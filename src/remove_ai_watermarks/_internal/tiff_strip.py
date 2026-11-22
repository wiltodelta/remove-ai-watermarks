"""Lossless in-place IFD metadata stripper for classic TIFF and DNG containers.

Preserves every pixel strip, tile, SubIFD, bit depth, multi-page chain, and RAW
DNG structure bit-for-bit without re-encoding or shifting any file offset.
"""

from __future__ import annotations

import logging
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_TYPE_SIZES: dict[int, int] = {
    1: 1,  # BYTE
    2: 1,  # ASCII
    3: 2,  # SHORT
    4: 4,  # LONG
    5: 8,  # RATIONAL
    6: 1,  # SBYTE
    7: 1,  # UNDEFINED
    8: 2,  # SSHORT
    9: 4,  # SLONG
    10: 8,  # SRATIONAL
    11: 4,  # FLOAT
    12: 8,  # DOUBLE
    13: 4,  # IFD
}

_TAG_IMAGE_DESCRIPTION = 270
_TAG_MAKE = 271
_TAG_STRIP_OFFSETS = 273
_TAG_STRIP_BYTE_COUNTS = 279
_TAG_SOFTWARE = 305
_TAG_ARTIST = 315
_TAG_TILE_OFFSETS = 324
_TAG_TILE_BYTE_COUNTS = 325
_TAG_SUB_IFDS = 330
_TAG_XMP = 700
_TAG_COPYRIGHT = 33432
_TAG_IPTC = 33723
_TAG_PHOTOSHOP = 34377
_TAG_EXIF_IFD = 34665
_TAG_ICC_PROFILE = 34675
_TAG_USER_COMMENT = 37510
_XP_TEXT_TAGS = frozenset({40091, 40092, 40093, 40094, 40095})

_TEXT_METADATA_TAGS = (
    frozenset({_TAG_IMAGE_DESCRIPTION, _TAG_MAKE, _TAG_SOFTWARE, _TAG_ARTIST, _TAG_COPYRIGHT, _TAG_USER_COMMENT})
    | _XP_TEXT_TAGS
)
_BLOB_METADATA_TAGS = frozenset({_TAG_XMP, _TAG_IPTC, _TAG_PHOTOSHOP})
_INSPECTABLE_METADATA_TAGS = _TEXT_METADATA_TAGS | _BLOB_METADATA_TAGS


def is_classic_tiff(head: bytes) -> bool:
    """Return True when ``head`` begins with a classic 32-bit TIFF/DNG header."""
    return len(head) >= 4 and head[:4] in (b"II*\x00", b"MM\x00*")


def _read_u16(buf: bytes | bytearray, endian: str, offset: int) -> int:
    return int(struct.unpack_from(f"{endian}H", buf, offset)[0])


def _read_u32(buf: bytes | bytearray, endian: str, offset: int) -> int:
    return int(struct.unpack_from(f"{endian}I", buf, offset)[0])


def _read_int_array(
    data: bytes | bytearray,
    endian: str,
    field_type: int,
    count: int,
    val_field: bytes,
) -> list[int]:
    """Decode a SHORT, LONG, or IFD tag array (inline or out-of-line)."""
    if field_type not in (3, 4, 13) or count <= 0:
        return []
    elem_size = _TYPE_SIZES[field_type]
    total = elem_size * count
    fmt = f"{endian}{count}{'H' if field_type == 3 else 'I'}"
    if total <= 4:
        return list(struct.unpack_from(fmt, val_field, 0))
    offset = _read_u32(val_field, endian, 0)
    if offset + total > len(data):
        return []
    return list(struct.unpack_from(fmt, data, offset))


def _external_span(
    size: int,
    endian: str,
    tag_id: int,
    field_type: int,
    count: int,
    val_field: bytes,
) -> tuple[int, int] | None:
    """Return the out-of-line ``(start, end)`` byte span for one IFD entry, or None if inline."""
    if field_type not in _TYPE_SIZES:
        return None
    byte_len = _TYPE_SIZES[field_type] * count
    if byte_len <= 4:
        return None
    offset = _read_u32(val_field, endian, 0)
    if offset < 8 or offset + byte_len > size:
        raise ValueError(f"TIFF tag {tag_id} value out of bounds: {offset}+{byte_len} > {size}")
    return (offset, offset + byte_len)


def _slice_tag_bytes(
    data: bytes | bytearray,
    field_type: int,
    count: int,
    val_field: bytes,
    ext_span: tuple[int, int] | None,
) -> bytes:
    """Materialize raw tag bytes only when a metadata predicate needs to inspect them."""
    if ext_span is not None:
        return bytes(data[ext_span[0] : ext_span[1]])
    byte_len = _TYPE_SIZES.get(field_type, 1) * count
    return bytes(val_field[:byte_len]) if byte_len > 0 else b""


def _bytes_carry_ai_markers(raw: bytes) -> bool:
    """Check one UTF-8/ASCII metadata payload against metadata.py's canonical rules."""
    from remove_ai_watermarks.metadata import payload_carries_ai_metadata

    return payload_carries_ai_metadata(raw)


def _payload_has_ai_markers(raw: bytes, *, is_xp_utf16: bool = False) -> bool:
    """Return True when raw TIFF/EXIF tag bytes carry canonical AI provenance markers."""
    if not raw:
        return False
    if _bytes_carry_ai_markers(raw):
        return True
    encodings: tuple[str, ...] = ()
    body = raw
    if is_xp_utf16:
        encodings = ("utf-16-le",)
    elif raw.startswith(b"UNICODE\x00"):
        body = raw[8:]
        encodings = ("utf-16-be", "utf-16-le")
    for enc in encodings:
        decoded = body.decode(enc, errors="ignore").encode("utf-8", errors="ignore")
        if decoded and _bytes_carry_ai_markers(decoded):
            return True
    return False


def _should_strip_tag(tag_id: int, raw: bytes, *, keep_standard: bool) -> bool:
    """Decide whether one non-structural TIFF/EXIF tag should be removed."""
    if tag_id == _TAG_ICC_PROFILE:
        return not keep_standard
    if tag_id in _INSPECTABLE_METADATA_TAGS:
        return (not keep_standard) or _payload_has_ai_markers(raw, is_xp_utf16=(tag_id in _XP_TEXT_TAGS))
    return False


def _overlaps_any(span: tuple[int, int], protected: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < p_end and p_start < end for p_start, p_end in protected)


def strip_tiff_metadata_lossless(source_path: Path, output_path: Path, *, keep_standard: bool = True) -> bool:
    """Strip AI metadata from a classic TIFF/DNG file without altering pixel offsets.

    Rewrites modified IFD tables in place (strictly shrinking entry counts) and
    zeroes removed external metadata buffers that do not overlap any protected
    pixel strip, tile, IFD, or kept tag payload. Returns ``False`` when the
    container is truncated, corrupt, or unsupported (for example BigTIFF).
    """
    from remove_ai_watermarks._internal.utils import staged_sibling

    with source_path.open("rb") as stream:
        head = stream.read(8)
        if len(head) < 8 or not is_classic_tiff(head):
            return False
        endian = "<" if head[:2] == b"II" else ">"
        first_ifd = _read_u32(head, endian, 4)
        stream.seek(0, 2)
        size = stream.tell()
        if first_ifd < 8 or first_ifd >= size:
            return False
        buf = bytearray(size)
        stream.seek(0)
        if stream.readinto(buf) != size:
            return False

    visited_ifds: set[int] = set()
    queue: list[tuple[int, bool, bool]] = [(first_ifd, False, False)]  # (offset, is_exif, drop_whole_ifd)

    protected: list[tuple[int, int]] = [(0, 8)]
    ifd_updates: list[tuple[int, list[bytes], int, int]] = []  # (offset, kept_entries, next_ifd, orig_count)
    zero_candidates: list[tuple[int, int]] = []

    try:
        while queue:
            ifd_off, is_exif, drop_whole_ifd = queue.pop(0)
            if ifd_off == 0 or ifd_off in visited_ifds:
                continue
            if ifd_off < 8 or ifd_off + 2 > size:
                return False
            visited_ifds.add(ifd_off)

            entry_count = _read_u16(buf, endian, ifd_off)
            table_end = ifd_off + 2 + entry_count * 12 + 4
            if table_end > size:
                return False

            if drop_whole_ifd:
                zero_candidates.append((ifd_off, table_end))
                for idx in range(entry_count):
                    e_off = ifd_off + 2 + idx * 12
                    tag_id = _read_u16(buf, endian, e_off)
                    field_type = _read_u16(buf, endian, e_off + 2)
                    count = _read_u32(buf, endian, e_off + 4)
                    val_bytes = bytes(buf[e_off + 8 : e_off + 12])
                    ext_span = _external_span(size, endian, tag_id, field_type, count, val_bytes)
                    if ext_span is not None:
                        zero_candidates.append(ext_span)
                continue

            protected.append((ifd_off, table_end))
            next_ifd = _read_u32(buf, endian, ifd_off + 2 + entry_count * 12)
            if next_ifd != 0:
                if next_ifd < 8 or next_ifd >= size:
                    return False
                queue.append((next_ifd, is_exif, False))

            strip_offsets: list[int] = []
            strip_counts: list[int] = []
            tile_offsets: list[int] = []
            tile_counts: list[int] = []
            kept_entries: list[bytes] = []
            removed_any = False

            for idx in range(entry_count):
                e_off = ifd_off + 2 + idx * 12
                entry = bytes(buf[e_off : e_off + 12])
                tag_id = _read_u16(entry, endian, 0)
                field_type = _read_u16(entry, endian, 2)
                count = _read_u32(entry, endian, 4)
                val_field = entry[8:12]
                ext_span = _external_span(size, endian, tag_id, field_type, count, val_field)

                if not is_exif:
                    if tag_id == _TAG_STRIP_OFFSETS:
                        strip_offsets = _read_int_array(buf, endian, field_type, count, val_field)
                    elif tag_id == _TAG_STRIP_BYTE_COUNTS:
                        strip_counts = _read_int_array(buf, endian, field_type, count, val_field)
                    elif tag_id == _TAG_TILE_OFFSETS:
                        tile_offsets = _read_int_array(buf, endian, field_type, count, val_field)
                    elif tag_id == _TAG_TILE_BYTE_COUNTS:
                        tile_counts = _read_int_array(buf, endian, field_type, count, val_field)
                    elif tag_id == _TAG_SUB_IFDS:
                        for sub_off in _read_int_array(buf, endian, field_type, count, val_field):
                            if 8 <= sub_off < size:
                                queue.append((sub_off, False, False))
                    elif tag_id == _TAG_EXIF_IFD:
                        exif_targets = _read_int_array(buf, endian, field_type, count, val_field)
                        if not keep_standard:
                            removed_any = True
                            if ext_span is not None:
                                zero_candidates.append(ext_span)
                            for exif_off in exif_targets:
                                if 8 <= exif_off < size:
                                    queue.append((exif_off, True, True))
                            continue
                        for exif_off in exif_targets:
                            if 8 <= exif_off < size:
                                queue.append((exif_off, True, False))
                        kept_entries.append(entry)
                        if ext_span is not None:
                            protected.append(ext_span)
                        continue

                raw_val = (
                    _slice_tag_bytes(buf, field_type, count, val_field, ext_span)
                    if keep_standard and tag_id in _INSPECTABLE_METADATA_TAGS
                    else b""
                )
                if _should_strip_tag(tag_id, raw_val, keep_standard=keep_standard):
                    removed_any = True
                    if ext_span is not None:
                        zero_candidates.append(ext_span)
                    continue

                kept_entries.append(entry)
                if ext_span is not None:
                    protected.append(ext_span)

            for off, length in zip(strip_offsets, strip_counts, strict=False):
                if off > 0 and length > 0 and off + length <= size:
                    protected.append((off, off + length))
            for off, length in zip(tile_offsets, tile_counts, strict=False):
                if off > 0 and length > 0 and off + length <= size:
                    protected.append((off, off + length))

            if removed_any:
                ifd_updates.append((ifd_off, kept_entries, next_ifd, entry_count))
    except (ValueError, struct.error):
        return False

    for ifd_off, kept_entries, next_ifd, orig_count in ifd_updates:
        new_count = len(kept_entries)
        struct.pack_into(f"{endian}H", buf, ifd_off, new_count)
        cursor = ifd_off + 2
        for entry in kept_entries:
            buf[cursor : cursor + 12] = entry
            cursor += 12
        struct.pack_into(f"{endian}I", buf, cursor, next_ifd)
        cursor += 4
        orig_end = ifd_off + 2 + orig_count * 12 + 4
        if cursor < orig_end:
            buf[cursor:orig_end] = b"\x00" * (orig_end - cursor)

    for span in zero_candidates:
        if not _overlaps_any(span, protected):
            start, end = span
            buf[start:end] = b"\x00" * (end - start)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with staged_sibling(output_path) as staged:
        staged.write_bytes(buf)
    return True
