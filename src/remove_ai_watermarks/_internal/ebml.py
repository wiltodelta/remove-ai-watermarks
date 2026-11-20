"""Bounded Matroska/WebM metadata reader for native TC260 AIGC labels.

TC260-PG-20257A stores the label as a Matroska ``SimpleTag`` whose
``TagName`` is ``AIGC`` and whose ``TagString`` is the normative JSON object.
The walker seeks over unrelated elements such as clusters instead of reading
their payloads.
"""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

from remove_ai_watermarks._internal.av1_obu import (
    patch_av1_obu_to_padding,
    scan_av1_obu_ai_spans,
    scan_av1_obu_stream_spans,
)
from remove_ai_watermarks.metadata import MAX_TC260_VALUE_BYTES, parse_tc260_aigc_json

EBML_MAGIC = b"\x1aE\xdf\xa3"
_SEGMENT_ID = 0x18538067
_TRACKS_ID = 0x1654AE6B
_TRACK_ENTRY_ID = 0xAE
_TRACK_NUMBER_ID = 0xD7
_CODEC_ID = 0x86
_CODEC_PRIVATE_ID = 0x63A2
_CONTENT_ENCODINGS_ID = 0x6D80
_CLUSTER_ID = 0x1F43B675
_SIMPLE_BLOCK_ID = 0xA3
_BLOCK_GROUP_ID = 0xA0
_BLOCK_ID = 0xA1
_CRC32_ID = 0xBF
_TAGS_ID = 0x1254C367
_TAG_ID = 0x7373
_SIMPLE_TAG_ID = 0x67C8
_TAG_NAME_ID = 0x45A3
_TAG_STRING_ID = 0x4487
_ATTACHMENTS_ID = 0x1941A469
_ATTACHED_FILE_ID = 0x61A7
_FILE_MEDIA_TYPE_ID = 0x4660
_FILE_DATA_ID = 0x465C
_MAX_TAG_NAME_BYTES = 256
_MAX_C2PA_ATTACHMENT_BYTES = 4 * 1024 * 1024


def _vint_length(first: int, *, maximum: int) -> int | None:
    """Return an EBML variable-integer length from its first byte."""
    mask = 0x80
    for length in range(1, maximum + 1):
        if first & mask:
            return length
        mask >>= 1
    return None


def _read_element_header(
    stream: BinaryIO,
    pos: int,
    limit: int,
) -> tuple[int, int, int] | None:
    """Return ``(element_id, payload_start, element_end)`` inside ``limit``."""
    if pos < 0 or pos >= limit:
        return None
    stream.seek(pos)
    first_raw = stream.read(1)
    if not first_raw:
        return None
    id_length = _vint_length(first_raw[0], maximum=4)
    if id_length is None or pos + id_length >= limit:
        return None
    element_id_raw = first_raw + stream.read(id_length - 1)
    if len(element_id_raw) != id_length:
        return None
    element_id = int.from_bytes(element_id_raw, "big")

    size_first_raw = stream.read(1)
    if not size_first_raw:
        return None
    size_length = _vint_length(size_first_raw[0], maximum=8)
    if size_length is None:
        return None
    size_rest = stream.read(size_length - 1)
    if len(size_rest) != size_length - 1:
        return None
    marker = 1 << (8 - size_length)
    size_value = int.from_bytes(bytes([size_first_raw[0] & (marker - 1)]) + size_rest, "big")
    payload_start = pos + id_length + size_length
    if payload_start > limit:
        return None
    unknown_size = size_value == (1 << (7 * size_length)) - 1
    element_end = limit if unknown_size else payload_start + size_value
    if element_end > limit:
        return None
    return element_id, payload_start, element_end


def _iter_elements(
    stream: BinaryIO,
    start: int,
    end: int,
) -> Iterator[tuple[int, int, int]]:
    """Yield valid direct children from one bounded EBML region."""
    pos = start
    while pos < end:
        header = _read_element_header(stream, pos, end)
        if header is None:
            return
        element_id, payload_start, element_end = header
        yield element_id, payload_start, element_end
        if element_end <= pos:
            return
        pos = element_end


def iter_top_level_ends(stream: BinaryIO, size: int) -> Iterator[int]:
    """End offset of each valid top-level element; stops early where a header fails.

    An element whose declared size runs past ``size`` fails its header, so a truncated
    file yields a last end short of ``size``. An unknown-size Segment runs to ``size``.
    """
    for _element_id, _payload_start, element_end in _iter_elements(stream, 0, size):
        yield element_end


def _read_bounded(
    stream: BinaryIO,
    start: int,
    end: int,
    maximum: int,
) -> bytes | None:
    size = end - start
    if size < 0 or size > maximum:
        return None
    stream.seek(start)
    value = stream.read(size)
    return value if len(value) == size else None


def _simple_tag_payloads(
    stream: BinaryIO,
    start: int,
    end: int,
) -> tuple[bytes, ...]:
    name: bytes | None = None
    values: list[bytes] = []
    for element_id, payload_start, element_end in _iter_elements(stream, start, end):
        if element_id == _TAG_NAME_ID:
            name = _read_bounded(stream, payload_start, element_end, _MAX_TAG_NAME_BYTES)
        elif element_id == _TAG_STRING_ID:
            value = _read_bounded(stream, payload_start, element_end, MAX_TC260_VALUE_BYTES)
            if value is not None:
                values.append(value)
    if name != b"AIGC":
        return ()
    return tuple(value for value in values if parse_tc260_aigc_json(value) is not None)


_C2PA_ATTACHMENT_MIMES = frozenset({b"application/c2pa", b"application/x-c2pa-manifest-store"})


def _iter_segment_children(stream: BinaryIO, file_size: int, target_id: int) -> Iterator[tuple[int, int]]:
    """Yield (payload_start, element_end) for top-level Segment children matching ``target_id``."""
    for element_id, payload_start, element_end in _iter_elements(stream, 0, file_size):
        if element_id != _SEGMENT_ID:
            continue
        for child_id, child_start, child_end in _iter_elements(stream, payload_start, element_end):
            if child_id == target_id:
                yield child_start, child_end


def _attached_c2pa_span(stream: BinaryIO, start: int, end: int) -> tuple[int, int] | None:
    mime: bytes | None = None
    data_span: tuple[int, int] | None = None
    for f_id, f_start, f_end in _iter_elements(stream, start, end):
        if f_id == _FILE_MEDIA_TYPE_ID:
            mime = _read_bounded(stream, f_start, f_end, _MAX_TAG_NAME_BYTES)
        elif f_id == _FILE_DATA_ID:
            data_span = (f_start, f_end)
    if mime is not None and mime.strip(b"\x00").lower() in _C2PA_ATTACHMENT_MIMES and data_span is not None:
        return data_span
    return None


def tc260_aigc_payloads(path: str | Path) -> tuple[bytes, ...]:
    """Read validated TC260 values from Matroska/WebM ``SimpleTag`` entries."""
    found: list[bytes] = []
    try:
        with open(path, "rb") as stream:
            if stream.read(4) != EBML_MAGIC:
                return ()
            stream.seek(0, 2)
            for child_start, child_end in _iter_segment_children(stream, stream.tell(), _TAGS_ID):
                for tag_id, tag_start, tag_end in _iter_elements(stream, child_start, child_end):
                    if tag_id != _TAG_ID:
                        continue
                    for simple_id, simple_start, simple_end in _iter_elements(stream, tag_start, tag_end):
                        if simple_id == _SIMPLE_TAG_ID:
                            found.extend(_simple_tag_payloads(stream, simple_start, simple_end))
    except OSError:
        return ()
    return tuple(found)


def c2pa_attachment_payloads(path: str | Path) -> tuple[bytes, ...]:
    """Read C2PA manifest attachments (application/c2pa) from Matroska/WebM files."""
    found: list[bytes] = []
    try:
        with open(path, "rb") as stream:
            if stream.read(4) != EBML_MAGIC:
                return ()
            stream.seek(0, 2)
            for child_start, child_end in _iter_segment_children(stream, stream.tell(), _ATTACHMENTS_ID):
                for att_id, att_start, att_end in _iter_elements(stream, child_start, child_end):
                    if att_id != _ATTACHED_FILE_ID:
                        continue
                    span = _attached_c2pa_span(stream, att_start, att_end)
                    if span is not None:
                        blob = _read_bounded(stream, span[0], span[1], _MAX_C2PA_ATTACHMENT_BYTES)
                        if blob is not None:
                            found.append(blob)
    except OSError:
        return ()
    return tuple(found)


def attachment_ordinals(path: str | Path) -> tuple[int, tuple[int, ...]]:
    """Return ``(total_attachments, c2pa_ordinals)`` for Matroska ``AttachedFile`` streams."""
    ordinals: list[int] = []
    ordinal = 0
    try:
        with open(path, "rb") as stream:
            if stream.read(4) != EBML_MAGIC:
                return 0, ()
            stream.seek(0, 2)
            for child_start, child_end in _iter_segment_children(stream, stream.tell(), _ATTACHMENTS_ID):
                for att_id, att_start, att_end in _iter_elements(stream, child_start, child_end):
                    if att_id != _ATTACHED_FILE_ID:
                        continue
                    if _attached_c2pa_span(stream, att_start, att_end) is not None:
                        ordinals.append(ordinal)
                    ordinal += 1
    except OSError:
        return 0, ()
    return ordinal, tuple(ordinals)


_MAX_AV1_BLOCK_BYTES = 4 * 1024 * 1024


def _read_ebml_vint_value(data: bytes, pos: int) -> tuple[int, int] | None:
    if pos >= len(data):
        return None
    length = _vint_length(data[pos], maximum=8)
    if length is None or pos + length > len(data):
        return None
    marker = 1 << (8 - length)
    raw = bytes([data[pos] & (marker - 1)]) + data[pos + 1 : pos + length]
    return int.from_bytes(raw, "big"), pos + length


def _read_ebml_signed_vint(data: bytes, pos: int) -> tuple[int, int] | None:
    parsed = _read_ebml_vint_value(data, pos)
    if parsed is None:
        return None
    val, next_pos = parsed
    length = next_pos - pos
    bias = (1 << (7 * length - 1)) - 1
    return val - bias, next_pos


def _block_frame_spans(
    header_bytes: bytes,
    block_file_start: int,
    total_len: int,
    av1_tracks: set[int],
) -> list[tuple[int, int]]:
    """Parse an EBML SimpleBlock/Block header prefix and return ``(file_start, file_end)`` per laced AV1 frame."""
    parsed = _read_ebml_vint_value(header_bytes, 0)
    if parsed is None:
        return []
    track_num, pos = parsed
    if track_num not in av1_tracks or pos + 3 > len(header_bytes) or pos + 3 > total_len:
        return []
    flags = header_bytes[pos + 2]
    pos += 3
    lacing = (flags >> 1) & 0x03
    if lacing == 0:
        return [(block_file_start + pos, block_file_start + total_len)] if pos < total_len else []
    if pos >= len(header_bytes):
        return []
    frame_count = header_bytes[pos] + 1
    pos += 1
    sizes: list[int] = []
    if lacing == 1:  # Xiph lacing
        for _ in range(frame_count - 1):
            sz = 0
            while pos < len(header_bytes) and header_bytes[pos] == 0xFF:
                sz += 255
                pos += 1
            if pos >= len(header_bytes):
                return []
            sz += header_bytes[pos]
            pos += 1
            sizes.append(sz)
    elif lacing == 2:  # Fixed-size lacing
        rem = total_len - pos
        if rem < 0 or rem % frame_count != 0:
            return []
        sizes = [rem // frame_count] * (frame_count - 1)
    else:  # EBML lacing
        first = _read_ebml_vint_value(header_bytes, pos)
        if first is None:
            return []
        cur_sz, pos = first
        sizes.append(cur_sz)
        for _ in range(frame_count - 2):
            delta = _read_ebml_signed_vint(header_bytes, pos)
            if delta is None:
                return []
            d_val, pos = delta
            cur_sz += d_val
            if cur_sz < 0:
                return []
            sizes.append(cur_sz)
    used = sum(sizes)
    if pos + used > total_len:
        return []
    sizes.append(total_len - pos - used)
    spans: list[tuple[int, int]] = []
    cur = pos
    for sz in sizes:
        if sz > 0:
            spans.append((block_file_start + cur, block_file_start + cur + sz))
        cur += sz
    return spans


def _scan_av1_block(
    stream: BinaryIO,
    b_start: int,
    b_end: int,
    av1_tracks: set[int],
    patches: list[tuple[int, int, int, int]],
    *,
    stop_on_first: bool,
) -> bool:
    total_len = b_end - b_start
    if total_len <= 4 or total_len > _MAX_AV1_BLOCK_BYTES:
        return False
    stream.seek(b_start)
    prefix = stream.read(min(1024, total_len))
    spans = _block_frame_spans(prefix, b_start, total_len, av1_tracks)
    if not spans:
        return False

    def _read_at(off: int, n: int) -> bytes:
        stream.seek(off)
        return stream.read(n)

    for f0, f1 in spans:
        found = scan_av1_obu_stream_spans(_read_at, f0, f1, stop_on_first=stop_on_first)
        if found:
            patches.extend(found)
            if stop_on_first:
                return True
    return False


def _scan_ebml_av1_stream(
    stream: BinaryIO,
    *,
    stop_on_first: bool = False,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int]]]:
    """Return ``(av1_obu_patches, crc_containers_touched)`` for ``V_AV1`` tracks in ``stream``."""
    if stream.read(4) != EBML_MAGIC:
        return [], []
    stream.seek(0, 2)
    file_size = stream.tell()
    av1_tracks: set[int] = set()
    patches: list[tuple[int, int, int, int]] = []
    crc_parents: list[tuple[int, int, int]] = []  # (crc_payload_start, body_after_crc_start, parent_end)

    for top_id, seg_start, seg_end in _iter_elements(stream, 0, file_size):
        if top_id != _SEGMENT_ID:
            continue
        for child_id, child_start, child_end in _iter_elements(stream, seg_start, seg_end):
            if child_id == _TRACKS_ID:
                crc_info: tuple[int, int, int] | None = None
                before_cnt = len(patches)
                for idx, (t_id, t_start, t_end) in enumerate(_iter_elements(stream, child_start, child_end)):
                    if idx == 0 and t_id == _CRC32_ID and t_end - t_start == 4:
                        crc_info = (t_start, t_end, child_end)
                        continue
                    if t_id != _TRACK_ENTRY_ID:
                        continue
                    track_num: int | None = None
                    codec_id: bytes | None = None
                    priv_span: tuple[int, int] | None = None
                    has_encodings = False
                    for f_id, f_start, f_end in _iter_elements(stream, t_start, t_end):
                        if f_id == _TRACK_NUMBER_ID and 1 <= f_end - f_start <= 8:
                            raw_n = _read_bounded(stream, f_start, f_end, 8)
                            if raw_n is not None:
                                track_num = int.from_bytes(raw_n, "big")
                        elif f_id == _CODEC_ID:
                            codec_id = _read_bounded(stream, f_start, f_end, 64)
                        elif f_id == _CODEC_PRIVATE_ID:
                            priv_span = (f_start, f_end)
                        elif f_id == _CONTENT_ENCODINGS_ID:
                            has_encodings = True
                    if (
                        track_num is not None
                        and codec_id is not None
                        and codec_id.strip(b"\x00") == b"V_AV1"
                        and not has_encodings
                    ):
                        av1_tracks.add(track_num)
                        if priv_span is not None and priv_span[1] - priv_span[0] > 4:
                            cfg_raw = _read_bounded(stream, priv_span[0] + 4, priv_span[1], _MAX_AV1_BLOCK_BYTES)
                            if cfg_raw is not None:
                                patches.extend(scan_av1_obu_ai_spans(cfg_raw, priv_span[0] + 4))
                                if stop_on_first and patches:
                                    return patches[:1], []
                if crc_info is not None and len(patches) > before_cnt:
                    crc_parents.append(crc_info)

            elif child_id == _CLUSTER_ID and av1_tracks:
                crc_info = None
                before_cnt = len(patches)
                for idx, (c_id, c_start, c_end) in enumerate(_iter_elements(stream, child_start, child_end)):
                    if idx == 0 and c_id == _CRC32_ID and c_end - c_start == 4:
                        crc_info = (t_start := c_start, c_end, child_end)
                        del t_start
                        continue
                    if c_id == _SIMPLE_BLOCK_ID:
                        if _scan_av1_block(stream, c_start, c_end, av1_tracks, patches, stop_on_first=stop_on_first):
                            return patches[:1], []
                    elif c_id == _BLOCK_GROUP_ID:
                        for g_id, g_start, g_end in _iter_elements(stream, c_start, c_end):
                            if g_id == _BLOCK_ID and _scan_av1_block(
                                stream, g_start, g_end, av1_tracks, patches, stop_on_first=stop_on_first
                            ):
                                return patches[:1], []
                if crc_info is not None and len(patches) > before_cnt:
                    crc_parents.append(crc_info)

    return patches, crc_parents


def scan_ebml_av1_ai_obus(
    path: str | Path,
    *,
    stop_on_first: bool = False,
) -> tuple[tuple[int, int, int, int], ...]:
    """Scan Matroska/WebM ``V_AV1`` tracks for AI ``OBU_METADATA`` payloads."""
    try:
        with open(path, "rb") as stream:
            patches, _crc = _scan_ebml_av1_stream(stream, stop_on_first=stop_on_first)
            return tuple(patches)
    except OSError:
        return ()


def scrub_ebml_av1_ai_obus(path: str | Path) -> int:
    """In-place rewrite AI ``OBU_METADATA`` units to ``OBU_PADDING`` and refresh EBML CRC-32s."""
    with open(path, "r+b") as stream:
        patches, crc_parents = _scan_ebml_av1_stream(stream)
        if not patches:
            return 0
        for hdr_off, orig_b0, pay_start, pay_end in patches:
            patch_av1_obu_to_padding(stream, hdr_off, orig_b0, pay_start, pay_end)
        for crc_pos, body_start, body_end in crc_parents:
            stream.seek(body_start)
            rem = body_end - body_start
            crc = 0
            while rem > 0:
                chunk = stream.read(min(rem, 1024 * 1024))
                if not chunk:
                    break
                crc = zlib.crc32(chunk, crc)
                rem -= len(chunk)
            stream.seek(crc_pos)
            stream.write((crc & 0xFFFFFFFF).to_bytes(4, "little"))
        stream.flush()
    return len(patches)
