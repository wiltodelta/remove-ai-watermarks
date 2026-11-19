"""Bounded Matroska/WebM metadata reader for native TC260 AIGC labels.

TC260-PG-20257A stores the label as a Matroska ``SimpleTag`` whose
``TagName`` is ``AIGC`` and whose ``TagString`` is the normative JSON object.
The walker seeks over unrelated elements such as clusters instead of reading
their payloads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

from remove_ai_watermarks.metadata import MAX_TC260_VALUE_BYTES, parse_tc260_aigc_json

EBML_MAGIC = b"\x1aE\xdf\xa3"
_SEGMENT_ID = 0x18538067
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
