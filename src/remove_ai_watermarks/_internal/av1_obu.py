"""AV1 Open Bitstream Unit (OBU) AI-metadata scanner and HDR-safe in-place scrubber.

Scans contiguous AV1 OBU sequences (inside ISOBMFF ``av1C`` / ``av01`` samples and
Matroska/WebM ``V_AV1`` ``CodecPrivate`` / ``SimpleBlock`` / ``Block`` frames) for
``OBU_METADATA`` (``obu_type == 5``) carrying C2PA, TC260 ``AIGC``, IPTC AI, or
Kling provenance while leaving ``METADATA_TYPE_HDR_CLL`` (1),
``METADATA_TYPE_HDR_MDCV`` (2), ``METADATA_TYPE_SCALABILITY`` (3), and
``METADATA_TYPE_TIMECODE`` (5) untouched. Scrubbing rewrites matching OBUs in-place
as same-length ``OBU_PADDING`` (``obu_type == 15``) with valid AV1 trailing bits so
all container box sizes, chunk offsets, and EBML element lengths stay invariant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from collections.abc import Callable

_C2PA_UUID = bytes.fromhex("d8fec3d61b0e483c92975828877ec481")
_OBU_METADATA = 5
_OBU_PADDING = 15
_PRESERVED_METADATA_TYPES = frozenset({1, 2, 3, 5})  # HDR_CLL, HDR_MDCV, SCALABILITY, TIMECODE
_METADATA_TYPE_ITUT_T35 = 4
_MAX_METADATA_PAYLOAD_BYTES = 4 * 1024 * 1024
_ZERO_CHUNK = b"\x00" * 65536


def read_leb128(data: bytes | bytearray | memoryview, pos: int) -> tuple[int, int] | None:
    """Decode an unsigned AV1 LEB128 integer starting at ``pos`` (up to 8 bytes)."""
    limit = len(data)
    value = 0
    for i in range(8):
        if pos + i >= limit:
            return None
        byte = int(data[pos + i])
        value |= (byte & 0x7F) << (7 * i)
        if (byte & 0x80) == 0:
            if value >= (1 << 32):
                return None
            return value, pos + i + 1
    return None


def encode_leb128(value: int) -> bytes:
    """Encode a non-negative integer as minimal unsigned AV1 LEB128 bytes."""
    if value < 0:
        raise ValueError("LEB128 value must be nonnegative")
    out = bytearray()
    rem = value
    while True:
        byte = rem & 0x7F
        rem >>= 7
        if rem:
            byte |= 0x80
        out.append(byte)
        if not rem:
            return bytes(out)


def is_ai_obu_metadata_payload(payload: bytes) -> bool:
    """Return True when an ``OBU_METADATA`` payload carries AI provenance rather than HDR/timing."""
    parsed = read_leb128(payload, 0)
    if parsed is None:
        return False
    meta_type, pos = parsed
    if meta_type in _PRESERVED_METADATA_TYPES:
        return False
    if meta_type == _METADATA_TYPE_ITUT_T35:
        if pos >= len(payload):
            return False
        country = payload[pos]
        pos += 2 if country == 0xFF else 1
        if pos > len(payload):
            return False
    body = payload[pos:]
    if not body:
        return False
    if _C2PA_UUID in body:
        return True
    lower = body.lower()
    return (
        b"urn:c2pa:" in lower
        or (b"jumb" in body and b"c2pa" in lower)
        or b"kling-ai" in lower
        or b"trainedalgorithmicmedia" in lower
        or (b'"contentproducer"' in lower and b'"label"' in lower)
        or (b'"serviceprovider"' in lower and b'"aigc"' in lower)
        or b"tc260:aigc" in lower
    )


def scan_av1_obu_stream_spans(
    read_at: Callable[[int, int], bytes],
    start_offset: int,
    end_offset: int,
    *,
    stop_on_first: bool = False,
) -> list[tuple[int, int, int, int]]:
    """Walk AV1 OBU headers in ``[start_offset, end_offset)`` without reading video frame slices."""
    found: list[tuple[int, int, int, int]] = []
    pos = start_offset
    while pos < end_offset:
        hdr = read_at(pos, min(10, end_offset - pos))
        if not hdr:
            break
        b0 = hdr[0]
        if (b0 & 0x81) != 0:
            break
        obu_type = (b0 >> 3) & 0x0F
        ext_flag = (b0 >> 2) & 1
        has_size = (b0 >> 1) & 1
        hdr_len = 1 + ext_flag
        if hdr_len > len(hdr):
            break
        if has_size:
            leb = read_leb128(hdr, hdr_len)
            if leb is None:
                break
            obu_size, rel_pay_pos = leb
            pay_pos = pos + rel_pay_pos
        else:
            pay_pos = pos + hdr_len
            obu_size = end_offset - pay_pos
        if obu_size < 0 or pay_pos + obu_size > end_offset:
            break
        if obu_type == _OBU_METADATA and 0 < obu_size <= _MAX_METADATA_PAYLOAD_BYTES:
            payload = read_at(pay_pos, obu_size)
            if len(payload) == obu_size and is_ai_obu_metadata_payload(payload):
                found.append((pos, b0, pay_pos, pay_pos + obu_size))
                if stop_on_first:
                    return found
        pos = pay_pos + obu_size
    return found


def scan_av1_obu_ai_spans(
    sample: bytes,
    base_file_offset: int,
) -> list[tuple[int, int, int, int]]:
    """Return ``(obu_hdr_offset, orig_hdr_byte, payload_start, payload_end)`` for AI ``OBU_METADATA`` units."""
    found: list[tuple[int, int, int, int]] = []
    pos = 0
    limit = len(sample)
    while pos < limit:
        b0 = sample[pos]
        if (b0 & 0x81) != 0:
            break
        obu_type = (b0 >> 3) & 0x0F
        ext_flag = (b0 >> 2) & 1
        has_size = (b0 >> 1) & 1
        hdr_end = pos + 1 + ext_flag
        if hdr_end > limit:
            break
        if has_size:
            leb = read_leb128(sample, hdr_end)
            if leb is None:
                break
            obu_size, pay_pos = leb
        else:
            pay_pos = hdr_end
            obu_size = limit - pay_pos
        if obu_size < 0 or pay_pos + obu_size > limit:
            break
        if obu_type == _OBU_METADATA and obu_size > 0:
            payload = sample[pay_pos : pay_pos + obu_size]
            if is_ai_obu_metadata_payload(payload):
                found.append(
                    (
                        base_file_offset + pos,
                        b0,
                        base_file_offset + pay_pos,
                        base_file_offset + pay_pos + obu_size,
                    )
                )
        pos = pay_pos + obu_size
    return found


def patch_av1_obu_to_padding(
    stream: BinaryIO,
    obu_hdr_offset: int,
    orig_hdr_byte: int,
    payload_start: int,
    payload_end: int,
) -> None:
    """Rewrite an ``OBU_METADATA`` unit in-place into same-length ``OBU_PADDING``."""
    new_b0 = (orig_hdr_byte & 0x87) | (_OBU_PADDING << 3)
    stream.seek(obu_hdr_offset)
    stream.write(bytes((new_b0,)))
    length = payload_end - payload_start
    if length > 0:
        stream.seek(payload_start)
        rem_zeros = length - 1
        while rem_zeros > 0:
            step = min(rem_zeros, len(_ZERO_CHUNK))
            stream.write(_ZERO_CHUNK[:step])
            rem_zeros -= step
        stream.write(b"\x80")
