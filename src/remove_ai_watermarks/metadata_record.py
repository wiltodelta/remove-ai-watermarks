"""Collect one image's provenance metadata into a portable, JSON-safe record.

WHY THIS EXISTS

``extract_provenance_evidence`` reads a file and hands back evidence in memory, so
collection and verdict must happen in the same process, on the machine holding the
image. This module splits them: collect here, judge anywhere, from a record that
survives JSON.

    record = collect_metadata_record(path)          # touches the file
    evidence = evidence_from_metadata_record(record, path=path)
    report = identify_from_evidence(evidence)       # touches nothing

WHAT GOES IN, AND WHY NOT SIMPLY THE FILE HEAD

The verdict reads a scan buffer that ``scan_head`` fills with the first mebibyte of
the file. Shipping that verbatim would make a record larger than a phone photo's
worth of metadata by two orders of magnitude, because for a PNG almost all of that
mebibyte is compressed pixel data in ``IDAT`` -- bytes no provenance token can ever
live in. A record carries the metadata REGIONS instead, walked per container: the
JPEG marker segments before the coded scan, every PNG chunk but ``IDAT``, the RIFF
chunks that are not coded image, the ISOBMFF provenance boxes, and in every case the
container's trailer.

COMPLETENESS IS A MEASURED PROPERTY, NOT A CLAIM

A region walker is only correct if nothing the verdict reads falls outside the
regions it keeps, and no test over fixtures can establish that: the failure mode is
a container placement nobody thought of. The contract is therefore ALSO verified
against the file path over a real corpus -- same image, both paths, identical
``ProvenanceReport``.

The placements that defeated an earlier draft of this collector, and the reason each
rule below exists, are recorded in ``docs/module-internals.md`` under "Portable
metadata record".
"""

from __future__ import annotations

import base64
import logging
import struct
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from remove_ai_watermarks._internal.constants import PNG_SIGNATURE, RIFF_CODED_IMAGE_CHUNKS
from remove_ai_watermarks._internal.schema import require_schema_version
from remove_ai_watermarks.metadata import (
    QUICK_SCAN_BYTES,
    SAMSUNG_EDITOR_MARKER,
    exif_bytes_from_image,
    exif_text,
    read_file_tail,
)

logger = logging.getLogger(__name__)

# The structural walk covers the same window the file path reads raw, so the two
# cannot disagree about a chunk type inside it. A smaller window would be cheaper but
# opens a blind spot: past the window only ``png_late_metadata``'s ALLOWLIST is
# collected, while the file path still sees every chunk type up to its own window --
# and a C2PA ``caBX`` chunk is in neither that allowlist nor ``IDAT``. Walking here
# costs little because the payload of the pixel stream is skipped, not copied.
HEAD_WINDOW = 1024 * 1024
# The window searched for the container's end marker. Matches the quick-scan window
# the file path uses when it goes looking for a Samsung trailer, so a trailer visible
# to one path is visible to the other.
TAIL_WINDOW = QUICK_SCAN_BYTES
# Kept from the tail when no end marker is found, so an unrecognized container still
# contributes its last bytes without carrying half a photo.
UNKNOWN_TRAILER_WINDOW = 64 * 1024

# PNG text keys the file path reads for a generator tag, in ITS order. NovelAI stamps
# Software/Source/Title rather than EXIF, and the first match wins, so order matters.
_GENERATOR_TEXT_KEYS = ("Software", "Source", "Title", "Description")
# Stable transport contract for records produced by this module. The version is
# deliberately separate from the verdict version: collection and interpretation can
# evolve independently as long as old records remain readable.
METADATA_RECORD_SCHEMA_VERSION = 1
METADATA_RECORD_TYPE = "provenance_metadata"


def _jpeg_regions(data: bytes) -> bytes:
    """Every marker segment up to the entropy-coded scan, plus the trailer after EOI.

    The scan itself is skipped by walking to SOS and then jumping to the trailing
    EOI, so a 20 MB photo contributes only its markers.

    TWIN: ``metadata._strip_jpeg_metadata_lossless`` walks the same marker chain. The
    two were left separate on purpose -- that one couples the walk to "return False and
    fall back to a PIL re-encode", a decision the lossless strip path owns and this one
    must not inherit -- so a fix to marker handling belongs in BOTH.
    """
    out = bytearray()
    index, size = 2, len(data)
    while index + 1 < size:
        if data[index] != 0xFF:
            break  # malformed boundary: keep what was collected, the tail still follows
        marker = data[index + 1]
        if marker in (0xDA, 0xD9):  # SOS / EOI: the coded scan follows
            break
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:  # standalone, no length
            index += 2
            continue
        if index + 4 > size:
            break
        segment_length = int.from_bytes(data[index + 2 : index + 4], "big")
        end = index + 2 + segment_length
        if segment_length < 2 or end > size:
            break
        out += data[index:end]
        index = end
    return bytes(out)


def _png_regions(data: bytes) -> bytes:
    """Every chunk except the ``IDAT`` payloads, plus whatever follows IEND.

    TWIN: ``metadata._png_late_metadata`` walks the same chunk chain by SEEKING over
    the file rather than over a buffer, and keeps an allowlist rather than skipping
    ``IDAT``. Both filters are deliberate: inside the window the file path sees every
    chunk type raw, past it only the allowlist survives.
    """
    out = bytearray()
    size = len(data)
    position = len(PNG_SIGNATURE)
    while position + 8 <= size:
        (length,) = struct.unpack(">I", data[position : position + 4])
        chunk_type = data[position + 4 : position + 8]
        start = position + 8
        # Clamp the length to the bytes that remain: a malformed 32-bit length must
        # not push the walk past EOF and abandon a genuine label chunk after it.
        safe_length = max(0, min(length, size - start))
        if chunk_type != b"IDAT":
            out += chunk_type + data[start : start + safe_length]
        position = start + safe_length + 4  # payload + CRC
        if chunk_type == b"IEND":
            out += data[position:]  # a trailer past IEND is metadata too
            break
    return bytes(out)


def _riff_regions(data: bytes) -> bytes:
    """Every RIFF chunk except the coded image payloads.

    TWIN: ``metadata._riff_late_metadata`` (seek-based, past the scan window) and
    ``_internal.riff`` (AVI ``LIST/INFO``). Same chunk-stepping arithmetic, three
    input models.
    """
    out = bytearray(data[:12])  # 'RIFF' + size + 'WEBP'
    declared_end = 8 + struct.unpack("<I", data[4:8])[0] if len(data) >= 12 else len(data)
    size = min(len(data), declared_end)
    position = 12
    while position + 8 <= size:
        chunk_type = data[position : position + 4]
        (length,) = struct.unpack("<I", data[position + 4 : position + 8])
        start = position + 8
        safe_length = max(0, min(length, size - start))
        if chunk_type not in RIFF_CODED_IMAGE_CHUNKS:
            out += chunk_type + data[start : start + safe_length]
        position = start + safe_length + (safe_length & 1)  # chunks are word-aligned
    return bytes(out)


def _isobmff_regions(image_path: Path, head: bytes) -> bytes:
    """Header window plus the provenance regions the bounded box walkers find.

    ISOBMFF hides a manifest in a ``uuid``/``jumb`` box that can sit after a
    multi-megabyte ``mdat``, and a TC260 label in ``moov.udta``. Both walkers seek
    rather than read the media, so neither pulls the payload in.
    """
    from remove_ai_watermarks._internal.isobmff import scan_c2pa_region, tc260_aigc_payloads

    out = bytearray(head[:HEAD_WINDOW])
    out += scan_c2pa_region(image_path, strict=True)
    for payload in tc260_aigc_payloads(image_path, strict=True):
        out += payload
    return bytes(out)


def _container_regions(image_path: Path, head: bytes) -> tuple[str, bytes]:
    """(container label, metadata bytes) for the container ``head`` starts with.

    ``head`` must be the file's raw first bytes. Handing this the ``scan_head``
    buffer instead is a trap: that buffer is the head CONCATENATED with late metadata
    payloads, so a structural walk runs off the end of the real head and parses the
    appended bytes as chunks, inflating the record and creating false signals.
    """
    from remove_ai_watermarks._internal.isobmff import is_isobmff
    from remove_ai_watermarks.metadata import png_late_metadata, riff_late_metadata

    if head.startswith(b"\xff\xd8"):
        return "jpeg", _jpeg_regions(head)
    if head.startswith(PNG_SIGNATURE):
        # Chunks placed after the pixel stream (an XMP packet at 2.7 MB, say) are
        # past the window; the same seek-past-IDAT reader the file path uses gets them.
        return "png", _png_regions(head) + png_late_metadata(image_path, HEAD_WINDOW, strict=True)
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "webp", _riff_regions(head) + riff_late_metadata(image_path, HEAD_WINDOW, strict=True)
    if is_isobmff(head):
        return "isobmff", _isobmff_regions(image_path, head)
    return "unknown", head


def _raw_head(image_path: Path) -> bytes:
    """The file's first bytes, unmodified -- the input every structural walk needs."""
    with open(image_path, "rb") as handle:
        return handle.read(HEAD_WINDOW)


def _trailer(image_path: Path, container: str) -> bytes:
    """The bytes that follow the container's end marker, and nothing else.

    A fixed-size tail read would be almost entirely pixels: the trailer of a 20 MB
    photo is a few kilobytes at most. So the end marker is located in the tail window
    and only what follows it is kept. When no marker is found (an unknown container,
    or one whose end lies before the window) the window is kept as-is, bounded --
    that is what a byte scan of the same file would have seen anyway.
    """
    if container == "webp":
        # RIFF declares its structural end in bytes 4..8. A fixed tail window is
        # normally the last animation/frame payload, not a trailer, so preserve
        # only bytes appended after the declared RIFF container.
        with open(image_path, "rb") as handle:
            header = handle.read(12)
            if len(header) < 12 or not header.startswith(b"RIFF"):
                return b""
            declared_end = 8 + struct.unpack("<I", header[4:8])[0]
            handle.seek(0, 2)
            file_size = handle.tell()
            if declared_end < 12 or declared_end >= file_size:
                return b""
            handle.seek(declared_end)
            return handle.read(min(file_size - declared_end, UNKNOWN_TRAILER_WINDOW))
    if container == "isobmff":
        # ISOBMFF has no out-of-container trailer convention. Its bounded box
        # walkers already collect late provenance while skipping ``mdat``; keeping
        # a blind tail here would carry coded media bytes.
        return b""

    tail = read_file_tail(image_path, TAIL_WINDOW, strict=True)
    if SAMSUNG_EDITOR_MARKER in tail:
        # Galaxy AI splits its evidence: the marker sits in the post-EOI trailer, but
        # the `genAIType` value it is gated on can sit INSIDE the entropy-coded scan.
        # Keeping only the trailer therefore carries the marker without the value and the
        # verdict silently drops the Samsung signal, so a marked file keeps the whole
        # window. Only Samsung-marked files pay for it.
        return tail
    marker = {"jpeg": b"\xff\xd9", "png": b"IEND\xae\x42\x60\x82"}.get(container)
    if marker is None:
        return tail[-UNKNOWN_TRAILER_WINDOW:]
    index = tail.rfind(marker)
    return tail[index + len(marker) :] if index >= 0 else tail[-UNKNOWN_TRAILER_WINDOW:]


def _decoder_info(image_path: Path) -> dict[str, Any]:
    """PIL's ``info`` mapping, read once.

    One open for both consumers below. They want different parts of the same mapping
    (the text keys, and the raw EXIF blob), and opening twice repeats the container
    header parse and, for a PNG carrying ``zTXt``, the zlib inflate with it.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(image_path) as img:
            # PIL types this mapping with a non-string key union (a DPI tuple key
            # exists), so the keys are normalized here rather than assumed.
            info = {str(key): value for key, value in img.info.items()}
            if "exif" not in info and (exif_bytes := exif_bytes_from_image(img)):
                info["exif"] = exif_bytes
            return info
    except (UnidentifiedImageError, ImportError) as exc:  # unsupported optional decoder
        logger.debug("PIL info unavailable for %s: %s", image_path, exc)
        return {}


def _exif_pairs(info: dict[str, Any]) -> dict[str, str]:
    """The 0th-IFD tags the verdict reads, under their tag NAMES.

    Not a convenience: two probes key on names rather than on the raw bytes already
    in the regions. ``xai_signature_pair`` wants an (ImageDescription, Artist) pair,
    and ``_external_exif_generator`` looks for Software / Make / Artist /
    ImageDescription. Ship the bytes alone and both silently return nothing, which
    is how a collector can silently lose Grok and NovelAI verdicts.
    """
    exif_bytes = info.get("exif")
    if not exif_bytes:
        return {}
    import piexif

    loaded = piexif.load(exif_bytes)
    tags = loaded.get("0th", {})
    exif_tags = loaded.get("Exif", {})

    pairs = {
        name: text
        for name, tag in (
            ("Software", piexif.ImageIFD.Software),
            ("Make", piexif.ImageIFD.Make),
            ("Artist", piexif.ImageIFD.Artist),
            ("ImageDescription", piexif.ImageIFD.ImageDescription),
        )
        if (text := exif_text(tags, tag))
    }
    if text := exif_text(exif_tags, piexif.ExifIFD.UserComment):
        pairs["UserComment"] = text
    return pairs


def _pil_info(info: dict[str, Any]) -> dict[str, str]:
    """PIL's ``info`` mapping as strings, the source of PNG text keys and ``hf-job-id``."""

    def text_of(value: Any) -> str:
        return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)

    # Emitted in the file path's own candidate order. ``generator_from_metadata``
    # returns the FIRST candidate carrying a known token. A record using PIL's natural
    # dict order can therefore choose a different platform string than the file path,
    # even though the two paths are supposed to be indistinguishable.
    out: dict[str, str] = {}
    for key in _GENERATOR_TEXT_KEYS:
        value = info.get(key)
        if value is not None and not isinstance(value, (dict, list, tuple)):
            out[f"info:{key}"] = text_of(value)
    for key, value in info.items():
        if key in _GENERATOR_TEXT_KEYS or isinstance(value, (dict, list, tuple)):
            continue
        out[f"info:{key}"] = text_of(value)
    return out


def collect_metadata_record(
    image_path: Path,
    *,
    schema_version: int = METADATA_RECORD_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Collect everything the provenance verdict reads, as a JSON-safe record.

    The record is the transport format for
    :func:`identify.evidence_from_metadata_record`: it carries the metadata regions
    (base64), the C2PA manifest store, and PIL's info mapping without carrying the
    primary coded-pixel stream. Schema and collection status are explicit so a
    consumer cannot mistake a failed read for an unknown provenance verdict.

    Args:
        image_path: Path to the image.
        schema_version: Output schema implemented by the consumer.

    Returns:
        A versioned JSON-serializable dict. ``metadata_base64`` holds the
        concatenated container regions, ``tail_base64`` the file trailer.
    """
    schema_version = require_schema_version(
        schema_version,
        contract="provenance metadata",
        supported=(1,),
    )

    from remove_ai_watermarks._internal.c2pa import read_manifest_store_json

    status = "complete"
    issues: list[dict[str, str]] = []
    container, regions, tail = "unknown", b"", b""
    info: dict[str, Any] = {}
    exif: dict[str, str] = {}
    store = None
    try:
        image_path.stat()
    except OSError as exc:
        logger.debug("metadata source unavailable for %s: %s", image_path, exc)
        status = "error"
        issues.append({"stage": "source", "code": "unavailable"})

    if status == "complete":
        try:
            head = _raw_head(image_path)
        except OSError as exc:
            logger.debug("metadata head read failed for %s: %s", image_path, exc)
            status = "error"
            issues.append({"stage": "head", "code": "read-failed"})
        else:
            # Each stage reports its own failure. Successful earlier regions stay
            # available for inspection, but the verdict rejects an incomplete record.
            try:
                container, regions = _container_regions(image_path, head)
            except Exception as exc:
                logger.debug("metadata region collection failed for %s: %s", image_path, exc)
                issues.append({"stage": "regions", "code": "collection-failed"})
            try:
                tail = _trailer(image_path, container)
            except Exception as exc:
                logger.debug("metadata trailer collection failed for %s: %s", image_path, exc)
                issues.append({"stage": "trailer", "code": "collection-failed"})
            try:
                info = _decoder_info(image_path)
                if container == "isobmff":
                    from remove_ai_watermarks._internal.isobmff import ai_metadata_tags

                    info.update(ai_metadata_tags(image_path, strict=True))
                exif = _exif_pairs(info)
            except Exception as exc:
                logger.debug("metadata decoder collection failed for %s: %s", image_path, exc)
                issues.append({"stage": "decoder", "code": "collection-failed"})
            try:
                store = read_manifest_store_json(image_path, strict=True)
            except Exception as exc:
                logger.debug("C2PA collection failed for %s: %s", image_path, exc)
                issues.append({"stage": "c2pa", "code": "collection-failed"})
            if issues:
                status = "partial"

    record: dict[str, Any] = {
        "schema_version": schema_version,
        "record_type": METADATA_RECORD_TYPE,
        "status": status,
        "issues": issues,
        "container": container,
        "name": image_path.name,
        "metadata_base64": base64.b64encode(regions).decode("ascii"),
        # Always collected: Samsung's Galaxy AI marker is a post-EOI trailer, and a
        # record without it loses that verdict outright.
        "tail_base64": base64.b64encode(tail).decode("ascii"),
        # PIL info BEFORE exif: the file path prefers a PNG text tag over an EXIF
        # one, and the normalizer walks the record in insertion order.
        "pil": _pil_info(info),
        "exif": exif,
    }
    if store is not None:
        record["c2pa_store"] = store
    return record
