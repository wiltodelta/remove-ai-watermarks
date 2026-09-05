"""Collect JSON-safe metadata and container forensics for one media file.

The collector is deliberately evidence-only: it preserves raw EXIF, IPTC, C2PA,
container metadata, encoder structure, hashes, timestamps, and bounded binary
payloads without deciding whether the content is AI-generated. Provenance verdicts
and pixel statistics are separate library stages.
"""

import base64
import contextlib
import hashlib
import io
import json
import os
import plistlib
import re
import struct
import zlib
from pathlib import Path
from typing import Any, cast

import piexif
from PIL import Image
from PIL.IptcImagePlugin import getiptcinfo

from remove_ai_watermarks import image_io
from remove_ai_watermarks._internal.constants import (
    PNG_METADATA_CHUNKS,
    RIFF_CODED_IMAGE_CHUNKS,
    RIFF_METADATA_CHUNKS,
)
from remove_ai_watermarks._internal.isobmff import (
    C2PA_BOX_TYPES,
    STREAM_SCAN_BYTES,
    iter_file_boxes,
)
from remove_ai_watermarks._internal.schema import require_schema_version
from remove_ai_watermarks.metadata import QUICK_SCAN_BYTES
from remove_ai_watermarks.metadata_record import HEAD_WINDOW

__all__ = [
    "FORENSIC_METADATA_RECORD_TYPE",
    "FORENSIC_METADATA_SCHEMA_VERSION",
    "SUPPORTED_EXTENSIONS",
    "collect_forensic_metadata",
]

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".heic",
    ".heif",
    ".avif",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    # video/px containers: no pixel decode, but C2PA reads them (Sora/Veo
    # carry C2PA manifests) and the byte scans still apply
    ".mp4",
    ".mov",
    ".m4v",
    ".jxl",
}

FORENSIC_METADATA_SCHEMA_VERSION = 1
FORENSIC_METADATA_RECORD_TYPE = "forensic_metadata"

_B64_CAP = 1 << 20  # 1 MB safety ceiling per embedded blob
_TEXT_CAP = 1 << 20  # decoded PNG text ceiling per chunk
# Preserve enough top-level ISOBMFF uuid/jumb payload data for downstream
# provenance algorithms without requiring them to reopen the source file.
_PROVENANCE_B64_CAP = STREAM_SCAN_BYTES
_RAW_SCAN_HEAD = HEAD_WINDOW
_RAW_SCAN_TAIL = QUICK_SCAN_BYTES


def _safe_str(v: Any) -> str:
    try:
        return str(v)
    except Exception:
        return repr(v)


def _b64(b: bytes, *, cap: int = _B64_CAP) -> str:
    """Legacy base64 value, with an explicit marker when the payload is capped."""
    encoded = base64.b64encode(b[:cap]).decode("ascii")
    return encoded + f"...TRUNCATED({len(b)} bytes total)" if len(b) > cap else encoded


def _decode_exif_value(v: Any) -> Any:
    """Make a piexif value JSON-safe; bytes are kept in full as hex."""
    if isinstance(v, bytes):
        if len(v) <= 64:
            try:
                return v.decode("utf-8", "strict")
            except (UnicodeDecodeError, ValueError):
                return f"hex:{v.hex()}"
        return f"hex:{v.hex()}"
    if isinstance(v, tuple | list):
        sequence = cast("list[Any] | tuple[Any, ...]", v)
        return [_decode_exif_value(item) for item in sequence]
    return v


def read_full_exif(
    path: Path, exif_blob: bytes | None = None, data: bytes | None = None
) -> tuple[dict[str, Any], bytes | None]:
    """All EXIF IFDs with decoded tag names (piexif, no re-encode), plus the
    raw embedded-thumbnail bytes for the caller's own thumbnail forensics.

    ``exif_blob`` is the PIL-exposed EXIF blob (PNG/WebP/HEIC path) so the
    caller's single Image.open is not repeated here. ``data`` is the
    already-read file bytes so piexif does not re-read the file."""
    try:
        exif: dict[str, Any] = piexif.load(data) if data is not None else piexif.load(str(path))
    except Exception:
        if not exif_blob:
            return {}, None
        try:
            exif = piexif.load(exif_blob)
        except Exception as exc:
            return {"error": _safe_str(exc)}, None
    out: dict[str, Any] = {}
    thumbnail: bytes | None = None
    for ifd, tags in exif.items():
        if ifd == "thumbnail":
            thumbnail = tags if isinstance(tags, bytes) else None
            out["thumbnail"] = f"{len(tags)} bytes" if isinstance(tags, bytes) else None
            continue
        if not isinstance(tags, dict):
            continue
        all_tag_names = cast("dict[str, dict[int, dict[str, Any]]]", getattr(piexif, "TAGS", {}))
        tag_names = all_tag_names.get(ifd, {})
        decoded: dict[str, Any] = {}
        for tag, value in cast("dict[int, Any]", tags).items():
            name = str(tag_names.get(tag, {}).get("name", f"tag_{tag}"))
            if name == "MakerNote" and isinstance(value, bytes):
                # full hex, no cap: measured on real uploads, Apple is ~2 KB
                # but Canon reaches 28 KB and Sony 38 KB (AF data, serials,
                # embedded previews) -- a cap would silently drop exactly the
                # camera-original evidence this scan exists to preserve
                decoded[name] = f"hex:{value.hex()}"
            else:
                decoded[name] = _decode_exif_value(value)
        out[ifd] = decoded
    return out, thumbnail


def png_text_decode(ctype: str, body: bytes) -> str:
    """Decode a tEXt/zTXt/iTXt chunk, inflating zlib where used.

    The compressed forms are where ComfyUI / Automatic1111 hide the
    generation workflow and prompt, so skipping the inflate would drop
    the strongest AI-provenance text a PNG can carry."""
    if ctype == "tEXt":
        suffix = b"...TRUNCATED" if len(body) > _TEXT_CAP else b""
        return (body[:_TEXT_CAP] + suffix).decode("utf-8", "replace")
    if ctype == "zTXt":
        nul = body.find(b"\x00")
        if nul == -1:
            return body[:_TEXT_CAP].decode("utf-8", "replace")
        keyword = body[:nul].decode("latin-1", "replace")
        # body[nul+1] = compression method (0 = zlib)
        try:
            inflater = zlib.decompressobj()
            decoded = inflater.decompress(body[nul + 2 :], _TEXT_CAP + 1)
            suffix = "...TRUNCATED" if len(decoded) > _TEXT_CAP else ""
            text = decoded[:_TEXT_CAP].decode("utf-8", "replace") + suffix
        except zlib.error:
            text = body[:_TEXT_CAP].decode("utf-8", "replace")
        return f"{keyword}\x00{text}"
    # iTXt: keyword\0 compflag(1) compmethod(1) lang\0 translated\0 text
    parts = body.split(b"\x00", 1)
    if len(parts) < 2:
        return body[:_TEXT_CAP].decode("utf-8", "replace")
    keyword = parts[0].decode("latin-1", "replace")
    rest = parts[1]
    if len(rest) < 2:
        return body[:_TEXT_CAP].decode("utf-8", "replace")
    compflag = rest[0]
    tail = rest[2:]
    for _ in range(2):  # skip language tag and translated keyword
        nul = tail.find(b"\x00")
        if nul == -1:
            return body[:_TEXT_CAP].decode("utf-8", "replace")
        tail = tail[nul + 1 :]
    if compflag:
        with contextlib.suppress(zlib.error):
            inflater = zlib.decompressobj()
            tail = inflater.decompress(tail, _TEXT_CAP + 1)
    if len(tail) > _TEXT_CAP:
        tail = tail[:_TEXT_CAP] + b"...TRUNCATED"
    return f"{keyword}\x00{tail.decode('utf-8', 'replace')}"


def read_png_chunks(data: bytes) -> tuple[list[dict[str, Any]], bytes]:
    """Every PNG chunk in order (type, length; text chunks decoded and
    inflated, binary chunks as base64) plus the post-IEND trailer bytes."""
    chunks: list[dict[str, Any]] = []
    post_iend = b""
    try:
        pos = 8
        while pos + 12 <= len(data):
            length = struct.unpack(">I", data[pos : pos + 4])[0]
            ctype = data[pos + 4 : pos + 8].decode("latin-1")
            body = data[pos + 8 : pos + 8 + length]
            entry: dict[str, Any] = {"type": ctype, "length": length}
            if ctype in ("tEXt", "zTXt", "iTXt"):
                entry["text"] = png_text_decode(ctype, body)
                if entry["text"].startswith("XML:com.adobe.xmp"):
                    entry["kind"] = "xmp"
            elif ctype == "tIME" and length == 7:
                y, mo, d, h, mi, s = struct.unpack(">HBBBBB", body)
                entry["time"] = f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:02d}Z"
            elif ctype == "gAMA" and length == 4:
                entry["gamma"] = struct.unpack(">I", body)[0] / 100000
            elif ctype == "sRGB" and length == 1:
                entry["rendering_intent"] = body[0]
            elif ctype == "iCCP":
                nul = body.find(b"\x00")
                if nul > 0:
                    entry["profile_name"] = body[:nul].decode("latin-1", "replace")
                entry["base64"] = _b64(body)
            elif ctype == "iDOT":
                # present in iOS/macOS screenshots
                entry["apple_screenshot_marker"] = True
            elif ctype in ("IHDR", "IDAT"):
                pass  # pixel-data / header chunks: length is signal enough
            elif length:
                entry["base64"] = _b64(body)
            chunks.append(entry)
            pos += 12 + length
            if ctype == "IEND":
                post_iend = data[pos:]
                break
    except Exception as exc:
        chunks.append({"error": _safe_str(exc)})
    return chunks, post_iend


def _set_jpeg_trailer(result: dict[str, Any], data: bytes, eoi: int) -> None:
    """Preserve bytes after JPEG EOI for Samsung Galaxy AI detection."""
    trailer = data[eoi + 2 :]
    result["post_eoi_bytes"] = len(trailer)
    if trailer:
        result["post_eoi_base64"] = _b64(trailer)


def read_jpeg_segments(data: bytes) -> dict[str, Any]:
    """Every JPEG APP segment in order, plus post-EOI trailer size.

    XMP APP1 segments are kept as full text; every other segment body is
    kept as full base64 (1 MB ceiling per segment).
    """
    result: dict[str, Any] = {"segments": [], "post_eoi_bytes": 0}
    try:
        pos = 2
        while pos + 4 <= len(data):
            if data[pos] != 0xFF:
                break
            marker = data[pos + 1]
            if marker == 0xD9:  # EOI
                _set_jpeg_trailer(result, data, pos)
                break
            if marker == 0xDA:  # SOS: entropy-coded data follows
                eoi = data.rfind(b"\xff\xd9")
                if eoi != -1:
                    _set_jpeg_trailer(result, data, eoi)
                break
            if not (0xE0 <= marker <= 0xEF):
                length = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
                pos += 2 + length
                continue
            length = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
            body = data[pos + 4 : pos + 2 + length]
            name = f"APP{marker - 0xE0}"
            entry: dict[str, Any] = {"marker": name, "length": length}
            # Adobe JPEG XMP APP1 magic (namespace URI in the packet, not a request).
            if body.startswith(b"http://ns.adobe.com/xap/1.0/\x00"):  # NOSONAR
                entry["kind"] = "xmp"
                entry["text"] = body[29:].decode("utf-8", "replace")
            elif name == "APP2" and body.startswith(b"MPF\x00"):
                # Multi-Picture Format: Ultra HDR gain map, Samsung dual shot
                entry["kind"] = "mpf"
                entry["base64"] = _b64(body)
            elif name == "APP2" and body.startswith(b"ICC_PROFILE"):
                entry["kind"] = "icc"
                entry["base64"] = _b64(body)
            elif name == "APP2" and body.startswith(b"FPXR"):
                entry["kind"] = "flashpix"
                entry["base64"] = _b64(body)
            elif name == "APP11":
                entry["kind"] = "c2pa_or_jumbf"
                # the parsed manifest is in c2pa_store, but the raw JUMBF
                # also carries assertion thumbnails the JSON may omit
                entry["base64"] = _b64(body)
            elif body.startswith(b"Exif\x00\x00"):
                entry["kind"] = "exif"
                entry["base64"] = _b64(body)
            elif body.startswith(b"Photoshop 3.0\x00"):
                entry["kind"] = "iptc_iim"
                entry["base64"] = _b64(body)
            else:
                entry["base64"] = _b64(body)
            result["segments"].append(entry)
            pos += 2 + length
    except Exception as exc:
        result["error"] = _safe_str(exc)
    return result


def read_pil_info(path: Path) -> tuple[dict[str, Any], dict[str, Any], bytes | None]:
    """One Image.open serving all PIL-derived data: container basics,
    img.info passthrough (XMP, comments), the IPTC-IIM dataset, and the
    raw EXIF blob (for the caller's piexif parse on PNG/WebP/HEIC)."""
    out: dict[str, Any] = {}
    iptc: dict[str, Any] = {}
    exif_blob: bytes | None = None
    try:
        with Image.open(path) as img:
            out["format"] = img.format
            out["mode"] = img.mode
            out["width"], out["height"] = img.size
            out["n_frames"] = getattr(img, "n_frames", 1)
            dpi = img.info.get("dpi")
            if dpi:
                out["dpi"] = [round(float(d), 2) for d in dpi]
            icc = img.info.get("icc_profile")
            if icc:
                out["icc_profile"] = {
                    "length": len(icc),
                    # header: profile class, color space, PCS (bytes 12-24)
                    "header_hex": icc[12:24].hex() if len(icc) >= 24 else "",
                    "base64": _b64(icc),
                }
            blob = img.info.get("exif")
            if isinstance(blob, bytes):
                exif_blob = blob
            try:
                info = getiptcinfo(img)
            except Exception:
                info = None
            if info:
                iptc = {f"{k[0]}:{k[1]}": _decode_exif_value(v) for k, v in info.items()}
            for key, value in img.info.items():
                if key in ("icc_profile", "exif", "dpi"):
                    continue
                if isinstance(value, bytes):
                    try:
                        out[f"info:{key}"] = value.decode("utf-8", "strict")
                    except (UnicodeDecodeError, ValueError):
                        out[f"info:{key}"] = f"base64:{_b64(value)}"
                else:
                    out[f"info:{key}"] = _safe_str(value)
    except Exception as exc:
        out["error"] = _safe_str(exc)
    return out, iptc, exif_blob


def read_c2pa_store(path: Path) -> dict[str, Any]:
    """Full C2PA manifest store through the package's cached reader."""
    from remove_ai_watermarks._internal.c2pa import read_manifest_store_json

    raw = read_manifest_store_json(path)
    if raw is None:
        return {}
    try:
        value: Any = json.loads(raw)
        return (
            cast("dict[str, Any]", value)
            if isinstance(value, dict)
            else {"error": "C2PA manifest store is not an object"}
        )
    except (TypeError, ValueError) as exc:
        return {"error": _safe_str(exc)}


def sniff_format(head: bytes) -> str:
    if head.startswith(b"\x89PNG"):
        return "png"
    if head.startswith(b"\xff\xd8"):
        return "jpeg"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "webp"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head.startswith(b"BM"):
        return "bmp"
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff"
    if head[4:8] == b"ftyp":
        return f"isobmff:{head[8:12].decode('latin-1', 'replace')}"
    return f"unknown:{head[:16].hex()}"


# --- JPEG encoder structure (metadata layer) ---


def _jpeg_forensics_bytes(data: bytes) -> dict[str, Any]:
    """Structure-level JPEG forensics: DQT tables (encoder fingerprint),
    SOF type (baseline/progressive) + chroma subsampling, DHT Huffman
    tables (custom = optimizing encoder), per-scan spectral selection
    (progressive scan script), JFIF/Adobe app markers, COM, DRI."""
    out: dict[str, Any] = {}
    try:
        if not data.startswith(b"\xff\xd8"):
            return out
        pos = 2
        scans: list[dict[str, int]] = []
        dqt: dict[str, list[int]] = {}
        dht: list[str] = []
        comments: list[str] = []
        while pos + 4 <= len(data):
            if data[pos] != 0xFF:
                break
            marker = data[pos + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                pos += 2
                continue
            if marker == 0xD9:
                break
            length = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
            body = data[pos + 4 : pos + 2 + length]
            if marker == 0xDB:  # DQT
                off = 0
                while off < len(body):
                    tid = body[off] & 0x0F
                    prec = body[off] >> 4
                    n = 128 if prec else 64
                    vals = list(body[off + 1 : off + 1 + n])
                    if prec:  # 16-bit entries
                        vals = [struct.unpack(">H", bytes(vals[i : i + 2]))[0] for i in range(0, len(vals) - 1, 2)]
                    dqt[str(tid)] = vals[:64]
                    off += 1 + n
            elif marker == 0xC4:  # DHT: custom tables mean an optimizing encoder
                dht.append(body.hex())
            elif marker == 0xDD and len(body) >= 2:  # DRI
                out["restart_interval"] = struct.unpack(">H", body[:2])[0]
            elif marker == 0xE0 and body.startswith(b"JFIF\x00") and len(body) >= 12:
                out["jfif"] = {
                    "version": f"{body[5]}.{body[6]}",
                    "density_units": body[7],
                    "x_density": struct.unpack(">H", body[8:10])[0],
                    "y_density": struct.unpack(">H", body[10:12])[0],
                }
            elif marker == 0xEE and body.startswith(b"Adobe") and len(body) >= 12:
                out["adobe_transform"] = body[11]
            elif marker in (0xC0, 0xC1, 0xC2) and len(body) >= 6:
                out["progressive"] = marker == 0xC2
                out["precision_bits"] = body[0]
                out["sof_height"] = struct.unpack(">H", body[1:3])[0]
                out["sof_width"] = struct.unpack(">H", body[3:5])[0]
                comps: list[dict[str, int]] = []
                for i in range(body[5]):
                    c = body[6 + i * 3 : 9 + i * 3]
                    if len(c) == 3:
                        comps.append({"h": c[1] >> 4, "v": c[1] & 0x0F, "tq": c[2]})
                if len(comps) >= 3:
                    lum = comps[0]
                    subs = {1: "4:4:4", 2: "4:2:2"}.get(lum["h"] * lum["v"])
                    out["subsampling"] = subs or f"{lum['h']}x{lum['v']}"
            elif marker == 0xFE:  # COM
                comments.append(body.decode("utf-8", "replace")[:2000])
            elif marker == 0xDA:
                # SOS spectral selection: the progressive scan script
                # differs across libjpeg / mozjpeg / Photoshop
                if len(body) >= 3:
                    ns = body[0]
                    tail = body[1 + ns * 2 :]
                    if len(tail) >= 3:
                        scans.append({"ss": tail[0], "se": tail[1], "ah": tail[2] >> 4, "al": tail[2] & 0x0F})
                # skip entropy-coded data to the next marker
                end = data.find(b"\xff\xd9", pos)
                nxt = data.find(b"\xff", pos + 2)
                while nxt != -1 and nxt + 1 < len(data) and data[nxt + 1] == 0x00:
                    nxt = data.find(b"\xff", nxt + 2)
                if nxt == -1 or (end != -1 and nxt >= end):
                    break
                pos = nxt
                continue
            pos += 2 + length
        if dqt:
            out["quant_tables"] = dqt
        if dht:
            out["huffman_tables_hex"] = dht
        if comments:
            out["comments"] = comments
        if scans:
            out["scan_count"] = len(scans)
            out["scan_script"] = scans
    except Exception as exc:
        out["error"] = _safe_str(exc)
    return out


def read_webp_chunks(data: bytes) -> list[dict[str, Any]]:
    """WebP RIFF chunk inventory (VP8X/VP8/VP8L/EXIF/XMP/ICCP/ANIM...)."""
    chunks: list[dict[str, Any]] = []
    try:
        pos = 12
        declared_end = 8 + struct.unpack("<I", data[4:8])[0] if len(data) >= 12 else len(data)
        container_end = min(len(data), declared_end)
        while pos + 8 <= container_end:
            chunk_type = data[pos : pos + 4]
            ctype = chunk_type.decode("latin-1")
            length = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
            body = data[pos + 8 : min(pos + 8 + length, container_end)]
            entry: dict[str, Any] = {"type": ctype, "length": length}
            if ctype == "XMP ":
                entry["kind"] = "xmp"
                entry["text"] = body.decode("utf-8", "replace")
            elif chunk_type in RIFF_CODED_IMAGE_CHUNKS:
                pass  # pixel-data chunks: length is signal enough
            elif length:
                entry["base64"] = _b64(body)
            chunks.append(entry)
            pos += 8 + length + (length & 1)  # chunks are 2-byte aligned
    except Exception as exc:
        chunks.append({"error": _safe_str(exc)})
    return chunks


def read_webp_late_metadata_path(path: Path, window: int = _RAW_SCAN_HEAD) -> list[dict[str, Any]]:
    """Stream metadata chunks after ``window`` while seeking over coded frames."""
    chunks: list[dict[str, Any]] = []
    try:
        file_size = path.stat().st_size
        with open(path, "rb") as handle:
            header = handle.read(12)
            if len(header) < 12 or not header.startswith(b"RIFF") or header[8:12] != b"WEBP":
                return chunks
            container_end = min(file_size, 8 + struct.unpack("<I", header[4:8])[0])
            position = 12
            while position + 8 <= container_end:
                handle.seek(position)
                chunk_header = handle.read(8)
                if len(chunk_header) < 8:
                    break
                chunk_type = chunk_header[:4]
                (length,) = struct.unpack("<I", chunk_header[4:8])
                start = position + 8
                safe_length = max(0, min(length, container_end - start))
                if chunk_type in RIFF_METADATA_CHUNKS and start >= window:
                    handle.seek(start)
                    body = handle.read(min(safe_length, _B64_CAP))
                    entry: dict[str, Any] = {
                        "type": chunk_type.decode("latin-1"),
                        "length": length,
                        "base64": _b64(body),
                    }
                    if len(body) < safe_length:
                        entry["truncated"] = True
                    chunks.append(entry)
                position = start + safe_length + (safe_length & 1)
    except (OSError, struct.error) as exc:
        chunks.append({"error": _safe_str(exc)})
    return chunks


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def xattr_where_from(path: Path) -> list[str]:
    """macOS download-source URLs (kMDItemWhereFroms), empty elsewhere."""
    try:
        getter = cast("Any", os.getxattr)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        raw = cast("bytes", getter(path, "com.apple.metadata:kMDItemWhereFroms"))
        value = plistlib.loads(raw)
        values = cast("list[Any]", value) if isinstance(value, list) else [value]
        return [str(item) for item in values]
    except (AttributeError, OSError, ValueError):
        return []


def xattr_quarantine(path: Path) -> str | None:
    """macOS quarantine string: flags; timestamp; downloading agent (Safari,
    Telegram, Chrome...). Presence alone means 'came from the internet'."""
    try:
        getter = cast("Any", os.getxattr)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        raw = cast("bytes", getter(path, "com.apple.quarantine"))
        return raw.decode("utf-8", "replace")[:500]
    except (AttributeError, OSError):
        return None


def read_isobmff_inventory(data: bytes) -> dict[str, Any]:
    """HEIC/AVIF/MOV box inventory: top-level boxes plus the meta item
    types (Exif, mime=XMP, auxl depth/gain-map, aae Apple-edits plist,
    irot derived images). Strong phone-provenance signal."""
    out: dict[str, Any] = {}
    try:
        stream = io.BytesIO(data)

        def boxes(start: int, end: int) -> list[tuple[str, int, int]]:
            return [
                (box_type.decode("latin-1"), payload_offset, box_end)
                for _, box_end, box_type, payload_offset in iter_file_boxes(stream, start, end)
            ]

        top = boxes(0, len(data))
        out["boxes"] = [t for t, _, _ in top]
        provenance_boxes: list[dict[str, Any]] = []
        for t, s, e in top:
            if t.encode("latin-1") in C2PA_BOX_TYPES:
                provenance_boxes.append(
                    {"type": t, "length": e - s, "base64": _b64(data[s:e], cap=_PROVENANCE_B64_CAP)}
                )
            if t == "moov":
                for ct, cs, ce in boxes(s, e):
                    if ct == "mvhd" and ce - cs >= 24:
                        # full box + creation/modification times (1904 epoch)
                        version = data[cs]
                        base = cs + 4
                        creation = struct.unpack(">I", data[base : base + 4])[0] if version == 0 else None
                        if creation:
                            out["mvhd_creation_time"] = creation - 2082844800
            elif t == "meta":
                # full box: 4 bytes version/flags, then child boxes
                for ct, cs, ce in boxes(s + 4, e):
                    if ct == "iinf":
                        # full box + entry count, then infe entries
                        count = struct.unpack(">H", data[cs + 4 : cs + 6])[0]
                        out["meta_item_count"] = count
                        item_types: list[str] = []
                        for it, is_, ie in boxes(cs + 6, ce):
                            if it == "infe" and ie - is_ >= 8:
                                # infe full box: version(1)+flags(3), then
                                # v2: item_ID(2)+protection(2)+item_type(4)
                                # v3: item_ID(4)+protection(2)+item_type(4)
                                version = data[is_]
                                off = is_ + 4 + (4 if version == 3 else 2) + 2
                                if off + 4 <= ie:
                                    item_types.append(data[off : off + 4].decode("latin-1", "replace"))
                        if item_types:
                            out["meta_item_types"] = sorted(set(item_types))
                    elif ct == "iprp":
                        out["has_iprp"] = True
                        for pt, ps, pe in boxes(cs, ce):
                            if pt == "ipco":
                                props = [t for t, _, _ in boxes(ps, pe)]
                                out["ipco_properties"] = props
                                # auxC holds the auxiliary image type URN
                                for box_type, qs, qe in boxes(ps, pe):
                                    if box_type == "auxC":
                                        out["auxc_types"] = (
                                            data[qs + 4 : qe].split(b"\x00")[0].decode("latin-1", "replace")
                                        )
                    elif ct == "iref":
                        out["has_iref"] = True
        if provenance_boxes:
            out["provenance_boxes"] = provenance_boxes
        # QuickTime metadata keys (©mak/©mod/©swr) for the MOV side of
        # Live Photos: tolerant printable-string grab after each atom
        qt: dict[str, str] = {}
        for atom, key in ((b"\xa9mak", "make"), (b"\xa9mod", "model"), (b"\xa9swr", "software")):
            idx = data.find(atom)
            if idx != -1:
                m = re.search(rb"[ -~]{4,80}", data[idx + 4 : idx + 200])
                if m:
                    qt[key] = m.group(0).decode("ascii", "replace")
        if qt:
            out["quicktime"] = qt
    except Exception as exc:
        out["error"] = _safe_str(exc)
    return out


def read_isobmff_provenance_path(path: Path) -> dict[str, Any]:
    """Stream top-level ISOBMFF boxes and preserve provenance payloads.

    This is the large-file counterpart to :func:`read_isobmff_inventory`.
    It seeks over media payloads instead of loading them into memory.
    """
    out: dict[str, Any] = {"boxes": []}
    provenance_boxes: list[dict[str, Any]] = []
    collected = 0
    try:
        file_size = path.stat().st_size
        with open(path, "rb") as f:
            for _, box_end, box_type_raw, payload_offset in iter_file_boxes(f, 0, file_size):
                box_type = box_type_raw.decode("latin-1")
                out["boxes"].append(box_type)
                payload_length = box_end - payload_offset
                if box_type_raw in C2PA_BOX_TYPES and collected < _PROVENANCE_B64_CAP:
                    to_read = min(payload_length, _PROVENANCE_B64_CAP - collected)
                    f.seek(payload_offset)
                    payload = f.read(to_read)
                    entry: dict[str, Any] = {
                        "type": box_type,
                        "length": payload_length,
                        "base64": _b64(payload, cap=_PROVENANCE_B64_CAP),
                    }
                    if to_read < payload_length:
                        entry["truncated"] = True
                    provenance_boxes.append(entry)
                    collected += len(payload)
    except (OSError, struct.error) as exc:
        out["error"] = _safe_str(exc)
    if provenance_boxes:
        out["provenance_boxes"] = provenance_boxes
    return out


def read_png_late_metadata_path(path: Path, window: int = _RAW_SCAN_HEAD) -> list[dict[str, Any]]:
    """Stream PNG metadata chunks whose payload starts after ``window``."""
    chunks: list[dict[str, Any]] = []
    try:
        file_size = path.stat().st_size
        with open(path, "rb") as f:
            if f.read(8) != b"\x89PNG\r\n\x1a\n":
                return chunks
            pos = 8
            while pos + 12 <= file_size:
                f.seek(pos)
                header = f.read(8)
                if len(header) < 8:
                    break
                length, chunk_type = struct.unpack(">I4s", header)
                data_start = pos + 8
                safe_length = max(0, min(length, file_size - data_start))
                if chunk_type in PNG_METADATA_CHUNKS and data_start >= window:
                    body = f.read(min(safe_length, _B64_CAP))
                    entry: dict[str, Any] = {
                        "type": chunk_type.decode("latin-1"),
                        "length": length,
                        "base64": _b64(body),
                    }
                    if len(body) < safe_length:
                        entry["truncated"] = True
                    chunks.append(entry)
                pos = data_start + safe_length + 4
                if chunk_type == b"IEND":
                    break
    except (OSError, struct.error) as exc:
        chunks.append({"error": _safe_str(exc)})
    return chunks


def apple_live_photo_id(head: bytes) -> str | None:
    """Apple Live Photo content identifier (links the still to its MOV).

    The UUID sits in the Apple MakerNote (tag 17) of the still and in the
    MOV metadata; a raw head scan finds it in either container."""
    # the UUID string sits next to "content.identifier" in the MOV, but in
    # the STILL it is a bare UUID inside the Apple MakerNote (whose header
    # is "Apple iOS"), so gate on either marker
    if b"content.identifier" not in head and b"com.apple.quicktime" not in head and b"Apple iOS" not in head:
        return None
    m = re.search(rb"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}", head)
    return m.group(0).decode("ascii") if m else None


_MAX_FULL_READ = 256 << 20  # files bigger than this are scanned head-only
_HEAD_READ = 4 << 20


def _sha256_stream(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def collect_forensic_metadata(
    path: Path,
    *,
    schema_version: int = FORENSIC_METADATA_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Collect the versioned, metadata-only forensic record for ``path``.

    This broad inspection record is not provenance-detector input. Use
    :func:`remove_ai_watermarks.metadata_record.collect_metadata_record` for the
    strict record accepted by ``identify_metadata_record``. Long-lived consumers
    should request the schema they implement; unsupported versions raise before the
    source is read.
    """
    schema_version = require_schema_version(
        schema_version,
        contract="forensic metadata",
        supported=(1,),
    )
    image_io._register_heif()  # pyright: ignore[reportPrivateUsage]
    stat = path.stat()
    oversized = stat.st_size > _MAX_FULL_READ
    if oversized:
        data = None
        with open(path, "rb") as f:
            head = f.read(_HEAD_READ)
    else:
        data = path.read_bytes()
        head = data
    record: dict[str, Any] = {
        "schema_version": schema_version,
        "record_type": FORENSIC_METADATA_RECORD_TYPE,
        "file": str(path),
        "name": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "birthtime": getattr(stat, "st_birthtime", None),
        "sha256": _sha256_stream(path) if data is None else sha256_of(data),
        "content_format": sniff_format(head),
    }
    if oversized:
        # Preserve the same bounded byte windows used by downstream provenance
        # algorithms while path-based readers (PIL, piexif, C2PA) run normally.
        record["oversized"] = {"head_scanned_bytes": len(head)}
        record["raw_metadata_windows"] = {"head_base64": _b64(head[:_RAW_SCAN_HEAD])}
        if stat.st_size > _RAW_SCAN_TAIL:
            with open(path, "rb") as f:
                f.seek(-_RAW_SCAN_TAIL, 2)
                record["raw_metadata_windows"]["tail_base64"] = _b64(f.read())
    where_from = xattr_where_from(path)
    if where_from:
        record["download_source_urls"] = where_from
    quarantine = xattr_quarantine(path)
    if quarantine:
        record["quarantine"] = quarantine
    live_photo_id = apple_live_photo_id(head[: 2 << 20])
    if live_photo_id:
        record["live_photo_content_id"] = live_photo_id
    record["pil"], record["iptc"], exif_blob = read_pil_info(path)
    record["exif"], thumbnail = read_full_exif(path, exif_blob, data)
    record["c2pa_store"] = read_c2pa_store(path)
    if data is not None:
        fmt = record["content_format"]
        if fmt == "png":
            record["png_chunks"], post_iend = read_png_chunks(data)
            if post_iend:
                record["png_post_iend_bytes"] = len(post_iend)
                record["png_post_iend_base64"] = _b64(post_iend)
        elif fmt == "jpeg":
            record["jpeg"] = read_jpeg_segments(data)
            record["jpeg_forensics"] = _jpeg_forensics_bytes(data)
        elif fmt == "webp":
            record["webp_chunks"] = read_webp_chunks(data)
        elif fmt.startswith("isobmff"):
            record["isobmff"] = read_isobmff_inventory(data)
    elif record["content_format"] == "png":
        late_chunks = read_png_late_metadata_path(path)
        if late_chunks:
            record["png_late_metadata_chunks"] = late_chunks
    elif record["content_format"] == "webp":
        late_chunks = read_webp_late_metadata_path(path)
        if late_chunks:
            record["webp_late_metadata_chunks"] = late_chunks
    elif record["content_format"].startswith("isobmff"):
        record["isobmff"] = read_isobmff_provenance_path(path)
    if thumbnail:
        record["has_exif_thumbnail"] = True
        # the embedded thumbnail is its own JPEG; after an edit its encoder
        # forensics commonly MISMATCH the main image (classic tamper tell)
        thumb_forensics = _jpeg_forensics_bytes(thumbnail)
        thumb_forensics["base64"] = _b64(thumbnail)
        record["exif_thumbnail_forensics"] = thumb_forensics
    return record
