"""Inventory AI-provenance signals AND all raw metadata over a dataset.

Read-only. For every image it writes one JSONL record containing:

- file basics: name, path, size in bytes, container format (content-sniffed),
  pixel dimensions, dpi, color mode, bit depth
- the library verdict: full `identify` ProvenanceReport (what we detect as AI)
- raw metadata the library does NOT interpret, so nothing has to be re-run:
  full EXIF (all IFDs, decoded tag names, MakerNote in full hex), all XMP
  packets, all PNG chunks (text chunks decoded and inflated, binary chunks
  as base64), all JPEG APP segments (marker + full base64), WebP RIFF
  chunks, ISOBMFF box/item inventory for HEIC/AVIF/MOV, IPTC-IIM dataset,
  ICC profile (full base64), C2PA manifest store JSON, post-EOI/IEND
  trailers, EXIF thumbnail (bytes + its own JPEG forensics), JPEG encoder
  forensics (quant tables, IJG quality estimate, progressive scan script,
  Huffman tables, subsampling, JFIF/Adobe markers), XMP edit-trail fields
  (creator tool, history agents/actions/timestamps, namespaces, motion
  photo / Ultra HDR / depth-map flags), file hashes and timestamps, macOS
  download provenance xattrs, Live Photo content identifier. Embedded
  binary blobs are base64'd in full with a 1 MB ceiling per blob.

Usage:
    python scan_dataset.py /path/to/dataset out_prefix

Writes out_prefix.jsonl (one record per file) and out_prefix.csv (flat
summary of the detection verdicts only, for quick sorting). If the prefix
ends with .gz the JSONL is gzip-compressed on the fly (3-5x smaller; still
streamable line by line).

For a large dataset, parallelize by sharding the input into folders and
running one process per shard (a multiprocessing pool hangs on macOS once
cv2/PIL are loaded; independent processes do not):

    for d in /dataset/*/; do
        python scan_dataset.py "$d" "out_$(basename "$d")" &
    done
    wait
    cat out_*.jsonl > dataset.jsonl   # or just read the shards lazily

Reading big results: never load the JSONL whole. Stream it:
polars.scan_ndjson (lazy), pandas.read_json(lines=True, chunksize=...),
or a plain line loop. One line = one self-contained JSON record.

Standalone use (no remove_ai_watermarks checkout):
    pip install pillow piexif c2pa-python pillow-heif
pillow-heif is only needed for HEIC/AVIF inputs. Without
remove_ai_watermarks the record simply carries no AI verdict; everything
raw is still collected.
"""

import base64
import contextlib
import csv
import json
import struct
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.IptcImagePlugin import getiptcinfo

try:
    from remove_ai_watermarks.identify import identify
except ImportError:  # standalone mode: raw metadata only, no AI verdict
    identify = None  # type: ignore[assignment]

try:  # optional reuse of the repo's cached C2PA reader (standalone-safe)
    from remove_ai_watermarks.noai.c2pa import read_manifest_store_json as _repo_c2pa_reader
except ImportError:
    _repo_c2pa_reader = None  # type: ignore[assignment]

SUPPORTED = {
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

_B64_CAP = 1 << 20  # 1 MB safety ceiling per embedded blob


def _safe_str(v: Any) -> str:
    try:
        return str(v)
    except Exception:
        return repr(v)


def _b64(b: bytes) -> str:
    """Full base64 of a binary blob, with a 1 MB safety ceiling."""
    if len(b) > _B64_CAP:
        return base64.b64encode(b[:_B64_CAP]).decode("ascii") + f"...TRUNCATED({len(b)} bytes total)"
    return base64.b64encode(b).decode("ascii")


def _decode_exif_value(v: Any) -> Any:
    """Make a piexif value JSON-safe; bytes are kept in full as hex."""
    if isinstance(v, bytes):
        if len(v) <= 64:
            try:
                return v.decode("utf-8", "strict")
            except (UnicodeDecodeError, ValueError):
                return f"hex:{v.hex()}"
        return f"hex:{v.hex()}"
    if isinstance(v, (tuple, list)):
        return [_decode_exif_value(x) for x in v]
    return v


def read_full_exif(path: Path, exif_blob: bytes | None = None) -> tuple[dict[str, Any], bytes | None]:
    """All EXIF IFDs with decoded tag names (piexif, no re-encode), plus the
    raw embedded-thumbnail bytes for the caller's own thumbnail forensics.

    ``exif_blob`` is the PIL-exposed EXIF blob (PNG/WebP/HEIC path) so the
    caller's single Image.open is not repeated here."""
    import piexif

    try:
        exif = piexif.load(str(path))
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
        tag_names = piexif.TAGS.get(ifd, {})
        decoded = {}
        for t, v in tags.items():
            name = tag_names.get(t, {}).get("name", f"tag_{t}")
            if name == "MakerNote" and isinstance(v, bytes):
                # full hex, no cap: measured on real uploads, Apple is ~2 KB
                # but Canon reaches 28 KB and Sony 38 KB (AF data, serials,
                # embedded previews) -- a cap would silently drop exactly the
                # camera-original evidence this scan exists to preserve
                decoded[name] = f"hex:{v.hex()}"
            else:
                decoded[name] = _decode_exif_value(v)
        out[ifd] = decoded
    return out, thumbnail


def _png_text_decode(ctype: str, body: bytes) -> str:
    """Decode a tEXt/zTXt/iTXt chunk, inflating zlib where used.

    The compressed forms are where ComfyUI / Automatic1111 hide the
    generation workflow and prompt, so skipping the inflate would drop
    the strongest AI-provenance text a PNG can carry."""
    import zlib

    if ctype == "tEXt":
        return body.decode("utf-8", "replace")
    if ctype == "zTXt":
        nul = body.find(b"\x00")
        if nul == -1:
            return body.decode("utf-8", "replace")
        keyword = body[:nul].decode("latin-1", "replace")
        # body[nul+1] = compression method (0 = zlib)
        try:
            text = zlib.decompress(body[nul + 2 :]).decode("utf-8", "replace")
        except zlib.error:
            text = body.decode("utf-8", "replace")
        return f"{keyword}\x00{text}"
    # iTXt: keyword\0 compflag(1) compmethod(1) lang\0 translated\0 text
    parts = body.split(b"\x00", 1)
    if len(parts) < 2:
        return body.decode("utf-8", "replace")
    keyword = parts[0].decode("latin-1", "replace")
    rest = parts[1]
    if len(rest) < 2:
        return body.decode("utf-8", "replace")
    compflag = rest[0]
    tail = rest[2:]
    for _ in range(2):  # skip language tag and translated keyword
        nul = tail.find(b"\x00")
        if nul == -1:
            return body.decode("utf-8", "replace")
        tail = tail[nul + 1 :]
    if compflag:
        with contextlib.suppress(zlib.error):
            tail = zlib.decompress(tail)
    return f"{keyword}\x00{tail.decode('utf-8', 'replace')}"


def read_png_chunks(data: bytes) -> tuple[list[dict[str, Any]], int]:
    """Every PNG chunk in order (type, length; text chunks decoded and
    inflated, binary chunks as base64) plus the post-IEND trailer size."""
    chunks: list[dict[str, Any]] = []
    post_iend = 0
    try:
        pos = 8
        while pos + 12 <= len(data):
            length = struct.unpack(">I", data[pos : pos + 4])[0]
            ctype = data[pos + 4 : pos + 8].decode("latin-1")
            body = data[pos + 8 : pos + 8 + length]
            entry: dict[str, Any] = {"type": ctype, "length": length}
            if ctype in ("tEXt", "zTXt", "iTXt"):
                entry["text"] = _png_text_decode(ctype, body)
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
                post_iend = len(data) - pos
                break
    except Exception as exc:
        chunks.append({"error": _safe_str(exc)})
    return chunks, post_iend


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
                result["post_eoi_bytes"] = len(data) - (pos + 2)
                break
            if marker == 0xDA:  # SOS: entropy-coded data follows
                eoi = data.rfind(b"\xff\xd9")
                if eoi != -1:
                    result["post_eoi_bytes"] = len(data) - (eoi + 2)
                break
            if not (0xE0 <= marker <= 0xEF):
                length = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
                pos += 2 + length
                continue
            length = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
            body = data[pos + 4 : pos + 2 + length]
            name = f"APP{marker - 0xE0}"
            entry: dict[str, Any] = {"marker": name, "length": length}
            if body.startswith(b"http://ns.adobe.com/xap/1.0/\x00"):
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
    """Full C2PA manifest store JSON. Prefers the repo's cached Reader
    wrapper when the package is installed; falls back to a direct
    c2pa-python Reader call in standalone mode."""
    if _repo_c2pa_reader is not None:
        try:
            store = _repo_c2pa_reader(path)
            return json.loads(store) if store else {}
        except Exception as exc:
            return {"error": _safe_str(exc)}
    try:
        from c2pa import Reader

        with Reader(str(path)) as reader:
            return json.loads(reader.json())
    except Exception as exc:
        return {"error": _safe_str(exc)}


def sniff_format(head: bytes) -> str:
    if head.startswith(b"\x89PNG"):
        return "png"
    if head.startswith(b"\xff\xd8"):
        return "jpeg"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "webp"
    if head[4:8] == b"ftyp":
        return f"isobmff:{head[8:12].decode('latin-1', 'replace')}"
    return f"unknown:{head[:16].hex()}"


# --- forensic helpers: signals that a file is not an untouched original ---

# IJG/libjpeg base quantization tables (quality 50), Annex K
_IJG_LUM_Q50 = [
    16,
    11,
    10,
    16,
    24,
    40,
    51,
    61,
    12,
    12,
    14,
    19,
    26,
    58,
    60,
    55,
    14,
    13,
    16,
    24,
    40,
    57,
    69,
    56,
    14,
    17,
    22,
    29,
    51,
    87,
    80,
    62,
    18,
    22,
    37,
    56,
    68,
    109,
    103,
    77,
    24,
    35,
    55,
    64,
    81,
    104,
    113,
    92,
    49,
    64,
    78,
    87,
    103,
    121,
    120,
    101,
    72,
    92,
    95,
    98,
    112,
    100,
    103,
    99,
]
_IJG_CHR_Q50 = [
    17,
    18,
    24,
    47,
    99,
    99,
    99,
    99,
    18,
    21,
    26,
    66,
    99,
    99,
    99,
    99,
    24,
    26,
    56,
    99,
    99,
    99,
    99,
    99,
    47,
    66,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
]


def _ijg_table(quality: int, base: list[int]) -> list[int]:
    scale = 5000 // quality if quality < 50 else 200 - quality * 2
    return [max(1, min(255, (b * scale + 50) // 100)) for b in base]


def estimate_jpeg_quality(lum_table: list[int]) -> int | None:
    """Best-fit IJG quality for a luminance quant table, else None if the
    table does not look IJG-derived (custom encoder, camera-specific)."""
    best_q, best_err = None, None
    for q in range(1, 101):
        ref = _ijg_table(q, _IJG_LUM_Q50)
        err = sum(abs(a - b) for a, b in zip(lum_table, ref, strict=True))
        if best_err is None or err < best_err:
            best_q, best_err = q, err
    return best_q if best_err is not None and best_err <= 200 else None


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
                comps = []
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
            q = estimate_jpeg_quality(dqt.get("0", []))
            if q is not None:
                out["estimated_ijg_quality"] = q
            else:
                out["ijg_derived"] = False
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


def parse_xmp_forensics(xmp_text: str) -> dict[str, Any]:
    """Edit-trail fields from an XMP packet: creator tool, history events,
    derived-from. These are the direct 'someone edited this' footprints."""
    import re

    out: dict[str, Any] = {}
    if not xmp_text:
        return out
    for field, patterns in {
        "creator_tool": [r'xmp:CreatorTool="([^"]*)"', r"<xmp:CreatorTool>([^<]*)"],
        "metadata_date": [r'xmp:MetadataDate="([^"]*)"'],
        "modify_date": [r'xmp:ModifyDate="([^"]*)"'],
        "derived_from": [r'stRef:documentID="([^"]*)"'],
        "original_document_id": [r'xmpMM:OriginalDocumentID="([^"]*)"'],
    }.items():
        for pat in patterns:
            m = re.search(pat, xmp_text)
            if m:
                out[field] = m.group(1)[:500]
                break
    agents = re.findall(r"stEvt:softwareAgent>([^<]*)", xmp_text)
    actions = re.findall(r"stEvt:action>([^<]*)", xmp_text)
    whens = re.findall(r"stEvt:when=\"([^\"]*)\"", xmp_text)
    if agents:
        out["history_software_agents"] = agents[:20]
    if actions:
        out["history_actions"] = actions[:20]
    if whens:
        out["history_when"] = whens[:20]
    # declared namespaces are an editor fingerprint on their own:
    # crs = Lightroom/Camera Raw (with edit settings attached), photoshop,
    # xmpG = GIMP, darktable, apple_photos, hdrgm = Ultra HDR, GDepth/GImage/
    # GCamera = Google computational photography (depth map, motion photo)
    namespaces = sorted(set(re.findall(r"xmlns:([A-Za-z0-9_]+)=", xmp_text)))
    if namespaces:
        out["namespaces"] = namespaces
    if re.search(r"crs:\w+[=>\s]", xmp_text):
        out["camera_raw_edited"] = True
    if "GCamera:MotionPhoto" in xmp_text or "Camera:MotionPhoto" in xmp_text:
        out["motion_photo"] = True
        m = re.search(r"MicroVideoOffset[=>\"\s]+(\d+)", xmp_text)
        if m:
            out["micro_video_offset"] = int(m.group(1))
    if "GDepth:" in xmp_text or "GImage:" in xmp_text:
        out["google_depth_map"] = True
    if "hdrgm:" in xmp_text:
        out["ultra_hdr_xmp"] = True
    return out


def xmp_texts_of(record_parts: dict[str, Any]) -> list[str]:
    """XMP packets, collected only from slots tagged at extraction (the
    container reader knows it is XMP; no substring guessing)."""
    texts: list[str] = []
    for key in ("info:xmp", "info:XML:com.adobe.xmp"):
        value = record_parts.get("pil", {}).get(key)
        if isinstance(value, str):
            texts.append(value)
    for seg in record_parts.get("jpeg", {}).get("segments", []):
        if seg.get("kind") == "xmp" and seg.get("text"):
            texts.append(seg["text"])
    for slot in ("png_chunks", "webp_chunks"):
        for chunk in record_parts.get(slot, []):
            if chunk.get("kind") == "xmp" and chunk.get("text"):
                texts.append(chunk["text"])
    return texts


def read_webp_chunks(data: bytes) -> list[dict[str, Any]]:
    """WebP RIFF chunk inventory (VP8X/VP8/VP8L/EXIF/XMP/ICCP/ANIM...)."""
    chunks: list[dict[str, Any]] = []
    try:
        pos = 12
        while pos + 8 <= len(data):
            ctype = data[pos : pos + 4].decode("latin-1")
            length = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
            body = data[pos + 8 : pos + 8 + length]
            entry: dict[str, Any] = {"type": ctype, "length": length}
            if ctype == "XMP ":
                entry["kind"] = "xmp"
                entry["text"] = body.decode("utf-8", "replace")
            elif ctype in ("VP8 ", "VP8L", "ALPH", "ANMF"):
                pass  # pixel-data chunks: length is signal enough
            elif length:
                entry["base64"] = _b64(body)
            chunks.append(entry)
            pos += 8 + length + (length & 1)  # chunks are 2-byte aligned
    except Exception as exc:
        chunks.append({"error": _safe_str(exc)})
    return chunks


def sha256_of(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def xattr_where_from(path: Path) -> list[str]:
    """macOS download-source URLs (kMDItemWhereFroms), empty elsewhere."""
    import os
    import plistlib

    try:
        raw = os.getxattr(path, "com.apple.metadata:kMDItemWhereFroms")
        value = plistlib.loads(raw)
        return [str(v) for v in value] if isinstance(value, list) else [str(value)]
    except (AttributeError, OSError, ValueError):
        return []


def xattr_quarantine(path: Path) -> str | None:
    """macOS quarantine string: flags; timestamp; downloading agent (Safari,
    Telegram, Chrome...). Presence alone means 'came from the internet'."""
    import os

    try:
        return os.getxattr(path, "com.apple.quarantine").decode("utf-8", "replace")[:500]
    except (AttributeError, OSError):
        return None


def read_isobmff_inventory(data: bytes) -> dict[str, Any]:
    """HEIC/AVIF/MOV box inventory: top-level boxes plus the meta item
    types (Exif, mime=XMP, auxl depth/gain-map, aae Apple-edits plist,
    irot derived images). Strong phone-provenance signal."""
    out: dict[str, Any] = {}
    try:

        def boxes(start: int, end: int) -> list[tuple[str, int, int]]:
            result = []
            pos = start
            while pos + 8 <= end:
                size, btype = struct.unpack(">I4s", data[pos : pos + 8])
                t = btype.decode("latin-1")
                header = 8
                if size == 1:
                    size = struct.unpack(">Q", data[pos + 8 : pos + 16])[0]
                    header = 16
                elif size == 0:
                    size = end - pos
                if size < header or pos + size > end:
                    break
                result.append((t, pos + header, pos + size))
                pos += size
            return result

        top = boxes(0, len(data))
        out["boxes"] = [t for t, _, _ in top]
        for t, s, e in top:
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
                        item_types = []
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
                                for qt, qs, qe in boxes(ps, pe):
                                    if qt == "auxC":
                                        out["auxc_types"] = (
                                            data[qs + 4 : qe].split(b"\x00")[0].decode("latin-1", "replace")
                                        )
                    elif ct == "iref":
                        out["has_iref"] = True
        # QuickTime metadata keys (©mak/©mod/©swr) for the MOV side of
        # Live Photos: tolerant printable-string grab after each atom
        import re

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


def apple_live_photo_id(head: bytes) -> str | None:
    """Apple Live Photo content identifier (links the still to its MOV).

    The UUID sits in the Apple MakerNote (tag 17) of the still and in the
    MOV metadata; a raw head scan finds it in either container."""
    import re

    # the UUID string sits next to "content.identifier" in the MOV, but in
    # the STILL it is a bare UUID inside the Apple MakerNote (whose header
    # is "Apple iOS"), so gate on either marker
    if b"content.identifier" not in head and b"com.apple.quicktime" not in head and b"Apple iOS" not in head:
        return None
    m = re.search(
        rb"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}",
        head,
    )
    return m.group(0).decode("ascii") if m else None


_MAX_FULL_READ = 256 << 20  # files bigger than this are scanned head-only
_HEAD_READ = 4 << 20


def _sha256_stream(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan_file(path: Path) -> dict[str, Any]:
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
        # too big to hold in memory: path-based readers (identify, PIL,
        # piexif, c2pa) still run; byte-level walkers are skipped
        record["oversized"] = {"head_scanned_bytes": len(head)}
    where_from = xattr_where_from(path)
    if where_from:
        record["download_source_urls"] = where_from
    quarantine = xattr_quarantine(path)
    if quarantine:
        record["quarantine"] = quarantine
    live_photo_id = apple_live_photo_id(head[: 2 << 20])
    if live_photo_id:
        record["live_photo_content_id"] = live_photo_id
    if identify is None:
        record["verdict"] = {"skipped": "remove_ai_watermarks not installed"}
    else:
        try:
            # visible-mark detectors are cv2-heavy and are the slow part; the
            # dataset scan only cares about metadata/embedded signals
            report = identify(path, check_visible=False)
            record["verdict"] = {
                "is_ai_generated": report.is_ai_generated,
                "platform": report.platform,
                "confidence": report.confidence,
                "ai_source_kind": report.ai_source_kind,
                "ai_from_metadata": report.ai_from_metadata,
                "watermarks": [str(w) for w in report.watermarks],
                "signals": [str(s) for s in report.signals],
                "caveats": [str(c) for c in report.caveats],
                "integrity_clashes": [str(c) for c in report.integrity_clashes],
            }
        except Exception as exc:
            record["verdict"] = {"error": _safe_str(exc)}
    record["pil"], record["iptc"], exif_blob = read_pil_info(path)
    record["exif"], thumbnail = read_full_exif(path, exif_blob)
    record["c2pa_store"] = read_c2pa_store(path)
    if data is not None:
        fmt = record["content_format"]
        if fmt == "png":
            record["png_chunks"], post_iend = read_png_chunks(data)
            if post_iend:
                record["png_post_iend_bytes"] = post_iend
        elif fmt == "jpeg":
            record["jpeg"] = read_jpeg_segments(data)
            record["jpeg_forensics"] = _jpeg_forensics_bytes(data)
        elif fmt == "webp":
            record["webp_chunks"] = read_webp_chunks(data)
        elif fmt.startswith("isobmff"):
            record["isobmff"] = read_isobmff_inventory(data)
    # edit-trail parse over every XMP packet found in any container slot
    xmp: dict[str, Any] = {}
    for text in xmp_texts_of(record):
        for key, value in parse_xmp_forensics(text).items():
            xmp.setdefault(key, value)
    if xmp:
        record["xmp_forensics"] = xmp
    if thumbnail:
        record["has_exif_thumbnail"] = True
        # the embedded thumbnail is its own JPEG; after an edit its encoder
        # forensics commonly MISMATCH the main image (classic tamper tell)
        thumb_forensics = _jpeg_forensics_bytes(thumbnail)
        thumb_forensics["base64"] = _b64(thumbnail)
        record["exif_thumbnail_forensics"] = thumb_forensics
    return record


def summary_row(record: dict[str, Any]) -> dict[str, Any]:
    v = record.get("verdict", {})
    pil = record.get("pil", {})
    return {
        "file": record.get("file"),
        "name": record.get("name"),
        "size_bytes": record.get("size_bytes"),
        "sha256": record.get("sha256"),
        "content_format": record.get("content_format"),
        "width": pil.get("width"),
        "height": pil.get("height"),
        "dpi": json.dumps(pil.get("dpi")),
        "is_ai_generated": v.get("is_ai_generated"),
        "platform": v.get("platform"),
        "confidence": v.get("confidence"),
        "ai_source_kind": v.get("ai_source_kind"),
        "watermarks": "; ".join(v.get("watermarks", [])),
        "signals": "; ".join(v.get("signals", [])),
        "integrity_clashes": "; ".join(v.get("integrity_clashes", [])),
    }


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    root, prefix = Path(sys.argv[1]), sys.argv[2]
    files = sorted(str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    # resume: skip files already present in the output of a previous
    # (interrupted) run with the same prefix
    gz = prefix.endswith(".gz")
    jsonl_name = prefix if gz else f"{prefix}.jsonl"
    csv_name = f"{prefix[:-3] if gz else prefix}.csv"
    import gzip

    jsonl_path = Path(jsonl_name)
    done: set[str] = set()
    if jsonl_path.exists():
        opener = gzip.open if gz else open
        with opener(jsonl_path, "rt") as existing:  # type: ignore[arg-type]
            for line in existing:
                with contextlib.suppress(Exception):
                    done.add(json.loads(line).get("file", ""))
        files = [f for f in files if f not in done]
    print(f"scanning {len(files)} files under {root} ({len(done)} already done)")
    n_done = 0
    text_opener = gzip.open if gz else open
    csv_exists = Path(csv_name).exists()
    with (
        text_opener(jsonl_path, "at") as jsonl,  # type: ignore[arg-type]
        open(csv_name, "a", newline="") as csvf,
    ):
        writer = csv.DictWriter(csvf, fieldnames=list(summary_row({})))
        if not csv_exists:
            writer.writeheader()
        for path_str in files:
            try:
                record = scan_file(Path(path_str))
            except Exception as exc:  # one corrupt file must not kill the scan
                record = {"file": path_str, "error": _safe_str(exc)}
            jsonl.write(json.dumps(record, default=str) + "\n")
            writer.writerow(summary_row(record))
            n_done += 1
            if n_done % 500 == 0:
                print(f"  {n_done}/{len(files)}", flush=True)
    print(f"done: {jsonl_name}, {csv_name}")


if __name__ == "__main__":
    main()
