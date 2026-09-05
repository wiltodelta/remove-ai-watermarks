"""Tests for the metadata-only forensic collector."""

from __future__ import annotations

import base64
import json
import zlib
from typing import TYPE_CHECKING

import piexif
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from remove_ai_watermarks.forensic_metadata import (
    FORENSIC_METADATA_RECORD_TYPE,
    FORENSIC_METADATA_SCHEMA_VERSION,
    SUPPORTED_EXTENSIONS,
    _b64,
    _decode_exif_value,
    _jpeg_forensics_bytes,
    _safe_str,
    apple_live_photo_id,
    collect_forensic_metadata,
    png_text_decode,
    read_full_exif,
    read_isobmff_inventory,
    read_isobmff_provenance_path,
    read_jpeg_segments,
    read_pil_info,
    read_png_chunks,
    read_png_late_metadata_path,
    read_webp_chunks,
    sha256_of,
    sniff_format,
    xattr_quarantine,
    xattr_where_from,
)

if TYPE_CHECKING:
    from pathlib import Path


def _jpeg(path: Path) -> Path:
    Image.new("RGB", (48, 32), (20, 80, 160)).save(path, "JPEG", quality=87)
    return path


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + payload).to_bytes(4, "big")
    return len(payload).to_bytes(4, "big") + chunk_type + payload + crc


def test_supported_extensions_are_media_not_documents():
    assert {".jpg", ".png", ".webp", ".heic", ".mp4"}.issubset(SUPPORTED_EXTENSIONS)
    assert ".pdf" not in SUPPORTED_EXTENSIONS


def test_json_helpers_and_format_sniffer():
    class BadString:
        def __str__(self):
            raise RuntimeError("no string")

    assert _safe_str("ok") == "ok"
    assert "BadString" in _safe_str(BadString())
    assert _b64(b"abc") == base64.b64encode(b"abc").decode("ascii")
    assert _b64(b"x" * 20, cap=4) == "eHh4eA==...TRUNCATED(20 bytes total)"
    assert _decode_exif_value(b"ascii") == "ascii"
    assert _decode_exif_value(b"\xff").startswith("hex:")
    assert _decode_exif_value((1, b"two")) == [1, "two"]
    assert sniff_format(b"\x89PNG\r\n\x1a\n") == "png"
    assert sniff_format(b"\xff\xd8\xff\xe0") == "jpeg"
    assert sniff_format(b"RIFF....WEBP") == "webp"
    assert sniff_format(b"....ftypheic").startswith("isobmff:")
    assert sniff_format(b"unknown").startswith("unknown:")


def test_png_text_and_container_metadata_are_preserved(tmp_path: Path):
    info = PngInfo()
    info.add_text("parameters", "Steps: 20, Model: SDXL", zip=True)
    path = tmp_path / "workflow.png"
    Image.new("RGB", (32, 32)).save(path, pnginfo=info)
    trailer = b'<TC260:AIGC>{"Label":"1"}</TC260:AIGC>'
    path.write_bytes(path.read_bytes() + trailer)

    record = collect_forensic_metadata(path)

    assert record["schema_version"] == FORENSIC_METADATA_SCHEMA_VERSION == 1
    assert record["record_type"] == FORENSIC_METADATA_RECORD_TYPE == "forensic_metadata"
    assert record["content_format"] == "png"
    assert any(chunk.get("type") == "zTXt" for chunk in record["png_chunks"])
    assert record["png_post_iend_bytes"] == len(trailer)
    assert base64.b64decode(record["png_post_iend_base64"]) == trailer
    assert "Steps: 20" in json.dumps(record)
    assert json.loads(json.dumps(record, allow_nan=False)) == record


def test_png_text_decoders_and_direct_chunk_reader(tmp_path: Path):
    assert "hello" in png_text_decode("tEXt", b"key\x00hello")
    compressed = b"prompt\x00\x00" + zlib.compress(b"workflow")
    assert "workflow" in png_text_decode("zTXt", compressed)
    assert "value" in png_text_decode("iTXt", b"key\x00\x00\x00\x00\x00value")

    path = tmp_path / "plain.png"
    Image.new("RGB", (8, 8)).save(path)
    chunks, trailer = read_png_chunks(path.read_bytes())

    assert chunks[0]["type"] == "IHDR"
    assert trailer == b""


def test_jpeg_exif_segments_encoder_and_trailer(tmp_path: Path):
    path = tmp_path / "camera.jpg"
    exif = piexif.dump(
        {
            "0th": {
                piexif.ImageIFD.Make: b"Camera Corp",
                piexif.ImageIFD.Software: b"Camera Firmware",
            },
            "Exif": {},
            "GPS": {},
            "1st": {},
        }
    )
    Image.new("RGB", (64, 48)).save(path, "JPEG", exif=exif, quality=82)
    trailer = b'PhotoEditor_Re_Edit_Data{"genAIType":1}'
    path.write_bytes(path.read_bytes() + trailer)

    record = collect_forensic_metadata(path)

    assert record["exif"]["0th"]["Make"] == "Camera Corp"
    assert record["jpeg"]["post_eoi_bytes"] == len(trailer)
    assert base64.b64decode(record["jpeg"]["post_eoi_base64"]) == trailer
    assert record["jpeg_forensics"]["quant_tables"]
    assert sha256_of(path.read_bytes()) == record["sha256"]


def test_direct_exif_pil_and_jpeg_readers(tmp_path: Path):
    path = _jpeg(tmp_path / "plain.jpg")

    exif, thumbnail = read_full_exif(path)
    pil, iptc, exif_blob = read_pil_info(path)
    segments = read_jpeg_segments(path.read_bytes())

    assert isinstance(exif, dict)
    assert thumbnail is None
    assert pil["width"] == 48
    assert pil["height"] == 32
    assert isinstance(iptc, dict)
    assert exif_blob is None or isinstance(exif_blob, bytes)
    assert isinstance(segments["segments"], list)
    assert _jpeg_forensics_bytes(path.read_bytes())["quant_tables"]
    assert _jpeg_forensics_bytes(b"not a jpeg") == {}


def test_webp_inventory_keeps_metadata_but_not_frame_pixels(tmp_path: Path):
    path = tmp_path / "image.webp"
    xmp = b"<x:xmpmeta>metadata</x:xmpmeta>"
    Image.new("RGB", (32, 32), (30, 40, 50)).save(path, "WEBP", xmp=xmp)

    chunks = read_webp_chunks(path.read_bytes())

    xmp_chunk = next(chunk for chunk in chunks if chunk["type"] == "XMP ")
    assert xmp_chunk["text"] == xmp.decode()
    assert all("base64" not in chunk for chunk in chunks if chunk["type"] in {"VP8 ", "VP8L", "ANMF"})


def test_isobmff_inventory_and_streaming_provenance(tmp_path: Path):
    path = tmp_path / "signed.mp4"
    ftyp = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    payload = b"jumb c2pa trainedAlgorithmicMedia"
    uuid_box = (8 + len(payload)).to_bytes(4, "big") + b"uuid" + payload
    path.write_bytes(ftyp + b"\x00\x00\x00\x08mdat" + uuid_box)

    inventory = read_isobmff_inventory(path.read_bytes())
    streamed = read_isobmff_provenance_path(path)

    assert "ftyp" in inventory["boxes"]
    assert base64.b64decode(inventory["provenance_boxes"][0]["base64"]) == payload
    assert base64.b64decode(streamed["provenance_boxes"][0]["base64"]) == payload


def test_oversized_path_keeps_bounded_windows_and_late_png_metadata(tmp_path: Path, monkeypatch):
    path = tmp_path / "late.png"
    Image.new("RGB", (16, 16)).save(path)
    source = path.read_bytes()
    iend = source.rfind(b"\x00\x00\x00\x00IEND")
    padding = _png_chunk(b"vpAg", b"\x00" * ((1 << 20) + 1))
    metadata = b'AIGC\x00{"Label":"1"}'
    path.write_bytes(source[:iend] + padding + _png_chunk(b"tEXt", metadata) + source[iend:])
    monkeypatch.setattr("remove_ai_watermarks.forensic_metadata._MAX_FULL_READ", 1)

    record = collect_forensic_metadata(path)

    assert record["oversized"]["head_scanned_bytes"] == path.stat().st_size
    assert base64.b64decode(record["raw_metadata_windows"]["head_base64"])
    assert base64.b64decode(record["png_late_metadata_chunks"][0]["base64"]) == metadata


def test_collection_registers_optional_heif_and_missing_file_raises(tmp_path: Path, monkeypatch):
    registered = False

    def mark_registered():
        nonlocal registered
        registered = True

    monkeypatch.setattr("remove_ai_watermarks.image_io._register_heif", mark_registered)
    collect_forensic_metadata(_jpeg(tmp_path / "plain.jpg"))

    assert registered is True
    with pytest.raises(FileNotFoundError):
        collect_forensic_metadata(tmp_path / "missing.jpg")


@pytest.mark.parametrize("schema_version", [2, True, 1.0])
def test_collection_rejects_unsupported_output_schema_before_reading(tmp_path: Path, schema_version: object):
    with pytest.raises(ValueError, match="Unsupported forensic metadata schema"):
        collect_forensic_metadata(
            tmp_path / "missing.jpg",
            schema_version=schema_version,  # type: ignore[arg-type]
        )


def test_xattrs_and_live_photo_probe_are_safe_on_plain_file(tmp_path: Path):
    path = _jpeg(tmp_path / "plain.jpg")

    assert xattr_where_from(path) == [] or isinstance(xattr_where_from(path), list)
    assert xattr_quarantine(path) is None or isinstance(xattr_quarantine(path), str)
    assert apple_live_photo_id(path.read_bytes()) is None


def test_late_png_reader_soft_fails_on_non_png(tmp_path: Path):
    path = tmp_path / "plain.bin"
    path.write_bytes(b"not png")

    assert read_png_late_metadata_path(path) == []
