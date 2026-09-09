"""Regression tests for container preservation and collection completeness."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import piexif
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from remove_ai_watermarks.identify import identify, identify_metadata_record
from remove_ai_watermarks.metadata import remove_ai_metadata
from remove_ai_watermarks.metadata_record import collect_metadata_record


def test_small_compressed_xmp_matches_portable_verdict(tmp_path):
    source = tmp_path / "compressed.png"
    info = PngInfo()
    info.add_text("XML:com.adobe.xmp", "<Iptc4xmpExt:AISystemUsed>ChatGPT</Iptc4xmpExt:AISystemUsed>", zip=True)
    Image.new("RGB", (16, 16)).save(source, pnginfo=info)
    assert source.stat().st_size < 1024
    assert b"ChatGPT" not in source.read_bytes()
    direct = identify(source, check_visible=False, check_invisible=False)
    portable = identify_metadata_record(collect_metadata_record(source), path=source)
    assert portable.platform is not None
    assert direct.to_dict() == portable.to_dict()


def test_jpeg_remove_all_drops_standard_exif(tmp_path):
    source = tmp_path / "source.jpg"
    exif = {
        "0th": {piexif.ImageIFD.Artist: b"Test Artist", piexif.ImageIFD.Orientation: 6},
        "GPS": {piexif.GPSIFD.GPSLatitudeRef: b"N", piexif.GPSIFD.GPSLatitude: ((1, 1), (2, 1), (3, 1))},
    }
    Image.new("RGB", (16, 8)).save(source, exif=piexif.dump(exif))
    assert piexif.load(str(source))["GPS"]
    output = tmp_path / "clean.jpg"
    remove_ai_metadata(source, output, keep_standard=False)
    assert not piexif.load(str(output))["0th"]
    assert not piexif.load(str(output))["GPS"]


def test_existing_directory_is_failed_collection(tmp_path):
    record = collect_metadata_record(tmp_path)
    assert record["status"] == "error"
    assert record["issues"]
    with pytest.raises(ValueError, match="collection failed"):
        identify_metadata_record(record, path=tmp_path)


def test_read_error_after_successful_stat_is_failed_collection(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8)).save(source)
    import remove_ai_watermarks.metadata_record as collector

    def denied(*args, **kwargs):
        raise PermissionError("synthetic read failure")

    monkeypatch.setattr(collector, "open", denied, raising=False)
    record = collect_metadata_record(source)
    assert record["status"] == "error"
    assert record["issues"]
    with pytest.raises(ValueError, match="collection failed"):
        identify_metadata_record(record, path=source)


def _run(arguments):
    return subprocess.run(arguments, capture_output=True, text=True, check=True)  # noqa: S603


def _packets(path):
    return json.loads(
        _run(
            [
                shutil.which("ffprobe"),
                "-v",
                "error",
                "-show_streams",
                "-show_packets",
                "-show_data_hash",
                "sha256",
                "-of",
                "json",
                str(path),
            ]
        ).stdout
    )


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg/ffprobe required")
@pytest.mark.parametrize("in_place", [False, True])
def test_remux_preserves_every_stream_and_packet(tmp_path, in_place):
    source = tmp_path / "source.mkv"
    _run(
        [
            shutil.which("ffmpeg"),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=0.1",
            "-map",
            "0:a",
            "-map",
            "1:a",
            "-c:a",
            "pcm_s16le",
            "-metadata",
            "title=Suno AI generated",
            str(source),
        ]
    )
    before = _packets(source)
    assert len(before["streams"]) == 2
    output = source if in_place else tmp_path / "clean.mkv"
    remove_ai_metadata(source, None if in_place else output)
    after = _packets(output)
    assert len(after["streams"]) == 2
    assert [(p["stream_index"], p["data_hash"]) for p in after["packets"]] == [
        (p["stream_index"], p["data_hash"]) for p in before["packets"]
    ]
    assert b"Suno AI generated" not in output.read_bytes()


@pytest.mark.parametrize("in_place", [False, True])
def test_failed_remux_does_not_publish_partial_output(tmp_path, monkeypatch, in_place):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"original media")
    output = source if in_place else tmp_path / "clean.mkv"
    if not in_place:
        output.write_bytes(b"existing output")
    original = source.read_bytes()
    previous = output.read_bytes()

    def fail(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"partial remux")
        return subprocess.CompletedProcess(cmd, 1, stderr="synthetic failure")

    monkeypatch.setattr(shutil, "which", lambda _: "/synthetic/ffmpeg")
    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        remove_ai_metadata(source, output)
    assert source.read_bytes() == original
    assert output.read_bytes() == previous
    assert set(tmp_path.iterdir()) == {source, output}


@pytest.mark.parametrize("stage", ["regions", "trailer", "decoder"])
def test_later_collection_failure_is_partial(tmp_path, monkeypatch, stage):
    import remove_ai_watermarks.metadata_record as collector

    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8)).save(source)
    called = []

    def fail(*args, **kwargs):
        called.append(True)
        raise RuntimeError("synthetic parser failure")

    helper = {"regions": "_container_regions", "trailer": "_trailer", "decoder": "_decoder_info"}[stage]
    monkeypatch.setattr(collector, helper, fail)
    record = collect_metadata_record(source)
    assert called
    assert record["status"] == "partial"
    assert {"stage": stage, "code": "collection-failed"} in record["issues"]
    with pytest.raises(ValueError, match="collection status"):
        identify_metadata_record(record, path=source)


@pytest.mark.parametrize("container", ["png", "webp", "isobmff"])
def test_structural_read_failure_reaches_collection_status(tmp_path, monkeypatch, container):
    import remove_ai_watermarks.metadata as metadata
    from remove_ai_watermarks._internal import isobmff

    source = tmp_path / f"source.{container}"
    if container == "isobmff":
        source.write_bytes(b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1")
    else:
        Image.new("RGB", (8, 8)).save(source, format=container)
    called = []

    def denied(*args, **kwargs):
        called.append(True)
        raise PermissionError("synthetic structural read failure")

    module = isobmff if container == "isobmff" else metadata
    monkeypatch.setattr(module, "open", denied, raising=False)
    record = collect_metadata_record(source)
    assert called
    assert record["status"] == "partial"
    assert {"stage": "regions", "code": "collection-failed"} in record["issues"]


def test_trailer_read_failure_reaches_collection_status(tmp_path, monkeypatch):
    import remove_ai_watermarks.metadata as metadata

    source = tmp_path / "source.jpg"
    Image.new("RGB", (8, 8)).save(source)

    def denied(*args, **kwargs):
        raise PermissionError("synthetic trailer read failure")

    monkeypatch.setattr(metadata, "open", denied, raising=False)
    record = collect_metadata_record(source)
    assert record["status"] == "partial"
    assert record["issues"] == [{"stage": "trailer", "code": "collection-failed"}]


@pytest.mark.parametrize("serialization_failure", [False, True])
def test_c2pa_reader_failure_is_partial_even_after_tolerant_cached_read(tmp_path, monkeypatch, serialization_failure):
    from remove_ai_watermarks._internal import c2pa

    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8)).save(source)
    calls = []

    class BrokenReader:
        @staticmethod
        def try_create(path):
            calls.append(path)
            if not serialization_failure:
                raise RuntimeError("synthetic C2PA open failure")
            return BrokenReader()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def json(self):
            raise RuntimeError("synthetic C2PA serialization failure")

    monkeypatch.setattr(c2pa, "_C2PA_READER_AVAILABLE", True)
    monkeypatch.setattr(c2pa, "_C2paReader", BrokenReader)
    assert c2pa.read_manifest_store_json(source) is None
    assert calls
    first = collect_metadata_record(source)
    second = collect_metadata_record(source)
    assert first["status"] == second["status"] == "partial"
    assert first["issues"] == [{"stage": "c2pa", "code": "collection-failed"}]
    assert len(calls) == 3, "failed strict reads must be retried, never cached as absence"


def test_exif_parser_failure_is_partial(tmp_path, monkeypatch):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (8, 8)).save(source, exif=piexif.dump({"0th": {piexif.ImageIFD.Artist: b"Test Artist"}}))
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("synthetic EXIF parser failure")

    monkeypatch.setattr(piexif, "load", fail)
    record = collect_metadata_record(source)
    assert calls
    assert record["status"] == "partial"
    assert record["issues"] == [{"stage": "decoder", "code": "collection-failed"}]
