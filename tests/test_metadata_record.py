"""Tests for the portable metadata record.

The contract is one property: a verdict built from the record equals the verdict
built from the file. Everything here exists to pin that, or to pin a placement the
record could plausibly drop.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from remove_ai_watermarks.identify import (
    PROVENANCE_REPORT_SCHEMA_VERSION,
    ProvenanceReport,
    evidence_from_metadata_record,
    identify,
    identify_from_evidence,
    identify_metadata_record,
)
from remove_ai_watermarks.metadata_record import (
    HEAD_WINDOW,
    METADATA_RECORD_SCHEMA_VERSION,
    METADATA_RECORD_TYPE,
    collect_metadata_record,
)

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "provenance"
COMPARED = ("is_ai_generated", "platform", "confidence", "ai_source_kind", "ai_from_metadata")


def _verdict_via_record(path: Path) -> ProvenanceReport:
    """The contractor's path: collect, serialize, judge -- without the file."""
    record = json.loads(json.dumps(collect_metadata_record(path)))
    return identify_metadata_record(record, path=path)


def _assert_same_verdict(path: Path) -> None:
    via_record = _verdict_via_record(path)
    via_file = identify(path, check_visible=False, check_invisible=False)

    for field in COMPARED:
        assert getattr(via_record, field) == getattr(via_file, field), field
    assert sorted(s.name for s in via_record.signals) == sorted(s.name for s in via_file.signals)
    assert sorted(via_record.watermarks) == sorted(via_file.watermarks)


def _noise_png(path: Path, size: tuple[int, int] = (700, 700)) -> Path:
    """A PNG whose IDAT is incompressible, so the file exceeds the head window."""
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)
    return path


def _insert_png_chunk(path: Path, chunk_type: bytes, payload: bytes) -> None:
    """Splice a chunk in just before IEND, i.e. AFTER the whole pixel stream."""
    data = path.read_bytes()
    end = data.rindex(b"IEND") - 4
    chunk = struct.pack(">I", len(payload)) + chunk_type + payload
    chunk += struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    path.write_bytes(data[:end] + chunk + data[end:])


class TestRecordReproducesTheFileVerdict:
    """The whole point of the record. Every tracked provenance fixture is compared,
    which covers C2PA, the China TC260 label, the xAI EXIF pair and the IPTC tag."""

    @pytest.mark.skipif(not FIXTURES.is_dir(), reason="provenance fixtures not present")
    @pytest.mark.parametrize("name", sorted(p.name for p in FIXTURES.iterdir()) if FIXTURES.is_dir() else [])
    def test_fixture(self, name: str):
        _assert_same_verdict(FIXTURES / name)

    @pytest.mark.skipif(not FIXTURES.is_dir(), reason="provenance fixtures not present")
    def test_the_fixtures_actually_exercise_several_signals(self):
        """Guards the test above: if the fixture set ever narrows to one signal
        family, equality across it stops meaning much."""
        found = {
            signal.name
            for path in FIXTURES.iterdir()
            for signal in identify(path, check_visible=False, check_invisible=False).signals
        }
        assert len(found) >= 4, found


class TestPlacementsTheRecordCouldDrop:
    def test_app_aigc_user_comment_survives(self, tmp_path: Path):
        """The file and record paths must both inspect EXIF UserComment."""
        import piexif

        path = tmp_path / "app-aigc.jpg"
        payload = json.dumps({"data": {"aigc_info": {"aigc_label_type": 1}}}, separators=(",", ":"))
        exif = piexif.dump(
            {
                "0th": {piexif.ImageIFD.Make: b"Canon"},
                "Exif": {piexif.ExifIFD.UserComment: payload.encode()},
                "GPS": {},
                "1st": {},
            }
        )
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path, "JPEG", exif=exif)

        assert identify(path, check_visible=False, check_invisible=False).is_ai_generated is True
        _assert_same_verdict(path)

    def test_a_trailer_after_eoi_survives(self, tmp_path: Path):
        """Samsung Galaxy AI appends its marker past the JPEG EOI, and the value it is
        gated on can sit further back still. A record that stopped at the last
        structural marker would report no signal at all."""
        path = tmp_path / "edited.jpg"
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path, "JPEG")
        with path.open("ab") as handle:
            handle.write(b'PhotoEditor_Re_Edit_Data{"genAIType":17}')

        assert identify(path, check_visible=False, check_invisible=False).confidence == "medium"
        _assert_same_verdict(path)

    def test_a_metadata_chunk_past_the_head_window_survives(self, tmp_path: Path):
        """A PNG encoder may put the label chunk after the pixels. The file is larger
        than the record's head window, so only the seek-past-IDAT reader finds it."""
        path = _noise_png(tmp_path / "late.png")
        assert path.stat().st_size > HEAD_WINDOW
        # An XMP packet in the namespaced TC260 form, which is how the label travels
        # when it is not a bare ``AIGC`` keyword chunk.
        label = json.dumps({"Label": "1", "ContentProducer": "001191110102MACQD9K64010000"})
        xmp = (
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
            b'xmlns:TC260="http://www.tc260.org.cn/ns/AIGC/1.0/"><TC260:AIGC>'
            + label.encode()
            + b"</TC260:AIGC></rdf:Description></rdf:RDF></x:xmpmeta>"
        )
        _insert_png_chunk(path, b"iTXt", b"XML:com.adobe.xmp\x00\x00\x00\x00\x00" + xmp)

        assert "aigc" in {s.name for s in identify(path, check_visible=False, check_invisible=False).signals}
        _assert_same_verdict(path)

    def test_an_exif_pair_survives(self, tmp_path: Path):
        """xAI is recognized from an (ImageDescription, Artist) PAIR, and both are read
        by tag NAME. A record carrying only raw bytes loses it."""
        import piexif

        path = tmp_path / "grok.jpg"
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path, "JPEG")
        exif = {
            "0th": {
                piexif.ImageIFD.ImageDescription: b"Signature: " + b"A" * 80,
                piexif.ImageIFD.Artist: b"3f2504e0-4f89-11d3-9a0c-0305e82c3301",
            }
        }
        piexif.insert(piexif.dump(exif), str(path))

        assert "xai_signature" in {s.name for s in identify(path, check_visible=False, check_invisible=False).signals}
        _assert_same_verdict(path)

    def test_an_imagemagick_raw_exif_profile_survives(self, tmp_path: Path):
        """Pillow decodes this PNG EXIF profile through getexif, not img.info."""
        import piexif

        exif = piexif.dump(
            {
                "0th": {
                    piexif.ImageIFD.ImageDescription: b"Signature: " + b"A" * 80,
                    piexif.ImageIFD.Artist: b"3f2504e0-4f89-11d3-9a0c-0305e82c3301",
                }
            }
        )
        profile = (
            "\nexif\n"
            + f"{len(exif):8d}\n"
            + "\n".join(exif.hex()[start : start + 72] for start in range(0, len(exif) * 2, 72))
        )
        info = PngInfo()
        info.add_text("Raw profile type exif", profile)
        path = tmp_path / "grok-converted.png"
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path, "PNG", pnginfo=info)

        assert "xai_signature" in {s.name for s in identify(path, check_visible=False, check_invisible=False).signals}
        _assert_same_verdict(path)

    def test_isobmff_generation_tags_survive(self, tmp_path: Path):
        def box(kind: bytes, payload: bytes) -> bytes:
            return (8 + len(payload)).to_bytes(4, "big") + kind + payload

        def key(name: bytes) -> bytes:
            return box(b"mdta", name)

        def value(index: int, payload: bytes) -> bytes:
            return box(index.to_bytes(4, "big"), box(b"data", b"\x00" * 8 + payload))

        keys = box(b"keys", b"\x00" * 4 + (2).to_bytes(4, "big") + key(b"workflow") + key(b"title"))
        ilst = box(b"ilst", value(1, b'{"app":"ComfyUI"}') + value(2, b"standard title"))
        meta = box(b"meta", b"\x00" * 4 + keys + ilst)
        path = tmp_path / "workflow.mp4"
        path.write_bytes(
            b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
            + box(b"mdat", b"synthetic-video")
            + box(b"moov", box(b"udta", meta))
        )

        assert "gen_params" in {s.name for s in identify(path, check_visible=False, check_invisible=False).signals}
        _assert_same_verdict(path)


class TestRecordShape:
    def test_the_record_survives_json(self, tmp_path: Path):
        record = collect_metadata_record(_noise_png(tmp_path / "plain.png"))

        assert json.loads(json.dumps(record, allow_nan=False))["container"] == "png"
        assert record["schema_version"] == METADATA_RECORD_SCHEMA_VERSION == 1
        assert record["record_type"] == METADATA_RECORD_TYPE == "provenance_metadata"

    @pytest.mark.parametrize("keep_version", [True, False])
    def test_transport_filename_is_not_evidence(self, tmp_path: Path, keep_version: bool):
        """The path labels a record; detector tokens in it do not describe pixels."""
        path = tmp_path / "jumb-c2pa-OpenAI-trainedAlgorithmicMedia.jpg"
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path, "JPEG")
        record = collect_metadata_record(path)
        if not keep_version:
            record.pop("schema_version")
            record.pop("record_type")

        report = identify_metadata_record(record, path=path)

        assert report.is_ai_generated is None
        assert report.platform is None
        assert report.confidence == "none"

    def test_pixels_are_not_carried(self, tmp_path: Path):
        """The reason the record walks regions instead of shipping the head: a record
        that carried the pixel stream would be the size of the image."""
        path = _noise_png(tmp_path / "big.png", size=(1200, 1200))
        record = collect_metadata_record(path)

        idat = path.read_bytes()
        start = idat.index(b"IDAT") + 4
        assert idat[start : start + 512] not in json.dumps(record).encode()
        assert len(json.dumps(record)) < path.stat().st_size // 10

    def test_an_unreadable_container_still_yields_a_record(self, tmp_path: Path):
        path = tmp_path / "junk.bin"
        path.write_bytes(b"\x00\x01\x02not an image at all" * 100)

        record = collect_metadata_record(path)

        assert record["container"] == "unknown"
        assert json.dumps(record)  # serializable, and no exception on the way here
        _assert_same_verdict(path)

    def test_a_missing_file_does_not_raise(self, tmp_path: Path):
        """Collection runs over whatever a caller hands it, including a path that
        vanished between listing and reading."""
        record = collect_metadata_record(tmp_path / "gone.png")

        assert record["container"] == "unknown"
        assert record["metadata_base64"] == ""
        assert record["status"] == "error"
        assert record["issues"] == [{"stage": "source", "code": "unavailable"}]

    def test_unknown_record_schema_is_rejected(self, tmp_path: Path):
        path = _noise_png(tmp_path / "plain.png")
        record = collect_metadata_record(path)
        record["schema_version"] = 2

        with pytest.raises(ValueError, match="Unsupported provenance metadata schema"):
            identify_metadata_record(record, path=path)

    @pytest.mark.parametrize("schema_version", [True, 1.0, "1", None])
    def test_native_record_schema_requires_the_integer_one(self, tmp_path: Path, schema_version: object):
        path = _noise_png(tmp_path / "plain.png")
        record = collect_metadata_record(path)
        record["schema_version"] = schema_version

        with pytest.raises(ValueError, match="Unsupported provenance metadata schema"):
            identify_metadata_record(record, path=path)

    @pytest.mark.parametrize("status", [None, "partial", "unknown", True])
    def test_native_record_requires_complete_collection_status(self, tmp_path: Path, status: object):
        path = _noise_png(tmp_path / "plain.png")
        record = collect_metadata_record(path)
        if status is None:
            record.pop("status")
        else:
            record["status"] = status

        with pytest.raises(ValueError, match="collection status"):
            identify_metadata_record(record, path=path)

    @pytest.mark.parametrize("schema_version", [2, True, 1.0])
    def test_collection_rejects_unsupported_output_schema_before_reading(self, tmp_path: Path, schema_version: object):
        with pytest.raises(ValueError, match="Unsupported provenance metadata schema"):
            collect_metadata_record(
                tmp_path / "missing.png",
                schema_version=schema_version,  # type: ignore[arg-type]
            )

    def test_broad_forensic_record_is_not_detector_input(self, tmp_path: Path):
        path = _noise_png(tmp_path / "plain.png")

        with pytest.raises(ValueError, match="Unsupported metadata record type"):
            identify_metadata_record(
                {"record_type": "forensic_metadata", "schema_version": 1},
                path=path,
            )

    def test_failed_collection_cannot_be_judged_as_an_unknown_image(self, tmp_path: Path):
        path = tmp_path / "missing.png"

        with pytest.raises(ValueError, match="collection failed"):
            identify_metadata_record(collect_metadata_record(path), path=path)


class TestReportTransport:
    def test_report_contract_is_versioned_json_and_omits_local_path(self, tmp_path: Path):
        path = _noise_png(tmp_path / "plain.png")

        payload = identify_metadata_record(collect_metadata_record(path), path=path).to_dict()

        assert payload["schema_version"] == PROVENANCE_REPORT_SCHEMA_VERSION == 1
        assert "path" not in payload
        assert json.loads(json.dumps(payload, allow_nan=False)) == payload

    @pytest.mark.parametrize("schema_version", [2, True, 1.0])
    def test_report_rejects_unsupported_output_schema(self, tmp_path: Path, schema_version: object):
        path = _noise_png(tmp_path / "plain.png")
        report = identify_metadata_record(collect_metadata_record(path), path=path)

        with pytest.raises(ValueError, match="Unsupported provenance report schema"):
            report.to_dict(schema_version=schema_version)  # type: ignore[arg-type]

    def test_convenience_entry_point_matches_explicit_sequence(self, tmp_path: Path):
        path = _noise_png(tmp_path / "plain.png")
        record = collect_metadata_record(path)

        explicit = identify_from_evidence(evidence_from_metadata_record(record, path=path))
        convenience = identify_metadata_record(record, path=path)

        assert convenience == explicit

    def test_c2pa_validation_survives_the_portable_record(self, tampered_chatgpt_png: Path):
        direct = identify(tampered_chatgpt_png, check_visible=False, check_invisible=False)
        portable = identify_metadata_record(collect_metadata_record(tampered_chatgpt_png), path=tampered_chatgpt_png)

        assert portable == direct
        assert portable.c2pa_validation is not None
        assert portable.c2pa_validation["integrity"] == "invalid"


def test_a_webp_record_matches(tmp_path: Path):
    """RIFF has its own walk; a chunk kept or dropped wrongly shows up here."""
    path = tmp_path / "image.webp"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path, "WEBP", xmp=b"<x:xmpmeta>plain</x:xmpmeta>")

    assert collect_metadata_record(path)["container"] == "webp"
    _assert_same_verdict(path)


def test_a_webp_record_collects_metadata_after_a_large_frame(tmp_path: Path):
    """The RIFF walker seeks past coded pixels instead of stopping at the head window."""
    path = tmp_path / "late.webp"
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 255, (900, 900, 3), dtype=np.uint8)
    xmp = b"<x:xmpmeta><photoshop:DigitalSourceType>trainedAlgorithmicMedia</photoshop:DigitalSourceType></x:xmpmeta>"
    Image.fromarray(pixels).save(path, "WEBP", lossless=True, xmp=xmp)

    assert path.stat().st_size > HEAD_WINDOW
    _assert_same_verdict(path)


def test_a_webp_record_does_not_carry_animation_frame_pixels(tmp_path: Path):
    """ANMF is a coded-frame container, not a metadata chunk."""
    from remove_ai_watermarks.metadata_record import _riff_regions

    frame = b"jumb c2pa OpenAI trainedAlgorithmicMedia" * 20
    riff = b"RIFF" + (4 + 8 + len(frame)).to_bytes(4, "little") + b"WEBP"
    riff += b"ANMF" + len(frame).to_bytes(4, "little") + frame

    assert frame not in _riff_regions(riff)


def test_an_invalid_short_riff_size_does_not_turn_the_container_into_a_trailer(tmp_path: Path):
    from remove_ai_watermarks.metadata_record import _trailer

    path = tmp_path / "invalid.webp"
    path.write_bytes(b"RIFF\x00\x00\x00\x00WEBPjumb c2pa trainedAlgorithmicMedia")

    assert _trailer(path, "webp") == b""


def test_the_record_never_reopens_the_source(tmp_path: Path, monkeypatch):
    """The reason the record exists: the verdict must run where the file is not."""
    path = FIXTURES / "chatgpt-1.png" if (FIXTURES / "chatgpt-1.png").exists() else _noise_png(tmp_path / "x.png")
    record = json.loads(json.dumps(collect_metadata_record(path)))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("the record path must not open the source file")

    monkeypatch.setattr("builtins.open", fail_if_called)
    monkeypatch.setattr(Path, "open", fail_if_called)
    monkeypatch.setattr(Image, "open", fail_if_called)

    report = identify_from_evidence(evidence_from_metadata_record(record, path=path))

    assert isinstance(report, ProvenanceReport)
