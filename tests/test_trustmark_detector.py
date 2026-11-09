"""Tests for the optional Adobe TrustMark detector.

TrustMark is an optional dependency (extra ``trustmark``) that downloads model
weights on first use, so the decode path is only exercised when it is installed
(mirrors the imwatermark handling). The always-on test pins the graceful
absent/error behavior: detect must return None, never raise.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from remove_ai_watermarks import trustmark_detector
from remove_ai_watermarks.identify import identify
from remove_ai_watermarks.trustmark_detector import (
    detect_trustmark,
    detect_trustmark_detailed,
    detect_trustmark_image,
    is_available,
    remove_trustmark,
)

_OFFICIAL_FIXTURE = (
    Path(__file__).resolve().parent.parent / "data" / "fixtures" / "provenance" / "adobe-trustmark-p.png"
)
_OFFICIAL_FIXTURE_SHA256 = "e58c5825ed7e5d9fb04710ea541b61bd55879cad65554c0d46260aa24b3d0755"


class _FakeDecoder:
    """A TrustMark decoder whose successive ``decode`` calls return scripted
    ``(secret, present, schema)`` tuples -- the first for the original image, the
    second for the re-encoded copy used by the false-positive durability gate."""

    def __init__(self, *results: tuple[str, bool, int]):
        self._results = list(results)
        self.calls = 0
        self.removed = 0

    def decode(self, _img: object, mode: str = "binary") -> tuple[str, bool, int]:
        assert mode == "binary"
        result = self._results[min(self.calls, len(self._results) - 1)]
        self.calls += 1
        return result

    def remove_watermark(self, img: Image.Image) -> Image.Image:
        self.removed += 1
        return img.copy()


def test_detect_never_raises(tmp_clean_png: Path):
    # Whether or not trustmark is installed, a clean image must yield None
    # (no watermark) without raising. When absent, the import guard returns None.
    assert detect_trustmark(tmp_clean_png) is None


def test_unreadable_file_returns_none(tmp_path: Path):
    bad = tmp_path / "not_an_image.txt"
    bad.write_bytes(b"not an image")
    assert detect_trustmark(bad) is None


@pytest.mark.skipif(not is_available(), reason="trustmark not installed")
def test_clean_image_reports_no_watermark(tmp_clean_png: Path):
    # With the decoder present, an un-watermarked image must report absent.
    assert detect_trustmark(tmp_clean_png) is None


class TestFalsePositiveGate:
    """The re-encode durability gate keeps real (durable) TrustMarks and drops
    BCH false positives that collapse under a mild JPEG round-trip."""

    @pytest.fixture(autouse=True)
    def _force_available(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(trustmark_detector, "is_available", lambda: True)

    def _patch_decoder(self, monkeypatch: pytest.MonkeyPatch, decoder: _FakeDecoder) -> None:
        monkeypatch.setattr(trustmark_detector, "_decoder", lambda: decoder)

    def test_durable_watermark_survives_and_is_reported(self, monkeypatch, tmp_clean_png: Path):
        decoder = _FakeDecoder(("secret", True, 2), ("secret", True, 2))
        self._patch_decoder(monkeypatch, decoder)
        result = detect_trustmark(tmp_clean_png)
        assert result == "Adobe TrustMark (variant P, schema 2)"
        assert decoder.calls == 2  # original + re-encode

    def test_already_decoded_image_uses_the_same_gate(self, monkeypatch):
        decoder = _FakeDecoder(("secret", True, 2), ("secret", True, 2))
        self._patch_decoder(monkeypatch, decoder)

        result = detect_trustmark_image(Image.new("RGB", (32, 32)), source="benchmark fixture")

        assert result == "Adobe TrustMark (variant P, schema 2)"
        assert decoder.calls == 2

    def test_false_positive_collapsing_on_reencode_is_dropped(self, monkeypatch, tmp_clean_png: Path):
        # Present on the original, absent after re-encode -> content-noise FP.
        decoder = _FakeDecoder(("01", True, 2), ("", False, -1))
        self._patch_decoder(monkeypatch, decoder)
        assert detect_trustmark(tmp_clean_png) is None

    def test_schema_drift_on_reencode_is_dropped(self, monkeypatch, tmp_clean_png: Path):
        # Present both times but the schema changes -> not a stable watermark.
        decoder = _FakeDecoder(("01", True, 2), ("01", True, 3))
        self._patch_decoder(monkeypatch, decoder)
        assert detect_trustmark(tmp_clean_png) is None

    def test_payload_drift_on_reencode_is_dropped(self, monkeypatch, tmp_clean_png: Path):
        decoder = _FakeDecoder(("first", True, 2), ("second", True, 2))
        self._patch_decoder(monkeypatch, decoder)
        assert detect_trustmark(tmp_clean_png) is None

    def test_weak_schema_three_is_dropped(self, monkeypatch, tmp_clean_png: Path):
        decoder = _FakeDecoder(("secret", True, 3))
        self._patch_decoder(monkeypatch, decoder)
        assert detect_trustmark(tmp_clean_png) is None
        assert decoder.calls == 1

    def test_absent_skips_reencode(self, monkeypatch, tmp_clean_png: Path):
        decoder = _FakeDecoder(("", False, -1))
        self._patch_decoder(monkeypatch, decoder)
        assert detect_trustmark(tmp_clean_png) is None
        assert decoder.calls == 1  # no second decode when the first is absent


@pytest.mark.skipif(not is_available(), reason="trustmark not installed")
def test_official_adobe_variant_p_fixture():
    assert detect_trustmark(_OFFICIAL_FIXTURE) == "Adobe TrustMark (variant P, schema 1)"


@pytest.mark.skipif(not is_available(), reason="trustmark not installed")
def test_official_adobe_variant_p_fixture_is_provenance_not_ai():
    report = identify(_OFFICIAL_FIXTURE, check_visible=False)
    assert report.is_ai_generated is None
    assert report.ai_from_metadata is False
    assert any(signal.name == "trustmark" for signal in report.signals)


def test_official_adobe_variant_p_fixture_digest():
    assert hashlib.sha256(_OFFICIAL_FIXTURE.read_bytes()).hexdigest() == _OFFICIAL_FIXTURE_SHA256


class TestVariantsAndRemoval:
    """Variant selection (`P`, `Q`, `C`), structured detection, and neural removal."""

    @pytest.fixture(autouse=True)
    def _force_available(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(trustmark_detector, "is_available", lambda: True)

    def test_variants_q_and_c_report_their_variant_labels(self, monkeypatch: pytest.MonkeyPatch, tmp_clean_png: Path):
        decoders = {
            "Q": _FakeDecoder(("bits-q", True, 1), ("bits-q", True, 1)),
            "C": _FakeDecoder(("bits-c", True, 2), ("bits-c", True, 2)),
        }
        monkeypatch.setattr(trustmark_detector, "_decoder_for_variant", lambda variant: decoders[variant])

        detailed_q = detect_trustmark_detailed(tmp_clean_png, variant="Q")
        assert detailed_q is not None
        assert detailed_q.variant == "Q"
        assert detailed_q.schema == 1
        assert detailed_q.payload == "bits-q"
        assert detailed_q.label == "Adobe TrustMark (variant Q, schema 1)"
        assert detect_trustmark(tmp_clean_png, variant="C") == "Adobe TrustMark (variant C, schema 2)"

    def test_unsupported_variant_raises_value_error(self, tmp_clean_png: Path):
        with pytest.raises(ValueError, match="unsupported TrustMark variant"):
            detect_trustmark(tmp_clean_png, variant="B")  # type: ignore[arg-type]

    def test_remove_trustmark_removes_detected_watermark_and_verifies_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_clean_png: Path, tmp_path: Path
    ):
        # First two decode calls verify durability before removal; the third verifies
        # that the written output no longer carries a durable watermark.
        decoder = _FakeDecoder(("1010", True, 1), ("1010", True, 1), ("", False, -1))
        monkeypatch.setattr(trustmark_detector, "_decoder", lambda: decoder)
        out = tmp_path / "cleaned.png"

        outcome = remove_trustmark(tmp_clean_png, out)

        assert outcome is not None
        assert outcome.variant == "P"
        assert outcome.schema == 1
        assert outcome.payload == "1010"
        assert outcome.output == out
        assert decoder.removed == 1
        assert out.exists()

    def test_remove_trustmark_returns_none_and_writes_nothing_on_clean_or_surviving_mark(
        self, monkeypatch: pytest.MonkeyPatch, tmp_clean_png: Path, tmp_path: Path
    ):
        clean_decoder = _FakeDecoder(("", False, -1))
        monkeypatch.setattr(trustmark_detector, "_decoder", lambda: clean_decoder)
        clean_out = tmp_path / "clean_out.png"

        assert remove_trustmark(tmp_clean_png, clean_out) is None
        assert clean_decoder.removed == 0
        assert not clean_out.exists()

        surviving_decoder = _FakeDecoder(("1010", True, 1), ("1010", True, 1), ("1010", True, 1), ("1010", True, 1))
        monkeypatch.setattr(trustmark_detector, "_decoder", lambda: surviving_decoder)
        surviving_out = tmp_path / "surviving_out.png"

        assert remove_trustmark(tmp_clean_png, surviving_out) is None
        assert surviving_decoder.removed == 1
        assert not surviving_out.exists()


@pytest.mark.skipif(not is_available(), reason="trustmark not installed")
def test_official_adobe_variant_p_fixture_removal_end_to_end(tmp_path: Path):
    out = tmp_path / "removed.png"
    outcome = remove_trustmark(_OFFICIAL_FIXTURE, out, variant="P")

    assert outcome is not None
    assert outcome.variant == "P"
    assert outcome.schema == 1
    assert outcome.output == out
    assert detect_trustmark(out) is None
