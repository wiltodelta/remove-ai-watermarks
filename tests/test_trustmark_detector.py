"""Tests for the optional Adobe TrustMark detector.

TrustMark is an optional dependency (extra ``trustmark``) that downloads model
weights on first use, so the decode path is only exercised when it is installed
(mirrors the imwatermark handling). The always-on test pins the graceful
absent/error behaviour: detect must return None, never raise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from remove_ai_watermarks import trustmark_detector
from remove_ai_watermarks.trustmark_detector import detect_trustmark, is_available

if TYPE_CHECKING:
    from pathlib import Path


class _FakeDecoder:
    """A TrustMark decoder whose successive ``decode`` calls return scripted
    ``(secret, present, schema)`` tuples -- the first for the original image, the
    second for the re-encoded copy used by the false-positive durability gate."""

    def __init__(self, *results: tuple[bytes, bool, int]):
        self._results = list(results)
        self.calls = 0

    def decode(self, _img: object) -> tuple[bytes, bool, int]:
        result = self._results[min(self.calls, len(self._results) - 1)]
        self.calls += 1
        return result


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
        decoder = _FakeDecoder((b"secret", True, 2), (b"secret", True, 2))
        self._patch_decoder(monkeypatch, decoder)
        result = detect_trustmark(tmp_clean_png)
        assert result == "Adobe TrustMark (variant P, schema 2)"
        assert decoder.calls == 2  # original + re-encode

    def test_false_positive_collapsing_on_reencode_is_dropped(self, monkeypatch, tmp_clean_png: Path):
        # Present on the original, absent after re-encode -> content-noise FP.
        decoder = _FakeDecoder((b"\x00\x01", True, 3), (b"", False, -1))
        self._patch_decoder(monkeypatch, decoder)
        assert detect_trustmark(tmp_clean_png) is None

    def test_schema_drift_on_reencode_is_dropped(self, monkeypatch, tmp_clean_png: Path):
        # Present both times but the schema changes -> not a stable watermark.
        decoder = _FakeDecoder((b"\x00", True, 2), (b"\x00", True, 3))
        self._patch_decoder(monkeypatch, decoder)
        assert detect_trustmark(tmp_clean_png) is None

    def test_absent_skips_reencode(self, monkeypatch, tmp_clean_png: Path):
        decoder = _FakeDecoder((b"", False, -1))
        self._patch_decoder(monkeypatch, decoder)
        assert detect_trustmark(tmp_clean_png) is None
        assert decoder.calls == 1  # no second decode when the first is absent
