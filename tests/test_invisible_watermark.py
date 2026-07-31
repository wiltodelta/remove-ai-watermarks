"""Tests for open DWT-DCT invisible-watermark detection.

Parity (``TestBackendParity``) is the required gate: vendored ``dwt_dct`` must
match upstream ``imwatermark`` bit-for-bit on committed images under
``data/fixtures/provenance/``. Install extras: ``detect`` (imwatermark) and/or
``detect-pywavelets`` (PyWavelets); ``dev`` ships both so CI always runs parity.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from remove_ai_watermarks.image_io import imread
from remove_ai_watermarks.invisible_watermark import (
    _BITS_48,
    _SD1_STRING,
    _bits_match,
    _bytes_match_frac,
    _decoder_backend,
    detect_invisible_watermark,
    is_available,
)
from remove_ai_watermarks.optional_deps import module_available

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "provenance"
# Midjourney sample: no open DWT-DCT mark, and a known-good carrier for embed
# round-trips (smoke_matrix uses the same file for the re-embed control).
CARRIER = SAMPLES_DIR / "mj-1.png"

pytestmark = pytest.mark.skipif(not is_available(), reason="neither detect nor detect-pywavelets backend installed")

_has_imwatermark = module_available("imwatermark")
_has_pywt = module_available("pywt")
requires_imwatermark = pytest.mark.skipif(
    not _has_imwatermark, reason="invisible-watermark (extra detect / dev) not installed"
)
requires_pywt = pytest.mark.skipif(not _has_pywt, reason="PyWavelets (extra detect-pywavelets / dev) not installed")
requires_parity_backends = pytest.mark.skipif(
    not (_has_imwatermark and _has_pywt),
    reason="parity requires both imwatermark and PyWavelets (extra dev)",
)
requires_samples = pytest.mark.skipif(not SAMPLES_DIR.is_dir(), reason="data/fixtures/provenance not present")
requires_carrier = pytest.mark.skipif(not CARRIER.is_file(), reason="mj-1.png fixture not present")


def _fixture_images() -> list[Path]:
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted(
        p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and p.is_file()
    )


def _carrier_bgr() -> np.ndarray:
    img = imread(CARRIER)
    assert img is not None, f"failed to read {CARRIER}"
    return img


def _write_bits_on_carrier(tmp_path: Path, message: int) -> Path:
    from imwatermark import WatermarkEncoder

    bits = [int(b) for b in format(message, "048b")]
    enc = WatermarkEncoder()
    enc.set_watermark("bits", bits)
    wm = enc.encode(_carrier_bgr(), "dwtDct")
    path = tmp_path / "wm.png"
    cv2.imwrite(str(path), wm)
    return path


def _as_bool_bits(bits: object) -> np.ndarray:
    return np.asarray([bool(b) for b in bits], dtype=bool)  # type: ignore[attr-defined]


def _decode_both_48(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from imwatermark import WatermarkDecoder

    from remove_ai_watermarks.dwt_dct import decode_dwt_dct

    upstream = _as_bool_bits(WatermarkDecoder("bits", 48).decode(img, "dwtDct"))
    ours = _as_bool_bits(decode_dwt_dct(img, wm_len=48))
    return ours, upstream


class TestHelpers:
    def test_bits_match_exact(self):
        assert _bits_match(0b1010, 0b1010, width=4) == 4

    def test_bits_match_one_off(self):
        assert _bits_match(0b1010, 0b1011, width=4) == 3

    def test_bytes_match_identical(self):
        assert _bytes_match_frac(_SD1_STRING, _SD1_STRING) == 1.0

    def test_bytes_match_length_mismatch_is_zero(self):
        assert _bytes_match_frac(b"abc", b"abcd") == 0.0


class TestDecoderBackend:
    """``[detect]`` must keep using imwatermark even when PyWavelets is present."""

    def test_prefers_imwatermark_when_both_present(self, monkeypatch: pytest.MonkeyPatch):
        def fake_available(name: str) -> bool:
            return name in {"imwatermark", "pywt"}

        monkeypatch.setattr(
            "remove_ai_watermarks.optional_deps.module_available",
            lambda *names: all(fake_available(n) for n in names),
        )
        assert _decoder_backend() == "imwatermark"

    def test_uses_dwt_dct_when_only_pywt(self, monkeypatch: pytest.MonkeyPatch):
        def fake_available(name: str) -> bool:
            return name == "pywt"

        monkeypatch.setattr(
            "remove_ai_watermarks.optional_deps.module_available",
            lambda *names: all(fake_available(n) for n in names),
        )
        assert _decoder_backend() == "dwt_dct"

    def test_none_when_neither(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "remove_ai_watermarks.optional_deps.module_available",
            lambda *names: False,
        )
        assert _decoder_backend() is None


@requires_imwatermark
@requires_carrier
class TestDetect:
    def test_detects_sdxl(self, tmp_path: Path):
        path = _write_bits_on_carrier(tmp_path, _BITS_48["Stable Diffusion XL"])
        assert detect_invisible_watermark(path) == "Stable Diffusion XL"

    def test_detects_flux(self, tmp_path: Path):
        path = _write_bits_on_carrier(tmp_path, _BITS_48["FLUX.2 (Black Forest Labs)"])
        assert detect_invisible_watermark(path) == "FLUX.2 (Black Forest Labs)"

    def test_detects_sd1_string(self, tmp_path: Path):
        from imwatermark import WatermarkEncoder

        enc = WatermarkEncoder()
        enc.set_watermark("bytes", _SD1_STRING)
        wm = enc.encode(_carrier_bgr(), "dwtDct")
        path = tmp_path / "sd1.png"
        cv2.imwrite(str(path), wm)
        assert detect_invisible_watermark(path) == "Stable Diffusion 1.x / 2.x"

    def test_mj_fixture_has_no_open_watermark(self):
        assert detect_invisible_watermark(CARRIER) is None

    def test_unreadable_file_is_none(self, tmp_path: Path):
        path = tmp_path / "not_image.png"
        path.write_bytes(b"not a png")
        assert detect_invisible_watermark(path) is None


@requires_parity_backends
@requires_samples
class TestBackendParity:
    """Mandatory gate: vendored dwt_dct must match imwatermark on repo fixtures."""

    @pytest.mark.parametrize("fixture", _fixture_images(), ids=lambda p: p.name)
    def test_decode_bits_match_on_fixture(self, fixture: Path):
        img = imread(fixture)
        assert img is not None, f"failed to read {fixture}"
        if min(img.shape[:2]) * max(img.shape[:2]) < 256 * 256:
            pytest.skip(f"{fixture.name} too small for dwtDct")
        ours, upstream = _decode_both_48(img)
        assert np.array_equal(ours, upstream), (
            f"{fixture.name}: ours ones={int(ours.sum())}/48 upstream ones={int(upstream.sum())}/48"
        )

    @pytest.mark.parametrize(
        "scheme",
        ["Stable Diffusion XL", "FLUX.2 (Black Forest Labs)"],
    )
    @requires_carrier
    def test_decode_matches_after_upstream_embed_on_mj(self, scheme: str):
        from imwatermark import WatermarkEncoder

        bits = [int(b) for b in format(_BITS_48[scheme], "048b")]
        enc = WatermarkEncoder()
        enc.set_watermark("bits", bits)
        wm = enc.encode(_carrier_bgr(), "dwtDct")
        ours, upstream = _decode_both_48(wm)
        assert np.array_equal(ours, upstream)
        assert np.array_equal(ours, np.asarray(bits, dtype=bool))

    @requires_carrier
    @requires_pywt
    def test_decode_matches_after_vendored_embed_on_mj(self):
        from remove_ai_watermarks.dwt_dct import encode_dwt_dct

        bits = [int(b) for b in format(_BITS_48["Stable Diffusion XL"], "048b")]
        wm = encode_dwt_dct(_carrier_bgr(), bits)
        ours, upstream = _decode_both_48(wm)
        assert np.array_equal(ours, upstream)
        assert np.array_equal(ours, np.asarray(bits, dtype=bool))

    @requires_carrier
    def test_sd1_string_decode_matches_on_mj(self):
        from imwatermark import WatermarkDecoder, WatermarkEncoder

        from remove_ai_watermarks.dwt_dct import decode_dwt_dct

        enc = WatermarkEncoder()
        enc.set_watermark("bytes", _SD1_STRING)
        wm = enc.encode(_carrier_bgr(), "dwtDct")
        wm_len = 8 * len(_SD1_STRING)
        upstream = _as_bool_bits(WatermarkDecoder("bits", wm_len).decode(wm, "dwtDct"))
        ours = _as_bool_bits(decode_dwt_dct(wm, wm_len=wm_len))
        assert np.array_equal(ours, upstream)
