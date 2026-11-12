"""Physics-invariant and regression tests for the 512-d v4 feature extractor."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from remove_ai_watermarks._internal.source_features_v4 import (
    FEATURE_DIM_V4,
    REGIME_SLICE_V4,
    extract_v4_features,
    extract_v4_features_from_rgb,
)


def test_size_boundary_and_determinism() -> None:
    rng = np.random.default_rng(42)
    small = Image.fromarray(rng.integers(0, 256, size=(256, 255, 3), dtype=np.uint8))
    assert extract_v4_features(small) is None
    assert extract_v4_features_from_rgb(rng.integers(0, 256, size=(255, 512, 3), dtype=np.uint8)) is None

    valid = Image.fromarray(rng.integers(20, 235, size=(256, 256, 3), dtype=np.uint8))
    v1 = extract_v4_features(valid)
    v2 = extract_v4_features(valid)
    assert v1 is not None
    assert v2 is not None
    assert v1.shape == (FEATURE_DIM_V4,)
    assert np.all(np.isfinite(v1))
    np.testing.assert_array_equal(v1, v2)


def test_sub512_images_retain_nonzero_intertile_spread() -> None:
    """Images in [256..511] px must keep >= 9 adaptive tiles and non-zero spread (fixes Bug #1)."""
    rng = np.random.default_rng(101)
    for side in (256, 384):
        canvas = np.full((side, side, 3), 128, dtype=np.uint8)
        # Top-left quadrant smooth, bottom-right quadrant noisy
        noise = rng.integers(-25, 26, size=(side // 2, side // 2, 3))
        canvas[side // 2 :, side // 2 :] = np.clip(128 + noise, 0, 255).astype(np.uint8)

        vec = extract_v4_features_from_rgb(canvas)
        assert vec is not None
        iqr_slice = vec[96:102]
        std_slice = vec[102:108]
        assert float(np.max(iqr_slice)) > 0.05
        assert float(np.max(std_slice)) > 0.05


def test_cropped_shifted_grid_detected_by_secondary_phase() -> None:
    """A shifted (3, 5) 8x8 grid before JPEG lights up secondary phase & r_grid_shift_score."""
    rng = np.random.default_rng(202)
    h, w = 384, 384
    base = rng.normal(128.0, 6.0, size=(h + 8, w + 8, 3))
    # Inject a subtle 8x8 block step pattern (simulating prior 8x8 lattice)
    yy = np.arange(h + 8)[:, None, None]
    xx = np.arange(w + 8)[None, :, None]
    block_step = ((yy // 8 + xx // 8) % 2) * 5.0
    img_full = np.clip(base + block_step, 0, 255).astype(np.uint8)

    aligned = Image.fromarray(img_full[:h, :w])
    shifted = Image.fromarray(img_full[3 : 3 + h, 5 : 5 + w])

    # Re-save both as Q88 JPEG so origin (0, 0) carries standard JPEG seams in both
    def _save_jpeg(im: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88, subsampling=2)
        buf.seek(0)
        with Image.open(buf) as dec:
            return dec.convert("RGB")

    v_aligned = extract_v4_features(_save_jpeg(aligned))
    v_shifted = extract_v4_features(_save_jpeg(shifted))
    assert v_aligned is not None
    assert v_shifted is not None

    # r_grid_shift_score is index 9 of REGIME_SLICE_V4 (dim 489)
    reg_aligned = v_aligned[REGIME_SLICE_V4]
    reg_shifted = v_shifted[REGIME_SLICE_V4]
    assert reg_shifted[9] > reg_aligned[9] + 0.05


def test_interior_vae_modulation_survives_jpeg_420() -> None:
    """Odd-even modulation strictly inside {1..6}^2 survives Q82 4:2:0 JPEG in Block B3."""
    rng = np.random.default_rng(303)
    h, w = 384, 384
    smooth = rng.normal(128.0, 3.0, size=(h, w, 3))

    yy, _xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    # Spatially varying amplitude modulated at stride-2 (so odd/even pixels have different residual variance)
    odd_mask = ((yy % 2) == 1).astype(np.float64)[:, :, None]
    vae_mod = smooth + odd_mask * rng.normal(0.0, 3.5, size=(h, w, 3))

    def _jpeg82(arr: np.ndarray) -> np.ndarray:
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82, subsampling=2)
        buf.seek(0)
        with Image.open(buf) as dec:
            return np.asarray(dec.convert("RGB"), dtype=np.uint8)

    v_plain = extract_v4_features_from_rgb(_jpeg82(smooth))
    v_vae = extract_v4_features_from_rgb(_jpeg82(vae_mod))
    assert v_plain is not None
    assert v_vae is not None

    # Block B3 spans indices 160..191 (112 + 24 + 24 = 160)
    b3_diff = float(np.linalg.norm(v_vae[160:192] - v_plain[160:192]))
    assert b3_diff > 0.08


def test_moisan_periodic_decomposition_suppresses_linear_ramp_cross() -> None:
    """A pure smooth linear ramp (strong opposite-edge jump) must not inject cardinal-vs-diagonal cross artifacts."""
    h, w = 256, 256
    yy = np.linspace(20.0, 235.0, h)[:, None]
    xx = np.linspace(30.0, 220.0, w)[None, :]
    ramp = np.clip(0.6 * yy + 0.4 * xx, 0, 255)
    rgb = np.stack([ramp, ramp, ramp], axis=-1).astype(np.uint8)

    vec = extract_v4_features_from_rgb(rgb)
    assert vec is not None
    # Block C2 (anisotropy cells for Luma) starts at 240 + 48 = 288 .. 304
    luma_aniso = vec[288:304]
    assert float(np.max(np.abs(luma_aniso))) < 0.25


def test_continuous_comb_scanner_tracks_non_integer_resized_grid() -> None:
    """A T=8 periodic comb downscaled by 0.75x (T=6 px -> k~42.7) activates resized_comb without inverting."""
    rng = np.random.default_rng(505)
    h0, w0 = 512, 512
    yy, xx = np.meshgrid(np.arange(h0), np.arange(w0), indexing="ij")
    comb_native = (
        128.0
        + rng.normal(0.0, 8.0, size=(h0, w0))
        + 6.0 * (np.cos(2.0 * np.pi * yy / 8.0) + np.cos(2.0 * np.pi * xx / 8.0))
    )
    rgb_nat = np.stack([comb_native] * 3, axis=-1)
    im_nat = Image.fromarray(np.clip(rgb_nat, 0, 255).astype(np.uint8))
    im_res = im_nat.resize((384, 384), Image.Resampling.BICUBIC)

    im_ctrl = Image.fromarray(
        np.clip(np.stack([128.0 + rng.normal(0.0, 8.0, size=(384, 384))] * 3, axis=-1), 0, 255).astype(np.uint8)
    )

    v_res = extract_v4_features(im_res)
    v_ctrl = extract_v4_features(im_ctrl)
    assert v_res is not None
    assert v_ctrl is not None

    reg_res = v_res[REGIME_SLICE_V4]
    reg_ctrl = v_ctrl[REGIME_SLICE_V4]
    # resized_comb is index 10 of REGIME_SLICE_V4, est_scale is index 11 (43 / 32 ~= 1.34)
    assert reg_res[10] > reg_ctrl[10] + 0.15
    assert 1.25 <= reg_res[11] <= 1.42


def test_shot_noise_slope_distinguishes_poisson_sensor_from_inverted_noise() -> None:
    """Block D3 slope (dim 444) is positive when noise scales with intensity and negative when inverted."""
    rng = np.random.default_rng(606)
    h, w = 384, 384
    # Smooth intensity tiles spanning [25..225]
    grad = np.repeat(np.linspace(25.0, 225.0, h)[:, None], w, axis=1)

    std_pos = 0.4 + 2.8 * np.sqrt(grad / 255.0)
    std_neg = 3.2 - 2.8 * np.sqrt(grad / 255.0)

    rgb_pos = np.clip(grad[:, :, None] + rng.normal(0.0, 1.0, size=(h, w, 3)) * std_pos[:, :, None], 0, 255).astype(
        np.uint8
    )
    rgb_neg = np.clip(grad[:, :, None] + rng.normal(0.0, 1.0, size=(h, w, 3)) * std_neg[:, :, None], 0, 255).astype(
        np.uint8
    )

    v_pos = extract_v4_features_from_rgb(rgb_pos)
    v_neg = extract_v4_features_from_rgb(rgb_neg)
    assert v_pos is not None
    assert v_neg is not None

    # Block D3 starts at 384 + 36 + 24 = 444; dim 444 is Luma shot-noise slope
    assert v_pos[444] > 0.3
    assert v_neg[444] < -0.3
