"""512-d physics-grounded pixel and spectral descriptor for source-classify v4.

Organizes image evidence into five deterministic blocks without Torch or ONNX:
- Block A (0..111,   112-d): Adaptive overlapping native-tile truncated SRM micro-residuals
  split across flat and textured regions (guarantees >= 9 tiles even at 256x256).
- Block B (112..239, 128-d): Phase-resolved modulo-8 / modulo-16 sub-lattice energy,
  cropped-grid secondary-phase detection, strictly interior {1..6}^2 VAE modulation,
  and 8x8 block-DCT AC quantization / comb statistics.
- Block C (240..383, 144-d): Moisan (2011) Periodic-Plus-Smooth (p + s) decomposition on
  the non-redundant rfft2 half-plane, 1D radial decay profile, and continuous
  scale-invariant period-comb scanner over T in [4.5, 18.3] px.
- Block D (384..479,  96-d): Patch-averaged cross-channel spectral phase coherence,
  Bayer/Quad-Bayer CFA periodicity, heteroscedastic Poisson-Gaussian shot-noise slope,
  and radial chromatic-aberration alignment.
- Block E (480..511,  32-d): Explicit deterministic regime and geometry vector r(x)
  conditioning the v4 router (lossless vs JPEG Q, 4:2:0/4:2:2/4:4:4, WebP 4x4,
  shifted grid, non-integer resize, resolution/lattice alignment, document/UI sparsity).
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from remove_ai_watermarks._internal.image_math import dct_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL import Image

MIN_SIDE_V4 = 256
MAX_ANALYSIS_SIDE = 2048
SPECTRAL_TILE = 256
FEATURE_DIM_V4 = 512
REGIME_SLICE_V4 = slice(480, 512)
REGIME_DIM_V4 = 32
EPS = 1e-8


def _f(val: Any) -> float:
    """Narrow a scalar numpy/python numeric value to builtin float."""
    return float(np.asarray(val, dtype=np.float64).item())


_DCT_AC_COORDS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (1, 1),
    (0, 2),
    (2, 0),
    (0, 3),
    (3, 0),
    (2, 2),
    (0, 4),
    (4, 0),
    (4, 4),
    (7, 7),
)

_AUTOCORR_LAGS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
    (0, 2),
    (2, 0),
    (0, 4),
    (4, 0),
)


@functools.lru_cache(maxsize=1)
def _srm_kernels() -> tuple[NDArray[np.float32], ...]:
    k_h1 = np.array([[0.0, -1.0, 1.0]], dtype=np.float32)
    k_v1 = np.array([[0.0], [-1.0], [1.0]], dtype=np.float32)
    k_h2 = np.array([[-0.5, 1.0, -0.5]], dtype=np.float32)
    k_v2 = np.array([[-0.5], [1.0], [-0.5]], dtype=np.float32)
    k_lap3 = (
        np.array(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, 8.0, -1.0],
                [-1.0, -1.0, -1.0],
            ],
            dtype=np.float32,
        )
        / 4.0
    )
    k_kb5 = (
        np.array(
            [
                [-1.0, 2.0, -2.0, 2.0, -1.0],
                [2.0, -6.0, 8.0, -6.0, 2.0],
                [-2.0, 8.0, -12.0, 8.0, -2.0],
                [2.0, -6.0, 8.0, -6.0, 2.0],
                [-1.0, 2.0, -2.0, 2.0, -1.0],
            ],
            dtype=np.float32,
        )
        / 4.0
    )
    return (k_h1, k_v1, k_h2, k_v2, k_lap3, k_kb5)


@functools.lru_cache(maxsize=1)
def _dct_basis_and_ac() -> tuple[NDArray[np.float32], NDArray[np.intp], NDArray[np.intp]]:
    basis = np.asarray(dct_matrix(np, 8), dtype=np.float32)
    r_idx = np.array([r for r, _ in _DCT_AC_COORDS], dtype=np.intp)
    c_idx = np.array([c for _, c in _DCT_AC_COORDS], dtype=np.intp)
    return basis, r_idx, c_idx


@functools.lru_cache(maxsize=2)
def _moisan_rfft_geometry(
    size: int = SPECTRAL_TILE,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.float64],
    tuple[NDArray[np.bool_], ...],
    tuple[NDArray[np.bool_], ...],
]:
    """Precompute exact Moisan (p + s) denominator, multiplicity weights, rings, and sectors on rfft2."""
    half_w = size // 2 + 1
    ky = (2.0 * math.pi * np.fft.fftfreq(size))[:, None]
    kx = (2.0 * math.pi * np.arange(half_w, dtype=np.float64) / size)[None, :]
    denom = 2.0 * np.cos(ky) + 2.0 * np.cos(kx) - 4.0
    denom[0, 0] = 1.0

    mult = np.full((size, half_w), 2.0, dtype=np.float64)
    mult[:, 0] = 1.0
    if size % 2 == 0:
        mult[:, -1] = 1.0

    fy = np.fft.fftfreq(size)[:, None] * size
    fx = np.arange(half_w, dtype=np.float64)[None, :]
    radius = np.hypot(fy, fx)
    max_r = math.ceil(float(radius.max())) + 1
    r_int = np.clip(np.rint(radius).astype(np.int32), 0, max_r - 1)
    r_weight_sum = np.bincount(r_int.ravel(), weights=mult.ravel(), minlength=max_r).astype(np.float64)
    r_weight_sum = np.maximum(r_weight_sum, 1.0)

    ring_edges = np.geomspace(2.0, size * 0.70710678, 13)
    rings_12 = tuple((radius >= ring_edges[i]) & (radius < ring_edges[i + 1]) for i in range(12))

    angle = np.mod(np.arctan2(fy, fx), math.pi)
    sec_0 = (angle < math.pi / 8.0) | (angle >= 7.0 * math.pi / 8.0)
    sec_45 = (angle >= math.pi / 8.0) & (angle < 3.0 * math.pi / 8.0)
    sec_90 = (angle >= 3.0 * math.pi / 8.0) & (angle < 5.0 * math.pi / 8.0)
    sec_135 = (angle >= 5.0 * math.pi / 8.0) & (angle < 7.0 * math.pi / 8.0)
    sectors = (sec_0, sec_45, sec_90, sec_135)

    coarse_edges = (3.0, 12.0, 32.0, 72.0, 126.0)
    cells_16: list[NDArray[np.bool_]] = []
    for r_idx in range(4):
        r_mask = (radius >= coarse_edges[r_idx]) & (radius < coarse_edges[r_idx + 1])
        for s_mask in sectors:
            cells_16.append(r_mask & s_mask)

    return denom, mult, r_int, r_weight_sum, rings_12, tuple(cells_16)


def _moisan_periodic_rfft2(tile_batch: NDArray[np.float32], denom: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Compute exact 2D rfft of Moisan's periodic component for a batch (N, P, P)."""
    tb64 = tile_batch.astype(np.float64)
    v = np.zeros_like(tb64)
    jump_tb = tb64[:, -1, :] - tb64[:, 0, :]
    v[:, 0, :] += jump_tb
    v[:, -1, :] -= jump_tb
    jump_lr = tb64[:, :, -1] - tb64[:, :, 0]
    v[:, :, 0] += jump_lr
    v[:, :, -1] -= jump_lr

    u_hat = np.fft.rfft2(tb64)
    v_hat = np.fft.rfft2(v)
    s_hat = v_hat / denom[None, :, :]
    s_hat[:, 0, 0] = 0.0
    return u_hat - s_hat


def _lattice_crop_cap(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Cap extreme dimensions at MAX_ANALYSIS_SIDE via a lattice-aligned (% 16 == 0) crop, never resizing."""
    h, w = rgb.shape[:2]
    if h <= MAX_ANALYSIS_SIDE and w <= MAX_ANALYSIS_SIDE:
        return rgb
    new_h = min(h, MAX_ANALYSIS_SIDE)
    new_w = min(w, MAX_ANALYSIS_SIDE)
    y0 = ((h - new_h) // 2) // 16 * 16
    x0 = ((w - new_w) // 2) // 16 * 16
    return rgb[y0 : y0 + new_h, x0 : x0 + new_w]


def _adaptive_tile_grid(height: int, width: int) -> tuple[int, list[tuple[int, int]]]:
    """Return (tile_size, origins) guaranteeing 9..16 lattice-aligned (% 8 == 0) tiles for all >= 256x256 inputs."""
    min_dim = min(height, width)
    tile = 128 if min_dim < 512 else 256

    def _axis_positions(length: int) -> list[int]:
        span = length - tile
        if span <= 0:
            return [0]
        n_steps = 3 if length < tile * 3 else 4
        raw_pos = [round(i * span / (n_steps - 1)) for i in range(n_steps)]
        aligned: list[int] = []
        for idx, pos in enumerate(raw_pos):
            snapped = min((pos // 8) * 8, span)
            if idx == n_steps - 1 and span >= 8:
                snapped = (span // 8) * 8
            aligned.append(snapped)
        return list(dict.fromkeys(aligned))

    ys = _axis_positions(height)
    xs = _axis_positions(width)
    return tile, [(y, x) for y in ys for x in xs]


def _spectral_256_origins(height: int, width: int, max_tiles: int = 9) -> list[tuple[int, int]]:
    """Select up to 9 lattice-aligned (% 16 == 0) 256x256 tiles covering the image."""

    def _axis_pos(length: int) -> list[int]:
        span = length - SPECTRAL_TILE
        if span < 16:
            return [0]
        n_steps = 2 if span < 256 else 3
        pts = [min(((round(i * span / (n_steps - 1))) // 16) * 16, span) for i in range(n_steps)]
        return list(dict.fromkeys(pts))

    ys = _axis_pos(height)
    xs = _axis_pos(width)
    return [(y, x) for y in ys for x in xs][:max_tiles]


def _opponent_planes_255(rgb: NDArray[np.uint8]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return float32 [0..255]-scale rgb_f32 and opponent planes (Y, R-G, B-0.5*(R+G))."""
    vals = rgb.astype(np.float32)
    r = vals[:, :, 0]
    g = vals[:, :, 1]
    b = vals[:, :, 2]
    planes = np.stack(
        [
            0.299 * r + 0.587 * g + 0.114 * b,
            r - g,
            b - 0.5 * (r + g),
        ],
        axis=-1,
    )
    return vals, planes


def _block_a_srm_and_tiles(
    planes: NDArray[np.float32],
    lap_raw: NDArray[np.float32],
) -> tuple[NDArray[np.float64], float]:
    """Compute Block A (112-d) adaptive overlapping tile truncated SRM statistics + flat_fraction."""
    h, w = planes.shape[:2]
    tile, origins = _adaptive_tile_grid(h, w)
    kernels = _srm_kernels()
    n_tiles = len(origins)

    k5_1d = np.array([-0.5, 1.0, -1.0, 1.0, -0.5], dtype=np.float32)
    raw_maps: list[NDArray[np.float32]] = []
    for k_i, k in enumerate(kernels):
        if k_i == 4:
            raw_maps.append(lap_raw)
        elif k_i == 5:
            raw_maps.append(
                np.asarray(
                    -cv2.sepFilter2D(planes, cv2.CV_32F, k5_1d, k5_1d, borderType=cv2.BORDER_REFLECT_101),
                    dtype=np.float32,
                )
            )
        else:
            raw_maps.append(
                np.asarray(
                    cv2.filter2D(planes, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT_101),
                    dtype=np.float32,
                )
            )

    grad_sq_y = raw_maps[0][:, :, 0] ** 2 + raw_maps[1][:, :, 0] ** 2
    tile_tex = np.array(
        [float(grad_sq_y[y : y + tile, x : x + tile].mean()) for y, x in origins],
        dtype=np.float64,
    )
    flat_fraction = float(np.mean(tile_tex < 25.0))

    order = np.argsort(tile_tex, kind="mergesort")
    half = max(1, n_tiles // 2)
    flat_idx = order[:half]
    tex_idx = order[n_tiles - half :]

    rms_t3 = np.empty((n_tiles, 6, 3), dtype=np.float64)
    rms_t15_sel = np.empty((n_tiles, 3, 3), dtype=np.float64)
    sel_k = (2, 4, 5)

    # Precompute contiguous clamped squared maps once (avoids 144 non-contiguous 3D clip+mul allocations)
    sq_t3_maps = [np.minimum(m * m, 9.0) for m in raw_maps]
    sq_t15_maps = [np.minimum(raw_maps[k_i] * raw_maps[k_i], 2.25) for k_i in sel_k]

    for t_i, (y, x) in enumerate(origins):
        for k_i in range(6):
            tile_m3 = sq_t3_maps[k_i][y : y + tile, x : x + tile].mean(axis=(0, 1), dtype=np.float64)
            rms_t3[t_i, k_i] = np.sqrt(tile_m3 + EPS)
        for s_i in range(3):
            tile_m15 = sq_t15_maps[s_i][y : y + tile, x : x + tile].mean(axis=(0, 1), dtype=np.float64)
            rms_t15_sel[t_i, s_i] = np.sqrt(tile_m15 + EPS)

    a1_flat = np.log1p(np.median(rms_t3[flat_idx], axis=0)).ravel()
    a1_tex = np.log1p(np.median(rms_t3[tex_idx], axis=0)).ravel()

    flat_sel = np.median(rms_t3[flat_idx][:, sel_k, :], axis=0)
    tex_sel = np.median(rms_t3[tex_idx][:, sel_k, :], axis=0)
    flat_t15 = np.median(rms_t15_sel[flat_idx], axis=0)
    a2_ratio_ft = (flat_sel / (tex_sel + 1e-4)).ravel()
    a2_ratio_q = (flat_t15 / (flat_sel + 1e-4)).ravel()

    # Subsample 2x2 inside flat tiles for order statistics (kurtosis / quantiles) -> 4x faster, identical distribution
    step_q = 2 if tile >= 256 else 1
    a3_list: list[float] = []
    for k_i in (4, 5):
        raw_patches = np.stack(
            [
                raw_maps[k_i][
                    origins[int(i)][0] : origins[int(i)][0] + tile : step_q,
                    origins[int(i)][1] : origins[int(i)][1] + tile : step_q,
                ]
                for i in flat_idx
            ],
            axis=0,
        )
        t3_patches = np.clip(raw_patches, -3.0, 3.0)
        for c in range(3):
            vals_t3 = t3_patches[:, :, :, c].ravel().astype(np.float64)
            vals_abs = np.abs(raw_patches[:, :, :, c]).ravel()
            std_v = float(vals_t3.std())
            if std_v > 1e-6:
                z = (vals_t3 - float(vals_t3.mean())) / std_v
                kurt = float(np.clip(np.mean(z**4) - 3.0, -3.0, 25.0))
            else:
                kurt = 0.0
            abs_t3 = np.abs(vals_t3)
            n_v = abs_t3.size
            part = np.partition(abs_t3, (n_v // 2, int(0.90 * (n_v - 1))))
            q50 = float(part[n_v // 2])
            q90 = float(part[int(0.90 * (n_v - 1))])
            q_ratio = float(q90 / (q50 + 1e-4))
            sat_frac = float(np.mean(vals_abs > 3.0))
            a3_list.extend((kurt / 10.0, min(q_ratio, 20.0) / 5.0, sat_frac))

    lap_t3_full = np.clip(lap_raw, -3.0, 3.0)
    flat_patches = np.stack(
        [
            lap_t3_full[
                origins[int(t_i)][0] : origins[int(t_i)][0] + tile,
                origins[int(t_i)][1] : origins[int(t_i)][1] + tile,
            ]
            for t_i in flat_idx
        ],
        axis=0,
    )
    centred = flat_patches - np.mean(flat_patches, axis=(1, 2), keepdims=True)
    energy = np.sum(centred * centred, axis=(1, 2), dtype=np.float64) + EPS
    lag_acc = np.empty((len(flat_idx), 3, len(_AUTOCORR_LAGS)), dtype=np.float64)
    for l_i, (dy, dx) in enumerate(_AUTOCORR_LAGS):
        y_src = slice(0, tile - dy) if dy > 0 else slice(0, tile)
        y_dst = slice(dy, tile) if dy > 0 else slice(0, tile)
        if dx >= 0:
            x_src = slice(0, tile - dx)
            x_dst = slice(dx, tile)
        else:
            x_src = slice(-dx, tile)
            x_dst = slice(0, tile + dx)
        cov = np.sum(centred[:, y_src, x_src, :] * centred[:, y_dst, x_dst, :], axis=(1, 2), dtype=np.float64)
        lag_acc[:, :, l_i] = cov / energy
    a4_autocorr = np.median(lag_acc, axis=0).ravel()

    log_rms_lk = np.log1p(rms_t3[:, (4, 5), :])
    q25, q75 = np.quantile(log_rms_lk, (0.25, 0.75), axis=0)
    iqr_lk = (q75 - q25).ravel()
    std_lk = np.std(log_rms_lk, axis=0).ravel()

    cc_01 = np.empty(n_tiles, dtype=np.float64)
    cc_02 = np.empty(n_tiles, dtype=np.float64)
    for t_i, (y, x) in enumerate(origins):
        p = lap_t3_full[y : y + tile : 2, x : x + tile : 2].reshape(-1, 3).astype(np.float64)
        p_c = p - np.mean(p, axis=0, keepdims=True)
        norms = np.sqrt(np.sum(p_c * p_c, axis=0)) + EPS
        cc_01[t_i] = float(np.dot(p_c[:, 0], p_c[:, 1]) / (norms[0] * norms[1]))
        cc_02[t_i] = float(np.dot(p_c[:, 0], p_c[:, 2]) / (norms[0] * norms[2]))

    cc_feats = np.array(
        [
            float(np.median(cc_01)),
            float(np.quantile(cc_01, 0.75) - np.quantile(cc_01, 0.25)),
            float(np.median(cc_02)),
            float(np.quantile(cc_02, 0.75) - np.quantile(cc_02, 0.25)),
        ],
        dtype=np.float64,
    )

    vec_a = np.concatenate(
        [
            a1_flat,
            a1_tex,
            a2_ratio_ft,
            a2_ratio_q,
            np.asarray(a3_list, dtype=np.float64),
            a4_autocorr,
            iqr_lk,
            std_lk,
            cc_feats,
        ]
    )
    if vec_a.shape != (112,):
        raise ValueError(f"Expected Block A shape (112,), got {vec_a.shape}")
    return vec_a, flat_fraction


def _phase_entropy(profile: NDArray[np.float64]) -> float:
    total = float(profile.sum())
    if total <= EPS:
        return 1.0
    p = np.maximum(profile / total, 1e-12)
    return float(-np.sum(p * np.log(p)) / math.log(len(profile)))


def _block_b_sublattice_and_dct(
    planes: NDArray[np.float32],
    lap_raw: NDArray[np.float32],
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Compute Block B (128-d): modulo-8/16 phase physics, interior {1..6}^2 VAE modulation, and 8x8 DCT stats."""
    h, w = planes.shape[:2]
    h16 = (h // 16) * 16
    w16 = (w // 16) * 16
    cropped = planes[:h16, :w16]

    # Zero-copy 1-pixel boundary steps locally detrended by each 8-pixel block's truncated mean step
    raw_jv = np.empty_like(cropped)
    raw_jv[1:, :, :] = np.minimum(np.abs(cropped[1:, :, :] - cropped[:-1, :, :]), 8.0)
    raw_jv[0, :, :] = raw_jv[16, :, :] if h16 > 16 else 0.0

    raw_jh = np.empty_like(cropped)
    raw_jh[:, 1:, :] = np.minimum(np.abs(cropped[:, 1:, :] - cropped[:, :-1, :]), 8.0)
    raw_jh[:, 0, :] = raw_jh[:, 16, :] if w16 > 16 else 0.0

    blk_jv = raw_jv.reshape(h16 // 8, 8, w16, 3)
    blk_jh = raw_jh.reshape(h16, w16 // 8, 8, 3)
    jump_v = np.maximum(0.0, blk_jv - 0.55 * np.mean(blk_jv, axis=1, keepdims=True)).reshape(h16, w16, 3)
    jump_h = np.maximum(0.0, blk_jh - 0.55 * np.mean(blk_jh, axis=2, keepdims=True)).reshape(h16, w16, 3)

    prof8_v = jump_v.reshape(h16 // 8, 8, w16, 3).mean(axis=(0, 2), dtype=np.float64)
    prof8_h = jump_h.reshape(h16, w16 // 8, 8, 3).mean(axis=(0, 1), dtype=np.float64)
    prof16_v = jump_v.reshape(h16 // 16, 16, w16, 3).mean(axis=(0, 2), dtype=np.float64)
    prof16_h = jump_h.reshape(h16, w16 // 16, 16, 3).mean(axis=(0, 1), dtype=np.float64)

    webp_4_jump = 0.5 * (
        _f(prof8_v[4, 0]) / (_f(np.median(prof8_v[[1, 2, 3, 5, 6, 7], 0])) + 1e-4)
        + _f(prof8_h[4, 0]) / (_f(np.median(prof8_h[[1, 2, 3, 5, 6, 7], 0])) + 1e-4)
    )

    b1_list: list[float] = []
    for c in range(3):
        int8_v = _f(np.median(prof8_v[1:, c])) + 1e-4
        int8_h = _f(np.median(prof8_h[1:, c])) + 1e-4
        r8_v = _f(prof8_v[0, c]) / int8_v
        r8_h = _f(prof8_h[0, c]) / int8_h
        b1_list.extend(
            (
                math.log1p(max(0.0, r8_v - 1.0)),
                math.log1p(max(0.0, r8_h - 1.0)),
                (_f(prof8_v[0, c]) - int8_v) / 6.0,
                r8_h - r8_v,
            )
        )
    for c in range(3):
        int16_v = _f(np.median(prof16_v[[2, 3, 5, 6, 10, 11, 13, 14], c])) + 1e-4
        int16_h = _f(np.median(prof16_h[[2, 3, 5, 6, 10, 11, 13, 14], c])) + 1e-4
        r16_0_v = _f(prof16_v[0, c]) / int16_v
        r16_8_v = _f(prof16_v[8, c]) / int16_v
        r16_0_h = _f(prof16_h[0, c]) / int16_h
        r16_8_h = _f(prof16_h[8, c]) / int16_h
        b1_list.extend(
            (
                (r16_0_v / (r16_8_v + 1e-4)) - 1.0,
                (r16_0_h / (r16_8_h + 1e-4)) - 1.0,
                math.log1p(max(0.0, 0.5 * (r16_0_v + r16_0_h) - 1.0)),
                (r16_0_h / (r16_8_h + 1e-4)) - (r16_0_v / (r16_8_v + 1e-4)),
            )
        )

    b2_list: list[float] = []
    shift_scores: list[float] = []
    for c in range(3):
        for prof in (prof8_v[:, c], prof8_h[:, c]):
            non_zero = prof[1:]
            s_sorted = np.sort(non_zero)
            med_nz = _f(np.median(non_zero)) + 1e-4
            max_nz = _f(s_sorted[-1])
            sec_nz = _f(s_sorted[-2])
            prominence = (max_nz - med_nz) / med_nz
            peak_vs_origin = max_nz / (_f(prof[0]) + 1e-4)
            gap_top2 = (max_nz - sec_nz) / med_nz
            ent = _phase_entropy(np.maximum(prof - _f(prof.min()), 0.0))
            b2_list.extend((prominence, peak_vs_origin, gap_top2, 1.0 - ent))
            if c == 0:
                shift_scores.append(prominence)

    # B3 (32-d): Strictly Interior {1..6}x{1..6} VAE Sub-Lattice Modulation reusing precomputed lap_raw
    res_lap = np.clip(lap_raw[:h16, :w16], -3.0, 3.0)
    res_sq = res_lap * res_lap
    grid8_sq = res_sq.reshape(h16 // 8, 8, w16 // 8, 8, 3).mean(axis=(0, 2), dtype=np.float64)

    blk_y = cropped[:, :, 0].reshape(h16 // 8, 8, w16 // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
    block_luma_var = blk_y.var(axis=1)
    med_var = _f(np.partition(block_luma_var, len(block_luma_var) // 2)[len(block_luma_var) // 2])
    flat_blk_mask = block_luma_var <= med_var
    flat_blocks_sq = (
        res_sq.reshape(h16 // 8, 8, w16 // 8, 8, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 8, 8, 3)[flat_blk_mask]
    )
    grid8_flat_sq = flat_blocks_sq.mean(axis=0, dtype=np.float64) if len(flat_blocks_sq) > 0 else grid8_sq

    interior_coords = slice(1, 7)
    yy6, xx6 = np.meshgrid(np.arange(1, 7), np.arange(1, 7), indexing="ij")
    par_y = np.where(yy6 % 2 == 0, 1.0, -1.0)
    par_x = np.where(xx6 % 2 == 0, 1.0, -1.0)
    par_xy = par_y * par_x
    mod4_y = np.cos(0.5 * math.pi * yy6)
    mod4_x = np.cos(0.5 * math.pi * xx6)

    b3_list: list[float] = []
    for c in range(3):
        int_maps = (
            grid8_sq[interior_coords, interior_coords, c],
            grid8_flat_sq[interior_coords, interior_coords, c],
        )
        for g_map in int_maps:
            mean_g = _f(g_map.mean()) + 1e-6
            norm_g = (g_map - mean_g) / mean_g
            b3_list.extend(
                (
                    _f(np.mean(norm_g * par_y)),
                    _f(np.mean(norm_g * par_x)),
                    _f(np.mean(norm_g * par_xy)),
                    _f(np.mean(norm_g * mod4_y)) + _f(np.mean(norm_g * mod4_x)),
                    _f(np.std(norm_g)),
                )
            )
    int_y = grid8_flat_sq[interior_coords, interior_coords, 0].ravel()
    int_rg = grid8_flat_sq[interior_coords, interior_coords, 1].ravel()
    int_by = grid8_flat_sq[interior_coords, interior_coords, 2].ravel()
    int_y_c = int_y - int_y.mean()
    int_rg_c = int_rg - int_rg.mean()
    int_by_c = int_by - int_by.mean()
    b3_list.append(_f(np.dot(int_y_c, int_rg_c) / (np.linalg.norm(int_y_c) * np.linalg.norm(int_rg_c) + EPS)))
    b3_list.append(_f(np.dot(int_y_c, int_by_c) / (np.linalg.norm(int_y_c) * np.linalg.norm(int_by_c) + EPS)))

    # B4 (48-d): Vectorized 8x8 DCT AC Quantization, Zero-Occupancy & Double-JPEG Comb Profile
    basis, ac_r, ac_c = _dct_basis_and_ac()
    n_by, n_bx = h16 // 8, w16 // 8
    total_blocks = n_by * n_bx
    max_sample_blocks = 2048
    all_blocks = cropped.reshape(n_by, 8, n_bx, 8, 3).transpose(0, 2, 1, 3, 4).reshape(total_blocks, 8, 8, 3)
    if total_blocks > max_sample_blocks:
        step_idx = np.linspace(0, total_blocks - 1, max_sample_blocks, dtype=np.intp)
        sub_blocks = all_blocks[step_idx]
    else:
        sub_blocks = all_blocks

    rows_b = basis[ac_r]
    cols_b = basis[ac_c]
    ac_vals = np.einsum("ki,bijc,kj->bkc", rows_b, sub_blocks, cols_b).astype(np.float64)
    abs_ac = np.abs(ac_vals)

    zero_y = np.asarray(np.mean(abs_ac[:, :, 0] < 0.5, axis=0), dtype=np.float64)
    zero_rg = np.array(
        [
            _f(np.mean(abs_ac[:, :4, 1] < 0.5)),
            _f(np.mean(abs_ac[:, 4:8, 1] < 0.5)),
            _f(np.mean(abs_ac[:, 8:, 1] < 0.5)),
        ],
        dtype=np.float64,
    )
    zero_by = np.array(
        [
            _f(np.mean(abs_ac[:, :4, 2] < 0.5)),
            _f(np.mean(abs_ac[:, 4:8, 2] < 0.5)),
            _f(np.mean(abs_ac[:, 8:, 2] < 0.5)),
        ],
        dtype=np.float64,
    )

    comb_steps = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
    comb_feats: list[float] = []
    max_comb_y = 0.0
    max_comb_c = 0.0
    for c in range(3):
        c_flat = ac_vals[:, :8, c].ravel()
        active = c_flat[np.abs(c_flat) >= 0.75]
        if active.size >= 32:
            for q_step in comb_steps:
                res_val = _f(np.mean(np.cos((2.0 * math.pi / q_step) * active)))
                comb_feats.append(res_val)
                if c == 0 and q_step >= 2.0:
                    max_comb_y = max(max_comb_y, res_val)
                elif c > 0 and q_step >= 2.0:
                    max_comb_c = max(max_comb_c, res_val)
        else:
            comb_feats.extend([0.0] * len(comb_steps))

    benford_target = np.log10(1.0 + 1.0 / np.arange(1, 10, dtype=np.float64))
    var_benford: list[float] = []
    for c in range(3):
        v_low = math.log1p(_f(np.mean(ac_vals[:, :4, c] ** 2)))
        v_mid = math.log1p(_f(np.mean(ac_vals[:, 4:8, c] ** 2)))
        v_hi = math.log1p(_f(np.mean(ac_vals[:, 8:, c] ** 2)))
        var_benford.extend((v_low / 5.0, v_mid / 5.0, v_hi / 5.0))

        nz = abs_ac[:, :, c].ravel()
        nz = nz[nz >= 1.0]
        if nz.size >= 64:
            lead = (nz / (10.0 ** np.floor(np.log10(nz)))).astype(np.int32)
            lead = lead[(lead >= 1) & (lead <= 9)]
            if lead.size >= 64:
                obs = np.bincount(lead, minlength=10)[1:10] / float(lead.size)
                ben_mad = _f(np.mean(np.abs(obs - benford_target)))
            else:
                ben_mad = 0.0
        else:
            ben_mad = 0.0
        var_benford.append(ben_mad * 10.0)

    vec_b = np.concatenate(
        [
            np.asarray(b1_list, dtype=np.float64),
            np.asarray(b2_list, dtype=np.float64),
            np.asarray(b3_list, dtype=np.float64),
            zero_y,
            zero_rg,
            zero_by,
            np.asarray(comb_feats, dtype=np.float64),
            np.asarray(var_benford, dtype=np.float64),
        ]
    )
    if vec_b.shape != (128,):
        raise ValueError(f"Expected Block B shape (128,), got {vec_b.shape}")

    luma_seam = 0.5 * (b1_list[0] + b1_list[1])
    chroma_seam = 0.25 * (b1_list[4] + b1_list[5] + b1_list[8] + b1_list[9])
    c420_score = _f(np.clip(0.5 * (b1_list[16] + b1_list[17] + b1_list[20] + b1_list[21]), -1.0, 3.0))
    c422_score = _f(np.clip(0.5 * (b1_list[19] + b1_list[23]), -1.0, 3.0))
    regime_b = {
        "luma_seam": luma_seam,
        "chroma_seam": chroma_seam,
        "q_luma_est": _f(np.clip(0.6 * _f(np.mean(zero_y[4:])) + 0.4 * max(0.0, max_comb_y), 0.0, 1.0)),
        "q_chroma_est": float(
            np.clip(0.6 * float(0.5 * (zero_rg[1] + zero_by[1])) + 0.4 * max(0.0, max_comb_c), 0.0, 1.0)
        ),
        "c420_score": c420_score,
        "c422_score": c422_score,
        "c444_score": float(np.clip(1.0 - 0.5 * (zero_rg[2] + zero_by[2]), 0.0, 1.0)),
        "double_jpeg": float(np.clip(max(0.0, max_comb_y), 0.0, 1.0)),
        "grid_shift": float(np.clip(max(shift_scores), 0.0, 1.0)),
        "webp_4x4": float(np.clip(webp_4_jump - 1.0, -0.5, 2.0)),
    }
    return vec_b, regime_b


def _extract_moisan_channel_profile(
    p_hat: NDArray[np.complex128],
    mult: NDArray[np.float64],
    r_int: NDArray[np.int32],
    r_weight_sum: NDArray[np.float64],
    rings_12: tuple[NDArray[np.bool_], ...],
    cells_16: tuple[NDArray[np.bool_], ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """From batch of rfft2 periodic spectra (N, 256, 129), extract radial_16, aniso_16, and comb_12."""
    power = np.mean(np.abs(p_hat) ** 2, axis=0)
    total_ac = float(np.sum(power * mult) - power[0, 0]) + EPS
    log_spec = np.log1p(np.sqrt(power / (total_ac / (SPECTRAL_TILE * SPECTRAL_TILE))))

    radial_sum = np.bincount(r_int.ravel(), weights=(log_spec * mult).ravel(), minlength=len(r_weight_sum))
    radial_mean = radial_sum / r_weight_sum
    residual = log_spec - radial_mean[r_int]
    residual[0, 0] = 0.0

    ring_vals = np.array([float(np.sum(log_spec[m] * mult[m]) / np.sum(mult[m])) for m in rings_12], dtype=np.float64)
    slopes = np.array(
        [
            ring_vals[11] - ring_vals[7],
            ring_vals[7] - ring_vals[3],
            ring_vals[3] - ring_vals[0],
            ring_vals[11] - 2.0 * ring_vals[6] + ring_vals[1],
        ],
        dtype=np.float64,
    )
    radial_16 = np.concatenate([ring_vals, slopes])

    aniso_16 = np.array(
        [float(np.sum(residual[m] * mult[m]) / (np.sum(mult[m]) + EPS)) for m in cells_16],
        dtype=np.float64,
    )

    ax_h = residual[0, :]
    ax_v = residual[:129, 0]
    cardinal = 0.5 * (ax_h + ax_v)

    def _local_prominence(k: int) -> float:
        nbrs = [cardinal[j] for j in (k - 3, k - 2, k + 2, k + 3) if 4 <= j <= 126]
        base = float(np.median(nbrs)) if nbrs else 0.0
        return float(cardinal[k] - base)

    p_k16 = _local_prominence(16)
    p_k32 = _local_prominence(32)
    p_k64 = _local_prominence(64)
    p_k96 = _local_prominence(96)
    p_2d_32 = float(0.5 * (residual[32, 32] + residual[SPECTRAL_TILE - 32, 32]))
    p_2d_16 = float(0.5 * (residual[16, 16] + residual[SPECTRAL_TILE - 16, 16]))

    prom_all = np.array([_local_prominence(k) for k in range(14, 61)], dtype=np.float64)
    ks = np.arange(14, 61, dtype=np.int32)
    off_grid_mask = (ks != 16) & (ks != 32) & (ks != 48)
    low_band = off_grid_mask & (ks >= 18) & (ks <= 30)
    hi_band = off_grid_mask & (ks >= 34) & (ks <= 58)

    best_low_idx = int(np.argmax(prom_all[low_band]))
    best_low_val = float(prom_all[low_band][best_low_idx])
    best_low_k = int(ks[low_band][best_low_idx])

    best_hi_idx = int(np.argmax(prom_all[hi_band]))
    best_hi_val = float(prom_all[hi_band][best_hi_idx])
    best_hi_k = int(ks[hi_band][best_hi_idx])

    if best_hi_val >= best_low_val:
        k_star = best_hi_k
        peak_off = best_hi_val
    else:
        k_star = best_low_k
        peak_off = best_low_val

    harm2_val = _local_prominence(2 * k_star) if 2 * k_star <= 124 else 0.0
    nyq_spike = _f(np.mean(cardinal[124:129])) - _f(np.mean(cardinal[112:120]))

    comb_12 = np.array(
        [
            p_k16,
            p_k32,
            p_k64,
            p_k96,
            p_2d_16,
            p_2d_32,
            best_low_val,
            best_hi_val,
            peak_off,
            float(k_star) / 32.0,
            harm2_val,
            nyq_spike,
        ],
        dtype=np.float64,
    )
    return radial_16, aniso_16, comb_12


def _block_c_and_coherence(
    rgb: NDArray[np.uint8],
    planes: NDArray[np.float32],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128], list[tuple[int, int]], dict[str, float]]:
    """Compute Block C (144-d) Moisan spectrum + D1 (36-d) cross-channel coherence in one FFT pass."""
    h, w = planes.shape[:2]
    denom, mult, r_int, r_weight_sum, rings_12, cells_16 = _moisan_rfft_geometry(SPECTRAL_TILE)
    origins = _spectral_256_origins(h, w, max_tiles=9)

    c1_list: list[NDArray[np.float64]] = []
    c2_list: list[NDArray[np.float64]] = []
    c3_list: list[NDArray[np.float64]] = []
    p_hats_by_ch: list[NDArray[np.complex128]] = []

    for c in range(3):
        ch_tiles = np.stack(
            [planes[y : y + SPECTRAL_TILE, x : x + SPECTRAL_TILE, c] for y, x in origins],
            axis=0,
        )
        p_hat = _moisan_periodic_rfft2(ch_tiles, denom)
        p_hats_by_ch.append(p_hat)
        rad16, ani16, cmb12 = _extract_moisan_channel_profile(p_hat, mult, r_int, r_weight_sum, rings_12, cells_16)
        c1_list.append(rad16)
        c2_list.append(ani16)
        c3_list.append(cmb12)

    if h == SPECTRAL_TILE and w == SPECTRAL_TILE:
        c4_shift = np.zeros(12, dtype=np.float64)
    else:
        global_rgb = np.asarray(
            cv2.resize(rgb, (SPECTRAL_TILE, SPECTRAL_TILE), interpolation=cv2.INTER_AREA),
            dtype=np.uint8,
        )
        _, global_planes = _opponent_planes_255(global_rgb)
        c4_parts: list[float] = []
        for c in range(3):
            gp_hat = _moisan_periodic_rfft2(global_planes[:, :, c][None, :, :], denom)
            grad16, _, _ = _extract_moisan_channel_profile(gp_hat, mult, r_int, r_weight_sum, rings_12, cells_16)
            native_rad = c1_list[c]
            c4_parts.extend(
                (
                    float(np.mean(grad16[:4] - native_rad[:4])),
                    float(np.mean(grad16[4:8] - native_rad[4:8])),
                    float(np.mean(grad16[8:12] - native_rad[8:12])),
                    float(grad16[12] - native_rad[12]),
                )
            )
        c4_shift = np.asarray(c4_parts, dtype=np.float64)

    vec_c = np.concatenate([*c1_list, *c2_list, *c3_list, c4_shift])
    if vec_c.shape != (144,):
        raise ValueError(f"Expected Block C shape (144,), got {vec_c.shape}")

    coherence_feats: list[float] = []
    merged_bands = tuple(rings_12[2 * i] | rings_12[2 * i + 1] for i in range(6))
    for c1, c2 in ((0, 1), (0, 2), (1, 2)):
        pa = p_hats_by_ch[c1]
        pb = p_hats_by_ch[c2]
        s_ab = np.mean(pa * np.conj(pb), axis=0)
        s_aa = np.mean(np.abs(pa) ** 2, axis=0)
        s_bb = np.mean(np.abs(pb) ** 2, axis=0)
        denom_coh = s_aa * s_bb + EPS
        gamma_sq = (np.abs(s_ab) ** 2) / denom_coh
        cospec = np.real(s_ab) / np.sqrt(denom_coh)
        for m in merged_bands:
            w_m = mult[m]
            w_sum = float(np.sum(w_m)) + EPS
            coherence_feats.append(float(np.sum(gamma_sq[m] * w_m) / w_sum))
            coherence_feats.append(float(np.sum(cospec[m] * w_m) / w_sum))

    d1_vec = np.asarray(coherence_feats, dtype=np.float64)
    if d1_vec.shape != (36,):
        raise ValueError(f"Expected D1 shape (36,), got {d1_vec.shape}")

    luma_comb = c3_list[0]
    regime_c = {
        "resized_comb": float(np.clip(luma_comb[8], 0.0, 2.0) / 2.0),
        "est_scale": float(luma_comb[9]),
    }
    return vec_c, d1_vec, p_hats_by_ch[0], origins, regime_c


def _block_d_sensor_physics(
    rgb: NDArray[np.uint8],
    vals_f32: NDArray[np.float32],
    planes: NDArray[np.float32],
    lap_opp_raw: NDArray[np.float32],
    d1_coherence: NDArray[np.float64],
    luma_p_hat: NDArray[np.complex128],
    spec_origins: list[tuple[int, int]],
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Compute Block D (96-d): D1 (36) + D2 CFA demosaicing (24) + D3 shot-noise slope (20) + D4 CA/gamut (16)."""
    h, w = rgb.shape[:2]
    r = vals_f32[:, :, 0]
    g = vals_f32[:, :, 1]
    b = vals_f32[:, :, 2]

    gm = g - 0.5 * (r + b)
    rb = r - b
    sub_origins = spec_origins[:6]
    denom, _, _, _, _, _ = _moisan_rfft_geometry(SPECTRAL_TILE)

    yy = np.arange(SPECTRAL_TILE, dtype=np.float32)[:, None]
    xx = np.arange(SPECTRAL_TILE, dtype=np.float32)[None, :]
    chk_2d = np.where((yy + xx) % 2 == 0, 1.0, -1.0).astype(np.float32)
    qb_2d = np.where(((yy // 2) + (xx // 2)) % 2 == 0, 1.0, -1.0).astype(np.float32)

    d2_list: list[float] = []
    for idx_ch, ch_map in enumerate((gm, rb, planes[:, :, 0])):
        tiles = np.stack(
            [ch_map[y : y + SPECTRAL_TILE, x : x + SPECTRAL_TILE] for y, x in sub_origins],
            axis=0,
        ).astype(np.float32)
        p_hat = luma_p_hat[: len(sub_origins)] if idx_ch == 2 else _moisan_periodic_rfft2(tiles, denom)
        mag = np.log1p(np.mean(np.abs(p_hat), axis=0))
        bg = _f(np.median(mag[16:240:2, 16:120:2])) + 1e-6

        pk_pipi = _f(np.max(mag[126:131, 125:129])) - bg
        pk_pi0 = _f(np.max(mag[126:131, 0:4])) - bg
        pk_0pi = _f(np.max(mag[[0, 1, 2, 254, 255], 125:129])) - bg
        pk_qb_diag = 0.5 * (_f(np.max(mag[62:67, 62:67])) + _f(np.max(mag[190:195, 62:67])) - 2.0 * bg)
        pk_qb_v = _f(np.max(mag[62:67, 0:4])) - bg
        pk_qb_h = _f(np.max(mag[[0, 1, 2, 254, 255], 62:67])) - bg

        chk_energy = _f(np.mean(np.abs(np.mean(tiles * chk_2d[None, :, :], axis=(1, 2)))))
        qb_energy = _f(np.mean(np.abs(np.mean(tiles * qb_2d[None, :, :], axis=(1, 2)))))
        d2_list.extend(
            (pk_pipi, pk_pi0, pk_0pi, pk_qb_diag, pk_qb_v, pk_qb_h, math.log1p(chk_energy), math.log1p(qb_energy))
        )

    # D3 (20-d): Heteroscedastic Poisson-Gaussian Camera Shot-Noise Curve sigma^2(I) = a * I + b
    # Reconstruct exact per-channel RGB laplacian from opponent laplacian without another filter2D pass!
    # Y = 0.299 R + 0.587 G + 0.114 B, Crg = R - G, Cby = B - 0.5 R - 0.5 G
    # Inverting the 3x3 linear transform gives exact R, G, B residuals:
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    ly = lap_opp_raw[:h8, :w8, 0]
    lrg = lap_opp_raw[:h8, :w8, 1]
    lby = lap_opp_raw[:h8, :w8, 2]
    lg = ly - 0.356 * lrg - 0.114 * lby
    lr = lg + lrg
    lb = lby + 0.5 * (lr + lg)

    y_crop = planes[:h8, :w8, 0]
    lap_y = np.clip(ly, -4.0, 4.0).astype(np.float32)
    lap_r = np.clip(lr, -4.0, 4.0).astype(np.float32)
    lap_g = np.clip(lg, -4.0, 4.0).astype(np.float32)
    lap_b = np.clip(lb, -4.0, 4.0).astype(np.float32)

    n_blk = (h8 // 8) * (w8 // 8)
    y_blocks = y_crop.reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3).reshape(n_blk, 64)
    blk_mean_y = y_blocks.mean(axis=1, dtype=np.float64)
    blk_std_y = y_blocks.std(axis=1, dtype=np.float64)

    def _blk_mse(arr2d: NDArray[np.float32]) -> NDArray[np.float64]:
        sq = arr2d * arr2d
        blocks_sq = sq.reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3).reshape(n_blk, 64)
        return blocks_sq.mean(axis=1, dtype=np.float64)

    blk_var_ly = _blk_mse(lap_y)
    blk_var_lr = _blk_mse(lap_r)
    blk_var_lg = _blk_mse(lap_g)
    blk_var_lb = _blk_mse(lap_b)

    q55_idx = int(0.55 * (n_blk - 1))
    std_thresh = _f(np.partition(blk_std_y, q55_idx)[q55_idx]) + 1e-3
    valid_blk = (blk_mean_y > 6.0) & (blk_mean_y < 248.0) & (blk_std_y <= std_thresh)
    if int(np.sum(valid_blk)) < 20:
        q75_idx = int(0.75 * (n_blk - 1))
        valid_blk = blk_std_y <= _f(np.partition(blk_std_y, q75_idx)[q75_idx]) + 1e-3

    iy = blk_mean_y[valid_blk] / 255.0
    d3_list: list[float] = []
    for noise_arr in (blk_var_ly[valid_blk], blk_var_lr[valid_blk], blk_var_lg[valid_blk], blk_var_lb[valid_blk]):
        if iy.size >= 10 and _f(np.std(iy)) > 1e-3:
            iy_c = iy - _f(np.mean(iy))
            nv = np.log1p(noise_arr)
            nv_c = nv - _f(np.mean(nv))
            denom_i = _f(np.sum(iy_c * iy_c)) + EPS
            slope = _f(np.sum(iy_c * nv_c)) / denom_i
            intercept = _f(np.mean(nv)) - slope * _f(np.mean(iy))
            corr = _f(np.sum(iy_c * nv_c)) / (math.sqrt(denom_i * _f(np.sum(nv_c * nv_c))) + EPS)
            q30_v = _f(np.quantile(iy, 0.30))
            q70_v = _f(np.quantile(iy, 0.70))
            q_lo = _f(np.median(nv[iy <= q30_v]))
            q_hi = _f(np.median(nv[iy >= q70_v]))
            d3_list.extend((slope, intercept, corr, q_hi - q_lo))
        else:
            d3_list.extend((0.0, _f(np.mean(np.log1p(noise_arr))) if noise_arr.size else 0.0, 0.0, 0.0))

    v_r = float(np.median(blk_var_lr[valid_blk])) + 1e-5 if np.any(valid_blk) else 1e-5
    v_g = float(np.median(blk_var_lg[valid_blk])) + 1e-5 if np.any(valid_blk) else 1e-5
    v_b = float(np.median(blk_var_lb[valid_blk])) + 1e-5 if np.any(valid_blk) else 1e-5
    d3_list.extend(
        (
            math.log(v_r / v_g),
            math.log(v_b / v_g),
            math.log(v_r / v_b),
            math.log((v_r + v_b) / (2.0 * v_g)),
        )
    )

    # D4 (16-d): Radial Lateral Chromatic Aberration (at stride 2 for >512px) + Gamut / Sparsity
    step_s = 2 if min(h, w) >= 512 else 1
    r_s = r[::step_s, ::step_s]
    g_s = g[::step_s, ::step_s]
    b_s = b[::step_s, ::step_s]
    hs, ws = g_s.shape

    gx_r = cv2.Sobel(r_s, cv2.CV_32F, 1, 0, ksize=3)
    gy_r = cv2.Sobel(r_s, cv2.CV_32F, 0, 1, ksize=3)
    gx_g = cv2.Sobel(g_s, cv2.CV_32F, 1, 0, ksize=3)
    gy_g = cv2.Sobel(g_s, cv2.CV_32F, 0, 1, ksize=3)
    gx_b = cv2.Sobel(b_s, cv2.CV_32F, 1, 0, ksize=3)
    gy_b = cv2.Sobel(b_s, cv2.CV_32F, 0, 1, ksize=3)

    cy, cx = 0.5 * (hs - 1), 0.5 * (ws - 1)
    norm_r = float(math.hypot(cy, cx)) + 1e-6
    ry = (np.arange(hs, dtype=np.float32)[:, None] - cy) / norm_r
    rx = (np.arange(ws, dtype=np.float32)[None, :] - cx) / norm_r
    rad_dist = np.hypot(ry, rx)

    grad_mag_g = np.hypot(gx_g, gy_g)
    edge_mask = grad_mag_g > 32.0
    outer_edge = edge_mask & (rad_dist > 0.55)
    inner_edge = edge_mask & (rad_dist <= 0.40)

    rad_g_proj = np.sign(gx_g * rx + gy_g * ry)
    ca_rg_map = ((gx_r - gx_g) * rx + (gy_r - gy_g) * ry) * rad_g_proj / (grad_mag_g + 1.0)
    ca_bg_map = ((gx_b - gx_g) * rx + (gy_b - gy_g) * ry) * rad_g_proj / (grad_mag_g + 1.0)

    ca_rg_out = float(np.mean(ca_rg_map[outer_edge], dtype=np.float64)) if np.any(outer_edge) else 0.0
    ca_rg_in = float(np.mean(ca_rg_map[inner_edge], dtype=np.float64)) if np.any(inner_edge) else 0.0
    ca_bg_out = float(np.mean(ca_bg_map[outer_edge], dtype=np.float64)) if np.any(outer_edge) else 0.0
    ca_bg_in = float(np.mean(ca_bg_map[inner_edge], dtype=np.float64)) if np.any(inner_edge) else 0.0

    max_rgb = np.maximum(np.maximum(r_s, g_s), b_s)
    min_rgb = np.minimum(np.minimum(r_s, g_s), b_s)
    chroma = max_rgb - min_rgb
    sat = (chroma / (max_rgb + 1.0)).ravel()
    sat_mean = float(np.mean(sat, dtype=np.float64))
    sat_std = float(np.std(sat, dtype=np.float64))
    p95_idx = int(0.95 * (sat.size - 1))
    sat_p95 = float(np.partition(sat, p95_idx)[p95_idx])
    achromatic_frac = float(np.mean(chroma < 4.0))

    clip_dark = float(np.mean(max_rgb <= 2.0))
    clip_bright = float(np.mean(min_rgb >= 253.0))
    high_edge_frac = float(np.mean(edge_mask))

    luma_s = planes[::step_s, ::step_s, 0]
    extreme_luma_frac = float(np.mean((luma_s < 30.0) | (luma_s > 225.0)))
    luma_hist = np.bincount(np.clip(luma_s.astype(np.int32).ravel(), 0, 255), minlength=256).astype(np.float64)
    luma_p = np.maximum(luma_hist / float(luma_s.size), 1e-12)
    luma_ent = float(-np.sum(luma_p * np.log(luma_p)) / math.log(256.0))

    rgb_s = rgb[::step_s, ::step_s]
    q_rgb = (
        ((rgb_s[:, :, 0].astype(np.int32) >> 4) << 8)
        | ((rgb_s[:, :, 1].astype(np.int32) >> 4) << 4)
        | (rgb_s[:, :, 2].astype(np.int32) >> 4)
    )
    q_counts = np.bincount(q_rgb.ravel(), minlength=4096)
    top8_color_share = float(np.sort(q_counts)[-8:].sum()) / float(q_rgb.size)
    nonzero_palette_frac = float(np.count_nonzero(q_counts)) / 4096.0

    d4_list = [
        ca_rg_out,
        ca_rg_out - ca_rg_in,
        ca_bg_out,
        ca_bg_out - ca_bg_in,
        ca_rg_out - ca_bg_out,
        sat_mean,
        sat_std,
        sat_p95,
        achromatic_frac,
        clip_dark,
        clip_bright,
        high_edge_frac,
        extreme_luma_frac,
        luma_ent,
        top8_color_share,
        nonzero_palette_frac,
    ]

    vec_d = np.concatenate(
        [
            d1_coherence,
            np.asarray(d2_list, dtype=np.float64),
            np.asarray(d3_list, dtype=np.float64),
            np.asarray(d4_list, dtype=np.float64),
        ]
    )
    if vec_d.shape != (96,):
        raise ValueError(f"Expected Block D shape (96,), got {vec_d.shape}")

    regime_d = {
        "high_edge_frac": high_edge_frac,
        "bimodal_bw": float(extreme_luma_frac * achromatic_frac),
        "palette_sparsity": top8_color_share,
        "sat_mean": sat_mean,
        "sat_p95": sat_p95,
        "clip_dark": clip_dark,
        "clip_bright": clip_bright,
        "luma_mean": float(np.mean(luma_s, dtype=np.float64)) / 255.0,
        "luma_std": float(np.std(luma_s, dtype=np.float64)) / 128.0,
    }
    return vec_d, regime_d


def extract_v4_features_from_rgb(
    rgb: NDArray[np.uint8],
    *,
    orig_size: tuple[int, int] | None = None,
) -> NDArray[np.float64] | None:
    """Extract the 512-d v4 feature vector from an HxWx3 uint8 RGB array, or None if < 256x256."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None
    h_raw, w_raw = int(rgb.shape[0]), int(rgb.shape[1])
    if min(h_raw, w_raw) < MIN_SIDE_V4:
        return None

    w_orig, h_orig = orig_size if orig_size is not None else (w_raw, h_raw)
    work_rgb = _lattice_crop_cap(rgb)
    vals_f32, planes = _opponent_planes_255(work_rgb)
    lap_opp_raw = np.asarray(
        cv2.filter2D(planes, cv2.CV_32F, _srm_kernels()[4], borderType=cv2.BORDER_REFLECT_101),
        dtype=np.float32,
    )

    vec_a, flat_fraction = _block_a_srm_and_tiles(planes, lap_opp_raw)
    vec_b, reg_b = _block_b_sublattice_and_dct(planes, lap_opp_raw)
    vec_c, d1_coh, luma_p_hat, spec_origins, reg_c = _block_c_and_coherence(work_rgb, planes)
    vec_d, reg_d = _block_d_sensor_physics(work_rgb, vals_f32, planes, lap_opp_raw, d1_coh, luma_p_hat, spec_origins)

    min_side = float(min(h_orig, w_orig))
    max_side = float(max(h_orig, w_orig))
    lossless_score = float(
        np.clip(
            1.0 - 1.8 * reg_b["luma_seam"] - 0.8 * reg_b["q_luma_est"] - 0.6 * reg_b["double_jpeg"],
            0.0,
            1.0,
        )
    )

    vec_e = np.array(
        [
            lossless_score,
            reg_b["luma_seam"],
            reg_b["chroma_seam"],
            reg_b["q_luma_est"],
            reg_b["q_chroma_est"],
            reg_b["c420_score"],
            reg_b["c422_score"],
            reg_b["c444_score"],
            reg_b["double_jpeg"],
            reg_b["grid_shift"],
            reg_c["resized_comb"],
            reg_c["est_scale"],
            reg_b["webp_4x4"],
            math.log2(min_side / 256.0),
            math.log2(max_side / 256.0),
            math.log2((float(h_orig) * float(w_orig)) / 65536.0),
            math.log2(float(w_orig) / float(h_orig)),
            1.0 if abs(w_orig - h_orig) <= 2 else 0.0,
            1.0 if (h_orig % 64 == 0 and w_orig % 64 == 0) else 0.0,
            1.0 if (h_orig % 16 == 0 and w_orig % 16 == 0) else 0.0,
            1.0 if (h_orig % 8 == 0 and w_orig % 8 == 0) else 0.0,
            1.0 if min_side < 512.0 else 0.0,
            flat_fraction,
            reg_d["high_edge_frac"],
            reg_d["bimodal_bw"],
            reg_d["palette_sparsity"],
            reg_d["sat_mean"],
            reg_d["sat_p95"],
            reg_d["clip_dark"],
            reg_d["clip_bright"],
            reg_d["luma_mean"],
            reg_d["luma_std"],
        ],
        dtype=np.float64,
    )
    if vec_e.shape != (REGIME_DIM_V4,):
        raise ValueError(f"Expected Block E shape ({REGIME_DIM_V4},), got {vec_e.shape}")

    full = np.concatenate([vec_a, vec_b, vec_c, vec_d, vec_e])
    if full.shape != (FEATURE_DIM_V4,) or not np.all(np.isfinite(full)):
        return None
    return full


def extract_v4_features(image: Image.Image) -> NDArray[np.float64] | None:
    """Extract the 512-d v4 feature vector from a PIL Image, or None if < 256x256."""
    w_orig, h_orig = image.size
    if min(w_orig, h_orig) < MIN_SIDE_V4:
        return None
    rgb_image = image if image.mode == "RGB" else image.convert("RGB")
    rgb = np.asarray(rgb_image, dtype=np.uint8)
    return extract_v4_features_from_rgb(rgb, orig_size=(w_orig, h_orig))
