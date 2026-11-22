"""Pure motion-compensated helpers shared by video pipelines."""

from __future__ import annotations

import warnings

# OpenCV exposes incomplete types for optical-flow and remap operations.
# Public signatures remain annotated while this third-party boundary is relaxed.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingTypeStubs=false, reportCallIssue=false, reportArgumentType=false, reportUnnecessaryComparison=false
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence

    from numpy.typing import NDArray


def _backward_map(
    current_gray: NDArray[Any],
    previous_gray: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Build a remap from a previous frame into current coordinates."""
    flow = cv2.calcOpticalFlowFarneback(
        current_gray,
        previous_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    height, width = current_gray.shape
    flow[..., 0] += np.arange(width, dtype=np.float32)[None, :]
    flow[..., 1] += np.arange(height, dtype=np.float32)[:, None]
    return flow[..., 0], flow[..., 1]


def _backward_warp(
    image: NDArray[Any],
    maps: tuple[NDArray[Any], NDArray[Any]],
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> NDArray[Any]:
    """Apply a precomputed backward optical-flow map."""
    return cv2.remap(
        image,
        maps[0],
        maps[1],
        interpolation=interpolation,
        borderMode=cv2.BORDER_REFLECT,
    )


def _motion_residual(
    current: NDArray[Any],
    previous: NDArray[Any],
    maps: tuple[NDArray[Any], NDArray[Any]],
) -> float:
    """Return mean absolute residual after warping the previous frame."""
    current_f32 = np.asarray(current, dtype=np.float32)
    previous_f32 = np.asarray(previous, dtype=np.float32)
    warped_previous = _backward_warp(previous_f32, maps)
    return float(np.mean(np.abs(current_f32 - warped_previous)))


def build_temporal_reference(
    reference: Sequence[NDArray[Any]],
) -> tuple[tuple[tuple[NDArray[Any], NDArray[Any]], ...], float]:
    """Precompute source motion maps and its mean residual."""
    if len(reference) < 2:
        raise ValueError("Temporal metric needs at least two frames")
    maps: list[tuple[NDArray[Any], NDArray[Any]]] = []
    reference_residuals: list[float] = []
    for index in range(1, len(reference)):
        current_gray = cv2.cvtColor(reference[index], cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(reference[index - 1], cv2.COLOR_BGR2GRAY)
        frame_maps = _backward_map(current_gray, previous_gray)
        maps.append(frame_maps)
        reference_residuals.append(_motion_residual(reference[index], reference[index - 1], frame_maps))
    return tuple(maps), float(np.mean(reference_residuals))


def temporal_residual_ratio(
    candidate: Sequence[NDArray[Any]],
    maps: Sequence[tuple[NDArray[Any], NDArray[Any]]],
    baseline: float,
) -> float:
    """Measure candidate flicker against a precomputed source residual."""
    if len(candidate) != len(maps) + 1:
        raise ValueError("Temporal metric needs one map per adjacent frame pair")
    candidate_residuals: list[float] = []
    for index, frame_maps in enumerate(maps, start=1):
        candidate_residuals.append(_motion_residual(candidate[index], candidate[index - 1], frame_maps))
    measured = float(np.mean(candidate_residuals))
    return measured / max(baseline, 1e-6)


def _to_gray_u8(frame_bgr: NDArray[Any]) -> NDArray[Any]:
    """Convert an 8-bit or 16-bit BGR frame to uint8 grayscale for optical flow."""
    gray: NDArray[Any] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return (gray >> 8).astype(np.uint8) if frame_bgr.dtype == np.uint16 else gray


def _depth_bounds(dtype: Any, threshold_u8: float) -> tuple[float, float]:
    """Return ``(scaled_threshold, peak_value)`` for an 8-bit or 16-bit frame dtype."""
    if np.dtype(dtype) == np.uint16:
        return threshold_u8 * 257.0, 65535.0
    return threshold_u8, 255.0


def _background_aligned_maps(
    current_source_crop: NDArray[Any],
    current_mask_crop: NDArray[Any],
    neighbor_source_crop: NDArray[Any],
    neighbor_mask_crop: NDArray[Any],
    *,
    fb_tolerance_px: float = 1.75,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Estimate background optical flow inside occluded watermark masks via context affine fit."""
    crop_h, crop_w = current_mask_crop.shape
    guard_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    curr_guard = cv2.dilate((current_mask_crop > 0).astype(np.uint8) * 255, guard_kernel)
    curr_guard_mask = curr_guard > 0
    nb_guard = cv2.dilate((neighbor_mask_crop > 0).astype(np.uint8) * 255, guard_kernel)

    curr_gray_raw = _to_gray_u8(current_source_crop)
    nb_gray_raw = _to_gray_u8(neighbor_source_crop)
    curr_gray = (
        cv2.inpaint(curr_gray_raw, curr_guard, 3.0, cv2.INPAINT_TELEA) if np.any(curr_guard_mask) else curr_gray_raw
    )
    nb_gray = cv2.inpaint(nb_gray_raw, nb_guard, 3.0, cv2.INPAINT_TELEA) if np.any(nb_guard > 0) else nb_gray_raw

    flow_to_nb = _flow_field(curr_gray, nb_gray)
    flow_from_nb = _flow_field(nb_gray, curr_gray)
    raw_maps = _displacement_to_maps(flow_to_nb)
    back_flow = _backward_warp(flow_from_nb, raw_maps)
    fb_err = np.linalg.norm(flow_to_nb + back_flow, axis=2)

    warped_nb_guard = _backward_warp(nb_guard, raw_maps, interpolation=cv2.INTER_NEAREST) > 0
    context = ~(curr_guard_mask | warped_nb_guard) & (fb_err <= fb_tolerance_px)
    if int(np.count_nonzero(context)) < 24:
        return raw_maps

    cy, cx = np.where(context)
    u_vals = flow_to_nb[cy, cx, 0]
    v_vals = flow_to_nb[cy, cx, 1]
    med_u, med_v = float(np.median(u_vals)), float(np.median(v_vals))
    inliers = (np.abs(u_vals - med_u) <= 3.0) & (np.abs(v_vals - med_v) <= 3.0)
    cx_in = cx[inliers].astype(np.float32)
    if len(cx_in) < 24:
        return raw_maps

    design = np.column_stack((cx_in, cy[inliers].astype(np.float32), np.ones_like(cx_in)))
    coef_u, *_ = np.linalg.lstsq(design, u_vals[inliers], rcond=None)
    coef_v, *_ = np.linalg.lstsq(design, v_vals[inliers], rcond=None)
    grid_x = np.arange(crop_w, dtype=np.float32)[None, :]
    grid_y = np.arange(crop_h, dtype=np.float32)[:, None]
    prop_flow = np.empty((crop_h, crop_w, 2), dtype=np.float32)
    prop_flow[..., 0] = coef_u[0] * grid_x + coef_u[1] * grid_y + coef_u[2]
    prop_flow[..., 1] = coef_v[0] * grid_x + coef_v[1] * grid_y + coef_v[2]
    keep_direct = ~curr_guard_mask & (fb_err <= fb_tolerance_px)
    prop_flow[keep_direct] = flow_to_nb[keep_direct]
    return _displacement_to_maps(prop_flow)


def stabilize_filled_frame_bidirectional(
    current_source: NDArray[Any],
    current_cleaned: NDArray[Any],
    current_mask: NDArray[Any],
    *,
    previous: tuple[NDArray[Any], NDArray[Any], NDArray[Any]] | None = None,
    nxt: tuple[NDArray[Any], NDArray[Any], NDArray[Any]] | None = None,
    blend: float = 0.5,
    max_context_residual: float = 12.0,
    copy: bool = True,
) -> NDArray[Any]:
    """Blend background-aligned prior and future fills when unmasked context agrees.

    Motion inside the watermark mask is estimated from the surrounding unmasked
    background so static corner logos do not pin optical flow to zero during
    camera pans. Each neighbor is independently gated by overlap and context
    photometric residual, rejecting scene cuts and occlusions.
    """
    if not 0.0 <= blend <= 1.0:
        raise ValueError("Temporal blend must be between 0 and 1")
    if max_context_residual <= 0.0:
        raise ValueError("Context residual threshold must be positive")
    if current_source.shape != current_cleaned.shape or current_mask.shape != current_source.shape[:2]:
        raise ValueError("Temporal fill inputs must share frame and mask geometry")

    neighbors = [entry for entry in (previous, nxt) if entry is not None]
    if not neighbors:
        return current_cleaned
    for nb_src, nb_clean, nb_mask in neighbors:
        if (
            nb_src.shape != current_source.shape
            or nb_clean.shape != current_cleaned.shape
            or nb_mask.shape != current_mask.shape
        ):
            raise ValueError("Temporal fill inputs must share frame and mask geometry")

    union = current_mask > 0
    for _nb_src, _nb_clean, nb_mask in neighbors:
        union = union | (nb_mask > 0)
    ys, xs = np.where(union)
    if len(xs) == 0:
        return current_cleaned
    height, width = current_mask.shape
    mask_width = int(xs.max() - xs.min() + 1)
    mask_height = int(ys.max() - ys.min() + 1)
    padding = max(24, round(max(mask_width, mask_height) * 0.75))
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(width, int(xs.max()) + padding + 1)
    y1 = min(height, int(ys.max()) + padding + 1)

    current_source_crop = current_source[y0:y1, x0:x1]
    current_cleaned_crop = current_cleaned[y0:y1, x0:x1]
    current_mask_crop = current_mask[y0:y1, x0:x1]
    current_hole = current_mask_crop > 0
    if not np.any(current_hole):
        return current_cleaned

    eff_max_residual, peak = _depth_bounds(current_cleaned.dtype, max_context_residual)

    dilation = max(7, round(max(mask_width, mask_height) * 0.25))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation | 1, dilation | 1))

    accepted: list[tuple[float, NDArray[Any], NDArray[Any]]] = []
    for nb_src, nb_clean, nb_mask in neighbors:
        nb_src_crop = nb_src[y0:y1, x0:x1]
        nb_clean_crop = nb_clean[y0:y1, x0:x1]
        nb_mask_crop = nb_mask[y0:y1, x0:x1]
        maps = _background_aligned_maps(current_source_crop, current_mask_crop, nb_src_crop, nb_mask_crop)
        warped_nb_src = _backward_warp(nb_src_crop, maps)
        warped_nb_clean = _backward_warp(nb_clean_crop, maps)
        warped_nb_mask = _backward_warp(nb_mask_crop, maps, interpolation=cv2.INTER_NEAREST)

        covered = current_hole & (warped_nb_mask > 0)
        if float(np.mean(covered[current_hole])) < 0.85:
            continue

        occupied = current_hole | (warped_nb_mask > 0)
        context = cv2.dilate(occupied.astype(np.uint8), kernel).astype(bool) & ~occupied
        if np.count_nonzero(context) < 64:
            continue
        residual = np.abs(current_source_crop.astype(np.float32) - warped_nb_src.astype(np.float32))
        context_residual = float(np.mean(residual[context]))
        if context_residual > eff_max_residual:
            continue
        conf = max(0.0, 1.0 - context_residual / eff_max_residual)
        accepted.append((conf, covered, warped_nb_clean.astype(np.float32)))

    if not accepted:
        return current_cleaned

    curr_f32 = current_cleaned_crop.astype(np.float32)
    nb_accum = np.zeros_like(curr_f32)
    w_accum = np.zeros(current_hole.shape, dtype=np.float32)
    if len(accepted) == 1:
        conf, covered, warped = accepted[0]
        w = blend * conf
        nb_accum[covered] = w * warped[covered]
        w_accum[covered] = w
    else:
        # Favor the already-stabilized prior frame slightly so frame-to-frame deltas stay smooth
        # while the future lookahead frame still cuts independent inpaint variance.
        weighted = [
            (conf * (1.35 if idx == 0 and previous is not None else 1.0), cov, warped)
            for idx, (conf, cov, warped) in enumerate(accepted)
        ]
        total_conf = sum(c for c, _, _ in weighted)
        if total_conf <= 1e-6:
            return current_cleaned
        max_conf = max(c for c, _, _ in accepted)
        total_blend = min(0.75, blend * 1.25) * max_conf
        for conf, covered, warped in weighted:
            w = total_blend * (conf / total_conf)
            nb_accum[covered] += w * warped[covered]
            w_accum[covered] += w

    active = w_accum > 0.0
    if not np.any(active):
        return current_cleaned
    blended = (1.0 - w_accum[active, None]) * curr_f32[active] + nb_accum[active]
    result = current_cleaned.copy() if copy else current_cleaned
    result_crop = result[y0:y1, x0:x1]
    result_crop[active] = np.clip(np.rint(blended), 0, peak).astype(current_cleaned.dtype)
    return result


def stabilize_filled_frame(
    previous_source: NDArray[Any],
    previous_cleaned: NDArray[Any],
    previous_mask: NDArray[Any],
    current_source: NDArray[Any],
    current_cleaned: NDArray[Any],
    current_mask: NDArray[Any],
    *,
    blend: float = 0.5,
    max_context_residual: float = 12.0,
    copy: bool = True,
) -> NDArray[Any]:
    """Blend a motion-aligned prior fill when nearby source pixels agree."""
    return stabilize_filled_frame_bidirectional(
        current_source,
        current_cleaned,
        current_mask,
        previous=(previous_source, previous_cleaned, previous_mask),
        nxt=None,
        blend=blend,
        max_context_residual=max_context_residual,
        copy=copy,
    )


def _coarse_shifted_flow(
    curr_gray: NDArray[Any],
    donor_gray: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Estimate dense optical flow with FFT coarse pre-alignment for >4 px camera pans."""
    h, w = curr_gray.shape[:2]
    shift, resp = cv2.phaseCorrelate(curr_gray.astype(np.float32), donor_gray.astype(np.float32))
    cdx, cdy = round(float(shift[0])), round(float(shift[1]))
    if resp >= 0.15 and max(abs(cdx), abs(cdy)) >= 4 and abs(cdx) < w // 2 and abs(cdy) < h // 2:
        # phaseCorrelate(src1, src2) returns shift such that src2(x + dx, y + dy) ~= src1(x, y)
        mat = np.array([[1.0, 0.0, -float(cdx)], [0.0, 1.0, -float(cdy)]], dtype=np.float32)
        aligned_donor = cv2.warpAffine(donor_gray, mat, (w, h), borderMode=cv2.BORDER_REFLECT)
        x_lo, x_hi = max(0, -cdx), min(w, w - cdx)
        y_lo, y_hi = max(0, -cdy), min(h, h - cdy)
        roi_curr = curr_gray[y_lo:y_hi, x_lo:x_hi].astype(np.float32)
        raw_mae = float(np.mean(np.abs(roi_curr - donor_gray[y_lo:y_hi, x_lo:x_hi].astype(np.float32))))
        aligned_mae = float(np.mean(np.abs(roi_curr - aligned_donor[y_lo:y_hi, x_lo:x_hi].astype(np.float32))))
        if aligned_mae < raw_mae * 0.65:
            res_to = _flow_field(curr_gray, aligned_donor)
            res_from = _flow_field(aligned_donor, curr_gray)
            res_maps = _displacement_to_maps(res_to)
            fb_err = np.linalg.norm(res_to + _backward_warp(res_from, res_maps), axis=2)
            flow_to = res_to.copy()
            flow_to[..., 0] += float(cdx)
            flow_to[..., 1] += float(cdy)
            return flow_to, fb_err

    flow_to = _flow_field(curr_gray, donor_gray)
    flow_from = _flow_field(donor_gray, curr_gray)
    raw_maps = _displacement_to_maps(flow_to)
    fb_err = np.linalg.norm(flow_to + _backward_warp(flow_from, raw_maps), axis=2)
    return flow_to, fb_err


def recover_unobscured_background(
    current_source: NDArray[Any],
    current_mask: NDArray[Any],
    donors: Sequence[tuple[NDArray[Any], NDArray[Any] | None]],
    *,
    max_context_residual: float = 12.0,
    fb_tolerance_px: float = 1.75,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Propagate real unobscured background pixels from motion-aligned neighbor frames.

    Unlike ``stabilize_filled_frame`` (which blends two already-inpainted guesses
    when masks overlap), this stage runs *before* single-frame inpainting: when a
    moving or intermittent watermark uncovers background pixels in nearby frames
    ``t +/- k``, forward-backward consistent optical flow warps those genuine
    background pixels into ``current_mask`` and shrinks the remaining hole. Scene
    cuts and occlusions are rejected via an unmasked context-ring residual gate.
    """
    if max_context_residual <= 0.0 or fb_tolerance_px <= 0.0:
        raise ValueError("Context residual and flow consistency thresholds must be positive")
    if current_mask.shape != current_source.shape[:2]:
        raise ValueError("Current source and mask must share spatial geometry")
    remaining = current_mask.copy()
    if not donors:
        return current_source, remaining
    ys, xs = np.where(current_mask > 0)
    if len(xs) == 0:
        return current_source, remaining

    height, width = current_mask.shape
    mask_width = int(xs.max() - xs.min() + 1)
    mask_height = int(ys.max() - ys.min() + 1)
    padding = max(40, round(max(mask_width, mask_height) * 1.0))
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(width, int(xs.max()) + padding + 1)
    y1 = min(height, int(ys.max()) + padding + 1)
    crop_h, crop_w = y1 - y0, x1 - x0

    eff_max_residual, peak = _depth_bounds(current_source.dtype, max_context_residual)

    curr_src_crop = current_source[y0:y1, x0:x1]
    curr_src_f32 = curr_src_crop.astype(np.float32)
    curr_mask_crop = current_mask[y0:y1, x0:x1]
    curr_hole = curr_mask_crop > 0

    guard_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    curr_guard = cv2.dilate(curr_hole.astype(np.uint8) * 255, guard_kernel)
    curr_guard_mask = curr_guard > 0
    curr_gray_raw = _to_gray_u8(curr_src_crop)
    # Pre-inpaint the watermark out of the flow-estimation grayscale image so
    # Farneback tracks background motion rather than the moving overlay itself.
    curr_gray_clean = cv2.inpaint(curr_gray_raw, curr_guard, 3.0, cv2.INPAINT_TELEA)

    ring_dilation = max(9, round(max(mask_width, mask_height) * 0.30)) | 1
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_dilation, ring_dilation))
    outer_ring = cv2.dilate(curr_guard_mask.astype(np.uint8), ring_kernel).astype(bool)
    grid_x = np.arange(crop_w, dtype=np.float32)[None, :]
    grid_y = np.arange(crop_h, dtype=np.float32)[:, None]

    def _project_donor_flow(
        flow: NDArray[Any],
        donor_guard_arr: NDArray[Any] | None,
    ) -> tuple[tuple[NDArray[Any], NDArray[Any]], NDArray[Any], NDArray[Any], NDArray[Any]]:
        maps_pair = _displacement_to_maps(flow)
        bounds = (
            (maps_pair[0] >= 0.0)
            & (maps_pair[0] <= float(crop_w - 1))
            & (maps_pair[1] >= 0.0)
            & (maps_pair[1] <= float(crop_h - 1))
        )
        donor_masked = (
            _backward_warp(donor_guard_arr, maps_pair, interpolation=cv2.INTER_NEAREST) > 0
            if donor_guard_arr is not None
            else np.zeros((crop_h, crop_w), dtype=bool)
        )
        ctx = outer_ring & ~(curr_guard_mask | donor_masked) & bounds
        return maps_pair, bounds, donor_masked, ctx

    accum_rgb = np.zeros((crop_h, crop_w, 3), dtype=np.float32)
    accum_w = np.zeros((crop_h, crop_w), dtype=np.float32)

    for rank, (donor_source, donor_mask) in enumerate(donors):
        if donor_source.shape != current_source.shape:
            raise ValueError("Donor frame must match current source geometry")
        if donor_mask is not None and donor_mask.shape != current_mask.shape:
            raise ValueError("Donor mask must match current mask geometry")

        donor_src_crop = donor_source[y0:y1, x0:x1]
        donor_gray_raw = _to_gray_u8(donor_src_crop)
        donor_gray_clean = donor_gray_raw
        donor_guard: NDArray[Any] | None = None
        if donor_mask is not None:
            donor_mask_crop = donor_mask[y0:y1, x0:x1]
            if np.any(donor_mask_crop > 0):
                donor_guard = cv2.dilate((donor_mask_crop > 0).astype(np.uint8) * 255, guard_kernel)
                donor_gray_clean = cv2.inpaint(donor_gray_raw, donor_guard, 3.0, cv2.INPAINT_TELEA)

        flow_to_donor, fb_err = _coarse_shifted_flow(curr_gray_clean, donor_gray_clean)
        _raw_maps, _raw_bounds, _raw_donor_masked, raw_context = _project_donor_flow(flow_to_donor, donor_guard)

        consistent_ctx = raw_context & (fb_err <= fb_tolerance_px)
        if int(np.count_nonzero(consistent_ctx)) < 48:
            continue

        # Inside curr_guard the source background is hidden by the watermark, so
        # dense optical flow has no true texture there. Fit a robust local affine
        # motion model on the verified unmasked context ring and evaluate the
        # context photometric gate on that propagated background motion field.
        cy, cx = np.where(consistent_ctx)
        u_vals = flow_to_donor[cy, cx, 0]
        v_vals = flow_to_donor[cy, cx, 1]
        med_u, med_v = float(np.median(u_vals)), float(np.median(v_vals))
        inliers = (np.abs(u_vals - med_u) <= 3.0) & (np.abs(v_vals - med_v) <= 3.0)
        cx_in = cx[inliers].astype(np.float32)
        if len(cx_in) < 32:
            continue
        design = np.column_stack((cx_in, cy[inliers].astype(np.float32), np.ones_like(cx_in)))
        coef_u, *_ = np.linalg.lstsq(design, u_vals[inliers], rcond=None)
        coef_v, *_ = np.linalg.lstsq(design, v_vals[inliers], rcond=None)
        prop_flow = np.empty((crop_h, crop_w, 2), dtype=np.float32)
        prop_flow[..., 0] = coef_u[0] * grid_x + coef_u[1] * grid_y + coef_u[2]
        prop_flow[..., 1] = coef_v[0] * grid_x + coef_v[1] * grid_y + coef_v[2]
        # Keep direct cycle-consistent Farneback outside the occluded region
        keep_direct = ~curr_guard_mask & (fb_err <= fb_tolerance_px)
        prop_flow[keep_direct] = flow_to_donor[keep_direct]

        maps, in_bounds, warped_donor_masked, context = _project_donor_flow(prop_flow, donor_guard)
        eligible = curr_hole & ~warped_donor_masked & in_bounds
        if not np.any(eligible) or int(np.count_nonzero(context)) < 48:
            continue

        warped_donor_src = _backward_warp(donor_src_crop.astype(np.float32), maps)
        residual = np.mean(np.abs(curr_src_f32 - warped_donor_src), axis=2)
        context_residual = float(np.mean(residual[context]))
        if context_residual > eff_max_residual:
            continue

        weight = max(0.05, (1.0 - context_residual / eff_max_residual) / (1.0 + 0.25 * rank))
        accum_rgb[eligible] += weight * warped_donor_src[eligible]
        accum_w[eligible] += weight
        if bool(np.all(accum_w[curr_hole] >= 1.1)):
            break

    recovered_px = curr_hole & (accum_w > 0.0)
    if not np.any(recovered_px):
        return current_source, remaining

    result = current_source.copy()
    result_crop = result[y0:y1, x0:x1]
    result_crop[recovered_px] = np.clip(
        np.rint(accum_rgb[recovered_px] / accum_w[recovered_px, None]),
        0,
        peak,
    ).astype(current_source.dtype)

    rem_crop = remaining[y0:y1, x0:x1]
    rem_crop[recovered_px] = 0
    if np.any(rem_crop > 0):
        seam_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        rem_crop[:] = cv2.dilate(rem_crop, seam_kernel) & curr_mask_crop
    return result, remaining


def _flow_field(
    from_gray: NDArray[Any],
    to_gray: NDArray[Any],
) -> NDArray[Any]:
    """Estimate dense Farneback displacement from ``from_gray`` into ``to_gray``."""
    return cv2.calcOpticalFlowFarneback(
        from_gray,
        to_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )


def _displacement_to_maps(flow: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
    height, width = flow.shape[:2]
    map_x = flow[..., 0] + np.arange(width, dtype=np.float32)[None, :]
    map_y = flow[..., 1] + np.arange(height, dtype=np.float32)[:, None]
    return map_x, map_y


def _neighbor_blend_term(
    current_source_f32: NDArray[Any],
    neighbor_source_f32: NDArray[Any],
    neighbor_regenerated_f32: NDArray[Any],
    flow_to_neighbor: NDArray[Any],
    flow_from_neighbor: NDArray[Any],
    *,
    neighbor_weight: float,
    fb_tolerance_px: float,
    photometric_tolerance: float,
    min_valid_fraction: float,
) -> tuple[NDArray[Any] | None, NDArray[Any] | None]:
    """Return ``(weight_map, warped_neighbor_regenerated)`` using source motion gates only."""
    maps = _displacement_to_maps(flow_to_neighbor)
    back_flow = _backward_warp(flow_from_neighbor, maps)
    fb_error = np.linalg.norm(flow_to_neighbor + back_flow, axis=2)
    warped_source = _backward_warp(neighbor_source_f32, maps)
    photo_error = np.mean(np.abs(current_source_f32 - warped_source), axis=2)

    valid = (fb_error <= fb_tolerance_px) & (photo_error <= photometric_tolerance)
    if float(np.mean(valid)) < min_valid_fraction:
        return None, None

    fb_conf = np.clip(1.0 - fb_error / fb_tolerance_px, 0.0, 1.0)
    photo_conf = np.clip(1.0 - photo_error / photometric_tolerance, 0.0, 1.0)
    weight = (neighbor_weight * fb_conf * photo_conf).astype(np.float32)
    warped_regenerated = _backward_warp(neighbor_regenerated_f32, maps)
    return weight, warped_regenerated


def _validate_stabilization_controls(
    *,
    neighbor_weight: float,
    fb_tolerance_px: float,
    photometric_tolerance: float,
    min_valid_fraction: float,
) -> None:
    if not 0.0 <= neighbor_weight <= 0.35:
        raise ValueError("Neighbor weight must be between 0 and 0.35")
    if fb_tolerance_px <= 0.0:
        raise ValueError("Forward-backward flow tolerance must be positive")
    if photometric_tolerance <= 0.0:
        raise ValueError("Photometric tolerance must be positive")
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("Minimum valid pixel fraction must be between 0 and 1")


def _prepare_frame_pair(
    source_frame: NDArray[Any],
    regenerated_frame: NDArray[Any],
    expected_shape: tuple[int, ...] | None,
) -> tuple[tuple[int, ...], NDArray[Any], NDArray[Any], NDArray[Any]]:
    shape = regenerated_frame.shape if expected_shape is None else expected_shape
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError("Stabilization expects 3-channel uint8 or uint16 video frames")
    if source_frame.shape != shape or regenerated_frame.shape != shape:
        raise ValueError("All source and regenerated frames must share identical geometry")
    if (
        source_frame.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16))
        or regenerated_frame.dtype != source_frame.dtype
    ):
        raise ValueError("Stabilization expects matching uint8 or uint16 video frames")
    return (
        shape,
        _to_gray_u8(source_frame),
        source_frame.astype(np.float32),
        regenerated_frame.astype(np.float32),
    )


def _blend_triplet(
    prev_entry: tuple[NDArray[Any], NDArray[Any]] | None,
    prev_flows: tuple[NDArray[Any], NDArray[Any]] | None,
    curr_src_f32: NDArray[Any],
    curr_reg_f32: NDArray[Any],
    next_entry: tuple[NDArray[Any], NDArray[Any]] | None,
    next_flows: tuple[NDArray[Any], NDArray[Any]] | None,
    *,
    neighbor_weight: float,
    fb_tolerance_px: float,
    photometric_tolerance: float,
    min_valid_fraction: float,
    out_dtype: np.dtype[Any] | None = None,
) -> NDArray[Any]:
    eff_dtype = np.dtype(np.uint8) if out_dtype is None else out_dtype
    eff_photo_tol, peak = _depth_bounds(eff_dtype, photometric_tolerance)
    accum = curr_reg_f32.copy()
    if prev_entry is not None and prev_flows is not None:
        fwd_prev_to_curr, bwd_curr_to_prev = prev_flows
        w_prev, warp_prev = _neighbor_blend_term(
            curr_src_f32,
            prev_entry[0],
            prev_entry[1],
            bwd_curr_to_prev,
            fwd_prev_to_curr,
            neighbor_weight=neighbor_weight,
            fb_tolerance_px=fb_tolerance_px,
            photometric_tolerance=eff_photo_tol,
            min_valid_fraction=min_valid_fraction,
        )
        if w_prev is not None and warp_prev is not None:
            accum += w_prev[:, :, None] * (warp_prev - curr_reg_f32)
    if next_entry is not None and next_flows is not None:
        fwd_curr_to_next, bwd_next_to_curr = next_flows
        w_next, warp_next = _neighbor_blend_term(
            curr_src_f32,
            next_entry[0],
            next_entry[1],
            fwd_curr_to_next,
            bwd_next_to_curr,
            neighbor_weight=neighbor_weight,
            fb_tolerance_px=fb_tolerance_px,
            photometric_tolerance=eff_photo_tol,
            min_valid_fraction=min_valid_fraction,
        )
        if w_next is not None and warp_next is not None:
            accum += w_next[:, :, None] * (warp_next - curr_reg_f32)
    return np.clip(np.round(accum), 0, peak).astype(eff_dtype)


def iter_stabilized_regenerated_frames(
    frame_pairs: Iterable[tuple[NDArray[Any], NDArray[Any]]],
    *,
    neighbor_weight: float = 0.25,
    fb_tolerance_px: float = 1.5,
    photometric_tolerance: float = 14.0,
    min_valid_fraction: float = 0.35,
) -> Generator[tuple[NDArray[Any], NDArray[Any]], None, None]:
    """Stream ``(source_frame, stabilized_regenerated_frame)`` with a 3-frame window."""
    _validate_stabilization_controls(
        neighbor_weight=neighbor_weight,
        fb_tolerance_px=fb_tolerance_px,
        photometric_tolerance=photometric_tolerance,
        min_valid_fraction=min_valid_fraction,
    )
    iterator = iter(frame_pairs)
    first = next(iterator, None)
    if first is None:
        return
    shape, curr_gray, curr_src_f32, curr_reg_f32 = _prepare_frame_pair(first[0], first[1], None)
    curr_raw_src = first[0]
    curr_raw_reg = first[1]

    if neighbor_weight == 0.0:
        yield curr_raw_src, curr_raw_reg.copy()
        for src_frame, reg_frame in iterator:
            _prepare_frame_pair(src_frame, reg_frame, shape)
            yield src_frame, reg_frame.copy()
        return

    prev_entry: tuple[NDArray[Any], NDArray[Any]] | None = None
    prev_flows: tuple[NDArray[Any], NDArray[Any]] | None = None

    for next_src, next_reg in iterator:
        shape, next_gray, next_src_f32, next_reg_f32 = _prepare_frame_pair(next_src, next_reg, shape)
        next_flows = (_flow_field(curr_gray, next_gray), _flow_field(next_gray, curr_gray))
        yield (
            curr_raw_src,
            _blend_triplet(
                prev_entry,
                prev_flows,
                curr_src_f32,
                curr_reg_f32,
                (next_src_f32, next_reg_f32),
                next_flows,
                neighbor_weight=neighbor_weight,
                fb_tolerance_px=fb_tolerance_px,
                photometric_tolerance=photometric_tolerance,
                min_valid_fraction=min_valid_fraction,
                out_dtype=curr_raw_reg.dtype,
            ),
        )
        prev_entry = (curr_src_f32, curr_reg_f32)
        prev_flows = next_flows
        curr_raw_src = next_src
        curr_raw_reg = next_reg
        curr_gray = next_gray
        curr_src_f32 = next_src_f32
        curr_reg_f32 = next_reg_f32

    if prev_entry is None:
        yield curr_raw_src, curr_raw_reg.copy()
    else:
        yield (
            curr_raw_src,
            _blend_triplet(
                prev_entry,
                prev_flows,
                curr_src_f32,
                curr_reg_f32,
                None,
                None,
                neighbor_weight=neighbor_weight,
                fb_tolerance_px=fb_tolerance_px,
                photometric_tolerance=photometric_tolerance,
                min_valid_fraction=min_valid_fraction,
                out_dtype=curr_raw_reg.dtype,
            ),
        )


def stabilize_regenerated_frames(
    source_frames: Sequence[NDArray[Any]],
    regenerated_frames: Sequence[NDArray[Any]],
    *,
    neighbor_weight: float = 0.25,
    fb_tolerance_px: float = 1.5,
    photometric_tolerance: float = 14.0,
    min_valid_fraction: float = 0.35,
) -> list[NDArray[Any]]:
    """Stabilize independently regenerated frames using motion-gated neighbor warps.

    Source frames are used exclusively to estimate optical flow and gate out
    occlusions, non-rigid motion, and scene cuts. Raw source pixels never enter
    the blended output, preventing direct re-injection of source watermarks.
    """
    if len(source_frames) != len(regenerated_frames):
        raise ValueError("Source and regenerated sequences must have identical frame counts")
    return [
        stabilized
        for _src, stabilized in iter_stabilized_regenerated_frames(
            zip(source_frames, regenerated_frames, strict=True),
            neighbor_weight=neighbor_weight,
            fb_tolerance_px=fb_tolerance_px,
            photometric_tolerance=photometric_tolerance,
            min_valid_fraction=min_valid_fraction,
        )
    ]


# Public internal-use aliases keep fidelity measurements on the producer math.
backward_motion_map = _backward_map
motion_residual_energy = _motion_residual


@dataclass(frozen=True)
class ClipAlphaMatte:
    """Self-calibrated YCrCb alpha/composite matte over a fixed bounding box."""

    bounds: tuple[int, int, int, int]  # (x0, y0, x1, y1)
    alpha_y: NDArray[Any]
    alpha_c: NDArray[Any]
    c_y: NDArray[Any]
    c_cr: NDArray[Any]
    c_cb: NDArray[Any]
    calibrated: NDArray[Any]


def estimate_clip_alpha_matte(
    frames_bgr: Sequence[NDArray[Any]],
    bounds: tuple[int, int, int, int],
) -> ClipAlphaMatte | None:
    """Self-calibrate (alpha, alpha*W) in YCrCb across moving frames via PyrLK+RANSAC warps.

    Returns ``None`` when the sequence has fewer than 12 frames or lacks sufficient
    background motion around ``bounds`` to identify per-pixel compositing parameters.
    """
    n_frames = len(frames_bgr)
    if n_frames < 12:
        return None
    x0, y0, x1, y1 = bounds
    height, width = frames_bgr[0].shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(width, int(x1)), min(height, int(y1))
    rh_box, rw_box = y1 - y0, x1 - x0
    if rh_box < 6 or rw_box < 6:
        return None

    ctx_pad = max(36, min(72, max(rh_box, rw_box)))
    cx0, cy0 = max(0, x0 - ctx_pad), max(0, y0 - ctx_pad)
    cx1, cy1 = min(width, x1 + ctx_pad), min(height, y1 + ctx_pad)
    ch, cw = cy1 - cy0, cx1 - cx0
    hx0, hy0 = x0 - cx0, y0 - cy0
    hx1, hy1 = x1 - cx0, y1 - cy0

    feat_mask = np.full((ch, cw), 255, dtype=np.uint8)
    feat_mask[max(0, hy0 - 2) : min(ch, hy1 + 2), max(0, hx0 - 2) : min(cw, hx1 + 2)] = 0
    ctx_bool = feat_mask > 0
    if int(np.count_nonzero(ctx_bool)) < 256:
        return None

    gx_c = np.arange(cw, dtype=np.float32)[None, :]
    gy_c = np.arange(ch, dtype=np.float32)[:, None]

    is_u16 = frames_bgr[0].dtype == np.uint16
    if is_u16:
        scale_to_u8 = 255.0 / 65535.0
        crops_ycrcb = [
            cv2.cvtColor(frame[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2YCrCb).astype(np.float32) * scale_to_u8
            for frame in frames_bgr
        ]
        grays = [np.clip(np.rint(crop[:, :, 0]), 0, 255).astype(np.uint8) for crop in crops_ycrcb]
    else:
        crops_ycrcb = [
            cv2.cvtColor(frame[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2YCrCb).astype(np.float32) for frame in frames_bgr
        ]
        grays = [crop[:, :, 0].astype(np.uint8) for crop in crops_ycrcb]

    step_mats: list[NDArray[Any] | None] = []
    for k in range(n_frames - 1):
        pts0 = cv2.goodFeaturesToTrack(
            grays[k],
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=4,
            mask=feat_mask,
        )
        if pts0 is None or len(pts0) < 8:
            step_mats.append(None)
            continue
        pts1, st, _err = cv2.calcOpticalFlowPyrLK(
            grays[k],
            grays[k + 1],
            pts0,
            None,
            winSize=(15, 15),
            maxLevel=3,
        )
        ok = st.ravel() == 1
        if int(np.count_nonzero(ok)) < 8:
            step_mats.append(None)
            continue
        aff, inl = cv2.estimateAffinePartial2D(
            pts0[ok],
            pts1[ok],
            method=cv2.RANSAC,
            ransacReprojThreshold=1.5,
        )
        if aff is None or inl is None or int(np.count_nonzero(inl)) < 6:
            step_mats.append(None)
            continue
        mat = np.eye(3, dtype=np.float64)
        mat[:2, :] = aff
        step_mats.append(mat)

    def _chain_map(k_src: int, k_dst: int) -> NDArray[Any] | None:
        mat = np.eye(3, dtype=np.float64)
        if k_dst > k_src:
            for j in range(k_src, k_dst):
                sj = step_mats[j]
                if sj is None:
                    return None
                mat = sj @ mat
        else:
            for j in range(k_src - 1, k_dst - 1, -1):
                sj = step_mats[j]
                if sj is None:
                    return None
                try:
                    mat = np.linalg.inv(sj) @ mat
                except np.linalg.LinAlgError:
                    return None
        return mat

    alpha_hat = np.zeros((rh_box, rw_box), dtype=np.float32)
    c_y = np.zeros((rh_box, rw_box), dtype=np.float32)
    c_cr = np.zeros((rh_box, rw_box), dtype=np.float32)
    c_cb = np.zeros((rh_box, rw_box), dtype=np.float32)
    calibrated = np.zeros((rh_box, rw_box), dtype=bool)
    support_crop = np.zeros((ch, cw), dtype=bool)
    support_crop[hy0:hy1, hx0:hx1] = True

    warps_by_frame: list[list[tuple[int, NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]]] = [
        [] for _ in range(n_frames)
    ]
    for k in range(n_frames):
        for delta in (-12, -9, -6, -4, -2, 2, 4, 6, 9, 12):
            kd = k + delta
            if kd < 0 or kd >= n_frames:
                continue
            mat = _chain_map(k, kd)
            if mat is None or float(np.hypot(mat[0, 2], mat[1, 2])) < 4.0:
                continue
            map_x = (mat[0, 0] * gx_c + mat[0, 1] * gy_c + mat[0, 2]).astype(np.float32)
            map_y = (mat[1, 0] * gx_c + mat[1, 1] * gy_c + mat[1, 2]).astype(np.float32)
            in_b = (map_x >= 1.0) & (map_x <= cw - 2.0) & (map_y >= 1.0) & (map_y <= ch - 2.0)
            ctx_in = ctx_bool & in_b
            if int(np.count_nonzero(ctx_in)) < 128:
                continue
            warps_by_frame[k].append((kd, map_x, map_y, in_b, ctx_in))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for _pass in range(3):
            if _pass > 0:
                dy = np.maximum(1.0 - alpha_hat, 0.20)
                ac_d = cv2.GaussianBlur(alpha_hat, (3, 3), 0.7)
                dc = np.maximum(1.0 - ac_d, 0.20)
                act_dy = (alpha_hat >= 0.02) & (alpha_hat <= 0.78)
                act_dc = (ac_d >= 0.02) & (ac_d <= 0.78)
                donor_pool: list[NDArray[Any]] = []
                for base_crop in crops_ycrcb:
                    unblended = base_crop.copy()
                    d_roi = unblended[hy0:hy1, hx0:hx1]
                    d_roi[:, :, 0] = np.where(act_dy, (d_roi[:, :, 0] - c_y) / dy, d_roi[:, :, 0])
                    d_roi[:, :, 1] = np.where(act_dc, (d_roi[:, :, 1] - c_cr) / dc, d_roi[:, :, 1])
                    d_roi[:, :, 2] = np.where(act_dc, (d_roi[:, :, 2] - c_cb) / dc, d_roi[:, :, 2])
                    donor_pool.append(unblended)
            else:
                donor_pool = crops_ycrcb
            sup_u8 = support_crop.astype(np.uint8)

            obs_list: list[NDArray[Any]] = []
            ref_list: list[NDArray[Any]] = []
            for k in range(n_frames):
                cand_stack: list[NDArray[Any]] = []
                cand_valid: list[NDArray[Any]] = []
                for kd, map_x, map_y, in_b, ctx_in in warps_by_frame[k]:
                    warped = cv2.remap(
                        donor_pool[kd],
                        map_x,
                        map_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT,
                    )
                    ctx_err = float(np.mean(np.abs(crops_ycrcb[k][:, :, 0][ctx_in] - warped[:, :, 0][ctx_in])))
                    if ctx_err > 9.0:
                        continue
                    warped_sup = (
                        cv2.remap(
                            sup_u8,
                            map_x,
                            map_y,
                            interpolation=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=1,
                        )
                        > 0
                    )
                    v_px = in_b[hy0:hy1, hx0:hx1] & ~warped_sup[hy0:hy1, hx0:hx1]
                    cand_stack.append(warped[hy0:hy1, hx0:hx1])
                    cand_valid.append(v_px)
                if cand_stack:
                    cs = np.stack(cand_stack, axis=0)
                    cv_m = np.stack(cand_valid, axis=0)
                    masked_cs = np.where(cv_m[..., None], cs, np.nan)
                    obs_list.append(crops_ycrcb[k][hy0:hy1, hx0:hx1])
                    ref_list.append(np.nanmedian(masked_cs, axis=0))

            if len(obs_list) < 8:
                return None
            i_arr = np.stack(obs_list, axis=0)
            b_arr = np.stack(ref_list, axis=0)
            valid = np.isfinite(b_arr[..., 0])
            cnt = np.sum(valid, axis=0)
            denom_cnt = np.maximum(cnt, 1).astype(np.float32)
            b_y = np.where(valid, b_arr[..., 0], 0.0)
            i_y = np.where(valid, i_arr[..., 0], 0.0)
            b_mean = np.sum(b_y, axis=0) / denom_cnt
            i_mean = np.sum(i_y, axis=0) / denom_cnt
            var_b = np.sum(valid * (b_y - b_mean[None, :, :]) ** 2, axis=0) / denom_cnt
            cov_ib = np.sum(valid * (b_y - b_mean[None, :, :]) * (i_y - i_mean[None, :, :]), axis=0) / denom_cnt
            calibrated = (cnt >= 6) & (var_b >= 25.0)
            if float(np.mean(calibrated)) < 0.35:
                return None
            slope = np.where(calibrated, np.clip(cov_ib / np.maximum(var_b, 1e-3), 0.0, 1.0), 1.0)
            a_raw = 1.0 - slope
            cy_raw = np.clip(
                np.nanmedian(np.where(valid, i_arr[..., 0] - slope[None, :, :] * b_arr[..., 0], np.nan), axis=0),
                0.0,
                255.0,
            )
            diff_med = np.nanmedian(np.where(valid, np.abs(i_arr[..., 0] - b_arr[..., 0]), np.nan), axis=0)
            cy_raw = np.nan_to_num(cy_raw, nan=0.0)
            diff_med = np.nan_to_num(diff_med, nan=0.0)
            act = calibrated & (a_raw >= 0.02) & (diff_med >= 1.5)
            alpha_hat = np.where(act, a_raw, 0.0).astype(np.float32)
            c_y = np.where(act, cy_raw, 0.0).astype(np.float32)

            ac = cv2.GaussianBlur(alpha_hat, (3, 3), 0.7)
            ccr_raw = np.nanmedian(
                np.where(valid, i_arr[..., 1] - (1.0 - ac[None, :, :]) * b_arr[..., 1], np.nan),
                axis=0,
            )
            ccb_raw = np.nanmedian(
                np.where(valid, i_arr[..., 2] - (1.0 - ac[None, :, :]) * b_arr[..., 2], np.nan),
                axis=0,
            )
            c_cr = np.where(ac >= 0.02, np.nan_to_num(ccr_raw, nan=0.0), 0.0).astype(np.float32)
            c_cb = np.where(ac >= 0.02, np.nan_to_num(ccb_raw, nan=0.0), 0.0).astype(np.float32)
            if _pass == 1:
                support_crop[hy0:hy1, hx0:hx1] = (~calibrated) | (alpha_hat > 0.78)
            else:
                support_crop[hy0:hy1, hx0:hx1] = (~calibrated) | (alpha_hat >= 0.04)

    if not np.any(alpha_hat >= 0.02):
        return None
    alpha_c = cv2.GaussianBlur(alpha_hat, (3, 3), 0.7).astype(np.float32)
    return ClipAlphaMatte(
        bounds=(x0, y0, x1, y1),
        alpha_y=alpha_hat,
        alpha_c=alpha_c,
        c_y=c_y,
        c_cr=c_cr,
        c_cb=c_cb,
        calibrated=calibrated,
    )


def unblend_ycbcr_roi(
    frame_bgr: NDArray[Any],
    mask: NDArray[Any],
    matte: ClipAlphaMatte,
    *,
    opaque_threshold: float = 0.72,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Algebraically invert translucent watermark pixels in YCrCb and shrink ``mask`` to opaque/uncalibrated core."""
    x0, y0, x1, y1 = matte.bounds
    out_bgr = frame_bgr.copy()
    out_mask = mask.copy()
    roi_mask = mask[y0:y1, x0:x1] > 0
    if not np.any(roi_mask):
        return out_bgr, out_mask

    is_u16 = frame_bgr.dtype == np.uint16
    full_scale = 65535.0 if is_u16 else 255.0
    c_scale = (65535.0 / 255.0) if is_u16 else 1.0
    ycrcb = cv2.cvtColor(out_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2YCrCb).astype(np.float32)
    translucent_y = matte.calibrated & (matte.alpha_y >= 0.02) & (matte.alpha_y <= opaque_threshold)
    translucent_c = matte.calibrated & (matte.alpha_c >= 0.02) & (matte.alpha_c <= opaque_threshold)
    denom_y = np.maximum(1.0 - matte.alpha_y, max(0.15, 1.0 - opaque_threshold))
    denom_c = np.maximum(1.0 - matte.alpha_c, max(0.15, 1.0 - opaque_threshold))

    ycrcb[:, :, 0] = np.where(translucent_y, (ycrcb[:, :, 0] - matte.c_y * c_scale) / denom_y, ycrcb[:, :, 0])
    ycrcb[:, :, 1] = np.where(translucent_c, (ycrcb[:, :, 1] - matte.c_cr * c_scale) / denom_c, ycrcb[:, :, 1])
    ycrcb[:, :, 2] = np.where(translucent_c, (ycrcb[:, :, 2] - matte.c_cb * c_scale) / denom_c, ycrcb[:, :, 2])
    out_bgr[y0:y1, x0:x1] = cv2.cvtColor(
        np.clip(np.rint(ycrcb), 0, full_scale).astype(frame_bgr.dtype),
        cv2.COLOR_YCrCb2BGR,
    )

    # Clear calibrated clean background (alpha < 0.02) AND calibrated translucent pixels (alpha <= opaque_threshold)!
    solved = matte.calibrated & (matte.alpha_y <= opaque_threshold)
    rem_roi = roi_mask & ~solved
    if np.any(rem_roi):
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        rem_roi = (cv2.dilate(rem_roi.astype(np.uint8) * 255, kern) > 0) & roi_mask
    out_mask[y0:y1, x0:x1] = rem_roi.astype(np.uint8) * 255
    return out_bgr, out_mask


def sobel_magnitude(gray: NDArray[Any]) -> NDArray[Any]:
    """Compute 3x3 float32 Sobel gradient magnitude for a single-channel image."""
    f32 = gray.astype(np.float32)
    gx = cv2.Sobel(f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f32, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


_sobel_mag = sobel_magnitude


def _match_patch_in_roi(
    roi_gray: NDArray[Any],
    roi_edge: NDArray[Any],
    ref_gray: NDArray[Any],
    ref_edge: NDArray[Any],
) -> tuple[float, int, int]:
    s_edge = cv2.matchTemplate(roi_edge, ref_edge, cv2.TM_CCOEFF_NORMED)
    s_gray = cv2.matchTemplate(roi_gray.astype(np.float32), ref_gray, cv2.TM_CCOEFF_NORMED)
    combined = 0.65 * np.nan_to_num(s_edge, nan=0.0) + 0.35 * np.nan_to_num(s_gray, nan=0.0)
    _, best_val, _, best_loc = cv2.minMaxLoc(combined)
    return float(best_val), int(best_loc[0]), int(best_loc[1])


def track_video_regions(
    frames_bgr: Iterable[NDArray[Any]],
    initial_regions: Sequence[tuple[int, int, int, int]],
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> list[tuple[tuple[int, int, int, int], ...] | None]:
    """Track one or more user-specified boxes across frames using PyrLK + Sobel/NCC matching."""
    if not initial_regions:
        raise ValueError("initial_regions must be non-empty")
    results: list[tuple[tuple[int, int, int, int], ...] | None] = []
    states: list[dict[str, Any]] = []
    prev_gray: NDArray[Any] | None = None

    for idx, frame_bgr in enumerate(frames_bgr):
        fh, fw = frame_bgr.shape[:2]
        if idx < start_frame:
            results.append(None)
            continue
        if end_frame is not None and idx > end_frame:
            results.append(None)
            continue
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        full_edge: NDArray[Any] | None = None
        if idx == start_frame or not states:
            states.clear()
            cur_boxes: list[tuple[int, int, int, int]] = []
            for rx, ry, rw, rh in initial_regions:
                cx = max(0, min(int(rx), fw - 1))
                cy = max(0, min(int(ry), fh - 1))
                cw = max(1, min(int(rw), fw - cx))
                ch = max(1, min(int(rh), fh - cy))
                p_x0, p_y0 = max(0, cx - 1), max(0, cy - 1)
                p_x1, p_y1 = min(fw, cx + cw + 1), min(fh, cy + ch + 1)
                crop_edge = sobel_magnitude(gray[p_y0:p_y1, p_x0:p_x1])
                states.append(
                    {
                        "x": cx,
                        "y": cy,
                        "w": cw,
                        "h": ch,
                        "ref_gray": gray[cy : cy + ch, cx : cx + cw].astype(np.float32),
                        "ref_edge": crop_edge[cy - p_y0 : cy - p_y0 + ch, cx - p_x0 : cx - p_x0 + cw].copy(),
                    }
                )
                cur_boxes.append((cx, cy, cw, ch))
            prev_gray = gray
            results.append(tuple(cur_boxes))
            continue

        cur_boxes = []
        for st in states:
            px, py, cw, ch = int(st["x"]), int(st["y"]), int(st["w"]), int(st["h"])
            dx, dy = 0.0, 0.0
            if prev_gray is not None and cw >= 6 and ch >= 6:
                patch_prev = prev_gray[py : py + ch, px : px + cw]
                pts0 = cv2.goodFeaturesToTrack(patch_prev, maxCorners=32, qualityLevel=0.05, minDistance=3)
                if pts0 is not None and len(pts0) >= 3:
                    pts0_abs = pts0.astype(np.float32) + np.array([[[px, py]]], dtype=np.float32)
                    pts1, ok_fwd, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, gray, pts0_abs, None, winSize=(15, 15), maxLevel=2
                    )
                    pts0_back, ok_bwd, _ = cv2.calcOpticalFlowPyrLK(
                        gray, prev_gray, pts1, None, winSize=(15, 15), maxLevel=2
                    )
                    fb_err = np.linalg.norm((pts0_abs - pts0_back).reshape(-1, 2), axis=1)
                    good = (ok_fwd.reshape(-1) == 1) & (ok_bwd.reshape(-1) == 1) & (fb_err < 1.5)
                    if int(np.count_nonzero(good)) >= 3:
                        flow_d = (pts1 - pts0_abs).reshape(-1, 2)[good]
                        dx = float(np.median(flow_d[:, 0]))
                        dy = float(np.median(flow_d[:, 1]))

            pred_x = max(0, min(round(px + dx), fw - cw))
            pred_y = max(0, min(round(py + dy), fh - ch))
            rad = max(36, round(max(cw, ch) * 0.85))
            sx0 = max(0, pred_x - rad)
            sy0 = max(0, pred_y - rad)
            sx1 = min(fw, pred_x + cw + rad)
            sy1 = min(fh, pred_y + ch + rad)

            nx, ny = pred_x, pred_y
            best_score = 0.0
            if (sx1 - sx0) >= cw and (sy1 - sy0) >= ch and cw >= 4 and ch >= 4:
                ex0, ey0 = max(0, sx0 - 1), max(0, sy0 - 1)
                ex1, ey1 = min(fw, sx1 + 1), min(fh, sy1 + 1)
                local_edge = sobel_magnitude(gray[ey0:ey1, ex0:ex1])[sy0 - ey0 : sy1 - ey0, sx0 - ex0 : sx1 - ex0]
                best_score, lx, ly = _match_patch_in_roi(
                    gray[sy0:sy1, sx0:sx1],
                    local_edge,
                    st["ref_gray"],
                    st["ref_edge"],
                )
                nx, ny = sx0 + lx, sy0 + ly

            # Jump recovery when a moving overlay teleports across the frame
            if best_score < 0.42 and fw >= cw and fh >= ch and cw >= 8 and ch >= 8:
                if full_edge is None:
                    full_edge = sobel_magnitude(gray)
                global_score, gx, gy = _match_patch_in_roi(gray, full_edge, st["ref_gray"], st["ref_edge"])
                if global_score >= 0.58:
                    nx, ny = gx, gy

            nx = max(0, min(nx, fw - cw))
            ny = max(0, min(ny, fh - ch))
            st["x"], st["y"] = nx, ny
            cur_boxes.append((nx, ny, cw, ch))

        prev_gray = gray
        results.append(tuple(cur_boxes))
    return results


def detect_ai_bumper_start(
    frames_bgr: Iterable[NDArray[Any]],
    *,
    fps: float,
    max_bumper_seconds: float = 4.0,
    min_bumper_seconds: float = 0.5,
) -> int | None:
    """Return the 0-based start frame of a trailing full-screen static AI outro bumper, or None."""
    eff_fps = fps if fps > 0 else 24.0
    min_frames = max(6, round(min_bumper_seconds * eff_fps))
    max_frames = max(min_frames + 1, round(max_bumper_seconds * eff_fps))
    keep_tail = max_frames + min_frames
    raw_tail: deque[NDArray[Any]] = deque(maxlen=keep_tail)
    total = 0
    for frame in frames_bgr:
        raw_tail.append(frame)
        total += 1
    if total < max(18, min_frames * 2):
        return None

    offset = total - len(raw_tail)
    thumbs_gray: list[NDArray[Any]] = []
    hists: list[NDArray[Any]] = []
    for frame in raw_tail:
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        thumbs_gray.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        hist = cv2.calcHist([small], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hists.append(hist.reshape(-1))

    border_mask = np.ones((90, 160), dtype=bool)
    border_mask[22:68, 32:128] = False  # Exclude central 60%x50% logo zone

    search_lo = max(min_frames, total - max_frames) - offset
    search_hi = (total - min_frames) - offset
    if search_lo > search_hi or search_lo < 1:
        return None

    best_cut: int | None = None
    best_jump = 0.0
    for cut in range(search_lo, search_hi + 1):
        # Require an abrupt visual cut from scene into card
        h_prev, h_curr = hists[cut - 1], hists[cut]
        denom = float(np.linalg.norm(h_prev) * np.linalg.norm(h_curr))
        hist_corr = float(np.dot(h_prev, h_curr) / denom) if denom > 1e-8 else 1.0
        luma_jump = float(
            np.mean(np.abs(thumbs_gray[cut].astype(np.float32) - thumbs_gray[cut - 1].astype(np.float32)))
        )
        if hist_corr >= 0.65 and luma_jump < 26.0:
            continue

        tail = thumbs_gray[cut:]
        outer_stds = [float(np.std(t[border_mask])) for t in tail]
        if float(np.median(outer_stds)) > 18.0:
            continue
        tail_settle = tail[min(2, len(tail) - 1) :]
        if len(tail_settle) >= 2:
            diffs = [
                float(
                    np.mean(
                        np.abs(
                            tail_settle[i][border_mask].astype(np.float32)
                            - tail_settle[i - 1][border_mask].astype(np.float32)
                        )
                    )
                )
                for i in range(1, len(tail_settle))
            ]
            if float(np.median(diffs)) > 2.5:
                continue

        # Ensure pre-cut video was not the exact same flat card from frame 0
        pre = thumbs_gray[max(0, cut - min_frames) : cut]
        pre_outer_std = float(np.median([float(np.std(t[border_mask])) for t in pre]))
        pre_motion = (
            float(
                np.mean(
                    [
                        float(np.mean(np.abs(pre[i].astype(np.float32) - pre[i - 1].astype(np.float32))))
                        for i in range(1, len(pre))
                    ]
                )
            )
            if len(pre) >= 2
            else 0.0
        )
        if pre_outer_std < 18.0 and pre_motion < 1.5 and abs(float(np.mean(pre[-1])) - float(np.mean(tail[0]))) < 20.0:
            continue

        cut_strength = (1.0 - hist_corr) * 50.0 + luma_jump
        if cut_strength > best_jump:
            best_jump = cut_strength
            best_cut = offset + cut
    return best_cut


def _ultra_low_freq_blur(image_f32: NDArray[Any], sigma: float) -> NDArray[Any]:
    """Compute a fast ultra-low-frequency Gaussian blur via 1/4-scale pyramid."""
    height, width = image_f32.shape[:2]
    small_w = max(8, width // 4)
    small_h = max(8, height // 4)
    small = cv2.resize(image_f32, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_sigma = max(1.0, sigma / 4.0)
    blurred_small = cv2.GaussianBlur(small, (0, 0), small_sigma, borderType=cv2.BORDER_REFLECT_101)
    return cv2.resize(blurred_small, (width, height), interpolation=cv2.INTER_LINEAR)


def anchor_low_freq_color(
    source: NDArray[Any],
    candidate: NDArray[Any],
    *,
    sigma: float | None = None,
) -> NDArray[Any]:
    """Anchor only ultra-low-frequency CIE Lab illumination and chroma to ``source``.

    Transfers spatial frequencies below ``sigma >= 32 px`` (~200 px wavelength) so
    VAE white-balance and exposure drift are corrected without leaking mid- or
    high-frequency watermark carriers from ``source``. Supports ``uint8`` and
    ``uint16`` BGR frames.
    """
    height, width = candidate.shape[:2]
    eff_sigma = max(32.0, max(height, width) / 32.0) if sigma is None else max(8.0, float(sigma))
    src_aligned = (
        source
        if source.shape[:2] == (height, width)
        else cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
    )
    src_scale = 65535.0 if src_aligned.dtype == np.uint16 else 255.0
    cand_scale = 65535.0 if candidate.dtype == np.uint16 else 255.0
    src_f32 = np.clip(src_aligned.astype(np.float32) / src_scale, 0.0, 1.0)
    cand_f32 = np.clip(candidate.astype(np.float32) / cand_scale, 0.0, 1.0)

    low_src = _ultra_low_freq_blur(src_f32, eff_sigma)
    low_cand = _ultra_low_freq_blur(cand_f32, eff_sigma)
    src_lab_low = cv2.cvtColor(low_src, cv2.COLOR_BGR2Lab)
    cand_lab_low = cv2.cvtColor(low_cand, cv2.COLOR_BGR2Lab)
    cand_lab = cv2.cvtColor(cand_f32, cv2.COLOR_BGR2Lab)

    delta_lab = src_lab_low - cand_lab_low
    cand_lab[..., 0] = np.clip(cand_lab[..., 0] + delta_lab[..., 0], 0.0, 100.0)
    cand_lab[..., 1] = np.clip(cand_lab[..., 1] + delta_lab[..., 1], -127.0, 127.0)
    cand_lab[..., 2] = np.clip(cand_lab[..., 2] + delta_lab[..., 2], -127.0, 127.0)

    anchored_bgr = np.clip(cv2.cvtColor(cand_lab, cv2.COLOR_Lab2BGR), 0.0, 1.0)
    if candidate.dtype == np.uint16:
        return np.clip(np.rint(anchored_bgr * 65535.0), 0.0, 65535.0).astype(np.uint16)
    return np.clip(np.rint(anchored_bgr * 255.0), 0.0, 255.0).astype(np.uint8)
