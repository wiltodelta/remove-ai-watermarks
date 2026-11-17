"""Pure motion-compensated helpers shared by video pipelines."""

from __future__ import annotations

# OpenCV exposes incomplete types for optical-flow and remap operations.
# Public signatures remain annotated while this third-party boundary is relaxed.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingTypeStubs=false, reportCallIssue=false, reportArgumentType=false
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
    """Blend a motion-aligned prior fill when nearby source pixels agree.

    The prior contributes only where its warped removal mask covers the current
    mask. A context ring outside both masks gates the blend, so scene cuts or
    non-rigid local changes keep the independent current-frame fill.
    """
    if not 0.0 <= blend <= 1.0:
        raise ValueError("Temporal blend must be between 0 and 1")
    if max_context_residual <= 0.0:
        raise ValueError("Context residual threshold must be positive")
    if (
        previous_source.shape != current_source.shape
        or previous_cleaned.shape != current_cleaned.shape
        or previous_source.shape != previous_cleaned.shape
        or previous_mask.shape != current_mask.shape
        or previous_mask.shape != current_source.shape[:2]
    ):
        raise ValueError("Temporal fill inputs must share frame and mask geometry")

    union = (previous_mask > 0) | (current_mask > 0)
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

    previous_source_crop = previous_source[y0:y1, x0:x1]
    current_source_crop = current_source[y0:y1, x0:x1]
    current_cleaned_crop = current_cleaned[y0:y1, x0:x1]
    previous_cleaned_crop = previous_cleaned[y0:y1, x0:x1]
    previous_mask_crop = previous_mask[y0:y1, x0:x1]
    current_mask_crop = current_mask[y0:y1, x0:x1]
    maps = _backward_map(
        cv2.cvtColor(current_source_crop, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(previous_source_crop, cv2.COLOR_BGR2GRAY),
    )
    warped_previous_source = _backward_warp(previous_source_crop, maps)
    warped_previous_cleaned = _backward_warp(previous_cleaned_crop, maps)
    warped_previous_mask = _backward_warp(
        previous_mask_crop,
        maps,
        interpolation=cv2.INTER_NEAREST,
    )

    current_hole = current_mask_crop > 0
    if not np.any(current_hole):
        return current_cleaned
    covered = current_hole & (warped_previous_mask > 0)
    if float(np.mean(covered[current_hole])) < 0.85:
        return current_cleaned

    occupied = current_hole | (warped_previous_mask > 0)
    dilation = max(7, round(max(mask_width, mask_height) * 0.25))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation | 1, dilation | 1))
    context = cv2.dilate(occupied.astype(np.uint8), kernel).astype(bool) & ~occupied
    if np.count_nonzero(context) < 64:
        return current_cleaned
    residual = np.abs(current_source_crop.astype(np.float32) - warped_previous_source.astype(np.float32))
    context_residual = float(np.mean(residual[context]))
    if context_residual > max_context_residual:
        return current_cleaned

    effective_blend = blend * (1.0 - context_residual / max_context_residual)
    blended = (1.0 - effective_blend) * current_cleaned_crop[covered].astype(
        np.float32
    ) + effective_blend * warped_previous_cleaned[covered].astype(np.float32)
    result = current_cleaned.copy() if copy else current_cleaned
    result_crop = result[y0:y1, x0:x1]
    result_crop[covered] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


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
        raise ValueError("Stabilization expects 3-channel uint8 video frames")
    if source_frame.shape != shape or regenerated_frame.shape != shape:
        raise ValueError("All source and regenerated frames must share identical geometry")
    if source_frame.dtype != np.uint8 or regenerated_frame.dtype != np.uint8:
        raise ValueError("Stabilization expects uint8 video frames")
    return (
        shape,
        cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY),
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
) -> NDArray[Any]:
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
            photometric_tolerance=photometric_tolerance,
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
            photometric_tolerance=photometric_tolerance,
            min_valid_fraction=min_valid_fraction,
        )
        if w_next is not None and warp_next is not None:
            accum += w_next[:, :, None] * (warp_next - curr_reg_f32)
    return np.clip(np.round(accum), 0, 255).astype(np.uint8)


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
