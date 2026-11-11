"""Neutral text-shape reconstruction from regenerated image colors.

Geometry is not verification. Callers own the line selection and independent
output validation; no detector or provider service runs in this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

Box = tuple[int, int, int, int]
Pixels = NDArray[np.uint8]


class ShapeGroup(TypedDict):
    roi: Box
    boxes: list[Box]
    accepted: NotRequired[bool]
    each_line_nonregression: NotRequired[list[bool]]
    full_support_nonregression: NotRequired[bool]


class ShapeDiagnostic(TypedDict):
    box: Box
    accepted: bool
    reason: NotRequired[str]
    color_eligible: NotRequired[bool]
    background: NotRequired[list[float]]
    foreground: NotRequired[list[float]]
    border_p80: NotRequired[float]
    residual_p95: NotRequired[float]
    support_pixels: NotRequired[int]
    quality_group: NotRequired[ShapeGroup]
    kernel_size: NotRequired[int]
    donor_foreground: NotRequired[list[float]]
    texture_attempt: NotRequired[ShapeDiagnostic]
    combined_quality_group: NotRequired[ShapeGroup]


def _contour_coverage(coverage: NDArray[Any]) -> NDArray[Any]:
    smooth = cv2.GaussianBlur(coverage, (3, 3), 0.35)
    high = cv2.resize(smooth, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    return cv2.resize(
        (high >= 0.5).astype(np.float32), (coverage.shape[1], coverage.shape[0]), interpolation=cv2.INTER_AREA
    )


def _restore_glyphs(
    source: NDArray[Any],
    candidate: NDArray[Any],
    boxes: Sequence[tuple[int, int, int, int]],
    coverage_builder: Callable[[NDArray[Any]], NDArray[Any]],
    *,
    allow_colored: bool = False,
) -> tuple[NDArray[Any], NDArray[Any], list[ShapeDiagnostic]]:
    if (
        source.ndim != 3
        or source.shape[-1] != 3
        or source.shape != candidate.shape
        or source.dtype != np.uint8
        or candidate.dtype != np.uint8
    ):
        raise ValueError("Expected matching uint8 RGB inputs")
    height, width = source.shape[:2]
    erase = np.zeros((height, width), np.uint8)
    alpha: NDArray[np.float32] | None = None
    color: NDArray[np.float32] | None = None
    diagnostics: list[ShapeDiagnostic] = []
    for original in boxes:
        x1, y1, x2, y2 = map(int, original)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Text box outside image")
        d: ShapeDiagnostic = {"box": original, "accepted": False}
        diagnostics.append(d)
        if y2 - y1 > 256:
            d["reason"] = "large line"
            continue
        x1, y1, x2, y2 = _padded_box((x1, y1, x2, y2), width, height)
        crop = source[y1:y2, x1:x2].astype(np.float32)
        border = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]))
        bg = np.median(border, axis=0)
        distance = np.linalg.norm(crop - bg, axis=2)
        far = distance >= max(80, cast("float", np.percentile(distance, 99)))
        if np.count_nonzero(far) < 2:
            d["reason"] = "insufficient contrast"
            continue
        fg = np.median(crop[far], axis=0)
        vector = fg - bg
        norm = float(vector @ vector)
        if (not allow_colored and np.ptp(fg) > 48) or norm < 80**2:
            d["reason"] = "colored or low contrast glyphs"
            d["color_eligible"] = not allow_colored and norm >= 80**2
            continue
        coverage = np.clip(((crop - bg) @ vector) / norm, 0, 1)
        residual = np.linalg.norm(crop - (bg + coverage[..., None] * vector), axis=2)
        spread = cast("float", np.percentile(np.linalg.norm(border - bg, axis=1), 80))
        error = cast("float", np.percentile(residual, 95))
        d.update(background=bg.tolist(), foreground=fg.tolist(), border_p80=float(spread), residual_p95=float(error))
        if spread > 20 or error > 32:
            d["reason"] = "complex background"
            continue
        coverage = coverage_builder(coverage)
        current = candidate[y1:y2, x1:x2].astype(np.float32)
        old_bg = np.median(np.concatenate((current[0], current[-1], current[:, 0], current[:, -1])), axis=0)
        old_vector = fg - old_bg
        old_norm = float(old_vector @ old_vector)
        if old_norm == 0:
            d["reason"] = "candidate contrast unavailable"
            continue
        old_alpha = np.clip(((current - old_bg) @ old_vector) / old_norm, 0, 1)
        old_peak = cast("float", np.percentile(old_alpha, 99))
        if old_peak <= 0.125:
            d["reason"] = "candidate contrast unavailable"
            continue
        candidate_far = old_alpha >= old_peak
        fg = np.median(current[candidate_far], axis=0)
        support = ((coverage > 0) | (old_alpha > 0.125)).astype(np.uint8) * 255
        support = np.asarray(cv2.dilate(support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))), dtype=np.uint8)
        erase[y1:y2, x1:x2] |= support
        if alpha is None or color is None:
            alpha = np.zeros((height, width), np.float32)
            color = np.zeros(source.shape, np.float32)
        old = alpha[y1:y2, x1:x2]
        take = coverage > old
        color[y1:y2, x1:x2][take] = fg
        old[take] = coverage[take]
        d.update(accepted=True, reason="neutral low complexity line", support_pixels=int(np.count_nonzero(support)))
    if alpha is None or color is None or not np.any(erase):
        return candidate.copy(), erase, diagnostics
    filled = cv2.inpaint(candidate, erase, 3, cv2.INPAINT_TELEA)
    rendered = (
        np.rint(filled.astype(np.float32) * (1 - alpha[..., None]) + color * alpha[..., None])
        .clip(0, 255)
        .astype(np.uint8)
    )
    out = candidate.copy()
    out[erase > 0] = rendered[erase > 0]
    return out, erase, diagnostics


def restore_contour_glyphs(
    source: NDArray[Any],
    candidate: NDArray[Any],
    boxes: tuple[tuple[int, int, int, int], ...],
    *,
    protected_boxes: tuple[tuple[int, int, int, int], ...] = (),
    allow_colored: bool = False,
) -> tuple[NDArray[Any], NDArray[Any], list[ShapeDiagnostic]]:
    """Use subpixel silhouettes, face exclusion and a grouped source-error guard.

    Skip an entire padded line if it intersects a protected face. Keep overlapping
    line groups together; accept their proposed pixels only if every eligible
    line and the complete erase support have nonincreasing squared source error.
    This guard measures fidelity, not watermark removal or semantic OCR accuracy.
    """
    height, width = source.shape[:2]
    for box in (*boxes, *protected_boxes):
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Text or protected box outside image")
    selected: list[Box] = []
    excluded: list[ShapeDiagnostic] = []
    for box in boxes:
        x1, y1, x2, y2 = _padded_box(box, width, height)
        intersects = any(max(x1, a) < min(x2, c) and max(y1, b) < min(y2, d) for a, b, c, d in protected_boxes)
        if intersects:
            excluded.append({"box": box, "accepted": False, "reason": "protected region intersection"})
        else:
            selected.append(box)
    proposal, mask, diagnostics = _restore_glyphs(
        source, candidate, selected, _contour_coverage, allow_colored=allow_colored
    )
    output, guarded_mask, groups = _guard_glyph_groups(source, candidate, proposal, mask, diagnostics)
    _annotate_quality_groups(diagnostics, groups)
    return output, guarded_mask, diagnostics + excluded


def _padded_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return max(0, x1 - 4), max(0, y1 - 4), min(width, x2 + 4), min(height, y2 + 4)


def _annotate_quality_groups(diagnostics: list[ShapeDiagnostic], groups: list[ShapeGroup]) -> None:
    for diagnostic in diagnostics:
        if not diagnostic["accepted"]:
            continue
        group = next(group for group in groups if diagnostic["box"] in group["boxes"])
        diagnostic["quality_group"] = group
        if not group.get("accepted", False):
            diagnostic.update(accepted=False, reason="group fidelity regression")


def _guard_glyph_groups(
    source: NDArray[Any],
    candidate: NDArray[Any],
    proposal: NDArray[Any],
    mask: NDArray[Any],
    diagnostics: list[ShapeDiagnostic],
) -> tuple[NDArray[Any], NDArray[Any], list[ShapeGroup]]:
    height, width = source.shape[:2]
    groups: list[ShapeGroup] = [
        {"roi": _padded_box(d["box"], width, height), "boxes": [d["box"]]} for d in diagnostics if d["accepted"]
    ]
    if not groups:
        return proposal, mask, groups
    output, guarded_mask = candidate.copy(), np.zeros_like(mask)
    merging = True
    while merging:
        merging = False
        for i in range(len(groups)):
            a = groups[i]["roi"]
            for j in range(i + 1, len(groups)):
                b = groups[j]["roi"]
                if max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3]):
                    groups[i] = {
                        "roi": (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])),
                        "boxes": groups[i]["boxes"] + groups[j]["boxes"],
                    }
                    groups.pop(j)
                    merging = True
                    break
            if merging:
                break
    for group in groups:
        x1, y1, x2, y2 = group["roi"]
        roi = np.s_[y1:y2, x1:x2]
        before = (source[roi].astype(np.float64) - candidate[roi].astype(np.float64)) ** 2
        after = (source[roi].astype(np.float64) - proposal[roi].astype(np.float64)) ** 2
        local = mask[roi] > 0
        line_wins: list[bool] = []
        for a, b, c, d in group["boxes"]:
            line_roi = np.s_[b - y1 : d - y1, a - x1 : c - x1]
            line_wins.append(bool(np.sum(after[line_roi]) <= np.sum(before[line_roi])))
        full_win = bool(np.sum(after[local]) <= np.sum(before[local]))
        group.update(
            accepted=full_win and all(line_wins), each_line_nonregression=line_wins, full_support_nonregression=full_win
        )
        if group.get("accepted", False):
            output[roi][local] = proposal[roi][local]
            guarded_mask[roi][local] = mask[roi][local]
    return output, guarded_mask, groups


def _restore_textured_glyphs(
    source: NDArray[Any],
    candidate: NDArray[Any],
    boxes: tuple[Box, ...],
    *,
    protected_boxes: tuple[Box, ...] = (),
) -> tuple[NDArray[Any], NDArray[Any], list[ShapeDiagnostic]]:
    """Propose neutral contours using spatial backgrounds and regenerated colors."""
    if (
        source.shape != candidate.shape
        or source.ndim != 3
        or source.shape[-1] != 3
        or source.dtype != np.uint8
        or candidate.dtype != np.uint8
    ):
        raise ValueError("Expected matching uint8 RGB inputs")
    height, width = source.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    alpha: NDArray[np.float32] | None = None
    color: NDArray[np.float32] | None = None
    diagnostics: list[ShapeDiagnostic] = []
    for box in boxes:
        a, b, c, d = box
        if not 0 <= a < c <= width or not 0 <= b < d <= height:
            raise ValueError("Text box outside image")
        row: ShapeDiagnostic = {"box": box, "accepted": False}
        diagnostics.append(row)
        x1, y1, x2, y2 = _padded_box(box, width, height)
        if any(
            max(x1, left) < min(x2, right) and max(y1, top) < min(y2, bottom)
            for left, top, right, bottom in protected_boxes
        ):
            row["reason"] = "protected region intersection"
            continue
        if d - b > 256:
            row["reason"] = "large line"
            continue
        crop = source[y1:y2, x1:x2].astype(np.float32)
        border = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]))
        bg = np.median(border, axis=0)
        distance = np.linalg.norm(crop - bg, axis=2)
        far = distance >= max(80, cast("float", np.percentile(distance, 99)))
        if np.count_nonzero(far) < 2:
            row["reason"] = "insufficient contrast"
            continue
        fg = np.median(crop[far], axis=0)
        vector = fg - bg
        norm = float(vector @ vector)
        if np.ptp(fg) > 48 or norm < 80**2:
            row["reason"] = "colored or low contrast glyphs"
            continue
        flat = np.clip(((crop - bg) @ vector) / norm, 0, 1)
        residual = np.linalg.norm(crop - (bg + flat[..., None] * vector), axis=2)
        spread = cast("float", np.percentile(np.linalg.norm(border - bg, axis=1), 80))
        if spread <= 20 and cast("float", np.percentile(residual, 95)) <= 32:
            row["reason"] = "handled by flat-background method"
            continue
        kernel_size = max(7, int((d - b) / 3) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        operation = cv2.MORPH_OPEN if np.sum(vector) > 0 else cv2.MORPH_CLOSE
        background = cv2.morphologyEx(crop, operation, kernel)
        vectors = fg - background
        norms = np.sum(vectors * vectors, axis=2)
        coverage = np.clip(np.sum((crop - background) * vectors, axis=2) / np.maximum(norms, 1), 0, 1)
        coverage[norms < 80**2] = 0
        neutral = np.ptp(crop, axis=2) <= 48
        extreme = (
            np.min(crop, axis=2) >= np.min(fg) - 80 if np.sum(vector) > 0 else np.max(crop, axis=2) <= np.max(fg) + 80
        )
        coverage[~(neutral & extreme)] = 0
        coverage = _contour_coverage(coverage)
        current = candidate[y1:y2, x1:x2].astype(np.float32)
        old_bg = cv2.morphologyEx(current, operation, kernel)
        old_vectors = fg - old_bg
        old_norms = np.sum(old_vectors * old_vectors, axis=2)
        old_alpha = np.clip(np.sum((current - old_bg) * old_vectors, axis=2) / np.maximum(old_norms, 1), 0, 1)
        old_alpha[old_norms < 80**2] = 0
        neutral = np.ptp(current, axis=2) <= 48
        extreme = (
            np.min(current, axis=2) >= np.min(fg) - 80
            if np.sum(vector) > 0
            else np.max(current, axis=2) <= np.max(fg) + 80
        )
        old_alpha[~(neutral & extreme)] = 0
        peak = cast("float", np.percentile(old_alpha, 99))
        if peak < 0.5:
            row["reason"] = "candidate contrast unavailable"
            continue
        donor_fg = np.median(current[old_alpha >= peak], axis=0)
        if np.ptp(donor_fg) > 48:
            row["reason"] = "colored candidate foreground"
            continue
        support = ((coverage > 0) | (old_alpha > 0.5)).astype(np.uint8) * 255
        support = np.asarray(
            cv2.dilate(support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, int((d - b) / 6) | 1),) * 2)),
            dtype=np.uint8,
        )
        mask[y1:y2, x1:x2] |= support
        if alpha is None or color is None:
            alpha = np.zeros((height, width), np.float32)
            color = np.zeros(source.shape, np.float32)
        existing = alpha[y1:y2, x1:x2]
        take = coverage > existing
        color[y1:y2, x1:x2][take] = donor_fg
        existing[take] = coverage[take]
        row.update(
            accepted=True,
            reason="spatial background contour",
            kernel_size=kernel_size,
            support_pixels=int(np.count_nonzero(support)),
            donor_foreground=donor_fg.tolist(),
        )
    if alpha is None or color is None or not np.any(mask):
        return candidate.copy(), mask, diagnostics
    filled = cv2.inpaint(candidate, mask, 3, cv2.INPAINT_TELEA)
    proposal = (
        np.rint(filled.astype(np.float32) * (1 - alpha[..., None]) + color * alpha[..., None])
        .clip(0, 255)
        .astype(np.uint8)
    )
    output, guarded, groups = _guard_glyph_groups(source, candidate, proposal, mask, diagnostics)
    _annotate_quality_groups(diagnostics, groups)
    return output, guarded, diagnostics


def restore_textured_glyphs(
    source: NDArray[Any],
    candidate: NDArray[Any],
    boxes: tuple[Box, ...],
    *,
    protected_boxes: tuple[Box, ...] = (),
) -> tuple[NDArray[Any], NDArray[Any], list[ShapeDiagnostic]]:
    """Follow flat-background contours with a guarded spatial-background stage.

    Colors and inpainted backgrounds come from the regenerated image. Source
    geometry remains unverified until each exact output has a provider verdict.
    The final guard also protects eligible flat-background lines from overlapping
    texture edits. This method is an explicit research option, not a default.
    """
    flat, flat_mask, flat_diagnostics = restore_contour_glyphs(
        source, candidate, boxes, protected_boxes=protected_boxes
    )
    textured, texture_mask, texture_diagnostics = _restore_textured_glyphs(
        source, flat, boxes, protected_boxes=protected_boxes
    )
    flat_by_box = {tuple(d["box"]): d for d in flat_diagnostics}
    texture_by_box = {tuple(d["box"]): d for d in texture_diagnostics}
    eligible: list[ShapeDiagnostic] = [
        {"box": box, "accepted": True}
        for box in boxes
        if flat_by_box[box]["accepted"] or texture_by_box[box]["accepted"]
    ]
    output, accepted_texture_mask, groups = _guard_glyph_groups(source, flat, textured, texture_mask, eligible)
    diagnostics: list[ShapeDiagnostic] = []
    for box in boxes:
        flat_diagnostic, texture_diagnostic = flat_by_box[box], texture_by_box[box]
        if texture_diagnostic["accepted"]:
            row = texture_diagnostic.copy()
            group = next(g for g in groups if box in g["boxes"])
            row["combined_quality_group"] = group
            if not group.get("accepted", False):
                row.update(accepted=False, reason="combined fidelity regression")
        else:
            row = flat_diagnostic.copy()
            row["texture_attempt"] = texture_diagnostic
        diagnostics.append(row)
    return output, flat_mask | accepted_texture_mask, diagnostics
