"""Opt-in restoration of verified text from a profile VAE reconstruction."""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

from remove_ai_watermarks._internal.schema import require_schema_version

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

TEXT_MANIFEST_SCHEMA = 2
_SUPPORTED_TEXT_MANIFEST_SCHEMAS = frozenset({1, TEXT_MANIFEST_SCHEMA})
GLYPH_FEATHER = 0.5
GLYPH_SIDE_PAD_RATIO = 0.12
GLYPH_SIDE_PAD_LIMIT = 1.0


@dataclass(frozen=True)
class VerifiedTextLine:
    """One operator-verified source region in source-pixel coordinates."""

    box: tuple[int, int, int, int]
    text: str | None = None
    script: str | None = None
    angle: float = 0.0


@dataclass(frozen=True)
class VerifiedTextManifest:
    """Verified text regions cryptographically bound to one decoded RGB source."""

    source_pixel_sha256: str
    width: int
    height: int
    lines: tuple[VerifiedTextLine, ...]


def source_pixel_sha256(image: Image.Image) -> str:
    """Hash decoded RGB geometry and bytes, independent of container metadata."""
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(rgb.width.to_bytes(8, "big"))
    digest.update(rgb.height.to_bytes(8, "big"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def load_verified_text_manifest(path: Path, source: Image.Image) -> VerifiedTextManifest:
    """Load and validate an operator-verified manifest for exactly ``source``."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read text manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Text manifest must be a JSON object")
    schema_version = require_schema_version(
        payload.get("schema_version"),
        contract="text manifest",
        supported=_SUPPORTED_TEXT_MANIFEST_SCHEMAS,
    )
    if payload.get("verified") is not True:
        raise ValueError("Text manifest must contain verified=true after operator verification")

    rgb = source.convert("RGB")
    width = _manifest_integer(payload, "width")
    height = _manifest_integer(payload, "height")
    if (width, height) != rgb.size:
        raise ValueError(f"Text manifest dimensions {width}x{height} do not match source {rgb.width}x{rgb.height}")
    expected_hash = payload.get("source_pixel_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("Text manifest source_pixel_sha256 must be a 64-character SHA-256")
    actual_hash = source_pixel_sha256(rgb)
    if expected_hash.casefold() != actual_hash:
        raise ValueError("Text manifest source_pixel_sha256 does not match the decoded source pixels")

    raw_lines = payload.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        raise ValueError("Text manifest lines must be a non-empty list")
    lines = tuple(_load_line(item, width, height, index, schema_version) for index, item in enumerate(raw_lines))
    if list(lines) != sorted(lines, key=lambda line: (line.box[1], line.box[0])):
        raise ValueError("Text manifest lines must be in top-to-bottom, left-to-right reading order")
    return VerifiedTextManifest(actual_hash, width, height, lines)


def _manifest_integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Text manifest {key} must be a positive integer")
    return value


def _load_line(item: Any, width: int, height: int, index: int, schema_version: int) -> VerifiedTextLine:
    if not isinstance(item, dict):
        raise ValueError(f"Text manifest line {index} must be an object")
    raw_box = item.get("box")
    if (
        not isinstance(raw_box, list)
        or len(raw_box) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_box)
    ):
        raise ValueError(f"Text manifest line {index} box must contain four integers")
    box = tuple(raw_box)
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Text manifest line {index} box is outside the source dimensions")
    text: str | None = None
    script: str | None = None
    if schema_version == 1:
        text = item.get("text")
        script = item.get("script")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Text manifest line {index} text must be non-empty")
        if not isinstance(script, str) or not script.strip():
            raise ValueError(f"Text manifest line {index} script must be non-empty")
    angle_value = item.get("angle", 0.0)
    if isinstance(angle_value, bool) or not isinstance(angle_value, int | float):
        raise ValueError(f"Text manifest line {index} angle must be numeric")
    angle = float(angle_value)
    if not math.isfinite(angle) or abs(angle) > 30.0:
        raise ValueError(f"Text manifest line {index} angle must be between -30 and 30 degrees")
    return VerifiedTextLine(box, text, script, angle)


def restore_verified_text(
    source: Image.Image,
    candidate: Image.Image,
    donor: Image.Image,
    lines: tuple[VerifiedTextLine, ...],
) -> Image.Image:
    """Erase candidate glyphs, then composite verified profile-VAE glyph cores."""
    from remove_ai_watermarks import region_eraser

    if not region_eraser.lama_available():
        raise RuntimeError(
            "Verified text restoration requires LaMa. Install: pip install 'remove-ai-watermarks[text-restoration]'"
        )
    source_rgb = np.asarray(source.convert("RGB"))
    candidate_rgb = np.asarray(candidate.convert("RGB"))
    donor_rgb = np.asarray(donor.convert("RGB"))
    if source_rgb.shape != candidate_rgb.shape or source_rgb.shape != donor_rgb.shape:
        raise ValueError("Source, candidate, and profile-VAE donor dimensions must match")

    source_masks = [source_silhouette_mask(source_rgb, line.box, line.angle) for line in lines]
    for index, mask in enumerate(source_masks):
        if not np.any(mask):
            raise ValueError(f"Verified text line {index} produced no source glyph pixels")
    candidate_masks = [source_silhouette_mask(candidate_rgb, line.box, line.angle) for line in lines]
    erase_masks = []
    for line, source_mask, candidate_mask in zip(lines, source_masks, candidate_masks, strict=True):
        radius = 5 if line.box[3] - line.box[1] >= 48 else 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
        erase_masks.append(cv2.dilate(np.maximum(source_mask, candidate_mask), kernel))
    del candidate_masks

    groups = group_text_lines(lines)
    background = cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2BGR)
    for group in groups:
        background = region_eraser.erase_lama(background, np.maximum.reduce([erase_masks[index] for index in group]))
    background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    residual_masks = [
        residual_glyph_mask(background_rgb, mask, line.box) for line, mask in zip(lines, erase_masks, strict=True)
    ]
    for group in groups:
        residual = np.maximum.reduce([residual_masks[index] for index in group])
        if np.any(residual):
            background = region_eraser.erase_lama(background, residual)
    del erase_masks, residual_masks
    restored = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    restored = composite_fresh_text_edges(source_rgb, restored, lines, source_masks)
    source_glyph_mask = np.maximum.reduce(source_masks)
    restored = composite_reconstructed_glyphs(donor_rgb, restored, source_glyph_mask)
    return Image.fromarray(restored)


def _glyph_crop_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    left_pad: int,
    right_pad: int,
) -> tuple[int, int, int, int]:
    """Widen the detector box so glyph edges stay inside the silhouette crop.

    Detector boxes can stop inside a leading flourish or trailing punctuation,
    and Paddle line boxes sit 2-5 px above the true ink bottom on the poster
    fixtures. The crop is otherwise the box itself, so those pixels never reach
    the donor composite. Start 12% sideways, expand each side independently
    while foreground reaches its boundary, and pad 8% up and 25% down.
    """
    x1, y1, x2, y2 = box
    line_h = max(1, y2 - y1)
    pad_top = max(1, round(line_h * 0.08))
    pad_bot = max(2, round(line_h * 0.25))
    return (
        max(0, x1 - left_pad),
        max(0, y1 - pad_top),
        min(width, x2 + right_pad),
        min(height, y2 + pad_bot),
    )


def _silhouette_crop_mask(
    source_rgb: NDArray[Any],
    crop_box: tuple[int, int, int, int],
    angle: float,
) -> NDArray[Any]:
    """Threshold one candidate crop so its horizontal boundaries can be tested."""
    height, width = source_rgb.shape[:2]
    x1, y1, x2, y2 = crop_box
    gray = cv2.cvtColor(source_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
    support = np.ones(gray.shape, dtype=np.uint8)
    if angle:
        box_width, box_height = x2 - x1, y2 - y1
        theta = math.radians(abs(angle))
        cosine, sine = math.cos(theta), math.sin(theta)
        denominator = cosine * cosine - sine * sine
        rect_width = (box_width * cosine - box_height * sine) / denominator
        rect_height = (box_height * cosine - box_width * sine) / denominator
        rotated = cv2.boxPoints(
            ((box_width / 2, box_height / 2), (max(1.0, rect_width * 0.92), max(1.0, rect_height * 0.62)), -angle)
        )
        support.fill(0)
        cv2.fillConvexPoly(support, np.rint(rotated).astype(np.int32), 1)
        values = gray[support > 0]
        background_luma = float(np.median(values))
    else:
        ring_pad = max(6, min(20, (y2 - y1) // 4))
        rx1, ry1, rx2, ry2 = _clip_box(crop_box, width, height, pad=ring_pad)
        context = cv2.cvtColor(source_rgb[ry1:ry2, rx1:rx2], cv2.COLOR_RGB2GRAY)
        ring = np.ones(context.shape, dtype=bool)
        ring[y1 - ry1 : y2 - ry1, x1 - rx1 : x2 - rx1] = False
        background_luma = float(np.median(context[ring])) if ring.any() else float(np.median(gray))
        values = gray.reshape(-1)
    low, high = float(np.percentile(values, 2)), float(np.percentile(values, 98))
    dark_contrast, light_contrast = background_luma - low, high - background_luma
    threshold = max(16.0, min(56.0, max(light_contrast, dark_contrast) * 0.22))
    if light_contrast > dark_contrast:
        crop_mask = (gray.astype(np.float32) >= background_luma + threshold).astype(np.uint8) * 255
    else:
        crop_mask = (gray.astype(np.float32) <= background_luma - threshold).astype(np.uint8) * 255
    crop_mask[support == 0] = 0
    return crop_mask


def _anchored_components_reach_sides(
    crop_mask: NDArray[Any],
    anchor_box: tuple[int, int, int, int],
) -> tuple[bool, bool]:
    """Whether anchored foreground reaches the left and right crop sides."""
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(crop_mask, 8)
    x1, y1, x2, y2 = anchor_box
    anchor_counts = np.bincount(labels[y1:y2, x1:x2].reshape(-1), minlength=count)
    anchor_counts[0] = 0

    def reaches(edge_labels: NDArray[Any]) -> bool:
        return any(
            label != 0 and anchor_counts[label] >= max(3, round(stats[label, cv2.CC_STAT_AREA] * 0.1))
            for label in np.unique(edge_labels)
        )

    return reaches(labels[y1:y2, :2]), reaches(labels[y1:y2, -2:])


def source_silhouette_mask(
    source_rgb: NDArray[Any],
    box: tuple[int, int, int, int],
    angle: float = 0.0,
) -> NDArray[Any]:
    """Recover a thresholded glyph shape without retaining source amplitudes."""
    height, width = source_rgb.shape[:2]
    box_x1, box_y1, box_x2, box_y2 = box
    line_h = max(1, box_y2 - box_y1)
    step = max(2, round(line_h * GLYPH_SIDE_PAD_RATIO))
    limit = max(step, round(line_h * GLYPH_SIDE_PAD_LIMIT))
    left_pad = right_pad = step
    while True:
        x1, y1, x2, y2 = _glyph_crop_box(box, width, height, left_pad, right_pad)
        crop_mask = _silhouette_crop_mask(source_rgb, (x1, y1, x2, y2), angle)
        core_y1 = max(0, box_y1 - y1)
        core_y2 = min(y2 - y1, box_y2 - y1)
        anchor_box = (max(0, box_x1 - x1), core_y1, min(x2 - x1, box_x2 - x1), core_y2)
        can_expand_left = x1 > 0 and left_pad < limit
        can_expand_right = x2 < width and right_pad < limit
        left_anchored, right_anchored = (
            _anchored_components_reach_sides(crop_mask, anchor_box)
            if can_expand_left or can_expand_right
            else (False, False)
        )
        left_touches = can_expand_left and left_anchored
        right_touches = can_expand_right and right_anchored
        if not left_touches and not right_touches:
            break
        if left_touches:
            left_pad = min(limit, left_pad + step)
        if right_touches:
            right_pad = min(limit, right_pad + step)
    result = np.zeros((height, width), dtype=np.uint8)
    result[y1:y2, x1:x2] = crop_mask
    return result


def residual_glyph_mask(
    background_rgb: NDArray[Any],
    original_mask: NDArray[Any],
    box: tuple[int, int, int, int],
) -> NDArray[Any]:
    """Find glyph-like contrast left after the first inpaint pass."""
    residual = _foreground_mask(background_rgb, box)
    residual = cv2.bitwise_and(residual, original_mask)
    return cv2.dilate(residual, np.ones((5, 5), np.uint8), iterations=1)


def composite_fresh_text_edges(
    source_rgb: NDArray[Any],
    background_rgb: NDArray[Any],
    lines: tuple[VerifiedTextLine, ...],
    masks: list[NDArray[Any]],
) -> NDArray[Any]:
    """Render fresh antialiased edges for source-derived glyph masks."""
    restored = background_rgb
    for line, mask in zip(lines, masks, strict=True):
        color = _sample_text_color(source_rgb, mask, line.box)
        restored = composite_fresh_silhouette(restored, mask, color)
    return restored


def composite_reconstructed_glyphs(
    donor_rgb: NDArray[Any],
    background_rgb: NDArray[Any],
    glyph_mask: NDArray[Any],
    *,
    feather: float = GLYPH_FEATHER,
) -> NDArray[Any]:
    """Composite an exact reconstructed core with a narrow donor edge."""
    if donor_rgb.shape != background_rgb.shape or donor_rgb.shape[:2] != glyph_mask.shape:
        raise ValueError("donor, background, and glyph mask dimensions must match")
    blurred = cv2.GaussianBlur(glyph_mask, (0, 0), feather) if feather > 0 else glyph_mask
    alpha = np.maximum(glyph_mask, blurred).astype(np.float32) / 255.0
    combined = donor_rgb.astype(np.float32) * alpha[..., None] + background_rgb.astype(np.float32) * (
        1.0 - alpha[..., None]
    )
    return np.clip(np.rint(combined), 0, 255).astype(np.uint8)


def composite_fresh_silhouette(
    background_rgb: NDArray[Any],
    glyph_mask: NDArray[Any],
    color: tuple[int, int, int],
    *,
    feather: float = 0.35,
) -> NDArray[Any]:
    """Render a binary source shape with fresh color and antialiasing."""
    if background_rgb.shape[:2] != glyph_mask.shape:
        raise ValueError("background and glyph mask dimensions must match")
    antialiased = cv2.GaussianBlur(glyph_mask, (0, 0), feather) if feather > 0 else glyph_mask
    alpha = antialiased.astype(np.float32)[..., None] / 255.0
    foreground = np.empty_like(background_rgb)
    foreground[:, :] = color
    combined = foreground.astype(np.float32) * alpha + background_rgb.astype(np.float32) * (1.0 - alpha)
    return np.clip(combined, 0, 255).astype(np.uint8)


def _clip_box(box: tuple[int, int, int, int], width: int, height: int, pad: int = 0) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)


def _foreground_mask(source_rgb: NDArray[Any], box: tuple[int, int, int, int]) -> NDArray[Any]:
    height, width = source_rgb.shape[:2]
    line_height = box[3] - box[1]
    x1, y1, x2, y2 = _clip_box(box, width, height, pad=max(6, int(line_height * 0.12)))
    gray = cv2.cvtColor(source_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
    ring_pad = max(8, min(24, (y2 - y1) // 5))
    rx1, ry1, rx2, ry2 = _clip_box((x1, y1, x2, y2), width, height, pad=ring_pad)
    context = cv2.cvtColor(source_rgb[ry1:ry2, rx1:rx2], cv2.COLOR_RGB2GRAY)
    ring = np.ones(context.shape, dtype=bool)
    ring[y1 - ry1 : y2 - ry1, x1 - rx1 : x2 - rx1] = False
    background_luma = float(np.median(context[ring])) if ring.any() else float(np.median(gray))
    low, high = float(np.percentile(gray, 4)), float(np.percentile(gray, 96))
    dark_contrast, light_contrast = background_luma - low, high - background_luma
    threshold = max(24.0, min(72.0, max(light_contrast, dark_contrast) * 0.32))
    if light_contrast > dark_contrast:
        mask = (gray.astype(np.float32) >= background_luma + threshold).astype(np.uint8) * 255
    else:
        mask = (gray.astype(np.float32) <= background_luma - threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    dilation = 5 if line_height >= 48 else 3
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1,) * 2))
    result = np.zeros((height, width), dtype=np.uint8)
    result[y1:y2, x1:x2] = mask
    return result


def _sample_text_color(
    source_rgb: NDArray[Any], mask: NDArray[Any], box: tuple[int, int, int, int]
) -> tuple[int, int, int]:
    height, width = source_rgb.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, width, height, pad=2)
    crop = source_rgb[y1:y2, x1:x2]
    pixels = crop[mask[y1:y2, x1:x2] > 0]
    luma = pixels.mean(axis=1)
    background_luma = float(crop[[0, -1], :, :].reshape(-1, 3).mean(axis=1).mean())
    selected = (
        pixels[luma <= np.percentile(luma, 20)] if background_luma >= 128 else pixels[luma >= np.percentile(luma, 80)]
    )
    return tuple(int(value) for value in np.median(selected, axis=0))


def group_text_lines(lines: Sequence[VerifiedTextLine]) -> list[list[int]]:
    """Group nearby same-script lines for a shared LaMa erase pass."""
    groups: list[list[int]] = []
    for index, line in enumerate(lines):
        if not groups:
            groups.append([index])
            continue
        previous = lines[groups[-1][-1]]
        gap = line.box[1] - previous.box[3]
        if line.script != previous.script or gap > max(60, int((previous.box[3] - previous.box[1]) * 1.1)):
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups
