"""Jimeng-basic ``AI生成`` pill visible mark (issue #54).

The Jimeng free-tier TC260 label is a rounded pill with 'AI生成' in the TOP-LEFT
corner -- distinct from the ``jimeng`` "★ 即梦AI" wordmark (bottom-right). It has no
captured alpha map for its outline. Detection therefore has two paths before the
shared localize -> fill step:

  * A high-confidence contrast template for the measured solid ``AI生成`` label
    (``assets/jimeng_label_alpha.png``). With Jimeng provenance this verified
    label can bypass the flat-background safety gate.
  * Edge-NCC of a font-rendered SILHOUETTE (``assets/jimeng_pill.png``,
    synthetic, data-safe -- see ``scripts/render_pill_silhouette.py``) against the
    top-left ROI, at the pill's known width fraction. Its calibrated
    ``_DETECT_THRESHOLD`` is 0.22 and it retains the provenance and flat-background
    product gates.
  * Remove: mask the verified label match or the stable outline footprint and
    inpaint it (MI-GAN / cv2 via the registry).

Geometry uses width ~0.161*W,
height ~0.091*W, top-left, margins ~0.02-0.05.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import cv2
import numpy as np

from remove_ai_watermarks import image_io
from remove_ai_watermarks._text_mark_engine import TextMarkConfig, TextMarkDetection, TextMarkEngine

if TYPE_CHECKING:
    from numpy.typing import NDArray

# cv2/numpy boundary: cv2 ships no usable type info, so strict pyright cannot know
# its array element types. Relax the unknown-type rules for this file only; the
# public signatures are still annotated with NDArray[Any].
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportAttributeAccessIssue=false, reportUnnecessaryComparison=false

_ASSET = Path(__file__).parent / "assets" / "jimeng_pill.png"

# Geometry (fractions of image WIDTH unless noted); top-left corner.
_WIDTH_FRAC = 0.161
_ROI_W_FRAC = 0.34  # search window width (of W)
_ROI_H_FRAC = 0.14  # search window height (of H)
_DETECT_THRESHOLD = 0.22  # calibrated edge-NCC gate
_PROVENANCE_DETECT_THRESHOLD = 0.195  # corroborated gate (gated by _keep_pill wordmark / flat-TC260 arms)
# Inpaint mask GEOMETRY (fractions of W unless noted): a generous fixed top-left box
# covering the pill (measured ~0.167*W wide, ~0.09*W tall, margin ~0.02-0.05) plus
# margin. The mask uses stable geometry, NOT the NCC match position -- the synthetic
# silhouette localizes only approximately, and the corner is negative space, so
# over-covering is harmless while a match-positioned box leaves outline residue.
_MASK_X0, _MASK_Y0 = 0.012, 0.006  # x0 of W, y0 of H
_MASK_W, _MASK_H = 0.205, 0.115  # width of W, height of W

# Background-flatness gate for the metadata-only pill arm (see remove_auto_marks).
# The pill detector is weak (~7% raw false-fire); metadata confirms the platform,
# not pill presence, so its false fires are real Jimeng-class content WITHOUT a pill.
# Those false fires cluster on TEXTURED top-left corners (ceiling fixtures, structure)
# where inpaint visibly SMEARS, while real pills and harmless false fires sit on FLAT
# corners (sky / wall / solid) where inpaint is invisible. So the metadata-only arm
# removes the pill only when the footprint background is flat enough for a safe,
# invisible inpaint. Threshold = median Sobel magnitude over the footprint box at a
# normalized width. The reliable bottom-right wordmark arm is NOT texture-gated:
# a wordmark-confirmed pill is removed regardless.
#
# Measured through the PRODUCT path (the `_keep_pill` gate), not the raw detector, by
# ``scripts/pill_gate_audit.py`` -- the raw path bypasses the gate and reads as a
# disaster that the shipped behavior does not have. Re-run it when the gate changes.
_FLAT_TEXTURE_MAX = 6.0

_silhouette: NDArray[Any] | None = None


class PillDetection(NamedTuple):
    detected: bool
    confidence: float
    region: tuple[int, int, int, int]  # x, y, w, h of the matched pill
    verified_label: bool = False


_LABEL_CONFIG = TextMarkConfig(
    name="Jimeng AI生成 pill label",
    asset_name="jimeng_label_alpha.png",
    corner="tl",
    margin_floor=2,
    width_frac=0.34,
    height_frac=0.14,
    margin_x_frac=0.0,
    margin_bottom_frac=0.0,
    max_saturation=255,
    logo_min_luma=100,
    tophat_delta=8,
    morph_open_size=3,
    detect_min_coverage=0.01,
    detect_ncc_threshold=0.55,
    detect_frontend="contrast",
    scale_basis="width",
    ladder=(0.95, 1.0, 1.05),
    alpha_width_frac=0.14,
    alpha_height_frac=0.14 * 128 / 353,
    min_gw=24,
    provenance_ncc_factor=1.0,
)


def _load_silhouette() -> NDArray[Any] | None:
    global _silhouette
    if _silhouette is None:
        if not _ASSET.exists():
            return None
        _silhouette = image_io.imread(str(_ASSET), cv2.IMREAD_GRAYSCALE)
    return _silhouette


def _grad(gray: NDArray[Any]) -> NDArray[Any]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX)


class PillEngine:
    """Detect the top-left ``AI生成`` label or its synthetic pill outline."""

    def __init__(self, *, include_label: bool = True) -> None:
        self._label = TextMarkEngine(_LABEL_CONFIG) if include_label else None

    @staticmethod
    def _label_box(detection: TextMarkDetection) -> tuple[int, int, int, int] | None:
        if not detection.detected or detection.match_box is None:
            return None
        x, y, _width, _height = detection.region
        x0, y0, x1, y1 = detection.match_box
        return x + x0, y + y0, x1 - x0 + 1, y1 - y0 + 1

    def _match(self, image: NDArray[Any]) -> tuple[float, tuple[int, int, int, int]] | None:
        sil = _load_silhouette()
        if sil is None or image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        if h < 64 or w < 64:
            return None
        gray = cv2.cvtColor(image_io.to_bgr(image), cv2.COLOR_BGR2GRAY)
        rh, rw = int(h * _ROI_H_FRAC), int(w * _ROI_W_FRAC)
        roi = gray[0:rh, 0:rw]
        tw = max(24, int(_WIDTH_FRAC * w))
        th = max(12, int(tw * sil.shape[0] / sil.shape[1]))
        if th >= rh or tw >= rw:
            return None
        tmpl = cv2.resize(sil, (tw, th))
        res = cv2.matchTemplate(_grad(roi.astype(np.float32)), _grad(tmpl.astype(np.float32)), cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        return float(score), (int(loc[0]), int(loc[1]), tw, th)

    def _detection(
        self,
        match: tuple[float, tuple[int, int, int, int]] | None,
        label: TextMarkDetection,
        *,
        provenance: bool = False,
    ) -> PillDetection:
        label_box = self._label_box(label)
        if match is None:
            return PillDetection(label.detected, label.confidence, label_box or label.region, label.detected)
        score, box = match
        threshold = _PROVENANCE_DETECT_THRESHOLD if provenance else _DETECT_THRESHOLD
        return PillDetection(
            score >= threshold or label.detected,
            max(score, label.confidence),
            label_box or box,
            label.detected,
        )

    def detect(self, image: NDArray[Any], *, provenance: bool = False) -> PillDetection:
        """Detect the outline, plus the solid label when Jimeng is corroborated."""
        label = self._label.detect(image) if provenance and self._label is not None else TextMarkDetection()
        return self._detection(self._match(image), label, provenance=provenance)

    def detect_both(self, image: NDArray[Any]) -> tuple[PillDetection, PillDetection]:
        """Return strict outline and Jimeng-corroborated label verdicts from one scan."""
        match = self._match(image)
        label = self._label.detect(image) if self._label is not None else TextMarkDetection()
        return (
            self._detection(match, TextMarkDetection(), provenance=False),
            self._detection(match, label, provenance=True),
        )

    def _footprint_box(self, image: NDArray[Any]) -> tuple[int, int, int, int] | None:
        h, w = image.shape[:2]
        x0, y0 = int(_MASK_X0 * w), int(_MASK_Y0 * h)
        x1, y1 = min(w, x0 + int(_MASK_W * w)), min(h, y0 + int(_MASK_H * w))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def footprint_texture(self, image: NDArray[Any]) -> float:
        """Median gradient magnitude over the fixed top-left footprint box at a
        normalized width. A robust flatness proxy: low = flat (sky / wall / solid,
        inpaint invisible), high = textured (ceiling fixtures / structure, inpaint
        smears). Median (not mean) so the pill's own edges -- a minority of the box --
        do not inflate it. Backs the metadata-only arm's safe-inpaint gate."""
        if image is None or image.size == 0:
            return 0.0
        box = self._footprint_box(image)
        if box is None:
            return 0.0
        x0, y0, x1, y1 = box
        crop = image[y0:y1, x0:x1]
        gray = cv2.cvtColor(image_io.to_bgr(crop), cv2.COLOR_BGR2GRAY)
        tw = 220
        gray = cv2.resize(gray, (tw, max(1, int(gray.shape[0] * tw / gray.shape[1])))).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.median(cv2.magnitude(gx, gy)))

    def footprint_is_flat(self, image: NDArray[Any], *, thresh: float = _FLAT_TEXTURE_MAX) -> bool:
        """True when the top-left footprint is flat enough for an invisible inpaint."""
        return self.footprint_texture(image) <= thresh

    def footprint_mask(
        self,
        image: NDArray[Any],
        *,
        force: bool = False,
        detection: PillDetection | None = None,
    ) -> NDArray[Any] | None:
        """Full-frame uint8 mask (255 = pill) over the pill's known top-left region.

        A verified label uses its detector-aligned box with measured padding. The
        weaker synthetic outline uses stable geometry because its NCC position is
        only approximate and a match-positioned mask leaves rim residue. The caller
        gates on :meth:`detect`, so a clean corner is never masked. ``force`` is
        accepted for a uniform engine signature but ignored."""
        if image is None or image.size == 0:
            return None
        if detection is not None and detection.verified_label:
            x, y, width, height = detection.region
            pad_x = int(image.shape[1] * 0.025)
            pad_y = int(image.shape[1] * 0.014)
            box = (
                max(0, x - pad_x),
                max(0, y - pad_y),
                min(image.shape[1], x + width + pad_x),
                min(image.shape[0], y + height + pad_y),
            )
        else:
            box = self._footprint_box(image)
        if box is None:
            return None
        # Same primitive the shared fill uses, rather than a private zeros/fill copy.
        # `dilate=0` because this footprint is already generous by construction; the
        # box is clamped to the frame in _footprint_box and both origins are positive
        # fractions, so boxes_to_mask's own clamping is a no-op here.
        from remove_ai_watermarks import region_eraser

        x0, y0, x1, y1 = box
        return region_eraser.boxes_to_mask(image.shape[:2], [(x0, y0, x1 - x0, y1 - y0)], dilate=0)
