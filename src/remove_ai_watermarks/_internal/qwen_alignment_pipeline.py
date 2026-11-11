"""Qwen-2.1 fidelity profiles with local quality acceptance and baseline fallback.

The strict research class accepts a caller-owned perceptual metric. The runtime
class supplies the fixed local metric and preserves uncalibrated legacy paths.
No provenance service runs in this module.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict, cast

from PIL import Image

from remove_ai_watermarks._internal.perceptual_metric import LocalPerceptualMetric
from remove_ai_watermarks._internal.qwen_zimage_pipeline import (
    qwen21_load_global,
    qwen21_run_global,
    qwen21_vae_roundtrip,
    qwen21_vram_config,
)
from remove_ai_watermarks._internal.two_stage_pipeline import DiffSynthVramConfig, TwoStageZImagePipeline
from remove_ai_watermarks._internal.watermark_profiles import fidelity_skip_reason

if TYPE_CHECKING:
    from remove_ai_watermarks._internal.text_restoration import VerifiedTextManifest
    from remove_ai_watermarks._internal.text_shapes import ShapeDiagnostic

log = logging.getLogger(__name__)
PerceptualMetric = Callable[[Image.Image, Image.Image], float]
Box = tuple[int, int, int, int]


class QualityDecision(TypedDict):
    baseline_mse: float
    candidate_mse: float
    baseline_perceptual: float | None
    candidate_perceptual: float | None
    selected: bool
    reason: str


@dataclass
class Qwen21AlignmentZImagePipeline(TwoStageZImagePipeline):
    """Finish the shared Qwen-2.1 recipe, then evaluate a bounded spatial warp.

    A candidate must reduce both full-frame RGB MSE and the supplied perceptual
    distance. Otherwise the exact baseline image object is returned. Supplied
    face boxes remain baseline-exact. This is a quality policy, not proof of
    watermark removal or of improvement at every image location.
    """

    profile_name: ClassVar[str] = "qwen21-alignment-zimage"
    perceptual_metric: PerceptualMetric = field(kw_only=True, repr=False)

    line_detector: Callable[[Image.Image], Sequence[Box]] | None = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not callable(self.perceptual_metric):
            raise TypeError("A local perceptual metric callback is required")
        self._qwen21_pipe: Any = None
        self.last_quality: QualityDecision | None = None
        self.last_text_quality: dict[str, Any] | None = None
        self.last_color_quality: dict[str, Any] | None = None

    def _qwen_vram_config(self) -> DiffSynthVramConfig:
        return qwen21_vram_config(self)

    def _load_global(self) -> Any:
        return qwen21_load_global(self)

    def _run_global(self, image: Image.Image, strength: float, seed: int | None) -> Image.Image:
        return qwen21_run_global(self, image, strength, seed)

    def _vae_roundtrip(self, image: Image.Image) -> Image.Image:
        return qwen21_vae_roundtrip(self, image)

    def _prepare_regeneration(
        self, image: Image.Image, *, tile: bool, text_manifest: VerifiedTextManifest | None
    ) -> tuple[Box, ...]:
        self.last_quality = None
        self.last_text_quality = None
        self.last_color_quality = None
        if tile or text_manifest is not None:
            raise ValueError("The experimental alignment profile is not calibrated with tiling or verified text")
        if min(image.size) < 64:
            raise ValueError("Alignment requires image sides of at least 64 pixels")
        if self.line_detector is None:
            return ()
        return tuple(self._freeze_box(box, image.size) for box in self.line_detector(image.copy()))

    @staticmethod
    def _freeze_box(box: Sequence[int], size: tuple[int, int]) -> Box:
        if len(box) != 4 or any(isinstance(value, bool) or not isinstance(value, Integral) for value in box):
            raise ValueError("Text boxes require four integer coordinates")
        x1, y1, x2, y2 = map(int, box)
        if not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
            raise ValueError("Text box outside image")
        return x1, y1, x2, y2

    @staticmethod
    def _metric_value(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None

    def _finish_alignment(
        self,
        image: Image.Image,
        result: Image.Image,
        boxes: list[tuple[int, int, int, int]],
        *,
        strength: float,
        seed: int | None,
        context: Any,
    ) -> Image.Image:
        import numpy as np

        from remove_ai_watermarks._internal.image_alignment import (
            align_regenerated,
            attenuate_regenerated,
            local_error_weights,
        )

        self._progress("Evaluating bounded regenerated-image alignment...")
        source = np.asarray(image.convert("RGB"))
        baseline = np.asarray(result.convert("RGB"))
        aligned, flow = align_regenerated(source, baseline, tuple(boxes), resampling="lanczos")
        weights = local_error_weights(source, baseline, aligned)
        candidate, _ = attenuate_regenerated(baseline, flow, weights)
        for x1, y1, x2, y2 in boxes:
            if not np.array_equal(candidate[y1:y2, x1:x2], baseline[y1:y2, x1:x2]):
                raise RuntimeError("Alignment changed protected baseline pixels")
        baseline_mse = float(np.mean((source.astype(np.float32) - baseline.astype(np.float32)) ** 2))
        candidate_mse = float(np.mean((source.astype(np.float32) - candidate.astype(np.float32)) ** 2))
        self.last_quality = {
            "baseline_mse": baseline_mse,
            "candidate_mse": candidate_mse,
            "baseline_perceptual": None,
            "candidate_perceptual": None,
            "selected": False,
            "reason": "pixel_error_did_not_improve",
        }
        if candidate_mse >= baseline_mse:
            return result
        self.last_quality["reason"] = "perceptual_metric_pending"
        reference = Image.fromarray(source)
        baseline_image = Image.fromarray(baseline)
        candidate_image = Image.fromarray(candidate)
        baseline_perceptual = self._metric_value(self.perceptual_metric(reference.copy(), baseline_image.copy()))
        candidate_perceptual = self._metric_value(self.perceptual_metric(reference.copy(), candidate_image.copy()))
        self.last_quality["baseline_perceptual"] = baseline_perceptual
        self.last_quality["candidate_perceptual"] = candidate_perceptual
        if baseline_perceptual is None or candidate_perceptual is None:
            self.last_quality["reason"] = "invalid_perceptual_metric"
            log.warning("Alignment metric returned an invalid nonnegative finite distance; retaining the baseline")
            return result
        if candidate_perceptual >= baseline_perceptual:
            self.last_quality["reason"] = "perceptual_error_did_not_improve"
            return result
        self.last_quality["selected"] = True
        self.last_quality["reason"] = "both_metrics_improved"
        return candidate_image

    def _finish_neutral_text(
        self,
        image: Image.Image,
        result: Image.Image,
        boxes: list[Box],
        *,
        strength: float,
        seed: int | None,
        context: Any,
    ) -> Image.Image:
        import numpy as np

        from remove_ai_watermarks._internal.text_shapes import restore_contour_glyphs

        parent = self._finish_alignment(image, result, boxes, strength=strength, seed=seed, context=context)
        if not context:
            return parent
        source = np.asarray(image.convert("RGB"))
        baseline = np.asarray(result.convert("RGB"))
        donor = np.asarray(parent.convert("RGB"))
        candidate, _mask, diagnostics = restore_contour_glyphs(source, donor, context, protected_boxes=tuple(boxes))
        for x1, y1, x2, y2 in boxes:
            if not np.array_equal(candidate[y1:y2, x1:x2], baseline[y1:y2, x1:x2]):
                raise RuntimeError("Text reconstruction changed protected baseline pixels")
        decision: dict[str, Any] = {"selected": False, "reason": "unchanged", "diagnostics": diagnostics}
        self.last_text_quality = decision
        if np.array_equal(candidate, donor):
            return parent

        source_float = source.astype(np.float64)
        baseline_mse = float(np.mean((source_float - baseline.astype(np.float64)) ** 2))
        parent_mse = (
            baseline_mse if parent is result else float(np.mean((source_float - donor.astype(np.float64)) ** 2))
        )
        candidate_mse = float(np.mean((source_float - candidate.astype(np.float64)) ** 2))
        del source_float
        decision.update(baseline_mse=baseline_mse, parent_mse=parent_mse, candidate_mse=candidate_mse)
        if candidate_mse >= baseline_mse or candidate_mse > parent_mse:
            decision["reason"] = "pixel_error_did_not_improve"
            return parent
        quality = self.last_quality
        if quality is None:
            raise RuntimeError("Alignment did not record its quality decision")
        baseline_perceptual = quality["baseline_perceptual"]
        reference = Image.fromarray(source)
        if baseline_perceptual is None:
            baseline_perceptual = self._metric_value(self.perceptual_metric(reference.copy(), result.copy()))
        parent_perceptual = quality["candidate_perceptual"] if quality["selected"] else baseline_perceptual
        candidate_image = Image.fromarray(candidate)
        candidate_perceptual = self._metric_value(self.perceptual_metric(reference.copy(), candidate_image.copy()))
        decision.update(
            baseline_perceptual=baseline_perceptual,
            parent_perceptual=parent_perceptual,
            candidate_perceptual=candidate_perceptual,
        )
        if baseline_perceptual is None or parent_perceptual is None or candidate_perceptual is None:
            decision["reason"] = "invalid_perceptual_metric"
            log.warning("Text metric returned an invalid distance; retaining the previous result")
            return parent
        if candidate_perceptual >= baseline_perceptual or candidate_perceptual > parent_perceptual:
            decision["reason"] = "perceptual_error_did_not_improve"
            return parent
        decision.update(selected=True, reason="both_metrics_improved")
        return candidate_image

    def _finish_regeneration(
        self,
        image: Image.Image,
        result: Image.Image,
        boxes: list[Box],
        *,
        strength: float,
        seed: int | None,
        context: Any,
    ) -> Image.Image:
        import numpy as np

        from remove_ai_watermarks._internal.text_shapes import restore_contour_glyphs

        parent = self._finish_neutral_text(image, result, boxes, strength=strength, seed=seed, context=context)
        diagnostics = (
            cast("list[ShapeDiagnostic]", self.last_text_quality["diagnostics"]) if self.last_text_quality else []
        )
        colored_boxes = tuple(
            self._freeze_box(d["box"], image.size) for d in diagnostics if d.get("color_eligible", False)
        )
        if not colored_boxes:
            return parent
        source = np.asarray(image.convert("RGB"))
        donor = np.asarray(parent.convert("RGB"))
        candidate, mask, details = restore_contour_glyphs(
            source, donor, colored_boxes, protected_boxes=tuple(boxes), allow_colored=True
        )
        if not np.array_equal(candidate[mask == 0], donor[mask == 0]):
            raise RuntimeError("Color reconstruction changed pixels outside its support")
        baseline = np.asarray(result.convert("RGB"))
        for x1, y1, x2, y2 in boxes:
            if not np.array_equal(candidate[y1:y2, x1:x2], baseline[y1:y2, x1:x2]):
                raise RuntimeError("Color reconstruction changed protected baseline pixels")
        decision: dict[str, Any] = {"selected": False, "reason": "unchanged", "diagnostics": details}
        self.last_color_quality = decision
        if np.array_equal(candidate, donor):
            return parent
        source_float = source.astype(np.float64)
        parent_mse = float(np.mean((source_float - donor.astype(np.float64)) ** 2))
        candidate_mse = float(np.mean((source_float - candidate.astype(np.float64)) ** 2))
        del source_float
        decision.update(parent_mse=parent_mse, candidate_mse=candidate_mse)
        if candidate_mse >= parent_mse:
            decision["reason"] = "pixel_error_did_not_improve"
            return parent
        reference = Image.fromarray(source)
        candidate_image = Image.fromarray(candidate)
        parent_perceptual = None
        if self.last_text_quality and self.last_text_quality.get("selected"):
            parent_perceptual = self._metric_value(self.last_text_quality.get("candidate_perceptual"))
        elif self.last_quality:
            selected_key = "candidate_perceptual" if self.last_quality["selected"] else "baseline_perceptual"
            parent_perceptual = self._metric_value(self.last_quality[selected_key])
        if parent_perceptual is None:
            parent_perceptual = self._metric_value(self.perceptual_metric(reference.copy(), parent.copy()))
        candidate_perceptual = self._metric_value(self.perceptual_metric(reference.copy(), candidate_image.copy()))
        decision.update(parent_perceptual=parent_perceptual, candidate_perceptual=candidate_perceptual)
        if parent_perceptual is None or candidate_perceptual is None:
            decision["reason"] = "invalid_perceptual_metric"
            log.warning("Color metric returned an invalid distance; retaining the previous result")
            return parent
        if candidate_perceptual >= parent_perceptual:
            decision["reason"] = "perceptual_error_did_not_improve"
            return parent
        decision.update(selected=True, reason="both_metrics_improved")
        return candidate_image


class _BaselineContext(tuple[Box, ...]):
    """Distinguish an unsupported fidelity path from a text-free normal input."""


@dataclass
class Qwen21FidelityZImagePipeline(Qwen21AlignmentZImagePipeline):
    """Apply calibrated fidelity stages only where the shared recipe supports them.

    Tiled inputs, reviewed-text calls and very small inputs retain the shared
    Qwen-2.1 behavior. Text geometry is optional and must be qualified by the
    caller; absent geometry still permits bounded alignment.
    """

    profile_name: ClassVar[str] = "qwen21-zimage"
    perceptual_metric: PerceptualMetric = field(default_factory=LocalPerceptualMetric, kw_only=True, repr=False)
    fidelity_enabled: bool = field(default=True, kw_only=True)
    last_fidelity_skip_reason: str | None = field(default=None, init=False)

    def _prepare_regeneration(
        self, image: Image.Image, *, tile: bool, text_manifest: VerifiedTextManifest | None
    ) -> tuple[Box, ...]:
        self.last_fidelity_skip_reason = fidelity_skip_reason(
            enabled=self.fidelity_enabled, tile=tile, has_reviewed_text=text_manifest is not None, size=image.size
        )
        if self.last_fidelity_skip_reason is not None:
            self.last_quality = None
            self.last_text_quality = None
            self.last_color_quality = None
            log.info("Retaining the shared Qwen-2.1 recipe: %s", self.last_fidelity_skip_reason)
            return _BaselineContext()
        self._progress("Checking image details...")
        return super()._prepare_regeneration(image, tile=tile, text_manifest=text_manifest)

    def _finish_regeneration(
        self,
        image: Image.Image,
        result: Image.Image,
        boxes: list[Box],
        *,
        strength: float,
        seed: int | None,
        context: Any,
    ) -> Image.Image:
        if isinstance(context, _BaselineContext):
            return result
        try:
            self._progress("Preserving image details...")
            return super()._finish_regeneration(image, result, boxes, strength=strength, seed=seed, context=context)
        except Exception:
            self.last_quality = None
            self.last_text_quality = None
            self.last_color_quality = None
            self.last_fidelity_skip_reason = "quality_stage_error"
            log.warning("Fidelity acceptance failed; retaining the completed baseline", exc_info=True)
            return result
