"""Project-native orchestration for diffusion-based pixel regeneration."""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
import os
import subprocess
from typing import TYPE_CHECKING, Any

from PIL import Image

from remove_ai_watermarks._internal.watermark_profiles import (
    CHROMA_ZIMAGE_PROFILE,
    DEFAULT_PROFILE,
    INVISIBLE_EXTRA,
    PROFILE_CHOICES,
    QWEN21_ZIMAGE_PROFILE,
    QWEN_ZIMAGE_PROFILE,
    REMOVAL_MODULES,
    SDXL_ZIMAGE_PROFILE,
    fidelity_skip_reason,
    global_offload_supported,
    normalize_profile,
    resolve_effective_profile,
    resolve_seed,
    resolve_strength,
)
from remove_ai_watermarks.optional_deps import module_available

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from remove_ai_watermarks._internal.text_restoration import VerifiedTextManifest

logger = logging.getLogger(__name__)

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


def _dtype_for_profile(profile: str) -> Any:
    """Return the weight dtype bound to a concrete regeneration profile."""
    if profile == SDXL_ZIMAGE_PROFILE:
        return torch.float16  # type: ignore[union-attr]
    return torch.bfloat16  # type: ignore[union-attr]


# Probed once at import. ``torch`` is imported above rather than probed because this
# module needs the object, not just the answer.
_HAS_REMOVAL_MODULES = module_available(*(name for name in REMOVAL_MODULES if name != "torch"))


def is_watermark_removal_available() -> bool:
    """Return whether the full removal runtime can be imported."""
    return _HAS_TORCH and _HAS_REMOVAL_MODULES


def _ensure_watermark_deps() -> None:
    if not is_watermark_removal_available():
        raise ImportError(
            f"Invisible watermark regeneration requires the 'qwen-zimage' extra: pip install {INVISIBLE_EXTRA}."
        )


def _has_nvidia_gpu() -> bool:
    try:
        subprocess.run(
            ["nvidia-smi"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def _cuda_works() -> bool:
    try:
        probe = torch.tensor([1.0], device="cuda")  # type: ignore[union-attr]
        _ = probe + probe
    except (AssertionError, RuntimeError):
        return False
    return True


def get_device() -> str:
    """Return ``"cuda"`` when a usable CUDA backend is present, else ``"cpu"``.

    Deliberately binary. All profiles are CUDA-only, so an XPU or MPS answer would
    only travel one frame further to the same refusal in :class:`WatermarkRemover`,
    while costing a probe on each. ``"cpu"`` here means "no CUDA", which is exactly
    what that refusal reports.
    """
    if not _HAS_TORCH:
        return "cpu"
    if torch.cuda.is_available() and _cuda_works():  # type: ignore[union-attr]
        return "cuda"
    if _has_nvidia_gpu():
        logger.warning(
            "NVIDIA GPU detected, but the installed PyTorch build (torch %s, CUDA %s) has no working CUDA backend",
            torch.__version__,  # type: ignore[union-attr]
            torch.version.cuda,  # type: ignore[union-attr]
        )
    return "cpu"


class WatermarkRemover:
    """Load one regeneration profile and write a metadata-clean raster output."""

    def __init__(
        self,
        device: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
        hf_token: str | None = None,
        pipeline: str = DEFAULT_PROFILE,
        controlnet_conditioning_scale: float = 1.0,
        cpu_offload: bool = False,
    ) -> None:
        self.configured_profile = normalize_profile(pipeline)
        if self.configured_profile not in PROFILE_CHOICES:
            raise ValueError(f"Unsupported pipeline '{pipeline}'. Use one of: {', '.join(PROFILE_CHOICES)}.")
        # Auto starts on the concrete Qwen fallback for preload, then resolves
        # from the immutable configured profile for every image. Keeping the
        # configured and active profiles separate prevents one image's vendor
        # from becoming the next image's policy input.
        self.model_profile = resolve_effective_profile(self.configured_profile, None)
        # There is no ``model_id`` parameter and no ``model_id`` attribute: each
        # profile pins a fixed model stack, and the dtype below is bound to that
        # stack's weights. Both used to be constructor overrides that existed only to
        # be rejected or to break the run, and the attribute only existed to echo the
        # rejected value back.
        _ensure_watermark_deps()
        selected_device = (device or get_device()).casefold()
        self.device = get_device() if selected_device == "auto" else selected_device
        # CUDA is a precondition of the object, not of the run. All profiles raise on
        # any other device, so accepting one here only defers a guaranteed failure to
        # model-load time, several layers down and under the wrong profile's name.
        if self.device != "cuda":
            raise ValueError(
                f"Invisible-watermark removal is CUDA-only, so '{self.device}' cannot run it. "
                "All profiles need an NVIDIA GPU. Visible-mark removal and "
                "every identify command still run on CPU."
            )

        # SDXL ships fp16 weights and an fp16-safe VAE. Qwen and Chroma use
        # bf16; auto starts on the concrete Qwen fallback and switches together
        # with its active profile per image.
        self.torch_dtype = _dtype_for_profile(self.model_profile)

        self.cpu_offload = cpu_offload
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._progress_callback = progress_callback
        self._qwen_zimage_pipeline: Any = None
        self._text_region_selector: Callable[[Image.Image], Any] | None = None
        self._fidelity_enabled = True

    def set_fidelity_enabled(self, enabled: bool) -> None:
        """Select fidelity or the shared baseline without reloading model weights."""
        if self.configured_profile != QWEN21_ZIMAGE_PROFILE:
            raise ValueError("Fidelity activation requires the qwen21-zimage profile")
        if type(enabled) is not bool:
            raise TypeError("Fidelity activation must be boolean")
        self._fidelity_enabled = enabled
        if self._qwen_zimage_pipeline is not None:
            self._qwen_zimage_pipeline.fidelity_enabled = enabled

    def set_text_region_selector(self, selector: Callable[[Image.Image], Any] | None) -> None:
        """Bind caller-qualified source geometry to the Qwen-2.1 profile."""
        if self.configured_profile != QWEN21_ZIMAGE_PROFILE:
            raise ValueError("A text-region selector requires the qwen21-zimage profile")
        if selector is not None and not callable(selector):
            raise TypeError("Text-region selector must be callable or None")
        self._text_region_selector = selector
        if self._qwen_zimage_pipeline is not None:
            self._qwen_zimage_pipeline.line_detector = selector

    def preload(self, *, global_only: bool = False) -> None:
        """Materialize the selected model stack before the first request."""
        self._load_qwen_zimage_pipeline().preload(global_only=global_only)

    def _warn_if_global_offload_unsupported(self) -> None:
        """Say so when --cpu-offload cannot reach the stack it was asked for.

        Warned here rather than at construction because ``auto`` picks its engine
        per-image. A normal run reaches this point with a concrete profile; preload
        reaches it as unresolved ``auto`` and builds its qwen-zimage fallback.
        Either way this remains ahead of the model load the user is waiting on.
        """
        if not self.cpu_offload or global_offload_supported(self.model_profile):
            return
        logger.warning(
            "--cpu-offload does not reach the global stack of the '%s' pipeline: only "
            "qwen-zimage, qwen21-zimage, and sdxl-zimage stream their global model "
            "between CUDA calls. The face stage still honours the flag, so below the "
            "face-stage residency floor this run behaves as if --cpu-offload were absent.",
            self.model_profile,
        )

    def _load_qwen_zimage_pipeline(self) -> Any:
        if self._qwen_zimage_pipeline is None:
            self._warn_if_global_offload_unsupported()
            if self.model_profile == SDXL_ZIMAGE_PROFILE:
                from remove_ai_watermarks._internal.sdxl_zimage_pipeline import (
                    SdxlZImagePipeline as _Pipeline,
                )
            elif self.model_profile == CHROMA_ZIMAGE_PROFILE:
                from remove_ai_watermarks._internal.chroma_zimage_pipeline import (
                    ChromaZImagePipeline as _Pipeline,
                )
            elif self.model_profile == QWEN21_ZIMAGE_PROFILE:
                from remove_ai_watermarks._internal.qwen_alignment_pipeline import (
                    Qwen21FidelityZImagePipeline as _Pipeline,
                )
            else:
                from remove_ai_watermarks._internal.qwen_zimage_pipeline import (
                    QwenZImagePipeline as _Pipeline,
                )

            self._qwen_zimage_pipeline = _Pipeline(
                device=self.device,
                torch_dtype=self.torch_dtype,
                hf_token=self.hf_token,
                progress_callback=self._progress_callback,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                keep_face_models_on_device=False if self.cpu_offload else None,
                keep_global_models_on_device=False if self.cpu_offload else None,
            )
            if self.model_profile == QWEN21_ZIMAGE_PROFILE:
                self._qwen_zimage_pipeline.line_detector = self._text_region_selector
                self._qwen_zimage_pipeline.fidelity_enabled = self._fidelity_enabled
        return self._qwen_zimage_pipeline

    def _write_output(self, image: Image.Image, output_path: Path, display_tags_from: Path | None = None) -> None:
        import numpy as np

        from remove_ai_watermarks import image_io
        from remove_ai_watermarks._internal.utils import staged_sibling
        from remove_ai_watermarks.metadata import strip_written_output

        output_path.parent.mkdir(parents=True, exist_ok=True)
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        # Staged and verified, so a marker that survives fails the run with nothing
        # published at output_path.
        with staged_sibling(output_path) as staged:
            if not image_io.imwrite(str(staged), bgr, display_tags_from=display_tags_from):
                image.save(staged)
            if leftover := strip_written_output(staged):
                raise RuntimeError(f"AI metadata survived the strip: {', '.join(sorted(leftover))}")

    def remove_watermark(
        self,
        image_path: Path,
        output_path: Path | None = None,
        strength: float | None = None,
        seed: int | None = None,
        vendor: str | None = None,
        tile: bool = False,
        tile_size: int = 1024,
        tile_overlap: int = 128,
        text_manifest: VerifiedTextManifest | None = None,
        fidelity_anchor: bool = False,
    ) -> Path:
        """Regenerate image pixels and write the result without AI metadata.

        Step count and CFG are not parameters. Each stage of every profile is a
        distilled schedule that owns its own, so the only thing a caller-supplied
        value could do is break the run or be rejected.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        destination = output_path or image_path
        with Image.open(image_path) as opened:
            source = opened.convert("RGB")

        # Resolve from the configured policy for every image, before strength
        # resolution, so a previous image's vendor cannot pin the next one to
        # its engine. Reset cached weights only when the concrete engine changes.
        resolved = resolve_effective_profile(self.configured_profile, vendor)
        if resolved != self.model_profile:
            self.model_profile = resolved
            self.torch_dtype = _dtype_for_profile(resolved)
            self._qwen_zimage_pipeline = None
        if text_manifest is not None and tile:
            raise ValueError("Verified text restoration is not calibrated with tiled diffusion")
        if fidelity_anchor and self.model_profile != QWEN_ZIMAGE_PROFILE:
            raise ValueError("The fidelity anchor is supported only by the qwen-zimage profile")
        fidelity_enabled = (
            self.model_profile == QWEN21_ZIMAGE_PROFILE
            and fidelity_skip_reason(
                enabled=self._fidelity_enabled,
                tile=tile,
                has_reviewed_text=text_manifest is not None,
                size=source.size,
            )
            is None
        )
        resolved_strength = resolve_strength(
            strength, vendor, self.model_profile, size=source.size, fidelity_enabled=fidelity_enabled
        )
        if not 0.0 <= resolved_strength <= 1.0:
            raise ValueError(f"Strength must be between 0.0 and 1.0, got {resolved_strength}")

        result = self._load_qwen_zimage_pipeline().run(
            source,
            strength=resolved_strength,
            seed=resolve_seed(seed),
            tile=tile,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            text_manifest=text_manifest,
            fidelity_anchor=fidelity_anchor,
        )
        self._write_output(result, destination, display_tags_from=image_path)
        return destination
