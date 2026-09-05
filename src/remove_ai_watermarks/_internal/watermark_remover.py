"""Project-native orchestration for diffusion-based pixel regeneration."""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
import os
import subprocess
from typing import TYPE_CHECKING, Any

from PIL import Image

from remove_ai_watermarks._internal.watermark_profiles import (
    AUTO_PROFILE,
    CHROMA_ZIMAGE_PROFILE,
    DEFAULT_PROFILE,
    INVISIBLE_EXTRA,
    PROFILE_CHOICES,
    QWEN_ZIMAGE_PROFILE,
    REMOVAL_MODULES,
    SDXL_ZIMAGE_PROFILE,
    global_offload_supported,
    normalize_profile,
    resolve_auto_profile,
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
        logger.warning("NVIDIA GPU detected, but the installed PyTorch build has no working CUDA backend")
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
        self.model_profile = normalize_profile(pipeline)
        if self.model_profile not in PROFILE_CHOICES:
            raise ValueError(f"Unsupported pipeline '{pipeline}'. Use one of: {', '.join(PROFILE_CHOICES)}.")
        # The auto profile resolves to a concrete engine per-image in
        # remove_watermark, once the provenance vendor is known. It never
        # resolves to sdxl-zimage, so the dtype below is correct either way.
        self._auto = self.model_profile == AUTO_PROFILE
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

        if self.model_profile == SDXL_ZIMAGE_PROFILE:
            # SDXL ships fp16 weights and an fp16-safe VAE; bf16 would give up the
            # variant without buying anything on this architecture.
            self.torch_dtype = torch.float16  # type: ignore[union-attr]
        else:
            # qwen-zimage, chroma-zimage, and auto all resolve to bf16 engines.
            self.torch_dtype = torch.bfloat16  # type: ignore[union-attr]

        self.cpu_offload = cpu_offload
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._progress_callback = progress_callback
        self._qwen_zimage_pipeline: Any = None

    def preload(self, *, global_only: bool = False) -> None:
        """Materialize the selected model stack before the first request."""
        self._load_qwen_zimage_pipeline().preload(global_only=global_only)

    def _warn_if_global_offload_unsupported(self) -> None:
        """Say so when --cpu-offload cannot reach the stack it was asked for.

        Warned here rather than at construction because ``auto`` picks its engine
        per-image: by the time a stack is built, ``model_profile`` is concrete, and
        this is still ahead of the model load the user is waiting on.
        """
        if not self.cpu_offload or global_offload_supported(self.model_profile):
            return
        logger.warning(
            "--cpu-offload does not reach the global stack of the '%s' pipeline: only "
            "qwen-zimage streams its global model between CUDA calls. The face stage "
            "still honours the flag, so below the face-stage residency floor this run "
            "behaves as if --cpu-offload were absent.",
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
        return self._qwen_zimage_pipeline

    def _write_output(self, image: Image.Image, output_path: Path) -> None:
        import numpy as np

        from remove_ai_watermarks import image_io

        output_path.parent.mkdir(parents=True, exist_ok=True)
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        if not image_io.imwrite(str(output_path), bgr):
            image.save(output_path)
        from remove_ai_watermarks.metadata import remove_ai_metadata

        remove_ai_metadata(output_path, output_path, keep_standard=True)

    def _resolve_chroma_strength(
        self,
        strength: float | None,
        vendor: str | None,
        source: Image.Image,
    ) -> float:
        """Resolve strength, running face detection for the chroma Google arm.

        The content-adaptive Google floor needs a face count. YuNet is fast
        (~50 ms) and the model download is already managed by the shared base.
        """
        if self.model_profile != CHROMA_ZIMAGE_PROFILE or (vendor or "").casefold() != "google":
            return resolve_strength(strength, vendor, self.model_profile, size=source.size)
        if strength is not None:
            return strength
        from remove_ai_watermarks._internal.two_stage_pipeline import detect_faces

        face_count = len(detect_faces(source))
        return resolve_strength(strength, vendor, self.model_profile, size=source.size, face_count=face_count)

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

        # The auto policy resolves per-image once the provenance vendor is
        # known, BEFORE the strength resolution so the right engine's floors
        # are used. If the vendor changed the engine, reset the cached pipeline.
        if self._auto:
            resolved = resolve_auto_profile(vendor)
            if resolved != self.model_profile:
                self.model_profile = resolved
                self._qwen_zimage_pipeline = None
        if text_manifest is not None and self.model_profile == SDXL_ZIMAGE_PROFILE:
            raise ValueError("Verified text restoration is not supported by the sdxl-zimage profile")
        if text_manifest is not None and tile:
            raise ValueError("Verified text restoration is not calibrated with tiled diffusion")
        if fidelity_anchor and self.model_profile != QWEN_ZIMAGE_PROFILE:
            raise ValueError("The fidelity anchor is supported only by the qwen-zimage profile")
        resolved_strength = self._resolve_chroma_strength(strength, vendor, source)
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
        self._write_output(result, destination)
        return destination
