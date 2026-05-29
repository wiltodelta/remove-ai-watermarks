"""Invisible watermark removal engine.

Wraps the vendored noai-watermark code for removing invisible AI watermarks
(SynthID, StableSignature, TreeRing) via diffusion-based regeneration.

This module requires the 'gpu' extra dependencies:
    uv pip install 'remove-ai-watermarks[gpu]'
"""

# cv2/torch boundary: this engine wraps cv2 (resize/imwrite/cvtColor), the YOLO
# face protector, and the humanizer, none of which carry usable element types;
# relax the unknown-type rules for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# Suppress verbose deprecation warnings from diffusers/transformers/huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")
warnings.filterwarnings("ignore", module="transformers")

# Suppress HuggingFace internal logging
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Check if invisible watermark removal dependencies are installed."""
    import importlib.util

    return importlib.util.find_spec("diffusers") is not None and importlib.util.find_spec("torch") is not None


def _target_size(width: int, height: int, max_resolution: int) -> tuple[int, int] | None:
    """Compute the downscaled (width, height) for a long-side cap, or None for native.

    Returns None when no pre-downscale is needed: ``max_resolution <= 0`` (native
    resolution, the default that matches the raiw.cc backend -- see issue #10) or
    the long side already fits the cap. Otherwise scales the long side down to
    ``max_resolution`` preserving aspect ratio (integer-truncated, matching the
    PIL ``resize`` call site). Pure function so the native-vs-downscale decision
    is unit-testable without loading the diffusion model.
    """
    if max_resolution > 0 and max(width, height) > max_resolution:
        ratio = max_resolution / max(width, height)
        # Clamp the short side to >=1: extreme aspect ratios (e.g. 5000x3 capped
        # at 1024) would otherwise truncate it to 0 and crash image.resize().
        return (max(1, int(width * ratio)), max(1, int(height * ratio)))
    return None


class InvisibleEngine:
    """Remove invisible AI watermarks using diffusion model regeneration.

    Based on noai-watermark by mertizci:
    https://github.com/mertizci/noai-watermark

    The approach encodes the image into latent space, injects controlled noise
    to break watermark patterns, and reconstructs via reverse diffusion.
    """

    # SDXL base is the default since May 2026: empirically defeats SynthID v2
    # at strength=0.05 / steps=50 / native ~1024px. See CLAUDE.md "Known
    # limitations" for the regression evidence ruling out SD-1.5 pipelines.
    DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    CTRLREGEN_MODEL_ID = "yepengliu/ctrlregen"

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        pipeline: str = "default",
        hf_token: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize the invisible watermark removal engine.

        Args:
            model_id: HuggingFace model ID. None = use default for pipeline.
            device: Device for inference (auto/cpu/mps/cuda/xpu). None = auto.
            pipeline: Pipeline profile. "default" (SDXL base, defeats SynthID
                v2) or "ctrlregen" (CtrlRegen).
            hf_token: HuggingFace API token.
            progress_callback: Optional callback for progress messages.
        """

        from remove_ai_watermarks.noai.watermark_remover import WatermarkRemover

        effective_model = model_id
        if pipeline == "ctrlregen" and model_id is None:
            effective_model = self.CTRLREGEN_MODEL_ID
        elif model_id is None:
            effective_model = self.DEFAULT_MODEL_ID

        self._remover = WatermarkRemover(
            model_id=effective_model,
            device=device,
            progress_callback=progress_callback,
            hf_token=hf_token,
        )
        self._progress_callback = progress_callback

    def preload(self) -> None:
        """Eagerly load the pipeline so download progress is visible."""
        self._remover.preload()

    def remove_watermark(
        self,
        image_path: Path,
        output_path: Path | None = None,
        strength: float | None = None,
        num_inference_steps: int = 100,
        guidance_scale: float | None = None,
        seed: int | None = None,
        humanize: float = 0.0,
        protect_faces: bool = True,
        protect_text: bool = True,
        max_resolution: int = 0,
    ) -> Path:
        """Remove invisible watermark from an image.

        Args:
            image_path: Path to the watermarked image.
            output_path: Output path (None = overwrite source).
            strength: Denoising strength (0.0-1.0). Default 0.04.
            steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility.
            humanize: Intensity of Analog Humanizer film grain (0 = off).
            protect_faces: Boolean to extract and restore faces intact.
            protect_text: Detect text regions and preserve them via Differential
                Diffusion when any are found, so glyphs (incl. CJK) survive the
                removal pass. On by default; the detector decides per image.
            max_resolution: Cap the long side (px) before diffusion. 0 (default)
                = native resolution, no pre-downscale -- matches the hosted
                raiw.cc backend. Set a positive value only to bound GPU/MPS
                memory on very large inputs (it reintroduces a lossy
                downscale->upscale round-trip).

        Returns:
            Path to the cleaned image.
        """
        import tempfile

        from PIL import Image, ImageOps

        # Process at native resolution by default (max_resolution=0). The hosted
        # raiw.cc backend (fal fast-sdxl) does NO pre-downscale either, and at
        # strength ~0.05 SDXL img2img does not need the input shrunk to ~1024 --
        # the old forced downscale->upscale round-trip was the main quality loss
        # (see issue #10). A positive max_resolution caps the long side only to
        # bound GPU/MPS memory on very large inputs.
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        orig_size = image.size  # (width, height)

        # Optional long-side downscale; native resolution by default (issue #10).
        target = _target_size(image.width, image.height, max_resolution)
        if target is not None:
            if self._progress_callback:
                self._progress_callback(
                    f"Downscaling {image.width}x{image.height} "
                    f"to {target[0]}x{target[1]} "
                    f"(max-resolution cap {max_resolution}px)..."
                )
            image = image.resize(target, Image.Resampling.LANCZOS)

        # Always persist to a temp file, even without downscaling: WatermarkRemover
        # reloads by path, so the EXIF-transposed pixels must be saved or rotation
        # is lost. Cleaned up in the finally block via _tmp_path.
        _tmp_fd, _tmp_str = tempfile.mkstemp(suffix=image_path.suffix)
        _tmp_path = Path(_tmp_str)
        image.save(_tmp_path)
        os.close(_tmp_fd)
        image_path = _tmp_path

        try:
            # Optional: Face protection (Phase 1 - Extraction)
            original_faces = []
            if protect_faces:
                try:
                    import cv2

                    from remove_ai_watermarks.face_protector import FaceProtector

                    if self._progress_callback:
                        self._progress_callback("Detecting and extracting faces (protect-faces)...")
                    # Convert PIL to CV2 BGR
                    import numpy as np

                    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    protector = FaceProtector(use_yolo=True)
                    original_faces = protector.extract_faces(cv_img)
                    if self._progress_callback:
                        self._progress_callback(f"Extracted {len(original_faces)} face(s) for protection.")
                except Exception as e:
                    logger.error("Failed to extract faces: %s", e)

            out_path = self._remover.remove_watermark(
                image_path=image_path,
                output_path=output_path,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                protect_text=protect_text,
            )

            # Optional: Face restoration & Humanizer (Phase 2 - Post-processing)
            if protect_faces or humanize > 0.0:
                import cv2
                import numpy as np

                from remove_ai_watermarks import image_io

                out_cv = image_io.imread(out_path, cv2.IMREAD_COLOR)

                if protect_faces and original_faces:
                    if self._progress_callback:
                        self._progress_callback("Restoring protected faces with soft blending...")
                    from remove_ai_watermarks.face_protector import FaceProtector

                    protector = FaceProtector(use_yolo=True)
                    out_cv = protector.restore_faces(out_cv, original_faces)

                if humanize > 0.0:
                    if self._progress_callback:
                        self._progress_callback(f"Applying Analog Humanizer (grain: {humanize})...")
                    from remove_ai_watermarks.humanizer import apply_analog_humanizer

                    out_cv = apply_analog_humanizer(out_cv, grain_intensity=humanize, chromatic_shift=1)

                # Restore original resolution
                if (out_cv.shape[1], out_cv.shape[0]) != orig_size:
                    if self._progress_callback:
                        self._progress_callback(
                            f"Upscaling result back to original resolution {orig_size[0]}x{orig_size[1]}..."
                        )
                    # Using INTER_LANCZOS4 for high-quality upscaling back to original
                    out_cv = cv2.resize(out_cv, orig_size, interpolation=cv2.INTER_LANCZOS4)

                image_io.imwrite(out_path, out_cv)

            else:
                # Even if no protect_faces or humanize, we must restore original size if needed
                import cv2

                from remove_ai_watermarks import image_io

                out_cv = image_io.imread(out_path, cv2.IMREAD_COLOR)
                if out_cv is not None and (out_cv.shape[1], out_cv.shape[0]) != orig_size:
                    if self._progress_callback:
                        self._progress_callback(
                            f"Upscaling result back to original resolution {orig_size[0]}x{orig_size[1]}..."
                        )
                    out_cv = cv2.resize(out_cv, orig_size, interpolation=cv2.INTER_LANCZOS4)
                    image_io.imwrite(out_path, out_cv)

            return out_path
        finally:
            # _tmp_path is always set above (we persist the image unconditionally).
            if _tmp_path.exists():
                _tmp_path.unlink()

    def remove_watermark_batch(
        self,
        input_dir: Path,
        output_dir: Path,
        strength: float = 0.04,
        steps: int = 50,
    ) -> list[Path]:
        """Remove invisible watermarks from all images in a directory."""
        return self._remover.remove_watermark_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            strength=strength,
            num_inference_steps=steps,
        )
