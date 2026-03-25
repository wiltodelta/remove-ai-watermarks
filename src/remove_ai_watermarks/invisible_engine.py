"""Invisible watermark removal engine.

Wraps the vendored noai-watermark code for removing invisible AI watermarks
(SynthID, StableSignature, TreeRing) via diffusion-based regeneration.

This module requires the 'invisible' extra dependencies:
    uv pip install 'remove-ai-watermarks[invisible]'
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Callable
from pathlib import Path

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
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


class InvisibleEngine:
    """Remove invisible AI watermarks using diffusion model regeneration.

    Based on noai-watermark by mertizci:
    https://github.com/mertizci/noai-watermark

    The approach encodes the image into latent space, injects controlled noise
    to break watermark patterns, and reconstructs via reverse diffusion.
    """

    DEFAULT_MODEL_ID = "Lykon/dreamshaper-8"
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
            device: Device for inference (auto/cpu/mps/cuda). None = auto.
            pipeline: Pipeline profile ("default" or "ctrlregen").
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
    ) -> Path:
        """Remove invisible watermark from an image.

        Args:
            image_path: Path to the watermarked image.
            output_path: Output path (None = overwrite source).
            strength: Denoising strength (0.0–1.0). Default 0.04.
            steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility.
            humanize: Intensity of Analog Humanizer film grain (0 = off).
            protect_faces: Boolean to extract and restore faces intact.

        Returns:
            Path to the cleaned image.
        """
        import tempfile

        from PIL import Image, ImageOps

        max_dimension = 768
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        orig_size = image.size  # (width, height)
        _tmp_path = None

        if max(image.width, image.height) > max_dimension:
            ratio = max_dimension / max(image.width, image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            if self._progress_callback:
                self._progress_callback(
                    f"Downscaling {image.width}x{image.height} "
                    f"to {new_size[0]}x{new_size[1]} "
                    f"(model trained at {max_dimension}px)..."
                )
            image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Save to a temp file instead of overwriting the original
            _tmp_fd, _tmp_str = tempfile.mkstemp(suffix=image_path.suffix)
            _tmp_path = Path(_tmp_str)
            image.save(_tmp_path)
            import os as _os

            _os.close(_tmp_fd)
            image_path = _tmp_path
        else:
            # We must save the transposed image back to a tmp file if it was rotated
            # otherwise WatermarkRemover will reload it without EXIF rotation!
            _tmp_fd, _tmp_str = tempfile.mkstemp(suffix=image_path.suffix)
            _tmp_path = Path(_tmp_str)
            image.save(_tmp_path)
            import os as _os

            _os.close(_tmp_fd)
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
            )

            # Optional: Face restoration & Humanizer (Phase 2 - Post-processing)
            if protect_faces or humanize > 0.0:
                import cv2
                import numpy as np

                out_cv = cv2.imread(str(out_path), cv2.IMREAD_COLOR)

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

                cv2.imwrite(str(out_path), out_cv)

            else:
                # Even if no protect_faces or humanize, we must restore original size if needed
                import cv2

                out_cv = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
                if out_cv is not None and (out_cv.shape[1], out_cv.shape[0]) != orig_size:
                    if self._progress_callback:
                        self._progress_callback(
                            f"Upscaling result back to original resolution {orig_size[0]}x{orig_size[1]}..."
                        )
                    out_cv = cv2.resize(out_cv, orig_size, interpolation=cv2.INTER_LANCZOS4)
                    cv2.imwrite(str(out_path), out_cv)

            return out_path
        finally:
            if _tmp_path is not None and _tmp_path.exists():
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
