"""Invisible watermark removal engine.

Wraps the vendored noai-watermark code for removing invisible AI watermarks
(SynthID, StableSignature, TreeRing) via diffusion-based regeneration.

This module requires the 'gpu' extra dependencies:
    uv pip install 'remove-ai-watermarks[gpu]'
"""

# cv2/torch boundary: this engine wraps cv2 (resize/imwrite/cvtColor) and the
# humanizer, none of which carry usable element types; relax the unknown-type
# rules for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .noai.watermark_profiles import DEFAULT_MODEL_ID as DEFAULT_SDXL_MODEL_ID

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
    from .optional_deps import module_available

    return module_available("diffusers", "torch")


def _target_size(width: int, height: int, max_resolution: int, min_resolution: int = 0) -> tuple[int, int] | None:
    """Compute the (width, height) to process at, or None for native.

    Two opposite long-side adjustments, in precedence order:

    - ``max_resolution`` (cap): if the long side exceeds it, scale DOWN to it
      (integer-truncated, matching the PIL ``resize`` call site). 0/negative = no
      cap. Set only to bound GPU/MPS memory on very large inputs (issue #10).
    - ``min_resolution`` (floor): else if the long side is below it, scale UP to it
      (rounded) so SDXL img2img runs near its ~1024 training resolution instead of
      degrading on a tiny latent (a 381x512 portrait distorts badly at native).
      The output is restored to the original size by the caller, so the floor is a
      transparent quality boost. 0 = no floor. Skipped on a ``min > max`` misconfig.

    Returns None when neither applies (native resolution). Pure function so the
    resolution decision is unit-testable without loading the diffusion model.
    """
    long_side = max(width, height)
    if max_resolution > 0 and long_side > max_resolution:
        ratio = max_resolution / long_side
        # Clamp the short side to >=1: extreme aspect ratios (e.g. 5000x3 capped
        # at 1024) would otherwise truncate it to 0 and crash image.resize().
        return (max(1, int(width * ratio)), max(1, int(height * ratio)))
    if min_resolution > 0 and long_side < min_resolution and (max_resolution <= 0 or min_resolution <= max_resolution):
        ratio = min_resolution / long_side
        return (max(1, round(width * ratio)), max(1, round(height * ratio)))
    return None


class InvisibleEngine:
    """Remove invisible AI watermarks using diffusion model regeneration.

    Based on noai-watermark by mertizci:
    https://github.com/mertizci/noai-watermark

    The approach encodes the image into latent space, injects controlled noise
    to break watermark patterns, and reconstructs via reverse diffusion.
    """

    # SDXL base is the default since May 2026; the vendor-adaptive strength
    # removes the current SynthID (see watermark_profiles + docs/synthid.md).
    DEFAULT_MODEL_ID = DEFAULT_SDXL_MODEL_ID

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        pipeline: str = "controlnet",
        hf_token: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
        controlnet_conditioning_scale: float = 1.0,
        cpu_offload: bool = False,
    ) -> None:
        """Initialize the invisible watermark removal engine.

        Args:
            model_id: HuggingFace model ID. None = use the SDXL base default.
            device: Device for inference (auto/cpu/mps/cuda/xpu). None = auto.
            pipeline: Pipeline profile. "controlnet" (DEFAULT; SDXL + canny ControlNet
                that preserves text/face structure via edge conditioning while removing
                SynthID), "sdxl" (plain SDXL img2img, lighter but leaves SynthID on
                flat-graphic content), or "qwen" (Qwen-Image 20B img2img, best text/
                structure preservation but CUDA/cloud-class). "default" aliases "sdxl".
            hf_token: HuggingFace API token.
            progress_callback: Optional callback for progress messages.
            controlnet_conditioning_scale: ControlNet structure-preservation
                strength (controlnet pipeline only).
            cpu_offload: Stream pipeline submodules to CUDA on demand instead of
                holding the whole fp16 pipeline in VRAM. Lets a low-VRAM card (e.g.
                8 GB) run SDXL that would otherwise OOM, at the cost of speed. CUDA only.
        """

        from remove_ai_watermarks.noai.watermark_remover import WatermarkRemover

        effective_model = model_id or self.DEFAULT_MODEL_ID

        self._remover = WatermarkRemover(
            model_id=effective_model,
            device=device,
            progress_callback=progress_callback,
            hf_token=hf_token,
            pipeline=pipeline,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            cpu_offload=cpu_offload,
        )
        self._progress_callback = progress_callback

    def preload(self) -> None:
        """Eagerly load the pipeline so download progress is visible."""
        self._remover.preload()

    def _esrgan_upscale(self, image: Any, target: tuple[int, int]) -> Any:
        """Upscale a PIL image to ``target`` with Real-ESRGAN, else Lanczos.

        Runs Real-ESRGAN at its native factor (on the remover's device, CPU fallback),
        then resizes to the exact ``target`` with Lanczos. Falls back to a plain Lanczos
        resize when the ``esrgan`` extra is absent or the model errors.
        """
        import cv2
        import numpy as np
        from PIL import Image

        from remove_ai_watermarks import upscaler

        if not upscaler.is_available():
            logger.debug("esrgan upscaler requested but the extra is absent; using Lanczos")
            return image.resize(target, Image.Resampling.LANCZOS)
        try:
            bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            big = upscaler.upscale(bgr, device=self._remover.device)
            if (big.shape[1], big.shape[0]) != target:
                big = cv2.resize(big, target, interpolation=cv2.INTER_LANCZOS4)
            return Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
        except Exception as e:  # never let an optional upscaler break removal
            logger.warning("Real-ESRGAN upscale failed (%s); using Lanczos", e)
            return image.resize(target, Image.Resampling.LANCZOS)

    def remove_watermark(
        self,
        image_path: Path,
        output_path: Path | None = None,
        strength: float | None = None,
        num_inference_steps: int = 100,
        guidance_scale: float | None = None,
        seed: int | None = None,
        humanize: float = 0.0,
        max_resolution: int = 0,
        min_resolution: int = 1024,
        vendor: str | None = None,
        unsharp: float = 0.0,
        adaptive_polish: bool = False,
        upscaler: str = "lanczos",
        tile: bool = False,
        tile_size: int = 1024,
        tile_overlap: int = 128,
    ) -> Path:
        """Remove invisible watermark from an image.

        Args:
            image_path: Path to the watermarked image.
            output_path: Output path (None = overwrite source).
            strength: Denoising strength (0.0-1.0). None -> the vendor-adaptive
                default.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility.
            humanize: Intensity of Analog Humanizer film grain (0 = off).
            unsharp: Final unsharp-mask sharpening strength (0 = off, default).
                Applied last to counter the soft / over-smoothed look of the
                diffusion pass; ~0.5-0.8 is a safe range, higher risks edge halos.
            adaptive_polish: When True (the CLI default), restore the input's detail
                level in the softened output: a capped unsharp + edge-masked grain
                targeting the input's Laplacian variance. Self-limiting -- a no-op when
                the output already meets the input's detail level (text/flat graphics),
                so it only acts on over-smoothed photo/face texture. Runs LAST.
            max_resolution: Cap the long side (px) before diffusion. 0 (default)
                = no cap. Set a positive value only to bound GPU/MPS memory on
                very large inputs (it reintroduces a lossy downscale->upscale
                round-trip).
            min_resolution: Upscale the long side UP to this (px) before diffusion
                when the input is smaller, so SDXL runs near its ~1024 training
                resolution (small inputs degrade/distort badly at native). 1024
                (default) = on; 0 = off. The output is restored to the original
                input size, so this is a transparent quality boost; it adds time
                and memory on small inputs. Ignored on a min > max misconfig.
            upscaler: How to upscale a small input to the ``min_resolution`` floor:
                ``"lanczos"`` (default, cv2, no deps) or ``"esrgan"`` (Real-ESRGAN
                via the ``esrgan`` extra). Only applies when UPscaling (the floor
                case); a ``max_resolution`` downscale always uses Lanczos. Falls back
                to Lanczos if the extra is absent.
            tile: Process the diffusion pass in overlapping tiles instead of one
                forward pass -- the lossless alternative to ``max_resolution`` for
                large inputs that OOM on MPS/GPU. Engages only when the long side
                exceeds ``tile_size``. Pair with ``max_resolution=0`` (the default)
                so the input keeps its native resolution.
            tile_size: Tile dimension in px (default 1024).
            tile_overlap: Overlap between adjacent tiles in px (default 128).

        Returns:
            Path to the cleaned image.
        """
        import tempfile

        from PIL import Image, ImageOps

        # Resolution policy: a max_resolution cap (0 = none) bounds memory on huge
        # inputs, and a min_resolution floor (1024 = default) upscales tiny inputs so
        # SDXL img2img runs near its ~1024 training size instead of distorting on a
        # tiny latent (a 381x512 portrait wrecks at native -- issue #36 follow-up).
        # The output is restored to orig_size below, so the floor is transparent.
        # Register the HEIF/AVIF opener so a .heic/.avif input (now a SUPPORTED_FORMAT)
        # decodes here too. The --force skip path bypasses image_io.imread, which is
        # what would otherwise register it, so a bare Image.open would fail on HEIC.
        from remove_ai_watermarks import image_io

        image_io._register_heif()
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        orig_size = image.size  # (width, height)
        # Full-res original, kept for the adaptive-polish detail target (image is
        # reassigned to the resized copy below; PIL resize returns a new object).
        reference_pil = image

        target = _target_size(image.width, image.height, max_resolution, min_resolution)
        if target is not None:
            upscaling = max(target) > max(image.width, image.height)
            if self._progress_callback:
                reason = (
                    f"min-resolution floor {min_resolution}px"
                    if upscaling
                    else f"max-resolution cap {max_resolution}px"
                )
                verb = "Upscaling" if upscaling else "Downscaling"
                self._progress_callback(f"{verb} {image.width}x{image.height} to {target[0]}x{target[1]} ({reason})...")
            # Real-ESRGAN only helps when UPscaling (the floor case); a downscale cap
            # always uses Lanczos. _esrgan_upscale falls back to Lanczos if the extra is absent.
            if upscaling and upscaler == "esrgan":
                image = self._esrgan_upscale(image, target)
            else:
                image = image.resize(target, Image.Resampling.LANCZOS)

        # Always persist to a temp file, even without downscaling: WatermarkRemover
        # reloads by path, so the EXIF-transposed pixels must be saved or rotation
        # is lost. Written as PNG (lossless) regardless of the input format, so a JPEG
        # input does not feed a re-compressed copy into the diffusion pass.
        # Cleaned up in the finally block via _tmp_path.
        _tmp_fd, _tmp_str = tempfile.mkstemp(suffix=".png")
        _tmp_path = Path(_tmp_str)
        # Convert to RGB before the PNG temp: the diffusion pass is RGB anyway, and a
        # non-RGB source mode (e.g. a CMYK JPEG) cannot be written as PNG and would raise.
        image.convert("RGB").save(_tmp_path)
        os.close(_tmp_fd)
        image_path = _tmp_path

        try:
            out_path = self._remover.remove_watermark(
                image_path=image_path,
                output_path=output_path,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                vendor=vendor,
                tile=tile,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )

            # Post-processing chain: decode the diffusion output ONCE, apply the
            # optional stages in memory in order (humanize -> restore original
            # resolution -> unsharp -> adaptive polish), and write ONCE. Previously
            # each stage independently imread/imwrote the full-res output, so a run
            # with several stages PNG-decoded+re-encoded the same image 2-4 times.
            # PNG is lossless, so the single-write output is byte-identical.
            # Diffusers rounds native dimensions down to the latent grid (multiples
            # of 8), even when our own resolution policy did not resize the input.
            # Route those outputs through the same final resize so --no-polish does
            # not silently change e.g. 1448x1086 into 1448x1080.
            needs_restore = target is not None or any(dimension % 8 for dimension in orig_size)
            if humanize > 0.0 or unsharp > 0.0 or adaptive_polish or needs_restore:
                import cv2

                from remove_ai_watermarks import image_io

                out_cv = image_io.imread(out_path, cv2.IMREAD_COLOR)
                if out_cv is None:
                    return out_path

                if humanize > 0.0:
                    if self._progress_callback:
                        self._progress_callback(f"Applying Analog Humanizer (grain: {humanize})...")
                    from remove_ai_watermarks.humanizer import apply_analog_humanizer

                    out_cv = apply_analog_humanizer(out_cv, grain_intensity=humanize, chromatic_shift=1)

                # Restore original resolution if the input was resized for diffusion.
                if (out_cv.shape[1], out_cv.shape[0]) != orig_size:
                    if self._progress_callback:
                        self._progress_callback(
                            f"Upscaling result back to original resolution {orig_size[0]}x{orig_size[1]}..."
                        )
                    out_cv = cv2.resize(out_cv, orig_size, interpolation=cv2.INTER_LANCZOS4)

                if unsharp > 0.0:
                    if self._progress_callback:
                        self._progress_callback(f"Sharpening (unsharp mask: {unsharp})...")
                    from remove_ai_watermarks.humanizer import unsharp_mask

                    out_cv = unsharp_mask(out_cv, amount=unsharp)

                # Adaptive polish (CLI default): restore the input's detail level in the
                # softened output, sparing text/edges. Self-limiting where no deficit.
                if adaptive_polish:
                    import numpy as np

                    from remove_ai_watermarks import humanizer

                    ref = cv2.cvtColor(np.array(reference_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
                    if (ref.shape[1], ref.shape[0]) != (out_cv.shape[1], out_cv.shape[0]):
                        ref = cv2.resize(ref, (out_cv.shape[1], out_cv.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                    if self._progress_callback:
                        self._progress_callback("Adaptive polish (sharpen + grain to the input's detail level)...")
                    out_cv = humanizer.adaptive_polish(out_cv, ref, seed=seed)

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
        strength: float | None = None,
        steps: int = 50,
    ) -> list[Path]:
        """Remove invisible watermarks from all images in a directory."""
        return self._remover.remove_watermark_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            strength=strength,
            num_inference_steps=steps,
        )
