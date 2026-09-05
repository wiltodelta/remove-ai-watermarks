"""Diffusion engine for regenerating images that carry invisible AI watermarks.

Requires the 'qwen-zimage' extra and a CUDA device:
    uv pip install 'remove-ai-watermarks[qwen-zimage]'
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

from ._internal.watermark_profiles import (
    DEFAULT_PROFILE,
    REMOVAL_MODULES,
    SDXL_ZIMAGE_PROFILE,
    resolve_adaptive_polish,
    resolve_seed,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

# Suppress verbose deprecation warnings from diffusers/transformers/huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")
warnings.filterwarnings("ignore", module="transformers")

# Suppress Hugging Face internal logging
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Whether the dependencies for a real removal run are installed.

    Shares :data:`REMOVAL_MODULES` with the remover's own precondition so the two
    cannot drift. When they did, a torch+diffusers-only environment passed this gate
    and then died at the DiffSynth face stage.
    """
    from .optional_deps import module_available

    return module_available(*REMOVAL_MODULES)


def _target_size(width: int, height: int, max_resolution: int) -> tuple[int, int] | None:
    """Compute the (width, height) to process at, or None for native.

    One long-side adjustment: if it exceeds ``max_resolution``, scale DOWN to it
    (integer-truncated, matching the PIL ``resize`` call site). 0/negative = no cap.
    Set only to bound GPU/MPS memory on very large inputs (issue #10).

    There was also a ``min_resolution`` floor that scaled small inputs UP toward
    SDXL's ~1024 training size. It went with the SDXL profiles: both surviving
    profiles run at native geometry, so the floor was forced to 0 on every path and
    could not fire.

    Returns None when the cap does not apply (native resolution). Pure function so the
    resolution decision is unit-testable without loading the diffusion model.
    """
    long_side = max(width, height)
    if max_resolution > 0 and long_side > max_resolution:
        ratio = max_resolution / long_side
        # Clamp the short side to >=1: extreme aspect ratios (e.g. 5000x3 capped
        # at 1024) would otherwise truncate it to 0 and crash image.resize().
        return (max(1, int(width * ratio)), max(1, int(height * ratio)))
    return None


def _apply_postprocessing(
    out_cv: NDArray[Any],
    reference: Callable[[], NDArray[Any]],
    *,
    humanize: float,
    unsharp: float,
    adaptive_polish: bool,
    orig_size: tuple[int, int],
    seed: int | None,
    progress: Callable[[str], None] | None = None,
) -> NDArray[Any]:
    """Run the optional post-processing stages in order on a decoded output.

    The order is: restore the original resolution, unsharp, adaptive polish,
    humanize. Grain comes LAST because the polish measures the image it is given
    against the reference's detail level, so grain applied first is read as detail
    that is already there -- see ``humanizer.adaptive_polish``. Applying it after
    the resize also puts the grain at output resolution instead of letting a
    Lanczos upscale smear it.

    ``reference`` is a callable so the full-res original is only decoded when the
    polish actually needs it. Pure function so the stage order is unit-testable
    without loading the diffusion model.
    """
    import cv2

    from remove_ai_watermarks import humanizer

    # Restore original resolution if the input was resized for diffusion.
    if (out_cv.shape[1], out_cv.shape[0]) != orig_size:
        if progress:
            progress(f"Upscaling result back to original resolution {orig_size[0]}x{orig_size[1]}...")
        out_cv = cv2.resize(out_cv, orig_size, interpolation=cv2.INTER_LANCZOS4)

    if unsharp > 0.0:
        if progress:
            progress(f"Sharpening (unsharp mask: {unsharp})...")
        out_cv = humanizer.unsharp_mask(out_cv, amount=unsharp)

    # Adaptive polish (CLI default): restore the input's detail level in the
    # softened output, sparing text/edges. Self-limiting where no deficit.
    if adaptive_polish:
        ref = reference()
        if (ref.shape[1], ref.shape[0]) != (out_cv.shape[1], out_cv.shape[0]):
            ref = cv2.resize(ref, (out_cv.shape[1], out_cv.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        if progress:
            progress("Adaptive polish (sharpen + grain to the input's detail level)...")
        out_cv = humanizer.adaptive_polish(out_cv, ref, seed=seed, on_skip=progress)

    if humanize > 0.0:
        if progress:
            progress(f"Applying Analog Humanizer (grain: {humanize})...")
        out_cv = humanizer.apply_analog_humanizer(out_cv, grain_intensity=humanize, chromatic_shift=1)

    return out_cv


class InvisibleEngine:
    """Remove invisible AI watermarks using diffusion model regeneration.

    The approach encodes the image into latent space, injects controlled noise
    to break watermark patterns, and reconstructs via reverse diffusion.
    """

    def __init__(
        self,
        device: str | None = None,
        pipeline: str = DEFAULT_PROFILE,
        hf_token: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
        controlnet_conditioning_scale: float = 1.0,
        cpu_offload: bool = False,
    ) -> None:
        """Initialize the invisible watermark removal engine.

        Args:
            device: Device for inference. All profiles are CUDA-only, so the
                usable values are "cuda" and None/"auto" (which detects it);
                anything else raises rather than falling back.
            pipeline: Pipeline profile, one of "qwen-zimage" (DEFAULT;
                Qwen-Image-2512 Lightning + Canny, then SAM-masked Z-Image face repair)
                or "sdxl-zimage" (the same recipe and the same face stage on an SDXL
                global pass, vendor-adaptive strength because an SDXL global stage
                needs more of it) or "chroma-zimage" (the same face stage on an
                Apache-2.0 Chroma1 global pass with its own flat vendor floors;
                see docs/chroma1-engine-research.md). ALL ARE CUDA-ONLY -- there is
                no CPU or MPS path for invisible-watermark removal.
            hf_token: Hugging Face API token.
            progress_callback: Optional callback for progress messages.
            controlnet_conditioning_scale: Canny ControlNet structure-preservation
                strength on the global stage (not used by chroma-zimage).
            cpu_offload: Offload model components to CPU between CUDA calls instead
                of keeping the whole pipeline in VRAM, at the cost of speed. For
                qwen-zimage, force the face stack to offload instead of using automatic
                residency. CUDA only.
        """

        from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover

        self._remover = WatermarkRemover(
            device=device,
            progress_callback=progress_callback,
            hf_token=hf_token,
            pipeline=pipeline,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            cpu_offload=cpu_offload,
        )
        self._progress_callback = progress_callback

    def preload(self, *, global_only: bool = False) -> None:
        """Eagerly load the pipeline so download progress is visible.

        For ``qwen-zimage``, ``global_only=True`` loads the mandatory Qwen stage
        and leaves the optional Z-Image and SAM face stack lazy until a face is
        detected. Other profiles have no optional stage and ignore the flag.
        """
        self._remover.preload(global_only=global_only)

    def remove_watermark(
        self,
        image_path: Path,
        output_path: Path | None = None,
        strength: float | None = None,
        seed: int | None = None,
        humanize: float = 0.0,
        max_resolution: int = 0,
        vendor: str | None = None,
        unsharp: float = 0.0,
        adaptive_polish: bool | None = None,
        tile: bool = False,
        tile_size: int = 1024,
        tile_overlap: int = 128,
        text_manifest: Path | None = None,
        fidelity_anchor: bool = False,
    ) -> Path:
        """Remove invisible watermark from an image.

        Args:
            image_path: Path to the watermarked image.
            output_path: Output path (None = overwrite source).
            strength: Denoising strength (0.0-1.0). None -> the profile's calibrated
                default (resolution-adaptive for qwen-zimage, vendor-adaptive flat
                floors for sdxl-zimage and chroma-zimage).
            seed: Random seed for reproducibility. None resolves to 0, because all
                profiles are certified at a fixed seed.
            humanize: Intensity of Analog Humanizer film grain (0 = off).
            unsharp: Final unsharp-mask sharpening strength (0 = off, default).
                Applied last to counter the soft / over-smoothed look of the
                diffusion pass; ~0.5-0.8 is a safe range, higher risks edge halos.
            adaptive_polish: Restore the input's detail level in the softened
                output: a capped unsharp + edge-masked grain targeting the input's
                Laplacian variance. Self-limiting -- a no-op when the output already
                meets the input's detail level (text/flat graphics), so it only acts on
                over-smoothed photo/face texture. Runs LAST. None (the default) follows
                the profile: off for qwen-zimage, on for sdxl-zimage. This resolves
                through the same ``resolve_adaptive_polish`` the CLI uses, so a library
                caller and a CLI caller on one profile get the same output.
            max_resolution: Cap the long side (px) before diffusion. 0 (default)
                = no cap. Set a positive value only to bound GPU memory on very large
                inputs (it reintroduces a lossy downscale->upscale round-trip).
            tile: Process the diffusion pass in overlapping tiles instead of one
                forward pass. This retains the input's native dimensions instead
                of applying ``max_resolution``, but each tile is still regenerated.
                Engages only when the long side exceeds ``tile_size``.
            tile_size: Tile dimension in px (default 1024).
            tile_overlap: Overlap between adjacent tiles in px (default 128).
            text_manifest: Operator-verified text lines bound to the decoded source
                pixels. Enables the experimental profile-VAE ``vae-glyphs`` post-pass.
                Requires the ``text-restoration`` extra and either ``qwen-zimage``
                or ``chroma-zimage``. Incompatible with downscaling, humanize,
                unsharp, and adaptive polish. Tiling is rejected because that
                combination has no provider-oracle calibration.
            fidelity_anchor: Blend 15% of the Qwen-VAE donor across the whole frame
                before glyph restoration. OFF by default since 0.27.1: that global
                blend was measured to return detector-visible OpenAI SynthID on
                poster-scale manifests (detected x6 with the anchor vs clean x6
                without it, base clean; official Content Provenance API,
                2026-08-19 - docs/text-protection-research.md). ``True`` reproduces
                the 0.27.0 research behavior. Requires ``text_manifest`` and the
                ``qwen-zimage`` profile.

        Returns:
            Path to the cleaned image.
        """
        import tempfile

        seed = resolve_seed(seed)
        adaptive_polish = resolve_adaptive_polish(adaptive_polish, self._remover.model_profile)

        if fidelity_anchor and text_manifest is None:
            raise ValueError("fidelity_anchor requires a text manifest")
        if text_manifest is not None:
            if self._remover.model_profile == SDXL_ZIMAGE_PROFILE:
                raise ValueError("--text-manifest is not supported by the sdxl-zimage profile")
            if tile:
                raise ValueError("--text-manifest is not calibrated with --tile")
            if max_resolution != 0:
                raise ValueError("--text-manifest requires --max-resolution 0")
            if humanize > 0.0 or unsharp > 0.0 or adaptive_polish:
                raise ValueError("--text-manifest requires humanize=0, unsharp=0, and adaptive polish disabled")
            from remove_ai_watermarks import region_eraser

            if not region_eraser.lama_available():
                raise RuntimeError(
                    "Verified text restoration requires LaMa. Install: "
                    "pip install 'remove-ai-watermarks[text-restoration]'"
                )

        from PIL import Image, ImageOps

        # Resolution policy: a max_resolution cap (0 = none) bounds memory on huge
        # inputs. See _target_size for why it is the only lever left.
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
        verified_text = None
        if text_manifest is not None:
            from remove_ai_watermarks._internal.text_restoration import load_verified_text_manifest

            verified_text = load_verified_text_manifest(text_manifest, reference_pil)

        # All profiles run at the input's native geometry, so only the explicit max
        # cap can move it, and it can only ever scale down.
        target = _target_size(image.width, image.height, max_resolution)
        if target is not None:
            if self._progress_callback:
                self._progress_callback(
                    f"Downscaling {image.width}x{image.height} to {target[0]}x{target[1]} "
                    f"(max-resolution cap {max_resolution}px)..."
                )
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
                seed=seed,
                vendor=vendor,
                tile=tile,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                text_manifest=verified_text,
                fidelity_anchor=fidelity_anchor,
            )

            # Post-processing chain: decode the diffusion output ONCE, apply the
            # optional stages in memory (see ``_apply_postprocessing`` for the order
            # and why), and write ONCE. Previously each stage independently
            # imread/imwrote the full-res output, so a run with several stages
            # PNG-decoded+re-encoded the same image 2-4 times. PNG is lossless, so the
            # single-write output is byte-identical.
            # Diffusers rounds native dimensions down to the latent grid (multiples
            # of 8), even when our own resolution policy did not resize the input.
            # Route those outputs through the same final resize so --no-polish does
            # not silently change e.g. 1448x1086 into 1448x1080.
            needs_restore = target is not None or any(dimension % 8 for dimension in orig_size)
            if humanize > 0.0 or unsharp > 0.0 or adaptive_polish or needs_restore:
                import cv2
                import numpy as np

                from remove_ai_watermarks import image_io

                out_cv = image_io.imread(out_path, cv2.IMREAD_COLOR)
                if out_cv is None:
                    return out_path

                out_cv = _apply_postprocessing(
                    out_cv,
                    lambda: cv2.cvtColor(np.array(reference_pil.convert("RGB")), cv2.COLOR_RGB2BGR),
                    humanize=humanize,
                    unsharp=unsharp,
                    adaptive_polish=adaptive_polish,
                    orig_size=orig_size,
                    seed=seed,
                    progress=self._progress_callback,
                )

                image_io.imwrite(out_path, out_cv)

            return out_path
        finally:
            # _tmp_path is always set above (we persist the image unconditionally).
            if _tmp_path.exists():
                _tmp_path.unlink()
