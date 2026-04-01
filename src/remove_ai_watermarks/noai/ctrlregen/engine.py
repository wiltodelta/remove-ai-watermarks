"""CtrlRegen engine — orchestrates the full watermark removal pipeline.

Loads the base SD 1.5 model with a ControlNet (spatial control from
canny edges) and a DINOv2-based IP Adapter (semantic control), then
runs controllable regeneration with optional color matching.

Attribution:
    Based on https://github.com/yepengliu/CtrlRegen .
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import torch
from PIL import Image

from remove_ai_watermarks.noai.progress import make_pipeline_progress

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability checks — these imports are optional.
# ---------------------------------------------------------------------------

_HAS_CONTROLNET_AUX = False
_HAS_COLOR_MATCHER = False
_HAS_DIFFUSERS = False

try:
    from ctrlregen.pipeline import CustomCtrlRegenPipeline
    from diffusers import AutoencoderKL, ControlNetModel, UniPCMultistepScheduler

    _HAS_DIFFUSERS = True
except ImportError:
    AutoencoderKL = None  # type: ignore[assignment,misc]
    ControlNetModel = None  # type: ignore[assignment,misc]
    UniPCMultistepScheduler = None  # type: ignore[assignment,misc]
    CustomCtrlRegenPipeline = None  # type: ignore[assignment,misc]

try:
    from controlnet_aux import CannyDetector

    _HAS_CONTROLNET_AUX = True
except ImportError:
    CannyDetector = None  # type: ignore[assignment,misc]

try:
    from ctrlregen.color import color_match

    _HAS_COLOR_MATCHER = True
except ImportError:
    color_match = None  # type: ignore[assignment]

CTRLREGEN_HF_REPO = "yepengliu/ctrlregen"
SPATIAL_SUBFOLDER = "spatialnet_ckp/spatial_control_ckp_14000"
SEMANTIC_SUBFOLDER = "semanticnet_ckp/models"
SEMANTIC_WEIGHT_NAME = "semantic_control_ckp_435000.bin"

DEFAULT_BASE_MODEL = "SG161222/Realistic_Vision_V4.0_noVAE"
CUSTOM_VAE_ID = "stabilityai/sd-vae-ft-mse"

PROCESS_SIZE = 512
DEFAULT_GUIDANCE_SCALE = 2.0
QUALITY_PROMPT = "best quality, high quality"
NEGATIVE_PROMPT = "monochrome, lowres, bad anatomy, worst quality, low quality"

CANNY_LOW_THRESHOLD = 100
CANNY_HIGH_THRESHOLD = 150

TILE_SIZE = 512
TILE_OVERLAP = 192


def is_ctrlregen_available() -> bool:
    """Return True when all CtrlRegen-specific dependencies are installed."""
    return _HAS_DIFFUSERS and _HAS_CONTROLNET_AUX and _HAS_COLOR_MATCHER


class CtrlRegenEngine:
    """End-to-end CtrlRegen watermark removal engine.

    Handles model loading, canny edge extraction, controlled denoising,
    and color-matched post-processing in a single ``run()`` call.
    """

    def __init__(
        self,
        base_model_id: str | None = None,
        device: str = "cpu",
        torch_dtype: torch.dtype | None = None,
        hf_token: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        if not is_ctrlregen_available():
            missing: list[str] = []
            if not _HAS_DIFFUSERS:
                missing.extend(["diffusers", "transformers", "accelerate"])
            if not _HAS_CONTROLNET_AUX:
                missing.append("controlnet-aux")
            if not _HAS_COLOR_MATCHER:
                missing.append("color-matcher")
            logger.info("Auto-installing missing dependencies: %s", missing)
            import subprocess

            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", *missing],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                raise ImportError(
                    "Failed to auto-install missing dependencies: "
                    + ", ".join(missing)
                    + ". Try manually: pip install --force-reinstall noai-watermark"
                ) from exc

        self.base_model_id = base_model_id or DEFAULT_BASE_MODEL
        self.device = device
        self.torch_dtype = torch_dtype or (torch.float32 if device in ("cpu", "mps") else torch.float16)
        self.hf_token: str | None = hf_token or os.environ.get("HF_TOKEN")
        self._progress_callback = progress_callback
        self._pipeline: CustomCtrlRegenPipeline | None = None  # type: ignore[assignment]
        self._canny_detector: CannyDetector | None = None  # type: ignore[assignment]

    def _set_progress(self, message: str) -> None:
        if self._progress_callback is None:
            return
        with contextlib.suppress(Exception):
            self._progress_callback(message)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Download and assemble the full CtrlRegen pipeline."""
        if self._pipeline is not None:
            return

        token_kwargs: dict[str, Any] = {}
        if self.hf_token:
            token_kwargs["token"] = self.hf_token

        self._set_progress(f"Loading CtrlRegen spatial ControlNet from {CTRLREGEN_HF_REPO}...")
        logger.info("Loading ControlNet from %s/%s", CTRLREGEN_HF_REPO, SPATIAL_SUBFOLDER)
        controlnet = [
            ControlNetModel.from_pretrained(
                CTRLREGEN_HF_REPO,
                subfolder=SPATIAL_SUBFOLDER,
                torch_dtype=self.torch_dtype,
                **token_kwargs,
            )
        ]

        self._set_progress(f"Loading SD base model ({self.base_model_id}) for CtrlRegen pipeline...")
        logger.info("Loading base pipeline from %s", self.base_model_id)
        pipe = CustomCtrlRegenPipeline.from_pretrained(
            self.base_model_id,
            controlnet=controlnet,
            torch_dtype=self.torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
            **token_kwargs,
        )

        self._set_progress(f"Loading CtrlRegen semantic IP-Adapter + DINOv2 from {CTRLREGEN_HF_REPO}...")
        logger.info("Loading IP-Adapter from %s/%s", CTRLREGEN_HF_REPO, SEMANTIC_SUBFOLDER)
        pipe.load_ctrlregen_ip_adapter(
            CTRLREGEN_HF_REPO,
            subfolder=SEMANTIC_SUBFOLDER,
            weight_name=SEMANTIC_WEIGHT_NAME,
            **token_kwargs,
        )

        from transformers import AutoImageProcessor, AutoModel

        pipe.image_encoder = AutoModel.from_pretrained("facebook/dinov2-giant").to(self.device, dtype=self.torch_dtype)
        pipe.feature_extractor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")

        self._set_progress(f"Loading custom VAE ({CUSTOM_VAE_ID})...")
        logger.info("Loading VAE from %s", CUSTOM_VAE_ID)
        pipe.vae = AutoencoderKL.from_pretrained(
            CUSTOM_VAE_ID,
            torch_dtype=self.torch_dtype,
            **token_kwargs,
        )

        self._set_progress("Configuring UniPC scheduler...")
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_ip_adapter_scale(1.0)

        self._set_progress(f"Moving CtrlRegen pipeline to {self.device}...")
        pipe = pipe.to(self.device)

        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            with contextlib.suppress(Exception):
                pipe.enable_xformers_memory_efficient_attention()

        self._pipeline = pipe
        self._canny_detector = CannyDetector()
        self._set_progress("CtrlRegen pipeline ready.")
        logger.info("CtrlRegen pipeline loaded on %s", self.device)

    # ------------------------------------------------------------------
    # Inference — public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        image: Image.Image,
        strength: float = 0.5,
        num_inference_steps: int = 50,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        seed: int | None = None,
    ) -> Image.Image:
        """Run CtrlRegen watermark removal on a single image.

        Images that fit within ``TILE_SIZE`` (512) are processed as a
        single pass.  Larger images are split into overlapping tiles.
        """
        self.load()
        assert self._pipeline is not None
        assert self._canny_detector is not None

        orig_w, orig_h = image.size
        orig_image = image
        t0 = time.monotonic()

        needs_tiling = orig_w > TILE_SIZE or orig_h > TILE_SIZE

        if needs_tiling:
            from ctrlregen.tiling import resize_center_crop, run_tiled

            aligned_w = orig_w // 8 * 8
            aligned_h = orig_h // 8 * 8
            if aligned_w != orig_w or aligned_h != orig_h:
                image = image.resize((aligned_w, aligned_h), Image.LANCZOS)
            regen_image = run_tiled(
                pipeline=self._pipeline,
                canny_detector=self._canny_detector,
                image=image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                tile_size=TILE_SIZE,
                tile_overlap=TILE_OVERLAP,
                quality_prompt=QUALITY_PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                canny_low=CANNY_LOW_THRESHOLD,
                canny_high=CANNY_HIGH_THRESHOLD,
                device=self.device,
                set_progress=self._set_progress,
                ip_adapter_image=orig_image,
            )
        else:
            from ctrlregen.tiling import resize_center_crop

            proc_image = resize_center_crop(image, PROCESS_SIZE)
            self._set_progress(f"Preprocessed {orig_w}x{orig_h}px → {proc_image.size[0]}x{proc_image.size[1]}px")
            regen_image = self._run_single(
                proc_image,
                strength,
                num_inference_steps,
                guidance_scale,
                seed,
            )

        if regen_image.size != (orig_w, orig_h):
            self._set_progress(f"Resizing {regen_image.size[0]}x{regen_image.size[1]}px → {orig_w}x{orig_h}px...")
            regen_image = regen_image.resize((orig_w, orig_h), Image.LANCZOS)

        self._set_progress(f"Applying color matching at {orig_w}x{orig_h}px...")
        output = color_match(reference=orig_image, source=regen_image)

        self._set_progress(f"✓ CtrlRegen done · {orig_w}x{orig_h}px · {time.monotonic() - t0:.0f}s total")
        return output

    # ------------------------------------------------------------------
    # Single-image path (image <= 512x512)
    # ------------------------------------------------------------------

    def _run_single(
        self,
        image: Image.Image,
        strength: float,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> Image.Image:
        """Process a single 512x512 image through the CtrlRegen pipeline."""
        w, h = image.size
        effective_steps = max(1, int(num_inference_steps * strength))

        self._set_progress(
            f"Extracting canny edges ({w}x{h}px, thresholds {CANNY_LOW_THRESHOLD}/{CANNY_HIGH_THRESHOLD})..."
        )
        control_image = self._canny_detector(
            image,
            low_threshold=CANNY_LOW_THRESHOLD,
            high_threshold=CANNY_HIGH_THRESHOLD,
        )

        generator = torch.manual_seed(seed if seed is not None else 0)

        self._set_progress(
            f"Config: strength={strength}, steps={num_inference_steps} "
            f"(~{effective_steps} effective), guidance={guidance_scale}"
        )

        step_cb, first_step, pipeline_done, start_updater = make_pipeline_progress(
            effective_steps,
            self.device,
            self._set_progress,
            label="CtrlRegen denoising",
        )
        start_updater()

        try:
            result = self._pipeline(
                prompt=QUALITY_PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=[image],
                control_image=[control_image],
                controlnet_conditioning_scale=1.0,
                ip_adapter_image=[image],
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                control_guidance_start=0.0,
                control_guidance_end=1.0,
                callback=step_cb,
                callback_steps=1,
            )
        except TypeError:
            first_step.set()
            result = self._pipeline(
                prompt=QUALITY_PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=[image],
                control_image=[control_image],
                controlnet_conditioning_scale=1.0,
                ip_adapter_image=[image],
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        finally:
            first_step.set()
            pipeline_done.set()

        return result.images[0]
