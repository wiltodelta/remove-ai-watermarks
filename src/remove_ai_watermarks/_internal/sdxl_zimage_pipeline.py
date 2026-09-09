"""The sdxl-zimage recipe with an SDXL global stage.

Only the global regeneration model changes. The face stage is inherited verbatim
from :class:`TwoStageZImagePipeline` -- same YuNet detection, same SAM masks, same
Z-Image Turbo repair of the original crops, same feathered compositing -- so a
change there cannot silently diverge between the profiles.

Three pieces cannot be shared, because they are bound to the architecture: the
ControlNet, the four-step distillation LoRA, and the sampler. Strength is bound to
it too, which is the part that is easy to miss: an SDXL global pass leaves SynthID
at the strength Qwen needs. See ``watermark_profiles.SDXL_ZIMAGE_OPENAI_STRENGTH``.
"""

# Diffusers and torch expose mostly untyped tensor APIs. Keep the relaxation local
# to this optional ML boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportPrivateUsage=false, reportPrivateImportUsage=false
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from PIL import Image

from remove_ai_watermarks._internal.two_stage_pipeline import (
    _GLOBAL_NEGATIVE,
    _GLOBAL_PROMPT,
    TwoStageZImagePipeline,
    _target_size,
    build_canny_control_image,
    requested_steps,
)
from remove_ai_watermarks._internal.watermark_profiles import (
    CONTROLNET_CANNY_MODEL,
    SDXL_LIGHTNING_MODEL_ID,
    SDXL_LIGHTNING_PATTERN,
    SDXL_MODEL_ID,
)

log = logging.getLogger(__name__)

SDXL_VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"
# SDXL aligns to an 8-pixel latent grid, against Qwen's 16.
_LATENT_GRID = 8
# SDXL-Lightning's own four-step distillation, at the strength its authors document.
# The reference graph loads the Qwen LoRA at 0.8; carrying that number to a different
# LoRA on a different architecture would be imitation, not parity.
SDXL_STEPS = 4


def sdxl_target_size(width: int, height: int) -> tuple[int, int]:
    """Floor dimensions to SDXL's latent grid without changing aspect."""
    return _target_size(width, height, _LATENT_GRID)


@dataclass
class SdxlZImagePipeline(TwoStageZImagePipeline):
    """Lazy runtime for the SDXL global stage plus the inherited face stage."""

    profile_name: ClassVar[str] = "sdxl-zimage"

    def __post_init__(self) -> None:
        super().__post_init__()
        self._sdxl_pipe: Any = None

    def _load_global(self) -> Any:
        if self._sdxl_pipe is not None:
            return self._sdxl_pipe
        self._require_cuda()
        import torch
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            EulerDiscreteScheduler,
            StableDiffusionXLControlNetImg2ImgPipeline,
        )
        from huggingface_hub import hf_hub_download

        self._progress("Loading SDXL, Lightning LoRA, and Canny ControlNet...")
        token = {"token": self.hf_token} if self.hf_token else {}
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_CANNY_MODEL, torch_dtype=torch.float16, **token)
        vae = AutoencoderKL.from_pretrained(SDXL_VAE_MODEL_ID, torch_dtype=torch.float16, **token)
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            SDXL_MODEL_ID,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            add_watermarker=False,
            **token,
        ).to(self.device)
        pipe.load_lora_weights(hf_hub_download(SDXL_LIGHTNING_MODEL_ID, SDXL_LIGHTNING_PATTERN, **token))
        pipe.fuse_lora()
        # SDXL-Lightning is distilled against trailing timestep spacing.
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        self._sdxl_pipe = pipe
        return pipe

    def _run_global(self, image: Image.Image, strength: float, seed: int | None) -> Image.Image:
        import torch

        pipe = self._load_global()
        target = sdxl_target_size(image.width, image.height)
        prepared = image if image.size == target else image.resize(target, Image.Resampling.LANCZOS)
        control = build_canny_control_image(prepared)
        steps = requested_steps(SDXL_STEPS, strength)
        self._progress(f"Running SDXL Canny pass: strength={strength:.4f}, steps={SDXL_STEPS} of {steps}...")
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        result = pipe(
            prompt=_GLOBAL_PROMPT,
            negative_prompt=_GLOBAL_NEGATIVE,
            image=prepared,
            control_image=control,
            controlnet_conditioning_scale=float(self.controlnet_conditioning_scale),
            strength=float(strength),
            num_inference_steps=steps,
            guidance_scale=1.0,
            generator=generator,
        ).images[0]
        if result.size != image.size:
            result = result.resize(image.size, Image.Resampling.LANCZOS)
        return result.convert("RGB")
