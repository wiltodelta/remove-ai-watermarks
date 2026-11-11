"""The Qwen global stages for the two-stage regeneration recipe.

The shared recipe (the Z-Image face stage, sizing, compositing, the prompt cache,
and the run orchestration) lives in
``remove_ai_watermarks._internal.two_stage_pipeline``; this module provides:

- :class:`QwenZImagePipeline` (``qwen-zimage``, default, Apache-2.0): the
  ``Qwen/Qwen-Image-2512`` global pass with Canny conditioning and its Lightning
  LoRA distillation.
- :class:`Qwen21ZImagePipeline` (``qwen21-zimage``, non-commercial research and
  personal use only under the Qwen Research License): the ``Qwen/Qwen-Image-2.1``
  4-step Canny global pass with ``alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union``
  and ``madebyollin/texture-fix-vae-for-qwen-image-2.1``.
"""

# DiffSynth, torch, transformers, and cv2 expose mostly untyped tensor/array APIs.
# Keep the relaxation local to this optional ML boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from PIL import Image

from remove_ai_watermarks._internal.two_stage_pipeline import (
    _GLOBAL_NEGATIVE,
    _GLOBAL_PROMPT,
    DiffSynthVramConfig,
    TwoStageZImagePipeline,
    _cache_static_prompt_embeddings,
    _resize_to_target,
    build_canny_control_image,
    edge_pad_to_grid,
)

log = logging.getLogger(__name__)

QWEN_IMAGE_2512_MODEL_ID = "Qwen/Qwen-Image-2512"
QWEN_CANNY_CONTROLNET_MODEL_ID = "DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny"
QWEN_LIGHTNING_MODEL_ID = "lightx2v/Qwen-Image-2512-Lightning"
QWEN_LIGHTNING_PATTERN = "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"

QWEN_IMAGE_21_MODEL_ID = "Qwen/Qwen-Image-2.1"
QWEN21_CANNY_CONTROLNET_MODEL_ID = "alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union"
QWEN21_CANNY_CONTROLNET_PATTERN = "Qwen-Image-2.1-Fun-Controlnet-Union.safetensors"
QWEN21_TEXTURE_FIX_VAE_MODEL_ID = "madebyollin/texture-fix-vae-for-qwen-image-2.1"
QWEN21_TEXTURE_FIX_VAE_FILENAME = "diffusion_pytorch_model.safetensors"

GLOBAL_STEPS = 4
GLOBAL_CFG = 1.0
# Below this floor the mandatory Qwen stack streams from disk, which is what makes a
# 20B model runnable on a consumer card at all; at or above it, streaming is pure
# waste. Set equal to the face floor rather than lower because that is the
# configuration actually measured with both stacks resident; a tighter gate is
# plausible but unvalidated. Benchmark in docs/module-internals.md, "CPU offload".
RESIDENT_GLOBAL_MODEL_MIN_VRAM_GIB = 64.0

# DiffSynth identifies a pipeline unit by what it produces, so these tuples are how
# the prompt stage is located inside each pipeline and how its cache file is keyed.
_QWEN_PROMPT_OUTPUTS = ("prompt_emb", "prompt_emb_mask")
_QWEN21_PROMPT_OUTPUTS = ("prompt_embeds", "prompt_embeds_mask", "edit_image_pad_mask")


def resolve_global_model_residency(
    requested: bool | None,
    *,
    total_memory_gib: float,
) -> bool:
    """Keep the mandatory Qwen stack resident when explicitly requested or safely sized."""
    if requested is not None:
        return requested
    return total_memory_gib >= RESIDENT_GLOBAL_MODEL_MIN_VRAM_GIB


def build_global_kwargs(
    image: Image.Image,
    *,
    strength: float,
    seed: int | None,
    controlnet_input: Any,
) -> dict[str, Any]:
    """Build the DiffSynth Qwen call shape without importing the ML runtime."""
    input_image = _resize_to_target(image)
    width, height = input_image.size
    return {
        "prompt": _GLOBAL_PROMPT,
        "negative_prompt": _GLOBAL_NEGATIVE,
        "cfg_scale": GLOBAL_CFG,
        "input_image": input_image,
        "denoising_strength": float(strength),
        "height": height,
        "width": width,
        "seed": seed,
        "rand_device": "cpu",
        "num_inference_steps": GLOBAL_STEPS,
        "exponential_shift_mu": math.log(3.0),
        "blockwise_controlnet_inputs": [controlnet_input],
    }


@dataclass
class QwenZImagePipeline(TwoStageZImagePipeline):
    """Lazy runtime for the two-stage Qwen/Z-Image profile."""

    profile_name: ClassVar[str] = "qwen-zimage"

    def __post_init__(self) -> None:
        super().__post_init__()
        self._qwen_pipe: Any = None

    def _keep_global_models_resident(self) -> bool:
        return resolve_global_model_residency(
            self.keep_global_models_on_device,
            total_memory_gib=self._total_vram_gib(),
        )

    def _qwen_vram_config(self) -> DiffSynthVramConfig:
        return _build_qwen_vram_config(keep_resident=self._keep_global_models_resident())

    def _load_global(self) -> Any:
        if self._qwen_pipe is not None:
            return self._qwen_pipe
        self._require_cuda()
        os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
        if self.hf_token:
            os.environ.setdefault("HF_TOKEN", self.hf_token)
        try:
            from diffsynth.pipelines.qwen_image import ControlNetInput, ModelConfig, QwenImagePipeline
        except ImportError as exc:
            raise ImportError(
                "The qwen-zimage pipeline needs the optional dependency group. "
                "Install: pip install 'remove-ai-watermarks[qwen-zimage]'"
            ) from exc

        self._progress("Loading Qwen-Image-2512, Lightning LoRA, and Canny ControlNet...")
        config = self._qwen_vram_config()
        text_encoder_config = ModelConfig(
            model_id=QWEN_IMAGE_2512_MODEL_ID,
            origin_file_pattern="text_encoder/model*.safetensors",
            **config,
        )
        model_configs = [
            ModelConfig(
                model_id=QWEN_IMAGE_2512_MODEL_ID,
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                **config,
            ),
            text_encoder_config,
            ModelConfig(
                model_id=QWEN_IMAGE_2512_MODEL_ID,
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                **config,
            ),
            ModelConfig(
                model_id=QWEN_CANNY_CONTROLNET_MODEL_ID,
                origin_file_pattern="model.safetensors",
                **config,
            ),
        ]
        prompt_cached = self._prompt_is_cached(QWEN_IMAGE_2512_MODEL_ID, _QWEN_PROMPT_OUTPUTS, _GLOBAL_PROMPT)
        if prompt_cached:
            model_configs.remove(text_encoder_config)
            log.info("Qwen prompt embedding is cached; loading the stack without its text encoder")
        pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=self.torch_dtype,
            device=self.device,
            model_configs=model_configs,
            tokenizer_config=ModelConfig(
                model_id=QWEN_IMAGE_2512_MODEL_ID,
                origin_file_pattern="tokenizer/",
            ),
            vram_limit=self._vram_limit(),
        )
        from diffsynth.diffusion import FlowMatchScheduler

        # Avoid the base Qwen scheduler's terminal rescale for the distilled LoRA.
        # With exponential_shift_mu=log(3), this is the closest DiffSynth equivalent
        # of the source graph's four-step sgm_uniform + AuraFlow shift 3.
        pipe.scheduler = FlowMatchScheduler("Qwen-Image-Lightning")
        lightning = ModelConfig(
            model_id=QWEN_LIGHTNING_MODEL_ID,
            origin_file_pattern=QWEN_LIGHTNING_PATTERN,
        )
        pipe.load_lora(pipe.dit, lightning, alpha=0.8)
        if self.cache_prompt_embeddings:
            _cache_static_prompt_embeddings(
                pipe,
                _QWEN_PROMPT_OUTPUTS,
                model_id=QWEN_IMAGE_2512_MODEL_ID,
                require_cache=prompt_cached,
            )
        self._qwen_pipe = (pipe, ControlNetInput)
        return self._qwen_pipe

    def _run_global(self, image: Image.Image, strength: float, seed: int | None) -> Image.Image:
        pipe, controlnet_input_cls = self._load_global()
        input_image = _resize_to_target(image)
        control = build_canny_control_image(input_image)
        control_input = controlnet_input_cls(
            image=control,
            scale=float(self.controlnet_conditioning_scale),
        )
        self._progress(f"Running Qwen-Image-2512 Canny pass: strength={strength:.4f}, steps={GLOBAL_STEPS}...")
        result = pipe(
            **build_global_kwargs(
                input_image,
                strength=strength,
                seed=seed,
                controlnet_input=control_input,
            )
        )
        if result.size != image.size:
            result = result.resize(image.size, Image.Resampling.LANCZOS)
        return result.convert("RGB")

    def _vae_roundtrip(self, image: Image.Image) -> Image.Image:
        """Reconstruct source pixels through the already loaded Qwen VAE."""
        pipe, _controlnet_input_cls = self._load_global()
        return _qwen_vae_roundtrip(
            pipe,
            image,
            grid=8,
            device=self.device,
            dtype=self.torch_dtype,
        )


def _build_qwen_vram_config(*, keep_resident: bool) -> DiffSynthVramConfig:
    """Build the DiffSynth fp8/bf16 placement map for a Qwen global stack."""
    import torch

    if keep_resident:
        # DiffSynth decides disk_offload once from offload_dtype=="disk" or
        # onload_dtype=="disk"; keeping "disk" out of every slot is what makes
        # the global stack stay resident in CUDA fp8 instead of streaming from
        # safetensors on every step.
        return {
            "offload_dtype": torch.float8_e4m3fn,
            "offload_device": "cuda",
            "onload_dtype": torch.float8_e4m3fn,
            "onload_device": "cuda",
            "preparing_dtype": torch.float8_e4m3fn,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
    return {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.float8_e4m3fn,
        "onload_device": "cpu",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }


def _qwen_vae_roundtrip(
    pipe: Any,
    image: Image.Image,
    *,
    grid: int,
    device: Any,
    dtype: Any,
    rgba: bool = False,
) -> Image.Image:
    """Encode and decode ``image`` through a loaded DiffSynth Qwen VAE."""
    import torch

    source_width, source_height = image.size
    padded = edge_pad_to_grid(image, grid)
    if rgba:
        padded = padded.convert("RGBA")
    pipe.load_models_to_device(["vae"])
    tensor = pipe.preprocess_image(padded).to(device=device, dtype=dtype)
    with torch.inference_mode():
        latents = pipe.vae.encode(tensor)
        decoded = pipe.vae.decode(latents)
    return pipe.vae_output_to_image(decoded).crop((0, 0, source_width, source_height)).convert("RGB")


def _build_qwen21_vram_config(*, keep_resident: bool) -> DiffSynthVramConfig:
    """Build the DiffSynth bf16 placement map for the 7B Qwen-Image-2.1 stack.

    Unlike 20B Qwen-Image-2512, the 7B Qwen-Image-2.1 stack fits in bf16 (17.6 GiB
    without its text encoder), and ``madebyollin/texture-fix-vae-for-qwen-image-2.1``
    rescales decoder weights by ``1 / 512`` (``2^-9``) to prevent fp16 overflow,
    which underflows to zero if cast to unscaled ``torch.float8_e4m3fn``.
    """
    import torch

    offload_device = "cuda" if keep_resident else "cpu"
    return {
        "offload_dtype": torch.bfloat16,
        "offload_device": offload_device,
        "onload_dtype": torch.bfloat16,
        "onload_device": offload_device,
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }


@dataclass
class Qwen21ZImagePipeline(TwoStageZImagePipeline):
    """Lazy runtime for the two-stage Qwen-Image-2.1/Z-Image profile.

    Uses ``Qwen/Qwen-Image-2.1`` (Qwen Research License: non-commercial research
    and personal use only) with ``alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union``
    Canny conditioning and ``madebyollin/texture-fix-vae-for-qwen-image-2.1``.
    """

    profile_name: ClassVar[str] = "qwen21-zimage"

    def __post_init__(self) -> None:
        super().__post_init__()
        self._qwen21_pipe: Any = None

    def _qwen_vram_config(self) -> DiffSynthVramConfig:
        keep_resident = resolve_global_model_residency(
            self.keep_global_models_on_device,
            total_memory_gib=self._total_vram_gib(),
        )
        return _build_qwen21_vram_config(keep_resident=keep_resident)

    def _load_global(self) -> Any:
        if self._qwen21_pipe is not None:
            return self._qwen21_pipe
        self._require_cuda()
        os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
        if self.hf_token:
            os.environ.setdefault("HF_TOKEN", self.hf_token)
        try:
            from diffsynth.configs import VRAM_MANAGEMENT_MODULE_MAPS
            from diffsynth.core import ModelConfig
            from diffsynth.diffusion.base_pipeline import ControlNetInput
            from diffsynth.pipelines.qwen_image_21 import (
                QwenImage21Pipeline,
                QwenImage21Unit_ControlNetEmbedder,
                QwenImage21Unit_PromptEmbedder,
                model_fn_qwen_image_21,
            )
        except ImportError as exc:
            raise ImportError(
                "The qwen21-zimage pipeline needs the optional dependency group. "
                "Install: pip install 'remove-ai-watermarks[qwen-zimage]'"
            ) from exc

        controlnet_map = VRAM_MANAGEMENT_MODULE_MAPS.get(
            "diffsynth.models.qwen_image_21_controlnet.QwenImage21ControlNet"
        )
        if isinstance(controlnet_map, dict):
            controlnet_map.setdefault(
                "diffsynth.models.qwen_image_21_dit.QwenImage21ZeroCenterRMSNorm",
                "diffsynth.core.vram.layers.AutoWrappedModule",
            )

        self._progress("Loading Qwen-Image-2.1, texture-fix VAE, and Canny ControlNet...")
        config = self._qwen_vram_config()
        text_encoder_config = ModelConfig(
            model_id=QWEN_IMAGE_21_MODEL_ID,
            origin_file_pattern="text_encoder/model*.safetensors",
            **config,
        )
        model_configs = [
            ModelConfig(
                model_id=QWEN_IMAGE_21_MODEL_ID,
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                **config,
            ),
            text_encoder_config,
            ModelConfig(
                model_id=QWEN21_TEXTURE_FIX_VAE_MODEL_ID,
                origin_file_pattern=QWEN21_TEXTURE_FIX_VAE_FILENAME,
                **config,
            ),
            ModelConfig(
                model_id=QWEN21_CANNY_CONTROLNET_MODEL_ID,
                origin_file_pattern=QWEN21_CANNY_CONTROLNET_PATTERN,
                **config,
            ),
        ]
        prompt_cached = self._prompt_is_cached(QWEN_IMAGE_21_MODEL_ID, _QWEN21_PROMPT_OUTPUTS, _GLOBAL_PROMPT)
        if prompt_cached:
            model_configs.remove(text_encoder_config)
            log.info("Qwen-Image-2.1 prompt embedding is cached; loading the stack without its text encoder")
        pipe = QwenImage21Pipeline.from_pretrained(
            torch_dtype=self.torch_dtype,
            device=self.device,
            model_configs=model_configs,
            processor_config=ModelConfig(
                model_id=QWEN_IMAGE_21_MODEL_ID,
                origin_file_pattern="processor/",
            ),
            vram_limit=self._vram_limit(),
        )
        from diffsynth.diffusion import FlowMatchScheduler

        pipe.scheduler = FlowMatchScheduler("Qwen-Image-Lightning")
        units = list(getattr(pipe, "units", ()))
        prompt_unit = next(
            (u for u in units if tuple(getattr(u, "output_params", ())) == _QWEN21_PROMPT_OUTPUTS),
            None,
        )
        if prompt_unit is None:
            prompt_unit = QwenImage21Unit_PromptEmbedder()
            units.append(prompt_unit)
            pipe.units = units
        if self.cache_prompt_embeddings:
            _cache_static_prompt_embeddings(
                pipe,
                _QWEN21_PROMPT_OUTPUTS,
                model_id=QWEN_IMAGE_21_MODEL_ID,
                require_cache=prompt_cached,
            )
        control_unit = QwenImage21Unit_ControlNetEmbedder()
        self._qwen21_pipe = (pipe, ControlNetInput, prompt_unit, control_unit, model_fn_qwen_image_21)
        return self._qwen21_pipe

    def _run_global(self, image: Image.Image, strength: float, seed: int | None) -> Image.Image:
        import torch

        pipe, controlnet_input_cls, prompt_unit, control_unit, model_fn = self._load_global()
        orig_w, orig_h = image.size
        padded = edge_pad_to_grid(image, 32)
        pad_w, pad_h = padded.size
        control = build_canny_control_image(padded)
        control_input = controlnet_input_cls(
            image=control,
            scale=float(self.controlnet_conditioning_scale),
        )
        self._progress(f"Running Qwen-Image-2.1 Canny pass: strength={strength:.4f}, steps={GLOBAL_STEPS}...")
        with torch.inference_mode():
            posi_prompt = prompt_unit.process(pipe, _GLOBAL_PROMPT, edit_image=None)
            pipe.load_models_to_device(["vae"])
            input_tensor = pipe.preprocess_image(padded.convert("RGBA")).to(
                device=self.device,
                dtype=self.torch_dtype,
            )
            input_latents = pipe.vae.encode(input_tensor)
            ctrl_dict = control_unit.process(
                pipe,
                controlnet_inputs=[control_input],
                height=pad_h,
                width=pad_w,
                tiled=False,
                tile_size=256,
                tile_stride=192,
            )
            pipe.scheduler.set_timesteps(
                num_inference_steps=GLOBAL_STEPS,
                denoising_strength=float(strength),
                exponential_shift_mu=math.log(3.0),
            )
            noise = pipe.generate_noise(
                input_latents.shape,
                seed=seed,
                rand_device="cpu",
                rand_torch_dtype=self.torch_dtype,
            )
            latents = pipe.scheduler.add_noise(input_latents, noise, pipe.scheduler.timesteps[0])
            pipe.load_models_to_device(["dit", "controlnet"])
            for timestep in pipe.scheduler.timesteps:
                t_step = timestep.reshape(1).to(dtype=self.torch_dtype, device=self.device)
                noise_pred = model_fn(
                    dit=pipe.dit,
                    latents=latents,
                    timestep=t_step,
                    prompt_embeds=posi_prompt["prompt_embeds"],
                    prompt_embeds_mask=posi_prompt["prompt_embeds_mask"],
                    edit_image_pad_mask=posi_prompt["edit_image_pad_mask"],
                    edit_latents=None,
                    kv_cache=None,
                    controlnet=pipe.controlnet,
                    control_context=ctrl_dict["control_context"],
                    control_scale=ctrl_dict["control_scale"],
                )
                latents = pipe.scheduler.step(noise_pred, timestep, latents)
            pipe.load_models_to_device(["vae"])
            decoded = pipe.vae.decode(latents)
        return pipe.vae_output_to_image(decoded).crop((0, 0, orig_w, orig_h)).convert("RGB")

    def _vae_roundtrip(self, image: Image.Image) -> Image.Image:
        """Reconstruct source pixels through the already loaded Qwen-Image-2.1 texture-fix VAE."""
        pipe, *_rest = self._load_global()
        return _qwen_vae_roundtrip(
            pipe,
            image,
            grid=16,
            device=self.device,
            dtype=self.torch_dtype,
            rgba=True,
        )


if TYPE_CHECKING:
    from collections.abc import Callable


# Unbound model kernels let experimental siblings share the fixed Qwen-2.1
# global stack while the common base retains run, preload, and face ownership.
qwen21_vram_config: Callable[[Any], DiffSynthVramConfig] = Qwen21ZImagePipeline._qwen_vram_config
qwen21_load_global: Callable[[Any], Any] = Qwen21ZImagePipeline._load_global
qwen21_run_global: Callable[[Any, Image.Image, float, int | None], Image.Image] = Qwen21ZImagePipeline._run_global
qwen21_vae_roundtrip: Callable[[Any, Image.Image], Image.Image] = Qwen21ZImagePipeline._vae_roundtrip
