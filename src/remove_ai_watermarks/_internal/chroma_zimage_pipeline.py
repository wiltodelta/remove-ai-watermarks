"""The chroma-zimage recipe with a Chroma1 global stage.

Only the global regeneration model changes. The face stage is inherited verbatim
from :class:`TwoStageZImagePipeline` -- same YuNet detection, same SAM masks, same
Z-Image Turbo repair of the original crops, same feathered compositing -- so a
change there cannot silently diverge between the profiles.

Pieces bound to this architecture: the diffusers ``ChromaImg2ImgPipeline``
(no DiffSynth here), the neutral faithful-regeneration prompt the floors were
calibrated with, guidance 5.0, and ``ceil(4 / strength)`` requested steps. Chroma
truncates the skipped interval, so this calibrated schedule executes four or
five steps depending on strength; SDXL rounds differently. Strength is bound
to the request schedule too: the flat
vendor floors in ``watermark_profiles`` come from the 2026-08-29/30 oracle
calibration (docs/chroma1-engine-research.md) and do not transfer to another
prompt, guidance, or effective step count.

No Canny conditioning: the floors were measured on a plain strength-controlled
img2img pass, so that is the calibrated path.
"""

# Diffusers and torch expose mostly untyped tensor APIs. Keep the relaxation local
# to this optional ML boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportPrivateUsage=false, reportPrivateImportUsage=false, reportOptionalMemberAccess=false
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from PIL import Image

from remove_ai_watermarks._internal.two_stage_pipeline import (
    TwoStageZImagePipeline,
    _load_prompt_payload,
    _prompt_cache_path,
    _store_prompt_payload,
    _target_size,
    edge_pad_to_grid,
    requested_steps,
)

log = logging.getLogger(__name__)

CHROMA_MODEL_ID = "lodestones/Chroma1-HD"

# Calibration used ceil(4 / strength) requested steps. Chroma truncates the
# skipped count, so some strengths execute five steps rather than four.
# Preserve this request schedule: changing it invalidates the measured floors.
CHROMA_NOMINAL_STEPS = 4
CHROMA_GUIDANCE = 5.0

# The neutral faithful-regeneration prompt the floors were measured with. It is
# deliberately NOT the canny-stage prompt the Qwen/SDXL profiles use: those were
# never calibrated against Chroma1, and swapping prompts silently moves the
# oracle boundaries.
CHROMA_PROMPT = "high quality, sharp, detailed, faithful to the original"
CHROMA_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"

# Prompt-embedding cache key for the Chroma1 T5 encoder. The cache lets a warm
# container skip the ~2 s T5 inference on every image, and after the first encode
# the text encoder is freed from VRAM (~9.5 GiB of the ~29 GiB peak).
_CHROMA_PROMPT_CACHE_KEY = (CHROMA_MODEL_ID, ("chroma1-prompt-embeds-v1",), CHROMA_PROMPT)

# The latent grid the calibration floors were generated on. The source is floored
# to this grid, generated, then resized back to the exact input size -- the same
# shape the prototype scripts used.
_LATENT_GRID = 16


def chroma_target_size(width: int, height: int) -> tuple[int, int]:
    """Floor dimensions to the latent grid without changing aspect."""
    return _target_size(width, height, _LATENT_GRID)


@dataclass
class ChromaZImagePipeline(TwoStageZImagePipeline):
    """Lazy runtime for the Chroma1 global stage plus the inherited face stage."""

    profile_name: ClassVar[str] = "chroma-zimage"

    def __post_init__(self) -> None:
        super().__post_init__()
        self._chroma_pipe: Any = None
        self._chroma_embeds: dict[str, Any] | None = None
        self._chroma_t5_freed: bool = False

    def _load_global(self) -> Any:
        if self._chroma_pipe is not None:
            return self._chroma_pipe
        self._require_cuda()
        try:
            import torch
            from diffusers import ChromaImg2ImgPipeline
        except ImportError as exc:
            raise ImportError(
                "The chroma-zimage pipeline needs the optional dependency group. "
                "Install: pip install 'remove-ai-watermarks[qwen-zimage]'"
            ) from exc

        self._progress("Loading Chroma1-HD (bf16)...")
        token = {"token": self.hf_token} if self.hf_token else {}
        pipe = ChromaImg2ImgPipeline.from_pretrained(CHROMA_MODEL_ID, torch_dtype=torch.bfloat16, **token)
        pipe = pipe.to(self.device)

        # Encode and cache the static prompt once; subsequent containers load
        # from the cache and skip both the T5 inference and the encoder's VRAM.
        cache_path = _prompt_cache_path(*_CHROMA_PROMPT_CACHE_KEY)
        if self.cache_prompt_embeddings and cache_path.exists():
            self._progress("Loading cached Chroma1 prompt embeddings...")
            payload = _load_prompt_payload(cache_path, self.device, torch.bfloat16)
            self._chroma_embeds = payload
            # Free the T5 encoder: setting to None releases the VRAM reference.
            pipe.text_encoder = None
            torch.cuda.empty_cache()
            self._chroma_t5_freed = True
        elif self.cache_prompt_embeddings:
            self._progress("Encoding and caching Chroma1 prompt...")
            prompt_embeds, _text_ids, prompt_mask, negative_embeds, _neg_ids, negative_mask = pipe.encode_prompt(
                prompt=CHROMA_PROMPT,
                negative_prompt=CHROMA_NEGATIVE,
                device=self.device,
                do_classifier_free_guidance=True,
            )
            p_mask = prompt_mask.cpu() if prompt_mask is not None else torch.zeros(1)
            n_mask = negative_mask.cpu() if negative_mask is not None else torch.zeros(1)
            payload = {
                "prompt_embeds": prompt_embeds.cpu(),
                "negative_prompt_embeds": negative_embeds.cpu(),
                "prompt_attention_mask": p_mask,
                "negative_prompt_attention_mask": n_mask,
            }
            _store_prompt_payload(cache_path, payload)
            loaded = _load_prompt_payload(cache_path, self.device, torch.bfloat16)
            self._chroma_embeds = loaded
            # Free the T5 encoder: setting to None releases the VRAM reference.
            pipe.text_encoder = None
            torch.cuda.empty_cache()
            self._chroma_t5_freed = True
        else:
            self._chroma_embeds = None
            self._chroma_t5_freed = False

        self._chroma_pipe = pipe
        return pipe

    def _run_global(self, image: Image.Image, strength: float, seed: int | None) -> Image.Image:
        import torch

        pipe = self._load_global()
        target = chroma_target_size(image.width, image.height)
        prepared = image if image.size == target else image.resize(target, Image.Resampling.LANCZOS)
        steps = requested_steps(CHROMA_NOMINAL_STEPS, strength)
        self._progress(f"Running Chroma1 pass: strength={strength:.4f}, requested steps={steps}...")
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None

        # Pass cached embeddings when available to skip the T5 inference.
        embeds_kwargs: dict[str, Any] = {}
        if self._chroma_embeds is not None:
            embeds_kwargs = {
                "prompt_embeds": self._chroma_embeds["prompt_embeds"],
                "negative_prompt_embeds": self._chroma_embeds["negative_prompt_embeds"],
                "prompt_attention_mask": self._chroma_embeds["prompt_attention_mask"],
                "negative_prompt_attention_mask": self._chroma_embeds["negative_prompt_attention_mask"],
            }
        else:
            embeds_kwargs = {"prompt": CHROMA_PROMPT, "negative_prompt": CHROMA_NEGATIVE}

        result = pipe(
            width=target[0],
            height=target[1],
            image=prepared,
            strength=float(strength),
            num_inference_steps=steps,
            guidance_scale=CHROMA_GUIDANCE,
            generator=generator,
            **embeds_kwargs,
        ).images[0]
        if result.size != image.size:
            result = result.resize(image.size, Image.Resampling.LANCZOS)
        return result.convert("RGB")

    def _vae_roundtrip(self, image: Image.Image) -> Image.Image:
        """Reconstruct source pixels through the already loaded Chroma VAE."""
        import torch

        pipe = self._load_global()
        source_width, source_height = image.size
        padded = edge_pad_to_grid(image, _LATENT_GRID)

        tensor = pipe.image_processor.preprocess(
            padded,
            height=padded.height,
            width=padded.width,
        ).to(device=self.device, dtype=self.torch_dtype)
        with torch.inference_mode():
            encoded = pipe.vae.encode(tensor)
            if hasattr(encoded, "latent_dist"):
                # A restoration donor must be deterministic. Diffusers samples this
                # distribution for img2img noise initialization; its mode is the
                # faithful VAE reconstruction needed here.
                latents = encoded.latent_dist.mode()
            elif hasattr(encoded, "latents"):
                latents = encoded.latents
            else:
                raise AttributeError("Chroma VAE encoder output contains no latents")
            decoded = pipe.vae.decode(latents, return_dict=False)[0]
        reconstructed = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        return reconstructed.crop((0, 0, source_width, source_height)).convert("RGB")
