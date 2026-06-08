"""SynthID-robust face identity restoration via PhotoMaker-V2.

The diffusion removal pass scrubs the pixel watermark from the WHOLE image, including
faces, but lets faces drift in identity. Unlike the GFPGAN restore pass in
``face_restore.py`` (which runs on the watermarked ORIGINAL and re-introduces SynthID
via partial pixel blending), PhotoMaker carries identity in a SEMANTIC EMBEDDING
(OpenCLIP-ViT-H/14 image embedding, finetuned by PhotoMaker-V2) and uses it to
CONDITION a fresh txt2img generation -- the pixels are new, so the watermark cannot
be transported.

That the embedding cannot carry an invisible pixel watermark like SynthID was
empirically confirmed 2026-06-04: on 31 face crops, the cosine similarity between
``embed(orig)`` and ``embed(synthid_proxy(orig))`` (a ±2 LSB low-frequency noise of
SynthID magnitude) is 0.9977 -- an order of magnitude less drift than JPEG90, which
SynthID survives at >=99% TPR by design. See ``docs/synthid-robust-identity-research.md``.

Architecture: PhotoMaker-V2 is a fine-tuned OpenCLIP-ViT-H/14 ID encoder plus LoRA on
the SDXL UNet attention layers. It ships as a single ``photomaker-v2.bin`` checkpoint
loaded into a ``PhotoMakerStableDiffusionXLPipeline`` (txt2img only -- there is no
PhotoMakerControlNetImg2img class in diffusers). We use it as a SECOND PASS after the
main controlnet/default removal:

  1. Main removal pass (`controlnet` at the certified strength) cleans SynthID
     everywhere but leaves faces drifted.
  2. For each face found in the CLEANED image (YuNet), this module takes the SAME
     face region from the ORIGINAL, computes a PhotoMaker ID embedding from it, and
     runs PhotoMaker txt2img to regenerate JUST that face crop from the embedding.
     The freshly generated face is feather-composited back into the cleaned image.

The generated face pixels are diffusion-fresh and inherit identity from the embedding
(not the pixels), so SynthID is not re-introduced.

Commercial-safe end-to-end:
- PhotoMaker-V2 weights: Apache-2.0 (TencentARC).
- ID encoder: OpenCLIP-ViT-H/14 (MIT) finetuned by PhotoMaker (still Apache-2.0).
- SDXL base: shared with the main pipeline (already used in `default`/`controlnet`).
- NO InsightFace / antelopev2 (which is the non-commercial blocker for IP-Adapter
  FaceID / InstantID / PuLID / Arc2Face).

Requires the optional ``photomaker`` extra: ``pip install
'remove-ai-watermarks[photomaker]'`` (pulls torch / diffusers / the upstream PhotoMaker
package, all commercial-safe). Weights download on first use; never bundled.
"""

# cv2/torch/diffusers boundary: relax unknown-type rules for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import importlib.util
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# PhotoMaker-V2 weights (Apache-2.0, TencentARC). Downloaded on first use.
_PHOTOMAKER_REPO = "TencentARC/PhotoMaker-V2"
_PHOTOMAKER_FILE = "photomaker-v2.bin"
# SDXL base shared with the main pipeline (same checkpoint as `default`/`controlnet`).
_SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# The neutral prompt PhotoMaker is designed around: a class noun + the trigger word
# `img`, which PhotoMaker replaces with the ID embedding at inference. Keeping it
# scene-neutral (no extra style words) maximises identity transfer from the embed and
# minimises hallucinated background/lighting that would not match the cleaned scene.
_PHOTOMAKER_PROMPT = "a portrait photo of a person img, natural lighting, sharp focus"
_PHOTOMAKER_NEGATIVE = "blurry, lowres, deformed, distorted, watermark"

# Square size used to feed PhotoMaker (must match a multiple of 64; 512 fits CPU/GPU
# comfortably and gives the encoder enough pixels for a stable embedding).
_PHOTOMAKER_FACE_SIZE = 512

_pipeline: Any | None = None
_pipeline_lock = threading.Lock()


def is_available() -> bool:
    """True when the optional PhotoMaker extra deps are importable."""
    return (
        importlib.util.find_spec("photomaker") is not None
        and importlib.util.find_spec("diffusers") is not None
        and importlib.util.find_spec("huggingface_hub") is not None
    )


def _select_device() -> str:
    """Pick the PhotoMaker pipeline device: CUDA when present, MPS on Apple, else CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception as e:
        logger.debug("photomaker_restore: device probe failed (%s); using CPU", e)
    return "cpu"


def _get_pipeline() -> Any:
    """Return the lazily-built PhotoMaker pipeline singleton (downloads weights on first use)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            import torch
            from huggingface_hub import hf_hub_download
            from photomaker import PhotoMakerStableDiffusionXLPipeline

            device = _select_device()
            dtype = torch.float16 if device == "cuda" else torch.float32
            logger.info("photomaker_restore: loading SDXL+PhotoMaker on %s (%s)", device, dtype)

            adapter_path = hf_hub_download(repo_id=_PHOTOMAKER_REPO, filename=_PHOTOMAKER_FILE)
            pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(_SDXL_MODEL_ID, torch_dtype=dtype)
            pipe.load_photomaker_adapter(
                str(Path(adapter_path).parent),
                subfolder="",
                weight_name=_PHOTOMAKER_FILE,
                trigger_word="img",
            )
            pipe.to(device)
            pipe.fuse_lora()
            _pipeline = pipe
    return _pipeline


def _face_crop_square(
    image_bgr: NDArray[Any],
    box: tuple[int, int, int, int],
    pad: float = 0.30,
) -> tuple[NDArray[Any], tuple[int, int, int, int]]:
    """Square crop around a face box (with padding), clipped to the image.

    Returns ``(crop_bgr, (x1, y1, x2, y2))``. The crop is the image content inside the
    returned square box -- callers use the box for the composite step. Pure numpy slicing,
    no model.
    """
    h, w = image_bgr.shape[:2]
    x, y, bw, bh = box
    cx, cy = x + bw // 2, y + bh // 2
    side = int(max(bw, bh) * (1.0 + 2.0 * pad))
    half = side // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    return image_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


def _composite_faces(
    base_bgr: NDArray[Any],
    restored_crops: list[tuple[NDArray[Any], tuple[int, int, int, int]]],
    feather_div: int = 6,
) -> NDArray[Any]:
    """Feather-composite a list of ``(restored_crop, (x1, y1, x2, y2))`` into ``base_bgr``.

    Pure cv2/numpy helper (no model), unit-testable. For each ``(crop, box)``: resize
    the crop to the box size, build a Gaussian-feathered rectangular alpha, and blend
    ``crop * a + base * (1 - a)``. Boxes that fall fully outside the image (or an empty
    list) leave ``base_bgr`` unchanged. Mirrors the alpha math in ``face_restore._composite_faces``.
    """
    import cv2
    import numpy as np

    out = base_bgr.astype(np.float32)
    h, w = base_bgr.shape[:2]

    for crop, (x1, y1, x2, y2) in restored_crops:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        resized = cv2.resize(crop, (bw, bh), interpolation=cv2.INTER_LANCZOS4)

        alpha = np.zeros((h, w), dtype=np.float32)
        alpha[y1:y2, x1:x2] = 1.0
        k = max(3, (min(bw, bh) // feather_div) | 1)
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)[:, :, None]

        full_restored = np.zeros_like(out)
        full_restored[y1:y2, x1:x2] = resized
        out = full_restored * alpha + out * (1.0 - alpha)

    return np.clip(out, 0, 255).astype(np.uint8)


def restore_faces_photomaker(
    original_bgr: NDArray[Any],
    cleaned_bgr: NDArray[Any],
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    style_strength: int = 20,
    seed: int | None = None,
    detect_faces_fn: Any | None = None,
) -> NDArray[Any]:
    """SynthID-robust face identity restoration via PhotoMaker txt2img.

    Pipeline:
      1. Detect faces in ``cleaned_bgr`` (YuNet via the package's ``auto_config`` by
         default; override via ``detect_faces_fn`` for tests).
      2. For each face: take the SAME box from ``original_bgr`` -> square crop -> PhotoMaker
         txt2img with that crop as the ID image -> a fresh face generated from the
         OpenCLIP embedding (the embedding is SynthID-invariant by ~3 orders of magnitude,
         see docs/synthid-robust-identity-research.md).
      3. Feather-composite each regenerated face into ``cleaned_bgr``.

    Faces are taken from ``original_bgr`` (the embedding ignores the watermark) but the
    PIXELS that land in the output are diffusion-fresh, so SynthID is not transported.

    Args:
        original_bgr: The original (watermarked) image as cv2 BGR. Source of identity.
        cleaned_bgr: The main-pass output as cv2 BGR. Faces drifted in identity; this
            module replaces those face regions.
        num_inference_steps: Diffusion steps inside PhotoMaker (def 30).
        guidance_scale: CFG scale inside PhotoMaker (def 5.0; the PhotoMaker recipe).
        style_strength: PhotoMaker's ``start_merge_step`` knob ~ 20-30 (def 20).
        seed: Optional seed for reproducibility.
        detect_faces_fn: Optional callable ``(bgr) -> list[(x,y,w,h)]`` to override the
            default YuNet detector (used by tests).

    Returns:
        ``cleaned_bgr`` with regenerated face regions composited in (or unchanged when
        no face is detected).
    """
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    if detect_faces_fn is None:
        from remove_ai_watermarks import auto_config as _ac

        def _default_detect(bgr: NDArray[Any]) -> list[tuple[int, int, int, int]]:
            h, w = bgr.shape[:2]
            model = Path(_ac.__file__).parent / "assets" / "face_detection_yunet_2023mar.onnx"
            det = cv2.FaceDetectorYN.create(str(model), "", (w, h), _ac._FACE_SCORE, 0.3, 5000)
            det.setInputSize((w, h))
            _, faces = det.detect(bgr)
            if faces is None:
                return []
            return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces if int(f[2]) > 0 and int(f[3]) > 0]

        detect_faces_fn = _default_detect

    boxes = detect_faces_fn(cleaned_bgr)
    if not boxes:
        logger.debug("photomaker_restore: no faces detected; returning cleaned image unchanged")
        return cleaned_bgr

    pipeline = _get_pipeline()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    restored: list[tuple[NDArray[Any], tuple[int, int, int, int]]] = []
    for box in boxes:
        id_crop_bgr, square_box = _face_crop_square(original_bgr, box)
        if id_crop_bgr.size == 0:
            continue
        id_crop_rgb = cv2.cvtColor(id_crop_bgr, cv2.COLOR_BGR2RGB)
        id_image_pil = Image.fromarray(id_crop_rgb)

        out = pipeline(
            prompt=_PHOTOMAKER_PROMPT,
            negative_prompt=_PHOTOMAKER_NEGATIVE,
            input_id_images=[id_image_pil],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            start_merge_step=style_strength,
            generator=generator,
            height=_PHOTOMAKER_FACE_SIZE,
            width=_PHOTOMAKER_FACE_SIZE,
            num_images_per_prompt=1,
        )
        gen_rgb = out.images[0]
        gen_bgr = cv2.cvtColor(np.array(gen_rgb), cv2.COLOR_RGB2BGR)
        restored.append((gen_bgr, square_box))

    return _composite_faces(cleaned_bgr, restored)
