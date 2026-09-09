"""The shared two-stage regeneration recipe: a profile global stage plus the Z-Image face stage.

Every removal profile implements this base rather than another profile:
``qwen-zimage``, ``sdxl-zimage`` and ``chroma-zimage`` each own one global regeneration stage, and a
future profile (a Flux 2 pass, for instance) plugs in the same way. Everything
here is shared verbatim, so the face stage, sizing, compositing, the prompt
cache, and the run orchestration cannot silently diverge between profiles.
"""

# DiffSynth, torch, transformers, and cv2 expose mostly untyped tensor/array APIs.
# Keep the relaxation local to this optional ML boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnnecessaryComparison=false
from __future__ import annotations

import contextlib
import hashlib
import logging
import math
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from PIL import Image

from remove_ai_watermarks._internal.watermark_profiles import resolve_seed

if TYPE_CHECKING:
    from collections.abc import Callable

    from remove_ai_watermarks._internal.text_restoration import VerifiedTextManifest

log = logging.getLogger(__name__)

ZIMAGE_TURBO_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
SAM_MODEL_ID = "facebook/sam-vit-base"

YUNET_MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_MODEL_NAME = "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
# This threshold retained the faces in the public validation set without admitting
# decorative background regions as faces.
YUNET_SCORE_THRESHOLD = 0.5

FACE_STEPS = 8
FACE_CFG = 1.0
RESIDENT_FACE_MODEL_MIN_VRAM_GIB = 64.0
FACE_DENOISE_SCALE = 0.5
GLOBAL_CONTROLNET_SCALE = 1.0

# Both stages prompt with compile-time constants, and at CFG 1.0 DiffSynth reuses the
# positive embedding for the negative side instead of encoding it, so exactly one
# embedding per stage is ever needed. Persisting it lets the next container load the
# stack without its text encoder at all: 15.45 GiB for Qwen and 7.49 GiB for Z-Image
# of a measured 87.6 GiB read per request. The stored tensors are what the encoder
# itself produced, so the output stays byte-identical. Bump the version when the
# stored shape changes; the model id and prompt are already part of the key, so a
# model or prompt change invalidates itself.
_PROMPT_CACHE_VERSION = 1
_PROMPT_CACHE_DIRNAME = "prompt-embeddings"
# DiffSynth identifies a pipeline unit by what it produces, so these tuples are how
# the prompt stage is located inside each pipeline and how its cache file is keyed.
_ZIMAGE_PROMPT_OUTPUTS = ("prompt_embeds",)

_CANNY_LOW = 13
_CANNY_HIGH = 64

# These short model inputs are retained as calibrated compatibility parameters.
# Changing them requires the same provider-oracle and identity evaluation as a model change.
_GLOBAL_PROMPT = "ultra clear and smoothe skin, spotless skin"
_GLOBAL_NEGATIVE = "moles, freckes, high detail skin"
_FACE_PROMPT = ""
_FACE_NEGATIVE = "blurry, ugly, bad quality,"


def edge_pad_to_grid(image: Image.Image, grid: int) -> Image.Image:
    """Pad RGB pixels at the lower/right edges to a latent-grid multiple."""
    width, height = image.size
    pad_width = (-width) % grid
    pad_height = (-height) % grid
    rgb = image.convert("RGB")
    if not (pad_width or pad_height):
        return rgb
    return Image.fromarray(
        np.pad(
            np.asarray(rgb),
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode="edge",
        )
    )


def resolve_face_model_residency(
    requested: bool | None,
    *,
    total_memory_gib: float,
) -> bool:
    """Keep the Z-Image stack resident when explicitly requested or safely sized."""
    if requested is not None:
        return requested
    return total_memory_gib >= RESIDENT_FACE_MODEL_MIN_VRAM_GIB


def _pin_vram_managed_models(pipe: Any) -> None:
    """Move the managed Z-Image stack to CUDA once and make offload a no-op."""
    model_names = ["text_encoder", "dit", "vae_encoder", "vae_decoder"]
    for name in model_names:
        model = getattr(pipe, name, None)
        if model is None:
            continue
        for module in model.modules():
            if not all(
                hasattr(module, attribute)
                for attribute in (
                    "offload_dtype",
                    "offload_device",
                    "onload_dtype",
                    "onload_device",
                    "preparing_dtype",
                    "preparing_device",
                    "computation_dtype",
                    "computation_device",
                )
            ):
                continue
            module.offload_dtype = module.computation_dtype
            module.offload_device = module.computation_device
            module.onload_dtype = module.computation_dtype
            module.onload_device = module.computation_device
            module.preparing_dtype = module.computation_dtype
            module.preparing_device = module.computation_device
    pipe.load_models_to_device(model_names)


def _cast_prompt_payload(payload: Any, device: Any, dtype: Any) -> Any:
    """Move a stored embedding payload onto the runtime device.

    Only floating tensors take the pipeline dtype. The attention masks travel in the
    same payload and are integer, so casting them would corrupt the prompt.
    """
    import torch

    if isinstance(payload, torch.Tensor):
        if dtype is not None and payload.is_floating_point():
            return payload.to(device=device, dtype=dtype)
        return payload.to(device=device)
    if isinstance(payload, dict):
        return {key: _cast_prompt_payload(value, device, dtype) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return type(payload)(_cast_prompt_payload(value, device, dtype) for value in payload)
    return payload


def _prompt_cache_path(model_id: str, output_params: tuple[str, ...], prompt: str) -> Path:
    """Locate the stored embedding for one model and one exact prompt string."""
    key = "\x1f".join((str(_PROMPT_CACHE_VERSION), model_id, *output_params, prompt))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return _model_cache_dir() / _PROMPT_CACHE_DIRNAME / f"{digest}.pt"


def _store_prompt_payload(path: Path, payload: Any) -> None:
    """Write the embedding atomically so a torn write can never read back as a hit."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        torch.save(_cast_prompt_payload(payload, "cpu", None), handle)
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_prompt_payload(path: Path, device: Any, dtype: Any) -> Any:
    import torch

    return _cast_prompt_payload(torch.load(path, map_location="cpu", weights_only=True), device, dtype)


def _cached_prompt_process(
    original_process: Any,
    *,
    model_id: str | None,
    output_params: tuple[str, ...],
    require_cache: bool,
) -> Any:
    cache: dict[str, dict[str, Any]] = {}

    def cached_process(
        runtime_pipe: Any,
        prompt: str,
        edit_image: Any = None,
    ) -> dict[str, Any]:
        if edit_image is not None:
            return original_process(runtime_pipe, prompt, edit_image=edit_image)
        if prompt in cache:
            return cache[prompt]
        path = None if model_id is None else _prompt_cache_path(model_id, output_params, prompt)
        if path is not None and path.exists():
            try:
                cache[prompt] = _load_prompt_payload(path, runtime_pipe.device, runtime_pipe.torch_dtype)
            except Exception:
                log.warning("Discarding unreadable prompt-embedding cache %s", path, exc_info=True)
            else:
                return cache[prompt]
        if require_cache:
            # The text encoder was left out of the model stack on the strength of
            # this file, so there is nothing left to fall back to.
            raise RuntimeError(
                f"The cached prompt embedding {path} disappeared after the stack was loaded without its text encoder."
            )
        produced = original_process(runtime_pipe, prompt, edit_image=None)
        if path is not None:
            try:
                _store_prompt_payload(path, produced)
            except OSError:
                log.warning("Could not persist the prompt embedding to %s", path, exc_info=True)
        cache[prompt] = produced
        return cache[prompt]

    return cached_process


def _cache_static_prompt_embeddings(
    pipe: Any,
    output_params: tuple[str, ...],
    *,
    model_id: str | None = None,
    require_cache: bool = False,
) -> bool:
    """Memoize a prompt unit when its embedding depends only on static text.

    With ``model_id`` the memo is also persisted, which is what lets the next
    container skip the text encoder entirely; without it the memo lives only as long
    as the pipeline. ``require_cache`` says the stack was already built without a
    text encoder, so a miss must fail loudly rather than call a model that is absent.
    """
    for unit in pipe.units:
        if tuple(getattr(unit, "output_params", ())) == output_params:
            unit.process = _cached_prompt_process(
                unit.process,
                model_id=model_id,
                output_params=output_params,
                require_cache=require_cache,
            )
            return True
    return False


def _clamp(value: float, minimum: float, maximum: float) -> float:
    if maximum < minimum:
        minimum, maximum = maximum, minimum
    return max(minimum, min(maximum, value))


def resolution_adaptive_denoise(
    width: int,
    height: int,
    *,
    adaptive_level: int = 6,
    denoise_min: float = 0.08,
    denoise_max: float = 0.15,
) -> float:
    """Choose the calibrated global strength from image area and operator level."""
    low, high = sorted((float(denoise_min), float(denoise_max)))
    megapixels = max(1.0, float(width) * float(height)) * 1e-6
    area_fraction = _clamp((megapixels - 0.30) / 3.40, 0.0, 1.0)
    strength_range = high - low
    strength = low + strength_range * area_fraction
    level_delta = float(int(adaptive_level)) - 5.0
    if level_delta >= 0.0:
        strength += level_delta * strength_range * (0.285714 / 5.0)
    else:
        strength += level_delta * strength_range * (0.257143 / 4.0)
    return _clamp(strength, 0.0001, 1.0)


def largest_face_denoise(
    boxes: list[tuple[int, int, int, int]],
    image_size: tuple[int, int],
    *,
    base_denoise: float = 0.10,
    adaptive_ratio: float = 0.03,
    denoise_min: float = 0.05,
    denoise_max: float = 0.28,
) -> float:
    """Choose the calibrated face strength from the largest detected face area."""
    width, height = image_size
    image_area = max(1.0, float(width) * float(height))
    largest_ratio = 0.0
    for x1, y1, x2, y2 in boxes:
        box_area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
        largest_ratio = max(largest_ratio, box_area / image_area)
    if largest_ratio <= 0.0:
        return _clamp(base_denoise, denoise_min, denoise_max)
    scaled = float(base_denoise) * largest_ratio / max(1e-6, float(adaptive_ratio))
    return _clamp(scaled, denoise_min, denoise_max)


def requested_steps(effective_steps: int, strength: float) -> int:
    """Scale a nominal denoising budget by strength for Diffusers img2img.

    SDXL truncates ``steps * strength``. Chroma truncates the skipped count
    instead, so the same request can execute one additional step. Both profiles
    were calibrated with this request formula; it is not an exact count across
    schedulers. DiffSynth instead executes every requested step over a shortened
    sigma range and does not use this compensation.
    """
    return max(1, math.ceil(effective_steps / max(float(strength), 1e-6)))


def _target_size(width: int, height: int, grid: int = 16) -> tuple[int, int]:
    """Floor image dimensions to the profile's latent grid."""
    return max(grid, (width // grid) * grid), max(grid, (height // grid) * grid)


def _resize_to_target(image: Image.Image) -> Image.Image:
    """Resize pixels to the exact latent-grid dimensions passed to DiffSynth."""
    target = _target_size(image.width, image.height)
    if image.size == target:
        return image
    return image.resize(target, Image.Resampling.LANCZOS)


def build_canny_control_image(image: Image.Image) -> Image.Image:
    """Build the calibrated three-channel Canny conditioning map."""
    import cv2

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, _CANNY_LOW, _CANNY_HIGH)
    return Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2))


def build_face_kwargs(crop: Image.Image, *, strength: float, seed: int | None) -> dict[str, Any]:
    """Build the DiffSynth Z-Image face-detail call shape."""
    input_image = _resize_to_target(crop)
    width, height = input_image.size
    return {
        "prompt": _FACE_PROMPT,
        "negative_prompt": _FACE_NEGATIVE,
        "cfg_scale": FACE_CFG,
        "input_image": input_image,
        "denoising_strength": float(strength),
        "height": height,
        "width": width,
        "seed": seed,
        "rand_device": "cpu",
        "num_inference_steps": FACE_STEPS,
    }


def composite_face(base: np.ndarray, detail: np.ndarray, mask: np.ndarray, *, feather: int = 10) -> np.ndarray:
    """Feather ``detail`` into ``base`` while preserving every zero-mask pixel."""
    import cv2

    if base.shape != detail.shape:
        raise ValueError("base and detail must have identical shapes")
    if mask.shape != base.shape[:2]:
        raise ValueError("mask must match the image height and width")

    alpha = mask.astype(np.float32) / 255.0
    if feather > 0:
        sigma = max(0.1, float(feather) / 3.0)
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
        # Blurring may introduce tiny values far outside the intended mask. Keep an
        # explicit support dilation so pixels beyond the feather radius stay exact.
        support = cv2.dilate((mask > 0).astype(np.uint8), np.ones((2 * feather + 1, 2 * feather + 1), np.uint8))
        alpha *= support
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
    merged = base.astype(np.float32) * (1.0 - alpha) + detail.astype(np.float32) * alpha
    return np.clip(np.rint(merged), 0, 255).astype(np.uint8)


def _expanded_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    factor: float = 2.5,
) -> tuple[int, int, int, int]:
    """Expand a face box around its center to include local lighting context."""
    x1, y1, x2, y2 = box
    image_width, image_height = image_size
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = max(1.0, (x2 - x1) * factor)
    height = max(1.0, (y2 - y1) * factor)
    return (
        max(0, round(center_x - width / 2.0)),
        max(0, round(center_y - height / 2.0)),
        min(image_width, round(center_x + width / 2.0)),
        min(image_height, round(center_y + height / 2.0)),
    )


def _model_cache_dir() -> Path:
    """Where this library persists downloaded and derived model assets.

    ``HF_HOME`` comes first because on a scale-to-zero runner it is the one path
    mounted persistently; anything under a container-local cache is re-derived on
    every request, which defeats both the YuNet download and the prompt cache.
    """
    root = os.environ.get("HF_HOME") or os.environ.get("XDG_CACHE_HOME")
    base = Path(root) if root else Path.home() / ".cache"
    return base / "remove-ai-watermarks"


def _yunet_model_path() -> Path:
    """Download the small MIT-licensed YuNet ONNX model on first use."""
    model_path = _model_cache_dir() / YUNET_MODEL_NAME
    if model_path.exists() and hashlib.sha256(model_path.read_bytes()).hexdigest() == YUNET_MODEL_SHA256:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading YuNet face detector: %s", YUNET_MODEL_URL)
    request = urllib.request.Request(
        YUNET_MODEL_URL,
        headers={"User-Agent": "remove-ai-watermarks"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed HTTPS source
        payload = response.read()
        log.info(
            "YuNet download response: status=%s content_length=%s",
            getattr(response, "status", None),
            len(payload),
        )
    digest = hashlib.sha256(payload).hexdigest()
    if digest != YUNET_MODEL_SHA256:
        raise OSError(
            "YuNet download failed integrity verification: "
            f"expected {YUNET_MODEL_SHA256}, got {digest} ({len(payload)} bytes)"
        )
    with tempfile.NamedTemporaryFile(dir=model_path.parent, suffix=".onnx", delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(model_path)
    return model_path


def _nms_boxes(
    boxes: list[tuple[int, int, int, int]],
    scores: list[float],
    *,
    threshold: float = 0.3,
) -> list[tuple[int, int, int, int]]:
    """Apply OpenCV NMS to detections collected at multiple image scales."""
    if not boxes:
        return []
    import cv2

    xywh = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
    indices = cv2.dnn.NMSBoxes(xywh, scores, score_threshold=0.2, nms_threshold=threshold)
    if len(indices) == 0:
        return []
    return [boxes[int(index)] for index in np.asarray(indices).reshape(-1)]


def detect_faces(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Detect face boxes with YuNet at two scales for large and small faces."""
    import cv2

    rgb = np.asarray(image.convert("RGB"))
    original_height, original_width = rgb.shape[:2]
    detections: list[tuple[int, int, int, int]] = []
    scores: list[float] = []
    model_path = _yunet_model_path()

    for long_side in (640, 1280):
        scale = min(1.0, long_side / max(original_width, original_height))
        width = max(1, round(original_width * scale))
        height = max(1, round(original_height * scale))
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb
        bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (width, height),
            YUNET_SCORE_THRESHOLD,
            0.3,
            5000,
        )
        _, rows = detector.detect(bgr)
        if rows is None:
            continue
        inverse = 1.0 / scale
        for row in rows:
            x, y, box_width, box_height = (float(value) for value in row[:4])
            x1 = max(0, round(x * inverse))
            y1 = max(0, round(y * inverse))
            x2 = min(original_width, round((x + box_width) * inverse))
            y2 = min(original_height, round((y + box_height) * inverse))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append((x1, y1, x2, y2))
            scores.append(float(row[-1]))
        if scale == 1.0:
            break
    return _nms_boxes(detections, scores)


def _ellipse_masks(
    boxes: list[tuple[int, int, int, int]],
    image_size: tuple[int, int],
) -> list[np.ndarray]:
    """Safe fallback masks when SAM is unavailable."""
    import cv2

    width, height = image_size
    masks: list[np.ndarray] = []
    for x1, y1, x2, y2 in boxes:
        mask = np.zeros((height, width), dtype=np.uint8)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = (max(1, int((x2 - x1) * 0.55)), max(1, int((y2 - y1) * 0.62)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        masks.append(mask)
    return masks


def _prepare_sam_inputs(inputs: Any, device: str, dtype: Any) -> Any:
    """Move SAM inputs to the target device without casting geometric prompts."""
    prepared = inputs.to(device)
    if "pixel_values" in prepared:
        prepared["pixel_values"] = prepared["pixel_values"].to(dtype=dtype)
    return prepared


def _sam_point_prompts(
    boxes: list[tuple[int, int, int, int]],
) -> tuple[list[list[list[list[float]]]], list[list[list[int]]]]:
    """Build Impact Pack's center-1 positive prompt for every face box."""
    points = [[[[(x1 + x2) / 2.0, (y1 + y2) / 2.0]] for x1, y1, x2, y2 in boxes]]
    labels = [[[1] for _box in boxes]]
    return points, labels


def _clip_sam_masks_to_boxes(
    masks: list[np.ndarray],
    boxes: list[tuple[int, int, int, int]],
    image_size: tuple[int, int],
) -> list[np.ndarray]:
    """Match Impact Pack by intersecting each SAM mask with its detector box."""
    # Same rectangle primitive the shared fill uses, rather than a private zeros/fill
    # copy. `dilate=0` because this box CLIPS a SAM mask -- growing it would admit the
    # pixels the clip exists to exclude. The function-local import keeps region_eraser's
    # module-scope cv2 off this module's import path.
    from remove_ai_watermarks.region_eraser import boxes_to_mask

    width, height = image_size
    clipped: list[np.ndarray] = []
    for mask, (x1, y1, x2, y2) in zip(masks, boxes, strict=True):
        # (x, y, w, h) with w = x2 - x1 keeps x + w == x2, so a negative origin clamps
        # to the same span the explicit max/min pair produced.
        box_mask = boxes_to_mask((height, width), [(x1, y1, x2 - x1, y2 - y1)], dilate=0)
        clipped.append(np.bitwise_and(mask.astype(np.uint8), box_mask))
    return clipped


def _select_sam_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float = 0.93,
) -> list[np.ndarray]:
    """Select and combine SAM proposals like Impact Pack's ``sub_threshold``."""
    mask_array = np.asarray(masks)
    score_array = np.asarray(scores)
    if mask_array.ndim == 5 and mask_array.shape[0] == 1:
        mask_array = mask_array[0]
    if score_array.ndim == 3 and score_array.shape[0] == 1:
        score_array = score_array[0]
    if mask_array.ndim == 3:
        mask_array = mask_array[:, None, :, :]
    if score_array.ndim == 1:
        score_array = score_array[:, None]
    if mask_array.ndim != 4 or score_array.ndim != 2:
        raise ValueError("SAM masks and scores have unexpected dimensions")
    if mask_array.shape[:2] != score_array.shape:
        raise ValueError("SAM mask proposals and IoU scores do not align")

    selected_masks: list[np.ndarray] = []
    for candidates, candidate_scores in zip(mask_array, score_array, strict=True):
        selected = np.flatnonzero(candidate_scores >= threshold)
        if selected.size == 0:
            selected = np.asarray([int(np.argmax(candidate_scores))])
        combined = np.any(candidates[selected] > 0, axis=0)
        selected_masks.append(combined.astype(np.uint8) * 255)
    return selected_masks


def _sam_outputs_to_numpy(masks: Any, scores: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert SAM tensors through float32 because NumPy rejects bfloat16."""
    mask_array = masks.detach().float().cpu().numpy()
    score_array = scores.detach().float().cpu().numpy()
    return mask_array, score_array


@dataclass
class TwoStageZImagePipeline:
    """Lazy runtime for the shared two-stage regeneration recipe.

    A profile subclass owns the global stage: ``_load_global`` loads and caches its
    model stack, ``_run_global`` regenerates a frame, and ``_vae_roundtrip``
    optionally exposes that stack's VAE as a verified-text donor. The Z-Image face
    stage with YuNet detection and SAM masks, the prompt cache, sizing, and the run
    orchestration live here and are inherited verbatim by every profile.
    """

    # Names the CUDA-only refusal after the concrete profile rather than after this
    # base, so a directly constructed subclass reports itself.
    profile_name: ClassVar[str] = ""

    device: str
    torch_dtype: Any
    hf_token: str | None = None
    progress_callback: Callable[[str], None] | None = None
    controlnet_conditioning_scale: float = GLOBAL_CONTROLNET_SCALE
    keep_face_models_on_device: bool | None = None
    keep_global_models_on_device: bool | None = None
    cache_prompt_embeddings: bool = True

    def __post_init__(self) -> None:
        self._zimage_pipe: Any = None
        self._sam_model: Any = None
        self._sam_processor: Any = None

    def _progress(self, message: str) -> None:
        if self.progress_callback is not None:
            with contextlib.suppress(Exception):
                self.progress_callback(message)

    def _require_cuda(self) -> None:
        if self.device != "cuda":
            name = self.profile_name or type(self).__name__
            raise RuntimeError(
                f"The {name} pipeline is CUDA-only; there is no CPU or MPS path "
                "for it. WatermarkRemover already refuses a non-CUDA device, so "
                "reaching this guard means the pipeline was constructed directly."
            )

    def _vram_limit(self) -> float | None:
        import torch

        with contextlib.suppress(Exception):
            return max(1.0, torch.cuda.mem_get_info("cuda")[1] / (1024**3) - 0.5)
        return None

    def _total_vram_gib(self) -> float:
        """Card capacity for the residency gates, 0.0 when it cannot be read.

        Separate from ``_vram_limit``, which answers a different question (the budget
        handed to DiffSynth) and must stay ``None`` rather than 0.0 when unknown, so
        an unreadable device means "no limit" there and "assume small" here.
        """
        import torch

        with contextlib.suppress(Exception):
            return torch.cuda.get_device_properties("cuda").total_memory / (1024**3)
        return 0.0

    def _keep_face_models_resident(self) -> bool:
        return resolve_face_model_residency(
            self.keep_face_models_on_device,
            total_memory_gib=self._total_vram_gib(),
        )

    def _prompt_is_cached(self, model_id: str, output_params: tuple[str, ...], prompt: str) -> bool:
        """Whether this stack can be built without its text encoder at all."""
        return self.cache_prompt_embeddings and _prompt_cache_path(model_id, output_params, prompt).exists()

    @classmethod
    def _face_stage_dtype(cls) -> Any:
        """The dtype every face-stage model loads and computes in.

        Deliberately independent of ``self.torch_dtype``, which belongs to the global
        stage. ``sdxl-zimage`` runs its global model in fp16, and inheriting that here
        built the Z-Image modules bf16 (per the VRAM config below) while handing them
        fp16 latents -- a Half/BFloat16 conv mismatch that crashed every face image
        while zero-face inputs passed, so no unit test could see it.

        Read by Z-Image and by SAM alike, so the whole stage moves together. SAM never
        crashed, because it casts its own inputs and leaves through ``.float()``, but it
        was reading the same wrong field and would re-land the bug for the next profile
        that changes its global dtype.
        """
        return cls._zimage_vram_config()["computation_dtype"]

    @staticmethod
    def _zimage_vram_config() -> dict[str, Any]:
        import torch

        return {
            "offload_dtype": torch.bfloat16,
            "offload_device": "cpu",
            "onload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }

    def _load_zimage(self) -> Any:
        if self._zimage_pipe is not None:
            return self._zimage_pipe
        self._require_cuda()
        os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
        if self.hf_token:
            os.environ.setdefault("HF_TOKEN", self.hf_token)
        try:
            from diffsynth.pipelines.z_image import ModelConfig, ZImagePipeline
        except ImportError as exc:
            raise ImportError(
                "The two-stage pipelines need the optional dependency group. "
                "Install: pip install 'remove-ai-watermarks[qwen-zimage]'"
            ) from exc

        self._progress("Loading Z-Image Turbo face-detail model...")
        keep_on_device = self._keep_face_models_resident()
        config = self._zimage_vram_config()
        text_encoder_config = ModelConfig(
            model_id=ZIMAGE_TURBO_MODEL_ID,
            origin_file_pattern="text_encoder/*.safetensors",
            **config,
        )
        model_configs = [
            ModelConfig(
                model_id=ZIMAGE_TURBO_MODEL_ID,
                origin_file_pattern="transformer/*.safetensors",
                **config,
            ),
            text_encoder_config,
            ModelConfig(
                model_id=ZIMAGE_TURBO_MODEL_ID,
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                **config,
            ),
        ]
        prompt_cached = self._prompt_is_cached(ZIMAGE_TURBO_MODEL_ID, _ZIMAGE_PROMPT_OUTPUTS, _FACE_PROMPT)
        if prompt_cached:
            model_configs.remove(text_encoder_config)
            log.info("Z-Image prompt embedding is cached; loading the stack without its text encoder")
        pipe = ZImagePipeline.from_pretrained(
            torch_dtype=self._face_stage_dtype(),
            device=self.device,
            model_configs=model_configs,
            tokenizer_config=ModelConfig(
                model_id=ZIMAGE_TURBO_MODEL_ID,
                origin_file_pattern="tokenizer/",
            ),
            vram_limit=self._vram_limit(),
        )
        if keep_on_device:
            _pin_vram_managed_models(pipe)
        if self.cache_prompt_embeddings:
            _cache_static_prompt_embeddings(
                pipe,
                _ZIMAGE_PROMPT_OUTPUTS,
                model_id=ZIMAGE_TURBO_MODEL_ID,
                require_cache=prompt_cached,
            )
        self._zimage_pipe = pipe
        return pipe

    def _load_sam(self) -> tuple[Any, Any]:
        if self._sam_model is not None and self._sam_processor is not None:
            return self._sam_model, self._sam_processor
        self._progress("Loading SAM face-mask model...")
        try:
            from transformers import AutoModelForMaskGeneration, AutoProcessor
        except ImportError as exc:
            raise ImportError("SAM needs transformers and torchvision from the qwen-zimage extra.") from exc
        kwargs: dict[str, Any] = {}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        processor = AutoProcessor.from_pretrained(SAM_MODEL_ID, **kwargs)
        model = AutoModelForMaskGeneration.from_pretrained(
            SAM_MODEL_ID,
            torch_dtype=self._face_stage_dtype(),
            **kwargs,
        ).to(self.device)
        model.eval()
        self._sam_model = model
        self._sam_processor = processor
        return model, processor

    def _sam_masks(
        self,
        image: Image.Image,
        boxes: list[tuple[int, int, int, int]],
    ) -> list[np.ndarray]:
        if not boxes:
            return []
        import torch

        try:
            model, processor = self._load_sam()
            input_points, input_labels = _sam_point_prompts(boxes)
            inputs = processor(
                images=image,
                input_boxes=[[list(box) for box in boxes]],
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt",
            )
            original_sizes = inputs["original_sizes"].clone()
            reshaped_sizes = inputs["reshaped_input_sizes"].clone()
            inputs = _prepare_sam_inputs(inputs, self.device, self._face_stage_dtype())
            with torch.inference_mode():
                outputs = model(**inputs, multimask_output=True)
            processed = processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                original_sizes,
                reshaped_sizes,
            )[0]
            mask_array, score_array = _sam_outputs_to_numpy(processed, outputs.iou_scores)
            binary_masks = _select_sam_masks(
                mask_array,
                score_array,
            )
            return _clip_sam_masks_to_boxes(binary_masks, boxes, image.size)
        except Exception as exc:
            log.warning("SAM face-mask refinement failed (%s); using box-derived ellipse masks", exc)
            return _ellipse_masks(boxes, image.size)

    def preload(self, *, global_only: bool = False) -> None:
        """Eagerly load the mandatory stage and, by default, the face stack."""
        self._load_global()
        _yunet_model_path()
        if not global_only:
            self._load_zimage()
            self._load_sam()

    def _load_global(self) -> Any:
        """Load and cache the profile's global model stack."""
        raise NotImplementedError("A profile pipeline must implement the global stage loader")

    def _run_global(self, image: Image.Image, strength: float, seed: int | None) -> Image.Image:
        """Regenerate one frame with the profile's global stage."""
        raise NotImplementedError("A profile pipeline must implement the global stage")

    def _vae_roundtrip(self, image: Image.Image) -> Image.Image:
        """Reconstruct source pixels through the profile's already loaded VAE."""
        raise NotImplementedError(f"{type(self).__name__} does not expose a VAE donor for verified text restoration")

    @staticmethod
    def _detail_size(
        crop_size: tuple[int, int],
        face_size: tuple[int, int],
    ) -> tuple[int, int]:
        """Scale a crop toward a 768px face guide while capping it at 1024px."""
        crop_width, crop_height = crop_size
        face_width, face_height = face_size
        scale_for_face = 768.0 / max(1, max(face_width, face_height))
        scale_for_crop = 1024.0 / max(1, max(crop_width, crop_height))
        scale = min(scale_for_face, scale_for_crop)
        # Never shrink below the crop's current size unless the cap requires it.
        if max(crop_width, crop_height) <= 1024:
            scale = max(1.0, scale)
        width = max(16, round(crop_width * scale / 16.0) * 16)
        height = max(16, round(crop_height * scale / 16.0) * 16)
        return width, height

    def _run_faces(
        self,
        original: Image.Image,
        global_result: Image.Image,
        boxes: list[tuple[int, int, int, int]],
        masks: list[np.ndarray],
        *,
        strength: float,
        seed: int | None,
    ) -> Image.Image:
        if not boxes:
            return global_result
        pipe = self._load_zimage()
        base = np.asarray(global_result.convert("RGB")).copy()
        source = np.asarray(original.convert("RGB"))
        detail_seed = None if seed is None else seed + 1

        for index, (box, mask) in enumerate(zip(boxes, masks, strict=True), start=1):
            crop_box = _expanded_box(box, original.size)
            cx1, cy1, cx2, cy2 = crop_box
            crop_source = source[cy1:cy2, cx1:cx2]
            crop_mask = mask[cy1:cy2, cx1:cx2]
            if crop_source.size == 0 or not np.any(crop_mask):
                continue

            face_width = box[2] - box[0]
            face_height = box[3] - box[1]
            process_size = self._detail_size((cx2 - cx1, cy2 - cy1), (face_width, face_height))
            crop_image = Image.fromarray(crop_source).resize(process_size, Image.Resampling.LANCZOS)
            self._progress(
                f"Regenerating face {index}/{len(boxes)} with Z-Image: strength={strength:.4f}, steps={FACE_STEPS}..."
            )
            detailed = pipe(**build_face_kwargs(crop_image, strength=strength, seed=detail_seed))
            detailed = detailed.convert("RGB").resize((cx2 - cx1, cy2 - cy1), Image.Resampling.LANCZOS)

            base_crop = base[cy1:cy2, cx1:cx2]
            base[cy1:cy2, cx1:cx2] = composite_face(
                base_crop,
                np.asarray(detailed),
                crop_mask,
                feather=10,
            )
        return Image.fromarray(base)

    def run(
        self,
        image: Image.Image,
        *,
        strength: float | None,
        seed: int | None,
        tile: bool = False,
        tile_size: int = 1024,
        tile_overlap: int = 128,
        text_manifest: VerifiedTextManifest | None = None,
        fidelity_anchor: bool = False,
    ) -> Image.Image:
        """Execute global regeneration and masked face repair."""
        if text_manifest is not None and tile:
            raise ValueError("Verified text restoration is not calibrated with tiled diffusion")
        self._require_cuda()
        seed = resolve_seed(seed)
        donor = None
        if text_manifest is not None:
            self._progress("Reconstructing the verified text donor...")
            donor = self._vae_roundtrip(image)
        global_strength = (
            resolution_adaptive_denoise(image.width, image.height) if strength is None else float(strength)
        )
        if tile and max(image.size) > tile_size:
            from remove_ai_watermarks._internal.tiling import run_tiled

            global_result = run_tiled(
                lambda tile_image: self._run_global(tile_image, global_strength, seed),
                image,
                tile_size,
                tile_overlap,
                self._progress,
            )
        else:
            global_result = self._run_global(image, global_strength, seed)

        self._progress("Detecting faces on the original image...")
        boxes = detect_faces(image)
        if not boxes:
            self._progress("No faces detected; keeping the global result.")
            result = global_result
        else:
            masks = self._sam_masks(image, boxes)
            face_strength = largest_face_denoise(boxes, image.size) * FACE_DENOISE_SCALE
            result = self._run_faces(
                image,
                global_result,
                boxes,
                masks,
                strength=face_strength,
                seed=seed,
            )
        if text_manifest is None:
            return result
        if donor is None:
            raise RuntimeError("Verified text restoration requires a VAE donor")
        from remove_ai_watermarks._internal.text_restoration import (
            blend_fidelity_anchor,
            restore_verified_text,
        )

        # The anchor is OFF by default since 0.27.1: blending 15% of the Qwen-VAE
        # donor ACROSS THE WHOLE FRAME returned detector-visible OpenAI SynthID on
        # poster-scale manifests (official Content Provenance API, 2026-08-19:
        # detected x6 with the anchor, clean x6 without it, base clean throughout;
        # see docs/text-protection-research.md). ``fidelity_anchor=True`` keeps the
        # 0.27.0 research behavior for reproduction.
        if fidelity_anchor:
            self._progress("Blending the Qwen-VAE fidelity anchor...")
            anchor = blend_fidelity_anchor(result, donor)
        else:
            anchor = result
        self._progress(f"Restoring {len(text_manifest.lines)} verified text lines...")
        return restore_verified_text(image, anchor, donor, text_manifest.lines)
