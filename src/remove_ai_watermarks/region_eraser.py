"""Universal region eraser: remove anything inside user-given boxes via inpainting.

Position- and content-agnostic. You supply the rectangle(s); the eraser inpaints
whatever is inside, so it removes any visible logo / watermark / object regardless
of colour, style, or location. Localisation is the user's responsibility (pass the
box); restoration runs on CPU. This is the universal fallback for marks the
deterministic per-generator engines (Gemini sparkle, Doubao) do not cover.

Backends:
  - ``cv2`` (default): ``cv2.inpaint`` (Telea / Navier-Stokes). Instant, no extra
    dependencies, lower quality on large or textured regions.
  - ``migan`` (optional, extra ``migan``): MI-GAN via onnxruntime
    (``andraniksargsyan/migan``, MIT). CPU, ~28 MB model, ~700-950 MB peak RAM,
    ~0.19 s/call -- the droplet-friendly tier: near-big-LaMa quality on small
    marks at ~5x less RAM and ~8x faster. Model downloaded on first use.
  - ``lama`` (optional, extra ``lama``): big-LaMa via onnxruntime
    (``Carve/LaMa-ONNX``, Apache-2.0). CPU, resolution-robust, best quality on
    texture but ~200 MB model and ~4.7 GB peak RAM (too heavy for a small host).
    The model is downloaded on first use and cached by huggingface_hub; it is
    never bundled in this repo.
"""

# cv2/numpy boundary: cv2 ships no usable type info, so strict pyright cannot know
# its array element types. Relax the unknown-type rules for this file only; the
# public signatures are still annotated with NDArray[Any].
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

Backend = Literal["cv2", "lama", "migan"]

_LAMA_REPO = "Carve/LaMa-ONNX"
_LAMA_FILE = "lama_fp32.onnx"

_MIGAN_REPO = "andraniksargsyan/migan"
_MIGAN_FILE = "migan.onnx"

# Cached onnxruntime sessions (loading is expensive; reuse across calls).
_lama_session: object | None = None
_migan_session: object | None = None


def boxes_to_mask(
    shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    dilate: int = 3,
) -> NDArray[Any]:
    """Build a uint8 mask (255 inside boxes) from ``(x, y, w, h)`` rectangles."""
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    for x, y, bw, bh in boxes:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    if dilate > 0 and mask.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        mask = cv2.dilate(mask, k)
    return mask


def erase_cv2(
    image_bgr: NDArray[Any],
    mask: NDArray[Any],
    *,
    method: Literal["telea", "ns"] = "telea",
    radius: int = 6,
) -> NDArray[Any]:
    """Inpaint ``mask`` with classical cv2 inpainting (CPU, no extra deps).

    Accepts 1-/3-channel BGR (passed straight to ``cv2.inpaint``) and 4-channel
    BGRA: ``cv2.inpaint`` rejects 4 channels, so the alpha plane is split off,
    the BGR is inpainted, and alpha is re-attached unchanged.
    """
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        bgr = cv2.inpaint(image_bgr[:, :, :3], mask, radius, flag)
        return np.dstack([bgr, image_bgr[:, :, 3]])
    return cv2.inpaint(image_bgr, mask, radius, flag)


def lama_available() -> bool:
    """True when the optional LaMa-ONNX backend can run (onnxruntime installed)."""
    from .optional_deps import module_available

    return module_available("onnxruntime")


def _get_lama_session() -> object:
    """Load (once) the big-LaMa ONNX session, downloading the model on first use."""
    global _lama_session
    if _lama_session is not None:
        return _lama_session

    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=_LAMA_REPO, filename=_LAMA_FILE)
    logger.info("Loading LaMa-ONNX model: %s", model_path)
    _lama_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _lama_session


def erase_lama(image_bgr: NDArray[Any], mask: NDArray[Any]) -> NDArray[Any]:
    """Inpaint ``mask`` with big-LaMa via onnxruntime (CPU).

    LaMa runs at a fixed square input size. To preserve full-image resolution we
    crop a padded region around the mask, inpaint that crop at the model size,
    and paste only the masked pixels back -- so untouched areas stay pixel-exact.

    Like ``erase_cv2``, accepts 1-channel (grayscale) and 4-channel (BGRA) input:
    LaMa runs on 3-channel BGR, so grayscale is promoted to BGR (result demoted
    back) and a BGRA alpha plane is split off and re-attached unchanged. Without
    this the ``cv2.cvtColor(..., BGR2RGB)`` below would crash on grayscale and
    silently drop alpha on BGRA.
    """
    if image_bgr.ndim == 2:
        bgr = erase_lama(cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR), mask)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        bgr = erase_lama(np.ascontiguousarray(image_bgr[:, :, :3]), mask)
        return np.dstack([bgr, image_bgr[:, :, 3]])
    session = _get_lama_session()
    inp = session.get_inputs()  # type: ignore[attr-defined]
    img_name = inp[0].name
    mask_name = inp[1].name
    # Model declares a fixed square spatial size (e.g. 512); fall back to 512.
    dims = inp[0].shape
    size = next((d for d in reversed(dims) if isinstance(d, int) and d > 1), 512)

    h, w = image_bgr.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return image_bgr.copy()

    # Padded crop around the mask (context for the inpainter).
    pad = max(16, int(0.4 * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
    cx0, cy0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
    cx1, cy1 = min(w, int(xs.max()) + 1 + pad), min(h, int(ys.max()) + 1 + pad)
    crop = image_bgr[cy0:cy1, cx0:cx1]
    crop_mask = mask[cy0:cy1, cx0:cx1]
    ch, cw = crop.shape[:2]

    # Resize crop + mask to the model size, normalise to [0,1] RGB CHW.
    crop_rs = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    mask_rs = cv2.resize(crop_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    img_in = cv2.cvtColor(crop_rs, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_in = np.transpose(img_in, (2, 0, 1))[None]  # (1,3,size,size)
    mask_in = (mask_rs > 127).astype(np.float32)[None, None]  # (1,1,size,size), 1=hole

    out = session.run(None, {img_name: img_in, mask_name: mask_in})[0]  # type: ignore[attr-defined]
    out = np.asarray(out)[0]  # (3,size,size)
    out = np.transpose(out, (1, 2, 0))
    if float(out.max()) <= 1.5:  # model emits [0,1]; otherwise already [0,255]
        out = out * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    # Resize back to crop size and paste only the masked pixels.
    out_crop = cv2.resize(out_bgr, (cw, ch), interpolation=cv2.INTER_LINEAR)
    result = image_bgr.copy()
    region = result[cy0:cy1, cx0:cx1]
    paste = crop_mask > 127
    region[paste] = out_crop[paste]
    result[cy0:cy1, cx0:cx1] = region
    return result


def migan_available() -> bool:
    """True when the optional MI-GAN backend can run (onnxruntime installed)."""
    from .optional_deps import module_available

    return module_available("onnxruntime")


def _get_migan_session() -> object:
    """Load (once) the MI-GAN ONNX session, downloading the model on first use."""
    global _migan_session
    if _migan_session is not None:
        return _migan_session

    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=_MIGAN_REPO, filename=_MIGAN_FILE)
    logger.info("Loading MI-GAN ONNX model: %s", model_path)
    _migan_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _migan_session


def erase_migan(image_bgr: NDArray[Any], mask: NDArray[Any]) -> NDArray[Any]:
    """Inpaint ``mask`` (255 = erase) with MI-GAN via onnxruntime (CPU).

    The MI-GAN ONNX pipeline crops around the mask bbox internally and re-composites,
    so the full image is fed at native resolution. Only the masked pixels are pasted
    back, so untouched areas stay pixel-exact.

    Mask polarity: the shipped ``andraniksargsyan/migan`` ONNX expects 0 = hole
    (inpaint) / 255 = known (keep) -- the INVERSE of this package's 255-erase
    convention -- so the mask is inverted before feeding the model (corpus-validated
    2026-07; feeding 255=hole regenerates the whole frame into stripes).

    Like ``erase_lama``, accepts 1-channel (grayscale) and 4-channel (BGRA) input.
    """
    if image_bgr.ndim == 2:
        bgr = erase_migan(cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR), mask)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        bgr = erase_migan(np.ascontiguousarray(image_bgr[:, :, :3]), mask)
        return np.dstack([bgr, image_bgr[:, :, 3]])

    session = _get_migan_session()
    inp = session.get_inputs()  # type: ignore[attr-defined]
    img_name, mask_name = inp[0].name, inp[1].name
    h, w = image_bgr.shape[:2]

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(rgb, (2, 0, 1))[None].astype(np.uint8)  # (1,3,H,W)
    # invert to MI-GAN polarity: 255 where KNOWN (keep), 0 where hole (erase)
    known = (mask <= 127).astype(np.uint8) * 255
    mask_in = known[None, None]  # (1,1,H,W)

    out = session.run(None, {img_name: img_in, mask_name: mask_in})[0]  # type: ignore[attr-defined]
    res = np.transpose(np.asarray(out)[0], (1, 2, 0)).astype(np.uint8)  # (H',W',3) RGB
    if res.shape[:2] != (h, w):
        res = cv2.resize(res, (w, h), interpolation=cv2.INTER_LINEAR)
    out_bgr = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

    result = image_bgr.copy()
    hole = mask > 127
    result[hole] = out_bgr[hole]
    return result


def erase(
    image_bgr: NDArray[Any],
    *,
    boxes: list[tuple[int, int, int, int]] | None = None,
    mask: NDArray[Any] | None = None,
    backend: Backend = "cv2",
    dilate: int = 3,
    cv2_method: Literal["telea", "ns"] = "telea",
    cv2_radius: int = 6,
) -> NDArray[Any]:
    """Erase the given boxes (or mask) via the chosen inpainting backend.

    Provide either ``boxes`` (list of ``(x, y, w, h)``) or a precomputed ``mask``
    (uint8, 255 = erase). Returns an unmodified copy when nothing is selected.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    if mask is None:
        if not boxes:
            return image_bgr.copy()
        mask = boxes_to_mask(image_bgr.shape[:2], boxes, dilate=dilate)
    if not mask.any():
        return image_bgr.copy()

    if backend == "migan":
        if not migan_available():
            raise RuntimeError(
                "MI-GAN backend requires onnxruntime. Install the extra: pip install 'remove-ai-watermarks[migan]'"
            )
        return erase_migan(image_bgr, mask)
    if backend == "lama":
        if not lama_available():
            raise RuntimeError(
                "LaMa backend requires onnxruntime. Install the extra: pip install 'remove-ai-watermarks[lama]'"
            )
        return erase_lama(image_bgr, mask)
    return erase_cv2(image_bgr, mask, method=cv2_method, radius=cv2_radius)
