"""Universal region eraser: remove anything inside user-given boxes via inpainting.

Position- and content-agnostic. You supply the rectangle(s); the eraser inpaints
whatever is inside, so it removes any visible logo / watermark / object regardless
of color, style, or location. Localization is the user's responsibility (pass the
box); restoration runs on CPU. This is the universal fallback for marks the
registered visible detectors do not cover.

Backends:
  - ``cv2`` (default): ``cv2.inpaint`` (Telea / Navier-Stokes). Instant, no extra
    dependencies, lower quality on large or textured regions.
  - ``migan`` (optional, extra ``migan``): MI-GAN via onnxruntime
    (``andraniksargsyan/migan``, MIT). CPU, ~28 MB model, ~0.19 s/call -- the
    droplet-friendly tier: near-big-LaMa quality on small marks. Model downloaded
    on first use. Like ``lama`` it crops a padded region around the mask before
    inference (at native resolution -- MI-GAN takes arbitrary dims), so peak RAM is
    bounded by the mark size (~0.6-0.9 GB) rather than scaling with the image, which
    is what lets a memory-tight host run it on a large upload.
  - ``lama`` (optional, extra ``lama``): big-LaMa via onnxruntime
    (``Carve/LaMa-ONNX``, Apache-2.0). CPU, resolution-robust, best quality on
    texture but ~200 MB model and ~4.7 GB peak RAM (too heavy for a small host).
    The model is downloaded on first use and cached by huggingface_hub; it is
    never bundled in this repo.

The per-backend RAM and wall-time figures above are reproduced by
``scripts/resource_ceilings.py`` (fresh subprocess per measurement, synthetic inputs).
Re-run it before changing any of them.
"""

# cv2/numpy boundary: cv2 ships no usable type info, so strict pyright cannot know
# its array element types. Relax the unknown-type rules for this file only; the
# public signatures are still annotated with NDArray[Any].
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

Backend = Literal["cv2", "lama", "migan"]

_LAMA_REPO = "Carve/LaMa-ONNX"
_LAMA_FILE = "lama_fp32.onnx"

_MIGAN_REPO = "andraniksargsyan/migan"
_MIGAN_FILE = "migan.onnx"

# Cached onnxruntime sessions, keyed by backend name (loading is expensive; reuse
# across calls).
_sessions: dict[str, object] = {}


@dataclass(frozen=True)
class _LearnedBackend:
    """One optional model-backed fill: its human label and the module-level names of
    its availability probe and erase function.

    NAMES, not function objects: the fallback tests monkeypatch these by attribute on
    the module, and a table holding bound references would not see the patch.
    """

    label: str
    available: str
    erase: str


# The learned tier, best-quality first. `resolve_backend`'s preference order and the
# CLI's choices are separate literals by design (see FILL_BACKENDS below), but the
# availability probe, install hint and dispatch all read this one table.
_LEARNED_BACKENDS: dict[str, _LearnedBackend] = {
    "lama": _LearnedBackend("LaMa", "lama_available", "erase_lama"),
    "migan": _LearnedBackend("MI-GAN", "migan_available", "erase_migan"),
}

# Every fill backend this module can execute, plus the caller-facing `auto`. Kept as a
# literal tuple (not derived through an import) so `watermark_registry` and the CLI can
# state their choices without importing this cv2-loading module at their import time --
# a test pins the two in sync.
FILL_BACKENDS: tuple[str, ...] = ("auto", "cv2", "migan", "lama")


def _full_scale(image: NDArray[Any]) -> float:
    """Full-scale value of an integer image dtype: 255 for uint8, 65535 for uint16."""
    if image.dtype.kind not in "ui":
        raise RuntimeError(
            f"Unsupported image depth {image.dtype}: the fill backends take integer "
            "images (8- or 16-bit). Convert the source before erasing."
        )
    return float(np.iinfo(image.dtype).max)


def _erase_via_uint8(
    backend: Callable[[NDArray[Any], NDArray[Any]], NDArray[Any]],
    image_bgr: NDArray[Any],
    mask: NDArray[Any],
) -> NDArray[Any]:
    """Run a uint8-only backend on a deeper image without flattening the whole frame.

    Two of the three backends cannot see more than 8 bits: ``cv2.inpaint`` takes
    8-bit 3-channel colour (its 16-bit support is single-channel only), and the
    shipped MI-GAN ONNX declares a uint8 input tensor. A 16-bit source therefore has
    to be narrowed for the fill itself -- but only for the fill. The filled pixels are
    written back into the original array, so everything outside the mask stays
    bit-exact at its original depth, which is the same guarantee the learned backends
    already make by pasting only masked pixels.

    LaMa is the exception and does not come through here: its model is float32, so
    ``erase_lama`` carries the source depth end to end.
    """
    scale = _full_scale(image_bgr)
    work = np.clip(image_bgr.astype(np.float32) * (255.0 / scale), 0, 255).astype(np.uint8)
    filled = backend(work, mask)
    result = image_bgr.copy()
    hole = mask > 127
    result[hole] = np.clip(filled[hole].astype(np.float32) * (scale / 255.0), 0, scale).astype(image_bgr.dtype)
    return result


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


def _padded_crop_box(
    mask: NDArray[Any], h: int, w: int, *, pad_frac: float, pad_min: int
) -> tuple[int, int, int, int] | None:
    """Bounding box of the set mask pixels, padded and clamped to the image.

    Returns ``(x0, y0, x1, y1)`` or ``None`` when the mask is empty. Both learned
    backends crop to this box so the ONNX working set is bounded by the mark size
    rather than the whole image; ``pad_frac``/``pad_min`` tune how much surrounding
    context the inpainter sees (LaMa then resizes the crop to its fixed square,
    MI-GAN feeds it at native resolution).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    pad = max(pad_min, int(pad_frac * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
    x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
    x1, y1 = min(w, int(xs.max()) + 1 + pad), min(h, int(ys.max()) + 1 + pad)
    return x0, y0, x1, y1


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

    A deeper-than-8-bit source is filled through :func:`_erase_via_uint8`: OpenCV's
    inpaint takes 16-bit only as a single channel, so 16-bit colour would otherwise
    raise a bare ``icvInpaint`` "Unsupported format" error.
    """
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    if image_bgr.dtype != np.uint8:
        return _erase_via_uint8(lambda img, m: erase_cv2(img, m, method=method, radius=radius), image_bgr, mask)
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        bgr = cv2.inpaint(image_bgr[:, :, :3], mask, radius, flag)
        return np.dstack([bgr, image_bgr[:, :, 3]])
    return cv2.inpaint(image_bgr, mask, radius, flag)


def lama_available() -> bool:
    """True when the optional LaMa-ONNX backend can run (onnxruntime installed)."""
    from .optional_deps import module_available

    return module_available("onnxruntime")


def _get_session(name: str, repo_id: str, filename: str, label: str) -> object:
    """Load (once) an ONNX session, downloading the model on first use."""
    cached = _sessions.get(name)
    if cached is not None:
        return cached

    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    logger.info("Loading %s model: %s", label, model_path)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    _sessions[name] = session
    return session


def _get_lama_session() -> object:
    """The big-LaMa ONNX session (kept as a named seam the tests monkeypatch)."""
    return _get_session("lama", _LAMA_REPO, _LAMA_FILE, "LaMa-ONNX")


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

    Depth is carried end to end: the model works in float32, which is finer than any
    integer source, so the normalisation divides by the SOURCE's full scale (65535 for
    a 16-bit input, not a hardcoded 255) and the result is written back at the source
    dtype. Dividing a 16-bit image by 255 pushed every value past 1.0 and the uint8
    cast then wrapped it, which filled the mask with near-black pixels -- a silent
    wrong answer rather than a crash.
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
    box = _padded_crop_box(mask, h, w, pad_frac=0.4, pad_min=16)
    if box is None:
        return image_bgr.copy()
    cx0, cy0, cx1, cy1 = box
    crop = image_bgr[cy0:cy1, cx0:cx1]
    crop_mask = mask[cy0:cy1, cx0:cx1]
    ch, cw = crop.shape[:2]

    # Resize crop + mask to the model size, normalize to [0,1] RGB CHW.
    scale = _full_scale(image_bgr)
    crop_rs = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    mask_rs = cv2.resize(crop_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    img_in = cv2.cvtColor(crop_rs, cv2.COLOR_BGR2RGB).astype(np.float32) / scale
    img_in = np.transpose(img_in, (2, 0, 1))[None]  # (1,3,size,size)
    mask_in = (mask_rs > 127).astype(np.float32)[None, None]  # (1,1,size,size), 1=hole

    out = session.run(None, {img_name: img_in, mask_name: mask_in})[0]  # type: ignore[attr-defined]
    out = np.asarray(out)[0]  # (3,size,size)
    out = np.transpose(out, (1, 2, 0))
    if float(out.max()) <= 1.5:  # model emits [0,1]
        out = out * scale
    elif scale != 255.0:  # model emitted 8-bit levels; lift them to the source's scale
        out = out * (scale / 255.0)
    out = np.clip(out, 0, scale).astype(image_bgr.dtype)
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
    """True when the optional MI-GAN backend can run (onnxruntime installed).

    Deliberately a separate function from :func:`lama_available` even though both
    currently reduce to the same onnxruntime probe: they are independent capability
    questions and `auto` resolves them separately (see
    ``watermark_registry.preferred_inpaint_backend``).
    """
    from .optional_deps import module_available

    return module_available("onnxruntime")


def _get_migan_session() -> object:
    """The MI-GAN ONNX session (kept as a named seam the tests monkeypatch)."""
    return _get_session("migan", _MIGAN_REPO, _MIGAN_FILE, "MI-GAN ONNX")


def erase_migan(image_bgr: NDArray[Any], mask: NDArray[Any]) -> NDArray[Any]:
    """Inpaint ``mask`` (255 = erase) with MI-GAN via onnxruntime (CPU).

    Like ``erase_lama``, we crop a padded region around the mask, feed only that crop
    to the ONNX model, and paste only the masked pixels back -- so untouched areas stay
    pixel-exact and the ONNX working set is bounded by the MARK size, not the whole
    image. MI-GAN accepts arbitrary spatial dims, so (unlike LaMa's fixed 512 square)
    the crop is fed at NATIVE resolution -- no resize, so the mark is seen at full
    scale. Feeding the whole frame instead made peak RAM scale with the image
    (~0.6 GB at 4 MP up to ~2.4 GB at 25 MP, measured 2026-07); cropping holds the
    inpaint working set roughly constant, so a memory-tight host (e.g. a 1-2 GB web
    worker) can run MI-GAN on a 25 MP upload. Cropping does not degrade the fill:
    a small mark only needs local context, and on real marks the cropped fill is on
    par with -- sometimes cleaner than -- the full-frame fill (a tighter view gives
    the GAN less room to hallucinate large background structure).

    Mask polarity: the shipped ``andraniksargsyan/migan`` ONNX expects 0 = hole
    (inpaint) / 255 = known (keep) -- the INVERSE of this package's 255-erase
    convention -- so the mask is inverted before feeding the model. Feeding 255=hole
    regenerates the whole frame into stripes.

    Accepts 1-channel (grayscale) and 4-channel (BGRA) input. A deeper-than-8-bit
    source goes through :func:`_erase_via_uint8`: the ONNX input tensor is uint8, so
    the model cannot see more depth than that whatever we hand it.
    """
    if image_bgr.dtype != np.uint8:
        return _erase_via_uint8(erase_migan, image_bgr, mask)
    if image_bgr.ndim == 2:
        bgr = erase_migan(cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR), mask)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        bgr = erase_migan(np.ascontiguousarray(image_bgr[:, :, :3]), mask)
        return np.dstack([bgr, image_bgr[:, :, 3]])

    h, w = image_bgr.shape[:2]
    # 2x the mark size (min 256 px) gives ample local context while a small corner
    # mask keeps the crop small regardless of the full image resolution; the crop is
    # clamped to the image, so a mark already spanning the frame degrades to the
    # whole image (the old behavior) rather than erroring.
    box = _padded_crop_box(mask, h, w, pad_frac=2.0, pad_min=256)
    if box is None:
        return image_bgr.copy()
    cx0, cy0, cx1, cy1 = box
    crop = np.ascontiguousarray(image_bgr[cy0:cy1, cx0:cx1])
    crop_mask = mask[cy0:cy1, cx0:cx1]
    ch, cw = crop.shape[:2]

    session = _get_migan_session()
    inp = session.get_inputs()  # type: ignore[attr-defined]
    img_name, mask_name = inp[0].name, inp[1].name

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(rgb, (2, 0, 1))[None].astype(np.uint8)  # (1,3,ch,cw)
    # invert to MI-GAN polarity: 255 where KNOWN (keep), 0 where hole (erase)
    known = (crop_mask <= 127).astype(np.uint8) * 255
    mask_in = known[None, None]  # (1,1,ch,cw)

    out = session.run(None, {img_name: img_in, mask_name: mask_in})[0]  # type: ignore[attr-defined]
    res = np.transpose(np.asarray(out)[0], (1, 2, 0)).astype(np.uint8)  # (ch',cw',3) RGB
    if res.shape[:2] != (ch, cw):
        res = cv2.resize(res, (cw, ch), interpolation=cv2.INTER_LINEAR)
    out_bgr = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

    result = image_bgr.copy()
    region = result[cy0:cy1, cx0:cx1]
    hole = crop_mask > 127
    region[hole] = out_bgr[hole]
    result[cy0:cy1, cx0:cx1] = region
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

    learned = _LEARNED_BACKENDS.get(backend)
    if learned is not None:
        # Probe and erase by MODULE NAME, not by the captured function object: the
        # availability probes and the erase functions are monkeypatched by name in the
        # fallback tests, and a table of bound references would not see those patches.
        if not globals()[learned.available]():
            raise RuntimeError(
                f"{learned.label} backend requires onnxruntime. "
                f"Install the extra: pip install 'remove-ai-watermarks[{backend}]'"
            )
        return globals()[learned.erase](image_bgr, mask)
    # cv2 and anything unrecognized (including "auto", which a library caller may pass
    # straight through) degrade to the classical fill rather than raising.
    return erase_cv2(image_bgr, mask, method=cv2_method, radius=cv2_radius)
