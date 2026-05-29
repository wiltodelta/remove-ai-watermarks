"""Registry of known visible watermarks.

A single catalog that ties each known visible mark to (a) where it usually sits,
(b) how to recognize it there, and (c) how to remove it while reconstructing the
background. One pass over the registry detects every known mark in its usual
place and removes the ones present.

Adding a new known mark = one ``KnownMark`` entry (plus its engine). Today's
entries:

  - ``gemini``  -- Google Gemini sparkle, bottom-right. Removed by reverse-alpha
    blending against a captured alpha map (exact background recovery).
  - ``doubao``  -- ByteDance Doubao "豆包AI生成" text strip, bottom-right.
    Removed by mask + inpaint (background is reconstructed, not recovered).
  - ``samsung`` -- Samsung Galaxy AI "AI-generated content" badge, bottom-left.
    Removed by mask + inpaint. Explicit-only (``in_auto=False``): the faint,
    localized badge can't be detected precisely enough for unattended routing.

Removal quality ladder (per mark, best first): reverse-alpha (exact, needs a
captured alpha map) > model inpaint (LaMa / diffusion, plausible) > cv2 inpaint
(cheap). Only ``gemini`` has an alpha map today; the others use cv2 inpaint and
leave the seam for a better backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

# Inpaint method for the mask-based removers' cv2 path (Telea / NS).
InpaintMethod = Literal["telea", "ns"]
# Background-reconstruction backend for the mask-based marks (gemini ignores it --
# it has an exact reverse-alpha map). ``lama`` (big-LaMa via region_eraser)
# reconstructs texture/structure far better than ``cv2`` (which only diffuses
# neighbours and smears on busy backgrounds), at ~3.5 GB RAM / ~5 s CPU. ``cv2``
# is the instant, zero-dependency fallback. The CLI default resolves ``lama`` when
# onnxruntime is installed, else ``cv2``. For ``lama`` the mask is the FILLED glyph
# bounding box (not the sparse glyph pixels): LaMa reconstructs a clean region,
# whereas a sparse mask leaves the glyphs' anti-aliased dark edges behind.
Backend = Literal["cv2", "lama"]
Region = tuple[int, int, int, int]


def default_backend() -> Backend:
    """Preferred backend: ``lama`` when its optional dep is installed, else ``cv2``."""
    from remove_ai_watermarks.region_eraser import lama_available

    return "lama" if lama_available() else "cv2"


@dataclass(frozen=True)
class MarkDetection:
    """Uniform detection result for a known mark (across heterogeneous engines)."""

    key: str
    label: str
    location: str
    detected: bool
    confidence: float
    region: Region


@dataclass(frozen=True)
class KnownMark:
    """A known visible watermark: where it lives, how to find and remove it."""

    key: str
    label: str
    location: str  # usual place, human-readable ("bottom-right", "bottom-left")
    in_auto: bool  # participate in `--mark auto` scanning
    recovery: str  # removal strategy: "reverse-alpha" (exact) | "inpaint" (reconstructed)
    _detect: Callable[[NDArray[Any]], MarkDetection]
    _remove: Callable[..., tuple[NDArray[Any], Region | None]]

    def detect(self, image: NDArray[Any]) -> MarkDetection:
        return self._detect(image)

    def remove(
        self,
        image: NDArray[Any],
        *,
        backend: Backend = "cv2",
        inpaint_method: InpaintMethod = "ns",
        inpaint: bool = True,
        inpaint_strength: float = 0.85,
        force: bool = False,
    ) -> tuple[NDArray[Any], Region | None]:
        """Remove this mark; returns ``(result, cleared_region)``. The region is
        the watermark bbox (for clearing alpha on save), or None if not removed.

        ``backend`` selects background reconstruction for the mask-based marks
        (``lama`` > ``cv2``); ``gemini`` ignores it (exact reverse-alpha).
        ``inpaint`` / ``inpaint_strength`` tune the Gemini residual-inpaint polish.
        ``force`` removes at the mark's usual location even without a positive
        detection (the ``--no-detect`` path).
        """
        return self._remove(image, backend, inpaint_method, inpaint, inpaint_strength, force)


# Gemini-sparkle confidence above which the registry treats it as a confident
# detection for arbitration. Matches identify's corpus-validated sparkle
# threshold (0.5): the gemini engine's own detect flag uses a looser internal
# threshold and weakly fires (~0.36) on unrelated bottom-right text (e.g. the
# Doubao mark), which would otherwise let it hijack `--mark auto` and apply
# reverse-alpha at the wrong place. 0.5 gives 0 false positives on the corpus.
_GEMINI_AUTO_MIN_CONF = 0.5

# ── Engine adapters (lazy singletons; engines are cv2-only, no model load) ──

_engines: dict[str, Any] = {}


def _engine(key: str) -> Any:
    if key not in _engines:
        if key == "gemini":
            from remove_ai_watermarks.gemini_engine import GeminiEngine

            _engines[key] = GeminiEngine()
        elif key == "doubao":
            from remove_ai_watermarks.doubao_engine import DoubaoEngine

            _engines[key] = DoubaoEngine()
        elif key == "samsung":
            from remove_ai_watermarks.samsung_engine import SamsungEngine

            _engines[key] = SamsungEngine()
        else:  # pragma: no cover - guarded by the registry keys
            raise KeyError(key)
    return _engines[key]


def _gemini_detect(image: NDArray[Any]) -> MarkDetection:
    d = _engine("gemini").detect_watermark(image)
    detected = bool(d.detected) and d.confidence >= _GEMINI_AUTO_MIN_CONF
    return MarkDetection("gemini", "Google Gemini sparkle", "bottom-right", detected, d.confidence, d.region)


def _gemini_remove(
    image: NDArray[Any], _backend: Backend, inpaint_method: InpaintMethod, inpaint: bool, strength: float, force: bool
) -> tuple[NDArray[Any], Region | None]:
    # Gemini ignores ``backend``: reverse-alpha against the captured map is the
    # exact recovery, strictly better than any inpaint.
    engine = _engine("gemini")
    det = engine.detect_watermark(image)
    if not det.detected:
        if not force:
            return image.copy(), None
        # Forced (--no-detect): remove at the default sparkle slot for the size.
        from remove_ai_watermarks.gemini_engine import get_watermark_config

        h, w = image.shape[:2]
        cfg = get_watermark_config(w, h)
        px, py = cfg.get_position(w, h)
        region = (px, py, cfg.logo_size, cfg.logo_size)
        result = engine.remove_watermark_custom(image, region)
        if inpaint:
            result = engine.inpaint_residual(result, region, strength=strength, method=inpaint_method)
        return result, region
    result = engine.remove_watermark(image)
    # Reverse-alpha leaves a faint residual at the sparkle edge; the engine's
    # residual inpaint cleans it (same polish the CLI applied before the registry).
    if inpaint:
        result = engine.inpaint_residual(result, det.region, strength=strength, method=inpaint_method)
    return result, det.region


def _lama_box_inpaint(engine: Any, image: NDArray[Any]) -> NDArray[Any] | None:
    """Inpaint the FILLED glyph bounding box with big-LaMa; None if no glyphs.

    A filled box (not the sparse glyph mask) is essential for LaMa: it then
    reconstructs a clean region, whereas a sparse mask leaves the glyphs'
    unmasked anti-aliased dark edges behind (verified on a wood-grain sample).
    """
    import numpy as np

    from remove_ai_watermarks.region_eraser import erase

    loc = engine.locate(image)
    gmask = engine.extract_mask(image, loc)
    ys, xs = np.where(gmask > 0)
    if len(xs) == 0:
        return None
    pad, h, w = 6, image.shape[0], image.shape[1]
    x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
    x1, y1 = min(w, int(xs.max()) + pad), min(h, int(ys.max()) + pad)
    boxmask = np.zeros((h, w), np.uint8)
    boxmask[y0:y1, x0:x1] = 255
    return erase(image, mask=boxmask, backend="lama")


def _glyph_remove(
    key: str, image: NDArray[Any], backend: Backend, inpaint_method: InpaintMethod, force: bool, *, gated: bool
) -> tuple[NDArray[Any], Region | None]:
    """Shared removal for the mask-based glyph marks (doubao/samsung).

    ``gated``: skip removal when the mark is not detected (and not forced) -- used
    for Samsung, whose faint badge mustn't be inpainted on a false positive.
    """
    engine = _engine(key)
    det = engine.detect(image)
    region = det.region if det.detected else None
    if gated and not det.detected and not force:
        return image.copy(), None
    if backend == "lama":
        from remove_ai_watermarks.region_eraser import lama_available

        if lama_available():
            out = _lama_box_inpaint(engine, image)
            if out is not None:
                return out, region
        # onnxruntime absent or no glyphs found -> fall through to the cv2 path.
    return engine.remove_watermark(image, inpaint_method=inpaint_method, force=force), region


def _doubao_detect(image: NDArray[Any]) -> MarkDetection:
    d = _engine("doubao").detect(image)
    return MarkDetection("doubao", "Doubao 豆包AI生成 text", "bottom-right", d.detected, d.confidence, d.region)


def _doubao_remove(
    image: NDArray[Any], backend: Backend, inpaint_method: InpaintMethod, _inpaint: bool, _strength: float, force: bool
) -> tuple[NDArray[Any], Region | None]:
    return _glyph_remove("doubao", image, backend, inpaint_method, force, gated=False)


def _samsung_detect(image: NDArray[Any]) -> MarkDetection:
    d = _engine("samsung").detect(image)
    return MarkDetection("samsung", "Samsung Galaxy AI badge", "bottom-left", d.detected, d.confidence, d.region)


def _samsung_remove(
    image: NDArray[Any], backend: Backend, inpaint_method: InpaintMethod, _inpaint: bool, _strength: float, force: bool
) -> tuple[NDArray[Any], Region | None]:
    return _glyph_remove("samsung", image, backend, inpaint_method, force, gated=True)


_REGISTRY: tuple[KnownMark, ...] = (
    KnownMark("gemini", "Google Gemini sparkle", "bottom-right", True, "reverse-alpha", _gemini_detect, _gemini_remove),
    KnownMark("doubao", "Doubao 豆包AI生成 text", "bottom-right", True, "inpaint", _doubao_detect, _doubao_remove),
    KnownMark("samsung", "Samsung Galaxy AI badge", "bottom-left", False, "inpaint", _samsung_detect, _samsung_remove),
)


def known_marks() -> tuple[KnownMark, ...]:
    """All registered known visible watermarks."""
    return _REGISTRY


def mark_keys() -> list[str]:
    """Keys of all registered marks (for CLI choices)."""
    return [m.key for m in _REGISTRY]


def get_mark(key: str) -> KnownMark:
    """Look up a known mark by key (raises KeyError if unknown)."""
    for m in _REGISTRY:
        if m.key == key:
            return m
    raise KeyError(key)


def detect_marks(image: NDArray[Any], *, include_explicit: bool = True) -> list[MarkDetection]:
    """Detect every known mark in its usual place.

    Returns one MarkDetection per scanned mark (``detected`` flags which fired).
    ``include_explicit=False`` scans only the ``in_auto`` marks (skips
    explicit-only marks like Samsung) -- the set used by ``--mark auto``.
    """
    return [m.detect(image) for m in _REGISTRY if include_explicit or m.in_auto]


def best_auto_mark(image: NDArray[Any]) -> MarkDetection | None:
    """The highest-confidence detected ``in_auto`` mark, or None if none fired."""
    fired = [d for d in detect_marks(image, include_explicit=False) if d.detected]
    return max(fired, key=lambda d: d.confidence) if fired else None
