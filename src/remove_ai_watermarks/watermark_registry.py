"""Registry of known visible watermarks.

A single catalog that ties each known visible mark to (a) where it usually sits,
(b) how to recognize it there, and (c) how to remove it. One pass over the
registry detects every known mark in its usual place and removes the ones
present.

**Reverse-alpha only.** A known mark is a fixed semi-transparent overlay, so it
is removed by inverting the alpha blend against a captured alpha map
(``original = (wm - a*logo)/(1-a)``) -- exact recovery of the true pixels, not an
inpaint guess. Detection is consistent with that: each mark is recognized by
matching its known shape/template (the thing we invert), not by heuristics. A
mark is therefore listed here only once a real alpha map has been captured for
it; everything else (arbitrary logos/objects) is the user-directed
``erase --region`` tool, not this catalog.

Entries:
  - ``gemini`` -- Google Gemini / Nano Banana sparkle, bottom-right.
  - ``doubao`` -- ByteDance Doubao "豆包AI生成" text strip, bottom-right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

# cv2 method for the Gemini reverse-alpha edge-residual cleanup (not a standalone
# remover): "ns" / "telea".
InpaintMethod = Literal["telea", "ns"]
Region = tuple[int, int, int, int]


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
    location: str  # usual place, human-readable ("bottom-right")
    in_auto: bool  # participate in `--mark auto` scanning
    recovery: str  # removal strategy (all reverse-alpha today)
    _detect: Callable[[NDArray[Any]], MarkDetection]
    _remove: Callable[..., tuple[NDArray[Any], Region | None]]

    def detect(self, image: NDArray[Any]) -> MarkDetection:
        return self._detect(image)

    def remove(
        self,
        image: NDArray[Any],
        *,
        inpaint_method: InpaintMethod = "ns",
        inpaint: bool = True,
        inpaint_strength: float = 0.85,
        force: bool = False,
    ) -> tuple[NDArray[Any], Region | None]:
        """Remove this mark by reverse-alpha; returns ``(result, cleared_region)``
        (region for clearing alpha on save, or None if nothing was removed).

        ``inpaint`` / ``inpaint_strength`` / ``inpaint_method`` tune the Gemini
        reverse-alpha edge-residual cleanup only. ``force`` removes at the mark's
        usual location even without a positive detection (the ``--no-detect`` path).
        """
        return self._remove(image, inpaint_method, inpaint, inpaint_strength, force)


# Gemini-sparkle confidence above which the registry treats it as a confident
# detection for arbitration. Matches identify's corpus-validated sparkle
# threshold (0.5): the gemini engine's own detect flag uses a looser internal
# threshold and weakly fires (~0.36) on unrelated bottom-right text (e.g. the
# Doubao mark), which would otherwise let it hijack `--mark auto`. 0.5 gives 0
# false positives on the corpus.
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
        else:  # pragma: no cover - guarded by the registry keys
            raise KeyError(key)
    return _engines[key]


def _gemini_detect(image: NDArray[Any]) -> MarkDetection:
    d = _engine("gemini").detect_watermark(image)
    detected = bool(d.detected) and d.confidence >= _GEMINI_AUTO_MIN_CONF
    return MarkDetection("gemini", "Google Gemini sparkle", "bottom-right", detected, d.confidence, d.region)


def _gemini_remove(
    image: NDArray[Any], inpaint_method: InpaintMethod, inpaint: bool, strength: float, force: bool
) -> tuple[NDArray[Any], Region | None]:
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
    # own residual inpaint cleans that seam (part of its reverse-alpha pipeline).
    if inpaint:
        result = engine.inpaint_residual(result, det.region, strength=strength, method=inpaint_method)
    return result, det.region


def _doubao_detect(image: NDArray[Any]) -> MarkDetection:
    d = _engine("doubao").detect(image)
    return MarkDetection("doubao", "Doubao 豆包AI生成 text", "bottom-right", d.detected, d.confidence, d.region)


def _doubao_remove(
    image: NDArray[Any], _inpaint_method: InpaintMethod, _inpaint: bool, _strength: float, force: bool
) -> tuple[NDArray[Any], Region | None]:
    # Reverse-alpha only: apply when the mark is present AND the resolution is in
    # the alpha map's calibrated band. Outside it we do NOT inpaint (no
    # hallucination) -- removal is skipped until a capture for that resolution.
    engine = _engine("doubao")
    det = engine.detect(image)
    if (det.detected or force) and engine.reverse_alpha_available(image):
        return engine.remove_watermark_reverse_alpha(image), (det.region if det.detected else None)
    return image.copy(), None


_REGISTRY: tuple[KnownMark, ...] = (
    KnownMark("gemini", "Google Gemini sparkle", "bottom-right", True, "reverse-alpha", _gemini_detect, _gemini_remove),
    KnownMark(
        "doubao", "Doubao 豆包AI生成 text", "bottom-right", True, "reverse-alpha", _doubao_detect, _doubao_remove
    ),
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
    ``include_explicit=False`` scans only the ``in_auto`` marks -- the set used
    by ``--mark auto``.
    """
    return [m.detect(image) for m in _REGISTRY if include_explicit or m.in_auto]


def best_auto_mark(image: NDArray[Any]) -> MarkDetection | None:
    """The highest-confidence detected ``in_auto`` mark, or None if none fired."""
    fired = [d for d in detect_marks(image, include_explicit=False) if d.detected]
    return max(fired, key=lambda d: d.confidence) if fired else None
