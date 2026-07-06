"""Registry of known visible watermarks.

A single catalog that ties each known visible mark to (a) where it usually sits,
(b) how to recognize it there, and (c) how to remove it. One pass over the
registry detects every known mark in its usual place and removes the ones
present.

**Reverse-alpha based.** A known mark is a fixed semi-transparent overlay, so it
is removed by inverting the alpha blend against a captured alpha map
(``original = (wm - a*logo)/(1-a)``) -- recovering the true pixels rather than
inpainting a guess. Gemini and Doubao recover exactly with no inpaint at native on
bright/flat backgrounds (Gemini falls back to inpainting the sparkle footprint when
reverse-alpha would over-subtract on a dark background -- issue #30, see gemini_engine);
Jimeng adds a thin residual inpaint over the glyph footprint to clear the outline
its per-image render variation leaves behind (still seeded by the reverse-alpha
recovery, not a blind inpaint). Detection is consistent with that: each mark is
recognized by matching its known shape/template (the thing we invert), not by
heuristics. A mark is therefore listed here only once a real alpha map has been
captured for it; everything else (arbitrary logos/objects) is the user-directed
``erase --region`` tool, not this catalog.

Entries:
  - ``gemini`` -- Google Gemini / Nano Banana sparkle, bottom-right.
  - ``doubao`` -- ByteDance Doubao "豆包AI生成" text strip, bottom-right.
  - ``jimeng`` -- ByteDance Jimeng / Dreamina "★ 即梦AI" wordmark, bottom-right.
  - ``samsung`` -- Samsung Galaxy AI "Contenuti generati dall'AI" strip, bottom-left.
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

# Removal method selection for the visible pass. ``auto`` prefers the inpaint
# fallback (LaMa footprint) when the ``lama`` extra is installed, else reverse-alpha
# for marks with a captured alpha map (cv2 inpaint for capture-less marks).
RemovalMethod = Literal["auto", "reverse-alpha", "inpaint"]


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
    recovery: str  # default removal strategy label ("reverse-alpha")
    _detect: Callable[[NDArray[Any]], MarkDetection]
    _remove: Callable[..., tuple[NDArray[Any], Region | None]]
    has_capture: bool = True  # a captured alpha map exists (reverse-alpha possible)

    def detect(self, image: NDArray[Any]) -> MarkDetection:
        return self._detect(image)

    def remove(
        self,
        image: NDArray[Any],
        *,
        method: RemovalMethod = "auto",
        inpaint_method: InpaintMethod = "ns",
        inpaint: bool = True,
        inpaint_strength: float = 0.85,
        force: bool = False,
    ) -> tuple[NDArray[Any], Region | None]:
        """Remove this mark; returns ``(result, region)`` where ``region`` is the
        removed mark's bbox (for residual-inpaint positioning), or None if nothing
        was removed. NB: the CLI does NOT use ``region`` to clear alpha on save --
        that zeroing caused the issue-#30 white box.

        ``method`` selects the removal path: ``reverse-alpha`` recovers the true
        pixels from the captured alpha map (lighter, exact, better on structured
        backgrounds); ``inpaint`` erases the footprint with LaMa (or cv2 when the
        ``lama`` extra is absent), needing no capture; ``auto`` (default) prefers
        inpaint when LaMa is installed and falls back to reverse-alpha otherwise
        (cv2 inpaint for capture-less marks). ``inpaint``/``inpaint_strength``/
        ``inpaint_method`` tune the Gemini reverse-alpha edge-residual cleanup only.
        ``force`` removes at the mark's usual location even without a positive
        detection (the ``--no-detect`` path).
        """
        if resolve_removal_method(method, self.has_capture) == "inpaint":
            return _inpaint_remove(self, image, force)
        return self._remove(image, inpaint_method, inpaint, inpaint_strength, force)


# Single source of truth for the Gemini-sparkle "trust this as a real mark"
# confidence, shared by BOTH the removal arbitration here (`best_auto_mark` /
# `_gemini_detect`) and the provenance detector in `identify` (which imports it
# as its sparkle threshold). Defining it once removes the detect-vs-remove
# threshold drift the retained-corpus mining surfaced (2026-06-20): identify
# would report a sparkle while removal declined it, or vice versa, whenever the
# two independently-maintained 0.5 constants fell out of step. Now they cannot.
#
# Value 0.5 is corpus-validated: the gemini engine's own `detected` flag uses a
# looser internal threshold (0.35) and weakly fires (~0.36-0.42) on unrelated
# bottom-right text -- a real Doubao mark scores ~0.40-0.42 as a gemini match,
# and its core-ring brightness margin is HIGHER than a genuine faint sparkle's,
# so neither confidence nor the brightness gate separates them in the [0.35, 0.5)
# band. Lowering this gate to recover faint sparkles was evaluated against that
# band (2026-06-20) and REJECTED: it cannot be done without re-admitting the
# Doubao-text / content false positives, trading a rare miss for false-positive
# removals on clean images. The band below the gate is therefore intentionally
# left to the higher-strength / metadata paths. 0.5 gives 0 false positives on
# the corpus.
GEMINI_SPARKLE_TRUST_CONF = 0.5
_GEMINI_AUTO_MIN_CONF = GEMINI_SPARKLE_TRUST_CONF

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
        elif key == "jimeng":
            from remove_ai_watermarks.jimeng_engine import JimengEngine

            _engines[key] = JimengEngine()
        elif key == "samsung":
            from remove_ai_watermarks.samsung_engine import SamsungEngine

            _engines[key] = SamsungEngine()
        elif key == "jimeng_pill":
            from remove_ai_watermarks.pill_engine import PillEngine

            _engines[key] = PillEngine()
        else:  # pragma: no cover - guarded by the registry keys
            raise KeyError(key)
    return _engines[key]


def inpaint_model_available() -> bool:
    """True when any ONNX inpaint-model backend (MI-GAN or big-LaMa) can run."""
    from remove_ai_watermarks import region_eraser

    return region_eraser.migan_available() or region_eraser.lama_available()


def preferred_inpaint_backend() -> str:
    """Backend used by the inpaint fallback: MI-GAN (light, droplet-friendly, the
    default) when its ONNX runtime is available, else cv2. big-LaMa is NOT auto-
    selected -- it is a heavier explicit opt-in via ``erase --backend lama`` (both
    models run on onnxruntime, so availability alone cannot express the user's
    intent; the light model is the safe default)."""
    from remove_ai_watermarks import region_eraser

    return "migan" if region_eraser.migan_available() else "cv2"


def resolve_removal_method(method: RemovalMethod, has_capture: bool) -> Literal["reverse-alpha", "inpaint"]:
    """Resolve the requested method to a concrete one. A capture-less mark has no
    alpha map, so it can only be inpainted -- even explicit ``reverse-alpha`` falls
    back to inpaint there.

    ``auto`` uses **reverse-alpha for capture marks** and **inpaint for capture-less**.
    Reverse-alpha recovers the true pixels under the mark (measured cleaner than
    MI-GAN inpaint on the capture marks -- doubao/gemini/jimeng -- especially on
    structured backgrounds, and it needs no model / RAM), so it stays the default
    where a capture exists; inpaint is reserved for marks that have no alpha map
    (the Jimeng pill). ``--method inpaint`` still forces inpaint for anyone who
    wants it."""
    # A capture-less mark can only be inpainted; a capture mark inpaints only when
    # explicitly asked (auto + reverse-alpha both recover the true pixels).
    if method == "inpaint" or not has_capture:
        return "inpaint"
    return "reverse-alpha"


def _inpaint_remove(mark: KnownMark, image: NDArray[Any], force: bool) -> tuple[NDArray[Any], Region | None]:
    """Remove ``mark`` by inpainting its footprint: the NCC-aligned captured
    silhouette (:meth:`footprint_mask`), erased with MI-GAN when its extra is
    installed, else cv2 (see :func:`preferred_inpaint_backend`). No-op (returns a
    copy) on a clean corner unless ``force``. Gated on the mark's trust-confidence
    detection, so a clean image is never touched."""
    from remove_ai_watermarks import region_eraser

    det = mark.detect(image)
    if not (det.detected or force):
        return image.copy(), None
    engine = _engine(mark.key)
    fm = engine.footprint_mask(image, force=force)  # uniform signature; text/pill ignore force
    if fm is None or not fm.any():
        return image.copy(), None
    if preferred_inpaint_backend() == "migan":
        result = region_eraser.erase_migan(image, fm)
    else:
        result = region_eraser.erase_cv2(image, fm, radius=6)
    return result, (det.region if det.detected else None)


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


# The three text-mark engines (Doubao/Jimeng/Samsung) share the TextMarkEngine
# interface, so one parameterized adapter pair drives all of them -- a new
# reverse-alpha text mark is one `_text_mark(...)` row below, not another copy-paste
# of these bodies. Removal is reverse-alpha only: applied when the mark is detected
# (or forced) and the alpha asset loads, otherwise skipped (no hallucination on a
# clean corner).
def _text_mark_detect(key: str, label: str, location: str) -> Callable[[NDArray[Any]], MarkDetection]:
    def detect(image: NDArray[Any]) -> MarkDetection:
        d = _engine(key).detect(image)
        return MarkDetection(key, label, location, d.detected, d.confidence, d.region)

    return detect


def _text_mark_remove(key: str) -> Callable[..., tuple[NDArray[Any], Region | None]]:
    def remove(
        image: NDArray[Any], _inpaint_method: InpaintMethod, _inpaint: bool, _strength: float, force: bool
    ) -> tuple[NDArray[Any], Region | None]:
        engine = _engine(key)
        det = engine.detect(image)
        if (det.detected or force) and engine.reverse_alpha_available(image):
            return engine.remove_watermark_reverse_alpha(image), (det.region if det.detected else None)
        return image.copy(), None

    return remove


def _text_mark(key: str, label: str, location: str) -> KnownMark:
    """A reverse-alpha text-mark registry row (Doubao/Jimeng/Samsung)."""
    return KnownMark(
        key, label, location, True, "reverse-alpha", _text_mark_detect(key, label, location), _text_mark_remove(key)
    )


# ── Capture-less mark: the Jimeng-basic "AI生成" pill (top-left, inpaint-only) ──
# No alpha map exists, so removal is inpaint only (has_capture=False routes every
# method to inpaint). Detection is edge-NCC of a synthetic silhouette; see pill_engine.
def _pill_detect(image: NDArray[Any]) -> MarkDetection:
    d = _engine("jimeng_pill").detect(image)
    return MarkDetection("jimeng_pill", "Jimeng AI生成 pill", "top-left", d.detected, d.confidence, d.region)


def _pill_noop_remove(
    image: NDArray[Any], _im: InpaintMethod, _ip: bool, _st: float, _force: bool
) -> tuple[NDArray[Any], Region | None]:
    # Capture-less: reverse-alpha is impossible. resolve_removal_method routes every
    # method to inpaint (_inpaint_remove), so this reverse-alpha slot is never reached;
    # return an untouched copy defensively.
    return image.copy(), None


_REGISTRY: tuple[KnownMark, ...] = (
    KnownMark("gemini", "Google Gemini sparkle", "bottom-right", True, "reverse-alpha", _gemini_detect, _gemini_remove),
    _text_mark("doubao", "Doubao 豆包AI生成 text", "bottom-right"),
    _text_mark("jimeng", "Jimeng 即梦AI wordmark", "bottom-right"),
    _text_mark("samsung", "Samsung Galaxy AI text", "bottom-left"),
    KnownMark("jimeng_pill", "Jimeng AI生成 pill", "top-left", True, "inpaint", _pill_detect, _pill_noop_remove, False),
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
    by ``--mark auto``."""
    return [m.detect(image) for m in _REGISTRY if include_explicit or m.in_auto]


def best_auto_mark(image: NDArray[Any]) -> MarkDetection | None:
    """The highest-confidence detected ``in_auto`` mark, or None if none fired."""
    fired = [d for d in detect_marks(image, include_explicit=False) if d.detected]
    return max(fired, key=lambda d: d.confidence) if fired else None


def remove_auto_marks(
    image: NDArray[Any],
    *,
    pill_metadata: bool = False,
    method: RemovalMethod = "auto",
    inpaint_method: InpaintMethod = "ns",
    inpaint: bool = True,
    inpaint_strength: float = 0.85,
) -> tuple[NDArray[Any], list[str]]:
    """Remove EVERY detected ``in_auto`` mark in one pass, chaining the result.

    Marks coexist in different corners -- a Jimeng-basic image carries BOTH the
    top-left "AI生成" pill AND the bottom-right "★ 即梦AI" wordmark -- and their
    confidences are on different scales, so ``best_auto_mark`` (single strongest)
    would clean only one and leave the other (issue #54). Each mark re-detects at
    its own corner on the progressively-cleaned image, so order does not matter.

    The capture-less ``jimeng_pill`` has a weak edge-NCC detector (~7% raw
    false-fire), so it is kept only when the image is CONFIRMED Jimeng and NOT
    Doubao. Confirmation is ``pill_metadata`` (the China-AIGC / TC260 metadata
    signal, supplied by the caller) OR the reliable bottom-right wordmark firing --
    the wordmark keeps recall on metadata-STRIPPED uploads (screenshots / re-saved
    files) that the metadata gate alone would miss. Returns ``(result, [labels
    removed])``; an empty list means nothing fired."""
    fired = [d for d in detect_marks(image, include_explicit=False) if d.detected]
    keys = {d.key for d in fired}
    jimeng_confirmed = pill_metadata or "jimeng" in keys
    if "jimeng_pill" in keys and (not jimeng_confirmed or "doubao" in keys):
        fired = [d for d in fired if d.key != "jimeng_pill"]
    result = image
    for det in fired:
        result, _ = get_mark(det.key).remove(
            result,
            method=method,
            inpaint_method=inpaint_method,
            inpaint=inpaint,
            inpaint_strength=inpaint_strength,
            force=False,
        )
    return result, [d.label for d in fired]
