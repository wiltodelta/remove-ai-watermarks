"""Registry of known visible watermarks.

A single catalog that ties each known visible mark to (a) where it usually sits,
(b) how to recognize it there, and (c) how to remove it. One pass over the
registry detects every known mark in its usual place and removes the ones
present.

**Localize -> fill.** A known mark is removed by LOCALIZING it (a template-free,
version-robust detector that returns a binary footprint MASK) and then handing
that mask to ONE shared, swappable fill backend (``region_eraser``: cv2 Telea/NS,
MI-GAN, or big-LaMa). No mark carries a reverse-alpha step any more: the old
``original = (wm - a*logo)/(1-a)`` recovery depended on a fixed captured alpha map
at a fixed position, broke whenever a vendor re-rendered or moved its mark, and was
not color-lossless even with the right map (it amplifies quantization/JPEG-chroma
error by ``1/(1-a)`` -- the "the color just changed, not removed" reports). The
localizer stays cheap (cv2/numpy, CPU) so a memory-tight caller can run it on a
small worker; the heavy fill (MI-GAN / LaMa) is opt-in and chosen by the caller.

Entries:
  - ``gemini`` -- Google Gemini / Nano Banana sparkle, bottom-right.
  - ``doubao`` -- ByteDance Doubao "豆包AI生成" text strip, bottom-right.
  - ``jimeng`` -- ByteDance Jimeng / Dreamina "★ 即梦AI" wordmark, bottom-right.
  - ``qwen`` -- Alibaba Tongyi Qianwen "千问AI生成" text strip, bottom-right.
  - ``samsung`` -- Samsung Galaxy AI "Contenuti generati dall'AI" strip, bottom-left.
  - ``jimeng_pill`` -- Jimeng-basic "AI生成" pill, top-left (capture-less).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

Region = tuple[int, int, int, int]

# Fill backend for the shared removal path. ``auto`` resolves best-first to the highest
# quality installed model -- LaMa, else MI-GAN, else cv2 (see ``resolve_backend``); the
# others force a specific backend (mirrors the ``erase`` command's ``--backend``).
Backend = Literal["auto", "cv2", "migan", "lama"]

# Detection sensitivity for the removal path -- how much to trust a borderline mark.
#   * ``strict``: high-precision visual gate only; never relaxed, so a clean image is
#     never touched (the gate demotes a sparkle-shaped content match, so it never fills
#     a clean corner). Lowest recall on faint/moved marks.
#   * ``auto`` (default): relax a mark's gate ONLY when the image carries same-product
#     evidence the mark is there -- metadata provenance for that vendor, or a confidently
#     detected sibling mark of the same product (see ``resolve_trust``). No evidence ->
#     stays strict. Safe: it only escalates where the mark is corroborated.
#
# REMOVED 2026-07-19: ``assume_ai`` relaxed every mark's gate on the caller's bare
# assertion that the image is AI. It was a statistical gamble, not an instruction:
# "this image is AI" says nothing about WHICH vendor or WHERE, which is exactly what a
# gate bypass needs, so it took a confidence floor to be tolerable at all (before that
# floor it filled a phantom sparkle on 59.8% of genuine camera photos). It also had no
# place in the product's own model -- detector finds a mark, remove it; detector finds
# nothing, leave the image alone; the USER sees a mark and says so, act on that. A user
# who can see the mark is better served by pointing at it (``erase --region``) or naming
# it (``--mark X --no-detect`` for a text mark), both of which execute an instruction
# instead of guessing. Removing it also collapsed the trust ladder from three levels to
# two. See docs/module-internals.md for the measurements.
Sensitivity = Literal["auto", "strict"]

# The trust level a mark's detection gate is resolved to (see ``resolve_trust``).
# ``confirmed`` bypasses the engine's false-positive gate, and that bypass is documented
# to require evidence naming THIS vendor (see GeminiEngine.detect_watermark's
# ``trust_provenance`` contract) -- so it is only ever reached from same-product evidence.
# A third ``assumed`` level existed for ``assume_ai`` and went with it (2026-07-19).
Trust = Literal["strict", "confirmed"]

# Product family per mark, for the ``auto`` cross-mark corroboration: a confidently
# detected mark relaxes only OTHER marks of the SAME product (different corners, one
# product -- the Jimeng wordmark + the Jimeng pill). Doubao and Jimeng are BOTH ByteDance
# but distinct products in the SAME bottom-right corner, so they must NOT cross-relax
# (relaxing Doubao on a Jimeng wordmark would spuriously fire Doubao on it).
_PRODUCT_OF: dict[str, str] = {
    "gemini": "gemini",
    "doubao": "doubao",
    "jimeng": "jimeng",
    "jimeng_pill": "jimeng",  # same product as the Jimeng wordmark
    "qwen": "qwen",
    "samsung": "samsung",
}


# Marks whose own detection is too weak to serve as EVIDENCE for a sibling of the
# same product, even though they share one. Sibling corroboration grants ``confirmed``
# trust, which bypasses the sibling's false-positive gate outright -- so a detector
# that false-fires often must not be able to hand that bypass to anyone.
#
# The pill qualifies on its own documented numbers: ~7% raw false-fire (5.5% measured
# on 578 vendor negatives, 2026-07-18). Letting it corroborate produced a closed loop
# on the DEFAULT auto path, no user flag involved:
#   pill false-fires on a clean non-ByteDance image
#     -> _PRODUCT_OF maps it to "jimeng", so jimeng resolves to `confirmed`
#     -> jimeng's NCC gate drops 0.45 -> 0.3825 and it false-fires too
#     -> _keep_pill now sees "jimeng" in keys and takes the WORDMARK arm, which
#        removes the pill unrestricted -- skipping the flatness guard that exists
#        precisely to stop the fill smearing a textured corner.
# Measured on the corpus: 3 of 578 negatives ran the full loop, one of them with
# footprint_flat=0 (the exact case the guard was written to block). Cutting the pill
# out of corroboration removed all 3 and cost NOTHING on 4417 TC260 carriers
# (jimeng fires 398 -> 398), so this is a defect fix, not a recall trade.
#
# `_keep_pill` already encodes the same distrust for the pill's own ACTION; this
# closes the gap that its TESTIMONY was never gated.
# Regression: tests/test_watermark_registry.py::TestArbiter::
#   test_weak_pill_detection_does_not_confirm_the_jimeng_wordmark
_CANNOT_CORROBORATE: frozenset[str] = frozenset({"jimeng_pill"})


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
class Localization:
    """A located mark: its detection verdict plus the full-frame removal mask.

    ``mask`` is a full-frame uint8 array (255 = remove) sized to the image, or None
    when nothing should be removed (no detection and not forced, or the footprint
    could not be placed). ``region`` is the mark's bbox (for logging / residual
    positioning)."""

    detected: bool
    confidence: float
    region: Region
    mask: NDArray[Any] | None


_REMOVED_SENSITIVITIES = {
    "assume_ai": (
        "sensitivity='assume_ai' was removed in 0.16: it relaxed EVERY mark's detection "
        "gate on the bare assertion that an image is AI, which says nothing about which "
        "vendor made it or where the mark is. If you can see a mark the detector missed, "
        "act on what you see: erase(image, region=(x, y, w, h)), or the CLI "
        "`--mark <name> --no-detect` for a known text mark. Use sensitivity='auto' for "
        "the default evidence-driven behaviour."
    )
}


def validate_sensitivity(value: str) -> Sensitivity:
    """Reject a removed sensitivity LOUDLY instead of silently falling back to ``auto``.

    ``Sensitivity`` is a ``Literal``, which is not enforced at runtime, so a caller
    upgrading from 0.15 would pass ``"assume_ai"`` and quietly get ``auto`` behaviour --
    a silent semantic change on the one release where they most need to be told.
    """
    if value in _REMOVED_SENSITIVITIES:
        raise ValueError(_REMOVED_SENSITIVITIES[value])
    if value not in ("auto", "strict"):
        raise ValueError(f"unknown sensitivity {value!r}; expected 'auto' or 'strict'")
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class Context:
    """The evidence + policy the removal arbiter decides against (perception is
    kept separate from this decision). ``sensitivity`` is the caller's intent
    (see :data:`Sensitivity`); ``provenance`` is the vendor keys local metadata
    confirms, the evidence that drives ``auto``. Bundling them into one object is
    why the arbiter can be a pure function of ``(candidates, context)``."""

    sensitivity: Sensitivity = "auto"
    provenance: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        validate_sensitivity(self.sensitivity)


@dataclass(frozen=True)
class Candidate:
    """One mark's PERCEPTION output -- what the engine sees, with NO policy applied.

    Carries the mark's verdict at BOTH trust levels (``detected_strict`` = the
    conservative gate, ``detected_relaxed`` = the gate the engine relaxes to under
    provenance), so the arbiter can pick per mark without re-running detection.

    ``features`` is a generic bag of physical measurements a mark's gate may need (the
    mark owns which it reports via ``KnownMark._features``); e.g. the pill supplies
    ``footprint_flat`` (0/1). Empty for marks whose gate needs no extra evidence."""

    key: str
    label: str
    detected_strict: bool
    detected_relaxed: bool
    features: dict[str, float]  # generic; both construction sites always supply it (empty when none)


@dataclass(frozen=True)
class Decision:
    """The arbiter's verdict for one fired mark: remove it, at the resolved trust
    level (``relax`` feeds the mark's mask build so the fill footprint matches the
    level the mark was accepted at)."""

    candidate: Candidate
    relax: bool


@dataclass(frozen=True)
class KnownMark:
    """A known visible watermark: where it lives, how to find and mask it.

    Removal is uniform (:meth:`remove`): localize the mark to a mask, then fill that
    mask with the chosen backend. Each mark supplies two cheap cv2/numpy callables --
    ``_detect`` (verdict + bbox, no mask; used by the identify scan) and ``_mask``
    (the full-frame footprint mask; used by removal)."""

    key: str
    label: str
    location: str  # usual place, human-readable ("bottom-right")
    in_auto: bool  # participate in `--mark auto` scanning
    _detect: Callable[..., MarkDetection]
    _mask: Callable[..., NDArray[Any] | None]
    # Optional physical-feature probe: the mark's OWN measurements its gate needs
    # (e.g. the pill's footprint flatness), so the perception pass stays uniform and
    # does not special-case any mark. None = the mark's gate needs no extra evidence.
    _features: Callable[..., dict[str, float]] | None = None

    def features(self, image: NDArray[Any]) -> dict[str, float]:
        """Physical features the mark reports for the arbiter's gate (empty if none)."""
        return self._features(image) if self._features is not None else {}

    def detect(self, image: NDArray[Any], *, provenance: bool = False) -> MarkDetection:
        """Detect the mark (verdict + bbox, no mask). ``provenance`` signals that
        external metadata already confirms this vendor, so the engine may relax its
        trust threshold (a mark it would otherwise demote as a content false positive
        is trusted when provenance says the vendor is present)."""
        return self._detect(image, provenance=provenance)

    def localize(self, image: NDArray[Any], *, provenance: bool = False, force: bool = False) -> Localization:
        """Detect and build the removal mask in one call. Returns a
        :class:`Localization`; ``mask`` is None unless the mark is detected (or
        ``force`` bypasses detection for the mark's usual footprint)."""
        det = self.detect(image, provenance=provenance)
        if not (det.detected or force):
            return Localization(det.detected, det.confidence, det.region, None)
        # Pass the (provenance-aware) detection to the mask builder so it does NOT
        # re-detect at a different trust level -- a relaxed sparkle must not be
        # re-demoted into a None mask (reported-removed-but-unchanged).
        mask = self._mask(image, force=force, detection=det)
        return Localization(det.detected, det.confidence, det.region, mask)

    def remove(
        self,
        image: NDArray[Any],
        *,
        backend: Backend = "auto",
        provenance: bool = False,
        force: bool = False,
    ) -> tuple[NDArray[Any], Region | None]:
        """Remove this mark by localize -> fill; returns ``(result, region)`` where
        ``region`` is the removed mark's bbox, or None if nothing was removed.

        ``backend`` picks the fill (``auto`` = LaMa > MI-GAN > cv2, best available; or force
        ``cv2``/``migan``/``lama``). ``provenance`` relaxes the detector's trust gate
        when metadata already confirms the vendor. ``force`` removes at the mark's
        usual footprint even without a positive detection (the ``--no-detect`` path).
        NB: the CLI does NOT use ``region`` to clear alpha on save -- that zeroing
        caused the issue-#30 white box."""
        loc = self.localize(image, provenance=provenance, force=force)
        if loc.mask is None or not loc.mask.any():
            return image.copy(), None
        return fill(image, loc.mask, backend=backend), (loc.region if loc.detected else None)


# Single source of truth for the Gemini-sparkle "trust this as a real mark"
# confidence, shared by BOTH the removal arbitration here (`_gemini_detect`) and
# the provenance detector in `identify` (which imports it as its sparkle threshold).
# Defining it once removes the detect-vs-remove
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
# band (2026-06-20) and REJECTED for the no-provenance case: it cannot be done
# without re-admitting the Doubao-text / content false positives. The band below
# the gate is therefore left to the metadata-confirmed path below.
GEMINI_SPARKLE_TRUST_CONF = 0.5
_GEMINI_AUTO_MIN_CONF = GEMINI_SPARKLE_TRUST_CONF

# Provenance-confirmed Gemini trust gate. When external metadata already proves the
# image is a Google generation (C2PA issuer "Google"/"Gemini"), the [gate, 0.5) band
# that the no-provenance gate leaves out is no longer ambiguous with Doubao text: a
# Doubao image carries ByteDance provenance, not Google, so it never reaches this
# relaxed gate. The vendor moving/re-rendering the sparkle (bigger, lighter, shifted
# north-west) drops a real sparkle into this band, and the fixed-slot detector demotes
# it -- provenance is exactly the extra evidence that lets us trust it.
#
# The gate was originally the engine's own `detected` floor (0.35). Raised to 0.42
# on 2026-07-18 after measuring what this arm actually admits, because the Doubao
# argument above -- while correct -- is not the binding constraint. Google C2PA is
# carried by Imagen, API generations and NotebookLM exports, none of which stamp a
# visible sparkle at all, so the relaxed gate spends most of its budget on images
# that never had a mark rather than on moved ones.
#
# Measured blind on 954 unique Google-metadata uploads (detector never saw the
# metadata), hand-labelled against a two-sided control (labeller sensitivity ~88%,
# specificity 100%). "Additions" = accepted with provenance but not without:
#
#   band         precision   95% CI    population
#   0.35-0.42          13%    5-30%           120
#   0.42-0.46          35%   19-54%            47
#   0.46-0.50          27%   14-46%            44
#   0.50-0.54          40%   20-64%            15
#
# Precision is flat above 0.42 and collapses below it, and that bottom band alone is
# half the arm's volume -- so this is a step, not a gradient, and 0.42 is where it
# sits. Raising the gate here drops ~16 genuine recoveries to prevent ~104 false
# fills (6.5:1), cutting false fills from 18.7% to 7.8% of Google-metadata uploads.
# A false fill is the worse error: it destroys pixels AND makes the caller report a
# removal that did not happen, while a miss leaves the image untouched.
#
# NOTE: even at 0.42 this arm runs at ~33% precision (two false fills per genuine
# recovery). Whether an arm that inaccurate should exist at all is a product call,
# not a tuning one -- do not read this constant as "now correct".
_GEMINI_PROVENANCE_MIN_CONF = 0.42

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
        elif key == "qwen":
            from remove_ai_watermarks.qwen_engine import QwenEngine

            _engines[key] = QwenEngine()
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


_warned_cv2_fallback = False


def preferred_inpaint_backend() -> Literal["lama", "migan", "cv2"]:
    """Backend the ``auto`` fill resolves to, best-first: LaMa > MI-GAN > cv2.

    LaMa is the highest-quality inpaint (it recovers the textured/structured backgrounds
    the classical fill smears), so ``auto`` prefers it whenever a learned backend can run
    (onnxruntime present). MI-GAN is the lighter learned model; both currently share the
    SAME onnxruntime availability check, so ``auto`` cannot tell them apart and always
    prefers the better one -- a memory-tight deployment that cannot afford LaMa's ~4.7 GB
    peak pins MI-GAN explicitly via ``--backend migan`` / ``backend="migan"`` (that is the
    deployment's call, not the library's). cv2 is the classical no-deps floor and the last
    resort: it smears textured/structured backgrounds, so a one-time quality warning fires
    when ``auto`` falls back to it."""
    from remove_ai_watermarks import region_eraser

    if region_eraser.lama_available():
        return "lama"
    if region_eraser.migan_available():
        return "migan"
    global _warned_cv2_fallback
    if not _warned_cv2_fallback:
        _warned_cv2_fallback = True
        logger.warning(
            "No learned-inpaint backend available (onnxruntime not installed); falling back "
            "to the cv2 classical inpaint, which can smear textured or structured backgrounds. "
            "Install the 'lama' (best) or 'migan' (lighter) extra for higher-quality fills."
        )
    return "cv2"


def resolve_backend(backend: Backend) -> Literal["cv2", "migan", "lama"]:
    """Resolve ``auto`` to the preferred installed backend; pass the rest through."""
    if backend == "auto":
        return preferred_inpaint_backend()
    return backend


def fill(image: NDArray[Any], mask: NDArray[Any], *, backend: Backend = "auto") -> NDArray[Any]:
    """The ONE shared, mark-agnostic removal: erase ``mask`` (255 = remove) via the
    chosen inpaint backend. Delegates to :func:`region_eraser.erase`; ``auto``
    resolves to MI-GAN when installed else cv2 (see :func:`resolve_backend`)."""
    from remove_ai_watermarks import region_eraser

    return region_eraser.erase(image, mask=mask, backend=resolve_backend(backend))


# ── Detection adapters (verdict + bbox; no mask work on this path) ──
# The identify scan calls `detect_marks`, which must stay cheap (it runs every
# detector on the memory-tight identify host), so detection never builds a mask.


def _gemini_detect(image: NDArray[Any], *, provenance: bool = False) -> MarkDetection:
    d = _engine("gemini").detect_watermark(image, trust_provenance=provenance)
    gate = _GEMINI_PROVENANCE_MIN_CONF if provenance else _GEMINI_AUTO_MIN_CONF
    detected = bool(d.detected) and d.confidence >= gate
    return MarkDetection("gemini", "Google Gemini sparkle", "bottom-right", detected, d.confidence, d.region)


def _gemini_mask(
    image: NDArray[Any], *, force: bool = False, detection: MarkDetection | None = None
) -> NDArray[Any] | None:
    # Reuse the decision's provenance-aware region (skip the strict re-detect that would
    # otherwise re-demote a relaxed sparkle to None); None region -> footprint_mask
    # falls back to its own detect-then-force path (direct/--no-detect callers).
    region = detection.region if (detection is not None and detection.detected) else None
    return _engine("gemini").footprint_mask(image, force=force, region=region)


# The three text-mark engines (Doubao/Jimeng/Samsung) share the TextMarkEngine
# interface, so one parameterized adapter pair drives all of them -- a new
# text mark is one `_text_mark(...)` row below, not another copy-paste of these
# bodies. Detection matches the glyph silhouette; the mask is the template-free
# glyph-bbox footprint (see TextMarkEngine.footprint_mask).
def _text_mark_detect(key: str, label: str, location: str) -> Callable[..., MarkDetection]:
    def detect(image: NDArray[Any], *, provenance: bool = False) -> MarkDetection:
        d = _engine(key).detect(image, provenance=provenance)
        return MarkDetection(key, label, location, d.detected, d.confidence, d.region)

    return detect


def _text_mark_mask(key: str) -> Callable[..., NDArray[Any] | None]:
    def mask(
        image: NDArray[Any], *, force: bool = False, detection: MarkDetection | None = None
    ) -> NDArray[Any] | None:
        # Text masks rebuild the glyph blob template-free (no trust gate to re-apply), so
        # the detection is not needed here; accepted for the uniform _mask signature.
        del detection
        return _engine(key).footprint_mask(image, force=force)

    return mask


def _text_mark(key: str, label: str, location: str) -> KnownMark:
    """A text-mark registry row (Doubao/Jimeng/Samsung): glyph-silhouette detect +
    template-free glyph-bbox mask."""
    return KnownMark(key, label, location, True, _text_mark_detect(key, label, location), _text_mark_mask(key))


# ── Capture-less mark: the Jimeng-basic "AI生成" pill (top-left) ──
# Detection is edge-NCC of a synthetic silhouette; the mask is a fixed top-left
# geometry box (see pill_engine). Removal is the same localize -> fill as the rest.
def _pill_detect(image: NDArray[Any], *, provenance: bool = False) -> MarkDetection:
    del provenance  # the pill detector is provenance-independent; its relaxation lives entirely in _keep_pill
    d = _engine("jimeng_pill").detect(image)
    return MarkDetection("jimeng_pill", "Jimeng AI生成 pill", "top-left", d.detected, d.confidence, d.region)


def _pill_mask(
    image: NDArray[Any], *, force: bool = False, detection: MarkDetection | None = None
) -> NDArray[Any] | None:
    # The pill mask is a fixed top-left geometry box, independent of the detection;
    # accepted for the uniform _mask signature.
    del detection
    return _engine("jimeng_pill").footprint_mask(image, force=force)


def _pill_features(image: NDArray[Any]) -> dict[str, float]:
    """The pill's own gate feature: top-left footprint flatness (1.0 = flat enough for
    an invisible fill), read by the metadata arm of :func:`_keep_pill`."""
    return {"footprint_flat": float(_engine("jimeng_pill").footprint_is_flat(image))}


_REGISTRY: tuple[KnownMark, ...] = (
    KnownMark("gemini", "Google Gemini sparkle", "bottom-right", True, _gemini_detect, _gemini_mask),
    _text_mark("doubao", "Doubao 豆包AI生成 text", "bottom-right"),
    _text_mark("jimeng", "Jimeng 即梦AI wordmark", "bottom-right"),
    _text_mark("qwen", "Qwen 千问AI生成 text", "bottom-right"),
    _text_mark("samsung", "Samsung Galaxy AI text", "bottom-left"),
    KnownMark("jimeng_pill", "Jimeng AI生成 pill", "top-left", True, _pill_detect, _pill_mask, _pill_features),
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


def detect_marks(
    image: NDArray[Any],
    *,
    include_explicit: bool = True,
    provenance: frozenset[str] = frozenset(),
) -> list[MarkDetection]:
    """Detect every known mark in its usual place.

    Returns one MarkDetection per scanned mark (``detected`` flags which fired).
    ``include_explicit=False`` scans only the ``in_auto`` marks -- the set used
    by ``--mark auto``. ``provenance`` names the vendor keys that external metadata
    already confirms, so each named mark's detector may relax its trust gate."""
    return [m.detect(image, provenance=m.key in provenance) for m in _REGISTRY if include_explicit or m.in_auto]


def resolve_trust(
    key: str,
    *,
    sensitivity: Sensitivity,
    provenance: frozenset[str],
    strict_keys: set[str],
) -> Trust:
    """The trust level mark ``key``'s detection gate is resolved to.

    The single place that turns the ``sensitivity`` policy + evidence into a per-mark
    level (which the engines consume as ``provenance = level != "strict"``). ``strict``
    never relaxes. A mark is ``confirmed`` only on same-product evidence -- the vendor
    confirmed by metadata (``key in provenance``) or a confidently strict-detected
    sibling of the same product (``_PRODUCT_OF``, minus the marks too weak to vouch,
    :data:`_CANNOT_CORROBORATE`). Without that evidence a mark stays ``strict``: there is
    no path that relaxes a gate on anything less than same-product evidence."""
    if sensitivity == "strict":
        return "strict"
    product = _PRODUCT_OF[key]
    confirmed = key in provenance or any(
        _PRODUCT_OF[k] == product for k in strict_keys if k != key and k not in _CANNOT_CORROBORATE
    )
    return "confirmed" if confirmed else "strict"


def _keep_pill(keys: set[str], *, provenance: frozenset[str], footprint_flat: bool) -> bool:
    """Whether to auto-remove the capture-less 'AI生成' pill given the fired marks.

    Pure decision (the flatness feature is precomputed at perception time and passed
    in). The pill detector is weak (~7% raw false-fire) and metadata/intent confirms
    the platform, not pill presence, so a naive confirmation-OR gate over-fires: on a
    32k real-upload corpus (2026-07) the metadata-only arm was only ~27% precise and
    its false fires were textured ceilings/walls that the fill visibly SMEARS. Arms:
      * bottom-right "★ 即梦AI" wordmark fired -> ~94% precise, and it survives
        metadata-STRIPPED uploads: remove the pill unrestricted;
      * TC260 metadata confirms Jimeng (``"jimeng" in provenance``, no wordmark) -> remove ONLY when the
        top-left footprint is flat enough for an invisible fill (``footprint_flat``),
        so real flat-scene pills (and harmless flat false fires) are cleaned while the
        damaging textured false fires are left untouched.
    A Doubao image is TC260 too but is not Jimeng-basic, so the pill never rides on a
    Doubao detection; a Qwen image likewise (another vendor's bottom-right mark naming
    its own product), so a confident Qwen detection suppresses the pill the same way.
    No confirmation at all -> never remove (blocks false fires on non-Jimeng content)."""
    if "doubao" in keys or "qwen" in keys:
        return False
    if "jimeng" in keys:
        return True
    if "jimeng" in provenance:
        return footprint_flat
    return False


def _build_candidates(image: NDArray[Any]) -> list[Candidate]:
    """PERCEPTION pass: run every ``in_auto`` mark's detector at both trust levels and
    package the raw verdicts + physical features into :class:`Candidate` objects. No
    policy here -- the arbiter (:func:`decide`) makes every keep/drop call.

    Each mark is detected at the strict AND the relaxed (``provenance=True``) level so
    :func:`decide` can pick per mark without re-running detection; a relaxed gate is
    monotonically more permissive, so this reproduces the old strict-then-relax pass
    exactly. The loop is uniform -- it knows nothing about any specific mark: each mark
    reports its own gate features via :meth:`KnownMark.features` (computed only when the
    mark is detected, so a clean image pays nothing extra)."""
    cands: list[Candidate] = []
    for m in _REGISTRY:
        if not m.in_auto:
            continue
        strict = m.detect(image, provenance=False)
        relaxed = m.detect(image, provenance=True)
        feats = m.features(image) if (strict.detected or relaxed.detected) else {}
        cands.append(Candidate(m.key, m.label, strict.detected, relaxed.detected, feats))
    return cands


def decide(candidates: list[Candidate], context: Context) -> list[Decision]:
    """The removal ARBITER: a pure function turning perception + context into the
    ordered list of marks to remove (and the trust level each was accepted at).

    All policy lives here, in one place: per-mark trust resolution (:func:`resolve_trust`,
    which needs the strict-detected siblings for ``auto`` cross-mark corroboration) and
    the capture-less pill gate (:func:`_keep_pill`). No image, no I/O -- so it is unit-testable in isolation and
    the same decision drives every caller."""
    strict_keys = {c.key for c in candidates if c.detected_strict}
    fired: list[Decision] = []
    for c in candidates:
        trust = resolve_trust(
            c.key, sensitivity=context.sensitivity, provenance=context.provenance, strict_keys=strict_keys
        )
        relax = trust != "strict"
        ok = c.detected_relaxed if relax else c.detected_strict
        if ok:
            fired.append(Decision(c, relax))
    keys = {d.candidate.key for d in fired}
    if "jimeng_pill" in keys:
        pill = next(d for d in fired if d.candidate.key == "jimeng_pill")
        flat = bool(pill.candidate.features.get("footprint_flat", 0.0))
        if not _keep_pill(keys, provenance=context.provenance, footprint_flat=flat):
            fired = [d for d in fired if d.candidate.key != "jimeng_pill"]
    return fired


def remove_auto_marks(
    image: NDArray[Any],
    *,
    sensitivity: Sensitivity = "auto",
    provenance: frozenset[str] = frozenset(),
    backend: Backend = "auto",
) -> tuple[NDArray[Any], list[str]]:
    """Remove EVERY decided ``in_auto`` mark in one pass, chaining the result.

    The three stages are separated: PERCEPTION (:func:`_build_candidates` -- engines
    report what they see, no policy), DECISION (:func:`decide` -- the pure arbiter over
    ``(candidates, Context)``), ACTION (localize -> :func:`fill` per winner). Marks
    coexist in different corners -- a Jimeng-basic image carries BOTH the top-left pill
    AND the bottom-right wordmark -- so every decided mark is removed, chained on the
    progressively-cleaned image (order does not matter, each re-localizes its corner).

    Three orthogonal knobs: ``sensitivity`` (how hard to trust a borderline mark --
    see :data:`Sensitivity`), ``provenance`` (vendor keys external metadata confirms,
    the evidence that drives ``auto``; the TC260 label maps to ``jimeng``/``doubao``),
    and ``backend`` (the shared fill). Returns ``(result, [labels removed])``; empty
    means nothing fired."""
    context = Context(sensitivity=sensitivity, provenance=provenance)
    result = image
    labels: list[str] = []
    for d in decide(_build_candidates(image), context):
        result, region = get_mark(d.candidate.key).remove(result, backend=backend, provenance=d.relax, force=False)
        # Only report the mark as removed when a fill actually happened: remove() returns
        # a None region when the localized mask came back empty, and reporting it anyway
        # would claim a removal that left the pixels unchanged.
        if region is not None:
            labels.append(d.candidate.label)
    return result, labels
