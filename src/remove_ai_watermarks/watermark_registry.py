"""Registry of known visible watermarks.

A single catalog that ties each known visible mark to (a) where it usually sits,
(b) how to recognize it there, and (c) how to remove it. One pass over the
registry detects every known mark in its usual place and removes the ones
present.

**Localize -> fill -> validate.** A known mark is removed by LOCALIZING it (a version-robust
detector that returns a binary footprint MASK) and then handing
that mask to ONE shared, swappable fill backend (``region_eraser``: cv2 Telea/NS,
MI-GAN, or big-LaMa). The same detector then checks the result without changing
the mask or running a second fill. No mark carries a reverse-alpha step any more: the old
``original = (wm - a*logo)/(1-a)`` recovery depended on a fixed captured alpha map
at a fixed position, broke whenever a vendor re-rendered or moved its mark, and was
not color-lossless even with the right map (it amplifies quantization/JPEG-chroma
error by ``1/(1-a)`` -- the "the color just changed, not removed" reports). The
localizer stays cheap (cv2/numpy, CPU) so a memory-tight caller can run it on a
small worker; the heavy fill (MI-GAN / LaMa) is opt-in and chosen by the caller.

Entries:
  - ``gemini`` -- Google Gemini / Nano Banana visible watermark (sparkle), bottom-right.
  - ``doubao`` -- ByteDance Doubao "豆包AI生成" text strip, bottom-right.
  - ``jimeng`` -- ByteDance Jimeng / Dreamina "★ 即梦AI" wordmark, bottom-right.
  - ``qwen`` -- Alibaba Cloud Qwen "千问AI生成" text or three-lobe symbol, bottom-right.
  - ``kling`` -- Kuaishou Kling AI "可灵AI 3.0" / "KlingAI 3.0" strip, bottom-right.
  - ``yuanbao`` -- Tencent Yuanbao "元宝 / AI生成" two-line mark, bottom-right.
  - ``samsung`` -- Samsung Galaxy AI "Contenuti generati dall'AI" strip, bottom-left.
  - ``jimeng_pill`` -- Jimeng-basic "AI生成" pill, top-left (capture-less).
  - ``runninghub`` -- RunningHub "RunningHub AI生成" text, top-left (gray front-end).
  - ``baidu`` -- Baidu "百度 AI生成" text + white tag, bottom-right.
  - ``liblib`` -- LiblibAI "LiblibAI" wordmark, bottom-center.
  - ``liblib_pill`` -- LiblibAI compact "AI生成" pill, top-left.
  - ``microsoft`` -- one measured Microsoft white AI-badge variant, top-right.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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

# Product family per mark now lives on the registry row (``KnownMark.product``);
# ``_PRODUCT_OF`` is derived from it right after ``_REGISTRY`` is built.


@dataclass(frozen=True)
class MarkDetection:
    """Uniform detection result for a known mark (across heterogeneous engines)."""

    key: str
    label: str
    location: str
    detected: bool
    confidence: float
    region: Region
    # The engine's OWN detection object, threaded back to this mark's mask builder so it
    # does not re-run the detector (the text-mark footprint is bounded by the ladder
    # sweep detection already ran). Opaque here: each mask adapter knows its engine's
    # type. Excluded from eq/repr so the uniform result stays comparable across engines.
    engine_detection: Any | None = field(default=None, compare=False, repr=False)


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
        "the default evidence-driven behavior."
    )
}


def validate_sensitivity(value: str) -> Sensitivity:
    """Reject a removed sensitivity LOUDLY instead of silently falling back to ``auto``.

    ``Sensitivity`` is a ``Literal``, which is not enforced at runtime, so a caller
    upgrading from 0.15 would pass ``"assume_ai"`` and quietly get ``auto`` behavior --
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
    # Retain the exact perception outputs for the action stage. The first selected
    # mark is still being localized on the unchanged input, so throwing these away
    # and re-detecting it paid for a duplicate scan. Reusing that scan makes room for
    # the read-only post-removal validation without adding another detector pass to
    # the common one-mark path. Optional for policy-only Candidates in tests.
    strict_detection: MarkDetection | None = field(default=None, compare=False, repr=False)
    relaxed_detection: MarkDetection | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Decision:
    """The arbiter's verdict for one fired mark: remove it, at the resolved trust
    level (``relax`` feeds the mark's mask build so the fill footprint matches the
    level the mark was accepted at)."""

    candidate: Candidate
    relax: bool


VisibleRemovalStatus = Literal["no_watermark", "cleaned", "partial", "unvalidated"]
MarkValidationStatus = Literal["cleaned", "partial", "unvalidated"]


@dataclass(frozen=True)
class MarkRemovalResult:
    """One visible-mark fill and its read-only post-removal validation.

    ``partial`` means the same detector still accepts an overlapping region after
    the one fill. It is useful evidence about this engine, not an independent oracle.
    Validation never expands the mask and never runs a second fill.
    """

    key: str
    label: str
    location: str
    region: Region
    mask_bbox: Region
    backend: Literal["cv2", "migan", "lama"]
    confidence_before: float
    confidence_after: float | None
    status: MarkValidationStatus
    residual_region: Region | None = None


@dataclass(frozen=True)
class VisibleRemovalResult:
    """Detailed result of one automatic visible-mark pass."""

    image: NDArray[Any] = field(compare=False, repr=False)
    marks: tuple[MarkRemovalResult, ...]

    @property
    def labels(self) -> list[str]:
        """Labels whose masks were filled, preserving the legacy list contract."""
        return [mark.label for mark in self.marks]

    @property
    def status(self) -> VisibleRemovalStatus:
        """Aggregate status, with a known residual taking precedence over unknown."""
        if not self.marks:
            return "no_watermark"
        if any(mark.status == "partial" for mark in self.marks):
            return "partial"
        if any(mark.status == "unvalidated" for mark in self.marks):
            return "unvalidated"
        return "cleaned"


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
    # Product family, for the `auto` cross-mark corroboration: a confidently detected
    # mark relaxes only OTHER marks of the SAME product (different corners, one product
    # -- the Jimeng wordmark + the Jimeng pill). Doubao and Jimeng are BOTH ByteDance but
    # distinct products in the SAME bottom-right corner, so they must NOT cross-relax
    # (relaxing Doubao on a Jimeng wordmark would spuriously fire Doubao on it).
    # REQUIRED, deliberately: a defaulted empty product would let two rows that forgot
    # the field corroborate each other and silently bypass the trust gate.
    product: str
    # The company responsible for the product. This is deliberately separate from the
    # labeling standard: Alibaba, ByteDance, Kuaishou, Tencent, and other manufacturers
    # all use TC260, but that does not make their marks one product family.
    manufacturer: str
    # Which provenance regime this mark's vendor labels under, or None. "tc260" means
    # the vendor stamps the China AIGC label; it does not identify the manufacturer.
    label_regime: str | None
    # The sentence `identify` reports when THIS mark is the strongest evidence. None for
    # a mark that never names a platform on its own: `gemini` has its own higher-
    # confidence sparkle path, and the capture-less pill is too weak to attribute.
    platform: str | None
    _detect: Callable[..., MarkDetection]
    _mask: Callable[..., NDArray[Any] | None]
    # Optional physical-feature probe: the mark's OWN measurements its gate needs
    # (e.g. the pill's footprint flatness), so the perception pass stays uniform and
    # does not special-case any mark. None = the mark's gate needs no extra evidence.
    _features: Callable[..., dict[str, float]] | None = None
    # Metadata signal names that confirm this mark's vendor, and platform substrings
    # that do. Both drive `api.visible_provenance`; empty means nothing confirms it.
    provenance_signals: tuple[str, ...] = ()
    provenance_platform_tokens: tuple[str, ...] = ()
    # TC260 ``ContentProducer`` identities that name THIS mark's vendor -- Unified
    # Social Credit Codes as normalized by ``metadata.uscc_of``, plus the bare product
    # names a few generators write instead. Here rather than in a separate table
    # because a newly registered TC260 mark whose codes were forgotten fails SILENTLY:
    # its own detector remains strict even on an image carrying that producer's label.
    tc260_producer_codes: tuple[str, ...] = ()
    # Optional single-pass dual verdict for the arbiter's perception stage (see
    # `detect_both`). None = fall back to two `_detect` calls.
    _detect_both: Callable[..., tuple[MarkDetection, MarkDetection]] | None = None
    # False for a weak detector that may be acted on after corroboration but must not
    # itself grant a sibling the relaxed trust level.
    can_corroborate: bool = True

    def features(self, image: NDArray[Any]) -> dict[str, float]:
        """Physical features the mark reports for the arbiter's gate (empty if none)."""
        return self._features(image) if self._features is not None else {}

    def detect(self, image: NDArray[Any], *, provenance: bool = False) -> MarkDetection:
        """Detect the mark (verdict + bbox, no mask). ``provenance`` signals that
        external metadata already confirms this vendor, so the engine may relax its
        trust threshold (a mark it would otherwise demote as a content false positive
        is trusted when provenance says the vendor is present)."""
        return self._detect(image, provenance=provenance)

    def detect_both(self, image: NDArray[Any]) -> tuple[MarkDetection, MarkDetection]:
        """``(strict, relaxed)`` from ONE pass over the image.

        The arbiter's perception stage needs a mark's verdict at BOTH trust levels so it
        can pick per mark without re-detecting. ``provenance`` never changes what a
        detector computes -- only the threshold it compares against, or (Gemini) whether
        a false-positive gate demotes the result afterwards -- so the expensive scan is
        shared. A mark without a ``_detect_both`` adapter falls back to two calls."""
        if self._detect_both is not None:
            return self._detect_both(image)
        return self.detect(image, provenance=False), self.detect(image, provenance=True)

    def localize(
        self,
        image: NDArray[Any],
        *,
        provenance: bool = False,
        force: bool = False,
        detection: MarkDetection | None = None,
    ) -> Localization:
        """Detect and build the removal mask in one call. Returns a
        :class:`Localization`; ``mask`` is None unless the mark is detected (or
        ``force`` bypasses detection for the mark's usual footprint).

        ``detection`` lets a caller that has ALREADY detected this mark at this trust
        level hand the result in rather than pay for the scan twice -- the explicit
        ``visible --mark <name>`` path detects once to report the confidence and would
        otherwise re-detect here."""
        det = detection if detection is not None else self.detect(image, provenance=provenance)
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
        detection: MarkDetection | None = None,
    ) -> tuple[NDArray[Any], Region | None]:
        """Remove this mark by localize -> fill; returns ``(result, region)`` where
        ``region`` is the removed mark's bbox, or None if nothing was removed.

        ``backend`` picks the fill (``auto`` = LaMa > MI-GAN > cv2, best available; or force
        ``cv2``/``migan``/``lama``). ``provenance`` relaxes the detector's trust gate
        when metadata already confirms the vendor. ``force`` removes at the mark's
        usual footprint even without a positive detection (the ``--no-detect`` path).
        NB: the CLI does NOT use ``region`` to clear alpha on save -- that zeroing
        caused the issue-#30 white box."""
        loc = self.localize(image, provenance=provenance, force=force, detection=detection)
        if loc.mask is None or not loc.mask.any():
            return image.copy(), None
        return fill(image, loc.mask, backend=backend), (loc.region if loc.detected else None)


# Single source of truth for the Gemini-sparkle "trust this as a real mark"
# confidence, shared by BOTH the removal arbitration here (`_gemini_detect`) and
# the provenance detector in `identify` (which imports it as its sparkle threshold).
# Defining it once removes the detect-vs-remove
# threshold drift found during compatibility testing: identify
# would report a sparkle while removal declined it, or vice versa, whenever the
# two independently-maintained 0.5 constants fell out of step. Now they cannot.
#
# Value 0.5 is calibrated: the Gemini engine's own `detected` flag uses a
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
# metadata), hand-labeled against a two-sided control (labeler sensitivity ~88%,
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

# key -> (module basename, class name). Only the NAMES live here: ``import_module``
# runs inside :func:`_engine` on first use, so importing this module -- and with it the
# metadata-only ``identify`` / ``visible_provenance`` path -- never pulls cv2 through
# an engine. Same lazy-import shape as ``_text_mark_engine._rival_config`` and ``fill``.
_ENGINE_CLASS: dict[str, tuple[str, str]] = {
    "gemini": ("gemini_engine", "GeminiEngine"),
    "doubao": ("doubao_engine", "DoubaoEngine"),
    "jimeng": ("jimeng_engine", "JimengEngine"),
    "qwen": ("qwen_engine", "QwenEngine"),
    "kling": ("kling_engine", "KlingEngine"),
    "yuanbao": ("yuanbao_engine", "YuanbaoEngine"),
    "samsung": ("samsung_engine", "SamsungEngine"),
    "jimeng_pill": ("pill_engine", "PillEngine"),
    "runninghub": ("runninghub_engine", "RunningHubEngine"),
    "baidu": ("baidu_engine", "BaiduEngine"),
    "liblib": ("liblib_engine", "LibLibEngine"),
    "liblib_pill": ("liblib_engine", "LibLibPillEngine"),
    "microsoft": ("microsoft_engine", "MicrosoftEngine"),
}


def _engine(key: str) -> Any:
    if key not in _engines:
        from importlib import import_module

        module_name, class_name = _ENGINE_CLASS[key]  # KeyError(key) for an unknown key
        _engines[key] = getattr(import_module(f"remove_ai_watermarks.{module_name}"), class_name)()
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
    resolves in quality order, LaMa then MI-GAN then cv2 (see
    :func:`resolve_backend`)."""
    from remove_ai_watermarks import region_eraser

    return region_eraser.erase(image, mask=mask, backend=resolve_backend(backend))


# ── Detection adapters (verdict + bbox; no mask work on this path) ──
# The identify scan calls `detect_marks`, which must stay cheap (it runs every
# detector on the memory-tight identify host), so detection never builds a mask.


def _gemini_wrap(d: Any, *, provenance: bool) -> MarkDetection:
    gate = _GEMINI_PROVENANCE_MIN_CONF if provenance else _GEMINI_AUTO_MIN_CONF
    detected = bool(d.detected) and d.confidence >= gate
    return MarkDetection(
        "gemini",
        "Google Gemini visible watermark (sparkle)",
        "bottom-right",
        detected,
        d.confidence,
        d.region,
    )


def _gemini_detect(image: NDArray[Any], *, provenance: bool = False) -> MarkDetection:
    return _gemini_wrap(_engine("gemini").detect_watermark(image, trust_provenance=provenance), provenance=provenance)


def _gemini_detect_both(image: NDArray[Any]) -> tuple[MarkDetection, MarkDetection]:
    strict, relaxed = _engine("gemini").detect_watermark_both(image)
    return _gemini_wrap(strict, provenance=False), _gemini_wrap(relaxed, provenance=True)


def _gemini_mask(
    image: NDArray[Any], *, force: bool = False, detection: MarkDetection | None = None
) -> NDArray[Any] | None:
    # Reuse the decision's provenance-aware region (skip the strict re-detect that would
    # otherwise re-demote a relaxed sparkle to None); None region -> footprint_mask
    # falls back to its own detect-then-force path (direct/--no-detect callers).
    region = detection.region if (detection is not None and detection.detected) else None
    return _engine("gemini").footprint_mask(image, force=force, region=region)


# Engines with the standard detect/detect_both/footprint_mask interface share
# parameterized adapters. Each engine still owns its footprint implementation.
def _engine_mark_detect(key: str, label: str, location: str) -> Callable[..., MarkDetection]:
    def detect(image: NDArray[Any], *, provenance: bool = False) -> MarkDetection:
        d = _engine(key).detect(image, provenance=provenance)
        return MarkDetection(key, label, location, d.detected, d.confidence, d.region, engine_detection=d)

    return detect


def _engine_mark_detect_both(key: str, label: str, location: str) -> Callable[..., tuple[MarkDetection, MarkDetection]]:
    def detect_both(image: NDArray[Any]) -> tuple[MarkDetection, MarkDetection]:
        strict, relaxed = _engine(key).detect_both(image)
        return (
            MarkDetection(
                key, label, location, strict.detected, strict.confidence, strict.region, engine_detection=strict
            ),
            MarkDetection(
                key, label, location, relaxed.detected, relaxed.confidence, relaxed.region, engine_detection=relaxed
            ),
        )

    return detect_both


def _engine_mark_mask(key: str) -> Callable[..., NDArray[Any] | None]:
    def mask(
        image: NDArray[Any], *, force: bool = False, detection: MarkDetection | None = None
    ) -> NDArray[Any] | None:
        # Thread the engine's OWN accepted detection into the mask builder. This avoids
        # repeating its scan and preserves the trust verdict selected by the arbiter.
        # A direct or --no-detect caller still reaches the engine's fallback path.
        return _engine(key).footprint_mask(
            image, force=force, detection=detection.engine_detection if detection is not None else None
        )

    return mask


def _text_mark(
    key: str,
    label: str,
    location: str,
    *,
    platform: str,
    manufacturer: str,
    product: str | None = None,
    label_regime: str | None = "tc260",
    provenance_signals: tuple[str, ...] = ("aigc",),
    tc260_producer_codes: tuple[str, ...] = (),
    provenance_platform_tokens: tuple[str, ...] = (),
) -> KnownMark:
    """Build a text-mark registry row from its shared detector and mask adapters.

    ``product`` defaults to the key (one mark, one product); pass it only when two
    marks share a product. ``manufacturer`` is separate because unrelated companies
    use the same TC260 standard. ``label_regime`` and ``provenance_signals`` default
    to the China-AIGC label because every text mark registered so far except Samsung
    and Microsoft uses it.
    """
    return KnownMark(
        key,
        label,
        location,
        True,
        product or key,
        manufacturer,
        label_regime,
        platform,
        _engine_mark_detect(key, label, location),
        _engine_mark_mask(key),
        provenance_signals=provenance_signals,
        tc260_producer_codes=tc260_producer_codes,
        provenance_platform_tokens=provenance_platform_tokens,
        _detect_both=_engine_mark_detect_both(key, label, location),
    )


# ── Capture-less mark: the Jimeng-basic "AI生成" pill (top-left) ──
# Detection is edge-NCC of a synthetic silhouette; the mask is a fixed top-left
# geometry box (see pill_engine). Removal is the same localize -> fill as the rest.
def _pill_detect(image: NDArray[Any], *, provenance: bool = False) -> MarkDetection:
    del provenance  # the pill detector is provenance-independent; its relaxation lives entirely in _keep_pill
    d = _engine("jimeng_pill").detect(image)
    return MarkDetection("jimeng_pill", "Jimeng AI生成 pill", "top-left", d.detected, d.confidence, d.region)


def _pill_detect_both(image: NDArray[Any]) -> tuple[MarkDetection, MarkDetection]:
    # The pill detector is provenance-independent (`_pill_detect` discards the flag), so
    # one call answers both levels. MarkDetection is frozen, so sharing it is safe.
    d = _pill_detect(image)
    return d, d


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
    # Gemini is a Google C2PA/SynthID product, not a China-AIGC labeler.
    KnownMark(
        "gemini",
        "Google Gemini visible watermark (sparkle)",
        "bottom-right",
        True,
        "gemini",
        "google",
        None,
        # No platform sentence: the sparkle has its own higher-confidence
        # `_visible_sparkle` path in identify, which names the platform itself.
        None,
        _gemini_detect,
        _gemini_mask,
        provenance_platform_tokens=("google", "gemini"),
        _detect_both=_gemini_detect_both,
    ),
    _text_mark(
        "doubao",
        "Doubao 豆包AI生成 text",
        "bottom-right",
        platform="ByteDance Doubao (visible 豆包AI生成 mark detected)",
        manufacturer="bytedance",
        tc260_producer_codes=("91110102MACQD9K640", "doubao"),
    ),
    _text_mark(
        "jimeng",
        "Jimeng 即梦AI wordmark",
        "bottom-right",
        platform="ByteDance Jimeng / Dreamina (visible 即梦AI mark detected)",
        manufacturer="bytedance",
        tc260_producer_codes=("9144030008867405X2",),
    ),
    _text_mark(
        "qwen",
        "Qwen 千问AI生成 text or three-lobe symbol",
        "bottom-right",
        platform="Alibaba Cloud Qwen (visible text or symbol mark detected)",
        manufacturer="alibaba",
        tc260_producer_codes=("91440101MA9Y9T4H7A",),
    ),
    _text_mark(
        "kling",
        "Kling AI 可灵AI / KlingAI 3.0 text",
        "bottom-right",
        platform="Kuaishou Kling AI (visible 可灵AI / KlingAI 3.0 mark detected)",
        manufacturer="kuaishou",
        tc260_producer_codes=("91110108335469089C",),
    ),
    _text_mark(
        "yuanbao",
        "Tencent Yuanbao 元宝 / AI生成 mark",
        "bottom-right",
        platform="Tencent Yuanbao (visible 元宝 / AI生成 mark detected)",
        manufacturer="tencent",
        tc260_producer_codes=("91440300708461136T",),
    ),
    # Samsung Galaxy AI is a device editing marker (samsung_genai), not a TC260 label.
    _text_mark(
        "samsung",
        "Samsung Galaxy AI text",
        "bottom-left",
        manufacturer="samsung",
        label_regime=None,
        provenance_signals=("samsung_genai",),
        platform="Samsung Galaxy AI (visible 'Contenuti generati dall'AI' mark detected)",
    ),
    _text_mark(
        "runninghub",
        "RunningHub AI生成 text",
        "top-left",
        platform="RunningHub (visible RunningHub AI生成 mark detected)",
        manufacturer="runninghub",
        tc260_producer_codes=("91340100MAEB4N8H76", "RunningHub"),
    ),
    _text_mark(
        "baidu",
        "Baidu 百度 AI生成 text",
        "bottom-right",
        platform="Baidu (visible 百度 AI生成 mark detected)",
        manufacturer="baidu",
        tc260_producer_codes=("91110000802100433B",),
    ),
    _text_mark(
        "liblib",
        "LiblibAI wordmark",
        "bottom-center",
        platform="LiblibAI (visible LiblibAI mark detected)",
        manufacturer="liblib",
        tc260_producer_codes=("91110105MACJ6K1C8A",),
    ),
    KnownMark(
        "liblib_pill",
        "LiblibAI AI生成 pill",
        "top-left",
        True,
        "liblib",
        "liblib",
        "tc260",
        # The compact pill is generic on its own; the bottom wordmark or LiblibAI
        # metadata is what makes the relaxed verdict actionable.
        None,
        _engine_mark_detect("liblib_pill", "LiblibAI AI生成 pill", "top-left"),
        _engine_mark_mask("liblib_pill"),
        provenance_signals=("aigc",),
        _detect_both=_engine_mark_detect_both("liblib_pill", "LiblibAI AI生成 pill", "top-left"),
        can_corroborate=False,
    ),
    # One measured Microsoft visible-mark variant: a white top-right pill with
    # dark internal shapes. Microsoft's documented feature also permits other
    # icon, text, and placement variants, which this detector does not cover.
    _text_mark(
        "microsoft",
        "Microsoft top-right AI badge",
        "top-right",
        manufacturer="microsoft",
        label_regime=None,
        provenance_signals=(),
        platform="Microsoft (visible top-right AI badge detected)",
        provenance_platform_tokens=("microsoft",),
    ),
    # Same product as the Jimeng wordmark -- the one pair that cross-relaxes.
    KnownMark(
        "jimeng_pill",
        "Jimeng AI生成 pill",
        "top-left",
        True,
        "jimeng",
        "bytedance",
        "tc260",
        # The capture-less pill is too weak a detector to attribute a platform on its
        # own; the Jimeng wordmark is what names ByteDance.
        None,
        _pill_detect,
        _pill_mask,
        _pill_features,
        _detect_both=_pill_detect_both,
        can_corroborate=False,
    ),
)

# Product family per mark, derived from the registry rows so registering a mark is one
# edit. See KnownMark.product for why Doubao and Jimeng must not cross-relax.
_PRODUCT_OF: dict[str, str] = {m.key: m.product for m in _REGISTRY}

# Weak pill detectors cannot vouch for sibling marks. Letting the Jimeng pill do so
# produced a closed false-positive loop: a clean-image pill fire relaxed the wordmark,
# and that wordmark then bypassed the pill's flatness guard. Keep this policy on each
# registry row so adding another weak companion cannot silently omit the safeguard.
_CANNOT_CORROBORATE: frozenset[str] = frozenset(m.key for m in _REGISTRY if not m.can_corroborate)


def _provenance_confirms_product(product: str, provenance: frozenset[str]) -> bool:
    """Whether external provenance names any registered mark of ``product``."""
    return any(_PRODUCT_OF.get(key) == product for key in provenance)


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
    already confirms, so every mark belonging to that product may relax its trust
    gate."""
    return [
        m.detect(image, provenance=_provenance_confirms_product(m.product, provenance))
        for m in _REGISTRY
        if include_explicit or m.in_auto
    ]


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
    never relaxes. A mark is ``confirmed`` only on same-product evidence -- metadata
    naming any mark of that product, or a confidently strict-detected sibling
    (``_PRODUCT_OF``, minus the marks too weak to vouch,
    :data:`_CANNOT_CORROBORATE`). Without that evidence a mark stays ``strict``: there is
    no path that relaxes a gate on anything less than same-product evidence."""
    if sensitivity == "strict":
        return "strict"
    product = _PRODUCT_OF[key]
    confirmed = _provenance_confirms_product(product, provenance) or any(
        _PRODUCT_OF[k] == product for k in strict_keys if k != key and k not in _CANNOT_CORROBORATE
    )
    return "confirmed" if confirmed else "strict"


def tc260_producer_mark(code: str) -> KnownMark | None:
    """Return the registry row for one TC260 ``ContentProducer`` identity."""
    return next((mark for mark in _REGISTRY if code in mark.tc260_producer_codes), None)


def _pill_suppressors() -> set[str]:
    """Other products from the pill's manufacturer whose detection vetoes it."""
    pill = get_mark("jimeng_pill")
    return {m.key for m in _REGISTRY if m.manufacturer == pill.manufacturer and m.product != pill.product}


def _keep_pill(keys: set[str], *, provenance: frozenset[str], footprint_flat: bool) -> bool:
    """Whether to auto-remove the capture-less 'AI生成' pill given the fired marks.

    Pure decision (the flatness feature is precomputed at perception time and passed
    in). The pill detector is weak and metadata/intent confirms the platform, not pill
    presence, so a naive confirmation-OR gate over-fires on textured ceilings and walls
    that the fill visibly smears. Arms:
      * bottom-right "★ 即梦AI" wordmark fired -> ~94% precise, and it survives
        metadata-STRIPPED images: remove the pill unrestricted;
      * TC260 metadata confirms Jimeng (``"jimeng" in provenance``, no wordmark) -> remove ONLY when the
        top-left footprint is flat enough for an invisible fill (``footprint_flat``),
        so real flat-scene pills (and harmless flat false fires) are cleaned while the
        damaging textured false fires are left untouched.
    A Doubao image is TC260 too but is not Jimeng-basic, so its mark suppresses the
    pill. Marks from other TC260 manufacturers are unrelated and do not participate in
    this ByteDance-family decision.
    No confirmation at all -> never remove (blocks false fires on non-Jimeng content).

    The suppressor set is derived from the manufacturer and product fields rather than
    from the shared TC260 standard. Marks from Google, Samsung, or another TC260
    manufacturer cannot enable or veto this ByteDance-specific arm."""
    if _pill_suppressors() & keys:
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
    exactly. Both levels come from ONE scan per mark (:meth:`KnownMark.detect_both`):
    the trust level moves a threshold, never the measurement, so running the detector
    twice was doing the expensive half of the work for a second time. The loop is
    uniform -- it knows nothing about any specific mark: each mark reports its own gate
    features via :meth:`KnownMark.features` (computed only when the mark is detected, so
    a clean image pays nothing extra)."""
    cands: list[Candidate] = []
    for m in _REGISTRY:
        if not m.in_auto:
            continue
        strict, relaxed = m.detect_both(image)
        feats = m.features(image) if (strict.detected or relaxed.detected) else {}
        cands.append(
            Candidate(
                m.key,
                m.label,
                strict.detected,
                relaxed.detected,
                feats,
                strict_detection=strict,
                relaxed_detection=relaxed,
            )
        )
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


def _mask_bbox(mask: NDArray[Any]) -> Region | None:
    """Smallest ``(x, y, w, h)`` box containing a mask, or None when empty."""
    import cv2

    x, y, w, h = cv2.boundingRect(mask)
    return (x, y, w, h) if w > 0 and h > 0 else None


def _regions_overlap(left: Region, right: Region) -> bool:
    """Whether two positive-area ``(x, y, w, h)`` regions overlap."""
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    if min(lw, lh, rw, rh) <= 0:
        return False
    return lx < rx + rw and rx < lx + lw and ly < ry + rh and ry < ly + lh


def remove_auto_marks_detailed(
    image: NDArray[Any],
    *,
    sensitivity: Sensitivity = "auto",
    provenance: frozenset[str] = frozenset(),
    backend: Backend = "auto",
) -> VisibleRemovalResult:
    """Fill every decided mark once, then validate each result without editing it.

    The three stages are separated: PERCEPTION (:func:`_build_candidates` -- engines
    report what they see, no policy), DECISION (:func:`decide` -- the pure arbiter over
    ``(candidates, Context)``), ACTION (localize -> :func:`fill` per winner). Marks
    coexist in different corners -- a Jimeng-basic image carries BOTH the top-left pill
    AND the bottom-right wordmark -- so every decided mark is removed, chained on the
    progressively-cleaned image (order does not matter, each re-localizes its corner).

    Three orthogonal knobs: ``sensitivity`` (how hard to trust a borderline mark --
    see :data:`Sensitivity`), ``provenance`` (vendor keys external metadata confirms,
    the evidence that drives ``auto``; the TC260 label maps to ``jimeng``/``doubao``),
    and ``backend`` (the shared fill).

    Each successful action has a fourth, read-only stage: the same mark detector runs
    once on the filled image. An accepted region counts as a residual only when it
    overlaps the mask that was actually filled. The result is ``partial`` in that case,
    ``unvalidated`` if the detector fails, and ``cleaned`` otherwise. This is a
    self-consistency check, not an independent removal oracle; it never changes the
    mask, retries the fill, or hides the candidate image.

    The first selected mark reuses the exact detection from perception because its
    input is still unchanged. Later marks retain the historical re-detection on the
    progressively modified image, so overlapping/co-firing marks cannot act on stale
    evidence."""
    context = Context(sensitivity=sensitivity, provenance=provenance)
    result = image
    removals: list[MarkRemovalResult] = []
    resolved_backend: Literal["cv2", "migan", "lama"] | None = None
    for d in decide(_build_candidates(image), context):
        mark = get_mark(d.candidate.key)
        cached_detection = None
        if not removals:
            cached_detection = d.candidate.relaxed_detection if d.relax else d.candidate.strict_detection
        localization = mark.localize(
            result,
            provenance=d.relax,
            force=False,
            detection=cached_detection,
        )
        mask = localization.mask
        if mask is None:
            continue
        bbox = _mask_bbox(mask)
        if bbox is None:
            continue

        if resolved_backend is None:
            resolved_backend = resolve_backend(backend)
        region = localization.region
        confidence_before = localization.confidence
        result = fill(result, mask, backend=resolved_backend)
        del localization, mask

        confidence_after: float | None = None
        residual_region: Region | None = None
        validation_status: MarkValidationStatus
        try:
            after = mark.detect(result, provenance=d.relax)
            confidence_after = after.confidence
            if after.detected and _regions_overlap(after.region, bbox):
                validation_status = "partial"
                residual_region = after.region
            else:
                validation_status = "cleaned"
        except Exception as exc:
            validation_status = "unvalidated"
            logger.warning(
                "Post-removal validation failed for %s after %s fill in mask %s: %s",
                mark.key,
                resolved_backend,
                bbox,
                exc,
            )

        removals.append(
            MarkRemovalResult(
                key=mark.key,
                label=mark.label,
                location=mark.location,
                region=region,
                mask_bbox=bbox,
                backend=resolved_backend,
                confidence_before=confidence_before,
                confidence_after=confidence_after,
                status=validation_status,
                residual_region=residual_region,
            )
        )

    return VisibleRemovalResult(result, tuple(removals))


def remove_auto_marks(
    image: NDArray[Any],
    *,
    sensitivity: Sensitivity = "auto",
    provenance: frozenset[str] = frozenset(),
    backend: Backend = "auto",
) -> tuple[NDArray[Any], list[str]]:
    """Compatibility wrapper returning the filled image and removed labels.

    The action still performs the read-only post-removal check. Call
    :func:`remove_auto_marks_detailed` when its ``cleaned`` / ``partial`` /
    ``unvalidated`` result is needed.
    """
    report = remove_auto_marks_detailed(
        image,
        sensitivity=sensitivity,
        provenance=provenance,
        backend=backend,
    )
    return report.image, report.labels
