"""High-level convenience API: clean an image (or array) in one call.

The low-level building blocks live in ``watermark_registry`` (localize -> fill) and
``image_io`` (Unicode-safe, alpha-preserving IO). This module ties them into the two
calls a caller usually wants, so a library user does not have to decode images, wire
up metadata provenance, or preserve the alpha channel by hand:

    import remove_ai_watermarks as raiw
    raiw.remove_visible("in.png", "out.png")                 # path -> file, provenance auto
    result, removed = raiw.remove_visible(bgr_array)         # array -> array
    report = raiw.remove_visible_detailed(bgr_array)         # includes post-fill status
    raiw.remove_visible("shot.png", "out.png", sensitivity="strict")
    raiw.visible_provenance("in.png")                        # -> frozenset({"gemini"})

Imports stay lazy (inside the functions), so ``import remove_ai_watermarks`` is cheap.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from remove_ai_watermarks.watermark_registry import Backend, Sensitivity, VisibleRemovalResult


def __getattr__(name: str) -> object:
    """Lazily expose the detailed visible result type without loading image code."""
    if name == "VisibleRemovalResult":
        from remove_ai_watermarks.watermark_registry import VisibleRemovalResult

        return VisibleRemovalResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass(frozen=True)
class _VisibleInput:
    """Normalized visible-removal input with its file-only context."""

    bgr: NDArray[Any]
    alpha: NDArray[Any] | None = None
    path: Path | None = None
    provenance: frozenset[str] = frozenset()


def visible_provenance(source: str | Path) -> frozenset[str]:
    """Vendor keys that the file's local metadata confirms, the evidence that drives
    the ``auto`` sensitivity (relaxing a corroborated mark's detection trust gate).

    Mapping: a Google/Gemini C2PA issuer -> ``"gemini"``; a ``samsung_genai`` marker ->
    ``"samsung"``; a China-AIGC (TC260) label -> the vendor its ``ContentProducer``
    names (``KnownMark.tc260_producer_codes``), falling back to ByteDance's two products
    when the producer is absent or unmapped.

    The TC260 label itself says only "this is AI", so it used to relax Doubao and
    Jimeng on EVERY China-AIGC image -- including one carrying a Qwen or Kling mark,
    where relaxing the wrong pair is pure false-positive risk and the mark actually
    present never reached the relaxed gate its own ``provenance_ncc_factor`` was
    calibrated for. The producer code identifies the signing entity, so it can.

    Best-effort: any read error yields an empty set (no relaxation). Metadata-only, so
    it never loads cv2/torch.
    """
    import contextlib

    path = Path(source)
    with contextlib.suppress(Exception):
        from remove_ai_watermarks import identify

        rep = identify.identify(path, check_visible=False, check_invisible=False)
        return _provenance_from_report(rep, path)
    return frozenset()


def _tc260_vendors(path: Path) -> frozenset[str]:
    """Vendor keys a TC260 label confirms, from its ``ContentProducer`` identity.

    An absent, unreadable or unmapped producer falls back to the historical pair rather
    than to nothing: the caller has already established that the AIGC signal fired, so
    the image IS China-AIGC labeled, and dropping to no relaxation would lose the
    detections the fallback recovers today. The re-read is deliberately isolated -- a
    failure here must narrow the answer, never discard the rest of the provenance.
    """
    import contextlib

    from remove_ai_watermarks._internal.constants import TC260_FALLBACK_VENDORS

    with contextlib.suppress(Exception):
        from remove_ai_watermarks.metadata import aigc_label, uscc_of
        from remove_ai_watermarks.watermark_registry import tc260_producer_vendors

        producer = (aigc_label(path) or {}).get("ContentProducer", "")
        if producer and (vendor := tc260_producer_vendors().get(uscc_of(producer))):
            return frozenset({vendor})
    return TC260_FALLBACK_VENDORS


def _load_visible_input(source: str | Path | NDArray[Any]) -> _VisibleInput:
    """Normalize a path/array source without making the public operation stateful."""
    if not isinstance(source, (str, Path)):
        return _VisibleInput(source)

    from remove_ai_watermarks import image_io

    path = Path(source)
    bgr, alpha = image_io.read_bgr_and_alpha(path)
    if bgr is None:
        raise ValueError(f"Could not read image: {source}")
    return _VisibleInput(bgr=bgr, alpha=alpha, path=path, provenance=visible_provenance(path))


def _write_visible_result(
    loaded: _VisibleInput,
    result: NDArray[Any],
    removed: list[str],
    output: str | Path,
    *,
    strip_metadata: bool,
    write_noop: bool,
) -> None:
    """Write one visible-removal result while preserving a true no-op losslessly."""
    if not removed and not write_noop:
        return

    from remove_ai_watermarks import image_io

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source_path = loaded.path
    if not removed and source_path is not None and source_path.suffix.lower() == out_path.suffix.lower():
        # Copy the ORIGINAL bytes instead of lossily re-encoding a no-op. An in-place
        # call needs no copy and would otherwise raise shutil.SameFileError.
        if source_path.resolve() != out_path.resolve():
            import shutil

            shutil.copyfile(source_path, out_path)
    else:
        # imwrite is contractually NON-RAISING, so this bool is the only signal the file
        # was not created. Unchecked, the metadata strip below ran on a nonexistent path
        # and surfaced as a confusing "cannot read image <INPUT>" naming the OUTPUT path
        # (Tier E, 2026-07-20). Raise here so a library caller and the CLI both get an
        # accurate message about the write.
        if not image_io.write_bgr_with_alpha(out_path, result, loaded.alpha):
            raise OSError(f"failed to write output (is the destination writable?): {out_path}")

    if strip_metadata:
        from remove_ai_watermarks import metadata

        metadata.remove_ai_metadata(out_path, out_path)


def remove_visible(
    source: str | Path | NDArray[Any],
    output: str | Path | None = None,
    *,
    sensitivity: Sensitivity = "auto",
    backend: Backend = "auto",
    strip_metadata: bool = True,
    write_noop: bool = True,
) -> tuple[NDArray[Any], list[str]]:
    """Remove every detected known visible AI mark through localize then fill.

    The registry currently covers the Gemini visible watermark; Doubao, Jimeng,
    Qwen, Kling AI, Yuanbao, Samsung, RunningHub, Baidu, and LiblibAI text marks;
    one Microsoft top-right AI-badge variant; and the Jimeng pill. Returns
    ``(result_bgr, [labels removed])``.

    ``source`` is a file path OR a BGR ndarray. For a PATH, metadata provenance is read
    automatically (so ``sensitivity="auto"`` recovers a moved/faint mark whenever the
    file still carries its provenance) and the alpha channel is preserved on write; for
    an ARRAY there is no metadata to read and no separate alpha plane. When ``output``
    is given the cleaned image is written there (alpha rejoined for a path source); the
    array is always returned as well, so an empty ``removed`` list tells a caller nothing
    known was found (e.g. route to the diffusion ``all`` path or ``erase``).

    ``sensitivity`` (``auto``/``strict``) and ``backend``
    (``auto``/``cv2``/``migan``/``lama``) are the same controls as the CLI.

    ``strip_metadata`` (default True, matching the CLI ``visible --strip-metadata``)
    also strips AI provenance metadata (C2PA/EXIF/XMP/IPTC) from the written output via
    the lossless :func:`metadata.remove_ai_metadata`, so a library call does exactly
    what the CLI does. Only applies when ``output`` is given.

    ``write_noop`` (default True) controls whether ``output`` is written when NOTHING was
    removed: True writes a clean passthrough copy (an idempotent clean); False leaves the
    output path untouched, so a caller that treats "no mark" as "produce nothing" (the CLI
    ``visible`` no-mark contract) does not clobber a pre-existing file at that path.
    """
    report = remove_visible_detailed(
        source,
        output,
        sensitivity=sensitivity,
        backend=backend,
        strip_metadata=strip_metadata,
        write_noop=write_noop,
    )
    return report.image, report.labels


def remove_visible_detailed(
    source: str | Path | NDArray[Any],
    output: str | Path | None = None,
    *,
    sensitivity: Sensitivity = "auto",
    backend: Backend = "auto",
    strip_metadata: bool = True,
    write_noop: bool = True,
) -> VisibleRemovalResult:
    """Remove known visible marks and report the read-only post-fill check.

    Unlike :func:`remove_visible`, this returns one :class:`VisibleRemovalResult`.
    Its aggregate ``status`` is ``no_watermark``, ``cleaned``, ``partial``, or
    ``unvalidated``. Each mark also records the before/after detector confidence,
    filled mask box, backend, and any overlapping residual region.

    The validator never edits the image: every selected mask is filled exactly once,
    then the same detector checks the result. All input, output, metadata, alpha, and
    no-op semantics otherwise match :func:`remove_visible`.
    """
    from remove_ai_watermarks import watermark_registry

    watermark_registry.validate_sensitivity(sensitivity)
    loaded = _load_visible_input(source)
    report = watermark_registry.remove_auto_marks_detailed(
        loaded.bgr,
        sensitivity=sensitivity,
        provenance=loaded.provenance,
        backend=backend,
    )
    if output is not None:
        _write_visible_result(
            loaded,
            report.image,
            report.labels,
            output,
            strip_metadata=strip_metadata,
            write_noop=write_noop,
        )
    return report


# ── The three-stage image pipeline (visible -> invisible -> metadata) ──
# This is the `all` / `batch` pipeline. It lived only in cli.py, written twice with
# divergences, so a library caller could not run the flagship path at all. The CLI is
# now a thin wrapper: it builds the options, prints the stage lines through
# `progress`, and turns the outcome into console text and an exit code.


@dataclass(frozen=True)
class InvisibleOptions:
    """The invisible stage's knobs, as one value instead of a dozen parameters.

    ENGINE KNOBS ONLY, under the engine's own names and defaults, so a bare
    ``InvisibleOptions()`` behaves exactly like calling the engine with no arguments.
    The engine takes them across TWO callables -- ``__init__`` for the ones that shape
    the loaded stack, ``remove_watermark`` for the per-image ones -- so this is not a
    splat-through bag; ``_run_invisible`` forwards each field to the right one.
    ``TestInvisibleOptionsMirrorTheEngine`` compares the signatures field by field with
    no exception table to maintain: a decision made before the engine runs, like
    ``force``, is a parameter of ``remove_all`` next to ``backend`` and ``sensitivity``,
    not a knob smuggled in here.

    Immutable so a batch can build it once and reuse it across every image while the
    engine itself is cached separately.
    """

    strength: float | None = None
    pipeline: str = "qwen-zimage"
    vendor: str | None = None
    seed: int | None = None
    hf_token: str | None = None
    humanize: float = 0.0
    unsharp: float = 0.0
    adaptive_polish: bool | None = None
    max_resolution: int = 0
    controlnet_conditioning_scale: float = 1.0
    cpu_offload: bool = False
    tile: bool = False
    tile_size: int = 1024
    tile_overlap: int = 128
    text_manifest: Path | None = None
    fidelity_anchor: bool = False


# What the invisible stage did. "unavailable" is the one outcome the caller must
# surface loudly: the file looks processed but still carries the watermark.
InvisibleOutcome = Literal["removed", "no-signal", "unavailable"]


@dataclass(frozen=True)
class RemoveAllResult:
    """Outcome of :func:`remove_all` -- what each stage actually did."""

    output: Path
    visible_label: str | None  # the removed mark(s), or None when nothing fired
    invisible: InvisibleOutcome


class MetadataStripIncomplete(RuntimeError):
    """AI metadata survived the strip, so no output was produced.

    Raised BEFORE the final write, deliberately: the contract is that an incomplete
    strip leaves nothing on disk rather than an AI-readable file the caller might ship.
    """

    def __init__(self, surviving: set[str]) -> None:
        self.surviving = surviving
        super().__init__(f"AI metadata survived the strip: {', '.join(sorted(surviving))}")


class _SourceEvidence:
    """One metadata extraction per source file, serving every stage of one call.

    ``remove_all`` asks the same file two provenance questions -- which vendor the
    metadata confirms (for the visible pass) and whether an invisible target exists
    (for the scrub gate). Both start from the same file-backed extraction, and running
    them independently paid for it twice.

    Per-CALL only, never module-level: the pipeline REWRITES its input in place on some
    paths (``batch`` with the output directory equal to the input), so a holder that
    outlived one call would answer from pre-write evidence. The individual metadata
    probes are memoized on content, which is what makes that safe -- this holder only
    removes the remaining assembly work.

    Every accessor fails safe exactly as the function it replaces does: an extraction or
    verdict error yields no provenance (no relaxation) and an invisible target of True
    (scrub rather than skip).
    """

    __slots__ = ("_evidence", "_extracted", "_path")

    def __init__(self, path: Path) -> None:
        self._path = path
        self._extracted = False
        self._evidence: Any | None = None

    def _extract(self) -> Any | None:
        if not self._extracted:
            self._extracted = True
            with suppress(Exception):
                from remove_ai_watermarks.identify import extract_provenance_evidence

                self._evidence = extract_provenance_evidence(self._path)
        return self._evidence

    def visible_provenance(self) -> frozenset[str]:
        """Vendor keys the metadata confirms; empty on any failure (no relaxation)."""
        evidence = self._extract()
        if evidence is None:
            return frozenset()
        # The suppress spans the VERDICT and the mapping, not just the extraction --
        # `visible_provenance` fails safe as a whole, and a raise here would escape as a
        # traceback where the old code returned an empty set.
        with suppress(Exception):
            from remove_ai_watermarks.identify import identify_from_evidence

            report = identify_from_evidence(evidence)
            return _provenance_from_report(report, self._path)
        return frozenset()

    def has_invisible_target(self) -> bool:
        """Whether a diffusion target exists. True on any failure -- see the fail-safe
        note on ``identify.has_invisible_target``: leaving a watermark on a paid removal
        is worse than over-regenerating a clean image."""
        evidence = self._extract()
        if evidence is None:
            return True
        try:
            from remove_ai_watermarks._internal.c2pa import c2pa_info_has_removal_hint
            from remove_ai_watermarks.identify import identify_from_evidence

            report = identify_from_evidence(evidence, image_path=self._path, check_invisible=True)
        except Exception:
            return True
        return report.ai_from_metadata or c2pa_info_has_removal_hint(evidence.c2pa_info)


def _provenance_from_report(report: Any, path: Path) -> frozenset[str]:
    """Map an already-built report to the vendor keys it confirms.

    Shared by :func:`visible_provenance` and the evidence holder so the two cannot
    drift; which evidence confirms which mark is data on the registry row.
    """
    from remove_ai_watermarks.watermark_registry import known_marks

    signal_names = {signal.name for signal in report.signals}
    platform = (report.platform or "").lower()
    keys: set[str] = set()
    aigc = False
    for mark in known_marks():
        if any(token in platform for token in mark.provenance_platform_tokens):
            keys.add(mark.key)
        if "aigc" in mark.provenance_signals and "aigc" in signal_names:
            aigc = True
        elif any(name in signal_names for name in mark.provenance_signals):
            keys.add(mark.key)
    if aigc:
        # The TC260 label is vendor-agnostic: which of its marks it confirms comes from
        # the producer identity, not from the signal firing.
        keys |= _tc260_vendors(path)
    return frozenset(keys)


def remove_all(
    source: str | Path,
    output: str | Path,
    *,
    backend: Backend = "auto",
    sensitivity: Sensitivity = "auto",
    invisible: InvisibleOptions | None = None,
    force: bool = False,
    engine: Any | None = None,
    progress: Callable[[str, str], None] | None = None,
) -> RemoveAllResult:
    """Remove visible marks, the invisible watermark, and AI metadata, in that order.

    Stages are chained through a file in the SYSTEM temp dir, not next to ``output``:
    the point of staging is that the user never sees a partial output file during a long
    model download, and writing the partial next to the final defeats that.

    ``force`` scrubs even when no invisible watermark is locally detectable. It sits
    here rather than in ``InvisibleOptions`` because it decides WHETHER the engine runs,
    which is settled before the engine is built; the options carry only what the engine
    itself takes.

    ``engine`` accepts an already-constructed ``InvisibleEngine`` so a batch can build
    the model once; leave it None to construct one per call.

    ``progress`` receives ``(stage, detail)`` per step -- ``stage`` is one of
    ``visible`` / ``invisible`` / ``metadata`` and ``detail`` is a stable token, not
    prose: the caller owns the wording. The invisible stage reports its
    :data:`InvisibleOutcome`, plus a ``strength=<value>`` line before it runs.

    Raises :class:`MetadataStripIncomplete` before writing anything when AI metadata
    survives, and ``OSError`` when the output cannot be written.
    """
    import os
    import tempfile

    from remove_ai_watermarks import image_io, watermark_registry
    from remove_ai_watermarks.metadata import strip_and_verify

    def say(stage: str, detail: str) -> None:
        if progress is not None:
            progress(stage, detail)

    opts = invisible if invisible is not None else InvisibleOptions()
    src, out = Path(source), Path(output)
    watermark_registry.validate_sensitivity(sensitivity)

    image, alpha = image_io.read_bgr_and_alpha(src)
    if image is None:
        raise ValueError(f"Could not read image: {src}")

    # One metadata extraction for the whole pipeline: the visible pass asks which
    # vendor is confirmed and the scrub gate asks whether a target exists, and both
    # start from the same evidence.
    evidence = _SourceEvidence(src)

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=src.suffix)
    os.close(tmp_fd)
    staged = Path(tmp_name)
    try:
        # ── 1. Visible marks ──
        result, removed = watermark_registry.remove_auto_marks(
            image,
            sensitivity=sensitivity,
            provenance=evidence.visible_provenance(),
            backend=backend,
        )
        visible_label = ", ".join(removed) if removed else None
        say("visible", visible_label or "")
        if not image_io.write_bgr_with_alpha(staged, result, alpha):
            raise OSError(f"failed to write the staged intermediate: {staged}")

        # ── 2. Invisible watermark ──
        outcome = _run_invisible(src, staged, staged, opts, engine, say, evidence, force)

        # ── 3. AI metadata ──
        # Read the pristine ORIGINAL for provenance above and the STAGED file here:
        # the visible pass has already dropped this file's C2PA.
        _, leftover = strip_and_verify(staged, staged)
        if leftover:
            # Before the write, on purpose -- see MetadataStripIncomplete.
            raise MetadataStripIncomplete(set(leftover))
        say("metadata", "stripped")

        # The invisible stage (and the cv2.IMREAD_COLOR paths under it) drops alpha, so
        # re-attach the ORIGINAL alpha plane unchanged for transparent formats.
        final_bgr, _ = image_io.read_bgr_and_alpha(staged)
        if final_bgr is None:
            raise OSError(f"failed to read back the staged intermediate: {staged}")
        out.parent.mkdir(parents=True, exist_ok=True)
        if not image_io.write_bgr_with_alpha(out, final_bgr, alpha):
            raise OSError(f"failed to write output (is the destination writable?): {out}")
    finally:
        if staged.exists():
            staged.unlink()
    return RemoveAllResult(out, visible_label, outcome)


def _run_invisible(
    vendor_source: Path,
    in_path: Path,
    out_path: Path,
    opts: InvisibleOptions,
    engine: Any | None,
    say: Callable[[str, str], None],
    evidence: _SourceEvidence,
    force: bool,
) -> InvisibleOutcome:
    """Run, or deliberately skip, the diffusion scrub.

    ``vendor_source`` is the PRISTINE original: the staged/output file has already lost
    its C2PA to the visible pass, so gating or resolving the vendor from it would always
    read as "no signal" and "unknown vendor". ``in_path``/``out_path`` are what the
    engine reads and writes (the same staged file for ``remove_all``, input->output for
    an invisible-only batch).
    """
    from remove_ai_watermarks.invisible_engine import is_available

    if not is_available():
        say("invisible", "unavailable")
        return "unavailable"
    if not (force or opts.vendor is not None or evidence.has_invisible_target()):
        say("invisible", "no-signal")
        return "no-signal"

    from remove_ai_watermarks._internal.watermark_profiles import resolve_strength, vendor_for_strength

    # An explicit vendor override wins over detection and implies the scrub runs:
    # naming the cohort (e.g. "meta" for Muse Image Content Seal, which carries no
    # provenance to detect) asserts the pixel watermark is present, so the no-signal
    # gate must not skip it.
    vendor = opts.vendor or vendor_for_strength(vendor_source)
    # Report the strength the engine will actually execute, resolved the same way it
    # resolves it, so the reported value cannot drift from the executed one.
    with suppress(Exception):
        from PIL import Image

        with Image.open(vendor_source) as probe:
            say("invisible", f"strength={resolve_strength(opts.strength, vendor, opts.pipeline, size=probe.size)}")

    if engine is None:
        from remove_ai_watermarks.invisible_engine import InvisibleEngine

        engine = InvisibleEngine(
            pipeline=opts.pipeline,
            hf_token=opts.hf_token,
            progress_callback=lambda message: say("invisible", message),
            controlnet_conditioning_scale=opts.controlnet_conditioning_scale,
            cpu_offload=opts.cpu_offload,
        )
    engine.remove_watermark(
        image_path=in_path,
        output_path=out_path,
        strength=opts.strength,
        seed=opts.seed,
        humanize=opts.humanize,
        unsharp=opts.unsharp,
        adaptive_polish=opts.adaptive_polish,
        max_resolution=opts.max_resolution,
        vendor=vendor,
        tile=opts.tile,
        tile_size=opts.tile_size,
        tile_overlap=opts.tile_overlap,
        text_manifest=opts.text_manifest,
        fidelity_anchor=opts.fidelity_anchor,
    )
    say("invisible", "removed")
    return "removed"


BatchMode = Literal["all", "visible", "metadata", "invisible"]


@dataclass(frozen=True)
class BatchSummary:
    """Per-directory outcome of :func:`remove_batch`."""

    processed: int
    failed: int
    # Files whose invisible watermark was left in place because the GPU extra is
    # missing. Non-empty means the outputs LOOK processed but still carry it.
    invisible_unavailable: list[Path]
    errors: list[tuple[Path, str]]


def remove_batch(
    directory: str | Path,
    output_dir: str | Path,
    *,
    mode: BatchMode = "all",
    backend: Backend = "auto",
    sensitivity: Sensitivity = "auto",
    invisible: InvisibleOptions | None = None,
    force: bool = False,
    engine: Any | None = None,
    progress: Callable[[Path, str, str], None] | None = None,
) -> BatchSummary:
    """Run one removal ``mode`` over every supported image in ``directory``.

    Never raises for a single bad image: a per-file failure is counted and recorded in
    ``BatchSummary.errors`` so one unreadable file cannot abandon the rest of the
    directory. ``force`` and ``engine`` are threaded straight through, so a caller that
    passes a constructed ``InvisibleEngine`` loads the model once for the whole run.

    ``progress`` receives ``(path, stage, detail)``. Every image ends with exactly one
    terminal stage -- ``done`` or ``failed`` -- whatever the mode does in between, so a
    caller driving a progress bar can advance on that alone.
    """
    from remove_ai_watermarks._internal.utils import is_supported_format
    from remove_ai_watermarks.optional_deps import pixels_available

    src_dir, out_dir = Path(directory), Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def say(path: Path, stage: str, detail: str) -> None:
        if progress is not None:
            progress(path, stage, detail)

    processed = failed = 0
    unavailable: list[Path] = []
    errors: list[tuple[Path, str]] = []
    for img_path in sorted(p for p in src_dir.iterdir() if is_supported_format(p)):
        out_path = out_dir / img_path.name
        try:
            outcome = _run_batch_one(img_path, out_path, mode, backend, sensitivity, invisible, force, engine, say)
        except Exception as exc:
            # A missing pixel stack is a property of the ENVIRONMENT, not of this
            # file, and per-file handling turned one absent package into an error
            # per image -- 84 of them on a directory that needed one install. Let
            # it out so the caller reports it once, with the extra named.
            if isinstance(exc, ImportError) and not pixels_available():
                raise
            failed += 1
            errors.append((img_path, str(exc)))
            say(img_path, "failed", str(exc))
            continue
        processed += 1
        if outcome == "unavailable":
            unavailable.append(img_path)
        # Exactly one terminal event per image, in every mode: a caller's progress bar
        # advances on this and nothing else. Keying it off a mode-specific stage line
        # left `visible` and `metadata` runs sitting at 0% for the whole batch.
        say(img_path, "done", outcome or "")
    return BatchSummary(processed, failed, unavailable, errors)


def _run_batch_one(
    img_path: Path,
    out_path: Path,
    mode: BatchMode,
    backend: Backend,
    sensitivity: Sensitivity,
    invisible: InvisibleOptions | None,
    force: bool,
    engine: Any | None,
    say: Callable[[Path, str, str], None],
) -> InvisibleOutcome | None:
    """One image, one mode. Returns the invisible outcome when that stage ran."""
    from remove_ai_watermarks import image_io
    from remove_ai_watermarks.metadata import strip_and_verify

    if mode == "all":
        result = remove_all(
            img_path,
            out_path,
            backend=backend,
            sensitivity=sensitivity,
            invisible=invisible,
            force=force,
            engine=engine,
            progress=lambda stage, detail: say(img_path, stage, detail),
        )
        return result.invisible

    # NOTE on holders in batch: `out_path` may EQUAL `img_path` (nothing forbids
    # `-o <the input directory>`), and the visible stage rewrites it. So each stage that
    # asks the file a provenance question builds its OWN holder, after any write that
    # precedes it. One holder spanning the write would answer the invisible gate from
    # pre-write evidence and scrub a file that today is correctly skipped.
    if mode == "visible":
        # Deliberately NOT `remove_visible`: its no-op branch copies the original bytes
        # through when nothing was removed, which is right for a single lossless call
        # and wrong here. A batch must produce every output through the one writer, so a
        # failed write RAISES and the run is counted and exits non-zero -- a read-only
        # output directory once produced zero files and still exited 0 (Tier E).
        # Always read the ORIGINAL: a stale out_path from a previous run must not be
        # re-processed as if it were the input.
        from remove_ai_watermarks import watermark_registry

        image, alpha = image_io.read_bgr_and_alpha(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
        result, _ = watermark_registry.remove_auto_marks(
            image,
            sensitivity=sensitivity,
            provenance=visible_provenance(img_path),
            backend=backend,
        )
        if not image_io.write_bgr_with_alpha(out_path, result, alpha):
            raise OSError(f"failed to write output (is the destination writable?): {out_path}")
        return None

    if mode == "metadata":
        _, leftover = strip_and_verify(img_path, out_path)
        if leftover:
            raise MetadataStripIncomplete(set(leftover))
        return None

    # invisible-only: no preceding visible pass, so out_path does not exist yet, and
    # the input IS the pristine original for both the gate and the vendor probe.
    outcome = _run_invisible(
        img_path,
        img_path,
        out_path,
        invisible if invisible is not None else InvisibleOptions(),
        engine,
        lambda stage, detail: say(img_path, stage, detail),
        _SourceEvidence(img_path),
        force,
    )
    if not out_path.exists():
        # Keep the output directory COMPLETE even when the pixels are deliberately
        # left alone; a hole the caller cannot see is worse than an unchanged copy.
        src_bgr, src_alpha = image_io.read_bgr_and_alpha(img_path)
        if src_bgr is None or not image_io.write_bgr_with_alpha(out_path, src_bgr, src_alpha):
            raise OSError(f"failed to copy input through to output: {out_path}")
    return outcome
