"""Unified CLI for remove-ai-watermarks.

Provides commands for:
  - Visible watermark removal (Gemini sparkle) - works offline, fast
  - Invisible watermark removal (SynthID etc.) - requires GPU/diffusion models
  - AI metadata stripping - lightweight, no ML deps needed
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn

import click

from remove_ai_watermarks import __version__, image_io, watermark_registry
from remove_ai_watermarks.noai.constants import SUPPORTED_FORMATS
from remove_ai_watermarks.noai.watermark_profiles import (
    resolve_strength,
    strength_default_help,
    vendor_for_strength,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray


# ── plain-text output layer (replaces rich: no colors, no markup, no boxes) ──


class _Table:
    """Plain-text stand-in for rich.Table."""

    def __init__(self, *args: Any, title: str | None = None, **kwargs: Any) -> None:
        self._title = title
        self._headers: list[str] = []
        self._rows: list[list[str]] = []

    def add_column(self, header: str = "", *args: Any, **kwargs: Any) -> None:
        self._headers.append(str(header))

    def add_row(self, *cells: Any) -> None:
        self._rows.append([str(c) for c in cells])

    def render(self) -> str:
        lines: list[str] = []
        if self._title:
            lines.append(self._title)
        if any(self._headers):
            lines.append("  ".join(self._headers))
        lines.extend("  ".join(row) for row in self._rows)
        return "\n".join(f"  {line}" for line in lines)


class _Progress:
    """No-op stand-in for rich.Progress; results are printed directly instead."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> _Progress:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def add_task(self, *args: Any, **kwargs: Any) -> int:
        return 0

    def advance(self, *args: Any, **kwargs: Any) -> None:
        pass

    def update(self, *args: Any, **kwargs: Any) -> None:
        pass


class _Console:
    """Minimal plain-text replacement for rich.Console."""

    def print(self, *objects: Any, **kwargs: Any) -> None:
        click.echo(" ".join(o.render() if isinstance(o, _Table) else str(o) for o in objects))

    @contextlib.contextmanager
    def status(self, message: str = "", **kwargs: Any) -> Generator[None, None, None]:
        if message:
            click.echo(message)
        yield


def _panel(text: str = "", *args: Any, **kwargs: Any) -> str:
    return text


def _column(*args: Any, **kwargs: Any) -> None:
    return None


Panel = _panel
Table = _Table
Progress = _Progress
SpinnerColumn = BarColumn = TextColumn = TimeElapsedColumn = _column
console = _Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(name)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )


def _banner() -> None:
    console.print(
        Panel(
            f"Remove-AI-Watermarks v{__version__}\nVisible & invisible watermark removal",
            border_style="cyan",
            padding=(0, 2),
        )
    )


def _validate_image(path: Path) -> Path:
    if not path.exists():
        console.print(f"Error: File not found: {path}")
        raise SystemExit(1)
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        console.print(f"Warning: {path.suffix} may not be supported (expected: {', '.join(SUPPORTED_FORMATS)})")
    return path


# Shared option decorator for commands that run the invisible-watermark pipeline.
# Both cmd_invisible and cmd_all expose this flag; defining it once avoids
# copy-paste drift.
_controlnet_scale_option = click.option(
    "--controlnet-scale",
    type=float,
    default=1.0,
    help="ControlNet conditioning scale (structure/text preservation strength); "
    "applies to the controlnet pipeline (the default). Higher = closer to original structure.",
)

_min_resolution_option = click.option(
    "--min-resolution",
    type=int,
    default=1024,
    help="Upscale long side UP to this (px) before diffusion when the input is smaller, so SDXL runs "
    "near 1024 (small inputs distort at native); output is restored to the input size. 0 = off. Default 1024.",
)

_unsharp_option = click.option(
    "--unsharp", type=float, default=0.0, help="Unsharp-mask sharpening strength (0 = off, typical: 0.3-0.8)."
)

_upscaler_option = click.option(
    "--upscaler",
    type=click.Choice(["lanczos", "esrgan"]),
    default="lanczos",
    help="How to upscale a small input to the --min-resolution floor: lanczos (default, cv2, no deps) or "
    "esrgan (Real-ESRGAN via the 'esrgan' extra; better detail, slower on CPU). Best for photo/texture "
    "content -- as a generic GAN with no face/glyph prior it can degrade faces (diffusion mitigates) and "
    "thin text, so lanczos stays the default. Falls back to lanczos if the extra is absent. Only when upscaling.",
)

_auto_option = click.option(
    "--auto",
    is_flag=True,
    default=False,
    help="DEPRECATED: controlnet is already the default pipeline, so --auto now only "
    "enables --adaptive-polish (the content detectors were removed). Use "
    "--adaptive-polish instead.",
)

_adaptive_polish_option = click.option(
    "--adaptive-polish/--no-adaptive-polish",
    default=True,
    help="Restore the input's detail level after removal (capped unsharp + edge-masked grain "
    "targeting the input's sharpness, sparing text), countering the over-smoothed look. ON by "
    "default; it self-limits where there is no detail deficit (text/flat graphics), so it is a "
    "no-op there. Pass --no-adaptive-polish to disable. Independent of --unsharp/--humanize.",
)


# Tiled-diffusion knobs, shared by the diffusion commands (invisible/all/batch).
# Tiling is the lossless alternative to --max-resolution for large inputs that OOM
# on MPS/GPU: process at native resolution in overlapping, feather-blended tiles.
def _tile_options(f: Any) -> Any:
    """Apply the --tile / --tile-size / --tile-overlap options to a command."""
    f = click.option(
        "--tile-overlap",
        type=int,
        default=128,
        help="Overlap between adjacent tiles in px (feather-blended, no seam). Default 128.",
    )(f)
    f = click.option(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile dimension in px for --tile (SDXL's training size). Default 1024.",
    )(f)
    return click.option(
        "--tile/--no-tile",
        default=False,
        help="Process large images in overlapping tiles instead of one forward pass -- the lossless "
        "alternative to --max-resolution for inputs that OOM on MPS/GPU. Engages only when the long "
        "side exceeds --tile-size; pair with --max-resolution 0 (default) to keep native resolution. Default off.",
    )(f)


# HuggingFace model + CFG knobs, shared by the diffusion commands (invisible/all/batch)
# so the surface stays identical across them.
_model_option = click.option(
    "--model",
    type=str,
    default=None,
    help="HuggingFace model ID for the diffusion pipeline. Default: the SDXL base checkpoint.",
)
_guidance_scale_option = click.option(
    "--guidance-scale",
    type=float,
    default=None,
    help="Classifier-free guidance scale (CFG). Default: 7.5 (the library default). "
    "Lower = follow the prompt less / stay closer to the input.",
)


def _normalize_pipeline(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
    """Resolve the legacy ``default`` profile name to ``sdxl`` (click option callback).

    Emits a one-line deprecation notice when the user explicitly passes the outdated
    ``default`` value, pointing at the two current choices (``sdxl`` / ``controlnet``).
    """
    if value is None:
        return None
    from remove_ai_watermarks.noai.watermark_profiles import normalize_profile

    normalized = normalize_profile(value)
    if value.strip().lower() == "default":
        click.echo(
            "Warning: --pipeline default is deprecated and maps to 'sdxl'. "
            "Use --pipeline sdxl (plain SDXL) or --pipeline controlnet (the default).",
            err=True,
        )
    return normalized


# ``controlnet`` (the default-SELECTED value), ``sdxl`` (plain SDXL img2img) and
# ``qwen`` (Qwen-Image, CUDA/cloud-class) are the current profiles; ``default`` is an
# OUTDATED back-compat alias for ``sdxl`` (warned + normalized away by _normalize_pipeline).
_PIPELINE_CHOICES = ["sdxl", "controlnet", "qwen", "default"]
_PIPELINE_HELP = (
    "Pipeline profile. controlnet (DEFAULT) = SDXL + canny ControlNet that preserves "
    "text/faces via edge conditioning while removing SynthID; sdxl = plain SDXL img2img "
    "(lighter, no extra model download, but leaves SynthID on flat-graphic content); "
    "qwen = Qwen-Image (20B, Apache-2.0) img2img, best text/structure preservation but "
    "CUDA/cloud-class (does not fit MPS). ('default' is an OUTDATED alias for 'sdxl'.)"
)

# Shared --pipeline / --strength decorators so the three diffusion commands
# (invisible/all/batch) keep an identical surface and the strength help can never
# drift from the watermark_profiles constants (strength_default_help derives it).
_pipeline_option = click.option(
    "--pipeline",
    type=click.Choice(_PIPELINE_CHOICES),
    default="controlnet",
    callback=_normalize_pipeline,
    help=_PIPELINE_HELP,
)
_strength_option = click.option(
    "--strength",
    type=float,
    default=None,
    help=f"Denoising strength (0.0-1.0). Default: {strength_default_help()}.",
)
_force_option = click.option(
    "--force/--no-force",
    default=False,
    help=(
        "Run the diffusion scrub even when no invisible AI watermark is locally "
        "detectable. Default: skip it (regeneration only degrades a clean image; a "
        "skip never claims the image is watermark-free -- a pixel SynthID is "
        "undetectable once its metadata proxy is gone)."
    ),
)
_cpu_offload_option = click.option(
    "--cpu-offload/--no-cpu-offload",
    default=False,
    help=(
        "Stream pipeline submodules to the GPU on demand instead of holding the whole "
        "fp16 pipeline in VRAM (CUDA only). Lets a low-VRAM card (e.g. 8 GB) run SDXL "
        "that would otherwise OOM, at the cost of speed. Pair with --pipeline sdxl on "
        "the tightest cards. No effect on cpu/mps."
    ),
)


_visible_backend_option = click.option(
    "--backend",
    "backend",
    type=click.Choice(["auto", "cv2", "migan", "lama"]),
    default="auto",
    help="Fill backend for visible-mark removal (localize -> fill). auto: best available, "
    "LaMa > MI-GAN > cv2 (a learned backend needs the 'lama' or 'migan' extra; else cv2, "
    "with a warning). cv2: classical inpaint (no deps, smears texture). migan: MI-GAN ONNX "
    "(light, ~1 GB, the memory-tight pick). lama: big-LaMa ONNX (best quality, ~4.7 GB).",
)


_visible_sensitivity_option = click.option(
    "--sensitivity",
    "sensitivity",
    type=click.Choice(["auto", "strict"]),
    default="auto",
    help="How hard to trust a borderline mark. auto: relax a mark only when metadata "
    "or a same-product sibling mark corroborates it (safe; clean images untouched). "
    "strict: high-precision visual gate only, never relaxed. To act on a mark YOU can "
    "see but the detector missed, use 'erase --region' or '--mark <name> --no-detect' "
    "rather than a blanket relaxation.",
)


def _resolve_auto_polish(auto: bool, adaptive_polish: bool) -> bool:
    """Warn on the retired ``--auto`` flag, returning ``adaptive_polish`` unchanged.

    ``--auto`` used to plan the pipeline + polish from content detection, but the
    pipeline is now always controlnet (the default) and the adaptive polish is ON by
    default (it self-gates by detail level), so the content detectors were removed and
    ``--auto`` is now a no-op alias: the polish it used to enable is already the default,
    and an explicit ``--no-adaptive-polish`` still wins. So it only emits a deprecation
    warning and passes ``adaptive_polish`` through.
    """
    if auto:
        click.echo(
            "Warning: --auto is deprecated and now does nothing (the adaptive polish it "
            "enabled is ON by default). Use --no-adaptive-polish to turn the polish off.",
            err=True,
        )
    return adaptive_polish


def _warn_if_esrgan_unavailable(upscaler: str) -> None:
    """Tell the user once if ``--upscaler esrgan`` will silently fall back to Lanczos.

    The engine downgrades to Lanczos when the ``esrgan`` extra is absent (fail-safe, so
    a batch never breaks mid-run) -- but without this notice the user would believe
    Real-ESRGAN ran. Surfaced at the CLI layer, once per invocation (not per image).
    """
    if upscaler != "esrgan":
        return
    from remove_ai_watermarks import upscaler as _upscaler

    if not _upscaler.is_available():
        console.print("  Note: --upscaler esrgan needs the 'esrgan' extra; falling back to Lanczos.")


def _visible_provenance(path: Path | None) -> frozenset[str]:
    """Vendor keys local metadata confirms, the EVIDENCE that drives ``auto``
    sensitivity. Thin wrapper over the public :func:`api.visible_provenance` (one
    implementation for the CLI and the library), with a None-path guard."""
    if path is None:
        return frozenset()
    from remove_ai_watermarks.api import visible_provenance

    return visible_provenance(path)


def _remove_visible_auto(
    image: NDArray[Any],
    *,
    source_path: Path | None = None,
    backend: str = "auto",
    sensitivity: str = "auto",
) -> tuple[NDArray[Any], str | None]:
    """Remove every auto-detected visible mark via the registry (localize -> fill).

    Routes the ``all``/``batch`` visible step through the same registry path the
    standalone ``visible`` command uses, so EVERY registered mark is handled (the
    Gemini sparkle AND the Doubao/Jimeng/Samsung text marks), not just the sparkle.
    Returns ``(result, label-or-None)``; when no ``in_auto`` mark fires the image is
    returned unchanged with ``None``. ``backend`` selects the shared fill; ``sensitivity``
    controls how hard a borderline mark is trusted (auto reads metadata provenance)."""
    from remove_ai_watermarks import watermark_registry

    bk: watermark_registry.Backend = backend  # type: ignore[assignment]
    sens = _parse_sensitivity(sensitivity)
    provenance = _visible_provenance(source_path)
    try:
        result, removed = watermark_registry.remove_auto_marks(
            image, sensitivity=sens, provenance=provenance, backend=bk
        )
    except RuntimeError as e:  # e.g. a selected migan/lama backend whose extra is absent
        console.print(f"  Error: {e}")
        raise SystemExit(1) from e
    if not removed:
        return image, None
    return result, ", ".join(removed)


def _parse_sensitivity(value: str) -> watermark_registry.Sensitivity:
    """Map the CLI ``--sensitivity`` choice to the registry literal.

    A pass-through since ``assume-ai`` was removed (2026-07-19); kept as the single
    conversion point so a future kebab-cased choice has an obvious home.
    """
    return "strict" if value == "strict" else "auto"


# Exit code for the standalone ``visible`` command when no visible mark was
# removed -- distinct from success (0) and a hard error (1) so a wrapping
# service can tell "nothing to do here" apart and surface guidance instead of
# re-serving the unchanged input as a finished result.
EXIT_NO_VISIBLE_MARK = 2


def _write_output_or_exit(output: Path, bgr: NDArray[Any], alpha: NDArray[Any] | None) -> None:
    """Write the final image, or fail with a readable error instead of a traceback.

    `image_io.imwrite` is contractually NON-RAISING: it returns False when the codec
    rejects the image or the path cannot be written. Every caller here follows its write
    with `output.stat()` to report the size, so a silently-failed write (read-only
    directory, full disk) died with a bare `FileNotFoundError` traceback pointing at the
    stat, not at the write. Found by the Tier E adversarial sweep 2026-07-20.
    Regression: `tests/test_cli_robustness.py::TestFailedWriteIsReported`.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if not image_io.write_bgr_with_alpha(output, bgr, alpha):
        console.print(f"  Error: failed to write output (is the destination writable?): {output}")
        raise SystemExit(1)


def _no_visible_mark_exit(source: Path) -> NoReturn:
    """Explain why no visible watermark was removed, then exit non-zero.

    The visible registry handles only known visual marks (the Gemini sparkle and
    the Doubao/Jimeng/Qwen/Samsung text strips). Most real uploads carry no such mark
    -- frequently an invisible/metadata watermark instead (e.g. an OpenAI or
    Gemini image whose only signal is C2PA + SynthID). Returning the input
    unchanged with exit 0 reads as success to a caller and re-serves the
    watermarked image -- the recurring "it didn't work" report. Instead, run a
    cheap metadata-only :func:`identify`, tell the user what the image actually
    carries and which command removes it, and exit
    :data:`EXIT_NO_VISIBLE_MARK`.

    When the user can SEE a mark the detector missed, the honest next step is one that
    executes their instruction rather than guessing harder. This used to recommend
    ``--sensitivity assume-ai``, which did the opposite -- it relaxed every mark's gate
    on a blanket assumption -- and that mode is gone (2026-07-19).

    The advice is per-mark, because the forced paths are not equally reliable
    (measured 2026-07-19):
      * ``erase --region`` is always sound: the user supplies the coordinates, so there
        is nothing to guess. This is the primary recommendation.
      * ``--mark <text-mark> --no-detect`` is reasonable for the TEXT marks: the forced
        mask is built from the actual glyph blob, non-empty on 13/13 real marks the
        detector missed.
      * ``--mark gemini --no-detect`` is NOT recommended and is deliberately not
        suggested here: with no detection it falls back to a fixed default sparkle slot,
        which covered the real sparkle on only **31% of 97** genuine sparkles the strict
        gate missed (median offset 63px up-and-left). The other 69% fill a clean corner
        AND report a removal that did not happen -- the worst outcome the tool has.
    """
    from remove_ai_watermarks.identify import identify

    report = identify(source, check_visible=False, check_invisible=False)
    if report.is_ai_generated and report.watermarks:
        plat = report.platform or "an unidentified platform"
        console.print(
            f"  This image carries an invisible/metadata watermark ({plat}), not a visible mark,\n"
            "  so the 'visible' command cannot remove it. Run the full pipeline instead:\n"
            f"    remove-ai-watermarks all {source.name}"
        )
    else:
        console.print(
            "  No visible mark and no readable AI provenance signal. This does not prove\n"
            "  the image is clean: an invisible pixel watermark such as SynthID cannot be\n"
            "  detected here once the metadata proxy is absent (it may have been stripped\n"
            "  earlier). If the image is AI-generated, regenerate the pixels with:\n"
            f"    remove-ai-watermarks all {source.name}\n"
            "  If instead there is a logo or object to remove, target it with the region eraser:\n"
            f"    remove-ai-watermarks erase {source.name} --region x,y,w,h"
        )
    console.print(
        "  If you can SEE a mark here that was not detected, point at it directly --\n"
        "  that removes what you actually see instead of guessing:\n"
        f"    remove-ai-watermarks erase {source.name} --region x,y,w,h\n"
        "  For a known CJK text mark you can also force it by name:\n"
        f"    remove-ai-watermarks visible {source.name} --mark doubao --no-detect"
    )
    raise SystemExit(EXIT_NO_VISIBLE_MARK)


# Same value as EXIT_NO_VISIBLE_MARK (2): a distinct-from-success / distinct-from-
# error code that tells a wrapping service (raiw.cc) "the diffusion scrub was skipped
# because no invisible watermark was locally detectable", so it can surface the
# message instead of charging for and serving an unchanged image as done.
EXIT_NO_INVISIBLE_SIGNAL = 2


def _no_invisible_signal_exit(source: Path) -> NoReturn:
    """Explain why the diffusion scrub was skipped, then exit non-zero.

    The ``invisible`` command regenerates pixels to remove SynthID / open
    watermarks; that regeneration also degrades a real photo. When
    :func:`identify` finds no locally-detectable invisible AI signal, running it
    anyway would damage a clean image for nothing -- the dominant paid score-0
    cause on no-watermark uploads. So skip it, but do NOT imply the image is
    clean: a pixel SynthID is undetectable here once its metadata proxy is gone.
    Write no output and exit :data:`EXIT_NO_INVISIBLE_SIGNAL`; ``--force`` runs
    the scrub regardless.
    """
    console.print(
        "  No invisible AI watermark detected (no C2PA/SynthID proxy, no open\n"
        "  watermark). Skipped the diffusion scrub -- regenerating the pixels would\n"
        "  only degrade the image with nothing to remove, so no output was written.\n"
        "  This does NOT prove the image is clean: a pixel watermark such as SynthID\n"
        "  cannot be detected here once its metadata proxy is absent (it may have\n"
        "  been stripped earlier). If you know the image is AI-generated and want the\n"
        "  pixels regenerated regardless, re-run with --force:\n"
        f"    remove-ai-watermarks invisible {source.name} --force"
    )
    raise SystemExit(EXIT_NO_INVISIBLE_SIGNAL)


def _should_skip_invisible_scrub(force: bool, image_path: Path) -> bool:
    """True when the diffusion scrub should be skipped for *image_path*.

    The shared no-signal gate for ``invisible`` / ``all`` / ``batch``: skip when
    ``--force`` is not set AND no invisible AI watermark is locally detectable
    (regenerating pixels would only degrade a clean image -- the dominant paid
    score-0 cause). Centralizes the condition + the lazy ``has_invisible_target``
    import so the three call sites cannot drift. ``--force`` short-circuits the
    detection entirely.
    """
    if force:
        return False
    from remove_ai_watermarks.identify import has_invisible_target

    return not has_invisible_target(image_path)


# ── Main group ──
@click.group(invoke_without_command=True)
@click.version_option(__version__, prog_name="remove-ai-watermarks")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Remove visible and invisible AI watermarks from images."""
    from dotenv import load_dotenv

    load_dotenv()  # Load .env (e.g. HF_TOKEN)

    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)

    if ctx.invoked_subcommand is None:
        _banner()
        click.echo(ctx.get_help())


# ── Visible (Gemini) watermark removal ──
def _run_visible_auto(
    source: Path,
    output: Path,
    *,
    backend: watermark_registry.Backend,
    sensitivity: watermark_registry.Sensitivity,
    strip_metadata: bool,
) -> None:
    """Run the registry-wide visible pass and render its CLI result."""
    from remove_ai_watermarks import api

    t0 = time.monotonic()
    try:
        with console.status("Detecting & removing visible marks..."):
            result, removed = api.remove_visible(
                str(source),
                str(output),
                sensitivity=sensitivity,
                backend=backend,
                strip_metadata=strip_metadata,
                write_noop=False,
            )
    except RuntimeError as e:  # selected migan/lama backend whose extra is absent
        console.print(f"  Error: {e}")
        raise SystemExit(1) from e
    except (ValueError, OSError) as e:
        # Covers BOTH an unreadable input and an unwritable output, so the message must
        # not assert which: it used to say "cannot read image <input>" while quoting the
        # OUTPUT path, blaming the wrong file (Tier E, 2026-07-20).
        console.print(f"  Error: {e}")
        raise SystemExit(1) from e

    elapsed = time.monotonic() - t0
    h, w = result.shape[:2]
    console.print(f"  Input:  {source.name}  ({w}x{h})")
    if not removed:
        # write_noop=False means nothing was written, so a pre-existing output is intact.
        console.print("  No known visible mark detected (gemini / doubao / jimeng / jimeng-pill / samsung).")
        _no_visible_mark_exit(source)
    console.print(f"  Removed: {', '.join(removed)}")
    size_kb = output.stat().st_size / 1024
    console.print(f"  Saved: {output}  ({size_kb:.0f} KB, {elapsed:.2f}s)")


def _run_visible_explicit(
    ctx: click.Context,
    source: Path,
    output: Path,
    *,
    detect: bool,
    mark: str,
    backend: watermark_registry.Backend,
    sensitivity: watermark_registry.Sensitivity,
    resolved_backend: str,
    strip_metadata: bool,
) -> None:
    """Run one explicitly selected visible-mark detector/remover."""
    image, alpha = image_io.read_bgr_and_alpha(source)
    if image is None:
        console.print(f"Error: Failed to read image: {source}")
        raise SystemExit(1)
    h, w = image.shape[:2]
    console.print(f"  Input:  {source.name}  ({w}x{h})")

    provenance = _visible_provenance(source)
    target = "gemini" if mark == "auto" else mark  # --no-detect auto: gemini fallback
    chosen = watermark_registry.get_mark(target)
    # A single explicit mark has no sibling corroboration. Keep its trust resolution
    # aligned with the registry arbiter.
    trust = watermark_registry.resolve_trust(
        chosen.key,
        sensitivity=sensitivity,
        provenance=provenance,
        strict_keys=set(),
    )
    relax = trust != "strict"
    detection = chosen.detect(image, provenance=relax)
    if detect and not detection.detected:
        console.print(f"  {chosen.label} not detected  (conf {detection.confidence:.2f}). Use --no-detect to force.")
        _no_visible_mark_exit(source)
    if detection.detected:
        console.print(f"  {chosen.label} detected  ({chosen.location}, conf {detection.confidence:.2f})")

    t0 = time.monotonic()
    try:
        with console.status(f"Removing {chosen.label}... ({resolved_backend})"):
            result, _ = chosen.remove(image, backend=backend, provenance=relax, force=not detect)
    except RuntimeError as e:  # selected migan/lama backend whose extra is absent
        console.print(f"  Error: {e}")
        raise SystemExit(1) from e
    elapsed = time.monotonic() - t0

    _write_output_or_exit(output, result, alpha)
    if strip_metadata:
        try:
            from remove_ai_watermarks.metadata import remove_ai_metadata

            remove_ai_metadata(output, output)
        except Exception as e:
            if ctx.obj.get("verbose"):
                console.print(f"  Warning: Failed to strip metadata: {e}")

    size_kb = output.stat().st_size / 1024
    console.print(f"  Saved: {output}  ({size_kb:.0f} KB, {elapsed:.2f}s)")


@main.command("visible")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o", "--output", type=click.Path(path_type=Path), default=None, help="Output path (default: <source>_clean.<ext>)."
)
@click.option("--detect/--no-detect", default=True, help="Detect watermark before removal.")
@click.option(
    "--mark",
    type=click.Choice(["auto", *watermark_registry.mark_keys()]),
    default="auto",
    help="Which known visible mark to target (auto picks every detected mark). "
    "The fill backend is chosen by --backend (default auto).",
)
@_visible_backend_option
@_visible_sensitivity_option
@click.option("--strip-metadata/--keep-metadata", default=True, help="Strip AI metadata from output.")
@click.pass_context
def cmd_visible(
    ctx: click.Context,
    source: Path,
    output: Path | None,
    detect: bool,
    mark: str,
    backend: str,
    sensitivity: str,
    strip_metadata: bool,
) -> None:
    """Remove a known visible AI watermark from an image.

    Finds a known mark in its usual place (Gemini sparkle / Doubao-Jimeng-Qwen-Samsung
    text) via the watermark registry and removes it by LOCALIZING the mark to a mask
    and filling that mask with the chosen ``--backend`` (auto: best available, LaMa >
    MI-GAN > cv2). ``--mark auto`` removes every detected mark in one
    pass. For arbitrary logos/objects, use ``erase``.
    """
    _banner()
    source = _validate_image(source)

    if output is None:
        output = source.with_stem(source.stem + "_clean")

    bk: watermark_registry.Backend = backend  # type: ignore[assignment]
    sens = _parse_sensitivity(sensitivity)
    resolved_backend = watermark_registry.resolve_backend(bk)
    if resolved_backend == "cv2" and not watermark_registry.inpaint_model_available():
        console.print("  Note: using cv2 fill (install the 'migan' extra for a lightweight ONNX model).")

    # ``auto`` removes EVERY detected in_auto mark in one pass (a Jimeng-basic image
    # carries the top-left pill AND the bottom-right wordmark). Delegate the whole
    # read -> provenance -> localize/fill -> write -> metadata-strip to the library
    # entry point, so the CLI and the library go through ONE path (no drift).
    if mark == "auto" and detect:
        _run_visible_auto(source, output, backend=bk, sensitivity=sens, strip_metadata=strip_metadata)
        return

    _run_visible_explicit(
        ctx,
        source,
        output,
        detect=detect,
        mark=mark,
        backend=bk,
        sensitivity=sens,
        resolved_backend=resolved_backend,
        strip_metadata=strip_metadata,
    )


# ── Universal region eraser ──
def _parse_region(spec: str) -> tuple[int, int, int, int]:
    """Parse an ``x,y,w,h`` region string into a 4-int tuple."""
    parts = spec.replace(" ", "").split(",")
    if len(parts) != 4:
        raise click.BadParameter(f"region must be 'x,y,w,h', got: {spec!r}")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError as e:
        raise click.BadParameter(f"region values must be integers: {spec!r}") from e
    if w <= 0 or h <= 0:
        raise click.BadParameter(f"region width/height must be positive: {spec!r}")
    return x, y, w, h


@main.command("erase")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--region", "regions", multiple=True, required=True, help="x,y,w,h box to erase (repeatable).")
@click.option(
    "-o", "--output", type=click.Path(path_type=Path), default=None, help="Output path (default: <source>_clean.<ext>)."
)
@click.option(
    "--backend",
    type=click.Choice(["cv2", "migan", "lama"]),
    default="cv2",
    help="Inpaint backend. cv2: instant, no deps. migan: light ONNX MI-GAN, ~1 GB RAM, "
    "near-LaMa quality (extra 'migan'). lama: big-LaMa, best quality but ~4.7 GB RAM (extra 'lama').",
)
@click.option("--inpaint-method", type=click.Choice(["telea", "ns"]), default="telea", help="cv2 inpaint method.")
@click.option("--dilate", type=int, default=3, help="Grow the box by this many px before inpainting.")
@click.option("--strip-metadata/--keep-metadata", default=True, help="Strip AI metadata from output.")
@click.pass_context
def cmd_erase(
    ctx: click.Context,
    source: Path,
    regions: tuple[str, ...],
    output: Path | None,
    backend: Literal["cv2", "migan", "lama"],
    inpaint_method: str,
    dilate: int,
    strip_metadata: bool,
) -> None:
    """Erase arbitrary region(s) from an image via inpainting.

    Universal and position-agnostic: removes any logo / watermark / object inside
    the boxes you pass, regardless of color or location. Runs on CPU. Use this
    for marks the dedicated ``visible`` engines (Gemini, Doubao) do not cover.
    """
    from remove_ai_watermarks.region_eraser import erase

    _banner()
    source = _validate_image(source)
    if output is None:
        output = source.with_stem(source.stem + "_clean")

    boxes = [_parse_region(r) for r in regions]

    image, alpha = image_io.read_bgr_and_alpha(source)
    if image is None:
        console.print(f"Error: Failed to read image: {source}")
        raise SystemExit(1)
    h, w = image.shape[:2]
    console.print(f"  Input:  {source.name}  ({w}x{h})  {len(boxes)} region(s), backend={backend}")

    t0 = time.monotonic()
    method: Literal["telea", "ns"] = "ns" if inpaint_method == "ns" else "telea"
    try:
        with console.status(f"Erasing ({backend})..."):
            result = erase(image, boxes=boxes, backend=backend, dilate=dilate, cv2_method=method)
    except RuntimeError as e:
        console.print(f"  Error: {e}")
        raise SystemExit(1) from e
    elapsed = time.monotonic() - t0

    _write_output_or_exit(output, result, alpha)

    if strip_metadata:
        try:
            from remove_ai_watermarks.metadata import remove_ai_metadata

            remove_ai_metadata(output, output)
        except Exception as e:
            if ctx.obj.get("verbose"):
                console.print(f"  Warning: Failed to strip metadata: {e}")

    size_kb = output.stat().st_size / 1024
    console.print(f"  Erased {len(boxes)} region(s) -> {output}  ({size_kb:.0f} KB, {elapsed:.2f}s)")


# ── Invisible watermark removal ──
@main.command("invisible")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o", "--output", type=click.Path(path_type=Path), default=None, help="Output path (default: <source>_clean.<ext>)."
)
@_strength_option
@click.option("--steps", type=int, default=50, help="Number of denoising steps. Default: 50.")
@_pipeline_option
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "mps", "cuda", "xpu"]),
    default="auto",
    help="Inference device.",
)
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility.")
@click.option("--hf-token", type=str, default=None, help="HuggingFace API token.")
@click.option(
    "--humanize", type=float, default=0.0, help="Analog Humanizer film grain intensity (0 = off, typical: 2.0-6.0)."
)
@click.option(
    "--max-resolution",
    type=int,
    default=0,
    help="Cap long side (px) before diffusion; 0 = native (best quality, like raiw.cc). Raise only on GPU/MPS OOM.",
)
@_controlnet_scale_option
@_min_resolution_option
@_unsharp_option
@_upscaler_option
@_model_option
@_guidance_scale_option
@_auto_option
@_adaptive_polish_option
@_tile_options
@_force_option
@_cpu_offload_option
@click.pass_context
def cmd_invisible(
    ctx: click.Context,
    source: Path,
    output: Path | None,
    strength: float | None,
    steps: int,
    pipeline: str,
    device: str,
    seed: int | None,
    hf_token: str | None,
    humanize: float,
    unsharp: float,
    max_resolution: int,
    min_resolution: int,
    controlnet_scale: float,
    upscaler: str,
    model: str | None,
    guidance_scale: float | None,
    auto: bool,
    adaptive_polish: bool,
    tile: bool,
    tile_size: int,
    tile_overlap: int,
    force: bool,
    cpu_offload: bool,
) -> None:
    """Remove invisible AI watermarks (SynthID, StableSignature, TreeRing).

    Uses diffusion-based regeneration. Requires GPU for reasonable speed.
    Requires the [gpu] extra: pip install 'remove-ai-watermarks[gpu]'
    """
    from remove_ai_watermarks.invisible_engine import is_available as invisible_available

    if not invisible_available():
        console.print(
            "Error: GPU dependencies not installed.\n  Install them with: pip install 'remove-ai-watermarks[gpu]'"
        )
        raise SystemExit(1)

    from remove_ai_watermarks.invisible_engine import InvisibleEngine

    source = _validate_image(source)
    _warn_if_esrgan_unavailable(upscaler)
    adaptive_polish = _resolve_auto_polish(auto, adaptive_polish)
    if output is None:
        output = source.with_stem(source.stem + "_clean")

    device_str = None if device == "auto" else device

    # Gate BEFORE building the engine: skip the destructive regeneration when no
    # invisible AI watermark is locally detectable (it would only degrade a clean
    # image -- dominant paid score-0 cause), so the common skip path pays nothing for
    # engine construction. A skip never claims the image is clean; --force overrides.
    if _should_skip_invisible_scrub(force, source):
        _no_invisible_signal_exit(source)

    def progress_cb(msg: str) -> None:
        console.print(f"  {msg}")

    engine = InvisibleEngine(
        model_id=model,
        device=device_str,
        pipeline=pipeline,
        hf_token=hf_token,
        progress_callback=progress_cb,
        controlnet_conditioning_scale=controlnet_scale,
        cpu_offload=cpu_offload,
    )

    # Detect the SynthID vendor from the ORIGINAL (before processing strips C2PA) so the
    # displayed and executed strength agree on the vendor-adaptive default.
    vendor = vendor_for_strength(source)
    console.print(f"  Input:    {source.name}")
    console.print(f"  Pipeline: {pipeline}")
    console.print(f"  Strength: {resolve_strength(strength, vendor, pipeline)}  Steps: {steps}")

    t0 = time.monotonic()
    result_path = engine.remove_watermark(
        image_path=source,
        output_path=output,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        humanize=humanize,
        unsharp=unsharp,
        adaptive_polish=adaptive_polish,
        max_resolution=max_resolution,
        min_resolution=min_resolution,
        upscaler=upscaler,
        vendor=vendor,
        tile=tile,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    elapsed = time.monotonic() - t0

    size_kb = result_path.stat().st_size / 1024
    console.print(f"\n  Saved: {result_path}  ({size_kb:.0f} KB, {elapsed:.1f}s)")


# ── Metadata operations ──
@main.command("metadata")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--check", is_flag=True, help="Check for AI metadata (don't modify).")
@click.option("--remove", is_flag=True, help="Remove AI metadata.")
@click.option(
    "-o", "--output", type=click.Path(path_type=Path), default=None, help="Output path (default: overwrite source)."
)
@click.option("--keep-standard/--remove-all", default=True, help="Keep standard metadata (Author, Title, etc.).")
@click.pass_context
def cmd_metadata(
    ctx: click.Context,
    source: Path,
    check: bool,
    remove: bool,
    output: Path | None,
    keep_standard: bool,
) -> None:
    """Check or remove AI-generation metadata (images, video, and audio).

    Strips EXIF AI tags, PNG text chunks, C2PA provenance manifests, and the
    China TC260 AIGC label. Beyond images (PNG/JPEG/WebP/AVIF/HEIF/JXL) it also
    strips provenance metadata from MP4/MOV/M4V/M4A containers and, via ffmpeg,
    from WebM/MP3/WAV/FLAC/OGG. The coded image, audio, and video data are left
    untouched.
    """
    from remove_ai_watermarks.metadata import get_ai_metadata, has_ai_metadata, strip_and_verify

    # No _validate_image() here: unlike the image-only commands, metadata also
    # accepts video/audio containers, so the image-format warning would misfire.
    # click's `exists=True` on the argument already enforces the file exists.
    _banner()

    if check or (not remove):
        has_ai = has_ai_metadata(source)
        if has_ai:
            console.print(f"  Warning: AI metadata detected in {source.name}:")
            meta = get_ai_metadata(source)
            if synthid := meta.get("synthid_watermark"):
                console.print(f"  Warning: SynthID watermark (inferred from C2PA metadata) {synthid}")
            table = Table(show_header=True, header_style="bold")
            table.add_column("Key", style="cyan")
            table.add_column("Value")
            for k, v in meta.items():
                table.add_row(k, str(v)[:80])
            console.print(table)
        else:
            console.print(f"  No AI metadata found in {source.name}")

        if not remove:
            return

    # Remove
    try:
        out, leftover = strip_and_verify(source, output, keep_standard=keep_standard)
    except (OSError, ValueError) as e:  # unreadable / truncated / non-image (PIL raises OSError subclasses)
        console.print(f"  Error: cannot process {source.name}: {e}")
        raise SystemExit(1) from e

    if leftover:
        console.print(f"  FAILED: {len(leftover)} AI metadata marker(s) survived in {out}")
        console.print(f"    still present: {', '.join(sorted(leftover))}")
        console.print("    the file could not be decoded, so it was copied through unchanged")
        raise SystemExit(1)
    console.print(f"  AI metadata stripped -> {out}")


# ── Provenance identification ──
@main.command("identify")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--no-visible",
    is_flag=True,
    help="Skip pixel-domain detectors (visible sparkle + invisible watermark); metadata-only.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the report as JSON instead of a table.")
@click.pass_context
def cmd_identify(ctx: click.Context, source: Path, no_visible: bool, as_json: bool) -> None:
    """Identify where an image was made and what watermarks it carries.

    Aggregates C2PA Content Credentials, IPTC "Made with AI" tags, embedded
    generation parameters, the SynthID metadata proxy, and the visible Gemini
    sparkle into a single provenance verdict. Absence of signals is reported as
    "unknown", never as "clean" (stripped metadata leaves no local proof).
    """
    from dataclasses import asdict

    from remove_ai_watermarks.identify import identify

    source = _validate_image(source)
    report = identify(source, check_visible=not no_visible, check_invisible=not no_visible)

    if as_json:
        click.echo(json.dumps(asdict(report), default=str, indent=2))
        return

    _banner()
    verdict = {True: "AI-generated", False: "not AI", None: "unknown"}[report.is_ai_generated]
    # Sharpen the True verdict when the C2PA source type says the image is a real
    # photo with an AI-composited region rather than a full AI generation, so the
    # caller (and the user) can tell "scrub the whole frame" from "scrub the AI region".
    if report.is_ai_generated and report.ai_source_kind == "enhanced":
        verdict = "AI-enhanced (real content with an AI-composited region)"
    elif report.is_ai_generated and report.ai_source_kind == "generated":
        verdict = "AI-generated (fully synthetic)"
    console.print(f"\n  Verdict: {verdict}  (confidence: {report.confidence})")
    console.print(f"  Platform: {report.platform or 'undetermined'}")

    if report.is_ai_generated is None:
        console.print(
            "  No locally-readable AI signal found. This is not the same as 'clean': "
            "metadata is often stripped by re-encoding, screenshots, or upload, and SynthID-class "
            "pixel watermarks (Gemini / Nano Banana / gpt-image) have no local detector. "
            "See caveats below."
        )

    if report.integrity_clashes:
        console.print("\n  Warning: Integrity clash (provenance signals contradict each other)")
        for clash in report.integrity_clashes:
            console.print(f"  - {clash}")

    if report.watermarks:
        table = Table(show_header=True, header_style="bold", title="Watermarks / provenance markers")
        table.add_column("Marker", style="cyan")
        for wm in report.watermarks:
            table.add_row(wm)
        console.print(table)
    else:
        console.print("  No watermarks or provenance markers found.")

    if report.caveats:
        console.print("\n  Caveats:")
        for c in report.caveats:
            console.print(f"  - {c}")


# ── Combined "all" mode ──
@main.command("all")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o", "--output", type=click.Path(path_type=Path), default=None, help="Output path (default: <source>_clean.<ext>)."
)
@_visible_backend_option
@_visible_sensitivity_option
@_strength_option
@click.option("--steps", type=int, default=50, help="Number of denoising steps for invisible removal.")
@_pipeline_option
@_model_option
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "mps", "cuda", "xpu"]),
    default="auto",
    help="Inference device.",
)
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility.")
@click.option("--hf-token", type=str, default=None, help="HuggingFace API token.")
@click.option(
    "--humanize", type=float, default=0.0, help="Analog Humanizer film grain intensity (0 = off, typical: 2.0-6.0)."
)
@click.option(
    "--max-resolution",
    type=int,
    default=0,
    help="Cap long side (px) before diffusion; 0 = native (best quality, like raiw.cc). Raise only on GPU/MPS OOM.",
)
@_controlnet_scale_option
@_min_resolution_option
@_unsharp_option
@_upscaler_option
@_guidance_scale_option
@_auto_option
@_adaptive_polish_option
@_tile_options
@_force_option
@_cpu_offload_option
@click.pass_context
def cmd_all(
    ctx: click.Context,
    source: Path,
    output: Path | None,
    backend: str,
    sensitivity: str,
    strength: float | None,
    steps: int,
    pipeline: str,
    model: str | None,
    device: str,
    seed: int | None,
    hf_token: str | None,
    humanize: float,
    unsharp: float,
    max_resolution: int,
    min_resolution: int,
    controlnet_scale: float,
    upscaler: str,
    guidance_scale: float | None,
    auto: bool,
    adaptive_polish: bool,
    tile: bool,
    tile_size: int,
    tile_overlap: int,
    force: bool,
    cpu_offload: bool,
) -> None:
    """Remove ALL watermarks: visible + invisible + metadata.

    Runs the full pipeline in order:
      1. Visible watermark removal (Gemini sparkle / text marks, localize -> fill)
      2. Invisible watermark removal (SynthID etc., diffusion regeneration)
      3. AI metadata stripping (EXIF, PNG text, C2PA)

    If invisible watermark deps are not installed, skips step 2 with a warning.
    """
    _banner()
    source = _validate_image(source)
    _warn_if_esrgan_unavailable(upscaler)
    adaptive_polish = _resolve_auto_polish(auto, adaptive_polish)

    if output is None:
        output = source.with_stem(source.stem + "_clean")

    t0 = time.monotonic()

    # Tracks whether step 2 (invisible / SynthID removal) was skipped because the
    # GPU extra is missing. A skipped step 2 still produces an output file (visible
    # mark + metadata stripped), so without a loud end-of-run notice + non-zero exit
    # the user mistakes it for a clean result and ships an image that still carries
    # the invisible watermark (recurring reports: #14, #47).
    synthid_skipped = False

    # Use a temp file for intermediate results so the user doesn't see
    # a partial output file during long model downloads.
    import tempfile

    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=source.suffix)
    tmp_path = Path(tmp_path_str)
    try:
        import os

        os.close(tmp_fd)

        # ── Step 1: Visible watermark ──
        console.print("\n  1) Visible watermark removal")
        image, alpha = image_io.read_bgr_and_alpha(source)
        if image is None:
            console.print(f"Error: Failed to read image: {source}")
            raise SystemExit(1)

        h, w = image.shape[:2]
        console.print(f"    Input: {source.name}  ({w}x{h})")

        with console.status("Removing visible watermark..."):
            result, removed_label = _remove_visible_auto(
                image, source_path=source, backend=backend, sensitivity=sensitivity
            )
            if removed_label is not None:
                console.print(f"    Visible watermark removed ({removed_label})")
            else:
                console.print("    Skipped (no visible watermark detected)")

        # Save to temp file for invisible engine input (preserve alpha if present)
        image_io.write_bgr_with_alpha(tmp_path, result, alpha)

        # ── Step 2: Invisible watermark ──
        console.print("\n  2) Invisible watermark removal")
        from remove_ai_watermarks.invisible_engine import is_available as invisible_available

        if not invisible_available():
            synthid_skipped = True
            console.print(
                "    Warning: Skipped - GPU dependencies not installed.\n"
                "    Install them with: pip install 'remove-ai-watermarks[gpu]'"
            )
        elif _should_skip_invisible_scrub(force, source):
            # No locally-detectable invisible watermark -> skip the destructive
            # regeneration (it would only degrade the image). The visible-removed
            # pixels in tmp_path are kept and step 3 still strips metadata, so this
            # is a SUCCESS (exit 0), unlike the GPU-missing skip above. Read the
            # pristine `source`, not tmp_path whose C2PA the visible pass already
            # dropped. Not a clean-image guarantee; --force overrides.
            console.print(
                "    Skipped (no invisible AI watermark detected; pixels left intact).\n"
                "    Not a clean-image guarantee: a pixel SynthID is undetectable once its\n"
                "    metadata proxy is gone. Re-run with --force to scrub regardless."
            )
        else:
            from remove_ai_watermarks.invisible_engine import InvisibleEngine

            device_str = None if device == "auto" else device

            def progress_cb(msg: str) -> None:
                console.print(f"    {msg}")

            inv_engine = InvisibleEngine(
                model_id=model,
                device=device_str,
                pipeline=pipeline,
                hf_token=hf_token,
                progress_callback=progress_cb,
                controlnet_conditioning_scale=controlnet_scale,
                cpu_offload=cpu_offload,
            )

            # Detect the vendor from the pristine ORIGINAL (`source`); `tmp_path` has
            # already lost its C2PA to the visible-removal pass, so reading it would
            # always resolve to the unknown-vendor default.
            vendor = vendor_for_strength(source)
            console.print(f"    Strength: {resolve_strength(strength, vendor, pipeline)}  Steps: {steps}")
            inv_engine.remove_watermark(
                image_path=tmp_path,
                output_path=tmp_path,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                humanize=humanize,
                unsharp=unsharp,
                adaptive_polish=adaptive_polish,
                max_resolution=max_resolution,
                min_resolution=min_resolution,
                upscaler=upscaler,
                vendor=vendor,
                tile=tile,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )
            console.print("    Invisible watermark removed")

        # ── Step 3: Metadata ──
        console.print("\n  3) AI metadata stripping")
        try:
            from remove_ai_watermarks.metadata import remove_ai_metadata

            remove_ai_metadata(tmp_path, tmp_path)
            console.print("    AI metadata stripped")
        except Exception as e:
            console.print(f"    Warning: Metadata strip failed: {e}")

        # ── Write final result ──
        # The invisible step (and downstream cv2.IMREAD_COLOR paths) drops alpha,
        # so re-attach the original alpha plane unchanged when writing the final
        # output for transparent formats.
        final_bgr, _ = image_io.read_bgr_and_alpha(tmp_path)
        if final_bgr is None:
            console.print(f"Error: Failed to read intermediate file: {tmp_path}")
            raise SystemExit(1)
        _write_output_or_exit(output, final_bgr, alpha)

    finally:
        # Clean up temp file if it still exists
        if tmp_path.exists():
            tmp_path.unlink()

    # ── Done ──
    elapsed = time.monotonic() - t0
    size_kb = output.stat().st_size / 1024
    console.print(f"\n  Done: {output}  ({size_kb:.0f} KB, {elapsed:.1f}s total)")

    # A skipped invisible step is the single most common "it didn't work" report:
    # the output looks processed but still carries the SynthID watermark. Make that
    # impossible to miss -- a prominent banner plus a non-zero exit so scripts and
    # batch callers can detect the incomplete run instead of trusting the file.
    if synthid_skipped:
        console.print(
            "\n  =====================================================================\n"
            "  WARNING: the invisible (SynthID) watermark was NOT removed.\n"
            "  Step 2 was skipped because the GPU dependencies are not installed,\n"
            "  so this output still carries the invisible watermark -- only the\n"
            "  visible mark and metadata were stripped.\n"
            "\n"
            "  Install the extra and rerun to remove it:\n"
            "    pip install 'remove-ai-watermarks[gpu]'\n"
            "  ====================================================================="
        )
        raise SystemExit(1)


# ── Batch command ──
def _passthrough_copy(img_path: Path, out_path: Path) -> None:
    """Copy the input's pixels through to ``out_path`` unchanged (the invisible-mode skip
    paths), so the output dir stays complete without touching the pixels."""
    src_bgr, src_alpha = image_io.read_bgr_and_alpha(img_path)
    if src_bgr is not None and not image_io.write_bgr_with_alpha(out_path, src_bgr, src_alpha):
        # The point of this copy is to keep the output dir COMPLETE. A silently-dropped
        # copy defeats that and leaves a hole the caller cannot see (Tier E, 2026-07-20).
        raise OSError(f"failed to copy input through to output: {out_path}")


@dataclass(frozen=True)
class _BatchOptions:
    """Validated processing options shared by every image in one batch.

    Click necessarily exposes these as individual command parameters, but the
    processing core should receive one coherent value instead of a 21-argument
    call. Keeping the object immutable also makes it safe to reuse while the
    batch caches model instances in ``ctx.obj``.
    """

    strength: float | None
    steps: int
    pipeline: str
    device: str
    seed: int | None
    hf_token: str | None
    humanize: float
    backend: str = "auto"
    sensitivity: str = "auto"
    unsharp: float = 0.0
    max_resolution: int = 0
    min_resolution: int = 1024
    controlnet_scale: float = 1.0
    upscaler: str = "lanczos"
    model: str | None = None
    guidance_scale: float | None = None
    adaptive_polish: bool = False
    tile: bool = False
    tile_size: int = 1024
    tile_overlap: int = 128
    force: bool = False
    cpu_offload: bool = False


def _run_batch_invisible(
    ctx: click.Context,
    img_path: Path,
    out_path: Path,
    mode: str,
    options: _BatchOptions,
) -> bool:
    """Run or safely skip the invisible pass for one batch image.

    Returns ``True`` only when a detectable target could not be processed because
    the GPU dependencies are missing. The availability probe is intentionally
    evaluated once so branching cannot observe inconsistent optional-dependency
    state.
    """
    from remove_ai_watermarks.invisible_engine import is_available as invisible_available

    skip_no_signal = _should_skip_invisible_scrub(options.force, img_path)
    available = invisible_available()
    if available and not skip_no_signal:
        from remove_ai_watermarks.invisible_engine import InvisibleEngine

        # Cache the engine in ctx.obj so the batch builds it once (pipeline is a
        # single CLI value, constant across the run).
        engines = ctx.obj.setdefault("_inv_engines", {})
        if options.pipeline not in engines:
            engines[options.pipeline] = InvisibleEngine(
                model_id=options.model,
                device=None if options.device == "auto" else options.device,
                pipeline=options.pipeline,
                hf_token=options.hf_token,
                controlnet_conditioning_scale=options.controlnet_scale,
                cpu_offload=options.cpu_offload,
            )
        engines[options.pipeline].remove_watermark(
            img_path if mode == "invisible" else out_path,
            out_path,
            strength=options.strength,
            num_inference_steps=options.steps,
            guidance_scale=options.guidance_scale,
            seed=options.seed,
            humanize=options.humanize,
            unsharp=options.unsharp,
            adaptive_polish=options.adaptive_polish,
            max_resolution=options.max_resolution,
            min_resolution=options.min_resolution,
            upscaler=options.upscaler,
            tile=options.tile,
            tile_size=options.tile_size,
            tile_overlap=options.tile_overlap,
            # Detect the vendor from the pristine original (`img_path`), not the
            # visible-processed `out_path` whose C2PA is already gone.
            vendor=vendor_for_strength(img_path),
        )
        return False

    # Invisible-only mode has no preceding visible pass to create ``out_path``.
    # Preserve a complete output directory while deliberately leaving pixels intact.
    if mode == "invisible" and not out_path.exists():
        _passthrough_copy(img_path, out_path)
    return not available and not skip_no_signal


def _process_batch_image(
    ctx: click.Context,
    img_path: Path,
    out_path: Path,
    mode: str,
    options: _BatchOptions,
) -> bool:
    """Process a single image for batch mode.

    Applies the requested watermark removal steps (visible, invisible,
    metadata) to *img_path* and writes the result to *out_path*.

    Returns True if the invisible (SynthID) scrub was skipped because the GPU deps
    are missing while a signal was present -- so the batch caller can warn + exit
    non-zero, mirroring the single ``all`` command.

    Raises:
        ValueError: If the image cannot be opened.
    """
    saved_alpha: NDArray[Any] | None = None
    synthid_skipped = False

    if mode in ("visible", "all"):
        # Always read the ORIGINAL source: the visible pass is the first step, so a
        # stale out_path from a previous run must not be re-processed as if it were
        # the input. (The invisible step below reads out_path for `all` -- that chain
        # is within a single run.)
        image, alpha = image_io.read_bgr_and_alpha(img_path)
        if image is None:
            raise ValueError("Failed to read image")

        result, _ = _remove_visible_auto(
            image,
            source_path=img_path,
            backend=options.backend,
            sensitivity=options.sensitivity,
        )

        # RAISE, never SystemExit: the batch loop catches per-image exceptions, counts
        # them and exits non-zero. Discarding this flag made a read-only output directory
        # produce ZERO files and still exit 0 -- silent data loss that also contradicted
        # the documented batch contract (Tier E, 2026-07-20).
        if not image_io.write_bgr_with_alpha(out_path, result, alpha):
            raise OSError(f"failed to write output (is the destination writable?): {out_path}")
        saved_alpha = alpha

    if mode in ("invisible", "all"):
        # Skip the destructive regeneration when no invisible watermark is locally
        # detectable (would only degrade a clean image). Read the pristine `img_path`;
        # `out_path` may already be the visible-processed result. --force overrides.
        synthid_skipped = _run_batch_invisible(ctx, img_path, out_path, mode, options)

    if mode in ("metadata", "all"):
        from remove_ai_watermarks.metadata import strip_and_verify

        # Same verification the single-image command does: the fail-safe copy-through
        # would otherwise leave an AI-reading output and still exit 0, contradicting the
        # batch contract that a failed image must make the run exit non-zero.
        _, leftover = strip_and_verify(img_path if mode == "metadata" else out_path, out_path)
        if leftover:
            msg = f"AI metadata survived the strip ({', '.join(sorted(leftover))}); file could not be decoded"
            raise RuntimeError(msg)

    # In "all" mode, the invisible step (color-only OpenCV paths) drops alpha,
    # so re-attach the cached alpha when the input had transparency.
    if mode == "all" and saved_alpha is not None:
        final_bgr, _ = image_io.read_bgr_and_alpha(out_path)
        if final_bgr is not None and not image_io.write_bgr_with_alpha(out_path, final_bgr, saved_alpha):
            raise OSError(f"failed to re-attach alpha to output: {out_path}")

    return synthid_skipped


@main.command("batch")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: <dir>_clean/).",
)
@click.option(
    "--mode", type=click.Choice(["visible", "invisible", "metadata", "all"]), default="visible", help="Processing mode."
)
@_strength_option
@click.option("--steps", type=int, default=50, help="Number of denoising steps (invisible mode).")
@_visible_backend_option
@_visible_sensitivity_option
@click.option(
    "--humanize", type=float, default=0.0, help="Analog Humanizer film grain intensity (0 = off, typical: 2.0-6.0)."
)
@_pipeline_option
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "mps", "cuda", "xpu"]),
    default="auto",
    help="Inference device.",
)
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility.")
@click.option("--hf-token", type=str, default=None, help="HuggingFace API token.")
@click.option(
    "--max-resolution",
    type=int,
    default=0,
    help="Cap long side (px) before diffusion; 0 = native (best quality, like raiw.cc). Raise only on GPU/MPS OOM.",
)
@_min_resolution_option
@_unsharp_option
@_upscaler_option
@_controlnet_scale_option
@_model_option
@_guidance_scale_option
@_auto_option
@_adaptive_polish_option
@_tile_options
@_force_option
@_cpu_offload_option
@click.pass_context
def cmd_batch(
    ctx: click.Context,
    directory: Path,
    mode: str,
    output_dir: Path | None,
    strength: float | None,
    steps: int,
    pipeline: str,
    device: str,
    seed: int | None,
    hf_token: str | None,
    backend: str,
    sensitivity: str,
    humanize: float,
    unsharp: float,
    max_resolution: int,
    min_resolution: int,
    controlnet_scale: float,
    upscaler: str,
    model: str | None,
    guidance_scale: float | None,
    auto: bool,
    adaptive_polish: bool,
    tile: bool,
    tile_size: int,
    tile_overlap: int,
    force: bool,
    cpu_offload: bool,
) -> None:
    """Process all images in a directory."""
    _banner()

    if output_dir is None:
        output_dir = directory.parent / (directory.name + "_clean")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_FORMATS)

    if not images:
        console.print(f"No supported images found in {directory}")
        return

    console.print(f"  Found {len(images)} images in {directory}")
    console.print(f"  Output -> {output_dir}")
    console.print(f"  Mode: {mode}")
    if mode in ("invisible", "all"):
        _warn_if_esrgan_unavailable(upscaler)
    adaptive_polish = _resolve_auto_polish(auto, adaptive_polish)
    options = _BatchOptions(
        strength=strength,
        steps=steps,
        pipeline=pipeline,
        device=device,
        seed=seed,
        hf_token=hf_token,
        humanize=humanize,
        backend=backend,
        sensitivity=sensitivity,
        unsharp=unsharp,
        max_resolution=max_resolution,
        min_resolution=min_resolution,
        controlnet_scale=controlnet_scale,
        upscaler=upscaler,
        model=model,
        guidance_scale=guidance_scale,
        adaptive_polish=adaptive_polish,
        tile=tile,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        force=force,
        cpu_offload=cpu_offload,
    )

    processed = 0
    errors = 0
    synthid_skipped_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=len(images))

        for img_path in images:
            out_path = output_dir / img_path.name
            progress.update(task, description=f"{img_path.name}")

            try:
                if _process_batch_image(
                    ctx=ctx,
                    img_path=img_path,
                    out_path=out_path,
                    mode=mode,
                    options=options,
                ):
                    synthid_skipped_count += 1
                processed += 1

            except Exception as e:
                errors += 1
                if ctx.obj.get("verbose"):
                    console.print(f"  {img_path.name}: {e}")

            progress.advance(task)

    console.print(f"\n  {processed} processed" + (f"  {errors} errors" if errors else ""))

    if synthid_skipped_count:
        # Mirror the single `all` command: a silently retained SynthID watermark is the
        # #1 "it didn't work" report, so make the skipped scrub impossible to miss.
        console.print(
            f"\n  WARNING: the invisible (SynthID) watermark was NOT removed on "
            f"{synthid_skipped_count} image(s) -- the GPU dependencies are not installed, "
            f"so those outputs still carry the invisible watermark.\n"
            f"  Install the extra and rerun: pip install 'remove-ai-watermarks[gpu]'"
        )

    # Non-zero exit so a wrapping service detects an incomplete/failed run (batch used
    # to always exit 0, hiding both per-image errors and skipped SynthID scrubs).
    if errors or synthid_skipped_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
