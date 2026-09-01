"""Unified CLI for remove-ai-watermarks.

Provides commands for:
  - Visible watermark removal (Gemini sparkle) - works offline, fast
  - Invisible watermark removal (SynthID etc.) - requires GPU/diffusion models
  - AI metadata stripping - lightweight, no ML deps needed
  - Video identification, visible-wordmark removal, and metadata stripping
  - Oracle-certified video SynthID removal
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn

import click

from remove_ai_watermarks import __version__, image_io, watermark_registry
from remove_ai_watermarks._internal.constants import SUPPORTED_FORMATS
from remove_ai_watermarks._internal.utils import is_supported_format
from remove_ai_watermarks._internal.watermark_profiles import (
    DEFAULT_PROFILE,
    INVISIBLE_EXTRA,
    PROFILE_CHOICES,
    resolve_strength,
    strength_default_help,
    vendor_for_strength,
)
from remove_ai_watermarks.video import VIDEO_VISIBLE_MARKS
from remove_ai_watermarks.video_synthid import (
    DEFAULT_VIDEO_SYNTHID_FPS,
    DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    DEFAULT_VIDEO_SYNTHID_NOISE_STD,
    VIDEO_SYNTHID_LATENT_MULTIPLE,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray

    from remove_ai_watermarks.api import InvisibleOptions


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

    def __exit__(self, *exc: object) -> Literal[False]:
        # Literal[False], not bool: a plain `bool` tells a type checker this context
        # manager MAY suppress an exception, which makes every name bound inside a
        # `with` block conditionally bound afterwards. It never suppresses.
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
    if not is_supported_format(path):
        console.print(f"Warning: {path.suffix} may not be supported (expected: {', '.join(SUPPORTED_FORMATS)})")
    return path


def _resolved_strength_for_display(
    source: Path,
    strength: float | None,
    vendor: str | None,
    pipeline: str,
) -> float:
    """Resolve the same profile-specific strength the engine will execute.

    One call for every profile, so the printed value cannot drift from the executed
    one; the size is what qwen-zimage derives its strength from.
    """
    from PIL import Image

    with Image.open(source) as image:
        return resolve_strength(strength, vendor, pipeline, size=image.size)


# -o/--output is the most-repeated option in this module. The image commands and the
# video commands differ only in the default they describe, so there are two decorators
# rather than one -- same reason as every other shared option here: define it once so
# the help text cannot drift between commands.
_output_option = click.option(
    "-o", "--output", type=click.Path(path_type=Path), default=None, help="Output path (default: <source>_clean.<ext>)."
)

_video_output_option = click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path (default: <source>_clean with the same container).",
)


# Shared option decorator for commands that run the invisible-watermark pipeline.
# Both cmd_invisible and cmd_all expose this flag; defining it once avoids
# copy-paste drift.
_controlnet_scale_option = click.option(
    "--controlnet-scale",
    type=float,
    default=1.0,
    help="Canny ControlNet conditioning scale on the global stage "
    "(structure/text preservation strength). Higher = closer to original structure.",
)

_unsharp_option = click.option(
    "--unsharp", type=float, default=0.0, help="Unsharp-mask sharpening strength (0 = off, typical: 0.3-0.8)."
)

_adaptive_polish_option = click.option(
    "--adaptive-polish/--no-adaptive-polish",
    default=None,
    help="Restore the input's detail level after removal (capped unsharp + edge-masked grain "
    "targeting the input's sharpness, sparing text), countering the over-smoothed look. "
    "Unset follows the profile: ON for sdxl-zimage, OFF for qwen-zimage, whose "
    "upstream-matching output is left unchanged. It self-limits where there is no detail "
    "deficit (text/flat graphics). Independent of --unsharp/--humanize.",
)


# Tiled-diffusion knobs, shared by the diffusion commands (invisible/all/batch).
# Tiling avoids an explicit resolution cap for large inputs that OOM on MPS/GPU:
# it regenerates overlapping tiles at the input's native dimensions.
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
        help="Tile dimension in px for --tile. Default 1024.",
    )(f)
    return click.option(
        "--tile/--no-tile",
        default=False,
        help="Process large images in overlapping tiles instead of one forward pass. This keeps "
        "the input's native dimensions instead of applying --max-resolution, but still regenerates "
        "every tile. Engages only when the long side exceeds --tile-size. Default off.",
    )(f)


# There is deliberately no --model, --steps, --guidance-scale or --device option.
# Each profile pins a fixed model stack, a distilled per-stage schedule, CFG 1.0 and
# CUDA; every one of those knobs existed only so the library could reject it several
# layers down. A flag whose sole outcome is an error is worse than no flag at all --
# it advertises a capability that does not exist.

# The two-stage profiles are the only ones left. The former controlnet, sdxl, qwen and
# default profiles were removed rather than kept as a CPU path: none matched this
# recipe's face preservation, so offering them implied a quality the library no longer
# delivers. BOTH remaining profiles are CUDA-only.
_PIPELINE_CHOICES = list(PROFILE_CHOICES)
_PIPELINE_HELP = (
    "Pipeline profile. qwen-zimage (DEFAULT) = Qwen-Image-2512 + Lightning + Canny, "
    "followed by SAM-masked Z-Image face repair; sdxl-zimage = the same recipe and the "
    "same face stage on an SDXL global pass, which needs more denoise; chroma-zimage = "
    "the same face stage on an Apache-2.0 Chroma1 global pass with lower OpenAI and "
    "Microsoft floors and higher Google/Meta floors; auto = pick the engine from the "
    "provenance (chroma-zimage for OpenAI/Microsoft, qwen-zimage for Google/Meta/unknown). "
    "All are CUDA-ONLY -- install the qwen-zimage extra. There is no CPU or MPS profile for "
    "invisible-watermark removal."
)

# Shared --pipeline / --strength decorators so the three diffusion commands
# (invisible/all/batch) keep an identical surface and the strength help can never
# drift from the watermark_profiles constants (strength_default_help derives it).
_pipeline_option = click.option(
    "--pipeline",
    type=click.Choice(_PIPELINE_CHOICES),
    default=DEFAULT_PROFILE,
    help=_PIPELINE_HELP,
)
_strength_option = click.option(
    "--strength",
    type=float,
    default=None,
    help=f"Denoising strength (0.0-1.0). Default: {strength_default_help()}.",
)
# Explicit strength-cohort override. Auto-detection reads the C2PA issuer, so it
# covers OpenAI / Google / Microsoft; Meta Content Seal has no provenance signal
# (no C2PA; the IPTC tag is a standard code), and an unknown or stripped manifest
# also leaves the resolution-adaptive curve in charge -- this flag is the way to
# name the cohort when the user knows what the file does not say.
_vendor_option = click.option(
    "--vendor",
    type=click.Choice(["auto", "openai", "google", "microsoft", "meta"]),
    default="auto",
    help=(
        "Strength cohort for the invisible-removal default, and it implies the scrub "
        "runs even without a local signal: naming the cohort asserts the pixel "
        "watermark is present. auto: derive from C2PA provenance, else "
        "resolution-adaptive. Set explicitly when the source is known but unreadable "
        "(e.g. meta for Muse Image Content Seal, which never carries C2PA)."
    ),
)


def _explicit_vendor(vendor: str | None) -> str | None:
    """Normalize --vendor's ``auto`` default to None for the engine/API seam.

    One helper so the three diffusion commands cannot drift on the spelling."""
    return None if vendor in (None, "auto") else vendor


_seed_option = click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility. Default 0: all profiles are certified "
    "at a fixed seed, because SynthID removal near the strength floor is seed-dependent.",
)
_hf_token_option = click.option("--hf-token", type=str, default=None, help="Hugging Face API token.")
_humanize_option = click.option(
    "--humanize", type=float, default=0.0, help="Analog Humanizer film grain intensity (0 = off, typical: 2.0-6.0)."
)
_max_resolution_option = click.option(
    "--max-resolution",
    type=int,
    default=0,
    help="Cap long side (px) before diffusion; 0 = native and preserves the most detail. Raise only on GPU OOM.",
)
_force_option = click.option(
    "--force/--no-force",
    default=False,
    help=(
        "Run the diffusion scrub even when no invisible AI watermark is locally "
        "detectable. Default: skip it (regeneration only degrades a clean image; a "
        "skip never claims the image is watermark-free -- this package has no local "
        "SynthID pixel decoder)."
    ),
)
_cpu_offload_option = click.option(
    "--cpu-offload/--no-cpu-offload",
    default=False,
    help=(
        "Offload model components to CPU between CUDA calls instead of keeping the "
        "whole pipeline in VRAM, at the cost of speed. For qwen-zimage, forces the "
        "face stack to offload instead of using automatic residency."
    ),
)

_text_manifest_option = click.option(
    "--text-manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Experimental verified-text restoration manifest. Requires qwen-zimage or "
        "chroma-zimage, the text-restoration extra, native untiled geometry, and no "
        "postprocessing."
    ),
)

_fidelity_anchor_option = click.option(
    "--fidelity-anchor/--no-fidelity-anchor",
    default=False,
    help=(
        "With --text-manifest: blend 15% of the Qwen-VAE donor across the whole "
        "frame. Off by default - the global blend was measured to return "
        "detector-visible OpenAI SynthID on poster-scale manifests."
    ),
)


_visible_backend_option = click.option(
    "--backend",
    "backend",
    type=click.Choice(["auto", "cv2", "migan", "lama"]),
    default="auto",
    help="Fill backend for visible-mark removal (localize -> fill). auto: best available, "
    "LaMa > MI-GAN > cv2 (a learned backend needs the 'lama' or 'migan' extra; else cv2, "
    "with a warning). cv2: classical inpaint (no model download, smears texture). migan: MI-GAN ONNX "
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


def _visible_provenance(path: Path | None) -> frozenset[str]:
    """Vendor keys local metadata confirms, the EVIDENCE that drives ``auto``
    sensitivity. Thin wrapper over the public :func:`api.visible_provenance` (one
    implementation for the CLI and the library), with a None-path guard."""
    if path is None:
        return frozenset()
    from remove_ai_watermarks.api import visible_provenance

    return visible_provenance(path)


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

    The visible registry handles only known visual marks. Most images carry no
    registered mark and may instead have an invisible or metadata watermark.
    Returning the input
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
# error code that tells a wrapping service "the diffusion scrub was skipped because
# no invisible watermark was locally detectable", so it can surface the message
# instead of treating an unchanged image as a completed removal.
EXIT_NO_INVISIBLE_SIGNAL = 2


def _no_invisible_signal_exit(source: Path) -> NoReturn:
    """Explain why the diffusion scrub was skipped, then exit non-zero.

    The ``invisible`` command regenerates pixels to remove SynthID / open
    watermarks; that regeneration also degrades a real photo. When
    :func:`identify` finds no locally-detectable invisible AI signal, running it
    anyway would damage a clean image for nothing -- the dominant paid score-0
    cause on no-watermark uploads. So skip it, but do NOT imply the image is
    clean: Google does not publish the SynthID payload decoder, and this package
    does not ship one, so a mark can still be present after its metadata proxy
    is gone. Write no output and exit :data:`EXIT_NO_INVISIBLE_SIGNAL`;
    ``--force`` runs the scrub regardless.
    """
    console.print(
        "  No supported invisible AI watermark detected (no provenance or open\n"
        "  watermark). Skipped the diffusion scrub -- regenerating the pixels would\n"
        "  only degrade the image with nothing to remove, so no output was written.\n"
        "  This does NOT prove the image is clean: this package has no local SynthID\n"
        "  pixel decoder. If you know the image is AI-generated and want the pixels\n"
        "  regenerated regardless, re-run with --force:\n"
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
    """Remove visible and invisible AI watermarks, plus metadata provenance marks, from images and video."""
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
        console.print(f"  No known visible mark detected. Checked: {', '.join(watermark_registry.mark_keys())}.")
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
            # Reuse the detection printed above instead of re-detecting inside remove():
            # nothing has touched `image` since, and the trust level is the same one.
            result, _ = chosen.remove(image, backend=backend, provenance=relax, force=not detect, detection=detection)
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
@_output_option
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

    Finds registered marks in their expected areas and removes them by localizing
    each mark to a mask, then filling that mask with the selected ``--backend``.
    ``--mark auto`` removes every detected registry entry in one pass. Run
    ``--help`` to see the current mark keys. For arbitrary logos and objects, use
    ``erase``.
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
@_output_option
@click.option(
    "--backend",
    type=click.Choice(["cv2", "migan", "lama"]),
    default="cv2",
    help="Inpaint backend. cv2: instant, no model download. migan: light ONNX MI-GAN, ~1 GB RAM, "
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
    for marks the dedicated ``visible`` registry does not cover.
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
@_output_option
@_strength_option
@_vendor_option
@_pipeline_option
@_seed_option
@_hf_token_option
@_humanize_option
@_max_resolution_option
@_controlnet_scale_option
@_unsharp_option
@_adaptive_polish_option
@_tile_options
@_force_option
@_cpu_offload_option
@_text_manifest_option
@_fidelity_anchor_option
@click.pass_context
def cmd_invisible(
    ctx: click.Context,
    source: Path,
    output: Path | None,
    strength: float | None,
    vendor: str | None,
    pipeline: str,
    seed: int | None,
    hf_token: str | None,
    humanize: float,
    unsharp: float,
    max_resolution: int,
    controlnet_scale: float,
    adaptive_polish: bool | None,
    tile: bool,
    tile_size: int,
    tile_overlap: int,
    force: bool,
    cpu_offload: bool,
    text_manifest: Path | None,
    fidelity_anchor: bool,
) -> None:
    """Attempt to disrupt invisible AI watermarks through pixel regeneration.

    Regenerates the pixels with the two-stage diffusion profile. CUDA-only:
    pip install 'remove-ai-watermarks[qwen-zimage]'
    """
    from remove_ai_watermarks.invisible_engine import is_available as invisible_available

    if not invisible_available():
        console.print(
            "Error: the invisible-removal dependencies are not installed.\n"
            f"  Install them with: pip install {INVISIBLE_EXTRA}"
        )
        raise SystemExit(1)

    from remove_ai_watermarks.invisible_engine import InvisibleEngine

    source = _validate_image(source)
    if output is None:
        output = source.with_stem(source.stem + "_clean")

    # An explicit --vendor wins over detection (see the option help) and implies the
    # scrub runs: naming the cohort asserts the pixel watermark is present, so the
    # no-signal gate must not skip it. Resolved BEFORE the gate for the same reason.
    resolved_vendor = _explicit_vendor(vendor)

    # Gate BEFORE building the engine: skip the destructive regeneration when no
    # invisible AI watermark is locally detectable (it would only degrade a clean
    # image -- dominant paid score-0 cause), so the common skip path pays nothing for
    # engine construction. A skip never claims the image is clean; --force and an
    # explicit --vendor override.
    if _should_skip_invisible_scrub(force or resolved_vendor is not None, source):
        _no_invisible_signal_exit(source)

    def progress_cb(msg: str) -> None:
        console.print(f"  {msg}")

    engine = InvisibleEngine(
        pipeline=pipeline,
        hf_token=hf_token,
        progress_callback=progress_cb,
        controlnet_conditioning_scale=controlnet_scale,
        cpu_offload=cpu_offload,
    )

    # Detect the SynthID vendor from the ORIGINAL (before processing strips C2PA) so the
    # displayed and executed strength agree on the vendor-adaptive default. An explicit
    # --vendor override wins over detection: it names a cohort the file cannot prove
    # (Meta Content Seal never carries C2PA; a stripped manifest proves nothing).
    detected_vendor = vendor_for_strength(source) if resolved_vendor is None else None
    vendor_label = resolved_vendor or detected_vendor
    vendor_note = " (override)" if resolved_vendor else ""
    console.print(f"  Input:    {source.name}")
    console.print(f"  Pipeline: {pipeline}")
    console.print(
        f"  Strength: {_resolved_strength_for_display(source, strength, vendor_label, pipeline)}"
        + (f"  [vendor: {vendor_label}{vendor_note}]" if vendor_label else "")
    )

    t0 = time.monotonic()
    try:
        result_path = engine.remove_watermark(
            image_path=source,
            output_path=output,
            strength=strength,
            seed=seed,
            humanize=humanize,
            unsharp=unsharp,
            adaptive_polish=adaptive_polish,
            max_resolution=max_resolution,
            vendor=vendor_label,
            tile=tile,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            text_manifest=text_manifest,
            fidelity_anchor=fidelity_anchor,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        console.print(f"  Error: {exc}")
        raise SystemExit(1) from exc
    elapsed = time.monotonic() - t0

    size_kb = result_path.stat().st_size / 1024
    console.print(f"\n  Saved: {result_path}  ({size_kb:.0f} KB, {elapsed:.1f}s)")


# ── Metadata operations ──
def _print_metadata_not_a_clean_verdict() -> None:
    """Repeat the identify empty-scan limit on metadata check and strip success.

    ``metadata --check`` and ``metadata --remove`` answer a narrower question than
    ``identify``: they report embedded AI metadata only. A quiet result used to
    stop at "No AI metadata found" / "AI metadata stripped", which readers treat
    as a clean-image verdict. The pixel channel is unchanged, and this project has
    no local SynthID decoder, so the command must say so in the same words
    ``identify`` already uses.
    """
    console.print(
        "  This is not the same as 'clean': a pixel watermark such as SynthID cannot be\n"
        "  detected here once its metadata proxy is absent."
    )


def _print_metadata_report(source: Path, has_ai: bool, metadata: dict[str, str]) -> None:
    """Render one metadata inspection result for the generic and video commands."""
    if not has_ai:
        console.print(f"  No AI metadata found in {source.name}")
        _print_metadata_not_a_clean_verdict()
        return

    console.print(f"  Warning: AI metadata detected in {source.name}:")
    if synthid := metadata.get("synthid_watermark"):
        console.print(f"  Warning: SynthID watermark {synthid}")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    for key, value in metadata.items():
        table.add_row(key, str(value)[:80])
    console.print(table)


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
    from WebM/MKV/MKA/AVI/FLV/MP3/WAV/FLAC/OGG/OGA/Opus/AAC. The coded image,
    audio, and video data are left untouched.
    """
    from remove_ai_watermarks.metadata import get_ai_metadata, has_ai_metadata, strip_and_verify

    # No _validate_image() here: unlike the image-only commands, metadata also
    # accepts video/audio containers, so the image-format warning would misfire.
    # click's `exists=True` on the argument already enforces the file exists.
    _banner()

    if check or (not remove):
        has_ai = has_ai_metadata(source)
        metadata = get_ai_metadata(source) if has_ai else {}
        _print_metadata_report(source, has_ai, metadata)

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
    _print_metadata_not_a_clean_verdict()


# ── Video pipeline ──
def _video_visible_options(f: Any) -> Any:
    """Apply the shared visible-video detector and fill options."""
    f = click.option(
        "--temporal-consistency/--no-temporal-consistency",
        default=True,
        help="Motion-align adjacent accepted fills to reduce frame-to-frame flicker.",
    )(f)
    f = click.option(
        "--backend",
        type=click.Choice(["auto", "cv2", "migan", "lama"]),
        default="cv2",
        help="Per-frame visible-fill backend.",
    )(f)
    return click.option(
        "--mark",
        type=click.Choice(["auto", *VIDEO_VISIBLE_MARKS]),
        default="auto",
        help="Visible AI mark to remove. Auto scans every supported provider in one decode pass.",
    )(f)


def _video_invisible_options(f: Any) -> Any:
    """Apply the shared invisible-video removal options."""
    f = click.option(
        "--device",
        type=click.Choice(["auto", "cuda", "mps", "cpu"]),
        default="auto",
        show_default=True,
        help="VAE inference device.",
    )(f)
    f = click.option("--seed", type=int, default=0, show_default=True)(f)
    f = click.option("--batch-size", type=click.IntRange(min=1), default=4, show_default=True)(f)
    f = click.option(
        "--fps",
        type=click.FloatRange(min=1.0),
        default=DEFAULT_VIDEO_SYNTHID_FPS,
        show_default=True,
        help="Output frame rate, capped at the source frame rate.",
    )(f)
    f = click.option(
        "--long-side",
        type=click.IntRange(min=VIDEO_SYNTHID_LATENT_MULTIPLE),
        default=DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
        show_default=True,
        help="Regenerated video long side in pixels.",
    )(f)
    return click.option(
        "--noise-std",
        type=click.FloatRange(min=0.0, max=1.0),
        default=DEFAULT_VIDEO_SYNTHID_NOISE_STD,
        show_default=True,
        help="Shared latent-noise strength. Higher values change more detail.",
    )(f)


@main.group("video")
def cmd_video() -> None:
    """Process AI watermarks in video files."""


@cmd_video.command("identify")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--no-visible", is_flag=True, help="Skip visible-mark detection; inspect metadata only.")
@click.option("--json", "as_json", is_flag=True, help="Emit the report as JSON.")
def cmd_video_identify(source: Path, no_visible: bool, as_json: bool) -> None:
    """Identify supported provenance and visible AI marks in video."""
    from dataclasses import asdict

    from remove_ai_watermarks.video import identify_video

    try:
        report = identify_video(source, check_visible=not no_visible)
    except (OSError, RuntimeError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    if as_json:
        click.echo(json.dumps(asdict(report), default=str, indent=2))
        return

    _banner()
    verdict = "AI-generated" if report.is_ai_generated else "unknown"
    console.print(f"  Verdict: {verdict}  (confidence: {report.confidence})")
    console.print(f"  Platform: {report.platform or 'undetermined'}")
    if report.visible_mark is not None:
        console.print(
            f"  Visible mark: {report.visible_mark} "
            f"({report.visible_detected_frames}/{report.total_frames} stable frames)"
        )
    else:
        console.print("  Visible mark: none found" if not no_visible else "  Visible mark: not checked")
    if report.metadata_markers:
        console.print(f"  AI metadata markers: {', '.join(sorted(report.metadata_markers))}")
    else:
        console.print("  AI metadata markers: none found")
    if report.caveats:
        console.print("  Caveats:")
        for caveat in report.caveats:
            console.print(f"  - {caveat}")


@cmd_video.command("metadata")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--check", is_flag=True, help="Check for AI metadata (don't modify).")
@click.option("--remove", is_flag=True, help="Remove AI metadata.")
@_video_output_option
@click.option("--keep-standard/--remove-all", default=True, help="Keep standard metadata.")
def cmd_video_metadata(
    source: Path,
    check: bool,
    remove: bool,
    output: Path | None,
    keep_standard: bool,
) -> None:
    """Check or remove AI metadata without transcoding video streams."""
    from remove_ai_watermarks.video import inspect_video_metadata, remove_video_metadata

    _banner()
    try:
        report = inspect_video_metadata(source)
    except (OSError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    if check or not remove:
        _print_metadata_report(source, report.has_ai_metadata, report.markers)

        if not remove:
            return

    try:
        result = remove_video_metadata(source, output, keep_standard=keep_standard)
    except (OSError, RuntimeError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    if result.remaining:
        console.print(f"  FAILED: {len(result.remaining)} AI metadata marker(s) survived in {result.output}")
        console.print(f"    still present: {', '.join(sorted(result.remaining))}")
        raise SystemExit(1)
    console.print(f"  AI metadata stripped -> {result.output}")
    _print_metadata_not_a_clean_verdict()


@cmd_video.command("invisible")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_video_output_option
@_video_invisible_options
def cmd_video_invisible(
    source: Path,
    output: Path | None,
    noise_std: float,
    long_side: int,
    fps: float,
    batch_size: int,
    seed: int,
    device: str,
) -> None:
    """Remove video SynthID with the oracle-certified VAE profile."""
    from remove_ai_watermarks.video import remove_video_invisible

    _banner()
    console.print(f"  Regenerating {source.name} with temporally shared VAE noise...")
    try:
        result = remove_video_invisible(
            source,
            output,
            noise_std=noise_std,
            long_side=long_side,
            fps=fps,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
    except (OSError, RuntimeError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    if result.remaining_metadata:
        console.print(f"  FAILED: {len(result.remaining_metadata)} AI metadata marker(s) survived in {result.output}")
        raise SystemExit(1)
    console.print(
        f"  SynthID removal complete: {result.width}x{result.height}, "
        f"{result.total_frames} frames at {result.fps:.4g} fps -> {result.output}"
    )


@cmd_video.command("visible")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_video_output_option
@_video_visible_options
@click.option("--strip-metadata/--keep-metadata", default=True, help="Strip AI metadata from the transcoded output.")
def cmd_video_visible(
    source: Path,
    output: Path | None,
    mark: str,
    backend: str,
    temporal_consistency: bool,
    strip_metadata: bool,
) -> None:
    """Remove a temporally stable visible AI wordmark from video."""
    from remove_ai_watermarks.video import remove_video_visible

    _banner()
    console.print(f"  Scanning {source.name} for a temporally stable {mark} mark...")
    try:
        result = remove_video_visible(
            source,
            output,
            mark=mark,
            backend=backend,
            strip_metadata=strip_metadata,
            temporal_consistency=temporal_consistency,
        )
    except (OSError, RuntimeError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    if result.output is None:
        console.print(f"  No stable {mark} watermark detected; no output written")
        raise SystemExit(EXIT_NO_VISIBLE_MARK)
    if result.remaining_metadata:
        console.print(f"  FAILED: {len(result.remaining_metadata)} AI metadata marker(s) survived in {result.output}")
        raise SystemExit(1)
    console.print(
        f"  Removed {result.mark} watermark from "
        f"{result.removed_frames}/{result.total_frames} frames -> {result.output}"
    )


@cmd_video.command("all")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_video_output_option
@_video_visible_options
@click.option(
    "--invisible/--no-invisible",
    default=False,
    help="Opt into oracle-certified lossy video SynthID removal.",
)
@_video_invisible_options
def cmd_video_all(
    source: Path,
    output: Path | None,
    mark: str,
    backend: str,
    temporal_consistency: bool,
    invisible: bool,
    noise_std: float,
    long_side: int,
    fps: float,
    batch_size: int,
    seed: int,
    device: str,
) -> None:
    """Remove stable visible marks and AI metadata from video."""
    from remove_ai_watermarks.video import remove_video_all

    _banner()
    stages = "visible marks + SynthID + verified AI metadata" if invisible else "visible marks + verified AI metadata"
    console.print(f"  Cleaning {source.name}: {stages}...")
    try:
        result = remove_video_all(
            source,
            output,
            mark=mark,
            backend=backend,
            temporal_consistency=temporal_consistency,
            include_invisible=invisible,
            noise_std=noise_std,
            long_side=long_side,
            fps=fps,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
    except (OSError, RuntimeError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    if result.remaining_metadata:
        console.print(f"  FAILED: {len(result.remaining_metadata)} AI metadata marker(s) survived in {result.output}")
        raise SystemExit(1)
    if result.visible_mark is None:
        detail = "" if result.invisible_removed else "; pixels preserved"
        console.print(f"  Visible mark: none found{detail}")
    else:
        console.print(
            f"  Visible mark: removed {result.visible_mark} from "
            f"{result.visible_removed_frames}/{result.total_frames} frames"
        )
    console.print(f"  AI metadata: stripped -> {result.output}")
    if result.invisible_removed:
        console.print("  SynthID: removed with the oracle-certified VAE profile")


@cmd_video.command("batch")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: <directory>_clean).",
)
@click.option(
    "--mode",
    type=click.Choice(["all", "visible", "metadata"]),
    default="all",
    show_default=True,
    help="Video processing mode.",
)
@_video_visible_options
@click.option(
    "--invisible/--no-invisible",
    default=False,
    help="Opt into oracle-certified lossy SynthID removal in all mode.",
)
@_video_invisible_options
def cmd_video_batch(
    directory: Path,
    output_dir: Path | None,
    mode: str,
    mark: str,
    backend: str,
    temporal_consistency: bool,
    invisible: bool,
    noise_std: float,
    long_side: int,
    fps: float,
    batch_size: int,
    seed: int,
    device: str,
) -> None:
    """Process every supported video in a directory."""
    from remove_ai_watermarks.video import remove_video_batch

    _banner()
    console.print(f"  Processing video directory {directory} in {mode} mode...")
    try:
        result = remove_video_batch(
            directory,
            output_dir,
            mode=mode,  # type: ignore[arg-type]
            mark=mark,
            backend=backend,
            temporal_consistency=temporal_consistency,
            include_invisible=invisible,
            noise_std=noise_std,
            long_side=long_side,
            fps=fps,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
    except (OSError, RuntimeError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    for item in result.items:
        if item.error is not None:
            console.print(f"  FAILED {item.source.name}: {item.error}")
        elif item.changed:
            detail = f" ({item.visible_mark})" if item.visible_mark is not None else ""
            console.print(f"  Processed {item.source.name}{detail} -> {item.output}")
        elif item.mode == "visible":
            console.print(f"  Copied {item.source.name} byte-for-byte -> {item.output}")
        else:
            console.print(f"  Completed {item.source.name}; no supported signal found -> {item.output}")
    console.print(
        f"  Batch complete: {result.processed} processed, {result.failed} failed -> {result.output_directory}"
    )
    if result.invisible_removed:
        console.print(f"  SynthID: removed from {result.invisible_removed} file(s)")
    if result.failed:
        raise SystemExit(1)


# ── Official OpenAI SynthID verification ──
@main.command("verify-openai-synthid")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--acknowledge-upload",
    is_flag=True,
    help="Confirm upload of a pixel-identical, AI-metadata-stripped copy to OpenAI.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the verifier result as JSON.")
def cmd_verify_openai_synthid(source: Path, acknowledge_upload: bool, as_json: bool) -> None:
    """Use OpenAI's official verifier on pixels, independently of C2PA.

    The command strips AI provenance metadata from a temporary copy, proves the
    decoded pixels are unchanged, and uploads that copy to OpenAI. It reads only
    the SynthID result. The source file is never modified.
    """
    if not acknowledge_upload:
        raise click.ClickException(
            "this command uploads a temporary pixel-identical copy to OpenAI; pass --acknowledge-upload to continue"
        )
    from remove_ai_watermarks.openai_provenance import verify_openai_synthid

    source = _validate_image(source)
    try:
        result = verify_openai_synthid(source, acknowledge_upload=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
        return

    _banner()
    console.print(f"\n  OpenAI SynthID pixel watermark: {result.status}")
    console.print("  Detector: official OpenAI Content Provenance API")
    if result.model is not None:
        console.print(f"  Model: {result.model}")
    if result.generated_at is not None:
        console.print(f"  Generated at: {result.generated_at}")
    console.print(
        "  Input: AI provenance metadata was stripped and decoded pixels were preserved.\n"
        "  Scope: supported OpenAI SynthID only. A not_detected result is not proof\n"
        "  that the image is human-created or contains no other watermark."
    )


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

    Aggregates supported C2PA, IPTC, EXIF, XMP, generator, visible-mark, and
    optional invisible-watermark signals into one provenance verdict. Absence of
    signals is reported as "unknown", never as "clean" because stripped metadata
    leaves no local proof.
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
    if report.c2pa_validation is not None:
        validation = report.c2pa_validation
        console.print(
            "  C2PA validation: "
            f"integrity={validation['integrity']}, signature={validation['signature']}, "
            f"signer trust={validation['signer_trust']}, signer validity={validation['signer_validity']}"
        )

    if report.is_ai_generated is None:
        console.print(
            "  No locally-readable AI signal found. This is not the same as 'clean': "
            "metadata is often stripped by re-encoding, screenshots, or upload, and this "
            "package has no local SynthID pixel decoder. See caveats below."
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


# ── Metadata-free photo classification (opt-in, never a tail of identify) ──
@main.command("classify")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit the classification as JSON.")
def cmd_classify(source: Path, as_json: bool) -> None:
    """Classify a photograph as AI or camera-like from pixels.

    Runs the frozen photo detector and, only on a DEFINITELY-AI result, the
    provider head. This is not provenance. identify never starts this command.
    Install: pip install 'remove-ai-watermarks[classify]'
    """
    from remove_ai_watermarks import classify as classify_mod

    if not classify_mod.is_available():
        console.print(
            "Error: the pixel-classification dependencies are not installed.\n"
            f"  Install them with: pip install {classify_mod.CLASSIFY_EXTRA}"
        )
        raise SystemExit(1)

    source = _validate_image(source)
    try:
        result = classify_mod.classify_pixels(source)
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
        return

    _banner()
    console.print("\n  Pixel classification (not a provenance verdict)")
    console.print(f"  Label: {result.label}")
    console.print(f"  Domain: {result.domain}")
    console.print(f"  Detector: {result.detector}")
    console.print(f"  Provider: {result.provider or 'none'}")
    console.print("  identify is unchanged. This command does not run cleanup and does not prove a file is clean.")


# ── Combined "all" mode ──
@main.command("all")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_output_option
@_visible_backend_option
@_visible_sensitivity_option
@_strength_option
@_vendor_option
@_pipeline_option
@_seed_option
@_hf_token_option
@_humanize_option
@_max_resolution_option
@_controlnet_scale_option
@_unsharp_option
@_adaptive_polish_option
@_tile_options
@_force_option
@_cpu_offload_option
@_text_manifest_option
@_fidelity_anchor_option
@click.pass_context
def cmd_all(
    ctx: click.Context,
    source: Path,
    output: Path | None,
    backend: str,
    sensitivity: str,
    strength: float | None,
    vendor: str | None,
    pipeline: str,
    seed: int | None,
    hf_token: str | None,
    humanize: float,
    unsharp: float,
    max_resolution: int,
    controlnet_scale: float,
    adaptive_polish: bool | None,
    tile: bool,
    tile_size: int,
    tile_overlap: int,
    force: bool,
    cpu_offload: bool,
    text_manifest: Path | None,
    fidelity_anchor: bool,
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

    if output is None:
        output = source.with_stem(source.stem + "_clean")

    t0 = time.monotonic()

    from remove_ai_watermarks.api import InvisibleOptions, MetadataStripIncomplete, remove_all

    stage_labels = {
        "visible": "\n  1) Visible watermark removal",
        "invisible": "\n  2) Invisible watermark removal",
        "metadata": "\n  3) AI metadata stripping",
    }
    # The library reports WHAT happened as a (stage, detail) pair of stable tokens; the
    # console wording is the CLI's business. These two skips in particular carry guidance
    # a library caller does not need but a user very much does.
    stage_text = {
        ("invisible", "no-signal"): (
            "Skipped (no invisible AI watermark detected; pixels left intact).\n"
            "    Not a clean-image guarantee: this package has no local SynthID pixel\n"
            "    decoder. Re-run with --force to scrub regardless."
        ),
        ("invisible", "unavailable"): (
            f"Warning: Skipped - GPU dependencies not installed.\n    Install them with: pip install {INVISIBLE_EXTRA}"
        ),
        ("invisible", "removed"): "Invisible watermark removed",
        ("metadata", "stripped"): "AI metadata stripped",
    }
    seen: set[str] = set()

    def progress(stage: str, detail: str) -> None:
        if stage in stage_labels and stage not in seen:
            seen.add(stage)
            console.print(stage_labels[stage])
        if (text := stage_text.get((stage, detail))) is not None:
            console.print(f"    {text}")
        elif stage == "visible":
            console.print(
                f"    Visible watermark removed ({detail})" if detail else "    Skipped (no visible watermark detected)"
            )
        elif detail.startswith("strength="):
            console.print(f"    Strength: {detail.removeprefix('strength=')}")
        else:
            console.print(f"    {detail}")

    try:
        outcome = remove_all(
            source,
            output,
            backend=backend,  # type: ignore[arg-type]
            sensitivity=_parse_sensitivity(sensitivity),
            invisible=InvisibleOptions(
                strength=strength,
                vendor=_explicit_vendor(vendor),
                pipeline=pipeline,
                seed=seed,
                hf_token=hf_token,
                humanize=humanize,
                unsharp=unsharp,
                adaptive_polish=adaptive_polish,
                max_resolution=max_resolution,
                controlnet_conditioning_scale=controlnet_scale,
                cpu_offload=cpu_offload,
                tile=tile,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                text_manifest=text_manifest,
                fidelity_anchor=fidelity_anchor,
            ),
            force=force,
            progress=progress,
        )
    except MetadataStripIncomplete as e:
        console.print(f"    Error: metadata stripping was incomplete; {', '.join(sorted(e.surviving))} survived")
        raise SystemExit(1) from e
    except ValueError as e:
        console.print(f"Error: {e}")
        raise SystemExit(1) from e
    except RuntimeError as e:  # a selected migan/lama backend whose extra is absent
        console.print(f"  Error: {e}")
        raise SystemExit(1) from e
    except OSError as e:
        console.print(f"  Error: {e}")
        raise SystemExit(1) from e

    # ── Done ──
    elapsed = time.monotonic() - t0
    size_kb = output.stat().st_size / 1024
    console.print(f"\n  Done: {output}  ({size_kb:.0f} KB, {elapsed:.1f}s total)")

    # A skipped invisible step is the single most common "it didn't work" report:
    # the output looks processed but still carries the SynthID watermark. Make that
    # impossible to miss -- a prominent banner plus a non-zero exit so scripts and
    # batch callers can detect the incomplete run instead of trusting the file.
    if outcome.invisible == "unavailable":
        console.print(
            "\n  =====================================================================\n"
            "  WARNING: the invisible (SynthID) watermark was NOT removed.\n"
            "  Step 2 was skipped because the GPU dependencies are not installed,\n"
            "  so this output still carries the invisible watermark -- only the\n"
            "  visible mark and metadata were stripped.\n"
            "\n"
            "  Install the extra and rerun to remove it:\n"
            f"    pip install {INVISIBLE_EXTRA}\n"
            "  ====================================================================="
        )
        raise SystemExit(1)


# ── Batch command ──
def _batch_engine(mode: str, options: InvisibleOptions) -> object | None:
    """Build the ONE invisible engine this batch reuses, or None.

    Called once per run, not per image: ``--pipeline`` is a single CLI value, constant
    across the batch, so building the model here and threading it down is what keeps the
    diffusion stack from reloading for every file. Modes that never scrub get None, so
    nothing is loaded at all.
    """
    if mode not in ("all", "invisible"):
        return None
    from remove_ai_watermarks.invisible_engine import InvisibleEngine, is_available

    if not is_available():
        return None
    return InvisibleEngine(
        pipeline=options.pipeline,
        hf_token=options.hf_token,
        controlnet_conditioning_scale=options.controlnet_conditioning_scale,
        cpu_offload=options.cpu_offload,
    )


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
@_visible_backend_option
@_visible_sensitivity_option
@_humanize_option
@_vendor_option
@_pipeline_option
@_seed_option
@_hf_token_option
@_max_resolution_option
@_unsharp_option
@_controlnet_scale_option
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
    vendor: str | None,
    pipeline: str,
    seed: int | None,
    hf_token: str | None,
    backend: str,
    sensitivity: str,
    humanize: float,
    unsharp: float,
    max_resolution: int,
    controlnet_scale: float,
    adaptive_polish: bool | None,
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

    images = sorted(p for p in directory.iterdir() if is_supported_format(p))

    if not images:
        console.print(f"No supported images found in {directory}")
        return

    console.print(f"  Found {len(images)} images in {directory}")
    console.print(f"  Output -> {output_dir}")
    console.print(f"  Mode: {mode}")
    from remove_ai_watermarks.api import InvisibleOptions
    from remove_ai_watermarks.api import remove_batch as api_remove_batch

    invisible_options = InvisibleOptions(
        strength=strength,
        vendor=_explicit_vendor(vendor),
        pipeline=pipeline,
        seed=seed,
        hf_token=hf_token,
        humanize=humanize,
        unsharp=unsharp,
        adaptive_polish=adaptive_polish,
        max_resolution=max_resolution,
        controlnet_conditioning_scale=controlnet_scale,
        cpu_offload=cpu_offload,
        tile=tile,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=len(images))
        done: set[str] = set()

        def on_progress(img: Path, stage: str, detail: str) -> None:
            # `remove_batch` emits exactly one terminal stage per image in EVERY mode,
            # so the bar advances on that and never on a mode-specific line.
            progress.update(task, description=img.name)
            if stage in ("done", "failed") and img.name not in done:
                done.add(img.name)
                progress.advance(task)
            if ctx.obj.get("verbose"):
                console.print(f"  {img.name}: {stage}{f' {detail}' if detail else ''}")

        summary = api_remove_batch(
            directory,
            output_dir,
            mode=mode,  # type: ignore[arg-type]
            backend=backend,  # type: ignore[arg-type]
            sensitivity=_parse_sensitivity(sensitivity),
            invisible=invisible_options,
            force=force,
            engine=_batch_engine(mode, invisible_options),
            progress=on_progress,
        )
        progress.update(task, completed=len(images))

    processed, errors = summary.processed, summary.failed
    synthid_skipped_count = len(summary.invisible_unavailable)

    if errors and ctx.obj.get("verbose"):
        for failed_path, message in summary.errors:
            console.print(f"  {failed_path.name}: {message}")

    console.print(f"\n  {processed} processed" + (f"  {errors} errors" if errors else ""))

    if synthid_skipped_count:
        # Mirror the single `all` command: a silently retained SynthID watermark is the
        # #1 "it didn't work" report, so make the skipped scrub impossible to miss.
        console.print(
            f"\n  WARNING: the invisible (SynthID) watermark was NOT removed on "
            f"{synthid_skipped_count} image(s) -- the GPU dependencies are not installed, "
            f"so those outputs still carry the invisible watermark.\n"
            f"  Install the extra and rerun: pip install {INVISIBLE_EXTRA}"
        )

    # Non-zero exit so a wrapping service detects an incomplete/failed run (batch used
    # to always exit 0, hiding both per-image errors and skipped SynthID scrubs).
    if errors or synthid_skipped_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
