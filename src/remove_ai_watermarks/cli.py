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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn

import click

from remove_ai_watermarks import __version__, watermark_registry
from remove_ai_watermarks.noai.constants import SUPPORTED_FORMATS
from remove_ai_watermarks.noai.watermark_profiles import (
    resolve_strength,
    strength_default_help,
    vendor_for_strength,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray


# --- plain-text output layer (replaces rich: no colors, no markup, no boxes) ---


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


_ALPHA_FORMATS = {".png", ".webp"}

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


_visible_method_option = click.option(
    "--method",
    "removal_method",
    type=click.Choice(["auto", "reverse-alpha", "inpaint"]),
    default="auto",
    help="Visible-mark removal method. auto: reverse-alpha for capture marks (exact "
    "pixels, lighter), inpaint for the capture-less pill. reverse-alpha recovers "
    "pixels from a captured alpha map; inpaint erases the footprint (MI-GAN with the "
    "'migan' extra, else cv2).",
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


def _aigc_metadata_present(path: Path) -> bool:
    """True when the file carries a China-AIGC (TC260) metadata label. Used to gate
    the weak-detector 'AI生成' pill: metadata confirms Jimeng-class provenance. NB
    this is only ONE of two confirmations -- ``remove_auto_marks`` also accepts the
    bottom-right wordmark, so a metadata-STRIPPED upload can still be handled."""
    with contextlib.suppress(Exception):
        from remove_ai_watermarks import metadata

        return bool(metadata.aigc_label(path))
    return False


def _remove_visible_auto(
    image: NDArray[Any],
    *,
    source_path: Path | None = None,
    removal_method: str = "auto",
    inpaint: bool = True,
    inpaint_method: str = "ns",
    inpaint_strength: float = 0.85,
) -> tuple[NDArray[Any], str | None]:
    """Remove the strongest auto-detected visible mark via the registry.

    Routes the ``all``/``batch`` visible step through the same registry path the
    standalone ``visible`` command uses, so EVERY registered mark is handled (the
    Gemini sparkle AND the Doubao/Jimeng/Samsung text marks), not just the sparkle.
    Returns ``(result, label-or-None)``; when no ``in_auto`` mark fires the image is
    returned unchanged with ``None``. ``removal_method`` selects reverse-alpha vs the
    inpaint fallback (see ``KnownMark.remove``); ``inpaint*`` tune the Gemini
    edge-residual cleanup only (the text engines ignore them).
    """
    from remove_ai_watermarks import watermark_registry

    rmethod: watermark_registry.RemovalMethod = removal_method  # type: ignore[assignment]
    method: Literal["telea", "ns"] = "ns" if inpaint_method == "ns" else "telea"
    pill_md = _aigc_metadata_present(source_path) if source_path is not None else False
    result, removed = watermark_registry.remove_auto_marks(
        image,
        pill_metadata=pill_md,
        method=rmethod,
        inpaint_method=method,
        inpaint=inpaint,
        inpaint_strength=inpaint_strength,
    )
    if not removed:
        return image, None
    return result, ", ".join(removed)


# Exit code for the standalone ``visible`` command when no visible mark was
# removed -- distinct from success (0) and a hard error (1) so a wrapping
# service can tell "nothing to do here" apart and surface guidance instead of
# re-serving the unchanged input as a finished result.
EXIT_NO_VISIBLE_MARK = 2


def _no_visible_mark_exit(source: Path) -> NoReturn:
    """Explain why no visible watermark was removed, then exit non-zero.

    The visible registry handles only known visual marks (the Gemini sparkle and
    the Doubao/Jimeng/Samsung text strips). Most real uploads carry no such mark
    -- frequently an invisible/metadata watermark instead (e.g. an OpenAI or
    Gemini image whose only signal is C2PA + SynthID). Returning the input
    unchanged with exit 0 reads as success to a caller and re-serves the
    watermarked image -- the recurring "it didn't work" report. Instead, run a
    cheap metadata-only :func:`identify`, tell the user what the image actually
    carries and which command removes it, and exit
    :data:`EXIT_NO_VISIBLE_MARK`.
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


def _read_bgr_and_alpha(path: Path) -> tuple[NDArray[Any] | None, NDArray[Any] | None]:
    """Read an image preserving its alpha channel separately.

    Returns ``(bgr, alpha)`` where ``alpha`` is a single-channel ndarray when the
    source has transparency, else ``None``. Greyscale inputs are promoted to BGR.
    Returns ``(None, None)`` if the image cannot be decoded.
    """
    import cv2

    from remove_ai_watermarks import image_io

    image = image_io.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), None
    if image.shape[2] == 4:
        return image[:, :, :3].copy(), image[:, :, 3].copy()
    return image, None


def _write_bgr_with_alpha(
    path: Path,
    bgr: NDArray[Any],
    alpha: NDArray[Any] | None,
) -> None:
    """Write BGR (with optional alpha) to ``path``.

    When ``alpha`` is provided and the output extension supports it, the original
    alpha plane is rejoined unchanged. The watermark region is NOT made
    transparent: reverse-alpha (and inpaint) recover real pixels there, so
    zeroing alpha would punch a transparent hole that renders as a white box on
    any non-transparent viewer (issue #30). Preserving the input alpha keeps
    genuinely transparent backgrounds intact without inventing new holes.
    """
    import numpy as np

    from remove_ai_watermarks import image_io

    if alpha is None or path.suffix.lower() not in _ALPHA_FORMATS:
        image_io.imwrite(path, bgr)
        return

    bgra = np.dstack([bgr, alpha])
    image_io.imwrite(path, bgra)


# -- Main group -------------------------------------------------------


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


# -- Visible (Gemini) watermark removal -------------------------------


@main.command("visible")
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--output", type=click.Path(path_type=Path), default=None, help="Output path (default: <source>_clean.<ext>)."
)
@click.option("--inpaint/--no-inpaint", default=True, help="Apply inpainting cleanup after removal.")
@click.option(
    "--inpaint-method", type=click.Choice(["ns", "telea", "gaussian"]), default="ns", help="Inpainting method."
)
@click.option("--inpaint-strength", type=float, default=0.85, help="Inpainting blend strength (0.0-1.0).")
@click.option("--detect/--no-detect", default=True, help="Detect watermark before removal.")
@click.option(
    "--mark",
    type=click.Choice(["auto", *watermark_registry.mark_keys()]),
    default="auto",
    help="Which known visible mark to target (auto picks the strongest detected). "
    "Removal method is chosen by --method (default auto).",
)
@_visible_method_option
@click.option("--strip-metadata/--keep-metadata", default=True, help="Strip AI metadata from output.")
@click.pass_context
def cmd_visible(
    ctx: click.Context,
    source: Path,
    output: Path | None,
    inpaint: bool,
    inpaint_method: Literal["ns", "telea", "gaussian"],
    inpaint_strength: float,
    detect: bool,
    mark: str,
    removal_method: str,
    strip_metadata: bool,
) -> None:
    """Remove a known visible AI watermark from an image.

    Finds a known mark in its usual place (Gemini sparkle / Doubao text) via the
    watermark registry and removes it. Default ``--method auto`` recovers the true
    pixels by exact reverse-alpha for the capture marks, and inpaints only the
    capture-less "AI生成" pill (MI-GAN with the ``migan`` extra, else cv2).
    ``--mark auto`` picks the strongest detected mark. For arbitrary logos/objects,
    use ``erase``.
    """
    from remove_ai_watermarks import watermark_registry as registry

    _banner()
    source = _validate_image(source)

    if output is None:
        output = source.with_stem(source.stem + "_clean")

    # Load image (preserving any alpha channel separately)
    image, alpha = _read_bgr_and_alpha(source)
    if image is None:
        console.print(f"Error: Failed to read image: {source}")
        raise SystemExit(1)

    h, w = image.shape[:2]
    console.print(f"  Input:  {source.name}  ({w}x{h})")

    method: Literal["telea", "ns"] = "ns" if inpaint_method == "ns" else "telea"

    # ``auto`` removes EVERY detected in_auto mark in one pass (a Jimeng-basic image
    # carries the top-left pill AND the bottom-right wordmark); an explicit
    # ``--mark <key>`` targets that one (the user asserts its presence).
    if mark == "auto" and detect:
        t0 = time.monotonic()
        with console.status("Detecting & removing visible marks..."):
            result, removed = registry.remove_auto_marks(
                image,
                pill_metadata=_aigc_metadata_present(source),
                method=removal_method,  # type: ignore[arg-type]
                inpaint_method=method,
                inpaint=inpaint,
                inpaint_strength=inpaint_strength,
            )
        elapsed = time.monotonic() - t0
        if not removed:
            console.print("  No known visible mark detected (gemini / doubao / jimeng / jimeng-pill / samsung).")
            _no_visible_mark_exit(source)
        console.print(f"  Removed: {', '.join(removed)}")
    else:
        target = "gemini" if mark == "auto" else mark  # --no-detect auto: gemini fallback
        chosen = registry.get_mark(target)
        det = chosen.detect(image)
        if detect and not det.detected:
            console.print(f"  {chosen.label} not detected  (conf {det.confidence:.2f}). Use --no-detect to force.")
            _no_visible_mark_exit(source)
        if det.detected:
            console.print(f"  {chosen.label} detected  ({chosen.location}, conf {det.confidence:.2f})")
        resolved = registry.resolve_removal_method(removal_method, chosen.has_capture)  # type: ignore[arg-type]
        if resolved == "inpaint" and not registry.inpaint_model_available():
            console.print(
                "  Note: --method inpaint using cv2 (install the 'migan' extra for a lightweight ONNX model)."
            )
        t0 = time.monotonic()
        with console.status(f"Removing {chosen.label}... ({resolved})"):
            result, _ = chosen.remove(
                image,
                method=removal_method,  # type: ignore[arg-type]
                inpaint_method=method,
                inpaint=inpaint,
                inpaint_strength=inpaint_strength,
                force=not detect,
            )
        elapsed = time.monotonic() - t0

    # Save (rejoins the original alpha plane unchanged)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_bgr_with_alpha(output, result, alpha)

    # Strip metadata
    if strip_metadata:
        try:
            from remove_ai_watermarks.metadata import remove_ai_metadata

            remove_ai_metadata(output, output)
        except Exception as e:
            if ctx.obj.get("verbose"):
                console.print(f"  Warning: Failed to strip metadata: {e}")

    size_kb = output.stat().st_size / 1024
    console.print(f"  Saved: {output}  ({size_kb:.0f} KB, {elapsed:.2f}s)")


# -- Universal region eraser -----------------------------------------


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
@click.argument("source", type=click.Path(exists=True, path_type=Path))
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
    the boxes you pass, regardless of colour or location. Runs on CPU. Use this
    for marks the dedicated ``visible`` engines (Gemini, Doubao) do not cover.
    """
    from remove_ai_watermarks.region_eraser import erase

    _banner()
    source = _validate_image(source)
    if output is None:
        output = source.with_stem(source.stem + "_clean")

    boxes = [_parse_region(r) for r in regions]

    image, alpha = _read_bgr_and_alpha(source)
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

    output.parent.mkdir(parents=True, exist_ok=True)
    _write_bgr_with_alpha(output, result, alpha)

    if strip_metadata:
        try:
            from remove_ai_watermarks.metadata import remove_ai_metadata

            remove_ai_metadata(output, output)
        except Exception as e:
            if ctx.obj.get("verbose"):
                console.print(f"  Warning: Failed to strip metadata: {e}")

    size_kb = output.stat().st_size / 1024
    console.print(f"  Erased {len(boxes)} region(s) -> {output}  ({size_kb:.0f} KB, {elapsed:.2f}s)")


# -- Invisible watermark removal -------------------------------------


@main.command("invisible")
@click.argument("source", type=click.Path(exists=True, path_type=Path))
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


# -- Metadata operations ---------------------------------------------


@main.command("metadata")
@click.argument("source", type=click.Path(exists=True, path_type=Path))
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
    from remove_ai_watermarks.metadata import get_ai_metadata, has_ai_metadata, remove_ai_metadata

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
    out = remove_ai_metadata(source, output, keep_standard=keep_standard)
    console.print(f"  AI metadata stripped -> {out}")


# -- Provenance identification ---------------------------------------


@main.command("identify")
@click.argument("source", type=click.Path(exists=True, path_type=Path))
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


# -- Combined "all" mode ----------------------------------------------


@main.command("all")
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--output", type=click.Path(path_type=Path), default=None, help="Output path (default: <source>_clean.<ext>)."
)
@click.option("--inpaint/--no-inpaint", default=True, help="Apply inpainting cleanup after visible removal.")
@click.option(
    "--inpaint-method", type=click.Choice(["ns", "telea", "gaussian"]), default="ns", help="Inpainting method."
)
@_visible_method_option
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
@click.pass_context
def cmd_all(
    ctx: click.Context,
    source: Path,
    output: Path | None,
    inpaint: bool,
    inpaint_method: Literal["ns", "telea", "gaussian"],
    removal_method: str,
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
) -> None:
    """Remove ALL watermarks: visible + invisible + metadata.

    Runs the full pipeline in order:
      1. Visible watermark removal (Gemini sparkle, reverse alpha blending)
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

        # -- Step 1: Visible watermark --------------------------------
        console.print("\n  1) Visible watermark removal")
        image, alpha = _read_bgr_and_alpha(source)
        if image is None:
            console.print(f"Error: Failed to read image: {source}")
            raise SystemExit(1)

        h, w = image.shape[:2]
        console.print(f"    Input: {source.name}  ({w}x{h})")

        with console.status("Removing visible watermark..."):
            result, removed_label = _remove_visible_auto(
                image, source_path=source, removal_method=removal_method, inpaint=inpaint, inpaint_method=inpaint_method
            )
            if removed_label is not None:
                console.print(f"    Visible watermark removed ({removed_label})")
            else:
                console.print("    Skipped (no visible watermark detected)")

        # Save to temp file for invisible engine input (preserve alpha if present)
        _write_bgr_with_alpha(tmp_path, result, alpha)

        # -- Step 2: Invisible watermark ------------------------------
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

        # -- Step 3: Metadata -----------------------------------------
        console.print("\n  3) AI metadata stripping")
        try:
            from remove_ai_watermarks.metadata import remove_ai_metadata

            remove_ai_metadata(tmp_path, tmp_path)
            console.print("    AI metadata stripped")
        except Exception as e:
            console.print(f"    Warning: Metadata strip failed: {e}")

        # -- Write final result ----------------------------------------
        # The invisible step (and downstream cv2.IMREAD_COLOR paths) drops alpha,
        # so re-attach the original alpha plane unchanged when writing the final
        # output for transparent formats.
        output.parent.mkdir(parents=True, exist_ok=True)
        final_bgr, _ = _read_bgr_and_alpha(tmp_path)
        if final_bgr is None:
            console.print(f"Error: Failed to read intermediate file: {tmp_path}")
            raise SystemExit(1)
        _write_bgr_with_alpha(output, final_bgr, alpha)

    finally:
        # Clean up temp file if it still exists
        if tmp_path.exists():
            tmp_path.unlink()

    # -- Done -----------------------------------------------------
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


# -- Batch command ----------------------------------------------------


def _process_batch_image(
    ctx: click.Context,
    img_path: Path,
    out_path: Path,
    mode: str,
    inpaint: bool,
    strength: float | None,
    steps: int,
    pipeline: str,
    device: str,
    seed: int | None,
    hf_token: str | None,
    humanize: float,
    removal_method: str = "auto",
    unsharp: float = 0.0,
    max_resolution: int = 0,
    min_resolution: int = 1024,
    controlnet_scale: float = 1.0,
    upscaler: str = "lanczos",
    model: str | None = None,
    guidance_scale: float | None = None,
    adaptive_polish: bool = False,
    tile: bool = False,
    tile_size: int = 1024,
    tile_overlap: int = 128,
    force: bool = False,
) -> None:
    """Process a single image for batch mode.

    Applies the requested watermark removal steps (visible, invisible,
    metadata) to *img_path* and writes the result to *out_path*.

    Raises:
        ValueError: If the image cannot be opened.
    """
    saved_alpha: NDArray[Any] | None = None

    if mode in ("visible", "all"):
        # Always read the ORIGINAL source: the visible pass is the first step, so a
        # stale out_path from a previous run must not be re-processed as if it were
        # the input. (The invisible step below reads out_path for `all` -- that chain
        # is within a single run.)
        image, alpha = _read_bgr_and_alpha(img_path)
        if image is None:
            raise ValueError("Failed to read image")

        result, _ = _remove_visible_auto(image, source_path=img_path, removal_method=removal_method, inpaint=inpaint)

        _write_bgr_with_alpha(out_path, result, alpha)
        saved_alpha = alpha

    if mode in ("invisible", "all"):
        from remove_ai_watermarks.invisible_engine import (
            is_available as invisible_available,
        )

        # Skip the destructive regeneration when no invisible watermark is locally
        # detectable (would only degrade a clean image). Read the pristine `img_path`;
        # `out_path` may already be the visible-processed result. --force overrides.
        skip_no_signal = _should_skip_invisible_scrub(force, img_path)
        if invisible_available() and not skip_no_signal:
            from remove_ai_watermarks.invisible_engine import InvisibleEngine

            # Cache the engine in ctx.obj so the batch builds it once (pipeline is a
            # single CLI value, constant across the run).
            engines = ctx.obj.setdefault("_inv_engines", {})
            if pipeline not in engines:
                engines[pipeline] = InvisibleEngine(
                    model_id=model,
                    device=None if device == "auto" else device,
                    pipeline=pipeline,
                    hf_token=hf_token,
                    controlnet_conditioning_scale=controlnet_scale,
                )
            engine_inv = engines[pipeline]
            engine_inv.remove_watermark(
                img_path if mode == "invisible" else out_path,
                out_path,
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
                tile=tile,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                # Detect the vendor from the pristine original (`img_path`), not the
                # visible-processed `out_path` whose C2PA is already gone.
                vendor=vendor_for_strength(img_path),
            )
        elif skip_no_signal and mode == "invisible" and not out_path.exists():
            # No invisible target and the visible/all pass did not write out_path
            # (invisible mode): copy the input through so the output dir is complete
            # with the pixels deliberately left intact.
            src_bgr, src_alpha = _read_bgr_and_alpha(img_path)
            if src_bgr is not None:
                _write_bgr_with_alpha(out_path, src_bgr, src_alpha)

    if mode in ("metadata", "all"):
        from remove_ai_watermarks.metadata import remove_ai_metadata

        remove_ai_metadata(img_path if mode == "metadata" else out_path, out_path)

    # In "all" mode, the invisible step (color-only OpenCV paths) drops alpha,
    # so re-attach the cached alpha when the input had transparency.
    if mode == "all" and saved_alpha is not None:
        final_bgr, _ = _read_bgr_and_alpha(out_path)
        if final_bgr is not None:
            _write_bgr_with_alpha(out_path, final_bgr, saved_alpha)


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
@click.option("--inpaint/--no-inpaint", default=True, help="Apply inpainting (visible mode).")
@_visible_method_option
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
    inpaint: bool,
    removal_method: str,
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

    processed = 0
    errors = 0

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
                _process_batch_image(
                    ctx=ctx,
                    img_path=img_path,
                    out_path=out_path,
                    mode=mode,
                    inpaint=inpaint,
                    strength=strength,
                    steps=steps,
                    pipeline=pipeline,
                    device=device,
                    seed=seed,
                    hf_token=hf_token,
                    humanize=humanize,
                    removal_method=removal_method,
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
                )
                processed += 1

            except Exception as e:
                errors += 1
                if ctx.obj.get("verbose"):
                    console.print(f"  {img_path.name}: {e}")

            progress.advance(task)

    console.print(f"\n  {processed} processed" + (f"  {errors} errors" if errors else ""))


if __name__ == "__main__":
    main()
