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
from typing import TYPE_CHECKING, Any, Literal

import click

from remove_ai_watermarks import __version__, watermark_registry
from remove_ai_watermarks.noai.watermark_profiles import resolve_strength

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray

    from remove_ai_watermarks.gemini_engine import DetectionResult

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

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp"}


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


def _watermark_region(det: DetectionResult, width: int, height: int) -> tuple[int, int, int, int]:
    """Pick a watermark bbox: detector's region if confident, else the default config slot."""
    if det.confidence > 0.15:
        return det.region
    from remove_ai_watermarks.gemini_engine import get_watermark_config

    config = get_watermark_config(width, height)
    px, py = config.get_position(width, height)
    return (px, py, config.logo_size, config.logo_size)


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
    "All marks are removed by exact reverse-alpha against a captured alpha map.",
)
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
    strip_metadata: bool,
) -> None:
    """Remove a known visible AI watermark from an image.

    Finds a known mark in its usual place (Gemini sparkle / Doubao text) via the
    watermark registry and removes it by exact reverse-alpha against a captured
    alpha map -- recovering the true pixels, not an inpaint guess. ``--mark auto``
    picks the strongest detected mark. For arbitrary logos/objects, use ``erase``.
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

    # Resolve the target mark from the known-watermark registry. ``auto`` scans
    # every in-auto mark in its usual place and picks the strongest; an explicit
    # ``--mark <key>`` targets that one (the user asserts its presence).
    if mark == "auto":
        best = registry.best_auto_mark(image)
        if best is None:
            console.print("  Warning: No known visible mark detected (gemini / doubao).")
            if detect:
                console.print("  Skipping. Use --mark <name> --no-detect to force.")
                raise SystemExit(0)
            target = "gemini"  # forced (no-detect): fall back to the default mark
        else:
            target = best.key
            console.print(f"  Mark auto: {best.label}  ({best.location}, conf {best.confidence:.2f})")
    else:
        target = mark

    chosen = registry.get_mark(target)
    det = chosen.detect(image)
    if detect and not det.detected:
        console.print(f"  Warning: {chosen.label} not detected  (conf {det.confidence:.2f}). Use --no-detect to force.")
        raise SystemExit(0)
    if det.detected:
        console.print(f"  {chosen.label} detected  ({chosen.location}, conf {det.confidence:.2f})")

    method: Literal["telea", "ns"] = "ns" if inpaint_method == "ns" else "telea"
    t0 = time.monotonic()
    with console.status(f"Removing {chosen.label}... ({chosen.recovery})"):
        result, _ = chosen.remove(
            image,
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
    type=click.Choice(["cv2", "lama"]),
    default="cv2",
    help="Inpaint backend. cv2: instant, no deps. lama: onnxruntime big-LaMa, better quality (extra 'lama').",
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
    backend: Literal["cv2", "lama"],
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
@click.option(
    "--strength",
    type=float,
    default=None,
    help="Denoising strength (0.0-1.0). Default: 0.30 (SDXL SynthID threshold); ctrlregen uses 1.0.",
)
@click.option("--steps", type=int, default=50, help="Number of denoising steps. Default: 50.")
@click.option(
    "--pipeline",
    type=click.Choice(["default", "ctrlregen"]),
    default="default",
    help="Pipeline profile (default=SDXL; ctrlregen=CtrlRegen, EXPERIMENTAL/destructive at clean-noise).",
)
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
@click.option(
    "--no-protect-text",
    is_flag=True,
    default=False,
    help="Disable automatic text protection (text/CJK is preserved by default on the SDXL pipeline).",
)
@click.option(
    "--no-protect-faces",
    is_flag=True,
    default=False,
    help="Disable face protection (skips the YOLO face detector; use when the image has no people).",
)
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
    max_resolution: int,
    no_protect_text: bool,
    no_protect_faces: bool,
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
    if output is None:
        output = source.with_stem(source.stem + "_clean")

    device_str = None if device == "auto" else device

    def progress_cb(msg: str) -> None:
        console.print(f"  {msg}")

    engine = InvisibleEngine(
        device=device_str,
        pipeline=pipeline,
        hf_token=hf_token,
        progress_callback=progress_cb,
    )

    console.print(f"  Input:    {source.name}")
    console.print(f"  Pipeline: {pipeline}")
    console.print(f"  Strength: {resolve_strength(strength, pipeline)}  Steps: {steps}")

    t0 = time.monotonic()
    result_path = engine.remove_watermark(
        image_path=source,
        output_path=output,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=None,
        seed=seed,
        humanize=humanize,
        protect_text=not no_protect_text,
        protect_faces=not no_protect_faces,
        max_resolution=max_resolution,
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
@click.option(
    "--strength",
    type=float,
    default=None,
    help="Invisible watermark denoising strength. Default: 0.30 (SDXL); ctrlregen uses 1.0.",
)
@click.option("--steps", type=int, default=50, help="Number of denoising steps for invisible removal.")
@click.option(
    "--pipeline",
    type=click.Choice(["default", "ctrlregen"]),
    default="default",
    help="Pipeline profile (default=SDXL; ctrlregen=CtrlRegen, EXPERIMENTAL/destructive at clean-noise).",
)
@click.option("--model", type=str, default=None, help="HuggingFace model ID for invisible removal.")
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
@click.option(
    "--no-protect-text",
    is_flag=True,
    default=False,
    help="Disable automatic text protection (text/CJK is preserved by default on the SDXL pipeline).",
)
@click.option(
    "--no-protect-faces",
    is_flag=True,
    default=False,
    help="Disable face protection (skips the YOLO face detector; use when the image has no people).",
)
@click.pass_context
def cmd_all(
    ctx: click.Context,
    source: Path,
    output: Path | None,
    inpaint: bool,
    inpaint_method: Literal["ns", "telea", "gaussian"],
    strength: float | None,
    steps: int,
    pipeline: str,
    model: str | None,
    device: str,
    seed: int | None,
    hf_token: str | None,
    humanize: float,
    max_resolution: int,
    no_protect_text: bool,
    no_protect_faces: bool,
) -> None:
    """Remove ALL watermarks: visible + invisible + metadata.

    Runs the full pipeline in order:
      1. Visible watermark removal (Gemini sparkle, reverse alpha blending)
      2. Invisible watermark removal (SynthID etc., diffusion regeneration)
      3. AI metadata stripping (EXIF, PNG text, C2PA)

    If invisible watermark deps are not installed, skips step 2 with a warning.
    """
    from remove_ai_watermarks.gemini_engine import GeminiEngine

    _banner()
    source = _validate_image(source)

    if output is None:
        output = source.with_stem(source.stem + "_clean")

    t0 = time.monotonic()

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
        engine = GeminiEngine()
        image, alpha = _read_bgr_and_alpha(source)
        if image is None:
            console.print(f"Error: Failed to read image: {source}")
            raise SystemExit(1)

        h, w = image.shape[:2]
        console.print(f"    Input: {source.name}  ({w}x{h})")

        with console.status("Removing visible watermark..."):
            det = engine.detect_watermark(image)
            if det.detected:
                result = engine.remove_watermark(image)
                if inpaint:
                    region = _watermark_region(det, w, h)
                    result = engine.inpaint_residual(result, region, method=inpaint_method)
                console.print("    Visible watermark removed")
            else:
                result = image.copy()
                console.print("    Skipped (no visible watermark detected)")

        # Save to temp file for invisible engine input (preserve alpha if present)
        _write_bgr_with_alpha(tmp_path, result, alpha)

        # -- Step 2: Invisible watermark ------------------------------
        console.print("\n  2) Invisible watermark removal")
        from remove_ai_watermarks.invisible_engine import is_available as invisible_available

        if not invisible_available():
            console.print(
                "    Warning: Skipped - GPU dependencies not installed.\n"
                "    Install them with: pip install 'remove-ai-watermarks[gpu]'"
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
            )

            console.print(f"    Strength: {resolve_strength(strength, pipeline)}  Steps: {steps}")
            inv_engine.remove_watermark(
                image_path=tmp_path,
                output_path=tmp_path,
                strength=strength,
                num_inference_steps=steps,
                seed=seed,
                humanize=humanize,
                protect_text=not no_protect_text,
                protect_faces=not no_protect_faces,
                max_resolution=max_resolution,
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
    max_resolution: int = 0,
) -> None:
    """Process a single image for batch mode.

    Applies the requested watermark removal steps (visible, invisible,
    metadata) to *img_path* and writes the result to *out_path*.

    Raises:
        ValueError: If the image cannot be opened.
    """
    saved_alpha: NDArray[Any] | None = None

    if mode in ("visible", "all"):
        from remove_ai_watermarks.gemini_engine import GeminiEngine

        if "_vis_engine" not in ctx.obj:
            ctx.obj["_vis_engine"] = GeminiEngine()
        engine = ctx.obj["_vis_engine"]
        read_path = img_path
        if mode == "all" and out_path.exists():
            read_path = out_path
        image, alpha = _read_bgr_and_alpha(read_path)
        if image is None:
            raise ValueError("Failed to read image")

        det = engine.detect_watermark(image)
        if det.detected:
            result = engine.remove_watermark(image)
            if inpaint:
                h, w = image.shape[:2]
                region = _watermark_region(det, w, h)
                result = engine.inpaint_residual(result, region)
        else:
            result = image.copy()

        _write_bgr_with_alpha(out_path, result, alpha)
        saved_alpha = alpha

    if mode in ("invisible", "all"):
        from remove_ai_watermarks.invisible_engine import (
            is_available as invisible_available,
        )

        if invisible_available():
            from remove_ai_watermarks.invisible_engine import InvisibleEngine

            if "_inv_engine" not in ctx.obj:
                ctx.obj["_inv_engine"] = InvisibleEngine(
                    device=None if device == "auto" else device,
                    pipeline=pipeline,
                    hf_token=hf_token,
                )
            engine_inv = ctx.obj["_inv_engine"]
            engine_inv.remove_watermark(
                img_path if mode == "invisible" else out_path,
                out_path,
                strength=strength,
                num_inference_steps=steps,
                seed=seed,
                humanize=humanize,
                max_resolution=max_resolution,
            )

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
@click.option("--strength", type=float, default=None, help="Denoising strength (invisible mode).")
@click.option("--steps", type=int, default=50, help="Number of denoising steps (invisible mode).")
@click.option("--inpaint/--no-inpaint", default=True, help="Apply inpainting (visible mode).")
@click.option(
    "--humanize", type=float, default=0.0, help="Analog Humanizer film grain intensity (0 = off, typical: 2.0-6.0)."
)
@click.option(
    "--pipeline",
    type=click.Choice(["default", "ctrlregen"]),
    default="default",
    help="Pipeline profile (default=SDXL; ctrlregen=CtrlRegen, EXPERIMENTAL/destructive at clean-noise).",
)
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
    humanize: float,
    max_resolution: int,
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
                    max_resolution=max_resolution,
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
