"""Unified CLI for remove-ai-watermarks.

Provides commands for:
  - Visible watermark removal (Gemini sparkle) — works offline, fast
  - Invisible watermark removal (SynthID etc.) — requires GPU/diffusion models
  - AI metadata stripping — lightweight, no ML deps needed
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from remove_ai_watermarks import __version__

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from remove_ai_watermarks.gemini_engine import DetectionResult, GeminiEngine

console = Console()

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
            f"[bold cyan]Remove-AI-Watermarks[/] [dim]v{__version__}[/]\n[dim]Visible & invisible watermark removal[/]",
            border_style="cyan",
            padding=(0, 2),
        )
    )


def _validate_image(path: Path) -> Path:
    if not path.exists():
        console.print(f"[red]Error:[/] File not found: {path}")
        raise SystemExit(1)
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        console.print(
            f"[yellow]Warning:[/] {path.suffix} may not be supported (expected: {', '.join(SUPPORTED_FORMATS)})"
        )
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
    clear_region: tuple[int, int, int, int] | None = None,
    pad: int = 6,
) -> None:
    """Write BGR (with optional alpha) to ``path``.

    When ``alpha`` is provided and the output extension supports it, writes a
    4-channel image. If ``clear_region`` is given as ``(x, y, w, h)``, alpha is
    forced to 0 inside that bbox (expanded by ``pad`` px) so the watermark area
    becomes fully transparent in the saved file.
    """
    import numpy as np

    from remove_ai_watermarks import image_io

    if alpha is None or path.suffix.lower() not in _ALPHA_FORMATS:
        image_io.imwrite(path, bgr)
        return

    alpha_out = alpha
    if clear_region is not None:
        alpha_out = alpha.copy()
        x, y, w, h = clear_region
        height, width = alpha.shape[:2]
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
        if x1 > x0 and y1 > y0:
            alpha_out[y0:y1, x0:x1] = 0

    bgra = np.dstack([bgr, alpha_out])
    image_io.imwrite(path, bgra)


def _run_doubao_if_selected(
    ctx: click.Context,
    image: NDArray[Any],
    alpha: NDArray[Any] | None,
    output: Path,
    mark: str,
    gemini_engine: GeminiEngine,
    detect: bool,
    detect_threshold: float,
    inpaint_method: str,
    strip_metadata: bool,
) -> bool:
    """Run the Doubao text-strip removal path when it is the selected mark.

    Returns True when this path handled the image (caller should stop). In
    ``auto`` mode the Doubao detector competes with the Gemini detector and wins
    only when it is both positive and at least as confident.
    """
    from remove_ai_watermarks.doubao_engine import DoubaoEngine

    doubao = DoubaoEngine()
    d_det = doubao.detect(image)

    if mark == "auto":
        g_det = gemini_engine.detect_watermark(image)
        use_doubao = d_det.detected and d_det.confidence >= g_det.confidence
        console.print(
            f"  [dim]Mark auto:[/] gemini={g_det.confidence:.2f} doubao={d_det.confidence:.2f} "
            f"-> {'doubao' if use_doubao else 'gemini'}"
        )
    else:
        use_doubao = mark == "doubao"

    if not use_doubao:
        return False

    if detect and not d_det.detected and d_det.confidence < detect_threshold:
        console.print(
            f"  [yellow]⚠[/] Doubao mark not detected  [dim](coverage {d_det.coverage:.1%}). "
            f"Use --no-detect to force.[/]"
        )
        raise SystemExit(0)

    method: Literal["telea", "ns"] = "ns" if inpaint_method == "ns" else "telea"
    t0 = time.monotonic()
    with console.status("[cyan]Removing Doubao watermark…[/]"):
        result = doubao.remove_watermark(image, inpaint_method=method)
    elapsed = time.monotonic() - t0

    output.parent.mkdir(parents=True, exist_ok=True)
    _write_bgr_with_alpha(output, result, alpha, clear_region=d_det.region)

    if strip_metadata:
        try:
            from remove_ai_watermarks.metadata import remove_ai_metadata

            remove_ai_metadata(output, output)
        except Exception as e:
            if ctx.obj.get("verbose"):
                console.print(f"  [yellow]⚠[/] Failed to strip metadata: {e}")

    size_kb = output.stat().st_size / 1024
    console.print(f"  [green]✓[/] Doubao mark removed → {output}  [dim]({size_kb:.0f} KB, {elapsed:.2f}s)[/]")
    return True


# ── Main group ───────────────────────────────────────────────────────


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


# ── Visible (Gemini) watermark removal ───────────────────────────────


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
@click.option("--detect-threshold", type=float, default=0.25, help="Detection confidence threshold.")
@click.option(
    "--mark",
    type=click.Choice(["auto", "gemini", "doubao"]),
    default="auto",
    help="Which visible mark to target. auto picks the stronger of the two detectors.",
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
    detect_threshold: float,
    mark: str,
    strip_metadata: bool,
) -> None:
    """Remove a visible AI watermark from an image.

    Targets the Gemini sparkle logo (reverse alpha blending) or the Doubao
    "豆包AI生成" text strip (locate -> mask -> inpaint). Fast, deterministic,
    offline. ``--mark auto`` picks whichever detector fires stronger.
    """
    from remove_ai_watermarks.gemini_engine import GeminiEngine

    _banner()
    source = _validate_image(source)

    if output is None:
        output = source.with_stem(source.stem + "_clean")

    engine = GeminiEngine()

    # Load image (preserving any alpha channel separately)
    image, alpha = _read_bgr_and_alpha(source)
    if image is None:
        console.print(f"[red]Error:[/] Failed to read image: {source}")
        raise SystemExit(1)

    h, w = image.shape[:2]
    console.print(f"  [dim]Input:[/]  {source.name}  ({w}x{h})")

    # Resolve which visible mark to target, then run the Doubao path if chosen.
    if _run_doubao_if_selected(
        ctx, image, alpha, output, mark, engine, detect, detect_threshold, inpaint_method, strip_metadata
    ):
        return

    # Detection (we always detect softly, to find dynamic region for inpainting)
    with console.status("[cyan]Detecting watermark…[/]"):
        det = engine.detect_watermark(image)

    if detect:
        if det.detected:
            console.print(
                f"  [green]✓[/] Watermark detected  "
                f"[dim](confidence: {det.confidence:.1%}, "
                f"spatial: {det.spatial_score:.3f}, "
                f"gradient: {det.gradient_score:.3f})[/]"
            )
        else:
            console.print(f"  [yellow]⚠[/] Watermark not detected  [dim](confidence: {det.confidence:.1%})[/]")
            if det.confidence < detect_threshold:
                console.print("  [dim]Skipping. Use --no-detect to force removal.[/]")
                raise SystemExit(0)

    # Removal
    t0 = time.monotonic()
    region: tuple[int, int, int, int] | None = None
    with console.status("[cyan]Removing watermark…[/]"):
        result = engine.remove_watermark(image)

        if inpaint:
            region = _watermark_region(det, w, h)
            result = engine.inpaint_residual(
                result,
                region,
                strength=inpaint_strength,
                method=inpaint_method,
            )

    elapsed = time.monotonic() - t0

    # Save (preserves transparency by clearing alpha in the watermark region)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_bgr_with_alpha(output, result, alpha, clear_region=region)

    # Strip metadata
    if strip_metadata:
        try:
            from remove_ai_watermarks.metadata import remove_ai_metadata

            remove_ai_metadata(output, output)
        except Exception as e:
            if ctx.obj.get("verbose"):
                console.print(f"  [yellow]⚠[/] Failed to strip metadata: {e}")

    size_kb = output.stat().st_size / 1024
    console.print(f"  [green]✓[/] Saved: {output}  [dim]({size_kb:.0f} KB, {elapsed:.2f}s)[/]")


# ── Universal region eraser ─────────────────────────────────────────


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
        console.print(f"[red]Error:[/] Failed to read image: {source}")
        raise SystemExit(1)
    h, w = image.shape[:2]
    console.print(f"  [dim]Input:[/]  {source.name}  ({w}x{h})  [dim]{len(boxes)} region(s), backend={backend}[/]")

    t0 = time.monotonic()
    method: Literal["telea", "ns"] = "ns" if inpaint_method == "ns" else "telea"
    try:
        with console.status(f"[cyan]Erasing ({backend})…[/]"):
            result = erase(image, boxes=boxes, backend=backend, dilate=dilate, cv2_method=method)
    except RuntimeError as e:
        console.print(f"  [red]Error:[/] {e}")
        raise SystemExit(1) from e
    elapsed = time.monotonic() - t0

    output.parent.mkdir(parents=True, exist_ok=True)
    clear = boxes[0] if len(boxes) == 1 else None
    _write_bgr_with_alpha(output, result, alpha, clear_region=clear)

    if strip_metadata:
        try:
            from remove_ai_watermarks.metadata import remove_ai_metadata

            remove_ai_metadata(output, output)
        except Exception as e:
            if ctx.obj.get("verbose"):
                console.print(f"  [yellow]⚠[/] Failed to strip metadata: {e}")

    size_kb = output.stat().st_size / 1024
    console.print(f"  [green]✓[/] Erased {len(boxes)} region(s) → {output}  [dim]({size_kb:.0f} KB, {elapsed:.2f}s)[/]")


# ── Invisible watermark removal ─────────────────────────────────────


@main.command("invisible")
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--output", type=click.Path(path_type=Path), default=None, help="Output path (default: <source>_clean.<ext>)."
)
@click.option("--strength", type=float, default=0.05, help="Denoising strength (0.0-1.0). Default: 0.05.")
@click.option("--steps", type=int, default=50, help="Number of denoising steps. Default: 50.")
@click.option(
    "--pipeline",
    type=click.Choice(["default", "ctrlregen"]),
    default="default",
    help="Pipeline profile (default=SDXL, ctrlregen=CtrlRegen).",
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
@click.pass_context
def cmd_invisible(
    ctx: click.Context,
    source: Path,
    output: Path | None,
    strength: float,
    steps: int,
    pipeline: str,
    device: str,
    seed: int | None,
    hf_token: str | None,
    humanize: float,
    max_resolution: int,
    no_protect_text: bool,
) -> None:
    """Remove invisible AI watermarks (SynthID, StableSignature, TreeRing).

    Uses diffusion-based regeneration. Requires GPU for reasonable speed.
    Requires the [gpu] extra: pip install 'remove-ai-watermarks[gpu]'
    """
    from remove_ai_watermarks.invisible_engine import is_available as invisible_available

    if not invisible_available():
        console.print(
            "[red]Error:[/] GPU dependencies not installed.\n"
            "  Install them with: [bold]pip install 'remove-ai-watermarks\\[gpu]'[/]"
        )
        raise SystemExit(1)

    from remove_ai_watermarks.invisible_engine import InvisibleEngine

    source = _validate_image(source)
    if output is None:
        output = source.with_stem(source.stem + "_clean")

    device_str = None if device == "auto" else device

    def progress_cb(msg: str) -> None:
        console.print(f"  [dim]{msg}[/]")

    engine = InvisibleEngine(
        device=device_str,
        pipeline=pipeline,
        hf_token=hf_token,
        progress_callback=progress_cb,
    )

    console.print(f"  [dim]Input:[/]    {source.name}")
    console.print(f"  [dim]Pipeline:[/] {pipeline}")
    console.print(f"  [dim]Strength:[/] {strength}  Steps: {steps}")

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
        max_resolution=max_resolution,
    )
    elapsed = time.monotonic() - t0

    size_kb = result_path.stat().st_size / 1024
    console.print(f"\n  [green]✓[/] Saved: {result_path}  [dim]({size_kb:.0f} KB, {elapsed:.1f}s)[/]")


# ── Metadata operations ─────────────────────────────────────────────


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
    """Check or remove AI-generation metadata from images.

    Strips EXIF AI tags, PNG text chunks, and C2PA provenance manifests.
    """
    from remove_ai_watermarks.metadata import get_ai_metadata, has_ai_metadata, remove_ai_metadata

    _banner()
    source = _validate_image(source)

    if check or (not remove):
        has_ai = has_ai_metadata(source)
        if has_ai:
            console.print(f"  [yellow]⚠[/] AI metadata detected in {source.name}:")
            meta = get_ai_metadata(source)
            if synthid := meta.get("synthid_watermark"):
                console.print(f"  [bold yellow]⚠ SynthID pixel watermark {synthid}[/]")
            table = Table(show_header=True, header_style="bold")
            table.add_column("Key", style="cyan")
            table.add_column("Value")
            for k, v in meta.items():
                table.add_row(k, str(v)[:80])
            console.print(table)
        else:
            console.print(f"  [green]✓[/] No AI metadata found in {source.name}")

        if not remove:
            return

    # Remove
    out = remove_ai_metadata(source, output, keep_standard=keep_standard)
    console.print(f"  [green]✓[/] AI metadata stripped → {out}")


# ── Provenance identification ───────────────────────────────────────


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
    verdict = {True: "[yellow]AI-generated[/]", False: "[green]not AI[/]", None: "[dim]unknown[/]"}[
        report.is_ai_generated
    ]
    console.print(f"\n  Verdict: {verdict}  [dim](confidence: {report.confidence})[/]")
    console.print(f"  Platform: {report.platform or '[dim]undetermined[/]'}")

    if report.is_ai_generated is None:
        console.print(
            "  [dim]No locally-readable AI signal found. This is not the same as 'clean': "
            "metadata is often stripped by re-encoding, screenshots, or upload, and SynthID-class "
            "pixel watermarks (Gemini / Nano Banana / gpt-image) have no local detector. "
            "See caveats below.[/]"
        )

    if report.integrity_clashes:
        console.print("\n  [bold red]⚠ Integrity clash[/] [dim](provenance signals contradict each other)[/]")
        for clash in report.integrity_clashes:
            console.print(f"  [red]- {clash}[/]")

    if report.watermarks:
        table = Table(show_header=True, header_style="bold", title="Watermarks / provenance markers")
        table.add_column("Marker", style="cyan")
        for wm in report.watermarks:
            table.add_row(wm)
        console.print(table)
    else:
        console.print("  [dim]No watermarks or provenance markers found.[/]")

    if report.caveats:
        console.print("\n  [dim]Caveats:[/]")
        for c in report.caveats:
            console.print(f"  [dim]- {c}[/]")


# ── Combined "all" mode ──────────────────────────────────────────────


@main.command("all")
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--output", type=click.Path(path_type=Path), default=None, help="Output path (default: <source>_clean.<ext>)."
)
@click.option("--inpaint/--no-inpaint", default=True, help="Apply inpainting cleanup after visible removal.")
@click.option(
    "--inpaint-method", type=click.Choice(["ns", "telea", "gaussian"]), default="ns", help="Inpainting method."
)
@click.option("--strength", type=float, default=0.05, help="Invisible watermark denoising strength (0.0-1.0).")
@click.option("--steps", type=int, default=50, help="Number of denoising steps for invisible removal.")
@click.option(
    "--pipeline",
    type=click.Choice(["default", "ctrlregen"]),
    default="default",
    help="Pipeline profile (default=SDXL, ctrlregen=CtrlRegen).",
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
@click.pass_context
def cmd_all(
    ctx: click.Context,
    source: Path,
    output: Path | None,
    inpaint: bool,
    inpaint_method: Literal["ns", "telea", "gaussian"],
    strength: float,
    steps: int,
    pipeline: str,
    model: str | None,
    device: str,
    seed: int | None,
    hf_token: str | None,
    humanize: float,
    max_resolution: int,
    no_protect_text: bool,
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

        # ── Step 1: Visible watermark ────────────────────────────────
        console.print("\n  [bold cyan]① Visible watermark removal[/]")
        engine = GeminiEngine()
        image, alpha = _read_bgr_and_alpha(source)
        if image is None:
            console.print(f"[red]Error:[/] Failed to read image: {source}")
            raise SystemExit(1)

        h, w = image.shape[:2]
        console.print(f"    [dim]Input:[/] {source.name}  ({w}x{h})")

        region: tuple[int, int, int, int] | None = None
        with console.status("[cyan]Removing visible watermark…[/]"):
            det = engine.detect_watermark(image)
            if det.detected:
                result = engine.remove_watermark(image)
                if inpaint:
                    region = _watermark_region(det, w, h)
                    result = engine.inpaint_residual(result, region, method=inpaint_method)
                console.print("    [green]✓[/] Visible watermark removed")
            else:
                result = image.copy()
                console.print("    [dim]Skipped (no visible watermark detected)[/]")

        # Save to temp file for invisible engine input (preserve alpha if present)
        _write_bgr_with_alpha(tmp_path, result, alpha, clear_region=region)

        # ── Step 2: Invisible watermark ──────────────────────────────
        console.print("\n  [bold cyan]② Invisible watermark removal[/]")
        from remove_ai_watermarks.invisible_engine import is_available as invisible_available

        if not invisible_available():
            console.print(
                "    [yellow]⚠[/] Skipped — GPU dependencies not installed.\n"
                "    Install them with: [bold]pip install 'remove-ai-watermarks\\[gpu]'[/]"
            )
        else:
            from remove_ai_watermarks.invisible_engine import InvisibleEngine

            device_str = None if device == "auto" else device

            def progress_cb(msg: str) -> None:
                console.print(f"    [dim]{msg}[/]")

            inv_engine = InvisibleEngine(
                model_id=model,
                device=device_str,
                pipeline=pipeline,
                hf_token=hf_token,
                progress_callback=progress_cb,
            )

            console.print(f"    [dim]Strength:[/] {strength}  Steps: {steps}")
            inv_engine.remove_watermark(
                image_path=tmp_path,
                output_path=tmp_path,
                strength=strength,
                num_inference_steps=steps,
                seed=seed,
                humanize=humanize,
                protect_text=not no_protect_text,
                max_resolution=max_resolution,
            )
            console.print("    [green]✓[/] Invisible watermark removed")

        # ── Step 3: Metadata ─────────────────────────────────────────
        console.print("\n  [bold cyan]③ AI metadata stripping[/]")
        try:
            from remove_ai_watermarks.metadata import remove_ai_metadata

            remove_ai_metadata(tmp_path, tmp_path)
            console.print("    [green]✓[/] AI metadata stripped")
        except Exception as e:
            console.print(f"    [yellow]⚠[/] Metadata strip failed: {e}")

        # ── Write final result ────────────────────────────────────────
        # The invisible step (and downstream cv2.IMREAD_COLOR paths) drops alpha,
        # so re-attach the original alpha (with the watermark region cleared)
        # when writing the final output for transparent formats.
        output.parent.mkdir(parents=True, exist_ok=True)
        final_bgr, _ = _read_bgr_and_alpha(tmp_path)
        if final_bgr is None:
            console.print(f"[red]Error:[/] Failed to read intermediate file: {tmp_path}")
            raise SystemExit(1)
        _write_bgr_with_alpha(output, final_bgr, alpha, clear_region=region)

    finally:
        # Clean up temp file if it still exists
        if tmp_path.exists():
            tmp_path.unlink()

    # ── Done ─────────────────────────────────────────────────────
    elapsed = time.monotonic() - t0
    size_kb = output.stat().st_size / 1024
    console.print(f"\n  [bold green]✓ Done:[/] {output}  [dim]({size_kb:.0f} KB, {elapsed:.1f}s total)[/]")


# ── Batch command ────────────────────────────────────────────────────


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
    saved_region: tuple[int, int, int, int] | None = None

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

        region: tuple[int, int, int, int] | None = None
        det = engine.detect_watermark(image)
        if det.detected:
            result = engine.remove_watermark(image)
            if inpaint:
                h, w = image.shape[:2]
                region = _watermark_region(det, w, h)
                result = engine.inpaint_residual(result, region)
        else:
            result = image.copy()

        _write_bgr_with_alpha(out_path, result, alpha, clear_region=region)
        saved_alpha = alpha
        saved_region = region

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
            _write_bgr_with_alpha(out_path, final_bgr, saved_alpha, clear_region=saved_region)


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
    help="Pipeline profile (default=SDXL, ctrlregen=CtrlRegen).",
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
        console.print(f"[yellow]No supported images found in {directory}[/]")
        return

    console.print(f"  Found [bold]{len(images)}[/] images in {directory}")
    console.print(f"  Output → {output_dir}")
    console.print(f"  Mode: [cyan]{mode}[/]")

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
        task = progress.add_task("Processing…", total=len(images))

        for img_path in images:
            out_path = output_dir / img_path.name
            progress.update(task, description=f"[cyan]{img_path.name}[/]")

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
                    console.print(f"  [red]✗[/] {img_path.name}: {e}")

            progress.advance(task)

    console.print(f"\n  [green]✓[/] {processed} processed" + (f"  [red]✗[/] {errors} errors" if errors else ""))


if __name__ == "__main__":
    main()
