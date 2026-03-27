"""Unified CLI for remove-ai-watermarks.

Provides commands for:
  - Visible watermark removal (Gemini sparkle) — works offline, fast
  - Invisible watermark removal (SynthID etc.) — requires GPU/diffusion models
  - AI metadata stripping — lightweight, no ML deps needed
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from remove_ai_watermarks import __version__

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
@click.option("--inpaint-strength", type=float, default=0.85, help="Inpainting blend strength (0.0–1.0).")
@click.option("--detect/--no-detect", default=True, help="Detect watermark before removal.")
@click.option("--detect-threshold", type=float, default=0.25, help="Detection confidence threshold.")
@click.option("--strip-metadata/--keep-metadata", default=True, help="Strip AI metadata from output.")
@click.pass_context
def cmd_visible(
    ctx: click.Context,
    source: Path,
    output: Path | None,
    inpaint: bool,
    inpaint_method: str,
    inpaint_strength: float,
    detect: bool,
    detect_threshold: float,
    strip_metadata: bool,
) -> None:
    """Remove visible Gemini watermark (sparkle logo) from an image.

    Uses reverse alpha blending — fast, deterministic, offline.
    """
    import cv2

    from remove_ai_watermarks.gemini_engine import GeminiEngine

    _banner()
    source = _validate_image(source)

    if output is None:
        output = source.with_stem(source.stem + "_clean")

    engine = GeminiEngine()

    # Load image
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        console.print(f"[red]Error:[/] Failed to read image: {source}")
        raise SystemExit(1)

    h, w = image.shape[:2]
    console.print(f"  [dim]Input:[/]  {source.name}  ({w}×{h})")

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
    with console.status("[cyan]Removing watermark…[/]"):
        result = engine.remove_watermark(image)

        if inpaint:
            if det.confidence > 0.15:
                region = det.region
            else:
                from remove_ai_watermarks.gemini_engine import get_watermark_config

                config = get_watermark_config(w, h)
                pos = config.get_position(w, h)
                region = (pos[0], pos[1], config.logo_size, config.logo_size)
            result = engine.inpaint_residual(
                result,
                region,
                strength=inpaint_strength,
                method=inpaint_method,
            )

    elapsed = time.monotonic() - t0

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), result)

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


# ── Invisible watermark removal ─────────────────────────────────────


@main.command("invisible")
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--output", type=click.Path(path_type=Path), default=None, help="Output path (default: <source>_clean.<ext>)."
)
@click.option("--strength", type=float, default=0.02, help="Denoising strength (0.0–1.0). Default: 0.02.")
@click.option("--steps", type=int, default=100, help="Number of denoising steps. Default: 100.")
@click.option("--pipeline", type=click.Choice(["default", "ctrlregen"]), default="default", help="Pipeline profile.")
@click.option("--device", type=click.Choice(["auto", "cpu", "mps", "cuda"]), default="auto", help="Inference device.")
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility.")
@click.option("--hf-token", type=str, default=None, help="HuggingFace API token.")
@click.option(
    "--humanize", type=float, default=0.0, help="Analog Humanizer film grain intensity (0 = off, typical: 2.0–6.0)."
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
) -> None:
    """Remove invisible AI watermarks (SynthID, StableSignature, TreeRing).

    Uses diffusion-based regeneration. Requires GPU for reasonable speed.
    Requires the [gpu] extra: pip install 'remove-ai-watermarks[gpu]'
    """
    from remove_ai_watermarks.invisible_engine import is_available as invisible_available

    if not invisible_available():
        console.print(
            "[red]Error:[/] GPU dependencies not installed.\n"
            "  Install them with: [bold]pip install 'remove-ai-watermarks[gpu]'[/]"
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
@click.option("--strength", type=float, default=0.02, help="Invisible watermark denoising strength (0.0–1.0).")
@click.option("--steps", type=int, default=100, help="Number of denoising steps for invisible removal.")
@click.option(
    "--pipeline",
    type=click.Choice(["default", "ctrlregen"]),
    default="default",
    help="Pipeline profile for invisible removal.",
)
@click.option("--model", type=str, default=None, help="HuggingFace model ID for invisible removal.")
@click.option("--device", type=click.Choice(["auto", "cpu", "mps", "cuda"]), default="auto", help="Inference device.")
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility.")
@click.option("--hf-token", type=str, default=None, help="HuggingFace API token.")
@click.option(
    "--humanize", type=float, default=0.0, help="Analog Humanizer film grain intensity (0 = off, typical: 2.0–6.0)."
)
@click.pass_context
def cmd_all(
    ctx: click.Context,
    source: Path,
    output: Path | None,
    inpaint: bool,
    inpaint_method: str,
    strength: float,
    steps: int,
    pipeline: str,
    model: str | None,
    device: str,
    seed: int | None,
    hf_token: str | None,
    humanize: float,
) -> None:
    """Remove ALL watermarks: visible + invisible + metadata.

    Runs the full pipeline in order:
      1. Visible watermark removal (Gemini sparkle, reverse alpha blending)
      2. Invisible watermark removal (SynthID etc., diffusion regeneration)
      3. AI metadata stripping (EXIF, PNG text, C2PA)

    If invisible watermark deps are not installed, skips step 2 with a warning.
    """
    import cv2

    from remove_ai_watermarks.gemini_engine import GeminiEngine, get_watermark_config

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
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            console.print(f"[red]Error:[/] Failed to read image: {source}")
            raise SystemExit(1)

        h, w = image.shape[:2]
        console.print(f"    [dim]Input:[/] {source.name}  ({w}×{h})")

        with console.status("[cyan]Removing visible watermark…[/]"):
            det = engine.detect_watermark(image)
            if det.detected:
                result = engine.remove_watermark(image)
                if inpaint:
                    if det.confidence > 0.15:
                        region = det.region
                    else:
                        config = get_watermark_config(w, h)
                        pos = config.get_position(w, h)
                        region = (pos[0], pos[1], config.logo_size, config.logo_size)
                    result = engine.inpaint_residual(result, region, method=inpaint_method)
                console.print("    [green]✓[/] Visible watermark removed")
            else:
                result = image.copy()
                console.print("    [dim]Skipped (no visible watermark detected)[/]")

        # Save to temp file for invisible engine input
        cv2.imwrite(str(tmp_path), result)

        # ── Step 2: Invisible watermark ──────────────────────────────
        console.print("\n  [bold cyan]② Invisible watermark removal[/]")
        from remove_ai_watermarks.invisible_engine import is_available as invisible_available

        if not invisible_available():
            console.print(
                "    [yellow]⚠[/] Skipped — GPU dependencies not installed.\n"
                "    Install them with: [bold]pip install 'remove-ai-watermarks[gpu]'[/]"
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
        import shutil

        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_path), str(output))

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
) -> None:
    """Process a single image for batch mode.

    Applies the requested watermark removal steps (visible, invisible,
    metadata) to *img_path* and writes the result to *out_path*.

    Raises:
        ValueError: If the image cannot be opened.
    """
    if mode in ("visible", "all"):
        import cv2

        from remove_ai_watermarks.gemini_engine import (
            GeminiEngine,
            get_watermark_config,
        )

        if "_vis_engine" not in ctx.obj:
            ctx.obj["_vis_engine"] = GeminiEngine()
        engine = ctx.obj["_vis_engine"]
        read_path = img_path
        if mode == "all" and out_path.exists():
            read_path = out_path
        image = cv2.imread(str(read_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to read image")

        det = engine.detect_watermark(image)
        if det.detected:
            result = engine.remove_watermark(image)
            if inpaint:
                if det.confidence > 0.15:
                    region = det.region
                else:
                    h, w = image.shape[:2]
                    config = get_watermark_config(w, h)
                    pos = config.get_position(w, h)
                    region = (pos[0], pos[1], config.logo_size, config.logo_size)

                result = engine.inpaint_residual(result, region)
        else:
            result = image.copy()

        cv2.imwrite(str(out_path), result)

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
            )

    if mode in ("metadata", "all"):
        from remove_ai_watermarks.metadata import remove_ai_metadata

        remove_ai_metadata(img_path if mode == "metadata" else out_path, out_path)


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
    "--humanize", type=float, default=0.0, help="Analog Humanizer film grain intensity (0 = off, typical: 2.0–6.0)."
)
@click.option("--pipeline", type=click.Choice(["default", "ctrlregen"]), default="default", help="Pipeline profile.")
@click.option("--device", type=click.Choice(["auto", "cpu", "mps", "cuda"]), default="auto", help="Inference device.")
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility.")
@click.option("--hf-token", type=str, default=None, help="HuggingFace API token.")
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
