"""Gradio GUI for remove-ai-watermarks."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from remove_ai_watermarks import __version__

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


class GuiError(RuntimeError):
    """User-facing GUI error."""


def _require_gradio() -> Any:
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Gradio is not installed. Install GUI dependencies with: pip install 'remove-ai-watermarks[gui]'"
        ) from exc
    return gr


def _input_path(file_obj: Any) -> Path:
    if file_obj is None:
        raise GuiError("Please upload an image first.")
    raw_path = getattr(file_obj, "name", file_obj)
    path = Path(str(raw_path))
    if not path.exists():
        raise GuiError(f"Uploaded file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise GuiError("Supported image formats: PNG, JPG, JPEG, WEBP.")
    return path


def _temp_output(source: Path, suffix: str = "_clean") -> Path:
    out_dir = Path(tempfile.mkdtemp(prefix="remove-ai-watermarks-gui-"))
    return out_dir / f"{source.stem}{suffix}{source.suffix}"


def _copy_to_output(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)


def _format_metadata(meta: dict[str, Any]) -> str:
    if not meta:
        return "No AI metadata found."
    lines: list[str] = []
    for key, value in meta.items():
        value_str = (
            json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else str(value)
        )
        lines.append(f"- **{key}**: {value_str}")
    return "\n".join(lines)


def _format_identify_report(report: Any) -> str:
    data = asdict(report)
    watermarks: list[str] = []
    raw_watermarks = data.get("watermarks")
    if isinstance(raw_watermarks, list):
        for item in cast("list[object]", raw_watermarks):
            if isinstance(item, str):
                watermarks.append(item)
    caveats: list[str] = []
    raw_caveats = data.get("caveats")
    if isinstance(raw_caveats, list):
        for item in cast("list[object]", raw_caveats):
            if isinstance(item, str):
                caveats.append(item)
    lines: list[str] = [
        f"## Verdict: {data.get('is_ai_generated')}",
        f"- Platform: {data.get('platform') or 'undetermined'}",
        f"- Confidence: {data.get('confidence')}",
    ]
    if watermarks:
        lines.append("\n### Watermarks / provenance markers")
        lines.extend(f"- {item}" for item in watermarks)
    else:
        lines.append("\nNo watermarks or provenance markers found.")
    if caveats:
        lines.append("\n### Caveats")
        lines.extend(f"- {item}" for item in caveats)
    return "\n".join(lines)


def _remove_visible(
    source: Path,
    output: Path,
    *,
    inpaint: bool,
    inpaint_method: str,
    inpaint_strength: float,
    detect: bool,
    detect_threshold: float,
    strip_metadata: bool,
) -> str:
    from click.testing import CliRunner

    from remove_ai_watermarks.cli import main as cli_main

    args = ["visible", str(source), "--output", str(output), "--inpaint-method", inpaint_method]
    args.append("--inpaint" if inpaint else "--no-inpaint")
    args.append("--detect" if detect else "--no-detect")
    args.extend(["--detect-threshold", str(detect_threshold), "--inpaint-strength", str(inpaint_strength)])
    args.append("--strip-metadata" if strip_metadata else "--keep-metadata")

    result = CliRunner().invoke(cli_main, args, catch_exceptions=False)
    if result.exit_code != 0:
        raise GuiError(result.output.strip() or f"Visible removal failed with exit code {result.exit_code}")
    if not output.exists():
        _copy_to_output(source, output)
    return result.output.strip() or f"Saved: {output.name}"


def _remove_invisible(
    source: Path,
    output: Path,
    *,
    strength: float,
    steps: int,
    pipeline: str,
    device: str,
    seed: int | None,
    hf_token: str | None,
    humanize: float,
    max_resolution: int,
    progress: Any | None = None,
) -> str:
    from remove_ai_watermarks.invisible_engine import is_available as invisible_available

    if not invisible_available():
        raise GuiError(
            "GPU/diffusion dependencies are not installed. Install them with: pip install 'remove-ai-watermarks[gpu]'"
        )

    from remove_ai_watermarks.invisible_engine import InvisibleEngine

    messages: list[str] = []

    def progress_cb(message: str) -> None:
        messages.append(message)
        if progress is not None:
            progress(0.5, desc=message)

    device_str = None if device == "auto" else device
    engine = InvisibleEngine(
        device=device_str,
        pipeline=pipeline,
        hf_token=hf_token or None,
        progress_callback=progress_cb,
    )
    result = engine.remove_watermark(
        image_path=source,
        output_path=output,
        strength=strength,
        num_inference_steps=steps,
        seed=seed,
        humanize=humanize,
        max_resolution=max_resolution,
    )
    return "\n".join([*messages, f"Saved: {result.name}"])


def process_image(
    file_obj: Any,
    mode: str,
    inpaint: bool,
    inpaint_method: str,
    inpaint_strength: float,
    detect: bool,
    detect_threshold: float,
    strip_metadata: bool,
    strength: float,
    steps: int,
    pipeline: str,
    device: str,
    seed: float | None,
    hf_token: str,
    humanize: float,
    max_resolution: int,
    progress: Any | None = None,
) -> tuple[str, str | None, str | None]:
    """Run selected processing mode and return status, preview path, download path."""
    try:
        source = _input_path(file_obj)
        output = _temp_output(source)
        seed_int = None if seed is None or int(seed) < 0 else int(seed)
        t0 = time.monotonic()

        if progress is not None:
            progress(0.05, desc="Starting…")

        if mode == "Visible watermark only":
            status = _remove_visible(
                source,
                output,
                inpaint=inpaint,
                inpaint_method=inpaint_method,
                inpaint_strength=inpaint_strength,
                detect=detect,
                detect_threshold=detect_threshold,
                strip_metadata=strip_metadata,
            )
        elif mode == "Invisible watermark only":
            status = _remove_invisible(
                source,
                output,
                strength=strength,
                steps=steps,
                pipeline=pipeline,
                device=device,
                seed=seed_int,
                hf_token=hf_token,
                humanize=humanize,
                max_resolution=max_resolution,
                progress=progress,
            )
        elif mode == "Metadata only":
            from remove_ai_watermarks.metadata import remove_ai_metadata

            remove_ai_metadata(source, output)
            status = "AI metadata stripped."
        else:
            tmp_visible = _temp_output(source, suffix="_visible")
            status_visible = _remove_visible(
                source,
                tmp_visible,
                inpaint=inpaint,
                inpaint_method=inpaint_method,
                inpaint_strength=inpaint_strength,
                detect=detect,
                detect_threshold=detect_threshold,
                strip_metadata=False,
            )
            try:
                status_invisible = _remove_invisible(
                    tmp_visible,
                    output,
                    strength=strength,
                    steps=steps,
                    pipeline=pipeline,
                    device=device,
                    seed=seed_int,
                    hf_token=hf_token,
                    humanize=humanize,
                    max_resolution=max_resolution,
                    progress=progress,
                )
            except GuiError as exc:
                _copy_to_output(tmp_visible, output)
                status_invisible = f"Invisible step skipped: {exc}"
            from remove_ai_watermarks.metadata import remove_ai_metadata

            remove_ai_metadata(output, output)
            status = "\n\n".join([status_visible, status_invisible, "AI metadata stripped."])

        elapsed = time.monotonic() - t0
        if progress is not None:
            progress(1.0, desc="Done")
        return f"{status}\n\nDone in {elapsed:.1f}s", str(output), str(output)
    except Exception as exc:
        return f"Error: {exc}", None, None


def identify_image(file_obj: Any, no_visible: bool) -> str:
    try:
        from remove_ai_watermarks.identify import identify

        source = _input_path(file_obj)
        report = identify(source, check_visible=not no_visible, check_invisible=not no_visible)
        return _format_identify_report(report)
    except Exception as exc:
        return f"Error: {exc}"


def check_metadata(file_obj: Any) -> str:
    try:
        from remove_ai_watermarks.metadata import get_ai_metadata, has_ai_metadata

        source = _input_path(file_obj)
        has_ai = has_ai_metadata(source)
        meta = get_ai_metadata(source) if has_ai else {}
        prefix = "AI metadata detected." if has_ai else "No AI metadata found."
        details = _format_metadata(meta)
        return f"{prefix}\n\n{details}"
    except Exception as exc:
        return f"Error: {exc}"


def create_app() -> Any:
    """Create the Gradio application."""
    gr = _require_gradio()

    with gr.Blocks(title="Remove-AI-Watermarks GUI") as app:
        gr.Markdown(
            f"# Remove-AI-Watermarks GUI\n"
            f"Version {__version__}. Process images locally from your browser. "
            "Visible watermark and metadata operations are lightweight; invisible removal needs the `[gpu]` extra."
        )
        with gr.Tab("Remove"):
            with gr.Row():
                input_file = gr.File(label="Input image", file_types=["image"], type="filepath")
                output_image = gr.Image(label="Preview", type="filepath")
            mode = gr.Radio(
                ["Visible watermark only", "Invisible watermark only", "Metadata only", "All available"],
                value="Visible watermark only",
                label="Mode",
            )
            with gr.Accordion("Visible watermark options", open=True):
                with gr.Row():
                    inpaint = gr.Checkbox(value=True, label="Inpaint residual artifacts")
                    detect = gr.Checkbox(value=True, label="Skip if no visible watermark is detected")
                    strip_metadata = gr.Checkbox(value=True, label="Strip metadata after visible removal")
                with gr.Row():
                    inpaint_method = gr.Dropdown(["ns", "telea", "gaussian"], value="ns", label="Inpainting method")
                    inpaint_strength = gr.Slider(0.0, 1.0, value=0.85, step=0.05, label="Inpainting strength")
                    detect_threshold = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Detection threshold")
            with gr.Accordion("Invisible watermark options", open=False):
                with gr.Row():
                    strength = gr.Slider(0.0, 1.0, value=0.05, step=0.01, label="Denoising strength")
                    steps = gr.Slider(1, 100, value=50, step=1, label="Steps")
                    humanize = gr.Slider(0.0, 10.0, value=0.0, step=0.5, label="Humanizer grain")
                with gr.Row():
                    pipeline = gr.Dropdown(["default", "ctrlregen"], value="default", label="Pipeline")
                    device = gr.Dropdown(["auto", "cpu", "mps", "cuda"], value="auto", label="Device")
                    seed = gr.Number(value=-1, label="Seed (-1 = random)")
                with gr.Row():
                    max_resolution = gr.Number(value=0, precision=0, label="Max resolution long side (0 = native)")
                    hf_token = gr.Textbox(value="", label="HuggingFace token", type="password")
            run_btn = gr.Button("Process image", variant="primary")
            status = gr.Textbox(label="Status", lines=8)
            download = gr.File(label="Download output")
            run_btn.click(
                process_image,
                inputs=[
                    input_file,
                    mode,
                    inpaint,
                    inpaint_method,
                    inpaint_strength,
                    detect,
                    detect_threshold,
                    strip_metadata,
                    strength,
                    steps,
                    pipeline,
                    device,
                    seed,
                    hf_token,
                    humanize,
                    max_resolution,
                ],
                outputs=[status, output_image, download],
            )

        with gr.Tab("Identify"):
            identify_file = gr.File(label="Input image", file_types=["image"], type="filepath")
            no_visible = gr.Checkbox(value=False, label="Metadata-only / skip pixel detectors")
            identify_btn = gr.Button("Identify provenance")
            identify_result = gr.Markdown(label="Result")
            identify_btn.click(identify_image, inputs=[identify_file, no_visible], outputs=identify_result)

        with gr.Tab("Metadata"):
            metadata_file = gr.File(label="Input image", file_types=["image"], type="filepath")
            check_btn = gr.Button("Check metadata")
            metadata_result = gr.Markdown(label="Metadata")
            check_btn.click(check_metadata, inputs=metadata_file, outputs=metadata_result)

    return app


def main() -> None:
    """Launch the GUI as a standalone script."""
    import click

    @click.command()
    @click.option("--host", default="127.0.0.1", show_default=True, help="Host interface to bind.")
    @click.option("--port", default=7860, show_default=True, type=int, help="Port to listen on.")
    @click.option("--share", is_flag=True, help="Create a temporary public Gradio share link.")
    def _run(host: str, port: int, share: bool) -> None:
        create_app().queue().launch(server_name=host, server_port=port, share=share)

    _run()


if __name__ == "__main__":
    main()
