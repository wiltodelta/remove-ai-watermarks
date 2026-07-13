"""Tests for the CLI entry point."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from remove_ai_watermarks.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_png(tmp_path: Path) -> Path:
    """Create a sample PNG for CLI testing."""
    # Seeded: an unseeded random corner can occasionally trip the Doubao
    # visible-mark detector, making `visible --mark auto` flaky.
    img = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    path = tmp_path / "input.png"
    cv2.imwrite(str(path), img)
    return path


def _make_batch_dir(tmp_path: Path, count: int = 3) -> Path:
    """Create a directory with test images for batch testing."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(count):
        img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / f"img_{i}.png"), img)
    return input_dir


def _make_batch_dir_with_metadata(tmp_path: Path, count: int = 3) -> Path:
    """Create a directory with PNG images containing AI metadata."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for i in range(count):
        img = Image.new("RGB", (64, 64), color=(100 + i, 150, 200))
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", f"Steps: 20, Sampler: Euler, img_{i}")
        pnginfo.add_text("prompt", "a test landscape")
        img.save(input_dir / f"img_{i}.png", pnginfo=pnginfo)
    return input_dir


def _mock_invisible_engine():
    """Create a mock InvisibleEngine that writes a copy of the input image."""

    def _mock_remove_watermark(image_path, output_path=None, **kwargs):
        out = output_path or image_path.with_stem(image_path.stem + "_clean")
        out.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(image_path)
        img.save(out)
        return out

    mock_engine = MagicMock()
    mock_engine.remove_watermark.side_effect = _mock_remove_watermark
    mock_cls = MagicMock(return_value=mock_engine)
    return mock_cls, mock_engine


def _mock_invisible_engine_drops_alpha():
    """Mock InvisibleEngine that mimics the real engine's BGR-only output path.

    The real diffusion-based engine reads with cv2.IMREAD_COLOR and writes a
    3-channel result. This mock simulates that so we can regression-test alpha
    preservation across the ``all`` pipeline.
    """

    def _mock_remove_watermark(image_path, output_path=None, **kwargs):
        out = output_path or image_path.with_stem(image_path.stem + "_clean")
        out.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        cv2.imwrite(str(out), bgr)
        return out

    mock_engine = MagicMock()
    mock_engine.remove_watermark.side_effect = _mock_remove_watermark
    mock_cls = MagicMock(return_value=mock_engine)
    return mock_cls, mock_engine


class TestMainGroup:
    """Tests for the top-level CLI group."""

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Remove visible and invisible" in result.output

    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "remove-ai-watermarks" in result.output
        assert "version" in result.output

    def test_no_command_shows_banner(self, runner):
        result = runner.invoke(main, [])
        assert result.exit_code == 0
        assert "Remove-AI-Watermarks" in result.output


class TestVisibleCommand:
    """Tests for the 'visible' subcommand."""

    def test_visible_help(self, runner):
        result = runner.invoke(main, ["visible", "--help"])
        assert result.exit_code == 0
        assert "visible AI watermark" in result.output
        assert "--mark" in result.output

    def test_visible_auto_no_mark_exits_two_with_eraser_hint(self, runner, sample_png, tmp_path):
        # No known visible mark and no AI provenance signal: the command must not
        # re-serve the input as a finished result. It exits EXIT_NO_VISIBLE_MARK
        # (2) -- distinct from success (0) and a hard error (1) -- writes no
        # output file, and points the user at the region eraser.
        output = tmp_path / "clean.png"
        result = runner.invoke(main, ["visible", str(sample_png), "-o", str(output)])
        assert result.exit_code == 2, result.output
        assert not output.exists()
        assert "erase" in result.output
        # The "no signal" branch must NOT imply the image is clean: a missing
        # metadata proxy is not proof an invisible pixel watermark (SynthID) is
        # absent, so the message preserves that uncertainty and routes to 'all'.
        assert "SynthID" in result.output
        assert "all" in result.output

    def test_visible_auto_no_mark_routes_to_all_when_metadata(self, runner, tmp_path):
        # An image whose only signal is an invisible/metadata watermark (here SD
        # generation parameters) has no visible mark to remove; the command must
        # exit 2 and upsell the full 'all' pipeline rather than the eraser.
        img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8))
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", "Steps: 20, Sampler: Euler, a test landscape")
        src = tmp_path / "ai.png"
        img.save(src, pnginfo=pnginfo)
        output = tmp_path / "clean.png"
        result = runner.invoke(main, ["visible", str(src), "-o", str(output)])
        assert result.exit_code == 2, result.output
        assert not output.exists()
        assert "all" in result.output

    def test_visible_basic(self, runner, sample_png, tmp_path):
        output = tmp_path / "clean.png"
        result = runner.invoke(
            main,
            ["visible", str(sample_png), "-o", str(output), "--no-detect"],
        )
        assert result.exit_code == 0
        assert output.exists()
        assert "Saved" in result.output

    def test_visible_default_output_name(self, runner, sample_png):
        result = runner.invoke(main, ["visible", str(sample_png), "--no-detect"])
        assert result.exit_code == 0
        expected = sample_png.with_stem(sample_png.stem + "_clean")
        assert expected.exists()

    def test_visible_backend_cv2(self, runner, sample_png, tmp_path):
        # The old --inpaint/--no-inpaint flags are gone; the fill backend is picked
        # with --backend (localize -> fill). cv2 needs no ONNX model.
        output = tmp_path / "clean.png"
        result = runner.invoke(
            main,
            [
                "visible",
                str(sample_png),
                "-o",
                str(output),
                "--backend",
                "cv2",
                "--no-detect",
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_visible_no_detect(self, runner, sample_png, tmp_path):
        output = tmp_path / "clean.png"
        result = runner.invoke(
            main,
            ["visible", str(sample_png), "-o", str(output), "--no-detect"],
        )
        assert result.exit_code == 0

    def test_visible_nonexistent_file(self, runner):
        result = runner.invoke(main, ["visible", "/nonexistent/file.png"])
        assert result.exit_code != 0

    def test_visible_preserves_rgba_transparency(self, runner, tmp_path):
        """Visible removal on an RGBA PNG must keep the alpha channel,
        not silently flatten the image onto an opaque background.
        """
        rgba = np.zeros((200, 200, 4), dtype=np.uint8)
        rgba[:, :, :3] = 200  # light grey foreground
        rgba[50:150, 50:150, 3] = 255  # opaque square in the middle, rest transparent
        src = tmp_path / "rgba_in.png"
        cv2.imwrite(str(src), rgba)

        output = tmp_path / "rgba_out.png"
        result = runner.invoke(
            main,
            ["visible", str(src), "-o", str(output), "--no-detect"],
        )

        assert result.exit_code == 0, result.output
        assert output.exists()

        out = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
        assert out.ndim == 3, f"output is not 3D: shape={out.shape}"
        assert out.shape[2] == 4, f"output is not RGBA: shape={out.shape}"
        # The transparent corners must remain transparent.
        assert out[0, 0, 3] == 0
        assert out[199, 199, 3] == 0
        # The opaque centre remains opaque (the watermark region default is bottom-right,
        # which doesn't overlap the centre square at 200x200).
        assert out[100, 100, 3] == 255

    def test_visible_keeps_alpha_opaque_in_watermark_region(self, runner, tmp_path):
        """Regression for issue #30 (white box): on an opaque RGBA image, the
        watermark region must stay OPAQUE. Reverse-alpha recovers real pixels
        there, so zeroing alpha would punch a transparent hole that renders as a
        solid white box on any non-transparent viewer.
        """
        rgba = np.full((200, 200, 4), 255, dtype=np.uint8)  # fully opaque white
        src = tmp_path / "rgba_full.png"
        cv2.imwrite(str(src), rgba)

        output = tmp_path / "rgba_kept.png"
        result = runner.invoke(
            main,
            ["visible", str(src), "-o", str(output), "--no-detect"],
        )

        assert result.exit_code == 0, result.output
        out = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
        assert out.shape[2] == 4
        # Default sparkle position is in the bottom-right; alpha there must stay 255.
        from remove_ai_watermarks.gemini_engine import get_watermark_config

        cfg = get_watermark_config(200, 200)
        px, py = cfg.get_position(200, 200)
        size = cfg.logo_size
        assert out[py + size // 2, px + size // 2, 3] == 255, "watermark region alpha was zeroed (white-box regression)"
        # No pixel anywhere should have been forced transparent.
        assert int((out[:, :, 3] == 0).sum()) == 0, "spurious transparent pixels introduced"

    def test_visible_rgb_input_stays_rgb(self, runner, sample_png, tmp_path):
        """Regression: a plain RGB PNG must NOT gain a spurious alpha channel."""
        output = tmp_path / "rgb_out.png"
        result = runner.invoke(
            main,
            ["visible", str(sample_png), "-o", str(output), "--no-detect"],
        )

        assert result.exit_code == 0, result.output
        out = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
        assert out.ndim == 3, f"output is not 3D: shape={out.shape}"
        assert out.shape[2] == 3, f"RGB input produced non-RGB output: shape={out.shape}"


class TestInvisibleCommand:
    """Tests for the 'invisible' subcommand."""

    def test_invisible_help(self, runner):
        result = runner.invoke(main, ["invisible", "--help"])
        assert result.exit_code == 0
        assert "invisible" in result.output.lower()

    def test_invisible_basic(self, runner, sample_png, tmp_path):
        mock_cls, mock_engine = _mock_invisible_engine()
        output = tmp_path / "clean.png"
        with (
            patch("remove_ai_watermarks.invisible_engine.is_available", return_value=True),
            patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True),
            patch("remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls),
        ):
            result = runner.invoke(
                main,
                ["invisible", str(sample_png), "-o", str(output), "--force"],
            )
        assert result.exit_code == 0, result.output
        assert output.exists()
        mock_engine.remove_watermark.assert_called_once()

    def test_invisible_default_output(self, runner, sample_png):
        mock_cls, _mock_engine = _mock_invisible_engine()
        with (
            patch("remove_ai_watermarks.invisible_engine.is_available", return_value=True),
            patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True),
            patch("remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls),
        ):
            result = runner.invoke(main, ["invisible", str(sample_png), "--force"])
        assert result.exit_code == 0, result.output
        expected = sample_png.with_stem(sample_png.stem + "_clean")
        assert expected.exists()

    def test_invisible_adaptive_polish_on_by_default(self, runner, sample_png):
        mock_cls, mock_engine = _mock_invisible_engine()
        with (
            patch("remove_ai_watermarks.invisible_engine.is_available", return_value=True),
            patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True),
            patch("remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls),
        ):
            result = runner.invoke(main, ["invisible", str(sample_png), "--force"])
        assert result.exit_code == 0, result.output
        # adaptive_polish is ON by default (self-gating, so a no-op where not needed).
        assert mock_engine.remove_watermark.call_args.kwargs["adaptive_polish"] is True
        # Default model is None (the SDXL base) and CFG is None (the library's 7.5).
        assert mock_cls.call_args.kwargs["model_id"] is None
        assert mock_engine.remove_watermark.call_args.kwargs["guidance_scale"] is None

    def test_invisible_no_adaptive_polish_disables(self, runner, sample_png):
        mock_cls, mock_engine = _mock_invisible_engine()
        with (
            patch("remove_ai_watermarks.invisible_engine.is_available", return_value=True),
            patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True),
            patch("remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls),
        ):
            result = runner.invoke(main, ["invisible", str(sample_png), "--no-adaptive-polish", "--force"])
        assert result.exit_code == 0, result.output
        assert mock_engine.remove_watermark.call_args.kwargs["adaptive_polish"] is False

    def test_invisible_model_and_guidance_scale_flow_to_engine(self, runner, sample_png):
        mock_cls, mock_engine = _mock_invisible_engine()
        with (
            patch("remove_ai_watermarks.invisible_engine.is_available", return_value=True),
            patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True),
            patch("remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls),
        ):
            result = runner.invoke(
                main,
                ["invisible", str(sample_png), "--model", "org/custom-sdxl", "--guidance-scale", "5.5", "--force"],
            )
        assert result.exit_code == 0, result.output
        assert mock_cls.call_args.kwargs["model_id"] == "org/custom-sdxl"
        assert mock_engine.remove_watermark.call_args.kwargs["guidance_scale"] == 5.5

    def test_pipeline_default_alias_warns_and_maps_to_sdxl(self, runner, sample_png):
        mock_cls, _mock_engine = _mock_invisible_engine()
        with (
            patch("remove_ai_watermarks.invisible_engine.is_available", return_value=True),
            patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True),
            patch("remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls),
        ):
            result = runner.invoke(main, ["invisible", str(sample_png), "--pipeline", "default", "--force"])
        assert result.exit_code == 0, result.output
        # The legacy value warns and is normalized to "sdxl" before the engine is built.
        assert "deprecated" in result.output.lower()
        assert mock_cls.call_args.kwargs["pipeline"] == "sdxl"

    def test_pipeline_sdxl_does_not_warn(self, runner, sample_png):
        mock_cls, _mock_engine = _mock_invisible_engine()
        with (
            patch("remove_ai_watermarks.invisible_engine.is_available", return_value=True),
            patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True),
            patch("remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls),
        ):
            result = runner.invoke(main, ["invisible", str(sample_png), "--pipeline", "sdxl", "--force"])
        assert result.exit_code == 0, result.output
        assert "deprecated" not in result.output.lower()
        assert mock_cls.call_args.kwargs["pipeline"] == "sdxl"

    def test_invisible_nonexistent_file(self, runner):
        result = runner.invoke(main, ["invisible", "/nonexistent/file.png"])
        assert result.exit_code != 0

    def test_invisible_no_signal_skips_and_exits_two(self, runner, sample_png, tmp_path):
        """P0#5: when no invisible AI watermark is locally detectable, the diffusion
        scrub must NOT run (it would only degrade a clean image). Mirrors the visible
        no-mark contract: write no output, exit 2, and DO NOT imply the image is
        clean (a stripped SynthID proxy is not proof of absence)."""
        mock_cls, mock_engine = _mock_invisible_engine()
        output = tmp_path / "clean.png"
        with (
            patch("remove_ai_watermarks.invisible_engine.is_available", return_value=True),
            patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True),
            patch("remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls),
        ):
            result = runner.invoke(main, ["invisible", str(sample_png), "-o", str(output)])
        assert result.exit_code == 2, result.output
        assert not output.exists()
        mock_engine.remove_watermark.assert_not_called()
        assert "--force" in result.output
        assert "SynthID" in result.output  # the message must preserve removal uncertainty

    def test_invisible_force_runs_scrub_on_no_signal(self, runner, sample_png, tmp_path):
        """--force overrides the no-signal skip: the scrub runs regardless."""
        mock_cls, mock_engine = _mock_invisible_engine()
        output = tmp_path / "clean.png"
        with (
            patch("remove_ai_watermarks.invisible_engine.is_available", return_value=True),
            patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True),
            patch("remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls),
        ):
            result = runner.invoke(main, ["invisible", str(sample_png), "-o", str(output), "--force"])
        assert result.exit_code == 0, result.output
        mock_engine.remove_watermark.assert_called_once()

    def test_invisible_runs_without_force_when_signal_present(self, runner, tmp_path):
        """An image carrying an AI metadata signal IS a scrub target, so the run
        proceeds with no --force needed."""
        img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8))
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", "Steps: 20, Sampler: Euler, a test landscape")
        src = tmp_path / "ai.png"
        img.save(src, pnginfo=pnginfo)
        output = tmp_path / "clean.png"
        mock_cls, mock_engine = _mock_invisible_engine()
        with (
            patch("remove_ai_watermarks.invisible_engine.is_available", return_value=True),
            patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True),
            patch("remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls),
        ):
            result = runner.invoke(main, ["invisible", str(src), "-o", str(output)])
        assert result.exit_code == 0, result.output
        mock_engine.remove_watermark.assert_called_once()


class TestAllCommand:
    """Tests for the 'all' subcommand (full pipeline)."""

    def test_all_help(self, runner):
        result = runner.invoke(main, ["all", "--help"])
        assert result.exit_code == 0
        assert "visible" in result.output.lower()

    def test_all_basic(self, runner, sample_png, tmp_path):
        mock_cls, _mock_engine = _mock_invisible_engine()
        output = tmp_path / "clean.png"
        with (
            patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True),
            patch("remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls),
            patch("remove_ai_watermarks.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(
                main,
                ["all", str(sample_png), "-o", str(output), "--force"],
            )
        assert result.exit_code == 0, result.output
        assert output.exists()

    def test_all_nonexistent_file(self, runner):
        result = runner.invoke(main, ["all", "/nonexistent/file.png"])
        assert result.exit_code != 0

    def test_all_visible_step_uses_registry(self, runner, sample_png, tmp_path):
        """Regression (#1): the `all` visible step must route through the registry
        (remove_auto_marks), so Doubao/Jimeng/Samsung/pill marks are handled -- not
        just the Gemini sparkle via a hardcoded GeminiEngine."""
        mock_cls, _mock_engine = _mock_invisible_engine()
        output = tmp_path / "clean.png"

        def _fake_remove_auto(image, **kwargs):
            return image, []

        with (
            patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True),
            patch("remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls),
            patch("remove_ai_watermarks.invisible_engine.is_available", return_value=True),
            patch(
                "remove_ai_watermarks.watermark_registry.remove_auto_marks", side_effect=_fake_remove_auto
            ) as mock_auto,
        ):
            result = runner.invoke(main, ["all", str(sample_png), "-o", str(output), "--force"])
        assert result.exit_code == 0, result.output
        mock_auto.assert_called()  # the registry auto-detector drove the visible pass

    def test_all_skips_invisible_on_no_signal_but_succeeds(self, runner, sample_png, tmp_path):
        """P0#5: with no detectable invisible watermark and no --force, `all` skips
        the destructive step 2 (pixels left intact) but STILL succeeds (exit 0) --
        visible removal + metadata strip ran and a file is written. Distinct from the
        GPU-missing skip, which is a non-zero failure."""
        mock_cls, mock_engine = _mock_invisible_engine()
        output = tmp_path / "clean.png"
        with (
            patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True),
            patch("remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls),
            patch("remove_ai_watermarks.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(main, ["all", str(sample_png), "-o", str(output)])
        assert result.exit_code == 0, result.output
        assert output.exists()
        mock_engine.remove_watermark.assert_not_called()
        assert "Skipped (no invisible" in result.output

    def test_all_loud_warning_and_nonzero_exit_when_gpu_missing(self, runner, sample_png, tmp_path):
        """Regression (#14/#47): when the GPU extra is absent the invisible step is
        skipped, but the output still looks processed -- the run must fail loudly
        (prominent banner + non-zero exit) so a skipped SynthID pass is not mistaken
        for a clean result. The output file is still written (visible + metadata)."""
        output = tmp_path / "clean.png"
        with patch("remove_ai_watermarks.invisible_engine.is_available", return_value=False):
            result = runner.invoke(main, ["all", str(sample_png), "-o", str(output)])
        assert result.exit_code != 0, result.output
        assert "NOT removed" in result.output
        assert "remove-ai-watermarks[gpu]" in result.output
        assert output.exists()  # visible + metadata still produced a file

    def test_all_preserves_rgba_across_invisible_step(self, runner, tmp_path):
        """Regression: ``all`` must keep transparency even when the invisible
        step writes a 3-channel result (as the real diffusion engine does).
        """
        rgba = np.zeros((200, 200, 4), dtype=np.uint8)
        rgba[:, :, :3] = 200
        rgba[50:150, 50:150, 3] = 255  # opaque square; corners transparent
        src = tmp_path / "rgba_in.png"
        cv2.imwrite(str(src), rgba)

        output = tmp_path / "rgba_out.png"
        mock_cls, _engine = _mock_invisible_engine_drops_alpha()
        with (
            patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True),
            patch("remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls),
            patch("remove_ai_watermarks.cli.invisible_available", return_value=True, create=True),
            patch("remove_ai_watermarks.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(main, ["all", str(src), "-o", str(output), "--force"])

        assert result.exit_code == 0, result.output
        out = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
        assert out.ndim == 3, f"output not 3D: shape={out.shape}"
        assert out.shape[2] == 4, f"output is not RGBA: shape={out.shape}"
        assert out[0, 0, 3] == 0
        assert out[100, 100, 3] == 255


class TestMetadataCommand:
    """Tests for the 'metadata' subcommand."""

    def test_metadata_help(self, runner):
        result = runner.invoke(main, ["metadata", "--help"])
        assert result.exit_code == 0

    def test_metadata_check_clean(self, runner, tmp_clean_png):
        result = runner.invoke(main, ["metadata", str(tmp_clean_png), "--check"])
        assert result.exit_code == 0
        assert "No AI metadata" in result.output

    def test_metadata_check_ai(self, runner, tmp_png_with_ai_metadata):
        result = runner.invoke(main, ["metadata", str(tmp_png_with_ai_metadata), "--check"])
        assert result.exit_code == 0
        assert "AI metadata detected" in result.output

    def test_metadata_remove(self, runner, tmp_png_with_ai_metadata, tmp_path):
        output = tmp_path / "stripped.png"
        result = runner.invoke(
            main,
            [
                "metadata",
                str(tmp_png_with_ai_metadata),
                "--remove",
                "-o",
                str(output),
            ],
        )
        assert result.exit_code == 0
        assert "stripped" in result.output

    def test_metadata_remove_in_place(self, runner, tmp_png_with_ai_metadata):
        """With ``-o`` omitted, the strip overwrites the source in place (default
        output_path=None). Previously every test passed an explicit ``-o``."""
        from remove_ai_watermarks.metadata import has_ai_metadata

        assert has_ai_metadata(tmp_png_with_ai_metadata)  # precondition
        result = runner.invoke(main, ["metadata", str(tmp_png_with_ai_metadata), "--remove"])
        assert result.exit_code == 0, result.output
        assert not has_ai_metadata(tmp_png_with_ai_metadata)  # source overwritten, AI metadata gone


class TestIdentifyCommand:
    """Tests for the 'identify' subcommand."""

    def test_identify_help(self, runner):
        result = runner.invoke(main, ["identify", "--help"])
        assert result.exit_code == 0

    def test_identify_clean_png(self, runner, tmp_clean_png):
        result = runner.invoke(main, ["identify", str(tmp_clean_png), "--no-visible"])
        assert result.exit_code == 0
        assert "unknown" in result.output

    def test_identify_unknown_explains_why(self, runner, tmp_clean_png):
        # An unknown verdict must explain itself inline (issue #22: users read a bare
        # "unknown" as the tool being broken) rather than only in the caveats section.
        result = runner.invoke(main, ["identify", str(tmp_clean_png), "--no-visible"])
        assert result.exit_code == 0
        assert "No locally-readable AI signal found" in result.output
        assert "not the same as 'clean'" in result.output

    def test_identify_ai_png_reports_platform(self, runner, tmp_png_with_ai_metadata):
        result = runner.invoke(main, ["identify", str(tmp_png_with_ai_metadata), "--no-visible"])
        assert result.exit_code == 0
        assert "AI-generated" in result.output
        assert "Stable Diffusion" in result.output

    def test_identify_reports_generated_source_kind(self, runner):
        """The C2PA trainedAlgorithmicMedia source type sharpens the verdict to
        'AI-generated (fully synthetic)' at the CLI (the ai_source_kind branch)."""
        from pathlib import Path

        sample = Path(__file__).resolve().parent.parent / "data" / "samples" / "chatgpt-1.png"
        if not sample.exists():
            pytest.skip("chatgpt sample not present")
        result = runner.invoke(main, ["identify", str(sample), "--no-visible"])
        assert result.exit_code == 0
        assert "AI-generated (fully synthetic)" in result.output

    def test_identify_json_is_valid(self, runner, tmp_png_with_ai_metadata):
        result = runner.invoke(main, ["identify", str(tmp_png_with_ai_metadata), "--no-visible", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["is_ai_generated"] is True
        assert payload["confidence"] == "high"

    def test_identify_nonexistent_file(self, runner):
        result = runner.invoke(main, ["identify", "/nonexistent/file.png"])
        assert result.exit_code != 0


class TestBatchCommand:
    """Tests for the 'batch' subcommand."""

    def test_batch_help(self, runner):
        result = runner.invoke(main, ["batch", "--help"])
        assert result.exit_code == 0

    def test_batch_empty_dir(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = runner.invoke(main, ["batch", str(empty_dir)])
        assert result.exit_code == 0
        assert "No supported images" in result.output

    def test_batch_visible_mode(self, runner, tmp_path):
        input_dir = _make_batch_dir(tmp_path)
        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            ["batch", str(input_dir), "-o", str(output_dir), "--mode", "visible"],
        )
        assert result.exit_code == 0
        assert "3 processed" in result.output
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.png"))) == 3

    def test_batch_metadata_mode(self, runner, tmp_path):
        input_dir = _make_batch_dir_with_metadata(tmp_path)
        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            ["batch", str(input_dir), "-o", str(output_dir), "--mode", "metadata"],
        )
        assert result.exit_code == 0
        assert "3 processed" in result.output
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.png"))) == 3
        # Verify AI metadata was stripped
        for out_img in output_dir.glob("*.png"):
            with Image.open(out_img) as img:
                assert "parameters" not in img.info

    def test_batch_invisible_mode(self, runner, tmp_path):
        input_dir = _make_batch_dir(tmp_path)
        output_dir = tmp_path / "output"
        mock_cls, _mock_engine = _mock_invisible_engine()
        with (
            patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True),
            patch("remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls),
            patch("remove_ai_watermarks.cli.invisible_available", return_value=True, create=True),
            patch("remove_ai_watermarks.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(
                main,
                ["batch", str(input_dir), "-o", str(output_dir), "--mode", "invisible", "--force"],
            )
        assert result.exit_code == 0, result.output
        assert "3 processed" in result.output

    def test_batch_invisible_skips_no_signal_and_copies_through(self, runner, tmp_path):
        """P0#5: batch invisible mode skips the scrub on signal-less images (no
        --force) and copies the input through, so the output dir is complete with the
        pixels left intact and the engine never called."""
        input_dir = _make_batch_dir(tmp_path)
        output_dir = tmp_path / "output"
        mock_cls, mock_engine = _mock_invisible_engine()
        with (
            patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True),
            patch("remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls),
            patch("remove_ai_watermarks.cli.invisible_available", return_value=True, create=True),
            patch("remove_ai_watermarks.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(
                main,
                ["batch", str(input_dir), "-o", str(output_dir), "--mode", "invisible"],
            )
        assert result.exit_code == 0, result.output
        assert "3 processed" in result.output
        assert len(list(output_dir.glob("*.png"))) == 3  # inputs copied through
        mock_engine.remove_watermark.assert_not_called()

    def test_batch_all_mode(self, runner, tmp_path):
        input_dir = _make_batch_dir(tmp_path)
        output_dir = tmp_path / "output"
        mock_cls, _mock_engine = _mock_invisible_engine()
        with (
            patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True),
            patch("remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls),
            patch("remove_ai_watermarks.cli.invisible_available", return_value=True, create=True),
            patch("remove_ai_watermarks.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(
                main,
                ["batch", str(input_dir), "-o", str(output_dir), "--mode", "all", "--force"],
            )
        assert result.exit_code == 0, result.output
        assert "3 processed" in result.output

    def test_batch_all_mode_preserves_rgba(self, runner, tmp_path):
        """Regression: batch ``all`` must keep transparency across the
        alpha-dropping invisible step (mirrors test_all_preserves_rgba_...).
        """
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        rgba = np.zeros((200, 200, 4), dtype=np.uint8)
        rgba[:, :, :3] = 200
        rgba[50:150, 50:150, 3] = 255
        cv2.imwrite(str(input_dir / "rgba.png"), rgba)

        output_dir = tmp_path / "output"
        mock_cls, _engine = _mock_invisible_engine_drops_alpha()
        with (
            patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True),
            patch("remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls),
            patch("remove_ai_watermarks.cli.invisible_available", return_value=True, create=True),
            patch("remove_ai_watermarks.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(
                main,
                ["batch", str(input_dir), "-o", str(output_dir), "--mode", "all", "--force"],
            )
        assert result.exit_code == 0, result.output

        out = cv2.imread(str(output_dir / "rgba.png"), cv2.IMREAD_UNCHANGED)
        assert out.ndim == 3, f"output not 3D: shape={out.shape}"
        assert out.shape[2] == 4, f"output is not RGBA: shape={out.shape}"
        assert out[0, 0, 3] == 0
        assert out[100, 100, 3] == 255

    def test_batch_auto_is_deprecated_and_enables_polish(self, runner, tmp_path):
        """--auto is retired: it warns and just enables the adaptive polish (the
        pipeline is always the default controlnet now)."""
        input_dir = _make_batch_dir(tmp_path, count=2)
        output_dir = tmp_path / "output"
        mock_cls, mock_engine = _mock_invisible_engine()
        with (
            patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True),
            patch("remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls),
            patch("remove_ai_watermarks.cli.invisible_available", return_value=True, create=True),
            patch("remove_ai_watermarks.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(
                main,
                ["batch", str(input_dir), "-o", str(output_dir), "--mode", "invisible", "--auto", "--force"],
            )
        assert result.exit_code == 0, result.output
        assert "2 processed" in result.output
        assert "deprecated" in result.output.lower()
        # Pipeline stays the default controlnet; --auto only turned the polish on.
        assert mock_cls.call_args.kwargs["pipeline"] == "controlnet"
        assert mock_engine.remove_watermark.call_args.kwargs["adaptive_polish"] is True

    def test_batch_default_output_dir(self, runner, tmp_path):
        input_dir = _make_batch_dir(tmp_path)
        result = runner.invoke(
            main,
            ["batch", str(input_dir), "--mode", "visible"],
        )
        assert result.exit_code == 0
        expected_dir = tmp_path / "input_clean"
        assert expected_dir.exists()

    def test_batch_errors_exit_nonzero(self, runner, tmp_path):
        """Regression: batch used to always exit 0 even when every image errored,
        hiding failure from a wrapping service. A corrupt image must yield a non-zero
        exit and an error count."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "corrupt.png").write_bytes(b"this is not a PNG at all" * 50)
        result = runner.invoke(main, ["batch", str(input_dir), "--mode", "visible"])
        assert result.exit_code != 0, result.output
        assert "error" in result.output.lower()

    def test_batch_invisible_gpu_missing_writes_output_and_exits_nonzero(self, runner, tmp_path):
        """Regression: batch --mode invisible with a signal-bearing image but no GPU
        deps used to write NO output for that image and still exit 0, silently dropping
        the files that most needed processing. It must now copy the input through (so the
        output dir is complete), warn about the retained SynthID watermark, and exit
        non-zero -- mirroring the single ``all`` command."""
        input_dir = _make_batch_dir_with_metadata(tmp_path, count=3)  # SD params = invisible signal
        output_dir = tmp_path / "output"
        with patch("remove_ai_watermarks.invisible_engine.is_available", return_value=False):
            result = runner.invoke(
                main,
                ["batch", str(input_dir), "-o", str(output_dir), "--mode", "invisible"],
            )
        assert result.exit_code != 0, result.output
        assert "NOT removed" in result.output
        assert len(list(output_dir.glob("*.png"))) == 3  # every input copied through, none dropped


class TestGpuHintMarkup:
    """The GPU-extra install hint must reach the user with the ``[gpu]`` token
    intact (plain output prints it verbatim, with no markup parsing)."""

    def test_invisible_install_hint_keeps_gpu_extra(self, runner, sample_png):
        with patch("remove_ai_watermarks.invisible_engine.is_available", return_value=False):
            result = runner.invoke(main, ["invisible", str(sample_png)])
        assert result.exit_code != 0
        assert "remove-ai-watermarks[gpu]" in result.output

    def test_all_install_hint_keeps_gpu_extra(self, runner, sample_png):
        # The `all` pipeline skips the invisible step with a warning that carries
        # the same hint; it must keep the [gpu] extra too.
        with patch("remove_ai_watermarks.invisible_engine.is_available", return_value=False):
            result = runner.invoke(main, ["all", str(sample_png)])
        assert "remove-ai-watermarks[gpu]" in result.output


class TestEraseCommand:
    """Tests for the 'erase' universal region eraser subcommand."""

    def test_erase_help(self, runner):
        result = runner.invoke(main, ["erase", "--help"])
        assert result.exit_code == 0
        assert "--region" in result.output
        assert "--backend" in result.output

    def test_erase_single_region(self, runner, sample_png, tmp_path):
        output = tmp_path / "erased.png"
        result = runner.invoke(
            main,
            ["erase", str(sample_png), "--region", "10,10,40,40", "-o", str(output)],
        )
        assert result.exit_code == 0, result.output
        assert output.exists()

    def test_erase_two_regions(self, runner, sample_png, tmp_path):
        output = tmp_path / "erased2.png"
        result = runner.invoke(
            main,
            [
                "erase",
                str(sample_png),
                "--region",
                "10,10,30,30",
                "--region",
                "120,120,30,30",
                "-o",
                str(output),
            ],
        )
        assert result.exit_code == 0, result.output
        assert output.exists()
        # The banner reports the region count it processed.
        assert "2 region(s)" in result.output

    def test_erase_default_output_name(self, runner, sample_png):
        result = runner.invoke(main, ["erase", str(sample_png), "--region", "10,10,40,40"])
        assert result.exit_code == 0, result.output
        assert sample_png.with_stem(sample_png.stem + "_clean").exists()

    def test_erase_malformed_region_exits_nonzero(self, runner, sample_png, tmp_path):
        output = tmp_path / "x.png"
        # Only three values: click.BadParameter -> non-zero exit, no output file.
        result = runner.invoke(
            main,
            ["erase", str(sample_png), "--region", "1,2,3", "-o", str(output)],
        )
        assert result.exit_code != 0
        assert not output.exists()

    def test_erase_nonexistent_file(self, runner):
        result = runner.invoke(main, ["erase", "/nonexistent/file.png", "--region", "0,0,10,10"])
        assert result.exit_code != 0

    def test_erase_lama_backend_without_onnxruntime(self, runner, sample_png, tmp_path):
        # The LaMa backend needs onnxruntime; without it the CLI must surface a
        # clear error and exit non-zero rather than crash. When onnxruntime IS
        # installed there is no missing-dep path to exercise, so skip.
        from remove_ai_watermarks.region_eraser import lama_available

        if lama_available():
            pytest.skip("onnxruntime installed; missing-dep error path not reachable")
        output = tmp_path / "y.png"
        result = runner.invoke(
            main,
            ["erase", str(sample_png), "--region", "10,10,40,40", "--backend", "lama", "-o", str(output)],
        )
        assert result.exit_code != 0
        assert "onnxruntime" in result.output.lower()
        assert not output.exists()


def test_visible_backend_runtime_error_exits_cleanly(runner, tmp_path, monkeypatch):
    # A backend whose extra is missing raises RuntimeError in region_eraser.erase;
    # the visible --mark auto path must surface it cleanly, not as a raw traceback (#9).
    from pathlib import Path

    from remove_ai_watermarks import region_eraser

    doubao = Path(__file__).resolve().parent.parent / "data" / "samples" / "doubao-1.png"
    if not doubao.exists():
        pytest.skip("doubao sample not present")

    def boom(*_a, **_k):
        raise RuntimeError("MI-GAN backend requires onnxruntime. Install the extra: ...")

    monkeypatch.setattr(region_eraser, "erase", boom)
    out = tmp_path / "out.png"
    result = runner.invoke(main, ["visible", str(doubao), "-o", str(out), "--backend", "migan"])
    assert result.exit_code == 1
    assert not isinstance(result.exception, RuntimeError), "RuntimeError leaked as a traceback"


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("empty.png", b""),
        ("notimage.jpg", b"plain text, not an image at all " * 20),
        ("truncated.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 40),
    ],
)
@pytest.mark.parametrize("cmd", [["metadata", "--remove"], ["visible", "--backend", "cv2"]])
def test_unreadable_input_exits_cleanly(runner, tmp_path, name, content, cmd):
    """Regression: a corrupt / empty / non-image file (real prod uploads include ~0.2%
    truncated files) must NEVER leak a raw PIL/OSError/ValueError traceback. `metadata
    --remove` is fail-safe -- an undecodable file is copied through unchanged (exit 0),
    a strip that cannot parse the file is a no-op, not a crash; `visible` must decode to
    remove a mark, so it is a clean error (exit 1). Found by the runtime mode fuzz."""
    bad = tmp_path / name
    bad.write_bytes(content)
    out = tmp_path / "out.png"
    result = runner.invoke(main, [cmd[0], str(bad), "-o", str(out), *cmd[1:]])
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"leaked a raw traceback: {result.exception!r}"
    )
    if cmd[0] == "metadata":
        assert result.exit_code == 0, result.output  # fail-safe copy-through
        assert out.read_bytes() == content  # input passed through unchanged
    else:
        assert result.exit_code == 1, result.output  # cannot remove a mark from unreadable input
        assert "Error" in result.output
