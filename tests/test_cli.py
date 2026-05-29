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

    def test_visible_no_inpaint(self, runner, sample_png, tmp_path):
        output = tmp_path / "clean.png"
        result = runner.invoke(
            main,
            [
                "visible",
                str(sample_png),
                "-o",
                str(output),
                "--no-inpaint",
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

    def test_visible_clears_alpha_in_watermark_region(self, runner, tmp_path):
        """When inpainting an RGBA image, the watermark region must be cleared
        in the alpha channel so the sparkle area becomes transparent, not opaque-black.
        """
        rgba = np.full((200, 200, 4), 255, dtype=np.uint8)  # fully opaque white
        src = tmp_path / "rgba_full.png"
        cv2.imwrite(str(src), rgba)

        output = tmp_path / "rgba_cleared.png"
        result = runner.invoke(
            main,
            ["visible", str(src), "-o", str(output), "--no-detect"],
        )

        assert result.exit_code == 0, result.output
        out = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
        assert out.shape[2] == 4
        # Default sparkle position is in the bottom-right; alpha there must be 0.
        from remove_ai_watermarks.gemini_engine import get_watermark_config

        cfg = get_watermark_config(200, 200)
        px, py = cfg.get_position(200, 200)
        size = cfg.logo_size
        assert out[py + size // 2, px + size // 2, 3] == 0, "alpha in the watermark region was not cleared"

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
                ["invisible", str(sample_png), "-o", str(output)],
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
            result = runner.invoke(main, ["invisible", str(sample_png)])
        assert result.exit_code == 0, result.output
        expected = sample_png.with_stem(sample_png.stem + "_clean")
        assert expected.exists()

    def test_invisible_nonexistent_file(self, runner):
        result = runner.invoke(main, ["invisible", "/nonexistent/file.png"])
        assert result.exit_code != 0


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
        ):
            result = runner.invoke(
                main,
                ["all", str(sample_png), "-o", str(output)],
            )
        assert result.exit_code == 0, result.output
        assert output.exists()

    def test_all_nonexistent_file(self, runner):
        result = runner.invoke(main, ["all", "/nonexistent/file.png"])
        assert result.exit_code != 0

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
            result = runner.invoke(main, ["all", str(src), "-o", str(output)])

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
                ["batch", str(input_dir), "-o", str(output_dir), "--mode", "invisible"],
            )
        assert result.exit_code == 0, result.output
        assert "3 processed" in result.output

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
                ["batch", str(input_dir), "-o", str(output_dir), "--mode", "all"],
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
                ["batch", str(input_dir), "-o", str(output_dir), "--mode", "all"],
            )
        assert result.exit_code == 0, result.output

        out = cv2.imread(str(output_dir / "rgba.png"), cv2.IMREAD_UNCHANGED)
        assert out.ndim == 3, f"output not 3D: shape={out.shape}"
        assert out.shape[2] == 4, f"output is not RGBA: shape={out.shape}"
        assert out[0, 0, 3] == 0
        assert out[100, 100, 3] == 255

    def test_batch_default_output_dir(self, runner, tmp_path):
        input_dir = _make_batch_dir(tmp_path)
        result = runner.invoke(
            main,
            ["batch", str(input_dir), "--mode", "visible"],
        )
        assert result.exit_code == 0
        expected_dir = tmp_path / "input_clean"
        assert expected_dir.exists()


class TestGpuHintMarkup:
    """The GPU-extra install hint must survive rich markup (the ``[gpu]`` token
    is otherwise parsed as a style tag and silently dropped)."""

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
