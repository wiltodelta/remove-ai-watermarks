"""Tests for the CLI entry point."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from remove_ai_watermarks.cli import main


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def sample_png(tmp_path: Path) -> Path:
    """Create a sample PNG for CLI testing."""
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    path = tmp_path / "input.png"
    cv2.imwrite(str(path), img)
    return path


def _make_batch_dir(tmp_path: Path, count: int = 3) -> Path:
    """Create a directory with test images for batch testing."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for i in range(count):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
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


class TestMainGroup:
    """Tests for the top-level CLI group."""

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Remove visible and invisible" in result.output

    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.2.0" in result.output

    def test_no_command_shows_banner(self, runner):
        result = runner.invoke(main, [])
        assert result.exit_code == 0
        assert "Remove-AI-Watermarks" in result.output


class TestVisibleCommand:
    """Tests for the 'visible' subcommand."""

    def test_visible_help(self, runner):
        result = runner.invoke(main, ["visible", "--help"])
        assert result.exit_code == 0
        assert "Gemini watermark" in result.output

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


class TestInvisibleCommand:
    """Tests for the 'invisible' subcommand."""

    def test_invisible_help(self, runner):
        result = runner.invoke(main, ["invisible", "--help"])
        assert result.exit_code == 0
        assert "invisible" in result.output.lower()

    def test_invisible_basic(self, runner, sample_png, tmp_path):
        mock_cls, mock_engine = _mock_invisible_engine()
        output = tmp_path / "clean.png"
        with patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True), patch(
            "remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls
        ):
            result = runner.invoke(
                main,
                ["invisible", str(sample_png), "-o", str(output)],
            )
        assert result.exit_code == 0, result.output
        assert output.exists()
        mock_engine.remove_watermark.assert_called_once()

    def test_invisible_default_output(self, runner, sample_png):
        mock_cls, mock_engine = _mock_invisible_engine()
        with patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True), patch(
            "remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls
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
        mock_cls, mock_engine = _mock_invisible_engine()
        output = tmp_path / "clean.png"
        with patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True), patch(
            "remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls
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
        mock_cls, mock_engine = _mock_invisible_engine()
        with patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True), patch(
            "remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls
        ), patch("remove_ai_watermarks.cli.invisible_available", return_value=True, create=True), patch(
            "remove_ai_watermarks.invisible_engine.is_available", return_value=True
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
        mock_cls, mock_engine = _mock_invisible_engine()
        with patch("remove_ai_watermarks.cli.InvisibleEngine", mock_cls, create=True), patch(
            "remove_ai_watermarks.invisible_engine.InvisibleEngine", mock_cls
        ), patch("remove_ai_watermarks.cli.invisible_available", return_value=True, create=True), patch(
            "remove_ai_watermarks.invisible_engine.is_available", return_value=True
        ):
            result = runner.invoke(
                main,
                ["batch", str(input_dir), "-o", str(output_dir), "--mode", "all"],
            )
        assert result.exit_code == 0, result.output
        assert "3 processed" in result.output

    def test_batch_default_output_dir(self, runner, tmp_path):
        input_dir = _make_batch_dir(tmp_path)
        result = runner.invoke(
            main,
            ["batch", str(input_dir), "--mode", "visible"],
        )
        assert result.exit_code == 0
        expected_dir = tmp_path / "input_clean"
        assert expected_dir.exists()

