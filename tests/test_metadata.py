"""Tests for AI metadata detection and removal."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from remove_ai_watermarks.metadata import (
    _is_ai_key,
    get_ai_metadata,
    has_ai_metadata,
    remove_ai_metadata,
)

# ── Key detection ───────────────────────────────────────────────────


class TestIsAiKey:
    """Tests for _is_ai_key helper."""

    def test_exact_match_lowercase(self):
        assert _is_ai_key("parameters")

    def test_exact_match_mixed_case(self):
        assert _is_ai_key("Parameters")

    def test_keyword_substring(self):
        assert _is_ai_key("stable_diffusion_model_v2")

    def test_c2pa_detected(self):
        assert _is_ai_key("c2pa_chunk")

    def test_standard_key_not_flagged(self):
        assert not _is_ai_key("Author")

    def test_innocuous_key_not_flagged(self):
        assert not _is_ai_key("Title")

    def test_dpi_not_flagged(self):
        assert not _is_ai_key("dpi")


# ── has_ai_metadata / get_ai_metadata ───────────────────────────────


class TestHasAiMetadata:
    """Tests for detecting AI metadata in images."""

    def test_detects_ai_metadata(self, tmp_png_with_ai_metadata):
        assert has_ai_metadata(tmp_png_with_ai_metadata)

    def test_clean_image_no_ai(self, tmp_clean_png):
        assert not has_ai_metadata(tmp_clean_png)


class TestGetAiMetadata:
    """Tests for extracting AI metadata."""

    def test_extracts_parameters_key(self, tmp_png_with_ai_metadata):
        meta = get_ai_metadata(tmp_png_with_ai_metadata)
        assert "parameters" in meta
        assert "Euler" in meta["parameters"]

    def test_extracts_prompt_key(self, tmp_png_with_ai_metadata):
        meta = get_ai_metadata(tmp_png_with_ai_metadata)
        assert "prompt" in meta

    def test_does_not_extract_author(self, tmp_png_with_ai_metadata):
        meta = get_ai_metadata(tmp_png_with_ai_metadata)
        assert "Author" not in meta

    def test_clean_image_empty_dict(self, tmp_clean_png):
        meta = get_ai_metadata(tmp_clean_png)
        assert meta == {}


# ── remove_ai_metadata ──────────────────────────────────────────────


class TestRemoveAiMetadata:
    """Tests for stripping AI metadata."""

    def test_removes_ai_keys(self, tmp_png_with_ai_metadata):
        output = tmp_png_with_ai_metadata.parent / "cleaned.png"
        remove_ai_metadata(tmp_png_with_ai_metadata, output)

        with Image.open(output) as img:
            assert "parameters" not in img.info
            assert "prompt" not in img.info

    def test_keeps_standard_metadata(self, tmp_png_with_ai_metadata):
        output = tmp_png_with_ai_metadata.parent / "cleaned.png"
        remove_ai_metadata(tmp_png_with_ai_metadata, output, keep_standard=True)

        with Image.open(output) as img:
            assert "Author" in img.info
            assert img.info["Author"] == "Test Author"

    def test_remove_all_metadata(self, tmp_png_with_ai_metadata):
        output = tmp_png_with_ai_metadata.parent / "cleaned.png"
        remove_ai_metadata(tmp_png_with_ai_metadata, output, keep_standard=False)
        with Image.open(output) as img:
            assert "Author" not in img.info
            assert "parameters" not in img.info

    def test_overwrite_in_place(self, tmp_path):
        """When output_path is None, should overwrite source."""
        img = Image.new("RGB", (32, 32))
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", "test data")
        path = tmp_path / "inplace.png"
        img.save(path, pnginfo=pnginfo)

        result = remove_ai_metadata(path)
        assert result == path

        with Image.open(path) as cleaned:
            assert "parameters" not in cleaned.info

    def test_jpeg_output(self, tmp_path):
        """Test metadata removal for JPEG format."""
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", "test")
        png_path = tmp_path / "source.png"
        img.save(png_path, pnginfo=pnginfo)

        jpg_path = tmp_path / "output.jpg"
        result = remove_ai_metadata(png_path, jpg_path)
        assert result == jpg_path
        assert jpg_path.exists()

    def test_creates_parent_directories(self, tmp_path):
        img = Image.new("RGB", (32, 32))
        pnginfo = PngInfo()
        pnginfo.add_text("prompt", "test")
        path = tmp_path / "source.png"
        img.save(path, pnginfo=pnginfo)

        output = tmp_path / "sub" / "dir" / "cleaned.png"
        remove_ai_metadata(path, output)
        assert output.exists()

    def test_returns_path(self, tmp_clean_png):
        output = tmp_clean_png.parent / "out.png"
        result = remove_ai_metadata(tmp_clean_png, output)
        assert isinstance(result, Path)
        assert result == output
