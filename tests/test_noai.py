"""Tests for vendored noai submodules: constants, extractor, cleaner, c2pa."""

from __future__ import annotations

from remove_ai_watermarks.noai.c2pa import (
    extract_c2pa_chunk,
    extract_c2pa_info,
    has_c2pa_metadata,
)
from remove_ai_watermarks.noai.cleaner import (
    has_ai_content,
)
from remove_ai_watermarks.noai.cleaner import (
    remove_ai_metadata as noai_remove_ai_metadata,
)
from remove_ai_watermarks.noai.constants import (
    AI_KEYWORDS,
    AI_METADATA_KEYS,
    C2PA_CHUNK_TYPE,
    PNG_SIGNATURE,
    SUPPORTED_FORMATS,
)
from remove_ai_watermarks.noai.extractor import (
    extract_ai_metadata,
    extract_metadata,
    get_ai_metadata_summary,
    has_ai_metadata,
)

# ── Constants ───────────────────────────────────────────────────────


class TestConstants:
    """Verify constant integrity."""

    def test_supported_formats_include_png(self):
        assert ".png" in SUPPORTED_FORMATS

    def test_supported_formats_include_jpg(self):
        assert ".jpg" in SUPPORTED_FORMATS

    def test_ai_metadata_keys_not_empty(self):
        assert len(AI_METADATA_KEYS) > 0

    def test_ai_keywords_not_empty(self):
        assert len(AI_KEYWORDS) > 0

    def test_png_signature_bytes(self):
        assert PNG_SIGNATURE == b"\x89PNG\r\n\x1a\n"

    def test_c2pa_chunk_type(self):
        assert C2PA_CHUNK_TYPE == b"caBX"


# ── Extractor ───────────────────────────────────────────────────────


class TestExtractor:
    """Tests for noai.extractor functions."""

    def test_extract_metadata_returns_dict(self, tmp_clean_png):
        meta = extract_metadata(tmp_clean_png)
        assert isinstance(meta, dict)

    def test_extract_metadata_gets_standard_keys(self, tmp_clean_png):
        meta = extract_metadata(tmp_clean_png)
        assert "Author" in meta

    def test_extract_ai_metadata_from_ai_image(self, tmp_png_with_ai_metadata):
        meta = extract_ai_metadata(tmp_png_with_ai_metadata)
        assert "parameters" in meta

    def test_extract_ai_metadata_from_clean_image(self, tmp_clean_png):
        meta = extract_ai_metadata(tmp_clean_png)
        assert len(meta) == 0

    def test_has_ai_metadata_detects(self, tmp_png_with_ai_metadata):
        assert has_ai_metadata(tmp_png_with_ai_metadata)

    def test_has_ai_metadata_clean(self, tmp_clean_png):
        assert not has_ai_metadata(tmp_clean_png)

    def test_summary_with_ai(self, tmp_png_with_ai_metadata):
        summary = get_ai_metadata_summary(tmp_png_with_ai_metadata)
        assert "AI Image Metadata" in summary

    def test_summary_clean(self, tmp_clean_png):
        summary = get_ai_metadata_summary(tmp_clean_png)
        assert "No AI metadata" in summary


# ── Cleaner ─────────────────────────────────────────────────────────


class TestCleaner:
    """Tests for noai.cleaner functions."""

    def test_remove_ai_metadata(self, tmp_png_with_ai_metadata, tmp_path):
        output = tmp_path / "cleaned.png"
        noai_remove_ai_metadata(tmp_png_with_ai_metadata, output)
        assert output.exists()
        # Verify AI metadata removed
        meta = extract_ai_metadata(output)
        assert "parameters" not in meta

    def test_has_ai_content(self, tmp_png_with_ai_metadata):
        assert has_ai_content(tmp_png_with_ai_metadata)


# ── C2PA ────────────────────────────────────────────────────────────


class TestC2PA:
    """Tests for C2PA detection on regular (non-C2PA) images."""

    def test_no_c2pa_on_regular_png(self, tmp_clean_png):
        assert not has_c2pa_metadata(tmp_clean_png)

    def test_no_c2pa_on_jpeg(self, tmp_jpeg_path):
        assert not has_c2pa_metadata(tmp_jpeg_path)

    def test_extract_c2pa_none_on_regular(self, tmp_clean_png):
        assert extract_c2pa_chunk(tmp_clean_png) is None

    def test_extract_c2pa_info_empty(self, tmp_clean_png):
        info = extract_c2pa_info(tmp_clean_png)
        assert info == {}

    def test_c2pa_returns_false_for_non_png(self, tmp_jpeg_path):
        assert not has_c2pa_metadata(tmp_jpeg_path)
