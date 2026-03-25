"""Tests for the invisible watermark engine (unit tests, no GPU required)."""

from __future__ import annotations

from remove_ai_watermarks.invisible_engine import InvisibleEngine, is_available


class TestIsAvailable:
    """Tests for dependency checking."""

    def test_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)

    def test_available_when_torch_installed(self):
        """torch + diffusers should be installed in dev env."""
        assert is_available() is True


class TestInvisibleEngineInit:
    """Tests for InvisibleEngine construction (no GPU required)."""

    def test_default_model_id(self):
        assert InvisibleEngine.DEFAULT_MODEL_ID == "Lykon/dreamshaper-8"

    def test_ctrlregen_model_id(self):
        assert InvisibleEngine.CTRLREGEN_MODEL_ID == "yepengliu/ctrlregen"
