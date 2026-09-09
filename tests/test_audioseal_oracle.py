"""Tests for the shared AudioSeal oracle's pure helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audioseal_oracle import SAMPLE_RATE, window_bounds  # noqa: E402


class TestWindowBounds:
    def test_exact_multiple_covers_every_sample_once(self) -> None:
        bounds = window_bounds(32_000, 1.0, SAMPLE_RATE)

        assert bounds == [(0, 16_000), (16_000, 32_000)]

    def test_trailing_remainder_keeps_its_own_window(self) -> None:
        bounds = window_bounds(32_500, 1.0, SAMPLE_RATE)

        assert bounds[-1] == (32_000, 32_500)
        assert len(bounds) == 3

    def test_short_signal_is_one_partial_window(self) -> None:
        assert window_bounds(100, 1.0, SAMPLE_RATE) == [(0, 100)]

    def test_empty_signal_has_no_windows(self) -> None:
        assert window_bounds(0, 1.0, SAMPLE_RATE) == []

    @pytest.mark.parametrize(
        ("total", "window_s", "rate"),
        [(-1, 1.0, 16_000), (100, 0.0, 16_000), (100, -1.0, 16_000), (100, 1.0, 0), (100, 0.00001, 16_000)],
    )
    def test_rejects_degenerate_arguments(self, total: int, window_s: float, rate: int) -> None:
        with pytest.raises(ValueError, match=r"total_samples|window_s|sample_rate"):
            window_bounds(total, window_s, rate)
