"""Tests for the pinned VideoSeal TorchScript oracle's pure helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import videoseal_oracle  # noqa: E402


class TestMessage:
    def test_message_is_deterministic_and_256_bits(self) -> None:
        first = videoseal_oracle.message_bits()
        second = videoseal_oracle.message_bits()

        assert first == second
        assert len(first) == 256
        assert set(first) <= {0, 1}

    def test_seed_changes_the_message(self) -> None:
        assert videoseal_oracle.message_bits(seed=8) != videoseal_oracle.message_bits(seed=7)


class TestLabel:
    def test_label_is_hex_of_the_decoded_bits(self) -> None:
        bits = tuple([1] * 8 + [0] * 8)
        reading = videoseal_oracle.VideoSealReading(
            bit_accuracy=1.0,
            decoded_bits=bits,
            per_frame_bit_accuracy=(1.0,),
        )

        assert reading.label == "ff00"

    def test_detection_rule_is_the_documented_threshold(self) -> None:
        below = videoseal_oracle.VideoSealReading(0.89, (0,), ())
        exact = videoseal_oracle.VideoSealReading(0.9, (0,), ())

        assert below.detected is False
        assert exact.detected is True


class TestPins:
    def test_checkpoint_pin_is_a_sha256_digest(self) -> None:
        digest = videoseal_oracle.JIT_SHA256

        assert len(digest) == 64
        assert all(char in "0123456789abcdef" for char in digest)

    def test_source_commit_is_pinned(self) -> None:
        assert len(videoseal_oracle.SOURCE_COMMIT) == 40


class TestCacheOverride:
    def test_cache_dir_env_override_is_honored(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        original_cache = videoseal_oracle.CACHE_DIR
        import importlib

        try:
            with monkeypatch.context() as environment:
                environment.setenv("VIDEOSEAL_CACHE_DIR", str(tmp_path / "cache"))
                importlib.reload(videoseal_oracle)
                assert tmp_path / "cache" == videoseal_oracle.CACHE_DIR
        finally:
            importlib.reload(videoseal_oracle)
        assert original_cache == videoseal_oracle.CACHE_DIR


class TestAggregationFormulas:
    def test_aggregations_match_the_upstream_formulas(self) -> None:
        torch = pytest.importorskip("torch")
        # Two frames, four bits; hand-computed from the upstream extract_message
        # formulas so a silent mutation of the port fails here.
        bit_preds = torch.tensor([[2.0, -1.0, 0.5, -3.0], [0.1, 0.2, -0.4, 0.3]])

        avg = videoseal_oracle._aggregate_bit_preds(bit_preds, "avg")
        squared = videoseal_oracle._aggregate_bit_preds(bit_preds, "squared_avg")
        l1 = videoseal_oracle._aggregate_bit_preds(bit_preds, "l1norm_avg")
        l2 = videoseal_oracle._aggregate_bit_preds(bit_preds, "l2norm_avg")

        torch.testing.assert_close(avg, bit_preds.mean(dim=0))
        torch.testing.assert_close(squared, (bit_preds * bit_preds.abs()).mean(dim=0))
        w1 = torch.norm(bit_preds, p=1, dim=1).unsqueeze(1)
        torch.testing.assert_close(l1, (bit_preds * w1).mean(dim=0))
        w2 = torch.norm(bit_preds, p=2, dim=1).unsqueeze(1)
        torch.testing.assert_close(l2, (bit_preds * w2).mean(dim=0))

    def test_unknown_aggregation_is_rejected(self) -> None:
        torch = pytest.importorskip("torch")

        with pytest.raises(ValueError, match="unknown aggregation"):
            videoseal_oracle._aggregate_bit_preds(torch.zeros(2, 4), "median")
