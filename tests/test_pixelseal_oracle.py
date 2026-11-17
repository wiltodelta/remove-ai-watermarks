"""Tests for the pinned PixelSeal oracle's pure helpers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pixelseal_oracle  # noqa: E402


class TestMessage:
    def test_message_is_deterministic_and_256_bits(self) -> None:
        first = pixelseal_oracle.message_bits()
        second = pixelseal_oracle.message_bits()

        assert first == second
        assert len(first) == 256
        assert set(first) <= {0, 1}

    def test_seed_changes_the_message(self) -> None:
        assert pixelseal_oracle.message_bits(seed=8) != pixelseal_oracle.message_bits(seed=7)


class TestReading:
    def test_label_is_hex_of_decoded_bits(self) -> None:
        reading = pixelseal_oracle.PixelSealReading(1.0, tuple([1] * 8 + [0] * 8))

        assert reading.label == "ff00"

    def test_detection_rule_is_the_documented_adapter_threshold(self) -> None:
        assert pixelseal_oracle.PixelSealReading(0.899, ()).detected is False
        assert pixelseal_oracle.PixelSealReading(0.9, ()).detected is True


class TestPins:
    @pytest.mark.parametrize(
        "value",
        [pixelseal_oracle.CHECKPOINT_SHA256, pixelseal_oracle.MODEL_CARD_SHA256],
    )
    def test_digests_are_sha256(self, value: str) -> None:
        assert len(value) == 64
        assert all(char in "0123456789abcdef" for char in value)

    def test_source_commit_is_pinned(self) -> None:
        assert len(pixelseal_oracle.SOURCE_COMMIT) == 40


def test_cache_dir_env_override_is_honored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original_cache = pixelseal_oracle.CACHE_DIR
    try:
        with monkeypatch.context() as environment:
            environment.setenv("PIXELSEAL_CACHE_DIR", str(tmp_path / "cache"))
            importlib.reload(pixelseal_oracle)
            assert tmp_path / "cache" == pixelseal_oracle.CACHE_DIR
    finally:
        importlib.reload(pixelseal_oracle)
    assert original_cache == pixelseal_oracle.CACHE_DIR


def test_video_embed_preserves_frame_order_and_uses_video_mode() -> None:
    torch = pytest.importorskip("torch")
    pixels = np.arange(3 * 4 * 5 * 3, dtype=np.uint8).reshape(3, 4, 5, 3)
    message = pixelseal_oracle.message_bits()

    class Model:
        def embed(self, tensor, *, msgs, is_video):
            assert is_video is True
            assert tensor.shape == (3, 3, 4, 5)
            assert tensor.dtype == torch.float32
            assert msgs.tolist() == [list(message)]
            return {"imgs_w": tensor + 0.01}

    marked = pixelseal_oracle.embed_video(Model(), pixels, message)
    np.testing.assert_allclose(marked, pixels.astype(np.float32) / 255.0 + 0.01, atol=1e-7)


def test_video_read_averages_raw_logits_before_thresholding() -> None:
    torch = pytest.importorskip("torch")
    message = (0, 1, *pixelseal_oracle.message_bits()[2:])
    logits = torch.tensor(message, dtype=torch.float32).mul(2).sub(1).repeat(3, 1)
    logits[:, 0] = torch.tensor([-4.0, 1.0, 1.0])
    logits[:, 1] = torch.tensor([4.0, -1.0, -1.0])

    class Model:
        def detect(self, tensor, *, is_video):
            assert is_video is True
            assert tensor.shape == (3, 3, 4, 5)
            assert float(tensor.max()) == pytest.approx(128 / 255)
            return {"preds": torch.cat((torch.zeros(3, 1), logits), dim=1)}

    reading = pixelseal_oracle.read_video(Model(), np.full((3, 4, 5, 3), 128, dtype=np.uint8), message=message)
    assert reading.decoded_bits == message
    assert reading.bit_accuracy == 1.0
    assert reading.detected is True
    assert reading.per_frame_bit_accuracy == (1.0, 254 / 256, 254 / 256)
    from videoseal_oracle import read_aggregation_matrix

    matrix = read_aggregation_matrix(Model(), np.full((3, 4, 5, 3), 128 / 255, dtype=np.float32), message=message)
    assert matrix["avg"] == 1.0


@pytest.mark.parametrize(
    "frames",
    [
        np.zeros((4, 5, 3)),
        np.zeros((0, 4, 5, 3)),
        np.zeros((2, 0, 5, 3)),
        np.zeros((2, 4, 5, 1)),
        np.full((2, 4, 5, 3), np.nan),
        np.full((2, 4, 5, 3), 2.0),
    ],
)
def test_video_oracle_rejects_invalid_frames(frames) -> None:
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="PixelSeal video"):
        pixelseal_oracle.embed_video(object(), frames, pixelseal_oracle.message_bits())
    with pytest.raises(ValueError, match="PixelSeal video"):
        pixelseal_oracle.read_video(object(), frames)
