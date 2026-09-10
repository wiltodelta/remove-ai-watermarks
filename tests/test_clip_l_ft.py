"""CLIP-L checkpoint loading without redundant parameter initialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def test_load_headed_clip_skips_random_initialization_and_materializes_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from remove_ai_watermarks._internal import clip_l_ft

    config = transformers.CLIPConfig(
        projection_dim=2,
        text_config={
            "vocab_size": 8,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 1,
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "max_position_embeddings": 8,
        },
        vision_config={
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "image_size": 4,
            "patch_size": 2,
            "num_channels": 3,
        },
    )
    expected = clip_l_ft.HeadedCLIP(transformers.CLIPModel(config))
    with torch.no_grad():
        for index, parameter in enumerate(expected.parameters(), start=1):
            parameter.fill_(float(index))
    checkpoint = tmp_path / "tiny-clip.pt"
    torch.save(expected.state_dict(), checkpoint)

    def reject_random_initialization(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("checkpoint loading must not initialize discarded random weights")

    monkeypatch.setattr(transformers.CLIPConfig, "from_pretrained", lambda _backbone: config)
    monkeypatch.setattr(torch.nn.init, "kaiming_uniform_", reject_random_initialization)

    loaded = clip_l_ft.load_headed_clip(checkpoint, torch.device("cpu"))

    assert not any(value.is_meta for value in (*loaded.parameters(), *loaded.buffers()))
    expected_state = expected.state_dict()
    loaded_state = loaded.state_dict()
    for name, expected_value in expected_state.items():
        assert torch.equal(loaded_state[name], expected_value)
    for (name, buffer), (expected_name, expected_buffer) in zip(
        loaded.named_buffers(), expected.named_buffers(), strict=True
    ):
        assert name == expected_name
        assert torch.equal(buffer, expected_buffer)
