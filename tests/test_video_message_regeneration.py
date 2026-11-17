"""VAE trust boundary and complete float32 checkpoint loading."""

import hashlib
from types import SimpleNamespace

import pytest
import torch

from remove_ai_watermarks._internal import video_message_regeneration as regen


@pytest.mark.parametrize("fault", [None, "digest", "partial", "scale"])
def test_vae_verifies_files_and_full_state_before_use(tmp_path, monkeypatch, fault):
    config = tmp_path / "config.json"
    weights = tmp_path / "diffusion_pytorch_model.safetensors"
    config.write_bytes(b"config")
    weights.write_bytes(b"weights")
    monkeypatch.setattr(regen, "FILES", {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (config, weights)})
    if fault == "digest":
        weights.write_bytes(b"corrupt")
    monkeypatch.setattr(regen, "is_available", lambda: True)
    calls = []

    def download(**kwargs):
        assert kwargs["revision"] == regen.REVISION
        assert kwargs["token"] is False
        return str(tmp_path / kwargs["filename"])

    vae = SimpleNamespace(config=SimpleNamespace(scaling_factor=0.5 if fault == "scale" else 0.18215))
    vae.to = lambda device: vae
    vae.train = lambda mode: calls.append(("train", mode))
    vae.enable_slicing = lambda: calls.append("slicing")

    def load(path, **kwargs):
        calls.append("load")
        assert kwargs == {
            "local_files_only": True,
            "use_safetensors": True,
            "torch_dtype": torch.float32,
            "output_loading_info": True,
        }
        torch.rand(4)
        return vae, {"missing_keys": ["bad"]} if fault == "partial" else {}

    modules = {
        "torch": torch,
        "huggingface_hub": SimpleNamespace(hf_hub_download=download),
        "diffusers": SimpleNamespace(AutoencoderKL=SimpleNamespace(from_pretrained=load)),
    }
    monkeypatch.setattr(regen, "import_module", modules.__getitem__)
    regen.load_runtime.cache_clear()
    state = torch.random.get_rng_state()
    try:
        if fault:
            with pytest.raises(RuntimeError, match=r"digest|completely|scaling"):
                regen.load_runtime("cpu")
            if fault == "digest":
                assert calls == []
        else:
            result = regen.load_runtime("cpu")
            assert result.scaling_factor == 0.18215
            assert ("train", False) in calls
        assert torch.equal(state, torch.random.get_rng_state())
    finally:
        regen.load_runtime.cache_clear()
