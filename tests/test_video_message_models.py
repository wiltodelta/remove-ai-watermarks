"""Asset integrity, decoder-only loading, and raw prediction guards."""

import hashlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

from remove_ai_watermarks._internal import video_message_models as models


@pytest.mark.parametrize("corrupt", ["videoseal", "pixelseal"])
def test_both_assets_are_verified_before_deserializing(tmp_path, monkeypatch, corrupt):
    monkeypatch.setattr(models, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(models, "is_available", lambda: True)
    assets = {}
    for scheme, suffix in [("videoseal", "jit"), ("pixelseal", "pth")]:
        data = scheme.encode()
        assets[scheme] = ("https://example.invalid/model", hashlib.sha256(data).hexdigest())
        (tmp_path / (scheme + "-" + assets[scheme][1] + "." + suffix)).write_bytes(
            b"corrupt" if corrupt == scheme else data
        )
    monkeypatch.setattr(models, "ASSETS", assets)
    torch = SimpleNamespace(jit=SimpleNamespace(load=lambda *a, **kw: pytest.fail("Unverified graph was loaded")))
    monkeypatch.setattr(models, "import_module", lambda name: torch)
    models.load_detector.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="pinned digest"):
            models.load_detector("pixelseal")
    finally:
        models.load_detector.cache_clear()


@pytest.mark.parametrize("valid", [False, True])
def test_only_verified_downloads_are_published_and_cached(tmp_path, monkeypatch, valid):
    monkeypatch.setattr(models, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        models, "ASSETS", {"videoseal": ("https://example.invalid/model", hashlib.sha256(b"trusted").hexdigest())}
    )
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"trusted" if valid else b"bad"))
    if valid:
        path = models._asset("videoseal")
        assert path.read_bytes() == b"trusted"
        monkeypatch.setattr(
            models.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("Cached file was redownloaded")
        )
        assert models._asset("videoseal") == path
    else:
        with pytest.raises(RuntimeError, match="pinned digest"):
            models._asset("videoseal")
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fault", [None, "empty_graph", "empty_detector", "not_dict", "no_model", "bad_key"])
def test_pixel_loading_is_weights_only_and_strict(tmp_path, monkeypatch, fault):
    calls = []
    checkpoint = {"model": {"detector.weight": "pixel", "embedder.weight": "unused"}}
    if fault == "empty_detector":
        checkpoint = {"model": {"embedder.weight": "unused"}}
    elif fault == "not_dict":
        checkpoint = []
    elif fault == "no_model":
        checkpoint = {"other": {}}
    elif fault == "bad_key":
        checkpoint = {"model": {3: "bad"}}

    def load(path, *, map_location, weights_only):
        assert map_location == "cpu"
        assert weights_only is True
        return checkpoint

    def load_state(state, *, strict):
        assert strict is True
        assert state == {"weight": "pixel"}
        calls.append("state")

    detector = SimpleNamespace(
        state_dict=lambda: {} if fault == "empty_graph" else {"weight": "video"}, load_state_dict=load_state
    )
    network = SimpleNamespace(detector=detector, train=lambda training: calls.append(training))
    torch = SimpleNamespace(jit=SimpleNamespace(load=lambda *a, **kw: network), load=load)
    monkeypatch.setattr(models, "is_available", lambda: True)
    monkeypatch.setattr(models, "_asset", lambda scheme: tmp_path / scheme)
    monkeypatch.setattr(models, "import_module", lambda name: torch)
    models.load_detector.cache_clear()
    try:
        if fault:
            with pytest.raises(RuntimeError):
                models.load_detector("pixelseal")
            assert not calls
        else:
            assert models.load_detector("pixelseal").scheme == "pixelseal"
            assert calls == ["state", False]
    finally:
        models.load_detector.cache_clear()


def test_frame_layout_and_message_columns_are_preserved():
    torch = pytest.importorskip("torch")
    frames = np.arange(2 * 4 * 6 * 3, dtype=np.float32).reshape(2, 4, 6, 3) / 144
    logits = torch.arange(514, dtype=torch.float32).reshape(2, 257)

    class Network:
        def detect(self, tensor, *, is_video):
            assert is_video is True
            torch.testing.assert_close(tensor, torch.from_numpy(frames).permute(0, 3, 1, 2))
            return logits

    result = models.decode_logits(models.VideoMessageDecoder("videoseal", Network()), frames)
    np.testing.assert_array_equal(result, logits[:, 1:].numpy())
    assert models.validate_logits(logits, 2) is logits


@pytest.mark.parametrize("fault", ["nan", "range", "rank", "channels", "empty"])
def test_bad_frames_never_reach_the_network(fault):
    frames = np.zeros((2, 4, 6, 3), dtype=np.float32)
    if fault == "nan":
        frames[0, 0, 0, 0] = np.nan
    elif fault == "range":
        frames[0, 0, 0, 0] = 255
    elif fault == "rank":
        frames = frames[0]
    elif fault == "channels":
        frames = frames[..., :1]
    else:
        frames = frames[:0]
    network = SimpleNamespace(detect=lambda *a, **kw: pytest.fail("Invalid frames reached the network"))
    with pytest.raises(ValueError, match="finite TxHxWx3"):
        models.decode_logits(models.VideoMessageDecoder("videoseal", network), frames)
