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


def test_chunkyseal_network_architecture_and_raw_slice_loader(tmp_path, monkeypatch):
    import struct
    import zlib

    torch = pytest.importorskip("torch")
    net = models._build_chunkyseal_network()
    state = net.detector.state_dict()
    assert len(state) == 381
    keys = list(state)
    assert keys[0] == "convnext.downsample_layers.0.0.weight"
    assert tuple(state[keys[0]].shape) == (362, 3, 4, 4)
    assert keys[-2] == "pixel_decoder.linear.weight"
    assert tuple(state[keys[-2]].shape) == (1025, 2896)
    assert keys[-1] == "pixel_decoder.linear.bias"
    assert tuple(state[keys[-1]].shape) == (1025,)

    frames = np.full((2, 32, 48, 3), 0.25, dtype=np.float32)
    decoded = models.decode_logits(models.VideoMessageDecoder("chunkyseal", net), frames)
    assert decoded.shape == (2, 1024)
    assert np.isfinite(decoded).all()

    # Build a synthetic PK\x03\x04 + PK\x07\x08 uncompressed slice for a 2-tensor stub detector
    class TinyDetector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 3)

    tiny = TinyDetector()
    expected_w = np.arange(12, dtype="<f4").reshape(3, 4)
    expected_b = np.array([-1.5, 0.5, 2.5], dtype="<f4")
    slice_buf = bytearray()
    for idx, payload_arr in enumerate((expected_w, expected_b)):
        fname = f"checkpoint/data/{models._CHUNKYSEAL_FIRST_STORAGE_KEY + idx}".encode()
        raw_bytes = payload_arr.tobytes()
        crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
        hdr = struct.pack("<4sHHHHHIIIHH", b"PK\x03\x04", 0, 0x0808, 0, 0, 0, 0, 0, 0, len(fname), 4)
        desc = struct.pack("<4sIII", b"PK\x07\x08", crc, len(raw_bytes), len(raw_bytes))
        slice_buf.extend(hdr + fname + b"ABCD" + raw_bytes + desc)

    slice_path = tmp_path / "chunkyseal.slice.pth"
    slice_path.write_bytes(bytes(slice_buf))
    assert models._peek_zip_entry_name(slice_path) == "checkpoint/data/228"
    models._load_chunkyseal_slice(slice_path, tiny)
    np.testing.assert_array_equal(tiny.linear.weight.detach().numpy(), expected_w)
    np.testing.assert_array_equal(tiny.linear.bias.detach().numpy(), expected_b)

    # Corrupting a single byte in the tensor payload must trip the per-entry CRC32 guard
    bad_buf = bytearray(slice_buf)
    bad_buf[-20] ^= 0xFF
    bad_path = tmp_path / "corrupt.slice.pth"
    bad_path.write_bytes(bytes(bad_buf))
    with pytest.raises(RuntimeError, match="Corrupted ZIP data descriptor"):
        models._load_chunkyseal_slice(bad_path, TinyDetector())

    # Central-directory digest mismatch must abort before downloading the slice
    monkeypatch.setattr(models, "_fetch_range_bytes", lambda _u, s, e: b"x" * (e - s + 1))
    with pytest.raises(RuntimeError, match="central directory differs"):
        models._download_chunkyseal_detector_slice("https://example.invalid/ckpt", tmp_path / "staged")
