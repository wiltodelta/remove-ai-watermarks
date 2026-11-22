"""Tests for Wave 6 features: PyAV grab/retrieve, video_messages VFR/non-zero PTS, and studio HF crossover."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import scipy.signal as sig

from remove_ai_watermarks import video_decode, video_invisible, video_messages
from remove_ai_watermarks._internal import audio_dac
from remove_ai_watermarks.video_tools import require_video_tool

if TYPE_CHECKING:
    from pathlib import Path


def test_pyav_capture_grab_skips_bgr24_conversion() -> None:
    """_PyAVCapture.grab() advances frames without calling to_ndarray until retrieve()."""
    converted: list[int] = []

    class _FakeFrame:
        def __init__(self, idx: int) -> None:
            self._idx = idx
            self.time = idx / 24.0

        def to_ndarray(self, *, format: str) -> np.ndarray[Any, Any]:
            assert format == "bgr24"
            converted.append(self._idx)
            return np.full((8, 8, 3), self._idx * 10, dtype=np.uint8)

    class _FakeStream:
        width = 8
        height = 8
        average_rate = 24
        guessed_rate = 24
        frames = 3

    class _FakeStreams:
        def __init__(self) -> None:
            self.video = [_FakeStream()]

    class _FakeContainer:
        def __init__(self) -> None:
            self.streams = _FakeStreams()

        def decode(self, _stream: Any) -> list[_FakeFrame]:
            return [_FakeFrame(0), _FakeFrame(1), _FakeFrame(2)]

        def close(self) -> None:
            pass

    cap = video_decode._PyAVCapture(_FakeContainer())
    assert cap.grab() is True  # skip frame 0 without conversion
    assert converted == []
    ok, f1 = cap.read()
    assert ok is True
    assert f1 is not None
    assert int(f1[0, 0, 0]) == 10
    assert converted == [1]
    assert cap.grab() is True
    ok2, f2 = cap.retrieve()
    assert ok2 is True
    assert f2 is not None
    assert int(f2[0, 0, 0]) == 20
    assert converted == [1, 2]
    assert cap.grab() is False
    cap.release()


def test_video_messages_supports_vfr_and_nonzero_start_pts(tmp_path: Path) -> None:
    """_processing_source and _expected_sampled_frames accept non-zero start PTS and VFR timelines."""
    ffmpeg = require_video_tool("ffmpeg")
    src = tmp_path / "nonzero_start.mp4"
    # Generate 24-frame clip with start_pts shifted by +1.5s
    subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=24:duration=1",
            "-vf",
            "setpts=PTS+1.5/TB",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(src),
        ],
        check=True,
    )
    geometry, audio_streams, timestamps = video_messages._processing_source(src)
    assert geometry[0] == 64
    assert geometry[1] == 64
    assert audio_streams == ()
    assert len(timestamps) >= 20
    assert timestamps[0] >= 1.4

    effective_fps = min(12.0, geometry[2])
    expected = video_messages._expected_sampled_frames(timestamps, effective_fps)
    sampled = list(
        video_invisible.iter_sampled_frames(
            src,
            source_fps=geometry[2],
            duration=None,
            effective_fps=effective_fps,
            size=(64, 64),
            timestamps=timestamps,
        )
    )
    assert len(sampled) == expected
    assert expected >= 10

    # VFR relative-timestamp sampling works cleanly
    vfr_ts = (0.5, 0.52, 0.60, 0.72, 0.90, 1.05)
    assert video_messages._expected_sampled_frames(vfr_ts, 4.0) == 3


def test_blend_hf_crossover_preserves_high_frequency_transients_and_validates() -> None:
    """Zero-phase Butterworth HF crossover preserves >14 kHz transients while keeping low-band processed audio."""
    sr = 48000
    t = np.arange(sr, dtype=np.float32) / sr
    low_orig = 0.3 * np.sin(2 * np.pi * 440 * t)
    low_proc = 0.3 * np.sin(2 * np.pi * 440 * t + 0.5)  # phase-shifted processed low band
    hi_orig = 0.2 * np.sin(2 * np.pi * 16500 * t)
    orig = np.column_stack([low_orig + hi_orig, low_orig + hi_orig]).astype(np.float32)
    proc = np.column_stack([low_proc, low_proc]).astype(np.float32)  # codec attenuated >14 kHz

    blended = audio_dac.blend_hf_crossover(proc, orig, sr, 14000.0)
    sos_hp = sig.butter(8, 14000.0, btype="high", fs=sr, output="sos")
    hf_orig = sig.sosfiltfilt(sos_hp, orig, axis=0)
    hf_proc = sig.sosfiltfilt(sos_hp, proc, axis=0)
    hf_blended = sig.sosfiltfilt(sos_hp, blended, axis=0)

    err_before = float(np.sqrt(np.mean((hf_orig - hf_proc) ** 2)))
    err_after = float(np.sqrt(np.mean((hf_orig - hf_blended) ** 2)))
    assert err_after < err_before * 0.05

    with pytest.raises(ValueError, match="preserve_hf_hz"):
        audio_dac.blend_hf_crossover(proc, orig, sr, 2000.0)
    with pytest.raises(ValueError, match="preserve_hf_hz"):
        audio_dac.blend_hf_crossover(proc, orig, sr, 23000.0)


def test_default_video_profile_is_auto_and_routes_seedance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """video all / video invisible default to profile='auto', routing Seedance seamlessly without a separate flag."""
    from remove_ai_watermarks import video, video_wan

    # 1. Default validation succeeds when include_invisible=False (auto does not error)
    video._validate_video_profile("auto", "delivery", enabled=False)

    # 2. ByteDance/Seedance clip resolves to seedance-experimental on CPU without passing --profile
    src = tmp_path / "seedance_clip.mp4"
    src.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
    monkeypatch.setattr(video_wan, "can_run_wan_on_source", lambda *a, **k: False)
    resolved = video._resolve_video_invisible_profile(
        "auto",
        src,
        device="cpu",
        quality="delivery",
        noise_std=video.DEFAULT_VIDEO_SYNTHID_NOISE_STD,
        long_side=video.DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
        fps=video.DEFAULT_VIDEO_SYNTHID_FPS,
        batch_size=4,
        seed=0,
        model=video.DEFAULT_VIDEO_SYNTHID_VAE,
        temporal_stabilize=True,
        visible_mark="jimeng",
    )
    assert resolved == "seedance-experimental"
