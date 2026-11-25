"""Regression and resource-efficiency tests for v0.75.0 audio/video optimizations."""

from __future__ import annotations

import io
import math
import subprocess
import tracemalloc
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import pytest

import remove_ai_watermarks as raiw
from remove_ai_watermarks import audio, cli, video, video_encoding, video_seedance, video_tools, video_visible

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_stereo_wav(path: Path, *, sample_rate: int = 48_000, seconds: float = 2.0) -> None:
    frames = int(sample_rate * seconds)
    t = np.arange(frames, dtype=np.float64) / sample_rate
    left = 0.22 * np.sin(2.0 * math.pi * 440.0 * t)
    right = 0.22 * np.cos(2.0 * math.pi * 659.25 * t)
    pcm = np.clip(np.column_stack((left, right)) * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _write_clean_mp4(path: Path, *, frames: int = 32, width: int = 160, height: int = 120) -> None:
    proc = video_encoding.start_raw_video_encoder(
        video_encoding.raw_video_command(
            path,
            width=width,
            height=height,
            fps=24.0,
            crf=22,
            profile=video_encoding.VideoEncodeProfile(),
        )
    )
    try:
        for idx in range(frames):
            frame = np.full((height, width, 3), 40 + (idx % 16), dtype=np.uint8)
            video_encoding.write_raw_frame_bytes(proc.stdin, frame)
        video_encoding.finish_raw_video_encoder(proc, path, operation="test clean mp4")
    except Exception:
        video_encoding.abort_raw_video_encoder(proc)
        raise


def test_probe_video_timestamps_prefers_packets_and_sorts_bframe_pts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "bframes.mp4"
    source.write_bytes(b"video")
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(video_tools.shutil, "which", lambda _name: "/usr/bin/ffprobe")

    def fake_run(cmd: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(tuple(cmd))
        return SimpleNamespace(
            returncode=0,
            stdout="0.000000,K__\n0.083333,___\n0.041667,___\n",
            stderr="",
        )

    monkeypatch.setattr(video_encoding.subprocess, "run", fake_run)
    pts = video_encoding.probe_video_timestamps(source)
    assert pts == (0.0, 0.041667, 0.083333)
    assert len(calls) == 1
    assert "-show_packets" in calls[0]


@pytest.mark.parametrize(
    "packet_stdout",
    [
        "0.000000,KD_\n0.041667,___\n",
        "N/A,K__\n0.041667,___\n",
        "0.000000,K__\n0.000000,___\n",
    ],
)
def test_probe_video_timestamps_falls_back_on_discard_missing_or_duplicate_pts(
    packet_stdout: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "editlist.mp4"
    source.write_bytes(b"video")
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(video_tools.shutil, "which", lambda _name: "/usr/bin/ffprobe")

    def fake_run(cmd: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(tuple(cmd))
        if "-show_packets" in cmd:
            return SimpleNamespace(returncode=0, stdout=packet_stdout, stderr="")
        return SimpleNamespace(returncode=0, stdout="0.000000\n0.041667\n", stderr="")

    monkeypatch.setattr(video_encoding.subprocess, "run", fake_run)
    pts = video_encoding.probe_video_timestamps(source)
    assert pts == (0.0, 0.041667)
    assert len(calls) == 2
    assert "-show_frames" in calls[1]


def test_write_raw_frame_bytes_zero_copy_and_noncontiguous_and_uint16() -> None:
    rng = np.random.default_rng(7)
    full_1080p = rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)

    class _NullPipe:
        def __init__(self) -> None:
            self.nbytes = 0

        def write(self, view: Any) -> int:
            n = len(view)
            self.nbytes += n
            return n

    null_pipe = _NullPipe()
    tracemalloc.start()
    video_encoding.write_raw_frame_bytes(null_pipe, full_1080p)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert null_pipe.nbytes == full_1080p.nbytes
    assert peak_bytes < 32 * 1024

    sliced = full_1080p[:16, ::2, :]
    assert not sliced.flags.c_contiguous
    buf_u8 = io.BytesIO()
    video_encoding.write_raw_frame_bytes(buf_u8, sliced)
    assert buf_u8.getvalue() == np.ascontiguousarray(sliced).tobytes()

    arr_u16 = rng.integers(0, 65535, size=(12, 16, 3), dtype=np.uint16)
    buf_u16 = io.BytesIO()
    video_encoding.write_raw_frame_bytes(buf_u16, arr_u16)
    assert buf_u16.getvalue() == arr_u16.tobytes()


def test_coarse_to_fine_scan_prunes_clean_clip_and_preserves_dense_parity(tmp_path: Path) -> None:
    clean_clip = tmp_path / "clean.mp4"
    _write_clean_mp4(clean_clip, frames=32)

    call_counts = {"a": 0, "b": 0}

    def det_a(_frame: np.ndarray, *, frame_index: int = 0, prepared: Any = None) -> video_visible.FrameLocalization:
        _ = prepared
        call_counts["a"] += 1
        return video_visible.FrameLocalization(frame_index, 0.0, None)

    def det_b(_frame: np.ndarray, *, frame_index: int = 0, prepared: Any = None) -> video_visible.FrameLocalization:
        _ = prepared
        call_counts["b"] += 1
        return video_visible.FrameLocalization(frame_index, 0.05, (4, 4, 12, 12))

    scans = video_visible._scan_video_detectors(clean_clip, {"sora": det_a, "veo": det_b}, collect_timestamps=False)
    assert len(scans["sora"].detections) == 32
    assert len(scans["veo"].detections) == 32
    assert call_counts["a"] == 8
    assert call_counts["b"] == 8

    veo_fixture = REPO_ROOT / "data" / "fixtures" / "visible" / "veo" / "example.mp4"
    cap = video_visible.open_video_capture(veo_fixture)
    try:
        dense_frames: list[video_visible.FrameLocalization] = []
        idx = 0
        while True:
            ok, frm = cap.read()
            if not ok:
                break
            dense_frames.append(video_visible.detect_veo_frame(frm, frame_index=idx))
            idx += 1
    finally:
        cap.release()
    multi_scans = video_visible.scan_video_marks(
        veo_fixture,
        ("veo", "pixverse", "luma"),
        collect_timestamps=False,
    )
    assert multi_scans["veo"].detections == tuple(dense_frames)

    empty_scan = video_visible.probe_empty_video_scan(clean_clip, collect_timestamps=False)
    assert len(empty_scan.detections) == 32
    assert all(det.region is None for det in empty_scan.detections)


def test_sora_pyramid_matches_exact_region_and_restores_cv2_threads() -> None:
    sora_fixture = REPO_ROOT / "data" / "fixtures" / "visible" / "sora" / "example.mp4"
    prev_threads = cv2.getNumThreads()
    rep = video.identify_video(sora_fixture)
    assert cv2.getNumThreads() == prev_threads
    assert rep.visible_mark == "sora"
    assert (rep.visible_detected_frames or 0) >= 5


def test_identify_audio_auto_fuses_ffmpeg_decode_and_matches_per_scheme(tmp_path: Path) -> None:
    if not (audio.audioseal.is_available() and audio.wavmark.is_available() and audio.perth.is_available()):
        pytest.skip("Optional public audio decoders are not all installed")

    wav_path = tmp_path / "stereo_48k.wav"
    _write_stereo_wav(wav_path, sample_rate=48_000, seconds=2.0)

    ffmpeg_spawns = 0
    orig_popen = subprocess.Popen

    def count_popen(*args: Any, **kwargs: Any) -> Any:
        nonlocal ffmpeg_spawns
        cmd = args[0] if args else kwargs.get("args", [])
        if isinstance(cmd, (list, tuple)) and cmd and "ffmpeg" in str(cmd[0]).lower():
            ffmpeg_spawns += 1
        return orig_popen(*args, **kwargs)

    with patch("subprocess.Popen", side_effect=count_popen):
        fused = audio.identify_audio(wav_path, scheme="auto")
    assert ffmpeg_spawns == 1

    expected_dets: list[audio.AudioDetection] = []
    for name in ("audioseal", "wavmark", "perth"):
        single = audio.identify_audio(wav_path, scheme=name)
        expected_dets.extend(single.detections)

    assert fused.detections == tuple(expected_dets)


def test_release_models_and_low_memory_cli_and_stage_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from click.testing import CliRunner

    runner = CliRunner()
    for subcmd in (
        ["audio", "invisible", "--help"],
        ["audio", "all", "--help"],
        ["audio", "batch", "--help"],
        ["video", "invisible", "--help"],
        ["video", "all", "--help"],
        ["video", "batch", "--help"],
    ):
        res = runner.invoke(cli.main, subcmd)
        assert res.exit_code == 0
        assert "--low-memory" in res.output

    if audio.audioseal.is_available():
        audio.audioseal.load_pinned_detector()
        assert audio.audioseal.load_pinned_detector.cache_info().currsize == 1
        raiw.release_models()
        assert audio.audioseal.load_pinned_detector.cache_info().currsize == 0

    wav_in = tmp_path / "in.wav"
    wav_out = tmp_path / "out.wav"
    _write_stereo_wav(wav_in, sample_rate=16_000, seconds=1.2)
    release_calls: list[str] = []
    monkeypatch.setattr(audio, "release_audio_models", lambda: release_calls.append("audio"))
    audio.remove_audio_invisible(
        wav_in,
        wav_out,
        method="spectral",
        allow_unverified=True,
        verify=False,
        low_memory=True,
    )
    assert release_calls == ["audio"]


def test_seedance_reconstruct_opens_single_source_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = tmp_path / "src.mp4"
    _write_clean_mp4(clip, frames=6, width=64, height=64)
    out = tmp_path / "out.mp4"

    source_opens = 0
    orig_frames = video_seedance._frames

    def counting_frames(path: Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal source_opens
        if path == clip:
            source_opens += 1
        return orig_frames(path, *args, **kwargs)

    class _FakeLatent:
        shape = (1, 4, 8, 8)
        dtype = None

    class _FakeRuntime:
        model = "stabilityai/sd-vae-ft-mse"
        requested_device = "cpu"
        resolved_device = "cpu"
        vae = object()
        scaling_factor = 0.18215

    import torch

    _FakeLatent.dtype = torch.float32
    monkeypatch.setattr(video_seedance, "_frames", counting_frames)
    monkeypatch.setattr(
        video_seedance,
        "_encode_frame_latents",
        lambda batch, **_kw: [torch.zeros((len(batch), 4, 8, 8), dtype=torch.float32)],
    )
    monkeypatch.setattr(
        video_seedance,
        "_decode_frame_latents",
        lambda latents, **_kw: [np.full((64, 64, 3), 42, dtype=np.uint8) for _ in range(latents[0].shape[0])],
    )

    metrics = video_seedance.regenerate_seedance_experimental(
        clip,
        out,
        device="cpu",
        quality="delivery",
        runtime=_FakeRuntime(),  # type: ignore[arg-type]
    )
    assert metrics.frames == 6
    # Exactly 1 decode during reconstruction + 1 decode during final _measure (previously 2 + 1 = 3)
    assert source_opens == 2
