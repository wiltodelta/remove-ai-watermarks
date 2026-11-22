"""Regression tests for Wave 9 (v0.70.0) video and audio quality improvements."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pytest

from remove_ai_watermarks import video_invisible, video_seedance, video_temporal, video_wan

if TYPE_CHECKING:
    from pathlib import Path


def _lab_delta_e(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    scale_a = 65535.0 if a_bgr.dtype == np.uint16 else 255.0
    scale_b = 65535.0 if b_bgr.dtype == np.uint16 else 255.0
    a_lab = cv2.cvtColor(a_bgr.astype(np.float32) / scale_a, cv2.COLOR_BGR2Lab)
    b_lab = cv2.cvtColor(b_bgr.astype(np.float32) / scale_b, cv2.COLOR_BGR2Lab)
    return float(np.mean(np.linalg.norm(a_lab - b_lab, axis=2)))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_anchor_low_freq_color_corrects_global_cast_without_leaking_midband(dtype: Any) -> None:
    peak = 65535.0 if dtype == np.uint16 else 255.0
    yy, xx = np.mgrid[0:128, 0:128]
    # Source has a smooth gradient + mid-band 8 px sinusoid
    base_b = 0.45 + 0.10 * (xx / 128.0)
    base_g = 0.50 + 0.05 * (yy / 128.0)
    base_r = 0.55 - 0.08 * (xx / 128.0)
    midband = 0.08 * np.sin(2.0 * np.pi * xx / 8.0)
    src_f32 = np.clip(np.stack([base_b + midband, base_g, base_r], axis=2), 0.0, 1.0)
    # Candidate has no mid-band wave, plus a strong warm color/exposure cast
    cand_f32 = np.clip(np.stack([base_b - 0.12, base_g + 0.04, base_r + 0.14], axis=2), 0.0, 1.0)

    src = np.rint(src_f32 * peak).astype(dtype)
    cand = np.rint(cand_f32 * peak).astype(dtype)
    anchored = video_temporal.anchor_low_freq_color(src, cand)

    assert anchored.dtype == dtype
    assert anchored.shape == cand.shape
    before_de = _lab_delta_e(src, cand)
    after_de = _lab_delta_e(src, anchored)
    assert after_de < before_de * 0.35

    # Verify mid-band sinusoid from source did NOT leak into anchored candidate
    anchored_b = anchored[..., 0].astype(np.float32) / peak
    mid_corr = float(np.mean((anchored_b - np.mean(anchored_b)) * midband))
    src_corr = float(np.mean((src_f32[..., 0] - np.mean(src_f32[..., 0])) * midband))
    assert abs(mid_corr) < abs(src_corr) * 0.05


def test_regenerate_video_candidate_defaults_temporal_stabilize_true() -> None:
    sig = inspect.signature(video_invisible.regenerate_video_candidate)
    assert sig.parameters["temporal_stabilize"].default is True


def test_wan21_mps_uses_compact_33_frame_window_and_9_frame_overlap() -> None:
    assert video_wan.WAN21_MPS_CHUNK_FRAMES == 33
    assert video_wan.WAN21_MPS_OVERLAP_FRAMES == 9
    eff_chunk = video_wan._effective_chunk_frames(193, 17, width=1280, height=720, resolved_device="mps")
    eff_overlap = video_wan._effective_overlap_frames(17, eff_chunk=eff_chunk, resolved_device="mps")
    assert eff_chunk == 33
    assert eff_overlap == 9


def test_wan21_high_depth_passes_float32_arrays_without_8bit_truncation() -> None:
    captured_videos: list[Any] = []

    def fake_pipe(**kwargs: Any) -> Any:
        captured_videos.append(kwargs["video"])
        frames_np = np.stack(kwargs["video"], axis=0)
        return SimpleNamespace(frames=[frames_np])

    runtime = video_wan.VideoWanRuntime(
        model=video_wan.WAN21_MODEL_ID,
        revision=video_wan.WAN21_MODEL_REVISION,
        resolved_device="cpu",
        pipe=fake_pipe,
        prompt_embeds=None,
    )
    # Two adjacent 10-bit levels that collapse to identical values under >> 8
    frame_a = np.full((16, 16, 3), 25600, dtype=np.uint16)
    frame_b = np.full((16, 16, 3), 25664, dtype=np.uint16)
    out = video_wan._run_wan_window(
        [frame_a, frame_b, frame_a, frame_b, frame_a],
        runtime,
        strength=0.12,
        seed=0,
        orig_h=16,
        orig_w=16,
    )
    assert len(captured_videos) == 1
    assert isinstance(captured_videos[0][0], np.ndarray)
    assert captured_videos[0][0].dtype == np.float32
    assert out.dtype == np.uint16
    assert int(out[1, 0, 0, 0]) - int(out[0, 0, 0, 0]) == pytest.approx(64, abs=1)


def test_seedance_eliminates_guide_mp4_and_stabilizes_color_cast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required")

    source = tmp_path / "src.mp4"
    subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=12:duration=0.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )

    encode_calls: list[Path] = []
    real_encode = video_seedance._encode_frames

    def spy_encode(frames: Any, src: Path, out: Path, *args: Any, **kwargs: Any) -> None:
        encode_calls.append(out)
        real_encode(frames, src, out, *args, **kwargs)

    monkeypatch.setattr(video_seedance, "_encode_frames", spy_encode)
    monkeypatch.setattr(video_seedance, "is_available", lambda: True)
    monkeypatch.setattr(video_seedance, "load_video_vae_runtime", lambda **_: SimpleNamespace())

    # Inject alternating color cast + frame-to-frame flicker in fake VAE guide
    def noisy_reconstruct(src_path: Path, _runtime: Any, size: tuple[int, int]) -> Any:
        for idx, frame in enumerate(video_seedance._frames(src_path)):
            small = cv2.resize(frame, size).astype(np.int16)
            small[..., 2] = np.clip(small[..., 2] + 25 + (6 if idx % 2 else -6), 0, 255)
            yield small.astype(np.uint8)

    monkeypatch.setattr(video_seedance, "_reconstruct", noisy_reconstruct)
    output = tmp_path / "out.mp4"
    metrics = video_seedance.regenerate_seedance_experimental(source, output, quality="master", device="cpu")
    assert all(p.name != "guide.mp4" for p in encode_calls)
    assert metrics.psnr_db > 25.0


def test_benchmark_suite_registers_native48k_audioseal_case() -> None:
    import sys
    from pathlib import Path

    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import media_removal_benchmark as bench  # type: ignore[import-not-found]

    assert "audioseal-native48k-music-marked" in bench.STRICT_BUILTIN_REQUIRED_CASE_IDS
