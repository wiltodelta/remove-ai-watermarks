"""Regression tests for the end-to-end video fidelity probe.

The probe needs no model and no GPU, so every behavior below is covered without a
download. Its whole output is a ranking, and a mispaired comparison still prints a
plausible number, so the pairing tests matter more than the metric ones.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
import click.testing
import numpy as np
import pytest

from remove_ai_watermarks import video_invisible

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

_SCRIPT = Path(__file__).parent.parent / "scripts" / "video_fidelity_probe.py"


@pytest.fixture(scope="module")
def probe() -> Iterator[ModuleType]:
    # The script inserts scripts/ on sys.path to reach its shared SSIM helper and
    # never removes it, which would otherwise outlive this module and defeat the
    # restore that test_fidelity_matching.py performs for the same directory.
    original_path = list(sys.path)
    spec = importlib.util.spec_from_file_location("video_fidelity_probe", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.path[:] = original_path


def _ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("the fidelity probe tests need ffmpeg to build their clips")
    return ffmpeg


def _distinct_frames(count: int, *, width: int = 64, height: int = 48) -> list[np.ndarray]:
    """Solid-color frames whose index is recoverable from the pixels.

    Distinguishable frames are the point: a pairing bug against a gradient or a
    static scene still scores well, which is how a reordering drift hides.
    """
    frames: list[np.ndarray] = []
    for index in range(count):
        # Odd multipliers, so each channel alone is injective for count <= 256. The
        # per-step deltas are what set the misalignment penalty the thresholds below
        # rely on; a constant offset would change neither and is left out.
        color = ((index * 23) % 256, (index * 71) % 256, (index * 137) % 256)
        frames.append(np.full((height, width, 3), color, dtype=np.uint8))
    return frames


def _write_clip(path: Path, frames: list[np.ndarray], *, fps: float, ffmpeg: str) -> None:
    height, width = frames[0].shape[:2]
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{fps:.12g}",
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "0",
        "-pix_fmt",
        "yuv444p",
        str(path),
    ]
    subprocess.run(  # noqa: S603
        command,
        input=b"".join(frame.tobytes() for frame in frames),
        capture_output=True,
        check=True,
    )


def test_the_probe_binds_the_engine_sampler_rather_than_its_own_copy(probe: ModuleType) -> None:
    """The "shared, not copied" contract, asserted at the seam that carries it.

    The behavioral test below pins the phase the sampler happens to select, which a
    private copy with the same phase would also satisfy. Identity is the only thing
    that fails the moment someone reimplements the rule here. It needs no ffmpeg, so
    it survives on runners where the rest of this file skips.
    """
    assert probe._iter_sampled_frames is video_invisible._iter_sampled_frames
    assert probe._probe_video is video_invisible._probe_video


def test_identical_source_and_delivery_score_as_untouched(probe: ModuleType, tmp_path: Path) -> None:
    ffmpeg = _ffmpeg()
    clip = tmp_path / "clip.mp4"
    _write_clip(clip, _distinct_frames(8), fps=12.0, ffmpeg=ffmpeg)

    row = probe._measure(clip, clip, source_geometry=probe._probe_video(clip), duration=None)

    assert row["source_psnr_db"] == float("inf")
    assert row["source_ssim"] == pytest.approx(1.0)
    assert row["pixel_ratio"] == 1.0
    assert row["frames"] == 8


def test_pairing_follows_the_engine_sampling_rule_not_just_the_frame_count(
    probe: ModuleType,
    tmp_path: Path,
) -> None:
    """The guard the frame-count check could not provide.

    Decimating 12 fps to 6 fps selects source frames 0, 2, 4, ... A rule that
    selected 1, 3, 5, ... instead returns exactly as many frames, so a count check
    passes while every comparison is against the wrong frame. Starting the sampler's
    accumulator half a period late produces exactly that, and this is the only test
    in the suite that constrains the sampler's phase at all.

    The clips encode losslessly, so the aligned pairing is exact and clears any
    floor; the misaligned one compares frames a full color step apart and lands
    near 9 dB, which is what leaves the 25 dB ceiling a wide moat rather than a
    tuned threshold.
    """
    ffmpeg = _ffmpeg()
    source = tmp_path / "source.mp4"
    frames = _distinct_frames(12)
    _write_clip(source, frames, fps=12.0, ffmpeg=ffmpeg)

    aligned = tmp_path / "aligned.mp4"
    _write_clip(aligned, frames[0::2], fps=6.0, ffmpeg=ffmpeg)
    shifted = tmp_path / "shifted.mp4"
    _write_clip(shifted, frames[1::2], fps=6.0, ffmpeg=ffmpeg)

    geometry = probe._probe_video(source)
    aligned_row = probe._measure(source, aligned, source_geometry=geometry, duration=None)
    shifted_row = probe._measure(source, shifted, source_geometry=geometry, duration=None)

    assert aligned_row["frames"] == shifted_row["frames"] == 6
    assert aligned_row["source_psnr_db"] > 40.0
    assert shifted_row["source_psnr_db"] < 25.0


def test_a_delivery_shorter_than_the_source_names_duration_rather_than_drift(
    probe: ModuleType,
    tmp_path: Path,
) -> None:
    """A short delivery is a duration prefix far more often than a real drift.

    The sweep harness trims its candidates by default, so pointing the probe at one
    is the common case rather than the pathological one. It has to refuse instead of
    scoring a truncated pairing, and the refusal has to name the fix.
    """
    ffmpeg = _ffmpeg()
    source = tmp_path / "source.mp4"
    frames = _distinct_frames(12)
    _write_clip(source, frames, fps=12.0, ffmpeg=ffmpeg)
    trimmed = tmp_path / "trimmed.mp4"
    _write_clip(trimmed, frames[:6], fps=12.0, ffmpeg=ffmpeg)

    geometry = probe._probe_video(source)
    with pytest.raises(click.ClickException, match="--duration"):
        probe._measure(source, trimmed, source_geometry=geometry, duration=None)

    row = probe._measure(source, trimmed, source_geometry=geometry, duration=0.5)

    assert row["frames"] == 6
    assert row["source_psnr_db"] > 40.0


def test_a_downscaled_delivery_is_upscaled_back_to_the_source_geometry(
    probe: ModuleType,
    tmp_path: Path,
) -> None:
    ffmpeg = _ffmpeg()
    source = tmp_path / "source.mp4"
    frames = _distinct_frames(6, width=64, height=48)
    _write_clip(source, frames, fps=12.0, ffmpeg=ffmpeg)
    delivered = tmp_path / "small.mp4"
    _write_clip(delivered, _distinct_frames(6, width=32, height=24), fps=12.0, ffmpeg=ffmpeg)

    row = probe._measure(source, delivered, source_geometry=probe._probe_video(source), duration=None)

    assert (row["width"], row["height"]) == (32, 24)
    assert row["pixel_ratio"] == 0.25
    # Geometry is the whole assertion. Solid colors survive both the downscale and
    # the upscale exactly, so any score here would pin the local ffmpeg's chroma
    # rounding rather than probe behavior. A missing upscale fails this test through
    # the shape mismatch in the metric, not through a number.
    assert row["frames"] == 6


def test_geometry_read_rejects_a_file_that_is_not_a_video(probe: ModuleType, tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")

    with pytest.raises(click.ClickException, match="Could not open video"):
        probe._delivered_geometry(broken)


def test_a_single_frame_delivery_is_rejected_before_the_temporal_metric(
    probe: ModuleType,
    tmp_path: Path,
) -> None:
    ffmpeg = _ffmpeg()
    clip = tmp_path / "one.mp4"
    _write_clip(clip, _distinct_frames(1), fps=12.0, ffmpeg=ffmpeg)

    with pytest.raises(click.ClickException, match="fewer than two frames"):
        probe._measure(clip, clip, source_geometry=probe._probe_video(clip), duration=None)


def test_the_command_writes_one_json_row_per_argument_in_order(probe: ModuleType, tmp_path: Path) -> None:
    """`main` is the only code the helper tests do not reach, and it has to round-trip.

    The rows exist to be read by something else, so the file has to parse. An
    identical source and delivery would serialize PSNR as bare ``Infinity``, which
    `json.loads` accepts but a strict reader does not; distinct deliveries keep the
    numbers finite and that stays out of the tool's normal output.
    """
    ffmpeg = _ffmpeg()
    source = tmp_path / "source.mp4"
    _write_clip(source, _distinct_frames(6), fps=12.0, ffmpeg=ffmpeg)
    small = tmp_path / "small.mp4"
    _write_clip(small, _distinct_frames(6, width=32, height=24), fps=12.0, ffmpeg=ffmpeg)
    smaller = tmp_path / "smaller.mp4"
    _write_clip(smaller, _distinct_frames(6, width=16, height=16), fps=12.0, ffmpeg=ffmpeg)
    report = tmp_path / "rows.json"

    result = click.testing.CliRunner().invoke(
        probe.main,
        [str(source), str(small), str(smaller), "--json-out", str(report)],
    )

    assert result.exit_code == 0, result.output
    rows = json.loads(report.read_text(encoding="utf-8"))
    assert [row["file"] for row in rows] == ["small.mp4", "smaller.mp4"]
    assert all(row["source"] == "source.mp4" for row in rows)


def test_fidelity_preserves_timestamps_when_native_opencv_cannot_decode(probe, tmp_path, monkeypatch):
    import cv2

    pytest.importorskip("av")
    source = tmp_path / "fallback.mp4"
    _write_clip(source, _distinct_frames(6), fps=12.0, ffmpeg=_ffmpeg())

    class ClosedCapture:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *_args: ClosedCapture())
    geometry = probe._delivered_geometry(source)
    assert geometry == (64, 48, 12.0)
    row = probe._measure(source, source, source_geometry=geometry, duration=None, preserve_timestamps=True)
    assert row["frames"] == 6
    assert row["source_ssim"] == pytest.approx(1.0)
    assert row["max_timestamp_error_seconds"] == 0
    assert row["video_duration_error_seconds"] == 0


def test_color_ssim_and_worst_frame_window_metrics_catch_chroma_inversion_and_spike_drop(
    probe: ModuleType,
    tmp_path: Path,
) -> None:
    ffmpeg = _ffmpeg()
    height, width = 48, 64
    c1 = np.array([220, 40, 20], dtype=np.uint8)
    c2 = np.array([16, 40, 98], dtype=np.uint8)
    xs = (np.arange(width) // 8) % 2
    ys = (np.arange(height) // 8) % 2
    mask = (ys[:, None] ^ xs[None, :]) == 0
    base = np.where(mask[:, :, None], c1, c2)
    swapped = np.where(mask[:, :, None], c2, c1)

    src_frames = [np.roll(base, shift=i * 2, axis=1) for i in range(16)]
    chroma_frames = [np.roll(swapped, shift=i * 2, axis=1) for i in range(16)]
    spike_frames = [f.copy() for f in src_frames]
    spike_frames[8] = np.zeros_like(base)

    src_mp4 = tmp_path / "isolum_src.mp4"
    chroma_mp4 = tmp_path / "isolum_chroma.mp4"
    spike_mp4 = tmp_path / "isolum_spike.mp4"
    _write_clip(src_mp4, src_frames, fps=12.0, ffmpeg=ffmpeg)
    _write_clip(chroma_mp4, chroma_frames, fps=12.0, ffmpeg=ffmpeg)
    _write_clip(spike_mp4, spike_frames, fps=12.0, ffmpeg=ffmpeg)

    geom = (width, height, 12.0)
    row_chroma = probe._measure(src_mp4, chroma_mp4, source_geometry=geom, duration=None)
    assert row_chroma["source_ssim"] >= 0.98
    assert row_chroma["source_color_ssim"] < 0.25

    row_spike = probe._measure(src_mp4, spike_mp4, source_geometry=geom, duration=None)
    assert row_spike["source_ssim"] > 0.90
    assert row_spike["min_frame_ssim"] < 0.15
    assert row_spike["min_frame_psnr_db"] < 16.0
    assert row_spike["worst_window_ssim"] < row_spike["source_ssim"]
