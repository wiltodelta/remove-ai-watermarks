"""Regression and end-to-end tests for Wave 3 audio/video capabilities.

Covers:
1. End-to-end 10-bit HDR (yuv420p10le, BT.2020, PQ/HLG) support in
   video_invisible (SD-VAE + temporal stabilization) and Wan 2.1.
2. Background-locked bidirectional multi-frame temporal video inpainting and
   transitive clean-plate propagation across slow pans.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from remove_ai_watermarks.video_encoding import VideoEncodeProfile
from remove_ai_watermarks.video_tools import require_video_tool

if TYPE_CHECKING:
    from pathlib import Path


def _run_cmd(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    assert res.returncode == 0, f"Command failed: {cmd}\nstderr={res.stderr}"


def _probe_stream(path: Path) -> dict[str, object]:
    ffprobe = require_video_tool("ffprobe")
    res = subprocess.run(  # noqa: S603
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=pix_fmt,color_transfer,color_primaries,color_space,width,height",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(res.stdout)["streams"][0]


def _make_hdr10_clip(path: Path, *, width: int = 64, height: int = 64, duration: float = 0.5, fps: int = 10) -> None:
    ffmpeg = require_video_tool("ffmpeg")
    _run_cmd(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate={fps}:duration={duration}",
            "-vf",
            "format=yuv420p10le",
            "-c:v",
            "libx265",
            "-tag:v",
            "hvc1",
            "-pix_fmt",
            "yuv420p10le",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "smpte2084",
            "-colorspace",
            "bt2020nc",
            "-x265-params",
            "log-level=error:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
            str(path),
        ]
    )


class _IdentityVae:
    """Fast deterministic fake AutoencoderKL preserving float32 [-1, 1] tensors."""

    def __init__(self) -> None:
        import torch

        self.dtype = torch.float32

    def encode(self, tensor):
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: tensor))

    def decode(self, tensor):
        return SimpleNamespace(sample=tensor)


class TestEndToEnd10BitHdrInvisible:
    def test_video_invisible_preserves_10bit_hdr_end_to_end_with_temporal_stabilize(self, tmp_path: Path) -> None:
        torch = pytest.importorskip("torch")
        from remove_ai_watermarks.video_invisible import (
            DEFAULT_VIDEO_SYNTHID_VAE,
            VIDEO_SYNTHID_VAE_SCALING_FACTOR,
            VideoVaeRuntime,
            regenerate_video_candidate,
        )

        source = tmp_path / "hdr10_src.mp4"
        output = tmp_path / "hdr10_clean.mp4"
        _make_hdr10_clip(source)

        runtime = VideoVaeRuntime(
            model=DEFAULT_VIDEO_SYNTHID_VAE,
            requested_device="cpu",
            resolved_device="cpu",
            vae=_IdentityVae(),
            scaling_factor=VIDEO_SYNTHID_VAE_SCALING_FACTOR,
        )
        assert torch is not None
        metrics = regenerate_video_candidate(
            source,
            output,
            noise_std=0.02,
            long_side=64,
            fps=10.0,
            batch_size=2,
            seed=7,
            device="cpu",
            runtime=runtime,
            temporal_stabilize=True,
        )

        assert output.is_file()
        assert metrics.frames >= 4
        assert metrics.psnr_db > 15.0
        stream = _probe_stream(output)
        assert stream["pix_fmt"] == "yuv420p10le"
        assert stream["color_transfer"] == "smpte2084"
        assert stream["color_primaries"] == "bt2020"
        assert stream["color_space"] == "bt2020nc"

    def test_video_invisible_still_rejects_unrepresentable_hdr_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("torch")
        from remove_ai_watermarks import video_invisible
        from remove_ai_watermarks.video_invisible import (
            DEFAULT_VIDEO_SYNTHID_VAE,
            VIDEO_SYNTHID_VAE_SCALING_FACTOR,
            VideoVaeRuntime,
            regenerate_video_candidate,
        )

        source = tmp_path / "hdr_unrep.mp4"
        _make_hdr10_clip(source)
        monkeypatch.setattr(
            video_invisible,
            "probe_video_encode_profile",
            lambda _path: VideoEncodeProfile(
                pixel_format="yuv420p",
                color_transfer="smpte2084",
                source_pixel_format="gbrp12le",
                component_depth=12,
            ),
        )
        runtime = VideoVaeRuntime(
            model=DEFAULT_VIDEO_SYNTHID_VAE,
            requested_device="cpu",
            resolved_device="cpu",
            vae=_IdentityVae(),
            scaling_factor=VIDEO_SYNTHID_VAE_SCALING_FACTOR,
        )
        with pytest.raises(RuntimeError, match="refusing to silently reduce gbrp12le"):
            regenerate_video_candidate(source, tmp_path / "out.mp4", device="cpu", runtime=runtime)

    def test_wan21_preserves_10bit_hdr_and_timestamps(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        from remove_ai_watermarks.video_wan import (
            WAN21_MODEL_ID,
            WAN21_MODEL_REVISION,
            VideoWanRuntime,
            can_run_wan_on_source,
            regenerate_wan_experimental,
        )

        source = tmp_path / "hdr10_wan.mp4"
        output = tmp_path / "hdr10_wan_out.mp4"
        _make_hdr10_clip(source, width=64, height=64, duration=0.5, fps=10)

        def _fake_pipe(**kwargs):
            frames = kwargs["video"]
            arr = np.stack(
                [
                    np.asarray(img, dtype=np.float32)
                    if isinstance(img, np.ndarray) and np.issubdtype(img.dtype, np.floating)
                    else np.asarray(img, dtype=np.float32) / 255.0
                    for img in frames
                ],
                axis=0,
            )
            return SimpleNamespace(frames=[arr])

        runtime = VideoWanRuntime(
            model=WAN21_MODEL_ID,
            revision=WAN21_MODEL_REVISION,
            resolved_device="cpu",
            pipe=_fake_pipe,
        )
        assert can_run_wan_on_source(source, device="cuda", runtime=runtime)
        metrics = regenerate_wan_experimental(
            source,
            output,
            profile="wan21-synthid-experimental",
            device="cuda",
            quality="delivery",
            seed=3,
            inference_steps=1,
            runtime=runtime,
        )
        assert output.is_file()
        assert metrics.frames == 5
        assert metrics.psnr_db > 25.0
        stream = _probe_stream(output)
        assert stream["pix_fmt"] == "yuv420p10le"
        assert stream["color_transfer"] == "smpte2084"
        assert stream["color_primaries"] == "bt2020"

    def test_stabilize_regenerated_frames_supports_uint16_hdr(self) -> None:
        from remove_ai_watermarks.video_temporal import stabilize_regenerated_frames

        rng = np.random.default_rng(101)
        base_u8 = cv2.GaussianBlur(rng.integers(32, 224, (64, 80, 3), dtype=np.uint8), (5, 5), 0)
        base_u16 = base_u8.astype(np.uint16) * 257
        sources = [
            cv2.warpAffine(
                base_u16,
                np.float32(((1, 0, idx * 1.5), (0, 1, idx * 0.5))),
                (80, 64),
                borderMode=cv2.BORDER_REFLECT,
            )
            for idx in range(5)
        ]
        noisy = [
            np.clip(frm.astype(np.float32) + rng.normal(0.0, 2600.0, frm.shape), 0, 65535).astype(np.uint16)
            for frm in sources
        ]
        stabilized = stabilize_regenerated_frames(sources, noisy)
        assert all(f.dtype == np.uint16 for f in stabilized)
        raw_mse = float(
            np.mean([(a.astype(np.float32) - b.astype(np.float32)) ** 2 for a, b in zip(noisy, sources, strict=True)])
        )
        stab_mse = float(
            np.mean(
                [(a.astype(np.float32) - b.astype(np.float32)) ** 2 for a, b in zip(stabilized, sources, strict=True)]
            )
        )
        assert stab_mse < raw_mse * 0.75


class TestMultiFrameTemporalInpainting:
    def test_bidirectional_background_locked_fill_beats_independent_and_rejects_scene_cut(self) -> None:
        from remove_ai_watermarks.video_temporal import (
            stabilize_filled_frame,
            stabilize_filled_frame_bidirectional,
        )

        # Textured panning background with a STATIC opaque white watermark in the exact same box
        height, width = 128, 160
        cy, cx = np.mgrid[0:height, 0 : width + 24].astype(np.float32)
        b = 128.0 + 65.0 * np.sin(cx * 0.35) * np.cos(cy * 0.32)
        g = 128.0 + 60.0 * np.cos(cx * 0.28 + cy * 0.25)
        r = 128.0 + 55.0 * np.sin((cx - cy) * 0.30)
        canvas = np.clip(np.stack([b, g, r], axis=2), 16, 240).astype(np.uint8)
        clean = [canvas[:, idx * 2 : idx * 2 + width].copy() for idx in range(3)]

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[42:82, 58:102] = 255
        hole = mask > 0

        # Static opaque watermark painted on all 3 sources
        sources = [frm.copy() for frm in clean]
        for s in sources:
            s[hole] = 255

        # Independent noisy single-frame fills on all 3 frames
        rng = np.random.default_rng(77)
        fills = []
        for frm in clean:
            f = frm.copy()
            f[hole] = np.clip(f[hole].astype(np.float32) + rng.normal(0.0, 22.0, f[hole].shape), 0, 255).astype(
                np.uint8
            )
            fills.append(f)

        indep_mse = float(np.mean((fills[1][hole].astype(np.float32) - clean[1][hole]) ** 2))
        uni = stabilize_filled_frame(sources[0], fills[0], mask, sources[1], fills[1], mask)
        bi = stabilize_filled_frame_bidirectional(
            sources[1],
            fills[1],
            mask,
            previous=(sources[0], fills[0], mask),
            nxt=(sources[2], fills[2], mask),
        )
        uni_mse = float(np.mean((uni[hole].astype(np.float32) - clean[1][hole]) ** 2))
        bi_mse = float(np.mean((bi[hole].astype(np.float32) - clean[1][hole]) ** 2))

        assert uni_mse < indep_mse * 0.70
        assert bi_mse < uni_mse
        assert bi_mse < indep_mse * 0.55
        assert np.array_equal(bi[~hole], fills[1][~hole])

        # Scene cut on BOTH neighbors must leave current fill untouched
        cut = rng.integers(0, 40, clean[1].shape, dtype=np.uint8)
        cut_out = stabilize_filled_frame_bidirectional(
            sources[1],
            fills[1],
            mask,
            previous=(cut, cut, mask),
            nxt=(cut, cut, mask),
        )
        assert np.array_equal(cut_out, fills[1])

    def test_transitive_clean_plate_propagation_recovers_wide_logo_on_slow_pan(self, tmp_path: Path) -> None:
        from remove_ai_watermarks.video_visible import FrameLocalization, VideoScan, iter_clean_video_frames

        ffmpeg = require_video_tool("ffmpeg")
        height, width, n_frames = 128, 160, 20
        cy, cx = np.mgrid[0:height, 0 : width + 60].astype(np.float32)
        b = 128.0 + 70.0 * np.sin(cx * 0.40) * np.cos(cy * 0.35)
        g = 128.0 + 65.0 * np.cos(cx * 0.31 + cy * 0.29)
        r = 128.0 + 70.0 * (((cx.astype(int) // 6) ^ (cy.astype(int) // 6)) * 2 - 1) * 0.6
        canvas = np.clip(np.stack([b, g, r], axis=2), 16, 240).astype(np.uint8)
        clean_frames = [canvas[:, idx * 2 : idx * 2 + width].copy() for idx in range(n_frames)]

        # Static 34x26 opaque logo; at 2 px/frame pan it takes 17 frames (> donor_radius=12) to clear fully
        box = (60, 48, 34, 26)
        x0, y0, bw, bh = box
        src_path = tmp_path / "slow_pan.mp4"
        proc = subprocess.Popen(  # noqa: S603
            [
                ffmpeg,
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                "10",
                "-i",
                "pipe:0",
                "-c:v",
                "libx264",
                "-crf",
                "0",
                "-pix_fmt",
                "yuv444p",
                str(src_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None
        for frm in clean_frames:
            wm = frm.copy()
            wm[y0 : y0 + bh, x0 : x0 + bw] = 250
            proc.stdin.write(wm.tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        assert proc.wait() == 0, stderr

        scan = VideoScan(
            width=width,
            height=height,
            fps=10.0,
            detections=tuple(FrameLocalization(i, 1.0, box) for i in range(n_frames)),
        )
        regions = [box for _ in range(n_frames)]

        no_temp = [
            f
            for f, _ in iter_clean_video_frames(
                src_path, scan, regions, backend="cv2", temporal_consistency=False, dilate_px=2
            )
        ]
        with_temp = [
            f
            for f, _ in iter_clean_video_frames(
                src_path, scan, regions, backend="cv2", temporal_consistency=True, dilate_px=2
            )
        ]
        assert len(with_temp) == n_frames
        # Evaluate the middle frame where slow-pan donors + transitive clean-plate propagation operate
        mid = n_frames // 2
        roi_slice = (slice(y0, y0 + bh), slice(x0, x0 + bw))
        mse_no_temp = float(np.mean((no_temp[mid][roi_slice].astype(np.float32) - clean_frames[mid][roi_slice]) ** 2))
        mse_temp = float(np.mean((with_temp[mid][roi_slice].astype(np.float32) - clean_frames[mid][roi_slice]) ** 2))
        assert mse_temp < mse_no_temp * 0.45
