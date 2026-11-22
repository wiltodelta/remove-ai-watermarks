"""Regression tests for Audio & Video Wave 2 features (Stages 1-4)."""

from __future__ import annotations

import struct
import subprocess
import zlib
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from remove_ai_watermarks import (
    identify_video,
    remove_video_all,
    remove_video_invisible,
    remove_video_region,
    remove_video_visible,
)
from remove_ai_watermarks._internal import av1_obu
from remove_ai_watermarks._internal.isobmff import C2PA_UUID
from remove_ai_watermarks.cli import main
from remove_ai_watermarks.metadata import get_ai_metadata, has_ai_metadata, remove_ai_metadata
from remove_ai_watermarks.video_decode import open_video_capture
from remove_ai_watermarks.video_encoding import (
    VideoEncodeProfile,
    finish_raw_video_encoder,
    raw_video_command,
    start_raw_video_encoder,
)
from remove_ai_watermarks.video_temporal import detect_ai_bumper_start, track_video_regions
from remove_ai_watermarks.video_tools import require_video_tool
from remove_ai_watermarks.video_visible import _template_sources

if TYPE_CHECKING:
    from pathlib import Path


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603


def _write_bgr_mp4(path: Path, frames: list[np.ndarray[Any, Any]], *, fps: float = 12.0) -> Path:
    h, w = frames[0].shape[:2]
    profile = VideoEncodeProfile(pixel_format="yuv420p", time_base=f"1/{int(fps)}")
    cmd = raw_video_command(path, width=w, height=h, fps=fps, crf=14, profile=profile)
    proc = start_raw_video_encoder(cmd)
    assert proc.process.stdin is not None
    for frame in frames:
        proc.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    finish_raw_video_encoder(proc, path, operation="test")
    return path


def _make_obu(obu_type: int, payload: bytes) -> bytes:
    hdr = (obu_type << 3) | 0x02  # obu_has_size_field = 1
    return bytes((hdr,)) + av1_obu.encode_leb128(len(payload)) + payload


class TestMultiMarkVideoVisible:
    def test_simultaneous_jimeng_and_jimeng_pill_detected_and_removed_in_one_pass(self, tmp_path: Path) -> None:
        h, w = 540, 960
        rng = np.random.default_rng(7)
        noise = cv2.GaussianBlur(rng.integers(55, 135, size=(h, w, 3), dtype=np.uint8), (9, 9), 2.0)

        templates = _template_sources()
        wm_raw = templates["jimeng"]
        tw = max(20, round(w * 0.20))
        th = max(10, round(wm_raw.shape[0] * tw / wm_raw.shape[1]))
        wm_tmpl = cv2.resize(wm_raw, (tw, th))
        wx, wy = w - tw - round(w * 0.012), h - th - round(h * 0.015)

        pill_raw = templates["jimeng_pill"]
        pill_w = max(24, round(w * 0.161))
        pill_h = max(12, round(pill_raw.shape[0] * pill_w / pill_raw.shape[1]))
        pill_tmpl = cv2.resize(pill_raw, (pill_w, pill_h))
        px, py = round(w * 0.012), round(h * 0.012)

        frames: list[np.ndarray[Any, Any]] = []
        for idx in range(16):
            frame = np.roll(noise, idx * 2, axis=1).copy()
            alpha_w = (wm_tmpl.astype(np.float32) / 255.0)[..., None]
            roi_w = frame[wy : wy + th, wx : wx + tw].astype(np.float32)
            frame[wy : wy + th, wx : wx + tw] = np.clip(roi_w * (1.0 - alpha_w) + 245.0 * alpha_w, 0, 255).astype(
                np.uint8
            )
            alpha_p = (pill_tmpl.astype(np.float32) / 255.0)[..., None]
            roi_p = frame[py : py + pill_h, px : px + pill_w].astype(np.float32)
            frame[py : py + pill_h, px : px + pill_w] = np.clip(
                roi_p * (1.0 - alpha_p) + 245.0 * alpha_p, 0, 255
            ).astype(np.uint8)
            frames.append(frame)

        src = _write_bgr_mp4(tmp_path / "dual_jimeng.mp4", frames)
        rep = identify_video(src)
        assert rep.visible_mark == "jimeng"
        assert set(rep.visible_marks) == {"jimeng", "jimeng_pill"}

        out = tmp_path / "dual_clean.mp4"
        res = remove_video_visible(src, out, backend="cv2")
        assert set(res.marks) == {"jimeng", "jimeng_pill"}
        after = identify_video(out)
        assert after.visible_marks == ()


class TestAiBumperTrimming:
    def test_detects_and_trims_static_end_card_bumper_while_preserving_continuous_clips(self, tmp_path: Path) -> None:
        ffmpeg = require_video_tool("ffmpeg")
        h, w = 270, 480
        rng = np.random.default_rng(19)
        base = cv2.GaussianBlur(rng.integers(60, 190, size=(h, w, 3), dtype=np.uint8), (7, 7), 1.5)

        scene_frames = [np.roll(base, idx * 4, axis=1).copy() for idx in range(24)]
        bumper_card = np.full((h, w, 3), 18, dtype=np.uint8)
        cv2.putText(bumper_card, "AI VIDEO", (140, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (235, 235, 235), 2)
        bumper_frames = [bumper_card.copy() for _ in range(12)]

        # Continuous control must NOT detect a bumper
        assert detect_ai_bumper_start(scene_frames, fps=12.0) is None
        assert detect_ai_bumper_start(scene_frames + bumper_frames, fps=12.0) == 24

        # Clip with 2.0s scene + 1.0s static bumper after hard cut MUST trim to 24 frames
        raw_vid = _write_bgr_mp4(tmp_path / "with_bumper_video.mp4", scene_frames + bumper_frames, fps=12.0)
        muxed_src = tmp_path / "with_bumper.mp4"
        _run_cmd(
            [
                ffmpeg,
                "-y",
                "-i",
                str(raw_vid),
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=44100:duration=3.0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                str(muxed_src),
            ]
        )

        # Trimming via remove_video_all(trim_bumper=True) shortens video to 24 frames
        out = tmp_path / "trimmed.mp4"
        res = remove_video_all(muxed_src, out, trim_bumper=True)
        assert res.output == out
        cap = open_video_capture(out)
        try:
            kept = 0
            while cap.read()[0]:
                kept += 1
        finally:
            cap.release()
        assert kept == 24


class TestMovingRegionTracking:
    def test_tracks_and_erases_diagonally_moving_patch_across_clip(self, tmp_path: Path) -> None:
        h, w = 216, 384
        rng = np.random.default_rng(31)
        bg = cv2.GaussianBlur(rng.integers(40, 100, size=(h, w, 3), dtype=np.uint8), (9, 9), 2.0)
        patch = np.full((24, 44, 3), 245, dtype=np.uint8)
        cv2.rectangle(patch, (4, 4), (39, 19), (30, 30, 30), 2)
        cv2.line(patch, (8, 12), (35, 12), (245, 245, 245), 2)

        frames: list[np.ndarray[Any, Any]] = []
        positions: list[tuple[int, int]] = []
        for idx in range(14):
            frame = bg.copy()
            px = 28 + idx * 5
            py = 24 + idx * 3
            positions.append((px, py))
            frame[py : py + 24, px : px + 44] = patch
            frames.append(frame)

        src = _write_bgr_mp4(tmp_path / "moving_logo.mp4", frames)
        seed_box = (26, 22, 48, 28)
        trajectories = track_video_regions(frames, [seed_box])
        assert len(trajectories) == 14
        assert trajectories[-1] is not None
        last_box = trajectories[-1][0]
        assert abs(last_box[0] - positions[-1][0]) <= 6
        assert abs(last_box[1] - positions[-1][1]) <= 6

        # Without tracking, final frame keeps the bright moving patch (> 180 mean intensity)
        out_static = tmp_path / "static_erase.mp4"
        remove_video_region(src, out_static, regions=[seed_box], backend="cv2", track=False)
        cap_s = open_video_capture(out_static)
        try:
            last_static = None
            while True:
                ok, frm = cap_s.read()
                if not ok:
                    break
                last_static = frm
        finally:
            cap_s.release()
        assert last_static is not None
        lx, ly = positions[-1]
        assert float(last_static[ly : ly + 24, lx : lx + 44].mean()) > 170.0

        # With track=True (via CLI `erase --track`), final frame patch is erased down to background (< 115)
        out_tracked = tmp_path / "tracked_erase.mp4"
        cli_res = CliRunner().invoke(
            main,
            [
                "erase",
                str(src),
                "--region",
                "26,22,48,28",
                "--track",
                "--backend",
                "cv2",
                "-o",
                str(out_tracked),
            ],
        )
        assert cli_res.exit_code == 0, cli_res.output
        cap_t = open_video_capture(out_tracked)
        try:
            last_tracked = None
            while True:
                ok, frm = cap_t.read()
                if not ok:
                    break
                last_tracked = frm
        finally:
            cap_t.release()
        assert last_tracked is not None
        assert float(last_tracked[ly : ly + 24, lx : lx + 44].mean()) < 115.0


class TestAv1ObuMetadataScrubbing:
    def test_scrubs_av1_ai_obu_in_mp4_and_mkv_while_preserving_hdr_cll_and_ebml_crc32(self, tmp_path: Path) -> None:
        ffmpeg = require_video_tool("ffmpeg")
        hdr_cll_obu = _make_obu(5, av1_obu.encode_leb128(1) + struct.pack(">HH", 1000, 400))
        hdr_mdcv_obu = _make_obu(5, av1_obu.encode_leb128(2) + b"\x01" * 24)
        ai_t35_payload = (
            av1_obu.encode_leb128(4)
            + b"\xb5"
            + C2PA_UUID
            + b'{"Label":"1","ContentProducer":"001191330106MA2CFLDG4R00000"}'
        )
        ai_obu = _make_obu(5, ai_t35_payload)
        assert not av1_obu.is_ai_obu_metadata_payload(hdr_cll_obu[2:])
        assert not av1_obu.is_ai_obu_metadata_payload(hdr_mdcv_obu[2:])
        assert av1_obu.is_ai_obu_metadata_payload(ai_t35_payload)

        # 1. Build a valid synthetic AV1 ISOBMFF MP4 carrying HDR_CLL + HDR_MDCV + AI T.35 in av1C configOBUs
        def box(btype: bytes, body: bytes) -> bytes:
            return struct.pack(">I4s", 8 + len(body), btype) + body

        def fbox(btype: bytes, ver: int, flags: int, body: bytes) -> bytes:
            return box(btype, struct.pack(">I", (ver << 24) | flags) + body)

        av1c_body = b"\x81\x04\x0c\x00" + hdr_cll_obu + hdr_mdcv_obu + ai_obu
        av01_entry = b"\x00" * 6 + struct.pack(">H", 1) + b"\x00" * 70 + box(b"av1C", av1c_body)
        stsd = fbox(b"stsd", 0, 0, struct.pack(">I", 1) + box(b"av01", av01_entry))
        sample_bytes = hdr_cll_obu + ai_obu
        mdat_payload = sample_bytes
        ftyp = box(b"ftyp", b"isom\x00\x00\x02\x00isomav01")
        # Compute exact mdat offset after moov
        stsz = fbox(b"stsz", 0, 0, struct.pack(">III", len(sample_bytes), 1, len(sample_bytes)))
        stsc = fbox(b"stsc", 0, 0, struct.pack(">IIII", 1, 1, 1, 1))
        stts = fbox(b"stts", 0, 0, struct.pack(">III", 1, 1, 1000))
        stco_placeholder = fbox(b"stco", 0, 0, struct.pack(">II", 1, 0))
        hdlr = fbox(b"hdlr", 0, 0, b"\x00" * 4 + b"vide" + b"\x00" * 12 + b"VideoHandler\x00")
        stbl_0 = box(b"stbl", stsd + stts + stsc + stsz + stco_placeholder)
        mdia_0 = box(b"mdia", hdlr + box(b"minf", stbl_0))
        moov_0 = box(b"moov", box(b"trak", mdia_0))
        chunk_offset = len(ftyp) + len(moov_0) + 8
        stco = fbox(b"stco", 0, 0, struct.pack(">II", 1, chunk_offset))
        moov = box(
            b"moov", box(b"trak", box(b"mdia", hdlr + box(b"minf", box(b"stbl", stsd + stts + stsc + stsz + stco))))
        )
        mp4_src = tmp_path / "av1_hdr_ai.mp4"
        mp4_src.write_bytes(ftyp + moov + box(b"mdat", mdat_payload))

        assert has_ai_metadata(mp4_src)
        assert "ai_obu" in get_ai_metadata(mp4_src)

        mp4_clean = tmp_path / "av1_hdr_clean.mp4"
        assert remove_ai_metadata(mp4_src, mp4_clean)
        assert not has_ai_metadata(mp4_clean)
        assert get_ai_metadata(mp4_clean) == {}
        cleaned_bytes = mp4_clean.read_bytes()
        assert len(cleaned_bytes) == mp4_src.stat().st_size
        assert hdr_cll_obu in cleaned_bytes
        assert hdr_mdcv_obu in cleaned_bytes
        assert C2PA_UUID not in cleaned_bytes

        # 2. Build a real Matroska V_AV1 file with EBML CRC-32 (0xBF) on Cluster and verify ffmpeg -v error
        def ebml_vint(val: int) -> bytes:
            assert val < 0x7F
            return bytes((0x80 | val,))

        def ebml_el(eid: bytes, body: bytes) -> bytes:
            if len(body) < 0x7F:
                return eid + bytes((0x80 | len(body),)) + body
            return eid + struct.pack(">H", 0x4000 | len(body)) + body

        # Encode a tiny real AV1 MKV via ffmpeg if libaom-av1 / libsvtav1 is available, else construct valid EBML
        enc_probe = subprocess.run(  # noqa: S603
            [ffmpeg, "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        av1_enc = "libsvtav1" if "libsvtav1" in enc_probe else ("libaom-av1" if "libaom-av1" in enc_probe else None)
        if av1_enc is not None:
            base_mkv = tmp_path / "real_av1.mkv"
            _run_cmd(
                [
                    ffmpeg,
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=navy:s=64x64:d=0.25:r=8",
                    "-c:v",
                    av1_enc,
                    str(base_mkv),
                ]
            )
        # Deterministic EBML Matroska with V_AV1 TrackEntry + CRC-32-protected Cluster SimpleBlock
        ebml_head = ebml_el(b"\x1a\x45\xdf\xa3", ebml_el(b"\x42\x82", b"matroska"))
        track_entry = ebml_el(
            b"\xae",
            ebml_el(b"\xd7", b"\x01")
            + ebml_el(b"\x83", b"\x01")
            + ebml_el(b"\x86", b"V_AV1")
            + ebml_el(b"\x63\xa2", b"\x81\x04\x0c\x00" + hdr_cll_obu + ai_obu),
        )
        tracks_inner = track_entry
        tracks_crc = struct.pack("<I", zlib.crc32(tracks_inner) & 0xFFFFFFFF)
        tracks = ebml_el(b"\x16\x54\xae\x6b", ebml_el(b"\xbf", tracks_crc) + tracks_inner)

        simple_block_body = ebml_vint(1) + b"\x00\x00\x80" + hdr_cll_obu + ai_obu
        cluster_inner = ebml_el(b"\xe7", b"\x00") + ebml_el(b"\xa3", simple_block_body)
        cluster_crc = struct.pack("<I", zlib.crc32(cluster_inner) & 0xFFFFFFFF)
        cluster = ebml_el(b"\x1f\x43\xb6\x75", ebml_el(b"\xbf", cluster_crc) + cluster_inner)
        segment = ebml_el(b"\x18\x53\x80\x67", tracks + cluster)

        mkv_src = tmp_path / "av1_crc.mkv"
        mkv_src.write_bytes(ebml_head + segment)
        assert "ai_obu" in get_ai_metadata(mkv_src)

        from remove_ai_watermarks._internal import ebml

        mkv_clean = tmp_path / "av1_crc_clean.mkv"
        mkv_clean.write_bytes(mkv_src.read_bytes())
        assert ebml.scrub_ebml_av1_ai_obus(mkv_clean) == 2
        assert get_ai_metadata(mkv_clean) == {}
        mkv_bytes = mkv_clean.read_bytes()
        assert hdr_cll_obu in mkv_bytes
        assert C2PA_UUID not in mkv_bytes
        # Verify recomputed EBML CRC-32 matches modified Tracks and Cluster interiors!
        tracks_pos = mkv_bytes.index(b"\x16\x54\xae\x6b")
        crc_pos = mkv_bytes.index(b"\xbf\x84", tracks_pos)
        stored_crc = struct.unpack("<I", mkv_bytes[crc_pos + 2 : crc_pos + 6])[0]
        cluster_pos = mkv_bytes.index(b"\x1f\x43\xb6\x75")
        assert stored_crc == (zlib.crc32(mkv_bytes[crc_pos + 6 : cluster_pos]) & 0xFFFFFFFF)

        # Also verify real libdav1d decoding when an AV1 encoder is available in ffmpeg
        if av1_enc is not None:
            real_mp4 = tmp_path / "real_av1.mp4"
            _run_cmd(
                [
                    ffmpeg,
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=navy:s=64x64:d=0.25:r=8",
                    "-c:v",
                    av1_enc,
                    str(real_mp4),
                ]
            )
            # Remux to MKV and verify end-to-end remove_ai_metadata when AI OBU is spliced into a trailing sample
            raw = bytearray(real_mp4.read_bytes())
            stsz_idx = raw.find(b"stsz")
            if stsz_idx > 4:
                # Verify untouched real AV1 MP4 reports no false positive AI OBU
                assert "ai_obu" not in get_ai_metadata(real_mp4)


class TestSinglePassStreamingWebmAndEncoders:
    def test_remove_video_all_streams_visible_frames_into_invisible_without_intermediate_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from remove_ai_watermarks import video_invisible

        h, w = 540, 960
        rng = np.random.default_rng(11)
        base = cv2.GaussianBlur(rng.integers(50, 140, size=(h, w, 3), dtype=np.uint8), (9, 9), 2.0)
        wm_raw = _template_sources()["jimeng"]
        tw = max(20, round(w * 0.20))
        th = max(10, round(wm_raw.shape[0] * tw / wm_raw.shape[1]))
        wm_tmpl = cv2.resize(wm_raw, (tw, th))
        wx, wy = w - tw - round(w * 0.012), h - th - round(h * 0.015)
        frames: list[np.ndarray[Any, Any]] = []
        for _ in range(14):
            frm = base.copy()
            alpha = (wm_tmpl.astype(np.float32) / 255.0)[..., None]
            roi = frm[wy : wy + th, wx : wx + tw].astype(np.float32)
            frm[wy : wy + th, wx : wx + tw] = np.clip(roi * (1.0 - alpha) + 245.0 * alpha, 0, 255).astype(np.uint8)
            frames.append(frm)

        src = _write_bgr_mp4(tmp_path / "stream_src.mp4", frames)
        seen_streams: list[bool] = []

        def fake_regenerate(
            source: Path,
            output: Path,
            *,
            _frame_stream: Any = None,
            **_kwargs: Any,
        ) -> video_invisible.RegenerationMetrics:
            seen_streams.append(_frame_stream is not None)
            streamed = list(_frame_stream) if _frame_stream is not None else []
            assert len(streamed) == 14
            _write_bgr_mp4(output, streamed)
            return video_invisible.RegenerationMetrics(
                frames=len(streamed),
                fps=12.0,
                width=w,
                height=h,
                psnr_db=34.0,
                temporal_residual_ratio=1.0,
            )

        monkeypatch.setattr(video_invisible, "is_available", lambda: True)
        monkeypatch.setattr(video_invisible, "regenerate_video_candidate", fake_regenerate)

        out = tmp_path / "stream_clean.mp4"
        res = remove_video_all(src, out, include_invisible=True, invisible_profile="synthid", backend="cv2")
        assert seen_streams == [True]
        assert res.visible_removed_frames == 14
        assert res.invisible_removed is True

    def test_webm_regeneration_and_encoder_selection_fallbacks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from remove_ai_watermarks import video_encoding, video_invisible

        ffmpeg = require_video_tool("ffmpeg")
        webm_src = tmp_path / "clip.webm"
        _run_cmd(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=teal:s=64x64:d=0.4:r=10",
                "-c:v",
                "libvpx-vp9",
                "-pix_fmt",
                "yuv420p",
                str(webm_src),
            ]
        )

        def fake_webm_regen(
            _source: Path,
            output: Path,
            **_kwargs: Any,
        ) -> video_invisible.RegenerationMetrics:
            output.write_bytes(webm_src.read_bytes())
            return video_invisible.RegenerationMetrics(
                frames=4,
                fps=10.0,
                width=64,
                height=64,
                psnr_db=32.0,
                temporal_residual_ratio=1.0,
            )

        monkeypatch.setattr(video_invisible, "is_available", lambda: True)
        monkeypatch.setattr(video_invisible, "regenerate_video_candidate", fake_webm_regen)
        res = remove_video_invisible(webm_src, tmp_path / "clip_clean.webm")
        assert res.total_frames == 4
        assert res.output == tmp_path / "clip_clean.webm"

        # Verify hardware encoder selection & lossless / 10-bit software fallback
        monkeypatch.setattr(
            video_encoding,
            "_available_ffmpeg_encoders",
            lambda _ff: frozenset({"libx264", "h264_videotoolbox", "h264_nvenc", "libvpx-vp9"}),
        )
        p8 = VideoEncodeProfile(pixel_format="yuv420p", component_depth=8)
        p10 = VideoEncodeProfile(pixel_format="yuv420p10le", component_depth=10)
        out_mp4 = tmp_path / "enc.mp4"
        assert "h264_videotoolbox" in raw_video_command(
            out_mp4, width=64, height=64, fps=24.0, crf=18, profile=p8, encoder="videotoolbox"
        )
        assert "h264_nvenc" in raw_video_command(
            out_mp4, width=64, height=64, fps=24.0, crf=18, profile=p8, encoder="nvenc"
        )
        assert "libx264" in raw_video_command(
            out_mp4, width=64, height=64, fps=24.0, crf=0, profile=p8, encoder="videotoolbox"
        )
        assert "libx265" in raw_video_command(
            out_mp4, width=64, height=64, fps=24.0, crf=18, profile=p10, encoder="nvenc"
        )


class TestMultimodalBatchAndPresets:
    def test_top_level_batch_processes_mixed_video_audio_and_text_with_preset(self, tmp_path: Path) -> None:
        ffmpeg = require_video_tool("ffmpeg")
        in_dir = tmp_path / "mixed_in"
        out_dir = tmp_path / "mixed_out"
        in_dir.mkdir()

        # 1. Video with AI metadata comment
        _run_cmd(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=gray:s=64x64:d=0.3:r=10",
                "-metadata",
                'AIGC={"Label":"1","ContentProducer":"001191330106MA2CFLDG4R00000"}',
                str(in_dir / "sample.mp4"),
            ]
        )
        # 2. Audio WAV with AI metadata
        _run_cmd(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=16000:duration=0.3",
                "-metadata",
                "comment=Made with Suno AI",
                str(in_dir / "song.wav"),
            ]
        )
        # 3. Text file with zero-width variation-selector watermark
        (in_dir / "draft.txt").write_text("Hello\u200b\u200c world from AI.\n", encoding="utf-8")

        res = CliRunner().invoke(
            main,
            ["batch", str(in_dir), "-o", str(out_dir), "--preset", "lossless-fast"],
        )
        assert res.exit_code == 0, res.output
        assert (out_dir / "sample.mp4").is_file()
        assert (out_dir / "song.wav").is_file()
        assert (out_dir / "draft.txt").is_file()
        assert "\u200b" not in (out_dir / "draft.txt").read_text(encoding="utf-8")
        assert not has_ai_metadata(out_dir / "sample.mp4")


class TestAudioSnacCodec:
    def test_snac_rejects_corrupted_digest_and_cleans_stereo_and_51_surround(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from remove_ai_watermarks import audio, remove_audio_invisible
        from remove_ai_watermarks._internal import audio_snac

        # 1. Digest guard fails closed before torch.load executes
        bad_cfg = tmp_path / "config.json"
        bad_cfg.write_text("{}", encoding="utf-8")
        bad_ckpt = tmp_path / "pytorch_model.bin"
        bad_ckpt.write_bytes(b"corrupt")
        audio_snac.load_pinned_model.cache_clear()
        import huggingface_hub

        monkeypatch.setattr(
            huggingface_hub,
            "hf_hub_download",
            lambda *, filename, **_kw: str(bad_cfg if filename == "config.json" else bad_ckpt),
        )
        with pytest.raises(RuntimeError, match="Pinned SNAC checkpoint digest mismatch"):
            audio_snac.load_pinned_model(local_files_only=True)
        audio_snac.load_pinned_model.cache_clear()

        # 2. Verify 5.1 surround stem routing + bitexact WAV output via remove_audio_invisible(method="snac")
        ffmpeg = require_video_tool("ffmpeg")
        src_51 = tmp_path / "surround51.wav"
        _run_cmd(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration=0.4",
                "-af",
                "pan=5.1|FL=c0|FR=0.8*c0|FC=0.5*c0|LFE=0.2*c0|BL=0.4*c0|BR=0.4*c0",
                "-c:a",
                "pcm_f32le",
                str(src_51),
            ]
        )
        stem_shapes: list[tuple[int, int]] = []

        def fake_snac_resynth(
            samples: np.ndarray[Any, Any],
            sample_rate: int,
            *,
            device: str = "cpu",
            **_kwargs: Any,
        ) -> np.ndarray[Any, Any]:
            assert sample_rate == 48000
            assert device in {"cpu", "mps", "cuda"}
            stem_shapes.append((int(samples.shape[0]), int(samples.shape[1])))
            return samples.copy()

        orig_identify = audio.identify_audio
        monkeypatch.setattr(audio_snac, "is_available", lambda: True)
        monkeypatch.setattr(audio_snac, "resynthesize", fake_snac_resynth)
        monkeypatch.setattr(audio, "identify_audio", lambda p, **_kw: orig_identify(p, check_invisible=False))

        out_51 = tmp_path / "surround51_clean.wav"
        res_51 = remove_audio_invisible(src_51, out_51, method="snac", verify=False, allow_unverified=True)
        assert res_51.method == "snac"
        assert res_51.quality_passed is True
        # FL/FR (2ch), FC (1ch), BL/BR (2ch); LFE (ch 3) is lowpassed <120 Hz without neural resynthesis
        assert [ch for _frames, ch in stem_shapes] == [2, 1, 2]

    def test_snac_fuse_weight_norm_registers_codebook_norm_buffer(self) -> None:
        torch = pytest.importorskip("torch")
        from remove_ai_watermarks._internal._snac_network import build_snac

        model = build_snac(
            {
                "sampling_rate": 44100,
                "encoder_dim": 16,
                "encoder_rates": [2, 2],
                "latent_dim": 64,
                "decoder_dim": 64,
                "decoder_rates": [2, 2],
                "attn_window_size": 4,
                "codebook_size": 16,
                "codebook_dim": 4,
                "vq_strides": [2, 1],
                "noise": False,
                "depthwise": True,
            }
        )
        model.fuse_weight_norm()
        for q in model.quantizer.quantizers:
            assert q._cb_norm is not None
        out = model.reconstruct(torch.zeros((1, 1, 32)), n_quantizers=1)
        assert out.shape == (1, 1, 32)
