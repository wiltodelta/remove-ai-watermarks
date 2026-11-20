"""Regression tests for the Audio & Video Market-Leadership Roadmap (Phases 0-5a)."""

from __future__ import annotations

import json
import struct
import subprocess
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from remove_ai_watermarks import (
    detect_video_watermark_presence,
    remove_audio_invisible,
    remove_video_all,
    remove_video_region,
)
from remove_ai_watermarks._internal import isobmff
from remove_ai_watermarks._internal.audio_io import probe_audio, read_audio, write_audio
from remove_ai_watermarks._internal.isobmff import C2PA_UUID
from remove_ai_watermarks.cli import main
from remove_ai_watermarks.metadata import get_ai_metadata, remove_ai_metadata
from remove_ai_watermarks.video_temporal import estimate_clip_alpha_matte, unblend_ycbcr_roi
from remove_ai_watermarks.video_tools import require_video_tool

if TYPE_CHECKING:
    from pathlib import Path


def _run_cmd(cmd: list[str], *, text: bool = False) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(cmd, check=True, capture_output=True, text=text)  # noqa: S603


def _box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), box_type) + payload


def _full_box(box_type: bytes, version: int, flags: int, payload: bytes) -> bytes:
    header = struct.pack(">I", ((version & 0xFF) << 24) | (flags & 0xFFFFFF))
    return _box(box_type, header + payload)


class TestPhase1StudioAudio:
    def test_vendor_flag_routes_stripped_metadata_track_to_dac_without_allow_unverified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from remove_ai_watermarks import audio
        from remove_ai_watermarks._internal import audio_dac

        ffmpeg = require_video_tool("ffmpeg")
        src = tmp_path / "stripped_suno.wav"
        out = tmp_path / "clean.wav"
        _run_cmd(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=44100:duration=0.5",
                "-c:a",
                "pcm_f32le",
                str(src),
            ]
        )

        monkeypatch.setattr(audio_dac, "is_available", lambda: True)
        called: list[tuple[int, str]] = []

        def fake_dac(
            waveform: np.ndarray[Any, Any],
            sample_rate: int,
            *,
            device: str = "cpu",
            **_kwargs: Any,
        ) -> np.ndarray[Any, Any]:
            called.append((sample_rate, device))
            return waveform.copy()

        orig_identify = audio.identify_audio
        monkeypatch.setattr(audio_dac, "resynthesize", fake_dac)
        monkeypatch.setattr(
            audio,
            "identify_audio",
            lambda p, **_kw: orig_identify(p, check_invisible=False),
        )

        res = remove_audio_invisible(src, out, scheme="auto", vendor="suno")
        assert res.method == "dac"
        assert res.quality_passed is True
        assert out.is_file()
        assert called

    @pytest.mark.parametrize(("codec", "expected_fmt"), [("pcm_s16le", "s16"), ("pcm_s24le", "s32")])
    def test_preserves_16bit_and_24bit_pcm_sample_format_when_not_bitexact(
        self, tmp_path: Path, codec: str, expected_fmt: str
    ) -> None:
        ffmpeg = require_video_tool("ffmpeg")
        src = tmp_path / f"in_{codec}.wav"
        out = tmp_path / f"out_{codec}.wav"
        _run_cmd(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration=0.25",
                "-c:a",
                codec,
                str(src),
            ]
        )
        stream = probe_audio(src)
        t = np.linspace(0.0, 0.25, 12000, endpoint=False, dtype=np.float32)
        samples = (0.25 * np.sin(2.0 * np.pi * 440.0 * t))[:, None]
        write_audio(stream, out, [samples], bitexact=False)
        out_stream = probe_audio(out)
        assert out_stream.codec == codec
        assert out_stream.sample_fmt == expected_fmt

    def test_soft_overshoot_guard_prevents_flac_clipping_while_rejecting_runaway_gain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from remove_ai_watermarks import audio
        from remove_ai_watermarks._internal import audio_dac

        ffmpeg = require_video_tool("ffmpeg")
        src = tmp_path / "hot.wav"
        out_mild = tmp_path / "mild.flac"
        out_runaway = tmp_path / "runaway.wav"
        _run_cmd(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=44100:duration=0.5:beep_factor=1",
                "-af",
                "volume=0.95",
                "-c:a",
                "pcm_f32le",
                str(src),
            ]
        )
        monkeypatch.setattr(audio_dac, "is_available", lambda: True)

        # 1.12x brings peak to ~1.064 (inside 0.9999 < peak <= 1.35): trimmed smoothly without FLAC clipping!
        monkeypatch.setattr(audio_dac, "resynthesize", lambda wav, _sr, **_kw: wav * 1.12)
        res = remove_audio_invisible(src, out_mild, method="dac", vendor="suno", verify=False, allow_unverified=True)
        assert res.quality_passed is True
        decoded_mild = np.concatenate(list(read_audio(out_mild, probe_audio(out_mild))), axis=0)
        assert float(np.max(np.abs(decoded_mild))) <= 1.0

        # 8.0x runaway gain (> 1.35) must NOT be hidden by the overshoot guard and must fail quality gate!
        monkeypatch.setattr(audio_dac, "resynthesize", lambda wav, _sr, **_kw: wav * 8.0)
        with pytest.raises(audio.AudioVerificationIncomplete):
            remove_audio_invisible(src, out_runaway, method="dac", vendor="suno", verify=False, allow_unverified=True)

    def test_surround_5_1_stem_processing_preserves_channels_and_lowpasses_lfe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from remove_ai_watermarks._internal import audio_dac

        ffmpeg = require_video_tool("ffmpeg")
        src = tmp_path / "surround51.wav"
        out = tmp_path / "surround51_clean.wav"
        _run_cmd(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                (
                    "aevalsrc="
                    "'0.2*sin(2*PI*440*t)|0.2*sin(2*PI*550*t)|0.2*sin(2*PI*330*t)|"
                    "0.25*sin(2*PI*40*t)+0.25*sin(2*PI*4000*t)|"
                    "0.15*sin(2*PI*660*t)|0.15*sin(2*PI*770*t)'"
                    ":s=48000:d=0.5:c=5.1"
                ),
                "-c:a",
                "pcm_f32le",
                str(src),
            ]
        )
        monkeypatch.setattr(audio_dac, "is_available", lambda: True)
        monkeypatch.setattr(audio_dac, "resynthesize", lambda wav, _sr, **_kw: wav.copy())

        res = remove_audio_invisible(
            src,
            out,
            method="dac",
            vendor="google",
            verify=False,
            allow_unverified=True,
            allow_quality_override=True,
        )
        assert res.method == "dac"
        out_probe = probe_audio(out)
        assert out_probe.channels == 6

        decoded = np.concatenate(list(read_audio(out, out_probe)), axis=0)
        lfe = decoded[:, 3]
        spec = np.abs(np.fft.rfft(lfe))
        freqs = np.fft.rfftfreq(lfe.shape[0], d=1.0 / 48000.0)
        sub_amp = float(spec[np.argmin(np.abs(freqs - 40.0))])
        high_amp = float(spec[np.argmin(np.abs(freqs - 4000.0))])
        assert sub_amp > 50.0 * high_amp


class TestPhase2VisibleAndErase:
    def test_ycbcr_reverse_alpha_unblending_recovers_moving_textured_background(self) -> None:
        rng = np.random.default_rng(7)
        canvas_h, canvas_w = 180, 260
        yy, xx = np.mgrid[0:canvas_h, 0:canvas_w].astype(np.float32)
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        canvas[..., 0] = 120.0 + 55.0 * np.sin(xx / 6.0) + 35.0 * np.cos(yy / 8.0)
        canvas[..., 1] = 130.0 + 50.0 * np.cos((xx + yy) / 7.0)
        canvas[..., 2] = 115.0 + 60.0 * np.sin((xx - yy) / 5.5)
        canvas = np.clip(canvas + rng.normal(0.0, 3.0, canvas.shape), 15.0, 235.0).astype(np.uint8)

        w, h = 160, 100
        rx, ry, rw, rh = 96, 62, 52, 26
        bounds = (rx, ry, rx + rw, ry + rh)
        alpha_gt = np.zeros((rh, rw), dtype=np.float32)
        cv2.putText(alpha_gt, "Veo", (4, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.72, 0.48, 2, cv2.LINE_AA)
        color_bgr = np.array([242.0, 244.0, 246.0], dtype=np.float32)

        gt_frames: list[np.ndarray[Any, Any]] = []
        comp_frames: list[np.ndarray[Any, Any]] = []
        for idx in range(24):
            ox = 10 + idx * 3
            oy = 12 + idx * 2
            bg = canvas[oy : oy + h, ox : ox + w].copy()
            gt_frames.append(bg)
            comp = bg.astype(np.float32)
            roi = comp[ry : ry + rh, rx : rx + rw]
            comp[ry : ry + rh, rx : rx + rw] = (1.0 - alpha_gt[..., None]) * roi + alpha_gt[..., None] * color_bgr
            comp_frames.append(np.clip(np.round(comp), 0, 255).astype(np.uint8))

        matte = estimate_clip_alpha_matte(comp_frames, bounds)
        assert matte is not None
        assert float(np.mean(np.abs(matte.alpha_y - alpha_gt))) < 0.01

        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[ry : ry + rh, rx : rx + rw] = 255

        psnrs: list[float] = []
        for gt, comp in zip(gt_frames, comp_frames, strict=True):
            restored, residual_mask = unblend_ycbcr_roi(comp, full_mask, matte)
            assert int(np.count_nonzero(residual_mask)) == 0
            mse = float(
                np.mean(
                    (
                        gt[ry : ry + rh, rx : rx + rw].astype(np.float32)
                        - restored[ry : ry + rh, rx : rx + rw].astype(np.float32)
                    )
                    ** 2
                )
            )
            psnrs.append(10.0 * np.log10((255.0**2) / max(mse, 1e-6)))
        assert float(np.mean(psnrs)) > 34.0

    def test_video_erase_partial_frame_range_preserves_outside_frames(self, tmp_path: Path) -> None:
        ffmpeg = require_video_tool("ffmpeg")
        src = tmp_path / "clip.mp4"
        out = tmp_path / "erased.mp4"
        _run_cmd(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=navy:s=128x96:r=12:d=1",
                "-vf",
                "drawbox=x=20:y=20:w=24:h=20:color=white:t=fill",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(src),
            ]
        )
        res = remove_video_region(
            src,
            out,
            regions=[(18, 18, 28, 24)],
            frame_range=(0, 5),
            backend="cv2",
        )
        assert res.total_frames == 12
        assert res.erased_frames == 6

        cap = cv2.VideoCapture(str(out))
        frames: list[np.ndarray[Any, Any]] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frames.append(frame)
        finally:
            cap.release()
        assert len(frames) == 12
        assert float(frames[2][30, 30, 2]) < 60.0
        assert float(frames[9][30, 30, 2]) > 200.0


class TestPhase3WanAndBlindDetect:
    def test_wan21_cpu_prompt_offload_releases_text_encoder_before_accelerator_move(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        import types

        torch = pytest.importorskip("torch")
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-cache"))

        from remove_ai_watermarks import video_wan

        events: list[str] = []

        class FakeVAE:
            @classmethod
            def from_pretrained(cls, *_a: Any, **_kw: Any) -> FakeVAE:
                return cls()

            def enable_tiling(self) -> None:
                events.append("vae_tiling")

            def enable_slicing(self) -> None:
                events.append("vae_slicing")

        class FakeScheduler:
            @classmethod
            def from_config(cls, _cfg: Any, **_kw: Any) -> FakeScheduler:
                return cls()

        class FakePipe:
            def __init__(self, *, text_encoder: Any = "default", tokenizer: Any = "default") -> None:
                self.text_encoder: Any = object() if text_encoder == "default" else text_encoder
                self.tokenizer: Any = object() if tokenizer == "default" else tokenizer
                self.scheduler: Any = types.SimpleNamespace(config={})

            @classmethod
            def from_pretrained(cls, *_a: Any, **kw: Any) -> FakePipe:
                if "text_encoder" in kw and kw["text_encoder"] is None:
                    events.append("from_pretrained_no_text_encoder")
                    return cls(text_encoder=None, tokenizer=kw.get("tokenizer"))
                return cls()

            def encode_prompt(self, **_kwargs: Any) -> tuple[Any, Any]:
                assert self.text_encoder is not None
                events.append("encode_prompt_on_cpu")
                return torch.ones((1, 4, 8), dtype=torch.float32) * 0.25, torch.zeros((1, 4, 8))

            def to(self, device: str) -> FakePipe:
                assert self.text_encoder is None, "text_encoder must be released before moving DiT+VAE to accelerator"
                events.append(f"to:{device}")
                return self

            def set_progress_bar_config(self, **_kw: Any) -> None:
                pass

        for pkg_name in (
            "diffusers",
            "diffusers.models",
            "diffusers.models.autoencoders",
            "diffusers.pipelines",
            "diffusers.pipelines.wan",
            "diffusers.schedulers",
        ):
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = []  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, pkg_name, pkg)

        wan_vae_mod = types.ModuleType("diffusers.models.autoencoders.autoencoder_kl_wan")
        wan_vae_mod.AutoencoderKLWan = FakeVAE  # type: ignore[attr-defined]
        wan_pipe_mod = types.ModuleType("diffusers.pipelines.wan.pipeline_wan_video2video")
        wan_pipe_mod.WanVideoToVideoPipeline = FakePipe  # type: ignore[attr-defined]
        wan_sched_mod = types.ModuleType("diffusers.schedulers.scheduling_unipc_multistep")
        wan_sched_mod.UniPCMultistepScheduler = FakeScheduler  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "diffusers.models.autoencoders.autoencoder_kl_wan", wan_vae_mod)
        monkeypatch.setitem(sys.modules, "diffusers.pipelines.wan.pipeline_wan_video2video", wan_pipe_mod)
        monkeypatch.setitem(sys.modules, "diffusers.schedulers.scheduling_unipc_multistep", wan_sched_mod)
        monkeypatch.setattr(video_wan, "is_available", lambda: True)
        monkeypatch.setattr(video_wan, "resolve_wan_device", lambda _d: "cpu")

        rt = video_wan.load_video_wan_runtime(device="cpu", local_files_only=True)
        assert rt.prompt_embeds is not None
        assert events == ["vae_tiling", "vae_slicing", "encode_prompt_on_cpu", "to:cpu"]
        assert video_wan._wan_prompt_cache_path().is_file()

        # Warm hit: skips UMT5-XXL altogether (text_encoder=None passed to from_pretrained, no encode_prompt call)
        events.clear()
        rt_warm = video_wan.load_video_wan_runtime(device="cpu", local_files_only=True)
        assert rt_warm.prompt_embeds is not None
        assert torch.allclose(rt_warm.prompt_embeds, rt.prompt_embeds)
        assert events == ["vae_tiling", "vae_slicing", "from_pretrained_no_text_encoder", "to:cpu"]

    def test_blind_video_watermark_presence_cli_and_api(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from remove_ai_watermarks import video_messages

        ffmpeg = require_video_tool("ffmpeg")
        clip = tmp_path / "marked.mp4"
        _run_cmd(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=darkslateblue:s=64x64:r=8:d=1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(clip),
            ]
        )

        fixed_bits = np.where(np.arange(1024) % 2 == 0, 3.0, -3.0).astype(np.float32)
        monkeypatch.setattr(video_messages, "is_available", lambda: True)
        monkeypatch.setattr(video_messages, "load_detector", lambda _scheme: object())
        monkeypatch.setattr(
            video_messages,
            "decode_presence_and_logits",
            lambda _model, frames: (
                np.full((len(frames),), 4.0, dtype=np.float32),
                np.tile(fixed_bits, (len(frames), 1)),
            ),
        )
        report = detect_video_watermark_presence(clip, scheme="chunkyseal")
        assert report.status == "detected"
        assert report.frames == 8
        assert report.bit_sign_consistency == pytest.approx(1.0)
        assert report.decoded_hex is not None
        assert len(report.decoded_hex) == 256

        cli_res = CliRunner().invoke(
            main, ["video", "message", "detect", str(clip), "--scheme", "chunkyseal", "--json"]
        )
        assert cli_res.exit_code == 0, cli_res.output
        payload = json.loads(cli_res.output)
        assert payload["status"] == "detected"
        assert payload["scheme"] == "chunkyseal"


class TestPhase4HdrAndContainerScrub:
    def test_10bit_hdr_bt2020_pq_erase_preserves_depth_and_hdr_color_tags(self, tmp_path: Path) -> None:
        ffmpeg = require_video_tool("ffmpeg")
        ffprobe = require_video_tool("ffprobe")
        if "libx265" not in _run_cmd([ffmpeg, "-hide_banner", "-encoders"], text=True).stdout:
            pytest.skip("libx265 encoder unavailable")
        src = tmp_path / "hdr10.mp4"
        out = tmp_path / "hdr10_clean.mp4"
        _run_cmd(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x120:rate=12:duration=0.5",
                "-vf",
                "format=yuv420p10le,drawbox=x=16:y=16:w=24:h=20:color=white:t=fill",
                "-c:v",
                "libx265",
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
                "-tag:v",
                "hvc1",
                str(src),
            ]
        )
        res = remove_video_region(src, out, regions=[(14, 14, 28, 24)], backend="cv2")
        assert res.erased_frames == 6

        probe = json.loads(
            _run_cmd(
                [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(out)],
                text=True,
            ).stdout
        )
        stream = probe["streams"][0]
        assert stream["pix_fmt"] == "yuv420p10le"
        assert stream["color_primaries"] == "bt2020"
        assert stream["color_transfer"] == "smpte2084"
        assert stream["color_space"] == "bt2020nc"

    def test_fmp4_c2pa_emsg_detected_and_scrubbed_in_place(self, tmp_path: Path) -> None:
        ftyp = _box(b"ftyp", b"isom\x00\x00\x02\x00isomiso6")
        moov = _box(b"moov", _box(b"mvhd", b"\x00" * 100))
        emsg_payload = b"urn:c2pa:verifiable-segment-info\x00fmp4\x00" + b"\x00" * 16 + b"c2pa-segment-sig"
        emsg = _full_box(b"emsg", 0, 0, emsg_payload)
        mdat = _box(b"mdat", b"\x01\x02\x03\x04" * 16)

        fmp4 = tmp_path / "segment.mp4"
        out = tmp_path / "segment_clean.mp4"
        fmp4.write_bytes(ftyp + moov + emsg + mdat)

        detected = get_ai_metadata(fmp4)
        assert "c2pa_manifest" in detected
        remove_ai_metadata(fmp4, out)
        assert get_ai_metadata(out) == {}
        assert b"urn:c2pa:verifiable-segment-info" not in out.read_bytes()

    def test_h264_ai_sei_payload_scrubbed_in_place(self, tmp_path: Path) -> None:
        sei_body = C2PA_UUID + b"urn:c2pa:sei-manifest-payload"
        rbsp = bytearray([5, len(sei_body)]) + sei_body + b"\x80"
        nal = bytes([0x06]) + bytes(rbsp)
        sample = struct.pack(">I", len(nal)) + nal

        avcc = _box(b"avcC", bytes([1, 0x64, 0x00, 0x1F, 0xFF, 0xE1, 0x00, 0x00, 0x00]))
        avc1 = _box(b"avc1", b"\x00" * 78 + avcc)
        stsd = _full_box(b"stsd", 0, 0, struct.pack(">I", 1) + avc1)
        stsc = _full_box(b"stsc", 0, 0, struct.pack(">IIII", 1, 1, 1, 1))
        stsz = _full_box(b"stsz", 0, 0, struct.pack(">III", 0, 1, len(sample)))
        stco_placeholder = _full_box(b"stco", 0, 0, struct.pack(">II", 1, 0))
        stbl = _box(b"stbl", stsd + stsc + stsz + stco_placeholder)
        hdlr = _full_box(b"hdlr", 0, 0, b"\x00\x00\x00\x00vide" + b"\x00" * 12 + b"VideoHandler\x00")
        mdia = _box(b"mdia", hdlr + _box(b"minf", stbl))
        trak = _box(b"trak", mdia)
        moov_tmp = _box(b"moov", trak)
        ftyp = _box(b"ftyp", b"isom\x00\x00\x02\x00isom")
        mdat_payload_offset = len(ftyp) + len(moov_tmp) + 8
        stco = _full_box(b"stco", 0, 0, struct.pack(">II", 1, mdat_payload_offset))
        moov = _box(
            b"moov",
            _box(b"trak", _box(b"mdia", hdlr + _box(b"minf", _box(b"stbl", stsd + stsc + stsz + stco)))),
        )
        mp4 = tmp_path / "sei_c2pa.mp4"
        out = tmp_path / "sei_clean.mp4"
        mp4.write_bytes(ftyp + moov + _box(b"mdat", sample))

        scan_before = isobmff.scan_isobmff_video_sei(mp4)
        assert len(scan_before.ai_sei_spans) == 1
        assert "ai_sei" in get_ai_metadata(mp4)

        remove_ai_metadata(mp4, out)
        scan_after = isobmff.scan_isobmff_video_sei(out)
        assert scan_after.ai_sei_spans == ()
        assert get_ai_metadata(out) == {}

    def test_mkv_c2pa_attachment_detected_and_stripped(self, tmp_path: Path) -> None:
        ffmpeg = require_video_tool("ffmpeg")
        ffprobe = require_video_tool("ffprobe")
        font_blob = tmp_path / "subtitle.ttf"
        font_blob.write_bytes(b"\x00\x01\x00\x00fontdata")
        attach_blob = tmp_path / "manifest.c2pa"
        attach_blob.write_bytes(b"jumbc2pa.manifest\x00trainedAlgorithmicMedia")
        mkv = tmp_path / "clip_c2pa.mkv"
        out = tmp_path / "clip_clean.mkv"
        _run_cmd(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:r=12:d=0.5",
                "-attach",
                str(font_blob),
                "-metadata:s:t:0",
                "mimetype=application/x-truetype-font",
                "-attach",
                str(attach_blob),
                "-metadata:s:t:1",
                "mimetype=application/c2pa",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(mkv),
            ]
        )
        detected = get_ai_metadata(mkv)
        assert "c2pa_manifest" in detected
        assert detected.get("source_type") == "trainedAlgorithmicMedia (AI-generated)"

        remove_ai_metadata(mkv, out)
        assert get_ai_metadata(out) == {}
        out_streams = json.loads(
            _run_cmd([ffprobe, "-v", "error", "-show_streams", "-of", "json", str(out)], text=True).stdout
        )["streams"]
        att_streams = [s for s in out_streams if s.get("codec_type") == "attachment"]
        assert len(att_streams) == 1

    def test_mkv_and_m4v_supported_across_remove_video_all_with_audio_and_invisible(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from shutil import copy2

        from remove_ai_watermarks import video_invisible
        from remove_ai_watermarks.video_invisible import RegenerationMetrics

        ffmpeg = require_video_tool("ffmpeg")

        def fake_regen(s: Path, o: Path, **_kw: Any) -> RegenerationMetrics:
            copy2(s, o)
            return RegenerationMetrics(
                frames=6,
                fps=12.0,
                width=96,
                height=64,
                psnr_db=34.0,
                temporal_residual_ratio=0.95,
            )

        monkeypatch.setattr(video_invisible, "regenerate_video_candidate", fake_regen)
        monkeypatch.setattr(video_invisible, "load_video_vae_runtime", lambda **_kw: object())
        monkeypatch.setattr(video_invisible, "is_available", lambda: True)
        monkeypatch.setattr(video_invisible, "pick_device", lambda _d: "cpu")

        for ext, vcodec, acodec in [(".mkv", "libx264", "pcm_s16le"), (".m4v", "libx264", "aac")]:
            src = tmp_path / f"in{ext}"
            out = tmp_path / f"out{ext}"
            _run_cmd(
                [
                    ffmpeg,
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=darkslategray:s=96x64:r=12:d=0.5",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=44100:duration=0.5",
                    "-c:v",
                    vcodec,
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    acodec,
                    str(src),
                ]
            )
            res = remove_video_all(
                src,
                out,
                include_invisible=True,
                include_audio_invisible=True,
            )
            assert res.invisible_removed is True
            assert res.audio_status is not None
            assert out.is_file()


class TestPhase5aMultimodalCli:
    def test_top_level_identify_and_erase_and_json_video_flags(self, tmp_path: Path) -> None:
        ffmpeg = require_video_tool("ffmpeg")
        vid = tmp_path / "sample.mp4"
        erased = tmp_path / "sample_erased.mp4"
        _run_cmd(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=96x64:r=12:d=0.5",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(vid),
            ]
        )
        runner = CliRunner()

        id_res = runner.invoke(main, ["identify", str(vid), "--json"])
        assert id_res.exit_code == 0, id_res.output
        assert "verdict" in json.loads(id_res.output) or "is_ai_generated" in json.loads(id_res.output)

        meta_res = runner.invoke(main, ["video", "metadata", str(vid), "--check", "--json"])
        assert meta_res.exit_code == 0, meta_res.output
        meta_payload = json.loads(meta_res.output)
        assert meta_payload["markers"] == {}
        assert meta_payload["has_ai_metadata"] is False

        erase_res = runner.invoke(
            main,
            ["erase", str(vid), "-o", str(erased), "--region", "8,8,20,16", "--frames", "0:2", "--json"],
        )
        assert erase_res.exit_code == 0, erase_res.output
        erase_payload = json.loads(erase_res.output)
        assert erase_payload["erased_frames"] == 3
        assert erased.is_file()
