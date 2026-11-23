"""Regression tests for v0.74.0 (Wave 12) audio, video, GOP stream-copy, and provenance features."""

from __future__ import annotations

import csv
import hashlib
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image, ImageDraw

from remove_ai_watermarks import metadata
from remove_ai_watermarks._internal import audio_snac
from remove_ai_watermarks._internal.c2pa import c2pa_info_from_manifest_store, soft_binding_vendors_in
from remove_ai_watermarks._internal.constants import C2PA_AI_VENDORS
from remove_ai_watermarks._internal.tc260_producers import producer_for_code
from remove_ai_watermarks.audio import _audio_metadata_vendor, remove_audio_invisible
from remove_ai_watermarks.cli import main
from remove_ai_watermarks.video import (
    VIDEO_VISIBLE_MARKS,
    _platform_from_video_metadata,
    identify_video,
    remove_video_invisible,
    remove_video_metadata,
    remove_video_region,
    remove_video_visible,
)
from remove_ai_watermarks.video_decode import open_video_capture
from remove_ai_watermarks.video_invisible import VideoVaeRuntime
from remove_ai_watermarks.video_temporal import _backward_map, _flow_field
from remove_ai_watermarks.video_tools import require_video_tool
from remove_ai_watermarks.video_visible import (
    _scalable_default_font,
    _template_sources,
    detect_heygen_frame,
    encode_clean_video,
    scan_video_marks,
    stabilize_localizations,
)


def _make_mp4_box(box_type: bytes, payload: bytes) -> bytes:
    return (len(payload) + 8).to_bytes(4, "big") + box_type + payload


def _write_minimal_mp4(path: Path, *, width: int = 320, height: int = 240, fps: int = 24, seconds: int = 1) -> None:
    ffmpeg = require_video_tool("ffmpeg")
    subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={width}x{height}:rate={fps}:duration={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def test_shipped_snac_operating_point_matches_certified_audio_synthid_manifest() -> None:
    repo = Path(__file__).resolve().parents[1]
    manifest = repo / "data" / "evaluations" / "audio-synthid-oracle.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    positive_controls = {
        row["source_sha256"]
        for row in rows
        if row["method"] == "source_wav_control" and row["output_verdict"] == "detected"
    } | {row["source_sha256"] for row in rows if row["source_verdict"] == "detected"}
    certified_snac: set[tuple[str, str, int]] = set()
    for row in rows:
        carrier = repo / row["source_path"]
        assert carrier.is_file()
        assert hashlib.sha256(carrier.read_bytes()).hexdigest() == row["source_sha256"]
        if (
            row["method"] == "snac"
            and row["output_verdict"] == "not_detected"
            and row["quantizers"]
            and row["source_sha256"] in positive_controls
        ):
            certified_snac.add((row["model"], row["revision"], int(row["quantizers"])))

    assert (audio_snac.MODEL_REPOSITORY, audio_snac.MODEL_REVISION, audio_snac.FIXED_QUANTIZERS) in certified_snac


def test_snac_defaults_to_preserve_hf_14khz_on_48khz_and_none_on_16khz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_hf: list[float | None] = []

    def _fake_snac_resynth(
        samples: np.ndarray,
        sample_rate: int,
        *,
        device: str = "cpu",
        preserve_hf_hz: float | None = None,
        **_kwargs: Any,
    ) -> np.ndarray:
        seen_hf.append(preserve_hf_hz)
        return samples.copy()

    monkeypatch.setattr(audio_snac, "is_available", lambda: True)
    monkeypatch.setattr(audio_snac, "resynthesize", _fake_snac_resynth)

    for sr in (48000, 16000):
        t = np.linspace(0.0, 1.0, sr, endpoint=False, dtype=np.float32)
        wav = np.stack([0.2 * np.sin(2.0 * np.pi * 440.0 * t), 0.2 * np.cos(2.0 * np.pi * 440.0 * t)], axis=1)
        src = tmp_path / f"in_{sr}.wav"
        out = tmp_path / f"out_{sr}.wav"
        pcm16 = np.clip(np.rint(wav * 32767.0), -32768, 32767).astype("<i2")
        with wave.open(str(src), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())
        res = remove_audio_invisible(src, out, method="snac", force=True, verify=False)
        assert res.output == out

    assert seen_hf == [14000.0, None]


def test_c2pa_pixverse_and_suno_vendors_and_soft_binding_parity() -> None:
    vendors = {v.issuer: v for v in C2PA_AI_VENDORS}
    assert vendors[b"PixVerse"].platform == "PixVerse"
    assert vendors[b"PixVerse"].asserts_ai is True
    assert vendors[b"Suno Inc"].platform == "Suno"
    assert vendors[b"Suno Inc"].asserts_ai is True

    store = {
        "active_manifest": "urn:c2pa:test",
        "manifests": {
            "urn:c2pa:test": {
                "claim_generator_info": [{"name": "Suno"}],
                "assertions": [
                    {
                        "label": "c2pa.actions.v2",
                        "data": {
                            "actions": [
                                {
                                    "action": "c2pa.created",
                                    "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
                                }
                            ]
                        },
                    },
                    {
                        "label": "c2pa.soft-binding",
                        "data": {"alg": "com.suno.fingerprint.v4", "blocks": [{"value": "abc-123"}]},
                    },
                ],
                "signature_info": {"issuer": "Suno Inc"},
            }
        },
    }
    info = c2pa_info_from_manifest_store(store)
    assert info["issuer"] == "Suno"
    assert info["soft_binding"] == "Suno Audio Fingerprint"
    assert "Suno Audio Fingerprint" in soft_binding_vendors_in(b"alg: com.suno.fingerprint.v4")
    assert _audio_metadata_vendor({"audio_generator": "Suno"}) == "suno"
    assert _platform_from_video_metadata({"audio_generator": "Suno"}) == "Suno"
    honor = producer_for_code("HZ_XYKJ")
    assert honor is not None
    assert honor.video_platform == "Honor"


def test_standalone_isobmff_xmp_iptc_detect_and_strip_parity(tmp_path: Path) -> None:
    base = tmp_path / "clean.mp4"
    _write_minimal_mp4(base)
    raw_mp4 = base.read_bytes()

    xmp_uuid = bytes.fromhex("be7acfcb97a942e89c71999491e3afac")
    meta_xmp = (
        b"<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>"
        b"<x:xmpmeta xmlns:x='adobe:ns:meta/'><rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>"
        b"<rdf:Description xmlns:Iptc4xmpExt='http://iptc.org/std/Iptc4xmpExt/2008-02-29/' "
        b"xmlns:dc='http://purl.org/dc/elements/1.1/'>"
        b"<Iptc4xmpExt:DigitalSourceType>http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia</Iptc4xmpExt:DigitalSourceType>"
        b"<dc:publisher><rdf:Bag><rdf:li>Meta</rdf:li></rdf:Bag></dc:publisher>"
        b"</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end='w'?>"
    )
    canva_xmp = (
        b'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        b'<rdf:Description ContainsAiGeneratedContent:ContainsAiGeneratedContent="Yes" '
        b'Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"/>'
        b'</rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
    )

    for name, payload, expected_plat in (
        ("meta.mp4", meta_xmp, "Meta AI"),
        ("canva.mp4", canva_xmp, "Canva (Magic Media)"),
    ):
        carrier = tmp_path / name
        carrier.write_bytes(raw_mp4 + _make_mp4_box(b"uuid", xmp_uuid + payload))
        assert metadata.has_ai_metadata(carrier) is True
        markers = metadata.get_ai_metadata(carrier)
        assert "source_type" in markers
        assert "app_aigc" in markers
        rep = identify_video(carrier, check_visible=False)
        assert rep.is_ai_generated is True
        assert rep.platform == expected_plat
        cleaned = tmp_path / f"cleaned_{name}"
        res = remove_video_metadata(carrier, cleaned)
        assert res.remaining == {}
        assert metadata.has_ai_metadata(cleaned) is False


def test_scaled_farneback_preserves_displacement_at_720p() -> None:
    rng = np.random.default_rng(42)
    tex = cv2.GaussianBlur(rng.integers(30, 220, size=(720, 1280), dtype=np.uint8), (9, 9), 2.0)
    shifted = np.roll(np.roll(tex, 6, axis=1), -4, axis=0)
    flow = _flow_field(shifted, tex)
    fx, fy = flow[..., 0], flow[..., 1]
    med_dx = float(np.median(fx[120:600, 160:1120]))
    med_dy = float(np.median(fy[120:600, 160:1120]))
    assert -7.2 <= med_dx <= -4.8
    assert 3.0 <= med_dy <= 5.0
    map_x, map_y = _backward_map(shifted, tex)
    assert map_x.shape == (720, 1280)
    assert map_y.shape == (720, 1280)


def test_native_guide_upscales_512p_vae_to_native_resolution_and_toggles_tiling(tmp_path: Path) -> None:
    import torch

    src = tmp_path / "src_960x540.mp4"
    _write_minimal_mp4(src, width=960, height=540, fps=24, seconds=1)

    tiling_calls: list[str] = []
    encoded_shapes: list[tuple[int, ...]] = []

    class _IdentityVae:
        dtype = torch.float32
        config = SimpleNamespace(scaling_factor=0.18215)

        def enable_tiling(self) -> None:
            tiling_calls.append("enable")

        def disable_tiling(self) -> None:
            tiling_calls.append("disable")

        def encode(self, tensor: Any) -> Any:
            encoded_shapes.append(tuple(int(d) for d in tensor.shape))
            lat = tensor[:, :1, ::8, ::8].repeat(1, 4, 1, 1)
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: lat / 0.18215))

        def decode(self, latents: Any) -> Any:
            up = torch.nn.functional.interpolate(latents[:, :3] * 0.18215, scale_factor=8, mode="bilinear")
            return SimpleNamespace(sample=up)

    rt = VideoVaeRuntime(
        model="stabilityai/sd-vae-ft-mse",
        requested_device="cpu",
        resolved_device="cpu",
        vae=_IdentityVae(),
        scaling_factor=0.18215,
    )
    out_guided = tmp_path / "guided.mp4"
    res = remove_video_invisible(
        src,
        out_guided,
        profile="synthid",
        fps=None,
        native_guide=True,
        device="cpu",
        _runtime=rt,
    )
    assert res.width == 960
    assert res.height == 540
    assert res.fps == pytest.approx(24.0, abs=0.1)
    assert encoded_shapes[0][2:] == (288, 512)
    assert "disable" in tiling_calls

    tiling_calls.clear()
    out_hi = tmp_path / "hi_1024.mp4"
    remove_video_invisible(src, out_hi, profile="synthid", long_side=800, device="cpu", _runtime=rt)
    assert "enable" in tiling_calls


def test_cli_wan_steps_and_native_guide_validation_and_forwarding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "clip.mp4"
    _write_minimal_mp4(src, width=320, height=240, fps=24, seconds=1)
    runner = CliRunner()

    bad = runner.invoke(main, ["video", "invisible", str(src), "--profile", "synthid", "--wan-steps", "6"])
    assert bad.exit_code != 0
    assert "wan_steps" in bad.output

    captured: dict[str, Any] = {}

    def _stub_inv(source: Any, output: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        out_p = Path(output) if output else Path(source).with_stem(Path(source).stem + "_clean")
        out_p.write_bytes(Path(source).read_bytes())
        return SimpleNamespace(
            source=Path(source),
            output=out_p,
            profile="synthid",
            width=320,
            height=240,
            total_frames=24,
            fps=24.0,
            remaining_metadata={},
        )

    monkeypatch.setattr("remove_ai_watermarks.video.remove_video_invisible", _stub_inv)
    ok = runner.invoke(main, ["video", "invisible", str(src), "-o", str(tmp_path / "o.mp4"), "--native-guide"])
    assert ok.exit_code == 0, ok.output
    assert captured["native_guide"] is True
    assert captured["fps"] is None


def test_selective_gop_stream_copy_preserves_untouched_gops_and_bumper_trim(tmp_path: Path) -> None:
    ffmpeg = require_video_tool("ffmpeg")
    src = tmp_path / "multi_gop.mp4"
    subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=30:duration=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=4",
            "-c:v",
            "libx264",
            "-g",
            "30",
            "-bf",
            "0",
            "-sc_threshold",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(src),
        ],
        check=True,
    )
    cap_src = open_video_capture(src)
    orig_frames: list[np.ndarray] = []
    while True:
        ok, frm = cap_src.read()
        if not ok:
            break
        orig_frames.append(frm)
    cap_src.release()
    assert len(orig_frames) == 120

    erased_out = tmp_path / "erased_head.mp4"
    res = remove_video_region(src, erased_out, regions=[(8, 8, 32, 24)], frame_range=(0, 10))
    assert res.erased_frames == 11
    cap_out = open_video_capture(erased_out)
    out_frames: list[np.ndarray] = []
    while True:
        ok, frm = cap_out.read()
        if not ok:
            break
        out_frames.append(frm)
    cap_out.release()
    assert len(out_frames) == 120
    # Untouched GOPs 1..3 (frames 30..119) must be bit-for-bit identical!
    for idx in range(30, 120):
        assert np.array_equal(orig_frames[idx], out_frames[idx]), f"Frame {idx} was unnecessarily re-encoded"

    # Pure keyframe-aligned bumper trim (0 dirty regions, trim at frame 60)
    scans = scan_video_marks(src, ("veo",))
    scan = scans["veo"]
    trimmed_out = tmp_path / "trimmed_60.mp4"
    rem = encode_clean_video(
        src,
        trimmed_out,
        scan,
        [None] * len(scan.detections),
        backend="cv2",
        strip_metadata=True,
        trim_end_frame=60,
    )
    assert rem == 0
    cap_trim = open_video_capture(trimmed_out)
    trim_frames: list[np.ndarray] = []
    while True:
        ok, frm = cap_trim.read()
        if not ok:
            break
        trim_frames.append(frm)
    cap_trim.release()
    assert len(trim_frames) == 60
    assert np.array_equal(orig_frames[0], trim_frames[0])
    assert np.array_equal(orig_frames[59], trim_frames[59])


def test_new_visible_marks_qwen_latin_and_heygen_tiled_preview_rejection(tmp_path: Path) -> None:
    for expected in ("pixverse", "luma", "heygen", "sync_labs", "tiktok_ai"):
        assert expected in VIDEO_VISIBLE_MARKS

    # International Latin Qwen ([3-lobe pinwheel] Qwen) detection and removal
    w, h = 640, 360
    tmpl = _template_sources()["qwen-en"]
    th = round(h * 0.056)
    tw = round(tmpl.shape[1] * th / tmpl.shape[0])
    alpha = (cv2.resize(tmpl, (tw, th)).astype(np.float32) / 255.0)[:, :, None]
    base = np.full((h, w, 3), 64, dtype=np.uint8)
    x, y = w - tw - 16, h - th - 10
    base[y : y + th, x : x + tw] = np.clip(base[y : y + th, x : x + tw] * (1.0 - alpha) + 245.0 * alpha, 0, 255).astype(
        np.uint8
    )
    qwen_mp4 = tmp_path / "qwen_latin.mp4"
    writer = cv2.VideoWriter(str(qwen_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
    for _ in range(24):
        writer.write(base)
    writer.release()

    rep = identify_video(qwen_mp4, check_visible=True)
    assert rep.visible_mark == "qwen"
    cleaned = tmp_path / "qwen_latin_clean.mp4"
    vis_res = remove_video_visible(qwen_mp4, cleaned, mark="auto")
    assert vis_res.output == cleaned
    assert vis_res.removed_frames == 24

    # HeyGen diagonal tiled preview MUST NOT be accepted as the corner AI badge
    tiled_img = Image.new("RGB", (720, 1280), (48, 52, 58))
    tile_layer = Image.new("L", (1600, 1600), 0)
    tdraw = ImageDraw.Draw(tile_layer)
    font = _scalable_default_font(44)
    for ty in range(0, 1600, 180):
        for tx in range(0, 1600, 320):
            tdraw.text((tx, ty), "HeyGen", font=font, fill=90)
    rotated = tile_layer.rotate(30, expand=False).crop((440, 160, 1160, 1440))
    tiled_bgr = np.asarray(tiled_img, dtype=np.uint8).copy()
    rot_a = (np.asarray(rotated, dtype=np.float32) / 255.0)[:, :, None]
    tiled_bgr = np.clip(tiled_bgr * (1.0 - rot_a) + 230.0 * rot_a, 0, 255).astype(np.uint8)
    dets = [detect_heygen_frame(tiled_bgr, frame_index=i) for i in range(24)]
    accepted = stabilize_localizations("heygen", dets, provenance=False)
    assert all(reg is None for reg in accepted)
