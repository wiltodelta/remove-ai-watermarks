"""Tests for Wave 8 (v0.69.0): cohort-accurate video/audio auto-routing, audio auto-device, and gain staging."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from remove_ai_watermarks import audio, video
from remove_ai_watermarks._internal import audio_dac, audio_io

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_video_invisible_target_cohort_gates_pixel_regeneration_accurately() -> None:
    """Only verified pixel-watermark cohorts trigger automatic pixel regeneration."""
    # Confirmed pixel-watermark cohorts -> trigger
    assert video.video_invisible_target_cohort({"synthid_watermark": "Google SynthID"}, None) == "synthid"
    assert video.video_invisible_target_cohort({}, "veo") == "synthid"
    assert video.video_invisible_target_cohort({}, "seedance") == "seedance"
    assert video.video_invisible_target_cohort({"app_aigc": "CapCut AI"}, None) == "seedance"
    assert video.video_invisible_target_cohort({}, None, explicit_vendor="google") == "synthid"
    assert video.video_invisible_target_cohort({}, None, explicit_vendor="bytedance") == "seedance"
    assert video.video_invisible_target_cohort({}, None, explicit_vendor="meta") == "meta"

    # Neutral C2PA signers, ComfyUI, and proven metadata/SEI-only AI models -> None (lossless pixels!)
    assert (
        video.video_invisible_target_cohort(
            {"c2pa_manifest": "store", "issuer": "YouTube Video Processing Services"}, None
        )
        is None
    )
    assert video.video_invisible_target_cohort({"workflow": '{"nodes": []}'}, None) is None
    assert video.video_invisible_target_cohort({"kling_sei": "kling-ai", "aigc_producer": "kling"}, "kling") is None
    assert (
        video.video_invisible_target_cohort({"c2pa_manifest": "store", "claim_generator": "Dreamina"}, "dreamina")
        is None
    )
    assert video.video_invisible_target_cohort({"c2pa_manifest": "store", "issuer": "RUNWAY AI, INC."}, None) is None
    assert (
        video.video_invisible_target_cohort({"c2pa_manifest": "store", "claim_generator": "xAI Grok Imagine"}, None)
        is None
    )
    assert video.video_invisible_target_cohort({"aigc_producer": "hailuo"}, "hailuo") is None
    assert video.video_invisible_target_cohort({}, None) is None


def test_audio_auto_scrub_restricts_to_certified_vendors_and_resolves_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto-scrub triggers only for certified Google/ElevenLabs; auto device resolves cleanly."""
    stream = audio_io.AudioStreamInfo(
        index=0,
        sample_rate=48000,
        channels=2,
        codec="pcm_s16le",
        start_time=Fraction(0),
        duration=Fraction(2),
        channel_layout="stereo",
    )
    monkeypatch.setattr(audio_dac, "is_available", lambda: True)

    google_rep = audio.AudioProvenanceReport(
        tmp_path / "g.wav",
        "wav",
        stream,
        {"claim_generator": "Google Veo", "synthid_watermark": "Google SynthID"},
        "not_detected",
        (),
        (),
    )
    eleven_rep = audio.AudioProvenanceReport(
        tmp_path / "e.wav", "wav", stream, {"claim_generator": "ElevenLabs"}, "not_detected", (), ()
    )
    cartesia_rep = audio.AudioProvenanceReport(
        tmp_path / "c.wav", "wav", stream, {"ai_tool": "Cartesia Sonic"}, "not_detected", (), ()
    )
    openai_rep = audio.AudioProvenanceReport(
        tmp_path / "o.wav",
        "wav",
        stream,
        {"issuer": "OpenAI", "source_type": "trainedAlgorithmicMedia"},
        "not_detected",
        (),
        (),
    )

    assert audio.should_auto_scrub_audio(google_rep, method="auto", explicit_vendor=None) is True
    assert audio.should_auto_scrub_audio(eleven_rep, method="auto", explicit_vendor=None) is True
    assert audio.should_auto_scrub_audio(cartesia_rep, method="auto", explicit_vendor=None) is False
    assert audio.should_auto_scrub_audio(openai_rep, method="auto", explicit_vendor=None) is False

    # CPU-only profiles always resolve "auto" to "cpu"
    assert audio.resolve_audio_device("auto", "spectral") == "cpu"
    assert audio.resolve_audio_device("auto", "spectral") == "cpu"
    assert audio.resolve_audio_device("cpu", "dac") == "cpu"
    assert audio.resolve_audio_device("auto", "dac") in {"cpu", "mps", "cuda"}


def test_blend_hf_crossover_matches_low_band_rms() -> None:
    """blend_hf_crossover preserves low-band RMS even when processed low band was attenuated."""
    sr = 48000
    t = np.arange(sr, dtype=np.float32) / sr
    low = 0.25 * np.sin(2 * np.pi * 440 * t)
    hi = 0.15 * np.sin(2 * np.pi * 16500 * t)
    orig = np.column_stack([low + hi, low + hi]).astype(np.float32)
    proc = np.column_stack([0.7 * low, 0.7 * low]).astype(np.float32)
    blended = audio_dac.blend_hf_crossover(proc, orig, sr, 14000.0)
    orig_rms = float(np.sqrt(np.mean(orig**2)))
    blended_rms = float(np.sqrt(np.mean(blended**2)))
    assert abs(blended_rms - orig_rms) / orig_rms < 0.05
