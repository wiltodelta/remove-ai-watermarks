"""Regression tests for Wave 14 (v0.76.0): CLI surface cleanup and native-resolution 2D SynthID default."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
from click.testing import CliRunner

from remove_ai_watermarks import api, cli, video, video_invisible, video_synthid, video_visible
from remove_ai_watermarks.invisible_engine import InvisibleEngine
from tests.test_agent_skill import _cli_commands, _long_options

if TYPE_CHECKING:
    import click
    import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _param_by_flag(command: click.Command, flag: str) -> click.Parameter:
    for param in command.params:
        if flag in param.opts or flag in param.secondary_opts:
            return param
    raise AssertionError(f"{flag} missing from {command.name}")


def test_video_synthid_default_preserves_native_fps_and_allows_wan_auto_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert video_synthid.DEFAULT_VIDEO_SYNTHID_FPS is None
    assert video_synthid.DEFAULT_VIDEO_SYNTHID_LONG_SIDE == 512
    assert video_synthid.DEFAULT_VIDEO_SYNTHID_NOISE_STD == 0.15

    # Default parameters must not count as 2D VAE overrides so --profile auto can select Wan 2.1 on CUDA.
    assert (
        video._has_2d_vae_overrides(
            noise_std=video_synthid.DEFAULT_VIDEO_SYNTHID_NOISE_STD,
            long_side=video_synthid.DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
            fps=video_synthid.DEFAULT_VIDEO_SYNTHID_FPS,
            batch_size=4,
            model=video_synthid.DEFAULT_VIDEO_SYNTHID_VAE,
            temporal_stabilize=True,
        )
        is False
    )
    # Passing an explicit fps cap counts as a 2D VAE override.
    assert (
        video._has_2d_vae_overrides(
            noise_std=video_synthid.DEFAULT_VIDEO_SYNTHID_NOISE_STD,
            long_side=video_synthid.DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
            fps=12.0,
            batch_size=4,
            model=video_synthid.DEFAULT_VIDEO_SYNTHID_VAE,
            temporal_stabilize=True,
        )
        is True
    )

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x10ftypmp42\x00\x00\x00\x00")
    monkeypatch.setattr("remove_ai_watermarks.video_wan.can_run_wan_on_source", lambda *_a, **_kw: True)
    monkeypatch.setattr("remove_ai_watermarks.metadata.get_ai_metadata", lambda _p: {})
    resolved = video._resolve_video_invisible_profile(
        "auto",
        clip,
        device="auto",
        quality="delivery",
        noise_std=video_synthid.DEFAULT_VIDEO_SYNTHID_NOISE_STD,
        long_side=video_synthid.DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
        fps=video_synthid.DEFAULT_VIDEO_SYNTHID_FPS,
        batch_size=4,
        seed=0,
        model=video_synthid.DEFAULT_VIDEO_SYNTHID_VAE,
        temporal_stabilize=True,
    )
    assert resolved == "wan21-synthid-experimental"


def test_removed_video_flags_are_absent_and_long_side_exists_only_on_video_message_remove() -> None:
    commands = _cli_commands()
    for path in (("video", "invisible"), ("video", "all"), ("video", "batch")):
        opts = _long_options(commands[path])
        for retired in ("--native-guide", "--keep-fps", "--wan-steps", "--long-side", "--no-temporal-stabilize"):
            assert retired not in opts, f"{path} still exposes {retired}"
        assert _param_by_flag(commands[path], "--fps").default is None

    with_long_side = {path for path, cmd in commands.items() if "--long-side" in _long_options(cmd)}
    assert with_long_side == {("video", "message", "remove")}

    for fn in (
        video_invisible.regenerate_video_candidate,
        video.remove_video_invisible,
        video.remove_video_all,
        video.remove_video_batch,
    ):
        sig = inspect.signature(fn)
        assert "native_guide" not in sig.parameters
        assert "wan_steps" not in sig.parameters


def test_fidelity_anchor_is_removed_across_options_and_engine() -> None:
    field_names = {f.name for f in dataclasses.fields(api.InvisibleOptions)}
    assert "fidelity_anchor" not in field_names
    assert "text_manifest" in field_names
    assert "fidelity_anchor" not in inspect.signature(InvisibleEngine.remove_watermark).parameters


def test_unified_cli_defaults_for_backend_scheme_and_batch_mode() -> None:
    commands = _cli_commands()

    backend_commands = (
        ("visible",),
        ("erase",),
        ("all",),
        ("batch",),
        ("video", "visible"),
        ("video", "erase"),
        ("video", "all"),
        ("video", "batch"),
    )
    for path in backend_commands:
        assert _param_by_flag(commands[path], "--backend").default == "auto", f"{path} backend default is not auto"

    scheme_commands = (
        ("audio", "identify"),
        ("audio", "invisible"),
        ("audio", "all"),
        ("audio", "batch"),
        ("video", "audio"),
    )
    for path in scheme_commands:
        assert _param_by_flag(commands[path], "--scheme").default == "auto", f"{path} scheme default is not auto"

    for path in (("video", "identify"), ("video", "all"), ("video", "batch")):
        assert _param_by_flag(commands[path], "--audio-scheme").default == "auto", (
            f"{path} audio-scheme default is not auto"
        )

    for path in (("batch",), ("video", "batch"), ("audio", "batch")):
        assert _param_by_flag(commands[path], "--mode").default == "all", f"{path} mode default is not all"


def test_video_audio_repeatable_stream_routes_single_and_multi_track_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00\x00\x00\x10ftypmp42\x00\x00\x00\x00")

    calls: list[tuple[str, Any]] = []

    def fake_single(source: Path, target: Path, **kwargs: Any) -> Any:
        calls.append(("single", kwargs["stream"]))
        target.write_bytes(b"ok")
        return SimpleNamespace(
            skipped=False,
            output=target,
            verified_removed=True,
            method="spectral",
            caveats=(),
            before=SimpleNamespace(caveats=()),
            after=None,
        )

    def fake_multi(source: Path, target: Path, **kwargs: Any) -> Any:
        calls.append(("multi", kwargs["streams"]))
        target.write_bytes(b"ok")
        return SimpleNamespace(
            skipped=False,
            output=target,
            processed_ordinals=(0, 2),
            verified_removed=True,
            caveats=(),
        )

    monkeypatch.setattr("remove_ai_watermarks.video_audio.remove_video_audio", fake_single)
    monkeypatch.setattr("remove_ai_watermarks.video_audio.remove_video_audio_all_streams", fake_multi)

    runner = CliRunner()
    r0 = runner.invoke(cli.main, ["video", "audio", str(src), "-o", str(tmp_path / "all.mkv")])
    assert r0.exit_code == 0, r0.output
    assert calls[-1] == ("multi", None)

    r1 = runner.invoke(cli.main, ["video", "audio", str(src), "--stream", "1", "-o", str(tmp_path / "one.mkv")])
    assert r1.exit_code == 0, r1.output
    assert calls[-1] == ("single", 1)

    r2 = runner.invoke(
        cli.main,
        ["video", "audio", str(src), "--stream", "0", "--stream", "2", "-o", str(tmp_path / "two.mkv")],
    )
    assert r2.exit_code == 0, r2.output
    assert calls[-1] == ("multi", (0, 2))


def test_detect_veo_frame_drops_sub_transition_noise_candidates() -> None:
    rng = np.random.default_rng(7)
    dark_noise = rng.integers(16, 48, size=(720, 1280, 3), dtype=np.uint8)
    det_noise = video_visible.detect_veo_frame(dark_noise, frame_index=3)
    assert det_noise.frame_index == 3
    assert det_noise.confidence == 0.0
    assert det_noise.region is None

    veo_cap = video_visible.open_video_capture(REPO_ROOT / "data" / "fixtures" / "visible" / "veo" / "example.mp4")
    try:
        ok, veo_frame = veo_cap.read()
    finally:
        veo_cap.release()
    assert ok
    det_veo = video_visible.detect_veo_frame(veo_frame, frame_index=0)
    assert det_veo.region is not None
    assert det_veo.confidence >= 0.55


def test_cleaned_veo_provider_original_does_not_false_positive_hailuo(tmp_path: Path) -> None:
    veo_orig = REPO_ROOT / "data" / "fixtures" / "visible" / "veo" / "provider-original.mp4"
    cleaned = tmp_path / "veo_cleaned.mp4"
    res = video.remove_video_visible(veo_orig, cleaned)
    assert res.detected_frames == 192

    after = video.identify_video(cleaned)
    assert after.visible_marks == ()
    assert after.visible_mark is None

    hailuo_orig = REPO_ROOT / "data" / "fixtures" / "visible" / "hailuo" / "provider-original.mp4"
    hailuo_rep = video.identify_video(hailuo_orig)
    assert hailuo_rep.visible_marks == ("hailuo",)
    assert hailuo_rep.visible_detected_frames == 124
