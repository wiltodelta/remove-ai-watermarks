"""Regression tests for the video SynthID removal engine."""

from __future__ import annotations

import csv
import math
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from remove_ai_watermarks import optional_deps, video_encoding, video_invisible, video_tools
from remove_ai_watermarks.video_synthid import (
    DEFAULT_VIDEO_SYNTHID_FPS,
    DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    DEFAULT_VIDEO_SYNTHID_NOISE_STD,
)

if TYPE_CHECKING:
    from typing import BinaryIO

ORACLE_MANIFEST = Path(__file__).resolve().parents[1] / "data" / "evaluations" / "video-synthid-oracle.csv"


def test_encoder_redirects_large_stderr_while_frames_are_streaming(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    diagnostic = "synthetic ffmpeg diagnostic"
    tail_diagnostic = "synthetic ffmpeg diagnostic tail"
    caplog.set_level("INFO", logger=video_encoding.__name__)
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            f"sys.stderr.buffer.write({diagnostic.encode()!r} + b'x' * 262144 + {tail_diagnostic.encode()!r}); "
            "sys.stderr.buffer.flush(); "
            "sys.stdin.buffer.read(); "
            "raise SystemExit(7)"
        ),
    ]
    encoder = video_encoding.start_raw_video_encoder(command)
    write_finished = threading.Event()
    write_errors: list[Exception] = []

    def write_frames() -> None:
        try:
            encoder.stdin.write(b"f" * 262144)
            encoder.stdin.flush()
        except Exception as exc:  # pragma: no cover - mutation cleanup path
            write_errors.append(exc)
        finally:
            write_finished.set()

    writer = threading.Thread(target=write_frames)
    writer.start()
    try:
        assert write_finished.wait(5), "stderr backpressure blocked the frame producer"
        assert write_errors == []
        with pytest.raises(RuntimeError, match=diagnostic):
            video_encoding.finish_raw_video_encoder(
                encoder,
                tmp_path / "unused.mp4",
                operation="synthetic encode",
            )
        assert "ffmpeg stderr truncated" in caplog.text
        assert tail_diagnostic in caplog.text
    finally:
        video_encoding.abort_raw_video_encoder(encoder)
        writer.join(timeout=5)


def test_availability_requires_both_optional_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        optional_deps,
        "module_available",
        lambda *names: set(names) <= {"torch"},
    )

    assert video_invisible.is_available() is False


def test_regeneration_rejects_noise_outside_unit_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        video_invisible.regenerate_video_candidate(
            tmp_path / "source.mp4",
            tmp_path / "candidate.mp4",
            noise_std=1.01,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fps": 0.0}, "fps must be at least 1"),
        ({"batch_size": 0}, "batch_size must be at least 1"),
        ({"duration": 0.0}, "duration must be positive"),
        ({"device": "tpu"}, "device must be auto, cuda, mps, or cpu"),
    ],
)
def test_regeneration_rejects_invalid_controls_before_probing(
    tmp_path: Path,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        video_invisible.regenerate_video_candidate(
            tmp_path / "source.mp4",
            tmp_path / "candidate.mp4",
            **kwargs,
        )


def test_fit_size_and_paired_psnr_validate_geometry() -> None:
    assert video_invisible._fit_size(1920, 1080, 512) == (512, 288)
    with pytest.raises(ValueError, match="positive"):
        video_invisible._fit_size(0, 1080, 512)
    with pytest.raises(ValueError, match="at least"):
        video_invisible._fit_size(1920, 1080, 4)

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    assert video_invisible.paired_psnr(frame, frame) == math.inf
    with pytest.raises(ValueError, match="matching shapes"):
        video_invisible.paired_psnr(frame, frame[:1])


def test_load_runtime_rejects_invalid_device_and_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="device must be"):
        video_invisible.load_video_vae_runtime(device="tpu")

    monkeypatch.setattr(video_invisible, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="diffusion extra"):
        video_invisible.load_video_vae_runtime()


def test_encoder_and_mux_commands_separate_streaming_frames_from_source_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "candidate.mp4"

    monkeypatch.setattr(video_tools.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    profile = video_encoding.VideoEncodeProfile(
        pixel_format="yuv420p",
        color_range="tv",
        color_space="bt709",
        color_transfer="bt709",
        color_primaries="bt709",
        time_base="1/90000",
    )

    command = video_encoding.raw_video_command(
        output,
        width=8,
        height=8,
        fps=2.0,
        crf=18,
        profile=profile,
    )

    metadata_index = command.index("-map_metadata")
    assert command[metadata_index + 1] == "-1"
    output_pixel_format_index = command.index("-pix_fmt", command.index("-c:v"))
    assert command[output_pixel_format_index + 1] == "yuv420p"
    assert command[command.index("-color_range") + 1] == "tv"
    assert command[command.index("-colorspace") + 1] == "bt709"
    assert command[command.index("-color_trc") + 1] == "bt709"
    assert command[command.index("-color_primaries") + 1] == "bt709"
    assert command[command.index("-enc_time_base:v") + 1] == "1/90000"
    assert command[command.index("-video_track_timescale") + 1] == "90000"
    assert command[command.index("-x264-params") + 1] == (
        "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited"
    )
    assert "pipe:0" in command
    assert str(source) not in command
    assert command[command.index("-map") + 1] == "0:v:0"
    assert command[command.index("-fps_mode") + 1] == "passthrough"
    assert "-shortest" not in command

    calls: list[list[str]] = []

    def fake_run(mux_command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(mux_command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(video_encoding.subprocess, "run", fake_run)
    encoded_video = tmp_path / "encoded.mp4"
    video_encoding.mux_encoded_video(
        encoded_video,
        source,
        output,
        strip_metadata=True,
    )

    mux_command = calls[0]
    assert mux_command.index(str(encoded_video)) < mux_command.index(str(source))
    assert mux_command[mux_command.index("-map") + 1] == "0:v:0"
    second_map = mux_command.index("-map", mux_command.index("-map") + 1)
    assert mux_command[second_map + 1] == "1:a?"
    assert mux_command[mux_command.index("-c") + 1] == "copy"
    assert mux_command[mux_command.index("-map_metadata") + 1] == "-1"
    assert "-shortest" not in mux_command


def test_mux_reports_bounded_disk_backed_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    diagnostic = b"synthetic mux diagnostic"
    tail_diagnostic = b"synthetic mux diagnostic tail"
    caplog.set_level("INFO", logger=video_encoding.__name__)
    monkeypatch.setattr(video_tools.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    def fake_run(_command: list[str], **kwargs: object) -> SimpleNamespace:
        stderr = cast("BinaryIO", kwargs["stderr"])
        stderr.write(diagnostic + b"x" * 262144 + tail_diagnostic)
        stderr.flush()
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(video_encoding.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=diagnostic.decode()):
        video_encoding.mux_encoded_video(
            tmp_path / "encoded.mp4",
            tmp_path / "source.mp4",
            tmp_path / "output.mp4",
            strip_metadata=True,
        )

    assert "ffmpeg stderr truncated" in caplog.text
    assert tail_diagnostic.decode() in caplog.text


def test_timestamped_encoder_reads_nut_and_passes_pts_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "candidate.mp4"
    monkeypatch.setattr(video_tools.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    command = video_encoding.raw_video_command(
        output,
        width=8,
        height=8,
        fps=24.0,
        crf=18,
        profile=video_encoding.VideoEncodeProfile(time_base="1/90000"),
        timestamped_input=True,
        copy_input_timestamps=True,
    )

    assert "-copyts" in command
    assert command[command.index("-f") : command.index("-f") + 4] == [
        "-f",
        "nut",
        "-i",
        "pipe:0",
    ]
    assert command[command.index("-fps_mode") + 1] == "passthrough"
    assert command[command.index("-avoid_negative_ts") + 1] == "disabled"
    assert command[command.index("-enc_time_base:v") + 1] == "1/90000"


def test_probe_encode_profile_preserves_supported_8_bit_properties(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(video_tools.shutil, "which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        video_encoding.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                '{"streams":[{"pix_fmt":"yuvj422p","color_range":"tv",'
                '"color_space":"bt709","color_transfer":"bt709",'
                '"color_primaries":"bt709","time_base":"2/180000",'
                '"start_pts":180000,"bits_per_raw_sample":"8"}]}'
            ),
            stderr="",
        ),
    )

    profile = video_encoding.probe_video_encode_profile(source)

    assert profile == video_encoding.VideoEncodeProfile(
        pixel_format="yuv422p",
        color_range="tv",
        color_space="bt709",
        color_transfer="bt709",
        color_primaries="bt709",
        time_base="1/90000",
        start_pts=180000,
        source_pixel_format="yuvj422p",
        component_depth=8,
    )


def test_probe_encode_profile_uses_compatible_defaults_without_ffprobe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_tools.shutil, "which", lambda _name: None)
    monkeypatch.setattr(video_tools, "_bundled_tools", lambda: None)

    assert video_encoding.probe_video_encode_profile(tmp_path / "source.mp4") == (video_encoding.VideoEncodeProfile())


@pytest.mark.parametrize("first_row", ["0.000000", "0.000000,", "0.000000,H.26[45] User Data Unregistered SEI message"])
def test_probe_video_timestamps_uses_best_effort_pts(
    first_row: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(video_tools.shutil, "which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        video_encoding.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"{first_row}\n0.041667\n",
            stderr="",
        ),
    )

    assert video_encoding.probe_video_timestamps(source) == (0.0, 0.041667)


def test_shipped_defaults_match_a_certified_manifest_row() -> None:
    """The shipped operating point must be one the provider oracle actually cleared.

    Pinning the constant alone was not enough. Only ``noise_std`` was asserted, so
    ``long_side`` and ``fps`` could move to an uncertified geometry with a green
    suite -- and they are two thirds of what the oracle was shown. Reading the
    manifest ties all three to the evidence: raising the resolution or the frame
    rate now fails here until a ``not_detected`` row exists for that exact triple.

    The tuple stops at three fields because the model is a fourth thing the oracle
    was shown and neither historical row recorded it. The manifest says
    ``unrecorded`` rather than leaving an ambiguous empty field. Add ``vae`` here in
    the same commit as the first row that records one.
    """
    with ORACLE_MANIFEST.open(newline="", encoding="utf-8") as stream:
        certified = {
            (float(row["noise_std"]), int(row["long_side"]), float(row["fps"]))
            for row in csv.DictReader(stream)
            # The manifest deliberately leaves unrecorded fields empty, so a row
            # missing part of its configuration certifies no triple and is skipped
            # rather than crashing the parse.
            if row["output_verdict"] == "not_detected"
            and all(row[field] for field in ("noise_std", "long_side", "fps"))
        }

    assert certified, f"{ORACLE_MANIFEST.name} records no fully configured certified row"
    shipped = (DEFAULT_VIDEO_SYNTHID_NOISE_STD, DEFAULT_VIDEO_SYNTHID_LONG_SIDE, DEFAULT_VIDEO_SYNTHID_FPS)
    assert shipped in certified, (
        f"shipped (noise_std, long_side, fps)={shipped} has no certified row in "
        f"{ORACLE_MANIFEST.name}; certified: {sorted(certified)}"
    )


def test_oracle_manifest_marks_missing_vae_identity_explicitly() -> None:
    with ORACLE_MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert rows
    assert all(row["vae"] for row in rows)
    assert all(row["vae"] == "unrecorded" or "/" in row["vae"] for row in rows)


def test_stream_batches_consumes_only_one_batch_ahead() -> None:
    consumed: list[int] = []

    def values():
        for value in range(5):
            consumed.append(value)
            yield value

    batches = video_invisible._stream_batches(values(), 2)

    assert next(iter(batches)) == [0, 1]
    assert consumed == [0, 1]
