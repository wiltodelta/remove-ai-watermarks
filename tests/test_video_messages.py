"""Known-message API gates, source integrity, and delivered-file publication."""

from types import SimpleNamespace

import numpy as np
import pytest

from remove_ai_watermarks import video_messages as video
from remove_ai_watermarks._internal.audio_quality import sha256_file

REFERENCE = "a5" * 32


def source_report(path, status="matched"):
    return video.VideoMessageReport(path, sha256_file(path), "videoseal", REFERENCE, status, None)


@pytest.mark.parametrize(("available", "enabled"), [(False, True), (True, False)])
def test_missing_or_skipped_checks_stay_unverified(tmp_path, monkeypatch, available, enabled):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(video, "is_available", lambda: available)
    monkeypatch.setattr(video, "load_detector", lambda *_: pytest.fail("No decoder should load"))
    report = video.identify_video_message(source, reference_hex=REFERENCE, check_message=enabled)
    assert report.status == "unverified"
    assert report.reading is None


@pytest.mark.parametrize("fault", ["decode", "model", "changed"])
def test_failed_inspection_closes_reader_and_cannot_be_negative(tmp_path, monkeypatch, fault):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    closed = []
    monkeypatch.setattr(video, "is_available", lambda: True)
    monkeypatch.setattr(video, "load_detector", lambda *_: object())

    def batches(_):
        try:
            yield np.zeros((1, 8, 8, 3))
            if fault == "decode":
                raise RuntimeError("broken decode")
            if fault == "changed":
                source.write_bytes(b"changed")
        finally:
            closed.append(True)

    def decode(*_):
        if fault == "model":
            raise RuntimeError("broken model")
        return np.zeros((1, 256))

    monkeypatch.setattr(video, "_video_batches", batches)
    monkeypatch.setattr(video, "decode_logits", decode)
    with pytest.raises(RuntimeError, match=r"broken|changed"):
        video.identify_video_message(source, reference_hex=REFERENCE)
    assert closed == [True]


@pytest.mark.parametrize(("count", "declared", "rotation"), [(13, 13, 90), (2, 3, 0), (0, 0, 0), (1, 1, 17)])
def test_decoder_rotation_batching_and_completeness(tmp_path, monkeypatch, count, declared, rotation):
    import av

    closed = []
    rgb = np.arange(6, dtype=np.uint8).reshape(2, 3, 1).repeat(3, axis=2)
    stream = SimpleNamespace(frames=declared, codec_context=SimpleNamespace(options={}))

    class Container:
        streams = SimpleNamespace(video=[stream])

        def __enter__(self):
            return self

        def __exit__(self, *_):
            closed.append(True)

        def decode(self, *_):
            for _ in range(count):
                yield SimpleNamespace(rotation=rotation, to_ndarray=lambda **_: rgb)

    monkeypatch.setattr(av, "open", lambda *a, **kw: Container())
    if count != declared or not count or rotation % 90:
        with pytest.raises((RuntimeError, ValueError)):
            list(video._video_batches(tmp_path / "video.mp4"))
    else:
        batches = list(video._video_batches(tmp_path / "video.mp4"))
        assert [len(batch) for batch in batches] == [8, 5]
        np.testing.assert_array_equal(batches[0][0], np.rot90(rgb) / np.float32(255))
    assert closed == [True]


def install_processing_stubs(monkeypatch, source, statuses):
    import remove_ai_watermarks.video_invisible as invisible
    import remove_ai_watermarks.video_quality as quality

    reports = []

    def identify(path, **kwargs):
        status = next(statuses)
        report = source_report(path, status)
        reports.append(report)
        return report

    monkeypatch.setattr(video, "identify_video_message", identify)
    monkeypatch.setattr(video, "_processing_source", lambda _: ((8, 8, 12.0), (), tuple(i / 12.0 for i in range(12))))
    monkeypatch.setattr(video, "_check_copied_audio", lambda *_: ())
    monkeypatch.setattr(invisible, "iter_sampled_frames", lambda *a, **kw: iter([np.zeros((8, 8, 3))] * 2))
    monkeypatch.setattr(
        invisible, "encode_video_frames", lambda frames, source, target, **kw: target.write_bytes(b"candidate")
    )
    monkeypatch.setattr(
        invisible, "regenerate_video_candidate", lambda source, target, **kw: target.write_bytes(b"vae")
    )
    monkeypatch.setattr(
        quality, "measure_video_fidelity", lambda *a, **kw: {"frames": 12, "source_psnr_db": 30, "source_ssim": 0.9}
    )
    from remove_ai_watermarks._internal import video_message_regeneration as regen

    monkeypatch.setattr(regen, "load_runtime", lambda _: object())
    return reports


@pytest.mark.parametrize("force", [False, True])
def test_matching_candidate_preserves_existing_output_unless_forced(tmp_path, monkeypatch, force):
    source, target = tmp_path / "source.mp4", tmp_path / "output.mp4"
    source.write_bytes(b"source")
    target.write_bytes(b"old")
    install_processing_stubs(monkeypatch, source, iter(["matched", "matched"]))
    if force:
        result = video.remove_video_message(source, target, reference_hex=REFERENCE, method="codec", force=True)
        assert result.verified_removed is False
        assert target.read_bytes() == b"candidate"
    else:
        with pytest.raises(video.VideoMessageVerificationIncomplete):
            video.remove_video_message(source, target, reference_hex=REFERENCE, method="codec")
        assert target.read_bytes() == b"old"
    assert not list(tmp_path.glob(".*.mp4"))


def test_auto_continues_to_fixed_vae_profile_and_only_publishes_negative(tmp_path, monkeypatch):
    source, target = tmp_path / "source.mp4", tmp_path / "output.mp4"
    source.write_bytes(b"source")
    install_processing_stubs(monkeypatch, source, iter(["matched", "matched", "not_matched"]))
    result = video.remove_video_message(source, target, reference_hex=REFERENCE)
    assert result.verified_removed is True
    assert result.method == "vae"
    assert result.noise_std == 0
    assert result.after.source == target
    assert target.read_bytes() == b"vae"


@pytest.mark.parametrize("source_status", ["not_matched", "unverified"])
def test_unqualified_source_skips_without_processing(tmp_path, monkeypatch, source_status):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(video, "identify_video_message", lambda *a, **kw: source_report(source, source_status))
    monkeypatch.setattr(video, "_processing_source", lambda *_: pytest.fail("Unqualified source was processed"))
    result = video.remove_video_message(source, tmp_path / "output.mp4", reference_hex=REFERENCE)
    assert result.skipped
    assert result.output is None
    assert result.verified_removed is None


def test_source_changed_before_publication_preserves_output(tmp_path, monkeypatch):
    source, target = tmp_path / "source.mp4", tmp_path / "output.mp4"
    source.write_bytes(b"source")
    target.write_bytes(b"old")
    install_processing_stubs(monkeypatch, source, iter(["matched", "not_matched"]))

    def changed(*_):
        source.write_bytes(b"changed")
        return ()

    monkeypatch.setattr(video, "_check_copied_audio", changed)
    with pytest.raises(RuntimeError, match="changed"):
        video.remove_video_message(source, target, reference_hex=REFERENCE, method="codec")
    assert target.read_bytes() == b"old"


@pytest.mark.parametrize("audio_codec", ["pcm_s16le", "aac"])
def test_actual_codec_delivery_preserves_reordered_audio_and_unverified_status(tmp_path, audio_codec):
    import json
    import shutil
    import subprocess

    from click.testing import CliRunner

    from remove_ai_watermarks.cli import main

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg required")
    source, target = tmp_path / "source.mov", tmp_path / "output.mov"
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=64x64:rate=12:duration=1",
        "-itsoffset",
        "0.25",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100:duration=1.25",
        "-itsoffset",
        "0.5",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=880:sample_rate=44100:duration=1.5",
        "-map",
        "1:a",
        "-map",
        "0:v",
        "-map",
        "2:a",
        "-c:v",
        "libx264",
        "-c:a",
        audio_codec,
        "-metadata:s:a:0",
        "language=eng",
        "-metadata:s:a:1",
        "language=rus",
        "-disposition:a:0",
        "0",
        "-disposition:a:1",
        "default",
        str(source),
    ]
    subprocess.run(command, check=True, capture_output=True)  # noqa: S603
    result = CliRunner().invoke(
        main,
        [
            "video",
            "message",
            "remove",
            str(source),
            "-o",
            str(target),
            "--reference",
            REFERENCE,
            "--method",
            "codec",
            "--long-side",
            "64",
            "--force",
            "--no-verify",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["verified_removed"] is None
    assert payload["after"]["status"] == "unverified"
    assert payload["copied_audio_streams"] == [0, 2]
    assert payload["quality"]["source_ssim"] > 0.9
    assert payload["quality"]["file"] == target.name
    assert target.is_file()


def test_pyav_display_rotation_matches_the_pixel_sampler(tmp_path):
    import shutil
    import subprocess

    import cv2

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    plain, rotated = tmp_path / "plain.mp4", tmp_path / "rotated.mp4"
    subprocess.run(  # noqa: S603 - Fixed local fixture encoder, no shell.
        [
            shutil.which("ffmpeg"),
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=48x32:rate=12:duration=0.25",
            "-c:v",
            "libx264",
            str(plain),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(  # noqa: S603 - Fixed local fixture encoder, no shell.
        [
            shutil.which("ffmpeg"),
            "-v",
            "error",
            "-y",
            "-i",
            str(plain),
            "-c",
            "copy",
            "-metadata:s:v:0",
            "rotate=90",
            str(rotated),
        ],
        check=True,
        capture_output=True,
    )
    capture = cv2.VideoCapture(str(rotated))
    try:
        ok, expected = capture.read()
    finally:
        capture.release()
    assert ok
    batches = list(video._video_batches(rotated))
    # Different linked swscale versions round YUV-to-RGB conversion differently.
    np.testing.assert_allclose(
        batches[0][0], cv2.cvtColor(expected, cv2.COLOR_BGR2RGB) / np.float32(255), rtol=0, atol=4 / 255
    )


@pytest.mark.parametrize("force", [False, True])
def test_auto_checks_every_fixed_profile_before_a_forced_final_match(tmp_path, monkeypatch, force):
    from remove_ai_watermarks import video_invisible

    source, target = tmp_path / "source.mp4", tmp_path / "output.mp4"
    source.write_bytes(b"source")
    target.write_bytes(b"old")
    install_processing_stubs(monkeypatch, source, iter(["matched"] * 5))
    noises = []

    def regenerate(source, target, **kwargs):
        noises.append(kwargs["noise_std"])
        target.write_bytes(b"vae")

    monkeypatch.setattr(video_invisible, "regenerate_video_candidate", regenerate)
    if force:
        result = video.remove_video_message(source, target, reference_hex=REFERENCE, force=True)
        assert result.verified_removed is False
        assert result.noise_std == 0.25
        assert target.read_bytes() == b"vae"
    else:
        with pytest.raises(video.VideoMessageVerificationIncomplete):
            video.remove_video_message(source, target, reference_hex=REFERENCE)
        assert target.read_bytes() == b"old"
    assert noises == [0, 0.15, 0.25]


def test_default_long_side_is_512_across_api_cli_and_mcp():
    import inspect as pyinspect

    from remove_ai_watermarks import cli, mcp_server

    sig = pyinspect.signature(video.remove_video_message)
    assert sig.parameters["long_side"].default == 512
    opt = next(p for p in cli.cmd_video_message_remove.params if p.name == "long_side")
    assert opt.default == 512
    schema = mcp_server._TOOL_SCHEMAS["video_message_remove"]["properties"]["long_side"]
    assert schema["default"] == 512


def test_premature_sampler_eof_is_rejected_by_ffprobe_timestamp_inventory(tmp_path, monkeypatch):
    import remove_ai_watermarks.video_quality as quality

    source, target = tmp_path / "source.mp4", tmp_path / "output.mp4"
    source.write_bytes(b"source")
    target.write_bytes(b"old")
    install_processing_stubs(monkeypatch, source, iter(["matched", "not_matched"]))
    # Source has 24 frames at 24 fps -> 12 fps sampling expects 12 frames, but OpenCV stopped at 5.
    monkeypatch.setattr(video, "_processing_source", lambda _: ((64, 64, 24.0), (), tuple(i / 24.0 for i in range(24))))
    monkeypatch.setattr(
        quality,
        "measure_video_fidelity",
        lambda *a, **kw: {"frames": 5, "source_psnr_db": 32.0, "source_ssim": 0.96},
    )
    with pytest.raises(RuntimeError, match="frame count"):
        video.remove_video_message(source, target, reference_hex=REFERENCE, method="codec")
    assert target.read_bytes() == b"old"


@pytest.mark.parametrize("force", [False, True])
def test_degraded_codec_candidate_fails_quality_gate_unless_forced(tmp_path, monkeypatch, force):
    import remove_ai_watermarks.video_quality as quality

    source, target = tmp_path / "source.mp4", tmp_path / "output.mp4"
    source.write_bytes(b"source")
    target.write_bytes(b"old")
    install_processing_stubs(monkeypatch, source, iter(["matched", "not_matched"]))
    monkeypatch.setattr(
        quality,
        "measure_video_fidelity",
        lambda *a, **kw: {"frames": 12, "source_psnr_db": 19.5, "source_ssim": 0.55},
    )
    if force:
        result = video.remove_video_message(source, target, reference_hex=REFERENCE, method="codec", force=True)
        assert result.verified_removed is True
        assert result.quality_passed is False
        assert target.read_bytes() == b"candidate"
    else:
        with pytest.raises(video.VideoMessageVerificationIncomplete, match="quality"):
            video.remove_video_message(source, target, reference_hex=REFERENCE, method="codec")
        assert target.read_bytes() == b"old"
