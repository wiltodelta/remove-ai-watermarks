"""Tests for the video processing API and CLI."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image, ImageDraw, ImageFont

from remove_ai_watermarks.cli import main
from remove_ai_watermarks.metadata import C2PA_UUID

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


_MP4_FTYP = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
_VIDEO_PAYLOAD = b"synthetic-video-payload"
_TC260_AIGC = (
    b'{"Label":"1","ContentProducer":"00119144030008867405X210002",'
    b'"ProduceID":"sample-001","ReservedCode1":"","ContentPropagator":"",'
    b'"PropagateID":"","ReservedCode2":""}'
)


def _box(box_type: bytes, payload: bytes) -> bytes:
    return (8 + len(payload)).to_bytes(4, "big") + box_type + payload


def _video_with_c2pa(path: Path) -> Path:
    manifest = C2PA_UUID + b"OpenAI trainedAlgorithmicMedia c2pa.watermarked.unbound"
    path.write_bytes(_MP4_FTYP + _box(b"uuid", manifest) + _box(b"mdat", _VIDEO_PAYLOAD))
    return path


def _metadata_key(name: bytes) -> bytes:
    return (8 + len(name)).to_bytes(4, "big") + b"mdta" + name


def _metadata_value(index: int, value: bytes) -> bytes:
    data = _box(b"data", b"\x00\x00\x00\x01\x00\x00\x00\x00" + value)
    return _box(index.to_bytes(4, "big"), data)


def _video_with_tc260(path: Path, *, media_payload: bytes = _VIDEO_PAYLOAD) -> Path:
    keys = _box(
        b"keys",
        b"\x00\x00\x00\x00" + (2).to_bytes(4, "big") + _metadata_key(b"AIGC") + _metadata_key(b"title"),
    )
    ilst = _box(
        b"ilst",
        _metadata_value(1, _TC260_AIGC) + _metadata_value(2, b"standard title"),
    )
    meta = _box(b"meta", b"\x00\x00\x00\x00" + keys + ilst)
    path.write_bytes(_MP4_FTYP + _box(b"mdat", media_payload) + _box(b"moov", _box(b"udta", meta)))
    return path


def _video_with_comfyui_metadata(path: Path, *, media_payload: bytes = _VIDEO_PAYLOAD) -> Path:
    workflow = b'{"nodes":[{"type":"KSampler"}],"app":"ComfyUI"}'
    prompt = b'{"1":{"class_type":"KSampler"}}'
    keys = _box(
        b"keys",
        b"\x00\x00\x00\x00"
        + (4).to_bytes(4, "big")
        + _metadata_key(b"workflow")
        + _metadata_key(b"prompt")
        + _metadata_key(b"title")
        + _metadata_key(b"software"),
    )
    ilst = _box(
        b"ilst",
        _metadata_value(1, workflow)
        + _metadata_value(2, prompt)
        + _metadata_value(3, b"standard title")
        + _metadata_value(4, b"ordinary editor"),
    )
    meta = _box(b"meta", b"\x00\x00\x00\x00" + keys + ilst)
    path.write_bytes(_MP4_FTYP + _box(b"mdat", media_payload) + _box(b"moov", _box(b"udta", meta)))
    return path


def _hdlr(handler: bytes) -> bytes:
    # version/flags + pre_defined + handler_type + reserved[3]; the TC260 walker
    # never parses it, but a real-shaped hdlr keeps the fixture honest.
    return _box(b"hdlr", b"\x00\x00\x00\x00" + b"\x00\x00\x00\x00" + handler + b"\x00" * 12)


def _quicktime_meta(*children: bytes) -> bytes:
    # QuickTime form: no 4-byte FullBox header before the children.
    return _box(b"meta", b"".join(children))


def _video_with_tc260_moov_meta(path: Path, *, media_payload: bytes = _VIDEO_PAYLOAD) -> Path:
    """Doubao's iOS export form: a QuickTime ``meta`` box as a direct ``moov`` child."""
    keys = _box(
        b"keys",
        b"\x00\x00\x00\x00"
        + (2).to_bytes(4, "big")
        + _metadata_key(b"com.apple.quicktime.artwork")
        + _metadata_key(b"AIGC"),
    )
    ilst = _box(
        b"ilst",
        _metadata_value(1, b'{"source_type":"","data":{"product":"doubao"}}') + _metadata_value(2, _TC260_AIGC),
    )
    meta = _quicktime_meta(_hdlr(b"mdta"), keys, ilst)
    path.write_bytes(_MP4_FTYP + _box(b"mdat", media_payload) + _box(b"moov", meta))
    return path


def _video_with_tc260_mdir_list(path: Path, *, media_payload: bytes = _VIDEO_PAYLOAD) -> Path:
    """Doubao's iOS QuickTime metadata list: ``udta.meta(hdlr=mdir)/ilst`` data
    items with numeric indices and no ``keys`` box at all."""
    ilst = _box(
        b"ilst",
        _metadata_value(0, b"vid:standard-video-id") + _metadata_value(0, _TC260_AIGC),
    )
    meta = _quicktime_meta(_hdlr(b"mdir"), ilst)
    path.write_bytes(_MP4_FTYP + _box(b"mdat", media_payload) + _box(b"moov", _box(b"udta", meta)))
    return path


def _ebml_size(value: int) -> bytes:
    for length in range(1, 9):
        if value < (1 << (7 * length)) - 1:
            return ((1 << (7 * length)) | value).to_bytes(length, "big")
    raise ValueError("EBML test value is too large")


def _ebml_element(element_id: bytes, payload: bytes) -> bytes:
    return element_id + _ebml_size(len(payload)) + payload


def _video_with_tc260_ebml(path: Path, *, value: bytes = _TC260_AIGC) -> Path:
    simple_tag = _ebml_element(
        b"\x67\xc8",
        _ebml_element(b"\x45\xa3", b"AIGC") + _ebml_element(b"\x44\x87", value),
    )
    tags = _ebml_element(b"\x12\x54\xc3\x67", _ebml_element(b"\x73\x73", simple_tag))
    segment = _ebml_element(b"\x18\x53\x80\x67", tags)
    path.write_bytes(_ebml_element(b"\x1a\x45\xdf\xa3", b"") + segment)
    return path


def _riff_chunk(chunk_id: bytes, payload: bytes) -> bytes:
    return chunk_id + len(payload).to_bytes(4, "little") + payload + (b"\x00" if len(payload) & 1 else b"")


def _video_with_tc260_avi(path: Path, *, value: bytes = _TC260_AIGC) -> Path:
    info = _riff_chunk(b"AIGC", value) + _riff_chunk(b"INAM", b"standard title\x00")
    body = b"AVI " + _riff_chunk(b"LIST", b"INFO" + info) + _riff_chunk(b"JUNK", _VIDEO_PAYLOAD)
    path.write_bytes(b"RIFF" + len(body).to_bytes(4, "little") + body)
    return path


def _amf0_string(value: bytes) -> bytes:
    return b"\x02" + len(value).to_bytes(2, "big") + value


def _video_with_tc260_flv(path: Path, *, value: bytes = _TC260_AIGC) -> Path:
    payload = (
        _amf0_string(b"onMetaData")
        + b"\x08\x00\x00\x00\x02"
        + len(b"AIGC").to_bytes(2, "big")
        + b"AIGC"
        + _amf0_string(value)
        + len(b"duration").to_bytes(2, "big")
        + b"duration"
        + b"\x00"
        + b"\x00\x00\x00\x00\x00\x00\x00\x00"
        + b"\x00\x00\x09"
    )
    tag_header = b"\x12" + len(payload).to_bytes(3, "big") + b"\x00" * 7
    path.write_bytes(
        b"FLV\x01\x05\x00\x00\x00\x09"
        + b"\x00\x00\x00\x00"
        + tag_header
        + payload
        + (11 + len(payload)).to_bytes(4, "big")
    )
    return path


_LEGACY_VIDEO_CASES = (
    (".avi", _video_with_tc260_avi),
    (".flv", _video_with_tc260_flv),
)


def _stamp_gray_mark(
    frame: np.ndarray,
    mark: Image.Image | np.ndarray,
    *,
    x: int,
    y: int,
    opacity: float,
) -> None:
    """Alpha-composite a grayscale synthetic mark onto a BGR test frame."""
    mark_array = np.asarray(mark, dtype=np.float32)
    height, width = mark_array.shape
    alpha = mark_array[:, :, None] / 255 * opacity
    crop = frame[y : y + height, x : x + width].astype(np.float32)
    frame[y : y + height, x : x + width] = np.clip(
        crop * (1 - alpha) + 255 * alpha,
        0,
        255,
    ).astype(np.uint8)


def _independent_sora_mark() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Render a Sora-like mark without using the detector's template."""
    mark = Image.new("L", (180, 64), 0)
    draw = ImageDraw.Draw(mark)
    draw.ellipse((1, 14, 32, 54), fill=255)
    draw.ellipse((25, 8, 62, 58), fill=255)
    draw.ellipse((15, 20, 28, 44), fill=0)
    draw.ellipse((37, 18, 50, 43), fill=0)
    try:
        font = ImageFont.load_default(size=49)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((68, 1), "Sora", font=font, fill=255, stroke_width=1)
    mark_array = cv2.resize(np.asarray(mark), (124, 44), interpolation=cv2.INTER_AREA)
    x, y = 620, 398
    return mark_array, (x, y, 124, 44)


def _independent_sora_frame() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Stamp the independent Sora-like mark onto a flat test frame."""
    frame = np.full((480, 840, 3), 36, dtype=np.uint8)
    mark_array, region = _independent_sora_mark()
    x, y, _width, _height = region
    _stamp_gray_mark(frame, mark_array, x=x, y=y, opacity=0.78)
    return frame, region


def _moving_video_background(frame_index: int) -> np.ndarray:
    """Return a smooth moving background with known clean pixels."""
    x = np.arange(840, dtype=np.float32)[None, :]
    y = np.arange(480, dtype=np.float32)[:, None]
    luma = 42 + 10 * np.sin((x + frame_index * 5) / 38) + 5 * np.cos((y - frame_index * 2) / 54)
    frame = np.stack(
        (
            np.clip(luma - 5, 0, 255),
            np.clip(luma + 1, 0, 255),
            np.clip(luma + 7, 0, 255),
        ),
        axis=2,
    ).astype(np.uint8)
    moving_x = 30 + frame_index * 8
    cv2.rectangle(frame, (moving_x, 70), (moving_x + 72, 132), (80, 140, 210), -1)
    cv2.line(frame, (0, 220 + frame_index), (839, 250 + frame_index), (110, 70, 45), 3)
    return frame


def _ffmpeg_test_tools() -> tuple[str, str]:
    """Return real ffmpeg tools or skip the integration test."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("full-clip video integration test requires ffmpeg and ffprobe")
    return ffmpeg, ffprobe


def _write_synthetic_sora_clip(
    path: Path,
    frames: list[np.ndarray],
    *,
    fps: float,
    ffmpeg: str,
    start_offset: float = 0.0,
) -> None:
    """Encode a synthetic marked MP4 with AAC audio and C2PA provenance."""
    height, width = frames[0].shape[:2]
    duration = len(frames) / fps
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
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=880:sample_rate=48000:duration={duration:.12g}",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "8",
        "-x264-params",
        "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_primaries",
        "bt709",
        "-c:a",
        "aac",
        "-shortest",
        "-video_track_timescale",
        "90000",
        "-movflags",
        "+faststart",
    ]
    if start_offset:
        command.extend(["-output_ts_offset", f"{start_offset:.12g}"])
    command.append(str(path))
    subprocess.run(  # noqa: S603
        command,
        input=b"".join(frame.tobytes() for frame in frames),
        capture_output=True,
        check=True,
    )
    with path.open("ab") as stream:
        stream.write(
            _box(
                b"uuid",
                C2PA_UUID + b"OpenAI Sora trainedAlgorithmicMedia c2pa.watermarked.unbound",
            )
        )


def _write_vfr_sora_clip(
    path: Path,
    frames: list[np.ndarray],
    *,
    durations: list[float],
    ffmpeg: str,
    start_offset: float = 0.0,
) -> None:
    """Encode marked stills at deliberately irregular presentation times."""
    assert len(frames) == len(durations)
    frame_paths: list[Path] = []
    for index, frame in enumerate(frames):
        frame_path = path.with_name(f"{path.stem}-frame-{index:03d}.png")
        assert cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)
    manifest = path.with_suffix(".ffconcat")
    lines = ["ffconcat version 1.0"]
    for frame_path, duration in zip(frame_paths, durations, strict=True):
        lines.extend((f"file '{frame_path.as_posix()}'", f"duration {duration:.12g}"))
    lines.append(f"file '{frame_paths[-1].as_posix()}'")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total_duration = sum(durations)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest),
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=880:sample_rate=48000:duration={total_duration:.12g}",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-frames:v",
        str(len(frames)),
        "-fps_mode",
        "vfr",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "8",
        "-bf",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-video_track_timescale",
        "90000",
    ]
    if start_offset:
        command.extend(["-output_ts_offset", f"{start_offset:.12g}"])
    command.append(str(path))
    subprocess.run(command, capture_output=True, check=True)  # noqa: S603
    with path.open("ab") as stream:
        stream.write(
            _box(
                b"uuid",
                C2PA_UUID + b"OpenAI Sora trainedAlgorithmicMedia c2pa.watermarked.unbound",
            )
        )


def _decode_video(path: Path) -> tuple[list[np.ndarray], float]:
    """Decode every frame through the same OpenCV boundary as production."""
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    return frames, fps


def _video_frame_timestamps(path: Path, *, ffprobe: str) -> list[float]:
    """Read display timestamps from the first video stream."""
    result = subprocess.run(  # noqa: S603
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    frames = json.loads(result.stdout)["frames"]
    return [float(frame["best_effort_timestamp_time"]) for frame in frames]


def _audio_bitstream(path: Path, *, ffmpeg: str) -> bytes:
    """Extract copied AAC packets in a container-independent form."""
    result = subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-c:a",
            "copy",
            "-f",
            "adts",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return result.stdout


def _container_duration(path: Path, *, ffprobe: str) -> float:
    """Read the container duration from the real ffprobe boundary."""
    result = subprocess.run(  # noqa: S603
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return float(result.stdout.strip())


def _video_stream_info(path: Path, *, ffprobe: str) -> dict[str, str]:
    """Read source-sensitive video stream properties through ffprobe."""
    result = subprocess.run(  # noqa: S603
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=pix_fmt,color_range,color_space,color_transfer,color_primaries,time_base",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    streams = json.loads(result.stdout)["streams"]
    assert len(streams) == 1
    return streams[0]


def _stream_start_times(path: Path, *, ffprobe: str) -> dict[str, float]:
    """Read the first video and audio stream start times."""
    result = subprocess.run(  # noqa: S603
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,start_time",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return {
        stream["codec_type"]: float(stream["start_time"])
        for stream in json.loads(result.stdout)["streams"]
        if stream["codec_type"] in {"video", "audio"}
    }


def _regeneration_metrics(
    *,
    frames: int = 24,
    fps: float = 12.0,
    width: int = 512,
    height: int = 288,
    psnr_db: float = 22.0,
    temporal_residual_ratio: float = 1.2,
):
    from remove_ai_watermarks.video_invisible import RegenerationMetrics

    return RegenerationMetrics(
        frames=frames,
        fps=fps,
        width=width,
        height=height,
        psnr_db=psnr_db,
        temporal_residual_ratio=temporal_residual_ratio,
    )


class TestVideoDependencies:
    def test_visible_runtime_reports_video_extra(self, monkeypatch):
        from remove_ai_watermarks import optional_deps
        from remove_ai_watermarks.video import _require_video_runtime

        monkeypatch.setattr(optional_deps, "module_available", lambda *_names: False)

        with pytest.raises(RuntimeError, match=r"remove-ai-watermarks\[video\]"):
            _require_video_runtime()


class TestVideoMetadataApi:
    def test_inspects_and_removes_comfyui_mp4_metadata(self, tmp_path: Path):
        from remove_ai_watermarks.video import inspect_video_metadata, remove_video_metadata

        source = _video_with_comfyui_metadata(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"

        report = inspect_video_metadata(source)
        assert set(report.markers) >= {"workflow", "prompt"}

        result = remove_video_metadata(source, output)
        cleaned = output.read_bytes()
        assert result.remaining == {}
        assert len(cleaned) == source.stat().st_size
        assert _VIDEO_PAYLOAD in cleaned
        assert b"standard title" in cleaned
        assert b"ordinary editor" in cleaned
        assert b"ComfyUI" not in cleaned

    def test_top_level_api_is_lazy_exported(self):
        import remove_ai_watermarks as raiw

        assert raiw.identify_video is not None
        assert raiw.inspect_video_metadata is not None
        assert raiw.remove_video_all is not None
        assert raiw.remove_video_batch is not None
        assert raiw.remove_video_invisible is not None
        assert raiw.remove_video_metadata is not None
        assert raiw.remove_video_visible is not None

    def test_inspects_video_metadata(self, tmp_path: Path):
        from remove_ai_watermarks.video import inspect_video_metadata

        source = _video_with_c2pa(tmp_path / "source.mp4")

        report = inspect_video_metadata(source)

        assert report.source == source
        assert report.has_ai_metadata is True
        assert report.markers

    def test_removes_metadata_without_touching_video_payload(self, tmp_path: Path):
        from remove_ai_watermarks.video import remove_video_metadata

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"

        result = remove_video_metadata(source, output)

        assert result.output == output
        assert result.detected
        assert result.remaining == {}
        assert result.audio.stream_action == "copied_if_present"
        assert result.audio.watermark_status == "unverified"
        assert _VIDEO_PAYLOAD in output.read_bytes()
        assert C2PA_UUID not in output.read_bytes()

    def test_default_output_preserves_source(self, tmp_path: Path):
        from remove_ai_watermarks.video import remove_video_metadata

        source = _video_with_c2pa(tmp_path / "source.mp4")
        original = source.read_bytes()

        result = remove_video_metadata(source)

        assert result.output == tmp_path / "source_clean.mp4"
        assert result.output.exists()
        assert source.read_bytes() == original

    @pytest.mark.parametrize("suffix", [".mp4", ".mov"])
    def test_inspects_native_tc260_metadata(self, tmp_path: Path, suffix: str):
        from remove_ai_watermarks.video import inspect_video_metadata

        source = _video_with_tc260(tmp_path / f"source{suffix}")

        report = inspect_video_metadata(source)

        assert report.has_ai_metadata is True
        assert report.markers["aigc_label"].endswith("producer 00119144030008867405X210002")

    def test_inspects_native_tc260_metadata_after_large_media_payload(self, tmp_path: Path):
        from remove_ai_watermarks.video import inspect_video_metadata

        source = _video_with_tc260(
            tmp_path / "source.mp4",
            media_payload=b"x" * (1024 * 1024),
        )

        report = inspect_video_metadata(source)

        assert report.has_ai_metadata is True
        assert "aigc_label" in report.markers

    def test_removes_native_tc260_metadata_without_touching_media_or_standard_tag(self, tmp_path: Path):
        from remove_ai_watermarks.video import remove_video_metadata

        source = _video_with_tc260(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"

        result = remove_video_metadata(source, output)
        cleaned = output.read_bytes()

        assert result.detected["aigc_label"].startswith("China AIGC label")
        assert result.remaining == {}
        assert len(cleaned) == source.stat().st_size
        assert _VIDEO_PAYLOAD in cleaned
        assert b"standard title" in cleaned
        assert b"AIGC" not in cleaned
        assert _TC260_AIGC not in cleaned

    @pytest.mark.parametrize("suffix", [".mp4", ".mov"])
    def test_inspects_native_tc260_in_quicktime_moov_meta(self, tmp_path: Path, suffix: str):
        # Doubao's iOS export stores the label in a QuickTime-form meta box
        # (no FullBox header) hanging directly off moov, not under udta.
        from remove_ai_watermarks.video import inspect_video_metadata

        source = _video_with_tc260_moov_meta(tmp_path / f"source{suffix}")

        report = inspect_video_metadata(source)

        assert report.has_ai_metadata is True
        assert report.markers["aigc_label"].endswith("producer 00119144030008867405X210002")

    def test_removes_native_tc260_in_quicktime_moov_meta(self, tmp_path: Path):
        from remove_ai_watermarks.video import remove_video_metadata

        source = _video_with_tc260_moov_meta(tmp_path / "source.mov")
        output = tmp_path / "clean.mov"

        result = remove_video_metadata(source, output)
        cleaned = output.read_bytes()

        assert result.detected["aigc_label"].startswith("China AIGC label")
        # The artwork item's Doubao product JSON is app-export provenance, a
        # separate signal the TC260 blanker does not touch -- as on the real
        # Doubao iOS export this fixture mirrors.
        assert result.remaining == {"app_provenance": "App export provenance (ByteDance Doubao)"}
        assert len(cleaned) == source.stat().st_size
        assert _VIDEO_PAYLOAD in cleaned
        assert _TC260_AIGC not in cleaned

    def test_inspects_native_tc260_in_quicktime_mdir_metadata_list(self, tmp_path: Path):
        # The second Doubao iOS variant: a QuickTime metadata list under
        # udta.meta with hdlr=mdir and ilst data items but no keys box.
        from remove_ai_watermarks.video import inspect_video_metadata

        source = _video_with_tc260_mdir_list(tmp_path / "source.mov")

        report = inspect_video_metadata(source)

        assert report.has_ai_metadata is True
        assert "aigc_label" in report.markers

    def test_removes_native_tc260_in_quicktime_mdir_metadata_list(self, tmp_path: Path):
        from remove_ai_watermarks.video import remove_video_metadata

        source = _video_with_tc260_mdir_list(tmp_path / "source.mov")
        output = tmp_path / "clean.mov"

        result = remove_video_metadata(source, output)
        cleaned = output.read_bytes()

        assert result.detected["aigc_label"].startswith("China AIGC label")
        assert result.remaining == {}
        assert len(cleaned) == source.stat().st_size
        assert _VIDEO_PAYLOAD in cleaned
        # The keyless entry has no key name to blank; only the JSON is spaced
        # out, and the neighboring standard ilst item survives untouched.
        assert _TC260_AIGC not in cleaned
        assert b"vid:standard-video-id" in cleaned

    def test_streams_large_isobmff_without_full_file_read(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks.video import inspect_video_metadata, remove_video_metadata

        media_payload = b"x" * (8 * 1024 * 1024)
        source = _video_with_tc260(tmp_path / "source.mp4", media_payload=media_payload)
        with source.open("ab") as stream:
            stream.write(
                _box(
                    b"uuid",
                    C2PA_UUID + b"OpenAI trainedAlgorithmicMedia c2pa.watermarked.unbound",
                )
            )
        output = tmp_path / "clean.mp4"
        source_size = source.stat().st_size
        source_media_digest = hashlib.sha256(media_payload).digest()
        original_read_bytes = type(source).read_bytes
        source_report = inspect_video_metadata(source)

        assert source_report.has_ai_metadata is True
        assert "synthid_watermark" in source_report.markers
        assert "aigc_label" in source_report.markers

        def reject_source_read_bytes(path: Path) -> bytes:
            if path.resolve() == source.resolve():
                raise AssertionError("video metadata removal must not read the complete source")
            return original_read_bytes(path)

        monkeypatch.setattr(type(source), "read_bytes", reject_source_read_bytes)

        result = remove_video_metadata(source, output)
        cleaned = output.read_bytes()
        mdat_start = cleaned.index(b"mdat") + 4
        mdat_end = mdat_start + len(media_payload)

        assert result.remaining == {}
        assert output.stat().st_size == source_size
        assert hashlib.sha256(cleaned[mdat_start:mdat_end]).digest() == source_media_digest
        assert b"standard title" in cleaned
        assert C2PA_UUID not in cleaned
        assert _TC260_AIGC not in cleaned

    def test_ignores_generic_mp4_aigc_tag_without_tc260_fields(self, tmp_path: Path):
        from remove_ai_watermarks.video import inspect_video_metadata

        source = _video_with_tc260(tmp_path / "source.mp4")
        source.write_bytes(source.read_bytes().replace(_TC260_AIGC, b'{"description":"' + b"x" * 146 + b'"}'))

        report = inspect_video_metadata(source)

        assert report.has_ai_metadata is False
        assert report.markers == {}

    @pytest.mark.parametrize("suffix", [".mkv", ".webm"])
    def test_inspects_native_tc260_ebml_metadata(self, tmp_path: Path, suffix: str):
        from remove_ai_watermarks.video import inspect_video_metadata

        source = _video_with_tc260_ebml(tmp_path / f"source{suffix}")

        report = inspect_video_metadata(source)

        assert report.has_ai_metadata is True
        assert report.markers["aigc_label"].endswith("producer 00119144030008867405X210002")

    def test_ignores_generic_ebml_aigc_tag_without_tc260_fields(self, tmp_path: Path):
        from remove_ai_watermarks.video import inspect_video_metadata

        source = _video_with_tc260_ebml(
            tmp_path / "source.mkv",
            value=b'{"description":"ordinary application metadata"}',
        )

        report = inspect_video_metadata(source)

        assert report.has_ai_metadata is False
        assert report.markers == {}

    @pytest.mark.parametrize(
        ("suffix", "factory"),
        _LEGACY_VIDEO_CASES,
    )
    def test_inspects_native_tc260_legacy_video_metadata(
        self,
        tmp_path: Path,
        suffix: str,
        factory: Callable[..., Path],
    ):
        from remove_ai_watermarks.video import inspect_video_metadata

        source = factory(tmp_path / f"source{suffix}")

        report = inspect_video_metadata(source)

        assert report.has_ai_metadata is True
        assert report.markers["aigc_label"].endswith("producer 00119144030008867405X210002")

    @pytest.mark.parametrize(
        ("suffix", "factory"),
        _LEGACY_VIDEO_CASES,
    )
    def test_ignores_generic_legacy_video_aigc_tag(
        self,
        tmp_path: Path,
        suffix: str,
        factory: Callable[..., Path],
    ):
        from remove_ai_watermarks.video import inspect_video_metadata

        source = factory(
            tmp_path / f"source{suffix}",
            value=b'{"description":"ordinary application metadata"}',
        )

        report = inspect_video_metadata(source)

        assert report.has_ai_metadata is False
        assert report.markers == {}

    def test_rejects_image_input(self, tmp_clean_png: Path):
        from remove_ai_watermarks.video import inspect_video_metadata

        with pytest.raises(ValueError, match="Unsupported video format"):
            inspect_video_metadata(tmp_clean_png)

    def test_rejects_image_with_video_extension(self, tmp_clean_png: Path, tmp_path: Path):
        from remove_ai_watermarks.video import inspect_video_metadata

        disguised = tmp_path / "image.mp4"
        disguised.write_bytes(tmp_clean_png.read_bytes())

        with pytest.raises(ValueError, match="does not match"):
            inspect_video_metadata(disguised)

    def test_rejects_output_container_change(self, tmp_path: Path):
        from remove_ai_watermarks.video import remove_video_metadata

        source = _video_with_c2pa(tmp_path / "source.mp4")

        with pytest.raises(ValueError, match="must match"):
            remove_video_metadata(source, tmp_path / "clean.mov")


class TestVideoProvenanceApi:
    def test_c2pa_platform_keeps_the_bytedance_surface_name(self):
        from remove_ai_watermarks.video import _platform_from_video_metadata

        assert _platform_from_video_metadata({"issuer": "BytePlus (ByteDance)"}) == "BytePlus (ByteDance)"
        assert _platform_from_video_metadata({"issuer": "ByteDance (Volcano Engine)"}) == "ByteDance Volcano Engine"

    def test_identifies_metadata_without_pixel_scan(self, tmp_path: Path):
        from remove_ai_watermarks.video import identify_video

        source = _video_with_c2pa(tmp_path / "source.mp4")

        report = identify_video(source, check_visible=False)

        assert report.source == source
        assert report.is_ai_generated is True
        assert report.confidence == "high"
        assert report.platform == "OpenAI (ChatGPT / GPT Image / DALL·E / Sora)"
        assert report.visible_mark is None
        assert report.total_frames is None
        assert report.has_ai_metadata is True
        assert report.metadata_markers
        assert "Visible video-mark detection was skipped." in report.caveats

    def test_identifies_stable_visible_mark_with_shared_arbiter(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video_visible
        from remove_ai_watermarks.video import VIDEO_VISIBLE_MARKS, identify_video
        from remove_ai_watermarks.video_visible import FrameLocalization, VideoScan

        source = _video_with_c2pa(tmp_path / "source.mp4")
        box = (4, 4, 20, 8)
        stable = VideoScan(
            width=64,
            height=64,
            fps=24.0,
            detections=tuple(FrameLocalization(index, 0.66, box) for index in range(5)),
        )
        empty = VideoScan(
            width=64,
            height=64,
            fps=24.0,
            detections=tuple(FrameLocalization(index, 0.0, None) for index in range(5)),
        )

        def fake_scan(
            _source: Path,
            marks: tuple[str, ...],
            *,
            collect_timestamps: bool,
        ) -> dict[str, VideoScan]:
            assert collect_timestamps is False
            return {candidate: stable if candidate == "sora" else empty for candidate in marks}

        monkeypatch.setattr(video_visible, "scan_video_marks", fake_scan)

        report = identify_video(source)

        assert report.platform == "OpenAI Sora"
        assert report.visible_mark == "sora"
        assert report.visible_detected_frames == 5
        assert report.total_frames == 5
        assert tuple(VIDEO_VISIBLE_MARKS) == ("sora", "veo", "seedance", "doubao", "dola", "hailuo", "kling")

    def test_reports_unknown_instead_of_clean(self, tmp_path: Path):
        from remove_ai_watermarks.video import identify_video

        source = tmp_path / "source.mp4"
        source.write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))

        report = identify_video(source, check_visible=False)

        assert report.is_ai_generated is None
        assert report.confidence == "unknown"
        assert report.has_ai_metadata is False
        assert any("not proof" in caveat for caveat in report.caveats)


class TestVideoMetadataCli:
    def test_help(self):
        runner = CliRunner()

        result = runner.invoke(main, ["video", "metadata", "--help"])

        assert result.exit_code == 0, result.output
        assert "AI metadata" in result.output

    def test_check_reports_metadata(self, tmp_path: Path):
        runner = CliRunner()
        source = _video_with_c2pa(tmp_path / "source.mp4")

        result = runner.invoke(main, ["video", "metadata", str(source), "--check"])

        assert result.exit_code == 0, result.output
        assert "AI metadata detected" in result.output
        assert "not the same as 'clean'" not in result.output

    def test_remove_reports_output(self, tmp_path: Path):
        runner = CliRunner()
        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"

        result = runner.invoke(main, ["video", "metadata", str(source), "--remove", "-o", str(output)])

        assert result.exit_code == 0, result.output
        assert "AI metadata stripped" in result.output
        assert "not the same as 'clean'" in result.output
        assert C2PA_UUID not in output.read_bytes()

    def test_rejects_image_input(self, tmp_clean_png: Path):
        runner = CliRunner()

        result = runner.invoke(main, ["video", "metadata", str(tmp_clean_png), "--check"])

        assert result.exit_code != 0
        assert "Unsupported video format" in result.output


class TestVideoProvenanceCli:
    def test_json_metadata_report(self, tmp_path: Path):
        source = _video_with_c2pa(tmp_path / "source.mp4")

        result = CliRunner().invoke(main, ["video", "identify", str(source), "--no-visible", "--json"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["is_ai_generated"] is True
        assert payload["visible_mark"] is None
        assert payload["has_ai_metadata"] is True


class TestVideoAllApi:
    def test_no_visible_mark_still_strips_metadata_and_writes_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video
        from remove_ai_watermarks.video import VideoVisibleResult, remove_video_all

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"
        monkeypatch.setattr(
            video,
            "remove_video_visible",
            lambda *_args, **_kwargs: VideoVisibleResult(
                source=source,
                output=None,
                mark="auto",
                total_frames=3,
                detected_frames=0,
                removed_frames=0,
                remaining_metadata={"c2pa_manifest": "present"},
            ),
        )

        result = remove_video_all(source, output)

        assert result.output == output
        assert result.visible_mark is None
        assert result.detected_metadata
        assert result.remaining_metadata == {}
        assert result.invisible_removed is False
        assert result.visual_invisible_action == "not_run"
        assert result.audio.stream_action == "copied_if_present"
        assert result.audio.watermark_status == "unverified"
        assert C2PA_UUID not in output.read_bytes()
        assert _VIDEO_PAYLOAD in output.read_bytes()

    def test_visible_output_is_the_final_locally_verified_result(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video
        from remove_ai_watermarks.video import VideoVisibleResult, remove_video_all

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"

        def fake_visible(_source: Path, target: Path, **_kwargs: object) -> VideoVisibleResult:
            target.write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))
            return VideoVisibleResult(
                source=source,
                output=target,
                mark="sora",
                total_frames=12,
                detected_frames=12,
                removed_frames=12,
                remaining_metadata={},
            )

        monkeypatch.setattr(video, "remove_video_visible", fake_visible)
        monkeypatch.setattr(
            video,
            "remove_video_metadata",
            lambda *_args, **_kwargs: pytest.fail("metadata must not reprocess an already stripped visible output"),
        )

        result = remove_video_all(source, output)

        assert result.visible_mark == "sora"
        assert result.visible_removed_frames == 12
        assert result.remaining_metadata == {}
        assert output.exists()

    def test_invisible_stage_is_explicit_and_uses_an_intermediate_visible_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video
        from remove_ai_watermarks.video import (
            VideoInvisibleResult,
            VideoVisibleResult,
            remove_video_all,
        )

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"
        intermediate_sources: list[Path] = []

        def fake_visible(_source: Path, target: Path, **_kwargs: object) -> VideoVisibleResult:
            target.write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))
            return VideoVisibleResult(
                source=source,
                output=target,
                mark="sora",
                total_frames=12,
                detected_frames=12,
                removed_frames=12,
                remaining_metadata={},
            )

        def fake_invisible(
            candidate_source: Path,
            target: Path,
            **_kwargs: object,
        ) -> VideoInvisibleResult:
            assert candidate_source != source
            assert candidate_source.exists()
            intermediate_sources.append(candidate_source)
            target.write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))
            return VideoInvisibleResult(
                source=candidate_source,
                output=target,
                noise_std=0.1,
                metrics=_regeneration_metrics(frames=12),
                remaining_metadata={},
            )

        monkeypatch.setattr(video, "remove_video_visible", fake_visible)
        monkeypatch.setattr(video, "remove_video_invisible", fake_invisible)

        result = remove_video_all(source, output, include_invisible=True)

        assert result.invisible_removed is True
        assert result.visual_invisible_action == "regenerated"
        assert result.audio.stream_action == "copied_if_present"
        assert result.audio.watermark_status == "unverified"
        assert output.exists()
        assert len(intermediate_sources) == 1
        assert not intermediate_sources[0].exists()

    def test_rejects_unsupported_invisible_container_before_visible_scan(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video
        from remove_ai_watermarks.video import remove_video_all

        source = _video_with_tc260_ebml(tmp_path / "source.webm")
        monkeypatch.setattr(
            video,
            "remove_video_visible",
            lambda *_args, **_kwargs: pytest.fail("visible scan must not start"),
        )

        with pytest.raises(ValueError, match="requires one of"):
            remove_video_all(source, include_invisible=True)


class TestVideoAllCli:
    def test_help_keeps_invisible_stage_opt_in(self):
        result = CliRunner().invoke(main, ["video", "all", "--help"])

        assert result.exit_code == 0, result.output
        assert "--invisible" in result.output
        assert "calibrated lossy regeneration" in result.output
        assert "video pixels" in result.output

    def test_reports_locally_verified_result(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from remove_ai_watermarks import video
        from remove_ai_watermarks.video import VideoAllResult

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"
        monkeypatch.setattr(
            video,
            "remove_video_all",
            lambda *_args, **_kwargs: VideoAllResult(
                source=source,
                output=output,
                visible_mark="sora",
                total_frames=12,
                visible_detected_frames=12,
                visible_removed_frames=12,
                detected_metadata={"c2pa_manifest": "present"},
                remaining_metadata={},
                invisible_removed=False,
            ),
        )

        result = CliRunner().invoke(main, ["video", "all", str(source), "-o", str(output)])

        assert result.exit_code == 0, result.output
        assert "removed sora from 12/12 frames" in result.output
        assert "Audio watermark: UNVERIFIED" in result.output

    def test_reports_video_regeneration_and_unverified_audio(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video
        from remove_ai_watermarks.video import VideoAllResult

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"
        monkeypatch.setattr(
            video,
            "remove_video_all",
            lambda *_args, **_kwargs: VideoAllResult(
                source=source,
                output=output,
                visible_mark=None,
                total_frames=12,
                visible_detected_frames=0,
                visible_removed_frames=0,
                detected_metadata={},
                remaining_metadata={},
                invisible_removed=True,
            ),
        )

        result = CliRunner().invoke(main, ["video", "all", str(source), "--invisible"])

        assert result.exit_code == 0, result.output
        assert "Video pixels: regenerated with the calibrated SynthID-removal profile" in result.output
        assert "not a per-file verification" in result.output
        assert "Audio watermark: UNVERIFIED" in result.output


class TestVideoBatchApi:
    def test_visible_mode_copies_noop_and_collects_per_file_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video
        from remove_ai_watermarks.video import VideoVisibleResult, remove_video_batch

        directory = tmp_path / "videos"
        directory.mkdir()
        clean = directory / "clean.mp4"
        clean.write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))
        broken = directory / "broken.mp4"
        broken.write_bytes(b"not a video")
        (directory / "notes.txt").write_text("ignored", encoding="utf-8")

        def fake_visible(source: Path, *_args: object, **_kwargs: object) -> VideoVisibleResult:
            if source == broken:
                raise ValueError("invalid container")
            return VideoVisibleResult(
                source=source,
                output=None,
                mark="auto",
                total_frames=3,
                detected_frames=0,
                removed_frames=0,
                remaining_metadata={},
            )

        monkeypatch.setattr(video, "remove_video_visible", fake_visible)

        result = remove_video_batch(directory, mode="visible")

        assert result.processed == 1
        assert result.failed == 1
        assert len(result.items) == 2
        assert (result.output_directory / clean.name).read_bytes() == clean.read_bytes()
        assert result.items[0].source.name == "broken.mp4"
        assert result.items[0].error == "invalid container"
        assert result.items[0].audio.stream_action == "not_written"
        assert result.items[1].changed is False
        assert result.items[1].audio.stream_action == "copied_if_present"

    def test_metadata_mode_processes_every_supported_video(self, tmp_path: Path):
        from remove_ai_watermarks.video import remove_video_batch

        directory = tmp_path / "videos"
        directory.mkdir()
        marked = _video_with_c2pa(directory / "marked.mp4")
        clean = directory / "clean.mp4"
        clean.write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))

        result = remove_video_batch(directory, mode="metadata")

        assert result.processed == 2
        assert result.failed == 0
        assert C2PA_UUID not in (result.output_directory / marked.name).read_bytes()
        assert (result.output_directory / clean.name).read_bytes() == clean.read_bytes()
        assert {item.changed for item in result.items} == {False, True}

    def test_rejects_invisible_stage_outside_all_mode(self, tmp_path: Path):
        from remove_ai_watermarks.video import remove_video_batch

        directory = tmp_path / "videos"
        directory.mkdir()

        with pytest.raises(ValueError, match="only in all mode"):
            remove_video_batch(directory, mode="visible", include_invisible=True)

    def test_invisible_mode_reuses_one_vae_runtime_for_the_batch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video, video_invisible
        from remove_ai_watermarks.video import VideoAllResult, remove_video_batch

        directory = tmp_path / "videos"
        directory.mkdir()
        for name in ("one.mp4", "two.mp4"):
            (directory / name).write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))
        runtime = object()
        load_calls: list[tuple[str, str]] = []
        used_runtimes: list[object] = []

        def fake_load(*, model: str, device: str) -> object:
            load_calls.append((model, device))
            return runtime

        def fake_all(source: Path, output: Path, **kwargs: object) -> VideoAllResult:
            used_runtimes.append(kwargs["_invisible_runtime"])
            return VideoAllResult(
                source=source,
                output=output,
                visible_mark=None,
                total_frames=2,
                visible_detected_frames=0,
                visible_removed_frames=0,
                detected_metadata={},
                remaining_metadata={},
                invisible_removed=True,
            )

        monkeypatch.setattr(video_invisible, "load_video_vae_runtime", fake_load)
        monkeypatch.setattr(video, "remove_video_all", fake_all)

        result = remove_video_batch(directory, include_invisible=True)

        assert result.processed == 2
        assert load_calls == [("stabilityai/sd-vae-ft-mse", "auto")]
        assert used_runtimes == [runtime, runtime]

    def test_invisible_runtime_failure_is_not_retried_for_every_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video_invisible
        from remove_ai_watermarks.video import remove_video_batch

        directory = tmp_path / "videos"
        directory.mkdir()
        for name in ("one.mp4", "two.mp4"):
            (directory / name).write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))
        load_calls = 0

        def fail_load(**_kwargs: object) -> object:
            nonlocal load_calls
            load_calls += 1
            raise RuntimeError("model unavailable")

        monkeypatch.setattr(video_invisible, "load_video_vae_runtime", fail_load)

        result = remove_video_batch(directory, include_invisible=True)

        assert result.failed == 2
        assert load_calls == 1
        assert {item.error for item in result.items} == {"model unavailable"}

    def test_rejects_source_as_output_directory(self, tmp_path: Path):
        from remove_ai_watermarks.video import remove_video_batch

        with pytest.raises(ValueError, match="must differ"):
            remove_video_batch(tmp_path, tmp_path)


class TestVideoBatchCli:
    def test_help(self):
        result = CliRunner().invoke(main, ["video", "batch", "--help"])

        assert result.exit_code == 0, result.output
        assert "all|visible|metadata" in result.output
        assert "--invisible" in result.output

    def test_returns_nonzero_when_any_file_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video
        from remove_ai_watermarks.video import VideoBatchItem, VideoBatchResult

        directory = tmp_path / "videos"
        directory.mkdir()
        output = tmp_path / "clean"
        failed_source = directory / "broken.mp4"
        monkeypatch.setattr(
            video,
            "remove_video_batch",
            lambda *_args, **_kwargs: VideoBatchResult(
                directory=directory,
                output_directory=output,
                items=(
                    VideoBatchItem(
                        source=failed_source,
                        output=None,
                        mode="all",
                        changed=False,
                        visible_mark=None,
                        invisible_removed=False,
                        error="invalid container",
                    ),
                ),
            ),
        )

        result = CliRunner().invoke(main, ["video", "batch", str(directory)])

        assert result.exit_code == 1
        assert "FAILED broken.mp4: invalid container" in result.output
        assert "1 failed" in result.output

    def test_reports_visual_regeneration_without_claiming_audio_removal(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video
        from remove_ai_watermarks.video import VideoBatchItem, VideoBatchResult

        directory = tmp_path / "videos"
        directory.mkdir()
        output = tmp_path / "clean"
        source = directory / "source.mp4"
        monkeypatch.setattr(
            video,
            "remove_video_batch",
            lambda *_args, **_kwargs: VideoBatchResult(
                directory=directory,
                output_directory=output,
                items=(
                    VideoBatchItem(
                        source=source,
                        output=output / source.name,
                        mode="all",
                        changed=True,
                        visible_mark=None,
                        invisible_removed=True,
                    ),
                ),
            ),
        )

        result = CliRunner().invoke(main, ["video", "batch", str(directory), "--invisible"])

        assert result.exit_code == 0, result.output
        assert "Video pixel regeneration: applied to 1 file(s)" in result.output
        assert "SynthID: removed" not in result.output
        assert "Audio watermark: UNVERIFIED" in result.output


class TestVideoInvisibleApi:
    def test_removes_synthid_and_strips_metadata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video_invisible
        from remove_ai_watermarks.video import remove_video_invisible

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"

        def fake_regenerate(_source: Path, target: Path, **_kwargs: object):
            target.write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))
            return _regeneration_metrics()

        monkeypatch.setattr(video_invisible, "regenerate_video_candidate", fake_regenerate)

        result = remove_video_invisible(source, output)

        assert result.output == output
        assert result.total_frames == 24
        assert result.remaining_metadata == {}
        assert result.visual_invisible_action == "regenerated"
        assert result.audio.stream_action == "copied_if_present"
        assert result.audio.watermark_status == "unverified"

    def test_default_output_is_named_as_clean(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video_invisible
        from remove_ai_watermarks.video import remove_video_invisible

        source = _video_with_c2pa(tmp_path / "source.mp4")

        def fake_regenerate(_source: Path, target: Path, **_kwargs: object):
            target.write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))
            return _regeneration_metrics(
                frames=2,
                fps=2.0,
                width=16,
                height=16,
                psnr_db=20.0,
                temporal_residual_ratio=1.0,
            )

        monkeypatch.setattr(video_invisible, "regenerate_video_candidate", fake_regenerate)

        result = remove_video_invisible(source)

        assert result.output == tmp_path / "source_clean.mp4"

    def test_rejects_webm_regeneration(self, tmp_path: Path):
        from remove_ai_watermarks.video import remove_video_invisible

        source = _video_with_tc260_ebml(tmp_path / "source.webm")

        with pytest.raises(ValueError, match="requires one of"):
            remove_video_invisible(source)


class TestVideoInvisibleCli:
    def test_help_describes_calibrated_video_pixel_action(self):
        runner = CliRunner()

        result = runner.invoke(main, ["video", "invisible", "--help"])

        assert result.exit_code == 0, result.output
        assert "calibrated SynthID-removal profile to video pixels" in result.output

    def test_reports_completed_removal(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video

        runner = CliRunner()
        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"

        def fake_remove(_source: Path, target: Path, **_kwargs: object):
            target.write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))
            return video.VideoInvisibleResult(
                source=_source,
                output=target,
                noise_std=0.1,
                metrics=_regeneration_metrics(),
                remaining_metadata={},
            )

        monkeypatch.setattr(video, "remove_video_invisible", fake_remove)

        result = runner.invoke(main, ["video", "invisible", str(source), "-o", str(output)])

        assert result.exit_code == 0, result.output
        assert "SynthID removal complete" not in result.output
        assert "Video pixels: regenerated with the calibrated SynthID-removal profile" in result.output
        assert "not a per-file verification" in result.output
        assert "Audio watermark: UNVERIFIED" in result.output


class TestSoraFrameLocalization:
    def test_localizes_independently_rendered_sora_like_mark(self):
        from remove_ai_watermarks.video_visible import _region_iou, detect_sora_frame

        frame, expected = _independent_sora_frame()

        detection = detect_sora_frame(frame)

        assert detection.region is not None
        assert detection.confidence >= 0.58
        assert _region_iou(detection.region, expected) >= 0.45

    def test_empty_frame_is_not_localized(self):
        from remove_ai_watermarks.video_visible import detect_sora_frame

        detection = detect_sora_frame(np.empty((0, 0, 3), dtype=np.uint8))

        assert detection.confidence == 0.0
        assert detection.region is None


class TestVeoFrameLocalization:
    def test_localizes_independently_rendered_diamond_at_relocated_position(self):
        from remove_ai_watermarks.video_visible import _region_iou, detect_veo_frame

        frame = np.full((720, 1280, 3), 28, dtype=np.uint8)
        size = 48
        x, y = 1080, 570
        mark = Image.new("L", (size, size), 0)
        points = (
            (size // 2, 1),
            (round(size * 0.61), round(size * 0.38)),
            (size - 2, size // 2),
            (round(size * 0.61), round(size * 0.62)),
            (size // 2, size - 2),
            (round(size * 0.39), round(size * 0.62)),
            (1, size // 2),
            (round(size * 0.39), round(size * 0.38)),
        )
        ImageDraw.Draw(mark).polygon(points, fill=255)
        _stamp_gray_mark(frame, mark, x=x, y=y, opacity=0.72)

        detection = detect_veo_frame(frame)

        assert detection.region is not None
        assert detection.confidence >= 0.70
        assert _region_iou(detection.region, (x, y, size, size)) >= 0.70

    def test_localizes_independently_rendered_legacy_text(self):
        from remove_ai_watermarks.video_visible import _region_iou, detect_veo_frame

        frame = np.full((720, 1280, 3), 42, dtype=np.uint8)
        mark = Image.new("L", (60, 24), 0)
        try:
            font = ImageFont.load_default(size=19)
        except TypeError:
            font = ImageFont.load_default()
        ImageDraw.Draw(mark).text((1, 0), "Veo", font=font, fill=255)
        mark_array = np.asarray(mark)
        ys, xs = np.where(mark_array > 0)
        mark_array = mark_array[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        mark_height, mark_width = mark_array.shape
        x = frame.shape[1] - mark_width - 20
        y = frame.shape[0] - mark_height - 18
        _stamp_gray_mark(frame, mark_array, x=x, y=y, opacity=0.66)

        detection = detect_veo_frame(frame)

        assert detection.region is not None
        assert detection.confidence >= 0.55
        assert _region_iou(detection.region, (x, y, mark_width, mark_height)) >= 0.65

    def test_empty_frame_is_not_localized(self):
        from remove_ai_watermarks.video_visible import detect_veo_frame

        detection = detect_veo_frame(np.empty((0, 0, 3), dtype=np.uint8))

        assert detection.confidence == 0.0
        assert detection.region is None

    def test_diamond_mask_preserves_transparent_box_corners(self):
        from remove_ai_watermarks.video_visible import _mask_for_region

        mask = _mask_for_region(
            np.zeros((100, 100, 3), dtype=np.uint8),
            (20, 20, 48, 48),
            padding_fraction=0.18,
            mask_style="veo",
        )

        assert mask[44, 44] == 255
        assert mask[20, 20] == 0
        assert mask[67, 67] == 0


class TestByteDanceFrameLocalization:
    def test_localizes_independently_rendered_seedance_box(self):
        from remove_ai_watermarks.video_visible import _region_iou, detect_seedance_frame

        frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
        mark = Image.new("L", (80, 60), 0)
        draw = ImageDraw.Draw(mark)
        draw.rounded_rectangle((2, 2, 70, 53), radius=14, outline=255, width=4)
        try:
            font = ImageFont.load_default(size=35)
        except TypeError:
            font = ImageFont.load_default()
        draw.text((18, 8), "AI", font=font, fill=255)
        x, y = 1130, 620
        _stamp_gray_mark(frame, mark, x=x, y=y, opacity=0.65)

        detection = detect_seedance_frame(frame)

        assert detection.region is not None
        assert detection.confidence >= 0.43
        assert _region_iou(detection.region, (x, y, 80, 60)) >= 0.75

    def test_localizes_independently_rendered_dola_text(self):
        from remove_ai_watermarks.video_visible import _region_iou, detect_dola_frame

        frame = np.full((720, 1280, 3), 35, dtype=np.uint8)
        mark = np.zeros((40, 150), dtype=np.uint8)
        cv2.putText(
            mark,
            "Dola AI",
            (2, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            255,
            2,
            cv2.LINE_AA,
        )
        ys, xs = np.where(mark > 0)
        mark = mark[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        mark_height, mark_width = mark.shape
        x = frame.shape[1] - mark_width - 18
        y = frame.shape[0] - mark_height - 14
        _stamp_gray_mark(frame, mark, x=x, y=y, opacity=0.75)

        detection = detect_dola_frame(frame)

        assert detection.region is not None
        assert detection.confidence >= 0.52
        assert _region_iou(detection.region, (x, y, mark_width, mark_height)) >= 0.75

    def test_seedance_box_mask_covers_the_full_localized_mark(self):
        from remove_ai_watermarks.video_visible import _mask_for_region

        mask = _mask_for_region(
            np.zeros((120, 160, 3), dtype=np.uint8),
            (20, 20, 80, 60),
            padding_fraction=0.0,
            mask_style="box",
        )

        assert mask[15, 15] == 0
        assert mask[16, 16] == 255
        assert mask[83, 103] == 255
        assert mask[84, 104] == 0


class TestAdditionalProviderFrameLocalization:
    def test_localizes_independently_rendered_hailuo_label(self):
        from remove_ai_watermarks.video_visible import _region_iou, detect_hailuo_frame

        frame = np.full((720, 1280, 3), 32, dtype=np.uint8)
        mark = Image.new("L", (330, 54), 0)
        draw = ImageDraw.Draw(mark)
        try:
            font = ImageFont.load_default(size=28)
        except TypeError:
            font = ImageFont.load_default()
        for index, height in enumerate((20, 34, 46, 34, 20)):
            x = 4 + index * 6
            draw.rounded_rectangle((x, 27 - height // 2, x + 2, 27 + height // 2), radius=1, fill=255)
        draw.text((39, 8), "MINIMAX", font=font, fill=255)
        draw.rectangle((164, 7, 166, 47), fill=255)
        draw.ellipse((178, 8, 224, 50), outline=255, width=5)
        draw.text((228, 8), "hailuo AI", font=font, fill=255)
        x, y = 930, 650
        _stamp_gray_mark(frame, mark, x=x, y=y, opacity=0.75)

        detection = detect_hailuo_frame(frame)

        assert detection.region is not None
        assert detection.confidence >= 0.24
        assert _region_iou(detection.region, (x, y, 330, 54)) >= 0.45

    def test_localizes_kling_core_and_covers_version_suffix(self):
        from remove_ai_watermarks.video_visible import detect_kling_frame

        frame = np.full((720, 1280, 3), 28, dtype=np.uint8)
        mark = np.zeros((42, 245), dtype=np.uint8)
        cv2.ellipse(mark, (20, 21), (15, 15), 0, 20, 330, 255, 4, cv2.LINE_AA)
        cv2.putText(
            mark,
            "KLING AI 1.6",
            (43, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            255,
            2,
            cv2.LINE_AA,
        )
        x, y = 1018, 664
        _stamp_gray_mark(frame, mark, x=x, y=y, opacity=0.72)

        detection = detect_kling_frame(frame)
        glyph_ys, glyph_xs = np.where(mark > 0)
        glyph_box = (
            x + int(glyph_xs.min()),
            y + int(glyph_ys.min()),
            int(glyph_xs.max() - glyph_xs.min() + 1),
            int(glyph_ys.max() - glyph_ys.min() + 1),
        )

        assert detection.region is not None
        assert detection.confidence >= 0.24
        detected_x, detected_y, detected_width, detected_height = detection.region
        glyph_x, glyph_y, glyph_width, glyph_height = glyph_box
        assert detected_x <= glyph_x
        assert detected_y <= glyph_y
        assert detected_x + detected_width >= glyph_x + glyph_width
        assert detected_y + detected_height >= glyph_y + glyph_height

    def test_rejects_a_saturated_fixed_kling_shape(self):
        from remove_ai_watermarks.video_visible import detect_kling_frame

        frame = np.full((720, 1280, 3), 24, dtype=np.uint8)
        cv2.circle(frame, (1040, 670), 15, (0, 220, 0), 5, cv2.LINE_AA)
        cv2.putText(
            frame,
            "KLING AI 1.6",
            (1065, 681),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

        detection = detect_kling_frame(frame)

        assert detection.confidence == 0.0
        assert detection.region is None

    def test_ignores_a_logo_candidate_left_of_the_wordmark(self, monkeypatch: pytest.MonkeyPatch):
        from remove_ai_watermarks import video_visible
        from remove_ai_watermarks.video_visible import FrameLocalization, detect_kling_frame

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[670:706, 1120:1280] = 255

        def fixed_mark(_frame, template_key, *, frame_index, **_kwargs):
            if template_key == "kling-3":
                return FrameLocalization(frame_index, 0.63, (1156, 678, 60, 20))
            if template_key == "kling-logo":
                return FrameLocalization(frame_index, 0.55, (900, 680, 20, 20))
            return FrameLocalization(frame_index, 0.0, None)

        monkeypatch.setattr(video_visible, "_detect_fixed_mark", fixed_mark)

        detection = detect_kling_frame(frame)

        assert detection.region == (1120, 670, 160, 36)
        assert detection.confidence == 0.63


class TestSoraTemporalArbiter:
    _BOX = (40, 60, 150, 54)

    def test_four_frame_lookalike_run_is_too_short(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        detections = [FrameLocalization(index, 0.70, self._BOX) for index in range(4)]

        assert stabilize_localizations("sora", detections, provenance=False) == [None] * 4

    def test_provenance_accepts_recurring_low_contrast_visual_match(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        detections = [
            FrameLocalization(0, 0.59, self._BOX),
            FrameLocalization(1, 0.61, self._BOX),
            FrameLocalization(2, 0.62, self._BOX),
            FrameLocalization(3, 0.60, self._BOX),
            FrameLocalization(4, 0.61, self._BOX),
        ]

        assert stabilize_localizations("sora", detections, provenance=True) == [self._BOX] * 5

    def test_confirmed_provenance_run_covers_transition_frames(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        detections = [
            FrameLocalization(0, 0.30, (500, 300, 54, 54)),
            FrameLocalization(1, 0.59, self._BOX),
            FrameLocalization(2, 0.61, self._BOX),
            FrameLocalization(3, 0.62, self._BOX),
            FrameLocalization(4, 0.60, self._BOX),
            FrameLocalization(5, 0.61, self._BOX),
            FrameLocalization(6, 0.30, (300, 100, 54, 54)),
        ]

        assert stabilize_localizations("sora", detections, provenance=True) == [self._BOX] * 7

    def test_transition_prefers_low_score_match_at_a_confirmed_position(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        other_box = (500, 300, 150, 54)
        detections = [
            FrameLocalization(0, 0.61, self._BOX),
            FrameLocalization(1, 0.62, self._BOX),
            FrameLocalization(2, 0.63, self._BOX),
            FrameLocalization(3, 0.61, self._BOX),
            FrameLocalization(4, 0.62, self._BOX),
            FrameLocalization(5, 0.20, (250, 180, 54, 54)),
            FrameLocalization(6, 0.52, self._BOX),
            FrameLocalization(7, 0.61, other_box),
            FrameLocalization(8, 0.62, other_box),
            FrameLocalization(9, 0.63, other_box),
            FrameLocalization(10, 0.61, other_box),
            FrameLocalization(11, 0.62, other_box),
        ]

        stabilized = stabilize_localizations("sora", detections, provenance=True)

        assert stabilized[6] == self._BOX
        assert stabilized[7:] == [other_box] * 5

    def test_transition_without_a_match_keeps_previous_stable_position(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        other_box = (500, 300, 150, 54)
        detections = [
            FrameLocalization(0, 0.61, self._BOX),
            FrameLocalization(1, 0.62, self._BOX),
            FrameLocalization(2, 0.63, self._BOX),
            FrameLocalization(3, 0.61, self._BOX),
            FrameLocalization(4, 0.62, self._BOX),
            FrameLocalization(5, 0.20, (250, 180, 54, 54)),
            FrameLocalization(6, 0.20, (300, 200, 54, 54)),
            FrameLocalization(7, 0.61, other_box),
            FrameLocalization(8, 0.62, other_box),
            FrameLocalization(9, 0.63, other_box),
            FrameLocalization(10, 0.61, other_box),
            FrameLocalization(11, 0.62, other_box),
        ]

        stabilized = stabilize_localizations("sora", detections, provenance=True)

        assert stabilized[5:7] == [self._BOX, self._BOX]

    def test_unproven_weak_run_is_rejected(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        detections = [
            FrameLocalization(0, 0.61, self._BOX),
            FrameLocalization(1, 0.62, self._BOX),
            FrameLocalization(2, 0.63, self._BOX),
            FrameLocalization(3, 0.62, self._BOX),
            FrameLocalization(4, 0.61, self._BOX),
        ]

        assert stabilize_localizations("sora", detections, provenance=False) == [None] * 5

    def test_strong_recurring_visual_run_needs_no_metadata(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        detections = [
            FrameLocalization(0, 0.61, self._BOX),
            FrameLocalization(1, 0.66, self._BOX),
            FrameLocalization(2, 0.62, self._BOX),
            FrameLocalization(3, 0.61, self._BOX),
            FrameLocalization(4, 0.62, self._BOX),
        ]

        assert stabilize_localizations("sora", detections, provenance=False) == [self._BOX] * 5

    def test_isolated_lookalikes_at_different_positions_are_rejected(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        detections = [
            FrameLocalization(0, 0.70, (10, 10, 150, 54)),
            FrameLocalization(1, 0.70, (400, 200, 150, 54)),
            FrameLocalization(2, 0.70, (650, 400, 150, 54)),
        ]

        assert stabilize_localizations("sora", detections, provenance=True) == [None, None, None]

    def test_short_dropout_between_matching_boxes_is_filled(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        detections = [
            FrameLocalization(0, 0.66, self._BOX),
            FrameLocalization(1, 0.20, (500, 300, 54, 54)),
            FrameLocalization(2, 0.67, self._BOX),
            FrameLocalization(3, 0.66, self._BOX),
            FrameLocalization(4, 0.66, self._BOX),
            FrameLocalization(5, 0.66, self._BOX),
        ]

        assert stabilize_localizations("sora", detections, provenance=False) == [self._BOX] * 6


class TestVeoTemporalArbiter:
    _BOX = (1132, 572, 56, 56)

    def test_eleven_frame_lookalike_run_is_too_short(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        detections = [FrameLocalization(index, 0.70, self._BOX) for index in range(11)]

        assert stabilize_localizations("veo", detections, provenance=False) == [None] * 11

    def test_strong_fixed_run_covers_video_without_metadata(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        detections = [FrameLocalization(index, 0.60, self._BOX) for index in range(12)]
        detections.extend(FrameLocalization(index, 0.20, (300, 200, 48, 48)) for index in range(12, 15))

        assert stabilize_localizations("veo", detections, provenance=False) == [self._BOX] * 15

    def test_google_provenance_accepts_recurring_low_contrast_diamond(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        detections = [FrameLocalization(index, 0.47, self._BOX) for index in range(12)]

        assert stabilize_localizations("veo", detections, provenance=True) == [self._BOX] * 12

    def test_unproven_weak_run_is_rejected(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        detections = [FrameLocalization(index, 0.52, self._BOX) for index in range(12)]

        assert stabilize_localizations("veo", detections, provenance=False) == [None] * 12


class TestByteDanceTemporalArbiter:
    _SEEDANCE_BOX = (1110, 610, 90, 66)
    _DOLA_BOX = (1160, 680, 96, 22)

    def test_seedance_requires_twelve_recurring_frames(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        detections = [FrameLocalization(index, 0.50, self._SEEDANCE_BOX) for index in range(11)]

        assert stabilize_localizations("seedance", detections, provenance=False) == [None] * 11

    def test_seedance_strong_run_covers_low_contrast_frames(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        detections = [FrameLocalization(index, 0.45, self._SEEDANCE_BOX) for index in range(12)]
        detections.extend(FrameLocalization(index, 0.20, (200, 100, 80, 60)) for index in range(12, 15))

        assert stabilize_localizations("seedance", detections, provenance=False) == [self._SEEDANCE_BOX] * 15

    def test_seedance_rejects_a_slowly_drifting_scene_detail(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        detections = [FrameLocalization(index, 0.46, (1110 - index * 3, 610, 90, 66)) for index in range(14)]

        assert stabilize_localizations("seedance", detections, provenance=False) == [None] * 14

    def test_dola_requires_twelve_recurring_frames(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        detections = [FrameLocalization(index, 0.60, self._DOLA_BOX) for index in range(11)]

        assert stabilize_localizations("dola", detections, provenance=True) == [None] * 11

    def test_dola_provenance_accepts_recurring_low_contrast_text(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        detections = [FrameLocalization(index, 0.49, self._DOLA_BOX) for index in range(12)]

        assert stabilize_localizations("dola", detections, provenance=True) == [self._DOLA_BOX] * 12

    def test_dola_without_provenance_needs_a_strong_frame(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        detections = [FrameLocalization(index, 0.51, self._DOLA_BOX) for index in range(12)]

        assert stabilize_localizations("dola", detections, provenance=False) == [None] * 12

    def test_bytedance_provenance_requires_ai_source_type(self):
        from remove_ai_watermarks.video_visible import has_bytedance_video_provenance

        assert has_bytedance_video_provenance(
            {
                "issuer": "BytePlus (ByteDance)",
                "source_type": "trainedAlgorithmicMedia (AI-generated)",
            }
        )
        assert not has_bytedance_video_provenance({"issuer": "BytePlus (ByteDance)"})


class TestAdditionalProviderTemporalArbiter:
    _HAILUO_BOX = (930, 650, 330, 54)
    _KLING_BOX = (1018, 664, 245, 42)

    @pytest.mark.parametrize(
        ("mark", "box", "weak_score", "strong_score"),
        [
            ("hailuo", _HAILUO_BOX, 0.31, 0.35),
            ("kling", _KLING_BOX, 0.21, 0.25),
        ],
    )
    def test_requires_a_strong_anchored_twelve_frame_run(
        self,
        mark: str,
        box: tuple[int, int, int, int],
        weak_score: float,
        strong_score: float,
    ):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        def stabilize(dets):
            return stabilize_localizations(mark, dets)

        weak = [FrameLocalization(index, weak_score, box) for index in range(12)]
        strong = [FrameLocalization(index, strong_score, box) for index in range(12)]

        assert stabilize(weak) == [None] * 12
        assert stabilize(strong) == [box] * 12

    def test_hailuo_provenance_accepts_sub_strong_stable_run(self):
        # A TC260 label naming MiniMax relaxes only the strong-frame requirement:
        # the entry bar stays the measured weak floor (0.30), so a stable run
        # between weak and strong (0.34) passes with provenance, fails without.
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_localizations

        detections = [FrameLocalization(index, 0.31, self._HAILUO_BOX) for index in range(12)]

        assert stabilize_localizations("hailuo", detections, provenance=False) == [None] * 12
        assert stabilize_localizations("hailuo", detections, provenance=True) == [self._HAILUO_BOX] * 12

    def test_hailuo_provenance_requires_a_minimax_producer_label(self):
        from remove_ai_watermarks.video_visible import has_hailuo_video_provenance

        assert has_hailuo_video_provenance({"aigc_producer": "MiniMax"})
        assert not has_hailuo_video_provenance({"aigc_producer": "001191110102MACQD9K64010000"})
        assert not has_hailuo_video_provenance({"aigc_label": "China AIGC label (TC260); producer MiniMax"})
        assert not has_hailuo_video_provenance({"issuer": "Some other vendor"})


class TestVideoVisibleScan:
    def test_auto_prepares_each_frame_once_for_every_detector(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video_visible
        from remove_ai_watermarks.video_visible import FrameLocalization, scan_video_marks

        frame = np.zeros((8, 12, 3), dtype=np.uint8)

        class FakeCapture:
            def __init__(self) -> None:
                self._read = False

            def isOpened(self) -> bool:
                return True

            def get(self, property_id: int) -> float:
                return {
                    video_visible.cv2.CAP_PROP_FRAME_WIDTH: 12.0,
                    video_visible.cv2.CAP_PROP_FRAME_HEIGHT: 8.0,
                    video_visible.cv2.CAP_PROP_FPS: 24.0,
                    video_visible.cv2.CAP_PROP_POS_MSEC: 0.0,
                }[property_id]

            def read(self) -> tuple[bool, np.ndarray | None]:
                if self._read:
                    return False, None
                self._read = True
                return True, frame

            def release(self) -> None:
                pass

        prepared_ids: list[int] = []

        def fake_detector(
            _frame: np.ndarray,
            *,
            frame_index: int,
            prepared: object,
        ) -> FrameLocalization:
            assert prepared is not None
            prepared_ids.append(id(prepared))
            return FrameLocalization(frame_index, 0.0, None)

        monkeypatch.setattr(video_visible.cv2, "VideoCapture", lambda _path: FakeCapture())
        monkeypatch.setattr(video_visible, "probe_video_timestamps", lambda _path: (0.25,))
        for detector_name in (
            "detect_sora_frame",
            "detect_veo_frame",
            "detect_seedance_frame",
            "detect_dola_frame",
            "detect_hailuo_frame",
            "detect_kling_frame",
        ):
            monkeypatch.setattr(video_visible, detector_name, fake_detector)

        scans = scan_video_marks(
            tmp_path / "synthetic.mp4",
            ("sora", "veo", "seedance", "doubao", "dola", "hailuo", "kling"),
        )

        assert set(scans) == {"sora", "veo", "seedance", "doubao", "dola", "hailuo", "kling"}
        assert all(scan.timestamps == (0.25,) for scan in scans.values())
        assert len(prepared_ids) == 6
        assert len(set(prepared_ids)) == 1


class TestVideoVisibleEncoding:
    @staticmethod
    def _patch_single_frame_encode(
        monkeypatch: pytest.MonkeyPatch,
        *,
        encoded_bytes: bytes,
        fail: bool,
    ) -> tuple[object, list[Path]]:
        from remove_ai_watermarks import video_visible, watermark_registry

        frame = np.full((8, 8, 3), 32, dtype=np.uint8)

        class FakeCapture:
            def __init__(self) -> None:
                self._read = False

            def isOpened(self) -> bool:
                return True

            def read(self) -> tuple[bool, np.ndarray | None]:
                if self._read:
                    return False, None
                self._read = True
                return True, frame.copy()

            def release(self) -> None:
                pass

        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = io.BytesIO()

            def poll(self) -> int:
                return 1

            def wait(self) -> int:
                return 1

            def discard_stderr(self) -> None:
                pass

        targets: list[Path] = []

        def fake_command(target: Path, **_kwargs: object) -> list[str]:
            targets.append(target)
            return ["ffmpeg", str(target)]

        def fake_finish(_process: object, target: Path, **_kwargs: object) -> None:
            target.write_bytes(encoded_bytes)
            if fail:
                raise RuntimeError("synthetic encode failure")

        def fake_mux(encoded: Path, _source: Path, target: Path, **_kwargs: object) -> None:
            target.write_bytes(encoded.read_bytes())

        process = FakeProcess()
        monkeypatch.setattr(video_visible.cv2, "VideoCapture", lambda _path: FakeCapture())
        monkeypatch.setattr(video_visible, "raw_video_command", fake_command)
        monkeypatch.setattr(video_visible, "start_raw_video_encoder", lambda _command: process)
        monkeypatch.setattr(video_visible, "finish_raw_video_encoder", fake_finish)
        monkeypatch.setattr(video_visible, "mux_encoded_video", fake_mux)
        monkeypatch.setattr(watermark_registry, "resolve_backend", lambda _backend: "cv2")
        return process, targets

    def test_publishes_completed_encode_atomically(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks.video_visible import FrameLocalization, VideoScan, encode_clean_video

        source = tmp_path / "source.mp4"
        source.write_bytes(b"source")
        output = tmp_path / "clean.mp4"
        scan = VideoScan(8, 8, 24.0, (FrameLocalization(0, 0.0, None),))
        _process, targets = self._patch_single_frame_encode(
            monkeypatch,
            encoded_bytes=b"complete",
            fail=False,
        )

        encode_clean_video(
            source,
            output,
            scan,
            [None],
            backend="cv2",
            strip_metadata=True,
        )

        assert output.read_bytes() == b"complete"
        assert targets[0] != output
        assert targets[0].suffix == output.suffix
        assert not targets[0].exists()

    def test_failed_encode_preserves_existing_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks.video_visible import FrameLocalization, VideoScan, encode_clean_video

        source = tmp_path / "source.mp4"
        source.write_bytes(b"source")
        output = tmp_path / "clean.mp4"
        output.write_bytes(b"previous")
        scan = VideoScan(8, 8, 24.0, (FrameLocalization(0, 0.0, None),))
        _process, targets = self._patch_single_frame_encode(
            monkeypatch,
            encoded_bytes=b"partial",
            fail=True,
        )

        with pytest.raises(RuntimeError, match="synthetic encode failure"):
            encode_clean_video(
                source,
                output,
                scan,
                [None],
                backend="cv2",
                strip_metadata=True,
            )

        assert output.read_bytes() == b"previous"
        assert not targets[0].exists()

    def test_failed_mux_preserves_existing_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video_visible
        from remove_ai_watermarks.video_visible import FrameLocalization, VideoScan, encode_clean_video

        source = tmp_path / "source.mp4"
        source.write_bytes(b"source")
        output = tmp_path / "clean.mp4"
        output.write_bytes(b"previous")
        scan = VideoScan(8, 8, 24.0, (FrameLocalization(0, 0.0, None),))
        _process, targets = self._patch_single_frame_encode(
            monkeypatch,
            encoded_bytes=b"complete video stream",
            fail=False,
        )

        def fail_mux(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("synthetic mux failure")

        monkeypatch.setattr(video_visible, "mux_encoded_video", fail_mux)

        with pytest.raises(RuntimeError, match="synthetic mux failure"):
            encode_clean_video(
                source,
                output,
                scan,
                [None],
                backend="cv2",
                strip_metadata=True,
            )

        assert output.read_bytes() == b"previous"
        assert not targets[0].exists()

    def test_rejects_high_bit_depth_before_silent_downconversion(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video_visible
        from remove_ai_watermarks.video_encoding import VideoEncodeProfile
        from remove_ai_watermarks.video_visible import FrameLocalization, VideoScan, encode_clean_video

        source = tmp_path / "source.mp4"
        source.write_bytes(b"source")
        output = tmp_path / "clean.mp4"
        output.write_bytes(b"previous")
        scan = VideoScan(
            8,
            8,
            24.0,
            (FrameLocalization(0, 1.0, (1, 1, 2, 2)),),
        )
        monkeypatch.setattr(
            video_visible,
            "probe_video_encode_profile",
            lambda _path: VideoEncodeProfile(
                pixel_format="yuv420p",
                color_transfer="smpte2084",
                source_pixel_format="yuv420p10le",
                component_depth=10,
            ),
        )
        monkeypatch.setattr(
            video_visible,
            "start_raw_video_encoder",
            lambda _command: pytest.fail("encoder must not start for a high-bit-depth source"),
        )

        with pytest.raises(RuntimeError, match="refusing to silently reduce yuv420p10le"):
            encode_clean_video(
                source,
                output,
                scan,
                [(1, 1, 2, 2)],
                backend="cv2",
                strip_metadata=True,
            )

        assert output.read_bytes() == b"previous"
        assert list(tmp_path.glob(".clean-*")) == []


class TestVideoVisibleFullClip:
    def test_removes_complete_clip_and_preserves_sequence_and_audio(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import get_ai_metadata
        from remove_ai_watermarks.video import remove_video_all, remove_video_metadata, remove_video_visible
        from remove_ai_watermarks.video_visible import scan_video_marks, stabilize_localizations

        ffmpeg, ffprobe = _ffmpeg_test_tools()
        frame_count = 18
        fps = 12.0
        clean_frames: list[np.ndarray] = []
        source_frames: list[np.ndarray] = []
        mark, mark_region = _independent_sora_mark()
        mark_x, mark_y, _mark_width, _mark_height = mark_region
        for frame_index in range(frame_count):
            clean_frame = _moving_video_background(frame_index)
            marked_frame = clean_frame.copy()
            _stamp_gray_mark(marked_frame, mark, x=mark_x, y=mark_y, opacity=0.78)
            clean_frames.append(clean_frame)
            source_frames.append(marked_frame)

        source = tmp_path / "marked.mp4"
        clean_control = tmp_path / "control.mp4"
        metadata_clean = tmp_path / "metadata-clean.mp4"
        output = tmp_path / "clean.mp4"
        all_output = tmp_path / "clean-all.mp4"
        independent_output = tmp_path / "clean-independent.mp4"
        _write_synthetic_sora_clip(source, source_frames, fps=fps, ffmpeg=ffmpeg)
        _write_synthetic_sora_clip(clean_control, clean_frames, fps=fps, ffmpeg=ffmpeg)
        assert get_ai_metadata(source)

        metadata_result = remove_video_metadata(source, metadata_clean)
        result = remove_video_visible(source, output, mark="sora", backend="cv2")
        all_result = remove_video_all(source, all_output, mark="sora", backend="cv2")
        independent_result = remove_video_visible(
            source,
            independent_output,
            mark="sora",
            backend="cv2",
            temporal_consistency=False,
        )

        assert metadata_result.remaining == {}
        assert metadata_clean.stat().st_size == source.stat().st_size
        assert get_ai_metadata(metadata_clean) == {}
        assert result.output == output
        assert result.total_frames == frame_count
        assert result.detected_frames == frame_count
        assert result.removed_frames == frame_count
        assert result.remaining_metadata == {}
        assert get_ai_metadata(output) == {}
        assert all_result.output == all_output
        assert all_result.visible_mark == "sora"
        assert all_result.visible_removed_frames == frame_count
        assert all_result.detected_metadata
        assert all_result.remaining_metadata == {}
        assert all_result.invisible_removed is False
        assert get_ai_metadata(all_output) == {}
        assert independent_result.removed_frames == frame_count

        decoded_source, source_fps = _decode_video(source)
        decoded_control, control_fps = _decode_video(clean_control)
        decoded_metadata_clean, metadata_clean_fps = _decode_video(metadata_clean)
        decoded_output, output_fps = _decode_video(output)
        decoded_all, all_fps = _decode_video(all_output)
        decoded_independent, independent_fps = _decode_video(independent_output)
        assert (
            len(decoded_source)
            == len(decoded_control)
            == len(decoded_metadata_clean)
            == len(decoded_output)
            == len(decoded_all)
            == len(decoded_independent)
            == frame_count
        )
        assert source_fps == pytest.approx(fps, abs=0.01)
        assert control_fps == pytest.approx(source_fps, abs=0.01)
        assert metadata_clean_fps == pytest.approx(source_fps, abs=0.01)
        assert output_fps == pytest.approx(source_fps, abs=0.01)
        assert all_fps == pytest.approx(source_fps, abs=0.01)
        assert independent_fps == pytest.approx(source_fps, abs=0.01)
        assert all(
            np.array_equal(original, metadata_cleaned)
            for original, metadata_cleaned in zip(decoded_source, decoded_metadata_clean, strict=True)
        )
        assert _container_duration(output, ffprobe=ffprobe) == pytest.approx(
            _container_duration(source, ffprobe=ffprobe),
            abs=1 / fps,
        )
        source_stream = _video_stream_info(source, ffprobe=ffprobe)
        output_stream = _video_stream_info(output, ffprobe=ffprobe)
        expected_stream_properties = {
            "pix_fmt": "yuv420p",
            "color_range": "tv",
            "color_space": "bt709",
            "color_transfer": "bt709",
            "color_primaries": "bt709",
            "time_base": "1/90000",
        }
        assert source_stream == expected_stream_properties
        assert _video_stream_info(metadata_clean, ffprobe=ffprobe) == source_stream
        assert output_stream == expected_stream_properties
        assert _video_stream_info(all_output, ffprobe=ffprobe) == expected_stream_properties
        source_audio = _audio_bitstream(source, ffmpeg=ffmpeg)
        assert source_audio
        assert _audio_bitstream(metadata_clean, ffmpeg=ffmpeg) == source_audio
        assert _audio_bitstream(output, ffmpeg=ffmpeg) == source_audio
        assert _audio_bitstream(all_output, ffmpeg=ffmpeg) == source_audio

        output_scan = scan_video_marks(output, ("sora",))["sora"]
        all_output_scan = scan_video_marks(all_output, ("sora",))["sora"]
        assert all(
            region is None
            for region in stabilize_localizations(
                "sora",
                output_scan.detections,
                provenance=False,
            )
        )
        assert all(
            region is None
            for region in stabilize_localizations(
                "sora",
                all_output_scan.detections,
                provenance=False,
            )
        )

        x, y, width, height = mark_region
        untouched = np.ones(decoded_source[0].shape[:2], dtype=bool)
        padding = 24
        untouched[
            max(0, y - padding) : min(untouched.shape[0], y + height + padding),
            max(0, x - padding) : min(untouched.shape[1], x + width + padding),
        ] = False
        untouched_psnr: list[float] = []
        for original, cleaned in zip(decoded_source, decoded_output, strict=True):
            squared_error = (original.astype(np.float32) - cleaned.astype(np.float32)) ** 2
            mean_squared_error = float(np.mean(squared_error[untouched]))
            untouched_psnr.append(float(10 * np.log10((255**2) / mean_squared_error)))
        assert min(untouched_psnr) >= 35.0

        filled_region = np.logical_not(untouched)
        temporal_errors: list[float] = []
        independent_temporal_errors: list[float] = []
        for frame_index in range(1, frame_count):
            expected_delta = decoded_control[frame_index].astype(np.float32) - decoded_control[frame_index - 1].astype(
                np.float32
            )
            cleaned_delta = decoded_output[frame_index].astype(np.float32) - decoded_output[frame_index - 1].astype(
                np.float32
            )
            independent_delta = decoded_independent[frame_index].astype(np.float32) - decoded_independent[
                frame_index - 1
            ].astype(np.float32)
            temporal_errors.append(float(np.mean(np.abs(cleaned_delta[filled_region] - expected_delta[filled_region]))))
            independent_temporal_errors.append(
                float(np.mean(np.abs(independent_delta[filled_region] - expected_delta[filled_region])))
            )
        assert float(np.median(temporal_errors)) <= 1.5
        assert float(np.percentile(temporal_errors, 95)) <= 2.0
        assert float(np.median(temporal_errors)) < float(np.median(independent_temporal_errors))
        assert float(np.percentile(temporal_errors, 95)) <= float(np.percentile(independent_temporal_errors, 95))

    def test_preserves_variable_frame_timestamps(self, tmp_path: Path):
        from remove_ai_watermarks.video import remove_video_visible

        ffmpeg, ffprobe = _ffmpeg_test_tools()
        durations = [value for _ in range(6) for value in (1 / 30, 1 / 12, 1 / 20)]
        mark, mark_region = _independent_sora_mark()
        mark_x, mark_y, _mark_width, _mark_height = mark_region
        frames: list[np.ndarray] = []
        for frame_index in range(len(durations)):
            frame = _moving_video_background(frame_index)
            _stamp_gray_mark(frame, mark, x=mark_x, y=mark_y, opacity=0.78)
            frames.append(frame)
        source = tmp_path / "marked-vfr.mp4"
        output = tmp_path / "clean-vfr.mp4"
        _write_vfr_sora_clip(
            source,
            frames,
            durations=durations,
            ffmpeg=ffmpeg,
            start_offset=2.0,
        )
        source_timestamps = _video_frame_timestamps(source, ffprobe=ffprobe)
        source_intervals = np.diff(source_timestamps)

        assert len(source_timestamps) == len(frames)
        assert source_timestamps[0] == pytest.approx(2.0, abs=1 / 90000)
        assert float(np.ptp(source_intervals)) >= 0.03

        result = remove_video_visible(source, output, mark="sora", backend="cv2")
        output_timestamps = _video_frame_timestamps(output, ffprobe=ffprobe)

        assert result.removed_frames == len(frames)
        assert len(output_timestamps) == len(source_timestamps)
        assert output_timestamps == pytest.approx(source_timestamps, abs=1 / 90000)
        assert _stream_start_times(output, ffprobe=ffprobe) == pytest.approx(
            _stream_start_times(source, ffprobe=ffprobe),
            abs=1 / 90000,
        )
        assert _container_duration(output, ffprobe=ffprobe) == pytest.approx(
            _container_duration(source, ffprobe=ffprobe),
            abs=max(durations),
        )
        assert _audio_bitstream(output, ffmpeg=ffmpeg) == _audio_bitstream(source, ffmpeg=ffmpeg)

    def test_preserves_nonzero_start_on_constant_rate_clip(self, tmp_path: Path):
        from remove_ai_watermarks.video import remove_video_visible

        ffmpeg, ffprobe = _ffmpeg_test_tools()
        mark, mark_region = _independent_sora_mark()
        mark_x, mark_y, _mark_width, _mark_height = mark_region
        frames = []
        for frame_index in range(18):
            frame = _moving_video_background(frame_index)
            _stamp_gray_mark(frame, mark, x=mark_x, y=mark_y, opacity=0.78)
            frames.append(frame)
        source = tmp_path / "marked-offset.mp4"
        output = tmp_path / "clean-offset.mp4"
        _write_synthetic_sora_clip(
            source,
            frames,
            fps=12.0,
            ffmpeg=ffmpeg,
            start_offset=2.0,
        )

        result = remove_video_visible(source, output, mark="sora", backend="cv2")

        assert result.removed_frames == len(frames)
        assert _video_frame_timestamps(output, ffprobe=ffprobe) == pytest.approx(
            _video_frame_timestamps(source, ffprobe=ffprobe),
            abs=1 / 90000,
        )
        assert _stream_start_times(output, ffprobe=ffprobe) == pytest.approx(
            _stream_start_times(source, ffprobe=ffprobe),
            abs=1 / 90000,
        )
        assert _audio_bitstream(output, ffmpeg=ffmpeg) == _audio_bitstream(source, ffmpeg=ffmpeg)


class TestVideoVisibleApi:
    def test_auto_prefers_specific_sora_run_over_hailuo_cross_match(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video_visible
        from remove_ai_watermarks.video import remove_video_visible
        from remove_ai_watermarks.video_visible import FrameLocalization, VideoScan

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"
        sora_box = (4, 4, 20, 8)
        hailuo_box = (30, 40, 28, 12)
        sora_scan = VideoScan(
            width=64,
            height=64,
            fps=24.0,
            detections=tuple(FrameLocalization(index, 0.66, sora_box) for index in range(12)),
        )
        hailuo_scan = VideoScan(
            width=64,
            height=64,
            fps=24.0,
            detections=tuple(FrameLocalization(index, 0.35, hailuo_box) for index in range(12)),
        )

        def fake_scan(_source: Path, marks: tuple[str, ...]):
            assert marks == ("sora", "veo", "seedance", "doubao", "dola", "hailuo", "kling")
            return {
                "sora": sora_scan,
                "veo": sora_scan,
                "seedance": sora_scan,
                "dola": sora_scan,
                "hailuo": hailuo_scan,
                "kling": sora_scan,
            }

        monkeypatch.setattr(video_visible, "scan_video_marks", fake_scan)

        def fake_encode(
            _source: Path,
            target: Path,
            _scan: VideoScan,
            regions: list[tuple[int, int, int, int] | None],
            **kwargs: object,
        ) -> int:
            assert regions == [sora_box] * 12
            assert kwargs["padding_fraction"] == 0.28
            assert kwargs["temporal_consistency"] is True
            target.write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))
            return 12

        monkeypatch.setattr(video_visible, "encode_clean_video", fake_encode)

        result = remove_video_visible(source, output)

        assert result.output == output
        assert result.mark == "sora"

    def test_removes_stable_sora_run_and_writes_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from remove_ai_watermarks import video_visible
        from remove_ai_watermarks.video import remove_video_visible
        from remove_ai_watermarks.video_visible import FrameLocalization, VideoScan

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"
        box = (4, 4, 20, 8)
        scan = VideoScan(
            width=64,
            height=64,
            fps=24.0,
            detections=tuple(FrameLocalization(index, 0.66, box) for index in range(5)),
        )
        monkeypatch.setattr(
            video_visible,
            "scan_video_marks",
            lambda _source, marks: {"sora": scan} if marks == ("sora",) else {},
        )

        def fake_encode(
            _source: Path,
            target: Path,
            _scan: VideoScan,
            regions: list[tuple[int, int, int, int] | None],
            **kwargs: object,
        ) -> int:
            assert regions == [box] * 5
            assert kwargs["temporal_consistency"] is False
            target.write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))
            return 5

        monkeypatch.setattr(video_visible, "encode_clean_video", fake_encode)

        result = remove_video_visible(
            source,
            output,
            mark="sora",
            temporal_consistency=False,
        )

        assert result.output == output
        assert result.detected_frames == 5
        assert result.removed_frames == 5
        assert result.remaining_metadata == {}
        assert result.audio.stream_action == "copied_if_present"
        assert result.audio.watermark_status == "unverified"

    def test_no_stable_mark_writes_no_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from remove_ai_watermarks import video_visible
        from remove_ai_watermarks.video import remove_video_visible
        from remove_ai_watermarks.video_visible import FrameLocalization, VideoScan

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"
        scan = VideoScan(
            width=64,
            height=64,
            fps=24.0,
            detections=(
                FrameLocalization(0, 0.70, (1, 1, 20, 8)),
                FrameLocalization(1, 0.70, (30, 30, 20, 8)),
                FrameLocalization(2, 0.70, (1, 30, 20, 8)),
            ),
        )
        monkeypatch.setattr(
            video_visible,
            "scan_video_marks",
            lambda _source, marks: {"sora": scan} if marks == ("sora",) else {},
        )

        result = remove_video_visible(source, output, mark="sora")

        assert result.output is None
        assert result.removed_frames == 0
        assert result.audio.stream_action == "not_written"
        assert result.audio.watermark_status == "unverified"
        assert not output.exists()

    def test_dispatches_veo_detector_and_uses_tighter_mask(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video_visible
        from remove_ai_watermarks.video import remove_video_visible
        from remove_ai_watermarks.video_visible import FrameLocalization, VideoScan

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"
        box = (4, 4, 20, 20)
        scan = VideoScan(
            width=64,
            height=64,
            fps=24.0,
            detections=tuple(FrameLocalization(index, 0.60, box) for index in range(12)),
        )
        monkeypatch.setattr(
            video_visible,
            "scan_video_marks",
            lambda _source, marks: {"veo": scan} if marks == ("veo",) else {},
        )

        def fake_encode(
            _source: Path,
            target: Path,
            _scan: VideoScan,
            regions: list[tuple[int, int, int, int] | None],
            **kwargs: object,
        ) -> int:
            assert regions == [box] * 12
            assert kwargs["padding_fraction"] == 0.18
            assert kwargs["mask_style"] == "veo"
            target.write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))
            return 12

        monkeypatch.setattr(video_visible, "encode_clean_video", fake_encode)

        result = remove_video_visible(source, output, mark="veo")

        assert result.output == output
        assert result.mark == "veo"
        assert result.detected_frames == 12
        assert result.removed_frames == 12

    @pytest.mark.parametrize(
        ("mark", "mask_style"),
        [
            ("seedance", "box"),
            ("dola", "box"),
            ("hailuo", "box"),
            ("kling", "box"),
        ],
    )
    def test_dispatches_fixed_mark_detectors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mark: str,
        mask_style: str,
    ):
        from remove_ai_watermarks import video_visible
        from remove_ai_watermarks.video import remove_video_visible
        from remove_ai_watermarks.video_visible import FrameLocalization, VideoScan

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"
        box = (40, 40, 20, 12)
        scan = VideoScan(
            width=64,
            height=64,
            fps=24.0,
            detections=tuple(FrameLocalization(index, 0.60, box) for index in range(12)),
        )
        monkeypatch.setattr(
            video_visible,
            "scan_video_marks",
            lambda _source, marks: {mark: scan} if marks == (mark,) else {},
        )

        def fake_encode(
            _source: Path,
            target: Path,
            _scan: VideoScan,
            regions: list[tuple[int, int, int, int] | None],
            **kwargs: object,
        ) -> int:
            assert regions == [box] * 12
            assert kwargs["mask_style"] == mask_style
            target.write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))
            return 12

        monkeypatch.setattr(video_visible, "encode_clean_video", fake_encode)

        result = remove_video_visible(source, output, mark=mark)

        assert result.output == output
        assert result.mark == mark
        assert result.detected_frames == 12
        assert result.removed_frames == 12


class TestVideoVisibleCli:
    def test_help(self):
        result = CliRunner().invoke(main, ["video", "visible", "--help"])

        assert result.exit_code == 0, result.output
        assert "temporally stable" in result.output
        assert "auto|sora|veo|seedance|doubao|dola|hailuo|kling" in result.output
        assert "--temporal-consistency" in result.output

    def test_reports_removed_frames(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from remove_ai_watermarks import video
        from remove_ai_watermarks.video import VideoVisibleResult

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"
        monkeypatch.setattr(
            video,
            "remove_video_visible",
            lambda *_args, **_kwargs: VideoVisibleResult(
                source=source,
                output=output,
                mark="sora",
                total_frames=12,
                detected_frames=10,
                removed_frames=10,
                remaining_metadata={},
            ),
        )

        result = CliRunner().invoke(main, ["video", "visible", str(source), "-o", str(output)])

        assert result.exit_code == 0, result.output
        assert "10/12 frames" in result.output
        assert "Audio watermark: UNVERIFIED" in result.output
