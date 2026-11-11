"""Shared ffmpeg frame-encoding helpers."""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from typing import Any, BinaryIO

from remove_ai_watermarks.video_tools import find_video_tool, require_video_tool

log = logging.getLogger(__name__)

_FFMPEG_STDERR_LIMIT = 64 * 1024
_FFMPEG_STDERR_TRUNCATION = b"\n...[ffmpeg stderr truncated]...\n"
_PIXEL_FORMATS = frozenset({"yuv420p", "yuv422p", "yuv444p"})
_PIXEL_FORMAT_ALIASES = {
    "yuvj420p": "yuv420p",
    "yuvj422p": "yuv422p",
    "yuvj444p": "yuv444p",
}
_COLOR_RANGES = frozenset({"tv", "pc"})
_COLOR_SPACES = frozenset({"bt709", "fcc", "bt470bg", "smpte170m", "smpte240m"})
_COLOR_TRANSFERS = frozenset(
    {
        "bt709",
        "gamma22",
        "gamma28",
        "smpte170m",
        "smpte240m",
        "linear",
        "log",
        "log_sqrt",
        "iec61966-2-4",
        "bt1361e",
        "iec61966-2-1",
        "bt2020-10",
        "bt2020-12",
        "smpte2084",
        "smpte428",
        "arib-std-b67",
    }
)
_COLOR_PRIMARIES = frozenset(
    {
        "bt709",
        "bt470m",
        "bt470bg",
        "smpte170m",
        "smpte240m",
        "film",
        "bt2020",
        "smpte428",
        "smpte431",
        "smpte432",
        "jedec-p22",
        "ebu3213",
    }
)


@dataclass(frozen=True)
class VideoEncodeProfile:
    """Source video properties that the raw-frame encoder can preserve."""

    pixel_format: str = "yuv420p"
    color_range: str | None = None
    color_space: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None
    time_base: str | None = None
    start_pts: int | None = None
    source_pixel_format: str | None = None
    component_depth: int | None = None


@dataclass
class _RawVideoEncoder:
    """Running ffmpeg process with diagnostics redirected outside a pipe."""

    process: subprocess.Popen[bytes]
    stdin: BinaryIO
    _stderr_buffer: BinaryIO

    def poll(self) -> int | None:
        return self.process.poll()

    def kill(self) -> None:
        self.process.kill()

    def wait(self) -> int:
        return self.process.wait()

    def collect_stderr(self) -> str:
        """Return bounded ffmpeg diagnostics and release the temporary file."""
        if self._stderr_buffer.closed:
            raise RuntimeError("ffmpeg diagnostics have already been collected")
        try:
            return _read_bounded_stderr(self._stderr_buffer)
        finally:
            self._stderr_buffer.close()

    def discard_stderr(self) -> None:
        """Release the diagnostic buffer after an abort."""
        if self._stderr_buffer.closed:
            return
        self._stderr_buffer.close()


def _read_bounded_stderr(buffer: BinaryIO) -> str:
    """Read bounded head-and-tail diagnostics from a seekable binary stream."""
    buffer.seek(0, os.SEEK_END)
    size = buffer.tell()
    if size <= _FFMPEG_STDERR_LIMIT:
        buffer.seek(0)
        raw_stderr = buffer.read()
    else:
        payload_limit = _FFMPEG_STDERR_LIMIT - len(_FFMPEG_STDERR_TRUNCATION)
        head_size = payload_limit // 2
        tail_size = payload_limit - head_size
        buffer.seek(0)
        head = buffer.read(head_size)
        buffer.seek(-tail_size, os.SEEK_END)
        tail = buffer.read(tail_size)
        raw_stderr = head + _FFMPEG_STDERR_TRUNCATION + tail
    return raw_stderr.decode("utf-8", errors="replace")


def _known_value(value: object, allowed: frozenset[str]) -> str | None:
    """Return a supported ffmpeg enum value, otherwise omit it."""
    return value if isinstance(value, str) and value in allowed else None


def _time_base(value: object) -> str | None:
    """Normalize a positive ffprobe time base."""
    if not isinstance(value, str):
        return None
    try:
        fraction = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    if fraction <= 0:
        return None
    return f"{fraction.numerator}/{fraction.denominator}"


def _pixel_component_depth(pixel_format: object, raw_bits: object) -> int | None:
    """Return the largest component depth reported by ffprobe or PyAV."""
    depths: list[int] = []
    if isinstance(raw_bits, str) and raw_bits.isdigit():
        depths.append(int(raw_bits))
    if isinstance(pixel_format, str):
        try:
            import av

            depths.extend(component.bits for component in av.VideoFormat(pixel_format).components)
        except (ImportError, ValueError):
            pass
    return max(depths) if depths else None


@dataclass(frozen=True)
class VideoEndRepair:
    """Durations for the separate decode-order and presentation-order ends."""

    last_packet_index: int
    last_packet_duration: int
    final_frame_pts: int
    final_frame_duration: int
    movie_timescale: int

    @property
    def bitstream_filter(self) -> str:
        return (
            "setts=pts=PTS:dts=DTS:duration='"
            f"if(eq(N,{self.last_packet_index}),{self.last_packet_duration},"
            f"if(eq(PTS,{self.final_frame_pts}),{self.final_frame_duration},DURATION))'"
        )


def video_end_repair(source_clock: dict[str, Any], encoded_clock: dict[str, Any]) -> VideoEndRepair:
    """Fit both packet ends to the original stream window without moving PTS/DTS."""
    try:
        source = source_clock["streams"][0]
        encoded = encoded_clock["streams"][0]
        source_base = Fraction(source["time_base"])
        encoded_base = Fraction(encoded["time_base"])
        packets = encoded_clock["packets"]
        source_duration = int(source["duration_ts"]) * source_base
        end_pts = round((int(source["start_pts"]) * source_base + source_duration) / encoded_base)
        final_pts = max(int(packet["pts"]) for packet in packets)
        presentation_duration = end_pts - final_pts
        decode_duration = round(source_duration / encoded_base) - (int(packets[-1]["dts"]) - int(packets[0]["dts"]))
        if int(packets[-1]["pts"]) == final_pts:
            decode_duration = max(decode_duration, presentation_duration)
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("Could not read the terminal video packet clocks") from exc
    if (
        source_base <= 0
        or encoded_base <= 0
        or source_duration <= 0
        or min(presentation_duration, decode_duration) <= 0
    ):
        raise ValueError("Encoded packet clocks do not fit the source video window")
    return VideoEndRepair(
        last_packet_index=len(packets) - 1,
        last_packet_duration=decode_duration,
        final_frame_pts=final_pts,
        final_frame_duration=presentation_duration,
        movie_timescale=encoded_base.denominator,
    )


def _run_ffprobe(
    source: Path,
    arguments: Sequence[str],
    *,
    purpose: str,
    output_format: str,
) -> str | None:
    """Run one ffprobe query with shared logging and failure handling."""
    ffprobe = find_video_tool("ffprobe")
    if ffprobe is None:
        log.warning("ffprobe is unavailable; cannot inspect %s of %s", purpose, source)
        return None
    command = [ffprobe, "-v", "error", *arguments, "-of", output_format, str(source)]
    result = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        check=False,
        text=True,
    )
    log.info(
        "ffprobe %s: command=%s status=%s stdout=%s stderr=%s",
        purpose,
        command,
        result.returncode,
        result.stdout,
        result.stderr,
    )
    if result.returncode != 0:
        log.warning("ffprobe could not inspect %s for %s", purpose, source)
        return None
    return result.stdout


def probe_video_encode_profile(source: Path) -> VideoEncodeProfile:
    """Read source properties that survive the package's 8-bit BGR boundary."""
    raw_profile = _run_ffprobe(
        source,
        (
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=pix_fmt,bits_per_raw_sample,color_range,color_space,color_transfer,color_primaries,time_base,start_pts",
        ),
        purpose="video profile",
        output_format="json",
    )
    if raw_profile is None:
        return VideoEncodeProfile()
    try:
        payload = json.loads(raw_profile)
        streams = payload.get("streams", [])
        stream = streams[0]
    except (AttributeError, IndexError, TypeError, json.JSONDecodeError):
        log.warning("ffprobe returned no usable video profile for %s; using yuv420p", source)
        return VideoEncodeProfile()

    raw_pixel_format = stream.get("pix_fmt")
    pixel_format = _PIXEL_FORMAT_ALIASES.get(raw_pixel_format, raw_pixel_format)
    if pixel_format not in _PIXEL_FORMATS:
        pixel_format = "yuv420p"
    time_base = _time_base(stream.get("time_base"))
    raw_start_pts = stream.get("start_pts")
    start_pts = raw_start_pts if isinstance(raw_start_pts, int) and time_base is not None else None
    return VideoEncodeProfile(
        pixel_format=pixel_format,
        color_range=_known_value(stream.get("color_range"), _COLOR_RANGES),
        color_space=_known_value(stream.get("color_space"), _COLOR_SPACES),
        color_transfer=_known_value(stream.get("color_transfer"), _COLOR_TRANSFERS),
        color_primaries=_known_value(stream.get("color_primaries"), _COLOR_PRIMARIES),
        time_base=time_base,
        start_pts=start_pts,
        source_pixel_format=raw_pixel_format if isinstance(raw_pixel_format, str) else None,
        component_depth=_pixel_component_depth(raw_pixel_format, stream.get("bits_per_raw_sample")),
    )


def probe_video_timestamps(source: Path) -> tuple[float, ...]:
    """Read authoritative display timestamps for the first video stream."""
    raw_timestamps = _run_ffprobe(
        source,
        (
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
        ),
        purpose="video timestamps",
        output_format="csv=p=0",
    )
    if raw_timestamps is None:
        return ()
    try:
        timestamps = tuple(float(line.split(",", 1)[0]) for line in raw_timestamps.splitlines() if line)
    except ValueError:
        log.warning("ffprobe returned unusable frame timestamps for %s", source)
        return ()
    if not timestamps or not all(math.isfinite(timestamp) for timestamp in timestamps):
        log.warning("ffprobe returned no finite frame timestamps for %s", source)
        return ()
    return timestamps


@contextmanager
def _temporary_video_path(output: Path, *, prefix: str) -> Generator[Path]:
    """Yield one sibling temporary path and remove it on exit."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=prefix,
        suffix=output.suffix,
        dir=output.parent,
        delete=False,
    ) as stream:
        temporary_output = Path(stream.name)
    try:
        yield temporary_output
    finally:
        temporary_output.unlink(missing_ok=True)


@contextmanager
def atomic_video_output(output: Path) -> Generator[Path]:
    """Yield a sibling temporary path and publish it only after success."""
    with _temporary_video_path(output, prefix=f".{output.stem}-") as temporary_output:
        yield temporary_output
        os.replace(temporary_output, output)


def _video_codec_args(suffix: str, *, crf: int, profile: VideoEncodeProfile) -> list[str]:
    if suffix == ".webm":
        return ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0"]
    args = ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf)]
    x264_params: list[str] = []
    if profile.color_primaries == "bt709":
        x264_params.append("colorprim=bt709")
    if profile.color_transfer == "bt709":
        x264_params.append("transfer=bt709")
    if profile.color_space == "bt709":
        x264_params.append("colormatrix=bt709")
    if profile.color_range is not None:
        x264_params.append(f"range={'full' if profile.color_range == 'pc' else 'limited'}")
    if x264_params:
        args.extend(["-x264-params", ":".join(x264_params)])
    return args


def _profile_args(profile: VideoEncodeProfile) -> list[str]:
    """Build generic output options for source properties ffmpeg understands."""
    args = ["-pix_fmt", profile.pixel_format]
    for option, value in (
        ("-color_range", profile.color_range),
        ("-colorspace", profile.color_space),
        ("-color_trc", profile.color_transfer),
        ("-color_primaries", profile.color_primaries),
    ):
        if value is not None:
            args.extend([option, value])
    if profile.time_base is not None:
        args.extend(["-enc_time_base:v", profile.time_base])
    return args


def raw_video_command(
    output: Path,
    *,
    width: int,
    height: int,
    fps: float,
    crf: int,
    profile: VideoEncodeProfile,
    timestamped_input: bool = False,
    copy_input_timestamps: bool = False,
) -> list[str]:
    """Build an ffmpeg command for BGR frames on standard input."""
    ffmpeg = require_video_tool("ffmpeg")
    frame_input = (
        ["-f", "nut", "-i", "pipe:0"]
        if timestamped_input
        else [
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
        ]
    )
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        *(["-copyts"] if copy_input_timestamps else []),
        *frame_input,
        "-map",
        "0:v:0",
        *_video_codec_args(output.suffix.lower(), crf=crf, profile=profile),
        *_profile_args(profile),
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
    ]
    command.extend(["-fps_mode", "passthrough"])
    if copy_input_timestamps:
        command.extend(["-avoid_negative_ts", "disabled"])
    if output.suffix.lower() in {".mp4", ".mov", ".m4v"} and profile.time_base is not None:
        numerator, denominator = (int(part) for part in profile.time_base.split("/", 1))
        if numerator == 1:
            command.extend(["-video_track_timescale", str(denominator)])
    command.append(str(output))
    return command


def mux_encoded_video(
    encoded_video: Path,
    source: Path,
    output: Path,
    *,
    strip_metadata: bool,
    copy_input_timestamps: bool = False,
) -> None:
    """Copy encoded video and source audio into the final container."""
    ffmpeg = require_video_tool("ffmpeg")
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        *(["-copyts"] if copy_input_timestamps else []),
        "-i",
        str(encoded_video),
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c",
        "copy",
        "-map_metadata",
        "-1" if strip_metadata else "1",
        "-map_chapters",
        "-1" if strip_metadata else "1",
    ]
    if copy_input_timestamps:
        command.extend(["-avoid_negative_ts", "disabled"])
    if output.suffix.lower() in {".mp4", ".mov", ".m4v"}:
        command.extend(["-movflags", "+faststart"])
    command.append(str(output))
    with tempfile.TemporaryFile(mode="w+b") as stderr_buffer:
        result = subprocess.run(  # noqa: S603
            command,
            stdout=subprocess.DEVNULL,
            stderr=stderr_buffer,
            check=False,
        )
        stderr = _read_bounded_stderr(stderr_buffer)
    log.info(
        "ffmpeg video mux: command=%s status=%s stderr=%s",
        command,
        result.returncode,
        stderr,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to mux {output}: {stderr.strip()[:500]}")


@contextmanager
def staged_video_output(output: Path) -> Generator[tuple[Path, Path]]:
    """Yield video-only and final temporary paths, then publish atomically."""
    with (
        atomic_video_output(output) as temporary_output,
        _temporary_video_path(
            output,
            prefix=f".{output.stem}-video-",
        ) as encoded_video,
    ):
        yield encoded_video, temporary_output


def start_raw_video_encoder(command: list[str]) -> _RawVideoEncoder:
    """Start ffmpeg and validate its raw-frame pipes."""
    log.info("Starting ffmpeg video encode: command=%s", command)
    stderr_buffer = cast(
        "BinaryIO",
        tempfile.TemporaryFile(mode="w+b"),  # noqa: SIM115 - encoder owns the lifetime
    )
    try:
        process = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr_buffer,
        )
    except Exception:
        stderr_buffer.close()
        raise
    if process.stdin is None:
        process.kill()
        process.wait()
        stderr_buffer.close()
        raise RuntimeError("Could not open ffmpeg input pipe")
    return _RawVideoEncoder(process, cast("BinaryIO", process.stdin), stderr_buffer)


def finish_raw_video_encoder(
    process: _RawVideoEncoder,
    output: Path,
    *,
    operation: str,
) -> None:
    """Close the frame stream and raise when ffmpeg rejects the encode."""
    process.stdin.close()
    return_code = process.wait()
    stderr = process.collect_stderr()
    log.info("ffmpeg %s finished: status=%s stderr=%s", operation, return_code, stderr)
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed to encode {output}: {stderr.strip()[:500]}")


def abort_raw_video_encoder(process: _RawVideoEncoder) -> None:
    """Stop an incomplete ffmpeg encode."""
    if process.poll() is None:
        process.kill()
    process.wait()
    process.discard_stderr()
