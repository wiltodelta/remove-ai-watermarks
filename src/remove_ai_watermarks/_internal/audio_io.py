"""Bounded audio decoding and atomic lossless output through ffmpeg."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from remove_ai_watermarks import video_tools
from remove_ai_watermarks._internal.utils import staged_sibling
from remove_ai_watermarks.optional_deps import module_available

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from typing import BinaryIO

    from numpy.typing import NDArray

log = logging.getLogger(__name__)
_MAX_DECODE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class AudioStreamInfo:
    """Native geometry and timing of one audio stream, without resampling."""

    index: int
    sample_rate: int
    channels: int
    codec: str
    start_time: Fraction | None
    duration: Fraction | None
    format_name: str = ""
    channel_layout: str | None = None
    sample_fmt: str | None = None
    bits_per_raw_sample: int | None = None


def is_available() -> bool:
    """Whether the optional sample runtime and both external tools exist."""
    return module_available("numpy") and (
        all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")) or video_tools.bundled_video_tools_available()
    )


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None and name in {"ffmpeg", "ffprobe"}:
        path = video_tools.find_video_tool("ffmpeg" if name == "ffmpeg" else "ffprobe")
    if path is None:
        raise RuntimeError(f"{name} is required for audio processing")
    return path


def _fraction(value: object) -> Fraction | None:
    if value in (None, "", "N/A"):
        return None
    return Fraction(str(value))


def probe_audio(source: Path, *, stream: int = 0) -> AudioStreamInfo:
    """Inspect the requested audio-stream ordinal; reject unreadable input."""
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if stream < 0:
        raise ValueError("Audio stream ordinal must be nonnegative")
    entries = (
        "stream=index,codec_name,sample_fmt,bits_per_raw_sample,sample_rate,channels,channel_layout,"
        "time_base,start_pts,start_time,duration_ts,duration:"
        "stream_tags=:format=format_name:format_tags="
    )
    command = [
        require_tool("ffprobe"),
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        entries,
        "-of",
        "json",
        str(source),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    log.info(
        "Audio probe: command=%s status=%s stdout=%s stderr=%s",
        command,
        result.returncode,
        result.stdout,
        result.stderr,
    )
    if result.returncode:
        raise ValueError(f"Cannot read audio from {source}: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
        streams = payload["streams"]
        raw = streams[stream]
        rate, channels = int(raw["sample_rate"]), int(raw["channels"])
        if rate <= 0 or channels <= 0:
            raise ValueError("Invalid audio sample rate or channel count")
        time_base = _fraction(raw.get("time_base"))
        start = _fraction(raw.get("start_time"))
        duration = _fraction(raw.get("duration"))
        if time_base is not None:
            if raw.get("start_pts") not in (None, "N/A"):
                start = int(raw["start_pts"]) * time_base
            if raw.get("duration_ts") not in (None, "N/A"):
                duration = int(raw["duration_ts"]) * time_base
        raw_bits = raw.get("bits_per_raw_sample")
        bits_per_raw = int(raw_bits) if str(raw_bits).isdigit() and int(raw_bits) > 0 else None
        raw_fmt = raw.get("sample_fmt")
        return AudioStreamInfo(
            int(raw["index"]),
            rate,
            channels,
            str(raw["codec_name"]),
            start,
            duration,
            str(payload["format"]["format_name"]),
            raw.get("channel_layout") or {1: "mono", 2: "stereo"}.get(channels),
            str(raw_fmt) if raw_fmt not in (None, "", "N/A") else None,
            bits_per_raw,
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"No readable audio stream {stream} in {source}") from exc


def finish_process(process: subprocess.Popen[bytes], stderr: BinaryIO, command: list[str]) -> None:
    status = process.wait()
    stderr.seek(0)
    detail = stderr.read().decode(errors="replace")
    log.info("Audio ffmpeg completed: command=%s status=%s stderr=%s", command, status, detail)
    if status:
        log.warning("Audio ffmpeg failed: command=%s status=%s stderr=%s", command, status, detail)
        raise RuntimeError(f"ffmpeg audio processing failed ({status}): {detail.strip()}")


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _close_input(process: subprocess.Popen[bytes]) -> None:
    if process.stdin is not None and not process.stdin.closed:
        with suppress(BrokenPipeError):
            process.stdin.close()


def read_audio(
    source: Path, info: AudioStreamInfo, *, chunk_frames: int = 65536
) -> Generator[NDArray[Any], None, None]:
    """Yield interleaved float32 frames; close this iterator when stopping early."""
    if not is_available():
        raise ImportError("Audio sample processing requires NumPy, ffmpeg, and ffprobe")
    import numpy as np

    if chunk_frames <= 0:
        raise ValueError("Audio chunk size must be positive")
    command = [
        require_tool("ffmpeg"),
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(Path(source).resolve()),
        "-map",
        f"0:{info.index}",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        str(info.channels),
        "-ar",
        str(info.sample_rate),
        "-c:a",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    ]
    frame_bytes = 4 * info.channels
    if frame_bytes > _MAX_DECODE_BYTES:
        raise ValueError("Audio channel layout exceeds the decoding memory budget")
    chunk_frames = min(chunk_frames, _MAX_DECODE_BYTES // frame_bytes)
    total = 0
    with tempfile.TemporaryFile() as stderr:
        log.info("Audio decode: command=%s", command)
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr)  # noqa: S603
        if process.stdout is None:
            stop_process(process)
            raise RuntimeError("Audio decoder did not open its output pipe")
        try:
            while raw := process.stdout.read(chunk_frames * frame_bytes):
                if len(raw) % frame_bytes:
                    raise RuntimeError("Audio decoder returned an incomplete sample frame")
                frames = np.frombuffer(raw, dtype="<f4").reshape(-1, info.channels).copy()
                total += len(frames)
                yield frames
            finish_process(process, stderr, command)
            if total == 0:
                raise ValueError("Audio stream contains no decoded samples")
        finally:
            process.stdout.close()
            stop_process(process)


_MP3_RATES = frozenset({8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000})


def _source_pcm_bit_depth(info: AudioStreamInfo) -> int | None:
    if info.codec in {"pcm_s16le", "pcm_s16be"} or (
        info.sample_fmt in {"s16", "s16p"} and info.bits_per_raw_sample in (None, 16)
    ):
        return 16
    if info.codec in {"pcm_s24le", "pcm_s24be"} or info.bits_per_raw_sample == 24:
        return 24
    return None


def _audio_codec_args(suffix: str, info: AudioStreamInfo, *, bitexact: bool) -> list[str]:
    pcm_depth = None if bitexact else _source_pcm_bit_depth(info)
    if suffix == ".wav":
        wav_codec = {16: "pcm_s16le", 24: "pcm_s24le"}.get(pcm_depth or 32, "pcm_f32le")
        return ["-c:a", wav_codec, "-rf64", "never" if bitexact else "auto"]
    if suffix == ".flac":
        if pcm_depth == 16:
            return ["-c:a", "flac", "-sample_fmt", "s16"]
        return ["-c:a", "flac", "-sample_fmt", "s32"]
    if suffix == ".m4a":
        return ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]
    if suffix == ".mp3":
        if info.channels not in (1, 2):
            raise ValueError("MP3 output supports mono or stereo; use WAV, FLAC, or M4A")
        if info.sample_rate not in _MP3_RATES:
            raise ValueError(f"MP3 output does not support {info.sample_rate} Hz; use WAV, FLAC, or M4A")
        return ["-c:a", "libmp3lame", "-b:a", "256k", "-write_xing", "1"]
    if suffix in {".ogg", ".opus"}:
        if info.channels not in (1, 2):
            raise ValueError("Opus/Ogg output supports mono or stereo; use WAV, FLAC, or M4A")
        return ["-c:a", "libopus", "-b:a", "192k"]
    raise ValueError("Audio sample output must be WAV, FLAC, MP3, M4A, OGG, or OPUS")


def write_audio(
    info: AudioStreamInfo,
    output: Path,
    chunks: Iterable[NDArray[Any]],
    *,
    expected_frames: int | None = None,
    bitexact: bool = False,
) -> int:
    """Publish a processed audio file only after every frame and the encoder succeed."""
    if not is_available():
        raise ImportError("Audio sample processing requires NumPy, ffmpeg, and ffprobe")
    import numpy as np

    output = Path(output)
    suffix = output.suffix.lower()
    if bitexact and suffix != ".wav":
        raise ValueError("Bitexact serialization requires floating-point WAV")
    codec = _audio_codec_args(suffix, info, bitexact=bitexact)
    with staged_sibling(output) as staged, tempfile.TemporaryFile() as stderr, ExitStack() as resources:
        iterator = iter(chunks)
        close = getattr(iterator, "close", None)
        if callable(close):
            resources.callback(close)
        command = [
            require_tool("ffmpeg"),
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(info.sample_rate),
            "-ac",
            str(info.channels),
            *(["-channel_layout", info.channel_layout] if info.channel_layout else []),
            "-i",
            "pipe:0",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            *codec,
            *(["-flags:a", "+bitexact", "-fflags", "+bitexact"] if bitexact else []),
            str(staged),
        ]
        log.info("Audio encode: command=%s", command)
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr)  # noqa: S603
        resources.callback(stop_process, process)
        resources.callback(_close_input, process)
        if process.stdin is None:
            raise RuntimeError("Audio encoder did not open its input pipe")
        count = 0
        try:
            for chunk in iterator:
                frames = np.asarray(chunk, dtype="<f4")
                if frames.ndim != 2 or frames.shape[1] != info.channels:
                    raise ValueError("Audio chunk channel count differs from the source stream")
                if bitexact and (count + len(frames)) * info.channels * 4 > 0xFFFF0000:
                    raise ValueError("Bitexact WAV exceeds the short RIFF size bound; use the default RF64 writer")
                if not np.isfinite(frames).all():
                    raise ValueError("Audio samples must be finite")
                if suffix == ".flac" and frames.size and (float(frames.min()) < -1 or float(frames.max()) > 1):
                    raise ValueError("FLAC cannot preserve samples outside [-1, 1]; use floating-point WAV")
                process.stdin.write(np.ascontiguousarray(frames).tobytes())
                count += len(frames)
            if count == 0 or (expected_frames is not None and count != expected_frames):
                raise ValueError(f"Audio frame count mismatch: wrote {count}, expected {expected_frames}")
            process.stdin.close()
            finish_process(process, stderr, command)
            delivered = probe_audio(staged)
            expected_sr = 48000 if suffix in {".ogg", ".opus"} else info.sample_rate
            if (delivered.sample_rate, delivered.channels) != (expected_sr, info.channels):
                raise RuntimeError("Audio encoder changed the sample rate or channel count")
            if (
                info.channel_layout
                and suffix in {".wav", ".flac", ".m4a"}
                and delivered.channel_layout != info.channel_layout
            ):
                raise RuntimeError("Audio encoder changed the channel layout")
        except BrokenPipeError as exc:
            _close_input(process)
            finish_process(process, stderr, command)
            raise RuntimeError("Audio encoder stopped accepting samples") from exc
    return count
