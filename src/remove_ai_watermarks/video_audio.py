"""Process one continuous audio track while preserving other encoded streams."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from remove_ai_watermarks import audio
from remove_ai_watermarks._internal import audio_io
from remove_ai_watermarks._internal.audio_quality import audio_quality_passed, measure_fidelity, sha256_file
from remove_ai_watermarks._internal.utils import staged_sibling

if TYPE_CHECKING:
    from collections.abc import Generator

log = logging.getLogger(__name__)

_CAVEATS = (
    "Only the selected audio track is processed; other tracks and video marks remain unchanged.",
    "Metadata is not intentionally stripped; container conversion can change it. "
    "Verification covers the named decoder.",
)
_PCM_CAVEAT = "The selected track is stored as uncompressed float PCM, which can increase file size."


@dataclass(frozen=True)
class VideoAudioRemovalResult:
    """Audio verification and packet-preservation evidence for a delivered video."""

    source: Path
    output: Path | None
    before: audio.AudioProvenanceReport
    after: audio.AudioProvenanceReport | None
    frames_processed: int
    verified_removed: bool | None
    skipped: bool
    fidelity: audio.AudioFidelity | None = None
    unchanged_streams: tuple[int, ...] = ()
    source_audio_start: Fraction | None = None
    output_audio_start: Fraction | None = None
    method: str = "spectral"
    bandwidth_kbps: float | None = None
    vocoder_device: str | None = None
    caveats: tuple[str, ...] = (
        *_CAVEATS,
        _PCM_CAVEAT,
    )
    quality_passed: bool | None = None


@dataclass(frozen=True)
class VideoAudioMultiStreamResult:
    """Multi-track audio verification and packet-preservation evidence for a video."""

    source: Path
    output: Path | None
    streams: tuple[VideoAudioRemovalResult, ...]
    processed_ordinals: tuple[int, ...]
    unchanged_streams: tuple[int, ...]
    verified_removed: bool | None
    skipped: bool
    caveats: tuple[str, ...] = ()
    quality_passed: bool | None = None


def _streams(source: Path) -> list[dict[str, Any]]:
    command = [
        audio_io.require_tool("ffprobe"),
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type,codec_name,sample_rate,channels,channel_layout,duration,time_base:"
        "stream_disposition:stream_tags=language",
        "-of",
        "json",
        str(source),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    log.info(
        "Video audio stream probe: command=%s status=%s stdout=%s stderr=%s",
        command,
        result.returncode,
        result.stdout,
        result.stderr,
    )
    if result.returncode:
        raise ValueError(f"Cannot inspect video streams: {result.stderr.strip()}")
    streams: list[dict[str, Any]] = json.loads(result.stdout)["streams"]
    if not any(s.get("codec_type") == "video" for s in streams):
        raise ValueError("Source has no video stream")
    return streams


def _lines(command: list[str]) -> Generator[str]:
    with tempfile.TemporaryFile(mode="w+b") as stderr:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr)  # noqa: S603
        log.info("Reading media evidence: command=%s", command)
        try:
            if process.stdout is None:
                raise RuntimeError("Media evidence pipe did not open")
            for line in process.stdout:
                yield line.decode("utf-8").strip()
            audio_io.finish_process(process, stderr, command)
        finally:
            if process.stdout is not None:
                process.stdout.close()
            audio_io.stop_process(process)


def _timeline(source: Path, index: int, rate: int, tick: Fraction) -> tuple[Fraction, int]:
    command = [
        audio_io.require_tool("ffprobe"),
        "-v",
        "error",
        "-select_streams",
        str(index),
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time,nb_samples",
        "-of",
        "compact=p=0",
        str(source),
    ]
    rows = _lines(command)
    first: Fraction | None = None
    count = 0
    try:
        for line in rows:
            fields = dict(part.split("=", 1) for part in line.split("|") if "=" in part)
            if not fields:
                continue
            try:
                start, samples = Fraction(fields["best_effort_timestamp_time"]), int(fields["nb_samples"])
            except (KeyError, ValueError) as exc:
                raise ValueError("Audio frame timing is unavailable") from exc
            if samples <= 0:
                raise ValueError("Audio frame has no samples")
            if first is None:
                first = start
            if abs(start - (first + Fraction(count, rate))) > max(tick, Fraction(1, rate)):
                raise ValueError("Discontinuous audio timestamps require a different processing path")
            count += samples
    finally:
        rows.close()
    if first is None:
        raise ValueError("Audio track has no decoded frames")
    return first, count


def _packets(source: Path, index: int) -> Generator[tuple[Fraction, tuple[Fraction, ...], tuple[str, ...]]]:
    command = [
        audio_io.require_tool("ffmpeg"),
        "-v",
        "error",
        "-nostdin",
        "-copyts",
        "-i",
        str(source),
        "-map",
        f"0:{index}",
        "-c",
        "copy",
        "-copytb",
        "1",
        "-f",
        "framemd5",
        "-hash",
        "sha256",
        "-",
    ]
    rows = _lines(command)
    tick: Fraction | None = None
    try:
        for line in rows:
            if line.startswith("#tb 0:"):
                tick = Fraction(line.split(":", 1)[1].strip())
            if not line or line.startswith("#"):
                continue
            if tick is None:
                raise RuntimeError("Packet verifier did not report a time base")
            fields = tuple(v.strip() for v in line.split(","))
            yield tick, tuple(Fraction(int(v)) * tick for v in fields[1:4]), fields[4:6]
    finally:
        rows.close()


def _verify_packets(source: Path, output: Path, index: int, *, output_index: int | None = None) -> None:
    before = _packets(source, index)
    after = _packets(output, index if output_index is None else output_index)
    try:
        for a, b in zip_longest(before, after):
            if a is None or b is None or a[2] != b[2]:
                raise RuntimeError(f"Stream {index} encoded packets changed during audio processing")
            tolerances = (b[0], b[0], max(a[0], b[0]))
            if any(abs(x - y) > tolerance for x, y, tolerance in zip(a[1], b[1], tolerances, strict=True)):
                raise RuntimeError(f"Stream {index} packet timing changed during audio processing")
    finally:
        before.close()
        after.close()


def _audio_packet_trimming(source: Path, index: int) -> Generator[tuple[int, int]]:
    """Read decoder priming and tail trimming independently of packet payloads."""
    rows = _lines(
        [
            audio_io.require_tool("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            str(index),
            "-show_packets",
            "-show_entries",
            "packet=stream_index:packet_side_data=side_data_type,skip_samples,discard_padding",
            "-of",
            "compact=p=0",
            str(source),
        ]
    )
    try:
        for line in rows:
            if not line:
                continue
            fields = dict(part.rsplit(":", 1)[-1].split("=", 1) for part in line.split("|") if "=" in part)
            yield int(fields.get("skip_samples", 0)), int(fields.get("discard_padding", 0))
    finally:
        rows.close()


def _verify_audio_trimming(source: Path, output: Path, index: int) -> None:
    before, after = _audio_packet_trimming(source, index), _audio_packet_trimming(output, index)
    try:
        if any(a != b for a, b in zip_longest(before, after)):
            raise RuntimeError(f"Stream {index} audio priming or tail trimming changed during audio processing")
    finally:
        before.close()
        after.close()


def remove_video_audio(
    source: str | Path,
    output: str | Path,
    *,
    stream: int = 0,
    force: bool = False,
    verify: bool = True,
    method: audio.AudioMethod = "auto",
    bandwidth_kbps: float = 24.0,
    scheme: audio.AudioScheme = "audioseal",
    vocoder_device: Literal["cpu", "mps", "cuda"] = "cpu",
    _before: audio.AudioProvenanceReport | None = None,
) -> VideoAudioRemovalResult:
    """Filter one audio track, copying every other stream without shortening tails.

    MKV and MOV store float PCM; MP4 encodes the selected track as AAC at a
    target 192 kbps. AAC is lossy and final verification covers its decoded audio. The source
    must have a continuous decoded audio timeline. Publication requires exact
    packet payload preservation and PTS/DTS within one output time-base tick
    for unchanged streams. Default watermark verification is source-positive
    gated and repeated on the actual mux, not inherited from an intermediate.
    Packet durations may differ by one source or output time-base tick.
    """
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser()
    if output.suffix.lower() not in {".mkv", ".mov", ".mp4"}:
        raise ValueError("Audio-only video output must be MKV, MOV, or MP4")
    lossy = output.suffix.lower() == ".mp4"
    caveats = (
        *_CAVEATS,
        "The selected track uses lossy AAC with a target bitrate of 192 kbps; "
        "verification and fidelity cover the delivered decoded audio."
        if lossy
        else _PCM_CAVEAT,
    )
    if output.resolve() == source or (output.exists() and output.samefile(source)):
        raise ValueError("Video output must differ from the source")
    streams = _streams(source)
    if [s["index"] for s in streams] != list(range(len(streams))):
        raise ValueError("Video stream indices must be contiguous")
    info = audio_io.probe_audio(source, stream=stream)
    selected = next(s for s in streams if s["index"] == info.index)
    start, count = _timeline(source, info.index, info.sample_rate, Fraction(selected["time_base"]))
    profiles = audio.resolve_audio_profiles(
        method=method,
        scheme=scheme,
        verify=verify,
        force=force,
        bandwidth_kbps=bandwidth_kbps,
        vocoder_device=vocoder_device,
    )
    before = (
        _before
        if _before is not None
        else audio.identify_audio(source, check_invisible=verify, stream=stream, scheme=scheme)
    )
    if method == "auto":
        profiles = audio.adaptive_auto_profiles(scheme, before)
    if not force and before.watermark_status != "detected":
        first = profiles[0]
        return VideoAudioRemovalResult(
            source,
            None,
            before,
            None,
            0,
            None,
            True,
            method=first,
            bandwidth_kbps=bandwidth_kbps if first in {"encodec", "combined"} else None,
            vocoder_device=vocoder_device if first in {"bigvgan", "mimi", "focalcodec", "dac"} else None,
            caveats=caveats,
        )
    if not verify or before.watermark_status != "detected" or (not force and audio.missing_decoder_reason(before)):
        profiles = profiles[:1]
    last: audio.AudioVerificationIncomplete | None = None
    with tempfile.TemporaryDirectory(prefix="raiw-video-audio-") as tmp:
        for attempt, profile in enumerate(profiles):
            attempt_force = force and attempt == len(profiles) - 1
            candidate = Path(tmp) / f"processed-{attempt}.wav"
            try:
                processed = audio.remove_audio_profile(
                    source,
                    candidate,
                    stream=stream,
                    force=attempt_force,
                    verify=verify,
                    method=profile,
                    bandwidth_kbps=bandwidth_kbps,
                    scheme=scheme,
                    _before=before,
                    _skip_post_verify=True,
                    vocoder_device=vocoder_device,
                )
                if processed.frames_processed != count:
                    raise RuntimeError("Audio timeline and decoded sample-frame counts disagree")
                output.parent.mkdir(parents=True, exist_ok=True)
                with staged_sibling(output) as staged:
                    command = [
                        audio_io.require_tool("ffmpeg"),
                        "-v",
                        "error",
                        "-nostdin",
                        "-y",
                        "-copyts",
                        "-i",
                        str(source),
                        "-itsoffset",
                        f"{float(start):.9f}",
                        "-i",
                        str(candidate),
                    ]
                    for s in streams:
                        command.extend(["-map", "1:a:0" if s["index"] == info.index else f"0:{s['index']}"])
                    for s in streams:
                        flags = "+".join(k for k, v in s.get("disposition", {}).items() if v) or "0"
                        command.extend([f"-disposition:{s['index']}", flags])
                        command.extend([f"-map_metadata:s:{s['index']}", f"0:s:{s['index']}"])
                    command.extend(
                        [
                            "-c",
                            "copy",
                            f"-c:{info.index}",
                            "aac" if lossy else "pcm_f32le",
                            "-map_metadata",
                            "0",
                            "-map_chapters",
                            "0",
                            "-avoid_negative_ts",
                            "disabled",
                            str(staged),
                        ]
                    )
                    if lossy:
                        command[-1:-1] = [f"-b:{info.index}", "192k"]
                    with tempfile.TemporaryFile(mode="w+b") as stderr:
                        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr)  # noqa: S603
                        try:
                            audio_io.finish_process(process, stderr, command)
                        finally:
                            audio_io.stop_process(process)
                    delivered_streams = _streams(staged)
                    if [(s["index"], s["codec_type"]) for s in streams] != [
                        (s["index"], s["codec_type"]) for s in delivered_streams
                    ]:
                        raise RuntimeError("Audio processing changed the video stream layout")
                    for before_stream, after_stream in zip(streams, delivered_streams, strict=True):
                        if before_stream.get("disposition") != after_stream.get("disposition") or (
                            before_stream.get("tags", {}).get("language", "und")
                            != after_stream.get("tags", {}).get("language", "und")
                        ):
                            raise RuntimeError("Audio processing changed stream language or disposition")
                    output_tick = Fraction(delivered_streams[info.index]["time_base"])
                    delivered_start, delivered_count = _timeline(staged, info.index, info.sample_rate, output_tick)
                    if count != delivered_count or abs(start - delivered_start) > output_tick:
                        raise RuntimeError("Mux changed the selected audio timeline")
                    after = audio.identify_audio(staged, check_invisible=verify, stream=stream, scheme=scheme)
                    verified = audio.verification_outcome(before, after)
                    if not attempt_force and verified is not True:
                        raise audio.AudioVerificationIncomplete(before, after, audio.verification_failure_reason(after))
                    fidelity = measure_fidelity(source, info, staged, stream=stream)
                    quality_ok = audio_quality_passed(fidelity, lossy=lossy)
                    if not attempt_force and not quality_ok:
                        raise audio.AudioVerificationIncomplete(
                            before, after, "Delivered audio failed the minimum quality gate"
                        )
                    unchanged = tuple(int(s["index"]) for s in streams if s["index"] != info.index)
                    for index in unchanged:
                        _verify_packets(source, staged, index)
                        if streams[index]["codec_type"] == "audio":
                            _verify_audio_trimming(source, staged, index)
                    if sha256_file(source) != before.sha256:
                        raise RuntimeError("Video source changed during audio processing")
                return VideoAudioRemovalResult(
                    source,
                    output,
                    before,
                    replace(after, source=output),
                    count,
                    verified,
                    False,
                    fidelity,
                    unchanged,
                    start,
                    delivered_start,
                    method=processed.method,
                    bandwidth_kbps=processed.bandwidth_kbps,
                    vocoder_device=processed.vocoder_device,
                    caveats=caveats,
                    quality_passed=quality_ok,
                )
            except audio.AudioVerificationIncomplete as exc:
                last = exc
            except ImportError as exc:
                if last is not None:
                    raise audio.AudioVerificationIncomplete(
                        last.before, last.after, f"Automatic fallback unavailable: {exc}"
                    ) from exc
                raise
    if last is None:
        raise RuntimeError("No audio processing profile was attempted")
    raise last


def preflight_video_audio_workload(
    source: str | Path,
    *,
    force: bool = False,
    verify: bool = True,
    method: audio.AudioMethod = "auto",
    bandwidth_kbps: float = 24.0,
    scheme: audio.AudioScheme = "auto",
    vocoder_device: Literal["cpu", "mps", "cuda"] = "cpu",
) -> None:
    """Validate audio dependencies, layout, and bounded-track limits before visual processing."""
    source_path = Path(source).expanduser().resolve()
    profiles = audio.resolve_audio_profiles(
        method=method,
        scheme=scheme,
        verify=verify,
        force=force,
        bandwidth_kbps=bandwidth_kbps,
        vocoder_device=vocoder_device,
    )
    streams = stream_inventory(source_path)
    if [s["index"] for s in streams] != list(range(len(streams))):
        raise ValueError("Video stream indices must be contiguous")
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio_streams:
        return
    if method != "auto":
        audio.require_audio_profile_runtime(method)
    if not any("sample_rate" in s and "channels" in s for s in audio_streams):
        return
    if force:
        target_profile = profiles[0]
        audio.require_audio_profile_runtime(target_profile)
        for s in audio_streams:
            if "sample_rate" not in s or "channels" not in s:
                continue
            raw_dur = s.get("duration")
            dur = Fraction(str(raw_dur)) if raw_dur not in (None, "", "N/A") else None
            info = audio_io.AudioStreamInfo(
                index=int(s["index"]),
                sample_rate=int(s["sample_rate"]),
                channels=int(s["channels"]),
                codec=str(s.get("codec_name", "")),
                start_time=None,
                duration=dur,
                channel_layout=s.get("channel_layout"),
            )
            audio.validate_audio_stream_for_profile(info, target_profile)
        return
    for ordinal in range(len(audio_streams)):
        rep = audio.identify_audio(source_path, check_invisible=verify, stream=ordinal, scheme=scheme)
        if rep.watermark_status != "detected":
            continue
        selected = audio.adaptive_auto_profiles(scheme, rep)[0] if method == "auto" else method
        audio.require_audio_profile_runtime(selected)
        audio.validate_audio_stream_for_profile(rep.stream, selected)


def remove_video_audio_all_streams(
    source: str | Path,
    output: str | Path,
    *,
    force: bool = False,
    verify: bool = True,
    method: audio.AudioMethod = "auto",
    bandwidth_kbps: float = 24.0,
    scheme: audio.AudioScheme = "auto",
    vocoder_device: Literal["cpu", "mps", "cuda"] = "cpu",
) -> VideoAudioMultiStreamResult:
    """Inspect and clean every watermarked audio track while preserving video and clean tracks."""
    source_path, output_path = Path(source).expanduser().resolve(), Path(output).expanduser()
    if output_path.suffix.lower() not in {".mkv", ".mov", ".mp4"}:
        raise ValueError("Audio-only video output must be MKV, MOV, or MP4")
    if output_path.resolve() == source_path or (output_path.exists() and output_path.samefile(source_path)):
        raise ValueError("Video output must differ from the source")
    profiles = audio.resolve_audio_profiles(
        method=method,
        scheme=scheme,
        verify=verify,
        force=force,
        bandwidth_kbps=bandwidth_kbps,
        vocoder_device=vocoder_device,
    )
    streams = _streams(source_path)
    if [s["index"] for s in streams] != list(range(len(streams))):
        raise ValueError("Video stream indices must be contiguous")
    audio_global_indices = [int(s["index"]) for s in streams if s.get("codec_type") == "audio"]
    if not audio_global_indices:
        raise ValueError("Source has no audio stream")
    lossy = output_path.suffix.lower() == ".mp4"
    caveats = (
        "All source-positive audio tracks are processed; clean tracks and video marks remain unchanged.",
        "Metadata is not intentionally stripped; container conversion can change it. "
        "Verification covers the named decoder.",
        "Processed tracks use lossy AAC with a target bitrate of 192 kbps; "
        "verification and fidelity cover the delivered decoded audio."
        if lossy
        else "Processed tracks are stored as uncompressed float PCM, which can increase file size.",
    )
    source_sha = sha256_file(source_path)
    befores = [
        audio.identify_audio(source_path, check_invisible=verify, stream=ordinal, scheme=scheme)
        for ordinal in range(len(audio_global_indices))
    ]
    to_process = [ordinal for ordinal, rep in enumerate(befores) if force or rep.watermark_status == "detected"]
    if not to_process:
        first = profiles[0]
        skipped_items = tuple(
            VideoAudioRemovalResult(
                source_path,
                None,
                rep,
                None,
                0,
                None,
                True,
                method=first,
                bandwidth_kbps=bandwidth_kbps if first in {"encodec", "combined"} else None,
                vocoder_device=vocoder_device if first in {"bigvgan", "mimi", "focalcodec", "dac"} else None,
                caveats=caveats,
            )
            for rep in befores
        )
        return VideoAudioMultiStreamResult(
            source=source_path,
            output=None,
            streams=skipped_items,
            processed_ordinals=(),
            unchanged_streams=tuple(range(len(streams))),
            verified_removed=None,
            skipped=True,
            caveats=caveats,
            quality_passed=None,
        )

    processed_global = {audio_global_indices[o] for o in to_process}
    unchanged = tuple(int(s["index"]) for s in streams if int(s["index"]) not in processed_global)
    processed_map: dict[int, VideoAudioRemovalResult] = {}
    final_items: list[VideoAudioRemovalResult] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with staged_sibling(output_path) as staged_output:
        if len(to_process) == 1:
            sole = to_process[0]
            sole_res = remove_video_audio(
                source_path,
                staged_output,
                stream=sole,
                force=force,
                verify=verify,
                method=method,
                bandwidth_kbps=bandwidth_kbps,
                scheme=scheme,
                vocoder_device=vocoder_device,
                _before=befores[sole],
            )
            if sole_res.skipped or sole_res.output is None:
                raise RuntimeError("Selected audio stream was unexpectedly skipped during multi-stream processing")
            processed_map[sole] = sole_res
            if sha256_file(source_path) != source_sha:
                raise RuntimeError("Video source changed during multi-stream audio processing")
        else:
            with tempfile.TemporaryDirectory(prefix="raiw-video-audio-multi-") as tmp:
                current = source_path
                for step, ordinal in enumerate(to_process):
                    is_last = step == len(to_process) - 1
                    step_target = staged_output if is_last else Path(tmp) / f"step-{step}{output_path.suffix.lower()}"
                    res = remove_video_audio(
                        current,
                        step_target,
                        stream=ordinal,
                        force=force,
                        verify=verify,
                        method=method,
                        bandwidth_kbps=bandwidth_kbps,
                        scheme=scheme,
                        vocoder_device=vocoder_device,
                        _before=befores[ordinal] if step == 0 else None,
                    )
                    if res.skipped or res.output is None:
                        raise RuntimeError(
                            f"Selected audio stream {ordinal} was unexpectedly skipped during multi-stream processing"
                        )
                    processed_map[ordinal] = res
                    current = step_target
            for idx in unchanged:
                _verify_packets(source_path, staged_output, idx)
                if streams[idx]["codec_type"] == "audio":
                    _verify_audio_trimming(source_path, staged_output, idx)
            if sha256_file(source_path) != source_sha:
                raise RuntimeError("Video source changed during multi-stream audio processing")

        last_processed = to_process[-1]
        for ordinal, before_rep in enumerate(befores):
            clean_before = replace(before_rep, source=source_path)
            if ordinal not in processed_map:
                first = profiles[0]
                final_items.append(
                    VideoAudioRemovalResult(
                        source_path,
                        output_path,
                        clean_before,
                        None,
                        0,
                        None,
                        True,
                        unchanged_streams=unchanged,
                        method=first,
                        caveats=caveats,
                    )
                )
                continue
            prev = processed_map[ordinal]
            if ordinal == last_processed and len(to_process) == 1:
                final_items.append(
                    replace(
                        prev,
                        source=source_path,
                        output=output_path,
                        before=clean_before,
                        after=replace(prev.after, source=output_path) if prev.after is not None else None,
                        unchanged_streams=unchanged,
                        caveats=caveats,
                    )
                )
                continue
            after_rep = (
                prev.after
                if ordinal == last_processed and prev.after is not None
                else audio.identify_audio(staged_output, check_invisible=verify, stream=ordinal, scheme=scheme)
            )
            verified = audio.verification_outcome(before_rep, after_rep)
            if not force and verified is not True:
                raise audio.AudioVerificationIncomplete(
                    before_rep, after_rep, audio.verification_failure_reason(after_rep)
                )
            info = audio_io.probe_audio(source_path, stream=ordinal)
            fidelity = measure_fidelity(source_path, info, staged_output, stream=ordinal)
            q_ok = audio_quality_passed(fidelity, lossy=lossy)
            if not force and not q_ok:
                raise audio.AudioVerificationIncomplete(
                    before_rep, after_rep, "Delivered audio failed the minimum quality gate"
                )
            final_items.append(
                replace(
                    prev,
                    source=source_path,
                    output=output_path,
                    before=clean_before,
                    after=replace(after_rep, source=output_path),
                    verified_removed=verified,
                    fidelity=fidelity,
                    unchanged_streams=unchanged,
                    caveats=caveats,
                    quality_passed=q_ok,
                )
            )

    active = [final_items[o] for o in to_process]
    inactive = [item for o, item in enumerate(final_items) if o not in processed_map]
    if any(r.verified_removed is False for r in active) or any(
        item.before.watermark_status == "detected" for item in inactive
    ):
        agg_verified: bool | None = False
    elif all(r.verified_removed is True for r in active) and all(
        item.before.watermark_status == "not_detected" for item in inactive
    ):
        agg_verified = True
    else:
        agg_verified = None
    agg_quality = all(r.quality_passed is True for r in active) if active else None
    return VideoAudioMultiStreamResult(
        source=source_path,
        output=output_path,
        streams=tuple(final_items),
        processed_ordinals=tuple(to_process),
        unchanged_streams=unchanged,
        verified_removed=agg_verified,
        skipped=False,
        caveats=caveats,
        quality_passed=agg_quality,
    )


# Shared stream-copy evidence for pixel processing and selected-audio processing.
stream_inventory = _streams
verify_stream_packets = _verify_packets
