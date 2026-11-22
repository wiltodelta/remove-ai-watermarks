"""Audio provenance inspection and source-positive-gated spectral processing."""

from __future__ import annotations

import logging
from contextlib import closing
from dataclasses import dataclass, replace
from importlib import import_module
from math import gcd
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from remove_ai_watermarks import metadata
from remove_ai_watermarks._internal import (
    audio_codec,
    audio_dac,
    audio_focalcodec,
    audio_io,
    audio_mimi,
    audio_snac,
    audio_spectral,
    audio_vocoder,
    audioseal,
    perth,
    silentcipher,
    silentcipher16,
    wavmark,
)
from remove_ai_watermarks._internal.audio_quality import (
    AudioFidelity,
    audio_quality_passed,
    measure_fidelity,
    sha256_file,
)
from remove_ai_watermarks._internal.utils import staged_sibling

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator

    from numpy.typing import NDArray

from remove_ai_watermarks._internal.constants import (
    AUDIO_BANDWIDTHS_KBPS as AUDIO_BANDWIDTHS_KBPS,
)
from remove_ai_watermarks._internal.constants import (
    AUDIO_EXTENSIONS,
    AUDIO_LOSSY_OUTPUT_EXTENSIONS,
    AUDIO_METHODS,
    AUDIO_OUTPUT_EXTENSIONS,
    AUDIO_VENDORS,
    AUDIO_VOCODER_DEVICES,
    AudioMethod,
    AudioProfile,
    AudioScheme,
    AudioVendor,
)
from remove_ai_watermarks._internal.constants import (
    AUDIO_PROFILES as AUDIO_PROFILES,
)
from remove_ai_watermarks._internal.constants import (
    AUDIO_SCHEMES as AUDIO_SCHEMES,
)

log = logging.getLogger(__name__)
AudioWatermarkStatus = Literal["detected", "not_detected", "unverified"]
_NAMED_SCHEMES = ("audioseal", "wavmark", "perth", "silentcipher", "silentcipher16")
_AUTO_SCHEMES = ("audioseal", "wavmark", "perth")
_AUTO_PROFILES: dict[AudioScheme, tuple[AudioProfile, ...]] = {
    "audioseal": ("spectral", "combined"),
    "wavmark": ("encodec", "combined"),
    "perth": ("spectral", "combined"),
    "silentcipher": ("encodec", "combined"),
    "silentcipher16": ("encodec", "combined"),
    "all": ("combined",),
    "auto": ("spectral", "combined"),
}
AudioModelReading = (
    audioseal.AudioSealReading
    | wavmark.WavMarkReading
    | wavmark.WavMarkSignatureReading
    | perth.PerthReading
    | silentcipher.SilentCipherReading
    | silentcipher16.SilentCipher16Reading
)
_MAX_WHOLE_SECONDS = 30
_MAX_WHOLE_VALUES = 16_000_000
_AUDIOSEAL_NATIVE_SAMPLE_RATES = (24_000, 48_000)


@dataclass(frozen=True)
class AudioWindowReading:
    """A decoder reading over native, half-open sample-frame bounds."""

    start_frame: int
    end_frame: int
    reading: AudioModelReading | None
    error: str | None = None
    inapplicable: bool = False


@dataclass(frozen=True)
class AudioDetection:
    """One compatible-model check, for a channel or the mono downmix."""

    scheme: str
    channel: int | Literal["mix"]
    status: AudioWatermarkStatus
    whole_clip: AudioModelReading | None
    windows: tuple[AudioWindowReading, ...]
    whole_clip_error: str | None = None
    decoder_sample_rate: int | None = None
    whole_segments: tuple[AudioWindowReading, ...] = ()
    inapplicable: bool = False


@dataclass(frozen=True)
class AudioProvenanceReport:
    """Metadata and named decoder evidence, without a general clean verdict."""

    source: Path
    sha256: str
    stream: audio_io.AudioStreamInfo
    metadata_markers: dict[str, str]
    watermark_status: AudioWatermarkStatus
    detections: tuple[AudioDetection, ...]
    caveats: tuple[str, ...]
    recommended_recipe: str | None = None


@dataclass(frozen=True)
class AudioMetadataReport:
    """AI metadata found in one supported audio container."""

    source: Path
    has_ai_metadata: bool
    markers: dict[str, str]


@dataclass(frozen=True)
class AudioMetadataResult:
    """Result of a verified metadata strip that preserves encoded audio."""

    source: Path
    output: Path
    detected: dict[str, str]
    remaining: dict[str, str]


@dataclass(frozen=True)
class AudioRemovalResult:
    """A completed output or a no-target skip, with its verification scope."""

    source: Path
    output: Path | None
    frames_processed: int
    before: AudioProvenanceReport
    after: AudioProvenanceReport | None
    verified_removed: bool | None
    skipped: bool
    removed_markers: dict[str, str]
    method: str = "spectral"
    fidelity: AudioFidelity | None = None
    verification_scope: tuple[str, ...] = ()
    bandwidth_kbps: float | None = None
    vocoder_device: str | None = None
    quality_passed: bool | None = None


@dataclass(frozen=True)
class AudioAllResult:
    """Result of verified AI metadata removal and optional audio-watermark processing."""

    source: Path
    output: Path
    detected_metadata: dict[str, str]
    remaining_metadata: dict[str, str]
    invisible_applied: bool
    invisible_result: AudioRemovalResult | None = None


@dataclass(frozen=True)
class AudioBatchItem:
    """Outcome for one source in an audio batch."""

    source: Path
    output: Path | None
    mode: Literal["all", "metadata"]
    changed: bool
    invisible_applied: bool
    error: str | None = None


@dataclass(frozen=True)
class AudioBatchSummary:
    """Aggregate outcome for a sequential audio batch."""

    directory: Path
    output_directory: Path
    items: tuple[AudioBatchItem, ...]

    @property
    def processed(self) -> int:
        return sum(item.error is None for item in self.items)

    @property
    def failed(self) -> int:
        return sum(item.error is not None for item in self.items)

    @property
    def invisible_applied(self) -> int:
        return sum(item.invisible_applied for item in self.items)


class AudioVerificationIncomplete(RuntimeError):
    """The candidate retained a compatible mark or could not be fully checked."""

    def __init__(self, before: AudioProvenanceReport, after: AudioProvenanceReport, reason: str | None = None) -> None:
        self.before = before
        self.after = after
        message = "Audio watermark verification did not confirm removal; candidate was not published"
        super().__init__(message if reason is None else f"{message}. {reason}")


def _source(source: str | Path, stream: int) -> tuple[Path, audio_io.AudioStreamInfo]:
    path = Path(source).expanduser().resolve()
    info = audio_io.probe_audio(path, stream=stream)
    with path.open("rb") as handle:
        head = handle.read(12)
    # Apply existing structural walkers where available. Other audio formats
    # use ffprobe plus the decoder, rather than the image-format fallback.
    structured = {"wav", "mov", "mp4", "m4a", "3gp", "3g2", "mj2", "matroska", "webm", "flv", "avi"}
    if structured.intersection(info.format_name.split(",")) and not head.startswith(b"RF64"):
        if reason := metadata.unreadable_reason(path):
            raise ValueError(f"cannot read {path.name}: {reason}")
    elif head.startswith(b"FORM") and len(head) == 12 and int.from_bytes(head[4:8], "big") + 8 > path.stat().st_size:
        raise ValueError(f"cannot read {path.name}: truncated AIFF container")
    return path, info


def _status(
    whole: AudioModelReading | None,
    windows: tuple[AudioWindowReading, ...],
    whole_segments: tuple[AudioWindowReading, ...] = (),
) -> AudioWatermarkStatus:
    if (
        (whole is not None and whole.detected)
        or any(s.reading is not None and s.reading.detected for s in whole_segments)
        or any(w.reading is not None and w.reading.detected for w in windows)
    ):
        return "detected"
    whole_verified = (
        bool(whole_segments) and all(s.reading is not None for s in whole_segments)
        if whole_segments
        else whole is not None
    )
    if whole_verified and windows and all(w.reading is not None for w in windows):
        return "not_detected"
    return "unverified"


def _aggregate(detections: tuple[AudioDetection, ...]) -> AudioWatermarkStatus:
    if any(d.status == "detected" for d in detections):
        return "detected"
    if detections and all(d.status == "not_detected" for d in detections):
        return "not_detected"
    return "unverified"


def _model_samples(samples: NDArray[Any], sample_rate: int) -> NDArray[Any]:
    import numpy as np

    if sample_rate == audioseal.SAMPLE_RATE:
        return np.ascontiguousarray(samples, dtype=np.float32)
    factor = gcd(sample_rate, audioseal.SAMPLE_RATE)
    signal = import_module("scipy.signal")
    return np.asarray(
        signal.resample_poly(samples, audioseal.SAMPLE_RATE // factor, sample_rate // factor), dtype=np.float32
    )


def _read_model(
    model: Any, samples: NDArray[Any], rate: int, scheme: str, *, native_audioseal: bool = False
) -> tuple[AudioModelReading | None, str | None, bool]:
    if scheme in {"silentcipher", "silentcipher16"}:
        backend = silentcipher16 if scheme == "silentcipher16" else silentcipher
        try:
            return backend.read(model, samples, sample_rate=rate), None, False
        except (RuntimeError, ValueError) as exc:
            log.warning(
                "SilentCipher check failed: scheme=%s sample_rate=%s sample_count=%s error=%s",
                scheme,
                rate,
                len(samples),
                exc,
            )
            return None, str(exc), False
    # AudioSeal 0.2 does not resample internally. Some encoders, including the
    # published LTX marker, stamp at the native sample clock instead of 16 kHz.
    prepared = samples if native_audioseal else _model_samples(samples, rate)
    if scheme in {"wavmark", "perth"} and len(prepared) < 16000:
        return None, "Audio is shorter than one complete model window", True
    try:
        decoder = {"audioseal": audioseal.read, "wavmark": wavmark.read, "perth": perth.read}[scheme]
        return decoder(model, prepared), None, False
    except RuntimeError as exc:
        log.warning(
            "Audio decoder check failed: scheme=%s sample_rate=%s sample_count=%s error=%s",
            scheme,
            rate,
            len(samples),
            exc,
        )
        return None, str(exc), False


def _scan_audio(
    source: Path, info: audio_io.AudioStreamInfo, scheme: str
) -> tuple[tuple[AudioDetection, ...], tuple[str, ...]]:
    checks, notes = _scan_audio_at_clock(source, info, scheme)
    if scheme == "audioseal" and info.sample_rate in _AUDIOSEAL_NATIVE_SAMPLE_RATES:
        native_checks, native_notes = _scan_audio_at_clock(source, info, scheme, native_audioseal=True)
        checks += native_checks
        notes = tuple(dict.fromkeys((*notes, *native_notes)))
    if scheme == "wavmark":
        signature_checks, signature_notes = _scan_openvoice_default(source, info)
        checks += signature_checks
        notes += signature_notes
    return checks, notes


def _scan_openvoice_default(
    source: Path, info: audio_io.AudioStreamInfo
) -> tuple[tuple[AudioDetection, ...], tuple[str, ...]]:
    import numpy as np

    model = wavmark.load_model()
    reader = audio_io.read_audio(source, info, chunk_frames=wavmark.OPENVOICE_PREFIX_SAMPLES)
    with closing(reader):
        frames = next(reader, np.empty((0, info.channels), dtype=np.float32))
    channels: list[int | Literal["mix"]] = list(range(info.channels))
    if info.channels > 1:
        channels.append("mix")
    checks: list[AudioDetection] = []
    for channel in channels:
        mono = frames.mean(axis=1) if channel == "mix" else frames[:, channel]
        if len(mono) < wavmark.MIN_READ_SAMPLES:
            checks.append(
                AudioDetection(
                    "wavmark",
                    channel,
                    "unverified",
                    None,
                    (),
                    "Audio is shorter than one complete OpenVoice signature window",
                    info.sample_rate,
                    inapplicable=True,
                )
            )
            continue
        try:
            whole, fragments = wavmark.read_openvoice_default(model, mono)
        except (RuntimeError, ValueError) as exc:
            log.warning(
                "OpenVoice signature check failed: channel=%s sample_rate=%s sample_count=%s error=%s",
                channel,
                info.sample_rate,
                len(mono),
                exc,
            )
            checks.append(AudioDetection("wavmark", channel, "unverified", None, (), str(exc), info.sample_rate))
            continue
        windows = tuple(
            AudioWindowReading(item.start_samples[0], item.start_samples[0] + wavmark.MIN_READ_SAMPLES, item)
            for item in fragments
        )
        checks.append(
            AudioDetection(
                "wavmark", channel, _status(whole, windows), whole, windows, decoder_sample_rate=info.sample_rate
            )
        )
    notes = (
        "WavMark also checks the independently known OpenVoice default signature at its native sample offsets. "
        "Custom messages and shifted/resampled framing are outside this receiver. "
        "Provider identity and complete erasure remain unverified.",
        "Signature binomial tails are correlation diagnostics, not calibrated false-positive probabilities.",
    )
    return tuple(checks), notes


def _scan_audio_at_clock(
    source: Path, info: audio_io.AudioStreamInfo, scheme: str, *, native_audioseal: bool = False
) -> tuple[tuple[AudioDetection, ...], tuple[str, ...]]:
    import numpy as np

    loader = {
        "audioseal": audioseal.load_pinned_detector,
        "wavmark": wavmark.load_model,
        "perth": perth.load_model,
        "silentcipher": silentcipher.load_model,
        "silentcipher16": silentcipher16.load_model,
    }[scheme]
    model = loader()
    channels: list[int | Literal["mix"]] = list(range(info.channels))
    if info.channels > 1:
        channels.append("mix")
    windows: dict[int | Literal["mix"], list[AudioWindowReading]] = {channel: [] for channel in channels}
    segments: dict[int | Literal["mix"], list[AudioWindowReading]] = {channel: [] for channel in channels}
    count = 0
    previous = np.empty((0, info.channels), dtype=np.float32)
    window_seconds = {"silentcipher": silentcipher.WINDOW_SECONDS, "silentcipher16": silentcipher16.WINDOW_SECONDS}
    window_frames = info.sample_rate * window_seconds.get(scheme, 1)
    max_segment_frames = min(
        _MAX_WHOLE_SECONDS * info.sample_rate,
        max(window_frames, _MAX_WHOLE_VALUES // max(1, info.channels)),
    )
    segment_chunks: list[NDArray[Any]] = []
    segment_start = 0
    segment_len = 0

    def flush_segment(block: NDArray[Any], start_frame: int, end_frame: int) -> None:
        for channel in channels:
            mono = block.mean(axis=1) if channel == "mix" else block[:, channel]
            seg_reading, seg_error, seg_inapplicable = _read_model(
                model, mono, info.sample_rate, scheme, native_audioseal=native_audioseal
            )
            segments[channel].append(
                AudioWindowReading(start_frame, end_frame, seg_reading, seg_error, inapplicable=seg_inapplicable)
            )

    for frames in audio_io.read_audio(source, info, chunk_frames=window_frames):
        if not np.isfinite(frames).all():
            raise ValueError("Audio input contains nonfinite samples")
        window, start = frames, count
        if len(frames) < window_frames and len(previous):
            window = np.concatenate((previous[-(window_frames - len(frames)) :], frames))
            start = count + len(frames) - len(window)
        for channel in channels:
            mono = window.mean(axis=1) if channel == "mix" else window[:, channel]
            reading, error, win_inapplicable = _read_model(
                model, mono, info.sample_rate, scheme, native_audioseal=native_audioseal
            )
            windows[channel].append(
                AudioWindowReading(start, count + len(frames), reading, error, inapplicable=win_inapplicable)
            )
        previous = frames
        count += len(frames)

        pending = frames
        while len(pending):
            if segment_len >= max_segment_frames:
                joined_seg = np.concatenate(segment_chunks)
                flush_segment(joined_seg, segment_start, segment_start + len(joined_seg))
                overlap = joined_seg[-min(window_frames, max(0, max_segment_frames - 1), len(joined_seg)) :]
                segment_start = segment_start + len(joined_seg) - len(overlap)
                segment_chunks = [overlap] if len(overlap) else []
                segment_len = len(overlap)
            space = max_segment_frames - segment_len
            take = pending[:space]
            segment_chunks.append(take)
            segment_len += len(take)
            pending = pending[space:]

    if segment_len > 0:
        joined_seg = np.concatenate(segment_chunks)
        flush_segment(joined_seg, segment_start, segment_start + len(joined_seg))

    multi_segment = False
    detections: list[AudioDetection] = []
    for channel in channels:
        seg_tuple = tuple(segments[channel])
        if len(seg_tuple) > 1:
            multi_segment = True
        reading = next(
            (item.reading for item in seg_tuple if item.reading is not None and item.reading.detected),
            seg_tuple[0].reading if seg_tuple else None,
        )
        error = next((item.error for item in seg_tuple if item.error is not None), None)
        intervals = tuple(windows[channel])
        decoder_rate = None
        if scheme == "audioseal":
            decoder_rate = info.sample_rate if native_audioseal else audioseal.SAMPLE_RATE
        elif scheme == "silentcipher":
            decoder_rate = silentcipher.SAMPLE_RATE
        elif scheme == "silentcipher16":
            decoder_rate = silentcipher16.SAMPLE_RATE
        detections.append(
            AudioDetection(
                scheme,
                channel,
                _status(reading, intervals, seg_tuple),
                reading,
                intervals,
                error,
                decoder_rate,
                seg_tuple,
                inapplicable=bool(seg_tuple or intervals)
                and all(s.inapplicable for s in seg_tuple)
                and all(w.inapplicable for w in intervals),
            )
        )
    caveats = [
        f"{scheme} checks cover its pinned public model. "
        "Production-provider compatibility and general AI origin remain unverified."
    ]
    if scheme == "audioseal":
        caveats.append(
            "AudioSeal adds native-clock checks for calibrated 24 kHz and 48 kHz profiles alongside 16 kHz checks; "
            "other native clocks are outside this coverage. Each channel and clock is independent. "
            "A matching public-model mark does not identify LTX."
        )
    elif scheme == "silentcipher":
        caveats.append(
            "SilentCipher checks use the pinned 44.1 kHz model, including for 16 kHz input audio. "
            "This receiver excludes the 16 kHz model. Separate silentcipher16 checks cover only an experimental "
            "terminated eight-symbol pattern, not the incompatible SDK byte API or general 32-bit framing."
        )
    elif scheme == "silentcipher16":
        caveats.append(
            "SilentCipher-16k checks cover only an experimental terminated eight-symbol pattern. "
            "payload_symbols contains seven ternary symbols, not bytes. The SDK byte API and general 32-bit "
            "profile remain unqualified. Known-message recovery failed on exploratory tone and noise carriers. "
            "A negative does not rule out other framing or certify erasure of every residual payload bit."
        )
    if multi_segment:
        caveats.append("Long-clip decoding used bounded overlapping segments alongside all sample windows.")
    if any(
        (not d.inapplicable) and (d.whole_clip_error or any(w.error and not w.inapplicable for w in d.windows))
        for d in detections
    ):
        caveats.append("Some decoder checks failed; those checks are unverified, not negative.")
    return tuple(detections), tuple(caveats)


def _audio_metadata_vendor(markers: dict[str, str]) -> Literal["elevenlabs", "google_synthid"] | None:
    """Identify proprietary AI-audio waveform vendors from pre-strip metadata."""
    if not markers:
        return None
    from remove_ai_watermarks._internal.c2pa import markers_claim_ai
    from remove_ai_watermarks._internal.constants import SYNTHID_EDIT_SIGNERS

    identity_text = " ".join(
        (
            markers.get("claim_generator", ""),
            markers.get("issuer", ""),
            markers.get("ai_tool", ""),
        )
    ).casefold()
    if "elevenlabs" in identity_text or "eleven labs" in identity_text:
        return "elevenlabs"
    synthid_marker = markers.get("synthid_watermark", "").casefold()
    soft_binding_text = f"{markers.get('soft_binding_algorithm', '')} {markers.get('soft_binding', '')}".casefold()
    if "google" in synthid_marker or ("google" in soft_binding_text and "synthid" in soft_binding_text):
        return "google_synthid"
    has_google_identity = "google" in identity_text and not any(
        signer.decode("utf-8", errors="ignore").casefold() in identity_text for signer in SYNTHID_EDIT_SIGNERS
    )
    if has_google_identity and ("synthid_watermark" in markers or markers_claim_ai(markers)):
        return "google_synthid"
    return None


def _recommend_audio_recipe(
    source: Path,
    *,
    info: audio_io.AudioStreamInfo,
    markers: dict[str, str],
    watermark_status: AudioWatermarkStatus,
) -> str | None:
    """Suggest an explicit read-only audio cleaning command without claiming hidden marks exist."""
    from remove_ai_watermarks._internal.c2pa import markers_claim_ai

    in_ext = source.suffix.lower() or ".wav"
    out_ext = in_ext if in_ext in AUDIO_OUTPUT_EXTENSIONS else ".wav"
    vendor = _audio_metadata_vendor(markers)
    if vendor == "elevenlabs" or (vendor == "google_synthid" and _is_mono_or_stereo(info)):
        method = "auto" if vendor == "elevenlabs" else "dac"
        perm = "" if watermark_status == "detected" else " --allow-unverified"
        return f"remove-ai-watermarks audio all input{in_ext} -o clean{out_ext} --invisible --method {method}{perm}"
    if watermark_status == "detected":
        return f"remove-ai-watermarks audio all input{in_ext} -o clean{out_ext} --invisible"
    if markers_claim_ai(markers):
        return f"remove-ai-watermarks audio metadata input{in_ext} --remove -o clean{in_ext}"
    return None


def identify_audio(
    source: str | Path,
    *,
    check_invisible: bool = True,
    stream: int = 0,
    scheme: AudioScheme = "audioseal",
    _metadata_markers: dict[str, str] | None = None,
) -> AudioProvenanceReport:
    """Inspect one audio track; missing decoders remain explicitly unverified.

    Track selection uses a zero-based audio ordinal. Channel checks and, for
    multichannel sources, a mono downmix are separate. Whole-clip decoding is
    bounded; skipping it never becomes a complete negative verdict. ``scheme="auto"``
    scans every installed compatible decoder without requiring absent optional extras.
    """
    if scheme not in _AUTO_PROFILES:
        raise ValueError("Audio scheme must be audioseal, wavmark, perth, silentcipher, silentcipher16, all, or auto")
    path, info = _source(source, stream)
    digest = sha256_file(path)
    markers = dict(_metadata_markers) if _metadata_markers is not None else metadata.get_ai_metadata(path)
    detections: tuple[AudioDetection, ...] = ()
    caveats: list[str] = []
    availability = {
        "audioseal": audioseal.is_available,
        "wavmark": wavmark.is_available,
        "perth": perth.is_available,
        "silentcipher": silentcipher.is_available,
        "silentcipher16": silentcipher16.is_available,
    }
    extras = {
        "audioseal": "audio-detect",
        "wavmark": "audio-wavmark",
        "perth": "audio-perth",
        "silentcipher": "audio-silentcipher",
        "silentcipher16": "audio-silentcipher",
    }
    if not check_invisible:
        caveats.append("Audio watermark decoding was skipped.")
    elif not audio_io.is_available() or not audio_spectral.is_available():
        caveats.append("Audio sample processing is unavailable; install remove-ai-watermarks[audio] and ffmpeg.")
    else:
        if scheme == "all":
            requested: tuple[str, ...] = _NAMED_SCHEMES
        elif scheme == "auto":
            requested = _AUTO_SCHEMES
        else:
            requested = (scheme,)
        found: list[AudioDetection] = []
        skipped_auto: list[str] = []
        for name in requested:
            if not availability[name]():
                if scheme == "auto":
                    skipped_auto.append(name)
                else:
                    caveats.append(f"{name} decoding is unavailable; install remove-ai-watermarks[{extras[name]}].")
                    if len(requested) > 1:
                        found.append(AudioDetection(name, 0, "unverified", None, ()))
                continue
            checks, notes = _scan_audio(path, info, name)
            found.extend(checks)
            caveats.extend(notes)
        if scheme == "auto":
            if len(skipped_auto) == len(requested):
                caveats.append("audioseal decoding is unavailable; install remove-ai-watermarks[audio-detect].")
            elif skipped_auto:
                caveats.append(
                    f"Auto audio inspection skipped uninstalled optional decoders: {', '.join(skipped_auto)}."
                )
            if any(not d.inapplicable for d in found):
                if any(d.inapplicable for d in found):
                    caveats.append(
                        "Auto audio inspection skipped checks whose minimum window (16000 samples / one second) "
                        "exceeds the clip duration."
                    )
                found = [d for d in found if not d.inapplicable]
        detections = tuple(found)
    if sha256_file(path) != digest:
        raise RuntimeError("Audio source changed during inspection")
    status = _aggregate(detections)
    recipe = _recommend_audio_recipe(path, info=info, markers=markers, watermark_status=status)
    return AudioProvenanceReport(path, digest, info, markers, status, detections, tuple(caveats), recipe)


def _missing_decoder_reason(report: AudioProvenanceReport) -> str | None:
    missing = [caveat for caveat in report.caveats if "decoding is unavailable; install " in caveat]
    return "; ".join(missing) if missing else None


def verification_failure_reason(report: AudioProvenanceReport) -> str | None:
    """Return a concise diagnostic when post-processing audio verification is incomplete."""
    if missing := _missing_decoder_reason(report):
        return missing
    if any(
        (not d.inapplicable)
        and (
            d.whole_clip_error
            or any(s.error and not s.inapplicable for s in d.whole_segments)
            or any(w.error and not w.inapplicable for w in d.windows)
        )
        for d in report.detections
    ):
        return "One or more audio decoder checks failed during output verification."
    return None


def verification_outcome(before: AudioProvenanceReport, after: AudioProvenanceReport) -> bool | None:
    """Return True only when every source-positive check has complete negative output coverage."""
    if after.watermark_status == "detected" or any(d.status == "detected" for d in after.detections):
        return False
    if before.watermark_status != "detected" or after.watermark_status != "not_detected":
        return None
    applicable_after = [d for d in after.detections if not d.inapplicable]
    if not applicable_after or any(d.status != "not_detected" for d in applicable_after):
        return None
    after_by_key: dict[tuple[str, int | Literal["mix"], int | None], list[AudioDetection]] = {}
    for det in applicable_after:
        after_by_key.setdefault((det.scheme, det.channel, det.decoder_sample_rate), []).append(det)
    source_positives = [d for d in before.detections if d.status == "detected"]
    if not source_positives:
        return None
    for pos in source_positives:
        matched = after_by_key.get((pos.scheme, pos.channel, pos.decoder_sample_rate))
        if not matched or any(item.status != "not_detected" for item in matched):
            return None
    return True


def _require_base_audio_runtime(method: str) -> None:
    if not audio_io.is_available() or not audio_spectral.is_available():
        extra = (
            "audio-vocoder"
            if method == "bigvgan"
            else "audio-codec"
            if method in {"mimi", "focalcodec", "dac", "snac"}
            else "audio"
        )
        raise ImportError(f"Audio processing requires 'remove-ai-watermarks[{extra}]' and ffmpeg")


def require_audio_profile_runtime(method: AudioProfile) -> None:
    """Ensure runtime dependencies for a concrete audio processing profile are installed."""
    if method in {"encodec", "combined"} and not audio_codec.is_available():
        raise ImportError("Neural audio processing requires remove-ai-watermarks[audio-codec]")
    if method == "mimi" and not audio_mimi.is_available():
        raise ImportError("Mimi processing requires remove-ai-watermarks[audio-codec]")
    if method == "focalcodec" and not audio_focalcodec.is_available():
        raise ImportError("FocalCodec processing requires remove-ai-watermarks[audio-codec]")
    if method == "dac" and not audio_dac.is_available():
        raise ImportError("DAC processing requires remove-ai-watermarks[audio-codec]")
    if method == "snac" and not audio_snac.is_available():
        raise ImportError("SNAC processing requires remove-ai-watermarks[audio-codec]")
    if method == "bigvgan" and not audio_vocoder.is_available():
        raise ImportError("BigVGAN processing requires remove-ai-watermarks[audio-vocoder]")
    _require_base_audio_runtime(method)


_SURROUND_51_LAYOUTS: frozenset[str] = frozenset({"5.1", "5.1(side)"})
_SURROUND_71_LAYOUTS: frozenset[str] = frozenset({"7.1", "7.1(wide)", "7.1(wide-side)"})
_RAW_CODEC_PROFILES: frozenset[str] = frozenset({"bigvgan", "mimi", "focalcodec", "dac", "snac"})


def _is_mono_or_stereo(info: audio_io.AudioStreamInfo) -> bool:
    return info.channel_layout in {"mono", "stereo"} and info.channels in (1, 2)


def _is_tagged_surround(info: audio_io.AudioStreamInfo) -> bool:
    return (info.channels == 6 and info.channel_layout in _SURROUND_51_LAYOUTS) or (
        info.channels == 8 and info.channel_layout in _SURROUND_71_LAYOUTS
    )


def validate_audio_stream_for_profile(info: audio_io.AudioStreamInfo, method: AudioProfile) -> None:
    """Validate channel layout and bounded full-track limits for a selected audio profile."""
    if method in {"dac", "snac"}:
        label = "DAC" if method == "dac" else "SNAC"
        if not (_is_mono_or_stereo(info) or _is_tagged_surround(info)):
            raise ValueError(f"{label} requires a mono or stereo (or tagged 5.1/7.1) channel layout")
    elif method in {"bigvgan", "mimi", "focalcodec"} and not _is_mono_or_stereo(info):
        label = {"bigvgan": "BigVGAN", "mimi": "Mimi", "focalcodec": "FocalCodec"}[method]
        raise ValueError(f"{label} requires a mono or stereo channel layout")
    if method in {"bigvgan", "mimi"} and info.duration is not None:
        label = "BigVGAN" if method == "bigvgan" else "Mimi"
        est_frames = int(info.duration * info.sample_rate)
        if est_frames > _MAX_WHOLE_SECONDS * info.sample_rate or est_frames * info.channels > _MAX_WHOLE_VALUES:
            raise ValueError(f"{label} full-track processing is limited to 30 seconds and 16000000 sample values")


def normalize_audio_vendor(vendor: str | None) -> str | None:
    if vendor in (None, "auto"):
        return None
    if vendor not in AUDIO_VENDORS:
        expected = ", ".join(AUDIO_VENDORS)
        raise ValueError(f"Audio vendor must be one of: {expected}")
    return vendor


_normalize_audio_vendor = normalize_audio_vendor


def adaptive_auto_profiles(
    scheme: AudioScheme,
    before: AudioProvenanceReport,
    *,
    vendor: str | None = None,
) -> tuple[AudioProfile, ...]:
    """Select minimal-distortion candidate profiles matched to detected schemes and vendor cohorts."""
    explicit_vendor = _normalize_audio_vendor(vendor)
    resolved_vendor = explicit_vendor or _audio_metadata_vendor(before.metadata_markers)
    if resolved_vendor is not None:
        can_use_dac = (
            _is_mono_or_stereo(before.stream) or _is_tagged_surround(before.stream)
        ) and audio_dac.is_available()
        if resolved_vendor in {"google", "google_synthid"}:
            return ("dac", "combined") if can_use_dac else ("dac",)
        if resolved_vendor in {"elevenlabs", "suno", "udio", "openai", "cartesia", "microsoft"}:
            return ("dac", "encodec", "combined") if can_use_dac else ("encodec", "combined")
        if resolved_vendor == "meta":
            return ("spectral", "dac", "combined") if can_use_dac else ("spectral", "combined")
        if resolved_vendor == "resemble":
            return ("dac", "spectral", "combined") if can_use_dac else ("spectral", "combined")
    if scheme not in {"all", "auto"} or before.watermark_status != "detected":
        return _AUTO_PROFILES[scheme]
    detected = {item.scheme for item in before.detections if item.status == "detected"}
    if detected and detected <= {"audioseal", "perth"}:
        return ("spectral", "combined")
    if detected and detected <= {"wavmark", "silentcipher", "silentcipher16"}:
        return ("encodec", "combined")
    return ("combined",)


def enforce_audio_publication_gates(
    before: AudioProvenanceReport,
    after: AudioProvenanceReport,
    *,
    verified: bool | None,
    quality_ok: bool,
    force: bool = False,
    allow_unverified: bool = False,
    allow_positive: bool = False,
    allow_quality_override: bool = False,
) -> None:
    """Enforce post-processing decoder verification and model-free audio integrity gates."""
    if (verified is False and not (force or allow_positive)) or (verified is None and not (force or allow_unverified)):
        raise AudioVerificationIncomplete(before, after, verification_failure_reason(after))
    if not (force or allow_quality_override) and not quality_ok:
        raise AudioVerificationIncomplete(before, after, "Delivered audio failed the minimum quality gate")


def validate_audio_arguments(
    *,
    method: AudioMethod,
    scheme: AudioScheme,
    verify: bool,
    force: bool,
    bandwidth_kbps: float,
    vocoder_device: Literal["cpu", "mps", "cuda"],
    allow_unverified: bool = False,
    vendor: str | None = None,
) -> tuple[AudioProfile, ...]:
    """Validate public audio processing arguments without requiring audio runtime dependencies."""
    explicit_vendor = _normalize_audio_vendor(vendor)
    if not verify and not (force or allow_unverified or explicit_vendor is not None):
        raise ValueError("Disabling verification requires force=True or allow_unverified=True")
    if method not in AUDIO_METHODS:
        raise ValueError(
            "Audio method must be auto, spectral, encodec, combined, bigvgan, mimi, focalcodec, dac, or snac"
        )
    if vocoder_device not in AUDIO_VOCODER_DEVICES:
        raise ValueError("Vocoder device must be cpu, mps, or cuda")
    if vocoder_device != "cpu" and method in {"spectral", "mimi", "focalcodec"}:
        raise ValueError("A non-CPU vocoder device requires method=bigvgan, dac, snac, encodec, combined, or auto")
    if scheme not in _AUTO_PROFILES:
        raise ValueError("Audio scheme must be audioseal, wavmark, perth, silentcipher, silentcipher16, all, or auto")
    if method in {"auto", "encodec", "combined"} and bandwidth_kbps not in AUDIO_BANDWIDTHS_KBPS:
        raise ValueError("EnCodec bandwidth must be 6, 12, or 24 kbps")
    return _AUTO_PROFILES[scheme] if method == "auto" else (method,)


def resolve_audio_profiles(
    *,
    method: AudioMethod,
    scheme: AudioScheme,
    verify: bool,
    force: bool,
    bandwidth_kbps: float,
    vocoder_device: Literal["cpu", "mps", "cuda"],
    allow_unverified: bool = False,
    vendor: str | None = None,
) -> tuple[AudioProfile, ...]:
    """Validate public audio processing arguments and return candidate profiles."""
    profiles = validate_audio_arguments(
        method=method,
        scheme=scheme,
        verify=verify,
        force=force,
        bandwidth_kbps=bandwidth_kbps,
        vocoder_device=vocoder_device,
        allow_unverified=allow_unverified,
        vendor=vendor,
    )
    _require_base_audio_runtime(method)
    return profiles


def remove_audio_invisible(
    source: str | Path,
    output: str | Path,
    *,
    force: bool = False,
    verify: bool = True,
    stream: int = 0,
    method: AudioMethod = "auto",
    bandwidth_kbps: float = 24.0,
    scheme: AudioScheme = "audioseal",
    vocoder_device: Literal["cpu", "mps", "cuda"] = "cpu",
    allow_unverified: bool = False,
    allow_positive: bool = False,
    allow_quality_override: bool = False,
    vendor: AudioVendor | None = None,
    preserve_hf_hz: float | None = None,
) -> AudioRemovalResult:
    """Process audio and publish only a fully checked negative by default.

    Auto chooses a model-specific first profile and falls back to combined
    processing only after local verification fails. Each attempt uses the same
    hash-bound source report. ``allow_unverified=True`` authorizes processing
    without a local source-positive decoder and publishing an unverified output
    while keeping positive-detection and quality gates active; ``allow_positive=True``
    permits a still-positive output; ``allow_quality_override=True`` overrides a
    failed integrity gate; ``force=True`` remains an alias enabling all three.
    Output files (.wav, .flac, .mp3, .m4a, .ogg, .opus) discard metadata and
    preserve channel geometry.
    Explicit bigvgan, mimi, and focalcodec are experimental bounded full-track
    raw M/S profiles, excluded from auto; dac is selected by auto only when
    pre-strip metadata identifies ElevenLabs or Google SynthID. None supplies a
    SynthID verifier. Mimi, FocalCodec, and DAC use CPU float32 (fixed
    32-quantizer Mimi; conditioned 50 Hz WavLM-to-Vocos FocalCodec; 3-codebook
    44.1 kHz DAC); BigVGAN retains its selected device. Codec reconstruction
    does not guarantee voice identity.
    """
    explicit_vendor = _normalize_audio_vendor(vendor)
    eff_allow_unverified = allow_unverified or explicit_vendor is not None
    profiles = resolve_audio_profiles(
        method=method,
        scheme=scheme,
        verify=verify,
        force=force,
        bandwidth_kbps=bandwidth_kbps,
        vocoder_device=vocoder_device,
        allow_unverified=eff_allow_unverified,
        vendor=vendor,
    )
    target, original = Path(output).expanduser(), Path(source).expanduser().resolve()
    if target.resolve() == original or (target.exists() and target.samefile(original)):
        raise ValueError("Audio output must differ from the source")
    before = identify_audio(source, check_invisible=verify, stream=stream, scheme=scheme)
    if method == "auto":
        profiles = adaptive_auto_profiles(scheme, before, vendor=explicit_vendor)
    eff_unverified = force or eff_allow_unverified
    if not (eff_unverified or allow_positive) and before.watermark_status != "detected":
        profile = profiles[0]
        return AudioRemovalResult(
            before.source,
            None,
            0,
            before,
            None,
            None,
            True,
            {},
            method=profile,
            bandwidth_kbps=bandwidth_kbps if profile in {"encodec", "combined"} else None,
            vocoder_device=vocoder_device if profile != "spectral" else None,
        )
    if not verify or before.watermark_status != "detected" or (not eff_unverified and _missing_decoder_reason(before)):
        profiles = profiles[:1]
    last: AudioVerificationIncomplete | None = None
    for index, profile in enumerate(profiles):
        is_last = index == len(profiles) - 1
        try:
            return _remove_audio_profile(
                source,
                output,
                force=force and is_last,
                verify=verify,
                stream=stream,
                method=profile,
                bandwidth_kbps=bandwidth_kbps,
                scheme=scheme,
                _before=before,
                vocoder_device=vocoder_device,
                allow_unverified=eff_allow_unverified and is_last,
                allow_positive=allow_positive and is_last,
                allow_quality_override=allow_quality_override and is_last,
                preserve_hf_hz=preserve_hf_hz,
            )
        except AudioVerificationIncomplete as exc:
            last = exc
        except ImportError as exc:
            if last is not None:
                raise AudioVerificationIncomplete(
                    last.before, last.after, f"Automatic fallback unavailable: {exc}"
                ) from exc
            raise
    if last is None:
        raise RuntimeError("No audio processing profile was attempted")
    raise last


def _remove_audio_profile(
    source: str | Path,
    output: str | Path,
    *,
    force: bool = False,
    verify: bool = True,
    stream: int = 0,
    method: AudioProfile = "spectral",
    bandwidth_kbps: float = 24.0,
    scheme: AudioScheme = "audioseal",
    _before: AudioProvenanceReport | None = None,
    _skip_post_verify: bool = False,
    vocoder_device: Literal["cpu", "mps", "cuda"] = "cpu",
    allow_unverified: bool = False,
    allow_positive: bool = False,
    allow_quality_override: bool = False,
    preserve_hf_hz: float | None = None,
) -> AudioRemovalResult:
    """Apply the selected audio profile and verify the saved candidate.

    The default requires a compatible source-positive decoder and publishes
    only a fully negative result under that decoder. ``force=True`` permits
    an unverified or still-positive candidate; the result never labels that
    outcome as verified removal. ``verify=False`` requires explicit force or
    ``allow_unverified=True``.
    Output is WAV, FLAC, MP3, M4A, OGG, or Opus, with metadata discarded during
    fresh encoding.
    """
    eff_unverified = force or allow_unverified
    eff_positive = force or allow_positive
    if not verify and not eff_unverified:
        raise ValueError("Disabling verification requires force=True or allow_unverified=True")
    if method not in AUDIO_PROFILES:
        raise ValueError("Audio method must be spectral, encodec, combined, bigvgan, mimi, focalcodec, dac, or snac")
    neural = method in {"encodec", "combined"}
    if neural and bandwidth_kbps not in AUDIO_BANDWIDTHS_KBPS:
        raise ValueError("EnCodec bandwidth must be 6, 12, or 24 kbps")
    require_audio_profile_runtime(method)
    target = Path(output).expanduser()
    original = Path(source).expanduser().resolve()
    if target.resolve() == original or (target.exists() and target.samefile(original)):
        raise ValueError("Audio output must differ from the source")
    before = (
        _before if _before is not None else identify_audio(source, check_invisible=verify, stream=stream, scheme=scheme)
    )
    validate_audio_stream_for_profile(before.stream, method)
    if not (eff_unverified or eff_positive) and before.watermark_status != "detected":
        return AudioRemovalResult(
            before.source,
            None,
            0,
            before,
            None,
            None,
            True,
            {},
            method=method,
            bandwidth_kbps=bandwidth_kbps if neural else None,
            vocoder_device=vocoder_device if method in _RAW_CODEC_PROFILES else None,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    source_exceeded_full_scale = False

    def inputs() -> Generator[NDArray[Any], None, None]:
        nonlocal count, source_exceeded_full_scale
        import numpy as np

        reader = audio_io.read_audio(before.source, before.stream, chunk_frames=before.stream.sample_rate * 8)
        try:
            for frames in reader:
                count += len(frames)
                if not source_exceeded_full_scale and frames.size and float(np.max(np.abs(frames))) > 1.0:
                    source_exceeded_full_scale = True
                yield frames
        finally:
            close = getattr(reader, "close", None)
            if callable(close):
                close()

    def _guard_full_scale(chunks: Iterator[NDArray[Any]]) -> Generator[NDArray[Any], None, None]:
        import numpy as np

        for chunk in chunks:
            arr = np.asarray(chunk, dtype=np.float32)
            if not source_exceeded_full_scale and arr.size:
                peak = float(np.max(np.abs(arr)))
                if bool(np.isfinite(peak)) and 0.9999 < peak <= 1.35:
                    arr = (arr * (0.9999 / peak)).astype(np.float32, copy=False)
            yield arr

    with staged_sibling(target) as staged:
        transformed: Iterator[NDArray[Any]] = inputs()
        if method in {"spectral", "combined"}:
            transformed = audio_spectral.filter_audio_chunks(transformed, before.stream.sample_rate)
        if neural:
            if _is_tagged_surround(before.stream):
                transformed = _encodec_surround_track(
                    transformed,
                    before.stream.sample_rate,
                    bandwidth_kbps=bandwidth_kbps,
                    device=vocoder_device,
                )
            else:
                stereo_mode: Literal["independent", "ms"] = "ms" if before.stream.channels == 2 else "independent"
                if vocoder_device == "cpu":
                    transformed = audio_codec.resynthesize_chunks(
                        transformed,
                        before.stream.sample_rate,
                        bandwidth_kbps=bandwidth_kbps,
                        stereo_mode=stereo_mode,
                    )
                else:
                    transformed = audio_codec.resynthesize_chunks(
                        transformed,
                        before.stream.sample_rate,
                        bandwidth_kbps=bandwidth_kbps,
                        stereo_mode=stereo_mode,
                        device=vocoder_device,
                    )
        if method == "bigvgan":
            transformed = _vocode_full_track(transformed, before.stream.sample_rate, vocoder_device)
        if method == "mimi":
            transformed = _mimi_full_track(transformed, before.stream.sample_rate)
        if method == "focalcodec":
            transformed = _focalcodec_full_track(transformed, before.stream.sample_rate)
        if method == "dac":
            transformed = _dac_full_track(
                transformed, before.stream.sample_rate, vocoder_device, preserve_hf_hz=preserve_hf_hz
            )
        if method == "snac":
            transformed = _snac_full_track(
                transformed, before.stream.sample_rate, vocoder_device, preserve_hf_hz=preserve_hf_hz
            )
        frames_written = audio_io.write_audio(
            before.stream,
            staged,
            _guard_full_scale(transformed),
            bitexact=method in _RAW_CODEC_PROFILES and staged.suffix.lower() == ".wav",
        )
        if frames_written != count:
            raise RuntimeError("Audio processing changed the decoded sample-frame count")
        if _skip_post_verify:
            if sha256_file(before.source) != before.sha256:
                raise RuntimeError("Audio source changed during processing")
            return AudioRemovalResult(
                before.source,
                target,
                frames_written,
                before,
                None,
                None,
                False,
                before.metadata_markers,
                method=method,
                bandwidth_kbps=bandwidth_kbps if neural else None,
                vocoder_device=vocoder_device if method in _RAW_CODEC_PROFILES else None,
            )
        fidelity = measure_fidelity(before.source, before.stream, staged)
        quality_ok = audio_quality_passed(fidelity, lossy=staged.suffix.lower() in AUDIO_LOSSY_OUTPUT_EXTENSIONS)
        after = identify_audio(staged, check_invisible=verify, scheme=scheme)
        if after.metadata_markers:
            raise RuntimeError(f"AI metadata remains in the processed audio: {sorted(after.metadata_markers)}")
        verified = verification_outcome(before, after)
        enforce_audio_publication_gates(
            before,
            after,
            verified=verified,
            quality_ok=quality_ok,
            force=force,
            allow_unverified=allow_unverified,
            allow_positive=allow_positive,
            allow_quality_override=allow_quality_override,
        )
        if sha256_file(before.source) != before.sha256:
            raise RuntimeError("Audio source changed during processing")
    after = replace(after, source=target)
    scope = tuple(sorted({d.scheme for d in before.detections + after.detections}))
    return AudioRemovalResult(
        before.source,
        target,
        frames_written,
        before,
        after,
        verified,
        False,
        before.metadata_markers,
        fidelity=fidelity,
        method=method,
        bandwidth_kbps=bandwidth_kbps if neural else None,
        vocoder_device=vocoder_device if method in _RAW_CODEC_PROFILES else None,
        verification_scope=scope,
        quality_passed=quality_ok,
    )


missing_decoder_reason = _missing_decoder_reason
remove_audio_profile = _remove_audio_profile


def _vocode_full_track(
    chunks: Iterator[NDArray[Any]], sample_rate: int, device: Literal["cpu", "mps", "cuda"]
) -> Iterator[NDArray[Any]]:
    """Use the bounded collector without changing BigVGAN math or arguments."""
    samples = _collect_full_track(chunks, sample_rate, "BigVGAN")
    yield audio_vocoder.resynthesize(samples, sample_rate, device=device, local_files_only=False)


def _mimi_full_track(chunks: Iterator[NDArray[Any]], sample_rate: int) -> Iterator[NDArray[Any]]:
    """Apply the explicit experimental CPU Mimi profile to one complete track."""
    samples = _collect_full_track(chunks, sample_rate, "Mimi")
    yield audio_mimi.resynthesize(samples, sample_rate, local_files_only=False)


def _focalcodec_full_track(chunks: Iterator[NDArray[Any]], sample_rate: int) -> Iterator[NDArray[Any]]:
    """Apply the explicit experimental CPU FocalCodec feature profile across short or long tracks."""
    yield from _stream_neural_track(
        chunks,
        sample_rate,
        "FocalCodec",
        lambda samples: audio_focalcodec.resynthesize_conditioned(samples, sample_rate, local_files_only=False),
    )


def _lowpass_lfe_channel(lfe: NDArray[Any], sample_rate: int) -> NDArray[Any]:
    """Strip out-of-band watermark energy (>120 Hz) from LFE while preserving subwoofer bass."""
    import numpy as np

    signal: Any = import_module("scipy.signal")
    arr = np.asarray(lfe, dtype=np.float64)
    if arr.shape[0] <= 16 or sample_rate <= 300:
        return arr.astype(np.float32)
    sos = np.asarray(signal.butter(4, 120.0, btype="lowpass", fs=sample_rate, output="sos"), dtype=np.float64)
    pad = min(3 * (2 * len(sos) + 1), arr.shape[0] - 1)
    filtered = np.asarray(signal.sosfiltfilt(sos, arr, axis=0, padlen=pad), dtype=np.float64)
    return filtered.astype(np.float32)


def _transform_surround_stems(
    samples: NDArray[Any],
    sample_rate: int,
    stem_fn: Callable[[NDArray[Any]], NDArray[Any]],
) -> NDArray[Any]:
    """Decompose 5.1 (6ch) or 7.1 (8ch) into FL/FR, FC, LFE (<120 Hz), and surround M/S pairs."""
    import numpy as np

    ch = samples.shape[1]
    if ch in (1, 2):
        return np.asarray(stem_fn(samples), dtype=np.float32)
    if ch not in (6, 8):
        raise ValueError("Surround stem decomposition requires 6 (5.1) or 8 (7.1) channels")
    fl_fr = np.asarray(stem_fn(samples[:, 0:2]), dtype=np.float32)
    fc = np.asarray(stem_fn(samples[:, 2:3]), dtype=np.float32)
    lfe = _lowpass_lfe_channel(samples[:, 3:4], sample_rate)
    sur1 = np.asarray(stem_fn(samples[:, 4:6]), dtype=np.float32)
    if ch == 6:
        return np.concatenate((fl_fr, fc, lfe, sur1), axis=1).astype(np.float32, copy=False)
    sur2 = np.asarray(stem_fn(samples[:, 6:8]), dtype=np.float32)
    return np.concatenate((fl_fr, fc, lfe, sur1, sur2), axis=1).astype(np.float32, copy=False)


def _encodec_surround_track(
    chunks: Iterator[NDArray[Any]],
    sample_rate: int,
    *,
    bandwidth_kbps: float,
    device: str = "cpu",
) -> Iterator[NDArray[Any]]:
    """Apply M/S EnCodec across tagged 5.1 or 7.1 surround stems."""

    def _stem(part: NDArray[Any]) -> NDArray[Any]:
        mode: Literal["independent", "ms"] = "ms" if part.shape[1] == 2 else "independent"
        if device == "cpu":
            return audio_codec.resynthesize(part, sample_rate, bandwidth_kbps=bandwidth_kbps, stereo_mode=mode)
        return audio_codec.resynthesize(
            part,
            sample_rate,
            bandwidth_kbps=bandwidth_kbps,
            stereo_mode=mode,
            device=device,
        )

    yield from _stream_neural_track(
        chunks,
        sample_rate,
        "EnCodec",
        lambda samples: _transform_surround_stems(samples, sample_rate, _stem),
        allow_surround=True,
    )


def _surround_codec_full_track(
    chunks: Iterator[NDArray[Any]],
    sample_rate: int,
    label: str,
    resynthesize_fn: Callable[..., Any],
    device: str = "cpu",
    *,
    preserve_hf_hz: float | None = None,
) -> Iterator[NDArray[Any]]:
    def _stem(part: NDArray[Any]) -> NDArray[Any]:
        kwargs: dict[str, Any] = {"local_files_only": False}
        if device != "cpu":
            kwargs["device"] = device
        if preserve_hf_hz is not None:
            kwargs["preserve_hf_hz"] = preserve_hf_hz
        return resynthesize_fn(part, sample_rate, **kwargs)

    yield from _stream_neural_track(
        chunks,
        sample_rate,
        label,
        lambda samples: _transform_surround_stems(samples, sample_rate, _stem),
        allow_surround=True,
    )


def _dac_full_track(
    chunks: Iterator[NDArray[Any]],
    sample_rate: int,
    device: str = "cpu",
    *,
    preserve_hf_hz: float | None = None,
) -> Iterator[NDArray[Any]]:
    """Apply the 44.1 kHz DAC profile across mono, stereo, 5.1, or 7.1 tracks on the selected device."""
    yield from _surround_codec_full_track(
        chunks, sample_rate, "DAC", audio_dac.resynthesize, device, preserve_hf_hz=preserve_hf_hz
    )


def _snac_full_track(
    chunks: Iterator[NDArray[Any]],
    sample_rate: int,
    device: str = "cpu",
    *,
    preserve_hf_hz: float | None = None,
) -> Iterator[NDArray[Any]]:
    """Apply the 44.1 kHz multi-scale SNAC profile across mono, stereo, 5.1, or 7.1 tracks."""
    yield from _surround_codec_full_track(
        chunks, sample_rate, "SNAC", audio_snac.resynthesize, device, preserve_hf_hz=preserve_hf_hz
    )


def _stream_neural_track(
    chunks: Iterator[NDArray[Any]],
    sample_rate: int,
    label: str,
    transform: Callable[[NDArray[Any]], NDArray[Any]],
    *,
    allow_surround: bool = False,
) -> Iterator[NDArray[Any]]:
    """Run single-pass resynthesis up to 30 s, and raised-cosine overlap windows on longer audio."""
    from collections import deque

    import numpy as np

    allowed_channels = (1, 2, 6, 8) if allow_surround else (1, 2)
    pending: deque[NDArray[Any]] = deque()
    buffered_frames = 0
    channels: int | None = None
    eof = False

    def _pull_one() -> bool:
        nonlocal buffered_frames, channels, eof
        if eof:
            return False
        try:
            chunk = next(chunks)
        except StopIteration:
            eof = True
            return False
        if chunk.ndim != 2 or chunk.shape[1] not in allowed_channels:
            raise ValueError(f"{label} processing requires mono or stereo")
        if channels is None:
            channels = int(chunk.shape[1])
        elif int(chunk.shape[1]) != channels:
            raise ValueError(f"{label} processing requires a constant channel count")
        if len(chunk):
            pending.append(np.asarray(chunk, dtype=np.float32))
            buffered_frames += len(chunk)
        return True

    def _pop_frames(target_frames: int) -> NDArray[Any]:
        nonlocal buffered_frames
        parts: list[NDArray[Any]] = []
        remaining = target_frames
        while remaining > 0 and pending:
            head = pending[0]
            if len(head) <= remaining:
                parts.append(pending.popleft())
                buffered_frames -= len(head)
                remaining -= len(head)
            else:
                parts.append(head[:remaining])
                pending[0] = head[remaining:]
                buffered_frames -= remaining
                remaining = 0
        return np.concatenate(parts, axis=0)

    try:
        if not _pull_one() or channels is None or buffered_frames == 0:
            raise ValueError(f"{label} requires nonempty audio")
        whole_limit = min(_MAX_WHOLE_SECONDS * sample_rate, max(1, _MAX_WHOLE_VALUES // channels))
        while buffered_frames <= whole_limit and _pull_one():
            pass

        if eof and buffered_frames <= whole_limit:
            yield np.asarray(transform(_pop_frames(buffered_frames)), dtype=np.float32)
            return

        window_frames = min(20 * sample_rate, max(4, whole_limit * 2 // 3))
        overlap_frames = max(1, min(2 * sample_rate, window_frames // 4))
        fade_in = (
            0.5 - 0.5 * np.cos(np.pi * np.arange(1, overlap_frames + 1, dtype=np.float32) / (overlap_frames + 1))
        )[:, None]
        fade_out = 1.0 - fade_in

        prev_tail: NDArray[Any] | None = None
        in_overlap: NDArray[Any] | None = None
        while True:
            need = window_frames + overlap_frames if in_overlap is None else window_frames
            while buffered_frames < need and _pull_one():
                pass
            if buffered_frames == 0:
                if prev_tail is not None:
                    yield prev_tail
                return
            step_in = _pop_frames(min(need, buffered_frames))
            win_in = step_in if in_overlap is None else np.concatenate((in_overlap, step_in), axis=0)
            win_out = np.asarray(transform(win_in), dtype=np.float32)
            if win_out.shape != win_in.shape:
                raise RuntimeError(f"{label} returned invalid window shape {win_out.shape}")
            if prev_tail is None:
                head = win_out
            else:
                blended = np.asarray(prev_tail * fade_out + win_out[:overlap_frames] * fade_in, dtype=np.float32)
                head = np.concatenate((blended, win_out[overlap_frames:]), axis=0)
            if eof and buffered_frames == 0:
                yield head
                return
            yield head[:-overlap_frames]
            prev_tail = win_out[-overlap_frames:]
            in_overlap = win_in[-overlap_frames:]
    finally:
        close = getattr(chunks, "close", None)
        if callable(close):
            close()


def _collect_full_track(chunks: Iterator[NDArray[Any]], sample_rate: int, label: str) -> NDArray[Any]:
    """Bound full-track inference and close the producer before invoking a model."""
    import numpy as np

    collected: list[NDArray[Any]] = []
    count = 0
    try:
        for chunk in chunks:
            count += len(chunk)
            if count > _MAX_WHOLE_SECONDS * sample_rate or count * chunk.shape[1] > _MAX_WHOLE_VALUES:
                raise ValueError(f"{label} full-track processing is limited to 30 seconds and 16000000 sample values")
            if chunk.shape[1] not in (1, 2):
                raise ValueError(f"{label} full-track processing requires mono or stereo")
            collected.append(chunk)
    finally:
        close = getattr(chunks, "close", None)
        if callable(close):
            close()
    if not collected:
        raise ValueError(f"{label} requires nonempty audio")
    return np.concatenate(collected, axis=0)


def _audio_source(source: str | Path) -> Path:
    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Audio source must be a file: {path}")
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported audio format {path.suffix or '<none>'}; expected one of: {supported}")
    if (reason := metadata.unreadable_reason(path)) is not None:
        raise ValueError(f"Cannot read audio {path}: {reason}")
    return path


def _audio_metadata_output(source: Path, output: str | Path | None) -> Path:
    path = Path(output).expanduser() if output is not None else source.with_stem(source.stem + "_clean")
    if path.suffix.lower() != source.suffix.lower():
        raise ValueError(
            f"Audio metadata output container must match the source ({source.suffix}); "
            f"metadata removal does not change containers to {path.suffix or '<none>'}"
        )
    if path.resolve() == source.resolve() or (path.exists() and path.samefile(source)):
        raise ValueError("Audio metadata removal requires a distinct output path")
    return path


def inspect_audio_metadata(source: str | Path) -> AudioMetadataReport:
    """Inspect supported AI-provenance metadata in an audio container."""
    source_path = _audio_source(source)
    markers = metadata.get_ai_metadata(source_path)
    return AudioMetadataReport(
        source=source_path,
        has_ai_metadata=bool(markers),
        markers=markers,
    )


def remove_audio_metadata(
    source: str | Path,
    output: str | Path | None = None,
    *,
    keep_standard: bool = True,
    _detected_metadata: dict[str, str] | None = None,
) -> AudioMetadataResult:
    """Remove AI metadata from an audio container without re-encoding audio samples.

    The default output is ``<source_stem>_clean<source_suffix>``. A separate
    output path is required so the operation never overwrites the original file.
    """
    source_path = _audio_source(source)
    output_path = _audio_metadata_output(source_path, output)
    detected = metadata.get_ai_metadata(source_path) if _detected_metadata is None else _detected_metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written, remaining = metadata.strip_and_verify(source_path, output_path, keep_standard=keep_standard)
    return AudioMetadataResult(
        source=source_path,
        output=written,
        detected=detected,
        remaining=remaining,
    )


def _transcode_clean_audio(source: Path, target: Path, *, stream: int = 0) -> None:
    """Write a metadata-free audio copy when output container differs from source."""
    info = audio_io.probe_audio(source, stream=stream)
    target.parent.mkdir(parents=True, exist_ok=True)
    with staged_sibling(target) as staged, closing(audio_io.read_audio(source, info)) as reader:
        audio_io.write_audio(info, staged, reader)


def remove_audio_all(
    source: str | Path,
    output: str | Path | None = None,
    *,
    include_invisible: bool = False,
    force: bool = False,
    verify: bool = True,
    stream: int = 0,
    method: AudioMethod = "auto",
    bandwidth_kbps: float = 24.0,
    scheme: AudioScheme = "auto",
    vocoder_device: Literal["cpu", "mps", "cuda"] = "cpu",
    keep_standard: bool = True,
    allow_unverified: bool = False,
    allow_positive: bool = False,
    allow_quality_override: bool = False,
    vendor: AudioVendor | None = None,
    preserve_hf_hz: float | None = None,
) -> AudioAllResult:
    """Remove AI metadata and optionally process compatible invisible audio watermarks.

    When ``include_invisible=False`` (default) and no explicit ``vendor`` cohort is
    supplied, strips AI metadata losslessly without re-encoding audio packets. Naming
    an explicit ``vendor`` cohort implies running the invisible waveform stage just
    like ``--vendor`` on images. When ``include_invisible=True``, inspects compatible
    audio watermark decoders (default ``scheme="auto"`` across installed public models)
    and removes detected marks; if no compatible source-positive mark is found and
    ``force=False``, falls back to lossless metadata stripping when the output container
    matches the source (or clean transcoding when an explicit cross-format output was
    requested), always delivering a metadata-clean output file.
    """
    source_path = _audio_source(source)
    explicit_vendor = _normalize_audio_vendor(vendor)
    if not (include_invisible or explicit_vendor is not None):
        meta_res = remove_audio_metadata(source_path, output, keep_standard=keep_standard)
        return AudioAllResult(
            source=source_path,
            output=meta_res.output,
            detected_metadata=meta_res.detected,
            remaining_metadata=meta_res.remaining,
            invisible_applied=False,
            invisible_result=None,
        )

    resolve_audio_profiles(
        method=method,
        scheme=scheme,
        verify=verify,
        force=force,
        bandwidth_kbps=bandwidth_kbps,
        vocoder_device=vocoder_device,
        allow_unverified=allow_unverified or explicit_vendor is not None,
        vendor=vendor,
    )
    if output is not None:
        target = Path(output).expanduser()
    elif source_path.suffix.lower() in AUDIO_OUTPUT_EXTENSIONS:
        target = source_path.with_stem(source_path.stem + "_clean")
    else:
        target = source_path.with_name(source_path.stem + "_clean.wav")
    if target.suffix.lower() not in AUDIO_OUTPUT_EXTENSIONS:
        supported = ", ".join(AUDIO_OUTPUT_EXTENSIONS)
        raise ValueError(f"Audio output format must be one of: {supported}")
    if target.resolve() == source_path.resolve() or (target.exists() and target.samefile(source_path)):
        raise ValueError("Audio output must differ from the source")

    inv_res = remove_audio_invisible(
        source_path,
        target,
        force=force,
        verify=verify,
        stream=stream,
        method=method,
        bandwidth_kbps=bandwidth_kbps,
        scheme=scheme,
        vocoder_device=vocoder_device,
        allow_unverified=allow_unverified,
        allow_positive=allow_positive,
        allow_quality_override=allow_quality_override,
        vendor=vendor,
        preserve_hf_hz=preserve_hf_hz,
    )
    detected_metadata = dict(inv_res.before.metadata_markers)
    if not inv_res.skipped and inv_res.output is not None:
        remaining = (
            dict(inv_res.after.metadata_markers)
            if inv_res.after is not None
            else metadata.get_ai_metadata(inv_res.output)
        )
        return AudioAllResult(
            source=source_path,
            output=inv_res.output,
            detected_metadata=detected_metadata,
            remaining_metadata=remaining,
            invisible_applied=True,
            invisible_result=inv_res,
        )

    if target.suffix.lower() == source_path.suffix.lower():
        meta_res = remove_audio_metadata(
            source_path,
            target,
            keep_standard=keep_standard,
            _detected_metadata=detected_metadata,
        )
        remaining = meta_res.remaining
        delivered = meta_res.output
    else:
        _transcode_clean_audio(source_path, target, stream=stream)
        remaining = metadata.get_ai_metadata(target)
        delivered = target
    return AudioAllResult(
        source=source_path,
        output=delivered,
        detected_metadata=detected_metadata,
        remaining_metadata=remaining,
        invisible_applied=False,
        invisible_result=inv_res,
    )


def remove_audio_batch(
    directory: str | Path,
    output_directory: str | Path | None = None,
    *,
    mode: Literal["all", "metadata"] = "all",
    include_invisible: bool = False,
    force: bool = False,
    verify: bool = True,
    method: AudioMethod = "auto",
    bandwidth_kbps: float = 24.0,
    scheme: AudioScheme = "auto",
    vocoder_device: Literal["cpu", "mps", "cuda"] = "cpu",
    keep_standard: bool = True,
    allow_unverified: bool = False,
    allow_positive: bool = False,
    allow_quality_override: bool = False,
    vendor: AudioVendor | None = None,
) -> AudioBatchSummary:
    """Process every supported audio file in one directory sequentially."""
    directory_path = Path(directory).expanduser()
    if not directory_path.exists():
        raise FileNotFoundError(f"Audio directory does not exist: {directory_path}")
    if not directory_path.is_dir():
        raise ValueError(f"Audio batch source must be a directory: {directory_path}")
    if mode not in {"all", "metadata"}:
        raise ValueError("Unsupported audio batch mode; expected all or metadata")
    explicit_vendor = _normalize_audio_vendor(vendor)
    eff_invisible = include_invisible or explicit_vendor is not None
    if eff_invisible and mode != "all":
        raise ValueError("The invisible audio stage is available only in all mode")
    if eff_invisible:
        resolve_audio_profiles(
            method=method,
            scheme=scheme,
            verify=verify,
            force=force,
            bandwidth_kbps=bandwidth_kbps,
            vocoder_device=vocoder_device,
            allow_unverified=allow_unverified or explicit_vendor is not None,
            vendor=vendor,
        )

    output_path = (
        Path(output_directory).expanduser()
        if output_directory is not None
        else directory_path.parent / f"{directory_path.name}_clean"
    )
    if output_path.resolve() == directory_path.resolve():
        raise ValueError("Audio batch output directory must differ from the source directory")

    sources = tuple(
        path
        for path in sorted(directory_path.iterdir(), key=lambda candidate: candidate.name.lower())
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    if sources:
        output_path.mkdir(parents=True, exist_ok=True)
    items: list[AudioBatchItem] = []
    for source_path in sources:
        suffix = (
            source_path.suffix
            if (not eff_invisible or source_path.suffix.lower() in AUDIO_OUTPUT_EXTENSIONS)
            else ".wav"
        )
        item_output = output_path / f"{source_path.stem}{suffix}"
        try:
            if mode == "all":
                all_res = remove_audio_all(
                    source_path,
                    item_output,
                    include_invisible=eff_invisible,
                    force=force,
                    verify=verify,
                    method=method,
                    bandwidth_kbps=bandwidth_kbps,
                    scheme=scheme,
                    vocoder_device=vocoder_device,
                    keep_standard=keep_standard,
                    allow_unverified=allow_unverified,
                    allow_positive=allow_positive,
                    allow_quality_override=allow_quality_override,
                    vendor=vendor,
                )
                if all_res.remaining_metadata:
                    raise RuntimeError(
                        f"{len(all_res.remaining_metadata)} AI metadata marker(s) survived audio cleaning"
                    )
                items.append(
                    AudioBatchItem(
                        source=source_path,
                        output=all_res.output,
                        mode=mode,
                        changed=bool(
                            all_res.detected_metadata or all_res.invisible_applied or suffix != source_path.suffix
                        ),
                        invisible_applied=all_res.invisible_applied,
                    )
                )
            else:
                meta_res = remove_audio_metadata(source_path, item_output, keep_standard=keep_standard)
                if meta_res.remaining:
                    raise RuntimeError(
                        f"{len(meta_res.remaining)} AI metadata marker(s) survived audio metadata removal"
                    )
                items.append(
                    AudioBatchItem(
                        source=source_path,
                        output=meta_res.output,
                        mode=mode,
                        changed=bool(meta_res.detected),
                        invisible_applied=False,
                    )
                )
        except Exception as exc:
            items.append(
                AudioBatchItem(
                    source=source_path,
                    output=None,
                    mode=mode,
                    changed=False,
                    invisible_applied=False,
                    error=str(exc),
                )
            )
    return AudioBatchSummary(
        directory=directory_path,
        output_directory=output_path,
        items=tuple(items),
    )
