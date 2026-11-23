"""High-level video processing API.

The product path covers provenance identification, container-level AI metadata
removal, temporally stabilized visible Sora, Veo, Seedance, Doubao, Dola,
Hailuo AI, Vidu, and Kling AI removal, and explicit VAE profiles for video
SynthID and experimental Seedance processing.
The visible pixel path reuses the image package's shared fill backends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal, cast

from remove_ai_watermarks._internal.constants import (
    ISOBMFF_VIDEO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    VIDEO_INVISIBLE_PROFILES,
    VIDEO_PROFILE_ALIASES,
    WAN_EXPERIMENTAL_PROFILES,
    VideoInvisibleProfile,
)
from remove_ai_watermarks._internal.tc260_producers import TC260_PRODUCERS
from remove_ai_watermarks.video_synthid import (
    DEFAULT_VIDEO_SYNTHID_FPS,
    DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    DEFAULT_VIDEO_SYNTHID_NOISE_STD,
    DEFAULT_VIDEO_SYNTHID_VAE,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from remove_ai_watermarks._internal.constants import (
        AudioDevice,
        AudioMethod,
        AudioScheme,
        AudioVendor,
        WanExperimentalProfile,
    )
    from remove_ai_watermarks.audio import AudioProvenanceReport
    from remove_ai_watermarks.video_invisible import RegenerationMetrics, VideoVaeRuntime
    from remove_ai_watermarks.video_visible import VideoScan
    from remove_ai_watermarks.video_wan import VideoWanRuntime

# Re-exported from _internal.constants for callers importing VIDEO_EXTENSIONS from video.
_VIDEO_EXTENSIONS_EXPORT: frozenset[str] = VIDEO_EXTENSIONS
VIDEO_VISIBLE_MARKS = (
    "sora",
    "veo",
    "dreamina",
    "jimeng",
    "jimeng_pill",
    "seedance",
    "doubao",
    "dola",
    "hailuo",
    "vidu",
    "wan",
    "qwen",
    "yuanbao",
    "baidu",
    "runninghub",
    "microsoft",
    "pixverse",
    "luma",
    "heygen",
    "sync_labs",
    "tiktok_ai",
    "generic_ai_label",
    "kling",
)
_EXPECTED_VIDEO_MARKS = f"{', '.join(('auto', *VIDEO_VISIBLE_MARKS[:-1]))}, or {VIDEO_VISIBLE_MARKS[-1]}"
_ISOBMFF_VIDEO_EXTENSIONS: frozenset[str] = ISOBMFF_VIDEO_EXTENSIONS
_EBML_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".webm", ".mkv"})
_RIFF_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".avi"})
_FLV_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".flv"})
_REGENERATED_VIDEO_EXTENSIONS: frozenset[str] = VIDEO_EXTENSIONS
_EXPECTED_REGENERATED_VIDEO_EXTENSIONS = ", ".join(sorted(_REGENERATED_VIDEO_EXTENSIONS))


def _require_video_runtime() -> None:
    """Raise with the public install command when a video runtime is absent.

    Shell-quoted, for the same reason as the other install hints: a bare
    ``pkg[extra]`` is a glob in zsh, so an unquoted copy-paste dies with
    "no matches found" before pip ever runs.
    """
    from remove_ai_watermarks.optional_deps import module_available

    if not module_available("cv2", "numpy", "av"):
        raise RuntimeError("Video pixel processing requires 'remove-ai-watermarks[video]'")


@dataclass(frozen=True)
class VideoMetadataReport:
    """AI metadata found in one supported video container."""

    source: Path
    has_ai_metadata: bool
    markers: dict[str, str]


@dataclass(frozen=True)
class VideoAudioStatus:
    """Audio disposition and watermark status for a video output."""

    stream_action: Literal["copied_if_present", "not_written", "processed", "absent"] = "copied_if_present"
    watermark_status: Literal["unverified", "not_detected", "detected"] = "unverified"


class _VideoAudioBoundary:
    """Derive the audio disposition from explicit audio processing or output presence."""

    @property
    def audio(self) -> VideoAudioStatus:
        """Return the audio status for this result."""
        explicit = getattr(self, "audio_status", None)
        if isinstance(explicit, VideoAudioStatus):
            return explicit
        action: Literal["copied_if_present", "not_written"] = (
            "copied_if_present" if getattr(self, "output", None) is not None else "not_written"
        )
        return VideoAudioStatus(stream_action=action)


@dataclass(frozen=True)
class VideoMetadataResult(_VideoAudioBoundary):
    """Result of a verified metadata strip that preserves source audio."""

    source: Path
    output: Path
    detected: dict[str, str]
    remaining: dict[str, str]


@dataclass(frozen=True)
class VideoProvenanceReport:
    """Locally verifiable provenance signals found in one video."""

    source: Path
    is_ai_generated: Literal[True] | None
    confidence: Literal["high", "unknown"]
    platform: str | None
    visible_mark: str | None
    visible_detected_frames: int
    total_frames: int | None
    has_ai_metadata: bool
    metadata_markers: dict[str, str]
    caveats: tuple[str, ...]
    invisible_watermark_hints: tuple[str, ...] = ()
    audio_streams: tuple[AudioProvenanceReport, ...] = ()
    recommended_recipe: str | None = None
    visible_marks: tuple[str, ...] = ()


@dataclass(frozen=True)
class VideoVisibleResult(_VideoAudioBoundary):
    """Result of visible AI-watermark removal with unverified copied audio."""

    source: Path
    output: Path | None
    mark: str
    total_frames: int
    detected_frames: int
    removed_frames: int
    remaining_metadata: dict[str, str]
    marks: tuple[str, ...] = ()
    trimmed_bumper_frames: int = 0


@dataclass(frozen=True)
class VideoEraseResult(_VideoAudioBoundary):
    """Result of user-directed rectangular region removal on video frames."""

    source: Path
    output: Path
    regions: tuple[tuple[int, int, int, int], ...]
    total_frames: int
    erased_frames: int
    frame_range: tuple[int, int] | None
    backend: str
    remaining_metadata: dict[str, str]
    tracked: bool = False


@dataclass(frozen=True)
class VideoInvisibleResult(_VideoAudioBoundary):
    """Result of applying the calibrated VAE profile to video pixels only."""

    source: Path
    output: Path
    noise_std: float
    metrics: RegenerationMetrics
    remaining_metadata: dict[str, str]
    profile: str = "synthid"

    @property
    def visual_invisible_action(self) -> Literal["regenerated"]:
        """Describe the pixel action without claiming a per-file watermark verdict."""
        return "regenerated"

    @property
    def total_frames(self) -> int:
        return self.metrics.frames

    @property
    def fps(self) -> float:
        return self.metrics.fps

    @property
    def width(self) -> int:
        return self.metrics.width

    @property
    def height(self) -> int:
        return self.metrics.height

    @property
    def psnr_db(self) -> float:
        return self.metrics.psnr_db

    @property
    def temporal_residual_ratio(self) -> float:
        return self.metrics.temporal_residual_ratio


@dataclass(frozen=True)
class VideoSeedanceExperimentalResult(VideoInvisibleResult):
    """Experimental Seedance reconstruction, without a per-file detector verdict."""

    quality: Literal["delivery", "master"] = "delivery"
    profile: Literal["seedance-texture-v1"] = "seedance-texture-v1"
    experimental: bool = True


@dataclass(frozen=True)
class VideoWanExperimentalResult(VideoInvisibleResult):
    """Experimental CUDA 3D Wan 2.1 reconstruction, without a per-file detector verdict.

    ``strength`` carries the calibrated flow-matching denoising strength;
    ``noise_std`` mirrors it for ``VideoInvisibleResult`` compatibility.
    """

    quality: Literal["delivery", "master"] = "delivery"
    profile: WanExperimentalProfile = "wan21-synthid-experimental"
    experimental: bool = True
    strength: float = -1.0

    def __post_init__(self) -> None:
        if self.strength < 0.0:
            object.__setattr__(self, "strength", self.noise_std)


@dataclass(frozen=True)
class VideoAllResult(_VideoAudioBoundary):
    """Result of visible, metadata, and optional video-pixel processing."""

    source: Path
    output: Path
    visible_mark: str | None
    total_frames: int
    visible_detected_frames: int
    visible_removed_frames: int
    detected_metadata: dict[str, str]
    remaining_metadata: dict[str, str]
    invisible_removed: bool
    invisible_profile: VideoInvisibleProfile | None = None
    audio_status: VideoAudioStatus | None = None
    audio_processed_streams: tuple[int, ...] = ()
    audio_verification_scope: tuple[str, ...] = ()
    audio_quality_passed: bool | None = None
    visible_marks: tuple[str, ...] = ()
    trimmed_bumper_frames: int = 0

    @property
    def visual_invisible_action(self) -> Literal["not_run", "regenerated"]:
        """Describe whether the opt-in visual regeneration stage ran."""
        return "regenerated" if self.invisible_removed else "not_run"


@dataclass(frozen=True)
class VideoBatchItem(_VideoAudioBoundary):
    """Outcome for one source in a video batch, including the audio boundary."""

    source: Path
    output: Path | None
    mode: Literal["all", "visible", "metadata"]
    changed: bool
    visible_mark: str | None
    invisible_removed: bool
    error: str | None = None
    audio_status: VideoAudioStatus | None = None
    audio_verification_scope: tuple[str, ...] = ()
    audio_quality_passed: bool | None = None
    visible_marks: tuple[str, ...] = ()

    @property
    def visual_invisible_action(self) -> Literal["not_run", "regenerated"]:
        """Describe whether visual regeneration ran for this item."""
        return "regenerated" if self.invisible_removed else "not_run"


@dataclass(frozen=True)
class VideoBatchResult:
    """Aggregate outcome for a sequential video batch."""

    directory: Path
    output_directory: Path
    items: tuple[VideoBatchItem, ...]

    @property
    def processed(self) -> int:
        return sum(item.error is None for item in self.items)

    @property
    def failed(self) -> int:
        return sum(item.error is not None for item in self.items)

    @property
    def invisible_removed(self) -> int:
        return sum(item.invisible_removed for item in self.items)


_VISIBLE_PLATFORM = {
    "sora": "OpenAI Sora",
    "veo": "Google Veo",
    "jimeng": "ByteDance Jimeng",
    "jimeng_pill": "ByteDance Jimeng",
    "seedance": "ByteDance Seedance",
    "dola": "ByteDance Dola",
    "runninghub": "RunningHub",
    "microsoft": "Microsoft",
    "pixverse": "PixVerse",
    "luma": "Luma AI",
    "heygen": "HeyGen",
    "sync_labs": "Sync Labs",
    "tiktok_ai": "TikTok",
    "generic_ai_label": "AI-generated content (GB 45438-2025 label)",
    **{
        row.video_mark: row.video_platform
        for row in TC260_PRODUCERS
        if row.video_mark is not None and row.video_platform is not None
    },
}

_VIDEO_MARK_VENDOR_GROUP: dict[str, str] = {
    "sora": "openai",
    "veo": "google",
    "dreamina": "bytedance",
    "jimeng": "bytedance",
    "jimeng_pill": "bytedance",
    "seedance": "bytedance",
    "doubao": "bytedance",
    "dola": "bytedance",
    "hailuo": "minimax",
    "vidu": "shengshu",
    "wan": "alibaba",
    "qwen": "alibaba",
    "yuanbao": "tencent",
    "baidu": "baidu",
    "runninghub": "runninghub",
    "microsoft": "microsoft",
    "pixverse": "pixverse",
    "luma": "luma",
    "heygen": "heygen",
    "sync_labs": "sync_labs",
    "tiktok_ai": "bytedance",
    "generic_ai_label": "generic_tc260",
    "kling": "kuaishou",
}


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    return min(ax0 + aw, bx0 + bw) > max(ax0, bx0) and min(ay0 + ah, by0 + bh) > max(ay0, by0)


def _union_box(regions: Sequence[tuple[int, int, int, int] | None]) -> tuple[int, int, int, int] | None:
    present = [r for r in regions if r is not None]
    if not present:
        return None
    x0 = min(r[0] for r in present)
    y0 = min(r[1] for r in present)
    x1 = max(r[0] + r[2] for r in present)
    y1 = max(r[1] + r[3] for r in present)
    return (x0, y0, x1 - x0, y1 - y0)


def _video_source(source: str | Path) -> Path:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Video does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Video source must be a file: {path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise ValueError(f"Unsupported video format {path.suffix or '<none>'}; expected one of: {supported}")
    from remove_ai_watermarks._internal.ebml import EBML_MAGIC

    with path.open("rb") as stream:
        head = stream.read(12)
    suffix = path.suffix.lower()
    matches_container = (
        (suffix in _ISOBMFF_VIDEO_EXTENSIONS and len(head) >= 8 and head[4:8] == b"ftyp")
        or (suffix in _EBML_VIDEO_EXTENSIONS and head.startswith(EBML_MAGIC))
        or (suffix in _RIFF_VIDEO_EXTENSIONS and len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"AVI ")
        or (suffix in _FLV_VIDEO_EXTENSIONS and head.startswith(b"FLV"))
    )
    if not matches_container:
        raise ValueError(f"Video content does not match its {suffix} extension: {path}")
    from remove_ai_watermarks.metadata import unreadable_reason

    if (reason := unreadable_reason(path)) is not None:
        raise ValueError(f"Cannot read video {path}: {reason}")
    return path


def _video_output(
    source: Path,
    output: str | Path | None,
    *,
    operation: str = "metadata removal",
) -> Path:
    path = Path(output) if output is not None else source.with_stem(source.stem + "_clean")
    if path.suffix.lower() != source.suffix.lower():
        raise ValueError(
            f"Video output container must match the source ({source.suffix}); "
            f"{operation} does not change containers to {path.suffix or '<none>'}"
        )
    if path.resolve() == source.resolve():
        raise ValueError(f"Video {operation} requires a distinct output path")
    return path


def _visible_removal_plan(
    selected_mark: str,
    selected_scan: VideoScan,
    markers: dict[str, str],
) -> tuple[list[tuple[int, int, int, int] | None], float, Literal["box", "veo"]]:
    """Resolve one provider's stable frame regions and fill geometry.

    Everything provider-specific -- the confidence floors, the run length, the fill
    padding and the mask style -- is data on ``VISIBLE_MARK_POLICIES``; which metadata
    confirms or contradicts which vendor lives beside the predicates in
    ``video_visible``.
    """
    from remove_ai_watermarks.video_visible import (
        VISIBLE_MARK_POLICIES,
        confirms_video_provenance,
        contradicts_video_provenance,
        stabilize_localizations,
    )

    policy = VISIBLE_MARK_POLICIES[selected_mark]
    if contradicts_video_provenance(selected_mark, markers):
        return [None] * len(selected_scan.detections), policy.padding_fraction, policy.mask_style
    confirmed = confirms_video_provenance(selected_mark, markers)
    regions = stabilize_localizations(selected_mark, selected_scan.detections, provenance=confirmed)
    return regions, policy.padding_fraction, policy.mask_style


def _select_stable_visible_marks(
    scans: dict[str, VideoScan],
    markers: dict[str, str],
    candidate_marks: tuple[str, ...],
    *,
    explicit_multi: bool = False,
) -> tuple[
    tuple[
        str,
        VideoScan,
        list[tuple[int, int, int, int] | None],
        float,
        Literal["box", "veo"],
    ],
    ...,
]:
    """Select one or more spatially disjoint stable marks without cross-vendor false merges."""
    accepted: list[
        tuple[
            str,
            VideoScan,
            list[tuple[int, int, int, int] | None],
            float,
            Literal["box", "veo"],
            tuple[int, int, int, int],
        ]
    ] = []
    for candidate_mark in candidate_marks:
        if accepted and not explicit_multi:
            primary_group = _VIDEO_MARK_VENDOR_GROUP.get(accepted[0][0])
            if primary_group is None or _VIDEO_MARK_VENDOR_GROUP.get(candidate_mark) != primary_group:
                continue
        candidate_scan = scans.get(candidate_mark)
        if candidate_scan is None:
            continue
        regions, padding, mask_style = _visible_removal_plan(candidate_mark, candidate_scan, markers)
        ubox = _union_box(regions)
        if ubox is None:
            continue
        if not accepted:
            accepted.append((candidate_mark, candidate_scan, regions, padding, mask_style, ubox))
            continue
        if any(_boxes_overlap(ubox, prev_ubox) for *_, prev_ubox in accepted):
            continue
        if any(
            r_cur is not None and r_prev is not None and _boxes_overlap(r_cur, r_prev)
            for _m, _s, prev_regs, _p, _ms, _ub in accepted
            for r_cur, r_prev in zip(regions, prev_regs, strict=True)
        ):
            continue
        accepted.append((candidate_mark, candidate_scan, regions, padding, mask_style, ubox))
    return tuple((m, s, r, p, ms) for m, s, r, p, ms, _ub in accepted)


def _select_stable_visible_mark(  # pyright: ignore[reportUnusedFunction]
    scans: dict[str, VideoScan],
    markers: dict[str, str],
    candidate_marks: tuple[str, ...],
) -> (
    tuple[
        str,
        VideoScan,
        list[tuple[int, int, int, int] | None],
        float,
        Literal["box", "veo"],
    ]
    | None
):
    """Select the first stable mark in specificity order that provenance does not veto."""
    selected = _select_stable_visible_marks(scans, markers, candidate_marks)
    return selected[0] if selected else None


def _merge_selected_visible_marks(
    selected_all: tuple[
        tuple[str, VideoScan, list[tuple[int, int, int, int] | None], float, Literal["box", "veo"]],
        ...,
    ],
    total_frames: int,
) -> tuple[
    tuple[str, ...],
    str,
    VideoScan,
    list[Any],
    int,
    float,
    Literal["box", "veo"],
    tuple[float, ...] | None,
    tuple[Literal["box", "veo"], ...] | None,
]:
    marks_tuple = tuple(item[0] for item in selected_all)
    primary_mark = marks_tuple[0]
    scan = selected_all[0][1]
    padding_fraction = selected_all[0][3]
    mask_style = selected_all[0][4]
    if len(selected_all) == 1:
        single_regions = selected_all[0][2]
        return (
            marks_tuple,
            primary_mark,
            scan,
            list(single_regions),
            sum(r is not None for r in single_regions),
            padding_fraction,
            mask_style,
            None,
            None,
        )
    slot_pads = tuple(item[3] for item in selected_all)
    slot_styles: tuple[Literal["box", "veo"], ...] = tuple(item[4] for item in selected_all)
    merged_regions: list[Any] = []
    detected_frames = 0
    for idx in range(total_frames):
        slot_tuple = tuple(item[2][idx] for item in selected_all)
        if any(r is not None for r in slot_tuple):
            merged_regions.append(slot_tuple)
            detected_frames += 1
        else:
            merged_regions.append(None)
    return (
        marks_tuple,
        primary_mark,
        scan,
        merged_regions,
        detected_frames,
        padding_fraction,
        mask_style,
        slot_pads,
        slot_styles,
    )


def _parse_mark_selector(mark: str) -> tuple[tuple[str, ...], bool]:
    if mark == "auto":
        return VIDEO_VISIBLE_MARKS, False
    if "," in mark:
        parts = tuple(dict.fromkeys(part.strip() for part in mark.split(",") if part.strip()))
        if not parts or any(part not in VIDEO_VISIBLE_MARKS for part in parts):
            raise ValueError(f"Unsupported visible video mark; expected {_EXPECTED_VIDEO_MARKS}")
        return parts, len(parts) > 1
    if mark not in VIDEO_VISIBLE_MARKS:
        raise ValueError(f"Unsupported visible video mark; expected {_EXPECTED_VIDEO_MARKS}")
    return (mark,), False


def _video_markers_claim_ai(markers: dict[str, str]) -> bool:
    """Return whether the metadata asserts AI, mirroring the image path's C2PA rule."""
    from remove_ai_watermarks._internal.c2pa import markers_claim_ai

    return markers_claim_ai(markers)


def _platform_from_video_metadata(markers: dict[str, str]) -> str | None:
    """Map supported C2PA-derived marker text to its generating platform."""
    from remove_ai_watermarks._internal.c2pa import c2pa_vendor_platform, claim_generator_platform
    from remove_ai_watermarks._internal.constants import C2PA_SIGNER_PLATFORM_BY_ORG
    from remove_ai_watermarks.metadata import (
        LOCAL_GENERATION_METADATA_KEYS,
        TOPAZ_AI_DESCRIPTION,
        TOPAZ_VIDEO_METADATA_KEYS,
    )

    generator = markers.get("claim_generator", "")
    ai_tool = markers.get("ai_tool", "")
    if platform := (
        claim_generator_platform(generator)
        or c2pa_vendor_platform(generator)
        or claim_generator_platform(ai_tool)
        or c2pa_vendor_platform(ai_tool)
        or c2pa_vendor_platform(markers.get("app_provenance", ""))
        or c2pa_vendor_platform("\n".join(markers.values()))
    ):
        return platform
    issuer = markers.get("issuer", "")
    if signer_platform := next(
        (C2PA_SIGNER_PLATFORM_BY_ORG[org] for org in C2PA_SIGNER_PLATFORM_BY_ORG if org in issuer),
        None,
    ):
        return signer_platform
    if "aigc_label" in markers:
        return _tc260_video_platform(markers.get("aigc_producer", "")) or (
            "China AIGC-labeled content (TC260 standard)"
        )
    if (app_aigc := markers.get("app_aigc", "")).startswith("App AIGC disclosure (") and app_aigc.endswith(")"):
        return app_aigc.removeprefix("App AIGC disclosure (").removesuffix(")")
    if (app_provenance := markers.get("app_provenance", "")).startswith(
        "App export provenance ("
    ) and app_provenance.endswith(")"):
        return app_provenance.removeprefix("App export provenance (").removesuffix(")")
    if "kling_sei" in markers:
        return _tc260_video_platform("kling")
    if "xai_signature" in markers:
        return "xAI Grok Imagine"
    if any(key.lower() in TOPAZ_VIDEO_METADATA_KEYS for key in markers) or (
        markers.get("description", "").strip() == TOPAZ_AI_DESCRIPTION.decode("ascii")
    ):
        return "Topaz Labs"
    if any(key.lower() in LOCAL_GENERATION_METADATA_KEYS for key in markers):
        return "Stable Diffusion / local pipeline (Automatic1111, ComfyUI, InvokeAI)"
    if audio_gen := markers.get("audio_generator", "").strip():
        return claim_generator_platform(audio_gen) or c2pa_vendor_platform(audio_gen) or audio_gen
    return None


def _tc260_video_platform(producer: str) -> str | None:
    """Name a registered TC260 producer, preserving multi-model organization labels."""
    from remove_ai_watermarks._internal.tc260_producers import producer_for_code

    row = producer_for_code(producer)
    return row.video_platform if row else None


def inspect_video_metadata(source: str | Path) -> VideoMetadataReport:
    """Inspect supported AI-provenance metadata in a video container."""
    from remove_ai_watermarks.metadata import get_ai_metadata

    source_path = _video_source(source)
    markers = get_ai_metadata(source_path)
    return VideoMetadataReport(
        source=source_path,
        has_ai_metadata=bool(markers),
        markers=markers,
    )


def _video_invisible_watermark_hints(markers: dict[str, str]) -> tuple[str, ...]:
    """Name declared schemes and provider-policy hints, without decoding pixels."""
    from remove_ai_watermarks._internal.c2pa import soft_binding_labels, soft_binding_registry_entries_in
    from remove_ai_watermarks._internal.constants import C2PA_SOFT_BINDING_REGISTRY

    entries = soft_binding_registry_entries_in(markers.get("soft_binding_algorithm", "").encode())
    hints = set(soft_binding_labels(entries, kind="watermark"))
    # The byte-scan fallback exposes labels without identifiers. A label shared
    # with a fingerprint (Trufo PawPrint) cannot establish the algorithm kind.
    if "soft_binding" in markers:
        declared_labels = f", {markers['soft_binding']}, "
        watermark_labels = set(soft_binding_labels(C2PA_SOFT_BINDING_REGISTRY, kind="watermark"))
        fingerprint_labels = set(soft_binding_labels(C2PA_SOFT_BINDING_REGISTRY, kind="fingerprint"))
        hints.update(label for label in watermark_labels - fingerprint_labels if f", {label}, " in declared_labels)
    if "synthid_watermark" in markers:
        hints.add("SynthID")
    if _video_markers_claim_ai(markers) and _platform_from_video_metadata(markers) == "Amazon Bedrock (Nova)":
        hints.add("Amazon Nova Reel")
    return tuple(sorted(hints))


def _recommend_video_recipe(
    *,
    markers: dict[str, str],
    selected_mark: str | None,
    claims_ai: bool,
    invisible_hints: tuple[str, ...],
    audio_streams: tuple[AudioProvenanceReport, ...],
) -> str | None:
    """Suggest an explicit read-only command recipe without claiming hidden marks exist."""
    if _is_bytedance_video_provenance(markers, selected_mark):
        if selected_mark is None:
            return (
                "remove-ai-watermarks video metadata input.mp4 --remove -o clean.mp4 "
                "(try lossless C2PA strip first; add video all --invisible --profile auto "
                "only for consumer CapCut pixel marks)"
            )
        return (
            "remove-ai-watermarks video all input.mp4 -o clean.mp4 "
            "(strips visible mark and C2PA; add --invisible --profile auto only for consumer CapCut pixel marks)"
        )
    platform = _platform_from_video_metadata(markers) if claims_ai else None
    is_google = (
        any("synthid" in h.lower() for h in invisible_hints)
        or (platform is not None and "google" in platform.lower())
        or selected_mark == "veo"
    )
    if is_google:
        if any(r.watermark_status != "detected" for r in audio_streams):
            return (
                "remove-ai-watermarks video all input.mp4 -o clean.mp4 "
                "--invisible --profile auto --audio-invisible --audio-method dac --audio-allow-unverified"
            )
        if audio_streams:
            return "remove-ai-watermarks video all input.mp4 -o clean.mp4 --invisible --profile auto --audio-invisible"
        return "remove-ai-watermarks video all input.mp4 -o clean.mp4 --invisible --profile auto"
    if any(r.watermark_status == "detected" for r in audio_streams):
        return "remove-ai-watermarks video all input.mp4 -o clean.mp4 --audio-invisible"
    if selected_mark is not None or claims_ai:
        return "remove-ai-watermarks video all input.mp4 -o clean.mp4"
    return None


def identify_video(
    source: str | Path,
    *,
    check_visible: bool = True,
    check_audio: bool = False,
    audio_scheme: AudioScheme = "auto",
) -> VideoProvenanceReport:
    """Identify locally readable AI provenance, stable visible marks, and optional audio marks.

    A negative result is reported as unknown, never clean. Proprietary pixel
    watermarks such as video SynthID have no public local decoder.
    """
    from remove_ai_watermarks.metadata import get_ai_metadata

    source_path = _video_source(source)
    markers = get_ai_metadata(source_path)
    selected_mark: str | None = None
    selected_marks_tuple: tuple[str, ...] = ()
    detected_frames = 0
    total_frames: int | None = None

    if check_visible:
        _require_video_runtime()
        from remove_ai_watermarks.video_visible import scan_video_marks

        scans = scan_video_marks(
            source_path,
            VIDEO_VISIBLE_MARKS,
            collect_timestamps=False,
        )
        total_frames = len(scans[VIDEO_VISIBLE_MARKS[0]].detections)
        selected_all = _select_stable_visible_marks(scans, markers, VIDEO_VISIBLE_MARKS)
        if selected_all:
            selected_marks_tuple = tuple(item[0] for item in selected_all)
            selected_mark = selected_marks_tuple[0]
            detected_frames = sum(any(item[2][idx] is not None for item in selected_all) for idx in range(total_frames))

    audio_reports: list[AudioProvenanceReport] = []
    if check_audio:
        from remove_ai_watermarks import audio as _audio_mod
        from remove_ai_watermarks import video_audio as _video_audio_mod

        inv = _video_audio_mod.stream_inventory(source_path)
        n_audio = sum(1 for s in inv if s.get("codec_type") == "audio")
        for ordinal in range(n_audio):
            audio_reports.append(
                _audio_mod.identify_audio(source_path, check_invisible=True, stream=ordinal, scheme=audio_scheme)
            )
    audio_tuple = tuple(audio_reports)
    any_audio_detected = any(r.watermark_status == "detected" for r in audio_tuple)
    claims_ai = _video_markers_claim_ai(markers)
    has_signal = claims_ai or selected_mark is not None or any_audio_detected
    invisible_hints = _video_invisible_watermark_hints(markers)
    caveats = ["No public local decoder can verify proprietary pixel watermarks such as video SynthID."]
    if invisible_hints:
        caveats.append(
            "Invisible watermark hints come from metadata declarations or provider policy, not a pixel detection. "
            "They do not confirm a surviving watermark or verified removal on this file."
        )
    if not check_visible:
        caveats.append("Visible video-mark detection was skipped.")
    if check_audio:
        caveats.append(
            "Audio stream inspection covers installed compatible public decoders only; "
            "proprietary audio marks such as SynthID are not checked locally."
        )
    if not has_signal and ("issuer" in markers or "actions" in markers):
        caveats.append(
            f"A C2PA manifest signed by {markers.get('issuer', 'an unregistered signer')} records "
            f"{markers.get('actions', 'no actions')} and no AI claim, as a platform re-encode does "
            "(YouTube re-signs every upload); that is unknown, not proof that the video is clean."
        )
    elif not has_signal and markers:
        caveats.append("Metadata does not establish AI origin; this is unknown, not proof that the video is clean.")
    elif not has_signal:
        caveats.append("No supported signal was found; absence is unknown, not proof that the video is clean.")
    recipe = _recommend_video_recipe(
        markers=markers,
        selected_mark=selected_mark,
        claims_ai=claims_ai,
        invisible_hints=invisible_hints,
        audio_streams=audio_tuple,
    )
    return VideoProvenanceReport(
        source=source_path,
        is_ai_generated=True if has_signal else None,
        confidence="high" if has_signal else "unknown",
        platform=(
            _VISIBLE_PLATFORM.get(selected_mark)
            if selected_mark is not None
            else _platform_from_video_metadata(markers)
            if claims_ai
            else None
        ),
        visible_mark=selected_mark,
        visible_detected_frames=detected_frames,
        total_frames=total_frames,
        has_ai_metadata=bool(markers),
        metadata_markers=markers,
        caveats=tuple(caveats),
        invisible_watermark_hints=invisible_hints,
        audio_streams=audio_tuple,
        recommended_recipe=recipe,
        visible_marks=selected_marks_tuple,
    )


def remove_video_metadata(
    source: str | Path,
    output: str | Path | None = None,
    *,
    keep_standard: bool = True,
    _detected_metadata: dict[str, str] | None = None,
) -> VideoMetadataResult:
    """Remove AI metadata without transcoding video or audio streams.

    The default output is ``<source_stem>_clean<source_suffix>``. A separate
    output is required so the operation never overwrites the original file.
    """
    from remove_ai_watermarks.metadata import get_ai_metadata, strip_and_verify

    source_path = _video_source(source)
    output_path = _video_output(source_path, output)
    detected = get_ai_metadata(source_path) if _detected_metadata is None else _detected_metadata
    written, remaining = strip_and_verify(source_path, output_path, keep_standard=keep_standard)
    return VideoMetadataResult(
        source=source_path,
        output=written,
        detected=detected,
        remaining=remaining,
    )


def _iter_decoded_bgr_frames(source_path: Path, *, max_frames: int | None = None) -> Iterator[Any]:
    from remove_ai_watermarks.video_decode import open_video_capture

    cap = open_video_capture(source_path)
    if not cap.isOpened():
        return
    try:
        idx = 0
        while max_frames is None or idx < max_frames:
            ok, raw = cap.read()
            if not ok or raw is None:
                break
            yield raw
            idx += 1
    finally:
        cap.release()


def _detect_source_bumper_cut(source_path: Path, fps: float) -> int | None:
    from remove_ai_watermarks.video_temporal import detect_ai_bumper_start

    return detect_ai_bumper_start(_iter_decoded_bgr_frames(source_path), fps=fps)


def remove_video_visible(
    source: str | Path,
    output: str | Path | None = None,
    *,
    mark: str = "auto",
    backend: str = "cv2",
    strip_metadata: bool = True,
    temporal_consistency: bool = True,
    trim_bumper: bool | Literal["auto"] = False,
    _metadata_markers: dict[str, str] | None = None,
) -> VideoVisibleResult:
    """Remove one or more supported visible AI marks (and optional trailing AI bumper) from a video."""
    _require_video_runtime()

    from remove_ai_watermarks.metadata import get_ai_metadata
    from remove_ai_watermarks.video_visible import encode_clean_video, scan_video_marks
    from remove_ai_watermarks.watermark_registry import resolve_backend

    candidate_marks, explicit_multi = _parse_mark_selector(mark)
    if backend not in {"auto", "cv2", "migan", "lama"}:
        raise ValueError("Unsupported fill backend; expected auto, cv2, migan, or lama")

    source_path = _video_source(source)
    output_path = _video_output(source_path, output, operation="visible watermark removal")
    markers = get_ai_metadata(source_path) if _metadata_markers is None else _metadata_markers

    scans = scan_video_marks(source_path, candidate_marks)
    base_scan = scans[candidate_marks[0]]
    total_frames = len(base_scan.detections)
    selected_all = _select_stable_visible_marks(scans, markers, candidate_marks, explicit_multi=explicit_multi)
    should_trim_bumper = trim_bumper is True or (trim_bumper == "auto" and (bool(markers) or bool(selected_all)))
    bumper_cut = _detect_source_bumper_cut(source_path, base_scan.fps) if should_trim_bumper else None
    trimmed_bumper_frames = (total_frames - bumper_cut) if bumper_cut is not None else 0

    if not selected_all and bumper_cut is None:
        return VideoVisibleResult(
            source=source_path,
            output=None,
            mark=mark,
            total_frames=total_frames,
            detected_frames=0,
            removed_frames=0,
            remaining_metadata=markers if strip_metadata else {},
            marks=(),
            trimmed_bumper_frames=0,
        )

    resolve_backend(backend)  # type: ignore[arg-type]
    encode_extra: dict[str, Any] = {}
    if bumper_cut is not None:
        encode_extra["trim_end_frame"] = bumper_cut

    if not selected_all:
        selected_marks_tuple: tuple[str, ...] = ()
        primary_mark = mark
        scan = base_scan
        merged_regions: list[Any] = [None] * total_frames
        padding_fraction = 0.12
        mask_style: Literal["box", "veo"] = "box"
        detected_frames = 0
    else:
        (
            selected_marks_tuple,
            primary_mark,
            scan,
            merged_regions,
            detected_frames,
            padding_fraction,
            mask_style,
            slot_pads,
            slot_styles,
        ) = _merge_selected_visible_marks(selected_all, total_frames)
        if slot_pads is not None:
            encode_extra["slot_padding_fractions"] = slot_pads
        if slot_styles is not None:
            encode_extra["slot_mask_styles"] = slot_styles

    removed_frames = encode_clean_video(
        source_path,
        output_path,
        scan,
        merged_regions,
        backend=backend,  # type: ignore[arg-type]
        strip_metadata=strip_metadata,
        padding_fraction=padding_fraction,
        mask_style=mask_style,
        temporal_consistency=temporal_consistency,
        **encode_extra,
    )
    remaining_metadata = get_ai_metadata(output_path) if strip_metadata else {}
    return VideoVisibleResult(
        source=source_path,
        output=output_path,
        mark=primary_mark,
        total_frames=total_frames,
        detected_frames=detected_frames,
        removed_frames=removed_frames,
        remaining_metadata=remaining_metadata,
        marks=selected_marks_tuple,
        trimmed_bumper_frames=trimmed_bumper_frames,
    )


def remove_video_region(
    source: str | Path,
    output: str | Path | None = None,
    *,
    regions: Sequence[tuple[int, int, int, int]],
    frame_range: tuple[int, int] | None = None,
    backend: str = "cv2",
    dilate: int = 3,
    strip_metadata: bool = True,
    temporal_consistency: bool = True,
    track: bool = False,
) -> VideoEraseResult:
    """Erase user-specified (x, y, w, h) region(s) across all or a slice of video frames."""
    import cv2

    from remove_ai_watermarks.metadata import get_ai_metadata
    from remove_ai_watermarks.video_decode import open_video_capture
    from remove_ai_watermarks.video_encoding import probe_video_timestamps
    from remove_ai_watermarks.video_visible import FrameLocalization, VideoScan, encode_clean_video
    from remove_ai_watermarks.watermark_registry import Backend, resolve_backend

    if not regions:
        raise ValueError("At least one (x, y, w, h) region is required")
    if dilate < 0:
        raise ValueError("dilate must be nonnegative")
    source_path = _video_source(source)
    output_path = _video_output(source_path, output, operation="video region erase")
    resolved_backend = resolve_backend(cast("Backend", backend))
    _require_video_runtime()

    boxes: list[tuple[int, int, int, int]] = []
    for box in regions:
        if len(box) != 4:
            raise ValueError("Each region must be an (x, y, w, h) 4-tuple")
        bx, by, bw, bh = (int(v) for v in box)
        if bw <= 0 or bh <= 0:
            raise ValueError("Region width and height must be positive")
        boxes.append((bx, by, bw, bh))

    capture = open_video_capture(source_path)
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not decode video: {source_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = 0
    raw_timestamps: list[float] = []
    try:
        if width <= 0 or height <= 0 or fps <= 0:
            raise RuntimeError(f"Video metadata is missing geometry or frame rate: {source_path}")
        while True:
            ok = capture.grab()
            if not ok:
                break
            raw_timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
            total_frames += 1
    finally:
        capture.release()
    if total_frames == 0:
        raise RuntimeError(f"Video contains no readable frames: {source_path}")

    clipped_boxes: list[tuple[int, int, int, int]] = []
    for bx, by, bw, bh in boxes:
        x0 = max(0, bx)
        y0 = max(0, by)
        x1 = min(width, bx + bw)
        y1 = min(height, by + bh)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Region {(bx, by, bw, bh)} lies completely outside {width}x{height} video frame")
        clipped_boxes.append((x0, y0, x1 - x0, y1 - y0))

    norm_range: tuple[int, int] | None = None
    if frame_range is not None:
        start_f, end_f = int(frame_range[0]), int(frame_range[1])
        if start_f < 0 or end_f < start_f or start_f >= total_frames:
            raise ValueError(f"Invalid frame_range {(start_f, end_f)} for {total_frames}-frame video")
        norm_range = (start_f, min(end_f, total_frames - 1))

    probed_ts = probe_video_timestamps(source_path)
    shared_ts = probed_ts if len(probed_ts) == total_frames else tuple(raw_timestamps)
    tuple_boxes = tuple(clipped_boxes)
    frame_entries: list[tuple[int, int, int, int] | tuple[tuple[int, int, int, int], ...] | None] = []
    detections: list[FrameLocalization] = []
    if track:
        from remove_ai_watermarks.video_temporal import track_video_regions

        t_start = norm_range[0] if norm_range is not None else 0
        t_end = norm_range[1] if norm_range is not None else None
        max_decode = (t_end + 1) if t_end is not None else None
        per_frame_boxes: list[tuple[tuple[int, int, int, int], ...] | None] = track_video_regions(
            _iter_decoded_bgr_frames(source_path, max_frames=max_decode),
            tuple_boxes,
            start_frame=t_start,
            end_frame=t_end,
        )
        if len(per_frame_boxes) < total_frames:
            per_frame_boxes.extend([None] * (total_frames - len(per_frame_boxes)))
        if len(per_frame_boxes) != total_frames:
            raise RuntimeError("Tracked frame count did not match probed video frame count")
    else:
        per_frame_boxes = [
            tuple_boxes if (norm_range is None or (norm_range[0] <= idx <= norm_range[1])) else None
            for idx in range(total_frames)
        ]

    for idx, t_regs in enumerate(per_frame_boxes):
        detections.append(FrameLocalization(idx, 1.0 if t_regs else 0.0, t_regs[0] if t_regs else None))
        if t_regs is None:
            frame_entries.append(None)
        elif len(t_regs) == 1:
            frame_entries.append(t_regs[0])
        else:
            frame_entries.append(t_regs)

    scan = VideoScan(width=width, height=height, fps=fps, detections=tuple(detections), timestamps=shared_ts)
    erased_frames = encode_clean_video(
        source_path,
        output_path,
        scan,
        frame_entries,
        backend=cast("Backend", resolved_backend),
        strip_metadata=strip_metadata,
        temporal_consistency=temporal_consistency,
        dilate_px=dilate,
        allow_unmasked_donors=bool(track and norm_range is None),
    )
    remaining = get_ai_metadata(output_path) if strip_metadata else {}
    return VideoEraseResult(
        source=source_path,
        output=output_path,
        regions=tuple_boxes,
        total_frames=total_frames,
        erased_frames=erased_frames,
        frame_range=norm_range,
        backend=resolved_backend,
        remaining_metadata=remaining,
        tracked=track,
    )


_BYTEDANCE_FALLBACK_TOKENS: tuple[str, ...] = (
    "bytedance",
    "byteplus",
    "seedance",
    "dreamina",
    "capcut",
    "doubao",
    "jimeng",
    "dola",
    "pippit",
    "volcengine",
    "spring (sg)",
)


def _has_2d_vae_overrides(
    *,
    noise_std: float,
    long_side: int,
    fps: float | None,
    batch_size: int,
    model: str,
    temporal_stabilize: bool,
    native_guide: bool = False,
) -> bool:
    return (
        noise_std != DEFAULT_VIDEO_SYNTHID_NOISE_STD
        or long_side != DEFAULT_VIDEO_SYNTHID_LONG_SIDE
        or fps != DEFAULT_VIDEO_SYNTHID_FPS
        or batch_size != 4
        or model != DEFAULT_VIDEO_SYNTHID_VAE
        or not temporal_stabilize
        or native_guide
    )


_BYTEDANCE_VISIBLE_MARKS = frozenset({"dreamina", "jimeng", "jimeng_pill", "seedance", "doubao", "dola"})


def _is_bytedance_video_provenance(
    metadata_markers: dict[str, str],
    visible_mark: str | None = None,
) -> bool:
    if visible_mark in _BYTEDANCE_VISIBLE_MARKS:
        return True
    if not metadata_markers:
        return False
    from remove_ai_watermarks._internal.tc260_producers import producer_for_code
    from remove_ai_watermarks.video_visible import has_bytedance_video_provenance

    if has_bytedance_video_provenance(metadata_markers):
        return True
    producer_row = producer_for_code(metadata_markers.get("aigc_producer", ""))
    if producer_row is not None and producer_row.vendor == "ByteDance":
        return True
    platform = (_platform_from_video_metadata(metadata_markers) or "").lower()
    if any(token in platform for token in _BYTEDANCE_FALLBACK_TOKENS):
        return True
    haystack = " ".join(f"{k} {v}" for k, v in metadata_markers.items()).lower()
    return any(token in haystack for token in _BYTEDANCE_FALLBACK_TOKENS)


def _normalize_video_profile(profile: str) -> str:
    return VIDEO_PROFILE_ALIASES.get(profile, profile)


def _resolve_video_invisible_profile(
    profile: VideoInvisibleProfile,
    source_path: Path,
    *,
    device: str,
    quality: Literal["delivery", "master"],
    noise_std: float,
    long_side: int,
    fps: float | None,
    batch_size: int,
    seed: int,
    model: str,
    temporal_stabilize: bool,
    native_guide: bool = False,
    wan_steps: int | None = None,
    detected_metadata: dict[str, str] | None = None,
    visible_mark: str | None = None,
    runtime: VideoVaeRuntime | VideoWanRuntime | None = None,
) -> Literal["synthid", "seedance-experimental", "wan21-synthid-experimental", "wan21-seedance-experimental"]:
    if profile != "auto":
        return cast(
            "Literal['synthid', 'seedance-experimental', 'wan21-synthid-experimental', 'wan21-seedance-experimental']",
            _normalize_video_profile(profile),
        )
    has_overrides = _has_2d_vae_overrides(
        noise_std=noise_std,
        long_side=long_side,
        fps=fps,
        batch_size=batch_size,
        model=model,
        temporal_stabilize=temporal_stabilize,
        native_guide=native_guide,
    )
    resolved: Literal["synthid", "seedance-experimental", "wan21-synthid-experimental", "wan21-seedance-experimental"]
    if has_overrides or source_path.suffix.lower() not in _REGENERATED_VIDEO_EXTENSIONS:
        if wan_steps is not None:
            raise ValueError(
                "wan_steps cannot be combined with 2D VAE SynthID tuning options or unsupported containers"
            )
        resolved = "synthid"
    else:
        from remove_ai_watermarks import video_wan

        can_wan = video_wan.can_run_wan_on_source(
            source_path, device=device, runtime=runtime, for_auto=wan_steps is None
        )
        if not can_wan and wan_steps is not None:
            raise ValueError("wan_steps requires an eligible CUDA or MPS Wan 2.1 device and runtime")
        if not can_wan and seed != 0:
            resolved = "synthid"
        else:
            from remove_ai_watermarks.metadata import get_ai_metadata

            markers = (
                detected_metadata
                if detected_metadata is not None
                else ({} if visible_mark in _BYTEDANCE_VISIBLE_MARKS else get_ai_metadata(source_path))
            )
            if _is_bytedance_video_provenance(markers, visible_mark):
                resolved = "wan21-seedance-experimental" if can_wan else "seedance-experimental"
            else:
                resolved = "wan21-synthid-experimental" if can_wan else "synthid"
    if quality != "delivery" and resolved == "synthid":
        raise ValueError("quality=master requires the enabled seedance-experimental or wan21-*-experimental profile")
    return resolved


def _has_capcut_pixel_watermark_provenance(
    metadata_markers: dict[str, str],
    visible_mark: str | None = None,
) -> bool:
    """Return True only for consumer CapCut Seedance surfaces that carry a pixel watermark."""
    if visible_mark == "seedance":
        return True
    if not metadata_markers:
        return False
    app_aigc = metadata_markers.get("app_aigc", "").casefold()
    if "capcut" in app_aigc:
        return True
    identity = " ".join(
        (
            metadata_markers.get("claim_generator", ""),
            metadata_markers.get("issuer", ""),
            metadata_markers.get("ai_tool", ""),
        )
    ).casefold()
    return "capcut" in identity and _video_markers_claim_ai(metadata_markers)


def video_invisible_target_cohort(
    detected_metadata: dict[str, str],
    visible_mark: str | None = None,
    *,
    explicit_vendor: str | None = None,
) -> Literal["synthid", "seedance", "meta", "declared_watermark"] | None:
    """Return the confirmed invisible-watermark cohort for auto pixel regeneration, or None.

    Only cohorts with verified pixel-domain watermarks (Google Veo SynthID,
    consumer CapCut Seedance, explicit vendor requests, or declared C2PA
    watermark soft bindings) trigger automatic pixel regeneration. Neutral C2PA
    signers (YouTube, non-AI CapCut/TikTok/Samsung/Topaz), local ComfyUI tags,
    and metadata/SEI-only AI models (Kling, Dreamina/BytePlus, Runway, Hailuo,
    Vidu, Wan, Grok, Sora) return None so ``all`` stays lossless on pixels.
    """
    if explicit_vendor in {"google", "openai"}:
        return "synthid"
    if explicit_vendor == "bytedance":
        return "seedance"
    if explicit_vendor == "meta":
        return "meta"
    if visible_mark == "veo":
        return "synthid"
    if _has_capcut_pixel_watermark_provenance(detected_metadata, visible_mark):
        return "seedance"
    if detected_metadata:
        if "synthid_watermark" in detected_metadata:
            return "synthid"
        if _video_invisible_watermark_hints(detected_metadata):
            return "declared_watermark"
    return None


def video_has_invisible_provenance(
    detected_metadata: dict[str, str],
    visible_mark: str | None = None,
    *,
    explicit_vendor: str | None = None,
) -> bool:
    """Return True when video provenance indicates a confirmed pixel-domain watermark cohort."""
    return (
        video_invisible_target_cohort(
            detected_metadata,
            visible_mark,
            explicit_vendor=explicit_vendor,
        )
        is not None
    )


def _validate_video_profile(
    profile: str,
    quality: str,
    *,
    enabled: bool | Literal["auto"] = True,
    noise_std: float = DEFAULT_VIDEO_SYNTHID_NOISE_STD,
    long_side: int = DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    fps: float | None = DEFAULT_VIDEO_SYNTHID_FPS,
    batch_size: int = 4,
    seed: int = 0,
    model: str = DEFAULT_VIDEO_SYNTHID_VAE,
    temporal_stabilize: bool = True,
    native_guide: bool = False,
    wan_steps: int | None = None,
) -> None:
    """Reject incompatible profile options before processing any stage."""
    if profile not in VIDEO_INVISIBLE_PROFILES:
        raise ValueError(
            "profile must be auto, synthid, seedance, wan21-synthid, or wan21-seedance "
            "(legacy *-experimental aliases are also accepted)"
        )
    profile = _normalize_video_profile(profile)
    if quality not in {"delivery", "master"}:
        raise ValueError("quality must be delivery or master")
    if profile not in {"auto", "synthid"} and not enabled:
        raise ValueError("Selecting an invisible profile requires include_invisible=True (--invisible)")
    if wan_steps is not None:
        if wan_steps < 1:
            raise ValueError("wan_steps must be >= 1")
        if not enabled:
            raise ValueError("wan_steps requires include_invisible=True (--invisible)")
        if profile in {"synthid", "seedance-experimental"}:
            raise ValueError(
                "wan_steps is only supported for Wan 2.1 profiles (wan21-synthid, wan21-seedance, or auto)"
            )
    fixed_quality_profiles = {
        "auto",
        "seedance-experimental",
        "wan21-synthid-experimental",
        "wan21-seedance-experimental",
    }
    if quality != "delivery" and (not enabled or profile not in fixed_quality_profiles):
        raise ValueError("quality=master requires the enabled seedance-experimental or wan21-*-experimental profile")
    has_overrides = _has_2d_vae_overrides(
        noise_std=noise_std,
        long_side=long_side,
        fps=fps,
        batch_size=batch_size,
        model=model,
        temporal_stabilize=temporal_stabilize,
        native_guide=native_guide,
    )
    if profile == "seedance-experimental" and (has_overrides or seed != 0):
        raise ValueError("Seedance uses a fixed recipe; SynthID tuning options must retain their defaults")
    if profile in WAN_EXPERIMENTAL_PROFILES and has_overrides:
        raise ValueError(
            "Wan 2.1 profiles use fixed native geometry and frame timing; "
            "2D VAE SynthID tuning options must retain their defaults"
        )
    if wan_steps is not None and has_overrides:
        raise ValueError("wan_steps cannot be combined with 2D VAE SynthID tuning options")


def remove_video_all(
    source: str | Path,
    output: str | Path | None = None,
    *,
    mark: str = "auto",
    backend: str = "cv2",
    temporal_consistency: bool = True,
    include_invisible: bool | Literal["auto"] = False,
    invisible_profile: VideoInvisibleProfile = "auto",
    invisible_quality: Literal["delivery", "master"] = "delivery",
    noise_std: float = DEFAULT_VIDEO_SYNTHID_NOISE_STD,
    long_side: int = DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    fps: float | None = DEFAULT_VIDEO_SYNTHID_FPS,
    batch_size: int = 4,
    seed: int = 0,
    model: str = DEFAULT_VIDEO_SYNTHID_VAE,
    device: str = "auto",
    temporal_stabilize: bool = True,
    native_guide: bool = False,
    wan_steps: int | None = None,
    include_audio_invisible: bool = False,
    audio_scheme: AudioScheme = "auto",
    audio_method: AudioMethod = "auto",
    audio_bandwidth_kbps: float = 24.0,
    audio_vocoder_device: AudioDevice = "auto",
    audio_force: bool = False,
    audio_verify: bool = True,
    audio_allow_unverified: bool = False,
    audio_allow_positive: bool = False,
    audio_allow_quality_override: bool = False,
    audio_vendor: AudioVendor | None = None,
    audio_streams: Sequence[int] | None = None,
    trim_bumper: bool | Literal["auto"] = False,
    max_duration_seconds: float | None = None,
    max_frame_pixels: int | None = None,
    require_gpu: bool = False,
    progress_callback: Callable[[str, float], None] | None = None,
    _invisible_runtime: VideoVaeRuntime | VideoWanRuntime | None = None,
    _detected_metadata: dict[str, str] | None = None,
    _runtime_getter: Callable[[Literal["vae", "wan"]], VideoVaeRuntime | VideoWanRuntime] | None = None,
) -> VideoAllResult:
    """Run visible, metadata, and optional video-pixel and audio-watermark processing.

    The default path removes a stable visible provider mark when present and
    always strips verified AI metadata. It writes a same-container passthrough
    when neither signal is present, giving product callers one predictable
    output contract. ``include_invisible=True`` additionally runs lossy VAE
    regeneration with the oracle-certified default SynthID profile (with
    motion-compensated temporal stabilization enabled by default) or the
    explicitly selected experimental Seedance profile. ``include_audio_invisible=True``
    additionally inspects all audio streams with compatible public decoders and
    cleans any watermarked tracks while preserving clean audio tracks unchanged.
    """
    import shutil

    from remove_ai_watermarks.metadata import get_ai_metadata
    from remove_ai_watermarks.video_encoding import atomic_video_output

    _validate_video_profile(
        invisible_profile,
        invisible_quality,
        enabled=include_invisible,
        noise_std=noise_std,
        long_side=long_side,
        fps=fps,
        batch_size=batch_size,
        seed=seed,
        model=model,
        temporal_stabilize=temporal_stabilize,
        native_guide=native_guide,
        wan_steps=wan_steps,
    )
    if max_duration_seconds is not None and max_duration_seconds <= 0.0:
        raise ValueError("max_duration_seconds must be positive")
    if max_frame_pixels is not None and max_frame_pixels <= 0:
        raise ValueError("max_frame_pixels must be positive")
    if progress_callback is not None:
        progress_callback("preflight", 0.05)
    source_path = _video_source(source)
    output_path = _video_output(source_path, output, operation="complete cleaning")
    if max_frame_pixels is not None or max_duration_seconds is not None:
        from remove_ai_watermarks.video_encoding import probe_video_timestamps
        from remove_ai_watermarks.video_invisible import probe_video_geometry

        w_probe, h_probe, fps_probe = probe_video_geometry(source_path)
        if max_frame_pixels is not None and w_probe * h_probe > max_frame_pixels:
            area = w_probe * h_probe
            raise ValueError(
                f"Video frame area {w_probe}x{h_probe} ({area} px) exceeds max_frame_pixels={max_frame_pixels}"
            )
        if max_duration_seconds is not None:
            ts_probe = probe_video_timestamps(source_path)
            est_dur = (
                (ts_probe[-1] - ts_probe[0]) + (1.0 / max(fps_probe, 1e-6))
                if len(ts_probe) >= 2
                else (len(ts_probe) / max(fps_probe, 1e-6))
            )
            if est_dur > max_duration_seconds:
                raise ValueError(
                    f"Video duration {est_dur:.2f}s exceeds max_duration_seconds={max_duration_seconds:.2f}s"
                )
    explicit_invisible = include_invisible is True
    auto_invisible = include_invisible == "auto"
    if explicit_invisible and source_path.suffix.lower() not in _REGENERATED_VIDEO_EXTENSIONS:
        raise ValueError(f"Video invisible regeneration requires one of: {_EXPECTED_REGENERATED_VIDEO_EXTENSIONS}")
    from remove_ai_watermarks.audio import normalize_audio_vendor

    preflight_befores: tuple[AudioProvenanceReport, ...] = ()
    explicit_audio_vendor = normalize_audio_vendor(audio_vendor)
    eff_include_audio_invisible = include_audio_invisible or explicit_audio_vendor is not None
    if eff_include_audio_invisible:
        if source_path.suffix.lower() not in {".mp4", ".mov", ".m4v", ".mkv", ".webm"}:
            raise ValueError("Video audio watermark processing requires MP4, MOV, M4V, MKV, or WebM input")
        from remove_ai_watermarks import video_audio as _video_audio_mod

        preflight_befores = _video_audio_mod.preflight_video_audio_workload(
            source_path,
            force=audio_force,
            verify=audio_verify,
            method=audio_method,
            bandwidth_kbps=audio_bandwidth_kbps,
            scheme=audio_scheme,
            vocoder_device=audio_vocoder_device,
            allow_unverified=audio_allow_unverified,
            allow_positive=audio_allow_positive,
            allow_quality_override=audio_allow_quality_override,
            vendor=audio_vendor,
            streams=audio_streams,
            _metadata_markers=_detected_metadata,
        )
    _require_video_runtime()
    detected_metadata = _detected_metadata if _detected_metadata is not None else get_ai_metadata(source_path)
    audio_status: VideoAudioStatus | None = None
    audio_processed_streams: tuple[int, ...] = ()
    audio_verification_scope: tuple[str, ...] = ()
    audio_quality_passed: bool | None = None

    with (
        atomic_video_output(output_path) as staged_final,
        TemporaryDirectory(prefix=f".{source_path.stem}-video-all-", dir=source_path.parent) as temp_dir,
    ):
        stage_target = (
            Path(temp_dir) / f"visual-final{source_path.suffix}" if eff_include_audio_invisible else staged_final
        )
        resolved_invisible_profile: VideoInvisibleProfile | None = None
        invisible_applied = False
        can_fuse = (
            explicit_invisible
            and not trim_bumper
            and remove_video_visible is _ORIG_REMOVE_VIDEO_VISIBLE
            and remove_video_invisible is _ORIG_REMOVE_VIDEO_INVISIBLE
        )
        if can_fuse:
            from remove_ai_watermarks.video_encoding import HIGH_BIT_DEPTH_PIXEL_FORMATS, probe_video_encode_profile
            from remove_ai_watermarks.video_visible import encode_clean_video, iter_clean_video_frames, scan_video_marks
            from remove_ai_watermarks.watermark_registry import resolve_backend

            candidate_marks, explicit_multi = _parse_mark_selector(mark)
            resolve_backend(backend)  # type: ignore[arg-type]
            scans = scan_video_marks(source_path, candidate_marks)
            base_scan = scans[candidate_marks[0]]
            total_frames = len(base_scan.detections)
            selected_all = _select_stable_visible_marks(
                scans, detected_metadata, candidate_marks, explicit_multi=explicit_multi
            )
            vis_mark = selected_all[0][0] if selected_all else None
            resolved_invisible_profile = _resolve_video_invisible_profile(
                invisible_profile,
                source_path,
                device=device,
                quality=invisible_quality,
                noise_std=noise_std,
                long_side=long_side,
                fps=fps,
                batch_size=batch_size,
                seed=seed,
                model=model,
                temporal_stabilize=temporal_stabilize,
                native_guide=native_guide,
                wan_steps=wan_steps,
                detected_metadata=detected_metadata,
                visible_mark=vis_mark,
                runtime=_invisible_runtime,
            )
            if require_gpu and resolved_invisible_profile == "synthid":
                raise RuntimeError(
                    "require_gpu=True refuses 512p 2D CPU fallback; run on a CUDA/MPS GPU or disable require_gpu"
                )
            if progress_callback is not None:
                progress_callback("invisible", 0.5)
            active_runtime = _invisible_runtime
            if active_runtime is None and _runtime_getter is not None:
                active_runtime = _runtime_getter(
                    "wan" if resolved_invisible_profile in WAN_EXPERIMENTAL_PROFILES else "vae"
                )
            if not selected_all:
                invisible_result = remove_video_invisible(
                    source_path,
                    stage_target,
                    profile=resolved_invisible_profile,
                    quality=invisible_quality,
                    noise_std=noise_std,
                    long_side=long_side,
                    fps=fps,
                    batch_size=batch_size,
                    seed=seed,
                    model=model,
                    device=device,
                    temporal_stabilize=temporal_stabilize,
                    native_guide=native_guide,
                    wan_steps=wan_steps,
                    _runtime=active_runtime,
                )
                remaining_metadata = invisible_result.remaining_metadata
                visible_result = VideoVisibleResult(
                    source=source_path,
                    output=None,
                    mark=mark,
                    total_frames=total_frames,
                    detected_frames=0,
                    removed_frames=0,
                    remaining_metadata={},
                    marks=(),
                    trimmed_bumper_frames=0,
                )
            else:
                (
                    selected_marks_tuple,
                    primary_mark,
                    scan,
                    merged_regions,
                    detected_frames,
                    padding_fraction,
                    mask_style,
                    slot_pads,
                    slot_styles,
                ) = _merge_selected_visible_marks(selected_all, total_frames)

                src_prof = probe_video_encode_profile(source_path)
                high_depth = src_prof.pixel_format in HIGH_BIT_DEPTH_PIXEL_FORMATS
                if resolved_invisible_profile == "seedance-experimental":
                    lossless_vis = Path(temp_dir) / f"visible-lossless{source_path.suffix}"
                    rem_cnt = encode_clean_video(
                        source_path,
                        lossless_vis,
                        scan,
                        merged_regions,
                        backend=backend,  # type: ignore[arg-type]
                        strip_metadata=True,
                        padding_fraction=padding_fraction,
                        mask_style=mask_style,
                        slot_padding_fractions=slot_pads,
                        slot_mask_styles=slot_styles,
                        temporal_consistency=temporal_consistency,
                        crf=0,
                    )
                    invisible_result = remove_video_invisible(
                        lossless_vis,
                        stage_target,
                        profile=resolved_invisible_profile,
                        quality=invisible_quality,
                        noise_std=noise_std,
                        long_side=long_side,
                        fps=fps,
                        batch_size=batch_size,
                        seed=seed,
                        model=model,
                        device=device,
                        temporal_stabilize=temporal_stabilize,
                        _runtime=active_runtime,
                    )
                else:
                    rem_cnt = 0

                    def _fused_clean_stream() -> Iterator[Any]:
                        nonlocal rem_cnt
                        for frm, was_rem in iter_clean_video_frames(
                            source_path,
                            scan,
                            merged_regions,
                            backend=backend,  # type: ignore[arg-type]
                            padding_fraction=padding_fraction,
                            mask_style=mask_style,
                            slot_padding_fractions=slot_pads,
                            slot_mask_styles=slot_styles,
                            temporal_consistency=temporal_consistency,
                            high_depth=high_depth,
                        ):
                            if was_rem:
                                rem_cnt += 1
                            yield frm

                    invisible_result = remove_video_invisible(
                        source_path,
                        stage_target,
                        profile=resolved_invisible_profile,
                        quality=invisible_quality,
                        noise_std=noise_std,
                        long_side=long_side,
                        fps=fps,
                        batch_size=batch_size,
                        seed=seed,
                        model=model,
                        device=device,
                        temporal_stabilize=temporal_stabilize,
                        native_guide=native_guide,
                        wan_steps=wan_steps,
                        _runtime=active_runtime,
                        _frame_stream=_fused_clean_stream(),
                    )
                remaining_metadata = invisible_result.remaining_metadata
                visible_result = VideoVisibleResult(
                    source=source_path,
                    output=stage_target,
                    mark=primary_mark,
                    total_frames=total_frames,
                    detected_frames=detected_frames,
                    removed_frames=rem_cnt if rem_cnt else detected_frames,
                    remaining_metadata=remaining_metadata,
                    marks=selected_marks_tuple,
                    trimmed_bumper_frames=0,
                )
            invisible_applied = True
        else:
            visible_output = (
                Path(temp_dir) / f"visible{source_path.suffix}"
                if (explicit_invisible or auto_invisible)
                else stage_target
            )
            vis_kwargs: dict[str, Any] = {
                "mark": mark,
                "backend": backend,
                "strip_metadata": True,
                "temporal_consistency": temporal_consistency,
                "_metadata_markers": detected_metadata,
            }
            if trim_bumper:
                vis_kwargs["trim_bumper"] = trim_bumper
            visible_result = remove_video_visible(source_path, visible_output, **vis_kwargs)
            current_source = visible_result.output or source_path
            detected_vis_mark = visible_result.mark if visible_result.output is not None else None
            run_invisible_now = explicit_invisible
            if (
                auto_invisible
                and video_has_invisible_provenance(detected_metadata, detected_vis_mark)
                and current_source.suffix.lower() in _REGENERATED_VIDEO_EXTENSIONS
            ):
                from remove_ai_watermarks import video_invisible as _video_inv_mod

                if _video_inv_mod.is_available():
                    run_invisible_now = True
                else:
                    log.warning(
                        "AI video provenance detected on %s; skipping invisible regeneration "
                        "(install remove-ai-watermarks[video,diffusion])",
                        source_path.name,
                    )

            if run_invisible_now:
                resolved_invisible_profile = _resolve_video_invisible_profile(
                    invisible_profile,
                    current_source,
                    device=device,
                    quality=invisible_quality,
                    noise_std=noise_std,
                    long_side=long_side,
                    fps=fps,
                    batch_size=batch_size,
                    seed=seed,
                    model=model,
                    temporal_stabilize=temporal_stabilize,
                    native_guide=native_guide,
                    wan_steps=wan_steps,
                    detected_metadata=detected_metadata,
                    visible_mark=detected_vis_mark,
                    runtime=_invisible_runtime,
                )
                if require_gpu and resolved_invisible_profile == "synthid":
                    raise RuntimeError(
                        "require_gpu=True refuses 512p 2D CPU fallback; run on a CUDA/MPS GPU or disable require_gpu"
                    )
                if progress_callback is not None:
                    progress_callback("invisible", 0.55)
                active_runtime = _invisible_runtime
                if active_runtime is None and _runtime_getter is not None:
                    active_runtime = _runtime_getter(
                        "wan" if resolved_invisible_profile in WAN_EXPERIMENTAL_PROFILES else "vae"
                    )
                invisible_result = remove_video_invisible(
                    current_source,
                    stage_target,
                    profile=resolved_invisible_profile,
                    quality=invisible_quality,
                    noise_std=noise_std,
                    long_side=long_side,
                    fps=fps,
                    batch_size=batch_size,
                    seed=seed,
                    model=model,
                    device=device,
                    temporal_stabilize=temporal_stabilize,
                    native_guide=native_guide,
                    wan_steps=wan_steps,
                    _runtime=active_runtime,
                )
                remaining_metadata = invisible_result.remaining_metadata
                invisible_applied = True
            elif visible_result.output is None:
                metadata_result = remove_video_metadata(
                    source_path,
                    stage_target,
                    _detected_metadata=detected_metadata,
                )
                remaining_metadata = metadata_result.remaining
            else:
                if visible_result.output != stage_target:
                    shutil.move(str(visible_result.output), str(stage_target))
                remaining_metadata = visible_result.remaining_metadata

        if eff_include_audio_invisible:
            if progress_callback is not None:
                progress_callback("audio", 0.8)
            from remove_ai_watermarks import video_audio

            streams = video_audio.stream_inventory(stage_target)
            has_audio = any(s.get("codec_type") == "audio" for s in streams)
            published_by_audio = False
            wm_state: Literal["unverified", "not_detected", "detected"] = "unverified"
            if has_audio:
                reusable_befores: Sequence[AudioProvenanceReport] | None = None
                if preflight_befores:
                    try:
                        src_a = [int(rep.stream.index) for rep in preflight_befores]
                        dst_a = [int(s["index"]) for s in streams if s.get("codec_type") == "audio"]
                        if len(src_a) == len(dst_a):
                            for sa, da in zip(src_a, dst_a, strict=True):
                                video_audio.verify_stream_packets(source_path, stage_target, sa, output_index=da)
                            reusable_befores = preflight_befores
                    except Exception:
                        reusable_befores = None
                effective_audio_vendor: AudioVendor | None = audio_vendor
                if explicit_audio_vendor is None and visible_result.output is not None and visible_result.mark == "veo":
                    effective_audio_vendor = "google"
                audio_res = video_audio.remove_video_audio_all_streams(
                    stage_target,
                    staged_final,
                    force=audio_force,
                    verify=audio_verify,
                    method=audio_method,
                    bandwidth_kbps=audio_bandwidth_kbps,
                    scheme=audio_scheme,
                    vocoder_device=audio_vocoder_device,
                    allow_unverified=audio_allow_unverified,
                    allow_positive=audio_allow_positive,
                    allow_quality_override=audio_allow_quality_override,
                    vendor=effective_audio_vendor,
                    streams=audio_streams,
                    _befores=reusable_befores,
                    _metadata_markers=detected_metadata,
                )
                seen_schemes: set[str] = set()
                for s in audio_res.streams:
                    for det in getattr(s.before, "detections", ()):
                        seen_schemes.add(det.scheme)
                    after_obj = getattr(s, "after", None)
                    if after_obj is not None:
                        for det in getattr(after_obj, "detections", ()):
                            seen_schemes.add(det.scheme)
                audio_verification_scope = tuple(sorted(seen_schemes))
                audio_quality_passed = getattr(audio_res, "quality_passed", None)
                if audio_res.skipped or audio_res.output is None:
                    all_neg = bool(audio_res.streams) and all(
                        s.before.watermark_status == "not_detected" for s in audio_res.streams
                    )
                    wm_state = "not_detected" if all_neg else "unverified"
                else:
                    published_by_audio = True
                    remaining_metadata = get_ai_metadata(staged_final)
                    unprocessed_detected = any(
                        getattr(s, "skipped", False) and s.before.watermark_status == "detected"
                        for s in audio_res.streams
                    )
                    unprocessed_unverified = any(
                        getattr(s, "skipped", False) and s.before.watermark_status != "not_detected"
                        for s in audio_res.streams
                    )
                    if audio_res.verified_removed is False or unprocessed_detected:
                        wm_state = "detected"
                    elif audio_res.verified_removed is True and not unprocessed_unverified:
                        wm_state = "not_detected"
                    else:
                        wm_state = "unverified"
                    audio_status = VideoAudioStatus(stream_action="processed", watermark_status=wm_state)
                    audio_processed_streams = audio_res.processed_ordinals
            if not published_by_audio:
                shutil.copyfile(stage_target, staged_final)
                audio_status = VideoAudioStatus(
                    stream_action="copied_if_present" if has_audio else "absent",
                    watermark_status=wm_state,
                )

        if remaining_metadata:
            raise RuntimeError(
                f"{len(remaining_metadata)} AI metadata marker(s) survived the complete pipeline: "
                f"{sorted(remaining_metadata)}"
            )

    if progress_callback is not None:
        progress_callback("complete", 1.0)
    return VideoAllResult(
        source=source_path,
        output=output_path,
        visible_mark=visible_result.mark if visible_result.output is not None else None,
        total_frames=visible_result.total_frames,
        visible_detected_frames=visible_result.detected_frames,
        visible_removed_frames=visible_result.removed_frames,
        detected_metadata=detected_metadata,
        remaining_metadata=remaining_metadata,
        invisible_removed=invisible_applied,
        invisible_profile=resolved_invisible_profile,
        audio_status=audio_status,
        audio_processed_streams=audio_processed_streams,
        audio_verification_scope=audio_verification_scope,
        audio_quality_passed=audio_quality_passed,
        visible_marks=getattr(visible_result, "marks", ()) if visible_result.output is not None else (),
        trimmed_bumper_frames=getattr(visible_result, "trimmed_bumper_frames", 0),
    )


def remove_video_batch(
    directory: str | Path,
    output_directory: str | Path | None = None,
    *,
    mode: Literal["all", "visible", "metadata"] = "all",
    mark: str = "auto",
    backend: str = "cv2",
    temporal_consistency: bool = True,
    include_invisible: bool | Literal["auto"] = False,
    invisible_profile: VideoInvisibleProfile = "auto",
    invisible_quality: Literal["delivery", "master"] = "delivery",
    noise_std: float = DEFAULT_VIDEO_SYNTHID_NOISE_STD,
    long_side: int = DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    fps: float | None = DEFAULT_VIDEO_SYNTHID_FPS,
    batch_size: int = 4,
    seed: int = 0,
    model: str = DEFAULT_VIDEO_SYNTHID_VAE,
    device: str = "auto",
    temporal_stabilize: bool = True,
    native_guide: bool = False,
    wan_steps: int | None = None,
    include_audio_invisible: bool = False,
    audio_scheme: AudioScheme = "auto",
    audio_method: AudioMethod = "auto",
    audio_bandwidth_kbps: float = 24.0,
    audio_vocoder_device: AudioDevice = "auto",
    audio_force: bool = False,
    audio_verify: bool = True,
    audio_allow_unverified: bool = False,
    audio_allow_positive: bool = False,
    audio_allow_quality_override: bool = False,
    audio_vendor: AudioVendor | None = None,
    audio_streams: Sequence[int] | None = None,
    trim_bumper: bool | Literal["auto"] = False,
) -> VideoBatchResult:
    """Process every supported video in one directory.

    Files are processed sequentially so model and ffmpeg resource use stays
    bounded. Per-file failures are returned in ``items`` and do not discard
    successful outputs. Visible-only no-op files are copied byte-for-byte so the
    output directory remains complete.
    """
    import shutil

    from remove_ai_watermarks.video_encoding import atomic_video_output

    _validate_video_profile(
        invisible_profile,
        invisible_quality,
        enabled=include_invisible,
        noise_std=noise_std,
        long_side=long_side,
        fps=fps,
        batch_size=batch_size,
        seed=seed,
        model=model,
        temporal_stabilize=temporal_stabilize,
        native_guide=native_guide,
        wan_steps=wan_steps,
    )
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Video directory does not exist: {directory_path}")
    if not directory_path.is_dir():
        raise ValueError(f"Video batch source must be a directory: {directory_path}")
    if mode not in {"all", "visible", "metadata"}:
        raise ValueError("Unsupported video batch mode; expected all, visible, or metadata")
    if mark not in {"auto", *VIDEO_VISIBLE_MARKS}:
        raise ValueError(f"Unsupported visible video mark; expected {_EXPECTED_VIDEO_MARKS}")
    if backend not in {"auto", "cv2", "migan", "lama"}:
        raise ValueError("Unsupported fill backend; expected auto, cv2, migan, or lama")
    from remove_ai_watermarks.audio import normalize_audio_vendor

    explicit_batch_vendor = normalize_audio_vendor(audio_vendor)
    eff_batch_audio_invisible = include_audio_invisible or explicit_batch_vendor is not None
    if include_invisible and mode != "all":
        raise ValueError("The invisible video stage is available only in all mode")
    if eff_batch_audio_invisible and mode != "all":
        raise ValueError("The audio watermark stage is available only in all mode")
    if eff_batch_audio_invisible:
        from remove_ai_watermarks import audio as _audio_mod

        _audio_mod.validate_audio_arguments(
            method=audio_method,
            scheme=audio_scheme,
            verify=audio_verify,
            force=audio_force,
            bandwidth_kbps=audio_bandwidth_kbps,
            vocoder_device=audio_vocoder_device,
            allow_unverified=audio_allow_unverified or explicit_batch_vendor is not None,
            vendor=audio_vendor,
        )
    if mode != "metadata":
        _require_video_runtime()

    output_path = (
        Path(output_directory)
        if output_directory is not None
        else directory_path.parent / f"{directory_path.name}_clean"
    )
    if output_path.resolve() == directory_path.resolve():
        raise ValueError("Video batch output directory must differ from the source directory")
    sources = tuple(
        path
        for path in sorted(directory_path.iterdir(), key=lambda candidate: candidate.name.lower())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if sources:
        output_path.mkdir(parents=True, exist_ok=True)
    items: list[VideoBatchItem] = []
    runtimes: dict[Literal["vae", "wan"], VideoVaeRuntime | VideoWanRuntime] = {}
    runtime_errors: dict[Literal["vae", "wan"], str] = {}

    def _get_batch_runtime(kind: Literal["vae", "wan"]) -> VideoVaeRuntime | VideoWanRuntime:
        if kind in runtimes:
            return runtimes[kind]
        if kind in runtime_errors:
            raise RuntimeError(runtime_errors[kind])
        try:
            if kind == "wan":
                from remove_ai_watermarks.video_wan import load_video_wan_runtime

                loaded: VideoVaeRuntime | VideoWanRuntime = load_video_wan_runtime(device=device)
            else:
                from remove_ai_watermarks.video_invisible import load_video_vae_runtime

                loaded = load_video_vae_runtime(model=model, device=device)
        except Exception as exc:
            runtime_errors[kind] = str(exc)
            raise
        runtimes[kind] = loaded
        return loaded

    for source_path in sources:
        item_output = output_path / source_path.name
        try:
            if mode == "all":
                selected_runtime: VideoVaeRuntime | VideoWanRuntime | None = None
                lazy_getter: Callable[[Literal["vae", "wan"]], VideoVaeRuntime | VideoWanRuntime] | None = None
                if include_invisible:
                    _video_source(source_path)
                    if invisible_profile == "auto" or include_invisible == "auto":
                        if source_path.suffix.lower() in _REGENERATED_VIDEO_EXTENSIONS:
                            lazy_getter = _get_batch_runtime
                    else:
                        norm_prof = _normalize_video_profile(invisible_profile)
                        needs_wan = norm_prof in WAN_EXPERIMENTAL_PROFILES
                        if source_path.suffix.lower() in _REGENERATED_VIDEO_EXTENSIONS:
                            selected_runtime = _get_batch_runtime("wan" if needs_wan else "vae")
                all_result = remove_video_all(
                    source_path,
                    item_output,
                    mark=mark,
                    backend=backend,
                    temporal_consistency=temporal_consistency,
                    include_invisible=include_invisible,
                    invisible_profile=invisible_profile,
                    invisible_quality=invisible_quality,
                    noise_std=noise_std,
                    long_side=long_side,
                    fps=fps,
                    batch_size=batch_size,
                    seed=seed,
                    model=model,
                    device=device,
                    temporal_stabilize=temporal_stabilize,
                    native_guide=native_guide,
                    wan_steps=wan_steps,
                    include_audio_invisible=include_audio_invisible,
                    audio_scheme=audio_scheme,
                    audio_method=audio_method,
                    audio_bandwidth_kbps=audio_bandwidth_kbps,
                    audio_vocoder_device=audio_vocoder_device,
                    audio_force=audio_force,
                    audio_verify=audio_verify,
                    audio_allow_unverified=audio_allow_unverified,
                    audio_allow_positive=audio_allow_positive,
                    audio_allow_quality_override=audio_allow_quality_override,
                    audio_vendor=audio_vendor,
                    audio_streams=audio_streams,
                    trim_bumper=trim_bumper,
                    _invisible_runtime=selected_runtime,
                    _runtime_getter=lazy_getter,
                )
                if all_result.remaining_metadata:
                    raise RuntimeError(
                        f"{len(all_result.remaining_metadata)} AI metadata marker(s) survived the complete pipeline"
                    )
                items.append(
                    VideoBatchItem(
                        source=source_path,
                        output=all_result.output,
                        mode=mode,
                        changed=bool(
                            all_result.visible_mark
                            or all_result.detected_metadata
                            or all_result.invisible_removed
                            or all_result.audio.stream_action == "processed"
                        ),
                        visible_mark=all_result.visible_mark,
                        invisible_removed=all_result.invisible_removed,
                        audio_status=all_result.audio_status,
                        audio_verification_scope=all_result.audio_verification_scope,
                        audio_quality_passed=all_result.audio_quality_passed,
                        visible_marks=all_result.visible_marks,
                    )
                )
            elif mode == "visible":
                visible_result = remove_video_visible(
                    source_path,
                    item_output,
                    mark=mark,
                    backend=backend,
                    strip_metadata=False,
                    temporal_consistency=temporal_consistency,
                )
                if visible_result.output is None:
                    with atomic_video_output(item_output) as temporary_output:
                        shutil.copyfile(source_path, temporary_output)
                items.append(
                    VideoBatchItem(
                        source=source_path,
                        output=item_output,
                        mode=mode,
                        changed=visible_result.output is not None,
                        visible_mark=visible_result.mark if visible_result.output is not None else None,
                        invisible_removed=False,
                        visible_marks=getattr(visible_result, "marks", ()) if visible_result.output is not None else (),
                    )
                )
            else:
                metadata_result = remove_video_metadata(source_path, item_output)
                if metadata_result.remaining:
                    raise RuntimeError(
                        f"{len(metadata_result.remaining)} AI metadata marker(s) survived metadata removal"
                    )
                items.append(
                    VideoBatchItem(
                        source=source_path,
                        output=metadata_result.output,
                        mode=mode,
                        changed=bool(metadata_result.detected),
                        visible_mark=None,
                        invisible_removed=False,
                    )
                )
        except Exception as exc:
            items.append(
                VideoBatchItem(
                    source=source_path,
                    output=None,
                    mode=mode,
                    changed=False,
                    visible_mark=None,
                    invisible_removed=False,
                    error=str(exc),
                )
            )

    return VideoBatchResult(
        directory=directory_path,
        output_directory=output_path,
        items=tuple(items),
    )


def remove_video_invisible(
    source: str | Path,
    output: str | Path | None = None,
    *,
    profile: VideoInvisibleProfile = "auto",
    quality: Literal["delivery", "master"] = "delivery",
    noise_std: float = DEFAULT_VIDEO_SYNTHID_NOISE_STD,
    long_side: int = DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    fps: float | None = DEFAULT_VIDEO_SYNTHID_FPS,
    batch_size: int = 4,
    seed: int = 0,
    model: str = DEFAULT_VIDEO_SYNTHID_VAE,
    device: str = "auto",
    temporal_stabilize: bool = True,
    native_guide: bool = False,
    wan_steps: int | None = None,
    _runtime: VideoVaeRuntime | VideoWanRuntime | None = None,
    _frame_stream: Iterable[Any] | None = None,
) -> VideoInvisibleResult:
    """Regenerate video pixels through SynthID, Seedance, or CUDA Wan 2.1 profiles.

    The function also strips source metadata during the transcode. The default
    SynthID profile was calibrated against a provider oracle and applies
    motion-compensated temporal stabilization by default; Seedance and the
    CUDA 3D Wan 2.1 profiles are experimental. This call performs no per-file
    verification. Source audio is copied if present and its watermark status
    remains unverified.
    """
    from remove_ai_watermarks.metadata import get_ai_metadata

    _validate_video_profile(
        profile,
        quality,
        noise_std=noise_std,
        long_side=long_side,
        fps=fps,
        batch_size=batch_size,
        seed=seed,
        model=model,
        temporal_stabilize=temporal_stabilize,
        native_guide=native_guide,
        wan_steps=wan_steps,
    )
    source_path = _video_source(source)
    clean_output = Path(output) if output is not None else source_path.with_stem(source_path.stem + "_clean")
    requested_profile = profile
    is_legacy_experimental = requested_profile.endswith("-experimental")
    if profile == "auto":
        profile = _resolve_video_invisible_profile(
            "auto",
            source_path,
            device=device,
            quality=quality,
            noise_std=noise_std,
            long_side=long_side,
            fps=fps,
            batch_size=batch_size,
            seed=seed,
            model=model,
            temporal_stabilize=temporal_stabilize,
            native_guide=native_guide,
            wan_steps=wan_steps,
            runtime=_runtime,
        )
    else:
        profile = cast("VideoInvisibleProfile", _normalize_video_profile(profile))
    if profile == "seedance-experimental":
        res_sd = remove_video_seedance_experimental(
            source_path,
            clean_output,
            device=device,
            quality=quality,
            _runtime=cast("VideoVaeRuntime | None", _runtime),
        )
        return res_sd if is_legacy_experimental else replace(res_sd, experimental=False)
    if profile in WAN_EXPERIMENTAL_PROFILES:
        res_wan = remove_video_wan_experimental(
            source_path,
            clean_output,
            profile=profile,
            device=device,
            quality=quality,
            seed=seed,
            inference_steps=wan_steps,
            _runtime=cast("VideoWanRuntime | None", _runtime),
            _frame_stream=_frame_stream,
        )
        return res_wan if is_legacy_experimental else replace(res_wan, experimental=False)
    if source_path.suffix.lower() not in _REGENERATED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(_REGENERATED_VIDEO_EXTENSIONS))
        raise ValueError(f"Video SynthID regeneration requires one of: {supported}")
    _require_video_runtime()
    from remove_ai_watermarks.video_invisible import regenerate_video_candidate

    output_path = _video_output(
        source_path,
        clean_output,
        operation="SynthID removal",
    )
    regen_extra: dict[str, Any] = {}
    if native_guide:
        regen_extra["native_guide"] = True
    if _frame_stream is not None:
        regen_extra["_frame_stream"] = _frame_stream
    metrics = regenerate_video_candidate(
        source_path,
        output_path,
        noise_std=noise_std,
        long_side=long_side,
        fps=fps,
        batch_size=batch_size,
        seed=seed,
        model=model,
        device=device,
        runtime=cast("VideoVaeRuntime | None", _runtime),
        temporal_stabilize=temporal_stabilize,
        **regen_extra,
    )
    return VideoInvisibleResult(
        source=source_path,
        output=output_path,
        noise_std=noise_std,
        metrics=metrics,
        remaining_metadata=get_ai_metadata(output_path),
    )


def remove_video_seedance_experimental(
    source: str | Path,
    output: str | Path | None = None,
    *,
    device: str = "auto",
    quality: Literal["delivery", "master"] = "delivery",
    _runtime: VideoVaeRuntime | None = None,
) -> VideoSeedanceExperimentalResult:
    """Apply the experimental Seedance texture-transfer recipe to an owned MP4.

    This pixel stage expects visible AI labels to have been handled separately.
    It preserves native geometry, all frame timestamps, and source audio, and
    strips AI metadata. CapCut verification remains a development-only check.
    ``delivery`` writes H.264 High CRF18; ``master`` retains lossless-coded YUV.
    Neither mode guarantees unchanged fine detail or a per-file detector result.
    """
    from remove_ai_watermarks.metadata import get_ai_metadata

    source_path = _video_source(source)
    if source_path.suffix.lower() not in _REGENERATED_VIDEO_EXTENSIONS:
        raise ValueError(f"Seedance processing requires one of: {_EXPECTED_REGENERATED_VIDEO_EXTENSIONS}")
    if quality not in {"delivery", "master"}:
        raise ValueError("quality must be delivery or master")
    _require_video_runtime()
    from remove_ai_watermarks.video_seedance import regenerate_seedance_experimental

    clean_output = (
        Path(output) if output is not None else source_path.with_stem(source_path.stem + "_seedance_experimental")
    )
    output_path = _video_output(source_path, clean_output, operation="Experimental Seedance processing")
    metrics = regenerate_seedance_experimental(
        source_path, output_path, device=device, quality=quality, runtime=_runtime
    )
    return VideoSeedanceExperimentalResult(
        source=source_path,
        output=output_path,
        noise_std=0.0,
        metrics=metrics,
        remaining_metadata=get_ai_metadata(output_path),
        quality=quality,
    )


def remove_video_wan_experimental(
    source: str | Path,
    output: str | Path | None = None,
    *,
    profile: WanExperimentalProfile = "wan21-synthid-experimental",
    device: str = "auto",
    quality: Literal["delivery", "master"] = "delivery",
    seed: int = 0,
    chunk_frames: int = 193,
    overlap_frames: int = 17,
    inference_steps: int | None = None,
    _runtime: VideoWanRuntime | None = None,
    _frame_stream: Iterable[Any] | None = None,
) -> VideoWanExperimentalResult:
    """Apply an experimental CUDA/MPS 3D Wan 2.1 video-to-video profile to an owned MP4, MOV, or M4V.

    Preserves native resolution, exact frame timestamps, and source audio while
    stripping AI metadata. Clips exceeding ``chunk_frames`` are processed through
    overlapping 3D causal windows blended with a raised-cosine ramp. Requires an
    NVIDIA CUDA GPU or explicit Apple Silicon MPS unless a preloaded test runtime
    is supplied. Performs no runtime detector verification.
    """
    from remove_ai_watermarks.metadata import get_ai_metadata

    source_path = _video_source(source)
    if source_path.suffix.lower() not in _REGENERATED_VIDEO_EXTENSIONS:
        raise ValueError(f"Wan 2.1 processing requires one of: {_EXPECTED_REGENERATED_VIDEO_EXTENSIONS}")
    if profile not in WAN_EXPERIMENTAL_PROFILES:
        raise ValueError("profile must be wan21-synthid-experimental or wan21-seedance-experimental")
    if quality not in {"delivery", "master"}:
        raise ValueError("quality must be delivery or master")
    _require_video_runtime()
    from remove_ai_watermarks.video_wan import profile_strength, regenerate_wan_experimental

    suffix = profile.replace("-", "_")
    clean_output = Path(output) if output is not None else source_path.with_stem(f"{source_path.stem}_{suffix}")
    output_path = _video_output(source_path, clean_output, operation="Experimental Wan 2.1 processing")
    wan_kwargs: dict[str, Any] = {
        "profile": profile,
        "device": device,
        "quality": quality,
        "seed": seed,
        "chunk_frames": chunk_frames,
        "overlap_frames": overlap_frames,
        "runtime": _runtime,
    }
    if inference_steps is not None:
        wan_kwargs["inference_steps"] = inference_steps
    if _frame_stream is not None:
        wan_kwargs["_frame_stream"] = _frame_stream
    metrics = regenerate_wan_experimental(source_path, output_path, **wan_kwargs)
    strength = profile_strength(profile)
    return VideoWanExperimentalResult(
        source=source_path,
        output=output_path,
        noise_std=strength,
        metrics=metrics,
        remaining_metadata=get_ai_metadata(output_path),
        quality=quality,
        profile=profile,
        strength=strength,
    )


def remove_video_seedance(
    source: str | Path,
    output: str | Path | None = None,
    *,
    device: str = "auto",
    quality: Literal["delivery", "master"] = "delivery",
    _runtime: VideoVaeRuntime | None = None,
) -> VideoSeedanceExperimentalResult:
    """Production Seedance video watermark removal (default output ``*_clean.mp4``)."""
    source_path = _video_source(source)
    clean_output = Path(output) if output is not None else source_path.with_stem(source_path.stem + "_clean")
    result = remove_video_seedance_experimental(
        source_path,
        clean_output,
        device=device,
        quality=quality,
        _runtime=_runtime,
    )
    return replace(result, experimental=False)


def remove_video_wan(
    source: str | Path,
    output: str | Path | None = None,
    *,
    profile: WanExperimentalProfile = "wan21-synthid",
    device: str = "auto",
    quality: Literal["delivery", "master"] = "delivery",
    seed: int = 0,
    chunk_frames: int = 193,
    overlap_frames: int = 17,
    inference_steps: int | None = None,
    _runtime: VideoWanRuntime | None = None,
    _frame_stream: Iterable[Any] | None = None,
) -> VideoWanExperimentalResult:
    """Production 3D Wan 2.1 video watermark removal (default output ``*_clean.mp4``)."""
    source_path = _video_source(source)
    clean_output = Path(output) if output is not None else source_path.with_stem(source_path.stem + "_clean")
    norm_profile = cast("WanExperimentalProfile", _normalize_video_profile(profile))
    result = remove_video_wan_experimental(
        source_path,
        clean_output,
        profile=norm_profile,
        device=device,
        quality=quality,
        seed=seed,
        chunk_frames=chunk_frames,
        overlap_frames=overlap_frames,
        inference_steps=inference_steps,
        _runtime=_runtime,
        _frame_stream=_frame_stream,
    )
    return replace(result, experimental=False)


_ORIG_REMOVE_VIDEO_VISIBLE = remove_video_visible
_ORIG_REMOVE_VIDEO_INVISIBLE = remove_video_invisible
