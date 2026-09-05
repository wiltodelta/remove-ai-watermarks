"""High-level video processing API.

The product path covers provenance identification, container-level AI metadata
removal, temporally stabilized visible Sora, Veo, Seedance, Dola, Hailuo AI, and
Kling AI removal, and an oracle-certified opt-in VAE profile for video SynthID.
The visible pixel path reuses the image package's shared fill backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal

from remove_ai_watermarks.video_synthid import (
    DEFAULT_VIDEO_SYNTHID_FPS,
    DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    DEFAULT_VIDEO_SYNTHID_NOISE_STD,
    DEFAULT_VIDEO_SYNTHID_VAE,
)

if TYPE_CHECKING:
    from remove_ai_watermarks.video_invisible import RegenerationMetrics, VideoVaeRuntime
    from remove_ai_watermarks.video_visible import VideoScan

VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".flv"})
VIDEO_VISIBLE_MARKS = ("sora", "veo", "seedance", "doubao", "dola", "hailuo", "kling")
_ISOBMFF_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".m4v"})
_EBML_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".webm", ".mkv"})
_RIFF_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".avi"})
_FLV_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".flv"})
_REGENERATED_VIDEO_EXTENSIONS: frozenset[str] = _ISOBMFF_VIDEO_EXTENSIONS
_EBML_MAGIC = b"\x1aE\xdf\xa3"


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
class VideoMetadataResult:
    """Result of a verified video metadata-removal operation."""

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


@dataclass(frozen=True)
class VideoVisibleResult:
    """Result of visible AI-watermark removal from a video."""

    source: Path
    output: Path | None
    mark: str
    total_frames: int
    detected_frames: int
    removed_frames: int
    remaining_metadata: dict[str, str]


@dataclass(frozen=True)
class VideoInvisibleResult:
    """Result of removing video SynthID through the oracle-certified VAE profile."""

    source: Path
    output: Path
    noise_std: float
    metrics: RegenerationMetrics
    remaining_metadata: dict[str, str]

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
class VideoAllResult:
    """Result of the complete video-cleaning pipeline."""

    source: Path
    output: Path
    visible_mark: str | None
    total_frames: int
    visible_detected_frames: int
    visible_removed_frames: int
    detected_metadata: dict[str, str]
    remaining_metadata: dict[str, str]
    invisible_removed: bool


@dataclass(frozen=True)
class VideoBatchItem:
    """Outcome for one source in a video batch."""

    source: Path
    output: Path | None
    mode: Literal["all", "visible", "metadata"]
    changed: bool
    visible_mark: str | None
    invisible_removed: bool
    error: str | None = None


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
    "seedance": "ByteDance Seedance",
    "doubao": "ByteDance Doubao",
    "dola": "ByteDance Dola",
    "hailuo": "MiniMax Hailuo AI",
    "kling": "Kuaishou Kling AI",
}


def _video_source(source: str | Path) -> Path:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Video does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Video source must be a file: {path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise ValueError(f"Unsupported video format {path.suffix or '<none>'}; expected one of: {supported}")
    with path.open("rb") as stream:
        head = stream.read(12)
    suffix = path.suffix.lower()
    matches_container = (
        (suffix in _ISOBMFF_VIDEO_EXTENSIONS and len(head) >= 8 and head[4:8] == b"ftyp")
        or (suffix in _EBML_VIDEO_EXTENSIONS and head.startswith(_EBML_MAGIC))
        or (suffix in _RIFF_VIDEO_EXTENSIONS and len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"AVI ")
        or (suffix in _FLV_VIDEO_EXTENSIONS and head.startswith(b"FLV"))
    )
    if not matches_container:
        raise ValueError(f"Video content does not match its {suffix} extension: {path}")
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
    padding and the mask style -- is data on ``VISIBLE_MARK_POLICIES``. The only thing
    left here is WHICH metadata predicate confirms which vendor, which genuinely is a
    mapping and not a tuning constant.
    """
    from remove_ai_watermarks.video_visible import (
        VISIBLE_MARK_POLICIES,
        has_bytedance_video_provenance,
        has_hailuo_video_provenance,
        has_sora_provenance,
        has_veo_provenance,
        stabilize_localizations,
    )

    confirms = {
        "sora": has_sora_provenance,
        "veo": has_veo_provenance,
        "seedance": has_bytedance_video_provenance,
        "dola": has_bytedance_video_provenance,
        "hailuo": has_hailuo_video_provenance,
    }.get(selected_mark)
    policy = VISIBLE_MARK_POLICIES[selected_mark]
    regions = stabilize_localizations(
        selected_mark,
        selected_scan.detections,
        provenance=bool(confirms and confirms(markers)),
    )
    return regions, policy.padding_fraction, policy.mask_style


def _select_stable_visible_mark(
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
    """Select the first stable provider result in the public specificity order."""
    for candidate_mark in candidate_marks:
        candidate_scan = scans[candidate_mark]
        candidate_regions, candidate_padding, candidate_mask_style = _visible_removal_plan(
            candidate_mark,
            candidate_scan,
            markers,
        )
        if any(region is not None for region in candidate_regions):
            return (
                candidate_mark,
                candidate_scan,
                candidate_regions,
                candidate_padding,
                candidate_mask_style,
            )
    return None


def _platform_from_video_metadata(markers: dict[str, str]) -> str | None:
    """Map supported C2PA-derived marker text to its generating platform."""
    from remove_ai_watermarks._internal.constants import C2PA_AI_VENDORS

    marker_text = "\n".join(markers.values()).casefold()
    if not marker_text:
        return None
    for vendor in C2PA_AI_VENDORS:
        if vendor.platform is not None and vendor.needle is not None and vendor.needle.casefold() in marker_text:
            return vendor.platform
    return None


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


def identify_video(
    source: str | Path,
    *,
    check_visible: bool = True,
) -> VideoProvenanceReport:
    """Identify locally readable AI provenance and stable visible video marks.

    A negative result is reported as unknown, never clean. Proprietary pixel
    watermarks such as video SynthID have no public local decoder.
    """
    from remove_ai_watermarks.metadata import get_ai_metadata

    source_path = _video_source(source)
    markers = get_ai_metadata(source_path)
    selected_mark: str | None = None
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
        selected = _select_stable_visible_mark(scans, markers, VIDEO_VISIBLE_MARKS)
        if selected is not None:
            selected_mark, _scan, regions, _padding, _mask_style = selected
            detected_frames = sum(region is not None for region in regions)

    has_signal = bool(markers) or selected_mark is not None
    caveats = ["No public local decoder can verify proprietary pixel watermarks such as video SynthID."]
    if not check_visible:
        caveats.append("Visible video-mark detection was skipped.")
    if not has_signal:
        caveats.append("No supported signal was found; absence is unknown, not proof that the video is clean.")
    return VideoProvenanceReport(
        source=source_path,
        is_ai_generated=True if has_signal else None,
        confidence="high" if has_signal else "unknown",
        platform=(
            _VISIBLE_PLATFORM.get(selected_mark)
            if selected_mark is not None
            else _platform_from_video_metadata(markers)
        ),
        visible_mark=selected_mark,
        visible_detected_frames=detected_frames,
        total_frames=total_frames,
        has_ai_metadata=bool(markers),
        metadata_markers=markers,
        caveats=tuple(caveats),
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


def remove_video_visible(
    source: str | Path,
    output: str | Path | None = None,
    *,
    mark: str = "auto",
    backend: str = "cv2",
    strip_metadata: bool = True,
    temporal_consistency: bool = True,
    _metadata_markers: dict[str, str] | None = None,
) -> VideoVisibleResult:
    """Remove a supported visible AI wordmark from a video.

    ``mark="auto"`` scans every supported provider in one decode pass and selects
    the first stable match in specificity order. Explicit marks are ``sora``,
    ``veo``, ``seedance``, ``dola``, ``hailuo``, and ``kling``. The full
    sequence is scanned before pixels change, and only recurring candidates are
    accepted. Complete audio is copied without re-encoding; video is transcoded
    because the pixels change. ``temporal_consistency=True`` motion-aligns a
    safely covered prior fill after each image-backend pass; scene cuts and
    disjoint masks keep the independent current fill. Completed output is
    published atomically. When no stable mark is found, no output is written
    and ``output`` in the result is ``None``.
    """
    _require_video_runtime()

    from remove_ai_watermarks.metadata import get_ai_metadata
    from remove_ai_watermarks.video_visible import encode_clean_video, scan_video_marks
    from remove_ai_watermarks.watermark_registry import resolve_backend

    if mark not in {"auto", *VIDEO_VISIBLE_MARKS}:
        raise ValueError("Unsupported visible video mark; expected auto, sora, veo, seedance, dola, hailuo, or kling")
    if backend not in {"auto", "cv2", "migan", "lama"}:
        raise ValueError("Unsupported fill backend; expected auto, cv2, migan, or lama")

    source_path = _video_source(source)
    output_path = _video_output(source_path, output, operation="visible watermark removal")
    markers = get_ai_metadata(source_path) if _metadata_markers is None else _metadata_markers

    candidate_marks = VIDEO_VISIBLE_MARKS if mark == "auto" else (mark,)
    scans = scan_video_marks(source_path, candidate_marks)
    selected = _select_stable_visible_mark(scans, markers, candidate_marks)
    if selected is None:
        scan = scans[candidate_marks[0]]
        return VideoVisibleResult(
            source=source_path,
            output=None,
            mark=mark,
            total_frames=len(scan.detections),
            detected_frames=0,
            removed_frames=0,
            remaining_metadata=markers if strip_metadata else {},
        )
    mark, scan, regions, padding_fraction, mask_style = selected
    detected_frames = sum(region is not None for region in regions)

    # Validate optional model availability before ffmpeg creates or overwrites
    # the requested output.
    resolve_backend(backend)  # type: ignore[arg-type]
    removed_frames = encode_clean_video(
        source_path,
        output_path,
        scan,
        regions,
        backend=backend,  # type: ignore[arg-type]
        strip_metadata=strip_metadata,
        padding_fraction=padding_fraction,
        mask_style=mask_style,
        temporal_consistency=temporal_consistency,
    )
    remaining_metadata = get_ai_metadata(output_path) if strip_metadata else {}
    return VideoVisibleResult(
        source=source_path,
        output=output_path,
        mark=mark,
        total_frames=len(scan.detections),
        detected_frames=detected_frames,
        removed_frames=removed_frames,
        remaining_metadata=remaining_metadata,
    )


def remove_video_all(
    source: str | Path,
    output: str | Path | None = None,
    *,
    mark: str = "auto",
    backend: str = "cv2",
    temporal_consistency: bool = True,
    include_invisible: bool = False,
    noise_std: float = DEFAULT_VIDEO_SYNTHID_NOISE_STD,
    long_side: int = DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    fps: float = DEFAULT_VIDEO_SYNTHID_FPS,
    batch_size: int = 4,
    seed: int = 0,
    model: str = DEFAULT_VIDEO_SYNTHID_VAE,
    device: str = "auto",
    _invisible_runtime: VideoVaeRuntime | None = None,
) -> VideoAllResult:
    """Run the complete video cleaning pipeline.

    The default path removes a stable visible provider mark when present and
    always strips verified AI metadata. It writes a same-container passthrough
    when neither signal is present, giving product callers one predictable
    output contract. ``include_invisible=True`` additionally runs lossy VAE
    regeneration with the oracle-certified default profile.
    """
    from remove_ai_watermarks.metadata import get_ai_metadata

    source_path = _video_source(source)
    output_path = _video_output(source_path, output, operation="complete cleaning")
    if include_invisible and source_path.suffix.lower() not in _REGENERATED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(_REGENERATED_VIDEO_EXTENSIONS))
        raise ValueError(f"Video SynthID regeneration requires one of: {supported}")
    _require_video_runtime()
    detected_metadata = get_ai_metadata(source_path)

    with TemporaryDirectory(prefix=f".{source_path.stem}-video-all-", dir=source_path.parent) as temp_dir:
        visible_output = Path(temp_dir) / f"visible{source_path.suffix}" if include_invisible else output_path
        visible_result = remove_video_visible(
            source_path,
            visible_output,
            mark=mark,
            backend=backend,
            strip_metadata=True,
            temporal_consistency=temporal_consistency,
            _metadata_markers=detected_metadata,
        )
        current_source = visible_result.output or source_path

        if include_invisible:
            invisible_result = remove_video_invisible(
                current_source,
                output_path,
                noise_std=noise_std,
                long_side=long_side,
                fps=fps,
                batch_size=batch_size,
                seed=seed,
                model=model,
                device=device,
                _runtime=_invisible_runtime,
            )
            remaining_metadata = invisible_result.remaining_metadata
        elif visible_result.output is None:
            metadata_result = remove_video_metadata(
                source_path,
                output_path,
                _detected_metadata=detected_metadata,
            )
            remaining_metadata = metadata_result.remaining
        else:
            remaining_metadata = visible_result.remaining_metadata

    return VideoAllResult(
        source=source_path,
        output=output_path,
        visible_mark=visible_result.mark if visible_result.output is not None else None,
        total_frames=visible_result.total_frames,
        visible_detected_frames=visible_result.detected_frames,
        visible_removed_frames=visible_result.removed_frames,
        detected_metadata=detected_metadata,
        remaining_metadata=remaining_metadata,
        invisible_removed=include_invisible,
    )


def remove_video_batch(
    directory: str | Path,
    output_directory: str | Path | None = None,
    *,
    mode: Literal["all", "visible", "metadata"] = "all",
    mark: str = "auto",
    backend: str = "cv2",
    temporal_consistency: bool = True,
    include_invisible: bool = False,
    noise_std: float = DEFAULT_VIDEO_SYNTHID_NOISE_STD,
    long_side: int = DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    fps: float = DEFAULT_VIDEO_SYNTHID_FPS,
    batch_size: int = 4,
    seed: int = 0,
    model: str = DEFAULT_VIDEO_SYNTHID_VAE,
    device: str = "auto",
) -> VideoBatchResult:
    """Process every supported video in one directory.

    Files are processed sequentially so model and ffmpeg resource use stays
    bounded. Per-file failures are returned in ``items`` and do not discard
    successful outputs. Visible-only no-op files are copied byte-for-byte so the
    output directory remains complete.
    """
    import shutil

    from remove_ai_watermarks.video_encoding import atomic_video_output

    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Video directory does not exist: {directory_path}")
    if not directory_path.is_dir():
        raise ValueError(f"Video batch source must be a directory: {directory_path}")
    if mode not in {"all", "visible", "metadata"}:
        raise ValueError("Unsupported video batch mode; expected all, visible, or metadata")
    if mark not in {"auto", *VIDEO_VISIBLE_MARKS}:
        raise ValueError("Unsupported visible video mark; expected auto, sora, veo, seedance, dola, hailuo, or kling")
    if backend not in {"auto", "cv2", "migan", "lama"}:
        raise ValueError("Unsupported fill backend; expected auto, cv2, migan, or lama")
    if include_invisible and mode != "all":
        raise ValueError("The invisible video stage is available only in all mode")
    if mode != "metadata":
        _require_video_runtime()

    output_path = (
        Path(output_directory)
        if output_directory is not None
        else directory_path.parent / f"{directory_path.name}_clean"
    )
    if output_path.resolve() == directory_path.resolve():
        raise ValueError("Video batch output directory must differ from the source directory")
    output_path.mkdir(parents=True, exist_ok=True)

    sources = tuple(
        path
        for path in sorted(directory_path.iterdir(), key=lambda candidate: candidate.name.lower())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    items: list[VideoBatchItem] = []
    invisible_runtime: VideoVaeRuntime | None = None
    invisible_runtime_error: str | None = None
    for source_path in sources:
        item_output = output_path / source_path.name
        try:
            if mode == "all":
                if (
                    include_invisible
                    and invisible_runtime is None
                    and source_path.suffix.lower() in _REGENERATED_VIDEO_EXTENSIONS
                ):
                    # Validate the container before paying the multi-GB model
                    # load, then retain one runtime for the complete batch.
                    _video_source(source_path)
                    if invisible_runtime_error is not None:
                        raise RuntimeError(invisible_runtime_error)
                    from remove_ai_watermarks.video_invisible import load_video_vae_runtime

                    try:
                        invisible_runtime = load_video_vae_runtime(model=model, device=device)
                    except Exception as exc:
                        invisible_runtime_error = str(exc)
                        raise
                all_result = remove_video_all(
                    source_path,
                    item_output,
                    mark=mark,
                    backend=backend,
                    temporal_consistency=temporal_consistency,
                    include_invisible=include_invisible,
                    noise_std=noise_std,
                    long_side=long_side,
                    fps=fps,
                    batch_size=batch_size,
                    seed=seed,
                    model=model,
                    device=device,
                    _invisible_runtime=invisible_runtime,
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
                            all_result.visible_mark or all_result.detected_metadata or all_result.invisible_removed
                        ),
                        visible_mark=all_result.visible_mark,
                        invisible_removed=all_result.invisible_removed,
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
    noise_std: float = DEFAULT_VIDEO_SYNTHID_NOISE_STD,
    long_side: int = DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    fps: float = DEFAULT_VIDEO_SYNTHID_FPS,
    batch_size: int = 4,
    seed: int = 0,
    model: str = DEFAULT_VIDEO_SYNTHID_VAE,
    device: str = "auto",
    _runtime: VideoVaeRuntime | None = None,
) -> VideoInvisibleResult:
    """Remove video SynthID through the oracle-certified VAE profile.

    The function also strips source metadata during the transcode. The default
    profile is provider-oracle certified; important outputs may still be
    rechecked with Google's verifier when the caller needs a per-file verdict.
    """
    from remove_ai_watermarks.metadata import get_ai_metadata

    source_path = _video_source(source)
    if source_path.suffix.lower() not in _REGENERATED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(_REGENERATED_VIDEO_EXTENSIONS))
        raise ValueError(f"Video SynthID regeneration requires one of: {supported}")
    _require_video_runtime()
    from remove_ai_watermarks.video_invisible import regenerate_video_candidate

    clean_output = Path(output) if output is not None else source_path.with_stem(source_path.stem + "_clean")
    output_path = _video_output(
        source_path,
        clean_output,
        operation="SynthID removal",
    )
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
        runtime=_runtime,
    )
    return VideoInvisibleResult(
        source=source_path,
        output=output_path,
        noise_std=noise_std,
        metrics=metrics,
        remaining_metadata=get_ai_metadata(output_path),
    )
