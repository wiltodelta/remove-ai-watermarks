"""Remove-AI-Watermarks: Unified tool for removing visible and invisible AI watermarks.

High-level API (lazy, so ``import remove_ai_watermarks`` stays cheap)::

    import remove_ai_watermarks as raiw
    raiw.remove_visible("in.png", "out.png")            # clean a file (provenance auto)
    result, removed = raiw.remove_visible(bgr_array)    # array -> array
    report = raiw.remove_visible_detailed(bgr_array)    # cleaned / partial / unvalidated
    raiw.visible_provenance("in.png")                   # -> frozenset of confirmed vendors
    raiw.identify_video("in.mp4")                       # -> VideoProvenanceReport
    raiw.inspect_video_metadata("in.mp4")               # -> VideoMetadataReport
    raiw.remove_video_all("in.mp4", "out.mp4")          # visible + verified metadata
    raiw.remove_video_batch("videos", "videos_clean")   # complete per-file results
    raiw.remove_video_metadata("in.mp4", "out.mp4")     # verified metadata strip
    raiw.remove_video_invisible("in.mp4", "out.mp4")    # calibrated video-pixel regeneration
    raiw.remove_video_seedance_experimental("in.mp4", "out.mp4")  # opt-in Seedance experiment
    raiw.remove_video_wan_experimental("in.mp4", "out.mp4")       # opt-in CUDA 3D Wan 2.1 experiment
    raiw.remove_video_visible("in.mp4", "out.mp4")      # stable visible video-mark removal
    raiw.identify_audio("in.wav", scheme="auto")        # -> AudioProvenanceReport
    raiw.inspect_audio_metadata("in.mp3")               # -> AudioMetadataReport
    raiw.remove_audio_metadata("in.mp3", "out.mp3")     # verified lossless metadata strip
    raiw.remove_audio_invisible("in.wav", "out.m4a", scheme="auto")  # verified audio-watermark removal
    raiw.remove_audio_all("in.mp3", "out.mp3")          # metadata + optional audio-watermark processing
    raiw.remove_audio_batch("tracks", "tracks_clean")   # complete per-file audio batch results
    raiw.identify_text("in.md")                         # -> TextProvenanceReport
    raiw.remove_text_metadata("in.md", "out.md")        # C2PATXT, C2PA HTML/ASCII armor, VSRMark, stego
    raiw.remove_text_invisible("in.md", "out.md")       # span-protected local prose regeneration
    raiw.remove_text_all("in.md", "out.md")             # metadata + optional prose regeneration
    raiw.classify_source("in.png")                      # 10-provider source pipeline + version, not SynthID detection

For a provenance verdict use the ``identify`` submodule::

    from remove_ai_watermarks.identify import identify
    report = identify("in.png")
"""

import os as _os
import warnings as _warnings
from typing import TYPE_CHECKING

# transformers prints a noisy deprecation for the Siglip2ImageProcessorFast
# alias when it is imported (by the optional GPU/ML path). Silence it before
# any submodule pulls transformers in, so the CLI startup stays quiet. Uses
# setdefault so a user-set TRANSFORMERS_VERBOSITY still wins.
_os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
_warnings.filterwarnings("ignore", message=r".*ImageProcessorFast.*")


__version__ = "0.72.0"

__all__ = [
    "AudioAllResult",
    "AudioBatchItem",
    "AudioBatchSummary",
    "AudioDetection",
    "AudioFidelity",
    "AudioMetadataReport",
    "AudioMetadataResult",
    "AudioProvenanceReport",
    "AudioRemovalResult",
    "AudioVerificationIncomplete",
    "AudioWindowReading",
    "BatchItemResult",
    "BatchSummary",
    "InvisibleOptions",
    "MetadataStripIncomplete",
    "RemoveAllResult",
    "SourceClassification",
    "SynthIDThreefryScore",
    "TextAllResult",
    "TextInvisibleResult",
    "TextMetadataResult",
    "TextProvenanceReport",
    "TextSpan",
    "TextWatermarkScore",
    "VideoAudioMultiStreamResult",
    "VideoAudioRemovalResult",
    "VideoEraseResult",
    "VideoMessageRemovalResult",
    "VideoMessageReport",
    "VideoMessageVerificationIncomplete",
    "VideoWanExperimentalResult",
    "VideoWatermarkPresenceReport",
    "VisibleRemovalResult",
    "__version__",
    "build_text_api_rewriter",
    "classify_source",
    "decode_stego_payloads",
    "detect_video_watermark_presence",
    "encode_innamark",
    "identify_audio",
    "identify_text",
    "identify_video",
    "identify_video_message",
    "inspect_audio_metadata",
    "inspect_video_metadata",
    "remove_all",
    "remove_audio_all",
    "remove_audio_batch",
    "remove_audio_invisible",
    "remove_audio_metadata",
    "remove_batch",
    "remove_text_all",
    "remove_text_invisible",
    "remove_text_metadata",
    "remove_video_all",
    "remove_video_audio",
    "remove_video_audio_all_streams",
    "remove_video_batch",
    "remove_video_invisible",
    "remove_video_message",
    "remove_video_metadata",
    "remove_video_region",
    "remove_video_seedance",
    "remove_video_seedance_experimental",
    "remove_video_visible",
    "remove_video_wan",
    "remove_video_wan_experimental",
    "remove_visible",
    "remove_visible_detailed",
    "score_gumbel_text",
    "score_kgw_text",
    "score_synthid_text",
    "score_synthid_threefry_text",
    "score_textgrain_text",
    "score_textseal2_text",
    "score_unigram_text",
    "split_protected_text_spans",
    "verify_text_rewrite_facts",
    "visible_provenance",
]

if TYPE_CHECKING:
    from remove_ai_watermarks.api import (
        BatchItemResult,
        BatchSummary,
        InvisibleOptions,
        MetadataStripIncomplete,
        RemoveAllResult,
        VisibleRemovalResult,
        remove_all,
        remove_batch,
        remove_visible,
        remove_visible_detailed,
        visible_provenance,
    )
    from remove_ai_watermarks.audio import (
        AudioAllResult,
        AudioBatchItem,
        AudioBatchSummary,
        AudioDetection,
        AudioFidelity,
        AudioMetadataReport,
        AudioMetadataResult,
        AudioProvenanceReport,
        AudioRemovalResult,
        AudioVerificationIncomplete,
        AudioWindowReading,
        identify_audio,
        inspect_audio_metadata,
        remove_audio_all,
        remove_audio_batch,
        remove_audio_invisible,
        remove_audio_metadata,
    )
    from remove_ai_watermarks.source_classify import SourceClassification, classify_source
    from remove_ai_watermarks.text import (
        SynthIDThreefryScore,
        TextAllResult,
        TextInvisibleResult,
        TextMetadataResult,
        TextProvenanceReport,
        TextSpan,
        TextWatermarkScore,
        build_text_api_rewriter,
        decode_stego_payloads,
        encode_innamark,
        identify_text,
        remove_text_all,
        remove_text_invisible,
        remove_text_metadata,
        score_gumbel_text,
        score_kgw_text,
        score_synthid_text,
        score_synthid_threefry_text,
        score_textgrain_text,
        score_textseal2_text,
        score_unigram_text,
        split_protected_text_spans,
        verify_text_rewrite_facts,
    )
    from remove_ai_watermarks.video import (
        VideoEraseResult,
        VideoWanExperimentalResult,
        identify_video,
        inspect_video_metadata,
        remove_video_all,
        remove_video_batch,
        remove_video_invisible,
        remove_video_metadata,
        remove_video_region,
        remove_video_seedance,
        remove_video_seedance_experimental,
        remove_video_visible,
        remove_video_wan,
        remove_video_wan_experimental,
    )
    from remove_ai_watermarks.video_audio import (
        VideoAudioMultiStreamResult,
        VideoAudioRemovalResult,
        remove_video_audio,
        remove_video_audio_all_streams,
    )
    from remove_ai_watermarks.video_messages import (
        VideoMessageRemovalResult,
        VideoMessageReport,
        VideoMessageVerificationIncomplete,
        VideoWatermarkPresenceReport,
        detect_video_watermark_presence,
        identify_video_message,
        remove_video_message,
    )


def __getattr__(name: str) -> object:
    """Lazily resolve the high-level API (PEP 562), so the heavy imports (cv2, the
    metadata/identify stack) load only when a caller actually reaches for them."""
    if name in (
        "VideoAudioMultiStreamResult",
        "VideoAudioRemovalResult",
        "remove_video_audio",
        "remove_video_audio_all_streams",
    ):
        from remove_ai_watermarks import video_audio

        return getattr(video_audio, name)
    if name in (
        "BatchItemResult",
        "BatchSummary",
        "InvisibleOptions",
        "MetadataStripIncomplete",
        "RemoveAllResult",
        "VisibleRemovalResult",
        "remove_all",
        "remove_batch",
        "remove_visible",
        "remove_visible_detailed",
        "visible_provenance",
    ):
        from remove_ai_watermarks import api

        return getattr(api, name)
    if name in (
        "AudioAllResult",
        "AudioBatchItem",
        "AudioBatchSummary",
        "AudioDetection",
        "AudioFidelity",
        "AudioMetadataReport",
        "AudioMetadataResult",
        "AudioProvenanceReport",
        "AudioRemovalResult",
        "AudioVerificationIncomplete",
        "AudioWindowReading",
        "identify_audio",
        "inspect_audio_metadata",
        "remove_audio_all",
        "remove_audio_batch",
        "remove_audio_invisible",
        "remove_audio_metadata",
    ):
        from remove_ai_watermarks import audio

        return getattr(audio, name)
    if name in (
        "VideoMessageReport",
        "VideoMessageRemovalResult",
        "VideoMessageVerificationIncomplete",
        "VideoWatermarkPresenceReport",
        "detect_video_watermark_presence",
        "identify_video_message",
        "remove_video_message",
    ):
        from remove_ai_watermarks import video_messages

        return getattr(video_messages, name)
    if name in ("SourceClassification", "classify_source"):
        from remove_ai_watermarks import source_classify

        return getattr(source_classify, name)
    if name in (
        "TextAllResult",
        "TextInvisibleResult",
        "TextMetadataResult",
        "TextProvenanceReport",
        "TextSpan",
        "TextWatermarkScore",
        "SynthIDThreefryScore",
        "build_text_api_rewriter",
        "decode_stego_payloads",
        "encode_innamark",
        "identify_text",
        "remove_text_all",
        "remove_text_invisible",
        "remove_text_metadata",
        "score_gumbel_text",
        "score_kgw_text",
        "score_synthid_text",
        "score_synthid_threefry_text",
        "score_textgrain_text",
        "score_textseal2_text",
        "score_unigram_text",
        "split_protected_text_spans",
        "verify_text_rewrite_facts",
    ):
        from remove_ai_watermarks import text

        return getattr(text, name)
    if name in (
        "VideoEraseResult",
        "VideoWanExperimentalResult",
        "identify_video",
        "inspect_video_metadata",
        "remove_video_all",
        "remove_video_batch",
        "remove_video_invisible",
        "remove_video_metadata",
        "remove_video_region",
        "remove_video_seedance",
        "remove_video_seedance_experimental",
        "remove_video_visible",
        "remove_video_wan",
        "remove_video_wan_experimental",
    ):
        from remove_ai_watermarks import video

        return getattr(video, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
