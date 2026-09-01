"""Remove-AI-Watermarks: Unified tool for removing visible and invisible AI watermarks.

High-level API (lazy, so ``import remove_ai_watermarks`` stays cheap)::

    import remove_ai_watermarks as raiw
    raiw.remove_visible("in.png", "out.png")            # clean a file (provenance auto)
    result, removed = raiw.remove_visible(bgr_array)    # array -> array
    raiw.visible_provenance("in.png")                   # -> frozenset of confirmed vendors
    raiw.identify_video("in.mp4")                       # -> VideoProvenanceReport
    raiw.inspect_video_metadata("in.mp4")               # -> VideoMetadataReport
    raiw.remove_video_all("in.mp4", "out.mp4")          # visible + verified metadata
    raiw.remove_video_batch("videos", "videos_clean")   # complete per-file results
    raiw.remove_video_metadata("in.mp4", "out.mp4")     # verified metadata strip
    raiw.remove_video_invisible("in.mp4", "out.mp4")    # oracle-certified SynthID removal
    raiw.remove_video_visible("in.mp4", "out.mp4")      # stable visible video-mark removal
    raiw.verify_openai_synthid("in.png", acknowledge_upload=True)  # remote

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


__version__ = "0.36.0"

__all__ = [
    "BatchSummary",
    "InvisibleOptions",
    "MetadataStripIncomplete",
    "OpenAIProvenanceError",
    "OpenAISynthIDDetection",
    "RemoveAllResult",
    "__version__",
    "identify_video",
    "inspect_video_metadata",
    "remove_all",
    "remove_batch",
    "remove_video_all",
    "remove_video_batch",
    "remove_video_invisible",
    "remove_video_metadata",
    "remove_video_visible",
    "remove_visible",
    "verify_openai_synthid",
    "visible_provenance",
]

if TYPE_CHECKING:
    from remove_ai_watermarks.api import (
        BatchSummary,
        InvisibleOptions,
        MetadataStripIncomplete,
        RemoveAllResult,
        remove_all,
        remove_batch,
        remove_visible,
        visible_provenance,
    )
    from remove_ai_watermarks.openai_provenance import (
        OpenAIProvenanceError,
        OpenAISynthIDDetection,
        verify_openai_synthid,
    )
    from remove_ai_watermarks.video import (
        identify_video,
        inspect_video_metadata,
        remove_video_all,
        remove_video_batch,
        remove_video_invisible,
        remove_video_metadata,
        remove_video_visible,
    )


def __getattr__(name: str) -> object:
    """Lazily resolve the high-level API (PEP 562), so the heavy imports (cv2, the
    metadata/identify stack) load only when a caller actually reaches for them."""
    if name in (
        "BatchSummary",
        "InvisibleOptions",
        "MetadataStripIncomplete",
        "RemoveAllResult",
        "remove_all",
        "remove_batch",
        "remove_visible",
        "visible_provenance",
    ):
        from remove_ai_watermarks import api

        return getattr(api, name)
    if name in (
        "identify_video",
        "inspect_video_metadata",
        "remove_video_all",
        "remove_video_batch",
        "remove_video_invisible",
        "remove_video_metadata",
        "remove_video_visible",
    ):
        from remove_ai_watermarks import video

        return getattr(video, name)
    if name in ("OpenAIProvenanceError", "OpenAISynthIDDetection", "verify_openai_synthid"):
        from remove_ai_watermarks import openai_provenance

        return getattr(openai_provenance, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
