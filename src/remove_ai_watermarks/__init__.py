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
    raiw.remove_video_visible("in.mp4", "out.mp4")      # stable visible video-mark removal

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


__version__ = "0.39.0"

__all__ = [
    "BatchItemResult",
    "BatchSummary",
    "InvisibleOptions",
    "MetadataStripIncomplete",
    "RemoveAllResult",
    "VisibleRemovalResult",
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
    "remove_visible_detailed",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
