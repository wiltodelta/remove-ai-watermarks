"""Share codec fallback across video scanning and reconstruction stages."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import cv2

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

log = logging.getLogger(__name__)


class _PyAVCapture:
    """Provide the small read-only capture interface used by video engines."""

    def __init__(self, container: Any) -> None:
        if not container.streams.video:
            container.close()
            raise ValueError("The input has no video stream")
        self._container = container
        stream: Any = container.streams.video[0]
        self._frames: Iterator[Any] = iter(container.decode(stream))
        self._properties = {
            cv2.CAP_PROP_FRAME_WIDTH: float(stream.width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(stream.height),
            cv2.CAP_PROP_FPS: float(stream.average_rate or stream.guessed_rate or 0),
            cv2.CAP_PROP_FRAME_COUNT: float(stream.frames),
        }
        self._timestamp = 0.0
        self._closed = False

    def isOpened(self) -> bool:
        return not self._closed

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_POS_MSEC:
            return self._timestamp * 1000
        return self._properties.get(property_id, 0.0)

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        if self._closed:
            return False, None
        try:
            frame = next(self._frames)
            self._timestamp = float(frame.time or 0.0)
            pixels: NDArray[np.uint8] = frame.to_ndarray(format="bgr24")
            return True, pixels
        except StopIteration:
            return False, None
        except Exception as error:
            raise RuntimeError(f"PyAV could not decode a video frame: {error}") from error

    def release(self) -> None:
        if not self._closed:
            self._container.close()
            self._closed = True


def open_video_capture(source: Path) -> Any:
    """Use OpenCV when available, otherwise decode BGR frames with PyAV."""
    capture = cv2.VideoCapture(str(source))
    if capture.isOpened():
        return capture
    capture.release()
    try:
        import av
        from av.error import FFmpegError
    except ImportError as error:
        raise RuntimeError("Video codec fallback needs remove-ai-watermarks[video]") from error
    try:
        fallback = _PyAVCapture(av.open(str(source), mode="r"))
    except (FFmpegError, ValueError) as error:
        log.warning("PyAV could not open video %s: %s", source, error)
        return capture
    log.info("Using PyAV because OpenCV could not open video: %s", source)
    return fallback
