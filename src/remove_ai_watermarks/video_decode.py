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


class _FFmpeg16BitCapture:
    """Stream 16-bit BGR48LE frames from ffmpeg for high-bit-depth / HDR sources."""

    def __init__(self, source: Path, width: int, height: int) -> None:
        import subprocess
        import tempfile

        from remove_ai_watermarks.video_tools import require_video_tool

        self._width = int(width)
        self._height = int(height)
        self._frame_bytes = self._width * self._height * 6
        self._stderr = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115 - capture owns the lifetime
        cmd = [
            require_video_tool("ffmpeg"),
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr48le",
            "pipe:1",
        ]
        self._proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
        )
        self._closed = False

    def isOpened(self) -> bool:
        return not self._closed and self._proc.stdout is not None

    def read(self) -> tuple[bool, NDArray[Any] | None]:
        import numpy as np

        if self._closed or self._proc.stdout is None:
            return False, None
        raw = self._proc.stdout.read(self._frame_bytes)
        if not raw:
            return False, None
        if len(raw) != self._frame_bytes:
            raise RuntimeError("Incomplete 16-bit BGR48LE frame read from ffmpeg")
        frame = np.frombuffer(raw, dtype="<u2").reshape(self._height, self._width, 3).copy()
        return True, frame

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc.stdout is not None:
            self._proc.stdout.close()
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()
        self._stderr.close()


def open_video_capture_16bit(source: Path, *, width: int, height: int) -> _FFmpeg16BitCapture:
    """Open a 16-bit BGR48LE frame reader for 10-bit/12-bit/HDR video pipelines."""
    return _FFmpeg16BitCapture(source, width=width, height=height)
