"""Real video decoding when the installed OpenCV has no usable codec backend."""

from fractions import Fraction

import av
import cv2
import numpy as np
import pytest


@pytest.fixture
def codec_clip(tmp_path):
    source = tmp_path / "frames.mkv"
    frames = [np.full((48, 64, 3), (20 + i * 30, 50, 90), dtype=np.uint8) for i in range(3)]
    with av.open(str(source), "w") as container:
        stream = container.add_stream("ffv1", rate=5)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "bgr0"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 10)
        for pixels, pts in zip(frames, (7, 9, 12), strict=True):
            frame = av.VideoFrame.from_ndarray(pixels, format="bgr24")
            frame.pts, frame.time_base = pts, Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return source, frames


class ClosedCapture:
    def isOpened(self):
        return False

    def release(self):
        pass


@pytest.mark.parametrize("stage", ["identify", "sample", "seedance"])
def test_video_stages_decode_without_an_opencv_backend(codec_clip, monkeypatch, stage):
    from remove_ai_watermarks.video import identify_video
    from remove_ai_watermarks.video_invisible import read_sampled_frames
    from remove_ai_watermarks.video_seedance import _frames

    source, expected = codec_clip
    before = source.read_bytes()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: ClosedCapture())
    if stage == "identify":
        report = identify_video(source)
        assert report.total_frames == len(expected)
    else:
        if stage == "sample":
            actual, fps = read_sampled_frames(source, duration=None, output_fps=5, size=(64, 48))
            assert fps == 5.0
        else:
            actual = list(_frames(source))
        np.testing.assert_array_equal(actual, expected)
    assert source.read_bytes() == before


def test_fallback_keeps_geometry_and_nonzero_variable_timestamps(codec_clip, monkeypatch):
    from remove_ai_watermarks import video_visible

    source, expected = codec_clip
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: ClosedCapture())
    monkeypatch.setattr(video_visible, "probe_video_timestamps", lambda path: ())
    scan = video_visible.scan_video_marks(source, ("sora",))["sora"]
    assert (scan.width, scan.height, scan.fps) == (64, 48, 5.0)
    assert len(scan.detections) == len(expected)
    assert scan.timestamps == pytest.approx((0.7, 0.9, 1.2))


def test_scan_closes_the_fallback_when_a_detector_fails(codec_clip, monkeypatch):
    from remove_ai_watermarks import video_visible
    from remove_ai_watermarks.video_decode import open_video_capture

    source, _ = codec_clip
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: ClosedCapture())
    capture = open_video_capture(source)
    monkeypatch.setattr(video_visible, "open_video_capture", lambda path: capture)

    def fail(*args, **kwargs):
        raise RuntimeError("Detector failed")

    with pytest.raises(RuntimeError, match="Detector failed"):
        video_visible._scan_video_detectors(source, {"sora": fail})
    assert not capture.isOpened()
