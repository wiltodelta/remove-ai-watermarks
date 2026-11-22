"""Tests for Wave 5 Track D2: 16-bit temporal recovery, grab(), multi-region CUDA batching, streaming ISOBMFF."""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import piexif

from remove_ai_watermarks import metadata, video_decode, video_temporal, video_visible, watermark_registry
from remove_ai_watermarks._internal import isobmff

if TYPE_CHECKING:
    import pytest


def test_uint16_recover_unobscured_background_and_bidirectional_stabilize() -> None:
    """10-bit HDR (uint16) frames run donor recovery and bidirectional stabilization at full 16-bit depth."""
    rng = np.random.default_rng(7)
    base = rng.integers(6000, 56000, size=(32, 32, 3), dtype=np.uint16)
    canvas = cv2.resize(base, (128, 128), interpolation=cv2.INTER_CUBIC)
    for y in range(8, 120, 16):
        for x in range(8, 120, 16):
            cv2.rectangle(canvas, (x, y), (x + 5, y + 5), (60000, 10000, 50000), -1)

    src0 = canvas[0:96, 0:96].copy()
    src1 = canvas[0:96, 2:98].copy()
    src2 = canvas[0:96, 4:100].copy()

    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[32:56, 32:56] = 255

    # Donor frame has no mask -> should recover clean 16-bit pixels inside mask
    marked1 = src1.copy()
    marked1[mask > 0] = 65000
    recovered, rem_mask = video_temporal.recover_unobscured_background(marked1, mask, [(src0, None), (src2, None)])
    assert recovered.dtype == np.uint16
    assert int(np.count_nonzero(rem_mask)) < int(np.count_nonzero(mask)) * 0.2
    before_err = float(np.mean(np.abs(marked1[mask > 0].astype(np.float32) - src1[mask > 0].astype(np.float32))))
    after_err = float(np.mean(np.abs(recovered[mask > 0].astype(np.float32) - src1[mask > 0].astype(np.float32))))
    assert after_err < before_err * 0.25

    # Bidirectional stabilization preserves uint16 and smooths noisy center fill toward clean neighbors
    noisy_center = src1.copy()
    noisy_center[mask > 0] = np.clip(src1[mask > 0].astype(np.int32) + 8000, 0, 65535).astype(np.uint16)
    stab = video_temporal.stabilize_filled_frame_bidirectional(
        src1,
        noisy_center,
        mask,
        previous=(src0, src0, mask),
        nxt=(src2, src2, mask),
    )
    assert stab.dtype == np.uint16
    noisy_err = float(np.mean(np.abs(noisy_center[mask > 0].astype(np.float32) - src1[mask > 0].astype(np.float32))))
    stab_err = float(np.mean(np.abs(stab[mask > 0].astype(np.float32) - src1[mask > 0].astype(np.float32))))
    assert stab_err < noisy_err * 0.65


def test_ffmpeg_16bit_capture_grab_skips_without_copy() -> None:
    """_FFmpeg16BitCapture supports grab()/retrieve() so skipped frames avoid NumPy allocation."""
    cap = object.__new__(video_decode._FFmpeg16BitCapture)
    cap._width = 4
    cap._height = 2
    cap._frame_bytes = 4 * 2 * 6
    cap._raw_buf = bytearray(cap._frame_bytes)
    cap._raw_view = memoryview(cap._raw_buf)
    cap._has_grabbed = False
    cap._closed = False

    f0 = np.full((2, 4, 3), 1000, dtype="<u2")
    f1 = np.full((2, 4, 3), 50000, dtype="<u2")

    class _FakeProc:
        stdout = io.BytesIO(f0.tobytes() + f1.tobytes())

    cap._proc = _FakeProc()  # type: ignore[assignment]
    assert cap.grab() is True  # skip frame 0
    ok, frame1 = cap.read()
    assert ok is True
    assert frame1 is not None
    assert frame1.dtype == np.uint16
    assert int(frame1[0, 0, 0]) == 50000
    assert cap.grab() is False


def test_iter_clean_video_frames_batches_multi_region_slots_on_cuda(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Multi-mark frames (2 disjoint regions per frame) batch each slot across the 8-frame window on CUDA."""
    frames = [np.full((128, 128, 3), 40 + i, dtype=np.uint8) for i in range(4)]

    class _FakeCap:
        def __init__(self) -> None:
            self._idx = 0

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray[Any, Any] | None]:
            if self._idx >= len(frames):
                return False, None
            f = frames[self._idx]
            self._idx += 1
            return True, f

        def release(self) -> None:
            pass

    monkeypatch.setattr(video_visible, "open_video_capture", lambda _src: _FakeCap())
    monkeypatch.setattr(watermark_registry, "resolve_backend", lambda _b: "migan")
    monkeypatch.setattr(watermark_registry, "backend_prefers_batching", lambda _b: True)

    batch_sizes: list[int] = []

    def _fake_fill_batch(
        items: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]], *, backend: Any = "auto"
    ) -> list[np.ndarray[Any, Any]]:
        batch_sizes.append(len(items))
        return [img.copy() for img, _m in items]

    monkeypatch.setattr(watermark_registry, "fill_batch", _fake_fill_batch)
    scan = video_visible.VideoScan(width=128, height=128, fps=24.0, detections=())
    # Two disjoint corners per frame -> 2 batched calls of size 4 (instead of 8 sequential calls)
    regions = [((4, 4, 16, 16), (104, 104, 16, 16)) for _ in range(4)]
    out = list(
        video_visible.iter_clean_video_frames(
            tmp_path / "dual.mp4",
            scan,
            regions,
            backend="migan",
            temporal_consistency=False,
        )
    )
    assert len(out) == 4
    assert batch_sizes == [4, 4]


def test_streaming_isobmff_image_strip_avoids_read_bytes_and_blanks_all_layers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """HEIF/AVIF/JXL metadata stripping streams in bounded chunks without calling Path.read_bytes()."""

    def _box(btype: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", 8 + len(payload)) + btype + payload

    ftyp = _box(b"ftyp", b"avif\x00\x00\x00\x00avifmif1")
    c2pa_uuid = _box(b"uuid", isobmff.C2PA_UUID + b"c2pa-manifest-payload")
    xmp = (
        b'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        b'<x:xmpmeta><rdf:Description iptcExt:DigitalSourceType="trainedAlgorithmicMedia"/></x:xmpmeta>'
        b'<?xpacket end="w"?>'
    )
    exif_bytes = piexif.dump({"0th": {piexif.ImageIFD.Software: b"ComfyUI 1.0", piexif.ImageIFD.Make: b"Canon"}})
    # Place XMP & EXIF inside an mdat box across a >1 MiB boundary to test sliding overlap
    pad_a = b"\xaa" * (isobmff._STREAM_COPY_BYTES - 100)
    pad_b = b"\xbb" * 2048
    mdat = _box(b"mdat", pad_a + xmp + pad_b + exif_bytes + b"\xcc" * 512)
    src = tmp_path / "large.avif"
    out = tmp_path / "large_clean.avif"
    src.write_bytes(ftyp + c2pa_uuid + mdat)
    orig_size = src.stat().st_size

    real_read_bytes = Path.read_bytes

    def _forbid_src_read_bytes(self: Path) -> bytes:
        if self.resolve() == src.resolve():
            raise AssertionError("Streaming HEIF/AVIF/JXL stripper must not call Path.read_bytes() on source")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _forbid_src_read_bytes)
    metadata.remove_ai_metadata(src, out)

    cleaned = real_read_bytes(out)
    assert len(cleaned) == orig_size
    assert b"c2pa-manifest-payload" not in cleaned
    assert b"trainedAlgorithmicMedia" not in cleaned
    assert b"ComfyUI 1.0" not in cleaned
    assert b"Canon" in cleaned
    # Pixel padding bytes untouched at exact offsets
    mdat_start = len(ftyp) + len(c2pa_uuid) + 8
    assert cleaned[mdat_start : mdat_start + len(pad_a)] == pad_a
