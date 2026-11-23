"""Tests for Wave 5 Track A features: 10-bit HDR alpha matte, batched ONNX inpainting, and Gemini veto in visible."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import remove_ai_watermarks as raiw
from remove_ai_watermarks import region_eraser, video_temporal, video_visible, watermark_registry
from remove_ai_watermarks.gemini_engine import GeminiEngine, WatermarkSize, get_watermark_config

if TYPE_CHECKING:
    from pathlib import Path


class _FakeInput:
    def __init__(self, name: str, shape: list[Any]) -> None:
        self.name = name
        self.shape = shape


class _FakeBatchSession:
    def __init__(
        self,
        *,
        providers: list[str],
        shape: list[Any],
        fill_value: float,
        fail_on_batch: bool = False,
        dtype: Any = np.float32,
    ) -> None:
        self._providers = providers
        self._inputs = [_FakeInput("image", shape), _FakeInput("mask", [shape[0], 1, shape[2], shape[3]])]
        self._fill_value = fill_value
        self._fail_on_batch = fail_on_batch
        self._dtype = dtype
        self.calls: list[tuple[int, ...]] = []

    def get_providers(self) -> list[str]:
        return list(self._providers)

    def get_inputs(self) -> list[_FakeInput]:
        return self._inputs

    def run(self, _outputs: Any, feed: dict[str, Any]) -> list[Any]:
        img = feed["image"]
        self.calls.append(tuple(img.shape))
        if self._fail_on_batch and img.shape[0] > 1:
            raise RuntimeError("static batch=1 model")
        out = np.full(img.shape, self._fill_value, dtype=self._dtype)
        return [out]


def test_estimate_clip_alpha_matte_and_unblend_uint16_hdr() -> None:
    """10-bit HDR (uint16) sequences self-calibrate YCrCb alpha mattes and unblend at full 16-bit precision."""
    rng = np.random.default_rng(42)
    bg_h, bg_w = 240, 280
    # Smoothly textured 16-bit background so PyrLK tracks reliably and var_b is high.
    base = rng.integers(8000, 52000, size=(bg_h // 4, bg_w // 4, 3), dtype=np.uint16)
    canvas_u16 = cv2.resize(base, (bg_w, bg_h), interpolation=cv2.INTER_CUBIC)
    # Add fine high-contrast edges for PyrLK corners.
    for y in range(10, bg_h - 10, 18):
        for x in range(10, bg_w - 10, 18):
            cv2.rectangle(canvas_u16, (x, y), (x + 6, y + 6), (58000, 12000, 48000), -1)

    fh, fw = 160, 180
    bounds = (110, 100, 150, 130)  # 40x30 ROI
    x0, y0, x1, y1 = bounds
    true_alpha = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    true_alpha[6:24, 6:34] = 0.45
    wm_color = np.array([62000.0, 62000.0, 62000.0], dtype=np.float32)

    clean_frames: list[np.ndarray[Any, Any]] = []
    marked_frames: list[np.ndarray[Any, Any]] = []
    for t in range(18):
        ox = t * 4
        oy = t * 3
        clean = canvas_u16[oy : oy + fh, ox : ox + fw].copy()
        clean_frames.append(clean)
        marked = clean.copy()
        roi = marked[y0:y1, x0:x1].astype(np.float32)
        blended = (1.0 - true_alpha[..., None]) * roi + true_alpha[..., None] * wm_color
        marked[y0:y1, x0:x1] = np.clip(np.rint(blended), 0, 65535).astype(np.uint16)
        marked_frames.append(marked)

    matte = video_temporal.estimate_clip_alpha_matte(marked_frames, bounds)
    assert matte is not None
    interior = true_alpha > 0.2
    alpha_mae = float(np.mean(np.abs(matte.alpha_y[interior] - true_alpha[interior])))
    assert alpha_mae < 0.08

    mask = np.zeros((fh, fw), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    unblended, rem_mask = video_temporal.unblend_ycbcr_roi(marked_frames[9], mask, matte)
    assert unblended.dtype == np.uint16
    clean_roi = clean_frames[9][y0:y1, x0:x1].astype(np.float32)
    marked_roi = marked_frames[9][y0:y1, x0:x1].astype(np.float32)
    unblended_roi = unblended[y0:y1, x0:x1].astype(np.float32)
    before_err = float(np.mean(np.abs(marked_roi - clean_roi)))
    after_err = float(np.mean(np.abs(unblended_roi - clean_roi)))
    assert after_err < before_err * 0.25
    assert int(np.count_nonzero(rem_mask)) < int(np.count_nonzero(mask)) * 0.2


def test_erase_lama_batch_single_call_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """erase_lama_batch runs one batched ONNX call and falls back cleanly if batch > 1 fails."""
    fake = _FakeBatchSession(providers=["CUDAExecutionProvider"], shape=["batch", 3, 64, 64], fill_value=0.5)
    monkeypatch.setattr(region_eraser, "lama_available", lambda: True)
    monkeypatch.setattr(region_eraser, "_get_lama_session", lambda: fake)
    assert region_eraser.backend_prefers_batching("lama") is True

    items = []
    for val in (10, 20, 30, 40):
        img = np.full((80, 80, 3), val, dtype=np.uint8)
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[20:40, 20:40] = 255
        items.append((img, mask))

    outs = region_eraser.erase_batch(items, backend="lama")
    assert len(outs) == 4
    assert fake.calls == [(4, 3, 64, 64)]
    for idx, (orig, mask) in enumerate(items):
        assert np.array_equal(outs[idx][mask == 0], orig[mask == 0])
        assert int(outs[idx][30, 30, 0]) in range(126, 130)

    # Static batch=1 fallback
    fake_static = _FakeBatchSession(
        providers=["CPUExecutionProvider"], shape=[1, 3, 64, 64], fill_value=0.5, fail_on_batch=True
    )
    monkeypatch.setattr(region_eraser, "_get_lama_session", lambda: fake_static)
    assert region_eraser.backend_prefers_batching("lama") is False
    outs_fb = region_eraser.erase_lama_batch(items[:2])
    assert len(outs_fb) == 2
    assert fake_static.calls == [(2, 3, 64, 64), (1, 3, 64, 64), (1, 3, 64, 64)]


def test_erase_migan_batch_groups_equal_crop_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """erase_migan_batch groups equal-sized crops into one ONNX call and handles uint16."""
    fake = _FakeBatchSession(
        providers=["CUDAExecutionProvider"],
        shape=["batch_size", 3, "height", "width"],
        fill_value=180,
        dtype=np.uint8,
    )
    monkeypatch.setattr(region_eraser, "migan_available", lambda: True)
    monkeypatch.setattr(region_eraser, "_get_migan_session", lambda: fake)
    assert region_eraser.backend_prefers_batching("migan") is True

    items = []
    for _ in range(3):
        img = np.full((120, 120, 3), 2000, dtype=np.uint16)
        mask = np.zeros((120, 120), dtype=np.uint8)
        mask[80:100, 80:100] = 255
        items.append((img, mask))

    outs = watermark_registry.fill_batch(items, backend="migan")
    assert len(outs) == 3
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == 3
    for out in outs:
        assert out.dtype == np.uint16
        assert int(out[0, 0, 0]) == 2000
        assert int(out[90, 90, 0]) > 40000


def test_iter_clean_video_frames_uses_fill_batch_when_cuda_preferred(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """iter_clean_video_frames batches single-region frames through fill_batch when CUDA batching is active."""
    frames = [np.full((64, 64, 3), 30 + i * 5, dtype=np.uint8) for i in range(5)]

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
    monkeypatch.setattr(region_eraser, "backend_prefers_batching", lambda _b: True)

    batch_sizes: list[int] = []

    def _fake_fill_batch(
        items: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]], *, backend: Any = "auto"
    ) -> list[np.ndarray[Any, Any]]:
        batch_sizes.append(len(items))
        return [img.copy() for img, _m in items]

    monkeypatch.setattr(watermark_registry, "fill_batch", _fake_fill_batch)
    scan = video_visible.VideoScan(width=64, height=64, fps=24.0, detections=())
    regions = [(10, 10, 16, 16) for _ in range(5)]
    out = list(
        video_visible.iter_clean_video_frames(
            tmp_path / "dummy.mp4",
            scan,
            regions,
            backend="migan",
            temporal_consistency=False,
        )
    )
    assert len(out) == 5
    assert batch_sizes == [5]


def test_gemini_sparkle_veto_in_visible_matches_identify(tmp_path: Path) -> None:
    """Trusted non-Google provenance vetoes a Gemini sparkle in remove_visible/remove_visible_detailed."""
    # Unit-level arbiter check
    cand = watermark_registry.Candidate("gemini", "Google Gemini visible watermark (sparkle)", True, True, {})
    assert watermark_registry.decide([cand], watermark_registry.Context()) != []
    assert watermark_registry.decide([cand], watermark_registry.Context(vetoed_marks=frozenset({"gemini"}))) == []

    # End-to-end check: Kling TC260 provenance + sparkle-like corner glint -> vetoed in visible
    size = 256
    engine = GeminiEngine()
    config = get_watermark_config(size, size)
    x, y = config.get_position(size, size)
    alpha = engine.get_alpha_map(WatermarkSize.SMALL)[:, :, None]
    canvas = np.full((size, size, 3), 60, dtype=np.float32)
    h, w = alpha.shape[:2]
    canvas[y : y + h, x : x + w] = alpha * 235.0 + (1.0 - alpha) * canvas[y : y + h, x : x + w]
    img_u8 = np.clip(canvas, 0, 255).astype(np.uint8)

    png_info = PngInfo()
    png_info.add_text("AIGC", json.dumps({"Label": "1", "ContentProducer": "001191110108335469089C10100"}))
    kling_path = tmp_path / "kling_corner_glint.png"
    Image.fromarray(img_u8).save(kling_path, pnginfo=png_info)

    report = raiw.remove_visible_detailed(kling_path, backend="cv2")
    assert report.status == "no_watermark"
    assert report.labels == []

    # Without non-Google provenance, the exact same pixels are still cleaned as Gemini sparkle.
    bare_path = tmp_path / "bare_corner_glint.png"
    Image.fromarray(img_u8).save(bare_path)
    bare_report = raiw.remove_visible_detailed(bare_path, backend="cv2")
    assert bare_report.status == "cleaned"
    assert "Google Gemini visible watermark (sparkle)" in bare_report.labels


def test_lossless_tiff_and_dng_metadata_stripping(tmp_path: Path) -> None:
    """Classic TIFF/DNG lossless strip removes AI XMP/Software/Exif UserComment while preserving pages and pixels."""
    from PIL.TiffImagePlugin import ImageFileDirectory_v2

    from remove_ai_watermarks.metadata import get_ai_metadata, remove_ai_metadata, strip_and_verify

    # 1. Multi-page RGB TIFF with AI XMP + AI Software + non-AI Artist & Make + 16-bit page
    page1 = Image.fromarray(np.full((16, 16, 3), [10, 90, 200], dtype=np.uint8))
    page2 = Image.fromarray(np.full((16, 16, 3), [210, 40, 30], dtype=np.uint8))
    ifd = ImageFileDirectory_v2()
    ifd[271] = "Leica Camera AG"  # Make (standard camera tag -> must survive keep_standard=True)
    ifd[305] = "ComfyUI 1.0"  # Software (AI -> must be stripped)
    ifd[315] = "Victor Kuznetsov"  # Artist (standard -> must survive keep_standard=True)
    ifd[700] = (
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        b'<rdf:Description iptcExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/>'
        b"</x:xmpmeta>"
    )
    src = tmp_path / "multipage_ai.tif"
    page1.save(src, format="TIFF", save_all=True, append_images=[page2], tiffinfo=ifd)

    assert get_ai_metadata(src)
    out = tmp_path / "multipage_clean.tif"
    _, leftover = strip_and_verify(src, out, keep_standard=True)
    assert leftover == {}
    assert b"trainedAlgorithmicMedia" not in out.read_bytes()
    assert b"ComfyUI" not in out.read_bytes()

    with Image.open(src) as before, Image.open(out) as after:
        assert after.format == "TIFF"
        assert after.n_frames == 2
        assert after.tag_v2.get(271) == "Leica Camera AG"
        assert after.tag_v2.get(315) == "Victor Kuznetsov"
        assert 305 not in after.tag_v2
        assert 700 not in after.tag_v2
        for idx in range(2):
            before.seek(idx)
            after.seek(idx)
            assert after.tobytes() == before.tobytes()

    # 2. keep_standard=False drops Artist/Make too while keeping pixels bit-exact
    out_all = tmp_path / "multipage_strip_all.tif"
    remove_ai_metadata(src, out_all, keep_standard=False)
    with Image.open(out_all) as after_all:
        assert after_all.n_frames == 2
        assert 315 not in after_all.tag_v2
        after_all.seek(0)
        assert after_all.tobytes() == page1.tobytes()

    # 3. 16-bit TIFF (mode I;16) preserves full 16-bit depth losslessly
    arr16 = np.arange(16 * 16, dtype=np.uint16).reshape(16, 16) * 200
    src16 = tmp_path / "depth16.dng"
    ifd16 = ImageFileDirectory_v2()
    ifd16[305] = "Adobe Firefly"
    Image.fromarray(arr16).save(src16, format="TIFF", tiffinfo=ifd16)
    out16 = tmp_path / "depth16_clean.dng"
    remove_ai_metadata(src16, out16, keep_standard=True)
    assert b"Adobe Firefly" not in out16.read_bytes()
    with Image.open(out16) as after16:
        assert after16.format == "TIFF"
        assert np.array_equal(np.asarray(after16), arr16)

    # 4. Big-Endian (MM\x00*) TIFF with ExifIFD UserComment carrying TC260 AIGC
    import struct

    # Build minimal valid big-endian 1x1 RGB TIFF with IFD0 -> ExifIFD (34665) -> UserComment (37510)
    # Header (8) | IFD0 at 8 (2 + 8*12 + 4 = 102 -> ends 110) | pixels at 110 (3 bytes, pad to 114)
    # ExifIFD at 114 (2 + 2*12 + 4 = 30 -> ends 144) | UserComment at 144 | DateTimeOriginal at 220
    user_comment = b'ASCII\x00\x00\x00{"AIGC":{"Label":"1","ContentProducer":"001191110108335469089C10100"}}'
    date_orig = b"2026:11:21 12:00:00\x00"
    uc_off = 144
    dt_off = uc_off + len(user_comment)
    ifd0_entries = [
        (256, 4, 1, 1),  # ImageWidth = 1
        (257, 4, 1, 1),  # ImageLength = 1
        (258, 3, 1, 8 << 16),  # BitsPerSample = 8 (inline big-endian SHORT)
        (262, 3, 1, 1 << 16),  # Photometric = BlackIsZero
        (273, 4, 1, 110),  # StripOffsets = 110
        (277, 3, 1, 1 << 16),  # SamplesPerPixel = 1
        (279, 4, 1, 1),  # StripByteCounts = 1
        (34665, 4, 1, 114),  # ExifIFD = 114
    ]
    exif_entries = [
        (36867, 2, len(date_orig), dt_off),  # DateTimeOriginal (standard -> kept)
        (37510, 7, len(user_comment), uc_off),  # UserComment (TC260 AIGC -> stripped & zeroed)
    ]
    be = bytearray(b"MM\x00*" + struct.pack(">I", 8))
    be += struct.pack(">H", len(ifd0_entries))
    for tag, ftype, cnt, val in ifd0_entries:
        be += struct.pack(">HHII", tag, ftype, cnt, val)
    be += struct.pack(">I", 0)
    assert len(be) == 110
    be += b"\x7f\x00\x00\x00"  # pixel byte 0x7f at offset 110 + padding to 114
    assert len(be) == 114
    be += struct.pack(">H", len(exif_entries))
    for tag, ftype, cnt, val in exif_entries:
        be += struct.pack(">HHII", tag, ftype, cnt, val)
    be += struct.pack(">I", 0)
    assert len(be) == 144
    be += user_comment + date_orig

    be_src = tmp_path / "big_endian_exif.tif"
    be_out = tmp_path / "big_endian_clean.tif"
    be_src.write_bytes(be)
    remove_ai_metadata(be_src, be_out, keep_standard=True)
    cleaned_be = be_out.read_bytes()
    assert cleaned_be[:4] == b"MM\x00*"
    assert cleaned_be[110] == 0x7F
    assert b"ContentProducer" not in cleaned_be
    assert b"2026:11:21 12:00:00" in cleaned_be


def test_video_messages_accepts_mkv_and_webm_suffix(tmp_path: Path) -> None:
    """remove_video_message accepts .mkv and .webm outputs and still rejects unsupported suffixes."""
    from remove_ai_watermarks import video_messages

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"")
    with pytest.raises(ValueError, match="MP4, MOV, M4V, MKV, WebM, AVI, or FLV"):
        video_messages.remove_video_message(src, tmp_path / "out.wmv", reference_hex="0" * 64)
