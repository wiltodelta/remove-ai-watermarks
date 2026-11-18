"""Regression tests for motion-compensated visible-video fill."""

from __future__ import annotations

import cv2
import numpy as np

from remove_ai_watermarks.video_temporal import (
    build_temporal_reference,
    iter_stabilized_regenerated_frames,
    stabilize_filled_frame,
    stabilize_regenerated_frames,
    temporal_residual_ratio,
)


def _translated_pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    previous = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
    previous = cv2.GaussianBlur(previous, (5, 5), 0)
    current = cv2.warpAffine(
        previous,
        np.float32(((1, 0, 2), (0, 1, 1))),
        (128, 96),
        borderMode=cv2.BORDER_REFLECT,
    )
    return previous, current


def test_motion_aligned_prior_reduces_independent_fill_error() -> None:
    previous, current = _translated_pair()
    mask = np.zeros(current.shape[:2], dtype=np.uint8)
    mask[30:66, 45:85] = 255
    rng = np.random.default_rng(11)
    current_fill = current.copy()
    noise = rng.normal(0, 18, (36, 40, 3))
    current_fill[30:66, 45:85] = np.clip(
        current_fill[30:66, 45:85].astype(np.float32) + noise,
        0,
        255,
    ).astype(np.uint8)

    stabilized = stabilize_filled_frame(
        previous,
        previous,
        mask,
        current,
        current_fill,
        mask,
    )

    hole = mask > 0
    before = float(np.mean((current_fill[hole].astype(np.float32) - current[hole]) ** 2))
    after = float(np.mean((stabilized[hole].astype(np.float32) - current[hole]) ** 2))
    assert after < before * 0.5
    assert np.array_equal(stabilized[~hole], current_fill[~hole])


def test_owned_fill_can_be_stabilized_without_full_frame_copy() -> None:
    previous, current = _translated_pair()
    mask = np.zeros(current.shape[:2], dtype=np.uint8)
    mask[30:66, 45:85] = 255
    current_fill = current.copy()
    current_fill[mask > 0] = 127

    stabilized = stabilize_filled_frame(
        previous,
        previous,
        mask,
        current,
        current_fill,
        mask,
        copy=False,
    )

    assert stabilized is current_fill


def test_scene_cut_keeps_independent_current_fill() -> None:
    previous, _current = _translated_pair()
    rng = np.random.default_rng(13)
    current = rng.integers(0, 256, previous.shape, dtype=np.uint8)
    mask = np.zeros(current.shape[:2], dtype=np.uint8)
    mask[30:66, 45:85] = 255
    current_fill = current.copy()
    current_fill[mask > 0] = 0

    stabilized = stabilize_filled_frame(
        previous,
        previous,
        mask,
        current,
        current_fill,
        mask,
    )

    assert np.array_equal(stabilized, current_fill)


def test_disjoint_prior_mask_cannot_reintroduce_old_mark() -> None:
    previous, current = _translated_pair()
    previous_mask = np.zeros(current.shape[:2], dtype=np.uint8)
    previous_mask[8:24, 8:24] = 255
    current_mask = np.zeros(current.shape[:2], dtype=np.uint8)
    current_mask[60:76, 96:112] = 255
    current_fill = current.copy()
    current_fill[current_mask > 0] = 127

    stabilized = stabilize_filled_frame(
        previous,
        previous,
        previous_mask,
        current,
        current_fill,
        current_mask,
    )

    assert np.array_equal(stabilized, current_fill)


def test_stabilize_regenerated_frames_reduces_flicker_without_source_pixel_leakage() -> None:
    rng = np.random.default_rng(23)
    base = cv2.GaussianBlur(rng.integers(32, 224, (96, 128, 3), dtype=np.uint8), (7, 7), 0)
    source_frames = [
        cv2.warpAffine(
            base,
            np.float32(((1, 0, idx * 1.5), (0, 1, idx * 0.75))),
            (128, 96),
            borderMode=cv2.BORDER_REFLECT,
        )
        for idx in range(6)
    ]
    noisy_regen = [
        np.clip(frame.astype(np.float32) + rng.normal(0.0, 14.0, frame.shape), 0, 255).astype(np.uint8)
        for frame in source_frames
    ]

    maps, baseline = build_temporal_reference(source_frames)
    raw_ratio = temporal_residual_ratio(noisy_regen, maps, baseline)
    stabilized = stabilize_regenerated_frames(source_frames, noisy_regen)
    stab_ratio = temporal_residual_ratio(stabilized, maps, baseline)

    def _frame_mse(seq_a: list[np.ndarray], seq_b: list[np.ndarray]) -> float:
        return float(
            np.mean([(a.astype(np.float32) - b.astype(np.float32)) ** 2 for a, b in zip(seq_a, seq_b, strict=True)])
        )

    raw_mse = _frame_mse(noisy_regen, source_frames)
    stab_mse = _frame_mse(stabilized, source_frames)
    assert stab_ratio < raw_ratio * 0.75
    assert stab_mse < raw_mse * 0.75

    # Zero raw-source pixel leakage: constant regenerated frames stay constant
    # even when source frames are bright/textured.
    constant_regen = [np.full_like(frame, 42) for frame in source_frames]
    no_leak = stabilize_regenerated_frames(source_frames, constant_regen)
    for out_frame in no_leak:
        assert np.array_equal(out_frame, np.full_like(out_frame, 42))


def test_stabilize_regenerated_frames_resets_across_scene_cut() -> None:
    rng = np.random.default_rng(41)
    shot_a = cv2.GaussianBlur(rng.integers(10, 80, (96, 128, 3), dtype=np.uint8), (5, 5), 0)
    shot_b = cv2.GaussianBlur(rng.integers(180, 250, (96, 128, 3), dtype=np.uint8), (5, 5), 0)
    sources = [shot_a, shot_a.copy(), shot_b, shot_b.copy()]
    regens = [
        np.full_like(shot_a, 20),
        np.full_like(shot_a, 30),
        np.full_like(shot_b, 210),
        np.full_like(shot_b, 220),
    ]

    stabilized = stabilize_regenerated_frames(sources, regens, neighbor_weight=0.30)
    # Across the hard cut between index 1 and index 2, shot B (210) must never bleed into index 1 (<= 30).
    assert int(stabilized[1].max()) <= 30
    assert int(stabilized[2].min()) >= 210


def test_iter_stabilized_regenerated_frames_streams_with_one_frame_lookahead() -> None:
    rng = np.random.default_rng(51)
    frames = [rng.integers(20, 200, (32, 48, 3), dtype=np.uint8) for _ in range(5)]
    consumed: list[int] = []

    def producer():
        for idx, frame in enumerate(frames):
            consumed.append(idx)
            yield frame, frame

    stream = iter_stabilized_regenerated_frames(producer())
    first_src, first_stab = next(stream)
    assert consumed == [0, 1]
    assert np.array_equal(first_src, frames[0])
    assert first_stab.shape == frames[0].shape


def _textured_panning_clip(frame_count: int = 5) -> list[np.ndarray]:
    height, width = 120, 160
    # Fine multi-frequency pattern inside the canvas that single-frame Telea smears badly
    canvas_w = width + 32
    cy, cx = np.mgrid[0:height, 0:canvas_w].astype(np.float32)
    b = 128.0 + 70.0 * np.sin(cx * 0.42) * np.cos(cy * 0.38)
    g = 128.0 + 65.0 * np.cos(cx * 0.31 + cy * 0.27)
    r = 128.0 + 70.0 * (((cx.astype(int) // 5) ^ (cy.astype(int) // 5)) * 2 - 1) * 0.65
    canvas = np.clip(np.stack([b, g, r], axis=2), 10, 245).astype(np.uint8)
    return [canvas[:, idx * 2 : idx * 2 + width].copy() for idx in range(frame_count)]


def test_recover_unobscured_background_restores_moving_mark_and_rejects_scene_cut() -> None:
    from remove_ai_watermarks.video_temporal import recover_unobscured_background

    clean = _textured_panning_clip(3)
    boxes = [(14, 16, 42, 38), (48, 44, 76, 66), (88, 72, 116, 94)]
    watermarked: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for frame, (x0, y0, x1, y1) in zip(clean, boxes, strict=True):
        wm = frame.copy()
        wm[y0:y1, x0:x1] = 250
        m = np.zeros(frame.shape[:2], dtype=np.uint8)
        m[y0:y1, x0:x1] = 255
        watermarked.append(wm)
        masks.append(m)

    hole = masks[1] > 0
    single_frame_inpaint = cv2.inpaint(watermarked[1], masks[1], 3.0, cv2.INPAINT_TELEA)
    inpaint_mse = float(np.mean((single_frame_inpaint[hole].astype(np.float32) - clean[1][hole]) ** 2))

    recovered, remaining = recover_unobscured_background(
        watermarked[1],
        masks[1],
        [(watermarked[0], masks[0]), (watermarked[2], masks[2])],
    )
    recovered_mse = float(np.mean((recovered[hole].astype(np.float32) - clean[1][hole]) ** 2))

    assert int(np.count_nonzero(remaining)) == 0
    assert recovered_mse < inpaint_mse * 0.10
    assert np.array_equal(recovered[~hole], watermarked[1][~hole])

    # Scene cut donor must be rejected without bleeding pixels into the target hole
    rng = np.random.default_rng(99)
    cut_donor = rng.integers(0, 40, clean[1].shape, dtype=np.uint8)
    cut_recovered, cut_remaining = recover_unobscured_background(
        watermarked[1],
        masks[1],
        [(cut_donor, None)],
    )
    assert np.array_equal(cut_remaining, masks[1])
    assert np.array_equal(cut_recovered, watermarked[1])


def test_encode_clean_video_recovers_moving_watermark_end_to_end(tmp_path) -> None:
    import shutil
    import subprocess

    import pytest

    from remove_ai_watermarks.video_decode import open_video_capture
    from remove_ai_watermarks.video_visible import FrameLocalization, VideoScan, encode_clean_video

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe required")

    clean = _textured_panning_clip(5)
    positions = [(14, 16), (46, 38), (80, 60), (46, 38), (14, 16)]
    wm_frames: list[np.ndarray] = []
    regions: list[tuple[int, int, int, int] | None] = []
    detections: list[FrameLocalization] = []
    for idx, (frame, (x, y)) in enumerate(zip(clean, positions, strict=True)):
        wm = frame.copy()
        wm[y : y + 22, x : x + 28] = 245
        wm_frames.append(wm)
        reg = (x, y, 28, 22)
        regions.append(reg)
        detections.append(FrameLocalization(frame_index=idx, confidence=0.95, region=reg))

    src_mp4 = tmp_path / "moving_wm.mp4"
    h, w = clean[0].shape[:2]
    subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{w}x{h}",
            "-r",
            "12",
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-crf",
            "10",
            "-pix_fmt",
            "yuv420p",
            str(src_mp4),
        ],
        input=b"".join(f.tobytes() for f in wm_frames),
        check=True,
    )

    scan = VideoScan(
        width=w,
        height=h,
        fps=12.0,
        detections=tuple(detections),
        timestamps=tuple(i / 12.0 for i in range(len(wm_frames))),
    )
    out_single = tmp_path / "single.mp4"
    out_temporal = tmp_path / "temporal.mp4"
    encode_clean_video(
        src_mp4,
        out_single,
        scan,
        regions,
        backend="cv2",
        strip_metadata=True,
        temporal_consistency=False,
    )
    encode_clean_video(
        src_mp4,
        out_temporal,
        scan,
        regions,
        backend="cv2",
        strip_metadata=True,
        temporal_consistency=True,
    )

    def _read_all(path):
        cap = open_video_capture(path)
        out = []
        try:
            while True:
                ok, frm = cap.read()
                if not ok:
                    break
                out.append(frm)
        finally:
            cap.release()
        return out

    single_decoded = _read_all(out_single)
    temporal_decoded = _read_all(out_temporal)
    assert len(single_decoded) == len(clean)
    assert len(temporal_decoded) == len(clean)

    # Compare interior target-box MSE against ground-truth clean frames
    single_err = 0.0
    temporal_err = 0.0
    for gt, sf_frm, tf_frm, (x, y) in zip(clean, single_decoded, temporal_decoded, positions, strict=True):
        gt_box = gt[y : y + 22, x : x + 28].astype(np.float32)
        single_err += float(np.mean((sf_frm[y : y + 22, x : x + 28].astype(np.float32) - gt_box) ** 2))
        temporal_err += float(np.mean((tf_frm[y : y + 22, x : x + 28].astype(np.float32) - gt_box) ** 2))

    assert temporal_err < single_err * 0.25
