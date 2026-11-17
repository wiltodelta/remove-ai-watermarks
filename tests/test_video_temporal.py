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
