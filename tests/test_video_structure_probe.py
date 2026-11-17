"""Known geometric counterexamples for the research structure measurements."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

probe = import_module("video_structure_probe")


def texture() -> np.ndarray:
    pixels = np.random.default_rng(834).integers(0, 256, (192, 256, 3), dtype=np.uint8)
    return cv2.GaussianBlur(pixels, (5, 5), 0)


def shifted(frame: np.ndarray, pixels: int) -> np.ndarray:
    transform = np.float32([[1, 0, pixels], [0, 1, 0]])
    return cv2.warpAffine(frame, transform, (frame.shape[1], frame.shape[0]))


def test_identity_and_translation_are_distinguished():
    source = texture()
    same = probe.frame_structure(source, source)
    moved = probe.frame_structure(source, shifted(source, 12))
    assert same["keypoint_displacement_px"]["p95"] == pytest.approx(0, abs=0.001)
    assert moved["keypoint_matches"] > 50
    assert moved["keypoint_displacement_px"]["p50"] == pytest.approx(12, abs=0.25)
    assert moved["source_keypoint_hull_fraction"] > 0.5
    assert same["source_edge_distance_px"]["p95"] == 0
    assert moved["source_edge_distance_px"]["p95"] > 0


def test_blank_or_erased_frames_do_not_receive_perfect_structure_scores():
    blank = np.zeros((192, 256, 3), dtype=np.uint8)
    no_edges = probe.frame_structure(blank, blank)
    erased = probe.frame_structure(texture(), blank)
    assert no_edges["edge_status"] == "unverified"
    assert no_edges["keypoint_displacement_px"] is None
    assert erased["edge_status"] == "candidate_edges_missing"
    assert erased["source_edge_distance_px"] is None


def test_native_geometry_is_required():
    with pytest.raises(ValueError, match="geometry"):
        probe.frame_structure(texture(), texture()[:-1])


def test_frozen_output_changes_motion_even_when_one_frame_is_exact():
    previous = texture()
    current = shifted(previous, 4)
    same = probe.motion_structure(previous, current, previous, current)
    frozen = probe.motion_structure(previous, current, previous, previous)
    assert same["flow_error_px"]["p95"] == pytest.approx(0, abs=0.001)
    assert frozen["reliable_source_fraction"] > 0.5
    assert frozen["flow_error_px"]["p50"] > 3


def test_textureless_motion_is_explicitly_unverified():
    blank = np.zeros((192, 256, 3), dtype=np.uint8)
    result = probe.motion_structure(blank, blank, blank, blank)
    assert result["status"] == "unverified"
    assert result["flow_error_px"] is None
