"""Research measurements of native-pixel structure and source-relative motion.

These are geometric diagnostics, not semantic object tracking, perceptual
acceptance criteria, or watermark verdicts. Sparse keypoint matches describe
only the matched part of an image; coverage and missing evidence stay explicit.
"""

from __future__ import annotations

# OpenCV's optional pixel stack exposes incomplete array and feature types.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _gray_pair(source: NDArray[Any], candidate: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
    if source.shape != candidate.shape or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("Structure measurement requires identical native RGB geometry")
    if source.dtype != np.uint8 or candidate.dtype != np.uint8:
        raise ValueError("Structure measurement requires uint8 RGB frames")
    return cv2.cvtColor(source, cv2.COLOR_RGB2GRAY), cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)


def _summary(values: NDArray[Any]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def frame_structure(source: NDArray[Any], candidate: NDArray[Any]) -> dict[str, Any]:
    """Measure bidirectional edge distances and unaligned SIFT displacement.

    Coordinates are never registered, resized, or normalized before scoring.
    A global translation therefore remains an error. Edge proximity can match
    unrelated edges in a dense texture; it must be read alongside keypoints,
    fidelity, temporal measurements, and visual review.
    """
    source_gray, candidate_gray = _gray_pair(source, candidate)
    source_edges = cv2.Canny(source_gray, 100, 200) != 0
    candidate_edges = cv2.Canny(candidate_gray, 100, 200) != 0
    source_count = int(np.count_nonzero(source_edges))
    candidate_count = int(np.count_nonzero(candidate_edges))
    row: dict[str, Any] = {
        "source_edge_count": source_count,
        "candidate_edge_count": candidate_count,
        "edge_status": "measured" if source_count and candidate_count else "unverified",
        "source_edge_distance_px": None,
        "candidate_edge_distance_px": None,
        "keypoint_matches": 0,
        "source_keypoint_hull_fraction": None,
        "keypoint_displacement_px": None,
    }
    if source_count and not candidate_count:
        row["edge_status"] = "candidate_edges_missing"
    if source_count and candidate_count:
        to_candidate = cv2.distanceTransform((~candidate_edges).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        to_source = cv2.distanceTransform((~source_edges).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        row["source_edge_distance_px"] = _summary(to_candidate[source_edges])
        row["candidate_edge_distance_px"] = _summary(to_source[candidate_edges])
    sift = cv2.SIFT.create(nfeatures=2000)
    mask = np.full(source_gray.shape, 255, dtype=np.uint8)
    source_points, source_descriptors = sift.detectAndCompute(source_gray, mask)
    candidate_points, candidate_descriptors = sift.detectAndCompute(candidate_gray, mask)
    if not source_points or len(candidate_points) < 2:
        return row
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(source_descriptors, candidate_descriptors, k=2)
    reverse = matcher.knnMatch(candidate_descriptors, source_descriptors, k=2)
    reverse_matches = {
        a.queryIdx: a.trainIdx
        for pair in reverse
        if len(pair) == 2
        for a, b in [pair]
        if a.distance < 0.75 * b.distance
    }
    matches = [
        a
        for pair in forward
        if len(pair) == 2
        for a, b in [pair]
        if a.distance < 0.75 * b.distance and reverse_matches.get(a.trainIdx) == a.queryIdx
    ]
    row["keypoint_matches"] = len(matches)
    if not matches:
        return row
    source_xy = np.asarray([source_points[match.queryIdx].pt for match in matches], dtype=np.float32)
    candidate_xy = np.asarray([candidate_points[match.trainIdx].pt for match in matches], dtype=np.float32)
    row["keypoint_displacement_px"] = _summary(np.linalg.norm(candidate_xy - source_xy, axis=1))
    height, width = source_gray.shape
    row["source_keypoint_hull_fraction"] = (
        float(cv2.contourArea(cv2.convexHull(source_xy))) / (height * width) if len(matches) >= 3 else 0.0
    )
    return row


def _flow(previous: NDArray[Any], current: NDArray[Any]) -> NDArray[Any]:
    initial = np.zeros((*previous.shape, 2), dtype=np.float32)
    return cv2.calcOpticalFlowFarneback(previous, current, initial, 0.5, 4, 21, 5, 7, 1.5, 0)


def motion_structure(
    previous_source: NDArray[Any],
    current_source: NDArray[Any],
    previous_candidate: NDArray[Any],
    current_candidate: NDArray[Any],
) -> dict[str, Any]:
    """Compare estimated flow at textured, forward/backward-consistent source pixels.

    The validity mask uses only the source. Candidate failures cannot choose
    their own easy pixels. Optical flow remains an estimator, not ground truth.
    No score is manufactured for flat frames or insufficient valid pixels.
    """
    previous_gray, previous_candidate_gray = _gray_pair(previous_source, previous_candidate)
    current_gray, current_candidate_gray = _gray_pair(current_source, current_candidate)
    if previous_gray.shape != current_gray.shape:
        raise ValueError("Motion measurement requires constant native geometry")
    forward = _flow(previous_gray, current_gray)
    backward = _flow(current_gray, previous_gray)
    height, width = previous_gray.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x, map_y = xx + forward[..., 0], yy + forward[..., 1]
    sampled_backward = cv2.remap(backward, map_x, map_y, cv2.INTER_LINEAR)
    consistent = np.linalg.norm(forward + sampled_backward, axis=2) < 1.5
    gx = cv2.Sobel(previous_gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(previous_gray, cv2.CV_32F, 0, 1)
    valid = consistent & (np.hypot(gx, gy) > 10)
    valid &= (map_x >= 1) & (map_x < width - 2) & (map_y >= 1) & (map_y < height - 2)
    row: dict[str, Any] = {
        "status": "unverified",
        "reliable_source_fraction": float(np.mean(valid)),
        "flow_error_px": None,
    }
    if np.count_nonzero(valid) < 32:
        return row
    candidate_flow = _flow(previous_candidate_gray, current_candidate_gray)
    row["status"] = "measured"
    row["flow_error_px"] = _summary(np.linalg.norm(candidate_flow - forward, axis=2)[valid])
    return row
