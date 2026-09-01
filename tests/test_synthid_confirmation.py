from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_affine_lattice_probe as research_probe
from synthid_runtime._synthid_confirmation import (
    RegisteredConfirmationComponents,
    registered_confirmation_components,
)


@pytest.fixture(scope="module")
def periodic_fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260818)
    template = rng.normal(0.0, 1.0, (16, 16, 3))
    template -= np.mean(template, axis=(0, 1), keepdims=True)
    template /= np.linalg.norm(template)
    coarse = rng.normal(0.0, 8.0, (16, 16, 3)).astype(np.float32)
    background = cv2.resize(coarse, (1024, 1024), interpolation=cv2.INTER_CUBIC) + 128.0
    carrier = np.tile(template, (64, 64, 1)) * 3.0
    pixels = np.clip(np.rint(background + carrier), 0, 255).astype(np.uint8)
    return pixels, template


def test_runtime_components_match_frozen_research_seam(
    periodic_fixture: tuple[np.ndarray, np.ndarray],
) -> None:
    pixels, template = periodic_fixture

    runtime = registered_confirmation_components(pixels, template, 16.0, 1.0)
    research = research_probe.score_lattice(
        pixels,
        template,
        periods=np.asarray([16.0]),
        rotations_degrees=np.asarray([0.0]),
    )

    assert runtime.period == research.selected_period
    assert runtime.joint_coherence == pytest.approx(research.joint_coherence)
    assert runtime.joint_amplitude == pytest.approx(research.joint_amplitude)
    assert runtime.unknown_codeword_fixed_confirmation == pytest.approx(research.unknown_codeword_fixed_confirmation)
    assert runtime.selection_patches == research.selection_patches
    assert runtime.confirmation_patches == research.confirmation_patches
    assert runtime.passes


def test_confirmation_rejects_independent_noise(periodic_fixture: tuple[np.ndarray, np.ndarray]) -> None:
    _pixels, template = periodic_fixture
    pixels = np.random.default_rng(20260819).integers(0, 256, (1024, 1024, 3), dtype=np.uint8)

    result = registered_confirmation_components(pixels, template, 16.0, 1.0)

    assert not result.passes


def test_period_aware_confirmation_boundaries() -> None:
    baseline = RegisteredConfirmationComponents(
        period=16.0,
        joint_coherence=0.30,
        joint_amplitude=0.0,
        unknown_codeword_fixed_confirmation=0.5,
        selection_patches=8,
        confirmation_patches=8,
    )

    assert baseline.passes
    assert not replace(baseline, period=9.99).passes
    assert not replace(baseline, joint_coherence=0.299).passes
    assert not replace(baseline, joint_amplitude=-0.001).passes
    assert not replace(baseline, period=18.28, unknown_codeword_fixed_confirmation=0.129).passes
    assert replace(baseline, period=18.28, unknown_codeword_fixed_confirmation=0.13).passes
    assert not replace(baseline, period=19.14, joint_coherence=0.399).passes
    assert replace(baseline, period=19.14, joint_coherence=0.40).passes
    assert not replace(baseline, period=21.31, unknown_codeword_fixed_confirmation=0.019).passes
    assert replace(baseline, period=21.31, unknown_codeword_fixed_confirmation=0.02).passes
