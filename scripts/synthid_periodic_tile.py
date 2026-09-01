"""Compatibility imports for shared periodic-residual helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthid_runtime.synthid_detector import fold_residual_template, unit_tile

__all__ = ["cyclic_tile_correlations", "fold_residual_template", "unit_tile"]


def cyclic_tile_correlations(template: np.ndarray, tile: np.ndarray) -> np.ndarray:
    """Return correlations for every cyclic spatial shift of TILE."""
    if template.shape != tile.shape or template.ndim != 3:
        raise ValueError("template and tile must have identical three-dimensional shapes")
    template_spectrum = np.fft.fft2(template, axes=(0, 1))
    tile_spectrum = np.fft.fft2(tile, axes=(0, 1))
    return np.fft.ifft2(np.sum(template_spectrum * np.conj(tile_spectrum), axis=2)).real
