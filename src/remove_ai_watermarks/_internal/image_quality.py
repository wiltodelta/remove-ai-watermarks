"""Shared grayscale structural similarity for delivered-file measurements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray


def structural_similarity_gray(a: NDArray[Any], b: NDArray[Any]) -> float:
    """Grayscale SSIM (single-window Gaussian, the Wang et al. formulation)."""
    import cv2
    import numpy as np

    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    mu_a = cv2.GaussianBlur(a, k, 1.5)
    mu_b = cv2.GaussianBlur(b, k, 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = cv2.GaussianBlur(a * a, k, 1.5) - mu_a2
    sb = cv2.GaussianBlur(b * b, k, 1.5) - mu_b2
    sab = cv2.GaussianBlur(a * b, k, 1.5) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * sab + c2)) / ((mu_a2 + mu_b2 + c1) * (sa + sb + c2))
    return float(ssim_map.mean())
