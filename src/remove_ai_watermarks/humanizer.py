"""Analog Humanizer: film grain and chromatic aberration injection.

Simulates analog film imperfections to defeat digital AI perfection
classifiers. Ported from NeuralBleach.
"""

import cv2
import numpy as np
from numpy.typing import NDArray


def apply_analog_humanizer(image: NDArray, grain_intensity: float = 4.0, chromatic_shift: int = 1) -> NDArray:
    """
    Apply Analog Humanizer (film grain and chromatic aberration) to an image.
    This simulates analog film imperfections to defeat digital AI perfection classifiers.

    Ported from NeuralBleach.

    Args:
        image: BGR image as numpy array (uint8).
        grain_intensity: Standard deviation of the Gaussian noise (film grain).
        chromatic_shift: Number of pixels to shift the red/blue color channels.

    Returns:
        Humanized BGR image.
    """
    # Ensure image is BGR
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image.copy()

    # Split channels (OpenCV uses BGR)
    # B = 0, G = 1, R = 2
    b, g, r = cv2.split(image)

    # 1. Chromatic Aberration
    # Shift R channel left, B channel right
    if chromatic_shift > 0:
        r = np.roll(r, -chromatic_shift, axis=1)
        b = np.roll(b, chromatic_shift, axis=1)

    merged = cv2.merge((b, g, r))

    # 2. Film Grain (Gaussian Noise)
    if grain_intensity > 0:
        img_f = merged.astype(np.float32)
        noise = np.random.normal(0, grain_intensity, img_f.shape).astype(np.float32)
        humanized = np.clip(img_f + noise, 0, 255).astype(np.uint8)
    else:
        humanized = merged

    return humanized
