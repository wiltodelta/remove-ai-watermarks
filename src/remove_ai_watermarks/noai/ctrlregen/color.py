"""Color matching post-processing for CtrlRegen output.

After diffusion-based regeneration, the output image may have slight
color shifts.  This module uses histogram-based color transfer to
align the regenerated image's color distribution back to the original.

Attribution:
    Adapted from https://github.com/yepengliu/CtrlRegen (Apache-2.0).
"""

from __future__ import annotations

import numpy as np
from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer
from PIL import Image


def color_match(reference: Image.Image, source: Image.Image) -> Image.Image:
    """Transfer the color distribution of *reference* onto *source*.

    Uses a two-pass histogram matching approach (``hm-mkl-hm``) that
    preserves fine-grained color relationships while correcting global
    shifts introduced by the regeneration pipeline.

    Args:
        reference: The original (watermarked) image whose colors should
            be preserved.
        source: The regenerated image whose colors will be adjusted.

    Returns:
        A new PIL Image with the structure of *source* but the color
        palette of *reference*.
    """
    cm = ColorMatcher()
    ref_np = Normalizer(np.asarray(reference)).type_norm()
    src_np = Normalizer(np.asarray(source)).type_norm()
    result = cm.transfer(src=src_np, ref=ref_np, method="hm-mkl-hm")
    result = Normalizer(result).uint8_norm()
    return Image.fromarray(result)
