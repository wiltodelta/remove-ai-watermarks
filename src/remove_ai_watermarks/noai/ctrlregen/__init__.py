"""CtrlRegen watermark removal via controllable regeneration.

Implements the pipeline from "Image Watermarks Are Removable Using
Controllable Regeneration from Clean Noise" (ICLR 2025) by Liu et al.

This sub-package uses a ControlNet for spatial guidance (canny edges)
and a DINOv2-based IP Adapter for semantic guidance to regenerate
watermarked images from partially noised latents.

Attribution:
    Based on https://github.com/yepengliu/CtrlRegen (Apache-2.0).
"""

from __future__ import annotations

from remove_ai_watermarks.noai.ctrlregen.engine import CtrlRegenEngine, is_ctrlregen_available

__all__ = ["CtrlRegenEngine", "is_ctrlregen_available"]
