"""Custom Stable Diffusion ControlNet Img2Img pipeline for CtrlRegen.

Extends ``StableDiffusionControlNetImg2ImgPipeline`` with the
``load_ctrlregen_ip_adapter`` method (via ``CustomIPAdapterMixin``)
that swaps in DINOv2-giant as the image encoder and loads the
CtrlRegen semantic-control adapter weights.

No ``encode_image`` override is needed — the CtrlRegen checkpoint
creates an ``IPAdapterPlusImageProjection`` which tells diffusers to
call ``encode_image`` with ``output_hidden_states=True``.  The
default implementation then uses ``hidden_states[-2]`` from DINOv2,
which is exactly what the projection was trained on.

Attribution:
    Adapted from https://github.com/yepengliu/CtrlRegen .
"""

from __future__ import annotations

from diffusers import StableDiffusionControlNetImg2ImgPipeline

from remove_ai_watermarks.noai.ctrlregen.ip_adapter import CustomIPAdapterMixin


class CustomCtrlRegenPipeline(
    StableDiffusionControlNetImg2ImgPipeline,
    CustomIPAdapterMixin,
):
    """SD ControlNet Img2Img pipeline with DINOv2 IP-Adapter support.

    MRO mirrors the original CtrlRegen repository: the base diffusers
    pipeline comes first so all standard methods are resolved from it,
    while ``CustomIPAdapterMixin`` only adds the
    ``load_ctrlregen_ip_adapter`` method.
    """
