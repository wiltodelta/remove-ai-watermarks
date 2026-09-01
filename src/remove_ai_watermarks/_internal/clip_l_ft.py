# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false
"""CLIP-L letterbox embedder for the frozen photo detector.

The freeze fine-tuned the last two vision blocks of
``openai/clip-vit-large-patch14``. Inference letterboxes to 224 with pad
``(123, 117, 104)`` and L2-normalizes ``get_image_features``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PIL import Image
from torch import nn

if TYPE_CHECKING:
    from pathlib import Path

    import torch
    from numpy.typing import NDArray

BACKBONE = "openai/clip-vit-large-patch14"
SIZE = 224
PAD_RGB = (123, 117, 104)


def letterbox(image: Image.Image, size: int = SIZE) -> Image.Image:
    """Return a square RGB letterbox matching the freeze preprocessor."""
    image = image.convert("RGB")
    width, height = image.size
    scale = size / max(width, height)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), PAD_RGB)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


class HeadedCLIP(nn.Module):
    """Checkpoint layout of ``clip-l-ft.pt``: CLIP plus a unused linear head."""

    def __init__(self, clip: Any) -> None:
        super().__init__()
        self.clip = clip
        self.head = nn.Linear(clip.config.projection_dim, 1)

    def embed(self, pixel_values: Any) -> Any:
        outputs = self.clip.get_image_features(pixel_values=pixel_values)
        vectors = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        return nn.functional.normalize(vectors, dim=-1)


def load_headed_clip(checkpoint: Path, device: torch.device) -> HeadedCLIP:
    """Build CLIP-L from config and load the freeze state dict."""
    import torch
    from transformers import CLIPConfig, CLIPModel

    clip = CLIPModel(CLIPConfig.from_pretrained(BACKBONE))
    model = HeadedCLIP(clip)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_processor() -> Any:
    from transformers import CLIPImageProcessor

    processor = CLIPImageProcessor.from_pretrained(BACKBONE)
    processor.do_resize = False
    processor.do_center_crop = False
    return processor


def embed_image(model: HeadedCLIP, processor: Any, image: Image.Image, device: torch.device) -> NDArray[Any]:
    """Return a 768-d L2-normalized CLIP-L-ft vector."""
    import numpy as np
    import torch

    boxed = letterbox(image)
    pixels = processor(images=[boxed], return_tensors="pt")["pixel_values"].to(
        device, dtype=next(model.parameters()).dtype
    )
    with torch.inference_mode():
        vector = model.embed(pixels).float().cpu().numpy()[0]
    return np.asarray(vector, dtype=np.float64)
