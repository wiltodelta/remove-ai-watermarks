"""Local, lazy perceptual distance for fidelity-profile acceptance."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image


class LocalPerceptualMetric:
    """Use the fixed AlexNet LPIPS model without loading it during construction."""

    def __init__(self) -> None:
        self._model: Any = None

    def __call__(self, reference: Image.Image, candidate: Image.Image) -> float:
        import numpy as np
        import torch
        from PIL import Image

        if self._model is None:
            import lpips

            self._model = lpips.LPIPS(net="alex", verbose=False).float().cpu().train(False)
        reference = reference.convert("RGB")
        reference.thumbnail((1024, 1024))
        candidate = candidate.convert("RGB").resize(reference.size, Image.Resampling.LANCZOS)
        tensors = [
            torch.from_numpy(np.asarray(image, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1).unsqueeze(0)
            for image in (reference, candidate)
        ]
        with torch.inference_mode():
            return float(self._model(*tensors).item())
