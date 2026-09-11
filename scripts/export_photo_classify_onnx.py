#!/usr/bin/env python3
"""Export the frozen CLIP-L vision tower as a static FP32 ONNX graph.

Run from the repository root with the ONNX exporter installed temporarily:

    uv run --with onnx python scripts/export_photo_classify_onnx.py \
      --checkpoint /path/to/clip-l-ft.pt \
      --out /path/to/clip-l-ft-vision-fp32.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger(__name__)


class VisionEmbedding(nn.Module):
    """Only the vision modules used by the classifier's embedding path."""

    def __init__(self, headed: Any) -> None:
        super().__init__()
        self.vision_model = headed.clip.vision_model
        self.visual_projection = headed.clip.visual_projection

    def forward(self, pixel_values: Any) -> Any:
        pooled = self.vision_model(pixel_values=pixel_values).pooler_output
        return nn.functional.normalize(self.visual_projection(pooled), dim=-1)


def export_onnx(checkpoint: Path, output: Path) -> None:
    """Write the static batch-one graph consumed by the ONNX runtime backend."""
    from remove_ai_watermarks._internal.clip_l_ft import SIZE, load_headed_clip

    checkpoint = checkpoint.expanduser().resolve()
    output = output.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {checkpoint}")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = VisionEmbedding(load_headed_clip(checkpoint, torch.device("cpu"))).eval()
    export_options: dict[str, Any] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_options["dynamo"] = False
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (torch.zeros(1, 3, SIZE, SIZE, dtype=torch.float32),),
            output,
            input_names=["pixel_values"],
            output_names=["embedding"],
            opset_version=17,
            do_constant_folding=True,
            **export_options,
        )
    with output.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    log.info("wrote %s bytes=%d sha256=%s", output, output.stat().st_size, digest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    export_onnx(args.checkpoint, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
