"""Tile-based processing for large images in the CtrlRegen pipeline.

Extracted from ``ctrlregen.engine`` to keep the engine focused on
single-image inference and model orchestration.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import torch
from PIL import Image


def tile_positions(total: int, tile: int, overlap: int) -> list[int]:
    """Compute evenly-spaced tile start positions covering *total* pixels."""
    if total <= tile:
        return [0]
    n = max(2, math.ceil((total - overlap) / (tile - overlap)))
    stride = (total - tile) / (n - 1)
    return [round(i * stride) for i in range(n)]


def make_blend_weight(h: int, w: int, overlap: int) -> np.ndarray:
    """2-D weight mask: 1.0 in center, cosine ramp in overlap margins."""
    wy = np.ones(h, dtype=np.float64)
    wx = np.ones(w, dtype=np.float64)
    if overlap > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, overlap))
        wy[:overlap] = np.minimum(wy[:overlap], ramp)
        wy[-overlap:] = np.minimum(wy[-overlap:], ramp[::-1])
        wx[:overlap] = np.minimum(wx[:overlap], ramp)
        wx[-overlap:] = np.minimum(wx[-overlap:], ramp[::-1])
    return np.outer(wy, wx)


def resize_center_crop(image: Image.Image, size: int = 512) -> Image.Image:
    """Resize shortest edge to *size*, then center-crop to a square.

    Matches the ``transforms.Resize(512) + CenterCrop(512)`` pipeline
    used in the original CtrlRegen repository.
    """
    w, h = image.size
    short = min(w, h)
    scale = size / short
    new_w, new_h = round(w * scale), round(h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return image.crop((left, top, left + size, top + size))


def run_tiled(
    pipeline: Any,
    canny_detector: Any,
    image: Image.Image,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int | None,
    *,
    tile_size: int,
    tile_overlap: int,
    quality_prompt: str,
    negative_prompt: str,
    canny_low: int,
    canny_high: int,
    device: str,
    set_progress: Callable[[str], None],
    ip_adapter_image: Image.Image | None = None,
) -> Image.Image:
    """Split a large image into overlapping tiles, process each, blend."""
    w, h = image.size
    xs = tile_positions(w, tile_size, tile_overlap)
    ys = tile_positions(h, tile_size, tile_overlap)
    n_tiles = len(xs) * len(ys)
    grid = f"{len(xs)}x{len(ys)}"
    effective_steps = max(1, int(num_inference_steps * strength))

    set_progress(f"Tiling {w}x{h}px → {n_tiles} tiles ({grid} grid, {tile_size}px, overlap {tile_overlap}px)")

    canvas = np.zeros((h, w, 3), dtype=np.float64)
    weight_sum = np.zeros((h, w), dtype=np.float64)
    blend_w = make_blend_weight(tile_size, tile_size, tile_overlap)

    t0 = time.monotonic()
    bar_len = 20
    tile_idx = 0

    for ty in ys:
        for tx in xs:
            tile_idx += 1
            prefix = f"[Tile {tile_idx}/{n_tiles}]"

            tile = image.crop((tx, ty, tx + tile_size, ty + tile_size))

            set_progress(f"{prefix} Extracting canny edges...")
            control = canny_detector(
                tile,
                low_threshold=canny_low,
                high_threshold=canny_high,
            )

            gen = None
            if seed is not None:
                gen = torch.Generator(device=device).manual_seed(seed + tile_idx)

            tile_t0 = time.monotonic()

            def _make_cb(
                _prefix: str = prefix,
                _t0: float = tile_t0,
                _es: int = effective_steps,
            ) -> Callable:
                def _cb(step: int, timestep: int, latents: Any) -> None:
                    elapsed = time.monotonic() - _t0
                    cur = step + 1
                    per = elapsed / max(1, cur)
                    rem = per * max(0, _es - cur)
                    filled = int(bar_len * cur / max(1, _es))
                    bar = "█" * filled + "░" * (bar_len - filled)
                    set_progress(f"{_prefix} [{bar}] {cur}/{_es} | {elapsed:.0f}s, ~{rem:.0f}s left")

                return _cb

            sem_image = ip_adapter_image if ip_adapter_image is not None else tile

            try:
                result = pipeline(
                    prompt=quality_prompt,
                    negative_prompt=negative_prompt,
                    image=[tile],
                    control_image=[control],
                    controlnet_conditioning_scale=1.0,
                    ip_adapter_image=[sem_image],
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=gen,
                    control_guidance_start=0.0,
                    control_guidance_end=1.0,
                    callback=_make_cb(),
                    callback_steps=1,
                )
            except TypeError:
                result = pipeline(
                    prompt=quality_prompt,
                    negative_prompt=negative_prompt,
                    image=[tile],
                    control_image=[control],
                    controlnet_conditioning_scale=1.0,
                    ip_adapter_image=[sem_image],
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=gen,
                )

            proc_arr = np.array(result.images[0], dtype=np.float64)
            th, tw = proc_arr.shape[:2]
            mask = blend_w[:th, :tw]
            canvas[ty : ty + th, tx : tx + tw] += proc_arr * mask[..., None]
            weight_sum[ty : ty + th, tx : tx + tw] += mask

            tile_time = time.monotonic() - tile_t0
            total_elapsed = time.monotonic() - t0
            set_progress(f"{prefix} Done ({tile_time:.0f}s) · Total: {total_elapsed:.0f}s")

    set_progress(f"Blending {n_tiles} tiles → {w}x{h}px...")
    canvas /= np.maximum(weight_sum[..., None], 1e-8)
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
