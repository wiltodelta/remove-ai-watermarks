"""Img2img pipeline execution with progress monitoring and MPS fallback.

Extracted from ``watermark_remover.py`` to keep the ``WatermarkRemover``
class focused on orchestration.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from PIL import Image

from remove_ai_watermarks.noai.progress import is_mps_error, make_pipeline_progress

logger = logging.getLogger(__name__)


def run_img2img(
    pipeline: Any,
    image: Image.Image,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    generator: Any,
    device: str,
    set_progress: Callable[[str], None],
) -> Image.Image:
    """Execute img2img with live progress and return the generated image."""
    effective_steps = max(1, int(num_inference_steps * strength))

    step_cb, first_step, done_ev, start_updater = make_pipeline_progress(
        effective_steps,
        device,
        set_progress,
    )
    start_updater()

    try:
        result = _call_pipeline(
            pipeline,
            image,
            strength,
            num_inference_steps,
            guidance_scale,
            generator,
            step_cb,
        )
        done_ev.set()
        return result.images[0]
    except TypeError:
        first_step.set()
        result = _call_pipeline(
            pipeline,
            image,
            strength,
            num_inference_steps,
            guidance_scale,
            generator,
            None,
        )
        done_ev.set()
        return result.images[0]
    finally:
        first_step.set()
        done_ev.set()


def run_img2img_with_mps_fallback(
    load_pipeline: Callable[[], Any],
    image: Image.Image,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    generator: Any,
    device: str,
    set_progress: Callable[[str], None],
    *,
    reload_on_cpu: Callable[[], Any],
) -> tuple[Image.Image, str]:
    """Run img2img; on MPS error, fall back to CPU.

    Returns:
        (result_image, final_device) — device may change to ``"cpu"`` on fallback.
    """
    pipeline = load_pipeline()

    try:
        img = run_img2img(
            pipeline,
            image,
            strength,
            num_inference_steps,
            guidance_scale,
            generator,
            device,
            set_progress,
        )
        return img, device
    except RuntimeError as error:
        if device == "mps" and is_mps_error(error):
            logger.warning("MPS error detected: %s. Falling back to CPU.", error)
            set_progress("MPS error! Clearing cache and retrying on CPU...")
            _try_clear_mps_cache()
            pipeline = reload_on_cpu()
            img = run_img2img(
                pipeline,
                image,
                strength,
                num_inference_steps,
                guidance_scale,
                None,
                "cpu",
                set_progress,
            )
            return img, "cpu"
        raise


def _call_pipeline(
    pipeline: Any,
    image: Image.Image,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    generator: Any,
    step_callback: Any,
) -> Any:
    kwargs: dict[str, Any] = {
        "prompt": "",
        "image": image,
        "strength": strength,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }
    if step_callback is not None:
        kwargs["callback"] = step_callback
        kwargs["callback_steps"] = 1
    return pipeline(**kwargs)


def _try_clear_mps_cache() -> None:
    with contextlib.suppress(Exception):
        import torch

        if hasattr(torch, "mps"):
            torch.mps.empty_cache()  # type: ignore[attr-defined]
