# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "diffusers>=0.35.0",
#   "transformers>=4.51.0",
#   "torch",
#   "accelerate",
#   "pillow",
#   "click",
# ]
# ///
"""Isolated GPU prototype: does a low-strength Qwen-Image img2img pass scrub the
invisible watermark while keeping text/structure legible?

This is the oracle-gated experiment behind Library roadmap P1#5 (migrate the
invisible pipeline onto Qwen-Image-Edit). It is DELIBERATELY standalone:

  * It is NOT imported by the package and NOT in ``uv.lock``. Qwen-Image needs a
    newer ``diffusers``/``transformers`` (Qwen2.5-VL text encoder) than the SDXL
    pipeline is pinned to, so wiring it into the locked env would risk the
    certified SDXL/ControlNet pipeline (the ``cannot import Qwen3VL...`` trap).
    PEP 723 inline metadata lets ``uv run`` build a throwaway env for it instead.
  * Qwen-Image is ~20B, so it needs a real GPU (CUDA) -- it will not fit on MPS.

Run (on a GPU box / Modal), then eyeball the outputs AND submit them to the
matching oracle (the Content Provenance API first for OpenAI, the Gemini app for
Google; inspect the current order with ``provider_oracles.py plan PROVIDER``):

    uv run scripts/qwen_scrub_prototype.py INPUT.png -o out/ --strengths 0.1,0.2,0.3,0.4

What to look for:
  * SCRUB: the oracle no longer reports the watermark at some strength.
  * FIDELITY: text stays legible and faces/structure stay faithful at that same
    strength -- the whole point of trying Qwen over SDXL (which garbles text).
The smallest strength that clears the oracle while keeping fidelity is the result
to compare against the SDXL/ControlNet floors (OpenAI 0.10 / Google 0.15).
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

log = logging.getLogger("qwen_proto")

# A neutral, faithful-regeneration prompt (we want to scrub, not restyle); mirrors
# the intent of the SDXL controlnet prompt. Qwen renders text natively, so a light
# pass should keep captions legible where SDXL would garble them.
_PROMPT = "high quality, sharp, detailed, faithful to the original"
_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"


def _pick_device(requested: str) -> tuple[str, object]:
    import torch

    if requested != "auto":
        device = requested
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    # bf16 on CUDA (Qwen's reference dtype); fp32 elsewhere for numerical safety.
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return device, dtype


@click.command()
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(path_type=Path), default=Path("qwen_out"))
@click.option("--strengths", default="0.1,0.2,0.3,0.4", help="Comma-separated img2img strengths to sweep.")
@click.option("--steps", type=int, default=40, help="Inference steps.")
@click.option("--cfg", type=float, default=4.0, help="true_cfg_scale (Qwen's CFG; reference default 4.0).")
@click.option("--model", default="Qwen/Qwen-Image", help="HF model id (Qwen-Image img2img base).")
@click.option("--device", default="auto", type=click.Choice(["auto", "cuda", "mps", "cpu"]))
@click.option("--seed", type=int, default=0, help="Reproducible seed.")
def main(
    source: Path,
    output_dir: Path,
    strengths: str,
    steps: int,
    cfg: float,
    model: str,
    device: str,
    seed: int,
) -> None:
    """Sweep Qwen-Image img2img strength over SOURCE and save one output per strength."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import torch
    from diffusers import QwenImageImg2ImgPipeline
    from PIL import Image

    dev, dtype = _pick_device(device)
    log.info("Loading %s on %s (%s)...", model, dev, dtype)
    pipe = QwenImageImg2ImgPipeline.from_pretrained(model, torch_dtype=dtype)
    pipe = pipe.to(dev)

    init_image = Image.open(source).convert("RGB")
    output_dir.mkdir(parents=True, exist_ok=True)
    values = [float(s) for s in strengths.split(",") if s.strip()]

    for strength in values:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        log.info("Generating strength=%.2f ...", strength)
        result = pipe(
            prompt=_PROMPT,
            negative_prompt=_NEGATIVE,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            true_cfg_scale=cfg,
            generator=generator,
        )
        out_path = output_dir / f"{source.stem}_qwen_s{strength:.2f}.png"
        result.images[0].save(out_path)
        log.info("  saved %s", out_path)

    log.info(
        "\nDone. Eyeball text/face fidelity, then submit each output to the matching oracle "
        "(OpenAI Content Provenance API first / Gemini app). The smallest strength that clears "
        "the oracle while keeping fidelity is the number to compare against the SDXL floors "
        "(OpenAI 0.10 / Google 0.15)."
    )


if __name__ == "__main__":
    main()
