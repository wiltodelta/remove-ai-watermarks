"""Watermark removal model profiles, the default strength, and profile detection.

Pure configuration and lookup functions with no ML dependencies.
"""

from __future__ import annotations

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CTRLREGEN_MODEL_ID = "yepengliu/ctrlregen"

# Single default denoising strength for the SDXL img2img scrub, overridable from
# the CLI (`--strength`). Raised 0.10 -> 0.30 after an oracle-verified GPU strength
# study (2026-05-31, Modal A100, native res, Gemini-app "Verify with SynthID", n=3
# FRESH Gemini images + protect_text/faces OFF): the CURRENT Google SynthID survives
# 0.10/0.15/0.2 and is only REMOVED at 0.3 (0.3 is the threshold; 0.2 still present).
# This supersedes the earlier n=1 "0.10 removes it" note, which is now stale -- Google
# has hardened SynthID and the threshold has climbed 0.05 -> 0.10 -> ~0.3 over time, so
# treat this as a moving target and re-test against fresh Gemini output periodically.
# Cost of 0.3: SSIM ~0.97 vs original (modest), but fine/dense typography softens, and
# it is OVERKILL for non-SynthID sources (OpenAI/ChatGPT carry C2PA, not Google SynthID
# -- 0.10 is plenty there). Two known tensions, documented but not auto-handled here:
# (1) higher strength deforms text more (why text protection runs by default), and
# (2) `protect_text` SHIELDS the text regions where SynthID hides, so text-region
# SynthID can survive at 0.3 unless `--no-protect-text` is passed. (Fixed LOW/MEDIUM/
# HIGH presets were removed -- the one knob is this default + the per-call override.)
DEFAULT_STRENGTH = 0.30

# CtrlRegen removes watermarks by regenerating from (near) clean Gaussian noise,
# NOT by the light-touch partial-noise img2img the SDXL default uses. The research
# is explicit (CtrlRegen, ICLR 2025, arXiv:2410.05470): partial-noise regeneration
# "struggles with high-perturbation watermarks" because a small noise step "retains"
# watermark information that diffuses back into the output; the fix is to start from
# clean noise. With the StableDiffusionControlNetImg2ImgPipeline that maps to a high
# strength (~1.0 = full noise at the first timestep, structure held by the canny
# ControlNet + DINOv2 IP-Adapter, not by the watermarked latent). So the ctrlregen
# profile must NOT inherit the SDXL default (`DEFAULT_STRENGTH`, a partial-noise
# value) -- at that low strength it loads ControlNet + DINOv2-giant and then barely
# changes the image (a no-op for removal). Tunable via
# `--strength`; lower it to trade removal strength for fidelity (the CtrlRegen+ regime).
#
# EXPERIMENTAL -- NOT recommended for production. The same GPU study that set the 0.3
# SDXL threshold tested ctrlregen at its clean-noise strength and found it DESTROYS
# images: smooth/background regions fill with hallucinated micro-text garbage, and it
# is heavy (~8.5 min / ~$0.30 vs ~25 s / ~$0.02 for SDXL on a large image). The pipeline
# is effectively binary -- low strength = no-op, high strength = destroys -- with no
# usable middle, so the literature's "clean-noise is the lever" (arXiv:2410.05470) did
# NOT survive empirical testing on real content. SDXL img2img at ~0.3 is the shippable
# path; ctrlregen stays opt-in and flagged experimental.
CTRLREGEN_DEFAULT_STRENGTH = 1.0


def resolve_strength(strength: float | None, profile: str) -> float:
    """Resolve the denoising strength, applying the profile-specific default when unset.

    ``None`` means "the user did not pass ``--strength``": the SDXL default profile
    resolves to ``DEFAULT_STRENGTH`` (the SynthID-removal default, ~0.3), while
    ``ctrlregen`` resolves to ``CTRLREGEN_DEFAULT_STRENGTH`` (clean-noise regeneration).
    An explicit value always wins. Shared by the CLI (for display) and the engine (for
    execution) so the two never disagree.
    """
    if strength is not None:
        return strength
    return CTRLREGEN_DEFAULT_STRENGTH if profile == "ctrlregen" else DEFAULT_STRENGTH


def get_model_id_for_profile(profile: str) -> str:
    """Map CLI model profile names to concrete Hugging Face model IDs."""
    normalized = profile.strip().lower()
    if normalized == "default":
        return DEFAULT_MODEL_ID
    if normalized == "ctrlregen":
        return CTRLREGEN_MODEL_ID
    raise ValueError(f"Unknown model profile '{profile}'. Use one of: default, ctrlregen.")


def detect_model_profile(model_id: str) -> str:
    """Infer model profile from model identifier."""
    if "ctrlregen" in model_id.lower():
        return "ctrlregen"
    return "default"
