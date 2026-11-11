"""Project-owned configuration for invisible-watermark regeneration profiles.

Four profiles are available, and all are CUDA-only: ``qwen-zimage`` (the default),
``qwen21-zimage`` (non-commercial research and personal use under the Qwen Research
License), ``sdxl-zimage`` and ``chroma-zimage``, plus the per-vendor ``auto``
selector. The older ``controlnet``, ``sdxl``, ``qwen`` and ``default`` profiles were
removed rather than kept as a CPU path, because none of them matched the two-stage
recipe's face preservation and keeping them implied a quality this library no longer
offers. Removing invisible watermarks therefore needs a CUDA device; the visible-mark
registry and every identify path still run anywhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

# SDXL base is no longer a profile of its own, but it is still the global stage of
# sdxl-zimage, so the checkpoint id stays. Named for what it is rather than
# ``DEFAULT_MODEL_ID``: there is no user-selectable model any more, so "default"
# implied an override that every profile rejects.
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_CANNY_MODEL = "xinsir/controlnet-canny-sdxl-1.0"

QWEN_ZIMAGE_PROFILE = "qwen-zimage"
QWEN21_ZIMAGE_PROFILE = "qwen21-zimage"
SDXL_ZIMAGE_PROFILE = "sdxl-zimage"
CHROMA_ZIMAGE_PROFILE = "chroma-zimage"
AUTO_PROFILE = "auto"
DEFAULT_PROFILE = QWEN_ZIMAGE_PROFILE


def fidelity_skip_reason(*, enabled: bool, tile: bool, has_reviewed_text: bool, size: tuple[int, int]) -> str | None:
    """Resolve the shared fidelity eligibility policy before strength or inference."""
    if not enabled:
        return "disabled"
    if tile:
        return "tiled_input"
    if has_reviewed_text:
        return "reviewed_text"
    if min(size) < 64:
        return "small_input"
    return None


PROFILE_CHOICES = (
    QWEN_ZIMAGE_PROFILE,
    QWEN21_ZIMAGE_PROFILE,
    SDXL_ZIMAGE_PROFILE,
    CHROMA_ZIMAGE_PROFILE,
    AUTO_PROFILE,
)

# The modules a real removal run needs, and the extra that installs them. Both live
# here, in the only profile module that imports nothing heavy, because the CLI's
# availability gate and the remover's own precondition must agree: when they drifted,
# the CLI passed on a torch+diffusers environment and the run then died at the
# DiffSynth face stage, telling the user to install an extra that does not contain it.
REMOVAL_MODULES = ("torch", "diffusers", "diffsynth")
# Shell-quoted, because this string is printed as a command the user copy-pastes.
# Bare brackets are a glob in zsh (the macOS default shell): an unquoted
# ``pip install remove-ai-watermarks[qwen-zimage]`` dies with "no matches found"
# before pip ever runs -- another install hint that does not install.
INVISIBLE_EXTRA = "'remove-ai-watermarks[qwen-zimage]'"
# The pixel stack every visible-mark command needs. The default package ships without
# it, as did legacy metadata-only Homebrew builds, so ``visible``, ``erase``, ``all`` and
# ``batch`` used to die on a bare ``ModuleNotFoundError: No module named 'cv2'``
# traceback on a first run that followed the project's own install instructions.
PIXELS_MODULES = ("cv2", "numpy")
VISIBLE_EXTRA = "'remove-ai-watermarks[visible]'"

# qwen-zimage's output already matches the input's detail level, so polishing it is a
# no-op at best. sdxl-zimage's global pass leaves the softer output the polish exists
# for. This is per-profile data, not a CLI concern: the flag defaults to None so that
# "the user did not choose" stays a value rather than an inference from Click state.
PROFILE_ADAPTIVE_POLISH = {
    QWEN_ZIMAGE_PROFILE: False,
    QWEN21_ZIMAGE_PROFILE: False,
    SDXL_ZIMAGE_PROFILE: True,
    CHROMA_ZIMAGE_PROFILE: False,
    # This fallback applies only before a vendor is known. The engine resolves
    # auto first so Google-to-SDXL receives SDXL's enabled default.
    AUTO_PROFILE: False,
}

# Which profiles let ``--cpu-offload`` reach their GLOBAL stack. The face stage
# honours the flag everywhere, because residency lives on the shared base, but the
# global stack is each profile's own: qwen-zimage and qwen21-zimage stream it through
# ``_qwen_vram_config`` and sdxl-zimage through Diffusers sequential offload.
# chroma-zimage loads its stack with a plain ``.to(device)``, so on a card below the
# face-stage residency floor the flag has nothing left to change there and the run
# behaves as if it were absent. Data rather than a subclass attribute, because the
# caller that must warn cannot import a profile module without pulling in Diffusers.
GLOBAL_OFFLOAD_PROFILES = frozenset({QWEN_ZIMAGE_PROFILE, QWEN21_ZIMAGE_PROFILE, SDXL_ZIMAGE_PROFILE})

SDXL_LIGHTNING_MODEL_ID = "ByteDance/SDXL-Lightning"
SDXL_LIGHTNING_PATTERN = "sdxl_lightning_4step_lora.safetensors"

# All profiles are certified at a fixed seed because SynthID removal near the
# strength floor is seed-dependent. The step count and CFG are not settable at all --
# each stage owns them (``GLOBAL_STEPS`` in qwen_zimage_pipeline, ``FACE_STEPS`` in two_stage_pipeline).
PROFILE_SEED = 0

# sdxl-zimage runs the qwen-zimage recipe on an SDXL global stage, and strength is
# architecture-bound. The source-fresh 2026-09-15 calibration found Google CJK
# detectable through 0.45 and clean at 0.50. The 0.50 candidate was then replicated
# clean on independent busy, flat-text, and face-plus-text carriers. OpenAI is
# the easier oracle: the profile cleared openai.com/verify at 0.1102, so 0.15
# sits above what was verified rather than on it. Unknown follows Google, the
# stricter measured cohort.
#
# Unlike qwen-zimage this is a flat vendor policy rather than a resolution curve,
# because flat values are what was measured. Every verdict above comes from a fixed
# strength at one size; no size dependence has been established for this stage.
SDXL_ZIMAGE_OPENAI_STRENGTH = 0.15
SDXL_ZIMAGE_GEMINI_STRENGTH = 0.50
SDXL_ZIMAGE_UNKNOWN_STRENGTH = SDXL_ZIMAGE_GEMINI_STRENGTH

# qwen-zimage keeps its resolution curve for unknown content, but measured vendor
# cohorts bypass it. A source-fresh 2026-09-15 sweep found Latin scenes and faces
# clean at 0.15 but CJK and Cyrillic text requiring 0.35. The 0.35 candidate was
# then independently replicated clean on busy, flat-text, and face-plus-text CJK
# carriers, so 0.35 is the content-agnostic operating floor.
QWEN_ZIMAGE_GOOGLE_STRENGTH = 0.35

# OpenAI source-fresh expansion (official Content Provenance API, 2026-09-21):
# six current gpt-image-2 carriers first cleared at 0.04375 / 0.09375 / 0.0625
# (YuNet faces) and 0.04375 / 0.03125 / 0.08125 (no faces), each bracketed to
# 0.00625 after an RGB-identical metadata-stripped positive control. Together
# with the earlier 0.06225 / 0.0695 sources this is n=8. Face and no-face ranges
# both span 0.05, while their worst boundaries differ by only 0.0125, so there
# is no content split. Add one full observed cross-source spread to the worst
# clean boundary: 0.09375 + (0.09375 - 0.03125) = 0.15625.
QWEN_ZIMAGE_OPENAI_STRENGTH = 0.15625

# Microsoft's public detector (https://ai.azure.com/nextgen/validate) returned
# Inconclusive rather than an API-level watermark-negative verdict. Three valid
# Paint sources first cleared at 0.04125,
# 0.055, and 0.095. Add one full observed cross-source spread to the worst clean
# boundary: 0.095 + (0.095 - 0.04125) = 0.14875, rounded up to 0.15. This is a
# measured corpus margin, not a universal InvisMark threshold.
QWEN_ZIMAGE_MICROSOFT_STRENGTH = 0.15

# Meta Muse Image's retained standalone IPTC AI tag is not a vendor-unique
# signature or a Content Seal pixel detection, so this floor requires an explicit
# --vendor meta override when the origin is independently known.
# Derivation (oracle meta.ai/identification, 2026-08-26/27, corpus in
# data/contentseal/): five independent 2.56 MP generations bracketed at
# lighthouse (0.0525, 0.06], fox (0.03, 0.0375], night_city (0.03, 0.0375],
# mug <= 0.03, text <= 0.015. Worst clean boundary plus one full observed
# cross-source spread: 0.06 + (0.0525 - 0.015) = 0.0975, rounded up to 0.1.
# sdxl-zimage has no measured Meta floor; its vendor map stays without a meta
# entry so an explicit --vendor meta there falls to the unknown 0.50, which is
# above this floor and therefore conservative.
QWEN_ZIMAGE_META_STRENGTH = 0.1

_QWEN_ZIMAGE_FLAT_STRENGTH_BY_VENDOR: dict[str, float] = {
    "google": QWEN_ZIMAGE_GOOGLE_STRENGTH,
    "openai": QWEN_ZIMAGE_OPENAI_STRENGTH,
    "microsoft": QWEN_ZIMAGE_MICROSOFT_STRENGTH,
    "meta": QWEN_ZIMAGE_META_STRENGTH,
}

# qwen21-zimage's 64-channel texture-fix-vae global pass preserves ~3.9 dB more
# background detail at s=0.15625 than qwen-zimage's 16-channel 8x VAE. On the
# 9-face OpenAI portrait (10_31_08 AM, where Z-Image face crops cover 46.2% of
# the frame), the global pass alone cleared at s=0.10, but the two-stage output
# with 9 Z-Image face patches remained detected at s=0.15625
# (req_3b58ff787cee4d22ba2934cb7e7e8ae4) and cleared at s=0.20
# (req_2e599929c32747a5b1a5a6c39623b802).
QWEN21_ZIMAGE_OPENAI_STRENGTH = 0.20

_QWEN21_ZIMAGE_FLAT_STRENGTH_BY_VENDOR: dict[str, float] = {
    **_QWEN_ZIMAGE_FLAT_STRENGTH_BY_VENDOR,
    "openai": QWEN21_ZIMAGE_OPENAI_STRENGTH,
}

QWEN21_FIDELITY_GOOGLE_STRENGTH = 0.45
_QWEN21_FIDELITY_STRENGTH_BY_VENDOR: dict[str, float] = {
    **_QWEN21_ZIMAGE_FLAT_STRENGTH_BY_VENDOR,
    "google": QWEN21_FIDELITY_GOOGLE_STRENGTH,
}


# sdxl-zimage always picks its strength from the vendor. qwen-zimage instead uses
# the vendor only for measured cohorts and image area for unknown content. An
# unlisted or unknown SDXL vendor falls back to the Gemini value.
_SDXL_ZIMAGE_STRENGTH_BY_VENDOR: dict[str, float] = {
    "openai": SDXL_ZIMAGE_OPENAI_STRENGTH,
    "google": SDXL_ZIMAGE_GEMINI_STRENGTH,
}

# chroma-zimage floors, from the 2026-08-29/30 four-cohort oracle calibration
# (docs/chroma1-engine-research.md; ChromaImg2ImgPipeline, neutral prompt,
# guidance 5.0, ceil(4 / strength) requested steps, seed 0). Flat per-vendor values derived
# by the same worst-clean-plus-one-spread rule as the other profiles:
#
# - OpenAI: the initial three fixtures first cleared at 0.06 / 0.06 / 0.075,
#   but two withheld carriers expanded the first-clean boundary to 0.10625 and
#   0.1375. The face-bearing cohort rule adds its observed spread to the worst
#   boundary: 0.1375 + (0.1375 - 0.075) = 0.20. Explicit Chroma uses that
#   operating point, verified clean three times on both withheld carriers.
#   Qwen cleared both withheld carriers at its then-existing 0.07675 operating
#   point; the later n=8 Qwen expansion raised its floor to 0.15625 without
#   changing the route, which still avoids Chroma's higher fidelity cost.
#   Three further withheld carriers, all zero-face by the same YuNet detector the
#   adaptive arms gate on, first cleared at 0.045 / 0.06 / 0.11 -> a zero-face
#   operating point of 0.11 + (0.11 - 0.045) = 0.175, which 0.20 already covers.
#   That is why OpenAI keeps a FLAT floor: the two content classes differ by 0.025
#   while the spread WITHIN each is 0.065 and 0.0625, so a face split would key on
#   a variable explaining a quarter of the variation it leaves behind. The earlier
#   fixture-only view (0.06 / 0.06 / 0.075, spread 0.015) understated the real
#   scatter fourfold; do not re-derive this cohort from fixtures alone.
# - Microsoft InvisMark: paint-1 (0.06, 0.08], paint-2 <= 0.04, paint-3
#   (0.06, 0.08] -> 0.08 + (0.08 - 0.04) = 0.12, rounded up to the measured
#   rung 0.125, oracle-verified clean on both worst sources. BELOW qwen's 0.15.
# - Google: the source-fresh 2026-09-15 sweep superseded the four-fixture 0.40
#   derivation: CJK remained detected at 0.40 and cleared at 0.50. The 0.50
#   candidate was independently replicated clean on busy, flat-text, and
#   face-plus-text CJK carriers. Face detection cannot select the older 0.125
#   arm safely because fresh face-plus-text carriers require more denoise and the
#   runtime has no measured text-content gate.
# - Meta Content Seal: lighthouse (0.08, 0.10], fox (0.06, 0.08], night_city
#   (0.045, 0.06], studio_mug (0.03, 0.045], text_poster <= 0.03 ->
#   0.10 + (0.10 - 0.03) = 0.17. ABOVE qwen's 0.1; the floors exist because
#   Chroma1's boundaries scatter wider than qwen's, not because the engine is
#   worse per unit strength.
CHROMA_ZIMAGE_OPENAI_STRENGTH = 0.20
CHROMA_ZIMAGE_MICROSOFT_STRENGTH = 0.125
CHROMA_ZIMAGE_GOOGLE_STRENGTH = 0.50
CHROMA_ZIMAGE_META_STRENGTH = 0.17
CHROMA_ZIMAGE_UNKNOWN_STRENGTH = CHROMA_ZIMAGE_GOOGLE_STRENGTH

_CHROMA_ZIMAGE_STRENGTH_BY_VENDOR: dict[str, float] = {
    "openai": CHROMA_ZIMAGE_OPENAI_STRENGTH,
    "google": CHROMA_ZIMAGE_GOOGLE_STRENGTH,
    "microsoft": CHROMA_ZIMAGE_MICROSOFT_STRENGTH,
    "meta": CHROMA_ZIMAGE_META_STRENGTH,
}
_ALIASES = {
    "qwen_zimage": QWEN_ZIMAGE_PROFILE,
    "qwen21_zimage": QWEN21_ZIMAGE_PROFILE,
    "sdxl_zimage": SDXL_ZIMAGE_PROFILE,
    "chroma_zimage": CHROMA_ZIMAGE_PROFILE,
    "auto": AUTO_PROFILE,
}

# The deterministic per-cohort selection policy: which profile wins on which vendor, from the
# 2026-08-29/30 four-cohort calibration (docs/chroma1-engine-research.md).
# chroma-zimage has the lower measured Microsoft floor. SDXL wins Google on
# paired quality at the independently clean operating points. Qwen wins OpenAI
# and Meta; unknown stays on Qwen, the shipped conservative fallback.
# qwen21-zimage is opt-in only and never selected by ``auto`` because
# Qwen/Qwen-Image-2.1 is licensed under the non-commercial Qwen Research License.
_ENGINE_BY_VENDOR: dict[str, str] = {
    "google": SDXL_ZIMAGE_PROFILE,
    "microsoft": CHROMA_ZIMAGE_PROFILE,
}


def resolve_auto_profile(vendor: str | None) -> str:
    """Pick the engine for --pipeline auto from the provenance vendor."""
    return _ENGINE_BY_VENDOR.get((vendor or "").casefold(), QWEN_ZIMAGE_PROFILE)


def normalize_profile(profile: str) -> str:
    """Normalize spelling and resolve the underscore spellings."""
    value = profile.strip().casefold()
    return _ALIASES.get(value, value)


def resolve_effective_profile(profile: str, vendor: str | None) -> str:
    """Resolve a configured profile to the concrete engine for one image."""
    normalized = normalize_profile(profile)
    if normalized == AUTO_PROFILE:
        return resolve_auto_profile(vendor)
    return normalized


def resolve_seed(seed: int | None) -> int:
    """Keep every profile reproducible by default."""
    return PROFILE_SEED if seed is None else seed


def resolve_adaptive_polish(adaptive_polish: bool | None, pipeline: str) -> bool:
    """Return an explicit polish choice, or the profile's calibrated default."""
    if adaptive_polish is not None:
        return adaptive_polish
    return PROFILE_ADAPTIVE_POLISH.get(normalize_profile(pipeline), True)


def global_offload_supported(pipeline: str) -> bool:
    """Whether ``--cpu-offload`` reaches this profile's global stack.

    Per-image ``auto`` runs resolve before loading a stack. An explicit preload is
    the only unresolved ``auto`` load, and that path deliberately uses qwen-zimage
    as its fallback, so it has the same offload support as qwen-zimage.
    """
    profile = resolve_effective_profile(pipeline, None)
    return profile in GLOBAL_OFFLOAD_PROFILES


def strength_default_help() -> str:
    """Describe the live default policy without duplicating its values."""
    return (
        "profile-adaptive (qwen-zimage uses resolution-adaptive denoise, with "
        f"flat OpenAI {QWEN_ZIMAGE_OPENAI_STRENGTH} / Google {QWEN_ZIMAGE_GOOGLE_STRENGTH} / "
        f"Microsoft InvisMark {QWEN_ZIMAGE_MICROSOFT_STRENGTH} / Meta Content Seal "
        f"{QWEN_ZIMAGE_META_STRENGTH} floors; qwen21-zimage uses the same curve and floors "
        f"with OpenAI {QWEN21_ZIMAGE_OPENAI_STRENGTH}, local "
        f"quality acceptance and Google {QWEN21_FIDELITY_GOOGLE_STRENGTH}; sdxl-zimage "
        f"uses OpenAI {SDXL_ZIMAGE_OPENAI_STRENGTH} / Google {SDXL_ZIMAGE_GEMINI_STRENGTH} / "
        f"unknown {SDXL_ZIMAGE_UNKNOWN_STRENGTH}; chroma-zimage uses OpenAI "
        f"{CHROMA_ZIMAGE_OPENAI_STRENGTH} / Microsoft {CHROMA_ZIMAGE_MICROSOFT_STRENGTH} / "
        f"Google {CHROMA_ZIMAGE_GOOGLE_STRENGTH} / Meta {CHROMA_ZIMAGE_META_STRENGTH}, from the C2PA issuer)"
    )


def resolve_strength(
    strength: float | None,
    vendor: str | None = None,
    pipeline: str | None = None,
    *,
    size: tuple[int, int] | None = None,
    fidelity_enabled: bool = True,
) -> float:
    """Resolve a user override or the calibrated policy for a profile and vendor.

    Total by design. qwen-zimage and qwen21-zimage pick their strength from image
    area rather than from the vendor for unknown content, so they need ``size``;
    returning ``None`` instead would push that branch onto every caller and move
    one of the two strength policies outside this module. ``size`` is required for
    qwen-zimage and qwen21-zimage without an explicit strength. Measured vendor
    exceptions bypass the area curve: OpenAI and Microsoft use one additional
    observed cross-source spread over their worst clean boundaries, while Google
    uses a separately repeated cross-source candidate. See the constants' comments
    for the exact derivations. ``fidelity_enabled=False`` selects the legacy
    Qwen-2.1 operating point for unsupported or explicitly disabled fidelity paths;
    it does not change other profiles or explicit strength overrides.
    """
    if strength is not None:
        return strength
    normalized = resolve_effective_profile(pipeline or "", vendor)
    if normalized == SDXL_ZIMAGE_PROFILE:
        return _SDXL_ZIMAGE_STRENGTH_BY_VENDOR.get((vendor or "").casefold(), SDXL_ZIMAGE_UNKNOWN_STRENGTH)
    if normalized == CHROMA_ZIMAGE_PROFILE:
        vendor_key = (vendor or "").casefold()
        return _CHROMA_ZIMAGE_STRENGTH_BY_VENDOR.get(vendor_key, CHROMA_ZIMAGE_UNKNOWN_STRENGTH)
    if size is None:
        raise ValueError(f"{normalized or 'qwen-zimage'} resolves strength from image area, so size is required")
    vendor_map = (
        _QWEN21_FIDELITY_STRENGTH_BY_VENDOR
        if normalized == QWEN21_ZIMAGE_PROFILE and fidelity_enabled
        else _QWEN21_ZIMAGE_FLAT_STRENGTH_BY_VENDOR
        if normalized == QWEN21_ZIMAGE_PROFILE
        else _QWEN_ZIMAGE_FLAT_STRENGTH_BY_VENDOR
    )
    vendor_strength = vendor_map.get((vendor or "").casefold())
    if vendor_strength is not None:
        return vendor_strength
    from remove_ai_watermarks._internal.two_stage_pipeline import resolution_adaptive_denoise

    return resolution_adaptive_denoise(*size)


def vendor_for_strength(image_path: Path) -> Literal["openai", "google", "microsoft", "meta"] | None:
    """Select the strength cohort from non-invalid pixel-watermark provenance.

    OpenAI / Google / Microsoft come from product-specific C2PA evidence. Meta has no
    automatic route because its companion IPTC ``trainedAlgorithmicMedia`` value is a
    shared standard, not manufacturer evidence; callers can select it explicitly."""
    try:
        from remove_ai_watermarks._internal.c2pa import (
            c2pa_info_has_invalid_credential,
            c2pa_info_has_invismark,
            extract_c2pa_info,
            soft_binding_vendors_in,
        )
        from remove_ai_watermarks.metadata import c2pa_marker_in, scan_head, synthid_source

        info = extract_c2pa_info(image_path)
        evidence = (synthid_source(image_path, c2pa_info=info) or "").casefold()
    except Exception:
        return None
    if "google" in evidence:
        return "google"
    if "openai" in evidence:
        return "openai"
    if c2pa_info_has_invalid_credential(info):
        return None
    if c2pa_info_has_invismark(info):
        return "microsoft"
    if info.get("c2pa_validation_source") != "reader":
        try:
            data = scan_head(image_path)
            if c2pa_marker_in(data) and "Microsoft InvisMark" in soft_binding_vendors_in(data):
                return "microsoft"
        except Exception:
            return None
    return None
