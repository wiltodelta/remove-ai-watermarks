"""Project-owned configuration for invisible-watermark regeneration profiles.

Three profiles remain, and all are CUDA-only: ``qwen-zimage`` (the default),
``sdxl-zimage``. The older ``controlnet``, ``sdxl``, ``qwen`` and ``default`` profiles
were removed rather than kept as a CPU path, because none of them matched the two-stage
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
SDXL_ZIMAGE_PROFILE = "sdxl-zimage"
CHROMA_ZIMAGE_PROFILE = "chroma-zimage"
AUTO_PROFILE = "auto"
DEFAULT_PROFILE = QWEN_ZIMAGE_PROFILE
PROFILE_CHOICES = (QWEN_ZIMAGE_PROFILE, SDXL_ZIMAGE_PROFILE, CHROMA_ZIMAGE_PROFILE, AUTO_PROFILE)

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
# it -- Homebrew installs exactly that build -- so ``visible``, ``erase``, ``all`` and
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
    SDXL_ZIMAGE_PROFILE: True,
    CHROMA_ZIMAGE_PROFILE: False,
    # Auto resolves to qwen-zimage or chroma-zimage, both of which keep the
    # input's detail level. If it ever resolves to an engine that needs polish,
    # the resolved profile's own default applies.
    AUTO_PROFILE: False,
}

SDXL_LIGHTNING_MODEL_ID = "ByteDance/SDXL-Lightning"
SDXL_LIGHTNING_PATTERN = "sdxl_lightning_4step_lora.safetensors"

# All profiles are certified at a fixed seed because SynthID removal near the
# strength floor is seed-dependent. The step count and CFG are not settable at all --
# each stage owns them (``GLOBAL_STEPS`` / ``FACE_STEPS`` in qwen_zimage_pipeline).
PROFILE_SEED = 0

# sdxl-zimage runs the qwen-zimage recipe on an SDXL global stage, and strength is
# architecture-bound: at Qwen's 0.154 an SDXL global pass leaves SynthID on a native
# 2816x1536 Gemini original, while 0.20, 0.25 and 0.30 all read clean in the Gemini
# app. 0.25 keeps a rung of margin over that boundary, which the historical SDXL
# certification argues for -- it recorded 0.20 as DETECTED against Gemini on an
# older SDXL pipeline. OpenAI is the easier oracle: the profile already cleared
# openai.com/verify at 0.1102, so 0.15 sits above what was verified rather than on
# it. Unknown follows Gemini, the stricter of the two.
#
# Unlike qwen-zimage this is a flat vendor policy rather than a resolution curve,
# because flat values are what was measured. Every verdict above comes from a fixed
# strength at one size; no size dependence has been established for this stage.
SDXL_ZIMAGE_OPENAI_STRENGTH = 0.15
SDXL_ZIMAGE_GEMINI_STRENGTH = 0.25
SDXL_ZIMAGE_UNKNOWN_STRENGTH = SDXL_ZIMAGE_GEMINI_STRENGTH

# qwen-zimage keeps its resolution curve for unknown content, but measured vendor
# cohorts bypass it. Google remained detectable through 0.24375; 0.25 cleared all
# three valid sources and 0.27 was separately repeated clean across them and three
# accounts, so the independently checked 0.27 candidate is the operating floor.
QWEN_ZIMAGE_GOOGLE_STRENGTH = 0.27

# The two OpenAI sources first cleared at 0.06225 and 0.0695. Add one full observed
# cross-source spread (0.00725) to the worst clean boundary: 0.0695 + 0.00725.
QWEN_ZIMAGE_OPENAI_STRENGTH = 0.07675

# Microsoft's public detector (https://ai.azure.com/nextgen/validate) returned
# Inconclusive rather than an API-level watermark-negative verdict. Three valid
# Paint sources first cleared at 0.04125,
# 0.055, and 0.095. Add one full observed cross-source spread to the worst clean
# boundary: 0.095 + (0.095 - 0.04125) = 0.14875, rounded up to 0.15. This is a
# measured corpus margin, not a universal InvisMark threshold.
QWEN_ZIMAGE_MICROSOFT_STRENGTH = 0.15

# Meta Muse Image stamps every output with Content Seal, but no provenance signal
# survives to route it: the outputs carry no C2PA, and their IPTC
# trainedAlgorithmicMedia companion tag is a standard code many platforms use, so
# it cannot key this cohort the way an issuer keys the others. The floor is
# therefore selected by an explicit --vendor meta override, never by detection.
# Derivation (oracle meta.ai/identification, 2026-08-26/27, corpus in
# data/contentseal/): five independent 2.56 MP generations bracketed at
# lighthouse (0.0525, 0.06], fox (0.03, 0.0375], night_city (0.03, 0.0375],
# mug <= 0.03, text <= 0.015. Worst clean boundary plus one full observed
# cross-source spread: 0.06 + (0.0525 - 0.015) = 0.0975, rounded up to 0.1.
# sdxl-zimage has no measured Meta floor; its vendor map stays without a meta
# entry so an explicit --vendor meta there falls to the unknown 0.25, which is
# above this floor and therefore conservative.
QWEN_ZIMAGE_META_STRENGTH = 0.1

_QWEN_ZIMAGE_FLAT_STRENGTH_BY_VENDOR: dict[str, float] = {
    "google": QWEN_ZIMAGE_GOOGLE_STRENGTH,
    "openai": QWEN_ZIMAGE_OPENAI_STRENGTH,
    "microsoft": QWEN_ZIMAGE_MICROSOFT_STRENGTH,
    "meta": QWEN_ZIMAGE_META_STRENGTH,
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
# guidance 5.0, four effective steps, seed 0). Flat per-vendor values derived
# by the same worst-clean-plus-one-spread rule as the other profiles:
#
# - OpenAI: first-clean boundaries 0.06 / 0.06 / 0.075 on three fixtures ->
#   0.075 + (0.075 - 0.06) = 0.09. Chroma1's OpenAI floor is BELOW qwen's
#   0.07675 while its matched-strength fidelity beats qwen on every metric.
# - Microsoft InvisMark: paint-1 (0.06, 0.08], paint-2 <= 0.04, paint-3
#   (0.06, 0.08] -> 0.08 + (0.08 - 0.04) = 0.12, rounded up to the measured
#   rung 0.125, oracle-verified clean on both worst sources. BELOW qwen's 0.15.
# - Google: 633uuy (0.20, 0.25], akdbei (0.20, 0.25], y48j3c (0.08, 0.12],
#   3mc4t9 (0.08, 0.12] -> 0.25 + (0.25 - 0.12) = 0.38, rounded up to 0.40,
#   oracle-verified clean on the worst fixture. ABOVE qwen's 0.27; at 0.40 the
#   regeneration destroys dense text and collapses face identity, which is why
#   the matched-strength addendum recommends the adaptive-strength follow-up.
# - Meta Content Seal: lighthouse (0.08, 0.10], fox (0.06, 0.08], night_city
#   (0.045, 0.06], studio_mug (0.03, 0.045], text_poster <= 0.03 ->
#   0.10 + (0.10 - 0.03) = 0.17. ABOVE qwen's 0.1; the floors exist because
#   Chroma1's boundaries scatter wider than qwen's, not because the engine is
#   worse per unit strength.
CHROMA_ZIMAGE_OPENAI_STRENGTH = 0.09
CHROMA_ZIMAGE_MICROSOFT_STRENGTH = 0.125
CHROMA_ZIMAGE_GOOGLE_STRENGTH = 0.40
CHROMA_ZIMAGE_META_STRENGTH = 0.17
CHROMA_ZIMAGE_UNKNOWN_STRENGTH = CHROMA_ZIMAGE_GOOGLE_STRENGTH

# Content-adaptive Google floor: the four-fixture calibration showed a clean
# face-count split. Both zero-face fixtures (dense-text cards 633uuy and akdbei)
# need 0.25 first-clean; both face fixtures (y48j3c with 17 detected faces and
# 3mc4t9 with 7) clear at 0.12. The flat 0.40 floor is the zero-face policy
# (worst 0.25 + spread 0.13 = 0.38, rounded to 0.40). Face content can use the
# measured 0.125 rung instead: both face fixtures first-cleaned at 0.12, and
# with identical boundaries the cross-source spread is zero, so 0.125 (the next
# measured rung above 0.12) is the operating point. This split is
# Google-SynthID-specific: OpenAI's face fixture was HARDER than its text
# fixture (0.075 vs 0.06), so no other cohort gets an adaptive arm.
# Oracle-verified 2026-08-30; see docs/chroma1-engine-research.md.
CHROMA_ZIMAGE_GOOGLE_FACE_STRENGTH = 0.125

_CHROMA_ZIMAGE_STRENGTH_BY_VENDOR: dict[str, float] = {
    "openai": CHROMA_ZIMAGE_OPENAI_STRENGTH,
    "google": CHROMA_ZIMAGE_GOOGLE_STRENGTH,
    "microsoft": CHROMA_ZIMAGE_MICROSOFT_STRENGTH,
    "meta": CHROMA_ZIMAGE_META_STRENGTH,
}
_ALIASES = {
    "qwen_zimage": QWEN_ZIMAGE_PROFILE,
    "sdxl_zimage": SDXL_ZIMAGE_PROFILE,
    "chroma_zimage": CHROMA_ZIMAGE_PROFILE,
    "auto": AUTO_PROFILE,
}

# The deterministic per-cohort selection policy: which profile wins on which vendor, from the
# 2026-08-29/30 four-cohort calibration (docs/chroma1-engine-research.md).
# chroma-zimage has lower floors AND better matched-strength fidelity on OpenAI
# and Microsoft; qwen-zimage wins on Google and Meta. Unknown stays on qwen
# (the shipped default, conservative).
_ENGINE_BY_VENDOR: dict[str, str] = {
    "openai": CHROMA_ZIMAGE_PROFILE,
    "microsoft": CHROMA_ZIMAGE_PROFILE,
}


def resolve_auto_profile(vendor: str | None) -> str:
    """Pick the engine for --pipeline auto from the provenance vendor."""
    return _ENGINE_BY_VENDOR.get((vendor or "").casefold(), QWEN_ZIMAGE_PROFILE)


def normalize_profile(profile: str) -> str:
    """Normalize spelling and resolve the underscore spellings."""
    value = profile.strip().casefold()
    return _ALIASES.get(value, value)


def resolve_seed(seed: int | None) -> int:
    """Keep every profile reproducible by default."""
    return PROFILE_SEED if seed is None else seed


def resolve_adaptive_polish(adaptive_polish: bool | None, pipeline: str) -> bool:
    """Return an explicit polish choice, or the profile's calibrated default."""
    if adaptive_polish is not None:
        return adaptive_polish
    return PROFILE_ADAPTIVE_POLISH.get(normalize_profile(pipeline), True)


def strength_default_help() -> str:
    """Describe the live default policy without duplicating its values."""
    return (
        "profile-adaptive (qwen-zimage uses resolution-adaptive denoise, with a "
        f"flat OpenAI {QWEN_ZIMAGE_OPENAI_STRENGTH} / Google {QWEN_ZIMAGE_GOOGLE_STRENGTH} / "
        f"Microsoft InvisMark {QWEN_ZIMAGE_MICROSOFT_STRENGTH} / Meta Content Seal "
        f"{QWEN_ZIMAGE_META_STRENGTH} floors; sdxl-zimage "
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
    face_count: int | None = None,
) -> float:
    """Resolve a user override or the calibrated policy for a profile and vendor.

    Total by design. qwen-zimage picks its strength from image area rather than
    from the vendor, so it needs ``size``; returning ``None`` for it instead would push that
    branch onto every caller and move one of the two strength policies outside this
    module. ``size`` is required for qwen-zimage without an explicit strength.
    Measured vendor exceptions bypass the area curve: OpenAI and Microsoft use one
    additional observed cross-source spread over their worst clean boundaries, while
    Google uses a separately repeated cross-source candidate. See the constants'
    comments for the exact derivations.
    """
    if strength is not None:
        return strength
    normalized = normalize_profile(pipeline or "")
    if normalized == SDXL_ZIMAGE_PROFILE:
        return _SDXL_ZIMAGE_STRENGTH_BY_VENDOR.get((vendor or "").casefold(), SDXL_ZIMAGE_UNKNOWN_STRENGTH)
    if normalized == CHROMA_ZIMAGE_PROFILE:
        vendor_key = (vendor or "").casefold()
        if vendor_key == "google" and face_count is not None and face_count > 0:
            return CHROMA_ZIMAGE_GOOGLE_FACE_STRENGTH
        return _CHROMA_ZIMAGE_STRENGTH_BY_VENDOR.get(vendor_key, CHROMA_ZIMAGE_UNKNOWN_STRENGTH)
    if size is None:
        raise ValueError("qwen-zimage resolves strength from image area, so size is required")
    vendor_strength = _QWEN_ZIMAGE_FLAT_STRENGTH_BY_VENDOR.get((vendor or "").casefold())
    if vendor_strength is not None:
        return vendor_strength
    from remove_ai_watermarks._internal.two_stage_pipeline import resolution_adaptive_denoise

    return resolution_adaptive_denoise(*size)


def vendor_for_strength(image_path: Path) -> Literal["openai", "google", "microsoft", "meta"] | None:
    """Select the strength cohort from non-invalid pixel-watermark provenance.

    OpenAI / Google / Microsoft come from their C2PA issuers. Meta is the
    fallback cohort: Muse Image carries no C2PA at all, and its only readable
    companion is the IPTC ``trainedAlgorithmicMedia`` XMP tag -- a standard code
    other platforms also use. Attributing that tag to Meta is a measured bet,
    not an identification: the other tag users in this project's model (ByteDance
    products, X) ship no invisible pixel watermark this profile targets, so the
    worst misroute spends the Meta floor (0.1) where the resolution curve would
    have spent a similar amount, and Google/OpenAI files never reach this arm
    because their C2PA matched first."""
    try:
        from remove_ai_watermarks._internal.c2pa import (
            c2pa_info_has_invalid_credential,
            c2pa_info_has_invismark,
            extract_c2pa_info,
        )
        from remove_ai_watermarks.metadata import synthid_source

        info = extract_c2pa_info(image_path)
        evidence = (synthid_source(image_path, c2pa_info=info) or "").casefold()
    except Exception:
        return None
    if "google" in evidence:
        return "google"
    if "openai" in evidence:
        return "openai"
    if not c2pa_info_has_invalid_credential(info) and c2pa_info_has_invismark(info):
        return "microsoft"
    if _standalone_iptc_ai_tag(image_path):
        return "meta"
    return None


def _standalone_iptc_ai_tag(image_path: Path) -> bool:
    """True when the file carries an AI IPTC marker with no C2PA around it.

    Mirrors identify's ``standalone_iptc`` condition (the tag is only
    trustworthy as platform evidence when no manifest supersedes it) without
    importing the heavy identify module: the shared chunk-aware
    :func:`metadata.scan_head` window -- Muse WebP outputs place their XMP
    packet in a tail chunk up to hundreds of KB past a plain head read, which
    is exactly what scan_head's extensions exist to catch.
    """
    try:
        from remove_ai_watermarks.metadata import IPTC_AI_MARKERS, c2pa_marker_in, scan_head

        scan = scan_head(image_path)
    except Exception:
        return False
    return any(marker in scan for marker in IPTC_AI_MARKERS) and not c2pa_marker_in(scan)
