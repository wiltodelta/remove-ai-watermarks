# Text protection research: crisp text under a "watermark removed everywhere" constraint

Date: 2026-05-29. Source: a deep-research run (104 agents, 5 search angles, sources
fetched and 3-vote adversarially verified). Not committed automatically — saved as a
research note for the next session.

## The constraint that frames everything

The invisible watermark (Google SynthID) must be removed **everywhere, including inside
text regions**. Therefore any technique that keeps or composites the **original
(watermarked) text pixels** is disqualified — the text must be *regenerated / freshly
synthesized* enough to scrub the watermark, yet rendered crisply. This single rule is the
filter applied to every candidate below.

## Problem recap

The `invisible` pipeline is SDXL base 1.0 img2img at low strength (~0.05) to defeat
SynthID with minimal visible change. Text is protected via Differential Diffusion with a
per-pixel change map (`preserve` ~0.9) driven by the PP-OCRv3 DB detector
(`text_protector.py`). Large text survives; **small text (sub ~8 px strokes) softens or
garbles** (issue #14, confirmed on real content).

## Executive summary

The fine-text softening is an **architectural consequence of latent-space processing, not
a tuning problem**: SDXL's 4-channel VAE (~48x compression) discards high-frequency signal
on encode, and Differential Diffusion blends in latent space with the change map
downsampled by 8x, so any stroke under ~8 px sits inside one latent cell and cannot be
preserved or edited cleanly **regardless of `preserve`** (the Differential Diffusion
authors state this limit explicitly). Two structurally sound directions keep the
"watermark removed everywhere" guarantee because they **synthesize fresh glyph pixels**
rather than compositing originals: (1) glyph/text-conditioned diffusion re-render of
detected text (AnyText2, EasyText), and (2) a two-stage architecture — global scrub, then
a dedicated text-restoration / text-aware super-resolution pass over detected regions
(TIGER, TextSR, TeReDiff/TAIR). **EasyText** and **TextSR** are the most promising for this
CJK-first pipeline (both multilingual via DiT/ByT5, both regenerate from glyph or
character-shape priors). The deepest fix — a 16-channel (SD3/FLUX) VAE — materially reduces
the softening but means switching the base model, not a drop-in VAE swap.

## Constraint reconciliation (important)

The generic research "quick win: bump `preserve` toward 1.0" is **invalid under our hard
constraint**: raising `preserve` freezes the text region, so SynthID there is **not
scrubbed**. Likewise, pixel paste-back of the original text is disqualified. The only
constraint-compatible quick win is **higher resolution / tiled diffusion** (strokes span
more latent cells, less VAE softening, while the text is still fully regenerated and thus
scrubbed). The real answer is **regenerate text crisply**, not freeze it.

## Findings (with confidence and sources)

### Finding 1 — confidence: high

**Claim.** The small-text softening is an architectural latent-space limit, not a tuning issue. SDXL's VAE compressively encodes (losing exact color and fine detail on every round-trip), and Differential Diffusion blends in latent space with the change map downsampled to latent resolution (8x), so the method explicitly caps edit/preserve granularity at ~8 px under SD settings. Text strokes below one latent cell cannot be cleanly preserved even at preserve ~0.9.

**Evidence.** Differential Diffusion's paper states a "cap on the resolution of the change map ... can limit the ability to precisely edit small objects (less than 8 pixels for Stable-Diffusion's settings)"; the official SDXL pipeline downsamples the map by `vae_scale_factor=8` and blends `latents = original*mask + latents*(1-mask)` in latent space. The VAE encode is "compressive ... exact color qualities and exact visual fine-details are lost." arXiv:2512.05198 confirms "resizing the pixel mask to latent resolution discards fine structure ... downsamples by 1/8" and that linear latent blending "cannot be pixel-equivalent." Higher compression = more high-frequency loss (arXiv:2305.02541).

**Sources.** https://onlinelibrary.wiley.com/doi/10.1111/cgf.70040 · https://differential-diffusion.github.io/ · https://github.com/exx8/differential-diffusion · https://arxiv.org/abs/2512.05198 · https://omriavrahami.com/blended-latent-diffusion-page/ · https://arxiv.org/pdf/2305.02541

### Finding 2 — confidence: low (do not build on it yet)

**Claim.** Pixel-space differential / blended-latent variants exist as a research direction, but the specific full-resolution-mask solution (PELC/DecFormer, arXiv:2512.05198) was NOT verified to deliver its claimed seam/edge improvements.

**Evidence.** arXiv:2512.05198 argues linear latent blending is not pixel-equivalent and proposes decoder-equivariant compositing; PixPerfect (arXiv:2512.03247) does pixel-space refinement of chromatic shifts at edit boundaries. But the specific PELC full-resolution-mask and DecFormer "53% error reduction" claims were **refuted on adversarial vote (0-3 and 1-2)**. Treat pixel-equivalent latent compositing as an emerging idea to watch, not a production fix.

**Sources.** https://arxiv.org/abs/2512.05198 · https://arxiv.org/abs/2512.03247

### Finding 3 — confidence: high

**Claim.** Glyph/text-conditioned diffusion can re-render detected text as freshly synthesized pixels (not copied), which inherently scrubs any watermark in the text region while rendering glyphs crisply. AnyText/AnyText2 inject text-rendering into a pretrained T2I model and support generation AND editing of existing scene images; multilingual including CJK and English.

**Evidence.** AnyText2 "enables precise control over multilingual text attributes in natural scene image generation and editing" (WriteNet+AttnX); +3.3% (Chinese) / +9.3% (English) accuracy over AnyText v1. AnyText "can be plugged into existing diffusion models ... for rendering or editing text" and synthesizes text latent features through diffusion (fresh pixels), supporting zh/en/ja/ko/ar/bn/hi. **Caveat:** both are SD1.5-based, so NOT a drop-in into the SDXL scrub (separate base model); AnyText's own limitation: "the inpainting manner ... impedes editing quality on small text," and it ranks weak on STRICT (EMNLP 2025) — small-text crispness not guaranteed.

**Sources.** https://github.com/tyxsspa/AnyText2 · https://arxiv.org/abs/2411.15245 · https://arxiv.org/abs/2311.03054

### Finding 4 — confidence: high

**Claim.** EasyText is a strong glyph-conditioned re-render candidate: built on the FLUX-dev DiT framework with LoRA tuning, renders compact per-character glyph patches (64px-high adaptive for alphabetic, 64x64 for logographic) concatenated in latent space, supports 10+ languages including Chinese, Japanese, Korean, Thai, Vietnamese, Greek, and Latin.

**Evidence.** AAAI 2025 + arXiv:2505.24417: "implemented based on the open-source FLUX-dev framework with LoRA-based parameter-efficient tuning," VAE and text encoder frozen, two-stage 512->1024 training. Glyph conditioning via "64-pixel-high images ... adaptive widths for alphabetic; fixed 64x64 for logographic," VAE-encoded and concatenated with denoised latents, "less than one-tenth the spatial size of layout-matching methods." FLUX-based (16-channel VAE, DiT) also sidesteps the SDXL 4-channel wall. Fresh-pixel generation preserves the watermark-removal guarantee. Cyrillic/Arabic crispness not separately benchmarked.

**Sources.** https://arxiv.org/html/2505.24417 · https://ojs.aaai.org/index.php/AAAI/article/view/37697

### Finding 5 — confidence: high

**Claim.** A two-stage "global watermark scrub then text-restoration pass" architecture is validated by recent literature, and the restoration stage can synthesize glyph pixels from priors (no original-pixel reintroduction). TIGER reconstructs stroke geometry then injects it as guidance into full-image super-resolution; TextSR uses a detector + multilingual OCR to regenerate text from character-shape priors; TeReDiff/TAIR couples a jointly-trained text-spotter with diffusion.

**Evidence.** TIGER (arXiv:2510.21590): "a diffusion-based local text refiner ... reconstructing fine-grained stroke geometry ... injected as conditional guidance into the subsequent full-image restoration." TextSR (arXiv:2505.23119, Google): "leverages a text detector ... then employs OCR to extract multilingual text," regenerating from "multilingual character-to-shape diffusion priors" that "produce character shapes solely based on text prompts, even without visual input" — fresh pixels. TAIR/TeReDiff (ICLR 2026): standard restoration "frequently generates plausible but incorrect textures"; TeReDiff feeds text-spotter outputs back as prompts. **Caveat:** TIGER orders text-first then global (reverse of scrub-then-text); these target degraded-input super-resolution, not watermark removal, so the SynthID-scrub of the restoration stage must be verified empirically (the stages are themselves diffusion-based, so fresh-pixel = no SynthID is plausible but unproven here).

**Sources.** https://arxiv.org/html/2510.21590v1 · https://arxiv.org/html/2505.23119v1 · https://cvlab-kaist.github.io/TAIR/ · https://arxiv.org/abs/2506.09993

### Finding 6 — confidence: high

**Claim.** Switching to a 16-channel VAE (SD3/FLUX class) materially reduces small-text/latent softening vs SDXL's 4-channel VAE, but it requires switching the base model — not a drop-in latent swap into an SDXL UNet img2img pipeline. RAE approaches are DiT-native and likewise not drop-in.

**Evidence.** SD3/FLUX moved from 4-channel (48x) to 16-channel (12x) VAEs specifically to preserve fine detail (diffusers Discussion #8713; madebyollin VAE notes; arXiv:2305.02541). RAE (arXiv:2510.11690) "should be the new default for diffusion transformer training" but produces high-dimensional latents needing a DiT wide-DDT head — NOT compatible with an SDXL 4-channel UNet. EasyText shows the practical path: adopt a FLUX-DiT base rather than retrofit SDXL. The VAE upgrade couples to a base-model migration.

**Sources.** https://arxiv.org/abs/2510.11690 · https://arxiv.org/pdf/2305.02541 · https://arxiv.org/html/2505.24417

## Recommendation

Under the hard constraint, the correct architecture is **not "protect text during the
scrub" (Differential Diffusion)** but **"scrub everywhere, then restore text crisply by
regeneration"**:

1. Global SDXL scrub with text protection OFF (text region is scrubbed too).
2. On detected text regions, a **glyph-conditioned restoration** that re-renders the same
   glyphs as fresh pixels (no original reused).

This is the only path that delivers both "watermark everywhere" and crisp text.

**Top-2 to prototype:**
- **TextSR** — detector + multilingual OCR + character-shape diffusion priors; closest to
  the existing detector-driven pipeline.
- **EasyText** — FLUX-DiT glyph re-render, multilingual incl. CJK; also gets the 16-channel
  VAE for free.

**Honest costs / unknowns:** this is a re-architecture, not a quick fix. It needs a new
**OCR-recognition** step (we currently only detect text; we must know *what* to re-render).
Models are FLUX/DiT-class (heavy) -> serverless GPU. Maturity is research-grade; CJK is
covered, Cyrillic/Arabic crispness is not separately benchmarked -> a prototype must
measure real fidelity. The restoration stage being diffusion-based makes "fresh pixels =
no SynthID" plausible but **must be verified empirically** (run the SynthID oracle on the
restored output).

**Constraint-compatible quick win to try first:** run the global scrub at **higher
resolution / tiled** so strokes exceed the latent cell — less softening, full scrub, no
freezing. Cheap to test; quantify recall/quality vs cost.

**Do not pursue:** raising `preserve` toward 1.0 or pixel paste-back (both leave original
watermarked pixels in text); PELC/DecFormer pixel-equivalent latent compositing (refuted,
not production-ready).

## Provenance

Deep-research workflow run `wf_118b9a03-3eb` (2026-05-29). Findings adversarially verified
(2/3 refutes required to kill a claim). This note records research only; no code change is
implied until a prototype validates fidelity and the SynthID-scrub guarantee on the
restored output.
