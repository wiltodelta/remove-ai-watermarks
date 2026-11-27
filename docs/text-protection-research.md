# Text protection research: crisp text under a "watermark removed everywhere" constraint

> Research archive. This page records evaluated ideas, including rejected
> prototypes. Current behavior is documented in the user guides and source.
> The `text_protector.py` module it describes was removed from the library
> afterwards.

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

The `invisible` pipeline is SDXL base 1.0 img2img to defeat SynthID. The default
strength rose over time as Google hardened SynthID (0.05 -> 0.10 -> **~0.30** in
this experiment; the later measured operating point is 0.27); higher strength deforms text more, which is
exactly why text protection matters. Text is protected via Differential Diffusion with a
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

## 2026-08-13 OCR plus LaMa prototype

A lightweight version of the recommended two-stage architecture was tested on
the three tracked text fixtures. It used the already-clean Qwen result as the
base, recognized the source text, removed source and Qwen glyph contours with
LaMa, and rendered the recognized strings as fresh pixels. No source pixels
were pasted back.

The result validates only part of the hypothesis. Character-weighted CER on the
two multilingual posters fell from 0.338/0.305 to 0.007/0.007, and OpenAI Verify
remained clean for both matched Qwen and restored pairs. However, replacement
fonts changed the design, whole-image LPIPS worsened by 0.067 on each poster,
and the light poster retained a shadow in one small English line. On the Chinese
sign, punctuation errors moved CER from 0.074 to 0.111. The Google verifier run
was inconclusive because the available account detected both the restored image
and the byte-identical Qwen control that a previous account had called clean.

The experiment supports a proper glyph-conditioned restorer, not shipping the
plain OCR/system-font compositor. Exact hashes, metrics, masks, and provider
verdicts are tracked in
[`data/evaluations/fidelity/text-restoration-2026-08-13.csv`](../data/evaluations/fidelity/text-restoration-2026-08-13.csv).

## 2026-08-13 AnyText2 follow-up

The official AnyText2 SD1.5 checkpoint was then tested as the glyph-conditioned
restorer. Its ModelScope entry and code are Apache 2.0, and the downloaded
checkpoint matched the published SHA-256. An official blackboard edit completed
successfully before the fixture run, establishing that the runtime reproduced
the model rather than silently exercising a fallback.

On the Chinese sign, a three-line local edit over the exact Qwen output scored
CER 0.185 under the standard detector. Font mimic from the source glyph masks
scored 0.222. Whole-image LPIPS changed from 0.289 to 0.338 and 0.345 respectively.
A padded crop-recognition check showed why the comparison matters: Qwen contained
the full correct text and scored 0.000, while default AnyText2 still scored 0.074
with two substituted characters and mimic remained at 0.222. The provider oracle
was deliberately not called because these variants had already failed the
content-fidelity gate.

AnyText2 is therefore not the missing production restorer in its published form.
The negative result is narrower than the model family: only the SD1.5 checkpoint
is public, while AnyText2XL remains listed as future work. The current wrapper
also truncates each quoted text line to 20 characters, which prevents a direct
test of several tracked English and Russian poster lines. Exact model provenance,
output hashes, and metrics are recorded in
[`data/evaluations/fidelity/anytext2-restoration-2026-08-13.csv`](../data/evaluations/fidelity/anytext2-restoration-2026-08-13.csv).

The same cross-check exposes a measurement bug in the earlier lightweight
restoration result. Paddle detection made tight boxes that omitted the final
Chinese full stop from two lines, producing Qwen CER 0.074 although the pixels
and padded recognition were correct. Adding 16-32 pixels of horizontal context
restored every punctuation mark. A deterministic rerender using the full OCR
strings and the closest of nine mask-scored CJK fonts also reached detector CER
0.000, but its heavier Hiragino Sans GB W6 glyphs raised LPIPS from 0.289 to 0.327.
The actionable design is selective restoration: compare padded source and output
recognition first, and preserve the Qwen output when they already match.

That policy was tested on the two multilingual posters. A manual prototype kept
the lines whose padded source and Qwen recognition agreed and rerendered only the
changed lines. The edited mask fell from 15.4%/17.0% in the full compositor to
5.5%/5.5%. On a single Paddle `en+ru+ch` route, CER changed from 0.378/0.413 for
Qwen to 0.101/0.112, while LPIPS was only 0.120/0.103 rather than the full
compositor's 0.174/0.162. The different OCR route is intentional and all three
variants were remeasured through it; these CER values are not directly
comparable with the earlier Vision/Paddle hybrid results.

Both selective outputs returned `No OpenAI signals detected` in OpenAI Verify,
and the original light poster returned `Generated with OpenAI tools` immediately
afterward as the positive control. The dark poster was visually clean. The light
poster still showed one local old-text shadow, so mask construction remains the
blocking defect. Exact hashes, metrics, mask fractions, and oracle controls are
in
[`data/evaluations/fidelity/selective-text-restoration-2026-08-13.csv`](../data/evaluations/fidelity/selective-text-restoration-2026-08-13.csv).

Uniformly expanding every selected glyph mask by two pixels removed that light
poster shadow and moved CER from 0.101 to 0.081 without a meaningful LPIPS cost.
It also expanded the edited area from 5.5% to 7.0%, and the identical rule made
the dark poster worse at CER 0.147 rather than 0.112. Those follow-up bytes have
not been oracle-checked. The next prototype should detect residual source glyphs
per line and expand only the failing component, rather than introducing another
poster-wide tuning constant.

The manual selection was then replaced with per-line padded recognition. A line
is left untouched only when the source recognizer is sufficiently consistent
with the verified line and normalized source/candidate recognition agree
exactly. The automatic rule reproduced the manual selection on the dark poster
and correctly kept one additional large Russian line on the light poster. It
reduced that poster's effective edited area from 5.5% to 4.1%.

Mask cleanup became a measured second pass rather than a global dilation. After
the first LaMa erase, the prototype finds contrast that remains specifically at
the original glyph positions, dilates only that residual, and erases it before
fresh text is drawn. This removed the visible double glyphs from both posters.
The automatic outputs scored LPIPS 0.113/0.104 and detector CER 0.123/0.119, with
effective edited fractions of 4.1%/5.7%. The light aggregate CER is pessimistic:
the page detector fragments its untouched Cyrillic line into Latin-like pieces,
while the padded Russian recognizer returns the exact expected text.

OpenAI Verify returned `No OpenAI signals detected` for both new hashes and then
`Generated with OpenAI tools` for the original light poster in the same browser
sequence. Reproducing the prototype from the tracked script exposed harmless
ONNX rounding on the light image: 386 pixels differed by at most one channel
value from the initially checked hash. The exact tracked output hash
`900def5a...` was therefore submitted separately, returned clean, and was
followed immediately by another positive source result. The dark tracked output
was byte-identical to the initially checked hash.

The Chinese sign provides a no-edit control for the selector. Detector boxes for
its three large lines overlap vertically, so Latin-style vertical padding
incorrectly mixed neighboring lines. Horizontal-only CJK padding captured the
terminal punctuation without mixing neighbors and made source and candidate
recognition agree exactly on all three lines.
The script then produced a zero mask and copied the Qwen candidate byte for byte.
This prevents the heavier-font regression seen in the earlier CJK compositor,
but does not resolve the unavailable Google-negative oracle control.

The result clears the measured OpenAI gate without a manual line selection, but
is not a production design yet. It still assumes verified source strings and
line boxes, relies on macOS system fonts, and needs evaluation on uncurated
layouts, rotated text, false OCR matches, automatic line-box discovery, and a
reproducible Google-negative control.

The later source-silhouette experiment removed the font lookup but did not meet
the actual visual requirement. Native-size review found changed stroke weight,
color variation, edge antialiasing, and decorative details even where crop OCR
was exact. A narrower `vae-glyphs` experiment uses the silhouette only as a
support mask: exact pixels come from a Qwen-VAE reconstruction, while a fresh
flat-color silhouette supplies only the outer edge beneath that core. Across 48
typography cases (548 annotated lines), this improved glyph-edge MAE in 48/48
and text-box SSIM in 47/48. The median values moved from 37.59 to 32.42 and from
0.854 to 0.914 respectively. One separate untracked core-only output with a
1.43% donor area returned `No OpenAI signals detected`; the tracked script then
reproduced those exact bytes. A 0.5-pixel feathered sibling with 2.75% nonzero
donor area improved mean text-box SSIM from 0.957 to 0.960, with a 0.918 minimum
across 15 verified lines, while full-image RGB SSIM reached 0.864. Crop OCR
recovered from 7/15 exact lines on the raw pass to 14/15, equal to the source's
own recognizer floor. Whole-image LPIPS was 0.082, but only 0.108% of pixels
remained exactly equal to the source and the detected face's Laplacian-variance
ratio was 0.670, confirming residual global smoothing. A same-session
OpenAI Verify sequence detected the exact source control in 1/1 check, returned
clean for the raw global pass and clean-fidelity anchor in 1/1 each, and returned
clean for the feathered output in 3/3 repeats. The tracked script reproduced
that feathered file byte for byte. This proves two materially better exact-output
Pareto points, not a general operating range. The 48 matrix outputs and other
mask sizes remain oracle-unverified.

The first Google oracle sample did not transfer. For the synthetic CJK sign,
two separate work accounts detected SynthID in both the resaved positive source
control and the exact Qwen-VAE donor candidate. The candidate's text-box SSIM
improved from 0.791 to 0.835 and its glyph-edge MAE from 35.48 to 22.49, but its
3.67% donor layer did not create a clean operating point. The Qwen silhouette
base was already detected, as was the earlier original-sign Qwen baseline in the
same account. The measured cause is therefore insufficient global Google
regeneration, not a demonstrated watermark regression from the text donor.
Google needs a stronger oracle-clean global anchor before the narrow donor can
be evaluated there.

A targeted follow-up supplied that missing anchor for one CJK case. An explicit
Qwen global pass at strength 0.30 returned no reliable SynthID signal in Gemini,
while the matched source control was detected. Applying the 0.5-pixel
`vae-glyphs` layer to that clean anchor changed 1.17% of the image and raised
mean text-box SSIM from 0.765 to 0.964 (minimum 0.963), with mean text-box MAE
falling from 20.74 to 5.38. The post-donor output then returned clean in 3/3
Gemini checks across two work accounts. A separate 18-face fixture with four
tiny UI-text lines also had a clean strength-0.30 anchor; its donor layer changed
0.70% of the image, raised mean text-box SSIM from 0.617 to 0.976 (minimum
0.972), and returned clean in 3/3 checks across the same two accounts. One
quota-exceeded response is excluded from both pass rates. These are two exact
oracle-certified outputs, not a general operating range: other layouts, masks,
strengths, seeds, and donor areas remain unverified.

An automatic-box follow-up merged Paddle word detections by vertical overlap.
It found exactly 20/20 poster lines and 3/3 sign lines, with mean IoU 0.857,
0.847, and 1.000 against the verified boxes. That structural match was not
sufficient: reusing the annotation crop padding changed recognition decisions,
reduced preserved dark-poster lines from 7 to 4, and expanded its edited
fraction from 5.7% to 11.2%.

A recognition-only sweep isolated the crop sensitivity. Limiting vertical
padding around detector boxes to 8-12% of line height reproduced the verified-
box decision vectors on both posters; 10% was used for a full follow-up. It kept
8/20 and 7/20 lines with edited fractions of 4.1% and 5.7%, and visual inspection
found no double glyphs. Whole-image LPIPS remained 0.113/0.104, but detector CER
was 0.127/0.154 instead of 0.123/0.119. The dark regression failed the fidelity
gate, so these hashes were not submitted to the provider oracle. The opt-in flag
remains only to reproduce the negative result. Count, IoU, and matching decision
vectors are therefore insufficient gates for automatic boxes; the next design
needs box rectification or recognition stability under crop jitter before it
can replace verified geometry.

Removing verified strings was tested separately with an annotation-seed dry
run. It detects boxes, chooses `en`, `ru`, or `ch` from Unicode script, and
accepts a draft only when three crop paddings normalize identically and every
confidence is at least 0.85. One execution proposed 20 and 18 poster lines, but
exact-text precision against the held-out annotations was only 90.0% and 94.4%.
The stable errors were punctuation: one lost English comma and an ideographic
comma consistently replaced by ASCII. Thus crop agreement and confidence do
not verify literal text. The helper is retained only to seed annotations; every
proposal needs manual review before it can enter the restoration compositor.

## Production-path anchor leak and fix (2026-08-19)

The engine `text_manifest=` path (unlike the research compositor above) blends a
15% Qwen-VAE fidelity anchor across the WHOLE frame before glyph restoration.
Certified through the official OpenAI Content Provenance API on the exact
production path (qwen-zimage, seed 0, no strength override, untiled, LANCZOS
resize-back, metadata strip; library 0.27.0 @ 50d9f448):

- controls (pixel-identical metadata-stripped sources): `synthid: detected` -
  the pixel channel was proven live before every session;
- base outputs: `clean` x6 (two 20-line posters, 3 attempts each);
- text-restored WITH the anchor: `synthid: detected` x6, deterministic;
- text-restored with `FIDELITY_BLEND_ALPHA = 0` (runtime ablation, base bytes
  identical between runs): `clean` x6.

The anchor was therefore the leak channel: a 15% global donor admixture over a
scrubbed frame returns enough OpenAI SynthID for the provider detector. Fidelity
cost of dropping it (source-referenced text-box MAE): 11.60->11.72, 7.79->7.86,
and 7.57->8.13 across the two posters and the CJK sign - well under one point
against base MAEs of 24.7/16.8/10.2. Since 0.27.1 the anchor was turned off by
default, and in 0.76.0 `fidelity_anchor` / `--fidelity-anchor` was removed from
the CLI and public options; Google rows were not re-queried because the default
strength leaves that fixture's base detected (the vendor-strength lane tracks
that separately). Certification artifacts: private archive
`text-restoration-2026-08-19-A1` (sha256-stamped rows).
