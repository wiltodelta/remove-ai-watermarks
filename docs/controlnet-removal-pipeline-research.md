# ControlNet-as-removal-pipeline research: can structure-conditioned regeneration scrub SynthID and keep text?

> Research archive. This document records experiments, superseded defaults, and
> deployment considerations from the time of the study. It does not define the
> current CLI or Python API. See `README.md`, `docs/cli.md`, and
> `docs/known-limitations.md` for current behavior. Everything it names on the
> face-restoration path was removed after this study: the `--restore-faces` and
> `--face-id` flags, the `instantid`, `photomaker` and `faceid` extras, and their
> modules. So were the `controlnet`, `sdxl`, `qwen` and `default` profiles.

Date: 2026-06-02. Source: a manual primary-source pass (WebSearch + WebFetch over the
watermark-removal-attack and SDXL-ControlNet literature). Prompted by issue #35
(@newideas99 / Jacob): "as we use SDXL even at low strength that kills small text ... Do you
think ControlNet could be added to preserve and still remove the watermark?" Clarified scope:
Jacob means **replacing the removal pipeline itself** with a ControlNet-conditioned
regeneration (structure held by the control signal), NOT a separate text-protection add-on.

A deep-research workflow run was attempted first (`wf_3244411d-ffd`) and failed at the harness
level (97 agents completed without emitting StructuredOutput; ~4.3 M tokens, no report). This
note is the hand-run replacement.

## The question, precisely

Can a single full-image ControlNet-conditioned diffusion pass **replace** plain SDXL base 1.0
img2img as the watermark remover, so that one structure-guided regeneration removes the
invisible robust pixel watermark (Google SynthID) **everywhere** while keeping fine detail and
small/CJK **text** legible across the whole image? The hard constraint is unchanged from
`text-protection-research.md`: the watermark must be scrubbed everywhere including inside text,
so any path that freezes or composites original text pixels is disqualified.

## Executive summary

The idea is **already academically validated as a watermark remover and is literally what we
already ship** — CtrlRegen (ICLR 2025) is a canny-ControlNet + DINOv2-semantic pipeline that
regenerates from clean noise. But the make-or-break gap is exact: **none of the
watermark-removal papers validate TEXT or fine-detail preservation at all** — CtrlRegen reports
only FID/PSNR/quality-model scores and explicitly contains no text, fine-detail, or
hallucination analysis. Our shipped ctrlregen's empirical failure ("destroys real content,
hallucinates micro-text in smooth regions") is precisely this unstudied failure mode of the
published method, most likely driven by our **512 px tiling** (text occupies too few pixels per
tile to regenerate legibly; edge-free smooth regions get DINOv2-semantic hallucination). The
constructive path is NOT to keep fixing the SD1.5 CtrlRegen, but to port the structure-control
idea onto an **SDXL-native** ControlNet (xinsir tile-sdxl / ControlNet-Union-SDXL) as a control
add to our existing SDXL base 1.0 img2img, run it at **1024+, not 512 tiles**, and empirically
sweep the (denoise strength x conditioning scale x resolution) cube against the SynthID oracle
AND a text-legibility check. The central tension may be fundamental and must be measured: the
conditioning strong enough to keep text legible may suppress regeneration enough to let SynthID
survive; the regeneration strong enough to scrub may deform text regardless of edges.

## Oracle validation 2026-06-04 — measured answer (AUTHORITATIVE; supersedes pre-oracle "scrubs SynthID" claims below)

The central tension the summary predicted is now MEASURED against the live oracles
(OpenAI `openai.com/verify` for OpenAI content; the Gemini app "Verify with SynthID"
for Google content — each detects only its own vendor's payload). **Verdict: at the
shipped low vendor-adaptive strength, controlnet is NOT a drop-in SynthID remover.**
It preserves structure so well that the watermark survives on exactly the photoreal
content it protects. Controlnet is the text/structure PRESERVATION pipeline; removal
is set by STRENGTH, separately calibrated, not by the pipeline choice.

This section is the single consolidated reference for the controlnet pipeline's
removal behavior. (Mirrored briefly in `docs/synthid.md` §5.5, which points here.)

### What we measured (real gpt-image + Gemini originals)

**1. Content × pipeline — neither pipeline clears all content at low strength.**
OpenAI set, strength = vendor-adaptive (0.10 OpenAI / 0.15 no-C2PA), `--max-resolution 1536`:

| content | controlnet (`--auto`) | plain `default` |
|---|---|---|
| flat text card | clean | clean |
| flat graphic (logo/poster, large flat fills) | clean | **SynthID detected** |
| photoreal (9-face grid) | **SynthID detected** | clean |
| photoreal (bracelet product photo) | **SynthID detected** | clean |

Mechanism: controlnet's dense edge map keeps the regeneration very close to the original
on photoreal, so the SynthID-destroying perturbation never happens; but it freely
repaints large flat fills. Plain img2img at low strength perturbs photoreal texture
enough yet barely touches flat fills. So the survivors FLIP by content type — pipeline
choice alone does not guarantee removal.

**2. Seed non-determinism near threshold.** img2img uses a random seed unless `--seed`
is passed, and these geometries are outside the current local detector's scope.
The bracelet survived controlnet @0.15 in one run and CLEARED @0.15 in another
(same pipeline+strength+res).
So a single clean run does NOT establish a strength as safe — characterizing a reliable
floor needs a seed-repeatability sweep (N runs, varied seed), not one pass.

**3. Per-vendor controlnet strength ladder (photoreal, oracle):**
- **OpenAI:** 0.10 detected → 0.15 borderline/non-deterministic → 0.20 cleared (with margin).
- **Gemini** (harder vendor; default 0.15 vs OpenAI 0.10): most cleared at 0.15–0.25.
- **Resolution is NOT the lever:** SynthID is robust to downscaling, and the study's
  trend says LOWER processing res needs LESS strength, so 1024 was never the wall. A
  Gemini face that resisted 0.15/0.20/0.25 was blocked by face-restore (#4), not strength.

**4. `--restore-faces` RE-INTRODUCES SynthID (was the "stubborn face" mystery).** A
Gemini face image stayed SynthID-detected through controlnet 0.15/0.20/0.25 WITH restore,
but CLEARED at 0.20 with `--no-restore-faces` (clean single-variable A/B). GFPGAN runs on
the **ORIGINAL watermarked** face at fidelity weight 0.5, blends ~half its pixels with the
StyleGAN2 prior, and composites that back OVER the diffusion-cleaned face → the watermark
returns in the face region. Content-dependent (smaller faces can clear with restore). So
raising strength cannot fix it — the face is re-pasted from the original after diffusion.
This also corrects the prior "GFPGAN scrubs SynthID / oracle-confirmed clean" claim (it was
checked on one lucky image).

### Certified controlnet strength floors (Modal GPU sweep + oracle, 2026-06-04)

Run via an isolated Modal certification harness:
controlnet, `restore_faces` OFF (it re-introduces SynthID), `--max-resolution 1536`,
each image checked on ITS OWN vendor oracle (OpenAI -> openai.com/verify, Gemini -> the
Gemini app; the two payloads are vendor-specific and never cross-checked):

| vendor | **floor** | evidence (oracle, restore OFF, <= 1536) |
|---|---|---|
| **OpenAI** | **0.20** | 2 photoreal images (9-face grid + bracelet) x seed {1,2,3} = **6/6 clean**; the bracelet that FLIPPED at 0.15 is seed-robust at 0.20 |
| **Gemini** | **0.30** | hardest face (gemini_3): 0.20 detected -> 0.30 clean on **2/2 seeds**; Gemini is the harder vendor (default 0.15 vs OpenAI 0.10) AND resolution-sensitive |

- **OpenAI 0.20 transfers to prod as-is** (OpenAI removal is resolution-independent:
  the study clears it at 0.05 across 1024-1600).
- **Gemini 0.30 is the floor at <= 1536 only.** Gemini is resolution-sensitive (study:
  native 2816 likely needs >= 0.30 even on `default`). A native-resolution
  deployment should either cap Gemini to <= 1536 and
  use 0.30, or run a native-resolution Gemini cert and expect a higher floor (~0.35+).

### Recommendations for a removal pipeline

- **Treat controlnet as PRESERVATION, not removal.** Choose it for text/structure content,
  `default` for photoreal; removal efficacy comes from STRENGTH in both.
- **Give controlnet a higher, per-vendor strength than `default`** (today both share
  `resolve_strength` 0.10/0.15, tuned for plain img2img). **Certified controlnet floors:
  OpenAI 0.20, Gemini 0.30** (see table above) — add a controlnet-specific per-vendor
  schedule to `resolve_strength` rather than reusing the `default` ladder.
- **Fix the seed in prod.** The non-determinism is purely `seed=None` (random); a fixed
  `--seed` makes every run reproduce the certified-clean result, so you ship a
  deterministic, re-certifiable config (and the seed sweep collapses to one config).
- **`--restore-faces` is OFF in prod and stays opt-in.** Two methods ship
  (`instantid` default, `photomaker`), both NON-COMMERCIAL. They REGENERATE the face
  from an ArcFace embedding via SDXL diffusion, making the output face look more
  AI-generated than the cleaned image (gloss, symmetric pores, SDXL "clean skin"
  aesthetic). For production face preservation the cleaned image from controlnet 0.20
  is the LEAST-AI state we can reach — any restore on top trades original-look for
  embedding-driven regeneration. Empirical sweep summary: GFPGAN-on-cleaned polished
  without identity recovery; PhotoMaker-V2 produced a different person; InstantID
  txt2img produced studio-portrait patchwork on group photos; InstantID
  img2img-on-cleaned with three parameter settings integrated scene context cleanly
  but never recovered original identity precisely — every setting traded one problem
  for another. See `docs/synthid-robust-identity-research-2026-06-08.md`
  "Empirical follow-up" for the full sweep.
- **No applicable local detector exists for these geometries** → the service
  can't self-verify; bake in strength margin and periodic oracle spot-checks.
- **Lesson:** visual-quality / face-identity recovery does NOT prove removal — only the
  oracle does, across MULTIPLE content types; never conclude from a partial result (the
  photoreal-only data first read as "controlnet shields, default removes"; the flat-graphic
  result reversed it; the face mystery was restore, not strength).

## Findings (with confidence and sources)

### Finding 1 — confidence: high

**Claim.** "ControlNet as the removal pipeline" is exactly CtrlRegen (ICLR 2025), and our
shipped `ctrlregen` profile is a faithful implementation of it. Its **spatial control is canny
edges** extracted from the watermarked image; its **semantic control is DINOv2-giant** via a
trainable projection + decoupled cross-attention. Clean-noise (full-strength) regeneration
scrubs the watermark from both pixel and latent space while the two control nets hold structure.

**Evidence.** CtrlRegen: spatial control "conditioned on Canny edge images extracted from the
watermarked image," integrated into the U-Net decoder blocks via a ControlNet structure;
semantic control on "DINOv2-giant" embeddings. Removal is strong: TPR@1%FPR driven from 1.00 ->
0.01 (StegaStamp) and 0.99 -> 0.12 (TreeRing). This matches our `ctrlregen/engine.py` exactly
(canny detector + `facebook/dinov2-giant` + spatial ControlNet from `yepengliu/ctrlregen`).

**Sources.** https://arxiv.org/html/2410.05470v1 · https://github.com/yepengliu/CtrlRegen ·
https://openreview.net/forum?id=mDKxlfraAn

### Finding 2 — confidence: high

**Claim.** Regeneration provably removes any bounded-perturbation pixel watermark **given enough
noise** — the operative constraint is the amount of regeneration, which is the same knob that
trades against fidelity.

**Evidence.** Zhao et al., "Invisible Image Watermarks Are Provably Removable Using Generative
AI" (NeurIPS 2024): a noise-then-reconstruct regeneration attack "guarantees the removal of any
invisible watermark" that perturbs the image within a bounded L2 distance. The guarantee is a
function of injected noise magnitude — low noise preserves detail but leaves the watermark; high
noise scrubs but discards original signal. This is the knob ControlNet conditioning is meant to
make survivable (push regeneration high while the control signal holds composition).

**Sources.** https://arxiv.org/abs/2306.01953 · https://github.com/XuandongZhao/WatermarkAttacker

### Finding 3 — confidence: high

**Claim.** The make-or-break gap: **no watermark-removal paper validates text or fine-detail
preservation.** CtrlRegen's "high perceptual quality" is FID/PSNR/quality-model only and
explicitly omits text, fine-detail, and hallucination analysis. So the literature does NOT
support the specific claim Jacob needs (text survives), it is simply unmeasured.

**Evidence.** CtrlRegen reports CLIP-FID, PSNR, Q-Align, LIQE; the fetched analysis confirms
"the paper contains no discussion of text preservation, fine-detail retention, or hallucination
artifacts," and "explicitly avoids discussing failure modes." Pixel metrics like PSNR are
acknowledged not to reflect perception, and text legibility is a different axis than FID.

**Sources.** https://arxiv.org/html/2410.05470v1

### Finding 4 — confidence: medium-high

**Claim.** Resolution is the prime suspect for our shipped ctrlregen's content destruction. We
tile to **512 px** and run full clean-noise per tile; at 512 px text occupies too few pixels per
tile to regenerate legibly, and smooth edge-free regions (no canny signal) are filled by the
DINOv2 semantic prior, which hallucinates texture/micro-text. The paper omits resolution
entirely, so this is an implementation regime it never characterized.

**Evidence.** Our `ctrlregen/engine.py`: `PROCESS_SIZE = 512`, `TILE_SIZE = 512`, full strength
on each tile. This mirrors the `_run_region_hires` insight (text needs MORE pixels under
regeneration so strokes exceed the VAE's ~8 px latent floor), but ctrlregen runs the regeneration
at LOW res, the opposite. CtrlRegen's paper gives no resolution/tiling spec to contradict this.

**Sources.** the former internal
`src/remove_ai_watermarks/_internal/ctrlregen/engine.py` (removed after this study);
resolution-omission
confirmed against https://arxiv.org/html/2410.05470v1

### Finding 5 — confidence: high

**Claim.** SDXL-native ControlNets exist, so the removal-pipeline upgrade need NOT be the SD1.5
re-architecture our current ctrlregen is. xinsir `controlnet-tile-sdxl-1.0` and
`controlnet-union-sdxl-1.0` (ControlNet++) run on SDXL base 1.0. The tile model has a `tile_var`
image-variation mode purpose-built to regenerate detail while preserving structure, at
`controlnet_conditioning_scale = 1.0`, optimal 1024 px. This is a drop-in control add to our
existing SDXL img2img.

**Evidence.** xinsir tile-sdxl model card: use cases = deblur/detail-repaint, **image variation
(preserving structure)**, super-resolution; `controlnet_conditioning_scale = 1.0`, ~30 steps,
optimal 1024x1024, works with `madebyollin/sdxl-vae-fp16-fix` (the same VAE our fp16 path
already swaps in). ControlNet-Union-SDXL / ControlNet++ merges 10+ control types (canny, HED,
tile, depth, lineart) into one SDXL model.

**Sources.** https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0 ·
https://huggingface.co/xinsir/controlnet-union-sdxl-1.0 · https://github.com/xinsir6/ControlNetPlus

### Finding 6 — confidence: high

**Claim.** The community tile-ControlNet upscale workflow runs at **LOW denoise (0.3-0.4)** —
the wrong regime for watermark removal. It preserves detail precisely by regenerating little, so
a naive tile-upscale preserves text AND preserves the watermark. The open empirical question is
whether at `conditioning_scale ~1.0` you can push denoise high enough to scrub SynthID while the
tile conditioning still holds text — the exact cell to test.

**Evidence.** Stable-Diffusion-Art ControlNet-tile upscale: denoise "typically 0.3, max ~0.4 to
avoid artifacts"; some users push 0.6 with ControlNet strength 0.5. Our own data: SynthID
survives below the removal-strength threshold (current Gemini needs notably higher denoise than
the tile-upscale regime). So the detail-preserving regime and the watermark-scrubbing regime are
on opposite ends of the denoise axis; ControlNet conditioning is the bet that they can meet.

**Sources.** https://stable-diffusion-art.com/controlnet-upscale/ ·
internal (`docs/synthid.md` strength data)

### Finding 7 — confidence: high

**Claim.** Forensic-stealth caveat: diffusion-based regeneration is among the MOST detectable
removal families. Even a ControlNet-regeneration that fools the SynthID oracle leaves forensic
traces flagging the output as "removal-processed" at >98% TPR@1%FPR. This bounds the claim (do
not over-promise "indistinguishable from an original") but does not block the use case — the
SynthID oracle still reads negative.

**Evidence.** "Removing the Watermark Is Not Enough: Forensic Stealth in Generative-AI Watermark
Removal" (arXiv:2605.09203, Goonatilake & Ateniese, GMU): across six removal attacks including
diffusion-based regeneration, independent forensic detectors separate removal-processed from
clean content at >98% TPR under a 1% FPR budget.

**Sources.** https://arxiv.org/html/2605.09203v1

### Finding 8 — confidence: low (watch, do not build on yet)

**Claim.** Partial/semantic-guided regeneration is an active sub-direction that explicitly targets
the removal-vs-fidelity tradeoff, but the specific fidelity-on-text claims were not verifiable
from the source in this pass.

**Evidence.** "Removing Watermarks with Partial Regeneration using Semantic Information"
(arXiv:2505.08234) proposes focusing regeneration on watermarked regions with semantic (VLM)
conditioning to preserve untouched areas; the PDF body did not render cleanly enough to confirm
its quantitative text/detail results. Treat as a pointer, not evidence.

**Sources.** https://arxiv.org/pdf/2505.08234

## Recommendation / decision

**ControlNet-as-removal-pipeline is worth prototyping — but not by fixing the SD1.5 ctrlregen.**
Port the structure-control idea onto an SDXL-native ControlNet as a control add to the existing
SDXL base 1.0 img2img, run it at full resolution (1024+, NOT 512 tiles), and treat the
text-vs-scrub tension as an empirical question to measure, not assume.

**Prototype (runs locally on 32 GB MPS — no dedicated GPU required):**

Compute is NOT the bottleneck. On a 32 GB Apple-silicon machine (M5 here) native SDXL already
runs entirely on MPS with no CPU fallback (~155 s at 1122x1402, verified 2026-05-31).
The prototype runs at **1024** (fewer pixels than that) with SDXL base + an SDXL
ControlNet + activations in **fp32** (MPS fp16 decodes to all-black NaN — issue #29 — confirmed
on run 1 below; fp32 is the required default on mps/cpu) — fits the 32 GB budget with vae-tiling +
attention-slicing; ~1-2 min/image, so a coarse sweep is a sub-hour background run. A dedicated GPU
is needed ONLY for the separate
native-large-Gemini (2816 px) case, which OOMs even without a ControlNet (that requires a
GPU task). The genuine external dependency is NOT compute but the **manual SynthID oracle**:
these geometries are outside the current local detector's scope, so removal is
verified by hand in the Gemini app ("Verify with SynthID") per image, regardless
of where the diffusion runs.

Runner: **`scripts/controlnet_sweep.py`** (built 2026-06-02) implements exactly this sweep —
SDXL base 1.0 + an SDXL-native ControlNet img2img, one output per (control x strength x scale)
cell, plus a `sweep_index.csv` with empty `synthid_oracle` / `text_legible` columns to fill by
hand. It uses the dedicated single-type xinsir models (`controlnet-canny-sdxl-1.0`,
`controlnet-tile-sdxl-1.0`) rather than the Union model to keep the diffusers API path robust.

    uv run python scripts/controlnet_sweep.py watermarked.png -o sweep_out

1. SDXL base 1.0 img2img + `xinsir/controlnet-canny-sdxl-1.0` / `controlnet-tile-sdxl-1.0`
   (sweep both `tile` and `canny` control), full image at 1024, `sdxl-vae-fp16-fix`.
2. Sweep the cube on fresh Gemini + gpt-image inputs that contain small/CJK text:
   - denoise strength {0.15, 0.3, 0.5, 0.7, 1.0}
   - `controlnet_conditioning_scale` {0.5, 0.8, 1.0}
   - control type {tile, canny}
3. Per cell, measure BOTH axes:
   - **removal**: Gemini app "Verify with SynthID" oracle (the only valid SynthID oracle; for
     gpt-image also openai.com/verify for provenance) — must read clean.
   - **text**: OCR round-trip / visual legibility of the small text.
   - secondary: SSIM/FID vs original for global fidelity.
4. Find the Pareto cell where the oracle is clean AND text stays legible.

**The honest fork the prototype resolves:**
- If such a cell exists -> the answer to Jacob is YES, ship an SDXL-native ControlNet removal
  profile (replacing the SD1.5 ctrlregen) tuned to that cell.
- If no cell clears both (the tension is fundamental: scrub-strength always deforms text, or
  text-preserving conditioning always spares the watermark) -> the canny/tile-ControlNet middle
  path is dead for text, and the standing answer reverts to `text-protection-research.md`: a full
  **glyph-conditioned re-render** (EasyText / TextSR on a FLUX-DiT base) is required, which is a
  base-model migration, not a control add.

**Do not:** keep tuning the 512 px SD1.5 ctrlregen for text (wrong resolution, wrong base model);
run tile-ControlNet at the community 0.3-0.4 upscale denoise and expect watermark removal (that
regime preserves the watermark); over-claim forensic invisibility (Finding 7).

## Prototype run 1 — 2026-06-02 (text axis measured; watermark axis pending the oracle)

First sweep on a real, SynthID-positive, text-dense input: the corpus tokyo-street-night
gpt-image (`88e61a38-chatgpt_tokyo.png`, 1023x1537 -> 680x1024, dense small CJK + Latin neon
signage; SynthID + C2PA confirmed, so its valid oracle is openai.com/verify). Grid: control
{canny, tile} x strength {0.3, 0.5, 0.7, 1.0} x `conditioning_scale` 1.0, fp32 on MPS. Outputs +
`sweep_index.csv` (text verdicts filled by visual inspection; `synthid_oracle` left for the
manual run) are under `/tmp/cnsweep/` (not committed — derived regenerations of corpus content).

**Measured — PSNR vs input (proxy for how much was regenerated):**
- canny: 0.3 -> 16.91, 0.5 -> 15.91, 0.7 -> 14.82, 1.0 -> 13.22 (monotonic drop = progressively
  more regeneration as strength rises; canny only pins edges, so flat regions change).
- tile: 0.3 -> 17.89, 0.5 -> 17.84, 0.7 -> 17.83, 1.0 -> 17.74 (**flat and high — near-identity
  even at strength 1.0**; tile@scale1.0 pins the whole image to the input and barely regenerates).

**Measured — text legibility (visual, focused on SMALL text; large high-contrast glyphs survive
everything because canny/tile hold their edges):**
- canny: legible at 0.3, softening at 0.5 (partial), garbling at 0.7, hallucinated pseudo-glyphs
  at 1.0 ("NEC" -> "NWENES"). Same plain-img2img small-text deformation, only big text protected.
- tile: near-identity through 0.7, only tiny alterations at 1.0 — small text preserved throughout.

**Reading (the make-or-break tension, now visible in the data):**
- **tile@scale1.0 does not actually regenerate** (flat PSNR), so it preserves all text but almost
  certainly leaves the watermark intact — it is a near-identity pass, exactly the community
  "tile-upscale preserves detail by not regenerating" regime (Finding 6), confirmed.
- **canny@scale1.0 regenerates progressively** (PSNR drops) and so could scrub — but small text
  breaks at exactly the strength where scrubbing would start to bite. canny saves big edges, not
  sub-stroke small text.
- Net on the text axis: neither cell at scale 1.0 cleanly gives "high regeneration + legible small
  text." This is the literature prior (Findings 3, 6) reproduced empirically. Lowering
  `conditioning_scale` to force small-text regeneration is the same tradeoff knob, not an escape.

**Still pending (the decisive half, cannot be done locally):** run the 8 cells through the SynthID
oracle and fill `synthid_oracle`. The most informative cells: canny 1.0 (text dead — does it at
least scrub? if not, the canny path is dead outright), canny 0.5 (text partial — does it scrub?),
tile 1.0 (text perfect — predicted to still read present). If no cell is `oracle=clean` AND
`text=yes`, the fork resolves to the glyph-re-render path (`text-protection-research.md`).

**Incidental bug caught:** the first run used fp16 on MPS (the script's original default) and
produced **all-black** outputs across every cell (2 KB PNGs, PSNR 9.22 flat) — the issue #29
fp16-VAE-NaN failure, and the fp16-fix VAE did not save it on MPS. Fixed `scripts/controlnet_sweep.py`
to default fp32 on mps/cpu (fp16 only on cuda/xpu), matching the production pipeline.

## Tuning ControlNet for text preservation across image types (research 2026-06-03)

Goal: how to configure the canny-ControlNet path to best preserve text (and faces) on diverse
images. Primary sources: diffusers ControlNet doc, the ControlNet paper (arXiv:2302.05543),
xinsir model cards, practitioner guides. The **critical reframe**: almost all community ControlNet
advice optimizes a txt2img *generation* tradeoff (control vs creative freedom). OUR context is
img2img *watermark removal*, where the objective is the opposite -- maximum faithful preservation
while regenerating just enough to scrub. So several common recommendations INVERT here.

**Removal is `strength`; everything below is preservation and does not change removal efficacy**
(only the watermark-shielding risk -- see the caveat). Set `strength` by the oracle/vendor need;
tune these to keep text/faces intact at that strength.

Knobs, ranked by impact for text:

1. **Canny edge density (the per-image lever, currently hardcoded `_CANNY_LOW=100`/`_CANNY_HIGH=200`).**
   Lower thresholds capture more/finer edges; higher thresholds keep only major outlines (diffusers
   doc + practitioner guides; ControlNet paper uses 100/200 as the default). Small-text strokes and
   fine facial features fall below the default 100/200 and are missed. **For dense small text
   (infographics, signage) lower the thresholds (~50/120, even 30/100 for facial likeness per
   practitioner tests); for high-contrast large text 100/200 already suffices.** Denser canny is
   still a BINARY thresholded edge map, so it does not carry the low-amplitude SynthID pixel pattern
   -- it passes more shape, not the watermark (still oracle-verify). This is the single highest-value
   unexplored lever and should become a CLI knob.

2. **`controlnet_conditioning_scale` -> keep at 1.0 (max structure hold).** Community defaults to 0.5
   for creative balance; we want maximum preservation, so 1.0 (xinsir canny/tile cards also recommend
   1.0). We measured text on a clean high-contrast image surviving across strength 0.1-0.5 at scale
   1.0 (PSNR ~26 flat), so scale 1.0 is the right default; only lower it if a specific image needs
   more regeneration to scrub (raises shielding risk the other way).

3. **`control_guidance_start=0.0`, `control_guidance_end=1.0` (full window) -- KEEP, do not shorten.**
   The common "end=0.5: establish structure early then let the model render detail freely" is a
   creative-generation recipe; for text it is HARMFUL -- the late free steps re-render and deform the
   glyphs. We want the edge control active through ALL denoise steps so text stays pinned. (Our
   pipeline already uses the 0->1 default; the point is to NOT adopt the shorten-the-window advice.)

4. **Control type, per image type:**
   - **Text / graphics / high-contrast -> canny** (the literature's reliable choice for defined edges
     and text; what we ship).
   - **Faces / smooth tonal content -> soft-edge / HED is a candidate worth testing.** Canny's hard
     binary threshold fractures smooth skin gradients; HED/soft-edge gives gradual edges that may hold
     faces better. UNVERIFIED for removal (softer edges may carry slightly more original signal ->
     oracle-check). A face-heavy image is the test (gemini group photos).
   - **tile -> NOT for removal.** It is near-identity (detail-enhancement at low denoise); it shields
     the watermark (measured flat PSNR ~17.8 across strength on the tokyo sweep). Do not use it as the
     removal control.

5. **Resolution** -- higher long-side = strokes span more VAE latent cells = less softening, while
   still fully regenerating. Already a knob (`--max-resolution`); for tiny text prefer native/large.

**Multi-ControlNet (canny + soft-edge), list scales e.g. `[1.0, 0.8]`** (diffusers MultiControlNet):
could hold text edges AND face geometry at once, but doubles ControlNet memory/latency and raises the
shielding risk; defer to a v2 after the single-canny path is dialed in.

**Image-type playbook (proposed, to validate with the oracle):**
- Clean high-contrast text (openai_1-style): canny 100/200, scale 1.0, full window -- already optimal.
- Dense small text / infographics (big_pic3, neon signage): canny **lower thresholds (~50/120)**,
  scale 1.0, full window, larger resolution.
- Faces / portraits: try **soft-edge/HED** control, scale 1.0; or multi-ControlNet canny+softedge.

**Hard caveat:** every change that increases preservation (higher scale, denser canny, fuller window,
softer edges) marginally REDUCES effective regeneration and so raises the chance the watermark
survives -- exactly the shielding failure mode. These geometries are outside the
current local detector's scope, so each tuning change must be re-confirmed on
the oracle. These are img2img-context recommendations derived from
generation-context sources plus our own measurements; treat the playbook as hypotheses to verify, not
settled defaults.

**Sources.** https://huggingface.co/docs/diffusers/en/using-diffusers/controlnet ·
https://arxiv.org/pdf/2302.05543 · https://huggingface.co/xinsir/controlnet-canny-sdxl-1.0 ·
https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0 · https://blog.cephalon.ai/canny-and-softedge/

## FaceID research: identity-preserving face conditioning (research 2026-06-03)

Motivation: canny alone preserves face STRUCTURE/position better than plain SDXL but does NOT hold
IDENTITY -- verified on a real Gemini group photo (gemini_3, s015): faces drift in expression and
likeness (the smile/mouth and eyes change), they are "a similar person," not the same one. Canny
carries edges, not identity, so the regenerated face is identity-drifted. To hold identity WITHOUT
copying original pixels (the hard constraint -- copied pixels carry SynthID), the conditioning must
be an identity EMBEDDING, not pixels. Primary sources: diffusers IP-Adapter doc, InstantID
(arXiv:2401.07519), IP-Adapter (arXiv:2308.06721), practitioner comparisons.

### Findings

**1. IP-Adapter FaceID conditions on an ArcFace identity VECTOR, not pixels (confidence: high).**
FaceID extracts `insightface` ArcFace `normed_embedding` (a ~512-d identity vector) via
`FaceAnalysis`, and passes it as `ip_adapter_image_embeds` -- NOT a CLIP image embedding, NOT the
original pixels. So it is constraint-compatible: the watermark (a pixel-amplitude pattern) is not in
the identity vector, and the img2img still regenerates the pixels (removal via `strength` unchanged).
It loads on any SDXL via `load_ip_adapter` (~100 MB), is fast/low-VRAM, but identity fidelity on SDXL
is ~5-10% lower than the SD1.5 line / dedicated methods.

**2. Multiple distinct faces ARE handled, via regional attention masks (confidence: high -- THE key
unlock).** This is the make-or-break for group photos (our hardest case). diffusers supports a LIST
of IP-Adapter face images each with its own binary region mask: `IPAdapterMaskProcessor` builds the
masks, `set_ip_adapter_scale([[s1, s2, ...]])`, and `cross_attention_kwargs={"ip_adapter_masks":
masks}`. So you detect each face, extract its own ArcFace embedding, assign it a region mask, and one
pass preserves N different identities simultaneously. (InstantID, by contrast, is single-subject --
it averages embeddings for multiple refs, which is wrong for distinct people -- so for group photos
**IP-Adapter FaceID + masks beats InstantID**.)

**3. IP-Adapter + ControlNet + img2img compose (confidence: high).** The doc shows IP-Adapter +
ControlNet (depth) in one pipeline and IP-Adapter + img2img (`strength`). Our target stack is the
union: `StableDiffusionXLControlNetImg2ImgPipeline` (canny = structure) + `load_ip_adapter` (FaceID =
identity) + `strength` (removal). `set_ip_adapter_scale` (1.0 = image-only, 0.5 = balanced) is the
identity-hold knob. API friction to verify in implementation: that `ip_adapter_masks` via
`cross_attention_kwargs` works on the *ControlNet img2img* pipeline (the masking is an attention-
processor feature, so it should be pipeline-agnostic, but confirm).

**4. InstantID / PuLID positioning (confidence: medium).** InstantID does not train the UNet so it
composes with canny/depth ControlNets, and gives better single-face fidelity than FaceID -- but it is
single-subject (needs its own landmark ControlNet + dedicated weights). PuLID has the best identity
fidelity but is heaviest and Flux-leaning. For our multi-face, constraint-bound, SDXL-canny case,
IP-Adapter FaceID + masks is the right first build; InstantID/PuLID are single-portrait upgrades.

### Architecture (proposed)

```
detect faces (insightface) -> per face: ArcFace embed + region mask
one img2img pass:
  image=init, control_image=canny(init),                      # structure (existing)
  ip_adapter_image_embeds=[face_embeds],                       # identity per face
  cross_attention_kwargs={"ip_adapter_masks": face_masks},     # each face -> its region
  controlnet_conditioning_scale=1.0, set_ip_adapter_scale(~0.6),
  strength=vendor-adaptive                                     # removal (unchanged)
```
Pixels are regenerated (SynthID removed by `strength`), structure held by canny, each face's identity
held by its masked ArcFace vector -- no original pixel copied.

### Risks / honest costs

- **Shielding risk (same wall):** FaceID conditioning, like canny, reduces effective regeneration ->
  higher `set_ip_adapter_scale` raises the chance SynthID survives in the face region (echo of why the
  old region-hires failed). MUST oracle-verify removal at the chosen FaceID scale; keep `strength` at
  the vendor threshold.
- **New heavy dependency:** `insightface` + `onnxruntime` + the `buffalo_l` model (~300 MB, downloaded
  on first use). Detection + embedding is CPU/ONNX, separate from the diffusion.
- **Detection floor:** insightface needs faces large enough (det_size ~640); tiny faces in a dense
  group may not be detected -> not preserved (falls back to canny-only for those).
- **Identity ceiling:** SDXL FaceID is ~5-10% off true identity -- a meaningful boost over canny-only
  drift, NOT a perfect face swap. Set expectations; PuLID/InstantID are the higher-fidelity (heavier)
  paths if needed.
- **Value scales with strength:** at low strength (OpenAI 0.10) faces barely drift, so FaceID is
  marginal; at the higher strength then used for a hard vendor (Google 0.30), FaceID earned its keep.

### Build plan (staged)

- v1: optional `--face-id` flag on `--pipeline controlnet`. Detect faces; if any, run the masked
  FaceID pass (works for 1 or N faces -- masks generalize). If none detected, fall through to plain
  canny. Oracle-verify SynthID removal is preserved at the default FaceID scale on a face image.
- v2 (if identity still short): InstantID for single-portrait, or PuLID, as a higher-fidelity opt-in.

**Sources.** https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter ·
https://huggingface.co/h94/IP-Adapter-FaceID · https://arxiv.org/pdf/2401.07519 (InstantID) ·
https://instantid.github.io/ · https://arxiv.org/abs/2308.06721 (IP-Adapter)

### FaceID prototype run 1 -- 2026-06-03 (NEGATIVE on dense small-face groups)

Built and shipped the masked multi-face FaceID layer (`--face-id`, `face_id.py`, `faceid` extra;
all three were removed after this run and never reached the tracked tree).
First real run on the gemini_3 group photo (Google, s015, scale 0.6, native 2816 via cap 1536):
insightface detected **17 faces**, the masked multi-face pass composed and ran end-to-end (non-black
output), so the API is correct. **At s015 the result is a clear FAILURE: every face corrupted --
melted/discolored/psychedelic, materially WORSE than canny-only.**

**ROOT CAUSE FOUND (confirmed by ablation, not speculation) -- it is STRENGTH, not scale/masks/faces.**
Investigated the real data: masks are fine (max overlap depth 2, 33% coverage, only 0.2% of pixels
double-covered -- NOT an overlap problem), embeddings are fine (`normed_embedding` norm 1.000), the
FaceID LoRA is not required for SDXL (h94 model card), and faces span 34-181 px (7 medium + 10 tiny).
None of those is the cause. The decisive test: the SAME image + FaceID at **strength 0.5** produces
**clean, coherent faces across the whole group** (no psychedelic artifacts). So FaceID needs
substantial regeneration: the h94 usage is full generation (txt2img, 30 steps); at our removal
strength (0.10-0.15 = ~7 effective steps) the strong identity cross-attention cannot reconcile with
a latent that is ~85% the untouched original, so it smears identity-colored noise onto the faces.

**This is a FUNDAMENTAL tension, not a tuning bug:** watermark removal wants LOW strength (minimal
degradation, just enough to scrub), FaceID wants HIGH strength (regenerate the face to impose
identity). They are opposed. At strength 0.5 FaceID works AND removes the watermark, but the whole
image regenerates much more (canny still holds text/edge structure, but texture/detail drifts well
beyond the 0.15 "minimal degradation" target). So `--face-id` is a HIGH-STRENGTH option: it trades
whole-image fidelity for face identity, and is a footgun at the low default strength (guaranteed
garbage). Required follow-up code guard: when `--face-id` is set, floor `strength` at ~0.5 (or refuse
+ warn) -- never run FaceID at the vendor-adaptive removal strength. Open question: whether
high-strength FaceID's whole-image drift is acceptable for face-centric images, or whether identity
preservation at LOW strength needs a different mechanism entirely (FaceID structurally cannot do it). (Infra lesson: the `faceid` extra must
stay numpy<2.0 -- pin `onnx<1.18` + `scipy<1.18`; pinning numpy UP, as the first build did, leaves a
numpy-1.26 env with a numpy-2-only scipy that crashes the diffusers import via `np.long`.)

## Face preservation, done properly (research 2026-06-03, after the FaceID failure)

The FaceID run failed and I wrongly concluded "faces can't be preserved." Re-research corrected the
understanding. The hard constraint is unchanged: to remove the watermark FROM a face the face MUST
be regenerated (freezing it leaves SynthID), so the goal is identity-preserving REGENERATION of the
face, at minimal overall image degradation. Three things I got wrong and the corrected picture:

**What I got wrong:** (1) I applied FaceID at GLOBAL high strength -- the literature is clear the
architecture must be REGION-ADAPTIVE (face region handled separately, background stays low-strength);
(2) I used IP-Adapter FaceID, the WEAKEST identity tool -- InstantID uses an ArcFace encoder and hits
82-86% face-recognition similarity vs FaceID's weak CLIP-ish signal; (3) I missed the entire
face-restoration class (CodeFormer / GFPGAN), which is purpose-built for "regenerate a face, keep
identity."

**The most promising mechanism -- CodeFormer face-restoration post-pass (confidence: high on the
mechanism, unverified on our watermark).** CodeFormer is a VQ-VAE: a frozen discrete CODEBOOK of HQ
facial priors + a Transformer that predicts code *tokens* from the input, and a frozen decoder that
regenerates the face FROM THE CODEBOOK ENTRIES -- "does not depend on feature fusion with low-quality
cues." So the output face pixels come from a finite learned codebook, NOT from the input pixels:
**the SynthID pixel-amplitude pattern physically cannot survive a codebook re-synthesis** -- a
stronger scrub than low-strength img2img (which keeps ~85% of the latent). Fidelity knob `w` in
[0,1]: higher w preserves identity but fuses MORE low-quality (input) cues (more watermark risk),
lower w leans on the codebook (cleaner scrub, identity drift) -- the same scrub-vs-fidelity tension,
settled per-image by the oracle; there is likely a `w` that holds identity AND clears the oracle.

**Constraint-compatible architecture:** run the normal canny low-strength controlnet removal globally
(minimal degradation everywhere), then detect+align each face, run CodeFormer on the **ORIGINAL** face
crop (to capture true identity AND re-synthesize from the codebook = scrub), and composite the
CodeFormer output (codebook-generated, not original pixels -> no copy, no watermark) into the cleaned
image. Decouples whole-image minimal-degradation from face identity -- no high GLOBAL strength needed.

**Honest costs/caveats:** (a) **License -- CodeFormer is NTU S-Lab 1.0 (non-commercial/research)**, so
it cannot be bundled in this MIT tool for general use; the license-clean alternative is **GFPGAN
(Apache-2.0)**, slightly lower quality. (b) Deps (basicsr/facexlib) are heavy and numpy-version-finicky
(same class of conflict as insightface). (c) CodeFormer is a *restoration* model -- it can subtly
alter expression/asymmetry; identity is held but not pixel-identical. (d) **The watermark-scrub is
mechanistically strong but UNVERIFIED -- must oracle-check.** InstantID + region-adaptive strength is
the alternative if the restoration route disappoints, but it is more complex (differential strength).
Prototype plan: validate CodeFormer on a real face in a THROWAWAY env (identity held? oracle clean?)
before any project-env integration or the license/GFPGAN decision.

### CodeFormer prototype -- VALIDATED end-to-end 2026-06-03 (oracle-confirmed)

Prototyped the CodeFormer face-restoration post-pass (codeformer-pip in a throwaway venv, forced CPU
-- the pip wrapper has an MPS device-mismatch bug) on the gemini_3 group photo (18 faces). Pipeline:
`all --pipeline controlnet --strength 0.15` (sparkle + SynthID removed from the whole image, minimal
degradation) -> CodeFormer on the ORIGINAL faces -> feather-composite the CodeFormer faces into the
all-cleaned image. Oracle results (Gemini app "Verify with SynthID"), isolating each part:
- pure controlnet-0.15 background (no faces): **clean** -> the background scrub works at 0.15 (no
  ControlNet-shielding problem for Google on this image).
- composite with CodeFormer faces at **w0.7**: **SynthID DETECTED** -> high fidelity fuses too much of
  the original face signal (the watermark) through.
- composite at **w0.5**: **clean**. composite at **w0.3**: **clean**.
So the scrub-vs-fidelity threshold is between 0.5 and 0.7; **w=0.5 is the sweet spot** (highest
fidelity / best identity that still clears the oracle). Identity at w0.3-0.7 all looks like the same
person (the face is large enough), so the lower w costs little.

**This VALIDATES the corrected face-preservation approach** (and refutes my earlier "faces can't be
preserved" / FaceID conclusion): controlnet low-strength background scrub + CodeFormer-codebook face
re-synthesis at w~0.5 + feather composite = oracle-clean SynthID removal everywhere (background AND
faces), identity preserved, minimal overall degradation, zero original-pixel copying (CodeFormer faces
are codebook-generated). CodeFormer's discrete-codebook re-synthesis DOES scrub the pixel watermark,
but only when w is low enough that the decoder leans on the codebook rather than fusing the input
(watermark-carrying) features -- exactly the predicted fidelity-vs-scrub tension, with an empirical
clean threshold at w<=0.5.

**Production TODO (not built -- still a throwaway prototype):** (1) license -- CodeFormer is NTU S-Lab
(non-commercial); decide CodeFormer-as-user-installed-extra vs GFPGAN (Apache-2.0, re-verify it scrubs
at its fidelity setting); (2) wire a `--restore-faces` post-pass (detect -> restore w~0.5 -> feather
composite) onto the controlnet pipeline; (3) handle the MPS device bug (force CPU for the face model
or fix); (4) re-verify the w threshold on more images / vendors (w=0.5 confirmed on one Gemini group
photo only).

**Sources.** https://arxiv.org/abs/2206.11253 (CodeFormer) · https://github.com/sczhou/CodeFormer ·
https://arxiv.org/pdf/2401.07519 (InstantID) ·
https://openaccess.thecvf.com/content/WACV2024/html/Suin_Diffuse_and_Restore_A_Region-Adaptive_Diffusion_Model_for_Identity-Preserving_Blind_WACV_2024_paper.html (region-adaptive) ·
https://arxiv.org/pdf/2504.12809 (saliency-aware watermark removal)

## Provenance

Hand-run primary-source pass, 2026-06-02. Sources fetched and quoted above; the central
make-or-break claim (structure-conditioned high-strength regeneration scrubs the watermark while
keeping text) is **unverified and explicitly flagged as the thing the local prototype must
measure** (against the manual Gemini SynthID oracle) — the literature supports removal (Findings 1, 2) and supports structure-preserving
regeneration (Finding 5) but never jointly validated text (Finding 3). No code change implied
until the prototype validates a Pareto cell on the SynthID oracle.
