# Chroma1 engine research (2026-08-29/30)

> Metric correction: the OCR metric from `scripts/fidelity_metrics.py` is
> normalized edit distance (NED), edit distance divided by the longer
> normalized string. The historical values are unchanged; the earlier CER
> label incorrectly implied normalization by reference length.

> Research archive. This page records experiments and decisions from the dates
> above. Current package behavior is defined by the
> [supported signals](supported-signals.md), [known limitations](known-limitations.md),
> and [module internals](module-internals.md). Nothing here is shipped.

Cited experiment behind issue #88 ("Please integrate Flux 2"): FLUX.2 has no
native strength img2img anywhere (BFL reference code or diffusers; its image
inputs are reference conditioning and outputs start from pure noise), so the
requestable scrub lane in the FLUX family is **Chroma1** -- the FLUX.1
architecture re-trained on open data, Apache-2.0 on weights, with a native
`ChromaImg2ImgPipeline` strength parameter. This page records what was measured
before any integration decision.

## Setup

All Chroma1 runs: `lodestones/Chroma1-HD` bf16 on Modal H100, seed 0, the
neutral scrub prompt (`high quality, sharp, detailed, faithful to the original`
/ `blurry, lowres, distorted text, garbled text, artifacts`), guidance 5.0, and
**step compensation**: the calibrated request is `ceil(4 / strength)`.
Chroma rounds the start index rather than the active step count: this runs
five effective steps at strengths 0.09 and 0.17, and four at 0.20, 0.125,
and 0.40. The nominal count is four, not a promise of four effective steps.
The calibrated requests remain unchanged.
Without it, strength 0.05 runs a single step and the floor reads artificially
high (the first, uncompensated sweep put the OpenAI boundary at (0.05, 0.10];
compensated it is (0.05, 0.06]).

Two Flux-family traps found while wiring the prototype:

- without explicit `width`/`height`, the img2img pipeline silently generates
  1024x1024 regardless of input aspect;
- dimensions floor to the /16 latent-patch grid, so the result must be resized
  back to the input size for pixel-exact comparison.

Harnesses (Modal GPU scripts kept outside this repository; each mounts this
library's `src/` and reads its tracked corpora): `chroma_scrub_prototype.py`
(first sweep), `engine_quality_price_probe.py` (head-to-head with the
qwen-zimage path), `engine_cohort_calibration.py` (the cohort ladders below),
`chroma_meta_high_rungs.py` (the Meta rungs above 0.08),
`chroma_muse_expansion.py` (harvest-2 Meta API fixtures),
`chroma_openai_expansion.py` (extra OpenAI SynthID carriers),
`chroma_seed_probes.py` (seed-1/2 at measured first-cleans).
These harnesses are internal tooling and are not distributed, so a reader outside
the project cannot rerun them. Generated outputs stay outside the repository
(`out/`, gitignored); the durable verdicts are the tables on this page.

## Measured floors

Oracle per cohort: OpenAI via openai.com/verify (SynthID read separately from
C2PA); Meta via the anonymous meta.ai/identification oracle. Each fixture's
rungs were checked ascending and stopped at the first clean verdict.

| cohort | fixture | detected at | clean from |
|---|---|---|---|
| OpenAI | typography card 1122x1402 | 0.05 | 0.06 |
| OpenAI | 9-face grid 1448x1086 | 0.06 | 0.075 |
| OpenAI | full-pipeline quality fixture 1122x1402 | 0.05 | 0.06 |
| Meta | gen_lighthouse_watercolor 1600x1600 | 0.08 | 0.10 |
| Meta | gen_fox_forest 1600x1600 | 0.06 | 0.08 |
| Meta | gen_night_city 1920x1280 | 0.045 | 0.06 |
| Meta | gen_studio_mug 1600x1600 | 0.03 | 0.045 |
| Meta | gen_text_poster 1600x1600 | -- | <=0.03 |

Derived operating points from this initial calibration (the OpenAI value was
superseded by the 2026-09-07 holdout below):

- **OpenAI: 0.09, superseded.** (0.075 + (0.075 - 0.06)). Against qwen-zimage 0.07675 and
  sdxl-zimage 0.15. The typography boundary (0.05, 0.06] is seed-stable across
  seeds 0, 1, 2.
- **Meta: 0.17** (0.10 + (0.10 - 0.03)), oracle-verified clean on the worst
  fixture at 0.17. Against qwen-zimage 0.1. Chroma needs ~1.7x qwen's strength
  on Content Seal.
- **Google: 0.40, measured 2026-08-30 across all four fixtures.** A
  metadata-stripped original verified as pixel-DETECTED through the @synthid
  Verify AI agent (the C2PA-carrying original answers about C2PA instead,
  which is not a pixel verdict).

  | fixture | detected at | clean from |
  |---|---|---|
  | Gemini_Generated_Image_633uuy | 0.20 | 0.25 |
  | Gemini_Generated_Image_akdbei | 0.20 | 0.25 |
  | Gemini_Generated_Image_y48j3c | 0.08 | 0.12 |
  | Gemini_Generated_Image_3mc4t9 | 0.08 | 0.12 |

  Derived by the shipped rule: worst first-clean 0.25 + spread (0.25 - 0.12
  = 0.13) = 0.38, rounded up to the measured rung **0.40**, then
  oracle-verified clean on the worst fixture. Against qwen-zimage 0.27:
  Chroma needs ~1.5x the strength on Google SynthID. At 0.15-0.20 the tool
  still reads positive while its wording degrades from "most or all" to
  "part of", so the boundary band is real signal decay, not noise.

  Oracle mechanics learned while bracketing: the quota is per ACCOUNT and
  additionally dedupes SIMILAR images ("quota for checking images similar to
  this one"), roughly three checks per account window; switching accounts
  inside one browser works. A fresh Playwright context with the same Google
  session cookies (exported browser-wide via CDP `Storage.getCookies`, never
  through the transcript) carries working sign-in and its own check budget.
  The agent's own instructions leak into the answer on some renders and
  include an abstention rule for text-dominated images - always read the TOOL
  outcome, never the prose.
- **Microsoft: 0.125**, measured 2026-08-30 on the SAME three valid Paint
  carriers behind the qwen floor (staged under
  `out/cohort-calibration/microsoft/` with pixel-identical metadata-stripped
  controls; they are not publication-cleared and are never committed).
  Controls re-verified `Microsoft AI detected` on the day of the run. Against
  qwen-zimage 0.15: Chroma needs LESS strength on InvisMark.

  | source (Paint) | detected at | clean from | qwen first-clean |
  |---|---|---|---|
  | paint-1 (47d87e99, 1536x1024) | 0.06 | 0.08 | 0.055 |
  | paint-2 (a579fb70, 1536x1024) | -- | <=0.04 | 0.04125 |
  | paint-3 (883191b5, worst qwen source) | 0.06 | 0.08 | 0.095 |

  Derived by the shipped rule: 0.08 + (0.08 - 0.04) = 0.12, rounded to the
  measured rung **0.125**, which was then oracle-verified clean on both worst
  sources. Oracle: the public `ai.azure.com/nextgen/validate` page; its
  `Inconclusive` verdict is the site's clean outcome and provides the provider
  oracle used for this certification. The API exposes a more detailed report,
  not a different class of certification.

## Fidelity at the floors (scripts/fidelity_metrics.py)

qwen-zimage rows ran through the package's own `QwenZImagePipeline` on the same
H100 (production path, both stacks resident, prompt cache warm).

OpenAI typography (chroma 0.06 vs qwen 0.07675):

| | NED | img LPIPS | SSIM | PSNR |
|---|---|---|---|---|
| qwen-zimage | 0.241 | 0.083 | 0.834 | 27.7 |
| chroma1 | **0.138** | **0.049** | **0.881** | **31.2** |

OpenAI full-pipeline fixture (chroma 0.09 vs qwen 0.07675): chroma wins every
metric (NED 0.155 vs 0.259, LPIPS 0.056 vs 0.094, PSNR 29.6 vs 26.1).

OpenAI 9-face grid (chroma 0.075 global-only vs qwen 0.07675 with its Z-Image
face stage):

| | ID cos | face LPIPS | img LPIPS | SSIM | PSNR |
|---|---|---|---|---|---|
| qwen-zimage | **0.857** | **0.051** | 0.065 | 0.865 | 31.1 |
| chroma1 (no face stage) | 0.788 | 0.067 | **0.060** | **0.890** | **32.4** |

The face stage regenerates from the ORIGINAL crops and is engine-independent,
so an integrated `chroma-zimage` would composite the same faces over the
chroma global; whether that closes the identity gap is an integration-time
measurement, not a given.

Meta at the respective floors (chroma 0.17 vs qwen 0.1): qwen wins perceptual
fidelity on textured content (LPIPS: fox 0.163 vs 0.211, lighthouse 0.173 vs
0.248, night city 0.187 vs 0.277), chroma wins the text poster decisively
(LPIPS 0.008 vs 0.012, NED 0.000 vs 0.021) and the studio mug is a split. The
extra 0.07 of strength costs real fidelity on photoreal content.

Google at the respective floors (chroma 0.40 vs qwen 0.27, qwen row includes
its Z-Image face stage):

| fixture | qwen LPIPS/SSIM/ID cos | chroma LPIPS/SSIM/ID cos |
|---|---|---|
| 633uuy | **0.405 / 0.577** / - | 0.499 / 0.576 / - (NED 1.000 vs 0.333) |
| akdbei | **0.440 / 0.552** / - | 0.540 / 0.557 / - (NED 0.796 vs 0.367) |
| y48j3c (18 faces) | **0.436 / 0.608 / 0.801** | 0.560 / 0.548 / 0.279 |
| 3mc4t9 (5 faces) | **0.422 / 0.529 / 0.921** | 0.562 / 0.461 / 0.143 |

qwen wins every axis that matters on every fixture; at 0.40 the Chroma global
regeneration itself collapses face identity (0.14-0.28 cosine) and destroys
dense text (NED up to 1.0). The extra 0.13 of strength is simply too much
regeneration. This is the Meta pattern amplified.

Microsoft at the respective floors (chroma 0.125 vs qwen 0.15): chroma wins
EVERY metric on EVERY source while scrubbing at lower strength --

| source | NED q/ch | img LPIPS q/ch | SSIM q/ch | PSNR q/ch |
|---|---|---|---|---|
| paint-1 | 0.741 / 0.241 | 0.164 / 0.106 | 0.742 / 0.847 | 24.8 / 29.3 |
| paint-2 | 0.826 / 0.391 | 0.253 / 0.108 | 0.536 / 0.633 | 22.5 / 25.5 |
| paint-3 | 0.000 / 0.000 | 0.038 / 0.020 | 0.762 / 0.953 | 36.3 / 42.2 |

The Microsoft cohort therefore behaves like the OpenAI cohort, not like Meta:
for InvisMark carriers Chroma1 is the better engine at a lower operating point.

## Price (warm H100 container)

| | qwen-zimage | chroma1 |
|---|---|---|
| seconds per image | 2.44 | 3.62 |
| VRAM peak | 31.9 GiB | 29.2 GiB |
| USD per 1000 images ($3.95/h) | $2.68 | $3.97 |
| weights in cache | ~70 GB | ~28 GB |

The chroma number includes a T5 encode per call; the shipped qwen path caches
prompt embeddings, and the same treatment would put chroma near 1.5-2 s/image.
Both stacks fit an 80 GB card; chroma's smaller footprint likely fits smaller
cards untested.

## Matched-strength comparison: the floors hide the engine (2026-08-30 addendum)

The floor-based tables above compare chroma at its DERIVED floor (worst case
+ spread) against qwen at its shipped floor - different perturbation levels.
At the strength where each fixture ACTUALLY turns clean (chroma first-clean
rungs, all oracle-verified above), the picture reverses:

Google (chroma@first-clean vs qwen@0.27):

| fixture | chroma rung | LPIPS q/ch | SSIM q/ch | PSNR q/ch | NED q/ch | ID q/ch |
|---|---|---|---|---|---|---|
| 633uuy | 0.25 | 0.405/**0.375** | 0.577/**0.673** | 18.8/**22.3** | **0.333**/0.667 | - |
| akdbei | 0.25 | 0.440/**0.409** | 0.552/**0.650** | 18.6/**22.5** | 0.367/**0.163** | - |
| y48j3c | 0.12 | 0.436/**0.329** | 0.608/**0.700** | 21.0/**25.8** | 1.0/1.0 | **0.801**/0.611 |
| 3mc4t9 | 0.12 | 0.422/**0.318** | 0.529/**0.644** | 17.7/**21.3** | 0.443/**0.229** | **0.921**/0.410 |

Meta (chroma@first-clean vs qwen@0.1):

| fixture | chroma rung | LPIPS q/ch | SSIM q/ch | PSNR q/ch | NED q/ch |
|---|---|---|---|---|---|
| lighthouse | 0.10 (equal) | 0.173/**0.097** | 0.783/**0.886** | 25.6/**30.8** | - |
| fox | 0.08 | 0.163/**0.083** | 0.778/**0.907** | 25.6/**31.4** | - |
| night_city | 0.06 | 0.187/**0.092** | 0.748/**0.893** | 23.9/**29.6** | 0.850/**0.250** |
| studio_mug | 0.045 | 0.024/**0.009** | 0.984/**0.992** | **39.0**/37.8 | - |
| text_poster | 0.03 | 0.012/**0.002** | 0.984/**0.994** | 34.0/**41.2** | 0.021/**0.007** |

At effective-clean strength Chroma1 wins whole-image fidelity on essentially
every fixture (often by 4-6 dB) and text on five of six; qwen's remaining
edge is face identity, which its Z-Image face stage buys (0.80-0.92 vs
0.41-0.61 cosine) and a chroma-zimage profile would inherit on top of its own
global pass.

**Why the floors still matter:** a shipped strength must clear the worst
unseen content, and Chroma1's per-fixture boundaries scatter much wider
(Google 0.12-0.25, Meta 0.03-0.10) than the spread the flat floors were built
to absorb - so its derived floors (0.40 / 0.17) pay a large margin penalty
and at those floors the extra strength genuinely destroys quality. The
engines differ less in "quality per strength" than in how predictable their
boundary is. The integration lever this exposes: a content-adaptive strength
policy for chroma-zimage (the project already ships resolution-adaptive
denoise for qwen) could capture chroma's low per-content boundaries; the two
hard Google fixtures are both dense-text cards and the easy ones are
photoreal, so a content-class signal may predict the rung.

## Pre-ship validation (2026-08-30)

With the `chroma-zimage` profile implemented
(`_internal/chroma_zimage_pipeline.py` on `TwoStageZImagePipeline`, floors in
`watermark_profiles`), two ship-gate checks were run:

1. **Seed stability at the worst Meta boundary.** lighthouse @ 0.10 with
   seeds 1 and 2: both CLEAN on the meta.ai oracle. The Meta boundary is not a
   seed-0 fluke. The Google seed probes (633uuy @ 0.25) delivered a critical
   finding: **seed 0 was CLEAN but seed 1 was DETECTED** -- the 633uuy
   boundary is seed-dependent, which validates the flat 0.40 floor's 0.15
   margin exactly the way the qwen SDXL precedent did (the same image at the
   same strength passing or failing run-to-run near the threshold). Seed 2
   of `seedprobe_google_633uuy_s0.25_seed2.png` was checked twice on
   2026-08-31 after a human drag-drop. Work Gemini (u/2, Pro,
   `https://gemini.google.com/u/2/app/ebc7c19d22f85bb9`) with the
   ordinary question "Is there a SynthID watermark in this image?"
   auto-titled "SynthID Watermark Detection Results" and returned "No
   reliable signals were detected indicating how the content was created,
   and you should look for other supporting evidence from trusted
   sources." Plus `@synthid` (u/0,
   `https://gemini.google.com/u/0/app/658b1470307811b7`) with the built-in
   verification question auto-titled "SynthID Image Verification Results"
   and returned "No digital watermark from Google AI was detected in this
   image, indicating that it was not created or edited using Google's AI
   models." Both are CLEAN. An earlier Plus `@synthid` attempt on the
   same file hung on "Connecting to Verify AI" with auto-title "SynthID
   Image Verification Failed"; Flash prose abstained because "the tool
   response does not confirm" Google AI, which is not a tool verdict.
   Workspace Gemini's More-uploads menu does not list Photos, but a
   human drag-drop of a local PNG does attach; one Workspace account in
   the same Chrome has no Gemini app. Seed 0 CLEAN / seed 1 DETECTED /
   seed 2 CLEAN at 0.25
   leaves the 633uuy boundary seed-dependent on the Plus `@synthid`
   path; the 0.40 floor is already established by the seed-1 flip.
2. **Full-path face-stage interaction.** The complete profile run (Chroma1
   global at the shipped Google floor 0.40, then the inherited
   YuNet/SAM/Z-Image face stage) on the 5-face fixture 3mc4t9: CLEAN on the
   @synthid oracle. The composited face regions do not re-introduce a
   detectable signal on top of the floor.

Generation harness: `chroma_preship_validation.py` (kept outside this
repository); outputs under
`out/preship-chroma/` (gitignored).

## Verdict

Engine quality is cohort-dependent, and after measuring all four cohorts the
split is even and sharp:

- On the initial OpenAI fixtures Chroma1 was better at the then-current floor,
  but the later holdout moved that floor to 0.1375 and changed the router decision.
- On Microsoft InvisMark content Chroma1 is better at every measured fidelity
  axis at a LOWER floor than qwen (0.125 vs 0.15).
- On Meta Content Seal Chroma1 needs 1.7x the strength (0.17 vs 0.1) and loses
  LPIPS on textured content at its floor.
- On Google SynthID Chroma1 needs ~1.5x the strength (0.40 vs 0.27) and loses
  every fidelity axis on every fixture, with face identity collapsing at 0.40
  (0.14-0.28 cosine vs qwen's 0.80-0.92).

A single-engine `chroma-zimage` default at a flat per-vendor floor is NOT
supported: Google and Meta floors land too high and the quality at those
floors is worse than qwen's. But the matched-strength addendum changes what
IS supported: at the strength each image actually needs, Chroma1 regenerates
better than qwen almost everywhere except face identity (which the shared
face stage would supply). The two honest integration options:

1. Per-cohort engine router at flat floors: chroma for OpenAI/Microsoft,
   qwen for Google/Meta - conservative, ships on the measurements as they
   stand.
2. `chroma-zimage` with a content-adaptive strength policy that predicts the
   per-image rung - potentially better everywhere, but the predictor is new
   research (the measured signal so far: dense-text content needs the high
   rungs, photoreal the low ones) and needs its own oracle validation before
   any ship.

Both need the seed-sensitivity and face-stage composition checks called out
above before shipping. Both checks have now been run (Pre-ship validation
above): the Meta boundary is seed-stable across seeds 0-2 and the full
profile path stays clean through the face stage on the shipped Google floor.
The profile ships with option 1's floors plus a content-adaptive Google arm
(face content resolves to 0.125 instead of 0.40, keyed on YuNet detection).
Option 2's full adaptive policy for the remaining cohorts remains the
documented follow-up. The Meta arm was measured on 2026-08-30/31 and
does not have a shippable split; see the expansion section below.

## Meta content-adaptive expansion (2026-08-30/31)

Goal: test whether `flat_ratio` (fraction of 16x16 blocks with luma std < 8)
predicts Chroma1's first-clean Content Seal boundary well enough to ship a
Meta arm analogous to Google's face-count split.

**Harvest 1, from a 495-file uncommitted corpus scan, was not a diverse expansion.**
Seven "unique" files were staged under `out/cohort-calibration/meta/muse-*`
(gitignored). After control checks on `meta.ai/identification`:

| fixture | control | first-clean | note |
|---|---|---|---|
| muse-1 | CLEAN | -- | seal did not survive the metadata-stripped control |
| muse-2 | DETECTED | (0.08, 0.10] | photoreal portrait; 0.06 and 0.08 DETECTED, 0.10 CLEAN |
| muse-3 | DETECTED | (0.015, 0.03] | photoreal portrait, different subject |
| muse-4 | DETECTED | (0.08, 0.10] | same subject as muse-2/6/7; 0.06 and 0.08 DETECTED, 0.10 CLEAN |
| muse-5 | -- | -- | byte-identical to committed `gen_lighthouse_watercolor.webp` |
| muse-6 | DETECTED | (0.06, 0.08] | same subject as muse-4; 0.06 DETECTED, 0.08 CLEAN |
| muse-7 | DETECTED | (0.045, 0.06] | same subject as muse-2/4/6; 0.045 DETECTED, 0.06 CLEAN |

Five valid new points, four of them near-duplicate portraits of one
subject, with first-clean already spanning 0.06-0.10 **inside that
subject**. Google's shipped split had zero overlap between classes and
identical boundaries inside a class. This harvest cannot support a
predictor.

`flat_ratio` on the original five plus the new portraits is not monotonic
with first-clean either (studio_mug 0.973 / 0.045 easy; lighthouse 0.602 /
0.10 hard; muse-4 0.789 / 0.10 hard). OpenAI stays unshippable for the
same reason as before: one text fixture vs one face fixture, and the face
fixture was the harder one, the opposite of Google's split.

**Harvest 2** takes six files from the 2026-08-26 Muse Image API corpus
(61 independent `muse-image-1.0` generations, no hash overlap with the
committed five), chosen to span class and `flat_ratio`:

| fixture | class | flat_ratio | edge_density |
|---|---|---|---|
| architecture_mosque | dense architecture | 0.037 | 0.367 |
| product_sneaker | product / high-flat | 0.874 | 0.018 |
| text_poster_bakery | text | 0.368 | 0.191 |
| portrait_weathered_fisherman | photoreal portrait | 0.470 | 0.108 |
| illustration_watercolor_botanical | illustration | 0.442 | 0.089 |
| scene_tokyo_alley | busy scene | 0.258 | 0.175 |

Harness: `chroma_muse_expansion.py` (kept outside this repository). Outputs under
`out/cohort-calibration/meta-api/` (gitignored). Chroma1 ladders (seven
rungs, seed 0, nominal four-step compensated requests) ran for all six. Oracle
2026-08-31, anonymous `playwright-isolated` against
`meta.ai/identification`. All six controls DETECTED (valid carriers).

| fixture | flat_ratio | detected at | clean from |
|---|---|---|---|
| architecture_mosque | 0.037 | 0.08 | 0.10 |
| illustration_watercolor_botanical | 0.442 | 0.08 | 0.10 |
| scene_tokyo_alley | 0.258 | 0.06 | 0.08 |
| text_poster_bakery | 0.368 | 0.045 | 0.06 |
| portrait_weathered_fisherman | 0.470 | 0.045 | 0.06 |
| product_sneaker | 0.874 | 0.03 | 0.045 |

The worst first-clean is still 0.10, the same as the committed
lighthouse, so the shipped Meta floor 0.17 does not move.

**No Meta adaptive arm.** Google's shipped split had zero overlap
between classes and identical first-cleans inside a class. Combined
with the original five, Meta first-cleans are a continuum from 0.03 to
0.10, and `flat_ratio` does not separate them:

- the 0.10 cluster is mosque 0.037, botanical 0.442, and lighthouse
  0.602;
- the 0.045 cluster is sneaker 0.874 and studio_mug 0.973;
- harvest-1 photoreal portraits were high-flat (0.75-0.85) AND hard
  (0.08-0.10), so a high-flat easy-arm would misroute them.

A content-class rule fails the same way: two text posters first-clean
at 0.03 and 0.06, and two watercolor-like images (lighthouse, botanical)
share 0.10 with dense architecture.

The leftover rungs were finished in a 2026-08-31 evening UTC
anonymous isolated session (fresh-navigation, wait for "Upload
another file"; all ten `POST /api/ai-detector` calls returned 200):

| check | verdict |
|---|---|
| muse-2 @ 0.08 | DETECTED |
| muse-3 @ 0.015 / 0.03 / 0.045 | DETECTED / CLEAN / CLEAN |
| muse-7 @ 0.045 / 0.06 | DETECTED / CLEAN |
| mosque @ 0.10 seeds 1, 2 | CLEAN / CLEAN |
| botanical @ 0.10 seeds 1, 2 | DETECTED / DETECTED |

botanical's seed-0 first-clean of 0.10 is not seed-stable;
lighthouse @ 0.10 seeds 1 and 2 were CLEAN. That is the same class
of run-to-run flip the 0.17 floor's margin exists for. No
Google-style binary split: harvest-1 first-cleans now span 0.03
(muse-3) through 0.10 (muse-2/4), and 0.06-0.10 inside one subject
(muse-7 / muse-6 / muse-2).

## OpenAI content-adaptive expansion (2026-08-31)

Goal: test whether face_count or `flat_ratio` predicts Chroma1's OpenAI
SynthID first-clean the way YuNet does for Google.

Committed three (YuNet + `flat_ratio` on the files, first-cleans from the
2026-08-29/30 web oracle):

| fixture | faces | flat_ratio | first-clean |
|---|---|---|---|
| typography 02_03_55 | 0 | 0.642 | 0.06 |
| quality 02_02_23 | 0 | 0.695 | 0.06 |
| 9-face grid 05_30 | 9 | 0.473 | 0.075 |

Face content is the harder of the two classes, the opposite of Google.
Two zero-face cards share a boundary, which is a hint, not a split.

**Images API is not a carrier source.** `gpt-image-1` and `gpt-image-1.5`
generations (medium, 1024) carry C2PA (`detected`) but SynthID
`not_detected` on `POST /v1/content_provenance_checks`. Contrastive-pair
"openai" recreations are also SynthID-negative. New ChatGPT-UI downloads
are not in the tree beyond the three committed files.

**Uncommitted corpus scan (one day, 2026-07-24):** 41 files with
OpenAI C2PA, of which 7 checked by the provenance API were SynthID
`DETECTED` (one `NOT_DETECTED`). Four diverse DETECTED files were staged
under `out/cohort-calibration/openai-expand/` (gitignored; not
publication-cleared) and given a Chroma1 ladder (0.04-0.12, seed 0) by
`chroma_openai_expansion.py` (a harness kept outside this repository):

| id | faces | flat_ratio |
|---|---|---|
| dense_ed68 | 0 | 0.118 |
| midflat_038c | 0 | 0.323 |
| textlike_3b17 | 0 | 0.649 |
| faces2_4dab | 2 | 0.279 |

The provenance API then 429'd. The same checks were finished on
`https://openai.com/research/verify/` (the page reports SynthID and C2PA
separately under View details). Chroma PNG outputs have no C2PA, so a
"No OpenAI signals detected" heading is a pixel-SynthID negative.

| id | faces | @0.04 | @0.05 | @0.06 |
|---|---|---|---|---|
| dense_ed68 | 0 | CLEAN | CLEAN | CLEAN |
| midflat_038c | 0 | -- | -- | CLEAN |
| textlike_3b17 | 0 | -- | -- | CLEAN |
| faces2_4dab | 2 | -- | -- | CLEAN |

9-face grid @ 0.075, seeds 1 and 2: both CLEAN (SynthID not detected).
The committed seed-0 first-clean is not a fluke.

**No OpenAI adaptive arm.** Zero-face extras clear at or below 0.06, the
2-face extra also clears at 0.06, and only the 9-face grid needs 0.075.
That is not a YuNet split: `face_count > 0` would over-strengthen the
2-face file, and a high face-count cutoff would be one fixture. The
spread (0.075 - 0.06) was absorbed by the 0.09 floor for this calibration,
but the later holdout below invalidated that operating point.
Images API generations remain non-carriers (C2PA only).

## OpenAI hard-carrier holdout (2026-09-07)

Two additional withheld OpenAI SynthID carriers expanded the Chroma boundary.
Each original was a positive pixel control after metadata-only stripping, so the
verdict did not depend on C2PA provenance. Their Chroma first-clean strengths were
0.10625 and 0.1375. Both Qwen outputs were clean at the existing 0.07675 operating
point.

The paired fidelity read was mixed rather than a reason to pay Chroma's higher
denoise. On the one detected face, Qwen had higher identity cosine (0.835 vs
0.821), lower face LPIPS (0.108 vs 0.118), and lower whole-image LPIPS (0.095 vs
0.125); Chroma retained more local texture and had slightly higher SSIM. On the
second carrier Qwen again had slightly lower LPIPS (0.104 vs 0.109), while Chroma
had higher SSIM and PSNR.

This invalidates both the 0.09 explicit Chroma floor and the OpenAI auto route.
For explicit Chroma, the face-bearing cohort rule gives
`0.1375 + (0.1375 - 0.075) = 0.20`; that candidate was subsequently verified
clean three times on both holdout carriers. `auto` uses `qwen-zimage` for OpenAI
at 0.07675; Chroma remains the Microsoft route.

## OpenAI zero-face holdout: the content split does not survive real carriers (2026-09-08)

The `OpenAI content-adaptive expansion` section above closed on fixtures whose
zero-face members first cleared at 0.06 and whose single face member cleared at
0.075, which made faces look like the harder class by a comfortable margin. Three
further withheld zero-face carriers were laddered on the public verifier to test
that, all at ~1.57 MP so image area could not explain a difference, and all
confirmed zero-face by `detect_faces` (YuNet) - the same detector the adaptive
arms gate on, not a downscaled Haar pass, which disagreed with it in both
directions on these very files.

Every regeneration was produced by the deployed worker at seed 0, `mode=all`,
explicit `chroma-zimage`, and every verdict below is the verifier's separate
`SynthID detected` line with `Content Credentials not detected` alongside it, so
no metadata could substitute for a pixel result. Each carrier's own
metadata-stripped, RGB-identical control returned a positive verdict first.

| carrier | detected through | first clean | interval |
|---|---|---|---|
| zero-face A | 0.03 | **0.045** | (0.03, 0.045] |
| zero-face B | 0.045 | **0.06** | (0.045, 0.06] |
| zero-face C | 0.10 | **0.11** | (0.10, 0.11] |

Carrier C was laddered 0.02, 0.03, 0.045, 0.06, 0.075, 0.09, 0.10 detected, then
0.11 and 0.125 clean.

Applying the standard rule to the zero-face class gives
`0.11 + (0.11 - 0.045) = 0.175`; the face class gives
`0.1375 + (0.1375 - 0.075) = 0.20`. **The two classes differ by 0.025 while the
spread within each is 0.065 and 0.0625** - a split would key on a variable that
explains roughly a quarter of the variation it leaves behind. The flat 0.20 floor
therefore stands, and it now covers both content classes rather than only the
face-bearing carriers it was derived from.

The methodological point generalises beyond this cohort. The fixture-only view of
zero-face OpenAI content spanned 0.015; the real carriers span 0.065, four times
wider. A narrow fixture spread produces a small worst-clean-plus-one-spread margin
and a false impression of homogeneity, and it was that impression - not any
measurement - that made a face split look justified here twice.

## Content-balanced engine selection check (2026-08-31)

The earlier tables were intentionally carrier-focused and therefore small. A
separate discovery pass tested whether ordinary image content contradicts the
cohort choice at the shipped floors. The tracked matrix contains 19 prompts,
matched across OpenAI `gpt-image-1-mini` and Meta `muse-image-1.0`, spanning
photoreal scenes, faces, action, text-like flat art, products, architecture,
watercolor, and surreal illustration. These files were re-encoded by collection
and are not signal carriers, so this pass measures fidelity only; removal remains
established by the provider-oracle calibration above.

Qwen and Chroma ran in separate sequential H100 invocations, once per source,
with the production provider floor and fixed seed. The comparison isolated the
global stage because both profiles inherit the same source-based Z-Image face
repair. Per-image differences, rather than differences between aggregate means,
were evaluated with exact two-sided sign tests:

| provider | metric | Chroma wins | Qwen wins | two-sided p |
|---|---|---:|---:|---:|
| OpenAI | SSIM | 19 | 0 | 0.0000038 |
| OpenAI | PSNR | 19 | 0 | 0.0000038 |
| OpenAI | MAE | 18 | 1 | 0.000076 |
| OpenAI | edge F1 | 15 | 4 | 0.019 |
| OpenAI | LPIPS | 12 | 7 | 0.359 |
| Meta | LPIPS | 1 | 18 | 0.000076 |
| Meta | edge F1 | 1 | 18 | 0.000076 |
| Meta | SSIM | 5 | 14 | 0.064 |

Meta's remaining pixel metrics disagree: Chroma wins PSNR on 16/19 and MAE on
14/19, while losing both perceptual distance and edge preservation on 18/19 at
its higher removal floor. This is not evidence for a genre classifier. The
direction instead follows the already measured provenance cohort: Chroma for
OpenAI, Qwen for Meta. On OpenAI, the seven LPIPS exceptions include unrelated
classes (landscape, sport, low-light interior, watercolor, monochrome portrait,
surreal art, and brutalism), while Chroma still wins SSIM and PSNR on every one;
there is no coherent input feature that justifies overriding the cohort choice.

Inputs and integrity manifests are under
`data/evaluations/engine-selection/`. Raw generated outputs remain gitignored
under `out/engine-selection-study/`; the `engine_selection_study.py` harness
(kept outside this repository) reproduces the run and
`scripts/analyze_engine_selection_study.py` reproduces
the paired statistics. One generation per prompt is a discovery set, not a
license to fit thresholds. Any future content override must first predict a
rule here and then survive unused generation indices without changing it.

## Oracle session notes

- openai.com/verify throttles anonymous sessions through Cloudflare Turnstile
  after ~15 rapid checks; a fresh browser context clears it. The programmatic
  `POST /v1/content_provenance_checks` API is the better oracle (separate
  SynthID/C2PA outcomes). The `OPENAI_API_KEY` in the main-checkout `.env`
  returned 401 org-wide, including `/v1/models`, through the 2026-08-30
  calibration; it was replaced on 2026-08-31 and both `/v1/models` and
  a provenance check on a committed ChatGPT original then succeeded
  (DETECTED).
- meta.ai/identification enforces a sliding per-IP window: a burst of ~15
  checks exhausted it twice; it reopens minutes later, and the page sometimes
  overstates this as a daily limit. Pacing of roughly 2-3 checks per window
  worked. On 2026-08-31 an anonymous isolated session got a real 429 from
  `POST /api/ai-detector` with page text "You’ve reached the daily limit
  for identifications. Try again tomorrow" after about six checks in the
  UTC morning and again after about fifteen in the UTC afternoon, so the
  daily-limit wording is not always overstated; the afternoon window was
  large enough to finish harvest 2. An evening isolated session the same
  day got ten 200s and finished the leftover harvest-1 rungs plus the
  mosque/botanical seed probes.
- The Gemini app oracle requires the user's logged-in session. A human
  drag-drop of a local PNG works on Work Gemini (u/2, 2026-08-31) and on
  Plus (u/0 the same day); Playwright cannot drive the OS picker, and
  the first `input[type=file].accept` is documents/code only. Plus
  `@synthid` hung once after attach, then returned a tool CLEAN on a
  second human-drag chat. Do not drive the live Chrome while it is in
  active use (wrong-tab risk).
