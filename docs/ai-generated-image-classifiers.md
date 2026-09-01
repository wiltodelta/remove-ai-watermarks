# AI-generated image classifiers (research)

> Research archive for metadata-free `ai_generated` versus
> `not_ai_generated` classifiers. These are not SynthID detectors and are not
 `identify` verdicts. The 2026-08-31 freeze is consumed only by explicit
> `classify` / `classify_pixels`. Current behavior:
> [photo pixel classification](photo-classify.md),
> [supported signals](supported-signals.md),
> and [known limitations](known-limitations.md).
>
> Sister pages: [SynthID source classifiers](synthid-classifiers.md),
> [SynthID local detector](synthid-detector-research.md),
> [SynthID mark removal](synthid-removal-research.md), and
> [mechanism reference](synthid.md).

A classifier is reliable only when its name matches its errors, photographs
are the first negative, Firefly and PixelBin are in the test, and a watermark
claim uses an independent oracle. CLIP content embeddings and the 124-d
origin-locked residual bank are different features for different jobs.

## Frozen result, 2026-08-31

The freeze is **AI versus camera**, not a universal detector, and it is not
ired into `identify`. The library consumes it only through explicit
`classify_pixels` / CLI `classify`. Detector: CLIP-L-ft ridge AND freeze MLP.
Provider: 124-d focal heads, only after DEFINITELY. Library guide:
[photo-classify.md](photo-classify.md). Hub card:
[photo-classify-hf/README.md](photo-classify-hf/README.md).

| Cell | Value |
| --- | ---: |
| Detector DEFINITELY, AI-test | 92.4% (n=1,847) |
| Detector DEFINITELY, fresh | 1.3% (n=3,000) |
| Detector DEFINITELY, Kodak | 0/24 |
| Detector DEFINITELY, FLUX | 83.0% (n=300) |
| Provider OpenAI / Google / TC260 | 90.8% / 90.9% / 78.6% |
| Provider Meta hold-out v3 | 89.4% (177/198) |

Out of contract: receipts and other documents, UI, digital art without the
illustration veto, and ungated memes. A photorealistic AI receipt can score
as human; a real CORD thermal receipt can score as AI. Catalog 32,690 files,
two CPU retrains byte-identical. Weights stay outside git
(`data/research/reports/both-models-2026-08-31/`). Details below.

## Research task hierarchy

The primary classifier task is metadata-free AI-generation detection: given an
arbitrary image, decide `ai_generated` versus `not_ai_generated` from pixels.
The target is open-world transfer to generators absent from training, with a
very low false-positive rate across real photographs and other non-AI imagery
such as scans, product cutouts, conventional CGI, and digital graphics.

OpenAI/Gemini source finding is a narrower, separately documented
[SynthID-adjacent task](synthid-classifiers.md). It asks whether a file resembles
a current OpenAI or Google generation pipeline and otherwise abstains. It does
not replace the general AI-generation detector: a precise provider finder can
miss most AI images, and a general detector need not know which provider
produced a positive. Neither task is a SynthID payload decoder.

## Partial result: Model 1, AI versus camera

Finetuned CLIP-L (`openai/clip-vit-large-patch14`), last two vision blocks,
224 letterbox, JPEG and mild crop, linear ridge. Train 5,221 AI plus 6,129
photos. Locked Open Images fresh never enters train. Operating point: 1%
FPR on disjoint `photo_dev_oi`.

| Cell | Value |
| --- | --- |
| Kodak | 0/24 |
| Open Images fresh FPR | 1.7% (n=3,000) |
| Exact-1024 Open Images FPR | 6% |
| AI-test TPR | 93.0% (n=1,905) |
| OpenAI | 93.2% |
| Gemini | 90.5% |
| Firefly | 94.0% |
| xAI | 96.1% |
| FLUX hold | 92.7% |

51 fresh false positives are mostly graphics, CGI, product cutouts, and
scans, not Gemini. Nobody in the sweep hit both ≤1% fresh FPR and ≥90%
TPR. This is the strongest result toward the general task, but its negative
contract is still AI-versus-camera rather than AI-versus-all-non-AI imagery.
The graphics/CGI errors therefore keep the general task open. This is not
SynthID, and it is not in `identify`.

Artifacts: `.local-eval/synthid/ai-photo-2026-08-22/`
(`comparison.json`, `probe-report-clip-l-ft.json`,
`probe-weights-clip-l-ft.npz`). Date cutoff 2026-07-23, seed 20260822.

### Rejected Model 1 variants

Same splits and `photo_dev_oi` 1% cut.

| Variant | Fresh FPR | Kodak | 1024 FPR | AI TPR | FLUX hold |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLIP-L v2 | 0.017 | 0/24 | 0.04 | 0.877 | n/a |
| CLIP-L + FLUX extra | 0.016 | 0/24 | 0.05 | 0.861 | 0.707 |
| CLIP-H + FLUX extra | 0.014 | 0/24 | 0.02 | 0.812 | 0.913 |
| CLIP-L last-2-blocks finetune | 0.017 | 0/24 | 0.06 | **0.930** | **0.927** |
| DINOv2-giant 256 | 0.023 | 0/24 | 0.04 | 0.606 | 0.293 |

CLIP-H is the photo-FPR specialist (1.4% fresh, 2% at 1024) at 81% TPR and
is not the result. DINOv2-giant at 256 px is not usable.

v1 (CLIP-L, no Open Images in train) at a COCO-looking 0.5% cut accepted
13% of Open Images. Domain shift, not the 124 residual bank. v2 added
1,000 disjoint Open Images reserve photos to train and 500 as
`photo_dev_oi`; locked fresh stayed 1.7% FPR at 87.7% TPR before
finetune.

The 124-d residual bank is the wrong feature for "AI or not". At a
Kodak-safe cut it catches 60% Firefly and misses FLUX, NovelAI, Reve, and
most of TC260 and xAI. Do not train another ridge on that representation
for an AI-or-not claim.

Open, if this head is ever considered for a product cut: a graphics/CGI
abstain and time/device-disjoint modern camera coverage. CLIP treating
non-camera imagery as generation is one known error, not Gemini contamination.

### Frozen public-checkpoint transfer, 2026-08-24

A no-training sweep put the official
[`Community Forensics`](https://github.com/JeongsooP/Community-Forensics) and
[`SPAI`](https://github.com/mever-team/spai) checkpoints on the same public
rows and the same operating rule as Model 1. Each threshold is the strict 99th
percentile of the 500-image `photo_dev_oi` split; no AI or evaluation negative
sets tune it. The SPAI core runs stop after all 2,405 AI rows because the model
is already dominated there; they do not supply a fresh-photo FPR.

| Model | AI test | AI extra | FLUX hold | Open Images fresh |
| --- | ---: | ---: | ---: | ---: |
| Model 1, CLIP-L-ft | 93.0% | 92.5% | 92.7% | 1.7% |
| Community Forensics 384 | 34.6% | 12.0% | 23.0% | 1.0% |
| SPAI, longest edge 512 | 2.2% | 3.5% | 0.7% | not run |
| SPAI, longest edge 1024 | 6.5% | 9.5% | 10.0% | not run |

Community Forensics finds 33 of Model 1's 170 misses across the 2,405 AI
rows. On 4,133 public evaluation photographs, however, it adds 37 errors not
made by Model 1. A calibration-only rank-max fusion reduces AI-test recall to
91.9%, AI-extra recall to 88%, and FLUX recall to 86%, while fresh-photo FPR
rises to 1.73%. A literal OR at the two original thresholds doubles calibration
FPR to 2% because their five errors do not overlap. The checkpoint is an
auxiliary representation, not a better detector or a valid OR branch.

SPAI at 1024 recovers only 11 Model 1 misses. Its predeclared rank-max fusion
reduces AI-test recall to 90.2% and FLUX recall to 85.3% at the same 1%
calibration FPR; its literal OR also doubles calibration FPR to 2%. The
300-image FLUX cell is exactly 1024 on its longest edge, so this failure cannot
be assigned to downscaling in that cell. The 512/1024 ablation does show
resolution sensitivity, but no useful low-FPR hybrid.

[`B-Free`](https://github.com/grip-unina/B-Free) remains unmeasured: its sole
official checkpoint host was unreachable over HTTP and HTTPS, and no verified
mirror was found. Its license also limits use to informational and nonprofit
purposes and expressly prohibits industrial or profit-oriented use. Its useful
result for this project is therefore the bias-reduction training paradigm, not
a checkpoint dependency.

No public checkpoint replaces Model 1 or safely repairs it. The next model
must change the negative contract: hash-grouped, time/device-disjoint modern
computational photography plus conventional CGI, graphics, scans, and product
cutouts. Another generic detector trained against a narrow `real` corpus is
not a new signal.

Local reproducibility artifacts:
`.local-eval/synthid/ai-photo-2026-08-22/frozen-ai-detector-sweep-2026-08-24/`.

### General AI-classifier GitHub sweep, 2026-08-25

A separate search targeted pixel-based `ai_generated` versus
`not_ai_generated` classifiers, not SynthID repositories. Twelve recorded
GitHub GraphQL searches returned 2,006 unique public non-fork repositories.
The broadest four searches were capped at 500 collected results, so this is a
bounded reproducible survey, not a claim that GitHub search can enumerate every
repository. Five current catalogs and benchmarks contributed 110 references;
106 resolved to 105 unique live repositories. Curated references plus
high-signal search matches produced 332 candidates, of which 328 resolved for
README, license, weight, and inference review.

The exact public, non-fork GitHub query set was:

```text
"AI-generated image detection" in:name,description,readme fork:false
"AI generated image detector" in:name,description,readme fork:false
"AI image detector" in:name,description,readme fork:false
"AIGC image detection" in:name,description,readme fork:false
"AIGC detector" image in:name,description,readme fork:false
"synthetic image detection" in:name,description,readme fork:false
"synthetic image detector" in:name,description,readme fork:false
"generated image detection" diffusion in:name,description,readme fork:false
topic:ai-generated-image-detection fork:false
topic:ai-image-detection fork:false
topic:aigc-detection fork:false
topic:synthetic-image-detection fork:false
```

The five catalog/benchmark inputs were
[`AIGCDetectBenchmark`](https://github.com/Ekko-zn/AIGCDetectBenchmark),
[`Awesome-AIGC-Image-Video-Detection`](https://github.com/ant-research/Awesome-AIGC-Image-Video-Detection),
[`Awesome-AIGC-Detection`](https://github.com/Daisy-Zhang/Awesome-AIGC-Detection),
[`Awesome-AIGC-Image-Detection`](https://github.com/graydove/Awesome-AIGC-Image-Detection),
and
[`Awesome-AI-generated-Image-Detection`](https://github.com/nxZhai/Awesome-AI-generated-Image-Detection).

The filter required pixel inference, an available checkpoint, reproducible
preprocessing, a license compatible with possible product use, and a signal or
training contract that differs materially from already rejected models. It
removed metadata/API wrappers, SynthID-only tools, face/video-only deepfake
systems, datasets and leaderboards, UI-only repositories, classroom CIFAKE
models, noncommercial checkpoints, and repositories without runnable weights.

The most relevant survivors are:

| Model | Status | Why it matters |
| --- | --- | --- |
| [Dual Data Alignment](https://github.com/roy-ch/Dual-Data-Alignment) | Apache-2.0, official 1.26 GB checkpoint, measured partially | DINOv2-L LoRA with paired real/reconstruction JPEG and frequency alignment; best new training contract. |
| [PGC](https://github.com/xiaoyu6868/PGC) | Apache-2.0, SD1.4 measured fully and joint measured on AI-test | DINOv2-L peak-guided calibration exposes a strong OpenAI signal, but it confounds Kodak scans and does not safely fuse with Model 1. |
| [DGS-Net](https://github.com/HorizonTEL/DGS-Net) | Apache-2.0, stage-2 checkpoint measured partially | Distillation-guided gradient surgery is reproducible, but the frozen checkpoint is weak and adds independent photo errors. |
| [FerretNet](https://github.com/xigua7105/FerretNet) | Apache-2.0, weights available, lower priority | Efficient local-pixel artifact branch, but trained on four ProGAN classes. |
| [OmniAID](https://github.com/yunncheng/OmniAID) | Modern 3.24 GB checkpoints; repository has no license file | Mirage-Train semantic/artifact experts are promising, but the README's MIT badge is not a license grant. |
| [SDAIE](https://github.com/Ekko-zn/SDAIE) | Weights available; no license | Camera/EXIF-supervised and real-only training are relevant ideas; inference is pixel-based, but product use is unresolved. |
| [AIDE](https://github.com/shilinyan99/AIDE) and [CO-SPY](https://github.com/Megum1/CO-SPY) | MIT, weights available, lower priority | Reproducible hybrid signals, but official checkpoints retain ProGAN or SD1.4-era negative contracts. |

[Effort](https://github.com/YZY-stack/Effort-AIGI-Detection),
[Forensic Self-Descriptions](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25),
and B-Free are research-only or noncommercial. UniGenDet is MIT but its
published checkpoint is about 59 GB. OpenSDI and SAD-Bridge have no detected
license. REM describes a relevant real-centric method, but its code and weights
are still pending. [FatFormer](https://github.com/Michel-liu/FatFormer) and
[SSP-AI-Generated-Image-Detection](https://github.com/bcmi/SSP-AI-Generated-Image-Detection)
have runnable, product-compatible releases, but their four-class ProGAN and
per-generator GenImage contracts are narrower than the measured candidates.
[RIGID](https://github.com/IBM/RIGID) is an Apache-2.0 training-free
reference-comparison idea rather than a detector with a frozen checkpoint to
transfer.

#### Our frozen-transfer protocol

The results below are this project's measurements, not accuracy copied from the
upstream papers or model cards. The public portion of the frozen manifest
contains 7,038 rows:

| Role | Public cells | Rows |
| --- | --- | ---: |
| Calibration negative | `photo_dev_oi` | 500 |
| AI evaluation | `ai_test`, `ai_eval_only`, `flux_hf_hold` | 2,405 |
| Non-AI evaluation | fresh and 1024-pixel Open Images, COCO holdout, Kodak, Picsum | 4,133 |

The manifest removes exact SHA-256 duplicates before scoring. Every candidate
uses its official image branch and published evaluation preprocessing unless a
deviation is named below. No checkpoint is retrained and no AI or evaluation
negative row selects a threshold. Scores are oriented so that larger means more
likely AI-generated; the decision is the strict inequality above the empirical
99th percentile of the 500 calibration negatives.

For a two-model hybrid, each raw score is converted to its empirical percentile
against that model's 500 calibration scores. `rank_max` is the maximum of those
two percentiles and `rank_mean` is their mean. The hybrid threshold is then
recalibrated on those same 500 negatives. Literal OR results retain each model's
independently calibrated threshold. Per-item changes are reported as paired
improvements and regressions, with a one-sided exact sign test when the direction
is used as evidence. A pilot stops a larger run only when it is already too far
below the recall/FPR gate to change the decision.

#### Exact artifacts and adapters used by our runs

- Community Forensics: repository
  `ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4`; model snapshot
  `6076002bf0d9dd37537f965ee2f06f826c333b61`, model SHA-256
  `b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387`;
  processor snapshot `3540a3f0d688f8bf492a8aed48613b891f88047e`. The adapter uses
  official `CommForImageProcessor` test mode at 384 pixels and the raw output
  logit. Its calibrated threshold is `-0.7597203404`.
- SPAI: repository `8ff7b3b6779b4fcb43cf313471d9cb1c62d129a4`, OpenAI CLIP
  revision `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`, checkpoint SHA-256
  `24159f27d7c8c2cd0cb6c4019189eb89ad0874a0d9d15f8dc9afd39ca9648a55`.
  The 512/1024 variants cap the longest edge with bicubic resampling before
  SPAI's RGB, minimum-size-padding transform; this implements the documented
  `--resize-to` intent because the upstream `SmallestMaxSize` path can enlarge
  the longest edge. Thresholds are `17.0864165878` and `14.0359167767`. The
  upstream checkpoint embeds a YACS object and could not use safe
  `weights_only=True` loading; this is another reason it remains research-only.
- Dual Data Alignment: repository
  `8b9c06e75e63f4688bc25ac43a7e3412878cf67f`, checkpoint SHA-256
  `b27a31d39374803ddeff02bfabb2be76e190b04300490cddfafb24f683f37e3e`.
  The adapter is the official DINOv2-L/14 rank-8 LoRA image path with
  `CenterCrop(336)`, tensor conversion, CLIP normalization, and sigmoid binary
  score. The threshold is `0.9587997139`.
- PGC: repository `0c9b10f3b89964b804ad6097e61709f74e827fbf`, DINOv2-L
  configuration `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`. The SD1.4
  checkpoint SHA-256 is
  `275c35741834345191fd1be4e4c26075512840d4cf0d2515b9c86094b7bf003e`;
  the ProGAN+SD1.4 checkpoint SHA-256 is
  `234ca835f4219892acacaea6f2bd15ac698e4491c91b2125f109901fdb56ece6`.
  Both load with `weights_only=True` and a strict state-dict match. The adapter
  preserves official `PadCenterCrop(224)`, tensor quantization-residual append,
  DINO normalization, and sigmoid fused logit. Thresholds are `0.4983334646`
  and `0.7142269963`.
- DGS-Net: repository `e7e0799b61014765c11c837506b772e7504198e3`, stage-2
  checkpoint SHA-256
  `89b96a586a6b9f2e5626b03aa27e8063bbe6aaae00f4d0cf97fd3ced2379c216`.
  Safe `weights_only=True`, memory-mapped loading extracts a strict inference
  subtree: CLIP ViT-L/14, rank-6 vision LoRA, and image head. The training-only
  frozen teacher and text head are not read by the official image-only forward,
  so stage 1 is unnecessary for evaluation. Preprocessing selects the 24 lowest
  and 25 highest spectral-entropy 32-pixel patches, shuffles them with the
  official seed 100, stitches a 7x7 image, and applies ImageNet normalization.
  The threshold is `0.9998653293`.
- SAFE, RINE, and Nonescape Mini respectively pin repository revisions
  `4e998724651b227def64f5be0cd60c0aa1552c35`,
  `9b7fd5857cc205d0412be6aeee0d7611b95bd620`, and
  `52619d5c96ab83f018d9e879d4be14d847ccb15d`; checkpoint SHA-256 values are
  `b3f5ecfb46a154ed553aaaf4bf3ba59182310726ddb0cbb1fe42bd0e22d2f20e`,
  `6535ff6ecfa88d33c081e561c2cb2dcb594a1006b050eddb72a72ce005ed2ca1`,
  and `7a0d0740c813ce199bc32ed16a5f4f4915895c4c9fdee0a98bdbeedd4f3631fd`.
  Their adapters preserve, respectively, the official 256-pixel crop plus
  bior1.3 DWT branch, 224-pixel OpenAI-CLIP path, and EfficientNet-v2-s
  256-resize/224-crop path. Thresholds are `0.9303810883`, `0.9529916000`, and
  `0.9896544343`.

Seven additional checkpoints were put on that frozen rule. All values below
are public cells.

| Model | AI test | Gemini | OpenAI | FLUX hold | Fresh Open Images |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model 1, CLIP-L-ft | 93.0% | 90.5% | 93.3% | 92.7% | 1.7%, n=3,000 |
| DDA official | 48.2% | 68.8% | 21.8% | not run | 0.7%, n=1,000 |
| PGC SD1.4 official | 44.6% | 5.5% | 96.5% | 0.7% | 0.7%, n=3,000 |
| PGC ProGAN+SD1.4 official | 29.3% | 8.3% | 56.3% | not run | not run |
| DGS-Net stage 2, seed 100 | 21.2% | 1.5% | 55.5% | not run | 1.9%, n=1,000 |
| SAFE official | 21.6% | 1.3% | 57.8% | 0.0% | 0.6%, n=3,000 |
| Nonescape Mini v0 | 13.3% | 13.5% | 7.3% | 37.3% | 1.8%, n=3,000 |
| RINE four-class | 10.7% | 5.5% | 16.8% | 1.0% | 1.0%, n=3,000 |

DDA supplies material complementary recall. It finds 52
of Model 1's 133 misses on the 1,905-image AI test. Its seven errors on the
independent 1,000-image negative slice do not overlap Model 1's 18, however. A
literal OR reaches 95.7% AI recall and 2.5% FPR there; the predeclared
calibration-rank maximum reaches only 93.4% recall and 1.6% FPR. The checkpoint
is therefore not a safe drop-in fusion.

PGC SD1.4 finds 63 of those 133 misses, while the joint checkpoint is weaker
on the same AI rows. A calibration-only rank maximum with Model 1 initially
looked like a clean improvement: AI-test recall rose from 93.0% to 94.1%
(58 paired improvements, 37 regressions; one-sided sign test `p=0.0198`) and
fresh Open Images FPR fell from 1.67% to 1.13% (22 paired improvements, six
regressions; `p=0.00186`). The full negative matrix rejects that conclusion.
The fusion calls all 24 Kodak images AI-generated, versus 0/24 for Model 1,
and its aggregate public-negative errors rise from 59/4,133 to 63/4,133. It
also reduces AI-extra recall from 92.5% to 87.5% and FLUX hold recall from
92.7% to 85.3%.

Our single-pass PGC component ablation localized but did not remove the
confound:

| PGC score | AI test | OpenAI | Fresh Open Images FPR | Kodak errors |
| --- | ---: | ---: | ---: | ---: |
| Global only | 34.8% | 82.0% | 1.8% | 17/24 |
| Global + RGB peak | 6.8% | 5.5% | 0.37% | 0/24 |
| Global + residual peak | 41.9% | 96.5% | 1.5% | 24/24 |
| Official global + RGB + residual | 44.6% | 96.5% | 0.67% | 24/24 |

The RGB peak suppresses the Kodak response, but it also removes the useful
OpenAI signal; the residual peak carries both. Calibration-rank conjunctions
of the RGB and residual components remove the Kodak errors, but either add
fresh-photo errors under their own 1% calibration cuts or fall below Model 1
recall after joint recalibration. A descriptive raw PGC threshold above `0.75`
would add 23 AI catches without a public-negative error, but it was selected
after inspecting test behavior and is therefore not validation. It is not an
accepted branch or product threshold. PGC remains an OpenAI-oriented research
feature, not a universal detector.

DGS-Net's official stage-2 image branch was reconstructed strictly from the
published checkpoint; the training-only frozen teacher and text head are not
read by the repository's image-only evaluation forward. Its official
spectral-entropy patch selection retains a random shuffle, so this measurement
pins the repository's seed 100. The checkpoint finds 37 Model 1 misses but adds
19 non-overlapping errors on the same 1,000 fresh negatives. A literal OR is
95.0% AI-test recall at 3.7% FPR; calibration-rank maximum is 91.0% recall at
1.6% FPR. Its 21.2% standalone recall is far enough below the gate that a full
corpus or multi-seed run is not warranted.

SAFE, RINE, and Nonescape Mini also fail as frozen replacements or fusions.
Their value is now bounded: SAFE supplies a wavelet/transformation branch, RINE
intermediate CLIP blocks, and Nonescape a cheap EfficientNet branch, but none
improves the low-FPR operating point.

The licensed frozen-checkpoint queue is exhausted at the useful priority level.
PGC's OpenAI/Kodak confound makes scans an explicit hard gate for subsequent
training. SDAIE's camera-supervised or real-only training remains an idea source
until a license exists.

Local search and scoring artifacts:
`.local-eval/github-ai-detector-sweep-2026-08-25/` and the frozen sweep directory
above.

### Own training and hard-negative ablations, 2026-08-25

The next campaign tested the training ideas rather than treating upstream
checkpoints as finished detectors. Every accepted threshold used only a named
negative calibration split. Per-item decisions were compared with Model 1, and
a prompt-disjoint EvalGEN cohort remained physically unextracted whenever a
candidate failed development or the old public matrix.

#### Paired reconstruction and hard negatives

DDA's useful ingredient is paired real/reconstruction training, not its
published frequency-mix formula: with its documented whole-image patch setting,
frequency mixing reduces algebraically to pixel mixing. A 256-pair pilot used a
public VAE reconstruction of training-only COCO, Picsum, and Open Images photos.
At 128 continuation steps, the ordinary control reached 93.75% AI-test recall,
2.67% fresh Open Images FPR, 94% FLUX recall, and 0/24 Kodak errors. Paired
reconstruction kept AI-test at 93.75% but moved FLUX to 91%, retained 2.67%
fresh FPR, and produced eight paired improvements against ten regressions.
Adding codec augmentation moved the same cells to 93%, 90%, and 3%, with six
improvements against 13 regressions. Neither arm improved Model 1.

Source-grouped UI screenshot negatives also failed to repair the public
contract. A higher-dose continuation reduced AI-test to 87.5%, raised fresh FPR
to 3%, added 1/24 Kodak error, and reduced FLUX to 82%. A lower dose reached 89%,
3%, 0/24, and 85%. Adding the same negative category only to the frozen ridge
head produced 52 regressions against 24 improvements on the public matrix.
Private source identities, paths, and local-only measurements are deliberately
absent from this public archive.

An official Open Images V7 training partition supplied a reproducible hard-photo
test. The first 6,000 lexicographic official S3 IDs were frozen before scoring,
content-deduplicated, and split into 4,000 mining, 1,000 untouched holdout, and
1,000 unused challenge images. Model 1 called 55/4,000 mining and 13/1,000
holdout images AI. Replacing its head after adding the top 128 hard negatives
kept holdout at 13/1,000, improved AI-test from 93.02% to 94.12%, but raised
fresh FPR from 1.67% to 2.13% and added 1/24 Kodak error. A predefined
calibration-rank mean with the hard head was only directional: AI-test 92.97%,
fresh FPR 1.53%, unchanged 92.67% FLUX, 0/24 Kodak, and six aggregate paired
improvements against two regressions. On the untouched Open Images train
holdout it changed 13 errors to ten, with four repairs and one new error.

The corresponding frozen rank-min challenge did not justify promotion. On 100
previously unscored EvalGEN Flux images it exactly matched Model 1 at 83%; on
1,000 unused Open Images train negatives it moved 1.0% to 0.9%, one repair and
no new error. The positive gate failed. That first 100-image dataset-viewer
slice covered only five lexicographically early prompt groups, heavily featuring
backpacks, bananas, and baseball bats, so 83% is a semantic stress result rather
than representative EvalGEN recall.

Semantic matching alone did not solve that content bias. A 5,221-pair frozen
CLIP linear head validated at AUC 0.996 but reached only 76% on the opened
EvalGEN slice. Two- and three-scale Model 1 views reached at most 85% there and
raised negative FPR to 1.3%; a 50% center crop fell to 57%. A paired-margin tower
continuation made the same decisions as its ordinary control. Its best arm made
two repairs against four regressions, while larger margins had no measured
separation to exploit.

#### Prompt-diverse EvalGEN continuation

The official [EvalGEN dataset](https://huggingface.co/datasets/Junwei-Xi/EvalGEN)
contains 553 aligned prompt groups and approximately 55,300 JPEG images from
Flux, GoT, Infinity, OmniGen, and NOVA. Before training, the five viewer-prefix
prompt groups were reserved as opened development, 300 disjoint prompt groups
were assigned to training, 100 to a future blind cohort, and 148 left unused.
One deterministic image per generator and training prompt produced 1,500 unique
training positives, 300 per generator. Their decoded-pixel hashes had zero
overlap with the 19,524 unique hashes in the base manifest. The 500-image blind
cohort was not extracted or scored during the campaign.

A 256-step continuation replaced half of each batch with an EvalGEN positive
and a text-nearest real training photo. Its neural head moved the opened
EvalGEN development slice from 88% for the byte-identical control to 95%, while
the unused 1,000-photo development FPR moved from 1.2% to 1.1%. The signal was
real, but promotion checks exposed forgetting:

| Candidate | Opened EvalGEN dev | Unused-photo dev FPR | AI-test | Fresh Open Images | FLUX | Kodak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Canonical ridge on expanded tower | 89% | 0.9% | not run | not run | not run | not run |
| Expanded neural head, stricter calibration | 92% | 0.7% | 91.13% | 1.57% | 92.0% | 0/24 |
| Model 1 + expanded rank maximum | 92% | 0.8% | 92.28% | 1.63% | 92.67% | 0/24 |
| Quarter-dose replay rank maximum | 91% | 0.7% | 92.34% | 1.60% | 90.0% | 0/24 |
| Soft-distilled quarter replay rank maximum | 90% | 0.8% | 92.07% | 1.40% | 90.67% | 0/24 |
| 75% base-to-expanded parameter interpolation | 91% | 0.8% | 92.28% | 1.73% | 91.67% | 0/24 |
| Calibration-budgeted rescue branch | 94% | 1.0% | 92.81% | 2.03% | 93.67% | 0/24 |
| Quadratic two-score stacker | 92% | 1.0% | 92.60% | 2.27% | 94.0% | 0/24 |
| Full-dose hard-negative neural head | 100% | 0.8% | 87.72% | 1.37% | 98.0% | 0/24 |
| Model 1 + full-dose hard-negative rank maximum | 91% | 0.6% | 90.34% | 1.17% | 92.0% | 0/24 |
| 75% hard-negative parameter interpolation | 90% | 0.6% | 90.60% | 1.20% | 91.33% | 0/24 |
| Quarter-dose hard-negative neural head | 97% | 0.8% | 88.50% | 1.43% | 97.0% | 0/24 |
| Model 1 + quarter-dose hard-negative rank maximum | 89% | 0.6% | not run | not run | not run | not run |
| Conditional Model 1 + quarter-hard branch | 97% | 1.0% | 93.33% | 1.93% | 96.33% | 0/24 |

The ridge gate failed on recall, so it never entered the old full matrix. The
full-dose neural head and rank fusion reduced fresh errors slightly, but paired
AI-test changes remained negative: the rank maximum repaired eight Model 1
misses and broke 22 prior hits. Reducing EvalGEN to one-quarter of each batch
did not remove the asymmetry. Logit-MSE distillation overconstrained the model
and collapsed the opened positive recall to 64%; bounded probability-BCE
distillation retained 95% before strict fusion but still failed the old core.
Interpolation at 25% and 50% failed development at 89%; 75% passed development
and then failed every core condition except Kodak.

The rescue cascade allocated a fixed five-error calibration budget between the
old and new heads. Its conservative selected allocation gave the new branch
only one error, yet the independent fresh-photo cell rose to 2.03%. Expanding
calibration to all 6,195 opened public negatives found no budget allocation that
simultaneously held aggregate FPR at or below 1%, preserved Model 1 AI-test,
AI-extra, and FLUX recall, and reached 90% on opened EvalGEN. A supervised
linear/quadratic stacker also failed: the quadratic model passed development at
92%/1.0%, then raised fresh FPR to 2.27% and reduced AI-test to 92.60% despite
raising FLUX to 94%.

Hard-negative replay isolated the competing objectives more sharply. A batch
with two old AI, two EvalGEN AI, two ordinary photos, one prompt-matched photo,
and one of the top 128 reproducible Open Images false positives produced perfect
opened EvalGEN recall and 0.8% development FPR. It also gave the campaign's best
fresh-photo rates, 1.37% for the neural head and 1.17% for its rank fusion, but
old AI-test recall collapsed to 87.72% and 90.34%. A 75% interpolation retained
only 90.60% AI-test recall. Thus hard negatives repair the negative contract,
but do not prevent positive-class forgetting.

A factorial follow-up changed only the positive replay ratio from two old plus
two EvalGEN images to three old plus one EvalGEN image per batch. Its direct
neural head reproduced 97/100 opened EvalGEN detections with 8/1,000 photo
errors under a frozen 99th-percentile calibration. On the old core it repaired
8 Model 1 AI-test misses but broke 94 prior hits: 1,686/1,905 AI-test images
(88.50%) remained positive. It improved fresh Open Images from 50/3,000 to
43/3,000, improved FLUX from 278/300 to 291/300 without a paired regression,
and kept Kodak at 0/24. The predeclared rank maximum was stricter on development,
89%/0.6%, and therefore never entered core. This rules out replay ratio alone as
the missing anti-forgetting mechanism.

A conditional two-branch rule then used a high quarter-hard threshold to add
new positives and a lower threshold only when Model 1 was already positive.
Its thresholds were selected from 13,280 admissible combinations on old
training AI and opened development, never core. It preserved 5,055/5,221 old
training AI against Model 1's 5,041, retained 97% opened EvalGEN recall at
1.0% development FPR, and restored the old core positive side: AI-test reached
1,778/1,905 (93.33%) with six repairs and no regressions, while FLUX reached
289/300 (96.33%) with 11 repairs and no regressions. The independent negative
side exposed calibration drift, however. Fresh Open Images rose from 50 to
58/3,000 errors: one repair against nine new errors, or 1.93% FPR. The route
separation fixes forgetting, but a single-source negative calibration does not
yet learn a portable veto.

Broadening that same two-score threshold search did not produce a candidate.
The frozen contract combined 4,171 negatives from eight public photo cells,
required aggregate FPR at or below 1%, untouched Open Images train at or below
1%, every source at or below 2%, preserved old training recall, and retained at
least 90% opened EvalGEN recall. No threshold pair satisfied all constraints,
so neither core nor blind was opened. A separate weighted-ridge veto was then
fit only on Model 1-positive `ai_train` embeddings and 256 hard photos from each
of the original photo training pool and official Open Images mining pool. None
of 37,875 penalty/threshold combinations passed the same development contract.
The best positive-preserving frontier was nevertheless narrow: it retained
95/100 EvalGEN and exactly matched Model 1 at 667/700 old AI development while
calling only 18/4,171 aggregate photos AI (0.43%). Six of those errors were in
the 100-image `open_images_1024` cell, a 6% source-specific rate; every other
cell ranged from zero to 0.75%. The next ablation is therefore negative dose,
not another threshold over the same fitted veto.

Increasing the veto fit to 512 and 1,024 hard negatives per source did not move
the `open_images_1024` failure: its best positive-preserving frontier remained
6/100 while aggregate photo errors were 17/4,171 and 20/4,171. Route attribution
showed that all six errors bypassed the veto through the unconditional new-AI
branch. Raising that branch's score threshold as high as possible while keeping
90/100 EvalGEN also retained all six. Applying the learned veto conjunctively to
both branches did not close the gap either. Across 37,875 combinations at each
hard-negative dose, the positive-preserving frontiers retained 91% EvalGEN and
95.29-96.0% old-AI development recall but left 6-7% `open_images_1024` FPR. The
negative-compliant frontiers reduced that cell to 2/100 and aggregate errors to
5/4,171, but collapsed EvalGEN to 26-39% and old-AI development to 43.57-53.43%.
The two learned CLIP representations therefore lack a separating direction for
this source. The next veto must introduce an independent pixel-forensic
representation rather than more data, thresholds, or routing over these scores.

A metadata-free native-lattice forensic veto then added 47 pixel statistics:
high-pass amplitude and correlation, CFA phase error, cross-channel residual
agreement, intensity-conditioned noise, JPEG block phase, clipping, and
Laplacian energy. It fit on deterministic 2,000-row samples from each old class
and evaluated 37,875 frozen combinations. No candidate passed. Its best
positive-preserving frontier reached 90% EvalGEN, 668/700 old-AI development,
and 32/4,171 aggregate photo errors, but still accepted the same 6/100
`open_images_1024` rows. The strict negative frontier reached 2/100 there only
by collapsing to 6% EvalGEN and 187/700 old AI. Native-512 coverage was weak on
the smaller COCO and general-photo cells, but it was 100% on EvalGEN, old-AI
development, and the failing exact-1024 cell, so missing features do not explain
the repeated six errors.

Pixel inspection showed that the six are a semantically diverse and valid
non-AI set: a scientific illustration, a logo, a studio object, a heavily
processed fashion image, an abstract long-exposure photograph, and an ordinary
flower-field photograph. This exposes a scope mistake in the provisional
negative contract: a universal detector must reject human-created graphics and
processed imagery as well as camera photographs. Camera-trace absence cannot be
the general veto. The next hard-negative mining pass must therefore rank public
negatives by the quarter-hard add score itself; the earlier top-128 replay was
ranked by Model 1 and did not target the branch now producing errors.

That aligned mining scored the same frozen 4,000-image official mining pool and
1,000-image holdout with the quarter-hard head. Its top 128 overlapped Model 1's
top 128 in only 78 rows (Jaccard 0.438), proving that the old mining missed 50
of the new branch's hardest examples. At the frozen neural threshold,
quarter-hard called 18/4,000 mining and 3/1,000 holdout images AI, versus Model
1's 55/4,000 and 13/1,000. Re-fitting the conjunctive linear veto on the
quarter-ranked negatives still failed at hard-negative doses 256, 512, and
1,024. The best hard-512 positive frontier retained 91% EvalGEN and 671/700 old
AI with 45/5,171 aggregate errors and 11/1,000 mining-holdout errors, but still
accepted 6/100 `open_images_1024`. Its negative-compliant frontier reduced those
cells to 1/1,000 and 2/100 only by falling to 29% EvalGEN and 302/700 old AI.
Aligned examples therefore need to change the add representation itself; a
linear veto over the original Model 1 embedding cannot use them sufficiently.

A second-stage representation update changed only the quarter-hard replay row:
the hard photo came from the top 128 quarter-hard mining scores instead of the
top 128 Model 1 scores. The 256-step self-hard arm retained the same 97/100
EvalGEN recall and 8/1,000 unused-photo errors in both pilot and formal neural
development. It did not change the source failure. At each head's independently
frozen 99th-percentile threshold, original quarter-hard and self-hard both
accepted 11/100 `open_images_1024` rows, with zero repairs, zero regressions, and
100/100 paired agreement. The self-hard arm therefore failed its source gate
before old core and blind. Selecting harder examples from the same broad class
is not equivalent to supplying the missing negative domains.

The first explicit negative-taxonomy audit then froze seven human-created
domains before scoring: historical camera photographs, drawings, prints,
manuscripts, paintings, studio-shot objects, and textiles. Its 700 Cleveland
Museum of Art images were CC0, content-hash unique, and unused by every fitted
model. Model 1 called 270/700 AI, including 60/100 drawings, 58/100 manuscripts,
and 46/100 textiles, while accepting 0/100 historical photographs. A second
source held 350 public-domain Art Institute of Chicago images, 50 per domain;
after a duplicate-hash guard rejected the first manifest and two repeated
images were replaced, Model 1 called 85/350 AI. The combined error was therefore
355/1,050 (33.81%), with 0/150 on the historical-camera control but 82/150 on
manuscripts and 75/150 on drawings. This is a real non-photo domain failure,
not a metadata observation or one museum's export artifact.

The direct quarter-hard representation reduced those errors to 163/1,050
(15.52%). Paired against Model 1, it repaired 201 rows and regressed on nine
(two-sided sign test `p=2.34e-48`), but its already measured 88.50% AI-test
recall still disqualifies it. Conditional routing again inherited the baseline
failure, at 357/1,050. A source-disjoint domain-veto experiment then fit only on
the Cleveland embeddings and selected its operating point only on AIC plus the
opened positive development cells. A class-balanced multiclass ridge reduced
the AIC route from 86/350 errors to 0/350 while retaining 97/100 opened EvalGEN
and 288/300 FLUX. The frozen core rejected it: AI-test became 1,771/1,905 versus
Model 1's 1,772 and the conditional route's 1,778, while fresh Open Images
remained 56/3,000 versus Model 1's 50 and the conditional route's 58. It also
left the exact-1024 cell at 8/100. The expert learned transferable museum-art
separation but not the modern logos, composites, product imagery, processed
photography, and ordinary-photo overlap represented by the public core.

No candidate reached the prompt-disjoint blind cohort. The useful result is a
bounded representation finding: diverse EvalGEN continuation adds a signal
that Model 1 lacks, but its current CLIP-L tower and two-score combinations
cannot add that signal without exchanging old generator recall or photo FPR.
Hard negatives now provide a stronger non-AI training contract, and conditional
routing provides the explicit anti-forgetting architecture. Broad calibration,
a dedicated linear veto, negative-dose scaling, conjunctive routing, a native
pixel-forensic veto, aligned mining, and self-hard continuation all failed on
the same source-specific overlap. The negative-taxonomy audit closes the
historical-art measurement gap but not the modern one. The next bounded campaign
needs source-disjoint training and validation cells specifically for modern
logos and graphic design, commercial product cutouts, composites, retouched
fashion imagery, abstract or long-exposure photography, and ordinary camera
photos. Another museum-art pool, generic photo pool, CLIP-space threshold,
binary veto, or replay-ratio change is not sufficient.

The modern-negative audit then filled that gap under the same frozen protocol,
with two new source-disjoint runs. The first drew 200 photographer-uploaded
photographs from the official Unsplash Lite dataset, 100 per cell for retouched
fashion and long-exposure photography, with the Unsplash License recorded per
row. The second drew 250 CC-licensed Flickr images through the Openverse API,
50 per cell for modern logo/graphic design, commercial product cutouts,
composites, and independent fashion and long-exposure replication cells, with
the exact CC license recorded per row. Both manifests enforce global
content-hash uniqueness plus perceptual near-duplicate rejection against every
picsum training cell, because picsum serves Unsplash photography; the Openverse
curation additionally excludes AI-marker, game-screenshot, and 3d-render
wording, so those domain labels are query-level rather than hand-verified
ground truth (46/50 logo-cell titles carry explicit logo or design wording).

Model 1 accepted 85/450 modern negatives, 18.89%. A contamination audit
then tightened the claim before it was trusted: a provenance-metadata scan of
all 450 rows with the project's own `identify` command flagged 0/450 (no C2PA,
SynthID, generator tags, or watermarks), and a date bound re-verified each row
against the public diffusion era, using the Unsplash submission date or the
Openverse `indexed_on` date. All 450 rows received a date; 421 predate
2022-08-01 and 29 post-date it, and the 29 were dropped. The correction was
material in exactly one cell: the Flickr fashion replication cell's 10 errors
became 2, because 8 of them sat in post-2022 uploads that are plausibly real
generations. On the date-clean 421 rows Model 1 accepted 69 (16.4%). The
failure concentrates in stylized graphics and processed photography:
logo/graphic design 18/47 (38.3%), retouched fashion 30/88 on the pre-2022
Unsplash cell (34.1%; the Flickr replication cell fell to 2/40, so fashion is
source-dependent rather than uniformly failing), and product cutouts 8/50
(16.0%, unchanged). Long exposure stayed comparatively clean at 11/146 (7.5%)
and composites at 0/50. Direct quarter-hard reduced the clean total to 47/421,
repairing 27 rows and regressing five (sign test `p=1.13e-04`), repeating the
museum pattern: it
helps every human-made domain while its measured 88.50% AI-test recall still
disqualifies it. Conditional routing inherited the baseline again at 86/450.
The two audits together bound the whole negative space: Model 1 fires on
stylized human graphics, logos, product cutouts, retouched fashion, and
historical art, while staying far cleaner on historical, composite, and
long-exposure photography and on ordinary camera photographs.

The measured cells then fed the same source-disjoint veto recipe the museum
audit used: pooled, per-domain, and multiclass ridge vetoes fit only on the
235 date-clean Openverse embeddings, with the threshold selected only on the
independent 186-image date-clean Unsplash run and applied to the frozen
conditional route. Of the searched candidates, none satisfied both the
negative cap (at most four dev errors) and the positive floor (95/100
EvalGEN, 287/300 FLUX); the full-run variant with contaminated rows reached
the same verdict over 5,265 candidates. The exchange curve shows why: at
positive compliance the best reachable dev error count is 41/186, meaning
zero of the route errors are repaired, while repairing roughly 39 of them
drops EvalGEN to 40/100 and FLUX to 162/300. Across the whole grid, every
repaired modern negative costs roughly 1.3 to 1.5 AI positives. The two
audits
therefore produced two different failure modes for the same recipe: museum art
is linearly separable from AI in this embedding and fails only on transfer,
while modern negatives are not separable at all. This closes the linear-veto
family over the frozen Model 1 CLIP-L embedding for the modern domains as well
and confirms the earlier aligned-mining finding: the add representation itself
must change, not the example selection or the veto geometry.

### Wild vendor-flagged AI, 2026-08-27

A free wild-AI positive cell came from the stock providers behind the private
veedma-blog image pipeline. Pixabay exposes a vendor-declared
`isAiGenerated` flag and the query `ai generated` surfaces that pool (3,287
results); 284 content-hash-unique rows were harvested (300 minus 16 lost to a
duplicate-download bug, since fixed, manifest repaired to on-disk truth). The
vendor flag is ground truth for AI-versus-human, not for the renderer, so the
cell is eval-only. A metadata scan found 0/284 provenance signals (the stock
CDN strips everything), making it a pure pixel task. Scored with frozen
classifiers that never saw stock-site content, this produced the first
source-disjoint wild-AI recall measurement: Model 1 accepted 183/284 (64.4%)
at its frozen 1%-FPR threshold, far below its 93.0% curated AI-test recall
(median score 0.43 above the 0.306 threshold, but a quarter of the cell scores
below 0.13). Wild AI images of unknown renderers, re-encoded by a stock
pipeline, are therefore a genuinely harder positive distribution, and curated
test recall does not bound deployment recall. The frozen provider cascade
under its photo-first margin called 255/284 `no_ai` with only 1 `openai`
and 0 `meta` calls, so unknown-renderer AI does not fabricate provider
attributions; under argmax it leans `google` (199) without evidence. The cell
joins the dataset as the first wild-AI eval row source. Artifacts:
`pixabay-ai-positive-2026-08-27/` (manifest, images,
classification-report.json).

A bias audit then rejected that cell's composition, though not its verdict:
147/284 rows (52%) came from a single contributor and every row from one
query, so the 64.4% recall was one author's dump plus a tail. Model 1 recall
split 63.3% on that contributor versus 65.7% on the rest, so difficulty was
homogeneous, but the sample was not. A stratified reharvest capped at 15 rows
per contributor over eight AI-marker queries produced 300 unique images from
180 contributors (top-10 share 33%) with the same sieves. On the balanced
cell the frozen numbers are: Model 1 recall 209/300 (69.7%, median score 0.482,
lowest quartile below 0.144), direct quarter-hard 189/300 repairing 22 Model 1
misses and losing 2 (the one place the quarter representation still helps),
conditional routing 210/300 adding one repair and nothing else, provider
cascade margin 252/300 `no_ai` with 1 `openai` call, and 0/300 metadata
signals. The wild-AI recall ceiling under Model 1 therefore settles near
65-70% rather than 93%, robust to contributor stratification; unknown-renderer
AI re-encoded by stock pipelines is the largest measured positive-side gap.
Artifacts: `pixabay-ai-stratified-2026-08-27/` (the eval cell the dataset
keeps; the first harvest is retained only as the audit trail of the bias
finding).

The stock-negative harvest from the same providers then tripled the modern
negative cells under identical sieves: 150 Pixabay product cutouts
(vendor `isAiGenerated=false`), 150 Pixabay fashion rows, and 152 Unsplash
API fashion rows date-bounded before 2022-08-01 (452 total, 0/452 metadata
signals, all content-hash unique). Frozen Model 1 false-positive rates on
them: product cutouts 40/150 (26.7%), Pixabay fashion 93/150 (62.0%),
Unsplash fashion 108/152 (71.1%). The fashion failure is therefore larger
than the earlier 34% measured on the 88-row date-clean cell and holds across
three independent sources; retouched fashion photography is now the single
worst negative domain for Model 1. The provider cascade stays quiet on all
three cells (no_ai 126/150, 105/152, 99/150; at most 15 stray `google` calls,
no openai or meta fabrications). A second stochastic pass over the Meta
muse-image-1.0 grid added 61 more API generations with zero hash overlap
with pass one (corpus now 127 + 5 checkout originals), moving the Meta class
toward margin calibrability. Artifacts:
`stock-negative-2026-08-27/`, `meta-muse-corpus-pass2-2026-08-26/`.

### Documents and receipts, 2026-08-27

Two public pre-diffusion research datasets filled the unmeasured document
domain: CORD-v2 thermal-printed photographed receipts (99 rows) and FUNSD
scanned forms (150 rows), both provenance-metadata-clean on a spot scan. The
split is the sharpest in the whole failure map: Model 1 accepted **89/99
receipts as AI-generated (89.9%, score median 0.469 against the 0.306
threshold)** while form scans stayed near the contract at 5/150 (3.3%,
median -0.001). Photographed thermal paper (small, noisy, low-texture,
layout-symmetric) is the single worst human negative domain measured,
surpassing retouched fashion (71.1%); clean high-fidelity scans pass. The
document domain therefore decomposes by capture quality rather than content
type, and any receipt-adjacent workload inherits a 90% false AI rate from
Model 1. Artifacts: `documents-negative-2026-08-27/`.

Diagnostics then fixed what can be done about it. Receipt embeddings sit
**99.0% closer to AI positives than to photos** in Model 1 space, so every
veto, head, or continuation over this embedding is structurally dead for
receipts, the fashion trap in its worst form; form scans sit at 26.7% and are
fine. A trivial white-background rule is also dead (CORD receipts are dark
noisy photographs of crumpled thermal paper, white fraction 0.021, at photo
levels). A receipt-document **gate** (ridge receipt-vs-rest over 499 receipt
embeddings with photo, museum, AI, and FLUX negatives) reaches **100%
receipt coverage**, but at the operating point chosen by inner CV it also
gates 14.4% of AI-test, 12.0% of FLUX, and 2% of EvalGEN, with family losses
up to 25.3% (fal) and 25.2% (xAI) — infographic-like generators trip the
document gate. As a product abstain this converts a 90% wrong-verdict domain
into unknowns at the price of a double-digit recall loss across AI families,
which violates the frozen recall contract; a high-precision variant with a
stricter threshold could gate fewer AI rows but the inner-CV F1 was only
0.444, so precision headroom is thin. The honest options are therefore:
(1) scope receipts out of the supported contract explicitly, or (2) a
calibrated abstain band presented as unknown rather than no_ai, accepting a
bounded recall loss decided as a product tradeoff, not a model fix.
Artifacts: `receipt-gate-2026-08-27/report.json`.

The 2026-08-31 layout probe (OpenCV stacked text-lines, no CLIP, no
retraining) splits that failure into two domains. Photoreal AI receipts
are easy layout positives (16 and 26 lines on the two probes, gated at
every cut from 3 to 15). Photographed thermal CORD is the hard cell
(median 6 lines). Photos and AI-test sit at median 1 line. A
high-precision cut `n_lines >= 11` covers both AI-receipt probes, 14.1%
of CORD, 0.3% of a 300-photo sample, 0/24 Kodak, and 3.7% of a 300-image
AI-test sample, versus the CLIP receipt-gate's 14.4% AI-test loss. Paddle
OCR on the two AI receipts reads brand and totals and garbles the
surrounding Spanish; that is a specialist signal, not a Model 1 feature.
Do not add receipts to CLIP-L train. Next: a layout-first router into an
OCR specialist, with thermal CORD still uncovered at a photo-safe cut.
Artifacts: `receipt-layout-gate-2026-08-31/report.json`.

Batch-1 then filled four more unmeasured domains under the same sieves
(428→421 rows after phantom repair: UI screenshots, 150; Danbooru community
digital art, 145; software-rendered matplotlib charts, 82; OpenStreetMap
tiles, 44). Frozen Model 1 false-positive rates: **UI screenshots 149/150
(99.3%)**, **digital art 144/145 (99.3%)**, charts 55/82 (67.1%), map tiles
21/44 (47.7%). The 150 UI files in that measurement were local product
screenshots; they were replaced on 2026-08-31 with CC-licensed Flickr
images (see the both-model freeze). The 149/150 figure is the original cell. The 1-NN diagnostic puts every failing
domain inside the AI cluster (UI 100%, art 100%, maps 100%, charts 95.1%
closer to AI than to photos), confirming the structural pattern: any
human-made content that is layout-structured, graphics-like, or
software-rendered lives inside the AI region of the CLIP-L space and cannot
be repaired there. Combined with receipts (89.9%), fashion (62-71%), logos
(38%), and museum art (33.8%), the boundary of the supported domain is now
sharp: clean photographs and high-fidelity scans pass; everything structured,
stylized, or degraded fails at 33-99%. The product contract and any future
routing gate must draw the line at photographic content, not at "images".
Artifacts: `batch1-negative-2026-08-27/`.

### Batch-2 and the full failure map, 2026-08-27

Batch-2 added four more domains under the same sieves (memes from imgflip
top-100, whiteboard photos, ticket/stamp scans, book pages; 396 rows):
memes 58/97 (59.8%), book pages 27/100 (27.0%), whiteboards 3/100 (3.0%),
tickets 3/99 (3.0%). Whiteboards and tickets join the passing side; memes
and book pages join the failing side. The complete failure map now covers
**20 measured negative domains**.

### H2 domain router falsified, 2026-08-27

A binary photographic-vs-structured router on Model 1 embeddings, trained on
all 16 failing domains vs photo cells, reached 98.2% structured coverage and
4.4% photo false-structured rate, but routed **97.9% of AI-test and 98.0% of
FLUX as structured**: every AI family (94.3-100%) looks non-photographic to
the router. This is the third independent confirmation (after 1-NN overlap
and linear-veto exchange) that provenance is not encoded in the CLIP-L
appearance space: routing cannot separate AI content from structured human
content because they occupy the same region.

### H3 process detector: generator-specific traces do not transfer, 2026-08-27

Contrastive pairs (same content, different process) were generated from 125
source images across five domains (charts, logos, art, documents, UI), each
recreated by three generators: OpenAI gpt-image-1-mini (edits/img2img), Meta
muse-image-1.0 (prompt), and Gemini 2.5-flash-image (prompt), totaling 369
pairs. A patch-level FFT-magnitude CNN trained on these pairs separates
human originals from AI recreations at pairwise AUC 0.909 (0.885 with two
generators, 0.949 with one VAE), confirming the process signal exists within
the training distribution. Transfer to production AI collapses: AI-test
score distributions overlap photo distributions (mean +0.013 to +0.026 above
photos), giving AUC 0.55-0.60 — near chance — while human receipts score
+0.095 above photos, higher than any AI family. Each VAE/decoder leaves a
unique spectral signature; three generators are insufficient to cover the
space, and the learned features do not generalize to unseen production
generators. This closes the process-trace family in its current form;
reopening would require 10+ diverse generators in training or a
fundamentally different feature space that captures what AI generation
processes share beyond their decoder-specific artifacts.

### Provider cascade across all 20 domains, 2026-08-27

The frozen provider cascade (margin 0.50) was scored on every measured
negative and positive domain. Photo negatives are quiet (COCO 0.7% stray,
Kodak 0%), museum/modern/stock cells leak 2.9-11.9% into `google` (mostly
stylized content that pattern-matches Imagen), but documents are the
exception: **69.8% of document/receipt content receives a stray provider
call** (48 openai, 19 google out of 96), and batch-2 content (memes, tickets,
whiteboards, book pages) leaks 27.0% (35 google calls out of 137). Wild AI
leans google at 15.1% with only 1 openai call. The provider cascade is
therefore safe on photographic content, partially safe on stylized content,
and unsafe on document-like content — mirroring the Model 1 failure map. A
production integration of the provider layer must gate on the same
photographic-content condition as the detector itself.
Artifacts: `provider-all-domains-2026-08-27/report.json`.

### Research-driven hypotheses H4-H6, 2026-08-27

Three approaches from 2025 literature (C2P-CLIP prompt learning from AAAI,
spectral invariant from CVPR, TTA voting) were run against the frozen gates.

**H4 (MLP prompt-probe on frozen CLIP, C2P-CLIP inspired):** a two-layer
MLP trained on frozen CLIP-L embeddings improves AI-test recall to
1,878/1,905 (98.6%, gate passed) and FLUX to 298/300, but at the cost of
fresh Open Images 473/3,000 (15.8% FPR, gate failed) and Kodak 6/24. The
nonlinear head trades photo safety for recall — same direction as quarter-hard
but via MLP instead of continuation. Gates: 2/4 passed.

**H5 (spectral invariant, CVPR 2025):** radial power-spectrum deviation from
a camera-image reference detected only 7/297 AI-test (2.4%) and 1/300 FLUX.
The 1/f^alpha spectral profile does not discriminate current AI generators
from camera photos at any useful operating point. Gates: 0/4 passed.

**H6 (TTA voting, 6 views):** averaging scores over center + 4 corners + flip
preserves photo gates (fresh 5/300, Kodak 0/24) but reduces AI-test from
273→264 and does not rescue any negative domain: documents still 65/80 FP,
batch1 80/80, museum drops to 1/80 but fashion stays 14/80. TTA helps museum
but hurts everything else. Gates: 2/4 passed (AI-test and fresh marginally,
Kodak and FLUX lost).

All three hypotheses are closed. The H4 MLP is the strongest non-continuation
result (98.6% AI-test recall from frozen embeddings) but fails the photo
contract, consistent with every previous finding: CLIP-L encodes appearance,
not provenance, and no head on top of it can satisfy both contracts
simultaneously.

### Y-phase campaign: frequency convention fix, 2026-08-27

The initial Y-phase campaign used FFT bin indices (16-32) instead of spatial
periods (16-32 pixels = 1/32 to 1/16 cycles/pixel), targeting the wrong band
at native resolution. After fixing to cycles/pixel, the full-octave Y-phase
scramble reproduced the known kill: **not_detected at 20.5 dB** on s1 at
1122x1402, confirming the mark lives in the 16-32 px period band of the
luminance channel. Partial sub-bands (10-75% of the octave) produced
visually invisible modifications (42.5 dB, baseline-level PSNR) but remained
detected — the mark is distributed across the full octave, not concentrated
in a sub-band. Point intervention within this octave is therefore not viable:
a quiet kill inherently requires modifying the entire 16-32 px octave, which
costs 20-25 dB.

### Universal AI detector v3-v8: diverse training shifts the curve, 2026-08-27

A fundamentally different approach from all closed hypotheses: instead of
repairing Model 1 (trained on photo negatives only), train a fresh classifier
on the FULL diversity of human-made and AI-generated content. The v3 ridge on
frozen CLIP embeddings with 3,224 structured negatives achieved 88.0% recall
at 0.1% photo FPR with structured FP reduced from 33-99% to 0-14%. The v4 MLP
pushed recall to 91.3%. Crucially, v5 (fine-tuning the CLIP tower on the same
data) UNDID the structured gains, proving the improvement comes from training
data diversity on frozen embeddings, not from representation change.

Iterations v6-v8 expanded the corpus to 49 negative domains (12,258 images)
plus 5,043 AI positives across 19+ families. The v7 MLP achieved **93.3%
AI-test recall** (matching Model 1's 93.02%) with **36 of 49 negative domains
at zero false positives**. The v8 auto-pipeline (650 additional targeted
images for weak domains) improved to **39/49 domains at zero FP** with digital
art at 24.8% (from 99.3% in Model 1), fashion at 12.0% (from 62-71%), and UI
at 12.0% (from 99.3%). Photo FP remained at 0.7% (COCO) and 0.7% (picsum).

The trade-off curve has shifted permanently: Model 1 traded 93% recall for
33-99% structured FP; v8 delivers 93% recall with 0-25% structured FP and
39/49 domains at exactly zero. The remaining gap is concentrated in digital
art (24.8%), fashion (12%), and UI (12%) — all domains where AI-generated and
human-made content share deep visual similarity. Photo-side gates (fresh OI
≤50, Kodak 0/24) remain the open constraint.
Artifacts: `universal-v7-2026-08-27/`, `auto-pipeline-2026-08-27/`.

### Three-tier ensemble detector, 2026-08-27

Combining Model 1 and v8 into a three-tier output transforms the binary
detector into a calibrated confidence system:

- **DEFINITELY_AI** (both models agree): 79.6% recall at 0.50% photo FPR,
  Kodak 0/24, 46 of 49 negative domains at zero FP, worst FP 2.1% (digital
  art). This tier is production-safe for verification and moderation.
- **POSSIBLY_AI** (one model fires): 13.8% additional recall at moderate
  FP. This tier flags images for human review.
- **LIKELY_HUMAN** (neither fires): high-confidence negative, includes
  2,946 of 3,000 fresh photos and all 24 Kodak images.

The union (DEFINITELY + POSSIBLY) achieves 93.3% recall at 1.80% photo FPR
with Kodak 0/24 — matching Model 1's recall while passing the Kodak gate
that Model 1 alone could not. The three-tier system is the first detector
that works across ALL content types (photographic, structured, documents,
UI, art) with per-tier calibrated error rates.
Artifacts: `three-tier-2026-08-27/`, `ensemble-v8-m1-2026-08-27/`.

### Provider v2: diversity training eliminates document leakage, 2026-08-27

Applying the same diversity-training insight to the provider cascade (adding
3,769 structured negative features to training alongside 1,936 photo
negatives) produced dramatic improvements. **Document/receipt leakage dropped
from 69.8% to 5.1%**, UI from 10% to 4.3%, fashion from 5-15% to 0-3%. AI
recall improved for OpenAI (84%, from 75%) and Google (88.8%, from 79%).
Photo leakage remained at 1.4%. The worst regression is memes at 47.9%
(from 27%) — text overlays on photos pattern-match to AI generators in the
provider feature space. Six domains achieve zero provider leakage.
Artifacts: `provider-v2-2026-08-27/`.

### Universal v11: full taxonomy coverage, 2026-08-30

The master data plan identified 148 visual domains of human-made content.
Massive collection (3,343 images across 73 new domains from Openverse)
plus gap-filler (724 images from Safebooru for anime/manga/chibi/mecha
and diverse generated documents) expanded the dataset to 19,768 training
images across 136 negative domains. Universal v11 (MLP-512 on frozen CLIP,
three-tier ensemble with Model 1) achieves:

- **DEFINITELY_AI**: 90.3% recall at 0.50% photo FPR, Kodak 0/24
- **ANY_AI**: 95.6% recall (exceeds Model 1's 93.0%)
- **124 of 134 negative domains at exactly 0% FP**
- Worst FP: digital_art 13.8% (human fan art visually inseparable from AI art)
- Only 57 false positives across 1,090 test images in all failing domains

This is the strongest result of the campaign: a single model that handles
photographic content, structured documents, UI screenshots, art, maps,
medical imagery, anime, and all other measured domains with near-zero false
positives on 92% of domains. The remaining failures are concentrated in two
categories (stylized digital art, heavily retouched fashion) where AI and
human content share deep visual similarity.
Artifacts: `universal-v11-2026-08-30/`, `massive-collection-2026-08-27/`,
`gap-filler-2026-08-27/`.

### Fine-tune v12: representation limit confirmed, 2026-08-30

Fine-tuning the CLIP tower on the full 19,518-image dataset (6x the data
that failed in v5) achieved 95.1% standalone AI-test recall at 2.9% photo
FPR — but destroyed structured-content discrimination: 100% FP on digital
art and UI screenshots, 80% on receipts, 14/59 domains at zero FP. This
confirms that the v5 failure was not a data-volume issue but a fundamental
property: fine-tuning the CLIP embedding space causes the model to lose the
structured-content separation that frozen embeddings provide. The frozen
CLIP + MLP approach (v11) remains the only working architecture. Provider v3
trained on all negative domains achieved 1.4% photo leak.
Artifacts: `fine-tune-v12-2026-08-30/`, `provider-v3-2026-08-30/`.

### v13: plateau confirmed, 2026-08-30

Top-plan data expansion (1,169 images: TC260 500, art 500 from Safebooru,
fashion 300 from Unsplash, memes 250, Meta 79, OpenAI 40) confirmed the
frozen-CLIP plateau. v13 achieves 89.6% DEFINITELY / 95.1% ANY with
**128 of 136 domains at zero FP** (up from 124/134 in v11) and only 8
non-zero FP domains. Fashion dropped from 9.3% to 5.3%. Digital art
remains at the representation limit (15.9%, was 13.8% — the Safebooru
expansion did not help). The frozen CLIP + MLP architecture has reached
its ceiling: further data diversity improves domain coverage but cannot
push DEFINITELY recall past ~90% or reduce art FP below ~15%.
Artifacts: `v13-final-2026-08-30/`, `top-plan-data-2026-08-30/`.

### CLIP-H ensemble, multi-scale TTA, and patch attention, 2026-08-30

The 19,950 CLIP-H embeddings (1024-d, `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`)
already had a ridge head from the Model 1 sweep (81.2% AI-test at 1.4% fresh
FPR). What had not been measured was using that ridge as a third voter next
to Model 1 and v13, plus a new MLP on the same cache, native-resolution TTA,
and patch-token attention.

On the original paired cells (n=1,905 AI-test, 3,000 fresh, 24 Kodak, 300
FLUX) the current DEFINITELY rule `Model 1 AND v13` is 89.6% / 1.17% / 0/24 /
85.3%. Replacing the AND of v13 with an OR of the CLIP-H ridge inside that
rule is the only positive result of the three:

| Rule | AI-test | Fresh FPR | Kodak | FLUX |
| --- | ---: | ---: | ---: | ---: |
| Model 1 AND v13 | 89.6% | 1.17% | 0/24 | 85.3% |
| Model 1 AND (v13 OR CLIP-H ridge) | **91.9%** | 1.40% | 0/24 | **91.7%** |
| CLIP-H MLP, photo negatives only | 89.8% | 2.40% | 0/24 | 56.3% |

The CLIP-H MLP without structured negatives repeats the v12 pattern: in-grid
recall holds, FLUX hold collapses. Multi-scale TTA on CLIP-L-ft (letterbox
224/336/448, mean embedding, ridge recalibrated on `photo_dev_oi`) raised
AI-test from 93.1% to 95.4% on 1,847 files and broke the operating point:
fresh 1.8% to 3.2%, Kodak 0/24 to 1/24, FLUX 92.7% to 87.3%. Patch-token
attention (256 projected patches, CLS-query, photo negatives only) was 92.2%
AI-test at 1.4% fresh and 0/24 Kodak, with FLUX at 39%. Extra views and extra
tokens without diversity training do not survive the 1% photo contract.

The frozen CLIP-L MLP plateau in v13 still stands as a single-backbone
result. A second frozen backbone as an OR branch inside DEFINITELY is a
different lever.
Artifacts: `cliph-ensemble-2026-08-30/`, `multiscale-tta-2026-08-30/`,
`patch-attention-2026-08-30/`.

### CLIP-H diversity MLP, 2026-08-30

The 8,333 taxonomy images that v13 added were embedded with the same CLIP-H
tower (1024-d). A v13-protocol MLP (5,330 pos including 809 extra AI, 6,129
photos, 6,674 structured negatives, 113 domains) was trained on that cache.

Diversity training gave CLIP-H the structured-content property: **113/113
taxonomy domains at 0% FP** for the CLIP-H MLP. It did not repair FLUX.
Standalone CLIP-H diversity is 83.0% AI-test / 1.20% fresh / 0/24 Kodak /
66.0% FLUX, better than the photo-only CLIP-H MLP (56.3% FLUX) and still
worse than the original CLIP-H ridge (91.3% FLUX). Linear CLIP-H stays the
robust voter; the MLP spends that robustness on domain coverage.

| Rule | AI-test | Fresh FPR | Kodak | FLUX |
| --- | ---: | ---: | ---: | ---: |
| Model 1 AND v13 | 89.6% | 1.17% | 0/24 | 85.3% |
| Model 1 AND (v13 OR CLIP-H ridge) | 91.9% | 1.40% | 0/24 | 91.7% |
| Model 1 AND (v13 OR CLIP-H diversity) | 91.1% | 1.27% | 0/24 | 88.3% |
| Model 1 AND (v13 OR ridge OR diversity) | 92.2% | 1.43% | 0/24 | 92.0% |

The production-shaped DEFINITELY rule remains Model 1 AND (v13 OR CLIP-H
ridge). Adding the diversity MLP on top is +0.3 pp recall at +0.03 pp fresh.
The 113/113 zero-FP count is only over taxonomy cells CLIP-H actually
embedded, not a re-score of v13's original `scores.npz` domains.
Artifacts: `cliph-diversity-2026-08-30/`, `embeddings-clip-h-taxonomy.npz`.

### Provider v4: focal loss and TC260 expansion, 2026-08-30

Provider v4 refit the five-head 124-d cascade on the v2 bank plus top-plan
expansion (TC260 1,213, Meta 199, OpenAI 2,038, 4,087 structured negatives,
250 imgflip memes as extra negatives). A one-vs-rest ridge and a focal-loss
MLP (`gamma=2`) were trained on the same rows. Full `ai_test` family cells,
not the 100-row v2 subsample:

| Head | OpenAI | Google | TC260 | photo COCO | `meme_template` |
| --- | ---: | ---: | ---: | ---: | ---: |
| v4 ridge | 83.2% | 83.9% | 68.1% | 1.4% | 50.7% |
| v4 focal | **93.4%** | 84.7% | **80.5%** | **0.7%** | **33.8%** |

Focal meets the TC260 ≥75% target and lifts OpenAI. Imgflip memes used in
training score 0% leak on themselves and are not a test; the honest meme
cell remains `meme_template` at 33.8%, down from 47.9% but short of 15%.
Meta 38/39 was a train-end slice and is not a hold-out. Artifacts:
`provider-v4-2026-08-30/`.

### Holes: digital art, memes, Meta hold-out, 2026-08-30

Three remaining cells were re-measured with disjoint splits.

**Digital art.** The 145 Danbooru rows are in v13 training and still
DEFINITELY at 15.9% (`m1 AND v13`). Adding CLIP-H ridge as an OR branch,
the rule that lifts AI-test to 91.9%, fires on 138/145 (95.2%): that
branch is unsafe on illustration. A CLIP-H diversity MLP, trained on
taxonomy art and not on these 145, scores 4.8% AI. A global AND with
that MLP drops FLUX from 91.7% to 64.7%. An illustration-versus-photo
ridge (807 taxonomy art/anime/paintings vs `photo_train`, 1% photo-dev
FPR) restricts the veto to illustration-like images: digital art 95.2%
to **4.8%**, AI-test 91.9% to 88.7%, FLUX 91.7% to 89.7%, Kodak 0/24,
fresh 1.40% to 1.27%. SD/Comfy takes the largest family drop (93.2% to
81.4%). The 145 were not in the illustration or diversity training
caches.

**Memes.** `meme_extra` is a content-hash duplicate of `meme_template`
(97/100). Holding the 97 hashes out of provider training and evaluating
them against 250 disjoint imgflip negatives yields **40.7% ungated
leak**. Training on 48 of the 97 and testing the other 49 still leaks
34.8%, so 124-d forensic features do not represent this domain.
Under the detector, DEFINITELY is 2/97 and those two stay `no_ai`:
**gated provider leak 0%**. Provider must run only on DEFINITELY /
POSSIBLY, not on every file.

**Meta.** Training on the 122 muse-image rows and testing the later
`photo_ai_meta_v2` pass (77 readable of 79) is **70.1%**. That is the
hold-out. The earlier 38/39 figure was a train-end slice and is
withdrawn. Adding the 95 disjoint `photo_ai_meta` files already on disk
from `targeted-data-2026-08-27` (same model, different collection, zero
file-sha256 overlap with muse or v2) lifts the same hold-out to **67/77
= 87.0%** without new generation.

That number is only reproducible from a frozen split. The collectors
hashed the API payload and then re-encoded PNG, so `targeted-data` and
`top-plan` Meta rows had a `sha256` that did not match the file on
disk (95/95 and 79/79). Muse webp rows already matched. The repaired
identity is sha256 of the file bytes; the old payload hash is kept as
`payload_sha256`. The freeze is `meta-split-2026-08-31/split.json`
(217 train = 122 muse + 95 targeted, 79 hold-out). Training reads that
file, re-hashes every path, and records extractor refusals instead of
dropping them: two hold-out 1280x1920 images and one 1600x1600 train
image return no 124-d features (`n_scored=77` of 79 listed). Seed
`20260956`. Re-run: `uv run python .local-eval/synthid/ai-photo-2026-08-22/meta_provider_split.py`.
Artifacts: `holes-2026-08-30/`, `meta-expand-2026-08-31/`,
`meta-split-2026-08-31/`.

### Both-model freeze and repeated training, 2026-08-31

A single catalog now identifies every file used by the detector and the
provider by the sha256 of the bytes on disk (`both-models-2026-08-31/catalog.json`,
32,690 files, 590 listed missing). Roles cover Model 1 splits, taxonomy
struct/AI extras, and provider OpenAI/Google/TC260/Meta/no_ai plus the Meta
and meme hold-outs. OpenAI and Google are a fresh seed-`20260821` sample of
2,000 existing labeled files each, with catalog sha256 verified at sample
time; they are not the anonymous `provider-class-features.npz` arrays.
`no_ai` is a documented 2,000-draw from `photo_train` (1,848 yielded 124-d
features).

The first CPU repeat that morning still trained on 150 local product UI
screenshots in `batch1-negative-2026-08-27` (`own_products_ui`). Those files
are not public internet. They were replaced the same day with 150
CC-licensed Flickr screenshots via Openverse (`license_type=all-cc`, host
`live.staticflickr.com` only): CC BY 2.0 42, CC BY-NC 2.0 42, CC BY-SA 2.0
29, CC BY-NC-SA 2.0 28, CC BY-NC-ND 2.0 6, CC BY-ND 2.0 2, CC0 1. Collector:
`replace_private_ui.py`. File sha256 after PNG save is identity. Catalog
`own_products_ui` is now 0; the 150 Flickr rows carry `source=openverse_flickr`
and a license. CLIP-L extras for those hashes live in `extra-clip-l.npz`
(150 × 768). 127 of 150 yielded 124-d features; 23 are listed in
`features-124d-failed.json` (1,808 refusals in the catalog after the swap).
CLIP-L embeddings keyed by sha256: 26,573.

Training is CPU, seeds `detector=20260940` and `provider=20260956`. Two
consecutive runs from the updated freeze were **byte-identical**
(`compare.json`, zero diffs). Pre-replacement: AI-test DEF 92.7%, FLUX
84.0%, OpenAI 90.0%, Google 89.8%, TC260 79.2%, Meta 83.1%, meme leak 33.7%.
After the public UI swap:

| Head | Cell | Run 1 = run 2 |
| --- | --- | ---: |
| Detector DEFINITELY | AI-test | 92.4% (n=1,847) |
| Detector DEFINITELY | Fresh | 1.3% (n=3,000) |
| Detector DEFINITELY | Kodak | 0/24 |
| Detector DEFINITELY | FLUX | 83.0% (n=300) |
| Provider | OpenAI test | 90.8% (345/380 of 381) |
| Provider | Google test | 90.9% (339/373 of 377) |
| Provider | TC260 test | 78.6% (298/379 of 384) |
| Provider | Meta hold-out v2 | **85.7%** (66/77 listed 79) |
| Provider | Meta hold-out v3 | **89.4%** (177/198) |
| Provider | Meta hold-out pooled | **88.4%** (243/275 listed 277) |
| Provider | meme_template ungated | 29.1% leak (86 of 97) |

A second Meta hold-out (`photo_ai_meta_v3`, 198 of 200 requested; two
near-duplicates rejected) was generated the same day from 50 new
photographic prompts, four passes, requested size 1024x1024 (delivered
1600x1600). It is eval-only: not in `provider_meta_train` and not in
detector extras. File sha256 after PNG save is identity. Scoring the
already-frozen `run1/provider.pt` and the retrained freeze both give
177/198. Pooled with v2 the lower 95% Wald bound sits above 80%
(243/275). Collector: `collect_meta_holdout_v3.py`. Split:
`meta-split-2026-08-31/split.json` (train 217, hold-out v2 79, hold-out
v3 198).

Re-run: `uv run python .local-eval/synthid/ai-photo-2026-08-22/replace_private_ui.py`
then `uv run python .local-eval/synthid/ai-photo-2026-08-22/collect_meta_holdout_v3.py`
then `uv run python .local-eval/synthid/ai-photo-2026-08-22/integrate_meta_holdout_v3.py`
then `uv run python .local-eval/synthid/ai-photo-2026-08-22/freeze_both_models.py`.
Durable copy: `data/research/reports/both-models-2026-08-31/` (gitignored).

Verification (`verify_both_models_freeze.py`) re-hashed all 32,690 catalog
files: every path exists, every sha256 matches disk, `own_products_ui` is
0, Meta v3 is disjoint from train by file hash and by prompt, v3 is
eval-only, all 198 v3 rows yielded 124-d, and the two freeze runs stay
byte-identical. One v2 hold-out is a Hamming-15 phash neighbor of a
Venice-canal train image (rainy Paris boulevard, different prompt and
aspect); that file also failed 124-d extraction, so it is not in the 77
scored. A 16-image visual sample of the Flickr UI cell is **12
screenshots, 3 photographs, 1 product still**. License and host checks
pass; the Openverse screenshot queries are not a UI-only cell. Re-run:
`uv run python .local-eval/synthid/ai-photo-2026-08-22/verify_both_models_freeze.py`.

### Five-head provider cascade with TC260, 2026-08-27

Adding TC260 (721 Chinese-AIGC images from the local research catalog) as a fifth
class to the provider cascade produced a working five-way system:
openai 68.8%, google 76.6%, **tc260 63.8%**, meta 71.8% mean recall under
uniform margin 0.50 (25 repeated splits). Per-class margin calibration
against a 1.67% aggregate photo-AI-rate budget (0.42% per class) yielded
margins openai 0.31, google 0.29, tc260 0.39, meta 0.55, and TC260 test
recall at calibrated margins reached **71.4%** on 196 held-out images. The
Chinese-AIGC domain is therefore a viable pixel class comparable to the
western providers, not an outlier. The per-class margin approach works:
different classes need different safety margins to share the same photo
budget, and the calibrated cascade is production-ready for the photographic
content domain. Artifacts: `tc260-five-head-2026-08-27/report.json`.

The combined-pool control then separated the data question from the geometry
question: refitting the same pooled/domain/multiclass ridge vetoes on a
687-row modern-negative pool (Openverse-clean plus all three stock cells,
three times the original fit size) reproduced the earlier Pareto curve to the
point: at positive compliance the best dev error count stayed 41/186 with
zero route errors repaired, and every repaired negative still cost about 1.4
AI positives. The linear-veto failure over the frozen Model 1 embedding is
therefore not a data-volume artifact; tripling diverse negatives moves
nothing. Artifacts:
`modern-domain-veto-combined-2026-08-27/` (development report and pareto).

The doubled Meta corpus then allowed its own margin sweep (127 API rows in
the bank after feature extraction, 25 repeated splits, photo leak measured on
held-out no_ai rows). The curve is steep exactly where the frozen 0.50 margin
sat: margin 0.00 recalls 92.2% meta at 5.3% photo AI-rate, 0.10 gives 90.1%
at 3.6%, 0.20 gives 85.2% at 2.2%, then the cliff: 0.30 falls to 70.2%, 0.40
to 39.4%, and the frozen 0.50 lands at 12.4%. A workable photo-safe
operating point exists at margin 0.20 (85% meta recall at 2.2% photo leak),
but 2.2% is still thirty times the Model 1 photo contract, so for provider
attribution the honest rule stays argmax until an external held-out meta cell
exists. The margin, not the head, was always the limiter on the small class.
Artifacts: `meta-margin-2026-08-27/report.json`.

## Representation-change campaign closed, 2026-08-27

A pre-registered two-candidate campaign (plan in `data/research/REPRESENTATION-CAMPAIGN.md`,
gates declared before any run: AI-test at least 1,772/1,905, fresh Open
Images at most 50/3,000, Kodak 0/24, FLUX at least 288/300, EvalGEN at least
95/100, no tuning) tested the two remaining representation families.
Candidate 1, a DINOv2-L ridge head (224 px CLS embeddings, nested-CV penalty),
failed all five gates simultaneously: AI-test 1,427/1,847, fresh 162/3,000,
Kodak 1/24, FLUX 29/300, EvalGEN 5/100 — the frozen DINOv2 space separates AI
from photos far worse than fine-tuned CLIP-L everywhere. Candidate 2, a
patch-level FFT tower on the DDA paired-reconstruction contract (1,024 fresh
sd-vae-ft-mse pairs, real 0.549 versus reconstruction 0.457 medians, 0.949
pairwise separation), learned its contract perfectly and still collapsed on
transfer: AI-test 14/1,847, FLUX 0/300, EvalGEN 0/100, because production
generators do not carry this VAE's reconstruction signature. The declared
falsification criterion triggered: neither family is the missing lever, and
the general metadata-free AI-versus-human classifier at the product contract
is beyond local scale as pursued. The product keeps the documented-remainder
contract; Model 1 stays the accepted research operating point with its
measured bounds. Reopening requires a materially new lever.

A taxonomy continuation then changed the training mix itself instead of the
veto: two arms re-ran the expanded quarter-hard recipe (ordinary AI replay,
EvalGEN pair positives, ordinary photos) with one or two of the eight negative
slots per batch drawn from the date-clean taxonomy pool (Cleveland museum plus
Openverse modern cells), 256 steps, thresholds calibrated at the 99th
percentile of `photo_dev_oi`. Both arms retained the positive contract
(EvalGEN 97-99/100, FLUX 289/300) and repaired the museum half (AIC dev
4/350), but the modern half did not move: fashion stayed at 33-35/88 on the
date-clean Unsplash cell against a 30/88 baseline, and long exposure at
11-12/98. Training on 40 date-clean Flickr fashion rows does not transfer to
Unsplash fashion, which mirrors the veto finding from the optimization side.
The development domain gate therefore fails on fashion for every arm, and no
arm reached the frozen core cells. With the linear-veto family, aligned
mining, self-hard continuation, and now in-mix taxonomy continuation all
closed on the same source-specific overlap, the CLIP-L program for the modern
negative domains is measured as exhausted: representation change is not one
more operating point away.

Local reproducibility artifacts are under
`.local-eval/synthid/ai-photo-2026-08-22/`, especially the
`paired-reconstruction`, `open-images-train-mining`, `evalgen-expanded`,
`evalgen-quarter`, `evalgen-rescue`, `evalgen-stacker`,
`negative-taxonomy-audit-2026-08-25`, `negative-taxonomy-aic-audit-2026-08-25`,
`modern-negative-unsplash-2026-08-26`, `modern-negative-openverse-2026-08-26`,
`modern-domain-veto-2026-08-26`, `modern-domain-veto-clean-2026-08-26`,
`contamination-scan-2026-08-26`, and `taxonomy-continuation-clean-2026-08-26`
run directories.

### Wild extras, not SynthID

| Hypothesis | 2026-08-23 | Use |
| --- | --- | --- |
| Missing camera PRNU | Gray `gpt-image-2` highpass RMS 0.25 vs COCO 14.6 | Texture confound. A Wiener PRNU residual on *photographs* vs Model 1 errors is the real test |
| JPEG ELA | COCO 3.13, s1 1.97, gray stamp 0.49 | Export history, leaks PNG vs JPEG, not a provider |
| CFA / Bayer presence | Photo-edit ratio 0.117 vs camera 0.184 vs gray 0.588 | Weak camera vote, overlap. Inverse of the Bayer remover |
| Double-JPEG ghosts | s1 / gpt-image-2 / camera all min at Q90 | Codec, not a provider |
| Perfect-circle / text-edge rate | Circles/MP 385 vs 536, edge 0.052 vs 0.072 | Too noisy for abstain |
| Wiener PRNU on photographs | Edits 4.61 vs camera 8.05 | Donor JPEG texture leftover, not a missing sensor |
| PNG Paeth filter mix | gpt-image-2 PNG 99.9% Paeth vs camera 73% | Export fingerprint |

None of these should be named a SynthID score.

## External literature (surveyed 2026-08-23)

AWPD / FSNet ([arXiv:2603.06723](https://arxiv.org/abs/2603.06723)) is
the published "is there any invisible watermark" task. Leave-one-algorithm-out
SynthID Acc 0.894 is *not* Model 1 and *not* a payload decoder. UniFreq's
SynthID split is 2,000 Imagen-API AIGC crops at 256x256, no photographs,
no Firefly, no OpenAI. A head trained that way can pass as watermark
presence while actually reading generator/size texture, which is the L1
failure mode.

Model 1 remains AI-versus-camera on CLIP-L-ft. That is a published
task, not a watermark task. Adjacent papers:

| Source | Claim | Map to Model 1 |
| --- | --- | --- |
| Ojha, Li, Lee, [arXiv:2302.10174](https://arxiv.org/abs/2302.10174) (CVPR 2023, UnivFD) | A classifier trained to see "fake" treats unseen generators as the real sink. Frozen CLIP + nearest neighbor / linear probe generalizes better than a trained CNN | This is the architecture. We finetuned the last two CLIP-L vision blocks instead of freezing, and put Firefly and a locked Open Images fresh set in the gate |
| Cozzolino et al., [arXiv:2312.00195](https://arxiv.org/abs/2312.00195) | CLIP linear probe, few shots from one generator, holds on DALL-E 3 / Midjourney / Firefly | Firefly is the cell we required. Their paper is why Firefly belongs in the test, not as a surprise |
| Corvi et al., [arXiv:2304.06408](https://arxiv.org/abs/2304.06408) | Spectral peaks and mid-high power differences, GAN and diffusion | Generator fingerprint, not a payload. Explains why a Fourier codebook lights up Google *and* Open Images |
| Zhong, Xu, Zou, [arXiv:2601.22778](https://arxiv.org/abs/2601.22778) (DCCT) | Self-supervised color-channel prediction under a Bayer mask; theoretical gap between photo CFA correlations and AIGC | Local Bayer interpolation-error ratio: edits 0.117 vs camera 0.184. Weak vote, not a payload |
| Klier and Baier, DFRWS EU 2026 | AI noise is not predominantly additive. Standard PCE vs smartphone PRNU: FPR 61% Firefly Image 4, 100% ChatGPT 5. Center crop kills those false positives without hurting true camera matches | Do not call missing PRNU a SynthID score. If we ever add a Wiener residual, crop and a recorded PCE threshold come with it |
| Popescu and Farid, IEEE Trans. Signal Process. 2005 | CFA interpolation leaves neighbor correlations; splicing breaks them | Classical forgery localization, not generation detection |
| Wang, Wang, Zhang, Owens, Efros, [arXiv:1912.11035](https://arxiv.org/abs/1912.11035) (CVPR 2020, CNNDetect) | Classifier on ProGAN + JPEG/crop aug transfers to many CNNs | The "one generator is enough" claim. Ojha is the correction once diffusion exists |
| Wang et al., DIRE, [arXiv:2303.09295](https://arxiv.org/abs/2303.09295) (ICCV 2023) | Reconstruction error under a frozen diffusion model | SDXL float32 at 512. VAE RMS: camera 11.84, photo edit 9.89, s1 9.15, gray stamp 1.24. DDIM DIRE RMS: camera 33.0, s1 31.5, photo 30.9, gray 2.40. Texture rank, not a payload. Float16 DDIM NaN'd on MPS |

They do not substitute for `verify-openai-synthid`.
