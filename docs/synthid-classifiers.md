# SynthID source classifiers (research)

> Audit correction, 2026-09-08: multi-period affine-probe results from
> schema 13 or earlier used confirmation patches during period selection
> (R04). They require rerunning and recalibration under schema 14; the
> frozen runtime detector uses separate code and its thresholds are unchanged.
> External-negative claims admitted without explicit independent evidence
> (R08) also need an evidence audit before reuse.

> Research archive for metadata-free OpenAI/Gemini source finding and
> provider-lineage classifiers. These are not SynthID payload decoders and are
> not shipped product verdicts. Current behavior:
> [supported signals](supported-signals.md) and
> [known limitations](known-limitations.md).
>
> Sister pages: [general AI-generated image classifiers](ai-generated-image-classifiers.md),
> [SynthID local detector](synthid-detector-research.md),
> [SynthID mark removal](synthid-removal-research.md), and
> [mechanism reference](synthid.md).

A source classifier is reliable only when photographs and non-target generators
are explicit negatives, a strict rule can abstain, and any watermark claim uses
an independent oracle. CLIP content embeddings and the 124-d origin-locked
residual bank are different features for different jobs.

Krawetz's Gemini-chat TPR critique is a verifier-quality claim, not a
feature we can ship. [Lead Stories, 2026-07](https://leadstories.com/analysis/2026/07/google-gemini-synthid-detector-confuses-results-within-same-chat.html)
documented Gemini repeating the first file's SynthID verdict inside a
chat; Google said that was fixed 2026-07-16. The OpenAI provenance API is
a different endpoint.

## Provider names from pixels

The task is: given a file with no metadata, is this OpenAI, Gemini, or
unknown, with almost no errors on camera photographs or other generators. That
is this section. `unknown` does not mean `not AI`; it includes AI images from
other providers and target-provider images the strict rule misses. This is not
a general AI-generation detector or a SynthID detector. Firefly, PixelBin, and
other generators have to sit in the test, because a head that only sees OpenAI
versus Gemini versus COCO will call Firefly a provider.

Three-way `openai` / `google` / `other` on Model 1 embeddings fails the
Firefly gate. CLIP-L-ft test accuracy 0.53; Firefly 35/31/18. CLIP-H 0.57;
Firefly 36/33/15. OpenAI versus Gemini AUC on CLIP-L-ft is 0.845; on the
124-d lattice bank it is 0.989. They are two pipelines, not one class.

### Renderer, not front-end (cross-carrier validation, 2026-08-26)

A local C2PA-labeled collection's issuer field allows a provider-label audit the
family labels cannot express: Bing Image Creator rows signed `Microsoft,
OpenAI` (renderer DALL-E, n=152), Microsoft-native rows (MAI-Image / Designer,
n=85), Designer rows signed `Microsoft, Google LLC` (renderer Imagen, n=36),
and the unattributed Instagram `made_with_ai` label set (n=275). Scored with
the frozen 124-feature openai/google/no_ai cascade, no training, metadata
used only to build cells: argmax called 124/152 (81.6%) of the Bing+DALL-E
cell `openai` and 34/36 (94.4%) of the Microsoft+Imagen cell `google`. Pixel
attribution therefore tracks the renderer behind the front-end, which is the
correct semantic for a stripped file: a Bing export is an OpenAI-rendered
image. The Microsoft-native cell split 48 openai / 35 google, consistent with
Designer historically routing between DALL-E and Imagen rather than one
pipeline. The Instagram `made_with_ai` set is not a Meta pixel class: argmax
leaned `google` at 196/275 with 68 `openai`, and the photo-first 0.50 margin
sent 182/275 to `no_ai`, so the label marks detected AI content of mixed
origin, not Meta-rendered pixels. A real Meta class needs fresh
`Imagined with AI` generations with known provenance; the catalog label is
not one. Artifacts:
`provider-renderer-cells-2026-08-26/report.json`.

### A Meta class from muse-image-1.0, 2026-08-26

The Meta Model API (`api.meta.ai/v1`, OpenAI-compatible images endpoint) made
a known-provenance Meta corpus possible without any account browser session:
61 images generated from a 61-prompt grid spanning portraits, product shots,
scenes, food, animals, architecture, illustration styles, text posters, and
abstract work across all three aspect ratios, plus the five oracle-verified
oracle-verified content-seal samples already on disk. All 66 are content-hash
unique, delivered at 1600x1600 / 1920x1280 / 1280x1920, and every API row
carries IPTC `trainedAlgorithmicMedia` plus a Content Seal generation id
recorded in the manifest. A four-head control with a `microsoft_native` class
(83 issuer-verified rows) failed first: repeated-split argmax recall 0.38 mean
with 0.16-0.56 spread, because Designer routes between renderers and the class
is mixed. The same recipe with `meta_muse_image` in its place holds: argmax
recall 0.86 mean (0.75-1.00) on just 66 images, openai 0.952 and google 0.956
unharmed, and the frozen cross-carrier cells keep their renderer semantics
(Bing+DALL-E 119/146 openai, Microsoft+Imagen 32/36 google) with only small
meta leakage (14 and 3 argmax rows). The class label names what the corpus is:
muse-image-1.0 API output, not the unverified assumption that the consumer
Imagine feature renders identically. The full-data model calls 59/66 meta rows
correctly, with seven leaking to google and none to openai. The photo-first
0.50 margin does not transfer to a 66-image class (0.398 mean recall); argmax
is the honest operating rule until the corpus grows. Muse Image is therefore
a separable fourth provider class on pixels, unlike the Microsoft front-end.
Artifacts: `meta-muse-corpus-2026-08-26/`, `four-head-provider-2026-08-26/`,
`meta-provider-cells-2026-08-26/`.

A paired chat-vs-API check then closed the label question. Two prompts from
the API grid (weathered fisherman portrait, white-sneaker product shot) were
submitted to the consumer Meta AI chat imagine flow in the user's own logged-in
browser session; the chat delivered 1280x1920 and 1920x1280 WebP files from the
`t39.105495-1` CDN family, each carrying IPTC `trainedAlgorithmicMedia` in the
downloaded bytes. Scored by the frozen API-trained four-head model, which had
never seen a chat image: both chat rows landed on `meta_muse_image` under
argmax (0.251 and 0.233) with score profiles matching their API twins almost
exactly, and the margin rule agreed pairwise (fisherman meta at margin, sneaker
no_ai at margin 0.50, same as its API counterpart). The chat imagine pipeline
and `muse-image-1.0` are therefore pixel-indistinguishable to the provider
classifier on this pair, so `meta_muse_image` honestly names both
distributions. Caveat: n=2 paired prompts from one session; this is a
consistency result, not a deployment-scale equivalence claim. Artifact:
`meta-chat-check-2026-08-26/`.

Collapsing OpenAI and Gemini into one pixel class versus other generators
does not fix that. Binary ridge AUC 0.686, TPR 75% at FPR 45%. Canva 98%,
Microsoft 75%, Firefly 68% leak into the union; FLUX HF hold stays out at
3%. Training the same union only against photographs recreates Model 1 with
a narrower train set (fresh FPR 2.1%, Firefly still 95%).

`provider-report-clip-l-ft.json`, `provider-union-report.json`.

### Strict abstention does not recover provider identity

An `unknown` class makes the claim narrower, but neither existing
representation has a useful high-precision operating point. A
leave-one-foreign-family-out ridge on the 124-d residual bank set each cut
above every calibration negative. Mean OpenAI recall fell to 1.3% and mean
Google recall to 4.4%, while a held-out foreign family was still attributed to
a provider up to 5.0% of the time and photographs were called OpenAI up to
0.52% of the time.

The CLIP-L-ft head has the same failure at a 0.1% calibration-negative cut.
On 400 disjoint queries per provider, OpenAI recall was 1.0% and Google recall
was 0.5%. It still called 1/11 held-out FLUX images OpenAI and 1/750 COCO
photographs OpenAI. Zero calibration false positives reduced both provider
recalls to zero. These are abstaining content classifiers, not precise source
finders.

A nonlinear head and local-neighbor decision do not expose a hidden provider
partition in the same embeddings. Across five balanced 768-128-3 MLP seeds,
strict test recall ranged from 0.5-7.0% for OpenAI and 0.75-1.0% for Google,
with 6-19 false attributions among 6,550 locked negatives. The best
calibration-selected `k=3` local-neighbor rule reached 0.5% OpenAI and 3.25%
Google recall and still called one TC260 image Google.

### Pixel finetuning learns the train split, not an exact provider boundary

The last two CLIP-L vision blocks were then finetuned directly for
`openai` / `google` / `unknown`: 9,063 fit images, 3,537 disjoint calibration
images, 400 balanced steps, and random JPEG 40-95, 85-100% crop, and mild blur.
Each provider cut was placed above every calibration negative. Calibration
recall was 4.3% OpenAI and 5.2% Google.

The time-disjoint locked result was 6/400 OpenAI and 10/400 Google. One Google
image and one TC260 image were called OpenAI. All 500 unseen-AI controls and
all 4,945 locked photographs stayed `unknown`, including 3,000 fresh Open
Images, but that photo specificity does not repair an AI-source error. An
oracle cut above both locked OpenAI errors leaves only 1/400 OpenAI; it is an
upper bound, not a valid post-test threshold. The model is not shippable.

The independent high-frequency route is already closed at the tested
capacity. A four-layer opponent-residual patch CNN reached AUC 0.44-0.56
against foreign generators, reversed to 0.15 on a fresh era, and accepted
95-100% of several held-out Firefly, Microsoft, fal.ai, and PixelBin families
at its photo-median threshold. It learned AI rendering versus photography,
not vendor identity.

### External surrogate and forensic-descriptor audit

The public
[`newideas99/gpt-image-synthid-detector`](https://github.com/newideas99/gpt-image-synthid-detector/tree/5495e09)
does not supply a causal SynthID contrast. Its negatives are lightly
regenerated positive images, so the trained ResNet/EfficientNet ensemble can
read the regeneration pipeline. On a blind 517-file local pilot, OpenAI versus
all AUC was 0.630. At the repository's 0.5 cut it retained 92/100 OpenAI and
accepted 307/417 negatives, including 104/120 Open Images, 26/30 COCO, 34/50
Google, and 8/10 Firefly. A later exact repeat on the hash-disjoint v7 challenge
retained 172/200 OpenAI but accepted 110/200 Google. It therefore fails source
specificity before any photograph gate is considered. It is a visual-domain
classifier, not an independent confirmation signal.

The current [`aloshdenny/reverse-SynthID`](https://github.com/aloshdenny/reverse-SynthID/tree/b110836)
V4 codebook also adds no useful hybrid evidence. A pickle-free exact inference
repeat on v7 accepted 77/200 Google and 76/200 OpenAI at its published 0.52
threshold. Applied only to v11 `unknown` rows, that threshold would rescue 26
Google files while misrouting two OpenAI files. The older V3 published cut
would add two v11 Google misses, but it previously accepted 5/499 controls and
6/1,000 fresh Open Images. A 1%-recall OR rule with that measured false-positive
history is also rejected.

The public [`Ristellise/REGRET`](https://github.com/Ristellise/REGRET/tree/7d449034bf323987e7e608e7886e029ed20fd847)
SPAM model is another forensic descriptor, not a decoder. The audited pickle
contained only an sklearn pipeline, scaler, logistic regression, and numeric
numpy globals; inference used an exact restricted allowlist. At the published
0.5 cut it accepted 139/200 Google and 141/200 OpenAI. On the disjoint public
extension, the same published cut also accepted 214/600 ImageNet photographs,
14/75 BigGAN, 17/75 Midjourney, 52/75 SDXL, and 10/75 VQDM. It adds no safe v11
rescue.

[`vordme2010/synthid-dataset`](https://github.com/vordme2010/synthid-dataset/tree/133a27088f6f4d695c79db9a1a70fa8e7fa3adad)
publishes a useful flat-field corpus but an invalid open-world classifier
contrast. Its Tier-1 matrix has 500 Gemini-flat positives and 1,500 synthetic,
spectrum-matched, or phase-scrambled negatives, with no real negative. The
33 features include noise scale and radial power as well as six hand-selected
carrier bins. Rebuilding the repository's seed-42 RBF SVM from the safe numeric
matrix, without loading joblib, accepted 1/200 current Google and 0/200 OpenAI
on v7. The reported AUC above 0.999 measures the synthetic negative recipe and
flat renderer epoch; it cannot confirm the current source finder or a SynthID
payload.

[Forensic Self-Descriptions](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25/tree/50f2eae)
(CVPR 2025) is a genuinely different representation: constrained prediction
residuals are summarized as a 960-d forensic descriptor. Its ready attribution
head is not usable here. Three current OpenAI files were all called `Real`,
while one Microsoft file was called `GPT-Image 1/1.5` at confidence 0.863.

A custom source head on a native 256-pixel center crop was more informative.
PCA-64 plus logistic regression, with each provider cut above every calibration
negative, gave OpenAI AUC 0.872 and 3/50 strict OpenAI recall with 0/388 test
false attributions. Its Google head failed on Bytedance, Canva, FLUX,
Microsoft, and an Open Images photograph. On a later unused-hash challenge the
OpenAI FSD head alone accepted 14/300 OpenAI, 3/300 TC260, and 3/90 PixelBin.
FSD is therefore an independent source cue, not a sufficient classifier and
not a watermark statistic.

### Rejected narrow two-signal cascades, 2026-08-23

The visible Gemini sparkle is absent from the target generation and is not a
feature or a gate. Two metadata-free intersections did survive a new challenge:

| Cascade | Development | Unused-hash challenge |
| --- | --- | --- |
| Google provider probability >= 0.8876249 **and** registered-v3 lattice >= 1.0 | 3/45 Google, 0/30 lattice-positive foreign files | 3/300 Google, 0/690 OpenAI/TC260/PixelBin |
| OpenAI provider probability >= 0.8822824 **and** FSD probability >= 0.9912320 | The two cuts were frozen independently | 3/300 OpenAI, 0/690 Google/TC260/PixelBin, 0/500 photographs |

The Google result is a low-recall intersection of a content head and an
origin-locked renderer lattice. A two-pixel crop destroys the lattice, so an
indeterminate result says nothing about SynthID. It is evidence for a narrow
Gemini/Imagen pipeline, not a public decoder.

All three OpenAI challenge positives were 1086x1448 RGB PNGs, the measured
ChatGPT export geometry. Thirty known non-OpenAI files at exactly 1086x1448
(TC260, Apple, Microsoft, Canva, Firefly, Google, and others) produced zero
first-gate positives. A scan of 1,194 unique files at that geometry with no
provider or platform label produced 29 pixel-gate candidates and four final
hybrid candidates. Three had only PNG image-data chunks; the fourth carried
generic DPI/orientation fields but no provider or provenance label. Visual
inspection found four plausible AI generations. They remain candidates, not
ground truth, until a held-out oracle or original source confirms them.

The full luma 16-32-pixel phase-ring head is not a third confirmation. It
rejected all three known and all four unlabeled OpenAI hybrid candidates while
accepting 3/30 known non-OpenAI same-geometry controls. Its global test AUC was
0.600 for OpenAI and it stays closed.

These cascades established that independent intersections could suppress the
observed false positives, but about 1% recall is not useful for the source-hunt
task. They are rejected as operating detectors. The FSD implementation and
weights are also research-only under CC BY-NC-SA 4.0.

Local artifacts: `lattice-pixel-hybrid-challenge.json`,
`fsd-openai-challenge-report.json`,
`unlabeled-1086x1448-openai-hybrid-candidates.json`. The research entry point
is `.local-eval/synthid/ai-photo-2026-08-22/hybrid_source_finder.py`.

### Canonical forensic source finder, 2026-08-23

An overlooked pixel representation materially improves the strict result. It
normalizes each file to 512x512, then extracts 1,794 metadata-free features
from opponent-color residuals: patch power and variation, phase coherence,
autocorrelation, low-bit correlations, and residual-error histograms. The
OpenAI branch intersects its canonical forensic score with a weak provider
pixel vote. The Gemini branch intersects a forensic/provider fusion with the
registered-v3 lattice. The visible Gemini sparkle remains excluded.

The single-scale cuts were frozen before a third hash-disjoint challenge. The
result was:

| Challenge cell | OpenAI candidate | Gemini candidate | Any candidate |
| --- | ---: | ---: | ---: |
| OpenAI | 51/300 | 1/300 | 52/300 (17.3%) |
| Google | 0/300 | 34/300 | 34/300 (11.3%) |
| Open Images photographs | 1/1,000 | 0/1,000 | 1/1,000 (0.1%) |
| TC260 | 0/7 | 0/7 | 0/7 |

For the requested union, 86/600 OpenAI-or-Google targets were found, 14.3%
recall instead of the rejected cascades' 1%. One OpenAI file crossed through
the Gemini branch, and one known photograph crossed through the OpenAI branch.
The seven remaining unused TC260 files are too few to establish specificity,
and no fresh PixelBin, Firefly, Microsoft, or other generator hashes remained
outside the earlier model and challenge manifests.

This v3 result established a strict source-candidate finder, not an exact
identifier or a SynthID decoder. It does not recover a payload, and an
`unknown` result says nothing about whether SynthID is present.

Local artifacts: `source-finder-v3-selection.json`,
`source-finder-v3-challenge.json`, `google-canonical-forensic.json`, and
`google-forensic-pixel-fusion.json`. The same local research entry point now
implements this rule without the FSD dependency.

### Multiscale strict finder, 2026-08-24

The 512x512 normalization was discarding usable scale-specific evidence. A
frozen union now takes the strongest OpenAI forensic response over 512, 640,
768, and native views, gated by the provider pixel head. The Google branch
takes the strongest forensic/provider fusion over 384, 512, 640, 768, and
native views, then requires the same registered-v3 lattice. Development on
the second and third challenges found 539/1,200 targets and 0/1,657
photograph/TC260 controls.

The views and cuts were written before opening a fourth challenge whose hashes
were disjoint from every model manifest and the first three challenges:

| Challenge cell | OpenAI candidate | Gemini candidate | Any candidate |
| --- | ---: | ---: | ---: |
| OpenAI | 92/300 | 11/300 | 103/300 (34.3%) |
| Google | 0/300 | 173/300 | 173/300 (57.7%) |
| Open Images photographs | 0/1,000 | 0/1,000 | 0/1,000 |
| TC260 | 1/25 | 0/25 | 1/25 |

For the requested union, the blind result is 276/600, 46.0% recall, with
1/1,025 non-target candidates. This is 3.2 times the single-scale v3 recall
and 46 times the rejected 1% cascades. The one false candidate is TC260, not a
camera photograph. Eleven OpenAI files crossed through the Gemini branch;
that is a provider-attribution error but still a correct hit for the declared
OpenAI-or-Google union.

This remains a source-candidate finder, not an exact identifier or a SynthID
decoder. Fresh unused paths from the other generator families were not
available for v4, so the 0.1% observed non-target rate is not an open-world
precision claim. Robustness to crop, resize, re-encoding, and screenshot
capture is also not established. Keep the models and paths in `.local-eval`;
do not add a runtime or public CLI until a new temporal challenge with fresh
foreign-generator families establishes positive precision.

Local artifacts: `source-finder-v4-selection.json`,
`source-finder-v4-rule.json`, `source-finder-v4-challenge.json`, and
`multiscale-forensic-development.json`. The local research entry point
implements the frozen multiscale rule and still uses no metadata or visible
sparkle.

A post-hoc OR over every per-view zero-development-error OpenAI cut is
rejected. It raised v4 OpenAI recall to 153/300 but also accepted 6/1,000
photographs and 3/25 TC260 controls. The apparent union of many individually
strict cuts was multiple-testing overfit, not additional independent evidence.

### Original-export hybrids, 2026-08-24

Three more hash-disjoint challenges tested whether multiscale fusion could be
made useful without metadata. The v5 ExtraTrees union improved exact provider
recall to 363/600 (60.5%) and provider-union recall to 373/600 (62.2%), but it
also accepted 8/1,000 photograph and foreign-generator controls. A revised
Google confirmation removed those eight development errors. Adding an
AI-versus-camera gate in v6 did not transfer: exact recall fell to 327/600
(54.5%) and union recall was 341/600 (56.8%).

The specificity failures exposed a stronger but narrower signal. Current
OpenAI exports in these sets are PNGs produced with adaptive scanline filters,
while the earlier TC260 error was a PNG encoded with filter zero on every row.
A strict PNG parser now requires a non-interlaced PNG with at least one adaptive
filter before the OpenAI branch can emit a result. This reads the image container
and pixels, not EXIF, C2PA, a filename, or a visible label. It also changes the
claim: a re-encoded OpenAI JPEG must abstain.

The complete frozen v7 rule reached 215/400 exact provider matches (53.8%) and
221/400 provider-union matches (55.3%) on a new challenge. Its cells were
114/200 exact OpenAI and 101/200 exact Google. The PNG gate repaired the
observed specificity problem, but the old OpenAI forensic head remained the
recall bottleneck.

A subsequent v8 development hybrid trains an ExtraTrees OpenAI head on v4-v5
multiscale forensic scores, pixel probabilities, and PNG encoding structure.
Model selection used v6. The final 0.47 precision cut was chosen after v7 had
been opened, so the following is a post-hoc development measurement, not
another blind result:

| v7 cell under v8 development rule | OpenAI | Gemini | Unknown |
| --- | ---: | ---: | ---: |
| OpenAI | 190/200 | 3/200 | 7/200 |
| Google | 0/200 | 102/200 | 98/200 |

That is 292/400 exact provider matches (73.0%) and 295/400 provider-union
matches (73.8%).

The remaining Google miss set contained two different export pipelines: PNG
and JPEG. A second development branch parses only JPEG codestream parameters,
including quantization tables, chroma sampling, and progressive encoding; it
explicitly skips APP0-APP15 and COM segments. Training one Google model per
encoding class on v4 and selecting zero-validation-error cuts on v5-v6 raised
the v11 transfer result to:

| v7 cell under v11 development rule | OpenAI | Gemini | Unknown |
| --- | ---: | ---: | ---: |
| OpenAI | 190/200 | 4/200 | 6/200 |
| Google | 0/200 | 126/200 | 74/200 |

This is 316/400 exact provider matches (79.0%) and 320/400 provider-union
matches (80.0%).

This is the best local source finder in the campaign, but it is still not a
SynthID detector, payload decoder, or open-world precision proof. The v8 rule
is post-hoc, and neither v8 nor v11 has an independent open-world negative
proof. The OpenAI branch is intentionally scoped to original-style PNG exports.
A new temporal blind challenge with new foreign generators and PNG
camera/editor controls is required before a runtime or public CLI is justified.

Local artifacts: `source-finder-v7-selection.json`,
`source-finder-v7-challenge.json`, `source-finder-v8-rule.json`,
`source-finder-v8-openai-extra-trees.joblib`, and
`source-finder-v11-google-per-codec.joblib`.

### Published few-shot attribution also fails the open-world gate

[OmniDFA](https://arxiv.org/abs/2509.25682) is a purpose-built few-shot source
attributor rather than a generic content embedding. Its published `part1`
checkpoint is the correct unseen-generator fold for DALL-E 2 and DALL-E 3:
those generators are in `part1` validation and absent from its training list.
The same checkpoint has seen Imagen, so its Google result is not a clean
unseen-Imagen benchmark; the OpenAI result is sufficient to reject the shared
runtime.

With 20 support images per provider and provider-specific similarity plus
margin cuts calibrated to zero false attributions over 160 negatives, a
content-hash-disjoint 745-image evaluation produced:

| Cell | Result |
| --- | ---: |
| OpenAI recall | 9/50 (18%) |
| Google recall | 4/50 (8%) |
| Microsoft called OpenAI | 3/15 (20%) |
| Kodak called OpenAI | 3/24 (12.5%) |
| Canva called Google | 1/15 (6.7%) |
| fal.ai called OpenAI | 1/15 (6.7%) |
| xAI called OpenAI | 1/15 (6.7%) |
| unseen Higgsfield called OpenAI | 1/11 (9.1%) |
| fresh Open Images / COCO false attributions | 0/100 / 0/50 |

Provider multimodality is not the missing fix. Choosing 1-10 spherical
prototypes only by calibration recall selected five: test recall fell to 16%
OpenAI and 4% Google, while false attributions remained on Firefly (2/15),
Kodak (2/24), Microsoft, ByteDance, TC260, and Made-with-AI samples.

Native files already fail, so JPEG, resize, crop, and screenshot variants were
not run for OmniDFA. Do not add a provider-attribution runtime or CLI from that
model. General exact OpenAI/Gemini identification remains unsupported. The
strict source finder above emits candidates; it does not read the SynthID
payload.

### 124-d lattice as pipeline ID, not a vendor CLIP head

Provider-class ridge on 124 native residual features (70/30 once, not a
watermark gate). OpenAI L1 n=285, Google corpus n=533, foreign n=218, COCO
n=289: OpenAI vs COCO 0.965; Google vs COCO 0.999; OpenAI vs Google 0.989;
OpenAI vs foreign 0.725; Google vs foreign 0.922. OpenAI vs Firefly-class
is the weak cell.

One-vs-rest: Google head at TPR 90% has FPR 0% vs COCO, 14% vs foreign, 2%
vs OpenAI. A Gemini-like pixel class is close to what `pipeline_lattice`
already is. An OpenAI-like pixel class on this bank would label Firefly as
OpenAI about half the time and is not shippable.

Three-class `openai` / `google` / `no_ai` on 2,000 catalog OpenAI, 2,000
catalog Google, and 1,936 COCO photos. Photo-first margin 0.50: openai
74.7%, google 78.9%, no_ai 99.8%; Kodak 24/24 `no_ai`. Other generators
are leakage, not classes:

| Platform | n | openai | google | no_ai |
| --- | ---: | ---: | ---: | ---: |
| Firefly | 106 | 37 | 27 | 42 |
| Microsoft | 117 | 39 | 22 | 56 |
| PixelBin | 90 | 14 | 46 | 30 |
| HuggingFace job | 82 | 3 | 62 | 17 |
| ByteDance C2PA | 86 | 5 | 35 | 46 |
| SD / Comfy | 120 | 30 | 11 | 79 |
| fal.ai | 98 | 11 | 18 | 69 |
| Made-with-AI tag | 115 | 4 | 32 | 79 |
| TC260 | 118 | 1 | 13 | 104 |
| xAI | 114 | 1 | 20 | 93 |
| Canva | 76 | 11 | 3 | 62 |
| Apple Clean Up | 114 | 1 | 23 | 90 |
| Aweme | 37 | 2 | 7 | 28 |
| FLUX | 11 | 0 | 0 | 11 |
| Reve | 10 | 0 | 0 | 10 |
| NovelAI | 9 | 0 | 0 | 9 |
| Higgsfield | 11 | 1 | 5 | 5 |

PixelBin and HuggingFace jobs lean `google` (shared renderer lineage).
FLUX, NovelAI, and Reve stay `no_ai`. Local probe:
`uv run python .local-eval/synthid/prc-oklab-attack-2026-08-15/classify_openai_gemini.py image.png`.

## Research lattice expert (google-lineage renderer)

Not a watermark and not in `identify`. `scripts/synthid_runtime/`
`detect_synthid` re-check on 628 frozen holdouts, seed 20260822, threshold
1.0.

| Family | n | detected | rate | max score |
| --- | ---: | ---: | ---: | ---: |
| Google / Gemini | 80 | 45 | 0.56 | 3.03 |
| Firefly | 84 | 15 | 0.18 | 3.00 |
| PixelBin | 80 | 11 | 0.14 | 2.48 |
| Microsoft | 60 | 2 | 0.03 | 2.40 |
| OpenAI | 80 | 1 | 0.01 | 1.38 |
| xAI | 40 | 1 | 0.03 | 1.26 |
| FLUX HF | 40 | 0 | 0 | 0.85 |
| TC260 | 40 | 0 | 0 | 0.97 |
| Kodak | 24 | 0 | 0 | 0.49 |
| Open Images fresh | 60 | 0 | 0 | 0.64 |
| COCO hold | 40 | 0 | 0 | 0.74 |

Firefly 18% and PixelBin 14% match the 2026-08-16 signed-foreign rates
(24% and 14%) in order of magnitude. Both Microsoft hits have issuer
`Microsoft, Google LLC`. A 2 px crop killed every sampled positive,
including Firefly and PixelBin. Google TPR 56% is mixed collection eras, not
the oracle-positive 147/148 cell. Honest name:
`google_lineage_renderer` = Gemini/Imagen + Firefly + PixelBin.

Registered-v3 photographic controls remain 0/5,993 Open Images and 0/2,366
COCO. Against 223 C2PA-named non-Google generators on 2026-08-16: 29
accepted (0.130), Firefly 0.241. `.local-eval/synthid/lattice-check-2026-08-22/`.

## Local collection note (2026-08-21)

Unlabeled rows in the local C2PA-labeled collection are not photographs.
Microsoft and Firefly rows also carry `synthid_from_provenance=true`, so
that flag is not an OpenAI-plus-Gemini class.
