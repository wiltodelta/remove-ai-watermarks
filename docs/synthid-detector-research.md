# SynthID local detector research

> Audit correction, 2026-09-08: multi-period affine-probe results from
> schema 13 or earlier used confirmation patches during period selection
> (R04). They require rerunning and recalibration under schema 14; the
> frozen runtime detector uses separate code and its thresholds are unchanged.
> External-negative claims admitted without explicit independent evidence
> (R08) also need an evidence audit before reuse.

> Research archive for the hunt for a local, keyless SynthID pixel detector.
> Not a statement of current product capability. Shipped behavior:
> [supported signals](supported-signals.md) and
> [known limitations](known-limitations.md).
>
> Sister pages: [SynthID source classifiers](synthid-classifiers.md),
> [SynthID mark removal](synthid-removal-research.md),
> [mechanism reference](synthid.md). Dated measurements live in the
> [chronological plan](synthid-detector-removal-plan.md).

## Result

The hunt for a local, keyless OpenAI SynthID pixel detector is open.
Google has no public payload decoder, and nothing in this project
reproduces one. Closed routes below failed to read the official oracle
contrast on photographs. The remaining work is a new waveform, a new
pair mint, or a new statistic, not a rerun of those routes.

The mark behaves like a keyed spread-spectrum residual: a content-adaptive
`x' = x + g(x)` with a paired decoder (DeepMind patent family, optional
secret in intermediate layers, encoder/decoder ensembles that do not
recognize each other). Without that pair, the residual sits under the
scene. Keyless energy, TrustMark transfer, L1 distillation, a flat-field
stamp, and a 16-32 band student all failed to read the official oracle
contrast on photographs.

S4, 2026-08-15: the two providers are not doing the same thing. Cross-image
correlation of the folded residual is high for Google (tile16 pos-pos
`+0.326`, chance `0.036`) and at chance for OpenAI (`+0.032`). Google
has phase-coherent residual structure also correlated with its controls;
normalized correlations do not establish an absolute amplitude ratio. OpenAI
shows no shared residual above chance in this measurement, consistent with a
content-dependent post-hoc encoder. Comb experts that work on
Google therefore cannot be reused as an OpenAI watermark detector.

M2, 2026-08-15, re-verified 2026-08-16: the tested Google generation APIs
exposed no encoder-off sibling. `imagen-*` ids 404 from the model garden. `gemini-*-image`
rejects `addWatermark` (`Cannot find field`). Current Gemini API docs
state that all generated images include a SynthID watermark. There is no
encoder-off Google path.

What the runtime product uses for the *watermark* is signed provenance. The
official OpenAI verifier and the periodic-lattice expert are development-only;
the latter lives under `scripts/synthid_runtime/` and is not called from
`identify` or the CLI.
Lineage measurements of that expert are in
[SynthID source classifiers](synthid-classifiers.md).

## Photographic edit pairs, 2026-10-07 UTC

A real photograph and its AI-edited copy provide a useful source/output pair.
They do not provide the same rendered pixels with the watermark encoder toggled:
their difference also contains editing, regeneration, color processing, and
export losses. A small requested edit does not establish a small pixel change.

Google's live [model reference](https://ai.google.dev/gemini-api/docs/models/gemini-nano-banana-2.1)
names the October update `gemini-nano-banana-2.1`. Its
[editing guide](https://ai.google.dev/gemini-api/docs/image-generation)
documents semantic masking. This pilot supplied a binary mask as a second
reference image, not a documented hard-mask parameter; preservation outside
the mask was measured rather than assumed.

One privately retained photographic source produced six independent API
responses: two requests with no intentional edit, two local removals with a
0.334% mask, and two with a 1.519% mask. All used the same EXIF-normalized,
once-resized 1536 x 2752 PNG input and the documented 2K output grid. All six
returned 1536 x 2752 JPEGs. Each was inspected for visible provider marks;
oracle preparation stripped provenance while proving decoded-pixel parity.

The official Gemini verifier returned `not_detected` for the prepared source
and `detected` for all six outputs, including both no-intentional-edit exports.
The first account returned the source negative and three output positives,
then explicitly refused the next file because its similar-image quota was
exceeded. The second existing account returned three remaining output
positives. Preserve that refusal separately. Modes differed, Flash-Lite versus
Flash, so this is file coverage across named slots, not an account agreement
or repeatability test. The immutable account batches remain intentionally
partial, 5/7 and 3/7 recorded rows; unused rows were never submitted. A third
prepared account batch remained entirely unsubmitted. Their hashes verify,
while `--require-complete` correctly rejects the partial account batches.

For the four masked outputs, 56.05-69.33% of pixels outside the requested mask
plus a 32-pixel guard changed by more than three 8-bit RGB channel levels.
The no-intentional-edit exports changed 68.31-70.55% by that criterion.
In a fixed sky region, output/source mean absolute channel differences were
1.868-2.100 levels, versus 0.082 for JPEG shams matching output quantization
tables and chroma sampling. Full-image PSNR was 19.96-20.57 dB. These copies
establish watermark presence but do not isolate its residual from regeneration.

Reanalysis of the earlier Google Photos photographic pairs found much smaller
sky differences, about 0.328 channel levels. Sky residual correlation between
the small removal, recoloring, and large Magic Eraser exports was
0.9951-0.999993. The ordinary crop control aligned at an integer source offset
of 840 rows; its overlapping-sky residual correlated only about 0.180 with the
AI-export residual. Simple JPEG shams did not reproduce that AI residual.
The historical oracle labels remain unchanged: large eraser positive, small
edits inconclusive or account-dependent. This is a candidate shared AI-export
component, not an attributed SynthID template. All comparisons have one parent
photograph; repeated outputs do not create independent scene evidence.

Next discriminating experiment: independent photographic parents, a measured
local-edit pipeline, same-grid non-AI export controls, and frozen official
watermark labels. Search for residual structure that survives the controls
and held-out parents. Retain private captures, requests, responses, hashes,
pixel measurements, and oracle evidence outside the public repository.

### Fixed 8-pixel output component, 2026-10-07

A continuation froze the two no-intentional-edit outputs as discovery inputs
and the four masked outputs as confirmation inputs. All still share one parent.
A fixed sky box, quadratic coordinate/source-RGB detrending, sigma-2 Gaussian
high-pass, and channel-wise source-noise regression separated smooth color
changes and a camera-noise confound from a periodic output component. Raw
pair correlations were 0.540-0.746, high-pass correlations 0.335-0.425, and
source-orthogonal correlations 0.271-0.365. Source subtraction itself creates
shared negative camera noise; regression is diagnostic and may also suppress
content-adaptive watermark components.

The source-orthogonal 8-pixel discovery template had RMS 0.496 RGB levels and
correlated 0.987-0.993 with the four confirmation outputs. Its 8-pixel repetition
explained 98.83% of the 64-pixel folded template's energy, not the full residual.
The 16/32/64-period matches therefore repeat largely the same component.
Twenty-four fixed JPEG/blur controls, using each output's quantization and
sampling with blur sigma 0/0.5/1/2, correlated only 0.118-0.168. They do not
reproduce the complete internal export pipeline.

Six further requests were frozen before generation: four independent
algorithmic references and two exact circular shifts of the same photograph.
One noisy-reference upload ended in a write timeout, with unknown remote
completion; no usable output or retry was admitted. Three gradient outputs
matched the frozen template at 0.7660-0.8604, versus 0.0975-0.1054 for matching
JPEG controls. These references test content diversity, not independent
photographic scenes. The one-pixel photographic shift matched at fixed output
phase, 0.9726, but not source-following phase, -0.2888. The (4,4) shift was
non-discriminating because the template itself is nearly invariant to that shift.
The component is visible in smooth sky patches; detailed foreground patches
are not reliable confirmation regions.

A frozen seven-file official-verifier batch tested one measured-amplitude
transplant onto the prior negative photograph, one untouched gradient, three
generated gradients, and two generated shifted photographs. The existing
account returned two photographic `detected` results and five `indeterminate`
results. The transplant's no-reliable-origin-signals wording was not an explicit
SynthID negative; gradient responses included too-little-information abstentions.
The batch passed `verify --require-complete`, 7/7, and two prepared fallback
account batches remained unsubmitted. No scale search or oracle-adaptive pixel
changes occurred. Positive photographic labels establish coexistence with
SynthID, not attribution or decoder sufficiency of the extracted template.

Retain this as an output-coordinate pipeline component. It is not a calibrated
SynthID detector or an attributed watermark template. Numeric tiles, private
inputs, API payloads, measurements, and hash-bound oracle evidence remain
outside the public repository. Same-renderer encoder-off controls or a
predeclared isolated-component oracle contrast are still needed for attribution.

### OpenAI API pairs and template sufficiency, 2026-10-07

Seven separate `gpt-image-2` Image API edit requests used fixed 1536 x 2752
PNG output, medium quality, and n=1: two no-intentional-edit copies, two identical
alpha-masked local removals, two independent algorithmic gradients, and one
exact one-pixel source shift. All returned the native grid, with no output
resizing or JPEG loss. A preceding built-in edit had an unreported model and
937 x 1678 output, so it was preserved separately and excluded from the API
cohort. Generation used the unmodified bundled CLI; request previews and
outputs were retained, but that CLI does not archive full generation responses.

The same fixed photographic sky region, detrending, high-pass, and source-noise
regression found an 8-pixel discovery template with RMS 0.36559 RGB levels.
It correlated 0.99749/0.99719 with the two masked confirmation copies. Its
8-pixel repetition explained 97.73% of the 64-pixel folded template's energy,
not the entire residual. Seven nonzero fixed blur/PNG/JPEG controls correlated
-0.311 to 0.041; an exact PNG control had zero residual and undefined correlation.
The photograph remained the sole real scene parent. Sky MAE was 3.428-6.417
RGB levels, and 80.26-89.07% of pixels outside the requested exclusion changed
by more than three levels. Small instructions still do not isolate watermarking.

The shifted photographic probe matched at fixed output phase, 0.99731,
versus -0.04495 at source-following phase. Independent gray/blue gradients
matched only 0.42590/0.45460, with folded RMS 0.01723/0.02469, much smaller than
the photographic template. These are not independent photographic scenes.
The OpenAI and Nano Banana 8-pixel templates correlated -0.04302 on the shared
parent; equal periods do not establish equal patterns or watermark attribution.
Explicit phase averaging and two independent PNG decoders verified the
measurement implementation.

A frozen ten-file batch used the official OpenAI Content Provenance API.
All seven generated outputs returned SynthID `detected`. The original
photograph, untouched gray gradient, and one measured-amplitude template
transplant onto the photograph returned `not_detected`. Preparation removed
AI metadata with pixel parity; the adapter's second sanitization was verified
byte-identical to each frozen upload. C2PA was not used as a verdict fallback.
The primary batch passed `verify --require-complete`, 10/10; two prepared
fallback batches remained unsubmitted. Full verification responses and hashes
remain privately retained, separate from generation-response limitations.

The transplant negative rejects sufficiency of this component at its measured
amplitude on that input. It does not prove absence of any relationship to
SynthID. Positive gradient labels despite a much weaker photographic pattern
also prevent calling this template a watermark-presence detector. Retain it
as a reproducible output-coordinate component pending attribution, not a
SynthID decoder or remover. No scale search or oracle-adaptive pixel changes
were performed.

### Independent photographic pilot, 2026-10-07

Eight distinct photographic parents produced 48 successful direct API outputs:
two no-intentional-edit requests and one small masked brightness request per
parent and provider. Models were `gemini-nano-banana-2.1` at 2K and
`gpt-image-2` at medium quality. Every native output retained the shared
1536 x 2752 grid, Google JPEG and OpenAI PNG. This continuation used direct
provider clients, without the built-in image-generation tool or bundled CLI.
The plan, split, prior templates, and expansion gate were frozen before
generation. Full requests, responses, and hashes remain outside the public repo.

Four parents were discovery inputs and four separate parents were confirmation
inputs. Controls comprised each source plus ten ordinary blur/PNG/JPEG exports.
The exact PNG re-save duplicates the source's pixels; these are correlated
controls within a parent, not 44 independent negatives. Inputs were EXIF-normalized,
center-cropped, and resized once; some were upscaled. One discovery parent lacked
camera metadata, so its lineage uncertainty remains explicit.

Blind inference uses only the candidate image: sigma-2 high-pass, an 8 x 8
phase fold averaged across three low-texture 512 x 512 patches from a fixed
grid, and maximum template correlation over 64 circular phases. Patches exclude
the requested edit region. Per-provider thresholds are the maximum discovery
score among controls and the other provider's outputs plus 0.02. Confirmation
rows contribute neither templates nor thresholds. Exactly one admitted provider
produces a label; neither produces `unknown`, and two produce `conflict`.

| Frozen-template confirmation | Correct provider labels | Controls assigned a provider |
| --- | ---: | ---: |
| Google | 9/12 | 0/44 across both providers |
| OpenAI | 12/12 | 0/44 across both providers |

All three Google misses came from one detailed scene. Independently estimated
discovery templates, whether extracted blindly or using source-orthogonal paired
residuals, produced the same confirmation counts. Their correlations with prior
templates were about 0.960 for Google and 0.952 for OpenAI. Each provider's
parent-level mean positive score exceeded the corresponding maximum control score
on all four confirmation parents, one-sided sign-test p=0.0625. Repeated outputs
do not increase that four-parent sample size or establish a production error rate.

Exploratory robustness retained the frozen thresholds. A one-pixel circular
image shift preserved 9/12 Google and 12/12 OpenAI labels. Common JPEG quality-95,
4:2:0 re-encoding reduced them to 0/12 and 6/12 respectively; all misses abstained.
These transformed copies were not oracle-checked. A local score loss is therefore
not evidence of lost SynthID. The measured component remains compatible with an
export or rendering fingerprint. It is not a robust watermark-presence decoder.

Six files were selected for oracle checks before generation. OpenAI's official
API returned `detected` for both selected outputs and `not_detected` for the
source; the primary batch passed `verify --require-complete`, 3/3. Gemini returned
`indeterminate` for a no-intentional-edit output, `detected` for a masked output,
and `indeterminate` for the source. One earlier masked-output attempt reported
no attached media; preserve it as `unreachable`, not a watermark verdict.
The next authorized account returned the positive for the same frozen upload.
Google coverage is 3/3 files across two accounts, while each immutable account
batch has only 2/3 recorded rows. Hash verification passed; strict completeness
correctly rejected those partial account batches. Unused fallback rows were not
submitted. Google used real authenticated Chrome through the CDP driver after
the native browser integration failed.

Masks actually contain 129 x 129 editable pixels because rectangle endpoints
are inclusive. Pixel-change measurements use that actual mask plus a 32-pixel
guard. Even no-intentional-edit outputs change substantial parts of each image:
across all cases, 51.16-96.71% for Google and 55.45-96.47% for OpenAI exceed three
8-bit levels in at least one RGB channel outside the guarded region. This still
is not a same-rendered-image encoder-on/off experiment.

The predeclared expansion requirement was at least 10/12 correct labels for
each provider with zero control false positives. Google failed it, so the larger
generation campaign was not started. Retain the private benchmark, feature
matrix, hashes, and separate origin/oracle labels for further research. Before
expansion, resolve detailed-scene misses and common-export sensitivity on new
parents; do not retune and reuse this confirmation set as an untouched test.

### Codec ablation and fresh API replication, 2026-10-07

A local factorial follow-up fixed each output's original three measurement
patches, preventing patch reselection from explaining score changes. All 24
Google JPEGs had identical quantization tables with every entry equal to one,
and 4:2:0 sampling. Header checks confirmed that Pillow quality-100 tables
matched these native tables. Re-encoding at quality 100 with either 4:4:4 or
4:2:0 preserved 9/12 Google and 12/12 OpenAI confirmation labels. Quality 95
reduced them to 0/12 and 6/12 with either sampling choice. In this ablation,
quantization explains the classification-count loss; changing chroma sampling
alone does not. Subsampling still changes individual scores, so it is not absent
from the measurement. Exact native-table ordinary exports of the sources
did not reproduce strong matches: confirmation maxima were 0.193 for the Google
template and 0.155 for the OpenAI template. This excludes that tested ordinary
encoder as a sufficient explanation, not every rendering-pipeline confound.

Separate JPEG-95 templates and thresholds were fitted using only the four
original discovery parents. On the already-inspected confirmation set they
produced 6/12 Google and 12/12 OpenAI labels with no control assignments. Treat
this as exploratory adaptation, not another independent successful validation.

Four new photographic parents then produced eight direct API outputs, one
no-intentional-edit request per provider and parent. The same models and native
grid were used. Prior and JPEG-conditioned templates, thresholds, and patch
selection were hash-frozen before these requests. Two inputs were upscaled;
one lacked camera EXIF. No new oracle requests were made.

| Fresh confirmation, four outputs per provider | Google | OpenAI | Provider assignments on 12 source-control rows |
| --- | ---: | ---: | ---: |
| Native exports, frozen prior classifier | 4/4 | 4/4 | 0 |
| JPEG 95, frozen prior classifier | 1/4 | 3/4 | 0 |
| JPEG 95, discovery-only codec refit | 4/4 | 4/4 | 3 |

The refit's three control hits all belong to one photographic parent and include
a pixel-identical PNG duplicate. Their Google scores were 0.497-0.506 against
the frozen 0.482 threshold. That source lacks camera EXIF and has no independent
Google SynthID verdict, so these are disagreements with experimental source-control
origin labels, not adjudicated watermark false positives. Do not tune the
threshold against them or silently relabel them as generated. The codec refit
was not accepted as a production detector.

Across the original and new confirmation sets, the unchanged native classifier
labels 13/16 Google outputs and 16/16 OpenAI outputs, with no provider assignments
on 56 control rows. Those controls have 48 unique decoded images and eight parent
groups; repeated exports remain correlated. The per-parent positive/control score
direction agrees on 8/8 groups for each provider, one-sided sign-test p=0.00390625.
This is origin association on a small selected cohort, not decoded SynthID or a
population error guarantee. Response-byte hashes and both image decoders matched
for all eight new outputs, and original-file hashes remained unchanged. Private
measurements and the fresh benchmark were retained separately from the first
pilot. No shipped detector or removal path changed.

### Clean-source provenance correction, 2026-10-07

Earlier personal photographic controls have unverified edit and export histories.
Camera EXIF, a camera filename, or a local negative score does not establish absent
SynthID. Their previously recorded provider-assignment counts describe origin-label
agreement only. They cannot calibrate watermark false-positive rates. Preserve the
frozen measurements rather than relabeling or overwriting those controls.

For new controlled pairs, the research campaign selected camera-native NEF files
from the official [RAISE dataset](https://loki.disi.unitn.it/RAISE/). Its publisher
states that the photographs were collected directly from three cameras in
2011-2014 without subsequent processing. Sixty downloaded files contained 59
unique byte hashes, including one exact duplicate under different identifiers.
After deduplication, the originals span 38 camera/capture-day groups. Local RAW
development records sensor dimensions, CFA patterns, decoder settings, source
URLs, and original and derived hashes. This supplies historical acquisition
provenance, not a provider oracle verdict or cryptographic proof of photography.

All cases, providers, ordinary exports, and near-event source controls stay in the
same capture-day/camera partition. Selected API parents come from 12 training,
six calibration, and six untouched test groups. No uncertain personal photograph
is admitted as a clean watermark negative. Separating these camera sources from
complete provider outputs remains an origin/export-pipeline task; it does not
causally isolate or decode SynthID.

RAISE permits scientific noncommercial research and personal experimentation.
Retain its copyright, license, source link, and paper attribution beside private
derivatives. A candidate trained on it is not cleared for commercial integration
or model distribution. No shipped classifier or removal path changed.

### RAW-grounded provider candidate, 2026-10-07

At the original freeze, the RAW campaign retained 140 native API outputs: 72 OpenAI and 68
Google. A Google HTTP 429 explicitly reported the project's monthly spending cap.
One request was refused and three further Google requests were not submitted;
only never-attempted OpenAI requests were subsequently completed. Training and
calibration have complete provider coverage. The final Google test has 14 outputs
from five capture-day groups, versus 18 OpenAI outputs from six groups. Do not
impute the missing Google cases or describe the planned campaign as complete.

The negative pool includes 59 deduplicated historical RAW originals, 236 ordinary
local edits (brightness, gamma, unsharp, and JPEG 100 export), and 128
publisher-labeled other-generator images from 64 prompt groups in the pinned
[Qwen-Image-Bench dataset](https://huggingface.co/datasets/Qwen/Qwen-Image-Bench/tree/d2493deb153b020cf169c7e3f57d15e4dd697038).
The latter constrain provider attribution, not SynthID absence. Their FLUX/Qwen
labels come from the publisher's prompt/model mapping. All derivatives share the
original or prompt's partition. Final negatives contain 18 RAW originals from ten
capture-day groups, 72 ordinary edits, and 32 foreign-generator images from 16
prompt groups. Export variants are correlated, not additional independent sources.

Scratch ridge heads on forensic and spectral features did not meet the calibration
gate. Low-texture 512-pixel phase-folded RGB templates reached a minimum primary
provider recall of 14/18 (77.8%). Color-only projections did not improve it.
Whole-frame eight-pixel phase folding improved textured calibration examples and
reached 18/18 per provider on both native and JPEG 95 exports. Training-only
group-balanced templates and calibration-only thresholds were then hash-frozen
before the single-use final test. The test selected no model or threshold.

| Frozen candidate, final test | OpenAI correct | Google correct | RAW/ordinary control provider assignments | Foreign-generator provider assignments |
| --- | ---: | ---: | ---: | ---: |
| Native | 18/18 | 14/14 | 4/90 | 1/32 |
| JPEG 95 | 18/18 | 14/14 | 0/90 | 1/32 |
| JPEG 75 stress | 12/18 | 4/14 | 8/90 | 1/32 |
| Resize to width 1024 stress | 0/18 | 0/14 | 0/90 | 1/32 |

The four native historical-camera control hits are one original and three of its
ordinary edits, all one capture-day group. The foreign-generator hit is one Qwen
image and repeats across exports. JPEG 75 also misattributes one Google output to
OpenAI. These are measured attribution failures; the foreign-generator lineage
alone does not adjudicate its watermark status. The zero-false-assignment gate
failed. The candidate is retained as experimental, not accepted or shipped, and
its threshold was not repaired against this test. Historical clean sources expose
a specificity failure that unverified personal controls could not establish.

Parent-level mean-positive-minus-max-source scores are positive on six OpenAI and
five available Google groups, one-sided sign-test p=0.015625 and p=0.03125.
That paired direction does not cancel the unrelated-control failures. All 140
response-byte/source-request bindings and both decoders matched; all historical
originals and separately retained personal inputs remained unchanged. Further
model selection needs new untouched groups, and fresh same-pipeline Google API
pairs are blocked by the observed spend cap. No library detector or removal path
changed. The private NPZ candidate, callable inference script, immutable result,
license notices, and acquisition evidence are retained for subsequent research.

### Magnitude-aware development follow-up, 2026-10-07

Post-test diagnostics found that the native historical-camera false hit has an
8-pixel component RMS of 0.0124 pixel levels, versus a native Google test median
of 0.2564 (minimum 0.1534). The falsely attributed Qwen image has RMS 0.0930.
Pearson correlation discards this magnitude information. These diagnostics explain
a candidate weakness, not an independently validated remedy.

An absolute matched-amplitude score alone failed calibration: unrelated high-energy
patterns raised the negative ceiling above weak genuine provider examples. A
separate development candidate therefore requires both the original correlation
cut and a train-supported matched-amplitude floor. Its templates remain trained
on the RAW training groups. A floor multiplier was selected using calibration
only; the largest of 1, 0.75, 0.5, and 0.25 meeting 90% primary calibration recall
was 1, giving floors of 0.2421 OpenAI and 0.0642 Google pixel levels.

On the already-consumed test used strictly as a development regression, this
joint candidate labels 18/18 OpenAI and 14/14 Google native results with 0/122
negative assignments. JPEG 95 drops to 16/18 and 12/14, below the 90% criterion;
negative assignments remain 0/122. JPEG 75 and resizing remain poorly supported.
Neither the successful native repair nor a repeated zero-control count is new
independent validation. The original failed result was preserved unchanged.

The separate private NPZ and input-only joint inference script reproduce the
four example decisions, including abstention on the genuine RAW and the Qwen
false hits. The candidate remains unvalidated and unshipped. Next research needs
robust codec-conditioned signal-strength modeling and new untouched source/event
and prompt groups. Same-pipeline independent Google positives remain blocked by
the observed monthly project spend cap. No account limit was modified.

### Independent specificity validation, 2026-10-08

The four missing API outputs were subsequently completed, giving 144/144 outputs
for the earlier generation plan. Its original failed single-use test remains
unchanged; the newly completed cases are development material, not retroactive
independent validation.

A separate candidate selected the amplitude multiplier by maximizing minimum
native/JPEG 95 calibration recall, breaking ties toward the strongest floor.
This chose 0.75 rather than the earlier largest multiplier merely meeting 90%.
The model was frozen before acquiring new inputs. The first new validation used
24 RAW originals from new camera/capture-day groups, 96 ordinary edits, and 64
other-generator images from 32 new prompt groups. All prior RAW hashes, source
days and adjacent days, and foreign prompt groups were excluded. Twelve new
positive source groups contributed 36 outputs per provider, three cases per
source. Export variants and repeated outputs do not increase independent support.

| First new single-use validation | OpenAI correct | Google correct | RAW/ordinary provider assignments | Other-generator provider assignments |
| --- | ---: | ---: | ---: | ---: |
| Native | 36/36 | 36/36 | 0/120 | 2/64 |
| JPEG 95 | 36/36 | 35/36 | 0/120 | 2/64 |
| JPEG 75 stress | 11/36 | 28/36 | 0/120 | 0/64 |
| Resize to width 1024 stress | 0/36 | 0/36 | 0/120 | 2/64 |

The candidate failed specificity acceptance. The false Google assignments are
one publisher-labeled FLUX image and one Qwen image in different prompt groups.
Their native correlations are 0.4960 and 0.4908, with matched RMS 0.3567 and
0.0990 pixel levels. An amplitude gate alone does not resolve these high-energy
pattern matches. All 72 successful response/source bindings and both decoders
matched. A separately retained HTTP 503 refusal was retried once as a new attempt.
Both provider-versus-own-control paired directions are positive on 12/12 source
groups, one-sided sign p=0.000244; this is not independent corroboration of a mark.

These results were then explicitly consumed as development data. A stricter
candidate keeps the templates and 0.75 magnitude floors fixed, raises each
direction cut above every magnitude-qualified non-provider primary score by
0.01, and selects the largest cut on a 0.025 grid retaining at least 90% admission
in each historical development partition and primary condition. The resulting
cuts are 0.90 OpenAI and 0.55 Google. Its minimum development admission is 94.4%
and 91.7%, with no primary wrong-provider assignments. These are development
figures; a second untouched validation is necessary.

The stricter candidate was frozen before a second new acquisition. Another 24
RAW originals and 96 ordinary edits use new capture-day groups; 64 other-generator
images use 32 new prompt groups. All earlier used and unused downloaded RAW days
and adjacent days were excluded. Twelve new API source groups contribute one
output per provider: six no-intentional-edit sources and six small masked edits.

| Second new single-use validation | OpenAI correct | Google correct | RAW/ordinary provider assignments | Other-generator provider assignments |
| --- | ---: | ---: | ---: | ---: |
| Native | 12/12 | 12/12 | 0/120 | 0/64 |
| JPEG 95 | 11/12 | 10/12 | 0/120 | 0/64 |
| JPEG 75 stress | 0/12 | 4/12 | 0/120 | 0/64 |
| Resize to width 1024 stress | 0/12 | 0/12 | 0/120 | 0/64 |

The native condition meets the observed per-condition target, but overall
acceptance still fails because Google JPEG 95 recall is 83.3%, below 90%.
There are no wrong-provider assignments, only abstentions. Zero control errors
span 56 independent source/prompt groups, not 184 independent trials; even the
one-sided 95% zero-failure group bound is 5.2%. These samples do not establish a
rare-population false-positive rate. All 24 response/source bindings and native
decoder outputs matched. Both earlier failed verdicts and all original inputs
remain unchanged. The library has no new detection or removal path.

A subsequent codec-aware development head retains the strict native cut and
uses cuts 0.85 OpenAI / 0.525 Google on JPEG 95, with train-minimum magnitude
multipliers 0.7 / 0.75. Routing reads the questioned file's actual matching JPEG
quantization tables and 4:2:0 sampling; it does not receive an expected provider,
paired source, or benchmark condition label. Other codecs and PNG exports of
JPEG pixels use the strict head. Producer-generated JPEG 95 / JPEG 75 fixtures
verify routing, and real input-only inference reproduces the two repaired misses
and the remaining Google abstention.

On the now-consumed second test, this head reaches 12/12 OpenAI and 11/12 Google
JPEG 95, with 0/184 control assignments. On the consumed first test, JPEG 95 is
36/36 and 35/36, again with 0/184 control assignments. These are post-test
regressions, not independent validation or acceptance. The immutable strict
model/result is retained separately. The broader FiveK validation below subsequently tests this candidate and
exposes additional false assignments. Model derivatives remain private
noncommercial RAISE research.

### FiveK camera diversity and period-two rejection, 2026-10-08

The codec-aware candidate was frozen before acquiring publisher input DNGs from
[MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/), published with the
2011 input/output photographic adjustment study. These are the input DNGs,
not the five expert-retouched TIFF renditions. Actual sensor mosaics, original
capture dates, downloaded byte hashes, and the local non-generative development
were verified. Historical publisher evidence is stronger than a contemporary
file's EXIF alone, but is not a cryptographic guarantee. Both publisher licenses
restrict use to research without commercial advantage; notices accompany the
private originals and derivatives.

The first acquisition covered 60 distinct capture dates and 35 distinct camera
model strings in EXIF. This counts labels, not independently verified physical
cameras or sensors. Two initial Google uploads timed out before returning a
response; remote completion remained unknown and neither request was repeated.
A new eligible source, selected before scoring, completed the 60 primary source
blocks. The two OpenAI outputs from the incomplete block remain diagnostic.
The archive therefore retains 244 attempts, 242 native outputs, and 240 primary
outputs: two cases per provider on 60 sources. Native image bytes and full
semantic request/response payloads are hash-bound; immutable image references
avoid duplicating base64 pixels in the records. PIL/OpenCV native decoding
agrees for all 242 outputs.

Each primary negative condition has 905 variants: 61 normalized historical RAW
renders, 244 ordinary edits, and 600 FLUX/Qwen images from 300 new prompt groups.
There are 361 origin groups, not 905 independent observations. The frozen codec
candidate failed:

| Export | OpenAI correct | Google correct | False provider assignments on 905 negatives |
|---|---:|---:|---:|
| Native | 117/120 | 120/120 | 3 |
| JPEG 95 | 108/120 | 115/120 | 3 |
| JPEG 85 | 7/120 | 77/120 | 1 |
| JPEG 75 | 0/120 | 54/120 | 1 |
| Resize to width 1024 | 0/120 | 0/120 | 2 |
| Crop 8 pixels from each edge | 117/120 | 120/120 | 2 |
| Crop 2 pixels from each edge | 117/120 | 120/120 | 2 |

The three primary false assignments are two Qwen images and one FLUX image
called Google. A separately declared camera-geometry audit developed all 61
RAWs without resizing or cropping. Two M8-source photographs were called
OpenAI: one native false assignment and two after JPEG 95. These views reuse
existing source groups and add no independent groups. Normalizing all clean
controls to the generation resolution had concealed a camera-processing alias.

A bounded official OpenAI API pass checked eight preselected files after
pixel-preserving metadata stripping. The two OpenAI cases, native and JPEG 95,
were SynthID `detected` in 4/4 checks. Historical native/JPEG 95 renders and two
foreign-generator files were OpenAI SynthID `not_detected` in 4/4; C2PA was
absent in all eight prepared uploads. This is an OpenAI-supported-signal result,
not verification of Google's mark or proof of general AI absence. The batch
completed with immutable media and recorded raw replies; no oracle adaptation
was performed.

Pixel fidelity also contradicts a lossless-pair interpretation. Median
outside-mask RGB absolute differences on the 0-255 scale are 11.60/10.40 for
OpenAI no-intentional-edit/masked-edit cases and 14.92/5.69 for Google. In every
case family, the median fraction of outside pixels differing by more than one
level in any channel exceeds 98%. Median whole-image PSNR spans 22.88-28.74 dB.
These unaligned differences include regeneration and compression and cannot
isolate SynthID simply by subtracting the source.

The subsequent development candidate subtracts the entire period-two lattice
subspace from both the 8x8 RGB residual tiles and the fixed development
templates. It uses no camera identity, expected provider, paired original, or
AI metadata at inference. Projected amplitude floors are calibrated against
the original training parents; cuts/floor multipliers use only previously
consumed cohorts, including the newly failed test and full-resolution M8
controls. The strict head conservatively guards foreign pixels reexported as
PNG. JPEG 95 routing still reads the questioned file's actual quantization
tables and 4:2:0 sampling. The original templates are projected, not refit on the
new validation images. All failed candidates and single-use verdicts remain
unchanged.

A new candidate was frozen before another acquisition: 24 previously unused
capture dates, excluding the preceding FiveK dates and their adjacent days,
including four M8 mosaics. Six API parents, including two M8 sources, produce
24 outputs. New negatives comprise 24 normalized RAW renders, 24 full-resolution
renders, 96 ordinary edits, 128 foreign images from 64 untouched prompt groups,
and 128 equal-pixel PNG counterparts of those foreign images. The 400 negative
variants belong to 88 origin groups; codec counterparts share groups. This
small independent test passes its private research acceptance:

| Export | OpenAI correct | Google correct | False provider assignments on 400 negative variants |
|---|---:|---:|---:|
| Native | 12/12 | 12/12 | 0 |
| JPEG 95 | 12/12 | 12/12 | 0 |
| JPEG 85 | 12/12 | 7/12 | 0 |
| JPEG 75 | 12/12 | 5/12 | 0 |
| Resize to width 1024 | 0/12 | 0/12 | 0 |
| Crop 8 pixels from each edge | 12/12 | 12/12 | 0 |
| Crop 2 pixels from each edge | 12/12 | 12/12 | 0 |

The frozen private model checksum begins `7afd80cd`. Its strict native cuts
are 0.45 OpenAI / 0.60 Google, with projected matched-RMS floors
0.055034 / 0.084615. The JPEG 95 head uses cuts 0.50 / 0.60 and floors
0.055034 / 0.028205. These are classifier operating points, not watermark
confidence scores. A standalone offline implementation is checked against all
424 native inputs and 24 actual JPEG 95 API exports, with score parity; neither
API credentials nor the research checkout is needed to run that implementation.

A fixed 2x2x2 factorial evaluates projection, correlation cuts, and amplitude
floors independently. On this fresh cohort, the original candidate already
passes both primary conditions with zero false assignments; the small test
therefore does not establish superior primary recall. With new cuts and floors
but no projection, there are 21 primary wrong-provider assignments, including
seven M8 full-resolution assignments; adding projection removes those errors.
Projection with unchanged cuts/floors misses OpenAI outputs, so recalibration
is necessary. JPEG stress improves OpenAI from 1/12 at JPEG 85 and 0/12 at
JPEG 75 to 12/12 each, while Google's corresponding recall decreases from
11/12 and 9/12 to 7/12 and 5/12. This is a precision/robustness tradeoff,
not an across-the-board improvement.

All six positive groups exceed their own historical/control projected signal
levels for each provider, with a paired one-sided sign-test p=0.015625.
The exact all-success one-sided 95% group lower bound is only 60.7% for six
positive groups. Zero failures across 88 negative groups gives a 3.35%
one-sided upper bound on group failure, conditional on the sampling and
independence assumptions. There is no rare-population false-positive guarantee.

This establishes a private provider/export-pipeline prototype with a small
untouched confirmation. It does not establish SynthID-specific detection,
scale robustness, arbitrary-camera coverage, or commercial deployability.
Remaining targets are a substantially larger untouched grouped validation,
scale-aware signal handling, and source-matched watermark contrasts that hold
the rendering pipeline fixed. No learned weights, retained images, API records,
or inference integration were added to the public library or `identify`.

### Extended validation protocol and bounded oracle evidence, 2026-10-08

A larger confirmation freezes the same projected model and operating points
before selecting 100 fresh historical capture-date groups. Each provider has
two API edit cases per parent. Negative conditions include normalized RAWs,
unscaled RAW development, four ordinary processing controls, foreign-generator
exports, and equal-pixel PNG counterparts. Native and JPEG 95 acceptance remains
at least 90% correct for each provider with zero primary wrong-provider
assignments. Compression, cropping, and resizing are separate stress conditions.
The archive preserves transport failures and every protocol amendment; source
replacements are selected before scoring, without repeating requests whose
remote completion is unknown. Nine transport-censored parents and two parents
with known moderation refusals, one input-stage and one output-stage, are
replaced before scoring, while their
RAW controls remain in the negative set. The primary cohort therefore measures
100 groups admitted by both APIs, not the refusal rate or all submitted photos.
A provider moderation category does not establish the depicted person's identity.
The full primary confirmation completed with the unchanged model and operating
points. The separately frozen recovery candidates failed their declared gates
as recorded below.

Four official OpenAI checks on a preselected source returned SynthID `detected`
for a masked API edit in native PNG and JPEG 95, and `not_detected` for the
corresponding historical RAW renders in those two formats. All four prepared
uploads had C2PA `not_present`; image pixels were preserved during metadata
stripping. The immutable batch completed and passed integrity verification.
These are OpenAI-supported-signal verdicts, not general AI-absence judgments.

Google's settled response explicitly reported SynthID in the corresponding
native Google edit. The JPEG 95 response combined a technical-error message
with a positive assertion and remains `indeterminate`. Another selected account
did not produce a settled JPEG 95 verdict, and a RAW-control request lost its
browser session during a network failure. Those unavailable checks are not
negative watermark results. Remaining rows stayed unsubmitted. Consequently,
Google JPEG 95 preservation and source-control absence are not authenticated
by this pass.

A resize diagnostic uses only the already-consumed small cohort: all 24 API
outputs and 400 negative variants from 88 negative origin groups. After reducing
images to width 1024, Lanczos restoration to their supplied original dimensions
recovers 12/12 OpenAI and 9/12 Google classifications, with zero provider
assignments on the 400 negatives. The reduced API images alone yield 0/12 for
each provider. This is development evidence with known geometry, not untouched
validation or input-only scale inference.

A separate secondary candidate was frozen before any new holdout predictions.
It retains the baseline result unless that result is `unknown` and the decoded
image width is exactly 1024. In that case it tries one fixed Lanczos 1.5x
restoration to width 1536, using the unchanged strict head, templates, cuts,
and amplitude floors. There is no scale grid or source-geometry input. Its
claim is limited to the controlled 1536-to-1024 export pipeline; its results
must be reported separately from baseline acceptance. A robustness claim
requires at least 90% correct for each provider and zero wrong-provider
assignments in the resize condition. Neither candidate establishes that its
periodic component is SynthID rather than another rendering fingerprint.

Additional consumed-data development rejects that unchanged-head restoration:
one FLUX parent originally exported at width 1024 is falsely assigned OpenAI
in six correlated native, JPEG 95, and resize views, including its equal-pixel
PNG counterpart. Restoring to supplied original geometry had not exercised
this blind 1.5x hypothesis. Image width alone does not establish that an image
was reduced from width 1536.

A separate recovery-only head is selected on the same consumed corpus before
new holdout scoring. The retained grid has 294 combinations of correlation cuts
and Google amplitude floors. It requires zero wrong-provider assignments over
all seven old stress conditions, then ranks minimum resize recall and more
conservative operating points. The selected cuts are 0.58 OpenAI / 0.55 Google,
with floors 0.055034 / 0.070000. It recovers 12/12 OpenAI and 11/12 Google resize
views with zero wrong-provider assignments across those development conditions.
These are fitted development results, not independent confirmation. The
baseline model remains unchanged. Both recovery candidates stay separately
recorded; combined acceptance additionally requires the baseline to pass and
the recovery candidate to meet the recall and zero-wrong-assignment gates in
native, JPEG 95, and resize conditions.

The completed confirmation has 100 API-admitted source groups, 400 native API
exports, 111 historical RAW-negative groups and 128 foreign prompt groups.
The requested API model identities are `gpt-image-2` and
`gemini-nano-banana-2.1`. All positive exports have decoded dimensions
1536 x 2752; this does not cover arbitrary aspect ratios, model versions,
or ordinary Google Photos edits.
There are 1,178 negative variants from 239 origin groups per condition, and
11,046 views over all seven conditions. Both provider case counts are 200 per
condition. The model passes the declared native/JPEG 95 row-recall and
zero-wrong-assignment acceptance gate without refitting.

| Condition | OpenAI correct / 200 | Google correct / 200 | Negative false assignments / 1,178 |
| --- | ---: | ---: | ---: |
| Native | 199 | 195 | 0 |
| JPEG 95 | 199 | 188 | 0 |
| JPEG 85 | 180 | 70 | 0 |
| JPEG 75 | 132 | 34 | 0 |
| Resize to width 1024 | 0 | 0 | 0 |
| Crop 8 pixels per edge | 199 | 195 | 0 |
| Crop 2 pixels per edge | 199 | 195 | 0 |

All mismatches across these conditions abstain as `unknown`; neither provider
is incorrectly assigned to another provider or a negative control. All primary
variants are correct in 99/100 OpenAI groups, 88/100 Google groups and 239/239
negative groups. Formats, edit cases and views within one group are correlated.
The all-success one-sided 95% negative-group failure upper bound is about
1.25%, conditional on independent, representative sampling assumptions that
capture-date grouping alone does not establish. This is not proof of a rare
population false-positive rate.

Per-parent mean positive matched amplitude exceeds the maximum of its own RAW,
full-resolution and ordinary-processing controls in 100/100 groups for each
provider. The paired one-sided sign test is approximately 7.89e-31 for each
provider under its sampling assumptions. This supports a repeatable processing
association; it does not identify the watermark payload or distinguish a
watermark component from a co-occurring renderer fingerprint. All 400 exports
have verified request/response byte bindings and Pillow/OpenCV native decode
parity. The unchanged portable implementation also matches the consumed corpus
numerically and through actual file-path CLI recovery calls. Images and API
payloads remain outside tracked public paths. The model retains private,
noncommercial research restrictions and is not integrated into `identify`.

The separate recovery holdout also completes without threshold changes. The
unchanged-head fixed 1.5x candidate and the supplied-original-geometry
diagnostic each recognize 197/200 OpenAI and 104/200 Google resize views, but
misassign four Google exports from two source groups to OpenAI. Their negative
controls still yield zero provider assignments. This rejects the candidate;
knowing source geometry does not rescue the strict head on this larger sample.

The separately calibrated recovery head recognizes 190/200 OpenAI and 169/200
Google resize views. It makes zero wrong-provider assignments across all seven
conditions, including native width-1024 foreign exports. Its 84.5% Google
resize recall is below the predeclared 90% gate, so combined acceptance fails
and the branch remains an explicitly enabled research option. Default inference
uses the accepted baseline. Neither head is retuned on this holdout. A decision
audit of baseline stored scores finds three ambiguous positive views, all from
one Google export, and no negative view with both provider heads admitted.

Remaining research targets are stronger JPEG and resize invariance, new export
sizes and aspect ratios, newer camera and phone pipelines, and a watermark
on/off causal comparison within the same renderer. Altering the model or its
operating points after these results requires another untouched confirmation.
Commercial or production use additionally needs a separately cleared corpus
and model license. The closed confirmation does not establish a proprietary
watermark decoder or generic AI-origin detection.


### Geometry and partial-image reading proposal, 2026-10-08

A text-only Claude Opus consultation (`claude-opus-5-5`, tools disabled)
recommended geometry synchronization before local aggregation and learned
residual classifiers. The consultant did not inspect images or run tests.
These are development proposals, not extensions of the accepted baseline.

An offline synthetic diagnostic used a period-8 horizontal cosine, amplitude
10 around RGB level 128, on a 1536-square image, then Pillow Lanczos reduction
to 1024 square. The existing sigma-2 residual folded at period 8 had RMS
4.987223 before reduction and 0.008072 afterward; period-16 folding of the
reduced residual had RMS 6.691334. This demonstrates grid cancellation for
this synthetic carrier, not recovered SynthID or a real-image recall gain.
At scale 2/3, the source fundamental moves to 3/16 cycles per observed pixel;
period-8 folding cancels it. Other harmonics can remain on the period-8 grid.
The different filtering responses also prevent interpreting the RMS ratio as
an SNR improvement.

Priority experiments, using the retained corpus as development data:

1. Measure template spectra and cross-provider overlap, then test direct
   forward-transformed templates or harmonic matching in observed coordinates.
   Include resampling kernels, rounded anisotropic geometry and filtering;
   naive inverse resizing has already failed its confirmation gate.
2. Read actual 256-, 512- and 1024-pixel crops without enlarging their evidence
   count. Estimate boundary effects and texture-dependent signal strength.
   Aggregate disjoint windows with common geometry and phase; use overlapping
   windows for localization only. Distinguish a detected region from attribution
   of the entire frame, especially for pasted composites.
3. Calibrate the complete maximum over providers, scales, phases and windows
   on hard negative origin groups. JPEG grids, foreign renderer patterns and
   resampling artifacts are nuisance signals. Sign-flip and decoy statistics
   require checked assumptions and empirical calibration; sample splitting
   does not automatically yield exact p-values for correlated image regions.

Under homogeneous signal and effectively independent noise, detection SNR
scales with the square root of effective area. This is an unvalidated crop
power hypothesis, not a supported minimum size. JPEG quantization and spectral
energy loss are candidate explanations, not measured causes of current misses.
Selective, hash-bound oracle checks on crop/transform ladders can compare
sensitivities; agreement does not identify the learned component as SynthID.
A new candidate selected using the closed corpus needs a fresh, frozen
confirmation with unseen source groups, export geometries and transform
families. The unchanged baseline's prior verdict remains valid for its scope.


### Offline crop and source-coordinate development, 2026-10-08

The geometry proposal was exercised offline on 767 retained sources: 400 API
exports from 100 shared source groups, 111 RAW renders and 256 foreign exports
from 128 prompt groups. All positives are 1536 x 2752; foreign originals are
1024 square. The consumed corpus was repartitioned by origin group into train,
calibration and internal validation. Decoded-pixel and original-content hash
identities do not cross those partitions. This is development, not another
independent confirmation or a broader camera/model claim.

Native-template energy in the selected source-coordinate low-frequency band
is 92.03% for OpenAI and 46.89% for Google; JPEG-95-template fractions are
65.08% and 67.58%. Maximum cross-provider phase correlation rises from 0.3533
in the full template space to 0.5704 in this band. These measurements support
investigating spectral loss and ambiguity; they do not establish the cause of
resize failures or identify the component as SynthID.

Actual crops use three fixed-seed positions per source at sides 256, 512 and
1024, without evidence upsampling. The frozen baseline makes four negative
false assignments among all 256-square crops and wrong-provider assignments
on some larger positive crops. Small edge trimming had not exposed these
failures. A Fourier-domain channel covariance filter, estimated only from
train negatives with 50% isotropic shrinkage, improves directional matching.
It retains a separate original-template physical matched-amplitude guard.
Cuts and floors are selected on calibration from a finite recorded grid.

| Candidate/condition | OpenAI correct | Google correct | Negative false assignments |
| --- | ---: | ---: | ---: |
| 1024-square spectral whitening | 152/156 | 154/156 | 0/258 |
| Fixed scale 2/3, empirical low-band reader, Lanczos | 49/52 | 47/52 | 0/86 |
| Same frozen resize reader, bicubic | 47/52 | 49/52 | 0/86 |
| Same frozen resize reader, bilinear | 42/52 | 46/52 | 0/86 |
| Same frozen resize reader, Lanczos then JPEG 95 | 0/52 | 14/52 | 0/86 |

These are internal-validation counts. Crop positives have 26 source groups
per provider and negatives 57 groups; positions and edit cases are correlated.
The 1024-square reader makes no wrong-provider assignment in this partition,
and all crop positions/cases are correct in 25/26 OpenAI and 24/26 Google
groups. Against the frozen baseline on these same groups, correct fractions
improve in four groups and decrease in none for each provider, with 22 ties.
The one-sided sign-test value is 0.0625 for each provider under its sampling
assumptions; the descriptive gains are not strong independent evidence.

The resize reader fits source-coordinate harmonics directly, with a fixed
1.5 inverse-scale hypothesis and rounded independent width/height geometry.
Its empirical template is fitted only on train source groups. It abstains on
all 256 untouched native-width-1024 foreign controls from 128 prompt groups.
Forward-only and mixed template banks falsely assign one such control and
are rejected. This is not an arbitrary blind scale or rotation estimator.

Whitening at crop sides 256 and 512 retains one negative false assignment in
each internal-validation condition. JPEG-85 crop whitening also has wrong
assignments and poor recall. Those branches remain rejected. Resize followed
by JPEG 95 remains a major uncovered composition, despite good resize-only
results; no thresholds are changed during the additional kernel/JPEG stress.

Two scoped portable research readers were packaged without images or API
records. Actual file-path CLI scores and decisions match stored development
features. Constant-carrier null, source-coordinate synthetic carrier and a
threshold mutation exercise the feature/decision seam. The accepted baseline
and original confirmation hashes are unchanged. No new image-generation or
oracle calls were used. Fresh source-group confirmation, complete scan
calibration, broader geometries and a watermark causal comparison remain
required before extending the accepted scope.


### Coherent-phase crop and JPEG follow-up, 2026-10-08

A second text-only Opus review (`claude-opus-5-5`, tools disabled) prompted
same-phase guards, input-local covariance and paired JPEG subsampling tests.
The previous spectral-whitening prototype maximized whitened direction and
unwhitened physical magnitude separately, potentially at different phases or
templates. The new rule evaluates both at the same winning match. The original
unwhitened frozen baseline already had a coherent projection and is unchanged.
Its RGB projection is identity; a suspected color-space mismatch was ruled out.
Correlation times residual RMS is a normalized projection, not an automatic
inflation merely because an image is textured.

Empirical crop templates are averaged over train source groups after restoring
known crop phase offsets. Negative covariance and templates use train data;
finite direction, physical-amplitude and orientation-support cuts use
calibration. Full covariance and frequency/channel covariance were compared.
Expanded negative controls add twenty crop positions per source at sides 256
and 512. These are correlated views, not extra independent groups.

| Scoped candidate | OpenAI correct | Google correct | Negative false assignments | Negative origin groups |
| --- | ---: | ---: | ---: | ---: |
| Actual 512-square lossless crop | 153/156 | 142/156 | 0/1,978 | 57 |
| Actual 1024-square lossless crop | 155/156 | 155/156 | 0/258 | 57 |

Both candidates have zero wrong-provider assignments on positive internal
validation. Each provider has 26 positive source groups, two cases and three
crop positions. All views are correct in 24/26 OpenAI and 19/26 Google groups
at side 512, and 25/26 for each provider at side 1024. Against the unchanged
baseline on these same groups, side-512 correct fractions improve in six
OpenAI and seven Google groups, decrease in none, with the remainder tied.
Exploratory one-sided sign values are 0.015625 and 0.0078125 under the unproved
independent-group assumptions. Side-1024 improves in four groups per provider,
with no decreases and a sign value of 0.0625. These repeated development
comparisons are not independent confirmation or multiplicity-adjusted claims.

The expanded side-256 candidate retains one negative false assignment among
1,978 internal-validation views and is rejected. A separate 16-subwindow
input-local covariance/conditional-template statistic makes zero negative
assignments, but Google recall falls to 70/156 at side 256 and 92/156 at side
512. It is not selected. The small-crop signal remains information-limited and
vulnerable to foreign-generator patterns; no supported minimum below 512 is
established by these results.

JPEG-conditioned empirical templates improve resize-then-JPEG-95 internal
validation from 0/52 to 31/52 OpenAI and from 14/52 to 29/52 Google. However,
four added native-width-1024 JPEG negative controls are falsely assigned.
After train/calibration-only control augmentation, recall is 32/52 and 28/52,
with zero assignments among 374 internal-validation negative views, but four
known negative training counterexamples remain. That bank is rejected despite
the clean validation partition. Dropping source-even/even harmonics overlapping
the output JPEG lattice does not eliminate every failure. JPEG-before-resize
and native-width-1024 JPEG histories are retained as explicit hard controls.

Paired transfer measurements on 52 train source groups per provider compare
JPEG 4:4:4 and 4:2:0 after Lanczos reduction. At quality 95, OpenAI periodic
residual Cr/Cb projection transfer is about 0.052/0.040 with 4:4:4 versus
0.0055/0.0034 with 4:2:0; Y transfer is about 0.171 in both. Google Y is about
0.689, with Cr/Cb 0.137/0.180 versus 0.041/0.050. Parent-bootstrap intervals
are retained locally. These measure residual transfer including content
nuisance, not isolated watermark amplitude. Chroma subsampling contributes;
it is not the sole explanation, and RGB channel-energy fractions do not by
themselves establish chromaticity.

Only the two lossless-crop modes are included in the selected research archive.
Actual file-path CLI direction, same-phase amplitude, support and predictions
match stored features. Synthetic checks cover known phase, constant null,
source-coordinate identity, rounded anisotropic resize and equality of the
16-subwindow mean to the whole-frame fold. Original baseline/confirmation hashes
are unchanged. No image-generation or oracle calls were added.

All source groups remain consumed development data. Fresh confirmation,
modern phone pipelines, additional generators/aspect ratios, complete scanning
calibration and a watermark causal comparison remain open. Fifty-seven
negative groups with zero failures imply only an approximately 5.12% one-sided
95% upper failure bound under independent, representative sampling assumptions.
Arbitrary scale/rotation, JPEG crops and resize/JPEG compositions remain outside
the scoped candidate claim. Neither reader authenticates SynthID or decodes it.

## Public GitHub sweep, 2026-08-24

GitHub's primary-repository search returned 78 repositories with `synthid` in
the name and 81 with it in the description, 133 unique repositories in total.
The fork-inclusive searches were much larger and capped or dominated by copies
(860 name hits and 1,311 description hits), so this is not a claim that every
fork or unindexed private repository was read. The sweep also used targeted
code searches for detector, verifier, correlation, codebook, bypass, and
removal implementations. SynthID-Text, visible-corner-mark removers, metadata
cleaners, wrappers, and literal ports were classified before inspecting the
remaining image-pixel candidates at pinned commits.

The audits are not null results. They separate three claims that must not be
collapsed: a repeatable image statistic exists, that statistic is a calibrated
single-image SynthID detector, and changing its score transfers to the
provider's production decoder. The repositories support parts of the first
claim. None supports the second or third on the current challenge data.

| Repository | Retained contribution | Rejected use | Decisive result |
| --- | --- | --- | --- |
| [`Rinne414/SynthID-detector`](https://github.com/Rinne414/SynthID-detector/tree/b08abff06b67c37db9dfa7c6ed63760a446d560f) | A reproducible fixed-residual correlator and a useful spoofability negative control | Current-image detector or fixed-template remover | All three templates detected 0/400 current OpenAI and Google images. Adding the expected template spoofed the public detector, while subtracting 0.25-2x from an official positive left SynthID `detected` down to 25.72 dB |
| [`newideas99/gpt-image-synthid-detector`](https://github.com/newideas99/gpt-image-synthid-detector/tree/5495e09c62dc26112f9a0e6cfab469f9a10f37e3) | Evidence that a CNN can learn a renderer/regeneration-domain signature, plus a white-box shortcut baseline | Payload decoder, independent hybrid confirmation, or removal oracle | The published cut retained 172/200 OpenAI but accepted 110/200 Google. A gradient drove the surrogate effectively to zero, but official SynthID stayed `detected` through +/-8/255 at 34.30 dB |
| [`aloshdenny/reverse-SynthID`](https://github.com/aloshdenny/reverse-SynthID/tree/b11083676fd3ee3ff97ce9d03c0e409e46905902) | Cross-color phase consensus and further evidence for a repeating Google renderer-lineage pattern | The published V3/V4 thresholds as presence detectors, or Round-06 as a verified quiet remover | V4 accepted 77/200 Google and 76/200 OpenAI at the same cut. Round-06 has no published per-image oracle evidence and uses regeneration/distortion with 14/11 dB PSNR floors |
| [`Ristellise/REGRET`](https://github.com/Ristellise/REGRET/tree/7d449034bf323987e7e608e7886e029ed20fd847) | A compact 147-d second-order SPAM descriptor and a safely auditable sklearn model | Google or OpenAI detector, or a high-precision hybrid rescue | The published cut accepted 139/200 Google, 141/200 OpenAI, and 214/600 disjoint ImageNet photographs |
| [`vordme2010/synthid-dataset`](https://github.com/vordme2010/synthid-dataset/tree/133a27088f6f4d695c79db9a1a70fa8e7fa3adad) | A public 2026 Gemini flat-field corpus, safe numeric features, and explicit candidate carrier bins | Its reported near-perfect Tier-1 AUC as open-world SynthID evidence | Training used 500 Gemini flats against 1,500 synthetic or phase-scrambled negatives and zero real photographs. Rebuilding the SVM from the safe matrix accepted 1/200 Google and 0/200 OpenAI on v7 |

Useful forensic signal is therefore not synonymous with a production
watermark decoder. The Google lattice idea is retained only as a
research-lineage component; it cannot be labelled SynthID. The fixed template
and CNN gradient are retained as negative controls. None of these public
implementations is shipped as a detector or remover. Detailed classifier and
attack measurements live in [SynthID source classifiers](synthid-classifiers.md) and
[mark removal](synthid-removal-research.md).

### Google verifier surfaces found in the sweep

Two real but non-local verifier paths surfaced. The historical Vertex SDK
exposed `WatermarkVerificationModel.from_pretrained("imageverification@001")`.
The class still exists in `google-cloud-aiplatform` 1.165.1, but warns that it
was deprecated 2025-06-24 with removal scheduled for 2026-06-24. Calls in two
eligible Vertex configurations were rechecked on 2026-09-15 with a real processed
variant and correctly attributed user credentials. The publisher
model resolved, but both `predict` calls returned HTTP 403 `The caller does not
have permission`. The surviving class reference therefore does not establish
public access to the backing model. A repository using
`response.watermark_detected` is also stale: the actual SDK property was
`watermark_verification_result`, backed by the prediction's `decision` field.

A newer unlisted upload route,
`POST https://synthid.googleapis.com/upload/v1:verifyContent`, is live enough
to identify itself as the SynthID API and return `SERVICE_DISABLED` for a real
public image. It has no public discovery document, did not appear in the
project's available-services listing, and no authorized successful response
was obtained. Enabling a cloud service was deliberately not attempted. This
may be an access-gated research oracle, but it is not a local decoder and does
not satisfy the metadata-free offline product requirement. Gemini chat remains
the only Google verifier actually evidenced by the public bypass corpora.

## Closed detector routes

| Route | Close | Why |
| --- | --- | --- |
| Wavelets / FFT / cepstrum as a single-image detector | 2026-08-09 | TrustMark proxy: 318-d wavelet/spectral summary AUC 0.653 and 0 TPR at a clean calibration cut. Complex FFT maps AUC 0.516. Spatial RGB still won |
| `aloshdenny/reverse-SynthID` V3 phase codebook | 2026-08-09/24 | Pickle-free numeric audit. The original pilot was 5/5 Google positives and 0/194 then-available negatives. The broader audit was 5/355 Google, 5/499 controls, and 6/1,000 fresh Open Images. On v7 it accepted 4/200 Google and 1/200 OpenAI. As a v11 `unknown` rescue it adds 2/200 Google on this test, but the historical control errors reject a 1%-recall OR rule |
| `aloshdenny/reverse-SynthID` V4 | 2026-08-13/24 | Commit `b110836`. Better-of-two profiles: 141/355 Google positives and 191/499 controls. Frozen 1,000 Open Images: 386 accepted. Paired AUC 0.517. An exact repeat on the hash-disjoint v7 challenge accepted 77/200 Google and 76/200 OpenAI. Strict GUI aspect routing reduced support but not false acceptance. The threshold sits 0.02 above chance phase similarity |
| `cebeuq/Synthid-Bypass` as a local decoder | 2026-08-13 | 12 public before/after diffusion pairs. Fixed carrier 10/12 before, 0/12 after. Gemini Verify still saw SynthID on two `before` files the carrier missed. Reconstruction removal, not a payload read |
| S4: one content-independent pattern for both providers | 2026-08-15 | Google has a shared phase-coherent tile. OpenAI is at chance. See Result |
| M2: mint Google encoder-off pairs | 2026-08-15 | No current Google path emits an unwatermarked image. See Result |
| U1: one feature space for both providers | 2026-08-15 | Transfer 0.535 and 0.469 |
| Hand statistics vs a perceptually masked keyed mark (M1b) | 2026-08-15 | At equal RMS, fixed 0.670, keyed 0.669, keyed+masked 0.546. Masking, not keying, defeats hand features |
| Decoder-substitution / foreign VAE as a verified negative | 2026-08-17 | OpenAI oracle stayed `detected` at 22.3 dB. Attacking a positive does not mint a labelled negative |
| `aoxo/reverse-synthid` as extra Google templates | 2026-08-19 | Hugging Face dump scored against the frozen era bank. Extra family means raised foreign scores instead of isolating a mark |
| Origin-locked lattice as SynthID | 2026-08-16 | Two-pixel crop kills it; published SynthID keeps 99.97% TPR under aggressive crop. See classifiers for `google_lineage_renderer` |
| Keyless energy in 16-32 px | 2026-08-20 | L1 AUC 0.53; official decoder reads phase structure, not energy |
| TrustMark / open-encoder transfer | 2026-08-20 | E3 leave-one-encoder-out at chance on TrustMark (0.505 ridge, 0.510 conv) |
| L1 distillation of oracle bits | 2026-08-20 | Geometry-only AUC 0.78-0.83 beats pixel 124-d (0.64). ChatGPT export and `opened` C2PA predict `not_detected`. Inside 1254x1254, permutation p=0.45 |
| Flat 16-32 matched filter | 2026-08-21 | Leave-one-out residual correlation 0.645 on gray flats; 0.007-0.025 on COCO photo residuals. L1 AUC 0.70 was a size confound (size-matched 0.59 / 0.44) |
| `gpt-image-1` as encoder-off pair | 2026-08-21 | Does not stamp. `gpt-image-2` does |
| CNN on raw L1 bits | 2026-08-21 | Do not train. Labels are export geometry and presentation, not the mark |
| 128-photo causal wavelet student on the flat template | 2026-08-21 | Photo 16-32 jam flips only at a=24 / 18.6 dB (`a*_photo < 0.53` failed). A student cannot beat a matched filter if the photo-domain mark is that quiet |
| Photo 16-32 amplitude without the flat-arm template | 2026-08-23 | 11 COCO `gpt-image-2` edits plus the red-dot pair. Residual is the camera JPEG resized onto the edit raster, top-left 32 px dropped. Median 16-32 residual RMS 12.2, same order as 8-16 (9.0) and 32-64 (16.3). On the five tightest pairs (full residual RMS < 16): 16-32 median 2.69, still ~24x the 0.11 flat stamp and not louder than 8-16 (3.81). The photo residual does not isolate a 16-32 codeword |
| Sort-by-luma 1D chroma stego | 2026-08-23 | Flat LOO 0.914 but a luma-order shuffle is already 0.792. COCO plus tiny noise: 0.642 vs shuffle 0.640. The sort invents the correlation |
| Retinex / local-gain then 16-32 LOO | 2026-08-23 | 0.742 vs plain 16-32 LOO 0.752 on the same eight gray flats. No new carrier |
| Distant 16-32 patch coupling | 2026-08-23 | Near lag 0.13 of a tile: -0.013. Far lag ~500 px: +0.054. No nonlocal lock |
| Bispectrum of the 16-32 ring | 2026-08-23 | Degenerate: 1.0 on flats and 1.0 on COCO. Not a statistic |
| Length-255 m-sequence vs 16x16 fold | 2026-08-23 | Max abs corr 0.059. Chance for n=255 is `1/sqrt(255) ≈ 0.063` |
| Row-as-audio pitch peak | 2026-08-23 | Peak-to-mean 51 on flats, 46 on COCO noise residuals. Shared FFT peakiness, not a pitch |
| 16x16 origin search of the flat 16-32 template on photos | 2026-08-23 | Zero-shift NCC 0.006-0.024, best of 256 shifts 0.013-0.026. Origin is not hiding a lock |
| Affine search of the 16-32 residual vs the flat template | 2026-08-23 | 11 `gpt-image-2` photos: zero-deg median 0.009, best rotation in ±4° 0.020, lift 0.011. Six COCO noise residuals: lift 0.008. Flats already peak at 0°. No hidden rotation or 3% scale lock |
| Fourier-angle scramble of 16-32 (true `(r, θ)` of the annulus) | 2026-08-23 | s1 24.1 dB official `not_detected`, cartesian replicate 24.8 dB `not_detected`. Decoder reads phase in that annulus, including angular organization. Not quieter than scramble |
| Baker-map of the 16-32 band | 2026-08-23 | s1 27.8 dB still `detected` |
| Poisson / photon-shot embedding | 2026-08-23 | Flat Anscombe LOO 0.7518 vs linear 0.7519. s1 Poisson 30.7 dB still `detected` |
| Nested LSB plus SynthID | 2026-08-23 | s1 55.9 dB still `detected`. Extra open stego does not hide or replace the mark |
| ICC sRGB rewrite | 2026-08-23 | No-op, PSNR inf. Mark is not in a profile that a pixel-preserving assign changes |
| Seam carve 8 columns of 16-32 energy | 2026-08-23 | s1 27.7 dB still `detected` |
| Extra OpenAI image ids as a new ensemble | 2026-08-23 | `gpt-image-1.5` and `gpt-image-1-mini` are encoder-off: C2PA `created, converted`, official `not_detected`, 16-32 corr vs the `gpt-image-2` gray stamp 0.003. Dated `gpt-image-2-2026-04-21` is the same family: C2PA `watermarked.unbound`, 16-32 corr 0.189. `chatgpt-image-latest` generate stays 403 until org verification |
| DIRE reconstruction error as a payload | 2026-08-23 | SDXL float32, 512 px. VAE RMS: gray 1.24, s1 9.15, photo edit 9.89, camera 11.84. DDIM DIRE RMS: gray 2.40, photo 30.9, s1 31.5, camera 33.0. Rank follows texture. Float16 DDIM NaN'd on MPS. Not a payload |
| HEIF / AV1 as a hidden carrier | 2026-08-23 | s1 HEIF q80 46.3 dB, q50 39.3 dB, AV1 CRF 32 37.2 dB, all official `detected`. Codec round-trip does not mint a new statistic |
| Print-scan simulation as the kill | 2026-08-23 | s1 24.95 dB still `detected`. Same PSNR class as 16-32 scramble, which kills. Generic 25 dB is not the mark |
| Mellin / log-polar \|F\| of the 16-32 residual | 2026-08-23 | Flat LOO 0.996-1.000, photo 0.996, COCO 0.991-0.995. The annulus envelope, not a payload. Scale chirps `r^{iα}`: flat 0.51, photo -0.27, COCO -0.58. No photo lock |
| Zernike n≤6 and PCET of the residual | 2026-08-23 | Flat LOO 0.77 / 0.75, same order as cartesian 16-32. Photo -0.007 / 0.044. COCO 0.23 / 0.35, above photos. Moments of the flat stamp, not a photograph decoder |
| Fresnel quadratic-phase of 16-32 (z=8,16,32,64) | 2026-08-23 | Flat LOO 0.750-0.753, photo 0.012, COCO -0.006. Unitary copy of the cartesian residual. No new shell |
| 8-bin Fourier orientation energy | 2026-08-23 | 0.999 flat, 0.966 photo, 0.995 COCO. Saturates |
| Chroma 16-32 as the payload | 2026-08-23 | Gray-flat Y LOO 0.682, Cb 0.064, Cr 0.025. s1 Y scramble 24.6 dB `not_detected`; Cb 45.0 dB and Cr 43.9 dB stay `detected`. Mark is in luma |
| One 90° Fourier sector of 16-32 | 2026-08-23 | s1 27.5 dB still `detected`. Decoder needs the whole annulus, not one wedge |
| Radial-phase-only scramble of 16-32 | 2026-08-23 | s1 25.0 dB `not_detected`. Phase as a function of radius, over the full ring, is enough to kill. Combined with Fourier-angle scramble, either polar coordinate of the annulus phase is a kill if the whole ring is hit |
| [`Rinne414/SynthID-detector`](https://github.com/Rinne414/SynthID-detector/tree/b08abff06b67c37db9dfa7c6ed63760a446d560f) fixed residual template | 2026-08-23/24 | The repository reports 0/33 held-out AI images and identifies its original result as self-correlation. Its pinned pre-May GPT-Image2 template first detected 0/11 current source-matched edits and 0/11 COCO sources. An exact repeat on 200 current OpenAI and 200 current Google images produced zero detections for all three published templates at the 0.08 cut; maximum GPT-Image2 score on OpenAI was 0.0167. Adding the expected template can spoof the public correlator, but that template is not present in held-out content. Independent repeat of the rejected fixed-template route, not a payload decoder |
| [`newideas99/gpt-image-synthid-detector`](https://github.com/newideas99/gpt-image-synthid-detector/tree/5495e09) CNN ensemble | 2026-08-24 | Exact frozen inference retained 172/200 OpenAI but accepted 110/200 Google at the published 0.5 cut. The training negatives are regenerated positives, while its confound probe transforms only positives and has no foreign-image control. A full-frame white-box attack drove its probability from 0.99995 to 0.25956 at ±2/255 and effectively zero at ±4/255, but the official OpenAI decoder stayed `detected` through ±8/255. Renderer/confound classifier, not a mark decoder |
| [`Ristellise/REGRET`](https://github.com/Ristellise/REGRET/tree/7d449034bf323987e7e608e7886e029ed20fd847) SPAM model | 2026-08-24 | At the published cut, exact model inference accepted 139/200 Google, 141/200 OpenAI, 214/600 disjoint ImageNet photographs, and substantial fractions of foreign generators. The apparent high-precision tail did not transfer |
| [`vordme2010/synthid-dataset`](https://github.com/vordme2010/synthid-dataset/tree/133a27088f6f4d695c79db9a1a70fa8e7fa3adad) Tier-1 SVM | 2026-08-24 | Rebuilt without loading joblib, using the repository's 2,000x33 numeric matrix, reference phases, noise template, seed-42 split, scaler, and RBF SVM. It accepted 1/200 current Google and no current OpenAI rows. Its published AUC above 0.999 distinguishes Gemini flats from generated derivatives, not current photographs |
| Generic GitHub heuristics and literal ports | 2026-08-24 | `hackerfactor/reverse-SynthID-C`, `BIRSAx2/ripmark`, and `xiaoyao9184/reverse-from-synthid` reproduce or wrap the already-rejected reverse-SynthID codebook. `AI-SCERN` uses uncalibrated annular energy, decoder-grid, and FFT-symmetry heuristics with no weights or held-out corpus. LSB flips, ±1 dither, and 100.5% resize tools publish no matching-oracle evidence. None adds an independent signal |

## Information budget on photographs (2026-08-21)

On 11 `gpt-image-2` photo edits, the flat-derived 16-32 template at full
amplitude against per-image whitening has median `d' = 0.93` (range
0.68-2.46). That is a best-case single-image AUC of about 0.75 even with a
perfect, perfectly aligned template. A P5 gate (FPR 0.1%, TPR 90%) needs
`d' >= 4.37`. The deficit is 4.7x in amplitude, 13.4 dB, *unless* the
photo-domain mark is louder than the flat measurement. Titration said it
is not: fish 16-32 additive jam stays `detected` through a=16 / 21.9 dB.

2026-08-23, without using that flat stamp as `G`: the aligned photo
residual in 16-32 is redraw, not a codeword. Median RMS 12.2 across 12
pairs; 2.69 on the five tightest. Neighboring octaves are as loud.
`.local-eval/synthid/prc-oklab-attack-2026-08-15/photo-band-amplitude-2026-08-23.json`.

Do not report an AUC from those 11 pairs as a detector result. With n=11
the standard error on AUC is about 0.12. Notes:
`.local-eval/synthid/prc-oklab-attack-2026-08-15/agent-detector-claude.md`.

## External literature (surveyed 2026-08-23)

Primary sources, not abstracts. Each row is mapped onto a closed or open
route in this campaign. Mechanism detail stays in
[synthid.md](synthid.md). Removal papers are on
[mark removal](synthid-removal-research.md). Classifier papers are on
[SynthID source classifiers](synthid-classifiers.md).

### Official mark, not a public decoder

| Source | What it is | Map to this campaign |
| --- | --- | --- |
| Gowal et al., [arXiv:2510.09263](https://arxiv.org/abs/2510.09263) | Post-hoc encoder `f` / decoder `g`. Detection logit is not payload recovery. SynthID-O (partner variant) 136 bits at 512x512. TPR at 0.1% FPR 99.98% aggregated, 99.97% on the hardest spatial crop+resize. Trains against sampled semantics-preserving transforms, including weak VAE regeneration. Production decoder unpublished | Matches the architecture we treat as keyed `x' = x + g(x)`. Explains why a two-pixel crop kills `pipeline_lattice` but not the official oracle, and why a 22.3 dB foreign VAE still reads `detected` |
| DeepMind [US12094474B1](https://patents.google.com/patent/US12094474B1/en) and continuation [US20250149048A1](https://patents.google.com/patent/US20250149048A1/en) | Residual U-Net encoder, separate decoder, optional key, encoder/decoder ensembles that need not recognize each other | Constraint, not a recipe. Ensemble non-recognition is why one recovered Google tile cannot be reused as an OpenAI detector (S4) |
| Dathathri et al., [Nature 634:818-823 (2024)](https://www.nature.com/articles/s41586-024-08025-4) | SynthID-Text: tournament sampling of LLM tokens, open-source. Anthropic's [Claude text watermark](https://www.anthropic.com/news/claude-text-watermark) (announced 2026-08-14 for future Claude models, updated 2026-09-01, shipped in Claude Fable 5.1/5.2 and Mythos 5.1/5.2) "is a version of the SynthID-Text approach", with its detection API in private preview | Different system from image/audio/video SynthID; text provenance and token-watermark removal are documented in [text watermark research](text-watermark-research.md) |
| OpenAI, [advancing content provenance](https://openai.com/index/advancing-content-provenance/) (2026-05-19, audio 2026-07-31) and [content provenance API](https://developers.openai.com/api/docs/guides/content-provenance) | ChatGPT / API / Codex images carry C2PA plus SynthID. Audio from 2026-07-31. `POST /v1/content_provenance_checks`. `not_detected` does not rule out another vendor | This is the oracle. C2PA and SynthID are independent entries. Do not abuse the endpoint as an adaptive reverse-engineering loop |

### Keyless presence detectors in the literature

| Source | Claim | Caveat against our gates |
| --- | --- | --- |
| Ao et al., [arXiv:2603.06723](https://arxiv.org/abs/2603.06723) (AWPD / FSNet, SAFE@CVPR 2026) | Leave-one-algorithm-out presence detection. SynthID held out: FSNet Acc 0.894 / F1 0.886, ResNet-50 Acc 0.845 / F1 0.812, ConvNeXt V2 Acc 0.866. LSB and Patchwork both fail below 60%. Hypothesis: modern invisible marks share dense high-frequency spectral anomalies | UniFreq Table 3: SynthID is 2,000 images, all AIGC, zero photographs, from `imagen-4.0-fast-generate-001` only, resized to 256x256. No OpenAI. No 0.1% FPR. Closest published analog to E3/L1 distillation. Our TrustMark leave-one-encoder-out was chance (0.505 ridge). Their high-frequency commonality is the same energy that L1 failed to read as an OpenAI payload |
| `aloshdenny/reverse-SynthID` V3/V4; Google to [The Verge, 2026-04-14](https://www.theverge.com/ai-artificial-intelligence/911579/google-synthid-ai-watermarking-system-reverse-engineered) | Spectral codebook from averaged Gemini flats. Spokesperson Myriam Khan: "It is incorrect to say this tool can systematically remove SynthID watermarks." The author also said the bypass confuses the decoder rather than deleting a payload | Our pickle-free V4 audit: paired AUC 0.517, frozen Open Images 386/1000 accepted. Not 0.1% FPR |
| Krawetz, [Hacker Factor "Reversing SynthID"](https://www.hackerfactor.com/blog/index.php?/archives/1092-Reversing-SynthID.html) and ["Meta's Un-Stable Signature"](https://hackerfactor.com/blog/index.php?/archives/1098-Metas-Un-Stable-Signature.html) | Gemini chat TPR closer to 1/20 than the paper's 99.97%. Detector weak on flats. reverse-SynthID field accuracy ~70% vs the author's 90% | Gemini-app chat is not the OpenAI provenance API. Our gray `gpt-image-2` flats are `detected` on the official OpenAI oracle. Do not mix the two verifiers |
| vitotitto LAB-a logistic (community, tiny holdout) | Reported 97.7% AUC on 20/20 | Not a P5 gate. Ignore as a detector claim |

### Different embedding loci (not post-hoc SynthID)

These papers are often cited as if they were SynthID. They are not.

| Source | Locus | Why it is not this mark |
| --- | --- | --- |
| Gunn, Zhao, Song, [arXiv:2410.07369](https://arxiv.org/abs/2410.07369) (PRC, ICLR 2025) | Pseudorandom error-correcting code in the *initial diffusion latent* | Cryptographic undetectability is a latent-code property. SynthID-Image is applied after the VAE has already decoded pixels. Our OKLab "PRC-style" noise was a category error |
| Francati et al., [arXiv:2509.10577](https://arxiv.org/abs/2509.10577) (EuroS&P 2026) | Crop-and-resize flipped about half of PRC latent signs and blocked belief-propagation decode | Confirms PRC is origin-locked in latent space. Matches why a 2 px shift kills `pipeline_lattice` and does not kill OpenAI SynthID |
| Fernandez et al. Stable Signature; Wen et al. Tree-Ring | Fine-tuned VAE decoder, or ring constraints on initial noise | In-generation. Google's paper is explicit that SynthID-Image does not modify the generator |
| TrustMark ([arXiv:2311.18297](https://arxiv.org/abs/2311.18297)), HiDDeN, StegaStamp | Open post-hoc encoder/decoder pairs | Transfer from these is E3: chance on TrustMark. WAVES ([arXiv:2401.08573](https://arxiv.org/abs/2401.08573)) stress-tests this family, not production SynthID |

Do not train a student on AWPD's Imagen-only 256 crop and call it an
OpenAI SynthID detector. That would repeat L1's contamination: a
generator/size cue, not a payload.

### Adjacent literature, not SynthID

These papers are the published versions of the wild hypotheses and of
Model 1. They detect *something else*: a camera pipeline, a generator
fingerprint, an open watermark, or steganography. None of them is a
keyless OpenAI payload decoder.

**Classical spread-spectrum and open neural marks**

| Source | What it is | Map |
| --- | --- | --- |
| Cox, Kilian, Leighton, Shamoon, IEEE TIP 1997 | i.i.d. Gaussian vector in perceptually significant spectral components. Informed detector. Collusion-resistant by construction | Ancestor of keyed CDMA. Our length-255 m-sequence vs 16x16 fold was at chance because we did not have the key, and because OpenAI is content-adaptive, not a fixed Gold code |
| Zhu et al., HiDDeN, ECCV 2018 | Joint encoder / noise layer / decoder | Open pair. AWPD leave-one-out Acc 0.985. Transfer to SynthID is E3 |
| Tancik, Mildenhall, Ng, StegaStamp, 2020 | 100 bits at 400x400, trained through print-scan | Open pair. Survives recapture. WAVES: TPR at 1% FPR collapses from 1.00 to 0.01 under regeneration |
| Wen et al., [arXiv:2305.20030](https://arxiv.org/abs/2305.20030) (Tree-Ring, NeurIPS 2023) | Pattern in the *initial diffusion noise*, recovered by DDIM inversion | In-generation, origin-locked in latent Fourier space. Crop-invariant by design. Not post-hoc SynthID |
| Fernandez et al., Stable Signature, ICCV 2023 | Fine-tune the LDM VAE decoder so every decode carries a bit string | In-generation. Gowal is explicit that SynthID-Image does not modify the generator |
| Lin and Juarez, [arXiv:2506.10502](https://arxiv.org/abs/2506.10502) (USENIX 2025) | Public-knowledge attack that removes Tree-Ring | Confirms Tree-Ring is a different object with a different kill |

**Steganalysis as a presence detector**

Fridrich and Kodovsky Spatial Rich Models (TIFS 2012) and Boroumand,
Chen, Fridrich SRNet (TIFS 2018) detect sub-bit-per-pixel spatial
stego by high-pass residuals, with pooling disabled in the front of
SRNet so the weak signal is not averaged away. AWPD cites both and
argues that transfer to modern deep / generative marks is difficult.
The preceding wavelet/FFT residual-energy baseline did not reproduce
SRM co-occurrence descriptors or pretrained SRNet features. Its failure
does not establish whether those specific representations transfer.

**Generator fingerprints in the Fourier domain**

Corvi, Cozzolino, Poggi, Nagano, Verdoliva,
[arXiv:2304.06408](https://arxiv.org/abs/2304.06408) (CVPRW 2023):
GAN, diffusion, and VQ-GAN images show spectral peaks and anomalous
autocorrelation; real vs synthetic differ in mid-high radial and
angular power. reverse-SynthID averaged Gemini flats and called the
peak a watermark codebook. Corvi's result says many generators leave
*some* peak. Our V4 Open Images 386/1000 is what a generator-fingerprint
detector looks like when you calibrate it as if it were a payload.

Yao and Juarez, [arXiv:2512.11771](https://arxiv.org/abs/2512.11771)
("Smudged Fingerprints"): 14 fingerprinting methods across RGB,
frequency, and learned features; removal attacks >80% white-box, >50%
black-box. A fingerprint you can see without a key is a fingerprint
you can wipe without a key.

### Image investigation and data hiding (any method)

These are not SynthID papers. They are the rest of the toolkit: how
people hide bits in pictures, and how people tell a picture was
touched. Several of our wild hypotheses already had a published form
here.

Hiding is not one problem. Cover modification (change an existing
image), coverless / generative (sample an image that already carries
the bits), and signed metadata (C2PA) fail under different attacks.

**Cover modification, classical**

| Source | Hide how | Detect / limit |
| --- | --- | --- |
| LSB, Patchwork (Bender et al., IBM SJ 1996) | Flip low bits, or luminance of random pixel pairs | AWPD Acc < 0.60. Sparse or ±1 amplitude. SRNet / FSNet average it away |
| Westfeld F5 (2001), Fridrich nsF5 | JPEG DCT coefficients, matrix embedding | Histogram attacks on F5; nsF5 was the shrinkage fix. Domain is the codec, not a mid-band residual |
| Pevny, Filler, Bas HUGO (2010); Holub and Fridrich WOW (2012); Holub, Fridrich, Denemark UNIWARD (IH&MMSec 2013 / EURASIP 2014) | Content-adaptive costs, bits placed in texture via syndrome-trellis codes (Filler, IH 2011) | The modern spatial/JPEG floor. Distortion is *designed* to look like the cover. A 16-32 energy detector is the wrong statistic |
| Cox et al. 1997 | Spread-spectrum in significant DCT | Keyed. Already mapped above |

**Cover modification, neural (hide a whole image, not 32 bits)**

| Source | What it does | Map |
| --- | --- | --- |
| Baluja, NeurIPS 2017, "Hiding Images in Plain Sight" | Full-resolution secret image distributed across all bits of a same-size cover | Capacity is the point. Not robust, not keyed provenance. HiDDeN later added a noise layer so the secret survives JPEG |
| Jing et al., HiNet, ICCV 2021 | Invertible neural net: hide and recover as one bijection | High-capacity lossless-ish hiding. Recovery needs the exact inverse, not an official oracle |
| Yang et al., PRIS, [arXiv:2309.13620](https://arxiv.org/abs/2309.13620) | Invertible net plus robustness modules | Same family, trained through distortion |

A HiNet-style student on OpenAI pairs would learn the *edit*, not a
SynthID codeword. That is the L1 contamination again.

**Coverless / generative hiding**

The cover is never modified because there is no cover. The sampler
emits an image whose latents already encode the message.

| Source | Locus | Map |
| --- | --- | --- |
| Yang et al., Gaussian Shading, [arXiv:2404.04956](https://arxiv.org/abs/2404.04956) (CVPR 2024) | Map bits onto Gaussian latents indistinguishable from ordinary noise. Recover by DDIM inversion | Sibling of Tree-Ring / PRC. Training-free, performance-lossless *for the generator*. Not a post-hoc pixel stamp |
| Peng et al., StegaDDPM (ACM MM 2023) and later LDStega | Bits in the diffusion sampling distribution | Spatial SRNet is the wrong detector (NS-DSer, [arXiv:2602.10219](https://arxiv.org/abs/2602.10219): move steganalysis into noise space) |
| CRoSS, Pulsar, MDDM | Message-to-noise projections | Same locus. A pixel scramble does not invert the sampler |

**Passive forensics (the picture was touched, no secret assumed)**

| Source | Cue | Map |
| --- | --- | --- |
| Krawetz, "A Picture's Worth", 2007 (ELA) | Re-JPEG at lower quality, subtract | Already measured: COCO 3.13, s1 1.97, gray stamp 0.49. Codec history, not a payload. Farid publicly called ELA as likely to mislabel originals as it is to catch edits |
| Farid, IEEE TIFS 2009, JPEG ghosts | Difference energy vs a sweep of JPEG qualities; spliced regions ghost at their original Q | Local: s1, a `gpt-image-2` photo, and a camera JPEG all minimize at Q90. Codec, not a payload |
| Popescu and Farid, TR2004-515 | Copy-move via duplicated regions | Not generation, not a watermark |
| Popescu and Farid, IEEE TSP 2005 | Resampling periodic correlations | Affine search cousin. A rotated SynthID residual is a different question |
| Wang et al., DIRE, [arXiv:2303.09295](https://arxiv.org/abs/2303.09295) (ICCV 2023) | Diffusion reconstruction error: generated images reconstruct, cameras do not | Model 1 sibling. SDXL float32 at 512: DIRE RMS camera 33.0, s1 31.5, photo edit 30.9, gray stamp 2.40. Texture rank, not a payload. Inverse of the foreign-VAE remover: there the mark survived 22.3 dB |
| Wang, Wang, Zhang, Owens, Efros, [arXiv:1912.11035](https://arxiv.org/abs/1912.11035) (CVPR 2020, CNNDetect) | One ProGAN classifier, heavy JPEG/crop aug, transfers to many CNNs | Ancestor of "train on one generator". Ojha showed the sink-class failure once diffusion arrived. We required Firefly for that reason |

C2PA is the non-pixel stack: a signed manifest, stripped by
`metadata --remove`. Durable Content Credentials (spec 2.4) add a
soft binding that can re-link a stripped file to a repository. That
is provenance, not hiding.

Do not train on ELA, JPEG ghosts, DIRE, or a HiNet reconstruction and
name the score SynthID.

### Waveforms that can live in a picture

A mark is a function on the pixel lattice. The literature does not
use one wave. It picks a basis whose symmetries match the attack it
fears, then hides a keyed coefficient vector in that basis. Cartesian
16-32 is one shell. Polar, scale, and diffraction are different
shells.

**Standing waves on a rectangle.** DFT / DCT / DST. A 2-D sinusoid
`cos(2π(ux + vy)/N)`. JPEG lives here. Our octave scramble destroys
one annular *radius* of these frequencies, not one orientation. A
Gabor packet is the same sinusoid windowed in space.

**Circular and log-radial waves.** Functions of `(r, θ)`, not
`(x, y)`.

| Basis | Wave | Invariance it buys |
| --- | --- | --- |
| Fourier-Mellin / log-polar (Ó Ruanaidh and Pun, Signal Processing 1998) | `r^{iα} exp(ikθ)` after a DFT magnitude | Rotation and scale become translations |
| Logarithmic radial harmonics (IH 2002) | Same family, added in pixels, detected by complex correlation | RST without going through the unstable log-polar resample |
| Polar harmonic transforms: PCET, PCT, PST | `exp(±i2π n r²)` and polar cos/sin | Rotation. Moments, not a dense codebook |
| Zernike / pseudo-Zernike | Orthogonal polynomials on the disk, radial part related to Bessel | Rotation. Classical moment watermark |
| Circular chirp (SPIE 6072, 2006) | Polar map of a 1-D chirp onto a ring | JPEG (tune chirp rate) plus rotation (the ring) |
| Tree-Ring (Wen 2023) | Concentric rings in the *latent* Fourier plane | Crop/flip by construction. Not a pixel wave |

The 2026-08-23 file named polar-1632 is cartesian annular phase
shuffle, a scramble replicate (`not_detected` at 25.6 dB). True
Fourier-angle scramble of the same annulus is `not_detected` at 24.1 dB.
Radial-phase-only scramble is `not_detected` at 25.0 dB. One 90°
sector plus its conjugate stays `detected` at 27.5 dB. Affine/rotation
search of the 16-32 residual against the flat template does not lock
(lift 0.011, COCO 0.008).

**Scale chirps.** Hyperbolic / Mellin monomials (arXiv:1208.5842):
real 1-D Mellin patterns tiled in 2-D. Run 2026-08-23:
`r^{iα}` coefficients flat LOO 0.51, photo -0.27, COCO -0.58.
Log-polar `|F|` and a 64-bin radial Mellin profile saturate on COCO
(0.99). Not a payload, and not quieter than cartesian 16-32.

**Directional packets.** Dual-tree complex wavelets (approximate
analytic wave), Gabor/Morlet, ridgelets, curvelets, shearlets,
contourlets, bandelets. Multiplicative spread-spectrum on curvelet
coefficients is a published detector-design paper, not a SynthID
decoder. Our wavelet summary AUC 0.653 already said a *generic*
packet energy is not the OpenAI payload.

**Optical diffraction, actual wave physics.**

| Transform | What the wave is | Map |
| --- | --- | --- |
| Fresnel | Quadratic phase `exp(iπ r² / λz)`, a radial chirp. Energy concentrates with distance | Holographic watermarks embed the diffracted field, not the picture |
| Fractional Fourier | Rotation of the time-frequency plane. Order is an extra key | Extra key space on top of DFT |
| Double random phase encoding (Réfrégier and Javidi, Optics Letters 1995) | Random phase at input *and* Fourier plane. Ciphertext is stationary white noise | Closest optical analog of a keyed `x' = x + g(x)` that looks like noise without both screens |
| Cascaded Fresnel / CGH | Two holograms at different depths | Hardware key. Not a pixel residual we can match-filter |

A Fresnel/FrFT student without the distance or the fractional order
is another unmatched filter. Fresnel at z=8, 16, 32, 64 on the 16-32
residual is a unitary copy of the cartesian band (flat LOO 0.75, photo
0.012). FrFT is the same unmatched-filter claim with an extra order
key; it does not isolate a photograph lock the four distances missed.

**Cyclostationary and clock-warped waves.** Periodic clock change
(Martin, Chabert, Lacaze, EUSIPCO 2005) turns a stationary host into
a cyclostationary one by a periodic time warp. Our H2 multiplicative
cyclostationarity test already put official positives *below*
controls. Do not rerun the same statistic in a new basis and call it
new evidence.

**How to read a candidate wave.** Name the coordinates `(x,y)` vs
`(r,θ)` vs `log r`, the generating function (sinusoid, chirp,
Bessel, Zernike), additive vs multiplicative, and the symmetry it
claims. Then pick one attack that kills that symmetry and one that
should not. Cartesian 16-32 scramble and a 2 px shift are done
(scramble kills, shift does not). True Fourier-angle scramble kills s1
at 24.1 dB. Radial-phase scramble kills at 25.0 dB. A 90° sector does
not. Y scramble kills; Cb/Cr do not. Affine, Mellin, Zernike, PCET,
and Fresnel of the 16-32 residual do not lock on photographs.

## Still open or incomplete

- Google: no public payload decoder and no usable programmatic oracle.
  The Gemini app remains quota-limited. Vertex was 403.
- `chatgpt-image-latest` is an optional third pair mint, not a blocker.
  `gpt-image-1`, `gpt-image-1.5`, and `gpt-image-1-mini` are encoder-off.
  `gpt-image-2` and dated `gpt-image-2-2026-04-21` are encoder-on, same
  16-32 stamp (corr 0.189 on gray). Latest would only matter if ChatGPT's
  current consumer id used a different ensemble. `images.generate` is 403
  until the organization is verified.
- Physical print-scan: Brother DCP-L2520DW is idle and accepting. No
  `scanimage`, no ImageCapture pyobjc. Recapture needs a human at the
  flatbed. Simulation at 24.95 dB stayed `detected`. Do not print
  unattended.
- Adding unmarked context (frame, pad, hstack, collage) to hunt a
  residual is a presentation gate, not a mark statistic. Those rungs
  belong with identity and layout below, and with
  [mark removal](synthid-removal-research.md).
- Camera-to-ChatGPT "unchanged crop" pairs are dirty counterfactuals.
  The instructed edit redraws; aligned PSNR on the frame pair was
  16.6 dB. Compare pixels only after registration and a frozen residual
  bound, as in the `gpt-image-2` red-dot row.

A Gemini app caption that an image "looks like a photograph" is not a
SynthID verdict. Use the provider pixel check or signed provenance.

## Wild hypotheses

Battery 2026-08-23, no official oracle. Local numbers:
`.local-eval/synthid/prc-oklab-attack-2026-08-15/wild-hypotheses-2026-08-23.json`.
Prepared attack rasters wait in `wild-attacks-2026-08-23/` for a later
verifier window. Jacobian / adaptive queries against
the OpenAI provenance oracle stay out: the endpoint forbids reverse-engineering.

### Tested locally, not a mark

| Hypothesis | Result |
| --- | --- |
| Chroma ordered by luma is a 1D payload | Correlation is an artifact of the sort |
| Mark is `g(x)` of a Retinex envelope | LOO unchanged vs the raw 16-32 residual |
| Nonlocal patch agreement in 16-32 | Far patches are not coupled |
| Quadratic phase coupling (bispectrum) | Statistic saturates on camera images too |
| CDMA Gold / m-sequence in the 16x16 fold | At chance for length 255 |
| Rows as a pitched waveform | COCO noise has the same peak-to-mean |
| The photo mark is the flat stamp at an unknown 16-phase | Exhaustive shift does not lift NCC |
| The photo mark is the flat stamp at a small rotation or 3% scale | Affine search lift 0.011, COCO noise 0.008 |
| Self-keyed predictor from a 16 px luma thumb | Already 0.48 vs L1, 0.50 vs COCO (`self-keyed.json`) |
| Global multiplicative cyclostationarity (H2) | Official positives below controls, 2026-08-14 |
| Mellin radial profile / log-polar \|F\| | Saturates on COCO (0.99). Annulus envelope |
| Scale chirps `r^{iα}` | Flat 0.51, photo -0.27, COCO -0.58 |
| Zernike n≤6 | Flat 0.77, photo -0.007, COCO 0.23 |
| PCET polar harmonics | Flat 0.75, photo 0.044, COCO 0.35 |
| Fresnel z=8..64 of 16-32 | Copy of cartesian LOO |
| 8-bin orientation energy | Saturates on COCO |
| Chroma 16-32 on gray flats | Cb 0.064, Cr 0.025 vs Y 0.682. Stamp is luma |

### Not a watermark, might still be an AI-vs-camera extra

Battery 2026-08-23, `remaining-hypotheses-2026-08-23.json`. Kodak corpus was empty here, camera arm is COCO plus the photo-pair sources.

| Hypothesis | Result |
| --- | --- |
| Absence of camera PRNU | Gray highpass RMS 0.25 vs COCO 14.6 was texture. Wiener residual on *photographs*: `gpt-image-2` edits 4.61 vs camera 8.05. Direction is camera-like leftover of the donor JPEG, not a missing sensor, and not a payload |
| JPEG ELA | Flat 0.49, s1 1.97, COCO 3.13. Compression history |
| CFA interpolation error (Bayer as detector) | Ratio photo-edit 0.117 vs camera 0.184 vs gray stamp 0.588. Weak camera vote, overlaps. Inverse of the Bayer remover, not SynthID |
| JPEG ghosts (Farid) | s1, a `gpt-image-2` photo, and a camera JPEG all minimize at Q90 with similar spreads. Codec, not a payload |
| PNG Paeth / filter-byte residual | `gpt-image-2` PNG is 99.9% Paeth. Camera re-saved PNG 73% Paeth, s1 74%. Export encoder fingerprint |
| Perfect-circle / text-edge rate | Hough circles per MP 385 (edits) vs 536 (camera). Edge fraction 0.052 vs 0.072. Too noisy for a payload or a provider |
| Face-gated 16-32 | s1: 3 Haar faces, inside RMS 12.6 vs outside 10.9. One photo with faces: outside/in 1.49. No systematic "spend the mark off faces" |
| CLIP-space watermark | That is Model 1. Not a payload decoder |

### Non-local, 2026-08-23

Battery `.local-eval/synthid/prc-oklab-attack-2026-08-15/nonlocal-2026-08-23/`.
Oracle after AI-metadata strip, 25 s gap, then one dated-id retry after a 429.

| Hypothesis | Result |
| --- | --- |
| HEIF q80 / q50 | 46.3 / 39.3 dB, official `detected` |
| AV1 CRF 32 still | 37.2 dB, `detected` |
| Print-scan simulation (0.35 deg, blur σ 0.9, 300/360 dpi, JPEG q42) | 24.95 dB, still `detected`. Same PSNR class as 16-32 scramble, which kills |
| Physical print-scan | Brother DCP-L2520DW idle and accepting. No `scanimage`, no ImageCapture pyobjc. Not printed unattended |
| `gpt-image-1.5` gray reproduce | C2PA `created, converted`. Oracle `not_detected`. 16-32 corr vs `gpt-image-2` stamp 0.003. Encoder-off. Dirty redraw 24.0 dB |
| `gpt-image-1-mini` gray reproduce | Same C2PA. Oracle `not_detected`. Corr 0.003. PSNR 31.1 dB |
| `gpt-image-2-2026-04-21` gray reproduce | C2PA `created, converted, watermarked.unbound`. Official `detected`. 16-32 corr vs unversioned `gpt-image-2` stamp 0.189. Same encoder family. PSNR 47.5 dB |
| `chatgpt-image-latest` generate | 403, org verification. Listed among image models, not callable |
| DIRE SDXL VAE + DDIM float32 | VAE RMS 512: gray 1.24, s1 9.15, photo edit 9.89, camera 11.84. DIRE RMS: gray 2.40, photo 30.9, s1 31.5, camera 33.0. Texture rank, not a payload |

### Still blocked after that battery

| Hypothesis | Why it stays blocked |
| --- | --- |
| Physical print-scan recapture | Human at the flatbed |
| Encoder ensemble that does not recognize itself | Needs another actual SynthID encoder/decoder pair. Extra OpenAI ids did not mint one. Open-proxy transfer is already closed: source-matched TrustMark, VideoSeal, DWT-DCT, and WAM pairs produced held-out-family AUC 0.437-0.562, so another generic public pair does not answer this hypothesis |
| `chatgpt-image-latest` as a third pair mint | `images.generate` 403 until org verification. Optional: only if ChatGPT latest used a different ensemble than `gpt-image-2` |

Do not train on official decoder bits beyond the frozen L1 study. Do not
mint a 128-photo student on the flat stamp. The information budget and
the 2026-08-23 residual measurement both say the photo-domain mark is
under the scene.

## Oracle and seeds

Official `POST /v1/content_provenance_checks`, metadata stripped, pixels
preserved. Raw files stay gitignored under
`.local-eval/synthid/prc-oklab-attack-2026-08-15/`.

| Seed | Size |
| --- | --- |
| s1 | 1024x1536 |
| s2 | 1024x1536 |
| s3 | 1086x1448 |

All three `detected` before attacks.

## Source-matched pairs (2026-08-21)

`gpt-image-2` (ChatGPT Images 2.0) can mint encoder-on/off pairs.
`chatgpt-image-latest` is optional and 403 until org verification.

| Input | C2PA | Oracle | Aligned PSNR |
| --- | --- | --- | ---: |
| Camera JPEG 960x1280 | none | not_detected | - |
| ChatGPT UI + frame 1086x1448 | watermarked.unbound | detected | 16.6 dB (redraw) |
| `gpt-image-1` edit, high fidelity, red dot | created, converted only | not_detected | 30.9 dB |
| `gpt-image-1.5` gray reproduce 1024 | created, converted only | not_detected | 24.0 dB (redraw) |
| `gpt-image-1-mini` gray reproduce 1024 | created, converted only | not_detected | 31.1 dB |
| `gpt-image-2` edit, red dot 1024x1536 | watermarked.unbound | detected | 28.2 dB |
| `gpt-image-2-2026-04-21` gray reproduce 1024 | watermarked.unbound | detected | 47.5 dB |

The `gpt-image-2` photo residual is broadband (std 9.1), not a 16-32 stamp.
COCO `gpt-image-2` edits: 11/12 `watermarked.unbound`, one safety block.

Solid gray/white through `gpt-image-2` "reproduce exactly" is
`watermarked.unbound`. Gray 128 at 1024x1536: PSNR 41.9 dB vs the flat,
source `not_detected`, output `detected`. Other gray levels 32-220: PSNR
38-49 dB. Eight gray 1024x1536 residuals, leave-one-out correlation after
highpass: 0.645 (shuffle 0). That shared stamp does not transfer to COCO
photo residuals (0.007-0.025).

## Presentation gate versus residual

An official `not_detected` is not one event. Collage, photo-pad, two-panel
self-concat, and 8x8 scramble can be a *presentation gate* (this upload is
not a single photograph). Blur sigma 7 and in-band 16-32 destruction sit on
the single-photo manifold and are residual damage; those belong in
[mark removal](synthid-removal-research.md).

Do not train a surrogate on mixed collage misses if the goal is a mark
student.

Identity (s1, 2026-08-19): JPEG q80/q20, crop 2 px, crop 8 px, keep 0.75
and 0.50, rot90, rotate 5, flip, grayscale all stay `detected`.

Token / layout (s1): each quadrant at 512x768 `detected`; `hstack` and
`vstack` of marked\|marked `not_detected`; 4x4 tile scramble `detected`;
8x8 `not_detected`. Two copies of a detecting image still miss. Tomography
3x3: all nine cells `not_detected` (a 1/9 window is below support).

Preprocess E1 (s1): stretch 2x `detected`; centre-crop of marked\|marked
hstack back to native `detected`; 0.20x pixels on a native-size white
canvas `not_detected` (the same 0.20x file uploaded alone was `detected`).

## L1 is not a mark task

A missing `watermarked.unbound` assertion is not a clean negative. On
2026-08-16 the official verifier called 65 of 94 OpenAI rows without that
assertion `detected` (69%), with 9 of 9 interleaved health positives
detected so the endpoint was answering. Google's own API emits watermarked
images with no assertion, per its documentation. Corpus AUCs that treat
"no assertion" as unmarked rest on a negative class that is roughly
two-thirds positive.

Control-only 283 rows: 203 detected, 80 not_detected. Forward-temporal
nested ridge on 124 pixel features: AUC 0.649 / 0.641. Geometry-only:
0.781 / 0.826. All 21 `claim_generator=ChatGPT` controls are
`not_detected`. C2PA `opened`: 24/24 `not_detected`. Inside 1254x1254,
mean-feature permutation p=0.45.

OpenAI-supervised ridge on 16-32 band-passed `image_features`:
forward-temporal AUC 0.53 vs L1 `not_detected`, 0.95 vs COCO, 0.97 vs COCO
after a 2 px crop. The student learns OpenAI-versus-photo and still does
not see the oracle mark contrast.

## Runtime product remainder for the watermark

Signed provenance (`identify`). The remote OpenAI oracle is development-only,
and a research lattice miss is not a clean SynthID negative.

## JPEG and resize development, 2026-10-21

> Development-label update, 2026-10-23: exact provider evidence is complete
> for every fixed cohort. Earlier incomplete snapshots below retain their
> chronology. The combined sensitivity targets and fresh acceptance remain unmet.

A periodic-residual candidate using fixed canonical image sizes improved
grouped development predictions over a fixed-scale baseline, including
previously missed JPEG and resized examples. Canonical sizes are internal
feature preprocessing; they do not supply watermark labels for new
derivatives. Every evaluated derivative requires its own provider evidence.

This result remains exploratory. Feature, hyperparameter and operating-point
selection consume the development data, and the transformed positive cohorts
are small. Selecting a threshold above the observed development negatives
does not establish a low false-positive rate on unseen images. The broader
geometry cohort and complete derivative labeling must finish before selecting
and freezing the next candidate. A completely fresh source-group holdout with
per-condition rates and confidence intervals is required for acceptance.

After completing the additional native-geometry development cohort, JPEG
false negatives remained even when the provider confirmed the watermark on
those exact JPEG bytes. Earlier small-subset success therefore does not
establish JPEG robustness. Canonical-size preprocessing and threshold
calibration alone have not resolved this gap. Final candidate selection and
fresh acceptance remain pending.

A subsequent native residual-phase template reduced the JPEG development
gap when combined with the residual-statistics candidate. Its source parents
remain separate during grouped template fitting. Calibration checks both
out-of-fold negatives and the final fitted template, since refitting can move
negative scores across an earlier cut. This still consumes development data,
and the small template cohort requires fresh independent confirmation.

A subsequent screen on intermediate downscales and upscales still found
prediction changes. Those transformed files require their own provider
evidence before a change can be called a detection error. Arbitrary-scale
robustness remains unconfirmed; the fixed resize development cohort is
retained in full for that check.

A same-provider specificity challenge is also required before acceptance.
Historical summaries identified disagreements on retained band-perturbed
OpenAI images, including artifacts whose archived byte hashes still match.
A new exact-byte provider check has now confirmed a false positive on a
retained spectral intervention. The remaining historical summaries do not
supply current labels when their full vendor payload is absent. Native OpenAI
archive controls
are selected independently of candidate scores; their model attribution does
not supply watermark labels. Missing historical originals cannot be scored or
silently substituted with other files. Fresh acceptance must exclude these
newly opened images and their known source and prompt identities as well as
the earlier development inventory. A filesystem census of the historical
archive does not establish calibration coverage. Reconcile retained
same-provider counterexamples across the main checkout and active worktree
against both the evaluated cohort and its source-exclusion manifests.

New exact API checks have already confirmed false positives on retained
OpenAI controls, including a native export. The combined candidate therefore
needs same-provider negative calibration before promotion. A separate fixed
intervention cohort pairs current provider-positive inputs with ordinary
controls, using the same predeclared spectral operation. Source labels are
known; intervention labels must be obtained independently. No intervention is
adjusted using the provider's answers, and the full prepared cohort is retained.

Provisional fits may use all currently exact-labeled development artifacts,
with unknown labels explicitly retained in the fixed pool. Such a snapshot
cannot satisfy the final-selection gate. Both the archival controls and the
paired intervention cohort must reconcile before that gate can pass; an older
supervisor covering fewer cohorts is insufficient.

An initial source-balanced refit corrected the newly confirmed OpenAI false
positives in grouped development predictions. Parent/class-balanced loss gives
each source within an exact class equal total weight despite unequal numbers
of transformed views. This does not make model selection independent. A new
phase-coupling descriptor was also compared on the same snapshot; it did not
improve the selected candidate and was not adopted in its inference program.
Resize results must report the number of independent source parents alongside
the number of transformed files. For repeated transforms of one source,
report parent-level all-view sensitivity and any-view false-positive risk
with their own intervals, separately from per-view rates. No observed false
positives in a small or adaptively selected development set establish a
population false-positive bound.

A separate candidate learns interactions between reference-phase matches and
residual statistics. Its reference features require nested source exclusion:
a held source is absent from every template used for that fold's training and
validation features, and a training row also excludes its own parent. Final
inference uses the complete frozen reference set, so calibration checks those
inference features separately from the training features. Selection checks
sensitivity on verified positive challenge examples alongside specificity;
stronger primary-route rates alone do not clear a challenge-positive regression.

Adding exact negative intervention labels removed the apparent robustness gain
of the earlier incomplete-specificity snapshot at the required false-positive
limit. A targeted comparison of separate frequency-band statistics and phase
matches also failed the development targets. Earlier low-band energy and
fixed-template failures remain relevant; another threshold sweep cannot
supply the missing source-paired mark contrast. Complete current-source
intervention labels are required before selecting an improved detector.

A bounded extra-trees comparison used the same fixed feature banks and exact
label snapshot, with parent/class-balanced training and nested source
exclusion for reference features. It also failed the combined development
targets. Changing the learner did not restore JPEG, resize, and intervention
sensitivity at the required false-positive limit. This comparison does not
justify a new acceptance campaign or additional feature sweeps.

The retained archival challenge is now completely labeled with current exact
provider evidence. It confirms both native same-provider negatives and spectral
intervention negatives, while many other transforms retain positive labels.
Forecasts recorded before the remaining answers still miss the combined
sensitivity and specificity targets. The newly labeled views remain development
data; their pre-existing forecasts do not constitute untouched acceptance.

The fixed current-source intervention cohort is also completely labeled. It
contains both changed and retained positive outcomes, with controls subjected
to the same operation. Source and derivative verification can occur at
different times; retain both provider timestamps and do not treat a changed
response alone as proof of causal watermark removal. Refitting the fixed
feature families on this cohort
still failed the joint development targets. Retaining the unprojected low-band
phase coordinates in a separate controlled comparison also failed to resolve
the gap. Feature displacement by itself describes the operation, not mark
presence. Every newly returned variant remains part of development.

A matched-operating-point comparison recalibrates the frozen v9 scores under
the same development false-positive constraints. The later candidate improves
grouped development predictions for native, JPEG, downscaled, and upscaled
inputs over that recalibrated baseline. This measured development gain still
falls short of the combined sensitivity targets and requires new untouched
acceptance. Keep the original v9 operating point separate from this
matched-FPR comparison. Standalone pixel inference has been reproduced
against the saved feature-bank predictions across the archival, intervention,
resize, and primary-route cases with external network access disabled. This
checks implementation parity, not independent detection quality.

A compact residual-network comparison uses fixed native RGB patch bags as
internal features under a single whole-image label. Patches are not separately
labeled evaluation artifacts. The experiment keeps source groups intact and
checks that a prediction for one image is independent of other images in its
inference batch. The [SRNet paper](https://ws.binghamton.edu/fridrich/Research/SRNet.pdf)
motivates learned residual extraction before spatial aggregation and warns
about pair-conditioned evaluation. This custom compact architecture is not a
reproduction of its published results or an established SynthID decoder.
The completed fixed recipe had strongly overlapping positive and negative
score distributions, weak source-held discrimination, and poor ordering of
same-source changed-outcome pairs. Its calibrated operating point failed
sensitivity and was not adopted. Saved weights reproduced actual local CPU
image inference with external operations blocked; implementation correctness
did not supply detection quality.

A controlled addition of source-paired ranking used the same data, source folds,
architecture, initialization seeds, base-image shuffle, and final epochs in
both arms. It improved held-source ordering for changed-outcome pairs, but the
score also fell for same-operation positive and negative controls whose provider
outcome did not change. That ordering is therefore insufficient evidence of a
mark-specific signal. The standalone neural operating point still failed the
sensitivity targets. A fixed two-score composition with the prior statistical
candidate showed a native-development gain at the same false-positive
constraints. Its standalone single-image program now reproduces saved scores
and decisions across the fixed actual-image verification cases with external
operations blocked. This composition remains provisional, falls short of the
combined development targets, and still needs untouched acceptance.

A subsequent controlled comparison added a consistency penalty to explicit
same-outcome source/derivative pairs of both classes. The ranking-only control
reproduced the preceding experiment's weights and training histories exactly.
The added penalty did not isolate the mark: changed-outcome and unchanged-outcome
pairs still showed a similar processing response. A fixed union of the two
statistical/neural compositions improved development coverage, including resize
cases, but continued to miss retained-positive interventions and failed the
joint targets. Its actual single-image scores and decisions reproduced saved
predictions with external operations blocked, and a mutation of the union rule
changed the expected exclusive-expert case. This remains adaptive development;
implementation parity and a processing response do not establish SynthID
specificity or independent acceptance.

The complete fixed development pool has now been reconciled against full
provider responses and current artifact bytes. The final comparison of all
predeclared statistical feature families, and the recorded neural candidates,
still failed the joint targets. Combining statistical and neural scores improves
coverage, but continued misses on retained-positive interventions prevent
promotion. An earlier fixed composition also violates a per-route false-positive
constraint on a newly labeled resize condition despite a low aggregate
false-positive rate. The aggregate therefore cannot replace condition-specific
checks. A complete development label inventory permits final selection; it does
not itself supply a passing detector or independent acceptance.

A targeted repeatability pass obtained new provider responses for difficult
retained-positive interventions, changed-outcome derivatives, same-operation
negative controls, legacy same-provider negatives, native positives, and the
new resize false positive. All repeated outcomes agreed with the original
adopted labels. Full responses had new identities and creation times, and the
input hashes still matched. This is a purposive development check, not an
estimate of population label error or independent decoder executions. Repeated
bookkeeping entries for one response do not count as repeated checks, and the
original labels remain separate from this corroborating evidence.

A controlled native-only versus native-plus-canonical patch comparison used
identical learned parameter shapes, exact labels, source folds, initialization
seeds, objective terms, and fixed epochs. Additional canonical patches improved
ranking discrimination, but neither neural arm supplied the required standalone
sensitivity at the false-positive constraint. Processing responses still occurred
in changed-outcome and unchanged-outcome controls. Combining fixed statistical
and neural components subsequently met the primary native, JPEG, and resize
point targets in development. Actual standalone pixel inference reproduced the
saved component scores and calibrated decisions, including boundary cases, with
external operations blocked and shared extraction counted explicitly. The
retained-positive intervention guard still failed on held sources even when the
final fit covered those training examples; untouched acceptance remains pending.

The training audit also found that changed-outcome ranking pairs all used one
predeclared transform producer, while equal-outcome controls from that producer
received only a small share of the general consistency sampler. This leaves an
operator cue competing with sparsely sampled matching controls. Comparing the
sampling scope requires the same image data, loss weight, architecture, and
training budget in both arms; the observed imbalance alone does not establish
that reweighting will recover a mark-specific signal.

The completed operator-matched sampling comparison improved overall source-held
ranking discrimination. Its unchanged control reproduced all learned tensors,
epoch histories, and CPU predictions exactly. Matching the consistency sampler
did not close the retained-positive intervention sensitivity gap, either alone
or in the predeclared compositions. Reduced normalized processing responses
also occurred in changed-outcome pairs, so this reduction alone cannot establish
mark specificity. The preceding verified prototype remains the retained
reference; the new neural arm is not promoted. Further development must retain
the original difficult cases as a separate guard when expanding source-paired
coverage, rather than hiding their failures in a wider aggregate.

A complete source-paired expansion retained both exact positive and negative
outcomes from the same fixed processing operation. Its wire evidence was
reconciled against actual prepared-file hashes; source pixel identities remain
separate from derivative pixel identities. Pending responses stayed unlabeled
in the frozen training snapshot. Later labels were evaluated using the existing
decisions, without retroactive fitting or threshold selection. Reusing completed
seed/fold checkpoints after an interruption requires matching input bindings,
weight hashes, and training histories; an incomplete model must restart.

The expanded-data comparison still failed the joint sensitivity and
false-positive targets. Statistical and neural combinations recovered some
retained-positive cases, but lost primary-condition coverage, and the original
difficult cases remained a separate failed guard. Diagnostics found constrained
operating-point misses even on full-fit predictions, together with a further
source-held generalization gap. These are separate contributors: a complete
label inventory does not establish sufficient representation or optimization,
and overall ranking discrimination does not establish low-false-positive
sensitivity. Any objective-focused continuation must use the same starting
checkpoints and training budget in its control and treatment arms.

A matched warm-start comparison added training-only hard-positive and
hard-negative ranking while preserving the starting checkpoints, source folds,
optimizer schedule, and continuation budget. The extra objective did not close
the joint development targets. Increasing native spatial coverage with the
same frozen weights recovered additional source-held positives, but did not
resolve the retained-positive intervention failures. Coverage and objective
changes therefore remain distinct experiments rather than interchangeable
explanations of the remaining gap.

The earlier tree comparison used an incomplete exact-label inventory and did
not assess the subsequently labeled intervention cohort. Replaying its original
feature families and depth settings across the expanded inventories reproduced
the old selected forest parameters, source-held scores, and operating policies
before comparing new fits. Adding those new trees to the frozen coverage
baseline recovered additional positives without observed negative errors in
this adaptive development set. The original and expanded retained-positive
guards still failed, so the combination was not promoted to a new verified
standalone detector. Full-fit success on a wider cohort cannot replace
source-held performance or erase the separate original guard.

The fresh-collection prerequisite now recomputes those same expanded
development targets and requires actual-image inference coverage for the
expanded interventions. An optimistic selection flag cannot override failed
rates. The source-exclusion inventory was widened alongside the development
pool without changing the planned fresh sources or acceptance criteria.

A frozen pretrained steganalysis feature probe recovered additional
source-held cases when combined with the retained predictors. Its original
released model, tensor values, and RGB input normalization were verified;
limited variation in the deep features was not evidence of a loader or
normalization failure. Early residual features contributed more than the
largely constant deep representation. Monotone coarse/readout score
compositions still missed the retained-positive guards, even under an
optimistic sensitivity bound that ignored the union's false-positive risk.
This limits that fixed score family, not steganalysis representations as a
whole. The combined scores remain adaptive development results requiring
actual standalone inference verification and untouched acceptance.

Full-image SRMQ1 and color rich-model extraction were checked against the
pinned implementation's bundled reference fixtures before fitting new heads.
The library merges some submodels through an unordered set, so flattening its
dictionary directly can change column identities between Python processes.
A canonical ordering by submodel name preserves their values and reproduced
identical vectors across independent processes. Histogram acceleration was
also checked against the original operators and full-image vectors; neither
implementation parity nor broader pixel coverage establishes detection quality.

Heads fitted on these full-image features recovered additional source-held
positives without adding observed negative errors to the previous composition.
A random-subspace FLD adaptation did not improve that composition. It reused
the original linear learner but sampled class-parent strata independently,
rather than reproducing the reference paired-cover/stego bootstrap or its
published error estimates. Bounded score conjunctions subsequently met the
observed format and archival thresholds in adaptive development. Both separate
retained-positive guards still failed in source-held predictions. An optimistic
bound for the wider frozen-score rule family also remained below those guard
targets, so that bound cannot justify further threshold tuning as a route to
the complete goal. Histogram-specific classifier comparisons continue with
the same source exclusion and quality criteria.

A histogram-kernel SVM probe subsequently passed the original retained-positive
guard while preserving the observed format and archival development targets.
The expanded retained-positive guard remains unmet. A matched follow-up changed
only class/condition loss weights, with feature selection and bandwidth estimation
shared between the control and treatment. It recovered additional source-held
positives without closing the expanded guard. A subsequent selector/loss factorial
comparison added no improvement. Exact parameter replay,
like-for-like score replay, and native-versus-portable numerical equivalence
are separate checks. Control models are matched by their explicit feature view
and recipe fields, rather than list position; different studies enumerate the
same recipes in different orders. Input preflights exercise the production data-loader path
and bind canonical corpus metadata independently from experiment revisions.

A bounded repeatability audit sent the same immutable missed positives and
near-threshold controls through each configured API account. Every repeated
watermark verdict agreed with its original label, with distinct vendor requests
and payloads reconciled against the completed batches. This purposive check does
not estimate a population label-error rate, replace the original labels, or
constitute independent detector acceptance. A capability bound for conjunctions
of the newly fitted histogram-kernel scores and frozen coarse scores also failed
the expanded guard, so further threshold tuning of that family was stopped.
Intervention-only expert comparisons retain the same source-parent exclusions;
confirmed source outcomes determine training eligibility, never inference inputs.

Intervention-only histogram experts and a phase-kernel comparison recovered
additional positives, but their bounded conjunctions still failed the expanded
guard. The selected new phase head used phase features alone, so the improvement
does not establish an advantage for mixing its kernel with rich-model features.
Direct signed-phase vectors without reference projections also failed to close
that guard. Their gated linear scores nevertheless contributed to a stronger
bounded interim composition. These completed score families are not extended
by further parameter searches.

Native patch-bag identity is audited separately from whole-image identity.
Different source images can yield identical model-input pixels even when their
full decoded images differ. The pretrained-network fine-tuning comparison
deduplicates training bags and excludes every bag represented in a held source
fold, in addition to excluding that fold's source parents. Both loss arms use
the same exclusions. This correction does not relabel or remove whole-image
evaluation cases.
The draft independent-collection protocol also checks current native model-input
identities before oracle labels. Check each transformed upload separately:
admitting a source image does not establish independence of its JPEG or resized
model input. A collision remains an unresolved planned case;
it is not replaced or silently removed. Older unavailable historical inputs
remain an explicit coverage limitation. Actual-image adapter checks separately
exercise the NumPy/PIL boundary, native crop pixels, full phase vectors, and
portable classifier scores; they do not constitute independent acceptance.

Apply sparse-input exclusions against the declared development inventory.
Identical sparse crops from fresh parents do not imply that their complete RGB
inputs match. Use complete RGB identity to check image duplication within the
fresh cohort, separately from the sparse-input check against development.

Check necessary conditions imposed jointly by class-support floors and
false-positive caps before completing a labeling campaign. A finite control
stratum may allow no false positives at all. If too few predictions are
negative to supply its required negative class, no possible label assignment
can pass acceptance. Preserve the failed or incomplete frozen evaluation and
retire it before using its inputs for development. Exact labels then distinguish
classifier errors from insufficient negative-class support in the cohort.

The fine-tuned research image adapter also normalizes native crop storage to
the contiguous layout used by its training and calibration bank. Equal pixel
values, dtype, and shape alone did not reproduce the same floating-point
readout when array strides differed. A regression through the actual runtime
method and an omission mutation check guard this conversion. It changes no
model weights, labels, or quality targets. Cold-process verification also
primes local platform metadata before blocking subprocess creation, because
an imported dependency queries that metadata during initialization. Network
and subprocess restrictions remain active for model execution.

Matched pretrained-network fine-tuning comparisons keep loss arms, paired
seeds, and source-held folds aligned. CPU replay verifies GPU scores within a
declared tolerance. Bounded combinations and combinations using per-condition
false-positive allowances require the same expanded retained-positive guard.
Ranking against mixed negative sources does not establish same-provider
watermark specificity.

A subsequent parameter audit found widespread subnormal values in the retained
checkpoint. Nominal trainability did not imply effective feature capacity:
late feature activity remained restricted after the matched fine-tuning runs.
Some subnormal values also round to zero in the training precision. A local
initialization experiment replaced only zero or subnormal parameters with values
from the same network constructor, preserving normally sized parameters and
all buffers. A real backward-and-optimizer-step check distinguishes effective
feature capacity from nominal trainability. Matched comparisons evaluate
source-held detection quality separately from feature activity. Actual RGB
inference must reproduce the selected classifier's decisions before the
candidate is frozen. Development point-rate targets do not substitute for
independent acceptance, which freezes the candidate, thresholds, source
identities, transformation recipes, and acceptance criteria before collecting
fresh inputs.

A specificity veto can only reject a base detector's positive decisions. It
cannot recover that detector's misses. Compare lost positive detections with
removed false alarms under the original route and retained-positive guards.
Training-negative score separation can overstate separation on unseen sources;
keep negative calibration sources separate from head fitting and outer
evaluation. A threshold chosen from all development negatives is a development
operating point, not an unbiased false-positive estimate.

For a head trained on neural embeddings, use the same outer-fold encoder for
its training, calibration, and held inputs. Training a head on a mixture of
out-of-fold embeddings can leak outer held labels indirectly through encoders
trained on other folds. When the composition is a conjunction, unfiltered
baseline false alarms form a conservative upper bound for negative cases whose
additional features were not extracted; positive recall still requires the
actual composed scores.

A matched warm-start comparison added exact current same-provider negatives
while retaining the prior positive and transformed-image guards. Neither loss
weighting restored the joint operating point. Correct ordering within a changed-
outcome pair can coexist with false positive decisions on both members; report
absolute operating-point errors alongside pair direction. Keep a complete newly
verified source-pair expansion separate until its intended training or validation
role is frozen, and retain unchanged positive outcomes of the same operation.

Adding matched source views to a shuffled dataset does not itself make the
training batches paired. Reconstruct the recorded sample and patch trace before
claiming that an experiment used within-batch source contrast. Class and source
loss weights balance expected contributions, while scarce high-weight cases can
still appear infrequently within minibatches. A sampling comparison must retain
a control with the same marginal sampling probabilities and correct importance
weights; pair coupling is a separate experimental factor from oversampling.

Inference-mode stress checks distinguish an implementation mismatch from a
failed classifier. For this network, recomputing BatchNorm on only one image's
patch bag collapses the mean linear readout toward its affine bias. That is not
a single-image remedy. Compare stored statistics with mixed-image training
batches separately. A small CPU precision probe does not establish training-
precision equivalence or repair detection quality. If no candidate passes the
GPU quality screen, record subsequent CPU qualification as skipped, never as a
successful CPU quality check.

A matched full-fit training-precision comparison kept initialization, sample
order, patch draws, objective, and backend fixed. Fixed CPU spot checks
reproduced both variants’ readouts, but changing autocast did not repair the
tested false-positive tail.
High training AUC and sensitivity at an arbitrary zero logit do not establish
sensitivity at the required false-positive operating point.

A subsequent batch comparison exactly replayed the original weight tensors and
prediction arrays before testing padding, marginal oversampling, and matched
pair coupling. Greater full-fit positive retention did not clear the original
quality guards. Pair coupling produced a tradeoff relative to its unpaired
sampling control. A failed full-fit prerequisite does not justify reporting a
source-held quality result for folds that were never trained.

For a head attached to an adapted encoder, excluding calibration rows from the
head alone does not exclude them from the encoder. Keep this distinction
explicit when the encoder has already learned those labels. Pooled source-held
and full-fit negative cuts remain development calibration; fresh acceptance is
still required. Compare exported numeric heads with the actual fitted estimator,
including held inputs, and check that held rows cannot change fitted scaling.

A solver that stops at its iteration limit has not completed the intended
comparison. Preserve the first result, complete the same fixed candidates to
verified convergence, and check exported inference against the fitted library
before attributing a quality failure to the representation. Greater full-fit
retention from a flexible kernel does not establish transfer to excluded source
parents. Audit the actual loss mass by class and domain before attributing that
gap to data imbalance.

Predicting the provider's outcome does not isolate a causal watermark signal
or demonstrate cross-provider coverage. These experiments do not change
`identify`, the shipped CLI, or the product capability claims above.

## Replaying preserved research inputs

Descriptor replay, fixed-patch training, and full-image replay have separate
input requirements. A retained, hash-bound descriptor bank can support new
learners without reopening source images. A retained raw patch bank can support
new pixel-model training under its recorded extraction contract. Neither bank
reconstructs the original image or permits arbitrary new pixel transformations.
Verify every reused bank against its original receipt, preserve exact outcome
labels and source exclusions, and report coverage separately for each capability.

When equal native patch inputs occur under distinct source-parent identifiers,
keep their connected source groups together or exclude the matching input from
training. Source-disjoint fitting and threshold calibration must both exclude
the outer evaluation groups. An actual-RGB replay with descriptor-cache reads
denied separately verifies exported feature inference; it is not an independent
quality evaluation.
