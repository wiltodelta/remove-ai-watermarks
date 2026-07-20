# Full verification plan

How we convince ourselves the library actually works, across its whole surface, on real data.

This is the pre-release and periodic-audit plan. It is deliberately organized by **oracle
strength** rather than by module, because the hard part is never "call the function" -- it
is "know what the right answer was". A sweep with no oracle proves only that nothing threw.

Measured throughput on the local corpus (39,430 images, M-series, 2026-07-19):

| path | per image | full corpus, 8 procs |
|---|---|---|
| `identify` | 0.58 s | ~0.8 h |
| `detect_marks` | 0.28 s | ~0.4 h |
| visible remove (cv2) | 0.58 s | ~0.8 h |
| diffusion @512px (MPS) | ~50 s | ~23 days -- sample only |

So every CPU path is affordable at FULL corpus scale; only the diffusion paths need
sampling. Plan accordingly: never sample where a full sweep costs an hour.

## Data sources

| source | size | committed | role |
|---|---|---|---|
| `data/spaces/originals/` | 39,430 imgs, 87.5 GB | no (gitignored) | the real-upload corpus |
| `data/spaces/identify/` | 39,314 JSON sidecars | no | **recorded `identify` verdict per image** |
| `data/spaces/_visible_datasets/` | 3,741 imgs, per vendor | no | mark-positive pools |
| `data/synthid_corpus/` | 39 imgs, labelled | yes | pos/neg/cleaned SynthID references |
| `data/samples/` | 11 fixtures | yes | deterministic fixtures |
| `data/*_capture/` | vendor captures | yes | detection silhouettes |
| synthesized | generated | no | constructed ground truth (tier B) |

**Data safety.** The corpus is user uploads: local analysis only. No run may copy, promote,
or commit corpus images into a tracked path, and no report may embed them. All harness
output goes to gitignored paths under `data/spaces/`.

## Tier A -- self-evident oracles (full corpus, unattended)

Properties that are true or false without anyone labelling anything. These are the
backbone: they scale to 39k images and catch regressions with zero human cost.

### A1. Sidecar regression -- the highest-value check we are not running

`data/spaces/identify/` holds 39,314 recorded `identify` verdicts, keyed by the same uid as
the image. Re-running `identify` today and diffing against them turns the corpus into a
**39k-image behavioral regression suite for free**. Any drift in verdict, platform,
confidence or signal set shows up as a diff, bucketed by cause.

Caveat that makes this honest: a diff is not automatically a bug -- the sidecars were
written by older versions, so intended improvements also show up. The output is therefore a
**classified diff** (new detections / lost detections / changed platform / changed
confidence), reviewed once, then re-baselined. Lost detections are the alarm.

Implemented as `scripts/sidecar_regression.py` (resumable, ~1.5 h at 8 workers).

#### First full run, 2026-07-19, all 39,314 sidecars

| class | n | share |
|---|---|---|
| unchanged | 37,326 | 94.9% |
| lost_signal | 1,302 | 3.3% |
| platform changed | 922 | 2.3% |
| confidence changed | 899 | 2.3% |
| new_signal | 694 | 1.8% |
| lost_ai | 747 | 1.9% |
| new_ai | 152 | 0.4% |

Two results worth keeping:

**No metadata signal regressed anywhere.** Lost families were exclusively visual
(`visible_sparkle` 1,256, `visible_doubao` 45, `visible_jimeng` 4) -- zero c2pa, synthid,
aigc_tc260, iptc, exif_generator or xai_signature losses across the whole corpus. And
`identify` raised on **none** of the 39,314 real uploads.

**The sparkle losses are mostly corrected false fires, but not entirely.** Sampling 400 of
the 1,256 and checking whether the file still carries Google provenance independently of
the sparkle: 12.5% (95% CI 9.6-16.1%) still do, i.e. ~120-200 corpus-wide are **genuine
misses**; the other ~1,050-1,135 had nothing backing them. Read the split as a trade the
FP-gate tightening made, not as a clean win.

Caveat on that split: "no Google provenance" is not proof of a false positive -- a
metadata-stripped Gemini screenshot also has none while still carrying the pixel sparkle.
So 87.5% is an **upper bound** on false fires; only the 12.5% genuine-miss figure is solid.

Doubao moved the other way (-45 / +628 net +583), which is the `scale_basis` landscape fix
showing up at corpus scale. `trustmark` +38 and `open_invisible` +9 are not behavior: those
extras were simply not installed when the sidecars were written.

### A2. Parity: whatever we detect, we must be able to remove

For every image where a signal fires: remove, re-scan with the same oracle, assert quiet.
- metadata: `scripts/metadata_removal_audit.py` (exists) -- run full corpus.
- visible: `scripts/visible_removal_audit.py` (exists) -- run **once per backend**
  (cv2 / migan / lama). It is single-process, so a full-corpus sweep is ~10 h and three
  backends ~30 h. Its expensive half is DETECTION, which does not depend on the backend,
  so run `scripts/visible_positives.py` once (parallel, ~40 min) and feed the result to
  the audit's `--paths-file` seam: a few thousand images per backend instead of 39k.

#### Metadata parity, first full run, 2026-07-19 (20,153 carriers + 1,500 clean controls)

Zero scan/strip/decode errors. Survival after strip:

| signal | carriers | survived |
|---|---|---|
| c2pa_manifest / claim_generator | 15,410 | **3** |
| synthid_watermark | 14,985 | 0 |
| aigc_label | 4,414 | 0 |
| the other 12 signal types | - | 0 |

The no-op control is clean: the strip **added** a signal to 0 of 1,500 clean images. Two
real defects fell out of the run.

**Defect 1 -- the fail-safe reports success on a file it did not strip.** All 3 parity
failures are Samsung Galaxy S22 camera PNGs (`Galaxy S22 c2pa-rs/0.37.0`) whose `caBX`
chunk survives. Cause: PIL raises `UnidentifiedImageError` on them, so
`remove_ai_metadata`'s fail-safe copies the file through byte-identical -- correct in
intent (never crash a worker on a partial upload) but it returns an output path
indistinguishable from a real strip. User-visible: `metadata --remove` prints
"AI metadata stripped ->", exits 0, and `identify` on the output still reports C2PA. The
warning is logged but the success line contradicts it. Rare here (3 of 20,153) but the
mechanism fires on ANY file PIL cannot decode. The fail-safe should stay; what needs
fixing is that the caller cannot tell a no-op from a strip.

**Defect 2 -- 16-bit PNGs are silently downconverted to 8-bit.** 5 of the 1,500 clean
controls failed the pixel-identity check; all are 16-bit PNGs, and the PIL re-save halves
their bit depth (one went 9.2 MB -> 2.5 MB). This is the known limitation recorded in
CLAUDE.md, now measured: a byte-level IHDR scan over every corpus PNG puts it at
**42 of 27,018 (0.16%)**.

Method note worth keeping: the first attempt to reproduce Defect 2 said "pixels
identical" and nearly closed it as a harness bug. That check read both files through
`image_io.imread`, which returns 8-bit -- **the reader destroyed the very property under
test**. The audit was right because it reads via `read_bgr_and_alpha`, which preserves
uint16. When verifying a fidelity property, check that the verification path can still
represent it.

### A3. Byte-level invariants

- no-op `remove_visible` returns the ORIGINAL bytes (not a re-encode)
- pixels outside the fill mask are bit-identical to the input
- JPEG metadata strip is pixel-lossless on the DEFAULT path (`--remove-all` re-encodes by
  design -- see `metadata.py`; assert the split, not losslessness everywhere)
- lossless source formats survive a misnamed extension

### A4. Idempotence and order-independence

- `remove_visible(remove_visible(x)) == remove_visible(x)`
- `strip(remove(x)) == remove(strip(x))` in signal terms
- a second `identify` on a cleaned output reports no metadata signals

### A5. Contract sweep across every parameter choice

`scripts/smoke_matrix.py` (exists, 68 rows, 0 skipped with `--diffusion`) covers every
choice-valued flag on fixtures. Extend from fixtures to a stratified corpus slice
(~500 images spanning format x provenance x aspect ratio), asserting exit-code semantics
rather than just absence of crash.

Known trap to encode: **exit 2 is triply overloaded** (no-visible-mark, no-invisible-signal,
Click usage error). A wrapper cannot distinguish them without parsing stderr. Either the
sweep asserts on stderr, or the codes get split -- the latter is the better fix.

#### Coverage before the extension (measured 2026-07-19, not estimated)

Comparing the flags the matrix actually executed against the flags the CLI declares:
**18 of 38 had never been executed even once** -- `--pipeline`, `--strength`, `--steps`,
`--guidance-scale`, `--device`, `--model`, `--upscaler`, `--tile`/`--tile-size`/
`--tile-overlap`, `--humanize`, `--unsharp`, `--adaptive-polish`, `--controlnet-scale`,
`--min-resolution`, `--hf-token`, `--auto`, `--verbose`. Plus uncovered VALUES of covered
flags: `--backend migan|lama` had never been driven through the CLI at all (only at
library level), `erase --backend` only ever ran `cv2`, `batch --mode all` never ran, and
`--pipeline` only ever ran its default.

Whole subsystems had unit tests but no real-data run: tiling (27 unit tests, never
processed a real image through the CLI), the region-targeted composite, the ESRGAN
upscaler, and the ffmpeg audio/video strip. The gap is not "logic untested" but
"never executed on real data", which is precisely what this campaign is for.

#### Bug found by the extension: `--steps` below ~7 crashes inside torch

Effective timesteps are `int(steps * strength)`. At the vendor-adaptive default strength
(0.15, or 0.10 for OpenAI) any `--steps` under 7 rounds to **zero**, and the pipeline dies
with a raw traceback:

```
$ remove-ai-watermarks invisible img.png --steps 5
RuntimeError: cannot reshape tensor of 0 elements into shape [0, -1, 1, 512]
```

Fully valid CLI arguments, no special flags, no `--force`. The value is accepted, the
crash is a torch internal, and nothing tells the user that steps and strength interact.
Fix is either a clamp to >=1 effective step or an up-front validation naming both values.

Method note: the first run of the knob rows failed 12 times with this identical error,
which read like twelve broken features. It was one bad harness parameter (`--steps 4`)
sitting on top of one real bug. An error that is IDENTICAL across unrelated rows is
evidence of a common cause, not of many faults -- check the shared input first.

## Tier B -- constructed ground truth (automatable, no labelling)

Where reality gives no answer key, build one. This is the tier that closes the two biggest
holes: fill quality has **never** been asserted, and recall was measured once at n=240.

### B1. Fill quality with a true reference

Real marks have no clean counterpart, so quality has only ever been eyeballed. Construct it
instead: take a clean corpus image, stamp a known mark at a known position (the captured
alpha maps make this exact), remove it, and compare against the **true original**.

Yields PSNR / SSIM per `--backend` (cv2 / migan / lama), sliced by background class, which
is exactly the axis where the docs say quality varies but no number exists.

Implemented as `scripts/fill_quality.py`. Two reporting rules are load-bearing: score
INSIDE the footprint (whole-frame PSNR sits near 60 dB whatever the backend does), and use
the MEDIAN (a fill that reproduces a flat background exactly scores PSNR=inf, and one inf
makes a mean inf -- the first run reported "+inf" for every flat bucket).

#### First run, 2026-07-19, 80 verified-clean sources, n=720 measurements

Median dB recovered inside the footprint (filled PSNR minus damaged PSNR):

| mark | bg | cv2 | migan | lama |
|---|---|---|---|---|
| doubao | flat | +10.79 | **+15.62** | +13.97 |
| doubao | mid | +9.25 | **+14.71** | +14.69 |
| doubao | textured | +2.15 | +0.66 | **+1.64** |
| jimeng | flat | +5.52 | +8.80 | **+10.30** |
| jimeng | mid | +10.78 | +11.05 | **+12.33** |
| jimeng | textured | +2.16 | +1.70 | **+3.03** |

**The `auto` order is CONFIRMED.** Median per-image recovery on the textured tercile:

| mark (textured) | cv2 | migan | lama |
|---|---|---|---|
| doubao | +0.06 | +1.53 | +1.38 |
| gemini | +1.79 | +2.75 | **+5.59** |
| jimeng | +1.24 | +2.69 | **+3.34** |
| jimeng_pill | +4.27 | +3.81 | **+5.16** |

LaMa > MI-GAN > cv2 holds on 3 of the 4 marks worth filling, and MI-GAN edges LaMa on
doubao. Nothing here argues for changing `--backend auto`.

**Correction, and the statistic that caused it.** An earlier version of this section
claimed the opposite -- that MI-GAN was the WORST on texture, below cv2 -- and it was
wrong. The report computed recovery as `median(filled) - median(damaged)`: a **difference
of medians**, not the median of the per-image differences. On skewed data those are
different statistics and here they disagreed in SIGN. A paired per-mark sign test settled
it: on the textured tercile cv2 vs MI-GAN is not significant for any mark (p 0.13-1.00;
pooled n=119, cv2 wins 69, p=0.099), while MI-GAN's medians are higher for 3 of 5.

The wrong statistic nearly shipped a change to `--backend auto`, which is the resolver a
memory-constrained CPU caller depends on. Two lessons: report the median of the per-image
DIFFERENCES when the question is paired, and confirm a ranking with a paired test before
acting on a table of independently-aggregated columns.

**Every backend collapses on texture**: recovery falls from ~+10-15 dB to ~+1-3 dB and SSIM
from ~0.79-0.95 to ~0.30-0.44. The documented "textured is where fills struggle" is
confirmed, with numbers, for the first time.

**The invariant held**: 0 violations of "the fill touches nothing outside its mask" across
all 720 measurements and all three backends.

#### The Jimeng pill: measure the GATED path or the number is meaningless

The parity audit calls `get_mark(key).remove`, which bypasses `_keep_pill`. On the pill
that path reports "detector still fires after removal" 68-75% of the time and reads as a
broken feature. It is not: the product gates the pill hard, and `--mark auto` behaves
completely differently.

Measured through the product path over **all 2,738 pill positives** (2026-07-20):

| | n | |
|---|---|---|
| raw detections | 2,738 | |
| corroborated real (wordmark or TC260) | 346 | 12.6% |
| the gate lets through to removal | 125 | 4.6% |
| **of those, corroborated real** | **125** | **precision 100% (95% CI 97.0-100)** |

Every single pill the product removed was corroborated. Headline recall is 36.1%, but that
denominator is wrong: **TC260 provenance maps to BOTH ByteDance products**, so a "TC260
says Jimeng" image may be a Doubao image with no pill at all. Split by evidence strength:

| corroboration | n | removed | recall |
|---|---|---|---|
| wordmark (names the product) | 49 | 49 | **100% (CI 92.7-100)** |
| TC260 only (cannot separate Doubao) | 297 | 76 | 25.6% (CI 21.0-30.8) |

So where the evidence actually names Jimeng, the pill is removed every time. The flatness
guard suppresses 186 TC260-only cases -- its measured recall cost, paid to avoid the
smeared textured fills it exists to prevent.

Two harness lessons, both of which produced a wrong number before being caught:
- **Gated marks must be measured through the gate.** A per-mark audit answers a question
  the product never asks.
- **Match mark labels exactly.** A substring test on `AI生成` also matches Doubao's label
  (`Doubao 豆包AI生成 text`), counting Doubao removals as pill removals and inflating the
  pill's precision.

#### The finding that was not being looked for: filling a faint mark is net negative

Samsung came out negative in every cell, which looked like a broken mark. It is not -- the
sign is set by how strongly the mark perturbs the image, not by which mark it is. Per-mark
recovery by damage band (a HIGH damage-PSNR means a FAINT mark):

| mark | 0-18 dB | 18-22 dB | 22-26 dB | 26+ dB |
|---|---|---|---|---|
| doubao | +9.78 (n=160) | -1.52 (36) | -3.37 (18) | -8.38 (23) |
| jimeng | +7.70 (187) | -2.49 (21) | -2.54 (15) | -2.34 (15) |
| samsung | - | **+7.23** (48) | -3.78 (102) | -10.78 (90) |

Samsung is POSITIVE where its mark is strong; doubao and jimeng go NEGATIVE where theirs
are faint. Linear fit over all 720: `recovery = -0.861 * damage_psnr + 19.47`, break-even
at **~22.6 dB**. Samsung's alpha map peaks at 0.37 against doubao 0.68 and jimeng 0.93, so
Samsung simply sits on the faint side of that line most of the time.

So: **below ~22.6 dB of mark damage, inpainting costs more fidelity than the mark did.**
The pipeline currently fills unconditionally once a mark is detected, so this cost is
invisible today.

Do not read this as "stop removing faint marks". A user who wants the watermark GONE is not
optimizing PSNR, and a faint mark is still a mark. What it says is that the fill has a real
cost, it is now measurable, and for faint marks it exceeds the thing it removes -- which
makes "how faint is too faint" a product decision that can finally be made on evidence.

### B2. Detector response curves

Stamp marks across a controlled grid -- size, contrast, background texture, aspect,
JPEG quality -- and measure detection rate per cell. Produces a recall **curve** instead of
a point estimate, and directly tests the `scale_basis` geometry that hid a 100% landscape
miss for months. Cheap, repeatable, no human in the loop.

### B3. Invisible round-trip, positive-control gated

The open DWT-DCT detector is positive-only and carrier-fragile: "not found" on a fragile
carrier proves nothing (measured: `chatgpt-1.png` recovers 114/128, below the 118 gate).
Every invisible assertion must first embed on the SAME carrier and confirm recovery, and
**degrade to a skip rather than a pass** when the control fails. Already implemented in
`smoke_matrix.py`; apply the same discipline anywhere else this detector is used.

### B4. Resource ceilings

Peak RSS and wall time per backend x input size, up to 25 MP. The memory-constrained CPU
tier is a real constraint (MI-GAN must stay ~0.6-0.9 GB by cropping around the mask); a
regression here is invisible today and would only surface under load.

## Tier C -- human-labelled accuracy (bounded by labelling effort)

The machinery exists: `visible_recall_sample.py` -> `visible_sheets.py` ->
`visible_groundtruth.py` -> `visible_eval.py`.

- **Recall** is the known weak spot: measured once, unbiased n=240, and that single
  measurement is what exposed the landscape miss. Expand per mark, especially jimeng
  (n=14) and jimeng_pill (n=6), whose numbers currently rest on almost nothing.
- **Precision** re-runs over the existing 779-cell ground truth; benchmark every detector
  change with `--vs <snapshot>`.
- **Coverage**, the largest known gap: ~6% of sampled images carry an uncovered vendor's
  mark (千问 / 百度 / 星绘 / 抖音-class) that no registered detector can fire on. This is a
  coverage problem, not a tuning problem, and no threshold work will move it.

Three harness rules are load-bearing and must not be relaxed: score a mark only within its
crop's adjudication scope; take provenance from metadata, never from labels; and never
report recall from the detector-sampled set.

## Tier D -- external oracles (manual, not automatable here)

SynthID removal cannot be verified locally by design -- no public decoder exists. Each
vendor has its own oracle and it covers only that vendor's content: `openai.com/verify` for
OpenAI (more accessible, the automation candidate), the Gemini app for Google (manual,
rate-limited). A quiet metadata proxy is **not** proof the pixel watermark is gone.

Scope honestly: this tier certifies strength floors on a handful of images per vendor, and
that is all it can do. See `docs/synthid.md`.

### D1. Sampling frame

The sidecars already classify the corpus: **15,000 images whose watermark list mentions
SynthID**, of which 9,071 carry `verify_oracle=openai` and 5,929 `verify_oracle=google`.
Stratify on the two axes that actually move removal efficacy -- vendor (the certified
floors differ: OpenAI 0.10, Gemini 0.15) and content class (photoreal vs flat graphic,
where the pipelines are documented to diverge).

### D2. The oracle is the bottleneck, not the GPU

Measured on MPS (2026-07-19), single invocation of `invisible`:

| `--max-resolution` | 256 | 384 | 512 | 768 | 1024 |
|---|---|---|---|---|---|
| wall time | 37.9 s | 58.9 s | 59.4 s | 65.2 s | 118.3 s |

384/512/768 are indistinguishable, so below ~768 the cost is dominated by **fixed model
load, not diffusion**. Confirmed by batching: 4 images in one process took 105.5 s
(26.4 s/image) against 59.4 s/image one at a time -- roughly 40 s fixed overhead per
invocation and ~15 s marginal per image at 512 (rough: run-to-run variance is large).

Two consequences for the harness:
- **Amortize the fixed cost**: one long-lived process over many images, never one
  invocation per image. That is a 2-4x win. Shrinking below 512 is not.
- The binding constraint is the **external oracle's throughput**, which is manual and rate
  limited. So do not run a uniform grid; spend each oracle check where the answer is
  uncertain -- **bisect strength per content class** to certify a floor in ~10 checks
  instead of ~100.

### D3. Mandatory control before trusting any reduced-size run

`--max-resolution` is downscale -> diffuse -> Lanczos upscale. If the **resize round trip
alone** damages SynthID, the oracle goes quiet for a reason unrelated to removal and the
result does not transfer to production at native resolution.

Before any reduced-size sweep, run the resize round trip with **no diffusion** and put the
result through the vendor oracle. If the watermark survives, the reduced size is a valid
test bed; if it does not, reduced-size results are measuring the resizer. This is the same
failure shape as the imwatermark carrier-fragility trap in B3: an oracle that falls silent
for the wrong reason reads exactly like success.

`docs/synthid.md` cites ~99.98% TPR across 30 transforms including resize, which predicts
the control passes -- but that is Google's claim about their own decoder, not our
measurement, so it is a hypothesis to test, not a reason to skip the control.

## Tier E -- robustness and adversarial inputs

Malformed and hostile inputs, since ~0.2% of real uploads are already truncated: truncation
at many offsets, corrupt headers, 16-bit and CMYK, absurd dimensions, decompression bombs,
zero-byte files, unicode and RTL filenames, symlinks, read-only output dirs, concurrent runs
on one file. The bar is never "handles it" but **never raises and never silently degrades**.

## Build order

1. **A1 sidecar regression** -- highest value per hour, unattended, needs no new labels.
2. **A2/A3/A4 parity and invariants** -- full corpus, reuses existing audit scripts.
3. **B1 fill quality** -- closes the oldest unmeasured claim in the project.
4. **B2 detector curves** -- cheap, and directly guards the geometry class of bug.
5. **A5 contract sweep at corpus scale**.
6. **B4 resource ceilings**, **E robustness**.
7. **C recall expansion** -- gated by labelling appetite.
8. **D oracles** -- manual, per release.

Every tier writes a versioned snapshot so runs are comparable over time; a run that cannot
be diffed against the last one is a one-off, not a regression suite.

## What the measurements imply for detection work

Recorded here because each item is grounded in a number from this campaign, not because
it is a prioritized plan (that lives elsewhere -- see the note at the end of this section).

### Metadata absence does not disable the detectors -- it disables the RELAXATION

The detectors are pixel-based and need no metadata. What metadata does is relax the
false-positive gate (`auto` vs `strict`). So "work better without metadata" means
strengthening the strict-path detectors themselves; it is not a gating problem.

Per mark, what actually goes away when metadata is stripped:

- **The pill loses an entire arm.** Its TC260 arm is dead without metadata, leaving only
  the wordmark arm. Measured: where the wordmark corroborates, pill recall is **100%**
  (49/49); on TC260-only evidence it is 25.6%. So on stripped uploads the pill's fate
  rests entirely on Jimeng wordmark detection -- whose own recall is **71% on n=14**.
  This is the weakest link with the most leverage: every point of wordmark recall pulls
  the pill along with it.
- **The sparkle already runs on pixels**, and the FP-gate tightening cost ~120-200 genuine
  detections corpus-wide (12.5% of the 1,256 lost). That headroom exists but the
  precision trade behind it was deliberate.
- **The largest gap is metadata-independent by nature**: ~6% of sampled images carry an
  uncovered vendor's mark (千问 / 百度 / 星绘 / 抖音-class) that no registered detector
  can fire on at all.

### Where the evidence points

1. **A generic CJK AI-mark detector.** GB 45438-2025 mandates the shared `AI生成` tail,
   and 千問 is already measured as non-separable from Doubao (AUC ~0.5) precisely because
   of it. The right shape is to detect the mark CLASS and treat vendor attribution as
   optional metadata. Closes the 6% coverage gap and is metadata-free by construction.
2. **Port the `tophat` front-end to the remaining marks.** It took Doubao from 89% to 92%
   recall at unchanged 99% precision. But the gate is front-end specific and **must be
   recalibrated, never ported**: a naive 0.40 produced 8 false fires instead of 1 and
   silently halved the pill's recall (because `_keep_pill` suppresses the pill whenever
   Doubao fires).
3. **The Jimeng wordmark.** Weak on its own (71%/71%) and it gates the pill. Its
   silhouette is also non-discriminative against Doubao's, which was patched with a 0.85
   threshold -- a patch on a detector problem, not a fix.

### Measure before improving

Jimeng recall rests on **n=14** and the pill's on **n=6**. Improving what is measured by
six samples means not knowing whether it improved. Tier B2 (detector response curves on
stamped marks) is the instrument to build first: recall as a function of size, contrast
and background texture, with no new hand labelling, and it catches the geometry class of
bug (`scale_basis`) directly.

This section records what the measurements imply technically. Prioritization is tracked
separately, outside this repo.

## Standing gap

None of this is in `maintain.sh`, and it should not all be -- the sweeps take hours. But
that means **no detector-accuracy or CLI-contract regression is caught automatically**
today. The endpoint of this plan is a cheap subset (fixtures-only smoke + a sidecar diff on
a fixed 500-image slice) that CI can run, with the full sweeps staying pre-release.
