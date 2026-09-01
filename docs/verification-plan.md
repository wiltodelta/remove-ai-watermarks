# Full verification plan

> Verification log and historical plan. Dated results describe the code and
> datasets at the time of each run. They are evidence for maintainers, not a
> current CLI contract. Recheck open items against the current code and issue
> tracker before treating them as work that remains.

How we convince ourselves the library actually works, across its whole surface, on real data.

This is the pre-release and periodic-audit plan. It is deliberately organized by **oracle
strength** rather than by module, because the hard part is never "call the function" -- it
is "know what the right answer was". A sweep with no oracle proves only that nothing threw.

Performance depends on format, enabled extras, and hardware. Measure it locally with representative inputs; do not publish private dataset sizes or run statistics.

## Public test sources

| source | role |
|---|---|
| `data/synthid/` | labeled SynthID oracle fixtures and evaluation manifest |
| `data/fixtures/provenance/` | deterministic format and provenance fixtures |
| `data/calibration/<vendor>/` | minimal controlled inputs used to derive detection silhouettes |
| synthesized inputs | constructed ground truth for pure and format-level checks |

## Tier A -- self-evident oracles (full corpus, unattended)

Properties that are true or false without anyone labeling anything. These are the
backbone: they scale to large local datasets and catch regressions with zero human cost.

### A1. Recorded-verdict regression

A dated local cache of `identify` verdicts can be replayed against the same
inputs to detect drift in verdict, platform, confidence, or signal set.

Caveat that makes this honest: a diff is not automatically a bug -- the sidecars were
written by older versions, so intended improvements also show up. The output is therefore a
**classified diff** (new detections / lost detections / changed platform / changed
confidence), reviewed once, then re-baselined. Lost detections are the alarm.

Implemented as `scripts/sidecar_regression.py` (resumable, ~1.5 h at 8 workers).

A 2026-08-15 metadata-only C2PA regression audit over the local historical
corpus caught two structured-parser gaps before release: an exact Firefly claim
generator without a repeated source type, and a Dreamina generator carried by a
reachable ingredient under a generic update manifest. Both now have focused
tests. The same audit confirmed that invalid hash/signature claims lose origin
attribution without losing the C2PA inventory, and exposed the absence of
production trust anchors in the SDK defaults. Dataset-derived counts and
identifiers remain in the gitignored audit output.

#### Local run protocol

Re-run `identify` against locally recorded sidecars, classify losses separately from intended new detections, and keep generated reports under `.local-eval/`. Do not commit dataset-derived counts or identifiers.

### A2. Parity: whatever we detect, we must be able to remove

For every image where a signal fires: remove, re-scan with the same oracle, assert quiet.
- metadata: `scripts/metadata_removal_audit.py`;
- visible: `scripts/visible_removal_audit.py`, once per backend.

Detection does not depend on the fill backend. Run `scripts/visible_positives.py` once
and pass its output through the audit's `--paths-file` option.

#### Local parity protocol

Run detection, removal, and re-detection over a representative local set. Confirm that decoded pixels remain unchanged for metadata-only operations. Keep reports untracked.

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

`scripts/smoke_matrix.py` covers every choice-valued flag on fixtures. Its knob rows
were rewritten when the CPU/MPS profiles and the ESRGAN chain were removed: most now
assert a knob is REJECTED, and the accepted-knob rows skip without a CUDA device, so
the row count is host-dependent rather than the fixed 68 recorded here before. Extend from fixtures to a stratified corpus slice
(~500 images spanning format x provenance x aspect ratio), asserting exit-code semantics
rather than just absence of crash.

Known trap to encode: **exit 2 is triply overloaded** (no-visible-mark, no-invisible-signal,
Click usage error). A wrapper cannot distinguish them without parsing stderr. Either the
sweep asserts on stderr, or the codes get split -- the latter is the better fix.

#### Coverage method

Compare the flags and values exercised by the matrix against the options declared by
the CLI. Include optional backends, batch modes, tiling, and the ffmpeg audio/video
strip. (Region-targeted diffusion composition was on this list until the parameter
was deleted: nothing but a caller-less convenience wrapper ever reached it, and the
user-facing region path is `erase`, which inpaints instead.) The gap to find is not only
"logic untested" but
"never executed on real data", which is precisely what this campaign is for.

#### Bug found by the extension: `--steps` below ~7 crashed inside torch

**Fixed by deletion.** `--steps` no longer exists, on the CLI or in the Python API,
so this class of failure is unreachable. Kept as a record of why.

Effective timesteps were `int(steps * strength)`. At the vendor-adaptive default
strength (0.15, or 0.10 for OpenAI) any `--steps` under 7 rounded to **zero**, and the
pipeline died with a raw traceback:

```
RuntimeError: cannot reshape tensor of 0 elements into shape [0, -1, 1, 512]
```

Fully valid CLI arguments, no special flags, no `--force`. The value was accepted, the
crash was a torch internal, and nothing told the user that steps and strength interact.
The considered fixes were a clamp to >=1 effective step or an up-front validation
naming both values; what shipped instead is that each stage owns its own distilled
schedule and no caller can set it. The general lesson stands: a knob whose valid range
depends on another knob's value needs the interaction validated where both are known,
or it needs to not be a knob.

Method note: the first run of the knob rows failed 12 times with this identical error,
which read like twelve broken features. It was one bad harness parameter (`--steps 4`)
sitting on top of one real bug. An error that is IDENTICAL across unrelated rows is
evidence of a common cause, not of many faults -- check the shared input first.

## Tier B -- constructed ground truth (automatable, no labeling)

Where reality gives no answer key, build one. This is the tier that closes the two biggest
holes: fill quality and detector response at the edge of the operating range.

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

#### Local fill-quality protocol

Construct marked images from clean local references, compare each backend against the known original inside the affected footprint, and keep the report under `.local-eval/`.

### B2. Detector response curves

Use `scripts/detector_response.py` to sweep mark size, opacity, background, and aspect. Report detection and maskability separately. Generated reports stay under `.local-eval/`.

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

## Tier C -- human-labeled accuracy (bounded by labeling effort)

The machinery exists: `visible_recall_sample.py` -> `visible_sheets.py` ->
`visible_groundtruth.py` -> `visible_eval.py`.

- **Recall** needs representative sampling per mark and aspect ratio.
- **Precision** must be benchmarked before and after every detector change with
  `--vs <snapshot>`.
- **Coverage** is separate from tuning. A missing vendor detector cannot be repaired by
  changing another detector's threshold.

Three harness rules are load-bearing and must not be relaxed: score a mark only within its
crop's adjudication scope; take provenance from metadata, never from labels; and never
report recall from the detector-sampled set.

## Tier D -- external oracles

Proprietary watermark removal cannot be verified locally by design -- no public decoder
exists. Each vendor has its own oracle and it covers only that vendor's content:
`openai.com/verify` for OpenAI, the Gemini app for Google, and Microsoft's
[Content Provenance Detection API](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/how-to/how-to-provenance-detection)
for InvisMark. The Microsoft API reports `Watermark` and `C2PA` separately. It therefore
needs a pixel-identical metadata-stripped control: the control must lose `C2PA` while
remaining `Watermark`-positive before a candidate's negative result can be attributed to
pixel regeneration. A quiet metadata proxy is **not** proof the pixel watermark is gone.

OpenAI's API documentation says not to use repeated queries to reverse-engineer, remove,
or evade a watermark. Using it as an adaptive research oracle therefore requires explicit
authorization. Without that authorization it must not become a training loss, search loop,
or automated removal gate. The provider-specific detector and pixel-only removal research
protocol is in [`synthid-detector-removal-plan.md`](synthid-detector-removal-plan.md).


Scope honestly: this tier certifies strength floors on a handful of images per vendor, and
that is all it can do. See `docs/synthid.md`.

### D1. Sampling frame

Select a representative local sample without committing images, identifiers, dataset sizes, or oracle results. Stratify by signal family and preserve the sampling method outside the public repository.

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

### D4. Video candidates require a matched transcode control

Video experiments add frame sampling, resizing, frame-rate conversion, and a
final video codec around the actual attack. `scripts/video_synthid_sweep.py`
therefore emits `control.mp4` from the same selected frames and encoder settings
as every VAE candidate.

Verify the control first in a new Gemini chat by invoking `@synthid` and using
the supported built-in content-verification question. Continue only when the
provider oracle still detects SynthID in it. Verify each candidate in its own
new chat. A generic response that discusses visual clues or metadata is not an
oracle result. Do not ask the chat model to reinterpret or second-guess the
built-in verdict; that follow-up is ordinary model reasoning. Record only the
explicit built-in SynthID verification verdict in the generated CSV.

The harness shares one latent-noise field across the sequence to avoid adding
independent frame noise. Its temporal-residual metric is a fidelity check, not a
watermark detector.

The 2026-07-29 two-carrier calibration produced genuine built-in verifier
results: both matched controls were positive, the stronger candidate
was negative on both carriers, and a weaker candidate was negative on one. On
2026-07-30, adversarial follow-up prompts returned `UNAVAILABLE` after asking
ordinary Gemini to ignore and reinterpret the detector result. Those follow-ups
were mistakenly treated as a stricter oracle; they were not detector reruns.
The implementation remains exposed as `video invisible` and
`remove_video_invisible`. Its oracle-certified default is the product operating
point. A fresh control-positive, candidate-negative pair remains an optional
per-file audit after provider changes or for unusually important files.

The 2026-07-31 full-clip check added a public eight-second Veo carrier. The
source and the complete `0.10` product output were both detected, proving the
surrounding resize / frame-rate / codec path had not silenced the oracle. The
complete `0.15` product output was not detected, so `0.15` became the default.
The tracked source and output hashes, fidelity metrics, and verdicts are in
`data/evaluations/video-synthid-oracle.csv`; generated media remains untracked.

## Tier E -- robustness and adversarial inputs

Malformed and hostile inputs, including truncated files:
at many offsets, corrupt headers, 16-bit and CMYK, absurd dimensions, decompression bombs,
zero-byte files, unicode and RTL filenames, symlinks, read-only output dirs, concurrent runs
on one file. The bar is never "handles it" but **never raises and never silently degrades**.

## Build order

1. **A1 sidecar regression** -- highest value per hour, unattended, needs no new labels.
2. **A2/A3/A4 parity and invariants** -- representative local set, reusing existing audit scripts.
3. **B1 fill quality** -- closes the oldest unmeasured claim in the project.
4. **B2 detector curves** -- cheap, and directly guards the geometry class of bug.
5. **A5 contract sweep over a representative local set**.
6. **B4 resource ceilings**, **E robustness**.
7. **C recall expansion** -- gated by labeling appetite.
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

- **The pill loses its metadata-confirmed arm.** Without metadata it depends on the
  bottom-right Jimeng wordmark.
- **The sparkle already runs on pixels.** Its threshold remains a deliberate
  recall-versus-precision trade.
- **Uncovered vendors are metadata-independent gaps.** They require a detector of their
  own.

### Where the evidence points

1. Prefer per-vendor CJK templates when fitted geometry separates similar marks.
2. Recalibrate every detector when changing its front-end. Thresholds do not transfer
   between `binary`, `tophat`, and `gray`.
3. Treat the Jimeng wordmark as a load-bearing confirmation path for the pill.

### Measure before improving

Use Tier B2 detector-response curves before tuning a detector with sparse labeled
examples. Sweep size, contrast, aspect, and background texture so geometry regressions
are visible without exposing private evaluation statistics.

This section records what the measurements imply technically. Prioritization is tracked
separately, outside this repo.

## Local end-to-end verification

Run `scripts/real_examples_e2e.py` against representative local inputs before releases that affect image handling. The script must read from `.local-eval/`, write only untracked temporary output, and report behavior without exposing dataset provenance or aggregate private measurements.

The cheap video seam is covered automatically by
`TestVideoVisibleFullClip::test_removes_complete_clip_and_preserves_sequence_and_audio`.
It constructs a full synthetic Sora-like MP4 with AAC audio and C2PA
provenance, then exercises detection, temporal arbitration, OpenCV fill, real
ffmpeg encoding, metadata stripping, audio stream copy, and atomic publication
through the public API. A separately encoded clean control supplies the paired
frame-to-frame deltas inside the filled region, so the gate detects temporal
flicker rather than treating all output motion as an error. The default
motion-compensated fill must score strictly below a separately encoded
frame-local opt-out on the median paired error, while its high-percentile error
cannot regress. The dedicated Linux CI job installs ffmpeg explicitly. Keep real-provider and learned-backend
sequence evaluation local because those inputs or model downloads do not
belong in the core matrix.

The local real-provider audit runs one complete clip for every registered video
mark. OpenCV and MI-GAN must remove every accepted frame, leave the second-pass
detector quiet, preserve source stream starts and duration, and copy AAC
packets exactly when present. Run one complete LaMa clip to verify wiring and
resource tier; its CPU throughput makes a six-provider online matrix
counterproductive. Store generated outputs and the detailed CSV only under
`.local-eval/`.

The same full-clip gate runs the public metadata-only path against its real MP4
and verifies unchanged file size, decoded frames, stream properties, and AAC
packets. A separate synthetic large-`mdat` test rejects full-source
`read_bytes()`, hashes the copied media payload, and mutation-checks both C2PA
and TC260 survival.

A companion full-clip VFR case alternates three frame durations, runs the
public visible-removal API, and compares every output display timestamp to the
source within one source time-base tick. Its source starts at a non-zero PTS,
so the test also verifies retained video/audio stream offsets, container
duration, and copied AAC identity. Mutations that disable the timestamped NUT
bridge or reset its start PTS must fail this gate.
A separate constant-rate clip with the same non-zero start guards the
start-offset routing without relying on the VFR branch.

## Standing gap

None of this is in `maintain.sh`, and it should not all be -- the sweeps take hours. But
that means **no detector-accuracy or CLI-contract regression is caught automatically**
today. The endpoint of this plan is a cheap subset (fixtures-only smoke + a sidecar diff on
a fixed 500-image slice) that CI can run, with the full sweeps staying pre-release.
