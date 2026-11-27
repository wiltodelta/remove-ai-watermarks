# Video SynthID quality research (2026-08-05)

> Research archive. This page records experiments and decisions from the date
> above. It may mention prototypes or defaults that were later changed. Use the
> user guides and current source code for the supported interface.

Cited research behind the question **"can the video SynthID path keep removing the
mark while giving up far less quality than the shipped 512 px / 12 fps profile?"**
Produced by a 13-agent workflow: 6 parallel scouts (pipeline ablation, SynthID
internals, attack literature, autoencoder landscape, perceptual masking, experiment
design), 11 change proposals from 3 independent design angles, and 3 adversarial
critics (signal theory, repository engineering, verifiability). No experiment was
run against the provider oracle for this page; every claim is labeled MEASURED,
REPORTED, or INFERRED.

Repository claims below were re-verified against source after the workflow
returned: the PSNR reference frame, the audio stream copy, the missing `_fit_size`
clamp, the absent `enable_tiling` call, the crf asymmetry, the test pins, and the
absent `scaling_factor` key in the cached checkpoint config.

## Context

`remove_video_invisible` regenerates video pixels through `sd-vae-ft-mse`: frames
are decimated to 12 fps, resized to a 512 px long side, encoded to latents with
`latent_dist.mode()`, perturbed by one seeded spatial noise field shared across
every frame, decoded, and streamed to an H.264 encoder at crf 18. A separate
stream-copy mux re-adds source audio and strips metadata. The shipped
`noise_std=0.15` is certified by one oracle row on one carrier.

The quality complaint is real: a 1080p source is delivered at roughly a quarter of
its linear resolution and half its frame rate. This page asks what part of that
cost is buying removal and what part is buying nothing.

## What the two oracle rows do and do not prove

`data/evaluations/video-synthid-oracle.csv`, one carrier (Veo 3 off-road,
sha256 `79a552b9...`), one seed, one geometry:

| noise_std | long_side | fps | verdict | psnr_db | temporal_residual_ratio |
| --- | --- | --- | --- | --- | --- |
| 0.10 | 512 | 12 | detected | 26.2932 | 1.0072 |
| 0.15 | 512 | 12 | not_detected | 25.3911 | 1.0578 |

They prove exactly one thing: the envelope of 512 px, 12 fps, crf 18, and a full
VAE round trip does **not** silence the oracle on its own. The verdict flip is
bought by the last 0.05 of latent noise.

They cannot decompose that conjunction. No row varies `long_side` or `fps`, so the
contribution of the downscale is unmeasured. The frequent reading "the 512 px
downscale was probably doing the removal work" is not supported, and neither is its
opposite.

## Measured facts from the repository

- **The fidelity metric is blind to the expensive steps.** `_iter_sampled_frames`
  applies `cv2.resize` at [`video_invisible.py:236`](../src/remove_ai_watermarks/video_invisible.py);
  the same generator feeds the loop at `:401-407`; the accumulator at `:429-435`
  zips those already-resized frames against `regenerated` captured at `:423`, which
  is before `frame_pipe.write` at `:430`. `psnr_db` therefore excludes the
  downscale, the decimation, and the encoder. It measures the VAE round trip plus
  latent noise at the working geometry, nothing else.
- **Frame decimation cannot destroy the carrier.** `_iter_sampled_frames`
  (`:224-238`) is pure subset selection: the only statement inside the threshold
  test is `yield cv2.resize(...)`, with no arithmetic across frames. The
  perturbation is equally time-blind: `shared_noise.expand(latents.shape[0], -1, -1, -1)`
  at `:295` broadcasts one CHW field across the batch and never mixes across time.
  Any fps effect on the verdict is a detection-probability effect, not carrier
  destruction.
- **`_fit_size` has no clamp at 1.0.** `scale = long_side / max(width, height)` at
  `:88` upscales any source whose long side is below 512, and both axes are floored
  independently to a multiple of 8 at `:89-96`, which introduces an anamorphic
  shift at scale 1.0 (1366x768 becomes 1360x768).
- **`enable_tiling` is never called** anywhere in `src/`, `scripts/`, or `tests/`.
  `enable_slicing()` at `:135` already splits both encode and decode to single
  frames, so activation VRAM is set by one frame and does not scale with
  `--batch-size`.
- **crf asymmetry:** 18 on the invisible path (`:321`, `:386`) against 14 on the
  visible path ([`video_visible.py:1343`](../src/remove_ai_watermarks/video_visible.py))
  through the same encoder.
- **No HDR guard on the invisible path.** `_HDR_TRANSFERS` (`video_visible.py:97`)
  and the `component_depth > 8` rejection (`:1322`) exist only for visible removal.
- **Audio is byte-copied.** `-map 0:v:0`, `-map 1:a?`, `-c copy` at
  [`video_encoding.py:413`](../src/remove_ai_watermarks/video_encoding.py).
- **Only `noise_std` is pinned.** `tests/test_video_invisible.py:285` asserts
  `DEFAULT_VIDEO_SYNTHID_NOISE_STD == 0.15`. Neither `long_side` nor `fps` is
  pinned anywhere, so changing either breaks no test.
- **`scaling_factor` 0.18215 is a class default, not a checkpoint fact.** The
  cached `config.json` for `stabilityai/sd-vae-ft-mse` has no `scaling_factor` key
  at all (verified locally: `_class_name`, `latent_channels`, `block_out_channels`,
  `sample_size`, and block types only). The value comes from the `AutoencoderKL`
  class default under `diffusers>=0.38.0` with no upper bound (`pyproject.toml:94`,
  `uv.lock` resolves 0.39.0), while `maintain.sh` runs `uv-outdated`. A dependency
  bump can move the certified operating point with a fully green test suite.
- **The manifest schema cannot record what a real program needs:** no source
  geometry, no control row, no track column, no verbatim verdict, no indeterminate
  state.

## Measured facts from external sources

- Gemini video verification quota: 10 checks per rolling 24 hours, up to 5 minutes
  of video total, under 90 seconds and 100 MB per file. The verifier reports which
  parts of the video carry the mark, and it has a **third** state beyond detected
  and not detected: unclear, with documented causes including "not enough details
  to watermark".
- The verifier scores **audio and visual tracks separately**. Google's published
  example verdict reads as SynthID detected in the audio over a time range with no
  SynthID detected in the visuals.
- SynthID-Image (arXiv:2510.09263) is a post-hoc, model-independent pixel-space
  watermark: a separate encoder network stamps an already-decoded image rather than
  being injected into the generator's latents. Its published payload figure is 136
  bits within a 512x512 image, and its product setup runs at 1536x1536.

## Inferred, with the reasoning that makes them weak

- **The noise axis is nearly exhausted.** Fitting `MSE = A + B * noise_std^2` to the
  two measured rows gives A = 124.5 and B = 2815, so a pure round trip
  (`noise_std = 0`) lands near 27.2 dB. The entire noise budget is worth at most
  **+1.79 dB**; everything else is the autoencoder. This is a two-parameter fit to
  two points with zero degrees of freedom, and the assumptions that MSE is
  quadratic in `noise_std` and that reconstruction and noise errors are orthogonal
  are untested. One local run at `noise_std=0` replaces it with a measurement.
- **Statistical power of the current certification.** With zero failures at n = 1,
  the exact one-sided Clopper-Pearson bound `1 - 0.05^(1/n)` is 95%: the data are
  compatible with removal failing almost always. n = 15 gives 18.1%, n = 30 gives
  9.5%.
- **Size of the geometry prize.** At a 1280x720 source, `_fit_size(1280, 720, 512)`
  is `(512, 288)` (pinned in `tests/test_video_synthid_sweep.py:30`), so 6.25x of
  the pixels are discarded by the downscale and another 2x by decimation from a
  24 fps source. Ladder rungs: 768 gives 2.25x the current area, 1024 gives 4.0x.
  The carrier's own geometry is recorded nowhere, so this is conditional.
- **The direction of the resolution effect is disputed, and both sides are
  inference.** Against raising it: the absolute frequency ceiling below which an f8
  VAE reconstructs faithfully is tied to the latent pitch, so 512x288 (a 64x36
  latent) preserves roughly up to 32 cycles per frame width while 1920 (a 240x135
  latent) preserves up to about 120. Raising resolution moves the carrier band out
  of the regime the decoder synthesizes and into the regime it reproduces
  faithfully, handing the detector more evidence. For raising it: Google's product
  operating point is 1536x1536, so native is closer to the distribution the
  watermark encoder targets. Note that the second argument cuts against the
  proposal rather than for it.

## First local measurements (2026-08-05)

Run without the oracle on a locally built carrier: a 6-second 1280x720 24 fps
clip panning across `data/fixtures/provenance/doubao-1.png`, processed by the
shipped path at 512 px / 12 fps on MPS. Generated media stayed outside the
repository.

| noise_std | engine `psnr_db` | end-to-end PSNR | end-to-end SSIM | bitrate |
| --- | --- | --- | --- | --- |
| 0.00 | 27.8049 | 27.6935 | 0.6955 | 2564 kbps |
| 0.15 | 25.8853 | 25.8541 | 0.6703 | 2716 kbps |

Three readings, all MEASURED, none of them about SynthID:

1. **The two-point fit's shape survives contact with independent content.** The
   pure round trip lands at 27.80 dB against the 27.2 dB the carrier fit
   predicted, and the whole noise budget costs 1.92 dB against the predicted
   1.79 dB. These are not the carrier's numbers, but the decomposition holds:
   the autoencoder is the floor and the entire `noise_std` axis is worth about
   2 dB.
2. **On this content the downscale is nearly free in PSNR terms.** End-to-end
   PSNR sits within 0.11 dB of the in-loop number, meaning the 512 px geometry
   cost almost nothing next to the VAE damage. This clip is a pan over a smooth
   generated image with little high-frequency detail, so it is the friendly
   case: real camera texture should widen that gap. Run the probe across content
   types before trusting any estimate of the geometry prize.
3. **`temporal_residual_ratio` is not meaningful on a near-static shot.** It read
   1.82 at `noise_std=0` and 2.47 at 0.15, far outside the [1.0072, 1.0578] band
   ever observed before. A slow pan gives the source almost no motion residual,
   so the `max(temporal_baseline, 1e-6)` denominator collapses and the ratio
   inflates. This is the predicted defect, now observed rather than argued.

## Measurements on the real carrier (2026-08-06)

The public Veo 3 off-road sample was downloaded and its sha256 matched the manifest
exactly, so everything below is measured on the same file the 2026-07-31
calibration used. Its properties, none of which the manifest recorded at the time:
H.264 1920x1080 at 24 fps, 8.000 s, 5,155,063 bps video, plus an **AAC 48 kHz
stereo 192,241 bps audio track of 7.914 s**.

The shipped profile therefore delivers 512x288 at 12 fps: **7.1% of the frame area
and 3.6% of the pixels per second.**

### The geometry prize, measured end to end

Scored with `scripts/video_fidelity_probe.py` against the untouched 1920x1080
source, all at `noise_std=0.15`:

| Geometry | Frame area | PSNR | SSIM | Temporal | Bitrate |
| --- | --- | --- | --- | --- | --- |
| 512x288 @12 (shipped) | 7.1% | 25.3971 | 0.8215 | 1.0407 | 2187 kbps |
| 768x432 @12 | 16.0% | 27.3796 | 0.8556 | 1.0323 | 3782 kbps |
| 1024x576 @12 | 28.4% | 28.8542 | 0.8819 | 1.0262 | 5604 kbps |
| 512x288 @24 | 7.1% | 25.6142 | 0.8270 | 1.0995 | 3181 kbps |

**512 to 1024 is worth +3.46 dB, against 1.92 dB for the entire `noise_std` axis.**
The geometry lever is roughly twice the noise lever, which is the first hard number
this investigation has for the size of the prize. What it still costs in removal is
unknown: none of these was ever submitted.

Two readings that are not obvious from the table:

- **Temporal residual improves as resolution rises** (1.0407 to 1.0262). The shared
  noise field lives on the latent grid, so at 1920 it is four times finer relative
  to the frame than at 512, where it stretches into low-frequency blotches. Higher
  resolution plausibly makes the perturbation *less* visible, not more.
- **The frame-rate arm looks bad here and the metric is the wrong judge.** 24 fps
  buys +0.22 dB and worsens the temporal ratio to 1.0995, but what 24 fps actually
  buys is smoothness, which neither PSNR nor SSIM measures at all. The metric sees
  only the cost. Do not decide the fps question from this table.

Encoder cost on an M-series MPS device, 8 seconds of source: 51 s at 512/12, 180 s
at 768/12, 380 s at 1024/12, 190 s at 512/24. Native 1920x1080 at 24 fps
extrapolates to roughly 46 minutes for the same 8 seconds, around 350x real time. A
native run was started and killed before producing anything, so that figure remains
an extrapolation.

### The certified row does not reproduce

Re-running the certified configuration (512/12/0.15, seed 0) on this machine gave
25.5002 dB and 1.0675 against the manifest's 25.3911 and 1.0578, and a different
output sha256. The manifest records neither device nor dtype, and the original was
almost certainly CUDA fp16 while this was MPS fp32. **The oracle certified a
specific file that cannot be rebuilt**, so any re-run produces an artifact the
verifier never saw. Device and dtype belong in the schema.

### Audio passes through untouched, and the output still reads clean

Established on two carriers, by comparing the extracted audio bitstream sha256 of
input and output:

- Veo carrier: `674063df...` in, `674063df...` out.
- A Lyria 3 track generated for this purpose: `6ee4c041...` in, `6ee4c041...` out.

The Lyria case is the sharper one because its provenance is known. Google's own
generator stamps it: `get_ai_metadata` reads a C2PA manifest naming
`Google C2PA Core Generator Library`, issuer `Google LLC`, `trainedAlgorithmicMedia`.
After the shipped path runs, that output reports `{}` -- every metadata marker
stripped, `remaining_metadata` empty, visuals regenerated -- while the audio the
generator produced is bit-for-bit intact.

So the pipeline can produce a file that **our own detector calls clean while an
untouched Google-generated audio track sits inside it**. The mechanism is proven;
whether that audio carries an audible-to-the-verifier SynthID mark is not, and only
the oracle can say. If it does, this is a shipped defect rather than a quality
question, and it outranks everything else in this document.

Worth recording alongside it: the Veo carrier itself returns `{}` from
`get_ai_metadata`. The public sample carries no C2PA at all, which is exactly why it
needed an oracle verdict in the first place.

## State when this work was parked (2026-08-06)

Landed in `main` through PR #78: the manifest-backed default pin, the
scaling-factor gate and its single seam, the sweep loading through the engine's
loader, `scripts/video_fidelity_probe.py` with its tests, the extended manifest
schema, and this document. **The shipped operating point did not move**: it is
still 512 px / 12 fps / `noise_std=0.15`.

A product requirement arrived late and reframes the target: output resolution,
frame rate and bitrate must not be reduced. Resolution, frame rate and crf are
parameters and can be preserved, but zero quality loss is not achievable by this
mechanism at all -- removal works by rebuilding pixels, so a regeneration can never
return its input. The honest goal is to preserve container properties and make the
perceptual loss imperceptible, judged by DISTS and VMAF rather than PSNR. Two
consequences follow: crf 18 on this path against crf 14 on the visible path is an
unexplained asymmetry and a free quality win, and the oracle's 100 MB limit means a
higher-quality profile is *harder* to certify than the one it replaces.

Nothing is blocked on analysis. Six oracle submissions were prepared and never run,
because file upload to the verifier failed at the tool level:

1. A Lyria carrier and the pipeline's output from it, answering the audio question.
2. A multiplexed control-plus-candidate file at the shipped geometry, validating
   whether the verifier reads a concatenated pair by time range. Verified locally
   frame by frame: half one is bit-identical to the control, half two to the
   candidate.
3. Three more multiplexed pairs for 768, 1024 and 24 fps, worth three queries
   instead of six if the instrument validates.

Every generated artifact for those submissions has since been lost with its
scratch directory, which is the intended lifecycle for generated media. Rebuilding
costs one download and roughly 25 minutes of encoding; the Lyria track is the
exception, since regenerating it would produce a different track, and the copies
that survive are the only ones.

## The single most important unknown

**How much the 512 px downscale contributes to removal.** Every quality gain routes
through this question, and the critics moved it from "probably free" to "direction
unknown", which is exactly what makes it the highest-information experiment
available.

## Ranked experiment program

Ranked by information per oracle query. The budget is roughly 10 checks per 24
hours (MEASURED), so the program is paced by calendar, not by money.

### E0. The zero-oracle tier (0 submissions) - do this first

Not an experiment on the oracle; the precondition that makes everything else
interpretable.

- `ffprobe` the carrier and record `source_width`, `source_height`, `source_fps`.
  The actual downscale factor of the existing rows is currently unrecoverable.
- Extend the manifest schema: variant/control, source geometry, `vae`, `track`
  (audio|visual), verdict state in {DETECTED, NOT_DETECTED, INDETERMINATE,
  REFUSED}, verbatim verdict text, detected time range for the output,
  `session_id`, content stratum.
- **Measure the `noise_std = 0.0` round trip locally** on the same carrier. This
  converts the inferred 27.2 dB ceiling into a measurement and bounds the whole
  noise axis for one GPU pass and zero oracle cost.
- Freeze the current defaults' metrics on a fixed clip set, matching the
  record-then-diff rule in `.claude/rules/development.md`.
- Determine whether the carrier has an audio track and whether it carries SynthID.

### E1. Instrument validation: multiplexing and a session anchor (1-2 submissions)

The verifier reports time ranges, so one file can carry several doses. A 24-second
file of `[control 8 s | certified 0.15 8 s | control 8 s]` should read detected on
the outer segments and not detected on the middle one, both halves already known
from the manifest.

`encode_video_frames` requires matching dimensions and one frame rate, so
multiplexing works **only along the `noise_std` axis**, not across geometries: one
multiplexed file per geometric envelope.

Separately, submitting the existing detected file first in each session costs 10%
of the quota, turns "the oracle may have changed" into a per-session gate, and
measures the flip rate on a byte-identical file, a quantity every plan silently
assumes is zero.

### E2. Resolution ladder at fixed dose - the decisive axis (2 submissions per rung)

A fixed value, not a ceiling. Rung 1 is `long_side = 768` with fps held at 12 so
exactly one destruction axis moves. Jumping straight to 1920 is a bad first rung:
a 3.75x jump makes a detected verdict uninformative about where the boundary lies.

- **Matched control:** 768 / 12 / `noise_std = 0.10`, expected DETECTED. It is
  strictly stronger than a plain re-encode control because it discharges the
  envelope, the VAE round trip, and a nonzero dose at once, and it diffs directly
  against the existing 512 row.
- **Candidate:** 768 / 12 / 0.15.
- **Positive:** the resolution axis is open; next rungs 1024 and then native with a
  clamp, 2 submissions each.
- **Negative:** the downscale contributes to removal. This is the most valuable
  available negative: it kills the native-resolution proposals outright and turns
  the question into how much dose must be spent to buy resolution back, at the
  known exchange rate B = 2815.
- Watch for a monotonicity violation: 768 / 0.10 reading NOT_DETECTED would
  overturn the critics' spectral argument.

### E3. Frame rate at fixed geometry and dose (2 submissions)

Control 512 / 24 / 0.10 expected DETECTED; candidate 512 / 24 / 0.15 expected
not detected. The mechanism is provable from source, so the prior is high, but the
only counter-mechanism - per-frame count aggregation - is probabilistic and n = 1
on an 8-second clip does not measure it. Certify on the longest clip that fits the
90 s / 100 MB limits, because the risk compounds with frame count.

Ranked below E2 because the outcome is nearly predetermined and the prize (judder)
is smaller than the geometry prize. It is the safest bet if a guaranteed win is
wanted.

### E4. `noise_std = 0.0` as an oracle probe (1 submission) - deliberately low rank

The local half of this probe is the cheapest high-information measurement in the
program and already sits in E0 at zero oracle cost. The **oracle** half ranks low:
it buys no quality by construction, and its likely DETECTED outcome is nearly
deducible from the 0.10 row under monotonicity. Its one real value is that
NOT_DETECTED here would be non-monotone against 0.10 and would refute the
assumption underneath every ladder and bisection in this program. Worth one query,
after E2 and E3.

### E5. Stratified certification of the surviving operating point (15-30 submissions)

Only after E2 and E3 produce a winner. Zero detections across at least 15
stratified carriers (at least 3 per stratum: face closeup, fast motion, flat sky,
dark night, text overlay) bounds the failure rate at 18.1%; 30 carriers bound it at
9.5%. Add the project's 1.5x margin convention, at least 2 seeds, at least 2
sessions, and per-track verdict recording.

For the flat-sky stratum, check the source reads detected first: Google documents a
"not enough details to watermark" state, so a clean reading there may say nothing
about removal.

## Surviving change candidates

**S1. Fix the metric's reference frame (0 oracle submissions).** Add
`source_psnr_db` (candidate upscaled back to native against the untouched source
frame) **and** a separate post-mux pass that decodes the delivered file. The
critics killed the original claim that this would expose crf 18: the candidate is
captured at `:423`, before `frame_pipe.write` at `:430`, so no in-loop metric can
see the codec. Only the post-mux pass covers resize, decimation, crf, and mux
together. Use `INTER_AREA` or a fixed analytic size for the upscale so the CPU
resize does not dominate the loop, and update both other consumers of the
generator in the same change (`read_sampled_frames` at `:188-209`, and
`scripts/video_synthid_sweep.py:147` where `np.stack(frames)` must keep receiving
resized frames). Do not redefine the existing `psnr_db`: the two manifest rows stay
comparable to each other only while that field keeps its exact current meaning.

**S2. Frame rate as a fixed certified value (after E3).**
`DEFAULT_VIDEO_SYNTHID_FPS` 12.0 to 24.0, keeping `min(fps, source_fps)`. A
ceiling of 60.0 was **rejected**: it makes the delivered operating point a function
of the user's source, so one sample from a family gets certified while the rest
ship uncertified, and the risk direction is unfavorable. Port the VFR/PTS bridge
(`probe_video_timestamps`, `timestamped_input`, currently only in
`video_visible.py`): at native frame rate the output stops reading as a proxy, and
silent CFR-ification becomes a master-quality defect. Add `fps` and `long_side`
pins next to `tests/test_video_invisible.py:285`, tied to the certifying manifest
row. *(Update `0.76.0`: `DEFAULT_VIDEO_SYNTHID_FPS` moved to `None` to preserve
native source frame rate after certifying the `512p`-guided native `1280x720@24fps`
operating point on `google-gemini-omni-flash-api` and `higgsfield-veo-3-1-lite` in
`data/evaluations/video-synthid-oracle.csv`, pinned by
`test_shipped_defaults_match_a_certified_manifest_row`.)*

**S3. Clamp and align `_fit_size` (0 oracle submissions, with a caveat).**
`scale = min(1.0, long_side / max(width, height))` at `:88`; align the long side
and derive the short side from the true aspect, rounding to the nearest multiple of
8. The pinned `_fit_size(1280, 720, 512) == (512, 288)` survives. The caveat: the
clamp is an **uncertified operator change for a whole class of users**, since a
320x240 source is upscaled to 512x384 today and would run at 320x240 afterwards,
and neither has been tested. Ship it as its own documented change, not as a
drive-by fix.

**S4. Extend the manifest schema and oracle protocol (0 submissions).** A
precondition for the whole program. Record the verbatim verdict text, because an
unclear state logged as not detected is exactly the silent regression the protocol
exists to prevent.

**S5. A sigma normalization contract (0 submissions, applies to the current path).**
No VAE swap survived, but two elements are real risks today: `vae.config.scaling_factor`
cannot be trusted, so log `latents.std()` at `:276` and assert the effective
scaling factor at load; and any cross-model sigma transfer must match on the **RMS
of the decoded pixel perturbation**, not on latent units.

**S6. Additive texture masking - contingency on an E2 negative only.** Not energy
preserving: the certified sigma stays as a floor everywhere and the mask only adds
on top in textured regions. The original energy-preserving form was refuted - the
mechanism normalizes and then clips, which breaks the claimed invariant by a
content-dependent amount, and the proposed unit test asserted the invariant before
the clip and would have stayed green. Energy preservation is also unsafe in
principle: the detector does not need a full frame, and the verifier reports
per-segment, so a flat region perturbed only at the floor is a crop carrying nearly
the full carrier. The salvaged version has **no standalone PSNR win**; its only
value is freeing distortion budget to spend on resolution if E2 shows the dose must
rise. It additionally needs motion compensation for the mask (flow is computed at
`:442`, after the decode at `:423`, so the loop must be restructured) and one
deliberately bad candidate so `temporal_residual_ratio` acquires a known failing
value: it has only ever been observed in [1.0072, 1.0578], so "the ratio looks
fine" is currently an unfalsified claim.

## Refuted proposals - do not resurrect

- **Carrier-scale, resolution-invariant noise field.** "Carrier scale" is set by
  our VAE's latent pitch, not the carrier's. The motivating arithmetic treats a
  field on the latent grid as white at pixel resolution and is wrong by exactly the
  square of the scale factor: white noise on a 240-wide latent already spans
  0-120 cycles per frame, entirely inside the band the decoder reproduces. The
  construction actually band-limits the perturbation from 0-120 to 0-32 cycles,
  making it smoother - the first thing any spread-spectrum extractor's content
  suppression removes - while moving its energy toward the peak of the contrast
  sensitivity function, making it more visible. Bilinear upsampling is also
  heteroscedastic, and global renormalization by `noise.std()` stamps a visible
  amplitude lattice rather than fixing it.
- **Native resolution plus tiling plus a carrier-scale field.** Inherits the
  arithmetic above, concedes that native processing destroys less carrier without
  proposing a replacement mechanism, and bundles four destruction axes into one
  candidate on a 4-query budget so a detected verdict is undecomposable. Its cost
  model was also wrong by roughly 10x. The salvageable parts moved into S1 and S3.
- **`AutoencoderKLTemporalDecoder`.** Refuted independently by three critics.
  Removal is a property of the encode-decode composition, so a strictly more
  faithful decoder preserves more carrier at the same sigma. The class implements
  neither slicing nor tiling, so `enable_slicing()` at `:135` raises
  `NotImplementedError` and the documented bounded-memory property dies. Its
  temporal receptive field needs windows of tens of frames, its license carries a
  revenue restriction incompatible with an Apache-2.0 default, and its own
  published table shows a **worse** FID (9.17 against 7.61).
- **Wan or another temporal video VAE.** Hard refusal in code: `AutoencoderKLWan._encode`
  uses `iter_ = 1 + (num_frame - 1) // 4`, so at the shipped `batch_size = 4` only
  frame 0 is encoded and a 4-frame batch decodes back to 1, which makes
  `zip(..., strict=True)` at `:429` raise. The proposed remedy of carrying the
  causal feature cache across chunks is impossible through the public API, since
  `clear_cache()` runs on both entry and exit of `_encode` and `_decode`.
  `float(vae.config.scaling_factor)` at `:270` and `:292` also raises, because Wan
  exposes `latents_mean`/`latents_std` instead. Logically, feeding 4:1 temporal
  compression frames 83 ms apart either reconstructs well (carrier preserved, no
  removal gain) or hallucinates (no quality prize); the two claims are mutually
  exclusive.
- **A 16-channel f8 VAE.** Self-refuted and confirmed by the critics. Quadrupling
  latent channels roughly doubles L in Zhao's theorem, so sigma must roughly double
  to hold removal, and the VAE frontier in the UnMarker results is monotone with no
  published point where a more faithful autoencoder removes **more**. Predicted net
  loss of 2-7.5 dB by its own arithmetic. Its config also carries
  `scaling_factor = 0.2614`, so an unchanged `noise_std = 0.15` is about 30% weaker
  in raw latent units. Only the normalization contract survived, as S5.
- **A per-channel luma/chroma probe.** The arms match in absolute latent units but
  not in relative dose per channel, since `scaling_factor` normalizes the aggregate
  latent, so the very reading the probe exists for is confounded with dose. The
  chroma arm is additionally subsampled by the yuv420p encode, which no current
  metric sees. Maximum prize under 1 dB by its own fit.
- **Post-processing with unsharp plus grain, as a shipped stage.** The
  carrier-reimport argument is sound and was verified line by line: `unsharp_mask`
  reads only its own argument ([`humanizer.py:79`](../src/remove_ai_watermarks/humanizer.py)),
  and `adaptive_polish` touches the source only through one float. But reimport was
  never the binding risk:
  `cv2.addWeighted(img_f, 1.0 + amount, blurred, -amount, 0.0)` at
  `_ADAPTIVE_MAX_UNSHARP = 1.0` multiplies the **surviving carrier residue** by up
  to 2x as the terminal operation before encoding, and whether that re-arms a given
  file depends on that file's unobservable margin. With no local decoder and only a
  sampled oracle the stage is structurally uncertifiable, not merely expensive to
  certify. The only rescue is moving sharpening **above** the VAE stage so the
  removal operator stays terminal. Grain, which monotonically lowers detector SNR,
  is the safe half and can be separated.

## Measuring quality properly

The current `psnr_db` cannot show the improvement this work exists to produce.
Minimum upgrade:

1. **`source_psnr_db`** - candidate upscaled back to native against the untouched
   source frame. Good for ranking configurations against a common reference,
   dominated by unrecoverable high-frequency content, so not a measure of what the
   VAE costs.
2. **A post-mux end-to-end pass** - decode the delivered file after
   `mux_encoded_video` (`:456`) and compare against the source. The only measure
   covering resize, decimation, crf, and mux together. Keep it out of the streaming
   loop so `test_stream_batches_consumes_only_one_batch_ahead` stays valid.
   Implemented as `scripts/video_fidelity_probe.py`, which streams, reports the
   delivered file's bitrate, and shares the engine's frame-selection rule rather
   than copying it.
3. **DISTS** (arXiv:2004.07728), built to tolerate texture resampling - exactly
   what the VAE does to foliage, skin, and grass, and exactly what PSNR and LPIPS
   punish even when the result is perceptually equivalent. It separates "the VAE
   resampled the grass" from "the VAE destroyed an edge".
4. **VMAF** through `ffmpeg` `libvmaf`, cheapest to add, whose ADM/DLM feature
   names this exact complaint. Its temporal term is only a mean absolute luma
   difference between neighboring frames, so it is not a flicker detector: keep the
   motion-compensated ratio.
5. **Encoded file size as a third axis.** Every current metric is taken before the
   pipe, so a bitrate explosion currently reads as "quality did not suffer". This
   matters most for any perturbation that varies frame to frame: at fixed crf it
   raises bitrate rather than lowering quality.
6. **`temporal_residual_ratio` repairs**: add a p95 across frames next to the mean
   so localized flicker stops being averaged away, add a long-horizon term against
   the first frame, and replace `max(temporal_baseline, 1e-6)` with an explicit
   undefined result on static shots.
7. **A no-reference metric** (DOVER or FAST-VQA) for the case where no
   full-reference measure can compare a 512/12 output against a 1024/24 output on
   one scale. Relative ranking within a sweep only; absolute values are
   uncalibrated for this artifact class.

Nothing local measures removal. `remove_video_invisible` checks `get_ai_metadata`
on the output, which is metadata, not pixels. Only a manifest row counts.

## Risks

**Error asymmetry is the governing constraint.** Shipping a leak means a user
receives a watermarked file believing it is clean, with no local pixel decoder to
catch it and no feedback path that would surface it. Staying conservative means a
user receives 512 px / 12 fps, a cost that is visible, bounded, and reversible with
an explicit flag. A symmetric test is therefore inadmissible, and the burden of
proof sits entirely on the new default.

**n = 1.** The current default is certified by one success on one carrier with one
seed. Any quality change inherits that weakness and must not deepen it.

**Content and seed dependence** is MEASURED on this project's image branch:
survivors switch by content type, and near the threshold the same input flipped
between runs on seed alone.

**Oracle instability.** Three verdict states, not two. Per-segment reporting.
Separate audio and visual tracks - and the pipeline byte-copies audio, so a Veo 3
clip with generated sound leaves with its audio SynthID intact, and the manifest
cannot even record which track the 2026-07-31 negative referred to. That is
potentially a shipped product hole, not only an experimental confound. Session
drift is currently unfalsifiable.

**Observability narrows exactly where quality rises.** At fixed crf the bitrate
grows with pixels per second, so the maximum duration fitting under 100 MB falls as
fps and resolution rise. The configuration users actually receive becomes **less**
observable than the one it replaces. That is a permanent property, not an
inconvenience.

**Ceilings instead of fixed values.** Any ceiling makes the delivered operating
point a function of the user's source, and with it the perturbation's cycles per
frame - the quantity carrying the certified margin. Ship fixed values.

**Documentation and test surface.** Only `DEFAULT_VIDEO_SYNTHID_NOISE_STD` is
pinned. Moving `long_side` or `fps` touches hardcoded numbers in `README.md`,
`docs/known-limitations.md`, `docs/cli.md`, `docs/python-api.md`,
`docs/module-internals.md`, `docs/synthid.md`, and `docs/verification-plan.md`.
Drift-prone operational numbers should not live in seven places of prose; their
source of truth is the manifest.

**Silent configuration drift.** `sd-vae-ft-mse/config.json` carries no
`scaling_factor`, so 0.18215 is a class default under an upper-unbounded
`diffusers>=0.38.0` while `maintain.sh` runs `uv-outdated`. A library bump can move
the certified operating point with a green suite.

**Missing guards that get worse at native resolution.** There is no HDR or >8-bit
rejection and no VFR/PTS bridge on the invisible path. A 512 px output plainly
reads as a proxy; a native-resolution output reads as a master, and the cost of
silently flattening a 10-bit PQ source to 8-bit SDR rises accordingly.

**Do not bundle axes.** Raising `long_side`, raising `fps`, and lowering crf all
reduce total destruction and all require recertification. The metric reference-frame
fix is the only exception, because it changes the measurement rather than the
pixels. Do not put a crf 18 to 14 change in the same candidate as a resolution
change.
