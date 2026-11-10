# Seedance video study

`scripts/seedance_video_study.py` is a local research runner for the first
Seedance removal experiments. It does not upload media, provide a Seedance
decoder, or add a supported removal profile to the public CLI. Its candidates
are transformations to evaluate, not verified watermark removal.

## Run a frozen comparison

Run from the repository root, using a new directory outside the repository:

```bash
uv run python scripts/seedance_video_study.py run \
  data/captures/2026-09/higgsfield/seedance-2-5.mp4 \
  --origin 'Seedance 2.5 / Higgsfield / repository capture 2026-09' \
  --output-dir /path/outside/repository/seedance-run \
  --long-side 512 --fps 12 --device mps
```

Use the diffusion extra if Torch and Diffusers are not installed. The default
VAE loader can download its public weights on the first run. `--skip-vae` runs
only the metadata, re-encode, and temporal arms without loading weights.
Omitting `--fps` keeps the source's nominal frame rate. Repeat `--long-side` for
additional geometries and `--noise` for additional VAE perturbation levels.
Repeat `--codec-crf` to add codec-only candidates, for example
`--codec-crf 28 --codec-crf 32`; the matched re-encode remains CRF18.
Keep the matrix fixed before looking at external detector responses.

### Fixed efficacy-first comparison

Use an owned source with its visible AI label already removed or identically
masked. This runner does not locate or remove visible labels. Keep that prepared
source for both controls and candidates.

```bash
uv run python scripts/seedance_video_study.py run /path/to/owned-label-cleaned.mp4 \
  --origin 'Owned Seedance 2.5 generation, visible AI label masked' \
  --output-dir /path/outside/repository/seedance-efficacy \
  --preserve-timestamps --include-lowpass64 --long-side 512 \
  --noise 0 --noise 0.15 --device mps
```

`--include-lowpass64` explicitly adds `reencode-native.mp4` and
`lowpass64-native.mp4`. Both preserve the source geometry and frame clock and
copy its audio. The low-pass arm strongly reduces spatial detail before
restoring the original dimensions. The exact filter is recorded in
`plan.json`, from the same definition passed to FFmpeg. It requires
`--preserve-timestamps`; there is no automatic quality gate or parameter search.
Add `--skip-vae` to run without loading model weights. The native-size arm does
not use the VAE's eight-pixel geometry rounding.

The VAE arms remain separate fixed reconstruction experiments. Their previous
Google calibration is not a CapCut or Seedance certification. Review the saved
outputs and attach captured provider receipts before drawing a signal-level
conclusion. Fine-detail preservation and universal erasure are not promised.

### Partial reconstruction for quality comparisons

An optional native-size mixture can retain more source detail:

```bash
uv run python scripts/seedance_video_study.py run /path/to/owned-label-cleaned.mp4 \
  --origin 'Owned Seedance generation, visible AI label masked' \
  --output-dir /path/outside/repository/seedance-quality \
  --preserve-timestamps --long-side 512 --noise 0 \
  --reconstruction-weight 0.25 --blend-crf 0 --device mps
```

`--reconstruction-weight` is the fraction of the decoded VAE reconstruction in
each mixed pixel; the remainder comes from the source. Repeat it to declare a
fixed comparison in advance. It accepts values strictly between zero and one,
requires `--preserve-timestamps`, and cannot be combined with `--skip-vae`.
Each guide is the saved, encoded VAE output, enlarged with Lanczos interpolation
to the source geometry. The frame-pair stream rejects mismatched timestamps.
The mixture uses float64 arithmetic, rounds to nearest even and clips to uint8.
It does not hold the entire decoded video in memory.

`--blend-crf` controls the native-size mixture encode and its own
`reencode-blend-crf<value>` control. Its default is zero, lossless YUV coding;
color conversion and chroma subsampling still prevent a pixel-exact round trip.
This control uses the same decoded-pixel path as the mixture and differs from
the direct-FFmpeg lowpass64 control. Mixture rows identify both their `guide_id`
and `reconstruction_weight`; their `control_id` names the correct codec control.
The plan records all weights, the blend codec setting and the SSIM implementation.

Returning source detail can also return the detectable signal. High SSIM alone
does not qualify a candidate, and an already negative codec control cannot
establish incremental suppression. Keep the complete planned matrix, including
failed candidates. Validate each new file with a positive matched control before
treating it as a signal-suppressed output. No mixture is enabled by default.

The complete source is processed. The preflight accepts only 8-bit SDR YUV
video. The default mode requires constant frame spacing. Use
`--preserve-timestamps` to process every source frame with its original PTS,
including variable spacing and nonzero start offsets. This mode cannot be
combined with `--fps`. It reuses the existing timestamped NUT bridge, restores
the decode-order and presentation-order ends through stream-copy operations
when needed, including after audio muxing, and requires the saved file's frame
times and video duration to match the source
within two microseconds. HDR and high bit depth remain unsupported. Geometry is rounded to the existing engine's eight-pixel
alignment and never intentionally upscaled. The actual dimensions and frame
rate appear in the delivered-file measurements.

The study writes:

- `plan.json`: source hash, operator-supplied origin, geometry, parameters,
  environment, and hashes of the principal implementation files. New runs also
  retain those files under `code/` so an uncommitted implementation remains
  inspectable after further edits.
- `results.json`: artifact hashes and paths, delivered-file quality, processing
  time excluding fidelity measurement and model loading, and independent
  watermark status. A finished batch has `complete: true`. An interrupted run
  retains its partial artifacts and report; do not reuse its directory.
- One metadata-only artifact and, for each geometry, a matched re-encode,
  temporal candidate, and VAE candidates. The VAE runtime report fingerprints
  the loaded tensor state and records the scaling factor and device. The final
  process RSS high-water mark includes evaluation and model loading; it is
  not a per-method memory measurement. `peak_process_rss_bytes` is `null` on
  platforms without the Unix `resource` module, including Windows.

## Experimental arms

| Arm | Purpose |
| --- | --- |
| Source | Establish that this delivery channel carries a detectable pixel watermark. |
| Metadata-only | Separate container provenance from pixel detection. |
| Re-encode | Control for resizing, frame selection, codec, and muxing shared by the candidate. |
| Codec | Change compression strength against the CRF18 re-encode control. |
| Temporal | Average motion-aligned neighboring source frames in a three-frame window. |
| Lowpass64, opt-in | Strongly suppress spatial detail, against a separate native-size codec control. |
| VAE, zero perturbation | Isolate reconstruction loss from added latent noise. |
| VAE, nonzero perturbation | Reuse the existing video regeneration engine with a fixed seed. |
| VAE/source mixture, opt-in | Trade reconstruction strength against source detail at native geometry, with a separate blend-codec control. |

The temporal arm is an original small experiment using the project's optical
flow helper. Forward/backward flow consistency, image boundaries, and local
photometric disagreement gate averaging. A large median disagreement rejects
a neighbor across a cut. Thresholds are exploratory, not learned or calibrated
against Seedance. It may preserve the watermark, suppress real texture, or
produce motion artifacts. It does not implement FlowDenoise or PS-SR.

Quality comes from `video_fidelity_probe.py`, which decodes the saved output
and compares it with source frames. PSNR/SSIM rank fidelity, not watermark
removal. With `--preserve-timestamps`, fidelity pairs all source and output frames in
order only after validating their PTS and video durations. The standalone
`video_fidelity_probe.py` accepts the same flag. Decoded-frame memory is bounded;
the timestamp and terminal-duration probes retain linear-size metadata lists.
The temporal ratio is unstable on nearly static scenes, and sampling
source frames at the output rate does not measure the perceptual cost of lost
frames. Always read geometry and frame rate alongside these metrics and review
motion. Null PSNR represents an unbounded/nonfinite result, not zero quality.

The encoder copies source audio; audio watermark status remains `unverified`.
No result certifies erasure of all provenance signals.

## Attach external evidence

Establish a positive matched video control with the
[CapCut checker](https://ai-generated-detect.capcut.com/). Its minimal-detail
result reports a supported provenance signal. It does not identify an invisible
watermark family. An empty `signals` array, C2PA verification, a general AI-origin
answer, or a negative result on an already negative control cannot establish
pixel-watermark removal. Keep separate raster checks separate from video verdicts.

Capture one immutable upload per artifact in a finite pass. Retain the selected
File's SHA-256 before submission, complete request/response records, pending job
identity, file echo and typed completion. Keep failures and contradictions. Do
not upload a source again merely to complete an assessment. `assess` performs no
network calls, uploads, retries or parameter selection.

Wrap each captured receipt in a JSON-array entry:

```python
observations = [
    {
        "session": "one-frozen-pass",
        "surface": "https://ai-generated-detect.capcut.com/",
        "artifact_id": "reencode-native",
        "receipt": captured_control_receipt,
    },
    {
        "session": "one-frozen-pass",
        "surface": "https://ai-generated-detect.capcut.com/",
        "artifact_id": "lowpass64-native",
        "receipt": captured_candidate_receipt,
    },
]
```

`receipt` is the complete captured record, not a manually assigned verdict or
quoted UI text. `scripts/capcut_receipts.py` checks its `capture.requests` and
`capture.responses`, selected-file hash, one file submission, job-only polls,
`exchange.final`, and `response`. A valid completion must echo the captured
filename, expected size and video media type, carry a real boolean
`signal_detected`, and have a compatible display state. Uploaded filenames may
differ from saved artifact names; their hashes and sizes must match. Local
receipts are auditable records, not signed CapCut attestations.

Only a detected `control_id` and a negative candidate in the same pass and
surface can yield `capcut_signal_status: observed_suppression`. Every matching
receipt must agree. Missing, malformed, failed, mismatched or contradictory
evidence remains `unverified`. Legacy manual `verdict` / `raw_response` entries
cannot qualify. `watermark_status` and audio watermark status remain
`unverified`, because this service result does not identify or decode a family.

```bash
uv run python scripts/seedance_video_study.py assess \
  /path/outside/repository/seedance-run/results.json \
  /path/outside/repository/observations.json \
  --output /path/outside/repository/assessment.json
```

Assessment rechecks every artifact and the plan hash, preserves the original
results, and refuses to overwrite an existing assessment. Even
`capcut_signal_status: observed_suppression` is scoped to this checker and pass,
with `certifies_erasure: false`. Assessment also records the hashes of the code
that interpreted the receipts.

## Next implementation gates

1. Establish positive matched video controls and separate raster evidence for
   each delivery channel. A minimal-detail service label does not identify the
   watermark family. Without positive controls, experiments measure execution
   and fidelity only.
2. Extend timing regression coverage before supporting additional containers.
   The timed path now preserves the retained Dreamina 2.5 capture's alternating
   frame spacing and shortened terminal packet, including after VAE processing.
3. Compare fixed VAE and temporal configurations at matched geometry and frame
   rate. Change one factor at a time; separate held-out clips from tuning clips.
4. Integrate a video reconstruction model only after auditing its code, weights,
   license, and runtime. PS-SR remains a candidate, not an implemented arm.
5. Validate the selected configuration on new positive sources and final editor
   exports, including faces, text, cuts, motion, audio, and synchronization.
   Promote a public Seedance profile only after that evidence exists.

For advertisements with independently added captions and app demonstrations,
evaluate processing the generated segment before compositing those assets.
Do not blend original marked pixels back into a processed segment to restore
quality without rechecking the result.

## Initial local pilot (2026-11-07)

The full Higgsfield capture at
`data/captures/2026-09/higgsfield/seedance-2-5.mp4` has SHA-256
`24857c4744ee9a7e4385d1f85df0acdd0aba05f1827210b7b97556d1a9d58841`.
The metadata control preserved all 97 encoded video packets. The following
SSIM values were measured on the delivered files against the original:

| Profile | Matched re-encode | Temporal | VAE, zero noise | VAE, noise 0.15 |
| --- | ---: | ---: | ---: | ---: |
| 512x288, 12 fps | 0.9771 | 0.9758 | 0.9539 | 0.9294 |
| 768x432, 24 fps | 0.9830 | 0.9821 | Not run | 0.9497 |

This compares two combined geometry/frame-rate profiles, not an isolated
resolution experiment. The carrier is a mostly static cup-and-steam scene,
so these measurements cannot establish quality on faces, typography, or fast
motion. Both profiles remain unverified for Seedance watermark suppression.
The optimized temporal implementation produced byte-identical saved artifacts
in a separate final-code validation run. Timing measurements include host
contention and should not be treated as a throughput benchmark.

A separate local VideoSeal probe used 16 frames from the publication-cleared
Sora carrier, resized to 256x256 at 12 fps, with the pinned oracle's fixed
message. The saved clean control decoded at 0.53125 bit accuracy; the marked
source and matched CRF18 re-encode both decoded at 1.0. The temporal output
decoded at 0.99609375, retaining 255 of 256 message bits and remaining detected.
This is evidence against this temporal configuration as a sufficient VideoSeal
remover. It neither proves nor disproves suppression of ByteDance's signal.

On the same saved marked source, the existing VAE at 256x256, 12 fps, seed 0,
and noise 0.15 reduced the decoded VideoSeal message accuracy to 0.5, below
the adapter's 0.9 threshold. Both the marked source and its matched re-encode
were positive. This demonstrates detector suppression in this one local
VideoSeal case, not Seedance removal or acceptable advertising quality.
Delivered-file fidelity against the saved unmarked control was PSNR 20.991 dB
and SSIM 0.4822. This large quality cost on the more detailed carrier is a
reason to optimize and validate reconstruction quality alongside detection.

The pilot scripts, hash-bound reports, logs, and media are retained outside the
repository in `/tmp/raiw-seedance-study-TIcXHL/`. The prior CapCut verdicts are
referenced by hash there; no new external uploads were made for this pilot.

## Timing and quality follow-up (2026-11-07)

The timestamped path completed on the full Dreamina 2.5 capture, including
VAE reconstruction with zero added latent noise. Every one of the 121 display
PTS values and the video stream duration matched the source at ffprobe's
reported precision. The metadata and temporal/control arms retained all 220
AAC packets with identical payload hashes, PTS, DTS, and durations. The rawvideo
bridge initially lengthened the terminal video packet; the explicit terminal
repair and a short-tail regression now guard that case. The VAE output's
end-to-end SSIM was 0.9332 at 512x288, against the 1280x720 original. This is
fidelity and timing evidence, not ByteDance watermark verification.

On the same Sora-carrier VideoSeal tuning clip as the initial pilot:

| Candidate | Message bit accuracy | Delivered SSIM against the unmarked control |
| --- | ---: | ---: |
| VAE 256, noise 0 | 0.484375 | 0.5465 |
| VAE 256, noise 0.025 | 0.5390625 | 0.5430 |
| VAE 256, noise 0.05 | 0.5234375 | 0.5358 |
| VAE 256, noise 0.10 | 0.50390625 | 0.5123 |
| VAE 256, noise 0.15 | 0.500000 | 0.4822 |
| VAE working grid 512, noise 0 | 0.85546875 | 0.7143 |
| Original/VAE blend, 25% reconstructed | 0.9921875 | 0.9522 |
| Original/VAE blend, 50% reconstructed | 0.953125 | 0.8638 |
| Original/VAE blend, 75% reconstructed | 0.66796875 | 0.7168 |
| Gaussian blur, sigma 0.8 | 0.99609375 | 0.7250 |
| Direct H.264 re-encode, CRF32 | 0.8671875 | 0.8275 |

The 512-grid experiment deliberately upscaled this 256x256 test input before
reconstruction. It is a separate ablation, not the study runner's clamped
geometry policy. Lowering noise alone did not remove the reconstruction quality
floor. Blending original pixels back improved fidelity but restored the signal.
Blur was ineffective in this case.

A fixed CRF32 profile was then evaluated through the study's BGR encoding path
on three independent carriers. Each carrier was newly stamped with the local
VideoSeal message; these are **not tests of the carriers' native provider marks**.
The recipe used the first 16 frames, resized to 256x256 and encoded at 12 fps.

| VideoSeal carrier | CRF18 control accuracy | CRF32 candidate accuracy | Candidate SSIM |
| --- | ---: | ---: | ---: |
| Higgsfield capture | 1.000000 | 0.65234375 | 0.9386 |
| Dreamina capture | 1.000000 | 0.70312500 | 0.9134 |
| Veo capture | 0.968750 | 0.61328125 | 0.9776 |

All clean controls were below the adapter threshold, and all marked controls
were positive. The Sora tuning case reproduced through the same BGR path at
0.87890625 accuracy and SSIM 0.8268. The direct-YUV and BGR paths are different
transforms and their numbers must not be combined as repeated identical runs.

**Below 0.9 is not equivalent to erased.** For example, 0.8789 still retains
most message bits. Keep the continuous scores and the threshold alongside the
binary result. These experiments support a codec baseline in future comparisons;
they do not establish universal removal or a validated Seedance profile.

The follow-up reports, recipes, and public experimental media remain under
`/tmp/raiw-seedance-next-8itRDK/`. The `private-marko/` subdirectory contains
private local-only artifacts and is not publication-cleared. Do not copy its
contents into the repository or upload them without separate authorization.

PS-SR remains unexecuted. Its [published setup](https://github.com/HiDream-ai/PS-SR)
targets Linux/CUDA and a separate dependency stack. The
[inference entry point](https://github.com/HiDream-ai/PS-SR/blob/main/inference_step_1.py)
accepts a device argument, but this does not establish MPS support or watermark
removal. Its decoded-frame video writers also require adaptation or a separate
frame handoff to preserve the timestamped contract used here.

## Native-detail follow-up (2026-11-10)

A frozen comparison on an owned forest clip evaluated native and doubled-grid
SD VAE reconstruction, the independent 16-channel Ostris autoencoder, and
source-texture transfer into a reconstructed guide. The texture profiles used
predeclared Gaussian scales, with separate luma-only and fixed global color
matching ablations. Every delivered video retained all 121 presentation times
and durations and the original audio payload, configuration, and packet clock.
All delivered videos decoded completely through AVFoundation on the test Mac.

| Forest profile | Mean gray SSIM | Minimum frame SSIM | CapCut video verdict |
| --- | ---: | ---: | --- |
| Native codec control | 0.996423 | 0.993151 | Detected |
| Doubled-grid VAE | 0.967309 | 0.953225 | Detected |
| Native 16-channel VAE | 0.956041 | 0.937470 | Detected |
| RGB texture transfer, sigma 1 | 0.899604 | 0.873579 | Not detected |
| RGB texture transfer, sigma 4 | 0.983026 | 0.953623 | Detected |
| Luma texture transfer, sigma 4, color matched | 0.991036 | 0.986144 | Detected |

The complete frozen pass contained twelve distinct videos and twelve matched
frame images. All requests settled and their artifact hashes, typed verdicts,
and request/response captures passed offline verification. The video control
was positive, so the sigma-1 result supports suppression of the service's
supported signal on this clip. It does not identify the watermark family or
prove complete erasure. Its texture and color remain visibly different from
the source. The high-fidelity profiles did not suppress the detected signal.

Separately, retained full-strength DRUNet and FastDVDnet predictions were
qualified locally against all forest frames, with the same visible-label mask
on both sides. Both failed the predeclared fidelity gate, so neither was sent
for a new provider check. No new certified removal profile follows from these
experiments. Raw media, model pins, designs, receipts, and per-frame metrics
remain in the private research archive outside the repository.

## Competitor and method audit (2026-11-10)

The product acceptance criterion for this study is a positive CapCut control
and a negative processed video in the same frozen pass. Universal watermark
erasure, forensic undetectability, and a change in social distribution are not
additional acceptance gates. Fidelity is measured separately on that same
delivered artifact. Competitor documentation, inspected code, author-reported
experiments, and our own provider results are distinct evidence classes.

### What competing products actually process

| Project or service | Inspected mechanism and video scope | Relevance to this study |
| --- | --- | --- |
| [SynthIDx video remover](https://synthidx.com/ai-video-watermark-remover) | Its video page explicitly limits the job to visible-label repair and metadata cleanup, leaving video SynthID untouched. | Its image regeneration offer is not evidence of invisible video removal. |
| [RemoveAILabel](https://removeailabel.io/) | The supported-format table offers pixel processing for images and C2PA-only processing for video. The page also contains conflicting broader claims. | The table and implemented media path matter more than the headline. No checked Seedance pixel-removal evidence. |
| [RemoveSynth](https://removesynth.com/) | The served uploader tries a configured remote `/process` endpoint, then silently falls back to a local cleaner on failure. The local video branch filters top-level container boxes; it never decodes frames. | The server implementation is unknown. Its local fallback cannot remove a pixel watermark. A synthetic four-frame MP4 passed through the exact helper byte-identically, including nested title metadata. This was a local code probe, not a server or watermark test. |
| [MarkCut](https://markcut.com/ai-video-watermark) and [Imaginoa](https://imaginoa.com/video-watermark-remover.html) | Both explicitly target visible marks. Imaginoa describes estimating fixed overlay transparency across frames and reversing the blend. | Useful visible-label methods, outside the present pixel-signal problem. |
| [CyberBara's Seedance API guide](https://github.com/ZeroLu/seedance-API) | Exposes `seedance-2-watermark-remover` as a hosted video-to-video route. The public client submits a task, rather than supplying removal weights or an algorithm. | A possible external comparison, not a reproducible local implementation. The route name does not establish invisible-mark coverage or ByteDance endorsement. |
| [wmr](https://github.com/froggeric/gemini-watermark-and-synthid-remover) | Invisible-image path uses lossy SDXL img2img, with optional source-detail restoration. The documented video path removes visible marks; video SynthID is listed as future work. | Its restoration code is useful, but its image verifier results do not certify Seedance videos. |
| [noai-watermark](https://github.com/mertizci/noai-watermark) | Image img2img and CtrlRegen, including tiled processing and color matching. The inspected CtrlRegen engine takes a PIL image and has no temporal model. | An image backend requiring a separate temporal wrapper and full-video validation. |
| [watermarks-remover](https://github.com/guillaumemeyer/watermarks-remover/blob/5a62be6338d2812372b1da440e55c82922f97d53/service/scripts/clean_video.py) | Extracts frames, runs an external CtrlRegen or DiffusionPurification backend, then remuxes audio and frame durations. Its frame-selection model assumes a TrustMark-B temporal vote. | A real video orchestration layer. The vote threshold is specific to that checker and cannot be assumed for CapCut. No cross-frame neural reconstruction is introduced by the wrapper. |
| [pagedMark](https://github.com/doofzoff/pagedMark) | The inspected video module uses a frame VAE with shared latent noise. Its Metal-oriented image restoration is a separate pipeline. | Similar video mechanism to our existing baseline, rather than an evidenced new Seedance operating point. |
| [hpdkhoa/rw](https://github.com/hpdkhoa/rw) | GitHub repository metadata identifies its parent as `wiltodelta/remove-ai-watermarks`. | Do not count a fork of this project as independent competitor validation. |

### Reproducible algorithm families

**Image regeneration with controlled structure.**
[CtrlRegen](https://github.com/yepengliu/CtrlRegen) starts from clean noise or a
partially noised latent and reconstructs using semantic and spatial controls.
Its released adapter weights are available on
[Hugging Face](https://huggingface.co/yepengliu/ctrlregen). Unlike a VAE-only
round trip, this performs learned denoising. Independent frame sampling can
change fine features differently in adjacent frames, so a video adaptation
needs temporal conditioning or consistency constraints. A fixed seed alone
does not establish temporal consistency. The original GitHub tree did not
contain a license file in the inspected revision; the model card declares
Apache-2.0. Keep code and weight licensing separate before distribution.

[Synthid-Bypass](https://github.com/cebeuq/Synthid-Bypass) uses a Qwen-based
ComfyUI image reconstruction with Canny conditioning and a separate face
detail pass. [DeSynth](https://github.com/0xROOTPLS/DeSynth) currently uses
Qwen-Image 2.1 img2img followed by frequency separation. Its current code
differs from the older Qwen-Image-2512 revision evaluated in the image research
archive; that older failure is not a test of the new revision. Neither is a
complete, validated Seedance video implementation.

**Source-detail restoration.** wmr's inspected
[restoration code](https://github.com/froggeric/gemini-watermark-and-synthid-remover/blob/37f338847c7fcdbaa1ab3f883c6a742173dd9595/src/core/regen_restore.cpp)
implements `original - blur(original - reconstructed)`, equivalent to keeping
the reconstructed low-frequency band and the original high-frequency band.
It disables automatic restoration below an empirical mean-luminance threshold
because the author's dim-image checks detected restored outputs. DeSynth
also provides an edge-aware split and a sharper Butterworth crossover. These
are different filters, not watermark decoders. Our forest texture-transfer
experiment already tests the Gaussian family; copying that formula alone
does not provide a new solution. A different reconstruction backbone and
temporally coherent reconstruction are the substantive remaining changes.

**Video reconstruction after controlled degradation.**
[ShadowMarkDefense](https://github.com/shikanon/ShadowMarkDefense) supplies real
adapters for compression plus BasicVSR, PS-SR, DCT perturbations, and RIFE.
Its documentation proposes scoring the outputs; it does not supply a checked
Seedance/CapCut positive-to-negative cohort in the inspected material. The
bounded DCT branch has already been tested in our private study and did not
clear the forest control. The external reconstruction branches remain
separate candidates.

[PS-SR](https://github.com/HiDream-ai/PS-SR) is an Apache-2.0 video
super-resolution implementation with base and draft checkpoints. It uses
video diffusion and a frequency-constrained refinement stage. It is a
restoration model, not a demonstrated Seedance remover. Its documented
environment is Linux/CUDA. Its stock frame writers use a nominal FPS; our
timestamp/audio contract therefore requires a decoded-frame adapter.

[SeedVR2](https://github.com/ByteDance-Seed/SeedVR) and
[FlashVSR+](https://github.com/OpenImagingLab/FlashVSR_Plus) provide additional
open video-restoration candidates. SeedVR2 is from ByteDance-Seed, but this
does not establish anything about watermark compatibility. Its authors warn
that lightly degraded AIGC clips, particularly small inputs, can become
oversharpened. FlashVSR+ publishes code and model links, targets Linux/CUDA,
and exposes a tensor API. Its stock CLI center-crops and resizes to dimensions
divisible by 128, so geometry preservation requires explicit padding and
cropping around the tensor path. Neither repository reports the required
CapCut result in the material checked here.

**Spectral optimization without a target decoder.**
[UnMarker](https://github.com/andrekassis/ai-watermark) optimizes spectral
changes under perceptual constraints instead of using detector feedback.
Its shipped SynthID configuration includes a crop and long per-image
optimization. Independent application to frames may flicker. A useful video
experiment would require temporally coupled perturbations and a geometry
control, rather than simply porting the image loop. Its
[source license](https://github.com/andrekassis/ai-watermark/blob/58ba69259dd1bd3391bf9c8b0fe93912df0125b1/LICENSE)
restricts use to noncommercial research or evaluation; it is not an
unrestricted production dependency.

**Motion-aware denoising and frame interpolation.**
[FlowDenoise](https://github.com/AIMZ-GFX/ComfyUI-FlowDenoise) contains MEMFOF
and RAFT motion-compensated averaging, color-similarity masks, scene-cut
handling, and occlusion checks. It treats video noise, not a decoded Seedance
payload. Its watermark effectiveness is a hypothesis requiring a positive
control. This is distinct from both single-frame DRUNet and five-frame
FastDVDnet. RIFE in ShadowMarkDefense changes temporal sampling; retaining
all original frames can retain their signals too. A changed cadence is not
equivalent to verified removal.

### Newer research with incomplete deployment evidence

[PoseRegen, revised November 6, 2026](https://arxiv.org/html/2610.22457v2),
jointly reconstructs a sequence using a video DiT, pose guidance, a trained
motion adapter, and a shared viewpoint perturbation. The authors report
near-random average bit recovery across eleven open watermark schemes.
Their experiment is not Seedance/CapCut, and its VBench motion scores are
not paired source SSIM. The linked
[project page](https://sites.google.com/view/poseregen) exposed a demo during
this audit, but no usable code/checkpoint download was found there. Pose
guidance also fits articulated foreground subjects better than a forest
texture scene. Treat this as architecture evidence, not a ready backend.

[NFPA](https://github.com/1249748036/NFPA) has inspectable next-frame-prediction
code, but its target is an image reconstructed as a hypothetical next frame.
It does not consume and faithfully restore an existing moving sequence.
[Disentangle and Drop](https://arxiv.org/abs/2609.31693) instead learns a
grayscale structural carrier and residual branch, then drops the residual.
It evaluates open image watermark families. No author code/checkpoint release
was located in the checked paper and title search. Both require substantial
work before a meaningful Seedance video comparison.

### Recommended experiment order

1. **Video restoration after a frozen signal-suppressed reconstruction.**
   Compare PS-SR and an efficient released video-restoration model such as
   FlashVSR+ on the same prepared sources. Include each reconstruction alone
   and its restoration result. This tests whether a temporal prior improves
   the picture without directly copying the original fine-detail residual.
   Reconstructed texture may be plausible rather than source-exact; measure
   both visual quality and paired fidelity.
2. **Controlled low-strength regeneration with temporal constraints.**
   Compare a source-structure-conditioned denoiser against the existing
   VAE-only path. Reuse the authoritative frame clock and original audio.
   Temporal attention or flow-aligned consistency is a real implementation
   task, not something supplied by an image CLI.
3. **Temporally coupled spectral optimization.** Use open watermark decoders
   and synthetic marked/clean pairs for development. Evaluate the fixed
   output with CapCut once the local profile is frozen. This is a separate
   research branch with more development and licensing work.
4. **Motion-compensated averaging or generated-frame replacement.** Run as
   cheaper independent ablations, with explicit edge, occlusion, and timing
   checks. Their removal effectiveness is not established by a noise-removal
   demo or a known-checker majority-vote calculation.

For every family, freeze a finite matrix on several owned, control-positive
scenes. Score the delivered video for full-frame and worst-frame fidelity,
color drift, text/face changes, and temporal artifacts. A CapCut-positive
control and negative candidate are sufficient for this task. Do not reuse a
negative control as proof of suppression or tune parameters adaptively against
the external checker. This audit did not upload media to competing services,
purchase credits, or execute their model pipelines. Public code snapshots,
commit bindings, service assets, local probe results, and retrieval failures
are retained in the private research archive outside the repository.

## Label-free TikTok delivery check (2026-11-10)

The owned forest generation also has an existing official CapCut export with
the visible `CapCut AI` overlay disabled. This export is a distinct encoded
artifact, not the earlier gray-masked research input. A new frozen pass tested
the complete label-free path, without another generation or paid operation.

All seven files settled in the official CapCut checker:

| Label-free forest artifact | CapCut verdict | Mean gray SSIM | Minimum frame SSIM |
| --- | --- | ---: | ---: |
| Official export | Detected | 1.000000 | 1.000000 |
| Metadata-only cleanup | Detected | 1.000000 | 1.000000 |
| Native CRF0 codec control | Detected | 0.996214 | 0.992612 |
| Reduced-grid zero-noise VAE guide | Not detected | 0.749428 | 0.671003 |
| Native RGB texture transfer, sigma 1 | Not detected | 0.896676 | 0.866547 |
| Delivery-codec control, H.264 High CRF18 | Detected | 0.973298 | 0.960581 |
| Delivery texture transfer, H.264 High CRF18 | Not detected | 0.873815 | 0.841170 |

Fidelity is relative to the official label-free export. The metadata-only
candidate retained every encoded video packet and decoded frame unchanged;
its positive verdict shows that container cleanup alone did not meet this
clip's acceptance criterion. Both processed native-size outputs have positive
same-pass codec controls and negative candidate verdicts. The delivery encode
was checked as its own final artifact, rather than inheriting a verdict from
its lossless-coded parent.

All seven outputs preserve the original audio payload, codec configuration,
and packet clock, and all 121 video presentation times and frame durations.
The source and seven outputs decoded fully through AVFoundation. Both final
files contain no recognized AI metadata. The former top-left label region
was inspected across all 121 frames of the official source and delivery
output, and a full-frame timeline comparison was reviewed. No artificial gray
mask is present. The final delivery file is 854x480, 24 fps, H.264 High,
8-bit YUV420p MP4, with the source audio copied.

The fixed operator is the previous sigma-1 texture-transfer recipe applied to
the unbranded source; no strength was selected from this pass's responses.
It passes the accepted CapCut criterion but does not meet the separate
high-fidelity target. Fine texture and color differences remain. There was no
TikTok publication or reach experiment. Raw captures, hashes, source and
delivery bindings, full-frame metrics, and local failed attempts are retained
outside the repository.

## Experimental library entry point

The fixed label-free texture-transfer recipe is available through
`remove_video_seedance_experimental` in the package's lazy Python API. The
implementation is `src/remove_ai_watermarks/video_seedance.py`, with explicit
`delivery` and `master` modes. It uses no research-script imports and performs
no provider requests. The common video invisible/all/batch pipeline also
accepts the explicit `seedance-experimental` profile. In all mode, the existing
visible-removal stage precedes it; the standalone function still expects a
label-cleaned source. Batch processing reuses a matching VAE runtime.
See [Python API](python-api.md#experimental-seedance-video-reconstruction).

A complete native forest replay through the public function reproduced both
previously checked MP4 files byte for byte, including the delivery encode.
This binds the earlier CapCut verdicts to the library's reproduced artifacts
without another upload. It does not change the scene coverage or measured
fidelity. Synthetic video tests additionally exercise variable timestamps,
nonzero start offsets, copied audio tails, and atomic failure behavior.

A subsequent replay through `remove_video_all` with
`include_invisible=True` and `invisible_profile="seedance-experimental"`
reproduced the checked delivery file byte for byte as well. Synthetic integration
tests cover visible-stage ordering, profile validation before processing,
shared batch runtimes, and CLI forwarding. The built wheel exposes the same
selectors after installation in a clean environment.

FFmpeg-dependent study tests require both `ffmpeg` and `ffprobe`; pure receipt,
assessment, texture, and importability checks remain active without them. CI's
mandatory `video full-clip end-to-end` job installs FFmpeg and runs both the
study and experimental library suites alongside the existing video checks.
