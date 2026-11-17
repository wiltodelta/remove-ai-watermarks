# VideoSeal temporal evaluation

> Audit correction, 2026-09-08: The historical 2026-09-07 matrix and its
> conclusions are superseded by the corrected run below. The audit found
> transposed rectangular video decoding (R01) and
> observations measured before encoding rather than on the hashed artifact
> (R12). The table remains an archive, not validated delivered-file evidence.

## VAE regeneration evaluation (2026-11-08)

Run the shipped video-pixel regeneration engine against the same standardized
synthetic and real-content carriers, with a matched codec control:

```bash
uv run --extra dev --extra diffusion python scripts/videoseal_temporal_study.py \
  --regenerate --output-dir /outside/repo/videoseal-regeneration
```

The same control protocol supports the pinned PixelSeal video model through
`--adapter pixelseal-video`. Its model and source pins are recorded independently;
the dated VideoSeal measurements below remain specific to VideoSeal. See
[non-SynthID video coverage](non-synthid-video.md) for the PixelSeal protocol.

This mode replaces the H.264 attack sweep. It retains the clean and marked
controls and adds four saved-artifact arms per carrier: the engine's sampling,
resize, and encoder without VAE; a zero-noise VAE round trip; the shipped
default VAE profile; and the same default applied to the clean negative.
Each positive carries a locally embedded fixed VideoSeal message from the
pinned research checkpoint. Veo and Sora name the content carriers. The real
clips are standardized square prefixes through `real_carrier_frames`; this
experiment does not establish native-resolution, full-clip population rates.
The control uses the same engine helpers and encoding profile as regeneration.
The VAE runtime loads through `load_video_vae_runtime`, including its latent
scaling-factor gate. `cases.jsonl` records the model revision and checkpoint
hashes, engine hash, profile parameters, artifact hashes, average-aggregation
message accuracy, and per-frame readings. `matrix.md` summarizes the run.

Delivered fidelity is measured after decoding and muxing against the decoded
marked source, or the clean source for the negative arm, using
`video_fidelity_probe`. Unlike the engine's in-loop PSNR, it includes resize and
codec effects. A source-positive, output-negative reading establishes loss of
the fixed message under this oracle's rule; only a positive codec control can
separate the VAE contribution from ordinary transcoding. These local VideoSeal
results do not certify proprietary video watermarks or the copied audio track.

### Saved-artifact results

The local MPS run used the shipped profile from `video_synthid.py`, the pinned
VAE revision `31f26fdeee1355a5c34592e401dd41e45d25a493`, and seed 0.
Each carrier had 64 frames at 256x256 and 12 fps; the shipped geometry produced
512x512 outputs. All four clean controls were negative and all four embedded
marked controls positive under the unchanged average-aggregation threshold of
0.9. Every retained artifact's SHA-256 matched its recorded measurement.

| Carrier | Codec control accuracy | Zero-noise VAE accuracy | Default VAE accuracy | Default detected | Delivered PSNR (dB) | SSIM | Temporal residual ratio |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| moving_gradient | 0.9961 | 0.9844 | 0.8203 | No | 33.92 | 0.9701 | 1.0120 |
| moving_texture | 0.9961 | 0.8438 | 0.7578 | No | 24.86 | 0.8423 | 1.0096 |
| real_veo | 1.0000 | 1.0000 | 0.9023 | Yes | 32.96 | 0.9657 | 1.7452 |
| real_sora | 1.0000 | 0.9727 | 0.8203 | No | 27.01 | 0.8073 | 1.8593 |

All codec controls remained positive, so the default profile's three negative
results have a matched positive transcode control. The regenerated clean
negatives remained negative on all four carriers, with accuracy 0.5117-0.5430.
The zero-noise VAE round trip crossed below threshold only on moving_texture.

The shipped profile therefore **does not consistently suppress VideoSeal below
the benchmark threshold**: the real_veo carrier remains positive. Even the
negative candidates retain 0.7578-0.8203 message accuracy, so a negative at 0.9
does not establish erasure of all watermark information. Fidelity and temporal
costs also vary by carrier. This run is a bounded comparison of the existing
profile, not a new VideoSeal removal certification or a change to the shipped
defaults. No stronger-noise profile was calibrated in this run.

The retained run is
`~/.cache/remove-ai-watermarks-dev/evaluations/videoseal-regeneration-2026-11-08-codex-CNRJ1FLC-v2/`.
Its `cases.jsonl` SHA-256 is
`393119c3abec8e5afa91f75027eab19b1e34f47872f11387572714459214fffd`.
Measurement source SHA-256:
`c60afb3dfc6dc593c482b0c8ee8736c8c651b3e6239164d707fce548c82229d4`;
engine SHA-256:
`b0bedbf2f8f04bec66533d403e7aa1c18bd4e14e13dba00ba07f55f7bab3e853`.
The archived measurement source precedes the report-only addition that renders
the regeneration table; the regenerated `matrix.md` uses report source SHA-256
`df706c57d3abda935ae298b518a99b03e919ba56474a1e7f11a17854564fbd93`.
An earlier interrupted attack-sweep attempt is retained separately and does
not contribute to these readings.

## Corrected artifact measurements (2026-09-08)

The corrected run measured 32 saved artifacts: four clean controls, four
marked controls, and 24 attack outputs. Every observation decoded its saved
MP4, with equal SHA-256 before and after decoding; all 32 recorded hashes
were also checked against the retained files. The four carriers and attack
recipes are the same as described in the historical study below.

| Carrier | crf18 | crf23 | crf28 | crf32 | scale .75 | fps .5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| moving_gradient | 0.879 | 0.559 | 0.520 | 0.512 | 0.711 | 1.000 |
| moving_texture | 0.582 | 0.547 | 0.535 | 0.562 | 0.879 | 0.969 |
| real_veo | 1.000 | 0.953 | 0.789 | 0.621 | 0.988 | 1.000 |
| real_sora | 1.000 | 1.000 | 0.992 | 0.879 | 1.000 | 1.000 |

Under the unchanged average-aggregation threshold of 0.9, frame-rate halving
retains detection on 4/4 carriers, downscale on 2/4, and CRF 32 on 0/4.
These are four fixed carriers, not population survival estimates. Norm-weighted
aggregation changes the moving-texture downscale result from 0.879 to 0.938;
it does not change the kernel's canonical average-aggregation rule.

This was a local CPU run with two Torch threads, Torch 2.14.0, ffmpeg 9.0.1,
and the existing pinned checkpoint listed below. Model downloads and provider
calls were disabled. Study source SHA-256:
`a440fc95a2712097ef47665eaf8a9e9a370545d7ba05748e961955c5222546a0`.
Corrected case rows SHA-256:
`de8d633f4f63ed45e1bd53cc29d30244cd5367034bf602c21baaadaab5ba85af`.
Artifacts remain outside the repository; this dated verdict and its limits
are the durable record.

## Historical study (2026-09-07, superseded)

Development-only study, run on 2026-09-07 from
`scripts/videoseal_temporal_study.py`, answering the three temporal questions
the case-level benchmark verdict aggregates away: how decoded bit accuracy
decays across H.264 quality, whether the frame-aggregation choice changes the
outcome, and how per-frame accuracy moves in time on real provider clips
versus synthetic carriers.

Carriers: the two synthetic cohort clips (moving gradient, moving texture)
plus the committed real provider videos for Veo and Sora, standardized to a
64-frame 256x256 prefix (first 64 frames, center-crop, INTER_AREA resize).
Attacks: crf 18/23/28/32 re-encodes, 75 percent downscale, and frame-rate
halving. Everything is local; the real clips are already-committed project
fixtures with public test clearance.

## The TorchScript aggregation trap

The first matrix run produced identical numbers under all four aggregations,
and that was a measurement artifact, not a finding: the standalone TorchScript
build ignores the `aggregation` argument of `detect_video_and_aggregate` and
returns the same thresholded bits for every choice (verified by checksum on a
single artifact). The oracle now computes the aggregations itself from the raw
per-frame logits using the exact upstream `extract_message` formulas - avg,
squared_avg, and l1/l2-norm-weighted averages - and the recorded matrix comes
from those. The formula port is pinned by a hand-computed unit test.

## Matrix (bit accuracy of the aggregated 256-bit decode)

| Carrier | crf18 | crf23 | crf28 | crf32 | scale .75 | fps .5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| moving_gradient (clean 0.52) | 0.887 | 0.570 | 0.559 | 0.500 | 0.691 | 0.996 |
| moving_texture (clean 0.48) | 0.602 | 0.559 | 0.531 | 0.586 | 0.879 | 0.980 |
| real_veo (clean 0.52) | 0.801 | 0.723 | 0.621 | 0.574 | 1.000 | 0.855 |
| real_sora (clean 0.51) | 0.996 | 1.000 | 0.992 | 0.902 | 1.000 | 0.996 |

Readings, not verdicts: the benchmark adapter's rule is 0.9 under plain avg.

- **Carrier content dominates.** The real Sora clip keeps the message through
  every attack including crf 32; the real Veo clip decays smoothly with
  quality; the smooth synthetic gradient dies at crf 23. Any "survives H.264"
  claim must name the carrier and the crf.
- **Downscale is not an attack on real content.** 0.75 downscale plus re-encode
  scored a perfect 1.000 on both real clips - resize was a training
  augmentation, and the antialias resample appears to help the detector. On
  the synthetic gradient it still costs 0.31.
- **Frame-rate halving was not uniformly above threshold.** Even the
  historical table puts real Veo at 0.855, below the 0.9 rule; three of four
  listed carriers exceed it. The corrected artifact-domain run must establish
  whether that pattern survives.
- **Aggregation choice matters at the margin.** Norm-weighted averaging
  recovered 2-6 points on damaged arms (moving_texture scale: 0.879 avg
  versus 0.941 l1/l2, crossing the 0.9 rule) but never rescued a collapsed
  case and never lifted a clean negative above chance. The kernel keeps plain
  avg as its canonical rule; the matrix is the evidence for when that choice
  is conservative.

Per-frame accuracy on the marked clips: real_sora sits at 0.988-1.000 on
every frame, real_veo spans 0.547-0.910, moving_texture 0.457-0.961 - the
temporal spread is itself carrier-dependent, and the kernel's
`detection.temporal` block now records it for every videoseal case.

## Pins

Study script
`sha256:682ccf14b21d6ac17bb6500a7efb364247eebae2d9e0e8895ffac33e6d294d4f`,
oracle
`sha256:5aa67ceaca1b0050727aa0a75440b1bf4c583435b3bf9e8ef2e97fadd46a035c`,
case rows
`sha256:5494c9bbe8a6085ecbd27fd9732dadf8d004dab7be159e905051e854dabe64aa`.
Checkpoint: TorchScript `y_256b_img.jit`
`sha256:5c7a4581c36fc6090aafdcfb3999123bae5172a4847f22e2da4e7fd1a39d1e1b`
at upstream commit `870ca7f`. The kernel re-run of the video cohort with
temporal records used kernel
`sha256:657b0d7e35af75eb97b9bd9b81c15fcae754399f27999e08a5c28c494e69cf05`
and produced case rows
`sha256:435215e9a55589f6d83e34d62f34469c079db078a0a174ead64ffaa74af664c5`
with unchanged verdicts.

## Follow-ups

- A crf sweep on more real carriers (the other committed provider clips) to
  widen the carrier-dominated picture beyond two samples.
- Temporal inpainting comparison remains open from the original research map;
  the per-frame rows recorded by the kernel are the substrate for it.
