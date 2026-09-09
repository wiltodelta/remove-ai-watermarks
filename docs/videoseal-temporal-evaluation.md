# VideoSeal temporal evaluation

> Audit correction, 2026-09-08: The historical 2026-09-07 matrix and its
> conclusions are superseded by the corrected run below. The audit found
> transposed rectangular video decoding (R01) and
> observations measured before encoding rather than on the hashed artifact
> (R12). The table remains an archive, not validated delivered-file evidence.

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
