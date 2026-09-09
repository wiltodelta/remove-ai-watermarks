# Watermark forgery study

> Audit correction, 2026-09-08: The historical VideoSeal results and
> cross-scheme conclusions are superseded by the corrected video run below.
> Historical video observations were taken before encoding
> rather than from the hashed output artifact (R12); rectangular video
> decoding also required correction (R01). AudioSeal observations remain
> separate from those invalidated video comparisons.

## Corrected VideoSeal measurements (2026-09-08)

Twelve observations now measure decoded saved artifacts, including controls
and double embeds. All 12 recorded SHA-256 digests match the retained MP4s;
the decoder also checks that each file remains unchanged during measurement.

| Cell | Gradient vs A | Gradient vs B | Real Sora vs A | Real Sora vs B |
| --- | ---: | ---: | ---: | ---: |
| clean | 0.516 | 0.555 | 0.559 | 0.535 |
| marked A | 1.000 | 0.469 | 1.000 | 0.469 |
| removed CRF 23 | 0.559 | 0.496 | 1.000 | 0.469 |
| forged B on clean | 0.469 | 1.000 | 0.469 | 1.000 |
| forged B on marked A | 0.469 | 1.000 | 0.473 | 0.996 |
| forged B on removed A | 0.469 | 1.000 | 0.469 | 1.000 |

Both marked controls meet the fixed-message A rule. All six B-embedding
outputs fall below its 0.9 threshold, while matching B at 0.996 or above.
CRF 23 removes the matched A verdict on the gradient but preserves it on
this Sora carrier. These two fixed carriers support this scoped observation;
they do not establish complete erasure of every trace of A or a universal
double-embedding rule. AudioSeal was not rerun in this correction.

The run used the same pinned checkpoint, Torch 2.14.0, ffmpeg 9.0.1, and
two CPU threads as the corrected [temporal study](videoseal-temporal-evaluation.md#corrected-artifact-measurements-2026-09-08).
No model downloads or provider calls ran. Study source SHA-256:
`031d9e0c58af58ab5f9227ea52a500eb86644fc464f9c37be1e894ce0b6b4935`.
Corrected case rows SHA-256:
`f8c5a65e5853de0f1ff67c2ef0549fbd81df971e13c79539c85c1b0255eb60a8`.
Artifacts remain outside the repository.

## Historical study (2026-09-07, superseded video results)

Development-only study, run on 2026-09-07 from
`scripts/watermark_forgery_study.py`, opening the forgery/remover-forensics
direction from the original research map. It measures the four-state model
(clean / marked / removed / forged) and the double-embedding question for both
pinned oracles: does embedding message B over an A-marked artifact leave A,
B, or mutual interference?

Carriers: the tone-stack audio carrier, the synthetic moving-gradient video
clip, and the real committed Sora clip. Message B is the oracle's message
generator at seed 8. Everything local; artifacts under `.local-eval/`.

## The double-embedding answer: the last writer wins, completely

| Cell | AudioSeal acc vs A | vs B | VideoSeal (gradient) vs A | vs B | VideoSeal (real Sora) vs A | vs B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| marked A | 1.000 | 0.44 | 1.000 | 0.47 | 1.000 | 0.47 |
| removed | 0.50 (presence 0.28) | 0.44 | 0.57 | 0.48 | 1.000 | 0.47 |
| forged B on clean | 0.44 | 1.000 | 0.47 | 1.000 | 0.47 | 1.000 |
| **forged B on marked A** | **0.44** | **1.000** | **0.47** | **1.000** | **0.48** | **0.99** |
| forged B on removed | 0.56 | 0.875 | 0.47 | 1.000 | 0.47 | 1.00 |

Embedding B over marked A reads as pure B: the original message decays to
coin-flip accuracy under both schemes, on synthetic and real content alike.
There is no mutual interference and no residue of A. With an unkeyed, openly
published embedder this is the expected outcome - and now it is a measurement,
not an assumption. The forensic consequence: payload attribution without key
control is meaningless against an attacker who can run the same open embedder;
only presence detection survives, and even that carries the attacker's
payload. Provider schemes that rely on secret keys are a different question
this study cannot answer.

Two boundary observations travel with the table:

- The removal transform is carrier-dependent, as the cohorts already showed:
  10 dB noise removes the tone stack (presence 0.28) and crf 23 removes the
  gradient clip (0.57), but the real Sora clip keeps a perfect 1.000 decode
  through the same crf 23 - the same number the temporal study recorded.
- A forged row reads `detected` under the AudioSeal presence rule (the label
  carries the foreign payload) and `not_detected` under the VideoSeal
  message-accuracy rule. Both cohort fixtures encode their scheme's correct
  expected value, and the benchmark docs state the two rules side by side.

## Remover-trace probe (scoped first evidence)

High-to-low frequency band energy ratio on clean versus noise-removed audio:
the tone stack jumps from 8.6e-05 to 2.3e-02 (a 269x trace), the pinkish
carrier 0.016 to 0.039, and the white-noise carrier does not separate at all
(0.330 to 0.332) - a noise carrier absorbs a noise remover. This detects the
trace of THIS study's gaussian-noise removal only; it is not a general
remover detector, and the probe JSON says so in its scope field.

## Cohort arms

The audio cohort (v2, 20 cases) and video cohort (v2, 15 cases) now carry the
four states as strict manifest rows: `removed` rows where the transform
measurably destroys the mark (audio: noise on tone stack; video: crf 23 on
both synthetic carriers), and `forged` rows embedding message B with the
scheme-correct expected verdict. Zero expected-result mismatches in both
recorded runs.

## Pins

Study script
`sha256:3985522c862ac44a0c7ef8aa02946a11701439a94d67f5867f00819f638c032b`,
case rows
`sha256:03d446c08a5daed623a4be2980e097bb8bbc0cd7a6d3ab3f931d25aa58570a16`.
Oracles and checkpoints are pinned as in the kernel document.

## Follow-ups

- A keyed-scheme perspective needs a scheme with actual key control; WmForger
  and WMCopier diffusion-attack baselines remain the map's heavyweight item.
- A general remover-trace detector would need features beyond one band ratio
  and validation against removals this project did not produce.
