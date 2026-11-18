# Repository data

Tracked data is organized by purpose:

```text
data/
  fixtures/
    provenance/   Real format and provenance fixtures used by tests
                  (source records live in fixtures/README.md)
    visible/      Per-mark example gallery: one synthetic canonical example per
                  registered mark plus documented provider originals where available
                  (see fixtures/visible/README.md)
    synthetic/    Frozen synthetic renders that unit tests stamp onto frames, each
                  produced by the script named in the test that loads it
  captures/
    2026-09/      Provider outputs from the maintainer's own accounts, stored as
                  served, indexed by manifest.csv (see its README)
  calibration/
    <vendor>/     Minimal controlled inputs needed to rebuild detector assets
  synthid/
    originals/    Canonical provider-oracle fixtures, stored once
    manifest.csv  Provenance and verification record for each original
    full-pipeline-quality.csv
                  Reusable full-pipeline evaluation selection
  evaluations/
    engine-selection/
                  Prompt-matched, publication-cleared AI content matrix for
                  deterministic auto-engine and watermark benchmark research
    fidelity/     Evaluation instructions and hand-verified ground truth
    audio-synthid-oracle.csv
                  Reproducible Gemini SynthID audio verdicts for DAC / FocalCodec
    video-synthid-oracle.csv
                  Reproducible full-clip Gemini SynthID 2D VAE verdicts
    video-wan-oracle.csv
                  Reproducible native-resolution Wan 2.1 SynthID verdicts
```

[`image-watermark-examples.csv`](image-watermark-examples.csv) and
[`video-watermark-examples.csv`](video-watermark-examples.csv) are the coverage
indexes for real carriers. They point to each canonical binary instead of
duplicating it and record unsupported gaps and source-specific licensing.

## Storage rules

Use only publication-cleared inputs in tracked data paths. Keep generated
evaluation outputs outside the repository.

1. Store each binary image once. Evaluation manifests and documentation point
   to its canonical location.
2. Put executable test fixtures in `fixtures/`.
3. Put only the minimal reproducible detector inputs in `calibration/`.
4. Put externally verified SynthID originals in `synthid/originals/` and keep
   both CSV files synchronized.
5. Keep evaluation outputs outside the repository. Record reproducible
   commands, hashes, and oracle verdicts instead of committing another corpus
   copy. A small curated before-and-after example may live in `docs/images/`
   when it is part of the public documentation.
   Local benchmark manifests may point to these canonical inputs or to an
   untracked corpus; every artifact and reference must carry its SHA-256 digest.
   `evaluations/engine-selection/` is the bounded exception for ORIGINAL study
   inputs: its manifest tracks a prompt-matched OpenAI/Meta content matrix, not
   generated model outputs or watermark-oracle evidence.
6. Put provider captures that no test reads in `captures/<yyyy-mm>/`, with a
   `manifest.csv` row per capture. A capture that is also a fixture is stored
   once, in `fixtures/`, and its manifest row points there. Request a new
   capture only for a service x vendor pair not yet captured, or for a
   service's own model: another version of a known pair adds no provenance
   evidence.
7. Keep third-party fixture license notices in `licenses/`, outside directories
   that tests enumerate as media inputs.
8. Runtime detector assets belong in `src/remove_ai_watermarks/assets/`.
   Unregistered research candidates belong in
   `scripts/assets/visible-mark-candidates/` so they are not shipped in the
   wheel.

The source distribution excludes `data/`; the wheel contains only package
runtime assets.

## Video SynthID oracle manifest

`evaluations/video-synthid-oracle.csv` is the only evidence that the shipped
video removal profile works, so it is also the source of truth for three shipped
defaults: `tests/test_video_invisible.py` asserts that `noise_std`, `long_side`,
and `fps` together match a row this manifest records as certified. Changing one
of those three without adding the row that certifies it fails the suite. Both
historical rows mark `vae` as `unrecorded`: the repository must not infer a model
identity from the current default. Add it to the assertion in the same commit as
the first oracle row that records and verifies one.

| Column | Meaning |
| --- | --- |
| `date`, `source_url`, `source_sha256` | Identify the carrier. |
| `source_width`, `source_height`, `source_fps` | Carrier geometry. Without it the actual downscale factor of a row cannot be recovered later. |
| `duration_seconds`, `source_verdict` | Clip length submitted and the verifier's reading of the untouched carrier. |
| `vae`, `noise_std`, `long_side`, `fps`, `seed` | The run configuration. `unrecorded` means the historical run did not preserve the VAE identity. |
| `output_sha256` | Identifies the exact submitted file. |
| `output_verdict` | One of `detected`, `not_detected`, `indeterminate`, `refused`. |
| `output_verdict_text` | The verifier's wording, verbatim. |
| `output_detected_range` | Time range the verifier reported for the output. |
| `track` | `visual`, `audio`, `both`, or empty. The verifier scores tracks separately and this path copies source audio unchanged. |
| `session_id` | Groups rows submitted in one oracle session, so per-session drift stays visible. |
| `stratum` | Content class of the carrier, for stratified certification. |
| `psnr_db`, `temporal_residual_ratio` | Fidelity measurements. Neither is a watermark verdict. |

Record `indeterminate` when the verifier answers with its unclear state rather
than a negative: an unclear reading logged as `not_detected` is exactly the
silent regression this manifest exists to prevent. Leave a field empty when it
was not recorded, and never backfill it with a plausible value.

`psnr_db` is measured against the already-resized frame and before the encode,
so it excludes the downscale, the decimation, and the codec. Rows stay
comparable to each other only while that definition holds.

The two 2026-07-31 rows predate this schema; their empty fields were never
recorded and are not recoverable from the row.

## Audio SynthID and Wan 2.1 video oracle manifests

`evaluations/audio-synthid-oracle.csv` and `evaluations/video-wan-oracle.csv`
record provider-oracle verdicts strictly over publication-cleared carriers
stored in `data/` (`fixtures/` and `captures/`):

- `tests/test_audio_dac.py` verifies that every `source_path` in
  `audio-synthid-oracle.csv` matches its tracked `source_sha256` and that
  `descript/dac_44khz` (`MODEL_REPOSITORY`, `MODEL_REVISION`,
  `FIXED_QUANTIZERS=3`) has a certified `not_detected` row paired with a
  matched `detected` control on the same carrier (alongside the held-out
  `focalcodec` counterexample that remained `detected`).
- `tests/test_video_wan.py` verifies that every `source_path` in
  `video-wan-oracle.csv` matches its tracked `source_sha256` and that
  `wan21-synthid-experimental` (`WAN21_MODEL_ID`, `WAN21_MODEL_REVISION`,
  `WAN21_SYNTHID_STRENGTH=0.38`, `WAN21_INFERENCE_STEPS=20`,
  `WAN21_FLOW_SHIFT=3.0`, `WAN21_GUIDANCE_SCALE=1.0`, and causal windowing
  `193 / 17`) has certified `not_detected` rows paired with matched `detected`
  controls. Pixel-marked CapCut Seedance captures (`WAN21_SEEDANCE_STRENGTH=0.12`)
  remain untracked in private research storage and are documented in
  [`docs/seedance-video-study.md`](../docs/seedance-video-study.md) until a
  publication-cleared pixel-marked CapCut fixture is added.

