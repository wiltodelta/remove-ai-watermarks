# Watermark benchmark kernel

`scripts/watermark_benchmark.py` is a development-only runner for reproducible
image-, audio-, and video-watermark experiments. It calls the existing local
DWT-DCT and TrustMark detectors plus the revision-pinned AudioSeal and
VideoSeal oracles through thin adapters. It does not add a runtime command,
download a corpus, or contact a provenance oracle or provider API. The
optional TrustMark package can fetch its official Adobe model weights when its
local package cache is incomplete, the audioseal adapter fetches its pinned
checkpoints the same way, and the videoseal adapter downloads one pinned
TorchScript file into a user cache; prepare those dependencies before an
offline run.

The kernel answers three different questions and never merges their answers:

- `detection` records whether one named adapter recognized its own signal.
- `removal` records that a removed-state artifact no longer produced a signal,
  or that the signal remained. It always sets `certifies_erasure` to `false`.
- `fidelity` compares decoded pixels with an explicit reference. It says
  nothing about whether an invisible watermark remains.

In particular, `not_detected` means only that the selected detector did not
recognize its signal. It is not a general `clean` verdict.

## Manifest

Input is strict JSONL, one case per line. Relative paths resolve from the
manifest directory. The runner verifies every named file against its lowercase
SHA-256 digest before invoking any detector, rejects duplicate `case_id` values,
and rejects missing or extra fields.

```json
{"schema_version":1,"case_id":"sample-removed","pair_id":"sample","media_type":"image","adapter":"dwt-dct","arm":"positive","state":"removed","path":"removed.png","sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","reference_path":"clean.png","reference_sha256":"abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789","source_revision":"local-corpus@2026-09-05","transform":{"name":"remove","revision":"remove-ai-watermarks@abc123","parameters":{"method":"auto"}},"seed":7,"expected":"not_detected"}
```

The fixed fields are:

| Field | Contract |
| --- | --- |
| `schema_version` | Integer `1`. |
| `case_id` | Unique case identity. |
| `pair_id` | Groups related clean, marked, attacked, and removed cases. |
| `media_type` | `image`, `audio`, or `video` in schema v1. |
| `adapter` | `dwt-dct` or `trustmark` for `image`; `audioseal` for `audio`; `videoseal` for `video`. The loader rejects a known adapter named against the wrong media type. |
| `arm` | `positive`, `matched_negative`, `wrong_key`, or `hard_negative`. |
| `state` | `clean`, `marked`, `attacked`, `removed`, or `forged`. `forged` names an artifact carrying a watermark with a message different from the adapter oracle's fixed one. Expected detection depends on the adapter: VideoSeal checks the fixed message, while AudioSeal's presence rule can still report `detected` for a foreign message. The study records message accuracy separately. |
| `path`, `sha256` | Artifact path and pinned content digest. |
| `reference_path`, `reference_sha256` | Both strings or both `null`; the fidelity reference. |
| `source_revision` | Corpus, generator, or acquisition revision. |
| `transform` | Exact `name`, `revision`, and JSON `parameters`. |
| `seed` | Integer or `null`; never an implicit random seed. |
| `expected` | `detected`, `not_detected`, or `unresolved`. |

The arms prevent a positive-only benchmark from being mistaken for a detector
evaluation. A useful detector run needs known positives plus matched negatives,
wrong-key cases where the method supports keys, and difficult watermark-free
images. Transformations such as resize, crop, recompression, or removal should
be separate rows with exact parameters, not edits made outside the manifest.

## Run and output

Keep the report outside the repository, and choose a new output path for every
run:

```bash
uv run python scripts/watermark_benchmark.py \
  /path/to/manifest.jsonl \
  --output /path/to/results/run-2026-09-05.jsonl
```

The runner refuses to overwrite an existing report. Every output row repeats
the case identity and records the repository commit, whether tracked files were
dirty, and SHA-256 digests of the benchmark kernel and adapter source. Detector
dependency absence is `unavailable`; an undecodable artifact or adapter
exception is `error`. Neither is counted as `not_detected`.

`detection.adapter_elapsed_ms` measures the wall time of the named detector
adapter call only. Image decoding and fidelity calculation are outside that
interval. The value is `null` when the adapter was unavailable or the artifact
could not be decoded, because no detector call occurred. Preserve the first
call separately when aggregating: it can include lazy imports and model loading,
while subsequent rows describe warm validation cost.

Fidelity is calculated only when a reference is present, both decoded images
are 8-bit, and their dimensions match. Other bit depths remain explicitly
incomparable until their sample-range contract is defined. The result records
8-bit MAE, the fraction of changed pixel positions, and PSNR. Identical decoded
pixels produce `psnr_db: null` with
`psnr_status: unbounded_identical`, never a misleading numeric zero. Missing or
incomparable references remain explicit states rather than fabricated metrics.

Schema v1 deliberately has no aggregate score, confidence interval, automatic
attack generator, removal runner, or provider oracle. Add those only after the
case-level evidence is large enough to define and test the corresponding
statistical or media contract.

## Audio rows

`media_type: audio` rows decode every artifact and reference to mono 16 kHz
float32 through the system `ffmpeg` before any adapter runs, mirroring the
image path where decoding also stays outside the detector interval. Audio rows
therefore require `ffmpeg` on `PATH`; a missing or failing decode is the same
explicit `error` status as an undecodable image.

The `audioseal` adapter reads the shared revision-pinned oracle in
`scripts/audioseal_oracle.py` (Meta AudioSeal, MIT license, Hugging Face
revision and checkpoint SHA-256 pinned and verified at load). Its `detected`
status is the detector's own documented rule: the fraction of per-sample
probabilities above 0.5 reaching 0.5. The `label` is the decoded 16-bit message
as a binary string, so a case whose payload differs from the embedded one is
still `detected` with a different label, never `not_detected`. A `wrong_key`
arm for this adapter is therefore a different-message row, not a clean row.

Audio fidelity compares the decoded sample arrays against the explicit
reference: identical decoded samples produce `snr_db: null` with
`snr_status: unbounded_identical`, never a misleading numeric zero; a silent
reference reports `zero_reference_power`; a length mismatch stays an explicit
`incomparable` state. SNR here is reference signal power over error power, the
sample-domain analogue of the image PSNR contract.

## Audio cohort v1

`scripts/watermark_benchmark_audio_cohort.py` builds the audio analogue of the
synthetic image cohort: three deterministic synthetic carriers (AM tone stack,
smoothed noise, white noise), the pinned AudioSeal message, matched clean
negatives, MP3 128k, gapless M4A AAC 128k, and deterministic 10 dB additive
noise attacks, plus one never-embedded white-noise hard negative. Since v2
the audio cohort also carries the four-state arms: a `removed` row where the
noise transform measurably destroys the mark (the tone stack only - the
synthetic noise carriers survive 10 dB noise, so their noise rows stay open
`attacked` measurements), and `forged` rows embedding a foreign message with
`expected: detected`, because the audioseal rule is watermark presence and
the decoded label is what exposes the foreign payload. 20 cases in total,
everything hashed into a strict manifest validated through the
benchmark loader.

```bash
uv run --extra dev python scripts/watermark_benchmark_audio_cohort.py \
  --output-dir .local-eval/watermark-benchmark-audio-cohort-v1
uv run --extra dev python scripts/watermark_benchmark.py \
  .local-eval/watermark-benchmark-audio-cohort-v1/manifest.jsonl \
  --output .local-eval/audio-cohort-v1-results.jsonl
```

### Development baseline, 2026-09-06

The first local run used kernel
`sha256:2216169ab7c33f7439720f0934e8ac6ba8379e510bfaeabe008f820c7658701d`,
oracle `sha256:bd67e2b1be5c486384ab1f64709f661241451ab1ea7592cc56f0eefb91c415ab`,
and recipe `sha256:f361163b4fbf23b885c026751c544e97be0bf054906c1aad455276992a2768d2`,
producing manifest `sha256:b6c48684f0c85e882159f6d5c4fbeea2f43c3b24d3a2c3a034209c6f531ba73d`
and case rows `sha256:66ac23bd926925dc5baf7216c03f8a7246eeb473315a17701358a3803ab7b0d1`.
Dependencies were audioseal 0.2.0, torch 2.13.0, and ffmpeg 9.0.1.

| State and transform | Cases | Detected | Not detected |
| --- | ---: | ---: | ---: |
| clean matched negatives + hard negative | 4 | 0 | 4 |
| marked, audioseal embed | 3 | 3 | 0 |
| attacked, MP3 128k | 3 | 3 | 0 |
| attacked, AAC 128k (gapless M4A) | 3 | 3 | 0 |
| attacked, 10 dB additive noise | 3 | 2 | 1 |

Every clean and never-embedded negative stayed below the decision rule. All
codecs retained detection with the full message on the tone-stack and
white-noise carriers; the pinkish-noise carrier decoded one flipped payload bit
while remaining detected, matching the experiment's per-carrier bit-accuracy
observation. The 10 dB noise arm destroyed the tone-stack detection but not
white or pinkish noise: carrier-dependent robustness is exactly what the
matched arms exist to expose. Nearest-rank adapter timing: cold first call
405.1 ms, warm median 124.2 ms, warm max 157.4 ms for 8-second clips on this
host. As with the image baselines, these are diagnostic host rows, not a
certified profile.

### Speech extension

`--with-speech` adds fifteen more rows: three system-voice speech carriers
(Samantha, Daniel, Milena) rendered by the local macOS `say` tool - no
network, byte-deterministic per voice on a fixed macOS version - each with a
matched clean negative, a marked positive, and the same three attacks. Every
speech row's transform parameters carry the full TTS provenance (tool, macOS
version, voice, text digest, duration). The builder verifies the configured
voices render distinct bytes and refuses an aliased set, because macOS 26.6.2
renders `Alex` and `Samantha` byte-identically and a name is not a carrier
identity.

The first `--with-speech` run (2026-09-06, 31 cases total, recipe
`sha256:3f291f49ab9ef2499085708fb614e3723ebee6d35f0b509fc36f017ec5a86c3b`,
manifest
`sha256:b9baa79066deb002402cc20166835887d67a5158ff9ce7bbc1963e5d603db3a5`,
case rows
`sha256:abc68590f59fbddab1c086371d2854622b41ea756ba61a7a87b76cb6df64c386`)
kept every synthetic verdict and added: speech clean 0/3 detected, speech
marked 3/3 with the full message, codecs 3/3, and full-clip 10 dB noise 0/3 -
unlike the synthetic white and pinkish carriers, speech has quiet moments
where the noise dominates. Speech marked fidelity sat at 27.9-32.2 dB SNR.

## Synthetic cohort v1

`scripts/watermark_benchmark_cohort.py` builds a reproducible local smoke cohort
without a provider API or provenance oracle. Its default 65 cases cover three
deterministic 512-pixel carriers, SDXL, FLUX, and Stable Diffusion 1.x DWT-DCT
patterns, TrustMark Variant P with schema 1, and JPEG quality 90, 75% resize
round-trip, and 5% crop-per-edge round-trip attacks. Each adapter also receives
a high-frequency hard negative. TrustMark cases include the package remover's
output as a separate `removed` state.

```bash
uv run python scripts/watermark_benchmark_cohort.py \
  --output-dir .local-eval/watermark-benchmark-cohort-v1 \
  --size 512
uv run python scripts/watermark_benchmark.py \
  .local-eval/watermark-benchmark-cohort-v1/manifest.jsonl \
  --output .local-eval/watermark-benchmark-cohort-v1-results.jsonl
```

The builder refuses an existing destination, hashes every generated file, pins
its own recipe digest and dependency versions, and validates the finished
manifest through the benchmark's strict loader. It reads the DWT-DCT patterns
from the production detector so the generated positive cannot silently drift
from the matcher. TrustMark's optional package and model weights must already be
available.

`--size` accepts square carrier sides of at least 256 pixels, divisible by 16.
The size is part of every case identity and transform record, so resource sweeps
at multiple boundaries can be aggregated without merging unlike artifacts.

Attacked and removed rows intentionally use `expected: unresolved`: they are
measurements, not assertions smuggled into fixtures. The cohort is too small
and too synthetic for prevalence, confidence, or release-quality claims. It is
a deterministic first baseline that proves the harness and exposes gross
detector behavior before the real-image cohort below is run.

### Development baseline, 2026-09-05

The first local v1 run used cohort recipe
`sha256:d98fd6402bcc149833005ec03aab77bd0074ee105696f9e1ac5da01ef9c666f7`
and kernel
`sha256:37e552ba53c3d2aff2c5f9e160a33ad47b0d32c877b0a8fbba39c90ca0ab0179`.
Dependencies were invisible-watermark 0.2.0, TrustMark 0.9.1, NumPy 1.26.4,
and Pillow 12.3.0. The tree was based on commit `ad53790` with the cohort and
timing changes still uncommitted, so the content hashes, not that commit alone,
identify the executed code. A second build in a different directory produced a
byte-identical manifest and artifact-hash inventory.

| Adapter and state | Unique artifacts | Detected | Not detected |
| --- | ---: | ---: | ---: |
| DWT-DCT clean | 4 | 0 | 4 |
| DWT-DCT marked | 9 | 6 | 3 |
| DWT-DCT attacked | 27 | 13 | 14 |
| TrustMark clean | 4 | 0 | 4 |
| TrustMark marked | 3 | 2 | 1 |
| TrustMark attacked | 9 | 4 | 5 |
| TrustMark removed | 3 | 0 | 3 |

All three plain-gradient DWT-DCT positives were missed, while all texture and
low-detail positives were recognized. DWT-DCT retained 7/9 positives through
JPEG, 6/9 through resize, and 0/9 through crop. TrustMark retained 2/3 through
JPEG and 1/3 through each resize and crop. Its three remover outputs produced no
recognized signal and measured 46.17-49.24 dB PSNR against their clean
carriers. Those three negatives are a post-removal observation, not proof that
the signal is absent beyond this decoder.

Ten independent sequential benchmark processes reproduced every case status.
Nearest-rank adapter timings were:

| Adapter call | n | p50 | p90 | max |
| --- | ---: | ---: | ---: | ---: |
| DWT-DCT cold, first call per process | 10 | 14.96 ms | 19.86 ms | 27.88 ms |
| DWT-DCT warm | 450 | 0.65 ms | 1.64 ms | 13.53 ms |
| TrustMark cold, first call per process | 10 | 1,989.47 ms | 2,085.62 ms | 2,086.02 ms |
| TrustMark warm | 180 | 17.96 ms | 34.20 ms | 39.95 ms |
| TrustMark warm, removed state only | 30 | 16.80 ms | 17.69 ms | 18.56 ms |

These intervals include only the detector adapter call. They exclude image
decoding, fidelity metrics, removal, and process startup. The roughly two-second
TrustMark cold start matters only when validation is the first TrustMark use in
the process; a workflow that already identified the source pays the warm cost.
The sample is deliberately diagnostic rather than statistically representative.

## Real-image cohort v1

`scripts/watermark_benchmark_real_cohort.py` replaces procedural carriers with
the 38 committed OpenAI/Meta images in the engine-selection content matrix. The
matrix has 19 prompt-matched content strata, one image per provider in every
stratum, and no duplicate file hash. Its images are project-collected model
outputs rather than camera originals; "real-image" here means real, decoded
content carriers rather than generated gradients or checkerboards.

Every source row carries
`reuse_basis: project-maintainer-public-test-clearance`. This records permanent
public test and benchmark permission for those exact committed bytes. It does
not claim that provider terms grant a general dataset license for arbitrary
generations. The builder refuses a source without that explicit basis, a hash
or dimension mismatch, repeated pixels, an incomplete provider pair, or a
duplicate provider/content stratum.

```bash
uv run python scripts/watermark_benchmark_real_cohort.py \
  --output-dir .local-eval/watermark-benchmark-real-v1 \
  --size 512
uv run python scripts/watermark_benchmark.py \
  .local-eval/watermark-benchmark-real-v1/manifest.jsonl \
  --output .local-eval/watermark-benchmark-real-v1-results.jsonl
```

Each source is deterministically center-cropped only if necessary, resized to
the selected square geometry with LANCZOS, converted to RGB, and written as a
PNG reference. At the default size all current sources are already square, so
only the downscale and RGB conversion apply. The source manifest hash, original
file and payload hashes, provider, model, content stratum, native dimensions,
reuse basis, prompt hash, and standardization recipe travel with every case.

The default build has 798 cases: all three production DWT-DCT patterns plus
TrustMark Variant P, each with matched negatives, marked positives, the same
three fixed attacks as the synthetic cohort, and TrustMark remover outputs.
Clean sources are expected negative only for the selected DWT-DCT or TrustMark
adapter; the result must not be generalized into a claim that no other
provenance signal exists. The aggregate report adds provider and content-stratum
tables when this source metadata is present.

The 12 existing SynthID and Content Seal signal carriers are deliberately not
mixed into this matched-negative baseline. They remain suitable for a separate
cross-watermark interference layer, where their original signal and oracle
provenance can be named instead of being mislabeled as globally clean.

### Development baseline, 2026-09-06

Three sequential local runs reproduced all 798 case verdicts with no unstable
case, adapter error, or unavailable result. Both adapters rejected all 38
matched negatives. The carrier set therefore removed the synthetic cohort's
false impression that TrustMark was broadly carrier-fragile, while confirming
and sharpening the DWT-DCT carrier-dependence finding.

| Adapter and state | Unique artifacts | Detected | Not detected |
| --- | ---: | ---: | ---: |
| DWT-DCT marked | 114 | 74 | 40 |
| DWT-DCT JPEG q90 | 114 | 76 | 38 |
| DWT-DCT resize 75% round-trip | 114 | 77 | 37 |
| DWT-DCT crop 5% per edge | 114 | 0 | 114 |
| TrustMark marked | 38 | 38 | 0 |
| TrustMark, all three attacked arms | 114 | 114 | 0 |
| TrustMark removed | 38 | 0 | 38 |

DWT-DCT marked recall was 26/38 for FLUX, 23/38 for Stable Diffusion
1.x, and 25/38 for SDXL. JPEG and resize retained every one of the 74 base
detections, then moved two and three previously missed cases above the decoder
threshold respectively. They did not create new ground-truth marks; this is why
the report presents pairwise transitions rather than describing 76/114 and
77/114 as ordinary attack retention. The crop destroyed every recognized
DWT-DCT signal.

The content strata expose a strong dependence that three procedural carriers
could not measure. Interior low light, person in context, and single portrait
were 0/6 on the un-attacked DWT-DCT embeds, while nine strata were 6/6. Meta
carriers were detected in 42/57 marked cases and their prompt-matched OpenAI
partners in 32/57. Those are descriptive counts over one image per provider
and stratum, not a provider-population estimate.

TrustMark was detected before and after every fixed attack in every stratum.
All 38 remover outputs produced no recognized TrustMark signal, with 41.26 to
49.80 dB PSNR and 45.27 dB median PSNR against the standardized carriers. This
remains a decoder-scoped post-removal observation, not proof of erasure against
another implementation.

The run was based on commit `f61a7ec` with tracked benchmark changes still
uncommitted. The exact inputs and implementations are therefore bound by hash:

- source manifest: `sha256:aa37aa97ed3b32a59242fd506fa3cee545df615f98690d373735929a9d7d0089`;
- content-manifest validator: `sha256:72ddb2e489d5a7f92cd948b880a0fd308d4083ed86d68fc49897d646aceb7ec5`;
- real-cohort recipe: `sha256:f3851dae5d4eee69510c3dc5b5b15b5116720d944cfeed631894e635c2a5e9c9`;
- shared embed/attack helper: `sha256:c78342f0d125d6636259b641e67e71b72b577455057e568052d2a0a4df00c9dd`;
- benchmark kernel: `sha256:a96874a41fa0a38cd29d957c6a69683de887e942b480ce7668090b0f1a9e3329`;
- generated manifest: `sha256:77dc6094fcfbbaa6fdb9a15d96dee6fb3d4349829b2b1d7cf5f006aa7fb0ae8a`;
- case-level results: `sha256:bb483be015c012fd28d0979510244c834475a23bbc215226c998fada73485ed3`,
  `sha256:35d735de9654a36d6d2a6e096f7a2c3f8a70b1b22ead4f6f414d6c1494e53b2a`, and
  `sha256:52a9a2bb78dcc9f40609b5ca13812470d21477e5e553892c2bbef4b803fee433`;
- aggregate report: `sha256:4496edf231a20cc7977f003c30313af5acbe8c99075e22d2cb2eeec5e104f3eb`.

The local timing rows were host-contended and are not used for a new latency
claim. The controlled resource profile below remains the performance evidence.

## Video rows

`media_type: video` rows decode every artifact and reference to float RGB
frames in [0, 1] as (T, H, W, 3) through the system `ffmpeg`, staying outside
the detector interval like the image and audio decodes. Video fidelity
compares decoded frames against the explicit reference clip: mean PSNR over
frames (`mean_psnr_db`, reported separately from image `psnr_db`), a
changed-frame fraction, `unbounded_identical` for bit-identical
pixel sequences, and an explicit `shape_mismatch` state when frame counts or
geometry differ - which is why geometry-changing attack arms carry no
reference instead of a fabricated comparison.

The `videoseal` adapter loads Meta's standalone TorchScript release of
VideoSeal (MIT license) from `scripts/videoseal_oracle.py` - one pinned file,
no `videoseal` package, whose wheel pulls a research dependency chain without
macOS arm64 wheels and whose loader resolves cards relative to the working
directory. The upstream video evaluation ignores the model's detection head
(it prepends a constant) and scores the aggregated 256-bit message decode, so
the adapter's `detected` is its own explicit rule: bit accuracy of the
averaged decode against the oracle's fixed message at or above 0.9. The
`label` is the decoded message as 64 hex characters. Because the rule is
message accuracy, a different-message row reads `not_detected` - the matched
verifier rejects the foreign payload - unlike the audioseal presence rule
above. Every videoseal detection record
carries a `temporal` block - frame count, min/mean/max and the full per-frame
bit-accuracy series - and the aggregation is computed in the oracle from raw
per-frame logits, because the TorchScript build ignores the aggregation
argument of `detect_video_and_aggregate`. The aggregation matrix itself
(avg versus norm-weighted) is evaluated in
[VideoSeal temporal evaluation](videoseal-temporal-evaluation.md), not in the
kernel, which keeps plain avg as its canonical rule.

## Video cohort v1

`scripts/watermark_benchmark_video_cohort.py` builds the video analogue of
the other cohorts: two deterministic synthetic 64-frame 256x256 clips (moving
gradient, moving texture), the pinned 256-bit message, matched clean
negatives, one never-embedded hard negative, and three attacks - a realistic
H.264 re-encode at crf 23, a 75 percent downscale, and a frame-rate halving
that drops half the keyframes. Since v2 the video cohort also carries the
four-state arms: `removed` rows reusing the crf 23 re-encode that measurably
takes both synthetic carriers below the rule, and `forged` rows embedding a
foreign message with `expected: not_detected`, because the videoseal rule is
message accuracy and the matched verifier rejects a foreign payload. 15
cases in total.

```bash
uv run --extra dev python scripts/watermark_benchmark_video_cohort.py \
  --output-dir .local-eval/watermark-benchmark-video-cohort-v1
uv run --extra dev python scripts/watermark_benchmark.py \
  .local-eval/watermark-benchmark-video-cohort-v1/manifest.jsonl \
  --output .local-eval/video-cohort-v1-results.jsonl
```

### Development baseline, 2026-09-07

The first local run used kernel
`sha256:7a3906cebc3c7cacbac7e204c014d1d6281615e7acbbc5b3a8fcecc939ca68c7`,
oracle `sha256:6c889bf399e8e3bada439eee4a1e9c0b04177be78dfee756d5daa79e73db71b3`,
recipe `sha256:dd06ae1c40ef7d7f6fd2b258b20444a3250254a5ef258542a4eaf43cc501342b`,
manifest
`sha256:bb6a7dda6b3897794f2fbfe47291329bb7b9ff7ee57c0634ebb386e58acd5e7a`,
and case rows
`sha256:5793b8ba33c55e135e5bef43f33fd2699bb57bae6ca8ead38b1dab0123d1dd12`.
The checkpoint is the TorchScript build `y_256b_img.jit`
`sha256:5c7a4581c36fc6090aafdcfb3999123bae5172a4847f22e2da4e7fd1a39d1e1b`
pinned to upstream commit `870ca7f`.

| State and transform | Cases | Detected | Not detected |
| --- | ---: | ---: | ---: |
| clean matched negatives + hard negative | 3 | 0 | 3 |
| marked, videoseal embed (crf 8 artifact) | 2 | 2 | 0 |
| attacked, H.264 crf 23 | 2 | 0 | 2 |
| attacked, 75 percent downscale + crf 18 | 2 | 0 | 2 |
| attacked, frame rate halved | 2 | 2 | 0 |

Direct oracle readings fill in what the threshold hides: the marked clips
decode at 0.992-0.996 bit accuracy (48.5 and 41.4 dB mean PSNR against their
clean carriers), the crf 23 re-encode collapses to 0.56-0.57, the downscale
to 0.69, and the halved frame rate keeps 0.996. Two cautions travel with
these numbers: the watermark survives H.264 only at generous quality - a
realistic crf 23 re-encode erases it on these smooth synthetic carriers,
while a high-bitrate uniform-noise carrier survived crf 18 in the isolated
probe, so retention is carrier- and quality-dependent - and dropping half
the frames does not touch the message because the secret repeats across
keyframes. These are first-baseline observations on synthetic carriers, not
robustness curves; a sweep across crf values and real-content carriers is
the natural follow-up.

### Four-state baseline, 2026-09-07 (v2 cohorts)

The v2 runs used kernel
`sha256:22f2c1dd5c3b7aeb96090c4c578c6d1841ee62d90958dd1c5c7ad7db71835eec`
with audio recipe
`sha256:31c021144212a43565fb3746fadc835bfc37ea8ac98998fad9bba9e72410e82c`
(manifest
`sha256:43c7f77f1d4537c00d4b0dd5128e9becc6c860cdd196a7ba2f7541f4276784ec`,
case rows
`sha256:c73d7dddfdb90262e96be4d67ec1567d9046812661928cc816cb9f97f4a01a7e`)
and video recipe
`sha256:8f25929304ef4f94721667968ee4bb89ebf584a2108de7e02db36c855dc2ddc8`
(manifest
`sha256:b5659848c0d05e205f80ab36480e70fdf7e5f9df4d8ada6ffadec42c97b88998`,
case rows
`sha256:c9ad56848bb7efe200aaf1189a8eb03f7571567e6702edd195a62f812fa05815`).
Every v1 verdict was unchanged; the new arms read: audio removed 0/1
detected, audio forged 3/3 detected under the presence rule, video removed
0/2, video forged 0/2 under the message rule - zero expected mismatches.
The mechanism underneath these rows, including double embedding, is measured
in [Watermark forgery study](watermark-forgery-study.md).

## Aggregate repeated runs

`scripts/watermark_benchmark_report.py` turns one or more result JSONL files
into a standalone Markdown report. Treat each input file as one independent
process run: the first measured call for each adapter in that file is cold, and
the remaining measured calls are warm.

```bash
uv run python scripts/watermark_benchmark_report.py \
  .local-eval/results/run-01.jsonl \
  .local-eval/results/run-02.jsonl \
  --output .local-eval/results/report.md
```

The report retains raw observation counts but deduplicates detector summaries
by artifact SHA-256 within each adapter and state. It reports state and
transform breakdowns, post-removal observations, nearest-rank timing, fidelity,
expected-result mismatches, unstable repeated verdicts, errors, and unavailable
adapters as separate evidence. A repeated `case_id` whose artifact, transform,
or other case identity changes is rejected rather than merged. Input file
digests in the final report bind every aggregate to its exact case-level source.
The runner decodes an artifact used by multiple cases only once per process and
passes the decoded pixels to both local adapters. Synthetic recipes record
the seed actually passed to generation; a manifest seed is not a substitute
for that generator input. Video observations must decode the saved artifact
whose digest the row records, including controls and double embeds. Adapter
timing does not include file decoding.

## Profile process cost

`scripts/watermark_benchmark_resources.py` runs each manifest in fresh Python
processes. It records parent-observed wall time and the child's absolute peak
RSS, while preserving the ordinary case-level JSONL outputs for the detector
report above.

```bash
uv run python scripts/watermark_benchmark_resources.py \
  .local-eval/cohort-256/manifest.jsonl \
  .local-eval/cohort-512/manifest.jsonl \
  .local-eval/cohort-1024/manifest.jsonl \
  --repeat 3 \
  --output-dir .local-eval/resource-profile
```

Wall time includes interpreter startup, imports, image decoding, detector calls,
fidelity metrics, and result writing. Peak RSS is the process's absolute
high-water mark, not incremental memory caused only by validation. Run DWT-DCT
and TrustMark manifests separately to avoid attributing one adapter's model load
to the other. The profiler uses only local adapters and never calls an oracle or
provider API.

### Development resource profile, 2026-09-06

Ten sequential fresh-process runs per size were measured on macOS arm64 with
Python 3.12.13. DWT-DCT manifests contained 46 cases; TrustMark manifests
contained 19. Every status count was stable across all ten repetitions.

| Adapter | Max geometry | Cases | Process wall p50 | p90 | Absolute peak RSS p50 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DWT-DCT | 256x256 | 46 | 263.07 ms | 322.61 ms | 74.66 MiB | 75.31 MiB |
| DWT-DCT | 512x512 | 46 | 482.67 ms | 498.07 ms | 90.56 MiB | 91.02 MiB |
| DWT-DCT | 1024x1024 | 46 | 1,360.10 ms | 1,389.40 ms | 159.58 MiB | 160.00 MiB |
| TrustMark | 256x256 | 19 | 3,191.18 ms | 6,074.68 ms | 741.11 MiB | 743.66 MiB |
| TrustMark | 512x512 | 19 | 9,172.26 ms | 15,430.20 ms | 752.92 MiB | 758.17 MiB |
| TrustMark | 1024x1024 | 19 | 7,301.26 ms | 9,797.40 ms | 848.91 MiB | 852.92 MiB |

The TrustMark wall-time variance, including one 18.34-second 512 px maximum,
did not coincide with comparable memory movement. The non-monotonic 512/1024
times prove that this host was contended, so the TrustMark wall rows are a
diagnostic range and cannot establish size scaling or throughput. Repeat them
on an idle host before making a performance claim. Peak RSS stayed much more
stable: its nearly flat 256/512 values establish the fixed model cost, while
1024 px adds about 100 MiB of high-water memory.

These whole-process rows are deliberately not divided by the case count. A
fresh-process average would hide the fixed model load and is not the cost of one
post-removal check. The current three-run real-image cohort measured warm p50
adapter calls of 0.57 ms for DWT-DCT and 29.11 ms for TrustMark at 512 px,
excluding file decoding. A workflow that identified the source before removal
already holds TrustMark's singleton decoder, so validation reuses that fixed
model cost. Starting a new process or loading TrustMark only after removal is
the expensive design and should be avoided.

The exact profiler source was
`sha256:caf9674dffceddebb8cbf26067581250334fbc11e0fbcb567e8dd45555357b54`.
The local DWT-DCT and TrustMark resource JSONL files were respectively
`sha256:b7ceb012db5c2f549720c37d6234526396c7120fb68e040705322be35dfd6743`
and
`sha256:f3639f46a9257a3161ed0197016c089ca5c084b2de2e07cb4b65fc774e20544d`.
They remain untracked because they contain host-specific paths and timings;
these hashes bind the summarized numbers to the exact local evidence.
