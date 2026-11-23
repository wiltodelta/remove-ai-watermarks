# Python API

Use the high level API for normal application integration. Low level detector
and pipeline modules are intended for maintainers and specialized workflows.

Dependency groups are identical for the CLI and Python API. The default install
covers metadata extraction, normalization, verdict logic, and stripping.
Array/pixel APIs use `pixels`; visible removal uses `visible`; DWT-DCT detection
uses `detect`; lightweight source-pipeline classification uses `source-classify`; invisible image removal uses `qwen-zimage` and an NVIDIA GPU; and
visible video processing uses `video`. The video-pixel SynthID-removal profile is
a separate VAE path that still runs on CPU and combines `video` and `diffusion`. Add `heif`
independently when path-based pixel APIs must decode HEIC, HEIF, or AVIF. See
the complete [feature-extra matrix](installation.md#feature-extras).

## Classify a source/export pipeline

Install `remove-ai-watermarks[source-classify]`, then call the explicit API:

```python
import remove_ai_watermarks as raiw

result = raiw.classify_source("metadata-stripped.png")
print(result.label)                  # one of 10 providers or "unknown"
print(result.version_subfamily)      # Stage-2 model subfamily or None
print(result.synthetic_detected)     # bool, stays True on transcoded AI when label is "unknown"
print(result.synthetic_probability)  # calibrated [0.0, 1.0] synthetic probability
print(result.reason)                 # "classified" | "abstained" | "feature_unavailable"
print(result.watermark_truth)        # always "unknown"
```

This signal classifies a complete image-generation and export
pipeline across 10 provider families (`openai`, `google`, `bytedance`, `flux`,
`stable_diffusion`, `meta`, `midjourney`, `ideogram`, `firefly`, `xai`); it is
not a SynthID detector and does not establish watermark
presence or absence. See [source-pipeline classification](source-classify.md)
for the result fields, model pin, evaluation boundary, and offline weights
override.

## Remove visible marks

Install `remove-ai-watermarks[visible]` before using the visible-removal API.

```python
import remove_ai_watermarks as raiw

result, removed = raiw.remove_visible(
    "watermarked.png",
    "clean.png",
)
```

The function returns:

- the result as a BGR NumPy array;
- a list of labels that were removed.

An empty `removed` list means that no registered visible mark was selected. It
does not prove the image has no metadata or invisible watermark.

Use the detailed form when the caller needs to distinguish a validated fill
from a residual or a validator failure:

```python
report = raiw.remove_visible_detailed("watermarked.png", "clean.png")
print(report.status)  # "no_watermark" | "cleaned" | "partial" | "unvalidated"
for mark in report.marks:
    print(mark.label, mark.status, mark.confidence_before, mark.confidence_after)
```

Each selected mark is localized and filled once. The same detector then checks
the result without changing the mask or running another fill. `partial` means
that detector still accepts a region overlapping the mask that was filled;
`unvalidated` means the check failed, not that the mark remains. The legacy
`remove_visible` tuple API performs the same check but intentionally discards
the detailed status.

### Path input

For a path input, `remove_visible` and `remove_visible_detailed`:

- reads metadata provenance for the default `auto` sensitivity;
- preserves a separate alpha channel;
- writes the output when an output path is supplied;
- strips AI metadata from the written output by default, and raises
  `MetadataStripIncomplete` without publishing it when a marker survives;
- preserves the original bytes for a same-format no-op copy.

```python
result, removed = raiw.remove_visible(
    "watermarked.png",
    "clean.png",
    sensitivity="auto",
    backend="auto",
    strip_metadata=True,
)
```

Set `write_noop=False` if the output path must remain untouched when nothing is
removed:

```python
result, removed = raiw.remove_visible(
    "input.png",
    "clean.png",
    write_noop=False,
)
```

### Array input

Array inputs are BGR NumPy arrays. They do not carry file metadata or a separate
alpha plane:

```python
import cv2
import remove_ai_watermarks as raiw

image = cv2.imread("input.png")
result, removed = raiw.remove_visible(image, backend="cv2")
```

## Run the full pipeline

`remove_all` is the library form of the `all` command: visible marks, then the
invisible watermark, then AI metadata. Stages are chained through a file in the
system temp directory, so a partial result never appears at the output path.

```python
import remove_ai_watermarks as raiw

result = raiw.remove_all("input.png", "clean.png")   # -> RemoveAllResult
print(result.output)          # the path written
print(result.visible_label)   # the marks removed, or None
print(result.visible_status)  # "no_watermark" | "cleaned" | "partial" | "unvalidated"
for mark in result.visible_marks:
    print(mark.label, mark.status, mark.confidence_before, mark.confidence_after)
print(result.invisible)       # "removed" | "no-signal" | "unavailable"
```

The visible status and per-mark records come from the same detector/fill/check
pass as `remove_visible_detailed`; `remove_all` does not run another detector or
fill. `partial` and `unvalidated` are quality statuses, not automatic retries.

`invisible` is also important. `"unavailable"` means the GPU extra is not
installed, so the output *looks* processed but still carries the watermark;
`"no-signal"` means the scrub was deliberately skipped because nothing was
locally detectable, which is a successful run.

Pass `InvisibleOptions` to tune the diffusion stage, and `engine` to reuse one
loaded model across many calls:

```python
from remove_ai_watermarks import InvisibleOptions

raiw.remove_all(
    "input.png",
    "clean.png",
    invisible=InvisibleOptions(strength=0.35),
    force=True,
    progress=print,
)
```

`vendor="meta"` names a strength cohort explicitly (the measured Content Seal
floor) for a stripped file whose provenance no longer carries the AI IPTC tag,
and implies the scrub runs:

```python
raiw.remove_all(
    "muse_output.webp",
    "clean.png",
    invisible=InvisibleOptions(vendor="meta"),
)
```

`InvisibleOptions` carries only what `InvisibleEngine` itself takes, and uses the
engine's own parameter names and defaults. `force`, which decides whether the
engine runs at all, is a parameter of `remove_all` and `remove_batch` alongside
`backend` and `sensitivity`.

Specialized callers can inspect or apply the clean-room local Paint pixel
operation directly. This is deliberately separate from the high-level diffusion
route. Passing the UUID signed in pristine C2PA makes agreement with that UUID a
write precondition:

```python
from remove_ai_watermarks.microsoft_invismark import (
    detect_local_invismark,
    disrupt_local_invismark,
    invismark_id_from_c2pa,
)

expected = invismark_id_from_c2pa("paint.png")
detection = detect_local_invismark("paint.png")
if expected is not None and detection is not None:
    result = disrupt_local_invismark(
        "paint.png",
        "clean.png",
        expected_watermark_id=expected,
    )
```

The direct call needs the `pixels` extra. It writes atomically only after the
encoded output no longer decodes as a valid local payload. The result does not
claim acceptance by a proprietary Microsoft detector.

Callers with the `trustmark` extra can similarly inspect or apply Adobe
TrustMark's variant-matched neural remover (`"P"`, `"Q"`, or `"C"`) directly:

```python
from remove_ai_watermarks.trustmark_detector import (
    detect_trustmark_detailed,
    remove_trustmark,
)

detection = detect_trustmark_detailed("adobe.jpg", variant="P")
if detection is not None:
    removed = remove_trustmark("adobe.jpg", "clean.jpg", variant=detection.variant)
```

`remove_trustmark` writes atomically only when the input passes the durable
re-encode gate for `variant` and the written file no longer passes that gate.

The complete field set, with the shipped defaults:

| Field | Default | Effect |
| --- | --- | --- |
| `strength` | `None` | Denoising strength. `None` resolves per profile and vendor cohort. |
| `pipeline` | `"qwen-zimage"` | Profile: `qwen-zimage`, `qwen21-zimage` (Qwen Research License: non-commercial/personal use only), `sdxl-zimage`, `chroma-zimage`, or `auto`. |
| `vendor` | `None` | Strength cohort, and an assertion that the watermark is present. |
| `seed` | `None` | `None` uses the certified seed 0. Removal near the floor is seed-dependent. |
| `hf_token` | `None` | Hugging Face token for the model download; the environment and a local `.env` are read when unset. |
| `humanize` | `0.0` | Analog film grain. 0 is off; 2.0-6.0 is the useful band. |
| `unsharp` | `0.0` | Unsharp-mask sharpening. 0 is off. |
| `adaptive_polish` | `None` | `None` lets the profile decide (off for `qwen-zimage` and `qwen21-zimage`, on for `sdxl-zimage`); `True`/`False` override it. |
| `max_resolution` | `0` | Cap the long side before diffusion. 0 keeps native geometry and the most detail. |
| `controlnet_conditioning_scale` | `1.0` | Canny ControlNet conditioning on the global stage. Higher stays closer to the original structure and text. |
| `cpu_offload` | `False` | Stream model components between CPU and GPU. Lower CUDA memory, slower. |
| `tile` | `False` | Regenerate large inputs in overlapping tiles instead of downscaling. |
| `tile_size` | `1024` | Tile side in pixels when `tile` is set. |
| `tile_overlap` | `128` | Tile overlap in pixels. More overlap hides seams and costs time. |
| `text_manifest` | `None` | Path to an operator-verified text manifest; see the text-restoration section. |
| `fidelity_anchor` | `False` | Opt into the fidelity-anchor pass. |

`tests/test_docs_cover_the_public_surface.py` fails when a field is added here
and not to this table, because this seam had already drifted by eight fields
before anything checked it.

If AI metadata survives the strip, `remove_all` and a path-input `remove_visible`
raise `MetadataStripIncomplete` **before** publishing anything: an AI-readable
output is worse than no output.

`remove_batch` runs one mode over a directory and never lets a single bad file
end the run:

```python
summary = raiw.remove_batch("in_dir", "out_dir", mode="visible")   # -> BatchSummary
print(summary.processed, summary.failed, summary.errors)
print(summary.invisible_unavailable)   # outputs that still carry the watermark
for item in summary.items:             # -> BatchItemResult
    print(item.source, item.output, item.visible_status, item.error)
    for mark in item.visible_marks:
        print(mark.label, mark.status)
```

`mode` is `all`, `visible`, `invisible`, or `metadata`. Pass a constructed
`InvisibleEngine` as `engine` to load the model once for the whole directory.
Visible and all items contain the aggregate visible status and lightweight
per-mark records; image arrays are not retained in the batch summary. Other
modes leave those two fields unset and empty. The original `BatchSummary`
aggregate fields and `RemoveAllResult.visible_label` remain available for
existing callers.

## Inspect provenance

The default installation evaluates file metadata. Add `visible`, `detect`, or
`trustmark` to enable the corresponding optional pixel signals.

Get the vendor keys used by visible removal:

```python
import remove_ai_watermarks as raiw

vendors = raiw.visible_provenance("input.png")
```

A known TC260 `ContentProducer` maps to one registered product. A missing or
unmapped producer, and the shared standalone IPTC AI tag, return no vendor key;
neither standard identifies a manufacturer by itself.

Get the full provenance report:

```python
from pathlib import Path

from remove_ai_watermarks.identify import identify

report = identify(Path("input.png"))
print(report.platform)
print(report.signals)
print(report.c2pa_validation)
```

`c2pa_validation`, when present, reports `integrity`, `signature`,
`signer_trust`, and `signer_validity` independently, plus the reader status
codes. A valid hash and signature is a high-confidence signed claim; an
unanchored or expired signer appears in `caveats` and in these fields, not as a
lower confidence, because the reader ships no trust anchors to check against and
`signer_trust` is therefore a missing input rather than a finding. A hash or
signature failure, or a revoked signing credential, does not confirm the claimed
platform or AI origin.

A consumer must read `integrity_clashes`. When a credential fails validation,
`is_ai_generated` becomes `None`, because a claim that cannot be tied to these
bytes cannot establish origin -- the manifest may have been transplanted from a
real AI image onto anything -- and the failure is reported in
`integrity_clashes` instead. That is a different question from whether an AI
watermark is physically present in the pixels, which is what
`has_invisible_target` answers, and it stays fail-safe `True` on the same file.
Reading only `is_ai_generated` turns a broken vendor manifest into silence.

`c2pa_validation["state"]` is the reader's own aggregate and is carried for
diagnostics only; no verdict is derived from it, because it collapses a
transplanted manifest and a merely expired certificate into one `Invalid`, and
its `Trusted` level depends on anchors no default installation has. Fallback parsing reports unknown validation
dimensions, while a raw marker in an unsupported or malformed container can
leave `c2pa_validation` as `None`.

Use `check_visible=False` and `check_invisible=False` for metadata-only
inspection through the compatible path-based API:

```python
report = identify(
    Path("input.png"),
    check_visible=False,
    check_invisible=False,
)
```

Extraction and detection are also available as separate steps. This is useful
when a file-reading worker collects the metadata once and another component
evaluates the resulting evidence:

```python
from remove_ai_watermarks.identify import (
    extract_provenance_evidence,
    identify_from_evidence,
)

evidence = extract_provenance_evidence(Path("input.png"))
report = identify_from_evidence(evidence)
```

### Collect once, judge elsewhere

`collect_metadata_record` splits the two halves apart: it is the only step that
touches the file, and it returns a JSON-serializable record the verdict can be
built from on another machine, in another process, or later.

```python
import json

from remove_ai_watermarks.identify import identify_metadata_record
from remove_ai_watermarks.metadata_record import collect_metadata_record

record = collect_metadata_record(Path("input.png"), schema_version=1)   # reads the file
blob = json.dumps(record)                             # ship it anywhere

report = identify_metadata_record(json.loads(blob), path=Path("input.png"))  # reads nothing
payload = report.to_dict(schema_version=1)            # versioned JSON contract
```

The collection record has `record_type="provenance_metadata"`,
`schema_version=1`, and a `status`. A vanished or unreadable source produces an
`error` record with structured `issues`; `identify_metadata_record` rejects that
record instead of turning a collection failure into an unknown-image verdict.
Unknown schema versions, non-integer aliases, and native records without a
`complete` collection status are rejected explicitly.

The verdict is the same one `identify(path, check_visible=False,
check_invisible=False)` returns for that file. That equality is the record's whole
contract and is verified over the tracked provenance fixtures and a separate local
evaluation corpus. `ProvenanceReport.to_dict()` is the stable service boundary: it
adds a `schema_version`, contains only JSON-safe values, and deliberately omits the
local source path.

Package and transport versions evolve independently. Long-lived consumers should
request the schema they implement, as above, instead of assuming the installed
package's latest schema. Within schema 1, existing fields, types, meanings,
`signals[].name` values, and `watermarks[]` labels remain compatible; releases may
add fields that consumers must ignore. A breaking change requires a new schema while
the schema 1 serializer remains available for rolling upgrades. Asking a release for
an unsupported schema raises `ValueError` rather than silently returning another
shape.

Microsoft InvisMark declarations emit both the compatible generic `soft_binding`
signal and the additive `invismark` signal. Consumers should use `invismark` to route
the image through pixel removal; the generic signal also covers content fingerprints
that must not trigger regeneration.

A record carries metadata regions, not the primary coded-pixel stream: marker
segments before the JPEG scan, every PNG chunk except `IDAT`, RIFF chunks except the
coded image, the ISOBMFF provenance boxes, the container's trailer, the parsed EXIF tags the
verdict reads by name, PIL's info mapping, and the C2PA manifest store. Record size
is bounded by those metadata regions and trailers; images with large embedded
manifests naturally produce larger records.

The `path` argument is metadata: it labels the report and is never opened by
either function, so a record collected elsewhere can be judged against a path that
does not exist locally.

If metadata was collected by another component instead, normalize its nested
record the same way:

```python
from remove_ai_watermarks.identify import (
    evidence_from_metadata_record,
    identify_from_evidence,
)

record = {
    "pil": {"info:parameters": "Steps: 20, Sampler: Euler"},
    "exif": {"0th": {"Software": "Stable Diffusion"}},
}
evidence = evidence_from_metadata_record(record, path=Path("input.png"))
report = identify_from_evidence(evidence)
```

Unversioned third-party records are normalized recursively for compatibility. The
normalizer preserves text and byte values. It also decodes
strings prefixed with `hex:` and fields named `base64` or ending in
`_base64`. Diagnostic, transport, timing, hash, provenance-result, and pixel-result
subtrees are ignored because they describe the collector or a derived result rather
than the source file. A versioned portable record is stricter still: only
`metadata_base64`, `tail_base64`, `pil`, `exif`, and `c2pa_store` are accepted as
source evidence. Other `record_type` values are rejected, so do not pass a broad
forensic inspection record to this API. Pass a C2PA manifest-store dictionary in
`record["c2pa_store"]`, or through the explicit
`c2pa_manifest_store` argument.

### Broad metadata inspection

`collect_forensic_metadata` provides the wide metadata-only record used by forensic
inspection and migration adapters. It preserves hashes and timestamps, full EXIF and
IPTC, C2PA, container inventories, bounded raw metadata payloads, and embedded
thumbnail forensics. It does not calculate a provenance verdict or pixel statistics.

```python
from remove_ai_watermarks.forensic_metadata import collect_forensic_metadata

record = collect_forensic_metadata(Path("input.png"), schema_version=1)
assert record["record_type"] == "forensic_metadata"
```

This record is intentionally not accepted by `identify_metadata_record`. Collect the
small strict provenance record separately and publish the resulting
`ProvenanceReport.to_dict()` as the detector contract.

### Pixel evidence

`extract_pixel_evidence` decodes once and calculates the DCT, FFT, residual, ELA,
gradient, and color families. Its versioned `to_dict()` result has a semantic
`status`: `complete`, `partial` when an individual family failed, or `error` when the
source could not be decoded. Transported errors contain only the exception class, so
local paths stay in the caller's logs rather than crossing the service boundary.

```python
from remove_ai_watermarks.pixel_evidence import extract_pixel_evidence

pixels = extract_pixel_evidence(Path("input.png"), artifacts=False, timings=True)
payload = pixels.to_dict(schema_version=1)
```

Timings and spatial artifacts are opt-in. Artifacts include image-identifying data
such as a thumbnail and perceptual hash; aggregate feature families do not.

`identify_from_evidence` does not reopen the source file by default: it evaluates
metadata only, and the pixel-backed checks remain in the path-based `identify`
call: registered visible marks and open invisible-watermark decoders.

Pass `image_path` together with `check_visible` or `check_invisible` to add those
pixel detectors on top of the SAME evidence. That is how a caller asking one file
two provenance questions — which vendor is confirmed, and is there an invisible
target — pays for the metadata extraction once:

```python
from remove_ai_watermarks.identify import extract_provenance_evidence, identify_from_evidence

evidence = extract_provenance_evidence(source)
metadata_only = identify_from_evidence(evidence)
with_pixels = identify_from_evidence(evidence, image_path=source, check_invisible=True)
```

## Strip metadata

```python
from pathlib import Path

from remove_ai_watermarks.metadata import has_ai_metadata, strip_and_verify

source = Path("input.png")
output = Path("clean.png")

if has_ai_metadata(source):
    output_path, surviving_markers = strip_and_verify(source, output)
    if surviving_markers:
        raise RuntimeError(
            f"AI metadata remains in {output_path}: {surviving_markers}"
        )
```

Use `strip_and_verify` when your application reports that stripping succeeded.
It checks the written output and returns `(output_path, surviving_markers)`.
When the first strip leaves markers in a malformed but raster-decodable image,
it normalizes the container through `image_io` and checks again. That recovery
path preserves the pixels but drops standard metadata. Treat a nonempty
`surviving_markers` mapping as a failure. A source that is not a readable image,
video or audio container raises `ValueError` before anything is written;
`metadata.unreadable_reason(path)` returns the same reason without stripping.

`remove_ai_metadata` is the lower level fail-safe transformer. It may copy an
undecodable input through unchanged, so its return alone must not be presented
as proof that metadata was removed.

Native Suno and Suno Studio audio creation comments appear as
`get_ai_metadata(path)["audio_generator"]` when ffprobe is available and the
container has audio without an actual video stream. This is an unsigned
creation declaration. Metadata removal preserves the encoded streams. The
ffmpeg route drops all container tags and chapters, including ordinary ones;
ISOBMFF uses targeted patches. This does
not remove a waveform watermark or a fingerprint stored elsewhere.

## Identify and process audio

```python
import remove_ai_watermarks as raiw

report = raiw.identify_audio("input.wav", scheme="auto")
meta = raiw.inspect_audio_metadata("input.mp3")
meta_clean = raiw.remove_audio_metadata("input.mp3", "clean.mp3")
result = raiw.remove_audio_invisible("input.wav", "processed.flac", scheme="auto")
complete = raiw.remove_audio_all("input.m4a", "clean.m4a", include_invisible=True)
batch = raiw.remove_audio_batch("tracks", "tracks_clean", mode="all", include_invisible=True)
```

Install `audio-detect` for compatible open AudioSeal checks, or `audio` for
explicit processing without verification. Both need ffmpeg and ffprobe.
`identify_audio(source, check_invisible=True, stream=0, scheme="audioseal")` selects a zero-based
audio-track ordinal. `AudioProvenanceReport` includes `sha256`, stream geometry,
`metadata_markers`, per-channel `detections`, `watermark_status`, `caveats`, and
optional `recommended_recipe`.
Each `AudioDetection` holds whole-clip and native-frame-bounded window readings;
multichannel inputs also get a mono-downmix check. For validated 24 and 48 kHz
sources, AudioSeal checks both 16 kHz and the original sample clock, with
`AudioDetection.decoder_sample_rate` identifying each receiver. This field is
also set for the native OpenVoice default-signature receiver and is 44100 for
`scheme="silentcipher"`, regardless
of the source rate; it is `None` for standard
synchronized WavMark and Perth. Other AudioSeal native
clocks are outside calibrated coverage
and are not added automatically. A complete negative requires every requested channel
and clock to be negative. Short trailing windows reuse preceding context,
including AudioSeal windows. Status is `detected`,
`not_detected`, or `unverified`, scoped to the pinned open model. Missing
explicitly requested decoders, failed checks, and bounded whole-clip checks never become negatives.
Pass `scheme="auto"` to scan every *installed* non-experimental compatible public decoder (`audioseal`, `wavmark`, `perth`) without failing when an optional decoder extra is absent or a clip is shorter than WavMark's 1-second minimum window, `scheme="wavmark"` for the pinned public WavMark model, `scheme="perth"` for public Perth v1, `scheme="silentcipher"` for the experimental
SilentCipher-44k receiver, or `scheme="all"`
for all named packaged decoders (including the experimental SilentCipher-44k receiver). The default on `identify_audio` and `remove_audio_invisible` is `"audioseal"`; `remove_audio_all` and `remove_audio_batch` default to `scheme="auto"`. WavMark also
checks the published OpenVoice default message at fixed native offsets. Those
readings have `framing="openvoice-default"`, fragment matches, and adjusted
binomial-tail diagnostics; they do not identify a provider or cover custom
messages and shifted/resampled framing. Missing requested
models keep a combined result unverified. WavMark windows include valid
endpoints and exactly one second; a shorter source stays unverified. Its
readings hold recovered payload bits and start offsets at the model sample rate.
SilentCipher-44k reads native-rate input through its own resampler and checks
three-second windows. A 16 kHz source still uses the 44.1 kHz model. Its reading includes the accepted five-byte payload, symbol
consistency, phase offset, phase count, and actual device. The project receiver
requires consistency at least 0.9 and checks all phases before a negative. A
source shorter than three seconds stays unverified. Symbol consistency is
agreement, not watermark probability; on a negative it includes invalid framing
hypotheses and can still be high.

`remove_audio_invisible(source, output, force=False, verify=True, stream=0)`
selects an adaptive minimal-distortion processing sequence. By default it skips without a compatible
positive source signal and publishes only a fully checked negative saved output.
`AudioVerificationIncomplete` exposes `before` and `after` on rejection and
preserves an existing output. `force=True` permits unverified or still-positive
results. `verify=False` requires `force=True`, `allow_unverified=True`, or an explicit `vendor`. Outputs accept `.wav`, `.flac`, `.mp3` (`libmp3lame` 256 kbps CBR with Xing header), `.m4a` (`aac` 192 kbps with faststart), `.ogg`, `.oga`, `.opus` (`libopus` 192 kbps), `.aac` (`aac` 192 kbps ADTS), and `.mka` (`flac` in Matroska); AI metadata is
discarded. The source must stay unchanged during processing.

`inspect_audio_metadata(source)` returns `AudioMetadataReport`, and `remove_audio_metadata(source, output=None, *, keep_standard=True)` strips AI creation tags and C2PA manifests without re-encoding audio frames (`AudioMetadataResult`). `remove_audio_all(source, output=None, *, include_invisible=False, scheme="auto", method="auto", ...)` returns `AudioAllResult`, always writing a metadata-cleaned output (`<source>_clean<ext>` by default) and optionally cleaning compatible waveform watermarks when `include_invisible=True`. `remove_audio_batch(input_dir, output_dir=None, *, mode="all", include_invisible=False, ...)` runs sequentially over a directory and returns `AudioBatchSummary` of `AudioBatchItem` rows.

`AudioRemovalResult` includes `before`, `after`, `frames_processed`,
`verified_removed`, `skipped`, `removed_markers`, `method`, `verification_scope`,
delivered-file `fidelity`, and `quality_passed`. `verified_removed=None` means unverified, not
success. Fidelity includes RMS and maximum error, SNR, peak amplitude, clipping
count, and frame count against the decoded marked source. Native channels,
sample rate, and decoded sample-frame count must survive processing. Long audio
tracks (`> 30 s`) are verified across bounded `<= 30 s` `whole_segments` passes
alongside sliding windows, and missing requested decoders include install hints.
`AudioStreamInfo.channel_layout` records the known spatial layout; encoding and
saved-output fidelity also reject changes to a known layout.
These APIs do not certify SynthID, Perth v2, or production-provider compatibility.
See [audio coverage](audio-video-removal.md).

Both audio removal functions accept `scheme` with the same semantics, and
`method="auto"` (default), `method="spectral"`, `method="encodec"`, or `method="combined"`
(spectral filtering followed by the codec), with `bandwidth_kbps=24.0`. Supported codec
bandwidths are defined in `_internal/audio_codec.py`. Install `audio-codec`
for this optional CPU backend. Result `method` and `bandwidth_kbps` identify
the applied profile. Neural resynthesis can alter speech timbre and music;
it retains the same strict saved-file verification policy, and 2-channel EnCodec passes operate in Mid/Side (`M/S`) space to preserve stereo phantom-center coherence. Auto tries spectral
then combined for AudioSeal and Perth v1, neural then combined for WavMark and both experimental SilentCipher receivers, routes ElevenLabs and Google SynthID metadata-identified mono, stereo, or tagged 5.1/7.1 audio to `dac` (falling back to `encodec`/`combined` when `dac` is absent) instead of `spectral`, and adapts to positively detected schemes under `scheme="auto"` or `"all"` before falling back to `combined`. A missing required neural fallback raises without publishing output.

Explicit `method="dac"` uses the `audio-codec` extra and fixed FP32
3-codebook 44.1 kHz Descript Audio Codec (`descript/dac_44khz` at revision
`c1bc521685adf9cfe247bc39a5ca58917eda1ac4`, MIT weights) on `vocoder_device`
(`"auto"`, `"cpu"`, `"mps"`, or `"cuda"`), defaulting `preserve_hf_hz=14000.0`
on `>= 44.1 kHz` tracks. It supports mono, raw mid/side stereo with stem
RMS preservation, and tagged `5.1` / `7.1` surround stem decomposition (`FL/FR`
M/S, `FC` mono, `<120 Hz` Butterworth LFE, surround M/S pairs), rational
half-open resampling around 44.1 kHz (preserving full 22.05 kHz bandwidth), and
overlap-window streaming on tracks over 30 seconds. EnCodec bandwidth is
ignored, and `auto` selects DAC automatically when pre-strip metadata or an
explicit `vendor` cohort (`"suno"`, `"udio"`, `"elevenlabs"`, `"google"`, etc.)
asserts a wideband AI audio generator. Tagged `5.1` / `7.1` tracks under
`encodec` and `combined` use the same stem decomposition. Explicit `method="snac"`
uses the `audio-codec` extra and fixed FP32 3-codebook 44.1 kHz Multi-Scale Neural
Audio Codec (`hubertsiuzdak/snac_44khz` at revision
`873ebef9718b89660340c6f55a2b515e98cfa1d9`, MIT weights, `vq_strides = (8, 4, 2)`)
on `vocoder_device` (`"cpu"`, `"mps"`, or `"cuda"`), sharing the same mono, raw
mid/side stereo, tagged `5.1` / `7.1` surround stem decomposition, and
overlap-window streaming while remaining excluded from `auto`.

`remove_audio_invisible`, `remove_audio_all`, `remove_audio_batch`,
`remove_video_audio`, and `remove_video_audio_all_streams` accept `vendor`
(`"auto"`, `"google"`, `"elevenlabs"`, `"suno"`, `"udio"`, `"openai"`,
`"cartesia"`, `"meta"`, `"resemble"`, or `"microsoft"`); `remove_video_all` and
`remove_video_batch` take it as `audio_vendor`. A named vendor (not `"auto"`)
selects that cohort's `method="auto"` candidate order and, because naming it
asserts the proprietary mark is present, implies `allow_unverified=True`.

`remove_video_audio(source, output, stream=0, force=False, verify=True)` processes
one selected audio track into MKV/MOV with float PCM, or MP4 with lossy AAC
(target 192 kbps), while copying other encoded streams. `remove_video_audio_all_streams(source, output=None, ...)` inspects every audio track in the container, stream-copying unmarked tracks and cleaning all positively detected audio tracks into one atomic mux (`VideoAudioMultiStreamResult`).
It returns `VideoAudioRemovalResult` with the same source/output decoder reports,
delivered fidelity, `quality_passed`, `frames_processed`, `verified_removed`, and `skipped`, plus
`unchanged_streams`, `source_audio_start`, `output_audio_start`, and caveats.
Under `method="auto"`, candidate selection runs across the final container mux so
an AAC-recompressed relapse on `spectral` automatically retries `combined` before
failing.
It checks all unchanged packet payloads and their timing, copied-audio priming
and tail trimming, stream languages and dispositions, selected-track frame
count and offset, and repeats verification on the final mux. Discontinuous
audio timelines and unsupported container/codec combinations are rejected.
Output format is selected by suffix; no samples are padded or truncated to repair
AAC count/timing failures, including nonzero-offset timelines the mux cannot
represent exactly. Fidelity and decoder checks cover the actual final
output. Other marks and metadata remain outside this operation's verification scope. It needs the audio extras, not the video
pixel runtime. Default verification failures preserve the existing output.

## Identify and clean video

The high level video API supports MP4, MOV, M4V, WebM, MKV, AVI, and FLV:
metadata-only calls work with the default install, while visible identification,
removal, and the complete pipeline require `remove-ai-watermarks[video]`.

```python
import remove_ai_watermarks as raiw

report = raiw.identify_video("input.mp4")
print(report.is_ai_generated)
print(report.platform)
print(report.visible_mark)
print(report.metadata_markers)
print(report.invisible_watermark_hints)
```

`identify_video` uses the same full-clip temporal arbiter as visible removal.
It reports a recurring registered mark and supported AI-origin metadata as positive
signals. When neither is present, `is_ai_generated` is `None`, never `False`.
The absence of a public local video SynthID decoder is included in `caveats`.
Pass `check_visible=False` for a bounded metadata-only inspection, or
`check_audio=True` (with `audio_scheme="auto"`) to inspect every audio stream in
the container with compatible public decoders (`VideoProvenanceReport.audio_streams`)
and populate `VideoProvenanceReport.recommended_recipe` when signals are detected.
`invisible_watermark_hints` names registered watermark algorithms declared by
C2PA, supported SynthID metadata, and Amazon Nova Reel inferred from Bedrock
video provenance. Fingerprint algorithms are excluded. These are metadata or
provider-policy hints, not pixel detections or per-file removal verification.
C2PA soft-binding names and payloads alone do not set `is_ai_generated`.
An independent AI-origin signal is required; a declaration-only video remains
unknown while retaining the metadata and any watermark hints.
For metadata-only platform attribution, a recognized C2PA claim generator
takes precedence over certificate issuers when a manifest names both.

For normal product integration, use the complete locally verifiable pipeline:

```python
result = raiw.remove_video_all("input.mp4", "clean.mp4")
if result.remaining_metadata:
    raise RuntimeError(f"AI metadata remains: {result.remaining_metadata}")

# User-directed video region erasing (supports inclusive frame_range=(start, end)):
erased = raiw.remove_video_region("input.mp4", "erased.mp4", regions=[(20, 20, 140, 44)], frame_range=(0, 96))
```

`remove_video_region(source, output=None, *, regions, frame_range=None, backend="cv2", dilate=3, strip_metadata=True, temporal_consistency=True)`
fills the given `(x, y, w, h)` boxes on every frame, or only inside the
inclusive `frame_range`, and returns `VideoEraseResult` with `regions`,
`total_frames`, `erased_frames`, `frame_range`, the resolved `backend`, and
`remaining_metadata`.

The default removes one stable supported visible provider mark when present,
always strips verified AI metadata, and copies the source file byte-for-byte
when no visual, metadata, or audio signal requires rewriting. This gives callers
one predictable output path without mutating clean containers.
It does not run lossy invisible regeneration by default.

`include_invisible=True` explicitly adds VAE regeneration for MP4, MOV, or M4V
with the default `invisible_profile="synthid"`, motion-compensated `temporal_stabilize=True`, and `invisible_fps=DEFAULT_VIDEO_SYNTHID_FPS` (pass `invisible_fps=None` to preserve native source frame rate). Select
`invisible_profile="seedance-experimental"` for the experimental MP4 texture-transfer
recipe, or `invisible_profile="wan21-synthid-experimental"` / `"wan21-seedance-experimental"`
on NVIDIA CUDA GPUs (or explicit Apple Silicon `device="mps"`) for full native-resolution 3D `Wan 2.1` video diffusion, and
`invisible_quality="delivery"` (default) or `"master"` for their output encode.
Pass `include_audio_invisible=True` (with `audio_scheme="auto"`, `audio_method="auto"`, `audio_allow_unverified=False`, `audio_allow_positive=False`, `audio_allow_quality_override=False`, `audio_streams=None`, or compatibility alias `audio_force=False`) to also inspect and clean compatible watermarked audio tracks in the same atomic pipeline.
These options also apply to `remove_video_batch(..., mode="all")`, where each `VideoBatchItem` also exposes `audio_verification_scope` and `audio_quality_passed`.
`VideoAllResult.invisible_profile` records the selected stage, or `None` when
it did not run.
`VideoAllResult.invisible_removed` remains the compatibility boolean for whether
the visual regeneration stage ran; it is an action record, not a per-file
watermark verdict. `visual_invisible_action` exposes the same distinction as
`not_run` or `regenerated`. By default `audio.stream_action` is `copied_if_present` while
`audio.watermark_status` remains `unverified`; when `include_audio_invisible=True` processes detected audio tracks, `audio.stream_action` is `processed` and `audio.watermark_status` reflects the verified saved-mux outcome (`not_detected`, `detected`, or `unverified`).

Process a top-level directory sequentially:

```python
batch = raiw.remove_video_batch("videos", "videos_clean", mode="all")
if batch.failed:
    for item in batch.items:
        if item.error:
            print(item.source, item.error)
```

Batch modes are `all`, `visible`, and `metadata`. Successful visible no-ops are
copied byte-for-byte, keeping the output directory complete. Per-file failures
are returned in `VideoBatchItem.error`; they do not discard successful outputs.
The invisible stage is available only as an explicit opt-in in `all` mode and
reuses one loaded VAE runtime across the batch. Each item carries the same
`audio` status, and `visual_invisible_action` distinguishes regeneration from a
run that left invisible video pixels untouched. A failed item whose output was
not published reports `audio.stream_action=not_written`.

## Inspect and strip video metadata

Metadata inspection and removal use the same supported video containers:

```python
import remove_ai_watermarks as raiw

report = raiw.inspect_video_metadata("input.mp4")
if report.has_ai_metadata:
    result = raiw.remove_video_metadata("input.mp4")
    if result.remaining:
        raise RuntimeError(f"AI metadata remains: {result.remaining}")
```

`remove_video_metadata` does not transcode video or audio streams. Its default
output is `input_clean.mp4`, leaving the source untouched. An explicit output
must use the same container extension as the source.

The returned `VideoMetadataResult` records the source, output, metadata detected
before removal, and any markers remaining after the verified strip. Its `audio`
field records that source audio is copied if present and not watermark-verified.
MP4/MOV
inspection recognizes the native TC260 `AIGC` entry in
`moov.udta.meta.keys/ilst` and the QuickTime-form `meta` variants Doubao's iOS
export writes. It also recognizes keyed `workflow` and `prompt` generation
metadata in the same metadata-list structure; removal preserves container size and encoded
stream bytes. MP4/MOV/M4V are copied in bounded chunks, so a large `mdat` is not
loaded into memory; publication is atomic. MKV/WebM inspection recognizes the corresponding
`Segment.Tags.Tag.SimpleTag` representation; its removal requires ffmpeg for a
stream-copy remux. AVI inspection reads `LIST/INFO/AIGC`, and FLV inspection
reads `script.onMetaData.AIGC`; both use the same verified ffmpeg stream-copy
removal path.

## Apply the video-pixel SynthID-removal profile

Install `remove-ai-watermarks[video,diffusion]` before using the video SynthID
API.

```python
import remove_ai_watermarks as raiw

result = raiw.remove_video_invisible(
    "input.mp4",
    "clean.mp4",
    device="auto",
)
if result.remaining_metadata:
    raise RuntimeError(f"AI metadata remains: {result.remaining_metadata}")
```

`remove_video_invisible` supports MP4, MOV, and M4V. It regenerates the complete
video through a VAE in bounded batches, shares one seeded latent-noise field
across all frames, applies default motion-compensated optical-flow blending (`temporal_stabilize=True`) to suppress VAE inter-frame flicker (`fps=None` preserves native source FPS), streams pixels to ffmpeg, copies complete audio, strips
source metadata, and publishes atomically. The default output is
`input_clean.mp4`; a distinct same-container output is required.

The returned `VideoInvisibleResult` includes output geometry, frame rate, frame
count, paired PSNR, and the motion-compensated temporal-residual ratio. Those
fields measure fidelity and flicker only. They are not a SynthID detector.
`visual_invisible_action` is `regenerated`; the nested `audio` status says
`copied_if_present` and `unverified`. These are action and evidence records, not
an assertion that every modality is clean.
The default `noise_std=0.15` is the current full-clip oracle floor; `0.10`
remained detected on the public eight-second Veo calibration carrier.
The default profile is oracle-certified. Google does not publish a local
decoder for this video payload, so a fresh source-positive, output-negative
pair from Gemini's built-in SynthID verifier remains an optional per-file audit.
A response inferred from a visible logo or metadata is not such a verdict, and
an adversarial follow-up asking ordinary Gemini to reinterpret the verifier is
not a second oracle run.

## Experimental Seedance video reconstruction

Install `remove-ai-watermarks[video,diffusion]` and FFmpeg, then explicitly opt
into the experimental Python API:

```python
import remove_ai_watermarks as raiw

result = raiw.remove_video_seedance_experimental(
    "owned-label-cleaned.mp4",
    "seedance-delivery.mp4",
    device="auto",
    quality="delivery",
)
assert result.experimental
assert result.profile == "seedance-texture-v1"
```

`quality="delivery"` writes a native-size H.264 High CRF18 MP4.
`quality="master"` retains the lossless-coded YUV420p reconstruction before
that final delivery encode; its H.264 profile and larger size may require
conversion in downstream editors. Both modes preserve the complete decoded
frame sequence, presentation timestamps, terminal video duration, and original
audio stream. They strip AI metadata and publish the output atomically after
clock and decoded-frame checks. Frame processing is streamed in bounded
batches; encoded intermediate videos use temporary disk space and are cleaned
after success or failure.

This API accepts owned SDR 8-bit MP4 videos with even dimensions. `device`
accepts `auto`, `cuda`, `mps`, or `cpu`. The default output is
`owned-label-cleaned_seedance_experimental.mp4`; input and output must be
distinct MP4 paths. The fixed profile uses the pinned SD VAE at a 512-pixel
working grid, zero latent noise, and sigma-1 source-texture restoration. These
settings are not exposed as tuning parameters.

The returned `VideoSeedanceExperimentalResult` extends `VideoInvisibleResult`
with `experimental`, `profile`, and `quality`. Its PSNR and temporal residual
measure the delivered native-size video. `visual_invisible_action` is
`regenerated`; copied audio retains an `unverified` watermark status. There
is no local CapCut detector or automatic external upload in this function.
Positive-control/negative-output CapCut evidence for the fixed recipe is
recorded in [the Seedance study](seedance-video-study.md). Fidelity loss can
be visible and the measured outcome is not a per-file runtime verdict.

The same recipe is integrated into the common pipeline:

```python
result = raiw.remove_video_all(
    "owned.mp4", "clean.mp4",
    include_invisible=True,
    invisible_profile="seedance-experimental",
    invisible_quality="delivery",
)
```

`remove_video_invisible(..., profile="seedance-experimental", quality="delivery")`
selects the pixel stage directly and keeps the common `<source>_clean.mp4`
output convention. Profile selection without `include_invisible=True` in
all/batch mode is rejected. Nondefault SynthID tuning options are rejected for
Seedance, whose recipe is fixed. A batch reuses one VAE runtime across files;
unsupported containers fail their individual rows.

Visible overlays must be removed separately, or disabled in the generator's
official export, before calling the standalone pixel stage. The function does not locate
or erase a visible `CapCut AI` label. Existing visible-removal APIs can precede
it for their supported marks. `remove_video_all` runs the existing visible
stage before the selected pixel stage. The CLI exposes the same choice through
`--profile seedance-experimental`; all/batch mode also requires `--invisible`.
SynthID remains the default.

On hosts with a hardware BF16-capable NVIDIA CUDA GPU (Ampere or newer; ROCm and
emulated BF16 are excluded), or with explicit `device="mps"` on Apple Silicon
(FP16 transformer; `profile="auto"` never selects MPS), `remove_video_wan`
(legacy alias `remove_video_wan_experimental`, and `profile="auto"`,
`"wan21-synthid"`, or `"wan21-seedance"` in `remove_video_invisible`,
`remove_video_all`, and `remove_video_batch`) runs pinned Apache-2.0 3D `Wan 2.1`
(`Wan-AI/Wan2.1-T2V-1.3B-Diffusers` at `0fad780a534b6463e45facd96134c9f345acfa5b`)
at native resolution and exact frame timestamps on SDR 8-bit and 10-bit HDR
clips across all 7 supported containers (`.mp4`, `.mov`, `.m4v`, `.mkv`,
`.webm`, `.avi`, `.flv`), using 1080p-guided texture restoration above
`1920 * 1080` and streaming clips exceeding `chunk_frames=193` across
`overlap_frames=17` raised-cosine 3D causal windows directly to the encoder
(pre-encoding `prompt_embeds` once so the T5 text encoder is freed before 3D
denoising), returning `VideoWanExperimentalResult` (which exposes `strength`
alongside the compatible `noise_std` mirror). `profile="auto"` selects
`wan21-seedance` on eligible BF16 CUDA sources (`>= 8 GiB` total VRAM, stepping
down to 81-frame chunks on 1080p or `< 16 GiB` CUDA and skipping `UMT5-XXL` when
the static prompt embedding is cached locally; `seedance` fallback on CPU/MPS)
when ByteDance/Seedance metadata or visible marks are present, and
`wan21-synthid` (`synthid` fallback) otherwise. `wan21-synthid` uses `strength=0.38`
and `wan21-seedance` uses `strength=0.12`. Both accept
`quality="delivery"` or `"master"` and `seed`, reuse one loaded pipeline across
batch items, and perform no runtime oracle verification. `VideoAllResult` also
records `audio_verification_scope` and `audio_quality_passed` when
`include_audio_invisible=True`.

## Remove a supported visible video mark

```python
import remove_ai_watermarks as raiw

result = raiw.remove_video_visible(
    "input.mp4",
    "clean.mp4",
    backend="cv2",
    strip_metadata=True,
    temporal_consistency=True,
)
if result.output is None:
    print("No temporally stable supported mark was found")
else:
    print(result.mark)

veo_result = raiw.remove_video_visible(
    "veo.mp4",
    "veo_clean.mp4",
    mark="veo",
)
seedance_result = raiw.remove_video_visible(
    "seedance.mp4",
    "seedance_clean.mp4",
    mark="seedance",
)
dola_result = raiw.remove_video_visible(
    "dola.mp4",
    "dola_clean.mp4",
    mark="dola",
)
hailuo_result = raiw.remove_video_visible(
    "hailuo.mp4",
    "hailuo_clean.mp4",
    mark="hailuo",
)
kling_result = raiw.remove_video_visible(
    "kling.mp4",
    "kling_clean.mp4",
    mark="kling",
)
```

`remove_video_visible` scans the complete video before writing output. It
combines synthetic multi-scale visual matching with temporal consistency, so an
isolated lookalike in one frame is not enough to authorize inpainting.
`mark="auto"` is the default: it evaluates all providers in one decode pass and
selects the first stable match in specificity order (`sora`, `veo`, `seedance`,
`doubao`, `dola`, `hailuo`, `vidu`, `kling`). Provider confidence values are calibrated
independently and are not compared across detectors. Pass one of those explicit
values to restrict the scan to a single provider. A stable Veo diamond takes
precedence over Sora only with valid Google AI-video provenance, no Sora
provenance, and either substantial overlap or a full-clip Veo run against a
partial Sora run. The Veo detector recognizes
the current four-point diamond and the
legacy `Veo` text. Seedance recognizes the boxed `AI` label, Doubao recognizes
the fixed `豆包AI生成` label, Dola recognizes
its compact text label, Hailuo AI recognizes the composite MINIMAX/Hailuo AI label,
Vidu recognizes the two-loop logo and `Vidu AI` wordmark,
and Kling AI recognizes its bottom-right logo, wordmark, and version suffix. Each
variant has an independent synthetic silhouette and calibrated temporal policy.
After each accepted frame is filled, `temporal_consistency=True` motion-aligns
the preceding accepted fill and blends it only when the warped prior mask
covers the current mask and a surrounding source-context ring agrees. Scene
cuts and disjoint masks keep the independent current fill. Pass
`temporal_consistency=False` for the frame-local baseline.

The returned `VideoVisibleResult` records the selected `mark`, the total,
detected, and removed frame counts, plus any AI metadata that survived the
output encode. Its `audio` field records `copied_if_present` and `unverified`.
The function returns `output=None` and writes no file when no
stable mark is selected; that result records `audio.stream_action=not_written`.
Video pixels are transcoded through ffmpeg while the
complete source audio stream is copied. The encoder preserves supported 8-bit
source chroma sampling, color tags, MP4/MOV track timescale, and relative
variable-frame timestamps. It also retains a non-zero source start PTS and the
copied audio offset. A failed encode preserves any existing output; only a
completed result is published atomically.
SDR 8-bit video is the supported pixel contract. High-bit-depth, PQ, and HLG
sources raise `RuntimeError` before encoding instead of being silently reduced
to 8-bit SDR.

## Remove invisible watermarks

Install `remove-ai-watermarks[qwen-zimage]`. All concrete profiles need it, and all
need an NVIDIA GPU.

```python
from pathlib import Path

from remove_ai_watermarks.invisible_engine import InvisibleEngine

engine = InvisibleEngine(
    pipeline="qwen-zimage",  # the default; also `qwen21-zimage` (non-commercial), "sdxl-zimage", "chroma-zimage", or "auto"
    device=None,
    cpu_offload=False,
)

engine.remove_watermark(
    Path("watermarked.png"),
    Path("clean.png"),
)
```

`device=None` and `device="auto"` both run detection. `"cuda"` pins it without
detecting. Every other value raises at construction rather than deferring a
guaranteed failure to model-load time.

For limited CUDA memory:

```python
engine = InvisibleEngine(
    pipeline="qwen-zimage",
    cpu_offload=True,
)
```

All profiles are CUDA-only, so on a machine without an NVIDIA GPU `device=None`
resolves to `cpu` and construction raises. For the SDXL global stage instead of
Qwen:

```python
engine = InvisibleEngine(pipeline="sdxl-zimage")
```

The `qwen-zimage` extra is required for every profile, including `auto`: each
concrete engine runs the same DiffSynth Z-Image face stage. `pipeline="auto"`
selects sdxl-zimage for Google provenance, chroma-zimage for Microsoft, and
qwen-zimage otherwise, after the vendor is known and before strength resolution.

### Fidelity and qualified text geometry

`InvisibleEngine(pipeline="qwen21-zimage")` selects Qwen-2.1 with fidelity refinement enabled by default,
under the non-commercial Qwen Research License.
It aligns geometry without OCR by default. A host integration can call
`engine.set_text_region_selector(selector)` before processing. The callable
receives an owned RGB PIL image and returns bounded `(x1, y1, x2, y2)` text
rectangles. It must independently reject non-text and exclude known visible
provider marks. A detector score alone does not establish eligibility.

`engine.set_text_region_selector(None)` disables text reconstruction while
retaining geometric alignment. `engine.set_fidelity_enabled(False)` retains the
shared Qwen-2.1 recipe and strength policy without reloading model weights;
restore it with `True` before the next eligible request. These setters accept only
the Qwen-2.1 profile. Eligibility and quality guards do not establish
watermark-removal evidence. See [profile limits](cli.md#fidelity-profile).

### Reviewed text manifests

The opt-in verified-text stage uses the same `text_manifest` argument as the CLI:

```python
engine.remove_watermark(
    Path("watermarked.png"),
    Path("clean.png"),
    text_manifest=Path("verified-lines.json"),
)
```

Install `remove-ai-watermarks[text-restoration]`. The manifest schema and safety
constraints are documented in the CLI guide. The engine verifies its decoded RGB
hash before loading the diffusion models. Qwen, SDXL, and Chroma reconstruct the
donor with the VAE already loaded for the selected profile; no second generative
profile runs. The engine turns an unset `adaptive_polish` off in verified-text mode
and rejects explicit postprocessing or downscaling combinations that were not
evaluated. Tiling is also rejected because the combined
tiled-restoration path has no provider-oracle calibration. `InvisibleOptions` exposes
the same field for `remove_all`; after a visible-stage edit, the manifest must be built
against the staged pixels rather than the pristine source.

Use manifest schema 1 for manually reviewed text plus script metadata. Automated
operators that verify only text-region geometry should emit schema 2 lines with a
`box` and optional `angle`; no placeholder transcription or script is required.

Since 0.27.1 the mode's global 15% Qwen-VAE fidelity-anchor blend is **off by
default** (`fidelity_anchor=False`): that whole-frame blend was measured to
return detector-visible OpenAI SynthID on poster-scale manifests (official
Content Provenance API, 2026-08-19 - detected x6 with the anchor, clean x6
without it, controls and base outputs validated in the same sessions). Pass
`fidelity_anchor=True` with `qwen-zimage` to reproduce the 0.27.0 research
behavior. Chroma rejects that Qwen-specific reproduction flag.

### Drafting manifest lines

`remove_ai_watermarks.text_draft` proposes lines for a manifest; it never
produces verified ones:

```python
from remove_ai_watermarks.text_draft import draft_text_lines

draft = draft_text_lines(Path("watermarked.png"))
for line in draft.accepted:
    print(line.box, line.script, line.min_score, line.text)
```

Install `remove-ai-watermarks[text-draft]` (CPU, no torch: PaddleOCR detection
plus three script-chosen recognition engines). A line lands in `accepted` only
when three crop paddings normalize identically and every confidence clears
`min_score` (default 0.85); `accepted` means crop-stable, NOT ground-truth-
correct - on the reference posters the draft's exact-text precision was 90.0%
and 94.4% because high-confidence OCR still lost punctuation. Every accepted
line needs a human yes/no before a manifest may claim `verified: true`.
`source_pixel_sha256` is re-exported here for building the manifest's
pixel-binding hash against the exact source the engine will decode.

`remove_watermark` takes strength, seed, tiling, resolution, and postprocessing
controls. It takes no model id, step count or guidance scale, and neither does the
constructor: each profile pins its model stack, its per-stage schedule and CFG
1.0, so passing one raises `TypeError` at the call rather than being accepted and
refused several layers down. Read the method signature in
[`invisible_engine.py`](../src/remove_ai_watermarks/invisible_engine.py) or use
the CLI guide for the concepts.
Defaults can differ between the Python method and CLI profile resolution, so
pass values explicitly when reproducibility matters.

## Public-model known video messages

```python
report = raiw.identify_video_message("input.mp4", reference_hex=encoder_message_hex, scheme="videoseal")
result = raiw.remove_video_message(
    "input.mp4", "processed.mp4", reference_hex=encoder_message_hex, scheme="pixelseal"
)
presence = raiw.detect_video_watermark_presence("input.mp4", scheme="chunkyseal")
```

`detect_video_watermark_presence` inspects channel-0 detection probability and
cross-frame payload sign consistency (`256` bits for `videoseal`/`pixelseal`,
`1024` bits for `chunkyseal`) without requiring a known reference hex key,
returning `VideoWatermarkPresenceReport`.

The reference is an independently known encoder message: 64 hex digits (`256`
bits) for `scheme="videoseal"` or `"pixelseal"`, or 256 hex digits (`1024` bits)
for `scheme="chunkyseal"`. Never infer it from the file being checked. The API
compares that target under pinned public weights; it does not detect unknown messages,
classify AI origin, or certify provider deployments or complete erasure.
`VideoMessageReport` binds `source`, `sha256`, `scheme`, `reference_hex`, and
`matched|not_matched|unverified` status to a reading over every decoded frame,
four aggregations, and stride-1 12-frame sliding windows with a context-completed
tail, verified against `ffprobe` packet timestamps.
`check_message=False` and missing optional models stay unverified. Decode and
model errors propagate, including incomplete declared streams and invalid logits.

`remove_video_message` accepts `method="auto"|"codec"|"vae"`, `noise_std=None`,
`long_side=512`, `device="auto"`, `force=False`, `verify=True`, and
`temporal_stabilize=False`.
An explicit noise level requires `method="vae"`. Auto tries codec, VAE with
zero noise, then fixed noise levels 0.15 and 0.25. `VideoMessageRemovalResult`
includes the selected profile, before/after reports, `verified_removed`,
`skipped`, delivered `quality`, `quality_passed`, and `copied_audio_streams`. A source that is
not matched skips by default. Failed saved-file checks raise
`VideoMessageVerificationIncomplete` with before/after evidence and preserve
an existing output. Disabling checks requires force. A verified result requires
both a source match and a complete saved-output negative.

The processing geometry, supported timelines and containers, audio guarantees,
and quality tradeoffs follow the [CLI contract](cli.md#public-model-video-messages-experimental).
Install `video-detect`, plus `diffusion` for VAE profiles. Imports remain lazy.

## Identify and clean text provenance and token watermarks

The high-level text API in [`text.py`](../src/remove_ai_watermarks/text.py) is
re-exported on the top-level `remove_ai_watermarks` package:

```python
import remove_ai_watermarks as raiw
from pathlib import Path

source = Path("article.md")  # str inputs are literal text, not filenames.
report = raiw.identify_text(source)
print(report.is_ai_generated, report.confidence, report.signals)
print(report.vsrmark_payload, report.unicode_tags_payload)

meta = raiw.remove_text_metadata(source, "article_clean.md")
print(meta.removed_signals, meta.removed_codepoints)

# Span-protected prose regeneration for token-level LLM watermarks:
inv = raiw.remove_text_invisible(source, "article_regenerated.md", seed=0)
print(inv.rewritten_spans, inv.locked_spans, inv.token_watermark_action)

# Combined pipeline (metadata strip + optional prose regeneration):
combined = raiw.remove_text_all(
    source,
    "article_all.md",
    include_invisible=True,
)
```

File input/output preserves UTF-8 line-ending bytes (LF, CRLF and CR). Segmentation
recognizes the same protected structures across those endings; rewritten prose
can still change its own internal layout.

Single-dollar inline math uses non-whitespace content at both delimiter edges,
and its closing dollar must not be followed by a digit. This prevents currency
pairs such as `$500` and `$1,750` from locking the intervening prose as math.
Numeric expressions such as `$5 + 2$` and `\(x + 1\)` remain protected.

`rewriter_backend="mlx"` selects optional local MLX generation on Apple silicon;
install `remove-ai-watermarks[text-mlx]` and pass a compatible `rewriter_model`
checkpoint path or Hugging Face ID. `device="auto"` uses the MLX default stream;
`device="mps"` selects a GPU stream. Other devices are rejected for this backend.
`is_text_invisible_available("mlx")` in the `text` module checks Apple silicon,
the concrete `mlx.core` module, `mlx_lm` and `py3langid`. The namespace-only `mlx` parent does
not make an installed runtime unavailable in a fresh process.
The default remains `rewriter_backend="transformers"`. Both backends share the
same factual/placeholder acceptance checks and three-attempt policy. MLX requires
an explicit `stop` finish reason; a length limit or unknown completion is refused.
The combined API requires `include_invisible=True` for a nondefault backend, and
a custom `rewriter` cannot also select MLX. No semantic judge is enabled by this
choice, and larger models can still change qualifiers or facts.

### Optional API text rewriter

Local generation remains the default. Install `remove-ai-watermarks[text-api]`
to use `build_text_api_rewriter` with an explicitly selected Chat Completions
endpoint. This extra needs the packaged language checker, without Torch,
Transformers or a provider SDK. The endpoint and model have no defaults; no
credentials or other API settings are read from the environment.

```python
from getpass import getpass
import remove_ai_watermarks as raiw

rewriter = raiw.build_text_api_rewriter(
    base_url="https://your-endpoint.example/v1",
    model="your-compatible-model",
    api_key=getpass("API key: "),  # Omit for an unauthenticated local endpoint.
    max_completion_tokens=1024,
    timeout=60.0,
)
result = raiw.remove_text_invisible("input.md", "output.md", rewriter=rewriter)
```

Construction makes no request. Each rewriter invocation sends one nonstreaming
POST to `base_url + "/chat/completions"`, with the shared local prompt and
masked factual spans. Technical spans remain protected by the outer library
pipeline. The endpoint must honor the chosen token-limit field and return
exactly one assistant text choice with `finish_reason="stop"`. The adapter
sends `max_completion_tokens` by default. For an endpoint that expects
`max_tokens`, pass `completion_token_field="max_tokens"`. The numeric limit
still comes from `max_completion_tokens`. Truncation, refusals,
tool calls, malformed replies and failed shared acceptance checks raise
`RuntimeError`. A failed request never triggers a local or remote fallback.
The adapter does not retry. An explicit outer `rewrite_validator` retains the
existing three-invocation policy; a document can contain multiple prose
clusters. Request count therefore depends on the text and caller policy.

`max_completion_tokens` is a positive integer, and `timeout` is a positive
finite number of seconds. The base URL must be HTTP(S), without embedded
credentials, query or fragment. The optional key is sent only in the
authorization header. Redirects are refused. Full request/response bodies are
logged at INFO and failures at WARNING, with the configured key redacted.
The adapter sends user prose to the configured service only when invoked.

The same selected fact, placeholder, extent, script and language checks apply
to built-in API and local generation. These heuristics do not establish full
meaning preservation. Transport tests against a local HTTP server are not
provider-model quality tests, and an API model may add its own watermark.
An original detector becoming negative does not verify the absence of another
provider's marking. No paid provider evaluation is implied by this API.
Protocol: [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

### Text verification and acceptance

For local calibration or testing when the deployment's watermark key is known,
[`remove_ai_watermarks.text`](../src/remove_ai_watermarks/text.py) also provides
keyed statistical scorers (`score_synthid_text`, `score_synthid_threefry_text`, `score_gumbel_text`,
`score_textgrain_text`, `score_textseal2_text`, `score_kgw_text`,
`score_unigram_text`), selected numeral/entity/polarity/season/frequency and
claim qualification checks
(`verify_text_rewrite_facts`), covert steganography payload decoding
(`decode_stego_payloads`), and deterministic wrapper encoders
(`encode_c2patxt`, `encode_c2pa_ascii_armor`, `encode_vsrmark`,
`encode_unicode_tags`, `encode_innamark`). See
[text watermark and provenance research](text-watermark-research.md).

Selected determiner-led affirmative English copular frames also reject a
membership statement becoming a critical stance, or the reverse, for matching
literal subject/object descriptions. Modality, negation, other grammar,
different participant descriptions and mixed claims remain unresolved. This
does not generally bind entities or establish equivalence.

The deterministic cleaner recognizes the [published Windows AI text sequence](https://learn.microsoft.com/en-us/windows/ai/apis/text-watermarking)
`U+200B U+034F U+2060 U+200B` as `windows_ai_text_watermark`. It removes complete
occurrences before classifying neighboring Unicode controls, preserving valid
emoji and joining-script context. The sequence is forgeable, so this signal does
not attribute the source to Microsoft. Tests cover synthetic published-format
fixtures; actual Windows AI output has not been generated in this environment.

The metadata cleaner preserves isolated `U+200B` between Thai word characters
and inside recognized inline and fenced code. It also preserves paired
`U+2060` word joiners around `U+00D7` or `U+00F7` between decimal digits.
Recognized zero-width payloads take precedence over these context rules.
The cleaner preserves ordinary non-breaking spaces (`U+00A0`) in prose.
A recognized Innamark payload can still trigger whitespace recovery.

The cleaner also retains `U+034F` when removing it would allow combining
marks to reorder across its position. This preserves the ordering distinction
described in [Unicode 17's layout controls](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-23/).
The complete Windows AI sequence is removed before the check. Collation-only
uses between ordinary letters are outside this rule.

The metadata cleaner also retains single selectors in registered Unicode 17.0
standardized VS1 through VS16 pairs. It removes unregistered BMP selectors
outside the separate emoji presentation cases. Supplementary CJK selectors
continue to use the existing ideograph context rule.

The local rewriter raises `RuntimeError` after three attempts if every candidate
hits the token budget without an EOS stop, severely collapses or expands a long
passage, fails selected numeral/entity/polarity/season/frequency checks, changes service placeholders, switches the
dominant writing script, or
repeats the original verbatim. Install `remove-ai-watermarks[text]` for the
default Transformers backend. Its configured chat template supplies native special
tokens once; a tokenizer without a template uses a plain prompt with automatic
special tokens. Both local backends additionally use a packaged
`py3langid==0.4.0` model to reject confident language changes, including those
within one script. After removing service placeholders and normalizing the
classification sample to NFC, both samples need at least 40 alphabetic characters.
A known source label needs a normalized score of at least 0.95; a different
candidate label needs at least 0.8. Short samples, uncertain labels and unknown
source labels abstain. These model scores are not calibrated accuracy estimates.
Only the model is cached, not classified user text. The script check remains an
independent backstop. For sufficiently long, predominantly Latin or Cyrillic sources,
it also rejects newly introduced letters of the other alphabet when that alphabet
was absent from the source. Existing mixed-script sources remain a limitation.
List prefixes are protected structural spans. A custom
rewriter returning an entire prose paragraph unchanged is also refused. A failed
rewrite does not replace the destination file. These deterministic checks do not
prove semantic equivalence; review the result. All-caps terms, including
sentence-initial acronyms, are treated as protected factual tokens. This is a
conservative lexical heuristic, not general named-entity recognition.
Numerical checks retain normalized occurrence counts, not only unique values.
An extra `two` is rejected even if `two` already occurs in the source. Whole
cardinals and selected measurements/durations contribute one amount per mention;
recurrence counts and service-marker indices are excluded from bare quantities.
This is conservative about mention multiplicity and can reject a valid
compression of repeated quantities. Equal occurrence multisets still do not
establish which actor or object a number belongs to.
Selected indefinite English durations (`an hour`, `a day`, `a week`, `a month`,
`a year`) bind amount one to the complete unit phrase, including recognized
numeric bounds. They may match `one day`/`1 day`, but cannot disappear or change
unit. Local backends mask the whole phrase. Recurrence phrases such as `once a
day` remain frequency facts rather than an additional duration measurement.
Ambiguous `a second`/`a minute`, hyphenated compounds and compound-unit ratios are
not interpreted by this indefinite-duration extension. It does not verify which
event a duration modifies or preserve arbitrary conditional meaning.
Numeric English compounds such as `four-week` and `two-hour` are separately
recognized as complete measurements for hour/day/week/month/year. They can match
ordinary `four weeks`/`two hours` forms while retaining values, units, bounds
and occurrence counts. This does not interpret arbitrary hyphenated expressions.
Selected English and Russian season mentions are also checked. Bare `spring`
and `fall` remain ambiguous; they may match explicit seasonal counterparts but
do not independently establish a season. A spring-to-autumn interval contributes
both seasons. This backstop does not establish temporal equivalence, dates,
frequency or which event a season modifies.
Selected explicit English recurrence phrases are checked separately. Hourly,
daily, weekly, monthly and yearly/annual phrases, selected `every N` intervals,
`every other`/`every second`/`on alternate` forms, and once/twice/thrice or numeric
repeat counts retain their occurrence count, period and unit. For example,
`every other day` may match `once every two days`, but cannot become `periodically`
or `every day`. Whole recognized phrases are masked for both local backends;
their internal numerals/units do not become unrelated quantity constraints.
`annual` matches `yearly`/`annually`, and bare `per week`-style periods match
their selected adverb aliases, including monetary rate wording. Frequency words
inside hyphen-prefixed compounds such as `semi-annual` are excluded, rather than
misread as annual recurrence. Unsupported compounds are not otherwise interpreted.
The check preserves mention multiplicity conservatively. It does not convert
calendar units, interpret arbitrary or bounded schedules, bind intervals to
specific actors/events, or certify semantic equivalence.
Selected explicit English type-coverage phrases (`each type`, `every appliance
type`, `all appliance types`) share a conservative mention count and whole-phrase
masking. Losing or inventing this selected coverage fails the local guard.
The check does not bind types to entities, interpret arbitrary quantifiers,
normalize every paraphrase, or certify universal semantic coverage. Adding an unsupported paper medium to a manual
process and swapping explicit event/date assignments can also pass. The examined
calibration pairs record these unresolved gaps. Full meaning review remains
separate from this heuristic verifier. Examined work/residence, beneficiary
class, goal/achievement, likelihood and urgency pairs also record current gaps;
retaining names and numerical values does not establish those relationships.
Consecutive recognized name words are masked as one span, so a name such as
`Willow Creek` cannot be split by moving its component markers. The fact checker
also requires those runs to remain contiguous, allowing case/whitespace changes.
This preserves recognized text, not the name's role or complete entity coverage.
Selected `Mr.`, `Mrs.`, `Ms.`, `Dr.` and `Prof.` phrases include capitalized
name words and initials even at sentence start. Native masking keeps the full
honorific/name phrase in one marker for each occurrence, so the honorific period
cannot leave the surname unprotected. Selected sentence/paragraph-initial ASCII
name runs before an explicit appositive, such as `Robert Johnson, the curator`,
are also masked as whole phrases and checked in full. Common leading function
words and numeric words are excluded. This prevents a recognized first name
being changed while only its surname survives. Single names, arbitrary initial
name runs, initials and general entity/role resolution are outside this extension.
The fact checker requires each recognized
full phrase to remain contiguous, allowing case/whitespace changes; it does not
require repeated mentions to survive a valid pronoun paraphrase or bind names
to actions.
The separate fact check recognizes some sentence-initial English actor
appositions such as `the curator, a woman with a silver badge`. For selected
actions, it rejects rewrites that make that same description a separate actor
or recipient in an event involving the curator. It compares specific literal
descriptions and action forms. Other appositions and general role or pronoun
resolution remain outside the check.
For selected English actions, the fact guard compares plans, attempts and
completions when both versions name the same actor and object. A completed
passive with an explicit actor can match an active clause. A separate shelter
frame distinguishes seeking refuge from finding or taking it when the actor
and any stated location match. Missing or ambiguous matches leave the phase
undetermined.
The guard also compares selected affirmative English `affect`, `impact` and
`influence` clauses with matching factors, `how` effects and stated conditions.
It flags a change between `may`, `might` or `could` and an assertion. For
`can`, it flags only a contrast with an explicit assertion such as
`directly influences`. Other expressions of likelihood and causal scope remain
outside this check.
Selected English/Russian approval, permission, responsibility, restriction and
requirement cues must also survive. For example, available funding does not
retain an explicit approval condition, and not performing repairs is different
from having no repair responsibility or being forbidden to repair. These are
selected literal categories, without general actor, event or scope binding; unrecognized
synonyms can cause conservative refusals. Additive idioms such as
"not only" are excluded, and the checked Russian necessity construction accepts
"registration is required" versus "access without registration is impossible".
The English requirement family includes `essential tool` and `essential instrument`
wording alongside `indispensable`, and still checks added or omitted requirements.
One paired English exception distinguishes selected technical capability from
permission: `A calculator allows students to calculate totals` can become
`A calculator enables students to compute totals`. The sentence-initial technical
subject, actor, functional action and bounded object must match, allowing only
selected action/object aliases. Only the matching `allow`/`enable` cues are
excluded from the constraint comparison. Human/policy permission, access verbs,
explicit permission elsewhere and unmatched or repeated frames remain checked.
The same bounded exception includes mechanical film advance when the action
still captures the same scene repeatedly without manual rewinding.
This does not establish general capability, authorization or object semantics.
These checks do not establish qualifier scope or complete claim coverage. Curly
apostrophes are normalized for English negation checks, including `mustn't`.
Selected explicit English record-container expressions are also checked: a
`communal notebook` or `shared ledger` cannot become `each tool has its own
notebook`, or the reverse. Separate entries/pages in a shared book are distinct
from separate books. This check abstains when either distribution is implicit,
unrecognized or mixed; it does not bind records to particular people or tools,
detect every omitted item, or certify full ownership/scope preservation.
Selected explicit English signal directions are also compared: `a nod from the
community` cannot become `a nod to the community`, including `a nod in return to
the community`. The check recognizes selected group/owner descriptions with
`nod`, `approval`, `permission` or `consent`. It compares the same literal signal
and participant only when each has one explicit direction. Missing cues,
different participant descriptions and mixed directions abstain. This does not
establish general approval-actor or manner-to-event binding: retaining carefulness
does not establish which action is performed carefully.
Selected English/Russian numeric bounds and approximations must retain their
kind and value: for example, `at least 20` can become `no fewer than 20`, but not
`20`, `at most 20` or `more than 20`. Quantified `no fewer/more than` is a bound,
not predicate negation; other negation still matters. This remains a lexical
heuristic. It does not cover all range/number expressions or bind each quantity
to its participant, and ambiguous words such as `around` can cause refusals.
Reciprocal/superlative `one` idioms are excluded from approximation matching;
explicit bounds such as `at least one` remain checked.
Selected adjacent English percentage change phrases distinguish endpoints
(`rose to 86%`) from change magnitudes (`rose by 86%` or
`an increase in satisfaction of 86%`). The check rejects a role swap
when the same normalized value has one recognized role on each side.
Unrecognized and mixed roles abstain.
Selected adjacent English `estimated N`, `estimated at N` and `estimate of N`
forms retain a distinct estimate qualifier and the supported value/unit. An
estimate cannot become an unqualified measured value or change its supported
value/unit. Nonadjacent descriptions such as `the estimated cost was ...`, general
forecast uncertainty and participant binding remain outside this check.
Selected Russian cardinal forms and descending thousand/million/billion scales
are normalized as whole values: `двадцать пять` and `25` agree, while changing
the value or its supported bound is refused. The local writer masks those
sequences and selected duration adjectives such as `пятилетняя` as complete spans.
Superlative `один из лучших` is not a quantity; `двадцать один из лучших` still
contains `21`. Ambiguous `сорока` is not parsed as forty. Unparsed recognized
sequences and indefinite bare plural scales retain literal signatures.
This does not cover every inflection, ordinal or fractional number word or
quantity-to-unit/participant relationship beyond the selected adjacent-unit checks below.
Selected English compounds also share whole values with digits: `twenty-five`
and `25`, or `one hundred and five` and `105`. Internal `and` differs from a
conjunction list such as `one and two`. English `minus`/`plus` and Russian
`минус`/`плюс` bind to the entire cardinal. Recognized numeric literals support
ASCII and Unicode minus, leading-point decimals and scientific notation without
float or decimal-context rounding. Ambiguous comma syntax and dotted numeric
identifiers retain literal signatures. The local writer freezes complete
recognized word-number spans. At a shared start, the longest phrase is frozen,
keeping full recognized names intact without splitting a larger cardinal on
name words seen elsewhere. Superlative `one` is excluded at a full-number boundary,
so `twenty-one of the best` retains `21`. This remains a selective lexical gate,
not a general quantity, range-order, arithmetic or semantic verifier.
An explicit English zero multiplier remains zero; a bare scale's implicit one
does not turn `zero thousand` into `1000`.

Explicit partial mode (`allow_partial=True`) keeps a rejected paragraph cluster
unchanged and continues with other prose. Results expose `retained_prose_spans`
and `token_watermark_action="partial_regeneration"`. If every prose cluster is
rejected, the call still raises without replacing the destination. This mode
catches rewrite `RuntimeError` only; unexpected exceptions still abort. It is not
a complete removal result. `remove_text_all` requires `include_invisible=True`
when requesting partial mode. Score retained segments separately: a lower whole-
document score can hide a marked passage.

An optional `rewrite_validator(source, candidate)` callback can reject a candidate
before publication. It receives visible text with protected inline content
restored, with outer whitespace stripped. It must return a literal `bool`, not a
dictionary, a truthy string or a confidence value. `False` retries up to three
candidate attempts, then raises `RuntimeError`; an invalid return type raises
`TypeError`. Exceptions propagate under the partial-mode rules above. The local
model's existing hard guards still run first. Caller-supplied rewriters retain
responsibility for their factual acceptance policy. Empty or unchanged custom
prose cannot be published; with a validator those candidates consume an attempt
without calling the validator. Protected-span fallback may call the rewriter and
validator on individual prose fragments. `remove_text_all` accepts the callback
only with `include_invisible=True`. No semantic judge is enabled by default.

For full-document context, pass `document_rewriter` to
`remove_text_invisible` or to `remove_text_all` with `include_invisible=True`.
It receives one tuple of trimmed, unlocked input spans in document order and
returns a list or tuple with one nonempty replacement per input span.
Positions distinguish repeated passages. The library restores the original
whitespace and protected content, checks protected structure before publication,
and counts identical spans as retained. A fully unchanged or malformed draft
raises `RuntimeError`. The document callback replaces passage generation and
cannot be combined with `rewriter` or `rewrite_validator`. The caller chooses
its model and fidelity policy.

Pass `document_validator(source, candidate)` to review the entire assembled
text with either generation path. It sees the originals of protected spans and
runs before the destination is written. A literal `False` vetoes that draft
even when `allow_partial=True`. An invalid verdict raises `TypeError`.

With both document callbacks set, `document_repairer(original, rejected)`
may offer one correction after a veto. Both arguments are tuples of trimmed,
unlocked prose spans in document order. Return a list or tuple of replacements
with the same shape, or return `None` to decline the repair. The library
assembles the correction under the same protection rules, rejects an unchanged
draft, and reviews the corrected document with the same validator. If the
correction fails, the destination is left intact. Callers can use the
validator's feedback to decide which drafts to repair.

`document_refiner(original, accepted)` receives the trimmed unlocked spans of
an accepted draft after any correction. Return replacements in the same order
to request a final pass through the structural and document validator checks,
or return `None` to keep the accepted draft. A refusal or exception leaves the
destination intact. Pass both `document_rewriter` and `document_validator` when
using a refiner.

All document callbacks are skipped if there is no prose to regenerate.
When combining metadata cleanup with regeneration, they operate on the
metadata-cleaned text.

For an explicitly configured Chat Completions endpoint, the optional
`remove-ai-watermarks[text-api]` extra provides generation and repair
callbacks. Supply a document reviewer separately:

```python
from remove_ai_watermarks import remove_text_invisible
from remove_ai_watermarks.text_api import build_text_api_document_rewriters

rewrite, repair = build_text_api_document_rewriters(
    base_url="http://127.0.0.1:11434/v1",
    model="your-compatible-model",
    max_completion_tokens=2048,
    completion_token_field="max_tokens",  # Ollama's Chat Completions field.
)

def process_document(input_path, output_path, review_document):
    return remove_text_invisible(
        input_path,
        output_path,
        document_rewriter=rewrite,
        document_validator=review_document,
        document_repairer=repair,
    )
```

The endpoint receives unlocked prose in order as JSON. A repair request
also includes the rejected prose. The endpoint receives the prose verbatim,
including its facts. Both requests specify the expected number of prose
slots. The adapter accepts that many strings in a JSON array, either
directly or inside one `json` code block. The library keeps protected
spans in the assembled document and the supplied reviewer
owns factual and semantic acceptance. An accepted initial draft
uses one POST. If the reviewer vetoes it, the configured correction adds
at most one POST, excluding any requests the reviewer itself makes.
Configuration, transport and credential handling match the passage API.

For the opt-in strict residual strategy, pass the callback mapping from
`build_text_api_strict_residual_rewriter`. It uses a fixed-length sentence
response, checks a rejected draft for sentences to restore locally, and makes at
most one second request for verbatim retained sentences. That second request
uses `residual_seed` and temperature 0.4. Every replacement in the second
request must differ from its source and pass the sentence fact check. The
assembled document must pass the whole-document guard.

```python
from remove_ai_watermarks import remove_text_invisible
from remove_ai_watermarks.text_api import build_text_api_strict_residual_rewriter

callbacks = build_text_api_strict_residual_rewriter(
    base_url="http://127.0.0.1:11434/v1",
    model="qwen3:8b-q4_K_M",
    seed=7291,
    completion_token_field="max_tokens",
)
remove_text_invisible(input_path, output_path, **callbacks)
```

Set `rewrite_style="recast"` to ask the model to change sentence openings
and clause structure in the initial draft. Repair requests still use
the fidelity correction instruction. The default style is `"standard"`.
Optional `temperature` and `seed` values pass through to the endpoint
for both requests. Omit them to use the endpoint's sampling settings.
Whether a seed makes output repeatable depends on the endpoint.

Set `rewrite_style="sentence"` to send all sentences in one request while
requiring one replacement per sentence in the same order. The adapter uses a
lightweight English sentence splitter and restores the original separators.
The caller still supplies the document reviewer.

For an endpoint that supports array-shaped JSON Schema output, set
`structured_output=True` to require a fixed-length array. Set
`disable_reasoning=True` if the endpoint supports `reasoning_effort="none"`.
Both settings apply to generation and correction. The adapter checks the reply
locally and surfaces endpoint errors without retrying with different settings.


For selected English loan-role reversals, an optional library helper can use a
caller-owned local spaCy-compatible parser:

```python
from functools import partial

import spacy

from remove_ai_watermarks.text import remove_text_invisible
from remove_ai_watermarks.text_roles import verify_text_rewrite_loan_roles

parser = spacy.load("en_core_web_trf", exclude=["ner"])
validator = partial(verify_text_rewrite_loan_roles, parser=parser)
result = remove_text_invisible("input.txt", rewrite_validator=validator)
```

Install the compatible English parser and its model separately. The helper does
not load or download a model, and the default text dependencies and processing
policy are unchanged. Select it only for English prose; it does not detect the
input language. It checks selected parsed `borrow`, `lend` and `loan`
clauses, including active/passive converses, against exact participant lemmas
and available item lemmas. It rejects missing/partial annotations and stale or
truncated parse text. Parser errors propagate. Its `True` result means no
supported reversal was found, not that meaning is preserved. Unmatched entities,
unresolved pronouns, omitted objects, timing, modality and general claims remain
outside this check. Lending does not establish ownership. Parser mistakes can
still hide reversals. Examined model findings are in
[`text-watermark-validation.md`](text-watermark-validation.md).

Service placeholder allocation avoids literal marker strings already in the
source, including inside locked code. User marker IDs are never parsed as
integers. This preserves literal marker-shaped content without namespace collisions.


Compatibility boundary: the legacy `score_gumbel_text`, `score_textgrain_text`,
`score_textseal2_text`, `score_kgw_text` and `score_unigram_text` helpers use this
module's BLAKE2 research PRF. They are simulations, not compatible reference
implementations of a vendor, MarkLLM or Meta deployment. A matching integer key
alone does not make their scores valid provider evidence. The independent
benchmark scripts use the external implementation's own detector instead.

`score_synthid_text` uses the optional torch/transformers reference processor,
including its context-repetition mask. Supply the generation's tokenizer, keys,
`ngram_len`, `sampling_table_seed`, `sampling_table_size`, `context_history_size`
and `device`. The default device is `cpu`; a table seeded on MPS or CUDA differs
from a CPU table with the same seed. The returned mean-g diagnostic z-score is
not the trained Bayesian detector or a provider's production detector. Thresholds
require calibration on matched marked and unmarked controls. Other named scorers
are local keyed simulations, not evidence of compatibility with private provider
implementations.

`score_synthid_threefry_text(ids, key, eos_token_ids=..., alpha=0.01)`
implements the [open single-key Threefry reference](https://github.com/google-deepmind/synthid-text/tree/72810caf309d51f5985dc871a4bce42c3d0cb54d)
in standard-library Python. Pass continuation tokens from the matching tokenizer
with the prompt removed. Set `ngram_len`, `depth`, `context_history_size` and
the generation's EOS token IDs to match the reference settings.
`SynthIDThreefryScore` reports the binomial upper-tail p-value, the numbers of
scored positions, one bits and trials, and the decision at `alpha`. Empty or
too-short sequences receive a p-value of 1.0. This scorer implements a distinct
format from the historical sampling-table detector and the dual-key mode.
Long histories and real text can violate the nominal independence assumption.
Calibrate against matching controls before drawing a watermark conclusion.

Selected adjacent physical measurements are checked as a multiset of normalized
amount/unit pairs and masked as whole phrases by the local rewriters. This covers
selected English/Russian units, case-sensitive symbols (`mW` differs from `MW`),
registered square-unit forms, slash compounds whose atoms are all recognized,
and the existing Russian duration adjectives. Numeric bounds include the unit
when recognized. Symbol/word aliases can compare equal; arithmetic conversion
such as `1000 m` to `1 km` is not implemented. Prefix currencies, unknown units,
unregistered products/powers, ambiguous unit words and source-to-role binding
remain outside this check. Short unit symbols can be ambiguous; English `in`
and Russian `г`/`с` are deliberately excluded. Quantity checks still do not
prove semantic equivalence or protect every measurement expression.

## Integration operation catalog

`remove_ai_watermarks.integration.operation_catalog()` derives typed operations
from every non-server CLI leaf, plus `api_draft_text` for the public API-only
OCR proposal feature. MCP, ComfyUI, and the Space consume the same catalog.
`input_schema(operation)` provides its JSON Schema; `widget_arguments` translates
optional visual controls without overriding profile defaults.
`run_operation(operation, arguments)` validates the payload, preserves the source,
and runs the selected operation in a child of the current interpreter. It returns
`exit_code`, `stdout`, `stderr`, and an output path only when a result was written.
Missing optional backends return installation guidance through the child report.
Credentials are passed on stdin and redacted from logs and returned diagnostics.
These adapters expose the same algorithms and hardware requirements as direct
CLI and Python calls; they do not provide a separate model implementation.
