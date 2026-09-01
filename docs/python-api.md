# Python API

Use the high level API for normal application integration. Low level detector
and pipeline modules are intended for maintainers and specialized workflows.

Dependency groups are identical for the CLI and Python API. The default install
covers metadata extraction, normalization, verdict logic, and stripping.
Array/pixel APIs use `pixels`; visible removal uses `visible`; DWT-DCT detection
uses `detect`; pixel photo classification uses `classify`; invisible image removal uses `qwen-zimage` and an NVIDIA GPU; and
visible video processing uses `video`. Video SynthID removal is a separate VAE
path that still runs on CPU and combines `video` and `diffusion`. Add `heif`
independently when path-based pixel APIs must decode HEIC, HEIF, or AVIF. See
the complete [feature-extra matrix](installation.md#feature-extras).

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

### Path input

For a path input, `remove_visible`:

- reads metadata provenance for the default `auto` sensitivity;
- preserves a separate alpha channel;
- writes the output when an output path is supplied;
- strips AI metadata from the written output by default;
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
print(result.invisible)       # "removed" | "no-signal" | "unavailable"
```

`invisible` is the field to check. `"unavailable"` means the GPU extra is not
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

If AI metadata survives the strip, `remove_all` raises `MetadataStripIncomplete`
**before** writing anything: an AI-readable output is worse than no output.

`remove_batch` runs one mode over a directory and never lets a single bad file
end the run:

```python
summary = raiw.remove_batch("in_dir", "out_dir", mode="visible")   # -> BatchSummary
print(summary.processed, summary.failed, summary.errors)
print(summary.invisible_unavailable)   # outputs that still carry the watermark
```

`mode` is `all`, `visible`, `invisible`, or `metadata`. Pass a constructed
`InvisibleEngine` as `engine` to load the model once for the whole directory.

## Classify a photograph from pixels

This is not provenance. `identify` does not call it. Install
`remove-ai-watermarks[classify]` first.

```python
from pathlib import Path

from remove_ai_watermarks.classify import classify_pixels

result = classify_pixels(Path("input.png"))
print(result.label, result.detector, result.provider)
```

`label` is `ai` only on a DEFINITELY detector result (ridge AND freeze MLP).
POSSIBLY is `unknown`. Camera-like photographs are `human`. `provider` is
`openai`, `google`, `muse-image`, or `tc260` only when `label` is `ai` and the 124-d
head beats `no_ai` by the freeze margin. `tc260` is the China AIGC label
standard (mixed producers), not a company. Otherwise it is `None`, including
when 124-d extraction refuses the file.

`device` is a library parameter: `None` / `"auto"` detect, `"cpu"` or `"cuda"`
pin. It is not a CLI option.

Missing extra raises `RuntimeError` with the quoted install command
`'remove-ai-watermarks[classify]'`. Guide:
[photo pixel classification](photo-classify.md).

## Inspect provenance

The default installation evaluates file metadata. Add `visible`, `detect`, or
`trustmark` to enable the corresponding optional pixel signals.

Get the vendor keys used by visible removal:

```python
import remove_ai_watermarks as raiw

vendors = raiw.visible_provenance("input.png")
```

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
metadata only, and registered visible marks and pixel-backed invisible watermarks
remain in the path-based `identify` call.

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
`surviving_markers` mapping as a failure.

`remove_ai_metadata` is the lower level fail-safe transformer. It may copy an
undecodable input through unchanged, so its return alone must not be presented
as proof that metadata was removed.

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
```

`identify_video` uses the same full-clip temporal arbiter as visible removal.
It reports a recurring registered mark and supported AI metadata as positive
signals. When neither is present, `is_ai_generated` is `None`, never `False`.
The absence of a public local video SynthID decoder is included in `caveats`.
Pass `check_visible=False` for a bounded metadata-only inspection.

For normal product integration, use the complete locally verifiable pipeline:

```python
result = raiw.remove_video_all("input.mp4", "clean.mp4")
if result.remaining_metadata:
    raise RuntimeError(f"AI metadata remains: {result.remaining_metadata}")
```

The default removes one stable supported visible provider mark when present,
always strips verified AI metadata, and writes a same-container output even
when neither signal is found. This gives callers one predictable output path.
It does not run lossy invisible regeneration by default.

`include_invisible=True` explicitly adds VAE regeneration for MP4, MOV, or M4V.
`VideoAllResult.invisible_removed` reports whether the oracle-certified SynthID
stage ran.

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
reuses one loaded VAE runtime across the batch.

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
before removal, and any markers remaining after the verified strip. MP4/MOV
inspection recognizes the native TC260 `AIGC` entry in
`moov.udta.meta.keys/ilst` and the QuickTime-form `meta` variants Doubao's iOS
export writes; its removal preserves container size and encoded
stream bytes. MP4/MOV/M4V are copied in bounded chunks, so a large `mdat` is not
loaded into memory; publication is atomic. MKV/WebM inspection recognizes the corresponding
`Segment.Tags.Tag.SimpleTag` representation; its removal requires ffmpeg for a
stream-copy remux. AVI inspection reads `LIST/INFO/AIGC`, and FLV inspection
reads `script.onMetaData.AIGC`; both use the same verified ffmpeg stream-copy
removal path.

## Remove video SynthID

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
across all frames, streams pixels to ffmpeg, copies complete audio, strips
source metadata, and publishes atomically. The default output is
`input_clean.mp4`; a distinct same-container output is required.

The returned `VideoInvisibleResult` includes output geometry, frame rate, frame
count, paired PSNR, and the motion-compensated temporal-residual ratio. Those
fields measure fidelity and flicker only. They are not a SynthID detector.
The default `noise_std=0.15` is the current full-clip oracle floor; `0.10`
remained detected on the public eight-second Veo calibration carrier.
The default profile is oracle-certified. Google does not publish a local
decoder for this video payload, so a fresh source-positive, output-negative
pair from Gemini's built-in SynthID verifier remains an optional per-file audit.
A response inferred from a visible logo or metadata is not such a verdict, and
an adversarial follow-up asking ordinary Gemini to reinterpret the verifier is
not a second oracle run.

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
`dola`, `hailuo`, `kling`). Provider confidence values are calibrated
independently and are not compared across detectors. Pass one of those explicit
values to restrict the scan to a single provider. The Veo detector recognizes
the current four-point diamond and the
legacy `Veo` text. Seedance recognizes the boxed `AI` label, Dola recognizes
its compact text label, Hailuo AI recognizes the composite MINIMAX/Hailuo AI label,
and Kling AI recognizes its bottom-right logo, wordmark, and version suffix. Each
variant has an independent synthetic silhouette and calibrated temporal policy.
After each accepted frame is filled, `temporal_consistency=True` motion-aligns
the preceding accepted fill and blends it only when the warped prior mask
covers the current mask and a surrounding source-context ring agrees. Scene
cuts and disjoint masks keep the independent current fill. Pass
`temporal_consistency=False` for the frame-local baseline.

The returned `VideoVisibleResult` records the selected `mark`, the total,
detected, and removed frame counts, plus any AI metadata that survived the
output encode. The function returns `output=None` and writes no file when no
stable mark is selected. Video pixels are transcoded through ffmpeg while the
complete source audio stream is copied. The encoder preserves supported 8-bit
source chroma sampling, color tags, MP4/MOV track timescale, and relative
variable-frame timestamps. It also retains a non-zero source start PTS and the
copied audio offset. A failed encode preserves any existing output; only a
completed result is published atomically.
SDR 8-bit video is the supported pixel contract. High-bit-depth, PQ, and HLG
sources raise `RuntimeError` before encoding instead of being silently reduced
to 8-bit SDR.

## Remove invisible watermarks

Install `remove-ai-watermarks[qwen-zimage]`. All three profiles need it, and all
need an NVIDIA GPU.

```python
from pathlib import Path

from remove_ai_watermarks.invisible_engine import InvisibleEngine

engine = InvisibleEngine(
    pipeline="qwen-zimage",  # the default; also "sdxl-zimage", "chroma-zimage", or "auto"
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
selects chroma-zimage for OpenAI and Microsoft provenance and qwen-zimage
otherwise, after the vendor is known and before strength resolution.

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
hash before loading the diffusion models. Qwen and Chroma reconstruct the donor with
the VAE already loaded for the one profile selected by `auto`; no second generative
profile runs. The engine rejects SDXL, downscaling, and postprocessing combinations
that were not evaluated. Tiling is also rejected because the combined
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
