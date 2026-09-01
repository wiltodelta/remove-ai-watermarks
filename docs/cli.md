# CLI guide

The command line interface is organized around the type of work you want to do.

```text
remove-ai-watermarks [OPTIONS] COMMAND [ARGS]
```

Run `remove-ai-watermarks COMMAND --help` for the complete option list and
defaults. This page focuses on choosing the right command.

## Command dependency map

| Command or signal | Required installation |
| --- | --- |
| `metadata` and metadata-only `identify` | Default package |
| `classify` | `remove-ai-watermarks[classify]` |
| Visible signals in `identify` | `remove-ai-watermarks[visible]` (`pixels` is the minimal runtime) |
| Open DWT-DCT signals in `identify` | `remove-ai-watermarks[detect]` |
| Adobe TrustMark signals in `identify` on Python 3.11-3.12 | `remove-ai-watermarks[trustmark]` |
| `visible` and `erase` with OpenCV | `remove-ai-watermarks[visible]` (`pixels` is the minimal runtime) |
| `visible` or `erase` with MI-GAN | `remove-ai-watermarks[migan]` |
| `visible` or `erase` with big-LaMa | `remove-ai-watermarks[lama]` |
| `invisible` and `all` (needs CUDA) | `remove-ai-watermarks[qwen-zimage]` |
| `video metadata` and `video identify --no-visible` | Default package |
| `video identify` | `remove-ai-watermarks[video]` |
| `video visible`, `video all`, and visible/all batch modes | `remove-ai-watermarks[video]` plus ffmpeg on PATH |
| `video invisible` and `video all --invisible` | `remove-ai-watermarks[video,diffusion]` plus ffmpeg on PATH |
| HEIC/HEIF/AVIF pixel input | Add `remove-ai-watermarks[heif]` |
| Every production command and Python-compatible backend | `remove-ai-watermarks[all]` |

`batch` requires the same extra as its selected mode. Extras can be combined in
one installation, for example `remove-ai-watermarks[visible,detect,heif]`.

## Inspect an image

```bash
remove-ai-watermarks identify image.png
```

`identify` always inspects supported metadata. When pixel extras are installed,
it also evaluates supported visible and invisible pixel signals. When no signal
is found, it reports the origin as unknown. It does not claim the image is
clean. For C2PA files, the text report shows asset integrity, claim-signature,
signer-trust, and signer-validity results separately. Confidence follows the
binding: an intact asset binding and claim signature is high confidence, and an
unanchored or expired signer is reported as a caveat rather than a lower score,
because no trust anchor list ships with the reader. A failed asset binding or
signature, or a revoked signing credential, does not confirm the claimed origin. When a structured
C2PA soft binding is present, the report also names its exact algorithm and
signed value; removing the manifest does not remove the referenced pixel
watermark or content fingerprint.

Machine readable output:

```bash
remove-ai-watermarks identify image.png --json
```

Metadata only inspection:

```bash
remove-ai-watermarks identify image.png --no-visible
```

Despite the historical option name, `--no-visible` skips both visible and open
invisible pixel detectors. Metadata inspection still runs.

## Classify a photograph from pixels

This is not provenance. `identify` never starts it, including after a
no-signal metadata scan. Install the extra first:

```bash
uv tool install --force "remove-ai-watermarks[classify]"
```

Then:

```bash
remove-ai-watermarks classify image.png
remove-ai-watermarks classify image.png --json
```

The command runs the frozen photo detector (CLIP-L-ft ridge AND freeze MLP).
Only a DEFINITELY result is reported as `ai`. On that result only, the 124-d
provider head may return `openai`, `google`, `muse-image`, or `tc260`. POSSIBLY is
`unknown`. Camera-like photographs are `human`. The call does not run cleanup
and does not set `is_ai_generated`.

The contract is AI versus camera. Receipts, UI, and digital art are out of
scope. Microsoft is not a pixel class: a DALL-E Bing image scores as `openai`,
an Imagen Designer image as `google`. `tc260` is the China AIGC label
standard, not one producer. Weights download on first use from
[`wiltodelta/raiw-models`](https://huggingface.co/wiltodelta/raiw-models),
or from `RAIW_CLASSIFY_WEIGHTS` if that directory already holds the freeze
files.
Guide: [photo pixel classification](photo-classify.md). Hub card:
[photo-classify-hf/README.md](photo-classify-hf/README.md).

## Remove known visible marks

Install `remove-ai-watermarks[visible]` before using `visible` or `erase`.

```bash
remove-ai-watermarks visible image.png -o clean.png
```

The default behavior:

- checks every registered visible mark;
- removes every detected match, except the weakly detected Jimeng label pill,
  which needs corroboration (see [supported signals](supported-signals.md));
- selects the best installed fill backend;
- strips AI metadata from the output.

Use a specific mark:

```bash
remove-ai-watermarks visible image.png --mark gemini -o clean.png
```

Available mark names are printed by:

```bash
remove-ai-watermarks visible --help
```

Keep metadata:

```bash
remove-ai-watermarks visible image.png --keep-metadata -o clean.png
```

Use the strict visual gate without metadata or sibling corroboration:

```bash
remove-ai-watermarks visible image.png --sensitivity strict -o clean.png
```

When no known mark is detected, the command does not write a new output. Use
`erase` if you can identify the affected region yourself.

## Erase a region

```bash
remove-ai-watermarks erase image.png \
  --region 1640,1930,400,100 \
  -o clean.png
```

The region format is `x,y,width,height`. Repeat `--region` to erase more than
one box:

```bash
remove-ai-watermarks erase image.png \
  --region 20,20,180,60 \
  --region 1640,1930,400,100 \
  -o clean.png
```

Choose the fill backend:

```bash
remove-ai-watermarks erase image.png \
  --region 1640,1930,400,100 \
  --backend migan \
  -o clean.png
```

`erase` accepts `cv2`, `migan`, and `lama`. The corresponding optional extra
must be installed for a learned backend.

Two more knobs tune the fill. `--dilate N` (default 3) grows every box by `N`
pixels before inpainting, which helps when a mark has a soft edge or a drop
shadow just outside the box you measured; it applies to every backend because it
shapes the mask. `--inpaint-method telea|ns` selects the classical algorithm and
only affects the `cv2` backend. Like `visible`, `erase` strips AI metadata from
the output by default; pass `--keep-metadata` to retain it.

## Strip AI metadata

Inspect metadata:

```bash
remove-ai-watermarks metadata image.png --check
```

Remove AI metadata and write a new file:

```bash
remove-ai-watermarks metadata image.png --remove -o clean.png
```

When `-o` is omitted, removal overwrites the source. Standard metadata is kept
unless you pass `--remove-all`.

A quiet `--check` or a successful `--remove` is not a clean verdict. The command
only inspects and strips embedded AI metadata; a pixel watermark such as SynthID
has no local decoder once that metadata proxy is gone. `identify` reports the
same limit.

The command also supports the audio and video containers listed in
[supported signals](supported-signals.md). ffmpeg must be available for the
non-ISOBMFF audio and video path.

## Identify and clean video

Install the video pixel and timestamp runtime for visible identification,
removal, and the complete pipeline:

```bash
uv tool install --force "remove-ai-watermarks[video]"
```

Inspect every locally supported video signal:

```bash
remove-ai-watermarks video identify input.mp4
remove-ai-watermarks video identify input.mp4 --json
remove-ai-watermarks video identify input.mp4 --no-visible
```

The default scans the complete clip for stable registered visible marks and
inspects supported metadata. A result with no signals is reported as unknown,
not clean, because proprietary pixel watermarks have no public local decoder.
`--no-visible` performs metadata-only inspection.

Use the complete locally verifiable cleaning path:

```bash
remove-ai-watermarks video all input.mp4 -o clean.mp4
```

It removes a stable supported visible mark when found and always strips
verified AI metadata. When neither signal is found, it writes a same-container
passthrough instead of returning a missing output. The source is never
overwritten.

Invisible regeneration is deliberately opt-in:

```bash
remove-ai-watermarks video all input.mp4 -o clean.mp4 --invisible
```

That option is supported only for MP4, MOV, and M4V. It is lossy and uses the
same oracle-certified profile as `video invisible`.

Process all supported files in a top-level directory:

```bash
remove-ai-watermarks video batch ./videos --mode all
remove-ai-watermarks video batch ./videos --mode visible
remove-ai-watermarks video batch ./videos --mode metadata
```

The batch runs sequentially, preserves successful outputs when another file
fails, and exits nonzero if any item failed. Visible no-op files are copied
byte-for-byte so the output directory remains complete. `--invisible` is
available only with `--mode all`.

## Strip AI metadata from video

Metadata inspection and removal are also available as an isolated operation:

```bash
remove-ai-watermarks video metadata input.mp4 --check
remove-ai-watermarks video metadata input.mp4 --remove -o clean.mp4
```

As with the generic `metadata` command, a quiet check or a successful strip is
not a clean verdict: video SynthID is not decoded locally after the metadata
proxy is gone.

Supported containers are MP4, MOV, M4V, WebM, MKV, AVI, and FLV. The operation
delegates to the same verified metadata scanner and stripper as the generic
`metadata` command, so detection and removal stay in parity. Video and audio
streams are not transcoded. For MP4 and MOV, this includes the native TC260
`AIGC` key and JSON value stored in `moov.udta.meta.keys/ilst`, plus the
QuickTime-form `meta` variants Doubao's iOS export writes (a bare `meta` box
as a direct `moov` child, and a keyless `hdlr=mdir` metadata list). The inspector
seeks past a large `mdat` to find a tail `moov`. Removal stream-copies the
container in bounded chunks, converts supported top-level provenance boxes to
same-size `free` boxes, and blanks the TC260 key/value in place. Box sizes,
media offsets, and encoded stream bytes do not move; the result is atomically
published only after the complete copy succeeds.

For MKV and WebM, the inspector reads the native TC260
`Segment.Tags.Tag.SimpleTag` entry. Removal uses ffmpeg stream copying to
discard container tags and chapters without transcoding the streams.
AVI uses the normative `LIST/INFO/AIGC` chunk, while FLV uses the
`script.onMetaData.AIGC` AMF0 string. Their bounded readers skip media payloads,
and removal also uses ffmpeg stream copying.

When `-o` is omitted, the command writes `<source>_clean` with the same
extension. It never overwrites the source, and it rejects an output with a
different container extension.

Visible video labels and invisible video watermarks are not handled by this
command.

## Remove video SynthID

```bash
uv tool install --force "remove-ai-watermarks[video,diffusion]"
remove-ai-watermarks video invisible input.mp4 -o clean.mp4
```

The command supports MP4, MOV, and M4V. It samples the complete
sequence at the configured frame rate, resizes frames to the configured long
side, regenerates them through a VAE, and applies one deterministic latent-noise
field to every frame. Reusing one spatial field avoids the unnecessary flicker
caused by independent per-frame noise. Frames are regenerated in bounded
batches and streamed directly to ffmpeg, which encodes the result, copies
audio, and drops source metadata.

The default `noise_std=0.15` profile is oracle-certified. The project has no
local video SynthID decoder, so an optional per-file recheck is still useful
for unusually important files or after provider changes. In a new Gemini chat,
upload the original first, invoke the built-in verifier with `@synthid`, and ask:

> For the video attached to this message, was it created or edited by Google
> AI? Use the built-in SynthID content verification result.

The source must be positive. Then upload the processed result in a separate new
chat and repeat the same built-in check. Only a source-positive, output-negative
pair is a fresh per-file verification. Do not ask an adversarial follow-up that tells the
chat model to ignore the verifier and reason about raw pixels: that is ordinary
Gemini reasoning, not a second oracle check.

The default output is `<source>_clean` in the same container. The
source is never overwritten. Use `--noise-std`, `--long-side`, `--fps`,
`--batch-size`, `--seed`, and `--device` to control the regeneration. The
default noise level is `0.15`. It cleared both carriers in the 2026-07-29
short-clip calibration and the complete public eight-second Veo carrier in the
2026-07-31 full-clip check; `0.10` remained detected on that complete clip. This
calibration certifies the shipped operating point; the paired check above is an
optional runtime audit, not a separate result state.

## Remove a supported visible video mark

```bash
remove-ai-watermarks video visible input.mp4 -o clean.mp4
remove-ai-watermarks video visible veo.mp4 --mark veo -o veo_clean.mp4
remove-ai-watermarks video visible seedance.mp4 --mark seedance -o seedance_clean.mp4
remove-ai-watermarks video visible dola.mp4 --mark dola -o dola_clean.mp4
remove-ai-watermarks video visible hailuo.mp4 --mark hailuo -o hailuo_clean.mp4
remove-ai-watermarks video visible kling.mp4 --mark kling -o kling_clean.mp4
```

The command supports the moving Sora mascot and wordmark, two Veo
corner variants, the Seedance boxed `AI` label, the `Dola AI` text label, the
composite `MINIMAX | hailuo AI` label, and the bottom-right Kling AI label. Sora
searches the whole frame at multiple scales. The other detectors search bounded
lower-frame regions with separate synthetic silhouettes. Kling additionally
requires its bright low-saturation label near the frame edge. Every mark
requires a spatially recurring candidate across adjacent frames. Fixed marks
must also remain anchored instead of drifting with a scene object. Matching
provider provenance may relax the visual score only for registered
provenance-aware marks; metadata alone never creates a detection.

`--mark auto` is the default. It evaluates all providers in one decode pass and
selects the first stable match in specificity order: Sora, Veo, Seedance,
Doubao, Dola, Hailuo AI, then Kling AI. Their confidence scores are independently calibrated and
are not compared across providers. Pass an explicit `--mark` to scan only that
provider.

The video stream is transcoded and the complete original audio stream is
copied without truncating an audio tail that extends beyond the final video
frame. The encoder probes the source stream and preserves supported 8-bit
chroma sampling, color range/matrix/transfer/primaries tags, and MP4/MOV track
timescale. For a variable-frame-rate source, decoded PTS are carried through a
timestamped in-memory NUT bridge so the output retains the source frame
intervals instead of flattening them to a constant rate. A non-zero source
start PTS and the copied audio start offset are preserved as well.
Supported input and output containers are MP4, MOV, M4V, WebM, MKV, AVI, and
FLV; the output extension must match the input. The default `cv2` backend is
fast but can smear structured backgrounds. Select `--backend migan` or
`--backend lama` for a learned fill, or `--backend auto` to choose the best
installed backend.

`--temporal-consistency` is enabled by default. It motion-aligns the preceding
accepted fill, requires overlapping removal masks and matching source context,
and blends only the safely covered pixels. Scene cuts, disjoint moving marks,
or a poor motion match keep the independent current-frame fill. Use
`--no-temporal-consistency` for an exact frame-local baseline.

The pixel path is intentionally limited to SDR 8-bit video. A high-bit-depth,
PQ, or HLG source is rejected before ffmpeg starts, preserving any existing
output instead of silently downconverting it through OpenCV's 8-bit boundary.
On CPU, MI-GAN is the practical learned tier. LaMa remains an explicit offline
quality option because full-sequence inference is too slow and memory-heavy
for an online worker.

AI metadata is stripped from the encoded output by default. Use
`--keep-metadata` to retain mapped container metadata. When no temporally stable
mark is found, the command writes no output and exits with the no-visible-mark
status. The final path is replaced atomically only after ffmpeg completes, so a
failed encode does not overwrite an existing result.

## Remove invisible watermarks

Install the removal dependencies first. All profiles are CUDA-only and all
run the DiffSynth Z-Image face stage, so this is the extra either one needs:

```bash
uv tool install --force "remove-ai-watermarks[qwen-zimage]"
```

Then run:

```bash
remove-ai-watermarks invisible image.png -o clean.png
```

The command normally skips regeneration when no supported local signal is
detected. Use `--force` when you know the image should be processed:

```bash
remove-ai-watermarks invisible image.png -o clean.png --force
```

### Choose a strength cohort

`--vendor` selects the cohort the default strength resolves from. `auto`
(the default) derives it from provenance: the C2PA issuer for
OpenAI/Google/Microsoft, and the standalone AI IPTC tag for Meta Content Seal
(Muse output carries no C2PA; the tag is a standard code, so C2PA evidence
always wins first). An explicit value both names the cohort for stripped files
and implies the scrub runs -- naming the cohort asserts the pixel watermark is
present -- exactly like `--force` plus a measured floor:

```bash
remove-ai-watermarks invisible muse_output.webp -o clean.png --vendor meta
```

The same option exists on `all` and `batch`, and as
`InvisibleOptions(vendor="meta")` in the Python API.

### Choose a pipeline

| Pipeline | When to use it |
| --- | --- |
| `qwen-zimage` | Default. Qwen-Image-2512 global pass plus a SAM-masked Z-Image face stage |
| `sdxl-zimage` | The same recipe and face stage on an SDXL global pass, at a higher denoise |
| `chroma-zimage` | The same face stage on an Apache-2.0 Chroma1 global pass, with lower OpenAI and Microsoft floors and higher Google and Meta floors |
| `auto` | Pick the engine from provenance: chroma-zimage for OpenAI and Microsoft, qwen-zimage for Google, Meta, and unknown |

**All four are CUDA-only.** There is no CPU or MPS profile for invisible-watermark
removal. The former `controlnet`, `sdxl`, `qwen` and `default` profiles were removed
rather than kept as a CPU path: none of them matched this recipe's face preservation,
so offering them implied a quality the library no longer delivers. Passing a retired
name is rejected at parse time rather than remapped. Visible-mark removal and every
identify path still run anywhere.

Example:

```bash
remove-ai-watermarks invisible image.png -o clean.png \
  --pipeline qwen-zimage --force
```

There is no `--model`, `--steps`, `--guidance-scale` or `--device` option, and the
deprecated `--auto` is gone. Each profile pins its model stack, its per-stage
schedule, CFG 1.0 and CUDA, so every one of those flags existed only to be refused
several layers down. They are not parsed at all now, which fails at the point the
user can act on rather than after a model load.

### Restore operator-verified text

`--text-manifest` enables the experimental `vae-glyphs` post-pass. It reconstructs
the source with the selected profile's VAE, erases the annotated candidate glyphs
with LaMa, and composites only the reconstructed glyph cores through source-derived
silhouettes. It does not run OCR or choose which strings are correct. The optional
Qwen-only `--fidelity-anchor` described below additionally blends 15% of the
reconstruction across the full frame.

Install the combined extra and run only with an operator-verified manifest:

```bash
uv tool install --force "remove-ai-watermarks[text-restoration]"
remove-ai-watermarks invisible image.png -o clean.png \
  --pipeline auto --text-manifest verified-lines.json --force
```

``verified: true`` may also be set by an automated operator that attests
machine-verified geometry: stability-gated detector boxes inside sane caps. Such
operators should use the geometry-only schema 2, which carries no transcription
or script metadata.

Since 0.27.1 the global 15% Qwen-VAE fidelity-anchor blend is off by default and
remains available only with `qwen-zimage`: it
was measured to return detector-visible OpenAI SynthID on poster-scale manifests
(official Content Provenance API, 2026-08-19). `--fidelity-anchor` restores the
0.27.0 research behavior; text-box fidelity lost by the default is well under one
MAE point on the measured fixtures.

The manifest is a JSON object with `verified: true`, decoded RGB dimensions,
`source_pixel_sha256`, and a non-empty `lines` array. Schema 1 is retained for
manually reviewed annotations: each line has an integer `[x1, y1, x2, y2]` box,
exact `text`, a non-empty `script`, and an optional angle from -30 to 30 degrees.
Schema 2 is geometry-only: each line has the box and optional angle, with no
required `text` or `script`. Lines must be in top-to-bottom, left-to-right order.
The hash binds the annotations to decoded RGB geometry and pixels, so metadata-only
container changes remain valid while a resized or edited source fails closed. The
experimental helper
`remove_ai_watermarks._internal.text_restoration.source_pixel_sha256` computes it.

This mode is supported by `qwen-zimage`, `chroma-zimage`, and `auto` at native
geometry with `humanize=0`, `unsharp=0`, and adaptive polish disabled. The legacy
`sdxl-zimage` profile does not expose a verified-text VAE donor. `all` also accepts the flag,
but its manifest must match the pixels entering the invisible stage; if visible-mark
removal changes those pixels, the hash check rejects the run. One oracle verdict does
not certify another manifest, seed, model/runtime version, or output hash.

### Work with limited memory

Lower CUDA memory pressure:

```bash
remove-ai-watermarks invisible image.png -o clean.png \
  --cpu-offload --force
```

Keep large images at native resolution while processing them in overlapping
tiles:

```bash
remove-ai-watermarks invisible image.png -o clean.png \
  --tile --max-resolution 0 --force
```

Or set a resolution cap:

```bash
remove-ai-watermarks invisible image.png -o clean.png \
  --max-resolution 2048 --force
```

Tiling avoids the explicit downscale but each tile is regenerated separately.
It is a memory strategy, not a guarantee of better quality.

## Run the full pipeline

The `all` command and the `all` installation extra are separate concepts. The
command runs every applicable stage. Installing `remove-ai-watermarks[all]`
makes every backend compatible with the active Python available; a smaller installation such as
`remove-ai-watermarks[visible,qwen-zimage]` can also run the command with fewer
optional backends.

```bash
remove-ai-watermarks all image.png -o clean.png
```

The command runs:

1. visible mark removal;
2. invisible watermark removal when available and applicable;
3. AI metadata stripping.

The visible options and diffusion options are also available on `all`.

When the `qwen-zimage` extra is unavailable, `all` still writes the result of
the visible and metadata stages, prints a prominent warning, and exits with code
1. That happens on every run without the extra: the skipped stage is what would
have decided whether a signal was there. This prevents a partial result from
being reported as complete.

If the extra is installed but the machine has no CUDA, which is the usual macOS
case, the run fails at engine construction instead: `all` prints
`Error: Invisible-watermark removal is CUDA-only ...`, writes no output at all,
and exits with code 1.

## Process a directory

```bash
remove-ai-watermarks batch ./images --mode visible
```

Modes:

- `visible`;
- `invisible`;
- `metadata`;
- `all`.

Set an output directory:

```bash
remove-ai-watermarks batch ./images \
  --mode all \
  --output-dir ./clean
```

The invisible and full modes accept the same main diffusion controls as their
single image counterparts. Run `batch --help` for the authoritative option
list.

## Exit behavior

The CLI uses nonzero exit codes for meaningful incomplete outcomes, including
no detected target on commands that would otherwise regenerate or create a
misleading unchanged result, processing errors, and a required invisible step
that could not run.

Scripts should check the process exit code and the output path. The detailed
per-command contract is maintained in
[module internals](module-internals.md#command-line-interface).
