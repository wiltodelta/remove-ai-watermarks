# Remove AI Watermarks

Remove AI provenance marks from images and video you generated yourself:

- known visible labels such as the Google Gemini sparkle watermark and vendor text marks;
- invisible pixel watermarks through diffusion regeneration;
- C2PA, EXIF, XMP, IPTC, and related AI metadata.

Video support covers provenance identification, complete visible-plus-metadata
cleaning, directory batches, visible Sora, Veo, Seedance, Dola, Hailuo AI, and
Kling AI mark removal, and oracle-certified VAE regeneration for video SynthID
removal.

> [raiw.cc](https://raiw.cc) runs this library as a hosted service, with the GPU
> included and nothing to install. Visible mark and metadata removal at Standard
> output up to 12 MP are free there; original resolution above 12 MP and invisible
> watermark removal are paid.

[![PyPI](https://img.shields.io/pypi/v/remove-ai-watermarks?logo=pypi&logoColor=white)](https://pypi.org/project/remove-ai-watermarks/)
[![Python](https://img.shields.io/pypi/pyversions/remove-ai-watermarks?logo=python&logoColor=white)](https://pypi.org/project/remove-ai-watermarks/)
[![Downloads](https://static.pepy.tech/badge/remove-ai-watermarks/month)](https://pepy.tech/project/remove-ai-watermarks)
[![License](https://img.shields.io/pypi/l/remove-ai-watermarks?color=blue)](LICENSE)
[![Tests](https://github.com/wiltodelta/remove-ai-watermarks/actions/workflows/test.yml/badge.svg)](https://github.com/wiltodelta/remove-ai-watermarks/actions/workflows/test.yml)
[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-db61a2?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/wiltodelta)

> This project is for lawful use on content you own. It does not target stock
> agency previews or other watermarks that protect third party paid content.
> See [scope, safety, and legal notes](docs/legal-and-safety.md).

## Choose what you want to do

| Goal | Command | GPU |
| --- | --- | --- |
| Find provenance signals and watermarks | `identify` | No |
| Classify a photograph from pixels (opt-in, not provenance) | `classify` | No |
| Verify supported OpenAI SynthID from pixels with the official remote API | `verify-openai-synthid` | No |
| Remove known visible AI marks | `visible` | No |
| Erase a region you select | `erase` | No |
| Strip AI metadata | `metadata` | No |
| Identify supported video provenance | `video identify` | No |
| Remove visible marks and AI metadata from video | `video all` | No |
| Strip AI metadata from video | `video metadata` | No |
| Remove a registered visible AI mark from video | `video visible` | No |
| Process a directory of videos | `video batch` | Depends on mode |
| Remove video SynthID with the certified VAE profile | `video invisible` | Recommended |
| Regenerate an image to disrupt invisible watermarks | `invisible` | Required (CUDA) |
| Run visible, invisible, and metadata removal | `all` | Recommended |
| Process a directory | `batch` | Depends on mode |

Microsoft Paint and Photos InvisMark declarations are routed automatically to
pixel regeneration. The `all` command removes both the hidden pixel watermark and
its linked C2PA manifest; metadata stripping alone removes only the manifest.

## Installation modes

| Need | Install |
| --- | --- |
| Metadata inspection and stripping | `remove-ai-watermarks` |
| Photograph AI-versus-camera classification | `remove-ai-watermarks[classify]` |
| Official remote OpenAI SynthID verification | `remove-ai-watermarks[verify]` |
| Visible detection and removal | `remove-ai-watermarks[visible]` |
| Visible video processing | `remove-ai-watermarks[video]` |
| Video SynthID removal | `remove-ai-watermarks[video,diffusion]` |
| Torch-free DWT-DCT detection | `remove-ai-watermarks[detect]` |
| Invisible image removal (needs CUDA) | `remove-ai-watermarks[qwen-zimage]` |
| Every production feature available on the active Python | `remove-ai-watermarks[all]` |

Lower-level and specialized extras include `pixels`, `heif`, `trustmark`,
`migan`, `lama`, and `diffusion`. The
[installation guide](docs/installation.md#feature-extras) documents their exact
dependency composition, Python compatibility, and model requirements.

## Quick start

Install the metadata-focused default CLI:

```bash
uv tool install remove-ai-watermarks
```

Inspect an image:

```bash
remove-ai-watermarks identify image.png
```

To classify a photograph from pixels (AI versus camera, optional provider),
install the extra and call `classify`. `identify` never starts it:

```bash
uv tool install --force "remove-ai-watermarks[classify]"
remove-ai-watermarks classify image.png
```

Guide: [photo pixel classification](docs/photo-classify.md).

Signed provenance is the supported route for SynthID and `identify` reads it.
There is no local SynthID pixel detector in the package. Research on a
periodic lattice expert is in `scripts/synthid_runtime/` and
[synthid-detector-research.md](docs/synthid-detector-research.md).

For supported OpenAI images, the optional official verifier provides a pixel
watermark verdict:

```bash
uv tool install --force "remove-ai-watermarks[verify]"
remove-ai-watermarks verify-openai-synthid image.png --acknowledge-upload
```

The command removes AI provenance metadata from a temporary copy, verifies that
the decoded pixels are unchanged, uploads only that copy to OpenAI, and consumes
only the independent SynthID response. It never runs implicitly from `identify`.
An API key, endpoint access, and explicit upload acknowledgement are required.

For visible watermark removal, install the pixel dependencies:

```bash
uv tool install --force "remove-ai-watermarks[visible]"
```

Then remove a known visible mark and AI metadata:

```bash
remove-ai-watermarks visible image.png -o clean.png
```

Strip metadata without running visible inpainting or diffusion:

```bash
remove-ai-watermarks metadata image.png --remove -o clean.png
```

Without `-o` this command overwrites the source in place.

Inspect or remove AI metadata from an MP4, MOV, M4V, WebM, MKV, AVI, or FLV
file:

```bash
remove-ai-watermarks video metadata input.mp4 --check
remove-ai-watermarks video metadata input.mp4 --remove -o clean.mp4
```

The `video metadata` command does not transcode video or audio streams. Unlike
the image command above, when `-o` is omitted it writes `<source>_clean` and
preserves the original. MP4 and MOV
inspection includes the native TC260 `AIGC` tag in
`moov.udta.meta.keys/ilst`, including a `moov` placed after the media payload,
plus the QuickTime-form `meta` variants Doubao's iOS export writes (a bare
`meta` box as a direct `moov` child, and a keyless `hdlr=mdir` metadata list).
MKV and WebM inspection reads the normative
`Segment.Tags.Tag.SimpleTag` placement. AVI uses `LIST/INFO/AIGC`, while FLV
uses `script.onMetaData.AIGC`. The non-ISOBMFF formats are remuxed with stream
copy for removal.

Use the product-oriented video path to identify or clean a file:

```bash
uv tool install --force "remove-ai-watermarks[video]"
remove-ai-watermarks video identify input.mp4
remove-ai-watermarks video all input.mp4 -o clean.mp4
```

`video all` removes a stable registered visible mark when present and always
strips verified AI metadata. If neither signal is found, it still writes a
same-container passthrough, so application callers get one predictable output
contract. Proprietary invisible-video removal is excluded by default.
`--invisible` opts into the lossy, oracle-certified video SynthID profile.

Process a directory with the same contract:

```bash
remove-ai-watermarks video batch ./videos --mode all
```

Remove a supported visible video mark:

```bash
remove-ai-watermarks video visible input.mp4 -o clean.mp4
remove-ai-watermarks video visible veo.mp4 --mark veo -o veo_clean.mp4
remove-ai-watermarks video visible seedance.mp4 --mark seedance -o seedance_clean.mp4
remove-ai-watermarks video visible dola.mp4 --mark dola -o dola_clean.mp4
remove-ai-watermarks video visible hailuo.mp4 --mark hailuo -o hailuo_clean.mp4
remove-ai-watermarks video visible kling.mp4 --mark kling -o kling_clean.mp4
```

This path scans the complete sequence before changing pixels. It accepts only a
mark that repeats at a stable position across adjacent frames, then reuses the
same OpenCV, MI-GAN, or LaMa fill backends as image removal. Audio is copied
without re-encoding and is allowed to reach its natural end; the video stream
is transcoded because its pixels change. By default, a guarded optical-flow
pass motion-aligns the preceding accepted fill and blends it only when the
nearby source context agrees; use `--no-temporal-consistency` to disable it.
The encoder preserves supported
8-bit source chroma sampling, color tags, and MP4/MOV track timescale instead
of relying on ffmpeg's implicit raw-BGR defaults. Variable frame intervals are
preserved through a timestamped in-memory NUT bridge instead of being flattened
to the average frame rate. Non-zero source start timestamps are retained
together with the copied audio offset. The default `--mark auto`
scans all providers in one decode pass and selects the first stable match in
the specificity order shown below. Pass an explicit mark to restrict detection
to one provider.
Sora covers the moving Sora 2 mascot and wordmark. Veo covers both the current
four-point diamond and the legacy `Veo` text. Seedance covers the fixed boxed
`AI` label, Dola covers the fixed `Dola AI` text, Hailuo AI covers the composite
`MINIMAX | hailuo AI` label, and Kling AI covers the bottom-right `KLING AI`
label with its version suffix. A completed encode is published atomically. No
output is written when no stable mark is found.
HDR, PQ/HLG, and greater-than-8-bit inputs are rejected before encoding rather
than silently reduced through OpenCV's 8-bit BGR boundary.

Remove video SynthID:

```bash
uv tool install --force "remove-ai-watermarks[video,diffusion]"
remove-ai-watermarks video invisible input.mp4 -o clean.mp4
```

This path regenerates the complete sequence with one latent-noise field shared
across time, copies complete audio, strips source metadata, and publishes the
completed encode atomically. The default `noise_std=0.15` profile passed both
the two-carrier calibration and a complete public eight-second Veo oracle
check. Google does not publish a local decoder, so a fresh provider check
remains useful for unusually important files or after provider changes, but it
is not a product result state.

For invisible watermark removal, install the `qwen-zimage` extra. **An NVIDIA GPU
is required**: all profiles are CUDA-only, and there is no CPU or MPS fallback.

```bash
uv tool install --force "remove-ai-watermarks[qwen-zimage]"
remove-ai-watermarks invisible image.png -o clean.png
```

If the local detectors cannot confirm an invisible watermark but you know the
image came from an AI generator, add `--force`:

```bash
remove-ai-watermarks invisible image.png -o clean.png --force
```

Typography-heavy images can opt into the experimental verified-text post-pass.
It accepts manually reviewed strings and line boxes or an operator-verified
geometry-only manifest; it never treats raw OCR output as ground truth or runs
automatically:

```bash
uv tool install --force "remove-ai-watermarks[text-restoration]"
remove-ai-watermarks invisible image.png -o clean.png \
  --text-manifest verified-lines.json --force
```

See the [CLI guide](docs/cli.md#restore-operator-verified-text) for the manifest
schema, compatibility restrictions, and oracle caveats.

See the [installation guide](docs/installation.md) for Homebrew, uv, optional
features, and development setup.

## Examples

### Visible Gemini mark

| Before | After |
| --- | --- |
| ![Image with a visible Gemini watermark](demo_banana_before.png) | ![Image after visible watermark removal](demo_banana_after.png) |

The `after` raster is generated from the tracked `before` raster by the public path:

```bash
uv run remove-ai-watermarks visible demo_banana_before.png \
  --backend cv2 -o demo_banana_after.png
```

### High quality invisible removal

`qwen-zimage` is the default profile: a Qwen-Image-2512 Lightning pass under Canny
ControlNet, followed by SAM-masked Z-Image repair of any detected face. The
alternative, `sdxl-zimage`, swaps the global stage for SDXL and keeps the same face
stage. A third profile, `chroma-zimage`, uses the Apache-2.0 Chroma1 global pass
with its own flat vendor floors; see `docs/chroma1-engine-research.md` for the
calibration. `--pipeline auto` picks chroma-zimage for OpenAI and Microsoft
provenance and qwen-zimage otherwise. All are CUDA only.

```bash
uv tool install --force "remove-ai-watermarks[qwen-zimage]"
remove-ai-watermarks invisible image.png -o clean.png --force
```

| OpenAI example before | OpenAI example after |
| --- | --- |
| [![OpenAI portrait grid before qwen-zimage](data/synthid/originals/ChatGPT%20Image%20May%2030,%202026,%2010_31_08%20AM.png)](data/synthid/originals/ChatGPT%20Image%20May%2030,%202026,%2010_31_08%20AM.png) | [![OpenAI portrait grid after qwen-zimage](docs/images/qwen-zimage/ChatGPT/ChatGPT%20Image%20May%2030,%202026,%2010_31_08%20AM_full_clean.png)](docs/images/qwen-zimage/ChatGPT/ChatGPT%20Image%20May%2030,%202026,%2010_31_08%20AM_full_clean.png) |

| Gemini example before | Gemini example after |
| --- | --- |
| [![Gemini sign before qwen-zimage](data/synthid/originals/Gemini_Generated_Image_633uuy633uuy633u.png)](data/synthid/originals/Gemini_Generated_Image_633uuy633uuy633u.png) | [![Gemini sign after qwen-zimage](docs/images/qwen-zimage/Gemini/Gemini_Generated_Image_633uuy633uuy633u_full_clean.png)](docs/images/qwen-zimage/Gemini/Gemini_Generated_Image_633uuy633uuy633u_full_clean.png) |

These exact output files were checked with the matching provider verifiers. That
result applies to these files, not to every seed, image, or future watermark
version.

## Common recipes

### Remove every detected visible mark

```bash
remove-ai-watermarks visible image.png -o clean.png
```

The default `--mark auto` checks all registered visible marks and removes every
match. If the mark is visible to you but the detector misses it, select its
region explicitly:

```bash
remove-ai-watermarks erase image.png \
  --region 1640,1930,400,100 \
  -o clean.png
```

`--region` uses `x,y,width,height` and may be repeated.

### Use a learned fill backend

The `visible` extra uses OpenCV inpainting when no learned backend is installed.
For more difficult backgrounds, the learned-backend extras include the same
pixel dependencies automatically:

```bash
uv tool install --force "remove-ai-watermarks[migan]"
remove-ai-watermarks visible image.png -o clean.png --backend migan
```

```bash
uv tool install --force "remove-ai-watermarks[lama]"
remove-ai-watermarks visible image.png -o clean.png --backend lama
```

### Reduce CUDA memory use

```bash
remove-ai-watermarks invisible image.png -o clean.png \
  --cpu-offload --force
```

CPU offload lowers CUDA memory pressure by moving model components between CPU
and GPU, at the cost of speed.

### Process a directory

```bash
remove-ai-watermarks batch ./images --mode visible
remove-ai-watermarks batch ./images --mode all
```

## What the tool can recognize

Visible mark support includes:

- Google Gemini and Nano Banana visible sparkle watermark;
- Doubao, the Jimeng wordmark and top-left `AI生成` pill, Qwen, Kling AI,
  Yuanbao, Baidu, LiblibAI, and RunningHub labels;
- one calibrated Microsoft top-right white AI-badge variant;
- one calibrated Samsung Galaxy AI label variant.

Metadata and provenance inspection covers C2PA, EXIF, XMP, IPTC, common
generator parameters, China TC260 AIGC labels, and several vendor specific
signals. Optional decoders add support for open DWT-DCT watermarks and Adobe
TrustMark.

The exact support matrix, including important locale and detector limits, lives
in [supported signals](docs/supported-signals.md). The
[synthetic example gallery](data/fixtures/visible/README.md) carries one canonical
sample for every registered image and video mark.

## How it works

Visible removal follows three steps:

1. Detect a registered mark in its expected area.
2. Build a mask around the mark.
3. Fill only the masked region with OpenCV, MI-GAN, or LaMa.

Metadata removal uses format aware stripping. JPEG metadata removal preserves
the encoded image scan instead of recompressing it. Native MP4/MOV TC260 values
are blanked without changing box sizes or media offsets. Other supported
containers use their corresponding metadata path.

Invisible removal is different. It regenerates the image through a diffusion
pipeline to disrupt pixel and frequency domain watermarks. This changes the
image and cannot guarantee that a proprietary verifier will reject every
output.

See [supported signals](docs/supported-signals.md) and
[known limitations](docs/known-limitations.md) for the full technical boundary.

## Python API

The visible-removal API requires `remove-ai-watermarks[visible]`.

```python
import remove_ai_watermarks as raiw

result, removed = raiw.remove_visible("watermarked.png", "clean.png")
print(removed)

openai_synthid = raiw.verify_openai_synthid("image.png", acknowledge_upload=True)
print(openai_synthid.status)

provenance = raiw.identify_video("input.mp4")
report = raiw.inspect_video_metadata("input.mp4")
complete = raiw.remove_video_all("input.mp4", "clean.mp4")
batch = raiw.remove_video_batch("videos", "videos_clean")
cleaned = raiw.remove_video_metadata("input.mp4")
synthid_cleaned = raiw.remove_video_invisible("input.mp4", "synthid_clean.mp4")
visible = raiw.remove_video_visible("input.mp4", "clean.mp4")
print(visible.mark)
veo = raiw.remove_video_visible("veo.mp4", "veo_clean.mp4", mark="veo")
seedance = raiw.remove_video_visible(
    "seedance.mp4",
    "seedance_clean.mp4",
    mark="seedance",
)
dola = raiw.remove_video_visible("dola.mp4", "dola_clean.mp4", mark="dola")
```

The high level API accepts a file path or a BGR NumPy array. For path inputs it
also reads provenance metadata, preserves alpha, and can strip AI metadata from
the written result.

See the [Python API guide](docs/python-api.md) for visible removal, the full
`remove_all` and `remove_batch` pipeline, provenance inspection, metadata
stripping, and diffusion usage.

## ComfyUI

The separate
[ComfyUI Remove AI Watermarks](https://github.com/wiltodelta/ComfyUI-remove-ai-watermarks)
package provides nodes for visible removal, detection, region erasing, and
invisible removal.

## Important limitations

- A missing local signal means unknown, not clean. Proprietary pixel
  watermarks may remain after metadata has been stripped.
- Visible removal reconstructs a small region. Results depend on the background
  and selected fill backend.
- Invisible removal changes the whole image and may alter faces, text, or fine
  detail.
- Visible video removal recognizes the moving Sora 2 wordmark, the current Veo
  diamond plus legacy `Veo` text, the Seedance boxed `AI` label, and the fixed
  Dola, Hailuo AI, and Kling AI labels. It does not recognize the older Sora Turbo
  corner swirl or unregistered layouts from those providers.
  The classical OpenCV backend can smear structured backgrounds; use MI-GAN or
  LaMa when recovery quality matters.
- Video SynthID regeneration changes resolution, frame rate, and image detail.
  The shipped profile is oracle-certified, but no public local decoder can
  certify an arbitrary output at runtime. Recheck unusually important outputs
  after provider changes.
- Invisible-watermark removal requires CUDA. All profiles refuse any other
  device at construction rather than falling back to one that cannot run them.
  Visible removal, metadata stripping and `identify` still run anywhere.
- Provider watermark systems can change. Validate important outputs with the
  provider's own verifier when one is available.

The shipped `video invisible` command uses the certified `noise_std=0.15`
profile. The companion `scripts/video_synthid_sweep.py` research harness builds
a matched re-encode control plus VAE-regenerated candidates and leaves the
verifier verdict blank:

```bash
uv run --extra video --extra diffusion python scripts/video_synthid_sweep.py input.mp4 -o sweep/
```

To score what an output actually cost, use `scripts/video_fidelity_probe.py`:
the engine's own PSNR is measured before the resize and the encode, so only the
probe sees the delivered picture.

```bash
uv run --extra video python scripts/video_fidelity_probe.py input.mp4 input_clean.mp4
```

The control must still be SynthID-positive before a negative candidate can
count as removal evidence. In the 2026-07-29 two-clip calibration, both matched
controls were positive in Gemini's built-in SynthID verifier; the stronger
candidate was negative on both carriers, while a weaker candidate was
negative on one. A later adversarial follow-up that asked ordinary Gemini to
reinterpret the pixel result returned `UNAVAILABLE`; that follow-up was not a
verifier rerun and does not invalidate the built-in verdicts. A 2026-07-31
full-clip check on a public eight-second Veo sample found `0.10` still detected
and `0.15` not detected, so `0.15` is now the certified default. The
reproducible hashes and verdicts live in
`data/evaluations/video-synthid-oracle.csv`.

## Documentation

Start with the [documentation index](docs/index.md).

- [Installation](docs/installation.md)
- [CLI guide](docs/cli.md)
- [Python API](docs/python-api.md)
- [Supported signals](docs/supported-signals.md)
- [Known limitations](docs/known-limitations.md)
- [Scope, safety, and legal notes](docs/legal-and-safety.md)
- [Module internals](docs/module-internals.md)
- [Release and distribution](docs/release-and-distribution.md)

Research notes and historical experiments are listed separately in the
[documentation index](docs/index.md). They explain past decisions but do not
define the current public API.

## Contributing

Install the development environment and run the project gate:

```bash
uv sync --frozen --extra dev
bash maintain.sh
```

See [module internals](docs/module-internals.md) before changing a subsystem
with documented invariants.

## License

[Apache 2.0](LICENSE). Copyright 2025-2026 wiltodelta.
