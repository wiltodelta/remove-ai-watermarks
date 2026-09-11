# Supported signals

This page describes the current support boundary. A check mark means that the
repository contains a corresponding code path. It does not guarantee detection
or removal on every future vendor version.

## Visible marks

The `visible` command registers these mark keys:

| Key | Mark | Expected area | Important limit |
| --- | --- | --- | --- |
| `gemini` | Google Gemini visible watermark (sparkle) | Usually bottom right | Detection includes a false positive gate. |
| `doubao` | `豆包AI生成` | Bottom right | Vendor specific text detector. |
| `jimeng` | `★ 即梦AI` | Bottom right | Vendor specific text detector. |
| `qwen` | `千问AI生成` or the three-lobe Qwen symbol | Bottom right | Separate strict templates cover the text and Qwen Create symbol variants. |
| `kling` | `可灵AI 3.0` or `KlingAI 3.0` | Bottom right | Separate calibrated silhouettes cover the older CJK and current IMAGE 3.0 Latin variants. |
| `yuanbao` | `元宝` over `AI生成` | Bottom right | Standard two-line variant only. |
| `samsung` | `✦ Contenuti generati dall'AI` | Bottom left | Calibrated for the Italian text variant. |
| `runninghub` | `RunningHub AI生成` | Top left | Strict visual and position gates. |
| `baidu` | `百度 AI生成` | Bottom right | Detector and extended removal footprint. |
| `liblib` | `LiblibAI` | Bottom center | Historical wordmark variant; includes a minimum image size gate. |
| `liblib_pill` | Compact `AI生成` pill | Top left | Auto removal requires LiblibAI metadata or the bottom-center wordmark, plus a flat background under the pill. |
| `microsoft` | One Microsoft white AI-badge variant | Top right | Strict uses the visual gate; auto can use Microsoft provenance for the measured [relaxed gate](module-internals.md#visible-mark-removal). Other documented icon, text, and position variants are not covered. |
| `jimeng_pill` | `AI生成` pill | Top left | Weak detector with additional product and background gates. |

`--mark auto` evaluates all registered marks and removes every selected match.
Known marks are localized to a mask, then the selected fill backend reconstructs
the masked area.

Marks from other vendors are not detected automatically. Use `erase --region`
when you can select the affected area yourself.

Synthetic canonical examples for every registered image and video mark live in
the [visible-mark gallery](../data/fixtures/visible/README.md), alongside
documented real provider originals where redistribution is cleared. The
canonical examples demonstrate detector geometry and house style; provider
originals exercise vendor raster fidelity.

Provider formats can change independently of the registered key. In particular,
the current LiblibAI sample in the gallery preserves a real new-format output
and its LiblibAI TC260 producer metadata. That metadata corroborates the compact
pill detector; the generic pill shape does not attribute LiblibAI on its own.

### Visible video marks

| Key | Mark | Motion | Important limit |
| --- | --- | --- | --- |
| `sora` | Sora 2 mascot and wordmark | Moves among frame positions | Requires a temporally recurring visual match; the older Sora Turbo corner swirl is a different unsupported mark. |
| `veo` | Current four-point diamond and legacy `Veo` text | Fixed bottom-right corner | Uses separate silhouettes and requires a recurring match; learned fill is preferable on structured backgrounds. |
| `seedance` | Boxed `AI` label | Fixed bottom-right corner | Requires an anchored recurring match; the full localized box is filled because a thinner synthetic shape mask leaves the real translucent rim behind. |
| `doubao` | `豆包AI生成` text run | Fixed bottom-right corner | Reuses the image engine's synthetic alpha as the template; a stable run of at least 12 frames at confidence >=0.35 is required. Without Doubao TC260 confirmation, the run also needs a strong anchor at >=0.55. |
| `dola` | `Dola AI` text | Fixed bottom-right corner | Requires an anchored recurring match; ByteDance or BytePlus provenance can relax only an existing visual run. |
| `hailuo` | `MINIMAX \| hailuo AI` composite label | Fixed lower edge | Uses a synthetic waveform, text, separator, and ring silhouette; the complete recurring label box is filled. A TC260 label naming MiniMax as producer can relax only an existing stable run. |
| `kling` | Kling AI swirl, `KLING AI` or `KlingAI`, version, and optional `PRO` suffix | Fixed bottom-right edge | Combines a synthetic logo rescue with font and capitalization variants, an edge gate, a white-label gate, and anchored temporal recurrence. |

`video identify`, `video visible`, and `video all` share this registry and the
same temporal arbiter. It is separate from the image registry because selection
is made over a sequence rather than one raster. The default `auto` mode scans
all seven entries in one decode pass and selects the first temporally stable
match in table order; an explicit mark restricts the scan to that row.
Accepted fills are motion-aligned across adjacent frames by default. The prior
fill contributes only where its warped mask covers the current removal mask and
nearby source context agrees. Scene cuts or disjoint marks retain the
independent frame fill.

## Fill backends

| Backend | Install | Behavior |
| --- | --- | --- |
| `cv2` | `remove-ai-watermarks[visible]` | Classical OpenCV inpainting |
| `migan` | `remove-ai-watermarks[migan]` | MI-GAN through ONNX Runtime; practical learned CPU video tier |
| `lama` | `remove-ai-watermarks[lama]` | big-LaMa through ONNX Runtime; offline video quality tier |
| `auto` | Depends on installed extras | Selects LaMa, then MI-GAN, then OpenCV |

The learned backends download model files on first use.

## Metadata and provenance

The inspection and stripping code handles signals in these groups:

- C2PA Content Credentials and supported cloud manifest references;
- EXIF and XMP generator fields;
- exact app-export provenance and AIGC disclosures from supported
  ByteDance-family products, with product-only provenance excluded from the
  generated-image verdict;
- IPTC AI disclosure fields;
- PNG text chunks and embedded generation parameters;
- China TC260 AIGC labels in supported image placements and the normative
  MP4/MOV `moov.udta.meta.keys/ilst`, MKV/WebM
  `Segment.Tags.Tag.SimpleTag`, AVI `LIST/INFO/AIGC`, and FLV
  `script.onMetaData.AIGC` placements;
- xAI and Grok EXIF signature fields;
- Samsung AI editing markers;
- Hugging Face job metadata;
- open Stable Diffusion style DWT-DCT watermarks with the `detect` extra;
- Adobe TrustMark Variant P schemas 0-2 with the `trustmark` extra. Variant Q
  needs a different model, while schema 3 is deliberately rejected because it
  produced persistent false positives on unrelated generators.

`identify` combines detected signals into a `ProvenanceReport`. It reports
unknown when evidence is absent. It never treats missing metadata as proof that
an image is human made. C2PA presence alone is not a verified identity: the
report distinguishes asset binding, claim signature, signer trust, and signer
validity. High-confidence C2PA attribution requires an intact asset binding and
claim signature; signer trust and certificate expiry are reported as their own
dimensions and as caveats, since no trust anchor list ships to evaluate them.
Fallback claims, which validate nothing, remain medium-confidence, while a failed
binding or signature or a revoked credential contributes no origin verdict.

C2PA soft-binding names and their official `watermark` or `fingerprint` type come
from a generated snapshot of the
[C2PA registry](https://github.com/c2pa-org/softbinding-algorithm-list). A registry
match names the declared algorithm but does not imply local payload decoding.
`identify` keeps both kinds in the generic `soft_binding` signal; only watermark
entries enter the watermark inventory, while a fingerprint remains a
re-linkability warning, never selects pixel regeneration, and does not suppress
independent SynthID evidence.

## File and container formats

Pixel based image commands discover these extensions:

- PNG;
- JPEG;
- WebP;
- HEIC and HEIF;
- AVIF.

HEIC, HEIF, and AVIF pixel decoding requires the independent `heif` extra in
addition to the selected pixel feature. Metadata scanning does not.

Metadata inspection and removal additionally have container paths for:

- JPEG XL metadata;
- MP4, MOV, M4V, and M4A;
- WebM, MKV, MKA, AVI, FLV, MP3, WAV, FLAC, OGG, OGA, Opus, and AAC when
  ffmpeg is available.

JPEG image metadata stripping removes targeted metadata segments without
re-encoding the entropy coded image scan. PNG and WebP removal preserves pixel
values through lossless output paths. HEIC, HEIF, AVIF, and other containers
use their format specific paths.

## Invisible watermarks

The `invisible` command uses diffusion regeneration. It targets watermark
patterns by changing the image rather than decoding and deleting a known
payload.

Current pipeline values, all CUDA-only:

- `qwen-zimage`, the default;
- `sdxl-zimage`, the same recipe and the same face stage on an SDXL global pass;
- `chroma-zimage`, the same face stage on a Chroma1 global pass;
- `auto`, chroma-zimage for Microsoft provenance, otherwise qwen-zimage.

The `controlnet`, `sdxl`, `qwen` and `default` values were removed. A retired name
is rejected at parse time rather than remapped onto a surviving profile.

Google does not publish the SynthID payload decoder. This package does not
ship a local pixel detector for that watermark. Research on a periodic
lattice expert is in [synthid-detector-research.md](synthid-detector-research.md)
and `scripts/synthid_runtime/`.

The tool recognizes presence from supported provenance: Google AI C2PA
under Google's all-media watermark policy, and current OpenAI C2PA carrying an
explicit `c2pa.watermarked.*` action. Legacy OpenAI C2PA without that action
does not assert SynthID.

Provider verifiers and provenance APIs are reserved for development-time
calibration. They are not exposed through the installed CLI, the top-level
Python API, or `identify`; production reports only locally available evidence.

Microsoft Paint can name `com.microsoft.invismark.1` in a C2PA soft-binding
assertion. Inspection reports both that exact algorithm and its signed `value`,
which Paint uses as the identifier carried by the pixel watermark, and emits an
additive `invismark` signal so callers can select pixel removal without parsing
the generic `soft_binding` detail. Photos uses a parallel local writer path.
Metadata stripping removes only the embedded manifest; `invisible` and `all`
guarantee the supported InvisMark removal contract by regenerating the pixel
layer as well. The project has no validated local InvisMark decoder, so local
inspection cannot independently verify the output. Microsoft's official
[Content Provenance Detection API](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/how-to/how-to-provenance-detection)
is the external oracle: it reports pixel `Watermark` and embedded `C2PA` results
separately; a control-positive, output-negative pair is the available per-file
verification path. The API needs Azure credentials; the page a human can check
without an account is <https://ai.azure.com/nextgen/validate>. That page is the
public provider oracle used for the measured removal certification. Its clean
outcome is rendered as `Inconclusive`; the authenticated API exposes separate
watermark and C2PA fields but is not a different class of certification.

Meta Muse Image stamps every output with Content Seal, a proprietary invisible
pixel watermark, and ships no visible mark (the legacy `Imagined with AI`
corner mark belongs to the pre-2026 Imagine pipeline and is not registered).
This project has no local Content Seal decoder. Meta Model API outputs and
Meta CDN copies carry an XMP `iptcExt:DigitalSourceType =
trainedAlgorithmicMedia` companion tag, which `identify` reports through the
existing Made-with-AI path; that IPTC code is a standard, not a Meta-exclusive
signal, so it neither asserts Content Seal nor selects a strength cohort. The
external oracle is `https://meta.ai/identification`: anonymous, no login,
accepts image, video, and audio, enforces an unspecified daily identification
limit, and answers with model attribution (`Muse Image 1 - Meta`) plus a
per-generation ID and creation timestamp read from the watermark payload. No
identification endpoint exists in the Meta Model API itself (the reference at
dev.meta.ai documents only generation and edits for images), and the official
documentation never mentions the seal. The
default `qwen-zimage` profile clears Content Seal at the default
resolution-adaptive strength (oracle-verified on 2.56 MP generations); measured
strength boundaries are recorded in `data/contentseal/manifest.csv` and
[module internals](module-internals.md#meta-content-seal-boundaries-for-qwen-zimage).
The derived Meta floor (0.1 by the standard spread method) ships as a measured
cohort. `invisible --vendor meta` (also `all` and `batch`, and
`InvisibleOptions.vendor` in the API) selects it when the source is independently
known. An explicit vendor implies the scrub runs: naming the cohort asserts the
pixel watermark is present.
The seal survives resizing, JPEG recompression, and metadata stripping; it dies
to center crops of a third to a half, matching the Reuters 2026-07-11 finding
that Meta's detector missed 55% of cropped Muse images.

For MP4, MOV, and M4V, `video invisible` or the explicit
`video all --invisible` option can regenerate the video through a VAE and strip
source metadata. The shipped profile is oracle-certified, but it is not a local
decoder. A fresh source-positive, output-negative pair from Gemini's built-in
SynthID verifier is an optional per-file audit. A normal Gemini answer may instead
infer from a visible logo or metadata; asking it to reinterpret a completed
verifier result is not a second oracle run.

The optional `detect` extra is different: it provides a local decoder for the
open DWT-DCT watermark used by some Stable Diffusion, SDXL, and FLUX workflows.
That signal is carrier and transformation sensitive, so a negative is still
not a universal clean verdict.

## Provider overview

| Provider or family | Visible | Invisible path | Metadata or provenance |
| --- | --- | --- | --- |
| Google Gemini | Sparkle | Diffusion regeneration | C2PA and related source signals |
| Google Veo video | Veo diamond and legacy text | Oracle-certified VAE removal for SynthID | C2PA and related source signals |
| OpenAI image generators | None registered | Diffusion regeneration for supported invisible signals | C2PA and generator provenance |
| Meta Muse Image | None on Muse output (legacy `Imagined with AI` unregistered) | Diffusion regeneration for Content Seal, oracle-verified on the default profile | XMP IPTC `trainedAlgorithmicMedia` companion tag; no local Content Seal decoder |
| Microsoft Paint and Photos | None registered | External Microsoft oracle for InvisMark; no validated local decoder | Paint C2PA soft-binding algorithm and identifier |
| Microsoft image outputs (measured variant) | One top-right white AI-badge variant | No registered pixel decoder | C2PA attribution |
| Stable Diffusion and SDXL | None registered | Diffusion regeneration; optional open decoder | Embedded parameters and text metadata |
| FLUX | None registered | Diffusion regeneration; optional open decoder | C2PA for supported sources |
| Adobe Firefly | None registered | Optional TrustMark Variant P decoder | C2PA |
| Midjourney | None registered | No registered pixel decoder | EXIF, XMP, and IPTC signals |
| Luma AI | None registered | No registered pixel decoder | PNG text generator tags (Uni-1) |
| ByteDance generators | Doubao and Jimeng marks | No registered pixel decoder | TC260 AIGC, supported C2PA, and exact app-export AIGC disclosures |
| Qwen | Qwen mark | No registered pixel decoder | TC260 AIGC |
| Kling AI | Kling AI image and video marks | No registered pixel decoder | TC260 AIGC |
| Hailuo AI / MiniMax video | Hailuo AI composite video label | No registered pixel decoder | TC260 AIGC where present |
| Baidu | Baidu mark | No registered pixel decoder | TC260 AIGC |
| LiblibAI | LiblibAI wordmark and compact pill | No registered pixel decoder | TC260 AIGC |
| RunningHub | RunningHub mark | No registered pixel decoder | TC260 AIGC |
| Samsung Galaxy AI | One locale specific mark | No registered pixel decoder | C2PA and Samsung markers |

For detector thresholds, measured limits, and incident history, see
[module internals](module-internals.md) and
[known limitations](known-limitations.md).
