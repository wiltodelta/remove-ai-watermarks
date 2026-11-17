# Supported signals

This page describes the current support boundary. A check mark means that the
repository contains a corresponding code path. It does not guarantee detection
or removal on every future vendor version.

## Visible marks

The `visible` command registers these mark keys:

| Key | Mark | Expected area | Important limit |
| --- | --- | --- | --- |
| `gemini` | Google Gemini visible watermark (sparkle) | Usually bottom right | Detection includes a false positive gate. |
| `doubao` | `豆包AI生成` | Bottom right | Covers filled and outlined text layouts; the weak outlined layout requires matching Doubao provenance. |
| `jimeng` | `★ 即梦AI` | Bottom right | Vendor specific text detector. |
| `qwen` | `千问AI生成` or the three-lobe Qwen symbol | Bottom right | Separate strict templates cover the text and Qwen Create symbol variants. |
| `wan` | Wan logo and `Wan` wordmark | Bottom right, hugging the corner | Provisional: one real Wan 2.7 Pro export. Requires NCC >= 0.55 and a match that ends within 0.015 of the short side from both edges. |
| `kling` | `可灵AI 3.0`/`4.0` or `KlingAI 3.0`/`4.0` | Bottom right | Separate calibrated silhouettes cover the CJK and Latin 3.0/4.0 variants; a glyph-detail gate rejects plain UI boundaries. |
| `yuanbao` | `元宝` over `AI生成` | Bottom right | Separate strict and provenance-corroborated templates cover the measured two-line rasterizations. |
| `dola` | `Dola AI` | Bottom right | Text detector and detector-aligned wordmark footprint. |
| `workbuddy` | `AI生成` and `WORKBUDDY>_` | Bottom right | Separate templates cover the single-line and stacked two-line layouts; corroborated by the registered Tencent WorkBuddy TC260 producer code. |
| `samsung` | Italian or Korean Galaxy AI disclosure | Bottom left | Independently gated text layouts cover `Contenuti generati dall'AI` and `AI로 생성한 콘텐츠`. |
| `runninghub` | `RunningHub AI生成` | Top left | Strict visual and position gates. |
| `baidu` | `百度 AI生成` | Bottom right | Detector and extended removal footprint. |
| `liblib` | `LiblibAI` | Bottom center | Historical wordmark variant; includes a minimum image size gate. |
| `liblib_pill` | Compact `AI生成` pill | Top left | Auto removal requires LiblibAI metadata or the bottom-center wordmark, plus a flat background under the pill. |
| `microsoft` | Microsoft white AI badge or beige `Made with AI` pill | Top right | Each layout has its own strict visual gate; auto can use Microsoft provenance for the original badge's measured [relaxed gate](module-internals.md#visible-mark-removal). |
| `jimeng_pill` | `AI生成` pill | Top left | A verified label layout can bypass the flat-background guard only with Jimeng provenance; the weak outline detector retains the product and background gates. |
| `generic_ai_label` | Bare `AI生成` text, no vendor wordmark | Bottom right | Brand-less TC260 compliance-stamp fallback (vivo and Xiaomi Gallery AI-edit output confirmed; see [module-internals.md](module-internals.md#generic-bare-ai生成-label)). Font-rendered synthetic asset; **calibrated on a synthetic corpus only**, not a real captured generator screenshot -- treat as weaker evidence than the vendor-tuned detectors above. |
| `openart` | `OpenArt` wordmark (bowtie/infinity icon + brand name) | Frame center, not a corner | Confirmed from one customer-reported production case; no known metadata signal, so the visible mark is the only attribution route. **UNCALIBRATED**: procedurally reconstructed asset, no captured corpus -- see [module-internals.md](module-internals.md#openart-wordmark). |

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
| `sora` | Sora 2 mascot, with or without the wordmark | Moves among frame positions | Requires a temporally recurring visual match; mascot-only matches also require complete-shape correlation, which rejects a measured Veo-diamond cross-match. The older Sora Turbo corner swirl is a different unsupported mark. OpenAI discontinued the Sora app on 2026-04-26 and the Sora API on 2026-09-24, so this covers existing exports only. |
| `veo` | Current four-point diamond and legacy `Veo` text | Fixed bottom-right corner | Uses separate silhouettes and requires a recurring match; a valid C2PA AI claim from another vendor (Google on a Sora match, for example) vetoes a mark its own vendor does not also confirm. Provenance alone does not create a visual detection. Learned fill is preferable on structured backgrounds. |
| `seedance` | Boxed `AI` label | Fixed bottom-right corner | Requires an anchored recurring match; the full localized box is filled because a thinner synthetic shape mask leaves the real translucent rim behind. |
| `doubao` | `豆包AI生成` text run | Fixed bottom-right corner | Reuses the image engine's synthetic alpha as the template; a stable run of at least 12 frames at confidence >=0.35 is required. Without Doubao TC260 confirmation, the run also needs a strong anchor at >=0.55. |
| `dola` | `Dola AI` text | Fixed bottom-right corner | Requires an anchored recurring match; ByteDance or BytePlus provenance can relax only an existing visual run. |
| `hailuo` | `MINIMAX \| hailuo AI` composite label | Fixed lower edge | Uses a synthetic waveform, text, separator, and ring silhouette; the complete recurring label box is filled. A TC260 label naming MiniMax as producer can relax only an existing stable run. |
| `vidu` | Vidu two-loop logo and `Vidu AI` wordmark | Fixed bottom-right corner | Provisional: one real Vidu Q3 export (0.72 per frame at 480-1080p, 0.60 at 360p; the 956 other local videos at most 0.41). Requires an anchored recurring run at >=0.52 with a strong frame at >=0.58. A ShengShu TC260 label confirms the vendor without lowering the bar. |
| `kling` | Kling AI swirl, `KLING AI` or `KlingAI`, version, and optional `PRO` suffix | Fixed bottom-right edge | Combines a synthetic logo rescue with font and capitalization variants, an edge gate, a white-label gate, and anchored temporal recurrence. It also scores the Vidu wordmark, so `vidu` is checked first, and a TC260 label naming a non-Kling producer vetoes the match. |

`video identify`, `video visible`, and `video all` share this registry and the
same temporal arbiter. It is separate from the image registry because selection
is made over a sequence rather than one raster. The default `auto` mode scans
all eight entries in one decode pass and selects the first temporally stable
match in table order, subject to the Veo/Sora overlap exception above; an
explicit mark restricts the scan to that row.
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
[VACE and ROSE were evaluated](video-inpainting-research.md) on the same
production masks and were not adopted: neither improved temporal stability
over MI-GAN, and ROSE retained the Veo mark.

## Metadata and provenance

The inspection and stripping code handles signals in these groups:

- C2PA Content Credentials and supported cloud manifest references;
- EXIF and XMP generator fields, including structured
  `generationParams.modelName` values from supported generators;
- exact app-export provenance and AIGC disclosures from supported
  ByteDance-family products, with product-only provenance excluded from the
  generated-image verdict;
- Kling's H.264 `user_data_unregistered` SEI tag (`kling-ai`) in MP4/MOV
  video, blanked in place by `video metadata --remove` without re-encoding;
- IPTC AI disclosure fields;
- PNG text chunks and embedded generation parameters, including keyed
  `workflow` and `prompt` tags in MP4/MOV metadata lists;
- China TC260 AIGC labels in supported image placements and the normative
  MP4/MOV `moov.udta.meta.keys/ilst`, MKV/WebM
  `Segment.Tags.Tag.SimpleTag`, AVI `LIST/INFO/AIGC`, and FLV
  `script.onMetaData.AIGC` placements, plus the audio placements of
  TC260-PG-202510A: a RIFF `AIGC` chunk in WAV, an ID3v2 `TXXX` frame described
  `AIGC` in MP3, and an `AIGC=` Vorbis comment in OGG, Opus and FLAC;
- TC260-PG-202511A `SecurityData` label signatures (SM3withSM2) in a label's
  `ReservedCode1`: a signature that verifies against a pinned producer key is a
  high-confidence `aigc_signature` signal (MiniMax is the one pinned signer), a
  signature that does not verify is a caveat, and every result appears as the
  `aigc_signature` metadata marker. Nested JSON reserved fields are preserved;
  Honor YOYO RSA/content signatures and Tencent Yuanbao raw-hex SM2 signatures
  without an embedded public key are reported as unsupported, with no integrity
  or signer-trust claim. Registered producer codes attribute images and videos
  to their respective vendors in [`_internal/tc260_producers.py`](../src/remove_ai_watermarks/_internal/tc260_producers.py);
- xAI and Grok signature-plus-UUID pairs in ordinary EXIF, ImageMagick PNG raw
  EXIF profiles, XMP description/creator fields, PNG Description/Author text,
  and IPTC Caption-Abstract/By-line fields;
- Samsung AI editing markers;
- Higgsfield job metadata (the `hf-job-id` PNG chunk);
- open Stable Diffusion style DWT-DCT watermarks with the `detect` extra;
- Adobe TrustMark Variant P schemas 0-2 by default with the `trustmark` extra,
  plus Variants Q (`com.adobe.trustmark.Q`) and C (`com.adobe.trustmark.C`) when
  explicitly requested or corroborated by a C2PA soft-binding assertion. Schema
  3 is deliberately rejected because it produced persistent false positives on
  unrelated generators;
- Text and HTML provenance and character steganography via `text identify` and
  `text metadata` ([`text.py`](../src/remove_ai_watermarks/text.py)): C2PA
  2.3/2.4 Appendix A.8 `C2PATextManifestWrapper` (`C2PATXT\0` in Variation
  Selectors), Appendix A.9 structured-text ASCII armor
  (`-----BEGIN C2PA MANIFEST-----`), Appendix A.7 HTML `<script type="application/c2pa">`
  and `<link rel="c2pa-manifest">`, `VSRMark` (`com.cognitive-proof.vsrmark.1`,
  `VSR1` + CRC32 in interleaved Variation Selectors), Unicode Tags
  (`U+E0000`..`U+E007F`), the [published Windows AI Unicode sequence](text-watermark-validation.md#published-format-fixtures)
  (synthetic format tests, no provider attribution), and zero-width steganography. Statistical
  token-watermark disruption uses best-effort span-protected local prose
  regeneration (`text invisible`, `text all --invisible`); this is not a verified
  private-provider detector or removal guarantee;
  see [text watermark research](text-watermark-research.md).

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

The opt-in `classify_source` API (and CLI `classify-source`) is not an
`identify` signal. It can suggest one of 10 AI source/export pipelines and its
model-version subfamily after metadata removal, but it neither detects nor
decodes a watermark and never changes the provenance verdict.

## File and container formats

Pixel based image commands discover these extensions:

- PNG;
- JPEG (`.jpg`, `.jpeg`, `.jpe`, and `.jfif`);
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

With ffprobe installed, audio-only MP3, WAV, FLAC, Ogg, and ISOBMFF inputs
can expose exact native Suno or Suno Studio creation comments as
`audio_generator`. Attached covers do not count as video. These unsigned
declarations are removable metadata, not proof of authenticity or a waveform
watermark. Arbitrary Suno mentions are not supported signals.

JPEG image metadata stripping removes targeted metadata segments without
re-encoding the entropy coded image scan. PNG and WebP removal preserves pixel
values through lossless output paths. HEIC, HEIF, AVIF, and other containers
use their format specific paths.

## Invisible watermarks

The optional audio API and `audio` CLI group inspect pinned public AudioSeal
and WavMark models, public Perth v1, and an experimental SilentCipher-44k receiver.
`--scheme` selects one model or all named models. AudioSeal checks both 16 kHz
and native sample clocks for validated 24 and 48 kHz sources. Other native
clocks are not added automatically; neither a quiet resampled receiver
nor a matching public-model mark establishes production-provider attribution.
WavMark separately checks the published OpenVoice default raw framing at
fixed native offsets, alongside its synchronized public-model scan. This
signature does not cover custom messages or shifted/resampled framing, and its
binomial-tail diagnostic is not a calibrated false-positive probability.
They provide
streaming spectral, neural-codec, and combined processing, plus explicit
experimental full-track BigVGAN, Mimi, and FocalCodec profiles. Mimi uses fixed
CPU FP32 and an internal 24 kHz grid; FocalCodec uses fixed CPU FP32 continuous
6-layer WavLM-to-Vocos resynthesis on an internal 16 kHz grid (`8 kHz` ceiling);
BigVGAN has an explicit device selector. All three use
bounded mono/stereo processing, preserve the native sample grid and expose
quality limitations separately. None implements a SynthID decoder.
Default removal requires a
source-positive and saved-output-negative check; resistant and incomplete
candidates are rejected. This does not establish compatibility with proprietary
audio marks. Public Perth v1 has a pinned runtime via `--scheme perth` and
`audio-perth`. The inspected default Chatterbox checkpoint has native Nano
source-positive and saved-output-negative evidence through this route, including
the original PerTh reader and final video mux. A PerTh response alone does not
identify Chatterbox. This does not certify proprietary PerTh v2. SilentCipher-44k uses
`--scheme silentcipher` and
`audio-silentcipher`, with three-second windows and a project-defined repeated-symbol
threshold. Neither its negative nor loss of a fixed message certifies complete erasure.
Reports identify its 44.1 kHz decoder clock separately from the source rate.
The published 16 kHz model has incompatible SDK byte framing and is outside
this receiver, including when the input itself is 16 kHz.
The separate `--scheme silentcipher16` route checks an experimental terminated
eight-symbol pattern with the pinned 16 kHz model and the same `audio-silentcipher`
extra. It labels readings `framing="terminated8"` and reports seven ternary
`payload_symbols`, not bytes, and a 16 kHz decoder
clock. `all` includes both receivers. The SDK byte API and general 32-bit profile
remain unqualified; a negative does not rule out other framing or deployed marks.
Negative results cover the requested decoder checks; they do not certify
erasure of every residual payload bit. See [audio coverage and local results](audio-video-removal.md).

The `invisible` command uses diffusion regeneration. Specialized Python APIs can
instead apply decoder-guided disruption to a positively identified local
Microsoft Paint payload or official neural removal to a confirmed Adobe
TrustMark payload.

Current diffusion pipeline values, all CUDA-only:

- `qwen-zimage`, the default (Apache-2.0);
- `qwen21-zimage`, opt-in Qwen-Image-2.1 global stage plus the same Z-Image face
  stage (`Qwen/Qwen-Image-2.1` is under the Qwen Research License: non-commercial
  research, evaluation, and personal use only);
- `sdxl-zimage`, the same recipe and the same face stage on an SDXL global pass;
- `chroma-zimage`, the same face stage on a Chroma1 global pass;
- `auto`, sdxl-zimage for Google provenance, chroma-zimage for Microsoft,
  otherwise qwen-zimage.

The `controlnet`, `sdxl`, `qwen` and `default` values were removed. A retired name
is rejected at parse time rather than remapped onto a surviving profile.

Google does not publish the SynthID payload decoder. This package does not
ship a local pixel detector for that watermark. Research on a periodic
lattice expert is in [synthid-detector-research.md](synthid-detector-research.md)
and `scripts/synthid_runtime/`.

The tool recognizes presence from supported provenance: Google AI C2PA
under Google's all-media watermark policy, and current OpenAI C2PA carrying an
explicit `c2pa.watermarked.*` action. Two Google signers are exceptions.
YouTube re-signs uploads as Google LLC (certificate `YouTube Video Processing
Services`, actions `opened` and `transcoded`, measured on Studio downloads), so
a YouTube-signed manifest establishes SynthID only through a SynthID action in
its chain, as a Gemini ingredient carries; with no AI claim in the chain,
`video identify` reports unknown. A Photos-signed AI edit is reported as
"Google Photos (AI edit)" with SynthID present, although the Photos manifest
records no SynthID action and Google documents the mark only for Reimagine:
Google's checker found SynthID in all four Photos AI edits tested on 2026-09-25
(Ask, eraser and two other edits). On 2026-10-01 UTC, a large Magic Eraser edit
and a watercolor restyle were positive, but two small `Help me edit` changes
were indeterminate; the source and crop control were negative. A caveat keeps
this provenance inference separate from a confirmed pixel watermark on the
particular file. Legacy OpenAI C2PA without that action
does not assert SynthID. GPT Image 2.5 Flare and Sunburst were verified on
2026-09-15: both API outputs carried valid OpenAI C2PA with
`c2pa.watermarked.unbound` and exercised the existing model-independent OpenAI
route. Automatic routing selected `qwen-zimage` with the OpenAI strength cohort,
and both outputs cleared the official OpenAI SynthID check.

Provider verifiers and provenance APIs are reserved for development-time
calibration. They are not exposed through the installed CLI, the top-level
Python API, or `identify`; production reports only locally available evidence.

Microsoft Paint can name `com.microsoft.invismark.1` in a C2PA soft-binding
assertion. Inspection reports both that exact algorithm and its signed `value`,
which Paint uses as the identifier carried by the pixel watermark, and emits an
additive `invismark` signal so callers can select pixel removal without parsing
the generic `soft_binding` detail.

For Paint's local `Watermarker.dll` format, the reader recovers the repeated
144-bit message and requires its prefix, GUID byte order, checksum, and per-bit
support. A caller can additionally require agreement with the UUID signed in
pristine C2PA. Only a positive local payload uses decoder-guided bit disruption;
the encoded output is read back and must no longer contain a valid local payload
before it is published. A Microsoft
soft-binding declaration without a matching local payload, including the
committed cloud Image Creator fixture, is rejected by this decoder. The CLI and
high-level API retain their Chroma diffusion route. Metadata stripping alone still
removes only the manifest.

This is local payload invalidation, not external removal certification. The
available Azure Content Provenance Detection API did not detect a known-positive
local-format control, so it cannot certify this direct output. Photos and future
Paint builds may use another pixel format and therefore remain on the regeneration
fallback unless this decoder positively identifies them.

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

Amazon Bedrock (`Nova Canvas Core Model` and `Titan Image Generator Core Model`)
embeds a proprietary invisible pixel watermark verified through the Bedrock
Watermark detection console and `POST /detect-generated-content`
(`DetectGeneratedContent`). A 2026-11-08 evaluation on six MIT-0-cleared
carriers from `aws-samples/amazon-nova-samples` (three Nova Canvas images at
1280x720 and 1024x1024 with C2PA signed `Amazon Web Services, Inc.`, and three
Titan Image Generator images at 1024x1024 and 768x768) confirmed that lossless
metadata stripping leaves `6/6` detected at `HIGH` confidence (`3/3 Nova Canvas
Core Model`, `3/3 Titan Image Generator Core Model`). Non-diffusion attacks
already clear Nova Canvas (`0/3` on `jpeg-q90`, `resize-75`, and `crop-5`),
while Titan Image partially survives (`2/3` `MEDIUM` on `jpeg-q90`, `1/3`
`HIGH` on `resize-75`, `2/3` `MEDIUM` on `crop-5`). Diffusion regeneration on
Modal (`NVIDIA H100 80GB HBM3`) cleared `6/6` across all three default profiles
(`qwen-zimage` at resolution-adaptive `0.090-0.099`, `sdxl-zimage` at `0.50`,
and `chroma-zimage` at `0.50`: `0/18` detected) and already cleared `6/6` on
`qwen-zimage` at `s = 0.03` and `s = 0.05` (`0/12` detected) as well as on
`sdxl-zimage` and `chroma-zimage` at `s = 0.05`, so the standard
resolution-adaptive `qwen-zimage` default (`0.090-0.099`) provides a 3x safety
margin with no separate vendor strength floor needed.

For MP4, MOV, and M4V, `video invisible` or the explicit
`video all --invisible` option can regenerate the video through a VAE and strip
source metadata. The shipped profile is oracle-certified for SynthID, but it is
not a local decoder. A fresh source-positive, output-negative pair from Gemini's built-in
SynthID verifier is an optional per-file audit. A normal Gemini answer may instead
infer from a visible logo or metadata; asking it to reinterpret a completed
verifier result is not a second oracle run.
The [VideoSeal VAE comparison](videoseal-temporal-evaluation.md#vae-regeneration-evaluation-2026-11-08)
found a surviving matched message on one of four standardized carriers under
the benchmark's threshold. The SynthID certification does not certify VideoSeal
removal.

The explicit [`video message`](cli.md#public-model-video-messages-experimental) route
compares an independently known message under pinned public VideoSeal or PixelSeal
weights and gates processing on saved-file checks. It does not add these signals
to generic AI-origin identification.

[Non-SynthID video coverage](non-synthid-video.md) separates registry
declarations, open-model VideoSeal/PixelSeal checks, and provider verification.
Amazon Nova Reel, Microsoft video watermarks, and Adobe FlowMark remain subject
to the verifier and positive-carrier limits recorded there.

Seedance has an explicit `seedance-experimental` profile in the common video
invisible/all/batch pipeline (`--profile seedance-experimental` in the CLI),
with `delivery` and `master` encoding modes. The standalone
`remove_video_seedance_experimental` function exposes the same recipe.
The fixed recipe has a positive-control/negative-output CapCut check on an
owned label-free forest clip. It preserves native geometry, the complete frame
clock and copied audio, while changing texture and color. This is an explicit
experimental pixel transform, not a local detector or the default SynthID
profile. Prepare visible labels separately. On NVIDIA CUDA GPUs, explicit
`wan21-synthid-experimental` (`strength=0.38`) and `wan21-seedance-experimental`
(`strength=0.12`) additionally provide native-resolution 3D `Wan 2.1` video-to-video
regeneration (`remove_video_wan_experimental`, using `chunk_frames=193` and
`overlap_frames=17` raised-cosine window blending for clips over 193 frames), and
`--profile auto` (`profile="auto"`) selects the Seedance or SynthID family from
pre-strip metadata and visible-mark signals while preferring CUDA `Wan 2.1` when
available. On the label-free Seedance `forest` and held-out `bottle`
carriers, `wan21-seedance-experimental` passed matched-control CapCut checks at
higher PSNR and lower temporal residual than `seedance-experimental` (whereas
Dreamina and BytePlus enterprise API exports cleared via C2PA metadata stripping
alone); on `flow-001`, `lite-003`, 2-window 240-frame
`google-gemini-omni-flash-api`, and `veo-3-1`, `wan21-synthid-experimental`
passed matched-control Google Gemini checks at native `1280x720 @ 24 fps`,
including full-audiovisual MP4 deliveries paired with 44.1 kHz `--method dac`. See the
[Python API](python-api.md#experimental-seedance-video-reconstruction) and
[recorded study](seedance-video-study.md).

The optional `detect` extra is different: it provides a local decoder for the
open DWT-DCT watermark used by some Stable Diffusion, SDXL, and FLUX workflows.
That signal is carrier and transformation sensitive, so a negative is still
not a universal clean verdict.

Video identification exposes `invisible_watermark_hints` for registered
C2PA-declared watermark schemes and supported SynthID metadata. It also infers
Amazon Nova Reel from AI-asserting Bedrock video provenance, following
[Amazon's Nova Reel service card](https://docs.aws.amazon.com/ai/responsible-ai/nova-reel/overview.html).
These hints do not verify a surviving pixel watermark or successful removal.
Registered fingerprint algorithms remain metadata, outside the watermark hints.

## Provider overview

| Provider or family | Visible | Invisible path | Metadata or provenance |
| --- | --- | --- | --- |
| Google Gemini | Sparkle | Diffusion regeneration | C2PA and related source signals |
| Google Veo video | Veo diamond and legacy text | Oracle-certified VAE removal for SynthID (`synthid`); optional CUDA 3D `Wan 2.1` native-resolution profile (`wan21-synthid-experimental`) | C2PA and related source signals |
| OpenAI image generators | None registered | Diffusion regeneration for supported invisible signals | C2PA and generator provenance |
| Meta Muse Image | None on Muse output (legacy `Imagined with AI` unregistered) | Diffusion regeneration for Content Seal, oracle-verified on the default profile | XMP IPTC `trainedAlgorithmicMedia` companion tag; no local Content Seal decoder |
| Microsoft Paint local `Watermarker.dll` format | None registered | Local repeated-payload decoder and decoder-guided disruption; no usable external oracle for this format | Paint C2PA soft-binding algorithm and matching UUID |
| Microsoft cloud and other InvisMark variants | None registered | Chroma regeneration fallback | Microsoft C2PA attribution and soft-binding declaration |
| Microsoft image outputs (measured variant) | One top-right white AI-badge variant | No registered pixel decoder | C2PA attribution |
| Stable Diffusion and SDXL | None registered | Diffusion regeneration; optional open decoder | Embedded parameters and text metadata |
| FLUX | None registered | Diffusion regeneration; optional open decoder | C2PA for supported sources |
| Adobe Firefly | None registered | Optional TrustMark decoder and neural remover (`P`, `Q`, `C`); diffusion fallback when C2PA declares `com.adobe.trustmark.*` | C2PA |
| Midjourney | None registered | No registered pixel decoder | EXIF, XMP, and IPTC signals |
| Luma AI | None registered | No registered pixel decoder | PNG text generator tags (Uni-1) |
| ByteDance generators | Doubao, Jimeng, Dola, and Seedance marks | Experimental Seedance reconstruction (`seedance-experimental` and CUDA `wan21-seedance-experimental`); no local pixel decoder | TC260 AIGC (Doubao, Jimeng, Volcano Engine), supported C2PA (Dreamina, Dola, Pippit AI, BytePlus), and exact app-export AIGC disclosures |
| Tencent Yuanbao and WorkBuddy | Yuanbao and WorkBuddy image marks | No registered pixel decoder | TC260 AIGC (with raw-hex SM2 `SecurityData` on Yuanbao) |
| Meitu, vivo, Xiaomi, and Agnes AI | Generic `AI生成` label on vivo and Xiaomi Gallery AI edits | No registered pixel decoder | TC260 AIGC producer attribution |
| Sync Labs, PixelBin.io, and TikTok AI | None registered | No registered pixel decoder | C2PA attribution |
| Qwen | Qwen mark | No registered pixel decoder | TC260 AIGC |
| Kling AI | Kling AI image and video marks | No registered pixel decoder | TC260 AIGC and the video SEI tag |
| Vidu (ShengShu) | Vidu video mark | No registered pixel decoder | TC260 AIGC in the MP4 `AIGC` key |
| Runway | None measured on paid-plan output | No registered pixel decoder | C2PA signed `RUNWAY AI, INC.` on its own models; third-party models keep their vendor's manifest |
| Amazon Bedrock (Nova Canvas, Titan Image, Nova Reel) | None | Diffusion regeneration for Nova Canvas and Titan Image, oracle-verified against `DetectGeneratedContent` (`0/18` across `qwen-zimage`, `sdxl-zimage`, `chroma-zimage`; already `0/6` at `qwen-zimage` `s=0.03`) | C2PA signed `Amazon Web Services, Inc.`, claim generator `Amazon Bedrock` |
| Wan (Alibaba) | Wan image mark | No registered pixel decoder | TC260 AIGC in XMP |
| Hailuo AI / MiniMax video | Hailuo AI composite video label | No registered pixel decoder | TC260 AIGC where present |
| Baidu | Baidu mark | No registered pixel decoder | TC260 AIGC |
| LiblibAI | LiblibAI wordmark and compact pill | No registered pixel decoder | TC260 AIGC |
| RunningHub | RunningHub mark | No registered pixel decoder | TC260 AIGC |
| Honor YOYO AI editing | No registered AI mark; forum account overlay is separate | No registered pixel decoder | EXIF UserComment TC260 AIGC with nested RSA/content signature, reported unsupported |
| Samsung Galaxy AI | One locale specific mark | No registered pixel decoder | C2PA and Samsung markers |
| Google Photos AI edits | None | SynthID provenance inference: 4/4 earlier edits positive (2026-09-25); large eraser and restyle positive, two small edits indeterminate (2026-10-01 UTC). No SynthID action in the manifest | C2PA signed "Google Photos" with a `compositeWithTrainedAlgorithmicMedia` action, plus XMP `photoshop:Credit` `Edited with Google AI` |
| Apple Image Playground and Photos Clean Up | None | No registered pixel decoder; SynthID announced by Apple for a later update | XMP `photoshop:Credit` (`Apple Image Playground`, `Apple Photos Generative Edit: Clean Up`) with or without IPTC `digitalSourceType`; no C2PA on measured 2026-09 output |
| xAI Grok / Aurora / Grok Imagine | None | No registered pixel decoder | C2PA signed `xAI Grok Imagine`, legacy EXIF/XMP/PNG/IPTC `Signature: <base64>` + UUID pair, and ISOBMFF `\xa9cmt` / `comment` `Signature: <base64>` video tag |
| Topaz Labs and CyberLink PhotoDirector | None | No registered pixel decoder | C2PA attribution when an AI `DigitalSourceType` is present (`asserts_ai=False`), plus ISOBMFF `topazcloud*` / `videoai*` video metadata |

For detector thresholds, measured limits, and incident history, see
[module internals](module-internals.md) and
[known limitations](known-limitations.md).
