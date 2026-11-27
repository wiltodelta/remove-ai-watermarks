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
| `classify-source` | `remove-ai-watermarks[source-classify]` |
| `text invisible` and `text all --invisible`, default backend | `remove-ai-watermarks[text]` |
| Text regeneration with an explicit Apple silicon MLX checkpoint | `remove-ai-watermarks[text-mlx]` |
| Text regeneration with a locally installed model served by Ollama | `remove-ai-watermarks[text-api]` |
| Visible signals in `identify` | `remove-ai-watermarks[visible]` (`pixels` is the minimal runtime) |
| Open DWT-DCT signals in `identify` | `remove-ai-watermarks[detect]` |
| Adobe TrustMark signals in `identify` | `remove-ai-watermarks[trustmark]` |
| `visible` and `erase` with OpenCV | `remove-ai-watermarks[visible]` (`pixels` is the minimal runtime) |
| `visible` or `erase` with MI-GAN | `remove-ai-watermarks[migan]` |
| `visible` or `erase` with big-LaMa | `remove-ai-watermarks[lama]` |
| `invisible` and `all` (needs CUDA) | `remove-ai-watermarks[qwen-zimage]` |
| `video metadata` and `video identify --no-visible` | Default package |
| `video identify` | `remove-ai-watermarks[video]` |
| `audio identify --no-invisible` | Default package plus ffmpeg and ffprobe |
| `audio identify` and verified `audio invisible` | `remove-ai-watermarks[audio-detect]` plus ffmpeg and ffprobe |
| Audio `--scheme silentcipher` | `audio-silentcipher`, for the experimental public SilentCipher-44k receiver |
| Audio `--scheme perth` | `audio-perth`, for pinned public Perth v1 |
| Audio `--scheme wavmark` | `audio-wavmark`; `--scheme all` needs `audio-detect`, `audio-wavmark`, `audio-perth`, and `audio-silentcipher` |
| `audio invisible --allow-unverified --no-verify` | `remove-ai-watermarks[audio]` plus ffmpeg and ffprobe |
| `audio invisible --method encodec` and `video audio --method encodec` | Add `audio-codec`; keep `audio-detect` for default verification |
| `video audio` | Same audio extras and tools; no video pixel runtime is required |
| `video visible`, `video all`, and visible/all batch modes | `remove-ai-watermarks[video]` (includes FFmpeg and ffprobe) |
| `video invisible` and `video all --invisible` | `remove-ai-watermarks[video,diffusion]` (includes FFmpeg and ffprobe) |
| HEIC/HEIF/AVIF pixel input | Add `remove-ai-watermarks[heif]` |
| Every production command and Python-compatible backend | `remove-ai-watermarks[all]` |
| `mcp` | Default package; each tool needs the extra of the command it runs |

`batch` requires the same extra as its selected mode. Extras can be combined in
one installation, for example `remove-ai-watermarks[visible,detect,heif]`.

## Inspect and process audio

```bash
remove-ai-watermarks audio identify input.wav --json
remove-ai-watermarks audio identify input.m4a --scheme auto --json
remove-ai-watermarks audio identify input.m4a --no-invisible
remove-ai-watermarks audio metadata input.mp3 --check
remove-ai-watermarks audio metadata input.mp3 --remove -o clean.mp3
remove-ai-watermarks audio invisible input.wav -o processed.flac --json
remove-ai-watermarks audio invisible input.m4a -o processed.m4a --scheme auto --json
remove-ai-watermarks audio all input.mp3 -o clean.mp3 --invisible
remove-ai-watermarks audio batch ./tracks -o ./tracks_clean --mode all --invisible
```

`audio identify` and `audio invisible` accept `--stream N`, the zero-based audio-track ordinal, and
`--json`. Inspection reports metadata, a source SHA-256, stream geometry, named
compatible decoder checks, and a suggested `recommended_recipe` when AI metadata
or a compatible watermark signal is found. `--no-invisible` skips model inference.
`--scheme` selects `audioseal` (default on `identify`/`invisible`), `auto` (scans every *installed* non-experimental compatible public decoder — `audioseal`, `wavmark`, and `perth` — without failing when an optional decoder extra is absent or a clip is shorter than WavMark's 1-second minimum window, leaving the overall status `unverified` only if no applicable decoder completes), `wavmark`, `perth`, `silentcipher`, or `all` (requires all named public
models, including the experimental SilentCipher-44k receiver). A missing explicitly requested decoder keeps the combined result unverified.
For validated 24 and 48 kHz sources, AudioSeal checks both a resampled 16 kHz
clock and the original sample clock. Other native clocks are not added automatically. JSON detections identify the clock in `decoder_sample_rate`;
verification requires both clocks to be negative, across channels and windows.
A short AudioSeal tail reuses preceding audio instead of decoding a tiny
remainder on its own.
WavMark checks exactly one-second inputs and the final valid search window;
inputs shorter than its model window remain unverified. The same `wavmark`
route separately checks the published OpenVoice default message at native
samples 0 and 32000. JSON labels these readings `framing="openvoice-default"`
and records their native `decoder_sample_rate`. Custom messages and
shifted/resampled framing are outside this receiver. SilentCipher-44k uses
three-second windows and an experimental consistency threshold, with exhaustive
phase search. Shorter sources stay unverified; CPU decoding can be slow.
Its JSON `decoder_sample_rate` is 44100 even for 16 kHz input. Neither negative
certifies a production provider, other framing, or complete payload erasure.

`audio invisible` defaults to `--method auto`, an adaptive minimal-distortion profile sequence for the
requested scheme and pre-strip vendor metadata. AudioSeal and Perth v1 try spectral then combined processing; WavMark and SilentCipher-44k
try neural then combined; when pre-strip C2PA/container metadata identifies ElevenLabs (`("dac", "encodec", "combined")` on mono, stereo, or tagged 5.1/7.1 when `dac` is installed, else `("encodec", "combined")`) or Google SynthID (`("dac", "combined")` on the same layouts when `dac` is installed, else `dac` alone, which fails closed), `--method auto` skips `spectral` and routes directly to neural codec resynthesis; an explicit `--vendor` cohort (`google`, `elevenlabs`, `suno`, `udio`, `openai`, `cartesia`, `meta`, `resemble`, `microsoft`) selects the same kind of vendor-matched sequence when metadata was already stripped and authorizes processing without `--allow-unverified`; when `--scheme auto` or `--scheme all` detects a specific subset of public schemes on the source, `--method auto` adapts to the lightest profile sequence covering only the positively detected schemes before falling back to `combined`. Publication requires
a positive compatible source check and proceeds only after the
saved output is negative in whole-clip and window checks. A resistant or
incompletely checked candidate raises an error and preserves an existing output.
No targeted source signal exits 2 without writing. Hard failures exit 1.

Use `--allow-unverified` (for example with `--method dac` on proprietary audio marks or alongside `--no-verify`) to process without a local source-positive decoder match while still enforcing the saved-file SNR, RMS, and 1-second active-window integrity gate. `--force` additionally overrides final decoder and quality checks on the last attempt. Unverified processing exits 0 with
`verified_removed: null`; this is not a removal certificate. `--verify` is the
default. The default output is `SOURCE_processed.wav`; `-o` / `--output`
accepts `.wav`, `.flac`, `.mp3` (`libmp3lame` 256 kbps CBR with Xing header), `.m4a` (`aac` 192 kbps with faststart), `.ogg`, `.oga`, `.opus` (`libopus` 192 kbps), `.aac` (`aac` 192 kbps ADTS), or `.mka` (`flac` in Matroska). Encoding discards AI metadata and preserves native sample
rate and channels (Opus internally codes on its RFC 7845 48 kHz grid and resamples back to the source clock during fidelity measurement). Reports measure fidelity (including `worst_window_rms_ratio` across 1-second active windows and stereo `max_side_mid_drift_db`) and run decoder verification on
the actual saved file, so lossy recompression relapses and mid-track dropouts are caught before publication. The filter changes audible content and can fail on tonal inputs.
See [audio coverage](audio-video-removal.md) for tested families and limits.

`audio metadata` inspects (`--check`, default) or strips (`--remove`) AI creation tags and C2PA manifests on standalone audio files without altering PCM/compressed audio frames via stream-copy remuxing (`-o` / `--output` defaults to `<source>_clean<ext>`; `--keep-standard/--remove-all` controls non-AI tags; `--json` emits JSON).

`audio all` provides the predictable single-file product contract (`<source>_clean<ext>` by default): it always strips verified AI metadata (`--keep-standard/--remove-all`) and, when `--invisible` is passed (`--no-invisible` is the default), runs compatible waveform watermark removal (`--scheme auto` by default, plus `--method`, `--vocoder-device`, `--vendor`, `--allow-unverified`, `--force`, `--verify/--no-verify`, `--low-memory`, `--stream`, `--json`). An explicit `--vendor` cohort implies the waveform stage even without `--invisible`. When `--invisible` finds no positive compatible watermark (and neither `--allow-unverified` nor `--force` is set) and the source carries no AI metadata, `audio all` copies the source file byte-for-byte rather than re-muxing container headers unnecessarily.

`audio batch` processes every supported audio file in a top-level directory (`-o` / `--output-dir` defaults to `<dir>_clean`, `--mode [all|metadata]` defaults to `all`, with `--invisible/--no-invisible`, `--scheme`, `--method`, `--vocoder-device`, `--vendor`, `--allow-unverified`, `--force`, `--verify/--no-verify`, `--low-memory`, `--keep-standard/--remove-all`, and `--json`). Pass `--low-memory` on `audio invisible`, `audio all`, or `audio batch` to evict cached neural codec and detector weights from RAM/VRAM immediately after each call or file.

`--method spectral` fixes the spectral profile. `--method encodec` selects
optional neural resynthesis at 24 kbps.
`--vocoder-device` (`auto` default, `cpu`, `mps`, or `cuda`) applies to EnCodec,
`combined`, DAC, and SNAC; `spectral` requires `cpu`.
Neural resynthesis can change speech timbre or music. On 2-channel stereo tracks, EnCodec resynthesizes in Mid/Side (`M/S`) space rather than independent L/R passes so stereo phantom-center phase coherence is preserved, while retaining native channels, rate, and sample-frame count using bounded context.
It has the same source-positive and saved-output verification gate, and can
fail even when a whole-clip reading is negative but a window remains positive.
`--method combined` applies the spectral filter followed by that neural codec.
It uses the same optional extra. Automatic neural
fallback also needs `audio-codec`; if unavailable, output remains unpublished.
`--method dac` selects the FP32 44.1 kHz 3-codebook
Descript Audio Codec (`audio-codec` extra, pinned to
`descript/dac_44khz@c1bc521685adf9cfe247bc39a5ca58917eda1ac4`, MIT weights in
`licenses/dac/NOTICE.txt`, defaulting `preserve_hf_hz=14000.0` on `>= 44.1 kHz` tracks). It accepts mono, stereo, and tagged 5.1/7.1 layouts,
and runs single-pass resynthesis up to 30 seconds
and raised-cosine overlap-window streaming on longer audio (`> 30 s`),
with raw mid/side stereo, stem RMS preservation, exact
rational half-open clock (`44.1 kHz` internal grid, preserving up to `22.05 kHz`
bandwidth), and bitexact WAV writer. EnCodec bandwidth does not apply;
`--vocoder-device` accepts `cpu` (default), `mps`, or `cuda`. `auto` selects DAC only when pre-strip metadata identifies
ElevenLabs (falling back to `encodec` / `combined` if DAC is absent or the
layout is unsupported) or Google SynthID (failing closed rather than falling back to
`encodec` if DAC is absent or the layout is unsupported), or when `--vendor` names a cohort.
`--method snac` selects the FP32 44.1 kHz 3-codebook
Multi-Scale Neural Audio Codec (`audio-codec` extra, pinned to
`hubertsiuzdak/snac_44khz@873ebef9718b89660340c6f55a2b515e98cfa1d9`, MIT weights in
`licenses/snac/NOTICE.txt`, defaulting `preserve_hf_hz=14000.0` on `>= 44.1 kHz` tracks), accepting mono, raw mid/side stereo, and tagged 5.1/7.1
layouts with single-pass resynthesis up to 30 seconds and overlap-window streaming
on longer tracks across `--vocoder-device [auto|cpu|mps|cuda]`; `auto` excludes `snac`.
Long audio tracks (`> 30 s`) are verified across
bounded `<= 30 s` whole-clip segments in addition to sliding windows, missing
requested decoders emit actionable install hints, and JSON output includes
`quality_passed`. Each method must pass
the requested decoder checks on the saved output; none guarantees universal
watermark removal or erasure of every residual message bit.

### Process audio inside a video

```bash
remove-ai-watermarks video audio input.mp4 -o processed.mkv --stream 0 --json
remove-ai-watermarks video audio input.mp4 -o processed.mp4 --scheme auto --json
```

This command inspects and cleans every watermarked audio track by default (or restricts processing to one or more zero-based ordinals via repeatable `--stream N`) while copying encoded video and subtitle streams.
It accepts `--allow-unverified`, `--force`, `--verify/--no-verify`, `--stream`, `--scheme`, `--json`, and
`-o` / `--output` with the same verification and exit policy as `audio invisible`.
It also accepts `--method`, `--vocoder-device`, and `--vendor` with the same audio backend rules.
When `--stream` is omitted (or repeated), every targeted audio track in the container is inspected sequentially; unmarked tracks are stream-copied unchanged while each positively detected track is cleaned and verified in the final mux.
The default output is `SOURCE_audio_processed.mkv`. MKV and MOV store the selected audio as uncompressed float PCM and can substantially
increase file size. An explicit `.mp4` output encodes only the selected track as
lossy AAC with a target bitrate of 192 kbps. Final verification and fidelity
measurements use the delivered decoded audio, including AAC priming and tail trimming.
Source formats and other stream codecs must be compatible with that container.

The source audio timeline must be continuous. Before publication the command
checks the selected track's native rate, channels, decoded sample-frame count,
start offset, and the actual muxed watermark readings. Every other stream must
retain its packet payloads and PTS/DTS within one output time-base tick;
packet durations may differ by one source or output tick. Stream languages and
dispositions and copied-audio priming/tail trimming must survive, so a container
conversion that changes them is rejected. No samples are padded or truncated to
repair a failed count or timestamp check. Some nonzero-offset AAC timelines cannot
preserve those values and are rejected.
Failures preserve an existing output. Each stream reaches its own end, including
audio tails. Other audio tracks, visible marks, invisible video pixels, and
metadata are not cleaned by this command. Container conversion can change tags.
Use a separate visual or metadata command when that work is also needed.

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
signed value. Removing the manifest does not remove a referenced pixel
watermark, and a referenced content fingerprint may still be recomputed. A
fingerprint is reported as a durable-provenance signal, not as a watermark or a
pixel-removal target.

Machine readable output:

```bash
remove-ai-watermarks identify image.png --json
```

Metadata only inspection:

```bash
remove-ai-watermarks identify image.png --no-visible
```

Despite the historical option name, `--no-visible` skips all pixel detectors,
including visible marks, open DWT-DCT, and TrustMark. Metadata inspection still
runs.

`identify` also routes other media by file suffix: a video runs `video identify`
(with `--audio --audio-scheme auto` unless `--no-visible` is passed), an audio
file runs `audio identify --scheme auto` (`--no-visible` maps to
`--no-invisible`), and a text file runs `text identify`.

## Classify source/export pipeline from pixels (opt-in)

To classify an image's likely AI generation/export pipeline and model version
subfamily from pixels after metadata removal, install the `source-classify`
extra and run `classify-source`. `identify` never runs this command, and it is
not a SynthID detector:

```bash
uv tool install --force "remove-ai-watermarks[source-classify]"
remove-ai-watermarks classify-source image.png
remove-ai-watermarks classify-source image.png --json
```

`classify-source` evaluates all 10 provider classes (`openai`, `google`,
`bytedance`, `flux`, `stable_diffusion`, `meta`, `midjourney`, `ideogram`,
`firefly`, `xai`) and abstains as `unknown` when no provider clears its
calibrated pixel-pipeline threshold, while also reporting `Synthetic` (`yes` or
`no`, backed by `synthetic_detected` and `synthetic_probability`) so heavily
resized or recompressed AI images remain detectable even when generator-family
attribution honestly abstains. When a provider is accepted, the stacked Stage-2
head also reports `version_subfamily` (for example
`google:gemini_3_pro_image`, `openai:gpt_image_1_5`, `flux:flux_2_pro`,
`stable_diffusion:sdxl`, `midjourney:v6_1`). `watermark_truth` is always
`"unknown"`. Guide:
[source-pipeline classification](source-classify.md).

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
- checks the filled region once without changing it or retrying the fill;
- strips AI metadata from the output.

The re-encode never costs display fidelity: the ICC colour profile and the EXIF
orientation tag are carried over from the source (see [strip AI metadata](#strip-ai-metadata)).

The automatic command reports `Removed and validated` when the detector no
longer accepts an overlapping mark, `Partial` when an overlapping residual is
still detected, and `Post-removal validation unavailable` when the check fails.
The latter two still write the once-filled output and return success; they are
result-quality statuses, not requests for an automatic second edit.

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

Act on a mark you can see but the detector will not confirm, by naming it and
skipping the gate:

```bash
remove-ai-watermarks visible image.png --mark gemini --no-detect -o clean.png
```

`--detect` is the default. `--no-detect` removes the named mark's footprint
unconditionally, so it only makes sense together with an explicit `--mark`; on
an image that never carried that mark it inpaints a region for nothing. Prefer
`erase --region` when you know the geometry.

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

`erase` accepts `cv2` (the default for images), `migan`, `lama`, and `auto`, which
selects the best installed backend. The corresponding optional extra must be
installed for a learned backend. A video input is routed to `video erase`, where
`--track` tracks moving user-specified boxes across frames via pyramidal Lucas-Kanade
flow and Sobel/NCC template refinement.

`--dilate N` (default 3) grows every box by `N`
pixels before inpainting, which helps when a mark has a soft edge or a drop
shadow just outside the box you measured; it applies to every backend because it
shapes the mask. Like `visible`, `erase` defaults to `--backend auto` and strips AI metadata from
the output by default; pass `--keep-metadata` to retain it. The ICC colour profile
and the EXIF orientation tag survive either way.

## Strip AI metadata

Inspect metadata:

```bash
remove-ai-watermarks metadata image.png --check
```

Remove AI metadata and write a new file:

```bash
remove-ai-watermarks metadata image.png --remove -o clean.png
```

The same metadata commands accept supported audio containers. With ffprobe
available, exact native Suno and Suno Studio creation comments are reported as
`audio_generator`; ffmpeg is required for stream-copy removal outside ISOBMFF.
The ffmpeg route drops all container tags and chapters, including ordinary
ones; ISOBMFF uses targeted patches. This unsigned declaration is separate
from C2PA, fingerprints, and waveform
watermarks. A quiet metadata report does not verify the audio samples.

When `-o` is omitted, removal overwrites the source. Standard metadata is kept
unless you pass `--remove-all`; that includes the display tags -- the ICC colour
profile and the EXIF orientation -- which also survive every pixel-command
re-encode (`all`, `visible`, `invisible`, `erase`), because colour and geometry
are display fidelity, not provenance. Orientation is re-tagged, not re-applied:
the raster is untouched, so what displayed upright still displays upright.

A quiet `--check` or a successful `--remove` is not a clean verdict. The command
only inspects and strips embedded AI metadata; a pixel watermark such as SynthID
has no local decoder once that metadata proxy is gone. `identify` reports the
same limit.

The command also supports the audio and video containers listed in
[supported signals](supported-signals.md). ffmpeg must be available for the
non-ISOBMFF audio and video path, either from the `video` extra or the system.

## Identify and clean video

Install the video pixel and timestamp runtime for visible identification,
removal, and the complete pipeline:

```bash
uv tool install --force "remove-ai-watermarks[video]"
```

Inspect every locally supported video signal:

```bash
remove-ai-watermarks video identify input.mp4
remove-ai-watermarks video identify input.mp4 --audio --audio-scheme auto
remove-ai-watermarks video identify input.mp4 --json
remove-ai-watermarks video identify input.mp4 --no-visible
```

The default scans the complete clip for stable registered visible marks and
inspects supported metadata. Pass `--audio` (`--no-audio` is the default) with
`--audio-scheme` (default `auto`) to also inspect every audio stream in the
container with compatible public decoders and include `audio_streams` plus a
suggested `recommended_recipe` when signals are detected. A result with no signals is reported as unknown,
not clean, because proprietary pixel watermarks have no public local decoder.
`--no-visible` performs metadata-only inspection (or metadata + audio when `--audio` is set).
The text report lists invisible watermark hints; JSON exposes them as
`invisible_watermark_hints`. These name C2PA-declared watermark schemes,
supported SynthID metadata, or Amazon Nova Reel from Bedrock video provenance.
They do not confirm a surviving pixel watermark. Registered fingerprints are
not watermark hints.
C2PA soft-binding declarations alone leave the video AI verdict unknown.
They remain visible as metadata and, for watermark algorithms, as hints.

Use the complete locally verifiable cleaning path:

```bash
remove-ai-watermarks video all input.mp4 -o clean.mp4
```

It removes a stable supported visible mark when found and always strips
verified AI metadata. When no visual, metadata, or audio signal requires rewriting, it copies the source file byte-for-byte instead of re-muxing container headers. The source is never
overwritten.

Invisible regeneration is deliberately opt-in:

```bash
remove-ai-watermarks video all input.mp4 -o clean.mp4 --invisible
remove-ai-watermarks video all input.mp4 -o clean.mp4 --invisible --audio-invisible
```

That option is supported across all seven video containers (`.mp4`, `.mov`, `.m4v`, `.mkv`, `.webm`, `.avi`, `.flv`). It is lossy and uses the
same default SynthID profile as `video invisible` (preserving native resolution and frame rate with motion-compensated optical-flow stabilization by default, and supporting `--low-memory` on `video invisible`, `video all`, and `video batch` to evict cached neural models and empty GPU/MPS allocator caches between stages and files). Pass `--audio-invisible` (`--no-audio-invisible` is the default) alongside `--audio-scheme` (default `auto`), `--audio-method` (default `auto`), repeatable `--audio-stream N` (to restrict inspection/processing to specific zero-based audio tracks), and optional `--audio-allow-unverified` (or `--audio-force`) to also inspect and clean compatible watermarked audio tracks in the same atomic pipeline, yielding 4-layer audiovisual cleaning (metadata + visible marks + video pixels + audio tracks). Silent videos skip `scipy` audio-extra requirements even when `--audio-invisible` is set. To select the Seedance recipe:

```bash
remove-ai-watermarks video all input.mp4 -o clean.mp4 --invisible --profile seedance
remove-ai-watermarks video batch ./videos --invisible --profile seedance
```

The complete pipeline runs supported visible-mark removal before this pixel
stage. Unsupported visible overlays still require separate preparation.
`--profile seedance-experimental` requires `--invisible` in all/batch mode.
The batch keeps one VAE runtime across files and reports unsupported containers
as per-file failures.

Process all supported files in a top-level directory:

```bash
remove-ai-watermarks video batch ./videos --mode all
remove-ai-watermarks video batch ./videos --mode all --invisible --audio-invisible
remove-ai-watermarks video batch ./videos --mode visible
remove-ai-watermarks video batch ./videos --mode metadata
```

The batch runs sequentially, preserves successful outputs when another file
fails, and exits nonzero if any item failed. Visible no-op files are copied
byte-for-byte so the output directory remains complete. `--invisible` and `--audio-invisible` are
available only with `--mode all`.

## Identify and clean text

Inspect deterministic text provenance and character steganography in plain text,
Markdown, or HTML files:

```bash
remove-ai-watermarks text identify article.md
remove-ai-watermarks text identify article.md --json
```

`text identify` runs with no optional ML or pixel extras. It detects and decodes:

- C2PA 2.3/2.4 Appendix A.8 `C2PATextManifestWrapper` (`C2PATXT\0` encoded in
  trailing Variation Selectors `U+FE00`..`U+FE0F` and `U+E0100`..`U+E01EF`,
  prefixed by `U+FEFF`);
- C2PA 2.3/2.4 Appendix A.9 structured-text ASCII-armored manifest blocks
  (`-----BEGIN C2PA MANIFEST----- ... -----END C2PA MANIFEST-----`);
- C2PA 2.3/2.4 Appendix A.7 HTML `<script type="application/c2pa">` and
  `<link rel="c2pa-manifest">` elements inside `<head>`;
- `VSRMark` (`com.cognitive-proof.vsrmark.1`, C2PA soft-binding ID 39)
  interleaved Variation Selector payloads (`VSR1` header + `zlib.crc32`);
- The published Windows AI four-codepoint sequence (see
  [validation boundaries](text-watermark-validation.md#published-format-fixtures));
  matching bytes are forgeable and do not establish provider attribution;
- Unicode Tags (`U+E0000`..`U+E007F`) and zero-width / Variation Selector
  steganography.

Check or strip deterministic text metadata and character steganography
losslessly:

```bash
remove-ai-watermarks text metadata article.md --check
remove-ai-watermarks text metadata article.md --remove -o article_clean.md
```

When `-o` is omitted, `text metadata --remove` writes `<source>_clean` and never
overwrites the input. Valid Emoji presentation selectors (`U+FE0E`/`U+FE0F`),
keycap emoji, and Emoji `U+200D` ZWJ sequences are preserved. ZWJ is also
preserved in standalone Devanagari half forms, legacy Malayalam chillus,
and explicit Saurashtra conjuncts. The cleaner also
preserves isolated `U+200B` between Thai word characters or within recognized
inline and fenced code, contextual `U+00AD` discretionary hyphens within words,
paired `U+2060` word joiners around multiplication or division signs between
decimal digits, and `U+034F` combining grapheme joiners that block canonical
mark reordering. The complete Windows AI watermark sequence is removed before
classifying its neighbors. Recognized binary and StegCloak streams take
precedence over the contextual rules for `U+200B`, `U+00AD` and `U+2060`.
File I/O retains LF, CRLF and CR line endings.
The cleaner preserves ordinary non-breaking spaces (`U+00A0`) in prose.
Single selectors in registered standardized and ideographic variation pairs
also retain their glyph selection.

Install `remove-ai-watermarks[text]` for default Transformers text regeneration.
The historical `diffusion` extra remains compatible.

For best-effort disruption of statistical token-level LLM watermarks, regenerate
unlocked prose spans with a
local causal LM while locking low-entropy technical spans
(fenced code blocks, inline code, LaTeX math, URLs, file paths, HTML blocks/tags,
and Markdown headings) byte-for-byte. Single-dollar formulas require content
without whitespace at the inner delimiter edges and no digit immediately after
the closing dollar. Currency pairs such as `$500` and `$1,750` stay prose.

Examples:

```bash
remove-ai-watermarks text invisible article.md -o article_clean.md --seed 0
remove-ai-watermarks text all article.md -o article_clean.md
remove-ai-watermarks text all article.md -o article_clean.md --invisible --seed 0
```

Apple silicon callers can install `remove-ai-watermarks[text-mlx]` and select
`--rewriter-backend mlx --rewriter-model /path/to/local-mlx-checkpoint` with either
`text invisible` or `text all --invisible`. The default backend remains
`transformers`; `--rewriter-model` also accepts a Hugging Face model ID. MLX uses
the same fact/placeholder guards and requires an explicit clean generation stop.
Its availability does not certify a model's semantic fidelity or watermark removal.

To use a locally installed model already served by Ollama, install
`remove-ai-watermarks[text-api]` and select the Ollama backend explicitly:

```bash
remove-ai-watermarks text invisible article.md -o article_clean.md \
  --rewriter-backend ollama --rewriter-model qwen3:8b-q4_K_M --seed 7291
```

This backend batches sentences in document order, requests a fixed-length JSON
array and checks the assembled document's facts before writing. It uses one
request and accepts only an HTTP loopback address. Set `--ollama-url` for a
different loopback port. The server must support structured array output and
serve the named model. The same options work with `text all --invisible`.
`--allow-partial` permits unchanged prose slots from a valid Ollama response.
A malformed response or failed document fact check aborts before writing.

With the Ollama backend, `--strict-residual` enables an optional second
request for sentences that remain verbatim after the first draft or a
local restoration. The edits must pass sentence and whole-document fact checks.
A failed second pass leaves the destination intact. Use this option without
`--allow-partial` with either `text invisible` or `text all --invisible`.

```bash
remove-ai-watermarks text invisible article.md -o article_clean.md \
  --rewriter-backend ollama --rewriter-model qwen3:8b-q4_K_M --seed 7291 --strict-residual
```

The Python API additionally supports an explicitly configured
[`build_text_api_rewriter`](python-api.md#optional-api-text-rewriter) with the
`text-api` extra. CLI generation remains local; there are no API-key flags or
implicit API fallbacks.

See [text watermark and provenance research](text-watermark-research.md) for the
measured token-perturbation survival tables and architecture.

Regeneration is best-effort. The native Transformers and MLX backends refuse
an unchanged passage, token-budget truncation without EOS, or
candidates that fail its fact checks after
three attempts; an existing destination stays untouched on failure. Those checks
cover selected numerals, entities and polarity, not full semantic equivalence.
Claude and other providers' production text detectors require private access.

Use `--allow-partial` with `text invisible` or `text all --invisible` to retain
rejected paragraph clusters verbatim and continue. The CLI prints `PARTIAL` with
the retained prose-span count. An explicitly requested partial output returns
exit 0, so inspect that warning; it does not certify watermark removal. If all
prose is rejected, the command fails and preserves any existing destination.
Unexpected exceptions still abort. Without the flag, any rejected cluster aborts.

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
as a direct `moov` child, and a keyless `hdlr=mdir` metadata list). Keyed
`workflow` and `prompt` entries in the same metadata list are also inspected.
The inspector seeks past a large `mdat` to find a tail `moov`. Removal stream-copies the
container in bounded chunks, converts supported top-level provenance boxes to
same-size `free` boxes, and blanks the TC260 or generation key/value in place. Box sizes,
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

## Apply the video-pixel SynthID-removal profile

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
source is never overwritten. By default, 2D `synthid` preserves native resolution and native source frame rate, upscaling cleanly from its internal 512p VAE guide with CIE Lab low-frequency color anchoring and motion-compensated optical-flow stabilization. Use `--noise-std`, optional `--fps` (to cap the output frame rate),
`--batch-size`, `--seed`, and `--device` to control the regeneration. The
default noise level is `0.15`. It cleared both carriers in the 2026-07-29
short-clip calibration and the complete public eight-second Veo carrier in the
2026-07-31 full-clip check; `0.10` remained detected on that complete clip. This
calibration certifies the shipped operating point; the paired check above is an
optional runtime audit, not a separate result state.

On completion the command reports that video pixels were regenerated with the
calibrated profile, not that this file received a fresh negative verdict. It
separately prints `Audio watermark: UNVERIFIED` because source audio is copied
unchanged when present. `video all` uses the same wording unless
`--audio-invisible` is enabled, in which case it prints `ABSENT` when the source
has no audio stream, `NOT DETECTED` naming the public decoders run (while noting
that proprietary audio marks such as SynthID are not checked locally), or
`UNVERIFIED` when forced on a public-decoder-negative track (or when any track
remains unverified).

For owned audiovisual clips carrying proprietary audio marks without a local
public decoder (such as short Google Veo/Gemini clips with SynthID audio), opt in
explicitly to both visual regeneration and the experimental 3-codebook DAC pass:

```bash
remove-ai-watermarks video all veo.mp4 -o veo_clean.mp4 --invisible --profile auto --audio-invisible --audio-method dac --audio-allow-unverified
```

Prefer `--audio-allow-unverified` over `--audio-force` when processing a
public-decoder-negative track: `--audio-allow-unverified` keeps the saved-file
SNR, RMS, and 1-second active-window integrity gate active, whereas
`--audio-force` also overrides final decoder and quality checks. When provenance metadata was already stripped
and you know the AI generator cohort (`google`, `elevenlabs`, `suno`, `udio`,
`openai`, `cartesia`, `meta`, `resemble`, `microsoft`), pass `--audio-vendor` on
`video all` or `video batch` (or `--vendor` on `audio invisible`, `audio all`,
`audio batch`, `video audio`, and multimodal `all`) to select `dac` and run the
audio scrub without needing `--audio-allow-unverified`. Current `dac`
provider-oracle evidence covers short single-pass clips (`<= 10 s`);
`focalcodec` failed on a held-out 10 s native Gemini audio output.

## Apply the Seedance profile

```bash
remove-ai-watermarks video invisible input.mp4 -o clean.mp4 --profile seedance
```

The fixed recipe accepts owned SDR 8-bit and **10-bit HEVC/HDR** video files
across all 7 supported containers (`.mp4`, `.mov`, `.m4v`, `.mkv`, `.webm`,
`.avi`, `.flv`) with even dimensions, preserving native bit depth (`yuv420p10le`), color tags
(`color_primaries`, `color_trc`, `colorspace`, `color_range`), geometry, frame
timestamps, terminal video duration, and source audio. `--quality delivery`
(default) writes H.264 High CRF18 for 8-bit or HEVC (`libx265`, `hvc1`) CRF18
for 10-bit; `--quality master` retains lossless output. Legacy
`--profile seedance-experimental`, `--profile wan21-synthid-experimental`, and
`--profile wan21-seedance-experimental` remain accepted as aliases for
`seedance`, `wan21-synthid`, and `wan21-seedance`. `--device` supports the same
devices as SynthID. The SynthID tuning options
(`--noise-std`, `--fps`, `--batch-size`, `--seed`) must retain
their defaults when Seedance is selected. `--quality master` is rejected for
SynthID. The default profile is `auto`, which routes ByteDance/Seedance clips
to the Seedance pipeline automatically and SynthID/other clips to the SynthID
pipeline (or Wan 2.1 on eligible BF16 CUDA hosts).

This transform reproduced the matched-control CapCut outcome
recorded in [the study](seedance-video-study.md). No detector or external upload
runs during processing. The standalone invisible command expects visible labels
to have been handled already; `video all` runs its existing visible stage first.

## Apply the 3D Wan 2.1 video diffusion profiles and `--profile auto`

On hosts with a hardware BF16-capable NVIDIA CUDA GPU (Ampere or newer; ROCm and
pre-Ampere emulation are excluded) or explicit Apple Silicon `--device mps`, two
3D causal video diffusion profiles preserve native resolution and frame rate on
SDR 8-bit and 10-bit HDR clips across all 7 containers (`.mp4`, `.mov`, `.m4v`,
`.mkv`, `.webm`, `.avi`, `.flv`), using 1080p-guided high-frequency texture
restoration above `1920x1080` and streaming clips longer than 193 frames through
overlapping 3D causal windows with 17-frame raised-cosine temporal crossfades
directly to the encoder without buffering the full clip in host RAM.
`--profile auto` automatically routes eligible ByteDance/Seedance clips
(`>= 8 GiB` total GPU VRAM on BF16 CUDA, stepping down from 193-frame to
81-frame chunks on 1080p or `< 16 GiB` CUDA and skipping the UMT5 text encoder
whenever the static neutral prompt embedding is cached locally) to
`wan21-seedance` on BF16 CUDA (`seedance` on CPU/MPS) and other eligible clips
to `wan21-synthid` on BF16 CUDA (`synthid` fallback otherwise):

```bash
remove-ai-watermarks video all input.mp4 -o clean.mp4 --invisible --profile auto
remove-ai-watermarks video invisible veo.mp4 -o veo_clean.mp4 --profile wan21-synthid
remove-ai-watermarks video all seedance.mp4 -o seedance_clean.mp4 --invisible --profile wan21-seedance
```

Both load pinned Apache-2.0 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (`0fad780a534b6463e45facd96134c9f345acfa5b`,
attribution in `licenses/wan/NOTICE.txt`) with an FP32 tiled/sliced 3D
`AutoencoderKLWan`, BF16 `WanVideoToVideoPipeline`, and `UniPCMultistepScheduler`
(`flow_shift=3.0`, 20 steps, `guidance_scale=1.0`), using `strength=0.38` for
`wan21-synthid-experimental` and `strength=0.12` for `wan21-seedance-experimental`.
Both accept `--quality [delivery|master]`, `--seed`, and `--device [auto|cuda|mps]`
(explicit `mps` runs the transformer in FP16 on Apple silicon; `auto` and
`--profile auto` never select MPS), while rejecting 2D VAE SynthID tuning options. Neither performs per-file oracle
verification.

## Remove a supported visible video mark

```bash
remove-ai-watermarks video visible input.mp4 -o clean.mp4
remove-ai-watermarks video visible veo.mp4 --mark veo -o veo_clean.mp4
remove-ai-watermarks video visible seedance.mp4 --mark seedance -o seedance_clean.mp4
remove-ai-watermarks video visible dola.mp4 --mark dola -o dola_clean.mp4
remove-ai-watermarks video visible hailuo.mp4 --mark hailuo -o hailuo_clean.mp4
remove-ai-watermarks video visible kling.mp4 --mark kling -o kling_clean.mp4
```

The command supports the moving Sora mascot with or without its wordmark, two Veo
corner variants, the Seedance boxed `AI` label, the `Dola AI` text label, the
composite `MINIMAX | hailuo AI` label, and the bottom-right Kling AI label. Sora
searches the whole frame at multiple scales and verifies the complete mascot
shape when the wordmark is absent. The other detectors search bounded
lower-frame regions with separate synthetic silhouettes. Kling additionally
requires its bright low-saturation label near the frame edge. Every mark
requires a spatially recurring candidate across adjacent frames. Fixed marks
must also remain anchored instead of drifting with a scene object. Matching
provider provenance may relax the visual score only for registered
provenance-aware marks; metadata alone never creates a detection.

`--mark auto` is the default. It evaluates all eighteen supported video marks in one decode pass and
selects the first stable match in specificity order (`sora`, `veo`, `dreamina`, `jimeng`,
`jimeng_pill`, `seedance`, `doubao`, `dola`, `hailuo`, `vidu`, `wan`, `qwen`, `yuanbao`,
`baidu`, `runninghub`, `microsoft`, `generic_ai_label`, then `kling`), co-selecting spatially
disjoint vendor-compatible marks (such as `jimeng` + `jimeng_pill`) in a single pass. Their confidence scores are independently calibrated and
are not compared across providers. Valid Google AI-video provenance resolves a
stable Sora/Veo match in favor of Veo when their regions overlap or Veo spans
the full clip while Sora does not; it does not create a visual detection.
Pass an explicit `--mark` to scan only that provider, or `--trim-bumper` (also available on `video all`) to detect and trim a trailing full-screen static AI end-card bumper.

The video stream is transcoded and the complete original audio stream is
copied without truncating an audio tail that extends beyond the final video
frame. The CLI reports that copied audio as watermark-unverified. The encoder
probes the source stream and preserves supported 8-bit
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

Supported 8-bit SDR and 10-bit HDR (`yuv420p10le`, `yuv422p10le`, `yuv444p10le`,
`bt2020nc`, `smpte2084`, `arib-std-b67`) sources preserve their native bit depth
and color transfer tags through a 16-bit `bgr48le` pipe; unsupported pixel
formats still fail closed before ffmpeg starts instead of silently downconverting
to 8-bit SDR. On CPU, MI-GAN is the practical learned tier. LaMa remains an
explicit offline quality option because full-sequence inference is too slow and
memory-heavy for an online worker.

AI metadata is stripped from the encoded output by default. Use
`--keep-metadata` to retain mapped container metadata. When no temporally stable
mark is found, the command writes no output and exits with the no-visible-mark
status. The final path is replaced atomically only after ffmpeg completes, so a
failed encode does not overwrite an existing result.

For user-specified rectangular overlays on owned video clips, use `video erase`
(or pass a video file to top-level `erase`):

```bash
remove-ai-watermarks video erase input.mp4 --region 24,24,180,56 --frames 0:120 -o clean.mp4
remove-ai-watermarks erase input.mp4 --region 24,24,180,56 --frames 0:120 --dilate 5 --backend auto --json
```

Pass one or more `--region x,y,w,h` boxes, optional `--track` to track moving
overlays across frames, optional inclusive `--frames START:END`
to restrict erasing to a frame interval, `--dilate`, `--backend` (default `auto`),
`--strip-metadata` / `--keep-metadata`, `--temporal-consistency` /
`--no-temporal-consistency`, and `--json`. Top-level `erase` on a video always
keeps temporal consistency enabled.

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

For transparent inputs, use PNG or WebP output to retain transparency. The engine
preserves the original alpha channel, including partially transparent edges,
while regenerating RGB pixels. JPEG output cannot preserve transparency. This
also applies to `batch --mode invisible`.

### Choose a strength cohort

`--vendor` selects the cohort the default strength resolves from. `auto`
(the default) derives it from product-specific C2PA provenance for
OpenAI/Google/Microsoft. The standalone AI IPTC tag is a shared standard and
does not select Meta; use `--vendor meta` only when the source is independently
known. An explicit value both names the cohort for stripped files
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
| `qwen-zimage` | Default (Apache-2.0). Qwen-Image-2512 global pass plus a SAM-masked Z-Image face stage |
| `qwen21-zimage` | Opt-in Qwen-Image-2.1 global pass (`texture-fix-vae` + Canny ControlNet) plus the same Z-Image face stage and guarded fidelity refinement. **Qwen Research License: non-commercial research, evaluation, and personal use only (not for commercial use)** |
| `sdxl-zimage` | The same recipe and face stage on an SDXL global pass, at a higher denoise |
| `chroma-zimage` | The same face stage on an Apache-2.0 Chroma1 global pass; explicit OpenAI use needs a higher floor than qwen-zimage |
| `auto` | Pick the commercial-safe engine from provenance: sdxl-zimage for Google, chroma-zimage for Microsoft, qwen-zimage for OpenAI, unknown, and an explicitly selected Meta cohort |

**All profiles are CUDA-only.** There is no CPU or MPS profile for invisible-watermark
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

### Fidelity profile

`--pipeline qwen21-zimage` retains the Qwen-2.1 global and face stages,
then tries bounded geometric alignment. Source MSE and CPU LPIPS, evaluated with
the reference bounded to 1024 pixels per side, must both improve before accepting
the aligned result. Protected face rectangles remain byte-identical to the
completed face stage. This quality decision is not watermark-detection evidence.

The CLI supplies no automatic text selector. Python integrations can supply
independently qualified text geometry for neutral and colored flat-text recovery;
see [the engine API](python-api.md#fidelity-and-qualified-text-geometry).
Each accepted text group must improve its local source error, and the complete
text stage must also pass the perceptual guard. Detector confidence alone is not
qualification. Textured text can remain regenerated.

The profile uses its own Google operating point; other provider strengths follow
the Qwen-2.1 policy. Tiled inputs, reviewed manifests, images smaller than 64 pixels
on either side, and explicit fidelity deactivation retain the shared Qwen-2.1
recipe and its strength policy. An optional quality-stage error retains the
completed baseline. The default profile and `auto` routing remain unchanged.
The Qwen-2.1 profile requires compliance with the non-commercial Qwen Research
License; it is never selected by `auto`.

### Restore operator-verified text (Python API)

Operator-verified text restoration (`text_manifest=Path("verified-lines.json")` on
`InvisibleOptions` / `InvisibleEngine.remove_watermark`) is exposed exclusively
through the [Python API](python-api.md#restore-operator-verified-text) and has no
CLI flag. Install `remove-ai-watermarks[text-restoration]` when calling it from
Python.

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
It is a memory strategy, not a guarantee of better quality. `--tile-size` and
`--tile-overlap` set the tile geometry (defaults 1024 and 128 px); a larger
overlap costs time and hides seams.

### Every tuning option

These apply to `invisible`, `all`, and `batch`. Each has a measured default, so
reach for one only when you know why the default is wrong for your file.

| Option | Effect |
| --- | --- |
| `--strength` | Denoising strength, 0.0-1.0. Default is profile- and vendor-adaptive; see [choose a strength cohort](#choose-a-strength-cohort). Raising it removes more and drifts further from the original. |
| `--vendor` | Strength cohort, and an assertion that the watermark is present. |
| `--pipeline` | Profile: `qwen-zimage` (default), `qwen21-zimage` (Qwen Research License: non-commercial/personal use only), `sdxl-zimage`, `chroma-zimage`, `auto`. |
| `--seed` | Fixed at 0 by default: every profile is certified at that seed, and removal near the floor is seed-dependent. |
| `--controlnet-scale` | Canny ControlNet conditioning on the global stage. Higher keeps the original structure and text closer. |
| `--adaptive-polish` / `--no-adaptive-polish` | Overrides the profile's own choice. Unset lets the profile decide: off for `qwen-zimage` and `qwen21-zimage`, whose outputs already match the input's detail level, on for `sdxl-zimage`. |
| `--max-resolution` | Cap the long side before diffusion; 0 = native. |
| `--tile`, `--tile-size`, `--tile-overlap` | Tiled regeneration for large inputs, as above. |
| `--cpu-offload` | Stream model components between CPU and GPU to lower CUDA memory pressure, at the cost of speed. |
| `--hf-token` | Hugging Face token for the model download; also read from the environment and a local `.env`. |

There is no `--model`, `--steps`, `--guidance-scale` or `--device` option; the
profile pins all four. See [exit behavior](#exit-behavior) for what each command
returns.

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

The visible options, diffusion options, and `--preset [lossless-fast|balanced|deep-clean]`
(`lossless-fast` restricts `all` to metadata + visible stages without pixel/waveform
regeneration, `balanced` enables learned visible fills and detector-gated audio
cleaning, and `deep-clean` activates `--force` pixel/waveform regeneration) are also available on `all`.

`all` also routes other media by file suffix, with fixed defaults: a video runs
`video all` with audio inspection (pixel regeneration only with `--force`, `--preset deep-clean`, or an
explicit `--vendor`), an audio file runs `audio all --invisible --scheme auto`,
and a text file runs `text all` (regeneration only with `--force` or `--preset deep-clean`). On these
inputs `--vendor` also accepts the audio cohorts (`elevenlabs`, `suno`, `udio`,
`cartesia`, `resemble`); an audio-only cohort on an image is a usage error. Use
the media-specific commands for any other option.
After the visible stage, `all` reports whether the once-filled result was
validated as cleaned, still produced an overlapping residual (`Partial`), or
could not be validated. These statuses do not trigger another fill and do not
change the exit code by themselves.

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
remove-ai-watermarks batch ./media \
  --mode all \
  --preset balanced \
  --output-dir ./clean
```

`batch` discovers images and videos in `--mode visible` (ignoring sidecar audio or
text notes), and discovers images, videos, audio tracks, and text files under
`--mode metadata`, `--mode invisible`, `--mode all`, or `--preset [lossless-fast|balanced|deep-clean]`
alongside the invisible and full-pipeline diffusion controls (`--vendor` accepts both
image and audio cohorts on mixed directories). Run `batch --help` for the authoritative
option list.

For `visible` and `all` modes, the final summary includes counts for
`no watermark`, `cleaned`, `partial`, and `unvalidated`. Partial and unavailable
validation results are printed explicitly, but the once-filled outputs are kept
and the batch does not retry them automatically.

## Serve the CLI to agent clients

```bash
remove-ai-watermarks mcp
```

Runs a Model Context Protocol server on stdin and stdout whose tools
(`identify`, `remove`, `batch`, the audio and video-message tools, and one
`cli_<command>` tool per CLI command) each run one command of this CLI. Client setup and
the tool contract are in the [MCP server guide](mcp-server.md).

## Exit behavior

The CLI uses nonzero exit codes for meaningful incomplete outcomes, including
no detected target on commands that would otherwise regenerate or create a
misleading unchanged result, a batch directory with nothing to process,
processing errors, an input that is not a readable image or video, and a
required invisible step that could not run.

Scripts should check the process exit code and the output path. The detailed
per-command contract is maintained in
[module internals](module-internals.md#command-line-interface).

## Public-model video messages (experimental)

```bash
remove-ai-watermarks video message identify input.mp4 --scheme videoseal --reference "$ENCODER_MESSAGE_HEX" --json
remove-ai-watermarks video message remove input.mp4 -o processed.mp4 --scheme pixelseal --reference "$ENCODER_MESSAGE_HEX" --json
remove-ai-watermarks video message detect input.mp4 --scheme chunkyseal --json
```

`video message detect` is a blind presence check that needs no reference. It
combines the pinned public model's presence probability with cross-frame payload
stability and reports `detected`, `not_detected`, or `unverified` (when
`video-detect` is not installed), plus any recovered payload hex. It covers only
the pinned public model family; proprietary keys and SynthID remain unverified.

`--reference` must be the independently known encoder message: 64 hexadecimal
digits (`256` bits) for `videoseal` and `pixelseal`, or 256 hexadecimal digits
(`1024` bits) for `chunkyseal`. Deriving it by decoding the tested file makes the
source check circular. These explicit commands compare a known target under
pinned public VideoSeal, PixelSeal, or ChunkySeal weights. They do not identify AI origin, arbitrary
unknown messages, proprietary deployments, or complete information erasure.
The generic `identify`, `all`, and SynthID routes do not run these checks.

Install `video-detect` for inspection and codec-only processing; add `diffusion`
for VAE processing. ffmpeg and ffprobe are required for processing. The receiver
checks every decoded frame, four full-clip aggregations, and stride-1 12-frame
sliding windows with a context-completed tail, verifying decoded frame counts
against `ffprobe` packet timestamps. Missing models yield `unverified`; decode,
checkpoint-integrity, numerical, and completeness errors fail the command.

Removal requires a matched source and a fully negative saved candidate before
atomic publication. `--method auto` tries codec-only, zero-noise VAE, then fixed
VAE noise levels 0.15 and 0.25. `--method codec` avoids the VAE;
`--method vae --noise-std 0` selects an explicit round trip. `--long-side`
defaults to 512; output sampling is at most 12 fps. VAE runs float32 on
`--device auto|cpu|cuda|mps`. Processing can visibly change texture and motion.
The JSON result includes delivered-file SSIM, PSNR, temporal measurements, and
`quality_passed`.

Processing accepts one video stream with optional audio streams and uniform,
zero-based video timestamps. Outputs are MP4, MOV, M4V, MKV, WebM, AVI, or FLV, defaulting to
`SOURCE_message_processed.mp4`. Audio is copied with packet, timing, decoded
sample, language, and disposition checks. Its watermarks remain unverified.
Source metadata is dropped. Additional stream types and unsupported timelines
fail before publication. Existing outputs survive failures.

Exit 2 means skipped because the source was not qualified, exit 1 means an
error or a forced still-matching result, and exit 0 means a published result.
`--force` permits an unqualified source or a final still-matching candidate;
`--force --no-verify` permits explicit unverified processing. Read
`verified_removed` and the before/after statuses, rather than interpreting
exit 0 alone as verified removal.
