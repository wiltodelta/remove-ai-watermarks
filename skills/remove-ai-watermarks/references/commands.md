# Command recipes

Run `remove-ai-watermarks COMMAND --help` when a flag is unclear. The runtime
help is the source of truth.

## Images

```bash
remove-ai-watermarks identify image.png
remove-ai-watermarks identify image.png --json
remove-ai-watermarks identify image.png --no-visible
```

`--no-visible` skips visible and open invisible pixel detectors. Metadata
still runs.

```bash
remove-ai-watermarks visible image.png -o clean.png
remove-ai-watermarks visible image.png --mark gemini -o clean.png
remove-ai-watermarks visible image.png --keep-metadata -o clean.png
```

Image mark keys: `gemini`, `doubao`, `jimeng`, `qwen`, `wan`, `kling`, `yuanbao`,
`dola`, `workbuddy`, `samsung`, `runninghub`, `baidu`, `liblib`, `liblib_pill`, `microsoft`,
`generic_ai_label`, `openart`, `genspark`, `jimeng_pill`. `dola` and `workbuddy` are new (since CLI 0.45.0); `genspark` is new (since CLI 0.68.0). Default
`--mark auto` removes every selected match. `microsoft` is the top-right AI
badge, separate from the invisible InvisMark watermark on the same vendor's
images. `jimeng_pill` is a weak detector and needs corroboration. The compact
`liblib_pill` likewise requires LiblibAI metadata or the bottom-center wordmark
(since CLI 0.39.0). `generic_ai_label` is a brand-less bare "AI生成" fallback for
OS/gallery-level AI-edit stamps with no vendor wordmark; synthetic-only
calibration, weaker evidence than the vendor-tuned marks above. `openart` and
`genspark` sit in the FRAME CENTER, not a corner; `openart` is UNCALIBRATED and
`genspark` carries provisional calibration (see `openart_engine.py` and
`genspark_engine.py`).

```bash
remove-ai-watermarks erase image.png --region 1640,1930,400,100 -o clean.png
```

Region is `x,y,width,height`. Repeat `--region` for more boxes, and pass
`--frames START:END` for video clips. Backends: `auto`, `cv2`, `migan`, `lama`.
Use this only with a user-supplied box on content they own.

```bash
remove-ai-watermarks classify-source image.png
remove-ai-watermarks classify-source image.png --json
```

Abstaining 10-provider source-pipeline classifier (`openai`, `google`,
`bytedance`, `flux`, `stable_diffusion`, `meta`, `midjourney`, `ideogram`,
`firefly`, `xai`) with Stage-2 model-version attribution and unified synthetic
detection (`synthetic_detected`, `synthetic_probability`) under resize and
recompression (since CLI 0.48.0), needs the `source-classify` extra. It is NOT
provenance and NOT a SynthID detector (`watermark_truth` is always `unknown`),
`unknown` is a family-attribution abstention rather than proof of camera
origin, and `identify` never runs it. Offer `classify-source` only when
`identify` found no signal and the user still wants a provider or synthetic
pixel estimate, and report it as a statistical pixel estimate rather than
provenance.

```bash
remove-ai-watermarks metadata image.png --check
remove-ai-watermarks metadata image.png --remove -o clean.png
```

Without `-o`, image metadata removal overwrites the source. Animated and static
`.gif` files are inspected and stripped losslessly across all frames (since CLI
0.68.0). Supported audio
containers use these metadata commands too. Exact native Suno/Studio creation
comments require ffprobe and appear as unsigned `audio_generator` declarations.
The ffmpeg route drops ordinary container tags and chapters too. Removing
these comments or C2PA does not verify or remove a waveform watermark.

```bash
remove-ai-watermarks invisible image.png -o clean.png
remove-ai-watermarks invisible image.png -o clean.png --force
```

CUDA only. Default pipeline is `qwen-zimage`. The local signals it acts on are
SynthID provenance, C2PA soft bindings, Microsoft InvisMark, and the open
DWT-DCT / TrustMark detectors. Use `--force` when the user knows the file
should be regenerated even though no local signal fired.

```bash
remove-ai-watermarks invisible image.png --vendor meta -o clean.png
```

Vendor cohorts: `auto`, `openai`, `google`, `microsoft`, `meta`. `auto` derives
the strength from C2PA provenance and falls back to a resolution curve. Naming a
cohort asserts the watermark IS present, so it also runs the scrub without a
local signal, exactly like `--force`. Set it only when the USER says where the
file came from. Image cohorts apply to `invisible`; multimodal `all` and `batch`
accept the full audio/video superset below.

Audio vendor cohorts: `auto`, `google`, `elevenlabs`, `suno`, `udio`, `openai`,
`cartesia`, `meta`, `resemble`, `microsoft`. Naming an audio cohort on `all`,
`batch`, `audio invisible`, `audio all`, `audio batch`, `video audio` (`--vendor`), or
`video all` / `video batch` (`--audio-vendor`) asserts the waveform mark is
present, selects a vendor-matched profile sequence (`dac` first for most cohorts,
`spectral` first for `meta`) when metadata was already stripped, and runs without needing `--allow-unverified` while keeping the audio
integrity gate active.

Meta Muse Image exports can retain standalone IPTC metadata, but that generic
code does not identify Meta or detect Content Seal in pixels. `--vendor meta`
supplies the route only when the user confirms the source.
There is no `--model`, `--steps`, or `--guidance-scale` flag, and no `--device`
on the image path: auto-detection already finds the only device these profiles
run on.

Pipeline profiles: `qwen-zimage` (the default), `qwen21-zimage`, `sdxl-zimage`,
`chroma-zimage`, `auto`. All are CUDA-only and all install from the same
`qwen-zimage` extra. `qwen21-zimage` (since CLI 0.51.0) uses `Qwen/Qwen-Image-2.1` under the
**Qwen Research License** (non-commercial research and personal use only; do
not use `qwen21-zimage` for commercial workflows).
Since CLI 0.57.0, `qwen21-zimage` includes bounded geometry by default and local quality acceptance
under the same license restrictions. Its CLI path does not run automatic OCR.
It falls back to the shared Qwen-2.1 recipe for tiles and reviewed manifests.
Quality improvement does not prove watermark removal.
`auto` picks a commercial-safe engine from the file's own provenance -- sdxl-zimage for Google,
chroma-zimage for Microsoft, qwen-zimage for OpenAI (since CLI 0.39.0), Meta and unknown
-- so prefer it over naming an engine when the user has no reason to care. `sdxl-zimage` is the
heavier alternative and `chroma-zimage` is the Apache-2.0 global stage.

Verified-text manifests work with every profile (since CLI 0.40.5). Each profile
uses its own already loaded VAE donor, so `auto` keeps the normal SDXL route for a
Google image instead of switching engines.

```bash
remove-ai-watermarks all image.png -o clean.png
remove-ai-watermarks all track.mp3 --preset deep-clean -o clean.mp3
remove-ai-watermarks batch ./media --mode all --preset balanced --output-dir ./clean
```

`batch` modes: `visible`, `invisible`, `metadata`, `all`. Both `all` and `batch` accept `--preset [lossless-fast|balanced|deep-clean]` (which switches default `batch` mode to `all`); `batch` processes visual media (images and videos) in `--mode visible` and mixed directories of images, videos, audio, and text files in `metadata`, `invisible`, and `all` modes.

## Video

`video identify` reports `invisible_watermark_hints` in JSON and names them in
text: registered C2PA watermark schemes, supported SynthID metadata, and Amazon
Nova Reel inferred from Bedrock video provenance. They are metadata or policy
hints, not pixel detections or removal verification; fingerprints are excluded.
C2PA soft-binding declarations alone keep the AI verdict unknown; the hints
remain available alongside any independent AI-origin evidence.

```bash
remove-ai-watermarks video identify input.mp4
remove-ai-watermarks video identify input.mp4 --audio --audio-scheme auto
remove-ai-watermarks video identify input.mp4 --json
remove-ai-watermarks video identify input.mp4 --no-visible
remove-ai-watermarks video all input.mp4 -o clean.mp4
remove-ai-watermarks video metadata input.mp4 --check
remove-ai-watermarks video metadata input.mp4 --remove -o clean.mp4
remove-ai-watermarks video visible input.mp4 -o clean.mp4
remove-ai-watermarks video visible input.mp4 --mark veo -o clean.mp4
remove-ai-watermarks video erase input.mp4 --region 10,10,120,40 --frames 0:48 -o clean.mp4
remove-ai-watermarks video invisible input.mp4 -o clean.mp4
remove-ai-watermarks video batch ./videos --mode all
```

Video mark keys: `sora`, `veo`, `dreamina`, `jimeng`, `jimeng_pill`, `seedance`, `doubao`,
`dola`, `hailuo`, `vidu`, `wan`, `qwen`, `yuanbao`, `baidu`, `runninghub`, `microsoft`,
`pixverse`, `luma`, `heygen`, `sync_labs`, `tiktok_ai`,
`generic_ai_label`, `kling` (since CLI 0.74.0). Default `--mark auto` selects the first temporally stable match in
that order and co-selects spatially disjoint vendor-compatible marks (such as `jimeng` +
`jimeng_pill`) in a single pass. Pass `--trim-bumper` on `video visible` or `video all` to trim
a trailing static AI outro card, and `--track` on `erase` or `video erase` to track moving
user-specified `--region` boxes across frames.

`video invisible` and `video all --invisible` accept MP4, MOV, M4V, MKV, WebM, AVI, and FLV.
That is the default SynthID profile. It is lossy; its shipped noise profile
is pinned, so do not lower it. Motion-compensated `--temporal-stabilize` is enabled by default (`--no-temporal-stabilize` disables it), `--keep-fps` preserves the native source frame rate instead of resampling to 12 fps, `--native-guide` (since CLI 0.74.0) runs fast 512p 2D VAE regeneration and upscales cleanly to native resolution and frame rate, and `--low-memory` (since CLI 0.75.0) on `video invisible`, `video all`, `video batch`, `audio invisible`, `audio all`, and `audio batch` evicts cached neural models and empties GPU/MPS caches after processing. On `video all` and `video batch`, `--audio-invisible` (`--audio-scheme`, `--audio-method`, `--audio-stream`, `--audio-allow-unverified`, `--audio-allow-positive`, `--audio-allow-quality-override`, or alias `--audio-force`) inspects and cleans compatible watermarked audio tracks in the same pipeline (since CLI 0.63.0). For proprietary audio marks with no local public decoder (such as short Veo/Gemini SynthID or ElevenLabs audio), pass `--audio-allow-unverified` (`--audio-invisible --audio-method dac --audio-allow-unverified`, or `--audio-method auto` when ElevenLabs/Google SynthID C2PA metadata is present), which processes unverified audio while keeping the SNR/RMS/window integrity gate active (`--audio-force` also enables `--audio-allow-positive` and `--audio-allow-quality-override`; `--audio-method auto` without vendor metadata or a positive public decoder runs `spectral` only).

For an owned Seedance video, `--profile auto` routes ByteDance/Seedance clips automatically, or opt in explicitly with `--profile seedance` (legacy alias `seedance-experimental`):

```bash
remove-ai-watermarks video all input.mp4 -o clean.mp4 --invisible --profile seedance
remove-ai-watermarks video invisible label-cleaned.mp4 -o clean.mp4 --profile seedance
remove-ai-watermarks video batch ./videos --invisible --profile seedance
```

Seedance accepts SDR 8-bit and 10-bit HEVC/HDR across all 7 containers (`.mp4`, `.mov`, `.m4v`, `.mkv`, `.webm`, `.avi`, `.flv`) with even dimensions. `--quality delivery`
(default) writes H.264/H.265 CRF18; `--quality master` retains a larger
lossless-coded master. Leave SynthID tuning flags at their defaults; nondefault
values are rejected for this fixed recipe. All/batch requires `--invisible`
with this profile. Only all mode runs the supported visible stage first;
unhandled visible overlays need separate preparation. The profile has
matched-control CapCut evidence but performs no per-file detector request.

On machines with a hardware BF16-capable NVIDIA CUDA GPU (Ampere or newer; ROCm
and pre-Ampere emulation are excluded) or explicit Apple Silicon `--device mps`, two
3D video diffusion profiles (`wan21-synthid` and `wan21-seedance`; legacy aliases `wan21-synthid-experimental`, `wan21-seedance-experimental`) preserve native resolution and frame
rate on SDR 8-bit and 10-bit HDR clips across all 7 containers (using 1080p-guided high-frequency texture restoration above `1920x1080` and streaming
clips over 193 frames via overlapping 3D causal windows), and
`--profile auto` (the default on `all`, `video all`, `video batch`, and `video invisible`) automatically routes eligible ByteDance/Seedance clips
(`>= 8 GiB` total GPU VRAM on BF16 CUDA, stepping down from 193-frame to 81-frame chunks on 1080p or `< 16 GiB` CUDA) to
`wan21-seedance` on BF16 CUDA (`seedance` fallback on CPU/MPS) and other eligible clips to `wan21-synthid`
on BF16 CUDA (`synthid` fallback otherwise):

```bash
remove-ai-watermarks video all input.mp4 -o clean.mp4 --invisible --profile auto
remove-ai-watermarks video all input.mp4 -o clean.mp4 --invisible --profile wan21-synthid
remove-ai-watermarks video all input.mp4 -o clean.mp4 --invisible --profile wan21-seedance
```

Both `wan21-*` profiles use pinned Apache-2.0 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
(`strength=0.38` for SynthID, `strength=0.12` for Seedance, 193-frame chunks
with 17-frame raised-cosine temporal crossfades on longer clips), accept
`--quality [delivery|master]`, `--seed`, and `--wan-steps` (default certified 20; e.g. `--wan-steps 6` for fast scrubbing, since CLI 0.74.0), require `device=auto`, `cuda`, or an
explicit `mps` (FP16 transformer; `auto` never selects MPS), and
reject 2D VAE SynthID tuning flags. Neither performs per-file oracle verification.

Unlike the image path, `video invisible`, `video all` and `video batch` do
accept `--device [auto|cuda|mps|cpu]`. Leave it at `auto`. Set it only when the
user names a device, and never to claim a device the probe did not report.

Without `-o`, video commands write `<source>_clean` and keep the original.
Output extension must match the input container.

## Audio (since CLI 0.63.0)

```bash
remove-ai-watermarks audio identify input.wav --json
remove-ai-watermarks audio identify input.m4a --scheme auto --json
remove-ai-watermarks audio identify input.m4a --no-invisible
remove-ai-watermarks audio metadata input.mp3 --check
remove-ai-watermarks audio metadata input.mp3 --remove -o clean.mp3
remove-ai-watermarks audio invisible input.wav -o processed.flac --json
remove-ai-watermarks audio all input.mp3 -o clean.mp3 --invisible
remove-ai-watermarks audio batch ./tracks -o ./tracks_clean --mode all --invisible
```

`audio identify` and `audio invisible` accept `--stream N`, a zero-based audio-track ordinal (`audio identify` also emits `recommended_recipe` when AI metadata or a compatible watermark signal is found). `audio metadata` checks (`--check`) or strips (`--remove`) AI metadata via stream-copy remuxing (`--keep-standard/--remove-all`, `-o`). `audio all` strips AI metadata and, with `--invisible`, removes compatible waveform watermarks (`--scheme auto` default, writing a metadata-cleaned passthrough when no positive compatible watermark is detected). `audio batch` runs `--mode [all|metadata]` across a directory (`-o` / `--output-dir`). Inspection without
models needs ffmpeg and ffprobe. `--scheme audioseal` (default on `identify`/`invisible`) needs
`audio-detect`; `--scheme auto` scans every installed compatible decoder without failing on absent optional extras; `--scheme wavmark` needs `audio-wavmark`; `--scheme perth` needs `audio-perth`; `--scheme silentcipher` needs `audio-silentcipher`; `--scheme all` checks
all named models and needs all named decoder extras. WavMark also checks the
published OpenVoice default raw framing at native samples 0 and 32000. JSON
labels it `framing="openvoice-default"` with its native `decoder_sample_rate`.
`silentcipher` reports `decoder_sample_rate=44100` regardless of input rate.
For OpenVoice, custom messages and shifted/resampled framing remain outside
the receiver; its binomial-tail diagnostic is not a calibrated false-positive probability. For validated 24 and 48 kHz sources, AudioSeal checks the resampled 16 kHz and native sample clocks independently; other native clocks are not added automatically. JSON `decoder_sample_rate` identifies each AudioSeal receiver; a verified negative requires both clocks to be negative. Removal requires a compatible source positive and a
fully negative saved output. No positive source exits 2 without writing.
Rejected verification exits 1 and preserves an existing output. `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`, `.oga`, `.opus`, `.aac`, and `.mka`
outputs preserve channels and sample clock (Opus codes internally at 48 kHz and resamples back during fidelity checks), with AI metadata
discarded and saved-file decoder verification enforced after encoding. The spectral profile changes audible content and can fail.

`--method auto` (default) tries spectral filtering then combined filtering and
neural resynthesis for AudioSeal and Perth v1, neural then combined for WavMark and SilentCipher (using Mid/Side stereo on 2-channel EnCodec passes), routes mono, stereo, or tagged 5.1/7.1 audio carrying ElevenLabs metadata to `dac` (or `encodec`/`combined` when `dac` is absent) and Google SynthID metadata exclusively to `dac`/`combined` (failing closed rather than falling back to `encodec` when `dac` is absent or the layout is unsupported) instead of `spectral`, and adapts to positively detected schemes under `--scheme auto` or `all` before falling back to `combined`. Each candidate must pass the saved-output gate before publication.
`--method spectral`, `encodec`, `combined`, `dac`, or `snac` fixes one profile. Neural profiles
need `audio-codec`; a missing required fallback fails without replacing output.
`--bandwidth` accepts `6`, `12`, or `24` kbps, default `24`, for EnCodec profiles.
Keep the requested decoder extras for verification. Neural processing changes speech
timbre or music and can retain a watermark. Whole-clip and window checks must
both pass; a negative whole-clip result alone is insufficient. These options
also apply to `video audio` and its MCP tool.

`--method dac` selects the 44.1 kHz Descript Audio Codec profile
(`descript/dac_44khz`, MIT weights) fixed at 3 codebooks on FP32 with the
`audio-codec` extra (since CLI 0.63.0). It runs single-pass resynthesis up to
30 seconds and raised-cosine overlap-window streaming on longer audio (`> 30 s`),
supporting mono, raw mid/side stereo, and tagged `5.1`/`7.1` surround stem
decomposition with `<120 Hz` LFE filtering while preserving full 22.05 kHz
bandwidth (defaulting `preserve_hf_hz=14000.0` on `>= 44.1 kHz` tracks). `--vocoder-device` accepts `auto` (default), `cpu`, `mps`, or `cuda`; EnCodec
bandwidth does not apply, and `auto` selects `dac` when pre-strip metadata or an
explicit `--vendor` cohort identifies a wideband AI audio generator.

`--method snac` selects the 44.1 kHz Multi-Scale Neural Audio Codec profile
(`hubertsiuzdak/snac_44khz`, MIT weights) fixed at 3 multi-scale codebooks
(strides 8, 4, 2) on FP32 with the `audio-codec` extra (defaulting `preserve_hf_hz=14000.0` on `>= 44.1 kHz` tracks). Like `dac`, it runs
single-pass resynthesis up to 30 seconds and overlap-window streaming on longer
audio (`> 30 s`), supporting mono, raw mid/side stereo, and tagged `5.1`/`7.1`
surround stems on `--vocoder-device [auto|cpu|mps|cuda]`; `auto` excludes `snac`.

`--allow-unverified` permits processing without a local source-positive decoder match (or with `--no-verify`) while keeping the audio SNR/RMS/window integrity gate active (`--allow-positive` permits a still-positive final candidate; `--allow-quality-override` overrides the integrity gate; `--force` enables all three).
`--method spectral --allow-unverified --no-verify` only needs the `audio` extra. Other
processing methods retain their own model extras. A still-positive written
candidate exits 1. Unverified processing exits 0 with
`verified_removed: null`, never a universal removal verdict. Default output:
`<source>_processed.wav`. Audio checks segment long tracks (`> 30 s`) into
`<= 30 s` bounded passes alongside short sliding windows; they do not certify
SynthID, Perth v2, or other production schemes. Visual video processing continues
to copy audio unchanged.

`video audio input.mp4 -o processed.mkv --stream 0 --json` (since CLI 0.63.0)
applies the same audio profile and verification policy inside a video (`--stream N` for one track, or `--all-streams` / repeatable `--target-stream N` to inspect and clean watermarked tracks while stream-copying unmarked tracks). It accepts
`--allow-unverified`, `--allow-positive`, `--allow-quality-override`, `--force`, `--verify/--no-verify`, `--stream`, `--all-streams`, `--target-stream`, `--scheme`, `--json`, and `-o` / `--output`.
Default output: `<source>_audio_processed.mkv`. MKV and MOV outputs store the
selected track as uncompressed float PCM, increasing file size. Explicit MP4 output
encodes only the selected track as lossy AAC (target 192 kbps); `--method auto`
evaluates the final muxed container and falls back to `combined` if `spectral`
relapses after AAC recompression. Other streams are copied and their packet
payloads, timing, and audio priming/tail trimming checked. Audio rate, channels,
sample-frame count, start offset, final-mux decoder results, and `quality_passed`
are checked.
Discontinuous audio, changed decoded counts/offsets, and incompatible muxes fail
without replacing an output. No samples are padded or truncated to repair AAC
priming or tail differences.
Other audio tracks and visual marks remain unchanged; metadata is not cleaned.

## Text

```bash
remove-ai-watermarks text identify article.md
remove-ai-watermarks text identify article.md --json
remove-ai-watermarks text metadata article.md --check
remove-ai-watermarks text metadata article.md --remove -o article_clean.md
remove-ai-watermarks text invisible article.md -o article_clean.md --seed 0
remove-ai-watermarks text all article.md -o article_clean.md
remove-ai-watermarks text all article.md -o article_clean.md --invisible --seed 0
```

`text identify` and `text metadata` need no optional extras: they inspect and
strip C2PA 2.3/2.4 Appendix A.8 (`C2PATXT\0`), Appendix A.9 (`-----BEGIN C2PA
MANIFEST-----`), Appendix A.7 (`text/html` `<head>`), `VSRMark`
(`com.cognitive-proof.vsrmark.1`), Unicode Tags (`U+E0000`..`U+E007F`), and
zero-width steganography while preserving valid Emoji `U+FE0E`/`U+FE0F` and
Emoji `U+200D` ZWJ sequences. Without `-o`, `text` commands write
`<source>_clean` and keep the original. `text invisible` and `text all
--invisible` lock code blocks, inline code, LaTeX math, URLs, file paths, HTML
tags, and Markdown headings verbatim and regenerate unlocked prose spans with an
unwatermarked local causal LM (default Transformers backend, `text` extra;
`diffusion` remains compatible).
Since CLI 0.63.0, both regeneration commands accept `--rewriter-model`,
`--rewriter-backend transformers|mlx|ollama`, `--ollama-url`, and `--strict-residual`. For local Apple silicon MLX checkpoints,
install `text-mlx` and pass
`--rewriter-backend mlx --rewriter-model /path/to/checkpoint`. `text all` requires
`--invisible` with these options. The MLX backend uses the same acceptance guards
and refuses truncated completions; it does not establish stronger removal or
meaning preservation. These options require a build whose `--help` lists them;
this pending surface is not a published release claim. This is best-effort:
invalid candidates can be refused, and a returned rewrite is not a verified
private-provider watermark removal. Review facts and language before using it.
Optional `--allow-partial` keeps rejected paragraph clusters unchanged and prints
`PARTIAL`; it requires regeneration and can leave marked prose. An explicitly
requested partial output returns 0. All-prose refusal still fails without replacing
the destination; default strict mode aborts on the first rejected cluster.
For local server backends and a second pass over retained sentences, check
the installed command's help for availability. See the
[CLI guide](https://github.com/wiltodelta/remove-ai-watermarks/blob/main/docs/cli.md)
for setup and constraints.

## MCP server

`remove-ai-watermarks mcp` (since CLI 0.46.0) serves `identify`, `remove`
and `batch` as MCP tools over stdin and stdout for MCP clients. Since 0.63.0,
it also serves `audio_identify`, `audio_invisible`, and `video_audio`.
The experimental `video_message_identify` and `video_message_remove` tools expose known-message video checks. It blocks
waiting for protocol messages, so do not run it from this skill; call the
commands above directly.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Requested work completed |
| 2 | No targeted signal (`visible` / no-signal `invisible`), or a `batch` directory with no supported file |
| 1 | Hard failure, an input that is not a readable image or video (`identify` and `metadata` included), or `all` skipped a required invisible stage |

A written file plus exit `1` from `all` is a partial result.

Click's own usage errors also exit `2` (missing file, unknown flag, directory
where a file was expected). They print `Usage:` or `Error: Invalid value` and
never touch the image, so they are not a no-signal verdict.

## Public video model messages

`video message identify SOURCE --reference HEX --scheme videoseal|pixelseal|chunkyseal --json`
checks a caller-known encoder message (`256`-bit / 64 hex digits for `videoseal`
and `pixelseal`, or `1024`-bit / 256 hex digits for `chunkyseal`).
`video message detect SOURCE --scheme videoseal|pixelseal|chunkyseal --json`
performs a blind public-model presence check without requiring `--reference`. Never derive
a verified removal reference circularly from the tested input. `video message remove` uses the same options,
with `-o OUTPUT`, `--method auto|codec|vae`, `--long-side 512`, and
`--device auto|cpu|cuda|mps`. Explicit `--noise-std` requires `--method vae`.

Install `video-detect`, plus `diffusion` for VAE. Auto tries codec, zero-noise VAE,
then noise 0.15 and 0.25. Output is at most 12 fps and defaults to
`SOURCE_message_processed.mp4`. Processing requires one video
stream (CFR, VFR, or non-zero start PTS) and optional audio, with
MP4/MOV/M4V/MKV/WebM output. Audio is copied and checked,
not cleaned. Metadata is dropped. Quality can change visibly; JSON output reports
delivered metrics and `quality_passed`.

Default publication requires a matched source and a saved-output negative over
every frame, four aggregations, and stride-1 12-frame sliding windows, plus an
`ffprobe`-verified complete frame count. Exit 2 skips unqualified
sources, exit 1 reports failure or a forced still-match, and exit 0 can include
explicitly unverified output. `--no-verify` requires `--force`; inspect
`verified_removed` and before/after reports. These checks do not establish AI
origin, detection beyond the pinned public models, provider compatibility, or
complete erasure.
