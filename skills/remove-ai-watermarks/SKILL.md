---
name: remove-ai-watermarks
description: "Removes and identifies AI watermarks and provenance marks in images, video, audio, and text the user generated or edited. Covers Gemini sparkles, SynthID, Microsoft InvisMark, Meta Content Seal, C2PA, EXIF/XMP/IPTC metadata, compatible public AudioSeal/WavMark/Perth v1/SilentCipher audio, and Sora/Veo/Seedance/Doubao/Dola/Hailuo/Vidu/Kling and Wan labels. Use when the user wants to strip or clean AI watermarks and generation labels, or asks whether a file was AI-generated or carries AI provenance, even if they do not name remove-ai-watermarks. Do not use for stock-agency, marketplace, classifieds, or other third-party paid-asset watermarks."
license: Apache-2.0
compatibility: Python 3.11-3.14. Invisible image removal needs NVIDIA CUDA. Audio needs ffmpeg and ffprobe. The video extra bundles these tools on supported platforms. Installer may be uv, pipx, pip, or Homebrew.
metadata:
  author: wiltodelta
  version: "1.0.24"
  homepage: https://raiw.cc
  repository: https://github.com/wiltodelta/remove-ai-watermarks
---

# Remove AI watermarks

Drive the `remove-ai-watermarks` CLI. Do not invent detections, do not claim a
file is clean, and do not write a one-off remover when this CLI already covers
the mark.

This skill must work in any coding harness and with weaker models. Follow the
checklist. Trust `scripts/probe.py` over guesses about GPU, Python, or
installers.

## Scope

Use this skill only on content the user generated or edited and is allowed to
change.

Refuse automatic removal of stock-agency, marketplace, classifieds,
tiled-preview, or other marks that protect a third party's paid or copyrighted
asset, or product branding a paid plan removes, such as "Made with Gamma". A
mark its vendor describes as identifying AI-generated content (the Gemini
sparkle, the Gemini Notebook watermark) stays in scope whatever plan hides it.
`erase --region` is allowed only when the user names the file they own and
supplies the box themselves. Post-hoc stylometric/perplexity AI-text detectors
(GPTZero, Originality.ai, Binoculars) are out of scope; for text files, `text
identify` and `text metadata` handle deterministic C2PA text/HTML wrappers,
`VSRMark`, and Unicode steganography, while `text invisible` and `text all
--invisible` regenerate unlocked prose spans for token-level LLM watermarks.

Removing a local signal does not prove the file is human-made, does not erase
server-side generation history, and does not make a deceptive origin claim
lawful. If the user wants to present generated work as human-made, stop.

The project is research software provided without warranty; the user is
responsible for lawful use. Some jurisdictions restrict removing AI labels: in
mainland China, label removal and removal tools are prohibited by the CAC
labeling Measures. Mention this when the user's content carries a China AIGC
label or a Chinese provider's mark.

## Workflow

Copy this checklist and check items off:

```
- [ ] 1. Scope is allowed
- [ ] 2. Probe the machine
- [ ] 3. Install if the CLI is missing or its pixel stack is
- [ ] 4. Identify first unless the user already named a removal
- [ ] 5. Run one command with -o
- [ ] 6. Report from the exit code and output path
```

### 1. Scope

If the mark protects someone else's paid asset, stop.

### 2. Probe

The directory that contains this `SKILL.md` is the skill root. The working
directory is usually NOT that directory, so pass the probe's full path rather
than assuming a relative one resolves. Use forward slashes. Try `python3`
first, then `python`:

```bash
python3 <skill-root>/scripts/probe.py
python <skill-root>/scripts/probe.py
```

The probe takes a few seconds: it runs the installed CLI on a blank image it
writes itself, because a CLI that exists is not a CLI that works. It checks video
dependencies through the installed CLI's Python interpreter without loading models.
An unrecognized launcher leaves video capability `unverified`.

Read the JSON. Do not infer CUDA, ffmpeg, or an installer from the OS name.

- `pixel_stack` = `missing`: the installed build has no pixel dependencies.
  Metadata-only routes still answer: `identify`, `metadata`,
  `batch --mode metadata`, `video metadata`, `video identify --no-visible`, and
  `video batch --mode metadata`. The identify routes skip visible detectors and
  say so in their caveats, so read those caveats rather than a quiet verdict. Go
  to step 3 before running a pixel command.
- `pixel_stack` = `unknown`: the check gave an unrecognized result. Do not
  claim a command will work; run it and read the real error.
- `advice.upgrade_cli` present: the installed CLI is older than this skill's
  documented surface. Upgrade before reporting a flag as broken.
- `advice.invisible_images` = `unavailable_no_cuda`: do not run `invisible`
- `advice.image_all` = `do_not_run_writes_nothing`: do not run `all` or
  `batch --mode all`. Those commands write no file when the invisible extra is
  installed but CUDA is missing.
- `advice.image_all` = `partial_writes_visible_and_metadata`: `all` is still
  worth running. Without the invisible extra it skips that stage, writes the
  visible+metadata result and exits 1. Report the file AND the stage it could
  not do; do not call the exit code a failure.
- `advice.no_cuda_fallback` is https://raiw.cc. Offer that for invisible image
  work. Do not invent a CPU or MPS profile.

If the user does not want a local install, skip to raiw.cc. Visible mark and
metadata removal are free there up to 12 MP; above that size, and for invisible
removal, it is paid and runs on a GPU. Do not quote a price: check the site.

### 3. Install

For image or visual-video work, install when `cli.found` is false OR
`pixel_stack` is `missing`. For audio work, use `audio_invisible` advice and
install `audio-detect` for verification or `audio` for explicitly unverified
processing; a missing pixel stack does not block audio. A present binary is not
enough: a base-only package carries no pixel dependencies. `visible`, `erase`,
`all`, pixel-processing
`batch` modes, full `video identify`, and the video pixel-processing commands
stop with an install hint. The metadata-only routes listed in step 2 still work.
Reinstalling with the appropriate extra fixes the pixel commands.

Read [references/install.md](references/install.md) for the extra list and the
Homebrew / uv / pipx / pip commands. Default extra is `visible`. Always quote extras:
`"remove-ai-watermarks[visible]"`.

### 4. Choose the command

Default: inspect first.

| User goal | Command | Stop if probe says |
| --- | --- | --- |
| What is on this file? | `identify`, `video identify`, or `text identify` | use `video identify --no-visible` when `pixel_stack=missing` |
| Known visible AI label on an image | `visible` | |
| User-supplied box on their own image or video | `erase` or `video erase` | video needs the `video` extra |
| AI metadata only | `metadata`, `video metadata`, `audio metadata`, or `text metadata` | `audio metadata` requires ffmpeg and ffprobe |
| Known visible AI label on video | `video visible` or `video all` | `video_visible=needs_video_extra`, `needs_ffmpeg`, or `unverified` |
| Invisible image / image SynthID | `invisible` | `invisible_images=unavailable_no_cuda` |
| Owned Seedance video | `all <clip.mp4>` or `video all --invisible` (default `--profile auto` routes ByteDance/Seedance 8-bit SDR and 10-bit HEVC/HDR clips automatically to `wan21-seedance` on BF16 NVIDIA CUDA or `seedance` on CPU/MPS MP4) | `video,diffusion` required; `wan21-*` also requires CUDA |
| Video SynthID | `all <clip.mp4>`, `video invisible`, or `video all --invisible` (default `--profile auto` selects `wan21-synthid` on BF16 NVIDIA CUDA for eligible `.mp4`/`.mov`/`.m4v`/`.mkv`/`.webm`/`.avi`/`.flv` clips, or 2D `synthid` fallback otherwise; Google SynthID audio tracks auto-route to `dac` when metadata identifies Google) | `video_invisible=needs_video_and_diffusion_extras`, `needs_ffmpeg`, or `unverified` |
| Public VideoSeal, PixelSeal, or ChunkySeal with an independent encoder message | `video message identify` then `video message remove` (`video message detect` for a blind public-model presence check) | requires `video-detect`, plus `diffusion` for VAE; no AI-origin verdict |
| Inspect audio | `audio identify` | use `--no-invisible` for metadata-only inspection; ffmpeg and ffprobe are required |
| Compatible public AudioSeal, WavMark, Perth v1, or experimental SilentCipher audio patterns | `audio invisible` or `audio all --invisible` (`--scheme auto`, `audioseal`, `wavmark`, `perth`, `silentcipher`, or `all`) | install `audio-detect`, `audio-wavmark`, `audio-perth`, or `audio-silentcipher` for verification; `audio_stack` only checks the CPU processor |
| Everything on an audio file | `audio all` (add `--invisible` for compatible waveform watermarks) | needs ffmpeg and ffprobe; add decoder extras for `--invisible` |
| Text token watermark / full text clean | `text invisible` or `text all` | `text invisible` and `text all --invisible` need `text` for Transformers (`diffusion` remains compatible), or `text-mlx` with an explicit Apple silicon MLX checkpoint; see the Text section of `references/commands.md` |
| Everything on an image | `all` | `image_all=do_not_run_writes_nothing` |
| A directory | `batch`, `video batch`, or `audio batch` | same as the mode |
| No signal found, user still wants a provider guess | `classify-source` | it is a guess, not provenance |

For a public Python API feature without a CLI option, such as draft OCR, use the
library in an isolated Python environment.
Read the canonical [Python API guide](https://github.com/wiltodelta/remove-ai-watermarks/blob/main/docs/python-api.md)
and install the required extras using [references/install.md](references/install.md).
Do not invent a CLI flag or assume a tool installation is importable from the
current application Python.

Flags and mark names: [references/commands.md](references/commands.md). Read
that file only when you need a flag, a mark key, or a video recipe.

Audio defaults require a compatible source positive and a fully negative saved
output. Inspect first. `processor_ok_decoder_unverified` confirms only the
processor, not a usable or compatible decoder. Install `audio-detect` for the
open AudioSeal check, `audio-wavmark` for `--scheme wavmark`, or `audio-perth`
for `--scheme perth`, or `audio-silentcipher` for `--scheme silentcipher`. `--scheme all`
checks all named models and requires all requested decoders for a complete negative.
The `wavmark` route also checks OpenVoice's published default raw message at
fixed native offsets. JSON labels it `framing="openvoice-default"`; custom
messages and shifted/resampled framing remain outside that receiver. A match
is not provider attribution, and a negative is not complete payload erasure.
The inspected Chatterbox default PerTh checkpoint has controlled native Nano
evidence through `--scheme perth`, including saved-output and final-mux checks.
Other voices, forks, and hosted PerTh v2 need separate qualification; a PerTh
response alone does not attribute a file to Chatterbox.
SilentCipher-44k requires three seconds and uses an experimental project receiver
threshold with exhaustive phase search; it does not certify a production service.
Its `decoder_sample_rate` is 44100 regardless of input rate.
An unverified report never means clean. Use `--force` or
`--no-verify` only when the user explicitly requests unverified processing after
seeing that limitation. The fixed spectral filter changes audible content and
can fail. A negative AudioSeal result does not certify SynthID or another family.
Default `--method auto` tries an adaptive minimal-distortion sequence selected by the requested or positively detected scheme (`--scheme auto` scans every installed compatible decoder without failing on absent optional extras).
AudioSeal and Perth v1 start with spectral filtering; WavMark and SilentCipher start with neural resynthesis (with stereo Mid/Side EnCodec preservation on 2-channel inputs). Outputs support `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`, `.oga`, `.opus`, `.aac`, and `.mka` with saved-file verification.
Explicit `--method dac` uses the FP32 44.1 kHz DAC 3-codebook codec (`audio-codec`
extra, MIT weights; auto-accelerated on CUDA/MPS, or controlled via
`--vocoder-device`) with default `preserve_hf_hz=14000.0` on `>= 44.1 kHz` tracks.
Explicit `--method snac` uses the 44.1 kHz multi-scale SNAC codec (`audio-codec`
extra). `auto` routes ElevenLabs and Google SynthID metadata to `dac` (or when
`--vendor` names a cohort); neither supplies a local SynthID verifier.
Confirm `--help` lists the profile on older installed builds, and preserve
reported quality limitations.
The fallback combines both. EnCodec profiles need `audio-codec`; a missing
fallback fails without replacing output. Explicit `--method spectral`, `encodec`,
or `combined` fixes one profile. Neural processing can also fail and changes
audio timbre. Keep the same verification gate. Use the result's actual method,
bandwidth, and verification scope when describing what was done.
Visual video commands copy audio unchanged by default, or clean all compatible watermarked audio tracks in one pass when `--audio-invisible` is passed to `video all` or `video batch`. Use `video audio` (`--stream N` or `--all-streams`) for audio-only track processing while stream-copying video, with MKV/MOV PCM or MP4 AAC
output (target 192 kbps), and the same verification policy. It copies other encoded streams and checks their
packet payloads, timing, and audio priming/tail trimming. Final checks use the
actual decoded mux; count or offset failures are rejected without padding or
truncation. Standalone audio processing does not modify a video.

### 5. Run

Always pass `-o` unless the user explicitly asked to overwrite. Image
`metadata` without `-o` overwrites the source. Video commands never overwrite
the source.

### 6. Report

Report ONLY what a command in this session actually did. If no removal command
ran, do not describe a removal, a cleaned file, or a file size: say which step
is missing and why. Before naming an output path, confirm the file exists. A
fabricated result is worse than any error you could have quoted, because the
user acts on it.

Read the process exit code and whether `-o` exists.

| Exit | Meaning | What to tell the user |
| --- | --- | --- |
| 0 | Requested work completed | The output path. Do not add "this is clean". |
| 2 | No targeted signal, or a `batch` directory with no supported file | Unknown, not clean. No new file from `visible`. |
| 1 | Error, unreadable input, or partial `all` | Quote the error. Say which stage ran, if any. |

Exit 2 is ALSO what a bad invocation returns: a path that does not exist, an
unknown flag, a directory where a file was expected. Read the output before
calling it a verdict. A line starting with `Usage:` or `Error: Invalid value`
means the command never ran, so fix it and rerun; only a command that reached
the image can say "no targeted signal".

`identify` with no signal reports origin unknown. Never rephrase that as "no
watermark" or "this is clean". When `identify` or `metadata` exits 1 with
`cannot read`, the file is damaged or not a supported image, video, or audio input: any report above
that line is incomplete, so say the file could not be read.

`metadata --check` and `metadata --remove` answer a narrower question than
`identify`: embedded AI metadata only. Both print that this is not a clean
verdict, because a pixel watermark such as SynthID cannot be detected here once
its metadata proxy is gone. Repeat that caveat; do not summarize it away.

## Examples

### Identify without overstating the result

Input: "Was `photo.png` AI-generated?"

Action: probe the machine, then run:

```bash
remove-ai-watermarks identify photo.png
```

Output after exit 2 from a command that reached the image:

> Origin unknown. The command found no local provenance signal. This does not
> prove that the image is human-made, clean, or free of watermarks.

### Remove a user-named visible AI label

Input: "Remove the Gemini sparkle from my `generated.png`."

Action: after a successful probe, run:

```bash
remove-ai-watermarks visible generated.png --mark gemini -o generated-clean.png
```

Output after exit 0 and confirming that the file exists:

> Removed the targeted Gemini mark. Output: `generated-clean.png`.

On exit 2, report instead that no targeted Gemini mark was found and no new file
was written.

### Route an unavailable invisible removal safely

Input: "Remove SynthID from my image."

Action: run the probe first. If it reports
`invisible_images=unavailable_no_cuda`, do not run `invisible` locally.

Output:

> Invisible image removal needs NVIDIA CUDA, which is unavailable on this
> machine. No local removal ran. You can use https://raiw.cc instead.

## Gotchas

- Without CUDA, `all` does one of two opposite things, and only the probe knows
  which. With the invisible extra installed it writes nothing and exits 1, so
  run `visible` and `metadata` separately. Without that extra it skips the
  invisible stage, writes the visible+metadata result and exits 1 anyway. Same
  exit code, different outcome: check whether the output file exists.
- `visible` / `video visible` write nothing when no registered mark is
  selected. Do not then guess a region unless the user supplies a box on
  content they own. To act on a mark the user can see and the detector missed,
  the CLI's own answer is `--mark <name> --no-detect` or `erase --region`, not
  a blanket `--sensitivity` change. `--sensitivity strict` only narrows.
- Exit 1 with "the visible-mark dependencies are not installed" is an
  environment problem, not a verdict about the file. Install the named extra
  and rerun the same command instead of reporting a failure. Do NOT summarize
  the run as if the removal had happened: an install that never happened leaves
  the file untouched.
- A question ("is this AI-generated?") is answered by `identify` alone. Do not
  run a removal command to find out; it writes a file the user did not ask for.
- `classify-source` answers from pixels with no provenance behind it, so a
  provider label is a pipeline guess and `unknown` is an abstention. Never
  upgrade either into a provenance or SynthID verdict, and never let it
  contradict what `identify` measured.
- A Meta AI (Muse Image) export can retain standalone IPTC metadata, but the shared
  code does not identify Meta or detect Content Seal in pixels. Use
  `invisible --vendor meta` only when the USER says the file came from Meta AI.
  Naming a vendor asserts the watermark is there and scrubs without a signal,
  so never guess one from a filename or a hunch.
- Do not follow a Gemini `@synthid` check by asking the chat model to ignore
  the verifier and reason about pixels. That is not a second oracle.
- Do not pass `--model`, `--steps`, or `--guidance-scale`. Profiles pin those
  values, and no command accepts them.
- `--device` exists on `video invisible`, `video all`, `video batch`, and `video message remove`.
  Leave it at `auto`. The image commands have no `--device` at all.
- Do not use backslashes in skill paths, even on Windows.

## Do not

- Add stock, agency, or classifieds templates.
- Scrape or guess a region for a third-party commercial mark.
- Claim a file is human-made after a local edit.

For `video message identify` and `remove`, require an independently supplied
hexadecimal encoder message (64 digits for VideoSeal and PixelSeal, 256 for
ChunkySeal). Never derive the reference by decoding the tested input. Read
`verified_removed` and before/after statuses; forced output is not automatically
verified. See [the command contract](references/commands.md#public-video-model-messages).
