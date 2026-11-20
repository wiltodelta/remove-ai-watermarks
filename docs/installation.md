# Installation

Python 3.11 through 3.14 are supported.

## Default metadata mode

The default package provides:

- provenance inspection;
- AI metadata inspection and removal.

It installs Pillow, piexif, and c2pa-python for reading metadata directly from
files. It does not install NumPy, OpenCV, pillow-heif, Torch, diffusion models,
or invisible-watermark decoders.

Native Suno audio creation-comment inspection additionally requires ffprobe
on `PATH`; audio stream-copy metadata removal requires ffmpeg. These tools
are not bundled. Neither is needed for ordinary image metadata inspection.

Install it as an isolated command with uv:

```bash
uv tool install remove-ai-watermarks
```

Or with pipx:

```bash
pipx install remove-ai-watermarks
```

You can also install the Homebrew package on macOS or Linux:

```bash
brew install wiltodelta/tap/remove-ai-watermarks
```

Homebrew installs the `visible` extra by default. Each extra published with the
pinned package has a matching `--with-<extra>` option, and options can be combined:

```bash
# Seedance and SynthID video processing
brew install wiltodelta/tap/remove-ai-watermarks --with-video --with-diffusion

# The package's all group
brew install wiltodelta/tap/remove-ai-watermarks --with-all

# Base metadata/provenance tools, without the default pixel dependencies
brew install wiltodelta/tap/remove-ai-watermarks --without-default-extras

# Base plus HEIF support only
brew install wiltodelta/tap/remove-ai-watermarks --without-default-extras --with-heif

# List the options for the currently pinned release
brew options wiltodelta/tap/remove-ai-watermarks
```

To add features to an existing installation, use `brew reinstall` with the desired
options. Homebrew retains previously selected options on reinstall and upgrade.
To remove old extras or replace the selection exactly, uninstall the formula and
install it again with the complete desired option set. The `all` option uses the
package's own `all` group; other groups remain individually selectable.

The formula creates an isolated Python environment and installs the requested
extras with their dependencies. Python is supplied by Homebrew; the `video` extra
supplies FFmpeg and ffprobe. Models still download on first use, and each backend's
platform, wheel, and hardware requirements still apply. In particular, image
regeneration profiles require NVIDIA CUDA; video VAE processing supports CPU,
MPS, or CUDA with a compatible Torch stack.

On Intel Macs, the formula stages checksum-verified build tools from the official
Rust distribution and uses Homebrew's OpenSSL and pkgconf to compile current
cryptography dependencies. The compiler is temporary; no manual setup is needed.
Video engines use the installed PyAV decoder when OpenCV cannot open a stream;
the `video` extra already supplies it, including on builds without OpenCV FFmpeg support.

Without pixel extras, `identify`, `metadata`, `batch --mode metadata`,
`video metadata`, `video identify --no-visible`, and `video batch --mode metadata`
still provide metadata-only results.

## Visible watermark removal

Visible mark detection, OpenCV inpainting, and manual region erasing need the
`visible` extra:

```bash
uv tool install --force "remove-ai-watermarks[visible]"
```

Add `heif` only when the pixel path must decode HEIC, HEIF, or AVIF:

```bash
uv tool install --force "remove-ai-watermarks[visible,heif]"
```

## Audio processing

Audio inspection and processing require both ffmpeg and ffprobe on PATH.
`audio identify --no-invisible` uses the default package. Install the CPU
spectral processor separately from the compatible AudioSeal decoder:

```bash
uv tool install --force "remove-ai-watermarks[audio]"
uv tool install --force "remove-ai-watermarks[audio-detect]"
```

`audio` installs NumPy and SciPy, without OpenCV or Torch. `audio-detect` adds
the pinned open AudioSeal runtime and downloads hash-checked detector weights
on first use. It needs no CUDA. Default removal requires that decoder; explicit
`--method spectral --allow-unverified --no-verify` processing only requires `audio` while retaining the audio integrity gate. Outputs are WAV, FLAC, MP3, M4A, OGG, or Opus.
See [audio coverage and limitations](audio-video-removal.md).

Add `audio-codec` for EnCodec resynthesis with `--method encodec`, `combined`,
or an automatic neural profile. The default AudioSeal sequence starts with
spectral filtering and requires this extra if its neural fallback is needed.
Combine it with `audio-detect` for default compatible-source verification:

```bash
uv tool install --force "remove-ai-watermarks[audio-detect,audio-codec]"
```

This backend uses Torch and Transformers on CPU and downloads pinned,
hash-checked model/configuration artifacts on first use. It is a distinct
processing method, with its own quality and watermark-verification results.
The same extra also supplies explicit experimental `--method mimi`, a pinned
CPU FP32 codec with 32 default quantizers, explicit experimental
`--method focalcodec`, a pinned CPU FP32 continuous 6-layer WavLM feature-to-Vocos
resynthesizer (`lucadellalib/focalcodec_50hz` at revision
`d6d6d9524e52155c85193c2c3b8da1cf8842f019`), and explicit experimental
`--method dac`, a pinned FP32 44.1 kHz 3-codebook Descript Audio Codec (CPU by
default; `--vocoder-device mps` or `cuda` also apply)
(`descript/dac_44khz` at revision `c1bc521685adf9cfe247bc39a5ca58917eda1ac4`).
Each downloads separate safetensors and configuration artifacts.
[Mimi weights](https://huggingface.co/kyutai/mimi) are CC-BY-4.0 with
attribution recorded in `licenses/mimi/NOTICE.txt`,
[FocalCodec code and weights](https://huggingface.co/lucadellalib/focalcodec_50hz)
are Apache-2.0 with attribution and third-party notices in
`licenses/focalcodec/`, and [DAC weights](https://huggingface.co/descript/dac_44khz)
are MIT with attribution in `licenses/dac/NOTICE.txt`. Model licenses remain
distinct from the library and Transformers code licenses. None of the three
methods is selected automatically, and all require the requested decoder extra
for checks.

Add `audio-vocoder` for explicit experimental `--method bigvgan`. This installs
Torch, librosa, and Hugging Face Hub alongside the numerical audio stack. The
runtime downloads revision-pinned configuration and weights on first use,
verifies both hashes, then loads weights strictly without remote Python.
CPU is the default; `--vocoder-device mps` or `cuda` selects that device explicitly.
Retain the chosen decoder extra for verification. See the [CLI limits](cli.md).

The explicit `video audio` route uses these same extras. It copies video packets
and does not require OpenCV, PyAV, or the `video` extra.

Add `audio-wavmark` for `--scheme wavmark`, the pinned public WavMark model.
`--scheme all` requests all named packaged decoders, so install `audio-detect`,
`audio-wavmark`, `audio-perth`, and `audio-silentcipher` together for complete checks. The public WavMark package and
checkpoint are hash-checked. This does not establish compatibility with a
production provider's WavMark deployment. The decoder includes the final valid
search position and exactly one-second inputs; shorter inputs remain unverified.
`audio-wavmark` also installs the public model's Torch, torchaudio, and librosa
dependencies. It is separate from metadata-only and CPU spectral processing.

Public Perth v1 uses `audio-perth` and `--scheme perth`. The loader bypasses the public package's legacy
initializer by loading validated model modules in a private namespace. It
requires no deprecated setuptools compatibility dependency. Source code, model
config, and weights are hash-checked
before loading. Missing or incompatible installations remain unverified.

Public SilentCipher-44k uses `audio-silentcipher` and `--scheme silentcipher`.
The same extra supports `--scheme silentcipher16`, an experimental terminated
eight-symbol pattern with seven ternary payload symbols. It does not qualify
the SDK byte API or general 32-bit profile. `all` includes both receivers.
It installs modern Torch, librosa, and Hugging Face Hub, without the legacy SDK
and its Torch upper bound. Pinned config and weights are checked before loading.
The experimental receiver requires at least three seconds and checks every
phase before a negative; CPU checks can be slow. See [receiver scope](audio-video-removal.md#experimental-silentcipher-44k-receiver).

## Video processing

Video metadata inspection works with the default package, and MP4 and MOV
stripping uses the in-tree ISOBMFF box walker. Stripping the non-ISOBMFF
containers (MKV, WebM, AVI, FLV, and the audio formats) and writing any cleaned
video need FFmpeg. The `video` extra installs both FFmpeg and ffprobe, so no
separate system installation is needed. Stable
visible-mark identification and removal, full video cleaning, and visible/all
batch modes need the `video` extra:

```bash
uv tool install --force "remove-ai-watermarks[video]"
```

The extra includes the visible pixel runtime and PyAV for preserving variable
frame timestamps, plus the offline `portable-ffmpeg-full` binary bundle. A complete
system FFmpeg/ffprobe pair takes precedence; otherwise the library extracts the
installed bundle on first use, without a network download or changing `PATH`.
The verified binaries are extracted into `~/.cache/remove-ai-watermarks/ffmpeg`,
so the installed package directory can be read-only. Separate CLI processes
serialize extraction through a file lock. Metadata-only installations can still
use system FFmpeg or add the `video` extra when needed.

The bundle supports Windows x64 (Windows 10 22H2+), macOS Intel/Apple Silicon
(macOS 12+), and Linux x64/ARM64 (glibc 2.28+, Linux 4.18+). Other platforms need
a system build and a source installation without the `video` extra, with the
pixel and PyAV dependencies installed separately. FFmpeg binaries have their
own GPL and third-party notices; see the
[distribution's licenses and corresponding source](https://github.com/mrkbac/portable-ffmpeg#licenses-and-corresponding-source).

Video SynthID regeneration and the experimental Seedance profile also need the diffusion stack:

```bash
uv tool install --force "remove-ai-watermarks[video,diffusion]"
```

## Invisible watermark removal

Install the `qwen-zimage` extra:

```bash
uv tool install --force "remove-ai-watermarks[qwen-zimage]"
```

The `qwen-zimage`, `qwen21-zimage`, and `sdxl-zimage` profiles run a Z-Image face
stage on the DiffSynth runtime, so they need this extra (`qwen21-zimage` use
`Qwen/Qwen-Image-2.1` under the non-commercial Qwen Research License, for research
and personal use only). `chroma-zimage` uses diffusers'
ChromaImg2ImgPipeline instead of DiffSynth for its global stage but inherits
the same face stage, so it needs this extra too. The extra also installs
SentencePiece for Chroma1-HD's binary `spiece.model`; without that backend,
Transformers cannot construct the Chroma tokenizer. It includes the `diffusion`
dependencies; `diffusion` on its own covers the torch and diffusers imports but
not the face stage or Chroma tokenizer, so it is not enough to run a removal.
The extra also installs LPIPS for the default Qwen-2.1 fidelity stages. Its AlexNet
weights load lazily on CPU and use the Torch cache (`TORCH_HOME`).

**An NVIDIA GPU is required for the diffusion profiles.** `qwen-zimage`,
`qwen21-zimage`, `sdxl-zimage` and `chroma-zimage` are CUDA-only, and
construction refuses any other device rather than falling back to a slow or broken
one. There is no CPU, MPS or XPU path for those profiles. The specialized API for
the validated local Microsoft Paint format uses the `pixels` extra and no GPU.
Visible-mark removal, metadata stripping and every `identify` command still run
anywhere.

Video SynthID regeneration is a separate VAE path and does still run on CPU or MPS;
it needs the `diffusion` extra, not this one.

The experimental verified-text post-pass additionally needs LaMa:

```bash
uv tool install --force "remove-ai-watermarks[text-restoration]"
```

That extra includes `qwen-zimage` and `lama`; it does not add OCR. Text strings and
line boxes must be reviewed before the run.

## Feature extras

Extras are composable. Install only the capabilities and file formats the
application actually uses:

| Extra | Capability | Automatically includes | Torch or model download |
| --- | --- | --- | --- |
| `pixels` | Shared BGR image runtime and direct local Paint InvisMark disruption | NumPy, headless OpenCV | No |
| `heif` | HEIC, HEIF, and AVIF pixel decoding | pillow-heif | No |
| `visible` | Visible mark detection, OpenCV inpainting, and manual erasing | `pixels` | No |
| `audio` | Native-geometry WAV/FLAC processing and spectral filtering | NumPy, SciPy; ffmpeg required | No |
| `audio-detect` | Public AudioSeal checks | `audio`, AudioSeal, Hugging Face Hub | Yes |
| `audio-wavmark` | Public WavMark checks | `audio`, WavMark | Yes |
| `audio-perth` | Public Perth v1 checks | `audio`, validated Perth model modules | Yes |
| `audio-silentcipher` | Experimental SilentCipher-44k and terminated 16k receivers | `audio`, Torch, librosa, Hugging Face Hub | Yes |
| `audio-codec` | Pinned EnCodec and experimental Mimi / FocalCodec / DAC resynthesis | `audio`, Torch, Transformers, safetensors | Yes |
| `audio-vocoder` | Experimental pinned full-track BigVGAN | `audio`, Torch, librosa, Hugging Face Hub | Yes |
| `video-detect` | Public VideoSeal/PixelSeal/ChunkySeal known-message and blind presence checks | `video`, Torch | Yes |
| `video` | Visible video identification/removal and timestamp preservation | `visible`, PyAV, FFmpeg/ffprobe bundle | No |
| `detect` | Open DWT-DCT detection for Stable Diffusion, SDXL, and FLUX | `pixels`, PyWavelets | No |
| `trustmark` | Adobe TrustMark detection | trustmark | Yes |
| `source-classify` | Abstaining 10-provider source-export, model-version, and synthetic-image classifier, not SynthID detection | `pixels`, Hugging Face Hub, ONNX Runtime | Model download, no Torch |
| `diffusion` | Torch and Diffusers runtime; video regeneration | `pixels`, Torch, Diffusers | Image profiles only |
| `migan` | MI-GAN ONNX fill backend | `visible`, ONNX Runtime | Model download, no Torch |
| `lama` | big-LaMa ONNX fill backend | `visible`, ONNX Runtime | Model download, no Torch |
| `qwen-zimage` | Invisible image-watermark removal, all CUDA-only profiles | `diffusion`, DiffSynth, SentencePiece | Yes |
| `text-restoration` | Opt-in verified profile-VAE glyph restoration | `qwen-zimage`, `lama` | Yes |
| `text-draft` | Draft OCR proposals for operator verification | PaddleOCR, PaddlePaddle | Model download, no Torch |
| `text` | Local text regeneration with the default Transformers backend | Torch, Transformers, packaged py3langid language model | Yes |
| `text-api` | Explicit Python API text regeneration and the local Ollama CLI backend | Packaged py3langid language model, standard-library HTTP client | Model served separately |
| `text-mlx` | Optional local text regeneration on Apple silicon | MLX LM, Transformers tokenizer, packaged py3langid language model | Yes, on Apple silicon |
| `all` | Production removal and default classifier backends | All runtime branches except proposal-only OCR | Yes |
| `dev` | Tests, linting, typing, upstream parity checks, and development oracles | `video`, `detect`, OpenAI SDK, upstream invisible-watermark | Yes, for parity tests |

The `text` extra does not install Diffusers or image runtimes. The historical
`diffusion` extra continues to include the Transformers text runtime and language
checker for compatibility. The language model ships inside `py3langid`; loading
it needs no service or model download. Generation checkpoints are downloaded
separately unless already cached.

`text-api` installs the shared language checker without Torch, Transformers or an
API-provider SDK. Configure `build_text_api_rewriter` explicitly with your
endpoint, model and optional key, then pass it as `rewriter=`. The library reads
no API environment settings and never switches a local request to an API.
See [the API example](python-api.md#optional-api-text-rewriter).
The same extra supports an explicit Ollama backend for the CLI. Start Ollama
with a locally installed model before using [the text CLI](cli.md#identify-and-clean-text).

Dependency composition:

```mermaid
flowchart LR
    visible --> pixels
    video --> visible
    detect --> pixels
    diffusion --> pixels
    migan --> visible
    lama --> visible
    qwen["qwen-zimage"] --> diffusion
    text["text-restoration"] --> qwen
    text --> lama
    draft["text-draft"]
    heif
    trustmark
    source["source-classify"] --> pixels
```

`heif`, `trustmark`, and `text-draft` are independent branches. Combine them
explicitly with another feature when required. `text-draft` is excluded from
`all` because it proposes unverified OCR annotations and is not a production
removal path.
The TrustMark branch requires `trustmark>=0.9.2`, the first release without a
NumPy 1.x pin, so it installs on every supported Python; its detector tests pass
on Python 3.13 with NumPy 2 (checked 2026-09-23). The `all` bundle selects the production removal and default classifier branches;
it never includes `dev`. Each selected dependency must support the active platform.

Examples:

```bash
# Metadata plus torch-free DWT-DCT detection
uv tool install --force "remove-ai-watermarks[detect]"

# Visible removal with HEIC/AVIF support and MI-GAN
uv tool install --force "remove-ai-watermarks[migan,heif]"

# Visible video removal with preserved timestamps
uv tool install --force "remove-ai-watermarks[video]"

# DWT-DCT and TrustMark detection without diffusion removal
uv tool install --force "remove-ai-watermarks[detect,trustmark]"

# Every production capability compatible with this Python
uv tool install --force "remove-ai-watermarks[all]"

# An arbitrary minimal combination
uv tool install --force "remove-ai-watermarks[migan,detect]"
```

`heif` stays independent so applications that only process PNG, JPEG, or WebP
do not install libheif. `detect` uses the in-tree torch-free decoder and does
not install the upstream `invisible-watermark` package. Optional models download
their weights on first use.

The old `gpu` and `remove` aliases are intentionally not provided. Use
`diffusion` and `visible` respectively.

## Installation coverage

The CLI and Python API come from the same package for pip, uv, pipx, Homebrew,
and source installs. The installer does not select a different implementation.
An extra adds a backend and its dependencies; a base-only install exposes metadata
operations and reports missing dependencies for pixel operations.

| Route | Selecting optional features | Interface coverage |
| --- | --- | --- |
| pip in a virtual environment | `python -m pip install "remove-ai-watermarks[all,text-draft]"` | CLI and public Python API |
| uv tool | `uv tool install --force --python 3.12 "remove-ai-watermarks[all,text-draft]"` | CLI; use a project environment to import the Python API |
| pipx | `pipx install --force --python python3.12 "remove-ai-watermarks[all,text-draft]"` | CLI; its environment is isolated from application Python |
| Homebrew | `brew install wiltodelta/tap/remove-ai-watermarks --with-all --with-text-draft` | CLI; its environment is isolated from application Python |
| Source checkout | `uv sync --extra all --extra text-draft` | CLI and public Python API in the project environment |
| Agent Skill and plugins | Choose the CLI extras through the installation reference | Guides the installed CLI; does not bundle backends |
| MCP | Start an existing CLI installation, or select extras with `uvx --from` | Every CLI operation and the public API-only backends via the [MCP catalog](mcp-server.md#complete-operation-tools) |
| ComfyUI nodes | The node package installs the complete public backend set | Existing image tensor nodes plus typed nodes for video, text, classifiers, and every catalog operation |
| Hugging Face Space | The Space installs the complete public backend set | Image, video, text, and classifier controls; image regeneration needs a CUDA Space |

These full-selection examples include the proposal-only OCR backend (`text-draft`)
that `all` intentionally omits. `dev` adds development tools and
oracles, not additional public runtime features.

Python 3.12 is the recommended starting point when every optional backend is
needed. The package's Python support range does not promise that every third-party
backend publishes wheels for every Python version or CPU architecture. A resolver
success verifies dependency compatibility, not model inference. Image regeneration
still requires NVIDIA CUDA; adding extras cannot supply a missing GPU.

The MCP server, ComfyUI operation nodes, and Space panels use the installed
library's shared operation catalog, so all CLI parameters reach each interface.
The catalog also exposes the public API-only ONNX classifier and OCR proposals.
ComfyUI accepts local file paths; the public Space stages uploads into isolated
request directories and supplies downloads rather than accepting server paths.
Hardware requirements still apply. Conda environments can use the pip route;
this project does not publish a conda package or a standalone Docker image.

## Install from the repository

```bash
git clone https://github.com/wiltodelta/remove-ai-watermarks.git
cd remove-ai-watermarks
uv sync --frozen
```

Add the feature groups required for your work:

```bash
uv sync --frozen --extra dev
uv sync --frozen --extra dev --extra diffusion
```

Run commands from the repository root:

```bash
uv run remove-ai-watermarks --help
```

## Development setup

Install development dependencies:

```bash
uv sync --frozen --extra dev
```

Run the complete project gate:

```bash
bash maintain.sh
```

The script syncs every optional backend on top of the `dev` environment above,
then runs dependency checks, linting, formatting, type checking, and the test
suite. It applies Ruff fixes and formatting in place rather than only reporting
them.

## Hugging Face authentication

Pass a Hugging Face token directly when the selected model or account requires
one:

```bash
remove-ai-watermarks invisible image.png --hf-token "$HF_TOKEN"
```

The CLI also loads `HF_TOKEN` from the environment and from a local `.env`
file. The same name is documented in `.env.example`.

## Troubleshooting

### The first model run is slow

Diffusion and learned fill backends may download model weights on first use.
Later runs reuse their caches.

### The command skips invisible removal

The normal behavior is to skip diffusion when no supported local signal is
found. A missing signal does not prove that the image is clean. If you know the
image came from a relevant generator, use `--force`.

If the CLI reports that the removal dependencies are unavailable, install the
`qwen-zimage` extra. `diffusion` alone covers Torch and Diffusers but not the
DiffSynth face stage that all profiles run. Video SynthID removal is a separate
path and needs `video` and `diffusion`.

## Public-model video message checks

`video-detect` adds Torch to the `video` stack for pinned public VideoSeal,
PixelSeal, and ChunkySeal decoders. Checkpoint downloads are checked against
packaged SHA-256 values before loading; the PixelSeal checkpoint uses
`weights_only=True`, and ChunkySeal loads a digest-pinned detector slice without
unpickling. The decoder does not install the upstream training SDK.

```bash
uv tool install --force "remove-ai-watermarks[video-detect]"
uv tool install --force "remove-ai-watermarks[video-detect,diffusion]"
```

The first supports inspection and codec-only processing. The second adds the
pinned float32 safetensors VAE. Processing needs ffmpeg and ffprobe. The caller
must supply an independently known encoder message for `identify` and `remove`.
`video message detect` is a blind presence check limited to the pinned public
models; these extras do not add a general AI-origin detector.
