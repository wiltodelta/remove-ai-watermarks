# Install extras

# Contents
- Installers
- Extra list
- Notes

## Installers

Use the first tool `scripts/probe.py` reported as present. Quote every extra
list. Forward slashes only.

uv:

```bash
uv tool install --force "remove-ai-watermarks[visible]"
```

pipx:

```bash
pipx install --force "remove-ai-watermarks[visible]"
```

pip (into the active environment):

```bash
python3 -m pip install --upgrade "remove-ai-watermarks[visible]"
```

On Windows, `python` may exist when `python3` does not.

Homebrew is available on macOS and Linux. Its default installation includes
`visible`. Every published extra has a matching `--with-<extra>` option:

```bash
brew install wiltodelta/tap/remove-ai-watermarks
brew install wiltodelta/tap/remove-ai-watermarks --with-video --with-diffusion
brew install wiltodelta/tap/remove-ai-watermarks --with-all
```

Use `--without-default-extras` for the base metadata-only package, or combine it
with explicit `--with-<extra>` flags for an exact minimal selection. List current
options with `brew options wiltodelta/tap/remove-ai-watermarks`. Add features using
`brew reinstall` with the flags; Homebrew preserves previous flags on reinstall
and upgrade. Removing old extras requires uninstalling and installing again with
the complete desired selection. Backend hardware requirements remain unchanged.
For every public optional backend, select `all,text-draft,classify-onnx` with
pip/uv/pipx, or combine `--with-all --with-text-draft --with-classify-onnx` in
Homebrew. Prefer Python 3.12 for this complete selection; optional backends can
have narrower Python and platform support than the base package.

When the user asks for Homebrew, use that route. A missing pixel stack can be an
older or explicitly metadata-only build; it does not mean Homebrew lacks extras.
Without pixel extras, `batch --mode metadata`, `video metadata`,
`video identify --no-visible`, and `video batch --mode metadata` still work.

After installing, rerun `scripts/probe.py` and confirm `pixel_stack` is `ok`
before promising a command will work.

Tool installs (uv tool, pipx, Homebrew) isolate their Python environments. For
library-only options, create an application virtual environment and install the
needed extras there, or run a script without changing the current project:

```bash
uv run --no-project --python 3.12 --with "remove-ai-watermarks[all,text-draft,classify-onnx]" python script.py
```

Read the [Python API guide](https://github.com/wiltodelta/remove-ai-watermarks/blob/main/docs/python-api.md)
for signatures and constraints. OCR proposals need operator verification before
they become verified-text manifests. Installing the Agent Skill through skills.sh,
ClawHub, or a plugin copies the same instructions; it does not install the Python
package or model weights. Probe and install the requested backend in every harness.

Read this file when the probe says the CLI is missing, the extra is wrong,
ffmpeg is missing, or the user asked about Homebrew, pipx, or pip.

## Extra list

| Need | Extra |
| --- | --- |
| Image/video/text metadata inspect and strip | no extra |
| Text token-watermark regeneration | `diffusion` |
| Image identify, `visible`, `erase` | `visible` |
| HEIC / HEIF / AVIF pixels | add `heif` |
| Video identify / visible / all / batch | `video` (includes FFmpeg and ffprobe) |
| Experimental Seedance video (`--profile seedance-experimental`) | `video,diffusion` (includes FFmpeg and ffprobe) |
| Video SynthID (`video invisible`) | `video,diffusion` (includes FFmpeg and ffprobe) |
| Invisible images | `qwen-zimage`, plus NVIDIA CUDA |
| Learned image/video fill | add `migan` or `lama` |
| Verified text restoration (`--text-manifest`) | add `text-restoration` |
| Open DWT-DCT detect | add `detect` |
| Pixel classifier (`classify`) | add `classify` |
| CPU ONNX photo classifier, library `backend="onnx"` only (since CLI 0.40.0) | add `classify-onnx` |
| Adobe TrustMark | add `trustmark` |
| Every production extra on this Python | `all` |

`all` the extra and `all` the command are different. The extra installs
backends. The command runs visible, then invisible when available, then
metadata.

ffmpeg is required to write cleaned video and to strip non-ISOBMFF containers
(MKV, WebM, AVI, FLV). The `video` extra installs an offline binary bundle for
Windows x64, macOS 12+ Intel/Apple Silicon, and Linux glibc 2.28+ x64/ARM64.
No separate FFmpeg installation is needed there. The library prefers a complete
system FFmpeg/ffprobe pair, then uses the bundle without changing `PATH` or
fetching executables at runtime. First use extracts into the user cache, so the package directory can be read-only.
The probe checks bundle availability inside the CLI interpreter, without extracting
it. Image work does not need FFmpeg.

Invisible image removal refuses CPU and MPS at construction. There is no
fallback profile.
