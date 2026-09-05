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

Homebrew is macOS/Linux only and CANNOT carry extras: the formula installs the
default package, which has no pixel dependencies, so `visible`, `erase`, `all`,
`batch` and the `video` commands stop with an install hint, and `identify`
reports metadata only. That is exactly what `pixel_stack: missing` means in the
probe. Use it for metadata-only work,
or when the user asks for brew by name and accepts that limit:

```bash
brew install wiltodelta/tap/remove-ai-watermarks
```

Reinstalling over a brew build with uv or pipx is the fix; the extra is what
was missing, not the tool.

After installing, rerun `scripts/probe.py` and confirm `pixel_stack` is `ok`
before promising a command will work.

Read this file when the probe says the CLI is missing, the extra is wrong,
ffmpeg is missing, or the user asked about Homebrew, pipx, or pip.

## Extra list

| Need | Extra |
| --- | --- |
| Metadata inspect and strip | no extra |
| Image identify, `visible`, `erase` | `visible` |
| HEIC / HEIF / AVIF pixels | add `heif` |
| Video identify / visible / all / batch | `video`, plus ffmpeg |
| Video SynthID (`video invisible`) | `video,diffusion`, plus ffmpeg |
| Invisible images | `qwen-zimage`, plus NVIDIA CUDA |
| Learned image/video fill | add `migan` or `lama` |
| Open DWT-DCT detect | add `detect` |
| Pixel classifier (`classify`) | add `classify` |
| OpenAI verifier (`verify-openai-synthid`) | add `verify` |
| Adobe TrustMark on Python 3.11-3.12 | add `trustmark` |
| Every production extra on this Python | `all` |

`all` the extra and `all` the command are different. The extra installs
backends. The command runs visible, then invisible when available, then
metadata.

ffmpeg is required to write cleaned video and to strip non-ISOBMFF containers
(MKV, WebM, AVI, FLV). Image work does not need it.

Invisible image removal refuses CPU and MPS at construction. There is no
fallback profile.
