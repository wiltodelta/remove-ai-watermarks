# Remove-AI-Watermarks

Unified tool for removing **visible** and **invisible** AI watermarks from images.

## Features

- **Visible watermark removal** — Gemini sparkle logo via reverse alpha blending (fast, offline, deterministic)
- **Invisible watermark removal** — SynthID, StableSignature, TreeRing via diffusion-based regeneration
- **AI metadata stripping** — EXIF, PNG text chunks, C2PA provenance manifests
- **Analog Humanizer** — film grain and chromatic aberration injection to bypass AI image classifiers
- **Smart Face Protection** — automatic extraction and blending of human faces to prevent AI distortion
- **High-Res Upscaler** — prevents resolution degradation during invisible watermark removal
- **Batch processing** — process entire directories
- **Detection** — three-stage NCC watermark detection with confidence scoring

## Examples

| Before (Watermarked) | After (Cleaned) |
| --- | --- |
| ![Before](demo_banana_before.png) | ![After](demo_banana_after.png) |

## Installation

### Recommended (macOS)

Install as an isolated CLI tool — no need to manage virtual environments:

```bash
# Using pipx (brew install pipx)
pipx install git+https://github.com/wiltodelta/remove-ai-watermarks.git

# Or using uv (brew install uv)
uv tool install git+https://github.com/wiltodelta/remove-ai-watermarks.git
```

To update to the latest version:

```bash
pipx install --force git+https://github.com/wiltodelta/remove-ai-watermarks.git

# or
uv tool install --force git+https://github.com/wiltodelta/remove-ai-watermarks.git
```

### Install from repository (macOS)

**Prerequisites:** Python 3.10+ and `pip` (or [`uv`](https://docs.astral.sh/uv/)).

```bash
# 1. Clone the repository
git clone https://github.com/wiltodelta/remove-ai-watermarks.git
cd remove-ai-watermarks

# 2. Install the package in editable mode
pip install -e .

# Or, if you use uv:
uv pip install -e .
```

After installation the `remove-ai-watermarks` command is available system-wide.

#### Invisible watermark removal (optional)

Invisible removal uses diffusion models and requires a **HuggingFace token** and a decent GPU (CUDA) or Apple Silicon (MPS).

```bash
# 1. Create a free token at https://huggingface.co/settings/tokens
# 2. Copy the example env file and paste your token
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_your_token_here

# 3. On first run, the model (~2 GB) will be downloaded automatically.
#    On macOS with Apple Silicon, MPS acceleration is used by default.
#    On macOS without GPU, add --device cpu (inference will be slow).
```

#### Developer setup

```bash
# Install with dev dependencies (pytest, ruff, pyright)
pip install -e ".[dev]"
# Or with uv:
uv pip install -e ".[dev]"

# Run tests
pytest

# Run linters
./maintain.sh
```

## Usage

### CLI

```bash
# Remove visible Gemini watermark
remove-ai-watermarks visible image.png -o clean.png

# Remove invisible watermarks (SynthID etc.) with optimal quality retention
remove-ai-watermarks invisible image.png -o clean.png --humanize 4.0

# Strip AI metadata
remove-ai-watermarks metadata image.png --check
remove-ai-watermarks metadata image.png --remove

# Batch processing
remove-ai-watermarks batch ./images/ --mode visible

# Full pipeline: visible + invisible + metadata
remove-ai-watermarks all image.png -o clean.png
```

### Python API

```python
from remove_ai_watermarks.gemini_engine import GeminiEngine
import cv2

engine = GeminiEngine()
image = cv2.imread("watermarked.png")

# Detect
result = engine.detect_watermark(image)
print(f"Detected: {result.detected} (confidence: {result.confidence:.1%})")

# Remove
clean = engine.remove_watermark(image)
cv2.imwrite("clean.png", clean)
```

### Metadata stripping

```python
from remove_ai_watermarks.metadata import has_ai_metadata, remove_ai_metadata
from pathlib import Path

if has_ai_metadata(Path("image.png")):
    remove_ai_metadata(Path("image.png"), Path("clean.png"))
```

## Requirements

- Python ≥ 3.10
- **Visible removal / metadata**: CPU only, no GPU required
- **Invisible removal**: GPU recommended (CUDA or MPS), works on CPU (slow)

## Troubleshooting

**SSL certificate error on macOS** (`CERTIFICATE_VERIFY_FAILED`):

```bash
# Option 1: Run the Python certificate installer
/Applications/Python\ 3.*/Install\ Certificates.command

# Option 2: Install certifi (the tool auto-detects it)
pip install certifi
```

**First run is slow** — this is expected. The tool downloads model weights (~2 GB) on first launch. Subsequent runs use cached models.

## Credits

- [noai-watermark](https://github.com/mertizci/noai-watermark) by mertizci — invisible watermark removal engine
- [GeminiWatermarkTool](https://github.com/allenk/GeminiWatermarkTool) by Allen Kuo (MIT) — visible watermark removal algorithm
- [CtrlRegen](https://github.com/yepengliu/CtrlRegen) by Liu et al. (ICLR 2025) — controllable regeneration pipeline
- [NeuralBleach](https://github.com/...) (MIT) — analog humanizer technique

## ⚠️ Disclaimer

This tool is provided for **educational and research purposes only**.

Removing AI watermarks to misrepresent AI-generated content as human-created
may violate applicable laws, including the U.S. Digital Millennium Copyright Act
(DMCA) and the COPIED Act. Users are solely responsible for ensuring their use
complies with all applicable laws and platform terms of service.

The authors do not condone the use of this tool for deception, fraud,
or any activity that violates applicable laws or regulations.

## License

MIT
