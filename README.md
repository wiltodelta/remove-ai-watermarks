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

## How it works

### 1. Visible watermark removal

Gemini embeds a sparkle logo using alpha blending:

```text
watermarked = α × logo + (1 − α) × original
```

We reverse this with a known alpha map (extracted from Gemini on a pure-black background):

```text
original = (watermarked − α × logo) / (1 − α)
```

A three-stage NCC (Normalized Cross-Correlation) detector finds the watermark position and scale dynamically, so it works even if the image was resized or cropped. After removal, residual sparkle-edge artifacts are cleaned via gradient-masked inpainting.

**Speed**: ~0.05s per image. No GPU needed.

### 2. Invisible watermark removal

AI services (Google SynthID, StableSignature, TreeRing) embed imperceptible patterns in the frequency domain. These survive cropping, resizing, and JPEG compression.

The removal pipeline:

```text
image → downscale to 768px → encode to latent space (VAE)
      → add controlled noise (forward diffusion)
      → denoise (reverse diffusion, ~2 steps at strength 0.02)
      → decode back to pixels (VAE) → upscale to original resolution
```

The key insight: even minimal noise injection (strength 0.02 = 2% perturbation) breaks the watermark signal while preserving visual quality. The diffusion model acts as a learned image prior — it reconstructs the image faithfully while destroying the watermark pattern.

**Face Protection**: before diffusion, YOLO detects people in the image and extracts them. After diffusion, the original faces are blended back with a soft elliptical mask to prevent AI distortion of facial features.

**Analog Humanizer**: optional film grain and chromatic aberration injection that makes the output indistinguishable from a photo of a screen, defeating AI-generated image classifiers.

### 3. AI metadata stripping

AI tools embed generation metadata in multiple layers:

- **EXIF tags** — prompt, seed, model hash, sampler settings
- **PNG text chunks** — ComfyUI workflows, Stable Diffusion parameters
- **C2PA manifests** — Google Imagen, OpenAI DALL-E, Adobe Firefly provenance

The cleaner parses each layer, removes AI-related fields, and preserves standard metadata (Author, Copyright, Title).

## Examples

| Before (Watermarked) | After (Cleaned) |
| --- | --- |
| ![Before](demo_banana_before.png) | ![After](demo_banana_after.png) |

## Installation

### Recommended

Install as an isolated CLI tool — no need to manage virtual environments:

```bash
# Using pipx (https://pipx.pypa.io)
pipx install git+https://github.com/wiltodelta/remove-ai-watermarks.git

# Or using uv (https://docs.astral.sh/uv)
uv tool install git+https://github.com/wiltodelta/remove-ai-watermarks.git
```

To update to the latest version:

```bash
pipx upgrade remove-ai-watermarks

# or
uv tool upgrade remove-ai-watermarks
```

### Install from repository

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

#### Invisible watermark removal

Invisible removal uses diffusion models and a GPU for reasonable speed.

```bash
# On first run, the model (~2 GB) will be downloaded automatically.
# Device is auto-detected: CUDA (Linux/Windows) > MPS (macOS) > CPU.
# To force a device: --device cuda / --device mps / --device cpu

# Optional: set a HuggingFace token for gated/private models
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_your_token_here
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
# Remove all watermarks from a single image (visible + invisible + metadata)
remove-ai-watermarks all image.png -o clean.png

# Process an entire directory
remove-ai-watermarks batch ./images/ --mode all
```

#### Individual commands

```bash
# Visible watermark only (Gemini sparkle) — fast, offline
remove-ai-watermarks visible image.png -o clean.png

# Invisible watermark only (SynthID etc.) — requires GPU
remove-ai-watermarks invisible image.png -o clean.png --humanize 4.0

# Check / strip AI metadata only
remove-ai-watermarks metadata image.png --check
remove-ai-watermarks metadata image.png --remove

# Batch with a specific mode
remove-ai-watermarks batch ./images/ --mode visible
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

**SSL certificate error** (`CERTIFICATE_VERIFY_FAILED`):

```bash
# Install certifi (the tool auto-detects it)
pip install certifi

# macOS only: run the Python certificate installer
/Applications/Python\ 3.*/Install\ Certificates.command
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
