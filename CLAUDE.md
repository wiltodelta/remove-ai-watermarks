# Remove-AI-Watermarks

You are a **principal Python engineer** maintaining a CLI tool and library for removing visible and invisible AI watermarks from images.

## How to run

- `uv run remove-ai-watermarks all <image.png> -o <output.png>`
- `uv run remove-ai-watermarks metadata <image.png> --check` — inspect AI metadata (C2PA, EXIF, PNG chunks)
- `uv run remove-ai-watermarks metadata <image.png> --remove -o <out.png>` — strip all AI metadata

## Configuration

- GPU/ML modules (invisible_engine, ctrlregen, watermark_remover) are optional — guard imports with `is_available()` checks
- Tests for ML modules are limited to availability checks (require multi-GB downloads)

## Key modules

- `noai/c2pa.py` — PNG chunk parser; use `extract_c2pa_chunk(path)` to get raw caBX payload, `has_c2pa_metadata(path)` to detect. Do not reimplement chunk parsing.
- `noai/constants.py` — PNG_SIGNATURE, C2PA_CHUNK_TYPE, C2PA_SIGNATURES constants
- `face_protector.py` — YOLO detect + soft-blend pattern; mirror this for any "protect region during diffusion" features

## Known limitations

- `invisible` pipeline downscales to 768 px before diffusion → degrades fine text in infographics. Tracked; fix is tile-based or skip-downscale approach.
- Pyright first run is slow (2-3 min) due to ML deps (torch/diffusers/transformers stubs)
