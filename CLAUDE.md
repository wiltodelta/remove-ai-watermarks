# Remove-AI-Watermarks

You are a **principal Python engineer** maintaining a CLI tool and library for removing visible and invisible AI watermarks from images.

## How to run

- `uv run remove-ai-watermarks all <image.png> -o <output.png>`

## Configuration

- GPU/ML modules (invisible_engine, ctrlregen, watermark_remover) are optional — guard imports with `is_available()` checks
- Tests for ML modules are limited to availability checks (require multi-GB downloads)
