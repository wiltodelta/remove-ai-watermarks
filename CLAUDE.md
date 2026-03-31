# Remove-AI-Watermarks

CLI tool and Python library for removing visible and invisible AI watermarks from images.

## Build & Run

```bash
uv pip install -e .            # base install (visible removal + metadata)
uv pip install -e ".[gpu]"     # + invisible removal (torch, diffusers, ultralytics)
uv pip install -e ".[dev]"     # + dev tools (pytest, ruff, pyright)
```

## Test & Lint

```bash
uv run pytest                  # run tests
uv run ruff check --fix        # lint
uv run ruff format             # format
uv run pyright                 # type check
./maintain.sh                  # all of the above + dependency audit
```

## Architecture

- `src/remove_ai_watermarks/` — main package (src-layout, hatchling build)
- `cli.py` — Click CLI with commands: `visible`, `invisible`, `metadata`, `all`, `batch`
- `gemini_engine.py` — visible watermark: reverse alpha blending + NCC detection
- `invisible_engine.py` — invisible watermark: diffusion regeneration wrapper
- `metadata.py` — EXIF/PNG/C2PA metadata detection and stripping
- `humanizer.py` — film grain + chromatic aberration (analog humanizer)
- `face_protector.py` — YOLO face detection + soft-blend restoration
- `noai/` — vendored invisible removal core (watermark_remover, cleaner, ctrlregen/)
- `assets/` — embedded alpha maps (gemini_bg_48.png, gemini_bg_96.png)
- `tests/` — pytest suite (uses tmp_path, no external data deps)

## Key Conventions

- Python 3.10+, ruff line-length 120, type hints everywhere
- GPU/ML modules (invisible_engine, ctrlregen, watermark_remover) are optional — guard imports with `is_available()` checks
- Tests for ML modules are limited to availability checks (require multi-GB downloads)
- Always run `./maintain.sh` before committing
- Use `uv` for all package operations, never raw `pip`

## Language

- Code, comments, docstrings, commits, docs: English only
- Communication with the user: Russian
