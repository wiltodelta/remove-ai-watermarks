# Remove-AI-Watermarks

You are a **principal Python engineer** maintaining a CLI tool and library for removing visible and invisible AI watermarks from images.

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

- Python 3.10+, ruff line-length 120, pyright strict mode, type hints everywhere
- GPU/ML modules (invisible_engine, ctrlregen, watermark_remover) are optional — guard imports with `is_available()` checks
- Tests for ML modules are limited to availability checks (require multi-GB downloads)
- Use `uv` for all package operations, never raw `pip`
- `_refs/` directory is excluded from all checks — contains third-party reference code

## Release Process

To release a new version (X.Y.Z):
1. Run all checks (`./maintain.sh`)
2. Update version in `pyproject.toml` and `src/remove_ai_watermarks/__init__.py`
3. Commit: `chore: bump version to X.Y.Z`
4. Tag: `git tag vX.Y.Z`
5. Push: `git push origin main && git push origin vX.Y.Z`

GitHub Actions will automatically create a GitHub Release with build artifacts.

## Pre-commit Hook

Git hooks live in `.githooks/`. Run `./maintain.sh` once to activate them (sets `core.hooksPath`). The pre-commit hook runs ruff check, ruff format --check, and pytest.

## Language

- Code, comments, docstrings, commits, docs: English only
- Communication with the user: Russian
