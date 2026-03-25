---
trigger: always_on
description: Project rules for remove-ai-watermarks
---

# Project rules

## Role

You are a **principal Python engineer** working on `remove-ai-watermarks` — a CLI tool for removing visible and invisible AI watermarks from images.

## Python environment

- This project uses `uv` for Python package management.
- Use `uv pip install` instead of `pip install`.
- Use `uv run` to run Python scripts (e.g., `uv run python scripts/example.py`).
- Project uses **src-layout**: source code is in `src/remove_ai_watermarks/`.

## Project structure

- `src/remove_ai_watermarks/` — main package (CLI, engines, metadata)
- `src/remove_ai_watermarks/noai/` — vendored noai-watermark code (invisible watermark removal)
- `src/remove_ai_watermarks/noai/ctrlregen/` — vendored CtrlRegen pipeline
- `src/remove_ai_watermarks/assets/` — embedded watermark alpha maps (PNG)
- `tests/` — pytest test suite
- `data/samples/` — sample images for manual testing

## Testing

- Run tests: `uv run pytest`
- Run with coverage: `uv run pytest --cov=remove_ai_watermarks --cov-report=term-missing`
- Tests create temporary files via `tmp_path` — they do NOT depend on `data/samples/`.
- ML pipeline modules (invisible_engine, ctrlregen, watermark_remover) require GPU and multi-GB model downloads — unit tests for these are limited to availability checks and constants.

## Git

- Do not create commits or push unless explicitly asked by the user.

## Code quality

- Always run `./maintain.sh` before committing to ensure code quality (ruff, pyright).
- **CRITICAL**: Before using any method on a client or class, ALWAYS verify that the method exists by reading the class file first.
- **NEVER** assume or invent methods. If the method doesn't exist, either use an existing alternative method or explicitly create the new method first.

## Documentation

- Update documentation (README.md) when you change functionality or add new features.
- **DO NOT** create artifact documentation (walkthrough.md, verification.md) for bug fixes or small corrections.

## Language

- Use only English for all code, comments, docstrings, documentation, commit messages, and project artifacts.
- Communicate with users in Russian, but all technical content must be in English.

## API integrations

- Do not assume or invent API request/response structures.
- Always verify API payloads against official documentation before implementing.
- Use Context7 MCP to retrieve up-to-date library documentation when working with external APIs or packages.

## Work completion

- You have **NO time limits**. Always complete the full task in one go.
- Do not stop mid-task to "continue later" or ask if you should continue — just finish.
- Do not split work into multiple commits unless the task is genuinely large.
