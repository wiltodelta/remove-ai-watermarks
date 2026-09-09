#!/usr/bin/env bash

set -euo pipefail

uv sync --all-extras
# uv-outdated / uv-secure run via uvx (isolated env), NOT `uv run`: resolving them
# inside the project env crashes and, with set -e, aborts the whole gate before
# ruff/pyright/tests (see CLAUDE.md "Test and lint").
uvx uv-outdated
# Findings and scanner failures both stop the gate, even if a success message
# preceded a failure. Scan only this lockfile, not sibling worktree copies.
uvx uv-secure uv.lock
uv run python scripts/sync_c2pa_soft_bindings.py --check
uv run ruff check --fix
uv run ruff format
# Scoped to src/: a full-project pyright run OOM-crashes node on this ML-heavy
# repo (see CLAUDE.md "Test and lint"); src/ is the authoritative strict gate.
uv run pyright src/
uv run pytest -n auto
