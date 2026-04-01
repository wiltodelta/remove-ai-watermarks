#!/usr/bin/env bash

set -euo pipefail

# Ensure git hooks are active
git config core.hooksPath .githooks

uv sync --all-extras
uv run uv-outdated
uv run uv-secure --ignore-unfixed
uv run ruff check --fix
uv run ruff format
uv run pytest
uv run pyright
