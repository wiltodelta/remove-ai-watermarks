#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-}"

if [[ -z "$VERSION" ]]; then
    echo "Usage: ./release.sh <version>"
    echo "Example: ./release.sh 0.4.0"
    exit 1
fi

if ! [[ "$VERSION" =~ ^[1-9][0-9]*\.[0-9]+\.[0-9]+$|^0\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: version must be in X.Y.Z format (e.g. 0.4.0)"
    exit 1
fi

TAG="v$VERSION"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$BRANCH" != "main" ]]; then
    echo "Error: releases must be made from the main branch (currently on $BRANCH)"
    exit 1
fi

if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "Error: tag $TAG already exists"
    exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: working tree is not clean. Commit or stash changes first."
    exit 1
fi

echo "=== Releasing $TAG ==="

echo "→ Running checks..."
uv run ruff check
uv run ruff format --check
uv run pytest
uv run pyright

echo "→ Updating version to $VERSION..."
# Portable sed -i: works on both macOS (BSD) and Linux (GNU)
if sed --version >/dev/null 2>&1; then
    # GNU sed
    sed -i "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
    sed -i "s/^__version__ = \".*\"/__version__ = \"$VERSION\"/" src/remove_ai_watermarks/__init__.py
else
    # BSD sed (macOS)
    sed -i '' "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
    sed -i '' "s/^__version__ = \".*\"/__version__ = \"$VERSION\"/" src/remove_ai_watermarks/__init__.py
fi

echo "→ Committing..."
git add pyproject.toml src/remove_ai_watermarks/__init__.py
git commit -m "chore: bump version to $VERSION"

echo "→ Tagging $TAG..."
git tag "$TAG"

echo "→ Pushing..."
git push origin main
git push origin "$TAG"

echo "=== Released $TAG ==="
