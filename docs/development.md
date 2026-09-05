# Development

Standalone evaluation, calibration, and release utilities are catalogued in
[`scripts/README.md`](../scripts/README.md). They are maintainer tools rather than
installed commands, and their local data inputs remain outside the repository.

Read this reference for environment setup, dependency recovery, CI behavior, and fixture policy. The always-loaded invariants remain in [`.claude/rules/development.md`](../.claude/rules/development.md).

## Local environment

- Use `uv sync --frozen --extra dev` and add only the feature extras needed for the task.
- Do not use `uv pip install` for development tools. It can re-resolve `uv.lock` outside the compatible ML dependency set.
- A default-only sync removes every pixel and model package by design. Package imports remain light through lazy exports.
- On an unreliable connection, sync `dev` plus only the required feature extras, such as `diffusion`, and run the checks directly instead of downloading every optional learned backend.
- Run `uv` from the repository root or it may create a bare environment without the project dependencies.

`maintain.sh` starts with `uv sync --all-extras`, so a lean environment is replaced whenever the gate runs. Syncing back down can leave `opencv-python-headless` half-removed: `import cv2` then succeeds against an empty namespace while `cv2.INTER_LINEAR` is gone, which surfaces as a suite that stalls or reports hundreds of unrelated failures. Repair it with `uv sync --frozen --extra dev --reinstall-package opencv-python-headless` rather than re-reading the diff.

Pyright is only meaningful with the extras its imports need. Scoped to `src/`, a lean environment reports `TrustMark`, `transformers` and `huggingface_hub` as unresolved, which reads as a type error and is not one; add `--extra trustmark --extra classify` before believing it.

The optional TrustMark decoder downloads weights into its installed package directory. After pruning that extra, a leftover weights directory can make availability checks see an empty namespace package. If Pyright reports an unknown `TrustMark` import and `find_spec("trustmark")` returns a loader-less spec, remove that regenerable remnant from the active virtual environment and resync.

### Known security-gate blocks

`maintain.sh` fails while PyPI ships no fixed release for a transitive CVE. Current case (triaged 2026-08-17): `lightning` 2.6.5, pulled only by the optional `trustmark` extra, carries PYSEC-2026-3624 (RCE via `load_from_checkpoint` on an attacker-crafted checkpoint). The vulnerable API is unreachable here, because the TrustMark decoder loads only its own pinned weights downloaded from the TrustMark release, never a user-supplied checkpoint. The upstream fix is merged but unreleased, and ignores are never added, so run and report the remaining core checks (Ruff, Pyright, tests) separately until a fixed `lightning` release lands; bump it with `uv lock --upgrade-package lightning` as soon as one does.

## CI

`.github/workflows/test.yml` runs Ruff, a test matrix over every supported Python
minor with default plus development dependencies, cross-platform coverage at the
oldest and newest minors, and a separate job that installs ffmpeg on Ubuntu to
run the full-clip video test. Diffusion and model-running tests skip in that
matrix; metadata, identification, visible removal, the DWT-DCT decoder, and the
OpenCV eraser remain covered across operating systems.

Keep `uv.lock` compatible with `uv sync --frozen`. Dependency pull-request checks use GitHub's merge result against current `main`; if `main` moves, merge it locally and rerun the full gate because a newer linter can expose stale directives in later code.

Release and distribution behavior is canonical in [`release-and-distribution.md`](release-and-distribution.md).

## Fixture and data policy

[`../data/README.md`](../data/README.md) is the source of truth:

- executable provenance fixtures live under `data/fixtures/`;
- minimal controlled detector inputs live under `data/calibration/`;
- canonical provider-oracle originals and their manifests live under `data/synthid/`;
- evaluation-only ground truth lives under `data/evaluations/`;
- runtime detector assets live in the package; unregistered research candidates remain outside the shipped wheel.

Store each binary once. Point tests and manifests at its canonical path. Keep generated and cleaned outputs outside the repository and retain only reproducible public records allowed by the data policy.

Use synthetic byte blobs for unsupported format paths and deterministic generated negatives where a real negative fixture is unnecessary. Detection and removal tests must preserve their format-specific invariants.
