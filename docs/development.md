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

`maintain.sh` starts with `uv sync --all-extras`, so a lean environment is replaced whenever the gate runs. Syncing back down can leave `opencv-python-headless` half-removed: `import cv2` then succeeds against an empty namespace while `cv2.INTER_LINEAR` is gone, which surfaces as a suite that stalls or reports hundreds of unrelated failures. Repair it with `uv sync --frozen --extra dev --reinstall-package opencv-python-headless` rather than re-reading the diff. `opencv-python-headless`, `opencv-python` (from the dev group's `invisible-watermark`) and `opencv-contrib-python` (from PaddleX under `text-draft`, pinned at 4.10.0.84) all install into one `cv2` directory, so the `text-draft` extra and a `[tool.uv]` constraint hold all three at that version. Mixed versions made a fresh `uv sync --all-extras` fail with `partially initialized module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline'` until 2026-10-23; if it reappears after a PaddleX upgrade, move the pin to PaddleX's new version rather than reinstalling packages.

Pyright is only meaningful with the extras its imports need. Scoped to `src/`, a lean environment reports `TrustMark`, `transformers` and `huggingface_hub` as unresolved, which reads as a type error and is not one; add `--extra trustmark --extra classify` before believing it.

The optional TrustMark decoder downloads weights into its installed package directory. After pruning that extra, a leftover weights directory can make availability checks see an empty namespace package. If Pyright reports an unknown `TrustMark` import and `find_spec("trustmark")` returns a loader-less spec, remove that regenerable remnant from the active virtual environment and resync.

### Known security-gate blocks

`maintain.sh` treats both vulnerability findings and scanner failures as fatal.
A success sentence followed by a nonzero scanner exit is still a failed check;
`tests/test_maintenance.py` exercises that case with isolated command stubs.
Do not add advisory ignores or suppress the scanner's exit status. Run and
report Ruff, Pyright scoped to `src/`, and tests separately when security blocks
the gate.

Rechecked on 2026-10-02 against the locked versions and the advisory sources:

- `lightning` and `pytorch-lightning`, pulled by the optional `trustmark`
  extra, are pinned to 2.6.6. That release clears PYSEC-2026-3624 and
  PYSEC-2026-3967; do not reintroduce an older transitive resolution.
- `accelerate`, required by the diffusion stack, is affected by
  [GHSA-4j2p-28q2-5m79](https://github.com/advisories/GHSA-4j2p-28q2-5m79)
  and PYSEC-2026-3804. The sharded-checkpoint loader accepts unsafe
  `weight_map` paths; neither advisory lists a patched release. The lock
  carries 1.15.0, which is outside the advisory's `<= 1.14.0` range, so the
  scanner no longer blocks the gate. Its
  `load_checkpoint_in_model` still joins `weight_map` entries with no
  containment check, and the sanitizer PRs
  ([#4138](https://github.com/huggingface/accelerate/pull/4138),
  [#4214](https://github.com/huggingface/accelerate/pull/4214)) were closed
  without merging. Do not treat the passing scan as a fix. The project has no
  direct call to `load_checkpoint_in_model` or
  `load_checkpoint_and_dispatch`, but this does not prove indirect
  model-loading paths are unaffected.

When a new block appears, or before carrying one forward, rerun `uvx uv-secure uv.lock` and check
PyPI for a fixed release. Upgrade a fixed package with
`uv lock --upgrade-package <package>` and rerun the gate. Removing the current
dependency would remove a supported optional feature, so that is not a
maintenance-only repair.

Minor-only lock refreshes must retain each package's current major and lower
bound. Use version-qualified `uv lock --upgrade-package` arguments, inspect a
`--dry-run` first, and compare the resulting package versions. NumPy has two
Python-dependent branches, so apply separate constraints on either side of
Python 3.13. A blanket upgrade can both cross a NumPy major and downgrade a
neighboring OCR package to satisfy the new graph.

### Dependency compatibility bounds

The published extras and Dependabot configuration carry matching bounds:

- Transformers stays below 5.16 while tokenizers stays on the verified 0.22
  line. [PyPI metadata for Transformers 5.16](https://pypi.org/pypi/transformers/5.16.0/json)
  and [5.17](https://pypi.org/pypi/transformers/5.17.0/json) requires tokenizers
  `>=0.23.1,<0.24`, so updating Transformers alone cannot resolve. Revalidate
  tokenizer construction and the model-loading paths before lifting both bounds.
- Hugging Face Hub stays below 2 in every extra that names it. Diffusers 0.40
  [requires `huggingface-hub>=1.23,<2`](https://pypi.org/pypi/diffusers/0.40.0/json);
  a Hub-only extra must remain composable with the diffusion extra. Recheck
  Diffusers' declared requirements before lifting the Hub bound.
- C2PA requires 0.38.0 or newer and uses an explicit per-reader certificate
  profile. Its Rust SDK 0.91.0 clears the bundled EKUs when loading empty trust
  settings, which otherwise rejects the OpenAI and Microsoft signing
  certificates in five committed fixtures. The profile restores those EKUs
  without installing trust anchors or bypassing certificate validation.
  Cause, upstream sources, and regression guards are in
  [Module internals, C2PA](module-internals.md#c2pa). Recheck both positive
  certificate acceptance and negative credential, signature, and asset-binding
  validation when upgrading the reader; retain every SDK validation status.

Dependabot ignore entries mirror these compatibility bounds. Its failed
2026-09-30 run named the active Python 3.11 interpreter in a generic resolver
hint, but the actual conflicts were the tokenizer and Hub requirements above;
reducing the supported Python range does not fix them.

## CI

`.github/workflows/test.yml` runs Ruff, a test matrix over every supported Python
minor with default plus development dependencies, cross-platform coverage at the
oldest and newest minors, and a separate job that installs ffmpeg on Ubuntu to
run the full-clip video test. Diffusion and model-running tests skip in that
matrix; metadata, identification, visible removal, the DWT-DCT decoder, and the
OpenCV eraser remain covered across operating systems.

A separate `security` job runs the `maintain.sh` lockfile scan, so an advisory
without a released fix turns `main` red; see
[Known security-gate blocks](#known-security-gate-blocks). Vulnerability
reporting is described in [`SECURITY.md`](../SECURITY.md).

Keep `uv.lock` compatible with `uv sync --frozen`. Dependency pull-request checks use GitHub's merge result against current `main`; if `main` moves, merge it locally and rerun the full gate because a newer linter can expose stale directives in later code.

`test.yml` runs one concurrency group per ref with `cancel-in-progress: true`, so a newer push cancels the previous run mid-flight. A cancelled run is not a failure verdict: the superseding run's head includes the cancelled commits as ancestors, so watch that run to completion instead of retrying the push.

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
