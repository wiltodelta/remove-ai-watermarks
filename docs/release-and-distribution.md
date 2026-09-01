# Release and distribution

This page describes the release behavior defined in this repository. External
registry state can change independently, so verify it during a release.

## Published surfaces

A release is complete only after all four published surfaces are verified:

| Surface | Published result | Automation |
|---|---|---|
| PyPI | The `remove-ai-watermarks` wheel and source distribution | `publish.yml` |
| Homebrew | The `remove-ai-watermarks` formula in `wiltodelta/homebrew-tap` | `distribute.yml` |
| Hugging Face Space | The Space demo deployed on the new release through a requirements-pin bump in `wiltodelta/raiw-hf-space` | `raiw-hf-space` sync workflow; the pin bump is manual, `distribute.yml` only re-installs the pinned version |
| ComfyUI Registry | A compatible release of `wiltodelta/ComfyUI-remove-ai-watermarks` with its own node version | `distribute.yml` and the node repository's workflows |

The GitHub Release is the trigger and release record for this flow. Conda is
not a supported publishing surface: this repository has no conda recipe or
conda publication job.

## Release sources of truth

The package version appears in:

- `pyproject.toml`;
- `src/remove_ai_watermarks/__init__.py`;
- the root package entry generated in `uv.lock`.

Update the first two, then refresh the lock file with uv. Do not edit a
line-number-specific location in `uv.lock`; its package order changes.

Before choosing a version, verify it is absent from git tags, GitHub Releases,
and PyPI. An unpublished remote tag still reserves that version; never move it,
publish the next version instead.

## Publish flow

PyPI publishing is triggered by a published GitHub Release, not by a tag push
alone.

The expected sequence:

1. update the version sources and lock file;
2. run the complete project gate;
3. commit the release change;
4. create an annotated `vX.Y.Z` tag;
5. push the commit and tag;
6. publish the GitHub Release.

`.github/workflows/publish.yml` then:

1. checks that the release tag matches `pyproject.toml`;
2. builds the package with uv;
3. publishes with `uv publish` through PyPI trusted publishing.

The workflow uses GitHub OIDC through the `pypi` environment. It does not read a
PyPI API token from the repository.

## Post-release distribution

`.github/workflows/distribute.yml` runs on the same published-release event. It
waits for the matching source distribution to appear on PyPI, then:

- updates the Homebrew tap formula URL and SHA-256;
- triggers a factory rebuild of the Hugging Face Space: this re-installs the
  version pinned in the Space repo, it does NOT upgrade the demo;
- synchronizes, tests, versions, and publishes the ComfyUI nodes.

Upgrading the Hugging Face Space is a separate, manual step: bump
`remove-ai-watermarks[visible,heif]` in `wiltodelta/raiw-hf-space` (`pyproject.toml`,
`uv lock`, and the re-exported `requirements.txt`), then push. That repository's
`sync-to-hf.yml` mirrors the files onto the Space, which rebuilds on the new
pin. Its callback smoke tests run in that repository's CI on every push; they
are the scenario check for the demo surface.

The Hugging Face **model** `wiltodelta/raiw-models` is not part of that
release fan-out. It holds the photo-classify freeze weights. The library extra
`classify` downloads that snapshot on first use, or reads
`RAIW_CLASSIFY_WEIGHTS`. Update the Hub repo only with
`.github/workflows/publish-photo-classify-hf.yml` (`workflow_dispatch`,
type `publish`). Full mode downloads the four weight files from the GitHub
Release tag `photo-classify-freeze-2026-08-31` (or the tag you pass) and
uploads them with the card in `docs/photo-classify-hf/`. Card mode updates
the README and `operating-point.json` only. A library version bump does not
upload new heads. Pin a Hub revision in production so a later freeze cannot
silently change installed clients.

The workflow can also be started manually with an optional version input.

If a distribution job fails because a repository or Hugging Face credential is
invalid, rotate the corresponding GitHub secret and rerun the failed job. A
manual Homebrew formula update is the fallback when its automation is blocked.

## Source distribution boundary

The wheel includes the package under `src/`.

The source distribution uses an explicit allowlist for `/src`, `/LICENSE`,
`/README.md`, and `/pyproject.toml` through
`[tool.hatch.build.targets.sdist]` in `pyproject.toml`. It also defensively
excludes `/data`, `/tmp`, and `/.sc`. Keep both controls: calibration captures,
test corpora, generated research outputs, and local session state do not belong
in the published package archive. Hatchling always adds the root `.gitignore` to
the sdist, so keep its comments generic and free of local operational context.
Ignore rules are not the build boundary: `data/` is deliberately tracked, while
the sdist configuration keeps it and the other excluded paths out of the archive.

## Build backend

The package uses hatchling through the unpinned `hatchling` build requirement in
`pyproject.toml`. Uploading uses uv rather than the older twine-based action.

## Other channels

The ComfyUI nodes are maintained and versioned in their own repository. After
the matching source distribution appears on PyPI, `distribute.yml` dispatches
that repository's sync workflow with the exact library version and waits for it
to finish. The sync updates the dependency floor, runs compatibility tests,
bumps the node patch version, and publishes to the ComfyUI Registry only when
those tests pass. Its daily schedule remains as a recovery path if a release
dispatch is interrupted. The `COMFYUI_RELEASE_TOKEN` repository secret is a
fine-grained token limited to the ComfyUI node repository, with Actions read and
write access.

## Release verification

Forensic transports are versioned independently from the package. Before publishing
a change to provenance metadata, provenance reports, broad forensic metadata, or
pixel evidence, run their schema 1 contract tests. Additive fields are compatible;
renaming a field, changing its type or meaning, changing a signal name or watermark
label, or removing a field requires a new output schema. Add the new serializer
without removing schema 1 so long-lived consumers can update separately. A package
release must never silently substitute its latest schema when a caller explicitly
requests an older supported one.

After publication, the `verify-release.yml` workflow runs automatically once
`distribute.yml` completes and checks every surface below. It can also be
dispatched manually with a specific version. The ComfyUI check verifies the
sync workflow's latest run succeeded rather than querying the registry directly
(the registry has no public JSON API). The full checklist for completeness:

- both wheel and source distribution exist on PyPI;
- the package version matches the tag;
- the Homebrew formula points to the new source distribution;
- the distribution workflow completed successfully;
- the ComfyUI Registry node requires the new library version;
- the Hugging Face Space serves the new release: its `requirements.txt` pin
  matches (raw file via the HF API), the runtime stage is `RUNNING` after the
  rebuild, and one live `identify` and one live `visible` API call succeed
  against the running Space;
- a clean install can run `remove-ai-watermarks --version`.

A clean-install check run immediately after publication can fail with "no
version of remove-ai-watermarks==X.Y.Z" even though the simple index already
lists it: uv serves its cached pre-release view of the index. Re-run with a
fresh cache (`UV_CACHE_DIR=$(mktemp -d)`) before treating it as a release
failure.

The ComfyUI sync run can fail the same way on its own: it resolves the version
from the PyPI JSON API (which updates first) but installs the test dependency
with pip against the simple index, whose CDN edges lag by minutes. Rerun the
failed `distribute.yml` comfyui job once the simple index lists the release;
nothing about the release itself is wrong.
