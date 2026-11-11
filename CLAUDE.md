# Remove AI Watermarks

You are a **principal Python engineer** maintaining a CLI tool and library for removing visible and invisible AI provenance watermarks.

## Scope and non-goals

The project gives users control over provenance marks on content they generated or edited themselves. It does not automatically remove stock-agency, marketplace, classifieds, tiled-preview, or other marks that protect a third party's paid or copyrighted asset.

- Add visible templates only for marks the vendor declares as AI-generation labels, whatever plan hides them; plan-removable product branding ("Made with Gamma") is not one.
- Do not add stock, agency, or classifieds marks to `watermark_registry.py`.
- Do not add templates for regulator-defined disclosure icons (EU Code of Practice Annex 1).
- Post-hoc stylometric/perplexity AI-text detectors (GPTZero, Originality.ai, Binoculars) are out of scope; deterministic text provenance (`C2PATXT`, C2PA HTML/ASCII armor, `VSRMark`, Unicode steganography) and span-protected local regeneration for token-level LLM watermarks (`text`) are supported ([`docs/text-watermark-research.md`](docs/text-watermark-research.md)).
- Keep `erase --region` generic and user-directed; do not build an automatic stock-watermark remover on it.

Full boundary and legal context: [`docs/legal-and-safety.md`](docs/legal-and-safety.md).

## How to run

```bash
uv run remove-ai-watermarks --help
bash maintain.sh
```

Run `uv` from the repository root. Command selection, options, defaults, and examples live in [`docs/cli.md`](docs/cli.md). Before changing command routing, no-signal behavior, or exit codes, read the command-line section of [`docs/module-internals.md`](docs/module-internals.md).

## Configuration

GPU and ML modules are optional. Guard their imports with `is_available()`.

Optional features and installation groups are documented in [`docs/installation.md`](docs/installation.md).

## Test and lint

`maintain.sh` scopes Pyright to `src/`, because full-project Pyright exhausts Node memory on the ML dependency graph.

The gate runs `scripts/sync_c2pa_soft_bindings.py --check`, which fails when the packaged C2PA soft-binding registry snapshot is stale; rerun it without `--check` to refresh the snapshot. Triaged security blocks and their recheck procedure, environment recovery, CI behavior, and fixture policy live in [`docs/development.md`](docs/development.md).

Before a release, read [`docs/release-and-distribution.md`](docs/release-and-distribution.md). Treat the release as complete only after PyPI, Homebrew, the Hugging Face Space, and the ComfyUI Registry are verified; conda is not published. The MCP registry listing publishes itself from `publish-mcp-registry.yml` and does not block a release. Keep the source-distribution public allowlist.

Victor's personal website (`wiltodelta/wiltodelta-app`) imports the README section "Choose what you want to do" from each stable release tag. Do not rename that section without updating `releaseSource` in that repository's `src/data/project-catalog.json` in the same change. After a release that changes what users can do, update that site by its `docs/product-updates.md`, "What product repositories owe the site".

The published Agent Skill is [`skills/remove-ai-watermarks/`](skills/remove-ai-watermarks/). Update it in the same change as any CLI edit that alters commands, extras, exit codes, mark keys, or the intended-use boundary. Install and publish steps: [`docs/agent-skill.md`](docs/agent-skill.md).

The same skill tree is the Codex plugin (`.codex-plugin/plugin.json`). `remove-ai-watermarks mcp` keeps a standard-library transport, discovering the shared operation catalog and executing optional backends in child processes. Its registry launcher selects the complete backend extras; bare `uvx remove-ai-watermarks mcp` remains supported for metadata-only use ([`docs/mcp-server.md`](docs/mcp-server.md)).

## Module architecture

[`docs/module-internals.md`](docs/module-internals.md) is the canonical per-module map, including design decisions, thresholds, calibration history, incident records, and regression guards. Read the relevant section before changing a subsystem.

Research, classifiers, the benchmark kernel, and current constraints are routed through [`docs/index.md`](docs/index.md); start with [`docs/known-limitations.md`](docs/known-limitations.md) and [`docs/supported-signals.md`](docs/supported-signals.md). Photo classification ([`docs/photo-classify.md`](docs/photo-classify.md)) and the export-pipeline signal ([`docs/source-classify.md`](docs/source-classify.md)) never run inside `identify` and are not SynthID detection.

## Data safety

For previously checked sources and baselines, reuse provider-oracle verdicts by exact sanitized upload SHA-256 and verification surface. Repeat a baseline only after the uploaded bytes change or concrete evidence invalidates the record; explain the reason to Victor before spending quota. Prioritize frozen new candidates.

Follow [`data/README.md`](data/README.md) for public fixture, calibration, oracle, and evaluation layout. Use only publication-cleared inputs in tracked data paths. Store each tracked binary once and keep generated evaluation outputs outside the repository.
