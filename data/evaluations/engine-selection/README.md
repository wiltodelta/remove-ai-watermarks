# Engine-selection fixture set

This directory is the tracked, content-balanced input set for researching a
deterministic `auto` engine choice. It is not a training corpus and it is not a
watermark-oracle corpus.

## Content matrix

`content-manifest.csv` contains 19 prompt-matched pairs. Each pair uses the
same prompt and generation index across OpenAI `gpt-image-1-mini` and Meta
`muse-image-1.0`, covering the content classes that materially stress image
regeneration:

- natural and urban scenes;
- single-person portraits and people in context;
- animals, food, products, interiors, architecture, macro detail, and action;
- flat, isometric, watercolor, and surreal illustration styles.

The files are exact copies of generation outputs collected for the
`docs/arxiv-paper-review` research branch. The manifest records the SHA-256 of
the committed bytes. The old OpenAI collection manifest stored the API payload
hash under `sha256`; `source_payload_sha256` preserves that value while
`sha256` always means the actual file bytes here. Meta's two hashes are the
same.

The project maintainer has cleared these exact committed outputs for permanent
public test and benchmark reuse. `reuse_basis` records that clearance on every
row. It is permission for this fixed fixture set, not a claim that either
provider grants a general-purpose dataset license for arbitrary generations.

One generation per prompt and provider is enough for the first paired
discovery pass. Any policy inferred from it must be challenged against unused
generation indices from the local research archive before shipping. Generated
outputs belong outside this directory.

`carrier-manifest.csv` adds the canonical, already tracked signal carriers to
the study without copying their bytes. Together the two manifests cover
ordinary content, text, single faces, face grids, and mixed face/text scenes.

## Signal carriers

The content matrix was re-encoded by its collection pipeline and has no local
provenance signal. It must not be used to set or certify watermark-removal
strengths. The tracked oracle/carrier inputs remain in their canonical homes:

- OpenAI and Google SynthID: `data/synthid/originals/` with
  `data/synthid/manifest.csv`;
- Meta Content Seal: `data/contentseal/originals/` with
  `data/contentseal/manifest.csv`.

There is no tracked, publication-cleared Microsoft InvisMark carrier yet. The
three Paint sources used by the earlier calibration are not publication-cleared
and remain outside git. A Microsoft row may be added only from a newly generated or
otherwise publication-cleared source, with a pixel-identical metadata-stripped
control that remains positive in Microsoft's detector.

## Integrity check

Run:

```bash
uv run python scripts/verify_engine_selection_fixtures.py
```

The check validates paths, byte hashes, dimensions, prompt pairing, and the
declared provider/content matrix without loading any ML model.
