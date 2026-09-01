# Private SynthID research manifest

The research manifest is a local CSV for detector and pixel-only removal
experiments. It is intentionally separate from `manifest.csv`, which describes
the small public regression corpus.

Store the CSV and its media below `.local-eval/synthid/`. Keep one manifest per
detector target, for example `google-manifest.csv` and `openai-manifest.csv`.
Separate manifests may reference the same external negative bytes without
duplicating an artifact inside either provider's split graph. Do not commit
private media, corpus sizes, oracle sessions, or provider-access details.

Before assigning labels, build a label-free inventory. It records byte and
decoded-pixel hashes, exact duplicates, geometry, and format, but deliberately
has no provider, oracle, outcome, or split columns:

```bash
uv run python scripts/synthid_research_inventory.py \
  --root .local-eval/synthid negatives google openai \
  --inventory-out .local-eval/synthid/inventory.csv
```

An inventory row becomes a manifest row only after its source, detector target,
label evidence, group lineage, and split are established independently. A
directory name is not evidence and must not be promoted mechanically.

Audit each local manifest with:

```bash
uv run python scripts/synthid_research_manifest.py \
  .local-eval/synthid/google-manifest.csv --verify-files
```

Once a provider manifest has ordinary positives and negatives in train,
validation, and locked-test splits, run the D1 confound challenge:

```bash
uv run --extra pixels python scripts/synthid_confound_probe.py \
  .local-eval/synthid/google-manifest.csv \
  --target-provider google \
  --report-out .local-eval/synthid/google-d1-confounds.json
```

D1 excludes candidate, sham, and source-control rows. Its `container`,
`thumbnail`, and `canonical` baselines intentionally measure how well export
fields, geometry, and coarse generator/content style can imitate detection. A
report is evidence-ready only when the locked test includes a same-provider hard
negative and the temporal split contains both labels. Evidence readiness does
not mean the D1 gate passed; a candidate signal still has to beat the frozen
canonical baseline on those controls.

## Columns

| Column | Meaning |
| --- | --- |
| `artifact_sha256` | SHA-256 of the exact file submitted or measured. |
| `pixel_sha256` | SHA-256 of decoded RGB bytes, used to catch lossless duplicates. |
| `artifact_path` | Safe path relative to the manifest. |
| `parent_sha256` | Exact parent artifact for a derivative, empty for an original. |
| `group_id` | Leakage boundary shared by an original and all semantic or transformed siblings. |
| `target_provider` | Detector target, `openai` or `google`. |
| `source_provider` | Actual source family: `openai`, `google`, `camera`, `other_ai`, `synthetic`, or `editor`. |
| `surface` | Product or export surface. |
| `model_epoch` | Model family plus a dated epoch when the exact version is unavailable. |
| `generation_session` | Groups outputs created in one provider session. |
| `content_stratum` | Predeclared content class. |
| `width`, `height`, `format` | Decoded geometry and file format. |
| `transform` | `original` or the reproducible transform applied to `parent_sha256`. |
| `split` | `discovery`, `train`, `validation`, `test`, or `temporal`. |
| `c2pa_outcome` | C2PA result, recorded independently from the pixel signal. |
| `synthid_outcome` | `detected`, `not_detected`, `indeterminate`, `refused`, or `not_checked`. |
| `verified_via` | Matching provider oracle, external source evidence, or `none`. |
| `evidence_reference` | Stable URL or local evidence-record reference required for `source-evidence`; never infer it from a directory name. |
| `oracle_session` | Groups allowed checks made in one session. |
| `oracle_role` | `ordinary`, `source_control`, `candidate`, or `sham`. |
| `captured_at`, `oracle_checked_at` | Timezone-aware ISO-8601 timestamps. |
| `notes` | Verbatim context that does not fit a structured column. |

`source-evidence` may establish an ordinary external negative, but never a
positive or a same-provider negative. Provider positives and same-provider hard
negatives require the matching provider verifier. An `indeterminate`, `refused`,
or unchecked row may remain in `discovery`; it cannot enter a train, validation,
test, or temporal split. Every `source-evidence` row must retain the source URL
or evidence-record reference that establishes the claim.

The auditor also rejects duplicate artifact hashes, identical decoded pixels in
different groups, derivatives without parents, cross-provider parentage,
lineage cycles, and any group crossing split boundaries. A `not_detected`
candidate or sham is valid only when the same provider, group, and oracle
session contains a detected `source_control`. A session that misses its control
cannot establish removal, even when its candidate response is negative.
