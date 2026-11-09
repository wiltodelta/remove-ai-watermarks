---
license: apache-2.0
library_name: numpy
pipeline_tag: image-classification
tags:
  - image-forensics
  - source-attribution
  - open-set-classification
---

# RAIW multi-provider source-pipeline classifier (`multi-provider-source-v3`)

This repository contains the pickle-free (`allow_pickle=False`) derivative weights for the abstaining `remove-ai-watermarks` pixel source-pipeline classifier across **10 AI provider families** (`openai`, `google`, `bytedance`, `flux`, `stable_diffusion`, `meta`, `midjourney`, `ideogram`, `firefly`, `xai`) plus `unknown` (`classify_source` / `remove-ai-watermarks classify-source`).

This model is **not a SynthID detector** and does not decode or verify any cryptographic or invisible watermark payload (`watermark_truth="unknown"`). Its labels describe complete generation and export pixel pipelines, so renderer, geometry, encoding, post-processing, and watermark state may be confounded. `unknown` does not mean that SynthID is absent, and a provider label does not prove that SynthID is present.

## Model architecture

- **Input (`956-d`):**
  - `124-d` spatial opponent-residual and gradient forensic features (`extract_124d`).
  - `768-d` phase-free `16x16` log-power and radial spectral features (`source_spectral`).
  - `64-d` directional / block-boundary (`8x8` and `16x16`) / chroma-aliasing / azimuthal-anisotropy features (`augment_892d_to_956d`).
- **Stage 1 (`54` primary specialist units + `1,590` auxiliary hard-margin subfamily, codec-recompression, cross-AI disambiguator, and domain-veto hinge descriptors across `10` canonical providers):**
  - `10` global provider specialists + `44` subfamily and codec/export-regime specialists (`37` active), each combining a block-diagonal MLP Soup (including 3-seed joint multi-task `54`-unit shared-trunk MLP blocks), a vectorized fixed-depth `HistGradientBoosting` tree ensemble, and a distilled `ExtraTrees -> MLP` student with calibration-reoptimized distilled blend weights (`w_e <= 0.25`), pooled via drift-bounded zero-inflation subfamily blending across 16 real-image varieties, refined by a Wave-24 Stage-1.5 2-knot zero-inflated hinge meta-router (`router_meta_tau`, `router_meta_w`) incorporating expanded-corpus tail-purified 3-component hybrid (`MLP Soup + HGB + Distilled ExtraTrees-MLP`), pre-ReLU extreme-tail zero-FP composite blends (`_tailmax`), pure-channel (`onlyhgb`, `onlymlp`, `onlyet`), extreme-value-safe steep zero-FP tail hinges (`SteepHinge`), provenance-cleaned cross-AI disambiguator (`Disambig` / `ParetoDis`), and real-domain veto specialists, and routed by a unified damped competitive margin (`gamma = 0.60`).
- **Stage 2 (`9` stacked version heads, `2600-d`):**
  - Feature-subsampled vectorized `HistGradientBoosting` tree ensembles (including multi-seed/multi-depth forest soups on `openai`, `google`, `flux`, `midjourney`, `ideogram`, `bytedance`, and `stable_diffusion`) taking the `956-d` features (and Stage-1 unit log-odds scores) to predict the canonical model/export subfamily (`version_subfamily`).
- **Artifact:** `multi-provider-source-v3.npz` (`allow_pickle=False`)
- **SHA-256:** `f65dac7cd7a6d8025e0b849bba9117635d6280628c226f47b599332898580f3c`

The artifact contains only learned parameters, normalization constants, tree arrays, labels, and calibrated thresholds. It does not contain training images, paths, image hashes, embeddings, or a training catalog.

## Evaluation (`204,050` clean audited samples; `10,586` locked test)

The operating point was selected on a disjoint `8,365`-image calibration split across 16 fine-grained real-image varieties and group-locked (`parent_sha256`) native, light-JPEG, and heavy-JPEG/resize views. On the `10,586`-image locked test split (`4,539` multi-variety `unknown` negatives + `6,047` AI provider images including preprocessed social/JPEG derivatives):

- **Unified 10-provider router:** `99.52%` `unknown` specificity (`0.48%` test `unknown` FPR; `0.73%` photographic FPR and `0.25%` non-photographic structured FPR across all 15 non-photo varieties), `69.97%` aggregate target recall across native (`76.72%`) and recompressed (`35.74%`) test views (`98.42%` provider accuracy among classified AI; `97.92%` overall precision when firing), and `0.9577` mean standalone ROC-AUC / `0.9682` binary PR-AUC on the monotonic `confidence` score (`_sigmoid_scalar(excess[best_cls])`).

| Provider | Test count | Standalone ROC AUC | Router recall | Router precision | Stage-2 version accuracy |
|---|---:|---:|---:|---:|---:|
| `openai` | 1,478 | 0.9740 | 77.3% | 97.9% | 86.5% |
| `google` | 1,382 | 0.9759 | 76.2% | 97.5% | 93.7% |
| `bytedance` | 389 | 0.9582 | 68.9% | 99.6% | 92.0% |
| `flux` | 1,357 | 0.9505 | 59.4% | 98.3% | 87.1% |
| `stable_diffusion` | 955 | 0.9742 | 71.5% | 98.0% | 90.8% |
| `meta` | 94 | 0.9806 | 81.9% | 100.0% | — (`meta:imagine`) |
| `midjourney` | 123 | 0.9594 | 57.7% | 94.7% | 91.1% |
| `ideogram` | 118 | 0.9361 | 47.5% | 98.2% | 65.3% |
| `firefly` | 103 | 0.9243 | 56.3% | 93.5% | 100.0% |
| `xai` | 48 | 0.9438 | 35.4% | 100.0% | 72.9% |

### Held-out real-image specificity by variety (`4,539` test negatives across 16 real classes + foreign AI)

| Real-image variety | Test count | False positives | False-positive rate | Specificity |
|---|---:|---:|---:|---:|
| `photo` | 2,183 | 16 | 0.73% | 99.27% |
| `photo_retouched` | 339 | 2 | 0.59% | 99.41% |
| `classical_inpaint` | 17 | 0 | 0.00% | 100.00% |
| `digital_art` | 441 | 1 | 0.23% | 99.77% |
| `game_art` | 174 | 0 | 0.00% | 100.00% |
| `vector` | 135 | 0 | 0.00% | 100.00% |
| `interface` | 180 | 0 | 0.00% | 100.00% |
| `diagram` | 98 | 1 | 1.02% | 98.98% |
| `chart_data_viz` | 159 | 0 | 0.00% | 100.00% |
| `technical_architecture` | 84 | 0 | 0.00% | 100.00% |
| `procedural_graphic` | 13 | 0 | 0.00% | 100.00% |
| `document` | 165 | 0 | 0.00% | 100.00% |
| `gov_scientific_doc` | 32 | 0 | 0.00% | 100.00% |
| `receipt_invoice` | 181 | 0 | 0.00% | 100.00% |
| `print_publishing` | 229 | 1 | 0.44% | 99.56% |
| `stress_recompressed` | 103 | 1 | 0.97% | 99.03% |
| `foreign_ai` | 6 | 0 | 0.00% | 100.00% |

### External blind multi-variety real-image verification (`selective-v8`, `2,267` untouched images)

Evaluated on the hash-disjoint external `selective-v8` blind real-image cohorts (`2,267` images never seen in train, calibration, or test): **2 / 2,267 false positives (`0.09%` overall FPR, `99.91%` specificity)**.

| Blind `selective-v8` cell | Evaluated | False positives | False-positive rate | Specificity |
|---|---:|---:|---:|---:|
| `photo` | 450 | 1 | 0.22% | 99.78% |
| `vector` | 332 | 0 | 0.00% | 100.00% |
| `game_art` | 230 | 1 | 0.43% | 99.57% |
| `interface` | 342 | 0 | 0.00% | 100.00% |
| `diagram` | 305 | 0 | 0.00% | 100.00% |
| `document` | 308 | 0 | 0.00% | 100.00% |
| `stress_recompressed` | 300 | 0 | 0.00% | 100.00% |

## Intended use

Install `remove-ai-watermarks[source-classify]` and call the explicit Python API `classify_source(path)` or CLI command `remove-ai-watermarks classify-source <IMAGE>`. `identify` does not run this model. Set `RAIW_SOURCE_CLASSIFY_WEIGHTS` to the `.npz` file or its parent directory for offline deployments.
