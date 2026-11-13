---
license: apache-2.0
library_name: numpy
pipeline_tag: image-classification
tags:
  - image-forensics
  - source-attribution
  - open-set-classification
---

# RAIW multi-provider source-pipeline classifier (`multi-provider-source-v5`)

This repository contains the compact (`15.08 MB`), pickle-free (`allow_pickle=False`) derivative weights for the abstaining `remove-ai-watermarks` pixel source-pipeline classifier across **10 AI provider families** (`openai`, `google`, `bytedance`, `flux`, `stable_diffusion`, `meta`, `midjourney`, `ideogram`, `firefly`, `xai`) plus `unknown` (`classify_source` / `remove-ai-watermarks classify-source`).

This model is **not a SynthID detector** and does not decode or verify any cryptographic or invisible watermark payload (`watermark_truth="unknown"`). Its labels describe complete generation and export pixel pipelines, so renderer, geometry, encoding, post-processing, and watermark state may be confounded. `unknown` does not mean that SynthID is absent, and a provider label does not prove that SynthID is present.

## Model architecture

- **Input (`512-d`, `extract_512d_v4_features`):**
  - `112-d` adaptive overlapping native-tile truncated SRM micro-residuals across flat and textured regions.
  - `128-d` phase-resolved modulo-8 / modulo-16 sublattice energy, cropped-grid secondary-phase detection, interior VAE modulation, and `8x8` block-DCT quantization comb statistics.
  - `144-d` Moisan periodic-plus-smooth (`p + s`) spectral decomposition, radial decay profile, and continuous scale-invariant period-comb scanner.
  - `96-d` patch-averaged cross-channel spectral phase coherence, Bayer/Quad-Bayer CFA periodicity, heteroscedastic shot-noise slope, and radial chromatic-aberration alignment.
  - `32-d` explicit codec, aspect-ratio, scale, and social-recompression regime vector (`x[480:512]`).
- **Stage 1 (`154` unit channels + `96` 2D multiplicative interaction gates across `10` canonical providers):**
  - Two-tower ensemble executed in pure NumPy: `5` multi-seed Regime-FiLM `LayerNorm` + `SiLU` residual MLP trunks (`12` provider/unknown/supported-AI logits + `45` subfamily logits) paired with `65` packed `HistGradientBoosting` specialist forests (global provider, subfamily, codec-regime, hard-domain veto, and `21` pairwise cross-AI disambiguator heads).
  - Decoupled Stage-1.5 router over the `558-d` gate-expanded feature vector: an `11`-channel regime-aware acceptance detector (per-provider acceptance channels plus a global supported-AI channel standardized against a regime-null baseline) paired with a prior-calibrated `LayerNorm` + `SiLU` conditional attribution MLP with linear skip projection (`attr_skip_w`), making provider attribution invariant to `unknown` threshold tightening.
- **Stage 2 (`9` stacked blended version heads, `666-d`):**
  - Vectorized `HistGradientBoosting` tree ensembles blended with standardized linear heads over the `512-d` descriptor concatenated with the `154` Stage-1 unit scores (`666-d`) to predict the canonical model/export subfamily (`version_subfamily`).
- **Artifact:** `multi-provider-source-v5.npz` (`allow_pickle=False`, `15.08 MB`)
- **SHA-256:** `6f9eaf2e8f2f168db019bcbf270fe5c827297f4812eb03fcf39c30b8192e1724`

The artifact contains only learned parameters, normalization constants, tree arrays, labels, and calibrated thresholds. It does not contain training images, paths, image hashes, embeddings, or a training catalog.

## Evaluation (`135,117` deduplicated feature views; `10,585` locked test)

The operating point was selected strictly on disjoint calibration partitions (`8,363` primary calibration views, `3,011` social-recompressed real negatives, and `2,972` crop/resize development views across 16 fine-grained real-image varieties and group-locked `parent_sha256` families). On the `10,585`-image locked test split (`4,538` multi-variety `unknown` negatives + `6,047` AI provider images across native, resized, and social-recompressed views):

- **Unified 10-provider router:** `99.89%` `unknown` specificity (`5 / 4,538` false positives, `0.11%` test `unknown` FPR; `0.09%` photographic FPR and `0.13%` non-photographic structured FPR), `81.23%` exact provider recall (`82.07%` on clean-provenance AI; `84.95%` any-AI recall; `91.90%` native, `79.74%` resized, and `74.05%` recompressed exact recall), `99.90%` any-AI precision when firing (`95.62%` exact-provider accuracy among classified AI; `96.19%` overall Stage-2 version accuracy given exact provider), and `0.9969` binary ROC-AUC / `0.9980` binary PR-AUC (`0.9978` mean standalone provider ROC-AUC).

| Provider | Test count | Standalone ROC AUC | Router recall | Router precision | Stage-2 version accuracy |
|---|---:|---:|---:|---:|---:|
| `openai` | 1,478 | 0.9997 | 91.1% | 96.8% | 95.7% |
| `google` | 1,382 | 0.9994 | 86.3% | 96.7% | 97.7% |
| `bytedance` | 389 | 0.9852 | 70.7% | 98.2% | 97.5% |
| `flux` | 1,357 | 0.9994 | 76.9% | 94.0% | 95.5% |
| `stable_diffusion` | 955 | 0.9994 | 72.1% | 97.7% | 97.5% |
| `meta` | 94 | 0.9994 | 86.2% | 95.3% | - (`meta:imagine`) |
| `midjourney` | 123 | 1.0000 | 81.3% | 85.5% | 98.0% |
| `ideogram` | 118 | 0.9998 | 81.4% | 76.8% | 76.0% |
| `firefly` | 103 | 0.9965 | 64.1% | 91.7% | 100.0% |
| `xai` | 48 | 0.9997 | 45.8% | 100.0% | 72.7% |

### Held-out real-image specificity by variety (`4,538` test negatives across 16 real classes + foreign AI)

| Real-image variety | Test count | False positives | False-positive rate | Specificity |
|---|---:|---:|---:|---:|
| `photo` | 2,183 | 2 | 0.09% | 99.91% |
| `photo_retouched` | 339 | 2 | 0.59% | 99.41% |
| `classical_inpaint` | 17 | 0 | 0.00% | 100.00% |
| `digital_art` | 440 | 0 | 0.00% | 100.00% |
| `game_art` | 174 | 1 | 0.57% | 99.43% |
| `vector` | 135 | 0 | 0.00% | 100.00% |
| `interface` | 180 | 0 | 0.00% | 100.00% |
| `diagram` | 98 | 0 | 0.00% | 100.00% |
| `chart_data_viz` | 159 | 0 | 0.00% | 100.00% |
| `technical_architecture` | 84 | 0 | 0.00% | 100.00% |
| `procedural_graphic` | 13 | 0 | 0.00% | 100.00% |
| `document` | 165 | 0 | 0.00% | 100.00% |
| `gov_scientific_doc` | 32 | 0 | 0.00% | 100.00% |
| `receipt_invoice` | 181 | 0 | 0.00% | 100.00% |
| `print_publishing` | 229 | 0 | 0.00% | 100.00% |
| `stress_recompressed` | 103 | 0 | 0.00% | 100.00% |
| `foreign_ai` | 6 | 0 | 0.00% | 100.00% |

### External blind multi-variety real-image verification (`selective-v8`, `2,267` untouched images)

Evaluated on the external `selective-v8` blind real-image cohorts (`2,267` images; `2,265` strictly parent-disjoint from `locked_test`): **1 / 2,267 false positives (`0.04%` overall FPR, `99.96%` specificity)**, plus **0 / 60 false positives (`0.00%` FPR)** on external phone thermal receipts.

| Blind `selective-v8` cell | Evaluated | False positives | False-positive rate | Specificity |
|---|---:|---:|---:|---:|
| `photo` | 450 | 0 | 0.00% | 100.00% |
| `vector` | 332 | 0 | 0.00% | 100.00% |
| `game_art` | 230 | 1 | 0.43% | 99.57% |
| `interface` | 342 | 0 | 0.00% | 100.00% |
| `diagram` | 305 | 0 | 0.00% | 100.00% |
| `document` | 308 | 0 | 0.00% | 100.00% |
| `stress_recompressed` | 300 | 0 | 0.00% | 100.00% |

## Intended use

Install `remove-ai-watermarks[source-classify]` and call the explicit Python API `classify_source(path)` or CLI command `remove-ai-watermarks classify-source <IMAGE>`. `identify` does not run this model. Set `RAIW_SOURCE_CLASSIFY_WEIGHTS` to the `.npz` file or its parent directory for offline deployments.
