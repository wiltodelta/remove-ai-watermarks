---
license: apache-2.0
library_name: numpy
pipeline_tag: image-classification
tags:
  - image-forensics
  - source-attribution
  - open-set-classification
---

# RAIW multi-provider source-pipeline classifier (`multi-provider-source-v4`)

This repository contains the compact (`13.35 MB`), pickle-free (`allow_pickle=False`) derivative weights for the abstaining `remove-ai-watermarks` pixel source-pipeline classifier across **10 AI provider families** (`openai`, `google`, `bytedance`, `flux`, `stable_diffusion`, `meta`, `midjourney`, `ideogram`, `firefly`, `xai`) plus `unknown` (`classify_source` / `remove-ai-watermarks classify-source`).

This model is **not a SynthID detector** and does not decode or verify any cryptographic or invisible watermark payload (`watermark_truth="unknown"`). Its labels describe complete generation and export pixel pipelines, so renderer, geometry, encoding, post-processing, and watermark state may be confounded. `unknown` does not mean that SynthID is absent, and a provider label does not prove that SynthID is present.

## Model architecture

- **Input (`512-d`, `extract_512d_v4_features`):**
  - `144-d` multi-scale spatial SRM residuals, directional kurtosis, and cross-channel Y/Co/Cg co-occurrence statistics.
  - `128-d` phase-invariant modulo-8 / modulo-16 sublattice energy, VAE upsampler checkerboards, `8x8` block-DCT quantization combs, and 4:2:0 chroma block-boundary ratios.
  - `128-d` phase-free multi-patch radial and azimuthal log-power spectra, high-frequency rolloff slopes, and resampling second-derivative spectral peaks.
  - `80-d` camera-sensor physics (Bayer CFA demosaicing periodicity, local noise-level heteroscedasticity, highlight clipping, edge ringing, and document/text binarization cues).
  - `32-d` explicit codec, aspect-ratio, scale, and social-recompression regime vector (`x[480:512]`).
- **Stage 1 (`132` unit channels + `68` 2D multiplicative interaction gates across `10` canonical providers):**
  - Two-tower ensemble executed in pure NumPy: `5` multi-seed Regime-FiLM `LayerNorm` + `SiLU` residual MLP trunks (`11` provider/unknown logits + `45` subfamily logits) paired with `44` packed `HistGradientBoosting` specialist forests (global provider, subfamily, codec-regime, hard-domain veto, and pairwise cross-AI disambiguator heads).
  - Smooth softplus hinge activations combined with `68` 2D multiplicative interaction gates (`provider x hard-domain veto`, `provider x pairwise disambiguator`, `MLP x HGB` agreement), standardized against a learned regime-conditioned null baseline `(z_raw - mu(r)) / sigma(r)` and routed via damped competitive margins ($\gamma = 0.18$).
- **Stage 2 (`9` stacked version heads, `644-d`):**
  - Vectorized `HistGradientBoosting` tree ensembles taking the `512-d` descriptor concatenated with the `132` Stage-1 unit scores to predict the canonical model/export subfamily (`version_subfamily`).
- **Artifact:** `multi-provider-source-v4.npz` (`allow_pickle=False`, `13.35 MB`)
- **SHA-256:** `bc62ee22331c0137e76c53f5026e14d08cae12a3483fa69f3000316806c146a3`

The artifact contains only learned parameters, normalization constants, tree arrays, labels, and calibrated thresholds. It does not contain training images, paths, image hashes, embeddings, or a training catalog.

## Evaluation (`127,001` deduplicated feature views; `10,585` locked test)

The operating point was selected strictly on a disjoint calibration partition (`8,363` primary calibration views + `3,011` social-recompressed real negatives across 16 fine-grained real-image varieties and group-locked `parent_sha256` families). On the `10,585`-image locked test split (`4,538` multi-variety `unknown` negatives + `6,047` AI provider images across native, resized, and social-recompressed views):

- **Unified 10-provider router:** `99.76%` `unknown` specificity (`0.24%` test `unknown` FPR; `0.14%` photographic FPR and `0.34%` non-photographic structured FPR), `81.35%` exact provider recall (`85.23%` any-AI recall; `88.08%` native, `84.58%` resized, and `76.81%` recompressed exact recall), `99.79%` any-AI precision when firing (`95.44%` exact-provider accuracy among classified AI; `96.32%` overall Stage-2 version accuracy given exact provider), and `0.9974` binary ROC-AUC / `0.9982` binary PR-AUC (`0.9976` mean standalone provider ROC-AUC).

| Provider | Test count | Standalone ROC AUC | Router recall | Router precision | Stage-2 version accuracy |
|---|---:|---:|---:|---:|---:|
| `openai` | 1,478 | 0.9997 | 92.5% | 96.4% | 95.8% |
| `google` | 1,382 | 0.9993 | 85.5% | 96.3% | 97.3% |
| `bytedance` | 389 | 0.9853 | 69.9% | 98.9% | 97.8% |
| `flux` | 1,357 | 0.9992 | 80.6% | 94.2% | 96.2% |
| `stable_diffusion` | 955 | 0.9990 | 70.5% | 95.3% | 97.8% |
| `meta` | 94 | 0.9987 | 89.4% | 93.3% | - (`meta:imagine`) |
| `midjourney` | 123 | 1.0000 | 63.4% | 92.9% | 100.0% |
| `ideogram` | 118 | 0.9997 | 75.4% | 73.0% | 74.2% |
| `firefly` | 103 | 0.9959 | 58.3% | 96.8% | 100.0% |
| `xai` | 48 | 0.9996 | 43.8% | 100.0% | 71.4% |

### Held-out real-image specificity by variety (`4,538` test negatives across 16 real classes + foreign AI)

| Real-image variety | Test count | False positives | False-positive rate | Specificity |
|---|---:|---:|---:|---:|
| `photo` | 2,183 | 3 | 0.14% | 99.86% |
| `photo_retouched` | 339 | 1 | 0.29% | 99.71% |
| `classical_inpaint` | 17 | 0 | 0.00% | 100.00% |
| `digital_art` | 440 | 1 | 0.23% | 99.77% |
| `game_art` | 174 | 3 | 1.72% | 98.28% |
| `vector` | 135 | 1 | 0.74% | 99.26% |
| `interface` | 180 | 0 | 0.00% | 100.00% |
| `diagram` | 98 | 0 | 0.00% | 100.00% |
| `chart_data_viz` | 159 | 0 | 0.00% | 100.00% |
| `technical_architecture` | 84 | 0 | 0.00% | 100.00% |
| `procedural_graphic` | 13 | 0 | 0.00% | 100.00% |
| `document` | 165 | 0 | 0.00% | 100.00% |
| `gov_scientific_doc` | 32 | 0 | 0.00% | 100.00% |
| `receipt_invoice` | 181 | 0 | 0.00% | 100.00% |
| `print_publishing` | 229 | 0 | 0.00% | 100.00% |
| `stress_recompressed` | 103 | 2 | 1.94% | 98.06% |
| `foreign_ai` | 6 | 0 | 0.00% | 100.00% |

### External blind multi-variety real-image verification (`selective-v8`, `2,267` untouched images)

Evaluated on the hash-disjoint external `selective-v8` blind real-image cohorts (`2,267` images never seen in train, calibration, or test): **2 / 2,267 false positives (`0.09%` overall FPR, `99.91%` specificity)**, plus **0 / 60 false positives (`0.00%` FPR)** on external phone thermal receipts.

| Blind `selective-v8` cell | Evaluated | False positives | False-positive rate | Specificity |
|---|---:|---:|---:|---:|
| `photo` | 450 | 1 | 0.22% | 99.78% |
| `vector` | 332 | 0 | 0.00% | 100.00% |
| `game_art` | 230 | 0 | 0.00% | 100.00% |
| `interface` | 342 | 0 | 0.00% | 100.00% |
| `diagram` | 305 | 0 | 0.00% | 100.00% |
| `document` | 308 | 0 | 0.00% | 100.00% |
| `stress_recompressed` | 300 | 1 | 0.33% | 99.67% |

## Intended use

Install `remove-ai-watermarks[source-classify]` and call the explicit Python API `classify_source(path)` or CLI command `remove-ai-watermarks classify-source <IMAGE>`. `identify` does not run this model. Set `RAIW_SOURCE_CLASSIFY_WEIGHTS` to the `.npz` file or its parent directory for offline deployments.
