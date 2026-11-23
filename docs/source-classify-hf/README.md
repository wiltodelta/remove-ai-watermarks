---
license: apache-2.0
library_name: onnx
pipeline_tag: image-classification
tags:
  - image-forensics
  - source-attribution
  - open-set-classification
  - ai-image-detection
  - ai-generated-content
  - provider-attribution
  - openai
  - gpt-image
  - dall-e
  - google
  - imagen
  - flux
  - midjourney
  - stable-diffusion
  - ideogram
  - firefly
  - seedream
  - grok
---

# RAIW multi-provider source-pipeline classifier (`multi-provider-source-v10`)

This repository contains the unified (`67.00 MB`), CPU-only, pickle-free (`allow_pickle=False`) multimodal weights for the abstaining [`remove-ai-watermarks`](https://github.com/wiltodelta/remove-ai-watermarks) pixel source-pipeline and synthetic-image classifier across **10 AI provider families** (`openai`, `google`, `bytedance`, `flux`, `stable_diffusion`, `meta`, `midjourney`, `ideogram`, `firefly`, `xai`) plus `unknown`, with Stage-2 model/export version attribution (`version_subfamily`) and calibrated binary synthetic detection (`synthetic_detected`, `synthetic_probability`) under foreign resize, social recompression, localized inpainting, and container transcoding.

This model is **not a SynthID detector** and does not decode or verify any cryptographic or invisible watermark payload (`watermark_truth="unknown"`). Its labels describe complete generation and export pixel pipelines, so renderer, geometry, encoding, post-processing, and watermark state may be confounded. `unknown` does not mean that SynthID is absent, and a provider label does not prove that SynthID is present.

## Quickstart (NumPy + ONNX Runtime CPU, no PyTorch required)

[![Hugging Face Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-wiltodelta%2Fraiw--source--classify-yellow)](https://huggingface.co/wiltodelta/raiw-source-classify)
[![GitHub Release](https://img.shields.io/github/v/release/wiltodelta/raiw-source-classify?label=GitHub%20weights)](https://github.com/wiltodelta/raiw-source-classify/releases/latest)
[![Leaderboard](https://img.shields.io/badge/%F0%9F%A4%97%20Leaderboard-20%20detectors-blue)](https://huggingface.co/spaces/wiltodelta/ai-image-detector-leaderboard)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiltodelta/raiw-source-classify/blob/main/demo.ipynb)

```bash
pip install "remove-ai-watermarks[source-classify]"
```

```python
import remove_ai_watermarks as raiw

res = raiw.classify_source("image.jpg")
print(res.label, res.version_subfamily, res.synthetic_detected, round(res.synthetic_probability, 4))
# e.g. 'openai' 'openai:gpt_image_1_5' True 0.9942  (native AI export)
# e.g. 'unknown' None True 0.9412                  (foreign 1024x1024 transcoded AI image)
# e.g. 'unknown' None False 0.0084                 (camera photo, UI screenshot, diagram, or scanned document)
```

Command line (installed CLI or zero-setup `uv run` with the bundled `infer.py` and open-detector `benchmark.py` from Hugging Face or GitHub):

```bash
remove-ai-watermarks classify-source image.jpg --json
uv run https://raw.githubusercontent.com/wiltodelta/raiw-source-classify/main/infer.py image.jpg --json
uv run https://raw.githubusercontent.com/wiltodelta/raiw-source-classify/main/infer.py --inspect-weights
uv run https://raw.githubusercontent.com/wiltodelta/raiw-source-classify/main/benchmark.py
uv run https://huggingface.co/wiltodelta/raiw-source-classify/resolve/main/benchmark.py --tier frontier
```

The verified `multi-provider-source-v10.npz` bundle is published identically on [Hugging Face](https://huggingface.co/wiltodelta/raiw-source-classify) and [GitHub Releases](https://github.com/wiltodelta/raiw-source-classify/releases) (automatic SHA-256-verified fallback when Hugging Face is unreachable). Try it in your browser on the [Remove AI Watermarks Hugging Face Space](https://huggingface.co/spaces/wiltodelta/remove-ai-watermarks) or at [raiw.cc](https://raiw.cc). Set `RAIW_SOURCE_CLASSIFY_WEIGHTS` to `multi-provider-source-v10.npz` or its parent directory for air-gapped deployments.

## Model architecture

- **Physics, sublattice, sensor, and codec branch (`512-d`, `extract_512d_v4_features`):**
  - `112-d` adaptive overlapping native-tile truncated SRM micro-residuals across flat and textured regions.
  - `128-d` phase-resolved modulo-8 / modulo-16 sublattice energy, cropped-grid secondary-phase detection, interior VAE modulation, and `8x8` block-DCT quantization comb statistics.
  - `144-d` Moisan periodic-plus-smooth (`p + s`) spectral decomposition, radial decay profile, and continuous scale-invariant period-comb scanner.
  - `96-d` patch-averaged cross-channel spectral phase coherence, Bayer/Quad-Bayer CFA periodicity, heteroscedastic shot-noise slope, and radial chromatic-aberration alignment.
  - `32-d` explicit codec, aspect-ratio, scale, and social-recompression regime vector (`x[480:512]`).
- **Dual-branch visual & spatial split-tail (`OwensLab/commfor-model-384` MIT `ViT-B/16 @ 384` selective weight-fp16 ONNX, `vit_onnx_bytes`, `66-d`):**
  - Permissive MIT `OwensLab/commfor-model-384` (`@ 6076002bf0d9`) frozen visual stem (`blocks[0..8]`) feeding Branch A (unadapted frozen `blocks[9..11]` preserving generalist `[orig_cf_cls, orig_cf_valid_patch_mean]` logits) and Branch B (adapted `blocks[9..11]` + 4-layer `RINE` intermediate gating + 3-teacher permissive logit distillation across `CommFor-MIT`, `DINOv2-Reg4-Base`, and `SigLIP2-Base` + `SubCenterArcFace-24d` + `24x24` `3x3` `SpatialPatchHead` with `patch_clean_24x24` and `patch_gfx_24x24`), stored as a selective weight-fp16 -> fp32-compute ONNX graph inside the pickle-free `.npz` bundle and executed via `onnxruntime` on CPU, paired with coordinate-routed `1:1` unresampled `384x384` Pass-2 native tiles (`preprocess_vit_384_routed_2chw_v10`).
- **Stage 1 & 1.5 (`154` unit channels + `96` 2D gates + `209-d` 3-view open-set provider head + `117-d` global fusion + `39-d` spatial-anomaly local edit head):**
  - Two-tower `512-d` ensemble in pure NumPy (`5` multi-seed Regime-FiLM `LayerNorm` + `SiLU` residual MLP trunks + `65` packed `HistGradientBoosting` specialist forests) feeding an `11`-channel regime-standardized acceptance detector, conditional attribution MLP, and a `209-d -> 96-d -> 12` 3-view open-set multimodal provider head (`10` providers vs `logsumexp(foreign_ai, real_non_ai)`) that recovers exact `label` on native and transcoded 10-provider images while keeping unseen generators as honest `unknown`.
  - Anchored `117-d -> 64-d -> 1` global synthetic fusion head and `39-d -> 32-d -> 1` `_LocalEditHeadV9` spatial-anomaly head combining the `66-d` dual-branch ViT output, Pass-2 tile contrast, bounded `11`-channel Stage-1.5 physics acceptance vector, positive/negative global physics cues, the `26-d` codec-invariant born-digital graphics score, and the `32-d` regime block.
- **Stage 2 (`9` stacked blended + multimodal residual version heads, `666-d` + ArcFace/ViT residual):**
  - Vectorized `HistGradientBoosting` tree ensembles blended with standardized linear heads (`666-d`) and multimodal `_MMStage2HeadV8` residual version heads (`SubCenterArcFace` + ViT + regime) to predict the canonical model/export subfamily (`version_subfamily`).
- **Artifact:** `multi-provider-source-v10.npz` (`allow_pickle=False`, `67.00 MB`)
- **SHA-256:** `b3eb961cc68cf283b399bebf11cdf61b2bf41b98fabca0579f8b61d9141f695f`

The artifact contains only learned parameters, weight-fp16 ONNX graph bytes, normalization constants, tree arrays, labels, and calibrated thresholds. It does not contain training images, paths, image hashes, embeddings, or a training catalog.

## Evaluation

### Open 578-image multi-regime benchmark (`benchmark.py`)

Interactive 20-model comparison matrix: [`wiltodelta/ai-image-detector-leaderboard`](https://huggingface.co/spaces/wiltodelta/ai-image-detector-leaderboard).

Evaluated end-to-end via `classify_source` on `200` non-AI false-positive traps (`50` camera photos, `50` UI screenshots, `50` science diagrams, `50` scanned documents), `125` untouched native AI exports (`25` each across FLUX.1-dev, GPT-Image-1, Gemini Nano Banana Pro, SDXL, Midjourney v6.1), and `253` square-resized/transcoded crowd-arena AI images (`Rapidata` and `bitmind` `1024x1024` JPEG/WebP/PNG):

- **Non-AI false-positive rate (`200` traps):** **`0.0%` (`0 / 200`)** across all four slices (`0%` camera photo, `0%` UI screenshot, `0%` science diagram, `0%` scanned document) and **`100.0%` precision**.
- **Overall AI detection (`378`):** **`82.8%` binary synthetic recall** (`313 / 378`) and **`38.6%` exact generator attribution** (`46.6%` when classified).
- **Native AI exports (`125`):** **`93.6%` binary synthetic recall** (`117 / 125`) and **`76.8%` exact generator attribution** (`96 / 125`; `82.1%` when classified, including `100.0%` on SDXL and `96.0%` on FLUX.1-dev).
- **Transcoded crowd-arena AI (`253`):** **`77.5%` binary synthetic recall** (`196 / 253`, including `87.1%` on xAI Grok and `96.3%` on Ideogram) and **`19.8%` exact generator attribution** (`50 / 253`, keeping honest `unknown` abstention when provider identity is unrecoverable; **`371.8 ms`** median single-image CPU latency).

### Internal locked test & external blind multi-variety verification

The operating point was selected strictly on parent-disjoint calibration partitions (`parent_sha256` grouped). Across sealed held-out benchmarks:

- **End-to-end RGB + ViT on `4,421` resolved `locked_test` parents (`2,074` supported AI + `2,347` real non-AI across 15 real varieties):** **`0.26%` real synthetic FPR** (`6 / 2,347` false positives; **`0.00%` real provider FPR**, `0 / 2,347`), **`95.76%` native synthetic recall** (**`93.35%` exact provider recall**; **`0.9994` ROC-AUC / `0.9993` PR-AUC**), and **`70.54%` synthetic recall** (**`43.30%` exact provider recall**) after forced `1024x1024` JPEG/WebP arena transcoding across all 10 provider families (`0.77%` transcoded real FPR, `0.9817` ROC-AUC).
- **Sealed 12-unseen-family open-world generator benchmark (`480` native + transcoded views across 12 unseen generator families):** **`87.71%` binary synthetic recall** with **`94.79%` honest `label="unknown"` abstention** (and `63.10%` recall / `94.05%` abstention on the `84`-view historical foreign-generator panel).
- **External `AI Detector Arena v0.1` (`2,038` images across 17 generators + 6 studio/real categories):** **`4.31%` real FPR** (`44 / 1,020`), **`91.02%` pixels-only F1** (`91.64` combined score; **`92.26` full combined score** with provenance/visible marks, `0.9689` ROC-AUC).
- **External blind `selective-v8` end-to-end RGB + ViT (`1,512` untouched real-variety parents):** **`0 / 965` (`0.00%` FPR)** across all camera photo (`0 / 25`), `diagram` (`0 / 305`), `document` (`0 / 308`), and `interface` (`0 / 327`) parents, `1 / 332` on `vector`, and `1 / 215` on `game_art` (`2 / 1,512 = 0.13%` overall synthetic FPR; **`0 / 1,512` provider FPR**; `1 / 2,267 = 0.04%` across all `2,267` `512-d` rows; `0 / 60` on real phone scanned documents).
- **External in-the-wild social/news benchmark (`WildAIGC-Bench V3`, `1,238` images):** **`2.35%` real FPR** (`15 / 638` real posts) with **`14.17%` processed AI recall** (`15.33%` raw recall).
- **Full `10,585`-row `locked_test` `512-d` provider router:** `99.93%` `unknown` specificity (`3 / 4,538` false positives, `0.07%` FPR), `83.28%` exact provider recall (`92.45%` native, `92.95%` resized, `77.06%` social-recompressed), `99.94%` any-AI precision, and `95.93%` Stage-2 version accuracy given exact provider.

| Provider | Test count | Standalone ROC AUC | Router recall | Router precision | Stage-2 version accuracy |
|---|---:|---:|---:|---:|---:|
| `openai` | 1,478 | 0.9997 | 92.1% | 96.1% | 95.6% |
| `google` | 1,382 | 0.9996 | 86.4% | 97.5% | 97.0% |
| `bytedance` | 389 | 0.9863 | 72.0% | 98.9% | 96.8% |
| `flux` | 1,357 | 0.9993 | 80.8% | 95.2% | 95.0% |
| `stable_diffusion` | 955 | 0.9987 | 77.2% | 98.1% | 98.0% |
| `meta` | 94 | 0.9995 | 86.2% | 98.8% | - (`meta:imagine`) |
| `midjourney` | 123 | 1.0000 | 82.1% | 87.8% | 98.0% |
| `ideogram` | 118 | 0.9999 | 84.7% | 78.7% | 77.0% |
| `firefly` | 103 | 0.9948 | 61.2% | 91.3% | 100.0% |
| `xai` | 48 | 0.9990 | 45.8% | 91.7% | 77.3% |

## Intended use

Install `remove-ai-watermarks[source-classify]` and call the explicit Python API `classify_source(path)` or CLI command `remove-ai-watermarks classify-source <IMAGE>`. `identify` does not run this model. Set `RAIW_SOURCE_CLASSIFY_WEIGHTS` to the `.npz` file or its parent directory for offline deployments.
