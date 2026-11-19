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

# RAIW multi-provider source-pipeline classifier (`multi-provider-source-v8`)

This repository contains the unified (`57.80 MB`), CPU-only, pickle-free (`allow_pickle=False`) multimodal weights for the abstaining [`remove-ai-watermarks`](https://github.com/wiltodelta/remove-ai-watermarks) pixel source-pipeline and synthetic-image classifier across **10 AI provider families** (`openai`, `google`, `bytedance`, `flux`, `stable_diffusion`, `meta`, `midjourney`, `ideogram`, `firefly`, `xai`) plus `unknown`, with Stage-2 model/export version attribution (`version_subfamily`) and calibrated binary synthetic detection (`synthetic_detected`, `synthetic_probability`) under foreign resize, social recompression, and container transcoding.

This model is **not a SynthID detector** and does not decode or verify any cryptographic or invisible watermark payload (`watermark_truth="unknown"`). Its labels describe complete generation and export pixel pipelines, so renderer, geometry, encoding, post-processing, and watermark state may be confounded. `unknown` does not mean that SynthID is absent, and a provider label does not prove that SynthID is present.

## Quickstart (NumPy + ONNX Runtime CPU, no PyTorch required)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiltodelta/remove-ai-watermarks/blob/main/docs/source-classify-hf/demo.ipynb)

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

Command line (installed CLI or zero-setup `uv run` with the bundled `infer.py` and open-detector `benchmark.py`):

```bash
remove-ai-watermarks classify-source image.jpg --json
uv run https://huggingface.co/wiltodelta/raiw-source-classify/resolve/main/infer.py image.jpg --json
uv run https://huggingface.co/wiltodelta/raiw-source-classify/resolve/main/infer.py --inspect-weights
uv run https://huggingface.co/wiltodelta/raiw-source-classify/resolve/main/benchmark.py
uv run https://huggingface.co/wiltodelta/raiw-source-classify/resolve/main/benchmark.py --tier frontier
```

Try it in your browser on the [Remove AI Watermarks Hugging Face Space](https://huggingface.co/spaces/wiltodelta/remove-ai-watermarks) or at [raiw.cc](https://raiw.cc). Set `RAIW_SOURCE_CLASSIFY_WEIGHTS` to `multi-provider-source-v8.npz` or its parent directory for air-gapped deployments.

## Model architecture

- **Physics, sublattice, sensor, and codec branch (`512-d`, `extract_512d_v4_features`):**
  - `112-d` adaptive overlapping native-tile truncated SRM micro-residuals across flat and textured regions.
  - `128-d` phase-resolved modulo-8 / modulo-16 sublattice energy, cropped-grid secondary-phase detection, interior VAE modulation, and `8x8` block-DCT quantization comb statistics.
  - `144-d` Moisan periodic-plus-smooth (`p + s`) spectral decomposition, radial decay profile, and continuous scale-invariant period-comb scanner.
  - `96-d` patch-averaged cross-channel spectral phase coherence, Bayer/Quad-Bayer CFA periodicity, heteroscedastic shot-noise slope, and radial chromatic-aberration alignment.
  - `32-d` explicit codec, aspect-ratio, scale, and social-recompression regime vector (`x[480:512]`).
- **Visual multi-task branch (`ViT-S/16 @ 384` weight-fp16 ONNX, `vit_onnx_bytes`, `59-d`):**
  - Permissive MIT `ViT-S/16 @ 384` frozen visual stem (`TensorFold/detectra-v1` lineage) paired with a zero-initialized parallel residual token adapter, multi-layer `RINE` intermediate hooks (`blocks[5, 9, 11]`), `SubCenterArcFace-24d` (`93` sub-centers across `31` generator subfamilies), a `24x24` `SpatialPatchHead`, and `AIDE` texture-vs-smooth high-frequency patch contrast (`59-d` output `[orig_syn(0), prov_10(1:11), foreign_ai(11), real_non_ai(12), gfx_veto(13), dom_syn(14), emb_24(15:39), proto_sim_10(39:49), sp_summary_10(49:59)]`) stored as a weight-fp16 -> fp32-compute ONNX graph inside the pickle-free `.npz` bundle and executed via `onnxruntime` on CPU, plus conditional `1:1` unresampled `modulo-16`-aligned `384x384` native-tile passes (`preprocess_vit_384_native_2chw`) for gray-zone rescue and widescreen attribution recovery.
- **Stage 1 & 1.5 (`154` unit channels + `96` 2D gates + `195-d` 3-view open-set provider head + `106-d` open-world synthetic fusion):**
  - Two-tower `512-d` ensemble in pure NumPy (`5` multi-seed Regime-FiLM `LayerNorm` + `SiLU` residual MLP trunks + `65` packed `HistGradientBoosting` specialist forests) feeding an `11`-channel regime-standardized acceptance detector, conditional attribution MLP, and a `195-d -> 96-d -> 12` 3-view open-set multimodal provider head (`10` providers vs `logsumexp(foreign_ai, real_non_ai)`) that recovers exact `label` on native and transcoded 10-provider images while keeping unseen generators as honest `unknown`.
  - Anchored `106-d -> 64-d -> 1` `LayerNorm` + `SiLU` skip multimodal synthetic fusion head combining the `59-d` ViT output, bounded `11`-channel Stage-1.5 physics acceptance vector, positive/negative global physics cues, the `26-d` codec-invariant born-digital graphics score, spatial/graphics interaction gates, and the `32-d` regime block. When container laundering or an unseen generator prevents 10-family attribution, `label` honestly abstains as `unknown` while `synthetic_detected` and `synthetic_probability` retain binary synthetic recall without sacrificing screenshot, diagram, or document specificity.
- **Stage 2 (`9` stacked blended + multimodal residual version heads, `666-d` + ArcFace/ViT residual):**
  - Vectorized `HistGradientBoosting` tree ensembles blended with standardized linear heads (`666-d`) and multimodal `_MMStage2HeadV8` residual version heads (`SubCenterArcFace` + ViT + regime) to predict the canonical model/export subfamily (`version_subfamily`).
- **Artifact:** `multi-provider-source-v8.npz` (`allow_pickle=False`, `57.80 MB`)
- **SHA-256:** `7e7a9ee708bc2d64b45b87829b2c45e487aed3936220fb09843267662c99e0da`

The artifact contains only learned parameters, weight-fp16 ONNX graph bytes, normalization constants, tree arrays, labels, and calibrated thresholds. It does not contain training images, paths, image hashes, embeddings, or a training catalog.

## Evaluation

### Open 578-image multi-regime benchmark (`benchmark.py`)

Evaluated end-to-end via `classify_source` on `200` non-AI false-positive traps (`50` camera photos, `50` UI screenshots, `50` science diagrams, `50` scanned documents), `125` untouched native AI exports (`25` each across FLUX.1-dev, GPT-Image-1, Gemini Nano Banana Pro, SDXL, Midjourney v6.1), and `253` square-resized/transcoded crowd-arena AI images (`Rapidata` and `bitmind` `1024x1024` JPEG/WebP/PNG):

- **Non-AI false-positive rate (`200` traps):** **`0.0%` (`0 / 200`)** across all four slices (`0%` camera photo, `0%` UI screenshot, `0%` science diagram, `0%` scanned document) and **`100.0%` precision**.
- **Overall AI detection (`378`):** **`92.9%` binary synthetic recall** (`351 / 378`) and **`42.6%` exact generator attribution** (`45.9%` when classified).
- **Native AI exports (`125`):** **`98.4%` binary synthetic recall** (`123 / 125`) and **`80.8%` exact generator attribution** (`101 / 125`; `82.1%` when classified, including `96.0%` on SDXL and `84.0%` on FLUX.1-dev, GPT-Image-1, and widescreen `1408x768` Gemini Nano Banana Pro).
- **Transcoded crowd-arena AI (`253`):** **`90.1%` binary synthetic recall** (`228 / 253`, up from `87.0%` in `v7` and `30.4%` in `v6`) and **`23.7%` exact generator attribution** (`60 / 253`, up from `18.6%` in `v7` and `9.5%` in `v6` via 3-view open-set multimodal attribution while keeping honest `unknown` abstention when provider identity is unrecoverable; **`385.9 ms`** median single-image CPU latency).

### Internal locked test & external blind multi-variety verification

The operating point was selected strictly on parent-disjoint calibration partitions (`parent_sha256` grouped). Across sealed held-out benchmarks:

- **End-to-end RGB + ViT on `4,415` resolved `locked_test` parents (`2,074` supported AI + `2,341` real non-AI across 15 real varieties):** **`0.09%` real synthetic FPR** (`2 / 2,341` false positives; **`0.04%` real provider FPR**, `1 / 2,341`), **`97.54%` native synthetic recall** (**`95.18%` exact provider recall**, up from `92.82%` in `v7`; **`0.9997` ROC-AUC / `0.9996` PR-AUC**), and **`79.36%` synthetic recall** (**`46.96%` exact provider recall**, up from `65.43%` / `25.41%` in `v7`) after forced `1024x1024` JPEG/WebP arena transcoding across all 10 provider families (`0.68%` transcoded real FPR, `0.9847` ROC-AUC).
- **Sealed open-world foreign-generator check (`84` held-out views outside the 10 provider families):** **`96.43%` honest `label="unknown"` abstention** (`81 / 84`; `0 / 51` false attributions on `cal-fit` foreign AI) with **`67.86%` binary synthetic recall** across unseen generators.
- **External blind `selective-v8` end-to-end RGB + ViT (`1,512` untouched real-variety parents):** **`0 / 638` (`0.00%` FPR)** across all camera photo (`0 / 25`), `diagram` (`0 / 305`), and `document` (`0 / 308`) parents, `1 / 327` on `interface`, `1 / 332` on `vector`, and `2 / 215` on `game_art` (`4 / 1,512 = 0.26%` overall synthetic FPR; `1 / 2,267 = 0.04%` across all `2,267` `512-d` rows).
- **External in-the-wild social/news benchmark (`WildAIGC-Bench V3`, `1,238` images):** **`19.67%` processed AI recall** (up from `10.50%` in `v7`) at **`3.92%` real FPR** (down from `4.86%` in `v7`, `57.87%` balanced accuracy).
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
