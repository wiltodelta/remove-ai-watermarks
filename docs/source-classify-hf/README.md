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

# RAIW multi-provider source-pipeline classifier (`multi-provider-source-v7`)

This repository contains the unified (`55.53 MB`), CPU-only, pickle-free (`allow_pickle=False`) multimodal weights for the abstaining [`remove-ai-watermarks`](https://github.com/wiltodelta/remove-ai-watermarks) pixel source-pipeline and synthetic-image classifier across **10 AI provider families** (`openai`, `google`, `bytedance`, `flux`, `stable_diffusion`, `meta`, `midjourney`, `ideogram`, `firefly`, `xai`) plus `unknown`, with Stage-2 model/export version attribution (`version_subfamily`) and calibrated binary synthetic detection (`synthetic_detected`, `synthetic_probability`) under foreign resize and container transcoding.

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

Try it in your browser on the [Remove AI Watermarks Hugging Face Space](https://huggingface.co/spaces/wiltodelta/remove-ai-watermarks) or at [raiw.cc](https://raiw.cc). Set `RAIW_SOURCE_CLASSIFY_WEIGHTS` to `multi-provider-source-v7.npz` or its parent directory for air-gapped deployments.

## Model architecture

- **Physics, sublattice, sensor, and codec branch (`512-d`, `extract_512d_v4_features`):**
  - `112-d` adaptive overlapping native-tile truncated SRM micro-residuals across flat and textured regions.
  - `128-d` phase-resolved modulo-8 / modulo-16 sublattice energy, cropped-grid secondary-phase detection, interior VAE modulation, and `8x8` block-DCT quantization comb statistics.
  - `144-d` Moisan periodic-plus-smooth (`p + s`) spectral decomposition, radial decay profile, and continuous scale-invariant period-comb scanner.
  - `96-d` patch-averaged cross-channel spectral phase coherence, Bayer/Quad-Bayer CFA periodicity, heteroscedastic shot-noise slope, and radial chromatic-aberration alignment.
  - `32-d` explicit codec, aspect-ratio, scale, and social-recompression regime vector (`x[480:512]`).
- **Visual multi-task branch (`ViT-S/16 @ 384` weight-fp16 ONNX, `vit_onnx_bytes`):**
  - Permissive MIT `ViT-S/16 @ 384` frozen visual stem (`OwensLab/commfor-model-384` / `TensorFold/detectra-v1` lineage) paired with open-world multi-task readout heads (`[cls_tok(384), patch_mean(384), patch_std(384)]` -> `28-d` vector `[orig_syn(0), prov_10(1:11), foreign_ai(11), real_non_ai(12), gfx_veto(13), dom_syn(14), emb(15:28)]`) stored as a weight-fp16 -> fp32-compute ONNX graph inside the pickle-free `.npz` bundle and executed via `onnxruntime` on CPU, plus a conditional `short_edge=672` 2-crop gray-zone pass for widescreen native exports.
- **Stage 1 & 1.5 (`154` unit channels + `96` 2D gates + `82-d` open-set provider head + `67-d` open-world synthetic fusion):**
  - Two-tower `512-d` ensemble in pure NumPy (`5` multi-seed Regime-FiLM `LayerNorm` + `SiLU` residual MLP trunks + `65` packed `HistGradientBoosting` specialist forests) feeding an `11`-channel regime-standardized acceptance detector, conditional attribution MLP, and an `82-d -> 64-d -> 12` open-set multimodal provider head (`10` providers vs `logsumexp(foreign_ai, real_non_ai)`) that recovers exact `label` on transcoded 10-provider images while keeping unseen generators as honest `unknown`.
  - Anchored `67-d -> 48-d -> 1` `LayerNorm` + `SiLU` skip multimodal synthetic fusion head combining the `28-d` ViT output, bounded `11`-channel Stage-1.5 physics acceptance vector, positive/negative global physics cues, the `26-d` codec-invariant born-digital graphics score, and the `32-d` regime block. When container laundering or an unseen generator prevents 10-family attribution, `label` honestly abstains as `unknown` while `synthetic_detected` and `synthetic_probability` retain binary synthetic recall without sacrificing screenshot, diagram, or document specificity.
- **Stage 2 (`9` stacked blended version heads, `666-d`):**
  - Vectorized `HistGradientBoosting` tree ensembles blended with standardized linear heads over the `512-d` descriptor concatenated with the `154` Stage-1 unit scores (`666-d`) to predict the canonical model/export subfamily (`version_subfamily`).
- **Artifact:** `multi-provider-source-v7.npz` (`allow_pickle=False`, `55.53 MB`)
- **SHA-256:** `4faef51eea48f8194ee804fab1782c48d626384e115e2ef70884f8e8393d7eb1`

The artifact contains only learned parameters, weight-fp16 ONNX graph bytes, normalization constants, tree arrays, labels, and calibrated thresholds. It does not contain training images, paths, image hashes, embeddings, or a training catalog.

## Evaluation

### Open 578-image multi-regime benchmark (`benchmark.py`)

Evaluated end-to-end via `classify_source` on `200` non-AI false-positive traps (`50` camera photos, `50` UI screenshots, `50` science diagrams, `50` scanned documents), `125` untouched native AI exports (`25` each across FLUX.1-dev, GPT-Image-1, Gemini Nano Banana Pro, SDXL, Midjourney v6.1), and `253` square-resized/transcoded crowd-arena AI images (`Rapidata` and `bitmind` `1024x1024` JPEG/WebP/PNG):

- **Non-AI false-positive rate (`200` traps):** **`0.0%` (`0 / 200`)** across all four slices (`0%` camera photo, `0%` UI screenshot, `0%` science diagram, `0%` scanned document) and **`100.0%` precision**.
- **Native AI exports (`125`):** **`99.2%` binary synthetic recall** (`124 / 125`, including `96.0%` on widescreen `1408x768` Gemini Nano Banana Pro and `100.0%` on widescreen `1456x816` Midjourney v6.1, FLUX.1-dev, GPT-Image-1, and SDXL) and **`71.2%` exact generator attribution** (`89 / 125`).
- **Transcoded crowd-arena AI (`253`):** **`87.0%` binary synthetic recall** (`220 / 253`, up from `30.4%` in `v6`) and **`18.6%` exact generator attribution** (`47 / 253`, up from `9.5%` in `v6` via open-set multimodal attribution while keeping honest `unknown` abstention when provider identity is unrecoverable; **`329 ms`** median single-image CPU latency).

### Internal locked test & external blind multi-variety verification

The operating point was selected strictly on parent-disjoint calibration partitions (`parent_sha256` grouped). Across sealed held-out benchmarks:

- **End-to-end RGB + ViT on `4,415` resolved `locked_test` parents (`2,074` supported AI + `2,341` real non-AI across 15 real varieties):** **`0.09%` real synthetic FPR** (`2 / 2,341` false positives, `0` on camera photos, UI, diagrams, vectors, and documents; **`0.04%` real provider FPR**, `1 / 2,341`), **`97.54%` native synthetic recall** (`92.82%` exact provider recall; **`0.9995` ROC-AUC / `0.9994` PR-AUC**), and **`65.43%` synthetic recall** (`25.41%` exact provider recall) after forced `1024x1024` JPEG/WebP arena transcoding across all 10 provider families (`0.9740` ROC-AUC).
- **Sealed open-world foreign-generator check (`84` held-out views outside the 10 provider families):** **`97.62%` honest `label="unknown"` abstention** (`82 / 84`; `0 / 58` false attributions on `cal-fit` foreign AI) with **`69.05%` binary synthetic recall** across unseen generators including hard-flash DreamBooth style transfers.
- **External blind `selective-v8` end-to-end RGB + ViT (`1,512` untouched real-variety parents):** **`0 / 1,297` (`0.00%` FPR)** across all camera photo (`0 / 25`), `diagram` (`0 / 305`), `document` (`0 / 308`), `interface` (`0 / 327`), and `vector` (`0 / 332`) parents, and `4 / 215` on `game_art` (`4 / 1,512 = 0.26%` overall synthetic FPR; **`0 / 1,512 = 0.00%` provider FPR**; `1 / 2,267 = 0.04%` across all `2,267` `512-d` rows).
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
