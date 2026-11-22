# Source-pipeline classification

`classify_source` (and the CLI command `classify-source`) is an opt-in,
metadata-free pixel source-pipeline classifier across **10 AI provider families**
with Stage-2 model-version attribution.

This is not a SynthID detector. It does not decode or verify a watermark
payload, and it must not be used to claim that SynthID is present or absent
(`watermark_truth="unknown"`). The model describes the complete source/export
pixel pipeline, where renderer, geometry, encoding, post-processing, and
watermark state can be confounded.

Install the runtime:

```bash
uv add "remove-ai-watermarks[source-classify]"
```

It evaluates a 512-dimensional physics-grounded forensic, sublattice, spectral,
sensor, and codec/geometry regime descriptor (`112-d` adaptive native-tile SRM
residuals + `128-d` phase-resolved modulo-8/16 sublattice and `8x8` DCT comb
features + `144-d` Moisan periodic-plus-smooth spectral and period-comb profiles
+ `96-d` CFA/shot-noise/chromatic-aberration sensor cues + `32-d` codec and
geometry regime features) together with an embedded weight-fp16 `ViT-S/16 @ 384`
multi-task ONNX visual backbone (`59-d` RINE + Sub-Center ArcFace + `24x24`
spatial patch and AIDE pools via `onnxruntime` CPU, no PyTorch required) inside
a single pickle-free `.npz` bundle (`multi-provider-source-v8.npz`,
`allow_pickle=False`). Two-tower `512-d` Regime-FiLM residual MLP trunks and
packed `HistGradientBoosting` specialists drive regime-aware 10-provider
acceptance and Stage-2 version attribution, a `195-d` 3-view open-set multimodal
provider head (`10` providers vs `foreign_ai` and `real_non_ai`) and multimodal
Stage-2 residual heads recover exact family and version attribution on native
and transcoded outputs while keeping unseen generators as `unknown`, and a
`106-d` open-world synthetic fusion head (paired with conditional `1:1`
unresampled `384x384` native-tile passes for gray-zone and widescreen exports)
combines the ViT visual/spatial representations, physics channels, regime
indicators, and a `26-d` codec-invariant born-digital graphics score.

It routes across all 10 canonical AI provider classes (`openai`, `google`,
`bytedance`, `flux`, `stable_diffusion`, `meta`, `midjourney`, `ideogram`,
`firefly`, `xai`) or abstains as `unknown`, while simultaneously returning a
calibrated binary synthetic verdict (`synthetic_detected`,
`synthetic_probability`) in a single call without extra flags.

Call the CLI command or Python API explicitly:

```bash
remove-ai-watermarks classify-source metadata-stripped.png
remove-ai-watermarks classify-source metadata-stripped.png --json
```

```python
import remove_ai_watermarks as raiw

result = raiw.classify_source("metadata-stripped.png")
print(result.label)                  # one of the 10 providers or "unknown"
print(result.version_subfamily)      # e.g. "google:gemini_3_pro_image", "openai:gpt_image_1_5", or None
print(result.synthetic_detected)     # calibrated binary synthetic verdict (bool)
print(result.synthetic_probability)  # calibrated binary synthetic probability (>= 0.5 iff True)
print(result.reason)                 # "classified" | "abstained" | "feature_unavailable"
print(result.margins)                # provider margins before thresholding
print(result.thresholds)             # calibrated provider thresholds
print(result.watermark_truth)        # always "unknown"
```

`feature_unavailable` is returned for images below the required 256x256
geometry or for a degenerate feature vector. `abstained` means the image did not
clear the calibrated 10-provider pixel-pipeline threshold.

## Behavior after foreign resize and container transcoding

When an external platform resizes an AI output to a foreign canvas (such as a
square `1024x1024` JPEG or WebP container) and overwrites native quantization
tables, aspect ratios, and VAE sublattice phase, exact 10-provider attribution
abstains as `unknown` rather than guessing a generator family, while the unified
multimodal synthetic head still evaluates `synthetic_detected` and
`synthetic_probability` in the same call:

- When a supported provider pipeline is accepted (`result.label != "unknown"`),
  `result.synthetic_detected` is guaranteed to be `True`
  (`result.synthetic_probability >= 0.5`).
- When container laundering or an unsupported generator prevents reliable
  10-provider family attribution, `result.label` stays `"unknown"`
  (`result.reason == "abstained"`) while `result.synthetic_detected` remains
  `True` whenever the multimodal synthetic head clears its calibrated threshold.
- Camera photographs, UI screenshots, scientific diagrams, vectors, and scanned
  documents are jointly guarded against false positives across both
  native and recompressed containers.

The first call downloads the exact pinned revision of
[`wiltodelta/raiw-source-classify`](https://huggingface.co/wiltodelta/raiw-source-classify)
(`multi-provider-source-v8.npz`, verified by SHA-256), falling back
automatically to the matching [GitHub Release](https://github.com/wiltodelta/raiw-source-classify/releases)
if Hugging Face Hub is unreachable. Set
`RAIW_SOURCE_CLASSIFY_WEIGHTS` to the `.npz` file itself or its parent directory
for an offline deployment.

This classifier is never invoked by `identify`, `has_invisible_target`, or `all`.
The [model card](source-classify-hf/README.md) records the evaluation and
robustness limits.
