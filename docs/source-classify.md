# Source-pipeline classification

`classify_source` (and the CLI command `classify-source`) is an opt-in,
metadata-free pixel source-pipeline classifier across **10 AI provider families**
with Stage-2 model-version attribution.

This is not a SynthID detector. It does not decode or verify a watermark
payload, and it must not be used to claim that SynthID is present or absent
(`watermark_truth="unknown"`). The model describes the complete source/export
pixel pipeline, where renderer, geometry, encoding, post-processing, and
watermark state can be confounded.

Install the lightweight runtime:

```bash
uv add "remove-ai-watermarks[source-classify]"
```

It evaluates a 512-dimensional physics-grounded forensic, sublattice, spectral,
sensor, and codec/geometry regime descriptor (`144-d` multi-scale SRM residuals
+ `128-d` phase-invariant modulo-8/16 sublattice and `8x8` DCT grid features +
`128-d` phase-free radial/azimuthal log-power spectra + `80-d` CFA/PRNU/kurtosis
sensor cues + `32-d` codec and resize regime features) through a two-tower pure
NumPy ensemble (Regime-FiLM residual MLP trunks + packed `HistGradientBoosting`
specialists), 2D multiplicative interaction gates, regime-conditioned null
standardization, damped cross-provider margin suppression ($\gamma = 0.18$), and
stacked Stage-2 version classifiers.

It routes across all 10 canonical AI provider classes (`openai`, `google`,
`bytedance`, `flux`, `stable_diffusion`, `meta`, `midjourney`, `ideogram`,
`firefly`, `xai`) or abstains as `unknown`.

Call the CLI command or Python API explicitly:

```bash
remove-ai-watermarks classify-source metadata-stripped.png
remove-ai-watermarks classify-source metadata-stripped.png --json
```

```python
import remove_ai_watermarks as raiw

result = raiw.classify_source("metadata-stripped.png")
print(result.label)              # one of the 10 providers or "unknown"
print(result.version_subfamily)  # e.g. "google:gemini_3_pro_image", "openai:gpt_image_1_5", or None
print(result.reason)             # "classified" | "abstained" | "feature_unavailable"
print(result.margins)            # provider margins before thresholding
print(result.thresholds)         # calibrated provider thresholds
print(result.watermark_truth)    # always "unknown"
```

`feature_unavailable` is returned for images below the required 256x256
geometry or for a degenerate feature vector. `abstained` means no active
provider cleared its calibrated damped margin. Both are honest `unknown`
results.

The first call downloads the exact pinned revision of
[`wiltodelta/raiw-source-classify`](https://huggingface.co/wiltodelta/raiw-source-classify)
(`multi-provider-source-v4.npz`, verified by SHA-256). Set
`RAIW_SOURCE_CLASSIFY_WEIGHTS` to the `.npz` file itself or its parent directory
for an offline deployment.

This classifier is never invoked by `identify`, `has_invisible_target`, or `all`.
The [model card](source-classify-hf/README.md) records the evaluation and
robustness limits.
