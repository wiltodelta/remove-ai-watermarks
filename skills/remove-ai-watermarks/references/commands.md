# Command recipes

Run `remove-ai-watermarks COMMAND --help` when a flag is unclear. The runtime
help is the source of truth.

## Images

```bash
remove-ai-watermarks identify image.png
remove-ai-watermarks identify image.png --json
remove-ai-watermarks identify image.png --no-visible
```

`--no-visible` skips visible and open invisible pixel detectors. Metadata
still runs.

```bash
remove-ai-watermarks visible image.png -o clean.png
remove-ai-watermarks visible image.png --mark gemini -o clean.png
remove-ai-watermarks visible image.png --keep-metadata -o clean.png
```

Image mark keys: `gemini`, `doubao`, `jimeng`, `qwen`, `wan`, `kling`, `yuanbao`,
`dola`, `workbuddy`, `samsung`, `runninghub`, `baidu`, `liblib`, `liblib_pill`, `microsoft`,
`generic_ai_label`, `openart`, `jimeng_pill`. `dola` and `workbuddy` are new (since CLI 0.45.0). Default
`--mark auto` removes every selected match. `microsoft` is the top-right AI
badge, separate from the invisible InvisMark watermark on the same vendor's
images. `jimeng_pill` is a weak detector and needs corroboration. The compact
`liblib_pill` likewise requires LiblibAI metadata or the bottom-center wordmark
(since CLI 0.39.0). `generic_ai_label` is a brand-less bare "AI生成" fallback for
OS/gallery-level AI-edit stamps with no vendor wordmark; synthetic-only
calibration, weaker evidence than the vendor-tuned marks above. `openart` sits
in the FRAME CENTER, not a corner, and is UNCALIBRATED (one confirmed real
case, no captured corpus) -- see `openart_engine.py`'s module docstring.

```bash
remove-ai-watermarks erase image.png --region 1640,1930,400,100 -o clean.png
```

Region is `x,y,width,height`. Repeat `--region` for more boxes. Backends:
`cv2`, `migan`, `lama`. `erase` has no `auto` backend; `visible`, `all`,
`batch` and the video commands do. Use this only with a user-supplied box on
content they own.

```bash
remove-ai-watermarks classify image.png
remove-ai-watermarks classify image.png --json
```

Pixel classifier, needs the `classify` extra. It is NOT provenance: it guesses
from pixels alone and `identify` never runs it. The CLI has no ONNX flag; the
optional CPU ONNX runtime is library-only (`backend="onnx"`) and needs
`classify-onnx` (since CLI 0.40.0). Labels are `ai` (both models
agree), `human`, and `unknown` -- `unknown` is an abstention, not a negative.
Photographed receipts abstain to `unknown` through the bundled receipt gate
(since CLI 0.38.0), so a receipt is not reported as `ai`. Offer `classify`
only when `identify` found no signal and the user still wants a guess, and
report it as a guess.

```bash
remove-ai-watermarks classify-source image.png
remove-ai-watermarks classify-source image.png --json
```

Abstaining 10-provider source-pipeline classifier (`openai`, `google`,
`bytedance`, `flux`, `stable_diffusion`, `meta`, `midjourney`, `ideogram`,
`firefly`, `xai`) with Stage-2 model-version attribution (since CLI 0.48.0), needs the
`source-classify` extra. Like `classify`, it is NOT provenance and NOT a SynthID
detector (`watermark_truth` is always `unknown`), and `identify` never runs it.

```bash
remove-ai-watermarks metadata image.png --check
remove-ai-watermarks metadata image.png --remove -o clean.png
```

Without `-o`, image metadata removal overwrites the source.

```bash
remove-ai-watermarks invisible image.png -o clean.png
remove-ai-watermarks invisible image.png -o clean.png --force
```

CUDA only. Default pipeline is `qwen-zimage`. The local signals it acts on are
SynthID provenance, C2PA soft bindings, Microsoft InvisMark, and the open
DWT-DCT / TrustMark detectors. Use `--force` when the user knows the file
should be regenerated even though no local signal fired.

```bash
remove-ai-watermarks invisible image.png --vendor meta -o clean.png
```

Vendor cohorts: `auto`, `openai`, `google`, `microsoft`, `meta`. `auto` derives
the strength from C2PA provenance and falls back to a resolution curve. Naming a
cohort asserts the watermark IS present, so it also runs the scrub without a
local signal, exactly like `--force`. Set it only when the USER says where the
file came from. `--vendor` is on `invisible`, `all` and `batch`.

Meta Muse Image exports can retain standalone IPTC metadata, but that generic
code does not identify Meta or detect Content Seal in pixels. `--vendor meta`
supplies the route only when the user confirms the source.
There is no `--model`, `--steps`, or `--guidance-scale` flag, and no `--device`
on the image path: auto-detection already finds the only device these profiles
run on.

Pipeline profiles: `qwen-zimage` (the default), `qwen21-zimage`, `sdxl-zimage`,
`chroma-zimage`, `auto`. All are CUDA-only and all install from the same
`qwen-zimage` extra. `qwen21-zimage` (since CLI 0.51.0) uses `Qwen/Qwen-Image-2.1` under the
**Qwen Research License** (non-commercial research and personal use only; do
not use `qwen21-zimage` for commercial workflows).
Since CLI 0.57.0, `qwen21-zimage` includes bounded geometry by default and local quality acceptance
under the same license restrictions. Its CLI path does not run automatic OCR.
It falls back to the shared Qwen-2.1 recipe for tiles and reviewed manifests.
Quality improvement does not prove watermark removal.
`auto` picks a commercial-safe engine from the file's own provenance -- sdxl-zimage for Google,
chroma-zimage for Microsoft, qwen-zimage for OpenAI (since CLI 0.39.0), Meta and unknown
-- so prefer it over naming an engine when the user has no reason to care. `sdxl-zimage` is the
heavier alternative and `chroma-zimage` is the Apache-2.0 global stage.

Verified-text manifests work with every profile (since CLI 0.40.5). Each profile
uses its own already loaded VAE donor, so `auto` keeps the normal SDXL route for a
Google image instead of switching engines.

```bash
remove-ai-watermarks all image.png -o clean.png
remove-ai-watermarks batch ./images --mode visible --output-dir ./clean
```

`batch` modes: `visible`, `invisible`, `metadata`, `all`.

## Video

```bash
remove-ai-watermarks video identify input.mp4
remove-ai-watermarks video identify input.mp4 --json
remove-ai-watermarks video identify input.mp4 --no-visible
remove-ai-watermarks video all input.mp4 -o clean.mp4
remove-ai-watermarks video metadata input.mp4 --check
remove-ai-watermarks video metadata input.mp4 --remove -o clean.mp4
remove-ai-watermarks video visible input.mp4 -o clean.mp4
remove-ai-watermarks video visible input.mp4 --mark veo -o clean.mp4
remove-ai-watermarks video invisible input.mp4 -o clean.mp4
remove-ai-watermarks video batch ./videos --mode all
```

Video mark keys: `sora`, `veo`, `seedance`, `doubao`, `dola`, `hailuo`,
`vidu`, `kling`; `vidu` is new (since CLI 0.43.0). Default `--mark auto` picks the first temporally stable match in that
order.

`video invisible` and `video all --invisible` accept MP4, MOV, and M4V only.
That is the default SynthID profile. It is lossy; its shipped noise profile
is pinned, so do not lower it.

For an owned Seedance MP4, explicitly opt into the experimental fixed recipe
(since CLI 0.56.0):

```bash
remove-ai-watermarks video all input.mp4 -o clean.mp4 --invisible --profile seedance-experimental
remove-ai-watermarks video invisible label-cleaned.mp4 -o clean.mp4 --profile seedance-experimental
remove-ai-watermarks video batch ./videos --invisible --profile seedance-experimental
```

Seedance requires SDR 8-bit MP4 with even dimensions. `--quality delivery`
(default) writes H.264 High CRF18; `--quality master` retains a larger
lossless-coded master. Leave SynthID tuning flags at their defaults; nondefault
values are rejected for this fixed recipe. All/batch requires `--invisible`
with this profile. Only all mode runs the supported visible stage first;
unhandled visible overlays need separate preparation. The experiment has
matched-control CapCut evidence but performs no per-file detector request.
These profile options require a build containing the experimental Seedance
integration; verify `video all --help` before using them.

Unlike the image path, `video invisible`, `video all` and `video batch` do
accept `--device [auto|cuda|mps|cpu]`. Leave it at `auto`. Set it only when the
user names a device, and never to claim a device the probe did not report.

Without `-o`, video commands write `<source>_clean` and keep the original.
Output extension must match the input container.

## Text

```bash
remove-ai-watermarks text identify article.md
remove-ai-watermarks text identify article.md --json
remove-ai-watermarks text metadata article.md --check
remove-ai-watermarks text metadata article.md --remove -o article_clean.md
remove-ai-watermarks text invisible article.md -o article_clean.md --seed 0
remove-ai-watermarks text all article.md -o article_clean.md
remove-ai-watermarks text all article.md -o article_clean.md --invisible --seed 0
```

`text identify` and `text metadata` need no optional extras: they inspect and
strip C2PA 2.3/2.4 Appendix A.8 (`C2PATXT\0`), Appendix A.9 (`-----BEGIN C2PA
MANIFEST-----`), Appendix A.7 (`text/html` `<head>`), `VSRMark`
(`com.cognitive-proof.vsrmark.1`), Unicode Tags (`U+E0000`..`U+E007F`), and
zero-width steganography while preserving valid Emoji `U+FE0E`/`U+FE0F` and
Emoji `U+200D` ZWJ sequences. Without `-o`, `text` commands write
`<source>_clean` and keep the original. `text invisible` and `text all
--invisible` lock code blocks, inline code, LaTeX math, URLs, file paths, HTML
tags, and Markdown headings verbatim and regenerate unlocked prose spans with an
unwatermarked local causal LM (needs the `diffusion` extra).

## MCP server

`remove-ai-watermarks mcp` (since CLI 0.46.0) serves `identify`, `remove`
and `batch` as MCP tools over stdin and stdout for MCP clients. It blocks
waiting for protocol messages, so do not run it from this skill; call the
commands above directly.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Requested work completed |
| 2 | No targeted signal (`visible` / no-signal `invisible`), or a `batch` directory with no supported file |
| 1 | Hard failure, an input that is not a readable image or video (`identify` and `metadata` included), or `all` skipped a required invisible stage |

A written file plus exit `1` from `all` is a partial result.

Click's own usage errors also exit `2` (missing file, unknown flag, directory
where a file was expected). They print `Usage:` or `Error: Invalid value` and
never touch the image, so they are not a no-signal verdict.
