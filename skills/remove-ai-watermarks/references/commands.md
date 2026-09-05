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

Image mark keys: `gemini`, `doubao`, `jimeng`, `qwen`, `kling`, `yuanbao`,
`samsung`, `runninghub`, `baidu`, `liblib`, `microsoft`, `jimeng_pill`. Default
`--mark auto` removes every selected match. `microsoft` is the top-right AI
badge, separate from the invisible InvisMark watermark on the same vendor's
images. `jimeng_pill` is a weak detector and needs corroboration.

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
from pixels alone and `identify` never runs it. Labels are `ai` (both models
agree), `human`, and `unknown` -- `unknown` is an abstention, not a negative.
Offer it only when `identify` found no signal and the user still wants a guess,
and report it as a guess.

```bash
remove-ai-watermarks verify-openai-synthid image.png --acknowledge-upload
```

This UPLOADS a copy of the user's image to OpenAI, so never run it without the
user's explicit go-ahead in this conversation; `--acknowledge-upload` is that
consent and the command refuses without it. It strips AI metadata from a
temporary pixel-identical copy, reads only the SynthID result, and never
modifies the source. Needs the `verify` extra.

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

Meta Muse Image is the case this exists for: Content Seal ships with no C2PA and
its IPTC tag is a generic code, so nothing routes it and `identify` reports
origin unknown. Without `--vendor meta` there is no way to clean it.
There is no `--model`, `--steps`, or `--guidance-scale` flag, and no `--device`
on the image path: auto-detection already finds the only device these profiles
run on.

Pipeline profiles: `qwen-zimage` (the default), `sdxl-zimage`, `chroma-zimage`,
`auto`. All are CUDA-only and all install from the same `qwen-zimage` extra.
`auto` picks the engine from the file's own provenance -- chroma-zimage for
OpenAI and Microsoft, qwen-zimage for Google, Meta and unknown -- so prefer it
over naming an engine when the user has no reason to care. `sdxl-zimage` is the
heavier alternative and `chroma-zimage` is the Apache-2.0 global stage.

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
`kling`. Default `--mark auto` picks the first temporally stable match in that
order.

`video invisible` and `video all --invisible` accept MP4, MOV, and M4V only.
They are lossy. The shipped noise profile is pinned; do not lower it.

Unlike the image path, `video invisible`, `video all` and `video batch` do
accept `--device [auto|cuda|mps|cpu]`. Leave it at `auto`. Set it only when the
user names a device, and never to claim a device the probe did not report.

Without `-o`, video commands write `<source>_clean` and keep the original.
Output extension must match the input container.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Requested work completed |
| 2 | No targeted signal (`visible` / no-signal `invisible`) |
| 1 | Hard failure, or `all` skipped a required invisible stage |

A written file plus exit `1` from `all` is a partial result.

Click's own usage errors also exit `2` (missing file, unknown flag, directory
where a file was expected). They print `Usage:` or `Error: Invalid value` and
never touch the image, so they are not a no-signal verdict.
