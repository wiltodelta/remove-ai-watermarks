# Doubao visible watermark capture

> **Status (captured 2026-05-29; alpha rebuilt 2026-05-31):** the black/gray/white captures were
> taken and are now **committed** in `captures/` (solid colour + watermark, content-free). The alpha
> map is rebuilt by `scripts/visible_alpha_solve.py doubao` (the careful gray-self solve shared with
> Jimeng). The first build claimed "pixel-exact" but left a readable outline on the real sample (issue
> #13 follow-up); removal now reverse-alphas, NCC-aligns, and applies a thin residual inpaint. See the
> `doubao_engine.py` notes in the root `CLAUDE.md`. The text below is kept as the historical capture plan.

Goal: capture the Doubao "豆包AI生成" visible watermark over known flat backgrounds so we can
build a per-pixel alpha map and a reverse-alpha-blend remover, the same way the Gemini sparkle
engine works (`src/remove_ai_watermarks/gemini_engine.py`).

## What we already know (verified from prior art, 2026-05-26)

- Blend model: **alpha compositing with a white logo** `watermarked = a*logo + (1-a)*original`,
  `logo = (255,255,255)`. Inversion: `original = (watermarked - a*logo) / (1-a)`.
  Confirmed by two independent sources (an open-source remover's algorithm doc + aiwatermarkremover.dev,
  both say "alpha map"). One commercial blog (pixelcleanai) claims "screen blend" instead; the gray
  capture below settles it empirically.
- Position: **bottom-right corner**, small margins (right ~8-20px, bottom ~5px), scales with image size.
  Confirmed by our sample `data/samples/doubao-1.png` (2048x2048) plus three sources.
- Size **scales with resolution**. Third-party numbers (~90x18 at <=1024, ~180x40 at >1024) are
  approximate and calibrated for ~1024-1280 outputs; at 2048 the strip is much larger. A shipped
  third-party alpha map is only 120x20, too small for our 2K/4K target -> capture fresh.
- The planning assumption was that clean inversion leaves residue on textured backgrounds, so the
  remover would pair the alpha map with inpainting. After the capture this turned out unnecessary at
  the native width (recovery is pixel-exact there and inpaint is off); the shipped remover is
  reverse-alpha only, with a residual inpaint applied off-native only.

## Use doubao.com specifically

The "豆包AI生成" mark is Doubao's. Jimeng / Dreamina use a different mark. Generate on doubao.com so
the captured template matches our target.

## How to capture (image-edit path, most reliable)

For each seed in `seeds/`:

1. Open Doubao image generation, use the image-edit / reference mode, upload the seed.
2. Prompt (Chinese preferred):
   `请完全按照原图重新生成这张图片，保持完全一致，不要添加或修改任何内容`
   (English: `Recreate this image exactly as it is, keep it identical, do not add or change anything`)
3. Download the ORIGINAL output file (not a screenshot). Do not crop / edit / re-save.

Prior art confirms uploading a pure-black image and letting Doubao stamp it works.

If edit mode is unavailable and text-to-image refuses a solid color, fall back to generating 10-12
normal-content images at one fixed resolution; the mark is the only constant across them and can be
extracted by per-pixel min/median.

## What to capture (priority top to bottom)

| Aspect | black | white | gray128 | why |
|--------|-------|-------|---------|-----|
| 1:1    | 3     | 1     | 1       | primary alpha map + confirm the stamp is pixel-identical across runs + settle blend mode |
| 16:9   | 2     | 1     | 1       | anchor rule in landscape |
| 9:16   | 2     | 1     | 1       | anchor rule in portrait |
| 4:3, 3:4 | 1 each | -   | -       | optional, refines anchor rule |

- 3 blacks on 1:1: if the first two are byte-identical in the watermark region, the third is optional.
- gray128 is the blend-mode test: predict the gray result from the black capture under alpha vs screen;
  whichever matches the real gray output is the true blend.
- If the UI offers multiple output resolutions (1K / 2K / 4K), capture one black per resolution on 1:1 -
  needed to learn how the watermark scales.
- Also grab 3-5 normal-content images on 1:1 for end-to-end removal validation.

## Hygiene

- Original download, never a screenshot. PNG preferred; if Doubao only gives JPEG, note it.
- No crop / edit / re-save. Default settings, watermark left ON.

## Naming, drop into `captures/`

```
doubao_black_1x1_1.png
doubao_white_1x1_1.png
doubao_gray128_1x1_1.png
doubao_black_16x9_1.png
doubao_content_1x1_1.png
```

## Also report back

1. Which resolutions and aspect ratios the Doubao UI actually offers.
2. Whether there is a watermark on/off toggle in the UI.
3. Download format (PNG or JPEG).
