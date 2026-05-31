# Gemini (Nano Banana) visible sparkle capture

> **Status (captured 2026-05-31):** black/gray/white captures taken and **committed**
> in `captures/` (solid colour + the sparkle, content-free). The sparkle-on-black assets
> `gemini_bg_{96,48}.png` are rebuilt by `scripts/visible_alpha_solve.py gemini`.

Google Gemini (Nano Banana) stamps a four-point sparkle icon in the bottom-right corner
via alpha compositing: `watermarked = a*logo + (1-a)*original`. Unlike the Doubao/Jimeng
text marks, the sparkle is captured over a **pure-black** background, where
`watermarked = a*255` (logo is near-white), so the alpha reads directly off the capture
(`alpha = max(R,G,B)/255`) -- no background fit needed. This is the "golden" capture case.

## What the captures confirmed (2026-05-31)

- The sparkle at a 2048-wide image is **96x96 px** (width_frac ~0.047), bottom-right,
  margins ~0.031, alpha max ~0.51 -- matching the engine's existing 96px asset.
- Our own controlled capture matches the previously third-party-sourced
  `gemini_bg_96.png` to **NCC 0.9998**, so the bundled asset is validated and now
  reproducible from our own capture.

## How to capture (image-edit path)

For each solid-colour seed (`seeds/seed_{black,gray,white}_2048.png`, gitignored):

1. Open Gemini image generation, image-edit / reference mode, upload the seed.
2. Prompt: `Recreate this image exactly as it is, keep it identical, do not add or change anything`
3. Download the ORIGINAL output (not a screenshot). Do not crop / edit / re-save.

Black is the key one (sparkle on black -> exact alpha). Gray/white cross-check.

## Naming, drop into `captures/`

```
gemini_black_2048.png   # the key capture (sparkle on black)
gemini_gray_2048.png
gemini_white_2048.png
```

The solid captures are **committed** (content-free). The synthetic `seeds/` and any
real-content `gemini_content_*.png` validation download are gitignored (local-only).
Rebuild the assets with:

```
uv run python scripts/visible_alpha_solve.py gemini   # or: all
```
