# Photo classifier training and data

How the 2026-08-31 freeze was fit, what a retrain needs, and where those
files come from. Inference is [photo-classify.md](photo-classify.md). The
campaign log is [ai-generated-image-classifiers.md](ai-generated-image-classifiers.md).

The shipped package does not include the catalog or the embeddings. Fitting
the heads again uses the cache pack. The original pixels still have to be
kept: a missing embedding, a new backbone, or an audit all need the files.

## Retrain pack (caches)

CPU retrain of both heads needs sha256-keyed matrices and the catalog of
roles. That step does not open JPEG or PNG files. It is not permission to
drop the pixels.

| File | Role |
| --- | --- |
| `catalog.json` | 32,690 rows: `sha256`, `roles`, optional `family` / public `source` |
| `embeddings-by-sha256.npz` | CLIP-L-ft 768-d vectors for Model 1 |
| `extra-clip-l.npz` | CLIP-L extras for the 150 Flickr UI hashes |
| `features-124d.npz` | 124-d residual vectors for Model 2 |
| `probe-weights-clip-l-ft.npz` | Ridge mean, scale, weights, `thr_oi_1pct` |
| `meta-split.json` | Meta train 217, hold-out v2 79, hold-out v3 198 |

Run:

```bash
uv run python scripts/retrain_photo_classify.py --pack DIR --out DIR
```

Seeds are fixed: detector `20260940`, provider `20260956`. Two consecutive
runs must be byte-identical.

Not in the **published** retrain pack: raw photographs, COCO/Open Images/Kodak
binaries, CORD, Danbooru. Those files stay in their public corpora or in the
private freeze archive. Pixel rebuild of a missing embedding still needs
`clip-l-ft.pt` and the original file.

## Where the rows come from

Identity is always sha256 of the file bytes.

### Public, named corpora

| Role | Source |
| --- | --- |
| `detector_photo_train`, `detector_photo_coco_hold` | COCO 2017 validation stills |
| `detector_open_images_*`, `detector_photo_dev_oi` | Open Images photographs. Locked fresh never enters train |
| `detector_photo_kodak` | Kodak lossless true color, 24 files |
| `detector_photo_picsum_eval` | Picsum |
| `detector_flux_hf_hold` | Hugging Face FLUX stills, hold-out |
| Flickr UI cell (150) | Openverse `license_type=all-cc`, host `live.staticflickr.com` only |
| Meta hold-out v3 (198) | Meta Muse Image API, eval-only, not in train |

### In-house labeled photographs, not redistributed

Provider OpenAI, Google, and TC260 train and test rows, and most detector AI
train and test rows, are labeled photographic generations held privately.
The published pack carries only their sha256, roles, and precomputed vectors.
It does not carry the pixels.

### Out of the freeze train

CORD receipts, FUNSD forms, Danbooru, charts, and map tiles were measured as
failure domains. They are not Model 1 train. A future universal router treats
them as other domains, not as extra CLIP-L negatives.

## What the freeze script does

Historically `freeze_both_models.py` built the catalog from local paths, then
trained. That campaign code is private and is not in this repository. The
public entry is `scripts/retrain_photo_classify.py`, which skips catalog
construction and fits the same heads from the pack above.

1. Load CLIP-L-ft vectors by sha256. Fit the 768-512-128-1 MLP on
   `detector_ai_train` plus extras versus `detector_photo_train` plus struct.
   Threshold is the 1.67% quantile on `detector_photo_dev_oi`.
2. AND that MLP with the frozen ridge in the probe file. DEFINITELY is both.
3. Load 124-d vectors. Fit one-vs-rest focal heads for openai, google, tc260,
   meta_muse_image, no_ai. Margin 0.30. The public class for the Muse Image
   head is `muse-image`. `tc260` is the China AIGC label standard, mixed
   producers, not one company.
4. Repeat with the same seeds. Compare bytes.

CLIP-L fine-tune of the last two vision blocks is earlier, seed 20260822, and
is not repeated by the retrain script. The backbone file `clip-l-ft.pt` is an
inference and embedding dependency, not a retrain-pack input.

## Next retrain

Keep the current `tc260` head until then. The next provider-head fit should
split that catalog by `ContentProducer` (Doubao, Jimeng, Qwen, Kling, and
the rest) instead of one label-standard class. Do not treat a `tc260` score
as a named manufacturer.
