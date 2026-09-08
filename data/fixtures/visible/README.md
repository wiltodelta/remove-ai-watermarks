# Visible-mark example gallery

One committed example per registered visible mark, so the repository carries a
working sample of everything it supports. `tests/test_visible_examples.py` holds
both sides to it: a mark registered without an example fails the suite, and so
does an engine that stops detecting its own example.

## What these files are

Every example is SYNTHETIC: `scripts/render_visible_examples.py` composites the
mark's committed alpha or silhouette onto a deterministic generated base photo at
the engine's measured geometry. Controlled provider captures were used to solve
some alpha assets, including Samsung's measured opacity; the remaining shapes are
repository-owned renders rather than copied vendor rasters.

The examples demonstrate DETECTION geometry and house style, not vendor raster
fidelity; real-world variants (fonts, opacities, sizes) are covered by the
engines' recorded calibration evidence.

The Qwen, Kling, Yuanbao, Baidu, Jimeng, LiblibAI, Microsoft, Sora, Hailuo, and Veo
directories additionally contain real provider-marked sources. These files
retain the source bytes and names used here are deliberately distinct from the
generated canonical examples. The current LiblibAI output exercises the
metadata-corroborated compact-pill path, not the historical wordmark detector:

| File | Source | SHA-256 | Expected detector |
| --- | --- | --- | --- |
| `qwen/provider-original.png` | OpenNoMark `examples/qwen/image_324855086256596.png`, commit `9455d2d` | `dce9e17c9b9b0483c672b71183b6eb5562f8205bb39a6da14451e07db13b1a50` | Qwen, confidence 0.73 |
| `kling/provider-original-direct.png` | Authenticated Kling AI IMAGE 3.0 generation, downloaded 2026-09-04 | `aefbf69a99cf7eb4211cc0513c6979b81731107a1bc605c972e01900b7c56f51` | Kling, confidence 0.43 |
| `yuanbao/provider-original.png` | Authenticated Tencent Yuanbao AI Image generation, downloaded 2026-09-05 | `e4fd856565d0434ef06deb78ff2088e597c4fec98640ad737213c7c0a917c82d` | Yuanbao, confidence 0.49 |
| `baidu/provider-original.jpeg` | OpenNoMark `examples/baidu/baidu_01.jpeg`, commit `9455d2d` | `52852c528f1a7b8918b665b2fcf883737ecc5016481d9c7683ebe07e152cc9d2` | Baidu, confidence 0.66 |
| `jimeng_pill/provider-published-example.jpg` | JPEG embedded on page 29 of Zhonglun W&D's 2026 *Bank Finance Legal Newsletter* | `1182fba74d3753785345f8a35d09a8c9b52ea8383ebe9ee999d0d027437f03fd` | Jimeng AI pill, confidence 0.28; Jimeng wordmark, confidence 0.61 |
| `liblib/current-provider-output.png` | Authenticated LiblibAI Smart Image V2 generation, downloaded 2026-09-05 | `60398d4f06fd50e047d3321c62405dae0a1c4a7070e77ac5e0548c516257d2e3` | `liblib_pill`, confidence 0.216 with LiblibAI metadata corroboration; no `liblib` wordmark |
| `microsoft/provider-original.png` | Authenticated Microsoft Copilot generation, downloaded 2026-09-03 | `0e76fb8aa0540b18fee4d63ac4b5fd63c7eee310474e896d21d27e668403afe7` | Microsoft, confidence 0.54 |
| `sora/provider-frame.jpg` | Sora 2 video frame from LLinked's Apache-2.0 `sora-watermark-dataset` | `bbcf70b737c30568668572a5bb23e70c52c9687808b0714ca98f9fac073f12c4` | Sora frame localization, confidence 0.69 |
| `sora/provider-original.mp4` | LLinked's Apache-2.0 `SoraWatermarkCleaner` fixture `resources/puppies.mp4`, commit `ad35ce9` | `7dd8a9d43d43cd5b901946c7ba14e8018e18d47e75f0f774157327ba7e9181c5` | Sora, 300/300 stable frames; Sora C2PA claim |
| `hailuo/provider-original.mp4` | Authenticated Hailuo AI MiniMax H3 generation, downloaded with `With Watermark` selected on 2026-09-03 | `6152609e0339c443af79e244440e1b1050f3c615a50f76bfaf6146584e68142f` | Hailuo, 124/124 stable frames |
| `kling/provider-original.mp4` | Authenticated Kling AI VIDEO 3.0 generation, downloaded 2026-09-04 | `f3e8d6a765fca699036d68fde28eb16f7aed97b6702f06f318e61ae8d4e53a69` | Kling, 121/121 stable frames |
| `veo/provider-original.mp4` | Authenticated Google Flow Veo 3.1 Fast generation with `Visible watermarking` enabled, downloaded 2026-09-03 | `e778194b545e34dafd2d5be58691031a070862890c41e6c527a652a114a7c7a4` | Veo, 192/192 stable frames |

OpenNoMark describes its corpus as real-image examples and distributes the
repository under CC BY-NC-SA 4.0. Its license text is reproduced in
`../../licenses/open-no-mark-CC-BY-NC-SA-4.0.txt`. The upstream repository does
not record per-image authorship beyond that repository-level license and claim.
The Qwen and Baidu files remain under that license and are not offered under
this repository's Apache-2.0 license.

The Microsoft file is the original 1024 x 1024 PNG downloaded from an
authenticated Copilot image-generation conversation. It uses the neutral
still-life prompt recorded in `../README.md`, carries the visible top-right
`Made with AI` pill, valid Microsoft C2PA, and InvisMark soft-binding ID
`975290d4-9c2f-4359-ad5a-92c5b5b850a4`. Microsoft does not claim ownership of
AI-service output; the maintainer distributes this generated output under the
repository license.

The Sora frame is the original JPEG frame
`images/train/images/failed_image_000016_frame_000150.jpg` from LLinked's
`sora-watermark-dataset`. The dataset card describes its images as frames
extracted from Sora-generated videos, provides manual watermark boxes, and
licenses the dataset under Apache-2.0. This neutral CGI frame carries the
current moving mascot-and-wordmark raster and passes the in-tree frame
localizer.

The Sora MP4 is the byte-for-byte `resources/puppies.mp4` fixture tracked by
LLinked's Apache-2.0 `SoraWatermarkCleaner` at commit `ad35ce9`. It preserves
the moving mascot-and-wordmark raster, timing, audio, and the source C2PA
manifest. The in-tree video path detects Sora on all 300 frames. The C2PA claim
names Sora as the generator and OpenAI/Truepic as the issuer; its content
integrity and signature verify, while the reader reports the signer certificate
as expired and untrusted.

- Sora watermark dataset:
  <https://huggingface.co/datasets/LLinked/sora-watermark-dataset>
- SoraWatermarkCleaner source fixture:
  <https://github.com/linkedlist771/SoraWatermarkCleaner/blob/ad35ce99a34fc9f9f6d448537e0c348311e74ccd/resources/puppies.mp4>

The Hailuo file is the original 2944 x 1248, 5-second MP4 downloaded from an
authenticated Hailuo AI generation. It was generated with MiniMax H3 for 60
top-up credits from the neutral prompt `A blue ceramic mug slowly rotates on a
plain light gray studio table, soft even lighting, fixed camera, clean
background.` The download menu explicitly offered clean and `With Watermark`
variants; this fixture is the latter. It carries the visible lower-right
`MINIMAX | hailuo AI` composite and a TC260 AIGC label naming MiniMax as the
producer. The maintainer generated and explicitly cleared this output for use
under the repository license.

- Hailuo AI paid-service and credit terms:
  <https://hailuoai.video/doc/payment-policy.html>

The Kling video is the original 1280 x 720, 5-second MP4 generated with VIDEO
3.0 for 30 credits at 720p without native audio. Its neutral prompt was `A
locked-off wide shot of a small white paper boat floating across a calm
dark-blue pond at dusk, gentle ripples, realistic cinematic lighting, no text,
no logos.` It carries the fixed lower-right `KlingAI 3.0` logo and no locally
readable provenance metadata. The maintainer generated and explicitly cleared
the output for this noncommercial deterministic computer-vision regression
fixture. Kling AI's terms say the user owns applicable rights in the output,
require a regular account to retain Kling branding when distributing it, and
prohibit commercial use without written permission. The fixture retains the
brand and remains subject to those terms; it is not offered under the
repository license.

- Kling AI terms of service: <https://kling.ai/docs/user-policy>
- Kling AI paid-service terms: <https://kling.ai/docs/payment-policy>

The direct Kling image is the original 1024 x 1024 RGB PNG generated with
IMAGE 3.0 in 1K SD mode for one credit. Its neutral prompt was `A documentary
photograph of three smooth river stones on pale linen beside a small blue glass
vase, soft natural daylight, plain uncluttered background, no text, no logos.`
It carries the current lower-right `KlingAI 3.0` mark and a TC260 AIGC field
whose producer code `001191110108335469089C10100` normalizes to Kuaishou's
registered USCC. The maintainer generated and explicitly cleared the output for
this noncommercial deterministic computer-vision regression fixture. The same
Kling terms and retained-brand restriction listed for the video apply; the file
is not offered under the repository license.

The Yuanbao image is the original 1776 x 1328 RGB PNG returned by an
authenticated AI Image conversation. It was generated at no charge from the
neutral prompt `A documentary photograph of a blue ceramic mug on a plain
light-gray studio table, soft even lighting, no text, no logos, no people.` It
carries the visible lower-right two-line `元宝` / `AI生成` mark and a TC260 AIGC
field naming Tencent's registered producer. The maintainer generated and
explicitly cleared this output for use under the repository license.

The Jimeng image is the byte-for-byte embedded JPEG extracted with
`pdfimages` from page 29 of Zhonglun W&D's publicly distributed 2026 *Bank
Finance Legal Newsletter*. It is a real Jimeng-generated debt-collection
poster carrying both the top-left `AI生成` pill and the lower-right `即梦AI`
wordmark. The repository retains one 705 x 705 image rather than the complete
33-page publication, solely to test identification and removal of the two
marks. Copyright remains with the source rights holders; this research fixture
is not licensed under Apache-2.0. See
`../../licenses/zhonglunwende-jimeng-fair-use.txt` for provenance and use
limitations.

- Zhonglun W&D source publication:
  <https://www.zhonglunwende.com/uploads/20260202/97f9a8f64892eac6b7e04e5e7ea45d0e.pdf>

The LiblibAI image is the original 2048 x 2048 RGBA PNG generated with Smart
Image V2 for 18 introductory points from the same neutral prompt. It carries a
top-left `AI生成` pill and a TC260 AIGC field whose producer code contains
LiblibAI's registered USCC, but it does not carry the historical bottom-center
`LiblibAI` wordmark that the `liblib` engine detects. It is the real-provider
regression for the metadata-corroborated `liblib_pill` path. The maintainer
generated and explicitly cleared this output for use under the repository
license.

- LiblibAI user agreement (generated-output ownership and watermark choices):
  <https://www.liblib.art/activities/468ad794ccc7408d81757fd91be003ec?hideHeader=1>

The Veo file is the original 1280 x 720, 8-second MP4 downloaded from an
authenticated Google Flow project. It was generated with Veo 3.1 Fast for 20
Flow credits from the neutral prompt `A blue mug rotates on gray. Static camera.
No people, text, logos, or audio.` after the account's `Visible watermarking`
setting was enabled. It carries the fixed, low-contrast `Veo` text at the bottom
right, valid Google C2PA, and an invisible SynthID marker. Google documents that
Flow clips can be downloaded and shared, states that Google does not claim
ownership of original generated content, and explicitly identifies SynthID in
all Flow video output. The maintainer distributes this generated output under
the repository license.

- Google Flow output and watermark documentation:
  <https://support.google.com/flow/answer/16353333>
- Google Flow download and sharing documentation:
  <https://support.google.com/flow/answer/16935308>

## Regeneration

    uv run python scripts/render_visible_examples.py

The generator self-verifies: it fails (exit 1) if any registered mark does not
detect on its own example, so regeneration is the fix point for drift.

## Layout

    <mark-key>/example.png   1536x2048..2048x2048 PNG, one per image mark
    <mark-key>/provider-original.*
                             Documented real provider-marked source where available
    <mark-key>/example.mp4   960x540 90-frame clip, one per video mark
                            (kling carries both: it is registered in both registries)

Special cases: `gemini` composites the sparkle alpha map at the provider's
configured position; `jimeng_pill` uses the measured 3:4 portrait geometry;
`liblib_pill` uses the shorter compact silhouette and requires product evidence;
`microsoft` is the opaque white pill with dark text
holes (the discriminator its detector keys on). Video examples composite the
detector's own synthetic template on every frame; where two marks share a
shape family the example carries the discriminative variant (`veo` the legacy
text form, `kling` the logo-plus-wordmark pair flush to the edge), because the
temporal selection resolves cross-template ties by table order.
