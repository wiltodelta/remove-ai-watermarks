# SynthID oracle fixtures

A compact set of externally verified originals for two uses:

1. **Signal and detector-limit research** with exact native resolutions and
   recorded verification sources.
2. **Removal regression set**: verify that a pipeline turns a SynthID-positive
   image into a negative one.

There is no reliable local detector of the SynthID pixel watermark (Google's
decoder is proprietary). The ground-truth label therefore comes from an
external oracle, recorded per image in `verified_via` (see below).

## Layout

```
data/synthid/
  README.md                  # collection and verification protocol
  manifest.csv               # one row per committed original
  full-pipeline-quality.csv  # reusable evaluation selection
  originals/                 # one canonical copy of each verified source
```

The originals are committed so the evaluation set is reproducible and its
hashes can be checked in CI. `manifest.csv` is kept in sync with
`originals/`, one row per file. Generated outputs do not belong here. Store
them outside the repository and record only a reproducible verdict in the
evaluation manifest or documentation.

## Reusable removal-quality set

`full-pipeline-quality.csv` is the canonical reusable input set for
full-pipeline visual-quality and watermark-removal tests. Its fixtures are
cleared for permanent public test reuse. Tests, benchmarks, and manual
evaluation bundles may read these images repeatedly without requesting new
permission.

The CSV preserves each platform's original filename and maps it to the one
canonical corpus copy. Use `corpus_path` to read the input, but preserve
`source_filename` in generated result names and keep OpenAI and Gemini outputs
in separate provider groups. This avoids duplicating large PNGs while retaining
the names needed for the corresponding provider oracle.

Rows with a dated negative `final_clean_oracle` were checked after the complete
`visible -> qwen-zimage -> metadata` route with the matching provider oracle.
`not-checked-in-final-candidate` means the input is available as an additional
quality stress fixture, but a newly generated output still needs the matching
oracle before claiming removal. The checked result applies to those exact
output bytes; re-run the oracle after changing the pipeline, seed, model
versions, or runtime settings.

## Verification levels (`verified_via`)

Ground-truth quality, strongest first:

- `gemini-app` — checked via the Gemini app "Verify with SynthID" feature. Gold standard for the pixel watermark (Google models).
- `openai-verify` — checked through the OpenAI web verifier. This is the
  historical manifest label for existing rows. OpenAI now also documents a
  Content Provenance API with a separate SynthID result; add a distinct manifest
  value before recording API-derived labels rather than silently reusing this one.
- `synthid-portal` — checked via Google's SynthID Detector portal.
- `c2pa-metadata` — supported provenance evidence (Google AI C2PA, or OpenAI
  C2PA with an explicit `c2pa.watermarked.*` action). Weaker than a provider
  oracle: C2PA can be stripped while the pixel watermark remains.
- `third-party` — label asserted by an external dataset, not independently verified.
- `none` — unverified.

Prefer `gemini-app` for any Google image that will gate a test.

## What to collect

For the **regression set**:

- A small number of positive originals with a recorded provider oracle.
- A quality-set row for every original used by repeatable evaluation.
- Re-run the provider oracle on each newly generated output. Do not commit the
  generated output as another corpus copy.

The corpus is committed to a public repo: review every image before adding it
and keep out anything private or identifiable you would not publish.

## Ingesting

Use `scripts/synthid_corpus.py` to copy a file into `originals/`, record its
SHA-256, resolution, format, and C2PA issuer through our detector, and append a
row to `manifest.csv`:

```bash
uv run python scripts/synthid_corpus.py ingest path/to/*.png \
    --label pos --source "Gemini app" --model gemini-3-pro \
    --verified-via gemini-app --notes "1024x1024 batch"

uv run python scripts/synthid_corpus.py status   # counts by label / resolution / verification
```

## Autonomous collection via Chrome MCP

> Historical collection notes. Browser interfaces and download behavior can
> change independently of this repository. These steps do not define package
> behavior.

Generation can be driven through the browser (the account must be logged in):

- **Gemini** (`gemini.google.com`): type `Create an image: <prompt>`, wait, hover the
  result, click the download icon (top-right). Single, reliable click. Outputs
  carry Google C2PA + SynthID. Occasionally the composer stalls in a
  "generating" state -> start a New chat to reset.
- **ChatGPT** (`chatgpt.com`): the UI download is flaky (the fullscreen viewer
  races and can grab the previous image; the share-modal path works but is
  multi-step). Reliable path is an in-page fetch of the rendered image, which
  preserves the original bytes (C2PA intact, unlike a canvas re-encode):

  ```js
  // run in the ChatGPT tab via the browser MCP javascript tool
  (async () => {
    const imgs = [...document.querySelectorAll('img')].filter(i => i.naturalWidth >= 400);
    const img = imgs[imgs.length - 1];                 // newest large image
    const b = await (await fetch(img.currentSrc || img.src)).blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(b); a.download = 'dl.png';
    document.body.appendChild(a); a.click(); a.remove();
    return 'size=' + b.size;                            // do NOT return the src (privacy guard blocks query strings)
  })()
  ```

  Gotcha: confirm the returned `size` differs from the previous image before
  ingesting -- if the new image has not finished rendering, the script grabs the
  prior one (the corpus dedups by sha256, but the notes would mislabel it).
  ChatGPT also shows an A/B "which is better?" picker; click Skip first.

**Originals, not previews.** Some platforms render a low-res preview in the chat
(Grok serves a ~20KB 1024px JPEG/PNG; the in-page `<img>` fetch grabs *that*, not
the original). Previews are re-encoded and **strip metadata**, so a "clean"
preview is not proof the original is clean. Always pull the original via the
platform's native Download / lightbox button and sanity-check the file size (a
20KB "1024x1024" is a preview). ChatGPT's in-chat `<img>` *is* the full-res
oaiusercontent original, so fetch+blob is fine there; Grok needs the lightbox
Download; Leonardo serves the original JPEG in-chat (download button matches).

## Per-platform watermark map (observed, May 2026)

This is a dated observation table, not a live provider compatibility promise.
The detector's coverage is complementary: metadata catches C2PA /
IPTC; `exif_generator` catches EXIF `Make`/`Software` + XMP `CreatorTool`;
`invisible_watermark.py` (imwatermark) catches the open SD/SDXL/FLUX DWT-DCT
watermark on pristine files; the visible detector catches the Gemini-family
sparkle; the SynthID *pixel* itself has no local detector (oracle only).

| Platform | C2PA issuer | SynthID pixel | IPTC "Made with AI" | Visible sparkle | imwatermark | Corpus label |
|---|---|---|---|---|---|---|
| Gemini app | Google | yes | - | yes | - | pos |
| ChatGPT / gpt-image | OpenAI | yes | - | - | - | pos |
| Microsoft Designer | OpenAI + Microsoft | yes (via OpenAI) | - | - | - | pos |
| Bing Image Creator | Microsoft (MAI-Image) | no | - | - | - | pos (C2PA "Microsoft", not OpenAI) |
| Google AI Studio (Nano Banana) | **none** | yes (oracle-confirmed) | - | yes | - | pos (metadata blind spot) |
| Stability AI (Brand Studio) | Stability AI Ltd | no | - | - | no | pos (C2PA only) |
| Ideogram | none | no | - | - | no | pos (EXIF `Make="Ideogram AI"` only) |
| Meta AI | none | no | **yes** | - | - | neg (for SynthID) |
| Leonardo.ai | none | no | no | - | no | neg |
| Recraft | none (export strips) | no | no | - | no | neg (re-encoded export, no signal) |
| Krea (FLUX 2 host) | none | no | no | - | no | neg (host omits the imwatermark encoder) |
| Grok (xAI) | none (non-adopter) | no | no | - | no | neg (captured: clean low-res preview) |

Key takeaways:
- The same model differs by *surface*: Gemini app wraps C2PA, AI Studio (API/playground) emits none -- only the pixel + sparkle survive.
- Microsoft Designer's DALL·E backend inherits OpenAI's C2PA+SynthID (issuer "OpenAI, Microsoft"); Bing now runs Microsoft's own **MAI-Image** and signs C2PA as "Microsoft" (not OpenAI/DALL·E).
- Meta uses the IPTC `digitalSourceType` marker, not C2PA or SynthID.
- The open imwatermark fires only on *pristine* output from a pipeline that runs the encoder (diffusers default, official BFL) -- not from re-hosts (Krea, Stability hosted SDXL) or re-encoded design exports (Recraft, Canva). Ideogram's only signal is the EXIF `Make` tag.
- Bing and Grok web UIs are uncooperative for autonomous capture (no document_idle for screenshots; blob downloads intermittently no-op; low-res in-chat previews). Use their native download button manually if a full-res sample is needed.
