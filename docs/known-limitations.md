# Known limitations

This page describes current product limits. Historical measurements and
superseded experiments live in the research archive listed in
[the documentation index](index.md).

## Pixel classification

`classify` / `classify_pixels` is AI versus camera, not a universal AI detector.
Receipts, UI screenshots, and digital art are out of contract: a thermal receipt
can score as AI, a photorealistic AI receipt as human. POSSIBLY is reported as
`unknown`. Provider attribution runs only after DEFINITELY and can abstain when
124-d extraction refuses the file. Microsoft is not a pixel class.

The command is never started by `identify`. A no-signal provenance result stays
unknown. This is not a clean verdict and it does not run cleanup. Full guide:
[photo pixel classification](photo-classify.md).

## Visible removal

### Fill quality depends on the background

Visible removal changes only the selected mask, but the hidden pixels still
have to be reconstructed.

- OpenCV is fast and requires no model download. It works well on flat
  backgrounds but
  can smear texture or repeated structure.
- MI-GAN is a lighter learned backend. It can improve natural texture but may
  ghost or invent structure.
- LaMa is the heaviest learned backend and is generally the strongest option
  for difficult backgrounds.

`--backend auto` selects LaMa when available, then MI-GAN, then OpenCV.

No backend can recover detail that is completely hidden by an opaque mark. A
successful detection therefore does not guarantee a visually perfect fill.

### Automatic detection covers registered variants only

The registry contains vendor and locale specific templates. A redesigned mark,
an unsupported locale, a different position, or a crop may be missed.

Known examples:

- The Microsoft detector covers one calibrated top-right white pill. Microsoft's
  [documented feature](https://support.microsoft.com/en-us/topic/include-a-watermark-when-content-from-microsoft-365-is-ai-generated-b00a656e-ae61-4692-8086-67d004421030)
  can instead use a Copilot icon, `AI-Generated` text, or another position.
- Samsung detection is calibrated for the Italian
  `Contenuti generati dall'AI` text variant.
- The Jimeng top-left pill has a weak visual detector and is intentionally
  subject to additional product and background checks.
- Kling AI support covers the calibrated variants rather than every Kling AI label.

Use `erase --region` when you can see and select an unsupported or missed mark.

### Strict and automatic sensitivity trade recall for precision

`--sensitivity strict` uses the visual gate alone. The default `auto` mode may
relax a mark only when metadata or a confidently detected sibling mark
corroborates the same product.

There is no blanket "this image is AI" relaxation. That information does not
identify the vendor, mark, or location and caused unacceptable false
detections in the removed experimental mode.

## Invisible removal

### Regeneration is lossy

Invisible removal does not decode and delete a payload. It regenerates the
image through a diffusion pipeline. Faces, text, colors, and fine detail can
change even when the watermark is successfully disrupted.

Text, tables, and UI screenshots are the worst case for a forced scrub: the
model redraws glyphs as plausible-but-wrong shapes. Do not run the invisible
stage on such content unless a locally-detectable invisible signal exists.
The default `all`, `invisible`, and `batch` commands already skip it then, and
`visible` (which strips AI metadata by default) or `metadata` alone never
redraws a glyph outside the filled watermark box. `--force` is for content you
know carries a pixel watermark despite no local signal; on a clean screenshot
it only buys distortion. Preserving or pasting back the original text pixels
during a real scrub is deliberately not offered: an invisible watermark such
as SynthID lives in the pixels everywhere, including inside the glyphs, so
frozen text regions would keep the watermark
([text protection research](text-protection-research.md)).

`qwen-zimage` is the default profile. `sdxl-zimage` and `chroma-zimage` keep
the same face stage and swap the global model. `auto` picks chroma-zimage for
OpenAI and Microsoft provenance and qwen-zimage otherwise. All are CUDA only.
Qwen and SDXL condition the global stage on a canny edge map, which preserves
structure but not identity or exact texture. Chroma1 is a plain strength pass
without Canny.
Existing face evaluations favor `qwen-zimage`, but there is no blanket fidelity
ordering across content types. A fixed-seed, three-scene text comparison at the
profile defaults found no stable winner: SDXL won one poster, Qwen won one, and
the Chinese sign tied. Both are large, slow, and may still alter small text or
difficult faces. The measurements and their OCR and oracle caveats are tracked
in [`data/evaluations/fidelity/`](../data/evaluations/fidelity/README.md).
A global Z-Image Turbo prototype preserved text substantially better at low
strength, but it has no useful cross-provider operating point and is not a
supported profile. Automatic text restorers also remain research-only:
fresh-font and silhouette variants visibly changed typography. The higher-fidelity
`vae-glyphs` route is available only as an experimental opt-in with verified strings
and line geometry. It builds its donor internally but still requires an independently
clean global anchor. Automatic OCR and line-box proposals are not reliable enough to
remove those requirements, and exact oracle results do not establish a general mask,
seed, runtime, or provider operating range. Qwen-Image-2.0 is hosted-only and exposes
no equivalent low-strength denoise control. Exact experiments, controls, and pass
rates are kept in
[`text-protection-research.md`](text-protection-research.md) and the
[`fidelity` evaluation record](../data/evaluations/fidelity/README.md).

### There is no local SynthID pixel detector in the package

Google does not publish the proprietary SynthID payload decoder, and the
package does not ship one. Signed provenance is the supported route:
Google AI C2PA or current OpenAI C2PA with an explicit watermark action.
`verify-openai-synthid` is the official remote pixel check for OpenAI.
Research on a periodic lattice expert is in
[synthid-detector-research.md](synthid-detector-research.md).

For important outputs:

1. preserve the original;
2. process a copy;
3. verify with the matching provider tool when available;
4. do not assume one provider's verifier covers another provider's payload.

Provider systems can change, so a result verified on one file, seed, or version
is not a permanent certification.

Meta Content Seal (Muse Image) is in the same family: no local decoder, presence
recognizable only through the XMP `trainedAlgorithmicMedia` companion tag while
that metadata survives, and removal verifiable only through the anonymous
`meta.ai/identification` oracle, which rate-limits by IP per day. The oracle's
negative is weaker than its positive: Reuters measured it missing 55% of cropped
Muse images, so treat a negative on a cropped or heavily edited file as
inconclusive rather than clean.

### Video SynthID removal is lossy and content-dependent

The `video invisible` command and `remove_video_invisible` API regenerate video
pixels through a VAE. The shipped `noise_std=0.15` profile is oracle-certified,
but Google does not publish a local decoder for arbitrary runtime outputs. A
quiet metadata scan, paired PSNR, and the temporal-residual metric are fidelity
measurements, not independent SynthID verdicts.

The control must use the same clip, frame rate, dimensions, and final codec as
the candidates. The separate `scripts/video_synthid_sweep.py` harness produces
that matched control. If the control is not detected by the matching provider
oracle, the experiment cannot attribute a quiet candidate to regeneration.

The 2026-07-29 two-clip calibration used Gemini's built-in content verifier:
both matched controls were SynthID-positive, the stronger candidate was
negative on both carriers, and a weaker candidate was negative on one. A
2026-07-30 adversarial follow-up incorrectly asked the ordinary chat model to
reinterpret the verifier while excluding every other input; its `UNAVAILABLE`
answer was not another detector run and does not invalidate the original
built-in results. The calibrated default remains content-dependent; a fresh
source-positive, output-negative pair is an optional audit for unusually
important files or after provider changes.

The 2026-07-31 full-clip check used the public eight-second Veo off-road sample.
The source was detected across the full clip, the complete product path at
`noise_std=0.10` remained detected, and `0.15` returned no SynthID detection.
The default was raised to `0.15`. At 512 px / 12 fps, the accepted candidate
measured 25.39 dB paired PSNR and a 1.058 motion-compensated temporal-residual
ratio. This is one carrier, not a universal guarantee; hashes and exact verdicts
are tracked in `data/evaluations/video-synthid-oracle.csv`.

### The invisible video path does not touch the audio track

`remove_video_invisible` regenerates the video stream and copies the source audio
verbatim: the extracted audio bitstream of input and output has an identical
sha256, measured on two carriers. It also strips every metadata marker, so the
output can report clean from `get_ai_metadata` and `identify_video` while carrying
an untouched generated audio track. Google's verifier scores audio and visual
tracks separately, and a track this path never modifies is a track it cannot have
cleaned.

Whether a given carrier's audio actually holds a mark the verifier reads has not
been established -- that needs a provider verdict, which has not been obtained.
Treat a clean local report on a clip with generated audio as unproven, not as a
guarantee, and check the audio separately when it matters.

The shipped engine streams sampled frames in bounded batches, computes its
fidelity metrics incrementally, and pipes regenerated pixels directly to
ffmpeg. Its frame and latent memory is therefore bounded by `--batch-size`
rather than clip duration. Runtime still grows linearly with duration, and the
separate multi-candidate research sweep deliberately retains its short sampled
prefix so it can reuse identical latents across candidate strengths.

The reported `psnr_db` is measured against the already-resized frame and before
the H.264 encode, so it excludes the downscale, the frame decimation, and the
encoder. It measures the VAE round trip plus latent noise at the working geometry.
[`video-synthid-quality-research.md`](video-synthid-quality-research.md) records
what the manifest rows constrain, what a higher-resolution or higher-frame-rate
profile would need in order to be certified, and the audio-track question this
path has not yet answered.

### Strength is content and seed dependent

The profiles resolve an unset strength differently, because different things
were measured for each.

`qwen-zimage` reads unknown content from the resolution-adaptive denoise curve.
Measured provider cohorts instead take flat operating points: OpenAI `0.07675`,
Google `0.27`, and Microsoft InvisMark `0.15`. OpenAI and Microsoft add one full
observed cross-source boundary spread to the worst clean source; Microsoft's three
first-clean boundaries were `0.04125`, `0.055`, and `0.095`, giving `0.14875` before
rounding up. Google's candidate was separately repeated across three valid sources
and three accounts. The small corpora make these operating points, not universal
thresholds.

`sdxl-zimage` reads it from the C2PA issuer, on a flat ladder:

- OpenAI: `0.15`;
- Google: `0.25`;
- unknown: `0.25`, following the stricter of the two.

An SDXL global pass needs more denoise than Qwen at the same fidelity, and the
values are flat rather than a curve because flat values are what was measured: each
verdict came from a fixed strength at one size, and no size dependence has been
established for that stage.

`chroma-zimage` also reads a flat ladder from the provenance cohort, measured
2026-08-29/30 across the four vendor oracles (full record:
[`chroma1-engine-research.md`](chroma1-engine-research.md)):

- OpenAI: `0.09` (below qwen's `0.07675` is false economy neither way -- the
  matched-strength fidelity favors Chroma1 on this cohort);
- Microsoft InvisMark: `0.125` (below qwen's `0.15`);
- Google: `0.40` for zero-face content (above qwen's `0.27`; at this floor the
  regeneration destroys dense text and collapses face identity -- the tradeoff
  that motivates the adaptive arm below), or `0.125` when YuNet detects at
  least one face (both measured face fixtures clear at 0.12 with zero
  cross-source spread; oracle-verified 2026-08-30);
- Meta Content Seal: `0.17` (above qwen's `0.1`);
- unknown: `0.40`, following the strictest measured cohort.

An explicit `--strength` overrides all three. The defaults are operating points,
not universal guarantees. Near a removal threshold, different content or a
different random seed may change the verifier result, which is why the profiles
are certified at a fixed seed. The live resolver is
[`watermark_profiles.py`](../src/remove_ai_watermarks/_internal/watermark_profiles.py).

### Pipelines have different quality tradeoffs

| Pipeline | Main limit |
| --- | --- |
| `qwen-zimage` | CUDA only, large model stack, and limited broad certification across seeds and content. |
| `sdxl-zimage` | CUDA only. Its strength ladder is flat per vendor, not a resolution curve, because flat values are what was measured. |
| `chroma-zimage` | CUDA only. Higher Google and Meta floors than qwen; Google face content uses 0.125 instead of 0.40. |
| `auto` | CUDA only. Routes OpenAI and Microsoft to chroma-zimage and everything else to qwen-zimage. |

Only manually verified `vae-glyphs` is an optional production stage, and it is
experimental rather than a default.
OCR plus LaMa recovered literal poster text but changed fonts and worsened whole-image
fidelity. Restricting it to OCR-mismatched lines improved the tradeoff but still
left a local shadow on one poster. The published AnyText2 SD1.5 checkpoint
substituted Chinese characters and increased CER on the sign fixture. AnyText2XL
is not published, and the released wrapper truncates individual text lines after
20 characters.

The `controlnet`, `sdxl`, `qwen` and `default` profiles were removed, not aliased
onward: a retired name is rejected at parse time rather than routed into a profile
the caller never chose. There is no `--model`, `--steps`, `--guidance-scale`,
`--device` or `--auto` option either; each profile pins its model stack, its
per-stage schedule, CFG 1.0 and CUDA.

## Resolution and memory

### Small images are processed at their native size

There is no minimum-resolution floor. It existed to enlarge small inputs toward
SDXL's ~1024 training resolution and was removed with the SDXL profiles, which
never applied it anyway. Both surviving profiles run at native geometry, so a
small input is neither enlarged before diffusion nor restored afterward.

`--max-resolution` still caps very large inputs, and only ever scales down.

### Large images stay at native resolution unless capped

`--max-resolution 0` means no explicit downscale cap. A positive value caps the
long side before diffusion and restores the result afterward. This reduces
memory use but introduces a downscale and upscale round trip.

`--tile` preserves the input dimensions while running the diffusion stage in
overlapping tiles. It avoids the explicit downscale, but it is not pixel
lossless: each tile is independently regenerated. With `qwen-zimage`, only the
global Qwen stage is tiled; the face stage runs after tile blending.

### CPU offload trades speed for VRAM

`--cpu-offload` forces both stacks of the two-stage profile out of automatic
device residency, streaming weights instead of pinning them. It reduces CUDA
memory pressure at the cost of speed.

Residency is otherwise chosen from the card's total VRAM. On a card large enough
to hold a stack, offloading is pure waste: DiffSynth drops weights to the meta
device and re-reads every parameter from disk on each stage transition.

### There is no CPU or MPS fallback

Invisible-watermark removal refuses any device but CUDA at construction. There is
no MPS out-of-memory fallback that continues on CPU, and no lighter profile to
drop to: all three profiles need an NVIDIA GPU.

When a run does not fit, the levers are `--tile` (native geometry, tiled
diffusion), `--max-resolution` (an explicit downscale), and `--cpu-offload`.
Memory needs depend on the profile, input size, dtype, and card.

## Metadata and formats

### Missing metadata does not mean clean

Screenshots, social platforms, and re-encoding can remove metadata while a
pixel watermark remains. `identify` therefore reports unknown rather than
clean when no supported signal is found.

### JPEG XL is metadata only

The metadata path recognizes JPEG XL containers, but the visible and diffusion
image paths do not list `.jxl` as a supported pixel format because the package
does not include a JPEG XL pixel decoder.

### HEIC, HEIF, and AVIF pixel decoding uses an optional Pillow fallback

OpenCV does not decode these formats in the project. `image_io.imread` falls
back to Pillow with `pillow-heif` when the `heif` extra is installed alongside
a pixel feature. The
default metadata path scans these containers without that plugin. A corrupt or
truncated file may still fail to decode.

### Some metadata removal requires ffmpeg

WebM, Matroska, MP3, WAV, FLAC, OGG, Opus, and AAC container metadata is stripped
through ffmpeg with stream copying. The operation fails if ffmpeg is absent or
cannot parse the input.

### Video pixel removal is provider-specific

The `video metadata` command and high level video API inspect and strip
supported AI provenance metadata without transcoding streams.

`video visible` and `remove_video_visible` additionally support the moving
Sora 2 mascot and wordmark, the current Veo four-point diamond, the legacy
`Veo` text, the Seedance boxed `AI` label, the fixed `Dola AI` text, the Hailuo AI
MINIMAX/Hailuo AI composite label, and the bottom-right Kling AI label with its
version suffix. Detection requires a recurring visual candidate across
adjacent frames. Fixed-mark candidates must remain anchored rather than
drifting with a scene object. Kling AI also requires a bright low-saturation
candidate near the expected frame edge. Provider provenance can recover
low-contrast runs only after visual evidence exists for the marks that define a
provenance prior, so metadata alone does not erase a clean API export.
The default auto-router evaluates all detectors in one decode pass but does not
rank their raw confidence values. Those scores are provider-specific and known
to cross-match in some layouts, so the router applies the independent temporal
policies and selects the first stable result in specificity order. Use an
explicit mark when the provider is already known.
Historical Sora Turbo exports use a small OpenAI swirl in the corner rather
than the moving mascot-and-wordmark design; that earlier variant is not
detected by the `sora` video mark. Hailuo AI and Kling AI coverage is specific to the
verified lower-edge layouts; a new provider layout needs a separate calibrated
silhouette. Other provider video labels are not supported yet. Google video
SynthID has an oracle-certified VAE removal path, while other proprietary
invisible video watermarks have no registered attack.

Visible removal transcodes the video stream and copies the complete audio
stream without shortening an audio tail. Completed visible and invisible
encodes are published atomically, so an encode failure preserves an existing
output. Visible removal now applies a guarded motion-compensated blend after
the per-frame fill. It uses adjacent optical flow only when the warped prior
mask covers the current mask and a source-context ring agrees; scene cuts and
disjoint marks keep the independent fill. This reduces measured paired
temporal error, but it is not a generative video-inpainting model and cannot
recover structure that no frame exposes. OpenCV can still leave a visible
smear where the mark overlaps a hard edge or structured texture. MI-GAN
improves difficult individual frames and is the practical learned CPU tier.
LaMa remains an offline quality option: a full real sequence confirmed its
multi-GB memory use and CPU throughput unsuitable for an online worker. The Veo diamond
uses a shape mask to limit damage outside the symbol. Seedance fills the full
localized box because a synthetic outline mask left part of the real
translucent border visible in an end-to-end check. OpenCV may therefore soften
texture inside that small box; use MI-GAN or LaMa when reconstruction quality
matters. Relative variable-frame timestamps are preserved through a timestamped
NUT bridge. A non-zero absolute video start PTS and the corresponding copied
audio offset are preserved through ffmpeg timestamp passthrough.
The encoder is source-aware for common 8-bit inputs: it probes and preserves
supported chroma sampling, recognized color metadata, encoder time base, and
MP4/MOV track timescale instead of accepting ffmpeg's implicit `yuv444p`
raw-BGR output. OpenCV still decodes through 8-bit BGR, so HDR/high-bit-depth
inputs are rejected before encoding rather than silently falling back to
`yuv420p`. The synthetic
Sora/OpenCV full-clip CI gate covers complete removal, untouched-region PSNR,
frame count, frame rate, duration, copied-audio identity, source stream
properties, paired temporal deltas against an independently encoded
frame-local baseline inside the filled region, and metadata
stripping through real ffmpeg. A second synthetic VFR clip verifies all display
timestamps to the source time-base tick, including a non-zero source start,
and checks both video and audio stream offsets. A separate constant-rate case
guards the non-zero-start routing independently of VFR detection. A local
full-sequence audit over all six providers passed both OpenCV and MI-GAN for
complete-frame removal, quiet second detection, stream starts, duration, and
copied audio. A full LaMa sequence passed the same checks but established that
the backend belongs in the offline tier on CPU. These bounded local checks are
still not universal evidence for every provider layout or source.

Native TC260 metadata in MP4/MOV is supported at its normative
`moov.udta.meta.keys/ilst` placement, including non-faststart files whose
`moov` follows a large media payload. MKV/WebM is supported at the normative
`Segment.Tags.Tag.SimpleTag` placement and uses ffmpeg for stream-copy removal.
AVI is supported at `LIST/INFO/AIGC`, and FLV at
`script.onMetaData.AIGC`; both use ffmpeg stream-copy removal. Stock ffmpeg can
write and strip the FLV form, but writing the nonstandard AVI child for fixture
generation requires dedicated muxer support, so the AVI reader is verified
against an exact synthetic RIFF structure.

MP4/MOV/M4V metadata removal now stream-copies the container in bounded chunks,
keeps every box size and media offset fixed, and publishes atomically. The
large-`mdat` regression rejects a full-source `read_bytes()` call and verifies
that the encoded payload is byte-identical. HEIF/AVIF/JPEG-XL image metadata
still uses the in-memory path because their XMP/EXIF items may live inside
`mdat`/`idat` and require bounded format-item parsing before that path can
stream safely.

### Metadata transformation is fail safe

`remove_ai_metadata` may copy an undecodable file through unchanged instead of
raising. User facing callers must use `strip_and_verify` and inspect its
surviving marker mapping before reporting success. `strip_and_verify` recovers
when `image_io` can still decode the raster by normalizing the container and
checking again. A truly undecodable file still reports the surviving markers.
The CLI uses this verified path.

### Sixteen bit PNG output is not preserved

The Pillow based PNG metadata rewrite uses the normal image save path and may
reduce a sixteen bit PNG to eight bits. A byte-level PNG metadata stripper
would be required to preserve that bit depth.

## Detection extras

The `detect` extra decodes an open DWT-DCT watermark used in some Stable
Diffusion, SDXL, and FLUX workflows. That decoder is sensitive to the carrier
and transformations. A negative result is not a universal negative.

The `trustmark` extra adds Adobe TrustMark decoding. The implementation retains
an additional JPEG re-encode gate and requires the binary payload and schema to
remain identical because isolated decoder hits can otherwise be content noise.
It accepts Variant P schemas 0-2. Variant Q requires a different model, and
schema 3 is rejected at the measured precision threshold. Its NumPy 1.x runtime
limits the extra to Python 3.11-3.12; the rest of the package remains supported
through Python 3.14. The current TrustMark dependency line also resolves lightning 2.6.5
(PYSEC-2026-3624, no fixed release yet); the vulnerable `load_from_checkpoint`
path is unreachable here because TrustMark loads its checksummed checkpoints
with plain `torch.load`. Bump lightning and cut a patch release when a fix
ships. The calibration history is in
[module internals](module-internals.md#metadata-and-provenance).

A generic metadata-free AI-generated-image classifier is not shipped. It is a
separate open research task from provenance detection: the current Model 1
result is AI-versus-camera and still confuses some conventional graphics, CGI,
product cutouts, and scans with generation. The OpenAI/Gemini source finder is
narrower again and cannot substitute for the general classifier. The shipped
product identifies concrete local provenance signals instead of presenting
either research classifier as a supported verdict.

Frozen transfers of Community Forensics, SPAI, SAFE, RINE, Nonescape Mini,
Dual Data Alignment, PGC, and DGS-Net did not close this gap at the required low
false-positive rate. DDA supplies a complementary representation but its
independent errors make simple hybrids worse. PGC SD1.4 is strong on OpenAI,
but its published output and the useful global/residual ablations misclassify
most or all of the Kodak scan set; removing that branch removes the OpenAI
gain. DGS-Net is weak at the same operating point and also adds independent
photo errors. A source-disjoint public-domain museum audit confirmed a separate
human-art failure: Model 1 called 355/1,050 drawings, prints, manuscripts,
paintings, textiles, and studio-object or historical-photo rows AI-generated.
Its error was 0/150 on the historical-photo control, so ordinary camera accuracy
does not bound accuracy on other human-created visual media. A domain veto
removed the museum errors in development but failed the existing AI-test and
fresh-photo core. A second source-disjoint modern-negative audit measured the
commercial cells: across 421 date-clean Unsplash and CC-licensed Flickr images,
Model 1 accepted 16.4% as AI, concentrated in logo/graphic design (38.3%),
retouched fashion photography (34.1% on the Unsplash cell; 2/40 on the Flickr
replication), and product cutouts (16.0%), while composites
stayed clean at 0/50 and long exposure near clean at 11/146 after a
two-part contamination control (a 0/450 provenance-metadata scan and a
pre-2022-08 date bound that dropped 29 rows, correcting one contaminated
Flickr fashion cell). The failure pattern therefore spans both historical art
and modern stylized graphics, not photographic exposure technique. A
source-disjoint linear veto fit on the date-clean modern cells was then
rejected one gate earlier: no searched operating point repaired even one
modern dev error without dropping below the frozen EvalGEN and FLUX floors,
because modern human graphics and AI generations overlap almost one-to-one in
the frozen CLIP embedding, at roughly 1.3 to 1.5 AI positives lost per negative
repaired. Model 1 also lacks a time/device-disjoint negative contract
for modern computational photography. The measured public protocol, GitHub
survey, and rejected fusions are recorded in
[AI-generated image classifier research](ai-generated-image-classifiers.md#general-ai-classifier-github-sweep-2026-08-25).

## Output and traceability

Removing file-local signals does not remove:

- provider account history;
- server side copies or provenance stores;
- perceptual fingerprints;
- evidence that an image passed through a removal pipeline;
- legal disclosure duties.

See [scope, safety, and legal notes](legal-and-safety.md).
