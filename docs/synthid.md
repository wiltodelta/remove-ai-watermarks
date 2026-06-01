# SynthID-Image: technical reference

This document covers how Google SynthID for images works mechanically, what it
survives, what removes it, and the current deployment landscape. It is written
for engineers working on watermark detection and removal -- specifically to
inform decisions about strength settings, test methodology, and what oracle
results mean.

Primary sources are cited inline. Marketing-only claims are flagged separately
from independently-verified results.

---

## 1. Mechanism

### 1.1 Post-hoc, model-independent design

SynthID-Image is **not** baked into a diffusion model's weights. It is a
post-hoc, model-independent system: a separate encoder `f` is applied to an
already-generated image, and a separate decoder `g` reads it back.

> "We deliberately designed SynthID-Image as a post-hoc, model-independent
> approach, a choice largely based on deployment considerations."
> -- Gowal et al., arXiv:2510.09263

The formal definition from the paper:

> "A post-hoc watermarking scheme is a pair f, g consisting of an encoder
> function f: X -> X, which adds an identification mark, and a decoder
> function g: X -> {+-1}, which tries to detect if the mark is present."

This is the key architectural fact: **the generative model (Imagen, Gemini's
image model) is not modified**. The watermark is stamped onto the pixel output
after generation, by a separate neural network. This means:

- The watermark is in **pixel space**, not in the model's latent activations.
- Replacing the generative model does not remove the watermarking capability.
- The encoder/decoder pair can be updated independently of the generative model.

The paper does not disclose the internal architecture of the encoder/decoder
networks (layer types, capacity). The external variant SynthID-O is available
to partners; the production internal variant is not published.

### 1.2 How it differs from classical DWT-DCT watermarks

The open watermarks used by Stable Diffusion / SDXL / FLUX (via the
`imwatermark` library) use classical **DWT-DCT** frequency-domain embedding: a
fixed bit pattern is added to specific frequency coefficients of the image's
wavelet transform. This is fast, key-free, and locally detectable with a public
decoder.

SynthID-Image uses **jointly-trained deep learning models**:

> "SynthID uses two deep learning models -- for watermarking and identifying --
> that have been trained together on a diverse set of images. The combined model
> is optimised on a range of objectives, including correctly identifying
> watermarked content and improving imperceptibility by visually aligning the
> watermark to the original content."
> -- Google DeepMind blog, 2023

The practical difference for robustness: the deep learning encoder learns to
spread the signal across the image in a way that is optimized to survive a
specific perturbation distribution seen during training. Classical DWT-DCT
embeds in fixed, predictable frequency bins, making it brittle to any
operation that hits those bins (e.g., JPEG re-quantization wipes it cleanly at
quality <= 90).

### 1.3 Payload capacity

SynthID-O (the external/partnership variant) encodes:

- **136 bits** within a **512x512 pixel image**

For comparison (from the same paper):

| Method      | Bits | Resolution |
|-------------|------|------------|
| SynthID-O   | 136  | 512x512    |
| StegaStamp  | 100  | 400x400    |
| TrustMark   | 100  | 256x256    |
| WAM         | 32   | 256x256    |

The payload carries an identification mark (not a user-readable secret). The
paper separates watermark **detection** (is this watermarked?) from payload
**recovery** (what does the payload say?): the detection path is what oracles
like the Gemini app's "Verify with SynthID" exercise.

### 1.4 Where in the pipeline it lives

```
[Diffusion model]
       |
  raw pixel output
       |
  [SynthID encoder f]   <-- separate neural net, stamps the watermark
       |
  watermarked image
       |
  [served / downloaded]
       |
  [SynthID decoder g]   <-- separate neural net, run by Google's verifier only
       |
  present / not present
```

The VAE decoder of the diffusion model is **not** involved in watermarking.
Some in-generation watermark approaches (like the research method "Tree Ring")
inject the signal into the initial noise latent so it propagates through the
diffusion process and appears in the final image; SynthID-Image does not do
this -- it is applied after the VAE has already decoded latents to pixels.

---

## 2. Robustness

### 2.1 What the paper claims it survives (primary-source verified)

The SynthID-Image paper (arXiv:2510.09263) evaluates SynthID-O against **30
image transformations** grouped into 6 categories:

| Category    | Examples                                      |
|-------------|-----------------------------------------------|
| Color       | brightness, contrast, saturation, hue shifts  |
| Combination | combinations of multiple transforms           |
| Noise       | Gaussian noise, impulse noise, median filter  |
| Overlay     | text overlays, logos, stickers                |
| Quality     | JPEG compression, WebP, format conversion     |
| Spatial     | crop, resize, rotate, flip, padding           |

**TPR at 0.1% FPR -- SynthID-O vs. baselines (resized to 512x512):**

| Category         | SynthID-O | Best baseline (WAM) | Worst baseline (StegaStamp spatial) |
|------------------|-----------|---------------------|--------------------------------------|
| Identity (none)  | 100.00%   | 100.00%             | 100.00%                              |
| Aggregated       | 99.98%    | 90.62%              | ~70%                                 |
| Color            | 100.00%   | 81.29%              | ~75%                                 |
| Combination      | 99.96%    | 96.08%              | ~22%                                 |
| Noise            | 99.98%    | 100.00%             | ~92%                                 |
| Overlay          | 100.00%   | 100.00%             | 100.00%                              |
| Quality          | 99.99%    | --                  | ~89%                                 |
| Spatial (worst)  | 99.97%    | 76.04%              | 15.25%                               |

The "Spatial worst" row is the hardest case (aggressive crop + resize).
SynthID-O retains 99.97% TPR; StegaStamp collapses to 15.25%. This is where
the deep-learning approach gains the most over classical methods.

Google's marketing page states the watermark is:

> "designed to stand up to modifications like cropping, adding filters, changing
> frame rates, or lossy compression."
> -- deepmind.google/models/synthid/

The marketing claim is broadly consistent with the paper's numbers for these
specific categories.

**JPEG and format conversion specifically** fall under the "Quality" category,
where SynthID-O achieves 99.99% TPR. This is the empirical basis for the fact
that **GitHub-recompressed JPEGs from issue attachments are valid SynthID test
subjects**: the re-encoding does not remove the pixel watermark.

### 2.2 Stated limits (vendor claim, not independently verified)

> "SynthID isn't foolproof against extreme image manipulations."
> -- Google DeepMind blog, 2023

This is the only public failure-mode statement Google has made. No specific
perturbation type, threshold, or quantitative boundary is named. The
Limitations section of the paper (Section 10) was not recoverable from the
public HTML version of arXiv:2510.09263v1 due to a rendering failure in the
conversion (the body text of Section 10 is absent from the HTML).

**What is known empirically from independent work and our own testing:**

- **Diffusion regeneration / img2img** at sufficient strength degrades or
  removes the watermark. Our testing (May-June 2026, Gemini oracle):
  - strength 0.05: insufficient for current Gemini SynthID (survives)
  - strength 0.10: removes Gemini SynthID (verified via Gemini app oracle, n=1)
  - strength 0.30: current DEFAULT; removes Gemini SynthID (verified n=3 via
    Gemini app oracle on fresh Gemini images, June 2026 oracle study)
  - strength 0.30: **does NOT reliably remove OpenAI gpt-image SynthID on
    1600x1600 images** (verified via openai.com/verify, issue #14 reports,
    June 2026)
  - strength 0.35 and 0.40: not yet oracle-verified on 1600x1600 gpt-image;
    0.40 visibly corrupts text-heavy images
  - **Resolution dependence confirmed**: same strength removes watermark on
    small images (376x429) but not on large ones (1600x1600) -- larger images
    appear to carry a stronger or more spatially distributed signal
  - The production SynthID has been progressively hardened: 0.05 worked earlier
    (pre-May 2026 Gemini), then 0.10 was needed, now 0.30 for Gemini and still
    failing at 0.30 for 1600x1600 gpt-image. It is a moving target.

- **Heavy JPEG compression** (quality < ~50-60): not specifically tested with
  oracle verification; the DL approach is more robust than DWT-DCT but Google
  acknowledges limits at "extreme" manipulation.

### 2.3 Removal attacks and forensic detectability

The paper arXiv:2605.09203 ("Removing the Watermark Is Not Enough",
Goonatilake & Ateniese, 2026) evaluates 6 removal attacks against a ResNet-50
forensic detector. All attacks defeat the watermark verifier but are detected
by the forensic classifier:

| Attack          | Family           | AUROC  | TPR @ 1% FPR | TPR @ 0.1% FPR |
|-----------------|------------------|--------|--------------|----------------|
| UnMarker        | Distortion       | 0.9994 | 99.81%       | 98.28%         |
| WatermarkAttacker| Regeneration    | 0.9997 | 99.95%       | 99.38%         |
| CtrlRegen+      | Regeneration     | 0.9999 | 99.97%       | 99.64%         |
| NFPA            | Inversion/Pert.  | 0.9984 | 99.24%       | 62.10%         |
| Boundary Leak.  | Inversion/Pert.  | 0.9991 | 99.24%       | 88.34%         |
| WiTS            | Erosion          | 0.9999 | 99.80%       | 99.55%         |

The forensic detector is a standard ResNet-50 fine-tuned end-to-end; no exotic
architecture needed. The key finding:

> "These removers do not return images to a clean forensic state. They often
> trade an explicit watermark for an implicit watermark: a detectable artifact
> introduced by the removal process itself."

This means: even when our SDXL img2img pass defeats the SynthID pixel
watermark (oracle reads negative), the output may still be classifiable as
"an image that went through a removal pipeline" by an independent detector --
even if that detector is not trained on SynthID specifically. **Defeating the
verifier does not restore forensic deniability.**

CtrlRegen+ is the most detectable removal method (AUROC 0.9999), which is
notable because it is also the most powerful removal attack. The paper notes
that diffusion regeneration "leaves a strong reconstruction signature from the
diffusion prior."

---

## 3. Detectability and verifier access

### 3.1 No public local detector

The SynthID decoder is proprietary and not released:

> "SynthID-Image has been used to watermark over ten billion images and video
> frames across Google's services and its corresponding verification service is
> available to trusted testers."
> -- Gowal et al., arXiv:2510.09263

There is no public API, no released decoder weights, and no reproducible
algorithm for local detection. The verification service (SynthID Detector) is:

> "a verification portal" in early testing with "journalists and media
> professionals" on a waitlist
> -- deepmind.google/models/synthid/

The external variant SynthID-O is available "through partnerships" only. Our
tool cannot locally detect SynthID presence or absence -- this is by design,
not a gap we can fill.

### 3.2 How our tool detects SynthID (metadata proxy)

We detect SynthID indirectly: if the image's C2PA manifest is signed by a
known SynthID-using issuer (Google, OpenAI), we infer SynthID is present. This
is a **metadata proxy**, not a pixel watermark decode. It works while the C2PA
manifest is intact, and is silent once the manifest is stripped or the image
is re-encoded without C2PA (e.g., a screenshot, a social-media re-upload, or
after `metadata --remove`).

This is why:
- `identify` on a GitHub-recompressed issue attachment returns Unknown (C2PA is
  gone) even though the pixel SynthID is still present and detectable by
  openai.com/verify.
- A quiet `identify` output is not proof that SynthID was removed -- it only
  means the metadata signal is gone.

### 3.3 Oracle scope: each vendor detects only their own

From openai.com/research/verify (verbatim, verified 2026-05-31):

> "OpenAI generation signals will only be detected if the image was generated
> with our tools."
> "Content could also still be AI-generated by another company's model, which
> the tool currently does not detect."

SynthID technology is used by multiple vendors, but each verifier is keyed to
its own payload:

| Oracle                        | Detects          | Does NOT detect         |
|-------------------------------|------------------|-------------------------|
| Gemini app "Verify with SynthID" | Google SynthID | OpenAI SynthID          |
| openai.com/research/verify    | OpenAI SynthID   | Google SynthID          |

A Google-SynthID image reads clean on openai.com/verify. An OpenAI image reads
clean in the Gemini oracle. They are different payloads within the same
framework.

---

## 4. Adoption and current state (as of June 2026)

### 4.1 Google products

Google has watermarked **over 10 billion** images and video frames. The
deployment split by surface matters for our tool:

| Surface                              | SynthID pixel | C2PA metadata | Visible sparkle |
|--------------------------------------|---------------|---------------|-----------------|
| Gemini app (generated images)        | YES           | YES (Google)  | YES             |
| Gemini API / AI Studio / Nano Banana | YES           | NO            | YES             |

The Gemini API surface is a key blind spot: it embeds the pixel watermark and
the visible sparkle but **no C2PA or IPTC at all**. Our `identify` returns
Unknown on API-generated images unless the visible sparkle is detected (via
`check_visible=True`) or the user runs the Gemini app oracle.

### 4.2 OpenAI

OpenAI confirmed SynthID adoption (Help Center, updated 2026-05-21):

> "ChatGPT images include both C2PA metadata and SynthID watermarks."

This is time-gated: pre-rollout ChatGPT/gpt-image images carry C2PA without
SynthID. Our C2PA proxy therefore over-reports SynthID presence on old images
(hence the `_OPENAI_CAVEAT` hedging flag in the codebase).

### 4.3 Other vendors

- **Kakao** (South Korea): SynthID adopter as of May 2026 (Google announcement)
- **NVIDIA Cosmos**: SynthID for video (not still images; different pipeline)
- **Meta AI**: does NOT use SynthID; uses IPTC `digitalSourceType` marker instead

### 4.4 Version evolution (v1 vs v2 hardening)

Google has not publicly documented version numbers for the SynthID image
watermark in a way that maps to our testing observations. What is known
empirically from oracle tests:

- **Before May 2026 (Gemini)**: strength 0.05 removed the watermark
- **May 2026 (Gemini)**: strength 0.05 insufficient; 0.10 required
- **Current (Gemini, June 2026)**: strength 0.10 insufficient for fresh images;
  0.30 verified clean (Gemini app oracle, n=3, A100 GPU, native resolution)
- **Current (OpenAI gpt-image 1600x1600, June 2026)**: strength 0.30 still
  detected by openai.com/verify (issue #14, user qw1212ss report)

The progression suggests Google has progressively hardened the watermark -- the
embedding signal strength or spatial distribution has increased across model
generations. No Google announcement confirms this; the observation is purely
empirical from oracle tests.

---

## 5. Practical implications for this tool

### 5.1 Text and face protection: OFF by default

**Text protection (`--protect-text`) can preserve SynthID in text regions.**
Verified June 2026 on gpt-image at 1600x1600: same image processed twice --
with `--protect-text` the oracle detected SynthID; without it, SynthID was
removed. The mechanism: the global img2img pass clears SynthID everywhere, but
the text-protection high-resolution re-scrub regenerates those pixels from an
upscaled crop. At the effective resolution of the upscaled crop, the per-region
pass may be insufficient to re-destroy the payload, reconstituting SynthID in
text regions.

**Face protection (`--protect-faces`) has an even more direct preservation
mechanism.** The pipeline extracts face regions from the ORIGINAL (watermarked)
image BEFORE the diffusion pass, runs the global pass (which removes SynthID
everywhere), then blends the original face pixels BACK onto the result
(`invisible_engine.py`: `original_faces = protector.extract_faces(cv_img)`
before `remove_watermark`, then `protector.restore_faces(out_cv, original_faces)`
after). Those restored pixels are the original watermarked pixels -- SynthID is
guaranteed to survive in face regions, not just possibly. The text-protection
case is at least re-generating (uncertain); face protection is literally
restoring the original SynthID-bearing pixels.

Both `--protect-text` and `--protect-faces` are therefore **experimental and
OFF by default**. Enable only when text/face fidelity matters more than
watermark removal completeness, and always verify the result with the oracle.

### 5.2 Strength setting

There is no single permanent correct strength. The default 0.30 was set based
on the June 2026 oracle study (Gemini, n=3). Known gaps:

- **OpenAI gpt-image at 1600x1600**: 0.30 does not clear it (oracle-verified,
  June 2026). 0.35 and 0.40 untested with oracle. 0.40 visibly corrupts text.
- **Resolution matters**: the same strength that clears a 376x429 image fails
  at 1600x1600 (qw1212ss observation, issue #14, multiple images)

If the watermark survives at 0.30, the correct guidance is to try 0.35 then
0.40, using the lowest value that reads clean on the vendor oracle.

### 5.3 Test methodology

- **GitHub-recompressed JPEGs from issue attachments are valid SynthID test
  subjects.** JPEG re-encoding removes C2PA metadata but does NOT remove the
  SynthID pixel watermark (verified June 2026 on issue #14 pic3). Do not
  dismiss these as "not faithful originals" for SynthID-removal tests.
- **The correct oracle for OpenAI images is openai.com/verify**, not the Gemini
  app. The two oracles detect different payloads.
- **A quiet `identify` output after processing is not proof of removal.** It
  means the metadata proxy is gone. The pixel watermark state is unknown without
  an oracle check.
- **After removal, the output may carry forensic artifacts** detectable by an
  independent classifier even if the vendor oracle reads negative. Defeating the
  verifier is not the same as being forensically indistinguishable from clean
  content (arXiv:2605.09203).

### 5.4 ctrlregen and img2img: the tradeoff

Both the paper and our testing confirm: higher img2img strength removes the
watermark but introduces detectable regeneration artifacts. The Goonatilake &
Ateniese paper shows CtrlRegen+ (the most powerful remover) is simultaneously
the most forensically detectable (AUROC 0.9999). The tradeoff is unavoidable
with current diffusion-based approaches.

---

## References

1. Gowal et al. (2025). **SynthID-Image: Image watermarking at internet scale.**
   arXiv:2510.09263. https://arxiv.org/abs/2510.09263

2. Google DeepMind. **Identifying AI-generated images with SynthID.** Blog post,
   2023. https://deepmind.google/blog/identifying-ai-generated-images-with-synthid/

3. Google DeepMind. **SynthID.** Product page.
   https://deepmind.google/models/synthid/

4. Goonatilake & Ateniese (2026). **Removing the Watermark Is Not Enough:
   Forensic Stealth in Generative-AI Watermark Removal.** arXiv:2605.09203.
   https://arxiv.org/abs/2605.09203

5. OpenAI. **Verify tool for AI-generated images.** openai.com/research/verify.
   Accessed 2026-05-31.
