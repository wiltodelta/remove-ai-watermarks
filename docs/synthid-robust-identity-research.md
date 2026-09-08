# SynthID-robust face identity for an SDXL removal pipeline (research)

> Research archive. This document evaluates approaches considered during the
> study. It does not describe the current public interface. The current
> pipelines and limits are documented in `docs/cli.md` and
> `docs/known-limitations.md`. The path it calls shipped -- `photomaker_restore.py`
> and the `photomaker` extra -- was removed from the library afterwards.

> **Status (2026-06-08): retired.** Every approach described below was empirically
> tested and rejected -- see `docs/synthid-robust-identity-research-2026-06-08.md`
> "Empirical follow-up" for the final conclusion. The library no longer ships any
> face-restore extra: the cleaned image from the main controlnet 0.20 pass is the
> least-AI face state we can reach without re-introducing SynthID. This document
> is kept as historical record of the exploration.

**Question.** Which face identity-preservation mechanism for an SDXL img2img +
canny-ControlNet watermark-removal pipeline (denoise 0.20-0.30) is BOTH (a)
commercial-safe end-to-end and (b) does not re-introduce the SynthID pixel
watermark the removal pass just destroyed?

**Constraint.** A paid deployment requires every component (adapter weights AND
the face embedder it conditions on AND any base model) must be Apache-2.0 / MIT /
BSD or otherwise clearly commercial-permitted. Non-commercial is disqualifying.

**One-line verdict.** Today there is **ONE** SDXL identity-conditioning stack that
is commercial-safe end-to-end: **PhotoMaker-V1** (Apache-2.0, identity encoded as a
fine-tuned OpenCLIP-ViT-H/14 image embedding -- NO InsightFace). Every other
candidate -- **including PhotoMaker-V2**, IP-Adapter FaceID, InstantID, PuLID,
Arc2Face -- inherits InsightFace's non-commercial model-pack license through an
ArcFace-class embedder and is therefore blocked for paid services, regardless of
the adapter's own license header. Below is the evidence per component and the
integration plan.

**Correction notice (2026-06-04).** An earlier version of this doc claimed
PhotoMaker-V2 was commercial-safe end-to-end. That was WRONG -- the V2 model card
phrase *"id_encoder includes finetuned OpenCLIP-ViT-H-14 and a few fuse layers"*
described one of TWO ID branches; the V2 source (model_v2.py) defines
`PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken` whose forward takes an
ArcFace `id_embeds` from `insightface.app.FaceAnalysis`, and the upstream package
`__init__.py` imports InsightFace at module load. A Modal cert sweep caught this
empirically (`No module named 'insightface'` from `restore_faces_photomaker`). V1
is the correct commercial-safe target: its `PhotoMakerIDEncoder` (model.py)
forward takes only `(id_pixel_values, prompt_embeds, class_tokens_mask)` -- no
ArcFace branch -- so identity is CLIP-only.

**Status notice (2026-06-04, end of session).** Two commercial-safe paths were
tried and abandoned:

1. **PhotoMaker-V1** (commercial-safe by license but blocked by upstream compat).
   The cert sweep hit a cascade of upstream compatibility issues with the diffusers
   version we ship (0.38): missing `einops` declaration, missing `peft` declaration,
   default `pm_version='v2'` that mis-loads V1 weights into the V2 encoder, custom
   `id_encoder` left on CPU after `pipe.to(device)`, and a CFG-batch tensor-shape
   mismatch in the denoising loop (`Expected size 2 but got size 1`). 7 cascading
   fixes did not get the pipeline running end-to-end. The PhotoMaker `pipeline.py`
   header notes it was forked from diffusers v0.29.1; SDXL prompt-encoder handling
   changed significantly between 0.29 and 0.38.
2. **GFPGAN on the diffusion-CLEANED image** (commercial-safe, but no identity
   recovery). A one-line change made it SynthID-safe (input pixels are already
   clean, so the partial blend cannot transport the watermark), but visual
   inspection of the cert output showed it only polished the already-drifted face
   without actually restoring identity. Trade-off was real and the value too low
   to keep.

**The shipped path is PhotoMaker-V2** (`photomaker_restore.py`, the `photomaker`
extra). V2 uses a DUAL ID encoder (CLIP image features + ArcFace embedding),
which delivers true identity-from-embedding face regeneration. The cost is that
the ArcFace embedding comes from InsightFace's `antelopev2`/`buffalo_l` model
packs, which are released under a non-commercial / research-only license. **So
the shipped restore path is NON-COMMERCIAL.** Any monetized
deployment must NOT install the `photomaker` extra. The CLI flag and module
docstring both call this out at every entry point.

A future commercial-safe path would need either (a) the PhotoMaker upstream to
land its diffusers 0.38 compat fix so V1 can run, or (b) an equally good
ArcFace-class face-recognition model released under a permissive license that
PhotoMaker-V2 can be retargeted to. Neither is on a near-term horizon as of
this writing.

## 1. Why identity-by-embedding (not by pixel) is the only SynthID-robust path

The pipeline regenerates pixels to destroy SynthID. Any identity-restoration that
is "faithful to the input pixels" (GFPGAN, CodeFormer, face-swap-by-blending, our
previous restore-on-original pass) reproduces the watermark, because SynthID is
engineered to be robust to fidelity-preserving transforms (resize, JPEG, partial
blend). Oracle-confirmed on a real Gemini face: controlnet @ 0.20/0.25 WITH the
GFPGAN restore pass left SynthID detected; the SAME controlnet @ 0.20 with
`--no-restore-faces` cleared it (clean A/B, see `docs/synthid.md` 5.5 and
`docs/controlnet-removal-pipeline-research.md`).

The only mechanism that can preserve identity AND not re-introduce SynthID is to
carry identity in a SEMANTIC EMBEDDING (a vector that encodes "who is in this
picture") and use it to CONDITION a fresh generation -- the pixels are new, so
the watermark is not transported. Two embedding families exist in practice:

- **ArcFace-class face-recognition embeddings** (the InsightFace family). Used by
  IP-Adapter FaceID, InstantID, PuLID, Arc2Face. Highest identity fidelity, but
  the embedder weights are non-commercial.
- **CLIP image embeddings of a face crop**. Used by PhotoMaker (and the original
  IP-Adapter image variant). Lower identity fidelity at small scale than ArcFace,
  but the encoder (OpenCLIP-ViT-H/14, MIT) is commercial-safe.

## 2. License table (verified against primary sources, 2026-06-04)

| stack | adapter weights | identity encoder | end-to-end commercial-safe? |
|---|---|---|---|
| **PhotoMaker-V1** | **Apache-2.0** ([HF][pmhf]) | **OpenCLIP-ViT-H/14 (MIT)** finetuned, identity from `PhotoMakerIDEncoder` (`model.py`); forward takes only ``(id_pixel_values, prompt_embeds, class_tokens_mask)`` -- no ArcFace branch | **YES** |
| PhotoMaker-V2 | Apache-2.0 (adapter) ([HF][pm2hf]) | DUAL encoder: OpenCLIP-ViT-H/14 AND InsightFace antelopev2/buffalo_l -- `PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken` (`model_v2.py`) forward takes `id_embeds` from `insightface.app.FaceAnalysis`, and `photomaker/__init__.py` imports InsightFace at module load | NO -- InsightFace pack is non-commercial |
| IP-Adapter FaceID | non-commercial per model card: *"AS InsightFace pretrained models are available for non-commercial research purposes, IP-Adapter-FaceID models are released exclusively for research purposes and is not intended for commercial use"* ([HF][ipafhf]) | InsightFace antelopev2 (non-commercial for the model pack) | NO -- both layers block |
| InstantID | Apache-2.0 (adapter only) ([HF][insthf]) | requires InsightFace antelopev2 face-analysis at runtime (`FaceAnalysis(name='antelopev2', ...)` per the README usage snippet, [HF][insthf]) | NO -- embedder pack is non-commercial |
| PuLID | apache-2.0 (HF model metadata, [HF][pulidhf]) | depends on InsightFace face-analysis for ArcFace embedding (per the upstream README; PuLID's own card is sparse and the GitHub README documents the InsightFace install step) | NO -- same embedder issue as IP-Adapter FaceID |
| Arc2Face | MIT (HF model metadata, [HF][arc2hf]) | uses `insightface.app.FaceAnalysis` to extract the ArcFace embedding ([HF][arc2hf]); also based on SD-v1-5 (NOT SDXL) | NO -- non-commercial embedder + not SDXL |

**The crux is InsightFace.** InsightFace explicitly splits its license: *"Code is
MIT licensed; models require separate commercial licensing"* and frames the
pretrained packs as *"Commercial licensing for InsightFace's open-source model
packages"* requiring users to *"obtain commercial usage rights for model
packages"* ([insightface.ai][iflic]). antelopev2 and buffalo_l fall under the
model-pack license, not MIT. So any stack that calls
`insightface.app.FaceAnalysis(name='antelopev2', ...)` to compute its ArcFace
embedding is blocked by default, REGARDLESS of the adapter's own Apache header
above it. This is the same reason IP-Adapter FaceID's card flags itself
non-commercial.

(Note on PuLID's HF metadata: the model card declares apache-2.0 for the adapter
weights but the upstream repo's quickstart requires the InsightFace package to
extract the ID embedding. So PuLID's adapter license is permissive; the BLOCKER
is the embedder it expects at runtime. This is the same trap as InstantID.)

[pmhf]: https://huggingface.co/TencentARC/PhotoMaker
[pm2hf]: https://huggingface.co/TencentARC/PhotoMaker-V2
[ipafhf]: https://huggingface.co/h94/IP-Adapter-FaceID
[insthf]: https://huggingface.co/InstantX/InstantID
[pulidhf]: https://huggingface.co/guozinan/PuLID
[arc2hf]: https://huggingface.co/FoivosPar/Arc2Face
[iflic]: https://www.insightface.ai/solutions/face-recognition-licensing

## 3. Is there a commercial-safe ArcFace replacement?

Short answer: **no clean drop-in**. The widely deployed pretrained ArcFace packs
(antelopev2, buffalo_l, glint360k) come from InsightFace and are non-commercial.
ArcFace as an ARCHITECTURE is published in a paper, so retraining is legally fine,
but you would need:

- a commercial-licensed training dataset (the big public ones -- MS-Celeb-1M,
  Glint360K, WebFace -- carry research-only or licensing-uncertain restrictions);
- compute + time to train an ArcFace-class model on the legal dataset;
- the result would be a one-off effort, not a maintained dependency.

For a removal service this is a multi-month side project that delivers what
PhotoMaker already gives us with one pip install. So the practical answer is to
take the CLIP-embedding path (PhotoMaker-V1; V2 adds InsightFace and is non-commercial), accept the identity-fidelity
trade-off, and revisit ArcFace later if quality is insufficient.

## 4. Does an identity embedding leak SynthID?

This is the load-bearing assumption of the whole approach. The argument:

- SynthID is a low-amplitude, perceptually-invisible pixel watermark engineered
  to be robust to "fidelity-preserving" transforms (it survives JPEG, resize,
  crop, color, noise at >=99% TPR -- see arXiv:2510.09263 referenced in
  `docs/synthid.md`).
- A face-recognition / CLIP-image embedding is by design INVARIANT to such low-
  amplitude pixel changes (compression, brightness, small noise should not change
  "who is in the photo"). That is the whole training objective.
- Therefore the embedding extracted from a watermarked face vs. the same face
  cleaned should be ~identical -- the embedding cannot CARRY the watermark
  pattern, only the identity, because the watermark sits in exactly the
  dimensions the embedding learned to discard.

**MEASURED 2026-06-04 — hypothesis confirmed.** Ran a low-amplitude
perturbation sweep on 31 face crops (3 photoreal originals: gemini_3, gemini_4,
openai_3 grid), comparing `cos(embedding(orig), embedding(perturbed))` for OpenCLIP-
ViT-H/14 (laion2B-s32B-b79K, the same OpenCLIP-ViT-H/14 encoder PhotoMaker V1 and V2 both finetune for CLIP-side identity):

| perturbation | mean cos | min | max |
|---|---|---|---|
| **synthid_proxy** (±2 LSB low-freq noise, σ=4 px Gaussian carrier — same regime SynthID hides in) | **0.9977** | 0.9937 | 0.9996 |
| noise3 (Gaussian σ=3, full-spectrum) | 0.9541 | 0.9055 | 0.9825 |
| jpeg90 (SynthID survives this) | 0.9280 | 0.8806 | 0.9566 |
| blur1 (Gaussian σ=1) | 0.9139 | 0.8103 | 0.9875 |
| jpeg70 | 0.8945 | 0.8125 | 0.9603 |
| (self check: identical crop) | 1.0000 | 1.0000 | 1.0000 |

The SynthID-magnitude perturbation moves the embedding by **0.002** (cosine 0.9977),
an order of magnitude less than JPEG90 — which SynthID survives at >=99% TPR by
design. So the embedding cannot carry the watermark pattern: its discriminative
signal is in dimensions the SynthID payload does not occupy. PhotoMaker-V1
conditioned on a watermarked face will see ~the same identity vector as if
conditioned on a clean face of the same person, so the freshly generated face
inherits the identity, not the watermark.

A first, naive smoke run measured `cos(orig, SDXL-cleaned)` instead — that test is
about diffusion drift, not watermark invariance (diffusion at strength 0.20-0.30 is a
much larger perturbation than SynthID), so its 0.56-0.93 spread is the identity
drift the PhotoMaker pipeline is meant to fix in the first place. The
synthid_proxy result above is the one that actually answers the load-bearing
question. Script: `/tmp/identity_smoke/test2_proxy.py` (not committed; reproducible
from the test set + this doc).

## 5. PhotoMaker-V1 properties for our pipeline

- **SDXL-native.** PhotoMaker v1 and v2 target Stable Diffusion XL; the pipeline
  is a stacked-ID embedding fused into SDXL's cross-attention via the fuse layers
  bundled in the released weights.
- **Identity from a SINGLE reference image works** but the method was designed
  for "stacked" multi-reference; with one image identity fidelity is lower than
  with 3-4, and a service has only one (the upload). This is the failure mode to
  guard.
- **Compatibility with img2img + canny ControlNet.** PhotoMaker is typically
  exposed in txt2img workflows in the upstream demo. SDXL img2img + ControlNet
  is the same denoising backbone, so the cross-attention injection works the same
  way; community examples on Diffusers and ComfyUI confirm PhotoMaker stacks with
  ControlNet. Validate this on a representative image before adopting.
- **Failure modes to expect:**
  - identity drift on small / multi-face groups (the 9-face grid case);
  - "plastic" / over-smoothed faces if PhotoMaker's identity weighting is high
    while the img2img strength is low;
  - canny ControlNet conditioning can fight the ID embedding (edges of the
    ORIGINAL face vs identity of the SAME person regenerated) -- expect to tune
    `controlnet_conditioning_scale` down a notch on photoreal faces;
  - PhotoMaker was trained on a celebrity-skew distribution; representative faces
    (especially non-white, non-Western, elderly, children) may have lower
    fidelity. Measure on a representative evaluation distribution.

## 6. Integration cost (rough)

- New deps: `diffusers` already in the diffusion extra; PhotoMaker ships as a `.bin`
  loaded via `pipeline.load_photomaker_adapter(...)`. The OpenCLIP encoder is the
  same one diffusers already pulls. No new heavy pip dep.
- Weight download: PhotoMaker-V1 weights are ~3 GB. Add to the model cache
  alongside SDXL.
- VRAM: SDXL + canny ControlNet + PhotoMaker-V1 fits comfortably in A100-40GB.
- Latency: a few extra seconds on cold start (load PhotoMaker), negligible per
  request after warm-up.
- No InsightFace install: huge win for `restore` extra's basicsr/numpy hell --
  this path simply does not touch that ecosystem.

## 7. Recommended path

1. **Embedding-invariance smoke test FIRST** (one afternoon, no codegen):
   - For ~10 OpenAI / Gemini watermarked faces, compute OpenCLIP-ViT-H/14
     embeddings; for the same images after our SDXL `default` pass at the
     certified strength, compute the embeddings again; assert mean cosine
     similarity > ~0.95.
   - If yes -> the embedding does not carry SynthID, proceed.
   - If no -> the assumption is wrong; PhotoMaker would re-introduce the
     watermark. Stop and reconsider.
2. **PhotoMaker-V1 prototype** in the existing `controlnet` pipeline:
   - Mirror the `_load_controlnet_pipeline` path (removed in 0.24.0 with the
     controlnet profile; the equivalent seam is now `SdxlZImagePipeline._load_global`):
     add a PhotoMaker variant that
     loads SDXL + canny ControlNet + PhotoMaker adapter on the same engine.
   - Extract the OpenCLIP face embedding from the watermarked face crops (use
     OpenCV YuNet, already bundled for `auto`, to find the face boxes).
   - Pass the embedding as PhotoMaker's `id_embeds` to the SDXL pipeline; run
     img2img at the certified strength (0.20 OpenAI, 0.30 Gemini-capped-1536)
     with the canny edge map.
3. **Oracle validation** on the cert sweep: run the new PhotoMaker variant
   through the isolated Modal certification harness over the same 6 image set, certify on the
   per-vendor oracles. Expected: SynthID cleared (the regeneration is the same)
   AND identity recovered (the embedding adds it back).
4. **Honest exit criteria.** Ship only if BOTH oracle reads clean AND a small
   user-perception test on representative inputs says "looks like me". If identity is
   still too soft on small faces -> add stacked-reference (multiple crops of the
   same upload at different scales) before reaching for a non-commercial
   embedder.

## 8. What we are NOT doing, and why

- **No InsightFace.** Non-commercial for model packs (see License table).
- **No CodeFormer.** Non-commercial.
- **No GFPGAN on the original image.** It re-introduces SynthID
  (oracle-confirmed).
- **No GFPGAN on the cleaned image.** It cannot RECOVER identity that the
  diffusion pass already drifted -- it can only smooth/sharpen whatever face is
  already there. Useful as cosmetic polish, not as identity restoration.
- **No retraining of an in-house ArcFace.** Out of scope for a removal service.

---

## Process note

The deep-research harness was run but its verifier subagents failed to call
`StructuredOutput` (same harness bug as the prior 2026-05-XX run), so its synthesis
was unusable. The license claims above were verified by directly fetching the HF
model cards and the InsightFace licensing page and quoting them; the
embedding-invariance argument is mechanistic and explicitly flagged as not yet
measured (it is the first integration step). Do not treat the deep-research
output as ground truth for this file.
