# Qwen-Image improvement research (2026-06-20)

> Metric correction: the OCR metric from `scripts/fidelity_metrics.py` is
> normalized edit distance (NED), edit distance divided by the longer
> normalized string. The historical values are unchanged; the earlier CER
> label incorrectly implied normalization by reference length.

> Research archive. This page records experiments and decisions from the date
> above. It may mention prototypes or defaults that were later changed. Use the
> user guides and current source code for the supported interface.

Cited research behind the decision **"ship the `qwen` pipeline as-is, or improve it
first?"** Produced by the multi-source deep-research harness (5 search angles, 22
sources fetched, 85 claims extracted, 25 verified by a 3-vote adversarial check, 20
confirmed / 5 killed, 104 agent calls). Findings carry their confidence and vote.

## Context

The `qwen` pipeline runs base Qwen-Image (20B MMDiT, Apache-2.0) as a low-strength
img2img scrub (removal comes from the denoising `strength`). Certified oracle scrub
floors: OpenAI 0.10 (seed-robust), Gemini 0.25 (pinned seed). Measured against the
SDXL + canny-ControlNet pipeline (`scripts/fidelity_metrics.py`): Qwen preserves
**text** markedly better (incl. CJK and Cyrillic, lower OCR NED) but preserves
**faces** worse, smoothing skin (Laplacian-variance retention 0.40 vs 0.62, face
LPIPS 0.17 vs 0.09, ArcFace identity 0.38 vs 0.55 at the scrub floors). The goal of
the research: keep Qwen's text advantage while fixing the face-smoothing, and judge
production-readiness.

## Verdict

Base Qwen-Image is **shippable now as an opt-in text-content lane** (Apache-2.0 on
code and weights, scrub lever confirmed), but it is not a universal upgrade (it loses
faces). The strongest verified improvement path is to **add structure conditioning**
(a Qwen-Image ControlNet) to the existing base pass, the direct analog of the SDXL +
canny conditioning that wins on faces. Separately, **Z-Image / Z-Image-Turbo** (6B,
Apache-2.0) is the best-verified lighter alternative to evaluate before committing to
the 20B cost. At research time none of the improvements had measured face-fidelity
numbers at our scrub floors. The later `qwen-zimage` follow-up below adds a crowded
fixture, two direct upstream comparisons, and a provider-oracle-negative final candidate.
A broader seeded text/face matrix is still needed for general certification.

## Follow-up: ControlNet experiment + deeper research (2026-06-20)

The verdict's strongest lead -- adding a Qwen-Image ControlNet -- was **built, measured, and
CLOSED**.

**Experiment** (Modal A100-80GB; DiffSynth-Studio `QwenImagePipeline` + the Apache-2.0
`DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny` -- the only framework exposing
Qwen-Image + canny ControlNet + img2img `denoising_strength` in ONE call; diffusers ships no
`QwenImageControlNetImg2ImgPipeline`, its three Qwen ControlNet pipelines are txt2img only).
Measured on `gemini_3` (18 faces) at the Gemini scrub floor 0.25 vs base-Qwen 0.25 with
`scripts/fidelity_metrics.py`:
- **The actual failure mode (face skin texture) was NOT restored:** Laplacian-variance
  retention stayed flat (base 0.40 -> qwen+canny 0.40; per-face 13/16 within +-0.02 after a
  one-to-one face match, sd 0.016 -- not an averaging artifact). The SDXL+canny target 0.62
  was not approached.
- Identity rose modestly and broadly (ArcFace 0.346 -> 0.415, 12/16 faces improved) but the
  absolute stays ~0.42 ("a different person, slightly closer").
- Mechanism (verified, not inferred): canny conditioning was applied fully (scale 1.0, full
  denoise schedule); the canny edge map is clean facial geometry with BLANK skin (4.83% edge
  density) -- canny carries edges, not skin grain. Root cause: Qwen's Gemini floor (0.25) is
  higher than SDXL+canny's (0.15), forcing more denoising -> more smoothing; structure
  conditioning cannot compensate for that.

**Deeper research** (deep-research harness, 103 agents, 3-vote adversarial):
- **[high, unanimous] No permissively-licensed Qwen-Image tile / detail / realism / skin
  ControlNet exists anywhere** -- DiffSynth first-party is Canny/Depth/Inpaint only, InstantX
  Union is canny/soft-edge/depth/pose, the official QwenLM repo ships none. Every Qwen
  conditioning is GEOMETRY, the same class as the tested canny. **The "add a Qwen ControlNet to
  fix faces" lead is closed for good.**
- **[high, unanimous] Z-Image / Z-Image-Turbo (6B, Apache-2.0 on code AND weights, ~1/3 of
  Qwen 20B)** ships a documented `ZImageImg2ImgPipeline` with standard strength denoising, so
  it preserves the scrub mechanism. Its own SynthID scrub floor and broad text fidelity
  remain unmeasured. The later `qwen-zimage` follow-up provides direct face metrics.
- **[medium] Lowering Qwen's scrub floor has no off-the-shelf SynthID answer:** the "partial
  img2img ~0.3 breaks robust watermarks" literature tests open schemes
  (StegaStamp/TrustMark/VINE), NEVER SynthID (proprietary decoder) -- analogy, not proof. No
  minimal-strength SynthID attack under a named permissive license was found.
- **REFUTED [0-3]:** "re-injecting high-frequency detail from a clean diffusion output would
  not carry the watermark back." So non-regenerative detail transfer is NOT safe by
  assumption -- the transferred high-frequency band must be gated against the SynthID oracle.

**Net for the single-pass `qwen` pipeline:** faces stay on SDXL+controlnet; Canny alone is not
a Qwen face fix. The next distinct architecture was Z-Image-Turbo on original masked face
crops, not another Qwen geometry conditioner.

**Implementation follow-up (2026-07-24, revised 2026-07-31):** an early experimental
`qwen-zimage` prototype reproduced a broad two-stage shape demonstrated by a public
experiment: structure-guided Qwen regeneration followed by masked Z-Image face
refinement. The maintained profile was subsequently
rewritten with project-owned prompts, adaptive strength and sizing policies, YuNet face
detection, SAM selection, masks, and compositing. It does not include the upstream workflow
JSON or source code. The upstream result supplied by the user was Gemini-oracle negative.

The first exact-path Modal run completed without SAM fallback or face seams. On one crowded
18-face `gemini_3` fixture, ArcFace identity improved materially over controlnet
(0.795 vs 0.587/0.588 raw/polished), while face LPIPS was 0.082 vs 0.087/0.078.
Two official upstream before/after pairs then reproduced the same identity advantage:
local `qwen-zimage` scored 0.950/0.947 ArcFace identity versus polished ControlNet's
0.701/0.548. The published upstream outputs remained slightly higher at 0.976/0.976.
On the matched-size group example, local face LPIPS essentially matched upstream
(0.015 vs 0.014) while whole-image LPIPS and SSIM were better (0.085/0.896 vs
0.111/0.777). ControlNet retained more texture but changed the identities.

That comparison also caught a real /16 alignment defect: DiffSynth dimensions were floored
without resizing the corresponding PIL input. The official non-grid input failed with a
VAE/noise-grid shape mismatch. Regression assertions were observed failing before global
and face pixels were aligned to the same grid as their explicit dimensions.

The final verifier follow-up is positive but exact-output scoped. On 2026-07-25 the user
checked all six current outputs in the provider-separated
`full-clean-final-candidate-2026-07-25-by-oracle` bundle with the corresponding provider
oracles and confirmed that none retained SynthID or the provider generation signal. The
checked bytes used the complete `visible -> qwen-zimage -> metadata` route, the calibrated
YuNet 0.5 gate, and the shipped prompt-cache/model-residency optimizations. This supersedes
the earlier prototype batch check as the release-candidate result, but it is not a
certification across seeds, resolutions, and content classes. YuNet's threshold was
calibrated independently from upstream YOLO: 0.5 retained the visible faces in the
comparison fixtures while removing the false and duplicate boxes admitted by the copied
0.2 gate. The remaining questions are now narrower: how well does text hold across a
broader set and how stable removal is across seeds and Google content types. See
`docs/known-limitations.md`.

The shipped profile therefore resolves an omitted seed to `0`; explicit seeds still
override it, and every other pipeline keeps its existing random default. This makes the
documented `--pipeline qwen-zimage` command reproduce the seed condition used for the
release-candidate oracle evidence without claiming cross-seed certification.

**Product recommendation:** keep ControlNet as the default because it is much cheaper and
supports CUDA, XPU, MPS, and CPU. Treat it as the compatibility baseline, not the
highest-fidelity result. When CUDA is available and visual quality matters more than
latency or cost, recommend `qwen-zimage`, especially for face-heavy content. This
recommendation is based on the measured identity advantage and the oracle-negative exact
candidate; it does not upgrade that exact-output result into broad certification.

**Follow-up (2026-06-20) — the content-routed lane / mixed dual-pass was tested and DROPPED.**
A `--pipeline auto` router (Haar+MSER → text→qwen / faces→controlnet / both→mixed) and a
faces+text mixed dual-pass (scrub the whole frame on both, graft qwen text regions onto the
controlnet base) were built and run on Modal (the abba poster: faces + display text). On that
canonical faces+text case **controlnet won EVERY metric, including text** (NED 0.114 vs qwen
0.379; ID 0.64 vs 0.36) — canny holds existing letter shapes, qwen re-renders display text and
garbles it, so grafting qwen text only hurts. Qwen beats controlnet on text ONLY for clean body
text on a plain background with no faces (openai_1/2), a niche `--pipeline qwen` alone covers;
the faces+clean-body-text intersection is near-empty, and "text→qwen" is undecidable cheaply
(body-vs-display text is what matters). So the router + mixed modules were removed and **`qwen`
is a manual `--pipeline qwen` opt-in only.** KEPT (independently valid): the qwen geometry fix
(it squished non-square inputs to 1024²), the pipeline-aware `resolve_strength` Qwen ladder, and
the `fidelity_metrics.py` one-to-one face matcher below.

**Tooling fix surfaced by this run:** `scripts/fidelity_metrics.py` face matching was changed
from per-face nearest-center to a collision-free one-to-one assignment
(`assign_faces_one_to_one`, gated by face size), after the 18-face `gemini_3` exposed
collisions (the regenerated variants detected 17 faces, so two originals mapped to the same
variant face, corrupting the identity metric). lapvar/LPIPS were always anchored to the
original bbox and stayed collision-immune. Regression-guarded by
`tests/test_fidelity_matching.py`.

## Findings

1. **[high, 3-0] A permissively-licensed Qwen-Image ControlNet exists today and is
   CUDA/diffusers-runnable.** InstantX Qwen-Image-ControlNet-Union supports
   canny/soft-edge/depth/pose; DiffSynth-Studio maintains blockwise Canny/Depth/Inpaint
   plus an In-Context-Control-Union; diffusers exposes `QwenImageControlNetPipeline`
   and `QwenImageMultiControlNetModel` with `controlnet_conditioning_scale` (default
   1.0) and `control_guidance_start`/`end`. This is the direct analog of the certified
   SDXL+canny structure conditioning that wins on faces. Caveat: canny/depth preserve
   geometric structure, not face identity per se, and none is a **tile**-ControlNet
   (the variant most tied to fine-detail/skin retention in the SDXL world).
   Sources: InstantX/Qwen-Image-ControlNet-Union, InstantX/Qwen-Image-ControlNet-Inpainting,
   DiffSynth-Studio Qwen-Image docs, diffusers qwenimage pipeline docs.

2. **[high, 3-0] The scrub mechanism is preserved, and the license is clean.**
   `QwenImageImg2ImgPipeline.strength` (default 0.6, range 0-1; DiffSynth names it
   `denoising_strength`) keeps the partial-regeneration scrub the project relies on,
   lower values staying closer to the input. Qwen-Image and Qwen-Image-Edit-2509 are
   Apache-2.0 on both code and weights.

3. **[medium, mixed 2-1 / 3-0] Qwen-Image-Edit improves identity consistency, but that
   is not proof it fixes our metric.** The instruction-edit pipeline (2511 better than
   2509) improves identity/character consistency, but only for identity *through edits*
   of an input portrait, which is not the same as measured face-skin Laplacian/LPIPS
   fidelity at a low scrub strength. Architecture: 20B base + Qwen2.5-VL (semantic
   control) + VAE Encoder (appearance control). Several stronger edit-model face claims
   were refuted (see below).

4. **[high, 3-0] Z-Image / Z-Image-Turbo is the best-verified lighter alternative.** A
   6B model (~1/3 of Qwen-Image's 20B), Apache-2.0 on code and weights, strong bilingual
   (Chinese + English) native text rendering, with an official diffusers
   `ZImageImg2ImgPipeline` exposing the same 0-1 denoising-strength scrub lever; Turbo
   runs at ~8 steps (guidance_scale=0.0) vs ~40. A material cost/footprint reduction vs
   20B/A100-80GB (but see caveat 4 on the refuted consumer-GPU claim).

5. **[high, 3-0] EliGen-V2 is NOT relevant** to the face-smoothing problem. It is an
   entity-level/regional control model (LoRA + regional attention placing entities via
   text + mask maps, plus entity-level inpainting); it provides no
   ControlNet/canny/depth/tile structure conditioning or face-skin-detail retention.

6. **[medium, 2-1] flymy-ai/qwen-image-realism-lora** is Apache-2.0 (code+weights) on
   base Qwen-Image, so it is permissively usable with the existing base img2img pass,
   but it is NOT verified to specifically fix the face/skin-smoothing failure mode.

## Caveats

1. The research did NOT surface verified evidence for two things specifically asked:
   (a) a Qwen-Image **tile**-ControlNet (the variant most tied to fine-detail/skin
   retention; only canny/soft-edge/depth/pose/inpaint were confirmed), and (b) any
   **non-regenerative detail-restoration** technique (high-frequency residual transfer,
   guided filtering) that recovers smoothed faces without re-introducing the watermark.
   Research angle 4 produced zero surviving claims, so it is unanswered.
2. No external claim provides measured face-fidelity numbers (ArcFace/LPIPS/Laplacian) for
   any recommended intervention at the project's scrub floors. The later direct
   `qwen-zimage` comparisons are the project's own measurements, not external evidence or
   a certification.
3. Several vendor model cards are marketing-register primary sources (Qwen blog,
   Z-Image card). Load-bearing facts (license, params, API levers) are independently
   corroborated, but comparative quality framings are author glosses.
4. Z-Image's "sub-second" figure is H800-specific and author-benchmarked; consumer-GPU
   third-party benchmarks are still limited (seconds, not sub-second, though within the
   <16GB envelope).
5. Time-sensitivity: Qwen-Image-Edit-2511 and Z-Image are late-2025/2026 releases; the
   diffusers pipelines cited are on the main/dev branch, so confirm released-version
   availability before pinning.
6. Five claims were refuted (below), clustering on over-strong edit-model face-fidelity
   and one over-strong Z-Image cost claim.

## Open questions

- Does a Qwen-Image **tile-ControlNet** (or equivalent high-resolution detail
  conditioning) exist under a permissive license?
- What **non-regenerative detail-restoration** method recovers smoothed faces WITHOUT
  re-introducing SynthID? Note: residual transfer from the ORIGINAL risks copying back
  watermark-carrying high frequencies, so it must be verified against the SynthID oracle.
- Head-to-head: does `qwen-zimage` retain its measured ArcFace gain across more portraits
  and mixed scenes, match Qwen's text advantage (CJK+Cyrillic NED), and clear SynthID
  robustly across content types and seeds?
- Can YuNet's serial face workload be bounded to foreground/relevant faces without losing the
  small faces the upstream YOLO path would process?

## Refuted claims (do NOT rely on these)

- [0-3] "Qwen-Image-Edit-2511 specifically targets/mitigates image drift, the same
  failure mode as face-detail loss in a low-strength scrub." (qwen.ai/blog, 2511)
- [0-3] "Qwen-Image-Edit-2509 explicitly improves facial identity preservation and
  supports portrait styles and pose transformations." (HF Qwen-Image-Edit-2509)
- [0-3] "Qwen-Image-Edit-2509 has native built-in ControlNet support (depth/edge/
  keypoint)." (HF Qwen-Image-Edit-2509)
- [1-2] "flymy realism LoRA specifically targets facial and skin detail, the exact
  failure mode." (HF flymy-ai/qwen-image-realism-lora)
- [0-3] "Z-Image-Turbo runs on consumer 16GB-VRAM hardware, far below the A100-80GB of
  Qwen-Image 20B, materially lowering per-image cost." (HF Tongyi-MAI/Z-Image-Turbo)

## Sources

1. https://qwen.ai/blog?id=qwen-image-edit-2511
2. https://qwenlm.github.io/blog/qwen-image-edit/
3. https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit
4. https://github.com/FurkanGozukara/Stable-Diffusion/wiki/Qwen-Image-Edit-2511-Free-and-Open-Source-Crushes-Qwen-Image-Edit-2509-and-Challenges-Nano-Banana-Pro
5. https://myaiforce.com/qie-2511/
6. https://huggingface.co/Qwen/Qwen-Image-Edit-2509
7. https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union
8. https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting
9. https://huggingface.co/DiffSynth-Studio/Qwen-Image-EliGen-V2
10. https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Qwen-Image.md
11. https://blog.comfy.org/p/day-1-support-of-qwen-image-instantx
12. https://learn.thinkdiffusion.com/how-to-use-qwen-image-with-instantx-union-controlnet-in-comfyui-guide-workflow/
13. https://huggingface.co/flymy-ai/qwen-image-realism-lora
14. https://huggingface.co/lightx2v/Qwen-Image-Lightning/discussions/4
15. https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage
16. https://www.diyphotography.net/skin-retouching-technique-frequency-separation/
17. https://link.springer.com/content/pdf/10.1007/978-3-642-15549-9_1.pdf
18. https://github.com/ShieldMnt/invisible-watermark/wiki/Frequency-Methods
19. https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
20. https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/z_image/pipeline_z_image_img2img.py
21. https://arxiv.org/pdf/2511.22699
22. https://github.com/ModelTC/LightX2V-Qwen-Image-Lightning
