# SynthID local detector research

> Audit correction, 2026-09-08: multi-period affine-probe results from
> schema 13 or earlier used confirmation patches during period selection
> (R04). They require rerunning and recalibration under schema 14; the
> frozen runtime detector uses separate code and its thresholds are unchanged.
> External-negative claims admitted without explicit independent evidence
> (R08) also need an evidence audit before reuse.

> Research archive for the hunt for a local, keyless SynthID pixel detector.
> Not a statement of current product capability. Shipped behavior:
> [supported signals](supported-signals.md) and
> [known limitations](known-limitations.md).
>
> Sister pages: [SynthID source classifiers](synthid-classifiers.md),
> [SynthID mark removal](synthid-removal-research.md),
> [mechanism reference](synthid.md). Dated measurements live in the
> [chronological plan](synthid-detector-removal-plan.md).

## Result

The hunt for a local, keyless OpenAI SynthID pixel detector is open.
Google has no public payload decoder, and nothing in this project
reproduces one. Closed routes below failed to read the official oracle
contrast on photographs. The remaining work is a new waveform, a new
pair mint, or a new statistic, not a rerun of those routes.

The mark behaves like a keyed spread-spectrum residual: a content-adaptive
`x' = x + g(x)` with a paired decoder (DeepMind patent family, optional
secret in intermediate layers, encoder/decoder ensembles that do not
recognize each other). Without that pair, the residual sits under the
scene. Keyless energy, TrustMark transfer, L1 distillation, a flat-field
stamp, and a 16-32 band student all failed to read the official oracle
contrast on photographs.

S4, 2026-08-15: the two providers are not doing the same thing. Cross-image
correlation of the folded residual is high for Google (tile16 pos-pos
`+0.326`, chance `0.036`) and at chance for OpenAI (`+0.032`). Google
has phase-coherent residual structure also correlated with its controls;
normalized correlations do not establish an absolute amplitude ratio. OpenAI
shows no shared residual above chance in this measurement, consistent with a
content-dependent post-hoc encoder. Comb experts that work on
Google therefore cannot be reused as an OpenAI watermark detector.

M2, 2026-08-15, re-verified 2026-08-16: Google unwatermarked pairs cannot
be minted. `imagen-*` ids 404 from the model garden. `gemini-*-image`
rejects `addWatermark` (`Cannot find field`). Current Gemini API docs
state that all generated images include a SynthID watermark. There is no
encoder-off Google path.

What the runtime product uses for the *watermark* is signed provenance. The
official OpenAI verifier and the periodic-lattice expert are development-only;
the latter lives under `scripts/synthid_runtime/` and is not called from
`identify` or the CLI.
Lineage measurements of that expert are in
[SynthID source classifiers](synthid-classifiers.md).

## Public GitHub sweep, 2026-08-24

GitHub's primary-repository search returned 78 repositories with `synthid` in
the name and 81 with it in the description, 133 unique repositories in total.
The fork-inclusive searches were much larger and capped or dominated by copies
(860 name hits and 1,311 description hits), so this is not a claim that every
fork or unindexed private repository was read. The sweep also used targeted
code searches for detector, verifier, correlation, codebook, bypass, and
removal implementations. SynthID-Text, visible-corner-mark removers, metadata
cleaners, wrappers, and literal ports were classified before inspecting the
remaining image-pixel candidates at pinned commits.

The audits are not null results. They separate three claims that must not be
collapsed: a repeatable image statistic exists, that statistic is a calibrated
single-image SynthID detector, and changing its score transfers to the
provider's production decoder. The repositories support parts of the first
claim. None supports the second or third on the current challenge data.

| Repository | Retained contribution | Rejected use | Decisive result |
| --- | --- | --- | --- |
| [`Rinne414/SynthID-detector`](https://github.com/Rinne414/SynthID-detector/tree/b08abff06b67c37db9dfa7c6ed63760a446d560f) | A reproducible fixed-residual correlator and a useful spoofability negative control | Current-image detector or fixed-template remover | All three templates detected 0/400 current OpenAI and Google images. Adding the expected template spoofed the public detector, while subtracting 0.25-2x from an official positive left SynthID `detected` down to 25.72 dB |
| [`newideas99/gpt-image-synthid-detector`](https://github.com/newideas99/gpt-image-synthid-detector/tree/5495e09c62dc26112f9a0e6cfab469f9a10f37e3) | Evidence that a CNN can learn a renderer/regeneration-domain signature, plus a white-box shortcut baseline | Payload decoder, independent hybrid confirmation, or removal oracle | The published cut retained 172/200 OpenAI but accepted 110/200 Google. A gradient drove the surrogate effectively to zero, but official SynthID stayed `detected` through +/-8/255 at 34.30 dB |
| [`aloshdenny/reverse-SynthID`](https://github.com/aloshdenny/reverse-SynthID/tree/b11083676fd3ee3ff97ce9d03c0e409e46905902) | Cross-color phase consensus and further evidence for a repeating Google renderer-lineage pattern | The published V3/V4 thresholds as presence detectors, or Round-06 as a verified quiet remover | V4 accepted 77/200 Google and 76/200 OpenAI at the same cut. Round-06 has no published per-image oracle evidence and uses regeneration/distortion with 14/11 dB PSNR floors |
| [`Ristellise/REGRET`](https://github.com/Ristellise/REGRET/tree/7d449034bf323987e7e608e7886e029ed20fd847) | A compact 147-d second-order SPAM descriptor and a safely auditable sklearn model | Google or OpenAI detector, or a high-precision hybrid rescue | The published cut accepted 139/200 Google, 141/200 OpenAI, and 214/600 disjoint ImageNet photographs |
| [`vordme2010/synthid-dataset`](https://github.com/vordme2010/synthid-dataset/tree/133a27088f6f4d695c79db9a1a70fa8e7fa3adad) | A public 2026 Gemini flat-field corpus, safe numeric features, and explicit candidate carrier bins | Its reported near-perfect Tier-1 AUC as open-world SynthID evidence | Training used 500 Gemini flats against 1,500 synthetic or phase-scrambled negatives and zero real photographs. Rebuilding the SVM from the safe matrix accepted 1/200 Google and 0/200 OpenAI on v7 |

Useful forensic signal is therefore not synonymous with a production
watermark decoder. The Google lattice idea is retained only as a
research-lineage component; it cannot be labelled SynthID. The fixed template
and CNN gradient are retained as negative controls. None of these public
implementations is shipped as a detector or remover. Detailed classifier and
attack measurements live in [SynthID source classifiers](synthid-classifiers.md) and
[mark removal](synthid-removal-research.md).

### Google verifier surfaces found in the sweep

Two real but non-local verifier paths surfaced. The historical Vertex SDK
exposed `WatermarkVerificationModel.from_pretrained("imageverification@001")`.
The class still exists in `google-cloud-aiplatform` 1.165.1, but warns that it
was deprecated 2025-06-24 with removal scheduled for 2026-06-24. Calls in two
projects with Vertex enabled now reject the publisher model id as invalid. A
repository using `response.watermark_detected` is also stale: the actual SDK
property was `watermark_verification_result`, backed by the prediction's
`decision` field.

A newer unlisted upload route,
`POST https://synthid.googleapis.com/upload/v1:verifyContent`, is live enough
to identify itself as the SynthID API and return `SERVICE_DISABLED` for a real
public image. It has no public discovery document, did not appear in the
project's available-services listing, and no authorized successful response
was obtained. Enabling a cloud service was deliberately not attempted. This
may be an access-gated research oracle, but it is not a local decoder and does
not satisfy the metadata-free offline product requirement. Gemini chat remains
the only Google verifier actually evidenced by the public bypass corpora.

## Closed detector routes

| Route | Close | Why |
| --- | --- | --- |
| Wavelets / FFT / cepstrum as a single-image detector | 2026-08-09 | TrustMark proxy: 318-d wavelet/spectral summary AUC 0.653 and 0 TPR at a clean calibration cut. Complex FFT maps AUC 0.516. Spatial RGB still won |
| `aloshdenny/reverse-SynthID` V3 phase codebook | 2026-08-09/24 | Pickle-free numeric audit. The original pilot was 5/5 Google positives and 0/194 then-available negatives. The broader audit was 5/355 Google, 5/499 controls, and 6/1,000 fresh Open Images. On v7 it accepted 4/200 Google and 1/200 OpenAI. As a v11 `unknown` rescue it adds 2/200 Google on this test, but the historical control errors reject a 1%-recall OR rule |
| `aloshdenny/reverse-SynthID` V4 | 2026-08-13/24 | Commit `b110836`. Better-of-two profiles: 141/355 Google positives and 191/499 controls. Frozen 1,000 Open Images: 386 accepted. Paired AUC 0.517. An exact repeat on the hash-disjoint v7 challenge accepted 77/200 Google and 76/200 OpenAI. Strict GUI aspect routing reduced support but not false acceptance. The threshold sits 0.02 above chance phase similarity |
| `cebeuq/Synthid-Bypass` as a local decoder | 2026-08-13 | 12 public before/after diffusion pairs. Fixed carrier 10/12 before, 0/12 after. Gemini Verify still saw SynthID on two `before` files the carrier missed. Reconstruction removal, not a payload read |
| S4: one content-independent pattern for both providers | 2026-08-15 | Google has a shared phase-coherent tile. OpenAI is at chance. See Result |
| M2: mint Google encoder-off pairs | 2026-08-15 | No current Google path emits an unwatermarked image. See Result |
| U1: one feature space for both providers | 2026-08-15 | Transfer 0.535 and 0.469 |
| Hand statistics vs a perceptually masked keyed mark (M1b) | 2026-08-15 | At equal RMS, fixed 0.670, keyed 0.669, keyed+masked 0.546. Masking, not keying, defeats hand features |
| Decoder-substitution / foreign VAE as a verified negative | 2026-08-17 | OpenAI oracle stayed `detected` at 22.3 dB. Attacking a positive does not mint a labelled negative |
| `aoxo/reverse-synthid` as extra Google templates | 2026-08-19 | Hugging Face dump scored against the frozen era bank. Extra family means raised foreign scores instead of isolating a mark |
| Origin-locked lattice as SynthID | 2026-08-16 | Two-pixel crop kills it; published SynthID keeps 99.97% TPR under aggressive crop. See classifiers for `google_lineage_renderer` |
| Keyless energy in 16-32 px | 2026-08-20 | L1 AUC 0.53; official decoder reads phase structure, not energy |
| TrustMark / open-encoder transfer | 2026-08-20 | E3 leave-one-encoder-out at chance on TrustMark (0.505 ridge, 0.510 conv) |
| L1 distillation of oracle bits | 2026-08-20 | Geometry-only AUC 0.78-0.83 beats pixel 124-d (0.64). ChatGPT export and `opened` C2PA predict `not_detected`. Inside 1254x1254, permutation p=0.45 |
| Flat 16-32 matched filter | 2026-08-21 | Leave-one-out residual correlation 0.645 on gray flats; 0.007-0.025 on COCO photo residuals. L1 AUC 0.70 was a size confound (size-matched 0.59 / 0.44) |
| `gpt-image-1` as encoder-off pair | 2026-08-21 | Does not stamp. `gpt-image-2` does |
| CNN on raw L1 bits | 2026-08-21 | Do not train. Labels are export geometry and presentation, not the mark |
| 128-photo causal wavelet student on the flat template | 2026-08-21 | Photo 16-32 jam flips only at a=24 / 18.6 dB (`a*_photo < 0.53` failed). A student cannot beat a matched filter if the photo-domain mark is that quiet |
| Photo 16-32 amplitude without the flat-arm template | 2026-08-23 | 11 COCO `gpt-image-2` edits plus the red-dot pair. Residual is the camera JPEG resized onto the edit raster, top-left 32 px dropped. Median 16-32 residual RMS 12.2, same order as 8-16 (9.0) and 32-64 (16.3). On the five tightest pairs (full residual RMS < 16): 16-32 median 2.69, still ~24x the 0.11 flat stamp and not louder than 8-16 (3.81). The photo residual does not isolate a 16-32 codeword |
| Sort-by-luma 1D chroma stego | 2026-08-23 | Flat LOO 0.914 but a luma-order shuffle is already 0.792. COCO plus tiny noise: 0.642 vs shuffle 0.640. The sort invents the correlation |
| Retinex / local-gain then 16-32 LOO | 2026-08-23 | 0.742 vs plain 16-32 LOO 0.752 on the same eight gray flats. No new carrier |
| Distant 16-32 patch coupling | 2026-08-23 | Near lag 0.13 of a tile: -0.013. Far lag ~500 px: +0.054. No nonlocal lock |
| Bispectrum of the 16-32 ring | 2026-08-23 | Degenerate: 1.0 on flats and 1.0 on COCO. Not a statistic |
| Length-255 m-sequence vs 16x16 fold | 2026-08-23 | Max abs corr 0.059. Chance for n=255 is `1/sqrt(255) ≈ 0.063` |
| Row-as-audio pitch peak | 2026-08-23 | Peak-to-mean 51 on flats, 46 on COCO noise residuals. Shared FFT peakiness, not a pitch |
| 16x16 origin search of the flat 16-32 template on photos | 2026-08-23 | Zero-shift NCC 0.006-0.024, best of 256 shifts 0.013-0.026. Origin is not hiding a lock |
| Affine search of the 16-32 residual vs the flat template | 2026-08-23 | 11 `gpt-image-2` photos: zero-deg median 0.009, best rotation in ±4° 0.020, lift 0.011. Six COCO noise residuals: lift 0.008. Flats already peak at 0°. No hidden rotation or 3% scale lock |
| Fourier-angle scramble of 16-32 (true `(r, θ)` of the annulus) | 2026-08-23 | s1 24.1 dB official `not_detected`, cartesian replicate 24.8 dB `not_detected`. Decoder reads phase in that annulus, including angular organization. Not quieter than scramble |
| Baker-map of the 16-32 band | 2026-08-23 | s1 27.8 dB still `detected` |
| Poisson / photon-shot embedding | 2026-08-23 | Flat Anscombe LOO 0.7518 vs linear 0.7519. s1 Poisson 30.7 dB still `detected` |
| Nested LSB plus SynthID | 2026-08-23 | s1 55.9 dB still `detected`. Extra open stego does not hide or replace the mark |
| ICC sRGB rewrite | 2026-08-23 | No-op, PSNR inf. Mark is not in a profile that a pixel-preserving assign changes |
| Seam carve 8 columns of 16-32 energy | 2026-08-23 | s1 27.7 dB still `detected` |
| Extra OpenAI image ids as a new ensemble | 2026-08-23 | `gpt-image-1.5` and `gpt-image-1-mini` are encoder-off: C2PA `created, converted`, official `not_detected`, 16-32 corr vs the `gpt-image-2` gray stamp 0.003. Dated `gpt-image-2-2026-04-21` is the same family: C2PA `watermarked.unbound`, 16-32 corr 0.189. `chatgpt-image-latest` generate stays 403 until org verification |
| DIRE reconstruction error as a payload | 2026-08-23 | SDXL float32, 512 px. VAE RMS: gray 1.24, s1 9.15, photo edit 9.89, camera 11.84. DDIM DIRE RMS: gray 2.40, photo 30.9, s1 31.5, camera 33.0. Rank follows texture. Float16 DDIM NaN'd on MPS. Not a payload |
| HEIF / AV1 as a hidden carrier | 2026-08-23 | s1 HEIF q80 46.3 dB, q50 39.3 dB, AV1 CRF 32 37.2 dB, all official `detected`. Codec round-trip does not mint a new statistic |
| Print-scan simulation as the kill | 2026-08-23 | s1 24.95 dB still `detected`. Same PSNR class as 16-32 scramble, which kills. Generic 25 dB is not the mark |
| Mellin / log-polar \|F\| of the 16-32 residual | 2026-08-23 | Flat LOO 0.996-1.000, photo 0.996, COCO 0.991-0.995. The annulus envelope, not a payload. Scale chirps `r^{iα}`: flat 0.51, photo -0.27, COCO -0.58. No photo lock |
| Zernike n≤6 and PCET of the residual | 2026-08-23 | Flat LOO 0.77 / 0.75, same order as cartesian 16-32. Photo -0.007 / 0.044. COCO 0.23 / 0.35, above photos. Moments of the flat stamp, not a photograph decoder |
| Fresnel quadratic-phase of 16-32 (z=8,16,32,64) | 2026-08-23 | Flat LOO 0.750-0.753, photo 0.012, COCO -0.006. Unitary copy of the cartesian residual. No new shell |
| 8-bin Fourier orientation energy | 2026-08-23 | 0.999 flat, 0.966 photo, 0.995 COCO. Saturates |
| Chroma 16-32 as the payload | 2026-08-23 | Gray-flat Y LOO 0.682, Cb 0.064, Cr 0.025. s1 Y scramble 24.6 dB `not_detected`; Cb 45.0 dB and Cr 43.9 dB stay `detected`. Mark is in luma |
| One 90° Fourier sector of 16-32 | 2026-08-23 | s1 27.5 dB still `detected`. Decoder needs the whole annulus, not one wedge |
| Radial-phase-only scramble of 16-32 | 2026-08-23 | s1 25.0 dB `not_detected`. Phase as a function of radius, over the full ring, is enough to kill. Combined with Fourier-angle scramble, either polar coordinate of the annulus phase is a kill if the whole ring is hit |
| [`Rinne414/SynthID-detector`](https://github.com/Rinne414/SynthID-detector/tree/b08abff06b67c37db9dfa7c6ed63760a446d560f) fixed residual template | 2026-08-23/24 | The repository reports 0/33 held-out AI images and identifies its original result as self-correlation. Its pinned pre-May GPT-Image2 template first detected 0/11 current source-matched edits and 0/11 COCO sources. An exact repeat on 200 current OpenAI and 200 current Google images produced zero detections for all three published templates at the 0.08 cut; maximum GPT-Image2 score on OpenAI was 0.0167. Adding the expected template can spoof the public correlator, but that template is not present in held-out content. Independent repeat of the rejected fixed-template route, not a payload decoder |
| [`newideas99/gpt-image-synthid-detector`](https://github.com/newideas99/gpt-image-synthid-detector/tree/5495e09) CNN ensemble | 2026-08-24 | Exact frozen inference retained 172/200 OpenAI but accepted 110/200 Google at the published 0.5 cut. The training negatives are regenerated positives, while its confound probe transforms only positives and has no foreign-image control. A full-frame white-box attack drove its probability from 0.99995 to 0.25956 at ±2/255 and effectively zero at ±4/255, but the official OpenAI decoder stayed `detected` through ±8/255. Renderer/confound classifier, not a mark decoder |
| [`Ristellise/REGRET`](https://github.com/Ristellise/REGRET/tree/7d449034bf323987e7e608e7886e029ed20fd847) SPAM model | 2026-08-24 | At the published cut, exact model inference accepted 139/200 Google, 141/200 OpenAI, 214/600 disjoint ImageNet photographs, and substantial fractions of foreign generators. The apparent high-precision tail did not transfer |
| [`vordme2010/synthid-dataset`](https://github.com/vordme2010/synthid-dataset/tree/133a27088f6f4d695c79db9a1a70fa8e7fa3adad) Tier-1 SVM | 2026-08-24 | Rebuilt without loading joblib, using the repository's 2,000x33 numeric matrix, reference phases, noise template, seed-42 split, scaler, and RBF SVM. It accepted 1/200 current Google and no current OpenAI rows. Its published AUC above 0.999 distinguishes Gemini flats from generated derivatives, not current photographs |
| Generic GitHub heuristics and literal ports | 2026-08-24 | `hackerfactor/reverse-SynthID-C`, `BIRSAx2/ripmark`, and `xiaoyao9184/reverse-from-synthid` reproduce or wrap the already-rejected reverse-SynthID codebook. `AI-SCERN` uses uncalibrated annular energy, decoder-grid, and FFT-symmetry heuristics with no weights or held-out corpus. LSB flips, ±1 dither, and 100.5% resize tools publish no matching-oracle evidence. None adds an independent signal |

## Information budget on photographs (2026-08-21)

On 11 `gpt-image-2` photo edits, the flat-derived 16-32 template at full
amplitude against per-image whitening has median `d' = 0.93` (range
0.68-2.46). That is a best-case single-image AUC of about 0.75 even with a
perfect, perfectly aligned template. A P5 gate (FPR 0.1%, TPR 90%) needs
`d' >= 4.37`. The deficit is 4.7x in amplitude, 13.4 dB, *unless* the
photo-domain mark is louder than the flat measurement. Titration said it
is not: fish 16-32 additive jam stays `detected` through a=16 / 21.9 dB.

2026-08-23, without using that flat stamp as `G`: the aligned photo
residual in 16-32 is redraw, not a codeword. Median RMS 12.2 across 12
pairs; 2.69 on the five tightest. Neighboring octaves are as loud.
`.local-eval/synthid/prc-oklab-attack-2026-08-15/photo-band-amplitude-2026-08-23.json`.

Do not report an AUC from those 11 pairs as a detector result. With n=11
the standard error on AUC is about 0.12. Notes:
`.local-eval/synthid/prc-oklab-attack-2026-08-15/agent-detector-claude.md`.

## External literature (surveyed 2026-08-23)

Primary sources, not abstracts. Each row is mapped onto a closed or open
route in this campaign. Mechanism detail stays in
[synthid.md](synthid.md). Removal papers are on
[mark removal](synthid-removal-research.md). Classifier papers are on
[SynthID source classifiers](synthid-classifiers.md).

### Official mark, not a public decoder

| Source | What it is | Map to this campaign |
| --- | --- | --- |
| Gowal et al., [arXiv:2510.09263](https://arxiv.org/abs/2510.09263) | Post-hoc encoder `f` / decoder `g`. Detection logit is not payload recovery. SynthID-O (partner variant) 136 bits at 512x512. TPR at 0.1% FPR 99.98% aggregated, 99.97% on the hardest spatial crop+resize. Trains against sampled semantics-preserving transforms, including weak VAE regeneration. Production decoder unpublished | Matches the architecture we treat as keyed `x' = x + g(x)`. Explains why a two-pixel crop kills `pipeline_lattice` but not the official oracle, and why a 22.3 dB foreign VAE still reads `detected` |
| DeepMind [US12094474B1](https://patents.google.com/patent/US12094474B1/en) and continuation [US20250149048A1](https://patents.google.com/patent/US20250149048A1/en) | Residual U-Net encoder, separate decoder, optional key, encoder/decoder ensembles that need not recognize each other | Constraint, not a recipe. Ensemble non-recognition is why one recovered Google tile cannot be reused as an OpenAI detector (S4) |
| Dathathri et al., [Nature 634:818-823 (2024)](https://www.nature.com/articles/s41586-024-08025-4) | SynthID-Text: tournament sampling of LLM tokens, open-source | Different system. Image/audio/video remain proprietary |
| OpenAI, [advancing content provenance](https://openai.com/index/advancing-content-provenance/) (2026-05-19, audio 2026-07-31) and [content provenance API](https://developers.openai.com/api/docs/guides/content-provenance) | ChatGPT / API / Codex images carry C2PA plus SynthID. Audio from 2026-07-31. `POST /v1/content_provenance_checks`. `not_detected` does not rule out another vendor | This is the oracle. C2PA and SynthID are independent entries. Do not abuse the endpoint as an adaptive reverse-engineering loop |

### Keyless presence detectors in the literature

| Source | Claim | Caveat against our gates |
| --- | --- | --- |
| Ao et al., [arXiv:2603.06723](https://arxiv.org/abs/2603.06723) (AWPD / FSNet, SAFE@CVPR 2026) | Leave-one-algorithm-out presence detection. SynthID held out: FSNet Acc 0.894 / F1 0.886, ResNet-50 Acc 0.845 / F1 0.812, ConvNeXt V2 Acc 0.866. LSB and Patchwork both fail below 60%. Hypothesis: modern invisible marks share dense high-frequency spectral anomalies | UniFreq Table 3: SynthID is 2,000 images, all AIGC, zero photographs, from `imagen-4.0-fast-generate-001` only, resized to 256x256. No OpenAI. No 0.1% FPR. Closest published analog to E3/L1 distillation. Our TrustMark leave-one-encoder-out was chance (0.505 ridge). Their high-frequency commonality is the same energy that L1 failed to read as an OpenAI payload |
| `aloshdenny/reverse-SynthID` V3/V4; Google to [The Verge, 2026-04-14](https://www.theverge.com/ai-artificial-intelligence/911579/google-synthid-ai-watermarking-system-reverse-engineered) | Spectral codebook from averaged Gemini flats. Spokesperson Myriam Khan: "It is incorrect to say this tool can systematically remove SynthID watermarks." The author also said the bypass confuses the decoder rather than deleting a payload | Our pickle-free V4 audit: paired AUC 0.517, frozen Open Images 386/1000 accepted. Not 0.1% FPR |
| Krawetz, [Hacker Factor "Reversing SynthID"](https://www.hackerfactor.com/blog/index.php?/archives/1092-Reversing-SynthID.html) and ["Meta's Un-Stable Signature"](https://hackerfactor.com/blog/index.php?/archives/1098-Metas-Un-Stable-Signature.html) | Gemini chat TPR closer to 1/20 than the paper's 99.97%. Detector weak on flats. reverse-SynthID field accuracy ~70% vs the author's 90% | Gemini-app chat is not the OpenAI provenance API. Our gray `gpt-image-2` flats are `detected` on the official OpenAI oracle. Do not mix the two verifiers |
| vitotitto LAB-a logistic (community, tiny holdout) | Reported 97.7% AUC on 20/20 | Not a P5 gate. Ignore as a detector claim |

### Different embedding loci (not post-hoc SynthID)

These papers are often cited as if they were SynthID. They are not.

| Source | Locus | Why it is not this mark |
| --- | --- | --- |
| Gunn, Zhao, Song, [arXiv:2410.07369](https://arxiv.org/abs/2410.07369) (PRC, ICLR 2025) | Pseudorandom error-correcting code in the *initial diffusion latent* | Cryptographic undetectability is a latent-code property. SynthID-Image is applied after the VAE has already decoded pixels. Our OKLab "PRC-style" noise was a category error |
| Francati et al., [arXiv:2509.10577](https://arxiv.org/abs/2509.10577) (EuroS&P 2026) | Crop-and-resize flipped about half of PRC latent signs and blocked belief-propagation decode | Confirms PRC is origin-locked in latent space. Matches why a 2 px shift kills `pipeline_lattice` and does not kill OpenAI SynthID |
| Fernandez et al. Stable Signature; Wen et al. Tree-Ring | Fine-tuned VAE decoder, or ring constraints on initial noise | In-generation. Google's paper is explicit that SynthID-Image does not modify the generator |
| TrustMark ([arXiv:2311.18297](https://arxiv.org/abs/2311.18297)), HiDDeN, StegaStamp | Open post-hoc encoder/decoder pairs | Transfer from these is E3: chance on TrustMark. WAVES ([arXiv:2401.08573](https://arxiv.org/abs/2401.08573)) stress-tests this family, not production SynthID |

Do not train a student on AWPD's Imagen-only 256 crop and call it an
OpenAI SynthID detector. That would repeat L1's contamination: a
generator/size cue, not a payload.

### Adjacent literature, not SynthID

These papers are the published versions of the wild hypotheses and of
Model 1. They detect *something else*: a camera pipeline, a generator
fingerprint, an open watermark, or steganography. None of them is a
keyless OpenAI payload decoder.

**Classical spread-spectrum and open neural marks**

| Source | What it is | Map |
| --- | --- | --- |
| Cox, Kilian, Leighton, Shamoon, IEEE TIP 1997 | i.i.d. Gaussian vector in perceptually significant spectral components. Informed detector. Collusion-resistant by construction | Ancestor of keyed CDMA. Our length-255 m-sequence vs 16x16 fold was at chance because we did not have the key, and because OpenAI is content-adaptive, not a fixed Gold code |
| Zhu et al., HiDDeN, ECCV 2018 | Joint encoder / noise layer / decoder | Open pair. AWPD leave-one-out Acc 0.985. Transfer to SynthID is E3 |
| Tancik, Mildenhall, Ng, StegaStamp, 2020 | 100 bits at 400x400, trained through print-scan | Open pair. Survives recapture. WAVES: TPR at 1% FPR collapses from 1.00 to 0.01 under regeneration |
| Wen et al., [arXiv:2305.20030](https://arxiv.org/abs/2305.20030) (Tree-Ring, NeurIPS 2023) | Pattern in the *initial diffusion noise*, recovered by DDIM inversion | In-generation, origin-locked in latent Fourier space. Crop-invariant by design. Not post-hoc SynthID |
| Fernandez et al., Stable Signature, ICCV 2023 | Fine-tune the LDM VAE decoder so every decode carries a bit string | In-generation. Gowal is explicit that SynthID-Image does not modify the generator |
| Lin and Juarez, [arXiv:2506.10502](https://arxiv.org/abs/2506.10502) (USENIX 2025) | Public-knowledge attack that removes Tree-Ring | Confirms Tree-Ring is a different object with a different kill |

**Steganalysis as a presence detector**

Fridrich and Kodovsky Spatial Rich Models (TIFS 2012) and Boroumand,
Chen, Fridrich SRNet (TIFS 2018) detect sub-bit-per-pixel spatial
stego by high-pass residuals, with pooling disabled in the front of
SRNet so the weak signal is not averaged away. AWPD cites both and
says they drift on modern deep / generative marks. That matches our
wavelet/FFT single-image detector (AUC 0.653, 0 TPR at a clean cut)
and the 16-32 energy miss on photographs: a residual energy detector
without the matching key is steganalysis of a mark that was trained
not to look like LSB.

**Generator fingerprints in the Fourier domain**

Corvi, Cozzolino, Poggi, Nagano, Verdoliva,
[arXiv:2304.06408](https://arxiv.org/abs/2304.06408) (CVPRW 2023):
GAN, diffusion, and VQ-GAN images show spectral peaks and anomalous
autocorrelation; real vs synthetic differ in mid-high radial and
angular power. reverse-SynthID averaged Gemini flats and called the
peak a watermark codebook. Corvi's result says many generators leave
*some* peak. Our V4 Open Images 386/1000 is what a generator-fingerprint
detector looks like when you calibrate it as if it were a payload.

Yao and Juarez, [arXiv:2512.11771](https://arxiv.org/abs/2512.11771)
("Smudged Fingerprints"): 14 fingerprinting methods across RGB,
frequency, and learned features; removal attacks >80% white-box, >50%
black-box. A fingerprint you can see without a key is a fingerprint
you can wipe without a key.

### Image investigation and data hiding (any method)

These are not SynthID papers. They are the rest of the toolkit: how
people hide bits in pictures, and how people tell a picture was
touched. Several of our wild hypotheses already had a published form
here.

Hiding is not one problem. Cover modification (change an existing
image), coverless / generative (sample an image that already carries
the bits), and signed metadata (C2PA) fail under different attacks.

**Cover modification, classical**

| Source | Hide how | Detect / limit |
| --- | --- | --- |
| LSB, Patchwork (Bender et al., IBM SJ 1996) | Flip low bits, or luminance of random pixel pairs | AWPD Acc < 0.60. Sparse or ±1 amplitude. SRNet / FSNet average it away |
| Westfeld F5 (2001), Fridrich nsF5 | JPEG DCT coefficients, matrix embedding | Histogram attacks on F5; nsF5 was the shrinkage fix. Domain is the codec, not a mid-band residual |
| Pevny, Filler, Bas HUGO (2010); Holub and Fridrich WOW (2012); Holub, Fridrich, Denemark UNIWARD (IH&MMSec 2013 / EURASIP 2014) | Content-adaptive costs, bits placed in texture via syndrome-trellis codes (Filler, IH 2011) | The modern spatial/JPEG floor. Distortion is *designed* to look like the cover. A 16-32 energy detector is the wrong statistic |
| Cox et al. 1997 | Spread-spectrum in significant DCT | Keyed. Already mapped above |

**Cover modification, neural (hide a whole image, not 32 bits)**

| Source | What it does | Map |
| --- | --- | --- |
| Baluja, NeurIPS 2017, "Hiding Images in Plain Sight" | Full-resolution secret image distributed across all bits of a same-size cover | Capacity is the point. Not robust, not keyed provenance. HiDDeN later added a noise layer so the secret survives JPEG |
| Jing et al., HiNet, ICCV 2021 | Invertible neural net: hide and recover as one bijection | High-capacity lossless-ish hiding. Recovery needs the exact inverse, not an official oracle |
| Yang et al., PRIS, [arXiv:2309.13620](https://arxiv.org/abs/2309.13620) | Invertible net plus robustness modules | Same family, trained through distortion |

A HiNet-style student on OpenAI pairs would learn the *edit*, not a
SynthID codeword. That is the L1 contamination again.

**Coverless / generative hiding**

The cover is never modified because there is no cover. The sampler
emits an image whose latents already encode the message.

| Source | Locus | Map |
| --- | --- | --- |
| Yang et al., Gaussian Shading, [arXiv:2404.04956](https://arxiv.org/abs/2404.04956) (CVPR 2024) | Map bits onto Gaussian latents indistinguishable from ordinary noise. Recover by DDIM inversion | Sibling of Tree-Ring / PRC. Training-free, performance-lossless *for the generator*. Not a post-hoc pixel stamp |
| Peng et al., StegaDDPM (ACM MM 2023) and later LDStega | Bits in the diffusion sampling distribution | Spatial SRNet is the wrong detector (NS-DSer, [arXiv:2602.10219](https://arxiv.org/abs/2602.10219): move steganalysis into noise space) |
| CRoSS, Pulsar, MDDM | Message-to-noise projections | Same locus. A pixel scramble does not invert the sampler |

**Passive forensics (the picture was touched, no secret assumed)**

| Source | Cue | Map |
| --- | --- | --- |
| Krawetz, "A Picture's Worth", 2007 (ELA) | Re-JPEG at lower quality, subtract | Already measured: COCO 3.13, s1 1.97, gray stamp 0.49. Codec history, not a payload. Farid publicly called ELA as likely to mislabel originals as it is to catch edits |
| Farid, IEEE TIFS 2009, JPEG ghosts | Difference energy vs a sweep of JPEG qualities; spliced regions ghost at their original Q | Local: s1, a `gpt-image-2` photo, and a camera JPEG all minimize at Q90. Codec, not a payload |
| Popescu and Farid, TR2004-515 | Copy-move via duplicated regions | Not generation, not a watermark |
| Popescu and Farid, IEEE TSP 2005 | Resampling periodic correlations | Affine search cousin. A rotated SynthID residual is a different question |
| Wang et al., DIRE, [arXiv:2303.09295](https://arxiv.org/abs/2303.09295) (ICCV 2023) | Diffusion reconstruction error: generated images reconstruct, cameras do not | Model 1 sibling. SDXL float32 at 512: DIRE RMS camera 33.0, s1 31.5, photo edit 30.9, gray stamp 2.40. Texture rank, not a payload. Inverse of the foreign-VAE remover: there the mark survived 22.3 dB |
| Wang, Wang, Zhang, Owens, Efros, [arXiv:1912.11035](https://arxiv.org/abs/1912.11035) (CVPR 2020, CNNDetect) | One ProGAN classifier, heavy JPEG/crop aug, transfers to many CNNs | Ancestor of "train on one generator". Ojha showed the sink-class failure once diffusion arrived. We required Firefly for that reason |

C2PA is the non-pixel stack: a signed manifest, stripped by
`metadata --remove`. Durable Content Credentials (spec 2.4) add a
soft binding that can re-link a stripped file to a repository. That
is provenance, not hiding.

Do not train on ELA, JPEG ghosts, DIRE, or a HiNet reconstruction and
name the score SynthID.

### Waveforms that can live in a picture

A mark is a function on the pixel lattice. The literature does not
use one wave. It picks a basis whose symmetries match the attack it
fears, then hides a keyed coefficient vector in that basis. Cartesian
16-32 is one shell. Polar, scale, and diffraction are different
shells.

**Standing waves on a rectangle.** DFT / DCT / DST. A 2-D sinusoid
`cos(2π(ux + vy)/N)`. JPEG lives here. Our octave scramble destroys
one annular *radius* of these frequencies, not one orientation. A
Gabor packet is the same sinusoid windowed in space.

**Circular and log-radial waves.** Functions of `(r, θ)`, not
`(x, y)`.

| Basis | Wave | Invariance it buys |
| --- | --- | --- |
| Fourier-Mellin / log-polar (Ó Ruanaidh and Pun, Signal Processing 1998) | `r^{iα} exp(ikθ)` after a DFT magnitude | Rotation and scale become translations |
| Logarithmic radial harmonics (IH 2002) | Same family, added in pixels, detected by complex correlation | RST without going through the unstable log-polar resample |
| Polar harmonic transforms: PCET, PCT, PST | `exp(±i2π n r²)` and polar cos/sin | Rotation. Moments, not a dense codebook |
| Zernike / pseudo-Zernike | Orthogonal polynomials on the disk, radial part related to Bessel | Rotation. Classical moment watermark |
| Circular chirp (SPIE 6072, 2006) | Polar map of a 1-D chirp onto a ring | JPEG (tune chirp rate) plus rotation (the ring) |
| Tree-Ring (Wen 2023) | Concentric rings in the *latent* Fourier plane | Crop/flip by construction. Not a pixel wave |

The 2026-08-23 file named polar-1632 is cartesian annular phase
shuffle, a scramble replicate (`not_detected` at 25.6 dB). True
Fourier-angle scramble of the same annulus is `not_detected` at 24.1 dB.
Radial-phase-only scramble is `not_detected` at 25.0 dB. One 90°
sector plus its conjugate stays `detected` at 27.5 dB. Affine/rotation
search of the 16-32 residual against the flat template does not lock
(lift 0.011, COCO 0.008).

**Scale chirps.** Hyperbolic / Mellin monomials (arXiv:1208.5842):
real 1-D Mellin patterns tiled in 2-D. Run 2026-08-23:
`r^{iα}` coefficients flat LOO 0.51, photo -0.27, COCO -0.58.
Log-polar `|F|` and a 64-bin radial Mellin profile saturate on COCO
(0.99). Not a payload, and not quieter than cartesian 16-32.

**Directional packets.** Dual-tree complex wavelets (approximate
analytic wave), Gabor/Morlet, ridgelets, curvelets, shearlets,
contourlets, bandelets. Multiplicative spread-spectrum on curvelet
coefficients is a published detector-design paper, not a SynthID
decoder. Our wavelet summary AUC 0.653 already said a *generic*
packet energy is not the OpenAI payload.

**Optical diffraction, actual wave physics.**

| Transform | What the wave is | Map |
| --- | --- | --- |
| Fresnel | Quadratic phase `exp(iπ r² / λz)`, a radial chirp. Energy concentrates with distance | Holographic watermarks embed the diffracted field, not the picture |
| Fractional Fourier | Rotation of the time-frequency plane. Order is an extra key | Extra key space on top of DFT |
| Double random phase encoding (Réfrégier and Javidi, Optics Letters 1995) | Random phase at input *and* Fourier plane. Ciphertext is stationary white noise | Closest optical analog of a keyed `x' = x + g(x)` that looks like noise without both screens |
| Cascaded Fresnel / CGH | Two holograms at different depths | Hardware key. Not a pixel residual we can match-filter |

A Fresnel/FrFT student without the distance or the fractional order
is another unmatched filter. Fresnel at z=8, 16, 32, 64 on the 16-32
residual is a unitary copy of the cartesian band (flat LOO 0.75, photo
0.012). FrFT is the same unmatched-filter claim with an extra order
key; it does not isolate a photograph lock the four distances missed.

**Cyclostationary and clock-warped waves.** Periodic clock change
(Martin, Chabert, Lacaze, EUSIPCO 2005) turns a stationary host into
a cyclostationary one by a periodic time warp. Our H2 multiplicative
cyclostationarity test already put official positives *below*
controls. Do not rerun the same statistic in a new basis and call it
new evidence.

**How to read a candidate wave.** Name the coordinates `(x,y)` vs
`(r,θ)` vs `log r`, the generating function (sinusoid, chirp,
Bessel, Zernike), additive vs multiplicative, and the symmetry it
claims. Then pick one attack that kills that symmetry and one that
should not. Cartesian 16-32 scramble and a 2 px shift are done
(scramble kills, shift does not). True Fourier-angle scramble kills s1
at 24.1 dB. Radial-phase scramble kills at 25.0 dB. A 90° sector does
not. Y scramble kills; Cb/Cr do not. Affine, Mellin, Zernike, PCET,
and Fresnel of the 16-32 residual do not lock on photographs.

## Still open or incomplete

- Google: no public payload decoder and no usable programmatic oracle.
  The Gemini app remains quota-limited. Vertex was 403.
- `chatgpt-image-latest` is an optional third pair mint, not a blocker.
  `gpt-image-1`, `gpt-image-1.5`, and `gpt-image-1-mini` are encoder-off.
  `gpt-image-2` and dated `gpt-image-2-2026-04-21` are encoder-on, same
  16-32 stamp (corr 0.189 on gray). Latest would only matter if ChatGPT's
  current consumer id used a different ensemble. `images.generate` is 403
  until the organization is verified.
- Physical print-scan: Brother DCP-L2520DW is idle and accepting. No
  `scanimage`, no ImageCapture pyobjc. Recapture needs a human at the
  flatbed. Simulation at 24.95 dB stayed `detected`. Do not print
  unattended.
- Adding unmarked context (frame, pad, hstack, collage) to hunt a
  residual is a presentation gate, not a mark statistic. Those rungs
  belong with identity and layout below, and with
  [mark removal](synthid-removal-research.md).
- Camera-to-ChatGPT "unchanged crop" pairs are dirty counterfactuals.
  The instructed edit redraws; aligned PSNR on the frame pair was
  16.6 dB. Compare pixels only after registration and a frozen residual
  bound, as in the `gpt-image-2` red-dot row.

A Gemini app caption that an image "looks like a photograph" is not a
SynthID verdict. Use the provider pixel check or signed provenance.

## Wild hypotheses

Battery 2026-08-23, no official oracle. Local numbers:
`.local-eval/synthid/prc-oklab-attack-2026-08-15/wild-hypotheses-2026-08-23.json`.
Prepared attack rasters wait in `wild-attacks-2026-08-23/` for a later
verifier window. Jacobian / adaptive queries against
the OpenAI provenance oracle stay out: the endpoint forbids reverse-engineering.

### Tested locally, not a mark

| Hypothesis | Result |
| --- | --- |
| Chroma ordered by luma is a 1D payload | Correlation is an artifact of the sort |
| Mark is `g(x)` of a Retinex envelope | LOO unchanged vs the raw 16-32 residual |
| Nonlocal patch agreement in 16-32 | Far patches are not coupled |
| Quadratic phase coupling (bispectrum) | Statistic saturates on camera images too |
| CDMA Gold / m-sequence in the 16x16 fold | At chance for length 255 |
| Rows as a pitched waveform | COCO noise has the same peak-to-mean |
| The photo mark is the flat stamp at an unknown 16-phase | Exhaustive shift does not lift NCC |
| The photo mark is the flat stamp at a small rotation or 3% scale | Affine search lift 0.011, COCO noise 0.008 |
| Self-keyed predictor from a 16 px luma thumb | Already 0.48 vs L1, 0.50 vs COCO (`self-keyed.json`) |
| Global multiplicative cyclostationarity (H2) | Official positives below controls, 2026-08-14 |
| Mellin radial profile / log-polar \|F\| | Saturates on COCO (0.99). Annulus envelope |
| Scale chirps `r^{iα}` | Flat 0.51, photo -0.27, COCO -0.58 |
| Zernike n≤6 | Flat 0.77, photo -0.007, COCO 0.23 |
| PCET polar harmonics | Flat 0.75, photo 0.044, COCO 0.35 |
| Fresnel z=8..64 of 16-32 | Copy of cartesian LOO |
| 8-bin orientation energy | Saturates on COCO |
| Chroma 16-32 on gray flats | Cb 0.064, Cr 0.025 vs Y 0.682. Stamp is luma |

### Not a watermark, might still be an AI-vs-camera extra

Battery 2026-08-23, `remaining-hypotheses-2026-08-23.json`. Kodak corpus was empty here, camera arm is COCO plus the photo-pair sources.

| Hypothesis | Result |
| --- | --- |
| Absence of camera PRNU | Gray highpass RMS 0.25 vs COCO 14.6 was texture. Wiener residual on *photographs*: `gpt-image-2` edits 4.61 vs camera 8.05. Direction is camera-like leftover of the donor JPEG, not a missing sensor, and not a payload |
| JPEG ELA | Flat 0.49, s1 1.97, COCO 3.13. Compression history |
| CFA interpolation error (Bayer as detector) | Ratio photo-edit 0.117 vs camera 0.184 vs gray stamp 0.588. Weak camera vote, overlaps. Inverse of the Bayer remover, not SynthID |
| JPEG ghosts (Farid) | s1, a `gpt-image-2` photo, and a camera JPEG all minimize at Q90 with similar spreads. Codec, not a payload |
| PNG Paeth / filter-byte residual | `gpt-image-2` PNG is 99.9% Paeth. Camera re-saved PNG 73% Paeth, s1 74%. Export encoder fingerprint |
| Perfect-circle / text-edge rate | Hough circles per MP 385 (edits) vs 536 (camera). Edge fraction 0.052 vs 0.072. Too noisy for a payload or a provider |
| Face-gated 16-32 | s1: 3 Haar faces, inside RMS 12.6 vs outside 10.9. One photo with faces: outside/in 1.49. No systematic "spend the mark off faces" |
| CLIP-space watermark | That is Model 1. Not a payload decoder |

### Non-local, 2026-08-23

Battery `.local-eval/synthid/prc-oklab-attack-2026-08-15/nonlocal-2026-08-23/`.
Oracle after AI-metadata strip, 25 s gap, then one dated-id retry after a 429.

| Hypothesis | Result |
| --- | --- |
| HEIF q80 / q50 | 46.3 / 39.3 dB, official `detected` |
| AV1 CRF 32 still | 37.2 dB, `detected` |
| Print-scan simulation (0.35 deg, blur σ 0.9, 300/360 dpi, JPEG q42) | 24.95 dB, still `detected`. Same PSNR class as 16-32 scramble, which kills |
| Physical print-scan | Brother DCP-L2520DW idle and accepting. No `scanimage`, no ImageCapture pyobjc. Not printed unattended |
| `gpt-image-1.5` gray reproduce | C2PA `created, converted`. Oracle `not_detected`. 16-32 corr vs `gpt-image-2` stamp 0.003. Encoder-off. Dirty redraw 24.0 dB |
| `gpt-image-1-mini` gray reproduce | Same C2PA. Oracle `not_detected`. Corr 0.003. PSNR 31.1 dB |
| `gpt-image-2-2026-04-21` gray reproduce | C2PA `created, converted, watermarked.unbound`. Official `detected`. 16-32 corr vs unversioned `gpt-image-2` stamp 0.189. Same encoder family. PSNR 47.5 dB |
| `chatgpt-image-latest` generate | 403, org verification. Listed among image models, not callable |
| DIRE SDXL VAE + DDIM float32 | VAE RMS 512: gray 1.24, s1 9.15, photo edit 9.89, camera 11.84. DIRE RMS: gray 2.40, photo 30.9, s1 31.5, camera 33.0. Texture rank, not a payload |

### Still blocked after that battery

| Hypothesis | Why it stays blocked |
| --- | --- |
| Physical print-scan recapture | Human at the flatbed |
| Encoder ensemble that does not recognize itself | Needs another actual SynthID encoder/decoder pair. Extra OpenAI ids did not mint one. Open-proxy transfer is already closed: source-matched TrustMark, VideoSeal, DWT-DCT, and WAM pairs produced held-out-family AUC 0.437-0.562, so another generic public pair does not answer this hypothesis |
| `chatgpt-image-latest` as a third pair mint | `images.generate` 403 until org verification. Optional: only if ChatGPT latest used a different ensemble than `gpt-image-2` |

Do not train on official decoder bits beyond the frozen L1 study. Do not
mint a 128-photo student on the flat stamp. The information budget and
the 2026-08-23 residual measurement both say the photo-domain mark is
under the scene.

## Oracle and seeds

Official `POST /v1/content_provenance_checks`, metadata stripped, pixels
preserved. Raw files stay gitignored under
`.local-eval/synthid/prc-oklab-attack-2026-08-15/`.

| Seed | Size |
| --- | --- |
| s1 | 1024x1536 |
| s2 | 1024x1536 |
| s3 | 1086x1448 |

All three `detected` before attacks.

## Source-matched pairs (2026-08-21)

`gpt-image-2` (ChatGPT Images 2.0) can mint encoder-on/off pairs.
`chatgpt-image-latest` is optional and 403 until org verification.

| Input | C2PA | Oracle | Aligned PSNR |
| --- | --- | --- | ---: |
| Camera JPEG 960x1280 | none | not_detected | - |
| ChatGPT UI + frame 1086x1448 | watermarked.unbound | detected | 16.6 dB (redraw) |
| `gpt-image-1` edit, high fidelity, red dot | created, converted only | not_detected | 30.9 dB |
| `gpt-image-1.5` gray reproduce 1024 | created, converted only | not_detected | 24.0 dB (redraw) |
| `gpt-image-1-mini` gray reproduce 1024 | created, converted only | not_detected | 31.1 dB |
| `gpt-image-2` edit, red dot 1024x1536 | watermarked.unbound | detected | 28.2 dB |
| `gpt-image-2-2026-04-21` gray reproduce 1024 | watermarked.unbound | detected | 47.5 dB |

The `gpt-image-2` photo residual is broadband (std 9.1), not a 16-32 stamp.
COCO `gpt-image-2` edits: 11/12 `watermarked.unbound`, one safety block.

Solid gray/white through `gpt-image-2` "reproduce exactly" is
`watermarked.unbound`. Gray 128 at 1024x1536: PSNR 41.9 dB vs the flat,
source `not_detected`, output `detected`. Other gray levels 32-220: PSNR
38-49 dB. Eight gray 1024x1536 residuals, leave-one-out correlation after
highpass: 0.645 (shuffle 0). That shared stamp does not transfer to COCO
photo residuals (0.007-0.025).

## Presentation gate versus residual

An official `not_detected` is not one event. Collage, photo-pad, two-panel
self-concat, and 8x8 scramble can be a *presentation gate* (this upload is
not a single photograph). Blur sigma 7 and in-band 16-32 destruction sit on
the single-photo manifold and are residual damage; those belong in
[mark removal](synthid-removal-research.md).

Do not train a surrogate on mixed collage misses if the goal is a mark
student.

Identity (s1, 2026-08-19): JPEG q80/q20, crop 2 px, crop 8 px, keep 0.75
and 0.50, rot90, rotate 5, flip, grayscale all stay `detected`.

Token / layout (s1): each quadrant at 512x768 `detected`; `hstack` and
`vstack` of marked\|marked `not_detected`; 4x4 tile scramble `detected`;
8x8 `not_detected`. Two copies of a detecting image still miss. Tomography
3x3: all nine cells `not_detected` (a 1/9 window is below support).

Preprocess E1 (s1): stretch 2x `detected`; centre-crop of marked\|marked
hstack back to native `detected`; 0.20x pixels on a native-size white
canvas `not_detected` (the same 0.20x file uploaded alone was `detected`).

## L1 is not a mark task

A missing `watermarked.unbound` assertion is not a clean negative. On
2026-08-16 the official verifier called 65 of 94 OpenAI rows without that
assertion `detected` (69%), with 9 of 9 interleaved health positives
detected so the endpoint was answering. Google's own API emits watermarked
images with no assertion, per its documentation. Corpus AUCs that treat
"no assertion" as unmarked rest on a negative class that is roughly
two-thirds positive.

Control-only 283 rows: 203 detected, 80 not_detected. Forward-temporal
nested ridge on 124 pixel features: AUC 0.649 / 0.641. Geometry-only:
0.781 / 0.826. All 21 `claim_generator=ChatGPT` controls are
`not_detected`. C2PA `opened`: 24/24 `not_detected`. Inside 1254x1254,
mean-feature permutation p=0.45.

OpenAI-supervised ridge on 16-32 band-passed `image_features`:
forward-temporal AUC 0.53 vs L1 `not_detected`, 0.95 vs COCO, 0.97 vs COCO
after a 2 px crop. The student learns OpenAI-versus-photo and still does
not see the oracle mark contrast.

## Runtime product remainder for the watermark

Signed provenance (`identify`). The remote OpenAI oracle is development-only,
and a research lattice miss is not a clean SynthID negative.
