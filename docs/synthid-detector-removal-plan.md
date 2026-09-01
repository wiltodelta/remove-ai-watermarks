# SynthID detector and pixel-only removal research plan

> Chronological mixed archive of detector, classifier, and removal work.
> Not a statement of current product capability. Read the split pages
> first:
>
> - [SynthID local detector research](synthid-detector-research.md)
> - [AI-generated image classifiers](ai-generated-image-classifiers.md)
> - [SynthID source classifiers](synthid-classifiers.md)
> - [SynthID mark removal research](synthid-removal-research.md)
>
> Shipped behavior remains in [supported signals](supported-signals.md),
> [known limitations](known-limitations.md), and
> [module internals](module-internals.md). Keep this file for dated H-gates,
> corpora, and session notes. Do not add new campaign results here; put
> them on the matching split page.

## Objective

Build two independent provider tracks with four capabilities:

1. a local, metadata-independent detector for the OpenAI SynthID image signal;
2. a local, metadata-independent detector for the Google SynthID image signal;
3. a pixel-only remover for the OpenAI signal that does not use diffusion,
   VAE reconstruction, semantic regeneration, or generative inpainting;
4. an independently calibrated pixel-only remover for the Google signal under
   the same constraints.

OpenAI is the first research track because it has a documented remote verifier.
Google follows with the same experimental protocol but its own corpus, labels,
model, thresholds, and oracle. No carrier, feature, score, or operating point
transfers between providers until a held-out experiment demonstrates that it
does.

The detector target is signal presence, not payload recovery, provider
classification, or general AI-image classification. The removal target is a
minimal pixel residual that makes a source-positive image negative in the
matching provider oracle while preserving the image's geometry and semantics.

## Non-negotiable evidence rules

1. **Detector before remover.** A remover may be prototyped against synthetic
   carriers, but no real-image removal claim is made until the local detector
   passes its own held-out gate.
2. **Provider-specific ground truth.** OpenAI and Google labels come from their
   matching verifier. C2PA is recorded separately and is never the pixel label.
3. **No metadata or export leakage.** Training and evaluation operate on decoded
   pixels after controlled metadata stripping and matched re-encoding. Geometry,
   filename, file size, chunks, encoder settings, and source directories cannot
   be model inputs.
4. **Signal identifiability is a gate.** A classifier trained only on provider
   positives and unrelated negatives can learn the provider's generator or
   export fingerprint. It is called a provider classifier, not a SynthID
   detector, until at least one causal control succeeds.
5. **The oracle is held out from optimization.** Candidate algorithms and
   hyperparameters are selected locally. Oracle batches are immutable and
   registered before submission. A remote binary verdict is never used as an
   online loss or hill-climbing signal.
6. **A local score decrease is not removal evidence.** Removal requires a
   source-positive, output-negative result from the matching provider oracle.
7. **One hypothesis does not become several signals.** Correlated spatial,
   spectral, and color statistics derived from one residual are reported as one
   line of evidence unless independent controls separate them.
8. **Every result is reproducible.** Input and output hashes, code revision,
   model artifact hash, preprocessing, transform lineage, seed, score, threshold,
   quality metrics, oracle result, session, and timestamp are retained.

## Oracle boundary

OpenAI documents a synchronous Content Provenance API at
`POST /v1/content_provenance_checks`. For images it returns separate `c2pa` and
`synthid` entries and supports PNG, JPEG, and WebP. It remains a remote,
OpenAI-scoped verifier, not a released local decoder. The same documentation
explicitly says not to use repeated queries to reverse-engineer, remove, or
evade a watermark. Adaptive detector or remover research against that endpoint
therefore requires explicit OpenAI authorization or a separate research oracle
whose terms permit the work. Without that authorization, the OpenAI track may
develop local hypotheses but stops before oracle-driven calibration and removal
certification.

Google's Gemini verification flow is a Google-scoped oracle. Its documented
result can be detected, not detected, or unclear, and the consumer flow has a
small rolling quota. An unclear result is `indeterminate`, never a negative.

For every permitted oracle batch:

- verify the untouched source first;
- submit one file per request;
- record C2PA and SynthID outcomes independently when both are returned;
- preserve the exact submitted bytes and SHA-256 outside the public repository;
- retry only a transient failure on the same bytes;
- record `detected`, `not_detected`, `indeterminate`, or `refused` verbatim;
- submit a matched transform-only control before attributing a negative result
  to an experimental edit;
- reserve a final temporal holdout that no feature, threshold, or remover has
  seen.

## Data design

### Corpus layers

Each provider gets a separate corpus with five layers:

| Layer | Purpose | Required controls |
| --- | --- | --- |
| Verified positives | Learn and evaluate the real signal | Matching provider oracle, original bytes |
| Same-provider hard negatives | Separate watermark from generator identity | Same surface or model, oracle-negative |
| External hard negatives | Measure false positives | Cameras, scans, edited photos, other generators, synthetic graphics |
| Low-texture probes | Expose weak shared structure | Solid colors, gradients, ramps, checkerboards, sparse edges |
| Causal pairs | Attribute a residual to the watermark | Same underlying pixels, positive and confirmed negative |

The causal-pair layer is the most valuable and the hardest to obtain. An
authorized encoder-off pair is ideal. A provider output and a pixel-only
processed version become a usable pair only after the original is positive and
the processed bytes are negative in the provider oracle. Public third-party
pairs are discovery material until their provenance and both labels are
independently verified.

A negative created by a remover trained against detector A cannot train or
validate detector A in the same experiment. Reserve it for detector B or a later
model epoch after closing the originating experiment. Otherwise detector and
remover can certify each other's shared blind spot.

If no same-provider hard negatives or causal pairs can be obtained, learned
real-image models stay explicitly labeled as provider classifiers. Spectral
repeatability on solid fills alone does not clear this gate.

### Strata

Record and split by:

- provider, product surface, model family, and generation date window;
- native width, height, aspect ratio, file format, and color mode;
- photoreal, face, text-heavy, flat graphic, illustration, low texture, and
  high texture content;
- untouched, metadata-stripped, lossless-normalized, JPEG/WebP, resized,
  cropped, and color-adjusted lineage;
- generation session and prompt family;
- parent hash for every derivative.

Store research media under `.local-eval/synthid/`, not in the public repository.
Only cleared fixtures may enter `data/synthid/`. Tracked files contain schemas,
scripts, synthetic fixtures, aggregate verdicts, and non-sensitive hashes.

### Split discipline

Deduplicate by decoded-pixel hash and perceptual similarity before splitting.
Keep every derivative, prompt sibling, and semantic near-duplicate in one hash
group. Split by group into train, validation, locked test, and a later temporal
test collected after the detector is frozen. A random image-level split is
invalid because it leaks transform and generation-family fingerprints.

The negative set must be large enough to support the claimed operating point.
At zero observed false positives, roughly 3,000 independent negatives are
needed merely to put the one-sided 95% upper bound near 0.1% by the rule of
three. The final evaluation should prefer at least 10,000 hard and ordinary
negatives per provider, report confidence intervals, and report every negative
stratum separately rather than hiding a weak stratum in an aggregate.

## Detector program

### Experiment D0: oracle and corpus integrity

Goal: prove that labels and bytes mean what the manifest says.

- Separate SynthID from C2PA in every response.
- Confirm that metadata stripping changes C2PA but does not silently define the
  SynthID label.
- Confirm every positive used for evaluation on its matching oracle.
- Mutation-test manifest ingestion with swapped provider, duplicate bytes,
  derivative leakage, an `indeterminate` result mislabeled as negative, and a
  changed file after hashing.
- Measure permitted same-byte verifier reproducibility on a small preregistered
  sample. Do not turn retries into adaptive querying.

Gate: no ambiguous label path, no cross-provider oracle substitution, and no
train/test family leakage.

### Experiment D1: export and generator confounds

Goal: determine how accurately SynthID can appear to be detected when no
watermark-specific evidence is available.

Train deliberately confounded baselines on file/container fields, dimensions,
RGB thumbnails, and generator-vs-camera content. Then repeat after canonical
decode, metadata removal, resolution matching, and hard-negative balancing.

Gate: any proposed signal feature must beat the canonicalized confound baseline
on same-provider hard negatives and on a temporal holdout. Otherwise the result
is generator attribution.

### Experiment D2: low-texture carrier discovery

Goal: test whether a repeatable carrier component is observable when scene
texture is suppressed.

Extend the existing `scripts/synthid_pixel_probe.py` measurements from grayscale
NCC to:

- per-channel and opponent-color residuals;
- two-dimensional FFT magnitude and circular phase coherence;
- wavelet bands and multi-scale autocorrelation;
- resolution and aspect-ratio registration;
- cross-color, cross-session, cross-model, and cross-date agreement;
- matched clean synthetic fills and same-provider oracle-negative probes.

Use leave-one-color, leave-one-session, and leave-one-resolution-out tests.
Fixed bins discovered and evaluated on the same images are descriptive only.

Gate: a template learned on one subset must detect held-out positive probes above
matched negatives and survive a later collection window. Failure kills the
fixed-carrier branch but not the content-dependent branch.

### Experiment D3: classical real-image detector

Goal: establish the strongest interpretable baseline before a neural model.

Candidate features include locally normalized high-pass residuals, FFT and
wavelet energy ratios, circular phase coherence, color-channel agreement,
block periodicity, and correlations against provider/resolution templates.
Fit regularized logistic regression and a shallow tree model. Calibration uses
only validation data.

Report TPR at 0.1% FPR as the primary metric, with bootstrap confidence
intervals. AUROC, average precision, and an arbitrary accuracy percentage are
secondary. Also report worst-stratum FPR, temporal-holdout TPR, and score drift.

Gate: advance the classical detector only if the locked test shows a stable
watermark-specific advantage over confound baselines. Do not choose a threshold
from the locked test.

### Mechanism hypotheses for D3-D5

The next detector epoch tests these hypotheses as one preregistered program:

1. **Counterfactual labels.** Train and calibrate on source-matched
   clean/watermarked examples, ideally the same underlying image before and
   after encoding. A result that vanishes when source, date, codec, and
   dimensions are balanced is a provider fingerprint, not watermark evidence.
2. **Canonical full-field evidence.** Preserve a 512x512 image-level field,
   absolute residual amplitude, and chroma alongside normalized residuals.
   Compare it against the frozen patch baseline on source-disjoint pairs.
   Failure to improve paired low-FPR detection rejects the added global context.
3. **Registration.** Score several bicubic canonical views spanning small scale,
   crop, and offset changes, then aggregate before calibration. This tests
   whether synchronization is distributed across the image rather than fixed to
   one global phase or local grid.
4. **Independent detection head.** Learn one presence logit directly. Treat
   payload-like or phase-consistency heads as auxiliary evidence, not as the
   presence decision. Their inclusion must improve a held-out paired test, not
   only attack-surface scores.
5. **Symmetric transformation channel.** Apply the identical sampled codec,
   resize, crop, color, noise, or overlay operation to both members of a pair. A
   transform is admitted only when the reference decoder, where available,
   confirms that the transformed positive remains valid.
6. **Encoder versions.** Compare one universal model with version- or
   epoch-specific experts on a cross-version transfer matrix. A version split is
   useful only if it improves held-out likelihood without source metadata at
   inference.
7. **Content-dependent watermarkability.** Measure flat, low-contrast,
   monochrome, logo, and pixel-art strata separately. Allow a
   content-conditioned expert or abstention instead of forcing one global
   operating point.
8. **Two-sided calibration.** Calibrate empirical evidence for both `not
   watermarked` and `watermarked`, returning positive, negative, or abstain.
   External generator corpora enter afterward as an untouched false-positive
   challenge, not as a substitute for counterfactual negatives.

### Experiment D4: learned residual detector

Goal: learn watermark presence from counterfactual image-level evidence that a
fixed template misses.

The primary model uses a full canonical field with raw RGB or luminance/chroma,
absolute-amplitude fine and coarse residuals, and optional stationary-wavelet
and complex-frequency branches. Every optional representation is encoded
separately and fused late; early channel concatenation is not a valid ablation.
Locally normalized patch evidence remains a frozen baseline, not the primary
input. Multi-view registration is aggregated into one image-level presence
logit. Payload-like, phase, localization, and content-watermarkability heads are
auxiliary and must prove an incremental held-out benefit.

Use group-aware pair sampling and apply every sampled transformation
symmetrically to the clean and watermarked members. Never expose metadata,
paths, native dimensions without normalization, or encoder-specific byte
patterns to the network. If causal pairs remain unavailable, do not advance a
learned model from provider classification to this experiment.

Train provider-specific models first. A shared backbone with provider-specific
heads is a later ablation, not the default architecture. Keep a second detector
family completely outside remover training so it can reveal surrogate overfit.
Compare a universal head with version-specific experts, and calibrate the
selected model with two-sided empirical evidence so ambiguous inputs abstain.

Gate for a detector release candidate:

- empirical FPR at or below 0.1% on the locked negative set;
- one-sided 95% TPR lower bound at or above 90% on untouched positives;
- no declared hard-negative stratum above 0.5% FPR;
- useful TPR on the temporal holdout and transformation suite;
- meaningful discrimination on causal pairs or same-provider hard negatives;
- calibrated `abstain` behavior outside supported providers and strata.

These are research gates, not promises that the proprietary decoder's operating
point has been reproduced.

### Experiment D5: robustness and drift

Evaluate identity, JPEG and WebP, resize, crop, padding, rotation, color changes,
blur, noise, overlays, screenshots, and combinations. Preserve a matched
transform-only positive control for every attack family. Report both average
and worst-transform TPR at the frozen threshold.

Repeat a small fixed collection after provider model or surface changes. A
shift in score distribution opens a new model epoch; it does not silently
recalibrate the old threshold.

## Localization program

A whole-image detector is not automatically a useful removal loss. Test whether
its evidence is spatially causal with three independent methods:

- tile occlusion and replacement with distortion-matched controls;
- detector-gradient attribution for the learned model;
- phase or template energy mapped back to spatial blocks.

For each source, create top-ranked, random, and bottom-ranked edits with equal
pixel norm and the same codec path. The oracle batch is registered before any
results return.

Gate: top-ranked edits must reduce matching-oracle detection more often than
random edits under paired analysis. Aggregate per-image differences and confirm
the direction with a sign test. If localization does not transfer, do not build
a region remover around it.

## Pixel-only removal program

All candidates preserve image dimensions and avoid a generative decoder. Run
the following ladder in order.

### Experiment R1: analytical carrier subtraction

Estimate provider, model-epoch, and geometry-specific residual components from
verified probes or causal pairs. Test:

- complex FFT projection with conjugate symmetry;
- wavelet-band projection;
- per-channel and opponent-color residual subtraction;
- spatially varying strength based on local texture and detector attribution.

Sweep signed amplitude, not only attenuation. Include a sham edit with identical
pixel norm outside candidate bins. This is the highest-value path because a real
shared post-hoc residual could be removed at very high fidelity.

### Experiment R2: constrained per-image optimization

Optimize pixels against an ensemble of frozen local detectors. The objective
combines detector margin with L-infinity and L2 bounds, LPIPS or DISTS, MS-SSIM,
edge consistency, OCR preservation, and face-embedding preservation where
applicable. Expectation over transformations covers lossless export, JPEG/WebP,
small resize, and color conversion so the result is not a fragile local
adversarial example.

The optimization may read only local detectors. The provider oracle evaluates
a frozen candidate batch afterward and never supplies gradients, search
direction, or per-step feedback.

### Experiment R3: feed-forward residual remover

If R2 transfers to the provider oracle, distill successful minimal residuals
into a compact image-to-residual network. Constrain the output amplitude and
frequency distribution explicitly. Train on one detector ensemble and select
on the held-out detector family. Preserve the per-image optimizer as the
reference implementation.

### Experiment R4: detector-remover co-evaluation

Evaluate four combinations separately:

1. local detector positive, provider oracle positive;
2. local detector negative, provider oracle positive;
3. local detector positive, provider oracle negative;
4. both negative.

Case 2 is the critical surrogate-overfit failure. Add every permitted example
to a future hard set only after the current experiment is closed; never tune and
report on the same oracle failure.

## Removal success gate

A provider-specific remover is a release candidate only when a locked,
source-positive evaluation shows all of the following:

- at least 90% matching-oracle `not_detected` results, with `indeterminate`
  counted as failure;
- a positive matched transform-only control for every evaluated source;
- exact-size output with no semantic regeneration stage;
- median PSNR at least 40 dB and fifth-percentile PSNR at least 35 dB;
- median SSIM at least 0.99, plus LPIPS or DISTS reported rather than optimized
  silently;
- no OCR regression on text strata and no material face-identity regression on
  face strata under preregistered thresholds;
- no worse oracle-negative rate after a standard downstream JPEG/WebP/resize
  suite;
- byte-identical pass-through for detector-negative inputs by default;
- a significant paired advantage over random and norm-matched sham edits;
- no claim of forensic cleanliness without a separately trained removal-artifact
  detector and a held-out evaluation.

If pixel-only methods fail this gate, retain diffusion regeneration as the
explicit fallback. Do not combine its success with pixel-only results.

## Provider sequence

### OpenAI first

1. Resolve the oracle authorization gate.
2. Build the corpus schema and confound challenge.
3. Verify or reject the shared-carrier hypothesis on low-texture probes.
4. Train classical and learned detectors.
5. Freeze the detector and temporal test.
6. Run R1, then R2, then R3 only after each preceding gate passes.
7. Submit one preregistered final oracle batch.

OpenAI work establishes the experimental machinery, not parameters for Google.

### Google second

Repeat the full sequence with Google-native positives, Google hard negatives,
and the Gemini oracle. Spend the small manual oracle budget on controls and
decisive boundary points, not uniform sweeps. Use local detector uncertainty to
choose a batch before submission, then freeze it. Test Gemini app and AI Studio
surfaces as separate strata because their export and metadata paths differ.

## Implementation order

The research harness remains outside the public API until the gates pass.

1. **Implemented:** use the research corpus schema and auditor documented in
   [`data/synthid/research-manifest.md`](../data/synthid/research-manifest.md) to
   record provider, surface, model epoch, session, content stratum, parent hash,
   transform lineage, separate C2PA and SynthID outcomes, oracle session, and
   artifact hashes.
2. **Implemented:** build a label-free local inventory before promotion so byte-identical files,
   decoded-pixel duplicates, and unsupported formats are visible without
   inferring evidence from directory names.
3. **Implemented:** add a corpus auditor that rejects hash-group leakage, missing parent links,
   ambiguous labels, and unsupported oracle-provider pairs.
4. **Harness implemented; evidence run pending:** run the manifest-driven D1
   challenge over container, thumbnail, and canonical decoded-content
   baselines. Freeze its validation threshold and report same-provider negative
   cohorts separately.
5. Generalize `scripts/synthid_pixel_probe.py` into reusable feature extraction
   while preserving its current synthetic tests.
6. Add reproducible train/evaluate commands whose output is a versioned model
   card and metrics snapshot, never an unversioned console claim.
7. Add analytical and optimization removal harnesses with norm-matched controls.
8. Reuse the existing fidelity scripts for PSNR, SSIM, OCR, face, and edge
   measurements, adding only missing metrics.
9. Package provider-specific detector weights behind an optional dependency only
   after the detector gate passes.
10. Add a runtime remover and CLI surface only after the removal gate passes.

Pure feature, manifest, split, threshold, and residual-constraint logic must be
unit-tested without model downloads. Real model and oracle runs stay explicit
research jobs.

The inventory, manifest auditor, and D1 confound harness now exist as local
research tools. D1 has not produced a real-corpus metric yet because existing
artifacts have not been promoted into evidence-bearing provider manifests. This
is an evidence gap, not permission to infer labels from their paths. The next
real D1 run begins only after ordinary rows cover train, validation, locked test,
same-provider hard negatives, and a two-class temporal holdout.

## Hypothesis register

What is open, what is closed, and what would move each one. A status changes only
against recorded evidence in the log below, never against an argument. `Refuted`
and `supported` name a scope: a hypothesis refuted for one provider or for one
question stays open for the others, and the scope column is the load-bearing part.

Statuses: `open` (never tested), `partial` (tested in one scope only), `supported`,
`refuted`, `running`.

### The labels

Everything else in this register is conditional on these. They were assumed for
months and tested for the first time on 2026-08-15.

| # | Hypothesis | Status | What moves it |
| --- | --- | --- | --- |
| L1 | A missing `watermarked.unbound` assertion means no watermark was embedded | `refuted` for both providers | Settled 2026-08-16. OpenAI: the official verifier called 65 of 94 non-asserted rows `detected`, a contamination rate of `69%`, with 9 of 9 interleaved health positives detected so the verifier was demonstrably answering throughout. Google: its own API emits watermarked images, per documentation that states so outright, carrying no assertion. Every corpus-labelled AUC in this document rests on a negative class that is roughly two-thirds positive |
| L2 | A watermark assertion means a pixel watermark is actually present | `partial` | Ten source controls returned `detected` across both providers. Ten of twelve thousand is a smoke test, not a rate. As of 2026-08-16 the verified-positive count is past thirty: Google web-verifier source arms across three accounts, the OpenAI provenance-API batches v1 and v2 (every asserted source detected), and 9 of 9 interleaved health positives. Still no denominator, so still not a rate. Extending the same runner to a frozen sample of asserted rows would give one. Update 2026-08-19: the first health miss (21/22) resolved as a bad label, not a verifier flip; the row `originals/2026-06-25/198ae69e...` is classified `openai/asserted` yet carries `claim_generator: Google C2PA Core Generator Library` and Google-side geometry, and the verifier's `not_detected` is the correct answer for it. The health design caught a corpus labeling error, which is the first direct evidence that assertion rows also carry at least one mislabeled member |
| L3 | Date and model version do not carry the label | `supported` | The matched sets fix size, generator and codec but not date, and 57 of 70 Google controls fall in one month. Tested directly on 2026-08-16: a date-only classifier scores AUC 0.524/0.491, so date carries nothing and the covariate re-run is no longer load-bearing |

### Where the signal lives

| # | Hypothesis | Status | What moves it |
| --- | --- | --- | --- |
| S1 | The periodic comb carries the watermark | `refuted` | Settled 2026-08-16. The structure every periodic expert reads is destroyed by a seven-pixel crop while the published mark retains 99.97% TPR under aggressive crop and resize, so it is a generation-pipeline lattice anchored at the image origin, not the watermark |
| S2 | The signal is the broadband content between the comb lines | `partial` | First direct oracle arm 2026-08-15: the equal-RMS broadband randomization (54.27 dB, local score unchanged by construction) remained `detected` by the Google verifier, as did every other arm of that account-isolated group. One image group at one perturbation class is a bound, not a sweep, but the between-lines content was randomized without moving the verdict, which is evidence against this scope. The local readout is blind between the comb lines by construction, so no local rerun can substitute; a wider amplitude ladder needs new provider quota |
| S3 | The signal is the shape of the normalized mid-band spectrum | `supported` for OpenAI, `open` for Google | OpenAI confirmed under nested family and lambda selection on 2026-08-15: `0.510` shipped to `0.605`, CI [+0.045, +0.147], above the permutation null. The Google branch was tested the same way on 2026-08-16 and did not improve: `0.678` against `0.689` shipped, CI [-0.071, +0.046]; at 63 controls a 0.02-0.03 gain is unprovable there in principle, so Google stays control-limited |
| S4 | The carrier is content-dependent, so no fixed key-derived pattern exists | `refuted` for Google, `supported` for OpenAI | Settled 2026-08-15 by cross-image tile correlation. Google carries one fixed phase-coherent pattern, present in the controls at about half the amplitude; OpenAI carries none at chance level. What remains open is whether the Google controls are unwatermarked at all, which is L1 |
| S5 | The signal needs a learned nonlinear representation, not hand statistics | `refuted` at the tested capacity | First slice 2026-08-17: a 4-layer conv on opponent-residual patches, translation-invariant by construction (unaligned random patch offsets), leave-one-era-out. Against photographs it reaches 0.6-0.94, against foreign generators 0.44-0.56 with reversals to 0.11-0.15 on fresh eras, at a permutation null of 0.54-0.59: it learns "AI render versus photograph", nothing vendor-specific. A larger model remains free to try but inherits the same leave-one-era-out and leave-one-vendor acceptance gate |
| S6 | Some hand-crafted statistic of the patch bank separates in every era, vendor and codec cell | `refuted` | Settled 2026-08-17 by the full-grid audit: of 124 features, zero clear minimum AUC 0.55 over 27 cells (best worst-cell 0.481), and a corpus-fitted ridge transfers to fresh eras at 0.388 native, 0.178 after JPEG-95 |

### Robustness

A detector that fails these is not shippable whatever its AUC.

| # | Hypothesis | Status | What moves it |
| --- | --- | --- | --- |
| R0 | A candidate expert survives a crop that is not tile-aligned | `refuted` for every expert built so far, including the shipped registered-v3 default | A seven-pixel offset takes the red-green agreement from `0.962` to `-0.111` at every size on large-v1, and on 2026-08-16 a two-pixel diagonal crop killed every one of the 36 registered-v3 detections across 28 foreign-generator acceptances and 8 Google provenance positives. This is now a precondition for calibration, not a later robustness check |

| # | Hypothesis | Status | What moves it |
| --- | --- | --- | --- |
| R1 | The patch-local feature survives JPEG, resize and crop | `partial`, and its numbers await clean labels | Re-score the matched set under each transform. The earlier patch-bag model kept one positive in 486 after JPEG-95, so this is a gate and not an improvement |
| R2 | The OpenAI result is not an artifact of PNG encoding or one geometry | `supported`, on contaminated labels | Repeat the evaluation inside a single exact size and a single month |

### Scope and universality

| # | Hypothesis | Status | What moves it |
| --- | --- | --- | --- |
| U1 | One feature space serves both providers | `refuted` at the current features | Transfer scores `0.535` and `0.469`. A shared representation, or an honest per-provider router with abstention |
| U2 | A watermark detector does not fire on non-watermarked AI images from other generators | `refuted` | Measured 2026-08-15 at feature level (the Google shipped feature accepts 34% of Adobe Firefly output) and 2026-08-16 through the production entry point: 29/223 signed non-Google generator images accepted, `0.130`, Firefly `0.241`. Photographic controls cannot bound this rate; see the two empirical entries below |
| U3 | The detector works on the distribution it will actually meet | `open` | Both classes are drawn from files that still carry C2PA. A re-saved image carries none, and that is the case a user brings |

### Method validation

These do not test SynthID. They test whether our method can find a watermark it
is known to contain, which is what separates "the features are weak" from "the
data is too small".

| # | Hypothesis | Status | What moves it |
| --- | --- | --- | --- |
| M1 | The pipeline recovers a watermark we embedded ourselves | `supported` for a fixed-structure mark at SynthID's amplitude | Open watermarkers, VideoSeal, Watermark Anything, Stable Signature, give unlimited matched pairs. A pipeline that misses those has a method problem, not a data problem. The 2026-08-15 dwtDct ladder adds the boundary: full strength AUC `0.990`, quarter strength (RMS `0.563`, our SynthID measurement range) `0.936`, so the pipeline is sound at the amplitudes we work at. The M1b/M1c ablation the same day bounds every hand-statistic branch: at equal RMS a fixed mark scores `0.670`, a keyed mark `0.669`, and a keyed perceptually masked mark `0.546`, and demasking does not recover it (`0.538`). Masking, not keying or content dependence, is what defeats hand features |
| M2 | Matched pairs can be minted rather than mined | `refuted` | Imagen, whose `addWatermark` the plan was built on, is absent from the model garden and every id 404s. The `gemini-*-image` models that replaced it reject the parameter, and the documentation states that all generated images carry a SynthID watermark. No current Google path can emit an unwatermarked image. Re-verified independently on 2026-08-16: `imagen-4.0-generate-001` and `imagen-3.0-generate-002` return 404 in us-central1 and us-east1 on both GCP projects, `gemini-2.5-flash-image` rejects `addWatermark` at request parsing (`Cannot find field`), and the current Gemini API documentation states for both the Gemini-image and Imagen paths that all generated images include a SynthID watermark, with no disable parameter documented. Settled on live API and documentation. Extended 2026-08-17 by the attack routes on the OpenAI side: pixel attacks to PSNR 20.1 (JPEG q40, 0.5 resize, sigma-8 noise plus JPEG q85, 0.35 resize plus JPEG q50) and a foreign-VAE decoder-substitution round-trip to PSNR 22.3 all left the official verifier answering detected, so attacking a verified positive cannot mint a verified negative on either provider by any mechanism tried |
| M3 | `folded_template_score` is correct | `supported` | Settled 2026-08-15: an independent reimplementation of the fold scored over the same sample agrees with the shipped implementation to `3.7e-4`, the residual traced to float32 in the cv2 blur. Removal and detection results still share the one implementation, but it now matches an independent one |

### The shipped decision

| # | Hypothesis | Status | What moves it |
| --- | --- | --- | --- |
| D1 | Some usable fraction of images can be decided at a precision-first threshold | `refuted` at one percent control acceptance | Asked and measured 2026-08-16: at one percent control acceptance the decidable positive fraction is 0-4.5%. A precision-first threshold that decides almost nothing is not a usable product surface |

## Empirical log

### 2026-08-09: D1 confound pilot and codec challenge

The manifest-driven D1 harness was run on the frozen Google pilot containing
five oracle-positive images and 330 deduplicated exact-geometry external
photographs. The manifest passed byte, decoded-pixel, lineage, and split
verification. It is not evidence-ready because the locked test contains no
same-provider hard negative.

The container-only baseline separated the labels perfectly because the pilot
still exposes format and export-geometry differences. That result is a measured
confound, not watermark evidence. The canonical 8x8 decoded-content baseline
was much weaker: locked-test AUC was 0.620 and temporal AUC was 0.533, and its
validation-frozen threshold detected neither held-out positive. The existing
positive-only RGB plus HSV S/V ensemble still detected all five positives and
emitted no positive verdict on the 330 frozen external images.

A codec challenge re-encoded every positive without changing geometry. The
ensemble remained positive on five of five JPEG-95 outputs and five of five
WebP-95 outputs; JPEG-90 retained three of five. This rejects a bare PNG versus
JPEG container explanation, but does not exclude a generator or source-pipeline
correlate. With only five positives, the one-sided 95% lower bound on TPR is
54.9%. With zero positives among 330 external images, the one-sided 95% upper
bound on FPR is 0.904%, still nine times the 0.1% detector target. Four positives
also participated in model fitting, leaving only one independent temporal
positive. The next valid detector claim still requires new oracle-positive
images and ordinary same-provider oracle-negative controls.

A post-freeze source-provenance challenge then added three exact-geometry
Google originals that had not influenced fitting or threshold selection. Each
carried the same signed Google LLC C2PA issuer, trained-algorithmic-media source
type, and explicit SynthID-present assertion as the detected temporal control.
The ensemble abstained on all three: two lacked active carrier support in both
branches, and the third passed RGB evidence but missed the HSV S/V evidence and
support gates. These are provider-signed embedding assertions rather than
matching-oracle pixel labels, but zero positives in three new same-geometry
images falsifies the current ensemble as a general Google SynthID detector. The
measured phase family is retained only as an epoch- or surface-specific
correlate pending a broader oracle-labeled corpus.

A follow-up cross-epoch check found that the correlate is not entirely confined
to the original five images. RGB and HSV models fitted only on the three later
images ranked each of the five earlier images above all 329 external negatives
in both color spaces (AUC 1.0), but thresholds derived from the later fitting
scores transferred poorly: the strict RGB-plus-HSV decision retained only one
of five earlier positives at zero false positives on the 279-image holdout. An
eight-fold leave-one-positive-out refit across both source groups then detected
six of eight excluded positives with the same RGB-plus-HSV conjunction and each
branch frozen just above its 50-image calibration maximum. RGB alone put all
eight excluded positives above that maximum, while HSV missed the geometric and
low-texture images; the per-fold RGB operating points also produced between one
and eight false positives on 279 held-out negatives.
This is evidence for a transferable but content- and epoch-sensitive Google
pixel correlate, not a shippable detector. The experiment still lacks ordinary
same-provider oracle negatives, and its external-negative set is too small for
the 0.1% false-positive target.

### 2026-08-09: fixed spectral-template baseline rejected

An exploratory Google template was reconstructed from four public, purported
clean/marked pairs. Their provenance and oracle status could not be established,
so they were used for discovery only. A scalar phase-consensus score and its
threshold were selected on those pairs, four older Google-oracle positives, and
30 external negatives.

The frozen threshold then produced six false positives on a new 100-image
external holdout. It also detected only one of four newly generated Gemini
images collected after threshold selection. Duplicate-image removal cannot
reduce the false-positive rate enough to approach the 0.1% detector gate, and
the temporal result is far below the required sensitivity. The baseline is
therefore rejected, not recalibrated on the holdout.

This result rules out the fixed phase template as a detector or removal loss.
It does not rule out a content-adaptive or model-epoch-specific signal. The
next detector must learn from independently labeled provider data, must retain
the failed holdout unchanged, and must demonstrate discrimination from export
format and generator identity.

### 2026-08-09: cross-color low-texture consensus rejected

A polarity-invariant consensus template was trained on black, white, and red
Gemini low-texture probes and frozen before evaluation. Its median score fell
from 0.574 on the training groups to 0.354 on held-out probes of the same
colors and 0.011 on unseen blue, green, and gray probes. External negatives had
a median score of 0.0047. The separation therefore depended on the training
colors and did not generalize across the intended low-texture stratum.

This branch is rejected as a current Google detector. It remains useful as a
negative control demonstrating why high training coherence is not evidence of
a shared carrier.

### 2026-08-09: external V3 codebook detector pilot

The numeric format-v2 V3 artifact from `reverse-SynthID` was loaded with pickle
disabled and evaluated independently; no third-party code was imported or
executed. A 256-bin phase score with a discovery-frozen threshold of 0.5
detected five of five Google-oracle-positive images and produced zero false
positives on 194 external images collected before and after threshold freeze.
The pilot includes four older positives and one newly generated temporal
positive. It is promising discovery evidence, not a released detector:

- zero errors on 194 negatives cannot support a 0.1% FPR claim;
- the set lacks enough same-provider, oracle-negative hard controls;
- the phase profile is a third-party artifact whose positive provenance is not
  independently reproducible from our current corpus;
- a secondary `active_weight_fraction` shortcut failed on an exact-resolution
  holdout, firing on 17 of 50 new negatives, and is rejected as a
  native-resolution/content confound.

The next detector milestone is to reproduce the phase carrier from our own
deduplicated positives, expand the negative set to at least the rule-of-three
minimum, and add same-provider hard negatives before fitting or publishing an
operating point.

### 2026-08-13: external V4 cross-color codebook audit

The later `aloshdenny/reverse-SynthID` V4 artifact was audited independently at
commit `b110836`. The audit read only numeric NPY members from its NPZ ZIP with
pickle disabled and reproduced the published native-resolution phase score;
no third-party Python or pickle payload was executed. Its bundled
`sample_watermarked.png` crossed the documented `phase_match > 0.52` threshold
at `0.535`, confirming that the reproduction follows the intended path.

The operating point did not transfer. Taking the better of the two published
model profiles detected 141 of 355 Google positives and 191 of 499 controls.
On a later frozen 1,000-image Open Images reserve it accepted 386 controls. The
phase score had AUC 0.517 on the paired corpus. Reproducing the July 2026 GUI's
0.3%-aspect-ratio gate reduced coverage without adding discrimination: it
supported 121 positives and detected 46, while supporting 343 controls and
detecting 132; it also detected 284 of 685 supported fresh controls.

The failure follows directly from the metric. Per-bin phase similarity is
`1 - abs(wrapped_difference) / pi`, whose expectation under independent uniform
phase is 0.5, only 0.02 below the threshold. The detector selects bins by
cross-color phase coherence alone and does not require image-side carrier
amplitude or a content-baseline excess. Its selected bins span nearly the full
spectrum, with a typical spatial period around 2.5 pixels. Cross-color
coherence is useful for discovering candidates from genuinely low-texture
references, but this single-image score is not a calibrated detector.
Two prespecified rerankings did not rescue it. Restricting carrier periods to
4-128 pixels and ranking by either coherence times reference magnitude or the
stored carrier weight times reference magnitude produced best-of-model AUCs of
0.521 and 0.524. They were evaluated as continuous scores without selecting a
new test-set threshold and are also rejected.

The repository's older V3 phases also failed this external challenge. Their
best dark/white phase score had AUC 0.473; the documented 0.78 phase threshold
accepted 5 of 355 positives, 5 of 499 controls, and 6 of the later 1,000 fresh
controls. The repository's current bundled sample and four validation sources
all remained below that phase threshold. The README's accuracy claim is
therefore not accepted as evidence for the current corpora.

The linked Hugging Face dataset cannot repair OpenAI calibration as published.
Its DALL-E 3 `black` and `white` buckets are not solid-color probes: per-image
mean luminance ranged from 11.66 to 143.74 and 124.94 to 235.66 respectively.
Across ten 1024-square images per bucket, only 39 and 32 RGB frequency bins had
phase coherence above 0.9, versus 20,443 and 100,252 for the corresponding
Gemini 3.1 buckets. Only four DALL-E bins cleared 0.9 in both colors and all
four were trivial DC or self-conjugate-axis coordinates; Gemini had 11,478
nontrivial joint bins and 3,341 also had cross-color absolute-cosine agreement
above 0.9. Those DALL-E folders measure heterogeneous content phase, not an
OpenAI solid-reference codebook. The official SynthID-Image paper also says
nearly uniform corner cases may be excluded from watermarking, so generating
flatter references does not guarantee a stronger watermark. A useful next
external experiment requires newly collected low-content OpenAI outputs whose
pixel watermark is separately confirmed, plus same-provider oracle-negative
controls, before freezing any phase bins.

### 2026-08-09: independently fitted phase-carrier pilot

An exact-geometry phase model was then fitted from four of our own
Google-oracle-positive images. The first unconstrained search was a clear
multiple-comparisons failure: training scores were 0.983-0.991, while the
newly confirmed fern positive scored 0.061 and an exact-resolution negative
reached 0.105. Leave-one-out discovery detected only two of four older
positives. That unrestricted model is rejected.

A second model used only the 256 frequency coordinates proposed by the numeric
V3 artifact, while relearning phase, weight, and expected magnitude entirely
from our four positives. This restricted model scored the fern validation
positive at 0.740 and the four fitting positives at 0.936-0.963. A threshold of
0.43 was frozen after 50 validation negatives whose maximum was 0.305.

The phase-only epoch failed its first new holdout with three false positives in
121 exact-size images. All three had active carrier weight between 0.07 and
0.14, so the failed epoch was closed and a new rule was frozen: phase score at
least 0.43 and active weight at least 0.50. That second epoch produced zero
false positives on 279 newly downloaded exact-size images.

A separate 3,000-image COCO check also produced zero joint-rule positives, but
it is not counted toward the supported-stratum FPR bound. Every COCO image had
been upscaled to the model geometry and every active weight was at most 0.201.
The rule therefore abstained because the inputs were outside its measured
carrier-support distribution. Treat active weight as an `abstain` gate, not as
negative SynthID evidence.

This is the first positive local-detector pilot derived from our labels, but it
does not clear the detector gate. It still depends on third-party candidate
coordinates, has only five confirmed positives including the fitting images,
has only 279 fresh negatives inside the current support stratum, and has no
same-provider hard negatives. The next epoch requires a larger independently
labeled positive set, at least 3,000 native-support negatives, same-provider
oracle negatives, and a new temporal positive that has not influenced feature
or threshold selection.

### 2026-08-09: color-space spectral comparison

The restricted phase-carrier experiment was repeated in RGB, full-range
YCbCr, YCoCg, an orthogonal opponent basis, CIE Lab, and HSV. Every branch
started from the same 102 spatial frequencies: the top 256 external codebook
coordinates contained 102 unique `(row, column)` pairs, which were expanded
over all three components. Each branch then independently selected 256 of the
306 component-frequency candidates and relearned phase, expected magnitude,
and weight from the same four oracle-positive fitting images. The score was
the phase score multiplied by active carrier weight, so weak spectral support
reduced rather than merely qualified the evidence.

Thresholds were frozen from the fern validation positive and 50 exact-size
validation negatives before scoring the 279-image comparison set. The
comparison set was locked for this color-space branch, although it had already
served as the second epoch's RGB negative set and is therefore not a globally
virgin corpus.

| Space | Validation gap | Comparison negative max | Fern minus negative max | False positives |
|---|---:|---:|---:|---:|
| RGB | 0.509 | 0.097 | 0.459 | 0/279 |
| YCbCr | 0.330 | 0.193 | 0.312 | 0/279 |
| YCoCg | 0.437 | 0.163 | 0.390 | 0/279 |
| Opponent | 0.351 | 0.167 | 0.339 | 0/279 |
| Lab | 0.408 | 0.126 | 0.382 | 0/279 |
| HSV | 0.530 | 0.084 | 0.495 | 0/279 |

HSV had the best observed worst-negative margin, narrowly ahead of RGB, but
did not generally shift paired negatives below RGB. After normalizing each
space by its fern score, the median paired HSV-minus-RGB difference was 0.0015
and a two-sided sign test gave `p=0.632` (144 higher, 135 lower). The result is
therefore a tail observation, not evidence that HSV dominates RGB.

Channel decomposition localized the useful effect. HSV hue contributed only
0.012 of fern evidence and its fern-minus-holdout-maximum channel gap was
-0.024; saturation and value contributed 0.333 and 0.233, with positive gaps
of 0.273 and 0.194. In YCbCr, YCoCg, opponent, and Lab, the luminance-like
channel supplied the largest and most stable separation, while chroma-channel
gaps were much smaller. This supports a carrier expressed across intensity and
contrast structure, with useful nonlinear saturation/value projections, not a
hue-specific watermark.

This experiment does not raise the detector's evidence grade. It still has
only one held-out confirmed positive, reuses the same third-party spatial
frequency prior, lacks same-provider hard negatives, and evaluates only one
native geometry. A future HSV S+V ablation must freeze its own operating point
and use a new negative epoch and new oracle-positive images; dropping hue now
would be a post-holdout model choice.

### 2026-08-09: pixel-only R1 pilot did not transfer

Three non-generative candidate families were frozen locally before Gemini
verification:

1. local visible-region inpainting changed 0.087% of pixels at 47.64 dB PSNR
   and 0.99963 SSIM;
2. sparse complex-spectrum subtraction used 256, 4,096, or 16,384 numeric V3
   bins while preserving native 2816x1536 geometry;
3. deterministic spatial fragmentation combined smooth local warps, one global
   resampling operation, mild color changes, and codec round-trips.

Local inpainting left the 256-bin phase score effectively unchanged. Healthy
Gemini sessions continued to detect both the target and its spatially shifted
sham, refuting an earlier one-off `not detected` response.

The widest spectral candidate reduced the frozen 256-bin phase score from
0.639 to 0.124 while retaining 51.94 dB PSNR and 0.99854 SSIM. A
pixel-identical, metadata-normalized source control was detected in two healthy
oracle sessions. The 16,384-bin candidate remained detected in one of them and
the second hit quota before returning a verdict. The 4,096-bin candidates also
remained detected. Therefore the external phase carrier is correlated with the
Google signal but is not a sufficient removal loss.

One additional profile returned `not detected` for the 16,384-bin candidate,
then also returned `not detected` for the untouched source. That entire session
is invalidated rather than counted as removal success. This failure adds a new
oracle-health rule: every account/session batch starts with the same
pixel-identical positive control; candidate verdicts are discarded when that
control is not positive. Quota and tool-unavailable responses remain
`indeterminate`.

The frozen fragmentation batch is not yet oracle-resolved because healthy
profiles exhausted their rolling quota. Its balanced and strong variants also
fall below the release fidelity gate locally, so even a future negative oracle
result would be a mechanism-discovery result, not a releasable remover.

A follow-up bounded-warp batch replaced RMS-normalized displacement with strict
per-axis maxima of 1.0, 1.8, and 2.8 pixels. The 1.0-pixel variant retained
37.79 dB PSNR and 0.9784 SSIM but left the local phase score essentially
unchanged at 0.643. The 2.8-pixel variant reduced the score only to 0.503 while
falling to 28.21 dB and 0.8570 SSIM. A multi-transform bounded variant reached
0.159 locally only at 21.97 dB and 0.6749 SSIM. Spatial fragmentation therefore
does not approach the release fidelity gate in this implementation and remains
a preregistered mechanism probe for the next healthy oracle window, not a
product candidate.

### 2026-08-09: positive-only RGB plus S/V detector epoch

The color-space result was converted into a deliberately one-sided research
detector. It requires exact 2816x1536 geometry, RGB evidence of at least 0.302,
HSV saturation-plus-value evidence of at least 0.302, and active carrier
support of at least 0.50 in both branches. It emits `positive` only when every
condition passes and emits `abstain` otherwise; it never claims that SynthID is
absent.

Leave-one-positive-out fitting passed all four older Google positives. Their
minimum held-out RGB evidence was 0.527, minimum held-out S+V evidence was
0.452, and minimum active support was 0.600. The final frozen models detected
all five available oracle-positive controls. A new Picsum epoch downloaded
IDs 600-999; after exact-geometry validation and content-hash deduplication
against all earlier epochs, 330 images remained. None produced a positive
verdict. However, 328 of 330 abstained for insufficient carrier support and
only two reached both support gates before falling below the evidence
thresholds. This validates a positive-only signal catcher inside one geometry,
not a two-sided detector or a general FPR claim.

### 2026-08-09: ensemble phase projection is not a sufficient removal loss

An analytical pixel-only attack removed the positive complex projection onto
the learned RGB phases and HSV S/V phases. One iteration at strength 0.5
cleared the local ensemble at 56.03 dB PSNR, 0.99909 SSIM, and a maximum pixel
change of two levels. A norm-matched random-noise control remained locally
positive.

The provider result rejected this loss as sufficient. In a Gemini session
where the untouched fern control returned `SynthID Status: Detected`, all
three preregistered projection candidates at strengths 0.5, 0.75, and 1.0 also
returned detected. The strongest candidate had negative local RGB and S+V
evidence at 51.97 dB PSNR. The ensemble therefore tracks a real correlate but
does not cover all evidence used by Google's decoder.

### 2026-08-09: 16x32 periodic-tile carrier hypothesis

The failed global projection exposed a more specific spatial structure. Of
the top 256 external V3 bins at 1536x2816, 208 (81.25%) lie exactly on an FFT
lattice spaced by 96 rows and 88 columns. This corresponds to a spatial cell
of `1536 / 96 = 16` rows by `2816 / 88 = 32` columns. Instead of subtracting a
small list of harmonics, a new pixel-only probe high-pass filters the image,
folds the residual modulo 16x32 over 8,448 repeats, and subtracts the complete
zero-mean folded template.

On the fern positive with its visible sparkle already removed, the mildest
locally clearing candidate used Gaussian sigma 0.6 and strength 1.5. The
phase-aligned template cleared the ensemble at 57.03 dB PSNR and 0.99911 SSIM;
the same template shifted by one row and one column remained positive at
57.02 dB and 0.99912 SSIM. Two additional aligned/shifted pairs reproduced the
direction at 55.96 and 54.19 dB.

The per-image estimator was then run on the four older positives with a frozen
strength ladder. The minimum locally clearing strength was 2.0 for three
images and 1.5 for one. All four phase-aligned outputs abstained, while all
four same-strength one-pixel-shifted controls remained positive. Selected
outputs retained at least 55.67 dB PSNR and 0.99874 SSIM. This is the strongest
local causal evidence so far because effect direction follows exact tile
phase at matched distortion across five contents.

It is not a removal result. In a healthy Google verifier session, the fern
source control was detected and the mild aligned sigma-0.6, strength-1.5
candidate was also detected. The stronger local variants therefore remain
unresolved rather than inheriting a negative label from the local ensemble.
Cross-resolution tile discovery and same-provider hard negatives also remain
open.

### 2026-08-09: provider-specific phase losses failed both remote verifiers

The Google result above and a separate OpenAI experiment closed the first
analytical-loss epoch. The OpenAI branch fitted an 8-pixel phase pattern from
OpenAI low-texture outputs and produced a 46.70 dB candidate whose local phase
score changed sign. A single frozen submission to OpenAI Verify still returned
`SynthID detected`; C2PA was absent because the candidate had been re-encoded.
This confirms that metadata removal and a local phase-score reversal do not
establish pixel-watermark removal.

The experiments are not cross-provider replications. Google images were sent
only to Google Verify AI and the OpenAI candidate only to OpenAI Verify. No
carrier, threshold, response, or verdict was transferred between tracks.
OpenAI's published verifier guidance prohibits repeated queries for watermark
reverse engineering or evasion, so that one failed query did not start an
adaptive loop. Further OpenAI candidates remain local hypotheses unless an
authorized research oracle becomes available.

### 2026-08-09: JPEG-robust periodic EOT surrogate pilot

Small residual CNNs were trained separately for each provider as differentiable
attack surfaces. They are provider classifiers, not SynthID detectors: their
corpora lack current same-provider oracle-negative controls, and several
external negatives approach the positive range. Two same-architecture models
were used for optimization and a third model with a different first kernel and
channel schedule was kept outside the attack.

For Google, all three models scored the temporal fern positive at a 75th-patch
percentile between 0.938 and 0.962. A patch-local L-infinity-8 attack initially
cleared the models and survived JPEG, but visual inspection exposed rectangular
grid boundaries; that candidate was rejected despite favorable scalar metrics.
A replacement optimized one 16x32 residual cell repeated over the full image,
with differentiable JPEG-95 and JPEG-90 branches. At half strength it retained
36.80 dB PSNR and 0.9241 SSIM against the icon-clean source. The three PNG
scores fell to 0.024, 0.047, and 0.016; after an actual JPEG-90 round trip they
were 0.182, 0.243, and 0.016. The independent RGB and HSV S+V phase ensemble
also abstained, and remained below threshold after JPEG-90, JPEG-95, WebP-95,
and a 0.98 resize round trip.

For OpenAI, the held-out complex market image scored 0.936, 0.929, and 0.813
on its three provider classifiers. An independently optimized 8x8 periodic EOT
residual was then attenuated in low-texture regions. The selected local
candidate retained 34.97 dB PSNR and 0.9356 SSIM. Its PNG scores were 0.030,
0.053, and 0.099; after actual JPEG-90 they were 0.042, 0.063, and 0.178.
WebP-95 behaved similarly, while a 0.98 resize raised the independent model to
0.692 and remains a measured weakness. A second optimization that included a
resize branch improved the two optimized models but transferred worse to the
independent model, so it was rejected.

The periodic residuals were then tested for cross-image transfer without
further optimization. The OpenAI 8x8 cell learned from the market image was
applied with the same low-texture attenuation to 11 other OpenAI positives.
Every image moved downward on all three models, both as PNG and after an actual
JPEG-90 round trip. Median PNG scores changed from 0.971, 0.970, and 0.931 to
0.016, 0.033, and 0.036. Each model had 11 of 11 paired decreases, giving a
two-sided exact sign-test `p=0.00098`; per-image PSNR ranged from 33.50 to
35.99 dB. Two complex scenes retained materially higher scores on the
independent model, so the transfer is strong but not uniform.

The Google 16x32 EOT cell was likewise applied to the four older images after
their own locally clearing phase-tile subtraction. All four remained phase
ensemble abstentions, and every CNN score decreased in the lossless outputs.
JPEG-90 preserved the direction but restored the two optimized models to
roughly 0.58-0.78 on the hardest images. Fidelity was approximately 37.25 dB
PSNR. With only four paired images this is descriptive transfer evidence, not a
statistical or oracle gate.

These are positive local robustness results, not watermark-removal results.
Neither candidate meets the release fidelity gate, neither CNN ensemble has
cleared the detector-identifiability gate, and neither selected EOT candidate
has a matching-provider negative oracle verdict. A three-file Google batch is
hash-frozen outside the repository for the next healthy quota window: the
positive source, the stronger phase-only ablation, and the combined periodic
EOT candidate. Results will be accepted only in that fixed order without
between-query adaptation. Two later attempts detected the positive source, then
returned a usage-limit response for the phase-only ablation. In the latest
attempt the verifier requested a retry after 17 hours. Both responses are
`indeterminate`, and the combined candidate was not submitted out of order. The
OpenAI candidate is not scheduled for the public verifier under the current
usage restriction.

### 2026-08-09: OpenAI fidelity and architecture-transfer frontier

Further local experiments tested whether the OpenAI periodic result could move
above the release fidelity gate without losing transport robustness. Chroma-only
projection reached 47.43 dB PSNR, but an actual JPEG round trip removed the
perturbation and restored the two classifier scores to approximately 0.9. Mixed
luminance/chroma projections and stronger texture masks did not improve that
frontier. A regularized Adam attack reached 41.21 dB on the source image, but
JPEG-90 restored the optimized classifiers to 0.90 and 0.91, and the independent
architecture scored 0.89. This rejected post-hoc color projection and simple
regularization as fidelity remedies.

A second experiment put actual OpenCV JPEG and WebP results in the forward pass
and used a straight-through gradient. At 40.08 dB, actual JPEG-90 still scored
0.89 and 0.91 on the optimized models. Removing the luminance penalty improved
the first model to 0.76 but left the second at 0.88 and reduced fidelity to
38.78 dB. The exact-codec forward pass therefore corrected an invalid transport
approximation, but did not produce a new Pareto candidate.

An architecture-diverse fourth classifier used fine RGB residuals, broader RGB
residuals, and luminance/opponent-color residuals. The market image remained
outside its training split. Its 75th-patch percentile ranged from 0.983 to 0.988
on four held-out OpenAI positives; the median across 120 held-out COCO negatives
was 0.041 and the maximum was 0.521. This is useful attack-surface diversity, not
a certified detector, because no current same-provider oracle-negative corpus is
available.

Adding that model to periodic optimization did not improve architecture transfer.
The preregistered local winner retained 34.86 dB and scored 0.057, 0.197, and
0.021 after JPEG-90 on the three optimized models, but the independent model
scored 0.573. A per-image half-resolution perturbation produced a higher-fidelity
41.86 dB candidate; JPEG-90 scores on its optimized models were 0.587, 0.572,
and 0.471, while the independent model remained at 0.875. Optimizing only the
two original models produced the same failure on both independent architectures.

The measured local frontier is therefore unchanged. The 34.97 dB periodic
candidate remains the strongest transport-robust and cross-architecture result;
the candidates above 40 dB are architecture-specific and cannot support a
removal claim. No additional public OpenAI verifier requests were made during
these experiments.

### 2026-08-09: mechanism reset and paired open-method control

The first detector epochs optimized the wrong statistical problem. The
[technical reference](synthid.md#11-post-hoc-model-independent-design) now
records the paper's paired training target, symmetric transformation channel,
independent detection logit, and two-sided calibration. Applied to the initial
pilots, those details reject three implicit assumptions: that a carrier phase
can stand in for the detection score, that mean patch classification is an
adequate image-level decoder, and that a sigmoid threshold calibrated on
unrelated negatives is a deployable decision rule.

A bounded open-method control tested the architecture concern without claiming
that another watermark reproduces SynthID. TrustMark P was used only because
its public encoder creates causal clean/watermarked pairs and its public decoder
can validate transformed labels. Ninety public COCO images were split by source
image into 60 training, 15 calibration, and 15 test pairs. Each pair received
the same identity, JPEG-90, 0.8 resize round trip, or 5% crop transformation.
The official decoder detected 14/15, 15/15, 14/15, and 15/15 transformed marked
test images respectively, with zero detections on the corresponding clean
images.

The existing normalized residual patch recipe reached only 0.707 identity AUC
and 0.668 aggregate AUC on the held-out pairs. A small full-field model that
retained RGB amplitude plus fine and coarse residuals reached 0.947 identity AUC
and 0.841 aggregate AUC. At a threshold above every calibration negative, their
aggregate TPRs were 5% and 20% respectively. Both models failed to reproduce
the official decoder's JPEG robustness. This small proxy is not a TrustMark
benchmark and says nothing directly about SynthID accuracy. It does falsify the
claim that the current patch-normalized architecture and training recipe are an
adequate generic neural-watermark detector.

The resulting eight falsifiable hypotheses now define the canonical
[D3-D5 mechanism program](#mechanism-hypotheses-for-d3-d5). In particular,
external generator corpora remain a final false-positive challenge and do not
substitute for counterfactual negatives.

Until the counterfactual-label and full-field gates pass, the current residual
CNN remains a useful vendor-triage stage for a future cascade, but it is not a
SynthID detector and should not be optimized as the final decision surface.

### 2026-08-09: paired wavelet and spectral ablation

The paired spectral harness was extended with three-level undecimated `db2`
wavelets, complex Fourier phase coherence and power, cepstral peaks, and a
cyclic clean/marked permutation control. The implementation streams wavelet and
spectrum field accumulators rather than stacking transformed fields across
pairs.

On 60 public TrustMark P training pairs at canonical size 256, the true
residuals had mean inter-pair RGB NCC of 0.097-0.121. Cyclically mismatching each
clean image with the next marked image reduced all three channels to about
-0.017. The strongest coherent Fourier cluster occurred around vertical offsets
9-11 and was strongest in the blue channel. Wavelet repeatability was greatest
in horizontal detail bands at levels 1 and 2. These are causal residual
measurements for the open proxy, not evidence about SynthID's carrier.

The representations then faced a source-disjoint single-image test with 60
training, 15 calibration, and 15 test pairs under identity, JPEG-90, 0.8 resize,
and 5% crop transformations. A fixed spectral-template score reached only 0.618
identity AUC and collapsed to 0.498 under crop. A 318-feature wavelet/spectral
summary reached 0.653 aggregate AUC and zero TPR at the threshold above every
calibration negative.

Full-field neural ablations at size 128 gave the following aggregate results:

| Input | AUC | Paired wins | Test FPR | Test TPR |
| --- | ---: | ---: | ---: | ---: |
| Spatial RGB plus residuals | 0.723 | 57/60 | 0% | 11.7% |
| SWT maps only | 0.671 | 59/60 | 0% | 0% |
| Complex FFT maps only | 0.516 | 41/60 | 0% | 0% |
| Spatial plus SWT, late fusion | 0.665 | 59/60 | 5.0% | 8.3% |
| Spatial plus SWT plus FFT, early fusion | 0.639 | 57/60 | 1.7% | 1.7% |
| Spatial plus SWT plus FFT, late fusion | 0.698 | 54/60 | 0% | 0% |

The table uses one threshold above every calibration negative; test FPR is
reported separately because a clean calibration result does not guarantee a
clean locked test.

The SWT branch contains weak, unusually consistent paired evidence, but it did
not improve the spatial model's source-disjoint discrimination in this small
proxy. The FFT map branch was indistinguishable from chance. Therefore neither
representation advances into the primary detector by default. SWT remains a
late-fusion ablation for a larger paired corpus; complex spectral analysis
remains a residual-discovery diagnostic unless a future held-out test reverses
this result.

### 2026-08-09: closed-corpus architecture and score ablations

A temporally split closed three-class corpus then tested whether the open-proxy
architecture result transferred to real external-source images. These labels
establish source classes, not watermark presence; they can measure
generalization and false positives but cannot substitute for counterfactual
clean/marked labels.

The frozen normalized-patch baseline was repeated over three training seeds.
Individual original-image AUCs ranged from 0.864 to 0.879. Averaging the three
models increased AUC to 0.884, but a threshold above every validation negative
still produced one false positive among 259 test negatives. At that operating
point the two positive-source TPRs were only 2.45% and 8.97%. Calibrating near
1% validation FPR yielded 1.54% test FPR and TPRs of 6.68% and 18.16%.
Accepted positives had 98.8-100% vendor accuracy, so vendor attribution is
useful only after an independently reliable presence decision.

A naive full-field spatial model reached only 0.599 AUC. Adding a separately
encoded stationary-wavelet branch and late fusion reduced AUC to 0.597. Both
had effectively zero strict TPR. The result does not contradict the causal
open proxy: it shows that global average pooling over source-labeled images
does not isolate the transferable signal and that SWT does not repair the
source shift.

Post-hoc image aggregation also failed to create an operating point. Averaging
presence logits raised AUC from 0.884 to 0.890, but increased strict test FPR to
0.77% while reducing TPR. Median, lower-quartile, upper-quartile, and top-tail
aggregation either preserved the overlap or suppressed almost every positive.
A separately trained binary presence head plus vendor head reached 0.880
ensemble AUC, 0.77% strict test FPR, and TPRs of 1.11% and 7.26%. Decoupling the
heads is architecturally faithful but insufficient without faithful labels.

These ablations close spectral feature engineering, naive full-field pooling,
and score aggregation as the next detector step. The next valid training epoch
requires same-content counterfactual pairs or independent watermark oracle
labels, matched transformations, and an image-level detection loss. External
generator corpora, including difficult non-target providers, remain hard
negative and FPR-challenge sets only.

### 2026-08-10: low-content controls and registered phase carrier

A same-resolution low-content matrix compared independently generated solid
outputs from two target model families against three per-image controls: exact
mean fill, amplitude-matched Gaussian noise, and a phase-randomized residual
with preserved Fourier magnitude. Raw stationary-wavelet summaries transferred
between the two target families with AUCs of 0.982 and 1.000, and reached 0.973
when blue and green were held out by color. This was not watermark evidence.
The frozen classifier accepted every one of 1,869 external negatives because
it had learned the distinction between real generator texture and artificial
controls. Removing absolute wavelet energy reduced external-negative
acceptance only to 61.6%, with similar 58.1-67.3% acceptance across all three
source classes. Both low-content wavelet branches are rejected as presence
detectors until real non-target solid outputs provide matched negatives.

The numeric V3 audit loader was then extended to support both dense and sparse
format-v2 profiles without pickle. Exact-profile evaluation exposed a sharp
encoder-version boundary. The 1024x1024 profile accepted none of 231 target
provider images and none of 26 exact-geometry negatives. The 1536x2816 profile
accepted 30 of 55 target-provider images, including all four temporal-test
images, while rejecting the one exact-geometry negative available in the
closed corpus. The independently fitted phase model accepted 24 of those 55
and also accepted all four temporal-test images. This is positive evidence for
a geometry- and epoch-specific carrier, not a universal SynthID decoder.

On the four temporal-test positives, the fixed V3 score survived JPEG-95 and a
75% downscale on all four images, survived JPEG-85 on two, and failed after a
5% center crop or a one-pixel translation on all four. Bounded analytical
translation registration recovered all four shifted images and selected the
known `(-1, -1)` offset. Searching up to 16 pixels produced no positives among
50 exact-resolution and 144 canonicalized frozen negatives. The shared
registration implementation now serves both the numeric V3 probe and the
independently fitted phase model.

A discovery-only scale-and-translation view search recovered all four 5%
cropped temporal images with the independently fitted model after lowering the
active-support gate from 0.50 to 0.40. It produced zero positives on the 194
frozen negatives and on the same preregistered 3,000-image COCO challenge used
by the identity scorer. The latter result has a zero-error one-sided 95% bound
of 0.0998% only for that abstention challenge: every COCO image remained
outside carrier support, with a maximum active fraction of 0.201. The scale
rule is not frozen because its support threshold was selected after inspecting
the crop examples. It requires a new temporal positive holdout before it can
join the detector rule.

The current actionable research candidate remains a positive-only,
provider-specific expert for the supported 1536x2816 carrier epoch. Identity
and bounded translation views use the frozen phase and support thresholds;
unsupported geometry, insufficient carrier magnitude, and ambiguous phase
return `abstain`. Vendor attribution may select the expert that supplied
accepted evidence, but it must not turn an abstention into a provider label.
The next calibration gate still requires at least 3,000 native-support
negatives, same-provider oracle negatives, matched non-target solid outputs,
and a new temporal positive that influenced neither profile nor threshold.

### 2026-08-10: 2048 periodic-tile detector

The phase hypothesis transferred to a second native geometry when the model
and operating point were kept separate. A 256-peak 2048x2048 model learned from
111 train positives used the already frozen phase threshold of 0.43, active
support threshold of 0.40, and a translation search of plus or minus four
pixels. It accepted 31 of 35 validation positives and 27 of 30 locked-test
positives while accepting none of 49 validation or 38 test negatives.
Translation registration recovered the shifted positives without changing
those identity counts. A threshold recalibrated from only 49 validation
negatives reached 30 of 30 test positives but accepted one test negative, so it
was rejected in favor of the transferred rule.

The wider native-geometry challenge exposed the remaining uncertainty. The
frozen 2048 rule accepted two of 182 earlier external-provider images, for two
accepted source negatives among all 269 native negatives. Both cases passed at
zero translation with high phase and support, and both also passed an
independently learned HSV phase branch. They are operational false positives
under source labels, but source provenance does not establish watermark
absence. They may instead expose a shared encoder or upstream backend. Without
an independent watermark oracle they cannot be relabeled either way. The same
experiment rejected the 1024x1024 and 768x1376 experts: they accepted 9 of 26
and 4 of 9 native source negatives, respectively.

The 2048 carrier has a concrete periodic mechanism. Its 256 peaks reduced to
108 unique spatial frequencies. Translating the frequency coordinates by 128
rows preserved 56 coordinates, while the maximum overlap in each of 1,000
uniform random controls was two. The permutation estimate was 0.001, and the
128-bin spacing implies a 16x16 spatial tile. A separate detector therefore
folded a high-pass residual modulo 16x16, averaged 16,384 repetitions, and
correlated the normalized tile against a train-positive template. After
float64 serialization and validation-only threshold calibration, the fixed
tile accepted 34 of 35 validation and 29 of 30 test positives, none of the 49
calibration or 38 held-out test negatives, and the same two of 182 earlier
source negatives. This second representation supports a real periodic carrier
rather than an arbitrary set of FFT peaks, but it does not resolve the two
labels.

The fixed tile also accepted none of a preregistered 3,000-image general-image
challenge after deterministic 2048x2048 canonicalization. The maximum
normalized correlation was 0.106 against a threshold of 0.174. The zero-error
one-sided 95% upper bound is 0.0998% for that challenge. Unlike the earlier
sparse-phase result, the tile score evaluates every image without a weak
carrier-support abstention. The challenge is still not native provider data
and cannot replace the required oracle-negative calibration.

A later exact-1024 content audit found that perceptual siblings can cross an
image-level fit split even when file and decoded-pixel hashes differ. The local
2048 reports retain aggregate counts and model hashes but not the 111 fitting
paths, so the old train/validation/test split cannot now be audited for that
specific leakage mode. This does not invalidate the independently collected
general-image challenges or justify changing the frozen runtime model, but the
reported positive rates remain conditional on the historical split. Any next
model calibration must retain content-group membership and prove that no group
crosses train, validation, test, or temporal partitions.

Symmetric attack evaluation established the robustness boundary. The fixed
tile accepted 29 of 30 original test positives, all 30 after a 75% downscale
round trip, 21 after JPEG-95, three after JPEG-85, and none after a 5% crop,
with no accepted held-out negatives under the identity threshold. A
validation-calibrated JPEG-95 tile threshold recovered 27 of 30 but accepted
one of 38 test negatives. Requiring both codec-conditioned tile and phase
scores reduced JPEG-95 to 16 of 30 with no held-out-negative acceptance, but
still accepted one of the 182 earlier source negatives. The corresponding
JPEG-85 consensus accepted 5 of 30 positives and none of all 269 native source
negatives. Scale-and-translation phase search recovered 15 of 35 validation
and 14 of 30 test crops with no held-out-negative acceptance, but remains
discovery-only because the test transformation had already influenced the
branch. Low-frequency peak subsets and transform-augmented phase training
improved JPEG sensitivity only by raising validation false positives to
2-10%, so both were rejected.

The reproducible implementation is `scripts/synthid_periodic_tile_probe.py`.
It stores the normalized template in float64 and calibrates only after loading
the serialized artifact; an earlier float32 experiment moved a boundary score
by approximately 2.5e-10 and demonstrated why calibration-before-serialization
is invalid. The resulting research detector is positive-only and limited to a
confirmed 2048x2048 carrier epoch. An accepted expert may suggest the encoder
family, but the two cross-source carrier matches prohibit a stronger vendor
claim until an oracle distinguishes direct provider output from shared-backend
output.

A pixel-space ablation then tested whether the frozen tile merely predicted the
local scores or controlled them. At twice the train-median tile norm, aligned
subtraction changed the fixed-tile decision from 29 of 30 accepted originals to
zero and the independently fitted sparse-phase decision from 27 of 30 to zero.
The median fidelity was 53.74 dB PSNR and 0.99681 SSIM. Cyclically shifting the
same tile by one row and column left one phase acceptance, while a seeded
zero-mean random tile orthogonal to the learned template left 13; their median
PSNRs were within 0.13 dB of the aligned edit. For both local representations,
the aligned edit reduced the score more than either control on all 30 paired
images, with a two-sided sign-test p-value of 1.86e-9 for each comparison.
The same aligned edit reversed both local decisions on each of the two disputed
cross-source matches, while the orthogonal control left both phase decisions
accepted. That two-item result is descriptive but makes an accidental threshold
crossing less likely; it still cannot distinguish a shared carrier from direct
provider output.

This is the strongest local causal evidence for the 16x16 mechanism, but the
strength was selected after inspecting this locked test and is therefore
discovery-only. The shifted control also suppressed the phase representation
substantially, so local score reversal cannot certify signal removal. A matching
provider oracle must still compare untouched sources, aligned candidates, and
norm-matched controls before any removal claim. The reproducible local harness
is `scripts/synthid_periodic_tile_ablation.py`; its report records input and
model hashes without writing derivative images.

### 2026-08-12: fast adaptive carrier projection

The fixed twice-median subtraction was intentionally conservative but spent
substantially more pixel norm than many images needed. A follow-up tested a
one-dimensional per-image solver. For source pixels `x`, frozen unit carrier
`t`, and scalar amplitude `a`, the candidate is
`clip(round(x - a * tile(t)))`. A bounded scalar search selects the smallest
`a` that reaches the local target. This is a direct pixel edit: it uses no
generative decoder, does not resize the image, and searches one scalar rather
than optimizing a full-resolution tensor.

The first challenge reused the 30 historical 2048x2048 causal-ablation
sources. Three were not simultaneously positive under the fixed-tile and
independently fitted phase rules and were excluded before counting a reversal.
All 27 conjunctive positives cleared both local rules after adaptive
subtraction. Median PSNR was 59.83 dB and median SSIM was 0.99890; the minima
were 55.19 dB and 0.99705. Repeating the same selected amplitude with a
one-row, one-column cyclic shift cleared zero of 27. Sixteen original images
remained conjunctively positive after JPEG-95. The aligned candidates cleared
all 16 after the same round trip, while the shifted controls cleared one. This
improves materially on the earlier fixed-strength median of 53.74 dB without
weakening the local causal control.

A three-image pilot also compared subtracting the complete per-image folded
high-pass residual and a regularized inverse-high-pass reconstruction. Both
could clear the two local scores, but their selected candidates retained only
about 54-57 dB PSNR, compared with 59-62 dB for the frozen carrier projection
on the same images. Removing the whole folded residual therefore spends pixel
norm on content-correlated periodic structure and was rejected in favor of the
narrow carrier direction.

The geometry challenge then selected three `first_only` carrier positives from
each of the 20 most frequent native Spaces geometries. The edit repeats the
same 16x16 carrier past the image boundaries and crops the repetition to the
decoded dimensions, so it does not require dimensions divisible by 16. A zero
score target cleared 60 of 60 aligned candidates and 5 of 60 matched shifted
controls. Median PSNR and SSIM were 60.37 dB and 0.99919; the minima were
53.56 dB and 0.99581. These samples come from an already inspected detector
corpus, so they measure mechanism transfer across geometry rather than fresh
detector or removal generalization.

The zero target left one WebP-95 and eight resize survivors among transformed
sources that were still locally positive. A frozen follow-up used a target
score of -0.25. It cleared all 60 native candidates at median 57.70 dB PSNR and
0.99863 SSIM, with minima of 52.38 dB and 0.99443. Conditional on the matching
transformed source remaining above the fixed carrier threshold, the aligned
candidate cleared 32 of 32 JPEG-95, 13 of 13 JPEG-90, 8 of 8 JPEG-85, 15 of 15
WebP-95, and 54 of 54 0.75x resize round trips. The corresponding one-pixel
shifted controls cleared 7, 1, 1, 5, and 17. The increased shifted-control rate
is the measured price of adding transform margin.

The frozen -0.25 recipe was finally run once on one deterministic first-carrier
positive from every represented native Spaces geometry. All 647 images at 647
distinct decoded sizes reached the target, with no maximum-amplitude failure.
Median fidelity was 57.67 dB PSNR and 0.99867 SSIM; the worst case was 51.78 dB
and 0.99400. Of 323 source images whose carrier survived JPEG-95, 323 aligned
candidates and 58 shifted controls cleared after the same round trip. Natively,
153 of 647 shifted controls also cleared at this deeper margin. The consistent
aligned result supports geometry-independent control, while the increased
shifted rate shows that the conservative edit perturbs a wider neighborhood of
the periodic carrier. Because the all-geometry corpus had already influenced
detector research, it is a coverage challenge rather than a fresh holdout.

This is a fast local carrier suppressor, not a verified SynthID remover. The
earlier Google oracle remained positive after analytical edits that already
looked favorable to local models, and the adaptive scalar recipe has not been
submitted to a matching-provider oracle. Provider terms and the project oracle
policy also prohibit turning verifier responses into an adaptive search loop.
The next valid test is one preregistered source/re-encode/aligned/shifted batch
at a healthy authorized oracle, with the amplitude rule and -0.25 target frozen
before submission. Until that batch returns a matching-provider negative result,
no production command exposes this method as removal.

The reproducible implementation is
`scripts/synthid_adaptive_carrier_suppress.py`. It writes only lossless PNG,
refuses overwrite and locally negative input, and records source/output hashes,
the selected scalar amplitude, local scores, fidelity metrics, and runtime in a
sidecar JSON report. Unit tests cover non-divisible geometry, target attainment,
an unreachable target, and the negative-input guard. A real CLI smoke test on a
Spaces source reproduced its reported score exactly through the runtime detector
after saving. It remains a research script rather than a routed package command.

### 2026-08-10: confirmatory oracle-batch readiness

The corpus audit found no new 2048x2048 positive collected after the tile rule
and subtraction strength were frozen. Existing test images cannot be renamed as
a temporal holdout, so the confirmatory oracle batch remains blocked on new
provider output at the supported native geometry.

The batch mechanics are ready in `scripts/synthid_oracle_batch.py`. For each new
source, it preregisters five requests in a fixed order: untouched source,
lossless re-encode, aligned subtraction, cyclic-shifted subtraction, and an
orthogonal norm-matched tile. It refuses to write media inside the repository,
records artifact and decoded-pixel hashes, pins both model artifacts and the
matching provider, and detects any post-registration byte change. Oracle results
live in a separate manifest-bound file so the preregistration remains immutable;
the evaluator requires complete ordered results, verbatim responses,
timezone-aware timestamps, and separate SynthID and C2PA verdicts.

A causal success requires the source, re-encode, shifted, and orthogonal views
all to remain `detected` while only the aligned view becomes `not_detected`.
`Indeterminate` and refusal remain indeterminate. A one-source dry run on an old
test positive verified all five artifacts and the local scoring path, but is
pipeline validation only and contributes no confirmatory evidence.

### 2026-08-10: first confirmatory oracle attempt

Four new 2048x2048 Gemini outputs were generated after the local rule and
strength were frozen, then registered as a 20-request batch. The manifest and
all 20 artifacts passed hash verification before submission. The first source
group produced these ordered results:

| Role | Local tile/phase decision | Provider result |
| --- | --- | --- |
| source | detected / detected | SynthID indeterminate; Google C2PA present |
| re-encode control | detected / detected | Google AI signal detected |
| aligned subtraction | detected / not detected | Google AI signal detected |
| shifted control | detected / detected | Google AI signal detected |
| orthogonal control | detected / detected | refused: verification quota exceeded |

The quota response instructed the account to wait 24 hours. No adaptive retry
was made, and the remaining 15 preregistered requests were not submitted. The
batch cannot receive its preregistered aggregate verdict until complete ordered
results exist.

Even incomplete, this attempt rejects the frozen removal recipe for the first
temporal source: aligned subtraction crossed the phase threshold but remained
above the tile threshold and did not clear the provider oracle. The remaining
sources can still measure transfer and disagreement between the two local
experts, but they cannot turn this first aligned result into a universal
pixel-only removal success.

### 2026-08-10: positive-only runtime detector

Removal is deferred while the transferred tile signal is exposed as a bounded
detector. `src/remove_ai_watermarks/synthid_detector.py` loads the frozen model
as a bundled pickle-free runtime asset and returns `detected`, `not_detected`,
or `unsupported` without resizing input. The direct API and
`detect-synthid` CLI initially covered only native 2048x2048 images; `identify`
consumes a positive match as high-confidence evidence but never treats a
negative or unsupported result as proof of absence.

The operating point and model are unchanged from the locked experiment. The
detector accepted all four post-freeze Gemini outputs, including the source
whose frozen phase score missed, while retaining the prior 29-of-30 locked-test
sensitivity and zero accepted validation/test negatives. The unresolved two
external-source matches remain the reason a runtime positive names the carrier
but does not attribute a provider.

### 2026-08-10: calibrated native-geometry extension

The frozen 16x16 template and threshold transfer across the common native
geometries in the provider-positive evidence. Non-divisible image dimensions
use direct modulo folding with per-cell sample counts; divisible dimensions,
including 2048x2048, retain the original numerical path. All prior 2048x2048
records were replayed and matched exactly, including floating-point scores.

The runtime registry was expanded only after the unchanged threshold accepted
none of 60,000 public COCO challenge views, 3,000 at each of 20 target
geometries. This is a geometry extension of the same positive-only carrier
expert, not an OpenAI pixel detector or a proprietary payload decoder.

### 2026-08-11: calibrated image-size range

The fixed template was then evaluated on every provider-positive image in the
local evidence set rather than only its common geometries. It accepted 3,928 of
4,698 images across 757 exact geometries. Sensitivity separated by pixel count:
1,987 of 2,021 images at or above three megapixels crossed the threshold, while
1,940 of 2,672 images from one through two megapixels did. This establishes a
carrier-family boundary, not universal SynthID recall: explicit C2PA watermark
actions also occur below threshold, and three strong carriers use a different
cyclic phase.

Two public-image geometry challenges tested whether geometry itself creates
false matches. The first balanced 5,000 COCO images across all 757 observed
geometries, with every geometry present in both development and final partitions;
the maximum fixed score was 0.12549 and none crossed the unchanged 0.17357
threshold. The second transformed the same 5,000 source images at 256 generated
geometries from one through 18 megapixels, covering every width/height remainder
pair modulo 16. Its maximum was 0.16387 and again none crossed the threshold.
Runtime support therefore uses that challenged pixel-count interval instead of
an enumerated geometry registry, still without resizing.

A cyclic-registration branch was rejected after the public-image challenge.
At a low threshold it recovered 130 fixed-phase misses but false-fired on
tracked OpenAI and Adobe controls. Raising the threshold above all development
controls recovered only 23 in-range provider positives while adding 84
cross-source matches beyond the fixed detector. Three strong shifted Google
carriers remain a documented miss rather than weakening the runtime rule.

Top-level source labels were not treated as watermark-negative truth. Of 237
non-Google-labeled corpus images crossing the fixed threshold, 81 retained
SynthID provenance. High-scoring Adobe manifests explicitly named Gemini Flash
as the remote provider, and perceptual matching found direct Google derivatives
under editing or relabeling categories. The remaining source labels cannot prove
absence of the carrier, so they constrain provider attribution but do not justify
raising the pixel threshold.

### 2026-08-11: spatial-resampling boundary

Arbitrary decoded geometry did not imply arbitrary resize robustness. A
stratified sample of 80 fixed-positive provider images was rescaled in memory at
0.5, 0.625, 0.75, 0.875, 1.125, 1.25, and 1.5. The unchanged 16x16 detector
accepted none of the 80 images at every nonidentity scale. Restoring each view
to its source dimensions recovered 58-80, which isolates the failure to the
scaled sampling lattice rather than simple information destruction.

Scaling the template and modulo fold to the corresponding integer periods 8,
10, 12, 14, 18, 20, and 24 recovered 73-80 of the discovery images at the
native threshold. Searching all eight periods raised control scores, so a
single threshold was frozen above the maximum of 3,000 deterministically
resized COCO development views. It accepted none of 2,000 final controls and
672 of 800 source-disjoint provider positives, with 77-88 accepted at each
scale. Six tracked OpenAI and Adobe controls remained below the frozen threshold.

This did not generalize to arbitrary scaling. Factors 0.8, 0.9, 1.1, 1.2, and
1.333 create noninteger carrier periods; rounding the period accepted zero of
80 images in the fixed-phase branch at every factor, and cyclic registration
accepted only 1-16. Period-specific thresholds improved sensitivity but accepted
five of the 2,000 final controls. The integer-period bank is therefore retained
as a research result, not added to the runtime. A universal resize branch needs
fractional-period registration or reliable reconstruction of the pre-resize
geometry, followed by a new multiple-search calibration.

A discovery follow-up tested that fractional-period direction with the 30
strongest template harmonics. A continuous 7.5-24.5 spectral search placed the
true period within 0.05 pixels among its top three candidates for 58 of 60
transformed positives. Searching nine neighboring integer reconstruction
geometries recovered 44 of 60 at the native threshold; even the true source
geometry recovered only 48, so resampling damage sets a separate upper bound.
The full search then failed a frozen 250-development/150-final COCO challenge:
the calibration threshold accepted two final controls. A preselected product
of canonical correlation and spectral confidence increased positive
acceptance from 49 to 52 of 60 at its development threshold but accepted the
same two final controls. This fractional FFT baseline is rejected for runtime
use; better period estimation alone does not solve multiple-search specificity.

### 2026-08-11: multiscale calibration and opt-in registered detector

The fractional search was revisited as a multiple-search calibration problem,
not as period estimation alone. A source-disjoint challenge used 200 confirmed
fixed-carrier positives, 40 independent sources at each scale 0.8, 0.9, 1.1,
1.2, and 1.333. Controls were 4,900 unique COCO images resized into the positive
pixel-count domain, split into 2,900 development and 2,000 final items. The ten
frozen candidates combined exact fractional canonical correlation, quadrant
agreement, and a three-level scale-space score through mean or minimum rules.
Pure box-counting dimensions, multifractal width, lacunarity, roughness,
wavelet-energy slope, and a generic log-Mellin response had already failed the
smaller pilot and were not promoted to the large challenge.

At the development-only threshold, the three-way mean accepted 177 of 200
positives versus 163 for canonical correlation alone, with 16 positives rescued
and two lost in the paired decision. Each produced one final false positive, on
different source images, so the development threshold was rejected. Recalibrating
above all 4,900 COCO controls left 176 of 200 three-way positives and 160 of 200
canonical-only positives. This recalibration is exploratory rather than a new
validation because the final controls had then been inspected.

A second source-disjoint corpus froze those thresholds before scoring 105 new
positives across scales 0.5, 0.65, 0.8, 1.0, 1.2, 1.333, and 1.5, plus all 879
content-deduplicated Kodak and Picsum controls available locally. Natural
2816x1536 controls had a heavier null tail than upscaled COCO: the three-way
mean accepted 82 of 105 positives and six controls at the COCO threshold,
whereas canonical-only accepted 79 positives and ten controls. A threshold
above all 5,779 controls retained 253 of the first 305 positives for the
three-way mean. Four coarse pixel-count bins recovered only six more positives,
so size normalization did not explain the main 0.5x miss.

At 0.5x, a fixed period-8 branch raised canonical correlation for seven of the
15 discovery positives but also won the unconstrained search on 629 of 879
controls. Searching it unconditionally merely raised the null threshold. The
frozen remedy requires period-8 native correlation of at least 0.35 and takes
the maximum of the old and period-8 three-way scores. It added four positives
without an accepted control in the discovery corpus. On the earlier 4,900
controls, the frozen gate never opened and none crossed the final three-way
threshold `0.2460603834083705`.

The improvement itself was then tested on 50 additional source-disjoint 0.5x
positives that influenced neither gate nor threshold. The ungated three-way
score accepted 6; the frozen gate accepted 17, rescuing 11 and losing none. The
exact paired sign test was `p = 0.0009765625`. These results support an opt-in
scale-registered detector over 250,000 through 10,000,000 decoded pixels. The
runtime now requires both sides to be at least 256 pixels because its frozen
confirmation needs complete disjoint 256-pixel patch groups. Narrower inputs
need a separately calibrated adaptive-patch expert. These results do not establish a
universal SynthID decoder: the control maximum helped set the operating point,
33 of 50 new 0.5x positives still missed, and crop, strong codec changes, other
carrier epochs, and provider attribution remain outside the claim. The fast
native fold remains the default and the `identify` path.

That period-8 conclusion was overturned by a later symmetric negative test.
Every one of the 11,506 known non-Google Spaces controls was downscaled by 0.5,
matching the positive transformation; 11,273 outputs remained in the registered
geometry range. The direct analytic 8-by-8 template was dominated by shared
resize structure: 80.8% of its energy projected onto a 2-by-2 lattice and 94.0%
onto a 4-by-4 lattice. The production period-8 gate opened on 73 controls, and
all 73 crossed the registered threshold. They spanned hosted jobs, Firefly,
China AIGC-labeled generators, Meta-tagged images, OpenAI, and local tools. No
tested secondary condition separated the 11 rescued positives from those 73
controls. The explicit period-8 rescue is therefore rejected and not part of
the runtime detector. The 0.5x rescue result is retained here as an incident
record against asymmetric transformed-positive evaluation.

The same symmetric corpus also invalidated the main single-threshold registered
rule: 68 of 11,273 supported controls crossed `0.2460603834083705`. Fixed
thresholds for nine recovered-period bins eliminated those errors while
retaining 238 of 355 transformed positives, but one of 4,900 COCO controls then
crossed through the third spectral candidate. Requiring the canonical winner to
match the strongest spectral candidate removed that case and all 68 Spaces
errors, with 239 of 355 positives retained. This version was not shipped: a
fresh, content-deduplicated 1,000-image Open Images validation corpus produced
six false positives.

All six Open Images errors concentrated their apparent match in one
middle-frequency ring. Their correlations disappeared or changed sign in the
two higher-frequency rings of the 16-by-16 carrier, unlike the accepted Gemini
positives. Adding a frozen minimum correlation of `0.075` across those two rings
removed the six development errors and retained 229 of 355 positives. A
previously untouched reserve of 499 supported Open Images controls then produced
five amplitude-gate candidates and zero final positives; their maximum
high-band score was `0.004682175209264222`.

The retained positives by scale were 0/65 at 0.5, 8/15 at 0.65, 53/55 at 0.8,
33/40 at 0.9, 13/15 at 1.0, 30/40 at 1.1, 38/55 at 1.2, 40/55 at 1.333, and
14/15 at 1.5. A physically transformed template recovered strong correlations
for some period-8 positives, but 975 top-candidate Spaces controls fully
overlapped every tested amplitude, spectral-band, and joint feature. No
zero-control operating point retained a period-8 positive. The calibrated
runtime therefore uses the top-candidate, period-bin, and high-band gates and
makes no 0.5x detection claim.

An exploratory amplitude-times-high-band rescue later tested whether the two
existing gates could trade evidence instead of both passing independently. A
threshold frozen above all 341 top-1 development controls recovered 11 of the
126 current false negatives and accepted none of a separate 499-image Open
Images reserve; the reserve maximum was 0.100583 against the 0.118168
threshold. The decisive symmetric challenge rejected it: 19 of all 11,506
Spaces controls after 0.5x resizing crossed the frozen threshold, with a
maximum of 0.179989. The errors spanned OpenAI, Firefly, Microsoft, hosted
jobs, and other generators. Raising the threshold post hoc above that maximum
would recover only seven false negatives, none at scale 0.65 or 0.8. The joint
rescue is rejected; the independent amplitude and high-band gates remain the
measured operating point.

### 2026-08-13: public bypass corpus and inverse-carrier audit

The public `cebeuq/Synthid-Bypass` V2 comparison set supplied 12 external
before/after pairs produced by a diffusion reconstruction workflow. The fixed
runtime carrier accepted 10 of 12 `before` images and none of the 12 `after`
images. The two misses were not assumed negative: Gemini Verify AI reported
SynthID on `before/03.webp` and `before/07.png`, and no SynthID on both paired
outputs. Outputs `01` through `06` were also oracle-negative; the verifier's
daily limit stopped the same-session sweep at `08`.

`07.png` is a low-SNR instance of the shipped positive direction. Its full-frame
score was `0.1248368819`; its signed RGB correlations were 0.073, 0.183, and
0.111, and all nine aligned region scores were positive. A proposed rescue at
`score >= 0.12` with Green dominance and spatial agreement separated it from
the two highest transformed-COCO controls, but failed a fresh exact-1024 Open
Images challenge. A 2009 natural photograph scored `0.2123730698`, had stronger
Green dominance and nine-region agreement, and Gemini returned `SynthID not
detected`. The downloaded bytes matched a fresh fetch from the official Open
Images validation bucket. This is a confirmed runtime false positive, not a
provider-label inference.

The failure is broader than that one JPEG. Symmetrically resizing the 24 Kodak
PNG controls to 1536 by 2752 with Lanczos produced fixed scores as high as
`0.3768339306`. Those synthetic views were not used as independent clean
samples, but they prove that rational resize lattices can create a stronger
signed 16-by-16 match than the runtime threshold. The existing 5,000-view
observed-geometry and 5,000-view generated-geometry COCO challenges remain
valid for their exact source-and-transform assignments; they do not justify a
universal transformation claim. No threshold relaxation for `07` is allowed,
and the current direct detector needs a fresh source-and-transform-grouped
specificity epoch before its support range can be broadened.

`03.webp` tested a different hypothesis. The two content-group-separated
alternate-Gemini experts both rejected it. Its shipped-template score was
instead `-0.3263447029`, all three channel scores were below `-0.28`, and every
one of nine aligned regions was negative. The paired oracle-negative output
weakened to `-0.1531420136`. Neither 5,000-view COCO geometry challenge nor the
fresh Open Images blocks contained a score at or below `-0.20`, making the
opposite-polarity response a plausible additional codeword or provider state.
It is not specific enough to ship: 40 of 11,506 heterogeneous non-Google-labeled
AI controls crossed `-0.20`, with examples from OpenAI, Microsoft, multiple
other generators, and unassigned made-with-AI files. An inverse-template expert
therefore remains a research hypothesis until matching-provider oracle labels
and a new untouched natural and AI control split exist.

A follow-up decomposed the two high-frequency rings instead of taking their
minimum. The 19 joint-rule false positives were broad-band, with median
`min(ring) / max(ring)` of 0.84, while the registered false negatives had a
median of 0.27. A development-selected rescue therefore multiplied normalized
amplitude by the stronger ring and required ring balance at or below 0.5. A
threshold above 341 development controls recovered 11 of 65 top-1 false
negatives and accepted none of the 499-image Open Images reserve. The full
symmetric challenge again rejected it: 13 of 11,506 resized Spaces controls
passed, with a maximum score of 0.694818 against the 0.308683 threshold. Strong
single-ring lattices occurred in Firefly, OpenAI, hosted jobs, China-labeled
generators, Canva, and other controls. Radial imbalance is therefore another
general renderer or resize feature, not a SynthID-specific rescue.

A multi-codeword hypothesis then projected the shared 16-by-16 template out of
each normalized residual tile and fitted source-disjoint spherical codebooks to
the remaining 768-dimensional vectors. Five prototypes selected on 300
positives retained 366 of 600 held-out positives. Three prototypes were highly
reproducible across an independent 300-positive fit, with matched correlations
of 0.88 through 0.95, while the additional states were less stable. A threshold
above 2,500 development natural-image controls accepted none of a separate
2,500-image reserve and selected 225 of 770 metadata-labeled Google images that
the shared-template threshold missed. The broader challenge decisively rejected
the feature: 828 of 11,506 metadata-labeled non-Google Spaces controls crossed
the frozen threshold. Errors occurred across OpenAI, Microsoft, Firefly,
China-labeled generators, hosted jobs, Stable Diffusion, and other sources. A
post-hoc intersection above every control that also passed the registered
min(R, B) gate retained 138 Google candidates, but none passed that independent
carrier gate. The residual clusters are reproducible generator or resampling
structure, not a specific additional SynthID codebook, and are excluded from
runtime.

A related spatial-payload pilot stopped averaging all 16-by-16 blocks and
measured carrier and codebook responses block by block. The positive carrier
response was spatially coherent: median adjacent-block correlation was 0.61 on
both axes, compared with 0.20 and 0.24 in a heterogeneous Spaces sample and
0.04 and 0.06 among codebook false positives. The development-selected rescue
was instead the standard deviation of the best codebook response across
blocks. It retained 11 of 60 reserve positives and none of 180 reserve controls,
but accepted six of 1,000 fresh heterogeneous controls. Those errors included
OpenAI, hosted, and China-labeled generators; raising the threshold above their
maximum retained only four of the 120 pilot positives. Per-block carrier or
codeword coherence is therefore diagnostic but not a usable rescue.

A geometry-stratified follow-up found a second reproducible native 16-by-16
carrier in the fixed-template misses. Templates fitted independently from two
halves of the 1024-by-1024 misses correlated at 0.869, and two independent fits
from the non-1024 misses correlated at 0.956. The latter correlated 0.82-0.84
with the 1024 template but only about 0.28 with the shipped carrier. This rules
out a single-resolution averaging accident and supports a distinct carrier
direction or state.

The initial 1024 template retained 8 of 91 held-out misses and accepted none of
81 exact-geometry reserve controls or 500 natural-image reserve controls. Seven
of those eight positives survived JPEG 95, six survived JPEG 85 and 70, and
three survived a 0.75x resize round trip; WebP 95 retained none. A threshold
frozen above the exact-geometry development controls accepted 25 of 11,506
heterogeneous controls when applied at arbitrary native geometries, so an
unrestricted second native template is rejected. Canonicalizing every control
to 1024 by 1024 eliminated all 11,506 errors, and the same path accepted only
the native 1024-by-1024 Google misses. The signal has a fixed native period,
not a size-normalized frequency.

A stricter chronological fit used only 1024-by-1024 misses through mid-June and
tested July misses. Its template correlated 0.839 with the independently fitted
July template, retained 5 of 95 temporal misses, and accepted none of 81 reserve
exact-geometry controls. At that frozen threshold, canonicalization accepted 18
of all 770 fixed-template Google misses and none of all 11,506 heterogeneous
controls; every accepted miss was natively 1024 by 1024. Requiring agreement
between two independently fitted early templates retained three temporal
misses and no reserve controls. This is credible evidence for a second carrier
direction within the Google cohort, but the positive and exact-geometry control
counts are still too small for a runtime operating point. The later external
natural-image challenge below further shows that this repeatability is not yet
specific enough for blind detection.

Projecting the shipped carrier out of every normalized tile did not remove the
second direction. The early orthogonal template was numerically orthogonal to the
shipped template, correlated 0.834 with an independently fitted July orthogonal
template, and retained the same 5 of 95 temporal misses with none of 81 reserve
exact-geometry controls. Its threshold was frozen above the development controls
before the decisive 1024-by-1024 canonical challenge; none of all 11,506
heterogeneous controls crossed it, and the maximum remained just below the
frozen boundary. The second direction is therefore not a weak projection of the
shipped carrier. This strengthens the research finding but does not enlarge the
small temporal-positive or exact-geometry control reserves required for runtime.

Mapping both frozen scores over all 4,698 Google-labeled images changed the
interpretation from an epoch replacement to coexisting carrier states. The map
contained 3,825 first-only, 69 second-only, 103 both, and 701 neither cases;
the two scores correlated at -0.563. Both directions occurred throughout the
collection interval and across multiple native geometries, while the presence
of both in 103 images rules out a simple dated switch from one template to the
other. The second native threshold remains exploratory outside exact
1024-by-1024 inputs because 8 of 11,506 heterogeneous native controls crossed
the orthogonal version's threshold; only the canonical exact-geometry challenge
retained the zero-of-11,506 result.

Freezing a stricter native threshold above the maximum of 5,753 alternating
development controls left zero crossings in the other 5,753 controls and 26
Google images missed by the first carrier above threshold. That retrospective
split is promising but not a certified operating point: the corpus had already
been inspected while developing the second carrier, so a fresh source- and
time-disjoint challenge is still required.

An external natural-image challenge then rejected that strict native threshold.
It accepted none of 1,500 unique Open Images controls and none of 12 Wikimedia
controls, but one of 637 content-deduplicated Picsum controls. The original
Picsum directory contained 1,000 filenames but only 637 unique hashes; the one
crossing appeared twice under different ids and is counted once. Its RGB and
quadrant contributions were highly uneven, unlike the small temporal-positive
set, but that post-error observation cannot retroactively define a gate. The
strict native threshold remains excluded from runtime.

The original exact-1024 threshold also failed on those external controls: four
of 1,500 Open Images and four of 637 unique Picsum images crossed it, including
native 1024-by-1024 cases. The earlier zero-of-11,506 canonical result was
therefore specific to that heterogeneous control composition, not evidence that
exact geometry alone makes the carrier specific. Requiring agreement between
the two independently fitted early templates did not close the gap; it retained
three temporal positives but accepted one Open Images and two unique Picsum
controls. The second direction remains a reproducible Google-cohort signal, not
a blind detector.

Requiring the orthogonal score to survive a JPEG-95 round trip was not a
specificity gate either. It retained four of the five temporal positives but
also retained all four Open Images and all four unique Picsum crossings at the
original threshold. The natural-image confound is itself transport-stable.
Together with the failed low-rank, two-template, channel-balance, quadrant, and
two-carrier-plane variants, this exhausts mean-template consensus for the
second direction on the current data.

The orthogonal carrier is not a cyclic shift or color permutation of the
shipped template. Its best absolute cyclic correlation was 0.212, Fourier
magnitude correlation was 0.480, and weighted phase lock was 0.200. Dominant
frequencies were axial, led by `(0, +/-8)`, `(0, +/-3)`, and `(0, +/-5)` rather
than the first direction's broader structure. A horizontal-axis-only ablation
retained all five temporal detections and no reserve exact-geometry control, but
then accepted three of the 11,506 canonicalized heterogeneous controls. The
errors included an OpenAI-labeled image and two watermark-remover outputs. The
full orthogonal template is more specific than its strongest axial component.

A final low-rank variant tested whether that direction contains several payload
states rather than one mean carrier. SVD bases of ranks 1, 2, 3, 4, 6, 8, 12,
and 16 were fitted only on early 1024-by-1024 misses. Rank was selected on a
separate June interval and a separate control third, then reported on July
misses and the final control third. The selected rank 12 retained 2 of 38
validation misses and 4 of 95 test misses, with no control error. Rank 2 happened
to retain 6 test misses but only one validation miss and therefore could not be
selected without post-test tuning. A multi-state subspace did not improve the
evidence-grade operating point.

Deflation then tested whether a third stable linear carrier remained. After
projecting both the shipped and second orthogonal templates out of every
normalized tile, early and July residual averages correlated only 0.0003. The
early residual template retained none of 95 temporal misses and none of 81
reserve exact-geometry controls. Within this 1024-by-1024 temporal corpus, the
linear periodic model therefore supports two reproducible carrier directions,
not an open-ended sequence of mean templates.

Treating those directions as a single two-dimensional carrier plane did not
provide a safer weak-signal rescue. A positive-quadrant Euclidean norm used the
two independently frozen carrier thresholds as axis scales, then froze its own
boundary above 81 exact-geometry development controls. It rescued none of the
Google images below both component thresholds and accepted one of 81 reserve
controls. The two directions must retain separate operating points; combined
subthreshold energy is rejected.

The five temporal positives above the original second-carrier threshold also
had unusually balanced normalized R, G, and B contributions: their worst
channel ranged from 0.297 to 0.320, while seven of eight native control errors
were below 0.297 and were commonly blue-heavy with a weak green contribution.
A clean chronological validation did not justify turning that observation into
a gate. A template fitted on half of the early positives produced no accepted
examples in the other early half at the threshold frozen from development
controls, despite accepting six July examples and no reserve control. The
channel-balance rule is therefore a post-test hypothesis, not detector logic.

A phase-only follow-up removed observed Fourier magnitude and averaged phase
agreement using only the early template's fixed spectral-energy weights. The
phase threshold alone retained 2 of 31 early validation misses and 10 of 95
July misses, but accepted one of 81 reserve exact-geometry controls. Requiring
both phase agreement and the independently calibrated linear projection
removed that control error and all external natural-image errors. The frozen
conjunction retained 1 of 31 early validation misses and 3 of 95 July misses,
accepted none of 162 heterogeneous exact-1024 controls, none of 1,500 unique
Open Images, none of 637 unique Picsum images, and none of 12 Wikimedia images.
Applied outside its declared geometry it accepted 3 of 11,506 heterogeneous
native controls, all at non-1024 sizes. This is the first independently
challenged phase-and-amplitude candidate for the second direction, but three
temporal positives are not enough to ship a runtime branch.

Across all 443 exact-1024 Google images, the frozen conjunction found 11 that
the shipped carrier missed and 16 that both branches accepted; 244 were found
only by the shipped carrier and 172 by neither. The candidate therefore adds
signal rather than merely confirming the first direction, but its incremental
native recall is still small.

Swapping the two early fit halves and recalibrating both thresholds retained 4
of 31 opposite-half validation misses and 4 of 95 July misses, again with none
of 81 reserve exact-geometry controls passing the conjunction. This supports
fit-split signal reproducibility, but the swapped operating point accepted one
of 92 unique exact-1024 Open Images controls in its external challenge; none of
637 unique exact-1024 Picsum controls crossed. A post-hoc consensus of the two
fit halves would require another untouched control set, so it is not selected
from these exposed results. The candidate remains research-only.

An initial two-expert rule required both independently fitted templates to pass
their own separately calibrated linear and phase thresholds. The next
nonoverlapping Open Images validation-id block was downloaded without scoring;
2,000 candidates yielded 100 unique native 1024-by-1024 controls. The first
expert accepted none, the second accepted one, and their conjunction accepted
none. That rule appeared to add 5 detections across the 443 exact-1024 Google
files, including 2 of 95 July fixed-carrier misses.

A subsequent content-group audit invalidated those recall numbers. The 443
files contained three exact decoded-pixel duplicate pairs and ten
perceptual-hash groups at a conservative 16-of-256-bit screening distance. One
early near-duplicate pair had been split between the two expert fits, so the
experts were not content-independent. After allocating whole groups to one
side, selecting one representative per group, and recalibrating both experts
from the same untouched development controls produced templates correlated at
0.720, and the consensus retained only 6 of 443 Google files. It added 2 files
missed by the shipped carrier and retained only 1 of 95 July misses. It still
accepted no conjunction among 81 reserve exact-geometry
controls, 92 exact-1024 Open Images, 637 exact-1024 Picsum images, or the 100
new Open Images controls. The corrected result is specific but far too small
to ship.

Removing the exact-geometry restriction invalidated even that corrected
consensus. Across all 11,506 heterogeneous native controls, the group-separated
experts accepted 5 and 6 respectively, and their conjunction accepted 2. Both
consensus errors were at non-1024 geometries. The result is explicitly an
exact-1024 research candidate and does not supply a universal-size detector.

A post-hoc margin above both arbitrary-geometry collisions retained the two
corrected incremental exact-1024 positives. Applying that frozen margin to all
4,698 Google files accepted 28 and added 16 files missed by the shipped carrier
across 11 native geometries. A full-corpus perceptual audit assigned those 16
files to 16 separate content groups, none containing a shipped-carrier-positive
variant. The same margin accepted none of 1,500 Open Images, 637 unique Picsum,
or 12 Wikimedia native views. This is the first concrete all-size hypothesis
for the second direction, but the margin was selected after all 11,506
heterogeneous controls were exposed and the natural corpora had already
challenged related experts. It requires a new content-deduplicated AI-control
corpus before it can be treated as validation or runtime logic.

The next source-independent model-cohort challenge sharply bounded that
hypothesis. With no exact-byte overlap against the Spaces inventory, the
original group-separated conjunction accepted 2 of 589 public Gemini 3.1 Flash
Image Preview images, none of 520 Nano Banana Pro Preview, and none of 280
DALL-E 3 images. The two Gemini hits were visually distinct diverse images at
1408 by 768. Each Google cohort also included solid-color and gradient probes
over several native geometries. The post-hoc 1.033 strict margin rejected every
cohort image: Gemini reached 1.011, Nano Banana 0.941, and DALL-E 3 0.514. The
base rule therefore has weak transfer to a current Gemini cohort but still
collides with 2 of 11,506 arbitrary-size controls; the strict rule removes both
the controls and the current-Gemini transfer only through a post-test threshold.
Neither is a validated universal Google-model or cross-provider SynthID
detector. The 16 Spaces all-size hits and two public Gemini base hits are useful
hard positives for epoch analysis, not justification for implementation.

The near-duplicate audit also provided a small mechanistic diagnostic. In all
three perceptually matched pairs where only one variant crossed the shipped
carrier, subtracting the shipped variant from the non-shipped variant reduced
the shipped-template direction and increased both independently refitted
second-template directions. The normalized mean paired difference correlated
-0.568 with the shipped template and 0.252 and 0.264 with the two second
templates. Three pairs are insufficient for inference; the two-sided sign-test
result is 0.25. The paired result supports a carrier-state interpretation but
is not detector validation.

All three temporal detections from the original first-expert conjunction survived JPEG round trips
at qualities 95, 85, and 70. None survived WebP 95 or a 0.75x down-and-up resize
round trip. A symmetric transform challenge prevented promoting that positive
retention to a JPEG claim: one of 162 heterogeneous exact-1024 controls crossed
the unchanged rule after JPEG 85. The 92 exact-1024 Open Images and 637 unique
exact-1024 Picsum controls stayed below threshold in every view. The candidate
was native-only; fixed-lattice JPEG retention is diagnostic, not a certified
transport operating point. The transform result applies only to that
first-expert rule. The corrected group-separated experts have not been
transport-calibrated, so it supplies no robustness claim for them.

The corrected carrier was then tested causally. The normalized sum of the two
content-group-separated expert directions correlated 0.927 with each expert.
Subtracting it from all 16 strict incremental hits cleared every linear and
phase component on every image; the same amplitudes with a one-pixel cyclic
shift cleared none. Selected amplitudes were 5.0 through 7.75 integer levels,
with median 6.5. Fidelity ranged from 58.92 through 69.97 dB PSNR and 0.99961
through 0.99996 SSIM. The carrier itself was transform-fragile: only one of 16
sources remained strict after JPEG 95 or 90, and the aligned edit cleared that
one in both views while the shifted edit did not. No source remained strict
after JPEG 85, WebP 95, or a 0.75x resize round trip.

A joint ablation then started from all 28 strict second-carrier hits. It first
suppressed the shipped carrier to the frozen -0.25 target where required, then
suppressed the second direction until all four expert components were below
their base thresholds. The original cohort contained 12 first-carrier and 28
strict second-carrier detections; the aligned candidates contained zero of
either. The shifted controls left one first-carrier and seven strict
second-carrier detections. Median fidelity was 61.69 dB PSNR and 0.99984 SSIM;
the minima were 58.15 dB and 0.99910. The two edits did not reactivate one
another. Together with the failed third-carrier fit, the current linear native
16x16 hypothesis is locally exhausted as two jointly controllable states.
These are detector-score interventions, not Google-oracle removals.

### 2026-08-12: Registered color and phase-lock challenge

A crop-specific research branch tested whether the recovered 16x16 carrier is
better represented in a perceptual color space or a shift-tolerant directional
transform. Every branch reused the direct RGB period-and-phase registration,
fitted on the first 30 positives, selected on the next 20 positives and 200
controls, and reported the remaining 50 positives and 299 controls separately.
Only records whose recovered period was 15.5 through 16.5 were eligible.

The frozen RGB template's channel norms were `0.887:1.000:0.930` for R:G:B,
including `0.903:1.000:0.916` above FFT radius 4.5. An SVD assigned 79.80% of
template energy to a nearly equal-channel component, 18.39% to a
green-vs-magenta component, and 1.81% to a red-vs-blue component. The common
component had strong diagonal energy, while 98.6% of the coarse
green-opponent orientation energy was axis-aligned. This measured carrier does
not support a fixed `0.85:1.00:0.70` channel rule or an exclusively diagonal
decoder.

Nonlinear cube-root LMS coordinates from the OKLab transform exposed the
signal most strongly in the direct color-fold comparison, but the
development-selected full-vector candidate retained 32 of 50 final positives
and accepted six of 299 controls. This is evidence that nonlinear color
projections can improve carrier SNR, not evidence that the deployed embedder
uses Lab or OKLab internally.

A three-level DTCWT probe retained all six orientations rather than only the
`+/-45` degree pair. Its development-selected near-vertical green energy
candidate retained 37 of 50 final positives and accepted eight controls. The
green-opponent near-vertical candidate retained 39 positives and accepted one
control. Separating energy, normalized complex correlation, and phase-only
coherence changed the development choice to all-orientation green-opponent
phase coherence, which retained 37 positives and accepted four controls. The
directional representation therefore exposes real carrier structure but fails
the low-false-positive gate.

An explicit FFT probe then froze 8, 16, 32, or 64 carrier bins from the fit
positives and compared signed cosine phase coherence with the proposed
`abs(cos(delta_phase)) > 0.90` rule. The absolute rule never produced a usable
operating point; treating phase and antiphase as equivalent increased the null
overlap. With 64 bins, the development-selected green-channel signed cosine
retained 26 of 50 final positives and initially accepted none of 299 controls.
That result was underpowered because only 39 of 499 controls selected the
eligible period-16 branch.

The green phase candidate and its threshold were frozen before a full
period-and-phase search over 1,000 COCO controls. It accepted five controls.
Among the 129 eligible controls, the maximum score was `0.6910271977636098`,
above the frozen `0.5443545641095117` threshold. Raising the threshold above
that maximum retained only eight of 100 positives. Explicit registered phase
lock is therefore a useful diagnostic of this carrier family but is rejected
as a standalone detector. Because it shares the RGB registration stage, it is
also not independent confirmation of the carrier.

### 2026-08-12: Green phase perturbation and blind SWT removal

A second removal probe tested the specific claim that changing only phase at
the recovered Green-channel carrier bins can preserve spectral magnitude and
image fidelity while defeating detection. It used the 100-image public Gemini
set at native period 16 and modified conjugate FFT pairs together so the
inverse remained real. Before pixel quantization, the maximum absolute change
in FFT magnitude was `5.24e-10` for the 32-pair inversion. After clipping and
rounding to 8-bit pixels, the maximum magnitude change relative to the largest
source coefficient was 0.175%.

The native periodic detector accepted 59 of the 100 sources. Inverting the 32
strongest unique Green-channel carrier pairs left one of those 59 positive.
Across all 100 images, PSNR was at least 43.36 dB and SSIM at least 0.99048;
the medians were 52.79 dB and 0.99739. Median signed phase coherence moved from
0.485 to -0.371 and the median full RGB tile score moved from 0.230 to 0.041.
This reproduces the proposed fidelity and local-score effect against the
recovered periodic detector.

It does not establish SynthID removal. The current test targets the same
carrier used by the local score, and earlier wider complex-spectrum and
phase-projection candidates already cleared local phase models at 51.94-56.03
dB while remaining detected in healthy Google verifier sessions. The 32-pair
Green candidate is therefore a preregistered oracle candidate, not a remover.
The official SynthID-Image description also identifies a post-hoc neural,
content-dependent encoder-decoder rather than a fixed Fourier codebook; it
does not document phase bins as its decision surface.

The preregistered Google-oracle challenge then used three of the strongest
local positives. Each fixed group contained the original, its 32-pair Green
phase inversion, and an RMS-matched random-noise sham. All three phase
candidates fell below the local `0.17357069773071196` threshold, with PSNR
from 55.41 to 57.86 dB and SSIM from 0.99709 to 0.99807. The three shams
remained locally positive. Nine files were submitted to Gemini's `Verify with
SynthID` tool in separate chats with the unchanged prompt. Eight verdicts
persisted and all eight reported Google AI signals. This includes all three
phase candidates, both recoverable source controls, and all three shams. The
second source request was submitted, but its chat did not persist and no
verdict could be recovered; it was not retried outside the fixed upload budget.

This falsifies the operative removal claim: Green-carrier phase inversion
clears the local periodic correlate on three of three images but clears the
Google oracle on zero of three. It is evidence that the local template observes
one stable component or correlate of SynthID, not that these FFT coefficients
form the proprietary decoder's sufficient decision surface. Do not productize
the Green phase inversion or report its local score change as SynthID removal.

The three oracle-confirmed phase candidates were then treated as adversarial
hard positives for detector analysis, without using them to relax the existing
runtime thresholds. Both runtime modes missed all three. Their native scores
were 0.126-0.160 against the 0.174 threshold; their registered decision scores
were 0, 0, and 0.553 against the threshold of 1. A phase-invariant magnitude
profile was therefore tested as a possible rescue. The development-selected
all-bin RGB profile retained 39 of 50 final positives but accepted one of 299
final Open Images controls and still missed all three hard positives.

A frozen-template, top-32 Blue-magnitude profile did accept all three hard
positives and initially accepted none of 499 Open Images controls, but this was
a post-hoc hard-positive-selected branch with low ordinary sensitivity: 17 of
70 development and final positives. Its frozen threshold accepted 39 of 1,000
fresh COCO controls. Combining it with a native-score floor of 0.10 happened to
accept none of all 3,000 COCO controls while retaining the three hard positives,
but failed the next independent challenge: 90 of 11,506 metadata-labeled
non-Google controls passed both gates. Accepted controls included Adobe
Firefly, local diffusion, Samsung, Microsoft, and several other generator
families. The phase-invariant magnitude rescue is therefore rejected as a
provider-specific detector; it measures a shared periodic rendering family.

Channel separation exposed a narrower attack-specific rescue. The Green-only
phase inversion leaves Red and Blue pixels unchanged, but the runtime RGB score
normalizes all three channels jointly. On the three oracle-confirmed hard
positives, separate signed correlations remained 0.781-0.804 in Red and
0.798-0.825 in Blue while Green moved to -0.400 through -0.645. A Red threshold
of `0.3770237083693908`, frozen above 200 Open Images calibration controls,
retained 16 of 70 development and final ordinary positives and accepted none
of the remaining 299 Open Images controls. It also accepted all three oracle
hard positives and none of 3,000 fresh COCO controls.

Red correlation alone remained a generic periodic-renderer correlate: it
accepted 96 of 11,506 metadata-labeled non-Google controls. Every one of those
96 had positive Green correlation. The attack-specific conjunction of Red at
or above the frozen threshold and negative Green accepted none of 15,005 total
controls across three independent corpora, while retaining all three
oracle-confirmed phase candidates. Across the 100-image synthetic inversion
probe, it recovered 19 of the 58 locally detected sources that the attack had
moved below the RGB threshold. One additional attacked image still passed the
existing detector.

This channel-discordance rule is the first zero-control rescue for the
oracle-confirmed hard positives, but it is not added to runtime. Its form was
motivated after inspecting the attack, only three examples carry external
oracle labels, and its synthetic rescue sensitivity is 19 of 58.

A follow-up challenge registered that frozen rule over all 256 modulo-16 phase
offsets and six fixed carrier periods: 8, 10.4, 12, 16, 20, and 24 pixels. The
Red-only form accepted one of 3,000 COCO controls during phase registration;
period search added no further acceptance. The accepted control scored 0.395
in Red but only 0.267 in Blue. This rejects Red-only registered rescue at the
existing threshold.

The previously frozen `min(Red, Blue) >= 0.3770237083693908` candidate, still
combined with negative Green, accepted none of the same 3,000 controls under
phase-only or phase-and-period registration. Its maximum registered control
score was 0.342. A symmetric transform stress test then evaluated all twelve
identity, crop, JPEG, and resize views for 100 fixed Open Images controls. None
of the 1,200 inputs was accepted; the maximum was 0.332.

The stricter rule retained all three oracle-confirmed hard positives at native
scale and after offsets of 1x1 and 7x11 pixels or a centered 5% crop. Period
registration retained all three after 0.65x, 0.75x, 1.25x, and 1.5x resizing.
It retained none after 0.5x resizing or JPEG round trips at qualities 95, 85,
and 70. Every source and RMS-noise sham remained negative in all twelve views.
The small-control zero was not sufficient for the complete period bank. A
later five-period search over all 11,506 metadata-labeled non-Google controls
accepted 16 at the original 0.377 threshold: two at period 20 and 14 at period
24. The maximum `min(Red, Blue)` score was 0.435813. This rejects one common
threshold across periods 10.4 through 24.

A conservative high-score branch was then frozen strictly above that measured
maximum. It retained all three hard positives at periods 12, 16, 20, and 24,
corresponding to 0.75x through 1.5x resizing, identity, and the three crop
views. A symmetric full-corpus resize challenge accepted none of 11,506
controls at 0.75x, 1.25x, or 1.5x; the respective maxima were 0.431352,
0.393949, and 0.372036. The 0.65x path needed separate treatment. Its native
period-10.4 null maximum was 0.361367, below all three hard positives, but a
symmetric 0.65x resize round trip raised one control to 0.397396, above the
weakest hard positive at 0.394911.

That collision motivated an exploratory low-period channel-ratio gate:
`min(Red, Blue) >= 0.3770237083693908`, negative Green, and
`Blue >= 1.1 * Red`, only at period 10.4. It retained all three 0.65x hard
positives. Among all 11,506 controls, the largest score satisfying the ratio
gate was 0.355165 natively and 0.354955 after the symmetric 0.65x round trip,
so neither challenge accepted a control. The combined exploratory rule also
retained all three hard positives in the identity, three crop, 0.65x, 0.75x,
1.25x, and 1.5x views, while every corresponding source and RMS-noise sham
remained negative.

Applying the combined rule without further tuning to the 100-image synthetic
Green-inversion cohort exposed its low sensitivity. Among the 58 attacks that
moved a natively detected source below the native detector threshold, the rule
recovered 4 at 0.65x, 5 at 0.75x, 16 at identity, 11 at 1.25x, and 10 at 1.5x.
The three oracle source images are members of that cohort; only one additional
synthetic source passed the 0.65x branch. The rule is therefore a
high-specificity rescue for a strong residual R/B carrier, not a general
solution to Green-phase perturbation.

A source-split Fisher projection then tested whether a learned three-channel
direction could improve that sensitivity. Red, Green, and Blue correlations
were standardized separately by scale, and the projection was fitted from 34
training attacks and 5,719 training controls. Per-scale thresholds were frozen
above every training-control score. They accepted 11 of 23,148 views from the
5,787 test controls. On the seven source-held-out attacks, the projection
recovered 0 at 0.65x, 0 at 0.75x, 2 at 1.25x, and 2 at 1.5x. It is both less
specific and less sensitive than the explicit rule and is rejected.

Spatial consistency at 0.65x was also insufficient. At the globally selected
period-10.4 phase, a 2x2 and 4x4 aligned grid measured local R/B strength,
negative-G coverage, and `Blue / Red` coverage on all 537 controls in the
global-score tail at or above 0.20 and on all 58 synthetic native-detector
misses. Every individual feature overlapped the control tail. Within a small
fixed grid of global floors and integer coverage counts, the best zero-control
conjunction recovered four attacks, no more than the simpler global ratio
rule. Lowering the global floor to 0.30 recovered five but accepted two
controls. The spatial branch is rejected as non-improving.

This establishes bounded transform behavior for the attack diagnostic, not a
universal detector. The 0.65x ratio was selected after seeing the full-corpus
collision, the conservative threshold was derived from an exposed control
corpus, the period bank is discrete, only three recovered images have oracle
labels, and JPEG sensitivity is zero in this probe. The next valid gate remains
new oracle-confirmed Green-perturbed positives plus a preregistered untouched
control corpus. The branch stays outside runtime until that gate passes.

An exploratory channel-contrast extension then tested whether the magnitude of
the negative Green response could recover the failed JPEG and 0.5x views. The
metrics were fixed as `min(Red, Blue) - Green` and
`mean(Red, Blue) - Green`; 50 Open Images controls across all twelve transforms
set thresholds before the remaining 50 controls were scored. Calibration over
600 views required thresholds of 0.655 and 0.695. Neither metric accepted an
evaluation control, but both still recovered zero of three JPEG cases at every
quality and only one of three 0.5x cases. A relaxed R/B-only threshold produced
the same result. Negative-G magnitude therefore adds no useful transform
coverage at the measured low-false-positive boundary and is rejected.

Codec-specific templates were then fitted from 50 other public positives after
JPEG 95, 85, and 70 round trips, excluding all three oracle hard-positive
sources. Neither a quality-matched template nor a three-quality bank recovered
any of the nine JPEG hard-positive views. They accepted respectively three and
two of 150 evaluation-control views. The templates primarily captured the JPEG
8x8 grid and were rejected.

Filtering the unchanged 16x16 template to radial frequency 0.5 through 4.5
exposed a real JPEG-surviving component. At a threshold above the first 300
Open Images JPEG views it retained all three hard positives at qualities 95,
85, and 70. The next frozen challenge rejected it: phase search accepted 35
native, 50 JPEG-95, 48 JPEG-85, and 46 JPEG-70 views among 3,000 COCO images.
Cross-band phase agreement did not separate the nine hard-positive views from
those 179 false positives.

A spatial gate required negative Green in all four quadrants and a median
quadrant R/B score above the COCO false-positive maximum. It retained all three
hard positives at JPEG 95 and 85 but none at JPEG 70. On 11,506
metadata-labeled non-Google controls it nevertheless accepted 35 native, 66
JPEG-95, and 88 JPEG-85 views. The errors included genuine camera images and
multiple unrelated generators. Scoring only the unchanged phase also failed,
accepting 38 native and 210 JPEG views among 3,000 COCO images; its maximum
control score of 0.673 exceeded the 0.464-0.609 hard-positive range. The
JPEG-stable band is a general low-frequency or codec lattice, not a
SynthID-specific rescue, and all of these branches are rejected.

The 0.5x boundary was challenged separately with a physically reduced 8x8
template. Projecting the dominant 2x2 lattice out of both observation and
template before scoring R/B with negative Green removed the known nuisance but
not the overlap. Across all 11,506 symmetrically downscaled controls, the
zero-control threshold was 0.396. The three oracle hard positives scored only
0.277-0.288. Channel separation therefore cannot rescue the rejected period-8
branch.

An untouched 2026-08-07 through 2026-08-10 temporal cohort then challenged the
runtime template and thresholds. Cohorts were recomputed from explicit
provenance because a provider-only split incorrectly placed mixed Google/OpenAI
lineage in the negative set. The frozen native detector accepted 183 of 252
files with explicit Google LLC SynthID provenance and 3 of 488 with explicit
OpenAI-only SynthID provenance. Scale registration accepted 130 and 1,
respectively. Among 170 files with no SynthID provenance assertion, native and
registered modes accepted eight and two. These operational errors are not
provider-oracle negatives and cannot establish the true false-positive rate.

The native detector remained positive on 153 of 252 explicit-Google images
after a symmetric 0.75x resize round trip, 66 after JPEG-95, and 54 after
WebP-95. A 5% center-crop round trip retained none. This fresh temporal result
confirms a real, partly codec-tolerant Google pixel signal across many stored
sizes, while rejecting the stronger claim that the fixed-phase runtime score
is a universal SynthID decoder. The post-hoc strict second carrier did not
generalize: it accepted only 2 of 234 Google-platform positives and none of 689
provider-split controls. Applying the already frozen scale-registration path
after the crop recovered only 8 of 252 explicit-Google files and accepted none
of 488 explicit OpenAI-only files. The existing scale search does not solve the
crop phase boundary.

A diagnostic joint-scale-and-phase probe then reversed the known 0.95 crop
scale and scored the carrier at the cyclic phase predicted from the removed
border. It reached 0.859 AUC for explicit Google provenance against all other
temporal cohorts. A post-hoc threshold above all 671 other rows retained 37 of
252 explicit-Google rows. Exhaustively searching all 256 tile phases was weaker
at 0.839 AUC and retained 41 rows above its own zero-control threshold. The
fixed geometric phase supplies a plausible crop mechanism and a candidate for
the next preregistered temporal challenge, not a calibrated runtime mode: both
the transform and reported operating point have now observed this cohort.

A blind three-level `db2` SWT probe then soft-thresholded only the Green
horizontal, vertical, or diagonal detail bands at 0.25, 0.5, 1, and 2 times the
band's robust noise estimate. Diagonal HH thresholding left all 59 original
positives detected at every strength. Vertical thresholding cleared four at
strength 1 and five at strength 2, but the respective minimum fidelity fell to
32.57 dB and 0.96971 SSIM, then 28.52 dB and 0.93646. At strength 1, vertical
score reduction exceeded diagonal reduction on 97 of 100 paired images
(`p = 2.63e-25`, exact two-sided sign test). Blind HH soft-thresholding is
therefore rejected; the stronger vertical response is consistent with the
separately measured axis-aligned green-opponent carrier component.

### 2026-08-13: patent architecture and amplitude confound audit

The official paper intentionally leaves the neural network architecture
unspecified, but a related DeepMind patent family with overlapping authors gives
more specific architectural evidence. Its image example constructs a
content-dependent residual as `x' = x + g(x)`, describes a U-Net-like encoder
and convolutional decoder, injects a message or secret into intermediate layers,
and trains clean and watermarked pairs under the same sampled transformations.
It also permits a bank of paired networks in which one decoder is deliberately
unable to recognize another pair's mark. These patent alternatives are not
proof of the exact production implementation, but they invalidate the working
assumption that SynthID must reduce to one provider-independent, fixed
spread-spectrum key. Multiple Google states and a distinct OpenAI carrier are
compatible with one technology family.

This changes the role of the shipped 16-by-16 template. It is a validated linear
expert for one observable encoder state, not a surrogate for the official
nonlinear detection logit. A universal local detector should be designed as a
versioned union of independently calibrated experts, with family-wise
false-positive control and an explicit abstention region. New experts still
need source-matched clean counterfactuals or provider-oracle labels; metadata
absence is not a negative label.

A symmetric transform-stability experiment challenged a simpler rescue. The
weak oracle-positive `07.png` scored 0.1248 natively, 0.0288 after JPEG 95,
-0.0023 after JPEG 85, 0.1179 after a 0.99x Lanczos round trip, and 0.0791 after
an eight-pixel crop round trip. The oracle-negative Open Images chestnut scored
0.2124, 0.1322, 0.0482, 0.2274, and 0.0601 under the same views. Both signals
degrade similarly under codec and crop operations. Multi-view stability does
not distinguish the weak positive from this natural lattice confound and is
rejected as a rescue gate.

Raw folded-tile amplitude was also tested because `07.png` had a norm of 6.924,
compared with 3.017 for the chestnut and 0.874 for the strongest tested resized
Kodak confound. The full corpus rejects a general amplitude gate. Among 4,698
Google rows, the current score threshold accepts 3,928, but accepted norms run
as low as 0.712. A post-hoc branch at `score >= 0.121` and `norm >= 5` would add
190 rows below the current threshold, yet an observed-geometry development
control already reaches score 0.120911 at norm 15.339. The margin to `07.png`
is under 0.004 and there is no untouched validation set. Amplitude is useful as
a support diagnostic, but neither a veto nor this low-score rescue is eligible
for runtime.

The next defensible Google experiment is consequently a preregistered
multi-expert cascade: preserve the current carrier unchanged; train candidate
experts only on source-grouped paired or oracle-labeled data; calibrate every
expert and their union on untouched natural, resize, codec, and provider
controls; and return `unknown` when neither the watermarked nor unwatermarked
hypothesis clears its calibrated boundary.

The newly published MarkNull work supplies an independent no-box removal
baseline. It reports 100% attack success on 20 Imagen-3 images checked through
Gemini's verification flow. The attack uses a clean Stable Diffusion proxy to
invert each image, optimize a bounded latent against a noise-latent alignment
loss plus LPIPS, MSE, and SSIM, and decode it again. Its milder SynthID setting
still reports only 25.36 dB PSNR and 0.80 SSIM. It therefore strengthens the
case for latent regeneration as a detector-independent fallback, but it does
not meet this project's pixel-only or fidelity gates.

The associated [public repository](https://github.com/JieJayCao/MarkNull)
contributed 50 metadata-free 512-by-512 Gemini/Imagen source candidates, but no
attacked SynthID pairs in the inspected Git tree. The paper reports only 20
oracle-evaluated sources and does not map those verdicts to filenames, so all 50
remain source-asserted candidates rather than individual oracle labels. Both
shipped experts detected zero of 50. More precisely, the fixed runtime expert
reported unsupported because 512 by 512 is below its calibrated pixel-count
floor. Its raw diagnostic score ranged from -0.1044 to 0.1058. Scale
registration supported all 50 and accepted none, with its highest decision
statistic at 0.9574 below the frozen 1.0 threshold. The shared scorer in
`scripts/synthid_runtime_expert_scores.py` now records this distinction and
exports both observations from one pixel decode.

An external-source split nevertheless exposed a highly reproducible period-4
component. Templates from the first and second 25 images correlated at 0.958.
A template fitted only on the first half accepted 22 of 25 held-out candidates
at a threshold above 250 canonicalized COCO development controls and accepted
zero of a separate 250 COCO controls. The broader challenge rejected watermark
specificity: it accepted two of 300 heterogeneous AI controls and five of 282
content-deduplicated internal negative-test controls at the same frozen
threshold.

The mechanism diagnostics also identify the confound. The period-4 mean placed
82.4% of its non-DC spectral power in the horizontal Nyquist bin, and stable
correlations at larger even periods vanished after projecting out all period-4
harmonics. Its strongest channel was Blue rather than Green, and its best
phase-aware correlation with a fourfold-reduced shipped carrier was only 0.375.
This is an axial renderer or resampling signature, not evidence for another
validated diagonal SynthID carrier. It is retained as a hard-negative feature
and excluded from the runtime expert bank.

A schema-versioned conformal cascade now implements the required aggregation in
`scripts/synthid_conformal_cascade.py`. Every configured expert must emit an
explicit supported score or unsupported result. For presence, the smallest
empirical clean-null upper-tail p-value receives a Bonferroni correction over
the supported expert bank. For absence, the largest lower-tail p-value against
each expert's positive calibration is used, so every possible encoder state
must be rejected. `not_detected` is disabled unless the calibration manifest
explicitly declares complete version coverage and every expert supports the
query. Input and output manifests are hash-pinned for reproducibility.

The first source-disjoint smoke test used only the unchanged fixed 16-by-16
expert. Alternating content hashes supplied 2,347 source-asserted Google
calibration rows and 2,346 evaluation rows. The existing observed-geometry COCO
partitions supplied 3,000 development controls and 2,000 final controls. At a
two-sided alpha of 0.001 with incomplete-version coverage, the cascade detected
2,084 of 2,346 Google rows and one of 2,000 final natural controls; all other
rows abstained and none returned `not_detected`. The corresponding frozen
runtime threshold detected 1,954 Google rows and zero final controls. The
conformal point therefore recovers 130 additional source-asserted positives at
the cost of one natural false positive. It is statistically compatible with
the target 0.1% FPR, but it is not a runtime replacement: the corpus has already
influenced detector development, Google labels are provenance assertions rather
than per-file oracle verdicts, and a multi-expert union still needs a fresh
family-wise challenge.

The fixed and scale-registered experts were then evaluated together on their
frozen source-disjoint resize challenge. Current production scores were
recomputed from pixels for all 355 transformed Google candidates and 499
untouched Open Images controls. Sorting each label by source identifier and
alternating rows produced 178 positive and 250 negative registered-calibration
examples, followed by 177 positive and 249 negative evaluation examples. The
fixed calibration used 4,516 Google sources after excluding every evaluation
source and 2,980 observed-geometry development controls. At the unchanged
runtime thresholds, fixed accepted 8 evaluation positives, registered accepted
116, and their union accepted 117 of 177 with zero of 249 controls. On the full
descriptive corpus the corresponding counts were 15, 229, and 231 of 355, again
with zero of 499 controls. Registered scale search therefore supplies 109
source-disjoint evaluation positives that the fixed expert misses; the fixed
expert adds one that registered misses.

Applying the empirical cascade at alpha 0.01 was less specific. It accepted 122
of 177 evaluation positives but also five of 249 controls. All five control hits
came from the registered branch below its frozen decision threshold. The
250-control calibration provides only 0.00398 p-value granularity, and lowering
the operating point merely traded five additional positives for five false
positives. This conformal operating point is rejected. The implementation is
retained as versioned calibration infrastructure, while the independently
frozen production thresholds remain the valid two-expert decisions.

A wider calibration reused the already saved component reports for 11,273
supported, symmetrically resized Spaces controls. The stored high-band value
made the current registered decision score exactly reconstructible throughout
the relevant upper tail; rows screened below the historical 0.1182 amplitude
ratio cannot affect this tail. At alpha 0.001, the two-expert cascade accepted
119 of 177 evaluation positives and zero of 249 controls. The two additions
over the runtime union had registered scores 0.9830 and 0.9731, with corrected
clean-null p-values of 0.000710. This initially appeared to be a non-failing
multi-expert calibration point, but the positive challenge had already been
exposed. The alpha, expert bank, and thresholds were frozen before the fresh
test below.

The decisive reserve used the next 1,500 sorted Open Images validation IDs,
3,500 through 4,999, frozen before scoring. Download and geometry checks yielded
1,000 registered-supported controls with no decoded-pixel hash overlap against
the prior 1,499 Open Images controls. Registered scale search accepted zero.
The fixed runtime threshold accepted 7 of the 81 controls in its supported
geometry, or 7 of 1,000 overall, so a simple OR of the two experts also accepted
seven. The frozen alpha-0.001 cascade was worse: it accepted nine controls,
eight through the fixed tail and one registered-only row at score 0.8690.
The candidate is rejected. In overlapping geometry, fixed-only evidence cannot
be added to registered evidence without a new specificity mechanism; keeping
the fixed detector available as a separately scoped diagnostic does not make
their union safe.

The fixed-only geometry does not provide a fallback. Among 89 saved non-Google
Spaces controls from 10 to 18 megapixels, where registered mode is unsupported
but fixed mode is supported, 10 crossed the fixed threshold. The errors span
multiple generator families. A geometry router therefore cannot extend recall
by selecting fixed whenever registered is unavailable.

`scripts/synthid_routed_expert_bank.py` implements the surviving conservative
policy without changing the standalone detectors. When registered mode is
supported, only its frozen threshold can produce `detected`; a fixed-only
crossing is retained in the report as `fixed_only_ambiguous`. When registered
mode is unsupported, fixed evidence is likewise reported but the bank abstains.
The bank never returns absence because encoder-version coverage is incomplete.
On all 355 transformed positives and 499 reserve controls it detected 229 and
zero. On the fresh 1,000-image reserve it detected zero and explicitly routed
the seven fixed crossings to abstention. This is a safe expert-bank baseline,
not a universal SynthID decoder: scale 0.5, unmeasured encoder states, and the
separate OpenAI family still require additional validated experts.

Directly enlarging registered mode's pixel ceiling also failed. The unchanged
registered components were computed on every saved 10-to-18-megapixel example:
37 Google candidates and 89 non-Google Spaces controls. The frozen threshold
accepted one positive and zero controls. Amplitude passed for 28 positives and
the selected spatial and spectral periods agreed for 21, but only three cleared
the high-band gate. Large full-frame spectra therefore lose almost all recall;
the absence of control errors does not justify widening the declared geometry.
Phase-aligned window aggregation inside the calibrated range is the next
bounded large-image hypothesis. That hypothesis was then challenged with every
phase-aligned 2,048-square window. Median-window scoring retained 36 of 37
positives but also 10 of 89 controls; requiring every window to pass retained
36 positives and eight controls. The false lattices are spatially coherent, so
window consensus cannot validate a large-image fixed fallback. Large images
outside registered scope remain abstentions.

A final frozen full-frame alternative isotropically reduced each large image to
eight megapixels with area interpolation before applying the unchanged
registered detector. This maps a native period 16 into approximately 10.7-14.3
pixels, inside the calibrated period range, without discarding spatial regions.
It still accepted only 1 of 37 positives and zero of 89 controls, and the one
hit was the same 4096-square image detected natively. Amplitude passed for 26
positives, period agreement for 16, and high-band agreement for only three.
Pre-resizing therefore does not recover the missing evidence and adds harmonic
ambiguity; no target-size sweep is justified on this exposed cohort.

The scale-0.5 failure was also rechecked without the registered detector's
sequential amplitude gate. High-band correlation was computed for all 65
period-8 positives and all 499 reserve controls. Positive and control medians
were 0.0184 and 0.0213, with ranges -0.2381 to 0.1930 and -0.1734 to 0.2148.
Normalized amplitude also overlapped: positive median and maximum were 0.4611
and 0.9259, compared with 0.3824 and 1.6296 for controls. Requiring amplitude
and high-band evidence jointly therefore cannot recover this scale at the
frozen specificity. The 0.5x case remains an explicit blind spot rather than an
eligible lower-threshold branch.

### 2026-08-10: OpenAI periodic-carrier challenge

The OpenAI track repeated the Google carrier method without runtime provenance
features. C2PA selected the research labels, but every candidate score consumed
only decoded RGB pixels. The temporally split corpus contained 4,615 OpenAI
training positives, 1,280 validation positives, 1,364 development-test
positives, 8,668 difficult external-source controls, and 3,000 COCO controls.
The development test is not called a locked holdout because earlier
representations had already exposed it.

High-pass residuals were folded modulo 8x8, 16x16, and 32x32 and normalized as
in the Google fixed-tile expert. A single 16x16 OpenAI template accepted only
one of 1,280 validation positives at the threshold above all 2,287 validation
controls. A 16-template geometry bank improved that to 12 positives but
accepted one of 3,372 development-test controls. The corresponding
development-test sensitivity was 9 of 1,364. The frozen Google template did
not transfer: at its unchanged threshold it accepted 14 validation positives
and 572 validation controls.

The apparent native-size signal was a confound. At 1024x1536, 91.6% of the
learned template energy projected onto a 2x2 periodic lattice, compared with
5.6% for the frozen Google template. Firefly, China AIGC-labeled, local
diffusion, and unidentified hosted-generator controls reached correlations of
0.92-0.98 against the uncorrected OpenAI template. This is operational evidence
for a shared decoder or upscaler lattice, not an OpenAI watermark carrier.

Projecting the 2x2 nuisance lattice out of every 16x16 fold isolated a weaker
epoch-specific residual. The 1024x1536 expert accepted 5 of 55 validation and
10 of 51 development-test positives while accepting none of the full
validation or development-test controls. Nine of the ten development-test
matches occurred from July 12 through July 14; later dates contributed only
one of 27. The 1536x1024 expert accepted 6 of 37 validation and 2 of 51
development-test positives with no development-test controls. Amplitude,
active-support, channel-consensus, payload-invariant power, and spherical
multi-template branches did not improve this frontier without false positives.

The OpenAI branch therefore remains research-only and is not added to the
runtime detector. The next valid training epoch requires same-content
OpenAI-oracle `detected` and `not_detected` pairs, or provider-authorized decoder
labels, followed by a fresh temporal holdout. Source provenance alone cannot
distinguish a true watermark carrier from the measured generator lattice.

### 2026-08-12: OpenAI opponent-color carrier challenge

The RGB periodic and learned branches did not test the hypothesis that a weak
carrier is clearer after the nonlinear LMS transform used by OKLab. A new
validation-only screen compared sRGB, linear RGB, cube-root LMS, OKLab, and an
orthonormal linear opponent basis. For each representation it tested periods
4, 8, and 16 plus whole-vector, single-channel, and chroma-vector folds. Every
candidate fitted a positive-minus-control direction independently in the native
and JPEG-95 views. A positive had to cross both thresholds, with each threshold
strictly above every validation control in the corresponding view.

The development corpus was limited to the three OpenAI native geometries:
1024x1024, 1024x1536, and 1536x1024. It contained 283/140/158 OpenAI
train/validation/test images and 252/145/135 heterogeneous controls. The best
validation candidate was the third cube-root LMS component at period 8. It
accepted 19 of 140 validation positives. The frozen locked test accepted 16 of
158 positives and none of 135 controls in the conjunctive native-plus-JPEG
rule. Native alone accepted 43 positives and one control; JPEG-95 alone accepted
22 positives and a different control. The native and JPEG templates correlated
at 0.890.

The 16 test hits were not a single date, geometry, or content family. They
spanned 11 collection dates; the 1024x1024, 1024x1536, and 1536x1024 cohorts
contributed 2, 8, and 6 hits. A 256-bit DCT-hash audit found two multi-member
perceptual groups in the complete 581-image OpenAI native cohort, but neither
connected an accepted test image to another split. Thresholding against the
union of heterogeneous validation controls and 140 matched 0.75x-down-and-up
OpenAI views did not change the operating point. The 158 matched test views
accepted none while their 16 corresponding source views remained accepted.

The transform boundary is mixed. All 16 source hits remained conjunctively
accepted after WebP-95; the complete test set then contained 26 accepted OpenAI
images and one accepted control. JPEG-90 retained four of the 16. Crop and 0.75x
resize round trips retained none. Searching all 64 cyclic period-8 phases did
not restore geometric robustness: resize retained 2 of 158 positives and one
of 135 controls, while crop retained no positive and one control. The loss is
not merely an unknown modulo origin.

The candidate was also scored without refitting on all 52 action-derived
same-provider controls at these geometries. It accepted one of 52 and none of
the 12 controls explicitly naming `gpt-image 2.0`. The exact-version subset is
only 8/2/2 train/validation/test by the native geometry restriction, and the
absence of `c2pa.watermarked.unbound` remains an action-derived candidate label,
not a provider-oracle negative. These counts cannot establish an OpenAI-specific
false-positive rate.

The dominant exporter lattice was removed explicitly. The raw period-8
template placed 84-90% of its energy in period-2 or period-4 subspaces. For
each image, projecting out the complete period-4 component before refitting
reduced native/JPEG template correlation to 0.564 but still accepted 18 of 140
validation and 15 of 158 test positives with zero test controls. The useful
residual therefore does not collapse to the previously documented 2x2/4x4
generator lattice.

A direct pixel ablation tested causality against the frozen local score. The
normalized native period-8 direction was subtracted only from the sRGB Blue
channel, with a scalar search for the smallest amplitude that cleared both the
native and JPEG-95 thresholds. All 16 locked-test sources cleared. Repeating
the same selected amplitude at the least-correlated cyclic phase cleared none
of 16. Median amplitude was two Blue-channel integer levels, median PSNR was
52.24 dB, and median SSIM was 0.99639. The minima were 40.62 dB and 0.97184,
so even as a local ablation this is not yet a production-quality remover.
More importantly, score causality does not establish that the candidate is the
OpenAI SynthID decoder surface.

An all-size map then evaluated the same fixed native period through partial
modulo folds, so dimensions did not need to be divisible by eight. It accepted
94 of all 7,259 OpenAI-labeled files across 13 of 605 geometries and 9 of 8,670
heterogeneous controls across 7 of 1,534 geometries. The three fit geometries
contributed 76 positives. The largest exported geometries, including 2,265
files at 1086x1448 and 992 at 1254x1254, contributed none. Two corpus rows were
excluded because their decoded geometry no longer matched the indexed metadata.

A scale search was frozen to integer periods 4 through 16 before examining the
all-size results. Its score was the maximum, across periods, of the minimum
native and JPEG-95 correlations. The common threshold was set above every
period-searched validation control, thereby charging the search itself against
specificity. It accepted 13 of 1,854 validation positives and none of 2,227
validation controls. On test it accepted 14 of 1,839 positives and one of 2,162
controls. Every accepted OpenAI image selected period 8; no other period added
a hit. The carrier does not scale continuously with stored resolution.

This is the strongest local OpenAI periodic candidate so far: it is codec
tolerant, survives explicit subperiod deflation, has a phase-specific causal
ablation, and is not explained by date, native geometry, or hash-group leakage.
It is still research-only. It has low sensitivity, fails geometric transforms,
has one all-size test-control collision, and lacks a matching-provider oracle.
The next independent branch should look for a resampling-invariant,
content-dependent residual on source-grouped data rather than broadening the
same period search. No runtime OpenAI pixel expert or remover is justified.

The next branch removed phase from the folded LMS spectrum and required three
views: native pixels, JPEG 95, and a 0.75x down-and-up resize. It screened
magnitude, power, and log-magnitude profiles at periods 4 through 16 on
validation only. The selected period-16 power candidate used a separately
fitted template for each view. It accepted 8 of 140 validation positives and 8
of 158 locked-test positives, with no native-geometry control accepted in
either split. On the complete wide challenge it accepted none of 2,227
validation and 2,162 test Spaces controls, one of all 261 action-derived
same-provider controls, and none of the 118 exact-version `gpt-image 2.0`
controls. Five of its eight test hits overlapped the signed period-8 candidate;
three were additional.

Subperiod deflation bounded that result. Removing the exact period-8 Fourier
support from each period-16 spectrum improved native-to-resize template
correlation from 0.061 to 0.925, but the locked test then retained five
positives and accepted two of 135 controls. Removing period-4 support retained
no test positive. The transform-tolerant energy is therefore part of the same
harmonic family rather than a clearly independent carrier.

Finally, a stricter experiment required one shared phase-free template across
all three views. Validation selected period-15 magnitude, whose independently
measured view directions correlated 0.992 for native versus JPEG and 0.900 for
native versus resize. It retained 7 of 158 native-geometry locked-test
positives and none of 135 controls. That apparent invariance failed on the
wide geometry challenge: the same thresholds accepted 11 of 2,227 validation
and 14 of 2,162 test Spaces controls, plus seven of 261 same-provider controls
and two of 118 exact-version controls. It is a general renderer or resampling
spectrum and is rejected.

The separately fitted period-16 three-view ensemble remains a useful research
diagnostic because it survived the broader control challenge, but it is not a
single codebook, has only eight native-geometry test positives, and derives
most of its signal from the same subperiod family. It does not rescue the
universal-detector claim or justify runtime code.

A non-learned fractal hypothesis was tested last. For each image it compared
the phase-free cube-root LMS spectrum with the same image after JPEG 95 and
after a physical 0.75x reduction. Candidate native periods 8, 12, 16, 20, and
24 were paired with scaled periods 6, 9, 12, 15, and 18. The score was the
minimum of native-to-JPEG and native-to-scaled spectral cosine, so no provider
template or fixed phase participated. Validation selected period 12-to-9
log-magnitude and retained 3 of 140 positives above all 145 controls. The
locked test accepted none of 158 OpenAI positives and one of 135 controls; the
OpenAI median was 0.515 versus 0.555 for controls. Simple phase-free spectral
self-similarity is not an OpenAI watermark feature and the fractal branch is
rejected.

The scale hypothesis was then challenged against the common non-native export
geometries. A nearest-native inverse-raster surrogate recovered none of 1,321
test OpenAI images and accepted two of 1,485 symmetrically resampled control
views. That mapping was intentionally treated as incomplete because it changed
aspect ratio for important cohorts. The largest cohort, 1086x1448, is instead
consistent with an isotropic `1/sqrt(2)` reduction from 1536x2048. Restoring
that exact hypothesized source raster still recovered none of 579 validation
or 562 test positives. A direct fractional-phase fold avoided inverse
interpolation and searched two raster-coordinate conventions plus all 64 cyclic
origins, charging the complete registration search against validation
controls. It retained 2 of 579 validation and 4 of 562 test positives while
accepting 2 of 135 test controls. Its median score was lower for OpenAI than
for controls. The fractional-carrier explanation for this dominant export
geometry is rejected.

A nonlinear cyclostationary screen next tested whether content-dependent or
payload-varying signs cancel the cross-image mean while leaving a periodic
variance. It compared folded magnitude, energy, locally normalized energy,
and cross-channel products in cube-root LMS and OKLab. Validation selected the
absolute residual of the third cube-root LMS component at period 4. It retained
33 of 140 validation and 43 of 158 locked-test positives, but also accepted 2
of 135 locked-test controls. The frozen wide challenge confirmed a renderer
confound: it accepted 48 of 1,854 validation and 52 of 1,839 test OpenAI images,
against 7 of 2,227 validation and 5 of 2,162 test Spaces controls. It also
accepted 10 of 261 action-derived same-provider controls, including 5 of 118
that explicitly name `gpt-image 2.0`. Requiring spatial agreement did not
repair specificity; the validation-selected median over four quadrants kept
41 of 158 test positives and the same two native-geometry test controls.

Finally, a second-order cyclic screen folded spatial products of the residual
with lags 1, 2, 4, and 8 along axial and diagonal directions. It evaluated 360
fixed LMS/OKLab, channel, period, lag, and normalization combinations. The
validation winner used the second cube-root LMS component, period 16, and a
vertical four-pixel lag. It retained 36 of 140 validation and 43 of 158 test
positives, but accepted 4 of 135 locked-test controls. Periodic magnitude,
patch consistency, and lag autocorrelation therefore expose reproducible
OpenAI renderer structure but do not meet the SynthID specificity gate.

These negative results bound the current manual feature search. Signed phase,
phase-free power, subperiods, integer and fractional scale, inverse raster
registration, spectral self-similarity, nonlinear periodic variance, spatial
consistency, and second-order lag products have all either lost sensitivity or
crossed held-out controls. Adding further hand-selected transforms to the same
exposed corpus would increase adaptive overfitting rather than confidence. A
defensible next OpenAI epoch requires provider-authorized oracle labels or
clean same-content counterfactual pairs and a fresh temporal holdout.

An untouched 2026-08-07 through 2026-08-10 temporal cohort supplied that
holdout for validation, but not the missing oracle negatives. It contained 486
explicit OpenAI-provenance positives and 504 heterogeneous controls. Only seven
additional files were pure OpenAI C2PA candidates without a SynthID assertion;
the remaining same-platform candidates contained inferred or mixed Google and
OpenAI watermark provenance and were not treated as clean negatives. The frozen
signed period-8 candidate accepted 5 of 486 positives and 1 of 504 controls.
The phase-free period-16 candidate accepted 2 of 40 supported positives and 1
of 50 supported controls. The nonlinear period-4 candidate accepted 14 of 486
positives and 8 of 504 controls. The manual carrier branches fail temporal
transfer.

### 2026-08-10: OpenAI content-dependent decoder challenge

The periodic result did not exhaust the neural-watermark hypothesis. OpenAI's
ChatGPT Images 2.0 system card describes its watermark as content-specific, and
the SynthID-Image paper says the encoder is content-dependent by design. The
paper also separates watermark detection from payload recovery and evaluates
the external SynthID-O model at its preferred 512x512 input size. These facts
support a learned full-field decoder, but they do not establish that OpenAI and
Google deploy identical model weights, payloads, or pixel spectra.

A second corpus audit found 261 same-provider candidate controls. Their signed
provenance contained `created` and `converted` actions without the
`c2pa.watermarked.unbound` action, but a subsequent software-agent audit showed
that they do not all come from the positive generation epoch. Only 118 name
`gpt-image 2.0`: 97 training, 16 validation, and five development-test images.
Another 108 name an earlier version and 35 do not expose a usable version. The
full 261-image split remains 217/26/18. These are useful confound controls, but
even the exact-version subset is action-derived rather than a provider-oracle
negative label.

Two pixel-only native-resolution experts were trained with the mixed-version
control set. A
local residual CNN accepted 132 of 1,280 validation positives and 149 of 1,364
development-test positives, with zero same-provider or COCO controls in both
splits and two of 1,373 difficult development-test controls. A separate
spectral, autocorrelation, bit-plane, and residual-error expert accepted 175
validation and 167 development-test positives with no accepted controls. A
validation-selected linear combination accepted 243 positives in each split,
with no accepted validation or development-test control. The zero-error
development-test bound across 3,391 controls is 0.0883% at one-sided 95%
confidence, but the same-provider subset contains only 18 images and therefore
does not independently establish provider specificity.

The exact-version audit did not invalidate the native score counts, but it
weakened their attribution. None of the 16 validation or five development-test
`gpt-image 2.0` controls crossed the residual, forensic, ensemble, canonical,
or JPEG-equalized thresholds. That is directionally useful, but five temporal
test controls cannot establish deployment specificity. Perceptual-hash and
local CLIP searches compared all 118 exact-version controls with 7,259
positives. They recovered a few near-content matches but no clean before/after
watermark pair. Sub-pixel affine alignment left content, edge, and photometric
changes in the residual, while 8x8, 16x16, and 32x32 folded residual
correlations varied in sign and magnitude. Those candidates cannot be used as
causal watermark differences.

Transformation challenges rejected the combined native expert as a SynthID
decoder. Of its 243 accepted development-test positives, 121 remained accepted
after a 5% crop round trip, six after a 75% resize round trip, and none after a
JPEG-95 round trip. No transformed same-provider control was accepted. The
codec fragility is inconsistent with treating the score as the documented
robust watermark without further evidence.

The 512x512 branch also failed the held-out specificity gate. Its selected
model accepted 238 validation and 194 development-test positives, but also
accepted two of 18 same-provider, four of 1,373 difficult, and nine of 2,000
COCO development-test controls. Finally, a CNN trained and evaluated only
after applying the same JPEG-95 round trip to every positive and control
accepted eight validation positives and six development-test positives while
accepting five of 2,000 development-test COCO controls. Equalizing the codec
therefore removed nearly all of the apparent signal instead of revealing a
robust hidden decoder feature.

An independent three-CNN OpenAI surrogate was also evaluated from its
published weights, without adopting its code or claims. At the published 0.5
threshold its ensemble marked 82.0% of COCO validation controls and 83.4% of
COCO development-test controls as watermarked. It also marked 56.3% of
difficult development-test controls and 15 of 16 exact-version OpenAI
validation controls, compared with 87.3% of OpenAI validation positives. The
repository does not publish its training corpus or logs, and its implementation
applies supposedly shared pair augmentation independently. This result rejects
the model as a SynthID decoder and demonstrates why same-content-looking
regenerations are not sufficient negative labels.

A final transform-augmented full-field branch tested the closest practical
approximation to the documented neural decoder. A 512x512 CNN consumed both
RGB and a fixed high-pass residual and trained with random crop/resize,
JPEG 70-100, horizontal flip, and mild color transforms. Training sampling
balanced watermarked positives, difficult controls, and the exact-version
OpenAI candidate controls. Selection required a score above every control for
both native and JPEG-95 validation views. All four fixed training epochs
accepted zero of 500 selection positives. On the full validation set the
frozen model accepted zero of 1,280 OpenAI positives in native, JPEG-95,
75%-resize, and 5%-crop views, while accepting up to two unrelated controls.
On development test it accepted zero of 1,364 OpenAI positives in native,
JPEG-95, and resize views. The crop view accepted one OpenAI positive and one
of 1,373 difficult controls. Transform augmentation therefore did not recover
a usable weaker OpenAI signal.

The measured OpenAI-native separation is best classified as an export or
rendering noiseprint, not a validated SynthID detector. None of these models is
added to runtime. Transform-augmented training has now also failed. A
defensible next epoch needs provider-authorized labels or clean same-content
before/after watermark pairs, plus a fresh temporal holdout that has not
influenced representation or threshold selection.

The same untouched August cohort then challenged the frozen 512-square forensic
model and its unchanged threshold. It accepted 82 of 486 explicit OpenAI
positives, one of 504 heterogeneous controls, and one of seven pure OpenAI C2PA
candidate controls. The aggregate score separation was real, but transformation
testing resolved its interpretation: all 82 native detections disappeared after
both JPEG-95 and WebP-95 round trips. A 0.75x resize retained 46, and a 5% crop
retained 23. The model therefore generalizes as a fragile OpenAI export
noiseprint, not as the documented robust SynthID signal.

The provider-key alternative was then tested directly. OpenAI training folds
were clustered into 1, 2, 4, 8, 16, or 32 spherical period-8 cube-root-LMS
directions, and the cluster count was selected on validation only. Eight
directions raised the development-test result from 16 to 27 of 158, with one
of 135 heterogeneous controls and two of 52 action-derived same-provider
controls accepted. This did not reveal eight independent codewords: the two
dominant clusters contained 234 of 283 training rows, and their templates
correlated at 0.982 natively and 0.924 after JPEG-95.

The frozen multi-direction model accepted none of 443 Google native-geometry
files, but accepted 5 of 39 Microsoft Designer files and 123 of 674 OpenAI-
platform files. All five Microsoft hits had joint `Microsoft, OpenAI`
provenance, one explicitly asserted an OpenAI SynthID watermark, and all five
selected the same dominant direction as 111 OpenAI hits. On the
fresh August native-geometry cohort, SHA-256 grouping reduced seven OpenAI row
hits to five unique content groups; two unique Microsoft controls crossed the
same dominant direction. A sign-invariant carrier-subspace fit retained only 1
of 158 development-test positives and none of 40 fresh positives. A final
OpenAI direction orthogonal to Google, Microsoft, and other-provider train
means likewise retained 1 of 158 and none of 40. The measured period-8 family
is distinct from Google but is shared with a Microsoft/OpenAI lineage.

Regrouping by the signed watermark assertion found 115 of 589 explicit OpenAI-
SynthID files accepted, against 13 of 117 OpenAI-lineage files without an
assertion, none of 443 Google files, and one of 89 other-provider files. The
aggregate one-sided exact-test result was `p = 0.0179`; the already exposed
development-test split was not independently significant at 27 of 159 versus
2 of 25 (`p = 0.202`). Missing watermark action is not an oracle-negative
label, so this is compatible with a provider-keyed family but does not prove
that the measured component is the watermark.

The frozen score did exhibit real codec persistence. WebP-95 retained 24 of 27
native development-test detections and added nine other positives, with the
same one of 135 controls retained. JPEG-90 retained eight of 27 and added no
positive, with zero controls. A 0.75x resize and 5% crop retained none. The
result is stronger than the full-image noiseprint and supports a genuine
OpenAI/Microsoft raster-phase component, but its geometric fragility excludes
it as the documented complete robust SynthID decoder surface.

The broader SynthID architecture permits provider-specific codes without
requiring one shared pixel template. The SynthID-Image paper separates binary
detection from payload recovery, identifies payloads as a mechanism for
distinguishing customers of one service, and assigns watermark versions to the
encoder. OpenAI and Google can therefore use the same watermark family while
deploying different encoder versions or payload distributions.

The narrower same-carrier/different-phase prediction was challenged directly.
Period-8 cube-root LMS folded residuals were collected from 581 byte-unique
OpenAI-asserted files, 443 Google-asserted files, 93 OpenAI-lineage files with
no watermark assertion, and 82 other-provider controls. Both provider
contrasts were internally stable across deterministic halves. Raw phase-free
power means had cross-provider cosine 0.480, but this shared imaging background
disappeared after subtracting the respective controls: positive power support
had cosine 0.0123, bootstrap median 0.0232 with a 95% interval of 0.0009 to
0.0856, and no overlap among the 12 strongest coordinates. The three
per-channel cosines were all at most 0.0155. The signed contrasts were negative,
not phase-locked. The evidence therefore rejects a simple Google template with
an OpenAI phase or payload substitution. It remains compatible with a distinct,
content-dependent SynthID encoder version and motivates separate learned
experts rather than further rotation of the Google carrier.

A fixed three-query OpenAI Verify validation then established attribution while
also falsifying completeness. An asserted OpenAI image accepted by the frozen
eight-direction period-8 candidate was submitted both in its original form and
as a pixel-identical PNG re-encoded without metadata. Both returned `SynthID
detected` and `Content Credentials not detected`. A second asserted OpenAI
image rejected by the local candidate, with native/JPEG scores 0.709/0.657
against thresholds 0.918/0.923, was also re-encoded without metadata; it
returned the same pixel-only SynthID verdict. The period-8 family is therefore
a genuine OpenAI SynthID component with known oracle-positive false negatives,
not merely an export fingerprint and not a complete detector. No adaptive
oracle sweep followed: OpenAI's Content Provenance documentation explicitly
disallows repeated queries to reverse-engineer, remove, or evade a watermark.
The next local epoch must treat period-8 as one expert and discover complementary
content-dependent experts using frozen corpus splits, without oracle-guided
feature selection.

Relaxing the eight-direction model to a separate validation-control threshold
for every direction increased validation recall to 36 of 140 with zero of 145
controls, but it did not preserve specificity. The locked development test
accepted 46 of 158 OpenAI rows and 4 of 135 controls. A wider provider challenge
accepted 180 of 603 explicit OpenAI-SynthID rows, 15 of 80 OpenAI-lineage rows
without an assertion, 6 of 443 Google rows, and 7 of 112 other-provider rows.
The fresh temporal result was 13 of 40 OpenAI rows, 4 of 50 general controls,
and 1 of 3 same-provider controls. The oracle-positive local false negative did
cross direction 4 under this relaxed calibration, but that direction survived
only 1 of 4 leave-one-positive-out refits; the oracle-positive row itself then
failed its JPEG threshold. Direction-specific thresholds therefore expose
unstable minority clusters rather than additional validated codewords.

A train-only bilinear follow-up tested whether a content-dependent or keyed
phase varies between images while retaining native-to-JPEG covariance within
each image. It subtracted the control within-class cross-covariance from the
OpenAI cross-covariance, selected the positive-eigenvalue rank on validation,
and kept the threshold above every validation control. Rank 1 retained only 1
of 158 locked-test OpenAI rows and zero of 135 controls, then zero of 40 fresh
OpenAI rows and zero of 53 fresh controls. A single codec-stable variable-phase
subspace is therefore rejected as the missing complementary expert.

Taking local spectral magnitude before spatial averaging produced a superficially
stronger complementary signal. Eighteen period, block-normalization, and spectral
variants were screened on validation; period-8 unit-block log-magnitude retained
8 of 158 locked-test OpenAI rows and zero of 135 controls, then 3 of 40 fresh
OpenAI rows and zero of 53 fresh controls. None of the three fresh hits overlapped
the frozen global eight-direction hits, and only one overlapped the unsafe
per-direction union. Both oracle-positive images, including the known local
false negative, remained below this expert's thresholds.

The full provider challenge rejected watermark attribution despite that clean
development transfer. The local-spectrum expert accepted 17 of 603 explicit
OpenAI-SynthID rows, 5 of 80 OpenAI-lineage rows without an assertion, zero of
443 Google rows, and 2 of 112 other-provider rows; both other-provider hits were
Microsoft. Explicit watermark assertion was not enriched over same-lineage
non-assertion (`p = 0.966`, one-sided exact test in the required direction).
Recalibrating both frozen view thresholds above the 32 same-lineage validation
maxima reduced locked-test recall from 8 of 158 to zero. Local pre-aggregation
magnitude therefore reveals another OpenAI/Microsoft raster component, not a
defensible complementary SynthID expert.

The proposed directional-wavelet branch was also tested directly on OpenAI.
A three-level DTCWT over the period-8 cube-root-LMS residual screened all six
orientations, the diagonal and axis-near groups, individual levels, and complex,
magnitude, and phase statistics. Validation selected all-orientation normalized
complex correlation at 19 of 140 OpenAI rows and zero of 145 controls. It
retained 16 of 158 locked-test OpenAI rows and zero of 135 controls, followed
by 2 of 40 fresh OpenAI rows and zero of 53 fresh controls.

This is a stronger representation of the known phase carrier, not a new expert.
Both fresh hits were already accepted by the frozen eight-direction model, the
known oracle-positive local false negative remained negative, and the other
oracle-positive row missed only the DTCWT JPEG threshold at 0.894 versus 0.897.
The provider challenge accepted 77 of 603 explicit OpenAI-SynthID rows, 3 of 80
OpenAI-lineage non-assertions, zero of 443 Google rows, and 2 of 112 other rows,
both Microsoft. The explicit-assertion enrichment was real (`p = 0.00879`), but
72 of the 77 hits overlapped the signed phase expert and their union increased
its explicit count only from 115 to 120. Same-lineage calibration retained 4
of 158 locked-test and 1 of 40 fresh OpenAI rows, while one Microsoft test row
still crossed.

The transform challenge then falsified operational shift invariance. Cyclic
rolls by `(1, 1)` and `(3, 5)`, a 0.75x resize round trip, and a 5% crop round
trip each retained zero of the 16 baseline detections. JPEG-90 retained four.
WebP-95 retained 15, accepted 25 OpenAI rows in total, and introduced one of
135 controls. DTCWT therefore improves codec tolerance for the already measured
raster phase but does not solve registration, scale, arbitrary resolution, or
the missing OpenAI payload variants.

Selecting the best DTCWT magnitude and phase candidates within their own
families did not change that conclusion. The off-diagonal magnitude candidate
retained 10 of 158 locked-test OpenAI rows and zero of 135 controls, then 3 of
40 fresh OpenAI rows and zero of 53 controls. All three fresh hits were already
signed-phase hits. The near-vertical phase candidate retained 9 of 158 and zero
of 135, but failed temporally at 1 of 40 OpenAI rows versus 3 of 50 general
controls. Magnitude also retained zero of its ten baseline detections after a
one-pixel roll, `(3, 5)` roll, resize, or crop; JPEG-90 retained one and WebP-95
retained two. Taking magnitude only after periodic complex folding preserves
the phase-origin dependency and does not realize DTCWT's intended shift
invariance.

A final DTCWT screen removed periodic folding entirely and summarized global
energy over six orientations and three levels. Nine validation-screened
translation-insensitive magnitude, power, orientation, and robust-quantile
statistics selected median magnitude proportions within each level. The locked
test accepted only 5 of 158 OpenAI rows and 1 of 135 controls; the fresh holdout
accepted zero of 40 OpenAI rows and zero of 53 controls. DTCWT therefore presents
a measured tradeoff in this corpus: preserving the periodic complex field
retains the real OpenAI carrier but its phase origin, while pooling enough to
be translation insensitive removes the transferable separation.

A metadata-stratified audit then tested whether the 77 provider-challenge hits
were confined to one encoder rollout or geometry. Within the exact `OpenAI
Media Service API` cohort, the frozen DTCWT expert accepted 75 of 581 asserted
rows and 1 of 52 same-generator rows without an assertion. Asserted-row rates
were 3/16 in May, 46/365 in June, and 26/200 in July, and 6/47, 44/320, and
25/214 across the three native geometries. The native/JPEG minimum score had
0.721 AUC against the same-generator non-assertion stratum. The component is
therefore continuous across time and geometry rather than a short-lived export
cohort, although missing assertion remains an indeterminate label rather than
a clean watermark negative.

A prespecified shift-invariant follow-up sorted all 64 cyclic period-8
template correlations within each image. This preserves the shape of the
correlation orbit while discarding its absolute phase; one train-only contrast
was fitted for native and JPEG-95 views, with both thresholds placed above all
145 validation controls. The individual views retained 5 and 7 of 140
validation positives, but their conjunction retained none of 158 locked-test
positives and none of 135 controls. JPEG, WebP, resize, and crop could not
rescue a baseline detection because there was none. Thus the transferable
OpenAI evidence is carried by signed raster phase, not merely by a
shift-invariant matched-filter peak shape.

Content-hash grouping also corrected the temporal denominators. The 486 OpenAI
rows represented 417 unique byte groups, and the 234 Google rows represented
211. The signed OpenAI period-8 result became 4 of 417 unique positives versus
1 of 454 general-control groups; the full-image noiseprint became 68 of 417;
the Google runtime result became 153 of 211. Repeated uploads did not change the
decision, but row-level counts must not be treated as independent evidence.

### 2026-08-13: OpenAI low-texture phase and masking-alignment challenge

The dominant 1086x1448 export geometry supplied 2,265 byte-unique images with
an explicit OpenAI SynthID assertion, 830 general controls at the same geometry,
and 26 OpenAI-lineage files without a watermark assertion. The exact-geometry
split contained 1,124/579/562 asserted OpenAI train/validation/test rows and
399/203/228 general controls. The lineage rows remained an indeterminate
stratum rather than being relabeled as negatives.

A train-only cross-image phase model selected 256 RGB bins from the 20
lowest-texture references in each of three luminance groups. It used
polarity-invariant axial phase and scored both native pixels and a JPEG-95 view
after canonicalizing to 512 square. Independent train halves shared only two
of 256 exact channel-frequency bins and three spatial coordinates. The full
model produced native AUC 0.526 on validation and 0.512 on test; JPEG-95 AUCs
were 0.521 and 0.512. A threshold above every validation control accepted none
of 562 test positives, while a validation 1% FPR threshold accepted four
positives and two of 228 test controls. Texture quartiles did not reproduce a
useful exception. The common absolute-phase hypothesis is rejected for this
OpenAI geometry.

The official architecture points to a different mechanism: watermark
generation is deliberately content-dependent to resist watermark exchange,
and nearly uniform content can be filtered rather than watermarked. A second
frozen probe therefore discarded absolute phase and global residual amplitude.
It measured 60 OKLab relations between blockwise high-frequency energy and
local texture, contrast, and chroma at two filter scales and two block sizes.
One train-only shrinkage-Fisher direction was scored in native and JPEG-95
views, and the joint decision used their minimum above every validation
control. Joint AUC was 0.563 on validation and 0.593 on test. The frozen test
count was 15 of 562 asserted OpenAI rows and one of 228 controls.

That weak enrichment was not a stable expert. Directions fitted on deterministic
train halves had raw-coordinate cosine 0.148. One half accepted 13 test
positives and two controls, the other accepted none; their test score
correlations were 0.335 on positives and 0.127 on controls, with no overlapping
positive decision. The branch is rejected as an unstable content or export
correlate. Together these experiments reinforce the existing boundary: a new
OpenAI epoch needs pixel-oracle labels or clean same-content counterfactuals,
not further adaptive feature selection on provenance-labeled exports.

### 2026-08-13: multi-scheme domain generalization and Gemini bicoherence

A deliberately broader experiment tested whether SynthID could inherit a
generic post-hoc watermark signature instead of a provider carrier. Ninety
source-matched COCO pairs were prepared for each of TrustMark P, Meta VideoSeal
Y-256, classical 64-bit DWT-DCT, and Meta WAM-MIT, always with random payloads.
The official VideoSeal decoder recovered a mean 99.6% of bits across identity,
JPEG-90, 0.8x resize round trip, and 5% crop round trip; its mean pair PSNR was
45.04 dB. WAM recovered 99.5-99.6% across the same views at mean PSNR 38.44 dB.
DWT-DCT supplied an architecture-diverse control, but only identity and resize
were retained as positives because its official decode collapsed to chance
after JPEG and crop.

A sign-symmetric 3,900-feature spatial rich model first learned residual
co-occurrences in luminance, opponent-color, and Lab channels. Its linear and
quadratic paired-delta variants recognized familiar schemes but did not transfer
to a held-out scheme. A stronger neural follow-up used a fixed residual bank,
worst-family training, cross-family residual transplantation, random sign,
phase, and amplitude, plus phase-scrambled equal-power residuals as hard
negatives. Each of four runs withheld one complete encoder family and all test
source images. Held-out aggregate AUCs were 0.562 for TrustMark, 0.474 for
VideoSeal, 0.497 for DWT-DCT, and 0.437 for WAM. The corresponding strict
true-positive rates were 5.0%, 3.3%, 0%, and 0%. This rejects the available
open-proxy path to a universal single-image detector: even causal paired data
from three distinct schemes did not generalize to a fourth.

A separate Gemini experiment tested a translation-invariant higher-order
signature. Block bicoherence measured normalized phase coupling at period-16
triplets satisfying `f1 + f2 = f3`; unlike ordinary phase correlation, the
global image-shift phase cancels from this product. Six Green, cube-root LMS,
and OKLab representations, two residual scales, two block sizes, and three
train-only harmonic counts produced 72 candidates. The data were separated
into harmonic-fit, candidate-selection, threshold-calibration, and final
splits: 30/20/50 Gemini positives and 100/100/100/199 Open Images controls.
The best selection candidate accepted none of 20 positives at the selection
control maximum and had AUC 0.523. After independent recalibration, the final
result was 0 of 50 Gemini positives and 0 of 199 controls, with AUC 0.374.
Period-16 bicoherence is therefore rejected as a scale or origin rescue. The
measured Gemini carrier remains a linear, registered periodic tile rather than
a transferable harmonic cascade.

### 2026-08-14: remaining Gemini resolution challenges

The 0.5x gap was tested with a local-consensus detector rather than another
global period-8 score. Each half-scaled image was divided into 64-pixel patches;
the 90th percentile of full-template patch correlations was selected without
using the locked test. It accepted 33 of 49 validation positives and zero of 166
validation controls at the frozen 0.552176 threshold, then 27 of 52 locked-test
positives and zero of 140 controls. This looked like a useful low-resolution
expert but did not survive scale-matched specificity testing.

The frozen challenge symmetrically resized every remaining Spaces control and a
separate 1,000-image Open Images reserve to 0.5x. The expert accepted 61 of
10,906 Spaces controls and 22 of 1,000 Open Images controls. All 83 errors were
rescored with the prespecified feature inventory. A negative-Green patch-spread
gate could remove them post hoc, but retained only 12 of 106 base-positive
development hits. On the later August 7-10 cohort the original patch expert
accepted 67 of 234 Google-provenance rows and 8 of 689 controls, while the new
conjunction retained only 2 Google rows and zero controls. The broad challenge
rejects the base expert and the temporal result rejects the post-hoc repair.
No 0.5x claim is added.

The large-image window family was completed separately. One centered,
phase-aligned 2,048-square registered window accepted 1 of 37 candidates above
10 megapixels and zero of 89 controls. Scoring every phase-aligned 2,048-square
window and accepting the maximum raised recall to only 2 of 37, still with zero
controls; the maximum control decision score was already 0.9677 against the 1.0
runtime boundary. Together with the previously rejected fixed-window consensus
and full-frame resize, these results leave the 10-megapixel ceiling intact.

Near-ceiling windows, color spaces, and JPEG-paired phase directions did not
provide a missing large-image expert. Three aligned windows of approximately
10 megapixels retained 1 of 37 candidates and zero of 89 controls. A native
fixed-score plus Green mid-band rule retained 15 of 19 candidates in four exact
recurring geometries and accepted one of 11,417 resized Spaces controls, but
collapsed to 0 of 19 after JPEG 90. A codebook fitted to the minimum native/JPEG
period-16 correlation selected RGB Green mid-band on the pre-July split. It
retained 23 of 26 candidates and zero of 40 controls there, then 10 of 11 and 7
of 49 on July. Lab and OKLab candidates failed on the same hard control. The
apparent native signal is a codec or upscaler lattice rather than a robust
watermark invariant.

Those 37 candidate labels came from Google C2PA and an inference that Google
media is likely watermarked, not an individual Google pixel-decoder result.
They cannot support a production recall claim. The next large-image step is an
immutable SHA-to-verdict Gemini batch, followed by fitting only on
oracle-detected files. The first live attempt on 2026-08-14 was stopped by
Google reCAPTCHA before the first new file submission; no CAPTCHA bypass was
attempted.

### 2026-08-14: official large oracle and opponent-color runtime branch

The frozen Gemini batch was subsequently submitted without bypassing CAPTCHA.
Every submitted file had AI provenance removed while preserving the decoded
RGBA fingerprint. Google returned a pixel-positive verdict for seven of seven
usable submissions across 3072x5504, 5504x3072, 5632x3072, 6336x2688,
3584x4800, 5016x3328, and 4096x4096. An eighth case was indeterminate after the
verifier quota ended. This is direct evidence for seven positives, not a label
upgrade for the remaining 30 C2PA-inferred candidates.

Revisiting the native period-16 fold by opponent color exposed the missing
separation. Google candidates preserve strong Red-minus-Green and
Blue-minus-Yellow phase agreement across every phase-aligned 2048-square
window, while natural resize aliases do not preserve the complete conjunction.
The frozen large-v1 rule requires minimum fixed score `0.14`, minimum
Red-minus-Green spatial correlation `0.90`, minimum Blue-minus-Yellow spatial
correlation `0.70`, and a minimum signed Blue-minus-Yellow mid-band correlation
at or below `-0.15`. The 3072x5504 portrait geometry additionally requires the
maximum Green mid-band correlation to stay at or below `0.06`. Its public score
is the minimum normalized gate margin and crosses at `1.0`.

The rule retained all 37 inferred candidates, including all seven official
pixel positives. It accepted none of 17,417 exposed COCO, Open Images, and
Spaces controls after the portrait alias gate. Because those controls informed
the final gate, they do not certify the operating point. The constants were
then challenged through the actual runtime on a frozen, decoded-pixel-deduped
local holdout: 2,000 COCO val2017 images excluded from the earlier 3,000-image
large color-phase challenge plus 637 unique Picsum controls deduplicated from
1,000 files. Four large geometries and four resampling kernels were balanced.
The result was 0 of 2,637 detections, with maximum score `0.0592777965`; the 95%
zero-error upper bound is about 0.114%. These source collections existed before
the freeze and supported other experiments, so the result is feature-unseen
rather than a fresh-acquisition holdout.

The post-freeze Open Images acquisition was too slow to complete the planned
3,000-image challenge, but its completed subset provides an independent source
audit. Incomplete `.aria2` files, all prior Open Images decoded-pixel hashes,
and duplicates were excluded before scoring, leaving 41 controls whose IDs and
URLs had been frozen after the constants. The unchanged runtime accepted 0/41;
the maximum score was `0.4083013324`. This small audit supports the source
boundary but does not replace the 2,637-control feature-unseen interval.

A prespecified post-freeze codec stress test re-encoded the seven official
positives at unchanged dimensions. Native detection was 7/7, while JPEG-95 and
JPEG-90 were each 0/7. Thresholds were not refitted. Large-v1 is therefore a
native-pixel and lossless-copy detector, not a lossy-transcode detector.

The old fixed-only rule still crosses on four locally saved rows representing
three unique pixel groups. One group is an Adobe file whose C2PA explicitly
reports Google SynthID present. The other two remain queued for an official
Gemini verdict and are not counted as clean negatives. The runtime branch is
therefore positive-only and does not turn a miss into an absence claim.

Production routing now selects the fixed expert from 1 through 10 megapixels
and the large expert above 10 through 18 megapixels when both sides are at least
2048 pixels. The score exporter emits fixed, registered, and large observations
separately, preventing the large result from being mislabeled as a fixed score.
The router accepts a registered or large threshold crossing only within that
expert's measured scope and otherwise abstains.

### 2026-08-14: final OpenAI local rescue checks

An assertion-contrast probe at the dominant exact 1254-square geometry tested
RGB, OKLab, and cube-root-LMS period-8 channel directions without treating
missing assertion as a clean label. Validation selected `srgb:p8:c2`, accepting
45 of 259 asserted rows and zero of 13 same-provider plus three other-provider
rows. The locked test accepted 39 of 233 asserted rows, zero of 15 same-provider
rows, and one of three other-provider rows. The all-row audit accepted 185 of
992 asserted rows, one of 55 same-provider rows without an assertion, and one
of 13 other-provider rows. A safe boundary above either wider control maximum
retained only a small minority of the asserted cohort. Exact geometry therefore
does not turn the weak OpenAI component into a production detector.

An independent published-decoder veto was then tested with the official
InvisMark checkpoint. The downloaded checkpoint hash was fixed before use, and
its bundled self-test reached 0.997 mean confidence with 0.97 recovered-bit
accuracy. In contrast, OpenAI DTCWT hits, OpenAI misses, exact-1254 hits and
misses, same-provider controls, Google, Microsoft, Canva, and unrelated controls
all had median confidence near 0.18. The decoder is working, but its watermark
family is absent from these cohorts; it cannot distinguish SynthID-bearing
OpenAI pixels from renderer confounds.

Two representation-level missing-codeword hypotheses were also rejected.
DTCWT modulus computed before spatial period folding selected off-diagonal
phase coherence and retained 5 of 158 locked-test OpenAI rows versus zero of 135
controls, but zero of 40 fresh OpenAI rows. Whitening the period-8 feature space
and clustering up to 16 codewords retained 5 of 158 and zero controls, then one
of 40 fresh OpenAI rows and one of 50 fresh controls. Neither captured the known
oracle-positive false negative. These were the remaining prespecified phase,
wavelet, codeword, and external-decoder variants; none earns a local runtime
route.

### 2026-08-15: OpenAI high-pass patch-bag challenge

One prepared but previously unexecuted content-dependent hypothesis aggregated
the distribution of nine normalized high-pass patch embeddings per image. Its
input contains luminance, Red-minus-Green, and Blue-minus-Yellow residuals but
no filenames, metadata, dimensions, or RGB thumbnails. Mean, standard
deviation, minimum, and maximum pooling allow spatially varying evidence without
requiring one global phase codeword.

The validation-selected epoch accepted 285 of 1,280 OpenAI rows above every
validation control. With weights and threshold frozen, the already exposed
development test accepted 324 of 1,364 OpenAI rows, 3 of 1,373 difficult
controls, and 1 of 2,000 COCO controls. The later August temporal cohort was
not used by this model or threshold: native scoring accepted 120 of 486 OpenAI
rows, 8 of 504 general controls, and 5 of 33 same-provider rows without a
watermark assertion. Assertion enrichment over the same-provider stratum was
not independently significant (`p = 0.151`, one-sided exact test). After
decoded-pixel grouping the counts were 107/417, 4/453, and 4/30. Requiring the
same unchanged score after JPEG-95 retained only
1/486 OpenAI rows and no controls.

None of the 120 native OpenAI hits overlapped the frozen signed period-8 hits;
four overlapped the nonlinear period-4 candidate. This supplies new evidence
for multiple complementary OpenAI raster components. It does not supply a
production watermark decoder: same-provider non-assertion rows crossed at a
high rate, general-control specificity failed, and ordinary JPEG destroyed all
but one positive. A post-hoc threshold above the temporal control maximum would
retain 19 unique OpenAI rows, but that threshold has observed the holdout and
is not a new expert. The official OpenAI verifier remains the provider-wide
pixel route.

### 2026-08-14: production OpenAI verifier boundary

OpenAI's official Content Provenance API now supplies the production-grade
OpenAI pixel verdict that the local experiments could not justify. The runtime
integration is deliberately a separate `verify-openai-synthid` command, never
an implicit `identify` call. It requires the independent `verify` extra,
`OPENAI_API_KEY`, endpoint access, and `--acknowledge-upload`.

The implementation establishes metadata independence before the request. It
computes a decoded RGBA fingerprint, strips AI provenance metadata to a
temporary copy, verifies that no AI markers survived and that the format and
pixel fingerprint stayed identical, enforces the documented 50 MiB limit, and
uploads only the temporary PNG, JPEG, or WebP. It parses exactly one `synthid`
entry and deliberately ignores the independent C2PA outcome. C2PA-only
positives, missing or duplicate SynthID fields, altered pixels, surviving
metadata, unsupported formats, and 400/404/429 failures are covered by mocked
tests. No credentialed API request was made during implementation because no
API key was available.

A separate live smoke used OpenAI's public web verifier after running the exact
production metadata-stripping and decoded-RGBA equality checks. Two OpenAI
images at opposite 3:2 orientations returned `SynthID detected` and `Content
Credentials not detected`. A Google SynthID oracle-positive control and a COCO
natural control returned OpenAI `SynthID not detected`, again with Content
Credentials absent. This confirms pixel-only, provider-specific semantics for
the integration, but four fixed cases do not estimate statistical error rates
and do not constitute a live SDK/API transport test.

On 2026-08-15, one separately preselected OpenAI-positive PNG completed the
credentialed SDK path after the same metadata removal and decoded-pixel equality
checks. The API returned an independent SynthID `detected` result with nullable
model and generation-time fields, and the command completed without a retry.
This closes the first positive PNG transport smoke only. It does not qualify
JPEG, WebP, negative cases, operational rate limits, retention behavior, or
local-detector accuracy.

This closes the production OpenAI detection surface through an official remote
backend, not by relabeling the incomplete local period-8 expert. It does not
expand the research oracle authority. The endpoint is not eligible for Zero
Data Retention, and its documentation prohibits repeated reverse-engineering or
evasion queries, so it cannot supply adaptive labels for detector or remover
optimization without separate authorization.

### 2026-08-15: periodic codeword replacement and its measurement limit

Suppressing the estimated periodic residual with independent noise spends
quality without reaching the decoder: at a strong budget the OKLab `a/b`
branch moved the Gemini score by a median of `-0.022` at `32.6 dB`, because
independent per-pixel noise averages out under the fold. Replacing the
estimated period-16 residual with a magnitude-matched random-phase codeword
instead moved all four Gemini positives below the frozen local threshold, with
a median score of `0.050` against `0.533` at `53.5 dB` and SSIM `0.9982`.
Adding non-periodic noise on top only cost `7 dB` and changed nothing.

Both providers show a threshold, not a slope. Gemini keeps all four positives
at replacement strength `0.75` and loses all four at `1.0`; the OpenAI period-8
proxy keeps `9/16` at `0.25`, `1/16` at `0.50`, and none at `0.75`. Random
injection without the subtraction left `1/16`, so the mechanism is
`subtract + randomize`, not added energy.

Radial banding of the folded residual put the Gemini carrier in the middle
band: `0.125-0.40 cycles/pixel` gives a median `-0.209` at `56.5 dB`, against
`-0.009` above `0.40`. The OpenAI proxy weights the upper band instead.

The follow-up question, whether the signal lives in the comb lines or in the
broadband content between them, cannot be answered by the local score at all.
`fold_residual_template` averages every tile, which is a comb filter: only
frequencies that are integer multiples of `1/16` cycles per pixel survive, and
everything between the lines is annihilated before the template is applied.
`probe_comb_vs_broadband.py` measures the consequence rather than asserting it.
Inside the same `0.125-0.40` annulus and at an identical residual RMS,
randomizing only the comb bins moved the four Gemini scores by `-0.177` to
`-0.519`, while randomizing only the bins between them moved them by `-0.0007`
to `-0.0012`. A local null on the broadband branch is therefore guaranteed by
the readout and is not evidence about the watermark.

That makes the two branches an oracle question. A frozen batch of four such
triples is recorded in
`.local-eval/synthid/prc-oklab-attack-2026-08-15/comb-vs-broadband-oracle-manifest.json`.
Its first attempt established a control rule that the earlier runs had missed:
the tracked originals carry a C2PA `caBX` chunk, and Google answered from it
without ever reaching the pixel path, so an original is not a valid control for
pixel-only candidates. Every arm is now re-encoded from decoded RGB with the
chunk dropped and decoded pixels asserted identical. The re-encoded source then
returned a pixel-path positive, which qualifies the control. The broadband
candidate returned a temporary service limit with a two-hour retry, so no
candidate verdict exists yet; an unavailable answer is not a negative.

### 2026-08-15: the local score is not the decoder margin

One group was then completed inside a single account, so that account tier and
request routing stay fixed across its arms; spreading arms of one comparison
across accounts would make a negative indistinguishable from a different
backend. All four arms returned a Google positive:

| Arm | Local score | PSNR | Google |
| --- | --- | --- | --- |
| source | `0.416` | - | detected |
| comb, equal RMS | `0.239` | `53.97 dB` | detected |
| broadband, equal RMS | `0.415` | `54.27 dB` | detected |
| full period-16 replacement at `0.95` | `0.087` | `52.56 dB` | detected |

The last row is the result that matters. That candidate sits well below the
frozen local threshold of `0.174`, which is the operating point the removal gate
would have used, and Google still reports the watermark. A local score at or
near zero therefore certifies nothing about the provider decoder, and no removal
claim may rest on it. The equal-RMS pair adds that neither a comb-targeted nor a
broadband-targeted perturbation at roughly `54 dB` moves the verdict, so this
run does not locate the signal either; it bounds what the local readout can say.

A replication attempt on a second image from a third account never produced a
verdict: that account hit the service limit on its first request, while another
had just served four. The limit therefore does not partition cleanly per
account, and holding several accounts does not buy proportionally more oracle
capacity. Plan oracle work around one window at a time and spend it on the arms
that decide something.

Querying stopped here rather than escalating the perturbation budget until the
verdict flipped. Searching for the breaking point by repeated queries is exactly
the adaptive use of a provider oracle that this plan rules out, and the sham
control was not spent because it only interprets a negative, and no arm returned
one.

### 2026-08-15: the same answer from the OpenAI verifier

The equivalent frozen batch went through the official OpenAI endpoint on
`verify-openai-synthid`, two OpenAI positives in a source and `replace-chroma`
pair each, four calls, one pass. Every arm returned `detected` with metadata
stripped, decoded pixels preserved, and the C2PA outcome ignored. The candidates
sit at `54.13 dB` and `53.97 dB` with residual RMS near `0.50`, the same
perturbation class that empties the local proxy.

Two providers, two independent verifiers, one conclusion: a periodic-codeword
replacement that drives the local score to zero does not remove what either
decoder reads. The removal program has no evidence of pixel-only removal for
either provider, and the local score cannot serve as its gate.

A third OpenAI positive was excluded before any call. Every periodic variant on
it produced a residual RMS of exactly zero and a periodic energy ratio of
exactly `1.0`, so the attack changed nothing: it can only redistribute periodic
energy it estimated, and it estimated none. That file is nevertheless recorded
in the manifest as `openai-verify: SynthID detected`. An estimator that finds no
periodic residual in an image the provider verifier flags is direct evidence
that the period-16 and period-8 grids do not carry the whole signal, and a
residual RMS of exactly zero is the signature to check for before reading any
attack result as a null.

This batch was a single verification pass over frozen files. It is not an
adaptive label source, and the endpoint's prohibition on repeated
reverse-engineering or evasion queries stands: no candidate was adjusted and
re-submitted, and no budget was escalated against the API.

A wider batch followed the same rule, sixteen pairs frozen in
`openai-oracle-batch-v2-manifest.json` before the first call. Their perturbation
budgets are not chosen, they fall out of each image's own estimated residual,
and they span `37.9` to `57.5 dB`. The API rate limit stopped the run after
fifteen calls with a one-hour retry, leaving seven complete pairs plus one
source. Every arm returned `detected`, including candidates as aggressive as
`44.2 dB` at residual RMS `1.58`, roughly three times the perturbation that
empties the local proxy.

The running total across both providers is nine source and candidate pairs and
zero verdict changes. The strongest frozen candidates, near `38 dB`, are still
queued behind the rate limit; completing them finishes one pass rather than
starting a new attempt.

### 2026-08-15: what the periodic features actually separate

The detector question needs no oracle at all. Both providers ship images from
the same pipeline with and without a watermark assertion, identical claim
generator and action list except for `watermarked.unbound`, so signed provenance
supplies the control group directly. `probe_render_fingerprint.py` scores five
groups with both shipped features.

| Group | n | Google feature accepts | OpenAI feature accepts |
| --- | --- | --- | --- |
| google watermarked | 300 | 83.3% | 0.0% |
| google, same pipeline, no assertion | 70 | 58.6% | 0.0% |
| openai watermarked | 300 | 2.3% | 2.3% |
| openai, same pipeline, no assertion | 287 | 0.7% | 4.5% |
| unrelated control | 299 | 7.4% | 1.0% |

Two results, and they land differently.

The OpenAI period-8 expert has no detection power on the general corpus. It
accepts `2.3%` of watermarked rows and `4.5%` of same-pipeline rows without an
assertion, so the difference is absent and its sign is inverted
(`p = 0.956` one-sided for the intended direction). The sixteen rows the frozen
proxy "clears" are a selected subset, not evidence of a working expert, and this
branch must not reach a shipped universal detector.

The Google period-16 feature does separate, but mostly by provider rather than by
watermark. It accepts `83.3%` of watermarked Google rows against `7.4%` of
unrelated controls, yet still accepts `58.6%` of same-pipeline rows carrying no
watermark assertion. The watermarked-versus-no-assertion gap is real
(`p = 1.6e-05`, medians `0.513` against `0.206`), so the feature is not purely a
render fingerprint, but a majority of its accepts on that generator's own
non-asserted output makes it a provider detector with a watermark-correlated
component, not a watermark detector.

One caveat bounds both readings: a missing C2PA watermark assertion is not proof
that no watermark was embedded. The no-assertion group is a lower bound on the
false-positive rate, not a measured one.

Neither feature transfers across providers, `0.0%` and `2.3%` in the off-diagonal
cells. That, and not the removal result, is the obstacle to a universal
detector: there is currently no shipped feature that reads a watermark rather
than a generator.

### 2026-08-15: a patch-local feature that beats the OpenAI branch

The evaluation set holds the confounds fixed rather than trusting them: one
claim generator per provider, PNG only, and every control matched to positives of
the exact same pixel dimensions, so size, codec and generator cannot carry the
label. Folds are taken over matched strata, content hashes are deduplicated
before assignment, and both the ridge strength and the feature family are chosen
inside the training folds. That yields `64` Google groups over `301` images and
`231` OpenAI groups over `1,081`.

The features are patch-local and every one is a ratio: normalized radial band
fractions, comb-to-background contrast at both periods, normalized periodic
autocorrelation, and the residual's own small-lag autocovariance grid, in
luminance and two opponent planes. Nothing is folded, so nothing is comb-blind.

| Provider | Shipped AUC | Patch-local AUC | Delta, 95% CI over groups |
| --- | --- | --- | --- |
| openai | 0.510 | 0.605 | `+0.095` `[+0.045, +0.147]` |
| google | 0.665 | 0.629 | `-0.036` `[-0.122, +0.050]` |

The OpenAI result is the first local expert on that provider with measurable
detection power, against a shipped branch that sits exactly at chance. Twenty
group-preserving label permutations put the null at `0.492 ± 0.020` with a
maximum of `0.525`, so the new score clears its own null by more than five
standard deviations and by more than the largest permutation.

The Google branch is not improved and the shipped feature stands. A full-sample
ablation showed the band subset alone reaching `0.676` there, above the shipped
`0.665`, but nested selection did not reproduce it, so that number was selection
noise on the evaluation set and is recorded here only as the trap it is.

Two limits bound the claim. Sensitivity at ten percent control acceptance is
`0.147` against the shipped `0.143`, so the gain lives in the middle of the ROC
and not at the precision-first operating point a shipped verdict needs. And a
model fitted on one provider scores the other at `0.535` and `0.469`, so this is
a better provider-specific expert, not a universal detector.

An ablation locates the OpenAI signal in the normalized spectral bands, `0.608`
alone against `0.571` for the autocovariance grid, `0.508` for comb contrast and
`0.460` for the periodic lags. The two periodic families, which are what the
shipped experts measure, contribute nothing on that provider. The across-patch
spread block currently covers only the luminance channel, which is a defect in
the feature vector rather than a finding; it scores at chance either way.

### 2026-08-15: the Google branch is control-limited, not feature-limited

A second attempt targeted three named causes rather than the score. The strata
were deepened to twelve positives each, `533` against `63` controls. The fold
assignment was repeated over seven seeds with the out-of-fold ranks averaged,
because a single assignment over 64 groups made the inner selection noisier than
the effect it selects on. And the shipped score was offered to that selection as
one candidate among the feature families, so a result below the baseline means
the selection failed rather than that the baseline is unbeatable.

It did not improve. Global AUC came to `0.678` against the shipped `0.689`, a
delta of `-0.010` with a group-bootstrapped interval of `[-0.071, +0.046]`.

Sensitivity at ten percent control acceptance did rise, `0.240` to `0.296`, and
that is the single number a precision-first reading would have quoted. It is an
artifact of one threshold. Partial AUC over the whole region below ten percent
control acceptance moves the other way, `0.164` shipped against `0.092`, so the
model is worse everywhere in the operating region except at the exact point the
first metric sampled. One threshold is not an operating point.

The binding constraint is the control count. Sixty-three controls put the
bootstrap interval on any delta near `±0.06`, so an improvement of two or three
points cannot be demonstrated here whatever the features do, and further model
search on this set is selection on the evaluation data rather than progress.
Unblocking the Google branch means more images from that generator known to
carry no watermark, not more features.

That is what a provider verifier would supply, and Google has no open route to
one. Three candidates were checked against their own pages on 2026-08-15; the
fetched evidence is archived under
`.local-eval/synthid/provider-access-2026-08-15/`.

**Vertex `WatermarkVerificationModel` is not a route.** The class survives in
`vertexai.preview.vision_models` with `verify_image` returning a `decision`
field, but the whole generative-AI preview module is marked deprecated as of
2025-06-24 with removal scheduled 2026-06-24, a date already past, superseded by
the Google Gen AI SDK. The documentation page
`vertex-ai/generative-ai/docs/image/verify-watermark` returns HTTP 404, including
its locale variants and the legacy `cloud.google.com` redirect. Whether the
backing endpoint still serves is unverified, and no successor page was found.
An earlier revision of this document named this the programmatic route; that was
wrong.

**Google Cloud's AI Content Detection API is real but answers a different
question.** It is documented at
`gemini-enterprise-agent-platform/models/ai-content-detection` (last updated
2026-08-13) as Private Preview behind an allowlist intake form whose own title is
`go/detect-ai-content-allowlist`. The page never uses the words SynthID or
watermark. It describes a classifier that analyzes "pixel-level artifacts, noise
patterns, and spectral anomalies" and states that "Support for C2PA metadata
detection is not included". So it decodes no watermark and no manifest: it is a
probabilistic AI-generated-image classifier. It cannot supply the ground-truth
negatives the Google branch is short of, though it would serve as an independent
comparison baseline for our own detector.

**The DeepMind SynthID Detector reads the watermark but is closed to us.** It is
a portal, not an API, and its early-tester form ANDs three requirements, the
first being "You are an active journalist or verification professional", with a
news-publisher corporate email and required News Publisher and Publisher URL
fields. The words research, academic and developer do not appear on the form at
all. Research access is excluded by construction rather than by refusal.

The Gemini app therefore stays the only working route, at roughly two checks per
account per two hours. The Google branch's control shortage has no near-term
provider-labelled fix.

### 2026-08-15: the two providers are architecturally different (S4)

Cross-image correlation of the folded residual, unit-normalized, over the matched
sets plus 200 unrelated controls. Three pairings are needed, not one: a high
positive-to-positive correlation alone cannot tell a watermark carrier from a
generator fingerprint, because the controls come off the same generator.

| Provider | Representation | pos-pos | ctl-ctl | pos-ctl | pos-unrelated | chance |
| --- | --- | --- | --- | --- | --- | --- |
| google | tile16 | `+0.326` | `+0.158` | `+0.224` | `+0.023` | `0.036` |
| google | tile8 | `+0.420` | `+0.223` | `+0.304` | `+0.035` | `0.072` |
| openai | tile16 | `+0.032` | `+0.024` | `+0.027` | `+0.008` | `0.036` |
| openai | tile8 | `+0.035` | `+0.031` | `+0.033` | `+0.009` | `0.072` |

Google images share a fixed, phase-coherent pattern at both geometries, an order
of magnitude above chance, and unrelated images do not carry it at all. OpenAI
images share nothing: every cell sits at or below the chance floor, which is what
a content-dependent post-hoc neural encoder produces and what the SynthID-Image
paper describes. The comb experts work on one provider and not the other because
the providers are not doing the same thing.

The Google numbers say something sharper than "positives correlate more". A
single shared pattern present in two classes at different amplitudes gives a
cross-correlation at the geometric mean of the two within-class values, whereas
two distinct components would fall below it. Observed against predicted:
`0.224` against `0.227` for tile16, `0.304` against `0.306` for tile8, ratios of
`0.985` and `0.993`. So it is one pattern, not a watermark carrier layered on a
separate generator fingerprint, and the controls carry that same pattern at
roughly half the amplitude of the positives.

Three readings survive that, and L1 decides between them: the controls are
watermarked after all and the label is wrong; the mark modulates the amplitude of
a fixed carrier the generator always emits; or the carrier is a generator
artifact whose strength happens to track the assertion. The first would also
explain the Google feature's 58.6% acceptance on controls without any of it being
a false positive.

The magnitude-only rows are reported in `shared-carrier-report.json` but carry
little: discarding phase leaves the generic image spectrum, and unrelated
photographs already correlate at `+0.348` there, so the floor is too high for the
measure to separate anything.

### 2026-08-15: the Google tile space is exhausted

S4's geometry suggests an estimator. If the classes differ by the amplitude of one
shared pattern, a discriminant whitened by the within-class covariance should beat
a plain cosine against a template, which weights all 768 tile dimensions alike.
`detector_tile_lda.py` fits exactly that, in the same space, with the family and
penalty chosen inside training folds and seven repeated fold assignments.

It does not. Google scores `0.644` against the shipped `0.689`, a delta of
`-0.044` with interval `[-0.090, +0.004]`.

The failure follows from the finding rather than contradicting it. When two
classes carry ONE pattern at different amplitudes, the optimal statistic IS the
projection onto that pattern, which the cosine already computes; whitening adds
estimation variance over 64 controls and buys no direction that was not there.
Read together with the earlier 124-dimensional attempt, this closes the question
the two of them were confounding: the Google shortfall is not the metric and not
the feature bag. The folded tile simply contains one usable degree of freedom,
and the shipped score already spends it.

Any Google improvement must therefore come from outside the fold, from the
content the comb filter destroys before the template is applied. That is S2, and
S2 needs the provider oracle.

One reporting caveat: this run scored both providers against
`folded_template_score`, which for OpenAI is the Google template applied to
foreign images and not that provider's shipped expert. The OpenAI delta printed
in `detector-tile-lda.json` is therefore against the wrong baseline. Against the
right one the tile discriminant reaches `0.565` versus the period-8 expert's
`0.510`, which is still below the `0.605` the patch-local features already reach,
so it is not an improvement on either provider.

### 2026-08-15: the implementation is sound, the date is innocent, the product is not

Four register entries closed offline in one pass.

**M3, the shared implementation, holds.** Every removal and detection result runs
through `folded_template_score`, so a defect there would produce both at once and
look like two findings agreeing. `probe_m3_independent_fold.py` re-derives the
statistic from its definition over 40 real images, folding by explicit modulo
accumulation and blurring with a hand-rolled separable Gaussian rather than the
OpenCV call, so a shared bug has nowhere to hide. Maximum score gap `3.7e-4`,
median `1.3e-4`, against scores spanning `0.01` to `0.85`; the residue is the
shipped path's float32 blur. The implementation is not the explanation.

**L3, the date confound, is refuted.** Month alone predicts the label at `0.524`
on Google and `0.491` on OpenAI, both chance. Inside a single month the Google
signal is not merely intact but stronger than pooled: `0.723` in 2026-06 over 44
controls and 336 positives, against `0.689` pooled. The OpenAI shipped expert
stays at chance within every month, as it is everywhere.

**R2 says the OpenAI result is not a geometry artifact, and adds a caution about
Google.** Restricted to one exact pixel size and pooled over 13 strata weighted by
pairs, the patch-local feature scores `0.577` against the shipped `0.478`, an edge
of `+0.099` that matches the `+0.095` measured pooled. So the improvement survives
the strongest available geometry control. Individual strata swing from `0.381` to
`0.799`, which is what 5 to 55 controls per stratum buys and is why the pooled
figure is the one to quote. Google has only three usable strata and one dominates
them, so its within-size figures, `0.767` patch and `0.763` shipped, are recorded
as underpowered rather than as an improvement.

**D1 is the finding that matters for the product, and it is bad.** Held at one
percent control acceptance, which with 63 and 229 controls means clearing every
control in the set, the fraction of positives that can be called at all is:
Google `0.045` shipped and `0.000` patch-local; OpenAI `0.025` shipped and
`0.018` patch-local. Every headline AUC in this document lives in the middle of
the ROC. At a precision-first threshold neither provider's detector can call more
than a twentieth of true positives, and on Google the newer feature calls none.

That reframes the program. The gap between `0.605` AUC and a shippable verdict is
not a matter of a few more points; nothing currently supports a positive claim at
a defensible false-positive rate. Either the operating point moves, which means
accepting a control acceptance far above one percent and saying so in the product
surface, or the representation has to change enough to move the tail rather than
the middle.

### 2026-08-15: U2, the detectors say "synthetic", not "watermarked"

Until now every control was a photograph or same-provider output. Neither can
catch the failure that decides whether a positive verdict may be reported as a
provenance claim: firing on an AI-generated image from a generator that embeds no
SynthID. The corpus carries 223 such images with their own signed provenance, so
the label is read from a manifest rather than inferred from a path.

Acceptance rates, each feature at its own threshold. The patch-local models have
no published threshold, so each is cut where one in ten of its own provider's
legitimate controls is already accepted.

| Generator | n | Google shipped | OpenAI shipped | patch Google | patch OpenAI |
| --- | --- | --- | --- | --- | --- |
| Adobe Firefly | 86 | 0.34 | 0.06 | 0.51 | 0.24 |
| Microsoft Designer | 20 | 0.20 | 0.00 | 0.40 | 0.25 |
| Microsoft Responsible AI 1.0 | 55 | 0.00 | 0.24 | 0.31 | 0.20 |
| PixelBin.io | 35 | 0.31 | 0.00 | 0.37 | 0.09 |
| fal-ai/seedvr | 10 | 0.00 | 0.00 | 0.80 | 0.00 |

The Google shipped feature accepts 7.4% of unrelated photographs and 83.3% of
watermarked Google images. It accepts 34% of Adobe Firefly output, which carries
no SynthID. The patch-local models are worse: cut at a level that admits one in
ten legitimate same-provider controls, they admit half of Firefly and four fifths
of fal-ai.

No threshold repairs this. A detector that fires this often on foreign synthetic
images is measuring generation, not the watermark, and a positive verdict from it
cannot be reported as "SynthID detected" without asserting a provenance the
evidence does not support. That is a correctness bound on the product surface,
not a performance target: it constrains what a positive may be called, whatever
the AUC.

It also bounds the reading of every earlier number here. The Google feature's
`0.665` to `0.689` AUC against same-provider controls was already suspected of
tracking the generator; these rows show it tracks synthesis more broadly than
that, across vendors with unrelated pipelines.

### 2026-08-15: a high number that must not be quoted

Following U2, the features were asked to separate watermarked images from
AI-generated images of other vendors, whose own provenance says they carry no
SynthID. Folds keep each vendor whole, so no fold trains and tests on the same
vendor.

| Provider | vs foreign generators | vs same-provider controls |
| --- | --- | --- |
| google | `0.917` | `0.636` |
| openai | `0.806` | `0.610` |

The high column is confounded and is recorded here so that it is not later
mistaken for a result. Separating a Google image from an Adobe Firefly image is
easy because the generators differ, and the same features already score `0.917`
and `0.806` at telling providers apart in every other experiment in this
document. The comparison cannot isolate the watermark, because changing the
control set also changed the generator.

The design flaw is not fixable by choosing better foreign vendors: any control
drawn from a different generator carries generator identity along with the
absence of the mark. The only control that varies the watermark while holding
the generator fixed is same-generator output without the mark, which is the
scarce set L1 is currently validating and the set M2 proposes to mint. Every
unconfounded discrimination this program has measured sits at `0.61` to `0.69`,
and that band is the honest state of the detector.

### 2026-08-15: M1, the pipeline is not the problem

Every negative result here is ambiguous between two causes: the features cannot
see watermarks, or SynthID is harder than the data allows. Embedding a watermark
ourselves separates them. The ground truth is not a provenance assertion that may
be wrong, and each carrier is its own control, so image and mark never straddle a
fold.

`invisible-watermark`'s `dwtDct` mark was embedded into 220 unrelated carriers at
three amplitudes, the library's own and two attenuations, because a mark loud
enough for a naive detector proves nothing about a signal at SynthID's level.

| Strength | AUC | TPR at 10% FPR | median residual RMS |
| --- | --- | --- | --- |
| full | `0.990` | `0.995` | `2.28` |
| half | `0.974` | `0.977` | `1.17` |
| quarter | `0.936` | `0.895` | `0.563` |

The last row is the one that matters. A residual RMS of `0.563` is the amplitude
band this program has been measuring SynthID in all along: the estimated periodic
residuals and every attack candidate sit between `0.45` and `0.6`. At that
amplitude the same feature extractor, the same folds and the same estimator reach
`0.936`, against `0.53` to `0.69` on real provider data.

So the machinery is sound at the relevant signal level, and the shortfall on real
data is a property of the task rather than of the code. Two candidate causes
survive, and they are not exclusive: SynthID is structured unlike a fixed-carrier
mark, which S4 already showed directly for OpenAI, and the labels are heavily
contaminated, which L1 is showing now.

The bound on this result: `dwtDct` applies one payload by one algorithm to every
carrier, so all marked images share a systematic perturbation. That is an easier
target than a content-adaptive neural encoder. M1 establishes that the pipeline
finds a fixed-structure watermark at SynthID's amplitude, not that it would find
any watermark. Repeating it with a neural watermarker is the next step, and it
would close the remaining ambiguity.

### 2026-08-15: what property of a mark actually defeats these features

M1 left one ambiguity: it found a fixed-structure mark and missed SynthID, and
content-adaptivity was the obvious suspect. Synthetic marks separate the
properties, all scaled to the same residual RMS so amplitude explains nothing.

| Mark | AUC | median residual RMS |
| --- | --- | --- |
| fixed, one pattern for every carrier | `0.670` | `0.626` |
| keyed, a different pattern per carrier, same band | `0.669` | `0.626` |
| keyed and perceptually masked by local texture | `0.546` | `0.577` |

Content-adaptivity is not the obstacle. Fixed and keyed are indistinguishable,
`0.670` against `0.669`, because the band features read the shared spectral
signature rather than the pattern. Perceptual masking is the obstacle, and it
costs three quarters of the separation.

That places every number in this document on one scale. At a residual RMS near
`0.6`: `dwtDct` scores `0.936`, being structured and unmasked; diffuse band noise
scores `0.670`; masked diffuse noise scores `0.546`. Real SynthID scores `0.53`
to `0.69`. The provider mark behaves like a masked, diffuse, mid-band
perturbation, which is what a competent perceptual encoder is built to be, and
our `0.605` is at the ceiling of what global hand-crafted statistics reach
against one, not short of it.

Undoing the mask does not help. Dividing the residual by the same local-activity
estimate before extraction scores `0.538` against `0.546` ordinary. The reason is
structural rather than a tuning failure: the band features are energy fractions
and therefore scale-invariant, so a mask that is nearly constant across a patch
already cancels. The loss comes from the correlation masking creates, not the
scaling. A mark whose amplitude tracks local texture is strongest exactly where
the image's own residual is strongest, so its local signal-to-noise is flat and
equal to its global value, and no region remains where it stands out. Statistics
of this family live on such regions.

The consequence for the program is concrete. Beating a perceptually masked mark
requires knowing its pattern, which is a decoder with the key, not a better
description of the residual. Absent the key, a local detector is bounded near
where ours already sits, and the remaining levers are the label quality that L1
is measuring and a learned decoder trained on genuinely clean pairs.

A defect found and fixed in passing, recorded because the first version of this
test returned a plausible number: the demasking probe originally rebuilt an RGB
image from the corrected residual with a hand-written inverse of the opponent
transform whose round-trip error was 15 grey levels, against marks of half a
level. It scored `0.529` and looked like a clean negative. The residual is now
handed to the extractor directly, and the probe asserts that a supplied residual
reproduces the extractor's own features exactly before it measures anything.

### 2026-08-15: the Google label rule is refuted, and the Google feature is blind

The M2 pilot was meant to mint unwatermarked pairs with Imagen's `addWatermark`.
It failed at the first step and produced something more decisive than it was
designed to.

Imagen is no longer in the model garden. All of `imagen-4.0-generate-001`,
`imagen-3.0-generate-002` and `imagegeneration@006` return 404, and a listing of
the 127 available publisher models contains no `imagen-*` entry. Image generation
now runs on `gemini-2.5-flash-image` and the `gemini-3.x-*-image` family, and
those reject the parameter outright: `Unknown name "addWatermark" at
'generation_config': Cannot find field`. There is no watermark toggle on any
current Google generation path, so M2 as designed cannot be built.

The Gemini API documentation states the policy directly, twice, rather than
leaving it to inference: "All generated images include a SynthID watermark." Its
optional-configuration section lists output format, aspect ratio and image size,
and nothing else. Generated images are therefore watermarked by construction.

Two results follow, and both are read off documentation plus four generated
images rather than off an oracle.

**The Google feature does not see SynthID on current output.** Four freshly
generated images scored `0.018`, `0.103`, `0.054` and `0.071` on
`folded_template_score`, a median of `0.063`, against a corpus median of `0.51`
for watermarked rows and `0.01` for unrelated photographs. The feature is not
over-firing on unwatermarked images; it fails to fire on watermarked ones. Taken
with the U2 result, it detects a generator era rather than a watermark, and that
is now demonstrated rather than suspected.

**L1's refutation extends to Google.** Every one of those four images carries a
C2PA manifest from the same claim generator as the corpus positives, "Google C2PA
Core Generator Library", and none of them asserts `watermarked.unbound`; their
action lists are `c2pa.created` and `c2pa.edited` only. Google's own API emits a
watermarked image with no watermark assertion. The rule that a missing assertion
means a missing watermark is therefore refuted for both providers, and the 70
Google controls are unreliable in exactly the way the 287 OpenAI controls were
measured to be. Every Google AUC in this document rests on that label.

A method note. The first attempt to settle this empirically was going to upload a
generated image to the Gemini app verifier. The composer renamed the attachment
from `.png` to `.jpg`: the interface re-encodes uploads. A `not_detected` verdict
after re-encoding would have been uninterpretable, and the query was abandoned
rather than spent. Today's earlier oracle runs went through the same interface
and returned `detected`, so the mark survived re-encoding there, but no negative
verdict obtained through that surface should be trusted without controlling for
it.

### 2026-08-15: what the corpus labels actually mean, and what that invalidates

Earlier entries in this log, written the same day, describe both providers'
label as the `watermarked.unbound` action. That is wrong for Google and the error
propagated into every Google experiment here. The corpus was counted directly.

`watermarked.unbound` appears in `0` of `4,780` Google rows and `7,427` of
`7,731` OpenAI rows. It is an OpenAI field. Google's `synthid_from_provenance`
flag is driven instead by `synthid_watermark`, "present according to Google LLC
provenance", which is set from the presence of a Google generation claim.

The two classes therefore differ by something other than a watermark:

| | Google positives, 4,710 | Google "controls", 70 |
| --- | --- | --- |
| claim generator | Google C2PA Core Generator Library | Google C2PA Core Generator Library |
| issuer | Google LLC | Google LLC |
| actions | `created, converted, edited, opened` and similar, always with `created` | `converted, edited, opened`, never `created` |
| platform | Google (Gemini / Imagen) | C2PA signer: Google LLC (no known AI generator named) |

The controls are images Google **edited or converted but did not create**. So
every Google number in this document, the `0.665` to `0.689` shipped AUC, the
patch-local comparisons, the within-month and within-size splits, measured
Google-generated against Google-edited. That is a synthetic-versus-photographic
task, not a watermark task, and it explains U2 without any further hypothesis:
a classifier trained that way fires on Adobe Firefly because Firefly output is
synthetic.

There is no Google watermark control set in this corpus and there never was.

Worse for the program, one cannot be collected. Google's documentation states
that all images generated by its image models carry a SynthID watermark, and the
`addWatermark` toggle that could produce an exception exists only on Imagen,
which returns 404 for every documented model id on an accessible project. The
same documentation adds a restriction that rules out the cleanest experiment even
with access: "You can use the `seed` field to get deterministic output only when
this field is set to `false`", so an identical image with and without the mark
cannot be generated at all.

For OpenAI the label is the right field but the negative class is contaminated,
measured at 20 of the first 28 non-asserted rows carrying a watermark.

What survives from the day's work is what did not depend on these labels: M1 and
M1b, which used marks embedded here with known ground truth, and M3. The rest
needs redoing on a task that is stated correctly first.

### 2026-08-15: the periodic lattice is JPEG, not SynthID

Classifier work was abandoned here for signal characterization. A discriminative
model finds whatever separates its classes, and this program has now measured
that what separates them is the generator and its era, so every trained number
carries that confound inside it. The question was changed to what the watermark
does to pixels.

Without a clean copy of the same image the perturbation is isolated by averaging:
a content-independent component accumulates over hundreds of images while content
cancels. Three groups make the comparison specific. Photographs carry neither
mark nor synthesis. Foreign-generator images are synthetic and carry no SynthID,
so subtracting them removes what is generic to generation and leaves what is
particular to Google.

The difference looked, at first, like a discovery. Google minus foreign
concentrates on the period-8 lattice far beyond chance: the lattice holds `0.024%`
of bins and takes `2.7%` of the difference in luminance, `10.8%` in red-green and
`7.8%` in blue-yellow, enrichments of `112x`, `441x` and `318x`. The strongest
individual bins sit at one eighth and three eighths of a cycle per pixel, on the
same lattice.

Period 8 is also the JPEG block size, and the control settles it. Measuring each
group's own lattice concentration rather than a difference:

| Group | L | R-G | B-Y |
| --- | --- | --- | --- |
| corpus Google | `17.8` | `117.4` | `70.1` |
| foreign generators | `18.9` | `73.1` | `68.9` |
| photographs | `9.1` | `30.4` | `25.5` |
| fresh Google from the API | `3.4` | `9.5` | `6.2` |

The fresh images are watermarked by documented policy and have never been through
a lossy codec, and they carry the least lattice structure of any group. If the
lattice were the watermark they would carry the most. Foreign generators carry
nearly as much as Google. The ordering tracks compression history.

So the periodic lattice this program has been measuring, attacking and building
experts on since the beginning is block-transform structure, not SynthID. That
one fact retro-explains the rest without further hypotheses: why the shipped
detector scores `0.51` on corpus positives and `0.063` on fresh output from the
same provider, why it fires on Adobe Firefly, and why S4 found a shared fixed
pattern among Google images that the controls also carried.

It also reinterprets the removal programme. Nine oracle pairs changed no verdict
at up to three times the perturbation that empties the local score, and the
reason is now plain: the attacks removed and replaced compression structure. The
watermark was never touched, so there was nothing for a provider decoder to lose.

### 2026-08-16: the large expert detects fresh out-of-distribution output

The previous entry concluded that the periodic structure this programme has been
measuring is block-transform artifact rather than SynthID. That conclusion was
drawn from `1024`-square API images and generalized too far. It is corrected
here.

Reading the earlier log supplied the constraint that prompted the check. The
2026-08-14 entry records `large-v1` validated against seven official Google pixel
verdicts and accepting none of `2,637` feature-unseen controls, and it records a
codec stress test in which JPEG-95 and JPEG-90 each took detection from seven of
seven to zero of seven. A compression artifact would be reinforced by
re-encoding, not destroyed by it.

`gemini-3.1-flash-image` on the global endpoint accepts `imageConfig.imageSize`
and returns `5632x3072`, `17.3` megapixels, inside the large expert's routed
scope and one of the geometries it was calibrated against. Eleven such images
were generated, never written through a lossy codec, and watermarked by the
documented policy that all generated images carry SynthID.

The shipped runtime detected eight of eleven. The crossing scores cluster just
above the gate at `1.036` to `1.061`; the three misses returned `0.637`, `0.568`
and `0.184` and were reported `indeterminate` rather than negative.

That model did not exist when the expert was calibrated, and these files have no
codec history, so this is generalization across both model generation and
pipeline. `large-v1` is a working local SynthID detector for large images at
roughly `73%` sensitivity on out-of-distribution positives.

The corrected reading of the two experts:

- the **large** expert measures something real that survives a new model
  generation and the absence of compression;
- the **fixed** period-16 expert scores a median of `0.063` on fresh `1024`-square
  output against `0.51` on the corpus, and the corpus small-image chroma lattice
  is ten times the fresh one at matched size with luminance nearly equal, which
  is the signature of 4:2:0 chroma subsampling. Its corpus performance is
  therefore still best explained by compression history, but that explanation is
  now specific to the small-image expert rather than to the periodic family.

What remains genuinely open is whether the fixed expert is confounded or merely
out of regime: `1024` squares may be below the size at which this signal is
recoverable at all, which the large expert's scope already hints at.

### 2026-08-16: a pre-registered test that failed, and why it could not have worked

The mid-band chroma lead from nine images was carried into a confirmation run on
fresh data the hypothesis had not seen. It is recorded here because a
pre-registered test is worth nothing if only its successes are written down.

Against 223 foreign-generated controls, band 4 in the opponent channels came back
at `z = -8.33` and `z = -5.69` on 150 fresh OpenAI images: enormous, and pointing
the opposite way to the lead. The exploratory scan behaves the same, with bands
0, 1 and 6 and the lag-one autocorrelation all separating at `|z| > 6`.

Two reasons this is not evidence about a watermark. The lead came from Google
images and only the OpenAI arm reached usable size, so the populations differ.
More fundamentally, the control differs in generator, so a large `z` here says
`gpt-image` at low quality has different spectral statistics from Firefly and
Microsoft output, which was never in doubt. This is the same confound identified
in the foreign-generator classification test, and averaging does not escape it:
without an unwatermarked control from the same generator, no amount of `n`
converts a generator difference into a watermark difference.

The Google arm stands at ten `1024`-square images, too few to test, and the same
objection would apply to it.

The practical consequence points at the day's other result. The signal that is
demonstrably present is the one `large-v1` reads at 17 megapixels, and every
OpenAI image collected here is `1024` square, which on the Google side is the
regime where nothing is recoverable. The next OpenAI attempt should be made at
large geometry rather than with more small images.

### 2026-08-16: a seven-pixel shift, and what `large-v1` actually reads

The crop ladder was run to find the size at which the signal stops being
recoverable. It found no such size, and answered a different question instead.

Cropping preserves pixels and lattice phase where resampling would not, so crops
of the eleven fresh 17-megapixel positives isolate size from every other
variable. Two ladders were taken, one from the origin and one from an offset of
seven pixels, which is deliberately not a multiple of the tile period. Components
are medians over the eleven images, folded whole rather than per window.

| Crop | Aligned fixed | Aligned R-G | Shifted fixed | Shifted R-G |
| --- | --- | --- | --- | --- |
| 5632x3072, 17.3 MP | `0.338` | `0.962` | - | - |
| 4096x2560, 10.5 MP | `0.342` | `0.962` | `0.044` | `-0.111` |
| 2048x2048, 4.2 MP | `0.335` | `0.963` | `0.044` | `-0.104` |
| 1024x1024, 1.0 MP | `0.366` | `0.959` | `0.039` | `-0.081` |
| 512x512, 0.3 MP | `0.350` | `0.937` | `0.046` | `-0.068` |

Two facts, and the second settles the programme's central question.

The structure does not decay with size at all. It is as strong in a
512-square crop as in the full 17 megapixels, so the fixed expert's blindness on
small images was never a size limit.

And it is phase-locked to the image origin. A seven-pixel shift takes the
red-green agreement from `0.962` to `-0.111` at every size.

That is disqualifying. The SynthID-Image paper reports `99.97%` TPR in its
"Spatial worst" category, aggressive crop combined with resize, evaluated at
512 squares, and Google's own page describes the mark as "designed to stand up to
modifications like cropping". A structure destroyed by a seven-pixel crop cannot
be a watermark whose published evaluation survives far worse.

So `large-v1` reads a generation-pipeline artifact: origin-anchored,
size-invariant, chroma-dominant, at the tile period. A tiled upsampling or
decoding grid fixed to the image corner has exactly those properties. Every
observation about it now follows without further hypotheses: seven official
positives that all came from one pipeline family, none of 2,637 photographic
controls which have no origin-locked lattice, eight of eleven fresh
`gemini-3.1-flash-image` outputs from the same family, `0.063` on
`gemini-2.5-flash-image` output from a different one, and destruction by JPEG,
which imposes its own grid and breaks the phase agreement.

A method note that should outlive this entry: the shift test costs one line and
would have settled this in a day at any point in the last three months. Any
candidate local expert should be required to survive a non-aligned crop before
it is calibrated, let alone shipped, because the mark it claims to detect
provably does.

Also recorded, since it produced a contradiction worth explaining: the table
above folds each crop whole, while `large_image_components` minimises its
components over phase-aligned 2,048 windows. The two are not the same statistic,
which is why the shipped runtime detected eight of eleven full images that this
table's gate arithmetic rejects. The aligned-versus-shifted contrast does not
depend on that difference.

### 2026-08-16: specificity measured against what the route can be confused by

Every control rate this project publishes was measured on photographs: `0/2,637`
for the large branch, `0/5,993` and `0/2,366` for registered-v3, `0/41` on a
post-freeze acquisition. Photographs are the easy negative, because they carry no
origin-anchored lattice at all. Once the route is named a generation-pipeline
detector, the negative that decides its worth is other generators' output.

The production entry point was run over the 223 corpus images whose C2PA claim
generator is a non-Google vendor, so the cohort's composition is signed rather
than assumed.

| Claim generator | Detected | n | Rate |
| --- | --- | --- | --- |
| Adobe Firefly | 21 | 87 | `0.241` |
| Microsoft Designer | 3 | 20 | `0.150` |
| PixelBin.io | 5 | 36 | `0.139` |
| Microsoft Responsible AI Provenance | 0 | 14 | `0.000` |
| fal-ai/seedvr | 0 | 10 | `0.000` |
| Microsoft Responsible AI 1.0 | 0 | 56 | `0.000` |
| **total** | **29** | **223** | **`0.130`** |

The shipped runtime accepts about one in eight images from other generators, with
a one-sided 95% lower bound of `0.095`, and reaches `0.241` on Adobe Firefly. The
highest foreign score was `3.01` against a threshold of `1.0`, so these are not
marginal crossings. Every acceptance came from registered-v3; the large branch
was either unsupported or indeterminate on this cohort.

That is the number a user meets. A tool that answers "is this AI-generated
content watermarked" will be pointed at AI-generated content, not at landscapes,
and against that population the published photographic control rates do not
describe its behaviour.

One measurement in the same run is reported but must not be read as a
false-positive rate. A cohort of 300 corpus rows with neither provider nor claim
generator was accepted at `0.090`. Those rows are unlabelled rather than verified
photographs and may contain AI images stripped of metadata, so the number is
uninterpretable in either direction. The COCO and Open Images collections behind
the published figures are not on disk, so the `0/2,637` result could not be
reproduced here and is neither confirmed nor contradicted.

### 2026-08-16: registered-v3 is phase-locked too, so the whole runtime is

Every acceptance in the entry above came from registered-v3, the default
ordinary-size route, and the seven-pixel shift test had only ever been run on
`large-v1` and the large branch's components. The same test was therefore run
through the production entry point on the ordinary-size expert: the 29
foreign-generator detections, plus 12 Google provenance-signed positives chosen
inside the registered geometry, each scored at diagonal crop offsets 0, 2, 4,
and 8 with a 32-pixel margin on every side.

| Cohort | n | Detected at 0 | Detected at 2 | Detected at 4 | Detected at 8 |
| --- | --- | --- | --- | --- | --- |
| Foreign-generator detections | 28 in cropped geometry | 28 | 0 | 27 | 25 |
| Google provenance positives | 12 (8 detected at 0) | 8 | 0 | 8 | 8 |

One foreign image left the registered geometry under the crop and could not be
shifted. Every one of the 36 detections that remained in geometry died at a
two-pixel diagonal offset: maximum foreign score `0.779` and maximum Google
score `0.311` against the `1.0` threshold, with medians of `0.0`. Four of the
twelve Google positives were indeterminate at every offset, a sensitivity miss
rather than a phase effect.

The recovery at offsets four and eight is itself the signature. Forcing the
period-16 fold on one Adobe Firefly detection gives `0.808` correlation at
offset zero, `-0.008` at offset one, about `0.14` at offset two, and back above
`0.77` at offset four: the matched structure repeats every four diagonal pixels
and is destroyed by anything in between. Period selection breaks as well; the
canonical winner jumps from `16.000` at offset zero to `9.590` at offset two and
the base statistic falls from `2.47` to `-0.58`.

A mark designed to survive cropping would score the same under a two-pixel
crop; the published SynthID evaluation retains 99.97% TPR under aggressive crop
and resize. Registered-v3 therefore reads the same origin-anchored,
crop-destroyed lattice established for `large-v1` in the seven-pixel entry, and
reads it in both directions: on the foreign generators it fires because those
pipelines also leave a sixteen-pixel origin-anchored structure that partially
matches the frozen Google template, and on the Google positives it reads the
same class of structure rather than a robust mark.

The S1 conclusion now covers every shipped periodic expert, not only the large
branch: the entire local pixel route detects generation-pipeline lattice, and
`identify`'s experimental `pipeline_lattice` signal is the honest name for all
of it. Any future expert must pass the two-pixel shift test before calibration,
per the R0 precondition.

### 2026-08-16: hypothesis register audit

A full pass over the register against the recorded evidence, closing what the
log already settles:

- M2 re-verified independently and settled. `imagen-4.0-generate-001` and
  `imagen-3.0-generate-002` return 404 in us-central1 and us-east1 on both GCP
  projects (`gen-lang-client-0926942364` and `raiw-cws-publish`);
  `gemini-2.5-flash-image` rejects `addWatermark` at request parsing
  (`Unknown name "addWatermark" at 'generation_config': Cannot find field`),
  and the current Gemini API documentation states for both the Gemini-image
  ("Nano Banana") and Imagen paths that all generated images include a SynthID
  watermark, with no disable parameter documented. No current Google path can
  emit an unwatermarked image.
- S2 moves from `open` to `partial`: the equal-RMS broadband arm of the
  account-isolated Google oracle group remained `detected`, first direct
  evidence against the between-lines scope at the tested perturbation class.
- S3 gains the Google nested test, which did not improve on the shipped score
  (`0.678` vs `0.689`), so the Google branch stays control-limited.
- L3 is settled by a direct date-only classifier (AUC 0.524/0.491).
- M3 is settled by the independent fold reimplementation (`3.7e-4`).
- D1 was asked and measured: 0-4.5% of positives decidable at one percent
  control acceptance.
- The ranked backlog gains an execution-state table: H1 executed and closed as
  a watermark route (it locks onto the S1-refuted pipeline lattice), H2
  rejected, H3 partially executed, H4 rejected as a standalone detector, H5
  advanced as research confirmation only, H6/H7 blocked on the oracle-positive
  corpus, H8 not started.

Nothing in this pass reopens a closed hypothesis, and no status moved on an
argument; every change cites a recorded measurement.

### 2026-08-17: era, vendor and codec audit of the patch-local bank

The power-spectrum bank cannot be blinded by translation, so the plain crop
test that killed the phase-locked experts certifies nothing for it. The axes
that can still separate a mark candidate from a pipeline signature are era
(the statistic must fire on corpus-era June-July positives, fresh
`gemini-2.5-flash-image` positives, and fresh `gemini-3.1-flash-image` 4K
positives alike, since the documented policy watermarks all three), vendor
(silence on C2PA-signed Firefly, Microsoft, PixelBin and fal-ai output), and
codec (survival of JPEG-95 applied to both classes, which the published mark
survives). `probe_era_vendor_codec.py` scored all 124 features over that grid:
positives 533 corpus rows plus 10 fresh-2.5 plus 11 fresh-4K, negatives 218
foreign-generator images, 300 seeded COCO photographs and 100 exact-1024 Open
Images photographs, three conditions each.

Result: zero features clear a minimum AUC of 0.55 over the 27 cells, zero
clear 0.6, 0.65 or 0.7. The best worst-cell is 0.481. The strongest cells are
the familiar ones, for example feature 47 at 0.9997 on corpus-native against
photographs, and every such feature collapses below chance in some other era
or codec cell. A ridge combination fitted on corpus-era positives transfers to
the fresh eras at AUC 0.388 native and 0.178 after JPEG-95; fitted on fresh
2.5-flash it transfers back to the corpus era at 0.243. The shipped folded
baseline repeats the phase-lock story per era: corpus 0.8845 native against
foreign generators falls to 0.4479 under a random 1-15 px crop, fresh 4K falls
from 0.859 to 0.4508. The one weak exception is the ten fresh 2.5-flash images
at 0.6114 under crop, which is also the set whose negatives are JPEG-stored
while the positives are PNG, so the codec asymmetry is untested there and the
sample is ten images.

What this settles: every separating statistic in the hand-crafted bank is
era-, vendor- or codec-bound, for the Google side measured directly and for the
OpenAI side already implied by the reversed pre-registered band test of
2026-08-16. Combined with the M1b masked-mark ceiling this closes the
hand-crafted branch as a SynthID route for both providers. The one untested
local route left is S5, a learned nonlinear model on residual patches, which
now carries the additional requirement demonstrated here: leave-one-era-out
transfer is the acceptance gate, because within-era separation is proven to be
pipeline separation.

### 2026-08-17: S5 first slice, learned patch detector under the era gate

`probe_s5_patch_conv.py` trains a 4-layer convolutional model on opponent-plane
Gaussian residual patches, eight unaligned random 128-px patches per image, so
no phase anchor to the image origin can exist by construction. Positives span
the eras available on disk: 535 corpus-era Google rows, 10 fresh
`gemini-2.5-flash-image`, 11 fresh `gemini-3.1-flash-image` 4K, 853 corpus-era
OpenAI rows and 151 fresh `gpt-image-1`. Negatives are 223 C2PA-signed foreign
generator images and 400 photographs (300 seeded COCO, 100 exact-1024 Open
Images). Each provider's model is trained on all eras but one and evaluated on
the held-out era, native, after JPEG-95 on both classes, and after a two-pixel
shift; the shift column equals the native one as designed, confirming the
construction. Three image-level label permutations put the null at 0.552,
0.553, 0.593 for Google and 0.540, 0.566, 0.554 for OpenAI.

The verdict repeats the U2 result for a learned model. Against photographs the
held-out eras score 0.711, 0.599 and 0.843 for Google and 0.791 and 0.767 for
OpenAI. Against foreign generators the same models score 0.442, 0.153 and
0.486 for Google and 0.455 and 0.153 for OpenAI: at or below the permutation
null, with the fresh eras driven below chance, meaning the model ranks
unwatermarked competitor output above freshly minted watermarked positives.
A Google-versus-photographs model that never saw a foreign generator accepts
94.6 to 100 percent of every foreign vendor at the photograph median, and
scores fal-ai output above Google's own fresh 4K era (AUC 0.946 against it).
One outlier cell, OpenAI fresh after JPEG-95 at 0.942 against photographs,
carries a storage asymmetry: the positives are stored PNG and the photograph
negatives JPEG, and re-encoding both does not equalize that history; it is
recorded as an anomaly, not a signal.

What this settles: the learned route at this capacity learns "AI render
versus photograph" and nothing vendor-specific. With S6 refuting every
hand-crafted feature across the same grid, R0 refuting the phase-locked
experts, and M1b bounding what per-image statistics can do against a properly
masked mark, three independent directions now agree: a per-image local
SynthID detector is not reachable from these pixels by statistical means at
any scale we can validate. Provider mark detection requires the provider
decoder, which exists as an API for OpenAI and as an access-gated service for
Google. The separable local signal that remains is the AI-render versus
photograph distinction, whose honest product framing is generation detection,
not watermark detection.

### 2026-08-17: matched-pair manufacturing and the distillation screen

Two measurements moved the "build our own detector" question:

The attack ladder pilot (`build_attack_ladder_batch.py` +
`run_attack_ladder_batch.py`, batch `attack-ladder-pilot-2026-08-17`, fifteen
arms, every arm submitted exactly once) took three verifier-confirmed OpenAI
positives through four attack mechanisms: JPEG q40, 0.5 bilinear resize,
sigma-8 noise plus JPEG q85, and 0.35 resize plus JPEG q50. Every single arm
returned `detected`, down to PSNR 20.1 dB and RMS 25.1. The OpenAI mark
outlives distortions that destroy the image as a product, and MarkNull-severity
attacks (which unwatermark Imagen-3 at PSNR 25.4 in their paper) do not flip
this verifier. Matched pairs cannot be manufactured by attacking positives:
the flip point sits beyond usable image quality, so same-content label
contrast is unreachable on this provider by the attack route. All three
re-verified sources stayed `detected`, so the run doubles as a health check.

The distillation screen (`probe_distillation_screen.py`) took the other
oracle-labeled contrast on disk, the 93 L1 control verdicts (65 detected, 28
not detected; metadata says "no watermark" for all of them, the oracle
supersedes it), and asked whether any local score reproduces those verdicts.
The shipped folded score reaches AUC 0.613 against a permutation null of
0.485 +/- 0.062, roughly two sigma. The best of 126 candidate scores reaches
0.649 before selection correction. A ridge fitted on the test rows themselves
(an explicit upper bound) reaches 0.693, while a ridge fitted on fresh mints
against photographs and foreign generators transfers at 0.409, below chance:
the mark-specific component does not survive provider-external training. The
confound audit shows the two verdict groups matched on median size (1.573 MP
both) and mixed across months and generators. Reading: a weak mark-specific
trace exists in the natural no-assertion region, with a per-score ceiling
around 0.61-0.65 at n=93. Tripling the labeled rows is the cheapest way to
either confirm that ceiling or climb it, so the L1 runner resumed on its
frozen 315-row plan the same day.

The distillation harness (`probe_distillation_train.py`, feature cache keyed
by content hash over all 329 factory-reachable rows, temporal folds by
collection month, nested-penalty ridge, max-statistic null) ran its dry pass
at the labels available that evening: 102 hard-contrast rows give temporal-CV
AUC 0.632, the 110-row pooled contrast 0.652, and one feature of 124 exceeds
the max-statistic null by two sigma (0.724 against 0.657 +/- 0.033). The
harness is the acceptance instrument for the growing label set; the decision
it will support at n near 280 is whether a weak-but-honest local component
ships under an `indeterminate`-heavy contract or the ceiling is declared.

### 2026-08-17 evening: decoder substitution and the state of Google access

Access re-checks first, both negative. The Vertex publisher model
`imagewatermarkdetector@001` returns "not servable in region us-central1" and
404 "not found or your project does not have access" in us-east4,
europe-west4 and asia-northeast1, on live credentials; nothing has opened
since the 2026-08-15 verification sweep. The hypothesized open release of
official image watermark weights ("synthid-suite") does not exist: GitHub,
the google-deepmind organization and the Hugging Face registries expose only
`synthid-text`. No official image detector or marker weights are public, so
no local distillation of the official encoder is possible either.

Decoder substitution was then tested as a matched-pair factory, on the
MarkNull mechanism reading that the mark lives in the decoder's output
distribution. `build_vae_pair_batches.py` froze five oracle-verified OpenAI
sources plus their float32 encode/decode round-trips through
`stabilityai/sd-vae-ft-mse` (latent mode, no noise, edge padding to a
multiple of eight), PSNR 22.3-32.0 dB, and `run_vae_pairs_openai.py`
submitted the five treated arms exactly once: every arm returned `detected`.
The OpenAI mark is not decoder-bound. It has now survived, with verdicts
unchanged, JPEG q40, 0.5 resize, sigma-8 noise plus JPEG q85, 0.35 resize
plus JPEG q50 down to PSNR 20.1, and a foreign-VAE reconstruction down to
PSNR 22.3. Every same-content pair-factory route is closed on this provider;
natural verdicts remain the only label source, which is the L1 extension
already running.

The matching Google batch `vae-pairs-google-2026-08-17` (three fresh
policy-watermarked generations, both arms, six submissions) is frozen for the
Gemini web verifier. MarkNull reports its SD-proxy regeneration flipping
Gemini verdicts at 25.4 dB, so this batch decides whether a plain
reconstruction (no latent optimization) already breaks the Google mark, and
with it whether the local negative factory is viable for Google. It requires
a browser session and is the next oracle action when one is available.

Resolved 2026-08-18 as `vae-pairs-google-v2-2026-08-18`, submitted manually by
the user through the Gemini web verifier. The first v1 attempt exposed a
known trap live: the source arm answered through C2PA Content Credentials
(metadata), and the vae arm returned visual analysis with no watermark check;
neither is a pixel verdict, and both were discarded. The v2 batch re-encoded
every arm through PIL so all arms carry identical chunk sets with the sources'
decoded pixels asserted identical, and added two attribution controls. All
eight arms returned explicit SynthID pixel verdicts: the three stripped
sources `detected`, the three VAE round-trips `detected` at 26.6-32.9 dB, the
COCO photograph `not detected`, and the C2PA-signed Adobe Firefly image `not
detected`. Two conclusions. First, the Google mark survives foreign-decoder
substitution exactly as the OpenAI mark does, so the plain-reconstruction
negative factory is dead on both providers; only MarkNull-style latent
optimization remains untested, and it destroys product quality (25.4 dB,
SSIM 0.80). Second, the controls give the web pixel verdict its first
attribution validation: it stays silent on another generator's AI image, so
within this session it read the Google mark or Google pipeline, not generic
AI forensics. That makes the earlier comb-vs-broadband and replace-095
verdicts, all answered on Google-generated pixels, retroactively cleaner as
mark-presence evidence than they were when recorded.

### 2026-08-18/19: the three-metadata-dataset program and the shared-component attribution

The negative-factory hypotheses being dead, the working program switched to
estimation on positives, organized as three datasets built only from
unambiguous metadata: G, Google `created` rows (4,710, watermarked by
documented policy); O, OpenAI `watermarked.unbound` rows (7,444, all oracle-
sampled rows detected); N, images that cannot carry SynthID (C2PA-signed
foreign generators plus C2PA-signed smartphone cameras; the 2026-08-18 web
control answered `not_detected` on a Firefly member). The 287 OpenAI
no-assertion rows and 70 Google non-`created` rows stay excluded (L1: 69% of
no-assertion rows are actually watermarked), and the 19,792 rows with unknown
provenance are not used as negatives.

`probe_shared_component.py` estimates, per class, the mean origin-aligned
opponent-residual pattern over up to six unit-RMS 256-px patches per image,
with split-half convergence bounds and photographs as the estimator null
(0.002 +/- 0.002, i.e. zero). Results: every AI class carries a strong shared
component (Google corpus 0.922 at n=535, OpenAI corpus 0.723, OpenAI fresh
0.978 at n=151, foreign generators 0.597, fresh Google 0.30-0.52 at n=10-11),
photographs carry none. Corpus-era components are compression-dominated (50-98%
of power on the period-8 comb); fresh-era components are not (4-10% comb).

The attribution of the shared component settles the mark question. In chroma,
Google eras correlate pairwise (R-G 0.29-0.59 full-field, 0.25-0.48 after
removing the JPEG comb) while OpenAI eras correlate at zero and OpenAI versus
Google at zero: the Google component is provider-specific but era-crossing.
But it is not the mark. Per-generator decomposition against the fresh 3.1-flash
mean shows Adobe Firefly at 0.572 and PixelBin at 0.406 full-band coherence
while Microsoft generators sit at 0.03-0.19 and fal-ai at 0.05; band-resolved,
Firefly tracks Google in every band including the high-frequency chroma bands
where the two fresh Google eras correlate (0.87 in the period-4 render band,
0.30-0.63 above 0.30 cyc/px). The 2026-08-18 oracle control verified Firefly
carries no SynthID. Therefore the era-crossing shared chroma component is a
renderer-architecture lineage shared by Google's decoders with Firefly and
PixelBin, absent from Microsoft, fal-ai and OpenAI, and the era genealogy is
visible directly (corpus and 3.1-flash share the period-4 band at 0.92 while
2.5-flash does not, matching the model lineages). No frequency band exists
where Google eras correlate and Firefly does not. A one-pixel diagonal shift
erases the component completely (0.48 to -0.05), the origin lock already
established for the lattice.

This closes the fixed-signature program: every image-independent signature
measurable from positives is renderer genealogy, and the mark itself presents
no fixed cross-image signature at all, consistent with a content-dependent
masked encoder (the M1b ceiling and the DeepMind patent model) and with S4's
zero carrier finding for OpenAI. The era-mean patterns are saved as amplified
PNGs (`shared-component/pattern-*.png`) so the pipeline signature is now
literally visible. The remaining measurable routes are the OpenAI natural-
verdict distillation (L1 at 227/315) and same-content differential analysis
inside near-duplicate groups, where a `created`+`edited` pair of the same
content is the one remaining place a mark-versus-mark contrast could exist;
the group miner `probe_near_duplicates.py` records mixed-assertion groups
explicitly for that.

The near-duplicate miner completed over all 12,511 provider rows: 170 groups
at Hamming 12, 21 cross-provider, 5 with mixed assertion, largest group 9.
Every cross-provider pair is a re-render (all 21 differ in raster size; the
closest, 896x1200 versus 895x1200, aligns at only 23.7 dB PSNR), so no
same-raster cross-provider pair exists and the cross-provider groups cannot
serve as same-content mark contrasts; they remain renderer-attribution
controls. The local half of decoder substitution also finished (`decoder-
substitution.json`, 287 photographs and 217 foreign round-trips completing
the earlier sets): zero of 124 paired-delta features clear the pre-registered
0.70 acceptance against both negative families (best 0.644), and the OpenAI
deltas point the wrong way entirely (0.23-0.30). This agrees with the oracle
outcome rather than contradicting it: since the mark survives the round-trip,
no mark-destruction signature exists to find, and the null result is the
expected shape.

A first product-shaped use of the renderer signature was measured the same
day: correlating an image's origin-aligned chroma residual against the frozen
era-mean signatures (matched filter over the three Google era patterns) gives
AUC 0.914 against foreign generators and 1.000 against OpenAI corpus images,
with 0.68 TPR at 10% FPR against foreign. The highest-scoring foreign images
are exclusively Adobe Firefly and PixelBin, the two generators that share the
Google renderer lineage, so the honest claim is "diffusion-render-family
generation detector" (Google, Firefly, PixelBin), not provider attribution,
and it inherits the one-pixel origin lock. It is a candidate second pixel
signal beside the pipeline lattice, to be frozen under the R0 gate before any
product integration.

The local half of the same probe (`probe_decoder_substitution.py`) runs the
round-trip over fresh and corpus Google, corpus and fresh OpenAI, foreign
generators and photographs, with a pre-registered acceptance rule: a paired
feature delta separating Google from BOTH photographs and foreign generators
at AUC 0.70 with a permutation null two sigma below. Separation from
photographs alone repeats the render-versus-photo result and is not
evidence. Its outcome is in `decoder-substitution.json`: zero of 124 features
clear 0.70 against both negative families (best `google_min` 0.644).

### 2026-08-19: aoxo/reverse-synthid scored against the frozen era bank

The Hugging Face dump `aoxo/reverse-synthid` was downloaded in full to
`.local-eval/synthid/reverse-synthid-hf` (2,197 files, 1,922 images) after
the public `reverse-SynthID` repository had already been audited as a
chance-level detector of our lattice, not of the mark. The question left
open was whether their solid-color and low-content images were usable as
independent copies of the Google-era signatures. `probe_aoxo_reverse_synthid.py`
reuses the shared-component residual, the origin-aligned 256-px unit-RMS
patches, and the frozen three-era matched filter. It does not call a
provider API.

Their Google-family means are the same renderer genealogy we already have.
Chroma R-G of the family mean versus `google_corpus` is 0.805 on
`gemini-3.1-flash-image-preview` (n=589, split-half 0.958), 0.905 on
`nano-banana-pro-preview` (n=518, split-half 0.938), 0.941 on `gemini_random`
(n=88), and 0.898 on `gemini_black_nb_pro` (n=91, split-half 0.964). The
same means sit at 0.48-0.56 versus `google_fresh31` and near zero versus
both OpenAI eras and photographs. A two-pixel crop on a `gemini_black_nb_pro`
2048-px black kills the match (`0.408` to `-0.063` versus corpus). The
amplified family mean is the period-8 vertical lattice. DALL-E 3 is a clean
negative for that filter: 0/280 images above the frozen foreign 10% FPR cut
(0.122), family-mean chroma 0.105 versus corpus, and the amplified mean is
unstructured noise.

Two of their folders are a different shared object and must not be mixed
into the Google bank. `gemini_black` and `gemini_white` have split-half
0.980-0.983, so they do share something, but chroma versus `google_corpus`
is 0.046 and `-0.140`. They correlate with each other at 0.527 and with the
Google-lineage aoxo families at 0.14 or below. The production folded
template accepts 0/20 in each folder.

The production lattice detector follows raster size, not folder label. It
accepts 20/20 `gemini_black_nb_pro` (2048 px, median 0.593) and 20/20
`gemini_random` (2816x1536, median 0.361), and rejects 20/20 of the leading
1024-px solids in the structured Gemini 3.1 and nano-banana folders. That is
the existing large-native scope, not a new carrier.

Adding the four matching aoxo means as extra matched-filter templates makes
the generation detector worse on the frozen holdout. Against foreign
generators, AUC falls from 0.878 to 0.857 and TPR at 10% FPR from 0.471 to
0.445 (n=535 vs 223). Against the DALL-E bank the same change drops TPR at
10% FPR from 0.991 to 0.907. Foreign and DALL-E scores rise more than Google
scores do. The frozen three-era bank is therefore left unchanged.
`shared-component/era-means-extended.npz` keeps the aoxo means as labeled
reference copies, including `aoxo_dalle3_solid` as a negative, and is not an
input to the detector. Per-family numbers live in
`aoxo-reverse-synthid/aoxo-reverse-synthid.json`.

Substitution was checked next, because addition can only raise scores.
Replacing `google_fresh31` with any aoxo Google-family mean does not beat
`google_corpus` alone. On the full in-sample holdout the three-era max is
already worse than the corpus mean by itself (full-field AUC 0.878 vs 0.892
against foreign). The leaked in-sample number is only slightly optimistic:
ten split-half re-estimates of the corpus mean, scored on the unused half,
give AUC 0.880 +/- 0.003 and TPR at 10% FPR 0.480 +/- 0.017. Chroma R-G
scoring is the one change that moves the needle on that honest split
(0.904 +/- 0.003, TPR 0.597 +/- 0.020). It is a scoring change, not a new
signature, and it is not frozen into the detector here.

The dump's actual use is as held-out positives the corpus mean never saw.
Against foreign generators, chroma correlation with the frozen corpus mean
separates `gemini_black_nb_pro` at AUC 0.990 / TPR 0.989 (n=91) and
`gemini_random` at 0.935 / 0.750 (n=88). The structured 1024-px solid
folders are the same lineage at much lower per-image SNR: `nano_banana_solid`
0.799 / 0.216 and `gemini31_solid` 0.801 / 0.153. `gemini_black` is chance
(0.594 / 0.000) and `gemini_white` is anti-aligned (0.315). Those two
folders are near-black and near-white Gemini dumps with a private shared
texture (split-half 0.98, pixel std about 3.5), not the period-8 lattice.
Numbers: `aoxo-reverse-synthid/generation-detector-aoxo-holdout.json` and
`generation-detector-aoxo-substitute.json`.

Deflation asks the remaining averaging question: after that lattice is
projected out, does a second shared component look like a mark.
Projecting a set's own mean out of itself forces split-half near `-1` and
is discarded as a tautology. The independent case is the one that counts.
Removing the frozen corpus mean from the aoxo Google families, or removing
the aoxo `gemini_black_nb_pro` mean from the corpus, leaves a leftover
chroma correlation of 0.74 between `nano_banana_solid` and
`gemini31_solid`. That leftover dies at a one-pixel roll (`0.742` to
`-0.005`) and returns at shifts 4/8/16 (`0.61` / `0.47` / `0.49`). It is
uncorrelated with photographs (`-0.01`), DALL-E solids (`0.11`), and the
OpenAI corpus (`0.09`). So rank-1 removal peels the JPEG-heavy corpus
lattice and uncovers a cleaner origin-locked period of the same renderer
family, which is why extra aoxo templates raised foreign scores instead of
finding a mark. OpenAI split-half is unchanged by Google-lattice removal
(0.723 / 0.978), as required if that component is a different renderer.
`aoxo-deflate.json` and `aoxo-deflate-leftover.json`.

### 2026-08-19: remaining local hypotheses

Four locally testable leftovers were closed by
`probe_remaining_hypotheses.py`. None of them is a mark.

**Rank deflation.** The leftover chroma between held-out `nano_banana_solid`
and `gemini31_solid` after removing the top K corpus chroma PCs stays at
0.57 from K=2 through K=16 and still dies at a one-pixel roll (0.045). It
is not in the corpus JPEG-era subspace. Fitting the PCs on `nano_banana_solid`
itself does remove it from `gemini31_solid` (0.70 at rank 1, 0.18 at rank 5,
0.12 at rank 8), so the leftover is a low-rank 3.1-family lattice the mixed
corpus does not span. Photographs stay at 0.01. Extra ranks do not uncover a
shift-robust residue.

**Mixed-assertion near-duplicates.** All five groups were opened. Three are
cross-provider re-renders at different rasters and cannot isolate a mark.
The two same-provider OpenAI pairs are same-raster: 1254 px at 29.6 dB
(RMS 8.4) and 1054x1492 at 22.1 dB (RMS 20.0). The pixel difference of
either pair is uncorrelated with the Google lattice, the OpenAI corpus mean,
and the OpenAI fresh mean (best 0.066). Both members of the 29.6 dB pair
match the OpenAI corpus fingerprint equally (0.079 vs 0.081). Assertion
metadata does not flip a shared residual, which matches the L1 finding that
most OpenAI no-assertion rows are still watermarked.

**OpenAI shared component.** The corpus-era OpenAI mean is origin-locked and
periodic: chroma self-correlation 1.0 at shift 0, `-0.446` at 1 px, then
0.57-0.68 at 2/4/8/16. Cross-era correlation with fresh `gpt-image-1` is
~0 at every shift. The fresh mean itself is origin-locked but not the same
period-8 object (self-shift 1 px `-0.083`, later shifts ~0). Both sit at
zero against photographs. This is an era-specific renderer fingerprint of
the same class as Google's, not a stable mark.

**aoxo `gemini_black` / `gemini_white`.** Their private shared object is also
origin-locked and periodic (black vs white 0.527 at 0 px, `-0.502` at 1 px,
0.51-0.53 at 2/4/8/16) and is not the Google corpus lattice (0.046 /
`-0.140`) nor `gemini_black_nb_pro` (0.139). It is a third phase-locked
texture, not a crop-robust mark.

Local averaging, deflation, near-duplicate contrast, and the OpenAI shared
component are therefore exhausted as mark routes. What remains needs a
provider oracle or a product decision: L1 distillation (OpenAI verifier),
H6/H7 (oracle-positive training pairs), freezing chroma scoring under R0 as
a generation detector, and H8 as a multi-image product mode. H3 is not a
new carrier; a GLRT on the same origin-locked lattice would rediscover S1.
Numbers: `aoxo-reverse-synthid/remaining-hypotheses.json`.

### 2026-08-19: new search class, lattice death

Hand-crafted templates and cross-image averages are exhausted because they
read origin-locked renderer structure. The mark, by publication and by our
oracle ladder, survives JPEG-95 and a few-pixel crop; the lattice does not.
The next class is therefore: destroy the lattice first, then ask whether any
statistic still separates Google from both Firefly and Microsoft.

`probe_lattice_death.py` applies JPEG-95 and an 8-24 px crop, then scores the
folded template, chroma correlation with the corpus mean, and the 124-feature
bank. The lattice dies as required: chroma AUC falls to 0.49-0.56 against
every family, and the folded template accept rate on Google falls from 0.88
to 0.04. The feature bank does not die. Best-of-124 AUC is 0.796 against
Firefly and 0.792 against Microsoft (n=149 vs 78/85). A 200-draw max-stat
null on the Firefly comparison has mean 0.600, 95th percentile 0.633, and
maximum 0.684; the observed 0.802 never occurs. Size and mean luma are not
the confound (AUC 0.51 and 0.35).

The shared winner is feature 106: median B-Y residual autocovariance at lag
(2, 3). It is a short-range opponent covariance, not a period-8 comb. On a
held-out scoring of that single feature the era gate fails. Corpus Google
median `-0.015`; fresh `gemini-3.1` `-0.011` (same side); fresh
`gemini-2.5-flash` `-0.073` (AUC 0.946 against the corpus, n=10). DALL-E 3
and fresh OpenAI sit with the other generators (0.815 and 0.796). PixelBin,
the other shared-lineage renderer, is the nearest foreign (0.663). A mark
required by policy on every Gemini era cannot miss 2.5-flash. The leftover
is therefore a crop-and-JPEG-stable renderer/era texture, a second pipeline
signature beside the lattice, not SynthID.

`lattice-death.json` and `feature-106-holdout.json`.

A linear follow-up asks whether the rest of the bank can recover the era
that feature 106 missed. Ridge on the 124 JPEG-plus-crop features, trained
on 99 corpus Google versus 164 Firefly+Microsoft rows (lambda 100 chosen
on an inner hold), reaches AUC 0.852 on the held-out 50 corpus Google
images against that foreign pool. It does not transfer: fresh 2.5-flash
0.306, fresh 3.1 0.403, DALL-E 3 0.485, fresh OpenAI 0.418. It does fire
on aoxo 3.1 solids (nano 0.962, gemini31 0.978), so what it learned is
corpus-era plus low-content 3.1 export texture, not a policy-wide mark.
`post-lattice-ridge.json`.

### 2026-08-19: L1 finished

The OpenAI control-validation runner completed the 315-row plan. Controls:
203 `detected` / 285 verdicts (contamination 0.712). Health positives: 27/28
`detected`, one miss, the same corpus-label error recorded earlier. Two
control rows remain `error` from an earlier unreachable-API window and were
not retried, because a request that may have reached the model is not a
second submission. The new key was accepted; a 429 after a burst slept 61
minutes and then drained the tail.

On provider-clean labels the frozen patch-local bank still does not separate
(`patch_local_auc` 0.572, shipped OpenAI 0.439, n=283). Distillation on the
full L1 set does move: temporal-CV ridge AUC 0.720 on the hard no-assertion
contrast (311 rows) and 0.728 pooled, with 8 and 6 univariate features
above a 2-sigma max-stat null (best 0.674 vs 0.595 +/- 0.016). In-sample
ridge permutation stays at 0.78, so the in-sample fit is still overfit; the
0.72 is the out-of-fold number. That is above the earlier 0.63 at n~110, and
still not a production expert. `openai-control-validation.json`,
`l1-snapshots.json`, `distillation-train.json`.

### 2026-08-19/20: OpenAI identity ladder

The 2026-08-17 attack ladder left crop, rotation, flip and grayscale unasked.
`identity-ladder-2026-08-19` takes the already-verified s1 positive and submits
ten frozen rungs once each. All ten returned `detected`:

| Rung | Output | PSNR (resized-back) | Oracle |
| --- | --- | --- | --- |
| JPEG q80 | 1024x1536 | 37.3 | detected |
| JPEG q20 | 1024x1536 | 31.4 | detected |
| crop 2 px top-left | 1022x1534 | 26.0 | detected |
| crop 8 px all sides | 1008x1520 | 21.0 | detected |
| center keep 75% | 768x1152 | 11.8 | detected |
| center keep 50% | 512x768 | 10.3 | detected |
| rotate 90 deg | 1536x1024 | 10.9 | detected |
| rotate 5 deg, expand | 1154x1620 | 13.6 | detected |
| horizontal flip | 1024x1536 | 10.7 | detected |
| grayscale | 1024x1536 | 26.1 | detected |

Together with the earlier 3x4 ladder (JPEG q40, 0.5x, noise+JPEG q85,
0.35x+JPEG q50, all detected, n=3) the official OpenAI mark, on this seed,
survives every identity-class transform that kills the pipeline lattice.
The two-pixel crop that zeroes the origin-locked chroma template is not a
mark failure. A half-image center crop is not a mark failure. Grayscale is
not a mark failure, so the decoder is not chroma-only. 90-degree rotation
and a flip are not mark failures, so the decoder is not phase-locked to the
original raster axes.

This is n=1 for the new rungs. It is enough to reject several detector
hypotheses and to constrain the embedding: a spatially redundant residual,
readable from luminance, trained with geometric and JPEG augmentations, not
a border, not a fixed tile, not the period-8 lattice. Local detection then
has to be a crop-equivariant learned decoder, which is the M1b ceiling, not
a better comb statistic. `identity-ladder-2026-08-19/manifest.json`.

### 2026-08-20: how the OpenAI mark is applied

The survival envelope plus the paper/patent plus the VAE pairs constrain the
embedding method. It is not a new measurement; it is the intersection of the
ones already taken.

The encoder is post-hoc and additive in pixel space: after the generator has
emitted an image, a separate network writes `x' = x + g(x)` and a separate
network reads it back. That is the paper's definition and the patent's image
example. Three oracle facts say the same thing in measurements. Metadata
stripping does not change the verdict, so the mark is not a C2PA field. A
VAE decoder substitution (`sd-vae-ft-mse`, 22.3-32.0 dB, 5/5 detected) does
not remove it, so the mark is not a latent of that decoder. Replacing the
generator would not remove the capability, which is why Google and OpenAI can
share a SynthID family name and still fail each other's official decoder.

Where it is written. Grayscale (PIL `L`, then RGB) stayed `detected`, so the
decoder reads luminance: stacking `L,L,L` zeroes chroma, and a chroma-only
carrier would have gone silent. JPEG q20 stayed `detected`, so the carrier
is not the high-frequency DCT tail that classical DWT-DCT uses. A center
crop keeping half the area stayed `detected`, so the residual is spatially
redundant rather than a border or a single tile. A two-pixel crop stayed
`detected` while our origin-locked lattice dies, so the written residual is
not that lattice. 90-degree rotation and a horizontal flip stayed `detected`,
so `g` is not a phase-locked overlay on the original raster axes.

How `g` is shaped. M1b at matched RMS put a fixed mid-band mark at AUC 0.670,
a keyed mid-band mark at 0.669, and a keyed perceptually masked mark at 0.546.
Masking, not keying, is what hides it from hand statistics: amplitude tracks
local texture, local SNR is flat, and there is no quiet region to average.
The patent's U-Net with optional key injection is the architecture that
produces exactly that residual. Content dependence (`g(x)` rather than a
fixed `g`) is why averaging thousands of OpenAI images never yielded a mark
template, only an era-specific renderer fingerprint.

The working model of application is therefore: a provider-specific neural
encoder, run after generation, adds a perceptually masked luminance residual
in the mid-band, redundant across the frame, trained to survive JPEG, crop,
resize, flip and small/90-degree rotation. The matching decoder holds the
provider key as weights. Nothing in the local pixel statistics we can write
by hand is that decoder.

### 2026-08-20: census of embedding methods

Every plausible way to write a SynthID-class mark, sorted by what the
measurements already forbid. Identity-ladder rungs (crop, rotation, flip,
grayscale, JPEG q20) are n=1 on OpenAI s1. Attack-ladder and VAE pairs are
n=3 and n=5. Google VAE is n=3. A method that survives those tests is not
proved; a method that dies on them is closed.

**Ruled out**

| Method | Why it is closed |
| --- | --- |
| C2PA, EXIF, filename, chunks, or any sidecar | Official OpenAI path strips AI metadata, proves pixels unchanged, still `detected` |
| LSB / bit-plane steganography | JPEG q20 and `sd-vae-ft-mse` both destroy LSBs; both left the mark `detected` |
| Classical DWT-DCT / `imwatermark` bins | That family dies near JPEG q90; SynthID stayed `detected` at q20 and at 0.35x+q50 |
| Chroma-only residual | Grayscale zeroes R-G and B-Y; oracle still `detected` |
| High-frequency-only tail | JPEG q20 and VAE keep mid-band and discard the tail; mark remained |
| Border, corner, or single-tile stamp | Center keep-50% still `detected` |
| Origin-locked periodic lattice (our S4/S1 object) | 2 px crop kills that lattice and not the oracle; Firefly shares the lattice and has no SynthID |
| Axis-locked overlay on the original raster | 90 deg rotation and horizontal flip still `detected` |
| Mark stored only in one generator's VAE latent | Decoder substitution through `sd-vae-ft-mse` left 5/5 OpenAI and 3/3 Google `detected` |
| In-model watermark baked into diffusion weights | Paper: post-hoc `f: X -> X` on an already generated image; swapping the generator does not remove the capability |
| One shared decoder for Google and OpenAI | OpenAI verifier returned `not_detected` on a Google positive and `detected` on OpenAI positives |
| Fixed additive template shared across images | OpenAI corpus vs fresh split-half cross-era ~0; averaging recovered renderer genealogy, not a mark |
| Keying alone as the reason hand features fail | M1b: fixed vs keyed mid-band 0.670 vs 0.669; masked keyed 0.546 |

**Still possible, not forced**

| Method | What would still fit |
| --- | --- |
| Additive `x' = x + g(x)` vs a learned invertible map | Additive is the patent example; a small invertible network is not ruled out if it is JPEG-and-crop stable |
| Luma-only vs luma-plus-chroma | Luma is required; extra chroma is allowed because grayscale kept the luma copy |
| Mid-band spatial residual vs mid-band learned latent of the watermarker itself | VAE test only kills *that* VAE; the encoder may still think in its own latent |
| U-Net vs other conv/attention generators | Patent describes U-Net; any local-and-global residual net that masks to texture would survive the same tests |
| Resize to a trained size, mark, resize back | Patent alternative; compatible with scale survival; decoder would canonicalize or be scale-trained |
| Ensemble of encoder/decoder pairs per provider or epoch | Patent continuation; explains Google vs OpenAI silence and era-specific keys without changing the method class |
| Per-image payload / secret injected into `g` | Presence detection does not need payload recovery; optional in the patent |
| Training augmentations beyond the ones we hit | Elastic warp, extreme downscale, JPEG q5, 10% area crop are untested; they may be outside the trained set |
| Slightly different Google vs OpenAI implementations of the same class | Same method, different weights, possibly different trained-size or mask strength |

**Forced by the intersection**

| Property | Forced by |
| --- | --- |
| Written in decoded pixels, after generation | Metadata strip; paper definition |
| Readable from luminance | Grayscale `detected` |
| Spatially redundant across the frame | Keep-50% `detected` |
| Not phase-locked to the source origin or axes | 2 px crop, rot90, flip `detected` |
| Mid-band, JPEG-stable | q20 and 0.35x+q50 `detected` |
| Provider-specific decoder weights | Cross-provider `not_detected` |
| Content-dependent residual, perceptually masked | No shared template under averaging; M1b mask gap |

The method class is therefore one: a provider-keyed neural residual in luma
mid-band, post-hoc, redundant, geometry-trained. Variants inside that class
(U-Net vs other nets, optional payload, resize-to-canonical, ensemble of
pairs) remain open and do not change the detector architecture. Everything
outside that class is closed.

### 2026-08-20: OpenAI kill ladder, first not_detected

Identity and attack ladders never silenced the official decoder. The kill
ladder on the same s1 seed asks which frozen destruction first does.

| Rung | Output | PSNR | Oracle |
| --- | --- | --- | --- |
| JPEG q10 | 1024x1536 | 29.1 | detected |
| JPEG q5 | 1024x1536 | 26.3 | detected |
| center keep 25% | 256x384 | 9.9 | not_detected |
| center keep 10% | 102x153 | 10.5 | not_detected |
| Gaussian blur sigma 3 | 1024x1536 | 26.9 | detected |
| Gaussian blur sigma 8 | 1024x1536 | 23.3 | not_detected |
| scale 0.20 | 204x307 | 27.0 | detected |
| median 7 | 1024x1536 | 28.6 | detected |
| elastic alpha 12 | 1024x1536 | 40.7 | detected |
| posterize 4 bits | 1024x1536 | 29.5 | detected |

The flip is not pixel count. Scale 0.20 has fewer pixels than keep-25% and
stayed `detected`, because it still holds the whole frame. Keep-50% was
`detected` on the identity ladder; keep-25% is the first crop miss. Blur
sigma 8 is the first full-frame miss; JPEG q5 is not. Mild elastic (40.7 dB)
is not reverse-SynthID's 18-24 dB warp and did not flip.

So the official decoder needs a large fraction of the original scene and is
more sensitive to spatial low-pass than to DCT quantization. That is not a
remover: blur-8 at 23 dB and a quarter-frame crop destroy the picture. It is
the first oracle-negative envelope on OpenAI. n=1 seed.
`kill-ladder-2026-08-20/manifest.json`.

The in-between batch `flip-ladder-2026-08-20` on the same seed:

| Rung | Output | PSNR | Oracle |
| --- | --- | --- | --- |
| blur sigma 5 | 1024x1536 | 25.1 | detected |
| blur sigma 6 | 1024x1536 | 24.4 | detected |
| blur sigma 7 | 1024x1536 | 23.8 | not_detected |
| center keep 40% linear | 409x614 | 10.5 | not_detected |
| center keep 33% linear | 337x506 | 10.1 | not_detected |
| elastic ~24.6 dB | 1024x1536 | 24.6 | detected |
| elastic ~20.8 dB | 1024x1536 | 20.8 | not_detected |

Flip points, still n=1: Gaussian blur between 6 and 7 (24.4 dB vs 23.8 dB);
crop between keep-50% linear (detected on the identity ladder, 512x768) and
keep-40% (409x614); elastic between 24.6 dB and 20.8 dB. JPEG q5 remains
`detected`. No rung that silences the decoder is a usable picture. `flip-ladder-2026-08-20/manifest.json`.

### 2026-08-20: OpenAI add ladder

The crop miss could have been "too few marked pixels in the upload" or
"too little of the original scene". Adding unmarked pixels distinguishes
them. Same s1 seed, donor COCO `000000000139`.

| Rung | Upload | Oracle |
| --- | --- | --- |
| pad white, original 70% linear | 1463x2194 | detected |
| pad white, original 40% linear | 2560x3840 | detected |
| inset on photo, 70% of 1600 canvas | 1600x1600 | not_detected |
| inset on photo, 40% of 1600 canvas | 1600x1600 | not_detected |
| blend 50% with photo | 1024x1536 | detected |
| blend 25% marked / 75% photo | 1024x1536 | detected |
| horizontal concat with photo | 2048x1536 | not_detected |
| Gaussian noise sigma 16 | 1024x1536 | detected |

White padding to 40% linear stays `detected`, while cropping the same seed
to 40% linear was `not_detected`. The decoder is not counting marked pixels
in the upload; it is reading coverage of the original scene. Uniform pad is
ignored. A second unmarked photo beside the intact marked half (`hstack`)
silences it, as does pasting a resized copy onto a photo (`inset`, with an
aspect-ratio confound). Mixing unmarked pixels into every location at 75%
still `detected`, so amplitude 0.25 of the residual is enough when it is
present everywhere. n=1. `add-ladder-2026-08-20/manifest.json`.

All OpenAI oracle ladder numbers are copied into tracked
[`docs/synthid-removal-research.md`](synthid-removal-research.md) so they survive
the gitignored eval directory.

Replication on s2 and s3 of the s1 flip points is complete. Blur sigma 6 vs 7
and crop keep 0.50 vs 0.40 hold on 3/3 seeds. White pad 40% stays `detected`
and hstack stays `not_detected` on 3/3. Blend 25% stays `detected` on 3/3.
Native-aspect inset onto a 2x photo canvas is `not_detected` on 3/3 (s1 added).
Elastic at ~21 dB does not generalize: s3 remains `detected` at 22.0 dB while
s1 and s2 miss. The durable crop and blur flips are therefore the ones to
keep; elastic PSNR is not a stable kill threshold.

Surround ladder on s1: photo fill at the white-pad geometries is
`not_detected` at both 70% and 40% linear, so extra canvas only kills when
the pad is a photograph. A side strip of unmarked photo stays `detected` at
10% and 25% of width and becomes `not_detected` at 35% and 40%. The decoder
treats a large unmarked scene in the same upload as a whole-frame negative,
while uniform white is ignored. Numbers in
[`docs/synthid-removal-research.md`](synthid-removal-research.md).

Photo-pad vs white-pad holds on 3/3. The hstack unmarked 25% vs 35% split
holds on 3/3. A photo frame around s1 stays `detected` at 5/8/10/12% each
side and becomes `not_detected` at 15%. Frame 15% holds on s2 and s3.
Frame 12% does not: s2 is already `not_detected`.

### 2026-08-20: close the local OpenAI mark hunt

Decision: stop searching for a local OpenAI SynthID detector or a
quality-preserving pixel remover on this evidence. The official decoder's
flip points are now replicated (blur 6 vs 7, crop keep 0.50 vs 0.40, photo
surround, hstack 25 vs 35, JPEG q5 still detected, all 3/3 except elastic
PSNR and frame 12%). Every `not_detected` that is not a collage is below
usable quality. Distillation on L1 labels remains 0.72 temporal-CV against a
0.78 in-sample permutation, so it is not an expert.

What the product already ships is the honest remainder: signed provenance for
the watermark, and `detect-synthid` / `identify`'s `pipeline_lattice` for the
origin-locked generation lattice, labeled as not SynthID. The unshipped
chroma matched-filter (AUC about 0.90 vs foreign on the frozen holdout) is a
second generation-family candidate and may only enter runtime under the R0
gate, never as a watermark claim.

Oracle numbers: [`docs/synthid-detector-research.md`](synthid-detector-research.md)
and [`docs/synthid-removal-research.md`](synthid-removal-research.md).

Open questions that are not a repeat of closed hunts, now that the official
OpenAI decoder can be queried:

1. Self-concat. `hstack` of marked|unmarked is `not_detected` on 3/3.
   `hstack` of marked|marked (same seed, two copies) is untested. If that
   stays `detected`, the kill is a second scene, not extra canvas. If it
   misses, any two-panel layout drops the score.
2. Surrogate. The ladders plus L1 are a small labeled set of (image,
   transform, official verdict). A student that copies those verdicts would
   let attack search run offline. It is not a mark decoder until it beats
   the 0.72 L1 temporal-CV and the 0.78 in-sample permutation on held-out
   seeds.
3. Luma-only blur between sigma 6 and 7 at PSNR above 30 dB. Closed
   2026-08-22 as a quiet-remover class together with Bayer, VNG, barrel, and
   scanline: none flipped the OpenAI oracle below scramble cost. The only
   full-frame residual kill remains ~24 dB.
4. The chroma matched-filter remains a generation-family candidate under R0,
   never a watermark claim.

Item 1 is the cheapest untested discriminator. Item 2 is the only path to a
local stand-in for the official decoder. Item 3 is the last quality-removal
check worth an oracle burst.

### 2026-08-20: cardinal directions, including the text analog

SynthID-Text is not a recipe for the image encoder. Text is biased at
sampling time among near-synonyms; the published image system is a post-hoc
map `f: X -> X`. The useful borrow is the *detector* shape: a keyed weak
statistic pooled over many tokens, so you need a long enough span and
unmarked tokens vote against. That is compatible with what the OpenAI image
oracle already did (keep-0.40 misses, whole-frame 0.20x hits, white pad
ignored, photo surround votes no). It does not revive averaging or a fixed
tile.

Directions that are actually untested, not a new comb:

- Token analog. Submit the four quadrants of a known-positive separately.
  4/4 `detected` means a locally redundant residual. 0/4 `detected` while
  the full frame hits means a global pool. Mixed means content-dependent
  support. Cheap, four uploads, not a transform we have run.
- Layout scramble. Cut the same positive into a grid of tiles, permute,
  reassemble, upload. Local-and-pool marks can survive; layout-locked or
  object-aligned `g(x)` should drop. Orthogonal to blur and crop.
- marked|marked concat. Distinguishes "second scene" from "any two-panel
  layout".
- Self-keyed residual. Text hashes the prefix; an image analog hashes a
  thumbnail and chips the residual. Averaging across images then must fail,
  which it did. The untested check is whether an image's high-pass residual
  is a stable function of its own low-pass thumbnail across seeds. A yes
  would be a local handle without the provider key. A no leaves only the
  official decoder.
- Surrogate of the oracle, not of the mark. Train on the labeled ladder plus
  L1. Risk: it learns collage/blur/crop, not SynthID. Gate it on held-out
  seeds and on JPEG q5 still calling `detected`.

Tree-Ring-style latent Fourier marks are already a poor fit: VAE
substitution left the OpenAI mark up. Do not reopen band templates, aoxo, or
elastic PSNR as a threshold.

Token-ladder on s1, 2026-08-20: all four quadrants `detected` at 512x768.
The mark is not a single global code that needs the whole frame; a corner
half is enough, matching keep-0.50. `hstack` and `vstack` of two copies of
the same marked image are both `not_detected`. Photo surround was therefore
misread: the decoder also fails when both halves are marked. A two-panel
layout is the kill, not unmarked pixels. 4x4 scramble stays `detected`;
8x8 scramble misses, so the local support is larger than ~128 px tiles and
can live in ~256 px tiles. Together with scale-0.20 still detecting the
full scene, the official decoder looks like a pool over photograph-shaped
regions of sufficient span, plus a gate that rejects obvious collages.

An independent Claude Code pass (2026-08-20, tools disabled) argued the
collage gate may be an artifact of **canonical-canvas resampling**: the
decoder resizes the whole upload, so a 0.20x full scene is restored, while
a photo-pad or two-panel layout leaves the mark at half canonical scale.
That one mechanism would reclassify most collage `not_detected` as
preprocessing, leaving blur-7 as the only confirmed residual kill. Cheapest
falsifier: submit 0.20x as-is (known detected) versus the same 0.20x pixels
centered on a white canvas of the original size. Identical marked pixels,
only canvas scale changes. If the padded copy misses, scale-on-canvas is
real. If it still detects, the collage/photo-content gate stands.
A second independent pass the same day used Claude Code Opus with full tools
and Codex `gpt-5.6-sol` at high reasoning. Full texts:
`.local-eval/synthid/prc-oklab-attack-2026-08-15/agent-claude-plan.md` and
`agent-codex-plan.md`.

Merged ordered plan from those two independent reviews, OpenAI first:

0. Repair L1 evaluation (0 calls). Codex: the 0.72 is leave-one-date-out on
   ~4-image folds, includes health positives, and the 0.78 permutation is
   in-sample. Forward-temporal control-only reanalysis is about 0.64-0.65.
   Use that as the baseline, not 0.72 vs 0.78.
1. The 0.20x-on-native-white discriminator was run anyway on s1. The 0.20x
   file alone was `detected`; the same pixels on a 1024x1536 white canvas are
   `not_detected`. Stretch 2x width is `detected`. Centre-crop of marked|marked
   back to native size is `detected`. Pure aspect and pure duplication at
   native size are not the hstack kill. Scale of the marked region on the
   uploaded canvas is real. It does not replace pad_photo vs pad_white, which
   still shows unmarked photographic content as a second mechanism.
2. L1 repair: control-only forward-temporal nested ridge is 0.649 / 0.641,
   not 0.72. June-15 cut beats a 40-run nested permutation at p=0; June-30
   cut p=0.075.
3. E3 leave-one-encoder-out on 80 COCO images: ridge and a 4-layer conv both
   fail to transfer to TrustMark (AUC 0.51 / 0.51). Ridge transfers only to
   dwtDct at 0.675. After VAE laundry of both classes: all three held-out
   families are chance or worse (0.41 / 0.49 / 0.47). The 0.65 gate fails.
   A local blind detector of this mark class is not learnable here at this
   capacity and data.

Independent debate 2026-08-20 (Opus full tools vs Codex gpt-5.6-sol high):
Codex votes CLOSED for the whole local-blind program. Opus splits: blind
transfer CLOSED, Google CLOSED, OpenAI OPEN by one experiment (E4: graded
phase rotation in 16-32 px, stay open only if some theta < pi kills the
oracle while a keyless paired AUC on that perturbation stays below 0.75).
Both texts: `agent-debate-claude.md`, `agent-debate-codex.md` under the eval
directory. Third-voice merge: honor E3 as closing surrogate-encoder transfer;
do not reopen templates. The only remaining OpenAI experiment that is not
shopping is a *short* E4: three seeds times theta in {pi/4, pi/8} in the
16-32 annulus, with a matched 64-128 control at the same theta. If nothing
below full scramble flips 2/3 seeds, close. If it flips, the pair layer is
real and D4 may be attempted once. That is 12 calls, not 57.

Unbounded open pass 2026-08-20 (Opus and Codex gpt-5.6-sol high, full tools,
local measurements, no oracle). Texts: `agent-open-claude.md`,
`agent-open-codex.md`.

Opus measured TrustMark mark/content power in the carrier octave at -33.9 dB
and showed a keyless energy detector's AUC is theoretically ~0.51 at s1 size
and ~0.54 at 4K; campaign keyless numbers (0.53-0.57) sit on that ceiling.
E4 pair-by-band-destruction is then dominated by the attack (+24 dB over the
mark at the gentlest killing theta). Micro-warp desync failed. An external
HF "gpt-image-synthid-detector" scores 0.50 vs COCO (positive on 80% of
photos) but 0.69 on L1 official labels; the 0.70 plateau is real and unnamed.

Codex measured that an 11-number geometry vector (size, aspect, modulo 64)
gets forward-temporal AUC 0.79-0.80 on the same 283 L1 controls where pixel
ridge is 0.64. Raster geometry currently beats every watermark representation
as a predictor of the official verdict. TrustMark-P analog: survives 0.20x
12/12, dies on crop and scramble; its decoder center-crops to 224, so lethal
octaves can be decoder coordinates.

Merged next action if the local hunt continues at all: zero-API,
geometry-conditioned L1 (Codex) plus naming the 0.70 (Opus). Do not run E4.
Do not train a CNN on raw L1 bits.

Ran 2026-08-20. ChatGPT rows are all `not_detected` (21/21). After dropping
them, geometry still beats pixels (0.72/0.78 vs 0.63/0.67). Largest mixed
API size 1254x1254 is chance (p=0.45). Self-keyed 16-32 map is 0.41 vs L1-neg
and 0.50 vs COCO. L1 official bits remain unidentifiable as a mark task on
this corpus. Do not train a CNN on them.

Detector debate 2026-08-21 after `gpt-image-2` pairs (Opus, Codex gpt-5.6-sol
high, this session). Notes: `agent-detector-claude.md`, `agent-detector-codex.md`.
Opus: photo matched-filter ceiling from the flat 16-32 amplitude (0.11 RMS) vs
in-band photo content (12.2 RMS) is d'~0.93 / AUC~0.75 even with a perfect
template; P5 needs d'>=4.37, a 13.4 dB deficit *unless* the photo-domain mark is
louder than the flat measurement. Next: 16-32 jamming titration on the fish edit
and the gray-128 flat (~14 oracle calls) to measure that amplitude. Codex: do not
reopen keyless search; train a frozen wavelet CNN only on procedural
flat-to-texture pairs, lock >=128 photos, require paired AUC LCB>0.65 plus a
five-file causal panel. Merge: titration first. A student cannot beat a matched
filter if the amplitude is as quiet as the flats; minting 128 photos before that
number is architecture shopping. Retract L1 matched-filter AUC 0.701 as size
confound.

Titration ran 2026-08-21. Gray arm is not a coding-gain calibrator (any extra
energy, including 4-8 px, is `not_detected`). Fish 16-32 additive jam stays
`detected` through a=16 / 21.9 dB and flips at a=24 / 18.6 dB; 4-8 px at a=16
stays `detected`. 4-8 px jam at a=24 / 18.6 dB stays `detected`, so the 16-32 flip is
band-specific. Phase scramble of 16-32 remains the cheaper kill at 24.3 dB.
The close `a*_photo < 0.53` failed. Do not mint a 128-photo student on that
number. Do not treat additive in-band noise as a photo remover.

### 2026-08-21: source-matched pairs and titration

Full numeric log: [`docs/synthid-detector-research.md`](synthid-detector-research.md)
and [`docs/synthid-removal-research.md`](synthid-removal-research.md).

`gpt-image-2` can mint encoder-on/off pairs. Flats are clean (PSNR 38-49 dB,
oracle `not_detected` to `detected`) and share a highpass residual (LOO 0.645)
that does not transfer to photographs. The 16-32 carrier is causal on both
flats (scramble 51 dB `not_detected`) and the fish photo (scramble 24.3 dB
`not_detected`; pyramid wipe 48 dB still `detected`; additive jam flips only
at 18.6 dB, while equal-energy 4-8 px jam stays `detected`).

That does not open a local photo detector: the flat stamp is not the photo
mark, L1 bits remain a geometry/export task, and the photo jam amplitude never
fell below the preregistered 0.53 RMS close. Product remainder unchanged.

Future analog track, not a current experiment: a self-owned encoder in the
16-32 px midband (thumbnail-keyed phase, JND mask) as a research stand, never
as a Gemini/OpenAI detector. Future evasion track, only with oracle quota: a
graded phase shift in 16-32 at PSNR > 30 dB on s1-s3. Collage, photo-pad, and
small-on-canvas stay presentation bypasses, not removers.

### 2026-08-22: Model 1 result, provider CLIP, lattice re-check, Bayer close

Numeric log: [`docs/ai-generated-image-classifiers.md`](ai-generated-image-classifiers.md),
[`docs/synthid-classifiers.md`](synthid-classifiers.md), and
[`docs/synthid-removal-research.md`](synthid-removal-research.md).

Model 1 is the AI-versus-camera research result, not a watermark detector.
Finetuned CLIP-L, 224 letterbox, last two vision blocks: Kodak 0/24, locked
Open Images fresh FPR 1.7%, AI-test TPR 93.0% (OpenAI 93.2, Gemini 90.5,
FLUX hold 92.7). False positives on Open Images are graphics, CGI, and
cutouts, not Gemini. It does not enter `identify`. CLIP-H and DINOv2-giant
are not the result.

Provider CLIP (`openai` / `google` / `other`) and an OpenAI-plus-Gemini
union fail Firefly. The 124-d residual bank remains the Google-lineage
lattice, not an AI-or-not feature set.

Production `pipeline_lattice` re-check (n=628): Google 56%, Firefly 18%,
PixelBin 14%, OpenAI 1%, photographs 0. A 2 px crop kills the positives.
Name stays generation-pipeline lattice.

Bayer bilinear, Bayer VNG, upscale-then-Bayer, barrel, and scanline left
OpenAI SynthID `detected` on s1 and s2. 16-32 phase scramble still flipped
both. VNG on s2 is dirtier than scramble and the mark remains. Camera
pipeline and mild geometry are closed as quiet removers. s3 and the fish
photo were not submitted in this pass (OpenAI 429).
4. E2 per-octave phase scramble on s1, replicated p16_32 on s2 and s3:
   destroying periods 16-32 px is `not_detected` 3/3; 2-4, 4-8, 8-16, 32-64,
   64-128 stay `detected`. The carrier octave is 16-32 px, not the lattice
   band. Keyless energy and RGB phase-agreement in that octave on L1 are
   AUC 0.53 and 0.57.
5. Google gets no independent hunt. If (3) transfers, validate on a small
   Gemini-app batch against Firefly/PixelBin. If (3) fails, say so.

Working decoder hypothesis, not proved: a net trained on single whole
photographs, probably after a canonical resize. That one story fits JPEG q5
(still one photo), scale 0.20 (same scene), white letterbox (still one
photo), keep-0.50 vs 0.40 (enough vs too-tight crop of the scene),
photo-pad and any two-panel concat including marked|marked (not a single
photo), scramble 4x4 (still reads as one image) vs 8x8 (a mosaic). It is a
hypothesis about the *decoder*, not a new encoder. Falsify it with a
concat that still looks like one photograph, or a mosaic that still looks
like one.

Ontology split, forced by marked|marked missing while every quadrant of the
same seed detects: an official `not_detected` is not one event. Collage,
photo-pad, two-panel self-concat, and 8x8 scramble can be a *presentation
gate* (this upload is not a single photograph). Blur sigma 7, keep-0.40, and
JPEG-surviving full-frame views sit on the single-photo manifold and are the
only current candidates for actually damaging the residual. Crop-keep-0.40
may be either: too little scene for the residual net, or a zoom the
preprocessor treats as a different image. Do not train a surrogate on mixed
collage misses if the goal is a mark student; those labels are the gate.

## Production hardening roadmap

The next milestone is a precision-first production detector system, not one
unqualified binary classifier. "Bulletproof" means that every positive verdict
is tied to a frozen, independently challenged expert; every negative verdict is
limited to a sensitivity-certified scope; and every other case is an explicit
abstention. It does not mean claiming that an undocumented watermark can be
recognized after every possible edit or future encoder change.

### Current starting point

| Route | Evidence already obtained | Production gap |
| --- | --- | --- |
| Gemini fixed native, 1-10 MP | Frozen period-16 template; 0/5,000 observed-geometry COCO controls and 0/5,000 generated-geometry controls | Too few matching-oracle positives to estimate recall by geometry, epoch, and product surface |
| Gemini large native, 10-18 MP | 7/7 official pixel positives; 0/2,637 feature-unseen controls; 0/41 post-freeze Open Images controls | Positive sample is too small; JPEG-95 and JPEG-90 each reduce 7/7 to 0/7; the fresh negative acquisition is incomplete |
| Gemini registered cascade, 0.25-10 MP | Registered-v3 plus bounded opponent fallback; the fallback recovered 49/49 lossless 0.5x-0.75x views from seven official positives, rejected 1,790 measured period-8 codec crossings, and retained the earlier 0/1,000 post-freeze control result | Provenance labels are not enough for a provider-wide recall claim; lossy retranscodes are rejected; additional encoder epochs remain abstentions |
| OpenAI official remote | Metadata-free, pixel-identical web smoke: 2/2 OpenAI positives detected, while one Google positive and one COCO control returned not detected; one preselected OpenAI-positive PNG also returned SynthID detected through the credentialed SDK path; mocked transport and parser tests pass | JPEG, WebP, approved negative transport cases, operational limits, privacy behavior, and live error paths are not yet certified |
| OpenAI local | Period-8, DTCWT, codeword, external-decoder, and generic-watermark transfer branches all failed their locked or fresh checks | No local route may ship without authorized pixel-oracle labels or source-matched counterfactual pairs |

These rows are separate claims. The OpenAI official verifier must never be used
as a fallback label for Gemini, and the local Gemini carrier must not be
reported as OpenAI or even as Google-exclusive until provider exclusivity is
measured against other SynthID deployments.

### P0: freeze the product contract before collecting more scores

Introduce one provider-neutral result envelope around the existing backends:

- `status`: `detected`, `not_detected`, `unsupported`, `indeterminate`, or
  `error`;
- `signal_family`, `provider_scope`, `backend`, `detector_version`, and
  `calibration_version`;
- decoded geometry, input format, score, threshold, and the exact support reason;
- `pixels_preserved` and `metadata_used_for_verdict` audit fields;
- a calibration-manifest digest and an optional decoded-pixel audit digest that
  stays in the local qualification manifest and is never emitted in normal user
  output or centralized telemetry;
- explicit caveats for lossy input, spatial resampling, unknown encoder epoch,
  and remote upload.

At the product boundary, a miss from the current positive-only Gemini experts
maps to `indeterminate`, not to a clean-image claim. `not_detected` is reserved
for an official provider verdict or a future local scope whose sensitivity gate
has passed. `unsupported`, remote failure, malformed response, quota exhaustion,
and verifier refusal remain distinct states and never collapse into a negative.

The expert registry owns support predicates and precedence. Fixed, registered,
large, and future codec experts keep separate identities and calibration
artifacts. A router may select one predeclared expert from input scope, but may
not OR together overlapping scores or relabel the default expert after seeing
its value. Offline commands never invoke a remote verifier implicitly.

The first P0 runtime slice is now implemented. Local results return
`detected`, `indeterminate`, or `unsupported`, include an explicit support or
miss reason, and expose `signal_family`, `provider_scope`, `backend`,
`metadata_used_for_verdict`, and `pixels_preserved`. The official OpenAI result
uses the same audit fields while retaining the provider's documented
`detected`/`not_detected` outcomes. Calibration-manifest digests and a unified
error envelope remain qualification work rather than being synthesized from
the historical research files.

### P1: build the immutable qualification corpus

All qualification media stay outside the public repository. The tracked
artifacts are schemas, hashes, split manifests, aggregate reports, and cleared
synthetic fixtures.

For Gemini, collect matching-oracle labels before any new threshold fitting:

- at least 200 decoded-pixel-unique native positives for the fixed route,
  balanced across five pixel-count bins, landscape, portrait, and square
  geometry, at least three collection windows, and every available Gemini image
  surface;
- at least 100 official native positives for the large route, balanced across
  10-12, 12-15, and 15-18 MP, the three aspect families, and at least three
  collection windows;
- at least 200 source-disjoint transformed positives before reconsidering the
  registered route, with the untouched parent verified first and every
  derivative grouped with that parent;
- at least 10,000 freshly frozen decoded-pixel-unique negatives per claimed
  local route. Each route needs cameras, scans, screenshots, flat graphics,
  synthetic textures, other generators, other watermark families, resampling
  aliases, and same-provider oracle-negative examples. No source family may
  dominate the aggregate result.

Complete the pending 3,000-image post-freeze Open Images large challenge rather
than treating the current 41 completed files as a final estimate. Deduplicate
every item against all prior train, development, threshold, and challenge
corpora by decoded hash and perceptual group.

For OpenAI, prepare a preregistered transport-qualification set with previously
established ordinary-use verdicts across PNG, JPEG, WebP, portrait, landscape,
square, alpha, maximum practical dimensions, and multiple model/date windows.
Run it through the SDK only within the published usage policy and the account's
authority. If the required batch volume or retention terms cannot be approved,
the remote backend remains opt-in preview rather than being called fully
qualified. These cases validate transport and response semantics; they are not
training labels for a local detector.

Every split is frozen by parent group before scoring. Mutation tests must prove
that swapped providers, changed bytes, duplicate pixels, derivative leakage,
indeterminate-as-negative conversion, and a wrong expert identity invalidate
the run.

### P2: finish the Gemini local expert bank

Rebaseline the fixed and large routes only on oracle-confirmed positives and
fresh negatives. Do not tune the existing holdouts again. Each failed gate
creates a new detector version and a new untouched test partition rather than a
patch to the same test set.

The all-resolution program is a registered bank of measured carrier periods,
not an assumption that arbitrary dimensions imply arbitrary scale. Test these
branches in order:

1. native period-16 fixed and large experts, with no threshold changes unless
   the new locked corpus rejects them;
2. integer-period experts for periods 8, 10, 12, 14, 18, 20, and 24, each with
   its own search-corrected null calibration;
3. a fractional-period expert that estimates period on one image region and
   scores a disjoint region, preventing the same noise peak from selecting and
   proving itself;
4. codec-specific experts trained and evaluated symmetrically on JPEG and WebP
   pixels, never by requiring a native file to survive a second lossy encoding;
5. crop and screenshot experts only if phase registration and boundary search
   pass a fresh multiple-search challenge.

Candidate features remain restricted to watermark-mechanism evidence:
registered residual tiles, signed opponent-color phase, cross-region phase
agreement, circular phase statistics, and directional wavelet residuals.
Scene semantics, thumbnails, filenames, container size, metadata, and provider
logos are forbidden. A learned model may be introduced only after a classical
expert passes the same source-matched gate, and it must consume high-pass
residuals rather than RGB content.

For each new expert, fit on train, choose features on validation, calibrate on a
separate null partition, and evaluate once on locked and later temporal tests.
Report per-parent paired differences and confirm claimed improvements with a
sign test. A larger aggregate AUC does not compensate for a failed hard-negative
stratum.

### Ranked detector hypothesis backlog

The evidence gap blocks threshold promotion, not mechanism research. The
following hypotheses are materially different from the already rejected extra
color spaces, raw FFT magnitude, standalone SWT/DTCWT, multifractal summaries,
folded variance, spatial lag products, and provider-versus-control CNNs. They
are ranked by expected information gain and must still follow the P1 split
discipline.

Execution state as of the 2026-08-16 audit (the entries below keep their
original wording as the research designs; this block records what happened):

| # | State | Outcome |
| --- | --- | --- |
| H1 | Executed 2026-08-14 | Split-confirm lattice synchronization works: 20/20 transformed positive views at exact expected periods, 0/800 locked negative views. But the lattice it locks onto is the origin-anchored generation-pipeline lattice refuted as the watermark under S1/R0, and the broad non-Google challenge accepted 3/276. Closed as a watermark-detection route; remains the mechanism baseline |
| H2 | Rejected 2026-08-14 | The complex spectral-correlation statistic did not beat its own phase-preserving mutation control; rejected without consuming another locked scale partition |
| H3 | Partially executed | Split-confirm selection and the content-matched same-image null ran inside the H1/H4/H5 probes; the full GLRT with empirical p-values folded into the conformal cascade is still open |
| H4 | Rejected as a standalone detector 2026-08-14 | Content-whitened matched filter added no boundary over the phase-preserving spatial template; survives only as an H1 synchronization refinement, calibrated inside that branch |
| H5 | Advanced as research confirmation | The unknown-payload patch-agreement statistic retained positive margins under JPEG and crop; explicitly not a runtime threshold, and it inherits the S1 scope question because it reads the same periodic structure |
| H6 | Blocked | Requires the matching-oracle positive corpus (P1 prerequisites, P3 boundary for OpenAI); not started |
| H7 | Blocked | Same corpus dependency as H6; not started |
| H8 | Not started | A separate batch product mode, not a substitute for the single-image detector |

#### H1: recover the complete reciprocal lattice

The current scale registration estimates mainly one carrier period. A resize,
rotation, shear, screenshot, or mild perspective warp transforms a two-
dimensional reciprocal lattice, not one scalar period. Fit two lattice basis
vectors from harmonic peak pairs with a robust consensus estimator, then
demodulate or resample the residual into canonical carrier coordinates.

Estimate the lattice on one fixed checkerboard of spatial blocks and score the
held-out blocks. This prevents content peaks used for synchronization from also
proving detection. Compare translation-only, isotropic-scale, affine, and mild
projective models in that order, charging every model and harmonic candidate to
the same maximum-statistic calibration.

Gate: improve paired sensitivity on resized, rotated, and screenshot positives
without raising the frozen hard-negative upper bound. If affine registration
selects natural resize lattices as often as watermark lattices, reject the
branch rather than adding a semantic veto.

#### H2: complex cross-spectral cyclostationary coherence

The rejected OpenAI cyclostationary screens measured folded magnitude,
periodic variance, and spatial lag products. They did not test the complex
spectral correlation density
`F(k + alpha/2) * conjugate(F(k - alpha/2))` across disjoint spatial blocks.
A spread-spectrum periodic modulation can couple neighboring frequencies even
when payload signs cancel the cross-image mean.

Measure normalized complex coherence only at preregistered cyclic frequencies
and compare it with same-radius off-carrier frequency pairs. Preserve phase and
cross-channel covariance instead of averaging magnitudes. Fit frequencies on
train, choose one statistic on validation, and confirm on locked images and
fresh same-provider controls.

Gate: the carrier-to-off-carrier contrast must remain positive within images,
across temporal windows, and after symmetric codec transforms. A difference
that appears only between providers is a renderer statistic and is rejected.

#### H3: select-confirm GLRT with a same-image null

Large external control sets remain necessary, but each image can also supply a
content-matched null. Estimate period, phase, orientation, and channel direction
on one region set. On disjoint regions, compare the selected carrier against
neighboring off-lattice bins, orthogonal orientations, phase-scrambled
templates, and nonselected lattice hypotheses. Convert the selected-versus-null
rank into an empirical p-value, then calibrate the final maximum over experts
with the existing conformal cascade.

This is stronger than scoring every candidate on the same pixels: selection
noise cannot confirm itself, and textured images define their own spectral
background.

Gate: p-values must be approximately super-uniform on every locked negative
stratum. A positive-only score improvement without null calibration does not
advance.

#### H4: content-whitened multichannel matched filter

The current template correlations normalize global energy but do not model the
image-specific complex noise covariance around carrier bins. Estimate local
power spectral density and the full RGB or LMS cross-channel covariance from
neighboring noncarrier frequencies, whiten the residual, and then apply the
frozen complex carrier template. Learn at most one generalized-eigenvector
channel projection per expert on training data; Green, Lab, and OKLab are
baselines rather than separately selected hypotheses.

Gate: whitening must improve within-parent carrier margin in high-texture
positives and reduce the heavy null tail on natural resampling aliases. It must
not depend on scene category or expose RGB thumbnails to a learned model.

#### H5: repeated unknown-payload consistency

Cross-image phase lock can fail when every image carries a different payload or
key. The watermark may still repeat one internally consistent phase or sign
codeword across distant patches of the same image. Estimate a short complex
phase/sign vector independently in each nonoverlapping patch, allow one global
cyclic phase and polarity nuisance parameter, and test patch-to-patch agreement.
Do not require the codeword to match another image.

This generalizes the existing quadrant minimum: it tests a structured vector
and its parity or low-rank consistency rather than one scalar correlation. Use
synthetic random codewords and phase-scrambled equal-power fields to mutation-
test that the statistic recognizes repetition, not merely periodic energy.

Gate: agreement must transfer to new source images and remain absent from
periodic graphics, demosaicing grids, JPEG blocks, and renderer lattices. If no
same-image consistency exists in low-texture oracle positives, reject the
payload-repetition model.

#### H6: calibrated encoder-state mixtures

One provider may deploy several encoder keys, payload families, strengths, or
model epochs. Fit a small mixture only on matching-oracle positives or causal
pairs, freeze every component before evaluation, and expose each component as
a separately identified expert. The union uses search-corrected conformal
p-values, not per-component thresholds chosen from the test set.

The earlier OpenAI multi-direction codebook does not justify this branch: it
was trained on assertion-labeled exports and failed fresh specificity. Reopen
the hypothesis only with the P1 OpenAI prerequisites. For Gemini, test whether
the known first and orthogonal carrier directions remain separate on official
pixel positives.

Gate: a mixture must rescue locked positives without increasing the familywise
false-positive bound, and every retained component must recur in a later time
window. Singletons and post-test clusters are discarded.

#### H7: residual-only equivariant network

If H1-H5 expose a causal statistic but a hand-built aggregation loses recall,
train a compact network on high-pass residual fields with explicit translation,
rotation, scale, and codec transformations. Give it complex carrier maps or
steerable-wavelet responses, not RGB content. Enforce source-matched pairs,
leave one encoder state and one transform family out, and keep the classical
expert as an independent confirmer.

Gate: the network must improve the paired locked test and the unseen-transform
test at the same low-FPR boundary. Saliency must remain on residual carrier
structure; a content or border shortcut kills the branch.

#### H8: optional multi-image evidence

A user may possess a set of outputs from one generation session. A separate
batch detector can aggregate weak phase or codeword evidence across diverse
images after content-hash grouping. It may detect a shared encoder state that
is too weak in one image, but it must return batch-level confidence and must
not relabel every member as individually detected.

Gate: define a minimum number of content-diverse parent groups, evaluate whole
sessions as the independent unit, and challenge same-renderer negative batches.
If gains disappear after session and content balancing, the result is provider
attribution rather than watermark detection.

### Hypothesis execution policy

Run H1, H3, and H4 first for Gemini because they directly target the measured
resize, texture, and large-image failure modes. Run H2 and H5 as the next
mechanism probes because they can detect payload-varying structure without a
shared cross-image template. H6 and H7 require the new oracle-positive corpus;
for OpenAI they additionally require the authorization or causal-pair boundary
in P3. H8 is a separate product mode, not a substitute for single-image
qualification.

At most one hypothesis family may consume a locked partition. A failed locked
test closes that partition for model selection and opens a new version only
after a fresh holdout exists. This is the cost of being creative without
turning repeated hypothesis search into an invisible false-positive generator.

### 2026-08-14: split-confirm lattice synchronization result

The first H1/H3 prototype is implemented in
[`synthid_affine_lattice_probe.py`](../scripts/synthid_affine_lattice_probe.py).
It samples strong complex harmonics from 16 nonoverlapping checkerboard patches,
selects period candidates on eight patches, and confirms the chosen lattice on
the other eight after correcting every patch for its global origin. Five
separated zero-rotation period candidates then enter an amplitude-aware
reranker. Cyclic phase and period are selected from one patch group and scored
on the other. The final full-image template correlation preserves the observed
phase; a separately reported cyclically registered score is diagnostic only.

The separation matters. A broad coherence search at 0.25-pixel spacing selected
harmonic aliases such as 10.5 instead of the true 12.8-pixel period. Refining the
grid to 0.1 pixels recovered the exact expected periods in all 20 transformed
positive views: 12.8, 14.4, 17.6, and 19.2 pixels for scale factors 0.8, 0.9,
1.1, and 1.2. A candidate-count mutation from five to one selected the same
periods and produced the same template scores in all 20 views. The grid
resolution, not the amplitude reranker, caused this recovery; amplitude remains
an independent confirmation statistic. Cyclic registration was also rejected
for the ordinary scale route. It increased scores on
resize-interpolation lattices, whereas the phase-preserving correlation kept
those aliases negative.

One provisional conjunction used the already frozen fixed-template threshold
`0.17357069773071196`, split-confirm lattice coherence of at least `0.52`, and
split-confirm amplitude of at least `0.20`. These latter two constants are
research thresholds, not runtime calibration. Results were:

| Challenge | Positive result | Negative or cross-provider result |
| --- | --- | --- |
| Native matching-oracle pilot | 5/5 | 0/130 matching-oracle Picsum controls |
| Four-scale development matrix | 20/20 transformed views from 5 parents | 0/200 views from 50 oracle-negative parents |
| Four-scale locked matrix | 20/20 transformed views from the same 5 positive parents | 0/800 views from 200 previously unused oracle-negative parents |
| Native large official positives | 7/7 | Not a new negative challenge; the frozen large-v1 controls remain the applicable evidence |
| Source-disjoint provider-labeled test | 295/455 Google-labeled candidates | 0/469 OpenAI-labeled candidates |
| Broad non-Google challenge | Not applicable | 3/276 scored rows, plus 6 unsupported small images |

The four transformed views of one parent are correlated observations, so the
scale rows establish transform consistency for 5 positive and 250 negative
parents, not independent sample sizes of 20 and 1,000. Across the locked scale
matrix, the phase-preserving template score had positive minimum `0.3603` and
negative maximum `0.1716`; split coherence had positive minimum `0.5344` in the
development matrix and locked negative maximum `0.5028`. All 12 independent
matching-oracle native positives, five ordinary-size and seven large, selected
period 16 and crossed the provisional conjunction.

The three broad-challenge crossings are not established false positives. Two
carry TC260 AIGC provenance and one carries Samsung Galaxy C2PA provenance;
all three independently crossed the earlier fixed carrier detector. They show
that the measured carrier family is not safe for provider attribution. They may
be another compatible watermark deployment, Google-backed editing inside a
different product, or a shared renderer lattice. Only a matching pixel oracle
or source-matched counterfactual can distinguish those explanations.

The cross-provider result is nevertheless useful: the Google codebook plus the
new synchronization path accepted none of 469 OpenAI-labeled test images. This
supports a different OpenAI payload or carrier family rather than a uniformly
weaker copy of the measured Google code. It does not provide a local OpenAI
detector; the official OpenAI pixel verifier remains the only production route.

Two related hypotheses were rejected. Phase-only cyclic codeword agreement
retained hard non-Google carrier rows, so amplitude cannot be discarded. A
period-8 decoder made by antialiasing the 16-by-16 codebook into 8-by-8 templates
with area, linear, cubic, or Lanczos kernels also failed: its best Google-versus-
control AUC was `0.567`, and split amplitude remained near chance. Period 8 is
therefore still an unsupported state, not a threshold adjustment.

This result promotes split-confirm synchronization to the next Gemini research
candidate, but not to runtime. The positive oracle set remains small, the
provider-labeled corpus is not a clean oracle, and the amplitude reranker
currently supports isotropic scale at zero rotation only. Rotation, crop,
translation, perspective, JPEG/WebP, independent temporal positives, and a
fresh clean negative source must pass before a production expert is calibrated.

### 2026-08-14: content-whitened matched-filter result

H4 was implemented as a complex multichannel matched filter on the 16 strongest
template harmonics. For each harmonic and patch, a ring of neighboring
noncarrier bins estimates the image-specific complex covariance of Green,
Red-minus-Green, and Blue-minus-Yellow. A fixed ridge regularizes that covariance
before the template and observed carrier vectors are whitened. Period is chosen
from the five H1 candidates on one checkerboard of patches; the reported score
is the smaller whitened cosine on the selection and disjoint confirmation
checkerboards. This retains the P1 separation and charges the candidate search
to the observed negative tail.

The implementation passed synthetic carrier, independent-noise, resize, and
cyclic-phase mutations. It also found a real synchronization failure that the
amplitude reranker missed: under a broad native search, one of five official
ordinary-size positives selected a period-24 harmonic alias. Choosing among the
same frozen candidate set by whitened selection score changed that result to
period 16, and all five ordinary-size positives then selected period 16.

The matched-filter score itself did not pass its preregistered improvement gate:

| Challenge | Whitened positives | Whitened negatives | Whitened margin | Existing template margin |
| --- | --- | --- | --- | --- |
| Native pilot | 5/5, range `0.7006-0.8243` | 130 controls, maximum `0.4797` | `0.2209` | `0.2650` |
| Locked 0.8 resize | 5/5, range `0.4268-0.5994` | 200 fresh-parent controls, maximum `0.2165` | `0.2103` | `0.2420` |
| Native large positives | 7/7, range `0.5401-0.6017` | No new negative partition consumed | Not estimated | Not estimated |

The different score scales make the absolute margins diagnostic rather than a
shared operating threshold, but both locked comparisons point in the same
direction: whitening separated the measured cohorts yet did not improve the
boundary over the phase-preserving spatial template. H4 is therefore rejected
as a replacement detection statistic. Its candidate-selection role remains a
useful H1 synchronization refinement and must be calibrated as part of that
search, not added as another post-hoc threshold.

The full complex code also sharpened the interpretation of the non-Google
challenge. The three earlier TC260- or Samsung-provenance crossings scored
`0.4764`, `0.6666`, and `0.7746`; two match the whitened code more strongly than
the weakest official large positive. They cannot be dismissed as generic
period-16 image structure. They remain compatible watermark-family detections,
Google-backed edits, or a shared implementation until a pixel oracle or causal
counterfactual labels them. Provider attribution from this code is still
invalid.

Finally, the ten OpenAI-labeled rows with the largest earlier Google-template
scores reached at most `0.2234` under the whitened Google code. This selected
challenge is not a sensitivity estimate and does not create an OpenAI detector,
but it is inconsistent with the hypothesis that those hardest OpenAI rows carry
the same code at merely lower amplitude. The next local OpenAI experiment still
requires authorized oracle labels or watermark-on/off pairs from which to learn
its own carrier and codebook.

### 2026-08-14: repeated unknown-codeword result

H5 was tested without requiring the observed phase vector to match the known
Google template. The codeword is estimated from one checkerboard of patches;
the score is the magnitude of its complex weighted inner product with the
independently estimated codeword from the other checkerboard. The 16 harmonic
coordinates remain preregistered from the carrier template, but the phase and
sign values are learned within each image. A global phase or polarity cancels
in the cross-product.

The initially proposed free cyclic-shift nuisance was rejected. Selecting a
shift on alternating harmonics and confirming it on the other harmonics gave
official-positive scores of `0.3651-0.6739`, but the 130-control maximum was
`0.4698`. A fixed zero-relative-shift variant also overlapped. Natural
periodic phase supplied enough shift aliases that search-corrected confirmation
did not separate the cohorts.

After global patch-origin correction, no relative shift is expected. The final
preregistered variant therefore uses all 16 harmonics, performs no shift or
harmonic selection, and confirms only across the two patch groups. It produced:

| Challenge | Official or oracle-positive range | Control maximum | Margin |
| --- | --- | --- | --- |
| Native fixed period 16 | 5/5, `0.4498-0.7291` | 130 controls, `0.2879` | `0.1619` |
| Locked 0.8 resize at period 12.8 | 5/5, `0.3922-0.6713` | 200 fresh-parent controls, `0.2381` | `0.1541` |
| Native large fixed period 16 | 7/7, `0.4044-0.6485` | No new negative partition consumed | Not estimated |

Synthetic mutation tests confirmed the intended invariance: the score accepted
the repeated random carrier, rejected independent equal-power noise, survived
isotropic resize after period canonicalization, and stayed high after a global
cyclic image translation even though the known phase-preserving template score
collapsed.

The three TC260- or Samsung-provenance compatible carriers scored `0.3994`,
`0.4541`, and `0.6052`. Their within-image codeword repeat is therefore also
inconsistent with an accidental single global Fourier spike. This strengthens
the compatible-signal-family interpretation without identifying a provider.
The ten OpenAI-labeled rows selected for their earlier Google-template scores
reached at most `0.2568` at fixed period 16, below the native-control maximum.
That selected result is not a full OpenAI screen, but supplies no evidence that
the same period-16 harmonic support carries a different repeating OpenAI
payload.

H5 advances as an independent research confirmation, not a runtime threshold.
The current evaluation supplies the correct period rather than charging the
full period search to this statistic, has only 12 independent official positive
parents, and reuses the available Picsum negative source. It must next survive
the broad H1 candidate search, the remaining scale and codec matrix, a second
clean negative family, and independent temporal positives. A free-shift version
is not reopened for the ordinary scale route; translation and crop belong in a
separately calibrated expert.

### 2026-08-14: complex cyclostationary result

H2 was implemented as the normalized complex spectral-correlation matrix
`F(k + alpha) * conjugate(F(k))` across all nine Green/opponent-color channel
pairs. The eight strongest preregistered template harmonics supplied `alpha`.
For every carrier shift, six one-bin neighboring shifts formed a same-radius
same-image null. Complex matrices were averaged separately over the selection
and confirmation checkerboards before taking their Frobenius norms, so content
phase had to repeat across distant patches to count.

Synthetic mutations verified that this was a genuine cyclostationary test. An
amplitude-modulated equal-power random field produced positive carrier-minus-
neighbor contrast on both checkerboards; independent noise and an additive
periodic carrier did not. The latter mutation distinguishes H2 from the H1/H5
additive-codeword family.

The native oracle pilot failed decisively:

| Statistic | Five official positives | 130 controls | Positive minimum minus control maximum |
| --- | --- | --- | --- |
| Selection contrast | `-0.0027-0.0342` | maximum `0.0593` | `-0.0620` |
| Confirmation contrast | `0.0067-0.0337` | maximum `0.0608` | `-0.0542` |
| Joint contrast | `-0.0027-0.0337` | maximum `0.0593` | `-0.0620` |

H2 is therefore rejected without consuming another locked scale partition or
searching frequency bands, harmonic subsets, color spaces, or null offsets on
the observed controls. The measured Google signal supports a repeated additive
or content-adaptive codeword, not a detectable global multiplicative
cyclostationary modulation. This result does not constrain an unknown OpenAI
carrier, but the OpenAI branch still lacks the oracle evidence required to fit
one without provider leakage.

### 2026-08-14: fixed-period codec and crop pilots

The H1/H4/H5 probe gained explicit in-memory JPEG and top-left crop transforms
so the same decoded parent pixels enter every paired comparison. These pilots
use the five ordinary-size official positive parents and split the 130 native
controls into 50 development and 80 locked controls. The transformed views are
not new independent positive evidence.

Lossy JPEG separates the useful statistics. Whitened and unknown-codeword
scores no longer maintain a positive development margin even at quality 95, so
they are unsupported for the lossy-codec route. The phase-preserving spatial
template remains usable at the known native period. Thresholds were frozen
after the 50-control development split and then applied unchanged:

| Transform | Frozen template threshold | Repeated positive views | Locked controls | Locked score margin |
| --- | --- | --- | --- | --- |
| JPEG 95 | `0.12` | 5/5 | 0/80 | `0.1054` |
| JPEG 90 | `0.09` | 5/5 | 0/80 | `0.0727` |
| JPEG 80 | `0.065` | 5/5 | 0/80 | `0.0257` |

The declining margin is monotonic and leaves JPEG 80 especially fragile. These
constants are research thresholds, not a runtime calibration: they use only
five positive parents, one negative family, fixed period 16, and one OpenCV
JPEG implementation. JPEG 100, chroma-subsampling variants, WebP, a second
encoder, codec-after-resize combinations, and independent positives remain
unqualified.

Cropping shows the opposite phase behavior. Removing 5% or 10% from the top
and left collapses the unregistered template score, while cyclic phase
registration recovers the carrier. A crop-only conjunction was frozen at
registered template at least `0.20`, split coherence at least `0.55`, and split
amplitude at least `0.20`:

| Transform | Repeated positive views | Locked controls | Registered-template locked margin |
| --- | --- | --- | --- |
| 5% top-left crop | 5/5 | 0/80 | `0.2275` |
| 10% top-left crop | 5/5 | 0/80 | `0.2378` |

The unknown-codeword H5 statistic independently retained positive margins of
`0.0872` and `0.0839` in those two cells. This supports a separate native
period-16 crop/translation expert. It does not reopen cyclic registration in
the ordinary scale route, where the same nuisance parameter was already shown
to amplify interpolation aliases. Arbitrary crop sides and offsets, one-pixel
translations, combined crop-plus-codec edits, rotation, screenshots, broad
negative sources, and new positive parents remain before qualification.

### 2026-08-14: fresh-source failure and frozen v3 confirmation candidate

The unchanged runtime experts were challenged on the first 3,000 decoded-
pixel-unique files from the official Open Images test split. The image-id list
was fixed before download and has SHA-256
`9b83bc221dd70823f39a9ad62b16c1dcc1f9c449f17c9ea3f6c9a51b581e0c1a`;
the ordered pixel-file hash list has SHA-256
`2c49ccdec79ec662f202670b9e4c5ee0dfdd80d5747929dd09795d945b3a22a6`.
This source-fresh challenge rejected both existing precision claims. The fixed
expert accepted 5 of its 211 supported images at its shipped `0.1735706977`
threshold. The registered expert accepted 4 of 2,997 supported images at its
shipped `1.0` threshold. These are false-positive measurements, not clean-image
labels inferred from detector misses.

The fixed v3 research candidate is frozen at `0.28`. It is above the largest
fresh crossing, `0.2592123`, and below the weakest of 12 source-diverse native
Google positives, `0.2978951`. That separation selected the threshold, so the
3,000-image set is development evidence for v3 and cannot validate it. The
candidate must remain unchanged on a second untouched source split.

The registered v3 research candidate preserves the unchanged registered base
decision and adds an independently split phase confirmation. Selection and
confirmation use disjoint checkerboards of patches. The confirmation requires
period at least `10.0`, joint harmonic coherence at least `0.30`, and nonnegative
joint cyclic amplitude. Three narrow period ranges add the following measured
alias controls:

| Registered period | Additional frozen gate |
| --- | --- |
| `18.0 <= p < 18.6` | held-out unknown-codeword confirmation at least `0.13` |
| `18.6 <= p < 20.0` | joint harmonic coherence at least `0.40` |
| `20.0 <= p < 22.0` | held-out unknown-codeword confirmation at least `0.02` |

The rule rejected all four fresh registered crossings while retaining 147 of
148 base-accepted positives from 226 source-diverse transformed parents. A
separate dense transform matrix applied 35 scales from 0.65 through 1.50 in
0.025 increments to 12 native positive parents; it retained all 359 views that
the unchanged registered base accepted. At the hard-negative period `18.28`,
all 11 base-accepted positive views passed, while the natural-image crossing
failed the held-out-codeword gate. Derivatives are not counted as independent
positive evidence. The exact candidate constants live in
`scripts/synthid_affine_lattice_probe.py` and are frozen before the second
negative holdout is acquired.

An explicit period-8 rescue is still rejected. A six-gate candidate selected
on earlier controls accepted 7 of these 3,000 fresh controls, so scale 0.5 has
no qualified local route.

### 2026-08-14: registered-v3 holdout qualification and weak-signal rescue

The registered confirmation rule above was frozen unchanged before a second
3,000-file Open Images test-split cohort was decoded. Its image-id list has
SHA-256 `91e1f19f559a432f61c5eb1f16797d40d706170c43094bc7d4cb9506309b662a`;
the ordered pixel-file hash list has SHA-256
`651f959689e7043989ded09fa4aece5cea2592e7b69e371cc97c35f581d9f57d`.
All 3,000 decoded hashes were unique and none overlapped the first cohort. The
unchanged registered-v2 base produced no threshold crossings among 2,996
supported images, so registered-v3 also produced 0/2,996. Together, the two
Open Images cohorts give registered-v3 0/5,993 false positives after the
confirmation rule was applied, with a one-sided 95% zero-error upper bound of
about 0.050%. They are source-disjoint samples from one source family, not two
independent acquisition mechanisms.

A second-family challenge used the first 3,000 sorted COCO val2017 files. The
newline-delimited file-name list has SHA-256
`7a2e5f47f6c775f0de839b8270ee0874526a8d0984c66cd7940e1a04fc89f4f5`;
the ordered per-file SHA-256 manifest has SHA-256
`b1b0c73891b6d5b49a7817c9bd0a47309a424e437ba36fabed7bd9077dfeda54`.
All 3,000 pixel hashes were unique. The unchanged registered base produced
0/2,366 crossings, with maximum score `0.932964706`; registered-v3 therefore
also produced zero. COCO predates this confirmation experiment and is not a
fresh acquisition, but it is a distinct natural-image source family. Across
both Open Images cohorts and this COCO challenge, registered-v3 has 0/8,359
supported-control crossings. This qualifies the precision-first positive route;
it does not turn a miss into proof of absence.

The fixed `0.28` candidate failed the second holdout and is rejected. It
accepted one of 213 supported images, at score `0.322542963`. The runtime keeps
fixed-v2 only as an explicitly requested diagnostic and never unions it with a
production positive route.

Split confirmation was also tested as a recall rescue below the registered base
threshold. The first relaxed rule recovered 16 of 76 source-diverse base misses
but accepted 94 of the 312 most dangerous first-cohort controls, so repeated
phase alone is decisively insufficient. A stricter H4/H5 conjunction required
matching spatial and spectral periods, base amplitude margin at least `0.4`,
high-band margin at least `0.3`, the frozen confirmation gates, and a minimum of
content-whitened template match, all-harmonic repeated-codeword match, and
split cyclic amplitude at least `0.225`. The threshold is above the first-
cohort maximum `0.222382188`. It added only 6 of 224 scored source-diverse
positives and 8 of 432 dense-scale views, while accepting none of the second-
cohort candidates and none of the scored COCO candidates. That small recall
gain does not justify another runtime branch: it was selected on exposed
development positives, adds a substantially more expensive whitening pass, and
lacks new positive parents. It remains a recorded research candidate rather
than being hidden inside registered-v3.

The runtime implementation was then checked on the transformed pixels rather
than the original source paths. Of 226 source-diverse transform assignments,
224 remained large enough for the split-patch research scorer. Registered-v3
detected 147/224 overall: 0/40 at 0.5x and 147/184 (79.9%) from 0.65x through
1.5x. The v2 base accepted 148 of the latter 184 and v3 confirmation rejected
only one base crossing. Failure attribution on the 184 supported non-half-scale
views was 10 period-winner mismatches, 17 amplitude failures, 9 high-band
failures, one confirmation failure, and 147 detections. This production-parity
measurement replaces the earlier accidental source-path check; transform
labels attached to untransformed files are not sensitivity evidence.

H3 then supplied a genuinely content-matched null. For each registered period,
the 16 fixed neighboring periods at offsets from plus or minus 0.35 through 2.0
pixels are scored on the same two checkerboards. The candidate statistic is the
smaller of the selection- and confirmation-group target coherence minus that
group's largest off-period coherence. Combining the two Open Images development
cohorts froze a recovery rule at registered base score at least `0.8`, the
existing v3 confirmation gates, and same-image joint excess at least
`0.05695968150082637`. It rescued 7 of 36 sub-threshold transformed views and
accepted 0/262 prescreened candidates from the first 3,000 COCO controls. The
untouched final 2,000 COCO files contained 1,598 registered-supported images
and seven near-threshold candidates. Four candidates were too small to form two
independent 256-pixel patch groups; all three scored candidates had joint excess
at most zero. The frozen rule and registered-v3 both accepted 0/1,598. This
passes the new negative-side test without changing the threshold. The candidate
remains outside runtime because all seven recovered positives influenced
feature selection and no new positive parent has yet confirmed the gain.

The same null did not justify relaxing the registered requirement that the
spatial reconstruction winner equal the strongest spectral-period candidate.
A full rescore of the two 3,000-file Open Images cohorts retained 469 plausible
mismatch candidates after amplitude and high-band prefilters; 212 also passed
the frozen split confirmation. Their same-image joint-excess maximum was
`0.1762448251`. Only five of the ten non-half-scale positive mismatch records
passed the same mechanism prefilters, and their maximum was lower at
`0.1464860382`. No zero-control same-image threshold retained a positive. A
post-hoc product of amplitude margin, high-band margin, and coherence could
retain two positives above the observed control maximum, but it was selected
from many combinations and offers too little gain to open another holdout.
Spatial/spectral winner equality therefore remains mandatory.

### 2026-08-14: content-adaptive patch-phase recovery candidate

The official SynthID-Image paper describes a content-dependent learned
watermark pattern, so a single whole-image template cannot be assumed to be the
complete decoder. A new research statistic therefore treats the frozen tile as
a synchronization basis rather than as a fixed global payload. At the
registered period, each 256-pixel patch is folded independently, every cyclic
tile shift is standardized against that patch's own shift distribution, and
the strongest half of the selection checkerboard chooses one phase. The score
is the smaller of that phase's trimmed standardized response on the selection
and disjoint confirmation checkerboards. Filenames, metadata, and scene labels
do not enter the statistic.

The research-only recovery rule was frozen at matching spatial and spectral
periods, the existing split confirmation, both registered gate margins at
least `0.45`, one gate margin at least `1.0`, and joint patch-shift score at
least `2.5`. It recovered 5 of 18 scored amplitude/high-band failures from the
224-view transformed-positive audit. Added to registered-v3 alone, that would
move the measured non-half-scale result from 147/184 to 152/184 (82.6%). It
accepted none of 88 gate-failure candidates in the first 3,000-image Open
Images development cohort and none of 334 previously scored control
candidates; those sets overlap and are not added as independent denominators.
The rule and threshold were recorded before reading a 350-image Picsum
feature-holdout. All 350 decoded rasters were unique. The frozen rule accepted
none; only two controls reached its mechanism pre-gates, and their maximum
joint patch-shift score was `0.7029283` against the `2.5` threshold. The
zero-error one-sided 95% upper bound is about 0.85% for this one-source
feature-holdout. The candidate remains outside runtime until new positive
parents pass unchanged.

That positive test then failed on a genuinely different carrier cohort. Seven
large Gemini images with independent official pixel-positive verdicts, none of
whose source paths appeared in the 224-view transform audit, were resized by
0.65, 0.70, and 0.75. Registered-v3 accepted 0/21 and the frozen patch-phase
recovery added 0/21. The base amplitude was often strong, but the RGB high-band
margin became negative on most views and several period searches selected an
octave alias. The branch is therefore a real recall improvement for the
measured registered carrier, but not a universal cross-epoch solution. A
scale-registered form of the independently validated large-v1 opponent-color
signal is the next hypothesis; weakening the existing high-band gate is not.

That hypothesis recovered the entire failed cohort. A signed spectral curve in
the Red-minus-Green and Blue-minus-Yellow planes searched periods 10.0 through
12.5 in 0.05-pixel steps, retained three separated candidates, canonicalized
each to the frozen period-16 tile, and reranked by the minimum of fixed RGB,
Red-minus-Green spatial, and Blue-minus-Yellow spatial agreement. Thresholds
were frozen at `0.16`, `0.60`, and `0.55`, respectively. Automatic period
selection recovered the exact injected period on all 21 views and accepted
21/21, versus registered-v3's 0/21. It accepted 0/21 symmetrically resized
views from seven large Open Images controls; the maximum control decision
margin was `0.9462124`. These positives and controls selected the gates, so
this is development evidence. The candidate was frozen before a 350-image
Picsum feature-holdout. It accepted 0/350 decoded-pixel-unique controls there;
the maximum decision margin was `0.7778322`. This negative-side result passed
unchanged.

The broader challenge rejected that unrestricted form. Extending the search to
periods 7.5-14.5 recovered 63/63 lossless views from the same seven official
parents at nine scales from 0.5 through 0.9, but accepted 605/2,997 and
587/2,996 controls in two Open Images cohorts. The crossings concentrated at
periods 8 and 12.8 and tracked JPEG sampling. A 3,964-image COCO challenge added
one period-10 crossing below one megapixel, while 3,772 heterogeneous
non-Google/non-OpenAI Spaces controls added 28 crossings, again only at periods
8 and 12.8. The unrestricted branch is a codec/resize-lattice detector and is
rejected.

A symmetric codec test separated lossless scale recovery from recompression.
At scales 0.5, 0.625, and 0.75 the raw views passed 21/21, while JPEG-95,
JPEG-85, and WebP-95 passed 0/63. At the supplied target period, half-scale JPEG
views inverted the opponent-color polarity and the other lossy views largely
lost spatial agreement. This is not a period-search-only failure.

The production candidate was then frozen at 1-10 megapixels, minimum side 768,
selected period 8.1-12.0, and the original `0.16`, `0.60`, and `0.55` gates.
An initially added spectral/spatial period-agreement condition was removed when
the real API path showed that a harmless 11.0 spectral alias reduced 0.55x
recall from 7/7 to 1/7; its only reserve-control counterpart was below the
already frozen geometry floor. The revised rule was fixed before a separate
1,000-image Picsum holdout. It accepted 0/1,000, with maximum unrestricted
margin `0.7953734`, and the production `detect_synthid` seam detected 42/42
lossless views across scales 0.55-0.75. This bounded expert now runs after
registered-v3 abstains as `synthid-periodic-tile-opponent-registered-v1`.

### 2026-08-15: period-8 codec veto and lossless 0.5x recovery

The original period exclusion conflated two observations: the 0.5x watermark
and the native JPEG block lattice both select period 8. Direct template polarity
did not resolve the confound. JPEG-95 and JPEG-85 half-scale positives produced
a strong inverted opponent template, but random JPEG 4:2:0 controls produced
the same profile. A matched resize-plus-JPEG residual experiment used 175
control parents for development, 175 disjoint parents for holdout, three JPEG
qualities per parent, and leave-one-positive-parent-out checks. The best ridge
variant accepted 19/21 fitted positive views but also 86/525 holdout controls;
the variants with fewer control errors lost most or all positive parents.

A nonlinear block-DCT follow-up evaluated 80 target-derived statistics in RGB,
Green, opponent-color, and YCbCr views on the same matched transform. Its best
zero-holdout-error statistic retained only 2/21 JPEG positives. The highest-AUC
zero-error opponent statistic reached about `0.735` but retained 0/21 at the
development maximum. Neither JPEG recovery hypothesis enters runtime.

The successful discriminator is instead a codec veto around the unchanged
opponent carrier gates. For each decoded opponent plane, it measures the mean
absolute edge energy on native 8-pixel boundaries relative to the median of
non-block phases. All seven lossless 0.5x positives had both Red-Green and
Blue-Yellow ratios no greater than the frozen `1.05` ceiling. The ceiling
rejected 600/600 development period-8 codec crossings, then 582/582 crossings
in the nonoverlapping Open Images holdout, 601/601 in the feature-unseen frozen
reserve, and 7/7 in the heterogeneous Spaces challenge. An additional 350
controls transformed through the identical lossless 0.5x resize produced zero
base crossings before the veto.

The production fallback now accepts periods 7.9-12.0 and requires the veto
through period 8.1. The public API detects 7/7 scale-0.5 views and 49/49 across
the measured 0.5-0.75 lossless matrix from seven official pixel-positive
parents. This is a bounded lossless extension, not evidence of JPEG robustness
or arbitrary scale coverage.

The provider challenge is evidence for a shared family, not attribution. The
unrestricted statistic accepted 95/7,254 OpenAI files with an explicit SynthID
assertion, all at periods 10.70-10.80, and 4/287 same-lineage files without an
assertion. The latter label is not an oracle-negative watermark verdict. The
95 decoded rasters were unique and had zero overlap with the 94 period-8
candidate hits, producing a 189/7,259 union. However, JPEG-95 and WebP-95 each
reduced the new component from 95/95 to 0/95. It remains compatible with either
a complementary carrier epoch or a geometry-linked export lattice. The runtime
result remains provider-neutral, and the official OpenAI verifier is still the
only qualified provider-wide OpenAI route.

### P3: productionize OpenAI without pretending the local route works

The near-term production route is the official pixel verifier. Remaining live
and deployment qualification:

- complete the approved SDK transport set beyond the finished positive PNG
  smoke, covering JPEG, WebP, and pre-established positive and negative
  expectations with pixel-identical, metadata-free files;
- malformed file, 50 MiB boundary, timeout, disconnect, 400, 404, 429, 5xx,
  unexpected response, duplicate SynthID result, and missing SynthID result
  qualification against recorded or contract-safe fixtures;
- deployment-level bounded backoff and a circuit breaker for transient API
  failures, without representing transport failure as a watermark verdict;
- confirmation of data-retention, access-control, audit-log, and user-consent
  requirements for the deployment environment;
- secret-safe logs that retain request ids, byte counts, timings, and status but
  never image bytes, credentials, user paths, or decoded-pixel hashes.

The local transport now fixes both the default SDK and per-request timeout at
120 seconds, sets automatic retries to zero so one acknowledgement cannot
retransmit media, and omits the decoded-pixel hash from request logs. Contract
tests cover exact and over-limit file sizes, caller cancellation, malformed
responses, all documented HTTP classes, timeout, and connection failure. A
structured error preserves status, API error code, request id, `Retry-After`,
and whether an explicit retry is appropriate. One credentialed positive PNG
smoke has now passed. Remaining format qualification, deployment retention
approval, and an operational circuit breaker remain environment-level work;
the research harness itself does not perform live uploads.

A local OpenAI detector reopens only after one of two prerequisites exists:
written authorization to obtain non-adaptive pixel-verifier labels at research
scale, or source-matched watermark-on/watermark-off pairs. The first local
epoch then starts from those causal residuals and uses same-provider negatives.
No additional model is selected on C2PA assertions, filenames, OpenAI export
geometry, or the already exhausted provenance-labeled cohorts.

### Updated local-detector execution order

The local program now advances through evidence gaps rather than adding more
unbounded feature searches.

1. Freeze the current routed Gemini bank, calibration artifacts, parent-group
   manifests, and support reasons as the reproducible baseline. Finish the
   active fine/anisotropic registration experiment against a fresh negative
   holdout and either promote it as a separately versioned expert or reject it
   before opening another synchronization branch.
2. Run the production router, not isolated expert functions, over the complete
   available size, scale, aspect, orientation, and codec matrix. Group results
   by parent and cluster misses and abstentions by measured carrier state. This
   produces the actual coverage backlog.
3. Address Gemini clusters one at a time. Use full reciprocal-lattice
   registration first, select-confirm same-image null and repeated unknown
   codeword confirmation second, and codec-specific residual experts third.
   Every branch selects on one region group, confirms on disjoint regions, and
   receives its own multiple-search null calibration.
4. Do not train a local OpenAI classifier from renderer style or provenance
   assertions. Existing verifier-confirmed cases may validate whether a
   candidate reaches the known pixel signal, but the training corpus requires
   source-matched watermark-on/off pairs or written authorization for a
   non-adaptive research label set. Same-provider hard negatives are mandatory.
5. Once that OpenAI prerequisite is met, learn causal residual differences,
   cluster stable carrier states across model and time windows, and expose each
   recurring state as a separate versioned expert. Test reciprocal-lattice,
   multichannel complex matched-filter, and within-image unknown-codeword
   families before any residual-only learned model.
6. Qualify both provider banks through one locked and later temporal matrix.
   A positive names only the provider scope proved by exclusivity data; a miss
   is `not_detected` only inside sensitivity-certified strata and otherwise
   remains `indeterminate` or `unsupported`.

### P4: run one symmetric robustness matrix

Every promoted expert and both official-verifier integrations receive the same
parent-grouped challenge:

| Axis | Frozen views |
| --- | --- |
| Codec | lossless PNG, source JPEG, JPEG 100/95/90/80, WebP lossless/95/80, 4:4:4 and 4:2:0 where controllable |
| Scale | 0.5, 0.625, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.333, 1.5, and 2.0 |
| Geometry | supported size-bin boundaries, modulo-period edges, portrait, landscape, square, and extreme aspect ratios |
| Spatial edit | 1-10% crop, one-pixel and multi-pixel translations, screenshot round trip, and 90/180/270-degree rotation |
| Pixel edit | mild blur, sharpen, denoise, gamma, contrast, saturation, and color-profile conversion |
| Content | photo, face, text-heavy, illustration, flat graphic, low texture, high texture, and alpha |

Transforms are applied identically to positives and controls. Scores and
verdicts are aggregated by parent, not by treating eleven derivatives as eleven
independent images. Unsupported cells are recorded as such; they do not lower
the denominator silently. Robustness work stops when the signal is no longer
identifiable at the required false-positive boundary.

### P5: use release-grade statistical gates

Freeze the following gates before the new corpus is scored:

- the one-sided 95% upper confidence bound for false-positive rate must be at
  most 0.1% overall and at most 0.5% in every critical negative stratum;
- the one-sided 95% lower confidence bound for sensitivity must be at least 90%
  in every scope advertised as detection-supported;
- a local `not_detected` absence claim needs a much stronger 99% sensitivity
  lower bound in every advertised stratum; otherwise a miss remains
  `indeterminate`;
- temporal, model-family, surface, geometry, and codec results are reported
  separately, with no claim wider than the weakest passing stratum;
- every selected threshold must remain unchanged on a newly acquired temporal
  holdout and on a second negative source family;
- provider attribution requires cross-provider exclusivity data. Until that
  passes, the local result names the signal family but not its generator.

Confidence intervals, operating thresholds, model hashes, calibration hashes,
split hashes, and code revision go into one machine-readable qualification
report. The report fails closed on missing records, duplicate parent groups,
NaN or infinite scores, unknown detector ids, or a changed artifact hash.

### P6: harden runtime and operations

Before release:

- benchmark wall time and peak memory at every supported size boundary; keep
  large-image folding and template subtraction stripe-bounded, and freeze
  numeric p95 latency, peak-memory, and concurrency budgets for the target
  deployment before qualification;
- fuzz corrupt, truncated, decompression-bomb, unusual color-mode, alpha, and
  oversized images; errors must be bounded and must not become negatives;
- prove deterministic scores across supported platforms and dependency pins, or
  version platform-specific calibration explicitly;
- run shadow mode before user-visible verdicts, collecting only consented,
  privacy-safe aggregate score histograms, support rates, latency, errors, and
  detector-version counts;
- alert on score-distribution drift, support-rate changes, remote outcome drift,
  error-rate spikes, and latency regressions;
- keep the previous detector and calibration artifacts available for immediate
  rollback, and make every result identify which version produced it;
- require the applicable dependency lock, security scan, Ruff, formatting,
  Pyright, unit, integration, mutation, packaging, and full corpus gates to be
  green. An untriaged production dependency advisory blocks release.

### Execution order and stop/go checkpoints

| Milestone | Deliverable | Go condition |
| --- | --- | --- |
| M1 | Unified result contract and expert registry | No miss, error, unsupported input, or remote failure can be presented as clean |
| M2 | Immutable Gemini and OpenAI qualification manifests | Parent-group split audit and all ingestion mutations pass |
| M3 | Gemini fixed and large requalification | P5 precision and sensitivity gates pass on locked and temporal data |
| M4 | OpenAI SDK/API qualification | Approved live transport matrix passes, privacy boundary is accepted, operational failures abstain |
| M5 | Optional registered, codec, and crop experts | Each expert independently passes P5; failed cells stay unsupported |
| M6 | Shadow deployment | No unexplained drift or SLO regression through the prespecified observation window |
| M7 | Production release candidate | All code, corpus, security, documentation, rollback, and versioning gates pass |

The immediate work queue is M1, freezing and auditing the current routed
baseline, completing the active Gemini fine/anisotropic candidate decision,
finishing the 3,000-image fresh Gemini large negative acquisition, and running
the router-level all-resolution matrix. In parallel at the planning and data
provenance level, identify a permitted source of OpenAI watermark-on/off pairs
or obtain written authorization for a non-adaptive research label set. The
first credentialed OpenAI positive PNG smoke is complete; further live API
calls are transport qualification, not local-detector training. New feature
search starts only from a measured coverage cluster and consumes a fresh
holdout once.

### 2026-08-15: OKLab random-codeword replacement probe

A pixel-only probe tested the hypothesis that the measured periodic residual is
only the carrier for a repeated coded message. It compared independent
high-pass OKLab noise, periodic-comb suppression, random periodic injection,
and complete replacement of the estimated period-16 residual with a
magnitude-matched random-phase tile. The four inputs were the cleared public
Google-oracle-positive originals already in the tracked research corpus.

Independent noise was inefficient. At the stronger tested allocation, with
three times more RMS assigned to each chroma channel than to luminance, the
median fixed-carrier score changed by only `-0.0219` while median PSNR fell to
`32.56 dB`. Comb suppression alone changed the median score by `-0.1962` at
`58.05 dB`, but moved only one of four inputs below the unchanged detector
threshold.

Replacing the complete tile was qualitatively different. At replacement
strength `0.95`, all four fixed-carrier scores moved below the frozen
`0.17357069773071196` threshold, from source scores `0.7322`, `0.6503`,
`0.3300`, and `0.4161` to `0.097`, `0.031`, `0.158`, and `0.087`. Median PSNR
was `53.85 dB`, minimum PSNR was `52.56 dB`, and minimum grayscale SSIM was
`0.99811`. All four paired score changes had the same direction, but four
parents give a two-sided sign-test result of only `p = 0.125`. The probe was
designed after the carrier family was known and is a causal local-score
intervention, not an independent detector or removal validation.

A preregistered provider check stopped at its source control. Gemini returned
the expected Google C2PA record but its pixel-verification stage remained at
`Connecting to Verify AI`; no candidate was submitted after the unhealthy
control. The next healthy oracle window must use the unchanged source control,
the frozen `0.95` replacement, and a norm-matched sham. Until that paired batch
returns a source-positive and candidate-negative pixel verdict, random-codeword
replacement remains a stronger mechanism hypothesis rather than a SynthID
remover.

A follow-up radial ablation separated the replacement tile into full-image
frequencies measured in cycles per pixel. On the same four public positives,
randomizing only `0.125-0.40` reduced the fixed score by a median `0.2093` at
`56.48 dB` median PSNR. The `0-0.125` band had no median effect, and frequencies
above `0.40` changed the median score by only `-0.0086`. Lower-middle and
upper-middle bands each contributed, with median changes of `-0.0594` and
`-0.0508`. This locates the measured Gemini carrier's most efficient causal
surface in the middle band, not at the highest frequencies.

The ablation still operates on a folded periodic tile, so every edited
coefficient remains a comb harmonic. It does not distinguish a sparse set of
carrier lines from broadband coded noise between those lines. The next
mechanism test must use continuous annular bands in local patch spectra, compare
spatially independent and cross-patch-coherent perturbations at matched OKLab
distance, and preserve a fixed random seed before any provider verdict.

## Decision record

The program has four possible honest outcomes per provider:

| Outcome | Product consequence |
| --- | --- |
| Causal signal and detector both generalize | Continue to pixel-only removal |
| Detector works but causal attribution fails | Ship no SynthID detector claim; retain as provenance research |
| Detector generalizes but pixel-only removal does not transfer | Keep local detection, retain regeneration fallback |
| Pixel-only removal clears the oracle with quality gates | Productize provider-specific detector and remover |

Stopping at a failed gate is a result. It prevents a local surrogate, export
fingerprint, or quality metric from being mistaken for control over SynthID.

### Decision taken 2026-08-16

The second row is the outcome that occurred, on the evidence in this log. No
local expert in this program detects SynthID. `large-v1` detects the generation
pipeline: its statistic is destroyed by a seven-pixel crop, while the published
evaluation of the mark retains `99.97%` TPR under aggressive crop and resize.
The direction agreed is therefore to keep what works under its true name and to
build toward provenance rather than toward a watermark claim.

**Ship what is real, renamed.** `large-v1` identifies large Google-pipeline
images with a measured `0/2,637` control rate and `8/11` on fresh
out-of-distribution positives. It may be shipped as generator or pipeline
identification, never as watermark detection, and its two failure modes belong
in the product surface rather than in a footnote: it breaks on a crop that is
not tile-aligned, and it breaks when the generation pipeline changes, as
`gemini-2.5-flash-image` output already demonstrates at `0.063`.

**Develop toward provenance.** Generator, pipeline and epoch identification is a
task with abundant data, reliable labels and a measurable gate. U2 shows it is
not solved, since the current features accept between a fifth and a half of
foreign synthetic images, but it is the task the evidence actually supports.

**Keep the watermark route open, blocked on access rather than on effort.** A
crop-robust detector of a keyed, perceptually masked mark needs a learned decoder
trained on clean pairs. Google emits no unwatermarked image from any accessible
path, and OpenAI's verifier may not be used to label a training set. If either
constraint lifts, this route reopens; until then it is not an engineering
problem.

**One rule adopted from the failure.** Every candidate local expert must survive
a non-tile-aligned crop before it is calibrated. The test costs one line, it is
the property the mark provably has, and its absence is why a pipeline artifact
was carried for three months as a watermark detector.

## Historical first milestone

The first local-research milestone was defined to produce evidence rather than
shipping code. Its scope was:

1. the private-corpus schema and auditor;
2. an OpenAI authorization decision for use of the remote provenance verifier;
3. an independently verified status for candidate causal pairs;
4. a canonicalized OpenAI pilot set with hard negatives;
5. the D1 confound report;
6. the D2 low-texture carrier report with leave-one-group-out results;
7. a go or no-go decision for real-image detector training.

The completed experiments produced a no-go decision for a local universal
OpenAI detector. The later official remote backend is a separate production
route and does not retroactively turn provenance-labeled exports into pixel
oracle training data.

## Primary sources

- OpenAI, [Content provenance](https://developers.openai.com/api/docs/guides/content-provenance).
- OpenAI, [ChatGPT Images 2.0 system card](https://deploymentsafety.openai.com/chatgpt-images-2-0/automated-evaluations-and-adversarial-testing).
- Google, [Verify AI-generated images, videos, and audio](https://support.google.com/gemini/answer/16722517?hl=en).
- Gowal et al., [SynthID-Image: Image watermarking at internet scale](https://arxiv.org/abs/2510.09263).
- Meta, [VideoSeal](https://github.com/facebookresearch/videoseal).
- Meta, [Watermark Anything](https://github.com/facebookresearch/watermark-anything).
