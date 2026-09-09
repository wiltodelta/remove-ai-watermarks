# Code and documentation audit

Review snapshot: 2026-09-08, working tree based on `f127ecb`, including the
uncommitted maintenance changes. The finding tables describe that initial
snapshot. The approved repairs are recorded below; no commit or release is
implied by their working-tree status.

## Repair status (2026-09-08)

All 32 package, research, and tooling findings (C01-C13, R01-R15, T01-T04)
have working-tree repairs. All 19 documentation discrepancies (D01-D19)
have corrections, with contradictory historical numbers withdrawn where the
underlying evidence could not establish a replacement. Existing maintenance
changes were preserved; the lockfile was unchanged during these repairs.

| Area | Repair and evidence |
| --- | --- |
| Image and metadata output | Current batch outcomes control publication; default invisible output survives temporary-file cleanup; ICC, orientation, and alpha travel through the real writer seams. ffmpeg maps all streams and stages in-place output. Synthetic MKV tests compare both audio packet hashes. Remove-all drops EXIF, and collection failures remain error/partial records. |
| Detection and calibrated profiles | Doubao receives vendor-specific structural confirmation; Baidu scores the actual Qwen rival. All 42 public image verdict records match the pre-fix baseline. A 15-frame Sora removal has identical output SHA-256 before and after. Chroma retains its calibrated requests; tests call the installed upstream timestep method and document four or five effective steps. |
| Research measurements | Period selection excludes confirmation; exported fallback identity is retained; old incompatible score files require rescoring. Geometry mismatch fails explicitly. Negative evidence, process outcomes, per-response persistence, executed seeds, and content deduplication have regression guards. The historical OCR formula is named NED. |
| Corrected VideoSeal evidence | Fresh local runs measured 32 temporal artifacts and 12 forgery artifacts. All 44 recorded hashes match the saved files. Results and pins are recorded in the [temporal study](videoseal-temporal-evaluation.md#corrected-artifact-measurements-2026-09-08) and [forgery study](watermark-forgery-study.md#corrected-videoseal-measurements-2026-09-08). The existing checkpoint was used without downloads or provider calls. |
| Release and skill tooling | The probe checks the installed CLI's video runtime; ClawHub uses its local executable; automatic release verification follows the triggering distribution run; ComfyUI dependencies are parsed as requirements. Skill metadata is consistently 1.0.5. No workflow was dispatched or skill published. |
| Final cleanup | Four independent simplify angles found shared crash-marker reuse, unnecessary in-place passthrough encoding, needless alpha copying, and nested probe conditionals. Those were simplified. The full suite also exposed a pre-existing environment/reload teardown leak; the test now restores the cache environment before reloading, with a mutation-checked regression. |

Repair agents wrote only disposable source archives. The coordinator inspected
and integrated their exact file lists, reproduced the failures, checked the
mutation guards, and ran the combined working tree.

The final pre-commit run also exposed a configuration-test isolation gap:
inherited Azure settings overrode the test's dotenv fixture. The test now
clears every fixture variable and seeds conflicting inherited values to verify
that isolation without making external calls.

| Post-repair check | Result |
| --- | --- |
| `bash maintain.sh` | Exit 2 at the existing Accelerate/Lightning security findings; versions retained by explicit decision |
| C2PA soft-binding snapshot | Current |
| Ruff check and format | Passed, 278 Python files formatted |
| `pyright src/` | Passed, no errors or warnings |
| `pytest -n auto` | 2,358 passed, 3 skipped, 5 upstream warnings; exit 0 |
| Relative Markdown file links | Checked for file existence, no missing targets; heading anchors are not included in this check |

Remaining evidence limits are explicit: historical multi-period affine
calibration and external-negative labels need fresh evaluation before reuse;
the inconsistent D07/D19 source numbers remain withdrawn. Image GPU inference,
provider-oracle certification, and remote release execution were not part of
this repair verification. The canonical regression rules are in
[development invariants](../.claude/rules/development.md#audit-regression-contracts).

## Initial review scope and method

The review covered the package, maintainer scripts, published skill probe,
configuration, workflows, tests, and tracked Markdown documentation. Seven
read-only review agents divided the package, research scripts, current
documentation, and historical archive. The coordinating review independently
checked the findings against source and exercised the important failure paths
with synthetic inputs or controlled substitutes for model and service calls.

| Baseline inventory | Coverage |
| --- | --- |
| 61 package Python files, 25,112 lines | Read in full across three subsystem reviews and the coordinator |
| 96 maintainer Python scripts, 28,151 lines | Read in full across two script reviews |
| Published skill probe, maintenance shell, local typing stub, project configuration and workflows | Read by the coordinator |
| 60 tracked Markdown files, 24,948 lines | 59 read in full by the documentation reviewer; the 6,416-line historical archive read in full by a separate reviewer |
| 111 tracked test modules plus the existing untracked maintenance regression module | Complete test execution; source review depth varies as described below |

Package and script source files were read in full. Test coverage combined a
complete suite run with source review; some large test modules were reviewed
by their assertions and relevant fixtures rather than every fixture body.
Documentation review distinguishes current contracts from dated historical
observations. It does not revalidate every external paper, provider statement,
historical oracle response, or inaccessible experiment artifact.

No production code, dependency, model, external account, or release was changed
by this review. Existing maintenance changes were preserved. This document and
its link from the verification plan are the review's repository changes.

## Initial review verification results

| Check | Result |
| --- | --- |
| `uv run --no-sync ruff check` | Passed |
| `uv run --no-sync ruff format --check` | Passed, 271 files |
| `uv run --no-sync pyright src/` | Passed, no errors or warnings |
| `uv run --no-sync pytest -n auto` | 2,208 passed, 3 skipped, 5 upstream warnings |
| `uvx uv-secure uv.lock` | Exit 2, existing Accelerate and Lightning findings |
| Relative Markdown file targets at the start of review | 402 checked across 60 documents, none missing |

The vulnerability scanner's three advisory entries include two entries for
the same Lightning vulnerability. The decision to retain both dependencies
is unchanged; see [Known security-gate blocks](development.md#known-security-gate-blocks).
The complete maintenance script was not rerun because this review uses checks
that do not synchronize dependencies or rewrite source.

The green suite does not cover the failure cases below. GPU inference,
provider uploads, production deployments, and complete historical experiments
were not run. Model-dependent reproductions test the surrounding real code
with the model operation substituted, not the quality of generated pixels.

## Package findings

P1 means a failure to address before relying on the affected data/output
contract. P2 means a concrete correctness or contract problem to repair in
the corresponding subsystem. Line numbers refer to the review snapshot.

| ID | Priority | Location | Finding and evidence |
| --- | --- | --- | --- |
| C01 | P1 | [api.py](../src/remove_ai_watermarks/api.py), line 779 | An invisible-only batch leaves an existing output untouched when the current outcome is `no-signal` or `unavailable`. A synthetic red source and old blue output produced `processed=1`, `failed=0`, with the old blue output. The copy decision must follow the current outcome, not file existence. |
| C02 | P1 | [invisible_engine.py](../src/remove_ai_watermarks/invisible_engine.py), line 335 | With `output_path=None`, the input path has already been replaced by a temporary PNG. The inner remover writes there; `finally` deletes the result. A real outer-method reproduction returned a nonexistent path and left the original present. Resolve the destination before replacing the input. |
| C03 | P1 | [metadata.py](../src/remove_ai_watermarks/metadata.py), line 1253 | The ffmpeg metadata stripper omits explicit stream mapping. A real synthetic MKV containing two audio streams became a file with one stream. `-c copy` alone does not preserve all streams. Test media inventory and payload preservation, not only tag removal. |
| C04 | P1 | [metadata.py](../src/remove_ai_watermarks/metadata.py), line 1768 | JPEG output re-emits EXIF even with `keep_standard=False`. Real synthetic Artist, Orientation, and GPS tags survived `remove_ai_metadata(..., keep_standard=False)`. This contradicts `metadata --remove-all`. The test currently checks which branch runs, not which tags remain. |
| C05 | P2 | [metadata.py](../src/remove_ai_watermarks/metadata.py), line 409 | Compressed metadata is decoded only when the scan buffer reaches its size limit. A 264-byte PNG containing compressed AI XMP yielded no file-based signal, while the portable record recognized it. A complete compressed byte buffer still needs decompression. |
| C06 | P2 | [metadata_record.py](../src/remove_ai_watermarks/metadata_record.py), line 379 | Collection completeness is decided by `stat`, while subsequent read errors are suppressed. Calling the real collector on an existing directory returned `status=complete`, no issues, and empty metadata. Preserve read failure separately from an unknown provenance verdict. |
| C07 | P2 | [metadata.py](../src/remove_ai_watermarks/metadata.py), line 1258 | The default in-place call passes the same input and output to ffmpeg. A real MKV reproduction failed with ffmpeg's same-input/output error. Stage the remux and publish it after success. |
| C08 | P2 | [api.py](../src/remove_ai_watermarks/api.py), lines 498, 519, 757, 783 | `all`, visible batch, and invisible passthrough omit `display_tags_from` when writing. A real no-signal `remove_all` run lost the input ICC profile. The unchanged-orientation decode also requires correct orientation handling. Cover each public writer seam; the existing low-level writer test supplies the missing argument itself. |
| C09 | P2 | [chroma_zimage_pipeline.py](../src/remove_ai_watermarks/_internal/chroma_zimage_pipeline.py), line 153 | Chroma's installed `get_timesteps` rounds the interval start, unlike the SDXL formula used by `requested_steps`. At strength 0.17, 24 requested steps produce five effective steps, not the claimed four. The test checks the wrong formula for this pipeline. Recover the calibrated schedule before changing behavior. |
| C10 | P2 | [invisible_engine.py](../src/remove_ai_watermarks/invisible_engine.py), line 315 | `tile=True` still applies `max_resolution`, contradicting the parameter's documented bypass. A synthetic 2048x1024 input reached the substituted remover as 512x256 with `tile=True, max_resolution=512`, too small to engage default tiling. |
| C11 | P2 | [video.py](../src/remove_ai_watermarks/video.py), line 289 | The provenance predicate table omits Doubao. Its stable weak-confidence branch therefore never receives TC260 confirmation. A controlled real-arbiter comparison accepted 12 matching frames with confirmation and none through the public planning seam. Preserve vendor specificity when repairing the mapping. |
| C12 | P2 | [_text_mark_engine.py](../src/remove_ai_watermarks/_text_mark_engine.py), line 250 | Baidu declares `qwen_alpha.png` as a rival, but the mapping lacks that asset. The fallback returns Baidu's entire configuration, so the runtime compares against Baidu instead of Qwen. The actual resolver returned `baidu_alpha.png` for this rival. Aggregate detector impact has not been measured. |
| C13 | P2 | [text_draft.py](../src/remove_ai_watermarks/text_draft.py), line 141 | CJK recognition ignores `vertical_pad_ratio`; all three crop-jitter passes inspect the same crop. Consequently `accepted` does not establish the claimed crop stability for CJK. The corresponding test changes responses by call count rather than by the crop supplied. |

C09 predates the helper deduplication in the uncommitted maintenance changes:
the previous Chroma helper used the same compensation formula. Sharing it did
not establish that the two upstream pipelines have identical timestep semantics.
Do not repair C09, C11, or C12 by changing a constant or mapping without the
project's detector/output baseline checks.

## Research and evaluation findings

These findings concern development tools and the evidence they produce.
They are not claims that the installed package contains a SynthID pixel detector.

| ID | Priority | Location | Finding and evidence |
| --- | --- | --- | --- |
| R01 | P1 | [watermark_benchmark.py](../scripts/watermark_benchmark.py), line 528 | `_decode_video` reads ffprobe's width,height output into height,width and reshapes raw pixels accordingly. A real 16x8 clip decoded as `(1,16,8,3)` instead of `(1,8,16,3)`. This scrambles rectangular media, including the real-carrier crop used by temporal studies. Uniform-color fixtures can miss the error. |
| R02 | P1 | [smoke_matrix.py](../scripts/smoke_matrix.py), line 91, and [robustness_suite.py](../scripts/robustness_suite.py), line 93 | `expect_exit=None` accepts every process return code. The real smoke runner classified a substituted SIGSEGV return code, -11, as `pass`, despite call-site comments requiring crashes to fail. The robustness checker likewise accepts a negative code without a textual crash marker. Use an explicit acceptable outcome set and check the command reached the intended behavior. |
| R03 | P2 | [synthid_runtime_expert_scores.py](../scripts/synthid_runtime_expert_scores.py), line 77 | Registered fallback observations are always named registered-v3. A fine-opponent detector result was exported under the wrong expert name, losing its threshold and calibration identity. Preserve the returned detector and account for every supported expert in consumers. |
| R04 | P2 | [synthid_affine_lattice_probe.py](../scripts/synthid_affine_lattice_probe.py), line 1005 | Confirmation values participate in the `argmax` that selects the period. Holding selection values fixed and changing only confirmation changed the selected period from 17 to 16. The selected result is therefore not independently confirmed by that group. Use an untouched final group or revise and recalibrate the methodology. |
| R05 | P2 | [synthid_periodic_tile_probe.py](../scripts/synthid_periodic_tile_probe.py), line 114 | `score --register` carries NumPy integer shifts into `asdict` and JSON. A real numeric-score reproduction raised `TypeError: Object of type int64 is not JSON serializable`. |
| R06 | P2 | [synthid_periodic_tile_probe.py](../scripts/synthid_periodic_tile_probe.py), line 175 | Discovery supports partial edge tiles, but loading requires exact divisibility. Three synthetic 65x65 PNGs with 8x8 tiles produced a saved model that the same module rejected on load. Producer and consumer need the same geometry contract. |
| R07 | P2 | [synthid_pixel_probe.py](../scripts/synthid_pixel_probe.py), line 132 | Positive and cleaned geometry are validated separately. Incompatible lengths become NCC zero and then `carrier attenuated`. A controlled real-command reproduction with 64x64 positives and 32x32 cleaned inputs produced this false success. Report incomparable inputs instead. |
| R08 | P2 | [synthid_research_manifest.py](../scripts/synthid_research_manifest.py), line 200 | An external training negative without evidence fails as `source-evidence` but passes after changing only `verified_via` to `none` or a nonmatching provider. The real auditor accepted empty evidence and session fields in both cases. Require independent label evidence before admission to a final split. |
| R09 | P2 | [provider_oracle_web.py](../scripts/provider_oracle_web.py), line 301 | Results are persisted only after the complete browser batch returns. A later browser exception loses the already-settled earlier results from the manifest and leaves those rows pending. Persist each response immediately with its own timestamp; do not repeat already-completed uploads. Verified by control flow, without sending media. |
| R10 | P2 | [visible_positives.py](../scripts/visible_positives.py), line 62, and [pill_gate_audit.py](../scripts/pill_gate_audit.py) | A pool timeout exits through executor shutdown, which waits for unfinished workers before reaching fallback. The serial fallback also runs potentially crashing native code inside the parent process; catching `BaseException` cannot isolate a native process crash. The advertised timeout and crash containment do not hold. No actual process crash was induced. |
| R11 | P2 | [watermark_benchmark_report.py](../scripts/watermark_benchmark_report.py), line 426 | Video fidelity emits `mean_psnr_db`, but aggregation reads only `psnr_db`. A measured video row with finite PSNR 40 produced `finite_psnr=0` and no percentile. Preserve metric semantics explicitly instead of silently dropping the observation. |
| R12 | P1 | [videoseal_temporal_study.py](../scripts/videoseal_temporal_study.py), line 212, and [watermark_forgery_study.py](../scripts/watermark_forgery_study.py), line 146 | Several baseline/forged arms measure pre-encode arrays while recording the hash of a lossy encoded MP4. Attack/removal arms instead decode the recorded artifact. These hashes do not bind the observations to the measured bytes, and comparisons mix domains. Re-read every encoded artifact before measurement or label the pre-encode measurements separately. |
| R13 | P2 | [fidelity_metrics.py](../scripts/fidelity_metrics.py), line 170 | The metric called CER is normalized by the longer string rather than reference length. The current helper reports 0.75 for reference `a`, hypothesis `aaaa`; reference-normalized character error rate would count three insertions per reference character. Rename the metric or correct its definition and re-evaluate affected reports. |
| R14 | P2 | [audio cohort](../scripts/watermark_benchmark_audio_cohort.py), line 204, and [video cohort](../scripts/watermark_benchmark_video_cohort.py), line 368 | Recorded seeds do not consistently describe the generators: audio synthesis uses a carrier-derived seed while the manifest records the shared base; the video hard-negative manifest derives its seed from a different name than the generator uses. Have the producer return the executed recipe and record that exact value. |
| R15 | P2 | [visible_recall_sample.py](../scripts/visible_recall_sample.py), line 81 | Sampling deduplicates by dimensions plus detector confidence vector rather than image content. Distinct inputs with the same detector observations collapse into one row, biasing the sample. Content equality must be established independently of the detector being evaluated. |

The reference-length definition used in R13 is also documented by
[TorchMetrics](https://lightning.ai/docs/torchmetrics/latest/text/char_error_rate.html).

R01 and R12 mean that affected VideoSeal research results need a fresh run
against decoded, correctly shaped, hash-bound artifacts before they support
new conclusions. R04 and R08 similarly require an evidence audit of affected
research outputs, not only a green unit test after a code edit. No historical
metric has been silently recalculated or replaced in this review.

## Tooling and release findings

| ID | Priority | Location | Finding and evidence |
| --- | --- | --- | --- |
| T01 | P2 | [skill probe](../skills/remove-ai-watermarks/scripts/probe.py), line 246 | The advice marks video removal ready from image pixels plus ffmpeg without checking PyAV or the video diffusion runtime. With `pixels=ok`, `ffmpeg=True`, and `invisible=missing`, the real helper reported `video_visible=ok` and `video_invisible=ok_cpu_or_gpu`. These statuses need actual capability checks. |
| T02 | P2 | [distribute.yml](../.github/workflows/distribute.yml), line 124 | ClawHub is installed locally, then invoked as a bare executable without adding its local bin directory to PATH or using npm execution. A fresh runner with a configured token cannot rely on that executable being globally present. npm documents local executables under `node_modules/.bin`; use an explicit executable path or npm's execution interface. |
| T03 | P2 | [verify-release.yml](../.github/workflows/verify-release.yml), line 44 | Automatic verification selects current PyPI latest rather than the release handled by the triggering distribution run. Concurrent releases or manual distribution of an older version can verify the wrong version. Carry a version identity from the triggering operation. |
| T04 | P2 | [verify-release.yml](../.github/workflows/verify-release.yml), line 130 | The registry requirement check uses substring membership for `>=version`, without checking the package name or parsing the requirement. The real expression accepts both a different package at the same version and a longer version sharing the prefix. Parse the distribution name and version constraint. |

T02 was checked against the workflow and npm's primary
[executable-location documentation](https://github.com/npm/cli/blob/latest/docs/lib/content/configuring-npm/folders.md#executables).
Release workflows were not dispatched, and these findings do not assert a
particular past remote run failed.

## Documentation corrections

The following are current internal contradictions or disagreements with the
checked source. They are separate from the runtime fixes above.

| ID | Page | Correction needed |
| --- | --- | --- |
| D01 | [README](../README.md), line 268 | `--pipeline auto` currently selects Chroma only for Microsoft, not OpenAI. `_ENGINE_BY_VENDOR` is the source of truth. |
| D02 | [Installation](installation.md), line 122 | The `pixels` extra does not install the research SynthID carrier detector. Remove that capability from the extras table. |
| D03 | [Published skill](../skills/remove-ai-watermarks/SKILL.md), line 185; its [commands reference](../skills/remove-ai-watermarks/references/commands.md); [module internals](module-internals.md), line 1953 | The absolute claim that Meta output has no usable provenance conflicts with the shipped standalone-IPTC attribution and automatic routing. Distinguish files that retain the tag from stripped files. Synchronize the related source comments too. |
| D04 | [Python API](python-api.md) and [known limitations](known-limitations.md) | Visible-video lists omit Doubao. [Supported signals](supported-signals.md) also places Doubao and Seedance in the opposite order from the live specificity order. Derive or verify both membership and order. |
| D05 | [Photo model card](photo-classify-hf/README.md), line 52 | When Model 1 is DEFINITELY and 124-d features are unavailable, runtime retains `label=ai, provider=None`; the card incorrectly promises `unknown`. Its own diagram and tests show the actual behavior. |
| D06 | [Photo model card](photo-classify-hf/README.md), architecture and files sections | Some lists describe five provider heads while the current manifest and other sections include ByteDance as the sixth class. Synchronize the list with the operating-point artifact and code. |
| D07 | [Photo model card](photo-classify-hf/README.md), line 197, and [classifier research](ai-generated-image-classifiers.md), line 1481 | The printed percentage disagrees with the accompanying fraction. Recover the intended source values before editing either value; no replacement was inferred. |
| D08 | [Photo training](photo-classify-training.md), final paragraph | The stated outstanding metadata prerequisite contradicts the preceding correction that the necessary metadata is already available. Reconcile the actual remaining prerequisite. |
| D09 | [VideoSeal temporal evaluation](videoseal-temporal-evaluation.md), line 47 | The claim that halving frame rate preserved the message on every carrier contradicts the table's real-Veo value below the stated threshold. R01 and R12 also require rechecking the underlying run. |
| D10 | [Module internals](module-internals.md), line 367 | Hailuo's policy accepts confirmed provenance in code; the text says it rejects it. Keep this distinct from the missing Doubao mapping in C11. |
| D11 | [Known limitations](known-limitations.md), line 499 | The claim that metadata stripping still requires a lossless 16-bit PNG path is stale: `_strip_png_metadata_lossless` is already routed for that input. Do not confuse this with pixel inpainting's separate precision limits. |
| D12 | [External benchmark licensing](external-benchmark-licensing.md), line 45 | This repository is Apache-2.0, not MIT. This is a local license-description correction, not a new legal assessment of external datasets. |
| D13 | [Benchmark kernel](benchmark-kernel.md), state table | Forged does not universally imply expected `not_detected`: the AudioSeal presence rule can detect a foreign message. Explain expectations per adapter and message-verification contract. |
| D14 | [README](../README.md), line 489 | Video engine PSNR is computed after the input resize and before final encoding. The current text says it precedes the resize. |
| D15 | [Engine-selection fixtures](../data/evaluations/engine-selection/README.md), line 50 | The absence of a cleared Microsoft fixture is no longer the right prerequisite: a licensed fixture is now tracked. A new paired oracle result is still a separate requirement. |
| D16 | [Historical SynthID archive](synthid-detector-removal-plan.md), lines 5874-5926 | The reported registered-v3 false-positive confidence bound combines the controls used to select the v3 gates with a subsequent holdout. Only the untouched evaluation set supports an independent bound. This is distinct from the code-level confirmation leakage in R04. Recompute the bound with the eligible denominator. |
| D17 | [Historical SynthID archive](synthid-detector-removal-plan.md), line 3692, and [detector research](synthid-detector-research.md), line 33 | Normalized residual correlations are interpreted as absolute carrier amplitude. The stated amplitude ratio does not follow from the reported correlations, and normalization discards the absolute scale. Retain the measured correlation statement without that unsupported inference. |
| D18 | [Historical SynthID archive](synthid-detector-removal-plan.md), line 4851 | A crop at half the width and half the height is described as half the area. It retains one quarter of the area; a later archive entry correctly describes the linear scale. |
| D19 | [Classifier research](ai-generated-image-classifiers.md), lines 755-757 | Baseline, candidate total, and paired gains/losses are arithmetically inconsistent. The candidate total must equal the baseline plus improvements minus regressions. Recover the source row before choosing which value to correct. |

The public documentation also needs a focused removal of local operational
details in [data layout](../data/README.md), lines 35-42, and the
[historical archive](synthid-detector-removal-plan.md), lines 4305-4306, as well
as the product-specific passage in [classifier research](ai-generated-image-classifiers.md),
lines 726-727.
Those details are deliberately not repeated here. Preserve only the public
fixture policy and the technical experiment conclusions.

Lower-priority drift remains in dated plans: some completed smoke, full-clip,
audio, and forgery checks are still listed as absent or future work. Keep the
historical chronology, but make current status pointers agree with the live
tests and newer studies.

## Original repair order

Start with C01-C04, R01-R02, and R12: they lose output/data, retain explicitly
removed metadata, or turn failed or mismatched validation into successful evidence. Add
regressions at the public seam using the synthetic scenarios above and
mutation-check that those tests distinguish the broken behavior.

Then repair the remaining subsystem contracts and their documentation
together. Detector, video, and calibrated pipeline changes still require
the parity and output checks in the development rules. GPU/oracle calibration
and historical evidence repair remain distinct from ordinary unit testing.
Run the full gate after fixes, and report core checks separately while the
documented security block remains.
