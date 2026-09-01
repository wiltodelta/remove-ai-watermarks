# Maintainer scripts

These scripts are development and evaluation tools, not installed CLI commands. Run
them from the repository root with `uv run python scripts/<name>.py --help`. Inputs
under `.local-eval/` and generated reports remain untracked unless a data README
explicitly names a tracked canonical result.

## Audits and release checks

| Script | Purpose |
| --- | --- |
| `corpus_gap_scan.py` | Compare a local image corpus with the library's `identify` results. |
| `detection_timing.py` | Record per-method metadata and verdict timings. |
| `detection_timing_report.py` | Aggregate timing records by method and segment. |
| `fidelity_metrics.py` | Compute objective image-fidelity metrics for paired outputs. |
| `invisible_quality_audit.py` | Pair originals and invisible-removal outputs for quality review. |
| `metadata_removal_audit.py` | Check metadata detection/removal parity over a corpus. |
| `pill_gate_audit.py` | Measure the Jimeng pill detector on the product path. |
| `real_examples_e2e.py` | Run end-to-end confidence checks over local examples. |
| `record_parity_audit.py` | Compare record-based and file-based identification. |
| `resource_ceilings.py` | Measure peak RSS and runtime for fill backends. |
| `robustness_suite.py` | Exercise CLI failures on adversarial and degenerate inputs. |
| `sidecar_regression.py` | Compare current identification with recorded sidecars. |
| `smoke_matrix.py` | Exercise CLI parameter choices on real local data. |
| `video_fidelity_probe.py` | Compare delivered video fidelity with its source. |
| `visible_eval.py` | Benchmark registered visible-mark detectors. |
| `visible_removal_audit.py` | Audit visible-removal results over a local corpus. |

## Calibration and corpus preparation

| Script | Purpose |
| --- | --- |
| `contentseal_transforms.py` | Reproduce and hash-check deterministic Content Seal variants. |
| `detector_response.py` | Measure detector response over mark size, contrast, background, and aspect. |
| `fill_quality.py` | Measure visible-fill quality against constructed ground truth. |
| `ladder_headroom.py` | Measure recall cost from the coarse scale ladder. |
| `registered_mark_calibrate.py` | Measure a registered detector without conflating visual positives, metadata cohorts, adjudicated negatives, and unlabeled controls. |
| `synthid_corpus.py` | Ingest and inspect the local SynthID reference corpus. |
| `vendor_cohort_harvest.py` | Partition TC260 carriers by producer code. |
| `vendor_mark_calibrate.py` | Calibrate a candidate vendor text detector. |
| `visible_alpha_solve.py` | Rebuild visible-watermark alpha assets from controlled captures. |
| `visible_groundtruth.py` | Consolidate blinded contact-sheet labels into ground truth. |
| `visible_positives.py` | List corpus images carrying a registered visible mark. |
| `visible_recall_sample.py` | Build an unbiased blinded sample for recall measurement. |
| `visible_sheets.py` | Build blinded contact sheets for relaxation candidates. |
| `publish_photo_classify_hf.py` | Upload the photo-classify card and freeze weights to Hugging Face `wiltodelta/raiw-models`. Manual; the Action `publish-photo-classify-hf.yml` is the write-token path. |

## Research and diagnostic prototypes

| Script | Purpose |
| --- | --- |
| `cjk_tail_probe.py` | Test a generic template for otherwise uncovered CJK labels. |
| `controlnet_sweep.py` | Sweep the historical ControlNet removal prototype. |
| `infer_text_lines.py` | Draft stable source-text lines without modifying pixels. |
| `qwen_scrub_prototype.py` | Probe low-strength Qwen regeneration on a GPU. |
| `selective_text_restoration.py` | Evaluate text restoration over a scrubbed image. |
| `synthid_pixel_probe.py` | Run the experimental local SynthID carrier probe. |
| `video_synthid_sweep.py` | Build oracle-gated video regeneration candidates. |

## Generated assets

| Script | Purpose |
| --- | --- |
| `render_pill_silhouette.py` | Render the synthetic Jimeng pill silhouette. |
| `render_vendor_silhouettes.py` | Render synthetic vendor text-mark silhouettes. |

## Shared helpers

`_plain_console.py` provides plain-text fallbacks for Rich output, and
`_text_eval.py` contains normalization helpers shared by text-evaluation scripts.
