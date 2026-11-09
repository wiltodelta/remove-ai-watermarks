---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
  - "scripts/**/*.py"
  - "skills/**"
  - "pyproject.toml"
  - "uv.lock"
  - "maintain.sh"
  - ".github/workflows/*.yml"
  - "README.md"
  - "docs/cli.md"
  - "docs/python-api.md"
  - "docs/provider-oracles.md"
  - "data/fixtures/visible/**"
  - "data/evaluations/**"
description: Command contracts, provider-oracle limits, local gate, typing boundaries, model-adjacent and seam tests, the docs-coverage seam, detection and video-path proof, and why cloud GPU harnesses stay out of scripts/.
---

# Development invariants

## Command contracts

Every single-image command declares `source` with `dir_okay=False`; `batch` declares its directory with `file_okay=False`. Keep `tests/test_cli_robustness.py::TestDirectoryInputIsRejected` as the regression guard.

Exit-code and no-signal behavior is a public contract; read the command-line section of [`../../docs/module-internals.md`](../../docs/module-internals.md) before changing it. The dev `.venv` always carries the pixel stack, so it cannot show what the default install does. Check a no-signal or fail-closed change in a throwaway venv outside the repository: `d=$(mktemp -d) && uv venv "$d" && VIRTUAL_ENV="$d" uv pip install . && "$d/bin/remove-ai-watermarks" ...`.

`tests/test_agent_skill.py` compares the skill's prose against the live Click tree. Raise `MIN_CLI_VERSION` in `skills/remove-ai-watermarks/scripts/probe.py` whenever a reference names a command, flag or value the previous release lacked.

Do not add an option whose only outcome is an error. Model id, step count and CFG are fixed by the profile, so none is a parameter of the CLI, `InvisibleEngine`, or `WatermarkRemover`; if a value cannot vary, delete the knob rather than validating it.

`device` is the deliberate exception: a library parameter (`None`/`"auto"` detect, `"cuda"` pins without detecting, anything else raises at construction), absent from the image-path CLI, and a `--device` flag (cuda, mps, cpu) on `video invisible`, `video all` and `video batch`. `test_device_exists_exactly_where_the_skill_says_it_does` pins that split; do not delete the video flag.

Install hints name the extra that actually makes the command work (`qwen-zimage`, not `diffusion`) and print the command shell-quoted.

## Provider oracles

Provider verification services and provenance APIs are development-only oracles. Keep their SDKs in the development extra and their adapters out of the installed CLI, top-level Python API, and runtime detection/removal pipelines.

The entry point is `scripts/provider_oracles.py`; workflow, slot schema, surface order and per-surface behavior live in [`docs/provider-oracles.md`](../../docs/provider-oracles.md). Read it before running or changing a provider check.

- Select each named slot explicitly from the gitignored `.oracle-slots.json` and snapshot it into a separate immutable batch manifest. Never commit the slot file.
- `check-openai` and `check-microsoft` make one submission per call; `run-web` submits each unrecorded row of one prepared manifest once. After a refusal, quota response, transport error, or indeterminate verdict, record the exact response and continue with the next slot or surface as its own command.
- Deviation from the global rules: do not ask for confirmation before the switch, because anyone doing the task holds standing permission to run the configured checks.
- Do not add new logins, unbounded retries, or address rotation inside one call; do not walk a prepared route list, and do not merge results from different sessions into one response.
- Bind every manual result to the exact sanitized upload hash and preserve the provider response verbatim. Keep each surface's verdict separate.
- Only `gemini-web`, and the `synthid-web` sign-in fallback, use the user's existing real Chrome; every other Web surface uses an isolated Playwright context that never attaches to or copies state from that browser. `synthid-web` terms forbid bulk verification and quota evasion: submit the prepared rows once and never start a sign-in.
- Credentials never enter a manifest. Proxy URLs and OpenAI keys are read only through a slot's named environment variable (`proxy_url_env`, `api_key_env`); Microsoft resource keys are read into memory through authenticated Azure CLI from the resource identities in `.env`. API transports keep `trust_env=False` and take no `proxy_url_env`.

## Local gate

The full `maintain.sh` run outlasts the agent's 10-minute foreground shell cap: run it detached with its exit code written to a file and poll that file. Stop a stray run by its own PID, never `pkill -f pytest`.

`tests/test_maintenance.py` checks clean, vulnerable, and broken scanner runs through the real shell entry point and pins the CI `security` job to the same scanner line with no `continue-on-error`. Do not revive the old text-matching bypass.

Boundary modules for cv2, Torch, and Diffusers may carry narrow per-file relaxations for unknown third-party types. Keep pure-logic files strict, preserve the local piexif stub, and fix real errors before widening a pragma.

From a worktree without its own `.venv`, `uv run` imports the package from the main checkout, where the editable install points. A script measuring a worktree's edit must assert `module.__file__` resolves inside that worktree (insert its `src` at `sys.path[0]` when it does not).

## Research harnesses and classifier training stay out of this repository

Keep any research script that needs a cloud GPU account outside this repository; `scripts/` carries no `modal` import, and `test_scripts_carry_no_cloud_gpu_harness` is the canary. Leave only the offline analysis step here, as `scripts/analyze_engine_selection_study.py` is. Do not mechanize the identifier half of the boundary audit in a tracked file: a tracked denylist publishes the strings it exists to keep out.

Classifier training scripts and local training caches/datasets live exclusively in the private `wiltodelta/raiw-classifier-training` repository (with `images/`, `data/`, `.npz`, `.joblib`, and `.pt` gitignored); this repository keeps only runtime inference (`classify.py`, `source_classify.py`) and Hugging Face Hub publishers (`scripts/publish_photo_classify_hf.py`, `scripts/publish_source_classify_hf.py`).

## Visible-mark example gallery

Every registered mark carries a committed `data/fixtures/visible/<key>/example.png` (image marks) or `example.mp4` (video marks), regenerated by `scripts/render_visible_examples.py` and guarded by `tests/test_visible_examples.py`. When generation fails after a geometry or gate change, fix the generator and the engine together; never hand-edit the binaries.

Visible-mark content must distinguish the registry key, the provider's canonical product name (from the provider's official source, never an asset filename or OCR text), and the literal text rendered in the mark whenever those differ.

## Model-adjacent tests

Do not classify an entire module as untestable because its main path downloads a model, and use availability checks only for paths that actually load large models. Keep pure behavior covered without downloads, including:

- target-size selection in `test_invisible_engine.py`;
- unsharp and adaptive-polish helpers in `test_humanizer.py`;
- tiling geometry and blending in `test_tiling.py`;
- prompt-embedding cache keying, storage round-trip, and the cross-pipeline reuse that lets a stack load without its text encoder, in `test_qwen_zimage_pipeline.py`;
- the face stack's dtype in `test_qwen_zimage_pipeline.py`: a subclass that changes the pipeline dtype must not change an inherited stage's. When one profile inherits another's stage, guard the invariants that stage relies on;
- the photo-classify gate in `test_classify.py`: DEFINITELY is ridge AND MLP, POSSIBLY does not emit a provider, and `identify` does not import `remove_ai_watermarks.classify`;
- the source-export classifier in `test_source_classify.py`: the Hub revision and artifact hash are immutable, the pickle-free schema and exact GELU are validated, threshold mutation changes the real decision, and `identify` never consumes the result as watermark evidence;
- the `InvisibleOptions` defaults in `test_api.py`, compared field by field with the signature they mirror and without an exception table: a field that needs one belongs elsewhere.

Keep `test_every_field_arrives_at_the_engine_with_the_caller_s_value`, which sets every `InvisibleOptions` field off its default and asserts each value arrives; a defaults comparison does not replace it.

Guard every path a knob takes to the engine: `force` goes through `remove_all` and `remove_batch(mode="all")`, `controlnet_conditioning_scale` through `_run_invisible` and `_batch_engine`. The mode-parametrized guards are `TestRemoveBatchLibrary::test_force_reaches_the_scrub_gate_in_every_scrubbing_mode` and `TestBatchCommand::test_batch_controlnet_scale_flows_to_the_cached_engine`.

A verdict branch must be reachable with the dependency stack as shipped. No C2PA trust anchor list ships, so `signingCredential.trusted` never appears; `TestIdentifyRealSamples::test_no_committed_fixture_reports_a_trusted_signer` asserts that against the committed fixtures. Read a capability the stack does not have as a missing input, never as a negative finding.

## The docs are a seam

`tests/test_docs_cover_the_public_surface.py` reads the live Click tree and `InvisibleOptions` and requires every knob to be named in the user-facing pages. Group option spellings by the Click parameter, never by a `--no-` prefix: a pair such as `--strip-metadata/--keep-metadata` is documented by either spelling. The same module requires every link and image in `README.md` to use an absolute `https://` URL (`raw.githubusercontent.com` for repo images, `github.com/.../blob/main/` for repo files) whose target exists on disk, because PyPI renders `README.md` at `pypi.org/project/remove-ai-watermarks/` without rewriting relative paths.

## One measurement, one gate seam

A detector is split into a trust-level-blind scan and a verdict that applies the threshold, so `detect` and `detect_both` reach the same numbers by construction.

- Put a per-mark demotion in the `_post_gate` hook (or, for a whole-scan precondition like LibLibAI's size floor, in `_scan`), never in a `detect` override, which `detect_both` cannot see. `TestSinglePassPerception` asserts `detect_both` equals two `detect` calls field for field.
- Detection and the removal mask read one sweep: the winning box travels on `TextMarkDetection.match_box` and the registry threads the detection into the mask builder. Never let a mask path re-run its own sweep.
- When the research lattice runtime in `scripts/synthid_runtime/` routes by image geometry, preserve the returned `SynthIDDetection.detector` in score manifests and downstream routers. Record an inactive expert as explicitly unsupported; never attribute a routed large-image score to the fixed expert. This expert is not part of the public package.

The C2PA manifest-store JSON is not stable across reads (the reader regenerates manifest URNs and instance ids). Compare the derived `c2pa_info`, never the raw store.

`collect_metadata_record` -> `evidence_from_metadata_record` -> `identify_from_evidence` must equal `identify(path, check_visible=False, check_invisible=False)` on the same image, and can break from either side. `tests/test_metadata_record.py` pins it over the tracked fixtures; when changing either side, also re-run the local evaluation corpus.

Before changing anything in the detection path, record the detectors' exact verdicts over a local sample and diff them after. A refactor is correct only if that record is byte-identical; a change meant to fix detection must change exactly the files named in advance. When the baseline is a published release, get it with `uv run --isolated --no-project --with 'remove-ai-watermarks[heif]==<version>' python <script>`.

In `dwt_dct.py`, the bit test `peak % 36 > 18.0` lands exactly on the threshold for uint8 input, so a 1-ulp difference flips real bits: leave the transform to `pywt` however slow it looks. `tests/test_invisible_watermark.py` skips the upstream-parity test without the `detect` extra, so the verdict record is required here.

## A certified operating point is data, not a constant

`test_shipped_defaults_match_a_certified_manifest_row` derives the video SynthID default from `data/evaluations/video-synthid-oracle.csv`; never restate its literals. Anything that produces oracle evidence loads through `load_video_vae_runtime`, which gates the latent scaling factor and carries it on `VideoVaeRuntime` into encode and decode.

Prove a video-path refactor by building a clip from a tracked fixture with ffmpeg and requiring an identical output sha256 before and after. Keep the generated media outside the repository, and run both arms in one synced environment: a full-extras `uv sync` alone changes re-encoded H.264 packet bytes.

Frame sampling compares `timestamp + 1e-9 >= next_sample_time`, so a `>=`-to-`>` mutant survives; mutate the phase or the period instead, which `test_pairing_follows_the_engine_sampling_rule_not_just_the_frame_count` catches.

## Audit regression contracts

Image output paths must describe the artifact published by the current call, including skipped invisible stages and in-place writes. Carry explicit EXIF decode state through image staging; equal dimensions do not prove that a mirror or 180-degree rotation was applied. Metadata collection must propagate late-read failures into an incomplete record, and stream-copy metadata removal must map every input stream before atomically publishing the output.

Research observations must describe the decoded, hash-bound saved artifact. Keep period selection independent of confirmation patches and require explicit negative evidence. Use the metric's actual denominator in its name; normalized edit distance is not reference-normalized character error rate.

Tests that reload a module after changing environment variables must restore the environment before reloading the original state, or a temporary cache path survives teardown and later model tests silently attempt downloads. Dotenv configuration tests clear every fixture variable and seed conflicting inherited values to prove isolation.

Cross-platform tests use host-absolute synthetic interpreter paths and explicit unchanged decodes when comparing stored pixels. Execute Ubuntu Bash workflow harnesses only on POSIX hosts with Bash; keep their pure Python checks active on every platform. Round floating-point image output before uint8 conversion when quantization must preserve constant values across OpenCV builds.

OpenCV 5 draws Hershey `putText` glyphs differently from 4.x, so a calibrated template drawn with `putText` ships as a committed PNG whose recipe refuses any OpenCV but 4.x (`scripts/render_video_hershey_templates.py`, `scripts/build_openart_alpha.py`); never render one at import time.
