# Text watermark validation

This work does not establish market leadership or removal of a private provider's
production watermark. Read the [coverage catalog](../data/evaluations/text-watermark/catalog.json)
for the reviewed provider/model cases, independent reference schemes, source URLs
and explicit access gaps. It is a source-backed inventory, not an exhaustive list
of every watermark paper or an assertion that every entry passed a live test.

## Published-format fixtures

The deterministic cleaner recognizes the [published Windows AI text sequence](https://learn.microsoft.com/en-us/windows/ai/apis/text-watermarking)
`U+200B U+034F U+2060 U+200B` as `windows_ai_text_watermark`. It removes complete
occurrences before classifying neighboring Unicode controls, preserving valid
emoji and joining-script context. The sequence is forgeable, so this signal does
not attribute the source to Microsoft. Tests cover synthetic published-format
fixtures; actual Windows AI output has not been generated in this environment.

### Independent C2PA text replay

`scripts/benchmark_c2pa_text.py` reads the external encoder's published
vectors and verifies that its decoder recognizes each marked input and
rejects its clean control. It reports exact byte recovery, preservation
of clean inputs, idempotence and reference-decoder residue. The two
supplemental Markdown examples use exact byte recovery because their
quoted manifest is literal content.

Use the [c2pa-text reference revision](https://github.com/encypherai/c2pa-text/tree/7e818755818281748c41aef8766037b30ce9fb04)
and, for the optional comparison, [dewatermark 0.8.0 at the tested revision](https://github.com/cyzanfar/text-watermark-remover/tree/fd620a93b9d96c7a4730d80be2d7aa90d5017575).
Run from the library checkout with the reference source already present
locally.

```bash
uv run --with requests python scripts/benchmark_c2pa_text.py \
  --reference-root ../c2pa-text \
  --dewatermark-root ../text-watermark-remover \
  --output ../c2pa-text-results.json
```

The library recovered all seven publisher originals and both Markdown
examples. The dewatermark safe and aggressive profiles recovered zero
and two of the nine originals, respectively. All three configurations
preserved each clean original. These results describe the tested C2PA
carriers and the two exact Markdown cases.

### Exact Unicode and serialized recovery through API and CLI

`benchmark_text_exact.py` has an independent synthetic Unicode panel in the
script. It inserts the published Windows sequence next to prose that contains
joining scripts, emoji, bidi isolates, decomposed text, numeric multiplication
with word joiners, and a fenced CRLF example. Results require byte-identical
recovery through `remove_text_all` and the CLI's top-level `all`. The harness
also checks the clean inputs and source-file preservation. The math control
covers the supported numeric multiplication form. Letter-based equalities
with WORD JOINER fall outside that preservation rule.

```bash
uv run python scripts/benchmark_text_exact.py --output ../unicode-exact.json
uv run --with requests python scripts/benchmark_text_exact.py \
  --reference-root ../c2pa-text --output ../text-exact.json
```

The second command adds the published C2PA vectors and the two fenced
Markdown examples as a separate report group. Reports contain input and
output hashes and fail the command on mismatches, clean-control changes,
non-idempotent API output, or residue found by the reference decoder.

## Partial regeneration

Partial processing is opt-in. Rejected paragraph clusters remain unchanged and
are counted explicitly. A whole-document negative after partial processing is
insufficient: inspect retained spans and test segment-level detection before
claiming complete removal. Fact-preservation heuristics are not semantic proof. The shared
benchmark assessor labels a global negative with retained prose as
`partial_unverified`. A frozen sentence-model trial demonstrated the failure
case directly: the whole output was below threshold while an unchanged paragraph
remained detected.

## Evidence rules

A useful trial needs a matched generator, tokenizer, key, configuration and
detector implementation. Record the literal source, an unmarked control, the
returned output, hashes, model revisions, device and raw detector scores. A
matching integer key is insufficient when hashing, vocabulary partitioning or
random tables differ. CPU and MPS sampling tables can differ with the same seed.

Count separately: an undetected source, a falsely detected unmarked control,
a runtime failure, a refusal, a retained mark and an accepted output below the
reference threshold. A rewritten passage is not automatically a removed watermark.
A single negative control does not calibrate a false-positive rate. Small studies
do not support cross-language or long-document guarantees.

The frozen study and expanded runs use synthetic local-model outputs. They are
not factual reference material: generation prompts can ask for facts the source
model fails to include. Compare rewrites to the saved source, not to an ideal
answer. Numeral/entity/polarity checks are heuristics. They do not establish full
semantic equivalence, detect every invented name or verify quantity-to-entity
relationships. Human quality review remains necessary. Spelled-number normalization
covers selected whole English and Russian cardinal sequences, not every language
or number expression. Whole-
paragraph NLI/LLM judgments can miss category narrowing even when short synthetic
pairs receive the expected labels. The writer and a fresh-context critic using
the same weights are not independent semantic verification.

Normalization checks are required when formatting changes lower a detector score.
Retest a declared canonicalized form and preserve both raw results. If a trivial
inverse transform restores the exact positive source, report parser sensitivity
rather than robust watermark removal.

## Meaning-preservation calibration

The optional `rewrite_validator` callback supports separately evaluated acceptance
policies inside the library's retry loop. It is not a semantic guarantee. A local
dependency-parser/WordNet experiment caught visitor-to-tourist narrowing, an added
child-age modifier and selected actor-role inversions, but still accepted other
meaning changes and rejected valid paraphrases. It was not adopted as the default.
Tests cover strict boolean verdicts, retries, visible protected content, literal
placeholder collisions, empty-output refusal and unchanged destination files.

The [synthetic fidelity pairs](../data/evaluations/text-watermark/fidelity-cases.json)
exercise quantities and bounds, modality, exclusivity, actor roles, causal evidence,
metrics, service scope and residency versus citizenship, including English,
Russian and selected Bulgarian cases.
They calibrate an acceptance judge, not a watermark detector. Track false accepts
and false rejects separately; no NLI or LLM judge is treated as semantic proof.

The corpus records its development/previously-held-out/discovered-failure history.
The former holdout is now public and cannot serve as fresh evaluation after tuning.
A Russian literacy pair was clarified after an ambiguous original wording; earlier
measurements do not evaluate the revised pair. Preserve that distinction when
comparing evidence. The mobility positive now retains the reader category, and
duration-expansion/supervision-omission pairs were added after an AliMark trial.
Neither revision creates fresh held-out evidence. Generated candidates still need
source-level review.
The Q64 sentence trial added split-name and owner/employee pairs after observed
failures; these are also examined development evidence. The stronger-inserter
PostMark trial adds supplied-object roles, cleanup timing, plan realization and
optional-demonstration pairs after observed failures. Older accuracy measurements
do not evaluate this revised input hash. Selected AutoMark candidates add
examined season loss/range and maintenance/visit frequency pairs; the seasonal
backstop does not claim to preserve every frequency, actor or event binding.
The visible-names follow-up adds examined reopening, oldest-harvest and
visitor-beneficiary pairs. These labels record failures; they do not imply that
the current fact checker detects them. Earlier corpus measurements do not
evaluate this revised input hash.

A separate [MiniCheck-Flan-T5-Large](https://huggingface.co/lytang/MiniCheck-Flan-T5-Large)
trial used the unchanged author sentence-support inference and document chunking.
It checked every English calibration pair, selected examined full-output failures
and their identity controls. The equivalence adaptation takes the minimum support
over every hypothesis sentence in both directions, without truncating inputs.
It still accepted role, scope and quantity-strength changes in full documents;
a higher cutoff also rejected correct paraphrases and identity controls.
An end-to-end `rewrite_validator` trial refused the frozen PostMark source after
retries rather than producing a qualified improvement. This checker was not
adopted as a default. Russian pairs were outside its declared English scope,
and these examined inputs do not establish fresh held-out performance.

Russian cardinal regressions found after that trial add examined calibration
pairs for whole values, scaled quantities, bounds and duration adjectives.
The MiniCheck run's archived corpus hash predates those additions.
Whole English cardinal, signed-value, exponent and literal-address pairs were
added after a subsequent numeric-parser review. They are examined regressions,
not fresh model evaluation; earlier judge metrics do not cover the new hash.

## Reproducible local NLI support benchmark

`benchmark_text_fidelity.py` evaluates an explicitly supplied local sequence
classifier on the synthetic pair corpus. Install the `text` extra for the local
Torch/Transformers runtime. No models are downloaded by this script. It reads the
exact `entailment` label from the checkpoint, refuses missing/unexpected weights
and keeps both directions. It reports whole-pair support and the minimum support
for every regex-delimited sentence against the complete opposite passage.
The sentence splitter is a diagnostic approximation, not a linguistic parser.

```bash
uv run python scripts/benchmark_text_fidelity.py \
  --model-dir /path/to/local-nli-checkpoint \
  --cases data/evaluations/text-watermark/fidelity-cases.json \
  --output-dir /path/to/new-run --device cpu --max-pair-tokens 512
```

The output directory must be new. `benchmark.json` retains complete paired input
tokens and probabilities, model-file/corpus/script hashes, versions and input limits.
It marks oversized complete inputs as `input_window_exceeded`, performs no
truncation or inference on them, and exits nonzero when any pair is unscored.
A saved partial run has `completed=false`; missing or incomplete reports must
not be counted as completed evaluation. Custom case arrays need unique IDs, complete `source`/`candidate`
strings and `equivalent=true|false|null`; null means an unlabeled diagnostic.
The script neither changes library acceptance nor detects a watermark.

A frozen follow-up used the author checkpoints
[BGE-M3 zeroshot v2.0](https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0)
revision `9abf1c8aaeb82a2447809c20753ed0b106b76652` and
[its -c variant](https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0-c)
revision `705510dfe0f35d223a4bf6f5d2695ec5fd162341`.
The corpus hash was
`1127fa780fd401898fe950cd2cfad986cfc97b41d9cc0b4bc2588483e0d291f3`.
The declared gate at 0.9 required no false accepts on the examined negatives and
at least 80% acceptance of the examined positives. Neither checkpoint passed.

| Checkpoint | Threshold | False accepts / 71 incorrect pairs | False rejects / 71 correct pairs |
| --- | --- | --- | --- |
| v2.0 | 0.9 | 9 | 13 |
| v2.0-c | 0.9 | 16 | 15 |
| v2.0 | 0.99 | 0 | 41 |
| v2.0-c | 0.99 | 2 | 35 |

These are examined development results, not held-out or production accuracy.
Combining the selected lexical fact helper with NLI at 0.9 still accepted seven
and twelve incorrect pairs respectively; that helper is not every local-generation
acceptance guard. Full-output and source-identity controls were recorded separately.
Sentence-wise scoring caught some errors missed by whole-pair scoring, but still
accepted intended allocation becoming dispatch and changes in temporal strength.
It also rejected one full-source identity control with the v2.0 checkpoint.
The 0.99 cutoff that eliminated short-pair v2.0 false accepts still accepted an
observed full-output temporal-strength change.
The checkers were not enabled by default.

All inputs were tokenized without truncation at an explicit 2048-token paired
limit, below the recorded XLM-RoBERTa position capacity. The serialized tokenizer
metadata declared 512; the experiment records that discrepancy rather than silently
using its default. The actual largest input was 757 tokens. Float32 eager MPS
inference, the original protocol, raw probabilities, independent script reruns,
identity controls and input-window/overwrite regressions are retained. Increasing
a score threshold does not establish semantic equivalence or watermark removal.

## Visible-name masking experiment

A paired local experiment kept recognized proper names visible while retaining
numeric and technical-span protection, the same prompt, all existing guards,
seed and default Qwen2.5-1.5B revision. It used the frozen AliMark, PostMark and
two selected AutoMark sources. Current masking returned one full output; visible
names returned three. Every returned output failed source-level fidelity review.
Examples include reopening becoming opening, visitors becoming villagers, the
oldest harvest becoming an older harvest, and added donation motivation.
PostMark returned byte-identical faulty output in both modes. RotationFirst
refused in both modes.

The experiment was not adopted. Increased acceptance did not establish improved
usable removal. No new detector replay was run because none of the returned
outputs qualified on fidelity. These are four examined synthetic sources at one
seed, not fresh held-out evidence or a provider/model coverage expansion.

## Sentence-model alternatives

The reviewed [English T5 checkpoint](https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base)
and [Russian ruT5 checkpoint](https://huggingface.co/cointegrated/rut5-base-paraphraser)
were tested on the same frozen SynthID source/control files through the library's
protected-span API. The experiment split sentences, rejected input-window overflow
instead of truncating, and compared preservation-oriented beam search with an
input-ngram repetition constraint. Model revisions, exact profiles, all candidate
outputs and matched scores are retained with the run.

These trials do not justify a default-model change. Observed outputs still narrow
visitor categories and invent a completion of a truncated source. A faster model
or a lower whole-document score does not resolve those quality failures. Retained
paragraphs and their diagnostic scores must be reported separately.

## Library-native MLX trial

The optional Apple silicon `text-mlx` backend uses the same protected-span, fact,
script, extent and caller-validator checks as Transformers. Streaming output must
end with an explicit stop reason; truncated or unknown completions are refused.
The backend is an execution option, not a stronger semantic acceptance policy.

A frozen local trial used `mlx-community/Qwen3.8-27B-4bit` revision
`b935172e4bb5dcd0543774a34748fb92422e3027`, MLX 0.32.4 and mlx-lm 0.32.2.
It rewrote the previously examined English AliMark Q64 source and Russian HF
SynthID source through the library, without detector-based candidate selection.
AliMark's author score fell from 3.0363935 to 2.6949550, still above the original
local-control threshold 1.2543291. The Russian matched mean-g diagnostic fell
from z=10.3279812 to 2.4924461: below the diagnostic cutoff 4 but above 2.3263.
Neither cutoff establishes a calibrated production or provider verdict.

The larger model still weakened qualifiers, including "many decades" to "decades"
and "ensuring everything was well-maintained" to "maintaining the facility".
The Russian output included a grammatical number error. Whole-document fidelity
is not established, and the default model remains unchanged.

A separate frozen punctuation experiment joined pairs of complete sentences with
semicolons while preserving the exact word sequence. Its raw AliMark score was
0.5410784 at a separately derived threshold 1.0736543, with one false positive in
32 evaluation controls. Replacing semicolon-space with period-space recovered
the exact original positive text and author score 3.0363935. This falsifies a
robust-removal interpretation of the raw negative. The three-sentence variant
failed on an unchanged evaluation control and has no completed result. Neither
punctuation strategy was adopted as a removal feature.

## Published removal methods and applicability

- [DIPPER](https://github.com/martiansideofthemoon/ai-detection-paraphrases) is a
  specialized discourse paraphraser. Its author-published GPT-2 XL completions
  and saved paraphrases provide external evaluation inputs. A fixed-index replay
  used the pinned author's NumPy-shuffle KGW functions, original prompt token and
  GPT-2 tokenization. This is distinct from the MarkLLM KGW implementation.
  Full completions and the author's minimum-200-token matched windows are kept
  separate. Some selected default-profile paraphrases retain the diagnostic
  signal. Reviewed candidates also omit a number, change participant roles and
  invent a completion of unfinished text. Fresh default-library runs refused the
  preselected source texts; none yielded a full output to qualify as removal.
  These are saved-data replays, not fresh GPT-2 XL/DIPPER generation, a recovered
  original checkpoint or an exhaustive model benchmark. Local model assets were
  inspected but not downloaded because they exceed the available disk budget.
- [SIRA](https://arxiv.org/abs/2505.05190) ranks tokens by self-information and
  fills masked text using a paraphrased reference. The paper also reports a
  removal-versus-meaning trade-off. Sparse single-word infill with a fixed source
  skeleton is an adaptation, not a reproduction of that pipeline.
  A bounded local adaptation using single-word synonym proposals preserved its
  source skeleton but introduced category changes and grammatical errors; the
  tested outputs retained the diagnostic signal. It was not adopted as a default.
- [Smoothing attacks](https://aclanthology.org/2025.findings-emnlp.264/) use
  target-model top-token probabilities during generation, with confidence-based
  resampling. Those capabilities are different from editing an arbitrary pasted
  document without access to the originating model's probabilities.
- [MarkSec](https://arxiv.org/abs/2609.16681) evaluates removal jointly with an
  explicit automatic quality gate. Its reported rankings depend on that gate and
  model/dataset support. Such a gate is a declared measurement, not proof of full
  claim preservation. Our diagnostic status and quality review remain separate.

## Reproducing the independent runs

The scripts print help before importing optional ML runtimes:

```bash
uv run python scripts/validate_text_watermarks.py --help
uv run python scripts/benchmark_text_schemes.py --help
uv run python scripts/benchmark_textseal_reference.py --help
uv run python scripts/benchmark_alimark_reference.py --help
```

- `validate_text_watermarks.py` generates marked/unmarked English and Russian
  samples with Hugging Face's reference SynthID processor. `--input-dir` replays
  frozen source/control text with hash checks; `--rewriter-model` compares another
  local rewrite model on the original source tokenizer and reference configuration.
- `benchmark_text_schemes.py` takes an external `--markllm-root`, records its
  revision/configuration, and attempts its selected schemes in bounded child
  processes. DIP/Unbiased history is cleared and its generation/detection phase
  reset at every boundary, so generation state cannot contaminate a control.
  `--model`, `--device`, `--languages` and `--schemes` define the run.
  `--input-dir` replays completed frozen cases; incomplete reference cases generate
  new controls. `--dependency-dir` can isolate optional research dependencies.
  Dense EXPGumbel tables are rejected before allocation when they exceed
  `--reference-memory-budget-gib`. PF uses a plain completion prompt because
  the reference EOS trim otherwise cuts a chat prompt at its first EOS token.
  Complete prompt prefixes are removed; prompt-only fragments are rejected.
  The CPU reference adapter keeps random sampling/scoring on CPU while forwarding
  neural inference to the selected model device, including entropy-based and
  author-custom generation paths. It does not substitute this library's toy PRF.
- `benchmark_textseal_reference.py` takes an external Meta `--reference-root`
  and isolated `--dependency-dir`. It runs the author's generators and detectors
  for TextSeal, TextSeal2, the one-key AutoMark copula and the published textGrain
  implementation. Research-owned keys do not reproduce OpenAI's production service.
- `benchmark_alimark_reference.py` replays frozen synthetic `source.txt`,
  `baseline.txt` and optional `after.txt` through the external author's detector.
  Supply `--reference-root`, optional isolated `--dependency-dir`, a local
  author-compatible `--embedder`, prepared `--nltk-data`, `--input-dir` and a new
  `--output-dir`. The input `manifest.json` requires `inputs_sha256` for every
  supplied text, a nonempty `generation` provenance object and an optional
  nonnegative integer `retained_prose_spans`. The replay records actual reference
  and encoder file hashes. It does not import or run vLLM generation. Without a
  caller-declared `--threshold`, the result remains `scored_uncalibrated`.

  A declared threshold is an experiment setting, not a calibrated FPR. The initial
  HF generation bridge used a smaller candidate budget than the paper; its source
  was not a valid positive at the separately tested cutoff. That is a failed
  positive control, not a removal success. Full native AliMark generation and
  broader calibration remain distinct work. A follow-up used the author-size
  candidate budget and a frozen threshold rule on separate local calibration and
  evaluation controls. Short generations were recorded and replaced by a
  length-only rule established before any control scoring. This conditions the
  population on length and does not calibrate production or human-text FPR.
  The source exceeded that threshold; default rewriting refused it, while an
  experimental sentence rewrite retained the signal and changed meaning.

- `benchmark_automark_reference.py` replays the selected RotationFirst and
  ConcordMark author detectors from the separate
  [AutoMark repository](https://github.com/eth-sri/automark). This is distinct
  from the one-key copula implementation above. Supply `--reference-root`,
  local `--model-dir`, the author's pinned `--reference-tokenizer` JSON,
  matching `--group-asset` and `--canon-asset`, optional isolated
  `--dependency-dir`, `--input-dir` and a new `--output-dir`. The frozen input
  directory includes `watermark.json` and the shared UTF-8 hash manifest.
  Generation provenance binds the scheme, key, reference revision, vocabulary,
  sampling settings, configuration digest and local model/asset digests.
  Current replay supports temperature 1, top-k 50 and top-p 1. Generation must
  declare stochastic sampling after the author processor: ConcordMark can return
  ordinary logits during its prelude/gates, so argmax there changes the method.
  Such stale fixtures are rejected before loading the model.
  Ordinary vocabulary, pre-tokenization and decoder must match the asset's
  reference tokenizer. Differing added special IDs must be ungrouped and
  identity-mapped; those IDs and unknown padding IDs are forbidden in inputs.
  ConcordMark requires model-derived token logprobs and top-eight competitor
  rows. `--model-device cpu` (default) or `mps` selects auxiliary inference; the
  watermark logic stays on CPU. Unchanged author local-HF signal methods run with an explicitly loaded
  local image/text model; failed or all-zero auxiliary data are not substituted
  for a working weighted detector. Numerical raw results, reference source
  hashes and runtime versions are retained. `--alpha` is a local diagnostic
  setting, not an empirically calibrated production FPR.

  The [submission inventory](../data/evaluations/text-watermark/automark-variants.json)
  lists the pinned published candidate source states and baselines. Every source
  object and configuration digest was checked. These are research submissions,
  not provider model versions. Inventory entries remain unexecuted until that
  exact source state is evaluated; selected-package results do not qualify a
  differently packaged submission with a similar name.

  `benchmark_automark_submission.py` separately replays exact content-addressed
  candidates 001 through 025 and unweighted states 037 through 048 using an explicit
  submission/class allowlist. It requires the author's `agent-submissions`
  source store, the complete materialized submission and its matching inventory
  row. The adapter checks the global manifest, source-map/configuration digests,
  every source object and materialized file, and rejects extra files or symlinks.
  It refuses an already imported author namespace rather than silently reusing
  another submission's classes. Each candidate needs its own source proof; a
  matching class name alone never qualifies a different inventory entry.
  The declared AAR-context3 and SynthID-context3 baselines use a separate tree
  proof: the complete original project must be available beside the source
  store at the manifest's declared path, inside the pinned reference. The
  original tree must match the inventory's canonical file-map digest, and its
  complete materialized copy must have the same files and contents. Both roots
  and their contents reject symbolic links. Project/configuration declarations
  must match the manifest; the configuration also needs its own matching digest.
  Baseline generation binds `source_tree_sha256`, while candidates bind
  `source_map_sha256`; mixed proof types are rejected. A source-tree proof alone
  does not enable the entropy-dependent TextSeal baseline.
  Before constructing a detector, it rejects any configuration declaring
  auxiliary signals: this plain replay cannot silently omit entropy, surprisal
  or row-weight inputs. TallyMark states 037 through 040 and the multivoice
  states 041 through 048 require no auxiliary signals or lexical grouping
  assets; their source and configuration proofs
  remain mandatory.

  Lexical candidates 002 through 014 additionally require the unchanged published
  WordNet grouping asset and its metadata, plus `--reference-tokenizer` pointing
  to the hash-pinned Llama-3.1-8B-Instruct tokenizer. The complete tokenizer JSON
  must match after normalizing only the old string/new pair serialization of
  BPE merges. Merge order, ordinary and special token IDs, normalizers, decoders,
  pre/post-processors, padding and truncation remain part of the comparison.
  Vocabulary size alone is insufficient. The grouping arrays must have valid
  types, bounds, idempotence and metadata counts. The recomputed proof must equal
  the frozen generation manifest before importing author code.

  FreshRace/RenewalRace candidates 015 through 025 require their own
  `freshrace_assets` manifest and the pinned reference tokenizer. Their published
  common-pair, predictable-hash, adaptive and lexical tables are checked for
  hashes, schemas, ID bounds, sorting, group idempotence and mask geometry.
  Unlike the earlier WordNet asset metadata, these source states do not embed
  their asset-build tokenizer identity. The declared reference follows the
  author's default Llama-3.1-8B evaluation model; matching the full tokenizer JSON
  and preserving assets does not recover missing build provenance. The adapter
  records that limitation explicitly and rejects stale frozen proofs before
  importing author code. Earlier candidate-001/002 replay scores were reproduced
  exactly after extending the adapter.

Use a separate author-detection Python environment with Torch, Transformers,
NumPy, Numba, SciPy, msgspec, Loguru and PyYAML. vLLM generation is not needed
for this replay; the frozen environment versions belong in the retained run.

```bash
/path/to/research-python scripts/benchmark_automark_submission.py \
  --source-store /path/to/automark/agent-submissions \
  --submission-dir /path/to/materialized-candidate-001 \
  --submission-id candidate-001 --tokenizer-dir /path/to/local-model \
  --input-dir /path/to/frozen-inputs --output-dir /path/to/new-run
```

  The input contract includes source/baseline and optional after text hashes,
  `watermark.json`, exact submission/map/configuration provenance, a research key,
  vocabulary size and tokenizer/model configuration hashes. Supported sampling is
  temperature 1, top-k 50 and top-p 1, with stochastic sampling after the author
  processor. Source and control completion/round-trip flags must bind to the exact
  scored text and token IDs, with no unsupported generated IDs and at least 64
  scored tokens each. An unqualified generation is distinct from an undetected
  source, failed negative control, refused rewrite or retained watermark.
  Scores do not establish fidelity; retained prose remains `partial_unverified`
  even if a whole-output score falls below the diagnostic threshold.

  One frozen candidate-001 study used Qwen2.5-1.5B-Instruct revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, unchanged author CPU logic and HF MPS
  float32 model logits. The author source/control p-values were respectively
  `1.5158813209343072e-60` and `1.0` at the predeclared alpha 0.01. Both texts
  completed with EOS and exact token round trips; fresh author-detector replay
  reproduced the contrast. Native strict library rewriting refused full output.
  Captured rejected attempts changed the supplied names and invented a location.
  This is one-key, smaller-model bridge evidence, not native vLLM/paper-model
  parity, production FPR calibration, successful removal or private-provider
  coverage. This run does not qualify any other inventory entry.

  A separate frozen lexical cohort executed exact candidates 002 through 014
  with unchanged CPU author processors and detectors, HF stochastic sampling and
  Llama-3.2-1B-Instruct revision `5a8abab4a5d6f164389b1079fb721cfab8d7126c`.
  The original tokenizer file at the author-pinned Llama-3.1 revision was verified
  against its official Git blob ID using a public byte-identical mirror. The full
  canonical tokenizer JSON, including every special token, matched the generator;
  the published lexical asset was unchanged. This smaller model is an explicit
  execution bridge, not paper-model or native vLLM parity.

  Every source completed with EOS and an exact text/token round trip. All source
  p-values were below the predeclared 0.01 threshold, and every detector accepted
  the one shared unmarked control. Fresh author-detector replay reproduced every
  score. There were eleven unique source texts: candidates 006, 007 and 008
  generated identical text. One prompt, key and seed, duplicate sources and a
  reused negative control do not establish independent sample coverage or FPR.
  The native strict default rewriter refused every complete document. Manual
  review found meaning changes even in intermediate paragraphs accepted by the
  selected fact checks, including invented instructions and changed event timing.
  Neither these intermediate checks nor detector scores establish semantic fidelity.
  Inventory statuses distinguish this source/control evidence from successful
  removal; remaining entries need their own exact source-state execution.

  A frozen local follow-up used Qwen3.5-0.8B with the same ordinary token map,
  pre-tokenizer and decoder as the author's pinned Qwen3.8-27B tokenizer. Seven
  unshared audio specials were ineligible and identity-mapped, and were excluded
  along with padding IDs. The smaller model is a declared execution surrogate,
  not a replication of the paper's model. The initial slow CPU attempt and an
  incorrect greedy post-processor bridge are retained as unqualified attempts.
  Corrected stochastic generation and fresh author-detector processes gave
  p=5.2968644e-46 for RotationFirst and p=3.8605231e-96 for ConcordMark. The shared
  unmarked control had p=1 for both at the predeclared diagnostic alpha 0.01.
  Default full-document rewriting refused both sources. This executes selected
  references; it does not show removal or calibrate an empirical production FPR.

  Saved candidates included damaged placeholders and a renamed event, which
  remain refusals. A separate accepted paragraph replaced autumn visits with
  year-wide activity and a spring-to-autumn period. The shared fact checker now
  rejects selected explicit season loss, insertion and substitution. Common
  English noun forms and selected Russian forms are covered; ambiguous bare
  `spring`/`fall` do not independently establish a season, but can explain an
  explicit counterpart. A spring-to-autumn range records both seasons. The guard
  does not bind seasons to actors or events, establish dates or prove meaning.
- `benchmark_postmark_reference.py` replays the external author's Nomic word
  selector and Paragram presence detector on frozen synthetic inputs with the
  same byte-hash manifest contract. Supply `--reference-root`, local Nomic v1
  `--embedder`, inspected `--encoder-code`, BERT `--tokenizer`, prepared
  `--nltk-data`, populated `--tiktoken-cache`, optional `--dependency-dir`,
  `--input-dir` and a new `--output-dir`. Prepare the author-referenced Paragram,
  Nomic document table and vocabulary files in the external reference checkout;
  the script records their hashes. The encoder uses CPU nonfused kernels and a
  local config redirect to supplied model code; reference selection and scoring
  functions remain unchanged. This bridge is recorded in the result. It requires
  a fresh Python process and does not call paid model APIs or download weights.
  Each response recomputes its own detection words. Reusing the source's words
  for an output would not reproduce the author's detector. `--ratio` defaults to
  0.12 and `--cosine-threshold` to 0.7. The latter is a word-match setting, distinct
  from the optional experimental decision `--threshold`. Without that decision
  threshold, results remain `scored_uncalibrated`. Empty word selection and
  nonfinite or out-of-range presence scores are failures, not passing skips.
  Declared generation ratios and word-table hashes must match the replay.
  In an initial synthetic local HF Qwen insertion bridge, author presence was
  0.2105263 for the unmarked baseline and 0.2051282 after attempted insertion.
  This did not establish a positive-control contrast. Library rewriting in that
  isolated reference environment refused the generated text after three attempts.
  That environment used Transformers 4.44.2, not the project's default runtime.
  These are failed-control/refusal findings, not a demonstrated removal success.
  A stronger local MLX insertion follow-up used the unchanged author v2 loop,
  a frozen handwritten baseline, the same ratio and a predeclared diagnostic
  decision threshold 0.5. The baseline presence was 0.0 and marked-source
  presence 0.64. A default-model rewrite scored 0.3913043 but changed supplied
  objects, cleanup timing and planned versus completed actions. It is not a
  quality-qualified removal. The initial MLX rewrite refused the full text;
  diagnosis found a false refusal of "materials needed" versus "materials
  required". After correcting equivalent English necessity forms, a fresh MLX
  rewrite returned the full text and scored 0.5909091, retaining the mark.
  It still strengthened accessible resources into abundant resources and an
  opportunity to exchange advice into described exchanges. Independent fresh
  author-detector processes reproduced the initial pair, default-output score
  and corrected MLX output. The single synthetic pair and arbitrary diagnostic
  threshold do not calibrate FPR or establish provider or market coverage.

- `benchmark_dipper_reference.py` replays the pinned DIPPER-paper NumPy detector
  on full frozen completions with the original context's last token. Supply
  `--reference-utils`, local `--tokenizer`, `--input-dir` and a new `--output-dir`.
  The shared `source.txt`, `baseline.txt`, optional `after.txt` and hash manifest
  contract also requires `context.txt`. Its generation block binds
  `detector_variant=dipper_paper_numpy`, the pinned reference revision/source
  digest, vocabulary and complete tokenizer-structure digests,
  `watermark_fraction=0.5`, `vocab_size=50257` and `context_sha256`.
  The adapter rejects token-mapping, merge/pre-tokenizer, prefix, parameter and
  source-code drift. It executes only the inspected numerical functions and
  requires the validated little-endian seed encoding. The author's strict z>4
  diagnostic distinguishes source/control failure, missing output and retained
  marks. This full-text replay is not the paper's matched-length ROC experiment;
  below-threshold output still needs independent fidelity review. It loads no
  generation model and does not download assets or instantiate API clients.

Every run requires a new `--output-dir`. Run ML jobs sequentially on memory-limited
hosts. Missing checkpoints, encoder models, compiled extensions or adapters must
remain visible coverage gaps. They must not turn into passing skips.

Generated evaluation outputs remain outside the repository by `data/README.md`.
Keep each run's `benchmark.json` or `summary.json`, its UTF-8 inputs/outputs, logs
and their SHA-256 digests in the retained artifact bundle. Record the source module
hash and external reference revisions before comparing runs.
Keep the source snapshot taken at job start: a later file hash may describe
different code from the module already imported by a running process.
Local artifact bundles are evidence for that run; they do not become public
release claims automatically.

### Comparing methods on the same marked and clean sources

Use `scripts/benchmark_text_pairs.py` from a source checkout to combine blinded
fidelity reviews with locally recorded detector results. Keep the run directory
and inputs outside the checkout if the texts are private.

The input JSON contains an `arms` list, a `detectors` object mapping named
reference configurations to their minimum scored positions, and a `pairs` list.
Each pair has an `id`, a `detector`, and `marked` and `clean` entries. Each
entry contains the exact `source` text and an `outputs` object with one exact
output per arm. Use JSON `null` for a refusal and the original text for an
unchanged output. Supply detector receipts as a JSON list with one
`detector`, UTF-8 `sha256`, boolean `detected`, and integer `positions` record
per distinct detector and text. Run the reference detector separately with
the same configuration and threshold for every source and output.

For any method that deliberately retains prose spans, record them under
`retained_prose` as a mapping from arm to the list of exact fragments present in
the source and output. Score those fragments separately with the same detector.
The report flags a partial output as `partial_unverified` if any declared
fragment is positive or has too few scored positions. A faithful, whole-document
negative with that flag stays in the denominator and earns no joint success.
The scorer checks declared fragments. It cannot infer which spans were retained
by an arm.

```bash
uv run python scripts/benchmark_text_pairs.py bind \
  --input trials.json --run paired-run
uv run python scripts/benchmark_text_pairs.py seal \
  --run paired-run --reviews reviews.json
uv run python scripts/benchmark_text_pairs.py score \
  --input trials.json --run paired-run --reviews reviews.json \
  --detections detections.json --baseline strict --output results.json
```

Give reviewers only `paired-run/blind-pairs.json`. Record a boolean
`faithful` verdict and a nonempty `reason` for each packet ID in
`reviews.json` and seal it before inspecting detector results. The script
checks that every marked source tests positive, every clean source tests
negative, and both meet the declared position minimum. Every attempted source
stays in the denominator. Only changed marked outputs with enough scored
positions, a negative detector result, and a faithful review count as joint
successes. The report also gives detector outcomes for changed clean controls
and exact paired tests against the declared baseline for each detector.

#### Generating matched KGW sources locally

`scripts/benchmark_text_kgw_sources.py` creates paired marked and clean
continuations from a pinned local chat checkpoint and the author's KGW
implementation. The plan requires `detector`, `key`, `wrong_key`, `gamma`,
`delta`, `z_threshold`, `minimum_positions`, `seed`, `batch_size`, `top_k`,
`temperature`, `max_new_tokens`, and a list of exact, distinct `prompts`.
Pin the checkpoint, tokenizer and generation assets in `model_sha256`.
Pin `watermark_processor.py` and `normalizers.py` in `reference_sha256`.
Keep the plan, batch size, device and runtime fixed when reproducing a panel.
The author processor runs on CPU and the model runs on the chosen device.

```bash
uv run --extra text --with nltk --with scipy python scripts/benchmark_text_kgw_sources.py \
  --config kgw-sources.json --reference-root /path/to/kgw-reference \
  --model-dir /path/to/local-checkpoint --device mps --output-dir kgw-sources
```

`qualification.json` records every attempt, including incomplete
continuations, exact token traces and both key checks. A pair enters
`trials.json` when both continuations end cleanly, round-trip through the
tokenizer, and meet the position minimum, with a positive marked source,
negative clean control, and negative wrong-key check on the marked source.
The trials are ready for the local sentence-repair runner. Apply any planned
readability screen before generating rewrites, and keep the original
qualification report for the full attempted denominator.

#### Generating local KGW detector receipts

`scripts/benchmark_text_kgw_receipts.py` consumes a sealed paired review and
emits `detections.json` for `score`. Use the author's local reference files and
the exact local tokenizer from source generation. This profile runs on CPU
with `simple_1`, distinct bigrams, and no normalization. The configuration
file has `detector`, `key`, `gamma`, `delta`, and `z_threshold` fields.
`detector` must match the manifest's detector ID. In the same file, map each
reference file name to its SHA-256 in `reference_sha256`, including
`watermark_processor.py` and `normalizers.py`. Map the tokenizer files in
`tokenizer_sha256`, including `config.json`, `tokenizer.json`, and
`tokenizer_config.json`. Include all other tokenizer assets used by the
generation checkpoint.

```bash
uv run --extra text --with nltk --with scipy python scripts/benchmark_text_kgw_receipts.py \
  --input trials.json --run paired-run --reviews reviews.json \
  --config kgw-config.json --reference-root /path/to/kgw-reference \
  --tokenizer-dir /path/to/local-tokenizer --output-dir kgw-receipts
```

Pass `--dependency-dir` for dependencies prepared in an isolated directory.
The output also records raw z scores by exact text hash, runtime versions,
and hashes of the configuration, review, and receipts. A text with no scored
positions has an absent z score and cannot qualify as cleared. The paired
scorer still checks the original marked and clean controls and applies the
minimum scored positions in the manifest.

#### Generating matched Threefry sources locally

`scripts/benchmark_text_threefry_sources.py` generates paired clean and
marked continuations from a pinned local chat checkpoint and the open
reference. It takes the same sampling fields as the KGW source runner.
Use `detector`, `alpha`, `eos_token_ids`, and `scheme` from the Threefry
receipt configuration below, plus a distinct `wrong_key` in the same
63-bit key space. Pin the checkpoint, generation config, and tokenizer
assets in `model_sha256`, and the reference files in `reference_sha256`.
`top_k` must be positive. The clean and marked arms use the same
temperature and candidate width. The reference watermark processor runs on CPU.

```bash
uv run --extra text --with scipy python scripts/benchmark_text_threefry_sources.py \
  --config threefry-sources.json --reference-root /path/to/threefry-reference \
  --model-dir /path/to/local-checkpoint --device mps --output-dir threefry-sources
```

`qualification.json` records all attempts, token traces, end markers,
and both key checks. The library and reference must agree on the raw scores.
The runner exports a pair to `trials.json` only when both continuations
end cleanly, round-trip exactly, and meet the scoring minimum. The marked
source must be positive with the matched key and negative with the wrong key.
The clean source must be negative with the matched key. Apply a planned
readability screen before generating rewrites and keep the full qualification
record for the attempted denominator.

#### Generating local Threefry detector receipts

`scripts/benchmark_text_threefry_receipts.py` uses the open frequentist
reference and checks its scored positions, counted one bits, nominal p values,
and verdicts against the library on the same token IDs. Use a configuration
with `detector`, `alpha`, `eos_token_ids`, and a `scheme` object containing
`key`, `ngram_len`, `depth`, `context_history_size`, and `num_leaves` set to 2.
Use the tokenizer from source generation and list its asset hashes in
`tokenizer_sha256` as for KGW. Under `reference_sha256`, pin paths relative to
the reference root, including `synthid_text/detector_frequentist.py`,
`synthid_text/hashing_function.py`, and `synthid_text/logits_processing.py`.
Include `synthid_text/__init__.py` if the local reference contains it.

```bash
uv run --extra text --with scipy python scripts/benchmark_text_threefry_receipts.py \
  --input trials.json --run paired-run --reviews reviews.json \
  --config threefry-config.json --reference-root /path/to/threefry-reference \
  --tokenizer-dir /path/to/local-tokenizer --output-dir threefry-receipts
```

Each receipt generator scores only trials with its configured detector ID.
For a mixed manifest, concatenate the receipt arrays, for example with
`jq -s add kgw-receipts/detections.json threefry-receipts/detections.json > detections.json`.
The paired scorer requires complete coverage and rejects duplicate receipts.
These settings describe the open single-key research profile. Its nominal
p values assume independent bits.

#### Running the local sentence-repair comparison

`scripts/benchmark_text_local_repair.py` prepares the scorer manifest from
source pairs and a local Chat Completions endpoint. Use the same input format
as above with arms `restore`, `strict_residual`, and `document_residual`, every
output set to `null`, and no `retained_prose` entries. For example, one pair
has this shape:

```json
{
  "id": "pair-1",
  "detector": "kgw-config-1",
  "marked": {
    "source": "Exact marked source",
    "outputs": {"restore": null, "strict_residual": null, "document_residual": null}
  },
  "clean": {
    "source": "Exact clean source",
    "outputs": {"restore": null, "strict_residual": null, "document_residual": null}
  }
}
```

```bash
uv run --extra text-api python scripts/benchmark_text_local_repair.py \
  --input sources.json --output-dir local-run \
  --url http://127.0.0.1:11434/v1 --model MODEL
```

The runner uses the same first draft and at most one shared residual request
per source. The first arm restores individually rejected sentences when the
document fails the fact guard. The residual arms rewrite verbatim retained
sentences and require the same whole-document check. The strict arm also
applies the fact guard to each rewritten sentence. The runner records exact
retained sentences for detector scoring and produces `local-run/trials.json`
for `bind`. It logs requests and responses to stderr, so keep the logs with the
private inputs. Pin the model revision and runtime separately for a live rerun.
Run the reference detector independently after sealing the blind review.
The strict arm is available through `build_text_api_strict_residual_rewriter`
and the Ollama CLI flag `--strict-residual`. Replaying saved model replies
through both surfaces checks the actual request sequence, exact file output,
and untouched destinations on refusal.

#### Adding the local ReFlip baseline

`scripts/benchmark_text_reflip.py` adds the pinned ReFlip Python
`paraphrase` pipeline to an existing paired manifest. Use the same marked
texts and clean controls as the library arms. For the comparison plan, record
the ReFlip checkout `revision`, SHA-256 paths in `source_sha256`, the
`ollama_version`, the Ollama `model_digest` of ReFlip's default checkpoint,
and a positive `workers` count. Include the imported ReFlip modules and its
facts lexicon in the source hashes. Keep the same runtime and checkpoint for
a rerun.

```bash
uv run --with requests python scripts/benchmark_text_reflip.py \
  --input local-run/trials.json --config reflip-plan.json \
  --reference-root /path/to/reflip --url http://127.0.0.1:11434/v1 \
  --output-dir combined-run
```

The runner checks the live local Ollama version and model digest before
generation. It keeps the library arms and records requests, replies and
refusals. An unrevised failed block, error, or length-limited reply counts as a
refusal. Run `bind` on
`combined-run/trials.json` before judging meaning or scoring the new text.
The two arms may use different checkpoints, so this compares complete local
pipelines. Keep the transport record private when it contains private text.

#### Adding the on-device Apple baseline

`scripts/benchmark_text_apple.py` adds an arm from Apple Intelligence's
on-device system language model. It requires an available Apple Intelligence
model and the macOS 27 Xcode SDK. The library and
its other benchmark commands have no Apple SDK dependency.
[Apple Foundation Models documentation](https://developer.apple.com/documentation/foundationmodels/systemlanguagemodel)

Set `arm` to a new name, `instructions` to the exact generation prompt,
`macos_build` to the output of `sw_vers -buildVersion`, and
`maximum_response_tokens` to a positive integer, such as 1024.

```bash
uv run python scripts/benchmark_text_apple.py \
  --input combined-run/trials.json --config apple-plan.json \
  --output-dir with-apple
```

The runner records the OS and SDK versions and uses greedy sampling in a
fresh private session for each distinct input. It keeps empty outputs and
generation errors as refusals. A sentence left verbatim is recorded for
independent detector scoring. Seal the blind review before scoring
`with-apple/trials.json`. This baseline uses the system model shipped with
the recorded macOS build. Its weights can change with system updates.

## Local language preservation trial

A fixed seven-source, three-prompt experiment with the default cached
Qwen2.5-1.5B-Instruct checkpoint produced fourteen full outputs. Manual review
found eight language changes accepted by the earlier script/fact guards,
including French to English and Bulgarian to Russian. Prompt variants did not
resolve the problem; the original generation prompt remains unchanged.

The local backends now add a packaged `py3langid==0.4.0` classifier. It sees
masked source and raw candidate prose before restoring facts, strips service
markers and normalizes only the classification sample to NFC. Both samples need
at least 40 alphabetic characters. A known source label needs a normalized score
of at least 0.95, a different candidate label at least 0.8; otherwise the check
abstains. These asymmetric thresholds were selected after examining this
development set, not preregistered or calibrated as an accuracy guarantee.
Only the model is cached. No text is sent to a language-identification service.

Replaying the fourteen formerly accepted candidates through the new guard
rejects all eight observed language changes. Twenty handwritten same-language
paragraph pairs produced no rejections, with several abstentions and an incorrect
source label among uncertain samples. They test language identity, not factual
or semantic equivalence. Six retained outputs from the first experiment still
need meaning review. Short texts, code-switching, related languages, model
misclassification and uncertain samples remain limitations. This trial contains
unmarked synthetic controls and establishes no watermark-removal success or
closed-provider coverage. A clean `text`-extra installation then ran the original seven sources offline
against the cached checkpoint: two English outputs returned, while the Russian,
Ukrainian, Bulgarian and French cases exhausted their retries. CLI verification
confirmed a returned English file and preservation of an existing destination
when the French rewrite was refused. Diffusers, Accelerate and SentencePiece
were absent from that installation. These are generation/acceptance controls,
not semantic certifications. Full runtime outcomes and source/model hashes are
retained separately from candidate replay.

## Smaller multilingual MLX follow-up

A frozen trial used [Qwen3.5-9B-MLX-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit)
revision `938d8919941c6e7efd3c7150eff7fe9d12afa631`, with MLX 0.32.4,
mlx-lm 0.32.2 and the unchanged strict library prompt/guards. It included the
seven examined language controls, two new unmarked probes and the frozen
selected-package RotationFirst/ConcordMark sources. Ten full outputs returned;
the abstract English number control was refused after three verbatim echoes.
Returned passages retained the source language in this review, but still changed
meaning: Bulgarian unscheduled became unknown scheduling status, Russian intended
allocation became dispatch, and during a trial became throughout a trial.
The RotationFirst rewrite also supplied a different instruction-role reading.
These observed failures prevent treating returned output as semantic success.

Fresh unchanged author detectors, checked against the archived source hashes,
reproduced both marked sources and their shared negative control. RotationFirst
remained detected after rewriting (p = 6.36008182732346e-14); ConcordMark also
remained detected (p = 7.399596243839886e-22), both at the frozen alpha = 0.01.
ConcordMark used the same pinned Qwen3.5-0.8B MPS float32 auxiliary signal bridge
as source generation. These are selected-package research-key results, not
native vLLM parity, every AutoMark submission or private-provider coverage.
The model was not adopted as a default.

Short handwritten scheduling-certainty, allocation/dispatch, explicit
visitor/staff role and during/throughout pairs now record these failure types in
`fidelity-cases.json`, including Bulgarian examples. The current lexical fact
checker accepts all four incorrect examples as well as their correct controls.
These are documented blind spots, not newly passing fidelity tests. The pairs
were written after inspecting model outputs and are examined development data;
previous judge measurements do not cover the revised corpus hash.

## Corrections from actual runs

- The former SynthID scorer used BLAKE2 simulation hashing and deduplicated whole
  ngrams. The corrected scorer delegates to HF's processor and repeated-context
  mask, with matching table seed/size, history and device.
- Exhausted local retries returned a failed candidate and verbatim echoes counted
  as regeneration. Invalid/unchanged candidates now raise without replacing output.
- List prefixes are locked separately from their prose. Comparative "one of the
  most" is not treated as a count; "one crate" remains a quantity.
  A selected pattern for an indefinite article with a modified participle
  handles noun substitutes such as "a partly filled one". The regression
  checks exact counts and local prompt masking, including a numeric compound.
- Local candidates must preserve the full service-placeholder multiset. An
  invented example marker was reproduced in a real rewrite and is now rejected.
- A dominant-script backstop rejects obvious unrequested translation; it is not
  a complete language detector. For long enough predominantly Latin/Cyrillic
  inputs, the backstop also rejects fragments in the other alphabet when it was
  absent from the source. Existing mixed-script sources remain a coverage gap.
- Recognized names, numeric literals and selected English/Russian cardinal spans are masked before local generation and
  restored after placeholder checks. Quantity checks reject added as well as
  missing values. These checks deliberately preserve source mistakes rather than
  silently rewriting its facts.
- Adjacent recognized name words now share one fact marker, and the checker
  requires their contiguous survival. This fixes an observed split place name;
  source-to-output roles and general entity coverage remain unverified.
- Selected English/Russian numeric bounds and approximations are attached to
  their normalized values. This rejects dropped or inverted bounds while allowing
  `at least 20`/`no fewer than 20`. Quantified negation is distinguished from
  predicate negation. This lexical check can refuse ambiguous phrasing; it does
  not establish participant attachment or general semantic equivalence.
- Selected English phrases distinguish a percentage reached from a percentage
  gained or lost. Positive and negative controls cover lexical paraphrases,
  inverted roles and multiple percentages in the same sentence. This guards a
  fact relation without changing any statistical detector or decision threshold.
- Selected Russian cardinal forms are read as whole integer sequences, including
  hundreds and descending thousand/million/billion scales. This distinguishes
  `двадцать пять` from `двадцать шесть` while allowing `25`, and attaches supported
  bounds to the whole value. Numeric words and selected duration adjectives such
  as `пятилетняя` are frozen before generation. Superlative `один из лучших` is
  excluded only when it is not inside a larger cardinal. Ambiguous `сорока`
  remains an ordinary word. Malformed recognized sequences and indefinite bare
  plural scales remain literal facts, rather than being guessed as exact values.
  This is not complete Russian morphology, fractional-word, ordinal or
  arithmetic parsing; quantity-to-entity/unit binding remains unverified.
- Selected English integer grammar handles hundreds and descending
  thousand/million/billion scales, whitespace/hyphen compounds and internal
  `and`. Conjunction lists remain separate values; malformed recognized sequences
  retain literal signatures. English `minus`/`plus` and Russian `минус`/`плюс`
  bind their sign to the whole cardinal. Superlative-one filtering happens at
  complete number boundaries, preserving `twenty-one of the best`.
  Explicit zero multipliers remain zero, rather than becoming the implicit one
  used by a bare English scale word.
- Recognized finite numeric literals preserve ASCII/Unicode-minus signs,
  leading-point decimals and scientific exponents. Decimal normalization uses
  exact digits with isolated context flags, not floats or precision rounding.
  Extreme exponents use exact compact keys instead of expanded zero strings.
  Ambiguous comma syntax and dotted identifiers retain literal signatures.
  At a shared start, masking chooses the longest phrase. This keeps a full
  recognized name intact without letting its words elsewhere split a larger
  cardinal. Pending matches are rescanned after overlaps while absent patterns
  remain cached, preserving unconsumed name suffixes without repeated full scans.
  These checks do not parse all fractions, vague quantities, formats, range
  ordering, units or participant attachment. The retained unmarked runtime
  controls exercise markers and numeric invariants, not watermark removal or
  a complete fidelity guarantee.

The focused tests include failing-before/passing-after regressions and deliberate
mutations of benchmark accounting, the model-device bridge, prompt extraction
and the reference-memory guard. Test counts and
runtime outcomes belong in the retained run reports, not a drifting prose total.

## Compatibility of older score helpers

Only `score_synthid_text` currently delegates to the matched HF reference processor.
The legacy Gumbel, textGrain, TextSeal2, KGW and Unigram score helpers use this
module's custom BLAKE2 research PRF. Their historical numerical examples are
simulations. They are not MarkLLM/Meta/provider-compatible detectors, even with a
matching integer key. The expanded benchmarks use external reference detectors.

## Remaining verification boundary

Anthropic's [model table](https://support.claude.com/en/articles/16266773-how-claude-marks-ai-generated-content)
identifies marked versions and private text-detector access. The catalog records
these individual models as blocked rather than implying a passed Claude test.
The catalog refresh also includes the currently listed older Opus/Sonnet/Haiku
models. For the marked legacy entries, non-default temperature/top-p/top-k can
make marking unreliable. Opus 3 cloud-partner text marking is not confirmed by
the table, and selected Bedrock rollouts have an additional delay caveat. These
conditions belong to source qualification, not a removal success.
[OpenAI](https://help.openai.com/en/articles/8912793-provenance-signals-in-openai-generated-content)
publishes textGrain product, region and API opt-in conditions with restricted text
checks. Its public policy does not enumerate every eligible model ID.
[Google](https://deepmind.google/models/synthid/) describes Gemini app/web text
marking; public image/audio/video checks are not a production text detector.

For each private-provider claim, the missing experiment remains an authorized
before/after detector check on exact source/output bytes, with known model and
marking settings and controls. Open reference results cannot substitute for it.

## Adjacent measurement preservation

Regression probes showed that equal number sets allowed kilometers to become
miles and milligrams to become grams. The local guards now compare selected
amount/unit multisets and protect the entire recognized measurement phrase.
Unit aliases normalize within a registered unit; case-sensitive symbols keep
`mW` distinct from `MW`. Known slash-unit chains, square-unit aliases and selected
Russian duration adjectives preserve their units. Numeric bounds include the
unit to reject swapped bounds on equal quantities of different units.

This is lexical protection, not a complete quantity parser. No value conversion
or dimensional algebra runs. Unknown units, unregistered powers/products,
prefix currencies and ambiguous unit words remain limits; common prepositions
are not inferred as units. Explicit proper-name regressions remain covered.
The symbol conventions follow [NIST's SI writing guidance](https://www.nist.gov/pml/owm/writing-si-metric-system-units);
this implementation does not claim coverage of every SI unit or notation.

An unmarked default-model runtime retained the protected English measurements
but changed a reading into a precision claim, so it did not establish full
fidelity. The Russian control exhausted its retries. These results test the
mechanical invariant and refusal behavior, not watermark removal. Examined
measurement and measurement-role pairs in `fidelity-cases.json` distinguish the
new unit checks from the remaining source-to-role gap. Earlier judge runs do
not evaluate this revised corpus hash.

### Native chat special tokens

A real Llama-3.2 tokenizer probe exposed duplicate BOS tokens when a rendered
chat template was encoded with default automatic special tokens. The native
Transformers rewriter now checks for a configured template, encodes rendered
chat without adding special tokens again, and uses automatic special tokens for
the plain-prompt fallback. Regression tests first failed on both paths before
the fix. A real Llama rewriter run received exactly one BOS. Its returned prose
changed meaning, so this is prompt-format evidence, not fidelity or removal proof.

The lexical cohort above was frozen before this fix, at text-module SHA-256
`8e80ac72da9c1cdc1a22547d4cc2e126ee6f68062848af6c35f5b9edb78f7467`.
All 44 recorded Qwen attempt prompts encoded identically with automatic special
tokens enabled or disabled. A fresh strict candidate-002 rewrite on the corrected
code reproduced every prompt, generated token and refusal from the frozen run.
The other neural rewrites were not rerun on the corrected module. Source generation
and fresh author-detector replays remain bound to their frozen inputs and code.

### Exact FreshRace/RenewalRace cohort and explicit frequencies

A frozen cohort executed exact source states 015 through 025 with the same pinned
Llama-3.2-1B generation bridge and original author CPU code. Every source ended
with EOS and an exact text/token round trip; source/control gates passed at the
predeclared diagnostic alpha 0.01. Every fresh author replay reproduced its
scores. There were eight unique source texts, one prompt/key/seed and one exact
unmarked control reused from the earlier lexical cohort. These are neither
independent samples nor FPR calibration or paper-model parity.

The strict default rewriter initially refused nine complete documents and
returned two, for candidates 019 and 025. Both outputs fell below the detector
threshold; both changed meaning. Manual review found lost alternate-day timing,
an invented weekly schedule, changed tool ownership and borrowing/lending roles,
and an omitted duration condition. These remain quality failures.

A shared explicit-English-frequency check and whole-phrase masking now reject
selected lost, changed or invented recurrence intervals, counts and units. The
observed accepted paragraphs first passed the prior fact checker and fail the
new one. A real strict follow-up refused candidate-019. Candidate-025 still
returned prose below the detector threshold, but manual review found remaining
ownership, duration-condition and participant-role changes. No faithful-removal
success is claimed. Source/control generation stays bound to the earlier frozen
module; the two follow-ups separately record the frequency-guard module hash.

The added short synthetic recurrence pairs are examined development evidence,
not a fresh holdout. Earlier judge measurements do not evaluate the revised
corpus hash. Selected English schedules are not general temporal reasoning:
calendar-unit conversion, arbitrary/bounded schedules, shared-modifier
compression and actor/event binding remain outside the guarantee.

### Indefinite-duration follow-up

The recurrence-guard candidate-025 output still omitted the source condition
`for a day`. The selected fact checker now interprets indefinite English
hour/day/week/month/year phrases as amount one with a complete unit, retains
selected numeric bounds, and masks the complete phrase in both local backends.
Frequency phrases are excluded from separate duration extraction. Ambiguous
second/minute senses, compounds, ratios and general conditional/event binding
remain outside this extension.

The actual previously accepted paragraph passes the prior checker and fails the
new checker. A native strict default-model rerun refused a complete output after
three attempts. Independent author replay still verifies the frozen marked source
and unmarked control; there is no after output to score. Source/control generation
was reused by exact bytes, and the new rewrite has its own module hash. Earlier
meaning-changed outputs are retained as failures. Other source states were not
rerun on this duration-guard revision, and no faithful-removal success is claimed.
The new short synthetic pairs are examined development evidence; earlier judge
measurements do not evaluate the revised corpus hash.

A separate fresh unmarked usability control was also refused by the native
default rewriter after three attempts: each candidate changed the sole protected
marker from index zero to index one. Candidate wording also changed inspection
into checking out a tool. This single control is not watermark-removal evidence
and does not establish usable meaning-preserving rewriting. Model instruction
adherence and participant/event preservation remain unresolved.

### Remaining entropy-weighted source states

A constructor/requirements audit of exact candidates 026 through 036 confirms
that each requires per-token `entropies`. The subsequent exact-source executions
are recorded below. The author's entropy-weighted condition can return no weights when
signals are omitted, so a plain detector call would not reproduce the declared
weighted method. The source-state adapter does not yet allow these candidates.

The unchanged auxiliary provider defaults to `google/gemma-3-270m-it` and consumes
the supplied integer token IDs directly, without retokenizing decoded prose.
It assigns zero at position zero and aligns each later float32 Shannon entropy
with the preceding position's prediction. Replaying these states requires
preserving that declared behavior and binding signal arrays to exact token IDs,
model revision/files, source/configuration and computation code. Zero-filled or
missing arrays and unweighted scores cannot qualify the author method.

An access check for the declared Gemma model's pinned configuration returned
HTTP 403. The model's [official card](https://huggingface.co/google/gemma-3-270m-it)
requires license acceptance before access. This historical access failure did
not establish that identical files were unavailable elsewhere. The later
byte-verified public-mirror execution below supersedes this dependency blocker;
a different auxiliary model or an unweighted detector was not substituted.

An additional exact-source constructor audit covers candidates 037 through 052
and the three baseline configurations, with their declared requirements recorded
in the submission inventory. TallyMark and states 041 through 048 declare no
auxiliary signals. State 049 requests `token_logprobs`; states 050 through 052
also request aligned `row_id_0` through `row_id_7` and `row_lp_0` through
`row_lp_7`. Their published provider can use the evaluated model or a compatible
local corpus model, rather than the separate Gemma entropy model. Its failure
paths can yield zero signals, and missing inputs can yield uniform weights.
These weighted states therefore need their own verified signal pipeline, rather
than being grouped under the Gemma access restriction or tested without weights.

The baseline source tree and each configuration digest were reverified before
constructors ran. The declared TextSeal baseline requires entropies; AAR and
SynthID do not declare auxiliary signals. These audits establish prerequisites,
not generated-text control qualification or removal coverage. Baseline source
verification uses the separate tree-based replay path, since these configurations
are not individual content-addressed submission maps. The declared AAR and
SynthID baseline executions are recorded below; the later TextSeal execution uses its entropy
pipeline; its qualification failure is recorded below.

### Native 9B rewrite ablations and numerical occurrences

A pinned Qwen3.5-9B MLX development experiment compared the current instruction
with four exploratory alternatives: minimal edits, protected-phrase references,
syntax-focused rewriting and a syntax/reference combination. The same library
masking, selected guards, retry/stop policy and sampler settings were used. Only
instruction or message construction changed in retained private prototypes;
model defaults and production instructions were not changed. Prompt length can
change the library's derived token cap, so actual caps and finish reasons are
recorded rather than claiming identical numerical budgets.

The frozen inputs comprise unmarked controls and two previously examined exact
AutoMark sources. The original source/control generation and research key are
reused, not independent new coverage or FPR calibration. Reviews compare each
output with predefined material claims and distinguish fidelity uncertainty,
meaning changes, fluency issues and refusal. These are assistant reviews, not
independent human annotations or a calibrated semantic judge. An initially
favorable location review was subsequently made more conservative; both the
initial review and revision remain visible.

More outputs did not establish better fidelity. Recurrence modifiers moved to
the wrong event, intended allocation became dispatch, resident affiliation became
tool provenance, and sentence transformations introduced ambiguous or ungrammatical
bindings. These results did not justify adopting an alternative. Detector scores remain
separate from these semantic assessments. Every complete marked output in
this experiment remained detected at the frozen diagnostic alpha 0.01. Fresh
author replays reproduced the original source/control scores; repeated controls
are not independent samples. The per-variant scores and review scopes are
recorded separately in the submission inventory.

The experiment also exposed a deterministic bug: `Two clinics two ...` passed
the old unique-number-value check. Numerical fact checking now preserves
normalized mention counts while keeping the old private set view available.
An actual native replay reproduces the same first prompt/tokens, rejects that
candidate and returns a second candidate preserving the two-clinic winter
finding, unknown summer persistence, conditional extension and lack of a decision.
This is an examined unmarked control, not watermark-removal evidence. Earlier
rewrite trials remain bound to the preceding module hash; the count-guard
follow-up has its own hash and does not rerun the whole model cohort.

Added short quantity, recurrence-event and provenance pairs are examined
synthetic development cases. Earlier judge metrics do not evaluate the new
corpus hash. Occurrence counts do not certify actor/event, qualifier or entity
binding, and a retained or reduced detector signal does not establish meaning.


### Exact TallyMark source-state execution

The exact allowlisted TallyMark states 037 through 040 completed generation,
source/control detection, strict native library rewriting and fresh-process
author detector replay on the declared Qwen2.5-1.5B MPS float32 bridge. No author
processor or detector was edited. Source and control both completed with EOS,
round-tripped to the scored token IDs and passed the predeclared scored-length
gate. The inventory records detector outcomes and the library module hash.

All sources were detected at the frozen diagnostic alpha 0.01; the shared
unmarked control was not. The library refused complete output in each case.
States 039 and 040 have identical source/configuration hashes, generated IDs and
native rewrite attempts; these are duplicate executions, not independent
evidence. Generation quality also varies, with malformed and ambiguous source
clauses. Qualified detector contrasts alone do not certify readable source text
or correct rewriting.

The accepted intermediate first span in states 039 and 040 adds a schedule for
keeping tools at home before use and substitutes a mallet for a hammer. A later
span fails retries, so neither document is returned. This preserves evidence of
a semantic-check gap rather than calling whole-document refusal a faithful
rewrite or removal success. Production model and rewriting instructions are
unchanged. Later sections record subsequent source-state and baseline executions;
the submission inventory is the source of truth for remaining execution gaps.

### Native Qwen3-4B study and exact multivoice states

A pinned MLX Qwen3-4B-Instruct-2507 4-bit runtime was compared with the current
Transformers Qwen2.5-1.5B configuration on the same examined development inputs.
Both used the library's unchanged production instruction, masking and guards
from the recorded pre-calibration snapshot. Model/backend and tokenizer-derived
token caps differ, so this is not an isolated model-size comparison or a replay
of a competitor's Ollama/GGUF runtime. The 4B runtime returned more complete
outputs but changed an approval condition into secured funding and an activity
statement into lack of responsibility. A Russian output preserved the selected
claims while introducing incorrect numeral/noun agreement. These results do not
justify changing production defaults.

The fact gate now preserves selected literal approval, permission and
responsibility cues separately from plain activity or funding availability.
Native follow-ups reproduce the identical erroneous candidates and reject them;
a later repair prohibition is also rejected. Both selected final follow-ups
refuse complete output. The original acceptances, intermediate failure and final
refusals remain distinct. These cue categories do not bind participants/events or
provide complete semantic validation, and unrecognized aliases can cause
conservative refusals. The associated synthetic fidelity additions are examined
development counterparts, not new held-out judge measurements.

Exact source states 041 through 048 completed unchanged author generation and
detection on the declared Qwen2.5-1.5B MPS float32 bridge, followed by the
experimental native MLX 4B rewriter. EOS, token round-trip, scored-length and
source/shared-negative-control contrasts passed; fresh-process author replays
reproduced scores. Both returned after texts remain detected at the frozen
diagnostic alpha 0.01; other complete rewrites are refused. Source prose itself
contains ambiguous or malformed details, so detector qualification cannot stand
in for source-quality assessment. The inventory binds these results to their
actual model, backend, guard snapshot and qualified-control diagnostics. One
prompt/key/seed, repeated source texts and a reused control do not establish FPR calibration, native
paper-model parity or private-provider coverage.

The ChoirMark output exposed an honorific masking fault: only `Mr` was protected
while `Smith` fell outside the span after the period. Selected honorific/name
phrases, including initials and capitalized Unicode name words, now form complete
literal spans at sentence start as well as mid-sentence. The same erroneous
output fails the updated fact checker. A strict native follow-up refuses output;
the existing explicit `allow_partial=True` mode retains the middle paragraph
byte-identically and correctly reports `partial_regeneration`. Its separate
author replay still detects the mark, and complete source-detail fidelity of
neighboring rewritten paragraphs is unestablished. Neither partial processing nor
whole-output refusal is credited as faithful removal.

### Declared AAR and SynthID baseline execution

The declared AAR-context3 and SynthID-context3 baseline configurations completed
generation, source/control detection, strict native library rewriting and
fresh-process author replay. Their original project/configuration declarations,
canonical source-tree digests and complete materialized trees were verified
before import. No author processor or detector was edited, and neither baseline
requires auxiliary signals. The tree verifier does not enable the separate
entropy-dependent TextSeal baseline.

The Qwen2.5-1.5B MPS float32 generation bridge retains stochastic sampling after
the CPU author processor. Both sources complete with EOS, round-trip to their
scored token IDs and pass the scored-length gate. The unchanged shared negative
control passes the frozen diagnostic alpha 0.01; both marked sources are
detected, and independent author replays reproduce the scores. AAR's numerical
source p-value is recorded as returned, without interpreting its reported zero
as an exact zero population error rate.

The current native library refuses both complete rewrites after three attempts.
There is no after text to assess for semantic fidelity or detector evasion.
Source wording itself is sometimes awkward; qualified detector contrasts do not
establish source quality, independent FPR calibration, native paper-model parity
or actual private-provider coverage. The submission inventory retains these
baseline outcomes separately from selected-package experiments and metadata-only
audits. No production model, prompt or backend default changed in this phase.

### Teacher-forced signal consistency

The reference adapter now checks the author's leading zero sentinel, token and
vocabulary types and bounds, per-position nonzero target probabilities, unique
and sorted top-eight rows, and agreement between a target probability and its
row entry. A target absent from the top eight cannot be more likely than the
last entry. Float32 tolerance and clamped tail ties remain accepted. These
checks reject structural and internal consistency errors; they cannot establish
model identity or detect every jointly shifted or stale array. Qualified model
experiments must additionally freeze the exact token arrays, model files,
provider source and configuration, and compare against a separate teacher-forced
forward pass. The source-state adapter's plain-signal gate remains in place.

Regression cases were exercised against the preceding implementation and against
intentional mutations of the new checks. This validation change alone does not
establish faithful watermark removal or enable previously unsupported methods.

### Exact AccentMark and RowMark weighted execution

Exact source states 049 and 050 completed the unchanged author generation and
weighted detection path with qualified EOS, token round-trip and source/shared
negative controls. The frozen Qwen2.5-1.5B MPS float32 bridge retains sampling
after the author CPU processor. AccentMark receives the declared token
log-probabilities; RowMark additionally receives the declared author top-eight
ID/probability rows. Direct local-HF provider calls avoid the outer provider's
zero fallback. Each signal is checked against a separate teacher-forced forward
on those exact tokens, including the previous-position alignment and the author's
probability floor. Fresh-process replays recompute the signals and reproduce the
author scores. The inventory records the frozen diagnostic values and source
hashes. The common negative control is reused, not an independent FPR sample.

Both complete native library rewrites are refused. RowMark accepts intermediate
paragraphs before refusing a later cluster; an accepted intermediate paragraph
omits listed objects and changes borrowing into lending. Whole-output refusal is
not credited as removal, and intermediate acceptance does not certify fidelity.
Examined, disambiguated borrower/lender-role and listed-object development pairs
record these gaps. The current fact checker accepts both negative examples;
they do not establish an improved checker or new held-out judge accuracy.

The reusable source-state replay command still rejects weighted methods. These
research executions do not silently enable its plain detection route, transfer
coverage from the selected package, establish native paper-model parity, or
verify inaccessible provider detectors. Production defaults remain unchanged.

### Optional parsed English loan-role check

`remove_ai_watermarks.text_roles.verify_text_rewrite_loan_roles` supplies an
optional callback helper for selected parsed borrower/lender reversals. It uses
an explicitly supplied local English dependency parser, not another rewriter.
Active/passive borrow/lend/loan clauses share borrower/lender roles; exact
participant lemmas and available item lemmas bind comparisons. Selected subject
control, coordination, recipient passives, datives and object-attached source
prepositions are handled. A complete parse must retain each exact input text;
missing/partial annotations, empty documents and stale/truncated text raise
instead of becoming acceptance. Importing the helper needs no ML dependencies.
The default rewriter, dependency groups and acceptance policy are unchanged.

An examined native RowMark trial with the unchanged current library rewriter
reproduced an intermediate borrower/lender reversal accepted by the existing
hard guards. The callback rejects that candidate inside the library retry loop.
The next candidate passes the selected role check but still changes other
wording; a later cluster is refused, so there is no complete marked output or
removal success. A separate short control returns prose after rejecting an
unsupported lending-role change, but its return-after-use condition becomes
less explicit. Whole-text fidelity remains uncertain, not certified.

Pinned `en_core_web_sm` and `en_core_web_trf` 3.8.0 model runs are kept separate.
The small model misparses some loan and dative examples; the transformer model
handles the examined additional constructions. Both reproduce the recorded
native callback decisions. Frozen dependency trees exercise rule behavior
without requiring models in unit tests. These are examined development inputs,
not held-out parser accuracy or proof of complete role binding. In the frozen
English fidelity corpus the helper rejects a selected role-change error that
the existing fact helper accepts; many other semantic errors remain accepted.
Matching roles do not establish object preservation, ownership, coreference,
negation scope, scheduling or general semantic equivalence. No default model or
semantic gate is adopted from these diagnostics.

### Exact UnisonMark and ConcordMark canonical source states

Exact states 051 and 052 completed unchanged author processing and weighted
scoring through the Llama3.2-1B float32 MPS bridge. The complete tokenizer is
canonically equal to the pinned declared Llama3.1 reference, including special
tokens and merge ranks. The source-state asset helper verifies the selected
nonidentity, idempotent, minimal-representative map and rejects foreign token
maps, missing modes, malformed arrays and unavailable assets. Both published
archive schemas are supported: the earlier six-array UnisonMark asset and the
later asset with the additional sense array. No embedded build-tokenizer identity
is claimed. Generation and fresh detectors also check that the actual loaded
map equals the published selected array instead of falling back to raw IDs.

Direct unchanged author logprob/top-eight providers are checked against separate
teacher-forced forwards. Fresh-process replay recomputes the signals and
reproduces the source/shared-control contrasts. The reused unmarked Llama control
has the same frozen prompt, including its synthetic chat-template date; it is
not an independent FPR sample. Source EOS, text/token round trips and scored
lengths qualify both diagnostic contrasts. The inventory records the scores,
model revisions, map hashes and scope of the bridge. No native vLLM/paper-model
or private-provider parity is established.

The complete UnisonMark rewrite is refused. ConcordMark returns a complete text
that the unchanged author detector does not detect at the frozen alpha, but a
communal notebook becomes a personalized notebook for each tool. That meaning
change was recorded before after-text detector replay. It is not credited as
faithful removal. Examined, disambiguated shared-record/per-item-record pairs
record this known fact-checker gap; earlier judge/parser/competitor measurements
retain their previous corpus hashes. The reusable replay command remains
plain-only and does not silently enable these weighted methods.


### Default-entropy OmniSeal and TextSeal execution

The inventory now records exact author-source executions of candidates 026
through 036 and the declared TextSeal baseline. The original processors,
configurations, detectors and default Gemma entropy provider are unchanged.
Nine runtime files from `daniel-dona/gemma-3-270m-it` revision
`99073d6b6edb0e298d163f964ec2b9d970b3408d` match official
`google/gemma-3-270m-it` revision
`ac82b4e820549b854eebf28ce6dedaf9fdfa17b3` byte identities: LFS SHA-256
or ordinary-file Git blob hashes, followed by local file verification. This
claims runtime-file identity, not identical model cards or complete repositories.

The author provider consumes raw Qwen generator IDs, without decoding and
retokenizing into Gemma IDs or adding BOS. Its unchanged loader selects CPU
bfloat16 and SDPA in the recorded environment. Entropy uses the author's
float32 log-softmax and reduction, with a zero leading sentinel and next-token
alignment. Every source/control signal is independently recomputed in a fresh
process and exactly matches the frozen author arrays; each detector explicitly
reports weighted mode. A separate model forward uses the same float32
normalization and independent float64 exponential/summation to check entropy.
Fully float64 normalization differs from the author calculation; that diagnostic
is retained rather than replacing the declared numerical method.

The reusable `validate_entropies` helper rejects malformed types, alignment,
sentinels, nonfinite values, zero fallbacks, invalid raw token IDs and impossible
entropy bounds. These structural checks do not establish model or source
provenance; the frozen file hashes and independent inference supply separate
evidence. Deliberate mutations demonstrate the new assertions. The public
plain-only replay command still rejects entropy-dependent configurations.

All eleven candidate sources pass the declared source/shared-negative-control
contrast and generation qualification. Native library rewriting refuses complete
output after three attempts per source. Attempt records expose dropped daily
frequencies, conditions and protected markers; no complete faithful removal is
credited. Candidates 026 through 029 produce identical source text and scores,
so they are separate source-state executions, not independent text samples.
Candidates 030 and 034 also share that source text, with different configured
weights and scores. The frozen summary records all six duplicate-source states.

The TextSeal baseline completes generation and independently reproduced weighted
scoring, but fails exact generation/tokenizer round-trip qualification. Its low
source p-value cannot qualify this trial, and rewriting is not attempted. The
failed frozen input is retained without detector-guided replacement generation.

These experiments use a declared Qwen2.5-1.5B MPS float32 generation bridge,
CPU author processing, and the native library's MPS float16 rewriter. They do
not establish native paper-model/vLLM parity, independent FPR calibration,
private-provider coverage or market leadership. The active goal still requires
effective faithful rewriting and independent full-product comparison. Inventory
coverage of source states must not be reported as coverage of provider models.

### Visible facts, sentence ablations and explicit record scope

Development ablations on frozen previously examined sources compared visible
facts, visible facts with an explicit preservation list, and contextual
sentence-wise rewriting with aggregate paragraph checks. Each used the library
pipeline and original fact guards plus exact selected-expression multiplicities.
The sentence strategy permits more total model calls and is not a fixed-cost
comparison. An initial sentence-count restriction was removed because splitting
a sentence is not inherently a meaning error; its frozen first answer was reused
in the refinement rather than regenerated. The returned complete texts still
change borrower/lender roles, omit listed objects, change a hanging notepad to a
resting notebook, or distribute a communal notebook per tool. No returned text
is credited as faithful removal, and these experimental strategies are not
adopted. After-detector scoring is not used to rehabilitate a known quality
failure.

The native fact checker now rejects selected explicit English contradictions
between a shared/communal record container and separate containers per item, in
either direction. It distinguishes notebook entries, ledger pages and register
codes from their containers. Missing cues, mixed distributions and bindings to
different people/items remain unresolved. Passing this check is not an ownership
or semantic-equivalence certificate. The existing communal-record development
negative changes from accepted to rejected; other decisions in the frozen
development corpus are unchanged. This is examined regression evidence, not
held-out semantic-judge accuracy.

A fresh native rewrite of the same frozen ConcordMark source keeps the original
model, prompt, masking and retry budget. The new guard rejects an intermediate
per-tool-notebook response that the preceding guard accepts. Complete rewriting
then refuses, so there is no new after output or detector verdict. The inventory
retains the earlier meaning-changed diagnostic and adds the separately hashed
guard follow-up. Subsequent type annotation and compound-word protection changes
are checked by replaying every frozen native candidate with identical decisions
under the final core. Regressions first fail against the preceding implementation; bypass
and over-refusal mutations demonstrate both negative and positive assertions.

### Local edit proposals and indexed sentence replacements

Two development prototypes use the library runtime and custom-rewriter hook on
the same frozen examined inputs. Exact old/new phrase proposals enforce unique
source matches, nonoverlap and selected-fact/technical-token locks. The small
model instead repeatedly edits listed protected expressions or copies source
strings inaccurately, preventing complete output. Accepted intermediate edits
can still change locations or participant roles; formal edit validity is not
semantic validation. An executor review also catches an adjacent-placeholder
boundary gap; the native placeholder guard already rejects the resulting token
loss, and the refined temporary executor rejects it directly.

Indexed sentence proposals remove exact source-string copying and do not expose
a separate protected-expression list. Invalid unit rewrites retain their original
sentences, with retained ids recorded separately from paragraph-level prose-span
accounting. Whole assembled paragraphs still pass native fact/language guards.
The sole returned full candidate retains original sentences and has unresolved
exchange and sign-off wording. Its manual quality review predates scoring and
does not establish equivalence.

The unchanged ConcordMark detector still detects this indexed candidate. Exact
source objects, canonical arrays, full tokenizer compatibility and required
weighted signals are checked with the original pinned generator checkpoint.
Fresh author providers reproduce frozen source/control signals and scores;
separate teacher-forced forwards verify probabilities and top-eight rows for the
new after text. The inventory records this retained-mark experiment separately
from earlier native trials. Neither proposal strategy is adopted, and there is
no new complete faithful removal or market-leadership evidence.

### Matched local runtime and Llama head precision

A local Qwen3-4B GGUF transport experiment reuses the unchanged library rewriter
factory, prompt, masking, retries and candidate guards. Every HF vocabulary id
and BPE merge matches the recorded GGUF representation, with unused padding ids
explicitly outside the equality claim. All observed prompt token counts match.
Raw local generation receives the library-rendered chat prompt; a counterfactual
completion test verifies that length-limited replies trigger the native retry
instead of becoming successful EOS stops. This remains a research transport,
with no shipped backend or numerical parity claim.

The returned RowMark and ConcordMark development candidates remain detected by
their unchanged weighted author oracles. Manual review also finds approval-signal
direction and manner-to-event binding changes. The inventory records separate
native default retests, raw-transport outputs and their code/input hashes. These
observations do not support a default model or strategy change.

An independent ConcordMark signal check initially compares the author's
32-position Llama output-head multiplication with a full-head forward. On an
examined input, the same hidden states produce small MPS float32 rounding
differences between those matrix shapes. Repeated backbone states are bit
identical; the full wrapper matches the full-head calculation, and the original
provider matches the chunked-head calculation exactly. The checker now performs
a separate backbone forward with the source-declared head chunk shape. The
original tolerance is retained, original author signals and generated outputs
remain unchanged, and frozen marked/control scores reproduce. Signal validation
must bind the declared numerical method as well as token ids and model files.

### Dependency-bound protected phrases

A native MLX development study pairs the existing mask with source-bound
dependency bundles, then adds a readable phrase glossary. The pinned English
parser's installed files match its model wheel. Source-offset binding, exact
restoration, literal marker collisions and stale-parser rejection are checked;
these checks do not establish parser accuracy or semantic equivalence. Neither
bundle strategy returns a complete accepted text on the examined inputs.

A format refinement permits exact protected-phrase expansions only with matching
literal multiplicities and the existing native guard. Removing the count check
admits a forbidden approval-direction change in a deliberate mutation. Reusing
the earlier first-paragraph replies recovers one paragraph, but subsequent
attempts still fail complete-output acceptance. The refinement is adaptive
development evidence, not an independent cohort or a shipped strategy.

The original-mask Qwen3-4B MLX baseline returns OmniSeal and RowMark candidates.
Pre-detector review identifies borrowing becoming lending in the former and
leaves loan direction and detail preservation unresolved in the latter. Both
remain detected by their unchanged weighted author oracles. Source identities,
model files and required signals are verified, separate forwards cross-check
signals, and frozen marked/control scores reproduce. No complete faithful
removal is credited. The ConcordMark baseline refuses, with no new after score.
The study does not change production masking, prompts, models or APIs.

### Larger MoE capacity and explicit signal direction

A pinned Qwen3-30B-A3B-Instruct-2507 MLX quantization runs the same examined
OmniSeal, RowMark and ConcordMark sources through the native library factory.
One paired ablation changes only the sampler to argmax. The installed runtime's
sampler identity and an argmax probe verify that intervention; disabling its
actual configuration branch fails the preflight expectations. Model files are
verified against the pinned repository's LFS SHA-256 and Git blob identities.

Both native and greedy OmniSeal rewrites refuse. RowMark and ConcordMark return
complete candidates in both modes, but pre-detector manual review finds signal
source/recipient reversals or lost inspection speed. Every complete candidate
remains detected by its unchanged weighted author oracle. Required signals are
independently checked with the declared numerical methods and frozen source and
control scores reproduce. These observations do not establish a better model
default, paper-model parity, an independent holdout or private-provider coverage.

A second development phase, fixed after the meaning review and before new keyed
after scores, applies the previously examined dependency bundles and readable
glossary with strict exact-expansion recovery. The pinned parser files match the
preceding wheel proof. Every generated candidate restores to its unchanged
source paragraph, so nonidentity rejection prevents counting copying as removal.
These adaptive constraint trials remain separate from the first paired phase.

The library fact checker now rejects selected explicit English signal-direction
contradictions, including a nod `from the community` becoming a nod `to the
community` or `in return to the community`. It compares the same literal signal
and recognized participant when both directions are unambiguous. Missing cues,
different participant descriptions and mixed directions abstain. Compound
participant descriptions are not shortened to recognized prefixes. This does
not establish general approval actors, ownership or modifier-to-event binding.

Regression cases fail against the preceding core. Bypassing the new extractor
fails the negative cases; blanket rejection fails the equivalent/unresolved
controls. Frozen candidate replay changes the two observed RowMark false
acceptances to rejection. A native RowMark rerun reproduces the initial prompts,
replies and stops until the changed guard decision, then retries that paragraph
and refuses the complete output. This prevents a meaning error, not successful
removal. New examined signal-direction and inspection-speed development pairs
retain earlier judge and competitor reports under their original corpus hashes.

### Published curated SynthID source and control traces

A further reference replay binds the published curated English corpus to its
pinned Qwen2.5-14B tokenizer and exact serialized CPU sampling table. All marked
and clean token sequences, complete context-repetition masks and g-value arrays
reproduce. The current library's reference scorer agrees with the exact g-value
means and the published decision threshold. The coverage catalog retains the
source URL, input identities and replay-report hash. This adds a reference
corpus, not a newly tested commercial provider or independent population FPR
calibration. The sources were curated after generation and are not represented
as untouched generation outputs.

The current default Qwen2.5-1.5B native factory is run on every marked source in
that fixed corpus with its pinned snapshot, MPS float16, original prompt/masking,
seed and per-cluster retry policy. Every complete document refuses acceptance.
The observed candidates lose protected markers and selected facts. This is an
availability failure, not a successful-removal result or a fidelity pass.

A separately frozen adaptive format ablation exposes the same native factual
literals inside visible tags. The decoder checks exact and case-insensitive
literal multiplicities, rejects malformed keep tags and runs the original native
guard on restored text. Disabling the actual multiplicity rejection admits a
forbidden protected-literal change in the format preflight; that probe uses a
guard stub and is not semantic certification. Some intermediate paragraphs pass,
but every complete document still refuses. Prompt length, output encoding and
total calls differ, so this is not a cost-matched visibility-only comparison.
No complete native after score is claimed and no default format, model or API
change is adopted.

### Larger-model corpus trial and technical-capability false refusal

The same ten frozen curated sources are evaluated with a pinned local
Qwen3-30B-A3B-Instruct-2507 MLX quantization, first with the native mask and then
with the previously examined visible native-fact format. All twenty complete
outputs refuse under that frozen core. Model files, source identities and the
unchanged source/control traces are verified before generation. The corpus is
examined development evidence, not fresh holdout or provider coverage.

Review exposes a false refusal: the system allowing staff to record information
is paraphrased as the system enabling staff to document information. A literal
permission cue is lost, while the observed technical capability is retained.
The fact checker now normalizes only matching selected sentence-initial technical
frames for this comparison. Subject, actor, functional-action family and bounded
object must match, with a small local alias set. Explicit permission elsewhere,
human/policy subjects, access verbs and unmatched/repeated frames retain ordinary
cue checks. This is a selected ambiguity exception, not general authorization
or semantic-equivalence inference.

The unchanged regressions fail against the preceding core. Bypassing the helper,
removing every allow cue, or deleting subject/actor/action/object binding from
its actual key each fails the relevant tests. The preceding development corpus
has unchanged decisions; the new calculator capability positive now passes.
Native and visible-format regression reruns reproduce the initial six
prompts/replies/stops exactly, then accept the formerly rejected capability
paragraph and complete the document. This targeted rerun is separate from the
earlier full-corpus experiment and does not silently replace its core hash.

Full-text review before new after scores leaves the native result unresolved
because qualifier and temporal wording equivalence is unproven. The visible
format result changes device availability for installation into an assertion
that installation began. The native result remains detected; the visible-format
result is not detected but has that meaning error. Fresh unchanged author-score
replay and the library scorer agree, with original source/control traces
reproduced. Neither result earns faithful-removal credit. No default model,
format, backend or API change is adopted; only the selective false-refusal
correction enters the working library.

### Source-bound nominal and verbal anchors

Three adaptive local phases use all ten fixed curated English sources with the
same pinned Qwen3-30B-A3B MLX model and native generation/retry policy. Source
tokenizers, sampling table and complete marked/control traces reproduce. Parser
model files match their pinned wheel proof; exact library paragraph inputs,
token offsets, dependencies and protected spans are frozen before each phase.
This verifies identity and examined coverage, not held-out parser accuracy.

The first phase shows nominal bundles inline, then adds availability/readiness,
planning/modality/negation and temporal anchors. A deliberate predicate-extension
bypass fails the coverage probes; the protected availability example rejects
conversion into an actual installation start while an unrelated passive rewrite
still passes. Only one complete nominal output returns. Pre-detector review
finds announcement/launch wording and development becoming implementation. The
unchanged author reference and library scorer agree that the output remains
detected. No faithful-removal credit is assigned.

The second phase uses identified readable fields, validates every id and an
ordered echo of source content, then restores the exact source field values.
Missing, foreign, duplicate, nested and malformed identifiers are rejected.
Bypassing the actual echo guard admits a role-swap probe through the native
guard, showing why identifier presence alone is insufficient. Reporting and
perfect/progressive source anchors are added before this phase's generation.
Both variants refuse complete rewrites. No keyed after feedback is used for refinement.

The third phase reduces wide headed-preposition spans to local preposition and
coordinated object groups, binds numeric heads and keeps selected predicate
chains compact. Every source round trip is exact, but all complete outputs
still refuse. The observed failures include omitted ids and altered protected
expressions. These results do not isolate a universal cause or prove that the
formats fail on other models. The phases change prompts, span rules and encoding;
they are not cost-matched independent cohorts or new commercial-model coverage.
No prototype is adopted into production masking, APIs or model defaults.
Examined development/implementation fidelity pairs retain earlier measurements
under their archived corpus hashes.

### Native dense Mistral family and currency boundaries

The curated-source catalog followup binds a pinned local Mistral Small 3.2 MLX
conversion to its verified file identities and actual native loader/template.
Every source runs with the unchanged library prompt, mask, sampler and guard
policy. Source/control tokenization, sampling table, complete masks, g-values
and decisions reproduce before generation. This is text-only runtime execution,
not numerical parity with the original model runtime or a multimodal study.

One complete initial output returns; full-text review before after scoring finds
a missing announcing subject and malformed staffing/iteration price relations.
The unchanged author scorer and library reference agree that it remains detected.
Other complete documents refuse. No faithful removal is credited.

The traces expose a segmentation defect: two currency dollars lock intervening
prose as inline math. Dollar math now requires non-whitespace at both inner
delimiter edges and no digit after its closing dollar, following the selected
[documented dollar-boundary rules](https://pandoc.org/MANUAL.html#extension-tex_math_dollars).
Currency pairs remain prose while genuine numeric and symbolic formulas stay
locked. New regression assertions fail on the preceding implementation; bypassing
each boundary and disabling dollar math fail their respective controls.

A targeted known-source native rerun after the correction reproduces the first
eight raw generations, then reaches an ordinary price-bearing prose paragraph.
It refuses rather than returning a complete text. Recorded differences include
`per week` versus `weekly` receiving different recurrence signatures. Other
examined traces show `annual`/`yearly` and `four-week period`/`four weeks` signature
asymmetries; these remain separately recorded limitations, not relaxed by the
dollar fix. No default model or backend changes. This followup reuses development
sources and does not add independent texts, provider coverage or market ranking.

### Selected duration/frequency signatures and numeric-style ablation

Observed native attempts expose three asymmetries: `four-week` versus `four weeks`,
`annual` versus `yearly`, and a bare `per week` rate versus `weekly`. The library
now recognizes selected numeric English hour/day/week/month/year compounds as
whole measurements, an annual recurrence alias and bare per-unit periods. A
word/hyphen boundary prevents annual being read inside semi-annual or non-annual.
This does not interpret arbitrary compound schedules or general subtraction.

The preceding core fails the new regression bank; removing each new branch,
relaxing the frequency prefix boundary, bypassing measurement/frequency checks
and disabling masking all fail corresponding controls. The preceding development
bank keeps its decisions. Added examined positive/negative duration, recurrence
and monetary-period pairs change in the intended direction; previous judge
measurements retain their original corpus hashes. Frozen known candidate replay
removes selected parser refusals without certifying those candidates' full meaning.

Three known-source native Mistral reruns use the same pinned model, unchanged
prompt/sampler/retry policy and corrected recognition/masking. Source/control
tokenization, table, masks, g-values and decisions reproduce. The garden and
museum documents advance past their initial paragraphs, but all complete outputs
still refuse. There is no native after text or successful removal to score.

A separately frozen deterministic callback ablation converts only selected
unambiguous English integer words greater than one to ASCII digits, skipping
recognized names and hyphen-boundary ambiguity. It runs through the library's
custom-rewriter API with explicitly requested partial mode: unchanged clusters
are retained and reported. One source has no eligible edit and yields no output.
All edits in the returned outputs are reviewed in context before scoring. An
explicit manually checked value map and a separate unchanged-region byte walk
verify that no other text changed. This supports fidelity for these examined
outputs, not general numeric disambiguation or a default implementation.

The unchanged author detector and library reference agree that every returned
numeric-style output remains detected. This establishes a meaning-preserving
failed ablation, not watermark removal. No prototype numeric API, model default,
backend or production policy is adopted. These are repeated development sources,
not new independent texts, commercial models or market-leadership evidence.

### Selected initial appositive names

An observed native mask leaves the first name outside the protected surname:
`Robert [[...]]`. A labeled synthetic mutation of that actual source changes the
first name while preserving the surname and passes the preceding native guard.
The selected initial ASCII appositive extension now masks and checks the full
phrase, such as `Robert Johnson, the curator`. Common leading function words
and numeric words abstain. Single names, arbitrary initial runs, initials,
non-ASCII names and general entity/role resolution remain outside this extension.

The new regression bank fails on the preceding core. Actual extraction,
checking, masking and leading-word bypasses fail their corresponding controls.
A separate CR-only boundary omission is exposed by a failing assertion and
corrected; LF, CRLF and CR name boundaries are tested. The preceding development
bank retains its decisions, while the added examined wrong-first-name pair is
rejected. Earlier model/judge/parser results retain their original input hashes.

A fixed known-source native Mistral rerun uses the same pinned files, unchanged
prompt, sampler and retry policy. Its first raw generation matches the preceding
trial exactly. The later name cluster retains the full person marker in every
attempt, but the complete rewrite still refuses. Remaining observed problems
include an omitted department component, download replacing required installation,
and invented seamless functioning or a scheduled entry time. No full after text
or successful-removal score exists.

Generation precedes the CR-only follow-up, and the catalog keeps both module
hashes. On the final core, every recorded LF mask, factual map and guard decision
reproduces exactly. Source/control detector traces also reproduce. This is frozen
trace replay, not fresh final-core model inference or full semantic certification.
The change does not adopt a model default, relax other checks or add provider
coverage, a new holdout or market-leadership evidence.

### Actor apposition and separate participants

A local sentence paraphrase trial found two constructions where a descriptive
apposition still appeared with the original role label while the rewrite
made that description a separate participant. One rewrite split the original
officer into two people in a greeting scene. Another made the officer the
thing handed over. A second beam made the woman the recipient of the handover.

The built-in fact guard now rejects these contrasts for selected English
actions, initial title descriptions and literal matching participants.
Active-to-passive and reordered apposition controls preserve the original
reading. The new paired tests fail on the earlier library and pass with the
rule. This is a bounded check for a known failure mode. General agent,
patient, recipient and pronoun resolution require separate evidence.

### Selected explicit event phases

A bidirectional entailment comparison accepted a candidate that changed finding
shelter into seeking refuge. The native guard now compares a bounded set of
English plans, attempts and completions when both clauses name the same actor
and object. It recognizes explicit completed passives with a stated actor.
The shelter frame separately matches the actor and stated location.

The paired tests cover shelter, repair, reopening, restoration, delivery,
unlocking and submission. Matching voice changes and lexical variants pass.
Unmatched objects and actors, mixed phases, modal wording, questions and
different locations abstain. This is a local fact check for supported frames.
It can miss omitted events and cannot assess broader entailment.

### Selected weekend timing expressions

The fact guard compares explicit weekend timing in adverbial clauses.
It groups single-occasion variants such as “on the weekend” and “over
the weekend” while preserving distinctions among relative weekends,
recurring weekends and whole-weekend duration. It leaves noun uses such
as “the weekend shift” to other checks. Local regeneration protects the
complete recognized phrase as one literal span.

A saved KGW rewrite changed a weekend event into a week-long activity.
The earlier guard accepted that paragraph. The updated guard rejects
the saved change and retains the identity and equivalent phrasing
controls. This is a regression check for the observed timing change.

### Qualified influence claims

An examined local rewrite replaced the possibility that paper surface
absorbency affects drying with a direct assertion about the same process.
The fact guard now compares selected affirmative English influence clauses
with matching factors, effects and conditions. The observed rewrite fails
against the current guard. Modal and equivalent wording controls pass, and
different factors, effects, conditions and mixed evidence abstain. This
check leaves general modal scope and unsupported sentence families open.

### Dense Qwen27 on the curated SynthID corpus

A pinned Qwen3.8-27B MLX checkpoint, previously executed on other English/Russian
marked texts, runs all fixed curated sources under the current native library
policy. Every model file is verified and all previously retained model-file
hashes match. The original source/control detector traces reproduce exactly.
The installed text loader and actual template execute with thinking disabled.
No new commercial model, untouched generator or original-model parity is claimed.

One complete output returns. Full-text review before after scoring identifies
each-type maintenance-checklist coverage becoming various types, initially
successful repairs becoming seemingly successful, and damaged referential/channel
wording. The unchanged author detector and library reference agree that this
output is no longer detected at the published threshold. It fails fidelity and
receives no successful-removal credit. Other complete attempts refuse.

The added examined each-type/various-types pair records a semantic-check gap
on that generation core: selected numeric/name/polarity checks can still accept that coverage loss.
The positive counterpart and earlier judge/model measurements remain bound to
their recorded corpus hashes. This addition is not a new held-out quality study.

### Newly inventoried research references

The catalog now separately records SAMark, SWAN, OpenStamp and SemField. Pinned
public source inventories were retrieved for the first three. The declared
SemField source URL returned HTTP 404; this observation does not rule out other
implementations. OpenStamp's listed checkpoint metadata and detection-artifact
filenames are retained; subsequent checkpoint-specific execution is recorded
separately below and in the catalog. None of these entries inherits a SynthID,
MarkLLM or AutoMark result. Semantic-structure/embedding and weight-level schemes require their own
unchanged author detectors, controls and meaning review.

The SAMark paper's [Algorithm 2](https://arxiv.org/html/2605.25796v3#S3.SS3)
has a separate response-matrix implementation in `_internal/samark.py`.
The scorer takes supplied cosines, infers flags and returns the equation's
statistic without assigning a detection threshold. Synthetic controls check
sentence permutation, tied channels and undefined constant-alignment cases.
On a simulated symmetric response distribution, estimating flags on the
scored document raised the exceedance rate above the nominal standard-normal
tail. Set thresholds on separate clean examples for the exact encoder,
text segmentation and key regime. The author code, text embeddings and
watermarked outputs have not been replayed by this mathematical check.

The SWAN source at revision `2ca0c2071c8d0c6bb3739b2b2bcfb34cfe63eb2a`
explicitly carries [CC BY-NC 4.0](https://github.com/amazon-science/SWAN/blob/2ca0c2071c8d0c6bb3739b2b2bcfb34cfe63eb2a/LICENSE).
The catalog marks it `blocked_license_clearance`: permission for this
product-oriented execution or redistribution has not been established. Its
author code has not been executed or included in the package. Independently
testing a separately licensed AMR parser cannot qualify SWAN's watermark
generator, detector or removal performance.

Separate rewriter references record RLCracker, the published adaptive-evasion
paraphraser collection and a masked-word substitution baseline. Their source
inventories are not executed trained-policy comparisons or full-fidelity results.
In particular, a sentence-similarity threshold does not establish factual
equivalence. These additions expand the explicit unfinished evaluation scope;
they do not turn reviewed papers or metadata into tested model coverage.

### OpenStamp artifact and numerical preflight

The OpenStamp catalog entry records a separate staging step for its published
SmolLM2 checkpoint and the base named in that checkpoint's README. The named base
revision is independently frozen because the author card does not specify one.
Git/LFS checksums cover both sets of artifacts. Tensor-by-tensor comparison finds
the shared input/output embedding changed while the remaining serialized tensors
match. This is artifact evidence, not source/control generation qualification.

Unchanged author LLR methods run on fixed synthetic hidden states and logits
against an independently summed float64 oracle. Real source mutations of the
prediction shift, sign, padding denominator and softmax axis fail the same fixed
comparison. The author scores labels from the third input position onward; an
input with no eligible positions yields nonfinite scores and must be rejected by
a future integration. Neither short input nor missing output earns clean status.

A tiny synthetic tied-weight model confirms that the unchanged author generation
constructor copies the modified head into the shared input embedding as well.
Replacing that actual copy with a no-op fails the assertion. The released selector
is finite BF16. Replaying the original BF16 matrix/head construction on CPU is
close to the released tensor but is not an exact match; the discrepancy remains
explicit in the catalog, with no post-hoc tolerance promoted to parity.

No full OpenStamp language model has run in these preflights. Qualified native
generation, detector controls, threshold calibration and library removal remain
unexecuted. None of the numerical or artifact checks adds a tested-model pass.

A subsequent diagnostic computes exact rational sums of the selector components
at the discrepant head coordinates, then independently compares neighboring BF16
values for each staged rounding operation. Fixed binary-fraction controls pass,
and removing the even-mantissa tie preference fails an actual function mutation.
This diagnoses the observed arithmetic without claiming the original device's
accumulation behavior or silently changing the author algorithm.

The original `from_augmented` constructor provides a separate published-head
path. On real released head tensors in a minimal tied-weight container, it
reproduces the saved shared embedding exactly; bypassing its actual copy fails
the same check. This is not a full causal-model forward pass. Tokenizer JSON
structures match after converting older string BPE merge pairs to the newer
pair-list representation. Padding and raw model-configuration differences remain
explicit. Future native qualification can use the released head through this
author path while retaining the nonexact CPU `from_config` observation separately.

### Completed first Qwen27 thinking mode

The first frozen thinking mode has every curated source terminal. It uses the
original library factory, prompt and acceptance policy with explicitly adapted
native thinking templates, author-recommended sampling and strict final-channel
projection. The readable-source mode is a separate unfinished part of the same
predeclared experiment. These are not unchanged-default or equal-cost comparisons.

The returned full text adds unsupported paper medium to the original manual
booking process. Its full-text development review precedes keyed after scoring.
The unchanged author detector and library reference still classify it as marked,
and it fails fidelity. Other complete attempts refuse after acceptance checks.
All recorded generations finish normally with final projections. Replaying saved
candidates reproduces every observed guard decision; selected-fact checks account
for the rejections. Intermediate accepted clusters are not complete outputs.
The catalog phase-summary/review/replay hashes bind these results separately from
the earlier nonthinking cohort and still-running readable-source mode. No
faithful removal is credited.

The original scheduling sentence has ambiguous date attachment, so its
rephrasing is recorded as uncertainty, not a proven date contradiction. Separate
examined synthetic event/date binding pairs and a medium-addition counterpart
record unresolved checker gaps. Earlier calibration measurements retain their
recorded corpus hashes. These additions are not a new held-out study.

### OpenStamp BF16 accumulation fingerprint

The original author loader selects BF16. A separate fixed synthetic preflight
reproduces BF16 log-probability sums and their difference, followed by division
by a float32 valid-position count. Independently summed exact rational values
with staged BF16 rounding reproduce the original scores. Promoting the actual
log-probability sums to float32 changes the results and fails the same fixed
comparison. This is a numerical fingerprint of the author algorithm, not an
endorsement of its precision or an independently calibrated detector. The oracle
starts from gathered BF16 values; it is not a float64 full-forward likelihood
calculation. Native qualification must preserve that distinction.

### Applied selected English type-coverage guard

The reusable library now recognizes selected explicit each/every/all type
phrases, preserves their mention count and masks the same whole phrases in local
rewriting. The examined each-type/various-types loss now fails the fact check. A
regression also feeds the originally accepted native Qwen paragraph to the shared
local guard and confirms rejection. Language identification is isolated in that
regression; its own tests remain separate.

The unchanged regression assertions fail on the preceding core and pass after
the fix. Actual production-source checker, signature, mask and boundary mutations
fail their corresponding assertions. Existing text tests, style and type checks
pass. In the current examined development corpus only the known type-coverage
negative changes decision; the faithful counterpart and earlier cases stay
unchanged. Earlier isolated-candidate measurements retain their original corpus
hash. This is not a new held-out study or a fresh model run on the corrected core.

The check preserves selected mention multiplicity conservatively. It does not
bind a type to a participant, recognize every quantifier/paraphrase, fix uppercase
name false rejections, or certify meaning. Unsupported paper medium and swapped
event/date assignments remain recorded gaps. The original Qwen generation
results remain bound to their earlier core; this guard does not retroactively
change those model outcomes or establish successful watermark removal.

### Completed readable-source Qwen27 thinking mode

The predeclared readable-source mode is now terminal too. It exposes actual
source wording through an explicitly empty factual mask while retaining native
acceptance checks and outer technical-span protection. Full-text review occurs
before keyed after scoring. The complete returned document strengthens potential
conflict to likely conflict and weakens immediate repair into priority handling.
It falls below the frozen author threshold, but fails fidelity and earns no
faithful-removal credit. Fresh original source/control traces and author/library
after-score decisions agree; frozen guard decisions replay on the generation core.

Both modes are complete. Their returned texts concern the same curated source,
not additional independent documents or providers. The experiment combines
explicit thinking/template, sampler, token-budget and final-projection adapters;
it is not an unchanged default, equal-cost comparison, new holdout or proof of
market leadership. The later type-coverage fix has a separate production core and
does not retroactively change this experiment.

### Native OpenStamp source/control qualification

The released SmolLM2 OpenStamp checkpoint now has a qualified local source/control
cohort. The original author loader, generation function, published-head
constructor and detector execute without edits. Explicit adapters provide local
paths/CPU placement, fixed prefixes from cleared fictional reports, paired seed
resets and the released head rather than nonexact CPU reconstruction. The author
requirements and complete runtime package versions are frozen separately.

Every clean and marked continuation is reported, with no rerolls or favorable
source selection. Original scores separate both conditions at the illustrative
author zero threshold. Original per-token traces independently replay through
exact rational sums and staged BF16 rounding followed by float32 division.
Serialized base parameters match before generation, and the original tied
embedding is restored before the unchanged detector scores either condition.
The protocol and result hashes are retained in the catalog.

This tests one released checkpoint. It is not independent full-forward numerical
parity, untouched data, a calibrated population false-positive rate, other
OpenStamp models, private provider verification, library removal or meaning
preservation. The complete marked cohort enters a separately fixed library
rewrite experiment; quality review must precede its keyed after scores.

### Native library rewriting of the qualified OpenStamp cohort

Every qualified marked continuation enters the fixed native-library rewrite
plan on the later type-coverage core. The same pinned Qwen27 files, default
system prompt, factual mask, sampler, retries and seed are used, with observation
only. Detector values and keys do not enter rewriting. Complete visible-text
reviews precede unchanged author after scoring.

The below-threshold outputs add unsupported residence or narrow individuals to
residents. They fail fidelity and earn no credit. The visible-text faithful
output only inserts an initial article and remains marked. Other full returns
remain marked, including ones whose malformed timing/condition clauses cannot
be confidently judged. Refusals are unscored. Raw malformed/truncated sources are
kept in the cohort rather than filtered for favorable results; indeterminate
fidelity is distinct from demonstrated contradiction and receives no credit.

The original source/control cohort is reused by exact artifact identity, and
a fixed clean/marked pair replays fresh before after scores. Original after
scores independently replay through their gathered BF16 traces and staged
rounding. This is not independent full-forward parity, calibrated population
FPR, another checkpoint/provider or market-leadership evidence. No successful
faithful removal is credited. The catalog retains exact outcome and report hashes.

New examined synthetic work/residence, beneficiary-class, goal/achievement,
likelihood and immediate/priority pairs record current guard gaps. Their positives
and negatives are not a held-out judge study or completed semantic fix. Earlier
judge/model/parser metrics and the type-coverage comparison retain their original
corpus hashes; the expanded calibration bank does not rewrite those measurements.

### Published EXP-tuned adapter: loaded controls and a fixed comparison

The published Qwen0.5 LoRA adapter and its named base are checksum-verified. The
base revision is independently frozen because the adapter config does not state
its training revision. Loaded adapter tensors match the file; original base
parameters stay unchanged. Disabling the unmerged adapter reproduces pristine
base probe logits and greedy tokens exactly; reenabling changes probe logits
and restores the original active result. Disabled-as-active fails the same
activation assertion. This is a loading/probe check, not watermark removal.

The model-card paraphrase function, template and postprocessor run unchanged in
a recording preflight. Actual cap, delimiter and output-slice mutations fail.
The completed fixed comparison uses native library policy and the author function
as an explicitly guarded library custom rewriter. Both have paired base-only
controls. All processing passes through the library; no new production default
or equal-cost/mask-only comparison is claimed. Reviews must precede after scores.

A new fixed source/control cohort uses unchanged original MarkLLM EXP and
BaseWatermark methods with explicit CPU, first-paragraph fiction prompts, raw
continuation-token slicing and sampling adapters. Every source/control pair is
reported. Both conditions separate at the original config threshold. Original
per-token scores independently reproduce via prefix-product seeding and an
integer-shape gamma survival sum. Actual seed, sampler and survival-tail mutations
fail fixed numerical controls. This is the intended EXP family; exact training
implementation/key parity and population FPR are unverified. Reused qualified
SynthID and OpenStamp corpora remain separate, with their original identities.
No faithful-removal credit follows from the loading/source preflights. The completed
model comparison is reported below, separately from those checks.

### Complete paired EXP-adapter and lexical-substitution comparisons

The paired adapter comparison executes the fixed four policy/activation variants
on the same qualified OpenStamp, SynthID and original EXP development sources.
Native policy returns no complete output for either the base or adapter. The
guarded author function returns two base outputs and one adapter output. Full
meaning reviews are frozen before after scores. All three change meaning:
workshops replace sharing experiences with judges, employee feedback becomes
customer feedback, or matching the prior year becomes surpassing prior years.
All become negative under the original EXP detector, but none earns faithful
removal credit. Fresh original EXP source/control replays reproduce their
per-token contributions and decisions. Counts, variant metadata and artifact
hashes are recorded in `catalog.json` under the adaptive-paraphraser entry.
Only the named checkpoint is evaluated; other family members remain unexecuted.

The offline lexical baseline uses the original BERT candidate and MPNet ranking
functions unchanged, inside the library custom-rewriter and native acceptance
guard. The author supplies model names without revisions; both model revisions
are independently pinned. The replacement fraction applies to eligible words,
not all words. Original character/token caps and threshold remain in place;
technical-span protection can split callbacks, so full CLI execution parity is
not claimed. No callback input exceeds the author character cap in this run.
Controlled fraction, threshold, candidate-limit and ranking mutations fail
actual assertions. Every real cosine operation independently reproduces from
recorded embeddings; that is numerical consistency, not factual equivalence.

All complete outputs receive frozen full-text reviews before original-author
after scoring. Clear changes include series membership becoming criticism,
parents becoming people, custom cakes becoming snack cakes, inventory software
becoming order software, and volunteers becoming a committee. One repair-history
paraphrase remains uncertain and stays detected. Only the bakery output becomes
negative under EXP, with changed meaning. Every OpenStamp output stays positive
at the explicitly illustrative zero threshold, which is not population-FPR
calibration. Fresh EXP controls and an OpenStamp source/control pair reproduce;
OpenStamp after traces independently replay with the preflighted BF16 staged
rounding. No faithful removal is credited. Fixed counts, model revisions and
artifact hashes are in the offline-baseline catalog entry.

These are fixed development cohorts with reused topics. Refusals and uncertain
fidelity receive no credit. Neither comparison establishes commercial-model
coverage, private-detector access, market leadership or a production default.

### Membership/stance acceptance follow-up

The lexical blog output exposes an accepted membership-to-criticism change.
The later shared guard compares selected affirmative determiner-led English
copular frames with matching literal subject/object descriptions. It handles
the observed joining of source line breaks. Different descriptions, unsupported
grammar, modality, negation and mixed claims remain unresolved. The fixed
regression file fails on the archived pre-fix core and passes afterward; actual
guard, relation, participant and mixed-claim mutations fail. Full saved-output
replay newly rejects the lexical blog case. Existing development-bank decisions
are unchanged; added counterparts are development evidence, not new holdout
quality measurements. The catalog records the later core/replay separately.
Historical cohort counts, reviews and detector scores retain their executed
core identities and are not retroactively changed.

The optional API factory shares this acceptance path, but its local HTTP and
packaging tests are transport evidence only. They add no commercial-model
watermark verification and do not change inaccessible-provider catalog states.


### NLI and lexical-constraint development diagnostics

The pinned English [DeBERTa NLI cross-encoder](https://huggingface.co/cross-encoder/nli-deberta-v3-base)
was executed as a quality critic, separately from watermark detection. Its saved
parameters match the loaded model. Both text directions are evaluated without
truncation; raw logits have independently replayed softmax probabilities. The
catalog's `quality_gate_references` records the frozen model revision, English
qualification, confusion counts, thresholds and artifact hashes. Unsupported
languages receive no passing decision. CPU/MPS numerical probes agree on the
fixed labels and threshold decisions, not bit-for-bit outputs or universal parity.

Combining the critic with the existing factual guard rejects more known bad
English pairs, while also rejecting more known good ones. Long complete outputs
still receive high entailment despite local factual changes. These reused
examined cases are development evidence, not held-out semantic-judge accuracy.

A separate fixed-cohort rewrite experiment uses the pinned original BERT proposal
helper with new eligibility, a larger edit budget, exact whitespace, bidirectional
local-window and whole-paragraph NLI checks, and the unchanged shared library
guard. This changes several policy components; it is not an isolated NLI or
equal-cost comparison. Complete returned texts were reviewed and hash-frozen
before original watermark-detector scores. Observed errors include installation
becoming configuration, exploring improvements becoming implementing them, and
food security becoming garden security. Every after text below its selected
reference threshold has changed meaning. No faithful removal is credited.

A subsequent NLP-only postfilter restricts previously selected proposals using
English POS, morphology, dictionary-backed verb forms and a shared WordNet
synset. It does not regenerate proposals or re-certify the new combinations with
NLI. The examined lexical controls pass and actual gate mutations fail, but
shared lexical senses are not contextual equivalence. A support-contact-points
change to contact details remains accepted. The stricter filter also leaves
unchanged clusters, which the library's default strict policy refuses. Merely
returning an unchanged callback does not enable partial processing. All returned
postfiltered outputs remain marked under their original reference detectors.

Fresh original SynthID and EXP source/control traces reproduce; the original
OpenStamp detector reproduces a fixed source/control pair and independently
replays its staged BF16 rounding. Its zero threshold is illustrative, not a
population-FPR calibration. Fixed counts, policies, quality reviews and score
hashes live in the catalog. Neither diagnostic establishes private-provider
coverage, a production default or market leadership. The optional API transport
remains separate from these entirely local experiments.


### Pegasus sentence benchmark and estimate-status follow-up

The pinned English [Pegasus paraphrasing checkpoint](https://huggingface.co/tuner007/pegasus_paraphrase)
was executed through the library with exact sentence-gap reconstruction,
probability-ranked beams, shared factual guards and bidirectional sentence NLI.
The new policy retains sentences outside the checkpoint's trained positional
window without truncation and refuses unchanged clusters under the default
strict library behavior. Natural termination is checked separately from the
length cap; the actual generation processor stack cannot force EOS.

All serialized parameters match the original checkpoint. Two omitted static
position tables already match the original deterministic Transformers formula;
these are not missing learned weights. The complete ordinary vocabulary and
preselected sentence encodings match the original SentencePiece model. Fixed
CPU/MPS numerical probes agree on labels and greedy IDs within the recorded
logit tolerance, not universal or CUDA parity. The catalog records exact models,
policies, qualifications, controls, counts and artifact hashes.

Complete returned texts were reviewed before after-detector scoring. The
examined equivalent outputs remain detected. The only output with a negative reference-detector
decision changes meaning. Other observed errors include assigning a leader
becoming performed leadership, an optional quantity feature becoming an order
condition, and a few steps becoming a few things. Unresolved source fragments
remain unresolved. No faithful removal or production-default improvement is
credited; this is a reused development cohort with executor review.

The museum output also drops an explicit numerical estimate. A later shared
lexical guard recognizes selected adjacent English `estimated N`, `estimated at N`
and `estimate of N` forms as a distinct estimate kind, preserving the supported
value and unit. The unchanged regression file fails on the executed old core and
passes on the new core; actual alias, guard and unit mutations fail assertions.
Saved callback replay newly rejects that museum output. Prior development-bank
decisions are unchanged, and the added synthetic counterparts are examined
regressions. Nonadjacent descriptions, general uncertainty, same-value participant
attachment and unrelated semantic errors remain unresolved. The original cohort is
not retrospectively reclassified. A separately frozen fresh museum-source run
reproduces the old beam prefix IDs, rejects the estimate-dropping candidate and
retains the original financial sentence. A broader availability claim remains
wrong, and the fresh output remains reference-detected. This targeted regression
adds no faithful-removal credit.

### Native OpenStamp Mistral evaluation

The released Mistral checkpoint differs from the pinned named base only in
`lm_head.weight` across 291 serialized tensors. All loaded base tensors match
the pinned checkpoint exactly. The original author generator and released-head
detector ran in BF16 on MPS using fixed synthetic prefixes. At the illustrative
zero threshold all ten clean controls scored negative and all ten marked
controls scored positive. Captured token probabilities independently reproduce
the original staged BF16 scores.

The strict library path with its default local rewriter refused all ten marked
inputs. Replaying the saved shared-guard calls reproduced every decision.
Thirty-two candidates failed placeholder preservation and four passed as
individual fragments. No full output existed for fidelity review or after
scoring. The catalog binds the exact checkpoint, protocols and artifacts.

One examined source also exposed unnecessary standalone article masks. A
separate rerun made only those entries visible to the model. This reduced fact
placeholders from nineteen to twelve on each attempt. All three candidates
still failed placeholder preservation. This single-source diagnostic did not
change library behavior.

### Independent AMR quality diagnostic

The English AMR diagnostic uses amrlib's published BART parser and compares
directed, labeled graphs with literal values and an explicit root. The
comparison ignores variable names and relation order. Mutations that discard
roles, concepts or roots fail the original controls.

All inputs in the English development subset yielded qualified graphs. On
this known set, exact identity rejects every labeled meaning change and
rejects most faithful paraphrases too. The catalog records the complete
confusion tables and comparisons on exact pairs shared with the earlier NLI
study. Exact identity was too restrictive for a mandatory rewrite check.
The diagnostic is separate from SWAN and contributes no watermark result.

### Paired local fact transport

The fixed SynthID and OpenStamp Mistral development cohorts each contributed
ten marked documents. Each document used the same pinned native Qwen2.5-1.5B
runtime with masked and visible facts. The two arms shared the prompt,
segmentation, locked-span handling, retry policy and restored-text guards.

All forty scheduled document attempts finished. The catalog separates document
returns, candidate rejection reasons and generation calls. Replay reproduced
all 157 observed candidate decisions. The visible-fact arm removed factual
placeholder mismatches but encountered factual-guard refusals on both cohorts.
One complete OpenStamp result reached fidelity review. It reversed the intended
users of the paper visitor log and added an unsupported statement about the
information being outdated. Meaning review was frozen before after scoring.
No full result qualified for after scoring or faithful-removal credit.
