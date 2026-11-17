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
that each requires per-token `entropies`. These entries remain unexecuted in the
inventory. The author's entropy-weighted condition can return no weights when
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
requires license acceptance before access. This leaves those exact weighted
states unverified; a different model or an unweighted detector is not substituted.

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
SynthID baseline executions are recorded below; TextSeal still needs its entropy
pipeline before execution can qualify.

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
