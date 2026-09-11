# Coverage and cadence

Use this reference to choose the audit modes and decide whether current-only or
historical data is sufficient.

## Coverage matrix

| Surface | Discovery evidence | Regression check | Removal proof |
| --- | --- | --- | --- |
| C2PA, EXIF, XMP, IPTC, PNG text, TC260, soft bindings, and container metadata | `scripts/corpus_gap_scan.py` plus review of unattributed and partially attributed reports | `scripts/sidecar_regression.py`; compare record-based and file-based identification when collection changes | `scripts/metadata_removal_audit.py`; signal absent after strip and decoded pixels/streams unchanged |
| Registered visible image marks | `scripts/visible_positives.py`; metadata-derived vendor cohorts and blinded sheets for uncovered labels | `scripts/visible_eval.py`, registered examples, and full-corpus crossfire | `scripts/visible_removal_audit.py`; same-mark redetection plus visual residual review and fill-quality checks |
| Candidate China-AIGC visible marks | `scripts/vendor_cohort_harvest.py` partitions by producer identity without using pixels; inspect full-width top and bottom sheets | `scripts/vendor_mark_calibrate.py` against independently labelled positives, clean negatives, and neighbouring marks | Register only after the candidate gate survives full-corpus crossfire; then use the normal visible-removal audit |
| Registered visible video marks | Review provider-labelled or controlled complete clips, not isolated frames alone | Full-clip tests and the local real-provider audit described in `docs/verification-plan.md` | Preserve sequence, timing, duration, and audio; second-pass detection must be quiet |
| Locally decodable invisible image marks, including open DWT-DCT and TrustMark | Positive-control-gated corpus or constructed carriers; a negative on an uncontrolled carrier is inconclusive | `scripts/watermark_benchmark.py` and the relevant decoder tests with the required extras installed | Re-run the same decoder on output and retain a positive control in the same run |
| Proprietary image marks: Google SynthID, Microsoft InvisMark, Meta Content Seal | Provider provenance can select a cohort but is not a local pixel decode; use bounded provider-labelled candidates | Hash-bound provider-oracle batches when a new route, profile, or claimed signal is evaluated | Source-positive and output-negative verdicts from that same provider; record C2PA separately from the pixel watermark |
| Invisible video and audio marks | Matched provider or open-model cohorts with codec and degradation controls | Benchmark-kernel audio/video rows and temporal aggregation checks | Preserve the non-target stream; use the matching oracle and a control-positive run |
| Explicit vendor regeneration with no detectable local signal | This is a user assertion or research cohort, not a discovered signal | Verify routing and profile selection separately from detection | Quality and provider-oracle evidence may certify a cohort, but must not create a local detection claim |

`classify` is a pixel-origin guess, not provenance, and must not be counted as a
new signal. `erase --region` is user-directed editing, not automatic watermark
detection, and is outside this audit.

## Gaps the legacy scanner cannot settle alone

`corpus_gap_scan.py` is a metadata serialization probe. It reports raw markers,
input shapes, identify errors, integrity clashes, unattributed results, and
blind-marker candidates, but it still cannot settle other useful classes, so
inspect them separately:

- a known generic signal with missing or wrong vendor attribution;
- a newly coexisting marker on a file that already has another AI verdict;
- an entirely novel marker absent from the scanner's hand-written list;
- visible image or video labels;
- proprietary invisible marks without a local decoder;
- detection/removal parity failures;
- decoder crashes, unsupported extensions, and incomplete reads;
- drift between stored identify sidecars and the current library;
- crossfire where a new detector fires on an older neighbouring cohort.

Marker matching must remain metadata-region bounded to avoid random hits in
compressed pixels, but its inventory should be reconciled with the current
metadata constants and registry on every audit. Do not infer completeness from
`No gap candidates` until the other rows of the matrix also ran.

## When to replay old data

Yes, old data must be rerun. Current code can learn to parse an old
serialization, and a new detector can crossfire on an old provider cohort even
when no new object demonstrates either problem. The canonical intervals,
change triggers, and sampling rules live in
[`docs/verification-plan.md#signal-discovery-cadence`](../../../../docs/verification-plan.md#signal-discovery-cadence).

Keep dated reports immutable enough to compare runs. Record the library SHA and
version, corpus minimum and maximum dates, input counts per media/format shape,
enabled extras, command line, and output paths. Never commit operational corpus
counts or retained identifiers; summarize only durable conclusions in the
canonical research document.
