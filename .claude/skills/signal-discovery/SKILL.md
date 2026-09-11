---
name: signal-discovery
description: Discover missing AI provenance and watermark coverage in the remove-ai-watermarks library from fresh or historical corpora, then validate detection and removal parity. Use for corpus gap mining, periodic signal audits, new-provider discovery, or deciding whether old retained data must be replayed. Do not use for ordinary per-file identification or removal.
---

# Signal discovery

Work from the `remove-ai-watermarks` repository root. This skill discovers and
closes library coverage gaps. Implementation, tests, fixtures, and
documentation changes belong in this library; do not patch a downstream
consumer to compensate for a detector or cleaner gap.

Treat retained inputs as sensitive and read-only. Keep reports, contact sheets,
sidecars, and derived samples under the library's gitignored `.local-eval/`.
Never commit a retained input or an artifact derived from one.
Use only publication-cleared or synthetic files for tracked fixtures.

Before acting, read:

- `CLAUDE.md` and `.claude/rules/development.md`;
- `docs/verification-plan.md`;
- `docs/supported-signals.md` and `docs/watermarking-landscape.md`;
- [references/coverage-and-cadence.md](references/coverage-and-cadence.md).

## Choose the run

Do not call `corpus_gap_scan.py` a complete signal audit. Select every mode the
question requires:

1. **Fresh discovery** finds new providers, serializations, labels, and formats
   in objects added since the preceding review.
2. **Historical replay** reruns current code on the complete retained history.
   Use it after detection-path or format-support changes and on the periodic
   backstop defined in the reference.
3. **Removal parity** proves that every detected removable signal is cleared
   without violating the signal family's fidelity invariant.
4. **Calibration or provider oracle** is required where the corpus has no
   self-evident answer key. Never treat a local negative as proof that a
   proprietary watermark is absent.

Start by inventorying the current code, not a remembered list. Enumerate image
and video visible-mark registries, metadata readers and strippers, local
invisible decoders, explicit invisible vendor routes, supported containers, and
the stable signal families emitted by `identify`. Reconcile that inventory with
the coverage matrix before interpreting any green result. A hand-written marker
list or report family map that omits a current registry member is itself a
finding.

## Run protocol

1. Record the library commit, version, enabled extras, corpus date range, and
   exact commands in an untracked run directory. Preserve the pre-run Git
   status so unrelated work is not attributed to the audit.
2. Refresh the local retained corpus through its owning, external collection
   process when the task calls for fresh data. Use an explicit gitignored
   destination, do not delete local objects, and finish with a zero-delta pass.
   Fetch paired outputs only for a removal or complaint investigation.
3. Run the applicable fresh and historical discovery modes from the reference.
   Use `corpus_gap_scan.py --since YYYY-MM-DD` for the overlapping fresh window
   and omit `--since` for the complete history. Report counts per input shape
   and per outcome; an example does not establish coverage.
4. Triage every candidate into one of: parser/serialization gap, missing vendor
   attribution, new visible mark, possible invisible signal, format/read error,
   expected non-AI provenance, or false candidate. Preserve unresolved items as
   unresolved.
5. Reproduce an actionable gap with a failing test before changing code. Derive
   the smallest publication-safe fixture from a controlled source or construct
   it synthetically; do not move the retained file into a tracked path.
6. Implement the fix in the library subsystem that owns it. Update the library
   registry, parser, remover, public docs, and published agent skill when their
   contracts move. Updating a downstream consumer to take a released version is
   a later, separately scoped task.
7. Rerun the affected focused audit and the complete historical regression
   mode. A detection change must name the files expected to move before the
   after-run; classify lost signals separately from intended new detections.
8. Run removal parity for every newly detected removable signal. A quiet
   post-removal detector is necessary but not always sufficient: check
   pixel/stream invariants and visually inspect residual-prone visible marks.
9. Run `bash maintain.sh`. Report evidence limits, incomplete rows, crashes,
   unsupported formats, and oracle-unavailable cohorts explicitly.

External oracle uploads are separate mutations. Use the `provider-oracles`
skill only for a bounded, already-selected candidate, and obtain the required
action-time upload approval. Do not adapt candidates against an oracle, retry
automatically, or substitute one provider's verdict for another's.
