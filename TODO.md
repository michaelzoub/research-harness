# TODOs For Phases 3-7

## Phase 3: Persistent World Model

- Add SQLite or Postgres backend behind `ArtifactStore`.
- Add cross-run source, claim, and hypothesis deduplication.
- Add provenance edges between claims, hypotheses, contradictions, and reports.
- Add migration scripts for schema evolution.

## Phase 4: Observability

- Track prompt versions and harness config snapshots per run.
- Add cost accounting for real model/tool calls.
- Add structured failure taxonomies.
- Add a run viewer or notebook export for traces and artifacts.

## Phase 5: Harness Debugger

- Expand failure localization from simple heuristics to component-level
  diagnosis.
- Compare trace patterns across runs.
- Score harness-change proposals against risk and expected value.

## Phase 6: Gated Adaptive Harness Evolution

- Save proposed changes as pending config variants.
- Run benchmark comparisons before accepting changes.
- Add keep/reject/ask-human decision records.
- Prevent code mutation; allow only reviewed config or branch variants.

## Phase 7: Long-Running Autonomous Research

- Add multi-cycle research loops.
- Implement entropy injection and periodic self-review.
- Add plateau detection using source, claim, hypothesis, and critic deltas.
- Add human checkpoints and continuation across sessions.
