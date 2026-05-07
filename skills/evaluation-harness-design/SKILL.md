---
name: evaluation-harness-design
description: Design or modify the evaluation harness for research, optimize, and challenge product agents. Use when adding EvalTask, EvalSuite, graders, trial isolation, trajectory metrics, aggregation, benchmark artifacts, or tests that judge agent-harness behavior.
---

# Evaluation Harness Design

Use this skill when working on `research_harness/evals.py`, benchmark outputs,
grader logic, or tests that measure agent behavior.

## Evaluation Shape

Follow this nesting:

```text
EvaluationHarness
  EvalSuite
    EvalTask
      EvalTrial
        production Agent Harness run
        trajectory + outcome
      GraderResult
    aggregate score
```

The eval harness should run the same `Orchestrator` path users run. Avoid mock
execution paths unless a test explicitly isolates a small pure function.

## Task Contract

Every `EvalTask` should specify:

- prompt/input
- expected product agent or task mode
- evaluator name when applicable
- success criteria
- grader IDs
- aggregation mode and threshold

The core suite should cover all product options:

- research: evidence, claims, report, trajectory
- optimize: evaluator score, optimal code artifact, trajectory
- challenge: seed context, proxy/official score, optimal code, solution shape

## Trial Isolation

Each trial gets a clean output root and temp directory. Preserve these checks:

- no stale artifacts can inflate score
- `TMPDIR` is per-trial
- artifacts stay under the trial root

## Grader Rules

Graders score trajectory plus outcome. Prefer deterministic code graders for
artifact contracts, routing, and exact metrics. Use model-style rubrics only for
report quality or qualitative judgment.

Optimization/challenge graders must check:

- `optimization_result.json`
- `optimized_candidate.txt`
- `optimal_code.py`
- `optimal_code_path` in the result payload

## Metrics To Track

Keep metrics explicit and machine-readable:

- status, pass/fail, aggregate score
- task mode and product agent
- best score and objective direction
- source/claim counts for research
- variant/evaluation counts for optimization
- trace/progress completeness
- latency/tokens/tool calls when available

