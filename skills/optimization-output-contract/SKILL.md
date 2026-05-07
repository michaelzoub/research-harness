---
name: optimization-output-contract
description: Use when adding or modifying optimization or challenge agents, evaluators, graders, artifacts, or run outputs in this repo. Enforces that every optimization/challenge run emits the exact best code selected by the agent, never only a score or prose summary.
---

# Optimization Output Contract

When working on optimization or challenge behavior in `research-harness`, treat
the selected candidate code as a required artifact.

## Required Contract

Every run that executes an optimization evaluator must emit all of:

- `optimized_candidate.txt`: exact best candidate payload selected by score.
- `optimal_code.py`: code artifact corresponding to the best candidate.
- `optimization_result.json`: machine-readable result with `optimal_code_path`.

Challenge adapters may also emit challenge-specific files such as `solution.py`,
but those are additive. Never make `solution.py` a one-off substitute for the
universal `optimal_code.py` contract.

## Implementation Rules

- Write `optimal_code.py` from the same best variant used for
  `optimization_result.json`.
- Include `optimal_code_path` in `optimization_result.json`.
- Add or update graders so evals fail when `optimal_code.py` is missing.
- For domain-specific challenge adapters, render evaluator-ready code in
  `optimal_code.py`; optionally mirror it to `solution.py` when the upstream
  challenge expects that filename.
- For generic/non-code optimization payloads, emit a Python module that preserves
  the exact selected payload and exposes a deterministic accessor.

## Prediction-Market Challenge

Model this challenge on:

`https://github.com/danrobinson/prediction-market-challenge/tree/main`

The upstream public API expects:

- `from orderbook_pm_challenge.strategy import BaseStrategy`
- `from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState`
- `class Strategy(BaseStrategy)`
- `def on_step(self, state: StepState)`

The official local command shape is:

```bash
uv run orderbook-pm run path/to/strategy.py --simulations 5 --steps 100
uv run orderbook-pm run path/to/strategy.py --workers 4
uv run orderbook-pm run path/to/strategy.py --sandbox
```

The official score is mean edge from `orderbook_pm_challenge.runner.run_batch`.
Local proxy scores are search signals only.

