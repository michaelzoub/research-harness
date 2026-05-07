---
name: challenge-adapter
description: Add or update optimization challenge adapters. Use when modeling an upstream challenge repo, writing a proxy evaluator, rendering optimal_code.py or solution.py, integrating official runners, or adding challenge-specific graders and docs.
---

# Challenge Adapter

Use this skill for files under `challenges/` and for challenge-specific paths in
the optimizer, eval harness, or docs.

## Adapter Contract

A challenge adapter should provide:

- challenge spec with upstream source-of-truth link
- local proxy evaluator for cheap search
- official runner contract when available
- solution renderer that emits evaluator-ready code
- eval graders for proxy score, official score status, and artifact shape

Every challenge run must also satisfy the universal optimization contract:

- `optimized_candidate.txt`
- `optimal_code.py`
- `optimization_result.json` with `optimal_code_path`

Challenge-specific files such as `solution.py` are additive.

## Upstream Modeling

Before changing a challenge adapter:

1. Inspect the upstream README/docs/API.
2. Identify package names, class/function interfaces, CLI commands, scoring
   fields, parallel execution options, and failure rules.
3. Encode those facts in the local `spec.md`.
4. Make generated code target the real upstream API.
5. Record whether official scoring was actually measured.

## Prediction-Market Notes

Source of truth:

```text
https://github.com/danrobinson/prediction-market-challenge/tree/main
```

Generated code should target:

```python
from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    def on_step(self, state: StepState):
        ...
```

Official command shape:

```bash
uv run orderbook-pm run path/to/strategy.py --workers 4
```

Proxy score is only a search signal. Official mean edge must come from the
upstream `orderbook_pm_challenge.runner.run_batch` or CLI path.

## Gotchas

- Never claim official profit unless the official runner actually executed.
- Do not let challenge-specific `solution.py` replace `optimal_code.py`.
- Keep proxy evaluators deterministic so eval trials are reproducible.

