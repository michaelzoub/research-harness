# research-harness

A compact research and optimization harness for running agent loops, preserving run artifacts, and experimenting with evaluator-driven optimization tasks.

```text
prompt → task ingestion → outer orchestrator → inner evaluator(s) → ranked variants → critique/synthesis → artifacts
```

The three product agents are `research`, `optimize`, and `challenge`. `optimize` and `challenge` share the same optimization core; challenge runs add challenge-specific specs, solution rendering, and official-result accounting.

## Quick Start

```bash
git clone <repo-url> research-harness && cd research-harness
./autore
```

Running `./autore` with no arguments opens a selection-based setup: enter the
goal, pick the run type, choose an evaluator/source when needed, and start the
run without memorizing flags. Use Up/Down to move through choices and Enter to
select.

You can still pass a prompt directly for scripted use:

```bash
./autore "Research adaptive agent harnesses for automated scientific discovery"
```

Optional install so `autore` is on `PATH`:

```bash
python3 -m pip install -e .
```

## Selection-Based Setup

```bash
./autore
```

The interactive setup asks for:

| Choice | Options |
| --- | --- |
| Run type | Auto decide, research, optimize, or research-then-optimize. |
| Evaluator | Prompt-derived, `length_score`, `prediction_market`, or a custom name. |
| Evidence source | Auto mix, local corpus, arXiv, OpenAlex, GitHub, web, docs/blogs, Twitter/X, or memory. |
| Budget | Iteration count. |

To start from existing flags but finish by selection, use:

```bash
./autore --interactive --retriever local
```

## Scriptable Options

| Flag / option | What it does |
| --- | --- |
| `--task-mode research` | Force evidence/article/query research. |
| `--task-mode optimize --evaluator <name>` | Score code/strategy variants with a deterministic evaluator. |
| `--task-mode optimize_query` | Research approaches first, write seed context, then seed the optimizer. |
| `--evaluator length_score \| prediction_market` | Select scoring function for optimize/optimize-query runs. |
| `--retriever local \| auto` | `local` uses the bundled corpus; `auto` mixes live sources. |
| `--llm-provider auto \| openai \| local` | `auto` uses OpenAI if `OPENAI_API_KEY` is set, otherwise falls back. |
| `--max-iterations N` | Set outer-loop iteration budget. |
| `--no-sessions` | Skip session JSONL logging for the run. |

## Examples

General research:

```bash
./autore "Research agent paradigms and workplace trends" --task-mode research
```

Optimize-query with the smoke-test evaluator:

```bash
./autore "Research approaches for improving a tiny scoring function" \
  --task-mode optimize_query --evaluator length_score --retriever local
```

Prediction-market challenge (`challenges/prediction_market/`, modeled on [danrobinson/prediction-market-challenge](https://github.com/danrobinson/prediction-market-challenge)):

```bash
./autore "Get to \$10 profit in the prediction market challenge, don't stop until profitable. \
Introduce entropy from AMM, prediction-market, and options literature before tweaking hyperparameters." \
  --task-mode optimize_query --evaluator prediction_market
```

Set `PREDICTION_MARKET_USE_UPSTREAM=1` to score through the real upstream CLI instead of the local proxy.

## Output Artifacts

Each run creates `outputs/<NNN>_run_<slug>/`:

| Artifact | Purpose |
| --- | --- |
| `prd.json` | Organized task map with status and dependencies. |
| `progress.txt` | Step-by-step progress log. |
| `variants.json` / `variant_evaluations.json` | Proposed variants and their scores. |
| `optimizer_seed_context.json` | Top query findings used to seed the optimizer. |
| `optimization_result.json` | Best score, candidate path, and official-result status. |
| `optimal_code.py` | Universal code artifact for the best selected candidate. |
| `solution.py` | Challenge-specific runnable solution (prediction-market only). |
| `final_report.md` | Final synthesis report. |
| `run_benchmark.html` | Per-run visual benchmark with Gantt timeline and decision DAG. |
| `run_notebook.ipynb` | Notebook export with trace summaries, artifact counts, cost, and diagnosis data. |
| `harness_diagnosis.json` | Component-level failure taxonomy, trace-pattern comparison, and debugger localization. |
| `provenance_edges.json` | Claim/source/hypothesis/contradiction/report provenance graph. |
| `cost.json` / `cost_events.json` | Per-run model usage totals and per-call accounting for live model calls. |
| `world_model.sqlite` | Cross-run SQLite mirror in the output root for dedupe, provenance, and observability queries. |
| `decision_dag.png` / `agent_timeline.png` | Flow and timing visuals. |

## Evaluation

```bash
python3 -m research_harness.evals --suite core   # core tasks
python3 -m research_harness.evals --suite edge   # edge/regression tasks
python3 -m research_harness.evals --suite all    # everything
```

Each trial runs the production `Orchestrator` path and writes artifacts to `eval_outputs/<task_id>/trial_<N>/<run_slug>/`, mirroring the `outputs/` layout. The core suite covers `research_open_ended`, `optimize_direct`, `optimize_query_seeded`, and `challenge_prediction_market`. The edge suite adds regression tasks for artifact containment, missing evaluators, fabricated sources, plateau recovery, and trajectory matching.

## Environment

Put secrets in `.env.local`:

```bash
OPENAI_API_KEY=sk-...
RESEARCH_HARNESS_LLM_PROVIDER=auto
RESEARCH_HARNESS_LLM_MODEL=gpt-5.2
```

## Development

```bash
python3 -m unittest tests.test_smoke
python3 -m research_harness.evals --suite core --trials 1
./autore-bench   # regenerate benchmark dashboard from existing runs
```

For architecture details and long-running task guidance see [`docs/`](docs/).
