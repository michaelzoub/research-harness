# research-harness

A compact research and optimization harness for running agent loops, preserving run artifacts, and experimenting with evaluator-driven optimization tasks.

The default mode is the nested evolutionary loop:

```text
prompt -> task ingestion -> outer orchestrator -> inner evaluator(s) -> ranked variants -> critique/synthesis -> run artifacts
```

Conceptually, each product option is an agent: model + harness. The harness is
the loop, tools/evaluators, state store, budgets, traces, and stopping policy.
The three product agents are `research`, `optimize`, and `challenge`; `optimize`
and `challenge` share the same optimization core, while challenge runs add
challenge-specific specs, solution rendering, and official-result accounting.

## Quick Start

Clone or download the repo, then run the repo-local wrapper:

```bash
git clone <repo-url> research-harness
cd research-harness
./autore "Research adaptive agent harnesses for automated scientific discovery"
```

If the wrapper is not executable:

```bash
chmod +x autore
```

Optional install, if you want `autore` available without `./`:

```bash
python3 -m pip install -e .
autore "Research adaptive agent harnesses"
```

## Commands And Options

| Use case | Command / option | What it does |
| --- | --- | --- |
| Default loop | `./autore "<prompt>"` | Runs the nested evolutionary agent loop. Auto-routes to `research`, `optimize`, or `optimize_query`. |
| General research | `--task-mode research` | Forces evidence/article/query research. |
| Direct optimization | `--task-mode optimize --evaluator <name>` | Skips research exploration and scores code/strategy variants with a deterministic evaluator. |
| Optimization query exploration | `--task-mode optimize_query` | Researches approaches first, writes `optimizer_seed_context.json`, then optionally seeds the optimizer. |
| Add deterministic evaluator | `--evaluator length_score` or `--evaluator prediction_market` | Selects the scoring function for optimize/optimize-query runs. |
| Local/offline retrieval | `--retriever local` | Uses `examples/corpus/research_corpus.json`; best for tests and reproducible demos. |
| Live/source mix retrieval | `--retriever auto` | Uses the configured mixed source strategy. |
| Legacy fan-out flow | `--mode standard` | Runs the older standard research fan-out/fan-in path. |
| Deterministic phase-1 flow | `--mode deterministic` | Runs the earliest deterministic research flow. |
| LLM auto-detect | `--llm-provider auto` | Uses OpenAI if `OPENAI_API_KEY` exists in `.env.local`; otherwise local fallback. |
| Force local fallback | `--llm-provider local` | Runs without paid/live LLM calls. |
| Force OpenAI | `--llm-provider openai` | Requires a valid OpenAI key in `.env.local`. |
| Iteration budget | `--max-iterations N` | Sets outer-loop iteration count. |
| Run eval suite | `python3 -m research_harness.evals --suite core` | Runs prewritten eval tasks for research, optimize, optimize-query, and challenge modes. |

## Common Examples

General research:

```bash
./autore "Research agent paradigms and workplace trends" --task-mode research
```

Optimize-query with the smoke-test evaluator:

```bash
./autore "Research approaches for improving a tiny scoring function before optimizing it" --task-mode optimize_query --evaluator length_score --retriever local
```

Bundled prediction-market challenge exploration:

```bash
./autore "Get to \$10 profit in the prediction market challenge, don't stop until you're profitable. Make sure to introduce entropy (AMM, PM, options, etc) literature if you start tweaking hyperparameters." --task-mode optimize_query --evaluator prediction_market --retriever local
```

## Output Artifacts

Each run creates `outputs/run_<slug>/` and prints the important paths:

| Artifact | Purpose |
| --- | --- |
| `prd.json` | Organized task map for the run, including task status and dependencies. |
| `progress.txt` | Terminal-style step-by-step progress log. |
| `variants.json` | Query/code/strategy variants proposed by the loop. |
| `variant_evaluations.json` | Scores and metrics for each variant. |
| `optimizer_seed_context.json` | Top query findings used to seed the optimizer in `optimize_query` mode. |
| `optimization_result.json` | Standard optimization summary: evaluator, objective variable, maximize/minimize direction, score, metrics, candidate path, and official-result status. |
| `optimized_candidate.txt` | Best generated candidate/payload that was scored by the optimizer. |
| `optimal_code.py` | Universal optimization/challenge code artifact for the exact best candidate selected by the agent. |
| `solution.py` | Best runnable solution sketch when the challenge can render one, currently prediction-market only. |
| `final_report.md` | Final synthesis report. |
| `run_benchmark.html` | Per-run visual benchmark and decision DAG. |
| `run_benchmark_summary.json` | Machine-readable benchmark summary. |
| `decision_dag.svg` | Flowchart of routing, evaluation, selection, and synthesis. |

For “where is the solution?”, check:

```text
outputs/run_<slug>/optimization_result.json
outputs/run_<slug>/optimized_candidate.txt
outputs/run_<slug>/optimal_code.py
outputs/run_<slug>/solution.py
```

`optimal_code.py` is required for every optimization or challenge run that
executes an evaluator. Challenge-specific files such as `solution.py` are
additive and should never replace the universal optimal-code artifact. If
`solution.py` does not exist, inspect `optimal_code.py`, `optimized_candidate.txt`,
`variants.json`, and `variant_evaluations.json`.

For the full component map and evaluation-harness diagram, see
[`docs/architecture.md`](docs/architecture.md).
For long-running optimization and challenge runs, see
[`docs/long_running_tasks.md`](docs/long_running_tasks.md).

## Repo Skills

Reusable project conventions live under `skills/` using the Agent Skills
format. The current set covers architecture, evaluation design, challenge
adapters, grounded research runs, skill authoring, and the universal
optimization output contract. Skill frontmatter is validated by the smoke test
suite.

## Evaluation Suites

Run the built-in eval suite:

```bash
python3 -m research_harness.evals --suite core --trials 1
```

Run edge-case regression evals:

```bash
python3 -m research_harness.evals --suite edge --trials 1
```

Run everything:

```bash
python3 -m research_harness.evals --suite all --trials 1
```

If installed with `python3 -m pip install -e .`, the console command is:

```bash
autore-eval --suite core --trials 1
```

The core suite includes one prewritten task for each run type:

| Eval task | Run type | Main graders |
| --- | --- | --- |
| `research_open_ended` | Open-ended research | Outcome completion, grounded claims, report artifact, transcript progress, deterministic model-style report rubric. |
| `optimize_direct` | Direct optimization | Mode routing, deterministic optimize score, transcript progress. |
| `optimize_query_seeded` | Optimization query | Seed context, query phase, optimizer phase, transcript progress. |
| `challenge_prediction_market` | Challenge optimization | Prediction-market solution file, local proxy score, seed context, transcript progress. |

The edge suite adds regression tasks for behaviors that are easy to get wrong:

| Eval task | Edge case |
| --- | --- |
| `optimize_query_missing_evaluator_skips_optimizer` | Missing evaluator must skip optimization cleanly instead of fabricating scores or code artifacts. |
| `prediction_market_outputs_are_contained` | Generated strategies must stay under `outputs/<run>/candidates/`, never repository source files. |
| `prediction_market_unmeasured_official_status` | Local fallback scores must be marked unmeasured and must not masquerade as upstream profit. |
| `challenge_prediction_market_official_unavailable_records_unmeasured` | If upstream scoring is unavailable or not requested, `official_result.measured` must stay false. |
| `challenge_prediction_market_candidate_files_only_in_outputs` | Candidate strategy files must be created only under the run output tree. |
| `parallel_trials_do_not_share_tmp_or_outputs` | Multiple trials for one task must have distinct trial, output, and temp directories. |
| `challenge_prediction_market_no_repo_root_strategy_files` | Prediction-market runs must not leak `pm_strategy*.py` or `tmp_pm*.py` files into the repository root. |
| `research_should_not_oversearch` | Simple bounded research prompts must stay inside a small search/claim/evolution budget. |

The trajectory checks are inspired by
[`langchain-ai/agentevals`](https://github.com/langchain-ai/agentevals): this
repo currently uses native loop artifacts rather than adopting the dependency
directly, because our trajectories are stored as `variant_evaluations.json`,
`evolution_rounds.json`, progress logs, and outcome files instead of OpenAI-style
message arrays.

Terminology used by the eval runner:

| Term | Meaning in this project |
| --- | --- |
| Task | A single eval problem with prompt, mode, success criteria, and graders. |
| Trial | One attempt at a task. Multiple trials can be run with `--trials N`. |
| Grader | Code, model-style, or human-review logic that scores one aspect of a trial. |
| Transcript | The recorded trace/progress artifacts for the trial. |
| Outcome | Final run state such as status, best score, report existence, and solution existence. |
| Harness | The infrastructure that runs tasks, records artifacts, grades, and aggregates. |
| Suite | A collection of tasks measuring related capabilities. |

Aggregation can be `binary`, `weighted`, or `hybrid`. The local suite mostly uses cheap deterministic code graders, with a deterministic model-style rubric placeholder for open-ended report quality and a human-grader placeholder for future calibration.

Eval isolation policy:

- Each trial uses the same production `Orchestrator` path as normal runs.
- Each trial gets its own clean output directory under `eval_outputs/runs/<task>_trial_<n>/`.
- Existing trial directories are removed before rerun so stale artifacts cannot inflate performance.
- `TMPDIR` is set to a per-trial temp folder while the trial runs.
- Shared local corpus files are read-only inputs; outputs, transcripts, solutions, and summaries are per-trial.

## Prediction-Market Challenge

The first bundled challenge lives in:

```text
challenges/prediction_market/
```

It is modeled on [danrobinson/prediction-market-challenge](https://github.com/danrobinson/prediction-market-challenge/tree/main), whose public API expects a Python `Strategy(BaseStrategy)` with an `on_step(state)` method and whose local CLI is `uv run orderbook-pm run <strategy.py>`.

Important scoring caveat:

- Generated strategy files are run artifacts and live under `outputs/<run>/candidates/`, with the selected winner copied to `outputs/<run>/optimal_code.py`.
- By default, `--evaluator prediction_market` uses a local challenge-semantics fallback so the harness can loop quickly without Python/`uv` setup becoming the bottleneck.
- Set `PREDICTION_MARKET_USE_UPSTREAM=1` to score generated candidates through the upstream challenge CLI. The real score/profit is the upstream challenge score from `danrobinson/prediction-market-challenge`, specifically its `orderbook_pm_challenge.runner.run_batch` / CLI path.
- The generated `optimal_code.py` and `solution.py` target the upstream `BaseStrategy` API so they can be run against the real challenge package.
- `optimization_result.json` records `official_result.measured`, `score_source`, `profit_usd`, and the winning `candidate_path`.
- Treat fallback scores as internal search signals; treat upstream measured scores as the final leaderboard/profit signal.

Useful upstream scoring variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PREDICTION_MARKET_USE_UPSTREAM` | empty | Set to `1` to run the upstream `orderbook-pm` scorer. |
| `PREDICTION_MARKET_CHALLENGE_PATH` | `/private/tmp/prediction-market-challenge-src` | Local checkout of the upstream repo. |
| `PREDICTION_MARKET_SIMULATIONS` | `40` | Number of upstream simulations per candidate during harness search. |
| `PREDICTION_MARKET_STEPS` | `600` | Number of upstream steps per simulation during harness search. |
| `PREDICTION_MARKET_WORKERS` | `4` | Parallel upstream simulation workers. |
| `PREDICTION_MARKET_TIMEOUT_SECONDS` | `120` | Timeout for one candidate scoring call. |

Current challenge files:

| File | Purpose |
| --- | --- |
| `challenges/prediction_market/spec.md` | Agent-readable challenge summary. |
| `challenges/prediction_market/interface.py` | Minimal interface sketch. |
| `challenges/prediction_market/evaluator.py` | Local proxy evaluator used by this harness. |
| `challenges/prediction_market/fixtures/` | Sanity examples for scoring. |

## Environment

The harness loads `.env`, then `.env.local`. Put secrets in `.env.local`:

```bash
cat > .env.local <<'EOF'
OPENAI_API_KEY=sk-...
RESEARCH_HARNESS_LLM_PROVIDER=auto
RESEARCH_HARNESS_LLM_MODEL=gpt-5.2
EOF
```

Useful variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | empty | Enables live OpenAI calls when present. |
| `RESEARCH_HARNESS_LLM_PROVIDER` | `auto` | `auto`, `openai`, or `local`. |
| `RESEARCH_HARNESS_LLM_MODEL` | `gpt-5.2` | Model used by live LLM calls. |
| `RESEARCH_HARNESS_RETRIEVER` | `auto` | Default retriever/source strategy. |
| `RESEARCH_HARNESS_CORPUS_PATH` | `examples/corpus/research_corpus.json` | Local corpus path. |
| `RESEARCH_HARNESS_OUTPUT_DIR` | `outputs` | Run artifact directory. |
| `RESEARCH_HARNESS_TASK_MODE` | `auto` | Default task mode. |
| `RESEARCH_HARNESS_EVALUATOR` | empty | Default evaluator name. |

## Development

Run tests:

```bash
python3 -m unittest tests.test_smoke
```

Compile-check Python files:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/research-harness-pycache python3 -m py_compile research_harness/*.py challenges/prediction_market/*.py
```

Generate benchmark dashboard from existing runs:

```bash
./autore-bench
```

## Current Limitations

- Parallel evaluators currently score proposed text/code sketches; they do not yet execute full submitted programs in a sandbox.
- Prediction-market `solution.py` is generated, but official profitability must be measured with the upstream challenge evaluator.
- The “parallel agents” are still lightweight loop workers; useful diversity depends heavily on live LLM proposal quality and retriever quality.
- `--retriever local` is deterministic and good for tests, but it will not bring in broad AMM/options/market-making literature unless that corpus contains it.

## Roadmap

Near-term useful upgrades:

1. Run generated `solution.py` directly through the upstream prediction-market package when installed.
2. Store official upstream score/profit beside the harness proxy score.
3. Make evaluator workers produce richer independent variants instead of near-duplicate seeded mutations.
4. Add a small challenge-local sandbox for executable strategy submissions.
