# research-harness

A research-first multi-agent harness for automated research and occasional optimization.

`research-harness` is designed to help researchers, builders, and technical teams run structured agentic research workflows: search broadly, explore multiple hypotheses in parallel, critique findings, synthesize evidence, and preserve useful artifacts for future runs.

The project starts with a deterministic research agent and a fan-out / fan-in parallel research harness. Over time, it is intended to evolve into an adaptive research harness that can inspect its own failures and propose constrained improvements to the research process.

The current MVP is intentionally simple and reproducible. By default, actual runs use a mixed live source strategy across papers, preprints, web/docs/blogs, GitHub, social web search, and prior artifact memory. An offline local JSON corpus is available for demo and test runs with `--retriever local`.

---

## Quick Start

### 1. Get The Project

Clone the repository, or download and unzip it, then enter the project folder:

```bash
git clone <repo-url> research-harness
cd research-harness
```

If you downloaded a ZIP instead, open the extracted folder in your terminal.

### 2. Run With The Repo-Local Bash Wrapper

The fastest way to start is the included `autore` wrapper script:

```bash
./autore "Research adaptive agent harnesses for automated scientific discovery"
```

If your shell says the script is not executable, run:

```bash
chmod +x autore
```

Then retry:

```bash
./autore "Research adaptive agent harnesses for automated scientific discovery"
```

### 3. Optional: Install `autore` As A Command

If you want to run `autore` without `./`, install the package locally:

```bash
python3 -m pip install -e .
```

Then run:

```bash
autore "Research adaptive agent harnesses for automated scientific discovery"
```

On macOS system Python, `pip install --user` may place scripts in `~/Library/Python/3.9/bin`. If `autore` installs but your shell says `command not found`, add that directory to your PATH:

```bash
echo 'export PATH="$HOME/Library/Python/3.9/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### 4. Try The Included Example

```bash
python3 run_example.py
```

New run folders are named from the research goal:

```text
outputs/run_adaptive-agent-harnesses-automated-scientific-discovery/
outputs/run_adaptive-agent-harnesses-automated-scientific-discovery-02/
```

### Common Runs

Run against arXiv:

```bash
./autore "Please research new agent paradigms on arxiv and determine which ones will be used in 5 years based on current workplace trends" --retriever arxiv
```

The default `--retriever auto` currently uses arXiv. Use `--retriever local` only when you want the offline demo corpus.

Run the deterministic Phase 1 flow:

```bash
./autore "Research critic agents for evidence checking" --mode deterministic
```

Run the Phase 2 fan-out / fan-in flow:

```bash
./autore "Research how multi-agent systems improve automated literature review quality"
```

Run the nested evolutionary loop:

```bash
./autore "Research how multi-agent systems improve automated literature review quality" --mode loop --retriever local
```

Loop mode routes the task to `research` or `optimize`, runs an outer evolutionary orchestrator that proposes variants, sends each batch to the selected inner loop for scoring, records ranked variants, and stops on a threshold or plateau signal.

Force research mode:

```bash
./autore "Research agent paradigms and workplace trends" --mode loop --task-mode research
```

Run optimize mode with a registered deterministic evaluator:

```bash
./autore "Optimize a tiny scoring function" --mode loop --task-mode optimize --evaluator length_score
```

Each run prints:

```text
Run: run_<id>
Status: completed
Artifacts: outputs/run_<id>
Report: outputs/run_<id>/final_report.md
Run benchmark: outputs/run_<id>/run_benchmark.html
Decision DAG: outputs/run_<id>/decision_dag.svg
```

Open `final_report.md` for the synthesis and `run_benchmark.html` for a run-specific benchmark with a decision DAG, mode routing, variant scores, and stopping signals. Inspect `trace.jsonl` and the JSON artifact files to see how the report was produced.

---

## AI Credits And LLM Usage

The harness is now wired as an actual LLM agent architecture: proposal, judge, and synthesis steps use a live LLM when configured.

By default, `RESEARCH_HARNESS_LLM_PROVIDER=auto`:

- If `OPENAI_API_KEY` is present in `.env.local`, the harness calls the OpenAI Responses API.
- If no valid-looking key is present, it falls back to `local-deterministic-fallback` so tests and offline demos still run.
- Set `--llm-provider openai` when you want to require live OpenAI calls.
- Set `--llm-provider local` when you want an offline deterministic run.

Live LLM calls are used for:

- Outer-loop code/query variant proposals.
- Research-loop judge scoring.
- Final synthesis report writing.

Search agents still retrieve evidence through the configured retriever. Optimize mode still requires a deterministic evaluator for scoring; the LLM proposes variants, but the inner optimize score comes from the evaluator.

---

## Environment Setup

The harness automatically loads `.env`, then `.env.local`. Put secrets such as `OPENAI_API_KEY` in `.env.local`; it overrides `.env` and should stay out of git.

The current MVP works without API keys. It uses public unauthenticated sources by default and can use the deterministic local corpus in `examples/corpus/research_corpus.json` for offline demos.

Copy the example env file if you want shared defaults:

```bash
cp .env.example .env
```

Create `.env.local` for your machine-specific secrets:

```bash
cat > .env.local <<'EOF'
OPENAI_API_KEY=sk-...
RESEARCH_HARNESS_LLM_PROVIDER=auto
RESEARCH_HARNESS_LLM_MODEL=gpt-4.1-mini
EOF
```

After that, run normally:

```bash
./autore "Research agent paradigms and trends" --mode loop
```

Supported environment variables:

```text
RESEARCH_HARNESS_CORPUS_PATH=examples/corpus/research_corpus.json
RESEARCH_HARNESS_OUTPUT_DIR=outputs
RESEARCH_HARNESS_RETRIEVER=auto
RESEARCH_HARNESS_LLM_PROVIDER=auto
RESEARCH_HARNESS_LLM_MODEL=gpt-4.1-mini
OPENAI_API_KEY=
```

Retriever options:

```text
auto:
  Use the mixed live source strategy.

local:
  Always use examples/corpus/research_corpus.json.

arxiv:
  Always query the public arXiv API.

openalex:
  Query OpenAlex works.

github:
  Query GitHub repositories.

web:
  Query general web results.

docs_blogs:
  Query technical docs and blog-style web results.

twitter:
  Query public indexed X/Twitter pages through web search. This is not the official X API.

memory:
  Search prior artifacts from local outputs/run_* folders.
```

Live model and optional search integrations:

```text
OPENAI_API_KEY=
RESEARCH_HARNESS_LLM_PROVIDER=auto
RESEARCH_HARNESS_LLM_MODEL=gpt-4.1-mini
SEARCH_API_KEY=
LITERATURE_API_KEY=
GITHUB_TOKEN=
X_API_KEY=
```

The CLI automatically loads `.env` and then `.env.local` if they exist. `.env`, `.env.*`, virtualenvs, logs, and generated run outputs are ignored by git.

---

## Problem

Most current AI agents are built around a simple deterministic loop:

```text
user prompt -> model thinks -> tool calls -> model observes -> final answer
```

This loop is powerful for bounded work, but it becomes limiting for open-ended research.

Research usually does not begin with a perfectly scoped task. It often starts with vague questions:

- What is the current state of this field?
- What are the strongest arguments for and against this idea?
- Which papers, datasets, or methods matter?
- What should we try next?
- Why did this optimization strategy fail?
- What assumptions are we making?
- Where is the hidden bottleneck?

For these kinds of tasks, a single agent running a fixed loop often fails in predictable ways:

- It explores too narrowly.
- It stops too early.
- It over-trusts weak sources.
- It misses contradictory evidence.
- It summarizes before it understands.
- It produces polished reports without enough traceability.
- It requires repeated human prompting to introduce new directions or entropy.
- It does not preserve useful intermediate research state.

The bottleneck is not always the model. Often, the bottleneck is the harness around the model.

A better research agent needs more than a chat loop. It needs orchestration, parallel exploration, structured memory, critique, citations, tracing, and eventually the ability to improve its own research process.

---

## Goal

The goal of `research-harness` is to build a practical automated research system that can:

1. Accept a high-level research or optimization goal.
2. Break the goal into useful research directions.
3. Run multiple agents in parallel.
4. Search, collect, and summarize relevant evidence.
5. Generate hypotheses and possible explanations.
6. Critique claims and identify missing evidence.
7. Synthesize findings into a coherent report.
8. Store structured artifacts for future use.
9. Track costs, tool calls, prompts, and outputs.
10. Eventually propose improvements to its own harness.

The immediate implementation target is **Phase 1 and Phase 2**:

- Phase 1: deterministic research agent
- Phase 2: fan-out / fan-in parallel research harness

Future phases will add persistent world models, stronger observability, harness debugging, and adaptive harness evolution.

---

## What This Project Is

`research-harness` is a research orchestration system.

It is not just a wrapper around an LLM. It is a framework for coordinating multiple model-driven agents around a shared research goal.

At a high level:

```text
User Goal
  |
  v
Orchestrator
  |
  |-- Search Agent A
  |-- Search Agent B
  |-- Search Agent C
  |-- Hypothesis Agent A
  |-- Hypothesis Agent B
  |-- Critic Agent
  |-- Synthesis Agent
  |-- Harness Debugger
  |
  v
Shared Artifact Store
  |
  v
Final Research Report
```

The system is built around the idea that open-ended research benefits from parallelism, structured artifacts, and critique.

---

## What This Project Is Not

This project is not intended to be:

- A generic chatbot
- A fully autonomous AI scientist on day one
- A replacement for expert judgment
- A black-box answer generator
- A swarm of unconstrained agents
- A system that lets agents freely rewrite their own tools or prompts without review

The goal is controlled autonomy, not chaos.

Agents should be useful, observable, bounded, and auditable.

---

## Core Design Principles

### 1. Harnesses Matter

An agent is not just a model. An agent is a model inside a harness.

The harness defines:

- System prompts
- Tools
- Skills
- Memory
- Subagents
- Handoffs
- Middleware
- Permissions
- Budgets
- Stopping rules
- Logging
- Evaluation

The harness determines how the model acts in the world.

### 2. Research Needs Parallel Exploration

Many research questions benefit from multiple simultaneous approaches.

For example:

- One agent searches foundational literature.
- One agent searches recent empirical evidence.
- One agent searches limitations and contradictory evidence.
- One agent generates hypotheses.
- One agent critiques assumptions.
- One agent synthesizes findings.

This allows the system to explore multiple framings before converging too early.

### 3. Open-Ended Research Should Not Be Over-Constrained Too Early

For bounded coding or optimization tasks, agents should have:

- Clear role
- Bounded task
- Strict budget
- Defined output schema

For open-ended research, early agents may need looser exploration space.

Instead of immediately forcing narrow subtasks, the orchestrator should allow different agents to explore different angles, assumptions, and search strategies.

### 4. Every Claim Should Be Traceable

Research outputs should be grounded in artifacts.

The system preserves:

- Sources
- Claims
- Citations
- Hypotheses
- Experiments
- Contradictions
- Open questions
- Failed paths
- Harness-change proposals
- Agent traces

A final report should not be the only useful output. The intermediate research state should also be valuable.

### 5. Critique Is Part Of The Loop

A research harness should not only collect evidence. It should attack its own conclusions.

The critic agent exists to ask:

- Which claims are unsupported?
- Which sources are weak?
- What evidence contradicts the current synthesis?
- What did the search agents miss?
- Are we converging too early?
- What should be checked next?

### 6. Adaptation Should Be Gated

The long-term vision is a self-adapting harness.

However, harness adaptation should be constrained and observable.

A proposed harness change should explain:

```json
{
  "change": "Add a contradiction-checking critic after each literature batch",
  "reason": "Prior runs accepted claims from abstracts without source verification",
  "expected_effect": "Higher precision, fewer unsupported claims",
  "risk": "More token cost and slower runs",
  "evaluation": "Compare unsupported-claim rate before and after"
}
```

The system should not silently mutate itself. Proposed changes should be logged, evaluated, and accepted or rejected deliberately.

---

## Current Scope

This repository currently implements Phase 1, Phase 2, and a scaffolded Phase 3 nested evolutionary loop.

### Phase 1: Deterministic Research Agent

A single-agent research loop that can:

- Accept a research prompt
- Search a deterministic local research corpus
- Summarize findings
- Extract claims
- Track citations
- Generate a final report
- Log tool usage and runtime metadata

Status: implemented.

### Phase 2: Parallel Research Harness

A fan-out / fan-in harness that can:

- Spawn multiple research agents in parallel
- Assign different research directions
- Collect structured outputs
- Run hypothesis agents
- Run a critic/reviewer agent
- Run a synthesis agent
- Run a harness debugger that proposes constrained changes
- Produce a final report
- Store useful artifacts
- Track traces for each agent

Status: implemented as an MVP.

### Phase 3: Nested Evolutionary Loop

The harness now includes a nested evolutionary loop inspired by AlphaEvolve, FunSearch, and parallel Ralph-style evaluation.

The architecture has two levels:

```text
Outer orchestrator loop
  Propose code/query variants
  Select mode at ingestion
  Send variants to inner loop
  Tournament-select winners
  Mutate/refine next batch
  Stop on threshold or plateau

Inner evaluator loop
  OptimizeLoop: code variant -> deterministic float score
  ResearchLoop: query variant -> retrieval + judge ensemble score
```

Status: scaffolded with live LLM wiring. With `OPENAI_API_KEY`, the outer loop uses an LLM for proposal, the research loop uses an LLM judge score, and synthesis uses an LLM report writer. Without a key, the same harness falls back to local deterministic behavior for tests and offline demos.

Task ingestion chooses one of two modes:

- `optimize`: selected only when a deterministic evaluator is registered or explicitly supplied.
- `research`: selected when there is no deterministic evaluator, or when the user explicitly requests research.

The fallback rule is intentional: if `register_evaluator(fn)` cannot resolve an evaluator, the harness should not fake optimization. It routes to research mode and records the reason in `task_ingestion_decisions.json`.

Shared inner-loop contract:

- Input: a batch of `Variant` records.
- Output: ranked `VariantEvaluation` records.
- Required fields: `variant_id`, scalar `score`, metric breakdown, judge scores, pass/fail flag, summary, and termination signal.
- The outer orchestrator only sees ranked variants and termination signals, so it stays mode-agnostic.

Research scoring uses a judge ensemble:

- `coverage`: how many useful sources were retrieved.
- `corroboration`: how many claims were extracted across those sources.
- `credibility`: average source credibility.
- `stable_judge_score`: deterministic tie-breaker from the query and metrics.
- `llm_judge_score`: live model assessment when an LLM is configured.
- Final score: median of judge scores to reduce variance.

Future judge ensembles can add more independent model calls or specialized rubrics, but should preserve the same output schema.

Plateau detection works across modes with different noise assumptions:

- Optimize mode uses a small improvement epsilon and shorter patience because deterministic scores are low-noise.
- Research mode uses a larger improvement epsilon and longer patience because retrieval and judge scores are noisier.
- Termination signals include `score_threshold`, `claim_corroboration_threshold`, `score_plateau`, and `coverage_plateau`.

Generated loop artifacts:

```text
tasks.json
loop_iterations.json
task_ingestion_decisions.json
variants.json
variant_evaluations.json
evolution_rounds.json
progress.txt
trace.jsonl
final_report.md
```

Use `--max-iterations` to cap loop turns:

```bash
./autore "Research critic agents for evidence checking" --mode loop --max-iterations 12
```

---

## Architecture

### System Overview

```text
┌──────────────────┐
│   User Request   │
└────────┬─────────┘
         │
         v
┌──────────────────┐
│   Orchestrator   │
└────────┬─────────┘
         │
         ├──────────────┬──────────────┬──────────────┐
         v              v              v              v
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Search Agent│ │ Search Agent│ │ Hypothesis  │ │ Critic Agent│
│      A      │ │      B      │ │   Agent     │ │             │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │               │
       └───────────────┴───────┬───────┴───────────────┘
                               v
                    ┌──────────────────────┐
                    │ Shared Artifact Store │
                    └──────────┬───────────┘
                               v
                    ┌──────────────────────┐
                    │   Synthesis Agent     │
                    └──────────┬───────────┘
                               v
                    ┌──────────────────────┐
                    │ Final Research Report │
                    └──────────────────────┘
```

---

## Main Components

### Orchestrator

The orchestrator coordinates the full research run.

Responsibilities:

- Receive the user goal
- Decide whether the task is bounded or open-ended
- Create an initial research plan
- Spawn parallel agents
- Assign budgets and constraints
- Track agent trace IDs
- Collect results
- Pass artifacts to critic and synthesis agents
- Produce final output

Example orchestration flow:

```text
1. Parse user goal.
2. Determine research mode.
3. Create initial search directions.
4. Launch parallel search agents.
5. Launch hypothesis agents.
6. Collect outputs.
7. Run critic agent.
8. Run synthesis agent.
9. Run harness debugger.
10. Save final report and traces.
```

Implementation: `research_harness/orchestrator.py`

### Search Agents

Search agents collect information from sources.

In the current MVP, search agents can use `examples/corpus/research_corpus.json`, public arXiv, OpenAlex, GitHub repository search, general web search, docs/blog search, public social web search over indexed X/Twitter pages, and prior artifact memory from `outputs/run_*`.

The orchestrator now creates a source strategy before fan-out. For live research, that strategy launches separate search tracks:

- Core academic mechanisms
- Methods, benchmarks, and empirical evidence
- Implementation signals and open-source adoption
- Adoption signals and workplace-relevant directions
- Public social trend signals
- Contradictory evidence, limitations, and risks
- Prior related harness artifacts

Each track gets its own generated queries. This keeps the harness from acting like a single arXiv pipe while preserving a simple retriever interface that can later include Semantic Scholar, internal docs, private connectors, and richer prior world-model artifacts.

Responsibilities:

- Search sources
- Summarize each relevant source
- Extract useful claims
- Preserve citations
- Score relevance and credibility
- Identify gaps in the search space

Search agents produce structured artifacts, not only prose.

Example artifact shape:

```json
{
  "sources": [
    {
      "title": "Example Paper",
      "url": "https://example.com/paper",
      "source_type": "paper",
      "summary": "Short summary of the source.",
      "relevance_score": 0.87,
      "credibility_score": 0.78
    }
  ],
  "claims": [
    {
      "text": "The paper argues that retrieval quality is a major bottleneck in deep research agents.",
      "source_ids": ["src_123"],
      "confidence": 0.82,
      "support_level": "moderate"
    }
  ]
}
```

Implementation: `research_harness/agents.py` and `research_harness/search.py`

### Hypothesis Agent

The hypothesis agent generates possible explanations, theories, strategies, or next research directions.

Responsibilities:

- Generate hypotheses from collected evidence
- Link hypotheses to supporting claims
- Identify contradicting evidence
- Suggest experiments or follow-up searches
- Rank hypotheses by confidence, novelty, and testability

Example output shape:

```json
{
  "hypotheses": [
    {
      "text": "Adaptive harnesses improve open-ended research mainly by preventing premature convergence.",
      "supporting_claim_ids": ["claim_123", "claim_456"],
      "contradicting_claim_ids": [],
      "confidence": 0.74,
      "novelty_score": 0.62,
      "testability_score": 0.81,
      "next_experiment": "Compare fixed-loop and adaptive-loop agents on a benchmark with hidden contradictory sources."
    }
  ]
}
```

### Critic Agent

The critic agent reviews the current research state.

Responsibilities:

- Identify unsupported claims
- Detect contradictions
- Challenge weak sources
- Find missing perspectives
- Flag premature synthesis
- Recommend follow-up work

The critic should be adversarial but constructive.

Example critic questions:

```text
Which claims are not directly supported by citations?
Which cited sources are weak or secondary?
What would a skeptical domain expert object to?
What evidence would change the conclusion?
What search direction has not been explored?
Is the synthesis converging too early?
```

### Synthesis Agent

The synthesis agent creates the final research output.

Responsibilities:

- Merge findings from all agents
- Deduplicate claims and sources
- Weigh conflicting evidence
- Rank hypotheses
- State confidence levels
- Explain remaining uncertainty
- Produce a readable final report

The synthesis separates:

- What is known
- What is likely
- What is uncertain
- What is contradicted
- What should be investigated next

Current report structure:

```text
# Research Report

## Executive Synthesis
## Key Claims
## Ranked Hypotheses
## Contradictions And Caveats
## Open Questions
## Sources
```

### Harness Debugger

The harness debugger inspects traces and artifacts after a run.

Responsibilities:

- Identify failed traces
- Identify weak points in the research process
- Propose one constrained harness change
- Save the change as a pending artifact

The debugger does not apply its own proposed changes.

---

## Agent Constraints

Every agent supports:

- `max_steps`
- `max_tokens`
- `max_tool_calls`
- `max_runtime_seconds`
- `write_policy`
- `reporting_schema`
- `trace_id`
- `cancelled`
- shared artifact store access

For bounded tasks, agents should also have:

- Clear role
- Bounded task
- Expected output

For open-ended research, agents may instead receive:

- Research direction
- Exploration budget
- Source preferences
- Uncertainty tolerance
- Required artifact schema

---

## Write Policies

Agents should not all have the same permissions.

Current MVP write policies:

```text
append_only:
  Agent appends structured artifacts.

upsert_by_url:
  Agent deduplicates sources by URL before writing.

upsert_by_text:
  Reserved for deduplicating text-like artifacts.
```

Recommended future write policies:

```text
none:
  Agent cannot write to filesystem or artifact store.

artifact_store_only:
  Agent can write structured research artifacts.

worktree_only:
  Agent can edit files only inside an isolated worktree.

full_workspace:
  Agent can modify the main workspace.
```

For research tasks, most agents should use `artifact_store_only`.

For optimization or coding tasks, implementation agents should use `worktree_only`.

---

## Shared Artifact Store

The artifact store preserves intermediate research state.

Current implementation: JSON files plus a JSONL trace stream.

Minimum entities:

- `sources`
- `claims`
- `hypotheses`
- `experiments`
- `open_questions`
- `contradictions`
- `failed_paths`
- `harness_changes`
- `runs`
- `agent_traces`

This can initially be implemented with JSON files or SQLite. As the project matures, SQLite or Postgres may be more appropriate.

---

## Suggested Data Model

The full schema is documented in `docs/schema.md`.

### `sources`

```text
id
url
title
author
date
source_type
retrieved_at
summary
relevance_score
credibility_score
```

### `claims`

```text
id
text
source_ids
confidence
support_level
contradicted_by
created_by_agent
run_id
```

### `hypotheses`

```text
id
text
supporting_claim_ids
contradicting_claim_ids
confidence
novelty_score
testability_score
next_experiment
```

### `experiments`

```text
id
description
hypothesis_id
expected_signal
priority
```

### `open_questions`

```text
id
question
priority
reason
created_by_agent
status
```

### `contradictions`

```text
id
claim_a
claim_b
explanation
severity
resolution_status
```

### `failed_paths`

```text
id
description
reason
created_by_agent
run_id
```

### `harness_changes`

```text
id
change
reason
expected_effect
risk
evaluation
status
created_at
run_id
```

### `runs`

```text
id
user_goal
task_type
started_at
completed_at
status
total_cost
total_tokens
harness_config_id
```

### `agent_traces`

```text
id
run_id
agent_name
role
prompt
model
tools_used
tool_calls
token_usage
runtime_ms
status
errors
output_summary
```

---

## Observability

The harness logs every meaningful action it currently performs.

Important logs:

- User prompt
- Agent prompts
- Tool calls
- Retrieved sources
- Extracted claims
- Generated hypotheses
- Critic findings
- Synthesis output
- Token usage estimate
- Runtime
- Errors
- Harness config ID

A good research harness should make it possible to answer:

```text
Why did the system produce this conclusion?
Which agent found this source?
Which source supports this claim?
Which claims were contradicted?
How much did this run cost?
Where did the process fail?
```

The MVP writes:

```text
outputs/run_<id>/
  sources.json
  claims.json
  hypotheses.json
  experiments.json
  open_questions.json
  contradictions.json
  failed_paths.json
  harness_changes.json
  runs.json
  agent_traces.json
  task_ingestion_decisions.json
  variants.json
  variant_evaluations.json
  evolution_rounds.json
  trace.jsonl
  final_report.md
  run_benchmark.html
  run_benchmark.md
  run_benchmark_summary.json
  decision_dag.svg
  decision_dag.mmd
```

Benchmark all runs in `outputs/`:

```bash
./autore-bench
```

Write benchmarks to a specific folder:

```bash
./autore-bench --benchmark-output benchmarks/latest
```

The benchmark report includes:

- Time charts for run duration and summed agent runtime
- Artifact-count charts for sources, claims, hypotheses, experiments, open questions, contradictions, failed paths, and harness changes
- Error count and failed-agent charts
- Error-type breakdowns
- Token usage charts
- Harness-change counts, useful for studying adaptive harness behavior
- Source-type breakdowns
- Average claim confidence, source relevance, and source credibility
- `summary.json`, `runs.csv`, SVG charts, and an `index.html` dashboard

Each individual run also writes non-global benchmarks directly into its run folder:

- `run_benchmark.html`: quick visual explanation of the run.
- `run_benchmark.md`: Markdown version with a Mermaid decision DAG.
- `run_benchmark_summary.json`: machine-readable local metrics.
- `decision_dag.svg`: flowchart of prompt routing, outer-loop proposal, inner-loop evaluation, selection, stopping, and synthesis.
- `decision_dag.mmd`: Mermaid source for the same DAG.

Benchmark dashboards under `benchmarks/` are generated artifacts and are ignored by git. Reusable example goals and commands live in `examples/`.

---

## Example Usage

Example command:

```bash
./autore "Research adaptive agent harnesses for automated scientific discovery"
```

Example output:

```text
Run: run_41edba777656
Status: completed
Artifacts: outputs/run_41edba777656
Report: outputs/run_41edba777656/final_report.md
```

Example generated files:

```text
outputs/run_41edba777656/final_report.md
outputs/run_41edba777656/trace.jsonl
outputs/run_41edba777656/sources.json
outputs/run_41edba777656/claims.json
outputs/run_41edba777656/hypotheses.json
```

---

## Example Research Flow

Input:

```text
What are adaptive agent harnesses, and how could they improve automated research?
```

The orchestrator may spawn:

```text
Search Agent A:
  Search core academic mechanisms.

Search Agent B:
  Search methods, benchmarks, and empirical evidence.

Search Agent C:
  Search adoption signals and workplace-relevant directions.

Search Agent D:
  Search contradictory evidence, limitations, and risks.

Hypothesis Agent A:
  Generate possible mechanisms.

Hypothesis Agent B:
  Generate research directions.

Critic Agent:
  Challenge the evidence and identify weak claims.

Synthesis Agent:
  Merge, deduplicate, rank, and write the final report.

Harness Debugger:
  Inspect traces and propose one constrained harness improvement.
```

The synthesis agent then produces:

```text
- Summary of current state
- Key sources
- Main claims
- Ranked hypotheses
- Contradictions and caveats
- Open questions
- Recommended next searches
```

---

## Configuration

The harness is configurable in code through `HarnessConfig`.

Current config shape:

```python
HarnessConfig(
    id="phase2-local-deterministic-v1",
    mode="fanout",
    retriever="auto",
    search_agent_count=7,
    hypothesis_agent_count=2,
    include_debugger=True,
)
```

Each agent also receives an `AgentBudget`:

```python
AgentBudget(
    max_steps=4,
    max_tokens=4000,
    max_tool_calls=8,
    max_runtime_seconds=30.0,
    write_policy="append_only",
    reporting_schema="structured_artifact_v1",
)
```

Future config may move to JSON or TOML:

```json
{
  "model": "gpt-5.4",
  "max_parallel_agents": 5,
  "default_agent_budget": {
    "max_steps": 8,
    "max_tokens": 12000,
    "max_tool_calls": 20,
    "max_runtime_ms": 300000
  },
  "artifact_store": {
    "type": "sqlite",
    "path": "./data/research-harness.sqlite"
  },
  "logging": {
    "trace_path": "./outputs/traces",
    "log_tool_outputs": true
  }
}
```

---

## Output Format

Each run produces:

```text
outputs/
  run_<id>/
    final_report.md
    trace.jsonl
    sources.json
    claims.json
    hypotheses.json
    experiments.json
    open_questions.json
    contradictions.json
    failed_paths.json
    harness_changes.json
    runs.json
    agent_traces.json
```

The final report is readable by humans.

The artifacts are structured enough for future runs.

---

## Safety And Permissions

Research agents should operate with explicit permissions.

Recommended defaults:

- Search agents: can read search tools and write artifacts.
- Hypothesis agents: can read artifacts and write hypotheses.
- Critic agents: can read artifacts and write critique.
- Synthesis agents: can read all artifacts and write final report.
- Coding/optimization agents: should use isolated worktrees.

Agents should not be allowed to:

- Delete user files
- Modify the harness without approval
- Execute destructive shell commands
- Spend unbounded tokens
- Run forever
- Hide tool failures
- Silently drop sources

The current MVP does not give agents arbitrary shell or workspace mutation access. Agents write only structured artifacts and the final report.

---

## Stopping Logic

The system should stop when marginal research value falls below marginal cost.

Possible stopping signals:

- No meaningful new sources after N cycles
- No new high-confidence claims
- Hypotheses converge
- Critic finds no new major objections
- Search results become repetitive
- Cost budget is reached
- Time budget is reached
- Human approval is required

For Phase 1 and Phase 2, stopping is simple:

```text
Stop after all parallel agents complete and synthesis finishes.
```

Later phases can introduce plateau detection and iterative continuation.

---

## Development Roadmap

### Phase 1: Deterministic Research Agent

Build a strong fixed-loop research agent.

Features:

- Accept research prompt
- Search local corpus or arXiv
- Store citations
- Summarize sources
- Extract claims
- Generate report
- Track tool calls and runtime

Status: implemented.

### Phase 2: Parallel Research Harness

Add fan-out / fan-in orchestration.

Features:

- Multiple search agents
- Multiple hypothesis agents
- Critic/reviewer agent
- Synthesis agent
- Parallel execution
- Shared artifact store
- Trace IDs per agent

Status: implemented as an MVP.

---

## Future Features

These are intentionally out of scope for the first implementation, but are part of the long-term vision.

### Phase 3: Persistent World Model

Add durable structured memory across runs.

Entities:

- `claims`
- `sources`
- `hypotheses`
- `experiments`
- `open_questions`
- `contradictions`
- `failed_paths`
- `harness_changes`

The world model should let future research runs build on previous work instead of starting from scratch.

### Phase 4: Full Observability Layer

Add detailed tracing and analysis tools.

Features:

- Prompt logs
- Tool-call logs
- Cost tracking
- Runtime tracking
- Source lineage
- Claim lineage
- Error tracking
- Harness config snapshots

The goal is to make every run inspectable and reproducible.

### Phase 5: Harness Debugger

Add an agent that reviews the research process itself.

It should answer:

```text
Where did this research process fail?
Which harness component caused the failure?
What constrained change should be tested next?
What metric should decide whether to keep it?
```

Harness-change proposals should use:

```json
{
  "change": "Add a contradiction-checking critic after each literature batch",
  "reason": "Prior runs accepted claims from abstracts without source verification",
  "expected_effect": "Higher precision, fewer unsupported claims",
  "risk": "More token cost and slower runs",
  "evaluation": "Compare unsupported-claim rate before and after"
}
```

Status: basic MVP implemented. Deeper failure localization is future work.

### Phase 6: Adaptive Harness Evolution

Allow the system to propose and evaluate harness changes.

Flow:

```text
1. Propose harness change.
2. Save change as pending.
3. Apply change in sandbox/config variant.
4. Run benchmark or comparison task.
5. Compare against previous harness.
6. Keep, reject, or ask human.
```

The system should not silently rewrite itself. Adaptation should be gated.

### Phase 7: Long-Running Autonomous Research

Support longer research runs.

Features:

- Entropy injection
- Periodic self-review
- Plateau detection
- Human checkpoints
- Recovery from failed agents
- Continuation across sessions
- Research agendas that persist over time

---

## Future Feature: Entropy Injection

Open-ended research benefits from new perspectives.

Possible entropy sources:

- New papers
- Adjacent fields
- Contrarian arguments
- Randomized search directions
- Alternative keywords
- Different models
- Critic-generated objections
- Human notes
- Prior failed paths

Entropy should prevent premature convergence.

---

## Future Feature: Evaluation And Benchmarks

Adaptive harnesses need feedback signals.

Possible evaluation metrics:

- Source recall
- Source precision
- Unsupported-claim rate
- Citation quality
- Contradiction detection rate
- Novelty of hypotheses
- Usefulness of final report
- Cost per useful claim
- Human rating
- Reproducibility across runs

Benchmarks are not used because open-ended research has one correct answer. They are used because harness evolution needs pressure. Without evaluation, a self-adapting harness may only become more verbose, not more useful.

---

## Repository Structure

Current structure:

```text
research-harness/
  README.md
  TODO.md
  setup.cfg
  setup.py
  autore
  autore-bench
  run_example.py
  research_harness/
    __init__.py
    agents.py
    benchmark.py
    cli.py
    orchestrator.py
    schemas.py
    search.py
    store.py
  prompts/
    literature_agent.md
    hypothesis_agent.md
    critic_agent.md
    synthesis_agent.md
    harness_debugger.md
  docs/
    schema.md
  examples/
    README.md
    goals.txt
    corpus/
      research_corpus.json
  outputs/
    run_<id>/
      final_report.md
      trace.jsonl
      *.json
  benchmarks/
    <timestamp>/
      index.html
      summary.json
      runs.csv
      charts/
  tests/
    test_smoke.py
```

---

## Implementation Notes

Current technical choices:

- Python for the harness
- Dataclasses for typed schemas
- JSON files for artifact tables
- JSONL for traces
- `asyncio` for parallel agents
- Markdown for final reports
- Config-driven harness behavior
- Deterministic local corpus for reproducible MVP runs
- Public arXiv retriever for live literature-oriented runs

Future technical options:

- SQLite for local artifact storage
- Zod or Pydantic-style validation for stricter structured outputs
- Broader web, paper, repository, and internal-document retrievers
- Model-backed extraction, critique, and synthesis
- Trace visualization
- Benchmark runner for harness evolution

The MVP prioritizes correctness and observability over autonomy.

A simple reliable system is better than a large swarm that cannot explain itself.

---

## Testing

Run the smoke test:

```bash
python3 -m unittest discover -s tests
```

The smoke test verifies that a Phase 2 run:

- Completes successfully
- Writes a final report
- Stores sources and claims
- Generates hypotheses
- Logs agent traces
- Saves one harness-change proposal
- Generates benchmark summaries and charts

---

## Contributing

This project is early.

Useful contributions include:

- New agent roles
- Better artifact schemas
- Search integrations
- Citation extraction
- Evaluation metrics
- Example research tasks
- Visualization of traces
- Benchmark tasks
- Better synthesis prompts
- Safer tool permissioning

---

## License

TBD.

---

## Status

This project currently implements the first two phases:

1. Deterministic research agent
2. Fan-out / fan-in parallel research harness

Adaptive harness evolution is part of the roadmap, but not part of the initial MVP.
