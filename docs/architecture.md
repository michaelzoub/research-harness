# Project Architecture

This project treats each product option as an agent in the conventional sense:

```text
agent = model + harness
```

The model is the inference component. The harness is the loop, tools,
evaluators, artifact store, budgets, traces, stopping rules, and orchestration
policy that make the model act on a task.

## Product Agents

There are three product agents:

| Product agent | Runtime loop mode | Primary objective |
| --- | --- | --- |
| `research` | `research` | Find papers/data, extract claims, synthesize grounded reports. |
| `optimize` | `optimize` or `optimize_query` | Improve a candidate against deterministic tests/evaluators. |
| `challenge` | `optimize_query` plus `optimize` | Solve benchmark/challenge tasks with proxy and official graders. |

`optimize` and `challenge` intentionally share the same optimization core. A
challenge is an optimization task with extra contract requirements: challenge
specs, solution rendering, optional official runner integration, and challenge
specific graders.

Every run writes a `prd.json`. The PRD records the selected product agent,
runtime mode, agent-harness definition, ordered tasks, acceptance criteria, and
artifact paths.

## Three-Loop Architecture

The harness runs three nested loops:

```text
Outer loop  — Session (sessions.py)
              Manages context isolation and parallel agent runs.
              Resets state between runs so each agent starts clean.

Middle loop — EvolutionaryOuterLoop (loops.py)
              Proposes and evaluates variants across N outer iterations.
              Drives research (query variants → retrieve → score) or
              optimize (code variants → evaluator → score).

Inner loop  — Ralph loop (orchestrator._run_loop + agent harness)
              The agent harness: model + loop policy + tools + store.
              Picks next story (passes: false) from prd.json.
              Executes it via research or optimization agent.
              Updates prd.json (passes: true) and appends progress.txt.
              Repeats until all stories pass or iteration budget exhausted.
```

```mermaid
flowchart TD
  prd["prd.json\nUS-001..US-N\npasses: false"] --> pick["Pick next story\npasses: false"]
  pick --> exec["Execute story\nresearch / optimize agent"]
  exec --> commit["EvolutionaryOuterLoop\npropose variants → evaluate → rank"]
  commit --> update["Update prd.json\npasses: true"]
  update --> log["Append progress.txt\nsave learnings"]
  log --> more{"More stories?"}
  more -->|yes| pick
  more -->|no| done["Done\nAll stories complete"]
```

## System Diagram

The visual architecture/roadmap diagram is available at
[`docs/assets/research_harness_architecture_phases_3_7.svg`](assets/research_harness_architecture_phases_3_7.svg).

```mermaid
flowchart LR
  user["User goal / CLI flags"] --> cli["CLI\nresearch_harness.cli"]
  cli --> config["HarnessConfig\nmode, retriever, evaluator, LLM, budgets"]
  config --> orchestrator["Agent Orchestrator\nresearch_harness.orchestrator"]

  orchestrator --> plan["ResearchPlan + SourceStrategy\nclassify goal, choose sources"]
  orchestrator --> run["RunRecord\nstatus, task_mode, product_agent"]
  orchestrator --> store["ArtifactStore\nJSON tables + traces + PRD"]
  orchestrator --> router["TaskRouter\nselect product agent + loop mode"]

  router -->|research| research_agent["Research Agent\nmodel + research loop harness"]
  router -->|optimize| optimize_agent["Optimize Agent\nmodel + optimization loop harness"]
  router -->|challenge| challenge_agent["Challenge Agent\noptimization harness + challenge contracts"]

  research_agent --> loop_research["ResearchLoop\nquery variants -> retrieve -> claims -> score"]
  optimize_agent --> loop_optimize["OptimizeLoop\ncandidate variants -> evaluator -> score"]
  challenge_agent --> loop_challenge["OptimizationQueryLoop -> OptimizeLoop\nstrategy research -> seed context -> candidate score"]

  loop_research --> store
  loop_optimize --> store
  loop_challenge --> store

  orchestrator --> roles["Role agents\nLiterature, Hypothesis, Critic, Synthesis, Debugger"]
  roles --> store
  store --> artifacts["Artifacts\nprd.json, progress.txt, trace.jsonl,\nvariants, evaluations, report, solution"]
```

## Agent Harness Internals

```mermaid
flowchart TD
  agent["Product Agent"] --> model["Model\nLLMClient live or local fallback"]
  agent --> harness["Harness Loop"]
  agent --> tools["Tools\nretrievers, evaluators, challenge runners"]
  agent --> state["State\nArtifactStore"]
  agent --> policy["Policy\nbudget, stopping rules, plateau detection"]
  agent --> trace["Trajectory\nAgentTrace, progress, loop iterations"]

  harness --> propose["Propose variants or tasks"]
  propose --> parallel{"Can run in parallel?"}
  parallel -->|yes| fanout["asyncio.gather\nparallel agents or evaluations"]
  parallel -->|no| sequential["dependency-ordered execution"]
  fanout --> evaluate["Evaluate / score / extract evidence"]
  sequential --> evaluate
  evaluate --> rank["Rank, select, or synthesize"]
  rank --> stop{"Threshold, plateau, or task complete?"}
  stop -->|continue| propose
  stop -->|stop| output["Persist outcome artifacts"]

  model <--> harness
  tools <--> harness
  state <--> harness
  trace --> state
  output --> state
```

## Evaluation Harness

The evaluation harness runs the product agents as black-box systems and grades
their trajectories and outcomes.

```mermaid
flowchart TD
  eval_harness["EvaluationHarness"] --> suite["EvalSuite"]
  suite --> task["EvalTask\nprompt, product expectation, mode,\nevaluator, success criteria, graders"]
  task --> trial1["EvalTrial 1"]
  task --> trial2["EvalTrial N"]

  trial1 --> isolated["Isolated output root + TMPDIR"]
  isolated --> orchestrator["Production Orchestrator"]
  orchestrator --> agent_harness["Selected Product Agent Harness"]
  agent_harness --> trajectory["Trajectory\nprogress, trace, agent_traces, artifacts"]
  trajectory --> outcome["Outcome\nstatus, product_agent, best score,\nreport/solution existence"]

  outcome --> graders["Graders"]
  trajectory --> graders
  graders --> aggregate["Aggregate score + pass/fail"]
```

## Product Agent Details

### Research Agent

```text
input goal
  -> route product_agent=research, loop_mode=research
  -> propose query variants
  -> run retrievers, possibly in parallel
  -> write sources and claims
  -> score evidence coverage/corroboration/credibility
  -> generate hypotheses
  -> critique contradictions
  -> synthesize final_report.md
```

### Optimize Agent

```text
input goal + evaluator
  -> route product_agent=optimize
  -> choose optimize or optimize_query loop
  -> propose candidate variants
  -> evaluate with deterministic evaluator/tests
  -> rank by score
  -> write optimized_candidate.txt, optimal_code.py, and optimization_result.json
```

### Challenge Agent

```text
input challenge goal + challenge evaluator
  -> route product_agent=challenge, loop_mode=optimize_query
  -> research challenge strategies
  -> write optimizer_seed_context.json
  -> propose candidate strategies
  -> evaluate local proxy
  -> render optimal_code.py for the selected strategy
  -> mirror to solution.py when a challenge adapter supports that upstream filename
  -> record official_result.measured=false until official runner executes
```

## Optimize And Challenge Relationship

`optimize` and `challenge` should stay fused at the loop/evaluator layer:

```text
Optimization core = propose candidates + evaluate + rank + stop + persist best
```

Challenge mode should remain a product-agent specialization:

```text
Challenge specialization = optimization core
  + challenge spec
  + proxy evaluator
  + optional official evaluator
  + solution renderer
  + challenge-specific graders
```

That keeps the implementation simple without erasing the product distinction
that matters to users and eval reporting.

## Optimization Output Contract

Every optimization or challenge run that executes an evaluator must write:

```text
optimized_candidate.txt
optimal_code.py
optimization_result.json
```

`optimization_result.json` must include `optimal_code_path`. Challenge-specific
artifacts such as `solution.py` are allowed, but they are additional artifacts,
not replacements for `optimal_code.py`.
