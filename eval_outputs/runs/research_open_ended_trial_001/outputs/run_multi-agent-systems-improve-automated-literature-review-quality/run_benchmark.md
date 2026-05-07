# Run Benchmark

- Run ID: `run_multi-agent-systems-improve-automated-literature-review-quality`
- Mode: `research`
- Tasks passed: 5 / 5
- Outer rounds: 1
- Variants evaluated: 3
- Best score: 0.957

## Decision DAG

```mermaid
flowchart TD
  prompt["Prompt: Research how multi-agent systems improve automated literature review quality"]
  route["Route: research"]
  outer["Outer orchestrator: propose variants"]
  inner["Inner loop: evaluate and rank"]
  select["Tournament selection"]
  stop{"Threshold or plateau?"}
  synth["Critic + synthesis + run benchmark"]
  prompt --> route --> outer --> inner --> select --> stop
  stop -- continue --> outer
  stop -- stop --> synth
  round1["Round 1: best=0.957, claim_corroboration_threshold"]
  inner --> round1 --> select
```

## Round Summary
- Round 1: best `variant_5c856588b3ca` score 0.957; signal `claim_corroboration_threshold`.
