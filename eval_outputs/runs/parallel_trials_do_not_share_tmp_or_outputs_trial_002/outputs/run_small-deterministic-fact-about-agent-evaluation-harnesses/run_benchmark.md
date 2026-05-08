# Run Benchmark

- Run ID: `run_small-deterministic-fact-about-agent-evaluation-harnesses`
- Product agent: `research`
- Mode: `research`
- Tasks passed: 5 / 5
- Outer rounds: 1
- Variants evaluated: 4
- Best score: 0.967

## Decision DAG

```mermaid
flowchart TD
  prompt["Prompt: Research a small deterministic fact about agent evaluation harnesses"]
  route["Product agent: research\nLoop mode: research"]
  outer["Agent harness loop: propose variants"]
  inner["Inner loop: evaluate and rank"]
  select["Tournament selection"]
  stop{"Threshold or plateau?"}
  synth["Critic + synthesis + run benchmark"]
  prompt --> route --> outer --> inner --> select --> stop
  stop -- continue --> outer
  stop -- stop --> synth
  round1["Round 1: best=0.967, claim_corroboration_threshold"]
  inner --> round1 --> select
```

## Round Summary
- Round 1: best `variant_8572c57db252` score 0.967; signal `claim_corroboration_threshold`.
