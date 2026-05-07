# Run Benchmark

- Run ID: `run_optimization-strategies-tiny-scoring-benchmark`
- Mode: `optimize_query`
- Tasks passed: 6 / 6
- Outer rounds: 2
- Variants evaluated: 7
- Best score: 0.805

## Decision DAG

```mermaid
flowchart TD
  prompt["Prompt: Research optimization strategies for a tiny scoring benchmark"]
  route["Route: optimize_query"]
  outer["Outer orchestrator: propose variants"]
  inner["Inner loop: evaluate and rank"]
  select["Tournament selection"]
  stop{"Threshold or plateau?"}
  synth["Critic + synthesis + run benchmark"]
  prompt --> route --> outer --> inner --> select --> stop
  stop -- continue --> outer
  stop -- stop --> synth
  round1["Round 1: best=0.805, claim_corroboration_threshold"]
  inner --> round1 --> select
  round2["Round 2: best=0.053, continue"]
  inner --> round2 --> select
```

## Round Summary
- Round 1: best `variant_c93f0cd62ada` score 0.805; signal `claim_corroboration_threshold`.
- Round 2: best `variant_8b1d117657c2` score 0.053; signal `continue`.
