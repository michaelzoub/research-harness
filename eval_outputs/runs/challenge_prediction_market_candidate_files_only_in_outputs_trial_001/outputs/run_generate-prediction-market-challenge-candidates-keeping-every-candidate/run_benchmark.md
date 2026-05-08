# Run Benchmark

- Run ID: `run_generate-prediction-market-challenge-candidates-keeping-every-candidate`
- Product agent: `challenge`
- Mode: `optimize_query`
- Tasks passed: 6 / 6
- Outer rounds: 2
- Variants evaluated: 7
- Best score: 0.869

## Decision DAG

```mermaid
flowchart TD
  prompt["Prompt: Generate prediction-market challenge candidates, keeping every candidate inside the run output directory"]
  route["Product agent: challenge\nLoop mode: optimize_query"]
  outer["Agent harness loop: propose variants"]
  inner["Inner loop: evaluate and rank"]
  select["Tournament selection"]
  stop{"Threshold or plateau?"}
  synth["Critic + synthesis + run benchmark"]
  prompt --> route --> outer --> inner --> select --> stop
  stop -- continue --> outer
  stop -- stop --> synth
  round1["Round 1: best=0.869, claim_corroboration_threshold"]
  inner --> round1 --> select
  round2["Round 2: best=0.502, improved"]
  inner --> round2 --> select
```

## Round Summary
- Round 1: best `variant_af3f241ebc4d` score 0.869; signal `claim_corroboration_threshold`.
- Round 2: best `variant_f791784ac944` score 0.502; signal `improved`.
