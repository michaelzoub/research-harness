# Run Benchmark

- Run ID: `run_run-prediction-market-challenge-locally-when-upstream-official-scorer-is`
- Product agent: `challenge`
- Mode: `optimize_query`
- Tasks passed: 6 / 6
- Outer rounds: 2
- Variants evaluated: 7
- Best score: 0.869

## Decision DAG

```mermaid
flowchart TD
  prompt["Prompt: Run the prediction-market challenge locally when the upstream official scorer is not required"]
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
- Round 1: best `variant_8b846bc37334` score 0.869; signal `claim_corroboration_threshold`.
- Round 2: best `variant_96c6325a69b4` score 0.502; signal `improved`.
