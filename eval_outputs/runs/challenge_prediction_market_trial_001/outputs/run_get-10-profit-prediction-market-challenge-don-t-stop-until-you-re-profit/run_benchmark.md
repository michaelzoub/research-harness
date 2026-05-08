# Run Benchmark

- Run ID: `run_get-10-profit-prediction-market-challenge-don-t-stop-until-you-re-profit`
- Product agent: `challenge`
- Mode: `optimize_query`
- Tasks passed: 5 / 6
- Outer rounds: 2
- Variants evaluated: 7
- Best score: 0.868

## Decision DAG

```mermaid
flowchart TD
  prompt["Prompt: Get to $10 profit in the prediction market challenge, don't stop until you're profitable. Introduce entropy fr"]
  route["Product agent: challenge\nLoop mode: optimize_query"]
  outer["Agent harness loop: propose variants"]
  inner["Inner loop: evaluate and rank"]
  select["Tournament selection"]
  stop{"Threshold or plateau?"}
  synth["Critic + synthesis + run benchmark"]
  prompt --> route --> outer --> inner --> select --> stop
  stop -- continue --> outer
  stop -- stop --> synth
  round1["Round 1: best=0.868, claim_corroboration_threshold"]
  inner --> round1 --> select
  round2["Round 2: best=0.502, improved"]
  inner --> round2 --> select
```

## Round Summary
- Round 1: best `variant_b2cf81241fc8` score 0.868; signal `claim_corroboration_threshold`.
- Round 2: best `variant_a2d346028b37` score 0.502; signal `improved`.
