# Run Benchmark

- Run ID: `run_who-founded-apple-answer-with-concise-sourced-summary`
- Product agent: `research`
- Mode: `research`
- Tasks passed: 5 / 5
- Outer rounds: 1
- Variants evaluated: 1
- Best score: 0.960

## Decision DAG

```mermaid
flowchart TD
  prompt["Prompt: Research who founded Apple and answer with a concise, sourced summary"]
  route["Product agent: research\nLoop mode: research"]
  outer["Agent harness loop: propose variants"]
  inner["Inner loop: evaluate and rank"]
  select["Tournament selection"]
  stop{"Threshold or plateau?"}
  synth["Critic + synthesis + run benchmark"]
  prompt --> route --> outer --> inner --> select --> stop
  stop -- continue --> outer
  stop -- stop --> synth
  round1["Round 1: best=0.960, claim_corroboration_threshold"]
  inner --> round1 --> select
```

## Round Summary
- Round 1: best `variant_f97207678010` score 0.960; signal `claim_corroboration_threshold`.
