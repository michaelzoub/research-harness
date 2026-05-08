# Research Report: Get to $10 profit in the prediction market challenge. Keep every generated strategy as a run artifact, never as a source file.

- Run ID: `run_get-10-profit-prediction-market-challenge-keep-every-generated-strategy`
- Task type: `open_ended`
- Completed: 2026-05-07T23:59:15.864866+00:00
- Sources reviewed: 9
- Claims extracted: 30
- Hypotheses ranked: 0

## Executive Synthesis
Evidence quality is 0.68 on average. Leading direction: No ranked hypothesis emerged. Contradictions require follow-up before acting.

## Key Claims
- Positive edge comes from capturing spread against uninformed retail order flow. Confidence: 0.85 (strong). Sources: Orderbook Prediction Market Challenge (2026-05-04)
- Negative edge comes from stale quotes that are swept by an informed arbitrageur after fair-value moves. Confidence: 0.85 (strong). Sources: Orderbook Prediction Market Challenge (2026-05-04)
- The static competitor does not re-anchor to fair value after jumps, creating an opportunity for adaptive strategies. Confidence: 0.85 (strong). Sources: Orderbook Prediction Market Challenge (2026-05-04)
- Adaptive fair-value estimates should use competitor midpoint, recent fills, and stale-fill signals. Confidence: 0.79 (strong). Sources: Prediction Market Strategy Design Notes (2026-05-06)
- Canceling or widening quotes after jump or adverse-selection signals reduces arbitrageur losses. Confidence: 0.79 (strong). Sources: Prediction Market Strategy Design Notes (2026-05-06)
- Inventory skew and moderate size help preserve profitability across randomized regimes. Confidence: 0.79 (strong). Sources: Prediction Market Strategy Design Notes (2026-05-06)
- Literature grounding (initial) found: Positive edge comes from capturing spread against uninformed retail order flow. Confidence: 0.74 (retrieved). Sources: Orderbook Prediction Market Challenge (2026-05-04)
- Literature grounding (initial) found: Parallel search agents increase evidence recall when each agent explores a distinct framing. Confidence: 0.74 (retrieved). Sources: Parallel Agent Review Improves Evidence Recall In Literature Tasks (2025-02-14)
- Literature grounding (initial) found: Typed artifact stores make multi-agent research runs easier to audit. Confidence: 0.74 (retrieved). Sources: Structured Artifact Stores For Reproducible Agent Research (2023-08-21)
- The evaluator rewards adaptive estimation, inventory control, stale-quote protection, and passive liquidity provision. Confidence: 0.71 (moderate). Sources: Prediction Market Local Evaluator Rubric (2026-05-06)
- The evaluator penalizes static ladders, excessive size, and tight quotes without jump guards. Confidence: 0.71 (moderate). Sources: Prediction Market Local Evaluator Rubric (2026-05-06)
- The registered evaluator name is prediction_market and can be used with optimize_query mode. Confidence: 0.71 (moderate). Sources: Prediction Market Local Evaluator Rubric (2026-05-06)
- Parallel search agents increase evidence recall when each agent explores a distinct framing. Confidence: 0.66 (moderate). Sources: Parallel Agent Review Improves Evidence Recall In Literature Tasks (2025-02-14)
- Fan-in synthesis improves report coherence when claims are deduplicated before ranking. Confidence: 0.66 (moderate). Sources: Parallel Agent Review Improves Evidence Recall In Literature Tasks (2025-02-14)
- Uncriticized parallel outputs can increase unsupported claims because weak abstracts are repeated. Confidence: 0.66 (moderate). Sources: Parallel Agent Review Improves Evidence Recall In Literature Tasks (2025-02-14)
- Typed artifact stores make multi-agent research runs easier to audit. Confidence: 0.65 (moderate). Sources: Structured Artifact Stores For Reproducible Agent Research (2023-08-21)
- JSONL trace logs are sufficient for early reproducibility when prompts, tools, runtime, and outputs are captured. Confidence: 0.65 (moderate). Sources: Structured Artifact Stores For Reproducible Agent Research (2023-08-21)
- Persistent world models should be introduced after the artifact schema stabilizes. Confidence: 0.65 (moderate). Sources: Structured Artifact Stores For Reproducible Agent Research (2023-08-21)
- Harness changes should be proposed, evaluated, and gated before modifying execution behavior. Confidence: 0.62 (moderate). Sources: Risk Controls For Adaptive Agent Harnesses (2025-01-30)
- Adaptive evolution without benchmark comparison can worsen reliability. Confidence: 0.62 (moderate). Sources: Risk Controls For Adaptive Agent Harnesses (2025-01-30)
- Pending change schemas improve reviewability and prevent arbitrary self-modification. Confidence: 0.62 (moderate). Sources: Risk Controls For Adaptive Agent Harnesses (2025-01-30)
- Single deterministic agents can match multi-agent systems on narrow bounded research questions. Confidence: 0.61 (moderate). Sources: Single-Agent Baselines Remain Competitive For Narrow Research Questions (2024-04-18)
- Multi-agent fan-out has less advantage when the task already has clear success criteria. Confidence: 0.61 (moderate). Sources: Single-Agent Baselines Remain Competitive For Narrow Research Questions (2024-04-18)
- Parallel research increases cost and can decrease precision if agents duplicate similar searches. Confidence: 0.61 (moderate). Sources: Single-Agent Baselines Remain Competitive For Narrow Research Questions (2024-04-18)
- Critic agents reduce unsupported claims by challenging evidence gaps after extraction. Confidence: 0.6 (moderate). Sources: Reviewer Agents Reduce Unsupported Claims In Automated Research (2024-11-09)
- A critic pass increases runtime and may decrease novelty if applied too early. Confidence: 0.6 (moderate). Sources: Reviewer Agents Reduce Unsupported Claims In Automated Research (2024-11-09)
- Contradiction checks are most useful before final synthesis. Confidence: 0.6 (moderate). Sources: Reviewer Agents Reduce Unsupported Claims In Automated Research (2024-11-09)
- Open-ended research benefits from broad early exploration before constraints are tightened. Confidence: 0.59 (moderate). Sources: Entropy And Framing Diversity In Open-Ended Research Agents (2025-05-03)
- Entropy injection can reveal overlooked mechanisms and assumptions. Confidence: 0.59 (moderate). Sources: Entropy And Framing Diversity In Open-Ended Research Agents (2025-05-03)
- Too much exploration decreases precision when no critic-driven follow-up is used. Confidence: 0.59 (moderate). Sources: Entropy And Framing Diversity In Open-Ended Research Agents (2025-05-03)

## Ranked Hypotheses

## Contradictions And Caveats
- claim_405249b169db vs claim_4dfbb058baf7: Claims appear directionally opposed and need source-level resolution. Severity: medium.
- claim_405249b169db vs claim_8d6fb7d87d00: Claims appear directionally opposed and need source-level resolution. Severity: medium.

## Open Questions
- P1: Which mechanisms or optimization paths should be explored next? Reason: No hypotheses were generated from the current evidence set.

## Sources
- [Orderbook Prediction Market Challenge](https://github.com/danrobinson/prediction-market-challenge) by Dan Robinson (2026-05-04)
- [Parallel Agent Review Improves Evidence Recall In Literature Tasks](https://example.org/multi-agent-review-quality-2025) by A. Rivera and S. Chen (2025-02-14)
- [Structured Artifact Stores For Reproducible Agent Research](https://example.org/artifact-stores-for-agent-traces) by N. Patel (2023-08-21)
- [Prediction Market Strategy Design Notes](challenges/prediction_market/spec.md) by research-harness (2026-05-06)
- [Prediction Market Local Evaluator Rubric](challenges/prediction_market/evaluator.py) by research-harness (2026-05-06)
- [Entropy And Framing Diversity In Open-Ended Research Agents](https://example.org/open-ended-agent-exploration) by L. Singh (2025-05-03)
- [Single-Agent Baselines Remain Competitive For Narrow Research Questions](https://example.org/single-agent-baseline-limitations) by D. Morales (2024-04-18)
- [Reviewer Agents Reduce Unsupported Claims In Automated Research](https://example.org/critic-agents-evidence-checking) by M. Okafor (2024-11-09)
- [Risk Controls For Adaptive Agent Harnesses](https://example.org/adaptive-harness-risk-controls) by E. Novak (2025-01-30)

## Optimizer Seed Context
- Has evaluator: True
- Summary: Get to $10 profit in the prediction market challenge. Keep every generated strategy as a run artifact, never as a source file. adverse selection arbitrage inventory and risk controls; Get to $10 profit in the prediction market challenge. Keep every generated strategy as a run artifact, never as a source file. prediction-market microstructure and market-making mechanisms; Get to $10 profit in the prediction market challenge. Keep every generated strategy as a run artifact, never as a source file. AMM LMSR entropy and scoring-rule literature
- Query seed variant_d351099c8ab9: score 0.873; Get to $10 profit in the prediction market challenge. Keep every generated strategy as a run artifact, never as a source file. adverse selection arbitrage inventory and risk controls
- Query seed variant_047092776ce0: score 0.869; Get to $10 profit in the prediction market challenge. Keep every generated strategy as a run artifact, never as a source file. prediction-market microstructure and market-making mechanisms
- Query seed variant_058fb75720ad: score 0.868; Get to $10 profit in the prediction market challenge. Keep every generated strategy as a run artifact, never as a source file. AMM LMSR entropy and scoring-rule literature
