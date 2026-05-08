# Research Report: Research a small deterministic fact about agent evaluation harnesses

- Run ID: `run_small-deterministic-fact-about-agent-evaluation-harnesses`
- Task type: `open_ended`
- Completed: 2026-05-07T23:59:17.497842+00:00
- Sources reviewed: 9
- Claims extracted: 27
- Hypotheses ranked: 1

## Executive Synthesis
Evidence quality is 0.61 on average. Leading direction: Mechanism path: Entropy injection can reveal overlooked mechanisms and assumptions. Contradictions require follow-up before acting.

## Key Claims
- Single deterministic agents can match multi-agent systems on narrow bounded research questions. Confidence: 0.7 (moderate). Sources: Single-Agent Baselines Remain Competitive For Narrow Research Questions (2024-04-18)
- Multi-agent fan-out has less advantage when the task already has clear success criteria. Confidence: 0.7 (moderate). Sources: Single-Agent Baselines Remain Competitive For Narrow Research Questions (2024-04-18)
- Parallel research increases cost and can decrease precision if agents duplicate similar searches. Confidence: 0.7 (moderate). Sources: Single-Agent Baselines Remain Competitive For Narrow Research Questions (2024-04-18)
- Parallel search agents increase evidence recall when each agent explores a distinct framing. Confidence: 0.67 (moderate). Sources: Parallel Agent Review Improves Evidence Recall In Literature Tasks (2025-02-14)
- Fan-in synthesis improves report coherence when claims are deduplicated before ranking. Confidence: 0.67 (moderate). Sources: Parallel Agent Review Improves Evidence Recall In Literature Tasks (2025-02-14)
- Uncriticized parallel outputs can increase unsupported claims because weak abstracts are repeated. Confidence: 0.67 (moderate). Sources: Parallel Agent Review Improves Evidence Recall In Literature Tasks (2025-02-14)
- Positive edge comes from capturing spread against uninformed retail order flow. Confidence: 0.64 (moderate). Sources: Orderbook Prediction Market Challenge (2026-05-04)
- Negative edge comes from stale quotes that are swept by an informed arbitrageur after fair-value moves. Confidence: 0.64 (moderate). Sources: Orderbook Prediction Market Challenge (2026-05-04)
- The static competitor does not re-anchor to fair value after jumps, creating an opportunity for adaptive strategies. Confidence: 0.64 (moderate). Sources: Orderbook Prediction Market Challenge (2026-05-04)
- The evaluator rewards adaptive estimation, inventory control, stale-quote protection, and passive liquidity provision. Confidence: 0.62 (moderate). Sources: Prediction Market Local Evaluator Rubric (2026-05-06)
- The evaluator penalizes static ladders, excessive size, and tight quotes without jump guards. Confidence: 0.62 (moderate). Sources: Prediction Market Local Evaluator Rubric (2026-05-06)
- The registered evaluator name is prediction_market and can be used with optimize_query mode. Confidence: 0.62 (moderate). Sources: Prediction Market Local Evaluator Rubric (2026-05-06)
- Adaptive fair-value estimates should use competitor midpoint, recent fills, and stale-fill signals. Confidence: 0.61 (moderate). Sources: Prediction Market Strategy Design Notes (2026-05-06)
- Canceling or widening quotes after jump or adverse-selection signals reduces arbitrageur losses. Confidence: 0.61 (moderate). Sources: Prediction Market Strategy Design Notes (2026-05-06)
- Inventory skew and moderate size help preserve profitability across randomized regimes. Confidence: 0.61 (moderate). Sources: Prediction Market Strategy Design Notes (2026-05-06)
- Critic agents reduce unsupported claims by challenging evidence gaps after extraction. Confidence: 0.6 (moderate). Sources: Reviewer Agents Reduce Unsupported Claims In Automated Research (2024-11-09)
- A critic pass increases runtime and may decrease novelty if applied too early. Confidence: 0.6 (moderate). Sources: Reviewer Agents Reduce Unsupported Claims In Automated Research (2024-11-09)
- Contradiction checks are most useful before final synthesis. Confidence: 0.6 (moderate). Sources: Reviewer Agents Reduce Unsupported Claims In Automated Research (2024-11-09)
- Typed artifact stores make multi-agent research runs easier to audit. Confidence: 0.59 (moderate). Sources: Structured Artifact Stores For Reproducible Agent Research (2023-08-21)
- JSONL trace logs are sufficient for early reproducibility when prompts, tools, runtime, and outputs are captured. Confidence: 0.59 (moderate). Sources: Structured Artifact Stores For Reproducible Agent Research (2023-08-21)
- Persistent world models should be introduced after the artifact schema stabilizes. Confidence: 0.59 (moderate). Sources: Structured Artifact Stores For Reproducible Agent Research (2023-08-21)
- Harness changes should be proposed, evaluated, and gated before modifying execution behavior. Confidence: 0.58 (moderate). Sources: Risk Controls For Adaptive Agent Harnesses (2025-01-30)
- Adaptive evolution without benchmark comparison can worsen reliability. Confidence: 0.58 (moderate). Sources: Risk Controls For Adaptive Agent Harnesses (2025-01-30)
- Pending change schemas improve reviewability and prevent arbitrary self-modification. Confidence: 0.58 (moderate). Sources: Risk Controls For Adaptive Agent Harnesses (2025-01-30)
- Open-ended research benefits from broad early exploration before constraints are tightened. Confidence: 0.51 (weak). Sources: Entropy And Framing Diversity In Open-Ended Research Agents (2025-05-03)
- Entropy injection can reveal overlooked mechanisms and assumptions. Confidence: 0.51 (weak). Sources: Entropy And Framing Diversity In Open-Ended Research Agents (2025-05-03)
- Too much exploration decreases precision when no critic-driven follow-up is used. Confidence: 0.51 (weak). Sources: Entropy And Framing Diversity In Open-Ended Research Agents (2025-05-03)

## Ranked Hypotheses
- Mechanism path: Entropy injection can reveal overlooked mechanisms and assumptions. Confidence: 0.41; novelty: 0.55; testability: 0.72. Next: Search for direct evaluations of: mechanism.

## Contradictions And Caveats
- claim_3f4ebd836099 vs claim_79a4dbc381d1: Claims appear directionally opposed and need source-level resolution. Severity: medium.
- claim_0830d193b0c4 vs claim_79a4dbc381d1: Claims appear directionally opposed and need source-level resolution. Severity: medium.

## Open Questions
- P2: What stronger evidence supports or refutes this claim: Open-ended research benefits from broad early exploration before constraints are tightened. Reason: Claim confidence is below the recommended synthesis threshold.
- P2: What stronger evidence supports or refutes this claim: Entropy injection can reveal overlooked mechanisms and assumptions. Reason: Claim confidence is below the recommended synthesis threshold.
- P2: What stronger evidence supports or refutes this claim: Too much exploration decreases precision when no critic-driven follow-up is used. Reason: Claim confidence is below the recommended synthesis threshold.

## Sources
- [Single-Agent Baselines Remain Competitive For Narrow Research Questions](https://example.org/single-agent-baseline-limitations) by D. Morales (2024-04-18)
- [Parallel Agent Review Improves Evidence Recall In Literature Tasks](https://example.org/multi-agent-review-quality-2025) by A. Rivera and S. Chen (2025-02-14)
- [Structured Artifact Stores For Reproducible Agent Research](https://example.org/artifact-stores-for-agent-traces) by N. Patel (2023-08-21)
- [Prediction Market Local Evaluator Rubric](challenges/prediction_market/evaluator.py) by research-harness (2026-05-06)
- [Reviewer Agents Reduce Unsupported Claims In Automated Research](https://example.org/critic-agents-evidence-checking) by M. Okafor (2024-11-09)
- [Risk Controls For Adaptive Agent Harnesses](https://example.org/adaptive-harness-risk-controls) by E. Novak (2025-01-30)
- [Orderbook Prediction Market Challenge](https://github.com/danrobinson/prediction-market-challenge) by Dan Robinson (2026-05-04)
- [Entropy And Framing Diversity In Open-Ended Research Agents](https://example.org/open-ended-agent-exploration) by L. Singh (2025-05-03)
- [Prediction Market Strategy Design Notes](challenges/prediction_market/spec.md) by research-harness (2026-05-06)
