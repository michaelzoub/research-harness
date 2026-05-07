# Artifact Schema

The MVP uses JSON files as database tables. Later phases can map these directly
to SQLite or Postgres rows.

## sources

- `id`
- `url`
- `title`
- `author`
- `date`
- `source_type`
- `retrieved_at`
- `summary`
- `relevance_score`
- `credibility_score`

## claims

- `id`
- `text`
- `source_ids`
- `confidence`
- `support_level`
- `contradicted_by`
- `created_by_agent`
- `run_id`

## hypotheses

- `id`
- `text`
- `supporting_claim_ids`
- `contradicting_claim_ids`
- `confidence`
- `novelty_score`
- `testability_score`
- `next_experiment`

## experiments

- `id`
- `description`
- `hypothesis_id`
- `expected_signal`
- `priority`

## open_questions

- `id`
- `question`
- `priority`
- `reason`
- `created_by_agent`
- `status`

## contradictions

- `id`
- `claim_a`
- `claim_b`
- `explanation`
- `severity`
- `resolution_status`

## failed_paths

- `id`
- `description`
- `reason`
- `created_by_agent`
- `run_id`

## harness_changes

- `id`
- `change`
- `reason`
- `expected_effect`
- `risk`
- `evaluation`
- `status`
- `created_at`
- `run_id`

## runs

- `id`
- `user_goal`
- `task_type`
- `task_mode`
- `product_agent`
- `started_at`
- `completed_at`
- `status`
- `total_cost`
- `total_tokens`
- `harness_config_id`

## task_ingestion_decisions

- `id`
- `requested_mode`
- `selected_mode`
- `product_agent`
- `evaluator_name`
- `reason`

## Optimization artifacts

Optimization and challenge runs that execute an evaluator write:

- `optimized_candidate.txt`: exact selected best candidate payload
- `optimal_code.py`: universal code artifact for the selected best candidate
- `optimization_result.json`: score, evaluator metadata, candidate path, and `optimal_code_path`
- `solution.py`: optional challenge-specific mirror for upstream runners

## agent_traces

- `id`
- `run_id`
- `agent_name`
- `role`
- `prompt`
- `model`
- `tools_used`
- `tool_calls`
- `token_usage`
- `runtime_ms`
- `status`
- `errors`
- `output_summary`
