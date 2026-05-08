from __future__ import annotations

"""Best optimization candidate emitted by research-harness.

This module is written for every optimization or challenge run so downstream
evaluators can always find the agent-selected code artifact at optimal_code.py.
When a domain adapter can render executable code, it should replace this generic
representation with evaluator-ready code.
"""

EVALUATOR_NAME = 'length_score'
OPTIMAL_CANDIDATE = 'vectorized implementation'


def selected_candidate() -> str:
    """Return the exact candidate payload that achieved the best score."""
    return OPTIMAL_CANDIDATE
