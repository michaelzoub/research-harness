from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import uuid4


TaskType = Literal["bounded", "open_ended"]
RunStatus = Literal["running", "completed", "failed", "cancelled"]
WritePolicy = Literal["append_only", "upsert_by_url", "upsert_by_text"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass
class AgentBudget:
    max_steps: int = 4
    max_tokens: int = 4000
    max_tool_calls: int = 8
    max_runtime_seconds: float = 30.0
    write_policy: WritePolicy = "append_only"
    reporting_schema: str = "structured_artifact_v1"
    trace_id: str = field(default_factory=lambda: new_id("trace"))
    cancelled: bool = False


@dataclass
class Source:
    url: str
    title: str
    author: str
    date: str
    source_type: str
    summary: str
    relevance_score: float
    credibility_score: float
    id: str = field(default_factory=lambda: new_id("src"))
    retrieved_at: str = field(default_factory=now_iso)


@dataclass
class Claim:
    text: str
    source_ids: list[str]
    confidence: float
    support_level: str
    created_by_agent: str
    run_id: str
    contradicted_by: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: new_id("claim"))


@dataclass
class Hypothesis:
    text: str
    supporting_claim_ids: list[str]
    contradicting_claim_ids: list[str]
    confidence: float
    novelty_score: float
    testability_score: float
    next_experiment: str
    id: str = field(default_factory=lambda: new_id("hyp"))


@dataclass
class Experiment:
    description: str
    hypothesis_id: Optional[str]
    expected_signal: str
    priority: int
    id: str = field(default_factory=lambda: new_id("exp"))


@dataclass
class OpenQuestion:
    question: str
    priority: int
    reason: str
    created_by_agent: str
    status: str = "open"
    id: str = field(default_factory=lambda: new_id("q"))


@dataclass
class Contradiction:
    claim_a: str
    claim_b: str
    explanation: str
    severity: str
    resolution_status: str = "unresolved"
    id: str = field(default_factory=lambda: new_id("contra"))


@dataclass
class FailedPath:
    description: str
    reason: str
    created_by_agent: str
    run_id: str
    id: str = field(default_factory=lambda: new_id("fail"))


@dataclass
class HarnessChange:
    change: str
    reason: str
    expected_effect: str
    risk: str
    evaluation: str
    run_id: str
    status: str = "pending"
    created_at: str = field(default_factory=now_iso)
    id: str = field(default_factory=lambda: new_id("change"))


@dataclass
class RunRecord:
    user_goal: str
    task_type: TaskType
    harness_config_id: str
    status: RunStatus = "running"
    total_cost: float = 0.0
    total_tokens: int = 0
    started_at: str = field(default_factory=now_iso)
    completed_at: Optional[str] = None
    id: str = field(default_factory=lambda: new_id("run"))


@dataclass
class AgentTrace:
    run_id: str
    agent_name: str
    role: str
    prompt: str
    model: str
    tools_used: list[str]
    tool_calls: list[dict[str, Any]]
    token_usage: int
    runtime_ms: int
    status: str
    errors: list[str]
    output_summary: str
    id: str = field(default_factory=lambda: new_id("trace"))


@dataclass
class ResearchPlan:
    task_type: TaskType
    goal: str
    strategy: str
    search_angles: list[str]
    hypothesis_angles: list[str]
    stopping_signals: list[str]


@dataclass
class SourceStrategyItem:
    name: str
    retriever: str
    purpose: str
    queries: list[str]
    limit: int = 6


def to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
