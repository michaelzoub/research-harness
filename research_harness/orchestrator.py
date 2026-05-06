from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from .agents import (
    CriticAgent,
    HarnessDebuggerAgent,
    HypothesisAgent,
    LiteratureAgent,
    SynthesisAgent,
)
from .schemas import AgentBudget, ResearchPlan, RunRecord, SourceStrategyItem, TaskType, now_iso
from .search import ArxivSearch, LocalCorpusSearch, SearchBackend
from .search import DocsBlogsSearch
from .search import GitHubSearch
from .search import OpenAlexSearch
from .search import PriorArtifactMemorySearch
from .search import SocialWebSearch
from .search import WebSearch
from .store import ArtifactStore


STOPPING_SIGNALS = [
    "no meaningful new sources after N cycles",
    "no new high-confidence claims",
    "hypotheses converge",
    "critic finds no new major objections",
    "benchmark/evaluation score plateaus",
    "cost budget reached",
    "time budget reached",
    "human approval needed",
]


BOUNDED_MARKERS = {
    "optimize",
    "debug",
    "fix",
    "implement",
    "compare",
    "benchmark",
    "choose",
    "rank",
}


RUN_SLUG_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "based",
    "be",
    "can",
    "for",
    "how",
    "in",
    "of",
    "on",
    "ones",
    "please",
    "research",
    "the",
    "to",
    "used",
    "what",
    "which",
    "will",
}


@dataclass
class HarnessConfig:
    id: str = "phase2-research-harness-v1"
    mode: str = "fanout"
    retriever: str = "auto"
    search_agent_count: int = 7
    hypothesis_agent_count: int = 2
    include_debugger: bool = True
    default_budget: AgentBudget = field(default_factory=AgentBudget)


class Orchestrator:
    def __init__(self, corpus_path: Path, output_root: Path, config: Optional[HarnessConfig] = None):
        self.corpus_path = corpus_path
        self.output_root = output_root
        self.config = config or HarnessConfig()

    async def run(self, goal: str, mode: Optional[str] = None) -> Tuple[RunRecord, ArtifactStore]:
        selected_mode = mode or self.config.mode
        plan = self.create_plan(goal)
        source_strategy = self.create_source_strategy(goal, plan)
        primary_retriever = self._retriever_for(source_strategy[0].retriever)
        run = RunRecord(
            id=self._next_run_id(goal),
            user_goal=goal,
            task_type=plan.task_type,
            harness_config_id=f"{self.config.id}:{selected_mode}:{primary_retriever.tool_name}:source_strategy_v1",
        )
        store = ArtifactStore(self.output_root / run.id)
        store.add_run(run)
        try:
            if selected_mode == "deterministic":
                await self._run_phase1(run, store, plan, source_strategy)
            elif selected_mode == "fanout":
                await self._run_phase2(run, store, plan, source_strategy)
            else:
                raise ValueError(f"Unknown mode: {selected_mode}")
            run.status = "completed"
        except Exception:
            run.status = "failed"
            raise
        finally:
            traces = store.list("agent_traces")
            run.total_tokens = sum(int(trace["token_usage"]) for trace in traces)
            run.completed_at = now_iso()
            store.update_run(run)
        return run, store

    def classify_task(self, goal: str) -> TaskType:
        normalized = goal.lower()
        if any(marker in normalized for marker in BOUNDED_MARKERS):
            return "bounded"
        return "open_ended"

    def create_plan(self, goal: str) -> ResearchPlan:
        task_type = self.classify_task(goal)
        if task_type == "bounded":
            search_angles = [
                "baseline evidence",
                "failure modes constraints",
                "evaluation metrics",
            ]
            hypothesis_angles = ["optimization path", "risk mitigation"]
            strategy = "Assign bounded roles and collect evidence against explicit success criteria."
        else:
            search_angles = [
                "foundational literature mechanisms",
                "recent empirical evidence",
                "contradictory evidence limitations",
            ]
            hypothesis_angles = ["mechanism", "research direction"]
            strategy = "Allow broad exploration across framings before critic-driven convergence."
        return ResearchPlan(
            task_type=task_type,
            goal=goal,
            strategy=strategy,
            search_angles=search_angles,
            hypothesis_angles=hypothesis_angles,
            stopping_signals=STOPPING_SIGNALS,
        )

    def create_source_strategy(self, goal: str, plan: ResearchPlan) -> list[SourceStrategyItem]:
        retriever = self.config.retriever.lower()
        if retriever == "local":
            return [
                SourceStrategyItem(
                    name=f"local_{index + 1}",
                    retriever="local",
                    purpose=angle,
                    queries=[f"{goal} {angle}"],
                    limit=self.config.default_budget.max_tool_calls,
                )
                for index, angle in enumerate(plan.search_angles)
            ]
        if retriever == "auto":
            return _mixed_source_strategy(goal, plan)
        if retriever in {"arxiv", "openalex", "github", "web", "docs_blogs", "twitter", "memory"}:
            return _single_retriever_strategy(goal, plan, retriever)
        if retriever not in {"auto", "arxiv"}:
            raise ValueError(f"Unknown retriever: {self.config.retriever}")
        return _mixed_source_strategy(goal, plan)

    async def _run_phase1(
        self, run: RunRecord, store: ArtifactStore, plan: ResearchPlan, source_strategy: list[SourceStrategyItem]
    ) -> None:
        item = source_strategy[0]
        literature = LiteratureAgent(
            name="deterministic_literature_agent",
            query_angle=item.purpose,
            corpus=self._retriever_for(item.retriever),
            prompt_template=_prompt("literature_agent"),
            budget=self._budget("append_only"),
            search_queries=item.queries,
        )
        critic = CriticAgent(
            name="deterministic_critic",
            role="critic_reviewer",
            prompt_template=_prompt("critic_agent"),
            budget=self._budget("append_only"),
        )
        synth = SynthesisAgent(
            name="deterministic_synthesis",
            role="synthesis_agent",
            prompt_template=_prompt("synthesis_agent"),
            budget=self._budget("append_only"),
        )
        for agent in [literature, critic, synth]:
            await agent.execute(run, store)

    async def _run_phase2(
        self, run: RunRecord, store: ArtifactStore, plan: ResearchPlan, source_strategy: list[SourceStrategyItem]
    ) -> None:
        search_agents = [
            LiteratureAgent(
                name=f"literature_agent_{index + 1}",
                query_angle=item.purpose,
                corpus=self._retriever_for(item.retriever),
                prompt_template=_prompt("literature_agent"),
                budget=self._budget("upsert_by_url"),
                search_queries=item.queries,
            )
            for index, item in enumerate(source_strategy[: self.config.search_agent_count])
        ]
        await asyncio.gather(*(agent.execute(run, store) for agent in search_agents))

        hypothesis_agents = [
            HypothesisAgent(
                name=f"hypothesis_agent_{index + 1}",
                hypothesis_angle=angle,
                prompt_template=_prompt("hypothesis_agent"),
                budget=self._budget("append_only"),
            )
            for index, angle in enumerate(plan.hypothesis_angles[: self.config.hypothesis_agent_count])
        ]
        await asyncio.gather(*(agent.execute(run, store) for agent in hypothesis_agents))

        critic = CriticAgent(
            name="critic_reviewer",
            role="critic_reviewer",
            prompt_template=_prompt("critic_agent"),
            budget=self._budget("append_only"),
        )
        synthesis = SynthesisAgent(
            name="synthesis_agent",
            role="synthesis_agent",
            prompt_template=_prompt("synthesis_agent"),
            budget=self._budget("append_only"),
        )
        await critic.execute(run, store)
        await synthesis.execute(run, store)

        if self.config.include_debugger:
            debugger = HarnessDebuggerAgent(
                name="harness_debugger",
                role="harness_debugger",
                prompt_template=_prompt("harness_debugger"),
                budget=self._budget("append_only"),
            )
            await debugger.execute(run, store)

    def _budget(self, write_policy: str) -> AgentBudget:
        return AgentBudget(
            max_steps=self.config.default_budget.max_steps,
            max_tokens=self.config.default_budget.max_tokens,
            max_tool_calls=self.config.default_budget.max_tool_calls,
            max_runtime_seconds=self.config.default_budget.max_runtime_seconds,
            write_policy=write_policy,  # type: ignore[arg-type]
            reporting_schema=self.config.default_budget.reporting_schema,
        )

    def _retriever_for(self, retriever: str) -> SearchBackend:
        retriever = retriever.lower()
        if retriever == "local":
            return LocalCorpusSearch(self.corpus_path)
        if retriever == "arxiv":
            return ArxivSearch()
        if retriever == "openalex":
            return OpenAlexSearch()
        if retriever == "github":
            return GitHubSearch()
        if retriever == "web":
            return WebSearch()
        if retriever == "docs_blogs":
            return DocsBlogsSearch()
        if retriever == "twitter":
            return SocialWebSearch()
        if retriever == "memory":
            return PriorArtifactMemorySearch(self.output_root)
        if retriever == "auto":
            return OpenAlexSearch()
        raise ValueError(f"Unknown retriever: {retriever}")

    def _next_run_id(self, goal: str) -> str:
        base = goal_slug(goal)
        candidate = f"run_{base}"
        if not (self.output_root / candidate).exists():
            return candidate
        index = 2
        while True:
            numbered = f"{candidate}-{index:02d}"
            if not (self.output_root / numbered).exists():
                return numbered
            index += 1


def _prompt(name: str) -> str:
    path = Path(__file__).resolve().parent.parent / "prompts" / f"{name}.md"
    return path.read_text(encoding="utf-8")


def goal_slug(goal: str, max_length: int = 72) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", goal.lower())
    selected = [word for word in words if word not in RUN_SLUG_STOPWORDS]
    if not selected:
        selected = words or ["research-run"]
    slug = "-".join(selected)
    slug = slug[:max_length].strip("-")
    return slug or "research-run"


def _mixed_source_strategy(goal: str, plan: ResearchPlan) -> list[SourceStrategyItem]:
    concepts = _goal_terms(goal)
    core = " ".join(concepts[:6]) or goal
    brain_terms = "brain neuroscience cognitive computational neuroscience intelligence"
    agent_terms = "agentic agents autonomous multi-agent planner executor tool use memory"
    workplace_terms = "workplace automation enterprise productivity software agents"
    trend_terms = "survey benchmark adoption evaluation future trends"
    contrarian_terms = "limitations failures safety reliability critique"
    if any(term in concepts for term in ["brain", "brains", "neuroscience", "cognitive"]):
        first_query = f"{core} {brain_terms}"
    else:
        first_query = f"{core} {agent_terms}"
    return [
        SourceStrategyItem(
            name="academic_papers",
            retriever="openalex",
            purpose="core academic mechanisms",
            queries=[first_query, f"{core} survey taxonomy"],
            limit=8,
        ),
        SourceStrategyItem(
            name="preprints_and_benchmarks",
            retriever="arxiv",
            purpose="methods, benchmarks, and empirical evidence",
            queries=[f"{core} {trend_terms}", f"{core} benchmark evaluation"],
            limit=8,
        ),
        SourceStrategyItem(
            name="implementation_signals",
            retriever="github",
            purpose="implementation signals and open-source adoption",
            queries=[f"{core} {agent_terms}", f"{core} framework tools"],
            limit=8,
        ),
        SourceStrategyItem(
            name="docs_blogs_workplace",
            retriever="docs_blogs",
            purpose="adoption signals and workplace-relevant directions",
            queries=[f"{core} {workplace_terms}", f"{core} human AI collaboration"],
            limit=8,
        ),
        SourceStrategyItem(
            name="social_trend_signals",
            retriever="twitter",
            purpose="public social trend signals",
            queries=[f"{core} workplace AI agents", f"{core} agentic AI"],
            limit=6,
        ),
        SourceStrategyItem(
            name="contrarian_limitations",
            retriever="web",
            purpose="contradictory evidence, limitations, and risks",
            queries=[f"{core} {contrarian_terms}", f"{core} failure mode"],
            limit=8,
        ),
        SourceStrategyItem(
            name="prior_artifact_memory",
            retriever="memory",
            purpose="prior related harness artifacts",
            queries=[core],
            limit=6,
        ),
    ]


def _single_retriever_strategy(goal: str, plan: ResearchPlan, retriever: str) -> list[SourceStrategyItem]:
    mixed = _mixed_source_strategy(goal, plan)
    selected = [item for item in mixed if item.retriever == retriever]
    if selected:
        return selected
    concepts = _goal_terms(goal)
    core = " ".join(concepts[:6]) or goal
    return [
        SourceStrategyItem(
            name=f"{retriever}_general",
            retriever=retriever,
            purpose=angle,
            queries=[f"{core} {angle}", f"{core} survey benchmark limitations"],
            limit=8,
        )
        for angle in plan.search_angles
    ]


def _goal_terms(goal: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9]+", goal.lower())
    return [word for word in words if word not in RUN_SLUG_STOPWORDS]
