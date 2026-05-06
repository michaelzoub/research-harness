from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from .agents import (
    AgentResult,
    CriticAgent,
    HarnessDebuggerAgent,
    HypothesisAgent,
    LiteratureAgent,
    SynthesisAgent,
)
from .llm import LLMClient
from .loops import EvaluatorRegistry, EvolutionaryOuterLoop, TaskRouter
from .run_benchmarks import write_run_benchmarks
from .schemas import AgentBudget, LoopIteration, LoopTask, ResearchPlan, RunRecord, SourceStrategyItem, TaskType, now_iso, to_dict
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
    mode: str = "evolutionary"
    retriever: str = "auto"
    search_agent_count: int = 7
    hypothesis_agent_count: int = 2
    include_debugger: bool = True
    max_loop_iterations: int = 12
    max_task_attempts: int = 2
    task_mode: str = "auto"
    evaluator_name: Optional[str] = None
    evolution_population_size: int = 4
    llm_provider: str = "auto"
    llm_model: str = "gpt-4.1-mini"
    echo_progress: bool = True
    default_budget: AgentBudget = field(default_factory=AgentBudget)


class Orchestrator:
    def __init__(self, corpus_path: Path, output_root: Path, config: Optional[HarnessConfig] = None):
        self.corpus_path = corpus_path
        self.output_root = output_root
        self.config = config or HarnessConfig()
        self.evaluator_registry = EvaluatorRegistry()
        self.llm = LLMClient(provider=self.config.llm_provider, model=self.config.llm_model)
        if self.config.llm_provider == "openai" and not self.llm.is_live:
            raise ValueError("--llm-provider openai requires OPENAI_API_KEY.")

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
        store = ArtifactStore(self.output_root / run.id, echo_progress=self.config.echo_progress)
        store.add_run(run)
        store.append_progress(f"Starting run {run.id}")
        store.append_progress(f"Execution mode: {selected_mode}")
        store.append_progress(f"Goal: {goal}")
        self._write_prd(store, run, plan, source_strategy, selected_mode, stage="planned")
        store.append_progress(f"PRD: {store.prd_path}")
        try:
            if selected_mode == "deterministic":
                await self._run_phase1(run, store, plan, source_strategy)
            elif selected_mode in {"standard", "fanout"}:
                await self._run_phase2(run, store, plan, source_strategy)
            elif selected_mode in {"evolutionary", "loop"}:
                await self._run_loop(run, store, plan, source_strategy)
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
            self._write_prd(store, run, plan, source_strategy, selected_mode, stage="completed")
            write_run_benchmarks(store)
        return run, store

    def _write_prd(
        self,
        store: ArtifactStore,
        run: RunRecord,
        plan: ResearchPlan,
        source_strategy: list[SourceStrategyItem],
        selected_mode: str,
        stage: str,
    ) -> None:
        loop_tasks = store.list("loop_tasks")
        if loop_tasks:
            tasks = [_prd_task_from_loop_task(task, index) for index, task in enumerate(loop_tasks, start=1)]
        else:
            tasks = self._standard_prd_tasks(selected_mode, plan, source_strategy, store.list("agent_traces"))
        payload = {
            "schema_version": "research_harness_prd_v1",
            "stage": stage,
            "run": to_dict(run),
            "execution_mode": selected_mode,
            "task_mode": run.task_mode,
            "goal": run.user_goal,
            "plan": to_dict(plan),
            "source_strategy": [to_dict(item) for item in source_strategy],
            "organized_tasks": tasks,
            "artifacts": {
                "final_report": str(store.report_path),
                "progress": str(store.progress_path),
                "run_benchmark": str(store.run_benchmark_path),
                "decision_dag": str(store.decision_dag_path),
                "trace": str(store.trace_log_path),
            },
            "notes": [
                "This file is the run-local PRD/task map.",
                "Use organized_tasks for the ordered work plan, status, dependencies, and acceptance criteria.",
            ],
        }
        store.write_prd(payload)

    def _standard_prd_tasks(
        self,
        selected_mode: str,
        plan: ResearchPlan,
        source_strategy: list[SourceStrategyItem],
        traces: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        tasks: list[dict[str, object]] = []
        if selected_mode == "deterministic":
            first = source_strategy[0]
            tasks.append(
                _prd_task(
                    1,
                    "Search initial evidence",
                    "search",
                    {"retriever": first.retriever, "purpose": first.purpose, "queries": first.queries},
                    ["Retrieve sources", "Extract traceable claims"],
                    [],
                    traces,
                    "search_literature",
                )
            )
            tasks.append(
                _prd_task(2, "Critique evidence", "critique", {}, ["Review claims", "Record contradictions or questions"], ["prd_task_001"], traces, "critic_reviewer")
            )
            tasks.append(
                _prd_task(3, "Synthesize report", "synthesize", {}, ["Write final_report.md"], ["prd_task_002"], traces, "synthesis_agent")
            )
            return tasks

        for index, item in enumerate(source_strategy[: self.config.search_agent_count], start=1):
            tasks.append(
                _prd_task(
                    index,
                    f"Search {item.purpose}",
                    "search",
                    {"retriever": item.retriever, "purpose": item.purpose, "queries": item.queries},
                    ["Retrieve sources", "Extract claims", "Record failed paths if evidence is unavailable"],
                    [],
                    traces,
                    "search_literature",
                )
            )
        offset = len(tasks)
        for index, angle in enumerate(plan.hypothesis_angles[: self.config.hypothesis_agent_count], start=offset + 1):
            tasks.append(
                _prd_task(
                    index,
                    f"Generate hypotheses for {angle}",
                    "hypothesize",
                    {"hypothesis_angle": angle},
                    ["Inspect claims", "Create hypotheses or open questions"],
                    [task["id"] for task in tasks if task["kind"] == "search"],
                    traces,
                    "hypothesis_generation",
                )
            )
        tasks.append(
            _prd_task(
                len(tasks) + 1,
                "Critique claims and hypotheses",
                "critique",
                {},
                ["Review claims", "Record contradictions and open questions"],
                [task["id"] for task in tasks],
                traces,
                "critic_reviewer",
            )
        )
        tasks.append(
            _prd_task(
                len(tasks) + 1,
                "Synthesize final report",
                "synthesize",
                {},
                ["Write final_report.md", "Summarize sources, claims, hypotheses, and caveats"],
                [tasks[-1]["id"]],
                traces,
                "synthesis_agent",
            )
        )
        if self.config.include_debugger:
            tasks.append(
                _prd_task(
                    len(tasks) + 1,
                    "Inspect harness behavior",
                    "debug_harness",
                    {},
                    ["Record constrained harness-change proposal"],
                    [tasks[-1]["id"]],
                    traces,
                    "harness_debugger",
                )
            )
        return tasks

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
            llm=self.llm,
        )
        critic = CriticAgent(
            name="deterministic_critic",
            role="critic_reviewer",
            prompt_template=_prompt("critic_agent"),
            budget=self._budget("append_only"),
            llm=self.llm,
        )
        synth = SynthesisAgent(
            name="deterministic_synthesis",
            role="synthesis_agent",
            prompt_template=_prompt("synthesis_agent"),
            budget=self._budget("append_only"),
            llm=self.llm,
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
                llm=self.llm,
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
                llm=self.llm,
            )
            for index, angle in enumerate(plan.hypothesis_angles[: self.config.hypothesis_agent_count])
        ]
        await asyncio.gather(*(agent.execute(run, store) for agent in hypothesis_agents))

        critic = CriticAgent(
            name="critic_reviewer",
            role="critic_reviewer",
            prompt_template=_prompt("critic_agent"),
            budget=self._budget("append_only"),
            llm=self.llm,
        )
        synthesis = SynthesisAgent(
            name="synthesis_agent",
            role="synthesis_agent",
            prompt_template=_prompt("synthesis_agent"),
            budget=self._budget("append_only"),
            llm=self.llm,
        )
        await critic.execute(run, store)
        await synthesis.execute(run, store)

        if self.config.include_debugger:
            debugger = HarnessDebuggerAgent(
                name="harness_debugger",
                role="harness_debugger",
                prompt_template=_prompt("harness_debugger"),
                budget=self._budget("append_only"),
                llm=self.llm,
            )
            await debugger.execute(run, store)

    async def _run_loop(
        self, run: RunRecord, store: ArtifactStore, plan: ResearchPlan, source_strategy: list[SourceStrategyItem]
    ) -> None:
        router = TaskRouter(self.evaluator_registry)
        decision = router.decide(run.user_goal, self.config.task_mode, self.config.evaluator_name)
        store.add_task_ingestion_decision(decision)
        run.task_mode = decision.selected_mode
        store.update_run(run)

        tasks = self.create_evolution_tasks(decision.selected_mode)
        for task in tasks:
            store.add_loop_task(task)
        store.append_progress(f"# Progress for {run.id}")
        store.append_progress(f"Prompt: {run.user_goal}")
        store.append_progress(f"Task ingestion: {decision.selected_mode} mode. {decision.reason}")

        await self._pass_task(
            run,
            store,
            tasks[0],
            1,
            "task_router",
            f"Selected {decision.selected_mode} mode. {decision.reason}",
        )

        tasks[1].status = "running"
        tasks[1].attempts += 1
        store.update_loop_task(tasks[1])
        outer_loop = EvolutionaryOuterLoop(
            run_id=run.id,
            goal=run.user_goal,
            task_mode=decision.selected_mode,
            source_strategy=source_strategy,
            search_factory=self._retriever_for,
            evaluator=self.evaluator_registry.get(decision.evaluator_name),
            max_outer_iterations=self.config.max_loop_iterations,
            population_size=self.config.evolution_population_size,
            llm=self.llm,
        )
        await outer_loop.run(store)
        await self._pass_task(
            run,
            store,
            tasks[1],
            2,
            "outer_orchestrator",
            f"Completed {len(store.list('evolution_rounds'))} evolutionary outer rounds.",
        )

        iteration = 3
        task_cursor = 2
        if decision.selected_mode == "research":
            hypothesis = HypothesisAgent(
                name="loop_research_hypothesis",
                hypothesis_angle=plan.hypothesis_angles[0],
                prompt_template=_prompt("hypothesis_agent"),
                budget=self._budget("append_only"),
                llm=self.llm,
            )
            result = await hypothesis.execute(run, store)
            await self._record_task_result(run, store, tasks[task_cursor], iteration, result)
            iteration += 1
            task_cursor += 1

        critic = CriticAgent(
            name="loop_evolution_critic",
            role="critic_reviewer",
            prompt_template=_prompt("critic_agent"),
            budget=self._budget("append_only"),
            llm=self.llm,
        )
        result = await critic.execute(run, store)
        await self._record_task_result(run, store, tasks[task_cursor], iteration, result)
        iteration += 1
        task_cursor += 1

        synthesis = SynthesisAgent(
            name="loop_evolution_synthesis",
            role="synthesis_agent",
            prompt_template=_prompt("synthesis_agent"),
            budget=self._budget("append_only"),
            llm=self.llm,
        )
        result = await synthesis.execute(run, store)
        await self._record_task_result(run, store, tasks[task_cursor], iteration, result)
        iteration += 1
        task_cursor += 1

        if self.config.include_debugger:
            debugger = HarnessDebuggerAgent(
                name="loop_evolution_debugger",
                role="harness_debugger",
                prompt_template=_prompt("harness_debugger"),
                budget=self._budget("append_only"),
                llm=self.llm,
            )
            result = await debugger.execute(run, store)
            await self._record_task_result(run, store, tasks[task_cursor], iteration, result)

        remaining = [row for row in store.list("loop_tasks") if not row.get("passes")]
        if remaining:
            store.append_progress(f"Stopped with {len(remaining)} incomplete loop tasks.")
        else:
            store.append_progress("<promise>COMPLETE</promise>")

    def create_evolution_tasks(self, selected_mode: str) -> list[LoopTask]:
        tasks = [
            LoopTask(
                title="Ingest prompt and select task mode",
                action="debug_harness",
                priority=1,
                params={"selected_mode": selected_mode},
                acceptance_criteria=[
                    "Explicit flags, evaluator availability, and prompt heuristics were checked",
                    "A task ingestion decision was persisted",
                ],
            ),
            LoopTask(
                title="Run outer evolutionary orchestrator",
                action="debug_harness",
                priority=2,
                params={"selected_mode": selected_mode},
                acceptance_criteria=[
                    "The outer loop proposed code or query variants",
                    "The selected inner loop returned ranked variants with scalar scores",
                    "Plateau or threshold stopping signals were evaluated",
                ],
            ),
        ]
        if selected_mode == "research":
            tasks.append(
                LoopTask(
                    title="Generate hypotheses from winning evidence",
                    action="hypothesize",
                    priority=3,
                    params={},
                    acceptance_criteria=["Claims from research variants were inspected", "Hypotheses or open questions were created"],
                )
            )
        tasks.extend(
            [
                LoopTask(
                    title="Critique ranked variants and claims",
                    action="critique",
                    priority=4,
                    params={},
                    acceptance_criteria=["Artifacts were reviewed for contradictions, weak evidence, or missing checks"],
                ),
                LoopTask(
                    title="Synthesize evolutionary run report",
                    action="synthesize",
                    priority=5,
                    params={},
                    acceptance_criteria=["Final report was written from run artifacts"],
                ),
            ]
        )
        if self.config.include_debugger:
            tasks.append(
                LoopTask(
                    title="Inspect harness behavior and propose improvements",
                    action="debug_harness",
                    priority=6,
                    params={},
                    acceptance_criteria=["A constrained harness-change proposal was recorded"],
                )
            )
        return tasks

    async def _record_task_result(
        self, run: RunRecord, store: ArtifactStore, task: LoopTask, iteration: int, result: AgentResult
    ) -> None:
        if result.errors:
            task.status = "failed"
            task.passes = False
            task.last_error = "; ".join(result.errors)
        else:
            task.status = "passed"
            task.passes = True
            task.completed_at = now_iso()
        task.attempts += 1
        task.result_summary = result.summary
        store.update_loop_task(task)
        store.add_loop_iteration(
            LoopIteration(
                run_id=run.id,
                iteration=iteration,
                task_id=task.id,
                task_title=task.title,
                agent_name=result.agent_name,
                status=task.status,
                summary=result.summary,
                errors=result.errors,
                completed_at=now_iso(),
            )
        )
        store.append_progress(f"Task {iteration}: {task.status} - {result.summary}")

    async def _pass_task(
        self,
        run: RunRecord,
        store: ArtifactStore,
        task: LoopTask,
        iteration: int,
        agent_name: str,
        summary: str,
    ) -> None:
        await self._record_task_result(run, store, task, iteration, AgentResult(agent_name, summary, []))

    def create_loop_tasks(self, plan: ResearchPlan, source_strategy: list[SourceStrategyItem]) -> list[LoopTask]:
        tasks: list[LoopTask] = []
        for index, item in enumerate(source_strategy[: self.config.search_agent_count], start=1):
            tasks.append(
                LoopTask(
                    title=f"Search {item.purpose}",
                    action="search",
                    priority=index,
                    params={
                        "query_angle": item.purpose,
                        "retriever": item.retriever,
                        "queries": item.queries,
                    },
                    acceptance_criteria=[
                        "Retriever was called for the assigned angle",
                        "Relevant sources and claims were written when evidence exists",
                        "Failure paths were recorded when no evidence exists",
                    ],
                )
            )
        base_priority = len(tasks) + 1
        for index, angle in enumerate(plan.hypothesis_angles[: self.config.hypothesis_agent_count], start=base_priority):
            tasks.append(
                LoopTask(
                    title=f"Generate hypotheses for {angle}",
                    action="hypothesize",
                    priority=index,
                    params={"hypothesis_angle": angle},
                    acceptance_criteria=[
                        "Existing claims were inspected",
                        "Hypotheses or open questions were created",
                        "Next experiments were captured for testable hypotheses",
                    ],
                )
            )
        tasks.extend(
            [
                LoopTask(
                    title="Critique claims and hypotheses",
                    action="critique",
                    priority=len(tasks) + 1,
                    params={},
                    acceptance_criteria=[
                        "Claims and hypotheses were reviewed",
                        "Contradictions or open questions were recorded when found",
                    ],
                ),
                LoopTask(
                    title="Synthesize final report",
                    action="synthesize",
                    priority=len(tasks) + 2,
                    params={},
                    acceptance_criteria=[
                        "Final report was written",
                        "Sources, claims, hypotheses, contradictions, and questions were summarized",
                    ],
                ),
            ]
        )
        if self.config.include_debugger:
            tasks.append(
                LoopTask(
                    title="Inspect harness behavior and propose improvements",
                    action="debug_harness",
                    priority=len(tasks) + 1,
                    params={},
                    acceptance_criteria=[
                        "Run traces and artifact yield were inspected",
                        "A constrained harness-change proposal was recorded",
                    ],
                )
            )
        return tasks

    def _next_loop_task(self, store: ArtifactStore) -> Optional[LoopTask]:
        candidates = []
        for row in store.list("loop_tasks"):
            if row.get("passes"):
                continue
            if row.get("status") == "failed" and int(row.get("attempts", 0)) >= self.config.max_task_attempts:
                continue
            candidates.append(LoopTask(**row))
        if not candidates:
            return None
        return sorted(candidates, key=lambda task: (task.priority, task.created_at))[0]

    async def _execute_loop_task(self, run: RunRecord, store: ArtifactStore, task: LoopTask):
        if task.action == "search":
            agent = LiteratureAgent(
                name=f"loop_search_{task.id}",
                query_angle=str(task.params["query_angle"]),
                corpus=self._retriever_for(str(task.params["retriever"])),
                prompt_template=_prompt("literature_agent"),
                budget=self._budget("upsert_by_url"),
                search_queries=[str(query) for query in task.params.get("queries", [])],
                llm=self.llm,
            )
        elif task.action == "hypothesize":
            agent = HypothesisAgent(
                name=f"loop_hypothesis_{task.id}",
                hypothesis_angle=str(task.params["hypothesis_angle"]),
                prompt_template=_prompt("hypothesis_agent"),
                budget=self._budget("append_only"),
                llm=self.llm,
            )
        elif task.action == "critique":
            agent = CriticAgent(
                name=f"loop_critic_{task.id}",
                role="critic_reviewer",
                prompt_template=_prompt("critic_agent"),
                budget=self._budget("append_only"),
                llm=self.llm,
            )
        elif task.action == "synthesize":
            agent = SynthesisAgent(
                name=f"loop_synthesis_{task.id}",
                role="synthesis_agent",
                prompt_template=_prompt("synthesis_agent"),
                budget=self._budget("append_only"),
                llm=self.llm,
            )
        elif task.action == "debug_harness":
            agent = HarnessDebuggerAgent(
                name=f"loop_debugger_{task.id}",
                role="harness_debugger",
                prompt_template=_prompt("harness_debugger"),
                budget=self._budget("append_only"),
                llm=self.llm,
            )
        else:  # pragma: no cover - guarded by LoopTaskAction literal
            raise ValueError(f"Unknown loop action: {task.action}")
        return await agent.execute(run, store)

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


def _prd_task_from_loop_task(task: dict[str, object], index: int) -> dict[str, object]:
    priority = int(task.get("priority", index) or index)
    dependencies = [] if priority <= 1 else [f"prd_task_{priority - 1:03d}"]
    return {
        "id": f"prd_task_{priority:03d}",
        "source_task_id": task.get("id"),
        "title": task.get("title"),
        "kind": task.get("action"),
        "priority": priority,
        "status": task.get("status"),
        "passes": bool(task.get("passes")),
        "attempts": int(task.get("attempts", 0) or 0),
        "dependencies": dependencies,
        "params": task.get("params", {}),
        "acceptance_criteria": task.get("acceptance_criteria", []),
        "result_summary": task.get("result_summary"),
        "last_error": task.get("last_error"),
    }


def _prd_task(
    index: int,
    title: str,
    kind: str,
    params: dict[str, object],
    acceptance_criteria: list[str],
    dependencies: list[str],
    traces: list[dict[str, object]],
    role: str,
) -> dict[str, object]:
    matching = [trace for trace in traces if trace.get("role") == role]
    failed = [trace for trace in matching if trace.get("status") != "completed"]
    status = "pending"
    if matching:
        status = "failed" if failed else "passed"
    return {
        "id": f"prd_task_{index:03d}",
        "title": title,
        "kind": kind,
        "priority": index,
        "status": status,
        "passes": bool(matching and not failed),
        "attempts": len(matching),
        "dependencies": dependencies,
        "params": params,
        "acceptance_criteria": acceptance_criteria,
        "result_summary": matching[-1].get("output_summary") if matching else None,
        "last_error": "; ".join(str(error) for trace in failed for error in trace.get("errors", [])) or None,
        "trace_ids": [trace.get("id") for trace in matching],
    }
