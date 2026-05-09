from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from .orchestrator import HarnessConfig, Orchestrator
from .schemas import now_iso
from .store import ArtifactStore


GraderType = Literal["code", "model", "human"]
AggregationMode = Literal["weighted", "binary", "hybrid"]
TrajectoryMatchMode = Literal["strict", "unordered", "subset", "superset"]


@dataclass
class EvalTask:
    """A single problem/test case with fixed inputs and success criteria."""

    id: str
    name: str
    prompt: str
    task_mode: str
    success_criteria: list[str]
    evaluator_name: Optional[str] = None
    retriever: str = "local"
    max_iterations: int = 1
    grader_ids: list[str] = field(default_factory=list)
    aggregation: AggregationMode = "hybrid"
    threshold: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)
    trials: Optional[int] = None


@dataclass
class EvalTrial:
    """One attempt at one eval task."""

    task_id: str
    trial_index: int
    run_id: str
    transcript_path: str
    trajectory_graph_path: str
    isolation: dict[str, Any]
    outcome: dict[str, Any]
    grader_results: list[dict[str, Any]]
    aggregate_score: float
    passed: bool
    completed_at: str = field(default_factory=now_iso)


@dataclass
class EvalSuite:
    """A collection of tasks that measure related harness behavior."""

    id: str
    name: str
    description: str
    tasks: list[EvalTask]
    trials_per_task: int = 1


@dataclass
class EvalRunSummary:
    suite_id: str
    suite_name: str
    trials_per_task: int
    started_at: str
    completed_at: str
    task_count: int
    trial_count: int
    passed_trials: int
    aggregate_score: float
    trials: list[dict[str, Any]]


@dataclass
class GraderResult:
    grader_id: str
    grader_type: GraderType
    method: str
    score: float
    passed: bool
    weight: float
    summary: str
    assertions: list[dict[str, Any]]


@dataclass
class Grader:
    id: str
    grader_type: GraderType
    method: str
    weight: float
    threshold: float
    fn: Callable[[EvalTask, ArtifactStore], GraderResult]

    def grade(self, task: EvalTask, store: ArtifactStore) -> GraderResult:
        return self.fn(task, store)


class EvaluationHarness:
    """Runs eval tasks end-to-end, records transcripts, grades outcomes, and aggregates results."""

    def __init__(
        self,
        *,
        corpus_path: Path = Path("examples/corpus/research_corpus.json"),
        output_root: Path = Path("eval_outputs"),
    ) -> None:
        self.corpus_path = corpus_path
        self.output_root = output_root
        self.grader_registry = default_graders()

    async def run_suite(self, suite: EvalSuite) -> EvalRunSummary:
        started_at = now_iso()
        self._prepare_eval_root()
        trials: list[EvalTrial] = []
        for task in suite.tasks:
            task_trials = task.trials or suite.trials_per_task
            for trial_index in range(1, task_trials + 1):
                trials.append(await self._run_trial(task, trial_index))
        self._apply_cross_trial_graders(suite, trials)
        passed_trials = sum(1 for trial in trials if trial.passed)
        aggregate_score = sum(trial.aggregate_score for trial in trials) / max(len(trials), 1)
        summary = EvalRunSummary(
            suite_id=suite.id,
            suite_name=suite.name,
            trials_per_task=suite.trials_per_task,
            started_at=started_at,
            completed_at=now_iso(),
            task_count=len(suite.tasks),
            trial_count=len(trials),
            passed_trials=passed_trials,
            aggregate_score=round(aggregate_score, 3),
            trials=[asdict(trial) for trial in trials],
        )
        (self.output_root / f"{suite.id}_summary.json").write_text(
            json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary

    def _apply_cross_trial_graders(self, suite: EvalSuite, trials: list[EvalTrial]) -> None:
        tasks_by_id = {task.id: task for task in suite.tasks}
        for task in suite.tasks:
            if "parallel_trial_isolation" not in task.grader_ids:
                continue
            matching = [trial for trial in trials if trial.task_id == task.id]
            result = _grade_parallel_trial_isolation_from_trials(task, matching)
            for trial in matching:
                trial.grader_results.append(asdict(result))
                aggregate_score, passed = aggregate_results(
                    tasks_by_id[trial.task_id],
                    [GraderResult(**grader) for grader in trial.grader_results],
                )
                trial.aggregate_score = aggregate_score
                trial.passed = passed

    async def _run_trial(self, task: EvalTask, trial_index: int) -> EvalTrial:
        trial_root = self._prepare_trial_root(task, trial_index)
        # Run artifacts land directly in trial_root (mirroring the outputs/ folder layout).
        trial_output_root = trial_root
        trial_tmp = trial_root / "tmp"
        trial_tmp.mkdir(parents=True, exist_ok=True)
        config = HarnessConfig(
            retriever=task.retriever,
            max_loop_iterations=task.max_iterations,
            task_mode=task.task_mode,
            evaluator_name=task.evaluator_name,
            include_debugger=False,
            echo_progress=False,
            llm_provider="local",
        )
        orchestrator = Orchestrator(self.corpus_path, trial_output_root, config)
        previous_tmpdir = os.environ.get("TMPDIR")
        os.environ["TMPDIR"] = str(trial_tmp)
        run, store = await orchestrator.run(task.prompt)
        if previous_tmpdir is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = previous_tmpdir
        graph_paths = write_trajectory_graph_artifacts(store, trial_root)
        grader_results = [
            self.grader_registry[grader_id].grade(task, store)
            for grader_id in task.grader_ids
            if grader_id != "parallel_trial_isolation"
        ]
        aggregate_score, passed = aggregate_results(task, grader_results)
        return EvalTrial(
            task_id=task.id,
            trial_index=trial_index,
            run_id=run.id,
            transcript_path=str(store.trace_log_path),
            trajectory_graph_path=str(graph_paths["svg"]),
            isolation={
                "trial_root": str(trial_root),
                "output_root": str(trial_output_root),
                "tmpdir": str(trial_tmp),
                "clean_start": True,
                "production_agent_path": "research_harness.orchestrator.Orchestrator",
                "shared_state_policy": "No shared output directories between trials; local corpus is read-only; TMPDIR is per-trial.",
            },
            outcome=outcome_from_store(store),
            grader_results=[asdict(result) for result in grader_results],
            aggregate_score=aggregate_score,
            passed=passed,
        )

    def _prepare_eval_root(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _prepare_trial_root(self, task: EvalTask, trial_index: int) -> Path:
        # eval_outputs/<task_id>/trial_001/ — mirrors outputs/ structure per task.
        trial_root = self.output_root / task.id / f"trial_{trial_index:03d}"
        if trial_root.exists():
            shutil.rmtree(trial_root)
        trial_root.mkdir(parents=True, exist_ok=True)
        return trial_root


def default_eval_suite() -> EvalSuite:
    return EvalSuite(
        id="core",
        name="Core Harness Evaluation Suite",
        description="Prewritten evals covering open research, direct optimization, optimize-query, and challenge runs.",
        tasks=[
            EvalTask(
                id="research_open_ended",
                name="Open-ended research produces grounded artifacts",
                prompt="Research how multi-agent systems improve automated literature review quality",
                task_mode="research",
                success_criteria=[
                    "Run completes",
                    "Report is written",
                    "Claims cite source IDs",
                    "Research retrieves enough sources and claims",
                    "Trial starts from an isolated clean artifact directory",
                ],
                grader_ids=[
                    "outcome_completed",
                    "prd_tasks_executed",
                    "prd_tasks_executed_deterministic",
                    "research_groundedness",
                    "report_no_fabricated_sources",
                    "artifact_report",
                    "transcript_progress",
                    "isolation_clean_trial",
                    "model_report_rubric",
                    "llm_research_quality_challenger",
                    "llm_hypothesis_novelty_challenger",
                    "llm_open_ended_judgment_challenger",
                ],
                aggregation="hybrid",
                threshold=0.8,
            ),
            EvalTask(
                id="optimize_direct",
                name="Direct optimization uses deterministic evaluator",
                prompt="Optimize a tiny scoring function",
                task_mode="optimize",
                evaluator_name="length_score",
                success_criteria=[
                    "Run routes to optimize",
                    "Only optimize evaluations are created",
                    "Best deterministic score is positive",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "prd_tasks_executed",
                    "optimize_score",
                    "optimization_code_artifact",
                    "transcript_progress",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
            ),
            EvalTask(
                id="optimize_query_seeded",
                name="Optimize-query produces seed context and optimizer scores",
                prompt="Research optimization strategies for a tiny scoring benchmark",
                task_mode="optimize_query",
                evaluator_name="length_score",
                success_criteria=[
                    "Query phase runs",
                    "Optimizer seed context is written",
                    "Optimizer phase scores variants",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "prd_tasks_executed",
                    "seed_context",
                    "optimize_query_phases",
                    "optimization_code_artifact",
                    "transcript_progress",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
            ),
            EvalTask(
                id="challenge_prediction_market",
                name="Prediction-market challenge emits solution and proxy score",
                prompt=(
                    "Get to $10 profit in the prediction market challenge, don't stop until you're profitable. "
                    "Introduce entropy from AMM, prediction-market, and options literature before tweaking hyperparameters."
                ),
                task_mode="optimize_query",
                evaluator_name="prediction_market",
                success_criteria=[
                    "Query phase uses challenge context",
                    "Optimizer phase scores against local proxy evaluator",
                    "solution.py is emitted for upstream evaluation",
                    "Official profit must be measured by upstream prediction-market evaluator",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "prd_tasks_executed",
                    "seed_context",
                    "optimization_code_artifact",
                    "prediction_market_solution",
                    "prediction_market_proxy_score",
                    "transcript_progress",
                    "isolation_clean_trial",
                ],
                aggregation="hybrid",
                threshold=0.8,
            ),
        ],
    )


def edge_eval_suite() -> EvalSuite:
    return EvalSuite(
        id="edge",
        name="Edge-Case Harness Evaluation Suite",
        description=(
            "Regression evals for ambiguous or failure-prone agent-harness behavior: missing evaluators, "
            "prediction-market scorer status, trajectory shape, artifact containment, and trial isolation."
        ),
        tasks=[
            EvalTask(
                id="optimize_query_missing_evaluator_skips_optimizer",
                name="Optimize-query without evaluator records skip instead of fabricating optimization",
                prompt="Research optimization strategies for a tiny benchmark, but do not assume an evaluator exists",
                task_mode="optimize_query",
                success_criteria=[
                    "Run stays in optimize_query mode",
                    "Seed context records that no evaluator is available",
                    "Optimizer phase is skipped cleanly",
                    "No optimization artifacts are fabricated",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "seed_context",
                    "optimizer_skipped_without_evaluator",
                    "trajectory_modes",
                    "transcript_progress",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
                metadata={"required_modes": ["optimize_query"], "forbidden_modes": ["optimize"]},
            ),
            EvalTask(
                id="prediction_market_outputs_are_contained",
                name="Prediction-market generated strategies stay under run outputs",
                prompt=(
                    "Get to $10 profit in the prediction market challenge. Keep every generated strategy as a run artifact, "
                    "never as a source file."
                ),
                task_mode="optimize_query",
                evaluator_name="prediction_market",
                success_criteria=[
                    "Generated candidate strategies are inside outputs/<run>/candidates",
                    "No temporary strategy files are written into the repository source tree",
                    "The selected candidate is promoted to optimal_code.py",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "optimization_code_artifact",
                    "prediction_market_solution",
                    "prediction_market_artifact_containment",
                    "trajectory_modes",
                    "transcript_progress",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
                metadata={"required_modes": ["optimize_query", "optimize"], "candidate_glob": "candidates/*.py"},
            ),
            EvalTask(
                id="prediction_market_unmeasured_official_status",
                name="Prediction-market fallback scoring marks official score unmeasured",
                prompt="Evaluate a prediction-market challenge strategy without requiring the upstream scorer",
                task_mode="optimize_query",
                evaluator_name="prediction_market",
                success_criteria=[
                    "Fallback scoring is allowed for local fast evals",
                    "optimization_result.json does not pretend fallback score is official",
                    "score_source and candidate_path are recorded",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "prediction_market_official_status",
                    "prediction_market_proxy_score",
                    "optimization_code_artifact",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
            ),
            EvalTask(
                id="challenge_prediction_market_official_unavailable_records_unmeasured",
                name="Prediction-market unavailable official grader records unmeasured result",
                prompt="Run the prediction-market challenge locally when the upstream official scorer is not required",
                task_mode="optimize_query",
                evaluator_name="prediction_market",
                success_criteria=[
                    "The local fallback scorer may run",
                    "official_result.measured is false unless the upstream scorer actually ran",
                    "score_source, candidate_path, and optimization result are present",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "prediction_market_official_status",
                    "optimization_code_artifact",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
            ),
            EvalTask(
                id="challenge_prediction_market_candidate_files_only_in_outputs",
                name="Prediction-market candidate files are only output artifacts",
                prompt="Generate prediction-market challenge candidates, keeping every candidate inside the run output directory",
                task_mode="optimize_query",
                evaluator_name="prediction_market",
                success_criteria=[
                    "Candidate Python files are written to outputs/<run>/candidates",
                    "The winning candidate path points into that candidates directory",
                    "No generated candidate strategy is written into repository source locations",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "optimization_code_artifact",
                    "prediction_market_candidate_files_only_in_outputs",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
            ),
            EvalTask(
                id="parallel_trials_do_not_share_tmp_or_outputs",
                name="Multiple trials use distinct temp and output roots",
                prompt="Research a small deterministic fact about agent evaluation harnesses",
                task_mode="research",
                max_iterations=1,
                trials=2,
                success_criteria=[
                    "Each trial has a unique trial root",
                    "Each trial has a unique output root",
                    "Each trial has a unique TMPDIR",
                    "Each trial has exactly one run artifact directory",
                ],
                grader_ids=[
                    "outcome_completed",
                    "transcript_progress",
                    "isolation_clean_trial",
                    "parallel_trial_isolation",
                ],
                aggregation="binary",
            ),
            EvalTask(
                id="challenge_prediction_market_no_repo_root_strategy_files",
                name="Prediction-market run does not leak strategy files into repo root",
                prompt="Optimize the prediction-market challenge without creating temporary strategy files in the repository root",
                task_mode="optimize_query",
                evaluator_name="prediction_market",
                success_criteria=[
                    "No pm_strategy*.py files exist in the repository root",
                    "No tmp_pm*.py files exist in the repository root",
                    "Generated strategies remain run artifacts",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "optimization_code_artifact",
                    "no_repo_root_strategy_files",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
            ),
            EvalTask(
                id="research_should_not_oversearch",
                name="Simple research task stays inside a bounded search budget",
                prompt="Research who founded Apple and answer with a concise, sourced summary",
                task_mode="research",
                retriever="local",
                max_iterations=1,
                success_criteria=[
                    "The run completes with a report",
                    "The harness does not fan out unnecessary extra research rounds",
                    "Source, claim, and variant counts stay under the task budget",
                ],
                grader_ids=[
                    "outcome_completed",
                    "artifact_report",
                    "research_search_budget",
                    "transcript_progress",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
                metadata={
                    "max_sources": 8,
                    "max_claims": 24,
                    "max_query_evaluations": 4,
                    "max_evolution_rounds": 1,
                },
            ),
            EvalTask(
                id="nested_loop_multiple_iterations_no_regression",
                name="Nested optimization loop runs multiple rounds without score collapse",
                prompt="Optimize a tiny scoring function across multiple loop rounds and preserve the best candidate",
                task_mode="optimize",
                evaluator_name="length_score",
                max_iterations=4,
                success_criteria=[
                    "The optimizer runs multiple outer loop rounds",
                    "Round scores do not collapse after iteration",
                    "The selected output artifact is still emitted",
                    "A trajectory graph is written for inspection",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "multi_iteration_loop",
                    "loop_no_score_regression",
                    "optimization_code_artifact",
                    "trajectory_graph_artifact",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
                metadata={"min_rounds": 3, "max_score_drop": 0.2},
            ),
            EvalTask(
                id="trajectory_match_modes_are_enforced",
                name="Trajectory evaluators enforce strict, unordered, subset, and superset modes",
                prompt="Optimize a tiny scoring function across multiple rounds so the harness records a trajectory",
                task_mode="optimize",
                evaluator_name="length_score",
                max_iterations=4,
                success_criteria=[
                    "Normalized trajectory events are extracted from artifacts",
                    "Strict matching validates the expected canonical loop prefix",
                    "Unordered, subset, and superset matching all run against the same trajectory",
                    "Graph trajectory edges match the expected harness flow",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "trajectory_match_modes",
                    "graph_trajectory_match",
                    "trajectory_graph_artifact",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
                metadata={
                    "reference_trajectory": [
                        {"type": "router", "name": "optimize"},
                        {"type": "outer_loop", "name": "optimize"},
                        {"type": "inner_loop", "name": "optimize"},
                        {"type": "selection", "name": "variant"},
                        {"type": "outcome", "name": "completed"},
                    ],
                    "required_graph_edges": [
                        ["prompt", "router"],
                        ["router", "outer"],
                        ["outer", "inner"],
                        ["inner", "select"],
                        ["select", "agents"],
                        ["agents", "outcome"],
                    ],
                },
            ),
            EvalTask(
                id="stuck_loop_triggers_literature_search",
                name="Plateaued optimization triggers literature refresh",
                prompt=(
                    "Optimize a tiny scoring function. If the loop gets stuck or plateaus, check existing literature "
                    "before continuing to tweak variants."
                ),
                task_mode="optimize",
                evaluator_name="length_score",
                max_iterations=4,
                success_criteria=[
                    "The loop reaches a plateau or stuck signal",
                    "The harness records a literature refresh trigger",
                    "A literature-refresh source and claim are created",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "multi_iteration_loop",
                    "literature_refresh_on_stuck",
                    "trajectory_graph_artifact",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
                metadata={"min_rounds": 3},
            ),
            EvalTask(
                id="optimize_runs_start_with_literature_grounding",
                name="Optimize and challenge-style runs search literature before producing outputs",
                prompt=(
                    "Optimize a tiny scoring function. Use existing literature and benchmark failure modes before "
                    "deciding which variants to try."
                ),
                task_mode="optimize",
                evaluator_name="length_score",
                max_iterations=2,
                success_criteria=[
                    "The optimize harness records an initial literature-grounding step",
                    "Retrieved grounding sources and claims are stored",
                    "The optimize output artifact is still emitted",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "literature_grounding_present",
                    "optimization_code_artifact",
                    "trajectory_graph_artifact",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
            ),
        ],
    )


def all_eval_suite() -> EvalSuite:
    core = default_eval_suite()
    edge = edge_eval_suite()
    return EvalSuite(
        id="all",
        name="All Harness Evaluation Suites",
        description=f"{core.description} {edge.description}",
        tasks=core.tasks + edge.tasks,
        trials_per_task=core.trials_per_task,
    )


def default_graders() -> dict[str, Grader]:
    return {
        "outcome_completed": Grader("outcome_completed", "code", "outcome verification", 1.0, 1.0, _grade_outcome_completed),
        "prd_tasks_executed": Grader("prd_tasks_executed", "code", "prd execution verification", 1.0, 1.0, _grade_prd_tasks_executed),
        "prd_tasks_executed_deterministic": Grader("prd_tasks_executed_deterministic", "code", "deterministic PRD execution verification", 1.0, 1.0, _grade_prd_tasks_executed_deterministic),
        "mode_selected": Grader("mode_selected", "code", "tool/output verification", 1.0, 1.0, _grade_mode_selected),
        "artifact_report": Grader("artifact_report", "code", "artifact existence", 0.75, 1.0, _grade_report_artifact),
        "research_groundedness": Grader("research_groundedness", "code", "groundedness assertions", 1.25, 0.8, _grade_research_groundedness),
        "report_no_fabricated_sources": Grader("report_no_fabricated_sources", "code", "source URL verification", 1.0, 1.0, _grade_report_no_fabricated_sources),
        "transcript_progress": Grader("transcript_progress", "code", "transcript analysis", 0.75, 1.0, _grade_transcript_progress),
        "model_report_rubric": Grader("model_report_rubric", "model", "deterministic rubric scoring", 0.8, 0.7, _grade_report_rubric),
        "llm_research_quality_challenger": Grader("llm_research_quality_challenger", "model", "LLM challenger research-quality rubric", 0.8, 0.7, _grade_llm_research_quality_challenger),
        "llm_hypothesis_novelty_challenger": Grader("llm_hypothesis_novelty_challenger", "model", "LLM challenger hypothesis novelty rubric", 0.6, 0.7, _grade_llm_hypothesis_novelty_challenger),
        "llm_open_ended_judgment_challenger": Grader("llm_open_ended_judgment_challenger", "model", "LLM challenger open-ended judgment", 0.6, 0.7, _grade_llm_open_ended_judgment_challenger),
        "optimize_score": Grader("optimize_score", "code", "outcome verification", 1.0, 0.01, _grade_optimize_score),
        "optimization_code_artifact": Grader("optimization_code_artifact", "code", "artifact contract", 1.0, 1.0, _grade_optimization_code_artifact),
        "seed_context": Grader("seed_context", "code", "artifact existence", 1.0, 1.0, _grade_seed_context),
        "optimize_query_phases": Grader("optimize_query_phases", "code", "trace/phase verification", 1.0, 1.0, _grade_optimize_query_phases),
        "prediction_market_solution": Grader("prediction_market_solution", "code", "static solution checks", 1.0, 1.0, _grade_prediction_market_solution),
        "prediction_market_proxy_score": Grader("prediction_market_proxy_score", "code", "local proxy outcome verification", 1.0, 0.5, _grade_prediction_market_proxy_score),
        "prediction_market_official_status": Grader("prediction_market_official_status", "code", "official-status verification", 1.0, 1.0, _grade_prediction_market_official_status),
        "prediction_market_artifact_containment": Grader("prediction_market_artifact_containment", "code", "artifact containment", 1.0, 1.0, _grade_prediction_market_artifact_containment),
        "prediction_market_candidate_files_only_in_outputs": Grader("prediction_market_candidate_files_only_in_outputs", "code", "artifact containment", 1.0, 1.0, _grade_prediction_market_candidate_files_only_in_outputs),
        "no_repo_root_strategy_files": Grader("no_repo_root_strategy_files", "code", "artifact containment", 1.0, 1.0, _grade_no_repo_root_strategy_files),
        "optimizer_skipped_without_evaluator": Grader("optimizer_skipped_without_evaluator", "code", "negative-path outcome verification", 1.0, 1.0, _grade_optimizer_skipped_without_evaluator),
        "research_search_budget": Grader("research_search_budget", "code", "search budget verification", 1.0, 1.0, _grade_research_search_budget),
        "trajectory_modes": Grader("trajectory_modes", "code", "trajectory subset/superset match", 1.0, 1.0, _grade_trajectory_modes),
        "multi_iteration_loop": Grader("multi_iteration_loop", "code", "multi-round trajectory check", 1.0, 1.0, _grade_multi_iteration_loop),
        "loop_no_score_regression": Grader("loop_no_score_regression", "code", "score regression check", 1.0, 1.0, _grade_loop_no_score_regression),
        "literature_refresh_on_stuck": Grader("literature_refresh_on_stuck", "code", "stuck-loop literature trigger", 1.0, 1.0, _grade_literature_refresh_on_stuck),
        "literature_grounding_present": Grader("literature_grounding_present", "code", "initial literature grounding", 1.0, 1.0, _grade_literature_grounding_present),
        "trajectory_graph_artifact": Grader("trajectory_graph_artifact", "code", "graph trajectory artifact", 1.0, 1.0, _grade_trajectory_graph_artifact),
        "trajectory_match_modes": Grader("trajectory_match_modes", "code", "trajectory match modes", 1.0, 1.0, _grade_trajectory_match_modes),
        "graph_trajectory_match": Grader("graph_trajectory_match", "code", "graph trajectory match", 1.0, 1.0, _grade_graph_trajectory_match),
        "parallel_trial_isolation": Grader("parallel_trial_isolation", "code", "cross-trial isolation check", 1.0, 1.0, _grade_parallel_trial_isolation_unavailable),
        "human_spot_check_placeholder": Grader("human_spot_check_placeholder", "human", "spot-check sampling", 0.0, 1.0, _grade_human_placeholder),
        "isolation_clean_trial": Grader("isolation_clean_trial", "code", "environment isolation check", 1.0, 1.0, _grade_isolation_clean_trial),
    }


def aggregate_results(task: EvalTask, results: list[GraderResult]) -> tuple[float, bool]:
    if not results:
        return 0.0, False
    total_weight = sum(result.weight for result in results if result.weight > 0)
    weighted_score = sum(result.score * result.weight for result in results if result.weight > 0) / max(total_weight, 1e-9)
    if task.aggregation == "binary":
        passed = all(result.passed for result in results if result.weight > 0)
    elif task.aggregation == "weighted":
        passed = weighted_score >= task.threshold
    else:
        required_pass = all(result.passed for result in results if result.weight >= 1.0)
        passed = required_pass and weighted_score >= task.threshold
    return round(weighted_score, 3), passed


def outcome_from_store(store: ArtifactStore) -> dict[str, Any]:
    runs = store.list("runs")
    run = runs[0] if runs else {}
    evaluations = store.list("variant_evaluations")
    best = max(evaluations, key=lambda row: float(row.get("score", 0.0)), default={})
    return {
        "run_id": run.get("id"),
        "status": run.get("status"),
        "task_mode": run.get("task_mode"),
        "product_agent": run.get("product_agent"),
        "best_score": best.get("score"),
        "best_inner_loop": best.get("inner_loop"),
        "report_exists": store.report_path.exists(),
        "solution_exists": store.solution_path.exists(),
        "optimized_candidate_exists": store.optimized_candidate_path.exists(),
        "optimal_code_exists": store.optimal_code_path.exists(),
        "optimization_result_exists": store.optimization_result_path.exists(),
        "seed_context_exists": store.optimizer_seed_context_path.exists(),
    }


def trajectory_graph(store: ArtifactStore) -> dict[str, Any]:
    decisions = store.list("task_ingestion_decisions")
    decision = decisions[0] if decisions else {}
    rounds = store.list("evolution_rounds")
    evaluations = store.list("variant_evaluations")
    traces = store.list("agent_traces")
    return {
        "nodes": [
            {"id": "prompt", "label": "Prompt"},
            {"id": "router", "label": f"Router: {decision.get('selected_mode', 'unknown')}"},
            {"id": "outer", "label": "Outer loop"},
            {"id": "inner", "label": "Inner evaluator"},
            {"id": "select", "label": "Selection"},
            {"id": "agents", "label": f"Role agents: {len(traces)}"},
            {"id": "outcome", "label": "Outcome"},
        ],
        "edges": [
            {"from": "prompt", "to": "router"},
            {"from": "router", "to": "outer"},
            {"from": "outer", "to": "inner"},
            {"from": "inner", "to": "select"},
            {"from": "select", "to": "outer", "kind": "loop"},
            {"from": "select", "to": "agents"},
            {"from": "agents", "to": "outcome"},
        ],
        "rounds": rounds,
        "evaluation_count": len(evaluations),
    }


def write_trajectory_graph_artifacts(store: ArtifactStore, trial_root: Path) -> dict[str, Path]:
    graph = trajectory_graph(store)
    mmd_path = trial_root / "trajectory_graph.mmd"
    svg_path = trial_root / "trajectory_graph.svg"
    json_path = trial_root / "trajectory_graph.json"
    json_path.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["flowchart TD"]
    for node in graph["nodes"]:
        lines.append(f'  {node["id"]}["{_mmd(str(node["label"]))}"]')
    for edge in graph["edges"]:
        connector = "-.->" if edge.get("kind") == "loop" else "-->"
        lines.append(f'  {edge["from"]} {connector} {edge["to"]}')
    for index, round_record in enumerate(graph["rounds"], start=1):
        round_id = f"round_{index}"
        label = (
            f"Round {round_record.get('outer_iteration')}: "
            f"{round_record.get('mode')} best={float(round_record.get('best_score', 0.0)):.3f} "
            f"{round_record.get('termination_signal')}"
        )
        lines.append(f'  {round_id}["{_mmd(label)}"]')
        lines.append(f"  inner --> {round_id} --> select")
    mmd_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    svg_path.write_text(_trajectory_graph_svg(graph), encoding="utf-8")
    return {"mmd": mmd_path, "svg": svg_path, "json": json_path}


def normalized_trajectory_events(store: ArtifactStore) -> list[dict[str, str]]:
    runs = store.list("runs")
    run = runs[0] if runs else {}
    decisions = store.list("task_ingestion_decisions")
    decision = decisions[0] if decisions else {}
    events = [
        {"type": "router", "name": str(decision.get("selected_mode", run.get("task_mode", "unknown")))},
    ]
    for round_record in store.list("evolution_rounds"):
        mode = str(round_record.get("mode", "unknown"))
        signal = str(round_record.get("termination_signal", "unknown"))
        events.extend(
            [
                {"type": "outer_loop", "name": mode},
                {"type": "inner_loop", "name": mode},
                {"type": "selection", "name": "variant"},
                {"type": "signal", "name": signal},
            ]
        )
    for trace in store.list("agent_traces"):
        role = str(trace.get("role", trace.get("agent_name", "agent")))
        events.append({"type": "role_agent", "name": role})
    status = str(run.get("status", "unknown"))
    events.append({"type": "outcome", "name": status})
    return events


def trajectory_match(actual: list[dict[str, str]], reference: list[dict[str, str]], mode: TrajectoryMatchMode) -> dict[str, Any]:
    if mode == "strict":
        passed = _trajectory_startswith(actual, reference)
    elif mode == "unordered":
        passed = _multiset_contains(actual, reference) and len(actual) >= len(reference)
    elif mode == "subset":
        allowed = {_event_key(item) for item in reference}
        passed = all(_event_key(item) in allowed for item in actual)
    elif mode == "superset":
        passed = _ordered_subsequence(actual, reference)
    else:
        passed = False
    return {
        "mode": mode,
        "passed": passed,
        "actual_count": len(actual),
        "reference_count": len(reference),
        "actual": actual,
        "reference": reference,
    }


def graph_trajectory_match(graph: dict[str, Any], required_edges: list[list[str]]) -> dict[str, Any]:
    actual_edges = {(edge.get("from"), edge.get("to")) for edge in graph.get("edges", [])}
    missing = [edge for edge in required_edges if tuple(edge) not in actual_edges]
    return {
        "passed": not missing,
        "missing_edges": missing,
        "actual_edges": sorted([list(edge) for edge in actual_edges]),
    }


def _event_key(event: dict[str, str]) -> tuple[str, str]:
    return (str(event.get("type", "")), str(event.get("name", "")))


def _trajectory_startswith(actual: list[dict[str, str]], reference: list[dict[str, str]]) -> bool:
    if len(actual) < len(reference):
        return False
    return [_event_key(item) for item in actual[: len(reference)]] == [_event_key(item) for item in reference]


def _ordered_subsequence(actual: list[dict[str, str]], reference: list[dict[str, str]]) -> bool:
    actual_keys = [_event_key(item) for item in actual]
    position = 0
    for ref in reference:
        key = _event_key(ref)
        try:
            position = actual_keys.index(key, position) + 1
        except ValueError:
            return False
    return True


def _multiset_contains(actual: list[dict[str, str]], reference: list[dict[str, str]]) -> bool:
    remaining = [_event_key(item) for item in actual]
    for ref in reference:
        key = _event_key(ref)
        if key not in remaining:
            return False
        remaining.remove(key)
    return True


def _mmd(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ")


def _trajectory_graph_svg(graph: dict[str, Any]) -> str:
    nodes = graph["nodes"]
    rounds = graph["rounds"]
    width = 1180
    height = 220 + max(0, len(rounds) - 1) * 42
    box_w = 132
    x0 = 28
    y = 42
    gap = 34
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#fbfaf7'/>",
        "<text x='28' y='26' font-size='18' font-family='Verdana' fill='#102a43'>Eval Trajectory Graph</text>",
    ]
    for index, node in enumerate(nodes):
        x = x0 + index * (box_w + gap)
        parts.append(f"<rect x='{x}' y='{y}' width='{box_w}' height='66' rx='8' fill='#ffffff' stroke='#52606d'/>")
        parts.append(f"<text x='{x + 10}' y='{y + 38}' font-size='12' font-family='Verdana' fill='#102a43'>{html.escape(str(node['label']))}</text>")
        if index < len(nodes) - 1:
            ax = x + box_w
            parts.append(f"<path d='M {ax + 3} {y + 33} L {ax + gap - 5} {y + 33}' stroke='#829ab1' stroke-width='2'/>")
            parts.append(f"<path d='M {ax + gap - 13} {y + 27} L {ax + gap - 4} {y + 33} L {ax + gap - 13} {y + 39}' fill='none' stroke='#829ab1' stroke-width='2'/>")
    ry = 148
    for index, round_record in enumerate(rounds, start=1):
        label = (
            f"Round {round_record.get('outer_iteration')} | {round_record.get('mode')} | "
            f"best {float(round_record.get('best_score', 0.0)):.3f} | {round_record.get('termination_signal')}"
        )
        parts.append(f"<rect x='198' y='{ry}' width='760' height='30' rx='6' fill='#eef2ff' stroke='#627d98'/>")
        parts.append(f"<text x='212' y='{ry + 20}' font-size='12' font-family='Verdana' fill='#243b53'>{html.escape(label)}</text>")
        ry += 42
    parts.append("</svg>")
    return "\n".join(parts)


def _result(
    grader: str,
    grader_type: GraderType,
    method: str,
    score: float,
    passed: bool,
    weight: float,
    summary: str,
    assertions: list[dict[str, Any]],
) -> GraderResult:
    return GraderResult(grader, grader_type, method, round(score, 3), passed, weight, summary, assertions)


def _grade_outcome_completed(task: EvalTask, store: ArtifactStore) -> GraderResult:
    status = outcome_from_store(store).get("status")
    passed = status == "completed"
    return _result("outcome_completed", "code", "outcome verification", 1.0 if passed else 0.0, passed, 1.0, f"Run status is {status}.", [{"status": status, "expected": "completed", "passed": passed}])


def _grade_prd_tasks_executed(task: EvalTask, store: ArtifactStore) -> GraderResult:
    prd = json.loads(store.prd_path.read_text(encoding="utf-8")) if store.prd_path.exists() else {}
    organized_tasks = prd.get("organized_tasks", []) if isinstance(prd, dict) else []
    loop_tasks = {row.get("id"): row for row in store.list("loop_tasks")}
    iterations_by_task = {}
    for iteration in store.list("loop_iterations"):
        iterations_by_task.setdefault(iteration.get("task_id"), []).append(iteration)
    task_checks = []
    for item in organized_tasks:
        source_task_id = item.get("source_task_id")
        loop_task = loop_tasks.get(source_task_id)
        iterations = iterations_by_task.get(source_task_id, [])
        status_matches = bool(loop_task) and item.get("status") == loop_task.get("status")
        task_checks.append(
            {
                "prd_task_id": item.get("id"),
                "source_task_id": source_task_id,
                "has_loop_task": bool(loop_task),
                "has_iteration": bool(iterations),
                "status_matches": status_matches,
                "passed": bool(loop_task) and bool(iterations) and status_matches,
            }
        )
    checks = [
        ("prd_exists", store.prd_path.exists()),
        ("prd_has_tasks", bool(organized_tasks)),
        ("all_prd_tasks_have_loop_tasks", all(check["has_loop_task"] for check in task_checks)),
        ("all_prd_tasks_have_iterations", all(check["has_iteration"] for check in task_checks)),
        ("all_prd_task_statuses_match", all(check["status_matches"] for check in task_checks)),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "prd_tasks_executed",
        "code",
        "prd execution verification",
        score,
        passed,
        1.0,
        f"Verified {sum(1 for check in task_checks if check['passed'])}/{len(task_checks)} PRD task(s) against loop tasks and iterations.",
        [{"check": name, "passed": passed} for name, passed in checks] + task_checks,
    )


def _grade_prd_tasks_executed_deterministic(task: EvalTask, store: ArtifactStore) -> GraderResult:
    prd = json.loads(store.prd_path.read_text(encoding="utf-8")) if store.prd_path.exists() else {}
    organized_tasks = prd.get("organized_tasks", []) if isinstance(prd, dict) else []
    loop_tasks = {row.get("id"): row for row in store.list("loop_tasks")}
    iterations_by_task: dict[str, list[dict[str, Any]]] = {}
    for iteration in store.list("loop_iterations"):
        iterations_by_task.setdefault(str(iteration.get("task_id")), []).append(iteration)

    task_assertions: list[dict[str, Any]] = []
    for item in organized_tasks:
        source_task_id = str(item.get("source_task_id") or "")
        loop_task = loop_tasks.get(source_task_id)
        iterations = iterations_by_task.get(source_task_id, [])
        terminal_status = str(item.get("status")) in {"passed", "failed", "skipped"}
        iteration_terminal = any(str(iteration.get("status")) in {"passed", "failed", "skipped", "completed"} for iteration in iterations)
        attempted = bool(loop_task) and int(loop_task.get("attempts") or 0) > 0
        evidence = bool(iterations) and bool(loop_task) and bool(loop_task.get("result_summary") or loop_task.get("last_error"))
        passed = bool(loop_task) and terminal_status and attempted and iteration_terminal and evidence
        task_assertions.append(
            {
                "prd_task_id": item.get("id"),
                "source_task_id": source_task_id,
                "title": item.get("title"),
                "has_loop_task": bool(loop_task),
                "terminal_prd_status": terminal_status,
                "attempted": attempted,
                "has_terminal_iteration": iteration_terminal,
                "has_result_or_error_evidence": evidence,
                "passed": passed,
            }
        )

    global_checks = [
        ("prd_exists", store.prd_path.exists()),
        ("prd_has_tasks", bool(organized_tasks)),
        ("no_pending_prd_tasks", bool(organized_tasks) and all(str(item.get("status")) in {"passed", "failed", "skipped"} for item in organized_tasks)),
        ("every_task_attempted_with_evidence", bool(task_assertions) and all(item["passed"] for item in task_assertions)),
    ]
    score = sum(1 for _, passed in global_checks if passed) / len(global_checks)
    passed = score == 1.0
    return _result(
        "prd_tasks_executed_deterministic",
        "code",
        "deterministic PRD execution verification",
        score,
        passed,
        1.0,
        f"Deterministically verified {sum(1 for item in task_assertions if item['passed'])}/{len(task_assertions)} PRD task(s) reached terminal execution with evidence.",
        [{"check": name, "passed": passed} for name, passed in global_checks] + task_assertions,
    )


def _grade_mode_selected(task: EvalTask, store: ArtifactStore) -> GraderResult:
    decisions = store.list("task_ingestion_decisions")
    selected = decisions[0].get("selected_mode") if decisions else None
    passed = selected == task.task_mode
    return _result("mode_selected", "code", "tool/output verification", 1.0 if passed else 0.0, passed, 1.0, f"Selected mode {selected}; expected {task.task_mode}.", [{"selected_mode": selected, "expected": task.task_mode, "passed": passed}])


def _grade_report_artifact(task: EvalTask, store: ArtifactStore) -> GraderResult:
    passed = store.report_path.exists() and len(store.report_path.read_text(encoding="utf-8").strip()) > 80
    return _result("artifact_report", "code", "artifact existence", 1.0 if passed else 0.0, passed, 0.75, "Final report artifact checked.", [{"path": str(store.report_path), "passed": passed}])


def _grade_report_no_fabricated_sources(task: EvalTask, store: ArtifactStore) -> GraderResult:
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    sources = store.list("sources")
    known_urls = {str(s.get("url", "")) for s in sources}
    report_urls = re.findall(r"\]\((https?://[^)]+)\)", report)
    fabricated = []
    for url in report_urls:
        if "example.org" in url or "example.com" in url:
            fabricated.append({"url": url, "reason": "placeholder/example domain"})
        elif url not in known_urls:
            fabricated.append({"url": url, "reason": "not in sources.json"})
    passed = not fabricated
    score = max(0.0, 1.0 - len(fabricated) * 0.25)
    return _result(
        "report_no_fabricated_sources",
        "code",
        "source URL verification",
        score,
        passed,
        1.0,
        f"Found {len(fabricated)} fabricated source URL(s) in report out of {len(report_urls)} cited.",
        [{"check": "no_fabricated_sources", "passed": passed, "fabricated_urls": fabricated}],
    )


def _grade_research_groundedness(task: EvalTask, store: ArtifactStore) -> GraderResult:
    sources = store.list("sources")
    claims = store.list("claims")
    grounded_claims = [claim for claim in claims if claim.get("source_ids")]
    source_score = min(1.0, len(sources) / 4)
    claim_score = min(1.0, len(claims) / 8)
    grounded_score = len(grounded_claims) / max(len(claims), 1)
    score = (source_score * 0.3) + (claim_score * 0.3) + (grounded_score * 0.4)
    passed = score >= 0.8
    return _result(
        "research_groundedness",
        "code",
        "groundedness assertions",
        score,
        passed,
        1.25,
        f"{len(sources)} sources, {len(claims)} claims, {len(grounded_claims)} grounded claims.",
        [
            {"check": "min_sources", "actual": len(sources), "expected_at_least": 4, "passed": len(sources) >= 4},
            {"check": "min_claims", "actual": len(claims), "expected_at_least": 8, "passed": len(claims) >= 8},
            {"check": "all_claims_have_sources", "actual": len(grounded_claims), "total": len(claims), "passed": grounded_score == 1.0},
        ],
    )


def _grade_transcript_progress(task: EvalTask, store: ArtifactStore) -> GraderResult:
    progress = store.progress_path.read_text(encoding="utf-8") if store.progress_path.exists() else ""
    traces = store.list("agent_traces")
    has_complete = "<promise>COMPLETE</promise>" in progress
    has_incomplete_stop = "Stopped with" in progress and "incomplete loop tasks" in progress
    has_steps = "Task 1:" in progress and len(progress.splitlines()) >= 5
    passed = (has_complete or has_incomplete_stop) and has_steps
    return _result(
        "transcript_progress",
        "code",
        "transcript analysis",
        1.0 if passed else 0.0,
        passed,
        0.75,
        f"Progress lines={len(progress.splitlines())}; traces={len(traces)}.",
        [
            {"check": "complete_or_incomplete_stop_marker", "passed": has_complete or has_incomplete_stop},
            {"check": "step_visibility", "passed": has_steps},
        ],
    )


def _grade_report_rubric(task: EvalTask, store: ArtifactStore) -> GraderResult:
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    evaluations = store.list("variant_evaluations")
    research_metrics = [
        row.get("metrics", {})
        for row in evaluations
        if row.get("inner_loop") == "research" and isinstance(row.get("metrics"), dict)
    ]
    rubric_dimensions = {"factual_accuracy", "citation_accuracy", "completeness", "source_quality", "tool_efficiency"}
    has_research_rubric_metrics = bool(research_metrics) and all(
        dimension in research_metrics[0] for dimension in rubric_dimensions
    )
    rubric_checks = [
        ("has_summary", "summary" in report.lower() or "findings" in report.lower()),
        ("mentions_sources", "source" in report.lower()),
        ("mentions_uncertainty", any(term in report.lower() for term in ["uncertain", "caveat", "contradiction", "limitation"])),
        ("has_research_rubric_metrics", has_research_rubric_metrics),
        ("substantial_length", len(report.split()) >= 80),
    ]
    score = sum(1 for _, passed in rubric_checks if passed) / len(rubric_checks)
    passed = score >= 0.7
    return _result("model_report_rubric", "model", "deterministic rubric scoring", score, passed, 0.8, "Local model-style rubric scored the report.", [{"check": name, "passed": passed} for name, passed in rubric_checks])


def _grade_llm_research_quality_challenger(task: EvalTask, store: ArtifactStore) -> GraderResult:
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    claims = store.list("claims")
    sources = store.list("sources")
    evaluations = store.list("variant_evaluations")
    research_metrics = [
        row.get("metrics", {})
        for row in evaluations
        if row.get("inner_loop") == "research" and isinstance(row.get("metrics"), dict)
    ]
    first_metrics = research_metrics[0] if research_metrics else {}
    grounded_claims = [claim for claim in claims if claim.get("source_ids")]
    dimensions = {
        "factual_accuracy": max(float(first_metrics.get("factual_accuracy", 0.0)), len(grounded_claims) / max(len(claims), 1)),
        "citation_accuracy": max(float(first_metrics.get("citation_accuracy", 0.0)), 1.0 if grounded_claims and len(grounded_claims) == len(claims) else 0.0),
        "completeness": max(float(first_metrics.get("completeness", 0.0)), min(1.0, len(report.split()) / 180.0)),
        "source_quality": max(float(first_metrics.get("source_quality", 0.0)), min(1.0, len(sources) / 4.0)),
    }
    score = sum(dimensions.values()) / len(dimensions)
    passed = score >= 0.7
    return _result(
        "llm_research_quality_challenger",
        "model",
        "LLM challenger research-quality rubric",
        score,
        passed,
        0.8,
        "Model-style challenger rated research quality across factual accuracy, citation accuracy, completeness, and source quality.",
        [{"dimension": name, "score": round(value, 3), "passed": value >= 0.7} for name, value in dimensions.items()],
    )


def _grade_llm_hypothesis_novelty_challenger(task: EvalTask, store: ArtifactStore) -> GraderResult:
    hypotheses = store.list("hypotheses")
    claims = {str(claim.get("text", "")).strip().lower() for claim in store.list("claims")}
    assertions: list[dict[str, Any]] = []
    scores: list[float] = []
    for hypothesis in hypotheses:
        text = str(hypothesis.get("text", "")).strip()
        novelty_score = float(hypothesis.get("novelty_score", 0.0) or 0.0)
        not_copy = text.lower() not in claims
        has_test = bool(hypothesis.get("next_experiment"))
        length_ok = len(text.split()) >= 6
        score = (novelty_score * 0.5) + (0.2 if not_copy else 0.0) + (0.2 if has_test else 0.0) + (0.1 if length_ok else 0.0)
        scores.append(min(1.0, score))
        assertions.append(
            {
                "hypothesis_id": hypothesis.get("id"),
                "question": "Is this hypothesis novel?",
                "novelty_score": novelty_score,
                "not_claim_copy": not_copy,
                "has_next_experiment": has_test,
                "passed": score >= 0.7,
            }
        )
    score = sum(scores) / max(len(scores), 1)
    passed = bool(hypotheses) and score >= 0.7
    return _result(
        "llm_hypothesis_novelty_challenger",
        "model",
        "LLM challenger hypothesis novelty rubric",
        score,
        passed,
        0.6,
        f"Model-style challenger judged novelty for {len(hypotheses)} hypothesis/hypotheses.",
        assertions or [{"check": "has_hypotheses", "passed": False}],
    )


def _grade_llm_open_ended_judgment_challenger(task: EvalTask, store: ArtifactStore) -> GraderResult:
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    progress = store.progress_path.read_text(encoding="utf-8") if store.progress_path.exists() else ""
    lower_report = report.lower()
    checks = [
        ("answers_user_prompt", any(term in lower_report for term in _keywords(task.prompt, limit=8))),
        ("uses_evidence_language", any(term in lower_report for term in ["source", "claim", "evidence", "citation"])),
        ("handles_uncertainty", any(term in lower_report for term in ["uncertain", "limitation", "caveat", "contradiction", "confidence"])),
        ("has_synthesis", any(term in lower_report for term in ["summary", "synthesis", "findings", "recommendation"])),
        ("run_reached_terminal_marker", "<promise>complete</promise>" in progress.lower() or "stopped with" in progress.lower()),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score >= 0.7
    return _result(
        "llm_open_ended_judgment_challenger",
        "model",
        "LLM challenger open-ended judgment",
        score,
        passed,
        0.6,
        "Model-style challenger made an open-ended judgment on relevance, evidence use, uncertainty, synthesis, and terminal progress.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _keywords(text: str, limit: int = 8) -> list[str]:
    stop = {"the", "and", "for", "with", "that", "this", "from", "into", "about", "research", "optimize", "how"}
    words = [word.lower() for word in re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", text) if word.lower() not in stop]
    unique: list[str] = []
    for word in words:
        if word not in unique:
            unique.append(word)
    return unique[:limit]


def _grade_optimize_score(task: EvalTask, store: ArtifactStore) -> GraderResult:
    optimize_evals = [row for row in store.list("variant_evaluations") if row.get("inner_loop") == "optimize"]
    best = max((float(row.get("score", 0.0)) for row in optimize_evals), default=0.0)
    passed = best > 0.0
    return _result("optimize_score", "code", "outcome verification", best, passed, 1.0, f"Best optimize score={best:.3f}.", [{"best_score": best, "passed": passed}])


def _grade_optimization_code_artifact(task: EvalTask, store: ArtifactStore) -> GraderResult:
    text = store.optimal_code_path.read_text(encoding="utf-8") if store.optimal_code_path.exists() else ""
    result = json.loads(store.optimization_result_path.read_text(encoding="utf-8")) if store.optimization_result_path.exists() else {}
    checks = [
        ("optimal_code_exists", store.optimal_code_path.exists()),
        ("optimal_code_nonempty", len(text.strip()) > 40),
        ("optimization_result_exists", store.optimization_result_path.exists()),
        ("optimization_result_points_to_optimal_code", result.get("optimal_code_path") == str(store.optimal_code_path)),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "optimization_code_artifact",
        "code",
        "artifact contract",
        score,
        passed,
        1.0,
        "Optimization run emitted the universal optimal_code.py artifact.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_seed_context(task: EvalTask, store: ArtifactStore) -> GraderResult:
    exists = store.optimizer_seed_context_path.exists()
    payload = json.loads(store.optimizer_seed_context_path.read_text(encoding="utf-8")) if exists else {}
    top = payload.get("top_query_findings", [])
    passed = exists and bool(top)
    return _result("seed_context", "code", "artifact existence", 1.0 if passed else 0.0, passed, 1.0, "Optimizer seed context checked.", [{"exists": exists, "top_query_findings": len(top) if isinstance(top, list) else 0, "passed": passed}])


def _grade_optimize_query_phases(task: EvalTask, store: ArtifactStore) -> GraderResult:
    loops = {row.get("inner_loop") for row in store.list("variant_evaluations")}
    passed = {"optimize_query", "optimize"}.issubset(loops)
    return _result("optimize_query_phases", "code", "trace/phase verification", 1.0 if passed else 0.0, passed, 1.0, f"Inner loops seen: {sorted(str(loop) for loop in loops)}.", [{"loops": sorted(str(loop) for loop in loops), "passed": passed}])


def _grade_prediction_market_solution(task: EvalTask, store: ArtifactStore) -> GraderResult:
    text = store.solution_path.read_text(encoding="utf-8") if store.solution_path.exists() else ""
    checks = [
        ("solution_exists", store.solution_path.exists()),
        ("optimized_candidate_exists", store.optimized_candidate_path.exists()),
        ("optimal_code_exists", store.optimal_code_path.exists()),
        ("optimization_result_exists", store.optimization_result_path.exists()),
        ("defines_strategy", "class Strategy" in text),
        ("has_on_step", "def on_step" in text),
        ("uses_upstream_api", "orderbook_pm_challenge" in text),
        ("uses_cancel_or_orders", "CancelAll" in text and "PlaceOrder" in text),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result("prediction_market_solution", "code", "static solution checks", score, passed, 1.0, "Generated solution.py checked against upstream API shape.", [{"check": name, "passed": passed} for name, passed in checks])


def _grade_prediction_market_proxy_score(task: EvalTask, store: ArtifactStore) -> GraderResult:
    optimize_evals = [row for row in store.list("variant_evaluations") if row.get("inner_loop") == "optimize"]
    best = max((float(row.get("score", 0.0)) for row in optimize_evals), default=0.0)
    # Threshold 0.45: normalized score maps 0-edge strategies to ~0.5; allow a
    # small margin so near-zero-edge strategies (acceptable baseline) still pass.
    passed = best >= 0.45
    return _result(
        "prediction_market_proxy_score",
        "code",
        "local proxy outcome verification",
        best,
        passed,
        1.0,
        f"Best local proxy score={best:.3f}. This is not the official upstream profit score.",
        [{"best_proxy_score": best, "official_score_required": True, "passed": passed}],
    )


def _grade_prediction_market_official_status(task: EvalTask, store: ArtifactStore) -> GraderResult:
    result = json.loads(store.optimization_result_path.read_text(encoding="utf-8")) if store.optimization_result_path.exists() else {}
    official = result.get("official_result", {}) if isinstance(result, dict) else {}
    source = official.get("score_source")
    measured = official.get("measured")
    candidate_path = official.get("candidate_path")
    # Accept either truthful outcome: if the upstream runner was used the result
    # must say so; if the fallback ran it must say it wasn't officially measured.
    if source == "upstream_orderbook_pm_challenge":
        expected_measured = measured is True
    else:
        expected_measured = measured is False
    checks = [
        ("optimization_result_exists", store.optimization_result_path.exists()),
        ("official_result_present", isinstance(official, dict) and bool(official)),
        ("measured_status_truthful", bool(expected_measured)),
        ("score_source_recorded", source in {"local_sandbox_strategy_execution", "local_official_semantics_fallback", "upstream_orderbook_pm_challenge"}),
        ("candidate_path_recorded", bool(candidate_path)),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "prediction_market_official_status",
        "code",
        "official-status verification",
        score,
        passed,
        1.0,
        f"official_result measured={measured}, score_source={source}, candidate_path={candidate_path}.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_prediction_market_artifact_containment(task: EvalTask, store: ArtifactStore) -> GraderResult:
    result = json.loads(store.optimization_result_path.read_text(encoding="utf-8")) if store.optimization_result_path.exists() else {}
    official = result.get("official_result", {}) if isinstance(result, dict) else {}
    candidate_path = Path(str(official.get("candidate_path", ""))) if official.get("candidate_path") else None
    candidate_files = list(store.candidates_dir.glob("*.py")) if store.candidates_dir.exists() else []
    repo_root = Path.cwd()
    leaked_files = [
        path
        for pattern in ("pm_strategy*.py", "tmp_pm*.py", "*prediction_market_strategy*.py")
        for path in repo_root.glob(pattern)
        if path.is_file()
    ]
    checks = [
        ("candidates_dir_exists", store.candidates_dir.exists()),
        ("candidate_files_exist", bool(candidate_files)),
        (
            "winner_inside_candidates_dir",
            bool(candidate_path) and candidate_path.exists() and store.candidates_dir.resolve() in candidate_path.resolve().parents,
        ),
        ("no_repo_root_strategy_leaks", not leaked_files),
        ("optimal_code_exists", store.optimal_code_path.exists()),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "prediction_market_artifact_containment",
        "code",
        "artifact containment",
        score,
        passed,
        1.0,
        f"{len(candidate_files)} candidate file(s); leaked repo-root strategy files={len(leaked_files)}.",
        [
            {"check": name, "passed": passed}
            for name, passed in checks
        ]
        + [{"check": "leaked_file", "path": str(path), "passed": False} for path in leaked_files],
    )


def _repo_root_strategy_leaks(repo_root: Path) -> list[Path]:
    return [
        path
        for pattern in ("pm_strategy*.py", "tmp_pm*.py", "*prediction_market_strategy*.py")
        for path in repo_root.glob(pattern)
        if path.is_file()
    ]


def _grade_prediction_market_candidate_files_only_in_outputs(task: EvalTask, store: ArtifactStore) -> GraderResult:
    result = json.loads(store.optimization_result_path.read_text(encoding="utf-8")) if store.optimization_result_path.exists() else {}
    official = result.get("official_result", {}) if isinstance(result, dict) else {}
    candidate_path = Path(str(official.get("candidate_path", ""))) if official.get("candidate_path") else None
    candidate_files = list(store.candidates_dir.glob("*.py")) if store.candidates_dir.exists() else []
    candidate_parent = store.candidates_dir.resolve()
    checks = [
        ("candidates_dir_exists", store.candidates_dir.exists()),
        ("candidate_files_exist", bool(candidate_files)),
        ("all_candidate_files_under_candidates_dir", all(candidate_parent in path.resolve().parents for path in candidate_files)),
        (
            "winner_candidate_under_candidates_dir",
            bool(candidate_path) and candidate_path.exists() and candidate_parent in candidate_path.resolve().parents,
        ),
        ("optimal_code_promoted", store.optimal_code_path.exists()),
        ("solution_promoted", store.solution_path.exists()),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "prediction_market_candidate_files_only_in_outputs",
        "code",
        "artifact containment",
        score,
        passed,
        1.0,
        f"{len(candidate_files)} candidate Python file(s) under {store.candidates_dir}.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_no_repo_root_strategy_files(task: EvalTask, store: ArtifactStore) -> GraderResult:
    leaked_files = _repo_root_strategy_leaks(Path.cwd())
    passed = not leaked_files
    return _result(
        "no_repo_root_strategy_files",
        "code",
        "artifact containment",
        1.0 if passed else 0.0,
        passed,
        1.0,
        f"Repository root generated strategy leak count={len(leaked_files)}.",
        [{"check": "no_repo_root_strategy_files", "passed": passed}]
        + [{"check": "leaked_file", "path": str(path), "passed": False} for path in leaked_files],
    )


def _grade_optimizer_skipped_without_evaluator(task: EvalTask, store: ArtifactStore) -> GraderResult:
    seed_context = json.loads(store.optimizer_seed_context_path.read_text(encoding="utf-8")) if store.optimizer_seed_context_path.exists() else {}
    loops = {row.get("inner_loop") for row in store.list("variant_evaluations")}
    progress = store.progress_path.read_text(encoding="utf-8") if store.progress_path.exists() else ""
    checks = [
        ("seed_context_exists", store.optimizer_seed_context_path.exists()),
        ("seed_context_has_no_evaluator", seed_context.get("has_evaluator") is False),
        ("no_optimize_inner_loop", "optimize" not in loops),
        ("skip_recorded_in_progress", "Optimizer phase skipped" in progress),
        ("no_optimization_result_fabricated", not store.optimization_result_path.exists()),
        ("no_optimal_code_fabricated", not store.optimal_code_path.exists()),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "optimizer_skipped_without_evaluator",
        "code",
        "negative-path outcome verification",
        score,
        passed,
        1.0,
        f"Inner loops seen: {sorted(str(loop) for loop in loops)}; has_evaluator={seed_context.get('has_evaluator')}.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_research_search_budget(task: EvalTask, store: ArtifactStore) -> GraderResult:
    max_sources = int(task.metadata.get("max_sources", 8))
    max_claims = int(task.metadata.get("max_claims", 24))
    max_query_evaluations = int(task.metadata.get("max_query_evaluations", 4))
    max_evolution_rounds = int(task.metadata.get("max_evolution_rounds", 1))
    sources = store.list("sources")
    claims = store.list("claims")
    query_evaluations = [
        row for row in store.list("variant_evaluations")
        if row.get("inner_loop") in {"research", "optimize_query"}
    ]
    rounds = store.list("evolution_rounds")
    checks = [
        ("sources_under_budget", len(sources) <= max_sources),
        ("claims_under_budget", len(claims) <= max_claims),
        ("query_evaluations_under_budget", len(query_evaluations) <= max_query_evaluations),
        ("evolution_rounds_under_budget", len(rounds) <= max_evolution_rounds),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "research_search_budget",
        "code",
        "search budget verification",
        score,
        passed,
        1.0,
        (
            f"sources={len(sources)}/{max_sources}, claims={len(claims)}/{max_claims}, "
            f"query_evaluations={len(query_evaluations)}/{max_query_evaluations}, rounds={len(rounds)}/{max_evolution_rounds}."
        ),
        [
            {"check": "sources_under_budget", "actual": len(sources), "max": max_sources, "passed": len(sources) <= max_sources},
            {"check": "claims_under_budget", "actual": len(claims), "max": max_claims, "passed": len(claims) <= max_claims},
            {
                "check": "query_evaluations_under_budget",
                "actual": len(query_evaluations),
                "max": max_query_evaluations,
                "passed": len(query_evaluations) <= max_query_evaluations,
            },
            {
                "check": "evolution_rounds_under_budget",
                "actual": len(rounds),
                "max": max_evolution_rounds,
                "passed": len(rounds) <= max_evolution_rounds,
            },
        ],
    )


def _grade_trajectory_modes(task: EvalTask, store: ArtifactStore) -> GraderResult:
    required = set(task.metadata.get("required_modes", []))
    forbidden = set(task.metadata.get("forbidden_modes", []))
    evaluation_modes = {str(row.get("inner_loop")) for row in store.list("variant_evaluations") if row.get("inner_loop")}
    evolution_modes = {str(row.get("mode")) for row in store.list("evolution_rounds") if row.get("mode")}
    modes = evaluation_modes | evolution_modes
    missing = sorted(required - modes)
    forbidden_seen = sorted(forbidden & modes)
    passed = not missing and not forbidden_seen
    score = 1.0 if passed else 0.0
    return _result(
        "trajectory_modes",
        "code",
        "trajectory subset/superset match",
        score,
        passed,
        1.0,
        f"Required modes={sorted(required)}; forbidden modes={sorted(forbidden)}; observed modes={sorted(modes)}.",
        [
            {"check": "required_modes_subset", "missing": missing, "passed": not missing},
            {"check": "forbidden_modes_absent", "forbidden_seen": forbidden_seen, "passed": not forbidden_seen},
        ],
    )


def _grade_multi_iteration_loop(task: EvalTask, store: ArtifactStore) -> GraderResult:
    min_rounds = int(task.metadata.get("min_rounds", 2))
    rounds = store.list("evolution_rounds")
    checks = [
        ("min_rounds", len(rounds) >= min_rounds),
        ("all_rounds_have_best_score", all("best_score" in row for row in rounds)),
        ("all_rounds_have_signal", all(row.get("termination_signal") for row in rounds)),
        ("variant_evaluations_exist", bool(store.list("variant_evaluations"))),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "multi_iteration_loop",
        "code",
        "multi-round trajectory check",
        score,
        passed,
        1.0,
        f"Observed {len(rounds)} evolution round(s); required at least {min_rounds}.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_loop_no_score_regression(task: EvalTask, store: ArtifactStore) -> GraderResult:
    max_drop = float(task.metadata.get("max_score_drop", 0.2))
    rounds = store.list("evolution_rounds")
    scores = [float(row.get("best_score", 0.0)) for row in rounds]
    if not scores:
        return _result("loop_no_score_regression", "code", "score regression check", 0.0, False, 1.0, "No round scores were recorded.", [{"check": "round_scores_exist", "passed": False}])
    first = scores[0]
    best = max(scores)
    final = scores[-1]
    worst_drop = max(0.0, best - final)
    result = json.loads(store.optimization_result_path.read_text(encoding="utf-8")) if store.optimization_result_path.exists() else {}
    selected_score = float(result.get("score", 0.0))
    checks = [
        ("selected_score_preserves_best", selected_score >= best - 1e-9),
        ("final_drop_recorded_within_budget_or_best_preserved", worst_drop <= max_drop or selected_score >= best - 1e-9),
        ("best_at_least_first", best >= first),
        ("scores_in_unit_interval", all(0.0 <= score <= 1.0 for score in scores)),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "loop_no_score_regression",
        "code",
        "score regression check",
        score,
        passed,
        1.0,
        f"round_scores={scores}; best={best:.3f}; final={final:.3f}; selected={selected_score:.3f}; worst_drop={worst_drop:.3f}.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_literature_refresh_on_stuck(task: EvalTask, store: ArtifactStore) -> GraderResult:
    progress = store.progress_path.read_text(encoding="utf-8") if store.progress_path.exists() else ""
    rounds = store.list("evolution_rounds")
    sources = store.list("sources")
    claims = store.list("claims")
    stuck_signals = {row.get("termination_signal") for row in rounds if row.get("termination_signal") in {"score_plateau", "coverage_plateau"}}
    refresh_sources = [source for source in sources if str(source.get("url", "")).startswith("memory://literature-refresh/")]
    refresh_claims = [claim for claim in claims if claim.get("created_by_agent") == "literature_refresh_policy"]
    checks = [
        ("stuck_signal_observed", bool(stuck_signals)),
        ("progress_records_literature_refresh", "Literature refresh triggered" in progress),
        ("refresh_source_created", bool(refresh_sources)),
        ("refresh_claim_created", bool(refresh_claims)),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "literature_refresh_on_stuck",
        "code",
        "stuck-loop literature trigger",
        score,
        passed,
        1.0,
        f"stuck_signals={sorted(str(signal) for signal in stuck_signals)}; refresh_sources={len(refresh_sources)}; refresh_claims={len(refresh_claims)}.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_literature_grounding_present(task: EvalTask, store: ArtifactStore) -> GraderResult:
    progress = store.progress_path.read_text(encoding="utf-8") if store.progress_path.exists() else ""
    claims = store.list("claims")
    sources = store.list("sources")
    grounding_claims = [claim for claim in claims if claim.get("created_by_agent") == "literature_grounding_policy"]
    grounding_source_ids = {source_id for claim in grounding_claims for source_id in claim.get("source_ids", [])}
    checks = [
        ("progress_records_initial_grounding", "Literature grounding (initial)" in progress),
        ("grounding_claim_created", bool(grounding_claims)),
        ("grounding_claim_has_source", bool(grounding_source_ids)),
        ("grounding_source_exists", any(source.get("id") in grounding_source_ids for source in sources)),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "literature_grounding_present",
        "code",
        "initial literature grounding",
        score,
        passed,
        1.0,
        f"grounding_claims={len(grounding_claims)}; grounding_source_ids={len(grounding_source_ids)}.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_trajectory_graph_artifact(task: EvalTask, store: ArtifactStore) -> GraderResult:
    trial_root = store.root.parent
    mmd = trial_root / "trajectory_graph.mmd"
    svg = trial_root / "trajectory_graph.svg"
    graph_json = trial_root / "trajectory_graph.json"
    text = mmd.read_text(encoding="utf-8") if mmd.exists() else ""
    checks = [
        ("mermaid_graph_exists", mmd.exists()),
        ("svg_graph_exists", svg.exists()),
        ("json_graph_exists", graph_json.exists()),
        ("graph_mentions_outer_loop", "outer" in text.lower()),
        ("graph_mentions_inner_loop", "inner" in text.lower()),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "trajectory_graph_artifact",
        "code",
        "graph trajectory artifact",
        score,
        passed,
        1.0,
        f"Trajectory graph artifacts: {mmd}, {svg}, {graph_json}.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_trajectory_match_modes(task: EvalTask, store: ArtifactStore) -> GraderResult:
    actual = normalized_trajectory_events(store)
    reference = [
        {"type": str(item.get("type", "")), "name": str(item.get("name", ""))}
        for item in task.metadata.get("reference_trajectory", [])
        if isinstance(item, dict)
    ]
    if not reference:
        reference = [
            {"type": "router", "name": task.task_mode},
            {"type": "outer_loop", "name": task.task_mode},
            {"type": "inner_loop", "name": task.task_mode},
            {"type": "selection", "name": "variant"},
            {"type": "outcome", "name": "completed"},
        ]
    strict_reference = reference[:4]
    minimal_reference = [reference[0], reference[1], reference[2], reference[-1]]
    allowed_reference = list({(event["type"], event["name"]): event for event in actual}.values())
    results = {
        "strict": trajectory_match(actual, strict_reference, "strict"),
        "unordered": trajectory_match(actual, minimal_reference, "unordered"),
        "superset": trajectory_match(actual, minimal_reference, "superset"),
        "subset": trajectory_match(actual, allowed_reference, "subset"),
    }
    checks = [(mode, result["passed"]) for mode, result in results.items()]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    match_summary = ", ".join(f"{mode}={result['passed']}" for mode, result in results.items())
    return _result(
        "trajectory_match_modes",
        "code",
        "trajectory match modes",
        score,
        passed,
        1.0,
        f"Matched normalized trajectory using modes: {match_summary}.",
        [{"check": mode, "passed": result["passed"], "actual_count": result["actual_count"], "reference_count": result["reference_count"]} for mode, result in results.items()],
    )


def _grade_graph_trajectory_match(task: EvalTask, store: ArtifactStore) -> GraderResult:
    graph = trajectory_graph(store)
    required_edges = task.metadata.get("required_graph_edges", [])
    if not isinstance(required_edges, list) or not required_edges:
        required_edges = [["prompt", "router"], ["router", "outer"], ["outer", "inner"], ["inner", "select"], ["agents", "outcome"]]
    result = graph_trajectory_match(graph, required_edges)
    passed = bool(result["passed"])
    return _result(
        "graph_trajectory_match",
        "code",
        "graph trajectory match",
        1.0 if passed else 0.0,
        passed,
        1.0,
        f"Graph trajectory required_edges={len(required_edges)} missing={len(result['missing_edges'])}.",
        [{"check": "required_graph_edges_present", "missing_edges": result["missing_edges"], "passed": passed}],
    )


def _grade_parallel_trial_isolation_unavailable(task: EvalTask, store: ArtifactStore) -> GraderResult:
    return _result(
        "parallel_trial_isolation",
        "code",
        "cross-trial isolation check",
        0.0,
        False,
        1.0,
        "parallel_trial_isolation is a cross-trial grader and must be applied after sibling trials finish.",
        [{"check": "cross_trial_context_available", "passed": False}],
    )


def _grade_parallel_trial_isolation_from_trials(task: EvalTask, trials: list[EvalTrial]) -> GraderResult:
    trial_roots = [trial.isolation.get("trial_root") for trial in trials]
    output_roots = [trial.isolation.get("output_root") for trial in trials]
    tmpdirs = [trial.isolation.get("tmpdir") for trial in trials]
    run_ids = [trial.run_id for trial in trials]
    expected = task.trials or len(trials)
    checks = [
        ("expected_trial_count", len(trials) >= expected),
        ("unique_trial_roots", len(set(trial_roots)) == len(trial_roots)),
        ("unique_output_roots", len(set(output_roots)) == len(output_roots)),
        ("unique_tmpdirs", len(set(tmpdirs)) == len(tmpdirs)),
        ("all_paths_exist", all(Path(str(path)).exists() for path in [*trial_roots, *output_roots, *tmpdirs] if path)),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "parallel_trial_isolation",
        "code",
        "cross-trial isolation check",
        score,
        passed,
        1.0,
        f"Checked {len(trials)} trial(s) for distinct roots, output roots, tmpdirs, and run ids.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_human_placeholder(task: EvalTask, store: ArtifactStore) -> GraderResult:
    return _result(
        "human_spot_check_placeholder",
        "human",
        "spot-check sampling",
        0.0,
        False,
        0.0,
        "Human grader slot recorded but not required in automated local suite.",
        [{"requires_human": True, "passed": False}],
    )


def _grade_isolation_clean_trial(task: EvalTask, store: ArtifactStore) -> GraderResult:
    trial_root = store.root.parent
    run_dirs = [path for path in trial_root.iterdir() if path.is_dir() and _is_run_dir(path.name)]
    has_trial_tmp = (trial_root / "tmp").exists()
    passed = len(run_dirs) == 1 and has_trial_tmp and str(store.root).startswith(str(trial_root))
    return _result(
        "isolation_clean_trial",
        "code",
        "environment isolation check",
        1.0 if passed else 0.0,
        passed,
        1.0,
        f"Trial root {trial_root} contains {len(run_dirs)} run artifact directory/directories.",
        [
            {"check": "single_run_directory", "actual": len(run_dirs), "expected": 1, "passed": len(run_dirs) == 1},
            {"check": "per_trial_tmpdir", "path": str(trial_root / "tmp"), "passed": has_trial_tmp},
            {"check": "store_under_trial_root", "path": str(store.root), "passed": str(store.root).startswith(str(trial_root))},
        ],
    )


def _is_run_dir(name: str) -> bool:
    return name.startswith("run_") or bool(re.match(r"^\d+_run_", name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prewritten research-harness eval suites.")
    parser.add_argument("--suite", choices=["core", "edge", "all"], default="core")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("eval_outputs"))
    parser.add_argument("--corpus", type=Path, default=Path("examples/corpus/research_corpus.json"))
    args = parser.parse_args()
    if args.suite == "edge":
        suite = edge_eval_suite()
    elif args.suite == "all":
        suite = all_eval_suite()
    else:
        suite = default_eval_suite()
    suite.trials_per_task = args.trials
    summary = asyncio.run(EvaluationHarness(corpus_path=args.corpus, output_root=args.output).run_suite(suite))
    print(f"Eval suite: {summary.suite_name}")
    print(f"Trials: {summary.passed_trials}/{summary.trial_count} passed")
    print(f"Aggregate score: {summary.aggregate_score:.3f}")
    print(f"Summary: {args.output / (suite.id + '_summary.json')}")


if __name__ == "__main__":
    main()
