from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from .orchestrator import HarnessConfig, Orchestrator
from .schemas import now_iso
from .store import ArtifactStore


GraderType = Literal["code", "model", "human"]
AggregationMode = Literal["weighted", "binary", "hybrid"]


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
        self.run_output_root = output_root / "runs"
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
        trial_output_root = trial_root / "outputs"
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
        self.run_output_root.mkdir(parents=True, exist_ok=True)

    def _prepare_trial_root(self, task: EvalTask, trial_index: int) -> Path:
        trial_root = self.run_output_root / f"{task.id}_trial_{trial_index:03d}"
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
                    "research_groundedness",
                    "artifact_report",
                    "transcript_progress",
                    "isolation_clean_trial",
                    "model_report_rubric",
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
        "mode_selected": Grader("mode_selected", "code", "tool/output verification", 1.0, 1.0, _grade_mode_selected),
        "artifact_report": Grader("artifact_report", "code", "artifact existence", 0.75, 1.0, _grade_report_artifact),
        "research_groundedness": Grader("research_groundedness", "code", "groundedness assertions", 1.25, 0.8, _grade_research_groundedness),
        "transcript_progress": Grader("transcript_progress", "code", "transcript analysis", 0.75, 1.0, _grade_transcript_progress),
        "model_report_rubric": Grader("model_report_rubric", "model", "deterministic rubric scoring", 0.8, 0.7, _grade_report_rubric),
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


def _grade_mode_selected(task: EvalTask, store: ArtifactStore) -> GraderResult:
    decisions = store.list("task_ingestion_decisions")
    selected = decisions[0].get("selected_mode") if decisions else None
    passed = selected == task.task_mode
    return _result("mode_selected", "code", "tool/output verification", 1.0 if passed else 0.0, passed, 1.0, f"Selected mode {selected}; expected {task.task_mode}.", [{"selected_mode": selected, "expected": task.task_mode, "passed": passed}])


def _grade_report_artifact(task: EvalTask, store: ArtifactStore) -> GraderResult:
    passed = store.report_path.exists() and len(store.report_path.read_text(encoding="utf-8").strip()) > 80
    return _result("artifact_report", "code", "artifact existence", 1.0 if passed else 0.0, passed, 0.75, "Final report artifact checked.", [{"path": str(store.report_path), "passed": passed}])


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
    has_steps = "Task 1:" in progress and len(progress.splitlines()) >= 5
    passed = has_complete and has_steps
    return _result(
        "transcript_progress",
        "code",
        "transcript analysis",
        1.0 if passed else 0.0,
        passed,
        0.75,
        f"Progress lines={len(progress.splitlines())}; traces={len(traces)}.",
        [{"check": "complete_marker", "passed": has_complete}, {"check": "step_visibility", "passed": has_steps}],
    )


def _grade_report_rubric(task: EvalTask, store: ArtifactStore) -> GraderResult:
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    rubric_checks = [
        ("has_summary", "summary" in report.lower() or "findings" in report.lower()),
        ("mentions_sources", "source" in report.lower()),
        ("mentions_uncertainty", any(term in report.lower() for term in ["uncertain", "caveat", "contradiction", "limitation"])),
        ("substantial_length", len(report.split()) >= 80),
    ]
    score = sum(1 for _, passed in rubric_checks if passed) / len(rubric_checks)
    passed = score >= 0.7
    return _result("model_report_rubric", "model", "deterministic rubric scoring", score, passed, 0.8, "Local model-style rubric scored the report.", [{"check": name, "passed": passed} for name, passed in rubric_checks])


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
    passed = best >= 0.5
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
    upstream_enabled = os.environ.get("PREDICTION_MARKET_USE_UPSTREAM") == "1"
    expected_measured = source == "upstream_orderbook_pm_challenge" if upstream_enabled else measured is False
    checks = [
        ("optimization_result_exists", store.optimization_result_path.exists()),
        ("official_result_present", isinstance(official, dict) and bool(official)),
        ("measured_status_truthful", bool(expected_measured)),
        ("score_source_recorded", source in {"local_official_semantics_fallback", "upstream_orderbook_pm_challenge"}),
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
    run_dirs = [path for path in store.root.parent.iterdir() if path.is_dir() and path.name.startswith("run_")]
    trial_root = store.root.parent.parent
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
