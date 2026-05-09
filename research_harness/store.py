from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Optional

from .schemas import (
    AgentTrace,
    Claim,
    Contradiction,
    Experiment,
    FailedPath,
    HarnessChange,
    Hypothesis,
    LoopIteration,
    LoopTask,
    LoopContinuationDecision,
    OpenQuestion,
    RunRecord,
    Source,
    TaskIngestionDecision,
    Variant,
    VariantEvaluation,
    EvolutionRound,
    to_dict,
)


ENTITY_FILES = {
    "sources": "sources.json",
    "claims": "claims.json",
    "hypotheses": "hypotheses.json",
    "experiments": "experiments.json",
    "open_questions": "open_questions.json",
    "contradictions": "contradictions.json",
    "failed_paths": "failed_paths.json",
    "harness_changes": "harness_changes.json",
    "runs": "runs.json",
    "agent_traces": "agent_traces.json",
    "loop_tasks": "tasks.json",
    "loop_iterations": "loop_iterations.json",
    "task_ingestion_decisions": "task_ingestion_decisions.json",
    "variants": "variants.json",
    "variant_evaluations": "variant_evaluations.json",
    "evolution_rounds": "evolution_rounds.json",
    "loop_continuation_decisions": "loop_continuation_decisions.json",
}


class ArtifactStore:
    """File-backed shared artifact store for one workspace.

    The store writes canonical JSON collections plus a JSONL trace stream. It is
    intentionally small, deterministic, and easy to inspect.
    """

    def __init__(self, root: Path, echo_progress: bool = False, session_store: Optional[Any] = None):
        self.root = root
        self.echo_progress = echo_progress
        self.session_store = session_store
        self.root.mkdir(parents=True, exist_ok=True)
        for filename in ENTITY_FILES.values():
            path = self.root / filename
            if not path.exists():
                path.write_text("[]\n", encoding="utf-8")
        self.trace_log_path = self.root / "trace.jsonl"
        self.cost_path = self.root / "cost.json"
        self.report_path = self.root / "final_report.md"
        self.prd_path = self.root / "prd.json"
        self.optimizer_seed_context_path = self.root / "optimizer_seed_context.json"
        self.optimized_candidate_path = self.root / "optimized_candidate.txt"
        self.optimal_code_path = self.root / "optimal_code.py"
        self.candidates_dir = self.root / "candidates"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.optimization_result_path = self.root / "optimization_result.json"
        self.solution_path = self.root / "solution.py"
        self.run_benchmark_path = self.root / "run_benchmark.html"
        self.decision_dag_path = self.root / "decision_dag.png"
        self.agent_timeline_path = self.root / "agent_timeline.png"
        self.loop_continuation_path = self.root / "loop_continuation_decisions.json"
        self.progress_path = self.root / "progress.txt"
        if not self.progress_path.exists():
            self.progress_path.write_text("", encoding="utf-8")

    def add_source(self, source: Source) -> Source:
        # Primary dedup: exact URL match.
        existing = self.find_by("sources", "url", source.url)
        if existing:
            return Source(**existing)
        # Secondary dedup: normalized title match catches the same paper returned
        # by different retrievers with different URLs (e.g. arXiv vs OpenAlex).
        normalized = _normalize_title(source.title)
        if normalized:
            for row in self.list("sources"):
                if _normalize_title(str(row.get("title", ""))) == normalized:
                    return Source(**row)
        self._append("sources", source)
        return source

    def add_claim(self, claim: Claim) -> Claim:
        self._append("claims", claim)
        return claim

    def add_hypothesis(self, hypothesis: Hypothesis) -> Hypothesis:
        self._append("hypotheses", hypothesis)
        return hypothesis

    def add_experiment(self, experiment: Experiment) -> Experiment:
        self._append("experiments", experiment)
        return experiment

    def add_open_question(self, question: OpenQuestion) -> OpenQuestion:
        self._append("open_questions", question)
        return question

    def add_contradiction(self, contradiction: Contradiction) -> Contradiction:
        self._append("contradictions", contradiction)
        return contradiction

    def add_failed_path(self, failed_path: FailedPath) -> FailedPath:
        self._append("failed_paths", failed_path)
        return failed_path

    def add_harness_change(self, change: HarnessChange) -> HarnessChange:
        self._append("harness_changes", change)
        return change

    def add_run(self, run: RunRecord) -> RunRecord:
        self._append("runs", run)
        return run

    def add_loop_task(self, task: LoopTask) -> LoopTask:
        self._append("loop_tasks", task)
        return task

    def update_loop_task(self, task: LoopTask) -> None:
        rows = self.list("loop_tasks")
        for index, row in enumerate(rows):
            if row["id"] == task.id:
                rows[index] = to_dict(task)
                self._write("loop_tasks", rows)
                return
        self.add_loop_task(task)

    def add_loop_iteration(self, iteration: LoopIteration) -> LoopIteration:
        self._append("loop_iterations", iteration)
        return iteration

    def add_task_ingestion_decision(self, decision: TaskIngestionDecision) -> TaskIngestionDecision:
        self._append("task_ingestion_decisions", decision)
        return decision

    def add_variant(self, variant: Variant) -> Variant:
        self._append("variants", variant)
        return variant

    def add_variant_evaluation(self, evaluation: VariantEvaluation) -> VariantEvaluation:
        self._append("variant_evaluations", evaluation)
        return evaluation

    def add_evolution_round(self, round_record: EvolutionRound) -> EvolutionRound:
        self._append("evolution_rounds", round_record)
        return round_record

    def add_loop_continuation_decision(self, decision: LoopContinuationDecision) -> LoopContinuationDecision:
        self._append("loop_continuation_decisions", decision)
        return decision

    def append_progress(self, text: str) -> None:
        if self.echo_progress:
            print(_format_progress_for_terminal(text), flush=True)
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(text.rstrip() + "\n")
        if self.session_store is not None:
            self.session_store.append_event("progress", {"text": text.rstrip()})

    def update_run(self, run: RunRecord) -> None:
        rows = self.list("runs")
        for index, row in enumerate(rows):
            if row["id"] == run.id:
                rows[index] = to_dict(run)
                self._write("runs", rows)
                return
        self.add_run(run)

    def add_trace(self, trace: AgentTrace) -> AgentTrace:
        self._append("agent_traces", trace)
        payload = to_dict(trace)
        with self.trace_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        if self.session_store is not None:
            self.session_store.append_event("agent_trace", payload)
        return trace

    def list(self, entity: str) -> list[dict[str, Any]]:
        path = self.root / ENTITY_FILES[entity]
        return json.loads(path.read_text(encoding="utf-8"))

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {entity: self.list(entity) for entity in ENTITY_FILES}

    def write_report(self, text: str) -> Path:
        self._snapshot_before_write(self.report_path, "before writing final report")
        self.report_path.write_text(text, encoding="utf-8")
        self._record_artifact_write(self.report_path, "report")
        return self.report_path

    def write_prd(self, payload: dict[str, Any]) -> Path:
        self._snapshot_before_write(self.prd_path, "before writing PRD")
        self.prd_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._record_artifact_write(self.prd_path, "prd")
        return self.prd_path

    def write_optimizer_seed_context(self, payload: dict[str, Any]) -> Path:
        self._snapshot_before_write(self.optimizer_seed_context_path, "before writing optimizer seed context")
        self.optimizer_seed_context_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._record_artifact_write(self.optimizer_seed_context_path, "optimizer_seed_context")
        return self.optimizer_seed_context_path

    def write_solution(self, text: str) -> Path:
        self._snapshot_before_write(self.solution_path, "before writing solution code")
        self.solution_path.write_text(text, encoding="utf-8")
        self._record_artifact_write(self.solution_path, "solution")
        return self.solution_path

    def write_optimized_candidate(self, text: str) -> Path:
        self._snapshot_before_write(self.optimized_candidate_path, "before writing optimized candidate")
        self.optimized_candidate_path.write_text(text, encoding="utf-8")
        self._record_artifact_write(self.optimized_candidate_path, "optimized_candidate")
        return self.optimized_candidate_path

    def write_optimal_code(self, text: str) -> Path:
        self._snapshot_before_write(self.optimal_code_path, "before writing optimal code")
        self.optimal_code_path.write_text(text, encoding="utf-8")
        self._record_artifact_write(self.optimal_code_path, "optimal_code")
        return self.optimal_code_path

    def write_optimization_result(self, payload: dict[str, Any]) -> Path:
        self._snapshot_before_write(self.optimization_result_path, "before writing optimization result")
        self.optimization_result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._record_artifact_write(self.optimization_result_path, "optimization_result")
        return self.optimization_result_path

    def write_cost(self, payload: dict[str, Any]) -> Path:
        self._snapshot_before_write(self.cost_path, "before writing cost summary")
        self.cost_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._record_artifact_write(self.cost_path, "cost")
        return self.cost_path

    def find_by(self, entity: str, key: str, value: Any) -> Optional[dict[str, Any]]:
        return next((row for row in self.list(entity) if row.get(key) == value), None)

    def _append(self, entity: str, value: Any) -> None:
        rows = self.list(entity)
        rows.append(to_dict(value) if is_dataclass(value) else value)
        self._write(entity, rows)

    def _write(self, entity: str, rows: list[dict[str, Any]]) -> None:
        path = self.root / ENTITY_FILES[entity]
        path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _snapshot_before_write(self, path: Path, reason: str) -> None:
        if self.session_store is not None and path.exists():
            self.session_store.snapshot_files([path], reason=reason)

    def _record_artifact_write(self, path: Path, kind: str) -> None:
        if self.session_store is not None:
            self.session_store.append_event("artifact_write", {"kind": kind, "path": str(path)})


def _normalize_title(title: str) -> str:
    """Return a lowercase, punctuation-stripped title prefix for dedup comparison."""
    normalized = re.sub(r"[^a-z0-9]", "", title.lower())
    return normalized[:80]


def _format_progress_for_terminal(text: str) -> str:
    if os.environ.get("NO_COLOR") or os.environ.get("RESEARCH_HARNESS_COLOR") == "0":
        return text
    if not sys.stdout.isatty() and os.environ.get("RESEARCH_HARNESS_COLOR") != "1":
        return text
    reset = "\033[0m"
    bold = "\033[1m"
    dim = "\033[2m"
    red_italic = "\033[31;3m"
    cyan_bold = "\033[1;36m"
    green_bold = "\033[1;32m"
    yellow_bold = "\033[1;33m"
    magenta_bold = "\033[1;35m"

    lowered = text.lower()
    if any(term in lowered for term in ["error", "failed", "traceback", "fallback:", "http error", "timeout"]):
        return f"{red_italic}{text}{reset}"
    if text.startswith("# "):
        return f"{bold}{text}{reset}"
    if re.match(r"^(Starting run|Execution mode|Goal|PRD|Run:|Status|Artifacts):", text):
        return f"{cyan_bold}{text}{reset}"
    if re.match(r"^(Task \d+: passed|<promise>COMPLETE</promise>)", text):
        return f"{green_bold}{text}{reset}"
    if re.match(r"^(Optimization-query phase|Optimizer phase|Prediction-market optimizer round|Outer \d+:|Literature grounding|Literature refresh)", text):
        return f"{magenta_bold}{text}{reset}"
    if re.match(r"^(Retriever search|Retriever done|Optimized candidate|Optimal code|Solution|Optimization result|Report|Run benchmark|Decision DAG):", text):
        return f"{yellow_bold}{text}{reset}"
    if text.startswith("  ") or "LLM judge" in text or "thinking" in lowered:
        return f"{dim}{text}{reset}"
    return text
