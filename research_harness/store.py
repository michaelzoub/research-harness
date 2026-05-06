from __future__ import annotations

import json
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
    OpenQuestion,
    RunRecord,
    Source,
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
}


class ArtifactStore:
    """File-backed shared artifact store for one workspace.

    The store writes canonical JSON collections plus a JSONL trace stream. It is
    intentionally small, deterministic, and easy to inspect.
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        for filename in ENTITY_FILES.values():
            path = self.root / filename
            if not path.exists():
                path.write_text("[]\n", encoding="utf-8")
        self.trace_log_path = self.root / "trace.jsonl"
        self.report_path = self.root / "final_report.md"

    def add_source(self, source: Source) -> Source:
        existing = self.find_by("sources", "url", source.url)
        if existing:
            return Source(**existing)
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
        return trace

    def list(self, entity: str) -> list[dict[str, Any]]:
        path = self.root / ENTITY_FILES[entity]
        return json.loads(path.read_text(encoding="utf-8"))

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {entity: self.list(entity) for entity in ENTITY_FILES}

    def write_report(self, text: str) -> Path:
        self.report_path.write_text(text, encoding="utf-8")
        return self.report_path

    def find_by(self, entity: str, key: str, value: Any) -> Optional[dict[str, Any]]:
        return next((row for row in self.list(entity) if row.get(key) == value), None)

    def _append(self, entity: str, value: Any) -> None:
        rows = self.list(entity)
        rows.append(to_dict(value) if is_dataclass(value) else value)
        self._write(entity, rows)

    def _write(self, entity: str, rows: list[dict[str, Any]]) -> None:
        path = self.root / ENTITY_FILES[entity]
        path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
