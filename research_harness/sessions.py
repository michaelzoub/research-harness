from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .schemas import new_id, now_iso


def default_session_projects_dir() -> Path:
    return Path(os.environ.get("AUTORE_PROJECTS_DIR", Path.home() / ".autore" / "projects")).expanduser()


def project_key(workspace: Path) -> str:
    resolved = str(workspace.resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:10]
    slug = "".join(char if char.isalnum() else "-" for char in resolved.strip("/")).strip("-").lower()
    return f"{slug[:80]}-{digest}" if slug else digest


@dataclass
class SessionRecord:
    id: str
    project_dir: Path
    jsonl_path: Path
    metadata_path: Path


class SessionStore:
    """Plaintext JSONL session log under ~/.autore/projects/.

    Sessions are deliberately independent: each new session gets a fresh JSONL
    file and can orient from durable run artifacts instead of prior context.
    """

    def __init__(self, workspace: Path, projects_dir: Optional[Path] = None):
        self.workspace = workspace.resolve()
        self.projects_dir = (projects_dir or default_session_projects_dir()).expanduser()
        self.project_dir = self.projects_dir / project_key(self.workspace)
        self.sessions_dir = self.project_dir / "sessions"
        self.snapshots_dir = self.project_dir / "snapshots"
        self.metadata_dir = self.project_dir / "metadata"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.current: Optional[SessionRecord] = None

    def start_session(
        self,
        *,
        goal: str,
        run_id: str,
        output_dir: Path,
        resume_from: Optional[str] = None,
        fork_from: Optional[str] = None,
    ) -> SessionRecord:
        session_id = new_id("session")
        record = SessionRecord(
            id=session_id,
            project_dir=self.project_dir,
            jsonl_path=self.sessions_dir / f"{session_id}.jsonl",
            metadata_path=self.metadata_dir / f"{session_id}.json",
        )
        self.current = record
        metadata = {
            "id": session_id,
            "workspace": str(self.workspace),
            "run_id": run_id,
            "goal": goal,
            "output_dir": str(output_dir),
            "resume_from": resume_from,
            "fork_from": fork_from,
            "started_at": now_iso(),
            "completed_at": None,
            "session_jsonl": str(record.jsonl_path),
            "snapshots_dir": str(self.snapshots_dir / session_id),
        }
        record.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.append_event("session_start", metadata)
        return record

    def complete_session(self, *, status: str, summary: str = "") -> None:
        if self.current is None:
            return
        metadata = json.loads(self.current.metadata_path.read_text(encoding="utf-8"))
        metadata["completed_at"] = now_iso()
        metadata["status"] = status
        metadata["summary"] = summary
        self.current.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.append_event("session_complete", {"status": status, "summary": summary})

    def append_event(self, kind: str, payload: dict[str, Any]) -> None:
        if self.current is None:
            return
        row = {
            "timestamp": now_iso(),
            "session_id": self.current.id,
            "event": kind,
            "payload": payload,
        }
        with self.current.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def snapshot_files(self, paths: Iterable[Path], *, reason: str) -> list[dict[str, str]]:
        if self.current is None:
            return []
        snapshot_root = self.snapshots_dir / self.current.id / new_id("snapshot")
        records: list[dict[str, str]] = []
        for path in paths:
            source = path.resolve()
            if not source.exists() or not source.is_file():
                continue
            try:
                relative = source.relative_to(self.workspace)
            except ValueError:
                relative = Path(source.name)
            destination = snapshot_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            records.append({"source": str(source), "snapshot": str(destination)})
        if records:
            self.append_event("snapshot", {"reason": reason, "files": records})
        return records

