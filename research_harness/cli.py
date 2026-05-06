from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from .orchestrator import HarnessConfig, Orchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the research harness MVP.")
    parser.add_argument("goal", help="High-level research goal.")
    parser.add_argument(
        "--mode",
        choices=["deterministic", "fanout"],
        default="fanout",
        help="Run Phase 1 deterministic or Phase 2 fan-out/fan-in.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(os.environ.get("RESEARCH_HARNESS_CORPUS_PATH", "examples/corpus/research_corpus.json")),
        help="Path to local deterministic search corpus.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("RESEARCH_HARNESS_OUTPUT_DIR", "outputs")),
        help="Directory where run artifacts are written.",
    )
    parser.add_argument(
        "--retriever",
        choices=["auto", "local", "arxiv", "openalex", "github", "web", "docs_blogs", "twitter", "memory"],
        default=os.environ.get("RESEARCH_HARNESS_RETRIEVER", "auto"),
        help="Evidence retriever/source mix. Auto uses a mixed strategy. Use local for the offline demo corpus.",
    )
    return parser


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    config = HarnessConfig(mode=args.mode, retriever=args.retriever)
    orchestrator = Orchestrator(args.corpus, args.output, config)
    run, store = asyncio.run(orchestrator.run(args.goal, mode=args.mode))
    print(f"Run: {run.id}")
    print(f"Status: {run.status}")
    print(f"Artifacts: {store.root}")
    print(f"Report: {store.report_path}")


if __name__ == "__main__":
    main()
