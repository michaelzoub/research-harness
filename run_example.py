from __future__ import annotations

import asyncio
from pathlib import Path

from research_harness.orchestrator import HarnessConfig, Orchestrator


async def run() -> None:
    goal = "Research how multi-agent systems improve automated literature review quality"
    orchestrator = Orchestrator(
        corpus_path=Path("examples/corpus/research_corpus.json"),
        output_root=Path("outputs"),
        config=HarnessConfig(retriever="local"),
    )
    run_record, store = await orchestrator.run(goal)
    print(f"Created example run {run_record.id}")
    print(store.report_path)
    print(store.run_benchmark_path)


if __name__ == "__main__":
    asyncio.run(run())
