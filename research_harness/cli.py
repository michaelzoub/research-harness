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
        choices=["standard", "deterministic"],
        default=os.environ.get("RESEARCH_HARNESS_MODE"),
        help="Optional legacy mode. Omit for the default nested evolutionary agent loop. Use standard for old fan-out/fan-in.",
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
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=12,
        help="Maximum outer-loop iterations for the default evolutionary agent loop.",
    )
    parser.add_argument(
        "--task-mode",
        choices=["auto", "optimize", "research", "optimize_query"],
        default=os.environ.get("RESEARCH_HARNESS_TASK_MODE", "auto"),
        help="Task ingestion mode for the evolutionary agent loop. Auto uses evaluator availability and prompt heuristics.",
    )
    parser.add_argument(
        "--evaluator",
        default=os.environ.get("RESEARCH_HARNESS_EVALUATOR"),
        help="Registered deterministic evaluator name for optimize-mode tasks.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["auto", "openai", "local"],
        default=os.environ.get("RESEARCH_HARNESS_LLM_PROVIDER", "auto"),
        help="LLM provider for agent proposal, judging, and synthesis. Auto uses OpenAI when OPENAI_API_KEY is set.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("RESEARCH_HARNESS_LLM_MODEL", "gpt-5.2"),
        help="LLM model name used when the selected provider is live.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not stream run progress to the terminal; artifacts are still written.",
    )
    return parser


def load_dotenv(path: Path = Path(".env"), *, override: bool = False) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        clean_key = key.strip()
        clean_value = value.strip().strip('"').strip("'")
        if override:
            os.environ[clean_key] = clean_value
        else:
            os.environ.setdefault(clean_key, clean_value)


def main() -> None:
    load_dotenv()
    load_dotenv(Path(".env.local"), override=True)
    args = build_parser().parse_args()
    config = HarnessConfig(
        mode=args.mode or "evolutionary",
        retriever=args.retriever,
        max_loop_iterations=args.max_iterations,
        task_mode=args.task_mode,
        evaluator_name=args.evaluator,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        echo_progress=not args.quiet,
    )
    orchestrator = Orchestrator(args.corpus, args.output, config)
    run, store = asyncio.run(orchestrator.run(args.goal, mode=args.mode))
    print(f"Run: {run.id}")
    print(f"Status: {run.status}")
    print(f"Artifacts: {store.root}")
    print(f"PRD: {store.prd_path}")
    if store.optimizer_seed_context_path.exists():
        print(f"Optimizer seed context: {store.optimizer_seed_context_path}")
    if store.optimization_result_path.exists():
        print(f"Optimization result: {store.optimization_result_path}")
    if store.optimized_candidate_path.exists():
        print(f"Optimized candidate: {store.optimized_candidate_path}")
    if store.optimal_code_path.exists():
        print(f"Optimal code: {store.optimal_code_path}")
    if store.solution_path.exists():
        print(f"Solution: {store.solution_path}")
    print(f"Report: {store.report_path}")
    print(f"Run benchmark: {store.run_benchmark_path}")
    print(f"Decision DAG: {store.decision_dag_path}")


if __name__ == "__main__":
    main()
