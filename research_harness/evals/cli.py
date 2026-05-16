from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .harness import EvaluationHarness
from .suites import all_eval_suite, default_eval_suite, edge_eval_suite


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
