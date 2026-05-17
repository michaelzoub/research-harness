from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .harness import EvaluationHarness
from .suites import SUITE_CHOICES, eval_suite_by_id, select_eval_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run prewritten research-harness eval suites.")
    parser.add_argument("--suite", choices=SUITE_CHOICES, default="core")
    parser.add_argument(
        "--eval",
        action="append",
        default=[],
        dest="eval_ids",
        help="Run only the selected eval id. May be repeated or comma-separated.",
    )
    parser.add_argument("--list", action="store_true", help="List eval ids for the selected suite and exit.")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("eval_outputs"))
    parser.add_argument("--corpus", type=Path, default=Path("examples/corpus/research_corpus.json"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        suite = select_eval_tasks(eval_suite_by_id(args.suite), args.eval_ids)
    except ValueError as exc:
        parser.error(str(exc))
    if args.list:
        print(f"Eval suite: {suite.name}")
        for task in suite.tasks:
            print(f"{task.id}\t{task.name}")
        return
    suite.trials_per_task = args.trials
    summary = asyncio.run(EvaluationHarness(corpus_path=args.corpus, output_root=args.output).run_suite(suite))
    print(f"Eval suite: {summary.suite_name}")
    print(f"Trials: {summary.passed_trials}/{summary.trial_count} passed")
    print(f"Aggregate score: {summary.aggregate_score:.3f}")
    print(f"Summary: {args.output / (suite.id + '_summary.json')}")


if __name__ == "__main__":
    main()
