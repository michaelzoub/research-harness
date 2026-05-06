from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from research_harness.benchmark import collect_runs, write_outputs
from research_harness.orchestrator import HarnessConfig, Orchestrator, goal_slug
from research_harness.search import OpenAlexSearch, _parse_arxiv_feed


class SmokeTest(unittest.TestCase):
    def test_phase2_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(mode="fanout", retriever="local"),
            )
            run, store = asyncio.run(
                orchestrator.run(
                    "Research how multi-agent systems improve automated literature review quality",
                    mode="fanout",
                )
            )

            self.assertEqual(run.status, "completed")
            self.assertTrue(store.report_path.exists())
            self.assertGreaterEqual(len(store.list("sources")), 2)
            self.assertGreaterEqual(len(store.list("claims")), 4)
            self.assertGreaterEqual(len(store.list("hypotheses")), 1)
            self.assertGreaterEqual(len(store.list("agent_traces")), 6)
            self.assertEqual(len(store.list("harness_changes")), 1)
            self.assertTrue(run.id.startswith("run_multi-agent-systems-improve-automated-literature-review-quality"))

    def test_duplicate_run_names_are_numbered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(mode="deterministic", retriever="local"),
            )
            first_run, _ = asyncio.run(orchestrator.run("Research agent memory systems", mode="deterministic"))
            second_run, _ = asyncio.run(orchestrator.run("Research agent memory systems", mode="deterministic"))

            self.assertEqual(first_run.id, "run_agent-memory-systems")
            self.assertEqual(second_run.id, "run_agent-memory-systems-02")

    def test_loop_mode_runs_nested_research_evolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(mode="loop", retriever="local", max_loop_iterations=3),
            )
            run, store = asyncio.run(
                orchestrator.run(
                    "Research how multi-agent systems improve automated literature review quality",
                    mode="loop",
                )
            )

            tasks = store.list("loop_tasks")
            iterations = store.list("loop_iterations")

            self.assertEqual(run.status, "completed")
            self.assertEqual(run.task_mode, "research")
            self.assertGreaterEqual(len(tasks), 5)
            self.assertTrue(all(task["passes"] for task in tasks))
            self.assertEqual(len(iterations), len(tasks))
            self.assertEqual(store.list("task_ingestion_decisions")[0]["selected_mode"], "research")
            self.assertGreaterEqual(len(store.list("variants")), 1)
            self.assertGreaterEqual(len(store.list("variant_evaluations")), 1)
            self.assertGreaterEqual(len(store.list("evolution_rounds")), 1)
            self.assertTrue(store.report_path.exists())
            self.assertTrue(store.run_benchmark_path.exists())
            self.assertTrue(store.decision_dag_path.exists())
            self.assertTrue((store.root / "run_benchmark_summary.json").exists())
            self.assertIn("<promise>COMPLETE</promise>", store.progress_path.read_text(encoding="utf-8"))

    def test_loop_mode_can_route_to_optimize_with_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(
                    mode="loop",
                    retriever="local",
                    max_loop_iterations=2,
                    task_mode="optimize",
                    evaluator_name="length_score",
                    include_debugger=False,
                ),
            )
            run, store = asyncio.run(orchestrator.run("Optimize a tiny scoring function", mode="loop"))

            self.assertEqual(run.status, "completed")
            self.assertEqual(run.task_mode, "optimize")
            self.assertEqual(store.list("task_ingestion_decisions")[0]["selected_mode"], "optimize")
            self.assertTrue(all(row["inner_loop"] == "optimize" for row in store.list("variant_evaluations")))
            self.assertTrue(all(task["passes"] for task in store.list("loop_tasks")))

    def test_goal_slug(self) -> None:
        self.assertEqual(
            goal_slug("Please research new agent paradigms on arxive and determine workplace trends"),
            "new-agent-paradigms-arxive-determine-workplace-trends",
        )


class BenchmarkTest(unittest.TestCase):
    def test_benchmark_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=root / "outputs",
                config=HarnessConfig(mode="fanout", retriever="local"),
            )
            asyncio.run(
                orchestrator.run(
                    "Research how multi-agent systems improve automated literature review quality",
                    mode="fanout",
                )
            )

            runs = collect_runs(root / "outputs")
            write_outputs(runs, root / "benchmarks")

            self.assertEqual(len(runs), 1)
            self.assertTrue((root / "benchmarks" / "index.html").exists())
            self.assertTrue((root / "benchmarks" / "summary.json").exists())
            self.assertTrue((root / "benchmarks" / "charts" / "artifact_counts.svg").exists())

class ArxivRetrieverTest(unittest.TestCase):
    def test_parse_arxiv_feed(self) -> None:
        payload = b"""<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/2601.00001v1</id>
            <updated>2026-01-01T00:00:00Z</updated>
            <published>2026-01-01T00:00:00Z</published>
            <title>Agent Paradigms For Workplace Automation</title>
            <summary>We introduce a benchmark for agentic workflows. Results suggest planner-executor systems improve reliability.</summary>
            <author><name>A. Researcher</name></author>
            <category term="cs.AI" />
          </entry>
        </feed>"""

        documents = _parse_arxiv_feed(payload)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].source_type, "arxiv_paper")
        self.assertEqual(documents[0].url, "http://arxiv.org/abs/2601.00001v1")
        self.assertIn("planner-executor", " ".join(documents[0].claims))

    def test_source_strategy_fans_out_for_general_research(self) -> None:
        orchestrator = Orchestrator(
            corpus_path=Path("examples/corpus/research_corpus.json"),
            output_root=Path("outputs"),
            config=HarnessConfig(retriever="auto"),
        )

        plan = orchestrator.create_plan("find research studying the human brain and artificial intelligence")
        strategy = orchestrator.create_source_strategy(
            "find research studying the human brain and artificial intelligence",
            plan,
        )

        self.assertGreaterEqual(len(strategy), 7)
        self.assertIn("openalex", {item.retriever for item in strategy})
        self.assertIn("arxiv", {item.retriever for item in strategy})
        self.assertIn("github", {item.retriever for item in strategy})
        self.assertIn("memory", {item.retriever for item in strategy})
        self.assertIn("brain", strategy[0].queries[0])
        self.assertIsInstance(orchestrator._retriever_for(strategy[0].retriever), OpenAlexSearch)


if __name__ == "__main__":
    unittest.main()
