from __future__ import annotations

import asyncio
import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from challenges.prediction_market import prediction_market_score
from research_harness.benchmark import collect_runs, write_outputs
from research_harness.cli import build_parser, configure_interactive_run
from research_harness.evals import (
    EvaluationHarness,
    GraderResult,
    aggregate_results,
    default_eval_suite,
    edge_eval_suite,
    graph_trajectory_match,
    trajectory_match,
)
from research_harness.loops import EvaluatorRegistry, ResearchLoop, TaskRouter
from research_harness.orchestrator import HarnessConfig, Orchestrator, goal_slug
from research_harness.schemas import Claim, Contradiction, Hypothesis, RunRecord, Source, Variant
from research_harness.search import LocalCorpusSearch, OpenAlexSearch, _parse_arxiv_feed
from research_harness.sessions import SessionStore
from research_harness.store import ArtifactStore


class SmokeTest(unittest.TestCase):
    def test_cli_allows_selection_based_setup_without_goal(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])

        self.assertIsNone(args.goal)
        self.assertFalse(args.interactive)

    def test_interactive_cli_setup_collects_run_choices(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--interactive", "--retriever", "local"])
        text_answers = iter(
            [
                "Research agent workflow evaluation",
                "5",
            ]
        )
        keys = iter(["down", "down", "enter", "down", "enter", "enter"])
        prompts: list[str] = []

        with contextlib.redirect_stdout(io.StringIO()):
            configured = configure_interactive_run(
                args,
                input_func=lambda prompt: prompts.append(prompt) or next(text_answers),
                output_func=lambda _message: None,
                key_reader=lambda: next(keys),
            )

        self.assertEqual(configured.goal, "Research agent workflow evaluation")
        self.assertEqual(configured.task_mode, "optimize")
        self.assertEqual(configured.evaluator, "length_score")
        self.assertEqual(configured.retriever, "local")
        self.assertEqual(configured.max_iterations, 5)
        self.assertEqual(configured.llm_provider, "auto")
        self.assertFalse(configured.quiet)
        self.assertFalse(any("Choose a number" in prompt for prompt in prompts))

    def test_interactive_prediction_market_routes_to_challenge_mode(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--interactive", "--retriever", "local"])
        text_answers = iter(["optimizing prediction market making", "12"])
        keys = iter(["down", "down", "enter", "down", "down", "enter", "enter"])

        with contextlib.redirect_stdout(io.StringIO()):
            configured = configure_interactive_run(
                args,
                input_func=lambda _prompt: next(text_answers),
                output_func=lambda _message: None,
                key_reader=lambda: next(keys),
            )

        self.assertEqual(configured.task_mode, "optimize_query")
        self.assertEqual(configured.evaluator, "prediction_market")

    def test_prediction_market_evaluator_overrides_plain_optimize_to_challenge_flow(self) -> None:
        router = TaskRouter(EvaluatorRegistry())

        decision = router.decide(
            "optimizing the prediction market market making strategy, get to 10 dollars",
            requested_mode="optimize",
            evaluator_name="prediction_market",
        )

        self.assertEqual(decision.selected_mode, "optimize_query")
        self.assertEqual(decision.product_agent, "challenge")
        self.assertEqual(decision.evaluator_name, "prediction_market")

    def test_plan_uses_llm_interpretation_for_typoed_prediction_market_goal(self) -> None:
        class FakePlannerLLM:
            is_live = True

            def complete_json(self, _system: str, user: str, **_kwargs: object) -> dict[str, object]:
                self.user_payload = json.loads(user)
                return {
                    "task_type": "bounded",
                    "topics": ["prediction_market", "market_making"],
                    "topic_queries": [
                        "prediction market order book market making adverse selection",
                        "stale quote arbitrage retail order flow inventory skew",
                    ],
                    "rationale": "Interpreted typoed predictionm arket and mm'ing as prediction-market market making.",
                }

        orchestrator = Orchestrator(
            corpus_path=Path("examples/corpus/research_corpus.json"),
            output_root=Path("outputs"),
            config=HarnessConfig(retriever="auto"),
        )
        fake_llm = FakePlannerLLM()
        orchestrator.llm = fake_llm  # type: ignore[assignment]

        plan = orchestrator.create_plan("optimizing the predictionm arket mm'ing, get to 10$")
        strategy = orchestrator.create_source_strategy(plan.goal, plan)
        queries = " ".join(query for item in strategy for query in item.queries).lower()

        self.assertEqual(plan.planner, "llm")
        self.assertIn("prediction_market", plan.topics)
        self.assertIn("prediction market order book market making", queries)
        self.assertIn("selected_evaluator", fake_llm.user_payload)

    def test_phase2_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(mode="standard", retriever="local", echo_progress=False),
            )
            run, store = asyncio.run(
                orchestrator.run(
                    "Research how multi-agent systems improve automated literature review quality",
                    mode="standard",
                )
            )

            self.assertEqual(run.status, "completed")
            self.assertTrue(store.report_path.exists())
            self.assertGreaterEqual(len(store.list("sources")), 2)
            self.assertGreaterEqual(len(store.list("claims")), 4)
            self.assertGreaterEqual(len(store.list("hypotheses")), 1)
            self.assertGreaterEqual(len(store.list("agent_traces")), 6)
            self.assertEqual(len(store.list("harness_changes")), 1)
            self.assertTrue(run.id.startswith("001_run_multi-agent-systems-improve-automated-literature-review-quality"))
            self.assertTrue(store.prd_path.exists())
            self.assertGreaterEqual(len(json.loads(store.prd_path.read_text(encoding="utf-8"))["organized_tasks"]), 1)

    def test_world_model_dedup_provenance_and_observability_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            store_a = ArtifactStore(output_root / "001_run_world")
            run_a = RunRecord(
                id="001_run_world",
                user_goal="Research provenance",
                task_type="open_ended",
                harness_config_id="test-config",
                prompt_versions={"literature_agent": "abc123"},
                harness_config_snapshot={"mode": "test"},
            )
            store_a.add_run(run_a)
            source_a = store_a.add_source(
                Source(
                    url="https://example.com/paper",
                    title="World model paper",
                    author="Ada",
                    date="2026",
                    source_type="paper",
                    summary="A paper about persistent world models.",
                    relevance_score=0.9,
                    credibility_score=0.9,
                )
            )
            claim_a = store_a.add_claim(
                Claim(
                    text="Persistent stores improve cross-run memory.",
                    source_ids=[source_a.id],
                    confidence=0.8,
                    support_level="strong",
                    created_by_agent="test",
                    run_id=run_a.id,
                )
            )
            hypothesis_a = store_a.add_hypothesis(
                Hypothesis(
                    text="Cross-run dedupe should reduce repeated claims.",
                    supporting_claim_ids=[claim_a.id],
                    contradicting_claim_ids=[],
                    confidence=0.7,
                    novelty_score=0.6,
                    testability_score=0.8,
                    next_experiment="Run the same source twice.",
                )
            )
            store_a.add_contradiction(
                Contradiction(
                    claim_a=claim_a.id,
                    claim_b=claim_a.id,
                    explanation="Self-check edge for provenance coverage.",
                    severity="low",
                )
            )
            store_a.write_report("Report citing the persistent world model claim.\n")
            store_a.write_harness_diagnosis()

            store_b = ArtifactStore(output_root / "002_run_world")
            source_b = store_b.add_source(
                Source(
                    url="https://example.com/paper",
                    title="World model paper",
                    author="Ada",
                    date="2026",
                    source_type="paper",
                    summary="A paper about persistent world models.",
                    relevance_score=0.9,
                    credibility_score=0.9,
                )
            )
            claim_b = store_b.add_claim(
                Claim(
                    text="Persistent stores improve cross-run memory.",
                    source_ids=[source_b.id],
                    confidence=0.8,
                    support_level="strong",
                    created_by_agent="test",
                    run_id="002_run_world",
                )
            )

            self.assertEqual(source_b.duplicate_of, source_a.id)
            self.assertEqual(claim_b.duplicate_of, claim_a.id)
            self.assertTrue(store_a.sqlite_path.exists())
            self.assertTrue(store_a.harness_diagnosis_path.exists())
            self.assertGreaterEqual(len(store_a.list("provenance_edges")), 4)
            self.assertEqual(hypothesis_a.run_id, run_a.id)
            with sqlite3.connect(store_a.sqlite_path) as connection:
                artifact_count = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
                migration_count = connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
            self.assertGreaterEqual(artifact_count, 4)
            self.assertGreaterEqual(migration_count, 1)

    def test_duplicate_run_names_are_numbered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(mode="deterministic", retriever="local", echo_progress=False),
            )
            first_run, _ = asyncio.run(orchestrator.run("Research agent memory systems", mode="deterministic"))
            second_run, _ = asyncio.run(orchestrator.run("Research agent memory systems", mode="deterministic"))

            self.assertEqual(first_run.id, "001_run_agent-memory-systems")
            self.assertEqual(second_run.id, "002_run_agent-memory-systems")

    def test_sessions_are_plaintext_jsonl_and_snapshot_previous_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=root / "outputs",
                config=HarnessConfig(
                    mode="deterministic",
                    retriever="local",
                    session_projects_dir=root / "autore" / "projects",
                    echo_progress=False,
                ),
            )
            run, store = asyncio.run(orchestrator.run("Research agent session memory", mode="deterministic"))

            self.assertIsNotNone(run.session_id)
            self.assertIsNotNone(run.session_jsonl_path)
            session_jsonl = Path(run.session_jsonl_path or "")
            self.assertTrue(session_jsonl.exists())
            rows = [json.loads(line) for line in session_jsonl.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["event"], "session_start")
            self.assertTrue(any(row["event"] == "progress" for row in rows))
            self.assertTrue(any(row["event"] == "agent_trace" for row in rows))
            self.assertTrue(any(row["event"] == "artifact_write" for row in rows))
            self.assertTrue(any(row["event"] == "snapshot" for row in rows))
            prd = json.loads(store.prd_path.read_text(encoding="utf-8"))
            self.assertEqual(prd["artifacts"]["session_jsonl"], str(session_jsonl))

    def test_session_store_records_resume_and_fork_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root / "workspace", root / "autore" / "projects")
            record = store.start_session(
                goal="Optimize from prior artifacts",
                run_id="001_run_optimize",
                output_dir=root / "outputs" / "001_run_optimize",
                resume_from="session_old",
                fork_from="session_branch",
            )
            store.complete_session(status="completed", summary="ok")

            metadata = json.loads(record.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["resume_from"], "session_old")
            self.assertEqual(metadata["fork_from"], "session_branch")
            self.assertEqual(metadata["status"], "completed")
            events = [json.loads(line)["event"] for line in record.jsonl_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events, ["session_start", "session_complete"])

    def test_loop_mode_runs_nested_research_evolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(retriever="local", max_loop_iterations=3, echo_progress=False),
            )
            run, store = asyncio.run(
                orchestrator.run(
                    "Research how multi-agent systems improve automated literature review quality",
                )
            )

            tasks = store.list("loop_tasks")
            iterations = store.list("loop_iterations")

            self.assertEqual(run.status, "completed")
            self.assertEqual(run.task_mode, "research")
            self.assertEqual(run.product_agent, "research")
            self.assertGreaterEqual(len(tasks), 5)
            self.assertTrue(all(task["passes"] for task in tasks))
            self.assertEqual(len(iterations), len(tasks))
            self.assertEqual(store.list("task_ingestion_decisions")[0]["selected_mode"], "research")
            self.assertGreaterEqual(len(store.list("variants")), 1)
            self.assertGreaterEqual(len(store.list("variant_evaluations")), 1)
            self.assertGreaterEqual(len(store.list("evolution_rounds")), 1)
            self.assertEqual(len(store.list("loop_continuation_decisions")), len(store.list("evolution_rounds")))
            self.assertTrue(all(row["decision"] in {"continue", "exit"} for row in store.list("loop_continuation_decisions")))
            self.assertTrue(store.report_path.exists())
            self.assertTrue(store.prd_path.exists())
            prd = json.loads(store.prd_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(prd["organized_tasks"]), 5)
            self.assertTrue(prd["research_architecture"]["enabled_for_mode"])
            self.assertEqual(prd["research_architecture"]["lead_agent"]["role"], "lead_research_orchestrator")
            self.assertEqual(prd["research_architecture"]["subagents"]["role"], "parallel_research_subagents")
            self.assertIn("asyncio.gather", prd["research_architecture"]["subagents"]["parallelism"])
            self.assertEqual(
                {item["name"] for item in prd["research_architecture"]["judge_rubric"]},
                {"factual_accuracy", "citation_accuracy", "completeness", "source_quality", "tool_efficiency"},
            )
            research_evaluations = [row for row in store.list("variant_evaluations") if row["inner_loop"] == "research"]
            self.assertTrue(research_evaluations)
            for metric in ["factual_accuracy", "citation_accuracy", "completeness", "source_quality", "tool_efficiency"]:
                self.assertIn(metric, research_evaluations[0]["metrics"])
            self.assertTrue(store.run_benchmark_path.exists())
            self.assertTrue(store.decision_dag_path.exists())
            self.assertTrue(store.agent_timeline_path.exists())
            self.assertTrue((store.root / "run_benchmark_summary.json").exists())
            self.assertIn("<promise>COMPLETE</promise>", store.progress_path.read_text(encoding="utf-8"))

    def test_loop_mode_can_route_to_optimize_with_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(
                    retriever="local",
                    max_loop_iterations=2,
                    task_mode="optimize",
                    evaluator_name="length_score",
                    include_debugger=False,
                    echo_progress=False,
                ),
            )
            run, store = asyncio.run(orchestrator.run("Optimize a tiny scoring function"))

            self.assertEqual(run.status, "completed")
            self.assertEqual(run.task_mode, "optimize")
            self.assertEqual(run.product_agent, "optimize")
            self.assertEqual(store.list("task_ingestion_decisions")[0]["selected_mode"], "optimize")
            self.assertEqual(store.list("task_ingestion_decisions")[0]["product_agent"], "optimize")
            self.assertTrue(all(row["inner_loop"] == "optimize" for row in store.list("variant_evaluations")))
            self.assertTrue(all(task["passes"] for task in store.list("loop_tasks")))
            self.assertTrue(store.optimal_code_path.exists())
            optimization_result = json.loads(store.optimization_result_path.read_text(encoding="utf-8"))
            self.assertEqual(optimization_result["optimal_code_path"], str(store.optimal_code_path))

    def test_optimize_query_mode_feeds_optimizer_with_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(
                    retriever="local",
                    max_loop_iterations=1,
                    task_mode="optimize_query",
                    evaluator_name="length_score",
                    include_debugger=False,
                    echo_progress=False,
                ),
            )
            run, store = asyncio.run(orchestrator.run("Research optimization strategies for a tiny scoring benchmark"))

            seed_context = json.loads(store.optimizer_seed_context_path.read_text(encoding="utf-8"))
            inner_loops = {row["inner_loop"] for row in store.list("variant_evaluations")}
            prd = json.loads(store.prd_path.read_text(encoding="utf-8"))

            self.assertEqual(run.task_mode, "optimize_query")
            self.assertEqual(run.product_agent, "optimize")
            self.assertTrue(seed_context["has_evaluator"])
            self.assertIn("optimize_query", inner_loops)
            self.assertIn("optimize", inner_loops)
            query_evaluations = [row for row in store.list("variant_evaluations") if row["inner_loop"] == "optimize_query"]
            self.assertTrue(all("novelty" in row["metrics"] for row in query_evaluations))
            self.assertTrue(all("implementability" in row["metrics"] for row in query_evaluations))
            self.assertTrue(all("evaluator_relevance" in row["metrics"] for row in query_evaluations))
            query_variants = [row for row in store.list("variants") if row["kind"] == "query"]
            self.assertTrue(any(row["metadata"].get("evaluator_name") == "length_score" for row in query_variants))
            self.assertIn("optimizer_seed_context", prd["artifacts"])
            self.assertTrue(any(task["title"] == "Compile optimizer seed context" for task in prd["organized_tasks"]))

    def test_optimize_query_mode_without_evaluator_skips_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(
                    retriever="local",
                    max_loop_iterations=1,
                    task_mode="optimize_query",
                    include_debugger=False,
                    echo_progress=False,
                ),
            )
            run, store = asyncio.run(orchestrator.run("Research optimization strategies for a tiny scoring benchmark"))

            seed_context = json.loads(store.optimizer_seed_context_path.read_text(encoding="utf-8"))
            inner_loops = {row["inner_loop"] for row in store.list("variant_evaluations")}
            tasks = store.list("loop_tasks")

            self.assertEqual(run.task_mode, "optimize_query")
            self.assertEqual(run.product_agent, "optimize")
            self.assertFalse(seed_context["has_evaluator"])
            self.assertIn("optimize_query", inner_loops)
            self.assertNotIn("optimize", inner_loops)
            self.assertTrue(any(task["status"] == "skipped" for task in tasks))

    def test_prediction_market_evaluator_rewards_adaptive_strategy(self) -> None:
        static_ladder = "Static ladder around midpoint with size=12 spread=2 and no inventory controls."
        adaptive_guarded = (
            "Adaptive fair value estimate from fills and competitor midpoint, CancelAll after stale adverse "
            "arbitrageur fills or jump volatility, size=5 spread=4 inventory limit=90 with inventory skew."
        )

        self.assertGreater(prediction_market_score(adaptive_guarded), prediction_market_score(static_ladder))

    def test_research_loop_falls_back_to_local_when_live_retriever_fails(self) -> None:
        class FailingSearch:
            tool_name = "failing_search"

            def search(self, query: str, limit: int = 4):
                raise RuntimeError("rate limited")

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory), echo_progress=False)

            def search_factory(name: str):
                if name == "local":
                    return LocalCorpusSearch(Path("examples/corpus/research_corpus.json"))
                return FailingSearch()

            loop = ResearchLoop("run_test", search_factory)
            variant = Variant(
                run_id="run_test",
                outer_iteration=1,
                kind="query",
                payload="prediction market stale arbitrageur retail flow",
                parent_ids=[],
                metadata={"retriever": "arxiv", "limit": 8},
            )

            result = asyncio.run(loop.evaluate([variant], store))

            self.assertEqual(len(result.ranked_evaluations), 1)
            self.assertGreater(len(store.list("sources")), 0)
            self.assertGreater(len(store.list("failed_paths")), 0)
            self.assertEqual(store.list("variant_evaluations")[0]["metrics"]["fallback_used"], 1.0)

    def test_optimize_query_prediction_market_challenge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(
                    retriever="local",
                    max_loop_iterations=1,
                    task_mode="optimize_query",
                    evaluator_name="prediction_market",
                    include_debugger=False,
                    echo_progress=False,
                ),
            )
            run, store = asyncio.run(
                orchestrator.run(
                    "Research approaches for the prediction market challenge: adaptive passive market making against stale quote arbitrage and retail flow"
                )
            )

            seed_context = json.loads(store.optimizer_seed_context_path.read_text(encoding="utf-8"))
            progress = store.progress_path.read_text(encoding="utf-8")
            inner_loops = {row["inner_loop"] for row in store.list("variant_evaluations")}

            self.assertEqual(run.task_mode, "optimize_query")
            self.assertEqual(run.product_agent, "challenge")
            self.assertTrue(seed_context["has_evaluator"])
            self.assertIn("optimize_query", inner_loops)
            self.assertIn("optimize", inner_loops)
            self.assertIn("prediction_market", progress)
            self.assertTrue(store.optimized_candidate_path.exists())
            self.assertTrue(store.optimal_code_path.exists())
            self.assertTrue(store.optimization_result_path.exists())
            self.assertTrue(store.solution_path.exists())
            self.assertIn("class Strategy", store.solution_path.read_text(encoding="utf-8"))
            self.assertIn("class Strategy", store.optimal_code_path.read_text(encoding="utf-8"))
            optimization_result = json.loads(store.optimization_result_path.read_text(encoding="utf-8"))
            self.assertEqual(optimization_result["objective_direction"], "maximize")
            self.assertEqual(optimization_result["objective_name"], "prediction_market_mean_edge")
            self.assertEqual(optimization_result["optimal_code_path"], str(store.optimal_code_path))
            score_source = optimization_result["official_result"]["score_source"]
            measured = optimization_result["official_result"]["measured"]
            self.assertIn(score_source, {"local_sandbox_strategy_execution", "local_official_semantics_fallback", "upstream_orderbook_pm_challenge"})
            if score_source == "local_sandbox_strategy_execution":
                self.assertTrue(optimization_result["official_result"]["sandbox_executed"])
            self.assertEqual(optimization_result["official_result"]["simulations"], 200)
            # measured must be truthful: True iff the upstream runner was used.
            if score_source == "upstream_orderbook_pm_challenge":
                self.assertTrue(measured)
            else:
                self.assertFalse(measured)
            self.assertTrue(Path(optimization_result["official_result"]["candidate_path"]).exists())
            self.assertTrue(any("Prediction Market" in source["title"] for source in store.list("sources")))
            prd = json.loads(store.prd_path.read_text(encoding="utf-8"))
            self.assertEqual(prd["product_agent"], "challenge")
            self.assertEqual(prd["agent_harness"]["runtime_mode"], "optimize_query")

    def test_prediction_market_report_filters_unrelated_sources_for_typos(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(
                    retriever="local",
                    max_loop_iterations=1,
                    task_mode="optimize",
                    evaluator_name="prediction_market",
                    include_debugger=False,
                    echo_progress=False,
                    llm_provider="local",
                ),
            )
            run, store = asyncio.run(orchestrator.run("optimizing the predictionm arket mm'ing, get to 10$"))

            report = store.report_path.read_text(encoding="utf-8")

            self.assertEqual(run.task_mode, "optimize_query")
            self.assertEqual(run.product_agent, "challenge")
            self.assertIn("Prediction Market", report)
            self.assertIn("prediction_market", report)
            self.assertNotIn("Review of deep learning", report)
            self.assertNotIn("Brain Tumor", report)
            self.assertNotIn("G*Power", report)

    def test_prediction_market_dont_stop_profit_target_keeps_prd_incomplete_until_met(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(
                    retriever="local",
                    max_loop_iterations=3,
                    task_mode="optimize_query",
                    evaluator_name="prediction_market",
                    include_debugger=False,
                    echo_progress=False,
                ),
            )
            run, store = asyncio.run(
                orchestrator.run(
                    "Get to $10 profit in the prediction market challenge, don't stop until you're profitable."
                )
            )

            rounds = store.list("evolution_rounds")
            optimize_rounds = [row for row in rounds if row["mode"] == "optimize"]
            query_rounds = [row for row in rounds if row["mode"] == "optimize_query"]
            prd = json.loads(store.prd_path.read_text(encoding="utf-8"))
            optimization_result = json.loads(store.optimization_result_path.read_text(encoding="utf-8"))
            optimizer_task = next(task for task in prd["organized_tasks"] if task["title"] == "Run optimizer variants from query seed context")

            self.assertEqual(run.product_agent, "challenge")
            self.assertGreaterEqual(len(query_rounds), 2)
            self.assertEqual(len(optimize_rounds), 3)
            self.assertEqual(prd["objective"]["kind"], "profit_usd")
            self.assertEqual(prd["objective"]["target"], 10.0)
            self.assertFalse(prd["objective"]["met"])
            self.assertEqual(optimizer_task["status"], "failed")
            self.assertFalse(optimizer_task["passes"])
            self.assertEqual(optimization_result["objective_target"]["target"], 10.0)
            self.assertFalse(optimization_result["objective_target"]["met"])

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
                config=HarnessConfig(mode="standard", retriever="local", echo_progress=False),
            )
            asyncio.run(
                orchestrator.run(
                    "Research how multi-agent systems improve automated literature review quality",
                    mode="standard",
                )
            )

            runs = collect_runs(root / "outputs")
            write_outputs(runs, root / "benchmarks")

            self.assertEqual(len(runs), 1)
            self.assertTrue((root / "benchmarks" / "index.html").exists())
            self.assertTrue((root / "benchmarks" / "summary.json").exists())
            self.assertTrue((root / "benchmarks" / "charts" / "artifact_counts.svg").exists())


class EvaluationHarnessTest(unittest.TestCase):
    def test_core_eval_suite_defines_all_run_types(self) -> None:
        suite = default_eval_suite()
        task_ids = {task.id for task in suite.tasks}

        self.assertIn("research_open_ended", task_ids)
        self.assertIn("optimize_direct", task_ids)
        self.assertIn("optimize_query_seeded", task_ids)
        self.assertIn("challenge_prediction_market", task_ids)
        self.assertTrue(all(task.success_criteria for task in suite.tasks))
        self.assertTrue(all(task.grader_ids for task in suite.tasks))
        self.assertTrue(all("prd_tasks_executed" in task.grader_ids for task in suite.tasks))
        research_task = next(task for task in suite.tasks if task.id == "research_open_ended")
        self.assertIn("prd_tasks_executed_deterministic", research_task.grader_ids)
        self.assertIn("llm_research_quality_challenger", research_task.grader_ids)
        self.assertIn("llm_hypothesis_novelty_challenger", research_task.grader_ids)
        self.assertIn("llm_open_ended_judgment_challenger", research_task.grader_ids)
        with tempfile.TemporaryDirectory() as directory:
            registry = EvaluationHarness(output_root=Path(directory)).grader_registry
            self.assertEqual(registry["prd_tasks_executed_deterministic"].grader_type, "code")
            self.assertEqual(registry["llm_research_quality_challenger"].grader_type, "model")

    def test_edge_eval_suite_defines_failure_prone_cases(self) -> None:
        suite = edge_eval_suite()
        task_ids = {task.id for task in suite.tasks}

        self.assertIn("optimize_query_missing_evaluator_skips_optimizer", task_ids)
        self.assertIn("prediction_market_outputs_are_contained", task_ids)
        self.assertIn("prediction_market_unmeasured_official_status", task_ids)
        self.assertIn("challenge_prediction_market_official_unavailable_records_unmeasured", task_ids)
        self.assertIn("challenge_prediction_market_candidate_files_only_in_outputs", task_ids)
        self.assertIn("parallel_trials_do_not_share_tmp_or_outputs", task_ids)
        self.assertIn("challenge_prediction_market_no_repo_root_strategy_files", task_ids)
        self.assertIn("research_should_not_oversearch", task_ids)
        self.assertIn("nested_loop_multiple_iterations_no_regression", task_ids)
        self.assertIn("stuck_loop_triggers_literature_search", task_ids)
        self.assertIn("trajectory_match_modes_are_enforced", task_ids)
        self.assertIn("optimize_runs_start_with_literature_grounding", task_ids)
        self.assertTrue(any("trajectory_modes" in task.grader_ids for task in suite.tasks))
        self.assertTrue(any("prediction_market_artifact_containment" in task.grader_ids for task in suite.tasks))
        self.assertTrue(any("parallel_trial_isolation" in task.grader_ids for task in suite.tasks))
        self.assertTrue(any("research_search_budget" in task.grader_ids for task in suite.tasks))
        self.assertTrue(any("trajectory_graph_artifact" in task.grader_ids for task in suite.tasks))
        self.assertTrue(any("literature_refresh_on_stuck" in task.grader_ids for task in suite.tasks))
        self.assertTrue(any("literature_grounding_present" in task.grader_ids for task in suite.tasks))
        self.assertTrue(any("trajectory_match_modes" in task.grader_ids for task in suite.tasks))
        self.assertTrue(any("graph_trajectory_match" in task.grader_ids for task in suite.tasks))

    def test_native_trajectory_match_modes(self) -> None:
        actual = [
            {"type": "router", "name": "optimize"},
            {"type": "outer_loop", "name": "optimize"},
            {"type": "inner_loop", "name": "optimize"},
            {"type": "selection", "name": "variant"},
            {"type": "signal", "name": "score_plateau"},
            {"type": "outcome", "name": "completed"},
        ]
        reference = [
            {"type": "router", "name": "optimize"},
            {"type": "outer_loop", "name": "optimize"},
            {"type": "inner_loop", "name": "optimize"},
        ]

        self.assertTrue(trajectory_match(actual, reference, "strict")["passed"])
        self.assertTrue(trajectory_match(list(reversed(actual)), reference, "unordered")["passed"])
        self.assertTrue(trajectory_match(actual, reference + [{"type": "outcome", "name": "completed"}], "superset")["passed"])
        self.assertFalse(trajectory_match(actual, reference, "subset")["passed"])

    def test_graph_trajectory_match(self) -> None:
        graph = {"edges": [{"from": "prompt", "to": "router"}, {"from": "router", "to": "outer"}]}

        self.assertTrue(graph_trajectory_match(graph, [["prompt", "router"]])["passed"])
        self.assertFalse(graph_trajectory_match(graph, [["inner", "select"]])["passed"])

    def test_eval_harness_runs_prediction_market_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            suite = default_eval_suite()
            suite.tasks = [task for task in suite.tasks if task.id == "challenge_prediction_market"]
            summary = asyncio.run(
                EvaluationHarness(
                    corpus_path=Path("examples/corpus/research_corpus.json"),
                    output_root=Path(directory),
                ).run_suite(suite)
            )

            self.assertEqual(summary.trial_count, 1)
            self.assertEqual(summary.passed_trials, 1)
            self.assertGreaterEqual(summary.aggregate_score, 0.8)
            self.assertTrue((Path(directory) / "core_summary.json").exists())
            trial = summary.trials[0]
            isolation = trial["isolation"]
            self.assertTrue(isolation["clean_start"])
            self.assertTrue(Path(isolation["trial_root"]).exists())
            self.assertTrue(Path(isolation["tmpdir"]).exists())
            self.assertIn("Orchestrator", isolation["production_agent_path"])
            graders = {result["grader_id"]: result for result in trial["grader_results"]}
            self.assertIn("optimization_code_artifact", graders)
            self.assertTrue(graders["optimization_code_artifact"]["passed"])
            self.assertIn("prediction_market_solution", graders)
            self.assertTrue(graders["prediction_market_solution"]["passed"])
            self.assertIn("isolation_clean_trial", graders)
            self.assertTrue(graders["isolation_clean_trial"]["passed"])

    def test_eval_aggregation_modes(self) -> None:
        suite = default_eval_suite()
        task = suite.tasks[0]
        task.aggregation = "weighted"
        results = [
            GraderResult("a", "code", "exact", 1.0, True, 1.0, "pass", []),
            GraderResult("b", "model", "rubric", 0.5, False, 1.0, "partial", []),
        ]
        score, passed = aggregate_results(task, results)

        self.assertEqual(score, 0.75)
        self.assertFalse(passed)

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
        self.assertEqual(strategy[0].name, "broad_landscape")
        self.assertLessEqual(len(strategy[0].queries[0].split()), 4)
        self.assertIsInstance(orchestrator._retriever_for(strategy[0].retriever), OpenAlexSearch)

    def test_source_strategy_uses_prompt_domain_lenses(self) -> None:
        orchestrator = Orchestrator(
            corpus_path=Path("examples/corpus/research_corpus.json"),
            output_root=Path("outputs"),
            config=HarnessConfig(retriever="auto"),
        )

        goal = "Get to $10 profit in the prediction market challenge using AMM LMSR entropy literature"
        plan = orchestrator.create_plan(goal)
        strategy = orchestrator.create_source_strategy(goal, plan)
        queries = " ".join(query for item in strategy for query in item.queries).lower()

        self.assertIn("prediction-market", plan.strategy)
        self.assertIn("lmsr", queries)
        self.assertIn("adverse selection", queries)
        self.assertIn("orderbook", queries.replace(" ", ""))
        self.assertNotIn("workplace automation", queries)

    def test_prediction_market_evaluator_forces_prediction_market_lens(self) -> None:
        orchestrator = Orchestrator(
            corpus_path=Path("examples/corpus/research_corpus.json"),
            output_root=Path("outputs"),
            config=HarnessConfig(retriever="auto", evaluator_name="prediction_market"),
        )

        goal = "Optimize an image compression routine with a deterministic benchmark"
        plan = orchestrator.create_plan(goal)
        strategy = orchestrator.create_source_strategy(goal, plan)
        queries = " ".join(query for item in strategy for query in item.queries).lower()

        self.assertIn("prediction-market", plan.strategy)
        self.assertIn("market scoring rules", queries)
        self.assertIn("adverse selection", queries)

    def test_source_strategy_does_not_force_prediction_market_lens_without_prompt_or_evaluator(self) -> None:
        orchestrator = Orchestrator(
            corpus_path=Path("examples/corpus/research_corpus.json"),
            output_root=Path("outputs"),
            config=HarnessConfig(retriever="auto"),
        )

        goal = "Optimize an image compression routine with a deterministic benchmark"
        plan = orchestrator.create_plan(goal)
        strategy = orchestrator.create_source_strategy(goal, plan)
        queries = " ".join(query for item in strategy for query in item.queries).lower()

        self.assertNotIn("prediction-market", plan.strategy)
        self.assertNotIn("lmsr", queries)
        self.assertNotIn("adverse selection", queries)


class SkillSpecTest(unittest.TestCase):
    def test_repo_skills_follow_agent_skills_frontmatter_spec(self) -> None:
        skills_root = Path("skills")
        skill_dirs = sorted(path for path in skills_root.iterdir() if path.is_dir())

        self.assertGreaterEqual(len(skill_dirs), 1)
        for skill_dir in skill_dirs:
            skill_file = skill_dir / "SKILL.md"
            self.assertTrue(skill_file.exists(), f"{skill_dir} is missing SKILL.md")
            text = skill_file.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("---\n"), f"{skill_file} must start with YAML frontmatter")
            _, frontmatter, body = text.split("---", 2)
            fields = _simple_frontmatter(frontmatter)
            name = fields.get("name", "")
            description = fields.get("description", "")

            self.assertEqual(name, skill_dir.name)
            self.assertRegex(name, r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
            self.assertLessEqual(len(name), 64)
            self.assertTrue(description.strip(), f"{skill_file} description is required")
            self.assertLessEqual(len(description), 1024)
            self.assertGreater(len(body.strip()), 20)


def _simple_frontmatter(frontmatter: str) -> dict[str, str]:
    fields = {}
    for line in frontmatter.splitlines():
        if not line.strip() or line.startswith(" "):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip().strip('"').strip("'")
    return fields


if __name__ == "__main__":
    unittest.main()
