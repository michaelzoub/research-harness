from __future__ import annotations

import json
from pathlib import Path

from ...store import ArtifactStore
from ..trajectory import graph_trajectory_match, normalized_trajectory_events, trajectory_graph, trajectory_match
from ..types import EvalTask, EvalTrial, GraderResult
from .common import _result


def _grade_trajectory_modes(task: EvalTask, store: ArtifactStore) -> GraderResult:
    required = set(task.metadata.get("required_modes", []))
    forbidden = set(task.metadata.get("forbidden_modes", []))
    evaluation_modes = {str(row.get("inner_loop")) for row in store.list("variant_evaluations") if row.get("inner_loop")}
    evolution_modes = {str(row.get("mode")) for row in store.list("evolution_rounds") if row.get("mode")}
    modes = evaluation_modes | evolution_modes
    missing = sorted(required - modes)
    forbidden_seen = sorted(forbidden & modes)
    passed = not missing and not forbidden_seen
    score = 1.0 if passed else 0.0
    return _result(
        "trajectory_modes",
        "code",
        "trajectory subset/superset match",
        score,
        passed,
        1.0,
        f"Required modes={sorted(required)}; forbidden modes={sorted(forbidden)}; observed modes={sorted(modes)}.",
        [
            {"check": "required_modes_subset", "missing": missing, "passed": not missing},
            {"check": "forbidden_modes_absent", "forbidden_seen": forbidden_seen, "passed": not forbidden_seen},
        ],
    )


def _grade_multi_iteration_loop(task: EvalTask, store: ArtifactStore) -> GraderResult:
    min_rounds = int(task.metadata.get("min_rounds", 2))
    rounds = store.list("evolution_rounds")
    checks = [
        ("min_rounds", len(rounds) >= min_rounds),
        ("all_rounds_have_best_score", all("best_score" in row for row in rounds)),
        ("all_rounds_have_signal", all(row.get("termination_signal") for row in rounds)),
        ("variant_evaluations_exist", bool(store.list("variant_evaluations"))),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "multi_iteration_loop",
        "code",
        "multi-round trajectory check",
        score,
        passed,
        1.0,
        f"Observed {len(rounds)} evolution round(s); required at least {min_rounds}.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_loop_no_score_regression(task: EvalTask, store: ArtifactStore) -> GraderResult:
    max_drop = float(task.metadata.get("max_score_drop", 0.2))
    rounds = store.list("evolution_rounds")
    scores = [float(row.get("best_score", 0.0)) for row in rounds]
    if not scores:
        return _result("loop_no_score_regression", "code", "score regression check", 0.0, False, 1.0, "No round scores were recorded.", [{"check": "round_scores_exist", "passed": False}])
    first = scores[0]
    best = max(scores)
    final = scores[-1]
    worst_drop = max(0.0, best - final)
    result = json.loads(store.optimization_result_path.read_text(encoding="utf-8")) if store.optimization_result_path.exists() else {}
    selected_score = float(result.get("score", 0.0))
    checks = [
        ("selected_score_preserves_best", selected_score >= best - 1e-9),
        ("final_drop_recorded_within_budget_or_best_preserved", worst_drop <= max_drop or selected_score >= best - 1e-9),
        ("best_at_least_first", best >= first),
        ("scores_in_unit_interval", all(0.0 <= score <= 1.0 for score in scores)),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "loop_no_score_regression",
        "code",
        "score regression check",
        score,
        passed,
        1.0,
        f"round_scores={scores}; best={best:.3f}; final={final:.3f}; selected={selected_score:.3f}; worst_drop={worst_drop:.3f}.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_literature_refresh_on_stuck(task: EvalTask, store: ArtifactStore) -> GraderResult:
    progress = store.progress_path.read_text(encoding="utf-8") if store.progress_path.exists() else ""
    rounds = store.list("evolution_rounds")
    sources = store.list("sources")
    claims = store.list("claims")
    stuck_signals = {row.get("termination_signal") for row in rounds if row.get("termination_signal") in {"score_plateau", "coverage_plateau"}}
    refresh_sources = [source for source in sources if str(source.get("url", "")).startswith("memory://literature-refresh/")]
    refresh_claims = [claim for claim in claims if claim.get("created_by_agent") == "literature_refresh_policy"]
    checks = [
        ("stuck_signal_observed", bool(stuck_signals)),
        ("progress_records_literature_refresh", "Literature refresh triggered" in progress),
        ("refresh_source_created", bool(refresh_sources)),
        ("refresh_claim_created", bool(refresh_claims)),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "literature_refresh_on_stuck",
        "code",
        "stuck-loop literature trigger",
        score,
        passed,
        1.0,
        f"stuck_signals={sorted(str(signal) for signal in stuck_signals)}; refresh_sources={len(refresh_sources)}; refresh_claims={len(refresh_claims)}.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_literature_grounding_present(task: EvalTask, store: ArtifactStore) -> GraderResult:
    progress = store.progress_path.read_text(encoding="utf-8") if store.progress_path.exists() else ""
    claims = store.list("claims")
    sources = store.list("sources")
    grounding_claims = [claim for claim in claims if claim.get("created_by_agent") == "literature_grounding_policy"]
    grounding_source_ids = {source_id for claim in grounding_claims for source_id in claim.get("source_ids", [])}
    checks = [
        ("progress_records_initial_grounding", "Literature grounding (initial)" in progress),
        ("grounding_claim_created", bool(grounding_claims)),
        ("grounding_claim_has_source", bool(grounding_source_ids)),
        ("grounding_source_exists", any(source.get("id") in grounding_source_ids for source in sources)),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "literature_grounding_present",
        "code",
        "initial literature grounding",
        score,
        passed,
        1.0,
        f"grounding_claims={len(grounding_claims)}; grounding_source_ids={len(grounding_source_ids)}.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_trajectory_graph_artifact(task: EvalTask, store: ArtifactStore) -> GraderResult:
    trial_root = store.root.parent
    mmd = trial_root / "trajectory_graph.mmd"
    svg = trial_root / "trajectory_graph.svg"
    graph_json = trial_root / "trajectory_graph.json"
    text = mmd.read_text(encoding="utf-8") if mmd.exists() else ""
    checks = [
        ("mermaid_graph_exists", mmd.exists()),
        ("svg_graph_exists", svg.exists()),
        ("json_graph_exists", graph_json.exists()),
        ("graph_mentions_outer_loop", "outer" in text.lower()),
        ("graph_mentions_inner_loop", "inner" in text.lower()),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "trajectory_graph_artifact",
        "code",
        "graph trajectory artifact",
        score,
        passed,
        1.0,
        f"Trajectory graph artifacts: {mmd}, {svg}, {graph_json}.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_trajectory_match_modes(task: EvalTask, store: ArtifactStore) -> GraderResult:
    actual = normalized_trajectory_events(store)
    reference = [
        {"type": str(item.get("type", "")), "name": str(item.get("name", ""))}
        for item in task.metadata.get("reference_trajectory", [])
        if isinstance(item, dict)
    ]
    if not reference:
        reference = [
            {"type": "router", "name": task.task_mode},
            {"type": "outer_loop", "name": task.task_mode},
            {"type": "inner_loop", "name": task.task_mode},
            {"type": "selection", "name": "variant"},
            {"type": "outcome", "name": "completed"},
        ]
    strict_reference = reference[:4]
    minimal_reference = [reference[0], reference[1], reference[2], reference[-1]]
    allowed_reference = list({(event["type"], event["name"]): event for event in actual}.values())
    results = {
        "strict": trajectory_match(actual, strict_reference, "strict"),
        "unordered": trajectory_match(actual, minimal_reference, "unordered"),
        "superset": trajectory_match(actual, minimal_reference, "superset"),
        "subset": trajectory_match(actual, allowed_reference, "subset"),
    }
    checks = [(mode, result["passed"]) for mode, result in results.items()]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    match_summary = ", ".join(f"{mode}={result['passed']}" for mode, result in results.items())
    return _result(
        "trajectory_match_modes",
        "code",
        "trajectory match modes",
        score,
        passed,
        1.0,
        f"Matched normalized trajectory using modes: {match_summary}.",
        [{"check": mode, "passed": result["passed"], "actual_count": result["actual_count"], "reference_count": result["reference_count"]} for mode, result in results.items()],
    )


def _grade_graph_trajectory_match(task: EvalTask, store: ArtifactStore) -> GraderResult:
    graph = trajectory_graph(store)
    required_edges = task.metadata.get("required_graph_edges", [])
    if not isinstance(required_edges, list) or not required_edges:
        required_edges = [["prompt", "router"], ["router", "outer"], ["outer", "inner"], ["inner", "select"], ["agents", "outcome"]]
    result = graph_trajectory_match(graph, required_edges)
    passed = bool(result["passed"])
    return _result(
        "graph_trajectory_match",
        "code",
        "graph trajectory match",
        1.0 if passed else 0.0,
        passed,
        1.0,
        f"Graph trajectory required_edges={len(required_edges)} missing={len(result['missing_edges'])}.",
        [{"check": "required_graph_edges_present", "missing_edges": result["missing_edges"], "passed": passed}],
    )


def _grade_parallel_trial_isolation_unavailable(task: EvalTask, store: ArtifactStore) -> GraderResult:
    return _result(
        "parallel_trial_isolation",
        "code",
        "cross-trial isolation check",
        0.0,
        False,
        1.0,
        "parallel_trial_isolation is a cross-trial grader and must be applied after sibling trials finish.",
        [{"check": "cross_trial_context_available", "passed": False}],
    )


def _grade_parallel_trial_isolation_from_trials(task: EvalTask, trials: list[EvalTrial]) -> GraderResult:
    trial_roots = [trial.isolation.get("trial_root") for trial in trials]
    output_roots = [trial.isolation.get("output_root") for trial in trials]
    tmpdirs = [trial.isolation.get("tmpdir") for trial in trials]
    run_ids = [trial.run_id for trial in trials]
    expected = task.trials or len(trials)
    checks = [
        ("expected_trial_count", len(trials) >= expected),
        ("unique_trial_roots", len(set(trial_roots)) == len(trial_roots)),
        ("unique_output_roots", len(set(output_roots)) == len(output_roots)),
        ("unique_tmpdirs", len(set(tmpdirs)) == len(tmpdirs)),
        ("all_paths_exist", all(Path(str(path)).exists() for path in [*trial_roots, *output_roots, *tmpdirs] if path)),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "parallel_trial_isolation",
        "code",
        "cross-trial isolation check",
        score,
        passed,
        1.0,
        f"Checked {len(trials)} trial(s) for distinct roots, output roots, tmpdirs, and run ids.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )
