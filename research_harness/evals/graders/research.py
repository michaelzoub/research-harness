from __future__ import annotations

import html
import json
import re

from ...store import ArtifactStore
from ..types import EvalTask, GraderResult
from .common import _result


TOOL_SOURCE_FAMILIES = {
    "alchemy_blockchain_search": "alchemy",
    "arxiv_api_search": "arxiv",
    "docs_blogs_search": "docs_blogs",
    "github_repo_search": "github",
    "local_corpus_search": "local",
    "openalex_api_search": "openalex",
    "prior_artifact_memory_search": "memory",
    "semantic_scholar_api_search": "semantic_scholar",
    "social_web_search": "social",
    "web_search": "web",
    "wikipedia_search": "wikipedia",
}


def _grade_research_source_diversity(task: EvalTask, store: ArtifactStore) -> GraderResult:
    traces = store.list("agent_traces")
    sources = store.list("sources")
    min_families = int(task.metadata.get("min_distinct_source_families", 4))
    tool_names = _called_tool_names(traces)
    families = sorted({_tool_source_family(tool_name) for tool_name in tool_names})
    passed = len(families) >= min_families
    score = min(1.0, len(families) / max(min_families, 1))
    return _result(
        "research_source_diversity",
        "code",
        "tool/API source-family superset check",
        score,
        passed,
        1.0,
        f"Research called {len(families)} distinct source family/families from {len(tool_names)} tool call(s); retained {len(sources)} source artifact(s).",
        [
            {
                "check": "min_distinct_tool_sources",
                "actual": len(families),
                "expected_at_least": min_families,
                "families": families,
                "tools": sorted(tool_names),
                "passed": passed,
            },
            {
                "check": "source_artifacts_present",
                "actual": len(sources),
                "expected_at_least": min_families,
                "passed": len(sources) >= min_families,
            },
        ],
    )


def _called_tool_names(traces: list[dict[str, object]]) -> set[str]:
    tool_names: set[str] = set()
    for trace in traces:
        for tool in trace.get("tools_used", []) if isinstance(trace.get("tools_used"), list) else []:
            clean = str(tool).strip()
            if clean:
                tool_names.add(clean)
        calls = trace.get("tool_calls", [])
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            clean = str(call.get("tool", "")).strip()
            if clean:
                tool_names.add(clean)
    return tool_names


def _tool_source_family(tool_name: str) -> str:
    return TOOL_SOURCE_FAMILIES.get(tool_name, tool_name.removesuffix("_search").removesuffix("_api"))


def _grade_report_no_fabricated_sources(task: EvalTask, store: ArtifactStore) -> GraderResult:
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    tex_path = getattr(store, "report_tex_path", store.root / "final_report.tex")
    tex = tex_path.read_text(encoding="utf-8") if tex_path.exists() else ""
    sources = store.list("sources")
    known_urls = {str(s.get("url", "")) for s in sources}
    report_urls = _cited_report_urls(report, tex)
    fabricated = []
    is_prediction_market_task = _is_prediction_market_eval_task(task)
    for url in report_urls:
        if _is_placeholder_report_url(url):
            fabricated.append({"url": url, "reason": "placeholder/example domain"})
        elif not is_prediction_market_task and _is_prediction_market_report_url(url):
            fabricated.append({"url": url, "reason": "prediction-market challenge source in non-challenge report"})
        elif url not in known_urls:
            fabricated.append({"url": url, "reason": "not in sources.json"})
    if not is_prediction_market_task and _references_prediction_market_challenge(report + "\n" + tex):
        fabricated.append({"url": "report text", "reason": "prediction-market challenge reference in non-challenge report"})
    passed = not fabricated
    score = max(0.0, 1.0 - len(fabricated) * 0.25)
    return _result(
        "report_no_fabricated_sources",
        "code",
        "source URL verification",
        score,
        passed,
        1.0,
        f"Found {len(fabricated)} fabricated source URL(s) in report out of {len(report_urls)} cited.",
        [{"check": "no_fabricated_sources", "passed": passed, "fabricated_urls": fabricated}],
    )


def _cited_report_urls(report: str, tex: str) -> list[str]:
    urls = re.findall(r"\]\(([^)]+)\)", report)
    urls.extend(re.findall(r"\\url\{([^}]+)\}", tex))
    deduped = []
    seen = set()
    for url in urls:
        clean = html.unescape(url).strip()
        if clean and clean not in seen:
            seen.add(clean)
            deduped.append(clean)
    return deduped


def _is_placeholder_report_url(url: str) -> bool:
    lowered = url.lower()
    return any(domain in lowered for domain in ["example.org", "example.com", "example.net", "example.invalid"])


def _is_prediction_market_report_url(url: str) -> bool:
    lowered = url.lower().replace("-", "_")
    return (
        "challenges/prediction_market" in lowered
        or "prediction_market/evaluator.py" in lowered
        or "prediction_market/spec.md" in lowered
        or "danrobinson/prediction_market_challenge" in lowered
    )


def _references_prediction_market_challenge(text: str) -> bool:
    lowered = text.lower().replace("-", "_")
    return any(
        phrase in lowered
        for phrase in [
            "prediction market strategy design notes",
            "prediction market local evaluator rubric",
            "orderbook prediction market challenge",
            "challenges/prediction_market",
            "danrobinson/prediction_market_challenge",
        ]
    )


def _is_prediction_market_eval_task(task: EvalTask) -> bool:
    normalized = task.prompt.lower().replace("-", " ")
    return task.evaluator_name == "prediction_market" or ("prediction" in normalized and "market" in normalized)


def _grade_research_groundedness(task: EvalTask, store: ArtifactStore) -> GraderResult:
    sources = store.list("sources")
    claims = store.list("claims")
    grounded_claims = [claim for claim in claims if claim.get("source_ids")]
    source_score = min(1.0, len(sources) / 4)
    claim_score = min(1.0, len(claims) / 8)
    grounded_score = len(grounded_claims) / max(len(claims), 1)
    score = (source_score * 0.3) + (claim_score * 0.3) + (grounded_score * 0.4)
    passed = score >= 0.8
    return _result(
        "research_groundedness",
        "code",
        "groundedness assertions",
        score,
        passed,
        1.25,
        f"{len(sources)} sources, {len(claims)} claims, {len(grounded_claims)} grounded claims.",
        [
            {"check": "min_sources", "actual": len(sources), "expected_at_least": 4, "passed": len(sources) >= 4},
            {"check": "min_claims", "actual": len(claims), "expected_at_least": 8, "passed": len(claims) >= 8},
            {"check": "all_claims_have_sources", "actual": len(grounded_claims), "total": len(claims), "passed": grounded_score == 1.0},
        ],
    )


def _grade_literature_section_evidence(task: EvalTask, store: ArtifactStore) -> GraderResult:
    sources = store.list("sources")
    paper_sources = [source for source in sources if "paper" in str(source.get("source_type", "")).lower() or "work" in str(source.get("source_type", "")).lower()]
    sectioned = []
    for source in paper_sources:
        sections = source.get("evidence_sections") if isinstance(source.get("evidence_sections"), dict) else {}
        present = [name for name in ["abstract", "introduction", "conclusion"] if str(sections.get(name, "")).strip()]
        if present:
            sectioned.append({"source_id": source.get("id"), "title": source.get("title"), "sections": present})
    ratio = len(sectioned) / max(len(paper_sources), 1)
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    report_mentions_basis = "evidence basis" in report.lower() or "paper context" in report.lower()
    passed = bool(paper_sources) and ratio >= 0.8 and report_mentions_basis
    score = (ratio * 0.75) + (0.25 if report_mentions_basis else 0.0)
    return _result(
        "literature_section_evidence",
        "code",
        "paper-section evidence verification",
        score,
        passed,
        1.0,
        f"{len(sectioned)}/{len(paper_sources)} paper-like sources included abstract/introduction/conclusion evidence sections.",
        [
            {"check": "paper_sources_present", "actual": len(paper_sources), "passed": bool(paper_sources)},
            {"check": "sectioned_paper_ratio", "ratio": round(ratio, 3), "passed": ratio >= 0.8, "examples": sectioned[:6]},
            {"check": "report_mentions_evidence_basis", "passed": report_mentions_basis},
        ],
    )


def _grade_hypothesis_evidence_matrix(task: EvalTask, store: ArtifactStore) -> GraderResult:
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    lower_report = report.lower()
    hypotheses = store.list("hypotheses")
    claims_by_id = {str(claim.get("id")): claim for claim in store.list("claims")}
    supported = [
        hypothesis
        for hypothesis in hypotheses
        if any(str(claim_id) in claims_by_id for claim_id in hypothesis.get("supporting_claim_ids", []))
    ]
    challenged = [
        hypothesis
        for hypothesis in hypotheses
        if hypothesis.get("contradicting_claim_ids") or "counterpoint" in lower_report or "limitation" in lower_report or "contradiction" in lower_report
    ]
    matrix_in_report = "hypothesis evidence matrix" in lower_report and "proof:" in lower_report and "counterpoint:" in lower_report
    support_ratio = len(supported) / max(len(hypotheses), 1)
    challenge_ratio = len(challenged) / max(len(hypotheses), 1)
    score = (support_ratio * 0.4) + (challenge_ratio * 0.3) + (0.3 if matrix_in_report else 0.0)
    passed = bool(hypotheses) and score >= 0.8
    return _result(
        "hypothesis_evidence_matrix",
        "code",
        "hypothesis proof/counterpoint verification",
        score,
        passed,
        1.0,
        f"{len(supported)}/{len(hypotheses)} hypotheses have retained proof claims; {len(challenged)}/{len(hypotheses)} have counterpoint or limitation handling.",
        [
            {"check": "hypotheses_present", "actual": len(hypotheses), "passed": bool(hypotheses)},
            {"check": "supporting_claim_ratio", "ratio": round(support_ratio, 3), "passed": support_ratio >= 0.8},
            {"check": "counterpoint_or_limitation_ratio", "ratio": round(challenge_ratio, 3), "passed": challenge_ratio >= 0.8},
            {"check": "report_contains_evidence_matrix", "passed": matrix_in_report},
        ],
    )


def _grade_transcript_progress(task: EvalTask, store: ArtifactStore) -> GraderResult:
    progress = store.progress_path.read_text(encoding="utf-8") if store.progress_path.exists() else ""
    traces = store.list("agent_traces")
    has_complete = "<promise>COMPLETE</promise>" in progress
    has_incomplete_stop = "Stopped with" in progress and "incomplete loop tasks" in progress
    has_steps = "Task 1:" in progress and len(progress.splitlines()) >= 5
    passed = (has_complete or has_incomplete_stop) and has_steps
    return _result(
        "transcript_progress",
        "code",
        "transcript analysis",
        1.0 if passed else 0.0,
        passed,
        0.75,
        f"Progress lines={len(progress.splitlines())}; traces={len(traces)}.",
        [
            {"check": "complete_or_incomplete_stop_marker", "passed": has_complete or has_incomplete_stop},
            {"check": "step_visibility", "passed": has_steps},
        ],
    )


def _grade_report_rubric(task: EvalTask, store: ArtifactStore) -> GraderResult:
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    evaluations = store.list("variant_evaluations")
    research_metrics = [
        row.get("metrics", {})
        for row in evaluations
        if row.get("inner_loop") == "research" and isinstance(row.get("metrics"), dict)
    ]
    rubric_dimensions = {"factual_accuracy", "citation_accuracy", "completeness", "source_quality", "tool_efficiency"}
    has_research_rubric_metrics = bool(research_metrics) and all(
        dimension in research_metrics[0] for dimension in rubric_dimensions
    )
    rubric_checks = [
        ("has_summary", "summary" in report.lower() or "findings" in report.lower()),
        ("mentions_sources", "source" in report.lower()),
        ("mentions_uncertainty", any(term in report.lower() for term in ["uncertain", "caveat", "contradiction", "limitation"])),
        ("has_research_rubric_metrics", has_research_rubric_metrics),
        ("substantial_length", len(report.split()) >= 80),
    ]
    score = sum(1 for _, passed in rubric_checks if passed) / len(rubric_checks)
    passed = score >= 0.7
    return _result("model_report_rubric", "model", "deterministic rubric scoring", score, passed, 0.8, "Local model-style rubric scored the report.", [{"check": name, "passed": passed} for name, passed in rubric_checks])


def _grade_llm_research_quality_challenger(task: EvalTask, store: ArtifactStore) -> GraderResult:
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    claims = store.list("claims")
    sources = store.list("sources")
    evaluations = store.list("variant_evaluations")
    research_metrics = [
        row.get("metrics", {})
        for row in evaluations
        if row.get("inner_loop") == "research" and isinstance(row.get("metrics"), dict)
    ]
    first_metrics = research_metrics[0] if research_metrics else {}
    grounded_claims = [claim for claim in claims if claim.get("source_ids")]
    dimensions = {
        "factual_accuracy": max(float(first_metrics.get("factual_accuracy", 0.0)), len(grounded_claims) / max(len(claims), 1)),
        "citation_accuracy": max(float(first_metrics.get("citation_accuracy", 0.0)), 1.0 if grounded_claims and len(grounded_claims) == len(claims) else 0.0),
        "completeness": max(float(first_metrics.get("completeness", 0.0)), min(1.0, len(report.split()) / 180.0)),
        "source_quality": max(float(first_metrics.get("source_quality", 0.0)), min(1.0, len(sources) / 4.0)),
    }
    score = sum(dimensions.values()) / len(dimensions)
    passed = score >= 0.7
    return _result(
        "llm_research_quality_challenger",
        "model",
        "LLM challenger research-quality rubric",
        score,
        passed,
        0.8,
        "Model-style challenger rated research quality across factual accuracy, citation accuracy, completeness, and source quality.",
        [{"dimension": name, "score": round(value, 3), "passed": value >= 0.7} for name, value in dimensions.items()],
    )


def _grade_llm_hypothesis_novelty_challenger(task: EvalTask, store: ArtifactStore) -> GraderResult:
    hypotheses = store.list("hypotheses")
    claims = {str(claim.get("text", "")).strip().lower() for claim in store.list("claims")}
    assertions: list[dict[str, Any]] = []
    scores: list[float] = []
    for hypothesis in hypotheses:
        text = str(hypothesis.get("text", "")).strip()
        novelty_score = float(hypothesis.get("novelty_score", 0.0) or 0.0)
        not_copy = text.lower() not in claims
        has_test = bool(hypothesis.get("next_experiment"))
        length_ok = len(text.split()) >= 6
        score = (novelty_score * 0.5) + (0.2 if not_copy else 0.0) + (0.2 if has_test else 0.0) + (0.1 if length_ok else 0.0)
        scores.append(min(1.0, score))
        assertions.append(
            {
                "hypothesis_id": hypothesis.get("id"),
                "question": "Is this hypothesis novel?",
                "novelty_score": novelty_score,
                "not_claim_copy": not_copy,
                "has_next_experiment": has_test,
                "passed": score >= 0.7,
            }
        )
    score = sum(scores) / max(len(scores), 1)
    passed = bool(hypotheses) and score >= 0.7
    return _result(
        "llm_hypothesis_novelty_challenger",
        "model",
        "LLM challenger hypothesis novelty rubric",
        score,
        passed,
        0.6,
        f"Model-style challenger judged novelty for {len(hypotheses)} hypothesis/hypotheses.",
        assertions or [{"check": "has_hypotheses", "passed": False}],
    )


def _grade_llm_open_ended_judgment_challenger(task: EvalTask, store: ArtifactStore) -> GraderResult:
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    progress = store.progress_path.read_text(encoding="utf-8") if store.progress_path.exists() else ""
    lower_report = report.lower()
    checks = [
        ("answers_user_prompt", any(term in lower_report for term in _keywords(task.prompt, limit=8))),
        ("uses_evidence_language", any(term in lower_report for term in ["source", "claim", "evidence", "citation"])),
        ("handles_uncertainty", any(term in lower_report for term in ["uncertain", "limitation", "caveat", "contradiction", "confidence"])),
        ("has_synthesis", any(term in lower_report for term in ["summary", "synthesis", "findings", "recommendation"])),
        ("run_reached_terminal_marker", "<promise>complete</promise>" in progress.lower() or "stopped with" in progress.lower()),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score >= 0.7
    return _result(
        "llm_open_ended_judgment_challenger",
        "model",
        "LLM challenger open-ended judgment",
        score,
        passed,
        0.6,
        "Model-style challenger made an open-ended judgment on relevance, evidence use, uncertainty, synthesis, and terminal progress.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _keywords(text: str, limit: int = 8) -> list[str]:
    stop = {"the", "and", "for", "with", "that", "this", "from", "into", "about", "research", "optimize", "how"}
    words = [word.lower() for word in re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", text) if word.lower() not in stop]
    unique: list[str] = []
    for word in words:
        if word not in unique:
            unique.append(word)
    return unique[:limit]


def _topic_keywords(text: str, limit: int = 10) -> list[str]:
    """Extract domain-specific topic terms, filtering generic verbs and filler words."""
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "about",
        "research", "optimize", "find", "make", "give", "take", "show", "have",
        "using", "used", "uses", "based", "also", "more", "some", "many", "most",
        "these", "those", "they", "their", "each", "when", "what", "which",
        "where", "there", "then", "than", "can", "could", "will", "would",
        "should", "shall", "must", "need", "want", "like", "just", "even",
        "only", "well", "very", "much", "such", "both", "after", "before",
        "over", "under", "other", "same", "different", "new", "long", "high",
        "data", "model", "models", "system", "systems", "method", "paper",
        "task", "tasks", "result", "results", "approach", "work",
    }
    words = [word.lower() for word in re.findall(r"[a-zA-Z][a-zA-Z-]{4,}", text) if word.lower() not in stop]
    unique: list[str] = []
    for word in words:
        if word not in unique:
            unique.append(word)
    return unique[:limit]


def _grade_prompt_output_relevance(task: EvalTask, store: ArtifactStore) -> GraderResult:
    """Check whether the report, claims, and sources are topically relevant to the original prompt."""
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    claims = store.list("claims")
    sources = store.list("sources")
    keywords = _topic_keywords(task.prompt, limit=10)
    if not keywords:
        return _result(
            "prompt_output_relevance", "code", "prompt-output topical relevance",
            0.0, False, 1.0,
            "Could not extract topic keywords from prompt.",
            [{"check": "keywords_extracted", "passed": False}],
        )
    lower_report = report.lower()
    report_hits = sum(1 for kw in keywords if kw in lower_report)
    report_ratio = report_hits / len(keywords)
    relevant_claims = [
        claim for claim in claims
        if any(kw in str(claim.get("text", "")).lower() for kw in keywords)
    ]
    claim_ratio = len(relevant_claims) / max(len(claims), 1)
    relevant_sources = [
        source for source in sources
        if any(kw in str(source.get("title", "")).lower() for kw in keywords)
    ]
    source_ratio = len(relevant_sources) / max(len(sources), 1)
    score = round((report_ratio * 0.4) + (claim_ratio * 0.4) + (source_ratio * 0.2), 3)
    passed = score >= 0.4
    return _result(
        "prompt_output_relevance",
        "code",
        "prompt-output topical relevance",
        score,
        passed,
        1.0,
        (
            f"Prompt keywords={keywords}; report={report_hits}/{len(keywords)} hits; "
            f"claims={len(relevant_claims)}/{len(claims)} relevant; "
            f"sources={len(relevant_sources)}/{len(sources)} relevant."
        ),
        [
            {"check": "report_keyword_ratio", "keywords": keywords, "hits": report_hits, "ratio": round(report_ratio, 3), "passed": report_ratio >= 0.4},
            {"check": "claim_relevance", "relevant": len(relevant_claims), "total": len(claims), "ratio": round(claim_ratio, 3), "passed": claim_ratio >= 0.4},
            {"check": "source_relevance", "relevant": len(relevant_sources), "total": len(sources), "ratio": round(source_ratio, 3), "passed": source_ratio >= 0.2},
        ],
    )
