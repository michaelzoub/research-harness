from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from .llm import LLMClient
from .schemas import (
    AgentBudget,
    AgentTrace,
    Claim,
    Contradiction,
    Experiment,
    FailedPath,
    HarnessChange,
    Hypothesis,
    OpenQuestion,
    RunRecord,
    now_iso,
)
from .search import SearchBackend
from .store import ArtifactStore


class Agent(Protocol):
    name: str
    role: str
    prompt_template: str
    budget: AgentBudget

    async def run(self, run: RunRecord, store: ArtifactStore) -> str: ...


@dataclass
class AgentResult:
    agent_name: str
    summary: str
    errors: list[str]


class BaseAgent:
    def __init__(
        self,
        name: str,
        role: str,
        prompt_template: str,
        budget: Optional[AgentBudget] = None,
        llm: Optional[LLMClient] = None,
        model: Optional[str] = None,
    ):
        self.name = name
        self.role = role
        self.prompt_template = prompt_template
        self.budget = budget or AgentBudget()
        self.llm = llm or LLMClient()
        self.model = model or self.llm.model_label
        self.tool_calls: list[dict[str, object]] = []
        self.tools_used: list[str] = []

    async def execute(self, run: RunRecord, store: ArtifactStore) -> AgentResult:
        started = time.perf_counter()
        errors: list[str] = []
        status = "completed"
        summary = ""
        store.append_progress(f"Agent start: {self.name} ({self.role}) using {self.llm.model_label}")
        try:
            if self.budget.cancelled:
                status = "cancelled"
                summary = "Agent cancelled before execution."
            else:
                summary = await asyncio.wait_for(
                    self.run(run, store),
                    timeout=self.budget.max_runtime_seconds,
                )
        except Exception as exc:  # pragma: no cover - defensive trace path
            status = "failed"
            errors.append(f"{type(exc).__name__}: {exc}")
            summary = "Agent failed; see errors."
        runtime_ms = int((time.perf_counter() - started) * 1000)
        store.append_progress(f"Agent {status}: {self.name} in {runtime_ms}ms - {summary}")
        store.add_trace(
            AgentTrace(
                id=self.budget.trace_id,
                run_id=run.id,
                agent_name=self.name,
                role=self.role,
                prompt=self.prompt_template.replace("{goal}", run.user_goal),
                model=self.model,
                tools_used=self.tools_used,
                tool_calls=self.tool_calls,
                token_usage=_estimate_tokens(self.prompt_template + summary),
                runtime_ms=runtime_ms,
                status=status,
                errors=errors,
                output_summary=summary,
            )
        )
        return AgentResult(self.name, summary, errors)

    async def run(self, run: RunRecord, store: ArtifactStore) -> str:
        raise NotImplementedError


class LiteratureAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        query_angle: str,
        corpus: SearchBackend,
        prompt_template: str,
        budget: Optional[AgentBudget] = None,
        search_queries: Optional[list[str]] = None,
        llm: Optional[LLMClient] = None,
    ):
        super().__init__(name, "search_literature", prompt_template, budget, llm=llm)
        self.query_angle = query_angle
        self.corpus = corpus
        self.search_queries = search_queries or []

    async def run(self, run: RunRecord, store: ArtifactStore) -> str:
        self.tools_used.append(self.corpus.tool_name)
        queries = self.search_queries or [f"{run.user_goal} {self.query_angle}"]
        seen_urls: set[str] = set()
        all_results = []
        for query in queries:
            try:
                results = self.corpus.search(query, limit=self.budget.max_tool_calls)
            except Exception as exc:
                store.add_failed_path(
                    FailedPath(
                        description=f"Retriever failed for angle: {self.query_angle}",
                        reason=f"{self.corpus.tool_name} error: {type(exc).__name__}: {exc}",
                        created_by_agent=self.name,
                        run_id=run.id,
                    )
                )
                raise
            self.tool_calls.append({"tool": self.corpus.tool_name, "query": query, "results": len(results)})
            for document, relevance in results:
                if document.url in seen_urls:
                    continue
                seen_urls.add(document.url)
                all_results.append((document, relevance))
        claim_count = 0
        for document, relevance in all_results:
            source = store.add_source(self.corpus.to_source(document, relevance))
            for text in document.claims[: self.budget.max_steps]:
                confidence = round((source.credibility_score * 0.7) + (relevance * 0.3), 2)
                store.add_claim(
                    Claim(
                        text=text,
                        source_ids=[source.id],
                        confidence=confidence,
                        support_level=_support_level(confidence),
                        created_by_agent=self.name,
                        run_id=run.id,
                    )
                )
                claim_count += 1
        if not all_results:
            store.add_failed_path(
                FailedPath(
                    description=f"No sources found for angle: {self.query_angle}",
                    reason=f"{self.corpus.tool_name} returned no matching documents.",
                    created_by_agent=self.name,
                    run_id=run.id,
                )
            )
        return (
            f"Retrieved {len(all_results)} sources from {len(queries)} queries and extracted "
            f"{claim_count} claims for angle '{self.query_angle}'."
        )


class HypothesisAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        hypothesis_angle: str,
        prompt_template: str,
        budget: Optional[AgentBudget] = None,
        llm: Optional[LLMClient] = None,
    ):
        super().__init__(name, "hypothesis_generation", prompt_template, budget, llm=llm)
        self.hypothesis_angle = hypothesis_angle

    async def run(self, run: RunRecord, store: ArtifactStore) -> str:
        claims = _dedupe_by_text(store.list("claims"))
        relevant = [
            claim
            for claim in claims
            if any(term in claim["text"].lower() for term in self.hypothesis_angle.lower().split())
        ] or claims[: self.budget.max_steps]
        created = 0
        for index, claim in enumerate(relevant[: self.budget.max_steps]):
            hypothesis = store.add_hypothesis(
                Hypothesis(
                    text=f"{self.hypothesis_angle.title()} path: {claim['text']}",
                    supporting_claim_ids=[claim["id"]],
                    contradicting_claim_ids=claim.get("contradicted_by", []),
                    confidence=max(0.2, round(float(claim["confidence"]) - 0.1, 2)),
                    novelty_score=round(0.55 + (index * 0.08), 2),
                    testability_score=0.72,
                    next_experiment=f"Search for direct evaluations of: {self.hypothesis_angle}.",
                )
            )
            store.add_experiment(
                Experiment(
                    description=hypothesis.next_experiment,
                    hypothesis_id=hypothesis.id,
                    expected_signal="Evidence that increases or decreases claim confidence.",
                    priority=max(1, 3 - index),
                )
            )
            created += 1
        if not claims:
            store.add_open_question(
                OpenQuestion(
                    question=f"What evidence exists for {self.hypothesis_angle}?",
                    priority=1,
                    reason="No claims were available before hypothesis generation.",
                    created_by_agent=self.name,
                )
            )
        return f"Generated {created} hypotheses for angle '{self.hypothesis_angle}'."


class CriticAgent(BaseAgent):
    async def run(self, run: RunRecord, store: ArtifactStore) -> str:
        claims = _dedupe_by_text(store.list("claims"))
        hypotheses = store.list("hypotheses")
        contradictions = 0
        max_contradictions = 50
        for left in claims:
            for right in claims:
                if left["id"] >= right["id"]:
                    continue
                if _looks_contradictory(left["text"], right["text"]):
                    if contradictions >= max_contradictions:
                        break
                    store.add_contradiction(
                        Contradiction(
                            claim_a=left["id"],
                            claim_b=right["id"],
                            explanation="Claims appear directionally opposed and need source-level resolution.",
                            severity="medium",
                        )
                    )
                    contradictions += 1
            if contradictions >= max_contradictions:
                break
        if contradictions >= max_contradictions:
            store.add_open_question(
                OpenQuestion(
                    question="Which contradiction clusters matter most for the final conclusion?",
                    priority=1,
                    reason="The critic hit the contradiction cap and needs clustering/reranking before deeper review.",
                    created_by_agent=self.name,
                )
            )
        low_support = [claim for claim in claims if claim["confidence"] < 0.55]
        for claim in low_support[:3]:
            store.add_open_question(
                OpenQuestion(
                    question=f"What stronger evidence supports or refutes this claim: {claim['text']}",
                    priority=2,
                    reason="Claim confidence is below the recommended synthesis threshold.",
                    created_by_agent=self.name,
                )
            )
        if not hypotheses:
            store.add_open_question(
                OpenQuestion(
                    question="Which mechanisms or optimization paths should be explored next?",
                    priority=1,
                    reason="No hypotheses were generated from the current evidence set.",
                    created_by_agent=self.name,
                )
            )
        return f"Reviewed {len(claims)} claims and {len(hypotheses)} hypotheses; found {contradictions} contradictions."


class SynthesisAgent(BaseAgent):
    async def run(self, run: RunRecord, store: ArtifactStore) -> str:
        sources = _dedupe_by_id(store.list("sources"))
        claims = _dedupe_by_text(store.list("claims"))
        hypotheses = sorted(_dedupe_by_id(store.list("hypotheses")), key=lambda row: row["confidence"], reverse=True)
        contradictions = store.list("contradictions")
        questions = sorted(store.list("open_questions"), key=lambda row: row["priority"])
        seed_context = _read_optional_json(store.optimizer_seed_context_path)
        report = _build_report_with_llm(self.llm, run, sources, claims, hypotheses, contradictions, questions, seed_context)
        store.write_report(report)
        return f"Synthesized final report with {len(sources)} sources, {len(claims)} claims, and {len(hypotheses)} hypotheses."


class HarnessDebuggerAgent(BaseAgent):
    async def run(self, run: RunRecord, store: ArtifactStore) -> str:
        claims = store.list("claims")
        contradictions = store.list("contradictions")
        traces = store.list("agent_traces")
        if contradictions:
            change = HarnessChange(
                change="Add a contradiction-checking critic after each literature batch",
                reason="The run produced directionally opposed claims that were only reviewed after fan-in.",
                expected_effect="Earlier contradiction capture and better follow-up search targeting",
                risk="More runtime and repeated critic work during exploratory runs",
                evaluation="Compare unresolved contradiction count before and after the change",
                run_id=run.id,
            )
        elif len(claims) < 4:
            change = HarnessChange(
                change="Increase search-agent query diversity for low-yield goals",
                reason="The run produced too few claims to support robust synthesis.",
                expected_effect="Higher recall and more diverse evidence",
                risk="More noisy sources and higher review burden",
                evaluation="Compare high-confidence claim count and critic objection rate",
                run_id=run.id,
            )
        else:
            change = HarnessChange(
                change="Add plateau detection across consecutive research cycles",
                reason="The MVP currently stops after one cycle instead of measuring marginal research value.",
                expected_effect="Cleaner stopping decisions for longer open-ended research",
                risk="Requires extra bookkeeping and benchmark thresholds",
                evaluation="Track new-source and new-claim yield per cycle",
                run_id=run.id,
            )
        store.add_harness_change(change)
        failures = [trace for trace in traces if trace["status"] != "completed"]
        return f"Proposed 1 constrained harness change; observed {len(failures)} failed traces."


def _support_level(confidence: float) -> str:
    if confidence >= 0.75:
        return "strong"
    if confidence >= 0.55:
        return "moderate"
    return "weak"


def _looks_contradictory(left: str, right: str) -> bool:
    pairs = [
        ("increases", "decreases"),
        ("improves", "worsens"),
        ("supports", "undermines"),
        ("high", "low"),
        ("more", "less"),
    ]
    left_text = left.lower()
    right_text = right.lower()
    return any(a in left_text and b in right_text or b in left_text and a in right_text for a, b in pairs)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _dedupe_by_id(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    unique = []
    for row in rows:
        row_id = str(row["id"])
        if row_id not in seen:
            seen.add(row_id)
            unique.append(row)
    return unique


def _dedupe_by_text(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    unique = []
    for row in rows:
        key = str(row.get("text", row["id"])).strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _citation(source: dict[str, object]) -> str:
    return f"{source['title']} ({source['date']})"


def _build_report(
    run: RunRecord,
    sources: list[dict[str, object]],
    claims: list[dict[str, object]],
    hypotheses: list[dict[str, object]],
    contradictions: list[dict[str, object]],
    questions: list[dict[str, object]],
) -> str:
    source_lookup = {source["id"]: source for source in sources}
    lines = [
        f"# Research Report: {run.user_goal}",
        "",
        f"- Run ID: `{run.id}`",
        f"- Task type: `{run.task_type}`",
        f"- Completed: {now_iso()}",
        f"- Sources reviewed: {len(sources)}",
        f"- Claims extracted: {len(claims)}",
        f"- Hypotheses ranked: {len(hypotheses)}",
        "",
        "## Executive Synthesis",
        _executive_summary(claims, hypotheses, contradictions),
        "",
        "## Key Claims",
    ]
    for claim in sorted(claims, key=lambda row: row["confidence"], reverse=True):
        citations = ", ".join(_citation(source_lookup[source_id]) for source_id in claim["source_ids"])
        lines.append(f"- {claim['text']} Confidence: {claim['confidence']} ({claim['support_level']}). Sources: {citations}")
    lines.extend(["", "## Ranked Hypotheses"])
    for hypothesis in hypotheses:
        lines.append(
            f"- {hypothesis['text']} Confidence: {hypothesis['confidence']}; "
            f"novelty: {hypothesis['novelty_score']}; testability: {hypothesis['testability_score']}. "
            f"Next: {hypothesis['next_experiment']}"
        )
    lines.extend(["", "## Contradictions And Caveats"])
    if contradictions:
        for contradiction in contradictions:
            lines.append(
                f"- {contradiction['claim_a']} vs {contradiction['claim_b']}: "
                f"{contradiction['explanation']} Severity: {contradiction['severity']}."
            )
    else:
        lines.append("- No major contradictions were detected in this pass.")
    lines.extend(["", "## Open Questions"])
    if questions:
        for question in questions:
            lines.append(f"- P{question['priority']}: {question['question']} Reason: {question['reason']}")
    else:
        lines.append("- No open questions were created.")
    lines.extend(["", "## Sources"])
    for source in sources:
        lines.append(f"- [{source['title']}]({source['url']}) by {source['author']} ({source['date']})")
    return "\n".join(lines) + "\n"


def _build_report_with_llm(
    llm: LLMClient,
    run: RunRecord,
    sources: list[dict[str, object]],
    claims: list[dict[str, object]],
    hypotheses: list[dict[str, object]],
    contradictions: list[dict[str, object]],
    questions: list[dict[str, object]],
    seed_context: dict[str, object],
) -> str:
    deterministic_report = _build_report(run, sources, claims, hypotheses, contradictions, questions)
    seed_section = _optimizer_seed_section(seed_context)
    if not llm.is_live:
        return deterministic_report + seed_section
    payload = {
        "goal": run.user_goal,
        "run_id": run.id,
        "task_type": run.task_type,
        "task_mode": run.task_mode,
        "product_agent": run.product_agent,
        "sources": sources[:12],
        "claims": sorted(claims, key=lambda row: row["confidence"], reverse=True)[:24],
        "hypotheses": hypotheses[:12],
        "contradictions": contradictions[:12],
        "open_questions": questions[:12],
        "optimizer_seed_context": seed_context,
    }
    system = (
        "You are the synthesis agent in a research harness. Write a concise, evidence-grounded "
        "Markdown report. Do not invent citations. Use only source IDs and URLs present in the payload. "
        "Call out uncertainty and contradictions."
    )
    user = "Create the final report from this JSON payload:\n" + json.dumps(payload, indent=2, sort_keys=True)
    try:
        response = llm.complete(system, user, max_output_tokens=1800)
    except Exception as exc:
        fallback_note = f"\n\n## LLM Synthesis Fallback\n- Live synthesis failed: {type(exc).__name__}: {exc}\n"
        return deterministic_report + fallback_note
    if not response.text.strip():
        return deterministic_report + "\n\n## LLM Synthesis Fallback\n- Live synthesis returned no text.\n"
    return response.text.strip() + "\n" + seed_section


def _read_optional_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _optimizer_seed_section(seed_context: dict[str, object]) -> str:
    if not seed_context:
        return ""
    lines = [
        "",
        "## Optimizer Seed Context",
        f"- Has evaluator: {seed_context.get('has_evaluator')}",
        f"- Summary: {seed_context.get('summary', '')}",
    ]
    for item in seed_context.get("top_query_findings", []) if isinstance(seed_context.get("top_query_findings"), list) else []:
        if isinstance(item, dict):
            lines.append(f"- Query seed {item.get('variant_id')}: score {item.get('score')}; {item.get('query')}")
    return "\n".join(lines) + "\n"


def _executive_summary(
    claims: list[dict[str, object]],
    hypotheses: list[dict[str, object]],
    contradictions: list[dict[str, object]],
) -> str:
    if not claims:
        return "The run did not find enough evidence to produce a reliable synthesis."
    avg_confidence = sum(float(claim["confidence"]) for claim in claims) / len(claims)
    best = hypotheses[0]["text"] if hypotheses else "No ranked hypothesis emerged."
    caveat = "Contradictions require follow-up before acting." if contradictions else "No major contradiction was detected."
    return f"Evidence quality is {avg_confidence:.2f} on average. Leading direction: {best} {caveat}"
