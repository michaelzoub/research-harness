from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from statistics import median
from typing import Callable, Optional, Protocol

from .llm import LLMClient
from .schemas import (
    Claim,
    EvolutionRound,
    SourceStrategyItem,
    TaskIngestionDecision,
    TaskMode,
    Variant,
    VariantEvaluation,
)
from .search import SearchBackend
from .store import ArtifactStore


EvaluatorFn = Callable[[str], float]
SearchFactory = Callable[[str], SearchBackend]


OPTIMIZE_HINTS = {
    "benchmark",
    "kernel",
    "latency",
    "optimize",
    "performance",
    "score",
    "speed",
    "strategy",
    "swe-bench",
    "throughput",
}


class EvaluatorRegistry:
    """Registry for deterministic evaluators.

    Optimize mode is only valid when a deterministic evaluator is available.
    The tiny built-in evaluator is for smoke tests and architecture demos; real
    tasks should register domain evaluators such as a benchmark harness.
    """

    def __init__(self) -> None:
        self._evaluators: dict[str, EvaluatorFn] = {
            "length_score": lambda payload: 1.0 / max(1, len(payload.split())),
        }

    def get(self, name: Optional[str]) -> Optional[EvaluatorFn]:
        if not name:
            return None
        return self._evaluators.get(name)

    def register(self, name: str, evaluator: EvaluatorFn) -> None:
        self._evaluators[name] = evaluator


class TaskRouter:
    def __init__(self, evaluator_registry: EvaluatorRegistry):
        self.evaluator_registry = evaluator_registry

    def decide(self, goal: str, requested_mode: str = "auto", evaluator_name: Optional[str] = None) -> TaskIngestionDecision:
        requested = requested_mode.lower()
        evaluator = self.evaluator_registry.get(evaluator_name)
        if requested == "research":
            return TaskIngestionDecision(
                requested_mode=requested_mode,
                selected_mode="research",
                reason="Research mode was explicitly requested.",
            )
        if requested == "optimize" and evaluator:
            return TaskIngestionDecision(
                requested_mode=requested_mode,
                selected_mode="optimize",
                evaluator_name=evaluator_name,
                reason=f"Optimize mode was explicitly requested with evaluator '{evaluator_name}'.",
            )
        if requested == "optimize" and not evaluator:
            return TaskIngestionDecision(
                requested_mode=requested_mode,
                selected_mode="research",
                evaluator_name=evaluator_name,
                reason="Optimize mode requested, but register_evaluator failed to resolve a deterministic evaluator; falling back to research mode.",
            )
        if evaluator:
            return TaskIngestionDecision(
                requested_mode=requested_mode,
                selected_mode="optimize",
                evaluator_name=evaluator_name,
                reason=f"Deterministic evaluator '{evaluator_name}' is registered.",
            )
        if any(hint in goal.lower() for hint in OPTIMIZE_HINTS):
            return TaskIngestionDecision(
                requested_mode=requested_mode,
                selected_mode="research",
                reason="The prompt looks optimization-shaped, but no deterministic evaluator was registered; using research mode.",
            )
        return TaskIngestionDecision(
            requested_mode=requested_mode,
            selected_mode="research",
            reason="No deterministic evaluator is available, so the task is routed to research mode.",
        )


@dataclass
class InnerLoopResult:
    ranked_evaluations: list[VariantEvaluation]
    termination_signal: str


class InnerLoop(Protocol):
    mode: TaskMode

    async def evaluate(self, variants: list[Variant], store: ArtifactStore) -> InnerLoopResult: ...


class OptimizeLoop:
    mode: TaskMode = "optimize"

    def __init__(self, run_id: str, evaluator: EvaluatorFn, pass_threshold: float = 0.8):
        self.run_id = run_id
        self.evaluator = evaluator
        self.pass_threshold = pass_threshold

    async def evaluate(self, variants: list[Variant], store: ArtifactStore) -> InnerLoopResult:
        evaluations = await asyncio.gather(*(self._evaluate_variant(variant) for variant in variants))
        for evaluation in evaluations:
            store.add_variant_evaluation(evaluation)
        ranked = sorted(evaluations, key=lambda item: item.score, reverse=True)
        signal = "score_threshold" if ranked and ranked[0].score >= self.pass_threshold else "continue"
        return InnerLoopResult(ranked_evaluations=ranked, termination_signal=signal)

    async def _evaluate_variant(self, variant: Variant) -> VariantEvaluation:
        raw_score = float(self.evaluator(variant.payload))
        score = max(0.0, min(1.0, raw_score))
        return VariantEvaluation(
            run_id=self.run_id,
            variant_id=variant.id,
            inner_loop="optimize",
            score=score,
            metrics={"deterministic_score": score},
            judge_scores=[score],
            summary=f"Deterministic evaluator returned {score:.3f}.",
            passed=score >= self.pass_threshold,
        )


class ResearchLoop:
    mode: TaskMode = "research"

    def __init__(self, run_id: str, search_factory: SearchFactory, llm: Optional[LLMClient] = None, pass_threshold: float = 0.78):
        self.run_id = run_id
        self.search_factory = search_factory
        self.llm = llm or LLMClient()
        self.pass_threshold = pass_threshold

    async def evaluate(self, variants: list[Variant], store: ArtifactStore) -> InnerLoopResult:
        evaluations = await asyncio.gather(*(self._evaluate_variant(variant, store) for variant in variants))
        for evaluation in evaluations:
            store.add_variant_evaluation(evaluation)
        ranked = sorted(evaluations, key=lambda item: item.score, reverse=True)
        signal = "claim_corroboration_threshold" if ranked and ranked[0].score >= self.pass_threshold else "continue"
        return InnerLoopResult(ranked_evaluations=ranked, termination_signal=signal)

    async def _evaluate_variant(self, variant: Variant, store: ArtifactStore) -> VariantEvaluation:
        retriever_name = str(variant.metadata.get("retriever", "local"))
        limit = int(variant.metadata.get("limit", 6))
        backend = self.search_factory(retriever_name)
        backend_results = backend.search(variant.payload, limit=limit)
        sources = []
        claim_count = 0
        for document, relevance in backend_results:
            source = store.add_source(backend.to_source(document, relevance))
            sources.append(source)
            for claim_text in document.claims[:3]:
                confidence = round((source.credibility_score * 0.7) + (relevance * 0.3), 2)
                store.add_claim(
                    Claim(
                        text=claim_text,
                        source_ids=[source.id],
                        confidence=confidence,
                        support_level=_support_level(confidence),
                        created_by_agent=f"research_loop:{variant.id}",
                        run_id=self.run_id,
                    )
                )
                claim_count += 1
        metrics = self._research_metrics(sources, claim_count)
        judge_scores = [
            metrics["coverage"],
            metrics["corroboration"],
            metrics["credibility"],
            _stable_judge_score(variant.payload, metrics),
        ]
        llm_score, llm_summary = self._llm_judge_score(variant, metrics, len(sources), claim_count)
        if llm_score is not None:
            judge_scores.append(llm_score)
        score = round(median(judge_scores), 3)
        return VariantEvaluation(
            run_id=self.run_id,
            variant_id=variant.id,
            inner_loop="research",
            score=score,
            metrics=metrics,
            judge_scores=judge_scores,
            summary=(
                f"Retrieved {len(sources)} sources and {claim_count} claims; {llm_summary}"
                f"median judge score {score:.3f}."
            ),
            passed=score >= self.pass_threshold,
        )

    def _llm_judge_score(
        self, variant: Variant, metrics: dict[str, float], source_count: int, claim_count: int
    ) -> tuple[Optional[float], str]:
        if not self.llm.is_live:
            return None, ""
        system = (
            "You are a research-loop judge. Return JSON only with keys score and rationale. "
            "Score from 0 to 1 based on evidence coverage, corroboration, relevance, and likely utility "
            "of the query variant for the user's research goal."
        )
        user = json.dumps(
            {
                "query_variant": variant.payload,
                "metadata": variant.metadata,
                "source_count": source_count,
                "claim_count": claim_count,
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
        try:
            payload = self.llm.complete_json(system, user, max_output_tokens=350)
            score = max(0.0, min(1.0, float(payload.get("score", 0.0))))
            rationale = str(payload.get("rationale", "LLM judge returned a score."))
            return round(score, 3), f"LLM judge: {rationale} "
        except Exception as exc:
            return None, f"LLM judge unavailable ({type(exc).__name__}). "

    def _research_metrics(self, sources: list[object], claim_count: int) -> dict[str, float]:
        if not sources:
            return {"coverage": 0.0, "corroboration": 0.0, "credibility": 0.0}
        credibility = sum(float(source.credibility_score) for source in sources) / len(sources)
        return {
            "coverage": round(min(1.0, len(sources) / 5), 3),
            "corroboration": round(min(1.0, claim_count / 10), 3),
            "credibility": round(credibility, 3),
        }


class PlateauDetector:
    def __init__(self, mode: TaskMode):
        self.mode = mode
        self.best_score = 0.0
        self.plateau_count = 0
        self.epsilon = 0.005 if mode == "optimize" else 0.03
        self.patience = 2 if mode == "optimize" else 3

    def update(self, score: float) -> str:
        if score > self.best_score + self.epsilon:
            self.best_score = score
            self.plateau_count = 0
            return "improved"
        self.plateau_count += 1
        if self.plateau_count >= self.patience:
            return "coverage_plateau" if self.mode == "research" else "score_plateau"
        return "continue"


class EvolutionaryOuterLoop:
    def __init__(
        self,
        run_id: str,
        goal: str,
        task_mode: TaskMode,
        source_strategy: list[SourceStrategyItem],
        search_factory: SearchFactory,
        evaluator: Optional[EvaluatorFn] = None,
        llm: Optional[LLMClient] = None,
        max_outer_iterations: int = 4,
        population_size: int = 4,
    ):
        self.run_id = run_id
        self.goal = goal
        self.task_mode = task_mode
        self.source_strategy = source_strategy
        self.search_factory = search_factory
        self.evaluator = evaluator
        self.llm = llm or LLMClient()
        self.max_outer_iterations = max_outer_iterations
        self.population_size = population_size

    async def run(self, store: ArtifactStore) -> None:
        inner_loop = self._inner_loop()
        plateau = PlateauDetector(self.task_mode)
        parents: list[Variant] = []
        for outer_iteration in range(1, self.max_outer_iterations + 1):
            variants = self._propose_variants(outer_iteration, parents)
            for variant in variants:
                store.add_variant(variant)
            store.append_progress(f"Outer {outer_iteration}: proposed {len(variants)} {self.task_mode} variants")
            for variant in variants:
                store.append_progress(f"  Variant {variant.id}: {_shorten(variant.payload)}")
            result = await inner_loop.evaluate(variants, store)
            best_eval = result.ranked_evaluations[0] if result.ranked_evaluations else None
            plateau_signal = plateau.update(best_eval.score if best_eval else 0.0)
            termination_signal = result.termination_signal
            if termination_signal == "continue":
                termination_signal = plateau_signal
            store.add_evolution_round(
                EvolutionRound(
                    run_id=self.run_id,
                    outer_iteration=outer_iteration,
                    mode=self.task_mode,
                    variant_ids=[variant.id for variant in variants],
                    best_variant_id=best_eval.variant_id if best_eval else None,
                    best_score=best_eval.score if best_eval else 0.0,
                    termination_signal=termination_signal,
                    plateau_count=plateau.plateau_count,
                )
            )
            store.append_progress(
                f"Outer {outer_iteration}: mode={self.task_mode} best_score="
                f"{best_eval.score if best_eval else 0.0:.3f} signal={termination_signal}"
            )
            for evaluation in result.ranked_evaluations[:3]:
                store.append_progress(
                    f"  Score {evaluation.score:.3f} for {evaluation.variant_id}: {_shorten(evaluation.summary)}"
                )
            winner_ids = {evaluation.variant_id for evaluation in result.ranked_evaluations[:2]}
            parents = [variant for variant in variants if variant.id in winner_ids]
            if termination_signal in {"score_threshold", "claim_corroboration_threshold", "score_plateau", "coverage_plateau"}:
                break

    def _inner_loop(self) -> InnerLoop:
        if self.task_mode == "optimize":
            if self.evaluator is None:
                raise ValueError("OptimizeLoop requires a deterministic evaluator.")
            return OptimizeLoop(self.run_id, self.evaluator)
        return ResearchLoop(self.run_id, self.search_factory, self.llm)

    def _propose_variants(self, outer_iteration: int, parents: list[Variant]) -> list[Variant]:
        if self.task_mode == "optimize":
            return self._propose_code_variants(outer_iteration, parents)
        return self._propose_query_variants(outer_iteration, parents)

    def _propose_query_variants(self, outer_iteration: int, parents: list[Variant]) -> list[Variant]:
        llm_variants = self._llm_query_variants(outer_iteration, parents)
        if llm_variants:
            return llm_variants
        if not parents:
            variants = []
            for item in self.source_strategy[: self.population_size]:
                variants.append(
                    Variant(
                        run_id=self.run_id,
                        outer_iteration=outer_iteration,
                        kind="query",
                        payload=item.queries[0],
                        parent_ids=[],
                        metadata={"retriever": item.retriever, "purpose": item.purpose, "limit": item.limit},
                    )
                )
            return variants
        suffixes = ["survey benchmark", "limitations contradictory evidence", "recent empirical results", "implementation signals"]
        variants = []
        for index, suffix in enumerate(suffixes[: self.population_size]):
            parent = parents[index % len(parents)]
            variants.append(
                Variant(
                    run_id=self.run_id,
                    outer_iteration=outer_iteration,
                    kind="query",
                    payload=f"{parent.payload} {suffix}",
                    parent_ids=[parent.id],
                    metadata=parent.metadata,
                )
            )
        return variants

    def _propose_code_variants(self, outer_iteration: int, parents: list[Variant]) -> list[Variant]:
        llm_variants = self._llm_code_variants(outer_iteration, parents)
        if llm_variants:
            return llm_variants
        if not parents:
            seeds = [
                "baseline direct implementation",
                "vectorized implementation",
                "cached implementation",
                "branch-reduced implementation",
            ]
        else:
            seeds = [f"{parent.payload} refined pass {outer_iteration}" for parent in parents]
            seeds.extend(f"{parent.payload} alternative mutation {outer_iteration}" for parent in parents)
        return [
            Variant(
                run_id=self.run_id,
                outer_iteration=outer_iteration,
                kind="code",
                payload=payload,
                parent_ids=[parent.id for parent in parents],
                metadata={"goal": self.goal},
            )
            for payload in seeds[: self.population_size]
        ]

    def _llm_query_variants(self, outer_iteration: int, parents: list[Variant]) -> list[Variant]:
        if not self.llm.is_live:
            return []
        parent_payloads = [parent.payload for parent in parents]
        strategy = [
            {"retriever": item.retriever, "purpose": item.purpose, "query": item.queries[0], "limit": item.limit}
            for item in self.source_strategy[: self.population_size]
        ]
        system = (
            "You are the outer orchestrator in an evolutionary research harness. "
            "Propose query variants as JSON only: {\"variants\": [{\"query\": str, \"retriever\": str, \"purpose\": str}]}."
        )
        user = json.dumps(
            {
                "goal": self.goal,
                "outer_iteration": outer_iteration,
                "parents": parent_payloads,
                "available_strategy": strategy,
                "population_size": self.population_size,
            },
            indent=2,
            sort_keys=True,
        )
        try:
            payload = self.llm.complete_json(system, user, max_output_tokens=800)
        except Exception:
            return []
        rows = payload.get("variants", [])
        if not isinstance(rows, list):
            return []
        variants = []
        for row in rows[: self.population_size]:
            if not isinstance(row, dict):
                continue
            query = str(row.get("query") or "").strip()
            if not query:
                continue
            retriever = str(row.get("retriever") or (self.source_strategy[0].retriever if self.source_strategy else "local"))
            variants.append(
                Variant(
                    run_id=self.run_id,
                    outer_iteration=outer_iteration,
                    kind="query",
                    payload=query,
                    parent_ids=[parent.id for parent in parents],
                    metadata={"retriever": retriever, "purpose": str(row.get("purpose") or "llm-proposed query"), "limit": 8},
                )
            )
        return variants

    def _llm_code_variants(self, outer_iteration: int, parents: list[Variant]) -> list[Variant]:
        if not self.llm.is_live:
            return []
        system = (
            "You are the outer orchestrator in an evolutionary optimization harness. "
            "Propose candidate code or strategy variants as JSON only: {\"variants\": [{\"payload\": str}]}."
        )
        user = json.dumps(
            {
                "goal": self.goal,
                "outer_iteration": outer_iteration,
                "parents": [parent.payload for parent in parents],
                "population_size": self.population_size,
            },
            indent=2,
            sort_keys=True,
        )
        try:
            payload = self.llm.complete_json(system, user, max_output_tokens=800)
        except Exception:
            return []
        rows = payload.get("variants", [])
        if not isinstance(rows, list):
            return []
        variants = []
        for row in rows[: self.population_size]:
            if not isinstance(row, dict):
                continue
            candidate = str(row.get("payload") or "").strip()
            if not candidate:
                continue
            variants.append(
                Variant(
                    run_id=self.run_id,
                    outer_iteration=outer_iteration,
                    kind="code",
                    payload=candidate,
                    parent_ids=[parent.id for parent in parents],
                    metadata={"goal": self.goal, "proposal_source": "llm"},
                )
            )
        return variants


def _support_level(confidence: float) -> str:
    if confidence >= 0.75:
        return "strong"
    if confidence >= 0.55:
        return "moderate"
    return "weak"


def _stable_judge_score(payload: str, metrics: dict[str, float]) -> float:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    jitter = int(digest[:4], 16) / 0xFFFF
    weighted = (metrics["coverage"] * 0.35) + (metrics["corroboration"] * 0.35) + (metrics["credibility"] * 0.25)
    return round(min(1.0, weighted + (jitter * 0.05)), 3)


def _shorten(text: str, limit: int = 140) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
