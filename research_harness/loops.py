from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Optional, Protocol

from challenges.prediction_market import prediction_market_score

from .llm import LLMClient
from .schemas import (
    Claim,
    EvolutionRound,
    FailedPath,
    ProductAgent,
    Source,
    SourceStrategyItem,
    TaskIngestionDecision,
    TaskMode,
    Variant,
    VariantEvaluation,
    now_iso,
)
from .search import SearchBackend
from .store import ArtifactStore


EvaluatorFn = Callable[[str], float]
SearchFactory = Callable[[str], SearchBackend]
PM_UPSTREAM_PATH = Path(os.environ.get("PREDICTION_MARKET_CHALLENGE_PATH", "/private/tmp/prediction-market-challenge-src"))


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
            "prediction_market": prediction_market_score,
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
                product_agent="research",
            )
        if requested == "optimize_query":
            product_agent = _product_agent_for("optimize_query", goal, evaluator_name)
            return TaskIngestionDecision(
                requested_mode=requested_mode,
                selected_mode="optimize_query",
                evaluator_name=evaluator_name if evaluator else None,
                product_agent=product_agent,
                reason=(
                    f"{product_agent.title()} agent selected optimization-query loop; query exploration will "
                    + (
                        f"feed the registered evaluator '{evaluator_name}'."
                        if evaluator
                        else "run without an optimizer evaluator."
                    )
                ),
            )
        if requested == "optimize" and evaluator:
            product_agent = _product_agent_for("optimize", goal, evaluator_name)
            return TaskIngestionDecision(
                requested_mode=requested_mode,
                selected_mode="optimize",
                evaluator_name=evaluator_name,
                product_agent=product_agent,
                reason=f"{product_agent.title()} agent selected optimize loop with evaluator '{evaluator_name}'.",
            )
        if requested == "optimize" and not evaluator:
            return TaskIngestionDecision(
                requested_mode=requested_mode,
                selected_mode="research",
                evaluator_name=evaluator_name,
                product_agent="research",
                reason="Optimize mode requested, but register_evaluator failed to resolve a deterministic evaluator; falling back to research mode.",
            )
        if _looks_like_optimization_query(goal):
            product_agent = _product_agent_for("optimize_query", goal, evaluator_name)
            return TaskIngestionDecision(
                requested_mode=requested_mode,
                selected_mode="optimize_query",
                evaluator_name=evaluator_name if evaluator else None,
                product_agent=product_agent,
                reason=(
                    f"The prompt maps to the {product_agent} agent and asks for research/query exploration around an optimization-style task"
                    + (" and an evaluator is available." if evaluator else ".")
                ),
            )
        if evaluator:
            product_agent = _product_agent_for("optimize", goal, evaluator_name)
            return TaskIngestionDecision(
                requested_mode=requested_mode,
                selected_mode="optimize",
                evaluator_name=evaluator_name,
                product_agent=product_agent,
                reason=f"{product_agent.title()} agent selected because deterministic evaluator '{evaluator_name}' is registered.",
            )
        if any(hint in goal.lower() for hint in OPTIMIZE_HINTS):
            return TaskIngestionDecision(
                requested_mode=requested_mode,
                selected_mode="research",
                product_agent="research",
                reason="The prompt looks optimization-shaped, but no deterministic evaluator was registered; using research mode.",
            )
        return TaskIngestionDecision(
            requested_mode=requested_mode,
            selected_mode="research",
            product_agent="research",
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
        backend, backend_results, retrieval_notes = await self._search_with_fallback(retriever_name, variant, limit, store)
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
        metrics["fallback_used"] = 1.0 if retrieval_notes else 0.0
        judge_scores = [
            metrics["factual_accuracy"],
            metrics["citation_accuracy"],
            metrics["completeness"],
            metrics["source_quality"],
            metrics["tool_efficiency"],
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
                f"Retrieved {len(sources)} sources and {claim_count} claims; "
                f"rubric factual={metrics['factual_accuracy']:.3f}, citation={metrics['citation_accuracy']:.3f}, "
                f"complete={metrics['completeness']:.3f}, source_quality={metrics['source_quality']:.3f}, "
                f"tool_efficiency={metrics['tool_efficiency']:.3f}; {llm_summary}"
                f"median judge score {score:.3f}."
                + (f" Retrieval notes: {'; '.join(retrieval_notes)}" if retrieval_notes else "")
            ),
            passed=score >= self.pass_threshold,
        )

    async def _search_with_fallback(
        self,
        retriever_name: str,
        variant: Variant,
        limit: int,
        store: ArtifactStore,
    ) -> tuple[SearchBackend, list[tuple[object, float]], list[str]]:
        backend = self.search_factory(retriever_name)
        notes: list[str] = []
        store.append_progress(f"Retriever search: {retriever_name} for {variant.id} (limit={limit})")
        try:
            results = await asyncio.to_thread(backend.search, variant.payload, limit)
            store.append_progress(f"Retriever done: {retriever_name} for {variant.id} returned {len(results)} result(s)")
            return backend, results, notes
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            store.add_failed_path(
                FailedPath(
                    description=f"Retriever '{retriever_name}' failed for variant {variant.id}",
                    reason=message,
                    created_by_agent=f"research_loop:{variant.id}",
                    run_id=self.run_id,
                )
            )
            store.append_progress(f"Retriever fallback: {retriever_name} failed for {variant.id}: {message}")
            notes.append(f"{retriever_name} failed ({type(exc).__name__})")
            if retriever_name == "local":
                return backend, [], notes
            fallback_backend = self.search_factory("local")
            store.append_progress(f"Retriever search: local fallback for {variant.id} (limit={limit})")
            try:
                results = await asyncio.to_thread(fallback_backend.search, variant.payload, limit)
            except Exception as fallback_exc:
                fallback_message = f"{type(fallback_exc).__name__}: {fallback_exc}"
                store.add_failed_path(
                    FailedPath(
                        description=f"Local fallback retriever failed for variant {variant.id}",
                        reason=fallback_message,
                        created_by_agent=f"research_loop:{variant.id}",
                        run_id=self.run_id,
                    )
                )
                store.append_progress(f"Retriever fallback: local failed for {variant.id}: {fallback_message}")
                notes.append(f"local fallback failed ({type(fallback_exc).__name__})")
                return fallback_backend, [], notes
            notes.append("local fallback used")
            store.append_progress(f"Retriever done: local fallback for {variant.id} returned {len(results)} result(s)")
            return fallback_backend, results, notes

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
            return {
                "coverage": 0.0,
                "corroboration": 0.0,
                "credibility": 0.0,
                "factual_accuracy": 0.0,
                "citation_accuracy": 0.0,
                "completeness": 0.0,
                "source_quality": 0.0,
                "tool_efficiency": 0.0,
            }
        credibility = sum(float(source.credibility_score) for source in sources) / len(sources)
        coverage = round(min(1.0, len(sources) / 5), 3)
        corroboration = round(min(1.0, claim_count / 10), 3)
        credibility = round(credibility, 3)
        citation_accuracy = 1.0 if claim_count > 0 else 0.0
        tool_efficiency = round(max(0.0, min(1.0, 1.15 - (max(0, len(sources) - 8) * 0.08))), 3)
        return {
            "coverage": coverage,
            "corroboration": corroboration,
            "credibility": credibility,
            "factual_accuracy": round((credibility * 0.65) + (corroboration * 0.35), 3),
            "citation_accuracy": citation_accuracy,
            "completeness": coverage,
            "source_quality": credibility,
            "tool_efficiency": tool_efficiency,
        }


class OptimizationQueryLoop(ResearchLoop):
    mode: TaskMode = "optimize_query"

    async def _evaluate_variant(self, variant: Variant, store: ArtifactStore) -> VariantEvaluation:
        evaluation = await super()._evaluate_variant(variant, store)
        metrics = dict(evaluation.metrics)
        metrics["evidence_coverage"] = metrics.get("coverage", 0.0)
        metrics["novelty"] = _novelty_score(variant.payload)
        metrics["implementability"] = _implementability_score(variant.payload)
        metrics["evaluator_relevance"] = _evaluator_relevance_score(variant.payload, str(variant.metadata.get("evaluator_name", "")))
        judge_scores = list(evaluation.judge_scores) + [
            metrics["novelty"],
            metrics["implementability"],
            metrics["evaluator_relevance"],
        ]
        llm_score, llm_summary = self._llm_optimization_query_score(variant, metrics)
        if llm_score is not None:
            judge_scores.append(llm_score)
        score = round(median(judge_scores), 3)
        return VariantEvaluation(
            run_id=evaluation.run_id,
            variant_id=evaluation.variant_id,
            inner_loop="optimize_query",
            score=score,
            metrics=metrics,
            judge_scores=judge_scores,
            summary=(
                evaluation.summary
                + f" novelty={metrics['novelty']:.3f}; "
                + f"implementability={metrics['implementability']:.3f}; "
                + f"evaluator_relevance={metrics['evaluator_relevance']:.3f}. "
                + llm_summary
            ),
            passed=score >= self.pass_threshold,
        )

    def _llm_optimization_query_score(self, variant: Variant, metrics: dict[str, float]) -> tuple[Optional[float], str]:
        if not self.llm.is_live:
            return None, ""
        system = (
            "You are judging whether a query result will help solve an optimization challenge. "
            "Return JSON only with keys score and rationale. Score 0 to 1 for actionable strategy value, "
            "implementability, and relevance to the evaluator."
        )
        user = json.dumps(
            {
                "query_variant": variant.payload,
                "metadata": variant.metadata,
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
        try:
            payload = self.llm.complete_json(system, user, max_output_tokens=350)
            score = max(0.0, min(1.0, float(payload.get("score", 0.0))))
            return round(score, 3), f"Optimization-query LLM judge: {payload.get('rationale', '')}"
        except Exception as exc:
            return None, f"Optimization-query LLM judge unavailable ({type(exc).__name__})."


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


@dataclass(frozen=True)
class LoopObjective:
    kind: str = "score"
    target: Optional[float] = None
    no_stop_until_target: bool = False

    @property
    def has_explicit_target(self) -> bool:
        return self.target is not None


class EvolutionaryOuterLoop:
    def __init__(
        self,
        run_id: str,
        goal: str,
        task_mode: TaskMode,
        source_strategy: list[SourceStrategyItem],
        search_factory: SearchFactory,
        evaluator: Optional[EvaluatorFn] = None,
        evaluator_name: Optional[str] = None,
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
        self.evaluator_name = evaluator_name or ""
        self.llm = llm or LLMClient()
        self.max_outer_iterations = max_outer_iterations
        self.population_size = population_size
        self.objective = _loop_objective_from_goal(goal, evaluator_name)

    async def run(self, store: ArtifactStore) -> None:
        if self.task_mode in {"optimize", "optimize_query"}:
            await self._record_literature_grounding(store, "initial")
        if self.task_mode == "optimize_query":
            await self._run_optimize_query(store)
            return
        inner_loop = self._inner_loop()
        plateau = PlateauDetector(self.task_mode)
        parents: list[Variant] = []
        last_result: Optional[InnerLoopResult] = None
        last_variants: list[Variant] = []
        best_eval: Optional[VariantEvaluation] = None
        best_variants: list[Variant] = []
        for outer_iteration in range(1, self.max_outer_iterations + 1):
            variants = self._propose_variants(outer_iteration, parents)
            last_variants = variants
            for variant in variants:
                store.add_variant(variant)
            store.append_progress(f"Outer {outer_iteration}: proposed {len(variants)} {self.task_mode} variants")
            for variant in variants:
                store.append_progress(f"  Variant {variant.id}: {_shorten(variant.payload)}")
            result = await inner_loop.evaluate(variants, store)
            last_result = result
            round_best = result.ranked_evaluations[0] if result.ranked_evaluations else None
            if round_best and (best_eval is None or round_best.score > best_eval.score):
                best_eval = round_best
                best_variants = variants
            plateau_signal = plateau.update(round_best.score if round_best else 0.0)
            termination_signal = result.termination_signal
            if termination_signal == "continue":
                termination_signal = plateau_signal
            store.add_evolution_round(
                EvolutionRound(
                    run_id=self.run_id,
                    outer_iteration=outer_iteration,
                    mode=self.task_mode,
                    variant_ids=[variant.id for variant in variants],
                    best_variant_id=round_best.variant_id if round_best else None,
                    best_score=round_best.score if round_best else 0.0,
                    termination_signal=termination_signal,
                    plateau_count=plateau.plateau_count,
                )
            )
            store.append_progress(
                f"Outer {outer_iteration}: mode={self.task_mode} best_score="
                f"{round_best.score if round_best else 0.0:.3f} signal={termination_signal}"
            )
            for evaluation in result.ranked_evaluations[:3]:
                store.append_progress(
                    f"  Score {evaluation.score:.3f} for {evaluation.variant_id}: {_shorten(evaluation.summary)}"
                )
            winner_ids = {evaluation.variant_id for evaluation in result.ranked_evaluations[:2]}
            parents = [variant for variant in variants if variant.id in winner_ids]
            if termination_signal in {"score_plateau", "coverage_plateau"}:
                self._record_literature_refresh(store, termination_signal, outer_iteration)
            if self._should_stop_outer_loop(termination_signal, round_best, outer_iteration):
                break
        if self.task_mode == "optimize" and last_result:
            self._write_optimization_outputs(store, best_variants or last_variants, best_eval)

    def _inner_loop(self) -> InnerLoop:
        if self.task_mode == "optimize":
            if self.evaluator is None:
                raise ValueError("OptimizeLoop requires a deterministic evaluator.")
            return OptimizeLoop(self.run_id, self.evaluator)
        if self.task_mode == "optimize_query":
            return OptimizationQueryLoop(self.run_id, self.search_factory, self.llm)
        return ResearchLoop(self.run_id, self.search_factory, self.llm)

    def _propose_variants(self, outer_iteration: int, parents: list[Variant]) -> list[Variant]:
        if self.task_mode == "optimize":
            return self._propose_code_variants(outer_iteration, parents)
        return self._propose_query_variants(outer_iteration, parents)

    async def _run_optimize_query(self, store: ArtifactStore) -> None:
        query_loop = OptimizationQueryLoop(self.run_id, self.search_factory, self.llm)
        plateau = PlateauDetector("research")
        parents: list[Variant] = []
        last_result: Optional[InnerLoopResult] = None
        for outer_iteration in range(1, self.max_outer_iterations + 1):
            variants = self._propose_query_variants(outer_iteration, parents)
            for variant in variants:
                variant.metadata.setdefault("challenge_goal", self.goal)
                variant.metadata.setdefault("evaluator_name", self.evaluator_name)
                variant.metadata.setdefault("query_intent", "optimization challenge strategy discovery")
                store.add_variant(variant)
            store.append_progress(f"Optimization-query phase {outer_iteration}: proposed {len(variants)} query variants")
            for variant in variants:
                store.append_progress(f"  Query {variant.id}: {_shorten(variant.payload)}")
            result = await query_loop.evaluate(variants, store)
            last_result = result
            best_eval = result.ranked_evaluations[0] if result.ranked_evaluations else None
            plateau_signal = plateau.update(best_eval.score if best_eval else 0.0)
            termination_signal = result.termination_signal
            if termination_signal == "continue":
                termination_signal = plateau_signal
            store.add_evolution_round(
                EvolutionRound(
                    run_id=self.run_id,
                    outer_iteration=outer_iteration,
                    mode="optimize_query",
                    variant_ids=[variant.id for variant in variants],
                    best_variant_id=best_eval.variant_id if best_eval else None,
                    best_score=best_eval.score if best_eval else 0.0,
                    termination_signal=termination_signal,
                    plateau_count=plateau.plateau_count,
                )
            )
            store.append_progress(
                f"Optimization-query phase {outer_iteration}: best_score={best_eval.score if best_eval else 0.0:.3f} signal={termination_signal}"
            )
            for evaluation in result.ranked_evaluations[:3]:
                store.append_progress(f"  Query score {evaluation.score:.3f} for {evaluation.variant_id}: {_shorten(evaluation.summary)}")
            winner_ids = {evaluation.variant_id for evaluation in result.ranked_evaluations[:2]}
            parents = [variant for variant in variants if variant.id in winner_ids]
            if self._should_stop_query_loop(termination_signal, outer_iteration):
                break

        seed_context = self._build_optimizer_seed_context(store, last_result)
        store.write_optimizer_seed_context(seed_context)
        store.append_progress(f"Optimizer seed context: {store.optimizer_seed_context_path}")
        if self.evaluator is None:
            store.append_progress("Optimizer phase skipped: no evaluator was registered for optimize_query mode.")
            return

        store.append_progress("Optimizer phase: starting code/strategy variants from query seed context")
        seed_parents = self._seed_context_variants(seed_context)
        if self.evaluator_name == "prediction_market":
            await self._run_prediction_market_optimizer(store, seed_parents, seed_context)
            return
        optimize_loop = OptimizeLoop(self.run_id, self.evaluator)
        await self._run_generic_optimizer_rounds(store, optimize_loop, seed_parents, seed_context)

    async def _run_generic_optimizer_rounds(
        self,
        store: ArtifactStore,
        optimize_loop: OptimizeLoop,
        parents: list[Variant],
        seed_context: dict[str, object],
    ) -> None:
        plateau = PlateauDetector("optimize")
        best_eval: Optional[VariantEvaluation] = None
        best_round_variants: list[Variant] = []
        for round_index in range(1, self.max_outer_iterations + 1):
            code_variants = self._propose_code_variants(round_index, parents)
            for variant in code_variants:
                variant.metadata["optimizer_seed_context_path"] = str(store.optimizer_seed_context_path)
                variant.metadata["query_seed_summary"] = seed_context.get("summary", "")
                store.add_variant(variant)
            result = await optimize_loop.evaluate(code_variants, store)
            round_best = result.ranked_evaluations[0] if result.ranked_evaluations else None
            if round_best and (best_eval is None or round_best.score > best_eval.score):
                best_eval = round_best
                best_round_variants = code_variants
            plateau_signal = plateau.update(round_best.score if round_best else 0.0)
            termination_signal = result.termination_signal if result.termination_signal != "continue" else plateau_signal
            store.add_evolution_round(
                EvolutionRound(
                    run_id=self.run_id,
                    outer_iteration=(len(store.list("evolution_rounds")) + 1),
                    mode="optimize",
                    variant_ids=[variant.id for variant in code_variants],
                    best_variant_id=round_best.variant_id if round_best else None,
                    best_score=round_best.score if round_best else 0.0,
                    termination_signal=termination_signal,
                    plateau_count=plateau.plateau_count,
                )
            )
            store.append_progress(
                f"Optimizer phase round {round_index}: best_score={round_best.score if round_best else 0.0:.3f} signal={termination_signal}"
            )
            for evaluation in result.ranked_evaluations[:3]:
                store.append_progress(f"  Optimize score {evaluation.score:.3f} for {evaluation.variant_id}: {_shorten(evaluation.summary)}")
            parents = [variant for variant in code_variants if variant.id in {evaluation.variant_id for evaluation in result.ranked_evaluations[:2]}]
            if termination_signal == "score_plateau":
                self._record_literature_refresh(store, termination_signal, round_index)
            if self._should_stop_optimizer_loop(termination_signal, round_best, round_index):
                break
        self._write_optimization_outputs(store, best_round_variants, best_eval)

    async def _run_prediction_market_optimizer(
        self,
        store: ArtifactStore,
        parents: list[Variant],
        seed_context: dict[str, object],
    ) -> None:
        plateau = PlateauDetector("optimize")
        best_eval: Optional[VariantEvaluation] = None
        best_round_variants: list[Variant] = []
        for round_index in range(1, self.max_outer_iterations + 1):
            code_variants = self._propose_prediction_market_variants(round_index, parents)
            for variant in code_variants:
                variant.metadata["optimizer_seed_context_path"] = str(store.optimizer_seed_context_path)
                variant.metadata["query_seed_summary"] = seed_context.get("summary", "")
                store.add_variant(variant)
            evaluations = await asyncio.gather(
                *(self._evaluate_prediction_market_variant(variant, store, round_index) for variant in code_variants)
            )
            for evaluation in evaluations:
                store.add_variant_evaluation(evaluation)
            ranked = sorted(evaluations, key=lambda item: item.score, reverse=True)
            round_best = ranked[0] if ranked else None
            if round_best and (best_eval is None or round_best.score > best_eval.score):
                best_eval = round_best
                best_round_variants = code_variants
            plateau_signal = plateau.update(round_best.score if round_best else 0.0)
            objective_met = self._prediction_market_objective_met(round_best)
            if self.objective.has_explicit_target:
                termination_signal = "profit_target" if objective_met else plateau_signal
            else:
                termination_signal = "score_threshold" if round_best and round_best.score >= 0.8 else plateau_signal
            store.add_evolution_round(
                EvolutionRound(
                    run_id=self.run_id,
                    outer_iteration=(len(store.list("evolution_rounds")) + 1),
                    mode="optimize",
                    variant_ids=[variant.id for variant in code_variants],
                    best_variant_id=round_best.variant_id if round_best else None,
                    best_score=round_best.score if round_best else 0.0,
                    termination_signal=termination_signal,
                    plateau_count=plateau.plateau_count,
                )
            )
            store.append_progress(
                f"Prediction-market optimizer round {round_index}: best_edge={_pm_edge_from_eval(round_best):.3f} "
                f"score={round_best.score if round_best else 0.0:.3f} signal={termination_signal}"
            )
            for evaluation in ranked[:3]:
                source = evaluation.metrics.get("score_source", "unknown")
                store.append_progress(f"  Prediction-market score {evaluation.score:.3f} via {source} for {evaluation.variant_id}: {_shorten(evaluation.summary)}")
            parents = [variant for variant in code_variants if variant.id in {evaluation.variant_id for evaluation in ranked[:2]}]
            if termination_signal == "score_plateau":
                self._record_literature_refresh(store, termination_signal, round_index)
            if self._should_stop_optimizer_loop(termination_signal, round_best, round_index):
                break
        self._write_optimization_outputs(store, best_round_variants, best_eval)

    def _should_stop_outer_loop(
        self,
        termination_signal: str,
        best_eval: Optional[VariantEvaluation],
        outer_iteration: int,
    ) -> bool:
        if termination_signal in {"score_plateau", "coverage_plateau"} and self.objective.no_stop_until_target:
            return outer_iteration >= self.max_outer_iterations
        if termination_signal == "score_threshold" and self.objective.has_explicit_target:
            return self._generic_objective_met(best_eval)
        return termination_signal in {"score_threshold", "claim_corroboration_threshold", "score_plateau", "coverage_plateau"}

    def _should_stop_query_loop(self, termination_signal: str, outer_iteration: int) -> bool:
        minimum_query_rounds = 2 if self.objective.no_stop_until_target else 1
        if outer_iteration < min(self.max_outer_iterations, minimum_query_rounds):
            return False
        if termination_signal == "coverage_plateau" and self.objective.no_stop_until_target:
            return outer_iteration >= self.max_outer_iterations
        return termination_signal in {"claim_corroboration_threshold", "coverage_plateau"}

    def _should_stop_optimizer_loop(
        self,
        termination_signal: str,
        best_eval: Optional[VariantEvaluation],
        round_index: int,
    ) -> bool:
        if self.objective.has_explicit_target:
            if termination_signal in {"profit_target", "score_threshold"}:
                return self._objective_met(best_eval)
            return round_index >= self.max_outer_iterations
        if termination_signal == "score_plateau" and self.objective.no_stop_until_target:
            return round_index >= self.max_outer_iterations
        return termination_signal in {"score_threshold", "profit_target"}

    def _objective_met(self, best_eval: Optional[VariantEvaluation]) -> bool:
        if self.evaluator_name == "prediction_market":
            return self._prediction_market_objective_met(best_eval)
        return self._generic_objective_met(best_eval)

    def _generic_objective_met(self, best_eval: Optional[VariantEvaluation]) -> bool:
        if not best_eval:
            return False
        if self.objective.target is None:
            return best_eval.score >= 0.8
        return best_eval.score >= self.objective.target

    def _prediction_market_objective_met(self, best_eval: Optional[VariantEvaluation]) -> bool:
        if not best_eval:
            return False
        edge = _pm_edge_from_eval(best_eval)
        if self.objective.target is None:
            return best_eval.score >= 0.8
        return edge >= self.objective.target

    async def _record_literature_grounding(self, store: ArtifactStore, reason: str) -> None:
        if any(claim.get("created_by_agent") == "literature_grounding_policy" for claim in store.list("claims")):
            return
        query = (
            f"{self.goal} existing literature benchmarks failure modes evaluation "
            "optimization strategy agent regressions"
        )
        item = self.source_strategy[0] if self.source_strategy else None
        retriever_name = item.retriever if item else "local"
        limit = min(3, item.limit if item else 3)
        notes: list[str] = []
        store.append_progress(f"Literature grounding ({reason}): searching {retriever_name} for existing evidence")
        try:
            backend = self.search_factory(retriever_name)
            results = await asyncio.to_thread(backend.search, query, limit)
        except Exception as exc:
            notes.append(f"{retriever_name} failed ({type(exc).__name__}: {exc})")
            store.append_progress(f"Retriever fallback: {retriever_name} failed during literature grounding: {type(exc).__name__}: {exc}")
            backend = self.search_factory("local")
            results = await asyncio.to_thread(backend.search, query, limit)
        sources = []
        for document, relevance in results[:limit]:
            source = store.add_source(backend.to_source(document, relevance))
            sources.append((source, document))
        for source, document in sources:
            claim_text = document.claims[0] if document.claims else document.summary
            store.add_claim(
                Claim(
                    text=f"Literature grounding ({reason}) found: {claim_text}",
                    source_ids=[source.id],
                    confidence=0.74,
                    support_level="retrieved",
                    created_by_agent="literature_grounding_policy",
                    run_id=self.run_id,
                )
            )
        store.append_progress(
            f"Literature grounding ({reason}): query='{query}' retrieved {len(sources)} source(s)"
            + (f"; notes={' ; '.join(notes)}" if notes else "")
        )

    def _record_literature_refresh(self, store: ArtifactStore, reason: str, round_index: int) -> None:
        existing_refresh = [
            source
            for source in store.list("sources")
            if str(source.get("url", "")).startswith(f"memory://literature-refresh/{self.run_id}/")
            and str(source.get("url", "")).endswith(f"/{reason}")
        ]
        if existing_refresh:
            return
        query = (
            f"{self.goal} literature after {reason} "
            "failure modes benchmark regressions optimization strategy"
        )
        source = store.add_source(
            Source(
                url=f"memory://literature-refresh/{self.run_id}/{round_index}/{reason}",
                title=f"Literature refresh for {reason}",
                author="research-harness",
                date=now_iso().split("T")[0],
                source_type="memory",
                summary=(
                    "Triggered because the agent loop plateaued or regressed. "
                    "Use external literature, challenge references, and eval failure cases before further hyperparameter changes."
                ),
                relevance_score=0.82,
                credibility_score=0.72,
            )
        )
        store.add_claim(
            Claim(
                text=(
                    f"Loop round {round_index} emitted {reason}; the harness triggered a literature-refresh query "
                    f"before further optimization: {query}."
                ),
                source_ids=[source.id],
                confidence=0.78,
                support_level="instrumented",
                created_by_agent="literature_refresh_policy",
                run_id=self.run_id,
            )
        )
        store.append_progress(f"Literature refresh triggered by {reason} at optimizer round {round_index}: {query}")

    async def _evaluate_prediction_market_variant(
        self,
        variant: Variant,
        store: ArtifactStore,
        round_index: int,
    ) -> VariantEvaluation:
        code = self._render_optimal_code(variant.payload)
        candidate_path = store.candidates_dir / f"round_{round_index:02d}_{variant.id}.py"
        candidate_path.write_text(code, encoding="utf-8")
        store.append_progress(f"  Candidate eval start {variant.id}: {candidate_path}")
        result = await asyncio.to_thread(_run_prediction_market_official, candidate_path)
        edge = float(result.get("mean_edge", 0.0))
        score = _normalize_prediction_market_edge(edge)
        result["candidate_path"] = str(candidate_path)
        store.append_progress(f"  Candidate eval done {variant.id}: mean_edge={edge:.3f} score={score:.3f}")
        score_source = str(result.get("score_source", "unknown"))
        measured_label = "upstream orderbook-pm" if result.get("official_measured") else "local challenge fallback"
        return VariantEvaluation(
            run_id=self.run_id,
            variant_id=variant.id,
            inner_loop="optimize",
            score=score,
            metrics=result,
            judge_scores=[score],
            summary=(
                f"{measured_label} mean_edge={edge:.3f}; "
                f"score_source={score_source}; "
                f"successes={int(result.get('success_count', 0))}; failures={int(result.get('failure_count', 0))}; "
                f"candidate={candidate_path}."
            ),
            passed=score >= 0.8,
        )

    def _write_optimization_outputs(
        self,
        store: ArtifactStore,
        variants: list[Variant],
        best_eval: Optional[VariantEvaluation],
    ) -> None:
        if not best_eval:
            return
        best_variant = next((variant for variant in variants if variant.id == best_eval.variant_id), None)
        candidate = best_variant.payload if best_variant else ""
        store.write_optimized_candidate(candidate + "\n")
        store.append_progress(f"Optimized candidate: {store.optimized_candidate_path}")
        optimal_code = self._render_optimal_code(candidate)
        optimal_code_path = str(store.write_optimal_code(optimal_code))
        store.append_progress(f"Optimal code: {store.optimal_code_path}")
        solution = self._render_solution(candidate)
        solution_path = None
        if solution:
            solution_path = str(store.write_solution(solution))
            store.append_progress(f"Solution: {store.solution_path}")
        objective = _objective_metadata(self.evaluator_name)
        official_result = objective["official_result"]
        if self.evaluator_name == "prediction_market":
            official_result = {
                "measured": bool(best_eval.metrics.get("official_measured", False)),
                "profit_usd": best_eval.metrics.get("mean_edge"),
                "target_profit_usd": self.objective.target,
                "target_met": self._prediction_market_objective_met(best_eval),
                "score_source": best_eval.metrics.get("score_source", "unknown"),
                "required_evaluator": "https://github.com/danrobinson/prediction-market-challenge",
                "candidate_path": best_eval.metrics.get("candidate_path"),
                "success_count": best_eval.metrics.get("success_count"),
                "failure_count": best_eval.metrics.get("failure_count"),
            }
        payload = {
            "run_id": self.run_id,
            "evaluator_name": self.evaluator_name or "unknown",
            "objective_name": objective["objective_name"],
            "objective_direction": objective["objective_direction"],
            "score_variable": "score",
            "score": best_eval.score,
            "metrics": best_eval.metrics,
            "best_variant_id": best_eval.variant_id,
            "best_candidate_path": str(store.optimized_candidate_path),
            "optimal_code_path": optimal_code_path,
            "solution_path": solution_path,
            "candidate": candidate,
            "official_result": official_result,
            "objective_target": {
                "kind": self.objective.kind,
                "target": self.objective.target,
                "no_stop_until_target": self.objective.no_stop_until_target,
                "met": self._objective_met(best_eval),
            },
            "note": objective["note"],
        }
        store.write_optimization_result(payload)
        store.append_progress(
            f"Optimization result: {store.optimization_result_path} "
            f"({payload['objective_direction']} {payload['objective_name']}={best_eval.score:.3f})"
        )

    def _build_optimizer_seed_context(self, store: ArtifactStore, result: Optional[InnerLoopResult]) -> dict[str, object]:
        evaluations = result.ranked_evaluations if result else []
        variant_lookup = {row["id"]: row for row in store.list("variants")}
        top_items = []
        for evaluation in evaluations[:5]:
            variant = variant_lookup.get(evaluation.variant_id, {})
            top_items.append(
                {
                    "variant_id": evaluation.variant_id,
                    "query": variant.get("payload", ""),
                    "score": evaluation.score,
                    "metrics": evaluation.metrics,
                    "summary": evaluation.summary,
                }
            )
        summary = "; ".join(str(item["query"]) for item in top_items[:3])
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "mode": "optimize_query",
            "summary": summary,
            "top_query_findings": top_items,
            "optimizer_instruction": "Use the top query findings as strategy context when proposing optimization variants.",
            "has_evaluator": self.evaluator is not None,
        }

    def _seed_context_variants(self, seed_context: dict[str, object]) -> list[Variant]:
        parents = []
        for item in seed_context.get("top_query_findings", []) if isinstance(seed_context.get("top_query_findings"), list) else []:
            if not isinstance(item, dict):
                continue
            parents.append(
                Variant(
                    run_id=self.run_id,
                    outer_iteration=0,
                    kind="query",
                    payload=str(item.get("query", "")),
                    parent_ids=[],
                    metadata={"seed_score": item.get("score", 0.0), "seed_summary": item.get("summary", "")},
                )
            )
        return parents

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
                    metadata={
                        "retriever": item.retriever,
                        "purpose": item.purpose,
                        "limit": item.limit,
                        "research_role": "parallel_research_subagent",
                        "search_phase": "broad" if outer_iteration == 1 else "narrow",
                    },
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
                    metadata={**parent.metadata, "search_phase": "narrow", "narrowing_suffix": suffix},
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
            if all(parent.kind == "query" for parent in parents):
                challenge_context = self._optimizer_seed_prefix()
                seeds = [
                    f"{challenge_context} Optimization variant informed by query finding: {parent.payload} refined pass {outer_iteration}"
                    for parent in parents
                ]
                seeds.extend(f"{challenge_context} Alternative optimization mutation from query finding: {parent.payload}" for parent in parents)
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

    def _optimizer_seed_prefix(self) -> str:
        if self.evaluator_name == "prediction_market":
            return (
                "Prediction-market strategy sketch for upstream orderbook_pm_challenge: passive only, "
                "avoid adverse arbitrageur fills, quote only outside hidden competitor ladder, infer drift "
                "from fills and competitor best quotes, cancel all each step, small size, wide spread, "
                "inventory limits, and skip quoting near unstable midpoints."
            )
        return "Optimization strategy sketch:"

    def _propose_prediction_market_variants(self, outer_iteration: int, parents: list[Variant]) -> list[Variant]:
        templates = [
            "pm_strategy=wide_top_of_competitor size=1 spread=8 inventory=30 skew=8 quote_mode=outside_competitor cancel_all=1",
            "pm_strategy=very_wide_small_size size=0.5 spread=12 inventory=20 skew=10 quote_mode=outside_competitor cancel_all=1",
            "pm_strategy=retail_only_extreme_edges size=1 spread=16 inventory=15 skew=12 quote_mode=extreme cancel_all=1",
            "pm_strategy=ask_fade_bid_fade size=0.75 spread=10 inventory=25 skew=9 quote_mode=outside_competitor cancel_all=1",
            "pm_strategy=patient_market_maker size=0.5 spread=20 inventory=10 skew=15 quote_mode=extreme cancel_all=1",
            "pm_strategy=no_trade_control size=0 spread=99 inventory=0 skew=0 quote_mode=none cancel_all=1",
        ]
        if parents and all(parent.kind == "code" for parent in parents):
            templates = [
                f"{parent.payload} mutation_round={outer_iteration} spread_delta={delta}"
                for parent in parents
                for delta in [2, 4, 8]
            ] + templates
        elif parents:
            context = self._optimizer_seed_prefix()
            templates = [f"{context} {template} query_seed={parents[index % len(parents)].payload}" for index, template in enumerate(templates)]
        return [
            Variant(
                run_id=self.run_id,
                outer_iteration=outer_iteration,
                kind="code",
                payload=payload,
                parent_ids=[parent.id for parent in parents],
                metadata={"goal": self.goal, "challenge": "prediction_market"},
            )
            for payload in templates[: self.population_size]
        ]

    def _render_solution(self, payload: str) -> str:
        if self.evaluator_name != "prediction_market":
            return ""
        return _prediction_market_solution(payload)

    def _render_optimal_code(self, payload: str) -> str:
        if self.evaluator_name == "prediction_market":
            return _prediction_market_solution(payload)
        return _generic_optimal_code(payload, self.evaluator_name)

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
            "Start wide with short broad queries, then narrow only after observing source yield. "
            "Fan out independent directions to parallel subagents. "
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
                    metadata={
                        "retriever": retriever,
                        "purpose": str(row.get("purpose") or "llm-proposed query"),
                        "limit": 8,
                        "research_role": "parallel_research_subagent",
                        "search_phase": "broad" if outer_iteration == 1 else "narrow",
                    },
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


def _looks_like_optimization_query(goal: str) -> bool:
    normalized = goal.lower()
    query_terms = {"research", "find", "query", "search", "investigate", "explore", "look for"}
    return any(term in normalized for term in query_terms) and any(term in normalized for term in OPTIMIZE_HINTS)


def _product_agent_for(selected_mode: TaskMode, goal: str, evaluator_name: Optional[str]) -> ProductAgent:
    normalized = goal.lower()
    if evaluator_name == "prediction_market" or "challenge" in normalized:
        return "challenge"
    if selected_mode == "research":
        return "research"
    return "optimize"


def _loop_objective_from_goal(goal: str, evaluator_name: Optional[str]) -> LoopObjective:
    normalized = goal.lower()
    no_stop = bool(re.search(r"\bdo\s*not\s*stop\b|\bdon't\s*stop\b|\bdont\s*stop\b|until\s+you", normalized))
    if evaluator_name == "prediction_market":
        target = _profit_target_from_goal(goal)
        return LoopObjective(kind="profit_usd", target=target, no_stop_until_target=no_stop and target is not None)
    return LoopObjective(kind="score", target=None, no_stop_until_target=no_stop)


def _profit_target_from_goal(goal: str) -> Optional[float]:
    normalized = goal.lower()
    patterns = [
        r"(?:get\s+to|reach|hit|achieve|make|earn)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:\$|usd|dollars?)?\s*(?:profit|edge)?",
        r"\$+\s*([0-9]+(?:\.[0-9]+)?)\s*(?:profit|edge|usd|dollars?)",
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:\$|usd|dollars?)\s*(?:profit|edge)",
    ]
    if not any(term in normalized for term in ["profit", "profitable", "edge", "$", "usd", "dollar"]):
        return None
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return float(match.group(1))
    return None


def _implementability_score(text: str) -> float:
    terms = {"implement", "code", "algorithm", "strategy", "heuristic", "benchmark", "test", "optimize", "latency", "throughput"}
    tokens = set(text.lower().replace("-", " ").split())
    return round(min(1.0, 0.35 + (len(tokens & terms) * 0.12)), 3)


def _novelty_score(text: str) -> float:
    normalized = text.lower().replace("-", " ")
    tokens = [token for token in normalized.split() if len(token) > 3]
    if not tokens:
        return 0.4
    distinct_ratio = len(set(tokens)) / len(tokens)
    novelty_terms = {"novel", "alternative", "contradictory", "recent", "mechanism", "frontier", "unusual", "ablation"}
    term_bonus = min(0.25, len(set(tokens) & novelty_terms) * 0.08)
    return round(min(1.0, 0.35 + (distinct_ratio * 0.35) + term_bonus), 3)


def _evaluator_relevance_score(text: str, evaluator_name: str) -> float:
    if not evaluator_name:
        return 0.45
    evaluator_terms = set(evaluator_name.lower().replace("_", " ").split())
    tokens = set(text.lower().replace("-", " ").split())
    overlap = len(tokens & evaluator_terms)
    return round(min(1.0, 0.55 + (overlap * 0.15)), 3)


def _prediction_market_solution(payload: str) -> str:
    escaped_payload = payload.replace('"""', '\\"\\"\\"')
    params = _prediction_market_params(payload)
    return f'''from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    """Adaptive passive prediction-market market maker.

    Generated by research-harness from the best optimize-query variant.
    This file targets the public upstream challenge API:
    https://github.com/danrobinson/prediction-market-challenge

    Source variant:
    """  # noqa: D205

    quote_size = {params["size"]!r}
    base_spread_ticks = {int(params["spread"])}
    inventory_limit = {params["inventory"]!r}
    skew_divisor = {params["skew_divisor"]!r}
    quote_mode = {params["quote_mode"]!r}

    def __init__(self) -> None:
        self.estimated_mid_ticks = 50.0
        self.last_buy_fill = 0.0
        self.last_sell_fill = 0.0

    def on_step(self, state: StepState):
        competitor_bid = state.competitor_best_bid_ticks
        competitor_ask = state.competitor_best_ask_ticks
        if competitor_bid is None and competitor_ask is None:
            return [CancelAll()]
        if competitor_bid is None:
            competitor_bid = max(1, competitor_ask - 8)
        if competitor_ask is None:
            competitor_ask = min(99, competitor_bid + 8)
        competitor_mid = (competitor_bid + competitor_ask) / 2.0

        buy_flow = state.buy_filled_quantity
        sell_flow = state.sell_filled_quantity
        net_flow = buy_flow - sell_flow
        self.last_buy_fill = state.buy_filled_quantity
        self.last_sell_fill = state.sell_filled_quantity

        # The upstream score is edge at fill time. Quoting too close to the
        # hidden ladder is punished by the informed arbitrageur, so this strategy
        # is deliberately patient and cancels every step before replacing quotes.
        midpoint_jump = abs(competitor_mid - self.estimated_mid_ticks)
        self.estimated_mid_ticks = (self.estimated_mid_ticks * 0.9) + (competitor_mid * 0.1) + (net_flow * 0.04)

        inventory = state.yes_inventory - state.no_inventory
        inventory_skew = max(-12.0, min(12.0, inventory / self.skew_divisor))
        spread = self.base_spread_ticks + (4 if midpoint_jump >= 4 else 0)
        if self.quote_mode == "none":
            return [CancelAll()]
        if self.quote_mode == "extreme":
            bid_reference = min(competitor_bid, self.estimated_mid_ticks)
            ask_reference = max(competitor_ask, self.estimated_mid_ticks)
            bid_ticks = int(max(1, min(98, round(bid_reference - spread - inventory_skew))))
            ask_ticks = int(max(bid_ticks + 1, min(99, round(ask_reference + spread - inventory_skew))))
        else:
            bid_ticks = int(max(1, min(98, round(competitor_bid - spread - inventory_skew))))
            ask_ticks = int(max(bid_ticks + 1, min(99, round(competitor_ask + spread - inventory_skew))))

        actions = [CancelAll()]
        ask_cost = max(0.01, ask_ticks / 100.0)
        bid_cost = max(0.01, bid_ticks / 100.0)
        if self.quote_size <= 0:
            return actions
        size = max(0.01, min(self.quote_size, state.free_cash / ask_cost))

        if state.yes_inventory < self.inventory_limit and state.free_cash >= bid_cost * size:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=bid_ticks, quantity=size))
        if state.no_inventory < self.inventory_limit:
            actions.append(PlaceOrder(side=Side.SELL, price_ticks=ask_ticks, quantity=size))
        return actions


SOURCE_VARIANT = """{escaped_payload}"""
'''


def _generic_optimal_code(payload: str, evaluator_name: str) -> str:
    return f'''from __future__ import annotations

"""Best optimization candidate emitted by research-harness.

This module is written for every optimization or challenge run so downstream
evaluators can always find the agent-selected code artifact at optimal_code.py.
When a domain adapter can render executable code, it should replace this generic
representation with evaluator-ready code.
"""

EVALUATOR_NAME = {evaluator_name!r}
OPTIMAL_CANDIDATE = {payload!r}


def selected_candidate() -> str:
    """Return the exact candidate payload that achieved the best score."""
    return OPTIMAL_CANDIDATE
'''


def _prediction_market_params(payload: str) -> dict[str, object]:
    params = {match.group("name").lower(): float(match.group("value")) for match in re.finditer(r"(?P<name>spread|size|quantity|inventory|limit|skew)\s*[=:]\s*(?P<value>-?\d+(?:\.\d+)?)", payload, re.I)}
    text = payload.lower()
    spread = int(max(2, min(30, params.get("spread", 12.0))))
    size = max(0.0, min(5.0, params.get("quantity", params.get("size", 1.0))))
    inventory = max(0.0, min(150.0, params.get("inventory", params.get("limit", 30.0))))
    skew = max(1.0, min(30.0, params.get("skew", 8.0)))
    if "quote_mode=none" in text or "no_trade" in text:
        quote_mode = "none"
    elif "quote_mode=extreme" in text or "extreme" in text:
        quote_mode = "extreme"
    else:
        quote_mode = "outside_competitor"
    return {
        "spread": spread,
        "size": size,
        "inventory": inventory,
        "skew_divisor": skew,
        "quote_mode": quote_mode,
    }


def _normalize_prediction_market_edge(edge: float) -> float:
    return round(max(0.0, min(1.0, (edge + 30.0) / 60.0)), 3)


def _pm_edge_from_eval(evaluation: Optional[VariantEvaluation]) -> float:
    if not evaluation:
        return 0.0
    return float(evaluation.metrics.get("mean_edge", 0.0))


def _run_prediction_market_official(strategy_path: Path) -> dict[str, object]:
    strategy_text = strategy_path.read_text(encoding="utf-8")
    if os.environ.get("PREDICTION_MARKET_USE_UPSTREAM") != "1":
        result = _prediction_market_local_semantic_score(strategy_text)
        result["error"] = "Set PREDICTION_MARKET_USE_UPSTREAM=1 to run the upstream orderbook-pm grader."
        return result
    if not PM_UPSTREAM_PATH.exists():
        result = _prediction_market_local_semantic_score(strategy_text)
        result["error"] = f"Missing upstream repo at {PM_UPSTREAM_PATH}"
        return result
    cmd = [
        "uv",
        "run",
        "--project",
        str(PM_UPSTREAM_PATH),
        "orderbook-pm",
        "run",
        str(strategy_path),
        "--simulations",
        os.environ.get("PREDICTION_MARKET_SIMULATIONS", "40"),
        "--steps",
        os.environ.get("PREDICTION_MARKET_STEPS", "600"),
        "--seed-start",
        os.environ.get("PREDICTION_MARKET_SEED_START", "0"),
        "--workers",
        os.environ.get("PREDICTION_MARKET_WORKERS", "4"),
        "--json",
    ]
    try:
        env = dict(os.environ)
        env.setdefault("UV_CACHE_DIR", "/private/tmp/research-harness-uv-cache")
        env.setdefault("UV_PYTHON_INSTALL_DIR", "/private/tmp/research-harness-uv-python")
        env.setdefault("UV_TOOL_DIR", "/private/tmp/research-harness-uv-tools")
        completed = subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=True,
            env=env,
            timeout=float(os.environ.get("PREDICTION_MARKET_TIMEOUT_SECONDS", "120")),
        )
    except Exception as exc:
        result = _prediction_market_local_semantic_score(strategy_text)
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    if completed.returncode != 0:
        result = _prediction_market_local_semantic_score(strategy_text)
        result["error"] = (completed.stderr or completed.stdout).strip()[:1000]
        return result
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        result = _prediction_market_local_semantic_score(strategy_text)
        result["error"] = f"JSONDecodeError: {exc}; stdout={completed.stdout[:500]}"
        return result
    results = payload.get("simulation_results", [])
    successes = [result for result in results if not result.get("failed")]
    mean_edge = sum(float(result.get("total_edge", 0.0)) for result in successes) / max(len(successes), 1)
    mean_arb_edge = sum(float(result.get("arb_edge", 0.0)) for result in successes) / max(len(successes), 1)
    mean_retail_edge = sum(float(result.get("retail_edge", 0.0)) for result in successes) / max(len(successes), 1)
    return {
        "official_measured": True,
        "mean_edge": round(mean_edge, 6),
        "mean_arb_edge": round(mean_arb_edge, 6),
        "mean_retail_edge": round(mean_retail_edge, 6),
        "success_count": len(successes),
        "failure_count": len(results) - len(successes),
        "simulations": len(results),
        "score_source": "upstream_orderbook_pm_challenge",
    }


def _prediction_market_local_semantic_score(strategy_text: str, simulations: int = 80, steps: int = 800) -> dict[str, object]:
    params = _params_from_strategy_text(strategy_text)
    rng = random.Random(20260507)
    edges = []
    retail_edges = []
    arb_edges = []
    failures = 0
    for sim in range(simulations):
        true_prob = max(0.02, min(0.98, 0.5 + rng.uniform(-0.22, 0.22)))
        competitor_mid = true_prob
        competitor_spread = rng.choice([1, 2, 3, 4])
        inventory = 0.0
        edge = 0.0
        retail_edge = 0.0
        arb_edge = 0.0
        for step in range(steps):
            if rng.random() < rng.uniform(0.0008, 0.003):
                true_prob += rng.gauss(0.0, rng.uniform(0.2, 0.6))
            true_prob += rng.gauss(0.0, 0.02)
            true_prob = max(0.01, min(0.99, true_prob))

            lower = max(1, min(99, int(true_prob * 100)))
            competitor_bid = max(1, lower - (competitor_spread - 1))
            competitor_ask = min(99, lower + 1 + (competitor_spread - 1))
            if params["quote_mode"] == "none" or params["size"] <= 0:
                continue
            skew = max(-12.0, min(12.0, inventory / params["skew_divisor"]))
            if params["quote_mode"] == "extreme":
                bid = int(max(1, min(98, round(min(competitor_bid, competitor_mid * 100) - params["spread"] - skew))))
                ask = int(max(bid + 1, min(99, round(max(competitor_ask, competitor_mid * 100) + params["spread"] - skew))))
            else:
                bid = int(max(1, min(98, round(competitor_bid - params["spread"] - skew))))
                ask = int(max(bid + 1, min(99, round(competitor_ask + params["spread"] - skew))))

            size = min(params["size"], max(0.01, params["inventory"] - abs(inventory)))
            if size <= 0:
                continue
            bid_price = bid / 100.0
            ask_price = ask / 100.0
            if ask_price < true_prob:
                fill_edge = size * (ask_price - true_prob)
                edge += fill_edge
                arb_edge += fill_edge
                inventory -= size
            if bid_price > true_prob:
                fill_edge = size * (true_prob - bid_price)
                edge += fill_edge
                arb_edge += fill_edge
                inventory += size

            arrivals = 1 if rng.random() < rng.uniform(0.154, 0.352) else 0
            for _ in range(arrivals):
                if rng.random() < 0.5:
                    # Retail buy crosses our ask only when we improve or equal
                    # the hidden competitor's visible ask.
                    if ask <= competitor_ask + 1:
                        q = min(size, rng.lognormvariate(1.0, 1.2))
                        fill_edge = q * (ask_price - true_prob)
                        edge += fill_edge
                        retail_edge += fill_edge
                        inventory -= q
                else:
                    if bid >= competitor_bid - 1:
                        q = min(size, rng.lognormvariate(1.0, 1.2) / max(true_prob, 0.05))
                        fill_edge = q * (true_prob - bid_price)
                        edge += fill_edge
                        retail_edge += fill_edge
                        inventory += q
        edges.append(edge)
        retail_edges.append(retail_edge)
        arb_edges.append(arb_edge)
    mean_edge = sum(edges) / len(edges)
    return {
        "official_measured": False,
        "mean_edge": round(mean_edge, 6),
        "mean_arb_edge": round(sum(arb_edges) / len(arb_edges), 6),
        "mean_retail_edge": round(sum(retail_edges) / len(retail_edges), 6),
        "success_count": simulations - failures,
        "failure_count": failures,
        "simulations": simulations,
        "score_source": "local_official_semantics_fallback",
    }


def _params_from_strategy_text(strategy_text: str) -> dict[str, object]:
    values = {}
    for name in ["quote_size", "base_spread_ticks", "inventory_limit", "skew_divisor"]:
        match = re.search(rf"{name}\s*=\s*([0-9.]+)", strategy_text)
        if match:
            values[name] = float(match.group(1))
    mode_match = re.search(r"quote_mode\s*=\s*['\"]([^'\"]+)['\"]", strategy_text)
    return {
        "size": float(values.get("quote_size", 1.0)),
        "spread": int(values.get("base_spread_ticks", 12)),
        "inventory": float(values.get("inventory_limit", 30.0)),
        "skew_divisor": max(1.0, float(values.get("skew_divisor", 8.0))),
        "quote_mode": mode_match.group(1) if mode_match else "outside_competitor",
    }


def _objective_metadata(evaluator_name: str) -> dict[str, object]:
    if evaluator_name == "prediction_market":
        return {
            "objective_name": "prediction_market_mean_edge",
            "objective_direction": "maximize",
            "official_result": {
                "measured": True,
                "profit_usd": None,
                "score_source": "upstream_orderbook_pm_challenge_when_available",
                "required_evaluator": "https://github.com/danrobinson/prediction-market-challenge",
                "reason": "Optimization evaluates generated candidates with the upstream orderbook-pm runner when available.",
            },
            "note": (
                "Prediction-market optimization uses upstream mean edge when the orderbook_pm_challenge repo is available. "
                "The normalized score is derived from mean edge for harness aggregation."
            ),
        }
    if evaluator_name == "length_score":
        return {
            "objective_name": "length_score",
            "objective_direction": "maximize",
            "official_result": {"measured": True, "score_source": "local_deterministic_evaluator"},
            "note": "This evaluator maximizes 1 / token_count for smoke-test optimization.",
        }
    return {
        "objective_name": evaluator_name or "deterministic_score",
        "objective_direction": "maximize",
        "official_result": {"measured": True, "score_source": "local_deterministic_evaluator"},
        "note": "Optimization result from the registered deterministic evaluator.",
    }


def _shorten(text: str, limit: int = 140) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
