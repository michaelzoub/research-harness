from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass


PARAM_RE = re.compile(r"(?P<name>spread|offset|size|quantity|limit|inventory|skew|guard|fade)\s*[=:]\s*(?P<value>-?\d+(?:\.\d+)?)", re.I)


@dataclass(frozen=True)
class StrategyProfile:
    bid_offset: int
    ask_offset: int
    quantity: float
    inventory_limit: float
    inventory_skew: float
    adaptive: float
    jump_guard: float
    cancel_rate: float


def prediction_market_score(payload: str) -> float:
    """Score a strategy idea against a deterministic prediction-market proxy.

    The upstream challenge evaluates Python market-making strategies on a FIFO
    YES limit order book. This evaluator keeps the same core objective: earn
    edge from retail flow while avoiding adverse selection from stale quotes
    after fair-value moves. It accepts strategy/code variant text and maps it to
    a compact parameterized strategy profile.
    """

    profile = _profile_from_payload(payload)
    raw_edges = [_simulate_regime(profile, seed) for seed in range(24)]
    mean_edge = sum(raw_edges) / len(raw_edges)
    risk_penalty = _risk_penalty(profile, raw_edges)
    normalized = 0.5 + (math.tanh((mean_edge - risk_penalty) / 90.0) * 0.45)
    return round(max(0.0, min(1.0, normalized)), 3)


def _profile_from_payload(payload: str) -> StrategyProfile:
    text = payload.lower()
    params = {match.group("name").lower(): float(match.group("value")) for match in PARAM_RE.finditer(payload)}
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    jitter = int(digest[:4], 16) / 0xFFFF

    adaptive_terms = {
        "adaptive",
        "infer",
        "bayesian",
        "estimate",
        "moving",
        "fill",
        "order flow",
        "re-anchor",
        "reanchor",
        "fair value",
        "midpoint",
    }
    guard_terms = {"jump", "stale", "adverse", "arb", "arbitrageur", "cancel", "volatility", "shock"}
    skew_terms = {"inventory", "skew", "risk", "hedge", "position", "limit"}

    adaptive = _term_score(text, adaptive_terms, base=0.18, step=0.1, cap=0.92)
    jump_guard = _term_score(text, guard_terms, base=0.08, step=0.11, cap=0.9)
    inventory_skew = _term_score(text, skew_terms, base=0.05, step=0.12, cap=0.85)
    if "cancelall" in text or "cancel all" in text:
        jump_guard = min(0.95, jump_guard + 0.1)

    bid_offset = int(params.get("offset", params.get("spread", 2 + round(jitter * 3))))
    ask_offset = int(params.get("spread", bid_offset))
    quantity = params.get("quantity", params.get("size", 4.0 + (jitter * 4.0)))
    inventory_limit = params.get("inventory", params.get("limit", 70.0 + (jitter * 80.0)))
    cancel_rate = min(1.0, max(0.15, 0.35 + jump_guard * 0.5))

    return StrategyProfile(
        bid_offset=max(1, min(12, bid_offset)),
        ask_offset=max(1, min(12, ask_offset)),
        quantity=max(0.5, min(30.0, quantity)),
        inventory_limit=max(10.0, min(300.0, inventory_limit)),
        inventory_skew=inventory_skew,
        adaptive=adaptive,
        jump_guard=jump_guard,
        cancel_rate=cancel_rate,
    )


def _simulate_regime(profile: StrategyProfile, seed: int) -> float:
    rng = random.Random(10_000 + seed)
    true_prob = 0.5 + rng.uniform(-0.04, 0.04)
    estimate = 0.5
    inventory = 0.0
    edge = 0.0

    jump_intensity = rng.uniform(0.004, 0.025)
    jump_sigma = rng.uniform(0.025, 0.12)
    retail_rate = rng.uniform(0.18, 0.55)
    competitor_spread = rng.choice([2, 3, 4, 5, 6])
    competitor_mid = 0.5

    for step in range(220):
        if rng.random() < jump_intensity:
            true_prob += rng.gauss(0.0, jump_sigma)
        true_prob += rng.gauss(0.0, 0.012)
        true_prob = max(0.02, min(0.98, true_prob))

        visible_mid = competitor_mid + (true_prob - competitor_mid) * (0.08 + profile.adaptive * 0.3)
        estimate = (estimate * (1.0 - profile.adaptive * 0.18)) + (visible_mid * profile.adaptive * 0.18)
        inventory_pressure = (inventory / profile.inventory_limit) * profile.inventory_skew

        bid = estimate - (profile.bid_offset / 100.0) - inventory_pressure * 0.03
        ask = estimate + (profile.ask_offset / 100.0) - inventory_pressure * 0.03
        bid = max(0.01, min(0.98, bid))
        ask = max(bid + 0.01, min(0.99, ask))

        stale_bid = max(0.0, bid - true_prob)
        stale_ask = max(0.0, true_prob - ask)
        adverse_fill_size = profile.quantity * (1.0 - profile.jump_guard * profile.cancel_rate)
        if stale_bid > 0:
            edge -= stale_bid * adverse_fill_size
            inventory += adverse_fill_size
        if stale_ask > 0:
            edge -= stale_ask * adverse_fill_size
            inventory -= adverse_fill_size

        if rng.random() < retail_rate:
            retail_buy = rng.random() < 0.5
            retail_size = min(profile.quantity, rng.lognormvariate(1.0, 0.45))
            if retail_buy and abs(inventory - retail_size) <= profile.inventory_limit:
                edge += max(0.0, ask - true_prob) * retail_size
                inventory -= retail_size
                estimate = (estimate * 0.96) + (ask * 0.04)
            elif not retail_buy and abs(inventory + retail_size) <= profile.inventory_limit:
                edge += max(0.0, true_prob - bid) * retail_size
                inventory += retail_size
                estimate = (estimate * 0.96) + (bid * 0.04)

        if step % 20 == 0:
            competitor_mid += rng.gauss(0.0, 0.002)
            competitor_mid = max(0.35, min(0.65, competitor_mid))
        edge -= abs(inventory) * 0.00008 * (1.0 - profile.inventory_skew * 0.45)
        edge += max(0.0, (competitor_spread / 100.0) - ((profile.bid_offset + profile.ask_offset) / 200.0)) * 0.015

    return edge * 100.0


def _risk_penalty(profile: StrategyProfile, raw_edges: list[float]) -> float:
    mean = sum(raw_edges) / len(raw_edges)
    variance = sum((edge - mean) ** 2 for edge in raw_edges) / len(raw_edges)
    volatility_penalty = math.sqrt(variance) * 0.12
    size_penalty = max(0.0, profile.quantity - 12.0) * 0.08
    inventory_penalty = max(0.0, profile.inventory_limit - 160.0) * 0.004
    return volatility_penalty + size_penalty + inventory_penalty


def _term_score(text: str, terms: set[str], base: float, step: float, cap: float) -> float:
    hits = sum(1 for term in terms if term in text)
    return min(cap, base + hits * step)
