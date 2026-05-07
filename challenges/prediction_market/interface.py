from __future__ import annotations


def strategy_variant() -> str:
    """Return a prediction-market strategy description or code sketch.

    Current harness integration evaluates generated variant text. A future
    sandboxed integration can replace this with a concrete Strategy class using
    the upstream `orderbook_pm_challenge.strategy.BaseStrategy` interface.
    """

    return (
        "Adaptive passive market maker: estimate fair value from competitor midpoint "
        "and recent fills, cancel all stale quotes after volatility jumps, quote "
        "moderate size with inventory skew and position limits."
    )
