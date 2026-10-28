"""Schema signal discovery and scoring primitives."""

from __future__ import annotations

from typing import Any

from src.signals.candidates import CATALOG_VIEWS, SignalCandidate, enumerate_signal_candidates
from src.signals.scorers import (
    DECIDER_BACKENDS,
    DeciderSignalScorer,
    MiniLMCrossEncoderScorer,
    ScoredSignal,
    create_choice_decider_client,
    default_decider_model,
)

__all__ = [
    "DeciderSignalScorer",
    "CATALOG_VIEWS",
    "DECIDER_BACKENDS",
    "DeciderGraphNavigator",
    "LLMGraphNavigator",
    "LLMDiscoveryConfig",
    "LLMPointwiseSignalScorer",
    "GraphDiscoveryResult",
    "MiniLMCrossEncoderScorer",
    "ScoredSignal",
    "SignalCandidate",
    "enumerate_signal_candidates",
    "create_choice_decider_client",
    "default_decider_model",
]


def __getattr__(name: str) -> Any:
    """Lazily expose graph and LLM classes without creating an eval import cycle."""
    if name in {"DeciderGraphNavigator", "GraphDiscoveryResult", "LLMGraphNavigator"}:
        from src.signals import graph_navigation

        return getattr(graph_navigation, name)
    if name in {"LLMDiscoveryConfig", "LLMPointwiseSignalScorer"}:
        from src.signals import llm_discovery

        return getattr(llm_discovery, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
