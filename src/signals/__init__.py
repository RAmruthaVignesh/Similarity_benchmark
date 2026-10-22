"""Schema signal discovery and scoring primitives."""

from src.signals.candidates import CATALOG_VIEWS, SignalCandidate, enumerate_signal_candidates
from src.signals.graph_navigation import DeciderGraphNavigator, GraphDiscoveryResult
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
    "GraphDiscoveryResult",
    "MiniLMCrossEncoderScorer",
    "ScoredSignal",
    "SignalCandidate",
    "enumerate_signal_candidates",
    "create_choice_decider_client",
    "default_decider_model",
]
