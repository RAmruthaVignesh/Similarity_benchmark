"""Validated model profiles used by signal-discovery entry points.

The command-line interface deliberately selects one profile with ``--model``.
Each profile binds a public, reproducible model name to the runner and concrete
checkpoint/service that can execute it.  Raw Hugging Face identifiers are not
accepted at the CLI: a new compatible model must be added here explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.llm_judge import SUPPORTED_MODELS


CATALOG_RANK = "catalog-rank"
GRAPH_SEARCH = "graph-search"
GRAPH_SEARCH_ORACLE_ROOT = "graph-search-oracle-root"
DISCOVERY_METHODS = (CATALOG_RANK, GRAPH_SEARCH, GRAPH_SEARCH_ORACLE_ROOT)
GRAPH_METHODS = frozenset({GRAPH_SEARCH, GRAPH_SEARCH_ORACLE_ROOT})


@dataclass(frozen=True)
class DiscoveryModelProfile:
    """One supported, reproducible model choice for signal discovery."""

    name: str
    runner: str
    model_id: str | None
    supported_methods: frozenset[str]
    llm_family: str | None = None


_CATALOG_ONLY = frozenset({CATALOG_RANK})
_ALL_METHODS = frozenset(DISCOVERY_METHODS)

DISCOVERY_MODEL_PROFILES: dict[str, DiscoveryModelProfile] = {
    "random": DiscoveryModelProfile("random", "random", None, _CATALOG_ONLY),
    "minilm": DiscoveryModelProfile(
        "minilm", "minilm", "cross-encoder/ms-marco-MiniLM-L-6-v2", _CATALOG_ONLY
    ),
    "strands-decider": DiscoveryModelProfile(
        "strands-decider",
        "strands-decider",
        "StrandsAgents/strands-decider-E4B-gemma4-v1-2610",
        _ALL_METHODS,
    ),
    "mistral-small-3.2": DiscoveryModelProfile(
        "mistral-small-3.2", "llm", SUPPORTED_MODELS["mistral"], _ALL_METHODS, "mistral"
    ),
    "muse-glimmer-30b": DiscoveryModelProfile(
        "muse-glimmer-30b", "llm", SUPPORTED_MODELS["muse-glimmer-30b"], _ALL_METHODS,
        "muse-glimmer-30b",
    ),
    "gemma-4-31b": DiscoveryModelProfile(
        "gemma-4-31b", "llm", SUPPORTED_MODELS["gemma-4-31b"], _ALL_METHODS,
        "gemma-4-31b",
    ),
    "gpt-oss-120b": DiscoveryModelProfile(
        "gpt-oss-120b", "llm", SUPPORTED_MODELS["gpt-oss-120b"], _ALL_METHODS,
        "gpt-oss-120b",
    ),
    "gpt-6-luna": DiscoveryModelProfile(
        "gpt-6-luna", "llm", SUPPORTED_MODELS["gpt-6-luna"], _ALL_METHODS, "gpt-6-luna"
    ),
}


def resolve_discovery_model(name: str) -> DiscoveryModelProfile:
    """Return a supported profile or a compact actionable error."""
    try:
        return DISCOVERY_MODEL_PROFILES[name]
    except KeyError as error:
        choices = ", ".join(DISCOVERY_MODEL_PROFILES)
        raise ValueError(f"unsupported discovery model {name!r}; choose one of: {choices}") from error
