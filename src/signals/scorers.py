"""Local scoring backends for ranking schema signal candidates.

The MiniLM backend is a conventional cross-encoder ranker.  The Decider backend
uses a Jev-style typed-decision model and preserves the model's returned option
probabilities.  They intentionally share one ``score`` interface so experiments
compare the model rather than a different candidate-generation procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import exp
import random
from typing import Any, Iterable, Mapping, Protocol, Sequence, TypeVar

from src.schema.schema_graph import SchemaGraph
from src.signals.candidates import SignalCandidate

__all__ = [
    "DECIDER_BACKENDS",
    "DeciderSignalScorer",
    "MiniLMCrossEncoderScorer",
    "RandomSignalScorer",
    "ScoredSignal",
    "SignalScorer",
    "create_choice_decider_client",
    "default_decider_model",
]


# The sole supported typed-choice backend exposes a categorical decision and a
# calibrated probability for every offered option.
DECIDER_BACKENDS = ("strands",)
_DEFAULT_DECIDER_MODELS = {
    "strands": "StrandsAgents/strands-decider-E4B-gemma4-v1-2610",
}


def default_decider_model(backend: str) -> str:
    """Return the documented default checkpoint for a typed-choice backend."""
    try:
        return _DEFAULT_DECIDER_MODELS[backend]
    except KeyError as error:
        raise ValueError(
            f"unsupported Decider backend {backend!r}; expected one of {DECIDER_BACKENDS}"
        ) from error

@dataclass(frozen=True)
class ScoredSignal:
    """A candidate plus the scalar score used to rank it."""

    candidate: SignalCandidate
    score: float
    ranking_logit: float | None = None
    metadata: Mapping[str, object] | None = None

    def to_dict(self, graph: SchemaGraph) -> dict[str, object]:
        return {
            **self.candidate.to_dict(graph),
            "score": self.score,
            **(
                {"ranking_logit": self.ranking_logit}
                if self.ranking_logit is not None
                else {}
            ),
            **(dict(self.metadata) if self.metadata is not None else {}),
        }


class RandomSignalScorer:
    """Stable random ordering for a reproducible ranking baseline."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.model_name = "random"

    def score(
        self,
        intent: str,
        candidates: Sequence[SignalCandidate],
        graph: SchemaGraph,
    ) -> tuple[ScoredSignal, ...]:
        del graph
        digest = hashlib.sha256(f"{self.seed}:{intent}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        return _sort_scored(
            [ScoredSignal(candidate, rng.random()) for candidate in candidates]
        )


class SignalScorer(Protocol):
    """A backend that ranks a fixed list of schema signals for one intent."""

    def score(
        self,
        intent: str,
        candidates: Sequence[SignalCandidate],
        graph: SchemaGraph,
    ) -> tuple[ScoredSignal, ...]: ...


class MiniLMCrossEncoderScorer:
    """Cross-encoder ranking with a Hugging Face sequence-classification model.

    The default is the common ``ms-marco-MiniLM-L-6-v2`` relevance model.  Its
    sigmoid-transformed logit ranks candidates; it is *not* a calibrated
    probability that a signal is truly relevant.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.device_name = device
        self.batch_size = batch_size
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def score(
        self,
        intent: str,
        candidates: Sequence[SignalCandidate],
        graph: SchemaGraph,
    ) -> tuple[ScoredSignal, ...]:
        if not candidates:
            return ()
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        device = self._model.device

        scored: list[ScoredSignal] = []
        query_context = _format_intent_context(intent, candidates, graph)
        for batch in _batches(candidates, self.batch_size):
            signal_texts = [candidate.describe(graph) for candidate in batch]
            encoded = self._tokenizer(
                [query_context] * len(batch),
                signal_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits
            values = _relevance_logits(logits, self._torch)
            ranking_logits = _ranking_logits(logits)
            for candidate, value, ranking_logit in zip(
                batch, values.tolist(), ranking_logits.tolist(), strict=True
            ):
                score = _sigmoid(float(value))
                scored.append(
                    ScoredSignal(candidate, score, float(ranking_logit))
                )
        return _sort_scored(scored)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:  # pragma: no cover - dependency environment
            raise RuntimeError(
                "MiniLM scoring requires torch and transformers; install them first"
            ) from error
        device = _resolve_device(self.device_name, torch)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(device)
        self._model.eval()
        self._torch = torch


class DeciderSignalScorer:
    """Score signals with a local typed-choice System One model.

    The model makes one categorical choice among the supplied signal candidates
    and returns a probability distribution across that exact list. Candidate
    scores are those selection probabilities. The supported ``strands``
    backend uses the Strands typed-choice runtime required by its Gemma-4
    checkpoint.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        backend: str = "strands",
        device: str = "auto",
        client: Any | None = None,
    ) -> None:
        if backend not in DECIDER_BACKENDS:
            raise ValueError(
                f"unsupported Decider backend {backend!r}; expected one of {DECIDER_BACKENDS}"
            )
        self.backend = backend
        self.model_name = model_name or default_decider_model(backend)
        self.device = device
        self._client = client

    def score(
        self,
        intent: str,
        candidates: Sequence[SignalCandidate],
        graph: SchemaGraph,
        *,
        history_context: str | None = None,
        explored_schema_context: str | None = None,
    ) -> tuple[ScoredSignal, ...]:
        if not candidates:
            return ()
        if len(candidates) == 1:
            # A one-option categorical choice has probability 1 by definition;
            # the Decider API itself requires at least two options.
            return (ScoredSignal(candidates[0], 1.0),)
        client = self._get_client()
        if history_context is not None and explored_schema_context is not None:
            raise ValueError("pass only one graph-rerank context")
        if explored_schema_context is not None:
            state = _format_explored_schema_rerank_context(
                intent, explored_schema_context
            )
        elif history_context is not None:
            # Retain the old explicit context argument for callers that use the
            # scorer directly. The graph-search pipeline uses the structured view.
            state = _format_history_rerank_context(intent, history_context)
        else:
            state = _format_intent_context(intent, candidates, graph)
        if len(candidates) > 255:
            raise ValueError(
                "Decider multi-choice mode accepts at most 255 candidates; reduce "
                "the hop limit or select fewer source tables"
            )
        options = [candidate.describe(graph) for candidate in candidates]
        answer = _answers(
            client.decide(
                state,
                [
                    {
                        "question": (
                            "Which schema signal is most useful for satisfying the "
                            "user intent?"
                        ),
                        "options": options,
                    }
                ],
            )
        )[0]
        raw_probabilities = _probabilities(answer)
        by_option = {option: candidate for option, candidate in zip(options, candidates)}
        scored = []
        for option, candidate in by_option.items():
            score = raw_probabilities.get(option, 0.0)
            scored.append(ScoredSignal(candidate, score))
        return _sort_scored(scored)

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = create_choice_decider_client(
                self.backend, self.model_name, device=self.device
            )
        return self._client


class _StrandsChoiceClient:
    """Adapt Strands' typed-choice API to the small internal ``decide`` protocol.

    Strands accepts many questions over one ``state`` in a single forward pass,
    sharing the state prefix/KV cache.  The current callers usually ask one
    question at a time because graph actions depend on the preceding answer,
    but keeping the adapter batch-capable avoids a different protocol later.
    """

    def __init__(self, model_name: str, *, device: str = "auto") -> None:
        try:
            from strands_decider.infer import load_engine
        except ImportError as error:  # pragma: no cover - dependency environment
            raise RuntimeError(
                "Strands Decider scoring requires the Gemma-4-capable package from "
                "https://github.com/strands-labs/strands-decider; see the README "
                "installation command."
            ) from error
        self.model_name = model_name
        self.device = _resolve_strands_device(device)
        self._engine = load_engine(model_name, device=self.device)

    def decide(
        self, state: str, questions: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, object]]:
        if not questions:
            return []
        try:
            from strands_decider.schema import ChoiceQuestion
        except ImportError as error:  # pragma: no cover - dependency environment
            raise RuntimeError("Strands Decider is installed incompletely") from error

        typed_questions: dict[str, Any] = {}
        original_options: dict[str, list[str]] = {}
        for index, question in enumerate(questions):
            raw_options = question.get("options")
            instruction = question.get("question")
            if not isinstance(instruction, str) or not isinstance(raw_options, Sequence):
                raise RuntimeError(f"unexpected typed-choice question: {question!r}")
            options = [str(option) for option in raw_options]
            if not 2 <= len(options) <= 255:
                raise ValueError(
                    "Strands choice decisions require 2 to 255 options; "
                    f"got {len(options)}"
                )
            if len(set(options)) != len(options):
                raise ValueError("Strands choice options must be unique")
            name = f"question_{index}"
            # Stable internal keys protect against the model interpreting a
            # lengthy human-readable option as an identifier. The descriptions
            # are still exactly the candidate strings seen by the model.
            criteria = {
                f"option_{option_index}": option
                for option_index, option in enumerate(options)
            }
            typed_questions[name] = ChoiceQuestion(
                instructions=instruction,
                criteria=criteria,
            )
            original_options[name] = options

        response = self._engine.ask(state, typed_questions)
        answers: list[dict[str, object]] = []
        for name, options in original_options.items():
            answer = response.answers[name]
            raw_probabilities = getattr(answer, "probabilities", None)
            raw_choice = getattr(answer, "choice", None)
            if not isinstance(raw_probabilities, Mapping) or not isinstance(raw_choice, str):
                raise RuntimeError(f"Strands returned a non-choice answer: {answer!r}")
            probabilities = {
                option: float(raw_probabilities.get(f"option_{index}", 0.0))
                for index, option in enumerate(options)
            }
            choice_index = raw_choice.removeprefix("option_")
            if not choice_index.isdigit() or int(choice_index) >= len(options):
                raise RuntimeError(f"Strands selected an invalid option: {raw_choice!r}")
            answers.append(
                {"choice": options[int(choice_index)], "probs": probabilities}
            )
        return answers


def create_choice_decider_client(
    backend: str,
    model_name: str | None = None,
    *,
    device: str = "auto",
) -> Any:
    """Load a local typed-choice client for the selected backend."""
    resolved_model = model_name or default_decider_model(backend)
    if backend == "strands":
        return _StrandsChoiceClient(resolved_model, device=device)
    raise ValueError(
        f"unsupported Decider backend {backend!r}; expected one of {DECIDER_BACKENDS}"
    )


def _resolve_strands_device(device: str) -> str:
    """Map the shared CLI device option to Strands' explicit device names."""
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:  # pragma: no cover - dependency environment
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _answers(result: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(result, Mapping):
        result = result.get("answers", result)
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
        raise RuntimeError(f"unexpected Decider response: {result!r}")
    if not all(isinstance(answer, Mapping) for answer in result):
        raise RuntimeError(f"unexpected Decider answers: {result!r}")
    return result


def _probabilities(answer: Mapping[str, Any]) -> dict[str, float]:
    raw = answer.get("probs", answer.get("probabilities"))
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"Decider answer has no option probabilities: {answer!r}")
    return {str(key): float(value) for key, value in raw.items()}


def _describe_schema(
    candidates: Sequence[SignalCandidate], graph: SchemaGraph
) -> str:
    """Summarize the same available schema for every condition's decision prompt."""
    attributes: dict[str, set[str]] = {table.name: set() for table in graph.get_tables()}
    for candidate in candidates:
        if candidate.kind == "attribute":
            assert candidate.attribute_table is not None
            assert candidate.attribute_column is not None
            attributes[candidate.attribute_table].add(candidate.attribute_column)

    tables = [
        f"- {table.name}: {', '.join(sorted(attributes[table.name])) or '(no candidate attributes)'}"
        for table in graph.get_tables()
    ]
    relationships = [
        f"- {edge.source_table}.{edge.source_column} references "
        f"{edge.target_table}.{edge.target_column}"
        for edge in graph.get_edges()
    ]
    lines = ["Tables and attributes:", *tables, "Relationships:"]
    lines.extend(relationships or ["- none declared"])
    return "\n".join(lines)


def _format_intent_context(
    intent: str, candidates: Sequence[SignalCandidate], graph: SchemaGraph
) -> str:
    """Build the shared schema-and-intent context supplied to both scorers."""
    return (
        "You select useful signals from a relational database schema for a "
        "retrieval or ranking task. Use the schema and relationship paths below "
        "to interpret each candidate. A path shows how its attribute or related "
        "table is reached by joining tables. Choose based on the user's intent; "
        "do not assume that a particular table is always the query entity. "
        "Each candidate description defines exactly what its signal contains. "
        "A relationship-only candidate represents only the association along its "
        "path; it does not include attributes from reached tables unless the "
        "candidate explicitly names one.\n\n"
        f"Schema:\n{_describe_schema(candidates, graph)}\n\n"
        f"User intent: {intent}"
    )


def _format_history_rerank_context(intent: str, history_context: str) -> str:
    """Prompt Decider to rank the shortlist using search context, not the schema."""
    if not history_context.strip():
        raise ValueError("history_context must not be empty")
    return (
        "You are performing a final ranking after a constrained graph search. "
        "Rank only the candidate signals offered as options. Use the user's intent "
        "and the recorded graph-search history to order them by usefulness. Do not "
        "invent or add signals, and do not assume that an unlisted signal was found.\n\n"
        f"User intent: {intent}\n\n"
        f"Graph-search history:\n{history_context}"
    )


def _format_explored_schema_rerank_context(
    intent: str, explored_schema_context: str
) -> str:
    """Rank the shortlist using an explored schema view, without search choices."""
    if not explored_schema_context.strip():
        raise ValueError("explored_schema_context must not be empty")
    return (
        "You are ranking only the candidate signals supplied as options after a "
        "constrained graph search. Use the intent, the explored schema view, and "
        "each option's explicit signal definition to rank them by usefulness. "
        "The schema view is background context; it does not add information to a "
        "candidate signal. A relationship-only option represents only its stated "
        "association or connected rows, not attributes of the reached table. "
        "Count an attribute as part of a signal only when that option explicitly "
        "names it. Do not invent or add signals.\n\n"
        f"User intent: {intent}\n\n"
        f"Explored schema view:\n{explored_schema_context}"
    )


T = TypeVar("T")


def _batches(values: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _relevance_logits(logits: Any, torch: Any) -> Any:
    if logits.ndim != 2:
        raise RuntimeError(f"expected two-dimensional logits, got {logits.shape!r}")
    if logits.shape[1] == 1:
        return logits[:, 0]
    if logits.shape[1] == 2:
        return torch.log_softmax(logits, dim=1)[:, 1]
    raise RuntimeError(
        "MiniLM model must have one relevance logit or two classification logits; "
        f"got {logits.shape[1]}"
    )


def _ranking_logits(logits: Any) -> Any:
    """Return unsquashed ranking logits for normalizing across candidates."""
    if logits.ndim != 2:
        raise RuntimeError(f"expected two-dimensional logits, got {logits.shape!r}")
    if logits.shape[1] == 1:
        return logits[:, 0]
    if logits.shape[1] == 2:
        return logits[:, 1] - logits[:, 0]
    raise RuntimeError(
        "MiniLM model must have one relevance logit or two classification logits; "
        f"got {logits.shape[1]}"
    )


def _resolve_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    exponent = exp(value)
    return exponent / (1.0 + exponent)


def _sort_scored(scored: Sequence[ScoredSignal]) -> tuple[ScoredSignal, ...]:
    return tuple(sorted(scored, key=lambda item: (-item.score, item.candidate.id)))
