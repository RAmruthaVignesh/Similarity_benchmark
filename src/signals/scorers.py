"""Local scoring backends for ranking schema signal candidates.

The MiniLM backend is a conventional cross-encoder ranker.  The Decider backend
uses a Jev-style typed-decision model and preserves the model's returned option
probabilities.  They intentionally share one ``score`` interface so experiments
compare the model rather than a different candidate-generation procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any, Iterable, Mapping, Protocol, Sequence, TypeVar

from src.schema.schema_graph import SchemaGraph
from src.signals.candidates import SignalCandidate

__all__ = [
    "DeciderSignalScorer",
    "MiniLMCrossEncoderScorer",
    "ScoredSignal",
    "SignalScorer",
]

@dataclass(frozen=True)
class ScoredSignal:
    """A candidate plus the scalar score used to rank it."""

    candidate: SignalCandidate
    score: float

    def to_dict(self, graph: SchemaGraph) -> dict[str, object]:
        return {
            **self.candidate.to_dict(graph),
            "score": self.score,
        }


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
        for batch in _batches(candidates, self.batch_size):
            signal_texts = [candidate.describe(graph) for candidate in batch]
            encoded = self._tokenizer(
                [intent] * len(batch),
                signal_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits
            values = _relevance_logits(logits, self._torch)
            for candidate, value in zip(batch, values.tolist(), strict=True):
                score = _sigmoid(float(value))
                scored.append(ScoredSignal(candidate, score))
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
    """Score signals with a local Decider-compatible System One model.

    The model makes one categorical choice among the supplied signal candidates
    and returns a probability distribution across that exact list. Candidate
    scores are those selection probabilities, so there can be at most 255
    candidates in one decision.
    """

    def __init__(
        self,
        model_name: str = "Mapika/decider-4b",
        *,
        client: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self._client = client

    def score(
        self,
        intent: str,
        candidates: Sequence[SignalCandidate],
        graph: SchemaGraph,
    ) -> tuple[ScoredSignal, ...]:
        if not candidates:
            return ()
        client = self._get_client()
        state = (
            "You select database schema signals for a retrieval or ranking task. "
            f"The user intent is: {intent}"
        )
        if len(candidates) > 255:
            raise ValueError(
                "Decider multi-choice mode accepts at most 255 candidates; reduce "
                "the hop limit or select fewer source tables"
            )
        options = [
            f"{candidate.id}: {candidate.describe(graph)}" for candidate in candidates
        ]
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
            try:
                from decider.infer import Decider
            except ImportError as error:  # pragma: no cover - dependency environment
                raise RuntimeError(
                    "Decider scoring requires the decider-ai package; install it "
                    "with the project dependencies first"
                ) from error
            self._client = Decider(self.model_name)
        return self._client


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
