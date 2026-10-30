"""Pointwise generative-LLM scoring for schema signal discovery.

This module deliberately does *not* reuse the relevance-judging prompt or its
0--3 grades. It asks an LLM for an absolute utility estimate for one
condition/signal (or one legal graph action) at a time. The default scale
is 0--10, with 0--100 available as an option; human qrels remain independent of discovery.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.evaluation.llm_judge import (
    LOCAL_VLLM_MODELS,
    MODEL_PROFILES,
    OCI_WAYFLOW_MODELS,
    SUPPORTED_MODELS,
    chat_template_kwargs,
    create_oci_wayflow_judge,
    create_vllm,
    describe_schema,
    render_chat_prompt,
    validate_reasoning_effort,
)
from src.schema.schema_graph import SchemaGraph
from src.signals.candidates import SignalCandidate
from src.signals.scorers import ScoredSignal, _sort_scored

LLM_DISCOVERY_PROMPT_VERSION = "llm-pointwise-discovery-v4"
UTILITY_SCORE_MAXIMA = (10, 100)


def utility_output_schema(score_max: int) -> dict[str, object]:
    """Return the constrained JSON schema for one selected utility scale."""
    _validate_utility_score_max(score_max)
    return {
        "type": "object",
        "properties": {
            "utility_score": {"type": "integer", "minimum": 0, "maximum": score_max},
            # A bounded field makes schema-constrained output finish its JSON object
            # well below the generation cap, rather than allowing an unbounded
            # explanation to be cut off mid-object.
            "evidence": {"type": "string", "minLength": 1, "maxLength": 512},
        },
        "required": ["utility_score", "evidence"],
        "additionalProperties": False,
    }


def _validate_utility_score_max(score_max: int) -> None:
    if score_max not in UTILITY_SCORE_MAXIMA:
        raise ValueError(
            f"utility score maximum must be one of {UTILITY_SCORE_MAXIMA}, got {score_max}"
        )


def utility_scale_instruction(score_max: int, *, noun: str = "evidence") -> str:
    """Describe the selected discrete utility scale without changing the method."""
    _validate_utility_score_max(score_max)
    if score_max == 10:
        return (
            f"Return utility_score from 0 to 10: 0 means no plausible useful {noun}; "
            f"5 means useful but incomplete {noun}; 10 means central, direct {noun}. "
            "Intermediate integer values are allowed."
        )
    return (
        f"Return utility_score from 0 to 100: 0 means no plausible useful {noun}; "
        f"25 means weak or indirect {noun}; 50 means useful but incomplete {noun}; "
        f"75 means strong {noun}; 100 means central, direct {noun}. Intermediate "
        "integer values are allowed."
    )


@dataclass(frozen=True)
class LLMDiscoveryConfig:
    model_family: str
    model_name: str
    model_revision: str = "main"
    prompt_version: str = LLM_DISCOVERY_PROMPT_VERSION
    seed: int = 0
    max_tokens: int = 256
    utility_score_max: int = 10
    temperature: float = 0.0
    reasoning_effort: str = "none"
    batch_size: int = 8
    max_attempts: int = 2
    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    oci_compartment_id: str | None = None
    oci_config: str = "~/.oci/config"
    oci_profile: str = "DEFAULT"

    def __post_init__(self) -> None:
        if self.model_family not in SUPPORTED_MODELS:
            raise ValueError(f"unsupported LLM model family: {self.model_family}")
        validate_reasoning_effort(self.model_family, self.reasoning_effort)
        _validate_utility_score_max(self.utility_score_max)
        if self.batch_size <= 0 or self.max_attempts <= 0 or self.max_tokens <= 0:
            raise ValueError("batch size, attempts, and max tokens must be positive")


def build_signal_prompt(
    condition: str,
    candidate: SignalCandidate,
    graph: SchemaGraph,
    schema_description: str,
    *,
    explored_context: str | None = None,
    utility_score_max: int = 10,
) -> list[dict[str, str]]:
    """Build a frozen pointwise discovery prompt for exactly one signal."""
    system = (
        "You identify useful schema signals for a retrieval or ranking condition. "
        "Score exactly one proposed signal, independently of any other signals. "
        "Use only the supplied schema context, condition, and signal definition. "
        "Do not assume row values, distributions, omitted "
        "joins, hidden fields, aggregation, or comparisons. A signal is evidence "
        "that a downstream model can use; it need not itself perform retrieval. "
        "For a relationship-only signal, credit only the association or connected "
        "rows on its stated path, not unlisted attributes of reached tables. "
        "Consider the starting entity, every path edge, and endpoint together.\n\n"
        f"{utility_scale_instruction(utility_score_max)} Return a concise schema-grounded evidence note. Return "
        "only the required JSON object and do not reveal private reasoning."
    )
    sections = []
    if explored_context is None:
        sections.append(f"Schema:\n{schema_description}")
    else:
        sections.append(f"Exploration record:\n{explored_context}")
    sections.extend((
        f"User condition:\n{condition}",
        f"Signal technical definition:\n{candidate.describe(graph)}",
        f"Signal meaning:\n{candidate.explain_for_human(graph)}",
        f"Return exactly {{\"utility_score\": integer 0..{utility_score_max}, \"evidence\": string}}.",
    ))
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(sections)}]


def build_action_prompt(
    context: str,
    question: str,
    action: str,
    action_kind: str,
    *,
    utility_score_max: int = 10,
) -> list[dict[str, str]]:
    """Score one legal graph action without exposing sibling-action ordering."""
    criteria = {
        "start": (
            "This is a root-selection action, not a signal. Score how promising "
            "the named table is as a starting point for discovering useful signals. "
            "Do not give it zero merely because it has not selected a signal yet."
        ),
        "traverse": (
            "This is a traversal action, not a signal. Score how promising it is "
            "to explore the stated relationship in order to discover useful signals. "
            "Do not give it zero merely because it has not selected a signal yet."
        ),
        "select_attribute": (
            "This is a signal-selection action. Score the stated attribute signal's "
            "ordinary usefulness for the condition."
        ),
        "select_relationship": (
            "This is a relationship-only signal-selection action. Score only the "
            "stated association or connected rows, not unlisted attributes."
        ),
        "stop_branch": (
            "This stops only the current branch. Score it highly only when no legal "
            "non-stop action from this state is worth exploring."
        ),
        "stop_search": (
            "This ends the whole search. Score it highly only when the already "
            "selected signals are sufficient and no legal non-stop action can add "
            "useful evidence."
        ),
    }
    try:
        action_criterion = criteria[action_kind]
    except KeyError as error:
        raise ValueError(f"unknown graph action kind: {action_kind!r}") from error
    system = (
        "You guide a bounded relational-schema graph search. Score exactly one "
        "program-generated legal action for how well it advances discovery of useful "
        "signals for the condition. Use only the supplied state and action. Do not "
        f"invent tables, columns, joins, or actions. {utility_scale_instruction(utility_score_max, noun='progress')} "
        "Return a concise "
        "evidence note and only the required JSON object.\n\n"
        f"Action-specific criterion: {action_criterion}"
    )
    user = (
        f"Search state:\n{context}\n\nDecision question:\n{question}\n\n"
        f"Legal action to score:\n{action}\n\n"
        f"Return exactly {{\"utility_score\": integer 0..{utility_score_max}, \"evidence\": string}}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_utility_response(raw_response: str, *, utility_score_max: int = 10) -> dict[str, Any]:
    """Parse a utility response, preserving a valid bounded score when possible.

    Local structured generation should return valid JSON.  Some models can still
    emit an unescaped quote in the free-text evidence field, however.  In that
    narrow case the numeric score remains useful for ranking, so recover only a
    syntactically valid, in-range ``utility_score`` and explicitly mark the
    evidence as recovered in the result metadata.
    """
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).removeprefix("json").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        return _recover_utility_score(text, error.msg, utility_score_max=utility_score_max)
    if not isinstance(value, dict):
        return {"status": "invalid_response", "parse_error": "response is not a JSON object"}
    score, evidence = value.get("utility_score"), value.get("evidence")
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= utility_score_max:
        return {"status": "invalid_response", "parse_error": f"utility_score must be an integer from 0 to {utility_score_max}"}
    if not isinstance(evidence, str) or not evidence.strip():
        return {"status": "invalid_response", "parse_error": "evidence must be non-empty"}
    return {
        "status": "scored",
        "utility_score": score,
        "evidence": evidence.strip(),
        "parse_status": "strict_json",
    }


def _recover_utility_score(
    text: str, parse_error: str, *, utility_score_max: int = 10
) -> dict[str, Any]:
    """Recover only a valid score when evidence has malformed JSON quoting."""
    match = re.match(
        r'^\s*\{.*?"utility_score"\s*:\s*(?P<score>-?\d+)\b',
        text,
        flags=re.DOTALL,
    )
    if match is None:
        return {"status": "invalid_response", "parse_error": parse_error}
    score = int(match.group("score"))
    if not 0 <= score <= utility_score_max:
        return {"status": "invalid_response", "parse_error": f"utility_score must be an integer from 0 to {utility_score_max}"}

    evidence_match = re.search(r'"evidence"\s*:\s*"(?P<evidence>.*)', text, flags=re.DOTALL)
    if evidence_match is None:
        evidence = "Evidence text could not be parsed from the model response."
    else:
        evidence = evidence_match.group("evidence")
        if evidence.endswith('"}'):
            evidence = evidence[:-2]
        evidence = " ".join(evidence.replace(r'\"', '"').replace(r'\n', " ").split())
        if not evidence:
            evidence = "Evidence text could not be parsed from the model response."

    return {
        "status": "scored",
        "utility_score": score,
        "evidence": evidence[:512],
        "parse_status": "recovered_score",
        "parse_error": parse_error,
    }


class LLMPointwiseSignalScorer:
    """Pointwise utility scorer backed by vLLM or OCI/Wayflow."""

    def __init__(self, config: LLMDiscoveryConfig) -> None:
        self.config = config
        self.model_name = config.model_name
        self._engine: Any | None = None
        self._tokenizer: Any | None = None
        self._oci_model: Any | None = None

    def score(
        self,
        intent: str,
        candidates: Sequence[SignalCandidate],
        graph: SchemaGraph,
        *,
        explored_schema_context: str | None = None,
    ) -> tuple[ScoredSignal, ...]:
        if not candidates:
            return ()
        schema_text = describe_schema(candidates, graph)
        prompts = [
            build_signal_prompt(
                intent,
                candidate,
                graph,
                schema_text,
                explored_context=explored_schema_context,
                utility_score_max=self.config.utility_score_max,
            )
            for candidate in candidates
        ]
        judgments = self._generate_with_retries(prompts)
        scored: list[ScoredSignal] = []
        for candidate, judgment in zip(candidates, judgments, strict=True):
            if judgment["status"] != "scored":
                raise RuntimeError(f"LLM could not score {candidate.id}: {judgment.get('parse_error')}")
            utility = int(judgment["utility_score"])
            scored.append(ScoredSignal(
                candidate,
                utility / self.config.utility_score_max,
                metadata={
                    "score_type": f"llm_pointwise_utility_0_{self.config.utility_score_max}",
                    "utility_score": utility,
                    "evidence": judgment["evidence"],
                    "token_usage": judgment.get("token_usage"),
                    "attempts": judgment["attempts"],
                    "parse_status": judgment.get("parse_status", "unknown"),
                    "parse_error": judgment.get("parse_error"),
                },
            ))
        return _sort_scored(scored)

    def score_actions(
        self,
        context: str,
        question: str,
        actions: Sequence[str],
        action_kinds: Sequence[str],
    ) -> list[dict[str, Any]]:
        if len(actions) != len(action_kinds):
            raise ValueError("each graph action must have exactly one action kind")
        judgments = self._generate_with_retries([
            build_action_prompt(
                context,
                question,
                action,
                action_kind,
                utility_score_max=self.config.utility_score_max,
            )
            for action, action_kind in zip(actions, action_kinds, strict=True)
        ])
        for action, judgment in zip(actions, judgments, strict=True):
            if judgment["status"] != "scored":
                raise RuntimeError(f"LLM could not score legal action {action!r}: {judgment.get('parse_error')}")
        return judgments

    def _generate_with_retries(self, prompts: Sequence[list[dict[str, str]]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any] | None] = [None] * len(prompts)
        usage_totals: list[dict[str, Any] | None] = [None] * len(prompts)
        pending = list(range(len(prompts)))
        for attempt in range(1, self.config.max_attempts + 1):
            if not pending:
                break
            raw = self._generate([prompts[index] for index in pending])
            next_pending: list[int] = []
            for index, response in zip(pending, raw, strict=True):
                if isinstance(response, tuple) and len(response) == 2:
                    response_text, response_usage = response
                else:  # Keep lightweight mock/custom backends compatible.
                    response_text, response_usage = response, None
                usage_totals[index] = _merge_token_usage(
                    usage_totals[index], response_usage
                )
                parsed = parse_utility_response(
                    response_text, utility_score_max=self.config.utility_score_max
                )
                parsed["attempts"] = attempt
                parsed["token_usage"] = usage_totals[index]
                if parsed["status"] == "scored":
                    results[index] = parsed
                else:
                    results[index] = parsed
                    next_pending.append(index)
            pending = next_pending
        return [result or {"status": "invalid_response", "parse_error": "no response", "attempts": self.config.max_attempts} for result in results]

    def _generate(self, messages: Sequence[list[dict[str, str]]]) -> list[tuple[str, dict[str, Any] | None]]:
        if self.config.model_family in LOCAL_VLLM_MODELS:
            return self._generate_vllm(messages)
        if self.config.model_family in OCI_WAYFLOW_MODELS:
            return self._generate_oci(messages)
        raise ValueError(f"unsupported LLM backend: {self.config.model_family}")

    def _generate_vllm(self, messages: Sequence[list[dict[str, str]]]) -> list[tuple[str, dict[str, Any]]]:
        engine = self._get_vllm()
        assert self._tokenizer is not None
        try:
            from vllm import SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
        except ImportError as error:  # pragma: no cover - runtime dependency
            raise RuntimeError("LLM discovery requires vLLM in the SimBench environment") from error
        prompts = [render_chat_prompt(self._tokenizer, message, template_kwargs=chat_template_kwargs(self.config.model_family, self.config.reasoning_effort)) for message in messages]
        sampling = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            seed=self.config.seed,
            structured_outputs=StructuredOutputsParams(
                json=utility_output_schema(self.config.utility_score_max)
            ),
        )
        outputs = engine.generate(prompts, sampling)
        generated = []
        for output in outputs:
            completion = output.outputs[0]
            prompt_tokens = len(output.prompt_token_ids or ())
            completion_tokens = len(completion.token_ids or ())
            generated.append((completion.text, {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "reasoning_tokens": None,
                "reasoning_tokens_available": False,
                "reasoning_token_responses_reported": 0,
                "exact_count": True,
                "requests": 1,
            }))
        return generated

    def _get_vllm(self) -> Any:
        if self._engine is None:
            profile = MODEL_PROFILES[self.config.model_family]
            self._engine = create_vllm(
                self.config.model_family,
                self.config.model_name,
                self.config.model_revision,
                tensor_parallel_size=self.config.tensor_parallel_size or profile.default_tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization or profile.default_gpu_memory_utilization,
                max_model_len=self.config.max_model_len or profile.default_max_model_len,
                max_num_seqs=self.config.max_num_seqs or profile.default_max_num_seqs,
                # Explicit XGrammar is the strict local JSON-schema constraint.
                # Do not rely on prompt wording for output validity.
                structured_outputs_config={"backend": "xgrammar"},
            )
            self._tokenizer = self._engine.get_tokenizer()
        return self._engine

    def _generate_oci(self, messages: Sequence[list[dict[str, str]]]) -> list[tuple[str, dict[str, Any] | None]]:
        model = self._get_oci()
        prompts = [
            _make_wayflow_utility_prompt(
                message, utility_score_max=self.config.utility_score_max
            )
            for message in messages
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.config.batch_size, len(prompts))) as executor:
            responses = list(executor.map(model.generate, prompts))
        return [(_wayflow_text(response), _wayflow_token_usage(response)) for response in responses]

    def _get_oci(self) -> Any:
        if self._oci_model is None:
            if not self.config.oci_compartment_id:
                raise RuntimeError(
                    "set OCI_GENAI_COMPARTMENT or --oci-compartment-id for OCI discovery"
                )
            try:
                import oci
            except ImportError as error:  # pragma: no cover - runtime dependency
                raise RuntimeError("OCI discovery requires the OCI SDK") from error
            oci_config = oci.config.from_file(self.config.oci_config, self.config.oci_profile)
            self._oci_model = create_oci_wayflow_judge(
                model_name=self.config.model_name,
                compartment_id=self.config.oci_compartment_id,
                config_path=self.config.oci_config,
                profile=self.config.oci_profile,
                region=oci_config["region"],
                max_tokens=self.config.max_tokens,
                seed=self.config.seed,
                reasoning_effort=self.config.reasoning_effort,
            )
        return self._oci_model


def _make_wayflow_utility_prompt(
    messages: Sequence[Mapping[str, str]], *, utility_score_max: int = 10
) -> Any:
    _validate_utility_score_max(utility_score_max)
    try:
        from wayflowcore.messagelist import Message
        from wayflowcore.models import Prompt
        from wayflowcore.property import IntegerProperty, ObjectProperty, StringProperty
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError("OCI discovery requires wayflowcore") from error
    return Prompt(
        messages=[Message(role=message["role"], content=message["content"]) for message in messages],
        response_format=ObjectProperty(
            name="signal_utility",
            properties={
                "utility_score": IntegerProperty(
                    description=f"Integer utility score from 0 to {utility_score_max}."
                ),
                "evidence": StringProperty(description="Short schema-grounded evidence."),
            },
            additional_properties=False,
        ),
    )


def _wayflow_text(response: Any) -> str:
    message = getattr(response, "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    for attribute in ("text", "content", "output_text"):
        value = getattr(response, attribute, None)
        if isinstance(value, str):
            return value
    if isinstance(response, str):
        return response
    return str(response)


def _wayflow_token_usage(response: Any) -> dict[str, Any] | None:
    """Read WayFlow's exact OCI token counters when the service returns them."""
    usage = getattr(response, "token_usage", None)
    if usage is None:
        return None
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    raw_reasoning_tokens = getattr(usage, "reasoning_tokens", None)
    # WayFlow normalizes an omitted OCI reasoning-token field to zero, so zero
    # cannot distinguish "reported as zero" from "not provided".
    reasoning_reported = isinstance(raw_reasoning_tokens, int) and raw_reasoning_tokens > 0
    return {
        "input_tokens": input_tokens if isinstance(input_tokens, int) else None,
        "output_tokens": output_tokens if isinstance(output_tokens, int) else None,
        "total_tokens": total_tokens if isinstance(total_tokens, int) else None,
        "reasoning_tokens": raw_reasoning_tokens if reasoning_reported else None,
        "reasoning_tokens_available": reasoning_reported,
        "reasoning_token_responses_reported": int(reasoning_reported),
        "exact_count": bool(getattr(usage, "exact_count", False)),
        "requests": 1,
    }


def _merge_token_usage(
    accumulated: dict[str, Any] | None,
    current: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Sum usage across retries while preserving missing and unknown counters."""
    if current is None:
        current = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "reasoning_tokens": None,
            "reasoning_tokens_available": False,
            "reasoning_token_responses_reported": 0,
            "exact_count": False,
            "requests": 1,
        }
    if accumulated is None:
        return dict(current)
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        left, right = accumulated.get(field), current.get(field)
        accumulated[field] = left + right if isinstance(left, int) and isinstance(right, int) else None
    left_reasoning = accumulated.get("reasoning_tokens")
    right_reasoning = current.get("reasoning_tokens")
    all_reasoning_available = bool(
        accumulated.get("reasoning_tokens_available")
        and current.get("reasoning_tokens_available")
    )
    accumulated["reasoning_tokens"] = (
        left_reasoning + right_reasoning
        if all_reasoning_available
        and isinstance(left_reasoning, int)
        and isinstance(right_reasoning, int)
        else None
    )
    accumulated["reasoning_tokens_available"] = all_reasoning_available
    accumulated["reasoning_token_responses_reported"] = (
        int(accumulated.get("reasoning_token_responses_reported", 0))
        + int(current.get("reasoning_token_responses_reported", 0))
    )
    accumulated["exact_count"] = bool(accumulated.get("exact_count") and current.get("exact_count"))
    accumulated["requests"] = int(accumulated.get("requests", 0)) + int(current.get("requests", 1))
    return accumulated


