"""Blinded, schema-only LLM relevance judgments using offline vLLM.

The runtime dependency on vLLM is intentionally imported only when a model is
created. Prompt construction, response parsing, cache keys, and metrics can
therefore be tested without a GPU or model download.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from src.schema.schema_graph import SchemaGraph
from src.signals.candidates import SignalCandidate

JUDGE_PROMPT_VERSION = "signal-relevance-v2"
RUBRIC_VERSION = "signal-relevance-1"

JUDGE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["judged", "unable_to_judge"]},
        "grade": {
            "anyOf": [
                {"type": "integer", "enum": [0, 1, 2, 3]},
                {"type": "null"},
            ]
        },
        "reason": {"type": "string"},
    },
    "required": ["status", "grade", "reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class JudgeModelProfile:
    """Stable local-judge preset, including conservative vLLM defaults."""

    model_name: str
    default_tensor_parallel_size: int
    default_gpu_memory_utilization: float
    default_max_model_len: int = 8192
    default_max_num_seqs: int | None = None


SUPPORTED_MODELS = {
    "mistral": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "qwen3.8-27b-fp8": "Qwen/Qwen3.8-27B-FP8",
    "qwen3.6-35b-a3b-fp8": "Qwen/Qwen3.6-35B-A3B-FP8",
    "muse-glimmer-30b": "meta-models/Muse-Glimmer-30B",
    "glm-4.7-flash": "zai-org/GLM-4.7-Flash",
    "gemma-4-31b": "google/gemma-4-31B-it",
}

MODEL_PROFILES = {
    "mistral": JudgeModelProfile(SUPPORTED_MODELS["mistral"], 1, 0.90),
    # Both FP8 Qwen checkpoints fit comfortably on one 80 GB H100 at this
    # short context length. Retain space for KV cache and batching.
    "qwen3.8-27b-fp8": JudgeModelProfile(
        SUPPORTED_MODELS["qwen3.8-27b-fp8"], 1, 0.85, default_max_num_seqs=64
    ),
    "qwen3.6-35b-a3b-fp8": JudgeModelProfile(
        SUPPORTED_MODELS["qwen3.6-35b-a3b-fp8"], 1, 0.85, default_max_num_seqs=64
    ),
    # The official BF16 checkpoints are approximately 60--63 GB.  They fit on
    # one 80 GB H100 for this short JSON-judging prompt, but only with a
    # deliberately small KV-cache budget.  A 2k context cap and four active
    # sequences leave headroom for model/runtime allocations; callers can
    # raise either after a successful throughput pilot.
    "muse-glimmer-30b": JudgeModelProfile(
        SUPPORTED_MODELS["muse-glimmer-30b"], 1, 0.92,
        default_max_model_len=2048, default_max_num_seqs=4,
    ),
    "glm-4.7-flash": JudgeModelProfile(
        SUPPORTED_MODELS["glm-4.7-flash"], 1, 0.92,
        default_max_model_len=2048, default_max_num_seqs=4,
    ),
    "gemma-4-31b": JudgeModelProfile(
        SUPPORTED_MODELS["gemma-4-31b"], 1, 0.92,
        default_max_model_len=2048, default_max_num_seqs=4,
    ),
}
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh")


@dataclass(frozen=True)
class JudgeConfig:
    """Frozen settings that identify a reproducible set of judgments."""

    model_family: str
    model_name: str
    model_revision: str
    prompt_version: str = JUDGE_PROMPT_VERSION
    rubric_version: str = RUBRIC_VERSION
    seed: int = 0
    max_tokens: int = 256
    temperature: float = 0.0
    reasoning_effort: str = "none"


def describe_schema(
    candidates: Sequence[SignalCandidate], graph: SchemaGraph
) -> str:
    """Return a concise schema description shared by all blinded prompts."""
    attributes: dict[str, set[str]] = {
        table.name: set() for table in graph.get_tables()
    }
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
    return "\n".join([
        "Tables and attributes:", *tables, "Relationships:",
        *(relationships or ["- none declared"]),
    ])


def build_judge_prompt(
    condition: str,
    candidate: SignalCandidate,
    graph: SchemaGraph,
    schema_description: str,
) -> list[dict[str, str]]:
    """Build one method-blind condition/signal judgment prompt."""
    system = (
        "You are an impartial evaluator for relational-schema signal discovery. "
        "Judge one proposed signal for one retrieval or ranking intent. "
        "Use only the stated schema, relationships, intent, and candidate; do not "
        "assume data values, distributions, or hidden fields. Do not infer a method, "
        "rank, score, or provenance beyond the displayed candidate path.\n\n"
        "Relevance grades: 0 = irrelevant; 1 = weak or indirect; "
        "2 = useful but not central; 3 = direct or central. "
        "Use unable_to_judge only when the schema and intent genuinely do not allow "
        "a relevance decision. Return a short, schema-grounded reason. Do not "
        "include private reasoning or extra fields. Return only the required "
        "top-level JSON object."
    )
    user = (
        f"Schema:\n{schema_description}\n\n"
        f"User intent:\n{condition}\n\n"
        f"Candidate signal:\n{candidate.describe(graph)}\n\n"
        "Return exactly one top-level JSON object with exactly these fields: "
        "status, grade, and reason. For status=judged, grade must be an integer "
        "from 0 to 3. For status=unable_to_judge, grade must be null. Do not "
        "wrap the result in another object or add fields."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def chat_template_kwargs(model_family: str, reasoning_effort: str) -> dict[str, Any]:
    """Return only documented model-template controls for the requested effort."""
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(
            f"unsupported reasoning effort {reasoning_effort!r}; "
            f"choose one of {', '.join(REASONING_EFFORTS)}"
        )
    if reasoning_effort == "none":
        if model_family in {"qwen3.8-27b-fp8", "qwen3.6-35b-a3b-fp8"}:
            return {"enable_thinking": False}
        return {}
    if model_family == "qwen3.8-27b-fp8":
        # Qwen 3.8 natively accepts low/medium/xhigh. Its documented maximum
        # effort is xhigh, so map the generic high option to that level.
        return {
            "enable_thinking": True,
            "reasoning_effort": "xhigh" if reasoning_effort == "high" else reasoning_effort,
            "preserve_thinking": False,
        }
    if model_family == "qwen3.6-35b-a3b-fp8":
        # The published Qwen 3.6 interface exposes thinking as an on/off
        # template option, not distinct effort levels.
        if reasoning_effort != "medium":
            raise ValueError(
                "Qwen 3.6 supports --reasoning-effort none (direct) or medium "
                "(thinking enabled); it has no documented graded effort control"
            )
        return {"enable_thinking": True, "preserve_thinking": False}
    raise ValueError(
        f"{model_family} does not expose a supported reasoning-effort control; "
        "use --reasoning-effort none"
    )


def render_chat_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    template_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Use the model's own chat template, failing clearly if it lacks one."""
    try:
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            **dict(template_kwargs or {}),
        )
    except (AttributeError, ValueError, TypeError) as error:
        raise RuntimeError(
            "the selected model tokenizer has no usable chat template; "
            "use a vLLM-compatible instruct checkpoint"
        ) from error


def parse_judge_response(raw_response: str) -> dict[str, Any]:
    """Validate the small public judgment schema; never coerce an invalid grade."""
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).removeprefix("json").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        return {"status": "invalid_response", "grade": None, "reason": None, "parse_error": error.msg}
    if not isinstance(value, dict):
        return {"status": "invalid_response", "grade": None, "reason": None, "parse_error": "response is not a JSON object"}
    status = value.get("status")
    grade = value.get("grade")
    reason = value.get("reason")
    if status == "judged" and isinstance(grade, int) and not isinstance(grade, bool) and grade in range(4):
        if not isinstance(reason, str) or not reason.strip():
            return {"status": "invalid_response", "grade": None, "reason": None, "parse_error": "judged response needs a non-empty reason"}
        return {"status": "judged", "grade": grade, "reason": reason.strip()}
    if status == "unable_to_judge" and grade is None:
        if not isinstance(reason, str) or not reason.strip():
            return {"status": "invalid_response", "grade": None, "reason": None, "parse_error": "unable_to_judge response needs a non-empty reason"}
        return {"status": "unable_to_judge", "grade": None, "reason": reason.strip()}
    return {"status": "invalid_response", "grade": None, "reason": None, "parse_error": "expected judged grade 0..3 or unable_to_judge with null grade"}


def cache_key(
    *,
    condition_id: str,
    condition: str,
    candidate: SignalCandidate,
    schema_version: str,
    config: JudgeConfig,
) -> str:
    """Version all semantic inputs so stale labels are never silently reused."""
    payload = {
        "condition_id": condition_id,
        "condition": condition,
        "candidate_id": candidate.id,
        "schema_version": schema_version,
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "prompt_version": config.prompt_version,
        "rubric_version": config.rubric_version,
        "seed": config.seed,
        "temperature": config.temperature,
        "reasoning_effort": config.reasoning_effort,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def schema_version(
    candidates: Sequence[SignalCandidate], graph: SchemaGraph, candidate_config: Mapping[str, Any]
) -> str:
    """Hash only schema/candidate definitions, never any source records."""
    payload = {
        "graph": graph.to_dict(),
        "candidate_ids": [candidate.id for candidate in candidates],
        "candidate_config": dict(candidate_config),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def create_vllm(model_family: str, model_name: str, model_revision: str, **kwargs: Any) -> Any:
    """Create an offline vLLM engine with the vendor-recommended Mistral mode."""
    if model_family not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported judge model family: {model_family}")
    try:
        from vllm import LLM
    except ImportError as error:
        raise RuntimeError(
            "LLM judging requires vLLM. Install a CUDA/H100-compatible vLLM build "
            "in the SimBench environment."
        ) from error
    options = {
        "model": model_name,
        "revision": model_revision,
        **{key: value for key, value in kwargs.items() if value is not None},
    }
    if model_family == "mistral":
        options.update({
            "tokenizer_mode": "mistral",
            "config_format": "mistral",
            "load_format": "mistral",
        })
    return LLM(**options)


def make_sampling_params(config: JudgeConfig) -> Any:
    """Request schema-constrained, deterministic local generation."""
    try:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as error:
        raise RuntimeError("LLM judging requires vLLM") from error
    return SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        seed=config.seed,
        structured_outputs=StructuredOutputsParams(json=JUDGE_OUTPUT_SCHEMA),
    )


def iter_chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]
