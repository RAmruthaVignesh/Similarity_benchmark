"""Blinded, schema-only LLM relevance judgments.

Local models are served by offline vLLM. OCI-hosted models are called through
WayFlow, with dependencies imported only when the corresponding backend is
created. Prompt construction, response parsing, cache keys, and metrics can
therefore be tested without a GPU, model download, or OCI credentials.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from src.schema.schema_graph import SchemaGraph
from src.signals.candidates import SignalCandidate

JUDGE_PROMPT_VERSION = "signal-relevance-v7"
RUBRIC_VERSION = "signal-evidence-rubric-1"
PROMPT_VARIANTS = ("standard", "paraphrase-a", "paraphrase-b")

_SYSTEM_PROMPT_VARIANTS = {
    "standard": (
        "You are an impartial evaluator of relational-schema signals. Judge one "
        "candidate for one retrieval or ranking intent.\n\n"
        "Evidence limits: Use only the supplied schema, declared relationships, "
        "intent, and candidate. Do not assume records, values, distributions, or "
        "hidden fields. A signal is evidence a downstream retrieval or ranking "
        "model can use, not a complete retrieval procedure; it need not itself "
        "compare records, aggregate values, or rank results.\n\n"
        "Assessment: Infer the sought records from the intent when it says what "
        "is being sought; do not assume a separate target type. Assess the "
        "candidate's source, every relationship in its displayed path, and its "
        "endpoint together. Account for a route that does not fit the intent.\n\n"
        "Candidate meaning: The technical definition and plain-language "
        "explanation describe the same signal. The explanation restates its "
        "source, path, and endpoint; it is a clearer rendering, not extra "
        "evidence. A relationship-only signal supplies only the association or "
        "connected-row information along its displayed path. It does not include "
        "attributes of reached tables unless an attribute is explicitly named as "
        "the endpoint. Other schema attributes are context only. Do not assume "
        "omitted joins, values, aggregation, or comparisons.\n\n"
        "Route interpretation: A direct attribute of the records named in the "
        "intent can be useful even with zero hops. An indirect route is not "
        "automatically irrelevant, and hop count alone is neither a benefit nor "
        "a penalty. Do not infer a method, rank, score, or provenance beyond the "
        "displayed candidate path."
    ),
    "paraphrase-a": (
        "Act as a neutral reviewer of relational-schema evidence. Give a "
        "relevance judgment for this one proposed signal and this one "
        "retrieval/ranking request.\n\n"
        "Use only the schema, its declared links, the request, and the candidate "
        "provided here. No row contents, field values, distributions, or "
        "unlisted fields may be presumed. The signal is a piece of evidence a "
        "retrieval or ranking system could consume, not the whole system: it "
        "does not have to perform comparisons, aggregation, or ranking itself.\n\n"
        "First determine from the request which records are wanted; there is no "
        "separate target-entity label to assume. Then judge the source entity, "
        "each link in the candidate's path, and its endpoint as one definition. "
        "A route that points to the wrong population can weaken its relevance.\n\n"
        "The technical form and plain-language form are two renderings of one "
        "candidate. The plain-language form restates the same source, path, and "
        "endpoint; it does not add evidence. If the candidate ends in a "
        "relationship rather than a named attribute, count only the association "
        "or connected-row information on that path—not attributes from tables "
        "it reaches. Attributes elsewhere in the schema do not become part of "
        "the candidate. Do not fill in missing joins or presume values, "
        "aggregates, or comparisons.\n\n"
        "A zero-hop attribute on the records requested can still be useful. A "
        "multi-hop signal is not irrelevant just because it is indirect, and "
        "path length by itself should not raise or lower its grade. Do not add "
        "a method, ranking, score, or provenance that the displayed path does "
        "not specify."
    ),
    "paraphrase-b": (
        "Review a single schema signal for its usefulness to one user intent, "
        "remaining impartial. The intent describes a retrieval or ranking task.\n\n"
        "Ground the judgment solely in the given schema and declared "
        "relationships, the intent, and the candidate. Do not rely on example "
        "rows, assumed values or distributions, or fields not shown. Treat the "
        "signal as input evidence for a downstream retriever or ranker, not as "
        "a self-contained retrieval algorithm; it need not compare or aggregate "
        "records or produce ranked results on its own.\n\n"
        "Read the intent to identify the requested records rather than supplying "
        "a separate target type. Relevance depends on the candidate's starting "
        "entity, the full displayed relationship route, and its endpoint, "
        "considered together. A route whose source or population is mismatched "
        "to the request is a reason to lower relevance.\n\n"
        "The technical definition and its plain-language explanation refer to "
        "one and the same signal; the explanation only restates the source, "
        "route, and endpoint in readable words. For a relationship-only "
        "candidate, the evidence is the association or connected rows along the "
        "route. Do not credit attributes of a reached table unless that "
        "attribute is explicitly the endpoint. The rest of the schema is context, "
        "not extra candidate content. Never supply an omitted join, field value, "
        "aggregation, or comparison.\n\n"
        "An attribute on the requested records may be relevant with no joins. "
        "Indirect routes are not disqualified by their length: hop count alone "
        "is not evidence for or against relevance. Do not claim methods, scores, "
        "rankings, or path provenance beyond what the candidate states."
    ),
}

_SHARED_RUBRIC_AND_OUTPUT = (
    "Relevance grades: 0 = no plausible useful evidence for the intent; "
    "1 = only weak or indirect evidence; 2 = meaningfully useful evidence, "
    "but incomplete, noisy, general, or not central; 3 = central and direct "
    "evidence for the intent. Apply these grades considering the intent and "
    "complete path. Use unable_to_judge only when the schema and intent "
    "genuinely do not allow a relevance decision. Return a short, "
    "schema-grounded reason. Do not include private reasoning or extra fields. "
    "Return only the required top-level JSON object."
)

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
    "muse-glimmer-30b": "meta-models/Muse-Glimmer-30B",
    "gemma-4-31b": "google/gemma-4-31B-it",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gpt-6-luna": "openai.gpt-6-luna",
}

# These names refer to model-serving backends rather than model capability.
# Keeping this explicit prevents an OCI model from accidentally going through
# vLLM (or causing a Hugging Face download) just because it is in
# SUPPORTED_MODELS.
OCI_WAYFLOW_MODELS = frozenset({"gpt-6-luna"})
LOCAL_VLLM_MODELS = frozenset(set(SUPPORTED_MODELS) - OCI_WAYFLOW_MODELS)

MODEL_PROFILES = {
    "mistral": JudgeModelProfile(SUPPORTED_MODELS["mistral"], 1, 0.90),
    # The official BF16 checkpoints are approximately 60--63 GB.  They fit on
    # one 80 GB H100 for this short JSON-judging prompt, but only with a
    # deliberately small KV-cache budget.  A 2k context cap and four active
    # sequences leave headroom for model/runtime allocations; callers can
    # raise either after a successful throughput pilot.
    "muse-glimmer-30b": JudgeModelProfile(
        SUPPORTED_MODELS["muse-glimmer-30b"], 1, 0.92,
        default_max_model_len=2048, default_max_num_seqs=4,
    ),
    "gemma-4-31b": JudgeModelProfile(
        SUPPORTED_MODELS["gemma-4-31b"], 1, 0.92,
        default_max_model_len=2048, default_max_num_seqs=4,
    ),
    # The official MXFP4 checkpoint is 60.8 GiB and is designed to fit on one
    # 80 GB H100. Keep context and concurrency conservative to leave room for
    # runtime allocations and the KV cache during local judging.
    "gpt-oss-120b": JudgeModelProfile(
        SUPPORTED_MODELS["gpt-oss-120b"], 1, 0.90,
        default_max_model_len=2048, default_max_num_seqs=8,
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
    prompt_variant: str = "standard"


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
    *,
    prompt_variant: str = "standard",
) -> list[dict[str, str]]:
    """Build one method-blind prompt from full, equivalent instruction variants."""
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"unsupported prompt variant {prompt_variant!r}; choose from {PROMPT_VARIANTS}")
    system = f"{_SYSTEM_PROMPT_VARIANTS[prompt_variant]}\n\n{_SHARED_RUBRIC_AND_OUTPUT}"
    user = (
        f"Schema:\n{schema_description}\n\n"
        f"User intent:\n{condition}\n\n"
        f"Candidate signal — technical definition:\n{candidate.describe(graph)}\n\n"
        f"Plain-language explanation of the same signal:\n{candidate.explain_for_human(graph)}\n\n"
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
        return {}
    if model_family == "gpt-oss-120b":
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError(
                "GPT-OSS supports --reasoning-effort none, low, medium, or high"
            )
        # GPT-OSS's Harmony chat template represents this setting in its
        # system message. The default is none: concise structured labels do
        # not need to spend their fixed generation budget on a long CoT.
        return {"reasoning_effort": reasoning_effort}
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
        "prompt_variant": config.prompt_variant,
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
    if model_family not in LOCAL_VLLM_MODELS:
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


def create_oci_wayflow_judge(
    *,
    model_name: str,
    compartment_id: str,
    config_path: str,
    profile: str,
    region: str,
    max_tokens: int,
    seed: int,
    reasoning_effort: str,
) -> Any:
    """Create a WayFlow OCI GenAI judge using the caller's OCI config file."""
    try:
        from wayflowcore.models import LlmGenerationConfig, OCIGenAIModel
        from wayflowcore.models.ociclientconfig import OCIClientConfigWithApiKey
    except ImportError as error:
        raise RuntimeError(
            "GPT-6 Luna judging requires wayflowcore and the OCI SDK in the "
            "SimBench environment."
        ) from error

    extra_args: dict[str, Any] = {"seed": seed}
    if reasoning_effort != "none":
        extra_args["reasoning_effort"] = reasoning_effort.upper()
    endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"
    client_config = OCIClientConfigWithApiKey(
        service_endpoint=endpoint,
        auth_profile=profile,
        _auth_file_location=config_path,
    )
    return OCIGenAIModel(
        model_id=model_name,
        compartment_id=compartment_id,
        client_config=client_config,
        # GPT-6 Luna's OCI endpoint only accepts its provider default
        # temperature. Do not send temperature=0, which OCI rejects.
        generation_config=LlmGenerationConfig(max_tokens=max_tokens, extra_args=extra_args),
    )


def make_wayflow_judge_prompt(messages: Sequence[Mapping[str, str]]) -> Any:
    """Translate judge messages into a WayFlow prompt with strict JSON output."""
    try:
        from wayflowcore.messagelist import Message
        from wayflowcore.models import Prompt
        from wayflowcore.property import (
            IntegerProperty,
            NullProperty,
            ObjectProperty,
            StringProperty,
            UnionProperty,
        )
    except ImportError as error:
        raise RuntimeError("GPT-6 Luna judging requires wayflowcore") from error

    response_format = ObjectProperty(
        name="signal_judgment",
        properties={
            "status": StringProperty(
                description="Either judged or unable_to_judge."
            ),
            "grade": UnionProperty(
                any_of=[IntegerProperty(), NullProperty()],
                description="A relevance grade from 0 to 3, or null if unable_to_judge.",
            ),
            "reason": StringProperty(description="A short schema-grounded reason."),
        },
        additional_properties=False,
    )
    return Prompt(
        messages=[
            Message(role=message["role"], content=message["content"])
            for message in messages
        ],
        response_format=response_format,
    )


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
