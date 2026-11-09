"""Shared model profiles and LLM backends for signal discovery and similarity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.schema.schema_graph import SchemaGraph
from src.signals.candidates import SignalCandidate


@dataclass(frozen=True)
class ModelProfile:
    """Local model preset, including conservative vLLM defaults."""

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
    "gpt-6-sol": "openai.gpt-6-sol",
    "gpt-5.6-terra": "openai.gpt-5.6-terra",
    "gpt-6-astra": "openai.gpt-6-astra",
}


OCI_WAYFLOW_MODELS = frozenset({"gpt-6-luna", "gpt-6-sol", "gpt-5.6-terra", "gpt-6-astra"})


LOCAL_VLLM_MODELS = frozenset(set(SUPPORTED_MODELS) - OCI_WAYFLOW_MODELS)


MODEL_PROFILES = {
    "mistral": ModelProfile(SUPPORTED_MODELS["mistral"], 1, 0.90),
    # The official BF16 checkpoints are approximately 60--63 GB.  They fit on
    # one 80 GB H100 for short JSON generation, but only with a
    # deliberately small KV-cache budget.  A 2k context cap and four active
    # sequences leave headroom for model/runtime allocations; callers can
    # raise either after a successful throughput pilot.
    "muse-glimmer-30b": ModelProfile(
        SUPPORTED_MODELS["muse-glimmer-30b"], 1, 0.92,
        default_max_model_len=2048, default_max_num_seqs=4,
    ),
    "gemma-4-31b": ModelProfile(
        SUPPORTED_MODELS["gemma-4-31b"], 1, 0.92,
        default_max_model_len=2048, default_max_num_seqs=4,
    ),
    # The official MXFP4 checkpoint is 60.8 GiB and is designed to fit on one
    # 80 GB H100. Keep context and concurrency conservative to leave room for
    # runtime allocations and the KV cache during local generation.
    "gpt-oss-120b": ModelProfile(
        SUPPORTED_MODELS["gpt-oss-120b"], 1, 0.90,
        default_max_model_len=2048, default_max_num_seqs=8,
    ),
}


REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh")


OCI_REASONING_EFFORTS = {
    "gpt-6-luna": frozenset({"none", "low", "medium", "high"}),
    "gpt-6-sol": frozenset({"none", "low", "medium", "high"}),
    "gpt-5.6-terra": frozenset({"none", "low", "medium", "high"}),
    "gpt-6-astra": frozenset({"low", "medium", "high"}),
}


def validate_reasoning_effort(model_family: str, reasoning_effort: str) -> None:
    """Reject effort levels that the selected runner cannot send."""
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(
            f"unsupported reasoning effort {reasoning_effort!r}; "
            f"choose one of {', '.join(REASONING_EFFORTS)}"
        )
    allowed = OCI_REASONING_EFFORTS.get(model_family)
    if allowed is not None and reasoning_effort not in allowed:
        raise ValueError(
            f"{model_family} on OCI supports --reasoning-effort "
            f"{', '.join(sorted(allowed))}"
        )


def describe_schema(
    candidates: Sequence[SignalCandidate], graph: SchemaGraph
) -> str:
    """Return a concise schema description for discovery prompts."""
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
        # system message. The default is none: concise structured outputs do
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


def create_vllm(model_family: str, model_name: str, model_revision: str, **kwargs: Any) -> Any:
    """Create an offline vLLM engine with the vendor-recommended Mistral mode."""
    if model_family not in LOCAL_VLLM_MODELS:
        raise ValueError(f"unsupported local model family: {model_family}")
    try:
        from vllm import LLM
    except ImportError as error:
        raise RuntimeError(
            "Local LLM generation requires vLLM. Install a CUDA/H100-compatible vLLM build "
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


def create_oci_wayflow_model(
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
    """Create a WayFlow OCI GenAI model using the caller's OCI config file."""
    try:
        from wayflowcore.models import LlmGenerationConfig, OCIGenAIModel
        from wayflowcore.models.ociclientconfig import OCIClientConfigWithApiKey
    except ImportError as error:
        raise RuntimeError(
            "OCI-backed generation requires wayflowcore and the OCI SDK in the "
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
