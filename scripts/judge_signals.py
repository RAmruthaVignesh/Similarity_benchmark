#!/usr/bin/env python3
"""Create resumable, blinded LLM relevance judgments using offline vLLM.

The script judges every condition/candidate pair in the fixed schema catalog.
It deliberately never receives discovery-method names, ranks, scores, or graph
search traces, so its qrels can be used to score all methods fairly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.annotate_signals import load_conditions, read_jsonl  # noqa: E402
from src.datasets.local_relbench_dataset import LocalRelBenchDataset  # noqa: E402
from src.evaluation.llm_judge import (  # noqa: E402
    MODEL_PROFILES,
    REASONING_EFFORTS,
    SUPPORTED_MODELS,
    JudgeConfig,
    JUDGE_OUTPUT_SCHEMA,
    build_judge_prompt,
    cache_key,
    chat_template_kwargs,
    create_vllm,
    describe_schema,
    iter_chunks,
    make_sampling_params,
    parse_judge_response,
    render_chat_prompt,
    schema_version,
)
from src.schema.schema_graph import SchemaGraph  # noqa: E402
from src.signals.candidates import enumerate_signal_candidates  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="resumable JSONL judgments (required unless --benchmark-only)")
    parser.add_argument("--model", choices=tuple(SUPPORTED_MODELS), required=True)
    parser.add_argument("--model-name", help="override the model ID for the selected family")
    parser.add_argument("--model-revision", default="main", help="Hugging Face commit/tag; pin a commit for final qrels")
    parser.add_argument("--hop-limit", type=int, default=2)
    parser.add_argument("--source-table", action="append", dest="source_tables")
    parser.add_argument("--include-structural-columns", action="store_true")
    parser.add_argument("--no-relationships", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16, help="pairs submitted to vLLM together")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, help="default depends on --model")
    parser.add_argument(
        "--max-num-seqs", type=int,
        help="maximum concurrent sequences held by vLLM; default depends on --model",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, help="default depends on --model")
    parser.add_argument("--tensor-parallel-size", type=int, help="GPUs serving one model replica; default depends on --model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reasoning-effort", choices=REASONING_EFFORTS, default="none",
        help="Qwen 3.8: none/low/medium/high(xhigh); Qwen 3.6: none or medium (thinking enabled); other presets: none",
    )
    parser.add_argument("--max-attempts", type=int, default=2, help="attempts for malformed model JSON")
    parser.add_argument("--overwrite", action="store_true", help="regenerate version-matched completed rows")
    parser.add_argument("--benchmark-only", action="store_true", help="measure model load and annotation throughput without writing judgments")
    parser.add_argument("--benchmark-pairs", type=int, default=32, help="condition/signal pairs to measure in --benchmark-only mode")
    parser.add_argument("--warmup-pairs", type=int, default=4, help="unmeasured pairs run after loading before throughput measurement")
    parser.add_argument(
        "--debug-response-samples", type=int, default=0,
        help="include this many measured raw responses and parse errors in benchmark metrics",
    )
    parser.add_argument("--metrics-output", type=Path, help="optional JSON file for load and throughput measurements")
    args = parser.parse_args()
    _validate_args(parser, args)

    conditions = load_conditions(args.conditions)
    dataset = LocalRelBenchDataset(args.dataset_root)
    graph = SchemaGraph.from_manifest(dataset.schema)
    candidates = enumerate_signal_candidates(
        dataset,
        graph,
        source_tables=args.source_tables,
        max_hops=args.hop_limit,
        include_relationships=not args.no_relationships,
        include_structural_columns=args.include_structural_columns,
    )
    if not candidates:
        parser.error("candidate enumeration produced no signals")
    config = JudgeConfig(
        model_family=args.model,
        model_name=args.model_name or SUPPORTED_MODELS[args.model],
        model_revision=args.model_revision,
        seed=args.seed,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    candidate_config = {
        "hop_limit": args.hop_limit,
        "source_tables": list(args.source_tables or dataset.get_tables()),
        "include_relationships": not args.no_relationships,
        "include_structural_columns": args.include_structural_columns,
    }
    version = schema_version(candidates, graph, candidate_config)
    rows_by_key = {} if args.benchmark_only else _load_existing(args.output)
    tasks = _make_tasks(conditions, candidates, version, config, rows_by_key, args.overwrite)
    print(
        f"Conditions: {len(conditions)} | candidates: {len(candidates)} | "
        f"stored rows: {len(rows_by_key)} | to judge: {len(tasks)}",
        flush=True,
    )
    if not tasks:
        return 0
    if args.model_revision == "main":
        print("WARNING: --model-revision=main is mutable; pin a commit before freezing qrels.", flush=True)

    profile = MODEL_PROFILES[args.model]
    max_model_len = args.max_model_len or profile.default_max_model_len
    gpu_memory_utilization = args.gpu_memory_utilization or profile.default_gpu_memory_utilization
    tensor_parallel_size = args.tensor_parallel_size or profile.default_tensor_parallel_size
    max_num_seqs = args.max_num_seqs or profile.default_max_num_seqs
    template_options = chat_template_kwargs(args.model, config.reasoning_effort)
    print(
        f"vLLM profile: model={config.model_name} | tensor parallel={tensor_parallel_size} | "
        f"max model len={max_model_len} | GPU memory utilization={gpu_memory_utilization:.2f} | "
        f"max sequences={max_num_seqs or 'vLLM default'} | "
        f"reasoning effort={config.reasoning_effort}",
        flush=True,
    )
    load_started = time.perf_counter()
    engine = create_vllm(
        args.model,
        config.model_name,
        config.model_revision,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=True,
    )
    model_load_seconds = time.perf_counter() - load_started
    tokenizer = engine.get_tokenizer()
    schema_text = describe_schema(candidates, graph)
    sampling_params = make_sampling_params(config)
    warmup_seconds = _warmup(
        engine, tokenizer, tasks[:args.warmup_pairs], graph, schema_text,
        sampling_params, template_options,
    )
    if args.benchmark_only:
        measured_tasks = tasks[:args.benchmark_pairs]
        if not measured_tasks:
            parser.error("--benchmark-pairs selected no condition/signal pairs")
        metrics = _benchmark_tasks(
            engine, tokenizer, measured_tasks, graph, schema_text, sampling_params,
            template_options, args.batch_size, args.debug_response_samples,
        )
        metrics.update({
            "mode": "throughput_benchmark",
            "model_family": args.model,
            "model": config.model_name,
            "model_revision": config.model_revision,
            "seed": config.seed,
            "reasoning_effort": config.reasoning_effort,
            "batch_size": args.batch_size,
            "max_tokens": config.max_tokens,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "model_load_seconds": model_load_seconds,
            "warmup_pairs": min(args.warmup_pairs, len(tasks)),
            "warmup_seconds": warmup_seconds,
        })
        _write_metrics(args.metrics_output, metrics)
        return 0
    metrics = _judge_tasks(
        engine=engine,
        tokenizer=tokenizer,
        tasks=tasks,
        rows_by_key=rows_by_key,
        graph=graph,
        candidates=candidates,
        candidate_config=candidate_config,
        schema_hash=version,
        config=config,
        batch_size=args.batch_size,
        max_attempts=args.max_attempts,
        output=args.output,
        sampling_params=sampling_params,
        template_options=template_options,
    )
    counts: dict[str, int] = {}
    for row in rows_by_key.values():
        status = str(row.get("status"))
        counts[status] = counts.get(status, 0) + 1
    print(f"wrote {args.output}; status counts: {counts}")
    metrics.update({
        "mode": "annotation",
        "model_family": args.model,
        "model": config.model_name,
        "model_revision": config.model_revision,
        "seed": config.seed,
        "reasoning_effort": config.reasoning_effort,
        "batch_size": args.batch_size,
        "max_tokens": config.max_tokens,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "model_load_seconds": model_load_seconds,
        "warmup_pairs": min(args.warmup_pairs, len(tasks)),
        "warmup_seconds": warmup_seconds,
    })
    _write_metrics(args.metrics_output, metrics)
    return 0


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.hop_limit < 0:
        parser.error("--hop-limit must be non-negative")
    if args.output is None and not args.benchmark_only:
        parser.error("--output is required unless --benchmark-only is set")
    if args.batch_size <= 0 or args.max_tokens <= 0 or args.max_attempts <= 0 or args.benchmark_pairs <= 0 or args.warmup_pairs < 0 or args.debug_response_samples < 0:
        parser.error("--batch-size, --max-tokens, --max-attempts, and --benchmark-pairs must be positive; --warmup-pairs must be non-negative")
    if args.tensor_parallel_size is not None and args.tensor_parallel_size <= 0:
        parser.error("--tensor-parallel-size must be positive")
    if args.max_model_len is not None and args.max_model_len <= 0:
        parser.error("--max-model-len must be positive")
    if args.max_num_seqs is not None and args.max_num_seqs <= 0:
        parser.error("--max-num-seqs must be positive")
    if args.gpu_memory_utilization is not None and not 0.05 <= args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be between 0.05 and 1.0")


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    rows_by_key: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = row.get("cache_key")
        if not isinstance(key, str):
            raise ValueError(f"{path}: existing row is missing cache_key")
        rows_by_key[key] = row
    return rows_by_key


def _make_tasks(
    conditions: list[dict[str, str]],
    candidates: tuple[Any, ...],
    schema_hash: str,
    config: JudgeConfig,
    rows_by_key: dict[str, dict[str, Any]],
    overwrite: bool,
) -> list[tuple[dict[str, str], Any, str]]:
    tasks = []
    for condition in conditions:
        for candidate in candidates:
            key = cache_key(
                condition_id=condition["id"], condition=condition["condition"],
                candidate=candidate, schema_version=schema_hash, config=config,
            )
            previous = rows_by_key.get(key)
            if not overwrite and previous and previous.get("status") in {"judged", "unable_to_judge"}:
                continue
            tasks.append((condition, candidate, key))
    return tasks


def _judge_tasks(
    *,
    engine: Any,
    tokenizer: Any,
    tasks: list[tuple[dict[str, str], Any, str]],
    rows_by_key: dict[str, dict[str, Any]],
    graph: SchemaGraph,
    candidates: tuple[Any, ...],
    candidate_config: dict[str, Any],
    schema_hash: str,
    config: JudgeConfig,
    batch_size: int,
    max_attempts: int,
    output: Path,
    sampling_params: Any,
    template_options: dict[str, Any],
) -> dict[str, Any]:
    schema_text = describe_schema(candidates, graph)
    pending = tasks
    generated_tokens = 0
    prompt_tokens = 0
    finish_reason_counts: dict[str, int] = {}
    token_limit_hit_count = 0
    token_limit_hit_invalid_count = 0
    measured_started = time.perf_counter()
    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        retry: list[tuple[dict[str, str], Any, str]] = []
        for batch_index, batch in enumerate(iter_chunks(pending, batch_size), 1):
            prompts = [
                render_chat_prompt(
                    tokenizer,
                    build_judge_prompt(condition["condition"], candidate, graph, schema_text),
                    template_kwargs=template_options,
                )
                for condition, candidate, _ in batch
            ]
            outputs = engine.generate(prompts, sampling_params)
            if len(outputs) != len(batch):
                raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(batch)} prompts")
            for (condition, candidate, key), prompt, response in zip(batch, prompts, outputs):
                raw = response.outputs[0].text if response.outputs else ""
                finish_reason = _response_finish_reason(response)
                finish_reason_counts[finish_reason] = finish_reason_counts.get(finish_reason, 0) + 1
                hit_token_limit = _hit_token_limit(finish_reason)
                token_limit_hit_count += int(hit_token_limit)
                generated_tokens += _generated_token_count(response)
                prompt_tokens += _prompt_token_count(tokenizer, response, prompt)
                parsed = parse_judge_response(raw)
                token_limit_hit_invalid_count += int(
                    hit_token_limit and parsed["status"] == "invalid_response"
                )
                rows_by_key[key] = _judgment_row(
                    key=key,
                    condition=condition,
                    candidate=candidate,
                    graph=graph,
                    candidate_config=candidate_config,
                    schema_hash=schema_hash,
                    config=config,
                    parsed=parsed,
                    raw=raw,
                    finish_reason=finish_reason,
                    attempt=attempt,
                )
                if parsed["status"] == "invalid_response" and attempt < max_attempts:
                    retry.append((condition, candidate, key))
            print(
                f"attempt {attempt}, batch {batch_index}: "
                f"processed {min(batch_index * batch_size, len(pending))}/{len(pending)}",
                flush=True,
            )
        write_jsonl_atomic(output, rows_by_key.values())
        pending = retry
    metrics = _throughput_metrics(
        elapsed_seconds=time.perf_counter() - measured_started,
        pairs=len(tasks), prompt_tokens=prompt_tokens, generated_tokens=generated_tokens,
    )
    metrics["finish_reason_counts"] = finish_reason_counts
    metrics["token_limit_hit_count"] = token_limit_hit_count
    metrics["token_limit_hit_invalid_attempt_count"] = token_limit_hit_invalid_count
    metrics["response_attempts"] = sum(finish_reason_counts.values())
    final_status_counts: dict[str, int] = {}
    for row in rows_by_key.values():
        status = str(row.get("status", "unknown"))
        final_status_counts[status] = final_status_counts.get(status, 0) + 1
    metrics["final_status_counts"] = final_status_counts
    metrics["final_token_limit_hit_invalid_response_count"] = sum(
        row.get("status") == "invalid_response" and row.get("hit_token_limit") is True
        for row in rows_by_key.values()
    )
    return metrics


def _warmup(
    engine: Any,
    tokenizer: Any,
    tasks: list[tuple[dict[str, str], Any, str]],
    graph: SchemaGraph,
    schema_text: str,
    sampling_params: Any,
    template_options: dict[str, Any],
) -> float:
    """Compile/load hot paths without contaminating the reported throughput."""
    if not tasks:
        return 0.0
    prompts = _render_prompts(tokenizer, tasks, graph, schema_text, template_options)
    started = time.perf_counter()
    outputs = engine.generate(prompts, sampling_params)
    if len(outputs) != len(tasks):
        raise RuntimeError(f"vLLM returned {len(outputs)} warmup outputs for {len(tasks)} prompts")
    return time.perf_counter() - started


def _benchmark_tasks(
    engine: Any,
    tokenizer: Any,
    tasks: list[tuple[dict[str, str], Any, str]],
    graph: SchemaGraph,
    schema_text: str,
    sampling_params: Any,
    template_options: dict[str, Any],
    batch_size: int,
    debug_response_samples: int = 0,
) -> dict[str, Any]:
    """Measure true prompt-to-judgment throughput without persisting qrels."""
    started = time.perf_counter()
    prompt_tokens = 0
    generated_tokens = 0
    status_counts: dict[str, int] = {}
    finish_reason_counts: dict[str, int] = {}
    token_limit_hit_count = 0
    token_limit_hit_invalid_count = 0
    response_samples: list[dict[str, Any]] = []
    for batch in iter_chunks(tasks, batch_size):
        prompts = _render_prompts(tokenizer, batch, graph, schema_text, template_options)
        outputs = engine.generate(prompts, sampling_params)
        if len(outputs) != len(batch):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(batch)} benchmark prompts")
        for (condition, candidate, _), prompt, response in zip(batch, prompts, outputs):
            prompt_tokens += _prompt_token_count(tokenizer, response, prompt)
            generated_tokens += _generated_token_count(response)
            raw = response.outputs[0].text if response.outputs else ""
            finish_reason = _response_finish_reason(response)
            finish_reason_counts[finish_reason] = finish_reason_counts.get(finish_reason, 0) + 1
            hit_token_limit = _hit_token_limit(finish_reason)
            token_limit_hit_count += int(hit_token_limit)
            parsed = parse_judge_response(raw)
            status = str(parsed["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
            token_limit_hit_invalid_count += int(hit_token_limit and status == "invalid_response")
            if len(response_samples) < debug_response_samples:
                response_samples.append({
                    "condition_id": condition["id"],
                    "candidate_id": candidate.id,
                    "status": status,
                    "parse_error": parsed.get("parse_error"),
                    "finish_reason": finish_reason,
                    "raw_response": raw,
                })
    metrics = _throughput_metrics(
        elapsed_seconds=time.perf_counter() - started,
        pairs=len(tasks),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
    )
    metrics["status_counts"] = status_counts
    metrics["finish_reason_counts"] = finish_reason_counts
    metrics["token_limit_hit_count"] = token_limit_hit_count
    metrics["token_limit_hit_invalid_count"] = token_limit_hit_invalid_count
    if debug_response_samples:
        metrics["debug_response_samples"] = response_samples
    return metrics


def _render_prompts(
    tokenizer: Any,
    tasks: Any,
    graph: SchemaGraph,
    schema_text: str,
    template_options: dict[str, Any],
) -> list[str]:
    return [
        render_chat_prompt(
            tokenizer,
            build_judge_prompt(condition["condition"], candidate, graph, schema_text),
            template_kwargs=template_options,
        )
        for condition, candidate, _ in tasks
    ]


def _generated_token_count(response: Any) -> int:
    return sum(len(getattr(output, "token_ids", []) or []) for output in response.outputs)


def _response_finish_reason(response: Any) -> str:
    """Return vLLM's termination reason for the single sampled completion."""
    if not response.outputs:
        return "no_output"
    reason = getattr(response.outputs[0], "finish_reason", None)
    return str(reason) if reason is not None else "unknown"


def _hit_token_limit(finish_reason: str) -> bool:
    """Recognize vLLM finish reasons indicating max_tokens ended generation."""
    return finish_reason in {"length", "max_tokens"}


def _prompt_token_count(tokenizer: Any, response: Any, prompt: str) -> int:
    token_ids = getattr(response, "prompt_token_ids", None)
    if token_ids is not None:
        return len(token_ids)
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def _throughput_metrics(
    *, elapsed_seconds: float, pairs: int, prompt_tokens: int, generated_tokens: int,
) -> dict[str, Any]:
    total_tokens = prompt_tokens + generated_tokens
    return {
        "pairs": pairs,
        "elapsed_seconds": elapsed_seconds,
        "pairs_per_second": pairs / elapsed_seconds if elapsed_seconds else 0.0,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": total_tokens,
        "generated_tokens_per_second": generated_tokens / elapsed_seconds if elapsed_seconds else 0.0,
        "total_tokens_per_second": total_tokens / elapsed_seconds if elapsed_seconds else 0.0,
        "mean_generated_tokens_per_pair": generated_tokens / pairs if pairs else 0.0,
    }


def _write_metrics(path: Path | None, metrics: dict[str, Any]) -> None:
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered, flush=True)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _judgment_row(
    *, key: str, condition: dict[str, str], candidate: Any, graph: SchemaGraph,
    candidate_config: dict[str, Any], schema_hash: str, config: JudgeConfig,
    parsed: dict[str, Any], raw: str, finish_reason: str, attempt: int,
) -> dict[str, Any]:
    return {
        "cache_key": key,
        "condition_id": condition["id"],
        "condition": condition["condition"],
        "candidate_id": candidate.id,
        "candidate": candidate.to_dict(graph),
        "schema_version": schema_hash,
        "candidate_config": candidate_config,
        "judge_model_family": config.model_family,
        "judge_model": config.model_name,
        "judge_model_revision": config.model_revision,
        "prompt_version": config.prompt_version,
        "rubric_version": config.rubric_version,
        "output_schema": JUDGE_OUTPUT_SCHEMA,
        "seed": config.seed,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "reasoning_effort": config.reasoning_effort,
        "status": parsed["status"],
        "grade": parsed["grade"],
        "reason": parsed["reason"],
        "raw_response": raw,
        "finish_reason": finish_reason,
        "hit_token_limit": _hit_token_limit(finish_reason),
        "parse_error": parsed.get("parse_error"),
        "attempt": attempt,
        "judged_at": datetime.now(timezone.utc).isoformat(),
    }


def write_jsonl_atomic(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in sorted(rows, key=lambda item: (item.get("condition_id", ""), item.get("candidate_id", ""), item.get("cache_key", ""))):
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


if __name__ == "__main__":
    raise SystemExit(main())
