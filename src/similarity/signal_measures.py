"""Materialize schema signals and compare entity pairs with suitable measures.

This is the first deterministic implementation of proposal section 3.2.  It
does not infer relevance and it does not use labels: a caller supplies the
signals selected by section 3.1, and this module turns each one into an
entity-level value, collection, distribution, or neighbourhood.  The chosen
measure and every component score remain visible in the result, so later
learned or LLM-assisted composition can reuse exactly the same evidence.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import math
import os
import re
import sqlite3
import tempfile
from contextlib import contextmanager
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa

from src.datasets.local_relbench_dataset import LocalRelBenchDataset
from src.schema.schema_graph import Edge, SchemaGraph
from src.signals.candidates import SignalCandidate

__all__ = [
    "MaterializedSignal",
    "PairSimilarity",
    "SignalMeasureExecutor",
    "MEASURE_VARIANTS",
    "combine_signal_scores",
]

_TEXT_COLUMN_HINTS = ("description", "text", "summary", "title")
DEFAULT_E5_MODEL = "intfloat/e5-small-v2"
DEFAULT_MODEL_CACHE_ROOT = Path(
    os.environ.get("SIMBENCH_MODEL_CACHE_ROOT", "/storage/lcostes/model-cache/huggingface/hub")
)
DEFAULT_TEXT_EMBEDDING_CACHE = Path(
    os.environ.get(
        "SIMBENCH_TEXT_EMBEDDING_CACHE",
        "/storage/lcostes/simbench-cache/text-embeddings.sqlite3",
    )
)
_E5_INPUT_PREFIX = "query: "
_E5_TEXT_TOKEN_LIMIT = 500
MEASURE_VARIANTS = (
    "auto",
    "jensen-shannon",
    "hellinger",
    "wl-subtree-kernel",
    "diffusion-distance",
)
_DISTRIBUTION_BIN_COUNT = 10
_DIFFUSION_RESTART_PROBABILITY = 0.2
_MAX_DIFFUSION_PROJECTED_EDGES = 8_000_000


@dataclass(frozen=True)
class MaterializedSignal:
    """One selected schema signal represented for every source-table entity."""

    candidate: SignalCandidate
    entity_table: str
    entity_id_kind: str
    entity_ids: tuple[object, ...]
    values_by_entity: Mapping[object, tuple[object, ...]]
    value_kind: str
    measure: str
    value_range: tuple[float, float] | None = None
    reference_values: tuple[float, ...] | None = None
    value_weights: Mapping[object, float] | None = None
    embeddings_by_entity: Mapping[object, np.ndarray] | None = None
    max_degree: int | None = None
    graph_features_by_entity: Mapping[object, Mapping[tuple[int, int, int], float]] | None = None
    diffusion_profiles_by_entity: Mapping[object, Mapping[int, float]] | None = None

    def values_for(self, entity_id: object) -> tuple[object, ...]:
        """Return the values reachable from one source entity."""
        return self.values_by_entity[entity_id]


@dataclass(frozen=True)
class PairSimilarity:
    """A transparent, combined score for one query/candidate entity pair."""

    query_entity_id: object
    candidate_entity_id: object
    similarity_score: float | None
    components: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "query_entity_id": self.query_entity_id,
            "candidate_entity_id": self.candidate_entity_id,
            "similarity_score": self.similarity_score,
            "components": list(self.components),
        }


class _VllmE5TextEmbedder:
    """Local, batched contextual E5 encoder served by vLLM's pooling runner.

    Each value in a path-derived collection is embedded independently and the
    resulting vectors are averaged for its source entity.  This is important:
    concatenating (for example) one hundred review texts into one prompt would
    silently discard almost all of them at an encoder's context limit.
    """

    def __init__(
        self,
        model_id: str,
        model_cache_root: Path | None,
        embedding_cache_path: Path,
        *,
        batch_size: int = 64,
    ) -> None:
        self.model_id = model_id
        self.model_cache_root = model_cache_root
        self.embedding_cache_path = embedding_cache_path
        self.batch_size = batch_size
        self._engine: Any = None
        self._tokenizer: Any = None
        self._cache: _TextEmbeddingCache | None = None

    def encode_collections(
        self, values_by_entity: Mapping[object, tuple[object, ...]]
    ) -> Mapping[object, np.ndarray]:
        flattened: list[tuple[object, str]] = []
        for entity_id, values in values_by_entity.items():
            for value in values:
                text = " ".join(str(value).split())
                if text:
                    flattened.append((entity_id, text))
        if not flattened:
            return {}

        vectors_by_entity: dict[object, list[np.ndarray]] = defaultdict(list)
        for offset in range(0, len(flattened), self.batch_size):
            batch = flattened[offset : offset + self.batch_size]
            vectors, _, _ = self._vectors_for_texts([text for _, text in batch])
            for (entity_id, _), embedding in zip(batch, vectors, strict=True):
                vectors_by_entity[entity_id].append(embedding)

        return {
            entity_id: np.mean(vectors, axis=0, dtype=np.float32)
            for entity_id, vectors in vectors_by_entity.items()
        }

    @property
    def cache_path(self) -> Path:
        return self.embedding_cache_path

    def precompute_texts(self, texts: Iterable[object]) -> dict[str, int | str]:
        """Embed and persist all nonblank text values, in bounded batches."""
        values_seen = 0
        cache_hits = 0
        embeddings_created = 0
        batch: list[str] = []
        chunk_size = self.batch_size * 8
        for value in texts:
            text = " ".join(str(value).split())
            if not text:
                continue
            values_seen += 1
            batch.append(text)
            if len(batch) >= chunk_size:
                _, hits, created = self._vectors_for_texts(batch)
                cache_hits += hits
                embeddings_created += created
                batch.clear()
        if batch:
            _, hits, created = self._vectors_for_texts(batch)
            cache_hits += hits
            embeddings_created += created
        cache = self._get_cache()
        return {
            "values_seen": values_seen,
            "cache_hits": cache_hits,
            "embeddings_created": embeddings_created,
            "cache_entries": cache.count(),
            "cache_path": str(self.embedding_cache_path),
        }

    def _vectors_for_texts(
        self, texts: Sequence[str]
    ) -> tuple[list[np.ndarray], int, int]:
        if not texts:
            return [], 0, 0
        prepared = self._prepare_batch(texts)
        prompts = [f"{_E5_INPUT_PREFIX}{text}" for text in prepared]
        prompt_by_key: dict[str, str] = {}
        keys = []
        for prompt in prompts:
            key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            keys.append(key)
            prompt_by_key.setdefault(key, prompt)

        cache = self._get_cache()
        vectors_by_key = cache.get_many(prompt_by_key)
        cache_hits = len(vectors_by_key)
        missing = [(key, prompt) for key, prompt in prompt_by_key.items() if key not in vectors_by_key]
        for offset in range(0, len(missing), self.batch_size):
            batch = missing[offset : offset + self.batch_size]
            outputs = self._load().embed([prompt for _, prompt in batch], use_tqdm=False)
            if len(outputs) != len(batch):  # pragma: no cover - backend contract guard
                raise RuntimeError("vLLM returned an unexpected embedding count")
            stored = {}
            for (key, _), output in zip(batch, outputs, strict=True):
                embedding = getattr(getattr(output, "outputs", None), "embedding", None)
                if embedding is None:
                    raise RuntimeError("vLLM pooling response did not contain an embedding")
                vector = np.asarray(embedding, dtype=np.float32)
                vectors_by_key[key] = vector
                stored[key] = vector
            cache.put_many(stored)
        return [vectors_by_key[key] for key in keys], cache_hits, len(missing)

    def _prepare_batch(self, texts: Sequence[str]) -> list[str]:
        """Safely truncate a batch to E5's 512-token input budget."""
        tokenizer = self._load_tokenizer()
        # Keep 500 model tokens for the text itself.  The E5 `query:` prefix
        # plus the special tokens added by vLLM remain below the 512 limit.
        token_ids = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=_E5_TEXT_TOKEN_LIMIT,
        )["input_ids"]
        return tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        snapshot = _local_huggingface_snapshot(self.model_id, self.model_cache_root)
        if snapshot is None:
            location = str(self.model_cache_root) if self.model_cache_root else "the configured Hugging Face cache"
            raise ValueError(f"local E5 tokenizer is absent from {location}: {self.model_id!r}")
        try:
            from transformers import AutoTokenizer
        except ImportError as error:  # pragma: no cover - optional runtime
            raise ValueError(f"text backend 'e5-vllm' requires transformers: {error}") from error
        self._tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        return self._tokenizer

    def _load(self) -> Any:
        if self._engine is not None:
            return self._engine
        snapshot = _local_huggingface_snapshot(self.model_id, self.model_cache_root)
        if snapshot is None:
            location = str(self.model_cache_root) if self.model_cache_root else "the configured Hugging Face cache"
            raise ValueError(
                f"text backend 'e5-vllm' needs the local model {self.model_id!r} in {location}; "
                "download it into the shared cache"
            )
        try:
            temporary_root = Path(os.environ.get("SIMBENCH_TMPDIR", "/storage/lcostes/tmp"))
            temporary_root.mkdir(parents=True, exist_ok=True)
            # vLLM computes and caches tempfile.gettempdir() during import, so
            # configure both the environment and Python's cached choice first.
            os.environ["TMPDIR"] = str(temporary_root)
            os.environ["TEMP"] = str(temporary_root)
            os.environ["TMP"] = str(temporary_root)
            tempfile.tempdir = str(temporary_root)
            from vllm import LLM
        except ImportError as error:  # pragma: no cover - optional runtime
            raise ValueError(f"text backend 'e5-vllm' requires vLLM: {error}") from error
        # Giving vLLM the snapshot path guarantees it does not contact the Hub
        # while scoring.  E5-small has a 512-token encoder context.
        self._engine = LLM(
            model=str(snapshot),
            runner="pooling",
            convert="embed",
            dtype="float16",
            max_model_len=512,
            gpu_memory_utilization=0.20,
            # An encoder is small and invoked in finite batches.  Avoid an
            # expensive first-run torch.compile step that can dwarf inference
            # time in a local exploratory workflow.
            enforce_eager=True,
        )
        return self._engine

    def _get_cache(self) -> "_TextEmbeddingCache":
        if self._cache is None:
            snapshot = _local_huggingface_snapshot(self.model_id, self.model_cache_root)
            if snapshot is None:
                location = str(self.model_cache_root) if self.model_cache_root else "the configured Hugging Face cache"
                raise ValueError(f"local E5 model is absent from {location}: {self.model_id!r}")
            namespace_data = {
                "model_id": self.model_id,
                "snapshot": snapshot.name,
                "prefix": _E5_INPUT_PREFIX,
                "max_text_tokens": _E5_TEXT_TOKEN_LIMIT,
                "tokenizer_decode": "skip-special-cleanup-v1",
                "runner": "vllm-pooling-embed-fp16",
            }
            namespace = hashlib.sha256(
                json.dumps(namespace_data, sort_keys=True).encode("utf-8")
            ).hexdigest()
            self._cache = _TextEmbeddingCache(self.embedding_cache_path, namespace)
        return self._cache


class _TextEmbeddingCache:
    """SQLite cache for individual E5 prompt embeddings, keyed by content."""

    def __init__(self, path: Path, namespace: str) -> None:
        self.path = path
        self.namespace = namespace
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=60) as connection:
            connection.execute("PRAGMA busy_timeout = 60000")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS embeddings (
                       namespace TEXT NOT NULL,
                       text_hash TEXT NOT NULL,
                       dimension INTEGER NOT NULL,
                       vector BLOB NOT NULL,
                       PRIMARY KEY (namespace, text_hash)
                   )"""
            )

    def get_many(self, prompts_by_hash: Mapping[str, str]) -> dict[str, np.ndarray]:
        hashes = list(prompts_by_hash)
        result: dict[str, np.ndarray] = {}
        if not hashes:
            return result
        with self._connect() as connection:
            for offset in range(0, len(hashes), 500):
                batch = hashes[offset : offset + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT text_hash, dimension, vector FROM embeddings "
                    f"WHERE namespace = ? AND text_hash IN ({placeholders})",
                    [self.namespace, *batch],
                )
                for text_hash, dimension, vector in rows:
                    result[text_hash] = np.frombuffer(vector, dtype=np.float32, count=dimension).copy()
        return result

    def put_many(self, vectors_by_hash: Mapping[str, np.ndarray]) -> None:
        if not vectors_by_hash:
            return
        rows = [
            (self.namespace, key, int(vector.size), np.asarray(vector, dtype=np.float32).tobytes())
            for key, vector in vectors_by_hash.items()
        ]
        with self._connect() as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO embeddings(namespace, text_hash, dimension, vector) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM embeddings WHERE namespace = ?", (self.namespace,)
            ).fetchone()
        return int(row[0])

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=60)
        connection.execute("PRAGMA busy_timeout = 60000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _local_huggingface_snapshot(model_id: str, cache_root: Path | None) -> Path | None:
    """Find a local cached snapshot without contacting Hugging Face."""
    if cache_root is None:
        return None
    repository = cache_root / f"models--{model_id.replace('/', '--')}"
    ref = repository / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text().strip()
        snapshot = repository / "snapshots" / revision
        if snapshot.is_dir():
            return snapshot
    snapshots = sorted((repository / "snapshots").glob("*")) if (repository / "snapshots").is_dir() else []
    return snapshots[-1] if snapshots else None


class SignalMeasureExecutor:
    """Execute candidate paths over a local relational dataset.

    The executor intentionally makes no decision about *which* signals are
    relevant.  It only materializes a path selected elsewhere and applies a
    measure appropriate to its observed Arrow type and path structure.
    """

    def __init__(
        self,
        dataset: LocalRelBenchDataset,
        graph: SchemaGraph,
        *,
        text_embedding_model: str = DEFAULT_E5_MODEL,
        model_cache_root: str | Path | None = None,
        text_embedding_cache: str | Path | None = None,
        embedding_batch_size: int = 64,
    ) -> None:
        if embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive")
        self.dataset = dataset
        self.graph = graph
        self.text_embedding_model = text_embedding_model
        self.model_cache_root = (
            Path(model_cache_root) if model_cache_root is not None else DEFAULT_MODEL_CACHE_ROOT
        )
        self.text_embedding_cache = (
            Path(text_embedding_cache)
            if text_embedding_cache is not None
            else DEFAULT_TEXT_EMBEDDING_CACHE
        )
        self.embedding_batch_size = embedding_batch_size
        self._columns: dict[tuple[str, str], list[object]] = {}
        self._indices: dict[tuple[str, str], dict[object, tuple[int, ...]]] = {}
        self._e5_embedder: _VllmE5TextEmbedder | None = None

    @property
    def text_embedding_cache_path(self) -> Path:
        return self.text_embedding_cache

    def precompute_text_embeddings(self) -> dict[str, object]:
        """Precompute embeddings for every column classified as semantic text."""
        text_columns: list[str] = []

        def values() -> Iterable[object]:
            for table_name in self.dataset.get_tables():
                table = self.dataset.get_table(table_name)
                for field in table.schema:
                    raw_values = table[field.name].to_pylist()
                    collections = [
                        tuple(
                            item
                            for item in _flatten_values(value)
                            if item is not None
                        )
                        for value in raw_values
                    ]
                    kind = _attribute_kind(
                        field.name,
                        field.type,
                        collections,
                        is_declared_time=(
                            self.dataset.get_time_column(table_name) == field.name
                        ),
                    )
                    if kind != "text":
                        continue
                    text_columns.append(f"{table_name}.{field.name}")
                    for collection in collections:
                        for item in collection:
                            yield item

        embedder = self._get_e5_embedder()
        report = embedder.precompute_texts(values())
        report["text_columns"] = text_columns
        report["text_column_count"] = len(text_columns)
        report["model"] = self.text_embedding_model
        report["cache_path"] = str(self.text_embedding_cache)
        return report

    def materialize(
        self,
        candidate: SignalCandidate,
        *,
        entity_table: str | None = None,
        measure_variant: str = "auto",
    ) -> MaterializedSignal:
        """Materialize ``candidate`` for every source entity.

        An entity table normally has a declared primary key.  For a table such
        as RelAmazon's ``review`` that has no primary key, stable row positions
        in the frozen local Parquet table are used explicitly as entity IDs.
        """
        table = entity_table or candidate.source_table
        if candidate.source_table != table:
            raise ValueError(
                f"candidate {candidate.id!r} starts from {candidate.source_table!r}, "
                f"not requested entity table {table!r}"
            )
        if candidate.path[0] != table:
            raise ValueError("candidate path must start at its source table")
        if candidate.kind not in {"attribute", "relationship"}:
            raise ValueError(f"unsupported signal kind {candidate.kind!r}")
        if measure_variant not in MEASURE_VARIANTS:
            raise ValueError(
                f"unknown measure variant {measure_variant!r}; choose from {', '.join(MEASURE_VARIANTS)}"
            )

        roots, entity_ids, entity_id_kind = self._root_rows(table)
        reached = roots
        current_table = table
        for next_table in candidate.path[1:]:
            reached = self._traverse(reached, current_table, next_table)
            current_table = next_table

        if candidate.kind == "relationship":
            raw_values, value_kind = self._relationship_values(reached, current_table)
            measure = (
                "jaccard_neighborhood"
                if self.dataset.get_primary_key(current_table) is not None
                else "degree_similarity"
            )
            max_degree = max((len(values) for values in raw_values.values()), default=0)
            graph_features = None
            diffusion_profiles = None
            if measure_variant in {"jensen-shannon", "hellinger"}:
                if self.dataset.get_primary_key(current_table) is None:
                    raise ValueError(
                        f"{measure_variant} on a relationship signal needs stable endpoint IDs; "
                        f"{current_table!r} has no primary key, so its row positions cannot be "
                        "compared as shared neighbors"
                    )
                _validate_distribution_variant(value_kind, raw_values)
                measure = (
                    "jensen_shannon_similarity"
                    if measure_variant == "jensen-shannon"
                    else "hellinger_similarity"
                )
            elif measure_variant == "wl-subtree-kernel":
                graph_features = self._wl_subtree_features(candidate, roots, entity_ids)
                measure = "wl_subtree_kernel_similarity"
            elif measure_variant == "diffusion-distance":
                if self.dataset.get_primary_key(current_table) is None:
                    raise ValueError(
                        "diffusion-distance on a relationship signal needs stable endpoint IDs; "
                        f"{current_table!r} has no primary key, so its row positions cannot form "
                        "a shared-neighbor graph"
                    )
                diffusion_profiles = _diffusion_profiles(entity_ids, raw_values)
                measure = "diffusion_distance_similarity_t1"
            elif measure_variant != "auto":
                raise ValueError(
                    f"measure variant {measure_variant!r} is not supported for relationship signals"
                )
            return MaterializedSignal(
                candidate=candidate,
                entity_table=table,
                entity_id_kind=entity_id_kind,
                entity_ids=entity_ids,
                values_by_entity=raw_values,
                value_kind=value_kind,
                measure=measure,
                max_degree=max_degree,
                graph_features_by_entity=graph_features,
                diffusion_profiles_by_entity=diffusion_profiles,
            )

        if candidate.attribute_table != current_table or candidate.attribute_column is None:
            raise ValueError("attribute candidate endpoint does not match its path")
        values = {
            entity_id: tuple(
                atomic_value
                for row_index in row_indices
                if (value := self._column(current_table, candidate.attribute_column)[row_index])
                is not None
                for atomic_value in _flatten_values(value)
                if atomic_value is not None
            )
            for entity_id, row_indices in reached.items()
        }
        arrow_type = self.dataset.get_table(current_table).schema.field(
            candidate.attribute_column
        ).type
        value_kind = _attribute_kind(
            candidate.attribute_column,
            arrow_type,
            values.values(),
            is_declared_time=(self.dataset.get_time_column(current_table) == candidate.attribute_column),
        )
        if value_kind == "categorical":
            values = {
                entity_id: tuple(_normalise_category(value) for value in entity_values)
                for entity_id, entity_values in values.items()
            }
        measure, value_range, reference_values, value_weights = _measure_for_attribute(
            value_kind, values.values()
        )
        embeddings = None
        if value_kind == "text":
            if measure_variant != "auto":
                raise ValueError(
                    f"measure variant {measure_variant!r} is not supported for text signals; use 'auto'"
                )
            embeddings = self._embed_e5_text_collections(values)
            measure = "e5_embedding_cosine"
        elif measure_variant in {"jensen-shannon", "hellinger"}:
            _validate_distribution_variant(value_kind, values)
            measure = (
                "jensen_shannon_similarity"
                if measure_variant == "jensen-shannon"
                else "hellinger_similarity"
            )
        elif measure_variant != "auto":
            raise ValueError(
                f"measure variant {measure_variant!r} is only supported for categorical, boolean, "
                "numeric, temporal, or relationship signals"
            )
        return MaterializedSignal(
            candidate=candidate,
            entity_table=table,
            entity_id_kind=entity_id_kind,
            entity_ids=entity_ids,
            values_by_entity=values,
            value_kind=value_kind,
            measure=measure,
            value_range=value_range,
            reference_values=reference_values,
            value_weights=value_weights,
            embeddings_by_entity=embeddings,
        )

    def available_measure_variants(self, candidate: SignalCandidate) -> tuple[str, ...]:
        """Return explicit measure choices supported by one signal.

        This is a lightweight inspection helper for interactive controls. It
        reads the signal values needed to distinguish scalar from collection
        attributes but does not load the text embedding model or encode text.
        """
        if candidate.kind == "relationship":
            variants = ["auto", "wl-subtree-kernel"]
            endpoint_table = candidate.path[-1]
            if self.dataset.get_primary_key(endpoint_table) is not None:
                variants[1:1] = ["jensen-shannon", "hellinger"]
                variants.append("diffusion-distance")
            return tuple(variants)
        if candidate.kind != "attribute" or candidate.attribute_column is None:
            raise ValueError(f"unsupported signal kind {candidate.kind!r}")

        value_kind, values = self._attribute_signal_values(candidate)
        if value_kind in {"categorical", "boolean"}:
            return ("auto", "jensen-shannon", "hellinger")
        if value_kind in {"numeric", "temporal"} and max(
            (len(entity_values) for entity_values in values.values()), default=0
        ) > 1:
            return ("auto", "jensen-shannon", "hellinger")
        return ("auto",)

    def automatic_measure_name(self, candidate: SignalCandidate) -> str:
        """Return a human-readable name for the measure selected by ``auto``."""
        if candidate.kind == "relationship":
            endpoint_table = candidate.path[-1]
            return (
                "Neighborhood Jaccard"
                if self.dataset.get_primary_key(endpoint_table) is not None
                else "Degree similarity"
            )
        value_kind, values = self._attribute_signal_values(candidate)
        measure, _, _, _ = _measure_for_attribute(value_kind, values.values())
        return {
            "categorical_agreement": "Categorical agreement",
            "idf_weighted_jaccard": "IDF-weighted Jaccard",
            "boolean_rate_similarity": "Boolean-rate similarity",
            "percentile_distance": "Percentile distance",
            "percentile_wasserstein": "Percentile Wasserstein similarity",
            "temporal_percentile_distance": "Temporal percentile distance",
            "temporal_percentile_wasserstein": "Temporal percentile Wasserstein similarity",
            "e5_embedding_cosine": "E5 embedding cosine similarity",
        }[measure]

    def _attribute_signal_values(
        self, candidate: SignalCandidate
    ) -> tuple[str, Mapping[object, tuple[object, ...]]]:
        if candidate.kind != "attribute" or candidate.attribute_column is None:
            raise ValueError(f"unsupported signal kind {candidate.kind!r}")
        roots, _, _ = self._root_rows(candidate.source_table)
        reached = roots
        current_table = candidate.source_table
        for next_table in candidate.path[1:]:
            reached = self._traverse(reached, current_table, next_table)
            current_table = next_table
        if current_table != candidate.attribute_table:
            raise ValueError("attribute candidate endpoint does not match its path")
        column_values = self._column(current_table, candidate.attribute_column)
        values = {
            entity_id: tuple(
                atomic_value
                for row_index in row_indices
                if (value := column_values[row_index]) is not None
                for atomic_value in _flatten_values(value)
                if atomic_value is not None
            )
            for entity_id, row_indices in reached.items()
        }
        arrow_type = self.dataset.get_table(current_table).schema.field(
            candidate.attribute_column
        ).type
        value_kind = _attribute_kind(
            candidate.attribute_column,
            arrow_type,
            values.values(),
            is_declared_time=(
                self.dataset.get_time_column(current_table) == candidate.attribute_column
            ),
        )
        return value_kind, values

    def _wl_subtree_features(
        self,
        candidate: SignalCandidate,
        root_rows: Mapping[object, tuple[int, ...]],
        entity_ids: tuple[object, ...],
    ) -> Mapping[object, Mapping[tuple[int, int, int], float]]:
        """Build WL subtree features for typed ego-graphs along one signal path."""
        path_tables = tuple(dict.fromkeys(candidate.path))
        offsets: dict[str, int] = {}
        node_tables: list[str] = []
        for table_name in path_tables:
            offsets[table_name] = len(node_tables)
            node_tables.extend(
                [table_name] * self.dataset.get_table(table_name).num_rows
            )
        adjacency: list[list[tuple[int, str]]] = [[] for _ in node_tables]

        def connect(left_node: int, right_node: int, left_label: str, right_label: str) -> None:
            adjacency[left_node].append((right_node, left_label))
            adjacency[right_node].append((left_node, right_label))

        for left_table, right_table in zip(candidate.path, candidate.path[1:]):
            edge = self.graph.get_relationship(left_table, right_table)
            if edge is None:
                raise ValueError(f"no relationship for graph kernel step {left_table!r} → {right_table!r}")
            edge_token = (
                f"{edge.source_table}.{edge.source_column}->"
                f"{edge.target_table}.{edge.target_column}"
            )
            if edge.source_table == left_table:
                foreign_keys = self._column(left_table, edge.source_column)
                target_index = self._index(right_table, edge.target_column)
                for left_row, key in enumerate(foreign_keys):
                    if key is None:
                        continue
                    for right_row in target_index.get(key, ()):
                        connect(
                            offsets[left_table] + left_row,
                            offsets[right_table] + right_row,
                            f"forward:{edge_token}",
                            f"reverse:{edge_token}",
                        )
            else:
                foreign_keys = self._column(right_table, edge.source_column)
                target_index = self._index(left_table, edge.target_column)
                for right_row, key in enumerate(foreign_keys):
                    if key is None:
                        continue
                    for left_row in target_index.get(key, ()):
                        connect(
                            offsets[left_table] + left_row,
                            offsets[right_table] + right_row,
                            f"reverse:{edge_token}",
                            f"forward:{edge_token}",
                        )

        table_labels = {name: index for index, name in enumerate(path_tables)}
        colors = [table_labels[table_name] for table_name in node_tables]
        color_layers = [colors]
        radius = min(max(candidate.hops, 1), 3)
        for _ in range(radius):
            signatures = [
                (colors[node], tuple(sorted((label, colors[neighbor]) for neighbor, label in neighbors)))
                for node, neighbors in enumerate(adjacency)
            ]
            palette = {signature: index for index, signature in enumerate(sorted(set(signatures)))}
            colors = [palette[signature] for signature in signatures]
            color_layers.append(colors)

        features_by_entity: dict[object, Mapping[tuple[int, int, int], float]] = {}
        source_offset = offsets[candidate.source_table]
        for entity_id in entity_ids:
            roots = root_rows.get(entity_id, ())
            distances: dict[int, int] = {}
            frontier = deque()
            for row_index in roots:
                node = source_offset + row_index
                distances[node] = 0
                frontier.append(node)
            while frontier:
                node = frontier.popleft()
                distance = distances[node]
                if distance >= radius:
                    continue
                for neighbor, _ in adjacency[node]:
                    if neighbor not in distances:
                        distances[neighbor] = distance + 1
                        frontier.append(neighbor)
            feature_counts: Counter[tuple[int, int, int]] = Counter()
            for node, distance in distances.items():
                for iteration, layer in enumerate(color_layers):
                    feature_counts[(distance, iteration, layer[node])] += 1
            features_by_entity[entity_id] = dict(feature_counts)
        return features_by_entity

    def score(
        self,
        materialized: MaterializedSignal,
        query_entity_id: object,
        candidate_entity_id: object,
    ) -> float | None:
        """Score one entity pair for one materialized signal, in ``[0, 1]``."""
        query_values = materialized.values_for(query_entity_id)
        candidate_values = materialized.values_for(candidate_entity_id)
        if not query_values or not candidate_values:
            return None
        if materialized.measure in {"jensen_shannon_similarity", "hellinger_similarity"}:
            return _distribution_similarity(
                query_values,
                candidate_values,
                materialized.value_kind,
                materialized.reference_values,
                materialized.measure,
            )
        if materialized.measure == "categorical_agreement":
            return float(query_values[0] == candidate_values[0])
        if materialized.measure == "idf_weighted_jaccard":
            return _weighted_jaccard(query_values, candidate_values, materialized.value_weights)
        if materialized.measure in {"jaccard_overlap", "jaccard_neighborhood"}:
            return _jaccard(query_values, candidate_values)
        if materialized.measure == "boolean_rate_similarity":
            return 1.0 - abs(_boolean_rate(query_values) - _boolean_rate(candidate_values))
        if materialized.measure == "e5_embedding_cosine":
            assert materialized.embeddings_by_entity is not None
            query_vector = materialized.embeddings_by_entity.get(query_entity_id)
            candidate_vector = materialized.embeddings_by_entity.get(candidate_entity_id)
            return None if query_vector is None or candidate_vector is None else _vector_cosine(query_vector, candidate_vector)
        if materialized.measure == "wl_subtree_kernel_similarity":
            assert materialized.graph_features_by_entity is not None
            query_features = materialized.graph_features_by_entity.get(query_entity_id)
            candidate_features = materialized.graph_features_by_entity.get(candidate_entity_id)
            return (
                None
                if query_features is None or candidate_features is None
                else _sparse_cosine(query_features, candidate_features)
            )
        if materialized.measure == "diffusion_distance_similarity_t1":
            assert materialized.diffusion_profiles_by_entity is not None
            query_profile = materialized.diffusion_profiles_by_entity.get(query_entity_id)
            candidate_profile = materialized.diffusion_profiles_by_entity.get(candidate_entity_id)
            return (
                None
                if query_profile is None or candidate_profile is None
                else _diffusion_profile_similarity(query_profile, candidate_profile)
            )
        if materialized.measure in {"percentile_distance", "temporal_percentile_distance"}:
            transform = _time_number if materialized.measure.startswith("temporal") else float
            return _percentile_distance(
                transform(query_values[0]), transform(candidate_values[0]), materialized.reference_values
            )
        if materialized.measure in {"percentile_wasserstein", "temporal_percentile_wasserstein"}:
            transform = _time_number if materialized.measure.startswith("temporal") else float
            return _percentile_wasserstein(
                query_values, candidate_values, materialized.reference_values, transform
            )
        if materialized.measure == "degree_similarity":
            denominator = materialized.max_degree or 0
            if denominator == 0:
                return 1.0
            return max(0.0, 1.0 - abs(len(query_values) - len(candidate_values)) / denominator)
        raise ValueError(f"unknown measure {materialized.measure!r}")

    def score_query(
        self,
        query_entity_id: object,
        signals: Sequence[MaterializedSignal],
        *,
        weights: Mapping[str, float] | None = None,
        include_self: bool = False,
        candidate_entity_ids: Sequence[object] | None = None,
    ) -> list[PairSimilarity]:
        """Score one query against all same-table entities using selected signals."""
        if not signals:
            raise ValueError("at least one materialized signal is required")
        entity_table = signals[0].entity_table
        if any(signal.entity_table != entity_table for signal in signals):
            raise ValueError("all selected signals must share one entity table")
        entity_ids = signals[0].entity_ids
        if any(signal.entity_ids != entity_ids for signal in signals[1:]):
            raise ValueError("selected signals must be materialized over the same entities")
        query = _resolve_entity_id(query_entity_id, entity_ids)
        results = []
        if candidate_entity_ids is None:
            candidate_ids = entity_ids
        else:
            from src.similarity.cohort import resolve_entity_ids
            candidate_ids = resolve_entity_ids(candidate_entity_ids, entity_ids)
        for candidate_id in candidate_ids:
            if not include_self and candidate_id == query:
                continue
            components = []
            for signal in signals:
                score = self.score(signal, query, candidate_id)
                confidence = _pair_confidence(signal, query, candidate_id)
                weight = (weights or {}).get(signal.candidate.id, 1.0)
                components.append(
                    {
                        "signal_id": signal.candidate.id,
                        "measure": signal.measure,
                        "score": score,
                        "weight": weight,
                        "confidence": confidence,
                        "effective_weight": weight * confidence,
                    }
                )
            results.append(
                PairSimilarity(
                    query_entity_id=query,
                    candidate_entity_id=candidate_id,
                    similarity_score=combine_signal_scores(components),
                    components=tuple(components),
                )
            )
        return sorted(
            results,
            key=lambda result: (
                result.similarity_score is None,
                -(result.similarity_score or 0.0),
                str(result.candidate_entity_id),
            ),
        )

    def _root_rows(self, table: str) -> tuple[dict[object, tuple[int, ...]], tuple[object, ...], str]:
        primary_key = self.dataset.get_primary_key(table)
        row_count = self.dataset.get_table(table).num_rows
        if primary_key is None:
            entity_ids = tuple(range(row_count))
            return ({index: (index,) for index in entity_ids}, entity_ids, "row_position")
        values = self._column(table, primary_key)
        if any(value is None for value in values):
            raise ValueError(f"table {table!r} has null primary-key values")
        if len(set(values)) != len(values):
            raise ValueError(f"table {table!r} has duplicate primary-key values")
        entity_ids = tuple(values)
        return ({value: (index,) for index, value in enumerate(values)}, entity_ids, "primary_key")

    def _traverse(
        self,
        rows_by_entity: Mapping[object, tuple[int, ...]],
        current_table: str,
        next_table: str,
    ) -> dict[object, tuple[int, ...]]:
        edge = self.graph.get_relationship(current_table, next_table)
        if edge is None:
            raise ValueError(f"no foreign-key relationship from {current_table!r} to {next_table!r}")
        if edge.source_table == current_table and edge.target_table == next_table:
            source_values = self._column(current_table, edge.source_column)
            target_index = self._index(next_table, edge.target_column)
        elif edge.target_table == current_table and edge.source_table == next_table:
            source_values = self._column(current_table, edge.target_column)
            target_index = self._index(next_table, edge.source_column)
        else:  # pragma: no cover - protected by SchemaGraph.get_relationship
            raise ValueError("schema edge does not match requested traversal")
        traversed = {}
        for entity_id, row_indices in rows_by_entity.items():
            targets = []
            for row_index in row_indices:
                value = source_values[row_index]
                if value is not None:
                    targets.extend(target_index.get(value, ()))
            traversed[entity_id] = tuple(dict.fromkeys(targets))
        return traversed

    def _relationship_values(
        self, rows_by_entity: Mapping[object, tuple[int, ...]], target_table: str
    ) -> tuple[dict[object, tuple[object, ...]], str]:
        primary_key = self.dataset.get_primary_key(target_table)
        if primary_key is None:
            return ({key: tuple(rows) for key, rows in rows_by_entity.items()}, "relationship")
        key_values = self._column(target_table, primary_key)
        return (
            {key: tuple(key_values[index] for index in rows) for key, rows in rows_by_entity.items()},
            "relationship",
        )

    def _embed_e5_text_collections(
        self, values_by_entity: Mapping[object, tuple[object, ...]]
    ) -> Mapping[object, np.ndarray]:
        return self._get_e5_embedder().encode_collections(values_by_entity)

    def _get_e5_embedder(self) -> _VllmE5TextEmbedder:
        if self._e5_embedder is None:
            self._e5_embedder = _VllmE5TextEmbedder(
                self.text_embedding_model,
                self.model_cache_root,
                self.text_embedding_cache,
                batch_size=self.embedding_batch_size,
            )
        return self._e5_embedder

    def _column(self, table: str, column: str) -> list[object]:
        key = (table, column)
        if key not in self._columns:
            self._columns[key] = self.dataset.get_table(table)[column].to_pylist()
        return self._columns[key]

    def _index(self, table: str, column: str) -> Mapping[object, tuple[int, ...]]:
        key = (table, column)
        if key not in self._indices:
            index: dict[object, list[int]] = defaultdict(list)
            for row_index, value in enumerate(self._column(table, column)):
                if value is not None:
                    index[value].append(row_index)
            self._indices[key] = {value: tuple(rows) for value, rows in index.items()}
        return self._indices[key]


def combine_signal_scores(components: Iterable[Mapping[str, object]]) -> float | None:
    """Return the weighted mean of available components, or ``None`` if absent."""
    weighted_total = 0.0
    total_weight = 0.0
    for component in components:
        score = component.get("score")
        weight = component.get("effective_weight", component.get("weight", 1.0))
        if score is None:
            continue
        if not isinstance(score, (int, float)) or not isinstance(weight, (int, float)):
            raise ValueError("component scores and weights must be numeric")
        if weight < 0:
            raise ValueError("signal weights must be non-negative")
        weighted_total += float(score) * float(weight)
        total_weight += float(weight)
    return None if total_weight == 0 else weighted_total / total_weight


def _attribute_kind(
    column: str,
    arrow_type: pa.DataType,
    values: Iterable[tuple[object, ...]],
    *,
    is_declared_time: bool,
) -> str:
    if is_declared_time:
        return "temporal"
    if pa.types.is_boolean(arrow_type):
        return "boolean"
    if pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type) or pa.types.is_decimal(arrow_type):
        return "numeric"
    if pa.types.is_date(arrow_type) or pa.types.is_timestamp(arrow_type) or pa.types.is_time(arrow_type):
        return "temporal"
    column_lower = column.lower()
    # Names normally identify people/organisations rather than describe their
    # semantic content. Keeping them categorical prevents accidental lexical
    # similarity between two unrelated entity names.
    if column_lower == "name" or column_lower.endswith("_name"):
        return "categorical"
    if any(hint in column_lower for hint in _TEXT_COLUMN_HINTS):
        return "text"
    strings = [str(value) for collection in values for value in collection]
    if strings and sum(len(value) for value in strings) / len(strings) > 40:
        return "text"
    return "categorical"


def _validate_distribution_variant(
    value_kind: str, values_by_entity: Mapping[object, Sequence[object]]
) -> None:
    if value_kind == "text":
        raise ValueError("Jensen-Shannon and Hellinger variants require discrete or numeric values")
    if value_kind not in {"categorical", "boolean", "numeric", "temporal", "relationship"}:
        raise ValueError(f"distribution distances do not support value kind {value_kind!r}")
    if value_kind in {"numeric", "temporal"} and max(
        (len(values) for values in values_by_entity.values()), default=0
    ) <= 1:
        raise ValueError(
            "Jensen-Shannon and Hellinger on numeric or temporal signals need a multi-valued "
            "path distribution; use the default percentile measure for scalar signals"
        )


def _distribution_similarity(
    left: Sequence[object],
    right: Sequence[object],
    value_kind: str,
    reference_values: Sequence[float] | None,
    measure: str,
) -> float:
    """Compare empirical categorical or quantile-binned distributions."""
    left_counts = _distribution_counts(left, value_kind, reference_values)
    right_counts = _distribution_counts(right, value_kind, reference_values)
    left_total, right_total = sum(left_counts.values()), sum(right_counts.values())
    if not left_total or not right_total:
        return 0.0
    support = left_counts.keys() | right_counts.keys()
    left_prob = {key: left_counts.get(key, 0) / left_total for key in support}
    right_prob = {key: right_counts.get(key, 0) / right_total for key in support}
    if measure == "hellinger_similarity":
        distance = math.sqrt(
            max(
                0.0,
                0.5
                * sum(
                    (math.sqrt(left_prob[key]) - math.sqrt(right_prob[key])) ** 2
                    for key in support
                ),
            )
        )
        return max(0.0, min(1.0, 1.0 - distance))
    midpoint = {key: 0.5 * (left_prob[key] + right_prob[key]) for key in support}
    divergence = 0.0
    for key in support:
        if left_prob[key]:
            divergence += 0.5 * left_prob[key] * math.log2(left_prob[key] / midpoint[key])
        if right_prob[key]:
            divergence += 0.5 * right_prob[key] * math.log2(right_prob[key] / midpoint[key])
    # The square root of base-2 Jensen-Shannon divergence is a metric in [0, 1].
    return max(0.0, min(1.0, 1.0 - math.sqrt(max(0.0, divergence))))


def _distribution_counts(
    values: Sequence[object], value_kind: str, reference_values: Sequence[float] | None
) -> Counter[object]:
    if value_kind not in {"numeric", "temporal"}:
        return Counter(values)
    transform = _time_number if value_kind == "temporal" else float
    counts: Counter[object] = Counter()
    for value in values:
        percentile = _percentile(transform(value), reference_values)
        bucket = min(_DISTRIBUTION_BIN_COUNT - 1, int(percentile * _DISTRIBUTION_BIN_COUNT))
        counts[bucket] += 1
    return counts


def _diffusion_profiles(
    entity_ids: Sequence[object],
    values_by_entity: Mapping[object, Sequence[object]],
) -> dict[object, Mapping[int, float]]:
    """Create one-step random-walk profiles on the shared-neighbour graph.

    Each signal value is a graph node connected to the source entities that
    reach it. Projection weights each shared value by the inverse number of
    source entities using it. A fixed restart mass keeps each profile rooted
    at its entity and makes sparse profiles comparable.
    """
    entity_index = {entity_id: index for index, entity_id in enumerate(entity_ids)}
    owners_by_value: dict[object, list[int]] = defaultdict(list)
    for entity_id in entity_ids:
        # Preserve encounter order: stable endpoint traversal keeps floating-
        # point accumulation and serialized results reproducible across runs.
        for value in dict.fromkeys(values_by_entity.get(entity_id, ())):
            owners_by_value[value].append(entity_index[entity_id])

    projected_edge_count = sum(len(owners) * (len(owners) - 1) for owners in owners_by_value.values())
    if projected_edge_count > _MAX_DIFFUSION_PROJECTED_EDGES:
        raise ValueError(
            "diffusion projection would contain "
            f"{projected_edge_count:,} directed edges (limit {_MAX_DIFFUSION_PROJECTED_EDGES:,}); "
            "use a smaller entity table or another relationship measure"
        )

    adjacency: list[dict[int, float]] = [defaultdict(float) for _ in entity_ids]
    for owners in owners_by_value.values():
        if len(owners) < 2:
            continue
        contribution = 1.0 / (len(owners) - 1)
        for source in owners:
            for target in owners:
                if source != target:
                    adjacency[source][target] += contribution

    restart = _DIFFUSION_RESTART_PROBABILITY
    profiles: dict[object, Mapping[int, float]] = {}
    for entity_id, row in zip(entity_ids, adjacency, strict=True):
        total = sum(row.values())
        if total <= 0:
            continue
        profile = {entity_index[entity_id]: restart}
        profile.update(
            {target: (1.0 - restart) * weight / total for target, weight in row.items()}
        )
        profiles[entity_id] = profile
    return profiles


def _diffusion_profile_similarity(
    left: Mapping[int, float], right: Mapping[int, float]
) -> float:
    support = left.keys() | right.keys()
    distance = math.sqrt(
        sum((left.get(index, 0.0) - right.get(index, 0.0)) ** 2 for index in support)
    )
    return max(0.0, min(1.0, 1.0 - distance / math.sqrt(2.0)))


def _sparse_cosine(
    left: Mapping[tuple[int, int, int], float],
    right: Mapping[tuple[int, int, int], float],
) -> float:
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _measure_for_attribute(
    value_kind: str, values: Iterable[tuple[object, ...]]
) -> tuple[
    str,
    tuple[float, float] | None,
    tuple[float, ...] | None,
    Mapping[object, float] | None,
]:
    collections = tuple(values)
    if value_kind == "text":
        return "e5_embedding_cosine", None, None, None
    if value_kind == "categorical":
        cardinality = max((len(collection) for collection in collections), default=0)
        if cardinality <= 1:
            return "categorical_agreement", None, None, None
        return "idf_weighted_jaccard", None, None, _inverse_document_frequency(collections)
    if value_kind == "boolean":
        cardinality = max((len(collection) for collection in collections), default=0)
        return ("categorical_agreement" if cardinality <= 1 else "boolean_rate_similarity"), None, None, None
    if value_kind == "numeric":
        flattened = [float(value) for collection in collections for value in collection]
        reference_values = tuple(sorted(flattened))
        cardinality = max((len(collection) for collection in collections), default=0)
        return (
            "percentile_distance" if cardinality <= 1 else "percentile_wasserstein",
            None,
            reference_values,
            None,
        )
    if value_kind == "temporal":
        flattened = [_time_number(value) for collection in collections for value in collection]
        reference_values = tuple(sorted(flattened))
        cardinality = max((len(collection) for collection in collections), default=0)
        return (
            "temporal_percentile_distance" if cardinality <= 1 else "temporal_percentile_wasserstein",
            None,
            reference_values,
            None,
        )
    raise ValueError(f"unknown attribute value kind {value_kind!r}")


def _flatten_values(value: object) -> tuple[object, ...]:
    """Flatten Arrow list values while preserving scalar text/numeric values."""
    if isinstance(value, (list, tuple, np.ndarray)):
        flattened: list[object] = []
        for item in value:
            flattened.extend(_flatten_values(item))
        return tuple(flattened)
    return (value,)


def _normalise_category(value: object) -> object:
    if not isinstance(value, str):
        return value
    value = html.unescape(value).casefold().strip()
    return re.sub(r"\s+", " ", value)


def _inverse_document_frequency(
    collections: Sequence[Sequence[object]],
) -> Mapping[object, float]:
    documents = len(collections)
    frequency: Counter[object] = Counter()
    for collection in collections:
        frequency.update(set(collection))
    return {
        value: math.log((documents + 1) / (count + 1)) + 1.0
        for value, count in frequency.items()
    }


def _weighted_jaccard(
    left: Sequence[object], right: Sequence[object], weights: Mapping[object, float] | None
) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    weights = weights or {}
    intersection_weight = sum(weights.get(value, 1.0) for value in left_set & right_set)
    union_weight = sum(weights.get(value, 1.0) for value in union)
    return intersection_weight / union_weight if union_weight else 1.0


def _boolean_rate(values: Sequence[object]) -> float:
    return sum(bool(value) for value in values) / len(values)


def _percentile(value: float, reference: Sequence[float] | None) -> float:
    if not reference:
        return 0.5
    left = bisect_left(reference, value)
    right = bisect_right(reference, value)
    return (left + right) / (2 * len(reference))


def _percentile_distance(left: float, right: float, reference: Sequence[float] | None) -> float:
    return max(0.0, 1.0 - abs(_percentile(left, reference) - _percentile(right, reference)))


def _percentile_wasserstein(
    left: Sequence[object],
    right: Sequence[object],
    reference: Sequence[float] | None,
    transform: Any,
) -> float:
    left_values = sorted(_percentile(transform(value), reference) for value in left)
    right_values = sorted(_percentile(transform(value), reference) for value in right)
    samples = max(len(left_values), len(right_values))
    if samples == 1:
        return 1.0 - abs(left_values[0] - right_values[0])
    distance = sum(
        abs(_quantile(left_values, index / (samples - 1)) - _quantile(right_values, index / (samples - 1)))
        for index in range(samples)
    ) / samples
    return max(0.0, 1.0 - distance)


def _vector_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator == 0 else float(np.dot(left, right) / denominator)


def _pair_confidence(
    signal: MaterializedSignal, query_entity_id: object, candidate_entity_id: object
) -> float:
    """Down-weight low-support path collections without altering direct attributes.

    A direct attribute is one observation by design. A collection reached by a
    path is evidence accumulated from several related rows; one observed review
    or one shared neighbour should not have the same influence as a large
    collection. The square-root schedule reaches full confidence at five values
    while retaining a non-zero contribution for sparse entities.
    """
    if signal.candidate.hops == 0 or signal.measure == "categorical_agreement":
        return 1.0
    if signal.measure not in {
        "idf_weighted_jaccard", "jaccard_neighborhood", "boolean_rate_similarity",
        "percentile_wasserstein", "temporal_percentile_wasserstein",
        "e5_embedding_cosine", "jensen_shannon_similarity", "hellinger_similarity",
        "wl_subtree_kernel_similarity", "diffusion_distance_similarity_t1",
    }:
        return 1.0
    support = min(
        len(signal.values_for(query_entity_id)), len(signal.values_for(candidate_entity_id))
    )
    return min(1.0, math.sqrt(support / 5.0))


def _jaccard(left: Sequence[object], right: Sequence[object]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def _scalar_distance(left: object, right: object, value_range: tuple[float, float] | None) -> float:
    if value_range is None or value_range[0] == value_range[1]:
        return float(left == right)
    span = value_range[1] - value_range[0]
    return max(0.0, 1.0 - abs(float(left) - float(right)) / span)


def _wasserstein_similarity(
    left: Sequence[object], right: Sequence[object], value_range: tuple[float, float] | None
) -> float:
    if value_range is None or value_range[0] == value_range[1]:
        return 1.0
    left_sorted = sorted(float(value) for value in left)
    right_sorted = sorted(float(value) for value in right)
    samples = max(len(left_sorted), len(right_sorted))
    if samples == 1:
        return _scalar_distance(left_sorted[0], right_sorted[0], value_range)
    distance = sum(
        abs(_quantile(left_sorted, index / (samples - 1)) - _quantile(right_sorted, index / (samples - 1)))
        for index in range(samples)
    ) / samples
    return max(0.0, 1.0 - distance / (value_range[1] - value_range[0]))


def _quantile(values: Sequence[float], probability: float) -> float:
    position = probability * (len(values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _time_number(value: object) -> float:
    if isinstance(value, dt.datetime):
        return value.timestamp()
    if isinstance(value, dt.date):
        return float(value.toordinal())
    if isinstance(value, dt.time):
        return float(value.hour * 3600 + value.minute * 60 + value.second)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError as error:
            raise ValueError(f"unsupported temporal value {value!r}") from error
    raise ValueError(f"unsupported temporal value {value!r}")


def _resolve_entity_id(requested: object, available: Sequence[object]) -> object:
    if requested in available:
        return requested
    matched = [entity_id for entity_id in available if str(entity_id) == str(requested)]
    if len(matched) == 1:
        return matched[0]
    raise KeyError(f"unknown or ambiguous entity ID {requested!r}")
