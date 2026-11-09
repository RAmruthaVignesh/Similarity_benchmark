"""Portable entity embeddings and ranking utilities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from src.datasets.local_relbench_dataset import LocalRelBenchDataset


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(dataset: LocalRelBenchDataset) -> str:
    digest = hashlib.sha256()
    for path in [dataset.root / 'manifest.yaml', *(dataset.root / f'{name}.parquet' for name in sorted(dataset.get_tables()))]:
        digest.update(path.name.encode())
        digest.update(file_sha256(path).encode())
    return digest.hexdigest()


def entity_frame(dataset: LocalRelBenchDataset, table: str) -> tuple[list[Any], pd.DataFrame]:
    frame = dataset.get_table(table).to_pandas()
    key = dataset.get_primary_key(table)
    ids = frame[key].tolist() if key else list(range(len(frame)))
    ids = [v.item() if isinstance(v, np.generic) else v for v in ids]
    if any(pd.isna(v) for v in ids) or len({str(v) for v in ids}) != len(ids):
        raise ValueError(f'{table}: entity IDs must be non-null and unique when converted to strings')
    return ids, frame


class EntityEmbeddingAdapter(Protocol):
    """Produce vectors, in dataset entity order, and descriptive metadata."""

    def __call__(self, dataset: LocalRelBenchDataset, table: str) -> tuple[list[Any], np.ndarray, dict[str, Any]]: ...


def encode_entities(adapter: EntityEmbeddingAdapter, dataset: LocalRelBenchDataset, table: str):
    """Run an adapter and validate its entity alignment and embedding matrix."""
    ids, vectors, metadata = adapter(dataset, table)
    expected, _ = entity_frame(dataset, table)
    if list(map(str, ids)) != list(map(str, expected)):
        raise ValueError('embedding IDs/order do not match the dataset')
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or len(array) != len(ids) or not np.isfinite(array).all():
        raise ValueError('embedding matrix must have one finite vector per entity')
    if not isinstance(metadata, dict):
        raise ValueError('embedding metadata must be a dictionary')
    return ids, array, metadata


def aligned_feature_frame(path, id_column, ids):
    if not id_column:
        raise ValueError('a feature parquet requires --feature-id-column')
    frame = pd.read_parquet(path)
    if id_column not in frame:
        raise ValueError(f'feature parquet has no ID column {id_column!r}')
    keys = frame[id_column].map(str)
    if keys.duplicated().any() or set(keys) != set(map(str, ids)):
        raise ValueError('feature parquet must contain each SimBench entity exactly once')
    return frame.set_index(keys).loc[list(map(str, ids))].reset_index(drop=True)




def _relbench_snapshot(dataset):
    """Recreate RelBench's index-based keys without changing the local dataset."""
    from relbench.base import Database, Table

    tables = {}
    original_ids = {}
    for name in dataset.get_tables():
        spec = dataset.schema.tables[name]
        frame = dataset.get_table(name).to_pandas()
        if spec.primary_key:
            ordered = frame.sort_values(spec.time_column) if spec.time_column else frame
            original_ids[name] = ordered[spec.primary_key].tolist()
        tables[name] = Table(frame, {f.column: f.target_table for f in spec.foreign_keys}, spec.primary_key, spec.time_column)
    db = Database(tables)
    db.reindex_pkeys_and_fkeys()
    return db, original_ids






def save_embeddings(path: Path, ids, vectors, metadata):
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or len(array) != len(ids) or not np.isfinite(array).all():
        raise ValueError('embedding matrix must have one finite vector per entity')
    if len(set(map(str, ids))) != len(ids):
        raise ValueError('entity IDs must be unique')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        np.savez_compressed(stream, vectors=array, ids_json=np.array(json.dumps(ids)), metadata_json=np.array(json.dumps(metadata, default=str)))


def load_embeddings(path: Path):
    with np.load(path, allow_pickle=False) as data:
        ids = json.loads(str(data['ids_json'].item()))
        metadata = json.loads(str(data['metadata_json'].item()))
        vectors = np.asarray(data['vectors'], dtype=np.float32)
    if vectors.ndim != 2 or len(ids) != len(vectors) or not np.isfinite(vectors).all() or len(set(map(str, ids))) != len(ids):
        raise ValueError('invalid entity embedding artifact')
    return ids, vectors, metadata


def rank_embeddings(query_id, ids, vectors, metric):
    indices = {str(value): i for i, value in enumerate(ids)}
    if str(query_id) not in indices:
        raise KeyError(f'query entity {query_id!r} is absent from the embeddings')
    i = indices[str(query_id)]
    if metric == 'cosine':
        norms = np.linalg.norm(vectors, axis=1)
        if norms[i] == 0:
            raise ValueError('query has a zero embedding; cosine similarity is undefined')
        scores = np.divide(vectors @ vectors[i], norms * norms[i], out=np.full(len(ids), np.nan, dtype=np.float32), where=norms > 0)
    elif metric == 'dot':
        scores = vectors @ vectors[i]
    else:
        raise ValueError(f'unknown embedding metric {metric!r}')
    pairs = [{'query_entity_id': ids[i], 'candidate_entity_id': candidate, 'similarity_score': float(scores[j]) if np.isfinite(scores[j]) else None, 'components': []} for j, candidate in enumerate(ids) if j != i]
    return sorted(pairs, key=lambda p: (p['similarity_score'] is None, -(p['similarity_score'] or 0.0), str(p['candidate_entity_id'])))
