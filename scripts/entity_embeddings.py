#!/usr/bin/env python3
"""Export relational data and rank saved entity embeddings."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.datasets.local_relbench_dataset import LocalRelBenchDataset
from src.similarity.entity_embeddings import (
    _relbench_snapshot, dataset_fingerprint, entity_frame, load_embeddings, rank_embeddings,
)




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    rank = sub.add_parser('rank', help='write rankings in the SimBench similarity JSONL format')
    rank.add_argument('--embeddings', type=Path, required=True)
    rank.add_argument('--dataset-root', type=Path, default=ROOT / 'data/dev/rel-amazon')
    rank.add_argument('--conditions', type=Path, default=ROOT / 'conditions/rel-amazon/conditions.jsonl')
    rank.add_argument('--condition-id', required=True)
    rank.add_argument('--query-id', required=True, action='append')
    rank.add_argument('--top-k', type=int, default=None)
    rank.add_argument('--output', type=Path, required=True)
    export = sub.add_parser('export-relbench', help='export the reindexed SimBench database and its ID map')
    export.add_argument('--dataset-root', type=Path, default=ROOT / 'data/dev/rel-amazon')
    export.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    try:
        dataset = LocalRelBenchDataset(args.dataset_root)
        if args.command == 'export-relbench':
            if args.output_dir.exists() and any(args.output_dir.iterdir()):
                raise ValueError('export directory must be empty')
            database, original_ids = _relbench_snapshot(dataset)
            db_dir = args.output_dir / 'db'
            db_dir.mkdir(parents=True, exist_ok=True)
            database.save(db_dir)
            with (args.output_dir / 'entity-id-map.jsonl').open('w', encoding='utf-8') as stream:
                for table, values in original_ids.items():
                    for index, original in enumerate(values):
                        stream.write(json.dumps({'table': table, 'entity_id': original, 'relbench_id': index}, default=str) + '\n')
            (args.output_dir / 'simbench-export.json').write_text(json.dumps({'dataset': dataset.schema.name, 'dataset_fingerprint': dataset_fingerprint(dataset), 'dataset_root': str(args.dataset_root.resolve()), 'tables': list(dataset.get_tables())}, indent=2) + '\n')
            print(f'Exported the reindexed database and entity ID map to {args.output_dir}')
        else:
            ids, vectors, metadata = load_embeddings(args.embeddings)
            if metadata.get('format') != 'simbench-entity-embeddings-v1':
                raise ValueError('unsupported embedding artifact format')
            if metadata.get('dataset_fingerprint') != dataset_fingerprint(dataset):
                raise ValueError('embedding artifact was created from a different dataset snapshot')
            expected, _ = entity_frame(dataset, metadata['entity_table'])
            if list(map(str, ids)) != list(map(str, expected)):
                raise ValueError('embedding IDs/order do not match the dataset')
            if args.top_k is not None and args.top_k <= 0:
                raise ValueError('--top-k must be positive')
            conditions = [json.loads(line) for line in args.conditions.read_text().splitlines() if line.strip()]
            condition = next((row for row in conditions if row['id'] == args.condition_id), None)
            if condition is None:
                raise ValueError(f'unknown condition {args.condition_id!r}')
            records = []
            for query in args.query_id:
                pairs = rank_embeddings(query, ids, vectors, metadata['metric'])
                records.append({'result_format': 'signal-similarity-v1', 'condition_id': condition['id'], 'condition': condition['condition'], 'entity_table': metadata['entity_table'], 'entity_id_kind': metadata['entity_id_kind'], 'query_entity_id': next(value for value in ids if str(value) == str(query)), 'model': metadata['model'], 'embedding_artifact': str(args.embeddings.resolve()), 'embedding_metadata': metadata, 'signals': [], 'pairs': pairs[:args.top_k] if args.top_k else pairs})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('w', encoding='utf-8') as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
            print(f'Wrote {len(records)} query rankings to {args.output}')
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
