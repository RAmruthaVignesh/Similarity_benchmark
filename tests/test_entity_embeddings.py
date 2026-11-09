"""Checks for entity alignment and comparison of saved rankings."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.compare_entity_rankings import compare_scores
from src.datasets.local_relbench_dataset import LocalRelBenchDataset
from src.similarity.entity_embeddings import (
    aligned_feature_frame, dataset_fingerprint, encode_entities, load_embeddings, rank_embeddings,
    save_embeddings,
)


class EntityEmbeddingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'manifest.yaml').write_text('''name: toy
tables:
  product:
    pkey: product_id
    time_col: null
    fkeys: {}
  review:
    pkey: null
    time_col: review_time
    fkeys:
      product_id: product
''')
        pq.write_table(pa.table({'product_id': [100, 300], 'price': [3.0, 8.0]}), self.root / 'product.parquet')
        pq.write_table(pa.Table.from_pandas(pd.DataFrame({'review_time': pd.to_datetime(['2010-01-01', '2011-01-01']), 'product_id': [300, 100]}), preserve_index=False), self.root / 'review.parquet')
        self.dataset = LocalRelBenchDataset(self.root)

    def test_artifact_round_trip_and_cosine_ranking(self):
        p = self.root / 'vectors.npz'
        save_embeddings(p, [100, 300, 500], np.array([[1, 0], [0.9, 0.1], [0, 1]]), {'metric': 'cosine'})
        ids, vectors, metadata = load_embeddings(p)
        self.assertEqual(ids, [100, 300, 500])
        self.assertEqual(metadata['metric'], 'cosine')
        pairs = rank_embeddings('100', ids, vectors, 'cosine')
        self.assertEqual([p['candidate_entity_id'] for p in pairs], [300, 500])
        self.assertAlmostEqual(pairs[1]['similarity_score'], 0.0)

    def test_comparison_keeps_ties_and_rejects_different_candidate_sets(self):
        a = {'a': 1., 'b': 1., 'c': 0., 'd': None}
        b = {'a': 1., 'b': 0., 'c': 0., 'd': 1.}
        result = compare_scores(a, b, 1)
        self.assertEqual(result['shared_scored'], 3)
        self.assertEqual((result['left_top_size'], result['right_top_size']), (2, 1))
        self.assertEqual(result['top_jaccard'], 0.5)
        with self.assertRaisesRegex(ValueError, 'candidate universes differ'):
            compare_scores(a, {'a': 1.}, 1)

    def test_fingerprint_changes_when_snapshot_changes(self):
        first = dataset_fingerprint(self.dataset)
        (self.root / 'manifest.yaml').write_text((self.root / 'manifest.yaml').read_text() + '\n')
        self.assertNotEqual(first, dataset_fingerprint(self.dataset))

    def test_feature_table_aligns_entity_ids_and_rejects_duplicates(self):
        path = self.root / 'features.parquet'
        pd.DataFrame({'entity_id': [300, 100], 'aggregate': [8., 3.]}).to_parquet(path)
        aligned = aligned_feature_frame(path, 'entity_id', [100, 300])
        self.assertEqual(aligned['aggregate'].tolist(), [3., 8.])
        pd.DataFrame({'entity_id': [100, 100], 'aggregate': [3., 4.]}).to_parquet(path)
        with self.assertRaisesRegex(ValueError, 'each SimBench entity exactly once'):
            aligned_feature_frame(path, 'entity_id', [100, 300])



    def test_adapter_validates_entity_alignment_and_vectors(self):
        adapter = lambda dataset, table: ([100, 300], [[1.0, 0.0], [0.0, 1.0]], {'source': 'example'})
        ids, vectors, metadata = encode_entities(adapter, self.dataset, 'product')
        self.assertEqual(ids, [100, 300])
        self.assertEqual(vectors.shape, (2, 2))
        self.assertEqual(metadata['source'], 'example')
        bad = lambda dataset, table: ([300, 100], [[1.0], [2.0]], {})
        with self.assertRaisesRegex(ValueError, 'IDs/order'):
            encode_entities(bad, self.dataset, 'product')


if __name__ == '__main__':
    unittest.main()
