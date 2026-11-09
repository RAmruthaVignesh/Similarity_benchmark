import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.similarity.conditioned_analysis import METHODS, compare_condition, load_condition, shared_scores


class ConditionedAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.run, self.robust = root / 'run', root / 'robust'
        (self.run / 'condition-001').mkdir(parents=True)
        (self.robust / 'condition-001').mkdir(parents=True)
        self.manifest = {'conditions': {'condition-001': 'Find products'},
                         'condition_tables': {'condition-001': 'product'},
                         'config': {'dataset_fingerprint': 'dataset'}}
        self.original = self.run / 'condition-001' / 'simbench.jsonl'
        self.baseline = self.run / 'condition-001' / 'baseline.jsonl'
        self.pairs = self.robust / 'condition-001' / 'pairs.jsonl'
        self.profile = self.robust / 'condition-001' / 'profile.json'
        def row(query, values):
            return {'condition_id': 'condition-001', 'condition': 'Find products',
                    'entity_table': 'product', 'query_entity_id': query,
                    'pairs': [{'candidate_entity_id': i, 'similarity_score': value}
                              for i, value in values.items()]}
        self.original.write_text(''.join(json.dumps(r) + '\n' for r in [
            row('train', {'a': 0.2}), row('test', {'a': 0.8, 'b': 0.8, 'c': None})]))
        self.baseline.write_text(json.dumps(row('test', {'a': 0.4, 'b': 0.3, 'c': 0.9})) + '\n')
        self.pairs.write_text(''.join(json.dumps({'query': 'test', 'candidate': c,
                                                  'experimental': score, 'coverage': 0.5}) + '\n'
                                      for c, score in {'a': 1.0, 'b': 0.0, 'c': 0.5}.items()))
        self.profile.write_text(json.dumps({
            'format': 'simbench-experimental-scoring-v6', 'normalization': 'bounded-robust-z', 'condition_id': 'condition-001',
            'condition': 'Find products', 'entity_table': 'product', 'dataset_fingerprint': 'dataset',
            'training_queries': ['train'], 'evaluation_queries': ['test'],
            'input_sha256': hashlib.sha256(self.original.read_bytes()).hexdigest(),
        }))

    def load(self):
        return load_condition(self.run, self.robust, 'condition-001', self.manifest)

    def test_all_pairs_use_same_heldout_reference_and_three_way_candidates(self):
        loaded = self.load()
        self.assertEqual(set(loaded['scores']), {'test'})
        common = shared_scores(loaded['scores']['test'])
        self.assertEqual({tuple(scores) for scores in common.values()}, {('a', 'b')})
        rows = compare_condition('condition-001', loaded, 1)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r['candidates'] == 3 and r['shared_scored'] == 2 for r in rows))
        original_robust = rows[0]
        self.assertEqual((original_robust['method_a'], original_robust['method_b']), METHODS[:2])
        self.assertEqual(original_robust['left_top_size'], 2)  # top-score tie is preserved
        self.assertEqual(original_robust['top_jaccard'], 0.5)

    def test_changed_original_is_rejected(self):
        self.original.write_text(self.original.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'changed since'):
            self.load()

    def test_different_candidate_pools_are_rejected(self):
        self.pairs.write_text(self.pairs.read_text().splitlines()[0] + '\n')
        with self.assertRaisesRegex(ValueError, 'candidate pools differ'):
            self.load()

    def test_training_reference_cannot_be_evaluated(self):
        profile = json.loads(self.profile.read_text())
        profile['training_queries'] = ['test']
        self.profile.write_text(json.dumps(profile))
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            self.load()


if __name__ == '__main__':
    unittest.main()
