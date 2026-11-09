#!/usr/bin/env python3
"""Compare saved entity rankings on matched conditions, queries and candidates."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def load_run(path: Path):
    result = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row['condition_id']), str(row['entity_table']), str(row['query_entity_id']))
            if key in result:
                raise ValueError(f'{path}: duplicate condition/table/query {key}')
            pairs = row['pairs']
            scores = {}
            for pair in pairs:
                candidate = str(pair['candidate_entity_id'])
                if candidate in scores:
                    raise ValueError(f'{path}: duplicate candidate {candidate}')
                value = pair.get('similarity_score')
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                    raise ValueError(f'{path}: invalid score for {candidate}')
                scores[candidate] = value
            result[key] = (row, scores)
    return result


def compare_scores(left, right, k):
    if set(left) != set(right):
        raise ValueError('candidate universes differ; compare full rankings over the same entities')
    shared = [key for key in left if left[key] is not None and right[key] is not None]
    if not shared:
        return {'candidates': len(left), 'shared_scored': 0, 'spearman': None, 'top_k': k, 'left_top_size': 0, 'right_top_size': 0, 'top_intersection': 0, 'top_jaccard': None}
    a = pd.Series([left[key] for key in shared], dtype=float)
    b = pd.Series([right[key] for key in shared], dtype=float)
    ar, br = a.rank(method='average'), b.rank(method='average')
    corr = float(ar.corr(br)) if len(shared) > 1 and ar.nunique() > 1 and br.nunique() > 1 else None
    n = min(k, len(shared))
    # Include all scores tied at the kth position; top sets can exceed k.
    ta = float(a.nlargest(n).iloc[-1]); tb = float(b.nlargest(n).iloc[-1])
    atop = {key for key in shared if left[key] >= ta}
    btop = {key for key in shared if right[key] >= tb}
    return {'candidates': len(left), 'shared_scored': len(shared), 'spearman': corr, 'top_k': k, 'left_top_size': len(atop), 'right_top_size': len(btop), 'top_intersection': len(atop & btop), 'top_jaccard': len(atop & btop) / len(atop | btop)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, metavar='NAME=FILE')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.top_k < 1 or len(args.run) < 2:
            raise ValueError('provide at least two runs and a positive --top-k')
        runs = {}
        for spec in args.run:
            name, sep, filename = spec.partition('=')
            if not sep or not name or not filename or name in runs:
                raise ValueError(f'invalid or duplicate --run {spec!r}')
            runs[name] = load_run(Path(filename))
        rows = []
        for (name_a, run_a), (name_b, run_b) in itertools.combinations(runs.items(), 2):
            common = sorted(run_a.keys() & run_b.keys())
            if not common:
                raise ValueError(f'{name_a} and {name_b} have no matching condition/table/query')
            for key in common:
                left_row, left = run_a[key]; right_row, right = run_b[key]
                if left_row.get('condition') != right_row.get('condition'):
                    raise ValueError(f'different condition text for {key}')
                values = compare_scores(left, right, args.top_k)
                rows.append({'condition_id': key[0], 'entity_table': key[1], 'query_entity_id': key[2], 'condition': left_row.get('condition'), 'run_a': name_a, 'run_b': name_b, **values})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f'Wrote {len(rows)} matched query comparisons to {args.output}')
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
