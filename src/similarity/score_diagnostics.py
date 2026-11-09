"""Ranking diagnostics shared by the experimental similarity evaluator."""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_queries(data: pd.DataFrame, seed: int = 42):
    queries = np.array(sorted(data["query"].unique()))
    if len(queries) < 2:
        raise ValueError("At least two reference queries are needed for a train/test split.")
    shuffled = np.random.default_rng(seed).permutation(queries)
    n_train = max(1, len(queries) // 2)
    return set(shuffled[:n_train]), set(shuffled[n_train:])


def tie_top_membership(scores: pd.Series, k: int):
    """Assign fractional top-k membership to scores tied at the boundary."""
    scores = scores.dropna()
    if scores.empty:
        return scores
    n = min(k, len(scores))
    cutoff = scores.nlargest(n).iloc[-1]
    above = scores > cutoff
    tied = scores == cutoff
    result = pd.Series(0.0, index=scores.index)
    result.loc[above] = 1.0
    result.loc[tied] = (n - int(above.sum())) / int(tied.sum())
    return result


def ranking_comparison(baseline: pd.Series, alternative: pd.Series, k: int = 10):
    """Compare common scored candidates with average ranks and inclusive cutoff ties."""
    both = pd.concat([baseline.rename("base"), alternative.rename("other")], axis=1)
    rows = []
    for query, group in both.groupby(level="query", sort=False):
        valid = group.dropna()
        n = len(valid)
        if not n:
            rows.append({"query": query, "coverage": 0.0, "spearman": np.nan, "top_jaccard": np.nan})
            continue
        left, right = valid["base"], valid["other"]
        ar, br = left.rank(method="average"), right.rank(method="average")
        rho = ar.corr(br) if n > 1 and ar.nunique() > 1 and br.nunique() > 1 else np.nan
        size = min(k, n)
        aset = set(left[left >= left.nlargest(size).iloc[-1]].index)
        bset = set(right[right >= right.nlargest(size).iloc[-1]].index)
        rows.append({"query": query, "coverage": n / len(group), "spearman": rho,
                     "top_jaccard": len(aset & bset) / len(aset | bset)})
    return pd.DataFrame(rows)
