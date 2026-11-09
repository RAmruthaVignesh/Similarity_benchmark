from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.similarity.experimental_scoring import (
    _robust_center_scale, fit_profile, score_components, rescore_pairs,
)
from src.similarity.signal_measures import PairSimilarity


def meta(signals):
    return {
        "condition_id": "c1", "condition": "example", "entity_table": "product",
        "signals": [{"candidate": {"id": signal}, "value_kind": "numeric"} for signal in signals],
    }


def test_robust_z_and_training_median_missing_reference():
    train = pd.DataFrame([
        {"query": "train", "candidate": candidate, "signal": signal, "measure": "percentile_distance",
         "raw_score": score, "confidence": 1., "base_weight": 1.}
        for signal, scores in (("x", [.1, .2, .3]), ("y", [.1, .7, .9]))
        for candidate, score in zip(("a", "b", "c"), scores)
    ])
    test = pd.DataFrame([
        {"query": "heldout", "candidate": "z", "signal": "x", "measure": "percentile_distance",
         "raw_score": .2, "confidence": 1., "base_weight": 1.},
        {"query": "heldout", "candidate": "z", "signal": "y", "measure": "percentile_distance",
         "raw_score": np.nan, "confidence": 0., "base_weight": 1.},
    ])
    drop = fit_profile(train, meta(["x", "y"]), missing_policy="drop")
    reference = fit_profile(train, meta(["x", "y"]), missing_policy="reference")
    original, _ = score_components(test, drop)
    imputed, components = score_components(test, reference)
    assert original.iloc[0]["score"] == pytest.approx(.5)
    assert imputed.iloc[0]["score"] == pytest.approx(.5)
    assert imputed.iloc[0]["coverage"] == pytest.approx(.5)
    assert bool(components.loc[components.signal.eq("y"), "imputed"].iloc[0])
    penalized_profile = fit_profile(train, meta(["x", "y"]),
                                    missing_policy="reference", coverage_penalty=.2)
    penalized, _ = score_components(test, penalized_profile)
    assert penalized.iloc[0]["score"] == pytest.approx(.4)
    assert drop["training_queries"] == ["train"]


def test_discrete_scores_use_robust_scale_fallbacks_without_special_cap():
    train = pd.DataFrame([
        {"query": "q", "candidate": "a", "signal": "verified", "measure": "categorical_agreement",
         "raw_score": 1., "confidence": 1., "base_weight": 7.},
        {"query": "q", "candidate": "b", "signal": "verified", "measure": "categorical_agreement",
         "raw_score": 0., "confidence": 1., "base_weight": 3.},
    ])
    profile = fit_profile(train, meta(["verified"]))
    assert profile["signals"]["verified"]["weight"] == 1.0
    scores, components = score_components(train, profile)
    z = 1 / (1.4826 * .5)
    assert scores.loc[("q", "a"), "score"] == pytest.approx(.5 + z / (2 * (1 + abs(z))))
    assert scores.loc[("q", "b"), "score"] == pytest.approx(.5 - z / (2 * (1 + abs(z))))
    assert scores["score"].between(0, 1).all()
    assert profile["signals"]["verified"]["scale_method"] == "mad"
    assert components["experimental_weight"].eq(1.0).all()


def test_robust_z_profile_and_online_pair_scoring():
    train = pd.DataFrame([
        {"query": "train", "candidate": str(i), "signal": "x", "measure": "percentile_distance",
         "raw_score": i / 100, "confidence": 1., "base_weight": 1.}
        for i in range(100)
    ])
    profile = fit_profile(train, meta(["x"]))
    assert profile["signals"]["x"]["normalization"] == "bounded-robust-z"
    candidate = SimpleNamespace(id="x", hops=0)
    signal = SimpleNamespace(candidate=candidate, measure="percentile_distance")
    pair = PairSimilarity(query_entity_id="heldout", candidate_entity_id="a", similarity_score=.2,
                          components=({"signal_id": "x", "measure": "percentile_distance",
                                       "score": .2, "weight": 1., "confidence": 1.},))
    result = rescore_pairs([pair], profile, [signal])
    assert result[0]["coverage"] == 1
    assert np.isfinite(result[0]["similarity_score"])
    assert result[0]["original_similarity_score"] == .2
    assert not result[0]["components"][0]["experimental_imputed"]


def test_robust_z_is_fitted_from_training_scores_and_applied_to_heldout():
    train = pd.DataFrame([
        {"query": "train", "candidate": str(i), "signal": "price", "measure": "percentile_distance",
         "raw_score": i / 30, "confidence": 1., "base_weight": 1.}
        for i in range(1, 31)
    ])
    profile = fit_profile(train, meta(["price"]))
    heldout = train[train["candidate"].isin(["2", "25"])].copy()
    heldout["query"] = "heldout"
    result, _ = score_components(heldout, profile)
    assert result.loc[("heldout", "25"), "score"] > result.loc[("heldout", "2"), "score"]
    assert profile["training_queries"] == ["train"]


def test_drop_policy_coverage_counts_confidence_once():
    train = pd.DataFrame([
        {"query": "train", "candidate": "a", "signal": "x", "measure": "e5_embedding_cosine",
         "raw_score": .7, "confidence": .5, "base_weight": 1.},
    ])
    profile = fit_profile(train, meta(["x"]), missing_policy="drop")
    scores, _ = score_components(train, profile)
    assert scores.iloc[0]["score"] == pytest.approx(0.)
    assert scores.iloc[0]["coverage"] == pytest.approx(.5)


def test_bounded_robust_z_is_monotonic_and_bounded():
    from src.similarity.experimental_scoring import _bounded_robust_z

    values = np.array([.20, .21, .90])
    center, scale, _ = _robust_center_scale(values)
    transformed = _bounded_robust_z((values - center) / scale)
    assert np.all(np.diff(transformed) > 0)
    assert np.all((transformed >= 0) & (transformed <= 1))
    assert transformed[0] == pytest.approx(.5 + ((values[0] - center) / scale) /
                                             (2 * (1 + abs((values[0] - center) / scale))))
