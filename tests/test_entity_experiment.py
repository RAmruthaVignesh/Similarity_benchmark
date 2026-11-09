import pytest

from src.similarity.cohort import sample_cohort as cohort
from src.similarity.cohort import resolve_entity_ids
from src.similarity.discovery_selection import select_discovery_signals


def test_sample_is_reproducible_and_queries_are_in_pool():
    ids = list(range(100))
    sample = cohort(ids, "product", 42, 40, 10)
    assert sample == cohort(ids, "product", 42, 40, 10)
    assert len(sample["candidate_ids"]) == 40
    assert len(sample["query_ids"]) == 10
    assert set(sample["query_ids"]) <= set(sample["candidate_ids"])
    assert sample != cohort(ids, "product", 43, 40, 10)
    assert len(cohort(ids, "product", 42, 0, 0)["candidate_ids"]) == 100


def test_pool_resolves_original_types_and_rejects_invalid_ids():
    assert resolve_entity_ids(["3", 1], [1, 2, 3]) == [3, 1]
    with pytest.raises(ValueError, match="duplicate"):
        resolve_entity_ids(["1", 1], [1, 2, 3])
    with pytest.raises(ValueError, match="unknown"):
        resolve_entity_ids([4], [1, 2, 3])


def test_threshold_includes_every_tied_signal_on_requested_table():
    row = {"condition_id": "c", "score_type": "llm_pointwise_utility_0_10", "signals": [
        {"id": "high", "source_table": "product", "utility_score": 9},
        {"id": "first-five", "source_table": "product", "utility_score": 5},
        {"id": "other-table", "source_table": "review", "utility_score": 10},
        {"id": "second-five", "source_table": "product", "utility_score": 5},
        {"id": "below", "source_table": "product", "utility_score": 4},
    ]}
    selected = select_discovery_signals(row, "product", min_utility_score=5)
    assert [item["id"] for item in selected] == ["high", "first-five", "second-five"]
    with pytest.raises(ValueError, match="0–10"):
        select_discovery_signals({**row, "score_type": "llm_pointwise_utility_0_100"}, "product", min_utility_score=5)
    with pytest.raises(ValueError, match="no product signals"):
        select_discovery_signals(row, "product", min_utility_score=10)
