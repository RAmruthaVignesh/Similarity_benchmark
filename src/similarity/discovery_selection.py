"""Select condition-relevant signals from a saved discovery run."""
from __future__ import annotations


def select_discovery_signals(record, table, *, top_signals=None, min_utility_score=None):
    if (top_signals is None) == (min_utility_score is None):
        raise ValueError("choose exactly one signal selection rule")
    raw = record.get("signals")
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        raise ValueError("discovery record must contain a list of signal objects")
    matching = [item for item in raw if item.get("source_table") == table]
    if min_utility_score is not None:
        if not isinstance(min_utility_score, int) or isinstance(min_utility_score, bool) or not 0 <= min_utility_score <= 10:
            raise ValueError("minimum utility score must be an integer from 0 to 10")
        if record.get("score_type") != "llm_pointwise_utility_0_10":
            raise ValueError("utility threshold selection requires a 0–10 LLM utility discovery run")
        for item in matching:
            score = item.get("utility_score")
            if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 10:
                raise ValueError(f"signal {item.get('id')!r} has a missing or invalid 0–10 utility score")
        selected = [item for item in matching if item["utility_score"] >= min_utility_score]
    else:
        if not isinstance(top_signals, int) or isinstance(top_signals, bool) or top_signals < 1:
            raise ValueError("top-signals must be positive")
        selected = matching[:top_signals]
    if not selected:
        raise ValueError(f"no {table} signals meet the requested selection rule for {record.get('condition_id')}")
    return selected
