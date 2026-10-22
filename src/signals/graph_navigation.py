"""Decider-guided, schema-constrained graph discovery.

This module implements the graph-navigation variant of signal discovery.  In
contrast to scoring one exhaustive candidate list, Decider repeatedly chooses
from a *legal* action menu: start at a table, select a reachable attribute or
relationship as a signal, traverse one foreign-key edge, or stop a branch.

The program owns graph state and executes every choice, so a model cannot
invent tables, columns, joins, or malformed paths.  Each decision includes the
original intent, full schema, current path, and prior branch decisions.  The
returned trace is deliberately verbose enough to audit and reproduce a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from math import log
from typing import Any, Mapping, Sequence

from src.schema.schema_graph import Edge, SchemaGraph
from src.signals.candidates import SignalCandidate
from src.signals.scorers import create_choice_decider_client, default_decider_model

__all__ = [
    "DeciderGraphNavigator",
    "GraphDiscoveryResult",
    "format_explored_schema_view",
    "format_search_history",
]


@dataclass(frozen=True)
class _Action:
    """One program-generated, valid Decider action."""

    kind: str
    option: str
    candidate: SignalCandidate | None = None
    target_table: str | None = None
    edge: Edge | None = None

    def to_dict(self, graph: SchemaGraph, probability: float) -> dict[str, object]:
        record: dict[str, object] = {
            "kind": self.kind,
            "option": self.option,
            "probability": probability,
        }
        if self.candidate is not None:
            record["candidate_id"] = self.candidate.id
            record["candidate"] = self.candidate.to_dict(graph)
        if self.target_table is not None:
            record["target_table"] = self.target_table
        if self.edge is not None:
            record["edge"] = self.edge.to_dict()
        return record


@dataclass(frozen=True)
class _State:
    """A single branch in the bounded graph search."""

    branch_id: int
    parent_branch_id: int | None
    root_table: str
    path: tuple[str, ...]
    selected_signal_ids: frozenset[str] = field(default_factory=frozenset)
    history: tuple[str, ...] = ()
    log_probability: float = 0.0


@dataclass(frozen=True)
class GraphDiscoveryResult:
    """Signals and the complete decision trace for one condition."""

    signals: tuple[SignalCandidate, ...]
    trace: tuple[dict[str, object], ...]
    model_calls: int
    stop_reason: str

    def to_dict(self, graph: SchemaGraph) -> dict[str, object]:
        return {
            "signals": [candidate.to_dict(graph) for candidate in self.signals],
            "search": {
                "model_calls": self.model_calls,
                "stop_reason": self.stop_reason,
                "trace": list(self.trace),
            },
        }


class DeciderGraphNavigator:
    """Discover schema signals by executing bounded Decider graph decisions.

    ``branch_width`` retains the most promising legal actions at every
    decision, including the initial table choice. The model-selected search
    can therefore explore multiple root tables. ``max_decisions`` is a strict
    cap on model calls, including that initial table-selection call.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        backend: str = "strands",
        device: str = "auto",
        max_hops: int = 2,
        max_decisions: int = 12,
        branch_width: int = 2,
        max_signals: int = 10,
        include_relationships: bool = True,
        client: Any | None = None,
    ) -> None:
        if max_hops < 0:
            raise ValueError("max_hops must be non-negative")
        if max_decisions <= 0:
            raise ValueError("max_decisions must be positive")
        if branch_width <= 0:
            raise ValueError("branch_width must be positive")
        if max_signals <= 0:
            raise ValueError("max_signals must be positive")
        self.backend = backend
        self.model_name = model_name or default_decider_model(backend)
        self.device = device
        self.max_hops = max_hops
        self.max_decisions = max_decisions
        self.branch_width = branch_width
        self.max_signals = max_signals
        self.include_relationships = include_relationships
        self._client = client

    def discover(
        self,
        intent: str,
        candidates: Sequence[SignalCandidate],
        graph: SchemaGraph,
        *,
        start_table: str | None = None,
    ) -> GraphDiscoveryResult:
        """Return signals discovered by a constrained, stateful graph search.

        ``start_table`` is an oracle-root ablation for controlled tasks that
        provide a known source table. Its root choice is forced rather than
        queried, and one decision from ``max_decisions`` is reserved for that
        root choice so the remaining search has the same decision budget as
        ordinary model-selected-root search.
        """
        catalog = _CandidateCatalog(candidates)
        catalog.require_compatible(graph, self.max_hops, self.include_relationships)
        if not graph.get_tables():
            raise ValueError("graph discovery requires a schema with at least one table")
        table_names = {table.name for table in graph.get_tables()}
        if start_table is not None and start_table not in table_names:
            raise ValueError(f"unknown graph-search start table: {start_table!r}")
        if start_table is not None and self.max_decisions < 2:
            raise ValueError("oracle-root search requires max_decisions of at least 2")
        client = self._get_client()
        trace: list[dict[str, object]] = []
        discovered: dict[str, SignalCandidate] = {}
        model_calls = 0
        next_branch_id = 0

        frontier: list[tuple[float, int, _State]] = []
        stop_reason = "frontier_exhausted"
        decision_budget = self.max_decisions
        if start_table is not None:
            decision_budget -= 1  # Match normal search, which spends one call selecting its root.
            state = _State(
                branch_id=1,
                parent_branch_id=None,
                root_table=start_table,
                path=(start_table,),
                history=(f"Oracle-forced start table: {start_table}",),
            )
            heapq.heappush(frontier, (-state.log_probability, state.branch_id, state))
            next_branch_id = 1
        else:
            root_actions = tuple(
                _Action("start", f"Start from table: {table.name}", target_table=table.name)
                for table in graph.get_tables()
            ) + (_Action("stop_search", "Finish discovery without selecting a start table."),)
            root_context = _format_navigation_context(
                intent,
                graph,
                candidates,
                current_path=None,
                history=(),
                discovered=(),
                remaining_hops=self.max_hops,
                remaining_signal_slots=self.max_signals,
            )
            root_answer, root_probabilities = self._decide(
                client,
                root_context,
                "Which table is the best starting point for discovering useful schema signals?",
                root_actions,
            )
            model_calls += 1
            root_selected = _top_actions(
                root_actions,
                root_probabilities,
                self.branch_width,
                chosen_option=root_answer,
            )
            trace.append(
                _trace_entry(
                    model_calls,
                    None,
                    None,
                    None,
                    root_context,
                    root_answer,
                    root_actions,
                    root_probabilities,
                    root_selected,
                    graph,
                )
            )

            for action, probability in root_selected:
                if action.kind == "stop_search":
                    continue
                assert action.target_table is not None
                next_branch_id += 1
                state = _State(
                    branch_id=next_branch_id,
                    parent_branch_id=None,
                    root_table=action.target_table,
                    path=(action.target_table,),
                    history=(action.option,),
                    log_probability=_log_probability(probability),
                )
                heapq.heappush(frontier, (-state.log_probability, state.branch_id, state))

        expanded_states: set[tuple[tuple[str, ...], frozenset[str]]] = set()
        if start_table is None and root_answer == "Finish discovery without selecting a start table.":
            return GraphDiscoveryResult((), tuple(trace), model_calls, "model_stopped_search")
        while frontier and model_calls < decision_budget and len(discovered) < self.max_signals:
            _, _, state = heapq.heappop(frontier)
            signature = (state.path, state.selected_signal_ids)
            if signature in expanded_states:
                continue
            expanded_states.add(signature)

            actions = self._legal_actions(state, catalog, graph, discovered)
            context = _format_navigation_context(
                intent,
                graph,
                candidates,
                current_path=state.path,
                history=state.history,
                discovered=tuple(discovered.values()),
                remaining_hops=self.max_hops - (len(state.path) - 1),
                remaining_signal_slots=self.max_signals - len(discovered),
            )
            answer, probabilities = self._decide(
                client,
                context,
                "Which single legal next action best advances discovery of useful signals?",
                actions,
            )
            model_calls += 1
            selected = _top_actions(
                actions, probabilities, self.branch_width, chosen_option=answer
            )
            trace.append(
                _trace_entry(
                    model_calls,
                    state.branch_id,
                    state.parent_branch_id,
                    state.path,
                    context,
                    answer,
                    actions,
                    probabilities,
                    selected,
                    graph,
                )
            )

            for action, probability in selected:
                if action.kind == "stop_search":
                    if action.option == answer:
                        stop_reason = "model_stopped_search"
                        frontier.clear()
                        break
                    continue
                if action.kind == "stop_branch":
                    continue
                selected_ids = state.selected_signal_ids
                if action.candidate is not None:
                    discovered.setdefault(action.candidate.id, action.candidate)
                    selected_ids = frozenset((*selected_ids, action.candidate.id))
                    if len(discovered) >= self.max_signals:
                        stop_reason = "max_signals_reached"
                        break

                next_path = state.path
                if action.kind == "traverse":
                    assert action.target_table is not None
                    next_path = (*state.path, action.target_table)

                next_branch_id += 1
                child = _State(
                    branch_id=next_branch_id,
                    parent_branch_id=state.branch_id,
                    root_table=state.root_table,
                    path=next_path,
                    selected_signal_ids=selected_ids,
                    history=(*state.history, action.option),
                    log_probability=state.log_probability + _log_probability(probability),
                )
                heapq.heappush(frontier, (-child.log_probability, child.branch_id, child))

        if len(discovered) >= self.max_signals:
            stop_reason = "max_signals_reached"
        elif model_calls >= decision_budget and frontier:
            stop_reason = "max_decisions_reached"
        return GraphDiscoveryResult(
            signals=tuple(discovered.values()),
            trace=tuple(trace),
            model_calls=model_calls,
            stop_reason=stop_reason,
        )
    def _legal_actions(
        self,
        state: _State,
        catalog: _CandidateCatalog,
        graph: SchemaGraph,
        discovered: Mapping[str, SignalCandidate],
    ) -> tuple[_Action, ...]:
        actions: list[_Action] = []
        for candidate in catalog.attributes(state.root_table, state.path):
            if candidate.id not in discovered:
                assert candidate.attribute_table is not None
                assert candidate.attribute_column is not None
                actions.append(
                    _Action(
                        "select_attribute",
                        "Select attribute: "
                        f"{candidate.attribute_table}.{candidate.attribute_column} "
                        f"(path: {_describe_path(state.path, graph)})",
                        candidate=candidate,
                    )
                )
        if self.include_relationships and len(state.path) > 1:
            relationship = catalog.relationship(state.root_table, state.path)
            if relationship is not None and relationship.id not in discovered:
                actions.append(
                    _Action(
                        "select_relationship",
                        f"Select relationship path: {_describe_path(state.path, graph)}",
                        candidate=relationship,
                    )
                )
        if len(state.path) - 1 < self.max_hops:
            for target, edge in _available_steps(graph, state.path):
                next_path = (*state.path, target)
                # In the feature-collapsed catalog there are deliberately no
                # path-specific candidates beyond a root. Do not spend a Decider
                # call exploring a branch which cannot emit any signal.
                if catalog.has_content_below(state.root_table, next_path):
                    actions.append(
                        _Action(
                            "traverse",
                            f"Traverse relationship: {_describe_step(state.path[-1], target, edge)}",
                            target_table=target,
                            edge=edge,
                        )
                    )
        actions.extend(
            (
                _Action(
                    "stop_branch",
                    "Stop this path; continue any other queued paths.",
                ),
                _Action(
                    "stop_search",
                    "Finish discovery and return the signals collected so far.",
                ),
            )
        )
        _require_unique_options(actions)
        return tuple(actions)

    def _decide(
        self,
        client: Any,
        context: str,
        question: str,
        actions: Sequence[_Action],
    ) -> tuple[str | None, dict[str, float]]:
        if not 2 <= len(actions) <= 255:
            raise ValueError(
                "Decider multi-choice decisions require 2 to 255 legal actions; "
                f"got {len(actions)}"
            )
        answer = _answers(
            client.decide(context, [{"question": question, "options": [a.option for a in actions]}])
        )[0]
        choice = _choice(answer)
        options = {action.option for action in actions}
        if choice is not None and choice not in options:
            raise RuntimeError(
                "Decider selected an action outside the program-generated legal menu: "
                f"{choice!r}"
            )
        return choice, _probabilities(answer)

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = create_choice_decider_client(
                self.backend, self.model_name, device=self.device
            )
        return self._client


def format_search_history(
    result: GraphDiscoveryResult,
    graph: SchemaGraph,
    *,
    forced_root: str | None = None,
) -> str:
    """Render explored actions for the final shortlist-ranking prompt.

    Deliberately excludes each trace event's ``state_context``: that contains the
    full schema, while this final Decider pass should be conditioned on what the
    graph search actually explored.
    """
    lines: list[str] = []
    if forced_root is not None:
        lines.append(f"Oracle-forced starting table: {forced_root}")
    for event in result.trace:
        decision_index = int(event["decision_index"])
        path = event.get("current_path")
        location = (
            "starting-table selection"
            if path is None
            else "path " + " → ".join(str(table) for table in path)
        )
        choice = event.get("model_choice") or "no explicit choice"
        lines.append(f"Decision {decision_index} ({location}): Decider chose: {choice}")
        for action in event.get("expanded_actions", []):
            probability = float(action.get("probability", 0.0))
            lines.append(
                f"  explored: {action.get('option')} "
                f"(choice probability {probability:.4f})"
            )
    if result.signals:
        lines.append("Signals collected:")
        lines.extend(f"- {candidate.describe(graph)}" for candidate in result.signals)
    else:
        lines.append("The graph search collected no signals.")
    return "\n".join(lines)


def format_explored_schema_view(
    result: GraphDiscoveryResult,
    graph: SchemaGraph,
    *,
    forced_root: str | None = None,
) -> str:
    """Summarize schema parts exposed during search without model choices/probabilities."""
    root_tables: set[str] = set()
    expanded_root_tables: set[str] = set()
    visited_tables: set[str] = set()
    relationships: set[tuple[str, str, str, str]] = set()
    traversed_relationships: set[tuple[str, str, str, str]] = set()
    attributes: set[tuple[tuple[str, ...], str, str]] = set()

    for event in result.trace:
        path = event.get("current_path")
        if isinstance(path, list) and all(isinstance(table, str) for table in path):
            visited_tables.update(path)

        actions = event.get("legal_actions", [])
        if isinstance(actions, list):
            for action in actions:
                if not isinstance(action, dict):
                    continue
                kind = action.get("kind")
                target_table = action.get("target_table")
                if kind == "start" and isinstance(target_table, str):
                    root_tables.add(target_table)
                if kind == "traverse":
                    edge = action.get("edge")
                    if isinstance(edge, dict) and all(
                        isinstance(edge.get(field), str)
                        for field in (
                            "source_table", "source_column",
                            "target_table", "target_column",
                        )
                    ):
                        relationships.add((
                            edge["source_table"], edge["source_column"],
                            edge["target_table"], edge["target_column"],
                        ))
                if kind == "select_attribute":
                    candidate = action.get("candidate")
                    if not isinstance(candidate, dict):
                        continue
                    candidate_path = candidate.get("path")
                    table = candidate.get("attribute_table")
                    column = candidate.get("attribute_column")
                    if (
                        isinstance(candidate_path, list)
                        and all(isinstance(item, str) for item in candidate_path)
                        and isinstance(table, str)
                        and isinstance(column, str)
                    ):
                        attributes.add((tuple(candidate_path), table, column))

        expanded = event.get("expanded_actions", [])
        if isinstance(expanded, list):
            for action in expanded:
                if not isinstance(action, dict):
                    continue
                kind = action.get("kind")
                target_table = action.get("target_table")
                if kind == "start" and isinstance(target_table, str):
                    expanded_root_tables.add(target_table)
                if kind == "traverse":
                    edge = action.get("edge")
                    if isinstance(edge, dict) and all(
                        isinstance(edge.get(field), str)
                        for field in (
                            "source_table", "source_column",
                            "target_table", "target_column",
                        )
                    ):
                        traversed_relationships.add((
                            edge["source_table"], edge["source_column"],
                            edge["target_table"], edge["target_column"],
                        ))

    lines = ["Schema context seen by graph search:"]
    if forced_root is not None:
        lines.append(f"Starting table used: {forced_root}")
    elif root_tables:
        lines.append("Starting table options offered: " + ", ".join(sorted(root_tables)))
        if expanded_root_tables:
            lines.append(
                "Starting table branches explored: "
                + ", ".join(sorted(expanded_root_tables))
            )
    lines.append(
        "Tables visited: " + (", ".join(sorted(visited_tables)) if visited_tables else "none")
    )
    lines.append("Foreign-key relationships offered at visited states:")
    if relationships:
        lines.extend(
            f"- {source_table}.{source_column} references {target_table}.{target_column}"
            for source_table, source_column, target_table, target_column in sorted(relationships)
        )
    else:
        lines.append("- none")
    lines.append("Foreign-key relationships traversed by the search:")
    if traversed_relationships:
        lines.extend(
            f"- {source_table}.{source_column} references {target_table}.{target_column}"
            for source_table, source_column, target_table, target_column
            in sorted(traversed_relationships)
        )
    else:
        lines.append("- none")
    lines.append("Attribute choices shown at visited search states:")
    if attributes:
        lines.extend(
            f"- {' → '.join(path)}: {table}.{column}"
            for path, table, column in sorted(attributes)
        )
    else:
        lines.append("- none")
    lines.append(
        "The listed attributes are schema context only. Each offered signal contains "
        "only its own displayed path and explicitly named endpoint."
    )
    return "\n".join(lines)


class _CandidateCatalog:
    """Index existing path-aware candidates without altering their semantics."""

    def __init__(self, candidates: Sequence[SignalCandidate]) -> None:
        self._attributes: dict[tuple[str, tuple[str, ...]], list[SignalCandidate]] = {}
        self._relationships: dict[tuple[str, tuple[str, ...]], SignalCandidate] = {}
        for candidate in candidates:
            key = (candidate.source_table, candidate.path)
            if candidate.kind == "attribute":
                self._attributes.setdefault(key, []).append(candidate)
            elif candidate.kind == "relationship":
                self._relationships[key] = candidate

    def require_compatible(
        self,
        graph: SchemaGraph,
        max_hops: int,
        include_relationships: bool,
    ) -> None:
        for table in graph.get_tables():
            if (table.name, (table.name,)) not in self._attributes:
                raise ValueError(
                    "graph discovery requires candidates for every schema table; "
                    "enumerate candidates without restricting --source-table"
                )
        if max_hops and include_relationships and not self._relationships:
            raise ValueError(
                "graph discovery with relationship selection requires relationship "
                "candidates; do not pass --no-relationships"
            )

    def attributes(
        self, source_table: str, path: tuple[str, ...]
    ) -> tuple[SignalCandidate, ...]:
        return tuple(self._attributes.get((source_table, path), ()))

    def relationship(
        self, source_table: str, path: tuple[str, ...]
    ) -> SignalCandidate | None:
        return self._relationships.get((source_table, path))

    def has_content_below(self, source_table: str, path: tuple[str, ...]) -> bool:
        """Whether this path can lead to an uncollapsed candidate.

        This prevents feature-collapsed graph discovery from traversing beyond
        a root when the catalog intentionally contains direct attributes only.
        """
        for root, candidate_path in (*self._attributes, *self._relationships):
            if (
                root == source_table
                and len(candidate_path) >= len(path)
                and candidate_path[: len(path)] == path
            ):
                return True
        return False


def _top_actions(
    actions: Sequence[_Action],
    probabilities: Mapping[str, float],
    count: int,
    *,
    chosen_option: str | None,
) -> tuple[tuple[_Action, float], ...]:
    ranked = sorted(
        ((action, max(0.0, probabilities.get(action.option, 0.0)), index) for index, action in enumerate(actions)),
        key=lambda row: (-row[1], row[2]),
    )
    if chosen_option is None:
        return tuple((action, probability) for action, probability, _ in ranked[:count])
    chosen = next(action for action in actions if action.option == chosen_option)
    chosen_probability = max(0.0, probabilities.get(chosen.option, 0.0))
    remainder = [
        (action, probability)
        for action, probability, _ in ranked
        if action.option != chosen_option
    ]
    return tuple([(chosen, chosen_probability), *remainder[: count - 1]])


def _trace_entry(
    decision_index: int,
    branch_id: int | None,
    parent_branch_id: int | None,
    path: tuple[str, ...] | None,
    context: str,
    model_choice: str | None,
    actions: Sequence[_Action],
    probabilities: Mapping[str, float],
    expanded: Sequence[tuple[_Action, float]],
    graph: SchemaGraph,
) -> dict[str, object]:
    return {
        "decision_index": decision_index,
        "branch_id": branch_id,
        "parent_branch_id": parent_branch_id,
        "current_path": None if path is None else list(path),
        "state_context": context,
        "model_choice": model_choice,
        "legal_actions": [
            action.to_dict(graph, max(0.0, probabilities.get(action.option, 0.0)))
            for action in actions
        ],
        "expanded_actions": [action.to_dict(graph, probability) for action, probability in expanded],
    }


def _format_navigation_context(
    intent: str,
    graph: SchemaGraph,
    candidates: Sequence[SignalCandidate],
    *,
    current_path: tuple[str, ...] | None,
    history: Sequence[str],
    discovered: Sequence[SignalCandidate],
    remaining_hops: int,
    remaining_signal_slots: int,
) -> str:
    lines = [
        "You are navigating a relational database schema to discover useful signals "
        "for a retrieval or ranking task.",
        "Choose only one offered action. Do not invent tables, attributes, joins, or paths.",
        "A relationship-only signal represents the association along its path; "
        "it does not bundle attributes from reached tables. Select a separate "
        "attribute action when that attribute is the intended evidence.",
        "The program validates and executes your choice before the next decision.",
        "",
        f"User intent: {intent}",
        "",
        f"Schema:\n{_describe_schema(candidates, graph)}",
        "",
        "Search state:",
        "- Current path: " + ("not selected yet" if current_path is None else _describe_path(current_path, graph)),
        "- Prior decisions on this branch: " + ("none" if not history else " | ".join(history)),
        "- Signals already selected across explored branches: "
        + (
            "none"
            if not discovered
            else "; ".join(candidate.describe(graph) for candidate in discovered)
        ),
        f"- Remaining join hops: {remaining_hops}",
        f"- Remaining signal slots: {remaining_signal_slots}",
    ]
    return "\n".join(lines)


def _describe_schema(candidates: Sequence[SignalCandidate], graph: SchemaGraph) -> str:
    attributes: dict[str, set[str]] = {table.name: set() for table in graph.get_tables()}
    for candidate in candidates:
        if candidate.kind == "attribute":
            assert candidate.attribute_table is not None and candidate.attribute_column is not None
            attributes[candidate.attribute_table].add(candidate.attribute_column)
    tables = [
        f"- {table.name}: {', '.join(sorted(attributes[table.name])) or '(no selectable attributes)'}"
        for table in graph.get_tables()
    ]
    relationships = [
        f"- {edge.source_table}.{edge.source_column} references "
        f"{edge.target_table}.{edge.target_column}"
        for edge in graph.get_edges()
    ]
    return "\n".join(["Tables and attributes:", *tables, "Relationships:", *(relationships or ["- none declared"])])


def _available_steps(graph: SchemaGraph, path: tuple[str, ...]) -> tuple[tuple[str, Edge], ...]:
    steps: list[tuple[str, Edge]] = []
    current = path[-1]
    for neighbor in graph.get_neighbors(current):
        if neighbor in path:
            continue
        edges = graph.get_relationships(current, neighbor)
        if len(edges) != 1:
            raise ValueError(
                f"graph traversal between {current!r} and {neighbor!r} is ambiguous; "
                "multiple foreign keys require edge-aware path candidates"
            )
        steps.append((neighbor, edges[0]))
    return tuple(steps)


def _describe_path(path: tuple[str, ...], graph: SchemaGraph) -> str:
    if len(path) == 1:
        return path[0]
    parts = [path[0]]
    for source, target in zip(path, path[1:]):
        edge = graph.get_relationships(source, target)
        if len(edge) != 1:
            raise ValueError(f"ambiguous path step {source!r} -> {target!r}")
        parts.append(_describe_step(source, target, edge[0]))
    return " ".join(parts)


def _describe_step(source: str, target: str, edge: Edge) -> str:
    if edge.source_table == source:
        return f"--{edge.source_table}.{edge.source_column}→{edge.target_table}.{edge.target_column}-- {target}"
    return f"--{edge.target_table}.{edge.target_column}←{edge.source_table}.{edge.source_column}-- {target}"


def _require_unique_options(actions: Sequence[_Action]) -> None:
    options = [action.option for action in actions]
    if len(options) != len(set(options)):
        raise ValueError("graph navigation generated duplicate Decider action labels")


def _log_probability(probability: float) -> float:
    return log(max(probability, 1e-12))


def _answers(result: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(result, Mapping):
        result = result.get("answers", result)
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
        raise RuntimeError(f"unexpected Decider response: {result!r}")
    if not all(isinstance(answer, Mapping) for answer in result):
        raise RuntimeError(f"unexpected Decider answers: {result!r}")
    return result


def _probabilities(answer: Mapping[str, Any]) -> dict[str, float]:
    raw = answer.get("probs", answer.get("probabilities"))
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"Decider answer has no option probabilities: {answer!r}")
    return {str(key): float(value) for key, value in raw.items()}


def _choice(answer: Mapping[str, Any]) -> str | None:
    choice = answer.get("choice")
    return str(choice) if choice is not None else None
