#!/usr/bin/env python3
"""Serve a small dependency-free browser UI for signal-based entity similarity."""

from __future__ import annotations

import argparse
import errno
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.local_relbench_dataset import LocalRelBenchDataset  # noqa: E402
from src.schema.schema_graph import SchemaGraph  # noqa: E402
from src.signals.candidates import SignalCandidate, candidate_from_dict  # noqa: E402
from src.similarity.signal_measures import MaterializedSignal, SignalMeasureExecutor  # noqa: E402

DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "dev" / "rel-amazon"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "rel-amazon" / "discovery"
DEFAULT_CONDITIONS_ROOT = REPO_ROOT / "conditions" / "rel-amazon"


class SimilarityExplorer:
    """Read-only backend for the signal-similarity explorer."""

    def __init__(
        self,
        dataset_root: Path,
        results_root: Path,
        conditions_root: Path,
        *,
        model_cache_root: Path | None = None,
    ) -> None:
        self.dataset = LocalRelBenchDataset(dataset_root)
        self.graph = SchemaGraph.from_manifest(self.dataset.schema)
        self.executor = SignalMeasureExecutor(
            self.dataset,
            self.graph,
            model_cache_root=model_cache_root,
        )
        self.results_root = results_root
        self.conditions_root = conditions_root
        self._jsonl_cache: dict[Path, list[dict[str, object]]] = {}
        self._materialized_cache: dict[str, MaterializedSignal] = {}

    def options(
        self, filename: str | None, condition_id: str | None, entity_table: str | None
    ) -> dict[str, object]:
        files = self.run_files()
        filename = filename if filename in files else (files[0] if files else None)
        conditions = self.conditions()
        if filename:
            available = {row.get("condition_id") for row in self._load_jsonl(self._run_dir() / filename)}
            conditions = {key: value for key, value in conditions.items() if key in available}
        condition_id = condition_id if condition_id in conditions else next(iter(conditions), None)
        record = self.result_record(filename, condition_id)
        source_tables = sorted(
            {
                candidate_from_dict(signal).source_table
                for signal in record.get("signals", [])
                if isinstance(signal, dict)
            }
        ) if record else []
        entity_table = entity_table if entity_table in source_tables else (source_tables[0] if source_tables else None)
        signals = self.compatible_signals(record, entity_table) if record and entity_table else []
        ids, id_kind = self.entity_ids(entity_table) if entity_table else ((), None)
        return {
            "runs": [{"filename": name, "label": _run_label(name)} for name in files],
            "conditions": [{"id": key, "condition": value["condition"]} for key, value in conditions.items()],
            "selected": {"filename": filename, "condition_id": condition_id, "entity_table": entity_table},
            "entity_tables": source_tables,
            "entity_ids": [str(value) for value in ids],
            "entity_id_kind": id_kind,
            "signals": [
                {
                    "id": candidate.id,
                    "rank": stored.get("rank"),
                    "label": candidate.explain_for_human(self.graph),
                }
                for candidate, stored in signals
            ],
        }

    def score(self, payload: dict[str, object]) -> dict[str, object]:
        filename = _required_string(payload, "filename")
        condition_id = _required_string(payload, "condition_id")
        entity_table = _required_string(payload, "entity_table")
        query_id = _required_string(payload, "query_id")
        selected_ids = payload.get("signal_ids")
        if not isinstance(selected_ids, list) or not selected_ids or not all(isinstance(item, str) for item in selected_ids):
            raise ValueError("select at least one signal")
        top_k = payload.get("top_k", 20)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer from 1 to 100")
        raw_weights = payload.get("weights", {})
        if not isinstance(raw_weights, dict):
            raise ValueError("weights must be a JSON object")
        weights = {str(key): _nonnegative_number(value) for key, value in raw_weights.items()}

        record = self.result_record(filename, condition_id)
        if record is None:
            raise ValueError("the selected run has no result for this condition")
        available = {candidate.id: (candidate, stored) for candidate, stored in self.compatible_signals(record, entity_table)}
        missing = [signal_id for signal_id in selected_ids if signal_id not in available]
        if missing:
            raise ValueError(f"signals do not start from {entity_table}: {', '.join(missing)}")
        if set(weights) - set(selected_ids):
            raise ValueError("weights may refer only to selected signals")
        signals = [self.materialize(available[signal_id][0]) for signal_id in selected_ids]
        pairs = self.executor.score_query(query_id, signals, weights=weights)[:top_k]
        resolved_query_id = pairs[0].query_entity_id if pairs else query_id
        return {
            "condition_id": condition_id,
            "condition": self.conditions()[condition_id]["condition"],
            "entity_table": entity_table,
            "query_entity_id": resolved_query_id,
            "query_record": self.entity_row(entity_table, resolved_query_id),
            "signals": [
                {
                    "id": signal.candidate.id,
                    "rank": available[signal.candidate.id][1].get("rank"),
                    "meaning": signal.candidate.explain_for_human(self.graph),
                    "measure": signal.measure,
                    "value_kind": signal.value_kind,
                    "weight": weights.get(signal.candidate.id, 1.0),
                    "query_values": list(signal.values_for(resolved_query_id)),
                }
                for signal in signals
            ],
            "pairs": [
                {
                    **pair.to_dict(),
                    "candidate_record": self.entity_row(entity_table, pair.candidate_entity_id),
                }
                for pair in pairs
            ],
        }

    def run_files(self) -> list[str]:
        return sorted(path.name for path in self._run_dir().glob("*.jsonl"))

    def conditions(self) -> dict[str, dict[str, str]]:
        return {
            row["id"]: {"id": row["id"], "condition": row["condition"]}
            for row in self._load_jsonl(self.conditions_root / "conditions.jsonl")
            if isinstance(row.get("id"), str) and isinstance(row.get("condition"), str)
        }

    def result_record(self, filename: str | None, condition_id: str | None) -> dict[str, object] | None:
        if not filename or not condition_id:
            return None
        path = self._run_dir() / filename
        if path.name != filename or not path.is_file():
            return None
        return next((row for row in self._load_jsonl(path) if row.get("condition_id") == condition_id), None)

    def compatible_signals(self, record: dict[str, object], entity_table: str) -> list[tuple[SignalCandidate, dict[str, object]]]:
        signals = []
        for stored in record.get("signals", []):
            if not isinstance(stored, dict):
                continue
            candidate = candidate_from_dict(stored)
            if candidate.source_table == entity_table:
                signals.append((candidate, stored))
        return signals

    def materialize(self, candidate: SignalCandidate) -> MaterializedSignal:
        if candidate.id not in self._materialized_cache:
            self._materialized_cache[candidate.id] = self.executor.materialize(candidate)
        return self._materialized_cache[candidate.id]

    def entity_ids(self, table: str) -> tuple[tuple[object, ...], str]:
        primary_key = self.dataset.get_primary_key(table)
        if primary_key is None:
            return tuple(range(self.dataset.get_table(table).num_rows)), "row position"
        return tuple(self.dataset.get_table(table)[primary_key].to_pylist()), "primary key"

    def entity_row(self, table: str, entity_id: object) -> dict[str, object]:
        primary_key = self.dataset.get_primary_key(table)
        if primary_key is None:
            return self.dataset.get_table(table).slice(int(entity_id), 1).to_pylist()[0]
        return self.dataset.get_entity(table, entity_id).to_pylist()[0]

    def _run_dir(self) -> Path:
        return self.results_root / "catalog-views"

    def _load_jsonl(self, path: Path) -> list[dict[str, object]]:
        if path not in self._jsonl_cache:
            with path.open() as handle:
                self._jsonl_cache[path] = [json.loads(line) for line in handle if line.strip()]
        return self._jsonl_cache[path]

class ExplorerHandler(BaseHTTPRequestHandler):
    explorer: SimilarityExplorer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_html(_HTML)
            elif parsed.path == "/api/options":
                query = parse_qs(parsed.query)
                self._send_json(self.explorer.options(
                    _query_value(query, "filename"),
                    _query_value(query, "condition_id"), _query_value(query, "entity_table"),
                ))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (OSError, ValueError, KeyError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/score":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            self._send_json(self.explorer.score(body))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, _format: str, *_args: object) -> None:
        pass

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, text: str) -> None:
        body = text.encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _query_value(query: dict[str, list[str]], key: str, default: str | None = None) -> str | None:
    return query.get(key, [default])[0]


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} is required")
    return value


def _nonnegative_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("weights must be non-negative numbers")
    return float(value)


def _run_label(filename: str) -> str:
    return filename.removesuffix(".jsonl").replace("catalog-rank-", "Catalog rank — ").replace("-path-aware", " [path-aware]").replace("-", " ")


_HTML = """<!doctype html><html><head><meta charset='utf-8'><title>SimBench signal similarity</title><style>
body{font-family:system-ui,sans-serif;max-width:1450px;margin:2rem auto;color:#1e293b}.controls{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:1rem}label{font-weight:600}select,input,textarea,button{display:block;width:100%;box-sizing:border-box;margin-top:.3rem;padding:.45rem;font:inherit}select[multiple]{height:13rem}textarea{height:5rem}.wide{grid-column:1/-1}button{background:#0f766e;color:white;border:0;font-weight:700;cursor:pointer}table{border-collapse:collapse;width:100%;font-size:.86rem;margin:.5rem 0 1.5rem}th,td{border:1px solid #cbd5e1;padding:.4rem;text-align:left;vertical-align:top}th{background:#f1f5f9}pre{background:#f1f5f9;padding:.6rem;white-space:pre-wrap}.error{color:#b91c1c;font-weight:700}.note{color:#475569}</style></head><body>
<h1>SimBench signal-similarity explorer</h1><p class='note'>Select a condition, entity type, and one or more signals rooted at that entity type. It shows component scores and raw nearest records.</p>
<div class='controls'><label>Discovery run<select id='run'></select></label><label class='wide'>Condition<select id='condition'></select></label><label>Retrieved entity table<select id='entity'></select></label><label>Query entity ID<input id='query' list='queryIds'><datalist id='queryIds'></datalist></label><label class='wide'>Compatible signals (Ctrl/Cmd-click for several)<select id='signals' multiple></select></label><label class='wide'>Optional weights: one SIGNAL_ID=WEIGHT per line<textarea id='weights'></textarea></label><label>Nearest entities to show<input id='topK' type='number' min='1' max='100' value='20'></label><div><label>&nbsp;</label><button id='score'>Score nearest entities</button></div></div><p id='status'></p><div id='output'></div>
<script>
const $=id=>document.getElementById(id),esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let busy=false;
async function get(url){const r=await fetch(url),b=await r.json();if(!r.ok)throw Error(b.error||r.statusText);return b}function fill(el,items,value,label,key){el.innerHTML='';for(const x of items){const o=document.createElement('option');o.value=key?x[key]:x;o.textContent=label?label(x):x;el.append(o)}if(value&&[...el.options].some(o=>o.value===value))el.value=value}function selected(){return[...$('signals').selectedOptions].map(x=>x.value)}
async function reload(){if(busy)return;busy=true;$('status').textContent='Loading options…';try{let p=new URLSearchParams({filename:$('run').value,condition_id:$('condition').value,entity_table:$('entity').value}),d=await get('/api/options?'+p);fill($('run'),d.runs,d.selected.filename,x=>x.label,'filename');fill($('condition'),d.conditions,d.selected.condition_id,x=>x.id+' — '+x.condition,'id');p=new URLSearchParams({filename:$('run').value,condition_id:$('condition').value,entity_table:$('entity').value});d=await get('/api/options?'+p);fill($('entity'),d.entity_tables,d.selected.entity_table);p=new URLSearchParams({filename:$('run').value,condition_id:$('condition').value,entity_table:$('entity').value});d=await get('/api/options?'+p);$('queryIds').innerHTML=d.entity_ids.map(id=>'<option value="'+esc(id)+'"></option>').join('');if(!d.entity_ids.includes($('query').value))$('query').value=d.entity_ids[0]||'';const old=new Set(selected());$('signals').innerHTML=d.signals.map(s=>'<option value="'+esc(s.id)+'">#'+esc(s.rank)+' | '+esc(s.id)+' | '+esc(s.label)+'</option>').join('');[...$('signals').options].forEach((o,i)=>o.selected=old.size?old.has(o.value):i<Math.min(3,d.signals.length));$('status').textContent=d.signals.length+' compatible signals rooted at '+d.selected.entity_table+'; IDs use '+d.entity_id_kind+'.'}catch(e){$('status').innerHTML='<span class="error">'+esc(e.message)+'</span>'}finally{busy=false}}
function weights(){const r={};for(const x of $('weights').value.split(/\\n/)){if(!x.trim())continue;const n=x.lastIndexOf('=');if(n<1)throw Error('Weights must be SIGNAL_ID=NUMBER.');const v=Number(x.slice(n+1));if(!Number.isFinite(v)||v<0)throw Error('Weights must be non-negative.');r[x.slice(0,n)]=v}return r}function table(h,rows){return'<table><thead><tr>'+h.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}
function render(r){const scoreRows=[],componentRows=[];r.pairs.forEach((p,i)=>{const available=p.components.filter(x=>x.score!=null);scoreRows.push([i+1,p.candidate_entity_id,Number(p.similarity_score).toFixed(3),available.length+'/'+p.components.length]);available.forEach(x=>componentRows.push([i+1,p.candidate_entity_id,x.signal_id,Number(x.score).toFixed(3),Number(x.confidence??1).toFixed(2),Number(x.effective_weight??x.weight??1).toFixed(2)]))});let h='<h2>'+esc(r.condition_id)+' — '+esc(r.entity_table)+' query '+esc(r.query_entity_id)+'</h2><p><b>Condition:</b> '+esc(r.condition)+'</p><h3>Selected evidence</h3>'+table(['rank','signal','measure','weight','query values'],r.signals.map(s=>[s.rank,s.id,s.measure,s.weight,(s.query_values||[]).slice(0,8).join(', ')]))+'<h3>Query record</h3><pre>'+esc(JSON.stringify(r.query_record,null,2))+'</pre><h3>Nearest entities</h3>'+table(['rank','candidate','combined similarity','available signals'],scoreRows)+'<h3>Available component scores</h3>'+table(['rank','candidate','signal','component similarity','evidence confidence','effective weight'],componentRows)+'<h3>Candidate records</h3>';r.pairs.forEach((p,i)=>h+='<details><summary>#'+(i+1)+' — '+esc(p.candidate_entity_id)+' (score '+Number(p.similarity_score).toFixed(3)+')</summary><pre>'+esc(JSON.stringify(p.candidate_record,null,2))+'</pre></details>');$('output').innerHTML=h}
async function score(){$('output').innerHTML='';$('status').textContent='Scoring…';try{const r=await fetch('/api/score',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:$('run').value,condition_id:$('condition').value,entity_table:$('entity').value,query_id:$('query').value,signal_ids:selected(),weights:weights(),top_k:Number($('topK').value)})}),b=await r.json();if(!r.ok)throw Error(b.error||r.statusText);render(b);$('status').textContent='Scored '+b.pairs.length+' nearest candidates.'}catch(e){$('status').innerHTML='<span class="error">'+esc(e.message)+'</span>'}}
for(const id of ['run','condition','entity'])$(id).addEventListener('change',reload);$('score').addEventListener('click',score);reload();
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--conditions-root", type=Path, default=DEFAULT_CONDITIONS_ROOT)
    parser.add_argument(
        "--model-cache-root",
        type=Path,
        help="Hugging Face hub cache containing the optional local text embedding model",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--port-attempts",
        type=int,
        default=50,
        help="number of consecutive ports to try when the requested port is occupied",
    )
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if args.port_attempts <= 0:
        parser.error("--port-attempts must be positive")
    ExplorerHandler.explorer = SimilarityExplorer(
        args.dataset_root,
        args.results_root,
        args.conditions_root,
        model_cache_root=args.model_cache_root,
    )
    server = None
    requested_port = args.port
    last_port = min(65535, requested_port + args.port_attempts - 1) if requested_port else 0
    ports = (0,) if requested_port == 0 else range(requested_port, last_port + 1)
    for port in ports:
        try:
            server = ThreadingHTTPServer((args.host, port), ExplorerHandler)
            break
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
    if server is None:
        parser.exit(
            2,
            f"error: ports {requested_port}-{last_port} are occupied; "
            "choose another port with --port or widen the search with --port-attempts\n",
        )
    actual_port = server.server_address[1]
    if actual_port != requested_port:
        print(f"Port {requested_port} is occupied; using available port {actual_port}.")
    print(f"Open http://{args.host}:{actual_port} (use VS Code port forwarding on a remote host).", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped similarity explorer.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
