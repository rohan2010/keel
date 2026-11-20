"""keel-core baseline: run an off-the-shelf agent (goose, opencode) on keel's agent tasks.

  uv run keel-core baseline --agent goose --name bl_goose_dev --split dev --per-task 1 --first-seed 30
  uv run keel-core baseline --agent opencode --name ... --tasks A3 E3 --concurrency 2

Needs `mlx_lm.server` running on 127.0.0.1:8080 with keel's weights and keel's sampling defaults:
  uv run mlx_lm.server --model models/Qwen3.5-4B-8bit --port 8080 \\
      --chat-template-args '{"enable_thinking":false}' --temp 0.7 --top-p 0.8 --top-k 20 --max-tokens 8192
Each episode runs sandboxed (see keel.evals.baselines). The run folder uses keel's agent format, so
`keel-core compare <keel run> <baseline run> --allow backend` works.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from keel.evals import baselines as B
from keel.evals import tasks as T
from keel.runlog import RunLog, ledger
from keel.specs import QWEN35_4B_8BIT


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--agent", choices=sorted(B.SETUPS), required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tasks", nargs="*", default=None, help="default: the original 13 tasks (as keel-core run)")
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--purpose", default="", help="required for test runs (goes into the test ledger)")
    ap.add_argument("--per-task", type=int, default=1)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4, help="episodes at once (keel arms use batch 4)")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=900, help="wall-clock cap per episode, seconds")


def _server_model() -> str:
    try:
        with urllib.request.urlopen(f"{B.SERVER}/v1/models", timeout=5) as r:
            return json.loads(r.read())["data"][0]["id"]
    except Exception as err:
        raise SystemExit(f"mlx_lm.server is not answering on {B.SERVER} ({err}); start it first (see --help)") from None


def _server_generates(model: str) -> bool:
    """A one-token chat request. /v1/models keeps answering after mlx_lm.server's generation thread dies (Metal out
    of memory, ch6dev 2026-11-19), so only a real completion shows the server can still serve an episode."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).encode()
    req = urllib.request.Request(f"{B.SERVER}/v1/chat/completions", body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status == 200
    except Exception:
        return False


class ServerDied(SystemExit):
    pass


def run(a: argparse.Namespace):
    if a.split == "test" and not a.purpose:
        raise SystemExit("--purpose is required for test runs")
    served = _server_model()
    tasks = a.tasks or list(T.TASKS)
    off = {"dev": 0, "test": 1000}[a.split]
    items = [T.build(t, off + a.first_seed + i) for t in tasks for i in range(a.per_task)]
    spec = B.agent_spec(a.agent, a.max_turns, a.timeout)
    sampling = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 0.0, "max_tokens": 8192,
                "seed": None}
    log = RunLog(a.name + f"_{QWEN35_4B_8BIT.name}", {
        # model: the server's template args (thinking off), not the preset's default (it said thinking on, 2026-11-19)
        "kind": "agent", "model": {**QWEN35_4B_8BIT.to_dict(), "chat_template_kwargs": {"enable_thinking": False}},
        "sampling": sampling, "tasks": tasks,
        "per_task": a.per_task, "first_seed": a.first_seed, "split": a.split, "max_steps": a.max_turns,
        "batch_size": a.concurrency, "toolset": a.agent, "rng": "server", "items": [it.id for it in items],
        "backend": f"baseline:{a.agent}", "baseline": {"agent": spec.name, "version": spec.version, **spec.config,
                                                       "server": B.SERVER, "served_model": served,
                                                       "server_defaults": "thinking off, T 0.7, top-p 0.8, top-k 20, "
                                                                          "8192 tokens"}})
    ledger(log.dir, a.split, a.purpose, {"script": "run_baseline", "agent": a.agent, "n": len(items)})
    out = (log.dir / "outputs.jsonl").open("w")
    t0 = time.time()

    dead = threading.Event()

    def one(it):
        if dead.is_set():                     # the server stopped serving: no more episodes (they'd score as fails)
            return None
        rec = B.run_episode(a.agent, it, log.dir, max_turns=a.max_turns, timeout_s=a.timeout)
        if rec["stop"] not in ("answer", "timeout") and not _server_generates(served):
            dead.set()                        # this episode failed because the server did: not a result
            print(f"  {rec['id']:8s} server not generating: run aborted", flush=True)
            return None
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"  {rec['id']:8s} {'OK ' if rec['correct'] else 'X  '} {rec['seconds']:6.1f}s steps "
              f"{rec['steps'] if rec['steps'] is not None else '?':>3} {rec['stop']:8s} {', '.join(rec['labels'])}",
              flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        recs = list(ex.map(one, items))
    out.close()
    if dead.is_set():
        log.write_json("metrics.json", {"aborted": "model server stopped generating", "completed": sum(
            r is not None for r in recs), "n": len(items)})
        raise ServerDied(f"ABORTED: {B.SERVER} stopped generating (see the server log); {log.dir} is not a result")
    n_ok = sum(r["correct"] for r in recs)
    log.write_json("metrics.json", {"passed": n_ok, "n": len(recs), "pass_rate": n_ok / len(recs) if recs else 0,
                                    "timeouts": sum(r["stop"] == "timeout" for r in recs),
                                    "unparsed": sum(not r.get("output_parsed") for r in recs)})
    log.time("total", time.time() - t0)
    print(f"{n_ok}/{len(recs)} passed  ({(time.time() - t0) / 60:.1f} min) -> {log.dir}")
    return log.dir
