"""BFCL v4 multi-turn through the continuous scheduler, scored with the official checker (see keel/evals/bfcl_mt.py).

  uv run keel-core bfcl-mt --name bfcl_mt_smoke --split dev --limit 3
  uv run keel-core bfcl-mt --name bfcl_mt_smoke --split dev --limit 3      (same, through the entry point)
  uv run keel-core bfcl-mt --name smoke --split dev --limit 3 --backend mlx-server   (a running mlx_lm.server)
  uv run keel-core bfcl-mt --name bfcl_mt_val_base --split val --purpose "..."

Splits: dev = design and tuning (read freely); val = validation (one comparison per module,
totals only, logged in experiments/test_ledger.jsonl); holdout = FINAL, once, frozen harness, --confirm-holdout.
"""

import argparse
import json
import time
from collections import defaultdict

from keel.backends.choose import (
    add_backend_arguments,
    add_gate_argument,
    load_gate,
    mlx_capabilities_record,
    server_setup,
)
from keel.core.backend import RunHooks, RunLimits
from keel.core.events import Event, JsonlSink, null_sink
from keel.evals import bfcl_mt as M
from keel.rendering import RENDERINGS, template_kwargs
from keel.runlog import RunLog, check_split, ledger
from keel.specs import QWEN35_4B_8BIT, add_thinking_arguments, run_spec_and_sampling
from keel.trim import make_trimmer

PRESETS = {"qwen3.5-4b-8bit": QWEN35_4B_8BIT}


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="qwen3.5-4b-8bit", choices=list(PRESETS))
    ap.add_argument("--categories", nargs="*", default=M.CATEGORIES, choices=M.CATEGORIES)
    ap.add_argument("--split", choices=["dev", "val", "holdout"], default="dev")
    ap.add_argument("--purpose", default="", help="required for val / holdout runs (goes into the test ledger)")
    ap.add_argument("--confirm-holdout", default=None)
    ap.add_argument("--limit", type=int, default=0, help="first N entries per category (0 = all in the split)")
    ap.add_argument("--ids", nargs="*", default=None, help="run only these entry ids")
    ap.add_argument("--skip-done", nargs="*", default=[], help="run folders whose finished entries are skipped (resume)")
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--context-budget", type=int, default=32000)
    ap.add_argument("--max-context", type=int, default=24000)
    ap.add_argument("--prefill-batch", type=int, default=4)
    ap.add_argument("--prefill-step", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    add_thinking_arguments(ap)
    ap.add_argument("--verify", choices=["none", "self", "state"], default="none",
                    help="experimental: a self-check or a state report, once per turn")
    ap.add_argument("--recovery", action="store_true", help="experimental: repeat detector + step-budget warning")
    ap.add_argument("--reminder", action=argparse.BooleanOptionalAction, default=True, help="(default on) re-show unfinished earlier requests (to-do reminder)")
    ap.add_argument("--loop-break", action=argparse.BooleanOptionalAction, default=True, help="(default on) stop the model repeating the same call")
    ap.add_argument("--arg-source", action="store_true",
                    help="experimental: hold a call whose argument values appear in no user message or tool output, and "
                         "tell the model to ask (off by default)")
    ap.add_argument("--trim-outputs", action=argparse.BooleanOptionalAction, default=True, help="(default on) shorten long tool outputs")
    ap.add_argument("--check-requests", choices=["current", "recent"], default="current",
                    help="what the check quotes: this turn's user message, or the previous + latest turn's")
    ap.add_argument("--rendering", choices=RENDERINGS, default="official",
                    help="official = BFCL chat template; cache = keep empty think blocks in history (dev only)")
    ap.add_argument("--state-saves", action="store_true",
                    help="experimental: keep the model state at each user turn's start (fewer re-read tokens; parity-checked)")
    ap.add_argument("--events", default="", metavar="PATH",
                    help="write the event stream (text, tool, layer, permission, stop) as JSON lines to PATH")
    ap.add_argument("--shared-prefix", action="store_true",
                    help="experimental: share the system+tools state across episodes (parity-checked)")
    add_backend_arguments(ap)
    add_gate_argument(ap)


def run(a: argparse.Namespace):
    check_split(a.split, a.confirm_holdout)
    if a.split != "dev" and not a.purpose:
        raise SystemExit("--purpose is required for val / holdout runs (it goes into the test ledger)")
    if a.rendering != "official" and a.split != "dev":
        raise SystemExit("--rendering cache is for dev runs only; Milestone and holdout keep the official rendering")
    base = PRESETS[a.model]
    spec, sampling = run_spec_and_sampling(base, a)     # --thinking / sampler overrides; else unchanged
    if a.thinking and a.backend != "mlx":
        raise SystemExit("--thinking is in-process MLX only for now (servers: later)")
    setup = server_setup(a, spec, sampling, trim_outputs=a.trim_outputs) if a.backend != "mlx" else None
    if setup is not None and a.rendering != "official":
        raise SystemExit("--rendering cache is an in-process MLX experiment; servers use the official rendering")
    caps = setup.capabilities_record() if setup else None
    gate = load_gate(a, spec)
    done_ids = set()
    for d in a.skip_done:
        done_ids |= {json.loads(l)["id"] for l in open(f"{d}/outputs.jsonl") if l.strip()}
    entries = []
    for c in a.categories:
        rows = M.load(c, a.split)
        if a.ids:
            rows = [r for r in rows if r["id"] in a.ids]
        rows = [r for r in rows if r["id"] not in done_ids]
        entries += rows[:a.limit] if a.limit else rows

    log = RunLog(a.name + f"_{spec.name}", {
        "kind": "bfcl_multi_turn", "bfcl_eval_version": "2026.3.23 (BFCL v4 data)", "model": spec.to_dict(),
        "sampling": sampling.to_dict(), "rng": "per-episode", "categories": a.categories, "split": a.split, "limit": a.limit,
        "skip_done": a.skip_done, "protocol": "official FC protocol: chat template + tools, no system prompt, turn ends on a reply without a "
                    "tool call, >20 steps in a turn = force quit; official multi-turn checker",
        "verify": a.verify, "recovery": a.recovery, "check_requests": a.check_requests,
        "reminder": a.reminder, "loop_break": a.loop_break, "trim_outputs": a.trim_outputs,
        **({"arg_source": True} if a.arg_source else {}),
        "state_saves": a.state_saves, "shared_prefix": a.shared_prefix,
        "rendering": a.rendering, "batch_size": a.batch_size, "context_budget": a.context_budget, "max_context": a.max_context,
        "items": [r["id"] for r in entries], "backend": a.backend, **({"server": caps} if caps else {}),
        "probe_gate": gate.record() if gate else None})
    ledger(log.dir, a.split, a.purpose, {"script": "run_bfcl_mt", "categories": a.categories, "n": len(entries),
                                           "verify": a.verify, "recovery": a.recovery,
                                           "check_requests": a.check_requests,
                                           "reminder": a.reminder, "loop_break": a.loop_break,
                                           "trim_outputs": a.trim_outputs,
                                           "rendering": a.rendering})
    exec_name = M.safe_name(log.dir.name)
    t0 = time.time()
    if setup is None:
        from keel.model import load_model
        model, tok = load_model(spec)
        caps = {**mlx_capabilities_record(a.trim_outputs), "probe_gate": gate.record() if gate else None}
        trimmer = make_trimmer(tok) if a.trim_outputs else None
    else:
        tok, trimmer = setup.tok, setup.trimmer
    log.write_json("capabilities.json", caps)
    render_kw = template_kwargs(tok, a.rendering) if tok is not None else {}

    def prompt_text(e):
        return tok.apply_chat_template(e.messages, tools=e.tools(), add_generation_prompt=True, tokenize=False,
                                       **spec.chat_template_kwargs, **render_kw)

    if setup is None:
        def prompt_fn(e):
            return tok.encode(prompt_text(e), add_special_tokens=False)
    elif setup.backend.mode == "raw":
        prompt_fn = prompt_text
    else:
        def prompt_fn(e):
            return {"messages": [dict(m) for m in e.messages], "tools": e.tools(), "funcs": e.entry["function"]}

    events = JsonlSink(a.events) if a.events else null_sink
    eps = [M.MTEpisode(r, exec_name, verify=a.verify, recovery=a.recovery, check_requests=a.check_requests,
                       reminder=a.reminder, loop_break=a.loop_break, arg_source=a.arg_source,
                       trimmer=trimmer, events=events,
                       thinking=bool(spec.chat_template_kwargs.get("enable_thinking"))).start()
           for r in entries]
    order = {"long_context": 0, "miss_func": 1, "miss_param": 1, "base": 2}
    eps.sort(key=lambda e: order[e.entry["category"]])          # long first

    out = (log.dir / "outputs.jsonl").open("w")
    (log.dir / "transcripts").mkdir(exist_ok=True)
    results = []

    def finish_fn(e):
        e.events(Event("stop", e.entry["id"], {"reason": e.stop, "steps": e.steps}))
        s = M.score(e.entry, e.result, exec_name + "_score")   # unfinished entries score as force-terminated
        rec = {**s, "stop": e.stop, "turns": len(e.entry["question"]), "turns_run": len(e.result), "steps": e.steps,
               "exec_errors": e.exec_errors, "arg_holds": e.argcheck.holds if e.argcheck else 0, "notes": e.notes, "todo_log": e.todo_log, "prompt_tokens": e.prompt_tokens, "processed_tokens": e.processed_tokens,
               "output_tokens": e.output_tokens, "seconds": round(e.seconds, 1), "result": e.result}
        results.append(rec)
        out.write(json.dumps(rec, default=str) + "\n")
        out.flush()
        (log.dir / "transcripts" / f"{e.entry['id']}.json").write_text(json.dumps(
            {**rec, "messages": e.messages, "tool_results": e.tool_results,
             "ground_truth": e.entry["ground_truth"]}, indent=1, default=str))
        print(f"  {e.entry['id']:28s} {'OK ' if s['valid'] else 'X  '} steps {e.steps:2d} {e.stop:13s} "
              f"{s.get('error_type') or ''}", flush=True)

    if setup is None:
        from keel.backends.mlx import run_scheduler  # MLX: loaded only when running
        stats = run_scheduler(model, tok, sampling, eps, prompt_fn=prompt_fn,
                              apply_fn=lambda e, raw, fr: e.apply(raw, fr), finish_fn=finish_fn,
                              max_context=a.max_context, batch_size=a.batch_size, prefill_step=a.prefill_step,
                              prefill_batch=a.prefill_batch, context_budget=a.context_budget,
                              label_fn=lambda e: e.entry["id"].replace("multi_turn_", ""),
                              seed_fn=lambda e: e.entry["id"], state_saves=a.state_saves,
                              shared_prefix=a.shared_prefix, gate=gate)
    else:
        hooks = RunHooks(prompt_fn=prompt_fn, apply_fn=lambda e, raw, fr: e.apply(raw, fr), finish_fn=finish_fn,
                         seed_fn=lambda e: e.entry["id"], label_fn=lambda e: e.entry["id"].replace("multi_turn_", ""))
        stats = {"backend_stats": setup.backend.run(eps, hooks, RunLimits(max_context=a.max_context,
                                                                          batch_size=a.batch_size))}
    out.close()
    wall = time.time() - t0

    by = defaultdict(list)
    for r in results:
        by[r["category"]].append(r)
    metrics = {}
    for c, v in sorted(by.items()):
        metrics[c] = {"n": len(v), "valid": sum(r["valid"] for r in v), "strict": sum(r["strict"] for r in v),
                      "accuracy": round(sum(r["valid"] for r in v) / len(v), 4),
                      "errors": dict(sorted(defaultdict(int, {k: sum(1 for r in v if r.get("error_type") == k)
                                                              for k in {r.get("error_type") for r in v} if k}).items())),
                      "stops": {k: sum(1 for r in v if r["stop"] == k) for k in {r["stop"] for r in v}},
                      "mean_steps": round(sum(r["steps"] for r in v) / len(v), 1)}
    n = len(results)
    metrics["overall"] = {"n": n, "valid": sum(r["valid"] for r in results),
                          "accuracy": round(sum(r["valid"] for r in results) / max(n, 1), 4),
                          "strict": sum(r["strict"] for r in results),
                          "category_mean_accuracy": round(sum(m["accuracy"] for c, m in metrics.items()) /
                                                          max(len(by), 1), 4)}
    log.write_json("metrics.json", metrics)
    log.write_json("timings.json", {"total_wall_s": round(wall, 1), **stats})
    for c, m in metrics.items():
        print(c, m)
    print(f"wall {wall / 60:.1f} min -> {log.dir}")
    return log.dir                                       # for keel-core eval; the run itself is unchanged


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
