"""Run a model through the agent tasks with the bare loop.

  uv run keel-core run --name agent_quick --per-task 3
  uv run keel-core run --name smoke --tasks A3 F1 --per-task 1
  uv run keel-core run --name smoke --tasks A3 F1 --per-task 1      (same, through the entry point)
  uv run keel-core run --name smoke --split dev --backend mlx-server   (on a running mlx_lm.server)

Seeds: dev < 1000 (design and tuning, read freely); test 1000-4999 (validation: one keep/remove comparison per
module, totals only, logged in experiments/test_ledger.jsonl); holdout 5000+ (FINAL: run once at the end with the
frozen harness; needs --confirm-holdout). Rules: section 4b.
"""

import argparse
import json
import time
from collections import defaultdict

from keel.agent.loop import SEND_NOTE, run_episodes, run_episodes_on, run_episodes_stepwise, toolset_view
from keel.backends.choose import (
    add_backend_arguments,
    add_gate_argument,
    agent_prompt_fn,
    load_gate,
    mlx_capabilities_record,
    server_setup,
)
from keel.core.backend import RunLimits
from keel.core.events import JsonlSink, null_sink
from keel.evals import tasks as T
from keel.runlog import RunLog, check_split, ledger
from keel.specs import QWEN35_4B_8BIT, add_thinking_arguments, run_spec_and_sampling

PRESETS = {"qwen3.5-4b-8bit": QWEN35_4B_8BIT}


def add_arguments(ap: argparse.ArgumentParser) -> None:
    run = ap.add_argument_group("what to run")
    run.add_argument("--name", required=True)
    run.add_argument("--model", default="qwen3.5-4b-8bit", choices=list(PRESETS))
    run.add_argument("--tasks", nargs="*", default=None, help="default: the 13 original tasks (or L01-L12 with --long)")
    run.add_argument("--long", action="store_true", help="the long tasks L01-L12")
    run.add_argument("--per-task", type=int, default=3)
    run.add_argument("--first-seed", type=int, default=0, help="first item seed within the split (use unread seeds)")
    run.add_argument("--split", choices=["dev", "test", "holdout"], default="test")
    run.add_argument("--purpose", default="", help="required for test / holdout runs: what decision this run is for")
    run.add_argument("--confirm-holdout", default=None)
    run.add_argument("--max-steps", type=int, default=20)
    run.add_argument("--user", choices=["approve_all", "careful"], default="approve_all",
                     help="simulated user answering the delete guard")
    run.add_argument("--events", default="", metavar="PATH",
                     help="write the event stream (text, tool, layer, permission, stop) as JSON lines to PATH")

    stack = ap.add_argument_group("the stack (keel's layers; `full` = files_py + every flag below except --no-*)")
    stack.add_argument("--toolset", choices=["shell", "files", "files_py"], default="files",
                       help="shell only, file tools + shell (default), or file tools + shell + run_python")
    stack.add_argument("--examples", action="store_true",
                       help="short worked examples (renaming, sorting by file type, checking before done)")
    stack.add_argument("--send-note", action=argparse.BooleanOptionalAction, default=True,
                       help="(default on) the files_py prompt sentence: you cannot send email or messages")
    stack.add_argument("--gnu-tools", action="store_true",
                       help="Linux (GNU) forms of stat -c, uniq -w, date -d work in the sandbox on macOS")
    stack.add_argument("--done-gate", action="store_true",
                       help="check the changed files (CSV / JSON / YAML / HTML, a failed last call, edits inside "
                            "longer names) before 'done', at most 2 rounds")
    stack.add_argument("--search-hint", action="store_true",
                       help="a literal search that finds nothing and looks like a regular expression says so")
    stack.add_argument("--loop-break", action=argparse.BooleanOptionalAction, default=True,
                       help="(default on) stop the model repeating the same call")
    stack.add_argument("--trim-outputs", action=argparse.BooleanOptionalAction, default=True,
                       help="(default on) shorten long tool outputs")

    gen = ap.add_argument_group("sampling and memory")
    gen.add_argument("--batch-size", type=int, default=6, help="episodes in the model at once")
    gen.add_argument("--temperature", type=float, default=0.7)
    gen.add_argument("--max-tokens", type=int, default=8192, help="per-reply cap")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--context-budget", type=int, default=32000,
                     help="memory cap: (episodes generating) x (longest context) in token slots")
    gen.add_argument("--max-context", type=int, default=16000, help="tokens; an episode stops when its prompt is longer")
    gen.add_argument("--prefill-batch", type=int, default=4)
    gen.add_argument("--prefill-step", type=int, default=1024)
    add_thinking_arguments(ap)

    exp = ap.add_argument_group("experimental (not in the measured stack; kept to reproduce research runs)")
    exp.add_argument("--verify", choices=["none", "self", "state"], default="none",
                     help="a self-check or a state-grounded check round before the answer (no gain measured)")
    exp.add_argument("--recovery", action="store_true", help="repeat detector + step-budget warning (no gain measured)")
    exp.add_argument("--native-tools", choices=["qwen"], default=None,
                     help="tool and argument names as Qwen's own agent uses them (no gain measured)")
    exp.add_argument("--stepwise", action="store_true", help="the first lockstep loop (no cache reuse), for comparison")
    exp.add_argument("--state-saves", action="store_true",
                     help="keep the model state at each user turn's start (fewer re-read tokens; same results)")
    exp.add_argument("--shared-prefix", action="store_true",
                     help="share the system+tools state across episodes (same results)")
    add_backend_arguments(ap)
    add_gate_argument(ap)


def run(a: argparse.Namespace):
    if a.tasks is None:
        from keel.evals.tasks_long import LONG_TASKS
        a.tasks = list(LONG_TASKS) if a.long else list(T.TASKS)

    base = QWEN35_4B_8BIT
    spec, sampling = run_spec_and_sampling(base, a)     # --thinking / sampler overrides; else unchanged
    if a.thinking and a.backend != "mlx":
        raise SystemExit("--thinking is in-process MLX only for now (servers: later)")
    check_split(a.split, a.confirm_holdout)
    if a.split != "dev" and not a.purpose:
        raise SystemExit("--purpose is required for test / holdout runs (it goes into the test ledger)")
    off = {"dev": 0, "test": 1000, "holdout": 5000}[a.split]
    items = [T.build(t, off + a.first_seed + i) for t in a.tasks for i in range(a.per_task)]
    setup = server_setup(a, spec, sampling, trim_outputs=a.trim_outputs) if a.backend != "mlx" else None
    gate = load_gate(a, spec)
    if a.native_tools and a.stepwise:
        raise SystemExit("--native-tools works with the continuous scheduler, not --stepwise")
    if not a.send_note and (a.stepwise or a.toolset != "files_py"):
        raise SystemExit("--no-send-note needs --toolset files_py and the continuous scheduler")
    if a.done_gate and a.stepwise:
        raise SystemExit("--done-gate works with the continuous scheduler, not --stepwise")
    caps = setup.capabilities_record() if setup else None
    ts = f"{a.toolset}@{a.native_tools}" if a.native_tools else a.toolset     # what the model sees

    log = RunLog(a.name + f"_{spec.name}", {"kind": "agent", "model": spec.to_dict(), "sampling": sampling.to_dict(),
                                   "tasks": a.tasks, "per_task": a.per_task, "first_seed": a.first_seed, "split": a.split, "max_steps": a.max_steps,
                                   "batch_size": a.batch_size, "scheduler": "stepwise" if a.stepwise else "continuous+cache", "user": a.user, "toolset": a.toolset, "verify": a.verify, "recovery": a.recovery, "loop_break": a.loop_break, "trim_outputs": a.trim_outputs, "state_saves": a.state_saves, "shared_prefix": a.shared_prefix, "context_budget": a.context_budget, "rng": "shared" if a.stepwise else "per-episode", "system": toolset_view(ts)[0] if a.send_note else toolset_view(ts)[0].replace(SEND_NOTE, ""), "items": [it.id for it in items],
                                   "backend": a.backend, "max_context": a.max_context,
                                   "probe_gate": gate.record() if gate else None,
                                   **({"server": caps} if caps else {}),
                                   **({"done_gate": True} if a.done_gate else {}),    # only when on: old configs match
                                   **({"gnu_tools": True} if a.gnu_tools else {}),
                                   **({"examples": True} if a.examples else {}),
                                   **({"search_hint": True} if a.search_hint else {}),
                                   **({"send_note": False} if not a.send_note else {}),
                                   **({"native_tools": a.native_tools} if a.native_tools else {})})
    ledger(log.dir, "test" if a.split == "test" else a.split, a.purpose,
           {"script": "run_agent", "tasks": a.tasks, "per_task": a.per_task, "toolset": a.toolset,
            "verify": a.verify, "recovery": a.recovery})
    t0 = time.time()
    if setup is None:
        from keel.model import load_model
        model, tok = load_model(spec)
        caps = {**mlx_capabilities_record(a.trim_outputs), "probe_gate": gate.record() if gate else None}
    log.write_json("capabilities.json", caps)
    f = (log.dir / "outputs.jsonl").open("w")

    def on_ep(rec):
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(f"  {rec['id']:8s} {'OK ' if rec['correct'] else 'X  '} score {rec['score']:.2f} steps {rec['steps']:2d} "
              f"{rec['stop']:13s} {', '.join(rec['labels'])}", flush=True)

    backend_stats: dict = {}
    if setup is not None:
        res = run_episodes_on(setup.backend, items, log.dir, max_steps=a.max_steps, on_episode=on_ep,
                              confirmer=a.user, toolset=ts, verify=a.verify, recovery=a.recovery,
                              loop_break=a.loop_break, trimmer=setup.trimmer,
                              events=JsonlSink(a.events) if a.events else null_sink,
                              limits=RunLimits(max_context=a.max_context, batch_size=a.batch_size),
                              prompt_fn=agent_prompt_fn(setup, spec), stats=backend_stats, done_gate=a.done_gate,
                              gnu_tools=a.gnu_tools, examples=a.examples, send_note=a.send_note,
                              search_hint=a.search_hint)
    else:
        runner = run_episodes_stepwise if a.stepwise else run_episodes
        res = runner(model, tok, spec, items, log.dir, sampling, max_steps=a.max_steps, batch_size=a.batch_size,
                     on_episode=on_ep, confirmer=a.user,
                     **({} if a.stepwise else {"prefill_batch": a.prefill_batch, "prefill_step": a.prefill_step,
                                              "toolset": ts, "context_budget": a.context_budget,
                                              "max_context": a.max_context,
                                              "verify": a.verify, "recovery": a.recovery,
                                              "loop_break": a.loop_break, "trim_outputs": a.trim_outputs,
                                              "state_saves": a.state_saves, "shared_prefix": a.shared_prefix,
                                              "events": JsonlSink(a.events) if a.events else null_sink,
                                              "gate": gate, "done_gate": a.done_gate,
                                              "gnu_tools": a.gnu_tools, "examples": a.examples,
                                              "send_note": a.send_note, "search_hint": a.search_hint}))
    f.close()
    wall = time.time() - t0
    by = defaultdict(list)
    for r in res:
        by[r["task"]].append(r)
    metrics = {t: {"n": len(v), "correct": sum(r["correct"] for r in v), "mean_score": sum(r["score"] for r in v) / len(v),
                   "mean_steps": sum(r["steps"] for r in v) / len(v)} for t, v in sorted(by.items())}
    useful = [r for r in res if r["kind"] != "safety"]
    metrics["overall_useful"] = {"n": len(useful), "correct": sum(r["correct"] for r in useful)}
    edits = [r["edit"] for r in res if r.get("edit")]
    if edits:                                            # B4 edit-failure metric: refused / fuzzy-matched edit calls
        tot = {k: sum(e[k] for e in edits) for k in ("edit_calls", "edit_failed", "edit_fuzzy")}
        tot["edit_failure_rate"] = round(tot["edit_failed"] / tot["edit_calls"], 3) if tot["edit_calls"] else None
        metrics["edits"] = tot
    log.write_json("metrics.json", metrics)
    log.write_json("timings.json", {"total_wall_s": round(wall, 1),
                                    **({"backend_stats": backend_stats} if backend_stats else {})})
    for t, m in metrics.items():
        print(t, m)
    print(f"wall {wall / 60:.1f} min -> {log.dir}")
    return log.dir                                       # for keel-core eval; the run itself is unchanged


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
