"""When2Call eval: call / ask / can't-answer decisions, with or without the probe gate.

  uv run keel-core when2call --name w2c_dev_off --split dev
  uv run keel-core when2call --name w2c_dev_on --split dev --probe-gate
  uv run keel-core when2call --name w2c_ho_off --split held_out --purpose "P8 held-out, gate off"
  uv run keel-core when2call --compare experiments/<off run> experiments/<on run>      (paired McNemar)
  uv run keel-core when2call --name w2c_mcq --split held_out --mcq --purpose "..."     (MCQ log-prob reference)

Pre-registered design (" pre-registration"): unseen-source items only for the
headline (dev = plumbing, held_out = the result, seen = a separate labelled table); BFCL single-turn FC protocol;
greedy, batch size 1, max 512 new tokens; gate margin 0.27 (frozen); exact McNemar on call-decision correctness.
In-process MLX only (the gate needs hidden states, and the comparison must be on one backend).
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from keel.backends.choose import add_gate_argument, load_gate
from keel.core.arg_source import ArgSourceCheck
from keel.core.parse import parse_calls
from keel.evals import when2call as W
from keel.runlog import RunLog, ledger
from keel.specs import QWEN35_4B_8BIT, ModelSpec, SamplingSpec

PRESETS = {"qwen3.5-4b-8bit": QWEN35_4B_8BIT}


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--name", help="run name (required unless --compare)")
    ap.add_argument("--model", default="qwen3.5-4b-8bit", choices=list(PRESETS))
    ap.add_argument("--split", choices=W.SPLITS, default="dev",
                    help="dev = plumbing; held_out = the result (needs --purpose); seen = probe saw the source question")
    ap.add_argument("--purpose", default="", help="required for held_out / seen (goes into the test ledger)")
    ap.add_argument("--limit", type=int, default=0, help="first N items (0 = the whole split)")
    ap.add_argument("--batch-size", type=int, default=1, help="1 = pre-registered (no batch-induced greedy drift)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--mcq", action="store_true", help="MCQ log-prob reference instead of generation (gate off)")
    ap.add_argument("--compare", nargs=2, metavar=("OFF_RUN", "ON_RUN"), default=None,
                    help="paired comparison of two finished runs (no model)")
    ap.add_argument("--arg-source", action="store_true",
                    help="experimental: a first reply whose call has an argument value found nowhere in the request is held; "
                         "the model gets one more reply, which is the one scored (off by default)")
    add_gate_argument(ap)
    ap.set_defaults(backend="mlx")


def _rows(run_dir: str) -> list[dict]:
    return [json.loads(l) for l in open(Path(run_dir) / "outputs.jsonl") if l.strip()]


def run_compare(a: argparse.Namespace) -> None:
    off, on = a.compare
    res = W.compare(_rows(off), _rows(on))
    cfg = [json.loads((Path(d) / "config.json").read_text()) for d in (off, on)]
    for k in ("split", "sampling", "batch_size", "items_sha1"):
        if cfg[0].get(k) != cfg[1].get(k):
            raise SystemExit(f"runs differ in {k}: {cfg[0].get(k)} vs {cfg[1].get(k)}")
    res.update({"off": off, "on": on, "split": cfg[0]["split"],
                "off_gate": cfg[0].get("probe_gate"), "on_gate": cfg[1].get("probe_gate")})
    from keel.evals.compare import compare_arms, load_arm  # the same verdict keel-core compare gives
    c = compare_arms(load_arm(off), load_arm(on))
    res.update({"verdict": c.verdict, "verdict_reason": c.reason})
    print(json.dumps(res, indent=1))
    Path(on, "compare_vs_off.json").write_text(json.dumps(res, indent=1))


def run(a: argparse.Namespace):
    if a.compare:
        return run_compare(a)
    if not a.name:
        raise SystemExit("--name is required")
    if a.split != "dev" and not a.purpose:
        raise SystemExit("--purpose is required for held_out / seen runs (it goes into the test ledger)")
    if a.mcq and getattr(a, "arg_source", False):
        raise SystemExit("--mcq scores fixed answers; --arg-source cannot act on it")
    if a.mcq and a.probe_gate:
        raise SystemExit("--mcq scores fixed answers; the gate cannot act on it")
    base = PRESETS[a.model]
    spec = ModelSpec(base.name, base.hf_repo, base.revision, base.local_dir, {"enable_thinking": False})
    sampling = SamplingSpec(temperature=0.0, top_p=1.0, top_k=0, presence_penalty=0.0, max_tokens=a.max_tokens, seed=0)
    gate = load_gate(a, spec)
    items = W.load(a.split)
    if a.limit:
        items = items[:a.limit]
    items_sha1 = hashlib.sha1(json.dumps([i["id"] for i in items]).encode()).hexdigest()
    log = RunLog(a.name + f"_{spec.name}", {
        "kind": "when2call_mcq" if a.mcq else "when2call", "data": W.ATTRIBUTION, "model": spec.to_dict(),
        "sampling": sampling.to_dict(), "split": a.split, "limit": a.limit, "batch_size": a.batch_size,
        "protocol": "BFCL single-turn FC protocol: chat template + tools, no system prompt, one user message, "
                    "thinking off; correct = called iff label tool_call",
        "items": [i["id"] for i in items], "items_sha1": items_sha1, "backend": "mlx",
        "probe_gate": gate.record() if gate else None, **({"arg_source": True} if a.arg_source else {})})
    ledger(log.dir, a.split, a.purpose, {"script": "run_when2call", "split": a.split, "n": len(items),
                                            "probe_gate": bool(gate), "mcq": a.mcq})
    from keel.model import load_model
    model, tok = load_model(spec)

    def prompt_ids(item, messages=None):
        text = tok.apply_chat_template(messages or W.messages_for(item), tools=W.tools_for(item) or None,
                                       add_generation_prompt=True, tokenize=False, **spec.chat_template_kwargs)
        return tok.encode(text, add_special_tokens=False)

    t0 = time.time()
    if a.mcq:
        results = run_mcq(model, tok, items, prompt_ids, log)
        metrics = mcq_metrics(results)
    else:
        results = run_generation(model, tok, sampling, items, prompt_ids, gate, a.batch_size, log,
                                 arg_source=a.arg_source)
        metrics = {"all": W.summarize(results)}
    wall = time.time() - t0
    log.write_json("metrics.json", metrics)
    log.write_json("timings.json", {"total_wall_s": round(wall, 1)})
    print(json.dumps(metrics, indent=1))
    print(f"wall {wall / 60:.1f} min -> {log.dir}")
    return log.dir                                       # for keel-core eval; the run itself is unchanged


class _Ep:
    """One When2Call item as a one-reply scheduler episode."""

    def __init__(self, item):
        self.item, self.done, self.stop, self.steps, self.notes, self.text = item, False, "", 0, [], ""
        self.prompt_tokens = self.processed_tokens = self.output_tokens = 0
        self.t_start = self.seconds = 0.0
        self.cache = self.seq = None
        self.finish_reason = ""
        self.messages = W.messages_for(item)
        self.held = ""                                   # the first reply, if it was held


def run_generation(model, tok, sampling, items, prompt_ids, gate, batch_size, log, arg_source: bool = False) -> list[dict]:
    from keel.backends.mlx import run_scheduler
    out = (log.dir / "outputs.jsonl").open("w")
    results = []

    def apply(e, text, fr):
        e.steps += 1
        if arg_source and e.steps == 1:                  # hold a call with an unsourced value, reply again
            calls = parse_calls(text, e.item["functions"])
            hold = ArgSourceCheck().check(calls, e.item["functions"], [e.item["question"]]) if calls else None
            if hold is not None:
                e.held = text
                e.notes.append("argsrc")
                e.messages = e.messages + [{"role": "assistant", "content": text}] + \
                    [{"role": "tool", "content": hold} for _ in calls]
                return
        e.text, e.finish_reason, e.done, e.stop = text, fr, True, "answer"

    def finish(e):
        s = W.score(e.item, e.text)
        g = next((n for n in e.notes if n.startswith("gate:")), None)
        rec = {"id": e.item["id"], "source_id": e.item["source_id"], "label": e.item["label"],
               "target": e.item["target"], **s, "text": e.text, "finish_reason": e.finish_reason,
               "prompt_tokens": e.prompt_tokens, "output_tokens": e.output_tokens, "seconds": round(e.seconds, 2),
               "gate": g.split(":")[1].split("@")[0] if g else None,
               "p_call": float(g.split("p=")[1]) if g else None,
               **({"arg_held": True, "held_text": e.held} if e.held else {})}
        results.append(rec)
        out.write(json.dumps(rec) + "\n")
        out.flush()
        if len(results) % 50 == 0:
            print(f"  {len(results)}/{len(items)}  {(time.time() - t0) / 60:.1f} min", flush=True)

    t0 = time.time()
    eps = [_Ep(i) for i in items]
    stats = run_scheduler(model, tok, sampling, eps, prompt_fn=lambda e: prompt_ids(e.item, e.messages), apply_fn=apply,
                          finish_fn=finish, batch_size=batch_size, max_context=16000, progress_every=0,
                          seed_fn=lambda e: e.item["id"], gate=gate)
    out.close()
    log.write_json("scheduler_stats.json", {k: v for k, v in stats.items() if isinstance(v, (int, float, dict, str))})
    order = {i["id"]: n for n, i in enumerate(items)}
    results.sort(key=lambda r: order[r["id"]])
    return results


def run_mcq(model, tok, items, prompt_ids, log) -> list[dict]:
    import mlx.core as mx
    out = (log.dir / "outputs.jsonl").open("w")
    results = []
    for n, item in enumerate(items):
        p = prompt_ids(item)
        lps = {}
        for k, text in W.mcq_candidates(item).items():
            c = tok.encode(text, add_special_tokens=False)
            ids = mx.array(p + c)[None]
            logits = model(ids)[0, len(p) - 1:-1, :].astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            tok_lp = mx.take_along_axis(lp, mx.array(c)[:, None], axis=-1)
            lps[k] = {"logprob": float(tok_lp.sum().item()), "bytes": len(text.encode())}
            mx.clear_cache()
        pick = max(lps, key=lambda k: lps[k]["logprob"])
        pick_norm = max(lps, key=lambda k: lps[k]["logprob"] / max(lps[k]["bytes"], 1))
        rec = {"id": item["id"], "source_id": item["source_id"], "label": item["label"], "pick": pick,
               "pick_norm": pick_norm, "correct": pick == item["label"], "correct_norm": pick_norm == item["label"],
               "logprobs": lps}
        results.append(rec)
        out.write(json.dumps(rec) + "\n")
        out.flush()
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(items)}", flush=True)
    out.close()
    return results


def mcq_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    by = {lab: [r for r in rows if r["label"] == lab] for lab in W.LABELS}
    return {"note": "keel's approximation of When2Call MCQ scoring (not NVIDIA's harness); reference only",
            "n": n, "acc": round(sum(r["correct"] for r in rows) / max(n, 1), 4),
            "acc_norm": round(sum(r["correct_norm"] for r in rows) / max(n, 1), 4),
            "acc_by_label": {lab: round(sum(r["correct"] for r in v) / max(len(v), 1), 4) for lab, v in by.items()},
            "picks": {k: sum(r["pick"] == k for r in rows) for k in ("direct", *W.LABELS)}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
