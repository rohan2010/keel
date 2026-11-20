"""keel-core eval: run any suite as a named arm, with tries and repeats, and write one record shape.

  uv run keel-core eval --list
  uv run keel-core eval --suite agent --arm base --split dev --per-task 1
  uv run keel-core eval --suite bfcl-mt --arm reminder --split dev --limit 3 --reminder
  uv run keel-core eval --suite agent --arm base --split dev --repeat 2          (noise band, P5)
  uv run keel-core eval --suite bfcl-mt --arm t07 --temperature 0.7 --tries 4    (pass@k)

Every flag other than --suite / --arm / --tries / --repeat goes to the suite's own command unchanged (`keel-core run`,
`bfcl-mt`, `when2call`), which keeps its own outputs, holdout refusal and test ledger. eval only adds three files to
each run folder (`eval.json`, `header.json`, `records.jsonl`) and a manifest `experiments/evals/<id>.json` listing the
runs of this arm. Differences from calling a suite directly, on purpose:
- greedy unless --temperature is given (the suites default to 0.7; P9 lost a run to that);
- eval names the runs (`<arm>[_t<try>][_r<repeat>]`); --name is refused;
- --tries k needs sampling (k greedy tries are the same run k times) and gives try t the seed `seed + t`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from keel.evals import runs as R
from keel.runlog import EXPERIMENTS, HOLDOUT_CONFIRM, LEDGER, SEALED_SPLITS

SUITES = ("agent", "bfcl-mt", "when2call")
EVALS = EXPERIMENTS / "evals"


def _suite(name: str):
    """The suite's command module, imported only when used (bfcl-mt / when2call need the BFCL checker)."""
    from keel import cli
    return cli.load({"agent": "run", "bfcl-mt": "bfcl-mt", "when2call": "when2call"}[name])


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--list", action="store_true", help="show suites, splits, item counts and which tiers are sealed")
    ap.add_argument("--suite", choices=SUITES)
    ap.add_argument("--arm", help="a short name for this configuration (names the runs)")
    ap.add_argument("--tries", type=int, default=1, help="independent sampled tries per item (pass@k); seed + t")
    ap.add_argument("--repeat", type=int, default=1, help="run the identical config this many times (noise band)")
    ap.set_defaults(_extra=[])


def suite_namespace(suite: str, argv: list[str]) -> argparse.Namespace:
    mod = _suite(suite)
    ap = argparse.ArgumentParser(prog=f"keel-core eval --suite {suite}", add_help=False)
    mod.add_arguments(ap)
    return ap.parse_args(argv)


def plan(a: argparse.Namespace) -> list[dict]:
    """The runs this eval will make: [{try, repeat, argv}], after the checks. No model is touched."""
    if not a.suite or not a.arm:
        raise SystemExit("eval needs --suite and --arm (or --list)")
    if not a.arm.replace("-", "").replace("_", "").isalnum():
        raise SystemExit("--arm: letters, digits, - and _ only (it names the run folders)")
    extra = list(a._extra)
    if "--name" in extra or any(x.startswith("--name=") for x in extra):
        raise SystemExit("eval names the runs from --arm; drop --name")
    if a.tries < 1 or a.repeat < 1:
        raise SystemExit("--tries and --repeat must be at least 1")
    ns = suite_namespace(a.suite, ["--name", "x", *extra])   # validates the suite flags before anything runs
    samples = hasattr(ns, "temperature") and hasattr(ns, "seed")   # When2Call is greedy by its pre-registration
    if samples and not any(x == "--temperature" or x.startswith("--temperature=") for x in extra):
        extra += ["--temperature", "0"]                    # greedy unless asked otherwise (suites default to 0.7)
        ns.temperature = 0.0
    if a.tries > 1 and not (samples and ns.temperature):
        raise SystemExit("--tries > 1 needs sampling (--temperature > 0 on a suite that samples): greedy tries are "
                         "the same run again")
    base_seed = getattr(ns, "seed", 0)
    out = []
    for rep in range(a.repeat):
        for t in range(a.tries):
            name = a.arm + (f"_t{t}" if a.tries > 1 else "") + (f"_r{rep}" if a.repeat > 1 else "")
            argv = ["--name", name, *extra]
            if a.tries > 1:
                argv += ["--seed", str(base_seed + t)]
            out.append({"try": t, "repeat": rep, "argv": argv})
    return out


def write_records(run_dir: Path, meta: dict) -> dict:
    """eval.json, header.json and records.jsonl next to the suite's own outputs (which are not touched)."""
    (run_dir / R.EVAL_META).write_text(json.dumps(meta, indent=1))
    h, recs = R.load_run(run_dir)
    (run_dir / "header.json").write_text(json.dumps(h.to_dict(), indent=1, default=str))
    (run_dir / "records.jsonl").write_text("".join(json.dumps(r.to_dict()) + "\n" for r in recs))
    return {"run": run_dir.name, "items": len(recs), "passed": sum(r.correct for r in recs), "complete": h.complete}


def run(a: argparse.Namespace):
    if a.list:
        return print_list()
    runs_planned = plan(a)
    eval_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{a.arm}"
    manifest = {"id": eval_id, "suite": a.suite, "arm": a.arm, "tries": a.tries, "repeat": a.repeat,
                "argv": list(a._extra), "runs": []}
    EVALS.mkdir(parents=True, exist_ok=True)
    mod = _suite(a.suite)
    for p in runs_planned:
        ns = suite_namespace(a.suite, p["argv"])
        run_dir = mod.run(ns)
        meta = {"eval": eval_id, "suite": a.suite, "arm": a.arm, "try": p["try"], "repeat": p["repeat"],
                "tries": a.tries, "repeats": a.repeat, "argv": p["argv"]}
        manifest["runs"].append({**meta, **write_records(Path(run_dir), meta)})
        (EVALS / f"{eval_id}.json").write_text(json.dumps(manifest, indent=1))
    if a.repeat > 1:                                       # the repeats of one config give its noise band
        manifest["noise"] = record_noise(manifest, a.tries)
        (EVALS / f"{eval_id}.json").write_text(json.dumps(manifest, indent=1))
    print(f"eval {eval_id}: {len(manifest['runs'])} run(s) -> {EVALS / (eval_id + '.json')}")
    for r in manifest["runs"]:
        print(f"  try {r['try']} repeat {r['repeat']}: {r['passed']}/{r['items']} passed  {r['run']}")
    return EVALS / f"{eval_id}.json"


def record_noise(manifest: dict, tries: int) -> list[dict]:
    """Repeat 0 against every later repeat (all tries of a repeat together, paired try by try)."""
    from keel.evals import noise
    from keel.evals.compare import load_arm
    by_rep: dict[int, list[str]] = {}
    for r in manifest["runs"]:
        by_rep.setdefault(r["repeat"], []).append(str(EXPERIMENTS / r["run"]))
    first = load_arm(",".join(by_rep[0]), fs=False)
    out = []
    for rep in sorted(by_rep)[1:]:
        m = noise.record(first, load_arm(",".join(by_rep[rep]), fs=False))
        print(f"noise band (repeat 0 vs {rep}): {m['flips']} of {m['n']} flipped, +-{100 * m['band']:.1f} points")
        out.append({k: m[k] for k in ("n", "flips", "flip_rate", "band", "net")})
    return out


# ---- --list ------------------------------------------------------------------------------------------------------

def _ledger_counts() -> dict:
    counts: dict = {}
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (row.get("script", "run_agent"), row.get("split"))
            counts[key] = counts.get(key, 0) + 1
    return counts


def suite_table() -> list[dict]:
    """One row per suite and split: items, tier, and whether the tier is sealed (ledgered / refused)."""
    from keel.evals import tasks as T
    rows = [{"suite": "agent", "split": s, "tier": R.TIERS[s], "items": f"{len(T.TASKS)} tasks x --per-task seeds "
             f"({rng})", "sealed": s in SEALED_SPLITS, "script": "run_agent"}
            for s, rng in (("dev", "seeds 0-999"), ("test", "seeds 1000-4999"), ("holdout", "seeds 5000+"))]
    try:
        from keel.evals import bfcl_mt as M
        for s in ("dev", "val", "holdout"):
            n = {c: len(M.load(c, s)) for c in M.CATEGORIES}
            rows.append({"suite": "bfcl-mt", "split": s, "tier": R.TIERS[s], "items": f"{sum(n.values())} "
                         f"({', '.join(f'{c} {k}' for c, k in n.items())})", "sealed": s in SEALED_SPLITS,
                         "script": "run_bfcl_mt"})
    except ImportError:
        rows.append({"suite": "bfcl-mt", "split": "-", "tier": "-", "items": "bfcl-eval not installed "
                     "(keel-core install-bfcl)", "sealed": False, "script": "run_bfcl_mt"})
    try:
        from keel.evals import when2call as W
        for s in W.SPLITS:
            rows.append({"suite": "when2call", "split": s, "tier": R.TIERS[s], "items": str(len(W.load(s))),
                         "sealed": s in SEALED_SPLITS, "script": "run_when2call"})
    except (FileNotFoundError, OSError, ValueError, SystemExit):   # load() says "not found / re-fetch" via SystemExit
        rows.append({"suite": "when2call", "split": "-", "tier": "-", "items": "data not fetched "
                     "(uv run python scripts/fetch_when2call.py)", "sealed": False, "script": "run_when2call"})
    led = _ledger_counts()
    for r in rows:
        r["ledgered_runs"] = led.get((r["script"], r["split"]), 0)
    return rows


def print_list() -> None:
    rows = suite_table()
    print(f"{'suite':10} {'split':9} {'tier':11} {'sealed':7} {'ledger':6}  items")
    for r in rows:
        print(f"{r['suite']:10} {r['split']:9} {r['tier']:11} {'yes' if r['sealed'] else 'no':7} "
              f"{r['ledgered_runs']:<6}  {r['items']}")
    print(f"\nsealed = every run is logged in {LEDGER.name} with a --purpose; holdout also needs "
          f"--confirm-holdout {HOLDOUT_CONFIRM}. Agent task cards: task-cards/README.md.")
