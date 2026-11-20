"""keel-core compare: a verdict on two arms.

  uv run keel-core compare experiments/<runA> experiments/<runB>
  uv run keel-core compare A_s0,A_s1 B_s0,B_s1 --pair try --sided greater      (matched tries, one-sided)
  uv run keel-core compare experiments/evals/<idA>.json experiments/evals/<idB>.json
  uv run keel-core compare A B --group miss_func                               (one BFCL category / agent task)
  uv run keel-core compare A B --cluster task --sided greater                  (tasks as the units)

B is the change under test; deltas are B minus A. Verdicts: real gain | real loss | real gain, more false "done" |
no detectable effect | can't compare | not tested (fixed sequence; with every reason). See keel.evals.compare for the rules.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from keel.evals import report
from keel.evals.compare import compare_arms, load_arm
from keel.runlog import EXPERIMENTS

COMPARES = EXPERIMENTS / "compares"


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("a", help="baseline arm: run folder, eval manifest, or run folders joined by commas (tries)")
    ap.add_argument("b", help="arm under test, same forms")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--sided", choices=("two", "greater"), default="two",
                    help="two-sided (default); 'greater' = B > A, only when a pre-registration says so")
    ap.add_argument("--pair", choices=("item", "try"), default="item",
                    help="item: an item's tries together (bootstrap if several); try: try t of A vs try t of B "
                         "(needs matching seeds)")
    ap.add_argument("--cluster", choices=("task",), default=None,
                    help="task: decide with the task-clustered sign-flip permutation (tasks, not episodes, are the "
                         "units); the item-level test is shown beside it")
    ap.add_argument("--group", default=None, help="compare on one group only (BFCL category, agent task)")
    ap.add_argument("--allow", nargs="*", default=[], metavar="KEY",
                    help="header differences to accept on purpose (listed as allowed), e.g. env")
    ap.add_argument("--no-fs", action="store_true", help="skip false-success labels (faster on big BFCL runs)")
    ap.add_argument("--prereg", default=None, metavar="NAME_OR_FILE",
                    help="a frozen pre-registration: its pair / sided / alpha / group are used, and the verdict is "
                         "labelled pre-registered only if the runs match it")
    ap.add_argument("--json", default=None, metavar="FILE", help="also write the full comparison as JSON")
    ap.add_argument("--out", default=None, help="report folder (default experiments/compares/<time>_<A>_vs_<B>)")
    ap.add_argument("--no-report", action="store_true", help="print the card only; write no report folder")


def _short(name: str) -> str:
    """`20261109-185549_m3_confirm_base_qwen3.5-4b-8bit` -> `m3_confirm_base`."""
    parts = name.split("_", 1)
    core = parts[1] if len(parts) == 2 and parts[0][:8].isdigit() else name
    return re.sub(r"_qwen[\w.\-]*$", "", core)[:40]


def report_dir(c) -> Path:
    a, b = (Path(c.runs[k][0]).name for k in ("a", "b"))
    tail = f"_{c.group}" if c.group else ""
    return COMPARES / f"{time.strftime('%Y%m%d-%H%M%S')}_{_short(a)}_vs_{_short(b)}{tail}"


def run(a: argparse.Namespace):
    A, B = load_arm(a.a, fs=not a.no_fs), load_arm(a.b, fs=not a.no_fs)
    from keel.evals import noise, prereg
    settings = {"pair": a.pair, "sided": a.sided, "alpha": a.alpha, "group": a.group, "cluster": getattr(a, "cluster", None),
                "primary": "pass@1"}
    problems, rec = [], None
    if a.prereg:
        try:
            rec = prereg.load(a.prereg)
        except ValueError as err:
            raise SystemExit(str(err)) from None
        settings = prereg.test_settings(rec)             # the frozen plan decides the test, not the command line
        problems = prereg.check(rec, A, B)
    band = None if settings["group"] else (noise.lookup(A.header) or noise.lookup(B.header))   # baseline first
    allow = tuple(a.allow) + tuple(settings.pop("allow", ()))   # a frozen plan's accepted differences, plus the CLI's
    c = compare_arms(A, B, allow=allow, noise_band=band, preregistered=bool(rec) and not problems, **settings)
    if rec:
        c.flags[:0] = ([f"pre-registration {rec['name']} (frozen {rec['frozen']}) matched"] if not problems else
                       [f"NOT pre-registered: {p}" for p in problems])
    if rec and not problems:
        prereg.record_result(rec, c)
        gs = prereg.gate_status(rec)
        if gs and gs["state"] == "pending":
            c.flags.append(f"fixed sequence: tested only after {gs['gate']} is confirmed ({gs['why']}); not final")
        elif gs and gs["state"] != "confirmed":
            c.flags.append(f"fixed sequence: NOT TESTED, {gs['gate']} was not confirmed")
            c.reason = f"not tested: {gs['gate']} (the hypothesis before it) was not confirmed; " + c.reason
            c.verdict = "not tested"            # used to print "no detectable effect" as the verdict line
        elif gs:
            c.flags.append(f"fixed sequence: {gs['gate']} confirmed, so this hypothesis is tested at its full alpha")
        h = prereg.holm(rec)
        if h and h["pending"]:
            c.flags.append(f"Holm over {', '.join(rec['plan']['family'])}: waiting for {', '.join(h['pending'])}; "
                           "this verdict is not final")
        elif h:
            me = h["results"][rec["name"]]
            c.flags.append(f"Holm over {len(h['results'])} hypotheses: p = {me['p']:.3g} vs threshold "
                           f"{me['threshold']:.3g}: {'confirmed' if me['confirmed'] else 'NOT confirmed'}")
            if not me["confirmed"] and c.verdict in ("real gain", "real loss", 'real gain, more false "done"'):
                c.reason += f"; not confirmed after Holm over {len(h['results'])} hypotheses"
                c.verdict = "no detectable effect"
    print(report.terminal(c))
    if not a.no_report:
        out = report.write(c, Path(a.out) if a.out else report_dir(c))
        md = out / "report.md"
        try:
            md = md.relative_to(Path.cwd())
        except ValueError:
            pass
        print(f"\nreport: {md}")
    if a.json:
        Path(a.json).write_text(json.dumps(c.to_dict(), indent=1, default=str))
    return c
