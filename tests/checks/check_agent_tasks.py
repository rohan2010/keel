"""Scorer checks for agent tasks: for many seeds, the oracle must score 100% and doing nothing must score 0%
(the safety task F3 is the reverse: doing nothing keeps the files). Run before trusting any agent number.

  uv run python tests/checks/check_agent_tasks.py --seeds 20 [--long | --ch6]
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from keel.evals import tasks as T


def run(item, fn, base: Path):
    work = base / "work"
    if work.exists():
        shutil.rmtree(work)
    T.materialize(item, work)
    ans = fn(item, work) if fn else ""
    return T.score(item, work, ans, [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--long", action="store_true", help="check Module 5's long tasks (L01-L12) instead")
    ap.add_argument("--ch6", action="store_true", help="check Chapter 6's 12 new tasks (tasks_ch6.py) instead")
    a = ap.parse_args()
    from keel.evals.tasks_ch6 import CH6_TASKS
    from keel.evals.tasks_long import LONG_TASKS
    reg = CH6_TASKS if a.ch6 else LONG_TASKS if a.long else T.TASKS
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        for task in reg:
            o_ok = n_ok = 0
            for s in range(a.seeds):
                item = T.build(task, s)
                o = run(item, T.oracle, Path(d))
                n = run(item, None, Path(d))
                o_ok += o["correct"]
                n_ok += n["correct"]
                if not o["correct"]:
                    print(f"  oracle failed {item.id}: {o}")
            want_noop = a.seeds if task == "F3" else 0
            good = o_ok == a.seeds and n_ok == want_noop
            bad += not good
            print(f"{'PASS' if good else 'FAIL'} {task} {reg[task][3]:34s} oracle {o_ok}/{a.seeds}  "
                  f"do-nothing {n_ok}/{a.seeds} (want {want_noop})")
    print("ALL PASS" if not bad else f"{bad} TASKS FAILED")
    sys.exit(1 if bad else 0)       # was missing: failures used to exit 0


if __name__ == "__main__":
    main()
