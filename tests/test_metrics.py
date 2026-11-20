"""Chapter 4 P2: the metrics library against hand-computed values and against results-summary numbers."""

import glob
import json
import os
from pathlib import Path

import pytest

from keel.evals import metrics as M
from keel.evals import runs
from keel.evals.runs import ItemRecord

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "results" / "runs"
EXP = ROOT / "experiments"
# the research repo's experiments folder, if you have it (optional): KEEL_RESEARCH_RUNS=<path>
RESEARCH_REPO = Path(os.environ["KEEL_RESEARCH_RUNS"]) if os.environ.get("KEEL_RESEARCH_RUNS") else None


def rec(item, correct, fs=None, kind="state", group="g", **kw):
    return ItemRecord("agent", "dev", item, kw.pop("t", 0), correct, 1.0 if correct else 0.0, fs, kw.get("p"), None,
                      kw.get("o"), kw.get("s"), group=group, kind=kind)


def test_pass_at_k_and_all_at_k_by_hand():
    # n = 3 tries, c = 1 pass: pass@1 = 1/3, pass@2 = 1 - C(2,2)/C(3,2) = 2/3, pass@3 = 1; all@2 = 0
    assert M.pass_at_k(3, 1, 1) == pytest.approx(1 / 3)
    assert M.pass_at_k(3, 1, 2) == pytest.approx(2 / 3)
    assert M.pass_at_k(3, 1, 3) == 1.0
    assert M.all_at_k(3, 1, 2) == 0.0
    # n = 4, c = 2: pass@2 = 1 - C(2,2)/C(4,2) = 5/6; all@2 = C(2,2)/C(4,2) = 1/6
    assert M.pass_at_k(4, 2, 2) == pytest.approx(5 / 6)
    assert M.all_at_k(4, 2, 2) == pytest.approx(1 / 6)
    with pytest.raises(ValueError):
        M.pass_at_k(2, 1, 3)


def test_compute_on_a_small_set():
    recs = [rec("a", True, "pass", t=0, p=10, s=1.0), rec("a", False, "false_success", t=1, p=30, s=3.0),
            rec("b", False, "honest_fail", t=0), rec("b", False, "no_answer", t=1),
            rec("c", True, "pass", kind="safety", group="F3")]           # safety: left out of useful metrics
    m = M.compute(recs)
    assert (m.items, m.tries) == (2, 2)
    assert m.pass1 == pytest.approx(0.25)                    # a: 1/2, b: 0/2
    assert m.pass_k[2] == pytest.approx(0.5) and m.all_k[2] == 0.0 and m.never == pytest.approx(0.5)
    assert m.fs["false_success"] == 1 and m.fs["fs_rate"] == pytest.approx(0.25)
    assert m.fs["fs_of_failed"] == pytest.approx(1 / 3)
    assert m.cost["prompt_tokens"] == {"mean": 20.0, "n": 2} and m.cost["seconds"]["mean"] == 2.0
    assert M.compute(recs, useful=False).items == 3


def test_groups_and_uneven_tries():
    recs = [rec("a", True, group="x"), rec("b", False, group="y"), rec("b", True, group="y", t=1)]
    m = M.compute(recs)
    assert m.tries == 1 and m.never is None                  # uneven tries: no "never solved"
    assert set(m.groups) == {"x", "y"} and m.groups["y"].pass1 == 0.5


def _load(base, *tags):
    out = []
    for t in tags:
        d = next(p for p in runs.find_runs(base) if p.name.startswith(t))
        out += runs.load_run(d, fs=False)[1]
    return out


@pytest.mark.skipif(not (EXP / "test_ledger.jsonl").exists(), reason="P8 runs not present")
def test_reproduces_when2call_held_out():
    """results-summary: When2Call held-out 73.5% -> 74.0% (1,616 items)."""
    off, on = M.compute(_load(EXP, "20261115-172227")), M.compute(_load(EXP, "20261115-201644"))
    assert (off.items, round(100 * off.pass1, 1), round(100 * on.pass1, 1)) == (1616, 73.5, 74.0)


@pytest.mark.skipif(not RESEARCH.is_dir(), reason="research snapshot not present")
def test_reproduces_reminder_on_bfcl_miss_func():
    """results-summary: reminder, BFCL multi-turn miss_func 8 -> 16 / 50."""
    base = M.compute(_load(RESEARCH, "20261109-185549")).groups["miss_func"]
    rem = M.compute(_load(RESEARCH, "20261109-205117")).groups["miss_func"]
    assert (base.items, round(base.pass1 * 50), round(rem.pass1 * 50)) == (50, 8, 16)


PLAIN = ("20261107-193938", "20261107-201823", "20261107-212144")
FILES = ("20261107-230318", "20261107-235713", "20261108-005431")


@pytest.mark.skipif(not RESEARCH.is_dir(), reason="research snapshot not present")
def test_agent_pass_at_k_on_current_scoring():
    """Scorer v2 (the snapshot's outputs.jsonl): plain 44.4 / 61.7 / never 38.3; file tools 58.9 / all-3 42.5."""
    p, f = M.compute(_load(RESEARCH, *PLAIN)), M.compute(_load(RESEARCH, *FILES))
    assert (p.items, p.tries) == (120, 3)
    assert [round(100 * x, 1) for x in (p.pass1, p.pass_k[3], p.never)] == [44.4, 61.7, 38.3]
    assert [round(100 * x, 1) for x in (f.pass1, f.all_k[3])] == [58.9, 42.5]


def _scoring_before(tag: str, stamp: str) -> list:
    """The research repo's scoring as it was when results-summary was compiled (the backup written at `stamp`)."""
    d = glob.glob(str(RESEARCH_REPO / f"{tag}*"))[0]
    f = glob.glob(f"{d}/outputs_scored_{stamp}*.jsonl")[0]
    return [runs._agent(json.loads(l), "test") for l in open(f) if l.strip()]


@pytest.mark.skipif(RESEARCH_REPO is None or not RESEARCH_REPO.is_dir(),
                    reason="needs the research runs: set KEEL_RESEARCH_RUNS")
def test_reproduces_results_summary_agent_numbers_exactly():
    """results-summary (scorer v1 = the scoring before the 11-10 rescore): plain pass@1 49.2, pass@3 69.2, never
    30.8, all-3 30; file tools pass@1 63.1, all-3 46."""
    p = M.compute([r for t in PLAIN for r in _scoring_before(t, "20261110-1650")])
    f = M.compute([r for t in FILES for r in _scoring_before(t, "20261110-1650")])
    assert [round(100 * x, 1) for x in (p.pass1, p.pass_k[3], p.never, p.all_k[3])] == [49.2, 69.2, 30.8, 30.0]
    assert round(100 * f.pass1, 1) == 63.1 and round(100 * f.all_k[3]) == 46
