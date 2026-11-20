"""Chapter 4 P3: keel compare verdicts on synthetic runs, the refusals, both pairings, and three published results."""

import json
import os
from pathlib import Path

import pytest

from keel.evals import runs
from keel.evals.compare import compare_arms, load_arm, mcnemar_exact, mcnemar_mde

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "results" / "runs"
EXP = ROOT / "experiments"

DONE, ADMIT = "Done! Everything is in place.", "I couldn't finish this; the file was missing."


def make(tmp, name, outcomes, *, batch=1, seed=0, env=None, items=None, answers=None):
    """outcomes: list of True/False per item i (ids t<i>_0); answers: list of final answers (default honest)."""
    d = tmp / name
    d.mkdir()
    ids = [f"T{i}_0" for i in range(len(outcomes))]
    cfg = {"kind": "agent", "split": "dev", "items": items or ids, "batch_size": batch, "model": {"name": "m"},
           "sampling": {"temperature": 0.0, "seed": seed}, "env": env or {"mlx": "0.32.4"}}
    (d / "config.json").write_text(json.dumps(cfg))
    rows = [{"id": i, "task": i.split("_")[0], "kind": "state", "seed": 0, "correct": ok, "score": float(ok),
             "stop": "answer", "answer": "Done." if ok else (answers[k] if answers else ADMIT), "steps": 3,
             "output_tokens": 10 + k} for k, (i, ok) in enumerate(zip(ids, outcomes, strict=True))]
    (d / "outputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(d)


def verdict(a, b, **kw):
    return compare_arms(load_arm(a), load_arm(b), **kw)


def test_mcnemar_and_mde_by_hand():
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert mcnemar_exact(2, 10) == pytest.approx(0.0386, abs=1e-4)
    assert mcnemar_exact(4, 27, "greater") == pytest.approx(1.7e-5, rel=0.05)
    assert mcnemar_mde(12, 0) == pytest.approx(6 / 12)          # 0 flips seen: 6 one-way flips would pass
    assert mcnemar_mde(4, 0) is None                              # 4 items can never reach p < 0.05


def test_real_gain(tmp_path):
    c = verdict(make(tmp_path, "a", [False] * 20), make(tmp_path, "b", [True] * 10 + [False] * 10))
    assert c.verdict == "real gain" and c.pass_test["only_b"] == 10 and c.pass_test["delta"] == 0.5


def test_real_loss(tmp_path):
    c = verdict(make(tmp_path, "a", [True] * 10 + [False] * 10), make(tmp_path, "b", [False] * 20))
    assert c.verdict == "real loss"


def test_no_detectable_effect_gives_the_detectable_size(tmp_path):
    a = [True, True, False, False] + [True] * 16
    b = [False, False, True, True] + [True] * 16
    c = verdict(make(tmp_path, "a", a), make(tmp_path, "b", b))
    assert c.verdict == "no detectable effect" and "could only detect" in c.reason
    assert c.identical == 16


def test_gain_with_more_false_done(tmp_path):
    a = make(tmp_path, "a", [False] * 30)                                     # every failure admits it
    b = make(tmp_path, "b", [True] * 10 + [False] * 20, answers=[DONE] * 30)  # failures now claim success
    c = verdict(a, b)
    assert c.verdict == 'real gain, more false "done"' and c.fs_test["significant"] and c.fs_test["delta"] > 0


@pytest.mark.parametrize("kw, key", [(dict(batch=8), "batch_size"), (dict(env={"mlx": "0.31.0"}), "env"),
                                     (dict(seed=1), "sampling")])
def test_cant_compare_lists_the_reason(tmp_path, kw, key):
    c = verdict(make(tmp_path, "a", [True] * 10), make(tmp_path, "b", [True] * 10, **kw))
    assert c.verdict == "can't compare" and key in c.reason and not c.pass_test


def test_allow_accepts_a_difference_on_purpose(tmp_path):
    c = verdict(make(tmp_path, "a", [False] * 10), make(tmp_path, "b", [True] * 10, env={"mlx": "0.31.0"}),
                allow=("env",))
    assert c.verdict == "real gain" and c.allowed == ["env"]


def test_incomplete_run_cant_be_compared(tmp_path):
    a = make(tmp_path, "a", [True] * 3, items=["T0_0", "T1_0", "T2_0", "T3_0"])
    b = make(tmp_path, "b", [True] * 3, items=["T0_0", "T1_0", "T2_0", "T3_0"])
    c = verdict(a, b)
    assert c.verdict == "can't compare" and "incomplete" in c.reason


def test_tries_bootstrap_and_try_pairing(tmp_path):
    a = ",".join(make(tmp_path, f"a{s}", [False] * 20, seed=s) for s in (0, 1))
    b = ",".join(make(tmp_path, f"b{s}", [True] * 12 + [False] * 8, seed=s) for s in (0, 1))
    c = verdict(a, b)
    assert c.pass_test["test"] == "paired bootstrap" and c.verdict == "real gain" and c.pass_test["ci95"][0] > 0
    c = verdict(a, b, pair="try")
    assert c.pass_test["test"] == "exact McNemar" and c.pass_test["n"] == 40 and c.pass_test["only_b"] == 24
    bad = ",".join(make(tmp_path, f"c{s}", [True] * 12 + [False] * 8, seed=s) for s in (5, 6))
    assert "try seeds" in verdict(a, bad, pair="try", allow=("sampling",)).reason


def test_runs_of_one_arm_must_be_tries_of_one_config(tmp_path):
    with pytest.raises(ValueError, match="not another try"):
        load_arm(make(tmp_path, "a", [True] * 4) + "," + make(tmp_path, "b", [True] * 4, batch=4))


# ---- published results ------------------------------------------------------------------------------------------

def _d(base, tag):
    return str(next(p for p in runs.find_runs(base) if tag in p.name))


@pytest.mark.skipif(not RESEARCH.is_dir(), reason="research snapshot not present")
def test_reproduces_validated_reminder():
    """Milestone 1: F > F minus reminder, 27 vs 4 of 100, exact one-sided p = 1.7e-5."""
    no = _d(RESEARCH, "m1_mf_noreminder_s0") + "," + _d(RESEARCH, "m1_mf_noreminder_s1")
    f = _d(RESEARCH, "m1_mf_F_s0") + "," + _d(RESEARCH, "m1_mf_F_s1")
    c = compare_arms(load_arm(no, fs=False), load_arm(f, fs=False), pair="try", sided="greater")
    t = c.pass_test
    assert (t["only_a"], t["only_b"], t["n"]) == (4, 27, 100) and t["p"] == pytest.approx(1.7e-5, rel=0.05)
    assert c.tier == "validation" and c.verdict.startswith("real gain")


@pytest.mark.skipif(not RESEARCH.is_dir(), reason="research snapshot not present")
def test_reproduces_bfcl_reminder_dev():
    """Module 3: reminder on BFCL miss_func 8 -> 16 / 50, 10 rescued / 2 broken, p = 0.039 (dev tier)."""
    c = compare_arms(load_arm(_d(RESEARCH, "20261109-185549"), fs=False),
                     load_arm(_d(RESEARCH, "20261109-205117"), fs=False), group="miss_func")
    assert (c.pass_test["only_a"], c.pass_test["only_b"]) == (2, 10) and round(c.pass_test["p"], 3) == 0.039
    assert c.tier == "dev"


@pytest.mark.skipif(not (EXP / "test_ledger.jsonl").exists(), reason="P8 runs not present")
def test_reproduces_when2call_no_effect():
    """P8: When2Call held-out gate off vs on, 66 broken / 75 fixed, p = 0.50: no detectable effect."""
    c = compare_arms(load_arm(_d(EXP, "20261115-172227")), load_arm(_d(EXP, "20261115-201644")))
    assert (c.pass_test["only_a"], c.pass_test["only_b"]) == (66, 75) and c.verdict == "no detectable effect"
    assert round(c.pass_test["p"], 2) == 0.50


# the research repo's experiments folder, if you have it (optional): KEEL_RESEARCH_RUNS=<path>
RESEARCH_REPO = Path(os.environ["KEEL_RESEARCH_RUNS"]) if os.environ.get("KEEL_RESEARCH_RUNS") else None


def research_arm(tags, stamp=None):
    """An arm from research runs, with the scoring saved at `stamp` (a rescore backup), else the current one."""
    import glob

    from keel.evals.compare import Arm
    hs, recs = [], []
    for i, t in enumerate(tags):
        d = glob.glob(str(RESEARCH_REPO / f"{t}*"))[0]
        f = glob.glob(f"{d}/outputs_scored_{stamp}*.jsonl")[0] if stamp else f"{d}/outputs.jsonl"
        hs.append(runs.header(RESEARCH / Path(d).name))
        for line in open(f):
            if line.strip():
                r = runs._agent(json.loads(line), "test")
                r.try_ = i
                recs.append(r)
    return Arm(hs[0].name, hs, recs)


@pytest.mark.skipif(RESEARCH_REPO is None or not RESEARCH_REPO.is_dir(),
                    reason="needs the research runs: set KEEL_RESEARCH_RUNS")
def test_reproduces_file_tools_vs_shell():
    """Module 2 (pre-registered, test seeds): pass@1 +13.9 points [+7.2, +20.6], paired over items, 3 tries each."""
    plain = ("20261107-193938", "20261107-201823", "20261107-212144")
    files = ("20261107-230318", "20261107-235713", "20261108-005431")
    c = compare_arms(research_arm(plain, "20261110-1650"), research_arm(files, "20261110-1650"))
    t = c.pass_test
    assert t["test"] == "paired bootstrap" and c.verdict == "real gain" and c.tier == "validation"
    assert [round(100 * x, 1) for x in (t["delta"], *t["ci95"])] == [13.9, 7.2, 20.6]


def test_one_sided_bootstrap_significance_follows_p():
    """Audit #12: with tries (bootstrap path), `significant` must agree with the one-sided p at alpha."""
    import random as _r

    from keel.evals.compare import paired_test
    rng = _r.Random(3)
    for shift in (0.0, 0.08, 0.15, 0.3):
        a = {f"i{k}": [rng.random() < 0.5 for _ in range(2)] for k in range(60)}
        b = {k: [rng.random() < 0.5 + shift for _ in range(2)] for k in a}
        for sided in ("greater", "two"):
            t = paired_test(a, b, 0.05, sided=sided)
            assert t["significant"] == (t["p"] < 0.05), (shift, sided, t["p"])
