"""Chapter 5 close-out: the task-clustered test in `keel compare`, and pre-registrations that freeze the deciding test
and a Holm family. Done-when: the tool and the card can no longer disagree (run 11: compare called both arms "real
gain" on items while the card's clustered test with Holm confirmed neither)."""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from keel.commands import run_compare
from keel.evals import prereg
from keel.evals.compare import cluster_test, compare_arms, load_arm


def run_dir(tmp, name, table, *, arm, started="2030-01-01 10:00:00"):
    """table: task -> list of outcomes (one per seed)."""
    d = tmp / name
    d.mkdir()
    rows = [(f"{t}_{s}", t, ok) for t, oks in table.items() for s, ok in enumerate(oks)]
    cfg = {"kind": "agent", "split": "test", "items": [r[0] for r in rows], "batch_size": 1, "started": started,
           "sampling": {"temperature": 0.7, "seed": 0}}
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "outputs.jsonl").write_text("".join(json.dumps({"id": i, "task": t, "kind": "state", "seed": 0, "correct": ok,
                                                         "score": float(ok), "stop": "answer", "answer": "ok",
                                                         "steps": 1}) + "\n" for i, t, ok in rows))
    (d / "eval.json").write_text(json.dumps({"arm": arm, "try": 0}))
    return str(d)


def counts(**kw):
    """task=(passed of 4)."""
    return {t: [True] * n + [False] * (4 - n) for t, n in kw.items()}


# run 11 per task (passed of 4): keel, full, full without the email sentence
KEEL = counts(A1=3, A3=3, A4=2, B1=4, C1=0, C3=3, D1=1, D2=4, E1=4, E3=0, F1=0, F2=4)
FULL = counts(A1=2, A3=4, A4=4, B1=4, C1=2, C3=3, D1=1, D2=3, E1=4, E3=2, F1=4, F2=4)
NONOTE = counts(A1=3, A3=4, A4=3, B1=4, C1=3, C3=4, D1=1, D2=4, E1=4, E3=2, F1=0, F2=4)


def test_cluster_test_reproduces_run_11(tmp_path):
    k = load_arm(run_dir(tmp_path, "k", KEEL, arm="val_keel"))
    f = load_arm(run_dir(tmp_path, "f", FULL, arm="val_full"))
    n = load_arm(run_dir(tmp_path, "n", NONOTE, arm="val_full_nonote"))
    h1 = compare_arms(k, f, sided="greater", cluster="task", noise_band={})
    h2 = compare_arms(k, n, sided="greater", cluster="task", noise_band={})
    assert round(h1.pass_test["p"], 4) == 0.0781 and round(h2.pass_test["p"], 4) == 0.0312
    assert h1.pass_test["moved"] == 7 and h1.pass_test["clusters"] == 12
    assert h1.verdict == "no detectable effect" and h2.verdict == "real gain"
    # the item test is kept beside it and would have passed: the run-11 disagreement, now visible on the card
    assert h1.item_test["significant"] and "would pass" in " ".join(h1.flags)


def test_cluster_test_basics():
    a = {k: [False] for k in ("x0", "x1", "y0", "y1")}
    b = {k: [True] for k in ("x0", "x1", "y0", "y1")}
    g = {"x0": "x", "x1": "x", "y0": "y", "y1": "y"}
    t = cluster_test(a, b, g, 0.05, "greater")
    assert t["p"] == 0.25 and t["exact"]                     # 2 tasks moved: 1 of 4 sign patterns is as extreme
    assert cluster_test(a, a, g, 0.05)["p"] == 1.0           # nothing moved
    assert cluster_test(a, b, g, 0.05, "two")["p"] == 0.5


def test_cluster_needs_two_tasks(tmp_path):
    k = load_arm(run_dir(tmp_path, "k", {"A1": [False] * 4}, arm="x"))
    f = load_arm(run_dir(tmp_path, "f", {"A1": [True] * 4}, arm="y"))
    with pytest.raises(ValueError, match="two tasks"):
        compare_arms(k, f, cluster="task", noise_band={})


PLAN = """\
name: {name}
suite: agent
split: test
hypothesis: b raises pass@1
arms:
  a: {{arm: val_keel}}
  b: {{arm: {arm}}}
primary: pass@1
sided: greater
alpha: 0.05
test: task-clustered
family: [h1, h2]
"""


def test_parse_checks_test_and_family():
    with pytest.raises(ValueError, match="test:"):
        prereg.parse(PLAN.format(name="h1", arm="b").replace("task-clustered", "bayes"))
    with pytest.raises(ValueError, match="family"):
        prereg.parse(PLAN.format(name="h3", arm="b"))        # the family must include this plan
    assert prereg.parse(PLAN.format(name="h1", arm="b").replace("test: task-clustered\n", ""))["test"] == "item"


def ns(a, b, plan):
    return Namespace(a=a, b=b, alpha=0.05, sided="two", pair="item", group=None, cluster=None, allow=[], no_fs=True,
                     json=None, out=None, no_report=True, prereg=plan)


def test_frozen_family_applies_holm(tmp_path, monkeypatch):
    monkeypatch.setattr(prereg, "_dir", lambda: tmp_path / "prereg")
    for name, arm in (("h1", "val_full"), ("h2", "val_full_nonote")):
        f = tmp_path / f"{name}.yaml"
        f.write_text(PLAN.format(name=name, arm=arm))
        prereg.freeze(f)
    k = run_dir(tmp_path, "k", KEEL, arm="val_keel")
    f = run_dir(tmp_path, "f", FULL, arm="val_full")
    n = run_dir(tmp_path, "n", NONOTE, arm="val_full_nonote")
    c2 = run_compare.run(ns(k, n, "h2"))                     # first member: real gain so far, Holm pending
    assert c2.preregistered and c2.pass_test["test"].startswith("task-clustered")
    assert c2.verdict == "real gain" and any("waiting for h1" in x for x in c2.flags)
    c1 = run_compare.run(ns(k, f, "h1"))                     # family complete: 0.031 needs < 0.025 -> neither
    assert c1.verdict == "no detectable effect" and any("NOT confirmed" in x for x in c1.flags)
    c2b = run_compare.run(ns(k, n, "h2"))                    # re-reading h2 now gives the Holm verdict
    assert c2b.verdict == "no detectable effect" and "not confirmed after Holm" in c2b.reason
    # a pre-registered result is recorded once and never replaced
    rec = json.loads((tmp_path / "prereg" / "h2.result.json").read_text())
    assert round(rec["p"], 4) == 0.0312


# ---- Chapter 6 combined headline: fixed-sequence gating and false success as the deciding metric ------------------

GATED = """\
name: {name}
suite: agent
split: test
hypothesis: x
arms:
  a: {{arm: val_keel}}
  b: {{arm: {arm}}}
primary: {primary}
sided: greater
alpha: 0.05
test: task-clustered
{gate}"""


def run_dir_answers(tmp, name, table, *, arm, answer_fail):
    d = Path(run_dir(tmp, name, table, arm=arm))
    rows = [json.loads(l) for l in (d / "outputs.jsonl").read_text().splitlines()]
    for r in rows:
        if not r["correct"]:
            r["answer"] = answer_fail
    (d / "outputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(d)


def freeze_plan(tmp_path, name, arm, primary="pass@1", gate=""):
    f = tmp_path / f"{name}.yaml"
    f.write_text(GATED.format(name=name, arm=arm, primary=primary, gate=f"gate: {gate}\n" if gate else ""))
    prereg.freeze(f)


def test_gate_and_family_are_exclusive():
    with pytest.raises(ValueError, match="gate and family"):
        prereg.parse(PLAN.format(name="h1", arm="b") + "gate: h0\n")


def test_fixed_sequence_tests_the_second_only_after_the_first(tmp_path, monkeypatch):
    monkeypatch.setattr(prereg, "_dir", lambda: tmp_path / "prereg")
    freeze_plan(tmp_path, "p1", "val_full")                                   # primary: pass rate
    freeze_plan(tmp_path, "p2", "val_full", primary="false_success", gate="p1")
    # every failure in keel says "Done" (false success); every failure in full admits it
    k = run_dir_answers(tmp_path, "k", KEEL, arm="val_keel", answer_fail="Done, all files are sorted.")
    f = run_dir_answers(tmp_path, "f", NONOTE, arm="val_full", answer_fail="I couldn't finish this task.")
    c2 = run_compare.run(ns(k, f, "p2"))                    # before p1 has a result: tested later, not final
    assert any("tested only after p1" in x for x in c2.flags)
    c1 = run_compare.run(ns(k, f, "p1"))                    # p1: clustered p = 0.031 < 0.05, confirmed
    assert c1.verdict == "real gain"
    c2b = run_compare.run(ns(k, f, "p2"))
    assert any("p1 confirmed" in x for x in c2b.flags)
    assert c2b.pass_test["metric"] == "false_success" and c2b.pass_test["delta"] < 0
    assert c2b.verdict == 'fewer false "done"'


def test_failed_gate_means_not_tested(tmp_path, monkeypatch):
    monkeypatch.setattr(prereg, "_dir", lambda: tmp_path / "prereg")
    freeze_plan(tmp_path, "q1", "val_full")
    freeze_plan(tmp_path, "q2", "val_full", gate="q1")
    k = run_dir(tmp_path, "k", KEEL, arm="val_keel")
    f = run_dir(tmp_path, "f", KEEL, arm="val_full")       # no change at all: q1 fails
    run_compare.run(ns(k, f, "q1"))
    c = run_compare.run(ns(k, f, "q2"))
    assert c.verdict == "not tested" and c.reason.startswith("not tested")


def _brute(a, b, g, sided):
    """The definition, by enumerating every sign pattern (only for small k)."""
    from itertools import product
    by = {}
    for i in a:
        by.setdefault(g[i], []).append(sum(b[i]) / len(b[i]) - sum(a[i]) / len(a[i]))
    d = [sum(v) / len(v) for v in by.values()]
    moved = [abs(x) for x in d if abs(x) > 1e-12]
    obs = sum(d)
    sums = [sum(s * m for s, m in zip(signs, moved, strict=True)) for signs in product((1, -1), repeat=len(moved))]
    if not moved:
        return 1.0
    if sided == "greater":
        return sum(x >= obs - 1e-9 for x in sums) / len(sums)
    return min(1.0, sum(abs(x) >= abs(obs) - 1e-9 for x in sums) / len(sums))


def test_cluster_test_exact_matches_enumeration():
    import random
    rng = random.Random(7)
    for _ in range(200):
        tasks, seeds = rng.randint(1, 12), rng.choice([1, 2, 3, 4])
        a, b, g = {}, {}, {}
        for t in range(tasks):
            for s in range(seeds):
                i = f"T{t}_{s}"
                a[i], b[i], g[i] = [rng.random() < 0.5], [rng.random() < 0.6], f"T{t}"
        for sided in ("greater", "two"):
            assert cluster_test(a, b, g, 0.05, sided)["p"] == pytest.approx(_brute(a, b, g, sided), abs=1e-12)


def test_cluster_test_exact_above_20_tasks():
    """24 tasks x 4 seeds, all moved up by one seed: the old code went Monte Carlo here. Every task moves by the same
    amount, so the null is binomial: p = P(all 24 signs up) = 1 / 2^24, exactly."""
    a, b, g = {}, {}, {}
    for t in range(24):
        for s in range(4):
            i = f"T{t}_{s}"
            a[i], b[i], g[i] = [False], [s == 0], f"T{t}"
    r = cluster_test(a, b, g, 0.05, "greater")
    assert r["exact"] and r["moved"] == 24 and r["p"] == 1 / 2 ** 24
    # 22 up, 2 down by the same amount: P(at least 22 of 24 up) = (1 + 24 + 276) / 2^24
    for t in (0, 1):
        for s in range(4):
            i = f"T{t}_{s}"
            a[i], b[i] = [s == 0], [False]
    assert cluster_test(a, b, g, 0.05, "greater")["p"] == (1 + 24 + 276) / 2 ** 24
