"""Chapter 4 P6: pre-registration. Done-when: a run that doesn't match its pre-registration is refused the label."""

import json
from argparse import Namespace

import pytest

from keel.commands import run_compare
from keel.evals import prereg
from keel.evals.compare import load_arm

PLAN = """\
name: reminder-keep
suite: agent
split: dev
hypothesis: the reminder raises pass@1
arms:
  a: {arm: base}
  b: {arm: reminder, layers: {reminder: true}}
primary: pass@1
pair: item
sided: greater
alpha: 0.05
"""


def run_dir(tmp, name, outcomes, *, arm, started="2030-01-01 10:00:00", layers=None, split="dev"):
    d = tmp / name
    d.mkdir()
    ids = [f"T{i}_0" for i in range(len(outcomes))]
    cfg = {"kind": "agent", "split": split, "items": ids, "batch_size": 1, "started": started,
           "sampling": {"temperature": 0.0, "seed": 0}, **(layers or {})}
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "outputs.jsonl").write_text("".join(json.dumps({"id": i, "task": "T", "kind": "state", "seed": 0,
                                                         "correct": ok, "score": float(ok), "stop": "answer",
                                                         "answer": "ok", "steps": 1}) + "\n"
                                             for i, ok in zip(ids, outcomes, strict=True)))
    (d / "eval.json").write_text(json.dumps({"arm": arm, "try": 0}))
    return str(d)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(prereg, "_dir", lambda: tmp_path / "prereg")
    f = tmp_path / "plan.yaml"
    f.write_text(PLAN)
    rec = prereg.freeze(f)
    return f, rec


def arms(tmp, **kw):
    a = run_dir(tmp, "a", [False] * 20, arm="base", **kw.get("a", {}))
    b = run_dir(tmp, "b", [True] * 10 + [False] * 10, arm=kw.get("b_arm", "reminder"),
                layers={"reminder": True}, **kw.get("b", {}))
    return load_arm(a), load_arm(b)


def test_parse_lists_every_problem():
    with pytest.raises(ValueError) as err:
        prereg.parse("name: x y\nsuite: nope\nprimary: pass@3\nsided: maybe\narms: {a: {}}\nextra: 1\n")
    msg = str(err.value)
    for want in ("unknown keys: extra", "missing split", "missing alpha", "name:", "suite must", "primary must",
                 "sided", "arms: exactly a and b"):
        assert want in msg, want


def test_freeze_is_idempotent_and_refuses_changed_text(frozen, tmp_path):
    f, rec = frozen
    assert prereg.freeze(f)["frozen"] == rec["frozen"]
    f.write_text(PLAN.replace("0.05", "0.10"))
    with pytest.raises(ValueError, match="already frozen with different text"):
        prereg.freeze(f)


def test_matching_runs_get_the_label(frozen, tmp_path):
    _, rec = frozen
    A, B = arms(tmp_path)
    assert prereg.check(rec, A, B) == []


@pytest.mark.parametrize("kw, why", [
    (dict(a={"started": "2020-01-01 00:00:00"}), "before the freeze"),
    (dict(b_arm="other"), "is not eval arm 'reminder'"),
    (dict(a={"split": "test"}), "split test"),
])
def test_mismatching_runs_are_refused_the_label(frozen, tmp_path, kw, why):
    _, rec = frozen
    A, B = arms(tmp_path, **kw)
    problems = prereg.check(rec, A, B)
    assert any(why in p for p in problems), problems


def test_wrong_layer_is_refused(frozen, tmp_path):
    _, rec = frozen
    a = run_dir(tmp_path, "a", [False] * 20, arm="base")
    b = run_dir(tmp_path, "b", [True] * 20, arm="reminder", layers={"reminder": False})
    assert any("reminder = False" in p for p in prereg.check(rec, load_arm(a), load_arm(b)))


def test_edited_file_is_refused(frozen, tmp_path):
    f, _ = frozen
    f.write_text(PLAN + "notes: changed my mind\n")
    rec = prereg.load(str(f))
    A, B = arms(tmp_path)
    assert "the file was edited after it was frozen" in prereg.check(rec, A, B)


def _cli(a, b, plan):
    return Namespace(a=a, b=b, alpha=0.05, sided="two", pair="item", group=None, allow=[], no_fs=True, json=None,
                     out=None, no_report=True, prereg=plan)


def test_compare_uses_the_frozen_settings_and_labels_the_verdict(frozen, tmp_path, capsys):
    A, B = arms(tmp_path)
    c = run_compare.run(_cli(A.header.dir, B.header.dir, "reminder-keep"))
    assert c.preregistered and c.sided == "greater" and "pre-registered" in capsys.readouterr().out
    assert any("matched" in f for f in c.flags)


def test_compare_with_a_mismatch_stays_exploratory(frozen, tmp_path):
    A, B = arms(tmp_path, a={"started": "2020-01-01 00:00:00"})
    c = run_compare.run(_cli(A.header.dir, B.header.dir, "reminder-keep"))
    assert not c.preregistered and any(f.startswith("NOT pre-registered") for f in c.flags)
