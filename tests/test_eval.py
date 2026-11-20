"""Ch4 P1: keel-core eval (planning, refusals, the record files it adds, CLI passthrough). No model."""

import argparse
import json

import pytest

from keel import cli
from keel.commands import run_eval as E
from keel.evals import runs as R


def ns(suite, extra, tries=1, repeat=1, arm="base"):
    return argparse.Namespace(suite=suite, arm=arm, tries=tries, repeat=repeat, _extra=list(extra), list=False)


def test_greedy_unless_asked():
    p = E.plan(ns("agent", ["--split", "dev"]))
    assert p == [{"try": 0, "repeat": 0, "argv": ["--name", "base", "--split", "dev", "--temperature", "0"]}]
    p = E.plan(ns("agent", ["--temperature", "0.6"]))
    assert p[0]["argv"].count("--temperature") == 1 and "0.6" in p[0]["argv"]


def test_tries_get_their_own_seed_and_repeats_their_own_name():
    p = E.plan(ns("bfcl-mt", ["--temperature", "0.7", "--seed", "10"], tries=2, repeat=2))
    assert [(x["try"], x["repeat"], x["argv"][1], x["argv"][-1]) for x in p] == [
        (0, 0, "base_t0_r0", "10"), (1, 0, "base_t1_r0", "11"), (0, 1, "base_t0_r1", "10"), (1, 1, "base_t1_r1", "11")]


def test_when2call_stays_greedy_and_gets_no_temperature_flag():
    p = E.plan(ns("when2call", ["--split", "dev", "--limit", "2"]))
    assert "--temperature" not in p[0]["argv"]


@pytest.mark.parametrize("args, msg", [
    (dict(suite="agent", extra=[], tries=3), "needs sampling"),
    (dict(suite="when2call", extra=[], tries=2), "needs sampling"),
    (dict(suite="agent", extra=["--name", "x"]), "drop --name"),
    (dict(suite="agent", extra=[], arm="a/b"), "letters, digits"),
    (dict(suite="agent", extra=[], repeat=0), "at least 1"),
])
def test_refusals(args, msg):
    with pytest.raises(SystemExit, match=msg):
        E.plan(ns(args["suite"], args["extra"], args.get("tries", 1), args.get("repeat", 1), args.get("arm", "base")))


def test_unknown_suite_flag_refused_before_running():
    with pytest.raises(SystemExit):
        E.plan(ns("bfcl-mt", ["--bogus", "1"]))


def test_cli_passes_unknown_flags_only_to_eval(monkeypatch):
    seen = {}
    monkeypatch.setattr(E, "run", lambda a: seen.update(extra=a._extra, suite=a.suite))
    monkeypatch.setitem(cli.COMMANDS, "eval", (E, cli.COMMANDS["eval"][1]))
    cli.main(["eval", "--suite", "agent", "--arm", "x", "--split", "dev", "--reminder"])
    assert seen == {"extra": ["--split", "dev", "--reminder"], "suite": "agent"}
    with pytest.raises(SystemExit):
        cli.main(["run", "--name", "x", "--bogus"])


def test_write_records_adds_three_files_and_sets_try(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"kind": "agent", "split": "dev", "items": ["A1_0"]}))
    row = {"id": "A1_0", "task": "A1", "kind": "state", "seed": 0, "correct": True, "score": 1.0, "stop": "answer",
           "answer": "done", "steps": 1}
    (d / "outputs.jsonl").write_text(json.dumps(row) + "\n")
    before = (d / "outputs.jsonl").read_bytes()
    s = E.write_records(d, {"eval": "e1", "arm": "a", "try": 2, "repeat": 0})
    assert s == {"run": "run", "items": 1, "passed": 1, "complete": True}
    assert (d / "outputs.jsonl").read_bytes() == before                    # the suite's file is untouched
    rec = json.loads((d / "records.jsonl").read_text())
    assert rec["try"] == 2 and rec["item_id"] == "A1_0"
    assert json.loads((d / "header.json").read_text())["suite"] == "agent"
    assert R.load_run(d)[1][0].try_ == 2


def test_list_table_has_every_suite():
    rows = E.suite_table()
    assert {r["suite"] for r in rows} == {"agent", "bfcl-mt", "when2call"}
    assert all(r["sealed"] for r in rows if r["tier"] in ("holdout", "validation"))


def test_repeats_record_a_noise_band(tmp_path, monkeypatch):
    from test_compare import make

    from keel.evals import noise
    exp = tmp_path / "experiments"
    exp.mkdir()
    monkeypatch.setattr(E, "EXPERIMENTS", exp)
    monkeypatch.setattr(noise, "_noise_dir", lambda: exp / "noise")
    r0 = make(exp, "base_r0", [True] * 10 + [False] * 10)
    r1 = make(exp, "base_r1", [False] * 3 + [True] * 7 + [False] * 10)
    manifest = {"runs": [{"run": "base_r0", "repeat": 0}, {"run": "base_r1", "repeat": 1}]}
    out = E.record_noise(manifest, tries=1)
    assert out[0]["flips"] == 3 and out[0]["n"] == 20
    from keel.evals.compare import load_arm
    assert noise.lookup(load_arm(r0).header)["flips"] == 3 and r1


def test_list_table_without_when2call_data(monkeypatch, tmp_path):
    """Ch7 P9 (fresh-clone check): `eval --list` on a fresh install, where When2Call was never fetched, must list it
    as not fetched instead of crashing on the loader's SystemExit."""
    import keel.evals.when2call as W
    real = W.load
    monkeypatch.setattr(W, "load", lambda split=None, path=None, check_sha=True: real(split, tmp_path / "none.jsonl"))
    rows = E.suite_table()
    assert [r["items"] for r in rows if r["suite"] == "when2call"] == [
        "data not fetched (uv run python scripts/fetch_when2call.py)"]
