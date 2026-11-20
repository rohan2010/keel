"""Chapter 4 P0: run folders of every suite and code version load into one shape (keel.evals.runs)."""

import json
from pathlib import Path

import pytest

from keel.evals import runs
from keel.false_success import first_failing_turn

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "results" / "runs"
needs_runs = pytest.mark.skipif(not RESEARCH.is_dir(), reason="research run snapshot not present")


def write_run(d: Path, config: dict, rows: list[dict]) -> Path:
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config))
    (d / "outputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


AGENT_ROW = {"id": "A1_0", "task": "A1", "kind": "state", "seed": 0, "correct": False, "score": 0.5, "stop": "answer",
             "answer": "Done! All files are sorted.", "steps": 3, "prompt_tokens": 100, "output_tokens": 20,
             "seconds": 1.0}


def test_agent_run_header_and_records(tmp_path):
    d = write_run(tmp_path / "r", {"kind": "agent", "model": {"name": "m", "revision": "abc"}, "split": "dev",
                                   "items": ["A1_0", "B1_0"], "batch_size": 1, "sampling": {"temperature": 0.0},
                                   "code": {"combined": "c0ffee"}, "loop_break": True}, [AGENT_ROW])
    h, recs = runs.load_run(d)
    assert (h.suite, h.tier, h.backend, h.code, h.batch_size) == ("agent", "dev", "mlx", "c0ffee", 1)
    assert h.layers == {"loop_break": True} and h.missing == ["B1_0"] and not h.complete
    r = recs[0]
    assert (r.item_id, r.try_, r.correct, r.score, r.group) == ("A1_0", 0, False, 0.5, "A1")
    assert r.fs_label == "false_success"                       # failed, but the answer claims success


def test_rescore_backups_are_not_read(tmp_path):
    """outputs.jsonl is the current scoring; outputs_scored_* hold EARLIER scorings and must be ignored."""
    d = write_run(tmp_path / "r", {"kind": "agent", "split": "dev", "items": ["A1_0"]},
                  [{**AGENT_ROW, "correct": True, "score": 1.0}])
    (d / "outputs_scored_20261110-165026.jsonl").write_text(json.dumps(AGENT_ROW) + "\n")
    _, recs = runs.load_run(d)
    assert recs[0].correct and recs[0].fs_label == "pass"


def test_unreadable_runs_say_why(tmp_path):
    with pytest.raises(runs.RunUnreadable, match="no config.json"):
        runs.header(tmp_path)
    d = write_run(tmp_path / "e", {"kind": "eval"}, [])
    with pytest.raises(runs.RunUnreadable, match="not an eval suite"):
        runs.load_run(d)
    d = write_run(tmp_path / "z", {"kind": "agent", "split": "dev"}, [])
    with pytest.raises(runs.RunUnreadable, match="empty"):
        runs.load_run(d)


def test_server_backend_and_when2call(tmp_path):
    d = write_run(tmp_path / "w", {"kind": "when2call", "split": "held_out", "items": ["x"],
                                   "server": {"backend": "server-raw:openai:http://h:1"}},
                  [{"id": "x", "correct": True, "label": "tool_call", "prompt_tokens": 5, "output_tokens": 2,
                    "seconds": 0.1, "finish_reason": "stop"}])
    h, recs = runs.load_run(d)
    assert (h.tier, h.backend) == ("holdout", "server-raw:openai:http://h:1")
    assert recs[0].fs_label is None and recs[0].group == "tool_call"


@needs_runs
def test_every_saved_research_run_of_a_known_suite_loads():
    """P0 done-when: all saved agent / BFCL / When2Call runs load; the only refusals are the two empty ones."""
    bad = []
    for d in runs.find_runs(RESEARCH):
        try:
            runs.load_run(d, fs=False)
        except runs.RunUnreadable as err:
            if "not an eval suite" not in str(err):
                bad.append(d.name)
    assert sorted(bad) == ["20261108-105321_bfcl_mt_smoke_qwen3.5-4b-8bit", "20261108-184135_m6dev_C_v1_qwen3.5-4b-8bit"]


@needs_runs
def test_first_failing_turn_does_not_depend_on_earlier_checks():
    """The official checker keeps executor state per model name; a reused name gave turn 2 then turn 1."""
    pytest.importorskip("bfcl_eval")
    from keel.evals import bfcl_mt as M
    d = next(p for p in runs.find_runs(RESEARCH) if p.name.startswith("20261109-185549"))
    r = next(json.loads(l) for l in (d / "outputs.jsonl").open() if '"multi_turn_miss_func_121"' in l)
    e = next(x for x in M.load("miss_func") if x["id"] == r["id"])
    assert [first_failing_turn(r, e) for _ in range(3)] == [2, 2, 2]


@needs_runs
def test_reproduces_the_published_false_success_finding():
    """results-summary finding 1: reminder vs base, 14 became false successes and 4 stopped (100 entries)."""
    pytest.importorskip("bfcl_eval")
    lab = {}
    for tag in ("20261109-185549", "20261109-205117"):
        d = next(p for p in runs.find_runs(RESEARCH) if p.name.startswith(tag))
        lab[tag] = {r.item_id: r.fs_label for r in runs.load_run(d)[1]}
    a, b = lab["20261109-185549"], lab["20261109-205117"]
    became = sum(a[i] != "false_success" and b[i] == "false_success" for i in a)
    stopped = sum(a[i] == "false_success" and b[i] != "false_success" for i in a)
    assert (became, stopped) == (14, 4)
