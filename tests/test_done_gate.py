"""Ch5 G1: concrete-signal checks that hold back "done" (keel.agent.done_gate + the agent loop). No model."""

import json
from pathlib import Path

from keel.agent import done_gate as G
from keel.agent.loop import run_episodes_on
from keel.backends.scripted import ScriptedBackend
from keel.evals import tasks as T


def call(name, **params):
    body = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
                   for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


# ---- the checks ---------------------------------------------------------------------------------------------------

def test_csv_field_counts_with_the_comma_hint(tmp_path):
    (tmp_path / "a.csv").write_text("date,vendor,total\nJul 24, 2025,Acme,10.00\n\"Jul 25, 2025\",Acme,5.00\n")
    p = G.file_problems(tmp_path, ["a.csv"])
    assert p == ["row 2 of a.csv has 4 fields, the header has 3 (a value that contains a comma must be in double "
                 "quotes)"]                                            # the quoted row 3 is fine


def test_json_yaml_html(tmp_path):
    (tmp_path / "a.json").write_text('{"a": 1,}')
    (tmp_path / "b.yaml").write_text("a: [1, 2\n")
    (tmp_path / "c.html").write_text('<a href="ok.html">x</a><a href="missing.html">y</a><a href="https://x">z</a>')
    (tmp_path / "ok.html").write_text("<p>ok</p>")
    p = G.file_problems(tmp_path, ["a.json", "b.yaml", "c.html", "ok.html"])
    assert any(x.startswith("a.json is not valid JSON") for x in p)
    assert any(x.startswith("b.yaml is not valid YAML") for x in p)
    assert "c.html links to missing.html, which doesn't exist" in p and len(p) == 3


def test_changed_files_sees_new_and_modified(tmp_path):
    (tmp_path / "keep.txt").write_text("x")
    (tmp_path / "edit.txt").write_text("x")
    before = G.snapshot(tmp_path)
    (tmp_path / "edit.txt").write_text("y")
    (tmp_path / "new.csv").write_text("a\n1\n")
    assert G.changed_files(tmp_path, before) == ["edit.txt", "new.csv"]


def test_gate_module_reads_nothing_task_specific():
    """Guardrail (plan v2.1): no layer may read a task id, a scorer or an answer key."""
    src = Path(G.__file__).read_text()
    for banned in ("import tasks", "from . import tasks", "agent.tasks", "item.task", "gold", ".score("):
        assert banned not in src, banned


# ---- in the agent loop ----------------------------------------------------------------------------------------------

BAD = "date,vendor,total\nJul 24, 2025,Acme,10.00\n"
GOOD = 'date,vendor,total\n"Jul 24, 2025",Acme,10.00\n'


def _run(tmp_path, task, replies, **kw):
    it = T.build(task, 0)
    res = run_episodes_on(ScriptedBackend({it.id: replies}), [it], tmp_path, **kw)
    msgs = json.loads((tmp_path / "transcripts" / f"{it.id}.json").read_text())["messages"]
    return res[0], msgs


def test_gate_holds_back_done_then_lets_the_fixed_answer_through(tmp_path):
    r, msgs = _run(tmp_path, "C1", [call("write_file", path="invoices.csv", content=BAD), "Done.",
                                    call("write_file", path="invoices.csv", content=GOOD), "Fixed and done."],
                   done_gate=True)
    gate = [m for m in msgs if m["role"] == "user" and m["content"].startswith("[keel check]")]
    assert len(gate) == 1 and "row 2 of invoices.csv has 4 fields" in gate[0]["content"]
    assert r["answer"] == "Fixed and done." and r["gate_rounds"] == 1 and r["answer_before_gate"] == "Done."
    assert r["correct_before_gate"] is not None and any(n.startswith("gate@2") for n in r["notes"])


def test_at_most_two_rounds_then_the_answer_stands(tmp_path):
    r, msgs = _run(tmp_path, "C1", [call("write_file", path="invoices.csv", content=BAD), "Done.", "Done!",
                                    "Really done."], done_gate=True)
    assert r["gate_rounds"] == 2 and r["answer"] == "Really done." and r["stop"] == "answer"


def test_edit_inside_a_longer_name_is_raised_at_done(tmp_path):
    it = T.build("D1", 0)
    page = next(p for p, b in it.files.items() if p.endswith(".html") and b"btn-primary-outline" in b)
    r, msgs = _run(tmp_path, "D1", [call("edit_file", path=page, old_text="btn-primary", new_text="button-main",
                                         replace_all=True), "Renamed everywhere."], done_gate=True)
    gate = [m["content"] for m in msgs if m["role"] == "user" and m["content"].startswith("[keel check]")]
    assert gate and "inside longer names" in gate[0] and "btn-primary-outline" in gate[0]


def test_failed_last_call_is_raised_unless_the_answer_admits_it(tmp_path):
    r, _ = _run(tmp_path, "C1", [call("read_file", path="nope.txt"), "All done."], done_gate=True)
    assert r["gate_rounds"] == 2                         # not fixed, so raised again (max 2), then the answer stands
    assert "failed and nothing after it succeeded" in r["gate_problems"][0][0]
    r, _ = _run(tmp_path / "b", "C1", [call("read_file", path="nope.txt"), "I couldn't read that file."],
                done_gate=True)
    assert r["gate_rounds"] == 0


def test_flag_off_records_and_messages_unchanged(tmp_path):
    r, msgs = _run(tmp_path, "C1", [call("write_file", path="invoices.csv", content=BAD), "Done."])
    assert "gate_rounds" not in r and not any(m["content"].startswith("[keel check]") for m in msgs
                                              if m["role"] == "user")


def test_whole_line_edits_are_not_raised(tmp_path):
    """Day 2: a whole-line edit that spells out the longer name is a choice, not an accident (G1 small run: it caused
    both blocked-correct answers). Only replace_all on one short name counts."""
    it = T.build("D1", 0)
    page = next(p for p, b in it.files.items() if p.endswith(".js") and b"querySelectorAll('.btn-primary')" in b)
    line = next(l for l in it.files[page].decode().splitlines() if "querySelectorAll('.btn-primary')" in l).strip()
    r, msgs = _run(tmp_path, "D1", [call("edit_file", path=page, old_text=line,
                                         new_text=line.replace("btn-primary", "button-main")), "Done."],
                   done_gate=True)
    assert not any(m["content"].startswith("[keel check]") for m in msgs if m["role"] == "user")
