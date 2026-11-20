"""When2Call eval (Ch3 P8): splits, scoring, summary, McNemar, MCQ candidates. No model."""

import json

import pytest

from keel.evals import when2call as W

FUNCS = [{"name": "get_weather", "description": "", "parameters": {"type": "dict", "required": ["city"],
                                                                   "properties": {"city": {"type": "string"}}}}]
CALL = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"


def item(label, target="get_weather", **kw):
    return {"id": kw.get("id", "x"), "source_id": "live_simple_0-0-0", "label": label, "question": "q",
            "functions": FUNCS, "target": target if label != "cannot_answer" else None, "answers": {}}


def test_splits_follow_preregistered_rules():
    # the research half rule: sha1(id) % 2 == 0 -> practice -> "seen"
    import hashlib
    for sid in ("live_multiple_0-0-0", "live_simple_12-3-4", "live_multiple_99-1-0"):
        h = int(hashlib.sha1(sid.encode()).hexdigest(), 16)
        if h % 2 == 0:
            assert W.split_of(sid) == "seen"
        else:
            d = int(hashlib.sha1(f"when2call-dev:{sid}".encode()).hexdigest(), 16) % 10 == 0
            assert W.split_of(sid) == ("dev" if d else "held_out")


@pytest.mark.skipif(not W.MCQ_FILE.exists(), reason="When2Call not fetched")
def test_split_counts_match_preregistration():
    items = W.load()
    count = {s: sum(i["split"] == s for i in items) for s in W.SPLITS}
    assert count == {"dev": 170, "held_out": 1616, "seen": 1866}
    for s in ("dev", "held_out"):          # a question never straddles dev and held-out
        ids = {i["source_id"] for i in items if i["split"] == s}
        assert not ids & {i["source_id"] for i in items if i["split"] != s}
    tc = next(i for i in items if i["label"] == "tool_call")
    assert tc["target"] in {f["name"] for f in tc["functions"]}
    assert set(W.mcq_candidates(tc)) == {"direct", "tool_call", "request_for_info", "cannot_answer"}


def test_score_call_decision():
    assert W.score(item("tool_call"), CALL) == {"called": True, "correct": True, "n_calls": 1,
                                                "first_call": "get_weather", "right_tool": True}
    s = W.score(item("tool_call", target="other"), CALL)
    assert s["correct"] and s["right_tool"] is False            # decision right, tool wrong
    assert W.score(item("tool_call"), "Which city?")["correct"] is False
    assert W.score(item("request_for_info"), "Which city?")["correct"] is True
    assert W.score(item("cannot_answer"), CALL)["correct"] is False
    assert W.score(item("request_for_info"), CALL)["right_tool"] is None


def test_summary_rates():
    rows = [{**W.score(item("tool_call"), CALL), "label": "tool_call", "gate": "force"},
            {**W.score(item("tool_call"), "no"), "label": "tool_call", "gate": "block"},
            {**W.score(item("request_for_info"), CALL), "label": "request_for_info", "gate": "force"},
            {**W.score(item("cannot_answer"), "no"), "label": "cannot_answer", "gate": "free"}]
    s = W.summarize(rows)
    assert s["n"] == 4 and s["accuracy"] == 0.5
    assert s["under_call"] == 0.5 and s["over_call"] == 0.5
    assert s["over_call_request_for_info"] == 1.0 and s["over_call_cannot_answer"] == 0.0
    assert s["right_tool_when_called"] == 1.0
    assert s["gate"]["tool_call"] == {"force": 1, "block": 1, "free": 0}


def test_mcnemar_exact():
    assert W.mcnemar_exact(0, 0) == 1.0
    assert W.mcnemar_exact(5, 5) == 1.0
    assert W.mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert W.mcnemar_exact(2, 10) == pytest.approx(2 * (1 + 12 + 66) / 4096)


def test_compare_pairs_by_id():
    a = [{"id": "1", "correct": True, "text": "a"}, {"id": "2", "correct": False, "text": "b"}]
    b = [{"id": "2", "correct": True, "text": "c"}, {"id": "1", "correct": True, "text": "a"}]
    c = W.compare(a, b)
    assert (c["only_a_correct"], c["only_b_correct"], c["identical_replies"]) == (0, 1, 1)
    with pytest.raises(ValueError):
        W.compare(a, b[:1])


def test_qwen_call_text_parses_back():
    t = W.qwen_call_text({"name": "get_weather", "arguments": {"city": "Paris", "days": 3}})
    from keel.core.parse import parse_calls
    assert parse_calls(t, FUNCS + [{"name": "x"}]) == [{"get_weather": {"city": "Paris", "days": "3"}}]
    it = {"answers": {"direct": "d", "tool_call": json.dumps({"name": "get_weather", "arguments": {}}),
                      "request_for_info": "r", "cannot_answer": "c"}}
    assert W.mcq_candidates(it)["tool_call"].startswith("<tool_call>\n<function=get_weather>")


def test_cli_refusals():
    from keel.cli import main
    with pytest.raises(SystemExit, match="purpose"):
        main(["when2call", "--name", "x", "--split", "held_out"])
    with pytest.raises(SystemExit, match="gate cannot act"):
        main(["when2call", "--name", "x", "--mcq", "--probe-gate"])
