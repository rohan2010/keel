"""Module 3 layers (src/keel/nudges.py): reply classification, to-do list, loop breaker, and both layers inside a real
BFCL multi-turn episode driven by canned replies (no model)."""

import copy

import pytest

from keel.nudges import LOOP_NOTE, LoopBreaker, TodoList, classify_reply


@pytest.mark.parametrize("text,kind", [
    ("Done! I created the folder and moved the file.", "done"),
    ("Done. Is there anything else I can help you with?", "done"),
    ("I moved it. Let me know if you need any other changes?", "done"),
    ("I'm ready to help! What would you like me to do?", "ask"),
    ("Which file do you mean?", "ask"),
    ("I can't move files because there is no move function.", "cant"),
    ("I don't have a tool to move files.", "cant"),
    ("I appreciate the update, but I don't see any new functions added.", "cant"),
    ("", "done"),
    ("Your vehicle cannot make the trip without refueling.", "done"),
    ("The engine cannot start until the doors are locked.", "done"),
    ("It appears there's no login function available.", "cant"),
    ("Unfortunately, I don't have a direct tool to count lines.", "cant"),
    # smoke fix 2026-11-11: quoted "?" is not a question back (long_context_111)
    ('I\'ve sent your message. The message "What are the new stock inclusion and the status of my current order?" '
     'has been delivered to Jane (USR002).', "done"),
    ("Sent “Are we still on for 3pm?” to Bob.", "done"),
    ("I wrote `why?` to notes.md.", "done"),
    ('I sent "Meeting moved?" to Jane. Which time do you prefer?', "ask"),
])
def test_classify_reply(text, kind):
    assert classify_reply(text) == kind


def test_todo_list_rules():
    t = TodoList()
    assert t.note_for_turn() == ""
    t.add(0, "list files")
    assert t.close_turn(0, "Here they are.", True)["cleared"] == [0]
    assert t.note_for_turn() == ""
    t.add(1, "move log.txt into archive")
    assert t.close_turn(1, "I can't, there is no move function.", True)["open"] == [1]
    note = t.note_for_turn()                     # holdout turn: item 1 shown
    assert '"move log.txt into archive"' in note
    rec = t.close_turn(2, "Done, moved it.", made_calls=False)   # no calls: shown item stays open
    assert rec["open"] == [1]
    t.note_for_turn()
    assert t.close_turn(3, "Done, moved it.", made_calls=True)["open"] == []


def test_todo_item_shown_at_most_twice():
    t = TodoList()
    t.add(0, "r0")
    t.close_turn(0, "Which one?", False)
    assert '"r0"' in t.note_for_turn()
    t.close_turn(1, "Which?", False)
    assert '"r0"' in t.note_for_turn()
    t.close_turn(2, "Which?", False)
    assert t.note_for_turn() == ""                 # shown twice: dropped
    assert t.items[0]["expired"]


def test_todo_list_caps_items():
    t = TodoList()
    for k in range(5):
        t.add(k, f"r{k}")
        t.close_turn(k, "Which one?", False)
    note = t.note_for_turn()
    assert '"r1"' not in note and '"r2"' in note and '"r4"' in note


def test_loop_breaker():
    lb = LoopBreaker()
    assert [lb.see("ls()", "x") for _ in range(4)] == [False, False, True, False]   # once per turn
    lb.new_turn()
    assert not lb.see("ls()", "x")
    for c in ["cd(a)", "cd(..)", "cd(a)", "cd(..)"]:                                 # cycles count too
        lb.see(c, "ok")
    assert lb.see("cd(a)", "ok")


def _episode(**kw):
    from keel.evals import bfcl_mt as M
    e = copy.deepcopy([r for r in M.load("miss_func") if r["id"] == "multi_turn_miss_func_1"][0])
    return M.MTEpisode(e, M.safe_name(f"test_m3_{len(kw)}_{sorted(kw)}"), **kw).start()


CALL = "<tool_call>\n<function=ls>\n<parameter=a>\nTrue\n</parameter>\n</function>\n</tool_call>"


def test_off_by_default_changes_nothing():
    from bfcl_eval.constants.default_prompts import DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC as HOLD
    e = _episode()
    for r in [CALL, CALL, CALL, "Here is the listing.", "I can't move files: there is no move function."]:
        e.apply(r, "stop")                                           # 3 identical calls: no loop note when off
    assert e.messages[-1]["content"] == HOLD and not e.notes and not e.todo_log and not e.ban_tool_call
    assert not any("Note from the system" in m["content"] for m in e.messages)


def test_reminder_in_episode():
    from bfcl_eval.constants.default_prompts import DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC as HOLD
    e = _episode(reminder=True)
    e.apply(CALL, "stop")
    e.apply("Here is the listing.", "stop")                          # turn 0 done -> cleared
    assert e.todo_log[-1]["cleared"] == [0]
    assert e.messages[-1]["content"].startswith("Go to workspace")   # turn 1: nothing earlier open, no note
    assert "Still unfinished" not in e.messages[-1]["content"]
    e.apply("I can't move files: there is no move function.", "stop")
    assert e.todo_log[-1]["open"] == [1]
    holdout = e.messages[-1]["content"]                               # turn 2 = holdout: note on BFCL's message
    assert holdout.startswith(HOLD) and "move one of the 'log.txt' files" in holdout
    assert e.notes == ["reminder@t2:1"]


def test_loop_breaker_in_episode():
    e = _episode(loop_break=True)
    e.apply(CALL, "stop")
    e.apply(CALL, "stop")
    assert not e.ban_tool_call
    e.apply(CALL, "stop")
    assert e.ban_tool_call and e.messages[-1]["content"].endswith(LOOP_NOTE)
    assert e.notes[-1].startswith("loop@t0")
    e.apply("Final answer: here is the listing.", "stop")            # the banned reply is text: turn ends
    assert not e.ban_tool_call and e.turn == 1
