"""Checks for the Module 6 layers in the agent loop (no model: scripted replies drive real episodes / sandboxes).

  uv run python tests/checks/check_m6_layers.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

from keel.agent.loop import CHECK, _apply_reply, _new_episodes
from keel.evals import tasks as T

fails = 0


def check(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {str(detail)[:600]}"))
    fails += not cond


def call(name, **params):
    body = "\n".join(f"<parameter={k}>\n{v}\n</parameter>" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}\n</function>\n</tool_call>"


def episode(out, verify="none", recovery=False, task="C1", seed=0):
    return _new_episodes([T.build(task, seed)], out / f"{task}_{verify}_{recovery}_{seed}", "approve_all", "files",
                         verify, recovery)[0]


def main():
    out = Path(tempfile.mkdtemp())

    # 1. verify=none: unchanged behaviour (a final answer ends the episode)
    e = episode(out)
    _apply_reply(e, call("write_file", path="invoices.csv", content="date,vendor,total"), "stop", 20)
    _apply_reply(e, "Done! I created invoices.csv.", "stop", 20)
    check("none: final answer ends episode", e.done and e.stop == "answer" and e.answer.startswith("Done"), e.stop)
    check("none: no check sent", not any(m["role"] == "user" and "Automatic check" in m["content"]
                                         for m in e.messages))

    # 2. verify=self (V0): first answer intercepted, check has no report; second answer is final
    e = episode(out, "self")
    _apply_reply(e, call("write_file", path="invoices.csv", content="date,vendor,total"), "stop", 20)
    _apply_reply(e, "Done! first answer", "stop", 20)
    last = e.messages[-1]
    check("self: not done after first answer", not e.done)
    check("self: check message sent as user", last["role"] == "user" and last["content"].startswith("[Automatic check"))
    check("self: contains request, no report", e.item.request in last["content"]
          and "CHANGE REPORT" not in last["content"], last["content"])
    _apply_reply(e, "Final: all correct.", "stop", 20)
    check("self: second answer final", e.done and e.answer == "Final: all correct."
          and e.answer_before_check == "Done! first answer", (e.answer, e.answer_before_check))
    check("self: pre-check correctness measured (bool)", isinstance(e.correct_before_check, bool), e.correct_before_check)
    check("self: only one check", sum("Automatic check" in m["content"] for m in e.messages if m["role"] == "user") == 1)

    # 3. verify=state (V1): same words + change report from the real folder
    e = episode(out, "state")
    _apply_reply(e, call("write_file", path="invoices/invoices.csv", content="date,vendor,total\n2025-01-01,X,1"),
                 "stop", 20)
    _apply_reply(e, "Done!", "stop", 20)
    msg = e.messages[-1]["content"]
    check("state: same check words as V0", msg.startswith(CHECK.format(request=e.item.request)), msg[:300])
    check("state: report lists the new file and its head", "CHANGE REPORT" in msg and "invoices/invoices.csv" in msg
          and "2025-01-01,X,1" in msg, msg)
    check("state: no host paths in report", "/Users/" not in msg, msg)

    # 4. V1 claim check: claims a change, nothing changed
    e = episode(out, "state", seed=1)
    _apply_reply(e, "Done! I've created invoices.csv.", "stop", 20)
    check("state: false-claim note", "your answer describes changes, but no file was changed" in e.messages[-1]["content"],
          e.messages[-1]["content"])

    # 5. check gets no extra steps: limit reached during the check -> earlier answer stands
    e = episode(out, "state", seed=2)
    for _ in range(17):
        _apply_reply(e, call("list_files", path="."), "stop", 20)
    _apply_reply(e, "Done! answer before check", "stop", 20)          # step 18 -> check
    _apply_reply(e, call("list_files", path="."), "stop", 20)          # step 19
    _apply_reply(e, call("list_files", path="."), "stop", 20)          # step 20 -> limit
    check("step limit during check: earlier answer stands", e.done and e.stop == "answer"
          and e.answer == "Done! answer before check" and "check_cut_by_step_limit" in e.notes, (e.stop, e.notes))

    # 6. no check when the first answer comes at the last step
    e = episode(out, "state", seed=3)
    for _ in range(19):
        _apply_reply(e, call("read_file", path="invoices/" + "x.txt"), "stop", 20)
    _apply_reply(e, "Final at step 20", "stop", 20)
    check("no check at the last step", e.done and e.stop == "answer" and not e.checked, (e.stop, e.checked))

    # 7. R1 repeat detector: note after the 3rd identical call + result, once
    e = episode(out, recovery=True, seed=4)
    for _ in range(4):
        _apply_reply(e, call("list_files", path="."), "stop", 20)
    tools = [m["content"] for m in e.messages if m["role"] == "tool"]
    check("repeat: note on 3rd identical call only", ["same result 3 times" in t for t in tools]
          == [False, False, True, False], [t[-120:] for t in tools])

    # 8. R1 step budget: warning appended to the tool message at step 17 (3 left)
    e = episode(out, recovery=True, seed=5)
    for i in range(17):
        _apply_reply(e, call("read_file", path=f"nofile{i}.txt"), "stop", 20)
    check("budget: warning at 3 steps left", "3 steps left" in e.messages[-1]["content"]
          and e.messages[-1]["role"] == "tool" and "budget@17" in e.notes, e.notes)
    check("errors collected", len(e.errors) == 17, len(e.errors))

    # 9. recovery off: no notes
    e = episode(out, seed=6)
    for _ in range(17):
        _apply_reply(e, call("list_files", path="."), "stop", 20)
    check("recovery off: no notes", not e.notes and not any("Note from the system" in m["content"]
                                                            for m in e.messages), e.notes)

    shutil.rmtree(out)
    print("ALL CHECKS PASS" if not fails else f"{fails} CHECK(S) FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
