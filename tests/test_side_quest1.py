"""Side quest 1 (2026-11-16): the fixes found after Chapter 3. No model.

- cut-off tool calls: a reply cut at the length limit inside a <tool_call> is not run; the model is told why
- known bug 2: the guard's own bookkeeping no longer shows real times to the command (eval mode)
- known bug 3: verify="state" without a start folder is refused up front
- known bug 9: chat mode writes booleans the way the model does (True / False)
- open issue 13: a refused backend config is a protocol error, not a torn-down event loop
"""

import asyncio
import json

import pytest

from keel.agent.guard import GuardedShell, approve_all
from keel.agent.loop import CUT_CALL, TOOLSETS, Episode, _cut_calls, run_episodes_on
from keel.agent.sandbox import CLOCK_BASE, Sandbox
from keel.backends.chat import calls_text
from keel.backends.scripted import ScriptedBackend
from keel.core.parse import parse_calls
from keel.evals import tasks as T


def call(name, **params):
    body = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
                   for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


FUNCS = TOOLSETS["files"][2]
CUT = "Here is the report.\n<tool_call>\n<function=write_file>\n<parameter=path>\nreport.md\n</parameter>\n" \
      "<parameter=content>\n# Report\n\n| a | b |\n|---|---|\n| 1 | "      # cut mid-content, as in L12_10


# ---- cut-off tool calls ----------------------------------------------------------------------------------------

def test_cut_calls_counts_only_the_unclosed_block():
    full = call("list_files") + "\n" + CUT
    assert len(parse_calls(full, FUNCS)) == 2                     # the parser still sees the half call (unchanged)
    assert _cut_calls(full, FUNCS, "length") == 1
    assert _cut_calls(full, FUNCS, "stop") == 0                   # not cut by the limit: unchanged behaviour
    assert _cut_calls(call("list_files"), FUNCS, "length") == 0  # closed call at the limit: runs


def test_cut_call_is_not_run_and_the_model_is_told(tmp_path):
    it = T.build("C1", 0)
    be = ScriptedBackend({it.id: [(call("list_files") + "\n" + CUT, "length"), "Done."]})
    res = run_episodes_on(be, [it], tmp_path)
    work = tmp_path / "sandboxes" / it.id / "work"
    assert not (work / "report.md").exists()                      # the half-written call never ran
    tools = [m for m in be.prompts[1][1] if m["role"] == "tool"]
    assert len(tools) == 2 and not tools[0]["content"].startswith("Error")   # the complete call before it ran
    assert tools[1]["content"] == CUT_CALL
    assert "cut_call@1" in res[0]["notes"]
    assert not any(c.startswith("write_file") for c in res[0]["commands"])


def test_complete_reply_at_the_limit_still_runs(tmp_path):
    it = T.build("C1", 0)
    be = ScriptedBackend({it.id: [(call("write_file", path="a.txt", content="hi\n"), "length"), "Done."]})
    res = run_episodes_on(be, [it], tmp_path)
    assert (tmp_path / "sandboxes" / it.id / "work" / "a.txt").read_text() == "hi\n"
    assert not any(n.startswith("cut_call") for n in res[0]["notes"])


# ---- known bug 2: clock gap --------------------------------------------------------------------------------------

def test_guard_bookkeeping_does_not_show_real_time(tmp_path):
    sb = Sandbox(tmp_path / "ep").create(None)
    (sb.work / "a.txt").write_text("x")
    sb.settle_clock(0)                                            # everything at logical time 0
    sh = GuardedShell(sb, approve_all)
    r, _ = sh.run("stat -f %m ..; stat -f %m ../.tmp; stat -f %m ../.snap; stat -f %m ../..")
    stamps = [int(x) for x in r.output.split()]                   # root, .tmp, the guard's .snap, the parent
    assert stamps == [int(CLOCK_BASE)] * 4, stamps                # none shows the real time (SQ2: .snap, parent)


def test_in_place_keeps_real_times(tmp_path):
    folder = tmp_path / "real"
    folder.mkdir()
    sb = Sandbox.for_folder(folder, "s0123456789")
    try:
        assert sb.hold_times() == {}                              # real folders: nothing held or re-stamped
    finally:
        sb.remove()


# ---- known bug 3 ------------------------------------------------------------------------------------------------

def test_state_verify_needs_a_start_folder(tmp_path):
    sb = Sandbox(tmp_path / "ep").create(None)
    item = T.Item(task="x", seed=0, kind="session", request="", files={})
    with pytest.raises(ValueError, match="start_dir"):
        Episode(item, sb, GuardedShell(sb, approve_all), [], verify="state", start_dir=None)
    Episode(item, sb, GuardedShell(sb, approve_all), [], verify="self", start_dir=None)   # fine


# ---- known bug 9 ------------------------------------------------------------------------------------------------

def test_chat_booleans_written_like_the_model():
    text = calls_text([("list_files", {"recursive": True, "hidden": False})])
    assert "<parameter=recursive>\nTrue\n</parameter>" in text and "\nFalse\n" in text
    assert parse_calls(text, FUNCS)[0]["list_files"]["recursive"] is True


# ---- open issue 13 ----------------------------------------------------------------------------------------------

def test_refused_backend_is_a_protocol_error(tmp_path):
    from acp.exceptions import RequestError

    from keel.acp_server import KeelAgent

    def refuse():
        raise SystemExit("raw mode renders keel's prompt and needs the model's tokenizer folder")
    agent = KeelAgent(refuse, tmp_path / "state")
    with pytest.raises(RequestError) as err:
        asyncio.run(agent.new_session(cwd=str(tmp_path)))
    assert "tokenizer folder" in str(err.value.args[0] if err.value.args else err.value)
