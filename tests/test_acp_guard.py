"""P5: the delete guard over ACP. A delete happens only on an explicit allow-once for that exact question; every other
outcome keeps the files. Each test runs keel-core as a subprocess with the scripted backend and a real temp folder.
"""

import asyncio
import json
import subprocess
import sys
import time

import pytest

acp = pytest.importorskip("acp")
from acp import schema as s  # noqa: E402


def call(name, **params):
    body = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
                   for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


class Client:
    """Answers the delete question in one of many ways (good and bad)."""

    def __init__(self, mode):
        self.mode, self.updates, self.asked, self.conn, self.sid = mode, [], [], None, None

    def on_connect(self, conn):
        pass

    async def session_update(self, session_id, update, **kw):
        self.updates.append(update)

    async def request_permission(self, session_id, tool_call, options, **kw):
        self.asked.append((tool_call, options, len(self.updates)))
        by_kind = {o.kind: o.option_id for o in options}
        sel = lambda oid: s.RequestPermissionResponse(outcome=s.AllowedOutcome(outcome="selected", option_id=oid))
        if self.mode == "allow":
            return sel(by_kind["allow_once"])
        if self.mode == "reject":
            return sel(by_kind["reject_once"])
        if self.mode == "cancelled":
            return s.RequestPermissionResponse(outcome=s.DeniedOutcome(outcome="cancelled"))
        if self.mode == "unknown_option":
            return sel("allow-always-please")
        if self.mode == "other_call_id":
            return sel(by_kind["allow_once"].replace("call_", "call_9"))     # an allow id for a different call
        if self.mode == "error":
            raise RuntimeError("client crashed while asking")
        if self.mode == "session_cancel":
            await self.conn.cancel(session_id=session_id)
            await asyncio.sleep(30)                                         # never answers
        raise AssertionError(self.mode)


async def _run(tmp_path, mode, script):
    folder, state = tmp_path / "proj", tmp_path / "state"
    folder.mkdir()
    (folder / "old.log").write_text("log")
    (folder / "keep.txt").write_text("keep")
    sp = tmp_path / "script.json"
    sp.write_text(json.dumps(script))
    client = Client(mode)
    async with acp.spawn_agent_process(client, sys.executable, "-m", "keel.cli", "acp", "--backend", "scripted",
                                       "--script", str(sp), "--state-dir", str(state)) as (conn, proc):
        client.conn = conn
        await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)
        new = await conn.new_session(cwd=str(folder))
        r = await conn.prompt(session_id=new.session_id, prompt=[s.TextContentBlock(type="text", text="clean up")])
    rec = json.loads((state / "sessions" / new.session_id / "transcript.json").read_text())
    return folder, client, r, rec


DELETE_TOOL = [call("delete", paths=["old.log"]), "Ok."]
DELETE_RM = [call("run_shell", command="rm old.log"), "Ok."]


@pytest.mark.parametrize("script", [DELETE_TOOL, DELETE_RM], ids=["delete-tool", "rm"])
def test_allow_once_deletes_and_the_question_is_well_formed(tmp_path, script):
    folder, c, r, rec = asyncio.run(asyncio.wait_for(_run(tmp_path, "allow", script), 60))
    assert not (folder / "old.log").exists() and (folder / "keep.txt").exists()
    (tool_call, options, n_before), = c.asked
    assert sorted(o.kind for o in options) == ["allow_once", "reject_once"]          # never "always"
    assert tool_call.kind == "delete" and str(folder.resolve() / "old.log") in tool_call.raw_input["files"]
    starts = [u for u in c.updates[:n_before] if getattr(u, "session_update", "") == "tool_call"]
    assert starts and starts[-1].tool_call_id == tool_call.tool_call_id               # the card came first
    assert rec["confirmations"][0]["approved"] is True


@pytest.mark.parametrize("mode", ["reject", "cancelled", "unknown_option", "other_call_id", "error"])
@pytest.mark.parametrize("script", [DELETE_TOOL, DELETE_RM], ids=["delete-tool", "rm"])
def test_anything_but_allow_once_keeps_the_file(tmp_path, mode, script):
    folder, c, r, rec = asyncio.run(asyncio.wait_for(_run(tmp_path, mode, script), 60))
    assert (folder / "old.log").read_text() == "log", mode
    assert len(c.asked) == 1 and rec["confirmations"][0]["approved"] is False
    tool_msgs = [m["content"] for m in rec["messages"] if m["role"] == "tool"]
    assert any("declined" in t.lower() for t in tool_msgs), tool_msgs                # the model is told


def test_session_cancel_while_asking_keeps_the_file(tmp_path):
    t0 = time.time()
    folder, c, r, rec = asyncio.run(asyncio.wait_for(_run(tmp_path, "session_cancel", DELETE_TOOL), 60))
    assert (folder / "old.log").exists() and r.stop_reason == "cancelled"
    assert time.time() - t0 < 25                                                     # did not wait for an answer


def test_client_killed_mid_question_keeps_the_file_and_the_core_exits(tmp_path):
    """Like killing the TUI while the delete prompt is on screen."""
    folder, state = tmp_path / "proj", tmp_path / "state"
    folder.mkdir()
    (folder / "old.log").write_text("log")
    sp = tmp_path / "script.json"
    sp.write_text(json.dumps([call("delete", paths=["old.log"]), call("write_file", path="after.txt", content="x"),
                              "Ok."]))
    proc = subprocess.Popen([sys.executable, "-m", "keel.cli", "acp", "--backend", "scripted", "--script", str(sp),
                             "--state-dir", str(state)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    send = lambda m: (proc.stdin.write(json.dumps(m) + "\n"), proc.stdin.flush())
    send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": 1}})
    send({"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {"cwd": str(folder), "mcpServers": []}})
    sid = None
    while True:
        m = json.loads(proc.stdout.readline())
        if m.get("id") == 2:
            sid = m["result"]["sessionId"]
            send({"jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                  "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "clean up"}]}})
        if m.get("method") == "session/request_permission":
            break                                                       # the question is open ...
    proc.stdin.close()                                                  # ... and the client vanishes
    proc.stdout.close()
    proc.wait(timeout=30)
    assert (folder / "old.log").read_text() == "log"
    assert not (folder / "after.txt").exists()                          # no work in the folder after the client left


def test_no_setting_turns_the_guard_off():
    """User rule: no flag, option or env var may skip the delete question."""
    import argparse
    import inspect

    from keel import acp_server
    ap = argparse.ArgumentParser()
    acp_server.add_arguments(ap)
    flags = " ".join(o for a in ap._actions for o in a.option_strings).lower()
    for word in ("guard", "confirm", "approve", "yolo", "unsafe", "allow", "dangerous", "skip"):
        assert word not in flags, word
    assert list(inspect.signature(acp_server.KeelAgent.__init__).parameters) == ["self", "make_engine", "state_dir",
                                                                                "stack"]
    # a stack (keel.stacks, 2026-11-19) picks tools and layers only: every stack runs through the same GuardedShell
    from dataclasses import fields

    from keel.stacks import Stack
    for f in fields(Stack):
        for word in ("guard", "confirm", "approve", "delete", "unsafe", "allow", "skip"):
            assert word not in f.name, f.name
    assert "KEEL_" not in inspect.getsource(acp_server)                  # no environment switch either
