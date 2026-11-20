"""P4: keel-core as an ACP agent, end to end over stdio (subprocess + the official Python ACP client).

The scripted backend stands in for the model, so this runs anywhere, with no MLX and no GPU. It checks what the Rust
TUI relies on: streamed prose without tool markup, tool calls with kinds, a diff for every write, follow-up prompts in
the same session, refused folders, and a stdout that carries nothing but protocol messages.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

acp = pytest.importorskip("acp")
from acp import schema as s  # noqa: E402


def call(name, **params):
    body = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
                   for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


class Recorder:
    """A minimal ACP client: records updates, answers permission requests with `answer`."""

    def __init__(self, answer="reject"):
        self.updates, self.permissions, self.answer = [], [], answer

    async def session_update(self, session_id, update, **kw):
        self.updates.append(update)

    async def request_permission(self, session_id, tool_call, options, **kw):
        self.permissions.append((tool_call, options))
        want = "allow_once" if self.answer == "allow" else "reject_once"
        opt = next(o for o in options if o.kind == want)
        return s.RequestPermissionResponse(outcome=s.AllowedOutcome(outcome="selected", option_id=opt.option_id))

    def on_connect(self, conn):
        pass


async def _session(folder: Path, state: Path, script, prompts, client=None):
    sp = state / "script.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(script))
    client = client or Recorder()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    async with acp.spawn_agent_process(client, sys.executable, "-m", "keel.cli", "acp", "--backend", "scripted",
                                       "--script", str(sp), "--state-dir", str(state), env=env) as (conn, proc):
        init = await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)
        new = await conn.new_session(cwd=str(folder))
        stops = []
        for p in prompts:
            r = await conn.prompt(session_id=new.session_id, prompt=[s.TextContentBlock(type="text", text=p)])
            stops.append(r.stop_reason)
        return init, new, stops, client


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=60))


def test_acp_session_end_to_end(tmp_path):
    folder, state = tmp_path / "proj", tmp_path / "state"
    folder.mkdir()
    (folder / "notes.md").write_text("# Notes\n- old line\n")
    script = [
        "I'll add the todo file.\n" + call("write_file", path="todo.md", content="- [ ] buy milk\n"),
        "Now I'll fix the notes.\n" + call("edit_file", path="notes.md", old_text="- old line", new_text="- new line"),
        "Done: created todo.md and updated notes.md.",
        "Yes, both files are in the folder.",              # answer to the follow-up prompt
    ]
    init, new, stops, c = run(_session(folder, state, script, ["make a todo and fix my notes", "did it work?"]))

    assert init.protocol_version == acp.PROTOCOL_VERSION
    assert init.field_meta["keel"]["deleteGuard"] == "always"
    assert new.field_meta["keel"]["folder"] == str(folder.resolve())
    assert stops == ["end_turn", "end_turn"]

    text = "".join(u.content.text for u in c.updates if getattr(u, "session_update", "") == "agent_message_chunk")
    assert "I'll add the todo file." in text and "Yes, both files are in the folder." in text
    assert "<tool_call>" not in text and "<function=" not in text

    starts = [u for u in c.updates if getattr(u, "session_update", "") == "tool_call"]
    ends = [u for u in c.updates if getattr(u, "session_update", "") == "tool_call_update"]
    assert [u.kind for u in starts] == ["edit", "edit"] and [u.status for u in ends] == ["completed", "completed"]
    diffs = [b for u in ends for b in (u.content or []) if getattr(b, "type", "") == "diff"]
    assert len(diffs) == 2
    assert diffs[0].old_text is None and diffs[0].new_text == (folder / "todo.md").read_text()   # a new file
    assert diffs[1].old_text == "# Notes\n- old line\n" and diffs[1].new_text == "# Notes\n- new line\n"
    assert (folder / "todo.md").read_text().startswith("- [ ] buy milk")

    rec = json.loads((state / "sessions" / new.session_id / "transcript.json").read_text())
    assert rec["folder"] == str(folder.resolve()) and [m["role"] for m in rec["messages"]].count("user") == 2


def test_acp_refuses_a_folder_that_would_hold_the_scratch(tmp_path):
    """The scratch folder lives in the system temp folder; working in a folder that contains it would put keel's
    own files (and the delete guard's backups) where the agent can reach them, so the session is refused."""
    folder = tmp_path / "proj"
    folder.mkdir()
    old_tmp = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = str(folder / "inner-tmp")             # temp folder inside the work folder
    (folder / "inner-tmp").mkdir()
    try:
        with pytest.raises(acp.RequestError) as err:
            run(_session(folder, tmp_path / "state", ["hi"], ["hi"]))
    finally:
        if old_tmp is None:
            os.environ.pop("TMPDIR")
        else:
            os.environ["TMPDIR"] = old_tmp
    assert "must not contain" in str(err.value) or "folder_refused" in json.dumps(err.value.data or {})


def test_acp_stdout_carries_only_protocol(tmp_path):
    """Every line keel-core writes to stdout must be a JSON-RPC message, even when code prints. (stdin stays open
    while we read, as with a real client: on end of input the core treats the client as gone and stops.)"""
    import subprocess
    folder = tmp_path / "p"
    folder.mkdir()
    proc = subprocess.Popen([sys.executable, "-c", "import keel.cli as c; print('stray print'); "
                             "c.main(['acp','--backend','scripted','--state-dir',%r])" % str(tmp_path / "st")],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for m in [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": 1}},
                  {"jsonrpc": "2.0", "id": 2, "method": "session/new",
                   "params": {"cwd": str(folder), "mcpServers": []}}]:
            proc.stdin.write(json.dumps(m) + "\n")
            proc.stdin.flush()
        lines = [proc.stdout.readline() for _ in range(2)]
    finally:
        proc.stdin.close()
        err = proc.stderr.read()
        proc.wait(timeout=30)
    msgs = [json.loads(l) for l in lines]
    assert [m.get("id") for m in msgs] == [1, 2] and all(m["jsonrpc"] == "2.0" and "result" in m for m in msgs)
    assert "stray print" in err and "keel-core" in err                 # the print went to stderr, not the protocol


def test_acp_stdout_to_a_regular_file(tmp_path):
    """A launcher may point stdout at a file: the core relays through a pipe instead of crashing."""
    import subprocess
    import time
    folder = tmp_path / "p"
    folder.mkdir()
    out = tmp_path / "out.jsonl"
    with out.open("wb") as f:
        proc = subprocess.Popen([sys.executable, "-m", "keel.cli", "acp", "--backend", "scripted", "--state-dir",
                                 str(tmp_path / "st")], stdin=subprocess.PIPE, stdout=f, stderr=subprocess.PIPE)
        proc.stdin.write((json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                      "params": {"protocolVersion": 1}}) + "\n").encode())
        proc.stdin.flush()
        for _ in range(100):
            if out.read_text().strip():
                break
            time.sleep(0.1)
        proc.stdin.close()
        proc.wait(timeout=30)
    assert json.loads(out.read_text().splitlines()[0])["id"] == 1, proc.stderr.read()[-800:]


def test_history_keeps_the_newest_50(tmp_path):
    import time

    from keel.acp_server import prune_sessions
    d = tmp_path / "sessions"
    for i in range(55):
        p = d / f"s{i:010x}"
        p.mkdir(parents=True)
        os.utime(p, (time.time() - 1000 + i,) * 2)                 # s...36 is the newest
    (d / "notes").mkdir()                                          # not keel's: never touched
    assert prune_sessions(d, 50, {"s0000000000"}) == 5             # an active session is never pruned ...
    left = sorted(p.name for p in d.iterdir())
    assert "notes" in left and "s0000000000" in left and len(left) == 51   # ... and counts toward the 50
    assert "s0000000001" not in left and f"s{54:010x}" in left


def test_scratch_is_removed_when_keel_core_exits(tmp_path):
    """Scratch goes to the system temp folder (here TMPDIR = tmp/t) and is deleted on exit; the transcript stays."""
    folder, state, tmpdir = tmp_path / "proj", tmp_path / "state", tmp_path / "t"
    folder.mkdir()
    tmpdir.mkdir()
    old_tmp = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = str(tmpdir)
    try:
        init, new, stops, c = run(_session(folder, state, ["ok"], ["hi"]))
    finally:
        if old_tmp is None:
            os.environ.pop("TMPDIR")
        else:
            os.environ["TMPDIR"] = old_tmp
    assert stops == ["end_turn"]
    assert not any(tmpdir.glob("keel-s*")), list(tmpdir.iterdir())          # scratch gone
    assert (state / "sessions" / new.session_id / "transcript.json").exists()
    assert sorted(p.name for p in folder.iterdir()) == []                    # nothing left in the user's folder
