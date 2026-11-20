"""Ch7 P5 safety fixes, each with the release audit's reproduction: read allowlist (#3), output cap while reading (#4),
failed calls by exit code not text (#13), run_python heredoc delimiter (#14), backups pruned with sessions (#15)."""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

from keel.agent.loop import _heredoc, run_episodes_on
from keel.agent.sandbox import STOP_AFTER, Sandbox
from keel.backends.scripted import ScriptedBackend
from keel.evals import tasks as T

mac = pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox")


def call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


@pytest.fixture
def outside_users():
    """A folder outside /Users (the audit's fresh clone in /private/tmp), with a file next to the episode."""
    base = Path(tempfile.mkdtemp(dir="/private/tmp"))
    (base / "secret.txt").write_text("TOP-SECRET-7731")
    yield base
    shutil.rmtree(base, ignore_errors=True)


@mac
def test_reads_outside_the_episode_are_blocked_wherever_it_lives(outside_users):
    sb = Sandbox(outside_users / "ep").create()
    for cmd in (f"cat {outside_users}/secret.txt", f"ls {outside_users}", "cat ../../secret.txt"):
        r = sb.run(cmd)
        assert r.exit_code != 0 and "TOP-SECRET-7731" not in r.output, cmd
    assert sb.run("echo hi > a.txt && cat a.txt").output.strip() == "hi"          # the episode itself works
    assert sb.run("head -1 /etc/hosts && python3 -c 'print(1)'").exit_code == 0     # system files and tools work


@mac
def test_endless_output_is_stopped_and_never_held_in_memory(tmp_path):
    sb = Sandbox(tmp_path / "ep").create()
    t0 = time.perf_counter()
    r = sb.run("yes")
    assert time.perf_counter() - t0 < 10 and r.exit_code == -9 and not r.timed_out
    assert f"more than {STOP_AFTER // 1_000_000} MB" in r.output and len(r.output) < 5000


@mac
def test_big_output_keeps_head_and_tail(tmp_path):
    sb = Sandbox(tmp_path / "ep").create()
    r = sb.run("python3 -c 'print(*range(1, 3000001), sep=chr(10))'")   # ~21 MB: past the kept-whole limit
    assert r.exit_code == 0 and r.output.startswith("1\n2\n") and r.output.rstrip().endswith("3000000")
    assert "characters cut" in r.output and len(r.output) < 5000


@mac
def test_output_under_the_limit_is_cut_exactly_as_before(tmp_path):
    sb = Sandbox(tmp_path / "ep").create()
    full = "".join(f"{i}\n" for i in range(1, 5001))
    half = sb.max_output // 2
    want = full[:half] + f"\n... [{len(full) - sb.max_output} characters cut] ...\n" + full[-half:]
    assert sb.run("seq 1 5000").output == want


def _episode(tmp_path, replies, **kw):
    it = T.build("C1", 0)
    res = run_episodes_on(ScriptedBackend({it.id: replies}), [it], tmp_path, **kw)
    msgs = json.loads((tmp_path / "transcripts" / f"{it.id}.json").read_text())["messages"]
    return res[0], msgs


@mac
def test_reading_a_file_that_starts_with_error_is_not_a_failed_call(tmp_path):
    r, _ = _episode(tmp_path, [call("write_file", path="errors.log", content="Error: disk full\n"),
                               call("read_file", path="errors.log"), "Done."], done_gate=True, toolset="files_py")
    assert not any("failed and nothing after it succeeded" in p for ps in r["gate_problems"] for p in ps)


@mac
def test_a_command_that_exits_non_zero_is_still_a_failed_call(tmp_path):
    r, _ = _episode(tmp_path, [call("run_shell", command="ls nope"), "All done."], done_gate=True)
    assert "failed and nothing after it succeeded" in r["gate_problems"][0][0]


def test_heredoc_delimiter_cannot_occur_in_the_code_and_is_stable():
    code = "print(1)\nKEEL_PY\nprint(2)"
    tag = _heredoc(code)
    assert tag not in code.splitlines() and tag == _heredoc(code) and tag != _heredoc("print(3)")


@mac
def test_a_line_keel_py_in_the_code_no_longer_ends_the_script(tmp_path):
    code = "x = '''\nKEEL_PY\n'''\nprint('end', len(x))"
    _, msgs = _episode(tmp_path, [call("run_python", code=code), "Done."], toolset="files_py")
    out = next(m["content"] for m in msgs if m["role"] == "tool")
    assert "end 9" in out and "[exit code" not in out


def test_backups_go_with_their_session(tmp_path):
    from keel.acp_server import prune_sessions
    sessions, backups = tmp_path / "sessions", tmp_path / "backups"
    for i in range(55):
        for base in (sessions, backups):
            p = base / f"s{i:010x}"
            p.mkdir(parents=True)
            t = time.time() - 1000 + i
            os.utime(p, (t, t))
    (backups / "s00000000ff").mkdir()                    # orphan: its session record is already gone
    (backups / "notes").mkdir()                          # not keel's: never touched
    assert prune_sessions(sessions, 50, set(), backups) == 5
    left = {p.name for p in backups.iterdir()}
    assert {p.name for p in sessions.iterdir()} | {"notes"} == left
    assert "s0000000000" not in left and "s00000000ff" not in left and len(left) == 51
