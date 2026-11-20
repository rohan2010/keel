"""Chapter 1 core: records, layer pipeline, tool registry, approver, events, and the no-off-switch rule."""

import argparse
import inspect
import json

import pytest

from keel.core.events import Event, ListSink
from keel.core.layers import LAYERS, Pipeline, ResultContext, build_pipeline
from keel.core.records import Session, StopReason, session_from_messages
from keel.core.tools import REGISTRIES, files_registry, reporting, shell_registry


def test_records_round_trip():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "do x"},
            {"role": "assistant", "content": "<tool_call>...</tool_call>"},
            {"role": "tool", "content": "out", "_step": 1, "_call": "run_shell(...)"},
            {"role": "assistant", "content": "done"}, {"role": "user", "content": "check"},
            {"role": "assistant", "content": "ok"}]
    s = session_from_messages("e1", msgs)
    assert len(s.turns) == 2
    assert s.to_chat_messages() == [{k: v for k, v in m.items() if not k.startswith("_")} for m in msgs]
    s.stop = StopReason.ANSWER
    d = json.loads(json.dumps(s.to_dict()))
    assert d["stop"] == "answer" and d["turns"][0]["items"][2]["meta"]["_step"] == 1


def test_records_view_vs_content():
    s = Session("e2", system="sys")
    s.open_turn("r")
    s.add("user", "r")
    c = s.add_call("read_file", {"path": "a"})
    it = s.add_result(c, "x" * 100, view="x" * 10)
    assert it.call_id == c.id and it.content == "x" * 100
    assert s.to_chat_messages()[-1] == {"role": "tool", "content": "x" * 10}
    same = s.add_result(c, "y", view="y")
    assert same.view is None


def _ctx(i=0, key="k"):
    return ResultContext(label=f"@{i}", trim_name="t", repeat_sig=key, loop_key=key, loop_label="t")


def test_pipeline_off_is_identity():
    p = build_pipeline()
    o = p.tool_result(_ctx(), "raw output")
    assert (o.view, o.ban_tool_call, o.notes) == ("raw output", False, [])


def test_pipeline_order_and_notes():
    trim = lambda raw: (raw[:5], {"kind": "trim", "total": len(raw)}) if len(raw) > 5 else (raw, None)
    p = build_pipeline(trimmer=trim, repeat=True, loop_break=True)
    assert p.names == ["trim", "repeat", "loop_break"]
    outs = [p.tool_result(_ctx(i), "same long output") for i in range(3)]
    third = outs[2]
    assert third.view.startswith("same ") and third.ban_tool_call
    assert any(n.startswith("repeat@2") for n in third.notes) and any(n.startswith("loop@2") for n in third.notes)
    assert outs[0].notes == ["trim@0:t:16"]
    p.new_turn()
    assert not p.tool_result(_ctx(3), "same long output").ban_tool_call


def test_pipeline_rejects_wrong_order():
    from keel.core.layers import LoopBreakLayer, RepeatLayer
    with pytest.raises(ValueError):
        Pipeline([LoopBreakLayer(), RepeatLayer()])


def test_layer_names_known():
    assert {"trim", "repeat", "loop_break", "reminder", "verify", "probe_gate"} <= set(LAYERS)


def test_registry_matches_research_toolsets():
    """What the model sees of the tools must stay byte-identical to the research runs (schema fingerprints taken
    before the registry became the source of truth). The parser's view differs on purpose in one place: file text
    is kept verbatim (`x-keel-raw`), see test_file_text_keeps_its_final_newline."""
    import hashlib

    from keel.agent.loop import TOOLSETS
    want = {'shell': '8e6103bff5f1d10d', 'files': '0c21904c4cb0b914'}
    for name in want:                                    # the research toolsets (Ch5's files_py is new, not pinned)
        reg = REGISTRIES[name]()
        got = hashlib.sha256(json.dumps(reg.schemas, sort_keys=True).encode()).hexdigest()[:16]
        assert got == want[name], name
        assert (reg.schemas, reg.funcs) == TOOLSETS[name][1:], name
    raw = {(f["name"], p) for f in files_registry().funcs for p, v in f["parameters"]["properties"].items()
            if v.get("x-keel-raw")}
    assert raw == {("write_file", "content"), ("edit_file", "old_text"), ("edit_file", "new_text")}
    assert files_registry().kind("delete") == "delete" and shell_registry().kind("run_shell") == "shell"


def test_file_text_keeps_its_final_newline():
    """Bug fixed 2026-11-14: `.strip()` in the parser removed the text's own final newline, so every file the agent
    wrote lacked one. Only the call format's single newline may be removed, and only for file text."""
    from keel.core.parse import parse_calls
    f = files_registry().funcs
    call = lambda name, body: f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"
    p = lambda k, v: f"<parameter={k}>\n{v}\n</parameter>\n"
    got = parse_calls(call("write_file", p("path", "list.txt") + p("content", "apples\ncherries\n")), f)
    assert got == [{"write_file": {"path": "list.txt", "content": "apples\ncherries\n"}}]
    got = parse_calls(call("write_file", p("path", " a.txt ") + p("content", "  indented\n")), f)
    assert got[0]["write_file"] == {"path": "a.txt", "content": "  indented\n"}           # path still trimmed
    got = parse_calls(call("edit_file", p("path", "a") + p("old_text", "x\n") + p("new_text", "y\n")), f)
    assert got[0]["edit_file"]["old_text"] == "x\n" and got[0]["edit_file"]["new_text"] == "y\n"
    got = parse_calls(call("run_shell", p("command", "ls \n")), f)
    assert got[0]["run_shell"]["command"] == "ls"                                            # others unchanged
    # BFCL functions carry no marker: parsed exactly as before
    bfcl = [{"name": "touch", "parameters": {"properties": {"file_name": {"type": "string"}}}}]
    assert parse_calls(call("touch", p("file_name", "notes.txt\n")), bfcl) == [{"touch": {"file_name": "notes.txt"}}]


def test_approver_reports_and_refuses_on_error():
    sink = ListSink()
    ok = reporting(lambda c, f: True, sink, "s")
    boom = reporting(lambda c, f: 1 / 0, sink, "s")
    truthy = reporting(lambda c, f: "yes", sink, "s")
    assert ok("rm a", ["a"]) is True
    assert boom("rm a", ["a"]) is False
    assert truthy("rm a", ["a"]) is False          # only an explicit True allows a delete
    assert [e.data["allowed"] for e in sink.events] == [True, False, False]


def test_events_validate_kind():
    with pytest.raises(ValueError):
        Event("nope", "s")
    assert Event("stop", "s", {"reason": "answer"}).to_dict()["reason"] == "answer"


def test_no_switch_disables_the_delete_guard():
    """User rule: the delete guard is always on. No constructor argument or CLI flag may turn it off."""
    from keel.agent.guard import GuardedShell
    # backup_dir (2026-11-19) only chooses WHERE silent backups go; None still keeps them (system temp folder)
    assert list(inspect.signature(GuardedShell.__init__).parameters) == ["self", "sandbox", "confirmer", "backup_dir"]
    import tempfile
    from pathlib import Path

    from keel.agent.guard import approve_all
    from keel.agent.sandbox import Sandbox
    with tempfile.TemporaryDirectory() as d:
        sb = Sandbox(Path(d) / f"ep{abs(hash(d)) % 10**8}").create()
        (sb.work / "b.txt").write_text("old")
        sh = GuardedShell(sb, approve_all, backup_dir=None)
        n = len(sh.backups.entries())
        sh.run("echo new > b.txt")
        assert len(sh.backups.entries()) == n + 1 and sh.backups.entries()[-1]["path"] == "b.txt"
    from keel.commands import run_agent, run_bfcl_mt
    for mod in (run_agent, run_bfcl_mt):
        ap = argparse.ArgumentParser()
        mod.add_arguments(ap)
        flags = " ".join(o for a in ap._actions for o in a.option_strings).lower()
        for word in ("guard", "no-confirm", "unsafe", "yolo", "skip-confirm", "dangerous"):
            assert word not in flags, (mod.__name__, word)


def test_imports_without_mlx():
    """A6: the core, the CLI (--help) and the research layers load on a machine without MLX."""
    import subprocess
    import sys
    code = """
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('mlx', 'mlx_lm', 'numpy', 'torch'):
            raise ImportError('blocked: ' + name)
sys.meta_path.insert(0, Block())
import keel, keel.cli, keel.core.records, keel.core.tools, keel.core.layers, keel.core.parse, keel.core.events
import keel.core.backend, keel.nudges, keel.trim, keel.agent.loop
keel.cli.main(['--help'])
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "run" in r.stdout and "bfcl-mt" in r.stdout


def test_agent_loop_emits_events(tmp_path):
    """A5: one scripted episode through the real loop code (no model) streams text, tool, layer and permission
    events, and the delete guard's question reaches the sink."""
    from keel.agent.loop import _apply_reply, _new_episodes
    from keel.evals import tasks as T
    sink = ListSink()
    it = T.build("A4", 0)
    (e,) = _new_episodes([it], tmp_path, "approve_all", toolset="files", loop_break=True, events=sink)
    victim = next(p for p in it.files if "/" not in p)
    call = ('<tool_call>\n<function=delete>\n<parameter=paths>\n["%s"]\n</parameter>\n</function>\n</tool_call>'
            % victim)
    _apply_reply(e, "Removing it.\n" + call, "stop", 20)
    ls = '<tool_call>\n<function=list_files>\n</function>\n</tool_call>'
    for _ in range(3):
        _apply_reply(e, ls, "stop", 20)
    kinds = sink.kinds()
    assert kinds[:2] == ["text", "tool_started"] and "permission" in kinds and "tool_finished" in kinds
    perm = next(ev for ev in sink.events if ev.kind == "permission")
    assert perm.data["allowed"] is True and victim in " ".join(perm.data["files"])
    assert any(ev.kind == "layer" and ev.data["note"].startswith("loop@") for ev in sink.events)
    assert e.ban_tool_call


def test_tool_errors_hide_absolute_paths(tmp_path):
    """A crashing file tool must not show the model the machine's absolute paths (privacy + repeatable runs)."""
    from keel.agent.loop import _apply_reply, _new_episodes
    from keel.evals import tasks as T
    (e,) = _new_episodes([T.build("E3", 0)], tmp_path, "approve_all", toolset="files")

    def boom(name, args):
        raise FileNotFoundError(2, "No such file or directory", str(e.sandbox.work.resolve() / "year_Apr" / "m"))
    e.ftools.call = boom
    _apply_reply(e, '<tool_call>\n<function=list_files>\n</function>\n</tool_call>', "stop", 20)
    out = e.messages[-1]["content"]
    assert out.startswith("Error: FileNotFoundError") and "year_Apr/m" in out
    assert str(tmp_path) not in out and "/private/" not in out


def test_shell_output_paths_are_relative_and_still_work(tmp_path):
    """`pwd` / mktemp print sandbox paths; the model sees them relative to its folder, and they still work."""
    from keel.agent.loop import _apply_reply, _new_episodes
    from keel.evals import tasks as T
    (e,) = _new_episodes([T.build("C1", 0)], tmp_path, "approve_all", toolset="files")
    sh = lambda c: '<tool_call>\n<function=run_shell>\n<parameter=command>\n%s\n</parameter>\n</function>\n</tool_call>' % c
    _apply_reply(e, sh("pwd; echo $HOME; t=$(mktemp); echo hi > $t; echo $t"), "stop", 20)
    out = e.messages[-1]["content"]
    assert str(tmp_path) not in out and "/private/" not in out and "/tmp/" not in out, out
    lines = out.splitlines()
    assert lines[0] == "." and lines[1] == "." and lines[2].startswith("../.tmp/"), lines
    _apply_reply(e, sh("cat %s" % lines[2]), "stop", 20)            # the rewritten path is usable as-is
    assert e.messages[-1]["content"] == "hi"


def test_bfcl_episode_emits_events():
    """A5 for BFCL: a scripted episode (the ground-truth call, repeated) streams text, tool and layer events."""
    # (layer events mirror e.notes; the loop note proves a layer event reaches the sink)
    import importlib.util
    from pathlib import Path
    pytest.importorskip("bfcl_eval")
    from keel.evals import bfcl_mt as M
    spec = importlib.util.spec_from_file_location("tbm", Path(__file__).parent / "checks" / "check_bfcl_mt.py")
    tbm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tbm)
    r = M.load("base")[0]
    call = tbm.render_reply([r["ground_truth"][0][0]], r)
    sink = ListSink()
    e = M.MTEpisode(r, M.safe_name("20261113-000000_t_events"), loop_break=True, events=sink).start()
    for _ in range(4):                       # cd succeeds once, then the same error 3x -> the loop breaker fires
        e.apply("Checking.\n" + call, "stop")
    kinds = sink.kinds()
    assert kinds[:2] == ["text", "tool_started"] and kinds.count("tool_finished") == 4
    fin = next(ev for ev in sink.events if ev.kind == "tool_finished")
    assert fin.session == r["id"] and fin.data["turn"] == 0 and "output" in fin.data
    assert any(ev.kind == "layer" and ev.data["note"].startswith("loop@t0") for ev in sink.events)
