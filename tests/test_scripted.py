"""P3: the scripted backend drives the real agent loop (tools, guard, layers, events, scoring) with no model."""

import json
import subprocess
import sys

from keel.agent.loop import run_episodes_on
from keel.backends.scripted import ScriptedBackend
from keel.core.events import ListSink
from keel.evals import tasks as T


def call(name, **params):
    body = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
                   for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def test_scripted_refusal_task_scores_correct(tmp_path):
    it = T.build("F3", 0)
    res = run_episodes_on(ScriptedBackend([
        "I can't delete everything on your computer. I can only see and change files in this folder."]),
        [it], tmp_path)
    assert res[0]["correct"] and res[0]["steps"] == 1 and res[0]["stop"] == "answer"


def test_scripted_tools_and_streaming(tmp_path):
    it = T.build("C1", 0)
    sink = ListSink()
    be = ScriptedBackend({it.id: ["Let me look first.\n" + call("list_files"), "I looked at the folder."]}, chunk=3)
    res = run_episodes_on(be, [it], tmp_path, events=sink, stream=True)
    kinds = sink.kinds()
    assert "tool_started" in kinds and "tool_finished" in kinds and kinds[-1] == "stop"
    streamed = "".join(e.data["text"] for e in sink.events if e.kind == "text_delta")
    assert streamed == "Let me look first.\nI looked at the folder."          # tool markup never streamed
    assert len(be.prompts) == 2 and be.prompts[1][1][-1]["role"] == "tool"    # the tool result reached the "model"
    assert res[0]["steps"] == 2


def test_scripted_delete_is_asked_and_declined(tmp_path):
    it = T.build("A4", 0)
    victim = next(p for p in it.files if "/" not in p)
    sink = ListSink()
    res = run_episodes_on(ScriptedBackend([call("delete", paths=[victim]), "Ok, I kept it."]), [it], tmp_path,
                          confirmer=lambda cmd, files: False, events=sink)
    perm = [e for e in sink.events if e.kind == "permission"]
    assert perm and perm[0].data["allowed"] is False and victim in perm[0].data["files"][0]
    assert (tmp_path / "sandboxes" / it.id / "work" / victim).exists()
    assert res[0]["confirmations"][0]["approved"] is False


def test_scripted_honours_the_loop_breaker_ban(tmp_path):
    it = T.build("C1", 0)
    ls = call("list_files")
    be = ScriptedBackend([ls, ls, ls, "Still looking.\n" + ls])               # 4th reply would call again
    res = run_episodes_on(be, [it], tmp_path, loop_break=True)
    assert any(n.startswith("loop@") for n in res[0]["notes"])
    assert res[0]["stop"] == "answer" and res[0]["answer"] == "Still looking."   # tool part dropped by the ban


def test_scripted_runs_without_mlx(tmp_path):
    code = f"""
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('mlx', 'mlx_lm', 'numpy', 'torch'):
            raise ImportError('blocked: ' + name)
sys.meta_path.insert(0, Block())
from pathlib import Path
from keel.evals import tasks as T
from keel.agent.loop import run_episodes_on
from keel.backends.scripted import ScriptedBackend
r = run_episodes_on(ScriptedBackend(["No, I won't do that."]), [T.build("F3", 0)], Path({str(tmp_path)!r}))
print("steps", r[0]["steps"])
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "steps 1" in r.stdout
