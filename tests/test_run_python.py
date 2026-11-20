"""Ch5 G2: run_python (toolset files_py) runs through the guarded shell; the default toolset is unchanged."""

import json
import sys

import pytest

from keel.agent.loop import TOOLSETS, run_episodes_on
from keel.backends.scripted import ScriptedBackend
from keel.core.parse import parse_calls
from keel.evals import tasks as T

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox")


def call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def run(tmp_path, replies, toolset="files_py"):
    it = T.build("C3", 0)
    res = run_episodes_on(ScriptedBackend({it.id: replies}), [it], tmp_path, toolset=toolset)
    msgs = json.loads((tmp_path / "transcripts" / f"{it.id}.json").read_text())["messages"]
    return it, res[0], msgs, tmp_path / "sandboxes" / it.id / "work"


CODE = "import os\nfiles = sorted(os.listdir('.'))\nfor f in files:\n    if f.endswith('.csv'):\n        print('csv:', f)\nprint(sum(range(5)))"


def test_script_runs_with_indentation_kept(tmp_path):
    _, r, msgs, _ = run(tmp_path, [call("run_python", code=CODE), "Done."])
    out = next(m["content"] for m in msgs if m["role"] == "tool")
    assert "csv:" in out and out.strip().endswith("10")


def test_a_delete_from_python_goes_through_the_guard(tmp_path):
    """The guard asks before a script deletes a file (dry run), exactly as for run_shell; a user who says no keeps it."""
    it = T.build("C3", 0)
    victim = next(p for p in it.files if "/" not in p)
    reply = call("run_python", code=f"import os\nos.remove({victim!r})\nprint('gone')")
    res = run_episodes_on(ScriptedBackend({it.id: [reply, "Done."]}), [it], tmp_path, toolset="files_py",
                          confirmer=lambda command, files: False)
    assert (tmp_path / "sandboxes" / it.id / "work" / victim).exists()
    c = res[0]["confirmations"][0]
    assert c["kind"] == "asked" and c["approved"] is False and c["files"] == [victim]


def test_default_toolset_has_no_run_python():
    assert "run_python" not in [s["function"]["name"] for s in TOOLSETS["files"][1]]
    assert "run_python" in [s["function"]["name"] for s in TOOLSETS["files_py"][1]]
    assert "run_python" in TOOLSETS["files_py"][0]                         # the prompt line


def test_parser_keeps_code_whitespace():
    funcs = TOOLSETS["files_py"][2]
    got = parse_calls(call("run_python", code=CODE), funcs)[0]["run_python"]["code"]
    assert got == CODE


def test_file_tools_still_work_with_run_python(tmp_path):
    """Regression (G2 run 1): files_py offered the file tools but did not wire them, so every call errored."""
    _, r, msgs, _ = run(tmp_path, [call("list_files", path="."), call("run_python", code="print(6 * 7)"), "Done."])
    outs = [m["content"] for m in msgs if m["role"] == "tool"]
    assert not any(o.startswith("Error") for o in outs), outs
    assert outs[1].strip() == "42"


def test_run_record_names_the_prompt_the_model_saw():
    """Regression (G2 audit): the run config recorded the shell-only prompt for every toolset."""
    import inspect

    from keel.commands import run_agent
    assert '"system": toolset_view(ts)[0]' in inspect.getsource(run_agent)   # ts = toolset (+ G8 family)
