"""Ch5 G8: native tool vocabulary (`--native-tools qwen`). The model sees Qwen Code's names; keel's code, scorers and
records keep keel's. Flag off = the toolsets are exactly as before."""

import json
import sys

import pytest

from keel.agent.loop import TOOLSETS, run_episodes_on, toolset_view
from keel.backends.scripted import ScriptedBackend
from keel.core import native
from keel.core.parse import parse_calls
from keel.evals import tasks as T

QWEN = {"list_directory", "read_file", "write_file", "edit", "grep_search", "move", "delete", "run_shell_command"}


def call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def names(schemas):
    return {s["function"]["name"] for s in schemas}


def test_flag_off_is_the_old_toolset():
    for t in ("shell", "files", "files_py"):
        assert toolset_view(t) is TOOLSETS[t]


def test_qwen_names_and_arguments():
    prompt, schemas, funcs = toolset_view("files@qwen")
    assert names(schemas) == QWEN
    edit = next(s for s in schemas if s["function"]["name"] == "edit")["function"]["parameters"]
    assert set(edit["properties"]) == {"file_path", "old_string", "new_string", "replace_all"}
    assert edit["required"] == ["file_path", "old_string", "new_string"]
    read = next(s for s in schemas if s["function"]["name"] == "read_file")["function"]["parameters"]
    assert set(read["properties"]) == {"file_path", "offset", "limit"} and "0" in read["properties"]["offset"]["description"]
    assert "run_shell_command" in prompt and "run_shell " not in prompt
    assert names(toolset_view("files_py@qwen")[1]) == QWEN | {"run_python"}
    assert names(TOOLSETS["files"][1]) == {"list_files", "read_file", "write_file", "edit_file", "search", "move",
                                           "delete", "run_shell"}           # the originals are not changed


def test_parser_keeps_raw_text_under_native_names():
    funcs = toolset_view("files@qwen")[2]
    old, new = "  a = 1\n", "  a = 2\n"
    got = parse_calls(call("edit", file_path="x.py", old_string=old, new_string=new), funcs)
    back = native.to_internal(got, "qwen")[0]["edit_file"]
    assert back["path"] == "x.py" and back["old_text"].endswith("a = 1\n") and back["new_text"].endswith("a = 2\n")


def test_offset_is_zero_based():
    assert native.to_internal([{"read_file": {"file_path": "a", "offset": 0}}], "qwen") == \
        [{"read_file": {"path": "a", "start_line": 1}}]
    assert native.to_internal([{"edit_file": {"path": "a"}}], "qwen") == [{"edit_file": {"path": "a"}}]  # passes through


def test_text_renames_keel_words_only():
    s = "Use read_file to look, then edit_file or write_file to fix (run_shell for anything else). old_text and new_text"
    out = native.text(s, "qwen")
    assert out == ("Use read_file to look, then edit or write_file to fix (run_shell_command for anything else). "
                   "old_string and new_string")
    assert native.text(out, "qwen") == out                                    # idempotent
    assert native.text("the path to my-run_shell-x", "qwen") == "the path to my-run_shell-x"
    assert native.text(s, None) == s


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox")
def test_episode_with_native_calls(tmp_path):
    it = T.build("D2", 0)
    page = next(p for p in it.files if p.endswith(".html"))
    replies = [call("read_file", file_path=page, offset=0, limit=5),
               call("edit", file_path=page, old_string="no such text here", new_string="x"),
               call("list_directory", path="."),
               call("run_shell_command", command="echo hi"),
               "Done."]
    res = run_episodes_on(ScriptedBackend({it.id: replies}), [it], tmp_path, toolset="files@qwen")
    msgs = json.loads((tmp_path / "transcripts" / f"{it.id}.json").read_text())["messages"]
    outs = [m["content"] for m in msgs if m["role"] == "tool"]
    assert outs[0].lstrip().startswith("1 ")                                  # offset 0 = line 1
    assert outs[1].startswith("Error") and "old_text" not in outs[1]
    assert not outs[2].startswith("Error") and outs[3].strip() == "hi"
    assert "run_shell_command" in msgs[0]["content"]
    assert res[0]["commands"][0].startswith("read_file ")                     # records keep keel's names
    assert any(c.startswith("edit_file ") for c in res[0]["commands"])


def test_unknown_family_refused():
    with pytest.raises(ValueError):
        native.text("x", "llama")
