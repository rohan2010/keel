"""Ch5 validation: `--no-send-note` removes the files_py email sentence (SEND_NOTE) so the stack can be measured
without it. Flag on (default) = the prompt is exactly as before."""

import json

from keel.agent.loop import SEND_NOTE, TOOLSETS, run_episodes_on
from keel.backends.scripted import ScriptedBackend
from keel.evals import tasks as T

OLD_FILES_PY_TAIL = ("write a parser for them. You cannot send email or messages from here: if the user asks you to, "
                     "say so at once without running any command, and tell them how to do it themselves.")


def system_of(tmp_path, it):
    return json.loads((tmp_path / "transcripts" / f"{it.id}.json").read_text())["messages"][0]["content"]


def test_default_prompt_unchanged():
    assert TOOLSETS["files_py"][0].endswith(OLD_FILES_PY_TAIL)
    assert SEND_NOTE.startswith(" ") and SEND_NOTE not in TOOLSETS["files"][0]


def test_flag_off_removes_only_the_sentence(tmp_path):
    it = T.build("F1", 0)
    run_episodes_on(ScriptedBackend({it.id: ["Done."]}), [it], tmp_path, toolset="files_py", send_note=False)
    s = system_of(tmp_path, it)
    assert "send email" not in s and "write a parser for them." in s
    assert s == TOOLSETS["files_py"][0].replace(SEND_NOTE, "")


def test_flag_on_keeps_it(tmp_path):
    it = T.build("F1", 0)
    run_episodes_on(ScriptedBackend({it.id: ["Done."]}), [it], tmp_path, toolset="files_py")
    assert SEND_NOTE in system_of(tmp_path, it)


def test_examples_still_appended_without_note(tmp_path):
    it = T.build("F1", 0)
    run_episodes_on(ScriptedBackend({it.id: ["Done."]}), [it], tmp_path, toolset="files_py", send_note=False,
                    examples=True)
    s = system_of(tmp_path, it)
    assert SEND_NOTE not in s and len(s) > len(TOOLSETS["files_py"][0].replace(SEND_NOTE, ""))
