"""Ch5 G6: worked examples are added only with the flag and contain nothing task-specific."""

import json

from keel.agent.examples import EXAMPLES
from keel.agent.loop import run_episodes_on
from keel.backends.scripted import ScriptedBackend
from keel.evals import tasks as T


def test_examples_name_no_task_and_no_task_file():
    for tid in T.TASKS:
        assert f"{tid}_" not in EXAMPLES and f" {tid} " not in EXAMPLES
    for tid in T.TASKS:
        it = T.build(tid, 0)
        for path in it.files:
            name = path.rsplit("/", 1)[-1]
            assert len(name) < 6 or name not in EXAMPLES, (tid, name)
    for leak in ("btn-primary", "button-main", "invoices.csv", "Images, Documents, Code and Other"):
        assert leak not in EXAMPLES


def _system(tmp_path, **kw):
    it = T.build("A1", 0)
    run_episodes_on(ScriptedBackend({it.id: ["Done."]}), [it], tmp_path, **kw)
    return json.loads((tmp_path / "transcripts" / f"{it.id}.json").read_text())["messages"][0]["content"]


def test_only_with_the_flag(tmp_path):
    assert EXAMPLES in _system(tmp_path / "on", examples=True)
    assert "Examples of careful work" not in _system(tmp_path / "off")
