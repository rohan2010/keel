"""B4: forgiving edit_file (exact -> whitespace-tolerant -> closest-lines hint), no-op refusal, edit metric."""

import pytest

from keel.agent.filetools import FileTools, ToolError
from keel.agent.guard import GuardedShell, approve_all
from keel.agent.sandbox import Sandbox


@pytest.fixture
def ft(tmp_path):
    sb = Sandbox(tmp_path / "sb").create()
    (sb.work / "app.py").write_text("def main():\n    x = 1\n    if x:\n        print('hi')\n    return x\n")
    return FileTools(GuardedShell(sb, approve_all))


def read(ft):
    return (ft.work / "app.py").read_text()


def test_exact_edit_unchanged(ft):
    out = ft.edit_file("app.py", "x = 1", "x = 2")
    assert "Replaced 1 occurrence" in out and "x = 2" in read(ft)
    assert ft.stats == {"edit_calls": 1, "edit_failed": 0, "edit_fuzzy": 0}


def test_noop_edit_refused(ft):
    with pytest.raises(ToolError, match="nothing would change"):
        ft.edit_file("app.py", "x = 1", "x = 1")
    assert ft.stats["edit_failed"] == 1


def test_whitespace_tolerant_reindents(ft):
    # the model dropped the indentation and added trailing spaces
    out = ft.edit_file("app.py", "if x:  \n    print('hi')", "if x:\n    print('bye')\n    print('!')")
    assert "ignoring differences in spaces" in out
    assert read(ft) == ("def main():\n    x = 1\n    if x:\n        print('bye')\n        print('!')\n"
                        "    return x\n")
    assert ft.stats["edit_fuzzy"] == 1 and ft.stats["edit_failed"] == 0


def test_ambiguous_loose_match_is_refused(ft):
    (ft.work / "app.py").write_text("a:\n  x\nb:\n  x\n")
    with pytest.raises(ToolError, match="not found"):
        ft.edit_file("app.py", "x ", "y")
    assert (ft.work / "app.py").read_text() == "a:\n  x\nb:\n  x\n"


def test_miss_shows_closest_lines(ft):
    with pytest.raises(ToolError) as err:
        ft.edit_file("app.py", "print('hello')", "print('x')")
    assert "closest text is at lines 4-4" in str(err.value) and "print('hi')" in str(err.value)
    assert read(ft).count("print('hi')") == 1


def test_delete_lines_loose(ft):
    out = ft.edit_file("app.py", "  if x:\n      print('hi')", "")
    assert read(ft) == "def main():\n    x = 1\n    return x\n" and "Replaced lines 3-4" in out
