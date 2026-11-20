"""Release audit #1 and #2 (2026-11-19): deletes written with ~ / $HOME are asked about like any other, and the
agent's folder itself can't be deleted (its contents can, with the user's yes). Needs macOS sandbox-exec."""

import shutil

import pytest

from keel.agent.guard import GuardedShell
from keel.agent.sandbox import Sandbox

pytestmark = pytest.mark.skipif(shutil.which("sandbox-exec") is None, reason="macOS sandbox-exec needed")


def _shell(tmp_path, answer=True):
    sb = Sandbox(tmp_path / "ep").create()
    (sb.work / "a.txt").write_text("x")
    (sb.work / "sub").mkdir()
    (sb.work / "sub" / "b.txt").write_text("y")
    asked = []
    sh = GuardedShell(sb, lambda c, f: asked.append(sorted(f)) or answer, backup_dir=tmp_path / "bk")
    return sb, sh, asked


@pytest.mark.parametrize("cmd,files", [
    ("rm -rf ~/*", ["a.txt", "sub/b.txt"]),
    ("rm -f $HOME/a.txt", ["a.txt"]),
    ("rm -f ~/sub/b.txt", ["sub/b.txt"]),
    ("rm a.txt", ["a.txt"]),
])
def test_home_deletes_are_asked(tmp_path, cmd, files):
    sb, sh, asked = _shell(tmp_path)
    res, note = sh.run(cmd)
    assert asked == [files]                       # the user saw exactly these files
    assert note == ""                             # nothing hidden to put back
    assert not any((sb.work / f).exists() for f in files)


def test_home_delete_declined_runs_nothing(tmp_path):
    sb, sh, asked = _shell(tmp_path, answer=False)
    res, note = sh.run("rm -rf ~/*")
    assert res is None and "declined" in note and asked
    assert (sb.work / "a.txt").exists() and (sb.work / "sub" / "b.txt").exists()


@pytest.mark.parametrize("cmd", ['rm -rf "$PWD"', "rm -rf ~", "cd .. && rm -rf work", "rm -rf ../work"])
def test_work_folder_itself_survives(tmp_path, cmd):
    sb, sh, asked = _shell(tmp_path)
    sh.run(cmd)
    assert sb.work.is_dir()                       # the folder stays (contents may go, with the user's yes)
    res, _ = sh.run("echo still here")            # and the next command runs instead of crashing
    assert res is not None and res.output.strip() == "still here"


def test_missing_folder_is_an_error_not_a_crash(tmp_path):
    sb = Sandbox(tmp_path / "ep").create()
    shutil.rmtree(sb.work)
    res = sb.run("echo hi")
    assert res.exit_code == 1 and "no longer exists" in res.output
