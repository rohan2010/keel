"""Silent backups (2026-11-19): the delete guard's overwrite gap is closed without changing what the model sees.
Done-when: every overwrite / edit / approved delete leaves a restorable copy outside the sandbox, the model's tool
output is unchanged, and nothing new appears inside the sandbox."""

from pathlib import Path

import pytest

from keel.agent.backups import Backups
from keel.agent.filetools import FileTools
from keel.agent.guard import GuardedShell, approve_all
from keel.agent.sandbox import Sandbox


@pytest.fixture
def env(tmp_path: Path):
    sb = Sandbox(tmp_path / "sandboxes" / "ep").create()
    (sb.work / "a.txt").write_text("A original\n")
    (sb.work / "b.txt").write_text("B original\n")
    sh = GuardedShell(sb, approve_all, backup_dir=tmp_path / "backups" / "ep")
    return sb, sh, tmp_path / "backups" / "ep"


def kept(bdir: Path) -> list[tuple[str, str, str]]:
    return [(e["path"], e["reason"], (bdir / str(e["n"]) / e["path"]).read_text()) for e in Backups(bdir).entries()]


@pytest.mark.parametrize("cmd", ["echo new > b.txt", "cp a.txt b.txt", "mv a.txt b.txt", "sed -i '' 's/B/X/' b.txt",
                                 "python3 -c \"open('b.txt','w').write('py')\""])
def test_overwrite_keeps_the_earlier_content(env, cmd):
    _, sh, bdir = env
    res, note = sh.run(cmd)
    assert res is not None and res.exit_code == 0
    if cmd.startswith("mv"):
        # unchanged guard behaviour (before backups too): a.txt's content landed on an existing path, so a.txt counts
        # as deleted and is put back, with the guard's usual note
        assert "put back: a.txt" in note
    else:
        assert note == ""                                               # the model is told nothing new
    assert ("b.txt", "changed", "B original\n") in kept(bdir)


def test_cp_over_a_file_is_recoverable_although_the_snapshot_is_hard_linked(env):
    sb, sh, bdir = env
    sh.run("cp a.txt b.txt")
    assert (sb.work / "b.txt").read_text() == "A original\n"
    out = Backups(bdir).restore(1, overwrite=True)
    assert out == sb.work / "b.txt" and out.read_text() == "B original\n"
    assert ("b.txt", "replaced by restore", "A original\n") in kept(bdir)  # what restore replaced is kept too


def test_approved_delete_goes_to_backups_declined_and_renames_do_not(tmp_path):
    sb = Sandbox(tmp_path / "sandboxes" / "ep").create()
    for n in ("a", "b", "c"):
        (sb.work / f"{n}.txt").write_text(n)
    bdir = tmp_path / "bk"
    sh = GuardedShell(sb, lambda cmd, files: "a.txt" in files, backup_dir=bdir)
    sh.run("rm a.txt")                                   # approved
    sh.run("rm b.txt")                                   # declined: nothing runs
    sh.run("mv c.txt d.txt")                             # rename: content moved, not lost
    assert kept(bdir) == [("a.txt", "deleted", "a")]
    assert (sb.work / "b.txt").exists() and (sb.work / "d.txt").read_text() == "c"


def test_nothing_new_inside_the_sandbox(env):
    sb, sh, bdir = env
    before = sorted(p.name for p in sb.root.iterdir())
    sh.run("echo x > b.txt")
    assert sorted(p.name for p in sb.root.iterdir()) == before          # no backup or stage folder in the sandbox
    assert not (bdir / ".stage").exists()                               # the stage is dropped after each command
    assert bdir.resolve() not in sb.work.parents and sb.root not in bdir.resolve().parents


def test_file_tools_back_up_before_changing(env):
    sb, sh, bdir = env
    ft = FileTools(sh)
    assert ft.write_file("a.txt", "A2").startswith("Replaced")         # tool output unchanged
    ft.write_file("new.txt", "fresh")                                   # a new file: nothing to keep
    ft.edit_file("b.txt", "original", "edited")
    ft.delete(["new.txt"])
    assert kept(bdir) == [("a.txt", "replaced", "A original\n"), ("b.txt", "edited", "B original\n"),
                          ("new.txt", "deleted", "fresh")]


def test_backups_refuse_to_live_inside_the_work_folder(tmp_path):
    sb = Sandbox(tmp_path / "ep").create()
    with pytest.raises(ValueError, match="outside"):
        GuardedShell(sb, approve_all, backup_dir=sb.work / "bk")


def test_restore_refuses_to_overwrite_without_the_flag(env):
    sb, sh, bdir = env
    sh.run("echo new > b.txt")
    with pytest.raises(ValueError, match="exists"):
        Backups(bdir).restore(1)
    to = Backups(bdir).restore(1, to=sb.root.parent / "elsewhere.txt")
    assert to.read_text() == "B original\n"
