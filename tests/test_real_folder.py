"""Chapter 2, P1: the in-place sandbox on a user's real folder (`keel run`).

The rules: the folder is never copied, removed or re-dated by keel itself; the agent can read and write only there;
oversized or overlapping folders are refused; the delete guard still asks and still undoes hidden deletes.
"""

import os
from pathlib import Path

import pytest

from keel.agent.filetools import FileTools
from keel.agent.guard import GuardedShell, approve_all
from keel.agent.sandbox import FolderRefused, Sandbox, check_folder

OLD = 1_600_000_000.0                       # 2020-09-13: a real, old file date that must survive


@pytest.fixture
def user(tmp_path):
    """A small 'real' folder with old file dates, and a separate keel state folder."""
    home = tmp_path / "home" / "Documents"
    (home / "sub").mkdir(parents=True)
    for rel, text in {"a.txt": "alpha", "b.txt": "beta", "sub/c.txt": "gamma"}.items():
        (home / rel).write_text(text)
        os.utime(home / rel, (OLD, OLD))
    return home, tmp_path / "state"


def snapshot(folder: Path) -> dict:
    return {str(p.relative_to(folder)): (p.read_bytes(), p.stat().st_mtime)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_create_leaves_the_folder_untouched_and_clock_off(user):
    home, state = user
    before = snapshot(home)
    sb = Sandbox.for_folder(home, "s1", state)
    assert sb.in_place and sb.work == home.resolve() and not sb.logical_clock
    assert snapshot(home) == before                           # nothing copied, removed or re-dated
    sb.run("cat a.txt > /dev/null; ls -la")
    sb.settle_clock(5)                                        # the loop calls this after every tool: must be a no-op
    assert snapshot(home) == before
    assert (state / "keel-s1" / ".bin" / "mktemp").exists()       # scratch: <scratch dir>/keel-<session>


def test_recreating_the_session_never_touches_the_folder(user):
    home, state = user
    before = snapshot(home)
    Sandbox.for_folder(home, "s1", state)
    Sandbox.for_folder(home, "s1", state)                     # create() rmtree's the old state folder, only that
    assert snapshot(home) == before


def test_agent_reads_and_writes_only_in_the_folder(user):
    home, state = user
    sb = Sandbox.for_folder(home, "s1", state)
    r = sb.run("cat a.txt; echo new > made.txt; mkdir -p d && echo x > d/y.txt")
    assert r.exit_code == 0 and r.output.startswith("alpha")
    assert (home / "made.txt").read_text() == "new\n" and (home / "d" / "y.txt").exists()
    outside = home.parent / "outside.txt"
    r = sb.run(f"echo no > {outside}")
    assert r.exit_code != 0 and not outside.exists()           # writes outside the folder are blocked by the OS
    sibling = home.parent / "secret.txt"
    sibling.write_text("TOPSECRET-42")
    r = sb.run(f"cat {sibling}")
    assert "TOPSECRET-42" not in r.output and "not permitted" in r.output   # reads outside the folder too


def test_size_limit_refuses_big_folders(user):
    home, state = user
    assert check_folder(home) == (3, len("alpha") + len("beta") + len("gamma"))
    with pytest.raises(FolderRefused, match="too big"):
        Sandbox.for_folder(home, "s1", state, max_files=2)
    with pytest.raises(FolderRefused, match="too big"):
        Sandbox.for_folder(home, "s1", state, max_bytes=8)
    assert not (state / "keel-s1").exists()                   # refused before anything was created


def test_overlapping_state_folder_is_refused(user):
    home, _ = user
    with pytest.raises(FolderRefused, match="must not contain"):
        Sandbox(home / ".keel-state", work=home)              # state inside the folder: create() would rmtree it
    with pytest.raises(FolderRefused, match="must not contain"):
        Sandbox(home.parent, work=home)                       # folder inside the state folder
    with pytest.raises(FolderRefused, match="not a folder"):
        Sandbox.for_folder(home / "a.txt", "s1", home.parent / "state")


def test_guard_asks_before_deleting_in_place(user):
    home, state = user
    asked = []
    sh = GuardedShell(Sandbox.for_folder(home, "s1", state), lambda cmd, files: asked.append(files) or False)
    res, note = sh.run("rm a.txt")
    assert res is None and "declined" in note and (home / "a.txt").read_text() == "alpha"
    assert asked == [["a.txt"]]
    ft = FileTools(sh)
    ft.delete(["b.txt"])                                      # the file tool asks the same confirmer
    assert (home / "b.txt").read_text() == "beta" and asked[-1] == ["b.txt"]


def test_guard_undoes_a_hidden_delete_in_place(user):
    home, state = user
    sh = GuardedShell(Sandbox.for_folder(home, "s1", state), approve_all)
    res, note = sh.run("perl -e 'unlink \"a.txt\"'")         # not caught by the delete pattern: the snapshot is
    assert "put back" in note and (home / "a.txt").read_text() == "alpha"
    assert (home / "a.txt").stat().st_mtime == OLD            # restored with its real date


def test_snapshot_falls_back_to_copy_across_disks(user, monkeypatch):
    home, state = user
    sh = GuardedShell(Sandbox.for_folder(home, "s1", state), approve_all)

    def no_link(*a, **k):
        raise OSError(18, "Cross-device link")                 # EXDEV: folder on another disk than the state
    monkeypatch.setattr(os, "link", no_link)
    res, note = sh.run("perl -e 'unlink \"sub/c.txt\"'")
    assert "put back" in note and (home / "sub" / "c.txt").read_text() == "gamma"


def test_path_rewrite_stays_correct_in_place(user):
    """Eval mode maps the sandbox root to `..` (it is the folder's parent). In place it is not, so only the folder
    itself is rewritten to `.`, and every printed path still works in the next command."""
    from keel.agent.loop import _relative
    home, state = user
    sb = Sandbox.for_folder(home, "s1", state)
    r = sb.run("pwd; t=$(mktemp); echo hi > $t; echo $t")
    lines = _relative(r.output.strip(), sb.work, sb.root).splitlines()
    assert lines[0] == "."
    assert not lines[1].startswith("../")                     # would point at the wrong place in place
    assert sb.run(f"cat {lines[1]}").output.strip() == "hi"


def test_remove_deletes_only_the_scratch_folder(user):
    home, scratch = user
    before = snapshot(home)
    sb = Sandbox.for_folder(home, "s1", scratch)
    sb.run("echo x > $TMPDIR/t.txt")
    assert (scratch / "keel-s1" / ".tmp" / "t.txt").exists()
    sb.remove()
    assert not (scratch / "keel-s1").exists() and snapshot(home) == before


def test_sweep_removes_only_old_keel_scratch(tmp_path):
    import time

    from keel.agent.sandbox import sweep_scratch
    old, fresh, other = tmp_path / "keel-s0123456789", tmp_path / "keel-sabcdefabcd", tmp_path / "keel-notes"
    for d in (old, fresh, other):
        d.mkdir()
    os.utime(old, (time.time() - 3 * 86400,) * 2)
    os.utime(other, (time.time() - 3 * 86400,) * 2)                # old, but not keel's session pattern
    assert sweep_scratch(scratch_dir=tmp_path) == 1
    assert not old.exists() and fresh.exists() and other.exists()
