"""Tests for the agent file tools (src/keel/agent/filetools.py): behaviour and folder confinement.

  uv run python tests/checks/check_filetools.py
"""

import os
import sys
import tempfile
from pathlib import Path

from keel.agent.filetools import FileTools, ToolError
from keel.agent.guard import GuardedShell
from keel.agent.sandbox import Sandbox

ROOT = Path(__file__).resolve().parents[2]
(ROOT / "experiments").mkdir(exist_ok=True)  # temp sandboxes live under the repo, not /tmp
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name} {info}")


def err(fn, *a, **k):
    try:
        fn(*a, **k)
        return None
    except ToolError as e:
        return str(e)


with tempfile.TemporaryDirectory(dir=ROOT / "experiments") as d:
    sb = Sandbox(Path(d) / "ep").create()
    w = sb.work
    (w / "a.txt").write_text("btn-primary\nbtn-primary-outline\nx\n")
    (w / "sub").mkdir()
    (w / "sub" / "b.md").write_text("hello\n")
    os.utime(w / "a.txt", (1600000000, 1600000000))
    os.symlink(ROOT, w / "link")
    asked = []
    g = GuardedShell(sb, lambda c, f: (asked.append(f), True)[1])
    t = FileTools(g)
    # confinement
    check("rejects ../ escape", err(t.read_file, "../../pyproject.toml"))
    check("rejects absolute path outside", err(t.read_file, str(ROOT / "pyproject.toml")))
    check("rejects symlink escape", err(t.read_file, "link/pyproject.toml"))
    check("rejects write outside", err(t.write_file, "../evil.txt", "x") and not (Path(d) / "ep" / "evil.txt").exists())
    check("rejects deleting the folder", err(t.delete, ["."]))
    # behaviour
    lst = t.list_files()
    check("list shows modified date", "modified 2020-09-13" in lst, lst.splitlines()[0])
    check("read numbers lines", "    1  btn-primary" in t.read_file("a.txt"))
    check("edit refuses ambiguous text", "occurs 2 times" in (err(t.edit_file, "a.txt", "btn-primary", "button-main") or ""))
    out = t.edit_file("a.txt", "btn-primary\n", "button-main\n")
    check("exact edit hits one line", (w / "a.txt").read_text() == "button-main\nbtn-primary-outline\nx\n", out.splitlines()[0])
    check("search finds", "a.txt:2" in t.search("outline"))
    check("search ignores case by default", "a.txt:2" in t.search("OUTLINE") and "(no matches)" == t.search("OUTLINE", ignore_case=False))
    (w / "w.css").write_text(".btn-primary{}\n.btn-primary-outline{}\n")
    out2 = t.edit_file("w.css", "btn-primary", "button-main", replace_all=True)
    check("warns on replacement inside a longer name", "inside a longer name: btn-primary-outline" in out2, out2.splitlines()[1][:80])
    out3 = t.edit_file("w.css", ".button-main{", ".btn-primary{")
    check("no warning for a clean edit", "Warning" not in out3)
    check("regex search with word boundary", "(no matches)" == t.search(r"btn-primary(?![\w-])", path="a.txt", regex=True))
    t.write_file("new/dir/c.txt", "c")
    check("write creates folders", (w / "new/dir/c.txt").read_text() == "c")
    check("move refuses overwrite", err(t.move, "new/dir/c.txt", "a.txt"))
    t.move("sub/b.md", "new")
    check("move into folder", (w / "new/b.md").exists() and not (w / "sub/b.md").exists())
    check("delete asks with exact files", t.delete(["new"]).startswith("Deleted 2") and asked[-1] == ["new/b.md", "new/dir/c.txt"])
    g2 = GuardedShell(sb, lambda c, f: False)
    t2 = FileTools(g2)
    check("declined delete keeps files", "declined" in t2.delete(["a.txt"]) and (w / "a.txt").exists())
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
