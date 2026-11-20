"""Checks for the Module 6 change report (no model).

  uv run python tests/checks/check_verify.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

from keel.agent.verify import change_report, diff_states

fails = 0


def check(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {detail}"))
    fails += not cond


def main():
    root = Path(tempfile.mkdtemp())
    start, end = root / "start", root / "end"
    (start / "docs").mkdir(parents=True)
    (start / "docs" / "a.txt").write_text("one\ntwo\nthree\n")
    (start / "b.txt").write_text("keep me\n")
    (start / "old.txt").write_text("moving content\n")
    (start / "gone.txt").write_text("bye\n")
    (start / "img.jpg").write_bytes(b"\xff\xd8\x00\x01binary")
    shutil.copytree(start, end)

    # no change
    r = change_report(start, end, answer="I've created the file.")
    check("no change reported", "No files were created" in r, r)
    check("false claim flagged", "your answer describes changes, but no file was changed" in r, r)
    r = change_report(start, end, answer="I could not find the file.")
    check("no flag without a claim", "NOTE" not in r, r)

    # changes
    (end / "docs" / "a.txt").write_text("one\nTWO\nthree\n")
    (end / "old.txt").rename(end / "docs" / "new.txt")
    (end / "gone.txt").unlink()
    (end / "out.csv").write_text("date,total\n2025-01-01,10\n")
    (end / "img.jpg").write_bytes(b"\xff\xd8\x00\x02binary!")
    d = diff_states(start, end)
    check("move detected", d["moved"] == [("old.txt", "docs/new.txt")], d["moved"])
    check("delete detected", d["deleted"] == ["gone.txt"], d["deleted"])
    check("create detected", d["created"] == ["out.csv"], d["created"])
    check("modify detected", sorted(d["modified"]) == ["docs/a.txt", "img.jpg"], d["modified"])
    r = change_report(start, end, errors=["Error: no such file x.txt"], answer="Done!")
    check("diff shows changed line", "-two" in r and "+TWO" in r, r)
    check("new file head shown", "2025-01-01,10" in r, r)
    check("binary not dumped", "binary or large file changed" in r, r)
    check("errors listed", "no such file x.txt" in r, r)
    r2 = change_report(start, end, errors=[f"Traceback: File \"{end.resolve()}/rename.py\", line 9"])
    check("host path hidden in errors", str(end.resolve()) not in r2 and "rename.py" in r2, r2)
    check("no false-claim note when files changed", "NOTE" not in r, r)

    # cap
    (end / "big.txt").write_text("\n".join(f"line {i}" for i in range(500)))
    r = change_report(start, end, max_lines=20)
    check("cap respected", len(r.splitlines()) < 45, len(r.splitlines()))
    check("cap says more", "more lines" in r or "more diff lines" in r, r)

    shutil.rmtree(root)
    print("ALL CHECKS PASS" if not fails else f"{fails} CHECK(S) FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
