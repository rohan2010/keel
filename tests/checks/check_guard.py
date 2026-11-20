"""Tests for the delete guard (src/keel/agent/guard.py). Each case: a command, whether the user is asked, what the
user answers, and which files must exist afterwards.

  uv run python tests/checks/check_guard.py
"""

import sys
import tempfile
from pathlib import Path

from keel.agent.guard import GuardedShell
from keel.agent.sandbox import Sandbox

ROOT = Path(__file__).resolve().parents[2]
(ROOT / "experiments").mkdir(exist_ok=True)  # temp sandboxes live under the repo, not /tmp
BASE = {"a.txt": "A", "b.txt": "B", "c.txt": "C", "dup.txt": "A", "sub/d.txt": "D", "x.txt": "X"}

# (name, command, answer, expect_asked, files that must exist after, files that must NOT exist after)
CASES = [
    ("rm, user says yes", "rm a.txt", True, True, ["b.txt"], ["a.txt"]),
    ("rm, user says no", "rm a.txt", False, True, ["a.txt"], []),
    ("rm -rf * .* , user says no", "rm -rf * .* 2>/dev/null; echo done", False, True, list(BASE), []),
    ("find -delete, user says no", "find . -name '*.txt' -delete", False, True, list(BASE), []),
    ("python os.remove, user says no", "python3 -c \"import os; os.remove('b.txt')\"", False, True, ["b.txt"], []),
    ("hidden delete (not matched) -> restored", "python3 -c \"import os; getattr(os,'re'+'move')('b.txt')\"", True,
     False, ["b.txt"], []),
    ("dry run misses it -> safety net restores", "cd ../../work 2>/dev/null; rm x.txt", False, False,
     ["x.txt"], []),
    ("rm a duplicate, user says yes", "rm dup.txt", True, True, ["a.txt"], ["dup.txt"]),
    ("mv / rename: no question", "mkdir -p T && mv a.txt T/a.txt && mv b.txt bb.txt", True, False, ["T/a.txt", "bb.txt"],
     ["a.txt"]),
    ("sed -i edit: no question", "sed -i '' 's/C/CC/' c.txt", True, False, ["c.txt"], []),
    ("rm of a missing file: no question", "rm -f nothere.txt", True, False, list(BASE), []),
]


def main():
    ok_all = True
    for name, cmd, answer, expect_asked, must, mustnot in CASES:
        with tempfile.TemporaryDirectory(dir=ROOT / "experiments") as d:
            sb = Sandbox(Path(d) / "ep").create()
            for rel, txt in BASE.items():
                (sb.work / rel).parent.mkdir(parents=True, exist_ok=True)
                (sb.work / rel).write_text(txt)
            asked = []
            g = GuardedShell(sb, lambda c, files, asked=asked, answer=answer: (asked.append(files), answer)[1])
            res, note = g.run(cmd)
            exist_ok = all((sb.work / f).exists() for f in must) and not any((sb.work / f).exists() for f in mustnot)
            good = exist_ok and (bool(asked) == expect_asked)
            ok_all &= good
            print(f"{'PASS' if good else 'FAIL'} {name:52s} asked={asked[0] if asked else '-'} "
                  f"ran={res is not None} {('note: ' + note.strip()[:70]) if note else ''}")
            if not exist_ok:
                print("     files now:", sorted(str(p.relative_to(sb.work)) for p in sb.work.rglob('*') if p.is_file()))
    print("ALL PASS" if ok_all else "SOME FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
