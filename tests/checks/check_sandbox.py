"""Escape tests for the agent sandbox. Every 'must fail' probe has to fail, every 'must work' probe has to work.

  uv run python tests/checks/check_sandbox.py
"""

import sys
import tempfile
from pathlib import Path

from keel.agent.filetools import FileTools
from keel.agent.guard import GuardedShell, approve_all
from keel.agent.sandbox import Sandbox

ROOT = Path(__file__).resolve().parents[2]
(ROOT / "experiments").mkdir(exist_ok=True)  # temp sandboxes live under the repo, not /tmp

MUST_WORK = [
    ("list the folder", "ls -la"),
    ("write a file", "echo hi > a.txt && cat a.txt"),
    ("make folders and move", "mkdir -p x/y && mv a.txt x/y/ && ls x/y"),
    ("python inside", "python3 -c \"open('p.txt','w').write('ok'); print(open('p.txt').read())\""),
    ("temp files (python)", "python3 -c \"import tempfile; f=tempfile.NamedTemporaryFile(delete=False); f.write(b't'); print(f.name)\""),
    ("node inside", "node -e \"require('fs').writeFileSync('n.txt','ok'); console.log('ok')\""),
    ("find + stat", "find . -name '*.txt' && stat -f '%Sm' p.txt"),
    ("stop its own child process", "sleep 30 & kill $! && wait $! ; echo stopped"),
    ("pipes and sed", "printf 'a\\nb\\n' | sed 's/a/c/' | sort"),
    # mktemp shim (macOS mktemp ignores TMPDIR; an empty $(mktemp -d) once turned a script into `mv /* .`)
    ("mktemp -d -> episode .tmp", 'd=$(mktemp -d) && [ -d "$d" ] && case "$d" in "${TMPDIR%/}"/*) echo "$d";; *) exit 1;; esac'),
    ("mktemp file -> episode .tmp", 'f=$(mktemp) && echo x > "$f" && case "$f" in "${TMPDIR%/}"/*) echo "$f";; *) exit 1;; esac'),
    ("mktemp -t prefix", 'f=$(mktemp -t photos) && case "$f" in "${TMPDIR%/}"/photos.*) echo "$f";; *) exit 1;; esac'),
    ("mktemp -dt prefix", 'd=$(mktemp -dt photos) && [ -d "$d" ] && case "$d" in "${TMPDIR%/}"/photos.*) echo "$d";; *) exit 1;; esac'),
    ("mktemp template in cwd", 'f=$(mktemp out.XXXX) && [ -f "./$f" ] && echo "$f"'),
    ("mktemp template in TMPDIR", 'd=$(mktemp -d "$TMPDIR/x.XXXX") && [ -d "$d" ] && echo "$d"'),
    ("bash script using mktemp", "printf '#!/bin/bash\\nT=$(mktemp -d) || exit 1\\ntouch \"$T/a\" && mv \"$T\"/* . && ls a\\n' > s.sh && bash s.sh"),
]

MUST_FAIL = [
    ("write to parent folder", "echo x > ../escape.txt"),
    ("write via cd ..", "cd .. && cd .. && echo x > escape2.txt"),
    ("write to repo", f"echo x > {ROOT}/ESCAPED.txt"),
    ("write to home", "echo x > ~/../../ESCAPED_HOME.txt"),
    ("write to /tmp", "echo x > /tmp/escaped.txt"),
    ("read the repo", f"cat {ROOT}/pyproject.toml"),
    ("list the repo", f"ls {ROOT}"),
    ("read user temp folders", "ls /private/var/folders"),
    ("read user home", "ls /Users/$(whoami 2>/dev/null || echo x)/Desktop"),
    ("python write outside", f"python3 -c \"open('{ROOT}/ESCAPED_PY.txt','w').write('x')\""),
    ("network: curl", "curl -sS -m 5 https://example.com"),
    ("network: python socket", "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 80), timeout=5)\""),
    ("symlink out and write", f"ln -s {ROOT} link && echo x > link/ESCAPED_LINK.txt"),
    ("overwrite the mktemp shim", "echo 'exit 0' > ../.bin/mktemp"),
    ("delete the mktemp shim", "rm ../.bin/mktemp"),
    ("signal the harness (parent)", "kill -0 $PPID"),
    ("signal launchd (pid 1)", "kill -0 1"),
]


def deletion_canaries(d: Path) -> bool:
    """Stand-in personal files just outside the episode; the user approves every deletion (so only the sandbox and
    the file tools' path check stand in the way). Every attempt must leave them intact."""
    canary = d / "PERSONAL.txt"
    canary.write_text("keep me")
    photos = d / "Photos"
    photos.mkdir()
    (photos / "a.jpg").write_text("photo")
    sb = Sandbox(d / "ep_del").create()
    sh = GuardedShell(sb, approve_all)
    attempts = [f"rm {canary}", f"rm -rf {photos}", "rm -rf ../../PERSONAL.txt ../../Photos", "rm -rf ~/../../*",
                f"python3 -c \"import os; os.remove('{canary}')\"",
                f"python3 -c \"import shutil; shutil.rmtree('{photos}')\"",
                f"find {d} -name '*.txt' -delete", f"mv {canary} ./stolen.txt", f"echo > {canary}",
                f"ln -s {photos} p && rm -rf p/ p/*", f"cd {d} && rm -rf PERSONAL.txt Photos"]
    ok = True
    for cmd in attempts:
        sh.run(cmd)
        intact = canary.exists() and canary.read_text() == "keep me" and (photos / "a.jpg").exists()
        ok &= intact
        print(f"{'PASS' if intact else 'FAIL'}  outside files intact after: {cmd[:70]}")
    ft = FileTools(sh)
    for args in (["../../PERSONAL.txt"], [str(canary)], [str(photos)], ["p"]):
        try:
            ft.delete(args)
            refused = False
        except Exception:
            refused = True
        intact = refused and canary.exists() and (photos / "a.jpg").exists()
        ok &= intact
        print(f"{'PASS' if intact else 'FAIL'}  delete tool refused: {args[0][-40:]}")
    return ok


def main():
    ok = True
    with tempfile.TemporaryDirectory(dir=ROOT / "experiments") as d:
        sb = Sandbox(Path(d) / "ep").create()
        for name, cmd in MUST_WORK:
            r = sb.run(cmd)
            good = r.exit_code == 0
            ok &= good
            print(f"{'PASS' if good else 'FAIL'}  works:   {name:28s} exit={r.exit_code} {r.output.strip()[:60]!r}")
        for name, cmd in MUST_FAIL:
            r = sb.run(cmd)
            good = r.exit_code != 0
            ok &= good
            print(f"{'PASS' if good else 'FAIL'}  blocked: {name:28s} exit={r.exit_code} {r.output.strip()[:70]!r}")
        ok &= deletion_canaries(Path(d))
        sb.timeout = 3
        r = sb.run("sleep 60")
        ok &= r.timed_out
        print(f"{'PASS' if r.timed_out else 'FAIL'}  timeout: sleep 60 stopped after {r.seconds:.0f} s")
    leaked = [p for p in [ROOT / "ESCAPED.txt", ROOT / "ESCAPED_PY.txt", ROOT / "ESCAPED_LINK.txt", Path("/tmp/escaped.txt")]
              if p.exists()]
    if leaked:
        ok = False
        print("FAIL  files written outside:", leaked)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
