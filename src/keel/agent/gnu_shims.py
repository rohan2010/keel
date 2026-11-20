"""Linux-style shell tools on a Mac.

Small models write Linux (GNU) shell; macOS has BSD tools that reject GNU options. In scoreboard run 1, 18 episodes
hit `stat: illegal option -- c`, `uniq: invalid option -- w` and the like; 15 of them failed (every E3 episode ran
out of steps retrying). These shims sit first on the sandbox's PATH (`<episode>/.bin`, like the mktemp shim) and do
what the GNU tool would for the GNU-only options the models use; any other use goes straight to the real macOS tool,
unchanged. Python standard library only (`/usr/bin/python3`, Command Line Tools). Off unless `--gnu-tools`.

Covered: stat -c / --format / --printf (GNU format letters), uniq -w / --check-chars (with -c -d -u -i), date -d /
--date (@seconds, ISO dates) with an optional +FORMAT. Not covered on purpose: anything that sends mail or talks to
the network.
"""

from __future__ import annotations

from pathlib import Path

PY = "#!/usr/bin/python3\n"           # placeholder: install() writes the real interpreter's path

# /usr/bin/python3 is macOS's xcrun stub: inside the sandbox it prints "couldn't create cache file" twice on every call
# (its cache folder is blocked), and that would land in every tool output. The Command Line Tools' own python3 doesn't.
PYTHONS = ("/Library/Developer/CommandLineTools/usr/bin/python3",
           "/Applications/Xcode.app/Contents/Developer/usr/bin/python3", "/usr/bin/python3")


def interpreter() -> str:
    import os
    return next((p for p in PYTHONS if os.access(p, os.X_OK)), PYTHONS[-1])

STAT = PY + r'''
import os, pwd, grp, stat as S, sys, time
args = sys.argv[1:]
fmt = None; nl = True; follow = False; files = []
i = 0
while i < len(args):
    a = args[i]
    if a in ("-c", "--format") and i + 1 < len(args): fmt, nl = args[i + 1], True; i += 2; continue
    if a.startswith("--format="): fmt, nl = a.split("=", 1)[1], True; i += 1; continue
    if a.startswith("--printf="): fmt, nl = a.split("=", 1)[1], False; i += 1; continue
    if a == "--printf" and i + 1 < len(args): fmt, nl = args[i + 1], False; i += 2; continue
    if a in ("-L", "--dereference"): follow = True; i += 1; continue
    files.append(a); i += 1
if fmt is None:
    os.execv("/usr/bin/stat", ["stat"] + args)        # not a GNU form: the real macOS stat
def human(t):
    lt = time.localtime(t); frac = int((t % 1) * 1e9)
    return time.strftime("%Y-%m-%d %H:%M:%S", lt) + ".%09d " % frac + time.strftime("%z", lt)
def kind(m):
    return ("directory" if S.S_ISDIR(m) else "regular file" if S.S_ISREG(m) else "symbolic link" if S.S_ISLNK(m)
            else "fifo" if S.S_ISFIFO(m) else "socket" if S.S_ISSOCK(m) else "file")
def one(path):
    st = os.stat(path) if follow else os.lstat(path)
    def user(u):
        try: return pwd.getpwuid(u).pw_name
        except KeyError: return str(u)
    def group(g):
        try: return grp.getgrgid(g).gr_name
        except KeyError: return str(g)
    birth = getattr(st, "st_birthtime", st.st_ctime)
    val = {"n": path, "N": "'%s'" % path, "s": str(st.st_size), "Y": str(int(st.st_mtime)), "y": human(st.st_mtime),
           "X": str(int(st.st_atime)), "x": human(st.st_atime), "Z": str(int(st.st_ctime)), "z": human(st.st_ctime),
           "W": str(int(birth)), "w": human(birth), "F": kind(st.st_mode), "a": oct(S.S_IMODE(st.st_mode))[2:],
           "A": S.filemode(st.st_mode), "u": str(st.st_uid), "g": str(st.st_gid), "U": user(st.st_uid),
           "G": group(st.st_gid), "i": str(st.st_ino), "h": str(st.st_nlink), "b": str(getattr(st, "st_blocks", 0)),
           "%": "%"}
    out, k = [], 0
    while k < len(fmt):
        c = fmt[k]
        if c == "%" and k + 1 < len(fmt): out.append(val.get(fmt[k + 1], "%" + fmt[k + 1])); k += 2; continue
        if c == "\\" and k + 1 < len(fmt): out.append({"n": "\n", "t": "\t", "\\": "\\"}.get(fmt[k + 1], fmt[k + 1])); k += 2; continue
        out.append(c); k += 1
    return "".join(out)
code = 0
for f in files:
    try:
        sys.stdout.write(one(f) + ("\n" if nl else ""))
    except OSError as e:
        sys.stderr.write("stat: cannot stat '%s': %s\n" % (f, e.strerror)); code = 1
sys.exit(code)
'''

UNIQ = PY + r'''
import os, sys
args = sys.argv[1:]
if not any(a in ("-w", "--check-chars") or a.startswith("--check-chars=") or (a.startswith("-w") and a[2:].isdigit())
           for a in args):
    os.execv("/usr/bin/uniq", ["uniq"] + args)        # not a GNU form: the real macOS uniq
count = dup = uniqonly = ignore = False; width = None; files = []
i = 0
while i < len(args):
    a = args[i]
    if a in ("-w", "--check-chars"): width = int(args[i + 1]); i += 2; continue
    if a.startswith("--check-chars="): width = int(a.split("=", 1)[1]); i += 1; continue
    if a.startswith("-w") and a[2:].isdigit(): width = int(a[2:]); i += 1; continue
    if a.startswith("-") and len(a) > 1 and not a.startswith("--"):
        for ch in a[1:]:
            count |= ch == "c"; dup |= ch == "d"; uniqonly |= ch == "u"; ignore |= ch == "i"
        i += 1; continue
    files.append(a); i += 1
src = open(files[0]) if files else sys.stdin
def key(line):
    k = line.rstrip("\n")
    k = k[:width] if width is not None else k
    return k.lower() if ignore else k
groups, prev, n, first = [], None, 0, None
for line in src:
    k = key(line)
    if n and k == prev: n += 1; continue
    if n: groups.append((n, first))
    prev, n, first = k, 1, line
if n: groups.append((n, first))
for n, line in groups:
    if (dup and n < 2) or (uniqonly and n > 1): continue
    sys.stdout.write(("%7d " % n if count else "") + line if line.endswith("\n") else (("%7d " % n if count else "") + line + "\n"))
'''

DATE = PY + r'''
import os, sys, time
from datetime import datetime
args = sys.argv[1:]
when = None; fmt = None; rest = []
i = 0
while i < len(args):
    a = args[i]
    if a in ("-d", "--date") and i + 1 < len(args): when = args[i + 1]; i += 2; continue
    if a.startswith("--date="): when = a.split("=", 1)[1]; i += 1; continue
    if a.startswith("+"): fmt = a[1:]; i += 1; continue
    rest.append(a); i += 1
if when is None:
    os.execv("/bin/date", ["date"] + args)            # not a GNU form: the real macOS date
w = when.strip()
try:
    if w.startswith("@"):
        t = float(w[1:])
    else:
        for f in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d"):
            try:
                t = time.mktime(datetime.strptime(w.split(".")[0], f).timetuple()); break
            except ValueError:
                continue
        else:
            raise ValueError(w)
except ValueError:
    sys.stderr.write("date: invalid date '%s'\n" % when); sys.exit(1)
print(time.strftime(fmt or "%a %b %e %H:%M:%S %Z %Y", time.localtime(t)))
'''

SHIMS = {"stat": STAT, "uniq": UNIQ, "date": DATE}


def install(bin_dir: Path) -> list[str]:
    """Write the shims into the sandbox's .bin (first on its PATH)."""
    bin_dir.mkdir(exist_ok=True)
    py = interpreter()
    for name, text in SHIMS.items():
        p = bin_dir / name
        p.write_text(text.replace(PY, f"#!{py}\n", 1))
        p.chmod(0o755)
    return sorted(SHIMS)
