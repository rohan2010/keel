"""OS-enforced sandbox for the agent's shell (one folder, shell + files only inside it, no network).

Each episode gets its own directory:
    <episode>/work/   the user's folder (the agent's cwd; what tasks are scored on)
    <episode>/.tmp/   TMPDIR for tools that need scratch space
Commands run under macOS `sandbox-exec` with a profile that
  - denies all network access,
  - denies signals to any process outside the sandbox (the agent can't kill the harness or the user's programs;
    its own child processes it can still stop),
  - denies every file write outside work/ and .tmp/ (except /dev/null and the terminal devices),
  - denies reading anything under /Users except the episode directory (stat-only metadata stays allowed so that
    getcwd() works), so the agent cannot read the repo, the model, or the user's home; per-user temp/cache
    folders (/private/var/folders) are unreadable too.
macOS `mktemp` ignores TMPDIR and uses the per-user temp folder, which the profile blocks; an empty `$(mktemp -d)`
once turned a model's script into `mv /* .`. Each episode therefore gets <episode>/.bin/mktemp (read-only to the
agent, first on PATH): with no template it adds `-p "$TMPDIR"`; calls with a template or -p pass through unchanged.
System binaries (/bin, /usr, /opt/homebrew, ...) stay readable so the shell and its tools work.

Two modes:
  eval (default)   work/ is a copy of a task template inside the episode folder; the logical clock is on.
  in place         `Sandbox.for_folder(folder)`: work/ IS the user's folder (never copied, never removed); .tmp/,
                   .bin/ and the guard's snapshot live in a scratch folder in the system temp folder
                   (`<tmp>/keel-<session>`), removed when the session ends (`remove()`); the logical clock is OFF (it
                   would rewrite the user's file dates); folders over a size limit are refused (the delete guard
                   snapshots and hashes every file before each command).
`sandbox-exec` is deprecated by Apple but still enforced on macOS 26; tests/checks/check_sandbox.py checks every rule.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Command Line Tools first: /usr/bin/python3, git, ... are xcrun shims that need a cache outside the sandbox.
PATH = "/Library/Developer/CommandLineTools/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin"

MKTEMP_SHIM = r'''#!/bin/bash
# Harness shim: macOS mktemp ignores $TMPDIR and uses the per-user temp folder, which the sandbox blocks.
# No template given -> create in $TMPDIR. A template or -p/--tmpdir given -> unchanged.
explicit=0; want_value=0
for a in "$@"; do
  if [ "$want_value" = 1 ]; then want_value=0; continue; fi
  case "$a" in
    -p|--tmpdir|--tmpdir=*|-p?*) explicit=1 ;;
    -t|--suffix|-[dqu]*t) want_value=1 ;;
    --) ;;
    -*) ;;
    *) explicit=1 ;;
  esac
done
if [ "$explicit" = 1 ] || [ -z "$TMPDIR" ]; then exec /usr/bin/mktemp "$@"; fi
exec /usr/bin/mktemp -p "${TMPDIR%/}" "$@"
'''


# What any command may read besides the episode: an allowlist of system folders, so the
# rule does not depend on where the user's files or keel's checkout live. Not readable: /Users, /Volumes, /private/tmp,
# /private/var/folders (other users' and apps' temp files), anything else. File metadata (stat, ls of a path) stays
# readable everywhere, as before; only file contents are covered.
SYSTEM_READ = " ".join(f'(subpath "{p}")' for p in (
    "/System", "/usr", "/bin", "/sbin", "/Library", "/Applications", "/opt", "/private/etc", "/private/var/db",
    "/private/var/select", "/dev"))


def _profile(root: str, writable: list[str], extra_read: tuple[str, ...] = ()) -> str:
    q = root.replace('"', '\\"')
    more = "".join(f'\n(allow file-read-data (subpath "{x.replace(chr(34), chr(92) + chr(34))}"))' for x in extra_read)
    esc = [p.replace('"', '\\"') for p in writable]
    w = " ".join(f'(subpath "{x}")' for x in esc)
    # the writable folders themselves can't be removed (their contents can): `rm -rf "$PWD"` used to delete the
    # agent's folder (in place: the user's folder), and every later command failed
    keep = " ".join(f'(literal "{x}")' for x in esc)
    return f"""(version 1)
(allow default)
(deny network*)
(deny signal (target others))
(deny file-write*)
(allow file-write*
  {w}
  (literal "/dev/null") (literal "/dev/zero") (literal "/dev/dtracehelper")
  (regex #"^/dev/tty") (regex #"^/dev/fd/"))
(deny file-write-unlink {keep})
(deny file-read-data (subpath "/"))
(allow file-read-data (literal "/") {SYSTEM_READ})
(allow file-read-data (subpath "{q}") {w}){more}
"""


# Command output: read while the command runs, never all of it in memory. Up to KEEP_FULL
# bytes are kept whole, so the usual head + tail cut below is exactly what it was; past that only a head and a tail are
# kept and the cut is counted in bytes. A command that prints more than STOP_AFTER bytes (`yes`, `cat /dev/urandom`)
# is stopped.
KEEP_FULL = 1_000_000
STOP_AFTER = 100_000_000


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


class _BoundedReader:
    def __init__(self, proc: subprocess.Popen, keep_full: int, stop_after: int):
        import threading
        self.proc, self.keep_full, self.stop_after = proc, keep_full, stop_after
        self.head, self.tail, self.total, self.stopped = bytearray(), bytearray(), 0, False
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()

    def _read(self) -> None:
        f = self.proc.stdout
        assert f is not None
        while chunk := f.read1(65536) if hasattr(f, "read1") else f.read(65536):
            self.total += len(chunk)
            if len(self.head) < self.keep_full:
                take = self.keep_full - len(self.head)
                self.head += chunk[:take]
                chunk = chunk[take:]
            if chunk:
                self.tail = (self.tail + chunk)[-self.keep_full:]
            if self.total > self.stop_after and not self.stopped:
                self.stopped = True
                _kill(self.proc)

    def join(self, timeout: float) -> None:
        self._t.join(timeout)

    def alive(self) -> bool:
        return self._t.is_alive()

    def text(self, max_output: int) -> str:
        half = max_output // 2
        if self.total <= self.keep_full:                 # everything kept: the cut is exactly as before
            text = bytes(self.head).decode("utf-8", errors="replace")
            if len(text) > max_output:
                text = text[:half] + f"\n... [{len(text) - max_output} characters cut] ...\n" + text[-half:]
            return text
        head = bytes(self.head).decode("utf-8", errors="replace")[:half]
        tail = bytes(self.tail).decode("utf-8", errors="replace")[-half:]
        return head + f"\n... [about {self.total - 2 * half} characters cut] ...\n" + tail


@dataclass
class ShellResult:
    command: str
    exit_code: int
    output: str          # stdout + stderr, truncated
    seconds: float
    timed_out: bool


# Logical clock (eval repeatability). Without it, every file the task or the agent creates carries the wall-clock
# time, `ls -la` / list_files show it, the model reads different text on every run and greedy runs drift apart.
# Template files keep their task-set times (all before CLOCK_BASE); anything newer than CLOCK_REAL is real-clock and
# is re-stamped to CLOCK_BASE + step minutes after each tool call. Residual: a command that creates a file and lists
# it in the same call still sees the real time once.
CLOCK_BASE = datetime(2026, 1, 5, 9, 0).timestamp()
CLOCK_REAL = CLOCK_BASE + 30 * 86400


# In-place limits: the delete guard hashes and snapshots every file before each command.
MAX_FILES = 5000
MAX_BYTES = 500 * 1024 * 1024


class FolderRefused(ValueError):
    """The folder can't be used in place (too big, or overlapping keel's own state folder)."""


SCRATCH_PREFIX = "keel-"


def sweep_scratch(max_age_s: float = 86400, scratch_dir: Path | None = None) -> int:
    """Remove scratch folders left by sessions that crashed (only `keel-s<10 hex>` folders, older than a day)."""
    import re
    base = Path(scratch_dir or tempfile.gettempdir())
    n = 0
    for p in base.glob(f"{SCRATCH_PREFIX}s*"):
        if p.is_dir() and re.fullmatch(rf"{SCRATCH_PREFIX}s[0-9a-f]{{10}}", p.name) and \
                time.time() - p.stat().st_mtime > max_age_s:
            shutil.rmtree(p, ignore_errors=True)
            n += 1
    return n


def default_state_dir() -> Path:
    """`KEEL_STATE_DIR` if set (tests, a second profile), else ~/Library/Application Support/keel (keel.paths)."""
    from .. import paths
    return paths.state_dir()


def check_folder(folder: Path, max_files: int = MAX_FILES, max_bytes: int = MAX_BYTES) -> tuple[int, int]:
    """(files, bytes) under `folder`; raises FolderRefused as soon as either limit is passed (stops scanning early)."""
    n = size = 0
    for d, _dirs, files in os.walk(folder):
        for f in files:
            try:
                st = os.lstat(os.path.join(d, f))
            except OSError:
                continue
            n += 1
            size += st.st_size
            if n > max_files or size > max_bytes:
                raise FolderRefused(
                    f"{folder} is too big for keel to work in safely (over {max_files} files or "
                    f"{max_bytes // (1024 * 1024)} MB). Before every command keel keeps a copy of each file so a "
                    "delete can always be undone. Pick a smaller folder.")
    return n, size


def _inside(a: Path, b: Path) -> bool:
    """True if a is b or lies inside b."""
    return a == b or b in a.parents


class Sandbox:
    def __init__(self, episode_dir: Path, timeout: float = 30.0, max_output: int = 4000, *, work: Path | None = None):
        self.root = Path(episode_dir).resolve()
        self.in_place = work is not None
        self.work = Path(work).resolve() if self.in_place else self.root / "work"
        self.tmp = self.root / ".tmp"
        self.bin = self.root / ".bin"
        self.timeout = timeout
        self.max_output = max_output
        self.logical_clock = not self.in_place          # never re-stamp a real user's file dates
        if self.in_place and (_inside(self.work, self.root) or _inside(self.root, self.work)):
            raise FolderRefused(f"the folder {self.work} and keel's state folder {self.root} must not contain each "
                                "other")
        self.profile = _profile(str(self.root), [str(self.work), str(self.tmp)],
                                (str(self.work),) if self.in_place else ())

    @classmethod
    def for_folder(cls, folder: Path, session_id: str, scratch_dir: Path | None = None, *,
                   max_files: int = MAX_FILES, max_bytes: int = MAX_BYTES, **kw) -> "Sandbox":
        """In-place sandbox on the user's real folder (`keel run`). Checks the size limit, then creates only the
        scratch folder `<scratch_dir or system temp>/keel-<session_id>`; the user's folder is not modified."""
        folder = Path(folder).resolve()
        if not folder.is_dir():
            raise FolderRefused(f"{folder} is not a folder")
        check_folder(folder, max_files, max_bytes)
        root = Path(scratch_dir or tempfile.gettempdir()).resolve() / f"{SCRATCH_PREFIX}{session_id}"
        return cls(root, work=folder, **kw).create()

    def remove(self) -> None:
        """End of an in-place session: delete the scratch folder (never the user's folder: __init__ proved the two
        do not contain each other)."""
        if self.in_place:
            shutil.rmtree(self.root, ignore_errors=True)

    def create(self, template: Path | None = None) -> "Sandbox":
        """Fresh folder; copies `template` (preserving modified times) if given. In place: only the state folder is
        (re)made; the user's folder is left exactly as it is."""
        if self.in_place:
            if template is not None:
                raise ValueError("an in-place sandbox works on the user's folder, not a template")
            if self.root.exists():
                shutil.rmtree(self.root)                # state only: __init__ proved it does not contain work
            self.root.mkdir(parents=True)
            self.tmp.mkdir()
            self.bin.mkdir()
            (self.bin / "mktemp").write_text(MKTEMP_SHIM)
            (self.bin / "mktemp").chmod(0o755)
            return self
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        if template is not None:
            shutil.copytree(template, self.work, copy_function=shutil.copy2)
        else:
            self.work.mkdir()
        self.tmp.mkdir()
        self.bin.mkdir()
        (self.bin / "mktemp").write_text(MKTEMP_SHIM)
        (self.bin / "mktemp").chmod(0o755)
        self.settle_clock(0)
        return self

    def settle_clock(self, step: int) -> None:
        """Re-stamp everything touched since the last call (real-clock mtime) to the logical time of `step`.
        Eval mode only: a no-op in place, where the dates are the user's."""
        if not self.logical_clock:
            return
        t = CLOCK_BASE + 60 * step
        for d, dirs, files in os.walk(self.root):
            for name in [*dirs, *files]:
                q = os.path.join(d, name)
                try:
                    if os.lstat(q).st_mtime > CLOCK_REAL:
                        os.utime(q, (t, t), follow_symlinks=False)
                except OSError:
                    pass
        if os.stat(self.root).st_mtime > CLOCK_REAL:
            os.utime(self.root, (t, t))

    def hold_times(self) -> dict:
        """Eval mode: the root's and .tmp's times, so the guard's own bookkeeping (snapshot, dry run) can be undone
        before the agent's command runs. Known bug 2: without this, `ls -la` showed the real time of `..` during a
        command, so identical runs differed. Empty in place (real folders keep real times)."""
        if not self.logical_clock:
            return {}
        held = {}
        for p in (self.root, self.tmp):
            try:
                s = os.stat(p)
                held[p] = (s.st_atime, s.st_mtime)
            except OSError:
                pass
        return held

    def put_back_times(self, held: dict) -> None:
        """Undo the bookkeeping's time stamps before the agent's command runs (eval mode; `held` from hold_times).
 (2026-11-16): `ls -la ..` also lists the guard's `.snap` folder (made just before each command)
        and the sandbox's parent, both at real time, so E3_0 still split at message 39. `.snap` now gets the root's
        held time and the parent the clock's start."""
        for p, (a, m) in held.items():
            try:
                os.utime(p, (a, m))
            except OSError:
                pass
        if not held or self.root not in held:
            return
        a, m = held[self.root]
        for p, t in ((self.root / ".snap", (a, m)), (self.root.parent, (CLOCK_BASE, CLOCK_BASE))):
            try:
                if p.exists():
                    os.utime(p, t)
            except OSError:
                pass
        try:
            os.utime(self.root, (a, m))                  # touching .snap's time doesn't change the root's, but be sure
        except OSError:
            pass

    def run(self, command: str, cwd: Path | None = None, writable: list[Path] | None = None) -> ShellResult:
        """Run in work/ (or `cwd`); `writable` narrows the writable folders (used for dry runs on a scratch copy).
        HOME is the folder the command runs in, so a dry run's `~` / `$HOME` point at the copy, not the real folder
        (2026-11-19: `rm -rf ~/*` deleted nothing in the dry run, so the user was never asked)."""
        here = Path(cwd) if cwd is not None else self.work
        if not here.is_dir():                            # never crash the session on a missing folder
            return ShellResult(command, 1, f"Error: the folder {here.name} no longer exists.", 0.0, False)
        env = {"PATH": f"{self.bin}:{PATH}" if self.bin.is_dir() else PATH, "HOME": str(here),
               "TMPDIR": str(self.tmp) + "/", "LANG": "en_US.UTF-8", "TERM": "dumb", "PYTHONDONTWRITEBYTECODE": "1"}
        t0 = time.perf_counter()
        profile = self.profile if writable is None else _profile(str(self.root), [str(Path(x).resolve()) for x in writable])
        proc = subprocess.Popen(["sandbox-exec", "-p", profile, "/bin/bash", "-c", command], cwd=here,
                                env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)
        out = _BoundedReader(proc, KEEP_FULL, STOP_AFTER)
        timed_out = False
        try:
            proc.wait(timeout=self.timeout)
            out.join(max(0.1, self.timeout - (time.perf_counter() - t0)))   # a child may still hold the pipe
            if out.alive():
                raise subprocess.TimeoutExpired(command, self.timeout)
        except subprocess.TimeoutExpired:
            timed_out = not out.stopped
            _kill(proc)
            out.join(5)
        text = out.text(self.max_output)
        if out.stopped:
            text += f"\n[command stopped: it printed more than {STOP_AFTER // 1_000_000} MB]"
        if timed_out:
            text += f"\n[command stopped after {self.timeout:.0f} s]"
        code = -9 if (timed_out or out.stopped) else proc.returncode
        return ShellResult(command, code, text, time.perf_counter() - t0, timed_out)
