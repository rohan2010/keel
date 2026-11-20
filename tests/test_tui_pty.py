"""P7: the real `keel` binary (full TUI) in a pseudo-terminal, with the scripted backend.

Drives it like a person: waits for the delete modal, presses Enter (must do nothing), then n (the file stays); and
kills the TUI with SIGKILL while the modal is open (the file stays, keel-core exits). Skipped if the binary is not
built (`cargo build`).
"""

import fcntl
import json
import os
import pty
import re
import select
import signal
import struct
import subprocess
import termios
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "debug" / "keel"
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[()][A-Z0-9]")
DELETE = '<tool_call>\n<function=delete>\n<parameter=paths>\n["old.log"]\n</parameter>\n</function>\n</tool_call>'

pytestmark = pytest.mark.skipif(not BIN.exists(), reason="keel binary not built (cargo build)")


class Pty:
    """Run a command in a pty that answers cursor-position queries like a real terminal."""

    def __init__(self, argv, cwd):
        self.pid, self.fd = pty.fork()
        if self.pid == 0:                                  # child: size the terminal BEFORE the UI starts
            fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
            os.chdir(cwd)
            os.execv(argv[0], argv)
        self.raw = ""
        self.answered = 0

    def read_until(self, pattern, timeout: float = 60):
        end = time.time() + timeout
        while time.time() < end:
            r, _, _ = select.select([self.fd], [], [], 0.2)
            if r:
                try:
                    chunk = os.read(self.fd, 65536).decode("utf-8", "replace")
                except OSError:
                    break
                self.raw += chunk
                while self.raw.count("\x1b[6n") > self.answered:  # every cursor query, even one split across reads
                    os.write(self.fd, b"\x1b[1;1R")              # "the cursor is at row 1, column 1"
                    self.answered += 1
                if re.search(pattern, self.text()):
                    return True
        return False

    def text(self):
        return ANSI.sub("", self.raw)

    def send(self, s):
        os.write(self.fd, s.encode())

    def wait(self, timeout=30):
        end = time.time() + timeout
        while time.time() < end:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                return os.waitstatus_to_exitcode(status)
            self.read_until(r"$^", timeout=0.2)
        raise TimeoutError("keel did not exit")


def _setup(tmp_path):
    folder = tmp_path / "proj"
    folder.mkdir()
    (folder / "old.log").write_text("log")
    script = tmp_path / "script.json"
    script.write_text(json.dumps([f"Cleaning up.\n{DELETE}", "Finished."]))
    state = tmp_path / "state"
    argv = [str(BIN), "run", "--dir", str(folder), "--backend", "scripted", "--script", str(script),
            "--state-dir", str(state), "tidy up"]                     # never the real ~/Library
    return folder, argv, state


def test_tui_delete_modal_enter_does_nothing_then_n_keeps(tmp_path, monkeypatch):
    folder, argv, _ = _setup(tmp_path)
    t = Pty(argv, tmp_path)
    assert t.read_until(r"Delete 1 file\?"), t.text()[-2000:]
    assert t.read_until(r"y\s+delete\s+n\s+keep"), t.text()[-2000:]
    t.send("\r")                                            # Enter: no default answer
    assert t.read_until(r"press y to delete or n to keep"), t.text()[-2000:]
    assert (folder / "old.log").exists()
    t.send("n")
    assert t.read_until(r"kept 1 file"), t.text()[-2000:]
    assert t.read_until(r"Finished\."), t.text()[-2000:]
    assert t.read_until(r"follow-up"), t.text()[-2000:]
    t.send("\x1b")                                          # Esc on an empty input: quit
    assert t.wait() == 0
    assert (folder / "old.log").read_text() == "log"
    text = t.text()
    assert "Cleaning up." in text and "<tool_call>" not in text


def test_tui_killed_mid_question_keeps_the_file(tmp_path):
    folder, argv, _ = _setup(tmp_path)
    t = Pty(argv, tmp_path)
    assert t.read_until(r"Delete 1 file\?"), t.text()[-2000:]
    os.kill(t.pid, signal.SIGKILL)                          # no cleanup code runs in keel
    # keep reading while it exits: a killed process can't finish closing its terminal until its unread output is
    # drained (a bare waitpid here deadlocked ~1 run in 4)
    assert t.wait() == -9
    time.sleep(1.5)
    assert (folder / "old.log").read_text() == "log"
    left = subprocess.run(["pgrep", "-f", f"--dir {folder}"], capture_output=True, text=True).stdout.strip()
    core = subprocess.run(["pgrep", "-f", "keel-core acp --backend scripted --script " + str(tmp_path)],
                          capture_output=True, text=True).stdout.strip()
    assert not core, f"keel-core still running: {core}"
    assert not left, f"keel still running: {left}"


def test_tui_without_a_task_waits_for_one(tmp_path):
    """Ch7 P6: `keel run` with no task opens the UI with a hint; the typed task runs."""
    folder, argv, _ = _setup(tmp_path)
    (tmp_path / "script.json").write_text(json.dumps(["Here is what I found."]))
    t = Pty(argv[:-1], tmp_path)                            # same command, task left out
    assert t.read_until(r"Type what you want done"), t.text()[-2000:]
    assert "▀" in t.text() and "a steady harness for small local models" in t.text()   # the mascot (any mood)
    assert t.read_until(r"type a task and press Enter"), t.text()[-2000:]
    t.send("list the files\r")
    assert t.read_until(r"Here is what I found\."), t.text()[-2000:]
    assert t.read_until(r"follow-up"), t.text()[-2000:]
    t.send("\x1b")
    assert t.wait() == 0
