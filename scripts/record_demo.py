"""Record a keel demo as an asciinema cast (v2): the real `keel` TUI with the local model, in a pseudo-terminal.

    uv run python scripts/record_demo.py <out.cast> [keel run flags, e.g. --backend mlx-server]

The story: tidy a messy Downloads folder; the delete question for the duplicate copies is answered "n"; a follow-up
("ok, delete the duplicates") is answered "y". Keystrokes are typed with small pauses so the cast reads like a person.
Also writes <out>.meta.json (folder before/after). Make the shareable GIF + MP4 (title card, captions, sped up) with
`uv run python scripts/cut_demo.py out.cast <stem>`.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_tui_pty import BIN, Pty  # noqa: E402

W, H = 100, 30
ASK = r"Delete \d+ files?\?"
IDLE = r"type a follow-up"
FILES = {
    "IMG_2041.jpg": b"\xff\xd8 photo of the beach",
    "IMG_2041 (1).jpg": b"\xff\xd8 photo of the beach",
    "IMG_2077.jpg": b"\xff\xd8 photo of the city",
    "invoice_march.pdf": b"%PDF invoice march",
    "invoice_march copy.pdf": b"%PDF invoice march",
    "notes.txt": b"call the bank on monday\n",
    "song.mp3": b"ID3 a song",
    "report_final.docx": b"PK report",
}


class Recorder(Pty):
    def __init__(self, argv, cwd):
        super().__init__(argv, cwd)
        self.t0 = time.time()
        self.frames: list = []
        self._seen = 0

    def read_until(self, pattern, timeout: float = 60):
        # Read in short slices and stamp each slice with its own time, so the cast shows output as it arrived
        # (one long wait used to land as a single frame: 48 s of blank screen, then the whole first turn at once).
        end = time.time() + timeout
        while True:
            ok = super().read_until(pattern, min(0.1, max(0.0, end - time.time())))
            new = self.raw[self._seen:]
            if new:
                self.frames.append([round(time.time() - self.t0, 3), "o", new])
                self._seen = len(self.raw)
            if ok or time.time() >= end:
                return ok

    def count(self, pattern):
        return len(re.findall(pattern, self.text()))

    def wait_more(self, pattern, have, timeout: float = 300):
        """Wait until `pattern` occurs more than `have` times."""
        end = time.time() + timeout
        while time.time() < end:
            self.read_until(r"$^", timeout=0.5)
            if self.count(pattern) > have:
                return True
        return False

    def type(self, text, delay=0.08):
        for ch in text:
            self.send(ch)
            self.read_until(r"$^", timeout=delay)


def main(out: str, extra: list[str] | None = None) -> int:
    """`extra` = more `keel run` flags, e.g. `--backend mlx-server` (Chapter 3 demo on a running mlx_lm.server)."""
    work = Path(tempfile.mkdtemp(prefix="keel-demo-"))
    folder = work / "Downloads"
    folder.mkdir()
    for name, data in FILES.items():
        (folder / name).write_bytes(data)
    env_cwd = work
    t = Recorder([str(BIN), "run", "--dir", str(folder), *(extra or []),
                  "Sort these files into folders by type (Images, Documents, Music) and remove duplicate copies."],
                 env_cwd)
    asked = t.read_until(r"Delete \d+ files?\?|follow-up", timeout=300)
    if re.search(r"Delete \d+ files?\?", t.text()):
        t.read_until(r"$^", timeout=2.5)                     # let the viewer read the question
        t.type("n")
        # The model may ask again on its own before the turn ends (seen on mlx-server, P9 (4)). Never type text into
        # an open question (a "y" in the follow-up would answer it): every question before the follow-up gets "n".
        for _ in range(5):
            asks, idles = t.count(ASK), t.count(IDLE)
            end = time.time() + 300
            while time.time() < end and t.count(ASK) <= asks and t.count(IDLE) <= idles:
                t.read_until(r"$^", timeout=0.5)
            if t.count(ASK) > asks:
                t.read_until(r"$^", timeout=2.5)
                t.type("n")
                continue
            break
        t.read_until(r"$^", timeout=1.5)
        asks, idles = t.count(ASK), t.count(IDLE)
        t.type("ok, you can delete the duplicate copies now\r", delay=0.05)
        end = time.time() + 300
        while time.time() < end and t.count(ASK) <= asks and t.count(IDLE) <= idles:
            t.read_until(r"$^", timeout=0.5)
        if t.count(ASK) > asks:
            t.read_until(r"$^", timeout=2.5)
            t.type("y")
            t.wait_more(IDLE, idles)
    t.read_until(r"$^", timeout=2.0)
    t.send("\x1b")
    code = t.wait(60)
    header = {"version": 2, "width": W, "height": H, "timestamp": int(t.t0), "title": "keel: a local model that asks before deleting",
              "env": {"TERM": "xterm-256color", "SHELL": "/bin/zsh"}}
    with open(out, "w") as f:
        f.write(json.dumps(header) + "\n")
        for fr in t.frames:
            f.write(json.dumps(fr) + "\n")
    tree = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
    meta = {"before": sorted(FILES), "after": tree, "exit": code, "extra": extra or []}
    Path(out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")   # read by scripts/cut_demo.py
    print(f"exit {code}; asked={asked}; {len(t.frames)} frames, {t.frames[-1][0] if t.frames else 0:.0f}s -> {out}")
    print("folder after:", tree)
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
