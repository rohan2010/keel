"""Silent backups (2026-11-19, closes the delete guard's overwrite gap; idea from Mingbird, arXiv 2610.02001).

The delete guard asks before any file is deleted, but overwriting a file's content (`mv a b` onto an existing b,
`cp a b`, `echo x > b`, `sed -i`, a script writing over b, `write_file` / `edit_file`) used to destroy b's earlier
content without a trace. Now, before any change, keel keeps the earlier version of every file whose content changes
or that is deleted with the user's approval, so the user can always get it back (`keel-core backups`).

Silent on purpose: the model is never told and never sees the copies. They live outside the sandbox (eval runs:
`<run folder>/backups/<item>`; sessions on a real folder: `<keel state folder>/backups/<session>`), so `ls`, the
logical clock and the delete guard's snapshot are unchanged, and so are results (golden gate 2026-11-19).

How a shell command is covered: just before it runs, every file is cloned into a staging folder (APFS clonefile:
copy-on-write, near-free; plain copy on other disks, up to a size cap). Afterwards, the staged copy of each file
whose content changed, or that was deleted with approval, is kept; the rest of the stage is dropped. (The guard's
own snapshot uses hard links, which share content with the live file, so it cannot recover an in-place overwrite.)
A file whose content moved to a new path (mv, rename) is not a change. Hidden deletions are put back by the guard
itself and need no backup.

Each kept copy is `<dir>/<n>/<relative path>`, with one line per copy in `<dir>/index.jsonl`.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

COPY_CAP = 200 * 1024 * 1024          # without clonefile, stage at most this many bytes per command (else: skip)

_libc = None


def _clone(src: Path, dst: Path) -> bool:
    """APFS copy-on-write clone (macOS clonefile); False if not possible (other file system, other volume)."""
    global _libc
    try:
        if _libc is None:
            _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        return _libc.clonefile(os.fsencode(src), os.fsencode(dst), 0) == 0
    except (OSError, AttributeError):
        return False


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not _clone(src, dst):
        shutil.copy2(src, dst)


def default_dir(name: str) -> Path:
    return Path(tempfile.gettempdir()) / "keel-backups" / name


class Backups:
    def __init__(self, directory: Path, work: Path | None = None):
        """`work` = the folder being protected (None: read / restore only)."""
        self.dir = Path(directory).resolve()
        self.work = Path(work).resolve() if work is not None else self.dir / ".none"
        if work is not None and (self.dir == self.work or self.work in self.dir.parents):
            raise ValueError(f"backups ({self.dir}) must be outside the folder being worked on ({self.work})")
        self.stage_dir = self.dir / ".stage"
        self.staged: dict[str, Path] = {}
        self.skipped = 0                                  # commands whose stage was skipped (too big, no clone)
        self._n = None

    # -- bookkeeping
    def _next(self) -> int:
        if self._n is None:
            self._n = len(self.entries())
        self._n += 1
        return self._n

    def entries(self) -> list[dict]:
        idx = self.dir / "index.jsonl"
        if not idx.exists():
            return []
        return [json.loads(l) for l in idx.read_text().splitlines() if l.strip()]

    def _record(self, src: Path, rel: str, reason: str, command: str) -> dict:
        n = self._next()
        dst = self.dir / str(n) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.parent == self.stage_dir or self.stage_dir in src.parents:
            os.replace(src, dst)                          # a staged clone: just move it
        else:
            _copy(src, dst)
        e = {"n": n, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "path": rel, "folder": str(self.work),
             "reason": reason, "command": command[:300], "bytes": dst.stat().st_size,
             "sha1": hashlib.sha1(dst.read_bytes()).hexdigest()}
        with open(self.dir / "index.jsonl", "a") as f:
            f.write(json.dumps(e) + "\n")
        return e

    # -- file tools: one file, before it changes
    def save(self, path: Path, reason: str, command: str = "") -> dict | None:
        p = Path(path)
        if not p.is_file() or p.is_symlink():
            return None
        rp = p.resolve()
        rel = str(rp.relative_to(self.work)) if self.work in rp.parents else f"outside/{rp.name}"
        return self._record(p, rel, reason, command)

    # -- shell commands: stage everything, keep what changed
    def stage(self, rels) -> None:
        self.drop_stage()
        rels = list(rels)
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        total, cloned = 0, True
        for rel in rels:
            src, dst = self.work / rel, self.stage_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if cloned and _clone(src, dst):
                self.staged[rel] = dst
                continue
            cloned = False                                # this disk can't clone: copy, within the cap
            try:
                total += src.stat().st_size
            except OSError:
                continue
            if total > COPY_CAP:
                self.skipped += 1
                self.drop_stage()
                return
            shutil.copy2(src, dst)
            self.staged[rel] = dst

    def keep(self, before: dict[str, str], after: dict[str, str], deleted: list[str], command: str) -> list[dict]:
        """After a command: keep the staged copy of every file whose content changed, and of every approved
        deletion (`deleted`). `before` / `after`: path -> content hash, as the guard computes them."""
        out = []
        for rel, h in sorted(before.items()):
            if rel not in self.staged:
                continue
            if rel in after and after[rel] != h:
                out.append(self._record(self.staged[rel], rel, "changed", command))
            elif rel in deleted:
                out.append(self._record(self.staged[rel], rel, "deleted", command))
        self.drop_stage()
        return out

    def drop_stage(self) -> None:
        self.staged = {}
        shutil.rmtree(self.stage_dir, ignore_errors=True)

    # -- getting a file back
    def restore(self, n: int, to: Path | None = None, overwrite: bool = False) -> Path:
        e = next((x for x in self.entries() if x["n"] == n), None)
        if e is None:
            raise ValueError(f"no backup number {n} in {self.dir}")
        src = self.dir / str(n) / e["path"]
        dst = Path(to) if to else Path(e["folder"]) / e["path"]
        if dst.exists():
            if not overwrite:
                raise ValueError(f"{dst} exists; pass overwrite (its current content is backed up first)")
            if dst.is_file():
                Backups(self.dir, Path(e["folder"])).save(dst, "replaced by restore", f"restore {n}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        _copy(src, dst)
        return dst
