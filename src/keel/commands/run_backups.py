"""keel-core backups: list and restore the silent backups keel keeps before a file changes or is deleted.

  uv run keel-core backups                                  sessions with backups (newest first)
  uv run keel-core backups <session>                        every backup in one session
  uv run keel-core backups <session> --restore 3            put backup 3 back where it was
  uv run keel-core backups <session> --restore 3 --to x.txt put it somewhere else
  uv run keel-core backups <path/to/backups dir> ...        an eval run's item: experiments/<run>/backups/<item>

A session's backups live in `<keel state folder>/backups/<session>`. Restoring over an existing file needs
--overwrite, and that file's current content is backed up first. Backups are never deleted automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from keel.agent.backups import Backups
from keel.agent.sandbox import default_state_dir


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("session", nargs="?", help="session id, or a backups folder")
    ap.add_argument("--restore", type=int, metavar="N", help="restore backup number N")
    ap.add_argument("--to", default=None, help="restore to this path instead of the original one")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing file (backed up first)")


def _dir(session: str) -> Path:
    p = Path(session)
    return p if (p / "index.jsonl").exists() else default_state_dir() / "backups" / session


def run(a: argparse.Namespace):
    if not a.session:
        base = default_state_dir() / "backups"
        dirs = sorted((d for d in base.iterdir() if (d / "index.jsonl").exists()), key=lambda d: d.stat().st_mtime,
                      reverse=True) if base.exists() else []
        if not dirs:
            print(f"no backups yet ({base})")
        else:
            print(f"backups in {base} (kept for the newest 50 sessions)")
        for d in dirs:
            es = Backups(d).entries()
            print(f"{d.name}  {len(es)} backup(s), last {es[-1]['time'] if es else '-'}, folder {es[0]['folder'] if es else '-'}")
        return
    d = _dir(a.session)
    if not (d / "index.jsonl").exists():
        raise SystemExit(f"no backups in {d}")
    es = Backups(d).entries()
    if a.restore is None:
        for e in es:
            print(f"{e['n']:>4}  {e['time']}  {e['reason']:<9} {e['path']}  ({e['bytes']:,} bytes)  <- {e['command'][:60]}")
        return
    try:
        out = Backups(d).restore(a.restore, Path(a.to) if a.to else None, a.overwrite)
    except ValueError as err:
        raise SystemExit(str(err)) from None
    print(f"restored backup {a.restore} -> {out}")
