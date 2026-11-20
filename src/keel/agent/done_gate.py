"""Done check: concrete-signal checks that hold back "done".

Why this shape: the research's state check printed what had changed when the model said it was done; the
4B saw its mistake in 25 of 32 wrong episodes and acted on it in 0, because it read the report as confirmation. So
The done check has no report. It runs only checks that fire on a concrete, generic signal, and when one fires it holds back the
answer with the exact problem and the tools to fix it (outcome monitors helped only when they named the recovery
tools, arXiv 2608.19303). At most 2 rounds per episode; after that the answer stands.

Every check is generic: it reads the files the model created or changed and the harness's own records, never a task
id, a scorer or an answer key (a test enforces that this module imports nothing task-specific).

Checks:
- csv:   a row whose field count differs from the header's (an unquoted comma is the usual cause)
- json:  the file doesn't load
- yaml:  the file doesn't load
- html:  a relative link or image that points at no file
- error: the last tool call failed and nothing succeeded after it
- edit:  an edit_file replacement also changed text inside a longer name (the tool warned; the model went on)
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from pathlib import Path

MAX_FILES = 200                  # files compared per episode (task folders are small; real folders are capped by
MAX_BYTES = 2_000_000            # the sandbox's own size limit)
PER_FILE = 2                     # problems reported per file
HREF = re.compile(r"""(?:href|src)\s*=\s*["']([^"'#?]+)""", re.I)
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv"}


def snapshot(work: Path) -> dict[str, str]:
    """{relative path: sha1} of the folder's small files: the state before the model's first tool call."""
    out: dict[str, str] = {}
    for d, dirs, files in os.walk(work):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        for f in files:
            p = Path(d) / f
            try:
                if p.stat().st_size <= MAX_BYTES:
                    out[str(p.relative_to(work))] = hashlib.sha1(p.read_bytes()).hexdigest()
            except OSError:
                continue
            if len(out) >= 5000:
                return out
    return out


def changed_files(work: Path, before: dict[str, str]) -> list[str]:
    """Files created or modified since the snapshot (sorted, capped)."""
    now = snapshot(work)
    return sorted(p for p, h in now.items() if before.get(p) != h)[:MAX_FILES]


def _csv(text: str, rel: str) -> list[str]:
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error as err:
        return [f"{rel} is not valid CSV ({err})"]
    rows = [r for r in rows if any(c.strip() for c in r)]
    if len(rows) < 2:
        return []
    width, out = len(rows[0]), []
    for i, r in enumerate(rows[1:], start=2):
        if len(r) != width:
            hint = " (a value that contains a comma must be in double quotes)" if len(r) > width else ""
            out.append(f"row {i} of {rel} has {len(r)} fields, the header has {width}{hint}")
            if len(out) >= PER_FILE:
                break
    return out


def _json(text: str, rel: str) -> list[str]:
    try:
        json.loads(text)
    except json.JSONDecodeError as err:
        return [f"{rel} is not valid JSON (line {err.lineno}, column {err.colno}: {err.msg})"]
    return []


def _yaml(text: str, rel: str) -> list[str]:
    import yaml
    try:
        yaml.safe_load(text)
    except yaml.YAMLError as err:
        mark = getattr(err, "problem_mark", None)
        where = f" (line {mark.line + 1})" if mark is not None else ""
        return [f"{rel} is not valid YAML{where}"]
    return []


def _html(text: str, rel: str, work: Path) -> list[str]:
    out = []
    base = (work / rel).parent
    for target in HREF.findall(text):
        t = target.strip()
        if not t or t.startswith(("http:", "https:", "mailto:", "tel:", "data:", "javascript:", "//", "/")):
            continue
        if not (base / t).resolve().exists():
            out.append(f"{rel} links to {t}, which doesn't exist")
            if len(out) >= PER_FILE:
                break
    return out


def file_problems(work: Path, changed: list[str]) -> list[str]:
    out: list[str] = []
    for rel in changed:
        p = work / rel
        ext = p.suffix.lower()
        if ext not in (".csv", ".json", ".yaml", ".yml", ".html", ".htm"):
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        if ext == ".csv":
            out += _csv(text, rel)
        elif ext == ".json":
            out += _json(text, rel)
        elif ext in (".yaml", ".yml"):
            out += _yaml(text, rel)
        else:
            out += _html(text, rel, work)
    return out


def message(problems: list[str]) -> str:
    """The one message the model gets: the exact problems and the tools to fix them. No general report."""
    lines = "\n".join(f"- {p}" for p in problems[:8])
    return ("[keel check] Not finished yet. Before you answer, fix these problems:\n" + lines +
            "\nUse read_file to look, then edit_file or write_file to fix (run_shell for anything else). Then give "
            "your final answer. If something can't be fixed, say so plainly in your answer.")
