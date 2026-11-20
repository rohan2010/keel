"""Dedicated file tools for the agent, next to run_shell.

Design goals (from the baseline failures): show modified dates unambiguously (A3 read access times from `stat`),
make edits exact and show what changed (D1: blind `sed` also renamed a look-alike class), keep moves and deletions
explicit (E3 lost photos), and never leave the folder: every path is resolved and must stay inside work/ (symlinks
included). Deletions go through the delete guard's confirmation like shell deletions do.
"""

from __future__ import annotations

import difflib
import os
import re
import shutil
import time
from pathlib import Path

from .guard import Confirmation, GuardedShell

MAX_LIST, MAX_LINES, MAX_HITS, MAX_OUT = 300, 200, 100, 6000


def _schema(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


FILE_TOOLS = [
    _schema("list_files", "List files and folders with size and last-modified date and time (YYYY-MM-DD HH:MM).",
            {"path": {"type": "string", "description": "Folder to list, relative to the current folder. Default '.'"},
             "recursive": {"type": "boolean", "description": "Include subfolders. Default true."}}, []),
    _schema("read_file", "Read a text file. Lines are numbered.",
            {"path": {"type": "string"}, "start_line": {"type": "integer", "description": "First line, default 1."},
             "max_lines": {"type": "integer", "description": f"Default {MAX_LINES}."}}, ["path"]),
    _schema("write_file", "Create a file or replace its whole content. Creates missing folders.",
            {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _schema("edit_file", "Replace exact text in a file. Fails if the text is not found, or is found more than once and "
            "replace_all is false. Shows the changed lines.",
            {"path": {"type": "string"}, "old_text": {"type": "string", "description": "Exact text to find."},
             "new_text": {"type": "string"},
             "replace_all": {"type": "boolean", "description": "Replace every occurrence. Default false."}},
            ["path", "old_text", "new_text"]),
    _schema("search", "Find lines containing a text (or a regular expression) in files, ignoring case by default. "
            "Shows file:line: text.",
            {"pattern": {"type": "string"}, "path": {"type": "string", "description": "Folder or file. Default '.'"},
             "regex": {"type": "boolean", "description": "Treat pattern as a Python regular expression."},
             "ignore_case": {"type": "boolean", "description": "Default true."}}, ["pattern"]),
    _schema("move", "Move or rename a file or folder. Creates missing folders. Never overwrites an existing file.",
            {"source": {"type": "string"}, "destination": {"type": "string",
             "description": "New path, or an existing folder to move into."}}, ["source", "destination"]),
    _schema("delete", "Delete files or folders. The user is asked to confirm first.",
            {"paths": {"type": "array", "items": {"type": "string"}}}, ["paths"]),
]
# parameter types for the call parser (it coerces XML parameter strings with these)
FILE_FUNCS = [{"name": t["function"]["name"], "parameters": t["function"]["parameters"]} for t in FILE_TOOLS]


class ToolError(Exception):
    pass


# search hint: the model writes regular expressions (`2\.8\.9`)
# without regex=true; in literal mode they match nothing and "(no matches)" sent it away (119 of 140 such searches in
# all saved runs). With the hint on, a literal search that finds nothing and looks like a regex says so; it never
# runs the regex itself (no hits the model didn't ask for).
REGEX_LIKE = re.compile(r"\\[.dDwWsSbB$^()\[\]{}|*+?]|\.[*+]|\[[^\]]+\]|\(\?|\|")


def looks_like_regex(pattern: str) -> bool:
    if not REGEX_LIKE.search(pattern):
        return False
    try:
        re.compile(pattern)
    except re.error:
        return False
    return True


class FileTools:
    def __init__(self, shell: GuardedShell, search_hint: bool = False):
        self.shell = shell
        self.search_hint = search_hint
        self.work = shell.sb.work.resolve()
        # edit-failure metric: calls, refusals, and edits that only matched ignoring whitespace
        self.stats = {"edit_calls": 0, "edit_failed": 0, "edit_fuzzy": 0}
        self.overmatches: list[tuple] = []        # (file, old_text, longer names hit): read by the done check, not by the model

    # -- path safety
    def _p(self, rel: str | None, must_exist: bool = True) -> Path:
        rel = (rel or ".").strip()
        p = Path(os.path.realpath(self.work / rel))
        if p != self.work and not str(p).startswith(str(self.work) + os.sep):
            raise ToolError(f"'{rel}' is outside the folder you can use.")
        if must_exist and not p.exists():
            raise ToolError(f"'{rel}' does not exist.")
        return p

    def _rel(self, p: Path) -> str:
        return "." if p == self.work else str(p.relative_to(self.work))

    @staticmethod
    def _when(p: Path) -> str:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))

    # -- tools
    def list_files(self, path=".", recursive=True):
        root = self._p(path)
        if root.is_file():
            return f"{self._rel(root)}  {root.stat().st_size} bytes  modified {self._when(root)}"
        it = sorted(root.rglob("*") if recursive in (True, "true", None) else root.iterdir())
        rows = []
        for p in it:
            if any(part.startswith(".") for part in p.relative_to(root).parts):
                continue
            rows.append(f"{self._rel(p)}/" if p.is_dir() else
                        f"{self._rel(p)}  {p.stat().st_size} bytes  modified {self._when(p)}")
        more = f"\n... {len(rows) - MAX_LIST} more" if len(rows) > MAX_LIST else ""
        return ("\n".join(rows[:MAX_LIST]) + more) if rows else "(empty folder)"

    def read_file(self, path, start_line=1, max_lines=MAX_LINES):
        p = self._p(path)
        if p.is_dir():
            raise ToolError(f"'{path}' is a folder; use list_files.")
        data = p.read_bytes()
        if b"\x00" in data[:4096]:
            return f"(binary file, {len(data)} bytes)"
        lines = data.decode("utf-8", errors="replace").splitlines()
        s = max(int(start_line or 1), 1)
        n = max(int(max_lines or MAX_LINES), 1)
        chunk = lines[s - 1: s - 1 + n]
        tail = f"\n... ({len(lines)} lines in total)" if s - 1 + n < len(lines) else ""
        return ("\n".join(f"{s + i:>5}  {l}" for i, l in enumerate(chunk)) + tail) if chunk else "(no lines there)"

    def write_file(self, path, content):
        p = self._p(path, must_exist=False)
        existed = p.exists()
        if existed:
            self.shell.backups.save(p, "replaced", f"write_file {self._rel(p)}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content if isinstance(content, str) else str(content))
        return f"{'Replaced' if existed else 'Created'} {self._rel(p)} ({p.stat().st_size} bytes)."

    def edit_file(self, path, old_text, new_text, replace_all=False):
        self.stats["edit_calls"] += 1
        try:
            return self._edit_file(path, old_text, new_text, replace_all)
        except ToolError:
            self.stats["edit_failed"] += 1
            raise

    def _edit_file(self, path, old_text, new_text, replace_all=False):
        p = self._p(path)
        text = p.read_text(errors="replace")
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise ToolError("old_text and new_text must be text.")
        if old_text == new_text:
            raise ToolError("old_text and new_text are the same, so nothing would change. Put the new version in "
                            "new_text.")
        n = text.count(old_text) if old_text else 0
        if n == 0 and old_text.strip():
            hit = _loose_match(text, old_text)
            if hit is not None:                          # unique match ignoring whitespace: apply it, and say so
                a, b, indent = hit
                lines = text.splitlines(keepends=True)
                repl = _reindent(new_text, indent)
                if repl and lines[b - 1].endswith("\n"):     # keep the line break the replaced block ended with
                    repl += "\n"
                new = "".join(lines[:a]) + repl + "".join(lines[b:])
                self.shell.backups.save(p, "edited", f"edit_file {self._rel(p)}")
                p.write_text(new)
                self.stats["edit_fuzzy"] += 1
                return (f"Replaced lines {a + 1}-{b} in {self._rel(p)} (matched ignoring differences in spaces and "
                        f"indentation).\n" + _changed_lines(text, new))
        if n == 0:
            raise ToolError(f"the text was not found in {self._rel(p)}. " + _closest(text, old_text))
        if n > 1 and replace_all not in (True, "true"):
            raise ToolError(f"the text occurs {n} times in {self._rel(p)}; add surrounding text to make it unique, "
                            "or set replace_all to true.")
        new = text.replace(old_text, new_text) if replace_all in (True, "true") else text.replace(old_text, new_text, 1)
        self.shell.backups.save(p, "edited", f"edit_file {self._rel(p)}")
        p.write_text(new)
        # warn when a replaced occurrence was part of a longer name (e.g. 'btn-primary' inside 'btn-primary-outline')
        inside = []
        for m in re.finditer(re.escape(old_text), text):
            a, b = m.start(), m.end()
            left = text[a - 1] if a > 0 else ""
            right = text[b] if b < len(text) else ""
            if (old_text[:1].isalnum() and (left.isalnum() or left in "-_")) or \
               (old_text[-1:].isalnum() and (right.isalnum() or right in "-_")):
                w0 = re.search(r"[\w-]*$", text[:a]).group(0)
                w1 = re.match(r"[\w-]*", text[b:]).group(0)
                inside.append(w0 + old_text + w1)
            if replace_all not in (True, "true"):
                break
        warn = (f"\nWarning: {len(inside)} replacement(s) were inside a longer name: "
                f"{', '.join(sorted(set(inside))[:5])}. Check that this is what you wanted.") if inside else ""
        if inside:                                       # recorded for the done check; the tool is unchanged
            self.overmatches.append((self._rel(p), old_text, sorted(set(inside))[:5], replace_all in (True, "true")))
        return (f"Replaced {n if replace_all in (True, 'true') else 1} occurrence(s) in {self._rel(p)}.{warn}\n"
                + _changed_lines(text, new))

    def search(self, pattern, path=".", regex=False, ignore_case=True):
        root = self._p(path)
        try:
            rx = re.compile(pattern if regex in (True, "true") else re.escape(pattern),
                            0 if ignore_case in (False, "false") else re.IGNORECASE)
        except re.error as err:
            raise ToolError(f"bad regular expression: {err}") from err
        files = [root] if root.is_file() else sorted(q for q in root.rglob("*") if q.is_file()
                                                    and not any(x.startswith(".") for x in q.relative_to(root).parts))
        hits = []
        for f in files:
            data = f.read_bytes()
            if b"\x00" in data[:4096]:
                continue
            for i, l in enumerate(data.decode("utf-8", errors="replace").splitlines()):
                if rx.search(l):
                    hits.append(f"{self._rel(f)}:{i + 1}: {l.strip()[:200]}")
        more = f"\n... {len(hits) - MAX_HITS} more" if len(hits) > MAX_HITS else ""
        if hits:
            return "\n".join(hits[:MAX_HITS]) + more
        if self.search_hint and regex not in (True, "true") and looks_like_regex(pattern):
            return (f"(no matches for the exact text '{pattern}'. It looks like a regular expression: to search with "
                    "it, call search again with regex=true.)")
        return "(no matches)"

    def move(self, source, destination):
        src = self._p(source)
        dst = self._p(destination, must_exist=False)
        if dst.is_dir():
            dst = dst / src.name
        if dst.exists():
            raise ToolError(f"'{self._rel(dst)}' already exists; moving would overwrite it. Choose another name.")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"Moved {self._rel(src)} -> {self._rel(dst)}."

    def delete(self, paths):
        if isinstance(paths, str):
            paths = [paths]
        targets = [self._p(x) for x in paths or []]
        if not targets:
            raise ToolError("no paths given.")
        if any(t == self.work for t in targets):
            raise ToolError("you cannot delete the whole folder you are working in.")
        files = sorted({self._rel(f) for t in targets for f in ([t] if t.is_file() else t.rglob("*")) if f.is_file()})
        ok = self.shell.confirm(f"delete {', '.join(self._rel(t) for t in targets)}", files)
        self.shell.log.append(Confirmation(f"delete({paths})", files, ok, "asked"))
        if not ok:
            return f"The user declined: nothing was deleted ({len(files)} file(s) would have been)."
        for t in targets:
            for f in ([t] if t.is_file() else sorted(t.rglob("*"))):
                if f.is_file():
                    self.shell.backups.save(f, "deleted", f"delete {self._rel(f)}")
            shutil.rmtree(t) if t.is_dir() else t.unlink()
        return f"Deleted {len(files)} file(s)" + (f" in {len(targets)} path(s)." if len(targets) > 1 else ".")

    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, name, None)
        if fn is None or name.startswith("_") or name == "call":
            raise ToolError(f"unknown tool {name}.")
        try:
            out = fn(**(args or {}))
        except TypeError as err:
            raise ToolError(f"wrong arguments for {name}: {err}") from err
        return out if len(out) <= MAX_OUT else out[:MAX_OUT] + f"\n... [{len(out) - MAX_OUT} characters cut]"


# --- edit_file helpers (B4) ---------------------------------------------------------------------------------------

def _changed_lines(old: str, new: str) -> str:
    changed = [f"{i + 1:>5}  {l}" for i, (a, l) in enumerate(zip(old.splitlines(), new.splitlines(), strict=False)) if a != l]
    a, b = old.splitlines(), new.splitlines()
    if len(a) != len(b):                                 # lines added or removed: show the new block's lines
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        changed = [f"{j + 1:>5}  {b[j]}" for tag, _, _, j1, j2 in sm.get_opcodes() if tag != "equal"
                   for j in range(j1, j2)]
    shown = "\n".join(changed[:20]) + (f"\n... {len(changed) - 20} more changed lines" if len(changed) > 20 else "")
    return f"Changed lines now:\n{shown}" if changed else "Changed lines now: (only line removals)"


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _loose_match(text: str, old: str) -> tuple[int, int, tuple[str, str]] | None:
    """The one line range of `text` whose lines equal `old`'s lines once surrounding whitespace is ignored (inner
    spaces collapsed). Returns (start, end, (file indent, old indent)) or None when there is no match or more than one.
    """
    norm = lambda l: " ".join(l.split())
    want = [norm(l) for l in old.strip("\n").splitlines()]
    while want and not want[-1]:
        want.pop()
    if not want:
        return None
    lines = text.splitlines(keepends=True)
    have = [norm(l) for l in lines]
    k = len(want)
    hits = [i for i in range(len(have) - k + 1) if have[i:i + k] == want]
    if len(hits) != 1:
        return None
    i = hits[0]
    first_old = next(l for l in old.strip("\n").splitlines() if l.strip())
    first_file = next(l for l in lines[i:i + k] if l.strip())
    return i, i + k, (_indent(first_file), _indent(first_old))


def _reindent(new: str, indent: tuple[str, str]) -> str:
    """Shift `new` from the model's indentation to the file's (the difference seen on the first matched line)."""
    have, wrote = indent
    out = []
    for l in new.strip("\n").splitlines():
        if l.strip() and l.startswith(wrote):
            l = have + l[len(wrote):]
        elif l.strip() and not wrote:
            l = have + l
        out.append(l)
    return "\n".join(out)


def _closest(text: str, old: str) -> str:
    """Point the model at the lines that look most like what it tried to replace, with line numbers."""
    lines = text.splitlines()
    want = old.strip("\n").splitlines() or [old]
    k = max(1, min(len(want), len(lines)))
    if not lines:
        return "The file is empty."
    if len(lines) * k > 200_000:                         # keep the hint cheap on very large files
        return "Read the file and copy the exact text."
    target = "\n".join(l.strip() for l in want)
    best, best_i = 0.0, 0
    for i in range(len(lines) - k + 1):
        r = difflib.SequenceMatcher(None, target, "\n".join(l.strip() for l in lines[i:i + k]), autojunk=False).ratio()
        if r > best:
            best, best_i = r, i
    if best < 0.5:
        return "Read the file and copy the exact text."
    shown = "\n".join(f"{best_i + j + 1:>5}  {lines[best_i + j]}" for j in range(k))
    return f"The closest text is at lines {best_i + 1}-{best_i + k}:\n{shown}\nCopy the exact text from there."
