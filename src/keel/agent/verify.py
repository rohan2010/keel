"""The change report (what the state-grounded check shows the model).

Computed by the harness from the real folder, never by the model and never from a scorer or gold answer:
files created / deleted / moved / modified between the start state (the task's template folder) and now, with short
diffs or heads for text files, plus tool calls that returned errors. Generic: nothing in here knows about tasks.
"""

from __future__ import annotations

import difflib
import hashlib
from pathlib import Path

TEXT_LIMIT = 200_000          # bytes; larger files are reported by size only

# --- layers. The self-check and the state check send the same words; the state check adds the change report. -------
CHECK = ("[Automatic check from the system, not from the user] Before you finish, check your work against the "
         "user's request: \"{request}\"\nIf something is wrong or missing, fix it now with the tools. If everything "
         "is correct, reply with your complete final answer for the user (it replaces your previous answer).")
CHECK_RECENT = ("[Automatic check from the system, not from the user] Before you finish, check your work against the "
                "user's recent requests:\nEarlier: \"{earlier}\"\nLatest: \"{latest}\"\nIf something is wrong or "
                "missing, fix it now with the tools. If everything is correct, reply with your complete final answer for "
                "the user (it replaces your previous answer).")
BUDGET_NOTE = ("\n\n[Note from the system: {left} steps left. Finish now, or tell the user clearly what is done and "
               "what is not.]")
REPEAT_NOTE = ("\n\n[Note from the system: this exact call has now returned the same result {n} times in a row. Try "
               "a different approach, or tell the user what is blocking you.]")


def _snapshot(root: Path) -> dict[str, bytes]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            out[p.relative_to(root).as_posix()] = p.read_bytes()
    return out


def _text(b: bytes) -> str | None:
    if len(b) > TEXT_LIMIT or b"\x00" in b[:4096]:
        return None
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _h(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def diff_states(start: Path, end: Path) -> dict:
    """{'created': [...], 'deleted': [...], 'moved': [(old, new)], 'modified': [...]}, plus the two snapshots."""
    a, b = _snapshot(start), _snapshot(end)
    deleted = [p for p in a if p not in b]
    created = [p for p in b if p not in a]
    modified = [p for p in a if p in b and a[p] != b[p]]
    # a deleted file whose exact content reappears under a new name = moved / renamed
    by_hash: dict[str, list[str]] = {}
    for p in created:
        by_hash.setdefault(_h(b[p]), []).append(p)
    moved = []
    for p in list(deleted):
        cands = by_hash.get(_h(a[p]))
        if cands:
            q = cands.pop(0)
            moved.append((p, q))
            deleted.remove(p)
            created.remove(q)
    return {"created": created, "deleted": deleted, "moved": moved, "modified": modified, "_a": a, "_b": b}


def change_report(start: Path, end: Path, errors: list[str] | None = None, answer: str | None = None,
                  max_lines: int = 60, max_list: int = 25) -> str:
    """Plain-text report, capped at about `max_lines` lines of diff / head text in total."""
    d = diff_states(start, end)
    a, b = d["_a"], d["_b"]
    lines: list[str] = []
    n_changes = sum(len(d[k]) for k in ("created", "deleted", "moved", "modified"))

    def listing(title: str, items: list[str]) -> None:
        if not items:
            return
        lines.append(f"{title} ({len(items)}):")
        lines.extend(f"  {x}" for x in items[:max_list])
        if len(items) > max_list:
            lines.append(f"  ... and {len(items) - max_list} more")

    lines.append("CHANGE REPORT (computed from the folder, not from your messages)")
    if n_changes == 0:
        lines.append("No files were created, deleted, moved or changed.")
    listing("Created", d["created"])
    listing("Deleted", d["deleted"])
    listing("Moved or renamed", [f"{x} -> {y}" for x, y in d["moved"]])
    listing("Modified", d["modified"])

    budget = max_lines
    details: list[str] = []
    for p in d["modified"]:
        if budget <= 0:
            break
        ta, tb = _text(a[p]), _text(b[p])
        if ta is None or tb is None:
            details.append(f"--- {p}: binary or large file changed ({len(a[p])} -> {len(b[p])} bytes)")
            budget -= 1
            continue
        diff = [l for l in difflib.unified_diff(ta.splitlines(), tb.splitlines(), lineterm="", n=0)
                if not l.startswith(("---", "+++"))]
        details.append(f"--- changes in {p}:")
        details.extend(diff[:budget])
        if len(diff) > budget:
            details.append(f"  ... {len(diff) - budget} more diff lines")
        budget -= min(len(diff), budget) + 1
    for p in d["created"]:
        if budget <= 0:
            break
        t = _text(b[p])
        if t is None:
            details.append(f"--- new file {p}: binary or large ({len(b[p])} bytes)")
            budget -= 1
            continue
        head = t.splitlines()
        show = head[:min(len(head), max(3, min(15, budget)))]
        details.append(f"--- new file {p} ({len(head)} lines):")
        details.extend(f"  {l}" for l in show)
        if len(head) > len(show):
            details.append(f"  ... {len(head) - len(show)} more lines")
        budget -= len(show) + 1
    if details:
        lines.append("")
        lines.extend(details)

    if errors:
        lines.append("")
        lines.append(f"Tool calls that returned errors ({len(errors)}):")
        root = str(Path(end).resolve())
        lines.extend(f"  {e.replace(root + '/', '').replace(root, '.')[:200]}" for e in errors[-5:])

    if answer is not None and n_changes == 0 and claims_change(answer):
        lines.append("")
        lines.append("NOTE: your answer describes changes, but no file was changed.")
    return "\n".join(lines)


_CLAIM_WORDS = ("i've created", "i have created", "i've moved", "i have moved", "i've renamed", "i have renamed",
                "i've deleted", "i have deleted", "i've updated", "i have updated", "i've added", "i have added",
                "i've sorted", "i have sorted", "i've organized", "i've organised", "i've saved", "i have saved",
                "i've written", "i've replaced", "i've copied", "i have successfully", "i've successfully",
                "has been created", "has been moved", "has been posted", "has been sent", "done!")


def claims_change(answer: str) -> bool:
    a = answer.lower()
    return any(w in a for w in _CLAIM_WORDS)
