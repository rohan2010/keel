"""Keeping a long interactive session inside the model's context.

Before a new prompt in the TUI, if the conversation plus the prompt would use more than TRIGGER of the context, keel
shortens it, cheapest first, until it fits in TARGET:
1. old tool outputs (all but the last few messages) are cut to their first lines;
2. if that is not enough, the oldest turns are replaced by a short written summary (the requests, the answers, the
   files involved), put in front of the first user message that is kept. No model call: the summary is extracted.
The user is told when this happens. If one turn alone outgrows the context, the session ends that turn with
`context_limit` and the user is told to start a fresh session. Eval runs never use this (one task, no follow-ups).
"""

from __future__ import annotations

import re
from collections.abc import Callable

TRIGGER = 0.70                 # of max_context: leaves room for the reply and this turn's tool outputs
TARGET = 0.50
KEEP_TAIL = 6                  # newest messages never touched by step 1
SHRUNK = "\n... [older tool output shortened by keel to keep the conversation within the model's memory]"
HEAD_CHARS = 300
_PATH = re.compile(r"<parameter=(?:path|paths|source|destination)>\s*(.+?)\s*</parameter>", re.S)


def size(messages: list[dict], count: Callable[[str], int]) -> int:
    return sum(count(str(m.get("content") or "")) for m in messages)


def _summary(dropped: list[dict]) -> str:
    asked, answered, calls, files = [], [], 0, []
    for m in dropped:
        text = str(m.get("content") or "")
        if m.get("role") == "user":
            asked.append(text.strip().splitlines()[0][:200] if text.strip() else "")
        elif m.get("role") == "assistant":
            calls += text.count("<tool_call>") + len(m.get("tool_calls") or [])
            for p in _PATH.findall(text):
                for f in re.split(r'[\[\]",\n]+', p):
                    if f.strip() and f.strip() not in files:
                        files.append(f.strip())
            if "<tool_call>" not in text and not m.get("tool_calls") and text.strip():
                answered.append(text.strip().splitlines()[0][:200])
    lines = ["[keel: the start of this conversation was shortened to fit the model's memory. Summary:]"]
    lines += [f"- the user asked: {a}" for a in asked if a]
    lines += [f"- you answered: {a}" for a in answered[-5:]]
    if calls:
        lines.append(f"- {calls} tool call(s) were made" + (f"; files involved: {', '.join(files[:20])}" if files
                                                            else ""))
    lines.append("Check the folder itself for the current state of the files.")
    return "\n".join(lines)


def compact(messages: list[dict], count: Callable[[str], int], max_context: int, incoming: str = "") -> tuple[
        list[dict], dict]:
    """(messages to use, what was done). `messages[0]` is the system prompt and is always kept; `incoming` is the next
    prompt (counted, not included). Returns the input list unchanged when it fits."""
    trigger, target = int(max_context * TRIGGER), int(max_context * TARGET)
    extra = count(incoming)
    if size(messages, count) + extra <= trigger or len(messages) < 3:
        return messages, {}
    msgs = [dict(m) for m in messages]
    shrunk = 0
    for m in msgs[1:max(1, len(msgs) - KEEP_TAIL)]:
        c = str(m.get("content") or "")
        if m.get("role") == "tool" and len(c) > HEAD_CHARS * 2 and not c.endswith(SHRUNK):
            m["content"] = c[:HEAD_CHARS] + SHRUNK
            shrunk += 1
    done: dict = {"shrunk_outputs": shrunk}
    if size(msgs, count) + extra <= target:
        return msgs, done
    users = [i for i, m in enumerate(msgs) if i > 0 and m.get("role") == "user"]
    cut = len(msgs)                                       # default: drop every old turn
    for i in users:                                       # the earliest user message from which the rest fits
        if size([msgs[0]] + msgs[i:], count) + extra + 300 <= target:
            cut = i
            break
    summary = _summary(msgs[1:cut])
    kept = msgs[cut:]
    if kept:
        kept[0] = {**kept[0], "content": summary + "\n\n" + str(kept[0].get("content") or "")}
        done["summary_prefix"] = ""
    else:
        done["summary_prefix"] = summary + "\n\n"         # goes in front of the incoming prompt
    done["summarised_messages"] = cut - 1
    return [msgs[0]] + kept, done


def notice(done: dict) -> str:
    """What the user is told (empty when nothing was done)."""
    if not done:
        return ""
    parts = []
    if done.get("shrunk_outputs"):
        parts.append(f"shortened {done['shrunk_outputs']} old tool output(s)")
    if done.get("summarised_messages"):
        parts.append(f"summarised the first {done['summarised_messages']} message(s)")
    if not parts:
        return ""
    return (f"(keel: this conversation is getting long for the model's memory, so I {' and '.join(parts)}. "
            "For a completely fresh start, quit and run keel again.)\n\n")


FULL = ("\n\n(keel: this conversation no longer fits the model's memory. Quit and run keel again to start fresh; "
        "your files stay as this session left them.)")
