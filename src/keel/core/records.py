"""Session records: what happened in an agent session, independent of any model or chat template.

Layout (Codex-style): a session holds turns (one per user request); a turn holds items (messages, tool calls, tool
results, notes), each with an id. Two views of the same history are kept apart:
- `content` / `raw`: what really happened (the full tool output, the model's raw reply text);
- `view`: what the model was shown (a trimmed output, a note appended by a layer).
Backends render the model view; scorers and the TUI read the real one.

Records are plain dataclasses with no MLX import, so the front end, scorers and tests can load them anywhere.
`to_chat_messages()` gives the exact message list the research loop built (`role` / `content` dicts), so rendering
through records stays byte-identical to the old code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class StopReason(str, Enum):
    ANSWER = "answer"                 # plain-text reply: the model says it is done
    STEP_LIMIT = "step_limit"
    CONTEXT_LIMIT = "context_limit"
    LENGTH = "length"                 # the reply hit max_tokens
    DONE = "done"                     # BFCL: every user turn was answered
    FORCE_QUIT = "force_quit"         # BFCL: a turn went past the step limit
    CANCELLED = "cancelled"           # the user stopped the session
    ERROR = "error"


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]
    raw: str = ""                     # the call as the model wrote it (backend-specific text), if any


@dataclass
class Item:
    """One entry in a turn. `kind`: system | user | assistant | tool_call | tool_result | note."""
    id: str
    kind: str
    content: str = ""                 # the real text (for tool_result: the full, untrimmed output)
    view: str | None = None           # what the model saw, when it differs from `content`
    call: ToolCall | None = None      # for tool_call items
    call_id: str | None = None        # for tool_result items: the call it answers
    meta: dict[str, Any] = field(default_factory=dict)

    def model_text(self) -> str:
        return self.content if self.view is None else self.view


@dataclass
class Turn:
    index: int
    request: str
    items: list[Item] = field(default_factory=list)


@dataclass
class Session:
    id: str
    turns: list[Turn] = field(default_factory=list)
    system: str = ""
    stop: StopReason | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    _n: int = field(default=0, repr=False, compare=False)

    def new_id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    # --- building ------------------------------------------------------------------------------------------------
    def open_turn(self, request: str) -> Turn:
        t = Turn(len(self.turns), request)
        self.turns.append(t)
        return t

    @property
    def turn(self) -> Turn:
        if not self.turns:
            raise ValueError("no open turn")
        return self.turns[-1]

    def add(self, kind: str, content: str = "", **kw) -> Item:
        it = Item(self.new_id(kind[0]), kind, content, **kw)
        self.turn.items.append(it)
        return it

    def add_call(self, name: str, args: dict, raw: str = "") -> ToolCall:
        c = ToolCall(self.new_id("c"), name, args, raw)
        self.add("tool_call", call=c)
        return c

    def add_result(self, call: ToolCall | None, content: str, view: str | None = None, **meta) -> Item:
        return self.add("tool_result", content, view=None if view == content else view,
                        call_id=call.id if call else None, meta=meta)

    # --- views ---------------------------------------------------------------------------------------------------
    def to_chat_messages(self) -> list[dict]:
        """The model's view as role/content messages (tool calls live inside the assistant text, as Qwen writes
        them; other backends render `ToolCall` items instead)."""
        out = [{"role": "system", "content": self.system}] if self.system else []
        for t in self.turns:
            for it in t.items:
                if it.kind in ("user", "assistant"):
                    out.append({"role": it.kind, "content": it.model_text()})
                elif it.kind == "tool_result":
                    out.append({"role": "tool", "content": it.model_text()})
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_n", None)
        d["stop"] = self.stop.value if self.stop else None
        return d


class SessionLog:
    """Append-only JSONL: one line per finished session. Feeds TUI replay and the scorers."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, s: Session) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False, default=str) + "\n")


def session_from_messages(sid: str, messages: list[dict], **meta) -> Session:
    """Convert a research transcript (role/content messages) into a Session. Each user message opens a turn.
    Lossless for the model view: `to_chat_messages()` returns the same list."""
    s = Session(sid, meta=meta)
    for m in messages:
        role, content = m["role"], m.get("content", "")
        if role == "system":
            s.system = content
        elif role == "user":
            s.open_turn(content)
            s.add("user", content)
        elif role == "assistant":
            if not s.turns:
                s.open_turn("")
            s.add("assistant", content)
        elif role == "tool":
            if not s.turns:
                s.open_turn("")
            s.add("tool_result", content, meta={k: v for k, v in m.items() if k.startswith("_")})
    return s
