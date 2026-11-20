"""Events the core emits while a session runs. The protocol server only translates these into ACP messages,
so the front end never needs to know how the core works inside.

Kinds:
  text            the model's reply text (whole reply for now; token deltas when a backend streams)
  tool_started    a tool call is about to run            {call_id, name, args}
  tool_finished   it returned                            {call_id, name, ok, output (what the model sees), raw_len}
  file_diff       a file tool changed a file             {path, diff}
  permission      the delete guard asked the approver    {command, files, allowed}
  layer           a layer fired                          {layer, note}
  stop            the session ended                      {reason, answer}
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

KINDS = ("text", "text_delta", "tool_started", "tool_finished", "file_diff", "permission", "layer", "stop")


@dataclass
class Event:
    kind: str
    session: str
    data: dict[str, Any] = field(default_factory=dict)
    t: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown event kind {self.kind!r}")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "session": self.session, "t": round(self.t, 3), **self.data}


Sink = Callable[[Event], None]


def null_sink(_: Event) -> None:
    pass


class ListSink:
    """Collects events (tests)."""

    def __init__(self):
        self.events: list[Event] = []
        self._lock = threading.Lock()

    def __call__(self, ev: Event) -> None:
        with self._lock:
            self.events.append(ev)

    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]


class JsonlSink:
    """One JSON line per event (thread-safe; tools run in worker threads)."""

    def __init__(self, path: Path | None = None, stream=None):
        self._f = Path(path).open("a") if path else None
        self._stream = stream
        self._lock = threading.Lock()

    def __call__(self, ev: Event) -> None:
        line = json.dumps(ev.to_dict(), ensure_ascii=False, default=str)
        with self._lock:
            if self._f:
                self._f.write(line + "\n")
                self._f.flush()
            if self._stream:
                print(line, file=self._stream, flush=True)

    def close(self) -> None:
        if self._f:
            self._f.close()


def stdout_sink() -> JsonlSink:
    return JsonlSink(stream=sys.stdout)
