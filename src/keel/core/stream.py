"""What a user sees of a reply while it streams.

A reply that calls a tool is prose followed by `<tool_call>...` markup in the model's own format. The front end shows
the prose live and the tool call as a card (from tool events), so the markup must never be printed. `VisibleText`
takes text deltas in the order the model writes them and returns only the part a person should see:

- prose passes through as soon as it can no longer be the start of a tag;
- a `<think>...</think>` block is hidden (thinking can be turned on later);
- everything from the first `<tool_call>` on is hidden;
- leading blank lines are dropped (the full text is `.strip()`ped the same way when the reply is recorded).

Pure Python, no model: the same rules serve any backend.
"""

from __future__ import annotations

TOOL = "<tool_call>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


class VisibleText:
    def __init__(self):
        self.buf = ""            # text not yet decided (may be the start of a tag)
        self.in_think = False
        self.done = False        # a tool call started: nothing more is shown
        self.started = False     # first visible character sent (to drop leading blank lines)

    def feed(self, delta: str) -> str:
        if self.done or not delta:
            return ""
        self.buf += delta
        out = []
        while self.buf and not self.done:
            if self.in_think:
                i = self.buf.find(THINK_CLOSE)
                if i < 0:
                    self.buf = self.buf[-(len(THINK_CLOSE) - 1):]     # keep a possible partial closing tag
                    break
                self.buf, self.in_think = self.buf[i + len(THINK_CLOSE):], False
                continue
            i = self.buf.find("<")
            if i < 0:
                out.append(self.buf)
                self.buf = ""
                break
            out.append(self.buf[:i])
            rest = self.buf[i:]
            if rest.startswith(TOOL):
                self.done, self.buf = True, ""
            elif rest.startswith(THINK_OPEN):
                self.buf, self.in_think = rest[len(THINK_OPEN):], True
            elif TOOL.startswith(rest) or THINK_OPEN.startswith(rest):
                self.buf = rest                                        # could still become a tag: wait
                break
            else:
                out.append("<")                                        # a plain "<" in prose
                self.buf = rest[1:]
        return self._emit("".join(out))

    def flush(self) -> str:
        """End of the reply: a held-back partial tag was just text after all."""
        if self.done or self.in_think:
            self.buf = ""
            return ""
        tail, self.buf = self.buf, ""
        return self._emit(tail)

    def _emit(self, text: str) -> str:
        if not self.started:
            text = text.lstrip()
            self.started = bool(text)
        return text
