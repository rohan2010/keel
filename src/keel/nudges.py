"""Two small notes keel adds to the conversation (their rules were fixed before any measurement).

A. To-do-list reminder (`--reminder`, BFCL multi-turn): every user request is an item; an item stays open while its
   turn (or a later turn that showed it) ends with a question back to the user or a refusal; open items from earlier
   turns are appended to the next turn's user message.
B. Loop breaker (`--loop-break`, BFCL + agent): the same (call, result) pair for the 3rd time in one turn -> a note on
   that tool result and <tool_call> is banned for the next reply, which therefore ends the turn with text.

Both are off by default; with them off, nothing here runs and episodes behave exactly as before.
"""

from __future__ import annotations

import re
from collections import Counter

REMINDER = "\n\nStill unfinished from earlier (do these too if you now can): {items}"
MAX_ITEMS = 3
MAX_SHOWN = 2          # smoke fix 2026-11-09: an item is shown at most twice, then dropped (no pile-up)
LOOP_NOTE = ("\n\n[Note from the system: you have repeated this 3 times with the same result. Stop calling tools and "
             "give your final answer now.]")
LOOP_N = 3

# Reply classification for clearing items (frozen 2026-11-09 before smoke; hand-checked in smoke).
# refusal: the model says IT can't do (part of) the request; first person only (smoke fix 2026-11-09: "your vehicle
# cannot make the trip" is a fact about the world, not a refusal)
CANT = re.compile(r"\b(I|we)\s+(cannot|can't|can not|couldn't|could not|am unable to|was unable to|am not able to|"
                  r"don't have|do not have|don't see any new|do not see any new)\b|"
                  r"\bno (\w+ )?(tool|function)s?\b", re.I)
# question back: a sentence ending in "?" that is not a closing pleasantry
PLEASANTRY = re.compile(r"(anything else|any other|further (help|assistance)|help (you )?with (anything|something) "
                        r"else|else (i|we) can)", re.I)
# smoke fix 2026-11-11 (user decision): a "?" inside quoted text is the user's or a message's words, not a question
# back (long_context_111 quoted the user's message 'What are ... my current order?' -> re-issued -> sent twice).
# Double quotes, curly quotes and backticks only (single quotes are apostrophes); question search only.
QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”|`[^`\n]*`')


def classify_reply(text: str) -> str:
    """'cant' | 'ask' | 'done' for a turn's final text reply."""
    if CANT.search(text or ""):
        return "cant"
    for q in re.findall(r"[^.?!\n]*\?", QUOTED.sub("", text or "")):
        if not PLEASANTRY.search(q):
            return "ask"
    return "done"


class TodoList:
    """The to-do reminder: a keel-kept list of the user's requests."""

    def __init__(self):
        self.items: list[dict] = []          # {"turn", "text", "open", "shown", "expired"}
        self.shown: list[int] = []            # indices listed in this turn's note

    def note_for_turn(self) -> str:
        """Called at a turn start, before the current request is added: the note listing open earlier items."""
        texts = self.open_for_turn()
        if not texts:
            return ""
        return REMINDER.format(items=" ".join(f'({k + 1}) "{t}"' for k, t in enumerate(texts)))

    def open_for_turn(self) -> list[str]:
        """Same selection as the reminder note (expiry after MAX_SHOWN, at most MAX_ITEMS), as a list of request texts;
        marks them shown. Used directly by the re-issue layer."""
        for it in self.items:                 # shown MAX_SHOWN times already: drop it
            if it["open"] and it["shown"] >= MAX_SHOWN:
                it["open"], it["expired"] = False, True
        open_idx = [i for i, it in enumerate(self.items) if it["open"]][-MAX_ITEMS:]
        self.shown = open_idx
        for i in open_idx:
            self.items[i]["shown"] += 1
        return [self.items[i]["text"] for i in open_idx]

    def add(self, turn: int, text: str) -> None:
        self.items.append({"turn": turn, "text": text, "open": True, "shown": 0, "expired": False})

    def close_turn(self, turn: int, final_text: str, made_calls: bool) -> dict:
        """The current turn's own item clears on a 'done' reply; earlier items shown this turn clear only if the turn
        also made tool calls (the model acted on them). Returns a log record."""
        kind = classify_reply(final_text)
        cleared = []
        if kind == "done":
            for i, it in enumerate(self.items):
                if not it["open"]:
                    continue
                if it["turn"] == turn or (i in self.shown and made_calls):
                    it["open"] = False
                    cleared.append(it["turn"])
        self.shown = []
        return {"turn": turn, "reply": kind, "made_calls": made_calls, "cleared": cleared,
                "open": [it["turn"] for it in self.items if it["open"]],
                "expired": [it["turn"] for it in self.items if it["expired"]]}


class LoopBreaker:
    """The loop breaker: counts (call, result) pairs in the current turn."""

    def __init__(self):
        self.counts: Counter = Counter()
        self.fired = False

    def new_turn(self) -> None:
        self.counts.clear()
        self.fired = False

    def see(self, call: str, result: str) -> bool:
        """True when this pair just reached LOOP_N (once per turn)."""
        key = (call, result[:2000])
        self.counts[key] += 1
        if self.counts[key] >= LOOP_N and not self.fired:
            self.fired = True
            return True
        return False


# BanToolCall (the <tool_call> logits mask) moved to keel.backends.mlx: it is MLX-specific.
