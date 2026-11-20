"""Harness layers as one ordered pipeline.

Every tool result passes through the result layers in a fixed order before the model sees it:
    trim -> repeat -> loop_break
Trim runs first so that notes added by later layers are never cut. Each layer is either append-only (it adds to
the newest message only) or history-rewriting (it changes text the model already saw). Append-only layers keep the
model's saved state valid; history-rewriting ones force a re-read, so the cache logic needs to know (E1).

Turn-level layers (to-do reminder, verify) stay inside the episode types for now, because what they read
(BFCL's executor state, an agent task's folder) is episode-specific; they are still named and recorded here.

A pipeline belongs to one episode (layers keep per-episode state). With every layer off it returns the raw
output unchanged, which is exactly the research behaviour with the flags off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..agent.verify import REPEAT_NOTE
from ..nudges import LOOP_NOTE, LoopBreaker

# name -> (rewrites history?, what it does). The run config records which are on.
LAYERS = {
    "trim": (False, "shortens a giant tool output before the model sees it"),
    "repeat": (False, "notes the same call + same result 3 times in a row"),
    "loop_break": (False, "3rd identical call+result in a turn: note + next reply must be text"),
    "reminder": (False, "re-shows unfinished earlier requests in the next user message"),
    "verify": (False, "checks the real state when the model says it is done"),
    "budget": (False, "warns 3 steps before the step limit"),
    "probe_gate": (False, "experimental: a linear probe decides call / no call before the reply (MLX only)"),
}


@dataclass
class ResultContext:
    """How one tool result is labelled in notes and keyed for repeat detection (the two episode types differ)."""
    label: str              # "@3" (agent step) or "@t0s1" (BFCL turn / step)
    trim_name: str          # tool name in the trim note
    repeat_sig: Any         # key for "same call + same result"
    loop_key: str           # call key for the loop breaker
    loop_label: str         # call text in the loop note


@dataclass
class ResultOutcome:
    view: str
    ban_tool_call: bool = False
    notes: list[str] = field(default_factory=list)


class TrimLayer:
    name = "trim"

    def __init__(self, trimmer: Callable[[str], tuple[str, dict | None]]):
        self.trimmer = trimmer

    def __call__(self, ctx: ResultContext, raw: str, out: ResultOutcome) -> None:
        view, info = self.trimmer(raw)
        out.view = view
        if info:
            out.notes.append(f"{info.get('kind', 'trim')}{ctx.label}:{ctx.trim_name}:{info['total']}")


class RepeatLayer:
    name = "repeat"

    def __init__(self):
        self.sigs: list = []

    def new_turn(self) -> None:
        self.sigs = []

    def __call__(self, ctx: ResultContext, raw: str, out: ResultOutcome) -> None:
        self.sigs.append(ctx.repeat_sig)
        n = 0
        for x in reversed(self.sigs):
            if x != ctx.repeat_sig:
                break
            n += 1
        if n == 3:
            out.view += REPEAT_NOTE.format(n=n)
            out.notes.append(f"repeat{ctx.label}")


class LoopBreakLayer:
    name = "loop_break"

    def __init__(self, loops: LoopBreaker | None = None):
        self.loops = loops if loops is not None else LoopBreaker()

    def new_turn(self) -> None:
        self.loops.new_turn()

    def __call__(self, ctx: ResultContext, raw: str, out: ResultOutcome) -> None:
        if self.loops.see(ctx.loop_key, raw):
            out.view += LOOP_NOTE
            out.ban_tool_call = True
            out.notes.append(f"loop{ctx.label}:{ctx.loop_label}")


class Pipeline:
    """Ordered result layers for one episode."""

    ORDER = ("trim", "repeat", "loop_break")

    def __init__(self, layers: list):
        names = [l.name for l in layers]
        if names != [n for n in self.ORDER if n in names]:
            raise ValueError(f"layers out of order: {names} (order is {self.ORDER})")
        self.layers = layers

    @property
    def names(self) -> list[str]:
        return [l.name for l in self.layers]

    def get(self, name: str):
        return next((l for l in self.layers if l.name == name), None)

    def new_turn(self) -> None:
        for l in self.layers:
            if hasattr(l, "new_turn"):
                l.new_turn()

    def tool_result(self, ctx: ResultContext, raw: str) -> ResultOutcome:
        out = ResultOutcome(view=raw)
        for l in self.layers:
            l(ctx, raw, out)
        return out


def build_pipeline(*, trimmer=None, repeat: bool = False, loop_break: bool = False,
                   loops: LoopBreaker | None = None) -> Pipeline:
    layers: list = []
    if trimmer is not None:
        layers.append(TrimLayer(trimmer))
    if repeat:
        layers.append(RepeatLayer())
    if loop_break:
        layers.append(LoopBreakLayer(loops))
    return Pipeline(layers)
