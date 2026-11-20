"""Named harness stacks: one definition of what "bare", "keel" and "full" mean, shared by the eval flags and the
interactive server (`keel-core acp --stack`), so what users get is exactly what was measured.

- bare: the plain model with one shell tool (no file tools, no layers). The baseline every gain is measured against.
- keel: file tools + shell, loop breaker, output trimming (the `keel-core run` defaults).
- full: keel + the done check + GNU tool shims + worked examples + run_python (for regular data only), without the
  send-note sentence, + the search hint (a small run: 17/30 -> 22/30). The stack the headline tested against bare.

SHIPPED is what the TUI uses when no stack is given. Decision (user, 2026-11-19): "full" once the headline
confirms H1 (full > bare). Confirmed 2026-11-20 (full 67.7% vs bare 53.1%, +14.6 points, task-clustered one-sided
p = 0.0128, pre-registered ch6-h1), so the TUI now runs full.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stack:
    toolset: str
    loop_break: bool
    trim: bool
    done_gate: bool = False
    gnu_tools: bool = False
    examples: bool = False
    send_note: bool = True
    search_hint: bool = False

    def run_flags(self) -> list[str]:
        """The `keel-core run` / `eval --suite agent` flags that select exactly this stack."""
        f = ["--toolset", self.toolset,
             "--loop-break" if self.loop_break else "--no-loop-break",
             "--trim-outputs" if self.trim else "--no-trim-outputs"]
        f += ["--done-gate"] * self.done_gate + ["--gnu-tools"] * self.gnu_tools + ["--examples"] * self.examples
        f += ["--search-hint"] * self.search_hint
        if not self.send_note:
            f.append("--no-send-note")
        return f

    def system_prompt(self) -> str:
        """The system prompt this stack's episodes start with (the toolset's prompt, then the stack's changes)."""
        from .agent.examples import EXAMPLES
        from .agent.loop import SEND_NOTE, toolset_view
        text = toolset_view(self.toolset)[0]
        if not self.send_note:
            text = text.replace(SEND_NOTE, "")
        if self.examples:
            text += "\n\n" + EXAMPLES
        return text


STACKS = {
    "bare": Stack(toolset="shell", loop_break=False, trim=False),
    "keel": Stack(toolset="files", loop_break=True, trim=True),
    "full": Stack(toolset="files_py", loop_break=True, trim=True, done_gate=True, gnu_tools=True, examples=True,
                  send_note=False, search_hint=True),
}

SHIPPED = "full"
