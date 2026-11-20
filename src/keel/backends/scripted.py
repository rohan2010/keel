"""Scripted backend: replays fixed replies instead of running a model.

It drives the real agent loop (tools, layers, delete guard, events, records) with no MLX and no GPU, so the ACP
server, the Rust TUI and CI can be tested end to end. It is never `validated`: no published number comes from it.

A script says what the "model" answers at each step of an episode, in one of three forms:
    ["reply 1", "reply 2", ...]                 the same list for every episode
    {"B1_0": [...], "s1": [...]}                per episode id (a task item id, or a session id)
    lambda e, k: "..."                          computed from the episode and the step number (0-based)
When a script runs out, the reply is "Done." (plain text), which ends the episode.

Like the MLX backend's logits mask, it honours the loop breaker's ban: when an episode forbids tool calls for its next
reply, any `<tool_call>` part of the scripted reply is dropped. Replies are streamed to `token_fn` in small chunks so
the live view is exercised too.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from ..core.backend import Capabilities, Price, RequestStats, RunHooks, RunLimits

CAPABILITIES = Capabilities("scripted", forbid_tools_next_turn=True)
END = "Done."
MAX_REPLIES = 1000                     # guard against a script that never ends an episode


def episode_id(e: Any) -> str:
    item = getattr(e, "item", None)
    if item is not None and hasattr(item, "id"):
        return item.id
    entry = getattr(e, "entry", None)
    if isinstance(entry, dict) and "id" in entry:
        return entry["id"]
    return getattr(e, "session_id", "") or str(id(e))


class ScriptedBackend:
    def __init__(self, script: list | dict[str, list] | Callable[[Any, int], Any], chunk: int = 4,
                 delay: float = 0.0):
        self.script = script
        self.chunk = max(1, chunk)
        self.delay = delay                       # seconds per chunk (to make a demo or a TUI test watchable)
        self.prompts: list[tuple[str, Any]] = []  # (episode id, model input) for every reply, for tests
        self.used: dict[str, int] = {}            # replies used per episode, kept across run() calls: a follow-up
                                                  # prompt continues the same episode, so its script continues too

    def capabilities(self) -> Capabilities:
        return CAPABILITIES

    def price(self) -> Price:
        return Price()

    def _reply(self, e: Any, k: int) -> str:
        if callable(self.script):
            return self.script(e, k)
        seq = self.script.get(episode_id(e), []) if isinstance(self.script, dict) else self.script
        return seq[k] if k < len(seq) else END

    def _one(self, e: Any, k: int) -> tuple[str, str]:
        """(reply, finish reason). A script entry is a reply, or (reply, "length") for a reply cut at the token
        limit."""
        reply, finish = self._reply(e, k), "stop"
        if isinstance(reply, tuple):
            reply, finish = reply
        if getattr(e, "ban_tool_call", False) and "<tool_call>" in reply:
            reply = reply.split("<tool_call>")[0].strip() or "I will stop here."
        return reply, finish

    def run(self, episodes, hooks: RunHooks, limits: RunLimits) -> dict:
        """Episodes one after another. With `hooks.handle` (P5b), an episode may submit more: they run after the
        current ones, or right away, inline, when an episode waits for them (this backend has no threads)."""
        n_replies = 0
        t0 = time.perf_counter()
        queue = list(episodes)
        handle = hooks.handle
        started: set[int] = set()

        def serve(e) -> None:
            nonlocal n_replies
            started.add(id(e))
            e.t_start = time.perf_counter()
            eid = episode_id(e)
            k = start = self.used.get(eid, 0)
            while not e.done:
                if k - start >= MAX_REPLIES:
                    raise RuntimeError(f"script for {episode_id(e)} never ended the episode")
                prompt = hooks.prompt_fn(e)
                self.prompts.append((episode_id(e), prompt))
                reply, finish = self._one(e, k)
                k += 1
                self.used[eid] = k
                e.prompt_tokens += len(str(prompt)) // 4          # rough, characters / 4 (no tokenizer here)
                e.processed_tokens += len(str(prompt)) // 4
                e.output_tokens += max(1, len(reply) // 4)
                if hooks.token_fn is not None:
                    for i in range(0, len(reply), self.chunk):
                        hooks.token_fn(e, reply[i:i + self.chunk])
                        if self.delay:
                            time.sleep(self.delay)
                hooks.apply_fn(e, reply, finish)
                n_replies += 1
            e.seconds = time.perf_counter() - e.t_start
            hooks.finish_fn(e)
            if handle is not None:
                handle.finished(e)

        def drive(e) -> None:                                  # a waiting episode needs `e` now
            assert handle is not None
            queue.extend(handle.take())
            if id(e) in started:
                raise RuntimeError(f"{episode_id(e)} is waited for while it is still running (a cycle)")
            queue[:] = [q for q in queue if q is not e]        # by identity: episodes are dataclasses (== by value)
            serve(e)

        if handle is not None:
            handle.attach(drive=drive)
        try:
            while queue or (handle is not None and handle.has_new()):
                if handle is not None:
                    queue.extend(handle.take())
                e = queue.pop(0)
                if id(e) not in started:
                    serve(e)
        finally:
            if handle is not None:
                handle.close()
        return {"replies": n_replies, "seconds": round(time.perf_counter() - t0, 3), "backend": "scripted",
                **({"submitted": handle.submitted} if handle is not None else {})}

    def complete(self, prompt: Any, *, forbid_tools: bool = False) -> tuple[str, str, RequestStats]:
        class _One:
            ban_tool_call = forbid_tools
            session_id = "complete"
        reply, finish = self._one(_One(), len(self.prompts))
        self.prompts.append(("complete", prompt))
        return reply, finish, RequestStats(output_tokens=max(1, len(reply) // 4))
