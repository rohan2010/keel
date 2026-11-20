"""Transcript <-> chat API messages.

keel's transcript has one format on every backend: an assistant message's `content` is the model's reply text, with
tool calls written in Qwen's `<tool_call><function=..>` markup, followed by one `{"role": "tool"}` message per call.
That is what the in-process chat template renders, and what `_apply_reply` parses.

A chat API (`/v1/chat/completions`, Ollama `/api/chat`) wants structured calls instead. This module converts both
ways, so the transcript, the layers and the records never know which mode ran:

    to_chat(messages, funcs, style)   transcript -> request messages. Calls become `tool_calls` with stable ids
                                      (`call_<assistant index>_<k>`: the same call has the same id in every request),
                                      tool results get the matching `tool_call_id` (Ollama: `tool_name`).
    calls_text(calls)                 structured calls from a reply -> Qwen markup, appended to the reply text.

Round trip: for any transcript, `parse_calls(calls_text(to_chat(...) calls))` gives the same calls as parsing the
original text (tests). Argument values are written the way the model writes them: strings as they are, booleans as
`True` / `False`, everything else as JSON.
"""

from __future__ import annotations

import json

from ..core.parse import parse_calls

TOOL_CALL = "<tool_call>"
STYLES = ("openai", "ollama")


def call_id(turn: int, k: int) -> str:
    return f"call_{turn}_{k}"


def split_reply(content: str, funcs: list[dict]) -> tuple[str, list[dict]]:
    """Assistant text -> (prose before the first call, [{name: args}])."""
    prose = content.split("</think>")[-1].split(TOOL_CALL)[0].rstrip()
    return prose, (parse_calls(content, funcs) if TOOL_CALL in content else [])


def to_chat(messages: list[dict], funcs: list[dict], style: str = "openai") -> list[dict]:
    """The transcript as chat API messages. Private keys (`_step`, `_call`) are dropped."""
    if style not in STYLES:
        raise ValueError(f"unknown chat style {style!r}")
    out: list[dict] = []
    pending: list[tuple[str, str]] = []              # (id, name) of calls still waiting for their tool message
    for i, m in enumerate(messages):
        role, content = m["role"], m.get("content") or ""
        if role == "assistant":
            prose, calls = split_reply(content, funcs)
            msg: dict = {"role": "assistant", "content": prose}
            if calls:
                tcs = []
                pending = []
                for k, c in enumerate(calls):
                    name, args = next(iter(c.items()))
                    args = args if isinstance(args, dict) else {}
                    if style == "openai":
                        tcs.append({"id": call_id(i, k), "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
                    else:
                        tcs.append({"function": {"name": name, "arguments": args}})
                    pending.append((call_id(i, k), name))
                msg["tool_calls"] = tcs
            out.append(msg)
        elif role == "tool":
            if not pending:                          # a result with no call to answer: keep it as a user note
                out.append({"role": "user", "content": content})
                continue
            cid, name = pending.pop(0)
            out.append({"role": "tool", "content": content,
                        **({"tool_call_id": cid} if style == "openai" else {"tool_name": name})})
        else:
            out.append({"role": role, "content": content})
    return out


def _value(v) -> str:
    """Known bug 9 (fixed 2026-11-16): a boolean was written as JSON (`true`), but the model writes `True` (seen in
    every in-process P9 transcript), so chat-mode transcripts weren't the model's text and diffs showed false
    differences. The parser reads both spellings, so execution never changed."""
    if isinstance(v, bool):
        return "True" if v else "False"
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def calls_text(calls: list[tuple[str, dict]]) -> str:
    """(name, args) pairs -> Qwen tool-call markup, as the model writes it in-process."""
    blocks = []
    for name, args in calls:
        params = "".join(f"<parameter={k}>\n{_value(v)}\n</parameter>\n" for k, v in args.items())
        blocks.append(f"{TOOL_CALL}\n<function={name}>\n{params}</function>\n</tool_call>")
    return "\n".join(blocks)


def reply_text(content: str, calls: list[tuple[str, dict]], reasoning: str = "") -> str:
    """A chat reply as transcript text: thinking (if any) in `<think>`, the prose, then the calls."""
    text = content or ""
    if calls:
        text = (text.rstrip() + "\n\n" if text.strip() else "") + calls_text(calls)
    if reasoning:
        text = f"<think>\n{reasoning.strip()}\n</think>\n\n" + text
    return text


def parse_args(raw) -> tuple[dict, bool]:
    """A tool call's arguments as sent by a server: a JSON string (OpenAI) or an object (Ollama). (args, ok)."""
    if isinstance(raw, dict):
        return raw, True
    if raw in (None, ""):
        return {}, True
    try:
        v = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}, False
    return (v, True) if isinstance(v, dict) else ({}, False)
