"""OpenAI-compatible local server backend, raw mode.

keel renders the prompt with its own chat template and parses `<tool_call>` itself, so the server is only an
inference engine: the model sees exactly what it sees in-process, and results compare with in-process MLX. The model
input (`prompt_fn(e)`) is the rendered prompt **text**.

Three wire flavours:
    "openai"   POST /v1/completions          mlx_lm.server, LM Studio, vLLM
    "ollama"   POST /api/generate, raw=true  Ollama (native API: has top_k and stop in `options`)
    "llama"    POST /completion              llama-server (llama.cpp)

mlx_lm.server 0.32.2 strips the `<tool_call>` block from raw completion text. With
`token_ids=True` (default for "openai") the request asks for `logprobs`, gets every generated token id back, and
`decode(ids)` rebuilds the exact text. Streamed chunks carry no ids, so token-id mode is never streamed: the reply
reaches `token_fn` in one piece.

Loop breaker ("this reply must be text", P3): one rung per request, the first in `LADDER` that the server is set to
honour (`ban_rungs`; default per mode + flavour, `keel doctor` will measure it) and that can be sent:
    logit_bias   -inf on the `<tool_call>` token (needs `tool_call_id`). Same as in-process MLX's logits mask.
    tool_choice  "none" (chat mode only). mlx_lm.server 0.32.2 ignores it, so it is not in that default.
    stop         a `<tool_call>` stop sequence: halts the reply where the call would start.
    none         nothing sent (not enforced; the loop breaker's note to the model is all there is).
Order from the P3 probe on mlx_lm.server (loop-breaker situation, note included): logit_bias gave a proper final answer
3/3; stop ended 1/3 replies mid-sentence where the model reached for a tool. Whatever the rung, call text or structured
calls that still come back are dropped (`cut`: the server ignored the rung). Each request records its rung and whether
it was cut (`run()` stats: `ban_methods`, `ban_cut`; each episode's `notes`: `ban:<rung>[+cut]@<step>`).

No new dependency: stdlib HTTP, one worker thread per server slot (`RunLimits.batch_size`). Never `validated`.

Chat mode (P2, `mode="chat"`): for models whose chat template keel doesn't render. The model input is
`{"messages", "tools", "funcs"}` (`agent.loop.chat_prompt`); the server renders the template and parses tool calls.
The transcript keeps keel's one format (Qwen `<tool_call>` text, see `backends.chat`): it is converted to
`tool_calls` + `tool_call_id` on the way out and the reply's structured calls are written back as text on the way in.
    "openai"   POST /v1/chat/completions     mlx_lm.server, LM Studio, vLLM, llama-server ("llama" uses this too)
    "ollama"   POST /api/chat                Ollama native (tool calls without ids: results carry `tool_name`)
Thinking: `thinking` -> `chat_template_kwargs.enable_thinking` (openai) or `think` (ollama); None sends neither.
Chat mode is a different backend from raw mode for eval purposes: the server's template and parser are in the loop.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from ..core.backend import Capabilities, Price, RequestStats, RunHooks, RunLimits
from ..specs import SamplingSpec
from .chat import calls_text, parse_args, reply_text, to_chat

TOOL_CALL = "<tool_call>"
END_TEXTS = ("<|im_end|>", "<|endoftext|>")       # stop tokens a decoded id list may end with
FLAVOURS = ("openai", "ollama", "llama")
MODES = ("raw", "chat")
DEFAULT_URLS = {"openai": "http://127.0.0.1:8080", "ollama": "http://127.0.0.1:11434",
                "llama": "http://127.0.0.1:8080"}

# Sampling fields each flavour's raw endpoint takes (from the server docs). Anything else is recorded as dropped.
SENT = {
    "openai": {"temperature", "top_p", "top_k", "presence_penalty", "seed"},
    "ollama": {"temperature", "top_p", "top_k", "presence_penalty", "seed"},
    "llama": {"temperature", "top_p", "top_k", "presence_penalty", "seed"},
}
# Sent, but the server's meaning differs from keel's in-process sampler (mlx_lm.server's presence penalty looks at
# the last 20 tokens of the whole context, prompt included; keel's counts generated tokens only).
APPROXIMATE = {"presence_penalty"}
LADDER = ("logit_bias", "tool_choice", "stop")
# What each wire format can carry (Ollama's native API has no logit_bias; tool_choice exists only for chat requests).
DEFAULT_RUNGS = {("raw", "openai"): ("logit_bias", "stop"), ("raw", "llama"): ("logit_bias", "stop"),
                 ("raw", "ollama"): ("stop",), ("chat", "openai"): ("logit_bias", "tool_choice", "stop"),
                 ("chat", "llama"): ("logit_bias", "tool_choice", "stop"), ("chat", "ollama"): ("stop",)}


class BackendError(RuntimeError):
    """The server could not be reached or answered with an error. The episode stops with `backend_error`."""


@dataclass
class Reply:
    text: str
    finish: str                                  # "stop" | "length"
    prompt_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    ban_method: str | None = None                # rung sent: "logit_bias" | "tool_choice" | "stop" | "none"
    ban_cut: bool = False                        # the server ignored it: call text / calls were dropped by keel
    bad_args: int = 0                            # chat mode: tool calls whose arguments were not a JSON object


@dataclass
class ServerBackend:
    url: str = ""
    flavour: str = "openai"
    sampling: SamplingSpec = field(default_factory=SamplingSpec)
    model: str = ""                              # the server's model name (Ollama needs it; others may ignore it)
    decode: Callable[[list[int]], str] | None = None   # token ids -> text (Qwen tokenizer); needed for token_ids
    tool_call_id: int | None = None              # `<tool_call>` token id, for logit_bias bans
    count_tokens: Callable[[str], int] | None = None   # prompt length before sending (context limit); else after
    token_ids: bool | None = None                # None = on for "openai" (when `decode` is given)
    stream: bool = False                         # stream text to token_fn (never with token_ids)
    seeded: bool = False                         # send a per-episode seed (mlx_lm.server: disables its batching)
    timeout: float = 600.0
    opener: Callable[..., Any] = urllib.request.urlopen
    mode: str = "raw"                            # "raw" (keel renders + parses) | "chat" (the server does)
    thinking: bool | None = False                # chat mode: the template's thinking switch; None = not sent
    ban_rungs: tuple[str, ...] | None = None     # loop-breaker rungs this server honours; None = DEFAULT_RUNGS

    def __post_init__(self):
        if self.flavour not in FLAVOURS:
            raise ValueError(f"unknown server flavour {self.flavour!r} (one of {', '.join(FLAVOURS)})")
        if self.mode not in MODES:
            raise ValueError(f"unknown server mode {self.mode!r} (one of {', '.join(MODES)})")
        self.url = (self.url or DEFAULT_URLS[self.flavour]).rstrip("/")
        if self.mode == "chat":
            if self.token_ids:
                raise ValueError("token_ids is raw mode only (chat mode gets structured tool calls)")
            self.token_ids = False
        if self.token_ids is None:
            self.token_ids = self.flavour == "openai" and self.decode is not None
        allowed = DEFAULT_RUNGS[(self.mode, self.flavour)]
        if self.ban_rungs is None:
            self.ban_rungs = allowed
        bad = [r for r in self.ban_rungs if r not in allowed]
        if bad:
            raise ValueError(f"ban rungs {bad} can't be sent in {self.mode} mode to a {self.flavour} server "
                             f"(possible: {', '.join(allowed)})")
        self.ban_rungs = tuple(self.ban_rungs)
        if self.token_ids and self.decode is None:
            raise ValueError("token_ids needs a decode function (the model's tokenizer)")
        if self.token_ids and self.flavour != "openai":
            raise ValueError("token_ids is only for the openai flavour (logprobs with token ids)")

    # ---- what this backend is ------------------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"server-{self.mode}:{self.flavour}:{self.url}"

    @property
    def chat_style(self) -> str:
        return "ollama" if self.flavour == "ollama" else "openai"

    def capabilities(self) -> Capabilities:
        rung = self.ban_rung()
        return Capabilities(self.name, forbid_tools_next_turn=rung != "none", logit_bias=rung == "logit_bias",
                            seeded=False, batched=True)

    def ban_rung(self) -> str:
        """The loop breaker's rung for this server: the first in LADDER that is set and can be sent."""
        for r in LADDER:
            if r in (self.ban_rungs or ()) and (r != "logit_bias" or self.tool_call_id is not None):
                return r
        return "none"

    def sampling_record(self) -> dict:
        """What the run asked for vs what was sent; goes into the run folder."""
        s = self.sampling
        asked = {"temperature": s.temperature, "top_p": s.top_p, "top_k": s.top_k,
                 "presence_penalty": s.presence_penalty}
        sent = sorted(k for k in asked if k in SENT[self.flavour])
        return {"asked": asked, "sent": sent, "dropped": sorted(set(asked) - set(sent)),
                "approximate": sorted(k for k in sent if k in APPROXIMATE and asked[k]),
                "seed": "per episode" if self.seeded else "none", "token_ids": self.token_ids,
                "stream": self.stream and not self.token_ids, "mode": self.mode, "ban_rung": self.ban_rung(),
                **({"thinking": self.thinking} if self.mode == "chat" else {})}

    def price(self) -> Price:
        return Price()

    # ---- one request -----------------------------------------------------------------------------------------------

    def _seed(self, label: str | None, step: int) -> int | None:
        if not self.seeded or label is None:
            return None
        h = hashlib.sha1(f"{self.sampling.seed}|{label}|{step}".encode()).digest()
        return int.from_bytes(h[:4], "big")

    def _ban(self) -> tuple[str, dict]:
        """(rung, request fields) for a reply that must be text. Ollama's stop goes into `options` (callers move it)."""
        rung = self.ban_rung()
        if rung == "logit_bias":
            if self.flavour == "llama" and self.mode == "raw":
                return rung, {"logit_bias": [[self.tool_call_id, False]]}
            return rung, {"logit_bias": {str(self.tool_call_id): -1e9}}
        if rung == "tool_choice":
            return rung, {"tool_choice": "none"}
        if rung == "stop":
            return rung, {"stop": [TOOL_CALL]}
        return rung, {}

    def _body(self, prompt: str, stream: bool, ban: dict, seed: int | None) -> tuple[str, dict]:
        s = self.sampling
        samp = {"temperature": s.temperature, "top_p": s.top_p, "top_k": s.top_k,
                "presence_penalty": s.presence_penalty}
        samp = {k: v for k, v in samp.items() if k in SENT[self.flavour]}
        if seed is not None:
            samp["seed"] = seed
        if self.flavour == "openai":
            body = {"prompt": prompt, "max_tokens": s.max_tokens, "stream": stream, **samp, **ban}
            if self.model:
                body["model"] = self.model
            if self.token_ids:
                body["logprobs"] = True
            if stream:
                body["stream_options"] = {"include_usage": True}
            return "/v1/completions", body
        if self.flavour == "ollama":
            opts = {**samp, "num_predict": s.max_tokens, **({"stop": ban["stop"]} if "stop" in ban else {})}
            return "/api/generate", {"model": self.model, "prompt": prompt, "raw": True, "stream": stream,
                                     "options": opts}
        return "/completion", {"prompt": prompt, "n_predict": s.max_tokens, "stream": stream, "cache_prompt": True,
                               **samp, **ban}

    def _open(self, path: str, body: dict):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            return self.opener(req, timeout=self.timeout)
        except urllib.error.HTTPError as err:
            detail = err.read().decode(errors="replace")[:300]
            raise BackendError(f"{self.url}{path}: HTTP {err.code}: {detail}") from err
        except (urllib.error.URLError, OSError) as err:
            raise BackendError(f"cannot reach {self.url} ({getattr(err, 'reason', err)}); is the server running?") \
                from err

    def _events(self, resp) -> Iterator[dict]:
        """Streamed chunks as dicts: SSE `data:` lines (openai, llama) or NDJSON lines (ollama)."""
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                return
            yield json.loads(line)

    def request(self, prompt: Any, *, forbid_tools: bool = False, label: str | None = None, step: int = 0,
                on_text: Callable[[str], None] | None = None, cancelled: Callable[[], bool] | None = None) -> Reply:
        """One reply for one model input: rendered text (raw mode) or `{"messages", "tools", "funcs"}` (chat mode).
        `on_text` gets text as it arrives (once, whole, when not streaming)."""
        if self.mode == "chat":
            return self._chat_request(prompt, forbid_tools, self._seed(label, step), on_text, cancelled)
        ban_method, ban = self._ban() if forbid_tools else (None, {})
        stream = self.stream and not self.token_ids
        path, body = self._body(prompt, stream, ban, self._seed(label, step))
        t0 = time.perf_counter()
        with self._open(path, body) as resp:
            r = self._read_stream(resp, on_text, cancelled) if stream else self._read_whole(resp)
        r.seconds = time.perf_counter() - t0
        if forbid_tools:
            r.ban_method = ban_method
            if ban_method == "stop" and self.token_ids and r.finish == "stop" and r.text.endswith(TOOL_CALL) \
                    and r.text.count(TOOL_CALL) == 1:
                # honoured: in token-id mode the returned ids include the stop token itself (mlx_lm.server, P6)
                r.text = r.text[: -len(TOOL_CALL)].rstrip()
            elif TOOL_CALL in r.text:                        # the server ignored the ban: cut the call off
                r.text, r.ban_cut = r.text.split(TOOL_CALL)[0].rstrip(), True
        if on_text is not None and not stream and r.text:
            on_text(r.text)
        return r

    def _read_whole(self, resp) -> Reply:
        try:
            o = json.loads(resp.read())
        except json.JSONDecodeError as err:
            raise BackendError(f"{self.url}: answer is not JSON ({err})") from err
        if isinstance(o, dict) and "error" in o:
            raise BackendError(f"{self.url}: {o['error']}")
        if self.flavour == "openai":
            c = (o.get("choices") or [{}])[0]
            text = c.get("text") or ""
            if self.token_ids:
                ids = [t.get("id") for t in ((c.get("logprobs") or {}).get("content") or [])]
                if not ids or any(i is None for i in ids):
                    raise BackendError(f"{self.url}: no token ids in the answer (logprobs not supported?); "
                                       "run with token_ids off")
                text = self.decode(ids)  # type: ignore[misc]  (token_ids implies decode, checked in __post_init__)
            u = o.get("usage") or {}
            return Reply(_strip_end(text), _finish(c.get("finish_reason")), u.get("prompt_tokens", 0),
                         (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
                         u.get("completion_tokens", 0))
        if self.flavour == "ollama":
            return Reply(_strip_end(o.get("response", "")), _finish(o.get("done_reason")),
                         o.get("prompt_eval_count", 0), 0, o.get("eval_count", 0))
        t = o.get("timings") or {}
        return Reply(_strip_end(o.get("content", "")), _finish(o.get("stop_type")),
                     o.get("tokens_evaluated", 0), t.get("cache_n", 0) or 0, o.get("tokens_predicted", 0))

    def _read_stream(self, resp, on_text, cancelled) -> Reply:
        parts: list[str] = []
        r = Reply("", "stop")
        for ev in self._events(resp):
            if isinstance(ev, dict) and "error" in ev:
                raise BackendError(f"{self.url}: {ev['error']}")
            if self.flavour == "openai":
                for c in ev.get("choices") or []:
                    piece = c.get("text") or ""
                    if c.get("finish_reason"):
                        r.finish = _finish(c["finish_reason"])
                    parts.append(piece)
                    if piece and on_text:
                        on_text(piece)
                if u := ev.get("usage"):
                    r.prompt_tokens, r.output_tokens = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
                    r.cached_tokens = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
            elif self.flavour == "ollama":
                piece = ev.get("response", "")
                parts.append(piece)
                if piece and on_text:
                    on_text(piece)
                if ev.get("done"):
                    r.finish = _finish(ev.get("done_reason"))
                    r.prompt_tokens, r.output_tokens = ev.get("prompt_eval_count", 0), ev.get("eval_count", 0)
            else:
                piece = ev.get("content", "")
                parts.append(piece)
                if piece and on_text:
                    on_text(piece)
                if ev.get("stop"):
                    t = ev.get("timings") or {}
                    r.finish = _finish(ev.get("stop_type"))
                    r.prompt_tokens, r.output_tokens = ev.get("tokens_evaluated", 0), ev.get("tokens_predicted", 0)
                    r.cached_tokens = t.get("cache_n", 0) or 0
            if cancelled is not None and cancelled():
                break                                        # closing the connection stops the server's work
        r.text = _strip_end("".join(parts))
        return r

    # ---- chat mode -------------------------------------------------------------------------------------------------

    def _chat_body(self, prompt: dict, stream: bool, forbid_tools: bool, seed: int | None) -> tuple[str, dict, str | None]:
        s = self.sampling
        samp = {"temperature": s.temperature, "top_p": s.top_p, "top_k": s.top_k,
                "presence_penalty": s.presence_penalty}
        if seed is not None:
            samp["seed"] = seed
        msgs = to_chat(prompt["messages"], prompt.get("funcs") or [], self.chat_style)
        method, ban = self._ban() if forbid_tools else (None, {})
        tools = prompt.get("tools") or []
        if self.chat_style == "ollama":
            opts = {**samp, "num_predict": s.max_tokens, **({"stop": ban["stop"]} if "stop" in ban else {})}
            body = {"model": self.model, "messages": msgs, "tools": tools, "stream": stream, "options": opts}
            if self.thinking is not None:
                body["think"] = self.thinking
            return "/api/chat", body, method
        body = {"messages": msgs, "tools": tools, "max_tokens": s.max_tokens, "stream": stream, **samp}
        if self.model:
            body["model"] = self.model
        if self.thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": self.thinking}
        if stream:
            body["stream_options"] = {"include_usage": True}
        return "/v1/chat/completions", {**body, **ban}, method

    def _chat_request(self, prompt: dict, forbid_tools: bool, seed: int | None, on_text, cancelled) -> Reply:
        if not isinstance(prompt, dict) or "messages" not in prompt:
            raise TypeError("chat mode needs {'messages', 'tools', 'funcs'} as the model input (agent.loop.chat_prompt)")
        stream = self.stream
        path, body, method = self._chat_body(prompt, stream, forbid_tools, seed)
        t0 = time.perf_counter()
        with self._open(path, body) as resp:
            r, content, calls, reasoning = (self._read_chat_stream(resp, on_text, cancelled) if stream
                                            else self._read_chat_whole(resp))
        r.seconds = time.perf_counter() - t0
        if forbid_tools:
            r.ban_method = method
            if calls or TOOL_CALL in content:      # the server ignored the ban: drop the calls
                calls, content, r.ban_cut = [], content.split(TOOL_CALL)[0].rstrip(), True
        r.text = _strip_end(reply_text(content, calls, reasoning))
        if on_text is not None and not stream and r.text:
            on_text(r.text)
        return r

    def _calls(self, raw_calls: list[dict], r: Reply) -> list[tuple[str, dict]]:
        out = []
        for tc in raw_calls:
            fn = tc.get("function") or {}
            args, ok = parse_args(fn.get("arguments"))
            r.bad_args += not ok
            out.append((fn.get("name") or "", args))
        return out

    def _read_chat_whole(self, resp):
        try:
            o = json.loads(resp.read())
        except json.JSONDecodeError as err:
            raise BackendError(f"{self.url}: answer is not JSON ({err})") from err
        if isinstance(o, dict) and "error" in o:
            raise BackendError(f"{self.url}: {o['error']}")
        if self.chat_style == "ollama":
            m = o.get("message") or {}
            r = Reply("", _finish(o.get("done_reason")), o.get("prompt_eval_count", 0), 0, o.get("eval_count", 0))
            return r, m.get("content") or "", self._calls(m.get("tool_calls") or [], r), m.get("thinking") or ""
        c = (o.get("choices") or [{}])[0]
        m = c.get("message") or {}
        u = o.get("usage") or {}
        r = Reply("", _finish(c.get("finish_reason")), u.get("prompt_tokens", 0),
                  (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0, u.get("completion_tokens", 0))
        reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
        return r, m.get("content") or "", self._calls(m.get("tool_calls") or [], r), reasoning

    def _read_chat_stream(self, resp, on_text, cancelled):
        """Content streams to `on_text` as it comes (thinking wrapped in <think>, which the live view hides); the
        calls are collected and written out as text at the end."""
        r = Reply("", "stop")
        content: list[str] = []
        reasoning: list[str] = []
        raw_calls: dict[int, dict] = {}              # index -> {"function": {"name", "arguments"}}
        state = {"think": False}

        def emit(piece: str, kind: str) -> None:
            if not on_text or not piece:
                return
            if kind == "reasoning" and not state["think"]:
                state["think"] = True
                piece = "<think>\n" + piece
            elif kind == "content" and state["think"]:
                state["think"] = False
                piece = "\n</think>\n\n" + piece
            on_text(piece)

        for ev in self._events(resp):
            if isinstance(ev, dict) and "error" in ev:
                raise BackendError(f"{self.url}: {ev['error']}")
            if self.chat_style == "ollama":
                m = ev.get("message") or {}
                for kind, piece in (("reasoning", m.get("thinking") or ""), ("content", m.get("content") or "")):
                    (reasoning if kind == "reasoning" else content).append(piece)
                    emit(piece, kind)
                for tc in m.get("tool_calls") or []:
                    raw_calls[len(raw_calls)] = tc
                if ev.get("done"):
                    r.finish = _finish(ev.get("done_reason"))
                    r.prompt_tokens, r.output_tokens = ev.get("prompt_eval_count", 0), ev.get("eval_count", 0)
            else:
                for c in ev.get("choices") or []:
                    d = c.get("delta") or {}
                    for kind, piece in (("reasoning", d.get("reasoning") or d.get("reasoning_content") or ""),
                                        ("content", d.get("content") or "")):
                        (reasoning if kind == "reasoning" else content).append(piece)
                        emit(piece, kind)
                    for tc in d.get("tool_calls") or []:
                        slot = raw_calls.setdefault(tc.get("index", len(raw_calls)),
                                                    {"function": {"name": "", "arguments": ""}})
                        fn = tc.get("function") or {}
                        if fn.get("name") and not slot["function"]["name"]:
                            slot["function"]["name"] = fn["name"]
                        args = fn.get("arguments")
                        if isinstance(args, dict):
                            slot["function"]["arguments"] = args
                        elif args:
                            slot["function"]["arguments"] += args
                    if c.get("finish_reason"):
                        r.finish = _finish(c["finish_reason"])
                if u := ev.get("usage"):
                    r.prompt_tokens, r.output_tokens = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
                    r.cached_tokens = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
            if cancelled is not None and cancelled():
                break
        calls = self._calls([raw_calls[k] for k in sorted(raw_calls)], r)
        text_content, text_reasoning = "".join(content), "".join(reasoning)
        if calls:                                    # the call markup, as in-process (the live view hides it)
            emit("\n\n" + calls_text(calls), "content")
        return r, text_content, calls, text_reasoning

    # ---- Backend protocol ------------------------------------------------------------------------------------------

    def complete(self, prompt: Any, *, forbid_tools: bool = False) -> tuple[str, str, RequestStats]:
        r = self.request(prompt, forbid_tools=forbid_tools)
        return r.text, r.finish, _stats(r)

    def run(self, episodes, hooks: RunHooks, limits: RunLimits) -> dict:
        """Serve every episode until done, `limits.batch_size` at a time (= the server's slots). With `hooks.handle`
        (P5b) episodes may submit more while the run goes on; a slot is handed back while its episode waits."""
        queue = list(episodes)
        handle = hooks.handle
        n_served = 0
        lock = threading.Lock()
        totals = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "errors": 0,
                  "ban_methods": {}, "ban_cut": 0, "context_limit": 0,
                  "bad_args": 0}
        t0 = time.perf_counter()

        def label(e) -> str | None:
            return hooks.seed_fn(e) if hooks.seed_fn else None

        def serve(e) -> None:
            e.t_start = time.perf_counter()
            step = 0
            try:
                while not e.done:
                    prompt = hooks.prompt_fn(e)
                    if (self.count_tokens is not None and isinstance(prompt, str)
                            and self.count_tokens(prompt) > limits.max_context):
                        e.done, e.stop = True, "context_limit"
                        with lock:
                            totals["context_limit"] += 1
                        break
                    on_text = (lambda t, e=e: _safe_token(hooks.token_fn, e, t)) if hooks.token_fn else None
                    r = self.request(prompt, forbid_tools=getattr(e, "ban_tool_call", False), label=label(e),
                                     step=step, on_text=on_text, cancelled=lambda e=e: getattr(e, "cancelled", False))
                    step += 1
                    if r.ban_method and isinstance(getattr(e, "notes", None), list):
                        cut = "+cut" if r.ban_cut else ""
                        e.notes.append(f"ban:{r.ban_method}{cut}@{getattr(e, 'steps', step - 1) + 1}")
                    e.prompt_tokens += r.prompt_tokens
                    e.processed_tokens += r.prompt_tokens - r.cached_tokens
                    e.output_tokens += r.output_tokens
                    with lock:
                        totals["requests"] += 1
                        totals["prompt_tokens"] += r.prompt_tokens
                        totals["cached_tokens"] += r.cached_tokens
                        totals["output_tokens"] += r.output_tokens
                        totals["bad_args"] += r.bad_args
                        totals["ban_cut"] += r.ban_cut
                        if r.ban_method:
                            totals["ban_methods"][r.ban_method] = totals["ban_methods"].get(r.ban_method, 0) + 1
                    counted = self.count_tokens is not None and isinstance(prompt, str)
                    if not counted and r.prompt_tokens > limits.max_context:
                        e.done, e.stop = True, "context_limit"       # same outcome as in-process: reply not used
                        with lock:
                            totals["context_limit"] += 1
                        break
                    hooks.apply_fn(e, r.text, r.finish)
            except BackendError as err:
                e.done, e.stop = True, "backend_error"
                e.backend_error = str(err)
                with lock:
                    totals["errors"] += 1
            e.seconds = time.perf_counter() - e.t_start
            with lock:                                              # records, events: one finished episode at a time
                hooks.finish_fn(e)
            if handle is not None:
                handle.finished(e)

        workers = max(1, limits.batch_size)
        if handle is None:                                          # the P1 path, unchanged
            n_served = len(queue)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for f in [pool.submit(serve, e) for e in queue]:
                    f.result()
        else:
            n_served = self._run_joinable(queue, serve, handle, workers)
        return {**totals, "episodes": n_served, "seconds": round(time.perf_counter() - t0, 3),
                "backend": self.name, "sampling": self.sampling_record(),
                **({"submitted": handle.submitted} if handle is not None else {})}

    @staticmethod
    def _run_joinable(queue: list, serve, handle, workers: int) -> int:
        """`workers` episodes run at once; an episode blocked in `handle.wait` gives its slot back meanwhile."""
        from concurrent.futures import FIRST_COMPLETED, wait
        running = threading.Semaphore(workers)
        handle.attach(before_wait=running.release, after_wait=running.acquire)
        futures = []

        def task(e) -> None:
            try:
                serve(e)
            finally:
                running.release()

        try:
            with ThreadPoolExecutor(max_workers=workers + 64) as pool:   # threads beyond `workers` = waiting parents
                while True:
                    queue.extend(handle.take())
                    if queue:
                        if running.acquire(timeout=0.1):
                            futures.append(pool.submit(task, queue.pop(0)))
                        continue
                    live = [f for f in futures if not f.done()]
                    if not live and not handle.has_new():
                        break
                    if live:
                        wait(live, timeout=0.1, return_when=FIRST_COMPLETED)
                for f in futures:
                    f.result()
        finally:
            handle.close()
        return len(futures)


def _finish(reason: str | None) -> str:
    """Server finish reasons -> the scheduler's ("stop" / "length")."""
    return "length" if reason in ("length", "limit", "max_tokens") else "stop"


def _strip_end(text: str) -> str:
    for end in END_TEXTS:
        if text.endswith(end):
            return text[: -len(end)]
    return text


def _stats(r: Reply) -> RequestStats:
    return RequestStats(prompt_tokens=r.prompt_tokens, reused_tokens=r.cached_tokens, output_tokens=r.output_tokens,
                        generate_seconds=r.seconds)


def _safe_token(fn, e, text) -> None:
    try:
        fn(e, text)
    except Exception as err:                                        # a display bug must never stop generation
        print(f"  [stream] on_token failed: {err!r}", flush=True)
