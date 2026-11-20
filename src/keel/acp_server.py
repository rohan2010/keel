"""keel-core as an ACP agent over stdio. The Rust TUI (and any ACP client, e.g. Zed) drives it.

    keel-core acp                                   the local model (MLX), loaded on the first session
    keel-core acp --backend scripted --script s.json   replayed replies, no model (tests, demos)
    keel-core acp --backend mlx-server              a running mlx_lm.server (or: server / ollama, see backends.choose)

If the model server can't be reached or answers with an error, `session/prompt` fails with a JSON-RPC error
(-32603, data.reason "backend_error") carrying the server's message, so the client shows it; a prompt that got no
reply at all is taken back out of the conversation, so trying again doesn't send it twice.

Protocol: Agent Client Protocol v1, JSON-RPC over stdin/stdout. Stdout carries protocol messages only: at start-up
the real stdout is kept for the protocol and file descriptor 1 is pointed at stderr, so a stray print (the scheduler's
progress lines, a library warning) can never corrupt the stream.

A session = one agent episode working IN PLACE in the folder given to `session/new` (`Sandbox.for_folder`: size
limit, clock off, delete guard). Each `session/prompt` adds a user message and runs the same agent loop as the evals
(tools, layers, guard) until the model answers in plain text, up to MAX_STEPS more steps. Core events become
`session/update`s:
    text_delta     -> agent_message_chunk (prose only; tool-call markup is never streamed)
    tool_started   -> tool_call (kind read / edit / move / delete / search / execute, status in_progress)
    tool_finished  -> tool_call_update (completed / failed, the output, and a `diff` for file writes)
Delete guard over ACP (P5). Every delete the guard catches (an `rm` after its dry run, the delete tool) becomes a
`session/request_permission` for the delete's tool call, sent only after that tool call reached the client. The only
options are `allow_once` and `reject_once` (never "always"). The files are deleted only when the reply is `selected`
with exactly that allow-once id; anything else is a refusal: reject, cancelled, an unknown option, a malformed reply,
an error, the client disconnecting, or `session/cancel` while the question is open. The model is told the user
declined. There is no setting that changes any of this.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import stat
import sys
import threading
import uuid
from pathlib import Path

import acp
from acp import schema as s
from acp.exceptions import RequestError

from .agent.compact import FULL, compact, notice
from .agent.filetools import FileTools
from .agent.guard import GuardedShell
from .agent.loop import Episode, SessionItem, base_toolset, configure, episode_hooks
from .agent.sandbox import FolderRefused, Sandbox, default_state_dir, sweep_scratch
from .core.backend import RunLimits
from .core.events import Event
from .core.tools import reporting
from .stacks import SHIPPED, STACKS

VERSION = "0.1.0"
MAX_STEPS = 20                                  # per prompt
KEEP_SESSIONS = 50                              # transcripts kept in <state>/sessions (user decision 2026-11-14)
MAX_DIFF_BYTES = 512 * 1024
STOP = {"answer": "end_turn", "length": "max_tokens", "step_limit": "max_turn_requests",
        "context_limit": "max_tokens", "cancelled": "cancelled"}
KIND = {"list_files": "read", "read_file": "read", "search": "search", "write_file": "edit", "edit_file": "edit",
        "move": "move", "delete": "delete", "run_shell": "execute"}
WRITES = {"write_file", "edit_file"}


def log(*a) -> None:
    print("[keel-core]", *a, file=sys.stderr, flush=True)


# --- backends ---------------------------------------------------------------------------------------------------

class Engine:
    """A backend plus what the agent loop needs from it: how to build the model input and how to trim outputs."""

    def __init__(self, backend, prompt_fn, trimmer, limits: RunLimits, name: str, caps: dict | None = None,
                 count=None):
        self.backend, self.prompt_fn, self.trimmer, self.limits, self.name = backend, prompt_fn, trimmer, limits, name
        self.caps = caps or {"backend": name}             # what this backend can do: session meta + capabilities.json
        self.count = count or (lambda text: len(text) // 2)   # tokens, for long sessions (over-counts without a tokenizer)


def scripted_engine(script_path: str | None, delay: float = 0.0) -> Engine:
    from .backends.scripted import ScriptedBackend
    from .trim import trim
    script = json.loads(Path(script_path).read_text()) if script_path else ["Hello from the scripted backend."]
    return Engine(ScriptedBackend(script, delay=delay), lambda e: [dict(m) for m in e.messages],
                  lambda text: trim(text, lambda x: len(x) // 4), RunLimits(batch_size=1), "scripted")


def mlx_engine() -> Engine:
    """Loads the model (seconds). Interactive sampling: Qwen's non-thinking settings, per-episode random stream."""
    from .agent.loop import prompt_ids
    from .backends.mlx import MLXBackend
    from .model import load_model
    from .specs import QWEN35_4B_8BIT, ModelSpec, SamplingSpec
    from .trim import make_trimmer
    b = QWEN35_4B_8BIT
    spec = ModelSpec(b.name, b.hf_repo, b.revision, b.local_dir, {"enable_thinking": False})
    sampling = SamplingSpec(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=0.0, max_tokens=4096, seed=0)
    model, tok = load_model(spec)
    from .backends.choose import mlx_capabilities_record
    return Engine(MLXBackend(model, tok, sampling), lambda e: prompt_ids(tok, spec, e.messages, e.toolset),
                  make_trimmer(tok), RunLimits(batch_size=1, max_context=16000, context_budget=48000,
                                               progress_every=0, extra={"prefill_batch": 1}), "mlx",
                  mlx_capabilities_record(True), count=lambda text: len(tok.encode(text, add_special_tokens=False)))


def _check_reachable(url: str, backend: str) -> None:
    """Fail at session start, with the fix, when nothing listens at the server's address (before, the session
    opened and the first task failed). Any HTTP answer, even an error page, means a server is there."""
    import urllib.error
    import urllib.request
    try:
        urllib.request.urlopen(url, timeout=3).close()
    except urllib.error.HTTPError:
        return
    except (urllib.error.URLError, OSError) as err:
        start = {"mlx-server": "start it with `uv run mlx_lm.server --model <model folder> --port 8080` (see "
                               "`keel doctor --backend mlx-server`)",
                 "ollama": "start Ollama (`ollama serve`) and pull the model"}.get(backend, "start the server")
        raise SystemExit(f"no model server answers at {url} ({getattr(err, 'reason', err)}): {start}, or point keel "
                         "at it with --url. Or leave out --backend to run the model inside keel.") from None


def server_engine(a: argparse.Namespace) -> Engine:
    """A local server (raw or chat mode). Same interactive sampling as MLX; replies stream where the server can
    (mlx_lm.server in raw mode returns them whole: token-id mode)."""
    from .backends.choose import agent_prompt_fn, server_setup
    from .specs import QWEN35_4B_8BIT, ModelSpec, SamplingSpec
    b = QWEN35_4B_8BIT
    spec = ModelSpec(b.name, b.hf_repo, b.revision, b.local_dir, {"enable_thinking": False})
    sampling = SamplingSpec(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=0.0, max_tokens=4096, seed=0)
    setup = server_setup(a, spec, sampling, stream=True)
    _check_reachable(setup.backend.url, a.backend)
    host = setup.backend.url.split("://", 1)[-1]
    name = f"{a.backend} {setup.backend.mode} · {host}"
    return Engine(setup.backend, agent_prompt_fn(setup, spec), setup.trimmer,
                  RunLimits(batch_size=1, max_context=16000, progress_every=0), name, setup.capabilities_record())


# --- one session -----------------------------------------------------------------------------------------------

class Session:
    def __init__(self, sid: str, folder: Path, engine: Engine, state_dir: Path, confirm, stack: str = SHIPPED):
        self.sid, self.engine = sid, engine
        self.sandbox = Sandbox.for_folder(folder, sid)            # scratch in the system temp folder
        self.work = self.sandbox.work
        self.record_dir = Path(state_dir) / "sessions" / sid     # what is kept: the transcript
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.busy = False
        self.updates: asyncio.Queue | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.calls = 0
        self.open: list[dict] = []               # tool calls started and not finished (sequential within a reply)
        self.asking: set = set()                 # open permission questions (futures), resolved as "no" on cancel
        approver = reporting(lambda command, files: confirm(self, command, files), self.on_event, sid)
        shell = GuardedShell(self.sandbox, approver, backup_dir=Path(state_dir) / "backups" / sid)
        item = SessionItem(task=sid)
        st = STACKS[stack]                       # the same stack definition the evals measure (keel.stacks)
        self.stack = stack
        self.ep = Episode(item, self.sandbox, shell, [{"role": "system", "content": st.system_prompt()}],
                          toolset=st.toolset,
                          ftools=FileTools(shell) if base_toolset(st.toolset) in ("files", "files_py") else None,
                          loop_break=st.loop_break, trimmer=engine.trimmer if st.trim else None,
                          events=self.on_event)
        # the stack's prompt already carries its examples / send note (Stack.system_prompt)
        configure(self.ep, done_gate=st.done_gate, search_hint=st.search_hint, gnu_tools=st.gnu_tools)

    def fit(self, incoming: str) -> dict:
        """Shorten the conversation if it plus `incoming` would crowd the context (keel.agent.compact). The full
        conversation is saved first, so the session record keeps everything."""
        e, limit = self.ep, self.engine.limits.max_context
        if not limit:
            return {}
        msgs, done = compact(e.messages, self.engine.count, limit, incoming)
        if not done:
            return {}
        n = len(list(self.record_dir.glob("before-shortening-*.json"))) + 1
        (self.record_dir / f"before-shortening-{n}.json").write_text(json.dumps(e.messages, default=str))
        e.messages[:] = msgs
        e.notes.append(f"shortened@{e.steps}: {done}")
        log("long session:", done)
        return done

    # core events (worker threads) -> ACP updates, in order, through one queue
    def push(self, update) -> None:
        if self.loop is not None and self.updates is not None:
            self.loop.call_soon_threadsafe(self.updates.put_nowait, update)

    def _read(self, rel: str) -> str | None:
        p = (self.work / rel).resolve()
        if self.work not in p.parents or not p.is_file() or p.stat().st_size > MAX_DIFF_BYTES:
            return None
        try:
            return p.read_text()
        except (UnicodeDecodeError, OSError):
            return None

    def on_event(self, ev: Event) -> None:
        d = ev.data
        if ev.kind == "text_delta":
            self.push(s.AgentMessageChunk(session_update="agent_message_chunk",
                                          content=s.TextContentBlock(type="text", text=d["text"])))
        elif ev.kind == "tool_started":
            self.calls += 1
            name, args = d["name"], d.get("args") if isinstance(d.get("args"), dict) else {}
            call = {"id": f"call_{self.calls}", "name": name, "path": None, "old": None}
            if name in WRITES and isinstance(args.get("path"), str):
                call["path"], call["old"] = args["path"], self._read(args["path"])
            self.open.append(call)
            self.push(s.ToolCallStart(session_update="tool_call", tool_call_id=call["id"], title=_title(name, args),
                                      kind=KIND.get(name, "other"), status="in_progress", raw_input=args))
        elif ev.kind == "tool_finished":
            call = self.open.pop(0) if self.open else {"id": f"call_{self.calls}", "path": None, "old": None}
            content: list = [s.ContentToolCallContent(type="content", content=s.TextContentBlock(
                type="text", text=str(d.get("output", ""))[:4000]))]
            if call["path"] and not d.get("error"):
                new = self._read(call["path"])
                if new is not None and new != call["old"]:
                    content.insert(0, s.FileEditToolCallContent(type="diff", path=str(self.work / call["path"]),
                                                                old_text=call["old"], new_text=new))
            self.push(s.ToolCallProgress(session_update="tool_call_update", tool_call_id=call["id"],
                                         status="failed" if d.get("error") else "completed", content=content))

    def finish(self, e: Episode) -> None:
        rec = {"session": self.sid, "folder": str(self.work), "stop": e.stop, "steps": e.steps,
               "backend": self.engine.name, **({"error": e.backend_error} if e.backend_error else {}),
               "commands": e.commands, "confirmations": [c.__dict__ for c in e.shell.log], "notes": e.notes,
               "messages": e.messages}
        (self.record_dir / "transcript.json").write_text(json.dumps(rec, indent=1, default=str))

    def close(self) -> None:
        self.sandbox.remove()


def prune_sessions(sessions_dir: Path, keep: int, active: set, backups_dir: Path | None = None) -> int:
    """Keep the newest `keep` session records (by last change); delete older ones. Only keel's own `s<hex>`
    folders are considered, and never an active session. A pruned session's backups (`backups_dir/<sid>`) go with it,
    and so do backups whose session record is already gone (backups used to grow forever)."""
    import re
    import shutil
    pattern = re.compile(r"s[0-9a-f]{10}")
    dirs = [p for p in sessions_dir.iterdir() if p.is_dir() and pattern.fullmatch(p.name)
            and p.name not in active] if sessions_dir.is_dir() else []
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    old = dirs[max(0, keep - len(active)):]
    for p in old:
        shutil.rmtree(p, ignore_errors=True)
    if backups_dir is not None and backups_dir.is_dir():
        for b in backups_dir.iterdir():
            if b.is_dir() and pattern.fullmatch(b.name) and b.name not in active and not (sessions_dir / b.name).exists():
                shutil.rmtree(b, ignore_errors=True)
    return len(old)


def _title(name: str, args: dict) -> str:
    if name == "run_shell":
        return f"run: {str(args.get('command', ''))[:80]}"
    target = args.get("path") or args.get("paths") or args.get("source") or args.get("pattern") or ""
    if isinstance(target, list):
        target = ", ".join(map(str, target[:3])) + (" ..." if len(target) > 3 else "")
    return f"{name} {target}".strip()


# --- the ACP agent -----------------------------------------------------------------------------------------------

class KeelAgent:
    def __init__(self, make_engine, state_dir: Path | None = None, stack: str = SHIPPED):
        self.make_engine = make_engine          # called once, on the first session (MLX loads the model there)
        self.stack = stack                      # keel.stacks name every session uses
        self.engine: Engine | None = None
        self.state_dir = Path(state_dir or default_state_dir())
        self.sessions: dict[str, Session] = {}
        self.conn = None
        self._engine_lock = threading.Lock()

    def on_connect(self, conn) -> None:
        self.conn = conn

    async def initialize(self, protocol_version, client_capabilities=None, client_info=None, **kw):
        log("initialize:", getattr(client_info, "name", "?"), "protocol", protocol_version)
        return s.InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_capabilities=s.AgentCapabilities(load_session=False),
            agent_info=s.Implementation(name="keel-core", title="keel", version=VERSION),
            field_meta={"keel": {"deleteGuard": "always", "coreVersion": VERSION}})

    def _get_engine(self) -> Engine:
        with self._engine_lock:
            if self.engine is None:
                self.engine = self.make_engine()
                log("backend ready:", self.engine.name)
            return self.engine

    async def new_session(self, cwd, additional_directories=None, mcp_servers=None, **kw):
        try:
            engine = await asyncio.to_thread(self._get_engine)
        except SystemExit as err:                # a refused backend config (e.g. raw mode without the model folder):
            msg = str(err.code) if err.code not in (None, 0) else "the model backend could not start"
            log("backend refused:", msg)         # a protocol error the client can show, not a torn-down event loop
            raise RequestError(-32602, msg, {"reason": "backend_refused"}) from None
        sid = "s" + uuid.uuid4().hex[:10]
        await asyncio.to_thread(prune_sessions, self.state_dir / "sessions", KEEP_SESSIONS, set(self.sessions),
                                self.state_dir / "backups")
        try:
            sess = await asyncio.to_thread(Session, sid, Path(cwd), engine, self.state_dir, self._confirm, self.stack)
        except FolderRefused as err:
            raise RequestError(-32602, str(err), {"reason": "folder_refused"}) from err
        self.sessions[sid] = sess
        (sess.record_dir / "capabilities.json").write_text(json.dumps(engine.caps, indent=1, default=str))
        log("session", sid, "in", sess.work)
        return s.NewSessionResponse(session_id=sid,
                                    field_meta={"keel": {"folder": str(sess.work), "backend": engine.name,
                                                         "capabilities": engine.caps.get("capabilities")}})

    async def prompt(self, session_id, prompt, **kw):
        sess = self.sessions.get(session_id)
        if sess is None:
            raise RequestError(-32602, f"unknown session {session_id}")
        if sess.busy:
            raise RequestError(-32602, "this session is already working on a prompt")
        text = "\n".join(b.text for b in prompt if getattr(b, "type", "") == "text").strip()
        if not text:
            raise RequestError(-32602, "empty prompt")
        sess.busy, sess.loop, sess.updates = True, asyncio.get_running_loop(), asyncio.Queue()
        e = sess.ep
        note = await asyncio.to_thread(sess.fit, text)  # long sessions: shorten old turns before this one
        if note.get("summary_prefix"):
            text = note["summary_prefix"] + text
        e.messages.append({"role": "user", "content": text})
        n_before, steps_before = len(e.messages), e.steps
        e.done, e.stop, e.answer, e.cancelled, e.backend_error = False, "", "", False, ""
        sender = asyncio.create_task(self._send(session_id, sess.updates))
        if notice(note):                         # on the event loop thread: straight into the queue (push() would
            sess.updates.put_nowait(s.AgentMessageChunk(  # schedule it, and a join() could pass before it lands)
                session_update="agent_message_chunk", content=s.TextContentBlock(type="text", text=notice(note))))
        hooks = episode_hooks(sess.engine.prompt_fn, e.steps + MAX_STEPS, sess.record_dir, [], stream=True,
                              finish=sess.finish)
        try:
            await asyncio.to_thread(sess.engine.backend.run, [e], hooks, sess.engine.limits)
            if e.stop == "context_limit":        # one turn outgrew the context: say what to do
                sess.updates.put_nowait(s.AgentMessageChunk(
                    session_update="agent_message_chunk", content=s.TextContentBlock(type="text", text=FULL)))
        finally:
            await sess.updates.join()            # every update sent before the prompt returns
            sender.cancel()
            sess.busy = False
        if e.stop == "backend_error":
            msg = e.backend_error or "the model server failed"
            if e.steps == steps_before and len(e.messages) == n_before:
                e.messages.pop()                 # no reply at all: take the prompt back, so a retry isn't doubled
                sess.finish(e)                   # the saved transcript too
            log("backend error:", msg)
            raise RequestError(-32603, f"model server error: {msg}", {"reason": "backend_error"})
        return s.PromptResponse(stop_reason=STOP.get(e.stop, "end_turn"),
                                field_meta={"keel": {"stop": e.stop, "steps": e.steps}})

    async def _send(self, sid: str, q: asyncio.Queue) -> None:
        while True:
            upd = await q.get()
            try:
                await self.conn.session_update(sid, upd)
            except Exception as err:             # a dead client must not hang the prompt
                log("update not sent:", repr(err))
                if isinstance(err, ConnectionError) and sid in self.sessions:
                    self.sessions[sid].ep.cancelled = True     # killed while the model was working: stop
            finally:
                q.task_done()

    async def cancel(self, session_id, **kw):
        sess = self.sessions.get(session_id)
        if sess is not None:
            sess.ep.cancelled = True
            for fut in list(sess.asking):        # an open delete question resolves as "no"
                fut.cancel()
            log("cancel", session_id)

    def close_all(self) -> None:
        """keel-core is exiting: delete every session's scratch folder (transcripts stay)."""
        for sess in self.sessions.values():
            sess.close()

    def client_gone(self) -> None:
        """The client disconnected (stdin closed / the TUI was killed): cancel every session and refuse every open
        delete question, so the agent stops before it touches the folder again."""
        for sess in self.sessions.values():
            sess.ep.cancelled = True
            for fut in list(sess.asking):
                fut.cancel()
        if self.sessions:
            log("client gone: all sessions cancelled")

    def _confirm(self, sess: Session, command: str, files: list[str]) -> bool:
        """The delete guard's question (called from a tool thread, which waits for the answer). True only for an
        explicit allow-once on this exact question; every other outcome, including any failure, is False."""
        if sess.loop is None or sess.ep.cancelled:
            return False
        fut = asyncio.run_coroutine_threadsafe(self._ask(sess, command, files), sess.loop)
        sess.asking.add(fut)
        try:
            return fut.result() is True
        except BaseException as err:             # cancelled, disconnected, malformed reply, client error
            log("delete refused:", type(err).__name__, str(err)[:120])
            if isinstance(err, ConnectionError):
                sess.ep.cancelled = True         # the client is gone: stop now, not at shutdown (it may be seconds
            return False                         # away, and the next step could write to the folder)
        finally:
            sess.asking.discard(fut)

    async def _ask(self, sess: Session, command: str, files: list[str]) -> bool:
        await sess.updates.join()                # the client has seen the delete's tool call first
        call_id = sess.open[-1]["id"] if sess.open else f"call_{sess.calls}"
        paths = [str(sess.work / f) for f in files]
        allow_id, reject_id = f"allow-{call_id}", f"reject-{call_id}"
        listing = "\n".join(paths[:50]) + (f"\n... and {len(paths) - 50} more" if len(paths) > 50 else "")
        resp = await self.conn.request_permission(
            sess.sid,
            s.ToolCallUpdate(tool_call_id=call_id, kind="delete",
                             title=f"Delete {len(files)} file{'s' if len(files) != 1 else ''}?",
                             raw_input={"command": command, "files": paths},
                             locations=[s.ToolCallLocation(path=p) for p in paths[:50]],
                             content=[s.ContentToolCallContent(type="content", content=s.TextContentBlock(
                                 type="text", text=f"{command}\n\nThis deletes:\n{listing}"))]),
            [s.PermissionOption(option_id=allow_id, name="Delete", kind="allow_once"),
             s.PermissionOption(option_id=reject_id, name="Keep", kind="reject_once")])
        out = getattr(resp, "outcome", None)
        return getattr(out, "outcome", None) == "selected" and getattr(out, "option_id", None) == allow_id


# --- stdio wiring ------------------------------------------------------------------------------------------------

def _protocol_stdout():
    """Keep the real stdout for the protocol; point fd 1 (and sys.stdout) at stderr for everything else."""
    proto = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return proto


def _pipe_for(out):
    """asyncio writes only to pipes, sockets and terminals. When stdout is a regular file (a launcher or a test
    redirecting to a file), relay through an internal pipe with a small copying thread."""
    mode = os.fstat(out.fileno()).st_mode
    if stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode) or stat.S_ISCHR(mode):
        return out
    r, w = os.pipe()

    def copy():
        with os.fdopen(r, "rb", buffering=0) as src:
            while chunk := src.read(65536):
                out.write(chunk)
                out.flush()
    threading.Thread(target=copy, daemon=True, name="keel-stdout-relay").start()
    return os.fdopen(w, "wb", buffering=0)


async def _serve(agent: KeelAgent, proto_out) -> None:
    from acp.stdio import _WritePipeProtocol
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=16 * 1024 * 1024)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    wp = _WritePipeProtocol()
    transport, _ = await loop.connect_write_pipe(lambda: wp, _pipe_for(proto_out))
    writer = asyncio.StreamWriter(transport, wp, None, loop)
    try:
        await acp.run_agent(agent, writer, reader)
    finally:
        agent.client_gone()                      # stop every session: no work in the folder after the client left


def add_arguments(ap: argparse.ArgumentParser) -> None:
    from .backends.choose import add_backend_arguments
    add_backend_arguments(ap, extra_choices=("scripted",))
    ap.add_argument("--script", default=None, help="scripted backend: JSON list of replies, or {session: [...]}")
    ap.add_argument("--script-delay", type=float, default=0.0, help="scripted backend: seconds per streamed chunk")
    ap.add_argument("--state-dir", default=None, help="keel state folder (default ~/Library/Application Support/keel)")
    ap.add_argument("--stack", choices=sorted(STACKS), default=SHIPPED,
                    help=f"harness stack for every session (default {SHIPPED}; see keel.stacks)")


def run(a: argparse.Namespace) -> None:
    proto = _protocol_stdout()
    if a.backend == "scripted":
        make = lambda: scripted_engine(a.script, a.script_delay)  # noqa: E731
    elif a.backend == "mlx":
        make = mlx_engine
    else:
        make = lambda: server_engine(a)  # noqa: E731
    agent = KeelAgent(make, Path(a.state_dir) if a.state_dir else None, a.stack)
    log(f"keel-core {VERSION} acp, backend {a.backend}, stack {a.stack}")
    swept = sweep_scratch()
    if swept:
        log(f"removed {swept} scratch folder(s) left by earlier crashed sessions")
    try:
        asyncio.run(_serve(agent, proto))
    finally:
        agent.close_all()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="keel-core ACP agent on stdio")
    add_arguments(ap)
    run(ap.parse_args(argv))
