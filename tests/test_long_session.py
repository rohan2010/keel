"""Ch7 P7 (D3, audit #16): a TUI session that grows past the model's context keeps working (old tool outputs cut,
then old turns summarised, the user told), instead of stopping with context_limit."""

import asyncio
import json
from types import SimpleNamespace

from keel.acp_server import Engine, KeelAgent
from keel.agent.compact import SHRUNK, compact, notice
from keel.backends.scripted import ScriptedBackend
from keel.core.backend import RunLimits

LIMIT = 16000


def count(text: str) -> int:                     # ~ 1 token per 4 characters
    return len(text) // 4


class LimitedScripted(ScriptedBackend):
    """The scripted backend, but a prompt over the context fails exactly like in-process MLX (stop = context_limit)."""

    def __init__(self, script, seen):
        super().__init__(script)
        self.seen = seen

    def run(self, episodes, hooks, limits):
        for e in episodes:
            used = sum(count(str(m.get("content") or "")) for m in e.messages)
            self.seen.append(used)
            if used > limits.max_context:
                e.done, e.stop = True, "context_limit"
                hooks.finish_fn(e)
                return {}
        return super().run(episodes, hooks, limits)


class Conn:
    def __init__(self):
        self.text = []

    async def session_update(self, sid, upd):
        t = getattr(getattr(upd, "content", None), "text", None)
        if t:
            self.text.append(t)


def call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def _session(tmp_path, turns):
    folder = tmp_path / "proj"
    folder.mkdir()
    for i in range(turns):
        (folder / f"notes{i}.txt").write_text(f"note {i}: " + "word " * 2400)        # ~3,000 tokens each
    script = []
    for i in range(turns):
        script += [call("read_file", path=f"notes{i}.txt"), f"notes{i}.txt is about note {i}."]
    seen = []
    engine = Engine(LimitedScripted(script, seen), lambda e: [dict(m) for m in e.messages], None,
                    RunLimits(batch_size=1, max_context=LIMIT), "scripted", count=count)
    agent = KeelAgent(lambda: engine, tmp_path / "state", stack="keel")
    agent.on_connect(conn := Conn())
    return agent, conn, seen, folder


def test_thirty_turns_past_the_context_keep_working(tmp_path):
    turns = 30
    agent, conn, seen, folder = _session(tmp_path, turns)

    async def go():
        sid = (await agent.new_session(cwd=str(folder))).session_id
        stops = []
        for i in range(turns):
            r = await agent.prompt(sid, [SimpleNamespace(type="text", text=f"what is in notes{i}.txt?")])
            stops.append(r.field_meta["keel"]["stop"])
        return sid, stops

    sid, stops = asyncio.run(go())
    assert stops == ["answer"] * turns                     # never context_limit
    assert max(seen) <= LIMIT and sum(seen) > 3 * LIMIT     # the whole talk was far bigger than the context
    said = "".join(conn.text)
    assert "getting long for the model's memory" in said and "fresh start" in said
    record = tmp_path / "state" / "sessions" / sid
    assert list(record.glob("before-shortening-*.json"))   # the full conversation is kept on disk
    assert "notes29.txt is about note 29." in said


def test_without_shortening_the_same_session_would_stop(tmp_path, monkeypatch):
    import keel.acp_server as A
    monkeypatch.setattr(A, "compact", lambda messages, *a, **k: (messages, {}))
    agent, conn, seen, folder = _session(tmp_path, 25)

    async def go():
        sid = (await agent.new_session(cwd=str(folder))).session_id
        out = []
        for i in range(25):
            r = await agent.prompt(sid, [SimpleNamespace(type="text", text=f"what is in notes{i}.txt?")])
            out.append(r.field_meta["keel"]["stop"])
        return out

    stops = asyncio.run(go())
    assert "context_limit" in stops
    assert "no longer fits the model's memory" in "".join(conn.text)


def test_old_tool_outputs_are_cut_first_and_the_tail_is_untouched():
    sys_ = {"role": "system", "content": "s"}
    msgs = [sys_]
    for i in range(14):
        msgs += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": call("read_file", path=f"f{i}")},
                 {"role": "tool", "content": "x" * 4000}, {"role": "assistant", "content": f"a{i}"}]
    out, done = compact(msgs, count, 16000)
    assert done["shrunk_outputs"] > 0 and "summarised_messages" not in done
    assert out[-2]["content"] == "x" * 4000                 # the newest tool output is kept whole
    assert all(m["content"].endswith(SHRUNK) for m in out[1:-6] if m["role"] == "tool")


def test_summary_names_requests_answers_and_files():
    msgs = [{"role": "system", "content": "s"}]
    for i in range(40):
        msgs += [{"role": "user", "content": f"rename photo {i}"},
                 {"role": "assistant", "content": call("move", source=f"p{i}.jpg", destination=f"2024/p{i}.jpg")},
                 {"role": "tool", "content": "moved " + "y" * 1500},
                 {"role": "assistant", "content": f"Renamed {i}. " + "detail " * 180}]
    out, done = compact(msgs, count, 16000, incoming="and the rest?")
    assert done["summarised_messages"] > 0 and len(out) < len(msgs)
    first = out[1]["content"]
    assert first.startswith("[keel: the start of this conversation was shortened")
    assert "the user asked: rename photo 0" in first and "p0.jpg" in first and "Renamed" in first
    assert out[1]["role"] == "user" and out[0]["role"] == "system"
    assert "summarised the first" in notice(done)


def test_short_conversations_are_left_alone():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    assert compact(msgs, count, 16000) == (msgs, {}) and notice({}) == ""
    assert json.dumps(msgs)                                 # unchanged objects
