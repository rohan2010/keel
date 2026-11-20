"""Chapter 3, P3: the loop breaker's ban ladder on every server mode + flavour (fake server, no model)."""

import pytest
from test_server_backend import TC_ID, Fake, call, decode, openai_ids
from test_server_chat import FUNCS, chat_answer, tc

from keel.agent.loop import chat_prompt, run_episodes_on
from keel.backends.server import DEFAULT_RUNGS, LADDER, ServerBackend
from keel.evals import tasks as T

CHAT = {"messages": [{"role": "user", "content": "hi"}], "tools": [], "funcs": FUNCS}


@pytest.fixture
def fake():
    made = []

    def make(respond, **kw):
        f = Fake(respond, **kw)
        made.append(f)
        return f
    yield make
    for f in made:
        f.close()


def sent(body):
    """Which ban fields a request carried."""
    opts = body.get("options") or {}
    return sorted(k for k in ("logit_bias", "tool_choice", "stop") if k in body or k in opts)


# (mode, flavour, kwargs, expected rung): the first rung in LADDER that is set and can be sent
CASES = [
    ("raw", "openai", {"tool_call_id": TC_ID}, "logit_bias"),
    ("raw", "openai", {}, "stop"),                                          # no token id: next rung
    ("raw", "llama", {"tool_call_id": TC_ID}, "logit_bias"),
    ("raw", "ollama", {"tool_call_id": TC_ID}, "stop"),                     # Ollama has no logit_bias
    ("chat", "openai", {"tool_call_id": TC_ID}, "logit_bias"),
    ("chat", "openai", {}, "tool_choice"),
    ("chat", "openai", {"ban_rungs": ("stop",)}, "stop"),                   # what doctor sets for mlx_lm.server
    ("chat", "ollama", {}, "stop"),
    ("raw", "openai", {"ban_rungs": (), "tool_call_id": TC_ID}, "none"),
    ("chat", "openai", {"ban_rungs": ()}, "none"),
]


@pytest.mark.parametrize("mode,flavour,kw,rung", CASES)
def test_each_rung_is_chosen_and_sent(fake, mode, flavour, kw, rung):
    if flavour == "ollama":
        ans = {"message": {"content": "ok"}, "done": True} if mode == "chat" else {"response": "ok", "done": True}
    elif flavour == "llama":
        ans = {"content": "ok", "stop_type": "eos"}
    else:
        ans = chat_answer("ok") if mode == "chat" else openai_ids("ok")
    f = fake(lambda p, b, n: ans)
    be = ServerBackend(f.url, flavour, mode=mode, decode=decode if (mode, flavour) == ("raw", "openai") else None,
                       **kw)
    assert be.ban_rung() == rung and be.sampling_record()["ban_rung"] == rung
    r = be.request(CHAT if mode == "chat" else "p", forbid_tools=True)
    assert r.ban_method == rung and not r.ban_cut and r.text == "ok"
    assert sent(f.requests[-1][1]) == ([] if rung == "none" else [rung])
    be.request(CHAT if mode == "chat" else "p")                               # no ban asked: nothing sent
    assert sent(f.requests[-1][1]) == []
    caps = be.capabilities()
    assert caps.logit_bias == (rung == "logit_bias") and caps.forbid_tools_next_turn == (rung != "none")


def test_calls_that_get_through_are_cut_on_every_rung(fake):
    f = fake(lambda p, b, n: openai_ids("Fine.\n" + call("list_files")))     # a server that ignores everything
    for kw, rung in (({"tool_call_id": TC_ID}, "logit_bias"), ({}, "stop"), ({"ban_rungs": ()}, "none")):
        r = ServerBackend(f.url, "openai", decode=decode, **kw).request("p", forbid_tools=True)
        assert r.text == "Fine." and (r.ban_method, r.ban_cut) == (rung, True)
    f2 = fake(lambda p, b, n: chat_answer("Fine.", [tc("list_files", path=".")]))
    for kw, rung in (({"tool_call_id": TC_ID}, "logit_bias"), ({}, "tool_choice"), ({"ban_rungs": ("stop",)}, "stop"),
                     ({"ban_rungs": ()}, "none")):
        r = ServerBackend(f2.url, "openai", mode="chat", **kw).request(CHAT, forbid_tools=True)
        assert r.text == "Fine." and (r.ban_method, r.ban_cut) == (rung, True)


def test_stop_rung_through_the_agent_loop_on_ollama_chat(fake, tmp_path):
    def respond(p, b, n):
        if "stop" in b["options"]:
            return {"message": {"content": "Nothing found."}, "done": True}
        return {"message": {"content": "", "tool_calls": [{"function": {"name": "list_files",
                                                                         "arguments": {"path": "."}}}]}, "done": True}
    f = fake(respond)
    stats = {}
    be = ServerBackend(f.url, "ollama", mode="chat")
    orig = be.run
    be.run = lambda eps, h, lim: stats.update(orig(eps, h, lim)) or stats  # type: ignore[method-assign]
    res = run_episodes_on(be, [T.build("C1", 0)], tmp_path, prompt_fn=chat_prompt, loop_break=True)
    assert res[0]["answer"] == "Nothing found." and "ban:stop@4" in res[0]["notes"]
    assert stats["ban_methods"] == {"stop": 1} and stats["ban_cut"] == 0


def test_rungs_a_wire_format_cannot_carry_are_refused():
    with pytest.raises(ValueError, match="tool_choice"):
        ServerBackend("http://x", "openai", ban_rungs=("tool_choice",))      # raw completions have no tool_choice
    with pytest.raises(ValueError, match="logit_bias"):
        ServerBackend("http://x", "ollama", mode="chat", ban_rungs=("logit_bias",))
    assert set(LADDER) == {r for v in DEFAULT_RUNGS.values() for r in v}
    assert ServerBackend("http://x", "openai", ban_rungs=["stop"]).ban_rungs == ("stop",)


def test_stop_token_in_returned_ids_is_not_a_cut(fake):
    """mlx_lm.server, token-id mode: a stop sequence halts the reply, but the ids it returns end with the stop token
    itself, so the decoded text ends in `<tool_call>` (found by keel doctor, P6). That is the rung working."""
    f = fake(lambda p, b, n: openai_ids("I'll stop here.\n<tool_call>", finish="stop", end=""))
    r = ServerBackend(f.url, "openai", decode=decode, ban_rungs=("stop",)).request("p", forbid_tools=True)
    assert (r.text, r.ban_method, r.ban_cut) == ("I'll stop here.", "stop", False)
    f2 = fake(lambda p, b, n: openai_ids("x " + call("list_files") + "\n<tool_call>", finish="stop", end=""))
    r = ServerBackend(f2.url, "openai", decode=decode, ban_rungs=("stop",)).request("p", forbid_tools=True)
    assert r.text == "x" and r.ban_cut                                       # a full call before it: really ignored
