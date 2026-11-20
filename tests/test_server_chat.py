"""Chapter 3, P2: chat-mode server backend and the transcript <-> chat conversion (fake server, no model, no MLX)."""

import json

import pytest
from test_server_backend import TC_ID, Fake, call  # noqa: F401  (Fake is used through the fixture)

from keel.agent.loop import TOOLSETS, chat_prompt, run_episodes_on
from keel.backends.chat import calls_text, parse_args, reply_text, to_chat
from keel.backends.server import ServerBackend
from keel.core.backend import RunLimits
from keel.core.events import ListSink
from keel.core.parse import parse_calls
from keel.evals import tasks as T

FUNCS = TOOLSETS["files"][2]


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


def tc(name, cid="srv-uuid", **args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def chat_answer(content="", calls=(), finish=None, prompt_tokens=100, cached=0, reasoning=None):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = list(calls)
    if reasoning:
        msg["reasoning"] = reasoning
    return {"choices": [{"message": msg, "finish_reason": finish or ("tool_calls" if calls else "stop")}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 10,
                      "prompt_tokens_details": {"cached_tokens": cached}}}


# ---- conversion ----------------------------------------------------------------------------------------------------

TRANSCRIPT = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "do it"},
    {"role": "assistant", "content": "Writing two things.\n\n"
        + call("write_file", path="a.txt", content="line one\n  indented\n") + "\n"
        + call("edit_file", path="b.md", old_text="x", new_text="y\n") + "\n"
        + call("run_shell", command="ls -la | wc -l")},
    {"role": "tool", "content": "wrote a.txt", "_step": 1, "_call": "write_file(...)"},
    {"role": "tool", "content": "edited b.md", "_step": 1, "_call": "edit_file(...)"},
    {"role": "tool", "content": "3", "_step": 1, "_call": "run_shell(...)"},
    {"role": "assistant", "content": call("list_files", path=".")},
    {"role": "tool", "content": "a.txt b.md", "_step": 2},
    {"role": "user", "content": "Check your work."},
    {"role": "assistant", "content": "All done."},
]


def test_transcript_to_chat_and_back_loses_nothing():
    msgs = to_chat(TRANSCRIPT, FUNCS, "openai")
    assert [m["role"] for m in msgs] == [m["role"] for m in TRANSCRIPT]
    a = msgs[2]
    assert a["content"] == "Writing two things." and [t["id"] for t in a["tool_calls"]] == [
        "call_2_0", "call_2_1", "call_2_2"]
    assert [m["tool_call_id"] for m in msgs[3:6]] == ["call_2_0", "call_2_1", "call_2_2"]
    assert msgs[7]["tool_call_id"] == "call_6_0" and msgs[6]["content"] == ""
    assert all(not k.startswith("_") for m in msgs for k in m)
    args0 = json.loads(a["tool_calls"][0]["function"]["arguments"])
    assert args0["content"] == "line one\n  indented\n"                    # file text keeps its own whitespace

    for i in (2, 6):                                    # structured calls -> markup -> the same parsed calls
        orig = parse_calls(TRANSCRIPT[i]["content"], FUNCS)
        back = [(t["function"]["name"], parse_args(t["function"]["arguments"])[0]) for t in msgs[i]["tool_calls"]]
        text = reply_text(msgs[i]["content"], back)
        assert parse_calls(text, FUNCS) == orig
        assert text.split("<tool_call>")[0].strip() == TRANSCRIPT[i]["content"].split("<tool_call>")[0].strip()


def test_ollama_style_uses_objects_and_tool_names():
    msgs = to_chat(TRANSCRIPT, FUNCS, "ollama")
    assert msgs[2]["tool_calls"][2] == {"function": {"name": "run_shell", "arguments": {"command": "ls -la | wc -l"}}}
    assert [m["tool_name"] for m in msgs[3:6]] == ["write_file", "edit_file", "run_shell"]
    assert "tool_call_id" not in msgs[3]


def test_orphan_tool_result_becomes_a_user_note_and_bad_style_refused():
    msgs = to_chat([{"role": "assistant", "content": "hi"}, {"role": "tool", "content": "?"}], FUNCS)
    assert msgs[1] == {"role": "user", "content": "?"}
    with pytest.raises(ValueError):
        to_chat([], FUNCS, "anthropic")


def test_non_string_values_and_args_parsing():
    text = calls_text([("search", {"pattern": "x", "max": 3, "flags": ["i"], "on": True})])
    assert "<parameter=max>\n3\n</parameter>" in text and '["i"]' in text and "\nTrue\n" in text   # bug 9: as the model writes it
    assert parse_args('{"a": 1}') == ({"a": 1}, True)
    assert parse_args({"a": 1}) == ({"a": 1}, True)
    assert parse_args("") == ({}, True)
    assert parse_args("{not json") == ({}, False) and parse_args("[1]") == ({}, False)
    assert reply_text("\n\n\n", [("list_files", {})]).startswith("<tool_call>")   # mlx_lm.server's leftover content
    assert reply_text("ok", [], reasoning="hmm").startswith("<think>\nhmm\n</think>")


# ---- through the agent loop ----------------------------------------------------------------------------------------

def test_multi_call_round_trip_through_the_agent_loop(fake, tmp_path):
    replies = [chat_answer("\n\n\n", [tc("write_file", "u1", path="new.txt", content="hello keel\n"),
                                      tc("list_files", "u2", path=".")]),
               chat_answer("Created new.txt.", prompt_tokens=200, cached=150)]
    f = fake(lambda p, b, n: replies[n - 1])
    be = ServerBackend(f.url, "openai", mode="chat")
    res = run_episodes_on(be, [T.build("C1", 0)], tmp_path, prompt_fn=chat_prompt)
    r = res[0]
    assert r["stop"] == "answer" and r["answer"] == "Created new.txt." and r["steps"] == 2
    assert r["commands"][0].startswith("write_file") and r["commands"][1].startswith("list_files")
    written = list((tmp_path / "sandboxes" / "C1_0").rglob("new.txt"))
    assert len(written) == 1 and written[0].read_text() == "hello keel\n"     # exact text, final newline kept

    path, body = f.requests[0]
    assert path == "/v1/chat/completions" and body["chat_template_kwargs"] == {"enable_thinking": False}
    assert [t["function"]["name"] for t in body["tools"]] == [s["function"]["name"] for s in TOOLSETS["files"][1]]
    assert "prompt" not in body and "logprobs" not in body and "tool_choice" not in body

    second = f.requests[1][1]["messages"]
    a = next(m for m in second if m["role"] == "assistant")
    assert [t["id"] for t in a["tool_calls"]] == ["call_2_0", "call_2_1"]       # keel's ids, not the server's
    tools = [m for m in second if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools] == ["call_2_0", "call_2_1"]
    assert json.loads(a["tool_calls"][0]["function"]["arguments"]) == {"path": "new.txt", "content": "hello keel\n"}

    saved = json.loads((tmp_path / "transcripts" / "C1_0.json").read_text())["messages"]
    assert "<tool_call>\n<function=write_file>" in saved[2]["content"]        # one transcript format on every backend


def test_ids_stay_the_same_in_every_request(fake, tmp_path):
    ls = tc("list_files", path=".")
    replies = [chat_answer("", [ls]), chat_answer("", [tc("read_file", path="a")]), chat_answer("Done.")]
    f = fake(lambda p, b, n: replies[n - 1])
    run_episodes_on(ServerBackend(f.url, "openai", mode="chat"), [T.build("C1", 0)], tmp_path, prompt_fn=chat_prompt)
    ids = [[t["id"] for m in b["messages"] if m["role"] == "assistant" for t in m.get("tool_calls", [])]
           for _, b in f.requests]
    assert ids == [[], ["call_2_0"], ["call_2_0", "call_4_0"]]


def test_ollama_chat(fake, tmp_path):
    replies = [{"message": {"role": "assistant", "content": "",
                            "tool_calls": [{"function": {"name": "list_files", "arguments": {"path": "."}}}]},
                "done": True, "done_reason": "stop", "prompt_eval_count": 30, "eval_count": 5},
               {"message": {"role": "assistant", "content": "Nothing here.", "thinking": "short"},
                "done": True, "done_reason": "stop", "prompt_eval_count": 40, "eval_count": 3}]
    f = fake(lambda p, b, n: replies[n - 1])
    be = ServerBackend(f.url, "ollama", mode="chat", model="qwen3.5:4b")
    res = run_episodes_on(be, [T.build("C1", 0)], tmp_path, prompt_fn=chat_prompt)
    assert res[0]["answer"] == "Nothing here." and res[0]["steps"] == 2          # thinking stripped from the answer
    path, body = f.requests[0]
    assert path == "/api/chat" and body["think"] is False and body["options"]["top_k"] == 20
    assert body["model"] == "qwen3.5:4b" and "stop" not in body["options"]
    tool = [m for m in f.requests[1][1]["messages"] if m["role"] == "tool"]
    assert tool[0]["tool_name"] == "list_files" and "tool_call_id" not in tool[0]


def test_streaming_with_split_call_arguments_and_thinking(fake, tmp_path):
    args = json.dumps({"path": "."})
    chunks = [{"choices": [{"delta": {"reasoning": "let me "}}]},
              {"choices": [{"delta": {"reasoning": "look"}}]},
              {"choices": [{"delta": {"content": "Looking now."}}]},
              {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "x", "type": "function",
                                                      "function": {"name": "list_files", "arguments": args[:5]}}]}}]},
              {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": args[5:]}}]}}]},
              {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
              {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 9,
                                        "prompt_tokens_details": {"cached_tokens": 40}}}]
    final = [{"choices": [{"delta": {"content": "It is empty."}, "finish_reason": "stop"}]}]
    f = fake(lambda p, b, n: chunks if n == 1 else final)
    be = ServerBackend(f.url, "openai", mode="chat", stream=True)
    sink = ListSink()
    res = run_episodes_on(be, [T.build("C1", 0)], tmp_path, prompt_fn=chat_prompt, events=sink, stream=True)
    shown = [e.data["text"] for e in sink.events if e.kind == "text_delta"]
    assert "".join(shown) == "Looking now.\n\nIt is empty."        # no thinking, no call markup (as in-process)
    assert res[0]["steps"] == 2 and res[0]["answer"] == "It is empty."
    assert f.requests[0][1]["stream"] is True and f.requests[0][1]["stream_options"] == {"include_usage": True}
    tool_msgs = [m for m in f.requests[1][1]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 1 and tool_msgs[0]["tool_call_id"] == "call_2_0"


def test_forbid_tools_in_chat_mode(fake):
    f = fake(lambda p, b, n: chat_answer("Text only."))
    p = {"messages": [{"role": "user", "content": "hi"}], "tools": [], "funcs": FUNCS}
    r = ServerBackend(f.url, "openai", mode="chat").request(p, forbid_tools=True)
    assert f.requests[-1][1]["tool_choice"] == "none" and "logit_bias" not in f.requests[-1][1]
    assert r.ban_method == "tool_choice"
    ServerBackend(f.url, "openai", mode="chat", tool_call_id=TC_ID).request(p, forbid_tools=True)
    assert f.requests[-1][1]["logit_bias"] == {str(TC_ID): -1e9}

    f2 = fake(lambda p_, b, n: chat_answer("I'll stop.", [tc("list_files", path=".")]))   # ban ignored
    r = ServerBackend(f2.url, "openai", mode="chat").request(p, forbid_tools=True)
    assert r.text == "I'll stop." and r.ban_method == "tool_choice" and r.ban_cut

    f3 = fake(lambda p_, b, n: {"message": {"content": "ok"}, "done": True})
    ServerBackend(f3.url, "ollama", mode="chat").request(p, forbid_tools=True)
    assert f3.requests[-1][1]["options"]["stop"] == ["<tool_call>"]


def test_loop_breaker_through_chat_mode(fake, tmp_path):
    ls = tc("list_files", path=".")
    f = fake(lambda p, b, n: chat_answer("Still looking.", [ls]) if n >= 4 else chat_answer("", [ls]))
    be = ServerBackend(f.url, "openai", mode="chat", tool_call_id=TC_ID)
    res = run_episodes_on(be, [T.build("C1", 0)], tmp_path, prompt_fn=chat_prompt, loop_break=True)
    assert any(n.startswith("loop@") for n in res[0]["notes"])
    assert res[0]["stop"] == "answer" and res[0]["answer"] == "Still looking."
    assert f.requests[3][1]["logit_bias"] == {str(TC_ID): -1e9} and "tool_choice" not in f.requests[3][1]
    assert "logit_bias" not in f.requests[2][1]
    assert "ban:logit_bias+cut@4" in res[0]["notes"]       # the rung sent, and that this server ignored it


def test_markup_in_content_still_parses(fake, tmp_path):
    """A server with no tool parser returns the model's markup as plain content: keel parses it as in raw mode."""
    replies = [chat_answer("Looking.\n" + call("list_files", path=".")), chat_answer("Empty.")]
    f = fake(lambda p, b, n: replies[n - 1])
    res = run_episodes_on(ServerBackend(f.url, "openai", mode="chat"), [T.build("C1", 0)], tmp_path,
                          prompt_fn=chat_prompt)
    assert res[0]["steps"] == 2 and res[0]["answer"] == "Empty."


def test_bad_arguments_counted_and_reach_the_tool_as_an_error(fake, tmp_path):
    bad = {"id": "z", "type": "function", "function": {"name": "read_file", "arguments": "{oops"}}
    replies = [chat_answer("", [bad]), chat_answer("Gave up.")]
    f = fake(lambda p, b, n: replies[n - 1])
    be = ServerBackend(f.url, "openai", mode="chat")
    r = be.request({"messages": [], "tools": [], "funcs": FUNCS})
    assert r.bad_args == 1 and "<function=read_file>" in r.text
    f2 = fake(lambda p, b, n: replies[n - 1])
    res = run_episodes_on(ServerBackend(f2.url, "openai", mode="chat"), [T.build("C1", 0)], tmp_path,
                          prompt_fn=chat_prompt)
    assert res[0]["n_errors"] == 1 and res[0]["answer"] == "Gave up."


def test_chat_config_and_records(fake):
    with pytest.raises(ValueError, match="mode"):
        ServerBackend("http://x", "openai", mode="tools")
    with pytest.raises(ValueError, match="raw mode only"):
        ServerBackend("http://x", "openai", mode="chat", token_ids=True)
    be = ServerBackend("http://x", "openai", mode="chat", decode=lambda ids: "")
    assert be.token_ids is False and be.name == "server-chat:openai:http://x"
    rec = be.sampling_record()
    assert rec["mode"] == "chat" and rec["thinking"] is False
    assert ServerBackend("http://x", "llama", mode="chat").chat_style == "openai"
    assert ServerBackend("http://x", "ollama", mode="chat", tool_call_id=TC_ID).capabilities().logit_bias is False
    assert ServerBackend("http://x", "openai", mode="chat", tool_call_id=TC_ID).capabilities().logit_bias is True

    f = fake(lambda p, b, n: chat_answer("hi"))
    ServerBackend(f.url, "openai", mode="chat", thinking=None).request({"messages": [], "tools": []})
    assert "chat_template_kwargs" not in f.requests[-1][1]
    with pytest.raises(TypeError, match="chat_prompt"):
        ServerBackend(f.url, "openai", mode="chat").request("rendered text")


def test_chat_run_stats(fake, tmp_path):
    f = fake(lambda p, b, n: chat_answer("No.", prompt_tokens=80, cached=60))
    be = ServerBackend(f.url, "openai", mode="chat")
    items = [T.build("F3", i) for i in range(2)]
    stats = {}
    orig = be.run

    def run(eps, hooks, limits):
        stats.update(orig(eps, hooks, limits))
        return stats
    be.run = run
    run_episodes_on(be, items, tmp_path, prompt_fn=chat_prompt, limits=RunLimits(batch_size=2))
    assert stats["requests"] == 2 and stats["cached_tokens"] == 120 and stats["bad_args"] == 0
    assert stats["backend"].startswith("server-chat:openai") and stats["sampling"]["mode"] == "chat"
