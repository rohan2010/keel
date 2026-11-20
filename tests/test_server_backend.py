"""Chapter 3, P1: the raw-mode server backend against a fake HTTP server (stdlib, no model, no MLX)."""

import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from keel.agent.loop import run_episodes_on
from keel.backends.server import BackendError, ServerBackend, _finish, _strip_end
from keel.core.backend import RunLimits
from keel.core.events import ListSink
from keel.evals import tasks as T
from keel.specs import SamplingSpec

TC_ID = 248058


def call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def encode(text):                       # fake tokenizer: one id per character
    return [ord(c) for c in text]


def decode(ids):
    return "".join(map(chr, ids))


def render(e):                          # fake chat template: the server only needs some text
    return json.dumps(e.messages)


class Fake:
    """A server that answers with `respond(path, body, n)`: a dict (whole answer), a list (streamed chunks),
    or an int (HTTP error status). Records every request and the peak number in flight."""

    def __init__(self, respond, stream_format="sse", delay=0.0):
        self.respond, self.stream_format, self.delay = respond, stream_format, delay
        self.requests, self.inflight, self.peak = [], 0, 0
        self.lock = threading.Lock()
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with fake.lock:
                    fake.requests.append((self.path, body))
                    n = len(fake.requests)
                    fake.inflight += 1
                    fake.peak = max(fake.peak, fake.inflight)
                try:
                    time.sleep(fake.delay)
                    out = fake.respond(self.path, body, n)
                    if isinstance(out, int):
                        self.send_response(out)
                        self.end_headers()
                        self.wfile.write(b'{"error": "boom"}')
                    elif isinstance(out, list):
                        self.send_response(200)
                        self.end_headers()
                        if fake.stream_format == "sse":
                            self.wfile.write(b": keepalive 1/1\n\n")
                            for ch in out:
                                self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode())
                            self.wfile.write(b"data: [DONE]\n\n")
                        else:
                            for ch in out:
                                self.wfile.write((json.dumps(ch) + "\n").encode())
                    else:
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(json.dumps(out).encode())
                finally:
                    with fake.lock:
                        fake.inflight -= 1

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


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


def openai_ids(text, finish="stop", prompt_tokens=100, cached=0, end="<|im_end|>"):
    """What mlx_lm.server sends in token-id mode: `text` without the tool block, every id in logprobs."""
    full = text + end
    return {"choices": [{"text": text.split("<tool_call>")[0], "finish_reason": "tool_calls" if "<tool_call>" in text
                         else finish, "logprobs": {"content": [{"id": i, "logprob": -0.1} for i in encode(full)]}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(full),
                      "prompt_tokens_details": {"cached_tokens": cached}}}


def test_openai_token_ids_recover_tool_calls_through_the_agent_loop(fake, tmp_path):
    replies = ["Let me look.\n" + call("list_files", path="."), "I looked at the folder."]
    f = fake(lambda p, b, n: openai_ids(replies[n - 1], prompt_tokens=50 * n, cached=40 * (n - 1)))
    be = ServerBackend(f.url, "openai", decode=decode)
    res = run_episodes_on(be, [T.build("C1", 0)], tmp_path, prompt_fn=render)
    assert res[0]["steps"] == 2 and res[0]["stop"] == "answer" and res[0]["answer"] == "I looked at the folder."
    path, body = f.requests[0]
    assert path == "/v1/completions" and body["logprobs"] is True and body["stream"] is False
    assert isinstance(body["prompt"], str) and body["top_k"] == 20 and "seed" not in body
    assert json.loads(f.requests[1][1]["prompt"])[-1]["role"] == "tool"       # the tool result reached the server


def test_run_stats_count_tokens_and_cache(fake):
    f = fake(lambda p, b, n: openai_ids("Done here.", prompt_tokens=100, cached=90))
    be = ServerBackend(f.url, "openai", decode=decode)
    text, fr, st = be.complete("hi")
    assert text == "Done here." and fr == "stop" and st.prompt_tokens == 100 and st.reused_tokens == 90


def test_openai_streaming_reaches_token_fn(fake, tmp_path):
    chunks = [{"choices": [{"text": t, "finish_reason": None}]} for t in ["Hel", "lo ", "there."]]
    chunks += [{"choices": [{"text": "", "finish_reason": "stop"}]},
               {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3,
                                         "prompt_tokens_details": {"cached_tokens": 6}}}]
    f = fake(lambda p, b, n: chunks)
    be = ServerBackend(f.url, "openai", stream=True)                          # no decode: plain text mode
    sink = ListSink()
    res = run_episodes_on(be, [T.build("F3", 0)], tmp_path, prompt_fn=render, events=sink, stream=True)
    streamed = "".join(e.data["text"] for e in sink.events if e.kind == "text_delta")
    assert streamed == "Hello there." and res[0]["answer"] == "Hello there."
    assert f.requests[0][1]["stream_options"] == {"include_usage": True} and "logprobs" not in f.requests[0][1]


def test_token_id_mode_is_never_streamed(fake):
    f = fake(lambda p, b, n: openai_ids("Hi."))
    got = []
    r = ServerBackend(f.url, "openai", decode=decode, stream=True).request("x", on_text=got.append)
    assert f.requests[0][1]["stream"] is False and got == ["Hi."] and r.text == "Hi."


def test_ollama_raw_generate_whole_and_streamed(fake):
    whole = {"response": "Fine.", "done": True, "done_reason": "stop", "prompt_eval_count": 12, "eval_count": 2}
    f = fake(lambda p, b, n: whole)
    be = ServerBackend(f.url, "ollama", model="qwen3.5:4b")
    text, fr, st = be.complete("prompt text")
    path, body = f.requests[0]
    assert path == "/api/generate" and body["raw"] is True and body["model"] == "qwen3.5:4b"
    assert body["options"]["top_k"] == 20 and body["options"]["num_predict"] == 8192 and "logit_bias" not in body
    assert (text, fr, st.prompt_tokens) == ("Fine.", "stop", 12)

    chunks = [{"response": "A", "done": False}, {"response": "B", "done": False},
              {"response": "", "done": True, "done_reason": "length", "prompt_eval_count": 5, "eval_count": 2}]
    f2 = fake(lambda p, b, n: chunks, stream_format="ndjson")
    got = []
    r = ServerBackend(f2.url, "ollama", stream=True).request("p", on_text=got.append)
    assert got == ["A", "B"] and r.text == "AB" and r.finish == "length" and r.output_tokens == 2


def test_llama_completion_and_logit_bias_ban(fake):
    f = fake(lambda p, b, n: {"content": "Just text.", "stop_type": "eos", "tokens_evaluated": 9,
                              "tokens_predicted": 3, "timings": {"cache_n": 8}})
    be = ServerBackend(f.url, "llama", tool_call_id=TC_ID)
    r = be.request("p", forbid_tools=True)
    path, body = f.requests[0]
    assert path == "/completion" and body["cache_prompt"] is True and body["n_predict"] == 8192
    assert body["logit_bias"] == [[TC_ID, False]] and r.ban_method == "logit_bias" and r.cached_tokens == 8


def test_ban_ladder_logit_bias_then_stop_then_cut(fake):
    f = fake(lambda p, b, n: openai_ids("Text only."))
    ServerBackend(f.url, "openai", decode=decode, tool_call_id=TC_ID).request("p", forbid_tools=True)
    assert f.requests[-1][1]["logit_bias"] == {str(TC_ID): -1e9} and "stop" not in f.requests[-1][1]

    f2 = fake(lambda p, b, n: {"response": "ok", "done": True})
    ServerBackend(f2.url, "ollama").request("p", forbid_tools=True)          # no logit_bias on Ollama: stop word
    assert f2.requests[-1][1]["options"]["stop"] == ["<tool_call>"]

    f3 = fake(lambda p, b, n: openai_ids("I'll stop.\n" + call("list_files")))  # server ignores the ban
    r = ServerBackend(f3.url, "openai", decode=decode, tool_call_id=TC_ID).request("p", forbid_tools=True)
    assert r.text == "I'll stop." and r.ban_method == "logit_bias" and r.ban_cut


def test_loop_breaker_ban_through_the_agent_loop(fake, tmp_path):
    ls = call("list_files", path=".")
    replies = [ls, ls, ls, "Still looking.\n" + ls]
    f = fake(lambda p, b, n: openai_ids(replies[min(n, 4) - 1]))
    be = ServerBackend(f.url, "openai", decode=decode, tool_call_id=TC_ID)
    res = run_episodes_on(be, [T.build("C1", 0)], tmp_path, prompt_fn=render, loop_break=True)
    assert any(n.startswith("loop@") for n in res[0]["notes"])
    assert res[0]["stop"] == "answer" and res[0]["answer"] == "Still looking."
    assert "logit_bias" in f.requests[3][1] and "logit_bias" not in f.requests[0][1]


def test_server_down_stops_with_backend_error(tmp_path):
    be = ServerBackend("http://127.0.0.1:9", "openai", timeout=2)            # nothing listens on port 9
    res = run_episodes_on(be, [T.build("F3", 0)], tmp_path, prompt_fn=render)
    assert res[0]["stop"] == "backend_error"
    with pytest.raises(BackendError, match="is the server running"):
        be.complete("hi")


def test_http_error_and_missing_ids(fake):
    f = fake(lambda p, b, n: 500)
    with pytest.raises(BackendError, match="HTTP 500.*boom"):
        ServerBackend(f.url, "openai").complete("hi")
    f2 = fake(lambda p, b, n: {"choices": [{"text": "no ids", "finish_reason": "stop"}], "usage": {}})
    with pytest.raises(BackendError, match="no token ids"):
        ServerBackend(f2.url, "openai", decode=decode).complete("hi")


def test_context_limit_from_usage_drops_the_reply(fake, tmp_path):
    f = fake(lambda p, b, n: openai_ids("Should not be used.", prompt_tokens=20000))
    be = ServerBackend(f.url, "openai", decode=decode)
    res = run_episodes_on(be, [T.build("F3", 0)], tmp_path, prompt_fn=render, limits=RunLimits(max_context=16000))
    assert res[0]["stop"] == "context_limit" and res[0]["steps"] == 0


def test_context_limit_checked_before_sending_with_a_counter(fake, tmp_path):
    f = fake(lambda p, b, n: openai_ids("x"))
    be = ServerBackend(f.url, "openai", decode=decode, count_tokens=lambda s: 10**6)
    res = run_episodes_on(be, [T.build("F3", 0)], tmp_path, prompt_fn=render)
    assert res[0]["stop"] == "context_limit" and f.requests == []


def test_episodes_run_concurrently_up_to_the_slots(fake, tmp_path):
    f = fake(lambda p, b, n: openai_ids("No."), delay=0.3)
    be = ServerBackend(f.url, "openai", decode=decode)
    items = [T.build("F3", i) for i in range(3)]
    res = run_episodes_on(be, items, tmp_path, prompt_fn=render, limits=RunLimits(batch_size=3))
    assert len(res) == 3 and f.peak == 3
    f.peak = 0
    run_episodes_on(be, items, tmp_path / "b", prompt_fn=render, limits=RunLimits(batch_size=1))
    assert f.peak == 1


def test_seeds_only_when_asked_and_stable(fake):
    f = fake(lambda p, b, n: openai_ids("ok"))
    be = ServerBackend(f.url, "openai", decode=decode, seeded=True, sampling=SamplingSpec(seed=3))
    be.request("p", label="A1_0", step=0)
    be.request("p", label="A1_0", step=0)
    be.request("p", label="A1_0", step=1)
    seeds = [b["seed"] for _, b in f.requests]
    assert seeds[0] == seeds[1] != seeds[2]


def test_sampling_record_says_what_was_sent():
    rec = ServerBackend("http://x", "openai").sampling_record()
    assert rec["dropped"] == [] and rec["approximate"] == ["presence_penalty"] and rec["seed"] == "none"
    assert ServerBackend("http://x", "openai", sampling=SamplingSpec(presence_penalty=0)).sampling_record()[
        "approximate"] == []


def test_config_errors_and_helpers():
    with pytest.raises(ValueError, match="flavour"):
        ServerBackend("http://x", "vllm-magic")
    with pytest.raises(ValueError, match="decode"):
        ServerBackend("http://x", "openai", token_ids=True)
    with pytest.raises(ValueError, match="openai"):
        ServerBackend("http://x", "ollama", token_ids=True, decode=decode)
    assert ServerBackend(flavour="ollama").url == "http://127.0.0.1:11434"
    assert ServerBackend(flavour="openai", decode=decode).token_ids is True
    assert ServerBackend(flavour="openai").token_ids is False
    assert [_finish(x) for x in ("stop", "tool_calls", "length", "limit", None)] == ["stop", "stop", "length",
                                                                                     "length", "stop"]
    assert _strip_end("hi<|im_end|>") == "hi" and _strip_end("hi") == "hi"
    caps = ServerBackend("http://x", "ollama", tool_call_id=TC_ID).capabilities()
    assert caps.logit_bias is False and caps.forbid_tools_next_turn and not caps.validated


def test_server_backend_imports_without_mlx(tmp_path):
    code = """
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('mlx', 'mlx_lm', 'numpy', 'torch'):
            raise ImportError('blocked: ' + name)
sys.meta_path.insert(0, Block())
from keel.backends.server import ServerBackend
print(ServerBackend(flavour='ollama').name)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "server-raw:ollama" in r.stdout
