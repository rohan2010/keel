"""Chapter 3, P6: keel doctor. Mac checks, and server conformance against fake servers that each break one thing.
Raw-mode probes need the local model folder's tokenizer (skipped where it is missing)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from keel import cli
from keel.doctor import mac_checks, measured_rungs, memory_advice, record_path
from keel.specs import QWEN35_4B_8BIT as SPEC

MODEL = Path(SPEC.local_dir)
needs_model = pytest.mark.skipif(not (MODEL / "tokenizer_config.json").is_file(), reason="local model folder missing")
TC = "<tool_call>"


def call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"{TC}\n<function={name}>\n{body}</function>\n</tool_call>"


WANTS = "I'll do that.\n\n" + call("write_file", path="notes.txt", content="hello keel") + "\n" + call("list_files",
                                                                                                    path=".")
TEXT_ONLY = "I created notes.txt."


class Server:
    """A fake model server. What it does right or wrong:
    raw_keeps_calls  raw completion text keeps the tool-call text (mlx_lm.server: no)
    ids              raw completions return token ids with logprobs (mlx_lm.server: yes)
    chat_parses      chat replies carry structured tool_calls
    honours          the ban rungs it obeys
    cache, repeat    reports cached tokens on a repeat / same reply twice at temperature 0"""

    def __init__(self, raw_keeps_calls=True, ids=True, chat_parses=True,
                 honours=("logit_bias", "tool_choice", "stop"), cache=True, repeat=True):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(MODEL)) if (MODEL / "tokenizer_config.json").is_file() else None
        self.cfg = dict(raw_keeps_calls=raw_keeps_calls, ids=ids, chat_parses=chat_parses, honours=set(honours),
                        cache=cache, repeat=repeat)
        self.n, self.seen = 0, set()
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(obj).encode())

            def do_GET(self):
                self._send({"data": [{"id": "fake-model"}]})

            def do_POST(self):
                b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(srv.answer(self.path, b))

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def banned(self, b) -> bool:
        h = self.cfg["honours"]
        return (("logit_bias" in b and "logit_bias" in h) or (b.get("tool_choice") == "none" and "tool_choice" in h)
                or (TC in (b.get("stop") or []) and "stop" in h))

    def answer(self, path, b):
        self.n += 1
        key = json.dumps(b.get("prompt") or b.get("messages"), sort_keys=True)
        cached = 50 if (self.cfg["cache"] and key in self.seen) else 0
        self.seen.add(key)
        text = TEXT_ONLY if self.banned(b) else WANTS
        if not self.cfg["repeat"]:
            text = text.replace("I", f"I ({self.n})", 1)
        usage = {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": cached}}
        if path == "/v1/completions":
            choice = {"text": text if self.cfg["raw_keeps_calls"] else text.split(TC)[0], "finish_reason": "stop"}
            if b.get("logprobs") and self.cfg["ids"]:
                choice["logprobs"] = {"content": [{"id": i} for i in
                                                  self.tok.encode(text + "<|im_end|>", add_special_tokens=False)]}
            return {"choices": [choice], "usage": usage}
        prose = text.split(TC)[0].strip()
        msg = {"role": "assistant", "content": prose if self.cfg["chat_parses"] else text}
        if self.cfg["chat_parses"] and TC in text:
            msg["tool_calls"] = [
                {"id": "a", "type": "function", "function": {"name": "write_file", "arguments": json.dumps(
                    {"path": "notes.txt", "content": "hello keel"})}},
                {"id": "b", "type": "function", "function": {"name": "list_files", "arguments": '{"path": "."}'}}]
        if not self.cfg["chat_parses"]:
            msg["content"] = prose                       # no parser and no markup either: no calls at all
        return {"choices": [{"message": msg, "finish_reason": "stop"}], "usage": usage}

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    made = []

    def make(**kw):
        s = Server(**kw)
        made.append(s)
        return s
    yield make
    for s in made:
        s.close()


def doctor(*argv, state):
    with pytest.raises(SystemExit) as ex:
        cli.main(["doctor", "--state-dir", str(state), *argv])
    return ex.value.code


def record(state, url):
    return json.loads(record_path(state, url, "openai").read_text())


# ---- servers ---------------------------------------------------------------------------------------------------

@needs_model
def test_a_server_that_does_everything(server, tmp_path, capsys):
    s = server()
    assert doctor("--backend", "server", "--url", s.url, state=tmp_path) == 0
    rec = record(tmp_path, s.url)
    assert rec["modes"]["raw"]["tool_calls"] == "text" and rec["modes"]["raw"]["ban_rungs"] == ["logit_bias", "stop"]
    assert rec["modes"]["chat"]["ban_rungs"] == ["logit_bias", "tool_choice", "stop"]
    assert rec["modes"]["chat"]["history_round_trip"] and rec["modes"]["raw"]["prefix_cache"]
    assert rec["recommended"]["mode"] == "raw" and rec["requests"] <= 12
    out = capsys.readouterr().out
    assert "all required checks passed" in out and "FAIL" not in out.split("server")[1]


@needs_model
def test_an_mlx_lm_server_like_one(server, tmp_path, capsys):
    """Strips raw call text (token ids recover it) and ignores tool_choice: both found, neither fails the run."""
    s = server(raw_keeps_calls=False, honours=("logit_bias", "stop"))
    assert doctor("--backend", "server", "--url", s.url, state=tmp_path) == 0
    rec = record(tmp_path, s.url)
    assert rec["modes"]["raw"]["call_text_kept"] is False and rec["modes"]["raw"]["tool_calls"] == "token_ids"
    assert rec["modes"]["chat"]["ban_rungs"] == ["logit_bias", "stop"]                 # tool_choice measured out
    out = capsys.readouterr().out
    assert "removes tool-call text" in out and "ban tool_choice" in out and "ignored" in out


@needs_model
def test_no_mode_with_tool_calls_fails(server, tmp_path, capsys):
    s = server(raw_keeps_calls=False, ids=False, chat_parses=False)
    assert doctor("--backend", "server", "--url", s.url, state=tmp_path) == 1
    out = capsys.readouterr().out
    assert "FAILED: tool calls work" in out and "no token ids in the answer" in out


@needs_model
def test_unrepeatable_and_uncached_are_warnings(server, tmp_path, capsys):
    s = server(cache=False, repeat=False, honours=())
    assert doctor("--backend", "server", "--url", s.url, state=tmp_path) == 0
    rec = record(tmp_path, s.url)
    assert rec["modes"]["raw"]["greedy_repeatable"] is False and rec["modes"]["raw"]["prefix_cache"] is False
    assert rec["modes"]["raw"]["ban_rungs"] == [] and rec["modes"]["chat"]["ban_rungs"] == []
    assert capsys.readouterr().out.count("warn") >= 6


def test_server_down_fails_and_writes_nothing(tmp_path, capsys):
    assert doctor("--backend", "server", "--url", "http://127.0.0.1:9", state=tmp_path) == 1
    assert "FAILED: server reachable" in capsys.readouterr().out
    assert not (tmp_path / "servers").exists()


def test_chat_only_without_a_model_folder(server, tmp_path):
    s = server(honours=("tool_choice",))
    assert doctor("--backend", "server", "--url", s.url, "--model-dir", str(tmp_path / "none"), state=tmp_path) == 0
    rec = record(tmp_path, s.url)
    assert rec["modes"]["raw"] == {"usable": False, "reason": "no tokenizer"} and rec["recommended"]["mode"] == "chat"
    assert rec["modes"]["chat"]["ban_rungs"] == ["tool_choice"]          # logit_bias skipped: no token id


def test_ollama_needs_a_model_name(server, tmp_path, capsys):
    s = server()
    assert doctor("--backend", "server", "--flavour", "ollama", "--url", s.url, state=tmp_path) == 1
    assert "server-model" in capsys.readouterr().out


@needs_model
def test_runs_use_the_measured_rungs(server, tmp_path):
    import argparse

    from keel.backends.choose import add_backend_arguments, server_setup
    from keel.specs import SamplingSpec
    s = server(honours=("stop",))
    doctor("--backend", "server", "--url", s.url, state=tmp_path)
    assert measured_rungs(tmp_path, s.url, "openai", "chat")[0] == ("stop",)
    ap = argparse.ArgumentParser()
    add_backend_arguments(ap)
    ap.add_argument("--state-dir")
    a = ap.parse_args(["--backend", "server", "--url", s.url, "--mode", "chat", "--state-dir", str(tmp_path)])
    st = server_setup(a, SPEC, SamplingSpec())
    assert st.backend.ban_rung() == "stop" and st.capabilities_record()["ban_rungs_source"].startswith("keel doctor")
    a.ban_rungs = "none"                                                  # an explicit flag still wins
    assert server_setup(a, SPEC, SamplingSpec()).backend.ban_rung() == "none"
    a.ban_rungs, a.url = None, "http://127.0.0.1:1"                       # a server never probed: defaults
    st = server_setup(a, SPEC, SamplingSpec())
    assert st.capabilities_record()["ban_rungs_source"] == "default for this server type"


# ---- the Mac -------------------------------------------------------------------------------------------------------

def test_memory_advice_matches_the_measured_16gb_mac():
    assert memory_advice(16)["context_budget"] == 32000 and memory_advice(16)["max_context"] == 16000
    assert memory_advice(8)["context_budget"] < 16000


def test_mac_checks_require_mlx_things_only_for_mlx(tmp_path, monkeypatch):
    import keel.modelstore as ms
    monkeypatch.setattr(ms, "find", lambda spec: None)          # no model anywhere
    empty = tmp_path / "models"
    empty.mkdir()
    need = {c.name: c for c in mac_checks(tmp_path / "state", empty, need_mlx=True)}
    assert need["model"].status == "FAIL" and need["model"].required
    assert "keel download" in need["model"].detail
    free = {c.name: c for c in mac_checks(tmp_path / "state", empty, need_mlx=False)}
    assert free["model"].status == "warn" and not free["model"].required
    assert free["state folder"].status == "ok" and list((tmp_path / "state").iterdir()) == []   # probe file removed


def test_plain_doctor_passes_on_this_mac(tmp_path, capsys):
    import platform
    if platform.machine() != "arm64":
        pytest.skip("not Apple Silicon")
    assert doctor(state=tmp_path) == 0
    assert "keel doctor: this Mac" in capsys.readouterr().out


def test_missing_bfcl_is_info_not_a_warning(tmp_path, monkeypatch):
    """bfcl-eval is optional (only the BFCL benchmark needs it): its absence must not look like a problem."""
    import keel.doctor as D
    monkeypatch.setattr(D, "_version", lambda pkg: None if pkg == "bfcl-eval" else "1.0")
    c = {x.name: x for x in mac_checks(tmp_path / "state", tmp_path, need_mlx=False)}["bfcl-eval"]
    assert c.status == "info" and not c.required and "BFCL benchmark" in c.detail
