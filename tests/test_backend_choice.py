"""Chapter 3, P5: choosing a backend end to end (`--backend` on run / bfcl-mt / acp), against a fake server.

No model is run: the fake server answers, and raw mode only needs the local model folder's tokenizer (skipped where
that folder is missing)."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from test_server_backend import Fake, call, encode, openai_ids  # noqa: F401
from test_server_chat import chat_answer, tc

import keel.runlog as runlog
from keel import cli
from keel.backends.choose import add_backend_arguments, server_setup
from keel.specs import QWEN35_4B_8BIT as SPEC
from keel.specs import SamplingSpec

MODEL = Path(SPEC.local_dir)
needs_model = pytest.mark.skipif(not (MODEL / "tokenizer_config.json").is_file(), reason="local model folder missing")


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


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """Run folders go to a temp folder, not experiments/."""
    monkeypatch.setattr(runlog, "EXPERIMENTS", tmp_path / "experiments")
    return tmp_path / "experiments"


def args(*argv, extra=()):
    ap = argparse.ArgumentParser()
    add_backend_arguments(ap, extra)
    ap.add_argument("--state-saves", action="store_true")
    return ap.parse_args(list(argv))


# ---- the choice ----------------------------------------------------------------------------------------------------

@needs_model
def test_mlx_server_shortcut_is_raw_openai_with_token_ids():
    st = server_setup(args("--backend", "mlx-server"), SPEC, SamplingSpec())
    b = st.backend
    assert (b.flavour, b.url, b.mode, b.token_ids) == ("openai", "http://127.0.0.1:8080", "raw", True)
    assert b.tool_call_id == 248058 and b.ban_rung() == "logit_bias" and b.count_tokens is not None
    rec = st.capabilities_record()
    assert rec["kind"] == "server" and rec["capabilities"]["validated"] is False
    assert rec["trim"]["method"] == "tokenizer.json" and rec["noise"].startswith("repeats")
    assert b.decode(encode_ids(st.tok, "hi <tool_call>")) == "hi <tool_call>"


@needs_model
def test_raw_mode_finds_the_model_folder_from_any_folder(tmp_path, monkeypatch):
    """P9 (4) bug: `keel run --backend mlx-server` from a user's folder looked for models/ relative to that folder and
    quit at start. The default model folder is relative to the keel root; the run records it as given (no abs path)."""
    monkeypatch.chdir(tmp_path)
    st = server_setup(args("--backend", "mlx-server"), SPEC, SamplingSpec())
    assert st.tok is not None and st.backend.mode == "raw"
    assert st.capabilities_record()["model_dir"] == SPEC.local_dir


def encode_ids(tok, text):
    return tok.encode(text, add_special_tokens=False)


@needs_model
def test_ollama_shortcut_and_flags():
    st = server_setup(args("--backend", "ollama", "--server-model", "qwen3.5:4b"), SPEC, SamplingSpec())
    assert (st.backend.flavour, st.backend.url, st.backend.model) == ("ollama", "http://127.0.0.1:11434", "qwen3.5:4b")
    assert st.backend.ban_rung() == "stop" and st.backend.token_ids is False
    st = server_setup(args("--backend", "server", "--flavour", "llama", "--url", "http://h:1/", "--ban-rungs", "none"),
                      SPEC, SamplingSpec())
    assert st.backend.url == "http://h:1" and st.backend.ban_rung() == "none"


def test_refusals_say_what_is_wrong(tmp_path):
    with pytest.raises(SystemExit, match="server-model"):
        server_setup(args("--backend", "ollama"), SPEC, SamplingSpec())
    with pytest.raises(SystemExit, match="in-process MLX only"):
        server_setup(args("--backend", "mlx-server", "--state-saves"), SPEC, SamplingSpec())
    with pytest.raises(SystemExit, match="tokenizer folder"):
        server_setup(args("--backend", "mlx-server", "--model-dir", str(tmp_path)), SPEC, SamplingSpec())
    with pytest.raises(SystemExit, match="use --backend server"):
        server_setup(args("--backend", "mlx-server", "--flavour", "ollama"), SPEC, SamplingSpec())
    with pytest.raises(SystemExit, match="tool_choice"):
        server_setup(args("--backend", "server", "--mode", "chat", "--flavour", "ollama", "--model-dir",
                          str(tmp_path), "--ban-rungs", "tool_choice"), SPEC, SamplingSpec())


def test_chat_mode_without_a_model_folder_counts_approximately(tmp_path):
    st = server_setup(args("--backend", "server", "--mode", "chat", "--model-dir", str(tmp_path)), SPEC,
                      SamplingSpec())
    assert st.tok is None and st.backend.tool_call_id is None and st.backend.ban_rung() == "tool_choice"
    assert st.trim_record["approximate"] is True and st.capabilities_record()["model_dir"] is None


# ---- keel-core run / bfcl-mt on a server ---------------------------------------------------------------------------

def test_agent_run_on_a_chat_server_writes_capabilities(fake, runs, tmp_path):
    def respond(p, b, n):
        has_tool = any(m["role"] == "tool" for m in b["messages"])
        return chat_answer("All done." if has_tool else "", [] if has_tool else [tc("list_files", path=".")])
    f = fake(respond)
    cli.main(["run", "--name", "p5", "--split", "dev", "--tasks", "C1", "F3", "--per-task", "1",
              "--backend", "server", "--mode", "chat", "--url", f.url, "--model-dir", str(tmp_path / "none"),
              "--batch-size", "2"])
    (run,) = list(runs.iterdir())
    caps = json.loads((run / "capabilities.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    tim = json.loads((run / "timings.json").read_text())
    assert caps["backend"].startswith("server-chat:openai") and caps["capabilities"]["validated"] is False
    assert cfg["backend"] == "server" and cfg["server"]["mode"] == "chat"
    assert tim["backend_stats"]["requests"] == 4 and tim["backend_stats"]["errors"] == 0
    outs = [json.loads(line) for line in (run / "outputs.jsonl").read_text().splitlines()]
    assert sorted(o["id"] for o in outs) == ["C1_0", "F3_0"] and all(o["stop"] == "answer" for o in outs)


@needs_model
def test_agent_run_raw_mode_sends_keels_rendering(fake, runs):
    f = fake(lambda p, b, n: openai_ids_from_text("Nothing to do."))
    cli.main(["run", "--name", "p5raw", "--split", "dev", "--tasks", "F3", "--per-task", "1",
              "--backend", "mlx-server", "--url", f.url])
    path, body = f.requests[0]
    assert path == "/v1/completions" and body["logprobs"] is True
    assert body["prompt"].startswith("<|im_start|>system") and body["prompt"].endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")                       # thinking off, as in-process


def openai_ids_from_text(text):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    ids = tok.encode(text + "<|im_end|>", add_special_tokens=False)
    return {"choices": [{"text": text, "finish_reason": "stop", "logprobs": {"content": [{"id": i} for i in ids]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": len(ids)}}


@needs_model
def test_bfcl_mt_on_a_raw_server(fake, runs):
    pytest.importorskip("bfcl_eval")
    f = fake(lambda p, b, n: openai_ids_from_text("I can't do that."))
    cli.main(["bfcl-mt", "--name", "p5bfcl", "--split", "dev", "--categories", "base", "--limit", "1",
              "--backend", "mlx-server", "--url", f.url])
    (run,) = list(runs.iterdir())
    assert json.loads((run / "capabilities.json").read_text())["mode"] == "raw"
    outs = [json.loads(line) for line in (run / "outputs.jsonl").read_text().splitlines()]
    assert len(outs) == 1 and outs[0]["turns_run"] == outs[0]["turns"]       # every turn ran through the server
    assert "<|im_start|>" in f.requests[0][1]["prompt"] and "tools" in f.requests[0][1]["prompt"].lower()


# ---- ACP ---------------------------------------------------------------------------------------------------------

acp = pytest.importorskip("acp")
from acp import schema as s  # noqa: E402
from acp.exceptions import RequestError  # noqa: E402
from test_acp_server import Recorder  # noqa: E402


async def _acp(state, folder, backend_args, prompts, before_prompts=None):
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    async with acp.spawn_agent_process(Recorder(), sys.executable, "-m", "keel.cli", "acp", *backend_args,
                                       "--state-dir", str(state), env=env) as (conn, proc):
        await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)
        new = await conn.new_session(cwd=str(folder))
        if before_prompts:
            before_prompts()
        out = []
        for p in prompts:
            try:
                r = await conn.prompt(session_id=new.session_id, prompt=[s.TextContentBlock(type="text", text=p)])
                out.append(r.stop_reason)
            except RequestError as err:
                out.append(err)
        return new, out


def test_acp_server_down_at_start_refuses_the_session_with_the_fix(tmp_path):
    """Ch7 P6: nothing listening at --url -> the session is refused at start with what to do (before, it opened and
    the first prompt failed)."""
    folder = tmp_path / "work"
    folder.mkdir()
    with pytest.raises(RequestError) as err:
        asyncio.run(_acp(tmp_path / "state", folder,
                         ["--backend", "server", "--mode", "chat", "--url", "http://127.0.0.1:9",
                          "--model-dir", str(tmp_path / "none")], ["hello"]))
    assert "no model server answers at http://127.0.0.1:9" in str(err.value) and "--url" in str(err.value)
    assert err.value.data == {"reason": "backend_refused"}


def test_acp_server_dying_mid_session_is_a_visible_error_and_the_prompt_is_taken_back(fake, tmp_path):
    folder = tmp_path / "work"
    folder.mkdir()
    f = fake(lambda p, b, n: as_stream(chat_answer("unused")))
    new, out = asyncio.run(_acp(tmp_path / "state", folder,
                                ["--backend", "server", "--mode", "chat", "--url", f.url,
                                 "--model-dir", str(tmp_path / "none")], ["hello", "again"], before_prompts=f.close))
    meta = new.field_meta["keel"]
    assert meta["backend"].startswith("server chat · 127.0.0.1:") and meta["capabilities"]["validated"] is False
    for err in out:
        assert isinstance(err, RequestError) and err.code == -32603 and "is the server running" in str(err)
        assert err.data == {"reason": "backend_error"}
    sess = tmp_path / "state" / "sessions" / new.session_id
    rec = json.loads((sess / "transcript.json").read_text())
    assert rec["stop"] == "backend_error" and "is the server running" in rec["error"]
    assert [m["role"] for m in rec["messages"]] == ["system"]                # both prompts taken back
    assert json.loads((sess / "capabilities.json").read_text())["kind"] == "server"


def as_stream(ans):
    """A whole chat answer as SSE chunks (the fake streams a list)."""
    c = ans["choices"][0]
    m = c["message"]
    out = [{"choices": [{"delta": {"content": m["content"]}}]}]
    for i, t in enumerate(m.get("tool_calls") or []):
        out.append({"choices": [{"delta": {"tool_calls": [{"index": i, **t}]}}]})
    return out + [{"choices": [{"delta": {}, "finish_reason": c["finish_reason"]}]}, {"choices": [], "usage": ans["usage"]}]


def test_acp_chat_server_session_works(fake, tmp_path):
    folder = tmp_path / "work"
    folder.mkdir()
    (folder / "a.txt").write_text("hi\n")
    def respond(p, b, n):
        if any(m["role"] == "tool" for m in b["messages"][-2:]):
            return as_stream(chat_answer("There is one file, a.txt."))
        return as_stream(chat_answer("Looking.", [tc("list_files", path=".")]))
    f = fake(respond)
    new, out = asyncio.run(_acp(tmp_path / "state", folder, ["--backend", "server", "--mode", "chat", "--url", f.url,
                                                             "--model-dir", str(tmp_path / "none")], ["what's here?"]))
    assert out == ["end_turn"] and f.requests[0][1]["stream"] is True        # the TUI gets live text in chat mode
