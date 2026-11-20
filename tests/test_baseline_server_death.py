"""ch6dev 2026-11-19: mlx_lm.server ran out of Metal memory, its generation thread died, every later request got a
404, and the baseline runner scored 42 dead-server episodes as goose failures (5/50). Now the run stops and is marked
aborted instead."""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from keel import runlog
from keel.commands import run_baseline as RB


def _serve(status: int):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(status)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.mark.parametrize("status,ok", [(200, True), (404, False)])
def test_server_generates(monkeypatch, status, ok):
    srv = _serve(status)
    monkeypatch.setattr(RB.B, "SERVER", f"http://127.0.0.1:{srv.server_port}")
    try:
        assert RB._server_generates("m") is ok
    finally:
        srv.shutdown()


def _args(tmp_path):
    return argparse.Namespace(agent="goose", name="t_death", tasks=["A3"], split="dev", purpose=None, per_task=6,
                              first_seed=0, concurrency=1, max_turns=30, timeout=900)


def _fake_episode(fail_from: int):
    calls = []

    def run_episode(agent, it, out_dir, **kw):
        calls.append(it.id)
        failed = len(calls) > fail_from
        return {"id": it.id, "correct": not failed, "seconds": 1.0, "steps": 1, "labels": [],
                "stop": "exit 1" if failed else "answer", "output_parsed": True}
    return run_episode, calls


def test_run_aborts_when_server_dies(monkeypatch, tmp_path):
    ep, calls = _fake_episode(fail_from=2)
    monkeypatch.setattr(RB, "_server_model", lambda: "m")
    monkeypatch.setattr(RB.B, "run_episode", ep)
    monkeypatch.setattr(RB, "_server_generates", lambda m: False)
    monkeypatch.setattr(RB, "ledger", lambda *a, **k: None)
    monkeypatch.setattr(runlog, "EXPERIMENTS", tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RB.ServerDied):
        RB.run(_args(tmp_path))
    assert len(calls) == 3                                    # stopped at the first dead-server episode
    run_dir = sorted(tmp_path.rglob("metrics.json"))[-1].parent
    assert json.loads((run_dir / "metrics.json").read_text())["aborted"]
    lines = (run_dir / "outputs.jsonl").read_text().splitlines()
    assert len(lines) == 2                                    # the dead-server episode is not recorded as a result


def test_real_failure_with_healthy_server_is_kept(monkeypatch, tmp_path):
    ep, calls = _fake_episode(fail_from=2)
    monkeypatch.setattr(RB, "_server_model", lambda: "m")
    monkeypatch.setattr(RB.B, "run_episode", ep)
    monkeypatch.setattr(RB, "_server_generates", lambda m: True)
    monkeypatch.setattr(RB, "ledger", lambda *a, **k: None)
    monkeypatch.setattr(runlog, "EXPERIMENTS", tmp_path)
    monkeypatch.chdir(tmp_path)
    RB.run(_args(tmp_path))
    assert len(calls) == 6
