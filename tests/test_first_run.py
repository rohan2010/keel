"""Ch7 P6: first-run problems give one line that says what to do (real `keel` binary, line mode). Skipped if the
binary is not built (`cargo build`)."""

import os
import socket
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "debug" / "keel"

pytestmark = pytest.mark.skipif(not BIN.exists(), reason="keel binary not built (cargo build)")


def keel(tmp_path, *args, env=None):
    e = {**os.environ, **(env or {})}
    r = subprocess.run([str(BIN), *args], capture_output=True, text=True, timeout=120, env=e,
                       stdin=subprocess.DEVNULL, cwd=tmp_path)
    return r.returncode, (r.stdout + r.stderr).strip()


@pytest.fixture
def demo(tmp_path):
    d = tmp_path / "demo"
    d.mkdir()
    (d / "a.txt").write_text("hi")
    return d


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_no_model_says_keel_download(tmp_path, demo):
    empty = tmp_path / "models"
    empty.mkdir()
    code, out = keel(tmp_path, "run", "--plain", "--state-dir", str(tmp_path / "s"), "--dir", str(demo), "list",
                     env={"KEEL_MODELS": str(empty), "HF_HUB_CACHE": str(tmp_path / "hf")})
    assert code != 0 and out.startswith("keel:") and "keel download" in out and "Repo id" not in out


def test_too_big_folder_says_pick_a_smaller_one(tmp_path):
    big = tmp_path / "big"
    big.mkdir()
    for i in range(5001):
        (big / f"f{i}").touch()
    code, out = keel(tmp_path, "run", "--plain", "--backend", "scripted", "--state-dir", str(tmp_path / "s"),
                     "--dir", str(big), "list")
    assert code != 0 and "too big" in out and "smaller folder" in out and "{" not in out


def test_missing_server_fails_at_start_with_the_fix(tmp_path, demo):
    url = f"http://127.0.0.1:{_free_port()}"
    code, out = keel(tmp_path, "run", "--plain", "--backend", "server", "--mode", "chat", "--url", url,
                     "--state-dir", str(tmp_path / "s"), "--dir", str(demo), "list")
    assert code != 0 and f"no model server answers at {url}" in out and "--url" in out


def test_unknown_backend_lists_the_real_ones(tmp_path, demo):
    code, out = keel(tmp_path, "run", "--plain", "--backend", "foo", "--dir", str(demo), "list")
    assert code != 0 and "possible values: mlx, mlx-server, ollama, server, scripted" in out


def test_line_mode_without_a_task_says_how(tmp_path, demo):
    code, out = keel(tmp_path, "run", "--plain", "--dir", str(demo))
    assert code != 0 and "line mode needs a task" in out


def test_missing_folder_is_one_line(tmp_path):
    code, out = keel(tmp_path, "run", "--plain", "--dir", str(tmp_path / "nope"), "x")
    assert code != 0 and out == f"keel: no such folder: {tmp_path / 'nope'}"


def test_bare_keel_shows_the_mascot_and_where_to_start(tmp_path):
    code, out = keel(tmp_path, env={"NO_COLOR": "1"})
    assert code == 0 and "▄█████▄" in out and "keel download" in out and "keel run --dir" in out
    assert "\x1b[" not in out                                # no colour codes when not a terminal / NO_COLOR
