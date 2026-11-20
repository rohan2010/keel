"""Chapter 6 baselines: off-the-shelf agents (no sandbox, no delete guard of their own) run under this profile.
Done-when: writes outside the task copy, reads of the home folder and any internet access are refused by the OS,
while localhost (the model server, an agent's own local server) works. The first draft allowed the internet through
a too-broad localhost rule (2026-11-19); this test keeps that from coming back."""

import subprocess
from pathlib import Path

import pytest

from keel.agent.sandbox import PATH, Sandbox
from keel.evals.baselines import SEND_SHIMS, SHIM, _profile_with_server


@pytest.fixture
def run(tmp_path: Path):
    sb = Sandbox(tmp_path / "ep").create()
    home = sb.root / ".home"
    home.mkdir()
    prof = _profile_with_server(sb.root, [sb.work, sb.tmp, home], (str(sb.work),), 8080)

    def go(cmd: str):
        p = subprocess.run(["sandbox-exec", "-p", prof, "/bin/bash", "-c", cmd], cwd=sb.work, capture_output=True,
                           text=True, env={"PATH": f"{sb.bin}:{PATH}", "HOME": str(home)}, timeout=30)
        return p.returncode, p.stdout + p.stderr
    go.sb, go.home = sb, home
    return go


def test_writes_only_inside_the_task_copy(run):
    assert run("echo ok > inside.txt && cat inside.txt")[1].strip() == "ok"
    target = Path.home() / "Desktop" / "keel_sandbox_probe_test.txt"
    code, out = run(f"echo x > '{target}'")
    assert code != 0 and "not permitted" in out and not target.exists()


def test_home_folder_is_unreadable(run):
    code, out = run(f"ls '{Path.home() / 'Documents'}'")
    assert "not permitted" in out


def test_no_internet(run):
    code, out = run("curl -s -m 5 -o /dev/null -w '%{http_code}' https://example.com")
    assert out.strip() in ("000", "") and code != 0
    code, out = run("curl -s -m 5 -o /dev/null -w '%{http_code}' http://1.1.1.1")
    assert out.strip() in ("000", "") and code != 0


def test_localhost_works(run):
    code, out = run("python3 -c \"import socket;s=socket.socket();s.bind(('127.0.0.1',0));s.listen();"
                    "socket.create_connection(('127.0.0.1',s.getsockname()[1]));print('ok')\"")
    assert out.strip() == "ok"


def test_mail_stand_ins_log_and_refuse(run):
    log = run.home / "sends.log"
    for prog in SEND_SHIMS:
        p = run.sb.bin / prog
        p.write_text(SHIM.format(log=log))
        p.chmod(0o755)
    code, out = run("echo hi | mail -s Report boss@example.com")
    assert code != 0 and "nothing was sent" in out
    assert log.read_text().strip() == "mail -s Report boss@example.com"
