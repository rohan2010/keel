"""Packaging (Ch7 P4): keel-core starts without the optional BFCL checker; the `keel` launcher finds its binary and
this environment's keel-core."""

import subprocess
import sys

import pytest

BLOCK_BFCL = "import sys; sys.modules['bfcl_eval'] = None; from keel import cli; cli.main(sys.argv[1:])"


def _core(*args):
    return subprocess.run([sys.executable, "-c", BLOCK_BFCL, *args], capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("cmd", ["doctor", "download", "run", "acp", "compare", "eval", "prereg", "backups"])
def test_core_commands_start_without_bfcl(cmd):
    r = _core(cmd, "--help")
    assert r.returncode == 0, r.stderr[-1500:]


def test_bfcl_commands_name_the_fix_without_bfcl():
    r = _core("bfcl-mt", "--help")
    assert r.returncode != 0 and "keel-core install-bfcl" in (r.stdout + r.stderr)


def test_install_bfcl_is_no_deps_and_pinned(capsys):
    from argparse import Namespace

    from keel.commands import run_install_bfcl as I
    assert I.run(Namespace(print=True)) == 0
    out = capsys.readouterr().out
    assert "--no-deps" in out and I.BFCL_PIN in out and sys.executable in out


def test_launcher_points_the_binary_at_this_environments_core(monkeypatch, tmp_path):
    from keel import _keel_bin as K
    fake = tmp_path / "keel"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setattr(K, "binary", lambda: fake)
    seen = {}
    monkeypatch.setattr(K.os, "execve", lambda path, argv, env: seen.update(path=path, argv=argv, env=env))
    monkeypatch.delenv("KEEL_CORE", raising=False)
    monkeypatch.setattr(sys, "argv", ["keel", "doctor"])
    K.main()
    assert seen["path"] == str(fake) and seen["argv"][1:] == ["doctor"]
    assert seen["env"]["KEEL_CORE"].endswith("keel-core")


def test_launcher_without_binary_says_what_to_do(monkeypatch):
    from keel import _keel_bin as K
    monkeypatch.setattr(K, "binary", lambda: None)
    with pytest.raises(SystemExit, match="cargo build"):
        K.main()
