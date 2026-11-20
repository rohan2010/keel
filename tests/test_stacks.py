"""Release audit #5 (2026-11-19): one definition of the bare / keel / full stacks, used by the evals' flags and by
the interactive server, so the TUI runs exactly what was measured."""

import argparse
import shutil
from types import SimpleNamespace

import pytest

from keel import acp_server
from keel.agent.loop import SEND_NOTE, TOOLSETS
from keel.commands import run_agent
from keel.stacks import SHIPPED, STACKS

# the Chapter 6 headline's flags (doc/product/spikes/ch6/headline.sh): bare arm, keel arm (defaults), full arm
HEADLINE = {"bare": "--toolset shell --no-loop-break --no-trim-outputs",
            "keel": "",
            "full": "--done-gate --gnu-tools --examples --toolset files_py --no-send-note --search-hint"}
FIELDS = ("toolset", "loop_break", "trim_outputs", "done_gate", "gnu_tools", "examples", "send_note", "search_hint")


def _run_args(flags: list[str]):
    ap = argparse.ArgumentParser()
    run_agent.add_arguments(ap)
    a = ap.parse_args(["--name", "x", *flags])
    return {f: getattr(a, f) for f in FIELDS}


@pytest.mark.parametrize("name", sorted(STACKS))
def test_stack_flags_equal_the_headline_arms(name):
    assert _run_args(STACKS[name].run_flags()) == _run_args(HEADLINE[name].split())


def test_shipped_default_is_full_after_h1():
    assert SHIPPED == "full"                                   # Ch6 H1 confirmed 2026-11-20 (p = 0.0128)
    assert STACKS["keel"].system_prompt() == TOOLSETS["files"][0]   # the keel stack itself is unchanged


def test_full_prompt():
    p = STACKS["full"].system_prompt()
    assert p.startswith(TOOLSETS["files_py"][0].replace(SEND_NOTE, "")) and SEND_NOTE not in p and "run_python" in p


@pytest.mark.skipif(shutil.which("sandbox-exec") is None, reason="macOS sandbox-exec needed")
@pytest.mark.parametrize("name", sorted(STACKS))
def test_session_uses_the_stack(tmp_path, name):
    folder = tmp_path / "proj"
    folder.mkdir()
    eng = SimpleNamespace(trimmer=object())
    s = acp_server.Session("s0123456789", folder, eng, tmp_path / "state", lambda *a: False, name)
    try:
        st, e = STACKS[name], s.ep
        assert e.toolset == st.toolset and e.loop_break == st.loop_break and e.done_gate == st.done_gate
        assert (e.trimmer is not None) == st.trim and (e.ftools is not None) == (st.toolset != "shell")
        assert e.messages[0]["content"] == st.system_prompt()
        assert ((s.sandbox.bin / "stat").exists()) == st.gnu_tools
    finally:
        s.close()


def test_acp_flag():
    ap = argparse.ArgumentParser()
    acp_server.add_arguments(ap)
    assert ap.parse_args([]).stack == SHIPPED and ap.parse_args(["--stack", "full"]).stack == "full"
