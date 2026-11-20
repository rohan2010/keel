"""Ch6 search hint (doc/findings/ch6-search-literal.md): a literal search that finds nothing and looks like a regular
expression says so; it never runs the regex itself. Off by default (output unchanged)."""

import shutil

import pytest

from keel.agent.filetools import FileTools, looks_like_regex
from keel.agent.guard import GuardedShell, approve_all
from keel.agent.sandbox import Sandbox

pytestmark = pytest.mark.skipif(shutil.which("sandbox-exec") is None, reason="macOS sandbox-exec needed")


@pytest.fixture
def tools(tmp_path):
    sb = Sandbox(tmp_path / "ep").create()
    (sb.work / "docs").mkdir()
    (sb.work / "pyproject.toml").write_text('version = "2.8.9"\n')
    (sb.work / "docs" / "conf.py").write_text('release = "2.8.9"\n')
    (sb.work / "tax-2020.txt").write_text("Tax paid: $1,234.00\n")
    return FileTools(GuardedShell(sb, approve_all, backup_dir=tmp_path / "bk"))


def test_off_by_default_unchanged(tools):
    assert tools.search(r"2\.8\.9") == "(no matches)"


@pytest.mark.parametrize("pat", [r"2\.8\.9", r"\$[0-9]+", r"paid.*\$", r"^\d+\.\d+$", r"Tax paid|total", r"v[0-9]"])
def test_hint_on_regex_like_no_match(tools, pat):
    tools.search_hint = True
    out = tools.search(pat)
    assert out.startswith("(no matches for the exact text") and "regex=true" in out
    assert "2020" not in out and "1,234" not in out          # the hint never shows what the regex would find


@pytest.mark.parametrize("pat", ["2.8.9", "Tax paid", "missing word", "a.txt", "C++", "what?"])
def test_no_hint_for_plain_text(tools, pat):
    tools.search_hint = True
    out = tools.search(pat)
    assert "regex=true" not in out                          # hits, or the plain "(no matches)"


def test_no_hint_when_regex_already_on_or_hits(tools):
    tools.search_hint = True
    assert tools.search(r"2\.8\.10", regex=True) == "(no matches)"
    assert "docs/conf.py:1" in tools.search(r"2\.8\.9", regex=True)
    assert "pyproject.toml:1" in tools.search("2.8.9")


def test_looks_like_regex():
    assert looks_like_regex(r"2\.8\.9") and looks_like_regex(r"a.*b") and looks_like_regex("x|y")
    assert not looks_like_regex("2.8.9") and not looks_like_regex("hello world") and not looks_like_regex("[unclosed")
