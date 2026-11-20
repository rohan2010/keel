"""Ch5 G5: GNU forms of stat / uniq / date work in the sandbox on macOS; BSD forms pass through; off by default."""

import json
import os
import sys

import pytest

from keel.agent import gnu_shims
from keel.agent.loop import run_episodes_on
from keel.agent.sandbox import Sandbox
from keel.backends.scripted import ScriptedBackend
from keel.evals import tasks as T

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox")


@pytest.fixture
def sb(tmp_path):
    s = Sandbox(tmp_path / "ep").create(None)
    gnu_shims.install(s.bin)
    (s.work / "a.jpg").write_text("x")
    (s.work / "b.jpg").write_text("x")
    os.utime(s.work / "a.jpg", (1689348600, 1689348600))
    return s


def out(s, cmd):
    r = s.run(cmd)
    assert r.exit_code == 0, r.output
    return r.output.strip()


def test_gnu_stat_uniq_date(sb):
    assert out(sb, 'stat -c "%Y %n" a.jpg') == "1689348600 a.jpg"
    assert out(sb, 'stat --format="%s %F" a.jpg') == "1 regular file"
    assert out(sb, "md5sum *.jpg | sort | uniq -d -w32").endswith("a.jpg")
    assert out(sb, 'printf "a1\\na2\\nb\\n" | uniq -c -w1').split() == ["2", "a1", "1", "b"]
    assert out(sb, "date -d @1689348600 +%Y") == "2023"
    assert out(sb, 'date -d "2023-07-14 15:30" +%Y-%m-%d_%H%M') == "2023-07-14_1530"


def test_bsd_forms_pass_through_and_no_noise(sb):
    assert out(sb, 'stat -f "%m %N" a.jpg') == "1689348600 a.jpg"
    assert "xcrun" not in sb.run('stat -c "%n" a.jpg').output          # no stub noise in what the model reads
    assert out(sb, "date +%Y").isdigit()


def call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def test_flag_in_the_agent_loop(tmp_path):
    it = T.build("E3", 0)
    script = [call("run_shell", command='stat -c "%n" *.jpg | head -1'), "Done."]
    for flag, ok in ((True, True), (False, False)):
        d = tmp_path / str(flag)
        run_episodes_on(ScriptedBackend({it.id: list(script)}), [it], d, gnu_tools=flag)
        tool = next(m["content"] for m in json.loads((d / "transcripts" / f"{it.id}.json").read_text())["messages"]
                    if m["role"] == "tool")
        assert ("illegal option" not in tool) == ok, tool
