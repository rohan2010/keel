"""The scripted checks in tests/checks/ (sandbox escapes, delete guard, file tools, task oracles, scorer fixes, BFCL
replay, verify layers, episode RNG). Each is a standalone script that prints PASS / FAIL lines and exits non-zero on
any failure; here each runs as `__main__` and must exit 0. On failure pytest shows its PASS / FAIL output.

Run one by hand: `uv run python tests/checks/check_sandbox.py`.
"""

import runpy
import sys
from pathlib import Path

import pytest

CHECKS = Path(__file__).resolve().parent / "checks"
CASES = [(f.stem, []) for f in sorted(CHECKS.glob("check_*.py"))]
CASES += [("check_agent_tasks", ["--long"]), ("check_agent_tasks", ["--ch6"])]


@pytest.mark.parametrize(("name", "args"), CASES, ids=[" ".join([n.removeprefix("check_"), *a]) for n, a in CASES])
def test_check(name, args, monkeypatch):
    script = CHECKS / f"{name}.py"
    monkeypatch.setattr(sys, "argv", [str(script), *args])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(script), run_name="__main__")
    assert e.value.code in (0, None), f"{name} {' '.join(args)} reported failures (see output above)"
