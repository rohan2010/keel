"""Logical sandbox clock: no wall-clock mtimes reach the model, so greedy eval runs are repeatable."""

import os
import time

from keel.agent.sandbox import CLOCK_BASE, Sandbox
from keel.evals import tasks as T


def test_clock_hides_wall_time(tmp_path):
    it = T.build("C1", 0)
    tmpl = T.materialize(it, tmp_path / "tmpl")
    sb = Sandbox(tmp_path / "sb").create(tmpl)
    for d, dirs, files in os.walk(sb.root):
        for n in [*dirs, *files]:
            assert os.lstat(os.path.join(d, n)).st_mtime < time.time() - 86400
    for rel, t in it.mtimes.items():                       # task-set times are kept
        assert os.stat(sb.work / rel).st_mtime == t
    (sb.work / "new.txt").write_text("x")
    sb.settle_clock(3)
    assert os.stat(sb.work / "new.txt").st_mtime == CLOCK_BASE + 180
