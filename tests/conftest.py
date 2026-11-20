"""Every test gets its own keel state folder, so nothing reads or writes the real one
(~/Library/Application Support/keel: session records, `keel doctor` server records)."""

import pytest


@pytest.fixture(autouse=True)
def _keel_state_dir(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("KEEL_STATE_DIR", str(tmp_path_factory.mktemp("keel-state")))
