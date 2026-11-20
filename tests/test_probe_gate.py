"""Chapter 3, P7: the probe gate's packaging, refusals and decision rule (no model run; the decisions themselves are
checked against the frozen research run by doc/product/spikes/p7-gate/port_check.py)."""

import hashlib
import json
import shutil
import struct
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

from keel import cli  # noqa: E402
from keel.backends import mlx_gate  # noqa: E402
from keel.backends.mlx_gate import GateRefused, Hooked, ProbeGate, gate_files, model_fingerprint  # noqa: E402
from keel.specs import QWEN35_4B_8BIT as SPEC  # noqa: E402

MODEL = Path(SPEC.local_dir)
needs_model = pytest.mark.skipif(not (MODEL / "config.json").is_file(), reason="local model folder missing")


def fake_model_dir(d: Path, header: dict) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text('{"model_type": "qwen3_5"}')
    h = json.dumps(header).encode()
    (d / "model.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + b"\0" * 16)
    return d


def test_packaged_probe_is_the_frozen_one():
    npz, js = gate_files()
    meta = json.loads(js.read_text())
    assert hashlib.sha1(npz.read_bytes()).hexdigest() == meta["sha1_npz"] == "77a9b8b43bb39a39b4de2b83034ddd11b7e7e8a7"
    assert meta["layer"] == 16 and meta["margin"] == 0.27 and "multi-turn" in meta["not_for"]
    assert meta["model"]["revision"].startswith("1704583a")


def test_decision_rule():
    g = ProbeGate.__new__(ProbeGate)
    g.margin = 0.27
    assert [g.action(p) for p in (0.99, 0.77, 0.76, 0.5, 0.24, 0.23, 0.0)] == [
        "force", "force", "free", "free", "free", "block", "block"]


@needs_model
def test_loads_for_the_model_it_was_fit_on():
    g = ProbeGate.load(MODEL)                          # fingerprint = headers only: no weights are read
    assert g.layer == 16 and g.margin == 0.27 and g.record()["experimental"] is True


def test_refuses_another_model(tmp_path):
    other = fake_model_dir(tmp_path / "other", {"w": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}})
    with pytest.raises(GateRefused, match="only works with the model it was fit on"):
        ProbeGate.load(other)
    with pytest.raises(GateRefused, match="no safetensors"):
        model_fingerprint(_no_weights(tmp_path / "noweights"))


def _no_weights(d: Path) -> Path:
    d.mkdir()
    (d / "config.json").write_text("{}")
    return d


def test_fingerprint_sees_header_changes(tmp_path):
    a = model_fingerprint(fake_model_dir(tmp_path / "a", {"w": {"shape": [1]}}))
    b = model_fingerprint(fake_model_dir(tmp_path / "b", {"w": {"shape": [2]}}))
    assert a != b and a == model_fingerprint(tmp_path / "a")


def test_refuses_a_changed_probe_file(tmp_path, monkeypatch):
    npz, js = gate_files()
    bad = tmp_path / "g.npz"
    shutil.copy(npz, bad)
    with open(bad, "ab") as f:
        f.write(b"x")
    monkeypatch.setattr(mlx_gate, "gate_files", lambda name=None: (bad, js))
    with pytest.raises(GateRefused, match="changed"):
        ProbeGate.load(MODEL)


def test_hook_is_restored_even_on_errors():
    class M:
        class language_model:
            class model:
                layers = ["l0", "l1"]
    with pytest.raises(RuntimeError):
        with Hooked(M, 1) as hook:
            assert M.language_model.model.layers[1] is hook
            raise RuntimeError("boom")
    assert M.language_model.model.layers == ["l0", "l1"]


@pytest.mark.parametrize("argv,msg", [
    (["run", "--name", "x", "--split", "dev", "--probe-gate", "--backend", "mlx-server"], "in-process MLX only"),
    (["run", "--name", "x", "--split", "dev", "--probe-gate", "--stepwise"], "not combined with --stepwise"),
    (["bfcl-mt", "--name", "x", "--split", "dev", "--probe-gate", "--state-saves"], "not combined with --state-saves"),
])
def test_cli_refuses_the_gate_where_it_cannot_work(argv, msg, tmp_path, monkeypatch, capsys):
    import keel.runlog as runlog
    monkeypatch.setattr(runlog, "EXPERIMENTS", tmp_path)
    with pytest.raises(SystemExit) as ex:
        cli.main(argv)
    assert msg in str(ex.value)
