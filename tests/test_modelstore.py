"""keel.modelstore + `keel-core download` (Ch7 P3): find order, verification, ask-first download. No network: the
Hub's record and the fetch are replaced with fakes."""

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

import keel.modelstore as M
from keel.commands import run_download as D
from keel.specs import ModelSpec

SPEC = ModelSpec("toy", "org/toy", "a" * 40, "models/Toy", {})
FILES = {"config.json": b'{"model_type": "toy"}', "model.safetensors": b"\x00" * 1000, "tokenizer.json": b"{}"}


def _record(files=FILES):
    out = []
    for name, data in files.items():
        if name.endswith(".safetensors"):
            out.append(M.RepoFile(name, len(data), hashlib.sha256(data).hexdigest(), None))
        else:
            out.append(M.RepoFile(name, len(data), None, hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()))
    return out


def _write(folder: Path, files=FILES) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (folder / name).write_bytes(data)
    return folder


@pytest.fixture
def models(monkeypatch, tmp_path):
    monkeypatch.setenv("KEEL_MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(M, "cached_snapshot", lambda spec: None)
    monkeypatch.setattr(M, "MODELS", {"toy": SPEC})
    return tmp_path / "models"


def test_complete_needs_config_and_every_indexed_weight(tmp_path):
    f = _write(tmp_path / "m")
    assert M.complete(f)
    (f / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "model-1.safetensors"}}))
    assert not M.complete(f)                             # the index names a file that is not there
    (f / "model-1.safetensors").write_bytes(b"x")
    assert M.complete(f)
    (f / "config.json").unlink()
    assert not M.complete(f)


def test_verify_catches_missing_resized_and_changed_files(tmp_path):
    f = _write(tmp_path / "m")
    assert M.verify(f, _record()) == []
    (f / "model.safetensors").write_bytes(b"\x01" + b"\x00" * 999)          # same size, one byte changed
    (f / "tokenizer.json").write_bytes(b"{} ")
    (f / "config.json").unlink()
    probs = "\n".join(M.verify(f, _record()))
    assert "model.safetensors: SHA-256" in probs and "tokenizer.json: 3 bytes" in probs and "config.json: missing" in probs


def test_find_prefers_models_folder_then_hf_cache(models, monkeypatch, tmp_path):
    assert M.find(SPEC) is None
    cache = _write(tmp_path / "hf" / "snapshots" / SPEC.revision)
    monkeypatch.setattr(M, "cached_snapshot", lambda spec: cache)
    assert M.find(SPEC) == cache
    local = _write(models / "Toy")
    assert M.find(SPEC) == local
    assert SPEC.path == local                             # ModelSpec.path goes through the same lookup


def test_resolve_names_the_models_folder_when_missing(models):
    assert M.resolve(SPEC) == models / "Toy"


def _args(**kw):
    base = {"model": "toy", "model_dir": None, "check": False, "yes": False, "list": False}
    return Namespace(**{**base, **kw})


def test_check_never_downloads(models, monkeypatch, capsys):
    monkeypatch.setattr(M, "repo_files", lambda spec: _record())
    monkeypatch.setattr(M, "fetch", lambda *a: pytest.fail("download attempted"))
    assert D.run(_args(check=True)) == 1
    assert "keel download" in capsys.readouterr().out
    _write(models / "Toy")
    assert D.run(_args(check=True)) == 0


def test_asks_first_and_refuses_without_a_terminal(models, monkeypatch, capsys):
    monkeypatch.setattr(M, "repo_files", lambda spec: _record())
    monkeypatch.setattr(M, "fetch", lambda *a: pytest.fail("download attempted"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    assert D.run(_args()) == 1
    out = capsys.readouterr().out
    assert "Size:" in out and "Free:" in out and "Not downloaded" in out


def test_refuses_when_disk_is_too_full(models, monkeypatch, capsys):
    monkeypatch.setattr(M, "repo_files", lambda spec: _record())
    monkeypatch.setattr(M, "free_bytes", lambda where: 10)
    monkeypatch.setattr(M, "fetch", lambda *a: pytest.fail("download attempted"))
    assert D.run(_args(yes=True)) == 1
    assert "Not enough space" in capsys.readouterr().out


def test_yes_downloads_then_verifies(models, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(M, "repo_files", lambda spec: _record())
    monkeypatch.setattr(M, "free_bytes", lambda where: 10**12)
    monkeypatch.setattr(M, "fetch", lambda spec, d=None: _write(tmp_path / "fetched"))
    assert D.run(_args(yes=True)) == 0
    assert "verified: 3 files" in capsys.readouterr().out


def test_bad_download_is_reported(models, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(M, "repo_files", lambda spec: _record())
    monkeypatch.setattr(M, "free_bytes", lambda where: 10**12)
    monkeypatch.setattr(M, "fetch", lambda spec, d=None: _write(tmp_path / "bad", {**FILES, "model.safetensors": b"x"}))
    assert D.run(_args(yes=True)) == 1
    assert "NOT VERIFIED" in capsys.readouterr().out


def test_offline_with_a_model_is_fine(models, monkeypatch, capsys):
    _write(models / "Toy")
    monkeypatch.setattr(M, "repo_files", lambda spec: (_ for _ in ()).throw(OSError("offline")))
    assert D.run(_args()) == 0
    assert "not verified" in capsys.readouterr().out


def test_cli_returns_the_exit_code(models, monkeypatch):
    from keel import cli
    monkeypatch.setattr(M, "repo_files", lambda spec: _record())
    monkeypatch.setattr(D.M, "MODELS", {"toy": SPEC})
    with pytest.raises(SystemExit) as e:
        cli.main(["download", "--model", "toy", "--check"])
    assert e.value.code == 1
