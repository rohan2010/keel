"""keel.paths (Ch7 P1): one place for locations; checkout layout unchanged, installed layout under the state folder."""

import re
from pathlib import Path

import pytest

from keel import paths

REPO = Path(__file__).resolve().parents[1]
ENVS = ("KEEL_EXPERIMENTS", "KEEL_MODELS", "KEEL_DATA", "KEEL_BASELINES")


@pytest.fixture
def clean_env(monkeypatch):
    for e in ENVS:
        monkeypatch.delenv(e, raising=False)


def test_checkout_layout_is_unchanged(clean_env):
    assert paths.source_root() == REPO
    assert paths.experiments_dir() == REPO / "experiments"
    assert paths.models_dir() == REPO / "models"
    assert paths.data_dir() == REPO / "data"
    assert paths.baselines_dir() == REPO / ".baselines"
    assert paths.model_path("models/Qwen3.5-4B-8bit") == REPO / "models" / "Qwen3.5-4B-8bit"


def test_installed_layout_uses_state_folder(clean_env, monkeypatch, tmp_path):
    site = tmp_path / "lib" / "python3.12" / "site-packages" / "keel"
    site.mkdir(parents=True)
    monkeypatch.setattr(paths, "PACKAGE", site)
    state = tmp_path / "state"
    monkeypatch.setenv("KEEL_STATE_DIR", str(state))
    assert paths.source_root() is None
    assert paths.experiments_dir() == state / "experiments"
    assert paths.models_dir() == state / "models"
    assert paths.data_dir() == state / "data"
    assert paths.baselines_dir() == state / "baselines"
    assert paths.model_path("models/Qwen3.5-4B-8bit") == state / "models" / "Qwen3.5-4B-8bit"
    assert paths.model_path("weird/rel") == state / "models" / "weird" / "rel"


def test_not_a_checkout_without_keels_pyproject(clean_env, monkeypatch, tmp_path):
    pkg = tmp_path / "src" / "keel"
    pkg.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "other"\n')
    monkeypatch.setattr(paths, "PACKAGE", pkg)
    assert paths.source_root() is None


def test_env_overrides_win(monkeypatch, tmp_path):
    for e in ENVS:
        monkeypatch.setenv(e, str(tmp_path / e))
    assert paths.experiments_dir() == tmp_path / "KEEL_EXPERIMENTS"
    assert paths.models_dir() == tmp_path / "KEEL_MODELS"
    assert paths.data_dir() == tmp_path / "KEEL_DATA"
    assert paths.baselines_dir() == tmp_path / "KEEL_BASELINES"
    assert paths.model_path("models/X") == tmp_path / "KEEL_MODELS" / "X"


def test_absolute_model_path_kept(tmp_path):
    assert paths.model_path(str(tmp_path / "m")) == tmp_path / "m"


def test_code_files_cover_the_package_only():
    files = paths.code_files()
    assert "keel/paths.py" in files and "keel/runlog.py" in files
    assert all(k.startswith("keel/") and k.endswith(".py") and "__pycache__" not in k for k in files)


def test_no_location_built_outside_paths():
    """Release audit #6: nothing but keel.paths anchors a location on the package file or the repo root."""
    bad = []
    for f in sorted((REPO / "src" / "keel").rglob("*.py")):
        if f.name == "paths.py":
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"\bROOT\b|parents\[\d\]", code) or ("__file__" in code and "bfcl_eval.__file__" not in code):
                bad.append(f"{f.relative_to(REPO)}:{i}: {line.strip()}")
    assert not bad, "\n".join(bad)
