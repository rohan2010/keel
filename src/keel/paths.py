"""Where keel keeps things: the only module that knows file locations.

Two layouts, picked automatically:
- **source checkout** (this package lives at `<repo>/src/keel` next to the repo's `pyproject.toml`): runs, models
  and data stay in the repo as before (`experiments/`, `models/`, `data/`), so development is unchanged;
- **installed** (wheel / `uv tool install`): everything lives under the state folder,
  `~/Library/Application Support/keel` (`experiments/`, `models/`, `data/`, `baselines/`, plus the TUI's sessions,
  backups and logs).

Each location can be moved with an environment variable; a set variable always wins:

  KEEL_STATE_DIR     the state folder (sessions, backups, logs; the installed layout's base)
  KEEL_EXPERIMENTS   run folders, pre-registrations, compares, noise bands, the test ledger
  KEEL_MODELS        model folders (a model's `local_dir` "models/<name>" resolves to <KEEL_MODELS>/<name>)
  KEEL_DATA          fetched datasets (`external/when2call/`)
  KEEL_BASELINES     goose / opencode installs for `keel-core baseline`

Nothing else in keel builds a location from `__file__` or the current folder.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent            # .../keel (the installed or checked-out package)


def _env(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else None


def source_root() -> Path | None:
    """The repo root when running from a source checkout (src layout + keel's pyproject.toml), else None."""
    root = PACKAGE.parent.parent
    pp = root / "pyproject.toml"
    if PACKAGE.parent.name == "src" and pp.is_file():
        try:
            if 'name = "keel"' in pp.read_text():
                return root
        except OSError:
            pass
    return None


def state_dir() -> Path:
    """`KEEL_STATE_DIR` if set (tests, a second profile), else ~/Library/Application Support/keel."""
    return _env("KEEL_STATE_DIR") or Path.home() / "Library" / "Application Support" / "keel"


def _base(env: str, checkout_name: str, installed_name: str) -> Path:
    if (p := _env(env)) is not None:
        return p
    root = source_root()
    return root / checkout_name if root else state_dir() / installed_name


def experiments_dir() -> Path:
    return _base("KEEL_EXPERIMENTS", "experiments", "experiments")


def models_dir() -> Path:
    return _base("KEEL_MODELS", "models", "models")


def data_dir() -> Path:
    return _base("KEEL_DATA", "data", "data")


def baselines_dir() -> Path:
    return _base("KEEL_BASELINES", ".baselines", "baselines")


def model_path(local_dir: str | Path) -> Path:
    """A model spec's folder. Absolute: as given. `models/<name>` (how the built-in specs and run records name it):
    <models_dir>/<name>. Anything else relative: under the source checkout if there is one, else the models folder."""
    p = Path(local_dir).expanduser()
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "models":
        return models_dir().joinpath(*p.parts[1:])
    root = source_root()
    return (root / p) if root else models_dir() / p


def code_files() -> dict[str, Path]:
    """The files the code hash covers: every .py of the keel package, keyed `keel/<path>` (same in a checkout and an
    installed wheel)."""
    return {f"keel/{f.relative_to(PACKAGE).as_posix()}": f for f in sorted(PACKAGE.rglob("*.py"))
            if "__pycache__" not in f.parts}
