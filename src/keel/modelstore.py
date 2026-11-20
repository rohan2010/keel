"""Where a model's weights are, and getting them.

Lookup, first match wins (`find`):
1. the models folder (`keel.paths.models_dir()`: the repo's `models/` in a source checkout, the state folder when
   installed, or `KEEL_MODELS`), folder named after the spec's `local_dir` (`models/Qwen3.5-4B-8bit`);
2. the Hugging Face cache (`~/.cache/huggingface/hub`, or `HF_HOME` / `HF_HUB_CACHE`) at the spec's pinned revision,
   so a model the user already has is not downloaded twice.

`download` fetches the pinned revision into the Hugging Face cache (or `model_dir`), after showing the size and the
free disk space and asking; an interrupted download resumes. `verify` checks every file against the Hub's record of
that revision: size, SHA-256 for large (LFS) files, the git blob hash for the rest. Nothing here runs at import time,
and nothing downloads without `download` being called.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .specs import QWEN35_4B_8BIT, QWEN35_9B_4BIT, ModelSpec

MODELS = {s.name: s for s in (QWEN35_4B_8BIT, QWEN35_9B_4BIT)}       # what `keel download` offers
DEFAULT = QWEN35_4B_8BIT
MARGIN = 1_000_000_000                                              # keep 1 GB free after a download


@dataclass
class RepoFile:
    name: str
    size: int
    sha256: str | None          # LFS files
    blob_id: str | None         # git blob sha1 (small files)


def local_folder(spec: ModelSpec) -> Path:
    """The models-folder location for this spec (whether or not it exists)."""
    return paths.model_path(spec.local_dir)


def complete(folder: Path) -> bool:
    """Has a config and every weight file the index names (or a single model.safetensors)."""
    if not (folder / "config.json").is_file():
        return False
    idx = folder / "model.safetensors.index.json"
    if idx.is_file():
        import json
        try:
            files = set(json.loads(idx.read_text()).get("weight_map", {}).values())
        except (OSError, ValueError):
            return False
        return bool(files) and all((folder / f).is_file() for f in files)
    return any(folder.glob("*.safetensors"))


def cached_snapshot(spec: ModelSpec) -> Path | None:
    """The Hugging Face cache's folder for this spec's pinned revision, if it is there and complete (no network)."""
    try:
        from huggingface_hub import snapshot_download
        p = Path(snapshot_download(spec.hf_repo, revision=spec.revision, local_files_only=True))
    except Exception:
        return None
    return p if complete(p) else None


def find(spec: ModelSpec) -> Path | None:
    """Where this model's weights are, or None (see the module docstring for the order)."""
    p = Path(spec.local_dir).expanduser()
    if p.is_absolute():
        return p if complete(p) else None
    local = local_folder(spec)
    if complete(local):
        return local
    return cached_snapshot(spec)


def resolve(spec: ModelSpec) -> Path:
    """`find`, else the models-folder location (so an error names a real place; `keel download` fills it)."""
    return find(spec) or local_folder(spec)


# ---- the Hub's record of a revision ------------------------------------------------------------------------------

def repo_files(spec: ModelSpec) -> list[RepoFile]:
    """Every file of the pinned revision with its size and hash (one metadata request, no download)."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(spec.hf_repo, revision=spec.revision, files_metadata=True)
    if info.sha != spec.revision:
        raise RuntimeError(f"the Hub answered for revision {info.sha}, not the pinned {spec.revision}")
    return [RepoFile(s.rfilename, int(s.size or 0), s.lfs.sha256 if s.lfs else None, getattr(s, "blob_id", None))
            for s in (info.siblings or [])]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _git_blob(p: Path) -> str:
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify(folder: Path, files: list[RepoFile]) -> list[str]:
    """Every way `folder` differs from the Hub's record (empty = verified)."""
    problems = []
    for f in files:
        p = folder / f.name
        if not p.is_file():
            problems.append(f"{f.name}: missing")
        elif p.stat().st_size != f.size:
            problems.append(f"{f.name}: {p.stat().st_size} bytes, expected {f.size}")
        elif f.sha256 and _sha256(p) != f.sha256:
            problems.append(f"{f.name}: SHA-256 does not match")
        elif not f.sha256 and f.blob_id and _git_blob(p) != f.blob_id:
            problems.append(f"{f.name}: content does not match")
    return problems


# ---- download ----------------------------------------------------------------------------------------------------

def hf_cache_dir() -> Path:
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE)


def missing_bytes(target: Path | None, files: list[RepoFile]) -> int:
    """Bytes still to fetch: files not yet present at full size (a resumed download only fetches the rest)."""
    if target is None:
        return sum(f.size for f in files)
    return sum(f.size for f in files if not ((target / f.name).is_file() and (target / f.name).stat().st_size == f.size))


def free_bytes(where: Path) -> int:
    p = where
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def fetch(spec: ModelSpec, model_dir: Path | None = None) -> Path:
    """Download (or resume) the pinned revision. Into the Hugging Face cache unless `model_dir` is given."""
    from huggingface_hub import snapshot_download
    if model_dir is not None:
        return Path(snapshot_download(spec.hf_repo, revision=spec.revision, local_dir=str(model_dir)))
    return Path(snapshot_download(spec.hf_repo, revision=spec.revision))


def gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"
