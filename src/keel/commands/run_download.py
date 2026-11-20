"""keel-core download: get the model keel runs on (pinned revision, size and free disk shown, asks first).

  keel download                         the default model (Qwen3.5-4B 8-bit, ~5.2 GB) into the Hugging Face cache
  keel download --check                 verify the model you have against the Hub's record; never downloads
  keel download --model-dir DIR         into DIR instead of the Hugging Face cache
  keel download --model qwen3.5-9b-4bit another model keel knows (see --list)
  keel download --yes                   don't ask (scripts)

A model already on disk (the models folder or the Hugging Face cache, at the pinned revision) is found and verified,
not downloaded again. An interrupted download resumes where it stopped. Every file is checked against the Hub's
record of the revision (size; SHA-256 for the weights).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from keel import modelstore as M


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", default=M.DEFAULT.name, choices=sorted(M.MODELS), help="which model (default: %(default)s)")
    ap.add_argument("--model-dir", default=None, help="download into this folder instead of the Hugging Face cache")
    ap.add_argument("--check", action="store_true", help="only verify the model on disk; never download")
    ap.add_argument("--yes", "-y", action="store_true", help="download without asking")
    ap.add_argument("--list", action="store_true", help="models keel can download, and where each one is now")


def _ask(question: str) -> bool:
    if not sys.stdin.isatty():
        return False
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _verify(where: Path, files) -> int:
    print(f"verifying {where} ...", flush=True)
    problems = M.verify(where, files)
    if problems:
        print("NOT VERIFIED:\n  " + "\n  ".join(problems))
        print("Run `keel download` again to repair it (it re-fetches what is missing or wrong).")
        return 1
    print(f"verified: {len(files)} files match the Hub's record of this revision")
    return 0


def run(a: argparse.Namespace) -> int:
    if a.list:
        for name, spec in sorted(M.MODELS.items()):
            have = M.find(spec)
            print(f"{name:18} {spec.hf_repo}@{spec.revision[:7]}  {have if have else 'not downloaded'}")
        return 0
    spec = M.MODELS[a.model]
    target = Path(a.model_dir).expanduser().resolve() if a.model_dir else None
    have = (target if M.complete(target) else None) if target else M.find(spec)
    label = f"{spec.hf_repo} at revision {spec.revision[:12]}"
    try:
        files = M.repo_files(spec)
    except Exception as err:                       # offline, Hub down
        if have:
            print(f"{spec.name} is at {have} (not verified: could not reach Hugging Face: {type(err).__name__})")
            return 0
        print(f"could not reach Hugging Face to look up {label}: {err}")
        return 1
    if a.check:
        if not have:
            print(f"{spec.name} is not on this Mac. Get it with `keel download`.")
            return 1
        return _verify(have, files)
    if have:
        print(f"{spec.name} is already at {have}")
        return _verify(have, files)

    dest = target or M.hf_cache_dir()
    partial = target if target and target.exists() else None
    need, total = M.missing_bytes(partial, files), sum(f.size for f in files)
    free = M.free_bytes(dest)
    print(f"Model:  {label}")
    print(f"Size:   {M.gb(total)}" + (f" ({M.gb(need)} still to fetch)" if need != total else ""))
    print(f"Into:   {dest}")
    print(f"Free:   {M.gb(free)} on that disk")
    if need + M.MARGIN > free:
        print(f"Not enough space: need {M.gb(need)} plus 1 GB to spare. Free some space, or use --model-dir on "
              "another disk.")
        return 1
    if not a.yes and not _ask(f"Download {M.gb(need)}?"):
        print("Not downloaded." + ("" if sys.stdin.isatty() else " (no terminal to ask in: pass --yes)"))
        return 1
    try:
        where = M.fetch(spec, target)
    except KeyboardInterrupt:
        print("\nStopped. Run `keel download` again to resume where it stopped.")
        return 130
    rc = _verify(where, files)
    if rc == 0 and target is not None and M.find(spec) != target:
        print(f"keel looks for this model in {M.local_folder(spec)} or the Hugging Face cache. To use {target}, set "
              f"KEEL_MODELS={target.parent} (if the folder is named {M.local_folder(spec).name}) or pass "
              f"--model-dir {target} to keel-core commands.")
    if rc == 0:
        print("Next: `keel doctor` to check everything, then `keel run --dir <folder> \"<task>\"`.")
    return rc
