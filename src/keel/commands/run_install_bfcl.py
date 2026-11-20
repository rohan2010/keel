"""keel-core install-bfcl: add the BFCL checker (bfcl-eval) to keel's own environment, for `keel-core bfcl-mt`.

  keel-core install-bfcl            installs bfcl-eval==2026.3.23 with --no-deps into this Python environment
  keel-core install-bfcl --print    only print the command

Why not a `keel[bfcl]` extra: bfcl-eval pins numpy 1.26.4 and pulls in torch-sized packages, which clash with keel's
own (numpy >= 2.5). keel only needs its data files and checker, which run on keel's dependencies, so it is installed
without its own dependencies (as the research did). Uses `uv pip` when uv is available, else `python -m pip`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

BFCL_PIN = "bfcl-eval==2026.3.23"


def command() -> list[str]:
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, "--no-deps", BFCL_PIN]
    return [sys.executable, "-m", "pip", "install", "--no-deps", BFCL_PIN]


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--print", action="store_true", help="only print the install command")


def run(a: argparse.Namespace) -> int:
    cmd = command()
    if a.print:
        print(" ".join(cmd))
        return 0
    print("running: " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False).returncode
