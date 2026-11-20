"""The `keel` command: start the Rust CLI/TUI.

In an installed wheel the binary is keel/bin/keel inside this package; in a source checkout it is
target/release/keel or target/debug/keel (`cargo build`). The binary is told where this environment's keel-core is
(`KEEL_CORE`), so it never depends on PATH or on where it was compiled.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from . import paths


def binary() -> Path | None:
    bundled = paths.PACKAGE / "bin" / "keel"
    if bundled.is_file():
        return bundled
    root = paths.source_root()
    if root:
        for profile in ("release", "debug"):
            p = root / "target" / profile / "keel"
            if p.is_file():
                return p
    return None


def core() -> str | None:
    here = Path(sys.executable).parent / "keel-core"       # same environment's entry point
    return str(here) if here.is_file() else shutil.which("keel-core")


def main() -> None:
    b = binary()
    if b is None:
        raise SystemExit("keel: the Rust binary is missing. In a source checkout run `cargo build`; otherwise reinstall "
                         "keel from a wheel built for macOS on Apple Silicon.")
    env = dict(os.environ)
    if not env.get("KEEL_CORE") and (c := core()):
        env["KEEL_CORE"] = c
    os.execve(str(b), [str(b), *sys.argv[1:]], env)
