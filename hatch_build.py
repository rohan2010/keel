"""Wheel build hook (Ch7 P4, decision D1): put the Rust `keel` CLI/TUI inside the wheel as keel/bin/keel.

`uv build --wheel` (or any PEP 517 frontend) runs `cargo build --release -p keel`, copies the binary into the wheel and
tags the wheel for macOS on Apple Silicon (the binary is native code). `KEEL_PREBUILT_BIN=<path>` uses a binary built
elsewhere (CI) instead of running cargo. Editable installs (`uv sync`) skip all of this: in a checkout the `keel`
entry point runs target/{release,debug}/keel.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

MACOS_TAG = "py3-none-macosx_11_0_arm64"       # Rust's default deployment target for aarch64-apple-darwin is 11.0


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel" or version == "editable":
            return
        root = Path(self.root)
        prebuilt = os.environ.get("KEEL_PREBUILT_BIN")
        if prebuilt:
            binary = Path(prebuilt)
        else:
            cargo = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo")
            subprocess.run([cargo, "build", "--release", "-p", "keel"], cwd=root, check=True)
            binary = root / "target" / "release" / "keel"
        if not binary.is_file():
            raise RuntimeError(f"no keel binary at {binary}")
        build_data["force_include"][str(binary)] = "keel/bin/keel"
        build_data["pure_python"] = False
        build_data["tag"] = MACOS_TAG
