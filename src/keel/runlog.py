"""One folder per run under experiments/: config, outputs (prompts + generations), metrics, timings, summary."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

from . import paths

EXPERIMENTS = paths.experiments_dir()       # KEEL_EXPERIMENTS moves it (e.g. re-checking an evidence pack)


def _env() -> dict:
    import mlx.core as mx
    import mlx_lm
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()
    return {"mlx": mx.__version__, "mlx_lm": mlx_lm.__version__, "python": platform.python_version(),
            "macos": platform.mac_ver()[0], "chip": chip, "memory_gb": int(mem) / 2**30 if mem else None}


def code_hashes() -> dict:
    """Code freeze record: sha1 of every .py of the keel package (`keel/<path>`, the same in a checkout and an
    installed wheel), plus one combined hash. Two runs with the same `combined` ran the same code. Before the
    paths rework the hash also covered scripts/*.py with keys `src/keel/...` (old hashes are not comparable)."""
    per = {k: hashlib.sha1(f.read_bytes()).hexdigest()[:12] for k, f in paths.code_files().items()}
    combined = hashlib.sha1(json.dumps(per, sort_keys=True).encode()).hexdigest()[:12]
    return {"combined": combined, "files": per}


# --- bias rules: every use of a test / validation / holdout set is logged ----------------
LEDGER = EXPERIMENTS / "test_ledger.jsonl"
SEALED_SPLITS = {"test", "val", "holdout", "held_out", "seen"}  # agent "test" seeds, BFCL "val" = validation; "holdout" = final;
# When2Call "held_out" / "seen"
HOLDOUT_CONFIRM = "frozen-final-run"


def check_split(split: str, confirm: str | None) -> None:
    """Refuse a final-holdout run unless explicitly confirmed (it is run once, at the end, with a frozen harness)."""
    if split == "holdout" and confirm != HOLDOUT_CONFIRM:
        raise SystemExit(f"Refusing to run the FINAL HOLDOUT. It is used once, at the end, with every layer frozen. "
                         f"If that is now, pass --confirm-holdout {HOLDOUT_CONFIRM}.")


def ledger(run_dir: Path, split: str, purpose: str, extra: dict | None = None) -> None:
    if split not in SEALED_SPLITS:
        return
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "run": run_dir.name, "split": split, "purpose": purpose,
           "code": code_hashes()["combined"], **(extra or {})}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(row) + "\n")


class RunLog:
    def __init__(self, name: str, config: dict):
        self.dir = EXPERIMENTS / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.t0 = time.time()
        self.timings: dict[str, float] = {}
        self.write_json("config.json", {**config, "env": _env(), "code": code_hashes(),
                                         "started": time.strftime("%Y-%m-%d %H:%M:%S")})

    def path(self, name: str) -> Path:
        return self.dir / name

    def write_json(self, name: str, obj) -> None:
        self.path(name).write_text(json.dumps(obj, indent=1))

    def append_jsonl(self, name: str, row: dict) -> None:
        with open(self.path(name), "a") as f:
            f.write(json.dumps(row) + "\n")

    def time(self, key: str, seconds: float) -> None:
        self.timings[key] = round(seconds, 2)
        self.write_json("timings.json", {**self.timings, "total_wall_s": round(time.time() - self.t0, 2)})

    def summary(self, text: str) -> None:
        self.path("summary.md").write_text(text)
