"""Re-check the Chapter 6 headline from this folder alone.

  python verify.py            from a keel checkout's venv (uv run python evidence/ch6-headline/verify.py)

Three checks:
  1. every file matches SHA256SUMS;
  2. "as run": the archived code (code/keel-1b1049713d33.tar.gz, the exact code of the counted runs) re-runs
     `keel-core compare --prereg` for H1 -> H4 in the frozen order;
  3. "current": the keel you have installed does the same.
Each compare runs in a fresh temp folder that holds only the frozen pre-registrations (no recorded results), so the
fixed-sequence gates are decided again from scratch. A check passes when every verdict, test and number equals the
stored report (reports/h*.json) and the recorded deciding p (recorded/*.result.json). Exit code 0 = all passed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = "1b1049713d33"
RUNS = {"bare": "20261119-221250_ch6_bare_qwen3.5-4b-8bit", "full": "20261119-233312_ch6_full_qwen3.5-4b-8bit",
        "goose": "20261120-001856_ch6_goose_qwen3.5-4b-8bit", "opencode": "20261120-014518_ch6_opencode_qwen3.5-4b-8bit"}
HYPS = [("h1", "ch6-h1-full-vs-bare", "bare"), ("h2", "ch6-h2-full-fs-vs-bare", "bare"),
        ("h3", "ch6-h3-full-vs-goose", "goose"), ("h4", "ch6-h4-full-vs-opencode", "opencode")]
IGNORE = {"runs"}                     # run folder paths differ by where the pack lives
# Later keel prints a gated hypothesis as "not tested" (Ch7 P7); the 2026-11-20 reports said "no detectable effect"
# with the same reason ("not tested: ...") and the same numbers. Only that rename is accepted, only in "current".
RENAMED = {"no detectable effect": "not tested"}


def check_sums() -> list[str]:
    bad = []
    for line in (HERE / "SHA256SUMS").read_text().splitlines():
        want, name = line.split("  ", 1)
        p = HERE / name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != want:
            bad.append(name)
    return bad


def run_compares(label: str, python_env: dict, experiments: Path, out: Path) -> bool:
    (experiments / "prereg").mkdir(parents=True)
    for _, name, _ in HYPS:
        shutil.copy2(HERE / "preregs" / f"{name}.json", experiments / "prereg" / f"{name}.json")
    ok = True
    print(f"\n[{label}]")
    for h, name, base in HYPS:
        js = out / f"{label}_{h}.json"
        cmd = [sys.executable, "-c", "import sys; from keel.cli import main; main(sys.argv[1:])", "compare",
               str(HERE / "runs" / RUNS[base]), str(HERE / "runs" / RUNS["full"]), "--prereg", name,
               "--no-report", "--json", str(js)]
        r = subprocess.run(cmd, env=python_env, capture_output=True, text=True)
        if r.returncode != 0 or not js.exists():
            print(f"  {h}: compare failed\n{r.stdout}\n{r.stderr}")
            ok = False
            continue
        fresh, stored = json.loads(js.read_text()), json.loads((HERE / "reports" / f"{h}.json").read_text())
        shown = fresh["verdict"]
        if label == "current" and fresh.get("verdict") == "not tested" == RENAMED.get(stored.get("verdict")) and \
                any("NOT TESTED" in f for f in fresh.get("flags", [])):
            fresh["verdict"] = stored["verdict"]        # same result, verdict renamed in keel Ch7 P7 (see README)
        diffs = sorted(k for k in set(fresh) | set(stored) if k not in IGNORE and fresh.get(k) != stored.get(k))
        res_fresh = json.loads((experiments / "prereg" / f"{name}.result.json").read_text())
        res_stored = json.loads((HERE / "recorded" / f"{name}.result.json").read_text())
        same_p = (res_fresh["p"], res_fresh["delta"]) == (res_stored["p"], res_stored["delta"])
        good = not diffs and same_p
        ok &= good
        t = fresh["pass_test"]
        print(f"  {h}: {shown.upper():22} deciding {t['delta']:+.3f} p = {t['p']:.4g}   "
              f"{'MATCH' if good else 'DIFFERS: ' + ', '.join(diffs) + ('' if same_p else ' recorded p')}")
    return ok


def main() -> int:
    bad = check_sums()
    print(f"SHA256SUMS: {'all files match' if not bad else 'MISMATCH ' + ', '.join(bad)}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with tarfile.open(HERE / "code" / f"keel-{CODE}.tar.gz") as tar:
            tar.extractall(tmp, filter="data")
        snap = tmp / f"keel-{CODE}"
        env_run = {**os.environ, "PYTHONPATH": str(snap / "src")}
        env_run.pop("KEEL_EXPERIMENTS", None)          # the archived code has no override: ROOT/experiments
        ok_run = run_compares("as run", env_run, snap / "experiments", tmp)
        cur = tmp / "current"
        env_cur = {**os.environ, "KEEL_EXPERIMENTS": str(cur)}
        env_cur.pop("PYTHONPATH", None)
        ok_cur = run_compares("current", env_cur, cur, tmp)
    ok = not bad and ok_run and ok_cur
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
