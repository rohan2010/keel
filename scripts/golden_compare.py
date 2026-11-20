"""Same-results gate (Chapter 1, A0 / A7): compare two golden runs made by scripts/golden.sh.

    uv run python scripts/golden_compare.py <tag_a> <tag_b>          e.g. a0_run1 a0_run2  (or a0_run1 a7)
    uv run python scripts/golden_compare.py <tag_a> <tag_b> --dirs D  also look for runs in D (frozen code copies)
    uv run python scripts/golden_compare.py <tag_a> <tag_b> --unstable E3_0:28   E3_0 only has to match up to message 28
    uv run python scripts/golden_compare.py <tag_a> <tag_b> --parity  pass counts only (for B2/B3, where batched
                                                                      float maths may change a few tokens)

Known-unstable items: E3_0 splits at message 28 between two runs of the SAME code (identical input, greedy): a
near-tie at one token flips on float noise (2026-11-13, a0_run1/2 and a0c_run1/2). `--unstable ID:N` requires the
first N messages to be identical and reports the rest instead of failing on it.

Compares every transcript (messages, tool calls, scores, stops, notes, token counts) of the agent and BFCL runs.
Wall-clock fields are ignored, and each run's own folder name is replaced by <RUN> (it appears in BFCL executor
names and sandbox paths). Exit code 0 = identical (or parity holds), 1 = different.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOLATILE = {"seconds"}


SEARCH = [ROOT / "experiments"]


def run_dir(tag: str, kind: str) -> Path:
    hits = sorted((h for d in SEARCH for h in d.glob(f"*golden_{tag}_{kind}_*")), key=lambda h: h.name)
    if not hits:
        raise SystemExit(f"no run folder for golden_{tag}_{kind}")
    return hits[-1]


def load(d: Path) -> dict[str, dict]:
    out = {}
    for p in sorted((d / "transcripts").glob("*.json")):
        text = p.read_text().replace(d.name, "<RUN>")
        safe = "r_" + "".join(c if c.isalnum() or c == "_" else "_" for c in d.name)
        text = text.replace(safe, "<RUN>")
        rec = json.loads(text)
        out[p.stem] = {k: v for k, v in rec.items() if k not in VOLATILE}
    return out


def first_diff(a, b, path="") -> str | None:
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: missing on one side"
            d = first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {str(a)[:120]!r} vs {str(b)[:120]!r}"


def passed(rec: dict) -> bool:
    return bool(rec.get("correct", rec.get("valid")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--unstable", nargs="*", default=[], metavar="ID:N",
                    help="item whose first N messages must match; differences after that are reported, not failed")
    ap.add_argument("--dirs", nargs="*", default=[], help="more folders holding run folders (e.g. a frozen copy's "
                                                          "experiments/)")
    args = ap.parse_args()
    SEARCH.extend(Path(d) for d in args.dirs)
    unstable = {u.split(":")[0]: int(u.split(":")[1]) for u in args.unstable}
    ok = True
    for kind in ("agent", "bfcl"):
        da, db = run_dir(args.a, kind), run_dir(args.b, kind)
        A, B = load(da), load(db)
        if set(A) != set(B):
            print(f"{kind}: different items: {sorted(set(A) ^ set(B))}")
            ok = False
            continue
        pa, pb = sum(passed(r) for r in A.values()), sum(passed(r) for r in B.values())
        diffs = {k: first_diff(A[k], B[k]) for k in A}
        diffs = {k: d for k, d in diffs.items() if d}
        excused = 0
        for k, n in unstable.items():
            if k in diffs:
                pre = first_diff(A[k]["messages"][:n], B[k]["messages"][:n], ".messages")
                if pre:
                    diffs[k] = f"(unstable item, but differs before message {n}) {pre}"
                else:
                    print(f"  {k}: known-unstable, first {n} messages identical; later: {diffs.pop(k)}")
                    excused += 1
        print(f"{kind}: {len(A)} items, passed {pa} vs {pb}, identical {len(A) - len(diffs) - excused}/{len(A)}"
              + (f" + {excused} known-unstable" if excused else ""))
        ta, tb = (sum(r.get("processed_tokens", 0) for r in X.values()) for X in (A, B))
        if ta:
            print(f"  processed tokens {ta} vs {tb} ({(tb - ta) / ta:+.1%})")
        for k, d in diffs.items():
            print(f"  {k}: {d}")
        ok &= (pa == pb) if args.parity else not diffs
    print("SAME" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
