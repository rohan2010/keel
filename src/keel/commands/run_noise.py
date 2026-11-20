"""keel-core noise: record or list noise bands (how far two identical runs disagree).

  uv run keel-core noise <runA> <runB> [--allow rng]     record a band from two runs of one configuration
  uv run keel-core noise --list                           every stored band

`keel-core eval --repeat 2` records one by itself. `keel-core compare` shows the band of the baseline's config.
"""

from __future__ import annotations

import argparse
import json

from keel.evals import noise
from keel.evals.compare import load_arm


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("a", nargs="?", help="a run folder (or tries joined by commas)")
    ap.add_argument("b", nargs="?", help="a second run of the same configuration")
    ap.add_argument("--allow", nargs="*", default=[], metavar="KEY",
                    help="header differences to accept (e.g. rng for research repeats); recorded with the band")
    ap.add_argument("--list", action="store_true")


def show(m: dict) -> str:
    return (f"{m['flips']} of {m['n']} flipped ({100 * m['flip_rate']:.1f}%): band +-{100 * m['band']:.1f} points; "
            f"net {100 * m.get('net', 0):+.1f}")


def run(a: argparse.Namespace):
    if a.list:
        base = noise._noise_dir()
        files = sorted(base.glob("*.json")) if base.exists() else []
        if not files:
            print("no noise bands stored yet (keel-core eval --repeat 2, or keel-core noise A B)")
        for p in files:
            doc = json.loads(p.read_text())
            cfg = doc["config"]
            print(f"{doc['key']}  {cfg['suite']} {cfg['split']} batch {cfg['batch_size']} "
                  f"T {cfg['sampling'].get('temperature')} {cfg['backend']}  ({len(doc['measurements'])} measurement(s))")
            for m in doc["measurements"]:
                print(f"    {show(m)}  {' vs '.join(','.join(r) for r in m['runs'])}")
        return None
    if not (a.a and a.b):
        raise SystemExit("noise needs two runs (or --list)")
    try:
        m = noise.record(load_arm(a.a, fs=False), load_arm(a.b, fs=False), tuple(a.allow))
    except ValueError as err:
        raise SystemExit(str(err)) from None
    print(show(m))
    return m
