"""keel-core prereg: freeze a pre-registration before the runs exist.

  uv run keel-core prereg plans/reminder-keep.yaml      freeze it (text + sha256 + time in experiments/prereg/)
  uv run keel-core prereg --list                         every frozen pre-registration
  uv run keel-core compare A B --prereg reminder-keep    use it: the verdict is labelled pre-registered only if
                                                         the runs match (else exploratory, with every reason)
Format: see keel.evals.prereg.
"""

from __future__ import annotations

import argparse
import json

from keel.evals import prereg


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("file", nargs="?", help="the pre-registration YAML to freeze")
    ap.add_argument("--list", action="store_true")


def run(a: argparse.Namespace):
    if a.list:
        base = prereg._dir()
        files = sorted(base.glob("*.json")) if base.exists() else []
        if not files:
            print("no pre-registrations frozen yet")
        for f in files:
            r = json.loads(f.read_text())
            p = r["plan"]
            print(f"{r['name']:28} frozen {r['frozen']}  {p['suite']} {p['split']}  {p['primary']}, "
                  f"{p['sided']}-sided, alpha {p['alpha']}  sha256 {r['sha256'][:12]}")
        return None
    if not a.file:
        raise SystemExit("prereg needs a YAML file (or --list)")
    try:
        r = prereg.freeze(a.file)
    except ValueError as err:
        raise SystemExit(f"not frozen: {err}") from None
    print(f"frozen {r['name']} at {r['frozen']} (sha256 {r['sha256'][:12]}). Runs must start after this time to be "
          "compared as pre-registered.")
    return r
