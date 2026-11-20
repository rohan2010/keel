"""Noise band: how much two identical runs disagree.

Two runs of the same configuration still differ (sampling, and under greedy decoding batched float math). For a
pair of identical runs we store the items whose pass/fail flipped. From `f` flips over `n` items:

    flip_rate = f / n
    band      = 2 * sqrt(f) / n     the net change (B - A) two identical runs reach about 95% of the time, if the
                                    flips go either way at random. The research's "28 of 120 flip (+-9 points)".

Bands live in `experiments/noise/<key>.json`, one file per configuration. The key is everything that changes
outputs except the try seed (suite, split, item set, model, backend, batch size, sampler, rng, layers, limits,
library versions), so a band is only used for the configuration it was measured on. `keel-core eval --repeat 2`
records one automatically; `keel-core noise A B` records one from two existing runs.
"""

from __future__ import annotations

import hashlib
import json
import time
from math import sqrt
from pathlib import Path

from .compare import Arm, header_diffs
from .runs import RunHeader

ENV_KEYS = ("mlx", "mlx_lm", "python", "bfcl_eval", "transformers")


def _noise_dir() -> Path:
    from ..runlog import EXPERIMENTS
    return EXPERIMENTS / "noise"


def config_of(h: RunHeader) -> dict:
    return {"suite": h.suite, "split": h.split, "items": h.items_sha1, "model": h.model, "backend": h.backend,
            "batch_size": h.batch_size, "sampling": {k: v for k, v in h.sampling.items() if k != "seed"},
            "rng": h.rng, "layers": h.layers, "limits": h.limits,
            "env": {k: v for k, v in h.env.items() if k in ENV_KEYS}}


def config_key(h: RunHeader) -> str:
    return hashlib.sha1(json.dumps(config_of(h), sort_keys=True, default=str).encode()).hexdigest()[:12]


def band_of(flips: int, n: int) -> float:
    return 2 * sqrt(flips) / n if n else 0.0


def measure(A: Arm, B: Arm, allow: tuple[str, ...] = ()) -> dict:
    """Flips between two runs of one configuration (one try per item, or matched tries)."""
    blocking = [d for d in header_diffs(A.header, B.header, ignore_seed=True)
                if d["blocking"] and d["key"] not in allow]
    listed_layers = [d for d in header_diffs(A.header, B.header) if d["key"] in ("layers", "limits", "template")]
    if blocking or listed_layers:
        why = ", ".join(d["key"] for d in blocking + listed_layers)
        raise ValueError(f"not two runs of one configuration (differ in {why}); a noise band needs identical runs")
    useful = (lambda r: r.kind != "safety") if A.records and A.records[0].suite == "agent" else (lambda r: True)
    a = {(r.item_id, r.try_): r.correct for r in A.records if useful(r)}
    b = {(r.item_id, r.try_): r.correct for r in B.records if useful(r)}
    units = sorted(set(a) & set(b))
    flips = [f"{i}#{t}" if t else i for i, t in units if a[(i, t)] != b[(i, t)]]
    n = len(units)
    return {"runs": [[Path(h.dir).name for h in A.headers], [Path(h.dir).name for h in B.headers]],
            "n": n, "flips": len(flips), "flip_rate": len(flips) / n if n else 0.0, "band": band_of(len(flips), n),
            "net": (sum(b[u] for u in units) - sum(a[u] for u in units)) / n if n else 0.0,
            "flipped": flips, "allowed": list(allow), "measured": time.strftime("%Y-%m-%d %H:%M:%S")}


def record(A: Arm, B: Arm, allow: tuple[str, ...] = (), base: Path | None = None) -> dict:
    """Measure and store under the configuration's key (both keys if an allowed difference splits them)."""
    m = measure(A, B, allow)
    base = base or _noise_dir()
    base.mkdir(parents=True, exist_ok=True)
    for h in {config_key(A.header): A.header, config_key(B.header): B.header}.values():
        p = base / f"{config_key(h)}.json"
        doc = json.loads(p.read_text()) if p.exists() else {"key": config_key(h), "config": config_of(h),
                                                          "measurements": []}
        if not any(x["runs"] == m["runs"] for x in doc["measurements"]):
            doc["measurements"].append(m)
        p.write_text(json.dumps(doc, indent=1, default=str))
    return m


def lookup(h: RunHeader, base: Path | None = None) -> dict | None:
    """The stored band for this run's configuration: pooled over every measurement (flips and items summed)."""
    p = (base or _noise_dir()) / f"{config_key(h)}.json"
    if not p.exists():
        return None
    ms = json.loads(p.read_text())["measurements"]
    if not ms:
        return None
    f, n = sum(m["flips"] for m in ms), sum(m["n"] for m in ms)
    return {"key": config_key(h), "measurements": len(ms), "flips": f, "n": n, "flip_rate": f / n if n else 0.0,
            "band": band_of(f, n) * sqrt(len(ms)) if n else 0.0}
