"""Metrics over item records. One library for every suite; `keel compare` and the report card use it.

Input: `ItemRecord`s (keel.evals.runs), one per item and try, from one run or several (tries of the same arm).
Items are matched by `item_id`; an item's tries are its records.

    pass@1        mean over items of c / n                        (c passes in n tries)
    pass@k        mean over items of 1 - C(n-c, k) / C(n, k)       unbiased (Chen et al. 2021); needs n >= k
    all@k         mean over items of C(c, k) / C(n, k)             all k tries pass (tau-bench's pass^k); unbiased
    never         share of items with c = 0                        (needs every item to have the same n)
    false success over every labelled try: FS / labelled (headline) and FS / failed (how failures are reported),
                  labels from keel.false_success (None = outside the measure, not counted)
    cost          mean prompt / processed / output tokens and seconds per item-try (missing values skipped, counted)

`useful=True` (agent default, as in the research's "overall_useful") leaves out the safety task kind.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from math import comb

from .runs import ItemRecord


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one item: chance that at least one of k tries drawn from the n passes."""
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} tries per item (have {n})")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def all_at_k(n: int, c: int, k: int) -> float:
    """Unbiased chance that all k tries drawn from the n pass."""
    if k > n:
        raise ValueError(f"all@{k} needs at least {k} tries per item (have {n})")
    return comb(c, k) / comb(n, k)


@dataclass
class Metrics:
    items: int
    tries: int                         # tries per item (min over items; equal for a clean run)
    pass1: float
    pass_k: dict[int, float] = field(default_factory=dict)
    all_k: dict[int, float] = field(default_factory=dict)
    never: float | None = None
    fs: dict = field(default_factory=dict)       # n, pass, false_success, honest_fail, no_answer, fs_rate, fs_of_failed
    cost: dict = field(default_factory=dict)     # mean per item-try, and how many had a value
    groups: dict[str, "Metrics"] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["groups"] = {g: m.to_dict() for g, m in self.groups.items()}
        return d


def _by_item(recs: list[ItemRecord]) -> dict[str, list[ItemRecord]]:
    out: dict[str, list[ItemRecord]] = defaultdict(list)
    for r in recs:
        out[r.item_id].append(r)
    return dict(out)


def _fs(recs: list[ItemRecord]) -> dict:
    from ..false_success import summarize
    labels = [r.fs_label for r in recs if r.fs_label is not None]
    return summarize(labels) if labels else {}


def _cost(recs: list[ItemRecord]) -> dict:
    out = {}
    for k in ("prompt_tokens", "processed_tokens", "output_tokens", "seconds"):
        vals = [getattr(r, k) for r in recs if getattr(r, k) is not None]
        out[k] = {"mean": sum(vals) / len(vals) if vals else None, "n": len(vals)}
    return out


def compute(recs: list[ItemRecord], ks: tuple[int, ...] = (), *, useful: bool | None = None,
            by_group: bool = True) -> Metrics:
    """Metrics for a set of records. `ks`: the k values for pass@k / all@k (default: every k up to the tries per
    item). `useful`: drop agent safety items (default True for agent records)."""
    if useful is None:
        useful = bool(recs) and recs[0].suite == "agent"
    if useful:
        recs = [r for r in recs if r.kind != "safety"]
    items = _by_item(recs)
    if not items:
        return Metrics(0, 0, 0.0)
    ns = {i: len(v) for i, v in items.items()}
    cs = {i: sum(r.correct for r in v) for i, v in items.items()}
    n_min = min(ns.values())
    ks = tuple(ks) or tuple(range(1, n_min + 1))
    m = Metrics(items=len(items), tries=n_min,
                pass1=sum(cs[i] / ns[i] for i in items) / len(items),
                pass_k={k: sum(pass_at_k(ns[i], cs[i], k) for i in items) / len(items) for k in ks if k <= n_min},
                all_k={k: sum(all_at_k(ns[i], cs[i], k) for i in items) / len(items) for k in ks if k <= n_min},
                never=(sum(cs[i] == 0 for i in items) / len(items)) if len(set(ns.values())) == 1 else None,
                fs=_fs(recs), cost=_cost(recs))
    if by_group:
        groups: dict[str, list[ItemRecord]] = defaultdict(list)
        for r in recs:
            groups[r.group or "-"].append(r)
        if len(groups) > 1:
            m.groups = {g: compute(v, ks, useful=False, by_group=False) for g, v in sorted(groups.items())}
    return m
