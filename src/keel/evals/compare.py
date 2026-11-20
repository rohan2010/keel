"""Compare two arms and give a verdict.

An arm is one run, or several runs of the same configuration (tries; e.g. research `_s0,_s1,_s2`, or a keel eval
manifest). `compare_arms(A, B)` answers, in this order:

1. **Can these be compared?** Headers must agree on model, backend, batch size, sampler (except a try's seed), rng,
   split and item set, and both arms must be complete, and on library versions (`env`): batched float math changes
   greedy answers, so each of these alone can move results. Every difference is listed. Code hash, layers and limits
   are listed, never blocking (usually the change under test). `allow` names differences the caller accepts on
   purpose; they are listed as allowed.
2. **Paired test on pass/fail** (B minus A): exact McNemar when every item has one try in each arm, else a paired
   bootstrap over items (10,000 resamples, fixed seed) of the per-item pass-rate difference.
3. **The same test on false success** (always run alongside; our sharpest finding is that layers which raise the pass
   rate also raise false "done" claims).
4. **Verdict**: real gain | real loss | real gain, more false "done" | no detectable effect | can't compare. A
   no-effect verdict also gives the smallest effect this sample could have detected, so it isn't read as "equal".
   With a stored noise band (P5), a delta inside it is flagged.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from math import comb, sqrt
from pathlib import Path

from . import metrics as M
from .runs import ItemRecord, RunHeader, eval_meta, load_run

BLOCKING = ("model", "backend", "batch_size", "sampling", "rng", "split", "items", "env")
LISTED = ("code", "layers", "limits", "template", "prompt")   # template = chat-template switches (thinking on /
                                                    # off); prompt = hash of the recorded system prompt (an
                                                    # undocumented prompt sentence moved a task 0 -> 4/4)
VERDICTS = ("real gain", "real loss", "real gain, more false \"done\"", "no detectable effect", "can't compare",
            "fewer false \"done\"",                    # primary = false_success, B's rate lower
            "not tested")                              # fixed sequence: the hypothesis before it was not confirmed


# ---- loading arms ------------------------------------------------------------------------------------------------

@dataclass
class Arm:
    name: str
    headers: list[RunHeader]
    records: list[ItemRecord]
    seeds: dict[int, object] = field(default_factory=dict)   # try -> sampler seed

    @property
    def header(self) -> RunHeader:
        return self.headers[0]


def _resolve(spec: str) -> list[Path]:
    """A run folder, an eval manifest (experiments/evals/<id>.json), or several run folders joined by commas."""
    p = Path(spec)
    if p.suffix == ".json" and p.is_file():
        m = json.loads(p.read_text())
        base = p.resolve().parent.parent                 # experiments/
        return [base / r["run"] for r in m["runs"]]
    return [Path(s) for s in spec.split(",")]


def load_arm(spec: str, name: str | None = None, fs: bool = True) -> Arm:
    dirs = _resolve(spec)
    loaded = [load_run(d, fs=fs) for d in dirs]
    hs = [h for h, _ in loaded]
    for h in hs[1:]:                                     # the runs of one arm differ only in a try's seed
        diff = [d for d in header_diffs(hs[0], h, ignore_seed=True) if d["key"] in BLOCKING]
        if diff:
            raise ValueError(f"{h.name} is not another try of {hs[0].name}: differs in "
                             f"{', '.join(d['key'] for d in diff)}")
    recs, seeds = [], {}
    for i, (d, (h, rs)) in enumerate(zip(dirs, loaded, strict=True)):
        if len(dirs) > 1 and "try" not in eval_meta(d):   # research tries are separate runs: their order is the try
            for r in rs:
                r.try_ = i
        for t in {r.try_ for r in rs}:
            seeds[t] = h.sampling.get("seed")
        recs += rs
    return Arm(name or (hs[0].name if len(hs) == 1 else f"{hs[0].name} (+{len(hs) - 1})"), hs, recs, seeds)


def try_seeds(arm: Arm) -> dict[int, object]:
    """try -> sampler seed (for pairing try t of A with try t of B)."""
    return arm.seeds or {t: arm.header.sampling.get("seed") for t in {r.try_ for r in arm.records}}


# ---- 1. compatibility --------------------------------------------------------------------------------------------

def header_diffs(a: RunHeader, b: RunHeader, ignore_seed: bool = False) -> list[dict]:
    out = []
    for key in BLOCKING + LISTED:
        if key == "items":
            va, vb = a.items_sha1, b.items_sha1
        elif key == "sampling" and ignore_seed:
            va, vb = ({k: v for k, v in h.sampling.items() if k != "seed"} for h in (a, b))
        elif key == "model":                             # thinking on / off is the change under test,
            va, vb = ({k: v for k, v in h.model.items() if k != "chat_template_kwargs"} for h in (a, b))   # not
        elif key == "template":                          # another model: listed, never blocking
            va, vb = ((h.model.get("chat_template_kwargs") or {}) for h in (a, b))
        elif key == "env":
            va, vb = ({k: v for k, v in h.env.items() if k in ("mlx", "mlx_lm", "python", "bfcl_eval", "transformers")}
                      for h in (a, b))
        else:
            va, vb = getattr(a, key), getattr(b, key)
        if va != vb:
            if isinstance(va, dict) and isinstance(vb, dict):
                ks = sorted(k for k in set(va) | set(vb) if va.get(k) != vb.get(k))
                va, vb = {k: va.get(k) for k in ks}, {k: vb.get(k) for k in ks}
            out.append({"key": key, "a": va, "b": vb, "blocking": key in BLOCKING})
    return out


# ---- 2/3. paired tests -------------------------------------------------------------------------------------------

def mcnemar_exact(b: int, c: int, sided: str = "two") -> float:
    """Exact McNemar on the discordant pairs: b = pass only in A, c = pass only in B. `sided="greater"` tests
    B > A (pre-registered one-sided)."""
    n = b + c
    if n == 0:
        return 1.0
    if sided == "greater":
        return sum(comb(n, i) for i in range(c, n + 1)) / 2 ** n
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def mcnemar_mde(n_items: int, discordant: int, alpha: float = 0.05) -> float | None:
    """Smallest |delta| (fraction of items) this sample could call significant: with the discordant pairs seen, the
    smallest one-sided split that would pass; and never less than the fewest one-way flips that could pass at all
    (so 0 flips seen does not read as "can't detect anything"). None if even every item flipping one way would not
    be significant."""
    floor = next((m for m in range(1, n_items + 1) if mcnemar_exact(0, m) < alpha), None)
    if floor is None:
        return None
    for gap in range(0, discordant + 1):
        if (discordant - gap) % 2:
            continue
        lo = (discordant - gap) // 2
        if mcnemar_exact(lo, discordant - lo) < alpha:
            return max(gap, floor) / n_items
    return floor / n_items if floor > discordant else None


def paired_test(a: dict[str, list[bool]], b: dict[str, list[bool]], alpha: float, sided: str = "two",
                resamples: int = 10000, seed: int = 0) -> dict:
    """a, b: item -> list of 0/1 outcomes (tries). Exact McNemar for one try each, else paired bootstrap."""
    ids = sorted(set(a) & set(b))
    n = len(ids)
    if n == 0:
        return {"test": "none", "n": 0, "delta": 0.0, "p": 1.0, "significant": False, "mde": None}
    single = all(len(a[i]) == 1 and len(b[i]) == 1 for i in ids)
    ma = {i: sum(a[i]) / len(a[i]) for i in ids}
    mb = {i: sum(b[i]) / len(b[i]) for i in ids}
    delta = sum(mb[i] - ma[i] for i in ids) / n
    if single:
        only_a = sum(a[i][0] and not b[i][0] for i in ids)
        only_b = sum(b[i][0] and not a[i][0] for i in ids)
        p = mcnemar_exact(only_a, only_b, sided)
        return {"test": "exact McNemar" + (" (one-sided)" if sided == "greater" else ""), "n": n, "delta": delta,
                "only_a": only_a, "only_b": only_b, "p": p, "significant": p < alpha,
                "mde": mcnemar_mde(n, only_a + only_b, alpha)}
    rng = random.Random(seed)
    diffs = [mb[i] - ma[i] for i in ids]
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    lo, hi = boots[int(resamples * alpha / 2)], boots[int(resamples * (1 - alpha / 2)) - 1]
    le, ge = sum(x <= 0 for x in boots) / resamples, sum(x >= 0 for x in boots) / resamples
    p = le if sided == "greater" else min(1.0, 2 * min(le, ge))
    p = max(p, 1 / resamples)                            # resolution of the bootstrap: shown as "< 1/resamples"
    mean = sum(diffs) / n
    se = sqrt(sum((d - mean) ** 2 for d in diffs) / max(n - 1, 1) / n)
    sig = p < alpha            # decided by p, one- or two-sided as asked (audit #12: one-sided p was judged against the
                               # two-sided interval's alpha/2 edge, so near the threshold `p` and `significant` disagreed)
    return {"test": "paired bootstrap" + (" (one-sided)" if sided == "greater" else ""), "n": n, "delta": delta,
            "ci95": [lo, hi], "p": p, "significant": sig, "mde": 2.8 * se,      # ~80% power at alpha 0.05
            "resamples": resamples, "seed": seed}


def cluster_test(a: dict[str, list[bool]], b: dict[str, list[bool]], groups: dict[str, str], alpha: float,
                 sided: str = "two") -> dict:
    """Task-clustered sign-flip permutation test. a, b: unit -> 0/1 outcomes; groups: unit -> cluster
    (agent task, BFCL category). Each cluster's difference is d_g = mean(B) - mean(A) over its units; the statistic
    is sum(d_g). Under "no effect" each d_g is as likely positive as negative, so p is the share of the 2^k sign
    patterns of the k clusters that moved whose sum is at least as extreme. Always exact: the d_g are fractions, so
    the sum's null distribution is counted on integers (one pass per cluster), with no Monte Carlo at any k (release
    audit #11, 2026-11-19: above 20 moved clusters it used to be Monte Carlo, seed-dependent, without the +1 term).
    Seeds of one task are not independent: this treats the task, not the episode, as the unit."""
    from fractions import Fraction
    from math import lcm
    ids = sorted(set(a) & set(b))
    by: dict[str, list[str]] = {}
    for i in ids:
        by.setdefault(groups.get(i, ""), []).append(i)

    def diff(i: str) -> Fraction:
        return Fraction(sum(b[i]), len(b[i])) - Fraction(sum(a[i]), len(a[i]))

    d = {g: sum((diff(i) for i in us), Fraction(0)) / len(us) for g, us in by.items()}
    n = len(ids)
    delta = float(sum((diff(i) for i in ids), Fraction(0)) / n) if n else 0.0
    moved = [abs(v) for v in d.values() if v != 0]
    k = len(moved)
    p = 1.0
    if k:
        scale = lcm(*(m.denominator for m in moved))
        mags = [int(m * scale) for m in moved]
        obs = int(sum(d.values()) * scale)           # exact: every d_g is a multiple of 1/scale
        counts = {0: 1}                               # null distribution of sum(+-m): sum -> number of sign patterns
        for m in mags:
            nxt: dict[int, int] = {}
            for t, c in counts.items():
                nxt[t + m] = nxt.get(t + m, 0) + c
                nxt[t - m] = nxt.get(t - m, 0) + c
            counts = nxt
        hits = sum(c for t, c in counts.items() if (t >= obs if sided == "greater" else abs(t) >= abs(obs)))
        p = min(1.0, hits / 2 ** k)
    up = sum(v > 0 for v in d.values())
    return {"test": "task-clustered sign-flip permutation" + (" (one-sided)" if sided == "greater" else ""),
            "n": n, "clusters": len(by), "moved": k, "clusters_up": up, "clusters_down": k - up, "delta": delta,
            "p": p, "significant": p < alpha, "exact": True, "mde": None,
            "cluster_deltas": {g: round(float(v), 4) for g, v in sorted(d.items())}}


def fmt_p(t: dict) -> str:
    """'= 0.039', or '< 0.0001' when a bootstrap p is at its resolution."""
    if t.get("resamples") and t["p"] <= 1 / t["resamples"]:
        return f"< {1 / t['resamples']:g}"
    return f"= {t['p']:.3g}"


def _outcomes(arm: Arm, fn, pair: str = "item") -> dict[str, list[bool]]:
    """unit -> outcomes. pair="item": an item's tries together (bootstrap if several); pair="try": each (item, try)
    is its own unit (try t of A against try t of B, same seed), so the exact test applies."""
    out: dict[str, list[bool]] = {}
    for r in arm.records:
        v = fn(r)
        if v is not None:
            key = r.item_id if pair == "item" else f"{r.item_id}#{r.try_}"
            out.setdefault(key, []).append(bool(v))
    return out


def _useful(arm: Arm) -> Arm:
    if arm.records and arm.records[0].suite == "agent":
        return Arm(arm.name, arm.headers, [r for r in arm.records if r.kind != "safety"], arm.seeds)
    return arm


# ---- 4. verdict --------------------------------------------------------------------------------------------------

@dataclass
class Comparison:
    a: str
    b: str
    suite: str
    tier: str
    verdict: str
    reason: str
    preregistered: bool
    alpha: float
    sided: str
    differences: list[dict] = field(default_factory=list)
    allowed: list[str] = field(default_factory=list)
    incomplete: dict = field(default_factory=dict)
    pass_test: dict = field(default_factory=dict)       # the deciding test
    item_test: dict = field(default_factory=dict)       # with cluster="task": the item-level test, shown beside
    fs_test: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)         # a / b: metrics.compute(...).to_dict()
    identical: int | None = None                        # items with the same pass, score, stop and output tokens in
                                                        # both (one try each): a cheap proxy for identical episodes
    group: str | None = None                            # compared on this group only (BFCL category, agent task)
    flips: dict = field(default_factory=dict)           # fixed / broken units (pass rate up / down, B vs A)
    runs: dict = field(default_factory=dict)            # a / b: run folders
    noise: dict | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _same_item(a: ItemRecord, b: ItemRecord) -> bool:
    return (a.correct, a.score, a.stop, a.output_tokens) == (b.correct, b.score, b.stop, b.output_tokens)


def compare_arms(A: Arm, B: Arm, *, alpha: float = 0.05, sided: str = "two", allow: tuple[str, ...] = (),
                 preregistered: bool = False, noise_band: dict | None = None, group: str | None = None,
                 pair: str = "item", cluster: str | None = None, primary: str = "pass@1") -> Comparison:
    """cluster="task": the deciding pass test is the task-clustered permutation (clusters = record groups: agent
    task, BFCL category); the item-level test is kept beside it in `item_test`, never deciding."""
    if pair not in ("item", "try"):
        raise ValueError("pair is 'item' or 'try'")
    if cluster not in (None, "task"):
        raise ValueError("cluster is None or 'task'")
    if group is not None:                                # e.g. one BFCL category, as a published result was
        A, B = (Arm(x.name, x.headers, [r for r in x.records if r.group == group], x.seeds) for x in (A, B))
        if not A.records or not B.records:
            raise ValueError(f"no records in group {group!r}")
    ha, hb = A.header, B.header
    diffs = header_diffs(ha, hb)
    blocking = [d for d in diffs if d["blocking"] and d["key"] not in allow]
    if ha.suite != hb.suite:
        blocking.insert(0, {"key": "suite", "a": ha.suite, "b": hb.suite, "blocking": True})
    incomplete = {arm.name: sum(len(h.missing) for h in arm.headers) for arm in (A, B)
                  if any(h.missing for h in arm.headers)}
    c = Comparison(A.name, B.name, ha.suite, ha.tier, "can't compare", "", preregistered, alpha, sided,
                   differences=diffs, allowed=[d["key"] for d in diffs if d["blocking"] and d["key"] in allow],
                   incomplete=incomplete, noise=noise_band, group=group,
                   runs={"a": [h.dir for h in A.headers], "b": [h.dir for h in B.headers]})
    ua, ub = _useful(A), _useful(B)
    c.metrics = {"a": M.compute(ua.records, useful=False).to_dict(), "b": M.compute(ub.records, useful=False).to_dict()}
    if pair == "try" and try_seeds(A) != try_seeds(B):
        blocking.append({"key": "try seeds", "a": try_seeds(A), "b": try_seeds(B), "blocking": True})
    if blocking or incomplete:
        parts = [f"{d['key']} differs ({d['a']} vs {d['b']})" for d in blocking]
        parts += [f"{k} is incomplete ({v} item(s) never finished)" for k, v in incomplete.items()]
        c.reason = "; ".join(parts)
        return c
    c.pass_test = paired_test(_outcomes(ua, lambda r: r.correct, pair), _outcomes(ub, lambda r: r.correct, pair),
                              alpha, sided)
    c.pass_test["pair"] = pair
    oa, ob = _outcomes(ua, lambda r: r.correct, pair), _outcomes(ub, lambda r: r.correct, pair)
    if cluster == "task":
        groups = {(r.item_id if pair == "item" else f"{r.item_id}#{r.try_}"): r.group for r in ua.records + ub.records}
        if len(set(groups.values())) < 2:
            raise ValueError("cluster='task' needs records from at least two tasks / groups")
        c.item_test = c.pass_test
        c.pass_test = cluster_test(oa, ob, groups, alpha, sided)
        c.pass_test["pair"] = pair
    rate = {k: (sum(oa[k]) / len(oa[k]), sum(ob[k]) / len(ob[k])) for k in sorted(set(oa) & set(ob))}
    c.flips = {"fixed": [k for k, (x, y) in rate.items() if y > x], "broken": [k for k, (x, y) in rate.items() if y < x]}
    fs_fn = (lambda r: None if r.fs_label is None else r.fs_label == "false_success")
    fa, fb = _outcomes(ua, fs_fn, pair), _outcomes(ub, fs_fn, pair)
    if fa and fb:
        c.fs_test = paired_test(fa, fb, alpha, "two")
    if all(len(v) == 1 for v in _outcomes(ua, lambda r: r.correct).values()):
        ra = {r.item_id: r for r in ua.records}
        c.identical = sum(_same_item(ra[r.item_id], r) for r in ub.records if r.item_id in ra)
    t = c.pass_test
    pts = f"{100 * t['delta']:+.1f} points"
    if t["significant"] and t["delta"] > 0:
        fs_up = bool(c.fs_test) and c.fs_test["significant"] and c.fs_test["delta"] > 0
        c.verdict = VERDICTS[2] if fs_up else VERDICTS[0]
        c.reason = f"{pts}, {t['test']} p {fmt_p(t)}" + (
            f"; false success also rose {100 * c.fs_test['delta']:+.1f} points (p {fmt_p(c.fs_test)})"
            if fs_up else "")
    elif t["significant"]:
        c.verdict, c.reason = VERDICTS[1], f"{pts}, {t['test']} p {fmt_p(t)}"
    else:
        mde = t.get("mde")
        c.verdict = VERDICTS[3]
        if "clusters" in t:
            c.reason = (f"{pts}, {t['test']} p {fmt_p(t)}; {t['moved']} of {t['clusters']} tasks moved "
                        f"({t['clusters_up']} up, {t['clusters_down']} down)")
        else:
            c.reason = (f"{pts}, {t['test']} p {fmt_p(t)}; " +
                        (f"this sample could only detect about {100 * mde:.1f} points or more" if mde is not None
                         else "this sample is too small to detect any effect"))
    if c.item_test:
        it = c.item_test
        c.flags.append(f"deciding test is task-clustered; item-level {it['test']} beside it (not deciding): "
                       f"p {fmt_p(it)}" + (" would pass" if it["significant"] else ""))
    if primary == "false_success":                       # the deciding test is "B makes fewer false done claims"
        if not (fa and fb):
            raise ValueError("primary false_success needs false-success labels in both arms")
        # one-sided B < A on the false-success indicator = "greater" with the arms swapped
        if cluster == "task":
            groups = {(r.item_id if pair == "item" else f"{r.item_id}#{r.try_}"): r.group for r in ua.records + ub.records}
            d = cluster_test(fb, fa, groups, alpha, "greater" if sided == "greater" else "two")
        else:
            d = paired_test(fb, fa, alpha, "greater" if sided == "greater" else "two")
        d = {**d, "delta": -d["delta"], "metric": "false_success", "pair": pair}
        c.item_test, c.pass_test = c.pass_test, d        # the pass test stays on the card, not deciding
        ok = d["significant"] and d["delta"] < 0
        c.verdict = VERDICTS[5] if ok else VERDICTS[3]
        c.reason = (f"false success {100 * d['delta']:+.1f} points (B minus A), {d['test']} p {fmt_p(d)}; "
                    f"pass rate {pts} (not deciding)")
    f = c.fs_test
    if f and f["significant"] and c.verdict not in (VERDICTS[2], VERDICTS[5]):   # a real change in false "done" claims is always said
        c.reason += (f"; false success {'rose' if f['delta'] > 0 else 'fell'} {100 * f['delta']:+.1f} points "
                     f"(p {fmt_p(f)})")
    if noise_band and noise_band.get("band") is not None:
        band = noise_band["band"]
        if abs(t["delta"]) <= band:
            c.flags.append(f"delta inside the noise band: identical runs of this config move by up to "
                           f"+-{100 * band:.1f} points ({noise_band['flips']} of {noise_band['n']} items flipped)")
    elif noise_band is None:
        c.flags.append("noise band unknown for this config")
    return c
