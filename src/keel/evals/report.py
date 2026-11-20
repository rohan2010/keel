"""Report card for a comparison: the same content as Markdown (`report.md`, renders on GitHub) and
as plain text for an 80-column terminal, plus `report.json` (the full comparison).

Order: verdict line first, then one table per arm (pass@1, pass@k / all@k when there are tries, false success,
noise band, tokens, time), the test used, what differs between the runs, per-group changes, the items that flipped,
and the run folders.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from .compare import Comparison, fmt_p

WIDTH = 80
MAX_IDS = 12


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _pts(a, b) -> str:
    return "-" if a is None or b is None else f"{100 * (b - a):+.1f}"


def _num(x, nd=0) -> str:
    return "-" if x is None else (f"{x:,.{nd}f}")


def rows(c: Comparison) -> list[tuple[str, str, str, str]]:
    """(label, A, B, delta) for the arm table."""
    ma, mb = c.metrics.get("a", {}), c.metrics.get("b", {})
    out = [("items x tries", f"{ma.get('items', 0)} x {ma.get('tries', 0)}", f"{mb.get('items', 0)} x {mb.get('tries', 0)}",
            ""),
           ("pass@1", _pct(ma.get("pass1")), _pct(mb.get("pass1")), _pts(ma.get("pass1"), mb.get("pass1")))]
    for k in sorted(set(ma.get("pass_k", {})) & set(mb.get("pass_k", {})), key=int):
        if int(k) > 1:
            out.append((f"pass@{k}", _pct(ma["pass_k"][k]), _pct(mb["pass_k"][k]), _pts(ma["pass_k"][k], mb["pass_k"][k])))
            out.append((f"all {k} tries pass", _pct(ma["all_k"][k]), _pct(mb["all_k"][k]),
                        _pts(ma["all_k"][k], mb["all_k"][k])))
    fa, fb = ma.get("fs") or {}, mb.get("fs") or {}
    if fa and fb:
        out.append(("false success (all)", _pct(fa.get("fs_rate")), _pct(fb.get("fs_rate")),
                    _pts(fa.get("fs_rate"), fb.get("fs_rate"))))
        out.append(("false success (of failed)", _pct(fa.get("fs_of_failed")), _pct(fb.get("fs_of_failed")),
                    _pts(fa.get("fs_of_failed"), fb.get("fs_of_failed"))))
    nb = c.noise or {}
    out.append(("noise band (identical runs)", "unknown" if nb.get("band") is None else
                f"+-{100 * nb['band']:.1f}", f"{nb['flips']}/{nb['n']} flip" if nb.get("band") is not None else "", ""))
    ca, cb = ma.get("cost") or {}, mb.get("cost") or {}
    for key, label, nd in (("prompt_tokens", "prompt tokens / item", 0), ("processed_tokens", "tokens read / item", 0),
                           ("output_tokens", "output tokens / item", 0), ("seconds", "seconds / item", 1)):
        va, vb = (ca.get(key) or {}).get("mean"), (cb.get(key) or {}).get("mean")
        if va is not None or vb is not None:
            d = "-" if va in (None, 0) or vb is None else f"{100 * (vb - va) / va:+.0f}%"
            out.append((label, _num(va, nd), _num(vb, nd), d))
    return out


def test_line(c: Comparison) -> str:
    t = c.pass_test
    if not t:
        return ""
    s = f"{t['test']}, n = {t['n']}" + (" (item, try) pairs" if t.get("pair") == "try" else " items")
    if "only_a" in t:
        s += f": pass only in A {t['only_a']}, only in B {t['only_b']}"
    if "clusters" in t:
        s += f" in {t['clusters']} tasks: {t['moved']} moved ({t['clusters_up']} up, {t['clusters_down']} down)"
    if "ci95" in t:
        s += f": 95% CI [{100 * t['ci95'][0]:+.1f}, {100 * t['ci95'][1]:+.1f}] points"
    s += f", p {fmt_p(t)} ({'one-sided' if c.sided == 'greater' else 'two-sided'}, alpha {c.alpha})"
    if c.fs_test:
        f = c.fs_test
        s += f". False success: {100 * f['delta']:+.1f} points, p {fmt_p(f)}"
    return s


def _ids(xs: list[str]) -> str:
    return ", ".join(xs[:MAX_IDS]) + (f" ... (+{len(xs) - MAX_IDS})" if len(xs) > MAX_IDS else "") if xs else "none"


def group_rows(c: Comparison) -> list[tuple[str, str, str, str]]:
    ga, gb = c.metrics.get("a", {}).get("groups", {}), c.metrics.get("b", {}).get("groups", {})
    return [(g, _pct(ga[g]["pass1"]), _pct(gb[g]["pass1"]), _pts(ga[g]["pass1"], gb[g]["pass1"]))
            for g in sorted(set(ga) & set(gb))]


def arm_label(name: str) -> str:
    """`20261112-222647_m1_mf_noreminder_s0_qwen3.5-4b-8bit (+1)` -> `m1_mf_noreminder_s0 (+1)`."""
    import re
    head, _, more = name.partition(" (+")
    head = re.sub(r"^\d{8}-\d{6}_", "", head)
    head = re.sub(r"_qwen[\w.\-]*$", "", head)
    return head + (f" (+{more}" if more else "")


def _label(c: Comparison) -> str:
    return (f"{c.suite}, {c.tier} tier, {'pre-registered' if c.preregistered else 'exploratory'}"
            + (f", group {c.group}" if c.group else ""))


# ---- Markdown ----------------------------------------------------------------------------------------------------

def markdown(c: Comparison) -> str:
    L = [f"# keel compare: {c.verdict}", "", f"**{c.verdict.upper()}** ({_label(c)}). {c.reason}.", ""]
    for f in c.flags:
        L.append(f"> Note: {f}.")
    if c.flags:
        L.append("")
    L += [f"A = `{arm_label(c.a)}`, B = `{arm_label(c.b)}` (the change under test).", "",
          "| | A | B | B - A |", "|---|---:|---:|---:|"]
    L += [f"| {a} | {b} | {x} | {d} |" for a, b, x, d in rows(c)]
    if c.pass_test:
        L += ["", f"**Test.** {test_line(c)}."]
        if c.identical is not None:
            L.append(f"Same outcome (pass, score, stop, output tokens) on {c.identical} of {c.pass_test['n']} items.")
    L += ["", "## What differs between the runs", ""]
    if c.differences:
        for d in c.differences:
            tag = "allowed" if d["key"] in c.allowed else ("blocks the comparison" if d["blocking"] else "listed")
            L.append(f"- **{d['key']}** ({tag}): `{d['a']}` -> `{d['b']}`")
    else:
        L.append("Nothing in the run headers (same model, backend, batch size, sampler, split, items, code, layers).")
    if c.incomplete:
        L += [f"- **incomplete:** {k} ({v} item(s) never finished)" for k, v in c.incomplete.items()]
    g = group_rows(c)
    if g and not c.group:
        L += ["", "## Per group (pass@1)", "", "| group | A | B | B - A |", "|---|---:|---:|---:|"]
        L += [f"| {a} | {b} | {x} | {d} |" for a, b, x, d in g]
    if c.flips:
        L += ["", "## Items that changed", "", f"- **Fixed** (better in B, {len(c.flips['fixed'])}): "
              f"{_ids(c.flips['fixed'])}", f"- **Broken** (worse in B, {len(c.flips['broken'])}): {_ids(c.flips['broken'])}"]
    L += ["", "## Runs", ""] + [f"- A: `{Path(d).name}`" for d in c.runs.get("a", [])] + \
         [f"- B: `{Path(d).name}`" for d in c.runs.get("b", [])]
    L += ["", "*Made by `keel-core compare`. Rules: `src/keel/evals/compare.py`.*", ""]
    return "\n".join(L)


# ---- terminal ----------------------------------------------------------------------------------------------------

def _wrap(text: str, indent: str = "  ") -> list[str]:
    return textwrap.wrap(text, WIDTH, initial_indent=indent, subsequent_indent=indent, break_on_hyphens=False) or [""]


def _short(name: str, n: int) -> str:
    return name if len(name) <= n else name[: n - 3] + "..."


def _bare(xs: list[str]) -> list[str]:
    """Item ids without a prefix they all share (BFCL's `multi_turn_`), for the terminal."""
    return [x.removeprefix("multi_turn_") for x in xs]


def terminal(c: Comparison) -> str:
    L = [_short(f"{c.verdict.upper()}  ({_label(c)})", WIDTH)]
    L += _wrap(c.reason + ".")
    L += [f"  A = {_short(c.a, WIDTH - 6)}", f"  B = {_short(c.b, WIDTH - 6)}", ""]
    w = [28, 14, 14, 12]
    L.append(f"  {'':{w[0]}}{'A':>{w[1]}}{'B':>{w[2]}}{'B - A':>{w[3]}}")
    for a, b, x, d in rows(c):
        L.append(f"  {_short(a, w[0] - 1):{w[0]}}{b:>{w[1]}}{x:>{w[2]}}{d:>{w[3]}}")
    if c.pass_test:
        L += [""] + _wrap("Test: " + test_line(c) + ".")
    for d in c.differences:
        tag = "allowed" if d["key"] in c.allowed else ("BLOCKS" if d["blocking"] else "listed")
        L += _wrap(f"[{tag}] {d['key']}: {d['a']} -> {d['b']}")
    for k, v in c.incomplete.items():
        L += _wrap(f"[BLOCKS] incomplete: {k} ({v} item(s) never finished)")
    g = group_rows(c)
    if g and not c.group:
        L += ["", f"  {'group':{w[0]}}{'A':>{w[1]}}{'B':>{w[2]}}{'B - A':>{w[3]}}"]
        L += [f"  {_short(a, w[0] - 1):{w[0]}}{b:>{w[1]}}{x:>{w[2]}}{d:>{w[3]}}" for a, b, x, d in g]
    if c.flips:
        L += [""] + _wrap(f"Fixed ({len(c.flips['fixed'])}): {_ids(_bare(c.flips['fixed']))}")
        L += _wrap(f"Broken ({len(c.flips['broken'])}): {_ids(_bare(c.flips['broken']))}")
    for f in c.flags:
        L += _wrap(f"note: {f}")
    return "\n".join(L)


def write(c: Comparison, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(markdown(c))
    (out_dir / "report.json").write_text(json.dumps(c.to_dict(), indent=1, default=str))
    return out_dir
