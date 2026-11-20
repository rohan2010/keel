"""Scorer checks for Module 5's long tasks beyond oracle 100% / do-nothing 0% (tests/checks/check_agent_tasks.py --long):
for each task, a realistic MISTAKE applied after the oracle must fail, and a correct answer in a DIFFERENT FORMAT
must pass. Lesson from the 2026-11-10 scorer audit: a scorer can pass wrong work (F1) or fail right work (A4).

  uv run python tests/checks/check_long_tasks.py
"""

import json
import re
import sys
import tempfile
from pathlib import Path

from keel.evals import tasks as T

ok = True


def check(name, got, want):
    global ok
    good = got == want
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name}: got {got}, want {want}")


def solve(task, seed, mutate=None, answer=None):
    it = T.build(task, seed)
    with tempfile.TemporaryDirectory() as tmp:
        w = T.materialize(it, Path(tmp) / "w")
        ans = T.oracle(it, w)
        if mutate:
            ans = mutate(it, w) or ans
        return T.score(it, w, answer if answer is not None else ans, [])["correct"]


def edit(w, rel, fn):
    p = w / rel
    p.write_text(fn(p.read_text()))


for seed in (0, 1, 2):
    # L01: keep a duplicate / skip index entry / alternative index layout (table)
    def l01_dup(it, w):
        f = next(p for p in sorted(w.rglob("*")) if p.is_file() and p.name != "index.md")
        (f.parent / ("Copy of " + f.name)).write_bytes(f.read_bytes())
    check(f"L01_{seed} duplicate kept: wrong", solve("L01", seed, l01_dup), False)
    check(f"L01_{seed} index as a table: right", solve("L01", seed, lambda it, w: (w / "index.md").write_text(
        "| file | folder |\n|---|---|\n" + "".join(f"| {p.name} | {p.parent.name} |\n" for p in w.rglob("*")
                                                    if p.is_file() and p.name != "index.md"))), True)

    # L02: off-by-one count / wrong first service / columns in another order
    def l02_bad(it, w):
        edit(w, "report.csv", lambda t: re.sub(r"^(\w+),(\d+),", lambda m: f"{m[1]},{int(m[2]) + 1},", t, count=2, flags=re.M))
    check(f"L02_{seed} count off by one: wrong", solve("L02", seed, l02_bad), False)
    check(f"L02_{seed} wrong first service named: wrong", solve("L02", seed, answer="The first error was in nowhere."), False)

    def l02_alt(it, w):
        g = it.gold["svc"]
        (w / "report.csv").write_text("First error,Service,ERROR count\n" + "".join(
            f"{v['first'] or 'N/A'},{s.title()},{v['errors']}\n" for s, v in g.items()))
    check(f"L02_{seed} other column order, titles, N/A: right", solve("L02", seed, l02_alt), True)

    # L03: one page still uses formatPrice / sitemap missing a page
    check(f"L03_{seed} one formatPrice left: wrong", solve("L03", seed, lambda it, w: edit(
        w, "js/app.js", lambda t: t.replace("formatAmount", "formatPrice", 1))), False)
    check(f"L03_{seed} sitemap missing a page: wrong", solve("L03", seed, lambda it, w: edit(
        w, "sitemap.xml", lambda t: t.replace("terms.html", "x.html"))), False)

    # L04: total for one category wrong / positive vs negative sign accepted / missing a duplicate
    check(f"L04_{seed} a total off by $5: wrong", solve("L04", seed, lambda it, w: edit(
        w, "summary.csv", lambda t: re.sub(r"(Dining,)([\d.]+)", lambda m: f"{m[1]}{float(m[2]) + 5:.2f}", t))), False)
    check(f"L04_{seed} totals as negatives, table header: right", solve("L04", seed, lambda it, w: (w / "summary.csv").write_text(
        "Category,Total spent (USD)\n" + "".join(f"{c},-{t:.2f}\n" for c, t in it.gold["totals"].items()))), True)
    check(f"L04_{seed} duplicates.txt empty: wrong", solve("L04", seed, lambda it, w: (w / "duplicates.txt").write_text("")), False)

    # L05: a person twice / phones in another format
    check(f"L05_{seed} a person listed twice: wrong", solve("L05", seed, lambda it, w: edit(
        w, "all_contacts.csv", lambda t: t + t.splitlines()[1] + "\n")), False)
    check(f"L05_{seed} phones formatted (xxx) xxx-xxxx: right", solve("L05", seed, lambda it, w: (w / "all_contacts.csv").write_text(
        "Name,Email,Phone\n" + "".join(f"{n},{e},({p[:3]}) {p[3:6]}-{p[6:]}\n" for e, (n, p) in it.gold["people"].items()))), True)

    # L06: one quantity wrong / extra ingredient from another recipe
    def l06_bad(it, w):
        i, q = next(iter(it.gold["total"].items()))
        edit(w, "shopping.md", lambda t: t.replace(f"- {i}: {q} ", f"- {i}: {q + 1} "))
    check(f"L06_{seed} one quantity wrong: wrong", solve("L06", seed, l06_bad), False)

    # L07: an old key not renamed / numbers as strings
    def l07_bad(it, w):
        cfg = json.loads((w / "config.json").read_text())
        svc = next(s for s, kv in cfg.items() if "hostname" in kv or "timeout" in kv)
        k = "hostname" if "hostname" in cfg[svc] else "timeout"
        cfg[svc][{"hostname": "host", "timeout": "timeout_ms"}[k]] = cfg[svc].pop(k)
        (w / "config.json").write_text(json.dumps(cfg))
    check(f"L07_{seed} old key kept: wrong", solve("L07", seed, l07_bad), False)

    def l07_str(it, w):
        cfg = {s: {k: str(v) for k, v in kv.items()} for s, kv in it.gold["cfg"].items()}
        (w / "config.json").write_text(json.dumps(cfg))
    check(f"L07_{seed} numbers written as strings: wrong", solve("L07", seed, l07_str), False)

    # L08: a closed action included / markdown table instead of csv lines still in tracker.csv
    def l08_bad(it, w):
        edit(w, "tracker.csv", lambda t: t + f"{it.gold['closed'][0]},Someone,2025-01-01\n")
    check(f"L08_{seed} a closed action included: wrong", solve("L08", seed, l08_bad), False)

    # L09: one wrong severity / duplicate written as '#5'
    def l09_bad(it, w):
        edit(w, "triage.csv", lambda t: re.sub(r"^1,(\w+),(\w+),", lambda m: f"1,{m[1]},{'low' if m[2] != 'low' else 'high'},", t, flags=re.M))
    check(f"L09_{seed} one severity wrong: wrong", solve("L09", seed, l09_bad), False)

    def l09_alt(it, w):
        (w / "triage.csv").write_text("Issue,Component,Severity,Duplicate of\n" + "".join(
            f"#{n:03d},{g['component'].title()},{g['severity'].upper()},{'#' + str(g['dup']) if g['dup'] else ''}\n"
            for n, g in it.gold["issues"].items()))
    check(f"L09_{seed} '#003', capitals, '#5' duplicates: right", solve("L09", seed, l09_alt), True)

    # L10: deliveries forgotten (orders only)
    def l10_bad(it, w):
        rows = (w / "stock.csv").read_text().splitlines()
        rows[1] = re.sub(r",(\d+),(\d+)$", lambda m: f",{int(m[1]) + 7},{m[2]}", rows[1])
        (w / "stock.csv").write_text("\n".join(rows) + "\n")
    check(f"L10_{seed} one quantity off: wrong", solve("L10", seed, l10_bad), False)

    # L11: cancelled booking kept / two lines swapped
    def l11_bad(it, w):
        edit(w, "itinerary.md", lambda t: t + f"- cancelled thing ({it.gold['cancelled']})\n")
    check(f"L11_{seed} cancelled booking kept: wrong", solve("L11", seed, l11_bad), False)

    def l11_swap(it, w):
        ls = (w / "itinerary.md").read_text().splitlines()
        ls[2], ls[3] = ls[3], ls[2]
        (w / "itinerary.md").write_text("\n".join(ls) + "\n")
    check(f"L11_{seed} two bookings out of order: wrong", solve("L11", seed, l11_swap), False)

    # L12: decoy todo_count reported / a TODO missing / table layout
    check(f"L12_{seed} decoy todo_count reported: wrong", solve("L12", seed, lambda it, w: edit(
        w, "todo-report.md", lambda t: t + "- line 3: todo_count = len(items)\n")), False)

    def l12_table(it, w):
        rows = "".join(f"| {Path(f).name} | {ln} | {tag}: {note} |\n" for f, ln, tag, note in it.gold["todos"])
        (w / "todo-report.md").write_text("| file | line | comment |\n|---|---|---|\n" + rows)
    check(f"L12_{seed} one table with file per row: right", solve("L12", seed, l12_table), True)

    # new traps (scaled-up versions, 2026-11-10)
    def l07_ms(it, w):
        cfg = json.loads((w / "config.json").read_text())
        for kv in cfg.values():
            if "timeout" in kv:
                kv["timeout"] = round(kv["timeout"] * 1000)
        (w / "config.json").write_text(json.dumps(cfg))
    if any("timeout" in kv for kv in T.build("L07", seed).gold["cfg"].values()):
        check(f"L07_{seed} timeout left in ms: wrong", solve("L07", seed, l07_ms), False)

    def l07_legacy(it, w):
        cfg = json.loads((w / "config.json").read_text())
        both = [p.stem for p in (w / "config" / "legacy").glob("*.ini") if (w / "config" / f"{p.stem}.ini").exists()]
        txt = (w / "config" / "legacy" / f"{both[0]}.ini").read_text()
        vals = dict(re.findall(r"^(\w+) = (.+)$", txt, re.M))
        from keel.evals.tasks_long import L07_RENAME
        cfg[both[0]] = {L07_RENAME.get(k, k): (int(v) / 1000 if k == "timeout_ms" else int(v) if v.isdigit() else v)
                        for k, v in vals.items()}
        (w / "config.json").write_text(json.dumps(cfg))
    check(f"L07_{seed} legacy file wins over config/: wrong", solve("L07", seed, l07_legacy), False)

    def l10_cancelled(it, w):
        order = next(p for p in sorted((w / "orders").glob("*.csv")) if "cancelled" in p.read_text())
        sku, n = order.read_text().splitlines()[2].split(",")
        edit(w, "stock.csv", lambda t: re.sub(rf"^({sku},[^,]+,)(\d+)", lambda m: f"{m[1]}{int(m[2]) - int(n)}", t, flags=re.M))
    check(f"L10_{seed} a cancelled order subtracted: wrong", solve("L10", seed, l10_cancelled), False)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
