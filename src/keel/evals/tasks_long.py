"""Long agent tasks for (user decision 2026-11-10: 10-15 new tasks).

Each task needs many steps (roughly 15-30) and builds up many medium tool outputs, so the plain model can fail by
the step limit or the context cap rather than by one giant output. Same contract as tasks.py: seeded generator ->
Item, automatic scorer (folder state and/or reply), oracle that solves it the intended way. tests/checks/check_agent_tasks.py
--long checks oracle 100% and do-nothing 0%.

Registry: LONG_TASKS (L01-L12). They are not in tasks.TASKS, so default runs keep the original 13 tasks.
"""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

from .tasks import FIRST, WORDS, Item, _filler, _h, _rng, walk


def _csv(text: str) -> list[dict]:
    """Rows of a CSV with lower-cased, stripped header names; tolerant of blank lines and extra spaces."""
    rows = list(csv.reader(io.StringIO(text.strip())))
    if not rows:
        return []
    head = [h.strip().lower() for h in rows[0]]
    return [{head[i]: (c.strip() if i < len(row) else "") for i, c in enumerate(row) if i < len(head)}
            for row in rows[1:] if any(x.strip() for x in row)]


def _col(row: dict, *names) -> str:
    for k, v in row.items():
        if any(n in k for n in names):
            return v
    return ""


def _num(s) -> float | None:
    try:
        return float(str(s).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def _read(work: Path, rel: str) -> str | None:
    hits = sorted(work.rglob(rel)) if "/" not in rel else [work / rel]
    hits = [h for h in hits if h.is_file()]
    return hits[0].read_text(errors="replace") if hits else None


def _result(checks: dict) -> dict:
    ok = sum(bool(v) for v in checks.values())
    return {"correct": ok == len(checks), "score": ok / len(checks), "labels": [k for k, v in checks.items() if not v]}


# ---------------------------------------------------------------- L01 downloads cleanup

L01_TYPES = {"Images": ["jpg", "png", "gif", "heic"], "Documents": ["pdf", "docx", "txt", "xlsx"],
             "Audio": ["mp3", "wav", "m4a"], "Video": ["mp4", "mov"], "Archives": ["zip", "tar.gz", "dmg"],
             "Code": ["py", "js", "html", "css"]}
L01_EXT = {e: t for t, es in L01_TYPES.items() for e in es}


def _ext(name: str) -> str:
    return "tar.gz" if name.endswith(".tar.gz") else name.rsplit(".", 1)[-1].lower()


def l01(seed: int) -> Item:
    r = _rng("L01", seed)
    files, uniq = {}, []
    for i in range(r.randint(90, 110)):
        ext = r.choice(list(L01_EXT))
        name = f"{r.choice(WORDS)}_{r.choice(WORDS)}_{i}.{ext}"
        files[name] = f"{name} {r.random()} {_filler(r, 8)}".encode()
        uniq.append(name)
    for src in r.sample(uniq, r.randint(12, 18)):
        stem, ext = (src[:-7], "tar.gz") if src.endswith(".tar.gz") else src.rsplit(".", 1)
        files[r.choice([f"{stem} (1).{ext}", f"Copy of {src}", f"{stem}_copy.{ext}"])] = files[src]
    gold = {_h(v): L01_EXT[_ext(k)] for k, v in files.items()}
    return Item("L01", seed, "state", "Clean up this Downloads folder: delete duplicate files (same content, keep one "
                "copy), move every file into a folder for its type (Images, Documents, Audio, Video, Archives, Code), "
                "and write index.md listing every file that is left and the folder it is in.", files,
                gold={"type": gold}, steps="L")


def l01_score(item, work, answer, commands):
    now = walk(work)
    by = {}
    for rel, data in now.items():
        if rel.lower() != "index.md":
            by.setdefault(_h(data), []).append(rel)
    lost = [h for h in item.gold["type"] if h not in by]
    dups = [h for h, ps in by.items() if h in item.gold["type"] and len(ps) > 1]
    wrong = [h for h, t in item.gold["type"].items() if h in by and len(by[h]) == 1
             and not (len(Path(by[h][0]).parts) == 2 and Path(by[h][0]).parts[0].lower() == t.lower())]
    idx = _read(work, "index.md") or ""
    listed = all(Path(ps[0]).name.lower() in idx.lower() for h, ps in by.items() if h in item.gold["type"])
    return _result({"no file lost": not lost, "no duplicates left": not dups, "every file in its type folder": not wrong,
                    "index.md lists every file": bool(idx) and listed})


def l01_oracle(item, work):
    seen, lines = set(), []
    for rel in sorted(walk(work), key=lambda p: (("copy" in p.lower() or "(1)" in p), p)):
        p = work / rel
        h = _h(p.read_bytes())
        if h in seen:
            p.unlink()
            continue
        seen.add(h)
        t = item.gold["type"][h]
        (work / t).mkdir(exist_ok=True)
        p.rename(work / t / p.name)
        lines.append(f"- {t}/{p.name}")
    (work / "index.md").write_text("# Index\n\n" + "\n".join(lines) + "\n")
    return "Removed duplicates, sorted by type, wrote index.md."


# ---------------------------------------------------------------- L02 log investigation

L02_SVC = ["auth", "billing", "search", "gateway", "mailer", "scheduler", "storage", "reports", "notifier", "profile"]
L02_MSG = {"INFO": ["request served in {n} ms", "cache hit ratio {n}%", "worker {n} started", "health check ok"],
           "WARN": ["slow query took {n} ms", "retrying connection (attempt {n})", "queue depth {n}"],
           "ERROR": ["connection refused by db-{n}", "timeout after {n} ms", "unhandled exception in handler {n}"]}


def l02(seed: int) -> Item:
    r = _rng("L02", seed)
    files, gold = {}, {}
    day = datetime(2025, r.randint(1, 12), r.randint(1, 28), 8, 0, 0)
    for svc in L02_SVC:
        t, lines, errs = day + timedelta(seconds=r.randint(0, 600)), [], []
        p_err = r.choice([0.0, 0.02, 0.05, 0.08])
        for _ in range(r.randint(150, 230)):
            t += timedelta(seconds=r.randint(5, 240))
            lvl = "ERROR" if r.random() < p_err else ("WARN" if r.random() < 0.12 else "INFO")
            lines.append(f"{t:%Y-%m-%d %H:%M:%S} {lvl:5s} [{svc}] {r.choice(L02_MSG[lvl]).format(n=r.randint(1, 999))}")
            if lvl == "ERROR":
                errs.append(f"{t:%H:%M:%S}")
        files[f"logs/{svc}.log"] = ("\n".join(lines) + "\n").encode()
        gold[svc] = {"errors": len(errs), "first": errs[0] if errs else None}
    firsts = [(g["first"], s) for s, g in gold.items() if g["first"]]
    return Item("L02", seed, "state", "Look through the logs folder. Write report.csv with one row per service: the "
                "service name, how many ERROR lines it has, and the time of its first ERROR (or 'none'). Then tell me "
                "which service had the first error of the day.", files,
                gold={"svc": gold, "first_svc": min(firsts)[1]}, steps="L")


def l02_score(item, work, answer, commands):
    rows = _csv(_read(work, "report.csv") or "")
    got = {}
    for row in rows:
        svc = next((s for s in L02_SVC if s in " ".join(row.values()).lower()), None)
        if svc:
            got[svc] = row
    counts = times = 0
    for svc, g in item.gold["svc"].items():
        row = got.get(svc)
        if not row:
            continue
        vals = list(row.values())
        counts += any(_num(v) == g["errors"] for v in vals if ":" not in v)
        # first-error time: the right HH:MM:SS, or no time at all for a service without errors
        times += (g["first"] in " ".join(vals)) if g["first"] else not re.search(r"\d\d:\d\d:\d\d", " ".join(vals))
    n = len(L02_SVC)
    return _result({"report.csv has every service": len(got) == n, "error counts right": counts == n,
                    "first-error times right": times == n,
                    "names the first service": item.gold["first_svc"] in answer.lower()})


def l02_oracle(item, work):
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["service", "errors", "first_error"])
    for svc, g in item.gold["svc"].items():
        w.writerow([svc, g["errors"], g["first"] or "none"])
    (work / "report.csv").write_text(out.getvalue())
    return f"The first error of the day was in {item.gold['first_svc']}."


# ---------------------------------------------------------------- L03 site update

L03_PAGES = ["index", "about", "pricing", "contact", "blog", "faq", "team", "careers", "shop", "cart", "terms", "privacy",
             "press", "partners", "support", "status", "docs", "api", "security", "jobs", "events", "news", "legal",
             "sitemap-page"]


def l03(seed: int) -> Item:
    r = _rng("L03", seed)
    files, uses = {}, {}
    broken = r.sample(L03_PAGES[1:], 6)
    for p in L03_PAGES:
        links = r.sample([q for q in L03_PAGES if q != p], 3)
        hrefs = [(f"{q}.htm" if q in broken and i == 0 else f"{q}.html") for i, q in enumerate(links)]
        n = r.randint(0, 3)
        body = "".join(f'<p class="price" id="p{i}"></p>\n<script>document.getElementById("p{i}").textContent = '
                       f'formatPrice({r.randint(5, 500)});</script>\n' for i in range(n))
        nav = " | ".join(f'<a href="{h}">{q.title()}</a>' for h, q in zip(hrefs, links, strict=False))
        files[f"{p}.html"] = (f"<!doctype html>\n<html><head><title>{p.title()}</title>"
                              f'<script src="js/utils.js"></script></head>\n<body>\n<nav>{nav}</nav>\n'
                              f"<h1>{p.title()}</h1>\n<p>{_filler(r, 25)}</p>\n{body}</body></html>\n").encode()
        uses[f"{p}.html"] = n
    files["js/utils.js"] = ("// shared helpers\nfunction formatPrice(n) {\n  return '$' + n.toFixed(2);\n}\n"
                            "function slug(s) { return s.toLowerCase().replace(/ /g, '-'); }\n").encode()
    files["js/app.js"] = ("document.querySelectorAll('.total').forEach(el => {\n"
                          "  el.textContent = formatPrice(Number(el.dataset.v));\n});\n").encode()
    uses["js/utils.js"], uses["js/app.js"] = 1, 1
    listed = r.sample(L03_PAGES, 16)
    files["sitemap.xml"] = ('<?xml version="1.0"?>\n<urlset>\n' + "".join(f"  <url><loc>https://example.com/{p}.html"
                            f"</loc></url>\n" for p in listed) + "</urlset>\n").encode()
    return Item("L03", seed, "state", "In this website: rename the JavaScript function formatPrice to formatAmount "
                "everywhere, fix the broken links between pages, and make sure sitemap.xml lists every page.", files,
                gold={"uses": uses}, steps="L")


def l03_score(item, work, answer, commands):
    renamed = links_ok = True
    for rel, n in item.gold["uses"].items():
        p = work / rel
        if not p.exists():
            return _result({"pages kept": False, "renamed": False, "links fixed": False, "sitemap complete": False})
        txt = p.read_text(errors="replace")
        if re.search(r"\bformatPrice\b", txt) or len(re.findall(r"\bformatAmount\b", txt)) < n:
            renamed = False
        if rel.endswith(".html"):
            for h in re.findall(r'href="([^"]+)"', txt):
                if not h.startswith(("http", "#", "mailto:")) and not (work / h).is_file():
                    links_ok = False
    sm = _read(work, "sitemap.xml") or ""
    return _result({"pages kept": True, "renamed": renamed, "links fixed": links_ok,
                    "sitemap complete": all(f"{p}.html" in sm for p in L03_PAGES)})


def l03_oracle(item, work):
    for rel in item.gold["uses"]:
        p = work / rel
        txt = p.read_text().replace("formatPrice", "formatAmount")
        txt = re.sub(r'href="([\w-]+)\.htm"', r'href="\1.html"', txt)
        p.write_text(txt)
    (work / "sitemap.xml").write_text('<?xml version="1.0"?>\n<urlset>\n' + "".join(
        f"  <url><loc>https://example.com/{p}.html</loc></url>\n" for p in L03_PAGES) + "</urlset>\n")
    return "Renamed, fixed links, completed the sitemap."


# ---------------------------------------------------------------- L04 expense year-end

L04_CAT = {"Groceries": ["Green Grocer", "Trader Joe's", "Corner Market"], "Dining": ["Sushi Bar", "Pizza Place", "Cafe Nero"],
           "Transport": ["Metro Card", "Shell Station", "City Taxi"], "Utilities": ["Electric Co", "Water Board", "Fiber ISP"],
           "Shopping": ["IKEA", "Bookshop", "Outdoor Store"], "Entertainment": ["Cinema", "Spotify", "Concert Hall"]}


def l04(seed: int) -> Item:
    r = _rng("L04", seed)
    files, totals, dups = {}, {c: 0.0 for c in L04_CAT}, []
    dup_months = set(r.sample(range(1, 13), r.randint(3, 5)))
    for m in range(1, 13):
        rows = []
        for _ in range(r.randint(14, 22)):
            c = r.choice(list(L04_CAT))
            rows.append([f"2025-{m:02d}-{r.randint(1, 28):02d}", r.choice(L04_CAT[c]), c, -round(r.uniform(4, 180), 2)])
        rows.append([f"2025-{m:02d}-01", "Salary ACME", "Income", round(r.uniform(2800, 3200), 2)])
        if m in dup_months:
            d = list(r.choice([x for x in rows if x[2] != "Income"]))
            rows.append(d)
            dups.append((d[0], d[1]))
        rows.sort(key=lambda x: x[0])
        out = io.StringIO()
        if m <= 6:
            w = csv.writer(out)
            w.writerow(["date", "description", "category", "amount"])
        else:   # new bank app from July: semicolons, dd/mm/yyyy, separate debit / credit columns
            w = csv.writer(out, delimiter=";")
            w.writerow(["Date", "Merchant", "Debit", "Credit", "Category"])
        for x in rows:
            if m <= 6:
                w.writerow(x)
            else:
                d = datetime.strptime(x[0], "%Y-%m-%d")
                w.writerow([f"{d:%d/%m/%Y}", x[1], f"{-x[3]:.2f}" if x[3] < 0 else "", f"{x[3]:.2f}" if x[3] > 0 else "", x[2]])
            if x[2] != "Income":
                totals[x[2]] += -x[3]
        files[f"statements/2025-{m:02d}.csv"] = out.getvalue().encode()
    return Item("L04", seed, "state", "Using the 12 monthly statements in the statements folder, write summary.csv "
                "with the total I spent in each category over the year (count every row, income excluded), and list "
                "any duplicate charges (same date, description and amount) in duplicates.txt.", files,
                gold={"totals": {k: round(v, 2) for k, v in totals.items()}, "dups": dups}, steps="L")


def l04_score(item, work, answer, commands):
    rows = _csv(_read(work, "summary.csv") or "")
    ok = 0
    for c, t in item.gold["totals"].items():
        row = next((x for x in rows if c.lower() in " ".join(x.values()).lower()), None)
        ok += bool(row) and any(v is not None and abs(abs(v) - t) <= 0.02 for v in map(_num, row.values()))
    dtxt = (_read(work, "duplicates.txt") or "").lower()
    found = sum((d in dtxt or f"{d[8:10]}/{d[5:7]}/{d[:4]}" in dtxt) and desc.lower() in dtxt
                for d, desc in item.gold["dups"])
    return _result({"every category total right": ok == len(item.gold["totals"]),
                    "every duplicate listed": bool(dtxt) and found == len(item.gold["dups"])})


def l04_oracle(item, work):
    (work / "summary.csv").write_text("category,total\n" + "".join(f"{c},{t:.2f}\n" for c, t in item.gold["totals"].items()))
    (work / "duplicates.txt").write_text("".join(f"{d} {desc}\n" for d, desc in item.gold["dups"]))
    return "Wrote summary.csv and duplicates.txt."


# ---------------------------------------------------------------- L05 contacts merge

L05_LAST = ["Smith", "Garcia", "Chen", "Okafor", "Novak", "Rossi", "Kim", "Silva", "Brown", "Ivanova", "Haddad", "Mehta"]


def l05(seed: int) -> Item:
    r = _rng("L05", seed)
    people = {}
    while len(people) < r.randint(40, 50):
        fn, ln = r.choice(FIRST), r.choice(L05_LAST)
        email = f"{fn.lower()}.{ln.lower()}{r.randint(1, 99)}@example.com"
        people[email] = (f"{fn} {ln}", f"{r.randint(200, 999)}{r.randint(1000000, 9999999)}")
    emails = list(people)
    src = {e: r.sample(["csv", "json", "vcf"], r.choice([1, 1, 2, 3])) for e in emails}
    fmt = lambda p: r.choice([f"({p[:3]}) {p[3:6]}-{p[6:]}", f"{p[:3]}-{p[3:6]}-{p[6:]}", f"+1 {p[:3]} {p[3:6]} {p[6:]}", p])
    case = lambda e: r.choice([e, e.upper(), e.title()])
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["Name", "E-mail", "Phone"])
    js, vcf = [], []
    for e in emails:
        name, ph = people[e]
        if "csv" in src[e]:
            w.writerow([name, case(e), fmt(ph)])
        if "json" in src[e]:
            js.append({"fullName": name.lower(), "contact": {"email": case(e), "mobile": fmt(ph)}})
        if "vcf" in src[e]:
            vcf.append(f"BEGIN:VCARD\nVERSION:3.0\nFN:{name.split()[1]}, {name.split()[0]}\nEMAIL:{case(e)}\n"
                       f"TEL:{fmt(ph)}\nEND:VCARD\n")
    files = {"exports/phone_contacts.csv": out.getvalue().encode(),
             "exports/webmail.json": json.dumps(js, indent=2).encode(), "exports/old_phone.vcf": "".join(vcf).encode()}
    return Item("L05", seed, "state", "Merge my three contact exports in the exports folder into one file, "
                "all_contacts.csv, with columns name, email and phone. Each person should appear once (match people "
                "by email address; write names as First Last).", files, gold={"people": people}, steps="L")


def l05_score(item, work, answer, commands):
    rows = _csv(_read(work, "all_contacts.csv") or "")
    seen, good = {}, 0
    for row in rows:
        em = next((v.lower() for v in row.values() if "@" in v), None)
        if em:
            seen[em] = seen.get(em, 0) + 1
    for e, (name, ph) in item.gold["people"].items():
        row = next((x for x in rows if any(v.lower() == e for v in x.values())), None)
        if row and any(re.sub(r"\D", "", v)[-10:] == ph for v in row.values()) and \
                any(v.lower() == name.lower() for v in row.values()):
            good += 1
    return _result({"every person once": all(seen.get(e) == 1 for e in item.gold["people"]) and
                    set(seen) == set(item.gold["people"]),
                    "name and phone right for everyone": good == len(item.gold["people"])})


def l05_oracle(item, work):
    (work / "all_contacts.csv").write_text("name,email,phone\n" + "".join(
        f"{n},{e},{p}\n" for e, (n, p) in item.gold["people"].items()))
    return "Merged into all_contacts.csv."


# ---------------------------------------------------------------- L06 shopping list

L06_ING = {"flour": "g", "sugar": "g", "butter": "g", "milk": "ml", "eggs": "pcs", "rice": "g", "onion": "pcs",
           "garlic": "cloves", "tomatoes": "pcs", "olive oil": "ml", "chicken": "g", "lentils": "g", "carrots": "pcs",
           "cheese": "g", "pasta": "g", "spinach": "g", "lemon": "pcs", "yogurt": "g"}
L06_DISH = ["Pancakes", "Risotto", "Lentil Soup", "Chicken Curry", "Pasta Bake", "Omelette", "Spinach Pie",
            "Tomato Salad", "Lemon Cake", "Fried Rice", "Mac and Cheese", "Shakshuka", "Carrot Soup", "Garlic Bread",
            "Yogurt Bowl", "Stuffed Peppers", "Minestrone", "Quiche", "Biryani", "Crepes"]


def l06(seed: int) -> Item:
    r = _rng("L06", seed)
    files, recipes = {}, {}
    for dish in L06_DISH:
        ings = {i: (r.randint(1, 4) if L06_ING[i] in ("pcs", "cloves") else r.randint(2, 30) * 10)
                for i in r.sample(list(L06_ING), r.randint(4, 7))}
        recipes[dish] = ings
        slug = dish.lower().replace(" ", "_")
        def show(i, q):   # some recipes write grams / ml as kg / l
            u = L06_ING[i]
            if u in ("g", "ml") and q >= 100 and r.random() < 0.4:
                return f"{q / 1000:g} {'kg' if u == 'g' else 'l'} {i}"
            return f"{q} {u} {i}"
        files[f"recipes/{slug}.txt"] = (f"{dish}\n\nServes 2.\n\nIngredients:\n" + "".join(
            f"- {show(i, q)}\n" for i, q in ings.items()) + f"\nMethod:\n{_filler(r, 30)}\n").encode()
    chosen = r.sample(L06_DISH, 8)
    total = {}
    for d in chosen:
        for i, q in recipes[d].items():
            total[i] = total.get(i, 0) + q
    return Item("L06", seed, "state", f"I'm cooking {', '.join(chosen[:-1])} and {chosen[-1]} this week (recipes are "
                "in the recipes folder). Write shopping.md with the total amount of each ingredient I need across "
                "those eight recipes, in grams, millilitres or pieces.", files, gold={"total": total}, steps="M")


def l06_score(item, work, answer, commands):
    txt = (_read(work, "shopping.md") or "").lower()
    lines = txt.splitlines()
    ok = sum(any(i in ln and re.search(rf"(?<![\d.]){q}(?![\d])", ln) for ln in lines) for i, q in item.gold["total"].items())
    extra = [i for i in L06_ING if i not in item.gold["total"] and any(re.search(rf"\b{i}\b", ln) and re.search(r"\d", ln)
                                                                          for ln in lines)]
    return _result({"every total right": bool(txt) and ok == len(item.gold["total"]), "nothing extra": not extra})


def l06_oracle(item, work):
    (work / "shopping.md").write_text("# Shopping list\n\n" + "".join(
        f"- {i}: {q} {L06_ING[i]}\n" for i, q in sorted(item.gold["total"].items())))
    return "Wrote shopping.md."


# ---------------------------------------------------------------- L07 config migration

L07_RENAME = {"host": "hostname", "timeout_ms": "timeout", "enable": "enabled", "user": "username", "pass_file": "password_file"}
L07_KEYS = ["host", "port", "timeout_ms", "enable", "user", "pass_file", "workers", "log_level", "retries", "region"]


def l07(seed: int) -> Item:
    r = _rng("L07", seed)
    files, gold = {}, {}
    svcs = r.sample(["api", "db", "cache", "queue", "mail", "search", "auth", "metrics", "storage", "billing",
                     "jobs", "cdn", "logs", "gateway", "notify", "geo"], 14)
    legacy = set(svcs[10:]) | set(r.sample(svcs[:10], 2))    # 2 services exist in both folders: config/ wins
    for svc in svcs + sorted(legacy & set(svcs[:10])):
        is_legacy = svc in legacy and (svc in svcs[10:] or f"config/{svc}.ini" in files)
        keys = r.sample(L07_KEYS, r.randint(5, 8))
        vals = {}
        for k in keys:
            vals[k] = {"host": f"{svc}.internal", "port": r.randint(1000, 9999), "timeout_ms": r.randint(1, 60) * 100,
                       "enable": r.choice(["true", "false"]), "user": f"{svc}_svc", "pass_file": f"/etc/secrets/{svc}",
                       "workers": r.randint(1, 16), "log_level": r.choice(["debug", "info", "warn"]),
                       "retries": r.randint(0, 5), "region": r.choice(["eu-west", "us-east", "ap-south"])}[k]
        path = f"config/legacy/{svc}.ini" if is_legacy else f"config/{svc}.ini"
        files[path] = (f"; {svc} service\n[{svc}]\n" + "".join(f"{k} = {v}\n" for k, v in vals.items())).encode()
        if not (is_legacy and svc in gold):
            gold[svc] = {L07_RENAME.get(k, k): (v / 1000 if k == "timeout_ms" else v) for k, v in vals.items()}
    files["MIGRATION.md"] = ("# Key renames for the new config format\n\n" + "".join(
        f"- `{a}` is now `{b}`\n" for a, b in L07_RENAME.items()) + "\nEverything else keeps its name. `timeout` is in "
        "seconds (old `timeout_ms` was milliseconds: divide by 1000).\nIf a service has a file in both config/ and "
        "config/legacy/, use the one in config/.\n").encode()
    return Item("L07", seed, "state", "Combine all the .ini files in the config folder (including config/legacy) "
                "into one config.json: one "
                "object per file, named after the file without .ini, holding every key. Rename the old keys listed in "
                "MIGRATION.md, and write numbers as numbers.", files, gold={"cfg": gold}, steps="M")


def l07_score(item, work, answer, commands):
    try:
        got = json.loads(_read(work, "config.json") or "")
    except (json.JSONDecodeError, TypeError):
        return _result({"config.json valid": False, "every value right": False, "old keys renamed": False})
    if not isinstance(got, dict):
        return _result({"config.json valid": False, "every value right": False, "old keys renamed": False})
    good = renamed = True
    for svc, kv in item.gold["cfg"].items():
        obj = got.get(svc) or got.get(f"{svc}.ini") or {}
        if not isinstance(obj, dict):
            good = False
            continue
        if set(L07_RENAME) & set(obj):
            renamed = False
        for k, v in kv.items():
            g = obj.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                good &= isinstance(g, (int, float)) and not isinstance(g, bool) and abs(g - v) < 1e-9
            else:
                good &= str(g).lower() == str(v).lower()
    return _result({"config.json valid": True, "every value right": good, "old keys renamed": renamed})


def l07_oracle(item, work):
    (work / "config.json").write_text(json.dumps(item.gold["cfg"], indent=2))
    return "Wrote config.json."


# ---------------------------------------------------------------- L08 meeting action tracker

L08_TASKS = ["send the budget draft", "book the venue", "update the website", "review the supplier contract",
             "write the hiring post", "chase unpaid invoices", "prepare client slides", "fix the onboarding doc",
             "order new laptops", "plan the offsite", "renew the domain", "audit the expense claims",
             "draft the newsletter", "set up the CRM", "collect survey results", "migrate the wiki"]


def l08(seed: int) -> Item:
    r = _rng("L08", seed)
    files, open_, closed = {}, {}, set()
    d = date(2025, r.randint(1, 6), r.randint(1, 20))
    pool = r.sample(L08_TASKS, len(L08_TASKS))
    for _ in range(10):
        lines = [f"# Team meeting {d.isoformat()}", "", f"Attendees: {', '.join(r.sample(FIRST, 4))}", "",
                 "## Notes", _filler(r, 25), ""]
        done_now = r.sample(list(open_), min(len(open_), r.randint(0, 2))) if open_ else []
        for t in done_now:
            lines.append(f"Done: {t} ({open_[t][0]})")
            closed.add(t)
            del open_[t]
        for _ in range(r.randint(1, 2)):
            if not pool:
                break
            t = pool.pop()
            owner, due = r.choice(FIRST), d + timedelta(days=r.randint(5, 30))
            open_[t] = (owner, due.isoformat())
            lines.append(f"Action: {owner} to {t} by {due.isoformat()}")
        files[f"meetings/{d.isoformat()}.md"] = ("\n".join(lines) + "\n").encode()
        d += timedelta(days=r.randint(5, 9))
    return Item("L08", seed, "state", "Go through my meeting notes in the meetings folder in order and write "
                "tracker.csv with the action items that are still open after the last meeting: task, owner and due "
                "date.", files, gold={"open": open_, "closed": sorted(closed)}, steps="L")


def l08_score(item, work, answer, commands):
    txt = (_read(work, "tracker.csv") or "").lower()
    lines = txt.splitlines()
    ok = sum(any(t.lower() in ln and o.lower() in ln and d in ln for ln in lines)
             for t, (o, d) in item.gold["open"].items())
    leaked = [t for t in item.gold["closed"] if t.lower() in txt]
    return _result({"every open action listed": bool(txt) and ok == len(item.gold["open"]),
                    "no closed action listed": not leaked})


def l08_oracle(item, work):
    (work / "tracker.csv").write_text("task,owner,due\n" + "".join(
        f"{t},{o},{d}\n" for t, (o, d) in item.gold["open"].items()))
    return "Wrote tracker.csv."


# ---------------------------------------------------------------- L09 bug triage

L09_COMP = {"auth": ["login", "password", "sign-in"], "billing": ["invoice", "payment", "refund"],
            "search": ["search", "filter", "results"], "ui": ["button", "layout", "dark mode"]}
L09_SEV = {"high": ["crash", "data loss", "cannot"], "low": ["typo", "colour", "alignment"]}


def l09(seed: int) -> Item:
    r = _rng("L09", seed)
    files, gold, titles = {}, {}, {}
    for n in range(1, r.randint(45, 55) + 1):
        if titles and r.random() < 0.15:
            orig = r.choice(list(titles))
            title, comp, sev, dup = titles[orig], gold[orig]["component"], gold[orig]["severity"], orig
        else:
            comp = r.choice(list(L09_COMP))
            sev = r.choice(["high", "medium", "low"])
            word = r.choice(L09_COMP[comp])
            sw = r.choice(L09_SEV[sev]) if sev != "medium" else "slow"
            title = f"{word} {sw} on {r.choice(['Safari', 'Chrome', 'iOS', 'Android', 'Windows'])} #{r.randint(100, 999)}"
            while title in titles.values():
                title += "!"
            dup = None
        titles[n] = title
        gold[n] = {"component": comp, "severity": sev, "dup": dup}
        files[f"issues/{n:03d}.md"] = (f"# {title}\n\nReported by {r.choice(FIRST)}.\n\n{_filler(r, 30)}\n").encode()
    files["TRIAGE.md"] = ("# Triage rules\n\nComponent, from words in the title:\n" + "".join(
        f"- {c}: {', '.join(ws)}\n" for c, ws in L09_COMP.items()) + "\nSeverity, from words in the title:\n"
        "- high: crash, data loss, cannot\n- low: typo, colour, alignment\n- medium: anything else\n\n"
        "Duplicates: an issue with exactly the same title as an earlier (lower-numbered) issue is a duplicate of it.\n").encode()
    return Item("L09", seed, "state", "Triage the issues in the issues folder using the rules in TRIAGE.md. Write "
                "triage.csv with columns issue, component, severity, duplicate_of (the earlier issue number, or empty).",
                files, gold={"issues": gold}, steps="L")


def l09_score(item, work, answer, commands):
    rows = _csv(_read(work, "triage.csv") or "")
    got = {}
    for row in rows:
        m = re.search(r"\d+", _col(row, "issue") or next(iter(row.values()), ""))
        if m:
            got[int(m.group())] = row
    comp = sev = dup = 0
    for n, g in item.gold["issues"].items():
        row = got.get(int(n))
        if not row:
            continue
        comp += _col(row, "comp").lower() == g["component"]
        sev += _col(row, "sev").lower() == g["severity"]
        dv = re.search(r"\d+", _col(row, "dup") or "")
        dup += (int(dv.group()) if dv else None) == g["dup"]
    n = len(item.gold["issues"])
    return _result({"every issue present": len(set(got) & {int(k) for k in item.gold["issues"]}) == n,
                    "components right": comp == n, "severities right": sev == n, "duplicates right": dup == n})


def l09_oracle(item, work):
    (work / "triage.csv").write_text("issue,component,severity,duplicate_of\n" + "".join(
        f"{n},{g['component']},{g['severity']},{g['dup'] or ''}\n" for n, g in item.gold["issues"].items()))
    return "Wrote triage.csv."


# ---------------------------------------------------------------- L10 stock reconciliation

L10_ITEMS = ["desk lamp", "office chair", "monitor arm", "usb hub", "keyboard", "mouse", "webcam", "headset",
             "laptop stand", "cable kit", "notebook", "pen set", "whiteboard", "printer paper", "stapler", "desk mat"]


def l10(seed: int) -> Item:
    r = _rng("L10", seed)
    stock = {f"SKU{100 + i}": [name, r.randint(10, 60), r.randint(5, 20)] for i, name in enumerate(L10_ITEMS)}
    qty = {s: v[1] for s, v in stock.items()}
    files = {}
    n_orders = r.randint(35, 45)
    cancelled = set(r.sample(range(n_orders), 2))
    for k in range(n_orders):
        lines = []
        for s in r.sample(list(stock), r.randint(2, 4)):
            n = r.randint(1, 4)
            if k not in cancelled:
                qty[s] -= n
            lines.append(f"{s},{n}")
        status = "cancelled" if k in cancelled else "shipped"
        files[f"orders/order_{1000 + k}.csv"] = (f"# status: {status}\nsku,quantity\n" + "\n".join(lines) + "\n").encode()
    for k in range(6):
        lines = []
        for s in r.sample(list(stock), r.randint(2, 4)):
            n = r.randint(5, 20)
            qty[s] += n
            lines.append(f"{s},{n}")
        files[f"received/delivery_{k + 1}.csv"] = ("sku,quantity\n" + "\n".join(lines) + "\n").encode()
    files["stock.csv"] = ("sku,name,quantity,reorder_level\n" + "".join(
        f"{s},{v[0]},{v[1]},{v[2]}\n" for s, v in stock.items())).encode()
    reorder = sorted(s for s in stock if qty[s] <= stock[s][2])
    return Item("L10", seed, "state", "Update stock.csv to today's quantities: subtract every shipped order in the orders "
                "folder (cancelled orders don't count) and add the deliveries in the received folder. Then write reorder.csv listing every item that is now at "
                "or below its reorder level.", files, gold={"qty": qty, "reorder": reorder}, steps="L")


def l10_score(item, work, answer, commands):
    rows = _csv(_read(work, "stock.csv") or "")
    q = {(_col(x, "sku") or "").upper(): _num(_col(x, "quant", "qty")) for x in rows}
    rtxt = (_read(work, "reorder.csv") or "").upper()
    listed = set(re.findall(r"SKU\d+", rtxt))
    return _result({"stock quantities right": all(q.get(s) == v for s, v in item.gold["qty"].items()),
                    "reorder list right": bool(rtxt) and listed == set(item.gold["reorder"])})


def l10_oracle(item, work):
    rows = _csv((work / "stock.csv").read_text())
    (work / "stock.csv").write_text("sku,name,quantity,reorder_level\n" + "".join(
        f"{x['sku']},{x['name']},{item.gold['qty'][x['sku']]},{x['reorder_level']}\n" for x in rows))
    (work / "reorder.csv").write_text("sku\n" + "".join(f"{s}\n" for s in item.gold["reorder"]))
    return "Updated stock.csv and wrote reorder.csv."


# ---------------------------------------------------------------- L11 itinerary

L11_CITY = ["Lisbon", "Porto", "Madrid", "Seville", "Barcelona", "Valencia"]


def l11(seed: int) -> Item:
    r = _rng("L11", seed)
    files, active = {}, []
    t = datetime(2026, r.randint(3, 9), r.randint(1, 15), 7, 0)
    cities = r.sample(L11_CITY, 4)
    kinds = ["Flight", "Hotel", "Train", "Hotel", "Car rental", "Train", "Hotel", "Flight"]
    for k, kind in enumerate(kinds):
        t += timedelta(hours=r.randint(10, 40))
        code = "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
        where = cities[min(k // 2, 3)]
        active.append({"code": code, "when": t.isoformat(), "kind": kind})
        body = (f"From: bookings@{kind.lower().replace(' ', '')}.example\nSubject: Your {kind.lower()} booking is "
                f"confirmed\nDate: {(t - timedelta(days=r.randint(10, 40))):%a, %d %b %Y}\n\nHello,\n\nYour {kind.lower()} "
                f"in {where} is confirmed for {t:%A %d %B %Y at %H:%M}.\nConfirmation number: {code}\n\n{_filler(r, 20)}\n")
        files[f"bookings/{r.randint(10000, 99999)}_{kind.lower().replace(' ', '_')}.eml"] = body.encode()
    cancelled = r.choice(active[1:-1])
    active.remove(cancelled)
    files[f"bookings/{r.randint(10000, 99999)}_cancellation.eml"] = (
        f"From: bookings@{cancelled['kind'].lower().replace(' ', '')}.example\nSubject: Cancellation\n\nHello,\n\nWe have "
        f"cancelled your booking with confirmation number {cancelled['code']} as requested.\n").encode()
    for _ in range(3):
        files[f"bookings/{r.randint(10000, 99999)}_newsletter.eml"] = (
            f"From: deals@travel.example\nSubject: Summer deals\n\n{_filler(r, 40)}\nUse code {r.randint(1000, 9999)} "
            f"for 10% off.\n").encode()
    return Item("L11", seed, "state", "From the emails in the bookings folder, write itinerary.md: my trip in date "
                "order, one line per booking with the date, what it is, and the confirmation number. Leave out anything "
                "that was cancelled.", files, gold={"active": active, "cancelled": cancelled["code"]}, steps="M")


def l11_score(item, work, answer, commands):
    txt = _read(work, "itinerary.md") or ""
    pos = [txt.find(b["code"]) for b in item.gold["active"]]
    return _result({"every booking listed": bool(txt) and all(p >= 0 for p in pos),
                    "in date order": all(p >= 0 for p in pos) and pos == sorted(pos),
                    "cancelled one left out": item.gold["cancelled"] not in txt})


def l11_oracle(item, work):
    (work / "itinerary.md").write_text("# Itinerary\n\n" + "".join(
        f"- {b['when'][:16].replace('T', ' ')} {b['kind']} ({b['code']})\n" for b in item.gold["active"]))
    return "Wrote itinerary.md."


# ---------------------------------------------------------------- L12 TODO sweep

L12_NOTES = ["handle the empty list case", "remove this hack after the migration", "add retries", "check for None",
             "use the config value instead", "this leaks a file handle", "cache the result", "validate the input",
             "log the error properly", "split this function", "write a test for this", "support pagination"]


def l12(seed: int) -> Item:
    r = _rng("L12", seed)
    files, gold = {}, []
    mods = r.sample(["users", "orders", "billing", "search", "auth", "export", "report", "mailer", "cache", "util",
                     "parser", "client", "server", "jobs", "models", "views", "forms", "admin", "api", "cli",
                     "config", "db", "events", "files", "geo", "i18n", "media", "notify", "queue", "stats"], 30)
    mods += [f"{m}_v2" for m in r.sample(mods, 30)]
    for m in mods:
        py = r.random() < 0.6
        cm = "#" if py else "//"
        lines = []
        for k in range(r.randint(40, 80)):
            if r.random() < 0.035:
                tag, note = r.choice(["TODO", "FIXME"]), r.choice(L12_NOTES)
                lines.append(f"    {cm} {tag}: {note}")
                gold.append((f"src/{m}.{'py' if py else 'js'}", len(lines), tag, note))
            elif r.random() < 0.05:
                lines.append(f"    todo_count = len(items)  {cm} counts open items" if py else
                             f"    const todoCount = items.length; {cm} counts open items")
            else:
                lines.append(f"    x{k} = compute_{r.choice(WORDS)}({k})" if py else f"    const x{k} = compute{k}();")
        head = f"def {m}_main():" if py else f"function {m}Main() {{"
        files[f"src/{m}.{'py' if py else 'js'}"] = (head + "\n" + "\n".join(lines) + ("\n" if py else "\n}\n")).encode()
    # line numbers in the file include the header line
    gold = [(f, ln + 1, tag, note) for f, ln, tag, note in gold]
    return Item("L12", seed, "state", "Find every TODO and FIXME comment in the src folder and write todo-report.md, "
                "grouped by file, giving the line number and the comment text of each.", files, gold={"todos": gold},
                steps="L")


def l12_score(item, work, answer, commands):
    txt = _read(work, "todo-report.md") or ""
    lines = txt.splitlines()
    found = 0
    for f, ln, _tag, note in item.gold["todos"]:
        name = Path(f).name
        for i, line in enumerate(lines):
            if note in line and re.search(rf"(?<!\d){ln}(?!\d)", line):
                ctx = " ".join(lines[max(0, i - 40):i + 1])
                if name in ctx:
                    found += 1
                    break
    extra = len(re.findall(r"todo_?count", txt, re.I))
    return _result({"every TODO/FIXME listed with file and line": bool(txt) and found == len(item.gold["todos"]),
                    "no false hits": extra == 0})


def l12_oracle(item, work):
    out, cur = ["# TODO report", ""], None
    for f, ln, tag, note in item.gold["todos"]:
        if f != cur:
            out += ["", f"## {f}"]
            cur = f
        out.append(f"- line {ln}: {tag}: {note}")
    (work / "todo-report.md").write_text("\n".join(out) + "\n")
    return "Wrote todo-report.md."


# ---------------------------------------------------------------- registry

LONG_TASKS = {
    "L01": (l01, l01_score, l01_oracle, "Downloads cleanup (long)"),
    "L02": (l02, l02_score, l02_oracle, "Log investigation (long)"),
    "L03": (l03, l03_score, l03_oracle, "Site update (long)"),
    "L04": (l04, l04_score, l04_oracle, "Expense year-end (long)"),
    "L05": (l05, l05_score, l05_oracle, "Contacts merge (long)"),
    "L06": (l06, l06_score, l06_oracle, "Shopping list from recipes (long)"),
    "L07": (l07, l07_score, l07_oracle, "Config migration (long)"),
    "L08": (l08, l08_score, l08_oracle, "Meeting action tracker (long)"),
    "L09": (l09, l09_score, l09_oracle, "Bug triage (long)"),
    "L10": (l10, l10_score, l10_oracle, "Stock reconciliation (long)"),
    "L11": (l11, l11_score, l11_oracle, "Itinerary from emails (long)"),
    "L12": (l12, l12_score, l12_oracle, "TODO sweep (long)"),
}
