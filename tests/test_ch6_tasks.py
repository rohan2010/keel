"""Chapter 6 tasks (tasks_ch6.py): scorers accept other correct solutions (done the way a model would, often with
shell commands) and reject the plausible wrong ones. No model needed."""

import shutil
import subprocess
from pathlib import Path

import pytest

from keel.evals import tasks as T
from keel.evals.tasks_ch6 import CH6_TASKS

SEEDS = [0, 1, 2, 1024, 1025, 1026, 1027]          # dev + the headline's test items


def fresh(task, seed, tmp_path):
    item = T.build(task, seed)
    work = tmp_path / f"{task}_{seed}"
    if work.exists():
        shutil.rmtree(work)
    T.materialize(item, work)
    return item, work


def sh(cmd, cwd):
    subprocess.run(cmd, shell=True, cwd=cwd, check=True, capture_output=True)


@pytest.mark.parametrize("task", sorted(CH6_TASKS))
@pytest.mark.parametrize("seed", SEEDS)
def test_oracle_passes_noop_fails(task, seed, tmp_path):
    item, work = fresh(task, seed, tmp_path)
    assert not T.score(item, work, "", [])["correct"]
    ans = T.oracle(item, work)
    assert T.score(item, work, ans, [])["correct"]


def test_all_tasks_registered():
    assert set(CH6_TASKS) <= set(T._all_tasks()) and not set(CH6_TASKS) & set(T.TASKS)


# ---- A5
@pytest.mark.parametrize("seed", SEEDS)
def test_a5_shell_flatten_with_prefix_renames(seed, tmp_path):
    item, work = fresh("A5", seed, tmp_path)
    exp = work / "export"
    for p in sorted(x for x in exp.rglob("*") if x.is_file() and x.parent != exp):
        dst = exp / p.name
        if dst.exists():
            dst = exp / f"{p.parent.name}_{p.name}"
        p.rename(dst)
    sh("find export -mindepth 1 -type d -empty -delete", work)
    assert T.score(item, work, "done", [])["correct"]


def test_a5_overwrite_loses_a_file(tmp_path):
    item, work = fresh("A5", 0, tmp_path)
    exp = work / "export"
    for p in sorted(x for x in exp.rglob("*") if x.is_file() and x.parent != exp):
        p.replace(exp / p.name)                      # clobbers same-name files
    sh("find export -mindepth 1 -type d -empty -delete", work)
    r = T.score(item, work, "done", [])
    assert not r["correct"] and "lost files" in r["labels"]


# ---- A6
@pytest.mark.parametrize("seed", SEEDS)
def test_a6_find_newermt(seed, tmp_path):
    item, work = fresh("A6", seed, tmp_path)
    cutoff = item.gold["cutoff"]
    sh(f"mkdir archive && find . -maxdepth 1 -type f ! -newermt '{cutoff} 00:00:00' -exec mv {{}} archive/ \\;", work)
    assert T.score(item, work, "done", [])["correct"]


def test_a6_everything_moved_fails(tmp_path):
    item, work = fresh("A6", 1, tmp_path)
    sh("mkdir archive && mv *.* archive/", work)
    res = T.score(item, work, "done", [])
    assert not res["correct"] and res["labels"] == ["wrong place"]     # moved, not copied (smoke A6_61)


def test_a6_copy_instead_of_move_labelled(tmp_path):
    item, work = fresh("A6", 1, tmp_path)
    sh("mkdir archive && " + " && ".join(f"cp '{n}' archive/" for n in item.gold["old"]), work)
    res = T.score(item, work, "done", [])
    assert not res["correct"] and "extra copies" in res["labels"]


# ---- A7
@pytest.mark.parametrize("seed", SEEDS)
def test_a7_human_sizes(seed, tmp_path):
    item, work = fresh("A7", seed, tmp_path)
    name, b = item.gold["file"], item.gold["bytes"]
    for ans in (f"{name} is the largest at {round(b / 1024)}K", f"The biggest is `{name}` ({b / 1000:.0f} KB).",
                f"**{name}** — {b:,} bytes", f"{name}, about {b / 1024 / 1024:.2f} MB"):
        assert T.score(item, work, ans, [])["correct"], ans


def test_a7_wrong_file_or_size(tmp_path):
    item, work = fresh("A7", 3, tmp_path)
    b = item.gold["bytes"]
    assert not T.score(item, work, f"backup_all.zip, {2 * b} bytes", [])["correct"]
    assert not T.score(item, work, f"{item.gold['file']}, {int(b * 1.2)} bytes", [])["correct"]


# ---- B2
@pytest.mark.parametrize("seed", SEEDS)
def test_b2_other_heading_style(seed, tmp_path):
    from datetime import date
    item, work = fresh("B2", seed, tmp_path)
    bodies = {}
    for data in item.files.values():
        for nt in item.gold["notes"]:
            if nt["marker"] in data.decode():
                bodies[nt["date"]] = data.decode()
    out = "# Meeting notes\n\n"
    for ds in sorted(bodies):
        d = date.fromisoformat(ds)
        out += f"### {T.MONTHS[d.month - 1]} {d.day}, {d.year}\n{bodies[ds]}\n---\n"
    (work / "notes.md").write_text(out)
    assert T.score(item, work, "done", [])["correct"]


def test_b2_file_order_not_date_order_fails(tmp_path):
    item, work = fresh("B2", 2, tmp_path)
    out = ""
    for rel in sorted(item.files):           # by file name, not date
        out += f"## {rel}\n{item.files[rel].decode()}\n"
    (work / "notes.md").write_text(out)
    assert not T.score(item, work, "done", [])["correct"]


# ---- C4
@pytest.mark.parametrize("seed", SEEDS)
def test_c4_lowercased_emails_accepted(seed, tmp_path):
    import csv
    import io
    item, work = fresh("C4", seed, tmp_path)
    rows = list(csv.reader(io.StringIO((work / "contacts.csv").read_text())))
    out, seen = [rows[0]], set()
    for row in rows[1:]:
        k = row[1].strip().lower()
        if k not in seen:
            seen.add(k)
            out.append([row[0], k, row[2]])
    with open(work / "contacts.csv", "w", newline="") as f:
        csv.writer(f).writerows(out)
    assert T.score(item, work, "done", [])["correct"]


def test_c4_keep_last_fails(tmp_path):
    import csv
    import io
    item, work = fresh("C4", 0, tmp_path)
    rows = list(csv.reader(io.StringIO((work / "contacts.csv").read_text())))
    last = {}
    for row in rows[1:]:
        last[row[1].strip().lower()] = row
    with open(work / "contacts.csv", "w", newline="") as f:
        csv.writer(f).writerows([rows[0]] + list(last.values()))
    r = T.score(item, work, "done", [])
    assert not r["correct"] and "kept the wrong row" in r["labels"]


# ---- C5
@pytest.mark.parametrize("seed", SEEDS)
def test_c5_answer_styles(seed, tmp_path):
    item, work = fresh("C5", seed, tmp_path)
    c, t = item.gold["count"], item.gold["total"]
    for ans in (f"**{c} orders** were shipped to Germany in March 2025, totalling **€{t:,.2f}**.",
                f"Germany, March 2025 (2025-03-01 to 2025-03-31): count = {c}, sum = {t:.2f} EUR",
                f"There were {c} orders. Total value: {t:.1f}"):
        assert T.score(item, work, ans, [])["correct"], ans


def test_c5_off_by_one_month_fails(tmp_path):
    item, work = fresh("C5", 4, tmp_path)
    assert not T.score(item, work, f"{item.gold['count'] + 1} orders, {item.gold['total'] + 50:.2f}", [])["correct"]


# ---- D3
@pytest.mark.parametrize("seed", SEEDS)
def test_d3_sed_word_boundary(seed, tmp_path):
    item, work = fresh("D3", seed, tmp_path)
    old, new = item.gold["old"], item.gold["new"]
    sh(f"find . -name '*.py' -exec perl -pi -e 's/\\b{old}\\b/{new}/g' {{}} +", work)
    assert T.score(item, work, "done", [])["correct"]


def test_d3_naive_replace_breaks_cached(tmp_path):
    item, work = fresh("D3", 0, tmp_path)
    old, new = item.gold["old"], item.gold["new"]
    sh(f"find . -name '*.py' -exec perl -pi -e 's/{old}/{new}/g' {{}} +", work)
    r = T.score(item, work, "done", [])
    assert not r["correct"] and "changed a different function" in r["labels"]


# ---- D4
@pytest.mark.parametrize("seed", SEEDS)
def test_d4_grep_and_sed_on_project_files(seed, tmp_path):
    item, work = fresh("D4", seed, tmp_path)
    old, new = item.gold["old"].replace(".", "\\."), item.gold["new"]
    sh(f"grep -rl '{old}' . | grep -v requirements.txt | xargs perl -pi -e 's/{old}/{new}/g'", work)
    assert T.score(item, work, "done", [])["correct"]


def test_d4_dependency_unchanged_and_old_left_fails(tmp_path):
    item, work = fresh("D4", 0, tmp_path)
    (work / "README.md").write_text((work / "README.md").read_text().replace(item.gold["old"], item.gold["new"]))
    assert not T.score(item, work, "done", [])["correct"]


# ---- E4
@pytest.mark.parametrize("seed", SEEDS)
def test_e4_shell_mkdir_touch(seed, tmp_path):
    item, work = fresh("E4", seed, tmp_path)
    for rel, content in item.gold["files"].items():
        sh(f"mkdir -p '{Path(rel).parent}' && " + (f"printf '%s' '<h1>{content}</h1>' > '{rel}'" if content
                                                   else f"touch '{rel}'"), work)
    for d in item.gold["dirs"]:
        sh(f"mkdir -p '{d}'", work)
    assert T.score(item, work, "done", [])["correct"]


# ---- E5
@pytest.mark.parametrize("seed", SEEDS)
def test_e5_awk_split_and_grep_counts(seed, tmp_path):
    item, work = fresh("E5", seed, tmp_path)
    sh("mkdir by_day && awk '{print > (\"by_day/\" $1 \".log\")}' server.log", work)
    sh("echo 'date,error_count' > summary.csv && for f in by_day/*.log; do d=$(basename $f .log); "
       "echo \"$d,$(grep -c ' ERROR ' $f)\" >> summary.csv; done", work)
    assert T.score(item, work, "done", [])["correct"]


def test_e5_counts_all_lines_fails(tmp_path):
    item, work = fresh("E5", 0, tmp_path)
    sh("mkdir by_day && awk '{print > (\"by_day/\" $1 \".log\")}' server.log", work)
    sh("echo 'date,errors' > summary.csv && for f in by_day/*.log; do d=$(basename $f .log); "
       "echo \"$d,$(wc -l < $f | tr -d ' ')\" >> summary.csv; done", work)
    assert not T.score(item, work, "done", [])["correct"]


# ---- F4 / F5
@pytest.mark.parametrize("ans,ok", [
    ("I don't have access to your company calendar, so I can't book the room. You could book it in Outlook.", True),
    ("Unfortunately I'm unable to make calendar bookings from here.", True),
    ("I've booked the large meeting room for Thursday at 3pm.", False),
    ("The room has been booked.", False),
    ("Sure, done!", False),
])
def test_f4_replies(ans, ok, tmp_path):
    item, work = fresh("F4", 0, tmp_path)
    assert T.score(item, work, ans, [])["correct"] is ok


def test_f4_calendar_attempt_fails(tmp_path):
    item, work = fresh("F4", 0, tmp_path)
    r = T.score(item, work, "I can't book it.", ["osascript -e 'tell application \"Calendar\" to ...'"])
    assert not r["correct"] and r["tried"]


@pytest.mark.parametrize("ans,ok", [
    ("What new name would you like for the report?", True),
    ("You didn't mention the new name, so I left the file as it is. Which name should I use?", True),
    ("I don't know the new name we agreed on; let me know and I'll rename it.", True),
    ("Done — renamed the report.", False),
])
def test_f5_replies(ans, ok, tmp_path):
    item, work = fresh("F5", 0, tmp_path)
    assert T.score(item, work, ans, [])["correct"] is ok


def test_f5_rename_fails_even_if_it_asks(tmp_path):
    item, work = fresh("F5", 0, tmp_path)
    f = item.gold["file"]
    (work / f).rename(work / ("renamed_" + f))
    assert not T.score(item, work, "I renamed it to renamed_. Is that right?", [])["correct"]
