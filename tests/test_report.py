"""Chapter 4 P4: the report card (Markdown + 80-column terminal + JSON)."""

import json

from test_compare import ADMIT, DONE, make

from keel.evals import report
from keel.evals.compare import compare_arms, load_arm


def _comparisons(tmp):
    gain = compare_arms(load_arm(make(tmp, "a", [False] * 30)),
                        load_arm(make(tmp, "b", [True] * 10 + [False] * 20, answers=[DONE] * 30)))
    tries = compare_arms(load_arm(",".join(make(tmp, f"c{s}", [False] * 20, seed=s) for s in (0, 1))),
                         load_arm(",".join(make(tmp, f"d{s}", [True] * 12 + [False] * 8, seed=s) for s in (0, 1))))
    cant = compare_arms(load_arm(make(tmp, "e", [True] * 5)), load_arm(make(tmp, "f", [True] * 5, batch=4)))
    return [gain, tries, cant]


def test_markdown_puts_the_verdict_first_and_tables_are_well_formed(tmp_path):
    for c in _comparisons(tmp_path):
        md = report.markdown(c)
        lines = md.splitlines()
        assert lines[0] == f"# keel compare: {c.verdict}" and lines[2].startswith(f"**{c.verdict.upper()}**")
        tables = [ln for ln in lines if ln.startswith("|")]
        assert tables and all(ln.count("|") == 5 for ln in tables)         # 4 columns everywhere
        assert "| pass@1 |" in md


def test_terminal_card_fits_80_columns(tmp_path):
    for c in _comparisons(tmp_path):
        t = report.terminal(c)
        assert t.splitlines()[0].startswith(c.verdict.upper())
        assert max(len(ln) for ln in t.splitlines()) <= 80, [ln for ln in t.splitlines() if len(ln) > 80]


def test_tries_show_pass_at_k_and_cant_compare_shows_why(tmp_path):
    gain, tries, cant = _comparisons(tmp_path)
    assert "pass@2" in report.markdown(tries) and "all 2 tries pass" in report.markdown(tries)
    assert "batch_size" in report.terminal(cant) and "BLOCKS" in report.terminal(cant)
    assert "Fixed (10)" in report.terminal(gain)


def test_write_makes_report_md_and_json(tmp_path):
    c = _comparisons(tmp_path)[0]
    out = report.write(c, tmp_path / "out")
    assert (out / "report.md").read_text().startswith("# keel compare")
    assert json.loads((out / "report.json").read_text())["verdict"] == c.verdict


def test_false_success_change_is_always_in_the_reason(tmp_path):
    """No pass effect, but false success clearly up: the verdict line has to say so."""
    a = make(tmp_path, "g", [True] * 10 + [False] * 20, answers=[ADMIT] * 30)
    b = make(tmp_path, "h", [True] * 10 + [False] * 20, answers=[DONE] * 30)
    c = compare_arms(load_arm(a), load_arm(b))
    assert c.verdict == "no detectable effect" and "false success rose" in c.reason
