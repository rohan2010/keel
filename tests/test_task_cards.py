"""Chapter 4 P7: one card per agent task, generated from the task code and kept up to date."""

import subprocess
import sys
from pathlib import Path

from keel.evals import tasks as T

ROOT = Path(__file__).resolve().parents[1]


def test_cards_are_up_to_date():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_task_cards.py"), "--check"], capture_output=True,
                       text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


def test_every_task_has_a_card_and_a_scoring_summary():
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_task_cards as M
    assert set(M.SCORING) == set(M.REG) and set(T.TASKS) <= set(M.REG)     # original 13 + Chapter 6's 12
    for t in M.REG:
        text = (ROOT / "task-cards" / f"{t}.md").read_text()
        assert text.startswith(f"# {t}: ") and "## How it is scored" in text and "| holdout | 5000+ |" in text
