"""Chapter 6 Day 0: the system prompt can't grow silently.

A 4B with a 24-32k window pays for every prompt token on every step, and Chapter 5 found an undocumented sentence
in the files_py prompt only after it moved a task 0 -> 4/4. These budgets are ~10% above the 2026-11-19 sizes
(Qwen tokenizer: shell 160, files 906, files_py 1,097, examples 296 tokens; full stack ~1,390). Counted in characters so the test
runs without the model. Raising a budget is allowed, but only with a change-log line saying what was added and why.
"""

import json

import pytest

from keel.agent.examples import EXAMPLES
from keel.agent.loop import TOOLSETS

BUDGET_CHARS = {"shell": 800, "files": 3_900, "files_py": 4_800}   # system prompt + tool definitions (JSON)
EXAMPLES_BUDGET_CHARS = 1_100


@pytest.mark.parametrize("name", sorted(BUDGET_CHARS))
def test_system_prompt_and_tools_stay_within_budget(name):
    prompt, schemas, _ = TOOLSETS[name]
    size = len(prompt) + len(json.dumps(schemas))
    assert size <= BUDGET_CHARS[name], (
        f"{name}: prompt + tool definitions are {size} chars (budget {BUDGET_CHARS[name]}). If the growth is wanted, "
        "raise the budget here and add a change-log line saying what was added and why.")


def test_examples_stay_within_budget():
    assert len(EXAMPLES) <= EXAMPLES_BUDGET_CHARS, (
        f"G6 examples are {len(EXAMPLES)} chars (budget {EXAMPLES_BUDGET_CHARS}); raise only with a change-log line.")


def test_every_toolset_has_a_budget():
    assert set(TOOLSETS) == set(BUDGET_CHARS), "a new toolset needs a prompt-size budget"
