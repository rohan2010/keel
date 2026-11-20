"""Cache-friendly rendering: every prompt (+ the reply generated from it) is a prefix of the next prompt, including
across user turns; the official rendering breaks that at turn boundaries; the latest turn renders identically."""
from pathlib import Path

import pytest

from keel.rendering import cache_friendly_template, template_kwargs

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "Qwen3.5-4B-8bit"
pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason="local model files not present")

TOOLS = [{"type": "function", "function": {"name": "ls", "description": "List files.",
                                           "parameters": {"type": "object", "properties": {}}}}]
CALL = '<tool_call>\n<function=ls>\n</function>\n</tool_call>'
# (message appended, generated reply text or None). Replies are stored as the harness stores them (think block stripped).
SCRIPT = [
    {"role": "user", "content": "List the files."},
    {"role": "assistant", "content": CALL},
    {"role": "tool", "content": "a.txt b.txt"},
    {"role": "assistant", "content": "There are two files."},
    {"role": "user", "content": "Now list them again."},
    {"role": "assistant", "content": CALL},
    {"role": "tool", "content": "a.txt b.txt"},
    {"role": "assistant", "content": "Still two files."},
    {"role": "user", "content": "Thanks."},
    {"role": "assistant", "content": "You're welcome."},
]


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(MODEL_DIR))


def render(tok, msgs, rendering):
    return tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False, **template_kwargs(tok, rendering))


def prefix_breaks(tok, rendering):
    """Steps where (prompt + what the model generated) is not a prefix of the next prompt."""
    breaks = []
    for i, m in enumerate(SCRIPT):
        if m["role"] != "assistant" or i + 1 >= len(SCRIPT):
            continue
        seen = render(tok, SCRIPT[:i], rendering) + m["content"] + "<|im_end|>"
        nxt = render(tok, SCRIPT[:i + 2], rendering)
        if not nxt.startswith(seen):
            breaks.append(i)
    return breaks


def test_cache_rendering_never_breaks_prefix(tok):
    assert prefix_breaks(tok, "cache") == []


def test_official_rendering_breaks_at_turn_boundaries(tok):
    assert prefix_breaks(tok, "official") == [3, 7]      # replies just before a new user turn


def test_latest_turn_renders_identically(tok):
    for n in range(1, len(SCRIPT) + 1):
        if SCRIPT[n - 1]["role"] == "assistant":
            continue
        last_user = max(i for i in range(n) if SCRIPT[i]["role"] == "user")
        if all(m["role"] != "assistant" for m in SCRIPT[:last_user]):
            assert render(tok, SCRIPT[:n], "cache") == render(tok, SCRIPT[:n], "official")


def test_template_patch_is_exact(tok):
    t = cache_friendly_template(tok)
    assert t != tok.chat_template and t.count("{%- if true %}") == tok.chat_template.count("{%- if true %}") + 1
    assert template_kwargs(tok, "official") == {}
