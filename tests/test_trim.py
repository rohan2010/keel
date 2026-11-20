"""Module 4 trimming (src/keel/trim.py): untouched below the threshold, cut at a separator with a size note above it,
the real BFCL giant list shrinks to the preview, and inside a BFCL episode only the model's view changes."""

import copy
import glob
import json

import pytest

from keel.trim import KEEP_TOKENS, MAX_TOKENS, trim

words = lambda s: len(s.split())          # a stand-in token counter: 1 token per whitespace-separated word


def test_short_output_untouched():
    t = "a b c"
    assert trim(t, words) == (t, None)


def test_long_list_cut_at_separator_with_note():
    items = [f"'T{i}'" for i in range(5000)]
    text = "[" + ", ".join(items) + "]"
    out, info = trim(text, words)
    assert info["total"] == 5000 and info["items"] == 5000
    body, note = out.split("\n[Output shortened", 1)
    assert body.endswith(",") or body.endswith("'")          # ends on a complete item
    assert words(body) <= KEEP_TOKENS
    assert f"of {info['total']} tokens" in note and "of 5000 items" in note


def test_multiline_cut_at_line():
    text = "\n".join(f"line {i} " + "x " * 5 for i in range(1000))
    out, info = trim(text, words)
    body = out.split("\n[Output shortened", 1)[0]
    assert body.split("\n")[-1].startswith("line ") and info["items"] is None


def _tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("models/Qwen3.5-4B-8bit")
    except Exception:
        pytest.skip("tokenizer not available")


def _giant():
    pat = "20261108-120706_bfcl_mt_dev_base_*/transcripts/*long_context_108.json"
    for f in sorted(glob.glob(f"experiments/{pat}") + glob.glob(f"results/runs/{pat}")):
        t = json.load(open(f))
        for res in t["tool_results"]:
            for outs in res:
                for o in outs:
                    if len(o) > 20000:
                        return o
    pytest.skip("dev transcript not available")


def test_real_bfcl_giant_list():
    from keel.trim import make_trimmer
    tok = _tok()
    big = _giant()
    out, info = make_trimmer(tok)(big)
    assert info["total"] > 30000
    assert len(tok.encode(out, add_special_tokens=False)) < MAX_TOKENS
    assert out.startswith(big[:200]) and '"price":' in out          # the part the task needs is kept


def test_bfcl_episode_view_only():
    from keel.evals import bfcl_mt as M
    from keel.nudges import LOOP_NOTE
    e = copy.deepcopy([r for r in M.load("miss_func") if r["id"] == "multi_turn_miss_func_1"][0])
    long_text = "[" + ", ".join(f"'X{i}'" for i in range(3000)) + "]"
    ep = M.MTEpisode(e, M.safe_name("test_m4_view"), loop_break=True, trimmer=lambda r: trim(r, words)).start()
    import keel.evals.bfcl_mt as mod
    real = mod.execute_multi_turn_func_call
    mod.execute_multi_turn_func_call = lambda calls, *a, **k: ([long_text for _ in calls], None)
    try:
        call = "<tool_call>\n<function=ls>\n<parameter=a>\nTrue\n</parameter>\n</function>\n</tool_call>"
        for _ in range(3):
            ep.apply(call, "stop")
    finally:
        mod.execute_multi_turn_func_call = real
    assert ep.tool_results[0][0][0] == long_text                # executed result kept in full (scoring)
    seen = ep.messages[-1]["content"]
    assert "[Output shortened" in seen and seen.endswith(LOOP_NOTE)   # note survives trimming
    assert any(n.startswith("trim@t0") for n in ep.notes)
