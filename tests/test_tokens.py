"""Chapter 3, P4: token counts without MLX. The `tokenizers` counter reads the model folder's tokenizer.json and must
match the mlx-lm / transformers tokenizer exactly, so trimming decides the same on every backend. (The full check over
13,112 research tool outputs: doc/product/spikes/p4-trim/check_counts.py -> 0 count and 0 trim differences.)"""

import glob
import json
import subprocess
import sys
from pathlib import Path

import pytest

from keel.specs import QWEN35_4B_8BIT as SPEC
from keel.tokens import CHARS_PER_TOKEN, approx_count, load_counter
from keel.trim import MAX_TOKENS, make_trimmer, trim, trimmer_for

MODEL = Path(SPEC.local_dir)
needs_model = pytest.mark.skipif(not (MODEL / "tokenizer.json").is_file(), reason="local model folder not present")


def _hf():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(MODEL))
    except Exception:
        pytest.skip("transformers tokenizer not available")


def _samples():
    long_list = "[" + ", ".join(f"'T{i}'" for i in range(5000)) + "]"
    lines = "\n".join(f"line {i} " + "x " * 5 for i in range(1000))
    texts = ["", "hello keel\n", "  indented\n\ttabs  ", "日本語のテキスト and émojis 🚀", "<tool_call>\n<function=x>",
             "<|im_end|> special text", json.dumps({"k": list(range(300))}), long_list, lines]
    for f in sorted(glob.glob("results/runs/*/transcripts/*.json"))[:40]:      # some real tool output, if present
        try:
            texts += [m["content"] for m in json.load(open(f)).get("messages", []) if m.get("role") == "tool"]
        except Exception:
            pass
    return texts


@needs_model
def test_tokenizer_json_counts_match_the_model_tokenizer():
    hf, c = _hf(), load_counter(MODEL)
    assert c.method == "tokenizer.json" and not c.approximate
    for t in _samples():
        assert c(t) == len(hf.encode(t, add_special_tokens=False)), t[:80]


@needs_model
def test_same_trim_decisions_as_the_model_tokenizer():
    hf = _hf()
    old = make_trimmer(hf)
    new, rec = trimmer_for(MODEL)
    assert rec == {"method": "tokenizer.json", "approximate": False, "source": str(MODEL / "tokenizer.json")}
    n_trimmed = 0
    for t in _samples():
        a, b = old(t), new(t)
        assert a == b, t[:80]
        n_trimmed += a[1] is not None
    assert n_trimmed >= 2                                                   # the long samples really were trimmed


def test_fallback_is_approximate_and_marked(tmp_path):
    for where in (None, tmp_path, tmp_path / "missing"):
        c = load_counter(where)
        assert c.approximate and c.method == f"chars/{CHARS_PER_TOKEN}" and c.record()["approximate"] is True
    (tmp_path / "tokenizer.json").write_text("not a tokenizer")             # unreadable file: fallback, no crash
    assert load_counter(tmp_path).approximate
    assert approx_count("") == 0 and approx_count("abcde") == 3
    trimmer, rec = trimmer_for(None)
    text = "x " * (MAX_TOKENS * 2)
    out, info = trimmer(text)
    assert rec["approximate"] and info is not None and "[Output shortened" in out
    assert trim("short", approx_count) == ("short", None)


def test_counter_imports_without_mlx_or_transformers():
    code = f"""
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('mlx', 'mlx_lm', 'transformers', 'torch'):
            raise ImportError('blocked: ' + name)
sys.meta_path.insert(0, Block())
from keel.trim import trimmer_for
t, rec = trimmer_for({str(MODEL)!r})
print(rec['method'], t('a b c')[1])
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    want = "tokenizer.json" if (MODEL / "tokenizer.json").is_file() else f"chars/{CHARS_PER_TOKEN}"
    assert r.stdout.split() == [want, "None"]
