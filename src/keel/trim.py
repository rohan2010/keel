"""Trimming long tool outputs.

Trim long tool outputs (`--trim-outputs`): a single tool result longer than MAX_TOKENS is cut to about KEEP_TOKENS,
at the last line break or list separator, and a note with the sizes is appended. Nothing is saved for later lookup.
Only what the model sees changes; executed results (and so BFCL scoring) are untouched. Off by default.
"""

from __future__ import annotations

MAX_TOKENS = 2000
KEEP_TOKENS = 1500
NOTE = ("\n[Output shortened by the system: showing the first {shown} of {total} tokens{items}. Ask a narrower "
        "question if you need something further in.]")



def _items(text: str) -> int | None:
    """Item count for a single flat list printed on one line ("['A', 'B']" or '{"k": ["A", "B"]}'); else None."""
    t = text.strip()
    if "\n" in t or "[" not in t or t.count("[") != 1:
        return None
    body = t[t.index("[") + 1:t.rindex("]")] if "]" in t else t[t.index("[") + 1:]
    return body.count(",") + 1 if body.strip() else 0


def trim(text: str, count, max_tokens: int = MAX_TOKENS, keep: int = KEEP_TOKENS) -> tuple[str, dict | None]:
    """`count(text) -> int` counts tokens. Returns (text the model sees, info or None if untouched)."""
    total = count(text)
    if total <= max_tokens:
        return text, None
    lo, hi = 0, len(text)                     # longest character prefix with <= keep tokens (binary search)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(text[:mid]) <= keep:
            lo = mid
        else:
            hi = mid - 1
    prefix = text[:lo]
    cut = max(prefix.rfind("\n"), prefix.rfind(","))   # end at a complete line or list item when possible
    if cut > lo // 2:
        prefix = prefix[:cut + (1 if prefix[cut] == "," else 0)]     # keep a trailing comma, drop a trailing newline
    n_all = _items(text)
    items = ""
    if n_all is not None:
        n_shown = prefix[prefix.index("[") + 1:].count(",") + 1 if "[" in prefix else 0
        items = f" ({n_shown} of {n_all} items)"
    shown = count(prefix)
    return prefix + NOTE.format(shown=shown, total=total, items=items), {"total": total, "shown": shown,
                                                                       "items": n_all}


def make_trimmer(tok):
    """Trimmer bound to a tokenizer (mlx-lm TokenizerWrapper or HF tokenizer)."""
    def count(s: str) -> int:
        return len(tok.encode(s, add_special_tokens=False))
    return lambda text: trim(text, count)


def trimmer_for(model_dir):
    """Trimmer for backends without MLX: counts with the model folder's `tokenizer.json`, or
    characters / 2.0 when there is none. Returns (trimmer, counter record for the run config)."""
    from .tokens import load_counter
    counter = load_counter(model_dir)
    return (lambda text: trim(text, counter)), counter.record()
