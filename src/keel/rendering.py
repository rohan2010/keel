"""Cache-friendly chat rendering for multi-turn BFCL dev runs (user decision 2026-11-10).

Qwen3.5's official template writes the empty `<think>\n\n</think>\n\n` block only on assistant messages after the
latest user turn. When a new user turn arrives, earlier assistant messages lose that block, so the rendered prefix
no longer matches what the model already processed and the hybrid model's recurrent state cannot be rewound: full
re-prefill at every turn boundary.

The cache-friendly rendering keeps the think block on every assistant message, i.e. history is rendered exactly as
the model generated it. Dev runs only, and only after a one-off equivalence check against the official rendering;
Milestone (val) and holdout runs keep the official rendering.
"""

OFFICIAL_COND = "{%- if loop.index0 > ns.last_query_index %}"
CACHE_COND = "{%- if true %}"
RENDERINGS = ("official", "cache")


def cache_friendly_template(tok) -> str:
    tmpl = tok.chat_template
    if not isinstance(tmpl, str) or tmpl.count(OFFICIAL_COND) != 1:
        raise ValueError("chat template does not have the expected think-block condition; cache rendering not applicable")
    return tmpl.replace(OFFICIAL_COND, CACHE_COND)


def template_kwargs(tok, rendering: str) -> dict:
    """Extra kwargs for tok.apply_chat_template: {} for official, {'chat_template': ...} for cache."""
    if rendering == "official":
        return {}
    if rendering == "cache":
        return {"chat_template": cache_friendly_template(tok)}
    raise ValueError(rendering)
