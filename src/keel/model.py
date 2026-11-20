"""Load a model and build chat prompts."""

from __future__ import annotations

from mlx_lm.utils import load as mlx_load

from .specs import ModelSpec


def load_model(spec: ModelSpec):
    """Return (model, tokenizer). A missing model says how to get it instead of an error from deep inside
    the loader."""
    from .modelstore import MODELS, complete
    path = spec.path
    if not complete(path):
        get = "`keel download`" + ("" if spec.name == "qwen3.5-4b-8bit" else f" --model {spec.name}")
        known = spec.name in MODELS
        raise SystemExit(f"the model {spec.name} is not on this Mac yet"
                         + (f": run {get} (it shows the size and asks first)" if known else f" (looked in {path})"))
    model, tok = mlx_load(str(path))
    model.eval()
    return model, tok


def stop_token_ids(tok) -> list[int]:
    ids = set(tok.eos_token_ids or [])
    for t in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0:
            ids.add(tid)
    return sorted(ids)


def build_prompt_tokens(tok, user_message: str, spec: ModelSpec, system: str | None = None) -> list[int]:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user_message}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, **spec.chat_template_kwargs)
