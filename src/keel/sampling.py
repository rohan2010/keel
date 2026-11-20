"""Sampling that matches vLLM (so runs compare with vLLM-served models), built on MLX.

Two differences from mlx-lm's defaults matter here and are fixed in this module:

1. Order of operations. mlx-lm's `make_sampler` applies top-p to the *untempered* distribution, then top-k, then
   temperature. vLLM applies penalties -> temperature -> top-k -> top-p. With temperature 0.6 the nucleus differs.
2. Presence penalty. mlx-lm's `make_presence_penalty` looks only at the last 20 tokens of the context (prompt
   included). vLLM subtracts the penalty once from every token that occurs anywhere in the *generated* output so far.
   Small models loop, and a penalty that only sees 20 tokens does not stop a long loop, so this is not cosmetic.
"""

from __future__ import annotations

import hashlib

import mlx.core as mx

from .specs import SamplingSpec


def _sample(spec: SamplingSpec, logprobs: mx.array, key=None) -> mx.array:
    """tokens[B] from logprobs[B, V]: temperature, then top-k, then top-p (vLLM order)."""
    temp, top_k, top_p = spec.temperature, spec.top_k, spec.top_p
    if temp == 0:
        return mx.argmax(logprobs, axis=-1)
    x = logprobs.astype(mx.float32) / temp
    if top_k and 0 < top_k < x.shape[-1]:
        idx = mx.argpartition(-x, kth=top_k - 1, axis=-1)[..., :top_k]
    else:
        idx = mx.broadcast_to(mx.arange(x.shape[-1]), x.shape)
    vals = mx.take_along_axis(x, idx, axis=-1)
    # sort the candidates in descending order so top-p can be applied on the cumulative mass
    order = mx.argsort(-vals, axis=-1)
    vals = mx.take_along_axis(vals, order, axis=-1)
    idx = mx.take_along_axis(idx, order, axis=-1)
    if 0 < top_p < 1.0:
        probs = mx.softmax(vals, axis=-1)
        mass_before = mx.cumsum(probs, axis=-1) - probs
        vals = mx.where(mass_before < top_p, vals, -mx.inf)    # always keeps the top token
    choice = mx.random.categorical(vals, axis=-1, key=key)
    return mx.take_along_axis(idx, choice[..., None], axis=-1).squeeze(-1)


def make_vllm_sampler(spec: SamplingSpec):
    """sampler(logprobs[B, V]) -> tokens[B], drawing from MLX's global random state."""
    return lambda logprobs: _sample(spec, logprobs)


def greedy(logprobs: mx.array) -> mx.array:
    return mx.argmax(logprobs, axis=-1)


def episode_base(seed: int, label: str) -> int:
    """Stable 63-bit number for one episode (run seed + episode id); same in every arm and every run."""
    return int.from_bytes(hashlib.sha1(f"{seed}|{label}".encode()).digest()[:8], "big") & (2**63 - 1)


class EpisodeSampler:
    """Per-episode random numbers, as the LAST logits processor of a sequence (the batch sampler is then greedy).

    The random key depends only on (run seed, episode id, position in the sequence). The position is the length of
    the token context mlx-lm passes to processors (prompt + generated so far, whole conversation), so it does not
    depend on scheduler timing, which episodes share the batch, or how many draws other episodes made. Two arms that
    feed an episode the same tokens draw the same random numbers, so their transcripts stay identical until the arms
    differ (e.g. a check message), up to small floating-point differences between batch compositions.

    Returns logits with 0 at the chosen token and -inf elsewhere, so the greedy batch sampler picks it."""

    def __init__(self, spec: SamplingSpec, base: int):
        self.spec, self.base = spec, base

    def key(self, pos: int):
        return mx.random.key((self.base ^ (pos * 0x9E3779B97F4A7C15)) & (2**63 - 1))

    def __call__(self, tokens, logits: mx.array) -> mx.array:
        tok = _sample(self.spec, logits, self.key(len(tokens)))
        hit = mx.arange(logits.shape[-1])[None, :] == tok[:, None]
        return mx.where(hit, mx.array(0.0, dtype=logits.dtype), mx.array(-mx.inf, dtype=logits.dtype))


class PresencePenalty:
    """vLLM-style presence penalty for one sequence, as an mlx-lm logits processor.

    mlx-lm's BatchGenerator calls processor(context_tokens, logits[1, V]) once per step, where context_tokens is the
    prompt followed by every token generated so far (verified in scripts/check_mlx.py). We keep a running penalty
    vector over the vocabulary and add only the newly generated tokens each call, so a step costs O(new tokens).
    """

    def __init__(self, prompt_len: int, penalty: float):
        self.prompt_len = prompt_len
        self.penalty = penalty
        self.seen = 0              # generated tokens already folded into self.vec
        self.vec = None

    def __call__(self, tokens, logits: mx.array) -> mx.array:
        if self.vec is None:
            self.vec = mx.zeros((logits.shape[-1],), dtype=mx.float32)
        n_gen = len(tokens) - self.prompt_len
        if n_gen > self.seen:
            new = tokens[self.prompt_len + self.seen:]
            new = new if isinstance(new, mx.array) else mx.array(list(new))
            self.vec[new] = self.penalty
            self.seen = n_gen
        return logits - self.vec.astype(logits.dtype)


def make_processors(spec: SamplingSpec, prompt_len: int) -> list:
    return [PresencePenalty(prompt_len, spec.presence_penalty)] if spec.presence_penalty else []
