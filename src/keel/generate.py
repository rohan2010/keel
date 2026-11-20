"""Batched sampling with mlx-lm's continuous-batching BatchGenerator, vLLM-style sampling, and timing/memory stats."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import ArraysCache

from .model import stop_token_ids
from .sampling import make_processors, make_vllm_sampler
from .specs import SamplingSpec


@dataclass
class Generation:
    index: int                 # position in the input list
    text: str
    n_tokens: int              # generated tokens (stop token excluded)
    prompt_tokens: int
    finish_reason: str         # "stop" or "length"
    seconds: float             # wall time from submission of the batch until this sequence finished

    @property
    def hit_cap(self) -> bool:
        return self.finish_reason == "length"


@dataclass
class GenStats:
    n: int
    generated_tokens: int
    prompt_tokens: int
    seconds: float
    peak_memory_gb: float

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.seconds if self.seconds else 0.0

    def to_dict(self) -> dict:
        return {**self.__dict__, "tokens_per_second": self.tokens_per_second}


def _flush_lazy_cache_state(gen: BatchGenerator) -> None:
    """Work around an mlx-lm 0.32 leak. ArraysCache.advance() (the linear-attention layers of hybrid models such as
    Qwen3.5) does `left_padding -= N` / `lengths -= N` every decode step, but nothing reads those arrays while
    decoding, so they become an ever-longer unevaluated graph. Every pending node pins a Metal buffer, and after
    ~30k steps x 18 layers MLX hits its 499,000-buffer limit (`[metal::malloc] Resource limit exceeded`): this
    crashed three 8k-token baseline runs at the same point. Evaluating them collapses the chain; values unchanged."""
    batch = getattr(gen, "_generation_batch", None)
    if batch is None:
        raise RuntimeError("mlx-lm internals changed: BatchGenerator._generation_batch missing; recheck this leak")
    arrs = [a for c in batch.prompt_cache if isinstance(c, ArraysCache)
            for a in (c.left_padding, c.lengths) if a is not None]
    if arrs:
        mx.eval(arrs)


def generate_batch(model, tok, prompts: list[list[int]], sampling: SamplingSpec, *, batch_size: int = 32,
                   on_done: Callable[[Generation], None] | None = None, progress_every: float = 60.0,
                   flush_every: int = 256, prefill_batch: int = 8,
                   prefill_step: int = 2048) -> tuple[list[Generation], GenStats]:
    """Sample one completion per prompt. Up to `batch_size` sequences decode concurrently; finished slots are
    refilled from the queue (continuous batching). `on_done` is called as each sequence finishes (for streaming
    results to disk so a crash loses nothing)."""
    mx.random.seed(sampling.seed)
    mx.reset_peak_memory()
    gen = BatchGenerator(
        model,
        max_tokens=sampling.max_tokens,
        stop_tokens=[[t] for t in stop_token_ids(tok)],
        sampler=make_vllm_sampler(sampling),
        completion_batch_size=batch_size,
        prefill_batch_size=min(prefill_batch, batch_size),
        prefill_step_size=prefill_step,     # smaller = lower peak memory at prompt time, same outputs
    )
    t0 = time.perf_counter()
    uids = gen.insert(prompts, logits_processors=[make_processors(sampling, len(p)) for p in prompts])
    pos = {u: i for i, u in enumerate(uids)}
    toks: dict[int, list[int]] = {u: [] for u in uids}
    results: list[Generation | None] = [None] * len(prompts)
    done = total_gen = 0
    last_report = t0
    stops = set(stop_token_ids(tok))
    steps = 0
    try:
        while responses := gen.next_generated():
            steps += 1
            if steps % flush_every == 0:
                _flush_lazy_cache_state(gen)
            for r in responses:
                if r.token not in stops:
                    toks[r.uid].append(r.token)
                if r.finish_reason is not None:
                    i = pos[r.uid]
                    g = Generation(i, tok.decode(toks[r.uid]), len(toks[r.uid]), len(prompts[i]),
                                   "length" if r.finish_reason == "length" else "stop", time.perf_counter() - t0)
                    results[i] = g
                    total_gen += g.n_tokens
                    done += 1
                    toks[r.uid] = []
                    if on_done:
                        on_done(g)
            now = time.perf_counter()
            if progress_every and now - last_report > progress_every:
                inflight = sum(len(v) for v in toks.values())
                print(f"  [{now - t0:7.0f}s] {done}/{len(prompts)} done, "
                      f"{(total_gen + inflight) / (now - t0):.0f} tok/s, peak {mx.get_peak_memory() / 1e9:.1f} GB",
                      flush=True)
                last_report = now
    finally:
        gen.close()
    secs = time.perf_counter() - t0
    assert all(r is not None for r in results)
    stats = GenStats(len(prompts), total_gen, sum(len(p) for p in prompts), secs, mx.get_peak_memory() / 1e9)
    return results, stats  # type: ignore[return-value]
