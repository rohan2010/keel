"""In-process MLX backend (the only path keel's validated numbers come from).

`run_scheduler` is the research scheduler, moved here unchanged from `agent/loop.py`: continuous
batching with per-episode KV-cache reuse, a memory cap on (episodes generating) x (longest context), tool calls in
worker threads, per-episode random streams. `MLXBackend` wraps it behind `keel.core.backend.Backend`.
MLX is imported only here (and in the modules this file imports), so `import keel` works without MLX.
"""

from __future__ import annotations

import copy
import time
from collections import OrderedDict

import mlx.core as mx
import numpy as np
from mlx_lm.generate import BatchGenerator

from ..core.backend import Capabilities, Price, RequestStats, RunHooks, RunLimits
from ..generate import _flush_lazy_cache_state
from ..model import stop_token_ids
from ..sampling import EpisodeSampler, episode_base, greedy, make_processors, make_vllm_sampler
from ..specs import SamplingSpec

CAPABILITIES = Capabilities("mlx", forbid_tools_next_turn=True, hidden_states=True, logit_bias=True, seeded=True,
                            state_save=True, batched=True, validated=True)


class BanToolCall:
    """Logits processor: <tool_call> can never be sampled (must run before the episode sampler). The mask is built
    from the logits' own width (the model's vocab can be padded beyond the tokenizer's)."""

    def __init__(self, tok_id: int):
        self.tok_id = tok_id
        self.vec = None

    def __call__(self, tokens, logits):
        if self.vec is None or self.vec.shape[-1] != logits.shape[-1]:
            v = np.zeros((logits.shape[-1],), dtype=np.float32)
            v[self.tok_id] = -np.inf
            self.vec = mx.array(v)
        return logits + self.vec.astype(logits.dtype)


def run_scheduler(model, tok, sampling: SamplingSpec, queue: list, *, prompt_fn, apply_fn, finish_fn,
                  max_context: int = 16000, batch_size: int = 6, prefill_step: int = 1024, prefill_batch: int = 4,
                  context_budget: int = 48000, reuse_cache: bool = True, progress_every: float = 60.0,
                  label_fn=None, seed_fn=None, state_saves: bool = False, shared_prefix: bool = False,
                  on_token=None, handle=None, gate=None) -> dict:
    """Continuous scheduler with per-episode KV-cache reuse, for any multi-step episode.

    An episode is any object with the fields cache, seq, done, stop, prompt_tokens, processed_tokens, output_tokens,
    t_start, seconds (see `Episode`). `prompt_fn(e)` gives its prompt token ids; `apply_fn(e, raw_text,
    finish_reason)` (run in a worker thread) records a reply, runs its tool calls and sets e.done when it ends;
    `finish_fn(e)` records a finished episode. Episodes start in `queue` order.

    - Memory cap: sequences decoding together are padded to the longest one, so memory scales with
      (episodes generating) x (longest context), not with the sum. An episode joins the batch only while that product
      stays under `context_budget` token slots; otherwise it waits outside the batch holding only its own cache
      (continuing episodes before new ones). A 6-episode run without this peaked at 12.3 GB on a 16 GB Mac.
    - Tool calls run in worker threads, so other episodes keep generating meanwhile.
    - Random numbers: with `seed_fn(e)` (a stable episode id), every episode draws from its own random stream keyed
      by (sampling.seed, id, position), so results do not depend on timing or batch-mates, and two arms see the same
      draws for the same episode (`EpisodeSampler`). Without it, all episodes share MLX's global stream (old runs).
    None of this changes what an episode sees; only the order and timing of work.

    Streaming: with `on_token(e, text)`, each generated token is decoded incrementally (one streaming
    detokenizer per sequence) and the new text is passed on as it appears, for a live view. It only observes:
    generation, the recorded reply and the cache are the same with or without it, and `None` skips it entirely.

    State saves (/ both off by default, and off means the exact old code path). Qwen3.5's DeltaNet
    state cannot be rewound, so a prompt that is not an exact extension of the cached tokens used to be re-read from
    scratch. Instead the prompt is inserted in segments and the state is copied at segment ends:
    - `state_saves`: at the first step of a user turn (last message is from the user), a copy is kept at the start
      of the assistant header. The official template re-renders the previous turn's assistant messages when a new
      user turn arrives, so the end-of-reply cache stops matching; this copy still does, and only the last turn is
      re-read. One copy per episode (the latest turn).
    - `shared_prefix`: a copy at the end of the system+tools block, shared by every episode with the same block
      (LRU of 8), so each new episode skips re-reading it.
    Restored copies are deep-copied (KV caches write in place). Prefill chunks now break at segment ends, so logits
    can differ by rounding: check with a parity run, not byte identity.

    Joining a run (off = `handle=None` = the code path above, unchanged). With a `RunHandle`, an
    episode's apply_fn may `handle.submit(child)`: the child joins the queue at the next scheduler pass. An episode
    blocked in `handle.wait(...)` sits in a tool thread and does not count against `batch_size`, so its children can
    run even with one slot; the tool pool has room for such waiting episodes.

    Probe gate (experimental; off = `gate=None` = the code path above, unchanged). With a `ProbeGate`,
    every request is prefilled here instead of in the batch generator (on the episode's reused cache when its prompt
    extends it, else a fresh one): the new tokens except the last, then the last with a hook on the probe layer
    (`mlx_gate.decide`). The decision picks the first token ("force" = <tool_call>; "block" = <tool_call> banned for the
    whole reply; "free" = as usual, through the same processors and per-episode random stream), and cache + first
    token go to the batch generator. Decisions are counted in the stats and noted on the episode (`gate:<action>@<step>
    p=<p>`). Not combined with state saves / shared prefix."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    queue = list(queue)
    n_total, n_done = len(queue), 0
    stops = stop_token_ids(tok)
    mx.random.seed(sampling.seed)
    batch_sampler = greedy if seed_fn else make_vllm_sampler(sampling)
    gen = BatchGenerator(model, max_tokens=sampling.max_tokens, stop_tokens=[[t] for t in stops],
                         sampler=batch_sampler, completion_batch_size=batch_size,
                         prefill_batch_size=min(prefill_batch, batch_size), prefill_step_size=prefill_step)
    mx.reset_peak_memory()
    active: dict = {}                        # uid -> (episode, generated tokens)
    pending: dict = {}                       # future -> episode (tools running)
    stats = {"reused": 0, "full": 0, "waited_for_budget": 0}
    saving = state_saves or shared_prefix
    hooked = None
    if gate is not None:
        from .mlx_gate import Hooked, decide
        if saving:
            raise ValueError("the probe gate is not combined with state saves / shared prefix")
        hooked = Hooked(model, gate.layer)
        hook = hooked.__enter__()
        tool_call_id = tok.convert_tokens_to_ids("<tool_call>")
        stats["gate"] = {"force": 0, "block": 0, "free": 0}
    if saving:
        stats.update({"restored_turn": 0, "restored_prefix": 0, "saved": 0})
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    prefix_store: OrderedDict = OrderedDict()     # tuple(tokens) -> cache, shared across episodes (shared prefix)
    turn_store: dict = {}                         # id(episode) -> (tokens, cache), latest turn start (state saves)
    marks: dict = {}                              # uid -> [(absolute boundary, kind)] still to save
    detoks: dict = {}                             # uid -> streaming detokenizer (only with on_token)

    def stream(e, text) -> None:
        if text:
            try:
                on_token(e, text)
            except Exception as err:                # a display bug must never stop generation
                print(f"  [stream] on_token failed: {err!r}", flush=True)
    pool = ThreadPoolExecutor(max_workers=batch_size if handle is None else batch_size + 64)
    if handle is not None:
        handle.attach()
    t0 = last = time.perf_counter()

    def finish(e) -> None:
        nonlocal n_done
        n_done += 1
        e.seconds = time.perf_counter() - e.t_start
        finish_fn(e)
        e.cache = e.seq = None
        turn_store.pop(id(e), None)
        if handle is not None:
            handle.finished(e)

    ready: list = []                          # continuing episodes waiting for room in the batch
    alen: dict[int, int] = {}                 # uid -> context length of sequences in the batch

    def fits(n_tokens: int) -> bool:
        if not active:
            return True
        longest = max(max(alen.values(), default=0), n_tokens)
        return (len(active) + 1) * longest <= context_budget

    def submit(e) -> None:
        ids = prompt_fn(e)
        if len(ids) > max_context:
            e.done, e.stop = True, "context_limit"
            finish(e)
            return
        if not fits(len(ids)):
            stats["waited_for_budget"] += 1
            ready.append(e)
            return
        e.prompt_tokens += len(ids)
        procs = make_processors(sampling, len(ids))
        if getattr(e, "ban_tool_call", False):          # this reply must be text
            procs.append(BanToolCall(tok.convert_tokens_to_ids("<tool_call>")))
        if seed_fn:
            procs.append(EpisodeSampler(sampling, episode_base(sampling.seed, seed_fn(e))))
        if gate is not None:
            submit_gated(e, ids, procs)
            return
        procs = [procs]
        if saving:
            uid = submit_saving(e, ids, procs)
        elif reuse_cache and e.cache is not None and e.seq and ids[:len(e.seq)] == e.seq and len(ids) > len(e.seq):
            suffix = ids[len(e.seq):]
            uid = gen.insert([suffix], caches=[e.cache], all_tokens=[e.seq], logits_processors=procs)[0]
            e.processed_tokens += len(suffix)
            stats["reused"] += 1
        else:
            uid = gen.insert([ids], logits_processors=procs)[0]
            e.processed_tokens += len(ids)
            stats["full"] += 1
        e.cache = e.seq = None
        active[uid] = (e, [])
        alen[uid] = len(ids)

    def submit_gated(e, ids, procs) -> None:
        """P7: prefill here, let the probe decide, pick the first token, hand cache + first token to the generator."""
        if reuse_cache and e.cache is not None and e.seq and extends(ids, e.seq):
            cache, start = e.cache, len(e.seq)
            stats["reused"] += 1
        else:
            cache, start = model.make_cache(), 0
            stats["full"] += 1
        e.cache = e.seq = None
        d, logits = decide(model, gate, hook, ids[start:], cache, prefill_step)
        e.processed_tokens += len(ids) - start
        if d.action == "force" and getattr(e, "ban_tool_call", False):
            d.action = "free(loop ban)"             # the loop breaker's ban wins over the gate
        stats["gate"][d.action] = stats["gate"].get(d.action, 0) + 1
        if isinstance(getattr(e, "notes", None), list):
            e.notes.append(f"gate:{d.action}@{getattr(e, 'steps', 0) + 1} p={d.p_call:.3f}")
        if d.action == "block":                     # banned for the whole reply; the episode sampler stays last
            procs.insert(len(procs) - 1 if seed_fn else len(procs), BanToolCall(tool_call_id))
        if d.action == "force":
            first = tool_call_id
        else:
            lg = logits
            ctx = mx.array(ids, dtype=mx.int32)
            for p in procs:
                lg = p(ctx, lg)
            first = int(batch_sampler(lg - mx.logsumexp(lg, axis=-1, keepdims=True)).item())
        mx.clear_cache()
        if first in stops:                          # an empty reply: nothing to decode
            e.cache, e.seq = cache, list(ids)
            pending[pool.submit(apply_fn, e, "", "stop")] = e
            return
        uid = gen.insert([[first]], max_tokens=[sampling.max_tokens - 1], caches=[cache], all_tokens=[list(ids)],
                         logits_processors=[procs])[0]
        active[uid] = (e, [first])
        alen[uid] = len(ids) + 1
        if on_token is not None:
            dt = detoks[uid] = tok.detokenizer
            dt.add_token(first)
            stream(e, dt.last_segment)

    def extends(ids, toks) -> bool:
        return bool(toks) and len(ids) > len(toks) and ids[:len(toks)] == list(toks)

    def submit_saving(e, ids, procs) -> int:
        """Shared-prefix / state-save insert: pick the longest usable cached state, then mark the boundaries to save on the way."""
        start, cache, kind = 0, None, "full"
        if reuse_cache and e.cache is not None and extends(ids, e.seq):
            start, cache, kind = len(e.seq), e.cache, "reused"
        if state_saves and id(e) in turn_store:
            toks, c = turn_store[id(e)]
            if len(toks) > start and extends(ids, toks):
                start, cache, kind = len(toks), copy.deepcopy(c), "restored_turn"
        if shared_prefix and start == 0:
            for toks, c in prefix_store.items():
                if extends(ids, toks):
                    start, cache, kind = len(toks), copy.deepcopy(c), "restored_prefix"
                    prefix_store.move_to_end(toks)
                    break
        stats[kind] += 1
        want = []
        if shared_prefix:
            first = next((i for i in range(1, len(ids)) if ids[i] == im_start), None)
            if first is not None and tuple(ids[:first]) not in prefix_store:
                want.append((first, "prefix"))
        msgs = getattr(e, "messages", None)
        if state_saves and msgs and msgs[-1].get("role") == "user":
            last = max((i for i in range(len(ids)) if ids[i] == im_start), default=None)
            if last is not None:
                want.append((last, "turn"))
        want = sorted({b: k for b, k in want if start < b < len(ids) - 1}.items())
        cuts = [start] + [b for b, _ in want] + [len(ids)]
        segs = [ids[a:b] for a, b in zip(cuts, cuts[1:], strict=False)]
        uid = gen.insert_segments([segs], caches=[cache] if cache is not None else None,
                                  all_tokens=[list(ids[:start])] if cache is not None else None,
                                  logits_processors=procs)[0]
        e.processed_tokens += len(ids) - start
        if want:
            marks[uid] = (e, want)
        return uid

    def save_states(prompt_responses) -> None:
        done = [r.uid for r in prompt_responses if r.end_of_segment and not r.end_of_prompt and r.uid in marks]
        if not done:
            return
        for uid, (c, toks) in gen.extract_cache(done).items():
            e, want = marks[uid]
            b, kind = want.pop(0)
            toks = list(toks)
            if len(toks) != b:                      # never store a state under the wrong tokens
                raise RuntimeError(f"state save at {len(toks)} tokens, expected {b}")
            if kind == "prefix":
                prefix_store[tuple(toks)] = c
                while len(prefix_store) > 8:
                    prefix_store.popitem(last=False)
            else:
                turn_store[id(e)] = (toks, c)
            stats["saved"] += 1
            if not want:
                del marks[uid]

    def next_generated():
        if not saving:
            return gen.next_generated()             # the old path, unchanged
        while True:                                 # same loop as BatchGenerator.next_generated, plus state saves
            prs, rs = gen.next()
            save_states(prs)
            if not rs and prs:
                continue
            return rs

    def take_new() -> None:
        nonlocal n_total
        if handle is not None:
            new = handle.take()
            queue.extend(new)
            n_total += len(new)

    def busy() -> int:
        """Episodes holding a place in the batch: generating or in tools, minus those blocked waiting (P5b)."""
        return len(active) + len(pending) - (handle.waiting if handle is not None else 0)

    def refill() -> None:
        # continuing episodes first (they already hold a cache), then new ones
        take_new()
        for e in list(ready):
            ids_len = len(e.seq) + 200 if e.seq else 2000
            if fits(ids_len):
                ready.remove(e)
                submit(e)
        while queue and not ready and busy() < batch_size and fits(2000):
            e = queue.pop(0)
            e.t_start = time.perf_counter()
            submit(e)

    try:
        refill()
        n_steps = 0
        while active or queue or pending or ready or (handle is not None and handle.has_new()):
            # tools that finished: back into the batch (or record the episode)
            for f in [f for f in pending if f.done()]:
                e = pending.pop(f)
                f.result()                                  # re-raise tool errors here
                if e.done:
                    finish(e)
                else:
                    submit(e)
            refill()
            if not active:
                if pending:
                    wait(list(pending), timeout=0.5, return_when=FIRST_COMPLETED)
                continue
            rs = next_generated()
            n_steps += 1
            if n_steps % 256 == 0:
                _flush_lazy_cache_state(gen)
            for r in rs or []:
                if r.uid not in active:
                    continue
                e, toks = active[r.uid]
                if r.token not in stops:
                    toks.append(r.token)
                    alen[r.uid] = alen.get(r.uid, 0) + 1
                    if on_token is not None:
                        d = detoks.get(r.uid)
                        if d is None:
                            d = detoks[r.uid] = tok.detokenizer
                        d.add_token(r.token)
                        stream(e, d.last_segment)
                if r.finish_reason is None:
                    continue
                if on_token is not None and r.uid in detoks:
                    d = detoks.pop(r.uid)
                    d.finalize()
                    stream(e, d.last_segment)
                del active[r.uid]
                alen.pop(r.uid, None)
                e.output_tokens += len(toks)
                e.cache, e.seq = r.prompt_cache, list(r.all_tokens)
                pending[pool.submit(apply_fn, e, tok.decode(toks), r.finish_reason)] = e
            now = time.perf_counter()
            if progress_every and now - last > progress_every:
                print(f"  [{now - t0:6.0f}s] {n_done}/{n_total} done, {len(active)} generating, "
                      f"{len(pending)} in tools, {len(ready)} waiting, slots {len(active)}x{max(alen.values(), default=0)}, cache reused {stats['reused']} / "
                      f"fresh {stats['full']}, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
                if label_fn:      # per-episode status: where the time goes
                    gen_s = ", ".join(f"{label_fn(e)} step {e.steps} ctx {alen.get(u, 0)} +{len(t)}"
                                      for u, (e, t) in active.items())
                    print(f"           generating: {gen_s}", flush=True)
                    if pending or ready:
                        print(f"           in tools: {', '.join(label_fn(e) for e in pending.values())}; "
                              f"waiting for memory: {', '.join(label_fn(e) for e in ready)}", flush=True)
                last = now
    finally:
        if hooked is not None:
            hooked.__exit__(None, None, None)
        if handle is not None:
            handle.close()
        gen.close()
        pool.shutdown(wait=True)
    stats["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    print(f"  done: {n_done}/{n_total} in {stats['seconds']:.0f}s; cache reused {stats['reused']}, "
          f"fresh {stats['full']}; waited for context budget {stats['waited_for_budget']}x; "
          f"peak {stats['peak_gb']:.1f} GB", flush=True)
    return stats

class MLXBackend:
    """keel.core.backend.Backend for an mlx-lm model already in memory."""

    def __init__(self, model, tok, sampling: SamplingSpec):
        self.model, self.tok, self.sampling = model, tok, sampling

    def capabilities(self) -> Capabilities:
        return CAPABILITIES

    def price(self) -> Price:
        return Price()

    def run(self, episodes, hooks: RunHooks, limits: RunLimits) -> dict:
        x = limits.extra
        return run_scheduler(self.model, self.tok, self.sampling, list(episodes), prompt_fn=hooks.prompt_fn,
                             apply_fn=hooks.apply_fn, finish_fn=hooks.finish_fn, max_context=limits.max_context,
                             batch_size=limits.batch_size, prefill_step=x.get("prefill_step", 1024),
                             prefill_batch=x.get("prefill_batch", 4), context_budget=limits.context_budget,
                             reuse_cache=x.get("reuse_cache", True), progress_every=limits.progress_every,
                             state_saves=x.get("state_saves", False), shared_prefix=x.get("shared_prefix", False),
                             label_fn=hooks.label_fn, seed_fn=hooks.seed_fn, on_token=hooks.token_fn,
                             handle=hooks.handle, gate=x.get("gate"))

    def complete(self, prompt, *, forbid_tools: bool = False):
        """One reply for one prompt (token ids), through the same scheduler as a one-episode run."""

        class _One:
            done = False; stop = ""; prompt_tokens = processed_tokens = output_tokens = 0
            t_start = seconds = 0.0; cache = seq = None; ban_tool_call = forbid_tools; steps = 0
        e, got = _One(), {}

        def apply(ep, text, fr):
            got["text"], got["fr"] = text, fr
            ep.done = True

        t0 = time.perf_counter()
        run_scheduler(self.model, self.tok, self.sampling, [e], prompt_fn=lambda ep: prompt, apply_fn=apply,
                      finish_fn=lambda ep: None, batch_size=1, progress_every=0)
        st = RequestStats(prompt_tokens=e.prompt_tokens, reused_tokens=e.prompt_tokens - e.processed_tokens,
                          output_tokens=e.output_tokens, generate_seconds=time.perf_counter() - t0)
        return got.get("text", ""), got.get("fr", ""), st
