"""Checks for per-episode random numbers (no model: fake logits).

  uv run python tests/checks/check_episode_rng.py
"""

import sys

import mlx.core as mx
import numpy as np

from keel.sampling import EpisodeSampler, _sample, episode_base, greedy
from keel.specs import SamplingSpec

fails = 0


def check(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {detail}"))
    fails += not cond


def main():
    spec = SamplingSpec(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=0.0, max_tokens=16, seed=0)
    V = 1000
    logits = mx.array(np.random.default_rng(1).normal(0, 2, (1, V)).astype(np.float32))
    ctx = list(range(57))

    a, b = EpisodeSampler(spec, episode_base(0, "C1_3")), EpisodeSampler(spec, episode_base(0, "C1_3"))
    pick = lambda s, c, lg=logits: greedy(s(c, lg)).item()
    check("same episode, same position -> same token", pick(a, ctx) == pick(b, ctx))
    mx.random.seed(123); x = pick(a, ctx); mx.random.categorical(mx.zeros((5,))); mx.random.seed(999)
    check("global random state does not matter", x == pick(a, ctx))
    seq_a = [pick(a, list(range(n))) for n in range(100, 160)]
    seq_c = [pick(EpisodeSampler(spec, episode_base(0, "C1_4")), list(range(n))) for n in range(100, 160)]
    check("different episodes -> different draws", seq_a != seq_c)
    seq_s = [pick(EpisodeSampler(spec, episode_base(1, "C1_3")), list(range(n))) for n in range(100, 160)]
    check("different run seed -> different draws", seq_a != seq_s)
    check("positions vary the draw", len(set(seq_a)) >= 2, set(seq_a))

    out = a(ctx, logits)
    check("output: one finite entry", int(mx.isfinite(out).sum().item()) == 1)
    lp = out - mx.logsumexp(out, axis=-1, keepdims=True)
    check("greedy batch sampler picks it", greedy(lp).item() == pick(a, ctx))
    check("bf16 logits ok", greedy(a(ctx, logits.astype(mx.bfloat16))).item() >= 0)

    # same distribution as the old global sampler (top-k / top-p respected; frequencies close)
    n = 4000
    new = np.array([pick(a, list(range(i))) for i in range(n)])
    mx.random.seed(0)
    old = np.array([_sample(spec, logits).item() for _ in range(n)])
    top20 = set(np.argsort(-np.array(logits[0]))[:20].tolist())
    check("only top-k tokens drawn", set(new.tolist()) <= top20)
    fn, fo = np.bincount(new, minlength=V) / n, np.bincount(old, minlength=V) / n
    check("frequencies match old sampler (TV < 0.05)", 0.5 * np.abs(fn - fo).sum() < 0.05, 0.5 * np.abs(fn - fo).sum())

    g = EpisodeSampler(SamplingSpec(temperature=0.0, top_p=1.0, top_k=0, presence_penalty=0.0, max_tokens=16, seed=0), 5)
    check("temperature 0 -> argmax", pick(g, ctx) == int(mx.argmax(logits).item()))

    print("ALL CHECKS PASS" if not fails else f"{fails} CHECK(S) FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
