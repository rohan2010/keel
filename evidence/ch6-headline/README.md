# Chapter 6 headline: evidence pack

Everything needed to re-check keel's pre-registered Chapter 6 result without trusting the author's machine.

**The claim.** On 24 file-chore tasks x 4 unseen test seeds (96 episodes per arm), Qwen3.5-4B (8-bit) with keel's
`full` stack passed **65/96 (67.7%)** against **51/96 (53.1%)** for the same model with one shell tool: **+14.6
points, task-clustered one-sided p = 0.0128** (H1, alpha 0.05, frozen before any test-seed run). False "done" claims
did not go down (H2, p = 0.62), so by the fixed order H3 (vs goose) and H4 (vs opencode) were **not tested**.

## Check it

From a keel checkout with its environment installed (`uv sync`):

```
uv run python evidence/ch6-headline/verify.py
```

It checks every file against `SHA256SUMS`, then re-runs `keel-core compare --prereg` for H1 -> H4 twice: once with
the archived code that produced the runs ("as run") and once with the keel you have installed ("current"). Each
pass starts from the frozen pre-registrations only, so the fixed-sequence gates are decided again. It takes well under a
minute and needs no model. Expected result:

```
SHA256SUMS: all files match

[as run]
  h1: REAL GAIN              deciding +0.146 p = 0.01276   MATCH
  h2: NO DETECTABLE EFFECT   deciding +0.013 p = 0.6191   MATCH
  h3: NO DETECTABLE EFFECT   deciding +0.115 p = 0.0448   MATCH
  h4: NO DETECTABLE EFFECT   deciding +0.125 p = 0.09111   MATCH

[current]
  h1, h2: the same lines
  h3: NOT TESTED             deciding +0.115 p = 0.0448   MATCH
  h4: NOT TESTED             deciding +0.125 p = 0.09111   MATCH

ALL CHECKS PASSED
```

"deciding" is the pre-registered metric: pass rate for H1, H3, H4 (full minus the other arm) and false success for H2
(full minus bare; H2 needed it to go down). H3 and H4 print their exploratory numbers, but their verdict is "not
tested": H2 was not confirmed, and the fixed order stops there. (H3's p = 0.0448 would have passed alone; it does not
count; keel since Ch7 P7 prints that verdict as "not tested", the 2026-11-20 reports said "no detectable
effect" with the same reason, and the verifier accepts only that rename.) The p values here are exact permutation p's and differ in the third digit from the write-up's rounded 0.62.

One compare by hand, from the repo root:

```
uv run keel-core compare evidence/ch6-headline/runs/20261119-221250_ch6_bare_qwen3.5-4b-8bit \
  evidence/ch6-headline/runs/20261119-233312_ch6_full_qwen3.5-4b-8bit --prereg ch6-h1-full-vs-bare --no-report
```

(This uses your own `experiments/prereg/`. For the pack's frozen copy, set `KEEL_EXPERIMENTS` to a folder holding
`prereg/ch6-h1-full-vs-bare.json` from `preregs/`.)

## What is here

| Path | What |
|---|---|
| `preregs/*.yaml`, `preregs/*.json` | The four pre-registrations as written, and their frozen records (text, sha256, freeze time 2026-11-19 22:12:42; every counted run started after it) |
| `recorded/*.result.json` | The deciding p of each hypothesis as recorded at the time (a pre-registered result is written once) |
| `runs/<run>/` | The five counted runs: `config.json` (settings, items, code hash, environment), `outputs.jsonl` (one scored line per episode), `metrics.json`, `timings.json`, `eval.json` (arm name). Sandboxes and transcripts are left out (about 1.2 GB) |
| `reports/h1..h4.{txt,json}` | The compare reports produced on 2026-11-20 |
| `code/keel-1b1049713d33.tar.gz` | The exact code of the runs: `src/`, `scripts/`, `pyproject.toml`, `uv.lock`, task cards and the task test. Its `.py` files reproduce the combined code hash `1b1049713d33` in every run's `config.json` (rule: `keel.runlog.code_hashes`) |
| `tasks.sha256` | The task-freeze hashes (generators, scorers, task cards), written before any arm ran |
| `headline.sh` | The commands that produced the five runs (one run each, 2026-11-19/20) |
| `originals.json` | sha256 of each run file before redaction, and the redaction rule |

**Redaction.** The goose and opencode runs captured the machine's user name (paths under `$HOME`, and a sign-off the
model wrote in a few email drafts). In this pack it is replaced by `user` / `User`; scores are stored per episode and
are unaffected (the verifier shows every verdict and number unchanged). keel's own arms needed no redaction.

## Setup in one paragraph

Qwen3.5-4B 8-bit (MLX), one try per item, temperature 0.7, top-p 0.8, top-k 20, 32k context, 8,192 tokens per reply,
thinking off, on a 16 GB M4 Mac. keel arms ran in-process (batch 4); goose 1.60.1 and opencode 1.19.6 ran against
mlx_lm.server with the same weights and sampling (2 at a time, 30 turns, 15 min cap), sandboxed, and were scored by
the same scorers. Deciding test: task-clustered sign-flip permutation (exact, one-sided), item-level exact McNemar
beside it. Full write-up: `doc/product/chapter-6-writeup.md`.

## Limits

Own tasks only (an outside benchmark is still to do), one model, one machine. Sampling at T 0.7 with one try per
item: a re-run of the model would give different episodes; this pack lets you re-check the scoring and the
statistics of the runs that were counted, not regenerate them.
