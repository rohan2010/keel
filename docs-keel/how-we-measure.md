# How we measure

A change to keel counts as better only if it beats the noise on tasks we did not tune on, with the test decided before
the run. This page explains each piece.

## The tasks

24 everyday file tasks (sort photos, find duplicates, fix a CSV, rename across files, refuse to send email, ...). Each
task is a small program that **builds** a fresh messy folder from a seed, a **checker** that looks at the folder (and the
answer) afterwards, and an **oracle** that solves it the intended way. Every task has a card in `task-cards/` saying
what is checked. A task only counts if its oracle scores 100% and doing nothing scores 0%
(`tests/checks/check_agent_tasks.py`).

## Seeds and tiers

The seed decides the folder's contents, so the same task gives new folders on new seeds.

| Tier | Seeds | Used for |
|---|---|---|
| dev | 0 to 999 | building and tuning, looked at freely |
| test (validation) | 1,000 to 4,999 | one pre-registered comparison per change; every look is logged |
| holdout | 5,000 and up | once, at the very end, with everything frozen |

## The scores

- **pass@1**: the share of tries the checker accepts.
- **false "done"**: the share of tries that failed but claimed success. It matters as much as the pass rate: some
  changes raise both.
- Time and tokens per try, reported beside them.

## Comparing two runs

`keel-core compare A B` first checks the runs are comparable (same model, settings, items; every difference listed),
then runs a **paired test**: the same items in both arms, so each item is compared with itself. With several tasks the
deciding test is **task-clustered** (an exact sign-flip permutation over tasks), so one lucky task can't carry the
result. The verdict is one of: real gain, real loss, no detectable effect, can't compare, or not tested. A "no
detectable effect" also says the smallest effect the sample could have seen.

## Pre-registration

Before a validation run, the plan is frozen: which runs, which metric, one-sided or not, alpha, and the test
(`keel-core prereg plans/<name>.yaml`). The comparison is only labelled pre-registered if the runs started after the
freeze and match the plan. Several hypotheses can be chained in a fixed order: a later one is tested only if the one
before it was confirmed (that is why two of the Chapter 6 comparisons say "not tested").

## Same results after every change

Before and after a code change, a fixed set of 13 tasks runs with greedy decoding (`scripts/golden.sh`) and every
transcript must be identical. That's how keel knows a refactor didn't quietly change what the model sees.

## Re-checking the headline yourself

`uv run python evidence/ch6-headline/verify.py` re-runs every comparison of the headline result from the files alone:
the frozen plans, the scored runs and the exact code. See `evidence/ch6-headline/README.md`.
