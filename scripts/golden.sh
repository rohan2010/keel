#!/bin/bash
# Same-results gate (Chapter 1, A0 / A7): a fixed small set at batch 1, greedy, fixed seeds.
# Usage: scripts/golden.sh <tag> [extra flags]   e.g. golden.sh a0_run1 ; golden.sh b3 --state-saves --shared-prefix
# Compare runs with scripts/golden_compare.py (add --parity for B2/B3).
# Agent: the 13 tasks x 1 dev item, file tools + loop breaker + trim. BFCL multi-turn: first 3 dev entries per
# category, reminder + loop breaker + trim. Exit code = first failing step.
set -u
tag="${1:?tag}"
shift
cd "$(dirname "$0")/.."
uv run keel-core run --name "golden_${tag}_agent" --split dev --per-task 1 --batch-size 1 --prefill-batch 1 \
  --temperature 0 --seed 0 --toolset files --loop-break --trim-outputs "$@" || exit 1
uv run keel-core bfcl-mt --name "golden_${tag}_bfcl" --split dev --limit 3 --batch-size 1 --prefill-batch 1 \
  --temperature 0 --seed 0 --reminder --loop-break --trim-outputs "$@" || exit 2
echo "golden ${tag} done"
