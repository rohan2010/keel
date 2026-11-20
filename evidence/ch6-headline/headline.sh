#!/bin/bash
# Chapter 6 headline, COMBINED design (DRAFT, not run): pre-registrations plans/ch6-h{1,2,3,4}-*.yaml (fixed sequence),
# frozen before this starts. Test split, fresh seeds 24-27, 25 tasks (24 counted + F3 safety) x 4 = 100 per arm.
# The 12 new tasks are built in the keel-26 session and frozen (hashed) before this runs. Baseline arms need:
#   uv run mlx_lm.server --model models/Qwen3.5-4B-8bit --port 8080 \
#     --chat-template-args '{"enable_thinking":false}' --temp 0.7 --top-p 0.8 --top-k 20 --max-tokens 8192
set -u
cd "$(dirname "$0")/../../../.."
P="Ch6 headline, combined design, pre-registered (plans/ch6-h1..h4), run once"
L="--batch-size 4 --max-context 32000 --max-tokens 8192 --temperature 0.7"
T="A1 A3 A4 A5 A6 A7 B1 B2 C1 C3 C4 C5 D1 D2 D3 D4 E1 E3 E4 E5 F1 F2 F3 F4 F5"
A="--split test --tasks $T --per-task 4 --first-seed 24 $L"
F="--done-gate --gnu-tools --examples --toolset files_py --no-send-note --search-hint"
echo "start $(date)"
uv run keel-core eval --suite agent --arm ch6_bare $A --toolset shell --no-loop-break --no-trim-outputs --purpose "$P" || echo "FAILED bare"
uv run keel-core eval --suite agent --arm ch6_keel $A --purpose "$P" || echo "FAILED keel"
uv run keel-core eval --suite agent --arm ch6_full $A $F --purpose "$P" || echo "FAILED full"
# baselines: the model server only now (two copies of the model would not fit next to the in-process arms)
uv run mlx_lm.server --model models/Qwen3.5-4B-8bit --port 8080 --chat-template-args '{"enable_thinking":false}' --decode-concurrency 2 --prompt-concurrency 1 --prompt-cache-bytes 1610612736 \
  --temp 0.7 --top-p 0.8 --top-k 20 --max-tokens 8192 > doc/product/spikes/ch6/headline_server.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do curl -s http://127.0.0.1:8080/v1/models >/dev/null 2>&1 && break; sleep 2; done
uv run keel-core baseline --agent goose --name ch6_goose --split test --tasks $T --per-task 4 --first-seed 24 --concurrency 2 --max-turns 30 --timeout 900 --purpose "$P" || echo "FAILED goose"
uv run keel-core baseline --agent opencode --name ch6_opencode --split test --tasks $T --per-task 4 --first-seed 24 --concurrency 2 --max-turns 30 --timeout 900 --purpose "$P" || echo "FAILED opencode"
kill $SRV
echo "done $(date)"
