#!/bin/bash
# One-shot orchestration: moe_test (4 configs in parallel) -> when done ->
# vision_test (4 configs in parallel). Runs inside a single container tmux
# session so it survives disconnect. prune=0, drop 1024/4096, n_evals=500,
# candidates=2000, Fengshui pool folded into each summary.json.
set -u
cd "$(dirname "$0")"
DB="${FENGSHUI_DB:-$(cd "$(dirname "$0")/.." && pwd)/unified_database.csv}"
MOZ="$(cd "$(dirname "$0")/.." && pwd)/archgym_results"
ARGS="--algorithm chain --n-start 1 --saeo-end 6 --n-end 8 --n-evals 500 \
--saeo-candidates 2000 --prune-pct 0 --drop-seqs 1024 4096 --seed 42 \
--database $DB --fengshui-dir $MOZ"

run_split () {   # $1 = split-mode, $2 = output prefix
  for c in energy_nocost energy_cost edp_nocost edp_cost; do
    python3 -u run_train_test.py --configs "$c" --split-mode "$1" $ARGS \
      --output-dir "results/$2_$c" > "$2_$c.log" 2>&1 &
  done
  wait
}

echo "MOE_START $(date)"
run_split moe_test moe2
echo "MOE_DONE $(date)"

echo "VISION_START $(date)"
run_split vision_test vision2
echo "VISION_DONE $(date)"

touch "$(dirname "$0")/MOE_VISION_ALL_DONE"
echo "ALL_DONE $(date)"
