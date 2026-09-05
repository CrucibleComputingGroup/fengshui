#!/bin/bash
# One-shot: prefill_test (4 configs parallel) -> seq2048_test (4 configs parallel).
# Absolute --output-dir (run_train_test does os.chdir(scripts/) at import, so a
# relative path would land under scripts/). Runs in a persistent container tmux.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
DB="${FENGSHUI_DB:-$(cd "$ROOT/.." && pwd)/unified_database.csv}"
MOZ="$ROOT/fengshui_pools"
ARGS="--algorithm chain --n-start 1 --saeo-end 6 --n-end 8 --n-evals 500 \
--saeo-candidates 2000 --prune-pct 0 --drop-seqs 1024 4096 --seed 42 \
--database $DB --fengshui-dir $MOZ"

run_split () {   # $1 = split-mode, $2 = output prefix
  for c in energy_nocost energy_cost edp_nocost edp_cost; do
    python3 -u run_train_test.py --configs "$c" --split-mode "$1" $ARGS \
      --output-dir "$ROOT/results/$2_$c" > "$ROOT/$2_$c.log" 2>&1 &
  done
  wait
}

echo "PREFILL_START $(date)"
run_split prefill_test prefill2
echo "PREFILL_DONE $(date)"

echo "SEQ2048_START $(date)"
run_split seq2048_test seq20482
echo "SEQ2048_DONE $(date)"

touch "$ROOT/PREFILL_SEQ_ALL_DONE"
echo "ALL_DONE $(date)"
