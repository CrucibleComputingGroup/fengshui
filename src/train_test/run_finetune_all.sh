#!/usr/bin/env bash
# Re-run strict-prefix n=8 -> n=9 fine-tuning from a completed train/test run.
# Five splits run sequentially; the four objective configurations within each
# split run in parallel.
set -Eeuo pipefail

readonly THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DRIVER="$THIS_DIR/run_finetune.py"
readonly EXPECTED_DB_SHA256="a1f5849d22a9e73251dfdf9b5565e918018aee8108a44b5894d3b6f2c12c4cc0"

INPUT_ROOT="$(realpath -m "${FENGSHUI_FINETUNE_INPUT_ROOT:?set FENGSHUI_FINETUNE_INPUT_ROOT to the completed results directory}")"
DB_PATH="$(realpath -m "${FENGSHUI_DB:?set FENGSHUI_DB to unified_database.csv}")"
RUNS_ROOT="$(realpath -m "${FENGSHUI_FINETUNE_RUNS_ROOT:-$THIS_DIR/finetune_runs}")"
RUN_ID="${FENGSHUI_FINETUNE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${FENGSHUI_PYTHON:-python3}"

readonly INPUT_ROOT DB_PATH RUNS_ROOT RUN_ID PYTHON_BIN
readonly RUN_ROOT="$RUNS_ROOT/$RUN_ID"
readonly RAW_ROOT="$RUN_ROOT/raw"
readonly LOG_ROOT="$RUN_ROOT/logs"
readonly MANIFEST="$RUN_ROOT/run_manifest.txt"
readonly -a SPLITS=(moe_test_v2 decode_test_v2 vision_test_v2 prefill_test_v2 seq2048_test_v2)
readonly -a CONFIGS=(energy_nocost energy_cost edp_nocost edp_cost)

ACTIVE_PIDS=()

die() {
  echo "ERROR: $*" >&2
  exit 1
}

terminate_active() {
  local pid
  for pid in "${ACTIVE_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
}
trap 'terminate_active; exit 130' INT TERM HUP

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python not found: $PYTHON_BIN"
command -v setsid >/dev/null 2>&1 || die "setsid is required"
[[ -f "$DRIVER" ]] || die "missing driver: $DRIVER"
[[ -r "$DB_PATH" ]] || die "database is not readable: $DB_PATH"
[[ -d "$INPUT_ROOT" ]] || die "input result root is missing: $INPUT_ROOT"

for split in "${SPLITS[@]}"; do
  [[ -s "$INPUT_ROOT/$split/split.json" ]] ||
    die "missing split file: $INPUT_ROOT/$split/split.json"
  for cfg in "${CONFIGS[@]}"; do
    csv="$INPUT_ROOT/$split/$cfg/train_sweep_phase2_isaeo.csv"
    [[ -s "$csv" ]] || die "missing train n=8 sweep: $csv"
    if grep -qi 'feather' "$csv"; then
      die "forbidden FEATHER architecture found in base pool input: $csv"
    fi
  done
done

echo "Verifying corrected database SHA-256 (4.9 GB; once)..."
ACTUAL_DB_SHA256="$(sha256sum "$DB_PATH" | awk '{print $1}')"
[[ "$ACTUAL_DB_SHA256" == "$EXPECTED_DB_SHA256" ]] ||
  die "database hash mismatch: got $ACTUAL_DB_SHA256, expected $EXPECTED_DB_SHA256"
readonly ACTUAL_DB_SHA256

if pgrep -f '[r]un_finetune.py' >/dev/null 2>&1; then
  die "run_finetune.py is already active"
fi

mkdir -p "$RUNS_ROOT"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$RUNS_ROOT/.run_finetune_all.lock"
  flock -n 9 || die "another fine-tune run owns $RUNS_ROOT/.run_finetune_all.lock"
fi
[[ ! -e "$RUN_ROOT" ]] || die "run directory already exists: $RUN_ROOT"
mkdir -p "$RAW_ROOT" "$LOG_ROOT"

export FENGSHUI_DETERMINISTIC=1
export FENGSHUI_GA_SEED=0
export INNER_GA_POP=30
export INNER_GA_GEN=30
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

{
  echo "run_id=$RUN_ID"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "input_root=$INPUT_ROOT"
  echo "database=$DB_PATH"
  echo "database_sha256=$ACTUAL_DB_SHA256"
  echo "raw_root=$RAW_ROOT"
  echo "scheduling=5 splits sequential; 4 configs parallel per split"
  echo "architecture_targets=eyeriss_like,simba_like,gemmini_like,PIM,switch_8port"
  echo "forbidden_architecture_targets=feather"
  echo "base_n=8"
  echo "finetune_n=9"
  echo "n_evals=500"
  echo "saeo_candidates=2000"
  echo "seed=42"
  echo "phase2_frac=0.0 (strict ordered-prefix freeze)"
} > "$MANIFEST"

for split in "${SPLITS[@]}"; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting $split"
  echo "split_start=$split $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
  ACTIVE_PIDS=()
  labels=()
  for cfg in "${CONFIGS[@]}"; do
    out="$RAW_ROOT/${split}__${cfg}"
    log="$LOG_ROOT/${split}__${cfg}.log"
    mkdir -p "$out"
    setsid "$PYTHON_BIN" -u "$DRIVER" \
      --input-root "$INPUT_ROOT" \
      --output-root "$out" \
      --database "$DB_PATH" \
      --splits "$split" \
      --configs "$cfg" \
      --base-n 8 \
      --n-evals 500 \
      --saeo-candidates 2000 \
      --saeo-mutation-rate 0.3 \
      --seed 42 >"$log" 2>&1 &
    ACTIVE_PIDS+=("$!")
    labels+=("$cfg")
    echo "  launched $cfg pid=$! log=$log"
  done

  failed=0
  for i in "${!ACTIVE_PIDS[@]}"; do
    if wait "${ACTIVE_PIDS[$i]}"; then
      echo "  completed ${labels[$i]}"
    else
      echo "  FAILED ${labels[$i]} (see $LOG_ROOT/${split}__${labels[$i]}.log)" >&2
      failed=1
    fi
  done
  ACTIVE_PIDS=()
  [[ "$failed" == "0" ]] || die "$split fine-tuning failed"

  for cfg in "${CONFIGS[@]}"; do
    summary="$RAW_ROOT/${split}__${cfg}/$split/$cfg/finetune_summary.json"
    [[ -s "$summary" ]] || die "missing fine-tune summary: $summary"
    echo "source=$split/$cfg $summary" >> "$MANIFEST"
  done
  echo "split_done=$split $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
done

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
(
  cd "$RUN_ROOT"
  find raw -type f -print0 | sort -z | xargs -0 -r sha256sum
) > "$RUN_ROOT/SHA256SUMS"

echo "All strict-prefix no-FEATHER fine-tune runs completed."
echo "Raw results: $RAW_ROOT"
echo "Logs       : $LOG_ROOT"
echo "Manifest   : $MANIFEST"
