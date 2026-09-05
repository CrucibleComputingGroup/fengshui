#!/usr/bin/env bash
# Re-run the five Figure-11 held-out splits on the corrected canonical DB.
#
# Scheduling is deliberately one split at a time and four objective configs in
# parallel. Each config already forks up to 64 outer workers, so launching all
# 20 combinations together severely oversubscribes a 256-core AE host.
set -Eeuo pipefail

MODE="full"
usage() {
  cat <<'EOF'
Usage: run_all_v2.sh [--validate-only | --smoke]

  --validate-only  Verify script syntax, AE pools, and corrected DB hash; run no jobs.
  --smoke          Run moe_test/energy_nocost only with the driver's n=1..2,
                   8-evaluation smoke shortcut; do not generate the 20-bar plot.
  (no option)      Run the full 5-split x 4-config corrected _v2 experiment.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --validate-only) MODE="validate" ;;
    --smoke) MODE="smoke" ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; echo "ERROR: unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

readonly MODE
readonly THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly AE_SRC="$(cd "$THIS_DIR/.." && pwd)"
readonly FENGSHUI_ROOT="$(cd "$AE_SRC/../.." && pwd)"
readonly DRIVER="$THIS_DIR/run_train_test.py"
readonly PLOTTER="$THIS_DIR/plot_v2_cross.py"
readonly EXPECTED_DB_SHA256="a1f5849d22a9e73251dfdf9b5565e918018aee8108a44b5894d3b6f2c12c4cc0"

DB_PATH="$(realpath -m "${FENGSHUI_DB:-$AE_SRC/unified_database.csv}")"
POOLS_ROOT="$(realpath -m "${FENGSHUI_ARCHGYM:-$AE_SRC/archgym_results}")"
RUNS_ROOT="$(realpath -m "${FENGSHUI_V2_OUTPUT_ROOT:-$THIS_DIR/corrected_v2_runs}")"
RUN_ID="${FENGSHUI_V2_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${FENGSHUI_PYTHON:-python3}"

readonly DB_PATH POOLS_ROOT RUNS_ROOT RUN_ID PYTHON_BIN
readonly RUN_ROOT="$RUNS_ROOT/$RUN_ID"
readonly RAW_ROOT="$RUN_ROOT/raw"
readonly RESULTS_ROOT="$RUN_ROOT/results"
readonly LOG_ROOT="$RUN_ROOT/logs"
readonly PLOT_ROOT="$RUN_ROOT/plot"
readonly MANIFEST="$RUN_ROOT/run_manifest.txt"

if [[ "$MODE" == "smoke" ]]; then
  SPLITS=(moe)
  CONFIGS=(energy_nocost)
  MODE_ARGS=(--smoke)
else
  SPLITS=(moe decode vision prefill seq2048)
  CONFIGS=(energy_nocost energy_cost edp_nocost edp_cost)
  MODE_ARGS=()
fi
readonly -a SPLITS CONFIGS MODE_ARGS
readonly -a COMMON_ARGS=(
  --algorithm chain
  --n-start 1
  --saeo-end 6
  --n-end 8
  --n-evals 500
  --saeo-candidates 2000
  --saeo-mutation-rate 0.3
  --prune-pct 0
  --drop-seqs 1024 4096
  --batches 1 8
  --seed 42
  --pim
  --switch
  --dag
)

ACTIVE_PIDS=()

die() {
  echo "ERROR: $*" >&2
  exit 1
}

terminate_active() {
  local pid
  for pid in "${ACTIVE_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      # Jobs are launched with setsid, so terminate the entire fork-worker group.
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
}

on_signal() {
  echo "Interrupted; terminating active train/test worker groups." >&2
  terminate_active
  exit 130
}
trap on_signal INT TERM HUP

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python not found: $PYTHON_BIN"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
command -v setsid >/dev/null 2>&1 || die "setsid is required for safe worker cleanup"
[[ -f "$DRIVER" ]] || die "missing driver: $DRIVER"
[[ -f "$PLOTTER" ]] || die "missing plotter: $PLOTTER"
[[ -r "$DB_PATH" ]] || die "canonical database is not readable: $DB_PATH"
[[ -d "$POOLS_ROOT" ]] || die "Fengshui pool root is missing: $POOLS_ROOT"

for pool_dir in ae_energy_chain ae_energy_cost_chain ae_edp_chain ae_edp_cost_chain; do
  [[ -d "$POOLS_ROOT/$pool_dir" ]] || die "missing AE pool directory: $POOLS_ROOT/$pool_dir"
  pool_csv="$(find "$POOLS_ROOT/$pool_dir" -maxdepth 1 -type f \
    -name 'saeo_isaeo_chain_*.csv' -print -quit)"
  [[ -n "$pool_csv" ]] || die "no chain CSV in AE pool directory: $POOLS_ROOT/$pool_dir"
  if grep -qi 'feather' "$pool_csv"; then
    die "forbidden FEATHER architecture found in AE pool: $pool_csv"
  fi
done

echo "Verifying corrected database SHA-256 (4.9 GB; this is done once)..."
ACTUAL_DB_SHA256="$(sha256sum "$DB_PATH" | awk '{print $1}')"
[[ "$ACTUAL_DB_SHA256" == "$EXPECTED_DB_SHA256" ]] ||
  die "database hash mismatch: got $ACTUAL_DB_SHA256, expected $EXPECTED_DB_SHA256"
readonly ACTUAL_DB_SHA256

if [[ "$MODE" == "validate" ]]; then
  bash -n "${BASH_SOURCE[0]}"
  "$PYTHON_BIN" -c \
    'import pathlib, sys; compile(pathlib.Path(sys.argv[1]).read_text(), sys.argv[1], "exec")' \
    "$DRIVER"
  echo "Validation passed: runner syntax, Python syntax, four AE pools, and corrected DB hash."
  if pgrep -f '[r]un_train_test.py' >/dev/null 2>&1; then
    echo "WARNING: run_train_test.py jobs are active; a full/smoke launch will refuse to overlap." >&2
  fi
  exit 0
fi

# Refuse to mix this exact run with any existing train/test optimizer. Override
# only when the caller has independently verified CPU and memory headroom.
if [[ "${FENGSHUI_V2_ALLOW_CONCURRENT:-0}" != "1" ]] &&
   pgrep -f '[r]un_train_test.py' >/dev/null 2>&1; then
  die "run_train_test.py is already active; stop it or explicitly set FENGSHUI_V2_ALLOW_CONCURRENT=1"
fi

mkdir -p "$RUNS_ROOT"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$RUNS_ROOT/.run_all_v2.lock"
  flock -n 9 || die "another run_all_v2.sh owns $RUNS_ROOT/.run_all_v2.lock"
fi
[[ ! -e "$RUN_ROOT" ]] || die "run directory already exists: $RUN_ROOT"
mkdir -p "$RAW_ROOT" "$RESULTS_ROOT" "$LOG_ROOT" "$PLOT_ROOT"

# Pin every environment-controlled search/evaluation knob used by this path.
export FENGSHUI_DB_SHA256="$ACTUAL_DB_SHA256"
export FENGSHUI_DETERMINISTIC=1
export FENGSHUI_GA_SEED=0
export FENGSHUI_EXCLUDE_PREFIXES=efficientnet,gpt,resnet
export INNER_GA_POP=30
export INNER_GA_GEN=30
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

{
  echo "run_id=$RUN_ID"
  echo "mode=$MODE"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "driver=$DRIVER"
  echo "python=$PYTHON_BIN"
  echo "database=$DB_PATH"
  echo "database_sha256=$ACTUAL_DB_SHA256"
  echo "fengshui_pools=$POOLS_ROOT"
  echo "raw_root=$RAW_ROOT"
  echo "results_root=$RESULTS_ROOT"
  echo "scheduling=5 splits sequential; 4 configs parallel per split"
  echo "architecture_targets=eyeriss_like,simba_like,gemmini_like,PIM,switch_8port"
  echo "forbidden_architecture_targets=feather"
  printf 'common_args='
  printf '%q ' "${COMMON_ARGS[@]}"
  printf '\n'
  echo "FENGSHUI_DETERMINISTIC=$FENGSHUI_DETERMINISTIC"
  echo "FENGSHUI_GA_SEED=$FENGSHUI_GA_SEED"
  echo "FENGSHUI_EXCLUDE_PREFIXES=$FENGSHUI_EXCLUDE_PREFIXES"
  echo "INNER_GA_POP=$INNER_GA_POP"
  echo "INNER_GA_GEN=$INNER_GA_GEN"
  echo "PYTHONHASHSEED=$PYTHONHASHSEED"
} > "$MANIFEST"

for split in "${SPLITS[@]}"; do
  split_mode="${split}_test"
  final_split="$RESULTS_ROOT/${split}_test_v2"
  mkdir -p "$final_split"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting $split_mode"
  echo "split_start=$split_mode $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"

  ACTIVE_PIDS=()
  job_labels=()
  for cfg in "${CONFIGS[@]}"; do
    raw_cfg="$RAW_ROOT/${split_mode}__${cfg}"
    log_file="$LOG_ROOT/${split_mode}__${cfg}.log"
    mkdir -p "$raw_cfg"
    setsid "$PYTHON_BIN" -u "$DRIVER" \
      --split-mode "$split_mode" \
      --configs "$cfg" \
      --database "$DB_PATH" \
      --fengshui-dir "$POOLS_ROOT" \
      --output-dir "$raw_cfg" \
      "${COMMON_ARGS[@]}" \
      "${MODE_ARGS[@]}" >"$log_file" 2>&1 &
    ACTIVE_PIDS+=("$!")
    job_labels+=("$cfg")
    echo "  launched $cfg pid=$! log=$log_file"
  done

  split_failed=0
  for i in "${!ACTIVE_PIDS[@]}"; do
    if wait "${ACTIVE_PIDS[$i]}"; then
      echo "  completed ${job_labels[$i]}"
    else
      echo "  FAILED ${job_labels[$i]} (see $LOG_ROOT/${split_mode}__${job_labels[$i]}.log)" >&2
      split_failed=1
    fi
  done
  ACTIVE_PIDS=()
  [[ "$split_failed" == "0" ]] || die "$split_mode did not complete successfully"

  # Each raw config root is freshly created and must contain exactly one driver
  # timestamp. Preserve the complete config directory, split, and per-config
  # summary_all rather than copying only summary.json.
  reference_split=""
  for cfg in "${CONFIGS[@]}"; do
    raw_cfg="$RAW_ROOT/${split_mode}__${cfg}"
    mapfile -t timestamp_dirs < <(
      find "$raw_cfg" -mindepth 1 -maxdepth 1 -type d -print | sort
    )
    [[ "${#timestamp_dirs[@]}" == "1" ]] ||
      die "expected one timestamp directory in $raw_cfg; found ${#timestamp_dirs[@]}"
    run_dir="${timestamp_dirs[0]}"

    required=(
      "$run_dir/split.json"
      "$run_dir/summary_all.json"
      "$run_dir/$cfg/summary.json"
      "$run_dir/$cfg/cross_eval.csv"
      "$run_dir/$cfg/train_sweep_phase1_saeo.csv"
      "$run_dir/$cfg/train_sweep_phase2_isaeo.csv"
      "$run_dir/$cfg/test_sweep_phase1_saeo.csv"
      "$run_dir/$cfg/test_sweep_phase2_isaeo.csv"
    )
    for required_file in "${required[@]}"; do
      [[ -s "$required_file" ]] || die "missing/empty required output: $required_file"
    done

    if [[ -z "$reference_split" ]]; then
      cp "$run_dir/split.json" "$final_split/split.json"
      reference_split="$run_dir/split.json"
    else
      cmp -s "$reference_split" "$run_dir/split.json" ||
        die "split.json differs across configs for $split_mode"
    fi
    cp -a "$run_dir/$cfg" "$final_split/$cfg"
    cp "$run_dir/summary_all.json" "$final_split/summary_all.${cfg}.json"
    echo "source=$split_mode/$cfg $run_dir" >> "$MANIFEST"
  done

  # Rebuild the conventional summary_all.json that parallel one-config jobs
  # cannot produce themselves. Keep every original per-config summary alongside it.
  "$PYTHON_BIN" - "$final_split" "${CONFIGS[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
configs = sys.argv[2:]
docs = {cfg: json.loads((root / f"summary_all.{cfg}.json").read_text())
        for cfg in configs}
combined = dict(docs[configs[0]])
combined["source_timestamps"] = {cfg: docs[cfg].get("timestamp") for cfg in configs}
combined["commands"] = {cfg: docs[cfg].get("command") for cfg in configs}
combined.pop("command", None)
combined["source_args"] = {cfg: docs[cfg].get("args") for cfg in configs}
combined["args"] = dict(combined.get("args", {}))
combined["args"]["configs"] = configs
combined["args"]["output_dir"] = str(root)
combined["configs"] = {}
for cfg in configs:
    combined["configs"].update(docs[cfg].get("configs", {}))
(root / "summary_all.json").write_text(json.dumps(combined, indent=2) + "\n")
PY

  echo "split_done=$split_mode $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] consolidated $final_split"
done

# The plotter hardcodes a sibling results/ directory. Run an unchanged copy in
# an isolated directory whose results symlink targets this run, so neither the
# shipped summaries nor shipped figures are overwritten.
if [[ "$MODE" == "smoke" ]]; then
  echo "plot=skipped_smoke_mode" >> "$MANIFEST"
elif [[ "${FENGSHUI_V2_SKIP_PLOT:-0}" == "1" ]]; then
  echo "plot=skipped_by_FENGSHUI_V2_SKIP_PLOT" >> "$MANIFEST"
elif "$PYTHON_BIN" -c 'import numpy, matplotlib' >/dev/null 2>&1; then
  cp "$PLOTTER" "$PLOT_ROOT/plot_v2_cross.py"
  ln -s "$RESULTS_ROOT" "$PLOT_ROOT/results"
  (
    cd "$PLOT_ROOT"
    "$PYTHON_BIN" plot_v2_cross.py
  ) >"$LOG_ROOT/plot_v2_cross.log" 2>&1
  echo "plot=$PLOT_ROOT" >> "$MANIFEST"
  echo "Generated isolated Figure-11 plot under $PLOT_ROOT"
else
  echo "Plot skipped: numpy/matplotlib unavailable to $PYTHON_BIN" >&2
  echo "plot=skipped_missing_numpy_or_matplotlib" >> "$MANIFEST"
fi

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
(
  cd "$RUN_ROOT"
  find results plot -type f -print0 | sort -z | xargs -0 -r sha256sum
) > "$RUN_ROOT/SHA256SUMS"

echo "All corrected _v2 runs completed."
echo "Results : $RESULTS_ROOT"
echo "Logs    : $LOG_ROOT"
echo "Manifest: $MANIFEST"
