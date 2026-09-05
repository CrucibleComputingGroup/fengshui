#!/usr/bin/env bash
# Determine the convergence endpoint N* of the chiplet-pool chain for all four
# objectives, and report how each endpoint was reached.
#
# WHY THIS EXISTS
# ---------------
# Figure 9's "Unconstrained" bar is not a hardcoded N. It is the point at which
# growing the shared pool stops helping: the marginal improvement in the search
# objective stays below --threshold-pct for --patience consecutive steps. That
# endpoint is a *result*, not a parameter, so it has to be visible.
#
# Some objectives reach N* inside the chain that already exists (no new compute);
# others must be extended past it (~1.5-2 h per added chiplet, growing with N).
# Both outcomes are reported the same way here, so a reader can see which is
# which instead of being handed a number. Run with --assess-only first: it
# prints the full marginal series and the verdict for all four objectives in
# seconds, without launching anything heavy.
#
# USAGE
#   bash run_convergence_all.sh --assess-only   # report status only; no compute
#   bash run_convergence_all.sh                 # assess, then extend what needs it
#   bash run_convergence_all.sh --metrics energy_cost,edp_cost
#
# ENVIRONMENT
#   Needs the conda env that has `gym` (the chain search imports run_archgym_chiplet,
#   which subclasses gym.Env). Per the project layout that is `base`, not `mozart`.
#   Must run with cwd = this scripts/ directory: the performance model opens
#   `network_analysis.csv` by relative path.
#
# OUTPUT (under convergence_runs/<run-id>/)
#   run_manifest.txt        inputs, DB hash, flags, env, per-metric outcome
#   logs/<metric>.log       full converge_chain.py output for that objective
#   convergence_summary.csv metric, N*, whether extension was needed, marginals
#   SHA256SUMS              hashes of everything above
# The consolidated chains themselves go to archgym_results/ae_<metric>_chain_converged/,
# which is where generate_paper_fig10.py looks for them first.
set -Eeuo pipefail

readonly THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CHIPLET_TL="$(cd "$THIS_DIR/.." && pwd)"
readonly DRIVER="$THIS_DIR/converge_chain.py"

DB_PATH="$(realpath -m "${FENGSHUI_DB:-$CHIPLET_TL/unified_database.csv}")"
ARCHGYM="$(realpath -m "${FENGSHUI_ARCHGYM:-$CHIPLET_TL/archgym_results}")"
CHAIN_VERSION="${FENGSHUI_CHAIN_VERSION:-ae}"
RUNS_ROOT="$(realpath -m "${FENGSHUI_CONV_OUTPUT_ROOT:-$THIS_DIR/convergence_runs}")"
RUN_ID="${FENGSHUI_CONV_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${FENGSHUI_PYTHON:-python3}"

THRESHOLD_PCT="${FENGSHUI_CONV_THRESHOLD:-1.0}"
PATIENCE="${FENGSHUI_CONV_PATIENCE:-2}"
N_MAX="${FENGSHUI_CONV_NMAX:-16}"
N_EVALS="${FENGSHUI_CONV_NEVALS:-500}"
SEED="${FENGSHUI_CONV_SEED:-42}"

ASSESS_ONLY=0
METRICS="energy,energy_cost,edp,edp_cost"

usage() {
  sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

while (( $# > 0 )); do
  case "$1" in
    --assess-only) ASSESS_ONLY=1 ;;
    --metrics) METRICS="${2:?--metrics needs a comma-separated list}"; shift ;;
    -h|--help) usage ;;
    *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

readonly RUN_ROOT="$RUNS_ROOT/$RUN_ID"
readonly LOG_ROOT="$RUN_ROOT/logs"
readonly MANIFEST="$RUN_ROOT/run_manifest.txt"
readonly SUMMARY_CSV="$RUN_ROOT/convergence_summary.csv"

die() { echo "ERROR: $*" >&2; exit 1; }

# ── preflight ───────────────────────────────────────────────────────────────
[[ -f "$DRIVER" ]] || die "missing driver: $DRIVER"
[[ -f "$DB_PATH" ]] || die "database not found: $DB_PATH (set FENGSHUI_DB)"
[[ -d "$ARCHGYM" ]] || die "archgym_results not found: $ARCHGYM (set FENGSHUI_ARCHGYM)"
[[ -f "$THIS_DIR/network_analysis.csv" ]] \
  || die "network_analysis.csv missing from $THIS_DIR; the perf model opens it by relative path"
"$PYTHON_BIN" -c 'import gym' 2>/dev/null \
  || die "'gym' not importable by $PYTHON_BIN. The chain search needs it; activate the env that has gym (conda 'base', not 'mozart')."

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

# Determinism: same knobs the AE pools were built under. Without these the inner
# GA draws from the global random stream and the endpoint can shift run to run.
export FENGSHUI_DETERMINISTIC="${FENGSHUI_DETERMINISTIC:-1}"
export FENGSHUI_GA_SEED="${FENGSHUI_GA_SEED:-0}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

{
  echo "run_id=$RUN_ID"
  echo "mode=$([[ $ASSESS_ONLY == 1 ]] && echo assess-only || echo full)"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "driver=$DRIVER"
  echo "python=$("$PYTHON_BIN" -V 2>&1)"
  echo "database=$DB_PATH"
  echo "database_sha256=$(sha256sum "$DB_PATH" | cut -d' ' -f1)"
  echo "archgym_results=$ARCHGYM"
  echo "chain_version=$CHAIN_VERSION"
  echo "metrics=$METRICS"
  echo "criterion=marginal improvement < ${THRESHOLD_PCT}% for ${PATIENCE} consecutive steps"
  echo "n_max=$N_MAX"
  echo "n_evals=$N_EVALS"
  echo "seed=$SEED"
  echo "FENGSHUI_DETERMINISTIC=$FENGSHUI_DETERMINISTIC"
  echo "FENGSHUI_GA_SEED=$FENGSHUI_GA_SEED"
  echo "PYTHONHASHSEED=$PYTHONHASHSEED"
} > "$MANIFEST"

echo "metric,objective,cost_aware,source_n,converged_n,extension_needed,steps_added,threshold_pct,patience" > "$SUMMARY_CSV"

echo "=============================================================================="
echo " Chiplet-pool convergence endpoints  (run $RUN_ID)"
echo " criterion: marginal improvement < ${THRESHOLD_PCT}% for ${PATIENCE} consecutive steps"
echo " mode:      $([[ $ASSESS_ONLY == 1 ]] && echo 'assess only (no heavy compute)' || echo 'assess, then extend what needs it')"
echo "=============================================================================="

overall_rc=0

IFS=',' read -ra METRIC_LIST <<< "$METRICS"
for metric in "${METRIC_LIST[@]}"; do
  case "$metric" in
    energy)      obj=energy; cost_flag="" ;;
    energy_cost) obj=energy; cost_flag="--cost-aware" ;;
    edp)         obj=edp;    cost_flag="" ;;
    edp_cost)    obj=edp;    cost_flag="--cost-aware" ;;
    *) die "unknown metric: $metric (expected energy|energy_cost|edp|edp_cost)" ;;
  esac

  src_dir="$ARCHGYM/${CHAIN_VERSION}_${metric}_chain"
  out_dir="$ARCHGYM/${CHAIN_VERSION}_${metric}_chain_converged"
  log="$LOG_ROOT/${metric}.log"
  [[ -d "$src_dir" ]] || die "no source chain for $metric: $src_dir"

  src_n="$("$PYTHON_BIN" - "$src_dir" <<'PY'
import sys, glob, os, csv
d = sys.argv[1]
f = sorted(glob.glob(os.path.join(d, 'saeo_isaeo_chain_*.csv')))[-1]
with open(f) as fh:
    print(max(int(r['n_chiplets']) for r in csv.DictReader(fh)))
PY
)"

  echo
  echo "─── $metric ─────────────────────────────────────────────────────────────"
  echo "    existing chain endpoint: N=$src_n"

  args=(--objective "$obj" --database "$DB_PATH"
        --source-dir "$src_dir" --output-dir "$out_dir"
        --threshold-pct "$THRESHOLD_PCT" --patience "$PATIENCE"
        --n-max "$N_MAX" --n-evals "$N_EVALS" --seed "$SEED")
  [[ -n "$cost_flag" ]] && args+=("$cost_flag")
  [[ $ASSESS_ONLY == 1 ]] && args+=(--dry-run)

  set +e
  ( cd "$THIS_DIR" && "$PYTHON_BIN" "$DRIVER" "${args[@]}" ) > "$log" 2>&1
  rc=$?
  set -e
  (( rc != 0 )) && overall_rc=$rc

  # Surface the lines that explain the verdict, not the whole log.
  grep -E "Trailing marginals|Consecutive sub-|ALREADY CONVERGED|NOT converged|=== DONE" "$log" \
    | sed 's/^.*INFO - /    /' || true

  conv_n="$(grep -oE 'converged_N=[0-9]+' "$log" | tail -1 | cut -d= -f2 || true)"
  [[ -z "$conv_n" ]] && conv_n="none"
  if grep -q "ALREADY CONVERGED" "$log"; then
    ext_needed=no
  else
    ext_needed=yes
  fi
  steps_added="0"
  [[ "$conv_n" != "none" && "$conv_n" -gt "$src_n" ]] && steps_added=$(( conv_n - src_n ))

  cost_aware=$([[ -n "$cost_flag" ]] && echo true || echo false)
  echo "$metric,$obj,$cost_aware,$src_n,$conv_n,$ext_needed,$steps_added,$THRESHOLD_PCT,$PATIENCE" >> "$SUMMARY_CSV"
  echo "outcome=$metric converged_n=$conv_n extension_needed=$ext_needed steps_added=$steps_added rc=$rc" >> "$MANIFEST"
  (( rc != 0 )) && echo "    !! converge_chain.py exited $rc - see $log"
done

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
( cd "$RUN_ROOT" && find . -type f ! -name SHA256SUMS -print0 | sort -z \
    | xargs -0 sha256sum > SHA256SUMS ) || true

echo
echo "=============================================================================="
column -t -s, "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
echo "=============================================================================="
echo "manifest: $MANIFEST"
echo "logs:     $LOG_ROOT"
if [[ $ASSESS_ONLY == 1 ]]; then
  echo
  echo "Assess-only: no chain was extended. Any metric with extension_needed=yes"
  echo "still needs a full run to establish its endpoint."
else
  echo
  echo "Consolidated chains -> $ARCHGYM/${CHAIN_VERSION}_<metric>_chain_converged/"
  echo "generate_paper_fig10.py prefers these over the base chains automatically."
fi
exit $overall_rc
