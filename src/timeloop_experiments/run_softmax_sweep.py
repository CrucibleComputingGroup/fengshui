#!/usr/bin/env python3
"""
Softmax sweep for LLaMA and Qwen3 models on simple_vector (1D) architecture.

Sweeps all 4 softmax sub-operations (max, sub_exp, sum, div) across
(model, phase, seq_len, GLB, PE_x, TP, DRAM) configurations.

Unlike run_sweep.py, softmax always uses the simple_vector architecture
(1D PE array), so there is no 2D arch sweep and only PE_x scale matters.

Usage (inside Docker my_timeloop_env_v2):
  export LD_LIBRARY_PATH=/workspace/accelergy-timeloop-infrastructure/src/timeloop/lib:$LD_LIBRARY_PATH

  # Dry run — count configs and estimate time
  python3 run_softmax_sweep.py --dry-run

  # Run all models
  python3 run_softmax_sweep.py --models all --output-csv softmax_sweep.csv --n-jobs 120

  # Run only LLaMA
  python3 run_softmax_sweep.py --models llama --output-csv llama_softmax.csv

  # Distributed execution (shard 0 of 4)
  python3 run_softmax_sweep.py --run-id 0 --total-runs 4 --output-csv shard0.csv
"""
import sys
import os
import argparse
import csv
import math
import time

import joblib
import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in [os.path.join(_THIS_DIR, "..", "scripts"),
                   os.path.join(_THIS_DIR, "..")]:
    if (os.path.isfile(os.path.join(_candidate, "utility_functions.py"))
            and _candidate not in sys.path):
        sys.path.insert(0, _candidate)

import timeloop_helper
from global_parameter import (
    prefill_seq_lens,
    decode_kv_lens,
    pe_scales,
    glb_scales,
    tp_degrees,
    num_mapping_per_arch,
    LLAMA_SOFTMAX_OPS,
    LLAMA_TP_CONFIG,
    QWEN_TP_CONFIG,
)

# ============================================================
# Model definitions
# ============================================================
LLAMA_MODELS = ["llama3.1_8b", "llama3.1_70b"]
QWEN_MODELS = ["qwen3_30b_a3b", "qwen3_235b_a22b"]

SOFTMAX_OPS = LLAMA_SOFTMAX_OPS  # same for all models

# ============================================================
# Sweep defaults
# ============================================================
PREFILL_SEQS = prefill_seq_lens
DECODE_KVS = decode_kv_lens
# Softmax always uses simple_vector — only PE_x scale matters (meshY=1)
PE_X_SCALES = pe_scales
GLB_SCALES = glb_scales
TP_DEGREES = tp_degrees
N_MAPPER = num_mapping_per_arch

DEFAULT_SWEEP_DRAM_OPTIONS = ["LPDDR5", "GDDR7"]

WORKLOAD_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "workloads"))
if not os.path.isdir(WORKLOAD_DIR):
    WORKLOAD_DIR = "/workspace/chiplet_timeloop/workloads"


# ============================================================
# Helpers
# ============================================================

def _model_family(model_name):
    if model_name.startswith("llama"):
        return "llama"
    return "qwen"


def _get_tp_config(model_name):
    if _model_family(model_name) == "llama":
        return LLAMA_TP_CONFIG
    return QWEN_TP_CONFIG


def get_workload_path(model, phase, length, op):
    """Resolve the workload YAML path for (model, phase, length, op).

    Returns (network_name, path_or_None).
    """
    if phase == "prefill":
        net = "%s_prefill_s%d" % (model, length)
    else:
        net = "%s_decode_kv%d" % (model, length)

    for candidate in ["layer0_%s" % op, op]:
        path = os.path.normpath(
            os.path.join(WORKLOAD_DIR, net, candidate + ".yaml"))
        if os.path.exists(path):
            return net, path
    return net, None


def _extract_model_from_net(net):
    if "_prefill_" in net:
        return net.split("_prefill_")[0]
    if "_decode_" in net:
        return net.split("_decode_")[0]
    return net


# ============================================================
# Run a single Timeloop configuration
# ============================================================

def parse_stats_file(stats_file):
    """Parse a timeloop-mapper stats file and return extracted metrics."""
    util = 0.0
    energy_uj = 0.0
    cycles = 0
    fj_per_compute = 0.0
    gflops = 0.0
    area_mm2 = 0.0

    if not os.path.exists(stats_file):
        return util, energy_uj, cycles, fj_per_compute, gflops, area_mm2

    with open(stats_file) as f:
        in_fj = False
        for line in f:
            ls = line.strip()
            if ls.startswith("Utilization:"):
                util = float(ls.split(":")[1].strip().replace("%", "")) / 100.0
            elif ls.startswith("Cycles:"):
                cycles = int(ls.split(":")[1].strip())
            elif ls.startswith("Energy:"):
                val = ls.split(":")[1].strip()
                if "uJ" in val:
                    energy_uj = float(val.replace("uJ", "").strip())
                elif "mJ" in val:
                    energy_uj = float(val.replace("mJ", "").strip()) * 1000
            elif ls.startswith("GFLOPs"):
                gflops = float(ls.split(":")[1].strip())
            elif ls.startswith("Area:"):
                area_mm2 = float(
                    ls.split(":")[1].strip().replace("mm^2", "").strip())
            elif ls == "fJ/Compute":
                in_fj = True
            elif in_fj and "Total" in ls and "=" in ls:
                fj_per_compute = float(ls.split("=")[-1].strip())
                in_fj = False

    return util, energy_uj, cycles, fj_per_compute, gflops, area_mm2


def run_one_config(problem_path, net, batch_size, glb_scale,
                   pe_x, dram_config, tp_degree, mapper_idx,
                   output_base_dir, remove_bw_limit=True):
    """Run a single softmax Timeloop config. Returns dict or None."""
    with open(problem_path) as f:
        pdata = yaml.safe_load(f)
    pdims = pdata["problem"]["instance"]

    op_basename = os.path.basename(problem_path).replace(".yaml", "")
    op_name = op_basename.replace("layer0_", "")

    # TP: H dimension for softmax
    tp_config = _get_tp_config(_extract_model_from_net(net))
    tp_dim = tp_config.get(op_name, "H")
    output_feature = pdims.get(tp_dim, 1)
    otc = int(math.floor(output_feature / tp_degree))

    # arch_target is overridden to simple_vector inside run_mapper for softmax
    result = timeloop_helper.run_mapper(
        net=net,
        problem=problem_path,
        batch_size=batch_size,
        mapper_idx=mapper_idx,
        output_tile_channels=otc,
        tp_degree=tp_degree,
        arch_target="simple_vector",
        glb_scale=glb_scale,
        pe_x_scale=pe_x,
        pe_y_scale=1,   # 1D array: meshY=1 always
        dram_config=dram_config,
        output_base_dir=output_base_dir,
        remove_bw_limit=remove_bw_limit,
    )

    config_id, problem_id, res = result
    if config_id is None:
        return None

    # Parse stats from the output directory
    output_dir = (
        "%s/outputs/%s/%s/%d/1/%d/single/%d/%d/"
        "arch=%s@glb_scale=%s@pe_x_scale=%s@pe_y_scale=%s/%s@%s"
        % (output_base_dir, net, op_basename, batch_size, mapper_idx,
           otc, tp_degree, "simple_vector", glb_scale, pe_x, 1,
           dram_config["I"], dram_config["O"])
    )
    stats_file = os.path.join(output_dir, "timeloop-mapper.stats.txt")
    util, energy_uj, cycles, fj_per_compute, gflops, area_mm2 = \
        parse_stats_file(stats_file)

    return {
        "model": _extract_model_from_net(net),
        "workload": net,
        "operator": op_name,
        "batch_size": batch_size,
        "tp_degree": tp_degree,
        "arch": "simple_vector",
        "pe_x_scale": pe_x,
        "pe_y_scale": 1,
        "pe_scale": "%dx%d" % (pe_x, 1),
        "glb_scale": glb_scale,
        "dram_I": dram_config["I"],
        "dram_O": dram_config["O"],
        "mapper_objective": "energy" if mapper_idx == 0 else "delay",
        "utilization": util,
        "energy_uj": energy_uj,
        "cycles": cycles,
        "latency_s": cycles * 1e-9,
        "fj_per_compute": fj_per_compute,
        "gflops": gflops,
        "area_mm2": area_mm2,
    }


# ============================================================
# Config builder
# ============================================================

def build_configs(models, dram_configs, run_id=None, total_runs=None):
    """Build all softmax sweep configs.

    Returns list of tuples:
        (problem_path, net, batch, glb, pe_x, dram, tp, mapper_idx)
    """
    configs = []

    hw_combos = [
        (glb, px, dram, tp, mi)
        for glb in GLB_SCALES
        for px in PE_X_SCALES
        for dram in dram_configs
        for tp in TP_DEGREES
        for mi in range(N_MAPPER)
    ]

    for model in models:
        for op in SOFTMAX_OPS:
            for phase in ["prefill", "decode"]:
                lengths = PREFILL_SEQS if phase == "prefill" else DECODE_KVS
                for length in lengths:
                    net, path = get_workload_path(model, phase, length, op)
                    if path is None or not os.path.exists(path):
                        continue
                    # Softmax: batch_size=1 (elementwise, no weight reuse)
                    for hw in hw_combos:
                        configs.append((path, net, 1) + hw)

    # Shard for distributed execution
    if run_id is not None and total_runs is not None:
        total = len(configs)
        chunk = total // total_runs
        start = run_id * chunk
        end = start + chunk if run_id < total_runs - 1 else total
        configs = configs[start:end]

    return configs


# ============================================================
# CSV output
# ============================================================

CSV_FIELDNAMES = [
    "model", "workload", "operator", "batch_size", "tp_degree",
    "arch", "pe_x_scale", "pe_y_scale", "pe_scale", "glb_scale",
    "dram_I", "dram_O", "mapper_objective",
    "utilization", "energy_uj", "cycles", "latency_s",
    "fj_per_compute", "gflops", "area_mm2",
]


def write_csv(results, output_csv):
    """Write result dicts to a CSV file."""
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print("Wrote %d rows to %s" % (len(results), output_csv))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Softmax sweep for LLaMA and Qwen3 on simple_vector")

    parser.add_argument(
        "--models", type=str, default="all",
        choices=["llama", "qwen", "all"],
        help="Which model family to sweep (default: all)")
    parser.add_argument(
        "--output-csv", type=str, default="softmax_sweep_results.csv",
        help="Output CSV path (default: softmax_sweep_results.csv)")
    parser.add_argument(
        "--output-base-dir", type=str, default=_THIS_DIR,
        help="Base directory for Timeloop outputs (default: script dir)")
    parser.add_argument(
        "--run-id", type=int, default=None,
        help="Shard index for distributed execution (0-based)")
    parser.add_argument(
        "--total-runs", type=int, default=None,
        help="Total number of shards for distributed execution")
    parser.add_argument(
        "--n-jobs", type=int, default=60,
        help="Number of parallel Timeloop jobs (default: 60)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just count configs and estimate time; do not run Timeloop")
    parser.add_argument(
        "--with-bw-limit", action="store_true",
        help="Enable real DRAM bandwidth limits. "
             "Default is no BW limit (1000x). "
             "When set, sweeps multiple DRAM configs.")

    args = parser.parse_args()

    # --- Resolve model list ---
    if args.models == "llama":
        models = LLAMA_MODELS
    elif args.models == "qwen":
        models = QWEN_MODELS
    else:
        models = LLAMA_MODELS + QWEN_MODELS

    # --- Resolve DRAM configs ---
    remove_bw_limit = not args.with_bw_limit
    if remove_bw_limit:
        dram_configs = [{"I": "LPDDR5", "O": "LPDDR5"}]
    else:
        dram_configs = [{"I": i, "O": o}
                        for i in DEFAULT_SWEEP_DRAM_OPTIONS
                        for o in DEFAULT_SWEEP_DRAM_OPTIONS]

    # --- Build configs ---
    configs = build_configs(
        models=models,
        dram_configs=dram_configs,
        run_id=args.run_id,
        total_runs=args.total_runs,
    )

    print("Models: %s" % [m for m in models])
    print("Softmax ops: %s" % SOFTMAX_OPS)
    print("Architecture: simple_vector (1D)")
    print("PE_x scales: %s" % PE_X_SCALES)
    print("GLB scales: %s" % GLB_SCALES)
    print("TP degrees: %s" % list(TP_DEGREES))
    print("DRAM configs: %d" % len(dram_configs))
    print("Total configs to run: %d" % len(configs))

    if args.dry_run:
        est_seconds = len(configs) / max(args.n_jobs, 1) * 15
        est_hours = est_seconds / 3600
        print("Estimated time: %.1f hours = %.1f days" % (
            est_hours, est_hours / 24))
        return

    # --- Run all configs ---
    print("Running with %d parallel jobs..." % args.n_jobs)
    t0 = time.time()

    results = joblib.Parallel(n_jobs=args.n_jobs, verbose=10)(
        joblib.delayed(run_one_config)(
            path, net, batch, glb, px, dram, tp, mi,
            args.output_base_dir,
            remove_bw_limit=remove_bw_limit,
        )
        for path, net, batch, glb, px, dram, tp, mi in configs
    )

    elapsed = time.time() - t0
    ok_results = [r for r in results if r is not None]
    print("\nDone in %.1f minutes. %d/%d succeeded." % (
        elapsed / 60, len(ok_results), len(configs)))

    # --- Write CSV ---
    if ok_results:
        write_csv(ok_results, args.output_csv)


if __name__ == "__main__":
    main()
