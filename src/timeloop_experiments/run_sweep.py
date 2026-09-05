#!/usr/bin/env python3
"""
Unified sweep for LLaMA and Qwen3 models.

Supports:
  - Both symmetric (1x1, 2x2, ...) and asymmetric (1x2, 2x1, ...) PE combos
  - LLaMA (8b, 70b) and Qwen3 (30b_a3b, 235b_a22b) models
  - Cross-network deduplication for projections
  - Distributed execution via --run-id/--total-runs

Usage:
  python3 run_sweep.py --dry-run                        # count all configs
  python3 run_sweep.py --models llama --output-csv llama.csv
  python3 run_sweep.py --models qwen --output-csv qwen.csv
  python3 run_sweep.py --models all --output-csv full.csv
  python3 run_sweep.py --with-bw-limit --output-csv bw.csv   # enable real BW
"""
import sys
import os
import argparse
import csv
import time
import math

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
    arch_targets,
    pe_scales,
    glb_scales,
    batch_configs,
    tp_degrees,
    num_mapping_per_arch,
    LLAMA_TP_CONFIG,
    QWEN_TP_CONFIG,
    cycle_time,
)

# ============================================================
# Model definitions
# ============================================================
LLAMA_MODELS = ["llama3.1_8b", "llama3.1_70b"]
QWEN_MODELS = ["qwen3_30b_a3b", "qwen3_235b_a22b"]

LLAMA_PROJECTION_OPS = ["q_proj", "k_proj", "o_proj", "gate_proj", "down_proj"]
LLAMA_ATTENTION_OPS = ["attn_qk", "attn_v"]

QWEN_PROJECTION_OPS = ["q_proj", "k_proj", "o_proj",
                        "expert_gate_proj", "expert_down_proj",
                        "router", "lm_head"]
QWEN_ATTENTION_OPS = ["attn_qk", "attn_v"]

# ============================================================
# Sweep defaults (overridden by CLI flags / --remove-bw-limit)
# ============================================================
PREFILL_SEQS = prefill_seq_lens
DECODE_KVS = decode_kv_lens
ARCHS = arch_targets
PE_SCALES = pe_scales
GLB_SCALES = glb_scales
BATCH_SIZES = batch_configs()
TP_DEGREES = tp_degrees
N_MAPPER = num_mapping_per_arch

# Only run bandwidth-distinct DRAM types by default.
# Same-bandwidth types (DDR5 vs LPDDR5, HBM3 vs GDDR7) produce identical
# mappings; energy can be derived analytically by scaling pJ/bit.
DEFAULT_SWEEP_DRAM_OPTIONS = ["LPDDR5", "GDDR7"]

WORKLOAD_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "workloads"))
if not os.path.isdir(WORKLOAD_DIR):
    WORKLOAD_DIR = "/workspace/chiplet_timeloop/workloads"


# ============================================================
# Helpers
# ============================================================

def _model_family(model_name):
    """Return 'llama' or 'qwen' for a model name."""
    if model_name.startswith("llama"):
        return "llama"
    return "qwen"


def _get_ops(model_name):
    """Return (projection_ops, attention_ops) for a given model."""
    if _model_family(model_name) == "llama":
        return LLAMA_PROJECTION_OPS, LLAMA_ATTENTION_OPS
    return QWEN_PROJECTION_OPS, QWEN_ATTENTION_OPS


def _get_tp_config(model_name):
    """Return the TP config dict for a given model."""
    if _model_family(model_name) == "llama":
        return LLAMA_TP_CONFIG
    return QWEN_TP_CONFIG


def get_workload_path(model, phase, length, op):
    """Resolve the workload YAML path for a given (model, phase, length, op).

    LLaMA workloads use ``layer0_{op}.yaml``.
    Qwen workloads may use ``layer0_{op}.yaml`` or ``{op}.yaml``.

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
    """Extract model name from network string like 'llama3.1_8b_prefill_s512'."""
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


def run_one_config(problem_path, net, batch_size, arch, glb_scale,
                   pe_x, pe_y, dram_config, tp_degree, mapper_idx,
                   output_base_dir, remove_bw_limit=True):
    """Run a single Timeloop config. Returns dict with results or None."""
    with open(problem_path) as f:
        pdata = yaml.safe_load(f)
    pdims = pdata["problem"]["instance"]

    # Determine op name (strip layer0_ prefix if present)
    op_basename = os.path.basename(problem_path).replace(".yaml", "")
    op_name = op_basename.replace("layer0_", "")

    # TP: divide the appropriate dimension
    tp_config = _get_tp_config(_extract_model_from_net(net))
    tp_dim = tp_config.get(op_name, "M")
    output_feature = pdims.get(tp_dim, 1)
    otc = int(math.floor(output_feature / tp_degree))

    result = timeloop_helper.run_mapper(
        net=net,
        problem=problem_path,
        batch_size=batch_size,
        mapper_idx=mapper_idx,
        output_tile_channels=otc,
        tp_degree=tp_degree,
        arch_target=arch,
        glb_scale=glb_scale,
        pe_x_scale=pe_x,
        pe_y_scale=pe_y,
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
           otc, tp_degree, arch, glb_scale, pe_x, pe_y,
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
        "arch": arch,
        "pe_x_scale": pe_x,
        "pe_y_scale": pe_y,
        "pe_scale": "%dx%d" % (pe_x, pe_y),
        "glb_scale": glb_scale,
        "dram_I": dram_config["I"],
        "dram_O": dram_config["O"],
        "mapper_objective": "energy" if mapper_idx == 0 else "delay",
        "utilization": util,
        "energy_uj": energy_uj,
        "cycles": cycles,
        "latency_s": cycles * cycle_time,
        "fj_per_compute": fj_per_compute,
        "gflops": gflops,
        "area_mm2": area_mm2,
    }


# ============================================================
# Config builder
# ============================================================

PE_COMBOS = [(px, py) for px in PE_SCALES for py in PE_SCALES]


def build_configs(models, dram_configs,
                  run_id=None, total_runs=None):
    """Build all configs with cross-network deduplication for projections.

    Returns (configs, dedup_info) where:
      - configs: list of tuples
            (problem_path, net, batch, arch, glb, pe_x, pe_y, dram, tp, mapper_idx)
      - dedup_info: list of dicts for post-hoc result duplication
    """
    configs = []
    dedup_info = []

    hw_combos = [
        (arch, glb, px, py, dram, tp, mi)
        for arch in ARCHS
        for glb in GLB_SCALES
        for (px, py) in PE_COMBOS
        for dram in dram_configs
        for tp in TP_DEGREES
        for mi in range(N_MAPPER)
    ]

    for model in models:
        proj_ops, attn_ops = _get_ops(model)

        # --- Projections: dedup across (network, batch) by N_eff ---
        for op in proj_ops:
            seen_n_eff = {}  # n_eff -> (net, problem_path, batch_size)

            all_combos = []
            for phase in ["prefill", "decode"]:
                lengths = PREFILL_SEQS if phase == "prefill" else DECODE_KVS
                for length in lengths:
                    net, path = get_workload_path(model, phase, length, op)
                    if path is None or not os.path.exists(path):
                        continue
                    with open(path) as f:
                        pdims = yaml.safe_load(f)["problem"]["instance"]
                    base_N = pdims.get("N", 1)
                    for batch in BATCH_SIZES:
                        n_eff = base_N * batch
                        all_combos.append((net, path, batch, n_eff))

            # Deduplicate: keep the first (net, batch) per n_eff
            for net, path, batch, n_eff in all_combos:
                if n_eff not in seen_n_eff:
                    seen_n_eff[n_eff] = (net, path, batch)
                    for hw in hw_combos:
                        configs.append((path, net, batch) + hw)
                else:
                    master_net, _master_path, master_batch = seen_n_eff[n_eff]
                    dedup_info.append({
                        "op": op,
                        "dup_net": net,
                        "dup_batch": batch,
                        "master_net": master_net,
                        "master_batch": master_batch,
                    })

        # --- Attention: no dedup (dims vary by seq/kv_len), batch=1 only ---
        for op in attn_ops:
            for phase in ["prefill", "decode"]:
                lengths = PREFILL_SEQS if phase == "prefill" else DECODE_KVS
                for length in lengths:
                    net, path = get_workload_path(model, phase, length, op)
                    if path is None or not os.path.exists(path):
                        continue
                    for hw in hw_combos:
                        configs.append((path, net, 1) + hw)

    # Shard for distributed execution
    if run_id is not None and total_runs is not None:
        total = len(configs)
        chunk = total // total_runs
        start = run_id * chunk
        end = start + chunk if run_id < total_runs - 1 else total
        configs = configs[start:end]

    return configs, dedup_info


def duplicate_deduped_results(ok_results, dedup_info):
    """Create duplicated result rows from dedup mapping.

    For each deduped combo, find the master results and create copies
    with the duplicate's network and batch size.

    Returns (extended_results, dup_count).
    """
    dup_count = 0

    # Build a lookup index for faster master result matching
    master_index = {}
    for r in ok_results:
        key = (r["operator"], r["workload"], r["batch_size"])
        master_index.setdefault(key, []).append(r)

    new_rows = []
    for dinfo in dedup_info:
        key = (dinfo["op"], dinfo["master_net"], dinfo["master_batch"])
        master_results = master_index.get(key, [])
        for mr in master_results:
            dup = mr.copy()
            dup["workload"] = dinfo["dup_net"]
            dup["batch_size"] = dinfo["dup_batch"]
            dup["model"] = _extract_model_from_net(dinfo["dup_net"])
            new_rows.append(dup)
            dup_count += 1

    ok_results.extend(new_rows)
    return ok_results, dup_count


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
        description="Unified sweep for LLaMA and Qwen3 models")

    parser.add_argument(
        "--models", type=str, default="all",
        choices=["llama", "qwen", "all"],
        help="Which model family to sweep (default: all)")
    parser.add_argument(
        "--output-csv", type=str, default="sweep_results.csv",
        help="Output CSV path (default: sweep_results.csv)")
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
    # Default: no BW limit (single LPDDR5, 1000x multiplier)
    # --with-bw-limit: real BW, sweep multiple DRAM types
    remove_bw_limit = not args.with_bw_limit
    if remove_bw_limit:
        dram_configs = [{"I": "LPDDR5", "O": "LPDDR5"}]
    else:
        dram_configs = [{"I": i, "O": o}
                        for i in DEFAULT_SWEEP_DRAM_OPTIONS
                        for o in DEFAULT_SWEEP_DRAM_OPTIONS]

    # --- Build configs ---
    configs, dedup_info = build_configs(
        models=models,
        dram_configs=dram_configs,
        run_id=args.run_id,
        total_runs=args.total_runs,
    )

    print("Models: %s" % [m for m in models])
    print("PE combos (%d): %s" % (len(PE_COMBOS), PE_COMBOS))
    print("DRAM configs: %d" % len(dram_configs))
    print("Total configs to run: %d" % len(configs))
    print("Deduped combos (will copy results): %d" % len(dedup_info))

    if args.dry_run:
        est_seconds = len(configs) / max(args.n_jobs, 1) * 25
        est_hours = est_seconds / 3600
        print("Estimated time: %.1f hours = %.1f days" % (
            est_hours, est_hours / 24))
        return

    # --- Run all configs ---
    print("Running with %d parallel jobs..." % args.n_jobs)
    t0 = time.time()

    results = joblib.Parallel(n_jobs=args.n_jobs, verbose=10)(
        joblib.delayed(run_one_config)(
            path, net, batch, arch, glb, px, py, dram, tp, mi,
            args.output_base_dir,
            remove_bw_limit=remove_bw_limit,
        )
        for path, net, batch, arch, glb, px, py, dram, tp, mi in configs
    )

    elapsed = time.time() - t0
    ok_results = [r for r in results if r is not None]
    print("\nDone in %.1f hours. %d/%d succeeded." % (
        elapsed / 3600, len(ok_results), len(configs)))

    # --- Duplicate deduped results ---
    ok_results, dup_count = duplicate_deduped_results(ok_results, dedup_info)
    print("Duplicated %d results from dedup mapping." % dup_count)
    print("Total result rows: %d" % len(ok_results))

    # --- Write CSV ---
    if ok_results:
        write_csv(ok_results, args.output_csv)


if __name__ == "__main__":
    main()
