#!/usr/bin/env python3
"""
CNN sweep for MobileNet V3 Small and RepLKNet31B workloads.

Tests utilization, weight sharing (batching), and energy across
three architectures (eyeriss_like, simba_like, gemmini_like)
at different PE array and GLB configurations.

Usage:
  python3 run_cnn_sweep.py --dry-run                 # count configs
  python3 run_cnn_sweep.py --output-csv cnn_test.csv  # run sweep
"""
import sys
import os
import argparse
import csv
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
from global_parameter import cycle_time

# ============================================================
# Workload definitions
# ============================================================
WORKLOAD_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "workloads"))
if not os.path.isdir(WORKLOAD_DIR):
    WORKLOAD_DIR = "/workspace/chiplet_timeloop/workloads"

# All layers from each CNN model
CNN_MODELS = {
    "mobilenet_v3_small": [
        # --- features.2 (block 2) ---
        "layer6_features_2_block_0_0",      # 1x1 expand: C=16, M=72, P=56, Q=56
        "layer7_features_2_block_1_0",      # DW 3x3: G=72, P=28, Q=28
        "layer8_features_2_block_2_0",      # 1x1 reduce: C=72, M=24, P=28, Q=28
        # --- features.5 (block 5) ---
        "layer17_features_5_block_0_0",     # 1x1 expand
        "layer18_features_5_block_1_0",     # DW 3x3
        "layer19_features_5_block_2_fc1",   # SE FC1: C=240, M=64, P=1, Q=1
        "layer20_features_5_block_2_fc2",   # SE FC2
        "layer21_features_5_block_3_0",     # 1x1 project
        # --- features.10 (block 10) ---
        "layer42_features_10_block_0_0",    # 1x1 expand
        "layer43_features_10_block_1_0",    # DW 3x3
        "layer44_features_10_block_2_fc1",  # SE FC1
        "layer45_features_10_block_2_fc2",  # SE FC2
        "layer46_features_10_block_3_0",    # 1x1 project
    ],
    "replknet31b": [
        # --- stages_0, block 0 ---
        "layer1_stages_0_blocks_0_pw1_conv",                       # pw1: C=128, M=128, 56x56
        "layer2_stages_0_blocks_0_large_kernel_lkb_origin_conv",   # DW 31x31: G=128, 56x56
        "layer3_stages_0_blocks_0_large_kernel_small_conv_conv",   # DW 5x5: G=128, 56x56
        "layer4_stages_0_blocks_0_pw2_conv",                       # pw2: C=128, M=128, 56x56
        # --- stages_0, block 1 ---
        "layer5_stages_0_blocks_1_pw1_conv",                       # pw1: C=128, M=512, 56x56
        "layer6_stages_0_blocks_1_pw2_conv",                       # pw2: C=512, M=128, 56x56
        # --- stages_1, block 0 ---
        "layer13_stages_1_blocks_0_pw1_conv",                      # pw1: C=256, M=256, 28x28
        "layer14_stages_1_blocks_0_large_kernel_lkb_origin_conv",  # DW 29x29: G=256, 28x28
        "layer15_stages_1_blocks_0_large_kernel_small_conv_conv",  # DW 5x5: G=256, 28x28
        "layer16_stages_1_blocks_0_pw2_conv",                      # pw2: C=256, M=256, 28x28
        # --- stages_1, block 1 ---
        "layer17_stages_1_blocks_1_pw1_conv",                      # pw1: C=256, M=1024, 28x28
        "layer18_stages_1_blocks_1_pw2_conv",                      # pw2: C=1024, M=256, 28x28
    ],
}

# ============================================================
# Sweep configuration
# ============================================================
ARCHS = ["eyeriss_like", "simba_like", "gemmini_like"]
PE_SCALES = [1, 2, 3, 4]
PE_COMBOS = [(px, py) for px in PE_SCALES for py in PE_SCALES]  # 16 asymmetric combos
GLB_SCALES = [1, 4, 9, 16]
BATCH_SIZES = [1, 4, 8, 16]
DEFAULT_SWEEP_DRAM_OPTIONS = ["LPDDR5", "GDDR7"]


# ============================================================
# Helpers
# ============================================================

def get_cnn_workload_path(model, layer_name):
    """Resolve workload YAML path for a CNN layer."""
    path = os.path.join(WORKLOAD_DIR, model, layer_name + ".yaml")
    if os.path.exists(path):
        return path
    return None


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


def run_one_config(problem_path, model, layer_name, batch_size,
                   arch, glb_scale, pe_x, pe_y, dram_config,
                   output_base_dir, remove_bw_limit=True):
    """Run a single CNN Timeloop config. Returns dict with results or None."""
    problem_name = os.path.basename(problem_path).replace(".yaml", "")

    result = timeloop_helper.run_mapper(
        net=model,
        problem=problem_path,
        batch_size=batch_size,
        mapper_idx=0,  # energy-first
        output_tile_channels=1,
        tp_degree=1,  # no TP for CNN
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

    # Parse stats
    proc_id = (
        f"arch={arch}"
        f"@glb_scale={glb_scale}"
        f"@pe_x_scale={pe_x}"
        f"@pe_y_scale={pe_y}"
    )
    dram_id = f"{dram_config['I']}@{dram_config['O']}"
    output_dir = (
        f"{output_base_dir}/outputs/{model}/{problem_name}/{batch_size}/"
        f"1/0/single/1/1/{proc_id}/{dram_id}"
    )
    stats_file = os.path.join(output_dir, "timeloop-mapper.stats.txt")
    util, energy_uj, cycles, fj_per_compute, gflops, area_mm2 = \
        parse_stats_file(stats_file)

    return {
        "model": model,
        "layer": layer_name,
        "batch_size": batch_size,
        "arch": arch,
        "pe_x_scale": pe_x,
        "pe_y_scale": pe_y,
        "pe_scale": "%dx%d" % (pe_x, pe_y),
        "glb_scale": glb_scale,
        "dram_I": dram_config["I"],
        "dram_O": dram_config["O"],
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

def load_existing_layers(db_path):
    """Load (model, layer) pairs already present in a database CSV."""
    existing = set()
    if not os.path.exists(db_path):
        return existing
    with open(db_path) as f:
        reader = csv.DictReader(f)
        # Support both run_cnn_sweep CSV (model/layer) and unified_database (net/layer_name)
        for row in reader:
            model = row.get("model") or row.get("net", "")
            layer = row.get("layer") or row.get("layer_name", "")
            if model and layer:
                existing.add((model, layer))
    return existing


def build_configs(dram_configs, output_base_dir, skip_layers=None):
    """Build all configs. Returns list of tuples.

    Args:
        skip_layers: optional set of (model, layer) tuples to skip.
    """
    if skip_layers is None:
        skip_layers = set()
    configs = []
    for model, layers in CNN_MODELS.items():
        for layer in layers:
            if (model, layer) in skip_layers:
                continue
            path = get_cnn_workload_path(model, layer)
            if path is None:
                print(f"WARNING: workload not found: {model}/{layer}")
                continue
            for arch in ARCHS:
                for glb in GLB_SCALES:
                    for (px, py) in PE_COMBOS:
                        for dram in dram_configs:
                            for batch in BATCH_SIZES:
                                configs.append(
                                    (path, model, layer, batch, arch, glb,
                                     px, py, dram, output_base_dir)
                                )
    return configs


# ============================================================
# CSV output
# ============================================================

CSV_FIELDNAMES = [
    "model", "layer", "batch_size",
    "arch", "pe_x_scale", "pe_y_scale", "pe_scale", "glb_scale",
    "dram_I", "dram_O",
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


def print_summary(results):
    """Print a summary of results grouped by analysis type."""
    if not results:
        print("No results to summarize.")
        return

    print("\n" + "=" * 80)
    print("CNN SWEEP RESULTS SUMMARY")
    print("=" * 80)

    # --- 1. Utilization across architectures and PE configs ---
    print("\n--- Utilization by Architecture and PE Config ---")
    print(f"{'Model':<25} {'Layer':<20} {'Arch':<15} {'PE':<6} "
          f"{'GLB':<5} {'B':<3} {'Util':>7}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: (
            x["model"], x["layer"], x["arch"], x["pe_scale"], x["glb_scale"])):
        print(f"{r['model']:<25} {r['layer'][:20]:<20} {r['arch']:<15} "
              f"{r['pe_scale']:<6} {r['glb_scale']:<5} {r['batch_size']:<3} "
              f"{r['utilization']:>6.1%}")

    # --- 2. Weight sharing: compare batch=1 vs batch=4 energy ---
    print("\n--- Weight Sharing Analysis (Energy per Sample) ---")
    print(f"{'Model':<25} {'Layer':<20} {'Arch':<15} {'PE':<6} "
          f"{'GLB':<5} {'E(B=1)':>10} {'E(B=4)/4':>10} {'Saving':>8}")
    print("-" * 105)

    # Index results by config
    by_config = {}
    for r in results:
        key = (r["model"], r["layer"], r["arch"], r["pe_scale"], r["glb_scale"])
        by_config.setdefault(key, {})[r["batch_size"]] = r

    for key in sorted(by_config.keys()):
        batches = by_config[key]
        if 1 in batches and 4 in batches:
            e1 = batches[1]["energy_uj"]
            e4 = batches[4]["energy_uj"]
            if e1 > 0:
                e4_per_sample = e4 / 4.0
                saving = 1.0 - (e4_per_sample / e1)
                model, layer, arch, pe, glb = key
                print(f"{model:<25} {layer[:20]:<20} {arch:<15} {pe:<6} "
                      f"{glb:<5} {e1:>10.3f} {e4_per_sample:>10.3f} "
                      f"{saving:>7.1%}")

    # --- 3. Energy comparison across architectures ---
    print("\n--- Energy (uJ) by Architecture (batch=1, PE=1x1, GLB=1) ---")
    print(f"{'Model':<25} {'Layer':<30} {'Eyeriss':>10} "
          f"{'Gemmini':>10} {'Simba':>10}")
    print("-" * 95)

    by_layer = {}
    for r in results:
        if r["batch_size"] == 1 and r["pe_scale"] == "1x1" and r["glb_scale"] == 1:
            key = (r["model"], r["layer"])
            by_layer.setdefault(key, {})[r["arch"]] = r["energy_uj"]

    for key in sorted(by_layer.keys()):
        model, layer = key
        archs = by_layer[key]
        e_ey = archs.get("eyeriss_like", 0)
        e_ge = archs.get("gemmini_like", 0)
        e_si = archs.get("simba_like", 0)
        print(f"{model:<25} {layer[:30]:<30} {e_ey:>10.3f} "
              f"{e_ge:>10.3f} {e_si:>10.3f}")

    print("\n" + "=" * 80)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CNN sweep for MobileNet V3 and RepLKNet31B")

    parser.add_argument(
        "--output-csv", type=str, default="cnn_sweep_results.csv",
        help="Output CSV path (default: cnn_sweep_results.csv)")
    parser.add_argument(
        "--output-base-dir", type=str, default=_THIS_DIR,
        help="Base directory for Timeloop outputs (default: script dir)")
    parser.add_argument(
        "--n-jobs", type=int, default=60,
        help="Number of parallel Timeloop jobs (default: 60)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just count configs; do not run Timeloop")
    parser.add_argument(
        "--with-bw-limit", action="store_true",
        help="Enable real DRAM bandwidth limits. "
             "Default is no BW limit (1000x). "
             "When set, sweeps multiple DRAM configs.")
    parser.add_argument(
        "--compensate", type=str, default=None, metavar="DB_CSV",
        help="Only run layers missing from the given database CSV. "
             "Reads existing (model, layer) pairs and skips them.")

    args = parser.parse_args()

    # --- Resolve DRAM configs ---
    remove_bw_limit = not args.with_bw_limit
    if remove_bw_limit:
        dram_configs = [{"I": "LPDDR5", "O": "LPDDR5"}]
    else:
        dram_configs = [{"I": i, "O": o}
                        for i in DEFAULT_SWEEP_DRAM_OPTIONS
                        for o in DEFAULT_SWEEP_DRAM_OPTIONS]

    os.makedirs(args.output_base_dir, exist_ok=True)

    # --- Compensate mode: skip layers already in DB ---
    skip_layers = set()
    if args.compensate:
        skip_layers = load_existing_layers(args.compensate)
        print("Compensate mode: found %d existing (model, layer) pairs in %s" %
              (len(skip_layers), args.compensate))

    configs = build_configs(dram_configs, args.output_base_dir,
                            skip_layers=skip_layers)

    print("CNN Models: %s" % list(CNN_MODELS.keys()))
    print("Architectures: %s" % ARCHS)
    print("PE combos (%d): %s" % (len(PE_COMBOS), PE_COMBOS))
    print("GLB scales: %s" % GLB_SCALES)
    print("Batch sizes: %s" % BATCH_SIZES)
    print("DRAM configs: %d" % len(dram_configs))
    print("Total configs to run: %d" % len(configs))

    if args.dry_run:
        est_seconds = len(configs) / max(args.n_jobs, 1) * 25
        est_hours = est_seconds / 3600
        print("Estimated time: %.1f hours = %.1f days" % (
            est_hours, est_hours / 24))
        return

    print("Running with %d parallel jobs..." % args.n_jobs)
    t0 = time.time()

    results = joblib.Parallel(n_jobs=args.n_jobs, verbose=10)(
        joblib.delayed(run_one_config)(
            path, model, layer, batch, arch, glb, px, py, dram,
            out_dir, remove_bw_limit,
        )
        for path, model, layer, batch, arch, glb, px, py, dram, out_dir
        in configs
    )

    elapsed = time.time() - t0
    ok_results = [r for r in results if r is not None]
    print("\nDone in %.1f minutes. %d/%d succeeded." % (
        elapsed / 60, len(ok_results), len(configs)))

    if ok_results:
        write_csv(ok_results, args.output_csv)
        print_summary(ok_results)


if __name__ == "__main__":
    main()
