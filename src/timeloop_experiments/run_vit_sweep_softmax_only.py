#!/usr/bin/env python3
"""
ViT sweep for vit_b16_s197, vit_l16_s197, vit_h14_s257.

Operator types
  - projection : q_proj / k_proj / v_proj / o_proj / gate_proj / down_proj / lm_head
                 shape (B, N, C, M) — eyeriss_like / simba_like / gemmini_like
  - attention  : attn_qk / attn_v
                 shape (B, H, Q, K, D) — eyeriss_like / simba_like / gemmini_like
  - softmax    : softmax_max / softmax_sub_exp / softmax_sum / softmax_div
                 shape (B, H, Q, K)  — simple_vector only

Usage examples
  # Quick sanity check: b16 model, all unique ops, all archs, default PE/GLB
  python3 run_vit_sweep.py --quick --output-csv vit_quick.csv

  # Sweep all 3 models, projection + attention, pe_scales 1 2 4
  python3 run_vit_sweep.py --ops proj attn --pe-scales 1 2 4 --output-csv vit_sweep.csv

  # Full sweep (all models, all ops, all configs)
  python3 run_vit_sweep.py --output-csv vit_full.csv

  # Dry-run: just count configs
  python3 run_vit_sweep.py --dry-run
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

# Ensure Timeloop shared libraries on LD_LIBRARY_PATH (Docker env)
_tl_lib = "/workspace/accelergy-timeloop-infrastructure/src/timeloop/lib"
if os.path.isdir(_tl_lib):
    os.environ["LD_LIBRARY_PATH"] = (
        _tl_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )

import timeloop_helper
from global_parameter import (
    VIT_MODELS, VIT_TP_CONFIG, cycle_time,
    pe_scales, glb_scales, batch_configs, tp_degrees, num_mapping_per_arch,
)

# ============================================================
# Workload directory
# ============================================================
WORKLOAD_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "workloads"))
if not os.path.isdir(WORKLOAD_DIR):
    WORKLOAD_DIR = "/workspace/chiplet_timeloop/workloads"

# Unique YAML stems per operator category
# (k/v_proj have same dims as q_proj; o_proj same as q_proj → skip duplicates)
PROJ_STEMS    = ["layer0_q_proj", "layer0_gate_proj", "layer0_down_proj"]
ATTN_STEMS    = ["layer0_attn_qk", "layer0_attn_v"]
SOFTMAX_STEMS = ["layer0_softmax_max", "layer0_softmax_sub_exp",
                  "layer0_softmax_sum", "layer0_softmax_div"]

MATMUL_ARCHS  = ["eyeriss_like", "simba_like", "gemmini_like"]
SOFTMAX_ARCH  = ["simple_vector"]

DEFAULT_MODELS    = ["vit_b16_s197"]
DEFAULT_OPS       = ["proj", "attn", "softmax"]
DEFAULT_PE_SCALES = pe_scales                    # [1, 2, 3, 4]  — from global_parameter
DEFAULT_GLB_SCALES= glb_scales                   # [1, 4, 9, 16] — from global_parameter
DEFAULT_BATCH     = batch_configs(transformer=True)  # [1, 4, 8, 16]
DEFAULT_DRAM      = [{"I": "LPDDR5", "O": "LPDDR5"}]
DEFAULT_TP        = tp_degrees                   # [1, 2]
N_MAPPER          = num_mapping_per_arch         # 1 = energy only; 2 = energy + delay


# ============================================================
# Config builder
# ============================================================

def _stems_for_ops(op_keys):
    stems = []
    if "proj"    in op_keys: stems += PROJ_STEMS
    if "attn"    in op_keys: stems += ATTN_STEMS
    if "softmax" in op_keys: stems += SOFTMAX_STEMS
    return stems


def build_configs(models, op_keys, pe_scales, glb_scales, batch_sizes,
                  dram_configs, output_base_dir,
                  matmul_archs=None, tp_list=None):
    """Return list of kwarg dicts for run_one_config."""
    if matmul_archs is None:
        matmul_archs = MATMUL_ARCHS
    if tp_list is None:
        tp_list = DEFAULT_TP
    stems = _stems_for_ops(op_keys)
    configs = []
    for model in models:
        model_dir = os.path.join(WORKLOAD_DIR, model)
        if not os.path.isdir(model_dir):
            print(f"[WARN] workload dir not found: {model_dir}")
            continue
        for stem in stems:
            yaml_path = os.path.join(model_dir, f"{stem}.yaml")
            if not os.path.isfile(yaml_path):
                print(f"[WARN] missing YAML: {yaml_path}")
                continue
            is_softmax = "softmax" in stem
            archs = SOFTMAX_ARCH if is_softmax else matmul_archs
            for arch in archs:
                # softmax ignores pe/glb scale (single-level vector arch)
                scale_combos = ([(px, 1, glb) for px in pe_scales for glb in glb_scales] if is_softmax
                                else [(px, py, glb)
                                      for px in pe_scales
                                      for py in pe_scales
                                      for glb in glb_scales])
                # attention and softmax are batch-agnostic (seq dims already in Q/K)
                effective_batches = [1] if ("attn" in stem or "softmax" in stem) else batch_sizes
                for (px, py, glb) in scale_combos:
                    for dram in dram_configs:
                        for batch in effective_batches:
                            for tp in tp_list:
                                for mi in range(N_MAPPER):
                                    configs.append(dict(
                                        model=model,
                                        stem=stem,
                                        yaml_path=yaml_path,
                                        arch=arch,
                                        pe_x=px,
                                        pe_y=py,
                                        glb_scale=glb,
                                        batch_size=batch,
                                        tp_degree=tp,
                                        dram_config=dram,
                                        mapper_idx=mi,
                                        output_base_dir=output_base_dir,
                                    ))
    return configs


# ============================================================
# Single-config runner
# ============================================================

def _parse_stats(stats_file):
    util = cycles = 0
    energy_uj = fj_per_compute = gflops = area = 0.0
    if not os.path.isfile(stats_file):
        return util, energy_uj, cycles, fj_per_compute, gflops, area
    in_fj = False
    with open(stats_file) as fh:
        for line in fh:
            ls = line.strip()
            if ls.startswith("Utilization:"):
                try: util = float(ls.split(":")[1].replace("%","")) / 100.0
                except ValueError: pass
            elif ls.startswith("Cycles:"):
                try: cycles = int(ls.split(":")[1])
                except ValueError: pass
            elif ls.startswith("Energy:"):
                val = ls.split(":")[1].strip()
                try:
                    if "uJ" in val:
                        energy_uj = float(val.replace("uJ",""))
                    elif "mJ" in val:
                        energy_uj = float(val.replace("mJ","")) * 1000
                except ValueError: pass
            elif ls.startswith("GFLOPs"):
                try: gflops = float(ls.split(":")[1])
                except ValueError: pass
            elif ls.startswith("Area:"):
                try: area = float(ls.split(":")[1].replace("mm^2",""))
                except ValueError: pass
            elif ls == "fJ/Compute":
                in_fj = True
            elif in_fj and "Total" in ls and "=" in ls:
                try: fj_per_compute = float(ls.split("=")[-1])
                except ValueError: pass
                in_fj = False
    return util, energy_uj, cycles, fj_per_compute, gflops, area


def run_one_config(model, stem, yaml_path, arch,
                   pe_x, pe_y, glb_scale, batch_size, tp_degree,
                   dram_config, mapper_idx, output_base_dir,
                   remove_bw_limit=True):
    import math
    # Compute output_tile_channels: dimension split by TP for this op
    op_name = stem.replace("layer0_", "")
    tp_dim  = VIT_TP_CONFIG.get(op_name, "M")
    with open(yaml_path) as f:
        pdims = yaml.safe_load(f)["problem"]["instance"]
    otc = int(math.floor(pdims.get(tp_dim, 1) / tp_degree))

    config_id, problem_id, res = timeloop_helper.run_mapper(
        net=model,
        problem=yaml_path,
        batch_size=batch_size,
        sequence_length=1,
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
    if config_id is None:
        return None

    proc_id  = (f"arch={arch}@glb_scale={glb_scale}"
                f"@pe_x_scale={pe_x}@pe_y_scale={pe_y}")
    dram_id  = f"{dram_config['I']}@{dram_config['O']}"
    out_dir  = (f"{output_base_dir}/outputs/{model}/{stem}/{batch_size}/1"
                f"/{mapper_idx}/single/{otc}/{tp_degree}/{proc_id}/{dram_id}")
    stats_file = os.path.join(out_dir, "timeloop-mapper.stats.txt")
    util, energy_uj, cycles, fj_per_compute, gflops, area = _parse_stats(stats_file)

    op_name = stem.replace("layer0_", "")
    return dict(
        model=model,
        workload=model,
        operator=op_name,
        batch_size=batch_size,
        tp_degree=tp_degree,
        arch=arch,
        pe_x_scale=pe_x,
        pe_y_scale=pe_y,
        pe_scale="%dx%d" % (pe_x, pe_y),
        glb_scale=glb_scale,
        dram_I=dram_config["I"],
        dram_O=dram_config["O"],
        mapper_objective="energy" if mapper_idx == 0 else "delay",
        utilization=util,
        energy_uj=energy_uj,
        cycles=cycles,
        latency_s=cycles * cycle_time,
        fj_per_compute=fj_per_compute,
        gflops=gflops,
        area_mm2=area,
    )


# ============================================================
# Output
# ============================================================

CSV_FIELDS = [
    "model", "workload", "operator", "batch_size", "tp_degree",
    "arch", "pe_x_scale", "pe_y_scale", "pe_scale", "glb_scale",
    "dram_I", "dram_O", "mapper_objective",
    "utilization", "energy_uj", "cycles", "latency_s",
    "fj_per_compute", "gflops", "area_mm2",
]


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows → {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ViT sweep: vit_b16_s197 / vit_l16_s197 / vit_h14_s257")

    parser.add_argument(
        "--models", nargs="+",
        choices=VIT_MODELS + ["all"], default=["vit_b16_s197"],
        help=f"ViT model(s) to sweep (default: vit_b16_s197). Choices: {VIT_MODELS}")
    parser.add_argument(
        "--ops", nargs="+",
        choices=["proj", "attn", "softmax", "all"], default=["all"],
        help="Operator categories (default: all)")
    parser.add_argument(
        "--archs", nargs="+",
        choices=MATMUL_ARCHS + ["all"], default=["all"],
        help="Architectures for projection/attention ops (default: all). "
             "Softmax always uses simple_vector.")
    parser.add_argument(
        "--pe-scales", nargs="+", type=int, default=DEFAULT_PE_SCALES,
        help=f"PE scale factors, applied to both X and Y (default: {DEFAULT_PE_SCALES})")
    parser.add_argument(
        "--glb-scales", nargs="+", type=int, default=DEFAULT_GLB_SCALES,
        help=f"GLB scale factors (default: {DEFAULT_GLB_SCALES})")
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH,
        help="Batch sizes (default: [1])")
    parser.add_argument(
        "--output-csv", type=str, default="vit_sweep_results.csv",
        help="Output CSV path (default: vit_sweep_results.csv)")
    parser.add_argument(
        "--output-base-dir", type=str, default=_THIS_DIR,
        help="Base directory for Timeloop output files (default: script dir)")
    parser.add_argument(
        "--n-jobs", type=int, default=12,
        help="Parallel Timeloop jobs (default: 12)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config count only; do not run Timeloop")
    parser.add_argument(
        "--with-bw-limit", action="store_true",
        help="Enable real DRAM bandwidth limits (default: 1000x / unlimited)")
    parser.add_argument(
        "--tp-degrees", nargs="+", type=int, default=DEFAULT_TP,
        help=f"Tensor-parallelism degrees (default: {DEFAULT_TP})")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick sanity test: vit_b16_s197 only, all unique ops, "
             "all archs, pe=1, glb=1, B=1, tp=1 (~22 configs)")

    args = parser.parse_args()

    # --quick overrides everything else
    if args.quick:
        models   = ["vit_b16_s197"]
        op_keys  = ["proj", "attn", "softmax"]
        pe_sc    = [1]
        glb_sc   = [1]
        batch_sz = [1]
        tp_list  = [1]
    else:
        models   = VIT_MODELS if "all" in args.models else args.models
        op_keys  = ["proj", "attn", "softmax"] if "all" in args.ops else args.ops
        pe_sc    = args.pe_scales
        glb_sc   = args.glb_scales
        batch_sz = args.batch_sizes
        tp_list  = args.tp_degrees

    # arch filter (softmax arch is fixed; this only affects matmul ops)
    matmul_archs = MATMUL_ARCHS
    if not args.quick and "all" not in args.archs:
        matmul_archs = [a for a in MATMUL_ARCHS if a in args.archs]

    remove_bw_limit = not args.with_bw_limit
    dram_configs = DEFAULT_DRAM

    os.makedirs(args.output_base_dir, exist_ok=True)
    configs = build_configs(models, op_keys, pe_sc, glb_sc, batch_sz,
                            dram_configs, args.output_base_dir,
                            matmul_archs=matmul_archs, tp_list=tp_list)

    print(f"Models     : {models}")
    print(f"Op cats    : {op_keys}")
    print(f"Archs      : {matmul_archs} (+ simple_vector for softmax)")
    print(f"PE scales  : {pe_sc}")
    print(f"GLB scales : {glb_sc}")
    print(f"Batch sizes: {batch_sz}")
    print(f"TP degrees : {tp_list}")
    print(f"Total configs: {len(configs)}")

    if args.dry_run:
        est = len(configs) / max(args.n_jobs, 1) * 10
        print(f"Estimated time: ~{est/60:.1f} min with {args.n_jobs} jobs")
        return

    print(f"\nRunning with {args.n_jobs} parallel jobs ...")
    t0 = time.time()

    rows = joblib.Parallel(n_jobs=args.n_jobs, verbose=5)(
        joblib.delayed(run_one_config)(
            c["model"], c["stem"], c["yaml_path"], c["arch"],
            c["pe_x"], c["pe_y"], c["glb_scale"], c["batch_size"], c["tp_degree"],
            c["dram_config"], c["mapper_idx"], c["output_base_dir"], remove_bw_limit,
        )
        for c in configs
    )

    elapsed = time.time() - t0
    ok_rows = [r for r in rows if r is not None]
    n_fail  = len(rows) - len(ok_rows)
    print(f"\nDone in {elapsed/60:.1f} min.  OK: {len(ok_rows)}/{len(rows)}  "
          f"FAILED: {n_fail}")

    if ok_rows:
        write_csv(ok_rows, args.output_csv)

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
