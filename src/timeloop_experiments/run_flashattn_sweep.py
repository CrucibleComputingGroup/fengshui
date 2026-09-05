#!/usr/bin/env python3
"""
Sweep Flash Attention (FuseMax) across arch configs × DRAM combinations.

Strategy: for each (model, arch, PE, GLB, TP) config, run FuseMax ONCE with
infinite DRAM bandwidth, then post-process to derive latency and energy for
every (dram_i, dram_o) combination — identical in spirit to postprocess_bw.py
but integrated into the FuseMax evaluation pipeline.

Usage (inside Docker my_timeloop_env_v2):
  export LD_LIBRARY_PATH=/workspace/accelergy-timeloop-infrastructure/src/timeloop/lib:$LD_LIBRARY_PATH

  # Dry run
  python3 run_flashattn_sweep.py --dry-run

  # Full sweep
  python3 run_flashattn_sweep.py --output-csv flashattn_dram_sweep.csv --n-jobs 60

  # Single model, single DRAM pair
  python3 run_flashattn_sweep.py --models llama --drams HBM3,GDDR7
"""
import sys
import os
import argparse
import csv
import time
import types
import traceback
import math
from pathlib import Path

import joblib

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))

# Add scripts/ so global_parameter can be imported
for _candidate in [os.path.join(_PROJECT_ROOT, "scripts"),
                   _PROJECT_ROOT]:
    if (os.path.isfile(os.path.join(_candidate, "global_parameter.py"))
            and _candidate not in sys.path):
        sys.path.insert(0, _candidate)

# Make 'src' → fusemax package so Proposal's internal imports resolve
_FUSEMAX_DIR = os.path.join(_PROJECT_ROOT, "flashattn", "fusemax")
src_mod = types.ModuleType("src")
src_mod.__path__ = [_FUSEMAX_DIR]
sys.modules["src"] = src_mod
if _FUSEMAX_DIR not in sys.path:
    sys.path.insert(0, _FUSEMAX_DIR)

from global_parameter import (
    prefill_seq_lens,
    decode_kv_lens,
    arch_targets,
    pe_scales,
    glb_scales,
    tp_degrees,
    ARCH_BASE_PE,
    dram_type_bandwidth_width_dict,
    word_size,
    timeloop_timeout,
    timeloop_victory_condition,
    timeloop_num_threads,
    timeloop_search_strategy,
    dram_options,
    cycle_time,
)

import pytimeloop.timeloopfe.v4 as tl

# ============================================================
# Constants
# ============================================================
LLAMA_MODELS = ["llama3.1_8b", "llama3.1_70b"]
QWEN_MODELS = ["qwen3_30b_a3b", "qwen3_235b_a22b"]

PREFILL_SEQS = prefill_seq_lens
DECODE_KVS = decode_kv_lens
ARCHS = arch_targets
PE_SCALES = pe_scales
GLB_SCALES = glb_scales
PE_COMBOS = [(px, py) for px in PE_SCALES for py in PE_SCALES]
TP_DEGREES = tp_degrees

# Default DRAM type used during the Timeloop mapper run (infinite BW)
SOURCE_DRAM = 'LPDDR5'

WORKLOAD_DIR = os.path.normpath(os.path.join(_PROJECT_ROOT, "workloads"))
if not os.path.isdir(WORKLOAD_DIR):
    WORKLOAD_DIR = "/workspace/chiplet_timeloop/workloads"

ARCH_BASE = Path(_PROJECT_ROOT) / "arch"
TOP_JINJA = ARCH_BASE / "top.yaml.jinja2"


# ============================================================
# Helpers
# ============================================================

def get_flashattn_dir(model, phase, length):
    if phase == "prefill":
        wl = f"{model}_prefill_s{length}"
    else:
        wl = f"{model}_decode_kv{length}"
    fa_dir = os.path.join(WORKLOAD_DIR, wl, "flashattn")
    if os.path.isdir(fa_dir):
        return wl, fa_dir
    return wl, None


def _ensure_src_module():
    """Ensure 'src' module alias exists in this worker process."""
    if "src" not in sys.modules:
        _proj = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".."))
        _fm = os.path.join(_proj, "flashattn", "fusemax")
        src = types.ModuleType("src")
        src.__path__ = [_fm]
        sys.modules["src"] = src
        if _fm not in sys.path:
            sys.path.insert(0, _fm)


# ============================================================
# Run one config (once with infinite BW) + post-process DRAMs
# ============================================================

def run_one_config(flashattn_dir, workload_name, arch, pe_x_scale, pe_y_scale,
                   glb_scale, output_base_dir, target_drams,
                   mapper_timeout=300, mapper_victory=100,
                   tp_degree=1, tp_dim="H"):
    """Run one FuseMax config with infinite BW, then post-process for all DRAM combos.

    Returns a list of result dicts (one per dram_i × dram_o pair), or None on failure.
    """
    _ensure_src_module()

    from accel.proposal import Proposal, ARCH_MESH
    from accel.constraints import make_flashattn_callback

    try:
        flashattn_path = Path(flashattn_dir)

        # Compute scaled PE dimensions
        bx, by = ARCH_BASE_PE.get(arch, (64, 64))
        pe_x = round(bx * pe_x_scale)
        pe_y = round(by * pe_y_scale)
        PE_dim = pe_x

        # Compute scaled GLB size
        base_l3_bytes = 16 * 2**20  # 16 MB (Proposal default)
        l3_sz = round(base_l3_bytes * glb_scale)

        p = Proposal(
            flashattn_path,
            arch_2d=arch,
            arch_1d="simple_vector",
            PE_dim=PE_dim,
            l3_sz=l3_sz,
            mode="mapper",
            tp_degree=tp_degree,
            tp_dim=tp_dim,
        )
        p._mapper_timeout = mapper_timeout
        p._mapper_victory = mapper_victory

        # Override mesh dims for constraint callback
        p.mesh_x = pe_x
        p.mesh_y = pe_y
        p._constraint_cb = make_flashattn_callback(
            arch_2d=arch,
            arch_1d="simple_vector",
            einsums_2d=set(p.einsums_2d),
            pe_x=pe_x,
            pe_y=pe_y,
        )

        base_constraint_cb = p._constraint_cb
        einsums_2d_set = set(p.einsums_2d)

        def spec_callback(spec, einsum):
            is_2d = einsum in einsums_2d_set or einsum == "2d"

            # Scale GLB
            shared_glb = spec.architecture.find("shared_glb")
            shared_glb.attributes["depth"] = round(
                shared_glb.attributes["depth"] * glb_scale)
            if glb_scale != 1.0:
                if "n_banks" in shared_glb.attributes:
                    shared_glb.attributes["n_banks"] = round(
                        shared_glb.attributes["n_banks"] * glb_scale)
                if "shared_bandwidth" in shared_glb.attributes:
                    shared_glb.attributes["shared_bandwidth"] = round(
                        shared_glb.attributes["shared_bandwidth"] * glb_scale)

            # Scale PE array
            if is_2d:
                if arch in ("eyeriss_like", "gemmini_like"):
                    spec.architecture.find("PE_column").spatial.meshX = pe_x
                    spec.architecture.find("PE").spatial.meshY = pe_y
                elif arch == "simba_like":
                    spec.architecture.find("PE").spatial.meshX = pe_x
                    spec.architecture.find("distributed_buffers").spatial.meshY = pe_y
            else:
                # 1D einsums use simple_vector
                spec.architecture.find("PE_column").spatial.meshX = pe_x

            # DRAM: infinite BW for mapper run (source DRAM type for energy)
            bw_multiplier = 1000
            for dram_index in ["I", "O"]:
                dram = spec.architecture.find(f"DRAM_{dram_index}")
                dram_info = dram_type_bandwidth_width_dict[SOURCE_DRAM]
                dram.attributes["type"] = dram_info["timeloop"]
                dram.attributes["width"] = dram_info["width"]
                dram.attributes["shared_bandwidth"] = (
                    dram_info["bandwidth"] * bw_multiplier * 8 / word_size
                    / tp_degree
                )

            # Mapper settings
            spec.mapper["algorithm"] = timeloop_search_strategy
            spec.mapper["timeout"] = mapper_timeout
            spec.mapper["victory_condition"] = mapper_victory
            spec.mapper["num_threads"] = timeloop_num_threads
            spec.mapper["diagnostics"] = False
            spec.mapper["optimization_metrics"] = ["energy", "delay"]

            # Apply FuseMax constraints (skip for Accelergy area/energy
            # estimation which passes "2d"/"1d" instead of einsum names)
            if einsum not in ("2d", "1d"):
                base_constraint_cb(spec, einsum)

        # Output directory
        config_tag = (
            f"arch={arch}@glb_scale={glb_scale}"
            f"@pe_x_scale={pe_x_scale}@pe_y_scale={pe_y_scale}"
            f"@tp={tp_degree}_{tp_dim}"
        )
        output_dir = Path(output_base_dir) / "flashattn_outputs" / workload_name / config_tag
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Run once with infinite BW ---
        traffic, latency = p.eval(output_dir, spec_callback=spec_callback)

        # Collect energy (Accelergy-based)
        energy_pj = p.eval_energy(output_dir, spec_callback=spec_callback)
        p.computed["dynamic_energy"] = energy_pj

        # Compute component latencies for output
        try:
            _, _, comp_2d_lat, comp_1d_lat = p.eval_components(output_dir)
        except Exception:
            comp_2d_lat = comp_1d_lat = 0

        model = (workload_name.split("_prefill_")[0] if "_prefill_" in workload_name
                 else workload_name.split("_decode_")[0])

        # Collect per-einsum details (for reference)
        per_einsum = {}
        for name, (t, ml, cl) in p.computed.get("results", {}).items():
            per_einsum[name] = {"traffic": t, "mem_lat": ml, "comp_lat": cl}

        # --- Post-process for each DRAM combo ---
        rows = []
        for di in target_drams:
            for do in target_drams:
                pp = p.postprocess_dram(di, do, source_dram=SOURCE_DRAM)

                rows.append({
                    "model": model,
                    "workload": workload_name,
                    "arch": arch,
                    "pe_x_scale": pe_x_scale,
                    "pe_y_scale": pe_y_scale,
                    "pe_scale": f"{pe_x_scale}x{pe_y_scale}",
                    "glb_scale": glb_scale,
                    "tp_degree": tp_degree,
                    "tp_dim": tp_dim,
                    "pe_x": pe_x,
                    "pe_y": pe_y,
                    "PE_dim": PE_dim,
                    "P1": p.P1,
                    "R": p.transformer.R,
                    "T": p.transformer.T,
                    "B_eff": p.transformer.B,
                    "H_eff": p.transformer.H,
                    "dram_i": di,
                    "dram_o": do,
                    "latency_s": pp['latency'],
                    "comp_latency_s": pp['comp_latency'],
                    "dram_I_latency_s": pp['dram_I_latency'],
                    "dram_O_latency_s": pp['dram_O_latency'],
                    "dynamic_energy_pj": pp['dynamic_energy_pj'],
                    "dram_I_accesses": pp['dram_I_accesses'],
                    "dram_O_accesses": pp['dram_O_accesses'],
                    "bw_throttled": pp['bw_throttled'],
                    "traffic": traffic,
                    "comp_2d_lat": comp_2d_lat,
                    "comp_1d_lat": comp_1d_lat,
                    # Per-einsum traffic breakdown
                    "qk_traffic": per_einsum.get("QK", {}).get("traffic", 0),
                    "slnv_traffic": per_einsum.get("SLNV", {}).get("traffic", 0),
                    "av_traffic": per_einsum.get("AV", {}).get("traffic", 0),
                    # Infinite-BW baseline for reference
                    "inf_bw_latency_s": latency,
                    "inf_bw_energy_pj": energy_pj,
                })

        return rows

    except Exception as e:
        print(f"FAIL {workload_name} {arch} pe={pe_x_scale}x{pe_y_scale} "
              f"glb={glb_scale} tp={tp_degree}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


# ============================================================
# Config builder
# ============================================================

def build_configs(models, run_id=None, total_runs=None, tp_dims=None,
                  arch_filter=None):
    """Build all (flashattn_dir, workload_name, arch, pe_x, pe_y, glb, tp, tp_dim) tuples."""
    if tp_dims is None:
        tp_dims = ["H"]
    archs = [a for a in ARCHS if a in arch_filter] if arch_filter else ARCHS
    configs = []

    for model in models:
        for phase in ["prefill", "decode"]:
            lengths = PREFILL_SEQS if phase == "prefill" else DECODE_KVS
            for length in lengths:
                wl, fa_dir = get_flashattn_dir(model, phase, length)
                if fa_dir is None:
                    continue
                for arch in archs:
                    for glb in GLB_SCALES:
                        for (px, py) in PE_COMBOS:
                            for tp in TP_DEGREES:
                                for td in tp_dims:
                                    configs.append((fa_dir, wl, arch, px, py,
                                                    glb, tp, td))

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
    "model", "workload", "arch",
    "pe_x_scale", "pe_y_scale", "pe_scale", "glb_scale",
    "tp_degree", "tp_dim",
    "pe_x", "pe_y", "PE_dim", "P1", "R", "T",
    "B_eff", "H_eff",
    "dram_i", "dram_o",
    "latency_s", "comp_latency_s", "dram_I_latency_s", "dram_O_latency_s",
    "dynamic_energy_pj",
    "dram_I_accesses", "dram_O_accesses",
    "bw_throttled",
    "traffic", "comp_2d_lat", "comp_1d_lat",
    "qk_traffic", "slnv_traffic", "av_traffic",
    "inf_bw_latency_s", "inf_bw_energy_pj",
]


def write_csv(results, output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print("Wrote %d rows to %s" % (len(results), output_csv))


# ---- run_sweep-compatible CSV output ----

COMPAT_FIELDNAMES = [
    "model", "workload", "operator", "batch_size", "tp_degree",
    "arch", "pe_x_scale", "pe_y_scale", "pe_scale", "glb_scale",
    "dram_I", "dram_O", "mapper_objective",
    "utilization", "energy_uj", "cycles", "latency_s",
    "fj_per_compute", "gflops", "area_mm2",
]


def _to_compat_row(row):
    """Convert a flashattn result row to run_sweep-compatible format."""
    return {
        "model": row["model"],
        "workload": row["workload"],
        "operator": "flashattn",
        "batch_size": row.get("B_eff", 1),
        "tp_degree": row["tp_degree"],
        "arch": row["arch"],
        "pe_x_scale": row["pe_x_scale"],
        "pe_y_scale": row["pe_y_scale"],
        "pe_scale": row["pe_scale"],
        "glb_scale": row["glb_scale"],
        "dram_I": row["dram_i"],
        "dram_O": row["dram_o"],
        "mapper_objective": "energy",
        "utilization": 0.0,
        "energy_uj": row["dynamic_energy_pj"] / 1e6,
        "cycles": round(row["latency_s"] / cycle_time),
        "latency_s": row["latency_s"],
        "fj_per_compute": 0.0,
        "gflops": 0.0,
        "area_mm2": 0.0,
    }


def write_compat_csv(results, output_csv):
    """Write results in run_sweep.py-compatible format."""
    compat_path = output_csv.replace(".csv", "_compat.csv")
    compat_rows = [_to_compat_row(r) for r in results]
    with open(compat_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPAT_FIELDNAMES,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(compat_rows)
    print("Wrote %d rows to %s (run_sweep-compatible)" % (
        len(compat_rows), compat_path))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sweep FlashAttn (FuseMax) across all configs: "
                    "run once per arch, post-process across DRAM combos")

    parser.add_argument("--models", type=str, default="all",
                        choices=["llama", "qwen", "all"],
                        help="Model family (default: all)")
    parser.add_argument("--output-csv", type=str,
                        default="flashattn_dram_sweep.csv",
                        help="Output CSV (default: flashattn_dram_sweep.csv)")
    parser.add_argument("--output-base-dir", type=str, default=_THIS_DIR,
                        help="Base dir for Timeloop outputs (default: script dir)")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Shard index for distributed execution")
    parser.add_argument("--total-runs", type=int, default=None,
                        help="Total shards for distributed execution")
    parser.add_argument("--n-jobs", type=int, default=60,
                        help="Parallel jobs (default: 60)")
    parser.add_argument("--mapper-timeout", type=int, default=300,
                        help="Mapper timeout per einsum in seconds (default: 300)")
    parser.add_argument("--mapper-victory", type=int, default=100,
                        help="Mapper victory condition (default: 100)")
    parser.add_argument("--tp-dims", type=str, default="H",
                        help="TP dimension(s) to sweep, comma-separated "
                             "(H, B, BH; default: H)")
    parser.add_argument("--drams", type=str,
                        default=",".join(dram_options),
                        help="Comma-separated DRAM types to sweep "
                             f"(default: {','.join(dram_options)})")
    parser.add_argument("--archs", type=str, default=None,
                        help="Comma-separated architectures to sweep "
                             "(e.g. simba_like,eyeriss_like; default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count configs without running")

    args = parser.parse_args()

    # Resolve models
    if args.models == "llama":
        models = LLAMA_MODELS
    elif args.models == "qwen":
        models = QWEN_MODELS
    else:
        models = LLAMA_MODELS + QWEN_MODELS

    target_drams = [d.strip() for d in args.drams.split(",")]
    tp_dims = [d.strip().upper() for d in args.tp_dims.split(",")]
    arch_filter = ([a.strip() for a in args.archs.split(",")]
                   if args.archs else None)
    configs = build_configs(models, args.run_id, args.total_runs,
                            tp_dims=tp_dims, arch_filter=arch_filter)

    n_dram_combos = len(target_drams) ** 2

    print("=" * 60)
    print("  FlashAttn (FuseMax) DRAM Sweep")
    print("=" * 60)
    print(f"  Models:       {models}")
    print(f"  Archs:        {ARCHS}")
    print(f"  PE combos ({len(PE_COMBOS)}): {PE_COMBOS}")
    print(f"  GLB scales:   {GLB_SCALES}")
    print(f"  TP degrees:   {TP_DEGREES}")
    print(f"  TP dims:      {tp_dims}")
    print(f"  Target DRAMs: {target_drams}")
    print(f"  DRAM combos:  {n_dram_combos} (per config)")
    print(f"  Workloads: {len(PREFILL_SEQS)} prefill + {len(DECODE_KVS)} decode = "
          f"{len(PREFILL_SEQS) + len(DECODE_KVS)} per model")
    print(f"  Arch configs: {len(configs)} (each runs Timeloop ONCE)")
    print(f"  Output rows:  {len(configs) * n_dram_combos}")
    print(f"  Each config runs 12 einsums via Proposal.eval()")

    if args.dry_run:
        est = len(configs) * 360 / max(args.n_jobs, 1)
        print(f"  Estimated time: {est/3600:.1f} hours = {est/86400:.1f} days "
              f"(at {args.n_jobs} parallel jobs)")
        return

    print(f"\nRunning with {args.n_jobs} parallel jobs...")
    t0 = time.time()

    raw_results = joblib.Parallel(n_jobs=args.n_jobs, verbose=10)(
        joblib.delayed(run_one_config)(
            fa_dir, wl, arch, px, py, glb,
            args.output_base_dir,
            target_drams,
            mapper_timeout=args.mapper_timeout,
            mapper_victory=args.mapper_victory,
            tp_degree=tp,
            tp_dim=td,
        )
        for fa_dir, wl, arch, px, py, glb, tp, td in configs
    )

    elapsed = time.time() - t0

    # Flatten: each successful result is a list of dicts
    all_rows = []
    ok_count = 0
    for result in raw_results:
        if result is not None:
            ok_count += 1
            all_rows.extend(result)

    print(f"\nDone in {elapsed/3600:.1f} hours. "
          f"{ok_count}/{len(configs)} configs succeeded → "
          f"{len(all_rows)} rows.")

    if all_rows:
        write_csv(all_rows, args.output_csv)
        write_compat_csv(all_rows, args.output_csv)


if __name__ == "__main__":
    main()
