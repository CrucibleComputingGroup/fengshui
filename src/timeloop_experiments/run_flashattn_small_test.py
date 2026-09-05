#!/usr/bin/env python3
"""
Small-scale test of FlashAttn DRAM sweep.

Covers:
  - 2 workloads: llama3.1_8b prefill_s512 (compute-heavy) + decode_kv512 (memory-heavy)
  - 3 architectures: eyeriss_like, simba_like, gemmini_like
  - 1 PE scale (1x1), 1 GLB scale (1), tp=1
  - 4 DRAM types: LPDDR5, DDR5, GDDR7, HBM3
  → 6 Timeloop runs × 16 DRAM combos = 96 output rows
"""
import sys
import os
import csv
import time
import types
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup (same as run_flashattn_sweep.py)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))

for _candidate in [os.path.join(_PROJECT_ROOT, "scripts"), _PROJECT_ROOT]:
    if (os.path.isfile(os.path.join(_candidate, "global_parameter.py"))
            and _candidate not in sys.path):
        sys.path.insert(0, _candidate)

_FUSEMAX_DIR = os.path.join(_PROJECT_ROOT, "flashattn", "fusemax")
src_mod = types.ModuleType("src")
src_mod.__path__ = [_FUSEMAX_DIR]
sys.modules["src"] = src_mod
if _FUSEMAX_DIR not in sys.path:
    sys.path.insert(0, _FUSEMAX_DIR)

from global_parameter import (
    ARCH_BASE_PE,
    dram_type_bandwidth_width_dict,
    word_size,
    dram_options,
)

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------
WORKLOAD_DIR = os.path.normpath(os.path.join(_PROJECT_ROOT, "workloads"))
if not os.path.isdir(WORKLOAD_DIR):
    WORKLOAD_DIR = "/workspace/chiplet_timeloop/workloads"

TEST_WORKLOADS = [
    ("llama3.1_8b_prefill_s512", "prefill"),
    ("llama3.1_8b_decode_kv512", "decode"),
]
TEST_ARCHS = ["eyeriss_like", "gemmini_like"]  # simba_like needs pe_y != pe_x handling
TARGET_DRAMS = dram_options  # ['LPDDR5', 'DDR5', 'GDDR7', 'HBM3']
SOURCE_DRAM = 'LPDDR5'

PE_X_SCALE = 1
PE_Y_SCALE = 1
GLB_SCALE = 1
TP_DEGREE = 1
TP_DIM = "H"
MAPPER_TIMEOUT = 120
MAPPER_VICTORY = 50


def run_one(wl_name, arch):
    from accel.proposal import Proposal, ARCH_MESH
    from accel.constraints import make_flashattn_callback

    fa_dir = os.path.join(WORKLOAD_DIR, wl_name, "flashattn")
    if not os.path.isdir(fa_dir):
        print(f"  SKIP {wl_name}: flashattn dir not found")
        return None

    bx, by = ARCH_BASE_PE.get(arch, (64, 64))
    pe_x = round(bx * PE_X_SCALE)
    pe_y = round(by * PE_Y_SCALE)

    p = Proposal(
        Path(fa_dir),
        arch_2d=arch,
        arch_1d="simple_vector",
        PE_dim=pe_x,
        l3_sz=16 * 2**20 * GLB_SCALE,
        mode="mapper",
        tp_degree=TP_DEGREE,
        tp_dim=TP_DIM,
    )
    p._mapper_timeout = MAPPER_TIMEOUT
    p._mapper_victory = MAPPER_VICTORY
    p.mesh_x = pe_x
    p.mesh_y = pe_y
    p._constraint_cb = make_flashattn_callback(
        arch_2d=arch, arch_1d="simple_vector",
        einsums_2d=set(p.einsums_2d), pe_x=pe_x, pe_y=pe_y,
    )

    base_cb = p._constraint_cb
    einsums_2d_set = set(p.einsums_2d)

    def spec_callback(spec, einsum):
        is_2d = einsum in einsums_2d_set

        if is_2d:
            if arch in ("eyeriss_like", "gemmini_like"):
                spec.architecture.find("PE_column").spatial.meshX = pe_x
                spec.architecture.find("PE").spatial.meshY = pe_y
            elif arch == "simba_like":
                spec.architecture.find("PE").spatial.meshX = pe_x
                spec.architecture.find("distributed_buffers").spatial.meshY = pe_y
        else:
            spec.architecture.find("PE_column").spatial.meshX = pe_x

        bw_multiplier = 1000
        for di in ["I", "O"]:
            dram = spec.architecture.find(f"DRAM_{di}")
            dram_info = dram_type_bandwidth_width_dict[SOURCE_DRAM]
            dram.attributes["type"] = dram_info["timeloop"]
            dram.attributes["width"] = dram_info["width"]
            dram.attributes["shared_bandwidth"] = (
                dram_info["bandwidth"] * bw_multiplier * 8 / word_size / TP_DEGREE
            )

        spec.mapper["algorithm"] = "random"
        spec.mapper["timeout"] = MAPPER_TIMEOUT
        spec.mapper["victory_condition"] = MAPPER_VICTORY
        spec.mapper["num_threads"] = 4
        spec.mapper["diagnostics"] = False
        spec.mapper["optimization_metrics"] = ["energy", "delay"]

        # Only apply FuseMax constraints for actual einsums, not for
        # Accelergy area/energy estimation which passes "2d"/"1d"
        if einsum not in ("2d", "1d"):
            base_cb(spec, einsum)

    output_dir = Path(_THIS_DIR) / "flashattn_test_outputs" / wl_name / f"arch={arch}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Running eval() for {wl_name} / {arch}...")
    t0 = time.time()
    traffic, latency = p.eval(output_dir, spec_callback=spec_callback)
    eval_time = time.time() - t0
    print(f"    eval() done in {eval_time:.1f}s  (inf-BW latency={latency*1e6:.2f} us)")

    print(f"  Running eval_energy()...")
    t0 = time.time()
    energy_pj = p.eval_energy(output_dir, spec_callback=spec_callback)
    p.computed["dynamic_energy"] = energy_pj
    print(f"    eval_energy() done in {time.time()-t0:.1f}s  (energy={energy_pj:.0f} pJ)")

    try:
        _, _, comp_2d_lat, comp_1d_lat = p.eval_components(output_dir)
    except Exception:
        comp_2d_lat = comp_1d_lat = 0

    model = wl_name.split("_prefill_")[0] if "_prefill_" in wl_name else wl_name.split("_decode_")[0]

    rows = []
    for di in TARGET_DRAMS:
        for do in TARGET_DRAMS:
            pp = p.postprocess_dram(di, do, source_dram=SOURCE_DRAM)
            rows.append({
                "model": model,
                "workload": wl_name,
                "arch": arch,
                "dram_i": di,
                "dram_o": do,
                "latency_s": pp['latency'],
                "comp_latency_s": pp['comp_latency'],
                "dram_I_latency_s": pp['dram_I_latency'],
                "dram_O_latency_s": pp['dram_O_latency'],
                "dynamic_energy_pj": pp['dynamic_energy_pj'],
                "dram_energy_pj": pp['dram_energy_pj'],
                "dram_I_accesses": pp['dram_I_accesses'],
                "dram_O_accesses": pp['dram_O_accesses'],
                "bw_throttled": pp['bw_throttled'],
                "traffic": traffic,
                "comp_2d_lat": comp_2d_lat,
                "comp_1d_lat": comp_1d_lat,
                "inf_bw_latency_s": latency,
                "inf_bw_energy_pj": energy_pj,
            })

    return rows


def main():
    print("=" * 60)
    print("  FlashAttn DRAM Sweep — Small Test")
    print("=" * 60)
    print(f"  Workloads:  {[w[0] for w in TEST_WORKLOADS]}")
    print(f"  Archs:      {TEST_ARCHS}")
    print(f"  DRAM types: {TARGET_DRAMS}")
    print(f"  Combos:     {len(TEST_WORKLOADS)} × {len(TEST_ARCHS)} = "
          f"{len(TEST_WORKLOADS)*len(TEST_ARCHS)} Timeloop runs")
    print(f"  Output:     × {len(TARGET_DRAMS)**2} DRAM combos = "
          f"{len(TEST_WORKLOADS)*len(TEST_ARCHS)*len(TARGET_DRAMS)**2} rows")
    print()

    all_rows = []
    t_total = time.time()

    for wl_name, phase in TEST_WORKLOADS:
        for arch in TEST_ARCHS:
            print(f"\n[{wl_name}] [{arch}]")
            try:
                rows = run_one(wl_name, arch)
                if rows:
                    all_rows.extend(rows)
                    print(f"  → {len(rows)} rows generated")
            except Exception as e:
                print(f"  FAIL: {e}")
                traceback.print_exc()

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Total: {len(all_rows)} rows in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    if all_rows:
        csv_path = os.path.join(_THIS_DIR, "flashattn_small_test.csv")
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote {len(all_rows)} rows to {csv_path}")

        # Quick analysis
        print(f"\n{'='*60}")
        print("  Quick Analysis")
        print(f"{'='*60}")
        for wl_name, _ in TEST_WORKLOADS:
            wl_rows = [r for r in all_rows if r['workload'] == wl_name]
            if not wl_rows:
                continue
            print(f"\n  [{wl_name}]")
            for arch in TEST_ARCHS:
                arch_rows = [r for r in wl_rows if r['arch'] == arch]
                if not arch_rows:
                    continue
                print(f"    {arch}:")
                print(f"      Inf-BW latency: {arch_rows[0]['inf_bw_latency_s']*1e6:.2f} us")
                print(f"      Compute lat:    {arch_rows[0]['comp_latency_s']*1e6:.2f} us")
                print(f"      DRAM_I accesses: {arch_rows[0]['dram_I_accesses']:.0f}")
                print(f"      DRAM_O accesses: {arch_rows[0]['dram_O_accesses']:.0f}")
                print(f"      {'dram_i':>8s} {'dram_o':>8s} {'latency_us':>12s} {'energy_pJ':>12s} {'throttled':>9s}")
                for r in arch_rows:
                    print(f"      {r['dram_i']:>8s} {r['dram_o']:>8s} "
                          f"{r['latency_s']*1e6:>12.2f} "
                          f"{r['dynamic_energy_pj']:>12.0f} "
                          f"{'YES' if r['bw_throttled'] else 'no':>9s}")


if __name__ == "__main__":
    main()
