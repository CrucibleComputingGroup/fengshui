#!/usr/bin/env python3
"""
Experiment: Sweep DRAM width and measure its impact on latency, utilization, and energy.

Keeps everything else constant (eyeriss_like arch, 64x64 PEs, LPDDR5 type, q_proj workload).
Only varies the DRAM "width" attribute: [16, 32, 64, 128, 256, 512, 1024].

Usage (inside Docker with timeloop):
  python3 sweep_dram_width.py
"""

import os
import sys
import re
import json
import csv

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_THIS_DIR, "..", "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pytimeloop.timeloopfe.v4 as tl
from pytimeloop.timeloopfe.v4.constraints import ProblemDataspaceList

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.normpath(os.path.join(_THIS_DIR, ".."))
ARCH_DIR = os.path.join(PROJECT_DIR, "arch")
TOP_JINJA_PATH = os.path.join(ARCH_DIR, "top.yaml.jinja2")
WORKLOAD_DIR = os.path.join(PROJECT_DIR, "workloads", "llama3.1_8b_prefill_s512")
PROBLEM_YAML = os.path.join(WORKLOAD_DIR, "layer0_q_proj.yaml")

OUTPUT_BASE = os.path.join(_THIS_DIR, "dram_width_sweep_results")

# Sweep parameter
DRAM_WIDTHS = [16, 32, 64, 128, 256, 512, 1024]

# Fixed parameters
ARCH_TARGET = "eyeriss_like"
DRAM_TYPE = '"LPDDR5"'
DRAM_BW_UNLIMITED = 70.4 * 1000 * 8 / 16  # remove_bw_limit=True
DRAM_BW_REALISTIC = 70.4 * 8 / 16          # realistic LPDDR5 BW
MAPPER_TIMEOUT = 120
MAPPER_VICTORY = 100
MAPPER_THREADS = 4


def parse_stats(stats_path):
    """Parse timeloop-mapper.stats.txt for key metrics."""
    with open(stats_path, "r") as f:
        content = f.read()

    # Cycles
    cycles_m = re.search(r"Cycles:\s*(\d+)", content)
    cycles = int(cycles_m.group(1)) if cycles_m else 0

    # Utilization
    util_m = re.search(r"Utilization:\s*([\d.]+)%", content)
    utilization = float(util_m.group(1)) if util_m else 0.0

    # Total energy (sum all "Energy (total)" lines)
    energy_total = 0.0
    for m in re.finditer(r"Energy \(total\)\s*:\s*([\d.]+)\s*pJ", content):
        energy_total += float(m.group(1))

    # Leakage energy
    leakage_total = 0.0
    for m in re.finditer(r"Leakage energy \(total\)\s*:\s*([\d.]+)\s*pJ", content):
        leakage_total += float(m.group(1))

    # fJ/Compute breakdown
    fj_breakdown = {}
    bd_section = re.search(r"fJ/Compute\s*(.*?)$", content, re.DOTALL)
    if bd_section:
        for line in bd_section.group(1).split("\n"):
            if "=" in line and "Total" not in line:
                k, v = line.split("=")
                fj_breakdown[k.strip()] = float(v.strip())
    fj_total = sum(fj_breakdown.values())

    return {
        "cycles": cycles,
        "utilization": utilization,
        "energy_pJ": energy_total + leakage_total,
        "dynamic_energy_pJ": energy_total,
        "leakage_energy_pJ": leakage_total,
        "fj_per_compute": fj_total,
        "fj_breakdown": fj_breakdown,
    }


def run_one_width(width, use_realistic_bw=False):
    """Run timeloop-mapper with a specific DRAM width. Returns metrics dict."""
    bw_tag = "realistic" if use_realistic_bw else "unlimited"
    output_dir = os.path.join(OUTPUT_BASE, f"width_{width}_{bw_tag}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Running DRAM width = {width} (BW: {bw_tag})")
    print(f"{'='*60}")

    # Load spec from Jinja template
    spec = tl.Specification.from_yaml_files(
        TOP_JINJA_PATH,
        jinja_parse_data={
            "architecture": ARCH_TARGET,
            "use_temp_arch": False,
            "use_bf_arch": True,
            "problem": PROBLEM_YAML,
            "mapping_path": False,
        },
    )

    # --- Modify DRAM width ---
    for dram_name in ["DRAM_I", "DRAM_O"]:
        dram = spec.architecture.find(dram_name)
        dram.attributes["type"] = DRAM_TYPE
        dram.attributes["width"] = width
        dram.attributes["datawidth"] = 16
        dram.attributes["shared_bandwidth"] = (
            DRAM_BW_REALISTIC if use_realistic_bw else DRAM_BW_UNLIMITED
        )

    # --- DRAM dataspace constraints (standard projection) ---
    ds_w, ds_a, ds_o = "Inputs1", "Inputs2", "Outputs"
    spec.architecture.find("DRAM_I")["constraints"]["dataspace"]["keep"] = (
        ProblemDataspaceList([ds_a, ds_w])
    )
    spec.architecture.find("DRAM_I")["constraints"]["dataspace"]["bypass"] = (
        ProblemDataspaceList([ds_o])
    )
    spec.architecture.find("DRAM_O")["constraints"]["dataspace"]["keep"] = (
        ProblemDataspaceList([ds_o])
    )
    spec.architecture.find("DRAM_O")["constraints"]["dataspace"]["bypass"] = (
        ProblemDataspaceList([ds_w, ds_a])
    )
    spec.architecture.find("DRAM_Backup")["constraints"]["dataspace"]["keep"] = (
        ProblemDataspaceList([])
    )
    spec.architecture.find("DRAM_Backup")["constraints"]["dataspace"]["bypass"] = (
        ProblemDataspaceList([ds_w, ds_a, ds_o])
    )

    # --- Apply eyeriss_like projection constraints ---
    # Use the same approach as timeloop_helper.py: modify existing constraint
    # objects' keep/bypass fields rather than replacing with raw dicts.
    from pytimeloop.timeloopfe.v4.constraints import Factors, Permutation

    pe_x, pe_y = 64, 64
    M, C = 4096, 4096  # from q_proj problem
    m_spatial = min(M, pe_x)
    c_spatial = min(C, pe_y)
    dim_order = ["B", "N", "C", "M"]
    all_ones = [f"{d}=1" for d in dim_order]

    # Spatial: PE_column tiles M, PE tiles C
    pe_col = spec.architecture.find("PE_column")
    pe_col["constraints"]["spatial"]["factors"] = Factors(
        [f"M={m_spatial}"] + [f"{d}=1" for d in dim_order if d != "M"]
    )
    pe_col["constraints"]["spatial"]["split"] = 999
    pe_col["constraints"]["spatial"]["permutation"] = Permutation(dim_order)

    pe = spec.architecture.find("PE")
    pe["constraints"]["spatial"]["factors"] = Factors(
        [f"C={c_spatial}"] + [f"{d}=1" for d in dim_order if d != "C"]
    )
    pe["constraints"]["spatial"]["split"] = 0
    pe["constraints"]["spatial"]["permutation"] = Permutation(dim_order)

    # shared_glb: keep Inputs2+Outputs, bypass Inputs1(weights)
    glb = spec.architecture.find("shared_glb")
    glb["constraints"]["dataspace"]["keep"] = ProblemDataspaceList([ds_a, ds_o])
    glb["constraints"]["dataspace"]["bypass"] = ProblemDataspaceList([ds_w])

    # ifmap_spad: all factors = 1
    ifmap = spec.architecture.find("ifmap_spad")
    ifmap["constraints"]["dataspace"]["keep"] = ProblemDataspaceList([ds_a])
    ifmap["constraints"]["dataspace"]["bypass"] = ProblemDataspaceList([ds_w, ds_o])
    ifmap["constraints"]["temporal"]["factors"] = Factors(all_ones)
    ifmap["constraints"]["temporal"]["permutation"] = Permutation(dim_order)

    # weights_spad: allow C,M temporal
    wspad = spec.architecture.find("weights_spad")
    wspad["constraints"]["dataspace"]["keep"] = ProblemDataspaceList([ds_w])
    wspad["constraints"]["dataspace"]["bypass"] = ProblemDataspaceList([ds_a, ds_o])
    wspad["constraints"]["temporal"]["factors"] = Factors(["B=1", "N=1"])
    wspad["constraints"]["temporal"]["permutation"] = Permutation(dim_order)

    # psum_spad: keep Outputs
    pspad = spec.architecture.find("psum_spad")
    pspad["constraints"]["dataspace"]["keep"] = ProblemDataspaceList([ds_o])
    pspad["constraints"]["dataspace"]["bypass"] = ProblemDataspaceList([ds_w, ds_a])

    # --- Mapper config ---
    mapper = spec.mapper
    mapper["algorithm"] = "random"
    mapper["timeout"] = MAPPER_TIMEOUT
    mapper["victory_condition"] = MAPPER_VICTORY
    mapper["num_threads"] = MAPPER_THREADS
    mapper["optimization_metrics"] = ["energy", "delay"]
    mapper["diagnostics"] = False

    # --- Run ---
    try:
        res = tl.call_mapper(spec, output_dir=output_dir, dump_intermediate_to=output_dir)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    # --- Parse results ---
    stats_path = os.path.join(output_dir, "timeloop-mapper.stats.txt")
    if not os.path.exists(stats_path):
        print(f"  ERROR: stats file not found at {stats_path}")
        return None

    metrics = parse_stats(stats_path)
    metrics["width"] = width
    latency_s = metrics["cycles"] * 1e-9  # 1 GHz
    metrics["latency_s"] = latency_s

    print(f"  Width={width}: cycles={metrics['cycles']}, "
          f"util={metrics['utilization']:.2f}%, "
          f"energy={metrics['energy_pJ']:.2e} pJ, "
          f"latency={latency_s:.6e} s")

    return metrics


def print_and_save(results, label, csv_suffix):
    """Print summary table and save CSV/JSON."""
    if not results:
        print(f"\nERROR: No successful runs for {label}!")
        return

    csv_path = os.path.join(OUTPUT_BASE, f"dram_width_sweep_{csv_suffix}.csv")
    fieldnames = ["width", "cycles", "latency_s", "utilization",
                  "energy_pJ", "dynamic_energy_pJ", "leakage_energy_pJ",
                  "fj_per_compute"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nCSV saved to: {csv_path}")

    print(f"\n{'='*90}")
    print(f"  {label}")
    print(f"  (eyeriss_like, q_proj B=1 N=512 C=4096 M=4096)")
    print(f"{'='*90}")
    print(f"{'Width':>8} {'Cycles':>12} {'Latency(s)':>14} {'Util%':>8} "
          f"{'Energy(pJ)':>14} {'fJ/Compute':>12}")
    print("-" * 90)
    for r in results:
        print(f"{r['width']:>8} {r['cycles']:>12,} {r['latency_s']:>14.6e} "
              f"{r['utilization']:>8.2f} {r['energy_pJ']:>14.2e} "
              f"{r['fj_per_compute']:>12.2f}")

    json_path = os.path.join(OUTPUT_BASE, f"dram_width_sweep_{csv_suffix}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # --- Experiment 1: Unlimited BW (isolate energy effect) ---
    print("\n" + "#" * 70)
    print("# Experiment 1: Unlimited DRAM Bandwidth (isolate energy effect)")
    print("#" * 70)
    results_unlimited = []
    for width in DRAM_WIDTHS:
        m = run_one_width(width, use_realistic_bw=False)
        if m:
            results_unlimited.append(m)
    print_and_save(results_unlimited, "DRAM Width Sweep — Unlimited BW", "unlimited_bw")

    # --- Experiment 2: Realistic BW (see latency/util impact) ---
    print("\n" + "#" * 70)
    print("# Experiment 2: Realistic DRAM Bandwidth (LPDDR5 70.4 GB/s)")
    print("#" * 70)
    results_realistic = []
    for width in DRAM_WIDTHS:
        m = run_one_width(width, use_realistic_bw=True)
        if m:
            results_realistic.append(m)
    print_and_save(results_realistic, "DRAM Width Sweep — Realistic BW (LPDDR5)", "realistic_bw")


if __name__ == "__main__":
    main()
