#!/usr/bin/env python3
"""
Ablation study for the Fengshui (Mozart) MICRO 2026 rebuttal.

Holds the co-designed N=8 chiplet pool FIXED (the same pool used for Figure 11 /
fig:big_eval) and toggles ONE BASIC-level technique at a time to isolate its
contribution. Uses the precomputed Timeloop database (no Timeloop reruns).

Default configurations (see CONFIGS; more via --configs):
  - Full Fengshui                 : all techniques on
  - w/o Heterogeneous Batching    : v_het_batch=False
  - w/o Heterogeneous Memory      : every buffer forced to GDDR7 (the strongest
                                    single-memory baseline; keeps near-bank PIM
                                    feasible, unlike the homo_hbm variant)
  - w/o Tensor Parallelism (TP=1) : disable tensor/spatial parallel split
  - w/o Layer Fusion              : every op in its own fusion group

Metrics (geomean over networks, each on its objective-matched N=8 pool):
  Energy (energy,F), Energy x Cost (energy,T), EDP (edp,F), EDP x Cost (edp,T)

Run from .../chiplet_timeloop/scripts (so the framework modules import cleanly).
"""
import os
import sys
import csv
import glob
import json
import math
import time
import argparse
import contextlib

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.normpath(os.path.join(THIS_DIR, ".."))
PROJECT_DIR = os.path.normpath(os.path.join(SCRIPTS_DIR, ".."))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

DEFAULT_DB = os.path.join(PROJECT_DIR, "unified_database.csv")
ARCHGYM = os.path.join(PROJECT_DIR, "archgym_results")

# Framework imports (after sys.path setup)
import global_parameter
import genetic_algo_opt_phy_net as ga
import cal_perf_phy_net as cpp
from chiplet_dataclass import ChipletConfig
from baseline import _build_virtual_nets
from chiplet_sel import run_single_optimization
from utility_functions import calculate_average_opt_value


# ---------------------------------------------------------------------------
# Pool loading (fixed N=8 pool from the I-SAEO chain summaries)
# ---------------------------------------------------------------------------
# Each metric uses its objective-matched pool, consistent with Figure 11.
POOL_DIRS = {
    ("energy", False): "v6_energy_chain",
    ("energy", True):  "v6_energy_cost_chain",
    ("edp",    False): "v6_edp_chain",
    ("edp",    True):  "v6_edp_cost_chain",
}


def load_pool(objective, cost_aware, n=8, chain_version="ae"):
    """Load the fixed N=n chiplet pool from the latest chain_summary JSON."""
    dirname = POOL_DIRS[(objective, cost_aware)].replace("v6", chain_version)
    pat = os.path.join(ARCHGYM, dirname, "chain_summary_*.json")
    files = sorted(glob.glob(pat), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"No chain_summary in {dirname} ({pat})")
    summary = json.load(open(files[-1]))
    key = f"n{n}"
    if key not in summary:
        raise KeyError(f"{key} not in {files[-1]} (have {list(summary.keys())})")
    idents = summary[key]["best_chiplets"]
    pool = [ChipletConfig.from_identifier(s) for s in idents]
    return pool, os.path.basename(files[-1]), idents


# ---------------------------------------------------------------------------
# Ablation toggles (monkeypatch context managers; no framework source edits)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def toggle(config):
    """Apply one ablation toggle, restore on exit."""
    saved = {}
    if config == "homo_hbm":
        # Force every per-layer buffer choice to HBM3 in the inner GA.
        # NOTE: PIM requires GDDR7 (cal_perf_phy_net.py:543-549), so HBM3-only
        # makes PIM infeasible -> disables the near-bank decode path.
        saved["ga_dram"] = ga.dram_options
        saved["cpp_dram"] = cpp.dram_options
        ga.dram_options = ["HBM3"]
        cpp.dram_options = ["HBM3"]
    elif config == "homo_gddr7":
        # Control: homogeneous GDDR7. PIM stays feasible (GDDR7), so this
        # isolates "lost PIM" (homo_hbm) from "lost cheap bulk DRAM" (cost).
        saved["ga_dram"] = ga.dram_options
        saved["cpp_dram"] = cpp.dram_options
        ga.dram_options = ["GDDR7"]
        cpp.dram_options = ["GDDR7"]
    elif config == "no_tp":
        # Remove the TP=2 option; only TP=1 is evaluated.
        saved["cpp_tp"] = cpp.tp_degrees
        saved["gp_tp"] = global_parameter.tp_degrees
        cpp.tp_degrees = [1]
        global_parameter.tp_degrees = [1]
    elif config == "no_fusion":
        # Force every gene to all-1s -> no inter-layer fusion.
        saved["harm_dag"] = ga.harmonize_dag_gene
        saved["harm_lin"] = ga.harmonize_gene
        _orig_dag = ga.harmonize_dag_gene
        _orig_lin = ga.harmonize_gene

        def _no_fuse_dag(gene, cp_spec):
            g = _orig_dag(gene, cp_spec)
            g["binary_string"] = "1" * len(g["binary_string"])
            return _orig_dag(g, cp_spec)  # re-harmonize buffers under no-fusion

        def _no_fuse_lin(gene, net_name):
            g = _orig_lin(gene, net_name)
            g["binary_string"] = "1" * len(g["binary_string"])
            return _orig_lin(g, net_name)

        ga.harmonize_dag_gene = _no_fuse_dag
        ga.harmonize_gene = _no_fuse_lin
    elif config in ("full", "no_pim", "homo_batch"):
        pass  # no_pim handled by pool filtering, homo_batch by v_het_batch in run_one
    else:
        raise ValueError(f"unknown config {config}")
    try:
        yield
    finally:
        if config in ("homo_hbm", "homo_gddr7"):
            ga.dram_options = saved["ga_dram"]
            cpp.dram_options = saved["cpp_dram"]
        elif config == "no_tp":
            cpp.tp_degrees = saved["cpp_tp"]
            global_parameter.tp_degrees = saved["gp_tp"]
        elif config == "no_fusion":
            ga.harmonize_dag_gene = saved["harm_dag"]
            ga.harmonize_gene = saved["harm_lin"]


# ---------------------------------------------------------------------------
# Network categorization (for per-category breakdown)
# ---------------------------------------------------------------------------
def categorize(net_unique_name):
    n = net_unique_name.lower()
    if "decode" in n:
        return "decode"
    if "prefill" in n:
        return "prefill"
    if "mobilenet" in n or "replknet" in n or "resnet" in n or "vit" in n:
        return "cnn"
    return "other"


def geomean(values):
    vals = [v for v in values if v is not None and math.isfinite(v) and v > 0]
    if not vals:
        return float("inf")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
METRICS = [
    ("Energy",         "energy", False),
    ("Energy x Cost",  "energy", True),
    ("EDP",            "edp",    False),
    ("EDP x Cost",     "edp",    True),
]
CONFIGS = [
    ("Full Fengshui",            "full"),
    ("w/o Heterogeneous Batching", "homo_batch"),
    ("w/o Heterogeneous Memory", "homo_gddr7"),
    ("w/o Tensor Parallelism",   "no_tp"),
    ("w/o Layer Fusion",         "no_fusion"),
]
# Other toggles remain available via --configs: "w/o Near-bank PIM"/no_pim,
# "homo HBM"/homo_hbm. Kept out of the default set.
_EXTRA_CONFIGS = [
    ("w/o Near-bank PIM",        "no_pim"),
    ("homo HBM",                 "homo_hbm"),
]


def run_one(virtual_nets, pool, objective, cost_aware, config, workers, database):
    """Run run_single_optimization for one (config, metric); return per-net dict.

    NOTE: `results_file` is the precomputed Timeloop performance database that the
    GA reads chiplet/layer stats from (NOT an output file)."""
    eff_pool = pool
    if config == "no_pim":
        # Fair PIM ablation: remove the PIM chiplet from the pool entirely.
        # Decode falls back to conventional off-chip-DRAM compute. No dependence
        # on memory-type data availability (unlike homo_hbm).
        eff_pool = [c for c in pool if c.arch_target != "PIM"]
    # Heterogeneous (non-uniform) batching: off-CP ops use a larger batch for
    # energy amortization. homo_batch disables it -> all ops use the base batch.
    v_het_batch = (config != "homo_batch")
    with toggle(config):
        _, results = run_single_optimization(
            virtual_nets=virtual_nets,
            chiplet_group=eff_pool,
            objective=objective,
            results_file=database,
            cost_aware=cost_aware,
            use_sequential=True,
            n_workers=workers,
            use_dag_cp=True,
            v_het_batch=v_het_batch,
        )
    avg, _ = calculate_average_opt_value(results, objective)
    per_net = {name: r["min_value"] for name, r in results.items()}
    return avg, per_net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", default=DEFAULT_DB)
    # 'ae' = the DETERMINISTIC artifact-evaluation pools (per-network GA reseed),
    # the ONLY chain_summary_*.json pools shipped in src/archgym_results/. They
    # supersede v7/v6 (pre-determinism, not shipped). Matches load_pool()'s own
    # default and ablation_factorial.py, so the heatmap's v2 sanity gate compares
    # like-for-like.
    ap.add_argument("--chain-version", default="ae")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--gen", type=int, default=10)
    ap.add_argument("--smoke", action="store_true",
                    help="2 nets, Energy metric only, validation oracle")
    ap.add_argument("--metrics", default="all",
                    help="comma list among: Energy,Energy x Cost,EDP,EDP x Cost")
    ap.add_argument("--out-prefix", default=None,
                    help="output prefix for <prefix>_{raw,summary}.csv; default "
                         "is a timestamped name so a re-run never clobbers the "
                         "shipped CSVs. Use 'arch_impl/ablation_v2_ae' to "
                         "regenerate the reference that the heatmap generator's "
                         "single-off sanity gate compares against.")
    ap.add_argument("--configs", default="all",
                    help="comma list of config labels to run (default all)")
    args = ap.parse_args()

    global CONFIGS
    if args.configs != "all":
        wanted_c = {c.strip() for c in args.configs.split(",")}
        CONFIGS = [c for c in (CONFIGS + _EXTRA_CONFIGS) if c[0] in wanted_c]

    t0 = time.time()
    print(f"[ablation] building virtual nets from {args.database} ...", flush=True)
    virtual_nets = _build_virtual_nets(database_file=args.database)
    print(f"[ablation] {len(virtual_nets)} virtual nets loaded:", flush=True)
    for vn in virtual_nets:
        print(f"    {vn.get_unique_name()}  ({categorize(vn.get_unique_name())})")

    metrics = METRICS
    if args.smoke:
        virtual_nets = virtual_nets[:2]
        metrics = [METRICS[0]]
        print(f"[ablation] SMOKE: nets={[v.get_unique_name() for v in virtual_nets]}")
    elif args.metrics != "all":
        wanted = {m.strip() for m in args.metrics.split(",")}
        metrics = [m for m in METRICS if m[0] in wanted]

    ts = time.strftime("%Y%m%d_%H%M%S")
    prefix = args.out_prefix or os.path.join(THIS_DIR, f"ablation_{ts}")
    raw_path = prefix + "_raw.csv"
    summ_path = prefix + "_summary.csv"

    raw_rows = []
    # results[metric_label][config_label] = (avg, per_net)
    results = {}

    for metric_label, objective, cost_aware in metrics:
        pool, pool_file, idents = load_pool(objective, cost_aware,
                                            chain_version=args.chain_version)
        print(f"\n=== METRIC: {metric_label}  (objective={objective}, "
              f"cost_aware={cost_aware}) pool={pool_file} ===", flush=True)
        for c in idents:
            print(f"      {c}")
        results[metric_label] = {}
        for config_label, config_key in CONFIGS:
            tcfg = time.time()
            avg, per_net = run_one(virtual_nets, pool, objective, cost_aware,
                                   config_key, args.workers, args.database)
            results[metric_label][config_label] = (avg, per_net)
            dt = time.time() - tcfg
            print(f"  [{config_label:28s}] geomean={avg:.4e}  ({dt:.1f}s)",
                  flush=True)
            for net, val in per_net.items():
                raw_rows.append({
                    "metric": metric_label, "objective": objective,
                    "cost_aware": cost_aware, "config": config_label,
                    "network": net, "category": categorize(net), "value": val,
                })

    # Write raw
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "objective", "cost_aware",
                                          "config", "network", "category", "value"])
        w.writeheader()
        w.writerows(raw_rows)

    # Build summary: per metric, per config, geomean overall + per category,
    # normalized to Full Fengshui = 1.0
    cats = ["decode", "prefill", "cnn"]
    summ_rows = []
    for metric_label, _, _ in metrics:
        full_avg = results[metric_label]["Full Fengshui"][0]
        # per-category full geomeans
        full_per_net = results[metric_label]["Full Fengshui"][1]
        full_cat = {cat: geomean([v for n, v in full_per_net.items()
                                  if categorize(n) == cat]) for cat in cats}
        for config_label, _ in CONFIGS:
            avg, per_net = results[metric_label][config_label]
            row = {
                "metric": metric_label, "config": config_label,
                "geomean": avg,
                "norm_to_full": (avg / full_avg) if (full_avg and math.isfinite(full_avg)) else float("inf"),
            }
            for cat in cats:
                cg = geomean([v for n, v in per_net.items() if categorize(n) == cat])
                row[f"{cat}_norm"] = (cg / full_cat[cat]) if (full_cat[cat] and math.isfinite(full_cat[cat])) else float("inf")
            summ_rows.append(row)

    fieldnames = ["metric", "config", "geomean", "norm_to_full",
                  "decode_norm", "prefill_norm", "cnn_norm"]
    with open(summ_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summ_rows)

    # Pretty-print markdown summary
    print("\n\n================ ABLATION SUMMARY (normalized to Full Fengshui = 1.00; "
          ">1.0 = worse) ================\n")
    for metric_label, _, _ in metrics:
        print(f"### {metric_label}")
        print(f"| Configuration | Overall | Decode | Prefill | CNN |")
        print(f"|---|---|---|---|---|")
        for config_label, _ in CONFIGS:
            r = next(x for x in summ_rows
                     if x["metric"] == metric_label and x["config"] == config_label)
            def fmt(x):
                return f"{x:.2f}x" if math.isfinite(x) else "inf"
            print(f"| {config_label} | {fmt(r['norm_to_full'])} | "
                  f"{fmt(r['decode_norm'])} | {fmt(r['prefill_norm'])} | "
                  f"{fmt(r['cnn_norm'])} |")
        print()

    print(f"\n[ablation] raw     -> {raw_path}")
    print(f"[ablation] summary -> {summ_path}")
    print(f"[ablation] total time {time.time()-t0:.1f}s")

    # Validation oracle (smoke mode)
    if args.smoke:
        full = results["Energy"]["Full Fengshui"][0]
        assert len(pool) == 8, f"expected 8 chiplets, got {len(pool)}"
        assert math.isfinite(full), "Full Fengshui energy not finite"
        for config_label, _ in CONFIGS:
            v = results["Energy"][config_label][0]
            assert math.isfinite(v), f"{config_label} not finite"
        print("\n[SMOKE] PASS: pool=8 chiplets, all configs finite.")


if __name__ == "__main__":
    main()
