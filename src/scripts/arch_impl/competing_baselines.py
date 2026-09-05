#!/usr/bin/env python3
"""
Competing-framework baselines for the Fengshui (Mozart) MICRO 2026 rebuttal (task C2).

Reproduces two recent chiplet design-space frameworks *inside Fengshui's own backend*
(precomputed Timeloop DB) so the comparison is apples-to-apples (same cost/perf model,
their methodology vs ours). Three frameworks, four metrics:

  - Fengshui (Full) : co-designed heterogeneous N=8 pool (objective-matched, incl. PIM +
                      switch), heterogeneous memory, heterogeneous batching.  == ablation "full".
  - SCAR-style      : FIXED same-size dataflow library {eyeriss(1,1), gemmini(1,1), simba(2,2)}
                      all = 4096 PE (glb=1 ~ SCAR's 10MB L2); homogeneous memory; no PIM/switch;
                      no operator-level non-uniform batching. SCAR's heterogeneity is *dataflow*
                      on same-size chiplets (cf. SCAR MICRO'24), NOT pool co-design.
  - Gemini-style    : HOMOGENEOUS single design whose size/granularity is cost-aware co-searched
                      (exhaustive arch x glb x pe); homogeneous memory; no PIM/switch; no het-batch.
                      Gemini's native objective MC*E*D == our EDP x Cost (cf. Gemini HPCA'24).

Both SCAR-style and Gemini-style are scored under Fengshui's shared backend (NOT their own
cost models) -- that is what makes the comparison iso-infrastructure.

Metrics (geomean over the 20 Fig-11 nets; Full uses its objective-matched pool per metric):
  Energy (energy,F), Energy x Cost (energy,T), EDP (edp,F), EDP x Cost (edp,T)

Run from .../chiplet_timeloop/scripts (so framework modules import cleanly).
"""
import os
import sys
import csv
import copy
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
from chiplet_dataclass import ChipletConfig, generate_all_chiplet_configs
from baseline import _build_virtual_nets
from chiplet_sel import run_single_optimization
from utility_functions import calculate_average_opt_value


# ---------------------------------------------------------------------------
# Fengshui (Full): objective-matched co-designed N=8 pool (same as Figure 11)
# ---------------------------------------------------------------------------
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
        raise KeyError(f"{key} not in {files[-1]}")
    idents = summary[key]["best_chiplets"]
    pool = [ChipletConfig.from_identifier(s) for s in idents]
    return pool, os.path.basename(files[-1]), idents


# ---------------------------------------------------------------------------
# SCAR-style: fixed same-size (4096-PE) heterogeneous-DATAFLOW library.
# eyeriss=(64,64), gemmini=(64,64), simba=(128,32) -> all 4096 PE. glb=1 ~ 8 MiB ~ SCAR 10MB L2.
# ---------------------------------------------------------------------------
def build_scar_pool(dram_type="GDDR7"):
    pool = [
        ChipletConfig("eyeriss_like", 1, 1, 1, dram_type=dram_type),  # 64 x 64  = 4096
        ChipletConfig("gemmini_like", 1, 1, 1, dram_type=dram_type),  # 64 x 64  = 4096
        ChipletConfig("simba_like",   1, 2, 2, dram_type=dram_type),  # 128 x 32 = 4096
    ]
    # Sanity: every chiplet must be 4096 PE (same-size grid, per SCAR).
    for c in pool:
        _, _, px, py = c.get_rounded_config()
        assert px * py == 4096, f"SCAR pool chiplet {c.get_identifier()} = {px}x{py} != 4096 PE"
    return pool


# ---------------------------------------------------------------------------
# Memory homogenization (SCAR + Gemini only). Mirrors ablation_study homo_gddr7.
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def homogenize_memory(dram_type):
    saved = (ga.dram_options, cpp.dram_options, global_parameter.dram_options)
    ga.dram_options = [dram_type]
    cpp.dram_options = [dram_type]
    global_parameter.dram_options = [dram_type]
    try:
        yield
    finally:
        ga.dram_options, cpp.dram_options, global_parameter.dram_options = saved


# ---------------------------------------------------------------------------
# Shared evaluation + categorization
# ---------------------------------------------------------------------------
def eval_pool(virtual_nets, pool, objective, cost_aware, v_het_batch, workers, database):
    """run_single_optimization for a fixed pool. `database` is the INPUT DB (framework quirk)."""
    _, results = run_single_optimization(
        virtual_nets=virtual_nets,
        chiplet_group=pool,
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


def categorize(net_unique_name):
    n = net_unique_name.lower()
    # qwen3_30b_a3b / qwen3_235b_a22b are MoE (active-expert variants -> use the
    # switch_8port expert-parallel chiplet); llama is dense. Split them out so the
    # MoE/EP regime (where SCAR/Gemini lack a switch) is visible, not bundled.
    is_moe = "qwen" in n
    if "decode" in n:
        return "decode-MoE" if is_moe else "decode"
    if "prefill" in n:
        return "prefill-MoE" if is_moe else "prefill"
    if "mobilenet" in n or "replknet" in n or "resnet" in n or "vit" in n:
        return "cnn"
    return "other"


def geomean(values):
    vals = [v for v in values if v is not None and math.isfinite(v) and v > 0]
    if not vals:
        return float("inf")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


# ---------------------------------------------------------------------------
# Per-framework runners
# ---------------------------------------------------------------------------
def run_full(virtual_nets, objective, cost_aware, workers, database, chain_version="ae"):
    pool, pool_file, idents = load_pool(objective, cost_aware, chain_version=chain_version)
    avg, per_net = eval_pool(virtual_nets, pool, objective, cost_aware,
                             v_het_batch=True, workers=workers, database=database)
    return avg, per_net, {"pool_file": pool_file, "chiplets": idents}


def run_scar(virtual_nets, objective, cost_aware, workers, database, dram_type="GDDR7"):
    pool = build_scar_pool(dram_type)
    with homogenize_memory(dram_type):
        avg, per_net = eval_pool(virtual_nets, pool, objective, cost_aware,
                                 v_het_batch=False, workers=workers, database=database)
    return avg, per_net, {"dram": dram_type,
                          "chiplets": [c.get_identifier() for c in pool]}


def run_gemini(virtual_nets, objective, cost_aware, workers, database,
               dram_type="GDDR7", candidates=None):
    """Homogeneous, cost-aware granularity co-search: pick the single design that
    minimizes this metric's geomean across the suite."""
    cand = candidates if candidates is not None else generate_all_chiplet_configs(
        arch_targets=["eyeriss_like", "simba_like", "gemmini_like"])
    best = None  # (geomean, design, per_net)
    with homogenize_memory(dram_type):
        for i, design in enumerate(cand):
            d = copy.deepcopy(design)
            d.dram_type = dram_type
            avg, per_net = eval_pool(virtual_nets, [d], objective, cost_aware,
                                     v_het_batch=False, workers=workers, database=database)
            if math.isfinite(avg) and (best is None or avg < best[0]):
                best = (avg, d, per_net)
            if (i + 1) % 20 == 0 or (i + 1) == len(cand):
                bid = best[1].get_identifier() if best else "none"
                print(f"      [gemini] {i+1}/{len(cand)} searched, "
                      f"best={bid} ({best[0]:.4e})" if best else
                      f"      [gemini] {i+1}/{len(cand)} searched", flush=True)
    if best is None:
        return float("inf"), {}, {"dram": dram_type, "design": "none"}
    return best[0], best[2], {"dram": dram_type, "design": best[1].get_identifier()}


def run_homo_pernet(virtual_nets, objective, cost_aware, workers, database,
                    dram_type="GDDR7", candidates=None):
    """Homogeneous PER-NET: best single (homogeneous) design chosen independently
    per workload (custom silicon per workload), homogeneous memory. The most
    generous homogeneous baseline -- the upper bound Gemini argues against."""
    cand = candidates if candidates is not None else generate_all_chiplet_configs(
        arch_targets=["eyeriss_like", "simba_like", "gemmini_like"])
    best_per_net = {}  # net -> (val, ident)
    with homogenize_memory(dram_type):
        for i, design in enumerate(cand):
            d = copy.deepcopy(design)
            d.dram_type = dram_type
            _, per_net = eval_pool(virtual_nets, [d], objective, cost_aware,
                                   v_het_batch=False, workers=workers, database=database)
            for net, val in per_net.items():
                if math.isfinite(val) and (net not in best_per_net or val < best_per_net[net][0]):
                    best_per_net[net] = (val, d.get_identifier())
            if (i + 1) % 40 == 0 or (i + 1) == len(cand):
                print(f"      [homo-per-net] {i+1}/{len(cand)} designs searched", flush=True)
    per_net = {net: v for net, (v, _) in best_per_net.items()}
    avg = geomean(list(per_net.values()))
    return avg, per_net, {"dram": dram_type, "mode": "best homogeneous design per workload"}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
METRICS = [
    ("Energy",         "energy", False),
    ("Energy x Cost",  "energy", True),
    ("EDP",            "edp",    False),
    ("EDP x Cost",     "edp",    True),
]
FRAMEWORKS = ["Fengshui (Full)", "SCAR-style", "Gemini-style"]
CATS = ["decode", "decode-MoE", "prefill", "prefill-MoE", "cnn"]
CAT_LABELS = {"decode": "Decode", "decode-MoE": "Decode-MoE",
              "prefill": "Prefill", "prefill-MoE": "Prefill-MoE", "cnn": "CNN"}


def emit_summary(results, metrics_used, frameworks, has_full, summ_path, meta=None):
    """results[metric][fw] = (avg, per_net). Write summary CSV + print markdown,
    normalized to Fengshui Full = 1.0, overall + per (MoE-split) category."""
    summ_rows = []
    for metric_label in metrics_used:
        if has_full:
            full_avg, full_per_net = results[metric_label]["Fengshui (Full)"]
            full_cat = {c: geomean([v for n, v in full_per_net.items()
                                    if categorize(n) == c]) for c in CATS}
        else:
            full_avg, full_cat = float("nan"), {c: float("nan") for c in CATS}
        for fw in frameworks:
            avg, per_net = results[metric_label][fw]
            row = {"metric": metric_label, "framework": fw, "geomean": avg,
                   "norm_to_full": (avg / full_avg) if (full_avg and math.isfinite(full_avg)) else float("inf"),
                   "detail": json.dumps(meta[metric_label][fw]) if meta else ""}
            for c in CATS:
                cg = geomean([v for n, v in per_net.items() if categorize(n) == c])
                row[f"{c}_norm"] = (cg / full_cat[c]) if (full_cat[c] and math.isfinite(full_cat[c])) else float("inf")
            summ_rows.append(row)

    fieldnames = (["metric", "framework", "geomean", "norm_to_full"]
                  + [f"{c}_norm" for c in CATS] + ["detail"])
    with open(summ_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summ_rows)

    print("\n\n========= C2 COMPETING-FRAMEWORK COMPARISON "
          "(normalized to Fengshui Full = 1.00; >1.0 = worse than Fengshui) =========\n")
    hdr = " | ".join(CAT_LABELS[c] for c in CATS)
    sep = "|---|---|" + "---|" * len(CATS)
    for metric_label in metrics_used:
        print(f"### {metric_label}")
        print(f"| Framework | Overall | {hdr} |")
        print(sep)
        for fw in frameworks:
            r = next(x for x in summ_rows if x["metric"] == metric_label and x["framework"] == fw)
            fmt = lambda x: f"{x:.2f}x" if math.isfinite(x) else "inf"
            cells = " | ".join(fmt(r[f"{c}_norm"]) for c in CATS)
            print(f"| {fw} | {fmt(r['norm_to_full'])} | {cells} |")
        print()
    print(f"[c2] summary -> {summ_path}")
    return summ_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", default=DEFAULT_DB)
    ap.add_argument("--chain-version", default="ae",
                    help="chiplet-pool generation for Fengshui (Full); v7 = energy/ViT-softmax corrected DB")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scar-dram", default="GDDR7")
    ap.add_argument("--gemini-dram", default="GDDR7")
    ap.add_argument("--metrics", default="all",
                    help="comma list among: Energy,Energy x Cost,EDP,EDP x Cost")
    ap.add_argument("--frameworks", default="all",
                    help="comma list among: Fengshui (Full),SCAR-style,Gemini-style")
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="2 nets, Energy only, Gemini over 3 candidates; validation oracle")
    ap.add_argument("--reaggregate", default=None,
                    help="path to a *_raw.csv: re-summarize with MoE split, no recompute")
    args = ap.parse_args()

    if args.reaggregate:
        rows = list(csv.DictReader(open(args.reaggregate)))
        rdict = {}
        for r in rows:
            rdict.setdefault(r["metric"], {}).setdefault(
                r["framework"], {})[r["network"]] = float(r["value"])
        results = {m: {fw: (geomean(list(pn.values())), pn) for fw, pn in fws.items()}
                   for m, fws in rdict.items()}
        metrics_used = [ml for (ml, _, _) in METRICS if ml in results]
        fw_present = [fw for fw in FRAMEWORKS if any(fw in results[m] for m in results)]
        has_full = "Fengshui (Full)" in fw_present
        base = (args.reaggregate[:-len("_raw.csv")]
                if args.reaggregate.endswith("_raw.csv") else args.reaggregate)
        summ_path = (args.out_prefix + "_summary_moe.csv") if args.out_prefix else base + "_summary_moe.csv"
        print(f"[c2] reaggregate {args.reaggregate} -> MoE-split summary")
        emit_summary(results, metrics_used, fw_present, has_full, summ_path, meta=None)
        return

    t0 = time.time()
    print(f"[c2] building virtual nets from {args.database} ...", flush=True)
    virtual_nets = _build_virtual_nets(database_file=args.database)
    print(f"[c2] {len(virtual_nets)} virtual nets loaded", flush=True)

    metrics = METRICS
    gemini_candidates = None
    if args.smoke:
        virtual_nets = virtual_nets[:2]
        metrics = [METRICS[0]]
        gemini_candidates = [
            ChipletConfig("eyeriss_like", 1, 1, 1),
            ChipletConfig("simba_like",   1, 2, 2),
            ChipletConfig("gemmini_like", 4, 2, 2),
        ]
        print(f"[c2] SMOKE: nets={[v.get_unique_name() for v in virtual_nets]}", flush=True)
    elif args.metrics != "all":
        wanted = {m.strip() for m in args.metrics.split(",")}
        metrics = [m for m in METRICS if m[0] in wanted]

    frameworks = FRAMEWORKS
    if not args.smoke and args.frameworks != "all":
        wanted_fw = {f.strip() for f in args.frameworks.split(",")}
        frameworks = [f for f in (FRAMEWORKS + ["homo-per-net"]) if f in wanted_fw]
    has_full = "Fengshui (Full)" in frameworks
    if not has_full:
        print("[c2] WARNING: 'Fengshui (Full)' not in --frameworks; norm_to_full will be NaN.")

    ts = time.strftime("%Y%m%d_%H%M%S")
    prefix = args.out_prefix or os.path.join(THIS_DIR, f"competing_{ts}")
    raw_path, summ_path = prefix + "_raw.csv", prefix + "_summary.csv"

    raw_rows = []
    results = {}   # results[metric_label][framework] = (avg, per_net)
    meta = {}      # meta[metric_label][framework] = info dict

    for metric_label, objective, cost_aware in metrics:
        print(f"\n=== METRIC: {metric_label} (objective={objective}, "
              f"cost_aware={cost_aware}) ===", flush=True)
        results[metric_label], meta[metric_label] = {}, {}
        for fw in frameworks:
            tcfg = time.time()
            if fw == "Fengshui (Full)":
                avg, per_net, info = run_full(virtual_nets, objective, cost_aware,
                                              args.workers, args.database, args.chain_version)
            elif fw == "SCAR-style":
                avg, per_net, info = run_scar(virtual_nets, objective, cost_aware,
                                              args.workers, args.database, args.scar_dram)
            elif fw == "homo-per-net":
                avg, per_net, info = run_homo_pernet(virtual_nets, objective, cost_aware,
                                                     args.workers, args.database, args.gemini_dram)
            else:  # Gemini-style
                avg, per_net, info = run_gemini(virtual_nets, objective, cost_aware,
                                                args.workers, args.database,
                                                args.gemini_dram, gemini_candidates)
            results[metric_label][fw] = (avg, per_net)
            meta[metric_label][fw] = info
            print(f"  [{fw:18s}] geomean={avg:.4e}  ({time.time()-tcfg:.1f}s)  {info}",
                  flush=True)
            for net, val in per_net.items():
                raw_rows.append({
                    "metric": metric_label, "objective": objective,
                    "cost_aware": cost_aware, "framework": fw,
                    "network": net, "category": categorize(net), "value": val,
                })

    # Raw
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "objective", "cost_aware",
                                          "framework", "network", "category", "value"])
        w.writeheader()
        w.writerows(raw_rows)

    # Summary (normalized to Fengshui Full = 1.0, overall + MoE-split categories)
    metrics_used = [ml for (ml, _, _) in metrics]
    emit_summary(results, metrics_used, frameworks, has_full, summ_path, meta=meta)
    print(f"[c2] raw     -> {raw_path}")
    print(f"[c2] total time {time.time()-t0:.1f}s")

    # Validation oracle (smoke)
    if args.smoke:
        m = "Energy"
        full_pool, _, _ = load_pool("energy", False, chain_version=args.chain_version)
        assert len(full_pool) == 8, f"expected Full pool=8, got {len(full_pool)}"
        scar_pool = build_scar_pool(args.scar_dram)
        assert len(scar_pool) == 3, f"expected SCAR pool=3, got {len(scar_pool)}"
        with homogenize_memory(args.scar_dram):
            assert ga.dram_options == [args.scar_dram], "homogenize did not patch ga.dram_options"
            assert cpp.dram_options == [args.scar_dram], "homogenize did not patch cpp.dram_options"
        for fw in frameworks:
            v = results[m][fw][0]
            assert math.isfinite(v), f"{fw} {m} not finite"
        full = results[m]["Fengshui (Full)"][0]
        scar = results[m]["SCAR-style"][0]
        gem = results[m]["Gemini-style"][0]
        gem_design = meta[m]["Gemini-style"]["design"]
        print(f"\n[SMOKE] Full=8 chiplets; SCAR=3 @4096PE; GDDR7 homogenize fires; "
              f"Gemini picked '{gem_design}'.")
        print(f"[SMOKE] ratios vs Full: SCAR={scar/full:.2f}x  Gemini={gem/full:.2f}x "
              f"(expected >= 1.0; 2-net sample is noisy).")
        print("[SMOKE] PASS: all frameworks finite, pools + homogenize validated.")


if __name__ == "__main__":
    main()
