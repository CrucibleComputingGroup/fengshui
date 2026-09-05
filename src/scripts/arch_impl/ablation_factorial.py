#!/usr/bin/env python3
"""
Full-factorial (2^4 = 16 combo) ablation for the Fengshui MICRO 2026 rebuttal (C3).

Extends ablation_study.py from one-at-a-time to ALL combinations of the four
BASIC-level techniques, on the FIXED co-designed N=8 pool (same as fig:big_eval):

  M  Heterogeneous Memory   (off -> homogeneous GDDR7; PIM stays feasible)
  B  Heterogeneous Batching (off -> v_het_batch=False)
  T  Tensor Parallelism     (off -> tp_degrees=[1])
  F  Layer Fusion           (off -> gene binary_string all-1s)

The single-toggle monkeypatches in ablation_study.toggle() touch disjoint module
attributes, so combos compose via contextlib.ExitStack. 16 combos x 4 metrics
(each on its objective-matched pool) = 64 runs over the precomputed DB.

Sanity gates (enforced downstream in generate_ablation_heatmap.sanity()):
  - single-off combos must reproduce the one-at-a-time ablation_v2 run. With the
    per-network GA reseed (scripts/chiplet_sel.py deterministic_ga_rng) this is
    run-order and worker-count independent, and matches to full float precision
    on the shipped data; the gate keeps a 3% slack band as a guard only.

Outputs (CSV): <prefix>_raw.csv (per-net values), <prefix>_summary.csv
(per metric x combo: geomean, norm_to_full, per-category norms). The default
prefix reproduces the canonical input of generate_ablation_heatmap.py.

Run from src/scripts:
    python3 arch_impl/ablation_factorial.py --workers 8
"""
import os
import sys
import csv
import math
import time
import argparse
import contextlib
import itertools

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.normpath(os.path.join(THIS_DIR, ".."))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from arch_impl.ablation_study import (  # noqa: E402
    DEFAULT_DB, METRICS, load_pool, toggle, categorize, geomean,
)
from baseline import _build_virtual_nets  # noqa: E402
from chiplet_sel import run_single_optimization  # noqa: E402
from utility_functions import calculate_average_opt_value  # noqa: E402

# Factor name -> ablation_study toggle key ("batch" is a run_single_optimization
# parameter, not a monkeypatch -- handled separately in run_combo).
FACTORS = [
    ("mem",    "homo_gddr7"),
    ("batch",  "homo_batch"),
    ("tp",     "no_tp"),
    ("fusion", "no_fusion"),
]
FACTOR_NAMES = [f for f, _ in FACTORS]


def combo_label(disabled):
    if not disabled:
        return "Full Fengshui"
    return "w/o " + "+".join(f.capitalize() for f in FACTOR_NAMES if f in disabled)


def all_combos():
    """All 2^4 subsets of disabled factors, ordered by #disabled then factor order."""
    combos = []
    for k in range(len(FACTOR_NAMES) + 1):
        for subset in itertools.combinations(FACTOR_NAMES, k):
            combos.append(frozenset(subset))
    return combos


def run_combo(virtual_nets, pool, objective, cost_aware, disabled, workers, database):
    v_het_batch = ("batch" not in disabled)
    with contextlib.ExitStack() as stack:
        for fname, tkey in FACTORS:
            if fname in disabled and tkey != "homo_batch":
                stack.enter_context(toggle(tkey))
        _, results = run_single_optimization(
            virtual_nets=virtual_nets,
            chiplet_group=pool,
            objective=objective,
            results_file=database,  # input DB the GA reads (framework naming quirk)
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
    # the only chain_summary_*.json pools shipped in src/archgym_results/.
    ap.add_argument("--chain-version", default="ae")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--metrics", default="all",
                    help="comma list among: Energy,Energy x Cost,EDP,EDP x Cost")
    # Default name carries the pool version so the generator's pinned canonical
    # input (ablation_factorial_ae_summary.csv) is produced by a bare re-run.
    ap.add_argument("--out-prefix",
                    default=os.path.join(THIS_DIR, "ablation_factorial_ae"))
    ap.add_argument("--smoke", action="store_true",
                    help="2 nets, Energy only, 4 spot combos, validation oracle")
    args = ap.parse_args()

    # A smoke run produces a PARTIAL summary (1 metric, 4 combos); never let it
    # clobber the shipped canonical CSVs that the heatmap generator reads.
    if args.smoke and args.out_prefix == os.path.join(THIS_DIR,
                                                      "ablation_factorial_ae"):
        args.out_prefix += "_smoke"

    t0 = time.time()
    print(f"[factorial] building virtual nets from {args.database} ...", flush=True)
    virtual_nets = _build_virtual_nets(database_file=args.database)
    print(f"[factorial] {len(virtual_nets)} virtual nets", flush=True)

    metrics = METRICS
    combos = all_combos()
    if args.smoke:
        virtual_nets = virtual_nets[:2]
        metrics = [METRICS[0]]
        combos = [frozenset(), frozenset({"mem"}), frozenset({"mem", "fusion"}),
                  frozenset(FACTOR_NAMES)]
        print(f"[factorial] SMOKE: nets={[v.get_unique_name() for v in virtual_nets]}")
    elif args.metrics != "all":
        wanted = {m.strip() for m in args.metrics.split(",")}
        metrics = [m for m in METRICS if m[0] in wanted]

    raw_path = args.out_prefix + "_raw.csv"
    summ_path = args.out_prefix + "_summary.csv"

    raw_rows = []
    results = {}  # results[metric_label][disabled] = (avg, per_net)
    for metric_label, objective, cost_aware in metrics:
        pool, pool_file, _ = load_pool(objective, cost_aware,
                                       chain_version=args.chain_version)
        print(f"\n=== METRIC: {metric_label} (objective={objective}, "
              f"cost_aware={cost_aware}) pool={pool_file} ===", flush=True)
        results[metric_label] = {}
        for disabled in combos:
            tcfg = time.time()
            avg, per_net = run_combo(virtual_nets, pool, objective, cost_aware,
                                     disabled, args.workers, args.database)
            results[metric_label][disabled] = (avg, per_net)
            print(f"  [{combo_label(disabled):34s}] geomean={avg:.4e} "
                  f"({time.time()-tcfg:.1f}s)", flush=True)
            for net, val in per_net.items():
                row = {"metric": metric_label, "objective": objective,
                       "cost_aware": cost_aware, "config": combo_label(disabled),
                       "network": net, "category": categorize(net), "value": val}
                for f in FACTOR_NAMES:
                    row[f] = 0 if f in disabled else 1
                raw_rows.append(row)

    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "objective", "cost_aware",
                                          "config"] + FACTOR_NAMES +
                                         ["network", "category", "value"])
        w.writeheader()
        w.writerows(raw_rows)

    # Summary normalized to the all-on combo, overall + per category.
    cats = ["decode", "prefill", "cnn"]
    summ_rows = []
    for metric_label, _, _ in metrics:
        full_avg, full_per_net = results[metric_label][frozenset()]
        full_cat = {cat: geomean([v for n, v in full_per_net.items()
                                  if categorize(n) == cat]) for cat in cats}
        for disabled in combos:
            avg, per_net = results[metric_label][disabled]
            row = {"metric": metric_label, "config": combo_label(disabled),
                   "n_disabled": len(disabled), "geomean": avg,
                   "norm_to_full": (avg / full_avg)
                   if (full_avg and math.isfinite(full_avg)) else float("inf")}
            for f in FACTOR_NAMES:
                row[f] = 0 if f in disabled else 1
            for cat in cats:
                cg = geomean([v for n, v in per_net.items()
                              if categorize(n) == cat])
                row[f"{cat}_norm"] = (cg / full_cat[cat]) \
                    if (full_cat[cat] and math.isfinite(full_cat[cat])) else float("inf")
            summ_rows.append(row)

    fieldnames = (["metric", "config"] + FACTOR_NAMES +
                  ["n_disabled", "geomean", "norm_to_full",
                   "decode_norm", "prefill_norm", "cnn_norm"])
    with open(summ_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summ_rows)

    def fmt(x):
        return f"{x:.2f}x" if math.isfinite(x) else "inf"

    print("\n\n========= FULL-FACTORIAL SUMMARY (norm to all-on=1.00; >1 worse) =========")
    for metric_label, _, _ in metrics:
        print(f"\n### {metric_label}   (M=mem B=batch T=tp F=fusion; x=disabled)")
        print("| M | B | T | F | Overall | Decode | Prefill | CNN |")
        print("|---|---|---|---|---|---|---|---|")
        for disabled in combos:
            r = next(x for x in summ_rows if x["metric"] == metric_label
                     and x["config"] == combo_label(disabled))
            flags = " | ".join("x" if f in disabled else "+" for f in FACTOR_NAMES)
            print(f"| {flags} | {fmt(r['norm_to_full'])} | {fmt(r['decode_norm'])} | "
                  f"{fmt(r['prefill_norm'])} | {fmt(r['cnn_norm'])} |")

    print(f"\n[factorial] raw     -> {raw_path}")
    print(f"[factorial] summary -> {summ_path}")
    print(f"[factorial] total time {time.time()-t0:.1f}s")

    if args.smoke:
        for disabled in combos:
            v = results["Energy"][disabled][0]
            assert math.isfinite(v) and v > 0, f"{combo_label(disabled)} not finite"
        full = results["Energy"][frozenset()][0]
        alloff = results["Energy"][frozenset(FACTOR_NAMES)][0]
        # Disabling every technique must not improve the objective.
        assert alloff >= full * 0.999, f"all-off {alloff} < full {full}"
        print("\n[SMOKE] PASS: all combos finite, all-off >= full.")


if __name__ == "__main__":
    main()
