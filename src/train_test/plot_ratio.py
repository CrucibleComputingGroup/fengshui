#!/usr/bin/env python3
"""Bar chart of the cross/test_own generalization ratio per config.

For each config, bar height = geometric mean of the PER-WORKLOAD ratios
(cross[w] / test_own[w]) at n=8; the whisker spans the min and max individual
workload ratio (so the bar shows the typical case, the whisker shows the tail).

Usage:
    python3 plot_ratio.py [results_subdir]   # default: results/random_split
"""
import csv, math, os, sys

THIS = os.path.abspath(os.path.dirname(__file__))
SUB = sys.argv[1] if len(sys.argv) > 1 else os.path.join("results", "random_split")
ROOT = SUB if os.path.isabs(SUB) else os.path.join(THIS, SUB)

CONFIGS = [  # (dir name, objective, pretty label)
    ("energy_nocost", "energy", "Energy"),
    ("energy_cost",   "energy", "Energy×\\$"),
    ("edp_nocost",    "edp",    "EDP"),
    ("edp_cost",      "edp",    "EDP×\\$"),
]
N = 8


def row_at(path, n):
    for r in csv.DictReader(open(path)):
        if r["n_chiplets"] == str(n):
            return r
    raise KeyError(f"n={n} not in {path}")


def per_workload_ratios(cfg_dir, obj):
    own = row_at(os.path.join(cfg_dir, "test_sweep_phase2_isaeo.csv"), N)
    cr = row_at(os.path.join(cfg_dir, "cross_eval.csv"), N)
    unames = [k[:-len("_min_" + obj)] for k in own if k.endswith("_min_" + obj)]
    ratios = []
    for u in unames:
        to = own[u + "_min_" + obj]
        cv = cr.get(u + "_" + obj)
        if cv in (None, "", "inf") or to in (None, "", "inf"):
            continue
        to, cv = float(to), float(cv)
        if to > 0 and math.isfinite(to) and math.isfinite(cv):
            ratios.append(cv / to)
    return ratios


def main():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, gms, mins, maxs = [], [], [], []
    for d, obj, pretty in CONFIGS:
        rs = per_workload_ratios(os.path.join(ROOT, d), obj)
        gm = math.exp(sum(math.log(r) for r in rs) / len(rs))
        labels.append(pretty)
        gms.append(gm)
        mins.append(min(rs))
        maxs.append(max(rs))
        print(f"{d:14} geomean={gm:.4f} min={min(rs):.4f} max={max(rs):.4f} (n={len(rs)})")

    x = np.arange(len(labels))
    yerr = [[g - lo for g, lo in zip(gms, mins)],   # lower whisker
            [hi - g for g, hi in zip(gms, maxs)]]   # upper whisker

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(x, gms, width=0.55, color="#4C9BD6", edgecolor="black",
                  linewidth=0.8, zorder=2)
    ax.errorbar(x, gms, yerr=yerr, fmt="none", ecolor="#222", elinewidth=1.4,
                capsize=8, capthick=1.4, zorder=3)
    ax.axhline(1.0, color="crimson", ls="--", lw=1.0, zorder=1,
               label="parity (cross = test-own)")

    # 2-decimal annotations: geomean on the bar, max/min at the whisker ends
    for xi, g, lo, hi in zip(x, gms, mins, maxs):
        ax.text(xi, g + 0.002, f"{g:.2f}", ha="center", va="bottom",
                fontweight="bold", fontsize=10)
        ax.text(xi + 0.30, hi, f"max {hi:.2f}", ha="left", va="center", fontsize=8,
                color="#555")
        ax.text(xi + 0.30, lo, f"min {lo:.2f}", ha="left", va="center", fontsize=8,
                color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Generalization ratio  (cross / test-own)")
    ax.set_title("Train-derived pool on unseen test set (random 70/30 split, n=8)")
    ax.set_ylim(0.85, max(maxs) * 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.tight_layout()

    out = os.path.join(THIS, "ratio_random_split.png")
    fig.savefig(out, dpi=200)
    fig.savefig(out.replace(".png", ".pdf"))
    print("saved:", out)


if __name__ == "__main__":
    main()
