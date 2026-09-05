#!/usr/bin/env python3
"""
Combined generalization figure across the 3 held-out splits.

For each split (random / decode_test / moe_test) and each of the 4 configs
(energy, energy x cost, edp, edp x cost) we draw TWO bars:
  - "Ours"   : the split's own TRAIN-derived 8-chiplet pool, evaluated on the test set
  - "Fengshui" : the paper's universal 8-chiplet pool, evaluated on the SAME test set
both normalised to that test set's OWN-optimal pool (generalization ratio,
= pool geomean on test / test-own-optimal geomean; 1.0 = matches the ceiling).

=> 3 groups x (4 configs x 2 bars) = 24 bars, one figure. Log y (decode_test
holds out the PIM-driven regime and blows up).

Reads: results/<split>/<config>/summary.json  and  fengshui_cross.json
Writes: figures/generalization_combined.png  and  generalization_combined.csv
"""
import os, json, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
MOZ_JSON = os.path.join(ROOT, "fengshui_cross.json")
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

SPLITS = [("random_split", "Random"), ("decode_test", "Decode held-out"),
          ("moe_test", "MoE (Qwen) held-out")]
CONFIGS = [("energy_nocost", "Energy"), ("energy_cost", "Energy×$"),
           ("edp_nocost", "EDP"), ("edp_cost", "EDP×$")]

fengshui = json.load(open(MOZ_JSON))

# Gather ratios
rows = []          # for CSV
data = {}          # data[(split,cfg)] = (ours_ratio, fengshui_ratio)
for sk, _ in SPLITS:
    for ck, _ in CONFIGS:
        summ = json.load(open(os.path.join(RES, sk, ck, "summary.json")))
        test_own = summ["test_own"]["8"]
        ours = summ["cross"]["8"]
        moz = fengshui[sk][ck]["fengshui_cross"]
        ours_r = ours / test_own
        moz_r = moz / test_own
        data[(sk, ck)] = (ours_r, moz_r)
        rows.append(dict(split=sk, config=ck, test_own=test_own,
                         ours_cross=ours, fengshui_cross=moz,
                         ours_ratio=ours_r, fengshui_ratio=moz_r))

with open(os.path.join(ROOT, "generalization_combined.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# ---- plot ----
fig, ax = plt.subplots(figsize=(15, 6))
bw = 0.38
cluster = 1.0          # spacing between config clusters
split_gap = 1.0        # extra gap between splits
c_ours, c_moz = "#2171b5", "#cb181d"

xticks, xlabels = [], []
split_centers = []
x = 0.0
for si, (sk, slabel) in enumerate(SPLITS):
    start = x
    for ci, (ck, clabel) in enumerate(CONFIGS):
        ours_r, moz_r = data[(sk, ck)]
        b1 = ax.bar(x, ours_r, bw, color=c_ours,
                    label="Ours (train pool)" if (si == 0 and ci == 0) else None)
        b2 = ax.bar(x + bw, moz_r, bw, color=c_moz,
                    label="Fengshui (paper pool)" if (si == 0 and ci == 0) else None)
        for b, val in ((b1, ours_r), (b2, moz_r)):
            if val >= 1.15:
                ax.text(b[0].get_x() + bw / 2, val * 1.02, f"{val:.1f}×",
                        ha="center", va="bottom", fontsize=7)
        xticks.append(x + bw / 2)
        xlabels.append(clabel)
        x += cluster
    split_centers.append((start + x - cluster + bw / 2) / 1.0
                         if False else (start + (x - cluster)) / 2 + bw / 2)
    x += split_gap

ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7)
ax.text(ax.get_xlim()[1], 1.0, " test-own optimum", va="center",
        ha="left", fontsize=8, color="gray")
ax.set_yscale("log")
ax.set_ylabel("Generalization ratio  (pool geomean on test / test-own optimum)")
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, fontsize=8, rotation=0)
# split labels under groups
for (sk, slabel), cx in zip(SPLITS, split_centers):
    ax.text(cx, -0.16, slabel, transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=11, fontweight="bold")
ax.set_title("Train-pool vs Fengshui-pool generalization to held-out test sets "
             "(8-chiplet pools, n_evals differ across splits)", fontsize=11)
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
out = os.path.join(FIGDIR, "generalization_combined.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
print("wrote", os.path.join(ROOT, "generalization_combined.csv"))

# console table
print(f"\n{'split':14}{'config':14}{'test_own':>12}{'ours/own':>10}{'fengshui/own':>12}")
for r in rows:
    print(f"{r['split']:14}{r['config']:14}{r['test_own']:12.3e}"
          f"{r['ours_ratio']:10.2f}{r['fengshui_ratio']:12.2f}")
