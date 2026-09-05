#!/usr/bin/env python3
"""V2 generalization figure WITH finetune overlay (n=8 -> n=9).

Same layout as plot_v2_cross.py (5 splits x 4 configs), but each bar now shows
two heights, SAME denominator (test-own n=8 optimum):

  light bar  = cross_ratio = train-8-pool on test / test-own(n8)   [n=8 generalization gap]
  dark  bar  = ft_ratio    = finetuned-9-pool on test / test-own(n8)  [after adding 1 chiplet]

The light part sticking out above the dark bar = the generalization gap that the
single finetuned chiplet removes. 1.0 (dashed) = test-own optimum. Log y.

Reads : finetune_results/finetune_master_table.csv
Writes: figures/generalization_v2_finetune.png/.pdf

With no arguments the historical in-tree locations are used, so existing
command-line usage is unchanged. Pass explicit paths to plot the master table
of an isolated run summarized by ``summarize_finetune.py --out-dir``.
"""
import argparse
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master",
        default=os.path.join(ROOT, "finetune_results",
                             "finetune_master_table.csv"),
        help="Master table written by summarize_finetune.py "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--figdir",
        default=os.path.join(ROOT, "figures"),
        help="Destination directory for the rendered figure "
             "(default: %(default)s)",
    )
    return parser.parse_args()


args = parse_args()
MASTER = os.path.abspath(os.path.expanduser(args.master))
FIGDIR = os.path.abspath(os.path.expanduser(args.figdir))
os.makedirs(FIGDIR, exist_ok=True)

SPLITS = [("vision_test_v2", "Vision\nheld-out"),
          ("moe_test_v2", "MoE (Qwen)\nheld-out"),
          ("seq2048_test_v2", "Seq-2048\nheld-out"),
          ("prefill_test_v2", "Prefill\nheld-out"),
          ("decode_test_v2", "Decode\nheld-out")]
# Set2 qualitative palette
CONFIGS = [("energy_nocost", "Energy", "#66c2a5"),
           ("energy_cost", "Energy×$", "#fc8d62"),
           ("edp_nocost", "EDP", "#8da0cb"),
           ("edp_cost", "EDP×$", "#e78ac3")]

CAP = 2.5  # linear y-axis ceiling; bars above are clipped + annotated "N×"
YFLOOR = 0.5  # linear y-axis floor (all bars/ratios sit above this)

import matplotlib.colors as _mcolors
def darken(c, f=0.98):
    r, g, b = _mcolors.to_rgb(c)
    return (r * f, g * f, b * f)

# short label for the added 9th chiplet's arch family (for annotation)
def arch_short(ident):
    a = ident.split("@")[0]
    return {"switch_8port": "switch", "eyeriss_like": "eyeriss",
            "gemmini_like": "gemmini", "simba_like": "simba", "PIM": "PIM"}.get(a, a)

# ---- gather ----
data = {}  # (split,config) -> dict
for r in csv.DictReader(open(MASTER)):
    to = float(r["test_own_n8"]); cross = float(r["cross_n8"]); ft = float(r["finetune_n9"])
    data[(r["split"], r["config"])] = dict(
        cross_ratio=cross / to, ft_ratio=ft / to,
        new9=arch_short(r["new_9th"]), gap=float(r["gap_closed_pct"]))

# ---- plot ----
fig, ax = plt.subplots(figsize=(8, 3.7))
bw = 0.038         # thinner bars
group_gap = 0.2
centers = []
for si, (sk, slabel) in enumerate(SPLITS):
    base = si * group_gap
    for ci, (ck, clabel, color) in enumerate(CONFIGS):
        if (sk, ck) not in data:
            continue
        d = data[(sk, ck)]
        x = base + (ci - 1.5) * bw
        cross = d["cross_ratio"]; ft = d["ft_ratio"]
        ft_disp = min(ft, CAP)
        # solid base = n=9 finetune; light top = remaining n=8 cross gap
        ax.bar(x, ft_disp, bw, color=darken(color), edgecolor="black",
               linewidth=0.4, alpha=0.88)
        gap_disp = min(max(cross - ft, 0.0), CAP - ft_disp)
        ax.bar(x, gap_disp, bw, bottom=ft_disp, color=color, edgecolor="black",
               linewidth=0.4, alpha=0.32)
        # annotate finetune ratio just below top of the solid base
        ax.text(x, ft_disp * 0.985, f"{ft:.2f}", ha="center",
                va="top", fontsize=5, color="white", weight="bold", rotation=90)
        # annotate cross ratio at top of light bar (at ceiling if clipped)
        if cross > CAP:
            ax.text(x, CAP * 0.985, f"{cross:.1f}×", ha="center", va="top",
                    fontsize=6, color=color, weight="bold", rotation=90)
        else:
            ax.text(x, cross * 1.02, f"{cross:.1f}×", ha="center", va="bottom",
                    fontsize=6, color="0.35")
    centers.append(base)

ax.axhline(1.0, color="gray", ls="--", lw=1.2, alpha=0.85)
ax.set_ylim(YFLOOR, CAP)
ax.set_ylabel("Normalized Value", fontsize=12, color="0.1", weight="bold")
split_fam = {sk: "/".join(sorted({data[(sk, ck)]["new9"]
                                  for ck, _, _ in CONFIGS if (sk, ck) in data}))
             for sk, _ in SPLITS}
ax.set_xticks(centers)
ax.set_xticklabels(["Vision", "MoE", "Longctx", "Prefill", "Decode"],
                   fontsize=11, color="0.1", weight="bold")
ax.tick_params(axis="both", colors="0.1", labelsize=10, width=1.4)
plt.setp(ax.get_yticklabels(), weight="bold", color="0.1")
for sp in ax.spines.values():
    sp.set_color("0.1"); sp.set_linewidth(1.6)
from matplotlib.transforms import blended_transform_factory
_trans = blended_transform_factory(ax.transAxes, ax.transData)
ax.text(1.005, 1.0, "test", transform=_trans, va="center", ha="left",
        fontsize=9, color="gray")

# legends on top: objective colors + solid/hatched meaning, single row, no title
obj_handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=l)
               for _, l, c in CONFIGS]
sem_handles = [Patch(facecolor="0.6", edgecolor="black", linewidth=0.5,
                     label="n=9 finetune (+1 chiplet)"),
               Patch(facecolor="0.6", alpha=0.4, edgecolor="black",
                     linewidth=0.5, label="n=8 cross gap (train pool)")]
ax.legend(handles=obj_handles + sem_handles, ncol=6,
          loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False,
          columnspacing=1.2, handletextpad=0.5,
          prop={"size": 9, "weight": "semibold"})
ax.grid(axis="y", which="both", alpha=0.25)
plt.tight_layout()
for ext in ("png", "pdf"):
    out = os.path.join(FIGDIR, f"generalization_v2_finetune.{ext}")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
