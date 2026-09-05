#!/usr/bin/env python3
"""Single-column COMBINED Figure 10: (a) per-unit cost breakdown (broken y-axis) and
(b) cost-model sensitivity tornado, side by side in ONE acmart \\columnwidth (~3.33in) figure.

Reuses the exact cost functions from generate_nre_cost_figure.py (breakdown) and the
precomputed sweep from generate_cost_sweep.py (tornado) -- no recomputation of the model,
only a compact combined layout tuned to stay readable at single-column width.
"""
import os, sys, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate_nre_cost_figure as G       # cost functions + VOLUMES + colors
import generate_cost_sweep as S            # SWEEPS (precomputed), BASE

AREAS = json.load(open(os.path.join(HERE, "fig10_areas.json")))
POWER = json.load(open(os.path.join(HERE, "fig10_power.json")))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 5.0, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
})

STRATS = ["homo_asic", "homo_basic", "het_unconstrained", "het_pool"]
# Paradigm names match the paper's 5-paradigm definition (codesign_framework/index.tex):
# Homogeneous ASIC / Homogeneous BASIC / Heterogeneous BASIC (unconstrained) /
# Heterogeneous BASIC (chiplet pool) == Fengshui.
SHORT = ["Homogeneous\nASIC", "Homogeneous\nBASIC",
         "Heterogeneous\nBASIC\n(unconstr.)", "Heterogeneous\nBASIC\n(Fengshui)"]
COL = {"die": "#7FB3D8", "pkg": "#F4A460", "nre": "#90EE90", "op": "#C39BD3"}


def _breakdown_data():
    re = {s: G.avg_re_cost(AREAS[s]) for s in STRATS}
    op = {s: G.op_cost(POWER[s]) for s in STRATS}
    data = {}
    for s in STRATS:
        d, p = re[s]
        data[s] = [(d, p, G.nre_per_unit(AREAS[s], v), op[s]) for v in G.VOLUMES]
    return data, re, op


def _kfmt(v):
    """Compact $ tick: 0, 1k, 2k, 10k ..."""
    return "0" if v == 0 else f"{v/1000:g}k"


def main(save=None):
    import matplotlib.patches as mpatches
    data, re, op = _breakdown_data()

    fig = plt.figure(figsize=(3.34, 1.75))
    # one shared legend strip at the very top; panels below it.
    # panel (b) widened to fill the previously-empty gap.
    ax_a = fig.add_axes([0.135, 0.275, 0.43, 0.55])     # breakdown (single linear axis)
    ax_t = fig.add_axes([0.675, 0.275, 0.305, 0.55])    # tornado

    # ---- shared top legend (out of the plot -> frees panel-a interior) ----
    handles = [mpatches.Patch(facecolor=COL[k], edgecolor="black", linewidth=0.3, label=l)
               for k, l in (("die", "Die"), ("pkg", "Packaging"), ("nre", "NRE"), ("op", "Operational"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=4,
               fontsize=5.2, frameon=False, handlelength=0.8, handleheight=0.8,
               handletextpad=0.35, columnspacing=0.8)

    # ---- breakdown bars (thin bars, wider group gap = more whitespace) ----
    bw, ig, gap = 0.17, 0.06, 0.42   # bar width, intra-group gap, inter-group gap
    step = bw + ig
    xpos, centers = [], []
    for g in range(len(STRATS)):
        base = g * (3 * step + gap)
        centers.append(base + step)   # middle bar of the 3-bar cluster
        for b in range(3):
            xpos.append(base + b * step)
    i = 0
    for s in STRATS:
        for b in range(3):
            die, pkg, nre, opc = data[s][b]
            x = xpos[i]
            ax_a.bar(x, die, bw, color=COL["die"], edgecolor="black", linewidth=0.25)
            ax_a.bar(x, pkg, bw, bottom=die, color=COL["pkg"], edgecolor="black", linewidth=0.25)
            ax_a.bar(x, nre, bw, bottom=die + pkg, color=COL["nre"], edgecolor="black", linewidth=0.25)
            ax_a.bar(x, opc, bw, bottom=die + pkg + nre, color=COL["op"], edgecolor="black", linewidth=0.25)
            i += 1

    max_tot = max(sum(data[s][b]) for s in STRATS for b in range(3))
    ax_a.set_ylim(0, max_tot * 1.05)
    ax_a.set_yticks([0, 5000, 10000]); ax_a.set_yticklabels(["0", "5k", "10k"])
    ax_a.set_xlim(xpos[0] - bw, xpos[-1] + bw)
    ax_a.tick_params(labelsize=4.6)

    # two-row x-axis: per-bar volume (1/2/3 M units) under the bars, paradigm name below
    xtr = ax_a.get_xaxis_transform()
    vlabels = [f"{v}M" for v in G.VOLUME_LABELS]   # 1M / 2M / 3M (million units)
    for i, x in enumerate(xpos):
        ax_a.text(x, -0.04, vlabels[i % 3], transform=xtr,
                  ha="center", va="top", fontsize=3.7)
    ax_a.set_xticks(centers); ax_a.set_xticklabels(SHORT, fontsize=3.5, linespacing=0.8)
    ax_a.tick_params(axis="x", length=0, pad=7.5)
    ax_a.set_ylabel("Cost (\\$/unit)", fontsize=5.4, labelpad=1)
    ax_a.set_title("(a) Cost breakdown", fontsize=5.6, pad=1.5)

    # ---- tornado ----
    rows = sorted(S.SWEEPS, key=lambda r: abs(r[4] - r[2]))
    short_lbl = {"NRE / design": "NRE", "Production volume": "Volume", "Packaging cost": "Pkg.",
                 "Interposer cost": "Interp.",
                 "Wafer cost": "Wafer", "Die defect density": "Defect",
                 "Electricity price": "Elec."}
    ax_t.set_xscale("log")
    # Endpoint labels are ABSOLUTE values with units, already matplotlib-ready in S.SWEEPS
    # (Reviewer B): no 0.5x/2x multipliers, so the swept parameter (domain) is never confused
    # with the x-axis advantage ratio (codomain).
    for i, (lbl, lv, lf, hv, hf) in enumerate(rows):
        lo, hi = min(lf, hf), max(lf, hf)
        ax_t.barh(i, hi - lo, left=lo, height=0.56, color="#7FB3D8", edgecolor="black", linewidth=0.3)
        # annotate the SWEPT PARAMETER value at each end (the param value that yields that factor)
        left_lbl = lv if lf <= hf else hv
        right_lbl = hv if hf >= lf else lv
        ax_t.text(lo / 1.05, i, left_lbl, va="center", ha="right", fontsize=3.7, color="#222")
        ax_t.text(hi * 1.05, i, right_lbl, va="center", ha="left", fontsize=3.7, color="#222")
    ax_t.axvline(S.BASE, color="#555", linewidth=0.7)   # default-parameter baseline (~12x)
    ax_t.axvline(1.0, color="#C0392B", linestyle="--", linewidth=0.7)
    ax_t.text(1.05, 2.0, "break-even", fontsize=4.4, color="#C0392B",
              va="center", ha="left", rotation=90)
    ax_t.set_yticks(range(len(rows)))
    ax_t.set_yticklabels([short_lbl[r[0]] for r in rows], fontsize=5.0)
    ax_t.set_xlabel("Pool cost advantage ($\\times$)", fontsize=5.4, labelpad=1)
    ax_t.set_xlim(0.8, max(hf for *_, hf in S.SWEEPS) * 1.9)
    ax_t.set_xticks([1, 10]); ax_t.set_xticklabels(["1", "10"])
    ax_t.tick_params(axis="x", labelsize=4.6)
    ax_t.tick_params(axis="y", length=0, pad=1.2)
    ax_t.grid(axis="x", which="both", linewidth=0.2, alpha=0.35)
    ax_t.set_title("(b) Sensitivity", fontsize=5.6, pad=1.5)

    if save:
        fig.savefig(save, dpi=500, bbox_inches="tight", pad_inches=0.02)
        print("Saved", save)
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None)
    main(ap.parse_args().save)
