#!/usr/bin/env python3
"""Generate the full-factorial ablation heatmap (fig:ablation, rebuttal C3).

Reads arch_impl/ablation_factorial_ae_summary.csv (from ablation_factorial.py:
all 2^4 combinations of {het-memory, het-batching, TP, fusion} x 4 metrics on
the fixed N=8 pool, normalized to all-on Fengshui = 1.0) and emits
  - the 16-row x 4-metric annotated heatmap (PDF/PNG) into $FENGSHUI_FIG_OUT
    (default: arch_impl/images/)
  - arch_impl/ablation_constants.tex -- the \\Abl... LaTeX macros that ARE the
    Table 3 / ablation prose numbers (single source of truth, 1-decimal policy).
    These are ALWAYS written, including under --no-overleaf.
  - optionally (when the destination tree exists and --no-overleaf is NOT given)
    a copy of the PDF + constants into the paper tree ($FENGSHUI_ABLATION_IMG /
    $FENGSHUI_ABLATION_CONST). That tree is not part of the artifact, so the
    copy is best-effort and is skipped with a message when absent.

Checks performed (see sanity()):
  - every one of the 16 combos is present for every metric, and finite
  - no combination beats Full Fengshui by more than 1%
  - the four single-disable rows reproduce ablation_v2_ae_summary.csv
  - advisory warning if a single-disable row is exactly 1.000000 on EVERY
    metric (would indicate a toggle that did nothing)

Run from src/scripts/:
    python3 arch_impl/generate_ablation_heatmap.py [--no-overleaf]
"""
import argparse
import csv
import itertools
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Canonical input = the CORRECTED-DB + v7-pool factorial run (post DRAM
# double-count fix, 2026-06-15). The old "ablation_factorial_summary.csv" was
# the pre-fix v6 snapshot (now quarantined as *_BUGGY_pre_dramfix_*.csv) — a
# bare re-run pointed there would silently overwrite the paper with buggy
# numbers, so the default is pinned to the v7 corrected artifact.
SUMMARY = os.path.join(THIS_DIR, "ablation_factorial_ae_summary.csv")
V2_SUMMARY = os.path.join(THIS_DIR, "ablation_v2_ae_summary.csv")
# Figure output directory. Same idiom as generate_paper_fig6_chain.py /
# generate_paper_fig10.py: the notebook sets FENGSHUI_FIG_OUT so every generator
# drops its figure where reproduce_all.ipynb looks; the arch_impl/images/
# fallback keeps a bare CLI run working (and is what the notebook falls back to).
OUT_DIR = os.environ.get("FENGSHUI_FIG_OUT", os.path.join(THIS_DIR, "images"))
# In-artifact constants file: ALWAYS written, independent of the paper tree.
LOCAL_CONST = os.path.join(THIS_DIR, "ablation_constants.tex")
OVERLEAF_IMG = os.environ.get("FENGSHUI_ABLATION_IMG",
                             os.path.join(THIS_DIR, "..", "..", "figures", "ablation_heatmap.pdf"))
OVERLEAF_CONST = os.environ.get("FENGSHUI_ABLATION_CONST",
                               os.path.join(THIS_DIR, "..", "..", "constants", "ablation_constants.tex"))

FACTORS = ["mem", "batch", "tp", "fusion"]
FACTOR_HEADERS = ["M", "B", "T", "F"]
# CSV metric label -> (display label, macro suffix); column order = paper order.
METRIC_COLS = [
    ("Energy",        "Energy",        "Energy"),
    ("EDP",           "EDP",           "Edp"),
    ("Energy x Cost", "Energy×\\$",    "Ec"),
    ("EDP x Cost",    "EDP×\\$",       "Edpc"),
]
# v2 single-ablation config names keyed by disabled factor (for the sanity gate).
V2_NAMES = {
    "mem": "w/o Heterogeneous Memory",
    "batch": "w/o Heterogeneous Batching",
    "tp": "w/o Tensor Parallelism",
    "fusion": "w/o Layer Fusion",
}


def row_order():
    """All 2^4 disabled-subsets, by #disabled then factor order (driver order)."""
    out = []
    for k in range(len(FACTORS) + 1):
        for subset in itertools.combinations(FACTORS, k):
            out.append(frozenset(subset))
    return out


def load_summary(path):
    """-> {(frozenset(disabled), metric): row-dict with float fields}."""
    data = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            disabled = frozenset(fn for fn in FACTORS if r[fn] == "0")
            for k in ("norm_to_full", "decode_norm", "prefill_norm", "cnn_norm"):
                r[k] = float(r[k])
            data[(disabled, r["metric"])] = r
    return data


def sanity(data):
    """Structural + cross-run consistency checks. Prints exactly what it checked.

    Note on what is NOT a check: the all-on row's norm_to_full is *defined* as
    full_avg/full_avg in ablation_factorial.py, so it is 1.0 by construction and
    verifying it would be vacuous. It is therefore not asserted here.
    """
    order = row_order()
    n_combos = 0
    for mlabel, _, _ in [(m[0], None, None) for m in METRIC_COLS]:
        for d in order:
            assert (d, mlabel) in data, f"missing combo {sorted(d)} for {mlabel}"
        for d in order:
            v = data[(d, mlabel)]["norm_to_full"]
            assert math.isfinite(v), f"{sorted(d)}/{mlabel} non-finite: {v}"
            # A disabled technique can never HELP; >1% improvement means a
            # toggle leaked into the wrong direction. (Note this cannot detect a
            # toggle that is a complete no-op — that yields exactly 1.0 and
            # passes; the advisory scan below covers that case.)
            assert v > 0.99, (f"{sorted(d)} beats Full by >1% on {mlabel}: {v}"
                              " — toggle leak?")
            n_combos += 1
    # Single-disable rows must reproduce the published one-at-a-time run.
    v2 = {}
    with open(V2_SUMMARY) as f:
        for r in csv.DictReader(f):
            v2[(r["config"], r["metric"])] = float(r["norm_to_full"])
    # Tolerance. The per-network GA reseed in scripts/chiplet_sel.py
    # (deterministic_ga_rng) removed the old process-global-RNG run-order
    # dependence, so the factorial and one-at-a-time runs now agree to full
    # float precision on the shipped data (max observed relative difference:
    # 0.00e+00). The 3% band is kept only as a slack guard for reruns on other
    # hardware/BLAS; a real toggle leak or sign error shows up as 10%+ on the
    # technique's home regime, far outside it.
    max_rel = 0.0
    for fac, cname in V2_NAMES.items():
        for mlabel, _, _ in METRIC_COLS:
            got = data[(frozenset({fac}), mlabel)]["norm_to_full"]
            ref = v2[(cname, mlabel)]
            rel = abs(got - ref) / ref
            max_rel = max(max_rel, rel)
            assert rel < 0.03, \
                f"single-off {fac}/{mlabel}: factorial {got:.4f} != v2 {ref:.4f}"

    # Advisory: a toggle that changed NOTHING on any metric would be a wiring
    # bug rather than a small effect. Warn only — some techniques legitimately
    # sit at ~1.00 on a subset of metrics for this workload suite.
    noop = []
    for fac in FACTORS:
        vals = [data[(frozenset({fac}), m[0])]["norm_to_full"] for m in METRIC_COLS]
        if all(abs(v - 1.0) < 1e-6 for v in vals):
            noop.append(fac)
    for fac in noop:
        print(f"[sanity] WARNING: disabling '{fac}' changed no metric "
              f"(all four exactly 1.000000) — possible no-op toggle.")

    print(f"[sanity] CHECKED: {n_combos} combo x metric cells present and finite; "
          f"no combo beats Full by >1%; {len(V2_NAMES) * len(METRIC_COLS)} "
          f"single-off values match ablation_v2 (max rel diff {max_rel:.2e}, "
          "tol 3%).")
    print("[sanity] NOT CHECKED: the all-on row (1.0 by construction), and "
          "absolute magnitudes against the database.")


# Compact metric headers for the narrow two-block layout (the ×$ variants are
# spelled out in the caption/prose; here they must fit a half-width column).
METRIC_HEADERS_SHORT = ["Energy", "EDP", "E×\\$", "ED×\\$"]


def _draw_block(ax_ind, ax_hm, block_order, block_vals, norm, vmax, n_cols):
    """Render one half of the factorial: indicator (✓/✗) + value heatmap."""
    n = len(block_order)

    # --- indicator matrix (M B T F on/off) ---
    ax_ind.set_xlim(-0.5, len(FACTORS) - 0.5)
    ax_ind.set_ylim(n - 0.5, -0.5)
    for i, d in enumerate(block_order):
        for j, fac in enumerate(FACTORS):
            on = fac not in d
            ax_ind.text(j, i, "✓" if on else "✗",
                        ha="center", va="center", fontsize=6.0,
                        color="#1a7a3c" if on else "#b03030",
                        fontweight="bold" if not on else "normal")
    for j, h in enumerate(FACTOR_HEADERS):
        ax_ind.text(j, -0.95, h, ha="center", va="center", fontsize=6.5,
                    fontweight="bold")
    ax_ind.axis("off")

    # --- value heatmap ---
    clipped = [[max(v, 1.0) for v in row] for row in block_vals]
    ax_hm.imshow(clipped, aspect="auto", cmap="Reds", norm=norm)
    for i in range(n):
        for j in range(n_cols):
            v = block_vals[i][j]
            frac = (math.log(max(v, 1.0)) / math.log(vmax)) if vmax > 1 else 0.0
            ax_hm.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.0,
                       color="white" if frac > 0.55 else "black")
    ax_hm.set_xticks(range(n_cols))
    ax_hm.set_xticklabels(METRIC_HEADERS_SHORT, fontsize=6.2)
    ax_hm.xaxis.tick_top()
    ax_hm.tick_params(length=0)
    ax_hm.set_yticks([])
    for j in range(1, n_cols):
        ax_hm.axvline(j - 0.5, color="white", lw=0.8)
    # heavy separators wherever the #disabled group changes between rows
    for i in range(1, n):
        heavy = len(block_order[i]) != len(block_order[i - 1])
        ax_hm.axhline(i - 0.5, color="white", lw=1.3 if heavy else 0.4)
        if heavy:
            ax_ind.axhline(i - 0.5, color="#999999", lw=0.4)
    for spine in ax_hm.spines.values():
        spine.set_visible(False)


def make_figure(data, out_pdf, out_png):
    order = row_order()
    vals = [[data[(d, m[0])]["norm_to_full"] for m in METRIC_COLS] for d in order]
    vmax = max(max(r) for r in vals)
    n_cols = len(METRIC_COLS)

    # Two-block reflow: the 16 combos are split into two 8-row half-tables placed
    # side-by-side, spending the column's spare WIDTH (each metric cell only holds
    # a 4-char number) to roughly HALVE the figure's height/area — no content
    # dropped. Colorbar dropped (cells annotated); caption notes "darker = worse".
    half = len(order) // 2
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans",
                         "pdf.fonttype": 42})
    fig = plt.figure(figsize=(3.4, 1.32))
    gs = fig.add_gridspec(
        1, 5, width_ratios=[0.62, 2.55, 0.42, 0.62, 2.55], wspace=0.04,
        left=0.012, right=0.992, top=0.80, bottom=0.02)
    norm = LogNorm(vmin=1.0, vmax=vmax)
    for b in range(2):
        sl = slice(b * half, (b + 1) * half)
        ax_ind = fig.add_subplot(gs[0 if b == 0 else 3])
        ax_hm = fig.add_subplot(gs[1 if b == 0 else 4])
        _draw_block(ax_ind, ax_hm, order[sl], vals[sl], norm, vmax, n_cols)

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(f"[figure] -> {out_pdf}")


def fmt1(x):
    return f"{x:.1f}"


def pct1(x):
    p = (x - 1.0) * 100.0
    return f"{p:.1f}" if abs(p) >= 0.05 else "0.0"


def write_constants(data, path):
    """Every ablation number the prose quotes, 1-decimal, single source."""
    single = {fac: {msfx: data[(frozenset({fac}), mlabel)]
                    for mlabel, _, msfx in METRIC_COLS}
              for fac in FACTORS}
    alloff = {msfx: data[(frozenset(FACTORS), mlabel)]
              for mlabel, _, msfx in METRIC_COLS}
    fac_macro = {"mem": "Mem", "batch": "Batch", "tp": "Tp", "fusion": "Fusion"}

    lines = [
        "% AUTO-GENERATED by src/scripts/arch_impl/generate_ablation_heatmap.py"
        " — do not edit by hand.",
        "% fig:ablation (C3): full-factorial 2^4 ablation on the fixed N=8 pool,",
        "% normalized to Full Fengshui = 1.0 (>1 = worse). x = multiplier, "
        "Pct = % increase.",
    ]
    for fac in FACTORS:
        for _, _, msfx in METRIC_COLS:
            v = single[fac][msfx]["norm_to_full"]
            lines.append(f"\\newcommand{{\\Abl{fac_macro[fac]}{msfx}}}{{{fmt1(v)}}}")
            lines.append(
                f"\\newcommand{{\\Abl{fac_macro[fac]}{msfx}Pct}}{{{pct1(v)}}}")
    # headline per-category levers (quoted in prose)
    lines.append("\\newcommand{\\AblMemDecodeEdp}{%s}"
                 % fmt1(single["mem"]["Edp"]["decode_norm"]))
    lines.append("\\newcommand{\\AblMemPrefillEdp}{%s}"
                 % fmt1(single["mem"]["Edp"]["prefill_norm"]))
    lines.append("\\newcommand{\\AblMemCnnEc}{%s}"
                 % fmt1(single["mem"]["Ec"]["cnn_norm"]))
    lines.append("\\newcommand{\\AblTpPrefillEdp}{%s}"
                 % fmt1(single["tp"]["Edp"]["prefill_norm"]))
    lines.append("\\newcommand{\\AblFusionCnnEnergy}{%s}"
                 % fmt1(single["fusion"]["Energy"]["cnn_norm"]))
    lines.append("\\newcommand{\\AblFusionCnnEc}{%s}"
                 % fmt1(single["fusion"]["Ec"]["cnn_norm"]))
    # batching: worst-case % increase across the four metrics
    bmax = max(single["batch"][msfx]["norm_to_full"] for _, _, msfx in METRIC_COLS)
    lines.append(f"\\newcommand{{\\AblBatchMaxPct}}{{{pct1(bmax)}}}")
    # all-off combo + additivity (observed vs product of the four singles)
    for _, _, msfx in METRIC_COLS:
        v = alloff[msfx]["norm_to_full"]
        pred = 1.0
        for fac in FACTORS:
            pred *= single[fac][msfx]["norm_to_full"]
        lines.append(f"\\newcommand{{\\AblAllOff{msfx}}}{{{fmt1(v)}}}")
        lines.append(f"\\newcommand{{\\AblAllOff{msfx}Pred}}{{{fmt1(pred)}}}")
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[constants] -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=SUMMARY)
    ap.add_argument("--no-overleaf", action="store_true",
                    help="skip ONLY the best-effort copy into the external paper "
                         "tree; the figure and the ablation constants are always "
                         "written inside the artifact")
    args = ap.parse_args()

    data = load_summary(args.summary)
    sanity(data)
    out_pdf = os.path.join(OUT_DIR, "ablation_heatmap.pdf")
    out_png = os.path.join(OUT_DIR, "ablation_heatmap.png")
    make_figure(data, out_pdf, out_png)

    # The \Abl... macros ARE the ablation table's numbers, so they are generated
    # unconditionally into the artifact — never gated on the paper-tree copy.
    write_constants(data, LOCAL_CONST)

    if not args.no_overleaf:
        # Best effort: the Overleaf/paper tree is NOT shipped with the artifact.
        # Copy only into an existing destination tree; never crash if it is absent.
        import shutil
        for label, src, dst in (("figure", out_pdf, OVERLEAF_IMG),
                                ("constants", LOCAL_CONST, OVERLEAF_CONST)):
            dst_dir = os.path.dirname(os.path.abspath(dst))
            if not os.path.isdir(dst_dir):
                print(f"[{label}] paper tree {dst_dir} not present — skipping "
                      f"external copy (this is expected in the artifact; the "
                      f"in-artifact copy above is the one that matters).")
                continue
            shutil.copy(src, dst)
            print(f"[{label}] -> {dst}")

    # console digest for the rebuttal text
    print("\nInteraction digest (observed all-off vs product of singles):")
    for mlabel, _, msfx in METRIC_COLS:
        obs = data[(frozenset(FACTORS), mlabel)]["norm_to_full"]
        pred = 1.0
        for fac in FACTORS:
            pred *= data[(frozenset({fac}), mlabel)]["norm_to_full"]
        print(f"  {mlabel:14s} observed {obs:5.2f}x  product-of-singles {pred:5.2f}x"
              f"  ratio {obs/pred:4.2f}")


if __name__ == "__main__":
    main()
