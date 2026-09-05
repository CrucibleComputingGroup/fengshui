#!/usr/bin/env python3
"""
Build the C2 competing-frameworks comparison table (tab:competitors) for the paper.

Reads the iso-infrastructure C2 results and emits, normalized to Fengshui = 1.00:
  - overall geomean and "excl. decode" geomean per metric (the headline table)
  - the full MoE-split per-category geomeans (printed, for prose / appendix)

Sources (all under arch_impl/):
  competing_full_raw.csv          -> Fengshui (Full), SCAR-style, Gemini-style
  competing_homopernet_raw.csv    -> homo-per-net  (optional; included if present)

Run from .../chiplet_timeloop/scripts:  python arch_impl/build_competitor_table.py
"""
import os
import csv
import math
import glob

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

METRICS = [("Energy", "Energy"), ("Energy x Cost", "\\gls{ec}"),
           ("EDP", "\\gls{edp}"), ("EDP x Cost", "\\gls{edpc}")]

# Display order + LaTeX labels for the baseline rows (Fengshui is the 1.00 ref).
ROWS = [
    ("homo-per-net", "Homogeneous, per-network"),
    ("Gemini-style", "Homogeneous, all-net (Gemini-style)"),
    ("SCAR-style",   "Heterog.\\ dataflow, fixed (SCAR-style)"),
]

DECODE_CATS = {"decode", "decode-MoE"}


def categorize(net):
    n = net.lower()
    is_moe = "qwen" in n
    if "decode" in n:
        return "decode-MoE" if is_moe else "decode"
    if "prefill" in n:
        return "prefill-MoE" if is_moe else "prefill"
    if "mobilenet" in n or "replknet" in n or "resnet" in n or "vit" in n:
        return "cnn"
    return "other"


def geomean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v) and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")


def load(path, rows, skip_existing_fw=False):
    """Load per-net values. If skip_existing_fw, ignore frameworks already loaded
    (keeps a single, consistent Fengshui normalizer across runs given GA stochasticity)."""
    if not os.path.exists(path):
        return
    seen_fw = {fw for (_, fw) in rows} if skip_existing_fw else set()
    for r in csv.DictReader(open(path)):
        if r["framework"] in seen_fw:
            continue
        rows.setdefault((r["metric"], r["framework"]), {})[r["network"]] = float(r["value"])


# Homo-per-net reuses the ORIGINAL submission's per-network baseline: a fully
# customized design per workload, INCLUDING heterogeneous memory + batching
# (a per-workload custom chip would naturally customize memory too -- so this is
# the most generous homogeneous point, and needs no recompute).
METRIC_OBJCOST = {"Energy": ("energy", "False"), "Energy x Cost": ("energy", "True"),
                  "EDP": ("edp", "False"), "EDP x Cost": ("edp", "True")}


def load_original_pernet(rows):
    scripts_dir = os.path.dirname(THIS_DIR)  # chiplet_timeloop/scripts
    for metric, (obj, cost) in METRIC_OBJCOST.items():
        cands = glob.glob(os.path.join(scripts_dir, f"optimal_single_chiplet_{obj}_{cost}_*.csv"))
        if not cands:
            print(f"  [warn] no original per-net CSV for {metric} "
                  f"(optimal_single_chiplet_{obj}_{cost}_*.csv)")
            continue
        f = max(cands, key=os.path.getmtime)
        for r in csv.DictReader(open(f)):
            rows.setdefault((metric, "homo-per-net"), {})[r["network"]] = float(r[obj])
        print(f"  [homo-per-net] {metric}: {os.path.basename(f)}")


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Build tab:competitors constants (competitor_constants.tex) from a competing raw CSV")
    ap.add_argument(
        "--competing", default=os.path.join(THIS_DIR, "competing_ae_raw.csv"),
        help="Path to the competing raw CSV (Fengshui Full / SCAR-style / Gemini-style rows). "
             "Default: competing_ae_raw.csv (corrected DB + v7 pool; reproduces the "
             "committed competitor_constants.tex). NOTE: competing_full_raw.csv is the STALE "
             "pre-DRAM-fix run and competing_corrected_full_raw.csv is Fengshui-only -- do not "
             "use either for tab:competitors.")
    args = ap.parse_args()
    data = {}  # (metric, framework) -> {net: val}
    load(args.competing, data)
    load_original_pernet(data)  # homo-per-net from the original submission (het-mem + het-batch)

    frameworks_present = {fw for (_, fw) in data}
    have_pernet = "homo-per-net" in frameworks_present
    rows = [r for r in ROWS if r[0] in frameworks_present]

    def scope_gm(metric, fw, scope):
        d = data.get((metric, fw), {})
        if scope == "all":
            vals = list(d.values())
        else:  # excl decode
            vals = [v for n, v in d.items() if categorize(n) not in DECODE_CATS]
        return geomean(vals)

    # ---- printed numbers (overall, excl-decode, per category) ----
    cats = ["decode", "decode-MoE", "prefill", "prefill-MoE", "cnn"]
    print("\n=== Normalized to Fengshui (Full) = 1.00  (>1 = worse than Fengshui) ===")
    for metric, _ in METRICS:
        fg_all = scope_gm(metric, "Fengshui (Full)", "all")
        fg_ex = scope_gm(metric, "Fengshui (Full)", "exdec")
        print(f"\n### {metric}")
        print(f"{'baseline':40s} {'overall':>9s} {'excl.dec':>9s}  | per-cat (decode/dec-MoE/pf/pf-MoE/cnn)")
        for key, label in rows + [("Fengshui (Full)", "Fengshui (ours)")]:
            ov = scope_gm(metric, key, "all") / fg_all
            ex = scope_gm(metric, key, "exdec") / fg_ex
            fg_cat = {c: geomean([v for n, v in data.get((metric, "Fengshui (Full)"), {}).items()
                                  if categorize(n) == c]) for c in cats}
            pc = []
            for c in cats:
                g = geomean([v for n, v in data.get((metric, key), {}).items() if categorize(n) == c])
                pc.append(g / fg_cat[c] if fg_cat[c] and math.isfinite(fg_cat[c]) else float("nan"))
            print(f"{label:40s} {ov:>9.2f} {ex:>9.2f}  | " + " ".join(f"{x:6.1f}" for x in pc))

    # ---- emit \newcommand macros: SINGLE SOURCE OF TRUTH for tab:competitors + its prose ----
    # tab:competitors uses per-workload-class cells \Cmp{Fw}{Metric}{Pf|Dec|Cnn}; the prose uses
    # the range / dense-vs-MoE macros. Re-run this script to update every number in one place.
    FW_ABBR = {"Gemini-style": "Gem", "SCAR-style": "Scar", "homo-per-net": "Pn"}
    MET_ABBR = {"Energy": "En", "Energy x Cost": "Ec", "EDP": "Edp", "EDP x Cost": "Edpc"}
    COARSE = {"Pf": {"prefill", "prefill-MoE"}, "Dec": {"decode", "decode-MoE"}, "Cnn": {"cnn"}}

    def cell(metric, fw, scope):  # overall / excl-decode (for prose ranges)
        return scope_gm(metric, fw, scope) / scope_gm(metric, "Fengshui (Full)", scope)

    def cellcat(metric, fw, coarse):  # per workload-class ratio (table cells)
        cs = COARSE[coarse]
        fg = geomean([v for n, v in data.get((metric, "Fengshui (Full)"), {}).items() if categorize(n) in cs])
        g = geomean([v for n, v in data.get((metric, fw), {}).items() if categorize(n) in cs])
        return g / fg if fg and math.isfinite(fg) else float("nan")

    def percat(metric, fw, cat):  # single fine category (dense vs MoE decode, for prose)
        fg = geomean([v for n, v in data.get((metric, "Fengshui (Full)"), {}).items() if categorize(n) == cat])
        g = geomean([v for n, v in data.get((metric, fw), {}).items() if categorize(n) == cat])
        return g / fg if fg and math.isfinite(fg) else float("nan")

    def fmt(x):
        if not math.isfinite(x):
            return "--"
        if x >= 100:
            return f"{x:.0f}"
        return f"{x:.1f}" if x >= 10 else f"{x:.2f}"

    def rng(vals, dec):
        vals = [v for v in vals if math.isfinite(v)]
        return f"{min(vals):.{dec}f}--{max(vals):.{dec}f}"

    m = []
    m.append("% Auto-generated by chiplet_timeloop/scripts/arch_impl/build_competitor_table.py")
    m.append("% SINGLE SOURCE OF TRUTH for Table~\\ref{tab:competitors} and its prose. DO NOT hand-edit;")
    m.append("% re-run the script (reads arch_impl/competing_*_raw.csv + optimal_single_chiplet_*.csv).")
    m.append("% Geometric-mean ratios to Fengshui (>1 = worse than Fengshui).")
    for fw, ab in FW_ABBR.items():
        if fw not in frameworks_present:
            continue
        for metric, mab in MET_ABBR.items():
            for coarse in ("Pf", "Dec", "Cnn"):
                m.append(f"\\newcommand{{\\Cmp{ab}{mab}{coarse}}}{{{fmt(cellcat(metric, fw, coarse))}}}")
    twofw = ("Gemini-style", "SCAR-style")
    m.append(f"\\newcommand{{\\CmpEnRange}}{{{rng([cell('Energy', f, 'all') for f in twofw], 1)}}}")
    m.append(f"\\newcommand{{\\CmpEcRange}}{{{rng([cell('Energy x Cost', f, 'all') for f in twofw], 1)}}}")
    m.append(f"\\newcommand{{\\CmpEdpEdpcRange}}{{{rng([cell(mt, f, 'all') for mt in ('EDP', 'EDP x Cost') for f in twofw], 0)}}}")
    m.append(f"\\newcommand{{\\CmpScarEdpDecDense}}{{{percat('EDP', 'SCAR-style', 'decode'):.0f}}}")
    m.append(f"\\newcommand{{\\CmpScarEdpDecMoE}}{{{percat('EDP', 'SCAR-style', 'decode-MoE'):.0f}}}")
    text = "\n".join(m) + "\n"

    outs = [os.path.join(THIS_DIR, "competitor_constants.tex")]
    overleaf = os.path.normpath(os.path.join(
        THIS_DIR, "..", "..", "constants", "competitor_constants.tex"))
    if os.path.isdir(os.path.dirname(overleaf)):
        outs.append(overleaf)
    for o in outs:
        with open(o, "w") as f:
            f.write(text)
        print(f"[constants] wrote {o}")
    print(f"(homo-per-net {'INCLUDED' if have_pernet else 'MISSING -- run still pending'})")
    print("\n----- competitor_constants.tex -----\n" + text)


if __name__ == "__main__":
    main()
