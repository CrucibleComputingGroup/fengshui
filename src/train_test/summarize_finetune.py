#!/usr/bin/env python3
"""Collect the 20 strict-freeze finetune runs into a tidy result tree + master
table, joined against the existing cross / test_own numbers.

With no arguments the historical in-tree locations are used, so existing
command-line usage is unchanged. Pass explicit roots to summarize an isolated
``finetune_runs/<run-id>/`` produced by ``run_finetune_all.sh``.
"""
import argparse
import os, json, csv, glob, shutil, math

THIS = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default=os.path.join(THIS, "finetune_raw"),
        help="Tree containing <split>__<config>/<split>/<config>/"
             "finetune_summary.json (default: %(default)s)",
    )
    parser.add_argument(
        "--gen-csv",
        default=os.path.join(THIS, "generalization_v2_cross.csv"),
        help="Cross-eval CSV supplying test_own/cross at n=8; written by "
             "plot_v2_cross.py (default: %(default)s)",
    )
    parser.add_argument(
        "--constants-out",
        default=None,
        help="Optional path for a LaTeX \\newcommand constants file "
             "(generalization_constants.tex) holding every number the paper's "
             "C1 prose cites. Omitted by default.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(THIS, "finetune_results"),
        help="Destination for the tidy tree and finetune_master_table.csv "
             "(default: %(default)s)",
    )
    return parser.parse_args()


args = parse_args()
RAW = os.path.abspath(os.path.expanduser(args.raw_root))
OUT = os.path.abspath(os.path.expanduser(args.out_dir))
GEN = os.path.abspath(os.path.expanduser(args.gen_csv))

SPLITS = ["moe_test_v2", "decode_test_v2", "vision_test_v2",
          "seq2048_test_v2", "prefill_test_v2"]
CONFIGS = ["energy_nocost", "energy_cost", "edp_nocost", "edp_cost"]

# existing cross + test_own (n=8)
gen = {}
with open(GEN) as f:
    for r in csv.DictReader(f):
        gen[(r["split"], r["config"])] = (float(r["cross"]), float(r["test_own"]))

os.makedirs(OUT, exist_ok=True)
rows = []
for split in SPLITS:
    for cfg in CONFIGS:
        sj = os.path.join(RAW, f"{split}__{cfg}", split, cfg, "finetune_summary.json")
        pn = os.path.join(RAW, f"{split}__{cfg}", split, cfg, "finetune_per_net.csv")
        if not os.path.isfile(sj):
            print(f"[miss] {split}/{cfg}")
            continue
        d = json.load(open(sj))
        # tidy copy
        dst = os.path.join(OUT, split, cfg)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(sj, os.path.join(dst, "finetune_summary.json"))
        if os.path.isfile(pn):
            shutil.copy(pn, os.path.join(dst, "finetune_per_net.csv"))

        base = d["base_cross"]          # n=8 train pool on test (== existing cross)
        ft = d["finetune_cross"]        # n=9 finetuned pool on test
        cross8, testown8 = gen.get((split, cfg), (float("nan"), float("nan")))
        new9 = d["new_chiplet_ids"][0] if d["new_chiplet_ids"] else ""
        chg = len(d["changed_prefix_ids"])
        ratio_ft_cross = ft / base if base else float("nan")
        ratio_ft_testown = ft / testown8 if testown8 else float("nan")
        # generalization gap closed: cross -> test_own is the gap; how much of it
        # the finetune recovers (can exceed 100% if finetune beats test_own's n=8).
        gap = base - testown8
        closed = ((base - ft) / gap * 100.0) if gap and abs(gap) > 0 else float("nan")
        rows.append(dict(split=split, config=cfg,
                         test_own_n8=testown8, cross_n8=base, finetune_n9=ft,
                         ft_vs_cross=ratio_ft_cross, ft_vs_testown=ratio_ft_testown,
                         gap_closed_pct=closed, new_9th=new9, chg_prefix=chg))

if not rows:
    raise FileNotFoundError(
        f"no finetune_summary.json files found under raw root: {RAW}"
    )

# master CSV
master = os.path.join(OUT, "finetune_master_table.csv")
with open(master, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)

# console table
print(f"\nWrote {master}  ({len(rows)} rows)\n")
hdr = f"{'split':16s} {'config':13s} {'test_own':>10s} {'cross(n8)':>10s} {'ft(n9)':>10s} {'ft/cross':>8s} {'gap%':>6s} {'chg':>3s}  new_9th"
print(hdr); print("-"*len(hdr))
for r in rows:
    print(f"{r['split']:16s} {r['config']:13s} {r['test_own_n8']:10.3e} {r['cross_n8']:10.3e} "
          f"{r['finetune_n9']:10.3e} {r['ft_vs_cross']:8.3f} {r['gap_closed_pct']:6.0f} {r['chg_prefix']:3d}  {r['new_9th']}")

# integrity check
bad = [r for r in rows if r["chg_prefix"] != 0]
print(f"\nstrict-freeze integrity: {len(rows)-len(bad)}/{len(rows)} have chg_prefix==0",
      "(ALL OK)" if not bad else f"-- VIOLATIONS: {[(r['split'],r['config']) for r in bad]}")


# ---------------------------------------------------------------- constants --
def _floor1(x):
    """Round down to 1 decimal: safe for the LOW end of a reported range."""
    return math.floor(x * 10.0) / 10.0


def _ceil1(x):
    """Round up to 1 decimal: safe for the HIGH end of a reported range."""
    return math.ceil(x * 10.0) / 10.0


def emit_constants(rows, path):
    r"""Write the \Gen... macros behind the paper's C1 prose.

    Rounding is directional so every macro keeps its sentence true: range
    lower bounds round down, upper bounds round up.
    """
    by = {(r["split"], r["config"]): r for r in rows}
    cross = lambda s_, c_: by[(s_, c_)]["cross_n8"] / by[(s_, c_)]["test_own_n8"]

    energy_cfgs = ("energy_nocost", "energy_cost")
    edp_cfgs = ("edp_nocost", "edp_cost")
    non_decode = [s_ for s_ in SPLITS if s_ != "decode_test_v2"]

    # transfer quality on the four non-decode families, energy + energy x $
    xfer = [cross(s_, c_) for s_ in non_decode for c_ in energy_cfgs
            if (s_, c_) in by]
    # fine-tuned n=9 vs the held-out family's own optimum, all 20 cells
    ft = [r["ft_vs_testown"] for r in rows]
    dec_gap = [r["gap_closed_pct"] for r in rows
               if r["split"] == "decode_test_v2"]

    macros = [
        ("GenXferLoPct", f"{_floor1((min(xfer) - 1) * 100):.1f}"),
        ("GenXferHiPct", f"{_ceil1((max(xfer) - 1) * 100):.1f}"),
        ("GenDecodeEnergyMax",
         f"{_ceil1(max(cross('decode_test_v2', c_) for c_ in energy_cfgs)):.1f}"),
        ("GenDecodeEdpLo",
         f"{_floor1(min(cross('decode_test_v2', c_) for c_ in edp_cfgs)):.1f}"),
        ("GenDecodeEdpHi",
         f"{_ceil1(max(cross('decode_test_v2', c_) for c_ in edp_cfgs)):.1f}"),
        ("GenMoeEdpMax", f"{_ceil1(cross('moe_test_v2', 'edp_nocost')):.1f}"),
        ("GenMoeEdpcMax", f"{_ceil1(cross('moe_test_v2', 'edp_cost')):.1f}"),
        ("GenVisionEdpMax",
         f"{_ceil1(cross('vision_test_v2', 'edp_nocost')):.1f}"),
        ("GenDecodeGapClosedMin", f"{math.floor(min(dec_gap)):d}"),
        ("GenFtMaxPct", f"{_ceil1((max(ft) - 1) * 100):.1f}"),
    ]

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("% AUTO-GENERATED by src/train_test/summarize_finetune.py "
                "--constants-out - do not edit by hand.\n")
        f.write("% Figure fig:generalization (C1): held-out transfer + "
                "prefix-frozen n=8->9 fine-tuning.\n")
        for name, val in macros:
            f.write("\\newcommand{\\%s}{%s}\n" % (name, val))
    print(f"\nWrote {path}")
    for name, val in macros:
        print(f"  \\{name} = {val}")


if args.constants_out:
    emit_constants(rows, os.path.abspath(os.path.expanduser(args.constants_out)))
