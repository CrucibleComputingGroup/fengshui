#!/usr/bin/env python3
"""
Post-processing correction for the DRAM-energy double-count bug
(parse_stats.py:622) WITHOUT rebuilding the database from outputs/.

Background
----------
Timeloop's reported dynamic_energy ALREADY embeds the DRAM-access energy at
`timeloop_e` pJ/bit (parse_stats sums the `=== DRAM_I/O ===` sections into the
total).  `post_process_mapping_results` then re-added a DRAM copy at `final_e`
WITHOUT removing the embedded one, so every row's dynamic_energy is

    buggy = correct + D,      where
    D = (i_access + w_access) * final_e[dram_i] * word_size * 1e-12
      +  o_access             * final_e[dram_o] * word_size * 1e-12

Because timeloop_e == final_e for every modeled DRAM type, `D` here reproduces
the spurious term EXACTLY from the DB columns.  Subtracting it gives the
corrected dynamic_energy = compute + buffers + one DRAM copy.

Safety
------
* PIM rows carry blank i/w/o_access -> D = 0 -> unchanged (PIM energy is
  power*latency, never went through the buggy path).
* Ideal DRAM has final_e = 0 -> D = 0 -> unchanged.
* inf rows (infeasibility markers) stay inf.
* Every other column is preserved byte-for-byte (read as str), so the corrected
  CSV is a drop-in replacement for the lookups in cal_perf_phy_net.py.

Usage
-----
    python arch_impl/correct_dram_double_count.py \
        --input ../unified_database.csv \
        --output ../unified_database.dramfix.csv
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from global_parameter import dram_type_bandwidth_width_dict as DRAM, word_size  # noqa: E402

FINAL_E = {k: float(v["final_e"]) for k, v in DRAM.items()}


def dram_term(chunk, source_dram):
    """Spurious double-counted DRAM-access energy (J) per row, from DB columns.

    CRITICAL: the spurious term is the energy of the SWEEP SOURCE DRAM, NOT the
    row's own (target) DRAM type.  Every Timeloop sweep was run at a single
    source (`outputs/` only has LPDDR5@LPDDR5; postprocess_bw SOURCE_DRAM='LPDDR5'),
    so the bug embedded + re-added one copy of the LPDDR5-energy term, and the
    analytical DRAM-type expansion (postprocess_bw) swapped only the *legitimate*
    copy's per-bit energy.  Hence the leftover spurious copy is ALWAYS valued at
    source_dram's pJ/bit, regardless of the row's dram_i/dram_o.  Verified against
    shipped rows: GDDR7@GDDR7 = C + D_LPDDR5 + D_GDDR7 (not C + 2*D_GDDR7), etc.
    """
    fe = FINAL_E[source_dram]            # source pJ/bit (= timeloop_e == final_e)
    i = pd.to_numeric(chunk["i_access"].replace("", "0"), errors="coerce").fillna(0.0)
    w = pd.to_numeric(chunk["w_access"].replace("", "0"), errors="coerce").fillna(0.0)
    o = pd.to_numeric(chunk["o_access"].replace("", "0"), errors="coerce").fillna(0.0)
    # source sweep was <SRC>@<SRC>, so both i/w and o sides used the same pJ/bit
    return (i + w + o) * fe * word_size * 1e-12


def classify(net):
    n = str(net).lower()
    if "decode" in n:
        return "decode"
    if "prefill" in n:
        return "prefill"
    if any(c in n for c in ("mobilenet", "replknet", "vit", "efficientnet", "resnet")):
        return "cnn"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source-dram", default="LPDDR5",
                    help="DRAM type the Timeloop sweep was run at (all outputs are "
                         "LPDDR5@LPDDR5). The double-counted term is valued at THIS "
                         "pJ/bit for every row regardless of the row's own dram.")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args()
    if args.source_dram not in FINAL_E:
        sys.exit(f"ERROR: unknown --source-dram {args.source_dram}; known={list(FINAL_E)}")

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        sys.exit("ERROR: refuse to overwrite the input in place; choose a different --output")

    total = 0
    changed = 0
    neg_after = 0
    # per-class accumulators for the ratio old/new (inflation actually removed)
    cls = {c: {"n": 0, "sum_old": 0.0, "sum_new": 0.0, "max_ratio": 0.0} for c in
           ("decode", "prefill", "cnn", "other")}

    first = True
    for chunk in pd.read_csv(args.input, chunksize=args.chunksize, dtype=str,
                             keep_default_na=False):
        D = dram_term(chunk, args.source_dram)
        old = pd.to_numeric(chunk["dynamic_energy"], errors="coerce")
        new = old - D

        fin = np.isfinite(new.to_numpy())
        neg_after += int(np.sum(fin & (new.to_numpy() < -1e-15)))
        changed += int(np.sum((D.to_numpy() > 0) & fin))

        # per-class stats on rows that actually changed and are finite/positive
        klass = chunk["net"].map(classify)
        good = fin & (D.to_numpy() > 0) & (new.to_numpy() > 0) & np.isfinite(old.to_numpy())
        for c in cls:
            m = good & (klass.to_numpy() == c)
            if m.any():
                o_ = old.to_numpy()[m]
                n_ = new.to_numpy()[m]
                cls[c]["n"] += int(m.sum())
                cls[c]["sum_old"] += float(o_.sum())
                cls[c]["sum_new"] += float(n_.sum())
                cls[c]["max_ratio"] = max(cls[c]["max_ratio"], float((o_ / n_).max()))

        # write corrected dynamic_energy at full round-trippable precision
        chunk["dynamic_energy"] = [
            ("inf" if math.isinf(v) else ("nan" if math.isnan(v) else f"{v:.17g}"))
            for v in new.to_numpy()
        ]
        chunk.to_csv(args.output, mode="w" if first else "a", header=first, index=False)
        first = False
        total += len(chunk)
        print(f"  ...{total:,} rows", flush=True)

    print("\n===== correction summary =====")
    print(f"total rows           : {total:,}")
    print(f"rows changed (D>0)   : {changed:,}")
    print(f"negative after fix   : {neg_after}   (should be 0)")
    print(f"source_dram={args.source_dram} ({FINAL_E[args.source_dram]} pJ/bit)  "
          f"word_size={word_size}")
    print("\nper-class aggregate inflation removed (sum_old / sum_new):")
    for c, s in cls.items():
        if s["n"]:
            print(f"  {c:8s}: n={s['n']:>10,}  agg_ratio={s['sum_old']/s['sum_new']:.3f}  "
                  f"max_row_ratio={s['max_ratio']:.3f}")


if __name__ == "__main__":
    main()
