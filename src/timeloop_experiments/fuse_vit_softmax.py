#!/usr/bin/env python3
"""
Fuse the 4 legacy ViT softmax sub-ops (layer0_softmax_{max,sub_exp,sum,div}) into a
single fused `layer0_softmax` row per config -- the same force-fusion that
database_builder.py applies to LLaMA/Qwen, which ViT missed because it was loaded via
convert_vit_xlsx.py (split into 4, never re-fused).  PURE POSTPROCESSING of rows already
in the DB -- no Timeloop.  Sequential softmax passes => SUM latency/dynamic_energy/
accesses; static_power/area/utilization are one vector unit => constant (take first).

Adding the fused row also fixes the silent load-time drop: db_layers is derived from the
DB, so once `layer0_softmax` exists for ViT, network_dataclass.load_from_dir stops
filtering it out and it lands back on the critical path.

Usage:
  python3 fuse_vit_softmax.py --db unified_database.csv --dry-run     # counts only
  python3 fuse_vit_softmax.py --db unified_database.csv --apply       # append fused rows
"""
import argparse, sys
import pandas as pd

SUB_OPS = ["layer0_softmax_max", "layer0_softmax_sub_exp",
           "layer0_softmax_sum", "layer0_softmax_div"]
FUSED = "layer0_softmax"
KEY = ["net", "batch_size", "sequence_length", "mapper_idx", "fused_layer_type",
       "tp_degree", "arch_target", "glb_scale", "pe_x_scale", "pe_y_scale",
       "dram_i", "dram_o"]
SUM_COLS = ["latency", "dynamic_energy", "i_access", "w_access", "o_access"]
FIRST_COLS = ["static_power", "area", "utilization"]


def load_subops(db):
    keep = []
    for chunk in pd.read_csv(db, chunksize=500000, low_memory=False):
        m = chunk["layer_name"].isin(SUB_OPS) & chunk["net"].astype(str).str.startswith("vit")
        if m.any():
            keep.append(chunk[m])
    return (pd.concat(keep, ignore_index=True) if keep
            else pd.DataFrame(columns=None))


def fuse(sub, header_cols):
    g = sub.groupby(KEY, sort=False)
    # GUARD: every config must contain exactly the 4 sub-ops
    sizes = g.size()
    bad = sizes[sizes != 4]
    if len(bad):
        print(f"  [WARN] {len(bad)} config-keys do NOT have exactly 4 sub-ops "
              f"(min={sizes.min()}, max={sizes.max()}); these would mis-sum.",
              file=sys.stderr)
    agg = {c: "sum" for c in SUM_COLS}
    agg.update({c: "first" for c in FIRST_COLS})
    fused = g.agg(agg).reset_index()
    fused["layer_name"] = FUSED
    # reorder to the DB header exactly
    return fused[header_cols], sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="unified_database.csv")
    ap.add_argument("--apply", action="store_true", help="append fused rows in place")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    header_cols = list(pd.read_csv(args.db, nrows=0).columns)
    sub = load_subops(args.db)
    if sub.empty:
        print("No ViT softmax sub-op rows found -- nothing to do.")
        return
    fused, sizes = fuse(sub, header_cols)

    print(f"DB: {args.db}")
    print(f"  ViT softmax sub-op rows read : {len(sub)}")
    print(f"  config-keys (rows/key)       : {len(sizes)} ({len(sub)/len(sizes):.1f})")
    print(f"  fused layer0_softmax rows out: {len(fused)}")
    print("  per-model fused rows:")
    print(fused.groupby("net").size().to_string().replace("\n", "\n    "))
    # spot-check
    sc = fused[fused.net == "vit_b16_s197"].iloc[0]
    print(f"  spot-check vit_b16 [{sc.fused_layer_type},tp{sc.tp_degree},"
          f"{sc.dram_i}/{sc.dram_o}]: latency={sc.latency:.4e}  dyn={sc.dynamic_energy:.4e}")

    if args.apply and not args.dry_run:
        fused.to_csv(args.db, mode="a", header=False, index=False)
        print(f"  APPLIED: appended {len(fused)} fused rows to {args.db}")
    else:
        print("  (dry-run: no rows written; pass --apply to append)")


if __name__ == "__main__":
    main()
