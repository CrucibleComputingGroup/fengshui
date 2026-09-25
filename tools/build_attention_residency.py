#!/usr/bin/env python3
"""Build src/scripts/attention_residency.csv from the Timeloop maps behind the database.

The evaluator (cal_perf_phy_net._softmax_row_cached) reads it to decide whether the scores S
and the probabilities P stay in the GLB between attn_qk, softmax and attn_v, or spill to DRAM.

For attn_qk (it writes S: Timeloop Outputs) and attn_v (it reads P: Timeloop Inputs1), per
(net, arch, glb, pe_x, pe_y, tp):
  glb_tile        = the tensor's shared_glb tile
  resident_words  = glb_tile x the bounds of the DRAM-level B/H/Q/K loops at or inside the
                    outermost DRAM-level K or D loop. A DRAM-level K loop splits rows; a D loop
                    leaves S as partial sums or re-reads P. D itself adds no S or P words, so it
                    is not multiplied in.
  other_glb_words = the mapping's other shared_glb tiles
  need_words      = resident_words + other_glb_words
  fits            = need_words <= glb_scale x global_parameter.glb_base_word
                    (shared_glb depth 1048576, width 64, datawidth 16 in all four arch files:
                    arch/eyeriss_like/arch_bf.yaml:55-61, simba_like :48-54, gemmini_like :44-50,
                    simple_vector :42-48)

Maps: <outputs>/<net>/layer0_<op>/1/1/0/single/<n>/<tp>/arch=.../LPDDR5@LPDDR5/
timeloop-mapper.map.txt, written by run_sweep.py (the layout tools/build_attn_compute_cycles.py
reads the stats from). Only LPDDR5@LPDDR5 at batch 1 was mapped; the database's other DRAM pairs
re-price these same mappings.

Check (the run fails otherwise): every map with no DRAM-level K or D loop fits, as Timeloop's
own capacity check requires.

The raw Timeloop outputs (90+ GB) are not shipped; the table is.  Usage:
    python3 tools/build_attention_residency.py --outputs /path/to/timeloop_experiments/outputs
"""
import argparse
import collections
import csv
import glob
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(_HERE), "src", "scripts")
sys.path.insert(0, SCRIPTS)

from global_parameter import glb_base_word                                        # noqa: E402

LOOP = re.compile(r"for (\w+) in \[0:(\d+)(?:,\d+)?\)")
PATH = re.compile(r"/([^/]+)/layer0_\w+/1/1/0/single/\d+/(\d+)/arch=(\w+)@glb_scale=(\d+)"
                  r"@pe_x_scale=(\d+)@pe_y_scale=(\d+)/")
TENSOR = (("attn_qk", "Outputs"), ("attn_v", "Inputs1"))


def parse(path, tensor):
    """(DRAM-level loops outermost first, the tensor's GLB tile, resident words, other GLB words)."""
    t = open(path).read()
    top = t.split("shared_glb [")[0]
    loops = [(d, int(b)) for d, b in LOOP.findall(top)]
    tiles = {k: int(v) for k, v in re.findall(r"(\w+):(\d+)",
                                               re.search(r"shared_glb \[([^\]]*)\]", t).group(1))}
    tile = tiles.get(tensor)
    resident = tile
    idx = [i for i, (d, _) in enumerate(loops) if d in ("K", "D")]
    if tile is not None and idx:
        for d, b in loops[idx[0]:]:
            if d in ("B", "H", "Q", "K"):
                resident *= b
    others = sum(v for k, v in tiles.items() if k != tensor)
    return loops, tile, resident, others


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", required=True, help="Timeloop sweep outputs directory (run_sweep.py)")
    ap.add_argument("--out", default=os.path.join(SCRIPTS, "attention_residency.csv"),
                    help="table to write (default: %(default)s)")
    a = ap.parse_args()
    rows = []
    for op, tensor in TENSOR:
        pattern = f"{a.outputs}/*/layer0_{op}/1/1/0/single/*/*/arch=*/LPDDR5@LPDDR5/timeloop-mapper.map.txt"
        for p in sorted(glob.glob(pattern)):
            m = PATH.search(p)
            net, tp, arch = m.group(1), int(m.group(2)), m.group(3)
            g, px, py = int(m.group(4)), int(m.group(5)), int(m.group(6))
            loops, tile, resident, others = parse(p, tensor)
            if tile is None:
                raise ValueError(f"{p}: no {tensor} tile at shared_glb")
            need = resident + others
            rows.append(dict(op=op, net=net, arch_target=arch, glb_scale=g, pe_x_scale=px, pe_y_scale=py,
                             tp_degree=tp, dram_loops=" ".join(f"{d}{b}" for d, b in loops),
                             glb_tile=tile, resident_words=resident, other_glb_words=others,
                             need_words=need, glb_words=g * glb_base_word,
                             fits=int(need <= g * glb_base_word)))
    keys = [(r["op"], r["net"], r["arch_target"], r["glb_scale"], r["pe_x_scale"], r["pe_y_scale"],
             r["tp_degree"]) for r in rows]
    if not rows or len(set(keys)) != len(keys):
        raise ValueError(f"{len(rows)} maps, {len(keys) - len(set(keys))} duplicate keys")
    no_split = [r for r in rows if not any(d[0] in "KD" for d in r["dram_loops"].split())]
    bad = [r for r in no_split if not r["fits"]]
    if bad:
        raise ValueError(f"{len(bad)} maps with no DRAM-level K or D loop do not fit: {bad[:3]}")
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {a.out}: {collections.Counter(r['op'] for r in rows)} maps, "
          f"{len({r['net'] for r in rows})} nets")
    print(f"maps with no DRAM-level K or D loop: {len(no_split)}, all fit")
    c = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        ph = "prefill" if "prefill" in r["net"] else ("decode" if "decode" in r["net"] else "other")
        k = (r["op"], ph, r["arch_target"])
        c[k][0] += 1
        c[k][1] += 1 - r["fits"]
    for k in sorted(c):
        print(f"  {k}: maps {c[k][0]:5d}  do not fit {c[k][1]:4d}")


if __name__ == "__main__":
    main()
