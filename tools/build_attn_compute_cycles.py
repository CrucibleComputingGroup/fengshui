#!/usr/bin/env python3
"""Build src/scripts/attn_compute_cycles.csv: Timeloop compute-only cycles of the KV ops.

The GQA correction (src/scripts/gqa_kv.py) re-applies postprocess_bw's roofline to the attention
rows with the corrected KV words, and cal_perf_phy_net._apply_attention_rows to attn_v's rebuilt
fused rows:
    latency = max(C, ceil((i + w) / bw_i), ceil(o / bw_o)) * cycle_time     (postprocess_bw.py:93-101)
C, the op's compute-only cycles on a chiplet, is not a database column.  It is the 'Cycles' line
of the infinite-bandwidth sweep (DRAM bandwidth x1000, run_sweep.py:449-453,
timeloop_helper.py:1067), whose stats run_sweep.py writes to (run_sweep.py:224-231)
    <outputs>/<net>/<op>/<batch>/1/<mapper_idx>/single/<otc>/<tp>/
        arch=<arch>@glb_scale=<g>@pe_x_scale=<x>@pe_y_scale=<y>/LPDDR5@LPDDR5/timeloop-mapper.stats.txt
This tool reads C for layer0_attn_qk and layer0_attn_v (batch 1, mapper 0: the only attention
rows in the database) for every network with stats, one value per
(net, layer_name, arch_target, glb_scale, pe_x_scale, pe_y_scale, tp_degree), then checks the
table against the database it will be used with: every non-PIM attn_qk / attn_v row, of every
network, fused type and DRAM pair, must satisfy the formula above with its own i/w/o_access, and
every key must be present.  The table is written only if the check passes.

The raw Timeloop outputs (90+ GB) are not shipped; the table is.  Usage:
    python3 tools/build_attn_compute_cycles.py \\
        --outputs /path/to/timeloop_experiments/outputs --db src/unified_database.csv
The database is streamed; only attention rows are parsed.
"""
import argparse
import collections
import csv
import glob
import io
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(_HERE), "src", "scripts")
sys.path.insert(0, SCRIPTS)

import gqa_kv                                                                     # noqa: E402
from global_parameter import cycle_time                                          # noqa: E402

PATH_RE = re.compile(r"/(?P<net>[^/]+)/(?P<layer>layer0_attn_(?:qk|v))/1/1/0/single/\d+/(?P<tp>\d+)/"
                     r"arch=(?P<arch>\w+)@glb_scale=(?P<glb>\d+)@pe_x_scale=(?P<px>\d+)@pe_y_scale=(?P<py>\d+)/"
                     r"LPDDR5@LPDDR5/timeloop-mapper\.stats\.txt$")
CYCLES_RE = re.compile(r"^Cycles:\s*(\d+)", re.M)
FIELDS = ["net", "layer_name", "arch_target", "glb_scale", "pe_x_scale", "pe_y_scale", "tp_degree",
          "compute_cycles"]


def read_stats(outputs):
    table, where = {}, {}
    for layer in gqa_kv.KV_OPS:
        pattern = os.path.join(outputs, "*", layer, "1", "1", "0", "single", "*", "*", "arch=*",
                               "LPDDR5@LPDDR5", "timeloop-mapper.stats.txt")
        for p in glob.glob(pattern):
            m = PATH_RE.search(p)
            if m is None:
                raise ValueError(f"unexpected stats path {p}")
            key = (m["net"], m["layer"], m["arch"], int(m["glb"]), int(m["px"]), int(m["py"]), int(m["tp"]))
            if key in table:
                raise ValueError(f"two stats files for {key}: {where[key]} and {p}")
            with open(p) as f:
                c = CYCLES_RE.search(f.read())
            if c is None:
                raise ValueError(f"no 'Cycles:' line in {p}")
            table[key], where[key] = int(c.group(1)), p
    if not table:
        raise FileNotFoundError(f"no layer0_attn_qk / layer0_attn_v stats under {outputs}")
    return table


def check_db(table, db):
    """Every non-PIM attention row of `db` must reproduce its latency from C and its own counts."""
    n, missing, worst = collections.Counter(), set(), 0.0
    needles = tuple(f",{layer}," for layer in gqa_kv.KV_OPS)
    with open(db, newline="") as f:
        header = next(csv.reader([f.readline()]))
        for line in f:
            if not any(s in line for s in needles):
                continue
            r = dict(zip(header, next(csv.reader(io.StringIO(line)))))
            if r["layer_name"] not in gqa_kv.KV_OPS or r["arch_target"] == "PIM":
                continue
            tp = int(r["tp_degree"])
            key = (r["net"], r["layer_name"], r["arch_target"], int(r["glb_scale"]),
                   int(r["pe_x_scale"]), int(r["pe_y_scale"]), tp)
            if key not in table:
                missing.add(key)
                continue
            cyc = gqa_kv.roofline_cycles(table[key], float(r["i_access"]), float(r["w_access"]),
                                         float(r["o_access"]), r["dram_i"], r["dram_o"], tp)
            d = abs(cyc * cycle_time / float(r["latency"]) - 1)
            worst = max(worst, d)
            if d > 1e-9:
                raise ValueError(f"{key} {r['dram_i']}@{r['dram_o']} {r['fused_layer_type']}: latency "
                                 f"{r['latency']} != max(C, DRAM bound) = {cyc} cycles")
            n[r["layer_name"]] += 1
    if missing:
        raise KeyError(f"{len(missing)} database chiplet keys have no stats, e.g. {sorted(missing)[:3]}")
    if not n:
        raise ValueError(f"no non-PIM attention rows in {db}")
    return n, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", required=True, help="Timeloop sweep outputs directory (run_sweep.py)")
    ap.add_argument("--db", required=True, help="unified database the table will be used with")
    ap.add_argument("--out", default=gqa_kv.COMPUTE_CYCLES_CSV, help="table to write (default: %(default)s)")
    a = ap.parse_args()
    table = read_stats(a.outputs)
    per_pair = collections.Counter(k[:2] for k in table)
    print(f"stats files: {len(table)}  (net, layer) pairs: {len(per_pair)}  "
          f"files per pair: {sorted(set(per_pair.values()))}")
    n, worst = check_db(table, a.db)
    print(f"database check: {dict(n)} non-PIM attention rows reproduce their latency "
          f"(worst rel. diff {worst:.1e}); no chiplet key missing")
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for k in sorted(table):
            w.writerow(list(k) + [table[k]])
    print(f"wrote {a.out} ({len(table)} rows)")


if __name__ == "__main__":
    main()
