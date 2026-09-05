#!/usr/bin/env python3
"""Final assembly: unified_database.csv = backup + prefill-copy rows + decode-new rows.

Validates that all three files share the EXACT same header (column order), then
streams them into one CSV. Additive only — every existing row is preserved verbatim;
we only append qwen3_30b_a3b prefill_s1024 / decode_kv1024 batch 32/64 rows.

Run from chiplet_timeloop/timeloop_experiments/ (env: mozart). Then propagate to
the root copy that scripts/ reads:  cp unified_database.csv ../unified_database.csv
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKUP = os.path.join(HERE, "unified_database_backup.csv")
PREFILL = os.path.join(HERE, "..", "case_study", "arch_impl", "new_rows_prefill.csv")
DECODE = os.path.join(HERE, "new_rows_decode.csv")
DECODE_SIMBA_INF = os.path.join(HERE, "new_rows_decode_simba_inf.csv")
OUT = os.path.join(HERE, "unified_database.csv")

_ap = argparse.ArgumentParser()
_ap.add_argument("--new-rows", nargs="+", default=None,
                 help="new-row CSVs to append onto the backup "
                      "(default: the qwen trio of prefill/decode/simba-inf)")
_args = _ap.parse_args()
NEW_ROW_FILES = _args.new_rows or [PREFILL, DECODE, DECODE_SIMBA_INF]


def header(path):
    with open(path, newline="") as f:
        return next(csv.reader(f))


def count_and_stream(path, writer, skip_header=True):
    n = 0
    with open(path, newline="") as f:
        r = csv.reader(f)
        if skip_header:
            next(r)
        for row in r:
            writer.writerow(row)
            n += 1
    return n


def main():
    srcs = [BACKUP] + NEW_ROW_FILES
    for p in srcs:
        if not os.path.isfile(p):
            print(f"MISSING: {p}")
            sys.exit(1)
    heads = [header(p) for p in srcs]
    if not all(h == heads[0] for h in heads):
        print("HEADER MISMATCH — aborting.")
        for p, h in zip(srcs, heads):
            print(" ", os.path.basename(p), ":", h)
        sys.exit(1)
    print(f"Headers match ({len(heads[0])} cols). Assembling -> {OUT}")
    total = 0
    with open(OUT, "w", newline="") as out:
        w = csv.writer(out)
        w.writerow(heads[0])
        nb = count_and_stream(BACKUP, w)
        total = nb
        print(f"  backup rows        : {nb}")
        for p in NEW_ROW_FILES:
            n = count_and_stream(p, w)
            total += n
            print(f"  {os.path.basename(p):<28}: {n}")
    print(f"  TOTAL              : {total}")
    print(f"Wrote {OUT}. Now propagate:  cp {OUT} ../unified_database.csv")


if __name__ == "__main__":
    main()
