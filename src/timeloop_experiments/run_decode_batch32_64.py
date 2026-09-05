#!/usr/bin/env python3
"""Stage B (stage-1): run Timeloop for ONLY the qwen3_30b_a3b decode_kv1024
literal-batch projection ops at batch {32,64}.

These are genuinely-new GEMM shapes (decode base_N=1 -> n_eff = batch = 32/64),
not coverable by the existing DB's n_eff set (min existing n_eff = 512). Attention,
softmax (batch=1 only) and experts (looked up at batch=1) need NOTHING, so they
are excluded. Reuses run_sweep.run_one_config so TP-dim / OTC / output-path /
parsing exactly match how every existing DB row was produced.

Run INSIDE Docker my_timeloop_env_v2 (pytimeloop), with LD_LIBRARY_PATH exported:
  cd /workspace/chiplet_timeloop/timeloop_experiments
  python3 run_decode_batch32_64.py --dry-run
  python3 run_decode_batch32_64.py --n-jobs 32            # real run -> populates outputs/
"""
import argparse
import os
import sys
import time

import joblib

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from run_sweep import (
    run_one_config, PE_COMBOS, ARCHS, GLB_SCALES, TP_DEGREES, N_MAPPER,
    WORKLOAD_DIR,
)

NET = "qwen3_30b_a3b_decode_kv1024"
# literal-batch projection ops (yaml basenames present in the decode dir).
# v_proj IS run directly from its yaml (no k_proj aliasing assumption).
OPS = ["layer0_q_proj", "layer0_k_proj", "layer0_v_proj", "layer0_o_proj",
       "lm_head", "router"]
# llama (dense): 7 projections; no router/lm_head in the llama DB/DAG.
LLAMA_OPS = ["layer0_q_proj", "layer0_k_proj", "layer0_v_proj", "layer0_o_proj",
             "layer0_gate_proj", "layer0_up_proj", "layer0_down_proj"]
NEW_BATCHES = [32, 64]
DRAM = {"I": "LPDDR5", "O": "LPDDR5"}   # infinite-BW sweep; stage-2 throttles to 16 combos


def build_configs(only_archs=None, only_batches=None, net=NET, ops=None):
    archs = only_archs if only_archs else ARCHS
    batches = only_batches if only_batches else NEW_BATCHES
    ops = ops if ops else OPS
    configs = []
    for op in ops:
        path = os.path.join(WORKLOAD_DIR, net, op + ".yaml")
        if not os.path.exists(path):
            print(f"  [WARN] missing yaml: {path}")
            continue
        for batch in batches:
            for arch in archs:
                for glb in GLB_SCALES:
                    for (px, py) in PE_COMBOS:
                        for tp in TP_DEGREES:
                            for mi in range(N_MAPPER):
                                configs.append((path, net, batch, arch, glb,
                                                px, py, DRAM, tp, mi))
    return configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="run only the first 2 configs to validate toolchain")
    ap.add_argument("--output-base-dir", default=_THIS_DIR)
    ap.add_argument("--only-archs", nargs="+", default=None,
                    help="restrict to these arch_targets (e.g. simba_like)")
    ap.add_argument("--only-batches", nargs="+", type=int, default=None,
                    help="restrict to these batches (e.g. 64)")
    ap.add_argument("--net", default=NET,
                    help="decode net to sweep (default: qwen3_30b_a3b_decode_kv1024)")
    ap.add_argument("--ops", nargs="+", default=None,
                    help="op yaml basenames; 'llama' = the 7 llama projections")
    args = ap.parse_args()

    ops = LLAMA_OPS if args.ops == ["llama"] else args.ops
    configs = build_configs(only_archs=args.only_archs, only_batches=args.only_batches,
                            net=args.net, ops=ops)
    print(f"NET={args.net}  ops={ops or OPS}  batches={NEW_BATCHES}  dram={DRAM}")
    print(f"grid: {len(ARCHS)} arch x {len(GLB_SCALES)} glb x {len(PE_COMBOS)} pe "
          f"x {len(TP_DEGREES)} tp x {N_MAPPER} mapper")
    print(f"Total configs: {len(configs)}")

    if args.dry_run:
        est = len(configs) / max(args.n_jobs, 1) * 25 / 60
        print(f"Estimated wall-time @ n_jobs={args.n_jobs}: ~{est:.1f} min "
              f"(tiny GEMMs likely faster)")
        return

    if args.smoke:
        configs = configs[:2]
        print(f"SMOKE: running {len(configs)} configs")

    t0 = time.time()
    results = joblib.Parallel(n_jobs=args.n_jobs, verbose=10)(
        joblib.delayed(run_one_config)(
            path, net, batch, arch, glb, px, py, dram, tp, mi,
            args.output_base_dir, remove_bw_limit=True,
        )
        for path, net, batch, arch, glb, px, py, dram, tp, mi in configs
    )
    ok = [r for r in results if r is not None]
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. {len(ok)}/{len(configs)} succeeded.")
    if ok:
        s = ok[0]
        print(f"sample row: op={s['operator']} b={s['batch_size']} "
              f"cycles={s['cycles']} energy_uj={s['energy_uj']:.4g} util={s['utilization']:.3f}")


if __name__ == "__main__":
    main()
