#!/usr/bin/env python3
"""
Finetune supplementary experiment for the train/test generalization study.

Idea
----
For each of the 5 held-out v2 splits and each of the 4 configs, take the
TRAIN-derived n=8 chiplet pool and "finetune" it on the TEST set: keep the 8
train chiplets as a fixed prefix and search ONE additional chiplet (n=8 -> 9)
with the framework's I-SAEO incremental extension -- the exact operator the
SAEO->I-SAEO chain already uses for n=7,8, just one more step, run on the TEST
set and seeded by the train pool. The resulting 9-chiplet pool is evaluated on
the test set; its value is the "finetuned cross".

This answers: does adding a single test-adapted chiplet close the
generalization gap between the frozen train pool (cross) and the test set's own
optimum (test_own)?

Why reconstructing the train n=8 pool from CSV is exact
-------------------------------------------------------
The result CSVs do not persist ChipletConfig.dram_type (only arch/glb/pe), but
dram_type is a no-op for the objective: evaluation recomputes the dram per
fusion-group from the bandwidth roofline (cal_perf_phy_net.get_buffer_config),
never reading the chiplet's dram_type. Verified bit-exact against the reported
cross. So `ChipletConfig.from_csv_for_n_chiplets` reconstruction is lossless.

What we record (per split, per config)
--------------------------------------
  - the finetuned 9-chiplet pool + which chiplet is NEW vs the train 8
  - the new (finetuned) cross on the test set
  - per test-network: value + the actual per-group dram (buffer_config)
  - baselines for comparison: n=8 cross (recomputed from the reconstructed
    pool, self-consistent), test_own at n=8 and (optionally) n=9

Run (inside the docker container, scripts/ import path handled here):
  python3 run_finetune.py \
    --input-root  <dir with <split>_v2/ subdirs> \
    --output-root <dir to write finetune results> \
    --database ../unified_database.csv
"""

import os
import re
import sys
import csv
import json
import time
import math
import copy
import argparse

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
PROJECT_DIR = os.path.normpath(os.path.join(THIS_DIR, ".."))
SCRIPTS_DIR = os.path.join(PROJECT_DIR, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
os.chdir(SCRIPTS_DIR)

import multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

from global_parameter import NET_DIR
from network_dataclass import VirtualNetwork
from cal_perf_phy_net import preload_database
from utility_functions import calculate_average_opt_value
from chiplet_sel import run_single_optimization
from chiplet_dataclass import ChipletConfig
import run_archgym_chiplet as rac

# The 5 held-out v2 splits and the 4 configs, mirroring run_train_test.py.
DEFAULT_SPLITS = ["moe_test_v2", "decode_test_v2", "vision_test_v2",
                  "seq2048_test_v2", "prefill_test_v2"]
CONFIGS = {
    "energy_nocost": ("energy", False),
    "energy_cost":   ("energy", True),
    "edp_nocost":    ("edp",    False),
    "edp_cost":      ("edp",    True),
}
FORBIDDEN_ARCH_TARGETS = frozenset({"feather"})

UNAME_RE = re.compile(r"^(?P<net>.+)_b(?P<b>\d+)_seq(?P<seq>\d+)$")


def parse_uname(u):
    m = UNAME_RE.match(u)
    if not m:
        raise ValueError(f"cannot parse virtual-net unique name: {u}")
    return m.group("net"), int(m.group("b")), int(m.group("seq"))


def build_test_nets(test_unames, db_layers_per_net):
    vnets = []
    for u in test_unames:
        net, b, seq = parse_uname(u)
        vn = VirtualNetwork(net, batch_size=b, sequence_length=seq)
        db_layers = db_layers_per_net.get(net)
        vn.load_from_dir(os.path.join(NET_DIR, net), db_layers=db_layers)
        if len(vn.layers) == 0:
            print(f"  [warn] {u} has no loadable layers, skipped")
            continue
        got = vn.get_unique_name()
        if got != u:
            print(f"  [warn] rebuilt name {got} != split name {u}")
        vnets.append(vn)
    return vnets


def chiplet_to_dict(c):
    return {"arch_target": c.arch_target,
            "glb_scale": c.global_buffer_size_scale,
            "pe_x_scale": c.pe_x_scale,
            "pe_y_scale": c.pe_y_scale}


def pool_identifiers(pool):
    return [c.get_identifier() for c in pool]


def assert_no_forbidden_arch_targets(pool, context):
    present = {
        str(getattr(c, "arch_target", "")).strip().lower() for c in pool
    }
    forbidden = sorted(present & FORBIDDEN_ARCH_TARGETS)
    if forbidden:
        raise RuntimeError(
            f"{context} contains forbidden architecture target(s): "
            + ", ".join(forbidden)
        )


def evaluate_pool(pool, test_nets, objective, cost_aware, database_file):
    """Fixed-pool evaluation on the test set; returns (geomean, per-net dict)."""
    _, results = run_single_optimization(
        virtual_nets=test_nets, chiplet_group=pool, objective=objective,
        results_file=database_file, cost_aware=cost_aware,
        use_sequential=True, n_workers=8,
        use_dag_cp=rac._USE_DAG_CP, v_het_batch=rac._V_HET_BATCH)
    geo, _ = calculate_average_opt_value(results, objective)
    per_net = {}
    for uname, r in results.items():
        gene = r.get("best_gene") or {}
        per_net[uname] = {
            "value": r.get("min_value", float("inf")),
            "latency": r.get("best_latency", float("inf")),
            # the actual per-fusion-group dram picked at evaluation time
            "buffer_config": gene.get("buffer_config") if isinstance(gene, dict) else None,
            "binary_string": gene.get("binary_string") if isinstance(gene, dict) else None,
        }
    return geo, per_net


def finetune_one(cfg_dir, work_dir, test_nets, objective, cost_aware, args):
    """Reconstruct the train n=8 pool, finetune +1 on the test set, evaluate."""
    train_csv = os.path.join(cfg_dir, "train_sweep_phase2_isaeo.csv")
    base_pool = ChipletConfig.from_csv_for_n_chiplets(args.base_n, train_csv)
    if len(base_pool) != args.base_n:
        raise ValueError(f"expected {args.base_n} base chiplets, got {len(base_pool)} "
                         f"from {train_csv}")
    assert_no_forbidden_arch_targets(base_pool, "base train pool")
    base_ids = pool_identifiers(base_pool)

    # --- baseline: n=8 cross from the reconstructed pool (self-consistent) ---
    base_cross, base_per_net = evaluate_pool(base_pool, test_nets, objective,
                                             cost_aware, args.database)

    # --- finetune: I-SAEO incremental extension n=base_n -> base_n+1 on TEST,
    #     seeded by the train n=8 pool. STRICT FREEZE: the 8 train chiplets are
    #     a fixed prefix and only the +1 chiplet is searched (phase2_frac=0
    #     disables I-SAEO's global fine-tune, so the train prefix is never
    #     altered -- a fixed tape-out + one added chiplet). ---
    saeo_kwargs = dict(candidates_per_round=args.saeo_candidates,
                       mutation_rate=args.saeo_mutation_rate,
                       phase2_frac=0.0)
    os.makedirs(work_dir, exist_ok=True)
    tmp_csv = os.path.join(work_dir, "finetune_sweep.csv")
    swept = rac.run_incremental_sweep(
        virtual_nets=test_nets, objective=objective, database_file=args.database,
        n_start=args.base_n + 1, n_end=args.base_n + 1,
        evals_per_n=args.n_evals, cost_aware=cost_aware, seed=args.seed,
        pruned_configs=None, include_pim=True, include_switch=True,
        algorithm='isaeo', algo_kwargs=saeo_kwargs, results_file=tmp_csv,
        initial_best_group=base_pool,
    )
    res = swept[args.base_n + 1]
    ft_pool = res["best_group"]
    assert_no_forbidden_arch_targets(ft_pool, "fine-tuned pool")
    ft_cross = res["best_value"]

    # per-net values + actual dram for the finetuned pool
    ft_per_net = {}
    for uname, r in res.get("network_results", {}).items():
        gene = r.get("best_gene") or {}
        ft_per_net[uname] = {
            "value": r.get("min_value", float("inf")),
            "latency": r.get("best_latency", float("inf")),
            "buffer_config": gene.get("buffer_config") if isinstance(gene, dict) else None,
            "binary_string": gene.get("binary_string") if isinstance(gene, dict) else None,
        }

    ft_ids = pool_identifiers(ft_pool)
    new_ids = ft_ids[args.base_n:]
    changed_prefix = [
        f"position {i + 1}: {base_id} -> {ft_id}"
        for i, (base_id, ft_id) in enumerate(zip(base_ids, ft_ids[:args.base_n]))
        if base_id != ft_id
    ]

    return {
        "base_n": args.base_n,
        "base_pool": [chiplet_to_dict(c) for c in base_pool],
        "base_pool_ids": base_ids,
        "base_cross": base_cross,
        "base_per_net": base_per_net,
        "finetune_n": args.base_n + 1,
        "finetune_pool": [chiplet_to_dict(c) for c in ft_pool],
        "finetune_pool_ids": ft_ids,
        "new_chiplet_ids": new_ids,
        "changed_prefix_ids": changed_prefix,
        "finetune_cross": ft_cross,
        "finetune_per_net": ft_per_net,
    }


def write_config_csv(out_dir, objective, test_unames, summary):
    """One CSV per config: the finetune recap + per-net values & dram."""
    os.makedirs(out_dir, exist_ok=True)
    # 1) per-net table
    pn_csv = os.path.join(out_dir, "finetune_per_net.csv")
    with open(pn_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["network", f"base_n{summary['base_n']}_{objective}",
                    f"finetune_n{summary['finetune_n']}_{objective}",
                    "improvement_ratio", "dram_buffer_config"])
        for u in test_unames:
            b = summary["base_per_net"].get(u, {}).get("value", float("inf"))
            ft = summary["finetune_per_net"].get(u, {}).get("value", float("inf"))
            ratio = (ft / b) if (b and math.isfinite(b) and b > 0
                                 and math.isfinite(ft)) else float("inf")
            bc = summary["finetune_per_net"].get(u, {}).get("buffer_config")
            w.writerow([u, b, ft, ratio,
                        "|".join(bc) if bc else ""])
    # 2) recap json
    with open(os.path.join(out_dir, "finetune_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-root", required=True,
                   help="Dir containing the <split>_v2/ result subdirs "
                        "(each with split.json + <config>/train_sweep_phase2_isaeo.csv)")
    p.add_argument("--output-root", required=True,
                   help="Dir to write finetune results into (mirrors split/config)")
    p.add_argument("--database", default=os.environ.get("FENGSHUI_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unified_database.csv")))
    p.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    p.add_argument("--base-n", type=int, default=8, help="Train pool size to finetune from")
    p.add_argument("--n-evals", type=int, default=500)
    p.add_argument("--saeo-candidates", type=int, default=2000)
    p.add_argument("--saeo-mutation-rate", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run: first split, edp_cost only, n_evals=20")
    args = p.parse_args()

    rac._USE_DAG_CP = True
    rac._V_HET_BATCH = True

    if args.smoke:
        args.splits = args.splits[:1]
        args.configs = ["edp_cost"]
        args.n_evals = min(args.n_evals, 20)

    search_targets = rac._build_arch_targets(True, True)
    forbidden_search_targets = sorted(
        {str(target).strip().lower() for target in search_targets}
        & FORBIDDEN_ARCH_TARGETS
    )
    if forbidden_search_targets:
        raise RuntimeError(
            "fine-tune candidate space contains forbidden architecture "
            "target(s): " + ", ".join(forbidden_search_targets)
        )
    print("Architecture targets (FEATHER excluded): "
          + ", ".join(search_targets))

    os.makedirs(args.output_root, exist_ok=True)
    grand = {"args": {k: getattr(args, k) for k in
                      ("base_n", "n_evals", "saeo_candidates",
                       "saeo_mutation_rate", "seed")},
             "splits": {}}

    for split in args.splits:
        split_dir = os.path.join(args.input_root, split)
        split_json = os.path.join(split_dir, "split.json")
        if not os.path.isfile(split_json):
            print(f"[skip] {split}: no split.json at {split_json}")
            continue
        info = json.load(open(split_json))
        test_unames = info["test"]
        print(f"\n{'#'*70}\n# SPLIT: {split}  ({len(test_unames)} test nets)\n{'#'*70}")

        # preload DB once per split (test nets shared across the 4 configs)
        needed = set(parse_uname(u)[0] for u in test_unames)
        print(f"Preloading DB for {len(needed)} networks...")
        t0 = time.perf_counter()
        preload_database(args.database, needed_nets=needed)
        print(f"  preload done in {time.perf_counter() - t0:.1f}s")

        from cal_perf_phy_net import CSV_CACHE
        df = CSV_CACHE.get(args.database)
        db_layers_per_net = ({net: set(g["layer_name"].unique())
                              for net, g in df.groupby("net", observed=True)}
                             if df is not None else {})
        test_nets = build_test_nets(test_unames, db_layers_per_net)
        built = [vn.get_unique_name() for vn in test_nets]

        grand["splits"][split] = {"test_nets": built, "configs": {}}

        for cfg_name in args.configs:
            cfg_dir = os.path.join(split_dir, cfg_name)
            if not os.path.isdir(cfg_dir):
                print(f"[skip] {split}/{cfg_name}: missing {cfg_dir}")
                continue
            objective, cost_aware = CONFIGS[cfg_name]
            print(f"\n--- {split}/{cfg_name} (obj={objective}, cost={cost_aware}) "
                  f"finetune n={args.base_n}->{args.base_n+1} ---")
            t0 = time.perf_counter()
            out_dir = os.path.join(args.output_root, split, cfg_name)
            summary = finetune_one(
                cfg_dir, out_dir, test_nets, objective, cost_aware, args
            )
            summary["time_s"] = time.perf_counter() - t0
            summary["objective"] = objective
            summary["cost_aware"] = cost_aware

            write_config_csv(out_dir, objective, built, summary)
            grand["splits"][split]["configs"][cfg_name] = {
                "base_cross": summary["base_cross"],
                "finetune_cross": summary["finetune_cross"],
                "new_chiplet_ids": summary["new_chiplet_ids"],
                "changed_prefix_ids": summary["changed_prefix_ids"],
                "time_s": summary["time_s"],
            }
            print(f"  base(n={args.base_n}) cross = {summary['base_cross']:.6e}")
            print(f"  finetune(n={args.base_n+1}) cross = {summary['finetune_cross']:.6e}")
            print(f"  new chiplet(s): {summary['new_chiplet_ids']}")
            if summary["changed_prefix_ids"]:
                print(f"  [note] I-SAEO phase2 also tweaked prefix: "
                      f"{summary['changed_prefix_ids']}")
            print(f"  ({summary['time_s']:.0f}s)")

            # write/refresh the grand summary incrementally (crash-safe)
            with open(os.path.join(args.output_root, "finetune_all.json"), "w") as f:
                json.dump(grand, f, indent=2)

    print(f"\nAll done. Results under {args.output_root}")


if __name__ == "__main__":
    main()
