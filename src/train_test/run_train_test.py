#!/usr/bin/env python3
"""
Train/Test generalization experiment for the chiplet-timeloop framework.

Pipeline
--------
1. Enumerate the workloads under ``workloads/``, EXCLUDING the families
   efficientnet_b0, gpt, resnet, vit.
2. Treat each (workload, batch) pair as one virtual network (batch in {1, 8},
   matching the framework's default).  Randomly split the resulting virtual
   networks 70/30 into a TRAIN set and a TEST set (fixed seed).
3. For each of the 4 objective/cost configurations
   (energy / edp)  x  (with cost / without cost):
     a. TRAIN  : run our framework (incremental chiplet-pool sweep n=1..N)
                 on the TRAIN set    -> per-n optimal pool + objective.
     b. TEST   : run our framework on the TEST set
                 -> per-n optimal pool + objective (the test set's *own* optimum).
     c. CROSS  : take the pool produced on the TRAIN set and evaluate it on the
                 TEST set WITHOUT changing the pool (fixed tape-out / ASIC; only
                 the per-workload inner mapping is re-optimised).  This measures
                 how well a TRAIN-derived pool generalises to unseen workloads.

Outputs (under ``train_test/results/<timestamp>/``)
   split.json                          the train/test split + seed
   <config>/train_sweep.csv            framework result on TRAIN set
   <config>/test_sweep.csv             framework result on TEST set
   <config>/cross_eval.csv             TRAIN pool evaluated on TEST set, per n
   <config>/summary.json               compact summary for the config
   summary_all.json                    summary across all configs

This script imports the framework's own optimisation routines, so it stays in
lock-step with the rest of the codebase (no logic is re-implemented).
"""

import os
import sys
import json
import time
import math
import random
import argparse

# --- Make the framework importable (scripts/ holds all the modules + relative
#     data files such as network_analysis.csv) ---------------------------------
THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
PROJECT_DIR = os.path.normpath(os.path.join(THIS_DIR, ".."))
SCRIPTS_DIR = os.path.join(PROJECT_DIR, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
# Several framework helpers read relative paths (e.g. network_analysis.csv);
# run everything with scripts/ as the working directory, exactly like the
# framework's own entry points expect.
os.chdir(SCRIPTS_DIR)

import multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

from global_parameter import NET_DIR
from network_dataclass import VirtualNetwork
from cal_perf_phy_net import preload_database, CSV_CACHE
from utility_functions import calculate_average_opt_value
from chiplet_sel import run_single_optimization
from chiplet_dataclass import ChipletConfig
import run_archgym_chiplet as rac
from chiplet_pruning import get_pruned_configs
import glob as _glob

# Families to exclude (prefix match on the workload directory name).
# ViT is now INCLUDED (the 3 vit_* workloads); efficientnet/gpt/resnet stay out.
# Workload families excluded from the split. NOTE: this list is the single source of
# truth -- keep README.md in sync with it. Override for sensitivity runs with
#   FENGSHUI_EXCLUDE_PREFIXES=efficientnet,gpt,resnet,vit
EXCLUDE_PREFIXES = tuple(
    p.strip() for p in os.environ.get(
        "FENGSHUI_EXCLUDE_PREFIXES", "efficientnet,gpt,resnet").split(",") if p.strip())

# The 4 configurations: (objective, cost_aware) -> short name.
CONFIGS = [
    ("energy", False, "energy_nocost"),
    ("energy", True,  "energy_cost"),
    ("edp",    False, "edp_nocost"),
    ("edp",    True,  "edp_cost"),
]

FORBIDDEN_ARCH_TARGETS = frozenset({"feather"})


def assert_no_forbidden_arch_targets(chiplet_group, context):
    """Fail rather than silently admitting an excluded architecture."""
    present = {
        str(getattr(chiplet, "arch_target", "")).strip().lower()
        for chiplet in chiplet_group
    }
    forbidden = sorted(present & FORBIDDEN_ARCH_TARGETS)
    if forbidden:
        raise RuntimeError(
            f"{context} contains forbidden architecture target(s): "
            + ", ".join(forbidden)
        )


# ---------------------------------------------------------------------------
# Workload enumeration / virtual-network construction
# ---------------------------------------------------------------------------
def seq_len_for(name):
    """Sequence length to feed VirtualNetwork, matching setup_virtual_nets()."""
    if "prefill_s" in name:
        try:
            return int(name.split("prefill_s")[1].split("_")[0])
        except (IndexError, ValueError):
            return 1
    # decode (kv-cache, generates one token) and CNNs use sequence_length = 1.
    return 1


def list_workloads(drop_seqs=()):
    """Return the sorted list of workload directory names, minus excluded ones.

    drop_seqs: sequence lengths to drop (e.g. [1024, 4096] removes both the
    prefill_s{N} and decode_kv{N} variants for each N) — fewer workloads, cheaper
    per-evaluation. CNN/ViT (no seq token) are unaffected."""
    drop_tokens = set()
    for s in drop_seqs:
        drop_tokens.add(f"s{s}")
        drop_tokens.add(f"kv{s}")
    names = []
    for entry in sorted(os.listdir(NET_DIR)):
        full = os.path.join(NET_DIR, entry)
        if not os.path.isdir(full):
            continue
        if entry.startswith(EXCLUDE_PREFIXES):
            continue
        if any(tok in entry for tok in drop_tokens):
            continue
        # A real workload dir contains per-layer YAMLs (NETWORK.yaml alone is
        # not enough). This also drops stray dirs like an inner "workloads/".
        has_layer = any(
            f.endswith(".yaml") and f != "NETWORK.yaml" for f in os.listdir(full)
        )
        if has_layer:
            names.append(entry)
    return names


def layers_from_preloaded(database_file):
    """Per-net set of layer names, taken from the already-preloaded DB cache.

    Avoids a second full read of the 4.8 GB CSV — ``preload_database`` already
    parsed it into ``CSV_CACHE`` (filtered to the needed networks)."""
    df = CSV_CACHE.get(database_file)
    if df is None:
        return {}
    return {
        net: set(grp["layer_name"].unique())
        for net, grp in df.groupby("net", observed=True)
    }


def build_virtual_nets(workload_names, batches, db_layers_per_net):
    """Construct + load VirtualNetwork objects for every (workload, batch) pair."""
    specs = []
    for name in workload_names:
        seq = seq_len_for(name)
        for b in batches:
            specs.append((name, b, seq))

    vnets = []
    for name, b, seq in specs:
        vn = VirtualNetwork(name, batch_size=b, sequence_length=seq)
        db_layers = db_layers_per_net.get(name, None)
        try:
            vn.load_from_dir(os.path.join(NET_DIR, name), db_layers=db_layers)
        except Exception as e:  # pragma: no cover - defensive
            print(f"  [warn] failed to load {name}: {e}")
        if len(vn.layers) > 0:
            vnets.append(vn)
        else:
            print(f"  [warn] {name} (b{b}) has no loadable layers, skipped")
    return vnets


# ---------------------------------------------------------------------------
# Framework runners
# ---------------------------------------------------------------------------
def run_sweep(vnets, objective, cost_aware, database_file, args, results_file,
              pruned_configs):
    """Run the framework's incremental chiplet-pool sweep on a set of nets."""
    algo_kwargs = {}
    if args.algorithm == "sa":
        algo_kwargs = dict(initial_temp=args.sa_initial_temp,
                           cooling_rate=args.sa_cooling_rate,
                           min_temp=args.sa_min_temp)
    elif args.algorithm in ("saeo", "isaeo"):
        algo_kwargs = dict(candidates_per_round=args.saeo_candidates,
                           mutation_rate=args.saeo_mutation_rate)
    return rac.run_incremental_sweep(
        virtual_nets=vnets,
        objective=objective,
        database_file=database_file,
        n_start=args.n_start,
        n_end=args.n_end,
        evals_per_n=args.n_evals,
        cost_aware=cost_aware,
        seed=args.seed,
        pop_size=args.ga_pop_size,
        mutation_rate=args.ga_mutation_rate,
        pruned_configs=pruned_configs,
        include_pim=args.pim,
        include_switch=args.switch,
        algorithm=args.algorithm,
        algo_kwargs=algo_kwargs,
        results_file=results_file,
    )


def run_chain(vnets, objective, cost_aware, database_file, args, results_file,
              pruned_configs):
    """The paper's proposed method: SAEO (global, n=n_start..saeo_end) chained
    into I-SAEO (prefix-fixed incremental, n=saeo_end+1..n_end).

    Mirrors ``run_saeo_isaeo_chain.py`` but returns the merged per-n result dict
    so the caller can reuse the train-derived pool for cross-evaluation."""
    base, ext = os.path.splitext(results_file)
    saeo_end = min(args.saeo_end, args.n_end)

    # Exploration knobs forwarded to the SAEO/I-SAEO optimizers (the chain's
    # actual search). pop_size/mutation_rate below are GA-only and ignored by
    # SAEO/I-SAEO; these algo_kwargs are what genuinely reach them.
    saeo_kwargs = dict(candidates_per_round=args.saeo_candidates,
                       mutation_rate=args.saeo_mutation_rate)

    # Phase 1: SAEO global search, n = n_start .. saeo_end
    phase1 = rac.run_incremental_sweep(
        virtual_nets=vnets, objective=objective, database_file=database_file,
        n_start=args.n_start, n_end=saeo_end,
        evals_per_n=args.n_evals, cost_aware=cost_aware, seed=args.seed,
        pop_size=args.ga_pop_size, mutation_rate=args.ga_mutation_rate,
        pruned_configs=pruned_configs,
        include_pim=args.pim, include_switch=args.switch,
        algorithm='saeo', algo_kwargs=saeo_kwargs,
        results_file=f"{base}_phase1_saeo{ext}",
    )
    merged = dict(phase1)

    # Phase 2: I-SAEO incremental extension seeded by SAEO's n=saeo_end group
    if args.n_end > saeo_end:
        seed_group = phase1[saeo_end]['best_group']
        phase2 = rac.run_incremental_sweep(
            virtual_nets=vnets, objective=objective, database_file=database_file,
            n_start=saeo_end + 1, n_end=args.n_end,
            evals_per_n=args.n_evals, cost_aware=cost_aware, seed=args.seed,
            pop_size=args.ga_pop_size, mutation_rate=args.ga_mutation_rate,
            pruned_configs=pruned_configs,
            include_pim=args.pim, include_switch=args.switch,
            algorithm='isaeo', algo_kwargs=saeo_kwargs,
            results_file=f"{base}_phase2_isaeo{ext}",
            initial_best_group=seed_group,
        )
        merged.update(phase2)
    return merged


def resolve_fengshui_pool_csv(fengshui_dir, objective, cost_aware):
    """Resolve the published pool CSV, preferring the deterministic AE pools.

    ``fengshui_dir`` is normally ``src/archgym_results``. Older bundles used
    ``v7_*`` or ``v6_*`` names, sometimes below an extra ``fengshui/`` or
    ``mozart/`` directory, so retain those layouts as fallbacks.
    """
    metric = f"{objective}_cost" if cost_aware else objective
    versions = (f"ae_{metric}_chain", f"v7_{metric}_chain",
                f"v6_{metric}_chain")
    roots = (fengshui_dir,
             os.path.join(fengshui_dir, "fengshui"),
             os.path.join(fengshui_dir, "mozart"))
    searched = []
    for version in versions:
        for root in roots:
            pool_dir = os.path.join(root, version)
            searched.append(pool_dir)
            csvs = sorted(_glob.glob(os.path.join(
                pool_dir, "saeo_isaeo_chain_*.csv")))
            if csvs:
                return csvs[-1], searched
    return None, searched


def fengshui_cross_eval(test_nets, objective, cost_aware, database_file,
                        fengshui_dir, n_start, n_end):
    """Evaluate the paper's Fengshui pool on the test set, fixed, per n.

    Returns ``({n: geomean_value}, resolved_pool_csv)``. The pool is loaded
    via ``ChipletConfig.from_csv_for_n_chiplets`` instead of the current train
    run, and its resolved path is retained in the result provenance.
    """
    mcsv, searched = resolve_fengshui_pool_csv(
        fengshui_dir, objective, cost_aware)
    if mcsv is None:
        print("  [warn] no Fengshui pool CSV found; searched:\n    " +
              "\n    ".join(searched))
        return {}, None
    print(f"  Fengshui pool: {mcsv}")
    out = {}
    for n in range(n_start, n_end + 1):
        try:
            pool = ChipletConfig.from_csv_for_n_chiplets(n, mcsv)
        except Exception:
            continue  # that n not present in the Fengshui sweep
        assert_no_forbidden_arch_targets(pool, f"Fengshui fixed pool n={n}")
        _, res = run_single_optimization(
            virtual_nets=test_nets, chiplet_group=pool, objective=objective,
            results_file=database_file, cost_aware=cost_aware,
            use_sequential=True, n_workers=8,
            use_dag_cp=rac._USE_DAG_CP, v_het_batch=rac._V_HET_BATCH)
        val, _ = calculate_average_opt_value(res, objective)
        out[n] = val
    return out, os.path.abspath(mcsv)


def cross_eval(train_sweep, test_nets, objective, cost_aware, database_file):
    """Evaluate each TRAIN-derived pool (per n) on the TEST set, fixed pool."""
    rows = []
    for n in sorted(train_sweep.keys()):
        group = train_sweep[n]["best_group"]
        if not group:
            rows.append({"n_chiplets": n, "cross_value": float("inf"),
                         "net_values": {}})
            continue
        assert_no_forbidden_arch_targets(group, f"train-derived pool n={n}")
        _, results = run_single_optimization(
            virtual_nets=test_nets,
            chiplet_group=group,
            objective=objective,
            results_file=database_file,
            cost_aware=cost_aware,
            use_sequential=True,
            n_workers=8,
            use_dag_cp=rac._USE_DAG_CP,
            v_het_batch=rac._V_HET_BATCH,
        )
        cross_value, _ = calculate_average_opt_value(results, objective)
        net_values = {name: r["min_value"] for name, r in results.items()}
        rows.append({"n_chiplets": n, "cross_value": cross_value,
                     "net_values": net_values})
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database", default=os.path.join(PROJECT_DIR, "unified_database.csv"),
                   help="Path to the Timeloop performance database CSV")
    p.add_argument("--output-dir", default=os.path.join(THIS_DIR, "results"))
    p.add_argument("--seed", type=int, default=42, help="Split + optimiser seed")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--split-mode", default="random",
                   choices=["random", "decode_test", "vision_test", "moe_test",
                            "prefill_test", "seq2048_test"],
                   help="'random': 70/30 random; otherwise a held-out family is "
                        "the test set: 'decode_test' (decode), 'vision_test' "
                        "(CNN+ViT), 'moe_test' (qwen MoE)")
    p.add_argument("--dry-split", action="store_true",
                   help="Print the train/test split and exit (no optimization)")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8],
                   help="Batch sizes; each (workload,batch) is one virtual net")
    # Framework / sweep knobs
    p.add_argument("--algorithm", default="chain",
                   choices=["chain", "ga", "sa", "random", "de", "pso", "memetic",
                            "eda", "saeo", "isaeo"],
                   help="'chain' = paper's SAEO->I-SAEO (default); others run a "
                        "single-algorithm incremental sweep")
    p.add_argument("--n-start", type=int, default=1)
    p.add_argument("--n-end", type=int, default=8)
    p.add_argument("--saeo-end", type=int, default=6,
                   help="chain: last n for the SAEO phase; I-SAEO covers "
                        "saeo_end+1..n_end")
    p.add_argument("--n-evals", type=int, default=500)
    p.add_argument("--drop-seqs", type=int, nargs="*", default=[],
                   help="Sequence lengths to drop, e.g. --drop-seqs 1024 4096 "
                        "(removes prefill_s{N} and decode_kv{N} for each N)")
    p.add_argument("--fengshui-dir", default="",
                   help="Dir with the paper's Fengshui chains (v6_*_chain/); if set, "
                        "also evaluate the Fengshui pool per n on the test set and "
                        "store it in summary.json as 'fengshui_cross'")
    p.add_argument("--ga-pop-size", type=int, default=20)
    p.add_argument("--ga-mutation-rate", type=float, default=0.1)
    p.add_argument("--sa-initial-temp", type=float, default=1.0)
    p.add_argument("--sa-cooling-rate", type=float, default=0.95)
    p.add_argument("--sa-min-temp", type=float, default=0.01)
    # SAEO / I-SAEO (the chain's actual optimizers) exploration knobs.
    p.add_argument("--saeo-candidates", type=int, default=2000,
                   help="chain/saeo/isaeo: candidate pool pre-screened by the RF "
                        "surrogate each round (higher = broader exploration)")
    p.add_argument("--saeo-mutation-rate", type=float, default=0.3,
                   help="chain/saeo/isaeo: per-gene mutation probability during "
                        "candidate generation (higher = more diversity)")
    p.add_argument("--prune-pct", type=int, default=30,
                   help="Roofline pruning: keep top X%% configs (0=off)")
    p.add_argument("--pim", action="store_true", default=True)
    p.add_argument("--no-pim", dest="pim", action="store_false")
    p.add_argument("--switch", action="store_true", default=True)
    p.add_argument("--no-switch", dest="switch", action="store_false")
    p.add_argument("--dag", action="store_true", default=True)
    p.add_argument("--no-dag", dest="dag", action="store_false")
    p.add_argument("--configs", nargs="+", default=[c[2] for c in CONFIGS],
                   help="Which of the 4 configs to run (by short name)")
    # Smoke-test shortcut
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run: energy_nocost only, n_end=2, n_evals=8")
    args = p.parse_args()

    if args.smoke:
        args.configs = ["energy_nocost"]
        args.n_end = min(args.n_end, 2)
        args.saeo_end = 1  # exercise both chain phases: SAEO n=1, I-SAEO n=2
        args.n_evals = min(args.n_evals, 8)

    search_targets = rac._build_arch_targets(args.pim, args.switch)
    forbidden_search_targets = sorted(
        {str(target).strip().lower() for target in search_targets}
        & FORBIDDEN_ARCH_TARGETS
    )
    if forbidden_search_targets:
        raise RuntimeError(
            "train-test candidate space contains forbidden architecture "
            "target(s): " + ", ".join(forbidden_search_targets)
        )
    print("Architecture targets (FEATHER excluded): "
          + ", ".join(search_targets))

    # DAG flags consumed by the framework's worker processes via fork.
    rac._USE_DAG_CP = args.dag
    rac._V_HET_BATCH = True

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.output_dir, ("smoke_" if args.smoke else "") + timestamp)
    os.makedirs(out_root, exist_ok=True)
    print(f"Output dir: {out_root}")
    print(f"Database  : {args.database}")

    # --- 1. enumerate workloads -------------------------------------------
    workloads = list_workloads(drop_seqs=args.drop_seqs)
    print(f"\n{len(workloads)} workloads after exclusion:\n  " + "\n  ".join(workloads))

    # --- 2. preload DB ONCE for all needed nets (the only big-CSV read) ----
    needed_nets = set(workloads)
    print(f"\nPreloading database for {len(needed_nets)} networks "
          "(single full-CSV read; all lookups afterwards are in-memory)...")
    t0 = time.perf_counter()
    preload_database(args.database, needed_nets=needed_nets)
    print(f"  preload done in {time.perf_counter() - t0:.1f}s")

    # --- 3. build virtual nets (layer sets reused from the preloaded cache) -
    db_layers_per_net = layers_from_preloaded(args.database)
    print("Building virtual networks...")
    vnets = build_virtual_nets(workloads, args.batches, db_layers_per_net)
    print(f"Built {len(vnets)} virtual networks "
          f"({len(workloads)} workloads x {len(args.batches)} batches).")

    # --- 4. train/test split ----------------------------------------------
    # Structured held-out splits: test = workloads matching the family predicate,
    # train = everything else. Probes whether a pool that never saw a regime can
    # still serve it.
    HELD_OUT = {
        "decode_test":  lambda n: "decode" in n,                      # decode -> test
        "vision_test":  lambda n: n.startswith(("mobilenet", "replknet", "vit")),
        "moe_test":     lambda n: "qwen" in n,                        # MoE (qwen) -> test
        "prefill_test": lambda n: "prefill" in n,                     # all prefill -> test
        "seq2048_test": lambda n: "s2048" in n or "kv2048" in n,      # all 2048-len -> test
    }
    if args.split_mode in HELD_OUT:
        pred = HELD_OUT[args.split_mode]
        test_nets = [vn for vn in vnets if pred(vn.network_name)]
        train_nets = [vn for vn in vnets if not pred(vn.network_name)]
        print(f"\nSplit[{args.split_mode}]: {len(train_nets)} train / "
              f"{len(test_nets)} test (held-out family -> test)")
    else:  # random
        rng = random.Random(args.seed)
        order = list(range(len(vnets)))
        rng.shuffle(order)
        n_train = int(round(len(vnets) * args.train_frac))
        train_idx = set(order[:n_train])
        train_nets = [vnets[i] for i in range(len(vnets)) if i in train_idx]
        test_nets = [vnets[i] for i in range(len(vnets)) if i not in train_idx]
        print(f"\nSplit[random]: {len(train_nets)} train / {len(test_nets)} test "
              f"(train_frac={args.train_frac}, seed={args.seed})")

    split_info = {
        "split_mode": args.split_mode,
        "drop_seqs": args.drop_seqs,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "batches": args.batches,
        "excluded_prefixes": list(EXCLUDE_PREFIXES),
        "workloads": workloads,
        "train": [vn.get_unique_name() for vn in train_nets],
        "test": [vn.get_unique_name() for vn in test_nets],
    }
    with open(os.path.join(out_root, "split.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    if args.dry_split:
        print("\n[dry-split] train:")
        for u in split_info["train"]:
            print("   ", u)
        print("[dry-split] test:")
        for u in split_info["test"]:
            print("   ", u)
        print("[dry-split] done (no optimization run).")
        return

    # --- 5. run the 4 configs ----------------------------------------------
    name2cfg = {c[2]: c for c in CONFIGS}
    # Keep enough provenance to audit a standalone summary.json. Earlier
    # shipped summaries omitted even the split mode, DB, seed and drop_seqs.
    provenance_args = dict(vars(args))
    provenance_args["database"] = os.path.abspath(args.database)
    provenance_args["output_dir"] = os.path.abspath(args.output_dir)
    if args.fengshui_dir:
        provenance_args["fengshui_dir"] = os.path.abspath(args.fengshui_dir)
    provenance_env = {
        "FENGSHUI_DETERMINISTIC": os.environ.get("FENGSHUI_DETERMINISTIC", "1"),
        "FENGSHUI_GA_SEED": os.environ.get("FENGSHUI_GA_SEED", "0"),
        "FENGSHUI_EXCLUDE_PREFIXES": os.environ.get(
            "FENGSHUI_EXCLUDE_PREFIXES", "efficientnet,gpt,resnet"),
        "INNER_GA_POP": os.environ.get("INNER_GA_POP", "30"),
        "INNER_GA_GEN": os.environ.get("INNER_GA_GEN", "30"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        # Set by run_all_v2.sh only after it verifies the actual file hash.
        "validated_database_sha256": os.environ.get("FENGSHUI_DB_SHA256"),
    }
    all_summary = {
        "timestamp": timestamp,
        "command": [sys.executable] + sys.argv,
        "script": os.path.abspath(__file__),
        "split": split_info,
        "args": provenance_args,
        "environment": provenance_env,
        "configs": {},
    }

    for cfg_name in args.configs:
        if cfg_name not in name2cfg:
            print(f"[warn] unknown config '{cfg_name}', skipping")
            continue
        objective, cost_aware, _ = name2cfg[cfg_name]
        cfg_dir = os.path.join(out_root, cfg_name)
        os.makedirs(cfg_dir, exist_ok=True)
        print(f"\n{'#'*70}\n# CONFIG: {cfg_name}  (objective={objective}, "
              f"cost_aware={cost_aware})\n{'#'*70}")

        # Roofline pruning is computed per (net-set, cost) — build for each set.
        def maybe_prune(net_names):
            if args.prune_pct <= 0:
                return None
            return get_pruned_configs(args.database, top_pct=args.prune_pct,
                                      needed_nets=set(net_names),
                                      cost_aware=cost_aware)

        runner = run_chain if args.algorithm == "chain" else run_sweep

        # a) TRAIN
        print(f"\n[{cfg_name}] === TRAIN sweep ({len(train_nets)} nets, "
              f"{args.algorithm}) ===")
        train_prune = maybe_prune([vn.network_name for vn in train_nets])
        train_sweep = runner(train_nets, objective, cost_aware, args.database,
                             args, os.path.join(cfg_dir, "train_sweep.csv"),
                             train_prune)

        # b) TEST (own optimum)
        print(f"\n[{cfg_name}] === TEST sweep ({len(test_nets)} nets, "
              f"{args.algorithm}) ===")
        test_prune = maybe_prune([vn.network_name for vn in test_nets])
        test_sweep = runner(test_nets, objective, cost_aware, args.database,
                            args, os.path.join(cfg_dir, "test_sweep.csv"),
                            test_prune)

        # c) CROSS: TRAIN pool -> TEST set (fixed pool)
        print(f"\n[{cfg_name}] === CROSS eval (TRAIN pool on TEST set) ===")
        cross_rows = cross_eval(train_sweep, test_nets, objective, cost_aware,
                                args.database)

        # write cross_eval.csv
        import csv as _csv
        test_unames = [vn.get_unique_name() for vn in test_nets]
        cross_csv = os.path.join(cfg_dir, "cross_eval.csv")
        with open(cross_csv, "w", newline="") as f:
            fieldnames = ["n_chiplets", f"cross_{objective}",
                          f"test_own_{objective}", "generalization_ratio"] + \
                         [f"{u}_{objective}" for u in test_unames]
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in cross_rows:
                n = row["n_chiplets"]
                own = test_sweep.get(n, {}).get("best_value", float("inf"))
                cross = row["cross_value"]
                ratio = (cross / own) if (own and math.isfinite(own) and own > 0
                                          and math.isfinite(cross)) else float("inf")
                d = {"n_chiplets": n, f"cross_{objective}": cross,
                     f"test_own_{objective}": own,
                     "generalization_ratio": ratio}
                for u in test_unames:
                    d[f"{u}_{objective}"] = row["net_values"].get(u, float("inf"))
                w.writerow(d)

        # c2) Fengshui: the paper's fixed pool (per n) on the TEST set
        fengshui_cross = {}
        fengshui_pool_csv = None
        if args.fengshui_dir:
            print(f"\n[{cfg_name}] === Fengshui pool on TEST set ===")
            fengshui_cross, fengshui_pool_csv = fengshui_cross_eval(
                test_nets, objective, cost_aware, args.database,
                args.fengshui_dir, args.n_start, args.n_end)

        # compact per-config summary
        cfg_summary = {
            "objective": objective, "cost_aware": cost_aware,
            "train": {n: train_sweep[n]["best_value"] for n in train_sweep},
            "test_own": {n: test_sweep[n]["best_value"] for n in test_sweep},
            "cross": {r["n_chiplets"]: r["cross_value"] for r in cross_rows},
            "fengshui_cross": fengshui_cross,
            "provenance": {
                "command": all_summary["command"],
                "script": all_summary["script"],
                "args": provenance_args,
                "environment": provenance_env,
                "split_file": os.path.abspath(os.path.join(out_root, "split.json")),
                "fengshui_pool_csv": fengshui_pool_csv,
            },
        }
        with open(os.path.join(cfg_dir, "summary.json"), "w") as f:
            json.dump(cfg_summary, f, indent=2)
        all_summary["configs"][cfg_name] = cfg_summary

        # console recap
        print(f"\n[{cfg_name}] n  train      test(own)   cross(train->test)  ratio")
        for n in sorted(train_sweep.keys()):
            tr = train_sweep[n]["best_value"]
            te = test_sweep.get(n, {}).get("best_value", float("inf"))
            cr = next((r["cross_value"] for r in cross_rows
                       if r["n_chiplets"] == n), float("inf"))
            ratio = cr / te if (te and math.isfinite(te) and te > 0) else float("inf")
            print(f"[{cfg_name}] {n:2d} {tr:10.3e} {te:10.3e} {cr:18.3e} {ratio:7.3f}")

    with open(os.path.join(out_root, "summary_all.json"), "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\nAll done. Results in {out_root}")


if __name__ == "__main__":
    main()
