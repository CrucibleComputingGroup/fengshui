"""
Chain SAEO (n=1→K) + I-SAEO (n=K+1→M) for incremental chiplet pool optimization.

SAEO does global search for the base pool, then I-SAEO extends it
incrementally with prefix-fixed search.

Usage:
    python3 run_saeo_isaeo_chain.py \
        --objective energy --database ../unified_database.csv \
        --n-evals 500 --seed 42 --cost-aware \
        --saeo-end 6 --isaeo-end 10
"""

import os
import sys
import time
import json
import csv
import copy
import logging

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
_CHIPLET_TL_DIR = os.path.normpath(os.path.join(THIS_DIR, '..'))
if _CHIPLET_TL_DIR not in sys.path:
    sys.path.insert(0, _CHIPLET_TL_DIR)
_EXP_DIR = os.path.normpath(os.path.join(THIS_DIR, '..', 'timeloop_experiments'))
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SAEO -> I-SAEO chained incremental sweep")
    parser.add_argument("--objective", type=str, choices=["energy", "edp"], default="energy")
    parser.add_argument("--database", type=str, default="../unified_database.csv")
    parser.add_argument("--n-evals", type=int, default=500, help="Evals per n for both phases")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-aware", action="store_true", default=False)
    parser.add_argument("--saeo-start", type=int, default=1)
    parser.add_argument("--saeo-end", type=int, default=6)
    parser.add_argument("--isaeo-end", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="archgym_saeo_isaeo_chain")
    parser.add_argument("--pim", action="store_true", default=True)
    parser.add_argument("--no-pim", dest="pim", action="store_false")
    parser.add_argument("--switch", action="store_true", default=True)
    parser.add_argument("--no-switch", dest="switch", action="store_false")
    parser.add_argument("--cnn", action="store_true", default=True)
    parser.add_argument("--no-cnn", dest="cnn", action="store_false")
    parser.add_argument("--dag", action="store_true", default=True)
    parser.add_argument("--no-dag", dest="dag", action="store_false")
    parser.add_argument("--v-het-batch", type=str, default="True")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    import run_archgym_chiplet as rac
    rac._USE_DAG_CP = args.dag
    rac._V_HET_BATCH = args.v_het_batch != "False"

    import multiprocessing as mp
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    logger.info("Loading virtual networks...")
    virtual_nets = rac.setup_virtual_nets(args.database, include_cnn=args.cnn)
    logger.info(f"Loaded {len(virtual_nets)} virtual networks")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_csv = os.path.join(args.output_dir,
        f'saeo_isaeo_chain_{args.objective}_{timestamp}.csv')
    # Use separate files per phase to avoid overwrite, merge at end
    results_csv_phase1 = os.path.join(args.output_dir,
        f'phase1_saeo_{args.objective}_{timestamp}.csv')
    results_csv_phase2 = os.path.join(args.output_dir,
        f'phase2_isaeo_{args.objective}_{timestamp}.csv')

    all_results = {}

    # =================================================================
    # Phase 1: SAEO (n=saeo_start -> saeo_end)
    # =================================================================
    logger.info(f"Phase 1: SAEO n={args.saeo_start} -> {args.saeo_end}")
    phase1_results = rac.run_incremental_sweep(
        virtual_nets, args.objective, args.database,
        n_start=args.saeo_start, n_end=args.saeo_end,
        evals_per_n=args.n_evals, cost_aware=args.cost_aware,
        seed=args.seed,
        include_pim=args.pim, include_switch=args.switch,
        algorithm='saeo',
        results_file=results_csv_phase1,
    )
    all_results.update(phase1_results)

    # Find best group from SAEO phase
    saeo_best_n = min(phase1_results.keys(),
                      key=lambda k: phase1_results[k]['best_value'])
    saeo_best_group = phase1_results[args.saeo_end]['best_group']
    saeo_best_value = phase1_results[saeo_best_n]['best_value']
    logger.info(f"Phase 1 done: best at n={saeo_best_n}, energy={saeo_best_value:.4e}")
    logger.info(f"Using n={args.saeo_end} group as seed for Phase 2 "
                f"({len(saeo_best_group)} chiplets)")

    for n, result in phase1_results.items():
        csv_file = os.path.join(args.output_dir,
            f"convergence_saeo_n{n}_{timestamp}.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f,
                fieldnames=['eval', 'value', 'best_value', 'eval_time'])
            writer.writeheader()
            for row in result['history']:
                writer.writerow(row)

    # =================================================================
    # Phase 2: I-SAEO (n=saeo_end+1 -> isaeo_end)
    # =================================================================
    if args.isaeo_end > args.saeo_end:
        logger.info(f"Phase 2: I-SAEO n={args.saeo_end + 1} -> {args.isaeo_end}")

        phase2_results = rac.run_incremental_sweep(
            virtual_nets, args.objective, args.database,
            n_start=args.saeo_end + 1, n_end=args.isaeo_end,
            evals_per_n=args.n_evals, cost_aware=args.cost_aware,
            seed=args.seed,
            include_pim=args.pim, include_switch=args.switch,
            algorithm='isaeo',
            results_file=results_csv_phase2,
            initial_best_group=saeo_best_group,
        )
        all_results.update(phase2_results)

        for n, result in phase2_results.items():
            csv_file = os.path.join(args.output_dir,
                f"convergence_isaeo_n{n}_{timestamp}.csv")
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f,
                    fieldnames=['eval', 'value', 'best_value', 'eval_time'])
                writer.writeheader()
                for row in result['history']:
                    writer.writerow(row)

    # =================================================================
    # Merge phase CSVs into one combined file
    # =================================================================
    import pandas as pd
    dfs = []
    if os.path.exists(results_csv_phase1):
        dfs.append(pd.read_csv(results_csv_phase1))
    if os.path.exists(results_csv_phase2):
        dfs.append(pd.read_csv(results_csv_phase2))
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(results_csv, index=False)
        logger.info(f"Merged {len(merged)} rows into {results_csv}")

    # =================================================================
    # Summary
    # =================================================================
    print(f"\n{'='*60}")
    print("SAEO -> I-SAEO Chain Summary")
    print(f"{'='*60}")
    print(f"{'n':>3} {'algo':>8} {'best_value':>12} {'time(s)':>8}")
    for n in sorted(all_results.keys()):
        r = all_results[n]
        algo = 'SAEO' if n <= args.saeo_end else 'I-SAEO'
        print(f"{n:3d} {algo:>8} {r['best_value']:12.4f} {r.get('time', 0):8.1f}")

    overall_best_n = min(all_results.keys(),
                         key=lambda k: all_results[k]['best_value'])
    print(f"\nOverall best: n={overall_best_n}, "
          f"energy={all_results[overall_best_n]['best_value']:.4e}")

    summary = {}
    for n in sorted(all_results.keys()):
        r = all_results[n]
        algo = 'saeo' if n <= args.saeo_end else 'isaeo'
        summary[f"n{n}"] = {
            'algorithm': algo,
            'best_value': r['best_value'],
            'time': r.get('time', 0),
            'best_chiplets': [c.get_identifier() for c in r['best_group']]
                             if r['best_group'] else [],
        }
    summary_file = os.path.join(args.output_dir, f"chain_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
