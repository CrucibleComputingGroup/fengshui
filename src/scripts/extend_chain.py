"""
Extend v6 chain results to larger N by running SAEO at a target pool size directly.

Loads the best chiplet group from a completed chain run (e.g., v6 n=10),
then runs SAEO at a target N (e.g., 20) warm-started from that group.

Usage:
    python3 extend_chain.py \
        --objective energy --database ../timeloop_experiments/unified_database.csv \
        --source-dir ../archgym_results/v6_energy_chain \
        --target-n 20 --n-evals 500 --seed 42 \
        --output-dir ../archgym_results/v6_energy_chain
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


def load_best_group_from_summary(source_dir):
    """Load the best chiplet group from the chain summary JSON (highest n)."""
    import glob as globmod
    from chiplet_dataclass import ChipletConfig

    summaries = sorted(globmod.glob(os.path.join(source_dir, 'chain_summary_*.json')))
    if not summaries:
        raise FileNotFoundError(f"No chain_summary_*.json in {source_dir}")
    summary_file = summaries[-1]  # most recent

    with open(summary_file) as f:
        summary = json.load(f)

    # Find the highest n
    max_n = max(int(k.replace('n', '')) for k in summary.keys())
    best_chiplets_ids = summary[f'n{max_n}']['best_chiplets']
    best_value = summary[f'n{max_n}']['best_value']

    group = [ChipletConfig.from_identifier(cid) for cid in best_chiplets_ids]
    print(f"Loaded n={max_n} group from {summary_file} (value={best_value:.4e})")
    return group, max_n, best_value


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extend chain to larger N")
    parser.add_argument("--objective", type=str, choices=["energy", "edp"], required=True)
    parser.add_argument("--database", type=str, required=True)
    parser.add_argument("--source-dir", type=str, required=True,
                        help="Directory with chain_summary_*.json from a completed run")
    parser.add_argument("--target-n", type=int, required=True,
                        help="Target pool size (e.g., 20)")
    parser.add_argument("--n-evals", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-aware", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--algorithm", type=str, default="saeo",
                        choices=["saeo", "isaeo", "sa"])
    parser.add_argument("--pim", action="store_true", default=True)
    parser.add_argument("--no-pim", dest="pim", action="store_false")
    parser.add_argument("--switch", action="store_true", default=True)
    parser.add_argument("--no-switch", dest="switch", action="store_false")
    parser.add_argument("--dag", action="store_true", default=True)
    parser.add_argument("--no-dag", dest="dag", action="store_false")
    parser.add_argument("--v-het-batch", type=str, default="True")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    import run_archgym_chiplet as rac
    rac._USE_DAG_CP = args.dag
    rac._V_HET_BATCH = args.v_het_batch != "False"

    import multiprocessing as mp
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    # Load previous best group
    prev_group, prev_n, prev_value = load_best_group_from_summary(args.source_dir)
    if args.target_n <= prev_n:
        raise ValueError(f"target-n ({args.target_n}) must be > source n ({prev_n})")

    logger.info("Loading virtual networks...")
    virtual_nets = rac.setup_virtual_nets(args.database, include_cnn=True)
    logger.info(f"Loaded {len(virtual_nets)} virtual networks")

    output_dir = args.output_dir or args.source_dir
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    results_csv = os.path.join(output_dir,
        f'extend_n{args.target_n}_{args.objective}_{timestamp}.csv')

    # Run SAEO/SA directly at target_n, warm-started from prev_group
    logger.info(f"Running {args.algorithm.upper()} at n={args.target_n}, "
                f"warm-started from n={prev_n}")

    results = rac.run_incremental_sweep(
        virtual_nets, args.objective, args.database,
        n_start=args.target_n, n_end=args.target_n,
        evals_per_n=args.n_evals, cost_aware=args.cost_aware,
        seed=args.seed,
        include_pim=args.pim, include_switch=args.switch,
        algorithm=args.algorithm,
        results_file=results_csv,
        initial_best_group=prev_group,
    )

    result = results[args.target_n]
    logger.info(f"Done: n={args.target_n}, best={result['best_value']:.4e}, "
                f"time={result.get('time', 0):.1f}s")

    # Save convergence
    conv_csv = os.path.join(output_dir,
        f'convergence_extend_n{args.target_n}_{timestamp}.csv')
    with open(conv_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f,
            fieldnames=['eval', 'value', 'best_value', 'eval_time'])
        writer.writeheader()
        for row in result['history']:
            writer.writerow(row)

    # Save summary
    summary = {
        f"n{args.target_n}": {
            'algorithm': args.algorithm,
            'best_value': result['best_value'],
            'time': result.get('time', 0),
            'warm_started_from': prev_n,
            'best_chiplets': [c.get_identifier() for c in result['best_group']]
                             if result['best_group'] else [],
        }
    }
    summary_file = os.path.join(output_dir,
        f'extend_summary_n{args.target_n}_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nExtension complete: n={args.target_n}, best={result['best_value']:.4e}")
    print(f"Results: {results_csv}")
    print(f"Summary: {summary_file}")


if __name__ == '__main__':
    main()
