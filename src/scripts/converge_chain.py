"""
converge_chain.py -- grow the chiplet pool past an existing chain's endpoint until the
marginal improvement in the search objective stays below a threshold for `patience`
CONSECUTIVE steps, then stop. This replaces the arbitrary "N=10 as proxy" for the
Unconstrained / hetero-ideal paradigm (generate_paper_fig10.py P4) with a principled,
convergence-defined endpoint.

Key design points
-----------------
* WARM START, no re-run of N=1..K. Loads the highest-N group from an existing chain's
  chain_summary_*.json (same mechanism as extend_chain.py) and continues incrementally
  (N=K+1 warm-started from N=K, N=K+2 from N=K+1, ...), one chiplet at a time.
* Convergence signal = the search's own objective column `best_{objective}` (which the
  engine monotonic-clamps, run_archgym_chiplet.py:1768). We additionally clamp at the
  driver level because each single-N call starts with an empty all_results (clamp cannot
  fire inside the engine).
* PATIENCE is mandatory. The per-step marginal has a "plateau-then-jump" shape (e.g.
  energy N=2->3 is +0.43% but N=3->4 is +7.4%). A naive stop-at-first-sub-1% halts far
  too early. Requiring `patience` consecutive sub-threshold steps defeats this; every
  observed false-positive is a *single* isolated sub-threshold step, so patience=2 is
  sufficient.
* If the EXISTING chain already satisfies the criterion at its endpoint, no new compute
  is run (converged in place).
* Non-destructive: writes a consolidated chain (N=1..N*) + summary into --output-dir
  (a *_converged sibling dir), leaving the canonical v7_*_chain dirs untouched so
  ablation / competing (which read chain_summary n8) are unaffected.

Usage (run in conda `base` -- gym lives there, not mozart):
    conda activate base
    python3 converge_chain.py \
        --objective edp --cost-aware \
        --database ../unified_database.csv \
        --source-dir ../archgym_results/v7_edp_cost_chain \
        --output-dir ../archgym_results/v7_edp_cost_chain_converged \
        --threshold-pct 1.0 --patience 2 --n-max 16
    # add --dry-run to assess convergence-from-existing and print the plan without any heavy step.
"""

import os
import sys
import time
import json
import csv
import copy
import glob as globmod
import logging

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
_CHIPLET_TL_DIR = os.path.normpath(os.path.join(THIS_DIR, '..'))
if _CHIPLET_TL_DIR not in sys.path:
    sys.path.insert(0, _CHIPLET_TL_DIR)
_EXP_DIR = os.path.normpath(os.path.join(THIS_DIR, '..', 'timeloop_experiments'))
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)


def _latest(pattern):
    files = sorted(globmod.glob(pattern))
    return files[-1] if files else None


def load_existing_chain(source_dir, objective):
    """Return (df_prev, best_col, marginals) for the latest chain CSV in source_dir.

    marginals: list of (n, best_value, pct_improvement_vs_prev) sorted by n
    (pct is None for the first n).
    """
    import pandas as pd
    csv_path = _latest(os.path.join(source_dir, 'saeo_isaeo_chain_*.csv'))
    if csv_path is None:
        raise FileNotFoundError(f'No saeo_isaeo_chain_*.csv in {source_dir}')
    df = pd.read_csv(csv_path).sort_values('n_chiplets').reset_index(drop=True)
    best_col = f'best_{objective}'
    if best_col not in df.columns:
        raise KeyError(f'{best_col} not in {csv_path} (cols: {list(df.columns)[:5]}...)')
    marginals = []
    prev = None
    for _, r in df.iterrows():
        n = int(r['n_chiplets'])
        v = float(r[best_col])
        pct = None if prev is None else (prev - v) / prev * 100.0
        marginals.append((n, v, pct))
        prev = v
    return df, csv_path, best_col, marginals


def consecutive_subthreshold(marginals, threshold_pct):
    """Count how many trailing steps (ending at the last n) are < threshold_pct."""
    cnt = 0
    for (_, _, pct) in reversed(marginals):
        if pct is None:
            break
        if pct < threshold_pct:
            cnt += 1
        else:
            break
    return cnt


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convergence-stopped chiplet-pool extension")
    parser.add_argument("--objective", type=str, choices=["energy", "edp"], required=True)
    parser.add_argument("--cost-aware", action="store_true", default=False)
    parser.add_argument("--database", type=str, required=True)
    parser.add_argument("--source-dir", type=str, required=True,
                        help="Dir with existing saeo_isaeo_chain_*.csv + chain_summary_*.json")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Dir to write the consolidated (N=1..N*) chain + summary")
    parser.add_argument("--threshold-pct", type=float, default=1.0,
                        help="Stop when marginal improvement < this %% for `patience` steps")
    parser.add_argument("--patience", type=int, default=2,
                        help="Consecutive sub-threshold steps required to declare convergence")
    parser.add_argument("--n-max", type=int, default=16, help="Hard cap on pool size")
    parser.add_argument("--n-evals", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pim", action="store_true", default=True)
    parser.add_argument("--no-pim", dest="pim", action="store_false")
    parser.add_argument("--switch", action="store_true", default=True)
    parser.add_argument("--no-switch", dest="switch", action="store_false")
    parser.add_argument("--cnn", action="store_true", default=True)
    parser.add_argument("--no-cnn", dest="cnn", action="store_false")
    parser.add_argument("--dag", action="store_true", default=True)
    parser.add_argument("--no-dag", dest="dag", action="store_false")
    parser.add_argument("--v-het-batch", type=str, default="True")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Assess convergence-from-existing + print plan; run no heavy step")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("converge_chain")

    import pandas as pd
    import run_archgym_chiplet as rac
    from extend_chain import load_best_group_from_summary
    rac._USE_DAG_CP = args.dag
    rac._V_HET_BATCH = args.v_het_batch != "False"

    import multiprocessing as mp
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    metric_name = args.objective + ("_cost" if args.cost_aware else "")
    logger.info(f"=== converge_chain: metric={metric_name} threshold={args.threshold_pct}% "
                f"patience={args.patience} n_max={args.n_max} ===")

    # 1) Existing chain + convergence assessment ----------------------------------
    df_prev, chain_csv, best_col, marginals = load_existing_chain(args.source_dir, args.objective)
    max_n = marginals[-1][0]
    logger.info(f"Existing chain: {chain_csv}  (N=1..{max_n})")
    logger.info("Trailing marginals: " +
                ", ".join(f"N{n}:{'--' if pct is None else f'{pct:.3f}%'}"
                          for n, _, pct in marginals[-5:]))
    already = consecutive_subthreshold(marginals, args.threshold_pct)
    logger.info(f"Consecutive sub-{args.threshold_pct}% steps at endpoint N={max_n}: {already} "
                f"(need {args.patience})")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    new_rows = []          # list of dict rows for N>max_n
    converged_n = None
    consec = already
    prev_value = marginals[-1][1]

    if consec >= args.patience:
        converged_n = max_n
        logger.info(f"ALREADY CONVERGED at N={max_n} (no new compute needed).")
    else:
        # 2) Warm-start group from the existing summary endpoint -------------------
        cur_group, seed_n, seed_val = load_best_group_from_summary(args.source_dir)
        if seed_n != max_n:
            logger.warning(f"summary endpoint n={seed_n} != chain CSV endpoint n={max_n}; "
                           f"using summary group (n={seed_n}).")
        plan_from = seed_n + 1
        logger.info(f"NOT converged -> will extend N={plan_from}..{args.n_max} "
                    f"(stop at {args.patience} consecutive sub-{args.threshold_pct}% steps).")
        logger.info("NOTE per-step wall-clock ~1.5-2h and grows with N.")

        if args.dry_run:
            logger.info("[dry-run] stopping before any heavy step.")
        else:
            virtual_nets = rac.setup_virtual_nets(args.database, include_cnn=args.cnn)
            logger.info(f"Loaded {len(virtual_nets)} virtual networks")
            for n in range(seed_n + 1, args.n_max + 1):
                tmp_csv = os.path.join(args.output_dir, f'_step_n{n}_{timestamp}.csv')
                t0 = time.perf_counter()
                res = rac.run_incremental_sweep(
                    virtual_nets, args.objective, args.database,
                    n_start=n, n_end=n, evals_per_n=args.n_evals,
                    cost_aware=args.cost_aware, seed=args.seed,
                    include_pim=args.pim, include_switch=args.switch,
                    algorithm='isaeo', results_file=tmp_csv,
                    initial_best_group=cur_group,
                )
                dt = time.perf_counter() - t0
                raw_val = float(res[n]['best_value'])
                # driver-level monotonic clamp (engine can't clamp across single-N calls)
                best_value = min(raw_val, prev_value)
                cur_group = copy.deepcopy(res[n]['best_group'])
                pct = (prev_value - best_value) / prev_value * 100.0
                consec = consec + 1 if pct < args.threshold_pct else 0
                logger.info(f"N={n}: best={best_value:.5e} (raw {raw_val:.5e})  "
                            f"Δ={pct:.3f}%  consec<thr={consec}  time={dt/60:.1f}min")

                # collect the row from the per-step CSV, overwrite best col with clamp
                row = pd.read_csv(tmp_csv)
                row = row[row['n_chiplets'] == n].iloc[0].to_dict()
                row[best_col] = best_value
                new_rows.append(row)

                # persist an incremental summary each step (crash-safe)
                _write_summary(args, df_prev, new_rows, best_col, timestamp,
                               converged_n=(n if consec >= args.patience else None),
                               threshold=args.threshold_pct, patience=args.patience)

                prev_value = best_value
                if consec >= args.patience:
                    converged_n = n
                    logger.info(f"CONVERGED at N={n} "
                                f"({args.patience} consecutive sub-{args.threshold_pct}% steps).")
                    break
            else:
                converged_n = args.n_max
                logger.warning(f"Hit n_max={args.n_max} without {args.patience} consecutive "
                               f"sub-{args.threshold_pct}% steps; reporting N={args.n_max}.")

    # 3) Write consolidated chain CSV (N=1..N*) + summary --------------------------
    if not (args.dry_run and converged_n is None):
        out_df = df_prev.copy()
        if new_rows:
            out_df = pd.concat([df_prev, pd.DataFrame(new_rows)], ignore_index=True)
        out_df = out_df.sort_values('n_chiplets').reset_index(drop=True)
        out_csv = os.path.join(args.output_dir,
                               f'saeo_isaeo_chain_{args.objective}_{timestamp}.csv')
        out_df.to_csv(out_csv, index=False)
        logger.info(f"Wrote consolidated chain (N=1..{int(out_df['n_chiplets'].max())}) -> {out_csv}")
        _write_summary(args, df_prev, new_rows, best_col, timestamp,
                       converged_n=converged_n, threshold=args.threshold_pct,
                       patience=args.patience)
        # tidy up per-step temp files
        for f in globmod.glob(os.path.join(args.output_dir, f'_step_n*_{timestamp}.csv')):
            try:
                os.remove(f)
            except OSError:
                pass

    logger.info(f"=== DONE metric={metric_name}: converged_N={converged_n} ===")


def _write_summary(args, df_prev, new_rows, best_col, timestamp,
                   converged_n, threshold, patience):
    """Write/refresh a converged_marker JSON describing the endpoint."""
    import pandas as pd
    rows = df_prev.copy()
    if new_rows:
        rows = pd.concat([df_prev, pd.DataFrame(new_rows)], ignore_index=True)
    rows = rows.sort_values('n_chiplets')
    curve = [{'n': int(r['n_chiplets']), 'best_value': float(r[best_col])}
             for _, r in rows.iterrows()]
    marker = {
        'metric': args.objective + ("_cost" if args.cost_aware else ""),
        'objective': args.objective,
        'cost_aware': bool(args.cost_aware),
        'threshold_pct': threshold,
        'patience': patience,
        'n_max': args.n_max,
        'converged_n': converged_n,
        'source_dir': args.source_dir,
        'curve': curve,
    }
    path = os.path.join(args.output_dir, f'converged_marker_{timestamp}.json')
    with open(path, 'w') as f:
        json.dump(marker, f, indent=2)


if __name__ == '__main__':
    main()
