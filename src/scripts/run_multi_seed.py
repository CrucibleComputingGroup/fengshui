"""Run GA and SAEO with 100 seeds in parallel."""
import subprocess, re, sys, os
from concurrent.futures import ProcessPoolExecutor, as_completed

seeds = list(range(10))

def run_one(seed):
    """Run one seed, return (seed, {algo: best_value})."""
    proc = subprocess.run(
        [sys.executable, 'run_archgym_chiplet.py',
         '--objective', 'energy', '--n-chiplets', '4',
         '--database', '../unified_database.csv',
         '--n-evals', '500', '--algorithms', 'ga', 'saeo',
         '--seed', str(seed), '--cost-aware',
         '--output-dir', f'archgym_results_seed{seed}'],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    result = {}
    for line in proc.stdout.splitlines():
        for algo in ['ga', 'saeo']:
            m = re.match(rf'^{algo}\s+(\S+)', line)
            if m:
                result[algo] = float(m.group(1))
    return seed, result

if __name__ == '__main__':
    all_results = {'ga': [], 'saeo': []}

    with ProcessPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(run_one, s): s for s in seeds}
        for future in as_completed(futures):
            seed, result = future.result()
            for algo in ['ga', 'saeo']:
                if algo in result:
                    all_results[algo].append(result[algo])
            ga_v = f"{result['ga']:.4f}" if 'ga' in result else 'N/A'
            saeo_v = f"{result['saeo']:.4f}" if 'saeo' in result else 'N/A'
            print(f"Seed {seed:>5} done: GA={ga_v}  SAEO={saeo_v}")

    print(f"\n{'='*60}")
    print(f"Summary across {len(seeds)} seeds:")
    print(f"{'='*60}")
    import numpy as np
    for algo in ['ga', 'saeo']:
        vals = all_results[algo]
        if vals:
            arr = np.array(vals)
            print(f"{algo.upper():>6}: best={arr.min():.4f}  worst={arr.max():.4f}  "
                  f"mean={arr.mean():.4f} +/- {arr.std():.4f}  median={np.median(arr):.4f}  ({len(vals)} runs)")

    # Per-seed wins (paired comparison)
    n = len(seeds)
    if len(all_results['ga']) == n and len(all_results['saeo']) == n:
        ga_vals = all_results['ga']
        saeo_vals = all_results['saeo']
        saeo_wins = sum(1 for g, s in zip(ga_vals, saeo_vals) if s < g)
        ties = sum(1 for g, s in zip(ga_vals, saeo_vals) if abs(s - g) < 1e-6)
        print(f"\nSAEO wins: {saeo_wins}/{n},  GA wins: {n - saeo_wins - ties}/{n},  Ties: {ties}/{n}")
