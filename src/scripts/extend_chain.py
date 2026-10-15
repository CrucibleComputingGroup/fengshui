"""
Load the best chiplet group of a completed chain run (its chain_summary_*.json, highest n).
converge_chain.py warm-starts the extension of a chain past its last N from this group.
"""

import os
import sys
import json

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

