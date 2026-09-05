#!/usr/bin/env python3
"""Re-run vit_op_breakdown.py on the CORRECTED canonical DB.

The committed ViT case study (av_vision_results.csv) and the first breakdown run
both read generate_paper_fig14.DATABASE = timeloop_experiments/unified_database.csv,
which is the OLD pre-fix snapshot (DRAM double-count NOT removed) — and the merged
ViT cache at /tmp/vit_merged_database.csv was built Jun-14 (pre-fix) and never
invalidated. This wrapper repoints fig14.DATABASE to the canonical root
unified_database.csv (DRAM double-count fixed + fused ViT softmax) and forces a
rebuild of the merged cache, then runs the identical breakdown so we can see
whether the conclusions move.
"""
import os, sys, types, tempfile, runpy
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, SCRIPTS); os.chdir(SCRIPTS)

if 'seaborn' not in sys.modules:
    _sns = types.ModuleType('seaborn')
    _sns.set_palette = lambda *a, **k: None
    _sns.barplot = lambda *a, **k: None
    sys.modules['seaborn'] = _sns

import generate_paper_fig14 as fig14
# Canonical corrected DB (both fixes). PIM rows still come from vit_pim_database.csv
# (merged in by fig14._build_database_for_workload) and are unaffected by the bug.
_root = os.path.join(fig14.PROJECT_DIR, 'unified_database.csv')
fig14.DATABASE = _root
print("=" * 92)
print("CORRECTED RE-RUN — fig14.DATABASE repointed to canonical root:")
print("   ", _root)
import os as _os
print("    mtime:", __import__('datetime').datetime.fromtimestamp(_os.path.getmtime(_root)))
print("=" * 92)

mp = os.path.join(tempfile.gettempdir(), 'vit_merged_database.csv')
if os.path.exists(mp):
    print("removing stale merged cache:", mp)
    os.remove(mp)

# Run the identical breakdown logic against the corrected DB.
runpy.run_path(os.path.join(HERE, 'vit_op_breakdown.py'), run_name='__corrected__')
