"""Quick verification: evaluate n=1's best chiplet alone, then as n=2 with PIM."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_archgym_chiplet import (
    evaluate_chiplet_group, _create_pim_chiplet, create_switch_chiplet,
    setup_virtual_nets,
)
from chiplet_dataclass import ChipletConfig

DATABASE = '../unified_database.csv'
OBJECTIVE = 'energy'

print("Loading virtual networks...")
virtual_nets = setup_virtual_nets(DATABASE, include_cnn=True)
print(f"Loaded {len(virtual_nets)} virtual networks\n")

# n=1 best chiplet from SAEO run
best_chiplet = ChipletConfig('eyeriss_like', 1, 1, 2, dram_type='HBM3')

# Test 1: n=1 — just the best chiplet
group_n1 = [best_chiplet]
val_n1, _ = evaluate_chiplet_group(group_n1, virtual_nets, OBJECTIVE, DATABASE)
print(f"n=1 [eyeriss_like alone]:          {val_n1:.4f}")

# Test 2: n=2 — best chiplet + PIM
group_n2_pim = [best_chiplet, _create_pim_chiplet()]
val_n2_pim, _ = evaluate_chiplet_group(group_n2_pim, virtual_nets, OBJECTIVE, DATABASE)
print(f"n=2 [eyeriss_like + PIM]:          {val_n2_pim:.4f}")

# Test 3: n=2 — best chiplet + switch
group_n2_sw = [best_chiplet, create_switch_chiplet()]
val_n2_sw, _ = evaluate_chiplet_group(group_n2_sw, virtual_nets, OBJECTIVE, DATABASE)
print(f"n=2 [eyeriss_like + switch]:       {val_n2_sw:.4f}")

# Test 4: n=2 — duplicate the same chiplet
group_n2_dup = [best_chiplet, ChipletConfig('eyeriss_like', 1, 1, 2, dram_type='HBM3')]
val_n2_dup, _ = evaluate_chiplet_group(group_n2_dup, virtual_nets, OBJECTIVE, DATABASE)
print(f"n=2 [eyeriss_like x2]:             {val_n2_dup:.4f}")

# Test 5: n=2 SAEO's best compute chiplet alone (without PIM)
saeo_n2_chiplet = ChipletConfig('gemmini_like', 9, 4, 3, dram_type='HBM3')
group_saeo = [saeo_n2_chiplet]
val_saeo, _ = evaluate_chiplet_group(group_saeo, virtual_nets, OBJECTIVE, DATABASE)
print(f"n=1 [gemmini_like from SAEO n=2]:  {val_saeo:.4f}")

print("\nIf n=2 values > n=1 value, evaluation itself treats n=2 differently.")
print("If n=2 values <= n=1 value, SAEO just didn't find the good chiplet at n=2.")

# ---- Detailed per-network breakdown for n=1 ----
print("\n" + "="*70)
print("Per-network breakdown: eyeriss_like@glb1@pe_x_scale1@pe_y_scale2@HBM3")
print("="*70)

from run_archgym_chiplet import evaluate_chiplet_group
from utility_functions import calculate_average_opt_value
from chiplet_sel import run_single_optimization
import math

group_test = [ChipletConfig('eyeriss_like', 1, 1, 2, dram_type='HBM3')]
_, results = run_single_optimization(
    virtual_nets=virtual_nets,
    chiplet_group=group_test,
    objective=OBJECTIVE,
    results_file=DATABASE,
    cost_aware=False,
    use_sequential=True,
    n_workers=8,
    use_dag_cp=True,
    v_het_batch=True,
)

values = []
for net_name, net_result in sorted(results.items()):
    v = net_result['min_value']
    values.append(v)
    print(f"  {net_name:<45} {v:.6e}")

import numpy as np
finite = [v for v in values if math.isfinite(v) and v > 0]
if finite:
    geo_mean = np.exp(np.mean(np.log(finite)))
    print(f"\nGeometric mean ({len(finite)} finite): {geo_mean:.6e}")
    print(f"Arithmetic mean: {np.mean(finite):.6e}")
    print(f"Min: {np.min(finite):.6e}, Max: {np.max(finite):.6e}")
