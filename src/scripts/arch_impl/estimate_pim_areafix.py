#!/usr/bin/env python3
"""
Impact of the CORRECT fix: price off-CP PIM at PIM_DIE_AREA_MM2 (224 mm^2) instead of
the buggy 3.88 mm^2 PE-array area, while KEEPING PIM available off-CP (preserving its
real energy/latency benefit). Monkeypatches _cached_chiplet_area for PIM only (the main
path never calls it for PIM, so only the off-CP path L1715 is affected).

Expectation: Energy/EDP unchanged (ratio 1.000); EC/EDPc shift slightly (off-CP PIM die
now costed correctly).
"""
import os, sys, math

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(THIS_DIR, ".."))
for _p in (SCRIPTS, THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cal_perf_phy_net as cpp
import global_parameter as gp
from baseline import _build_virtual_nets
from estimate_pim_offcp_impact import load_pool, geomean, evaluate, DB

_orig_area = cpp._cached_chiplet_area
def _pim_area_fixed(arch, pe_x_scale, pe_y_scale, glb_scale):
    if arch == 'PIM':
        return {'total_area_mm2': gp.PIM_DIE_AREA_MM2}
    return _orig_area(arch, pe_x_scale, pe_y_scale, glb_scale)

METRICS = [("Energy", "energy", False), ("Energy x Cost", "energy", True),
           ("EDP", "edp", False), ("EDP x Cost", "edp", True)]

vnets = _build_virtual_nets(database_file=DB)
print(f"\n{'metric':14s} {'now (buggy)':>13s} {'area-fixed':>13s} {'ratio':>7s}   decode-b1 nets")
for label, obj, ca in METRICS:
    cpp._cached_chiplet_area = _orig_area
    cur = evaluate(vnets, obj, ca)
    cpp._cached_chiplet_area = _pim_area_fixed
    fix = evaluate(vnets, obj, ca)
    cpp._cached_chiplet_area = _orig_area
    gnow, gfix = geomean(cur.values()), geomean(fix.values())
    changed = sorted(((fix[n]/cur[n]-1)*100, n) for n in cur
                     if cur[n] and math.isfinite(cur[n]) and abs(fix[n]/cur[n]-1) > 0.001)
    tag = "; ".join(f"{n.split('_')[0]}-{'dec' if 'decode' in n else 'pf'}-b{'1' if '_b1' in n else '8'} {d:+.1f}%"
                    for d, n in changed[-3:]) or "none"
    print(f"{label:14s} {gnow:13.4e} {gfix:13.4e} {gfix/gnow:7.3f}   {tag}")
