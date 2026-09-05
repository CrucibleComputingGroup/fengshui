#!/usr/bin/env python3
"""Figure 10 panel (b): cost-model parameter sensitivity (tornado).
Defends: the chiplet pool stays cost-effective -- far cheaper in TOTAL cost than fully-bespoke
(unconstrained) heterogeneity -- across all plausible cost-model assumptions. Metric is the
advantage factor = unconstrained_total / pool_total per unit, at the reference volume.
Reuses the SAME cost model + areas/energy as panel (a) (fig10_areas/power.json).

C4 (Reviewer B): the tornado now names INTERPOSER cost and ELECTRICITY ($/kWh) as their own swept
knobs (previously interposer was folded inside "Packaging cost" and operational cost was a fixed
component never swept). Both enter the pool-vs-unconstrained dollar ratio directly through get_cost
-- this is post-processing on the FIXED per-net areas / workload memory footprints, NOT a B(N)
re-search: interposer $/mm^2 scales via CostParams.interposer_cost_scale (already threaded into
compute_assembly), and the operational $ term = (per-net energy) x lifetime-inferences x
price/3.6e6 x PUE.

WHY MEMORY $/GB IS NOT A TORNADO KNOB (Reviewer B's question):
  per-unit DRAM $ = geomean over nets of (provisioned_GB(net) x $/GB).  This is a COMMON-MODE
  term: the chiplet pool and the unconstrained-bespoke design serve the SAME 200-net fleet, at
  the SAME precision, with the SAME baseline DRAM type (GDDR7) -- so each ships the identical
  ~$28/unit of memory.  In the advantage ratio  uncon_total / pool_total = (U0 + M)/(P0 + M)
  the memory term M adds to BOTH numerator and denominator, so changing $/GB can only pull the
  ratio toward break-even by a bounded amount -- it can never flip the verdict (even across the
  full LPDDR5 $2.31 -> HBM3 $110/GB market it moved the advantage only ~12.6x -> 10.1x).  Memory
  $ is therefore kept as a FIXED component of total() but is NOT swept as its own bar.  Memory's
  *design* value in Fengshui is the per-workload DRAM-TYPE choice (LPDDR5 vs HBM3), which shows up
  as energy/area in Fig 9 and the C3 ablation (homogeneous-GDDR7 costs 1.43x more) -- not a $/GB
  lever in this pool-vs-bespoke robustness panel."""
import os, sys, json, re, argparse
import numpy as np
import matplotlib.pyplot as plt

_SCRIPTS = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
_TIMELOOP = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'timeloop'))
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, _TIMELOOP)
from get_cost import calculate_die_cost, compute_assembly, CostParams, ASSEMBLY_DB
from mem_spec import get_memory_spec   # DRAM provisioning -> $ term (cost = provisioned_gb * cost_per_GB)

HERE = os.path.dirname(__file__)
AREAS = json.load(open(os.path.join(HERE, 'fig10_areas.json')))
POWER = json.load(open(os.path.join(HERE, 'fig10_power.json')))

REF_VOLUME = 1_000_000
NRE_DEFAULT = 14.8e6
PRICE_DEFAULT, PUE, LIFE_DEFAULT = 0.10, 1.2, 1.0e11
INCLUDE_OPCOST = True     # op-cost IS a component of total cost, matching the breakdown
MEM_DRAM = 'GDDR7'        # baseline DRAM for the per-unit memory $ term (== Fig 9 / panel-(a) normalization)


def _geo(xs):
    a = np.asarray(xs, float)
    return float(np.exp(np.mean(np.log(a + 1e-30))))


# --- per-net DRAM capacity (GB) from the SAME static workload table the perf model reads ---
# DRAM footprint per net = sum of per-layer weight + input + output memory (the 'single' fusion
# variant, to avoid double counting boundaries). Both paradigms serve this identical 20-net suite
# at identical precision, so the provisioned DRAM -- and hence its $ -- is the same for pool and
# unconstrained; the sweep scales it via CostParams.mem_cost_scale. No pytimeloop / DB needed.
def _build_net_dram_gb():
    import pandas as pd
    df = pd.read_csv(os.path.join(_SCRIPTS, 'network_analysis.csv'))
    out = {}
    for tag in AREAS['meta']['nets']:
        m = re.match(r'(.+?)_b(\d+)_seq(\d+)', tag)
        base, b, s = m.group(1), int(m.group(2)), int(m.group(3))
        sub = df[(df.net_name == base) & (df.batch_size == b) & (df.sequence_length == s)]
        one = sub[sub.fused_layer_type == 'single'] if 'single' in sub.fused_layer_type.values else sub
        out[tag] = float(one.groupby('layer_name')['weight_mem'].first().sum()
                         + one.groupby('layer_name')['in_mem'].first().sum()
                         + one.groupby('layer_name')['out_mem'].first().sum())
    return out


_NET_DRAM_GB = _build_net_dram_gb()


def mem_cost(mem_cost_scale=1.0):
    """Per-unit DRAM $ (geomean over nets) at the baseline DRAM type, scaled by mem_cost_scale.
    Provision each net's footprint via get_memory_spec, take its $ (= provisioned_gb * cost_per_GB),
    geomean over nets (the figure's standard aggregation), then apply the sweep knob."""
    return _geo([get_memory_spec(gb, MEM_DRAM)['cost'] for gb in _NET_DRAM_GB.values()]) * mem_cost_scale


def re_cost(parad, params):
    if parad.get('monolithic'):
        return calculate_die_cost(parad['area'], '2D', params)
    bonding = parad.get('bonding', '2.5D')
    dies, pks = [], []
    for areas in parad['per_net_areas'].values():
        if not areas:
            continue
        dies.append(sum(calculate_die_cost(a, bonding, params) for a in areas))
        asm = compute_assembly(sum(areas), bonding, n_chips=len(areas), params=params)
        pks.append(asm['C_assembly'] / asm['Y_assembly'])   # raw assembly yield, no floor
    return _geo(dies) + _geo(pks)


def op_cost(key, price, life):
    return _geo([e for e, _l in POWER[key].values()]) * life / 3.6e6 * price * PUE


def total(key, params, nre, price, life, volume):
    p = AREAS[key]
    t = re_cost(p, params) + p['n_unique_designs'] * nre / volume
    if INCLUDE_OPCOST:
        t += op_cost(key, price, life)
    t += mem_cost(params.mem_cost_scale)   # DRAM $ term (memory-cost sweep knob)
    return t


def factor(params=None, nre=NRE_DEFAULT, price=PRICE_DEFAULT, life=LIFE_DEFAULT, volume=REF_VOLUME):
    params = params or CostParams(process_node='16nm', nre_fixed_usd=nre)
    return (total('het_unconstrained', params, nre, price, life, volume) /
            total('het_pool', params, nre, price, life, volume))


def P(**kw):
    return CostParams(process_node='16nm', nre_fixed_usd=NRE_DEFAULT, **kw)


# --- absolute swept-parameter endpoints, with UNITS (Reviewer B) -------------------------------
# Each tornado endpoint is annotated with the ABSOLUTE parameter value (and its unit), never a
# 0.5x/2x multiplier, so the swept knob (domain) is never confused with the x-axis advantage ratio
# (codomain, the only quantity in "x"). Defaults below are pulled from the cost model itself
# (CostParams / ASSEMBLY_DB / MEM_SPECS) so the labels can't drift from the numbers they sweep.
_DP        = CostParams()                                   # cost-model defaults = source of truth
_WAFER0    = _DP.wafer_cost                                 # $1375 / wafer
_D0        = _DP.defect_density_D0                          # 0.008 defects / mm^2
_PKG_MAT0  = ASSEMBLY_DB['2.5D']['materials_cost_per_mm2']  # $0.12 / mm^2 assembly materials
_INTERP0   = 0.0219                                         # $/mm^2 interposer (get_cost.compute_assembly)
_MM2 = "/mm$^2$"   # matplotlib-ready superscript unit


# (label, lo_param_label, lo_factor, hi_param_label, hi_factor); labels are matplotlib-ready.
# Physical cost-model constants are swept +-2x around their default (the standard "how wrong could
# this constant be" band); VOLUME / NRE / ELECTRICITY use realistic deployment ranges. The label is
# the ABSOLUTE value at each end -- see _DP-derived numbers above. (Memory $/GB is a FIXED common-
# mode term in total(), NOT a tornado knob -- it cancels in the ratio; see the module docstring.)
SWEEPS = [
    ("NRE / design",       "\\$6M",                       factor(nre=6e6),
                           "\\$50M",                      factor(nre=50e6)),
    ("Production volume",  "5M",                          factor(volume=5_000_000),
                           "0.5M",                        factor(volume=500_000)),
    ("Packaging cost",     f"\\${_PKG_MAT0*0.5:.2f}{_MM2}", factor(params=P(assembly_material_scale=0.5)),
                           f"\\${_PKG_MAT0*2.0:.2f}{_MM2}", factor(params=P(assembly_material_scale=2.0))),
    ("Interposer cost",    f"\\${_INTERP0*0.5:.3f}{_MM2}",  factor(params=P(interposer_cost_scale=0.5)),
                           f"\\${_INTERP0*2.0:.3f}{_MM2}",  factor(params=P(interposer_cost_scale=2.0))),
    ("Wafer cost",         f"\\${_WAFER0*0.5:.0f}",       factor(params=P(wafer_cost=_WAFER0*0.5)),
                           f"\\${_WAFER0*2.0:.0f}",       factor(params=P(wafer_cost=_WAFER0*2.0))),
    ("Die defect density", f"{_D0*0.5:.3f}{_MM2}",        factor(params=P(yield_defect_scale=0.5)),
                           f"{_D0*2.0:.3f}{_MM2}",        factor(params=P(yield_defect_scale=2.0))),
    ("Electricity price",  "\\$0.05/kWh",                 factor(price=0.05),
                           "\\$0.20/kWh",                 factor(price=0.20)),
]

BASE = factor()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    # sort by impact (range), largest at top of the tornado
    rows = sorted(SWEEPS, key=lambda r: abs(r[4] - r[2]))
    print(f"baseline advantage factor (uncon/pool) = {BASE:.2f}x  @ vol={REF_VOLUME:,}")
    for lbl, lv, lf, hv, hf in reversed(rows):
        print(f"  {lbl:20s} {lv:>6s}->{lf:6.2f}x   {hv:>6s}->{hf:6.2f}x   (range {abs(hf-lf):.2f})")

    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.set_xscale('log')
    for i, (lbl, lv, lf, hv, hf) in enumerate(rows):
        lo, hi = min(lf, hf), max(lf, hf)
        ax.barh(i, hi - lo, left=lo, height=0.62, color="#7FB3D8", edgecolor='black', linewidth=0.6)
        # annotate the parameter value at each end
        lo_lab, hi_lab = (lv, hv) if lf <= hf else (hv, lv)
        ax.text(lo, i, f" {lo_lab}", va='center', ha='right', fontsize=7.5, color='#333')
        ax.text(hi, i, f"{hi_lab} ", va='center', ha='left', fontsize=7.5, color='#333')
    ax.axvline(BASE, color='#555', linestyle='-', linewidth=1.0)
    ax.text(BASE, len(rows) - 0.3, f' baseline {BASE:.0f}×', fontsize=7.5, color='#555', va='bottom', ha='left')
    ax.axvline(1.0, color='#C0392B', linestyle='--', linewidth=1.1)
    ax.text(1.0, len(rows) - 0.3, ' break-even', fontsize=7.5, color='#C0392B', va='bottom', ha='left')

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("Pool total-cost advantage over\nunconstrained heterogeneity ($\\times$)", fontsize=10)
    ax.set_xlim(0.8, max(hf for *_, hf in SWEEPS) * 1.7)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='x', which='both', linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=300, bbox_inches='tight')
        print("Saved", args.save)
    else:
        plt.show()
