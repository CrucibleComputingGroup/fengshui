#!/usr/bin/env python3
"""
Reproduce the NRE cost figure (nre_cost.png) from the MICRO-Mozart-2026 paper.

Figure caption: "System cost breakdown under different manufacturing volumes,
averaged across all evaluated networks, assuming a total of 200 different networks.
The panel on the right further details the major components of NRE cost."

Cost model: CATCH (arXiv:2503.15753) -- RE + NRE/V framework, sourced ENTIRELY from
chiplet_timeloop/scripts/get_cost.py (no hard-coded cost numbers):
  * RE cost (die + packaging): calculate_die_cost + compute_assembly, evaluated on the
    REAL per-net chiplet areas chosen by the optimizer for each paradigm, then averaged
    over the full Fig-9 workload suite (20 nets, energy-optimal non-cost-aware designs).
  * NRE cost: get_cost's C4 NRE model (nre_per_chiplet_type), calibrated to $14.8M per distinct
    design @16nm (the CATCH/IBS per-chiplet build-up used elsewhere in the paper; --nre-usd to
    override) -- times the number of distinct designs a paradigm must tape out, amortized / V.

Areas come from source_fig10_areas.py -> fig10_areas.json (real optimization results).

Usage:
    python generate_nre_cost_figure.py [--areas fig10_areas.json] [--save nre_cost.png]
"""

import sys
import os
import math
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox

# ─── Add scripts dir so we can import the real cost model ───
_SCRIPTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
sys.path.insert(0, _SCRIPTS_DIR)
from get_cost import calculate_die_cost, compute_assembly, nre_per_chiplet_type, CostParams

# ══════════════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════════════

# Number of total target networks (from paper caption)
N_NETWORKS = 200

# Manufacturing volumes to evaluate (in units)
VOLUMES = [1_000_000, 2_000_000, 3_000_000]
VOLUME_LABELS = ["1", "2", "3"]

# ── Operational (energy) cost ──
# ISO-WORK model: the electricity to serve a FIXED inference demand scales with energy
# PER INFERENCE (not instantaneous power), so a more energy-efficient design (lower Fig-9
# energy) costs less to run. op$/unit = geomean_energy(J) * lifetime_inferences * price/3.6e6 * PUE.
# (power x time would instead reward the slow generalist ASIC -- the wrong story.)
ELEC_PRICE_USD_PER_KWH = 0.10      # datacenter electricity price (assumption)
PUE = 1.2                           # datacenter power-usage-effectiveness overhead (assumption)
LIFETIME_INFERENCES = 1.0e11        # inferences over the unit's life (~5-yr deployment) (assumption)

def _packaging_cost(assembly):
    # Raw assembly cost straight from the cost model: C_assembly / Y_assembly, NO yield floor.
    # (The earlier ASSEMBLY_YIELD_FLOOR=0.95 hack was removed -- a hard floor masks the model
    #  rather than fixing it. Caveat: get_cost.compute_assembly models per-bond yield with no
    #  known-good-die / bond-repair, so it is pessimistic for very large 2.5D modules; the
    #  geomean-over-nets aggregation keeps the figure robust to the few large-module outliers.)
    return assembly["C_assembly"] / assembly["Y_assembly"]

# NRE per distinct chiplet design, taken through the cost model's amortization path.
# We CALIBRATE it to the CATCH/IBS per-chiplet build-up ($14.8M @16nm = sum of the 7 inset
# components) that the motivation figure (fig:nre_dilemma) and the paper prose already use,
# so Figure 10 stays internally consistent. The C4 model's RAW table value at 16nm is $50M
# (a full-SoC-design-cost UPPER bound); pass --nre-usd 50e6 to use that instead.
PROCESS_NODE = "16nm"
NRE_PER_DESIGN_USD = 14.8e6
COST_PARAMS = CostParams(process_node=PROCESS_NODE, nre_fixed_usd=NRE_PER_DESIGN_USD)
NRE_PER_DESIGN = nre_per_chiplet_type(0.0, COST_PARAMS)   # $ per distinct design (via cost model)

# ── NRE component breakdown, split per-DESIGN vs per-PRODUCT ──
# The cost model exposes only a per-design NRE *total*; the component split is the
# standard CATCH/IBS decomposition (note: NOT from CATCH's code -- CATCH's own NRE is
# design + mask + ATPG per die, with no software or per-package line).
#
# Corrected 2026-09-17.  The seven components do not all scale with the number of
# chiplet DESIGNS.  A "design" is one tapeout (one mask set); a "product" is one
# shipped accelerator (one package, one interposer layout).  They are different
# counts: the pool has 8 designs but serves a 200-model fleet.
#
#   per-DESIGN  Logic Mask, IP Licensing, Labor, Equip Validation, SW + System
#               SW + System is per-design for the same reason CUDA is: software
#               enablement (driver, compiler backend, firmware, validation suite)
#               is written once per architecture and shared by every product that
#               uses it -- you do not rewrite it per SKU.
#   per-PRODUCT Interposer Mask, Package Design
#               One interposer per package, one package layout per package.  These
#               are the only two lines whose attribution is unambiguous.
_NRE_PER_DESIGN_M = {
    "Logic Mask":       8.5,
    "IP Licensing":     2.5,
    "SW + System":      2.2,
    "Labor":            0.5,
    "Equip Validation": 0.3,
}
_NRE_PER_PRODUCT_M = {
    "Interposer Mask":  0.3,
    "Package Design":   0.5,
}
_NRE_PROPORTIONS_M = {**_NRE_PER_DESIGN_M, **_NRE_PER_PRODUCT_M}   # inset display order

# Products in the fleet.  Uniform across paradigms ON PURPOSE: every paradigm must
# serve all N_PRODUCTS networks, so the per-product term is the SAME constant for
# each and therefore cannot bias the comparison.  Counting distinct package
# geometries per paradigm instead would need a 20-net -> 200-net extrapolation that
# the data does not support (homo_asic's geometry count saturates -- it has one die
# at 1/2/8/9 copies -- while the pool's does not), and it would hand each paradigm a
# DIFFERENT additive constant.  Slight over-charge for paradigms that reuse a
# package layout; uniform, assumption-free and conservative for the pool.
N_PRODUCTS = 200

_prop_sum = sum(_NRE_PROPORTIONS_M.values()) * 1e6
_scale = NRE_PER_DESIGN / _prop_sum          # rescale to the calibrated $14.8M total
NRE_BREAKDOWN = {k: (v * 1e6) * _scale for k, v in _NRE_PROPORTIONS_M.items()}
NRE_DESIGN_TERM  = sum(_NRE_PER_DESIGN_M.values())  * 1e6 * _scale   # $14.0M
NRE_PRODUCT_TERM = sum(_NRE_PER_PRODUCT_M.values()) * 1e6 * _scale   # $0.8M


# ══════════════════════════════════════════════════════════════════════════════
# Cost computation (RE from real areas; NRE from the cost model)
# ══════════════════════════════════════════════════════════════════════════════

def _chiplet_re_cost(areas, bonding):
    """RE (die + packaging) for one chiplet-based accelerator."""
    total_die = sum(calculate_die_cost(a, bonding) for a in areas)
    total_area = sum(areas)
    assembly = compute_assembly(total_area, bonding, n_chips=len(areas))
    return total_die, _packaging_cost(assembly)


def _geomean(xs):
    """Geometric mean over networks -- the project convention (CLAUDE.md sec 9) and the
    same aggregation Figure 9 uses. Robust to the few MoE-prefill RE outliers that make
    the arithmetic mean unrepresentative (e.g. homo BASIC pkg mean $1226 vs median $62)."""
    a = np.asarray(xs, dtype=float)
    return float(np.exp(np.mean(np.log(a + 1e-30))))


def avg_re_cost(paradigm):
    """Geomean-over-nets (die_cost, packaging_cost) for a paradigm."""
    if paradigm.get("monolithic"):
        area = paradigm["area"]
        die_cost = calculate_die_cost(area, "2D")        # monolithic = single 2D die
        assembly = compute_assembly(area, "2D", n_chips=1)
        packaging = _packaging_cost(assembly)
        die_cost_only = die_cost - packaging             # calculate_die_cost includes substrate
        return die_cost_only, packaging
    bonding = paradigm.get("bonding", "2.5D")
    dies, pkgs = [], []
    for areas in paradigm["per_net_areas"].values():
        if not areas:
            continue
        d, p = _chiplet_re_cost(areas, bonding)
        dies.append(d)
        pkgs.append(p)
    return _geomean(dies), _geomean(pkgs)


def nre_per_unit(paradigm, volume):
    """Per-shipped-unit NRE, split per-design and per-product.

        (n_designs * NRE_DESIGN_TERM + N_PRODUCTS * NRE_PRODUCT_TERM) / volume

    Self-consistency check: when n_designs == N_PRODUCTS (homo_basic, 200 of each)
    this is identical to the old n_designs * $14.8M, because
    14.0 + 0.8 == 14.8.  That invariant is what keeps the paper's "$3,000 for a
    200-model fleet" claim unchanged.
    """
    return (paradigm["n_unique_designs"] * NRE_DESIGN_TERM
            + N_PRODUCTS * NRE_PRODUCT_TERM) / volume


def op_cost(energy_latency_per_net):
    """Lifetime operational (energy) cost in $/unit -- iso-work over LIFETIME_INFERENCES.
    energy_latency_per_net: {net: [energy_J, latency_s]} for this paradigm (geomean over nets)."""
    geo_energy_j = _geomean([e for e, _l in energy_latency_per_net.values()])
    kwh = geo_energy_j * LIFETIME_INFERENCES / 3.6e6
    return kwh * ELEC_PRICE_USD_PER_KWH * PUE


# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════

def make_figure(areas, power, save_path=None, include_op=True):
    strategies = ["homo_asic", "homo_basic", "het_unconstrained", "het_pool"]
    strategy_labels = [
        "Homogeneous\nASIC (all networks)",
        "Homogeneous\nBASIC (all networks)",
        "Heterogeneous\nBASIC(unconstrained)",
        "Heterogeneous\nBASIC(chiplet pool)",
    ]

    # Precompute RE (die, packaging) and operational cost per strategy (volume-independent)
    re = {s: avg_re_cost(areas[s]) for s in strategies}
    op = {s: (op_cost(power[s]) if include_op else 0.0) for s in strategies}

    # Gather data
    data = {}  # strategy -> list of (die, pkg, nre, op) per volume
    for strat in strategies:
        data[strat] = []
        avg_die, avg_pkg = re[strat]
        for vol in VOLUMES:
            data[strat].append((avg_die, avg_pkg, nre_per_unit(areas[strat], vol), op[strat]))

    # ── Layout ──
    # Single panel with a broken y-axis (NRE breakdown is described in prose, not inset).
    fig = plt.figure(figsize=(9, 4.8))
    ax_low = fig.add_axes([0.12, 0.26, 0.84, 0.28])   # bottom part (linear)
    ax_high = fig.add_axes([0.12, 0.58, 0.84, 0.37])  # top part (log scale)

    # Bar parameters
    n_groups = len(strategies)
    n_bars_per_group = len(VOLUMES)
    bar_width = 0.22
    group_gap = 0.4
    colors = {"die": "#7FB3D8", "pkg": "#F4A460", "nre": "#90EE90", "op": "#C39BD3"}

    x_positions = []
    for g in range(n_groups):
        base = g * (n_bars_per_group * bar_width + group_gap)
        for b in range(n_bars_per_group):
            x_positions.append(base + b * bar_width)

    # Plot bars on BOTH axes
    for ax in [ax_low, ax_high]:
        idx = 0
        for g, strat in enumerate(strategies):
            for b in range(n_bars_per_group):
                die, pkg, nre, op_c = data[strat][b]
                x = x_positions[idx]
                ax.bar(x, die, bar_width, color=colors["die"],
                       edgecolor='black', linewidth=0.5,
                       label='Die cost' if idx == 0 else None)
                ax.bar(x, pkg, bar_width, bottom=die, color=colors["pkg"],
                       edgecolor='black', linewidth=0.5,
                       label='Packaging cost' if idx == 0 else None)
                ax.bar(x, nre, bar_width, bottom=die + pkg, color=colors["nre"],
                       edgecolor='black', linewidth=0.5,
                       label='NRE cost' if idx == 0 else None)
                if include_op:
                    ax.bar(x, op_c, bar_width, bottom=die + pkg + nre, color=colors["op"],
                           edgecolor='black', linewidth=0.5,
                           label='Operational cost' if idx == 0 else None)
                idx += 1

    # ── Configure broken axis ──
    # Bottom (linear) panel must fully contain (a) every paradigm's RE (die+pkg) and
    # (b) the FULL stack of the low-NRE paradigms (ASIC / chiplet pool) so their small
    # totals stay readable; the high-NRE paradigms tower into the top (log) panel.
    re_totals = [re[s][0] + re[s][1] for s in strategies]
    low_nre_totals = [re[s][0] + re[s][1] + nre_per_unit(areas[s], VOLUMES[0]) + op[s]
                      for s in strategies if areas[s]['n_unique_designs'] <= 8]
    _need = max(max(re_totals), max(low_nre_totals))
    y_break_low = math.ceil(_need * 1.12 / 500) * 500
    y_break_high_min = y_break_low * 1.15
    _max_total = max(sum(data[s][b]) for s in strategies for b in range(n_bars_per_group))

    ax_low.set_ylim(0, y_break_low)
    ax_low.spines['top'].set_visible(False)

    ax_high.set_ylim(y_break_high_min, _max_total * 1.4)
    ax_high.set_yscale('log')
    ax_high.spines['bottom'].set_visible(False)
    ax_high.tick_params(bottom=False, labelbottom=False)

    # Break marks
    d = 0.015
    kwargs = dict(transform=ax_high.transAxes, color='k', clip_on=False, linewidth=1)
    ax_high.plot((-d, +d), (-d, +d), **kwargs)
    ax_high.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs = dict(transform=ax_low.transAxes, color='k', clip_on=False, linewidth=1)
    ax_low.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Shared x-axis settings
    group_centers = []
    for g in range(n_groups):
        base = g * (n_bars_per_group * bar_width + group_gap)
        center = base + (n_bars_per_group - 1) * bar_width / 2
        group_centers.append(center)

    ax_low.set_xticks(group_centers)
    ax_low.set_xticklabels(strategy_labels, fontsize=9)

    # Volume labels just under each bar; strategy names pushed further down (no overlap)
    _xtrans = ax_low.get_xaxis_transform()   # x in data coords, y in axes fraction
    for g in range(n_groups):
        base = g * (n_bars_per_group * bar_width + group_gap)
        for b in range(n_bars_per_group):
            x = base + b * bar_width
            ax_low.text(x, -0.04, VOLUME_LABELS[b], transform=_xtrans,
                        ha='center', va='top', fontsize=8)
    ax_low.tick_params(axis='x', length=0, pad=20)

    ax_low.set_ylabel("Cost ($)", fontsize=11)
    # single x-axis caption for the per-group volume sub-bars (1/2/3 M units)
    fig.text(0.54, 0.015, "Manufacturing volume (million units)", fontsize=9, ha='center')

    # Legend on top axes
    ax_high.legend(loc='upper left', fontsize=9, framealpha=0.9)

    # Align x-limits
    x_min = x_positions[0] - bar_width
    x_max = x_positions[-1] + bar_width * 2
    ax_low.set_xlim(x_min, x_max)
    ax_high.set_xlim(x_min, x_max)

    plt.suptitle("")
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    # Echo the numbers behind the figure for the record
    print(f"\nNRE per distinct design (cost model @ {PROCESS_NODE}): ${NRE_PER_DESIGN/1e6:.1f}M")
    print(f"Operational cost: ${ELEC_PRICE_USD_PER_KWH}/kWh, PUE {PUE}, {LIFETIME_INFERENCES:.0e} lifetime inferences")
    print(f"{'strategy':22s} {'avg_die$':>9s} {'avg_pkg$':>9s} {'op$':>8s} {'#designs':>9s}  NRE/unit @ 1M/2M/3M")
    for strat in strategies:
        ad, ap = re[strat]
        nd = areas[strat]['n_unique_designs']
        nres = "  ".join(f"${nre_per_unit(areas[strat], v):,.0f}" for v in VOLUMES)
        print(f"{strat:22s} {ad:9.2f} {ap:9.2f} {op[strat]:8.1f} {nd:9d}  {nres}")

    # NRE component breakdown (for the prose that replaces the old inset)
    print(f"\nNRE component breakdown (${NRE_PER_DESIGN/1e6:.1f}M total/design @ {PROCESS_NODE}):")
    for comp, val in NRE_BREAKDOWN.items():
        print(f"  {comp:18s} ${val/1e6:5.2f}M  ({100*val/NRE_PER_DESIGN:4.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _default_areas = os.path.join(os.path.dirname(__file__), 'fig10_areas.json')
    _default_power = os.path.join(os.path.dirname(__file__), 'fig10_power.json')
    parser.add_argument("--areas", type=str, default=_default_areas,
                        help="JSON of real per-net chiplet areas (from source_fig10_areas.py)")
    parser.add_argument("--power", type=str, default=_default_power,
                        help="JSON of per-net [energy_J, latency_s] (from source_fig10_power.py)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to this path (e.g., nre_cost.png)")
    parser.add_argument("--nre-usd", type=float, default=None,
                        help="Override NRE per distinct design (USD). Default $14.8M (paper-"
                             "consistent). Use 50e6 for the C4 model's raw 16nm table value.")
    parser.add_argument("--elec-price", type=float, default=None, help="electricity $/kWh")
    parser.add_argument("--lifetime-inferences", type=float, default=None,
                        help="inferences served over the unit's life (op-cost magnitude knob)")
    parser.add_argument("--no-opcost", action="store_true",
                        help="omit the operational-cost segment (3-segment die/pkg/NRE breakdown)")
    args = parser.parse_args()

    if args.nre_usd is not None:
        NRE_PER_DESIGN = float(args.nre_usd)
        NRE_BREAKDOWN = {k: (v * 1e6) * (NRE_PER_DESIGN / _prop_sum)
                         for k, v in _NRE_PROPORTIONS_M.items()}
    if args.elec_price is not None:
        ELEC_PRICE_USD_PER_KWH = args.elec_price
    if args.lifetime_inferences is not None:
        LIFETIME_INFERENCES = args.lifetime_inferences

    with open(args.areas) as f:
        areas = json.load(f)
    power = {}
    if not args.no_opcost:
        with open(args.power) as f:
            power = json.load(f)
    make_figure(areas, power, save_path=args.save, include_op=not args.no_opcost)
