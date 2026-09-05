"""C4: Cost-model sensitivity sweep -- is the optimal chiplet-pool size robust?

Background (see Phase-0 finding): the per-unit cost model in get_cost.py is
monotone in pool size N -- adding chiplet types only ever improves the optimized
objective, so there is no true optimum until we add a per-distinct-design NRE
amortized over production volume. With that term:

    TCO(N) = opex_anchor_usd * B(N)/B(1)        # operational value of flexibility
             + amortized_NRE(N)                  # = sum_over_N(NRE_type) / volume

B(N) is the empirical "best objective vs N" curve from the incremental sweep
(incremental_chiplet_sweep_*.csv). The first term decreases and saturates in N;
the second rises linearly -> TCO is U-shaped with a real argmin. C4 asks whether
that argmin is robust as we vary the cost parameters.

Two classes of sweep:
  * NRE, production volume, GPU baseline   -> additive on top of B(N); pure
    post-processing, runs anywhere (get_cost.py is standalone). DONE HERE.
  * interposer/packaging, chiplet yield, memory cost -> baked into B(N) via the
    cost-aware objective, so they require RE-RUNNING the incremental sweep with
    modified CostParams. That needs pytimeloop + the (gitignored) area/perf
    database. Wired up in rerun_benefit_curve(); guarded so it fails loudly if
    the pipeline is unavailable.

All NRE / volume / GPU / anchor numbers are LITERATURE-DEFAULT ASSUMPTIONS -- see
get_cost.NRE_FIXED_BY_NODE_USD -- and should be confirmed with the cost-model
owner (the maintainer) before final figures.

Run:  python3 sweep_cost_sensitivity.py            (uses env with pandas; no pytimeloop needed)
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import get_cost as gc

# Per-unit operational/manufacturing dollar anchor at N=1. Used only to put the
# (dimensionless) empirical benefit curve and the (dollar) NRE term in the same
# units. Representative accelerator silicon+memory cost. ASSUMPTION.
DEFAULT_OPEX_ANCHOR_USD = 400.0


# ---------------------------------------------------------------------------
# Empirical benefit curve B(N) from the incremental sweep CSVs
# ---------------------------------------------------------------------------
def load_benefit_curve(csv_path: str, objective: str) -> Dict[int, float]:
    """Return {N: best_objective(N)} from an incremental_chiplet_sweep CSV."""
    col = f"best_{objective}"
    curve: Dict[int, float] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if col not in reader.fieldnames:
            raise ValueError(f"{csv_path} has no column {col}; has {reader.fieldnames[:4]}...")
        for row in reader:
            n = int(float(row["n_chiplets"]))
            curve[n] = float(row[col])
    if not curve:
        raise ValueError(f"no rows parsed from {csv_path}")
    return dict(sorted(curve.items()))


# ---------------------------------------------------------------------------
# Total cost of ownership and optimal pool size
# ---------------------------------------------------------------------------
def tco_curve(
    benefit: Dict[int, float],
    params: gc.CostParams,
    opex_anchor_usd: float,
    chiplet_area_mm2: float = 0.0,
) -> Dict[int, float]:
    """TCO(N) in USD/unit = opex_anchor * B(N)/B(Nmin) + amortized_NRE(N)."""
    n_min = min(benefit)
    b_ref = benefit[n_min]
    out: Dict[int, float] = {}
    for n, b in benefit.items():
        opex = opex_anchor_usd * (b / b_ref)
        nre = gc.amortized_nre_per_unit([chiplet_area_mm2] * n, params)
        out[n] = opex + nre
    return out


def optimal_pool_size(
    benefit: Dict[int, float],
    params: gc.CostParams,
    opex_anchor_usd: float,
    chiplet_area_mm2: float = 0.0,
) -> Tuple[int, Dict[int, float]]:
    tco = tco_curve(benefit, params, opex_anchor_usd, chiplet_area_mm2)
    best_n = min(tco, key=tco.get)
    return best_n, tco


def near_optimal_basin(tco: Dict[int, float], tol: float = 0.02) -> List[int]:
    """Pool sizes whose TCO is within `tol` of the minimum. The SA-derived benefit
    curve is noisy, so the basin is a more honest robustness object than argmin."""
    mn = min(tco.values())
    return [n for n, v in tco.items() if v <= mn * (1.0 + tol)]


# ---------------------------------------------------------------------------
# Post-processing sweeps (run anywhere)
# ---------------------------------------------------------------------------
def sweep_one_knob(
    benefit: Dict[int, float],
    knob: str,
    scales: List[float],
    base_params: gc.CostParams,
    opex_anchor_usd: float,
) -> List[dict]:
    """Vary a single CostParams field by `scales` and record the optimal N."""
    rows = []
    for s in scales:
        if knob == "production_volume":
            p = replace(base_params, production_volume=base_params.production_volume * s)
        elif knob == "nre":
            p = replace(base_params, nre_scale=base_params.nre_scale * s)
        elif knob == "gpu_baseline":
            p = replace(base_params, gpu_baseline_cost_usd=base_params.gpu_baseline_cost_usd * s)
        else:
            raise ValueError(f"knob '{knob}' is not a post-processing knob; use rerun_benefit_curve")
        best_n, tco = optimal_pool_size(benefit, p, opex_anchor_usd)
        basin = near_optimal_basin(tco)
        rows.append({"knob": knob, "scale": s, "optimal_N": best_n,
                     "basin_lo": min(basin), "basin_hi": max(basin)})
    return rows


def sweep_nre_volume_grid(
    benefit: Dict[int, float],
    base_params: gc.CostParams,
    opex_anchor_usd: float,
    nre_scales: List[float],
    vol_scales: List[float],
) -> List[dict]:
    """2-D NRE x production-volume heatmap of optimal N."""
    rows = []
    for ns in nre_scales:
        for vs in vol_scales:
            p = replace(
                base_params,
                nre_scale=base_params.nre_scale * ns,
                production_volume=base_params.production_volume * vs,
            )
            best_n, tco = optimal_pool_size(benefit, p, opex_anchor_usd)
            basin = near_optimal_basin(tco)
            rows.append({"nre_scale": ns, "vol_scale": vs, "optimal_N": best_n,
                         "basin_lo": min(basin), "basin_hi": max(basin)})
    return rows


# ---------------------------------------------------------------------------
# Re-run sweeps (need pytimeloop + database) -- interposer / yield / memory
# ---------------------------------------------------------------------------
def rerun_benefit_curve(
    params: gc.CostParams,
    n_start: int = 1,
    n_end: int = 16,
    objective: str = "energy",
    database_file: str = "final_database.csv",
) -> Dict[int, float]:
    """Recompute B(N) with modified cost params. Requires the full pipeline
    (pytimeloop + area/perf database). The interposer/yield/memory sweeps change
    B(N) itself, so they must go through here rather than the post-processing path.

    NOTE: get_cost.calculate_die_cost / compute_assembly already accept `params`;
    to make this end-to-end, cal_perf_phy_net must thread `params` into its
    calculate_die_cost(...) call (cal_perf_phy_net.py:370) and the memory-cost path
    must use get_cost.scaled_mem_cost_per_gb(..., params). Left as the wiring TODO
    for the env that has the database.
    """
    try:
        from chiplet_sel import run_incremental_n_chiplet_sweep  # noqa: F401
        from network_dataclass import VirtualNetwork  # noqa: F401
    except Exception as e:  # pragma: no cover - depends on pytimeloop
        raise RuntimeError(
            "rerun_benefit_curve needs the full pipeline (pytimeloop + "
            f"{database_file}); not available here. Underlying import error: {e}\n"
            "Run this sweep in the maintainer's environment, or provide the database so "
            "the post-processing path can be extended."
        )
    raise NotImplementedError(
        "Pipeline import succeeded -- finish wiring `params` into cal_perf_phy_net "
        "(see docstring) before enabling interposer/yield/memory sweeps."
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _fmt_curve(curve: Dict[int, float]) -> str:
    return "  ".join(f"N{n}={v:.3f}" for n, v in curve.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benefit-csv", default="incremental_chiplet_sweep_energy_True.csv",
                    help="incremental_chiplet_sweep CSV providing B(N)")
    ap.add_argument("--objective", default="energy", choices=["energy", "edp"])
    ap.add_argument("--opex-anchor", type=float, default=DEFAULT_OPEX_ANCHOR_USD,
                    help="per-unit operational $ anchor at N=1 (ASSUMPTION)")
    ap.add_argument("--node", default="7nm", help="process node for NRE lookup")
    ap.add_argument("--out-prefix", default="c4_sensitivity")
    args = ap.parse_args()

    if not os.path.exists(args.benefit_csv):
        raise SystemExit(f"benefit CSV not found: {args.benefit_csv}")

    benefit = load_benefit_curve(args.benefit_csv, args.objective)
    base = replace(gc.DEFAULT_COST_PARAMS, process_node=args.node)

    print("=" * 72)
    print(f"C4 cost-model sensitivity sweep   (objective={args.objective}, node={args.node})")
    print(f"benefit curve B(N) from {args.benefit_csv}")
    print(f"  B(N): {_fmt_curve(benefit)}")
    print(f"  opex anchor = ${args.opex_anchor:.0f}/unit (assumption)")
    nre_unit = gc.nre_per_chiplet_type(0.0, base) / base.production_volume
    print(f"  baseline NRE/type = ${gc.nre_per_chiplet_type(0.0, base):,.0f}, "
          f"volume = {base.production_volume:,.0f} -> ${nre_unit:.1f}/unit/type")

    base_n, base_tco = optimal_pool_size(benefit, base, args.opex_anchor)
    base_basin = near_optimal_basin(base_tco)
    print(f"\nBaseline optimal pool size N* = {base_n}   near-optimal basin(<=2%) = {base_basin}")
    print(f"  TCO(N): {_fmt_curve(base_tco)}")

    # 1-D sweeps
    scales = [0.5, 0.7, 1.0, 1.5, 2.0]
    all_rows = []
    print("\n--- single-knob sweeps (optimal N* vs scale; basin in CSV) ---")
    for knob in ["nre", "production_volume", "gpu_baseline"]:
        rows = sweep_one_knob(benefit, knob, scales, base, args.opex_anchor)
        all_rows.extend(rows)
        summary = "  ".join(f"{r['scale']}x->N{r['optimal_N']}" for r in rows)
        print(f"  {knob:18s}: {summary}")

    with open(f"{args.out_prefix}_1d.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["knob", "scale", "optimal_N", "basin_lo", "basin_hi"])
        w.writeheader()
        w.writerows(all_rows)

    # 2-D NRE x volume grid
    grid_scales = [0.5, 0.7, 1.0, 1.5, 2.0]
    grid = sweep_nre_volume_grid(benefit, base, args.opex_anchor, grid_scales, grid_scales)
    with open(f"{args.out_prefix}_nre_volume_grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["nre_scale", "vol_scale", "optimal_N", "basin_lo", "basin_hi"])
        w.writeheader()
        w.writerows(grid)
    ns_set = sorted({r["optimal_N"] for r in grid})
    print(f"\n--- NRE x volume grid ({len(grid)} points) ---")
    print(f"  optimal N* ranges over {ns_set} across the grid")

    # robustness verdict
    all_optima = [r["optimal_N"] for r in all_rows] + [r["optimal_N"] for r in grid]
    lo, hi = min(all_optima), max(all_optima)
    print("\n" + "=" * 72)
    print(f"ROBUSTNESS: across all post-processing sweeps, N* in [{lo}, {hi}] "
          f"(baseline {base_n}).")
    print("  Note: interposer/yield/memory sweeps perturb B(N) itself and need the "
          "SA rerun (rerun_benefit_curve) in an env with pytimeloop + the database.")
    print(f"  wrote {args.out_prefix}_1d.csv, {args.out_prefix}_nre_volume_grid.csv")


if __name__ == "__main__":
    main()
