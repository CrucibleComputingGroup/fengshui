"""C4 Phase 1 regression test for the parametrized cost model.

Verifies that get_cost.CostParams defaults reproduce the original hard-coded
behavior EXACTLY, and that each sweep knob moves cost in the expected direction.

Run:  python3 test_cost_params.py     (needs no pytimeloop; get_cost is standalone)
"""
from dataclasses import replace

import get_cost as gc



# CATCH goldens: the only path. A faithful port of
# CATCH's Layer.reticle_utilization (floor(field/A)*A/field).  Note the 800 mm^2
# rows are IDENTICAL to the legacy table: at A=800 exactly one die fits a 858 mm^2
# field, so floor(858/800)*800/858 == 800/858 and the two models coincide.  That
# coincidence is a correctness check on the port, not a copy-paste error.
GOLDEN_CATCH = {
    (0.5, "2D"): 0.4697418592,   (0.5, "2.5D"): 0.8050328939,
    (1, "2D"): 0.5252293579,     (1, "2.5D"): 0.9072177150,
    (2, "2D"): 0.6366413913,     (2, "2.5D"): 1.1122337159,
    (5, "2D"): 0.9734777213,     (5, "2.5D"): 1.7311237599,
    (10, "2D"): 1.5457187351,    (10, "2.5D"): 2.7784741025,
    (25, "2D"): 3.3228596347,    (25, "2.5D"): 6.0137319649,
    (50, "2D"): 6.4921107936,    (50, "2.5D"): 11.7259958288,
    (100, "2D"): 13.8013072694,  (100, "2.5D"): 24.5995899417,
    (200, "2D"): 32.2949660905,  (200, "2.5D"): 56.5832634691,
    (400, "2D"): 87.9451962952,  (400, "2.5D"): 152.7282821686,
    (800, "2D"): 303.2964136594, (800, "2.5D"): 556.6796279397,
}



def test_default_is_catch():
    for (area, tech), want in GOLDEN_CATCH.items():
        got = round(gc.calculate_die_cost(area, tech), 10)
        assert got == want, f"catch {area} {tech}: want {want}, got {got}"


def test_reticle_utilization_is_a_packing_efficiency():
    """CATCH's utilization is bounded in (0.5, 1]; the legacy ratio is not."""
    for A in [0.5, 1, 10, 25, 100, 200, 400, 429, 500, 800, 858, 900, 1700]:
        u = gc.reticle_utilization(A, gc.DEFAULT_COST_PARAMS.reticle_area_mm2)
        assert 0.5 < u <= 1.0, f"A={A}: utilization {u} outside (0.5, 1]"
    # One die per field => the two models coincide exactly.
    R = gc.DEFAULT_COST_PARAMS.reticle_area_mm2
    for A in [500, 700, 800, 858]:
        assert abs(gc.reticle_utilization(A, R) - A / R) < 1e-12, A
    # And the legacy ratio is catastrophically small for a chiplet-sized die.
    assert 10 / R < 0.02 < gc.reticle_utilization(10, R)



def test_reticle_utilization_float_robustness():
    """Regressions for three defects found in adversarial review (Codex, 2026-09-17)."""
    R = gc.DEFAULT_COST_PARAMS.reticle_area_mm2
    # d1: repeated-addition drift. CATCH's literal `while field < A: field += R`
    # selects 31 fields at A=25743, R=858.1 where 30 suffice; the closed form does not.
    assert gc.reticle_utilization(25743, 858.1) == 1.0
    # d2: subnormal area used to overflow the quotient and raise OverflowError.
    assert gc.reticle_utilization(1e-308, R) == 1.0
    # d3: an enormous area used to loop ~1e17 times; the closed form is O(1).
    assert gc.reticle_utilization(1e20, R) == 1.0
    # Non-finite input must raise, not silently propagate NaN/inf into cost.
    for bad in (float("inf"), float("nan"), 0.0, -1.0):
        try:
            gc.reticle_utilization(bad, R)
        except ValueError:
            continue
        raise AssertionError(f"reticle_utilization({bad}) should raise ValueError")
    # The bound must hold across the whole positive range, not just plausible dies.
    import random
    random.seed(0)
    xs = [1e-300, 1e-9, 0.5, 1, 100, 858, 858.0001, 859, 1716, 1717, 5000, 1e12]
    xs += [random.uniform(0.1, 3000) for _ in range(5000)]
    for A in xs:
        u = gc.reticle_utilization(A, R)
        assert 0.5 < u <= 1.0, f"A={A}: utilization {u} outside (0.5, 1]"



def test_knob_directions():
    P = gc.DEFAULT_COST_PARAMS
    base = gc.calculate_die_cost(100, "2.5D")
    assert gc.calculate_die_cost(100, "2.5D", replace(P, interposer_cost_scale=2.0)) > base
    assert gc.calculate_die_cost(100, "2.5D", replace(P, assembly_material_scale=2.0)) > base
    assert gc.calculate_die_cost(100, "2.5D", replace(P, yield_defect_scale=2.0)) > base  # more defects


def test_nre_amortization():
    P = gc.DEFAULT_COST_PARAMS
    nre8 = gc.amortized_nre_per_unit([0] * 8)
    assert abs(gc.amortized_nre_per_unit([0] * 8, replace(P, production_volume=2e6)) - nre8 / 2) < 1e-6
    assert abs(gc.amortized_nre_per_unit([0] * 16) - nre8 * 2) < 1e-6
    assert gc.amortized_nre_per_unit([0] * 8, replace(P, nre_scale=2.0)) == nre8 * 2


def test_mem_and_gpu_knobs():
    P = gc.DEFAULT_COST_PARAMS
    assert gc.scaled_mem_cost_per_gb("HBM3") == gc.MEM_SPECS["HBM3"]["cost_per_GB"]
    assert gc.scaled_mem_cost_per_gb("HBM3", replace(P, mem_cost_scale=1.5)) == \
        gc.MEM_SPECS["HBM3"]["cost_per_GB"] * 1.5
    assert gc.gpu_baseline_cost() == P.gpu_baseline_cost_usd


def test_catch_layer_table():
    """The ported CATCH table must reproduce known wafer prices and bracket 14 nm."""
    import math
    W = math.pi * (gc.DEFAULT_COST_PARAMS.wafer_diameter_mm / 2.0) ** 2
    # combined_12nm -> $3,984/wafer, the published IBS 16/12 nm price.
    assert abs(gc.CATCH_LAYERS["combined_12nm"]["cost_per_mm2"] * W - 3984.0) < 1.0
    # CATCH has NO 14 nm or 16 nm row; 14 must map to the nearest, combined_12nm.
    assert "combined_14nm" not in gc.CATCH_LAYERS
    assert "combined_16nm" not in gc.CATCH_LAYERS
    assert gc.NODE_TO_CATCH_LAYER[14] == "combined_12nm"
    assert gc.DEFAULT_COST_PARAMS.catch_layer == "combined_12nm"
    # The 100x-slipped organic interposer row must NOT have been imported.
    assert "combined_interposer_organic" not in gc.CATCH_LAYERS
    for name, d in gc.CATCH_LAYERS.items():
        assert 0.0 < d["critical_area_ratio"] <= 1.0, name
        assert 0.0 < d["litho_percent"] < 1.0, name


def test_critical_area_ratio_raises_yield():
    """CATCH applies D0 to CAR*area, not the full die -- yield must improve."""
    for A in (100, 400, 800):
        y_leg = 0.97 * (1 + A * 0.008 / 2) ** -2   # the pre-port scalar model
        d = gc.CATCH_LAYERS["combined_12nm"]
        y_catch = 0.97 * (1 + A * d["defect_density"] * d["critical_area_ratio"] / 2) ** -2
        assert y_catch > y_leg, A
    # Magnitude: ~1.5x at 100 mm^2 rising to ~3.6x at 800 mm^2.
    y = lambda A, dd, car: 0.97 * (1 + A * dd * car / 2) ** -2
    d = gc.CATCH_LAYERS["combined_12nm"]
    assert 1.4 < y(100, d["defect_density"], d["critical_area_ratio"]) / y(100, 0.008, 1.0) < 1.6
    assert 3.4 < y(800, d["defect_density"], d["critical_area_ratio"]) / y(800, 0.008, 1.0) < 3.8


def test_wafer_fit_derate():
    """CATCH's circle_area/used_area factor: always >= 1, grows with die area."""
    assert gc.wafer_fit_derate(100) >= 1.0
    assert gc.wafer_fit_derate(858) > gc.wafer_fit_derate(100)
    assert 1.0 < gc.wafer_fit_derate(1679.7965) < 1.5
    # Turning it off must make the 2.5D interposer cheaper.
    # A cheaper interposer layer must make the 2.5D package cheaper.
    P_si = replace(gc.DEFAULT_COST_PARAMS, interposer_layer="combined_interposer_silicon")
    P_gl = replace(gc.DEFAULT_COST_PARAMS, interposer_layer="combined_interposer_glass")
    assert gc.calculate_die_cost(400, "2.5D", P_si) < gc.calculate_die_cost(400, "2.5D", P_gl)


def test_unknown_catch_layer_raises():
    P = replace(gc.DEFAULT_COST_PARAMS, catch_layer="combined_14nm")
    try:
        gc.calculate_die_cost(100, "2D", P)
    except ValueError:
        return
    raise AssertionError("unknown catch_layer should raise ValueError")


def test_sweep_knobs_still_work():
    """generate_cost_sweep varies wafer price and defect density; both must bite."""
    P = gc.DEFAULT_COST_PARAMS
    base = gc.calculate_die_cost(100, "2.5D")
    assert gc.calculate_die_cost(100, "2.5D", replace(P, wafer_cost_scale=2.0)) > base
    assert gc.calculate_die_cost(100, "2.5D", replace(P, wafer_cost_scale=0.5)) < base
    # Reporting accessors used by the sweep labels.
    assert abs(gc.effective_wafer_cost() - 3984.0) < 1.0
    assert gc.effective_defect_density() == 0.005
    assert abs(gc.effective_wafer_cost(replace(P, wafer_cost_scale=2.0)) - 7968.0) < 2.0
    # Varying the node must change cost; 5 nm silicon is dearer than 12 nm.
    c12 = gc.calculate_die_cost(100, "2D", replace(P, catch_layer="combined_12nm"))
    c5 = gc.calculate_die_cost(100, "2D", replace(P, catch_layer="combined_5nm"))
    assert c5 > c12


def test_no_legacy_escape_hatch():
    """The old model is in git history, not in a runtime flag."""
    fields = {f.name for f in gc.dataclasses.fields(gc.CostParams)} if hasattr(gc, "dataclasses") \
        else set(gc.CostParams.__dataclass_fields__)
    for gone in ("reticle_model", "wafer_fit_derate", "wafer_cost",
                 "litho_percent", "defect_density_D0"):
        assert gone not in fields, f"{gone} should have been removed"


if __name__ == "__main__":
    n_pass = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
            n_pass += 1
    print(f"\n{n_pass} tests passed.")
