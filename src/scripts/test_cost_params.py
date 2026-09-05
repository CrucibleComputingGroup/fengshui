"""C4 Phase 1 regression test for the parametrized cost model.

Verifies that get_cost.CostParams defaults reproduce the original hard-coded
behavior EXACTLY, and that each sweep knob moves cost in the expected direction.

Run:  python3 test_cost_params.py     (needs no pytimeloop; get_cost is standalone)
"""
from dataclasses import replace

import get_cost as gc

# Golden calculate_die_cost(area, tech) values captured from the pre-parametrization
# code. Defaults MUST reproduce these to 1e-10.
GOLDEN = {
    (0.5, "2D"): 6.5725047270, (0.5, "2.5D"): 6.9093590311,
    (1, "2D"): 6.6326585149,   (1, "2.5D"): 7.0171709202,
    (2, "2D"): 6.7559946731,   (2, "2.5D"): 7.2359805803,
    (5, "2D"): 7.1071035108,   (5, "2.5D"): 7.8747449504,
    (10, "2D"): 7.7185227493,  (10, "2.5D"): 8.9698420867,
    (25, "2D"): 9.5715329123,  (25, "2.5D"): 12.3061487745,
    (50, "2D"): 12.8722375392, (50, "2.5D"): 18.1926800668,
    (100, "2D"): 20.4687269137,(100, "2.5D"): 31.4407296970,
    (200, "2D"): 40.5079175906,(200, "2.5D"): 65.2170780531,
    (400, "2D"): 104.3131606299,(400, "2.5D"): 171.5214757329,
    (800, "2D"): 372.8433361078,(800, "2.5D"): 655.3968140345,
}


def test_default_byte_identical():
    for (area, tech), want in GOLDEN.items():
        got = round(gc.calculate_die_cost(area, tech), 10)
        assert got == want, f"{area} {tech}: want {want}, got {got}"


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


if __name__ == "__main__":
    n_pass = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
            n_pass += 1
    print(f"\n{n_pass} tests passed.")
