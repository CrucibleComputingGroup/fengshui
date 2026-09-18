import argparse
import math
from typing import Dict, Any, List, Tuple

tech_parameters = {
    "65nm": {"k_silicon": 0.03080, "k_exposures": 1122, "defect_density": 0.0004},
    "40nm": {"k_silicon": 0.04531, "k_exposures": 1649, "defect_density": 0.0007},
    "28nm": {"k_silicon": 0.07097, "k_exposures": 2584, "defect_density": 0.001},
    "22nm": {"k_silicon": 0.08527, "k_exposures": 3104.54, "defect_density": 0.0015},
    "16nm": {"k_silicon": 0.10370, "k_exposures": 3774, "defect_density": 0.003},
    "7nm":  {"k_silicon": 0.14311, "k_exposures": 5208.8, "defect_density": 0.006},
}

MEM_SPECS: Dict[str, Dict[str, float]] = {
    'LPDDR5': {
        'shared_bw': 25.6,
        'cost_per_GB': 2.31,
        'area_per_GB': 29.3,
        'ctrl_area': 0.07,
        'phy_area': 7.5,
        'module_caps': 4
    },
    'DDR5': {
        'shared_bw': 70.4,
        'cost_per_GB': 4.38,
        'area_per_GB': 18.0,
        'ctrl_area': 0.07,
        'phy_area': 7.5,
        'module_caps': 8
    },
    'GDDR7': {
        'shared_bw': 128.0,
        'cost_per_GB': 12.0, 
        'area_per_GB': 84.0,
        'ctrl_area': 0.07,
        'phy_area': 8.0,
        'module_caps': 2
    },
    'HBM3': {
        'shared_bw': 819.0,
        'cost_per_GB': 110.0,
        'area_per_GB': 62.5,
        'ctrl_area': 1.00,
        'phy_area': 19.28,
        'module_caps': 8
    }
}

a_reticle = 858
yield_wp = 0.97
alpha = 2

interposer_cost = 0.0219
machine_cost = 0.0209
placement_time = 2
bonding_time = 2

yield_alignment = 0.999

yield_pins = 0.999999

# =============================================================================
# C4: Parametrized cost knobs + NRE / production-volume / GPU-baseline model.
#
# Goal: make every number the sensitivity sweep needs to vary injectable, WITHOUT
# changing default behavior. `CostParams()` defaults reproduce the pre-existing
# hard-coded constants EXACTLY, so calculate_die_cost(area, tech) is byte-identical
# when `params` is omitted. The sweep builds variants with dataclasses.replace(...).
#
# NRE / production-volume / GPU-baseline numbers are LITERATURE DEFAULTS marked as
# ASSUMPTIONS -- confirm with the cost-model owner (the maintainer) before final figures.
# =============================================================================
from dataclasses import dataclass, field, replace as _dc_replace


# Per-distinct-chiplet-design NRE (design + verification + mask set), USD, by node.
# Order-of-magnitude public IBS-style full-design-cost figures; chiplets are simpler
# than full SoCs so these are an UPPER bound. ASSUMPTION -- tune with the maintainer.
NRE_FIXED_BY_NODE_USD = {
    "65nm": 5.0e6,
    "40nm": 12.0e6,
    "28nm": 28.0e6,
    "22nm": 40.0e6,
    "16nm": 50.0e6,
    "7nm": 175.0e6,
}


# ---------------------------------------------------------------------------
# CATCH layer definitions, ported from nanocad-lab/CATCH `layer_definitions.xml`
# (Apache-2.0; arXiv:2503.15753) -- the cost model this file implements.
#
# `cost_per_mm2` is CATCH's wafer price divided by wafer area, so the Fengshui
# equivalent is  wafer_cost = cost_per_mm2 * pi*(wafer_diameter/2)^2.  Sanity
# check: combined_12nm -> $3,984/wafer, exactly the published IBS 16/12 nm
# price, which is what validates the unit convention.
#
# NOTE CATCH has NO 14 nm and NO 16 nm row -- its logic rows are 40/45/12/10/7/
# 5/3 nm, with a gap between 12 and 40.  Fengshui's compute is 14 nm
# (global_parameter.technology_node), so `combined_12nm` is the nearest row and
# is the default below.  Anything else has to be interpolated and said out loud.
#
# `critical_area_ratio` is the fraction of die area that is defect-sensitive.
# Fengshui previously had no such term and applied D0 to the FULL die area,
# which is why its yield was 1.5x (100 mm^2) to 3.5x (800 mm^2) more pessimistic
# than CATCH's own model.
CATCH_LAYERS: Dict[str, Dict[str, float]] = {
    "combined_40nm":               {"cost_per_mm2": 0.033497,   "defect_density": 0.005,    "litho_percent": 0.15, "critical_area_ratio": 0.50},
    "combined_45nm":               {"cost_per_mm2": 0.032171,   "defect_density": 0.005,    "litho_percent": 0.15, "critical_area_ratio": 0.60},
    "combined_12nm":               {"cost_per_mm2": 0.05636207, "defect_density": 0.005,    "litho_percent": 0.25, "critical_area_ratio": 0.60},
    "combined_10nm":               {"cost_per_mm2": 0.084769,   "defect_density": 0.005,    "litho_percent": 0.25, "critical_area_ratio": 0.62},
    "combined_7nm":                {"cost_per_mm2": 0.132219,   "defect_density": 0.005,    "litho_percent": 0.27, "critical_area_ratio": 0.64},
    "combined_5nm":                {"cost_per_mm2": 0.25024,    "defect_density": 0.005,    "litho_percent": 0.30, "critical_area_ratio": 0.67},
    "combined_3nm":                {"cost_per_mm2": 0.29461,    "defect_density": 0.005,    "litho_percent": 0.35, "critical_area_ratio": 0.70},
    "combined_interposer_silicon": {"cost_per_mm2": 0.021905,   "defect_density": 0.00001,  "litho_percent": 0.10, "critical_area_ratio": 0.30},
    "combined_interposer_glass":   {"cost_per_mm2": 0.043910,   "defect_density": 0.00001,  "litho_percent": 0.10, "critical_area_ratio": 0.30},
    # DELIBERATELY OMITTED: combined_interposer_organic.  Its upstream value
    # (5.3820e-7 $/mm^2) is a 100x ft^2->mm^2 exponent slip -- CATCH's own file
    # states the derivation ("$5 per square foot"), and 5/92,903.04 = 5.381955e-5,
    # the same five digits two decades higher.  Even corrected, $5/ft^2 is raw
    # panel laminate, not a build-up ABF substrate.  Do not import it.
}

# Map Fengshui's integer technology_node (nm) onto the nearest CATCH row.
NODE_TO_CATCH_LAYER: Dict[int, str] = {
    40: "combined_40nm", 45: "combined_45nm",
    14: "combined_12nm", 12: "combined_12nm", 16: "combined_12nm",
    10: "combined_10nm", 7: "combined_7nm", 5: "combined_5nm", 3: "combined_3nm",
}


@dataclass
class CostParams:
    """All knobs the C4 sweep can vary. Defaults == current hard-coded values."""
    # --- die / wafer ---
    wafer_diameter_mm: float = 300.0
    edge_exclusion_mm: float = 3.0
    reticle_area_mm2: float = 858.0       # 26 x 33 mm exposure field
    die_yield_para: float = 0.97          # CATCH's wafer_process_yield
    alpha: float = 2.0                    # CATCH's clustering_factor

    # --- CATCH layer selection -------------------------------------------
    # Supplies cost_per_mm2, defect_density, litho_percent, critical_area_ratio.
    # CATCH has no 14 nm or 16 nm row (logic rows: 40/45/12/10/7/5/3 nm), so
    # combined_12nm is the nearest to Fengshui's technology_node = 14.
    catch_layer: str = "combined_12nm"
    interposer_layer: str = "combined_interposer_silicon"

    # --- multiplicative sweep knobs (1.0 == baseline) ---
    # Folded in where the underlying absolute value is consumed so a sweep can scale
    # one physical quantity without rewriting the absolute default.
    wafer_cost_scale: float = 1.0         # scales the layer's wafer price (wafer-cost sweep)
    yield_defect_scale: float = 1.0       # scales the layer's defect density (yield sweep)
    interposer_cost_scale: float = 1.0    # scales interposer $/mm^2 (packaging sweep)
    assembly_material_scale: float = 1.0  # scales assembly materials $/mm^2 (packaging sweep)
    mem_cost_scale: float = 1.0           # scales memory $/GB (memory-cost sweep)



    # --- NRE / production volume / GPU baseline (NEW for C4 Option A) ---
    process_node: str = "7nm"             # node used for NRE lookup
    nre_fixed_usd: float = None           # if None, taken from NRE_FIXED_BY_NODE_USD[node]
    nre_area_coeff_usd_per_mm2: float = 0.0  # optional area-scaling of NRE (default off)
    nre_scale: float = 1.0                # multiplicative knob for the NRE sweep
    production_volume: float = 1.0e6      # units over which NRE is amortized (volume sweep)
    gpu_baseline_cost_usd: float = 1500.0 # reference monolithic-GPU unit cost (GPU-cost sweep)

    def base_nre_per_type(self) -> float:
        """Fixed NRE for one distinct chiplet design at this node (before area term)."""
        if self.nre_fixed_usd is not None:
            return self.nre_fixed_usd
        return NRE_FIXED_BY_NODE_USD.get(self.process_node, NRE_FIXED_BY_NODE_USD["7nm"])


DEFAULT_COST_PARAMS = CostParams()


def nre_per_chiplet_type(area_mm2: float = 0.0, params: "CostParams" = None) -> float:
    """Amortizable NRE for ONE distinct chiplet design (USD, before /volume)."""
    p = params or DEFAULT_COST_PARAMS
    return p.nre_scale * (p.base_nre_per_type() + p.nre_area_coeff_usd_per_mm2 * area_mm2)


def amortized_nre_per_unit(chiplet_areas: List[float], params: "CostParams" = None) -> float:
    """Per-shipped-unit NRE for a pool of distinct designs = sum(NRE_type)/volume.

    `chiplet_areas` is one entry per DISTINCT design in the pool (length == pool size N).
    Pass zeros (or a list of length N) if per-design areas are unknown -- the fixed
    per-node NRE term still applies.
    """
    p = params or DEFAULT_COST_PARAMS
    total = sum(nre_per_chiplet_type(a, p) for a in chiplet_areas)
    return total / max(p.production_volume, 1.0)


def scaled_mem_cost_per_gb(mem_type: str, params: "CostParams" = None) -> float:
    """Memory $/GB scaled by the memory-cost sweep knob. Wiring point for the
    memory-cost sweep: the SA pipeline reads cost via get_memory_spec(); multiply
    its cost_per_GB by params.mem_cost_scale to apply this sweep end-to-end."""
    p = params or DEFAULT_COST_PARAMS
    return MEM_SPECS[mem_type]["cost_per_GB"] * p.mem_cost_scale


def gpu_baseline_cost(params: "CostParams" = None) -> float:
    """Reference monolithic-GPU unit cost used to normalize the chiplet system."""
    p = params or DEFAULT_COST_PARAMS
    return p.gpu_baseline_cost_usd
# def calculate_die_cost(
#     chiplet_area: float,
#     k_silicon: float = tech_parameters["16nm"]["k_silicon"],
#     k_exposures: float = tech_parameters["16nm"]["k_exposures"]* 858 / 57962.38,
#     a_reticle: float = a_reticle,
#     defect_density: float = tech_parameters["16nm"]["defect_density"],
#     yield_wp: float = yield_wp,
#     alpha: float = alpha,
#     die_type: str = 'sram'
# ):
#     if chiplet_area == float("inf"):
#         return float("inf")
#     # use 0.5 to simulate sram/compute mixed logic
#     defect_density_final = defect_density * 0.1 if die_type == "sram" else defect_density * 0.5
    
#     k_die_silicon = k_silicon * chiplet_area
#     k_die_exposure = k_exposures / math.ceil(a_reticle / chiplet_area)
#     k_die = k_die_silicon + k_die_exposure

#     yield_die = yield_wp * (1 + (chiplet_area * defect_density_final) / alpha) ** (-alpha)
#     die_cost = k_die / yield_die
    
#     return die_cost

# def calculate_assembly_cost(
#     chiplets: list[str],
#     interposer_cost: float = interposer_cost,
#     machine_cost: float = machine_cost,
#     placement_time: float = placement_time,
#     bonding_time: float = bonding_time,
#     yield_alignment: float = yield_alignment,
#     yield_pins: float = yield_pins
# ):
    
#     n_chiplets = len(chiplets)
#     total_area = 0
#     for chiplet in chiplets:
#         area, chiplet_type = chiplet.split(",")
#         area = float(area)
#         total_area += area
#     n_pins = int(200 * math.sqrt(total_area))
#     assembly_time = n_chiplets * placement_time + bonding_time
#     raw_assembly_cost = interposer_cost * total_area + machine_cost * assembly_time
#     yield_assembly = (yield_alignment ** n_chiplets) * (yield_pins ** n_pins)

#     return raw_assembly_cost, yield_assembly

# get_cost.py
from typing import Literal, Dict, Any

# Public types
StackTech = Literal["2D", "2.5D"]

SECONDS_PER_YEAR = 365 * 24 * 60 * 60

# Assembly parameters keyed directly by stack technology
# (parameter names slightly shortened/renamed)
ASSEMBLY_PARAMS: Dict[StackTech, Dict[str, float]] = {
    "2D": {
        "material_cost_mm2": 0.1,
        "pnp_machine_cost": 600_000.0,
        "pnp_machine_life_yrs": 5.0,
        "pnp_uptime": 0.9,
        "pnp_tech_cost_per_year": 100_000.0,
        "pnp_time_sec": 10.0,
        "pnp_group": 1,
        "bond_machine_cost": 200_000.0,
        "bond_machine_life_yrs": 5.0,
        "bond_uptime": 0.9,
        "bond_tech_cost_per_year": 100_000.0,
        "bond_time_sec": 20.0,
        "bond_group": 64,
    },
    "2.5D": {
        "material_cost_mm2": 0.1,
        "pnp_machine_cost": 500_000.0,
        "pnp_machine_life_yrs": 5.0,
        "pnp_uptime": 0.9,
        "pnp_tech_cost_per_year": 100_000.0,
        "pnp_time_sec": 10.0,
        "pnp_group": 1,
        "bond_machine_cost": 500_000.0,
        "bond_machine_life_yrs": 5.0,
        "bond_uptime": 0.9,
        "bond_tech_cost_per_year": 100_000.0,
        "bond_time_sec": 20.0,
        "bond_group": 1,
    },
}

# UCIe IO parameter table (exposed for convenience)
UCIE_PARAMS: Dict[str, Dict[str, Any]] = {
    "UCIe_standard": {
        "tx_area": 0.75438,
        "rx_area": 0.75438,
        "shoreline": 0.5715,
        "bandwidth_Gbps": 1024,
        "wire_count": 44,
        "bidirectional": True,
        "energy_per_bit_J": 1e-12,
        "reach_mm": 10.0,
    },
    "UCIe_advanced": {
        "tx_area": 0.4055184,
        "rx_area": 0.4055184,
        "shoreline": 0.3888,
        "bandwidth_Gbps": 4096,
        "wire_count": 140,
        "bidirectional": True,
        "energy_per_bit_J": 1e-12,
        "reach_mm": 10.0,
    },
}

STACK_TO_UCIE = {"2D": "UCIe_standard", "2.5D": "UCIe_advanced"}


def _cost_per_second(
    machine_cost: float,
    life_yrs: float,
    uptime: float,
    tech_cost_per_year: float,
) -> float:
    """Convert (machine depreciation + technician yearly cost) to an effective $/s rate."""
    if life_yrs <= 0 or uptime <= 0:
        raise ValueError("life_yrs and uptime must be positive.")
    annual_cost = (machine_cost / life_yrs) + tech_cost_per_year
    effective_seconds = SECONDS_PER_YEAR * uptime
    return annual_cost / effective_seconds


def get_ucie_type(tech: StackTech) -> str:
    """Map '2D' -> 'UCIe_standard', '2.5D' -> 'UCIe_advanced'."""
    if tech not in STACK_TO_UCIE:
        raise ValueError(f"Unknown stack tech: {tech}. Expected one of {list(STACK_TO_UCIE.keys())}.")
    return STACK_TO_UCIE[tech]


def get_ucie_params(tech: StackTech) -> Dict[str, Any]:
    """Return the UCIe parameter dictionary for a given stack tech ('2D' or '2.5D')."""
    return UCIE_PARAMS[get_ucie_type(tech)]


import math

def squares_in_circle(diameter, square_side):
    if diameter <= 0 or square_side <= 0:
        raise ValueError("diameter and square_side must be positive")
    
    radius = diameter / 2
    count = 0
    
    y = -radius + square_side / 2
    while y <= radius - square_side / 2:
        horizontal_radius = math.sqrt(radius**2 - y**2)
        squares_in_row = int((2 * horizontal_radius) // square_side)
        count += squares_in_row
        y += square_side
    
    return count

def reticle_utilization(die_area_mm2: float, reticle_area_mm2: float) -> float:
    """Fraction of an exposure field occupied by WHOLE dies (a packing efficiency).

    Faithful port of ``Layer.reticle_utilization`` from CATCH
    (github.com/nanocad-lab/CATCH, Apache-2.0; arXiv:2503.15753), the cost model
    this file implements.  A die larger than one field is assumed to be stitched
    across whole additional fields.

    Returns a value in (0.5, 1] -- NOT the raw area ratio ``A / reticle_area``.

    Why this matters: ``base_cost`` is proportional to die area, so dividing the
    litho share by a raw area ratio cancels the area and charges every die the
    same absolute lithography cost (~$5.91 here) whether it is 10 mm^2 or
    858 mm^2 -- an 85x overcharge for a small chiplet, where litho then accounts
    for 98% of die cost.  Shots per wafer are set by the field, not by how the
    wafer is diced, so litho per die is proportional to area; the only
    reticle-related penalty is imperfect tiling of the field, which is what this
    function measures.
    """
    if not (math.isfinite(die_area_mm2) and math.isfinite(reticle_area_mm2)):
        raise ValueError("die_area_mm2 and reticle_area_mm2 must be finite")
    if die_area_mm2 <= 0 or reticle_area_mm2 <= 0:
        raise ValueError("die_area_mm2 and reticle_area_mm2 must be positive")

    # CATCH stitches a too-large die across whole additional fields with
    #     field = R;  while field < A:  field += R
    # We use the closed form instead.  It is exactly equivalent in exact
    # arithmetic (both give the least n >= 1 with n*R >= A) but is O(1) rather
    # than O(A/R), and avoids a real float-accumulation drift in the loop:
    # at A = 25743, R = 858.1 repeated addition selects 31 fields where 30
    # suffice.  Unreachable at the shipped R = 858 mm^2 (every Fengshui die is
    # sub-reticle, so n = 1), but the function is general.
    n_fields = max(1, math.ceil(die_area_mm2 / reticle_area_mm2))
    field_area = n_fields * reticle_area_mm2

    # A vanishingly small die tiles the field essentially perfectly, and the
    # quotient below would overflow for subnormal areas (float division of a
    # normal by ~1e-308 is +inf), so short-circuit instead of crashing.
    ratio = field_area / die_area_mm2
    if not math.isfinite(ratio) or ratio > 1.0e15:
        return 1.0

    n_whole_dies = max(1, int(ratio))         # n_fields*R >= A, so >=1 die fits
    return (n_whole_dies * die_area_mm2) / field_area


def _effective_layer(p: "CostParams") -> Dict[str, float]:
    """Per-node cost/yield parameters in force, with sweep scales applied."""
    try:
        layer = dict(CATCH_LAYERS[p.catch_layer])
    except KeyError:
        raise ValueError(f"unknown catch_layer {p.catch_layer!r}; "
                         f"expected one of {sorted(CATCH_LAYERS)}")
    layer["cost_per_mm2"] *= p.wafer_cost_scale
    layer["defect_density"] *= p.yield_defect_scale
    return layer


def effective_wafer_cost(p: "CostParams" = None) -> float:
    """$ per processed wafer implied by the selected layer (for sweeps/reporting)."""
    p = p or DEFAULT_COST_PARAMS
    return _effective_layer(p)["cost_per_mm2"] * math.pi * (p.wafer_diameter_mm / 2.0) ** 2


def effective_defect_density(p: "CostParams" = None) -> float:
    """Defects/mm^2 implied by the selected layer (for sweeps/reporting)."""
    p = p or DEFAULT_COST_PARAMS
    return _effective_layer(p)["defect_density"]


def wafer_fit_derate(die_area_mm2: float, p: "CostParams" = None) -> float:
    """CATCH's circle_area/used_area factor (design.py:1129).

    Dies do not tile a circular wafer perfectly; the wasted area is paid for.
    Always >= 1.0.  At the shipped 300 mm wafer this is ~1.29x for an 858 mm^2
    interposer and ~1.32x at 1679.8 mm^2.
    """
    p = p or DEFAULT_COST_PARAMS
    effective_diameter = p.wafer_diameter_mm - 2 * p.edge_exclusion_mm
    n = squares_in_circle(effective_diameter, math.sqrt(die_area_mm2))
    if n <= 0:
        return 1.0
    circle_area = math.pi * (p.wafer_diameter_mm / 2.0) ** 2
    return circle_area / (n * die_area_mm2)


def calculate_die_cost(
    die_area_mm2: float,
    bonding_tech: str,
    params: "CostParams" = None
):
    p = params or DEFAULT_COST_PARAMS
    layer = _effective_layer(p)

    wafer_diameter_mm = p.wafer_diameter_mm
    edge_exclusion_mm = p.edge_exclusion_mm
    litho_percent = layer["litho_percent"]
    reticle_area_mm2 = p.reticle_area_mm2
    die_yield_para = p.die_yield_para          # CATCH's wafer_process_yield
    D_0 = layer["defect_density"]              # sweep scale already applied
    CAR = layer["critical_area_ratio"]
    Alpha = p.alpha                            # CATCH's clustering_factor

    # wafer_cost = cost_per_mm2 * full wafer area.
    wafer_cost = layer["cost_per_mm2"] * math.pi * (wafer_diameter_mm / 2.0) ** 2

    effective_diameter = wafer_diameter_mm - 2 * edge_exclusion_mm
    dies_per_wafer = squares_in_circle(effective_diameter, math.sqrt(die_area_mm2))
    base_cost = wafer_cost / dies_per_wafer

    # CATCH: defect_yield = (1 + D0*A*CAR/N)^-N, times wafer_process_yield.
    die_yield = die_yield_para * (1 + (die_area_mm2 * D_0 * CAR)/Alpha)**(-Alpha)
    util = reticle_utilization(die_area_mm2, reticle_area_mm2)
    litho_cost = base_cost * litho_percent / util
    die_cost = (base_cost * (1 - litho_percent) + litho_cost) / die_yield

    assembly = compute_assembly(die_area_mm2, bonding_tech, n_chips=1, params=p)

    final_cost = (die_cost + assembly["C_assembly"]) / assembly["Y_assembly"]

    return final_cost

SECONDS_PER_YEAR = 365 * 24 * 60 * 60

ASSEMBLY_DB = {
    "2D": {
        "materials_cost_per_mm2": 0.05,
        "bb_cost_per_second": 0.0147979,
        "picknplace_machine_cost": 400000,
        "picknplace_machine_lifetime": 5,
        "picknplace_machine_uptime": 0.9,
        "picknplace_technician_yearly_cost": 100000,
        "picknplace_time": 8,
        "picknplace_group": 64,
        "bonding_machine_cost": 200000,
        "bonding_machine_lifetime": 5,
        "bonding_machine_uptime": 0.9,
        "bonding_technician_yearly_cost": 120000,
        "bonding_time": 20,
        "bonding_group": 64,
        "die_separation": 0.10,
        "edge_exclusion": 0.20,
        "max_pad_current_density": 250.0,
        "bonding_pitch": 0.110,
        "alignment_yield": 0.9999,
        "bonding_yield": 0.9999995,
        "dielectric_bond_defect_density": 0.0,
        "tsv_area": 0.0,
        "tsv_yield": 1.0,
        "tsv_pitch": 0.0,
    },
    "2.5D" : {
        "materials_cost_per_mm2": 0.12,
        "bb_cost_per_second": 0.0123316,
        "picknplace_machine_cost": 600000,
        "picknplace_machine_lifetime": 5,
        "picknplace_machine_uptime": 0.9,
        "picknplace_technician_yearly_cost": 120000,
        "picknplace_time": 12,
        "picknplace_group": 64,
        "bonding_machine_cost": 1000000,
        "bonding_machine_lifetime": 5,
        "bonding_machine_uptime": 0.9,
        "bonding_technician_yearly_cost": 150000,
        "bonding_time": 45,
        "bonding_group": 1,  
        "die_separation": 0.10,  
        "edge_exclusion": 0.15,
        "max_pad_current_density": 250.0,
        "bonding_pitch": 0.045,    
        "alignment_yield": 0.9998, 
        "bonding_yield": 0.999999, 
        "dielectric_bond_defect_density": 0.0,
        "tsv_area": 0.0001, 
        "tsv_yield": 0.999998,    
        "tsv_pitch": 0.045,  
    },
}

def _cost_per_second(bb, machine_cost, lifetime, uptime, tech_yearly):
    if bb is not None:
        return float(bb)
    yearly = machine_cost / lifetime + tech_yearly
    return (yearly / SECONDS_PER_YEAR) * uptime

def estimate_bonds(area_mm2, pitch_mm):
    if pitch_mm <= 0:
        return 0
    return int(area_mm2 / (pitch_mm ** 2))

def estimate_tsvs(area_mm2, tsv_pitch_mm, tsv_density_factor=0.02):
    if tsv_pitch_mm <= 0:
        return 0
    base = area_mm2 / (tsv_pitch_mm ** 2)
    return int(base * tsv_density_factor)

def compute_assembly(area_mm2, assembly_name, n_chips=1, params=None):
    p = params or DEFAULT_COST_PARAMS
    # CATCH's combined_interposer_silicon (0.021905), derated for wafer fit the
    # way CATCH does (design.py:1129).  Previously a flat 0.0219 with no derate,
    # which under-charged the interposer by ~30%.
    # Derate uses FENGSHUI's own squares_in_circle, for consistency with how the
    # die path counts dies_per_wafer.  CATCH's figure is larger (1.287x at
    # 858 mm^2 vs 1.113x here) because its compute_dies_per_wafer models dicing
    # lanes and a fill grid squares_in_circle does not; importing CATCH's factor
    # on top of Fengshui's packing would double-count the waste.
    _itp_rate = CATCH_LAYERS[p.interposer_layer]["cost_per_mm2"]
    if area_mm2 > 0:
        _itp_rate *= wafer_fit_derate(area_mm2, p)
    interposer_cost_per_mm2 = _itp_rate * p.interposer_cost_scale
    if assembly_name not in ASSEMBLY_DB:
        raise ValueError(f"unknown assembly '{assembly_name}'")
    P = ASSEMBLY_DB[assembly_name]

    pick_steps = math.ceil(n_chips / P["picknplace_group"])
    bond_steps = math.ceil(n_chips / P["bonding_group"])
    t_pick = P["picknplace_time"] * pick_steps
    t_bond = P["bonding_time"] * bond_steps

    cps_pick = _cost_per_second(P["bb_cost_per_second"],
                                P["picknplace_machine_cost"], P["picknplace_machine_lifetime"],
                                P["picknplace_machine_uptime"], P["picknplace_technician_yearly_cost"])
    cps_bond = _cost_per_second(P["bb_cost_per_second"],
                                P["bonding_machine_cost"], P["bonding_machine_lifetime"],
                                P["bonding_machine_uptime"], P["bonding_technician_yearly_cost"])

    time_cost = cps_pick * t_pick + cps_bond * t_bond
    materials_cost = P["materials_cost_per_mm2"] * p.assembly_material_scale * area_mm2
    if assembly_name == "2.5D":
        interposer_area_mm2 = area_mm2
        interposer_cost = interposer_area_mm2 * interposer_cost_per_mm2
        materials_cost += interposer_cost


    n_bonds = estimate_bonds(area_mm2, P["bonding_pitch"])
    if assembly_name.startswith("2.5D"):
        n_tsvs = estimate_tsvs(area_mm2, P["tsv_pitch"])
    else:
        n_tsvs = 0

    Y_align = (P["alignment_yield"]) ** n_chips
    Y_bond  = (P["bonding_yield"])  ** max(n_bonds, 1)
    Y_tsv   = (P["tsv_yield"])      ** max(n_tsvs, 0)
    Y_dielectric = 1.0 / (1.0 + P["dielectric_bond_defect_density"] * area_mm2)

    Y_assembly = Y_align * Y_bond * Y_tsv * Y_dielectric
    Y_assembly = min(max(Y_assembly, 0.0), 1.0)

    C_assembly = time_cost + materials_cost

    return {
        "C_assembly": C_assembly,
        "Y_assembly": Y_assembly,
        "n_bonds": n_bonds,
        "n_tsvs": n_tsvs,
        "time_sec": t_pick + t_bond,
        "breakdown": {
            "time_cost": time_cost,
            "materials_cost": materials_cost,
            "cps_pick": cps_pick,
            "cps_bond": cps_bond,
            "t_pick": t_pick,
            "t_bond": t_bond,
        },
    }

__all__ = ["calculate_die_cost", "get_ucie_type", "get_ucie_params"]

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Compute per-die assembly cost.")
    parser.add_argument("--area", type=float, required=True, help="Total die area in mm^2 (non-negative).")
    parser.add_argument(
        "--tech",
        type=str,
        choices=["2D", "2.5D"],
        required=True,
        help="Stack/bonding technology.",
    )
    parser.add_argument(
        "--show-ucie",
        action="store_true",
        help="Print the mapped UCIe type and parameters."
    )
    args = parser.parse_args()

    cost = calculate_die_cost(args.area, args.tech)  # type: ignore[arg-type]
    payload = {
        "area_mm2": args.area,
        "tech": args.tech,
        "cost_usd": cost,
    }
    if args.show_ucie:
        ucie_type = get_ucie_type(args.tech)  # type: ignore[arg-type]
        payload["ucie_type"] = ucie_type
        payload["ucie_params"] = get_ucie_params(args.tech)  # type: ignore[arg-type]

    print(json.dumps(payload, indent=2))



def calculate_total_cost(
    chiplets: list[str]
):
    total_die_cost = 0
    yield_die_final = 1
    for chiplet in chiplets:
        area, chiplet_type = chiplet.split(",")
        die_cost = calculate_die_cost(chiplet_area=float(area),die_type=chiplet_type)
        total_die_cost += die_cost

    raw_assembly_cost, yield_assembly = calculate_assembly_cost(chiplets)
    total_cost = (total_die_cost + raw_assembly_cost) / yield_assembly
    return total_cost,yield_die_final

# LPDDR5 and DDR5 share one PHY; GDDR7 and HBM3 each have their own.
# Compute chiplets carry all 3 unique PHY blocks so they can drive any DRAM.
_ALL_PHY_GROUPS = [
    ['LPDDR5'],   # shared with DDR5
    ['GDDR7'],
    ['HBM3'],
]

def area_with_mem_overheads(
    area: float,
    bonding: str,
    dram_type: str = "HBM3",
    mem_specs: Dict[str, Dict[str, float]] = MEM_SPECS,
    all_phy: bool = False,
) -> Dict[str, Any]:
    """Return die area with memory-interface overheads.

    Args:
        all_phy: If True, sum PHY/ctrl areas for all DRAM types (compute
                 chiplets that support every interface).  When False, only
                 the single *dram_type* PHY is added (PIM / legacy path).
    """
    bonding_norm = bonding.strip().upper()
    if bonding_norm not in ("2D", "2.5D"):
        raise ValueError("bonding tech needs to be '2D' or '2.5D'")

    if all_phy:
        total_phy = 0.0
        total_ctrl = 0.0
        for group in _ALL_PHY_GROUPS:
            rep = group[0]
            base_phy = float(mem_specs[rep]["phy_area"])
            total_phy += base_phy if bonding_norm == "2.5D" else base_phy * 2.2
            total_ctrl += float(mem_specs[rep]["ctrl_area"])
        final_area = float(area) + total_ctrl + total_phy
        return {
            "core_area": float(area),
            "ctrl_area": total_ctrl,
            "phy_area": total_phy,
            "final_area": final_area
        }

    if dram_type not in mem_specs:
        raise KeyError(f"Unknown DRAM type: {dram_type}")

    ctrl_area = float(mem_specs[dram_type]["ctrl_area"])
    base_phy = float(mem_specs[dram_type]["phy_area"])
    phy_area = base_phy if bonding_norm == "2.5D" else base_phy * 2.2

    final_area = float(area) + ctrl_area + phy_area
    return {
        "core_area": float(area),
        "ctrl_area": ctrl_area,
        "phy_area": phy_area,
        "final_area": final_area
    }
