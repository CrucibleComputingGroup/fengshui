#!/usr/bin/env python3
"""
Fact-check: is PIM modeled as "memory -> compute -> memory, all three replaced by
the PIM die" consistently across the perf/cost model?

Normal operator  : DRAM read (operands) -> compute chiplet (PE array) -> DRAM write.
PIM operator      : one near-bank in-memory op -> NO external DRAM read of operands,
                    NO separate PE-array compute, NO DRAM write-back; the PIM die IS
                    memory+compute. Energy/latency come straight from the PIM DB, and
                    area/cost are the PIM die(s) (PIM_DIE_AREA_MM2), not a PE array.

This script checks the runtime-verifiable invariant (per-die PIM area must be the PIM
die everywhere) and prints the inspection-verified invariants with line refs.

Run from .../chiplet_timeloop/scripts:  python arch_impl/verify_pim_integration.py
Exit code 0 = consistent; non-zero = inconsistency found (regression guard).
"""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(THIS_DIR, ".."))
EXP = os.path.normpath(os.path.join(SCRIPTS, "..", "timeloop_experiments"))
for p in (SCRIPTS, EXP):
    if p not in sys.path:
        sys.path.insert(0, p)

from compute_area import get_chiplet_area_mm2          # noqa: E402
from global_parameter import PIM_DIE_AREA_MM2, ARCH_BASE_PE  # noqa: E402
import cal_perf_phy_net as cpp                          # noqa: E402
from chiplet_dataclass import ChipletConfig             # noqa: E402

fails = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        fails.append(name)


print("=== PIM integration fact-check ===\n")

# --- Inspection-verified invariants (energy/latency = 'all three replaced') ---
print("Energy/latency model (verified by code inspection of cal_perf_phy_net.py):")
print("  [OK] main path L573-577: PIM dynamic_energy = DB value (no e_DDRtoLPDDR DRAM")
print("       adjustment, no inter-chiplet comm) -> in-memory compute already in DB.")
print("  [OK] main path L566: PIM skips BW-contention; L563 latency = DB value.")
print("  [OK] off-CP path: PIM skips DRAM adjustment + weight transfer; charges activation")
print("       operand transport (in_mem/out_mem inter-chiplet link) only -- no weights.")
print("  [OK] main path L630-650: PIM cost/area = 2x PIM dies, 'no separate DRAM chips'.")
print("  [OK] boundary L1104-1126: PIM->non-PIM input buffer deducted (no double-count).")
print("  [OK] feasibility L543-546: PIM is GDDR7-only (non-GDDR7 -> inf).\n")

# --- Runtime-checkable invariant: per-die PIM core area must be the PIM die ---
# Exercise the REAL call-site decision (_offcp_core_mm2) rather than re-implement it,
# so this guard tracks _build_off_cp_functions. The raw compute-core primitive
# get_chiplet_area_mm2('PIM',...) legitimately returns the small PE core (~3.88 mm^2);
# the invariant is that the off-CP builder prices PIM at the PIM die instead.
print("Cost/area model (runtime check):")
core_primitive = get_chiplet_area_mm2('PIM', pe_x_scale=1, pe_y_scale=1, glb_scale=1)
core_primitive = core_primitive['total_area_mm2'] if isinstance(core_primitive, dict) else core_primitive
pim_cc = ChipletConfig.from_identifier('PIM@glb1@pe_x_scale1@pe_y_scale1')
core_off_cp = cpp._offcp_core_mm2(pim_cc)

check("main-path PIM die area constant",
      abs(PIM_DIE_AREA_MM2 - 224.0) < 1.0,
      f"PIM_DIE_AREA_MM2 = {PIM_DIE_AREA_MM2:.2f} mm^2 (used at cal_perf_phy_net.py:465)")

check("off-CP PIM core area == PIM die area",
      abs(core_off_cp - PIM_DIE_AREA_MM2) / PIM_DIE_AREA_MM2 < 0.05,
      f"_build_off_cp_functions prices PIM via _offcp_core_mm2 = {core_off_cp:.2f} mm^2 "
      f"(PIM die {PIM_DIE_AREA_MM2:.2f} mm^2; raw compute-core primitive is "
      f"{core_primitive:.2f} mm^2, correctly NOT used for PIM cost)")

print()
if fails:
    print(f"VERDICT: {len(fails)} inconsistency(ies) found: {', '.join(fails)}")
    print("Energy/latency model is correct (PIM replaces mem->compute->mem); the gap is")
    print("in OFF-CRITICAL-PATH cost/area only. Fix: in _build_off_cp_functions, branch")
    print("`if is_pim_chiplet: core_mm2 = PIM_DIE_AREA_MM2` (mirror main path L463-465),")
    print("and add GDDR7 die leakage + PHY to off-CP PIM static_power (mirror L640-644).")
    sys.exit(1)
print("VERDICT: PIM integration consistent across all paths.")
sys.exit(0)
