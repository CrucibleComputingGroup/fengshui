#!/usr/bin/env python3
"""
Compute chiplet core area as a function of hardware config (arch, pe_scale, glb_scale).

Area is purely a hardware property — it doesn't depend on the workload, operator,
batch size, or DRAM type. So we compute it once per (arch, pe_scale, glb_scale).

Component areas (um²) are from Timeloop's Accelergy ERT at 14nm technology:
  - MAC (bf16mac):     334.43 um² per instance
  - Per-PE buffers:    varies by arch (from arch_bf.yaml component specs)
  - shared_glb:        base 3,881,150 um² (depth=1048576, width=64, 32 banks)
                        scales linearly with glb_scale (depth × glb_scale)

Usage:
  from compute_area import get_chiplet_area_mm2
  area = get_chiplet_area_mm2("eyeriss_like", pe_scale=2, glb_scale=4)
"""
import sys, os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in [os.path.join(_THIS_DIR, "..", "scripts"), os.path.join(_THIS_DIR, "..")]:
    if os.path.isfile(os.path.join(_candidate, "utility_functions.py")) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from global_parameter import ARCH_BASE_PE, layer_factors

# Per-instance areas in um² (from Accelergy ERT, 14nm BF16)
# Extracted from Timeloop stats for each architecture at pe_scale=1, glb_scale=1

MAC_AREA_UM2 = 334.43  # bf16mac per instance (1371824.12 / 4096)

# Per-PE buffer areas (um² per instance)
BUFFER_AREAS = {
    "eyeriss_like": {
        # Instances scale with PE count (pe_x * pe_y)
        "per_pe": {
            "psum_spad":    949.93,   # 24-entry RF
            "weights_spad": 242.01,   # 224-entry SRAM
            "ifmap_spad":   477.32,   # 12-entry RF
        },
        # Instances = 1 (shared)
        "shared": {}
    },
    "gemmini_like": {
        "per_pe": {
            "pe_spad":              10061.40,  # 256-entry RF
            "output_activation_reg": 40.82,    # 1-entry RF
            "input_activation_reg":  40.82,    # 1-entry RF
            "weight_reg":            40.82,    # 1-entry RF
        },
        "shared": {}
    },
    "simba_like": {
        # PEWeightRegs: 1 per MAC (pe_x * pe_y * reg_mac)
        # PEAccuBuffer/PEWeightBuffer: 1 per distributed_buffers group (pe_x * pe_y)
        # PEInputBuffer: 1 per PE (pe_x)
        "per_mac": {
            "PEWeightRegs": 1855.16,  # 1-entry, cluster_size=64
        },
        "per_dist_buf": {
            "PEAccuBuffer":   154.53,   # 128-entry SRAM
            "PEWeightBuffer": 17006.40, # 4096-entry SRAM
        },
        "per_pe": {
            "PEInputBuffer": 37312.20,  # 8192-entry SRAM
        },
        "shared": {}
    },
}

# shared_glb area at each glb_scale (um²)
# From Accelergy ERT (smartbuffer_SRAM, width=64, n_banks=32×scale, depth=1048576×scale)
# Area scales sub-linearly due to SRAM bank overhead sharing
GLB_AREA_UM2 = {
    1:  3_881_150.0,
    4:  14_309_800.0,
    9:  31_776_500.0,
    16: 56_293_700.0,
}


def get_chiplet_area_mm2(arch: str, pe_scale: int = 1, glb_scale: int = 1,
                         pe_x_scale: int = None, pe_y_scale: int = None) -> dict:
    """Compute chiplet core area breakdown in mm².

    Args:
        arch: "eyeriss_like", "gemmini_like", or "simba_like"
        pe_scale: PE array scale factor (applied to both x and y, used if pe_x/y_scale not given)
        glb_scale: Global buffer depth multiplier
        pe_x_scale: PE X scale factor (overrides pe_scale for X)
        pe_y_scale: PE Y scale factor (overrides pe_scale for Y)

    Returns:
        dict with area breakdown in mm²:
          mac_area, buffer_area, glb_area, total_area
    """
    bx, by = ARCH_BASE_PE.get(arch, (64, 64))
    px = pe_x_scale if pe_x_scale is not None else pe_scale
    py = pe_y_scale if pe_y_scale is not None else pe_scale
    pe_x = bx * px
    pe_y = by * py
    reg_mac_spatial = 4 if arch == "simba_like" else 1
    n_macs = pe_x * pe_y * reg_mac_spatial

    # MAC area
    mac_area = MAC_AREA_UM2 * n_macs

    # Buffer area
    buffer_area = 0.0
    buf_spec = BUFFER_AREAS.get(arch, {})

    if arch in ("eyeriss_like", "gemmini_like"):
        # All per-PE buffers scale with pe_x * pe_y
        for name, area_per in buf_spec.get("per_pe", {}).items():
            buffer_area += area_per * n_macs
    elif arch == "simba_like":
        # simba has different instance counts per buffer level
        # n_macs already includes reg_mac (pe_x * pe_y * 4)
        n_pe = pe_x                          # PEInputBuffer: 1 per PE column
        n_dist = pe_x * pe_y                 # PEAccuBuffer, PEWeightBuffer: 1 per dist_buf group

        for name, area_per in buf_spec.get("per_mac", {}).items():
            buffer_area += area_per * n_macs  # PEWeightRegs: 1 per MAC (= n_macs)
        for name, area_per in buf_spec.get("per_dist_buf", {}).items():
            buffer_area += area_per * n_dist
        for name, area_per in buf_spec.get("per_pe", {}).items():
            buffer_area += area_per * n_pe

    # GLB area from lookup table (or interpolate for non-standard scales)
    if glb_scale in GLB_AREA_UM2:
        glb_area = GLB_AREA_UM2[glb_scale]
    else:
        # Linear interpolation between known points
        keys = sorted(GLB_AREA_UM2.keys())
        if glb_scale <= keys[0]:
            glb_area = GLB_AREA_UM2[keys[0]] * glb_scale / keys[0]
        elif glb_scale >= keys[-1]:
            glb_area = GLB_AREA_UM2[keys[-1]] * glb_scale / keys[-1]
        else:
            for i in range(len(keys) - 1):
                if keys[i] <= glb_scale <= keys[i + 1]:
                    frac = (glb_scale - keys[i]) / (keys[i + 1] - keys[i])
                    glb_area = GLB_AREA_UM2[keys[i]] + frac * (GLB_AREA_UM2[keys[i + 1]] - GLB_AREA_UM2[keys[i]])
                    break

    # Vector unit for softmax (1D, scales with pe_x only)
    # Each chiplet has a vector unit attached alongside the 2D array,
    # sharing the same GLB. It idles when the 2D array runs and vice versa.
    # Components: softmax sub-units (max, sn, a) + bf16 ALUs
    vec_area = 0.0
    for unit_spec in layer_factors.values():
        vec_area += pe_x * unit_spec["area"]       # um²
    vec_area += 16 * 39.41 / 2 * pe_x             # bf16 ALU array

    total = mac_area + buffer_area + glb_area + vec_area

    return {
        "mac_area_mm2": mac_area / 1e6,
        "buffer_area_mm2": buffer_area / 1e6,
        "glb_area_mm2": glb_area / 1e6,
        "vector_area_mm2": vec_area / 1e6,
        "total_area_mm2": total / 1e6,
    }


if __name__ == "__main__":
    print("%-16s %4s %4s  %8s %8s %8s %10s" % (
        "Arch", "PE", "GLB", "MAC", "Buffer", "GLB", "Total"))
    print("-" * 68)

    for arch in ["eyeriss_like", "gemmini_like", "simba_like"]:
        for pe in [1, 2, 4]:
            for glb in [1, 4]:
                a = get_chiplet_area_mm2(arch, pe, glb)
                print("%-16s %4d %4d  %7.2f %7.2f %7.2f %9.2f mm²" % (
                    arch, pe, glb,
                    a["mac_area_mm2"], a["buffer_area_mm2"],
                    a["glb_area_mm2"], a["total_area_mm2"]))
