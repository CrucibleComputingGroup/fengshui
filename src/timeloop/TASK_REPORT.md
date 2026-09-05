## Task: Convert MAC units from INT8 to BF16
**Date:** 2026-03-13

### Research Findings
- **No existing GitHub repo** has BF16 Timeloop/Accelergy configurations
- Searched: Accelergy-Project org (accelergy-library-plug-in, timeloop-accelergy-exercises, timeloop-accelergy-tutorial) — all only support integer MACs
- The Aladdin library plugin only has integer multiplier/adder primitives (32-bit at 40nm/15nm)

### Energy & Area Numbers Source

**Primary source: Jouppi et al., "Ten Lessons From Three Generations Shaped Google's TPUv4i", ISCA 2021**
Table 2 extends Horowitz's energy table to 7nm and adds BF16:

| Operation | INT8 (45nm) | INT32 (45nm) | FP16 (45nm) | FP32 (45nm) | **BF16 (7nm)** | INT8 (7nm) | INT32 (7nm) | FP16 (7nm) | FP32 (7nm) |
|-----------|-------------|-------------|-------------|-------------|---------------|------------|-------------|------------|------------|
| Add (pJ)  | 0.03        | 0.1         | 0.4         | 0.9         | **0.110**     | 0.007      | 0.030       | 0.160      | 0.380      |
| Mult (pJ) | 0.2         | 3.1         | 1.1         | 3.7         | **0.210**     | 0.070      | 1.480       | 0.340      | 1.310      |

**BF16/INT32 energy ratios (from 7nm column):**
- BF16 multiply / INT32 multiply = 0.210 / 1.480 = **0.142**
- BF16 add / INT32 add = 0.110 / 0.030 = **3.667**

**Area estimation method:**
- Dally (NVIDIA), AHA Retreat 2023 keynote: **"Area is proportional to energy"**
- Therefore the same BF16/INT32 ratios (0.142 for multiply, 3.667 for add) are applied to both energy and area
- Applied to existing Aladdin INT32 values in the repo at 15nm and 40nm

### Files Created
- `timeloop/bf16_multiplier.csv` — BF16 FP multiplier primitive energy/area at 15nm and 40nm
- `timeloop/bf16_adder.csv` — BF16 FP adder primitive energy/area at 15nm and 40nm
- `arch/_components/bf16mac.yaml` — BF16 MAC compound component (multiplier + adder)
- `arch/eyeriss_like/arch_bf.yaml` — BF16 variant of eyeriss architecture
- `arch/simba_like/arch_bf.yaml` — BF16 variant of simba architecture
- `arch/simple_output_stationary/arch_bf.yaml` — BF16 variant of output-stationary architecture
- `arch/simple_vector/arch_bf.yaml` — BF16 variant of vector architecture

### Files Modified
- `arch/top.yaml.jinja2` — Added `use_bf_arch` flag to select `arch_bf.yaml`
- `scripts/timeloop_helper.py` — Added `use_bf_arch` parameter to `run_mapper()`

### BF16 Component Energy/Area Summary (Jouppi 2021 + Dally area∝energy)

| Component | 15nm Energy | 15nm Area | 40nm Energy | 40nm Area |
|-----------|------------|-----------|-------------|-----------|
| bf16_multiplier (multiply) | 0.118 pJ | 78.7 µm² | 1.80 pJ | 901 µm² |
| bf16_adder (add) | 0.106 pJ | 136 µm² | 0.770 pJ | 1020 µm² |
| **BF16 MAC total** | **0.224 pJ** | **215 µm²** | **2.57 pJ** | **1921 µm²** |

### Derivation Chain (fully traceable)
```
Jouppi ISCA 2021 Table 2 (7nm, Google synthesis)
  → BF16 multiply: 0.210 pJ, BF16 add: 0.110 pJ
  → INT32 multiply: 1.480 pJ, INT32 add: 0.030 pJ
  → BF16/INT32 ratios: multiply=0.142, add=3.667

Dally AHA Retreat 2023 keynote
  → "Area is proportional to energy" (same ratios for area)

Aladdin INT32 in repo (15nm and 40nm)
  → Apply BF16/INT32 ratios to get BF16 energy AND area
```

### Important Considerations
1. **Buffer capacity**: With datawidth 16 (vs 8), buffers store same number of entries but each entry is 2× larger. The `width` (bus width) may need adjustment for throughput.
2. **Accumulator precision**: In real BF16 implementations (e.g., TPU), partial sums are accumulated in FP32. This is not modeled here — a more accurate model would use `adder_width: 32`.
3. **Accelergy plugin discovery**: The BF16 CSV files need to be discoverable by Accelergy's table-based library plugin.
4. **Original files preserved**: All `arch.yaml` and `arch_temp.yaml` files remain at INT8. BF16 configs are in separate `arch_bf.yaml` files. Pass `use_bf_arch=True` to `run_mapper()` to select them.

### References
- Jouppi, N. et al. "Ten Lessons From Three Generations Shaped Google's TPUv4i." **ISCA 2021.** Table 2: BF16 energy at 7nm (0.210 pJ multiply, 0.110 pJ add). Source of BF16/INT32 ratios.
- Dally, W. "Energy Efficiency and AI Hardware." **Stanford AHA Retreat keynote, Aug 2023.** "Area is proportional to energy." Justification for using energy ratios as area ratios.
- Horowitz, M. "Computing's Energy Problem (and what we can do about it)." **ISSCC 2014.** Original 45nm energy table (INT8, INT32, FP16, FP32).
- Aladdin energy estimation tables (existing project baseline at 15nm/40nm).
