#!/usr/bin/env python3
"""Analyze FlashAttn DRAM sweep results."""
import pandas as pd
import sys

csv_path = sys.argv[1] if len(sys.argv) > 1 else "flashattn_small_test.csv"
df = pd.read_csv(csv_path)

print("=" * 80)
print("  FlashAttn DRAM Sweep — Analysis Report")
print("=" * 80)

# ── 1. Overview ──
print(f"\nDataset: {len(df)} rows")
print(f"Workloads: {df['workload'].unique().tolist()}")
print(f"Architectures: {df['arch'].unique().tolist()}")
print(f"DRAM types: {df['dram_i'].unique().tolist()}")

# ── 2. Per-workload, per-arch summary ──
for wl in df['workload'].unique():
    phase = "prefill" if "prefill" in wl else "decode"
    print(f"\n{'='*80}")
    print(f"  Workload: {wl}  ({phase})")
    print(f"{'='*80}")

    wdf = df[df['workload'] == wl]

    for arch in wdf['arch'].unique():
        adf = wdf[wdf['arch'] == arch]
        print(f"\n  Architecture: {arch}")
        print(f"  {'─'*60}")

        # Baseline info
        row0 = adf.iloc[0]
        print(f"  Compute latency:     {row0['comp_latency_s']*1e6:>10.2f} µs")
        print(f"  Inf-BW latency:      {row0['inf_bw_latency_s']*1e6:>10.2f} µs")
        print(f"  Inf-BW energy:       {row0['inf_bw_energy_pj']/1e6:>10.2f} mJ")
        print(f"  DRAM_I accesses:     {row0['dram_I_accesses']:>10.0f} words")
        print(f"  DRAM_O accesses:     {row0['dram_O_accesses']:>10.0f} words")
        print(f"  DRAM_I bytes:        {row0['dram_I_accesses']*2/1024/1024:>10.2f} MB")
        print(f"  DRAM_O bytes:        {row0['dram_O_accesses']*2/1024/1024:>10.2f} MB")

        # Latency & energy table
        print(f"\n  {'dram_i':>8s} {'dram_o':>8s} {'lat(µs)':>10s} {'speedup':>8s} "
              f"{'energy(mJ)':>11s} {'dram_E(mJ)':>11s} {'throttle':>8s}")
        print(f"  {'─'*8} {'─'*8} {'─'*10} {'─'*8} {'─'*11} {'─'*11} {'─'*8}")

        baseline_lat = adf[(adf['dram_i']=='LPDDR5') & (adf['dram_o']=='LPDDR5')]['latency_s'].values[0]

        for _, r in adf.iterrows():
            lat = r['latency_s'] * 1e6
            spd = baseline_lat / r['latency_s']
            eng = r['dynamic_energy_pj'] / 1e6
            dram_eng = r['dram_energy_pj'] / 1e6
            thr = "YES" if r['bw_throttled'] else ""
            print(f"  {r['dram_i']:>8s} {r['dram_o']:>8s} {lat:>10.2f} {spd:>8.3f}x "
                  f"{eng:>11.2f} {dram_eng:>11.2f} {thr:>8s}")

    # ── 3. Cross-arch comparison for same DRAM ──
    print(f"\n  Cross-architecture comparison (same DRAM combo):")
    print(f"  {'dram_i':>8s} {'dram_o':>8s}", end="")
    archs = wdf['arch'].unique()
    for a in archs:
        print(f" {a:>20s}", end="")
    print(f" {'ratio':>8s}")

    for di in df['dram_i'].unique():
        for do in df['dram_o'].unique():
            lats = []
            print(f"  {di:>8s} {do:>8s}", end="")
            for a in archs:
                row = wdf[(wdf['arch']==a) & (wdf['dram_i']==di) & (wdf['dram_o']==do)]
                if len(row):
                    lat = row.iloc[0]['latency_s'] * 1e6
                    lats.append(lat)
                    print(f" {lat:>18.2f}µs", end="")
                else:
                    lats.append(None)
                    print(f" {'N/A':>20s}", end="")
            if len(lats) == 2 and all(l is not None for l in lats):
                print(f" {lats[0]/lats[1]:>8.3f}", end="")
            print()

# ── 4. Memory-boundedness analysis ──
print(f"\n{'='*80}")
print("  Memory-Boundedness Analysis")
print(f"{'='*80}")

for wl in df['workload'].unique():
    wdf = df[df['workload'] == wl]
    for arch in wdf['arch'].unique():
        adf = wdf[wdf['arch'] == arch]
        throttled = adf[adf['bw_throttled'] == 1]
        not_throttled = adf[adf['bw_throttled'] == 0]
        print(f"\n  {wl} / {arch}:")
        print(f"    BW-throttled configs: {len(throttled)}/{len(adf)}")
        if len(throttled) > 0:
            print(f"    Throttled DRAMs (input): {throttled['dram_i'].unique().tolist()}")
            max_slowdown = (throttled['latency_s'] / throttled['comp_latency_s']).max()
            print(f"    Max slowdown: {max_slowdown:.3f}x")

# ── 5. Energy breakdown ──
print(f"\n{'='*80}")
print("  Energy Breakdown (compute vs DRAM)")
print(f"{'='*80}")

for wl in df['workload'].unique():
    wdf = df[df['workload'] == wl]
    for arch in wdf['arch'].unique():
        adf = wdf[wdf['arch'] == arch]
        print(f"\n  {wl} / {arch}:")
        print(f"  {'dram_i':>8s} {'dram_o':>8s} {'total(mJ)':>10s} {'DRAM(mJ)':>10s} {'DRAM%':>7s}")
        for _, r in adf.iterrows():
            total = r['dynamic_energy_pj'] / 1e6
            dram = r['dram_energy_pj'] / 1e6
            pct = dram / total * 100 if total > 0 else 0
            print(f"  {r['dram_i']:>8s} {r['dram_o']:>8s} {total:>10.2f} {dram:>10.2f} {pct:>6.1f}%")

print(f"\n{'='*80}")
print("  Key Takeaways")
print(f"{'='*80}")

# Compute vs memory bound
for wl in df['workload'].unique():
    wdf = df[df['workload'] == wl]
    phase = "prefill" if "prefill" in wl else "decode"
    any_throttled = wdf['bw_throttled'].any()
    print(f"\n  {phase.upper()}:")
    if any_throttled:
        print(f"    - Memory-bound with slow DRAM (LPDDR5/DDR5)")
        fast = wdf[(wdf['dram_i'].isin(['GDDR7','HBM3'])) & (wdf['dram_o'].isin(['GDDR7','HBM3']))]
        slow = wdf[(wdf['dram_i']=='LPDDR5') & (wdf['dram_o']=='LPDDR5')]
        if len(fast) and len(slow):
            speedup = slow['latency_s'].values[0] / fast['latency_s'].min()
            print(f"    - Max speedup from fastest DRAM: {speedup:.2f}x")
    else:
        print(f"    - Compute-bound across all DRAM types")
        print(f"    - DRAM choice affects energy only, not latency")
