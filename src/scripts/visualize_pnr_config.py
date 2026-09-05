#!/usr/bin/env python3
"""
visualize_pnr_config.py — DAG visualization of PnR configs.

Pure DAG: Chiplet A → DRAM → Chiplet B
- PIM-to-PIM: direct arrow (GDDR7 integrated as compute+buffer)
- Off-CP operators: proper DAG edges showing feeds_from/feeds_to
- Parallel stages: annotate TP degree / number of chiplets
- Edges show data movement volume (read/write) between chiplet and DRAM
"""

import os
import json
import argparse
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

ARCH_COLORS = {
    'eyeriss_like': '#4e79a7',
    'simba_like': '#f28e2b',
    'gemmini_like': '#59a14f',
    'PIM': '#e15759',
    'switch_8port': '#b07aa1',
    'unknown': '#999999',
}
ARCH_SHORT = {
    'eyeriss_like': 'RS (Eyeriss)',
    'simba_like': 'WS (Simba)',
    'gemmini_like': 'OS (Gemmini)',
    'PIM': 'PIM',
    'switch_8port': 'Switch',
}
DRAM_COLOR = '#ffd700'
DRAM_SPECS = {
    'LPDDR5': {'bw': 70.4, 'ctrl_mm2': 0.07, 'phy_mm2': 7.5, 'area_per_GB': 29.3},
    'DDR5':   {'bw': 70.4, 'ctrl_mm2': 0.07, 'phy_mm2': 7.5, 'area_per_GB': 18.0},
    'GDDR7':  {'bw': 320,  'ctrl_mm2': 0.07, 'phy_mm2': 8.0, 'area_per_GB': 84.0},
    'HBM3':   {'bw': 819,  'ctrl_mm2': 1.00, 'phy_mm2': 19.28, 'area_per_GB': 62.5},
    'HBM3E':  {'bw': 1229, 'ctrl_mm2': 1.00, 'phy_mm2': 19.28, 'area_per_GB': 62.5},
}


def shorten_op(name):
    name = name.replace('layer0_', '').replace('stages_', 's')
    name = name.replace('blocks_', 'b').replace('_conv', '')
    name = name.replace('_proj', '').replace('expert_', 'exp_')
    return name

def fmt_area(um2):
    if um2 <= 0: return '?'
    return f"{um2/1e6:.1f} mm\u00b2"

def fmt_power(w):
    if w <= 0: return '?'
    return f"{w*1e3:.0f} mW" if w < 1 else f"{w:.2f} W"

def fmt_lat(s):
    if s <= 0: return '?'
    if s < 1e-6: return f"{s*1e9:.0f} ns"
    if s < 1e-3: return f"{s*1e6:.0f} \u00b5s"
    return f"{s*1e3:.2f} ms"

def fmt_energy(j):
    if j <= 0: return '?'
    if j < 1e-6: return f"{j*1e9:.1f} nJ"
    if j < 1e-3: return f"{j*1e6:.1f} \u00b5J"
    return f"{j*1e3:.2f} mJ"

def fmt_bytes(b):
    if b <= 0: return ''
    if b < 1024: return f"{b} B"
    if b < 1e6: return f"{b/1024:.1f} KB"
    if b < 1e9: return f"{b/1e6:.1f} MB"
    return f"{b/1e9:.2f} GB"


def stage_data_movement(stage):
    """Total read/write bytes (FP16) for a stage."""
    rd = sum(op.get('input_accesses', 0) + op.get('weight_accesses', 0)
             for op in stage.get('operators', []))
    wr = sum(op.get('output_accesses', 0) for op in stage.get('operators', []))
    return rd * 2, wr * 2


def is_pim(stage):
    return (stage.get('assigned_chiplet', {}).get('architecture') or '') == 'PIM'


def draw_chiplet_box(ax, x, y, w, h, stage):
    """Draw a single chiplet box with all annotations."""
    arch = stage['assigned_chiplet']['architecture'] or 'unknown'
    color = ARCH_COLORS.get(arch, '#999')
    arch_label = ARCH_SHORT.get(arch, arch)

    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.06",
                           facecolor=color, edgecolor='black',
                           linewidth=1.8, alpha=0.9)
    ax.add_patch(rect)

    cx = x + w / 2
    tp = stage['tensor_parallelism']
    bonding = stage['bonding']
    pe_dim = stage['assigned_chiplet']['pe_array_dim']
    glb_kb = stage['assigned_chiplet']['glb_capacity_KB']
    dram = stage['dram_type']

    # Header
    header = f"S{stage['stage_index']}: {arch_label}"
    ax.text(cx, y + h - 0.15, header,
            ha='center', va='top', fontsize=8.5, fontweight='bold', color='white')

    # Config - show PE scale factors and array dims
    pe_x_scale = stage['assigned_chiplet'].get('pe_x_scale', 1) or 1
    pe_y_scale = stage['assigned_chiplet'].get('pe_y_scale', 1) or 1
    config_str = f"PE: {pe_x_scale}\u00d7{pe_y_scale} ({pe_dim}) | GLB: {glb_kb}KB | {bonding}"
    ax.text(cx, y + h - 0.42, config_str,
            ha='center', va='top', fontsize=5.5, color='#ffffffcc')

    # For PIM: note GDDR7 is integrated
    if is_pim(stage):
        ax.text(cx, y + h - 0.62, f"GDDR7 integrated (compute+buffer)",
                ha='center', va='top', fontsize=5, color='#ffdddd', style='italic')

    # Operators — show per-op chiplet if VG with different chiplets
    ops_entries = stage.get('operators', [])
    has_vg_chiplets = (stage.get('is_virtual_group', False) and
                       any('assigned_chiplet' in op for op in ops_entries))
    if has_vg_chiplets:
        # Show each op with its own chiplet
        op_lines = []
        for op in ops_entries:
            oname = shorten_op(op['name'])
            if 'assigned_chiplet' in op:
                oarch = ARCH_SHORT.get(op['assigned_chiplet'].get('architecture', ''), '')
                op_lines.append(f"{oname} \u2192 {oarch}")
            else:
                op_lines.append(oname)
        ops_text = ' | '.join(op_lines)
        par_tag = " [parallel]"
    else:
        ops = [shorten_op(o) for o in stage['fusion_group_operators']]
        par_tag = ""
        if stage['is_parallel']:
            par_tag = f" [\u00d7{tp} parallel]" if tp > 1 else " [parallel]"
        elif tp > 1:
            par_tag = f" [\u00d7{tp} TP]"
        ops_text = ', '.join(ops)
        if len(ops_text) > 32:
            ops_text = ', '.join(ops[:3]) + f' +{len(ops)-3}'
    ax.text(cx, y + h / 2 + 0.15, ops_text + par_tag,
            ha='center', va='center', fontsize=7, color='white', style='italic')

    # Metrics — read area from JSON; fallback computes unified PHY if JSON is stale
    agg = stage['aggregate']
    core_mm2 = agg.get('core_area_mm2', agg.get('area_mm2', 0)) or 0
    # Unified PHY: DDR5/LPDDR5 shared (7.5) + GDDR7 (8.0) + HBM3 (19.28) = 34.78 mm2 @2.5D
    if 'unified_phy_area_mm2' in agg:
        phy_mm2 = agg['unified_phy_area_mm2'] or 0
        ctrl_mm2 = agg.get('unified_ctrl_area_mm2', 0) or 0
        total_mm2 = agg.get('total_area_mm2', 0) or (core_mm2 + phy_mm2 + ctrl_mm2)
    else:
        phy_groups = [7.5, 8.0, 19.28]
        bonding_val = stage.get('bonding', '2.5D')
        phy_mm2 = sum(p if bonding_val == '2.5D' else p * 2.2 for p in phy_groups)
        ctrl_mm2 = 0.07 + 0.07 + 1.00
        total_mm2 = core_mm2 + phy_mm2 + ctrl_mm2

    power = stage['aggregate']['static_power_W']
    lat = stage['aggregate']['latency_s']
    energy = stage['aggregate']['dynamic_energy_J']

    area_str = f"Core: {core_mm2:.1f} + PHY: {phy_mm2:.1f} + Ctrl: {ctrl_mm2:.2f} = {total_mm2:.1f} mm\u00b2"
    m2 = f"Power: {fmt_power(power)}  |  Lat: {fmt_lat(lat)}  |  E: {fmt_energy(energy)}"
    ax.text(cx, y + 0.50, area_str, ha='center', va='bottom',
            fontsize=5.5, color='white', family='monospace')
    ax.text(cx, y + 0.18, m2, ha='center', va='bottom',
            fontsize=5.5, color='white', family='monospace')


def draw_dram_diamond(ax, cx, cy, w, h, dram_type):
    """Draw a DRAM diamond node with area annotation."""
    diamond = plt.Polygon([
        [cx, cy + h/2], [cx + w/2, cy], [cx, cy - h/2], [cx - w/2, cy],
    ], facecolor=DRAM_COLOR, edgecolor='black', linewidth=1.2, alpha=0.9)
    ax.add_patch(diamond)
    dspec = DRAM_SPECS.get(dram_type, {})
    bw = dspec.get('bw', 0)
    buf_area = dspec.get('area_per_GB', 0)
    ax.text(cx, cy + 0.12, dram_type,
            ha='center', va='center', fontsize=7, fontweight='bold')
    ax.text(cx, cy - 0.08, f"{bw} GB/s",
            ha='center', va='center', fontsize=5, color='#555')
    ax.text(cx, cy - h/2 - 0.08, f"{buf_area} mm\u00b2/GB",
            ha='center', va='top', fontsize=4.5, color='#777')


def draw_dag(net_config, output_path):
    """Draw pure DAG with proper PIM handling and off-CP connections."""
    stages = net_config['stages']
    cp_stages = [s for s in stages if not s['is_off_critical_path']]
    offcp_stages = [s for s in stages if s['is_off_critical_path']]
    n_cp = len(cp_stages)
    if n_cp == 0:
        return

    # Layout
    max_per_row = 4
    n_rows = math.ceil(n_cp / max_per_row)

    cw = 2.8   # chiplet width
    ch = 2.8   # chiplet height (taller for PIM note)
    dw = 1.6   # dram diamond width
    dh = 0.9
    gap = 0.5
    row_gap = 2.5
    pim_direct_w = 0.8  # width for PIM direct connection (no DRAM diamond)

    # Compute cell width: chiplet + (dram or pim_direct) + gaps
    cell_w = cw + dw + 2 * gap
    cols_this = min(n_cp, max_per_row)
    row_w = cols_this * cell_w
    total_h = n_rows * (ch + row_gap)
    if offcp_stages:
        total_h += 2.5

    fig_w = max(row_w + 2, 10)
    fig_h = max(total_h + 3, 6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-1, row_w + 1)
    ax.set_ylim(-total_h + ch, ch + 2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    net_name = net_config['network']
    bs = net_config['batch_size']
    total_e = net_config.get('total_energy_J')
    total_l = net_config.get('total_latency_s')
    title = f"{net_name}  (B={bs})"
    if total_l: title += f"  |  Lat: {fmt_lat(total_l)}"
    if total_e: title += f"  |  Obj: {total_e:.3e}"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    # Place chiplet boxes
    positions = []  # (cx, cy, x, y, row, col) per CP stage
    for idx in range(n_cp):
        row = idx // max_per_row
        col = idx % max_per_row
        # Serpentine
        if row % 2 == 1:
            ncols = min(max_per_row, n_cp - row * max_per_row)
            col = ncols - 1 - col

        x = col * cell_w
        y = -(row * (ch + row_gap))
        draw_chiplet_box(ax, x, y, cw, ch, cp_stages[idx])
        positions.append((x + cw/2, y + ch/2, x, y, row, col))

    # ---- Draw connections between consecutive CP stages ----
    # Track DRAM diamond positions by buffer_config boundary index for sharing
    dram_diamond_pos = {}  # boundary_index → (cx, cy)

    for i in range(n_cp - 1):
        s_cur = cp_stages[i]
        s_nxt = cp_stages[i + 1]
        p_cur = positions[i]
        p_nxt = positions[i + 1]

        cur_row, nxt_row = p_cur[4], p_nxt[4]
        # Shared boundary = buf_start of next stage (from JSON)
        boundary_idx = s_nxt.get('buf_start', i + 1)
        dram_type = s_nxt['dram_type']
        both_pim = is_pim(s_cur) and is_pim(s_nxt)
        nxt_is_pim = is_pim(s_nxt)

        _, write_bytes = stage_data_movement(s_cur)
        read_bytes, _ = stage_data_movement(s_nxt)

        is_inter = (s_cur['assigned_chiplet']['identifier'] !=
                     s_nxt['assigned_chiplet']['identifier'])
        arrow_color = '#cc0000' if is_inter else '#333333'
        arrow_lw = 2.0 if is_inter else 1.5

        # PIM consumer: no separate DRAM diamond (GDDR7 is integrated in PIM)
        skip_diamond = both_pim or nxt_is_pim

        if cur_row == nxt_row:
            going_right = p_cur[5] < p_nxt[5]

            if skip_diamond:
                # Direct connection (PIM has integrated GDDR7)
                if going_right:
                    ax.annotate('', xy=(p_nxt[2] - 0.05, p_cur[1]),
                               xytext=(p_cur[2] + cw + 0.05, p_cur[1]),
                               arrowprops=dict(arrowstyle='->', color=arrow_color,
                                              lw=arrow_lw))
                    mid_x = (p_cur[2] + cw + p_nxt[2]) / 2
                else:
                    ax.annotate('', xy=(p_nxt[2] + cw + 0.05, p_cur[1]),
                               xytext=(p_cur[2] - 0.05, p_cur[1]),
                               arrowprops=dict(arrowstyle='->', color=arrow_color,
                                              lw=arrow_lw))
                    mid_x = (p_cur[2] + p_nxt[2] + cw) / 2

                label = 'GDDR7 local' if both_pim else f'{dram_type} (integrated)'
                lbl_color = '#e15759' if both_pim else arrow_color
                ax.text(mid_x, p_cur[1] + 0.2, label,
                        ha='center', fontsize=6, color=lbl_color, style='italic')
                if is_inter:
                    ax.text(mid_x, p_cur[1] + 0.45, 'inter-chiplet',
                           ha='center', fontsize=5, color='#cc0000', fontweight='bold')
            else:
                # Normal: chiplet → DRAM diamond → chiplet
                if going_right:
                    dram_cx = p_cur[2] + cw + gap + dw/2
                else:
                    dram_cx = p_nxt[2] + cw + gap + dw/2
                dram_cy = p_cur[1]

                draw_dram_diamond(ax, dram_cx, dram_cy, dw, dh, dram_type)
                dram_diamond_pos[boundary_idx] = (dram_cx, dram_cy)

                if is_inter:
                    ax.text(dram_cx, dram_cy + dh/2 + 0.12, 'inter-chiplet',
                           ha='center', fontsize=5, color='#cc0000', fontweight='bold')

                if going_right:
                    ax.annotate('', xy=(dram_cx - dw/2, dram_cy),
                               xytext=(p_cur[2] + cw + 0.05, p_cur[1]),
                               arrowprops=dict(arrowstyle='->', color=arrow_color, lw=arrow_lw))
                    ax.annotate('', xy=(p_nxt[2] - 0.05, dram_cy),
                               xytext=(dram_cx + dw/2, dram_cy),
                               arrowprops=dict(arrowstyle='->', color=arrow_color, lw=arrow_lw))
                    w_str = fmt_bytes(write_bytes)
                    if w_str:
                        ax.text((p_cur[2] + cw + dram_cx - dw/2) / 2, dram_cy + 0.28,
                                f"W: {w_str}", ha='center', fontsize=5, color=arrow_color)
                    r_str = fmt_bytes(read_bytes)
                    if r_str:
                        ax.text((dram_cx + dw/2 + p_nxt[2]) / 2, dram_cy - 0.28,
                                f"R: {r_str}", ha='center', fontsize=5, color=arrow_color)
                else:
                    ax.annotate('', xy=(dram_cx + dw/2, dram_cy),
                               xytext=(p_cur[2] - 0.05, p_cur[1]),
                               arrowprops=dict(arrowstyle='->', color=arrow_color, lw=arrow_lw))
                    ax.annotate('', xy=(p_nxt[2] + cw + 0.05, dram_cy),
                               xytext=(dram_cx - dw/2, dram_cy),
                               arrowprops=dict(arrowstyle='->', color=arrow_color, lw=arrow_lw))
        else:
            # Cross-row vertical
            mid_y = (p_cur[1] + p_nxt[1]) / 2
            dram_cx = p_cur[0]
            dram_cy = mid_y

            if skip_diamond:
                ax.annotate('', xy=(p_nxt[0], p_nxt[3] + ch),
                           xytext=(p_cur[0], p_cur[3]),
                           arrowprops=dict(arrowstyle='->', color=arrow_color,
                                          lw=arrow_lw, connectionstyle='arc3,rad=0.2'))
                label = 'GDDR7\nlocal' if both_pim else f'{dram_type}\n(integrated)'
                ax.text(dram_cx + 0.3, mid_y, label,
                        fontsize=5, color='#e15759' if both_pim else arrow_color,
                        style='italic', va='center')
            else:
                draw_dram_diamond(ax, dram_cx, dram_cy, dw, dh, dram_type)
                dram_diamond_pos[boundary_idx] = (dram_cx, dram_cy)
                ax.annotate('', xy=(dram_cx, dram_cy + dh/2),
                           xytext=(p_cur[0], p_cur[3]),
                           arrowprops=dict(arrowstyle='->', color=arrow_color, lw=arrow_lw))
                ax.annotate('', xy=(p_nxt[0], p_nxt[3] + ch),
                           xytext=(dram_cx, dram_cy - dh/2),
                           arrowprops=dict(arrowstyle='->', color=arrow_color, lw=arrow_lw))
                if is_inter:
                    ax.text(dram_cx + dw/2 + 0.1, dram_cy, 'inter-chiplet',
                           fontsize=5, color='#cc0000', fontweight='bold', va='center')

    # ---- Off-CP stages ----
    if offcp_stages:
        offcp_y_base = -(n_rows * (ch + row_gap)) - 0.3
        for j, s in enumerate(offcp_stages):
            ox = j * (cw + 0.5) + 0.5
            oy = offcp_y_base
            oh = 1.8
            arch = s['assigned_chiplet']['architecture'] or 'unknown'
            color = ARCH_COLORS.get(arch, '#999')

            rect = FancyBboxPatch((ox, oy), cw, oh,
                                   boxstyle="round,pad=0.04",
                                   facecolor=color, edgecolor='black',
                                   linewidth=1.2, alpha=0.7, linestyle='--')
            ax.add_patch(rect)

            ops = [shorten_op(o) for o in s['fusion_group_operators']]
            arch_label = ARCH_SHORT.get(arch, arch)
            lat = s['aggregate']['latency_s']
            energy = s['aggregate']['dynamic_energy_J']
            area = s['aggregate'].get('total_area_mm2', 0)
            dram = s['dram_type']

            ax.text(ox + cw/2, oy + oh - 0.12,
                    f"Off-CP S{s['stage_index']}: {', '.join(ops)}",
                    ha='center', va='top', fontsize=7, fontweight='bold', color='white')
            ax.text(ox + cw/2, oy + oh/2 - 0.1,
                    f"{arch_label} | {dram}\n"
                    f"Area: {area:.1f} mm\u00b2 | Lat: {fmt_lat(lat)}\n"
                    f"Energy: {fmt_energy(energy)}",
                    ha='center', va='center', fontsize=5.5, color='white',
                    linespacing=1.3)

            # Use buf_start/buf_end from JSON for correct boundary indexing
            offcp_buf_start = s.get('buf_start', 0)
            offcp_buf_end = s.get('buf_end', 0)
            feeds_to = s.get('offcp_feeds_to_stage')

            offcp_dram_i = s['operators'][0].get('dram_i', dram) if s['operators'] else dram
            offcp_dram_o = s['operators'][0].get('dram_o', dram) if s['operators'] else dram

            # --- Input connection ---
            # Off-CP reads from buffer_config[buf_start] — the pipeline input DRAM.
            # This is NOT a data dependency on S0; both S0 and off-CP read from the same input.
            if offcp_buf_start in dram_diamond_pos:
                # Reuse existing DRAM diamond at this boundary
                dcx, dcy = dram_diamond_pos[offcp_buf_start]
                ax.annotate('',
                    xy=(ox + cw/2, oy + oh),
                    xytext=(dcx, dcy - dh/2),
                    arrowprops=dict(arrowstyle='->', color='#888', lw=1.2,
                                   linestyle='--', connectionstyle='arc3,rad=-0.15'))
            else:
                # No existing diamond (e.g., pipeline input boundary before S0).
                # Draw "input" label on the left side of the off-CP box.
                ax.text(ox - 0.15, oy + oh/2,
                        f'input\n({offcp_dram_i})',
                        ha='right', va='center', fontsize=5.5, color='#888',
                        style='italic', fontweight='bold')
                ax.annotate('',
                    xy=(ox, oy + oh/2),
                    xytext=(ox - 0.6, oy + oh/2),
                    arrowprops=dict(arrowstyle='->', color='#888', lw=1.2,
                                   linestyle='--'))

            # --- Output connection: off-CP → feeds_to ---
            if feeds_to is not None and feeds_to < len(positions):
                dst = positions[feeds_to]
                dst_stage = cp_stages[feeds_to]

                if is_pim(dst_stage):
                    # Destination is PIM — GDDR7 integrated, connect directly
                    ax.annotate('',
                        xy=(dst[0], dst[3]),
                        xytext=(ox + cw/2, oy + oh),
                        arrowprops=dict(arrowstyle='->', color='#888', lw=1.2,
                                       linestyle='--', connectionstyle='arc3,rad=-0.15'))
                    ax.text((ox + cw/2 + dst[0]) / 2 + 0.3,
                            (oy + oh + dst[3]) / 2,
                            f'{offcp_dram_o} (integrated)',
                            fontsize=5, color='#888', style='italic')
                elif offcp_buf_end in dram_diamond_pos:
                    # Reuse existing DRAM diamond at the output boundary
                    dcx, dcy = dram_diamond_pos[offcp_buf_end]
                    ax.annotate('',
                        xy=(dcx, dcy - dh/2),
                        xytext=(ox + cw/2, oy + oh),
                        arrowprops=dict(arrowstyle='->', color='#888', lw=1.2,
                                       linestyle='--', connectionstyle='arc3,rad=-0.15'))
                else:
                    # No existing diamond — draw directly with label
                    ax.annotate('',
                        xy=(dst[0], dst[3]),
                        xytext=(ox + cw/2, oy + oh),
                        arrowprops=dict(arrowstyle='->', color='#888', lw=1.2,
                                       linestyle='--', connectionstyle='arc3,rad=-0.15'))
                    ax.text((ox + cw/2 + dst[0]) / 2 + 0.3,
                            (oy + oh + dst[3]) / 2,
                            f'{offcp_dram_o}', fontsize=6, color='#888',
                            style='italic', fontweight='bold')

    # Legend
    used_archs = set()
    for s in stages:
        a = s['assigned_chiplet']['architecture']
        if a: used_archs.add(a)
    handles = [mpatches.Patch(facecolor=ARCH_COLORS.get(a, '#ccc'),
                              label=ARCH_SHORT.get(a, a)) for a in sorted(used_archs)]
    handles.append(mpatches.Patch(facecolor=DRAM_COLOR, label='DRAM'))
    handles.append(plt.Line2D([0], [0], color='#cc0000', lw=2, label='Inter-chiplet'))
    handles.append(plt.Line2D([0], [0], color='#888', lw=1.2, linestyle='--', label='Off-CP'))
    ax.legend(handles=handles, loc='upper left', fontsize=7, framealpha=0.9)

    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='pnr_configs_edp/pnr_all_networks.json')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    out_dir = args.output_dir or os.path.join(os.path.dirname(args.input), 'figures')
    os.makedirs(out_dir, exist_ok=True)

    for net_key, net_config in data['networks'].items():
        print(f"--- {net_key} ---")
        draw_dag(net_config, os.path.join(out_dir, f'dag_{net_key}.png'))


if __name__ == '__main__':
    main()
