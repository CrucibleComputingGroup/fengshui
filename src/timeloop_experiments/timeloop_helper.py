"""Timeloop helper for LLaMA / Qwen / ViT BF16 workloads.

Uses Approach C: all mapping constraints are applied directly on
architecture nodes via spec.architecture.find(name).constraints —
no temporary mapping_constraints.yaml files are written.

Architecture files: arch_bf.yaml (selected via TOP_JINJA_PATH with use_bf_arch=True).

ViT support (vit_b16_s197, vit_l16_s197, vit_h14_s257):
  ViT operators reuse the same GEMM4D / Attn5D forms as LLaMA:
    Projection (B,N,C,M):     q_proj/k_proj/v_proj/o_proj/gate_proj/down_proj/lm_head
    Attention  (B,H,Q,K,D):   attn_qk / attn_v
    Softmax    (B,H,Q,K):     softmax_max/sub_exp/sum/div  →  simple_vector arch
  The existing _apply_projection_constraints / _apply_attention_constraints /
  _apply_softmax_constraints functions handle all ViT ops without modification.
  TP is disabled for ViT (tp_degree=1, VIT_TP_CONFIG used for routing only).
"""

import os
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, Dict
import sys
import yaml

# ---------------------------------------------------------------------------
# Path bootstrapping — add scripts/ so utility_functions, global_parameter,
# and chiplet_dataclass can be imported.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in [os.path.join(_THIS_DIR, "..", "scripts"), os.path.join(_THIS_DIR, "..")]:
    if os.path.isfile(os.path.join(_candidate, "utility_functions.py")) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import pytimeloop.timeloopfe.v4 as tl
from pytimeloop.timeloopfe.v4.constraints import (
    ProblemDataspaceList,
    Factors,
    Permutation,
)

import utility_functions
from chiplet_dataclass import *
from global_parameter import *


# ===================================================================
# Internal: apply mapping constraints directly on architecture nodes
# ===================================================================

def _apply_projection_constraints(spec, arch: str, pe_x: int, pe_y: int, dims: dict):
    """Apply projection-operator constraints via Approach C.

    Dims: [B, N, C, M]
    Dataspaces:
        Inputs1 = weights  (M, C)
        Inputs2 = activations (B, N, C)
        Outputs = result   (B, N, M)
    """
    B = dims.get("B", 1)
    C = dims.get("C", 1)
    M = dims.get("M", 1)

    dim_order = ["B", "N", "C", "M"]
    all_ones = [f"{d}=1" for d in dim_order]

    ds_w, ds_a, ds_o = "Inputs1", "Inputs2", "Outputs"

    # ── Per-architecture, per-level permutation & factor strategy ──────────
    #
    # Translated from the official Timeloop exercise example_designs:
    #   - eyeriss_like  → Row Stationary   (each spad has its own permutation)
    #   - gemmini_like  → Output Stationary (pe_spad keeps Outputs)
    #   - simba_like    → Weight Stationary (PEWeightBuffer keeps Weights)
    #
    # Conv7D→GEMM4D dimension mapping:
    #   Conv: N,P,Q,R,S,C,M  →  GEMM: B,N,C,M
    #   N(batch)→B | P,Q(output spatial)→N | R,S(filter)→(none) | C→C | M→M
    #
    # Official eyeriss_like exercise permutations (Conv7D):
    #   ifmap_spad:   [N, M, C, P, Q, R, S]  factors: all=1
    #   weights_spad: [N, M, P, Q, S, C, R]  factors: N=1,M=1,P=1,Q=1,S=1
    #   psum_spad:    [N, C, P, Q, R, S, M]  factors: N=1,C=1,R=1,S=1,P=1,Q=1
    #   PE_column:    [N, C, P, R, S, Q, M]  spatial
    #   PE:           [N, P, Q, R, S, C, M]  spatial
    #
    # Official simple_weight_stationary exercise:
    #   pe_spad (keeps Weights): permutation: [P, Q, R, S]
    #
    # Official simple_output_stationary exercise:
    #   pe_spad (keeps Outputs): permutation: [R, S, P, Q]

    # ------------------------------------------------------------------
    if arch == "eyeriss_like":
        # Row Stationary: each spad level has a distinct permutation
        # following the official eyeriss_like example_design.
        #
        # GEMM translation of official Conv permutations (inner→outer):
        #   Conv.N→B, Conv.P/Q→N, Conv.R/S→(skip), Conv.C→C, Conv.M→M
        #
        #   ifmap_spad:   [N,M,C,P,Q,R,S] → [B,M,C,N]
        #   weights_spad: [N,M,P,Q,S,C,R] → [B,M,N,C]  (S,R→skip, only C,M remain outer)
        #   psum_spad:    [N,C,P,Q,R,S,M] → [B,C,N,M]
        #   PE_column:    [N,C,P,R,S,Q,M] → [B,C,N,M] split=999 (all X)
        #   PE:           [N,P,Q,R,S,C,M] → [B,N,C,M] split=0 (all Y)

        m_spatial = min(M, pe_x)
        c_spatial = min(C, pe_y)

        # Spatial: PE_column tiles M along X, PE tiles C along Y
        pe_col = spec.architecture.find("PE_column")
        pe_col.constraints.spatial["factors"] = Factors(
            [f"M={m_spatial}", "B=1", "N=1", "C=1"]
        )
        pe_col.constraints.spatial["split"] = 999
        pe_col.constraints.spatial["permutation"] = Permutation(["B", "C", "N", "M"])

        pe = spec.architecture.find("PE")
        pe.constraints.spatial["factors"] = Factors(
            [f"C={c_spatial}", "B=1", "N=1", "M=1"]
        )
        pe.constraints.spatial["split"] = 0
        pe.constraints.spatial["permutation"] = Permutation(["B", "N", "C", "M"])

        # shared_glb: keep ALL three dataspaces (weights cached for batch reuse)
        # B=batch at GLB: weights stay cached across all B iterations at GLB level.
        # Experiment result: g{B}_i1 gives -13% energy/token at B=4, -7% at B=8.
        glb = spec.architecture.find("shared_glb")
        glb.constraints.dataspace["keep"] = ProblemDataspaceList([ds_w, ds_a, ds_o])
        glb.constraints.dataspace["bypass"] = ProblemDataspaceList([])
        glb.constraints.temporal["factors"] = Factors([f"B={B}"])

        # ifmap_spad (12 entries): keep activations
        # Official: perm=[N,M,C,P,Q,R,S] → GEMM: [B,M,C,N] all=1
        ifmap = spec.architecture.find("ifmap_spad")
        ifmap.constraints.dataspace["keep"] = ProblemDataspaceList([ds_a])
        ifmap.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w, ds_o])
        ifmap.constraints.temporal["factors"] = Factors(all_ones)
        ifmap.constraints.temporal["permutation"] = Permutation(["B", "M", "C", "N"])

        # weights_spad (224 entries): keep weights
        # Official: perm=[N,M,P,Q,S,C,R] → GEMM: [B,M,N,C] factors=B=1,N=1
        wspad = spec.architecture.find("weights_spad")
        wspad.constraints.dataspace["keep"] = ProblemDataspaceList([ds_w])
        wspad.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_a, ds_o])
        wspad.constraints.temporal["factors"] = Factors(["B=1", "N=1"])
        wspad.constraints.temporal["permutation"] = Permutation(["B", "M", "N", "C"])

        # psum_spad (24 entries): keep outputs
        # Official: perm=[N,C,P,Q,R,S,M] → GEMM: [B,C,N,M] factors=C=1,B=1,N=1
        pspad = spec.architecture.find("psum_spad")
        pspad.constraints.dataspace["keep"] = ProblemDataspaceList([ds_o])
        pspad.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w, ds_a])
        pspad.constraints.temporal["factors"] = Factors(["C=1", "B=1", "N=1"])
        pspad.constraints.temporal["permutation"] = Permutation(["B", "C", "N", "M"])

    # ------------------------------------------------------------------
    elif arch == "gemmini_like":
        # Output Stationary: pe_spad keeps outputs.
        # Official OS example: pe_spad permutation=[R,S,P,Q]
        # → GEMM: put reduction dim C inner (like R,S), output dims outer
        # → perm=[C,B,N,M]

        m_spatial = min(M, pe_x)
        c_spatial = min(C, pe_y)

        pe_col = spec.architecture.find("PE_column")
        pe_col.constraints.spatial["factors"] = Factors(
            [f"M={m_spatial}", "B=1", "N=1", "C=1"]
        )
        pe_col.constraints.spatial["split"] = 999
        pe_col.constraints.spatial["permutation"] = Permutation(["C", "M"])

        pe = spec.architecture.find("PE")
        pe.constraints.spatial["factors"] = Factors(
            [f"C={c_spatial}", "B=1", "N=1", "M=1"]
        )
        pe.constraints.spatial["split"] = 0
        pe.constraints.spatial["permutation"] = Permutation(["C", "M"])

        # shared_glb: keep activations + outputs
        glb = spec.architecture.find("shared_glb")
        glb.constraints.dataspace["keep"] = ProblemDataspaceList([ds_a, ds_o])
        glb.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w])
        glb.constraints.temporal["factors"] = Factors([])

        # pe_spad (256 entries): keep outputs — output stationary
        # Official OS: perm=[R,S,P,Q] → GEMM: perm=[C,B,N,M]
        # (C inner = reduction streams through, outputs stay)
        # B=batch at pe_spad: outputs for all B cached here, weights reused.
        # Experiment result: -20% energy/token at B=4, -18% at B=8.
        pe_spad = spec.architecture.find("pe_spad")
        pe_spad.constraints.dataspace["keep"] = ProblemDataspaceList([ds_o])
        pe_spad.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w, ds_a])
        pe_spad.constraints.temporal["factors"] = Factors([f"B={B}"])
        pe_spad.constraints.temporal["permutation"] = Permutation(["C", "B", "N", "M"])

        # Registers (depth=1): each keeps exactly one dataspace
        # Official: weight_reg factors=[R=1,S=1,M=1,C=1] → all_ones
        #           input_reg  factors=[P=1,Q=1,C=1,N=1] → all_ones
        #           output_reg factors=[P=1,Q=1,M=1,N=1] → all_ones
        for reg_name, keep_ds, bypass_ds in [
            ("weight_reg", ds_w, [ds_a, ds_o]),
            ("input_activation_reg", ds_a, [ds_w, ds_o]),
            ("output_activation_reg", ds_o, [ds_w, ds_a]),
        ]:
            reg = spec.architecture.find(reg_name)
            reg.constraints.dataspace["keep"] = ProblemDataspaceList([keep_ds])
            reg.constraints.dataspace["bypass"] = ProblemDataspaceList(bypass_ds)
            reg.constraints.temporal["factors"] = Factors(all_ones)

    # ------------------------------------------------------------------
    elif arch == "simba_like":
        # Weight Stationary: PEWeightBuffer keeps weights.
        # Official WS example: pe_spad (keeps weights) perm=[P,Q,R,S]
        # → GEMM: put output dims B,N inner (like P,Q), weight dims outer
        # → perm=[B,N,C,M]

        C_val = dims.get("C", 1)
        M_val = dims.get("M", 1)

        # PE spatial: tile C
        C_pe = min(C_val, pe_x)
        pe_node = spec.architecture.find("PE")
        pe_node.constraints.spatial["factors"] = Factors(
            [f"C={C_pe}", "B=1", "N=1", "M=1"]
        )
        pe_node.constraints.spatial["permutation"] = Permutation(["C", "M"])

        # reg_mac spatial: tile M (output channel)
        M_reg = min(M_val, reg_mac_spatial)
        reg_node = spec.architecture.find("reg_mac")
        reg_node.constraints.spatial["factors"] = Factors(
            [f"M={M_reg}", "B=1", "N=1", "C=1"]
        )
        reg_node.constraints.spatial["permutation"] = Permutation(["M", "C"])

        # distributed_buffers spatial: tile M (output channel)
        M_dist = min(M_val, pe_y)
        dist_node = spec.architecture.find("distributed_buffers")
        dist_node.constraints.spatial["factors"] = Factors(
            [f"M={M_dist}", "B=1", "N=1", "C=1"]
        )
        dist_node.constraints.spatial["permutation"] = Permutation(["M", "C"])

        # shared_glb: keep activations + outputs, bypass weights
        glb = spec.architecture.find("shared_glb")
        glb.constraints.dataspace["keep"] = ProblemDataspaceList([ds_a, ds_o])
        glb.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w])
        glb.constraints.temporal["factors"] = Factors(["B=1"])

        # PEInputBuffer (8KB): keep activations
        pib = spec.architecture.find("PEInputBuffer")
        pib.constraints.dataspace["keep"] = ProblemDataspaceList([ds_a])
        pib.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w, ds_o])
        pib.constraints.temporal["factors"] = Factors(["M=1"])

        # PEWeightBuffer (32KB): keep weights — weight stationary
        # Official WS: pe_spad (keeps weights) perm=[P,Q,R,S]
        # → GEMM: perm=[B,N,C,M], B,N inner (not in wt), C,M outer (stay)
        pwb = spec.architecture.find("PEWeightBuffer")
        pwb.constraints.dataspace["keep"] = ProblemDataspaceList([ds_w])
        pwb.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_a, ds_o])
        pwb.constraints.temporal["factors"] = Factors(["B=1", "N=1"])
        pwb.constraints.temporal["permutation"] = Permutation(["B", "N", "C", "M"])

        # PEAccuBuffer (128 entries): keep outputs
        # B=batch at PEAccuBuffer: all B outputs cached, weights reused from
        # PEWeightBuffer (B=1). Experiment: sa{B}_w1 gives -11% energy at B=16.
        pab = spec.architecture.find("PEAccuBuffer")
        pab.constraints.dataspace["keep"] = ProblemDataspaceList([ds_o])
        pab.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w, ds_a])
        pab.constraints.temporal["factors"] = Factors([f"B={B}"])

        # InputPassthrough: keep activations
        ipt = spec.architecture.find("InputPassthrough")
        ipt.constraints.dataspace["keep"] = ProblemDataspaceList([ds_a])
        ipt.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w, ds_o])

        # PEWeightRegs (depth=1): keep weights
        pwr = spec.architecture.find("PEWeightRegs")
        pwr.constraints.dataspace["keep"] = ProblemDataspaceList([ds_w])
        pwr.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_a, ds_o])
        pwr.constraints.temporal["factors"] = Factors(all_ones)


def _apply_attention_constraints(spec, arch: str, pe_x: int, pe_y: int, dims: dict, op_name: str):
    """Apply attention-operator constraints via Approach C.

    Dims: [B, H, Q, K, D]
    attn_qk: Inputs1(B,H,Q,D) @ Inputs2(B,H,K,D) -> Outputs(B,H,Q,K)
    attn_v:  Inputs1(B,H,Q,K) @ Inputs2(B,H,K,D) -> Outputs(B,H,Q,D)
    """
    Q = dims.get("Q", 1)
    K = dims.get("K", 1)
    D = dims.get("D", 1)
    H = dims.get("H", 1)

    dim_order = ["B", "H", "Q", "K", "D"]
    all_ones = [f"{d}=1" for d in dim_order]

    ds_i, ds_w, ds_o = "Inputs1", "Inputs2", "Outputs"

    is_qk = "qk" in op_name

    # Attention dim roles (same for all architectures):
    #   attn_qk: reduction=D, output-channel=K, output-spatial=Q
    #   attn_v:  reduction=K, output-channel=D, output-spatial=Q
    if is_qk:
        red_dim, red_val = "D", D
        out_ch, out_ch_val = "K", K
    else:
        red_dim, red_val = "K", K
        out_ch, out_ch_val = "D", D

    # ------------------------------------------------------------------
    if arch == "eyeriss_like":
        # Row Stationary: PE_column(X) tiles output-channel (K for qk, D for v),
        # PE(Y) tiles reduction dim (D for qk, K for v).
        # Per-level permutations follow the official eyeriss_like design.
        #
        # Conv7D→Attn5D permutation mapping (inner→outer):
        #   PE_column: [N,C,P,R,S,Q,M] → [B,H,red,Q,out-ch]  (X tiles out-ch, like M in proj)
        #   PE:        [N,P,Q,R,S,C,M] → [B,H,Q,red,out-ch]   (Y tiles red, like C in proj)
        #   ifmap:     [N,M,C,P,Q,R,S] → [B,H,out-ch,red,Q]
        #   wspad:     [N,M,P,Q,S,C,R] → [B,H,out-ch,Q,red]
        #   pspad:     [N,C,P,Q,R,S,M] → [B,H,red,Q,out-ch]

        out_ch_spatial = min(out_ch_val, pe_x)
        red_spatial = min(red_val, pe_y)

        # Spatial: PE_column tiles out_ch along X (like M in proj), PE tiles reduction along Y
        pe_col = spec.architecture.find("PE_column")
        pe_col.constraints.spatial["factors"] = Factors(
            [f"{out_ch}={out_ch_spatial}", "B=1", "H=1", "Q=1", f"{red_dim}=1"]
        )
        pe_col.constraints.spatial["split"] = 999
        pe_col.constraints.spatial["permutation"] = Permutation(
            ["B", "H", red_dim, "Q", out_ch]
        )

        pe = spec.architecture.find("PE")
        pe.constraints.spatial["factors"] = Factors(
            [f"{red_dim}={red_spatial}", "B=1", "H=1", "Q=1", f"{out_ch}=1"]
        )
        pe.constraints.spatial["split"] = 0
        pe.constraints.spatial["permutation"] = Permutation(
            ["B", "H", "Q", red_dim, out_ch]
        )

        # shared_glb: keep Inputs1 + Outputs, bypass Inputs2
        # Pin output-channel at GLB (compiler hint for accumulation)
        glb = spec.architecture.find("shared_glb")
        glb.constraints.dataspace["keep"] = ProblemDataspaceList([ds_i, ds_o])
        glb.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w])
        glb.constraints.temporal["factors"] = Factors(
            [f"{out_ch}={dims[out_ch]}"]
        )

        # ifmap_spad (12 entries): keep Inputs1
        # Perm: batch, out-ch, reduction, spatial (inner→outer)
        ifmap = spec.architecture.find("ifmap_spad")
        ifmap.constraints.dataspace["keep"] = ProblemDataspaceList([ds_i])
        ifmap.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w, ds_o])
        ifmap.constraints.temporal["factors"] = Factors(all_ones)
        ifmap.constraints.temporal["permutation"] = Permutation(
            ["B", "H", out_ch, red_dim, "Q"]
        )

        # weights_spad (192 entries): keep Inputs2
        # Perm: batch, out-ch, spatial, reduction (inner→outer)
        # Factors: B=1,H=1,Q=1 (batch+spatial pinned; out-ch & reduction free)
        wspad = spec.architecture.find("weights_spad")
        wspad.constraints.dataspace["keep"] = ProblemDataspaceList([ds_w])
        wspad.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_i, ds_o])
        wspad.constraints.temporal["factors"] = Factors(["B=1", "H=1", "Q=1"])
        wspad.constraints.temporal["permutation"] = Permutation(
            ["B", "H", out_ch, "Q", red_dim]
        )

        # psum_spad (16 entries): keep Outputs
        # Perm: batch, reduction, spatial, out-ch (inner→outer)
        # Factors: red=1,B=1,H=1,Q=1 (only out-ch free)
        pspad = spec.architecture.find("psum_spad")
        pspad.constraints.dataspace["keep"] = ProblemDataspaceList([ds_o])
        pspad.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w, ds_i])
        pspad.constraints.temporal["factors"] = Factors(
            [f"{red_dim}=1", "B=1", "H=1", "Q=1"]
        )
        pspad.constraints.temporal["permutation"] = Permutation(
            ["B", "H", red_dim, "Q", out_ch]
        )

    # ------------------------------------------------------------------
    elif arch == "gemmini_like":
        # Output Stationary (following simple_output_stationary reference):
        #   pe_spad keeps Outputs, bypasses Inputs1 + Inputs2
        #   PE spatial: split=1, perm [C, M] → [red_dim, out_ch]
        #     PE_column(X) tiles out_ch, PE(Y) tiles red_dim
        #   pe_spad temporal: OS perm [R,S,P,Q] → [red,B,H,Q, out_ch]
        #     reduction (red) inner (streams through), output dims outer (stay)
        #
        # For Gemmini 2-level spatial: out_ch and red_dim are both ≥64,
        # so the array is fully utilized without needing the Q trick.

        # PE_column(X): tile out_ch (like M in conv ref split=1 perm [C,M])
        out_ch_spatial = min(out_ch_val, pe_x)
        pe_col = spec.architecture.find("PE_column")
        pe_col.constraints.spatial["factors"] = Factors(
            [f"{out_ch}={out_ch_spatial}", "B=1", "H=1", "Q=1", f"{red_dim}=1"]
        )
        pe_col.constraints.spatial["split"] = 999
        pe_col.constraints.spatial["permutation"] = Permutation(
            [red_dim, out_ch]
        )

        # PE(Y): tile red_dim (like C in conv ref)
        red_spatial = min(red_val, pe_y)
        pe = spec.architecture.find("PE")
        pe.constraints.spatial["factors"] = Factors(
            [f"{red_dim}={red_spatial}", "B=1", "H=1", "Q=1", f"{out_ch}=1"]
        )
        pe.constraints.spatial["split"] = 0
        pe.constraints.spatial["permutation"] = Permutation(
            [red_dim, out_ch]
        )

        # shared_glb: keep Inputs1 + Outputs, bypass Inputs2
        # Loose constraint — pin batch dims only
        glb = spec.architecture.find("shared_glb")
        glb.constraints.dataspace["keep"] = ProblemDataspaceList([ds_i, ds_o])
        glb.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w])
        glb.constraints.temporal["factors"] = Factors(["B=1", "H=1"])

        # pe_spad (256 entries): OUTPUT STATIONARY — keep Outputs
        # OS ref perm: [R,S,P,Q] → Attn: [red,B,H,Q, out_ch]
        #   reduction (red) inner (streams through), output dims outer (stay)
        pe_spad = spec.architecture.find("pe_spad")
        pe_spad.constraints.dataspace["keep"] = ProblemDataspaceList([ds_o])
        pe_spad.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_i, ds_w])
        pe_spad.constraints.temporal["factors"] = Factors(["B=1", "H=1"])
        pe_spad.constraints.temporal["permutation"] = Permutation(
            [red_dim, "B", "H", "Q", out_ch]
        )

        # Registers (depth=1): each keeps its dataspace, all_ones
        # Ref: weight_reg R=1 S=1 M=1 C=1, input_reg P=1 Q=1 C=1 N=1,
        #      output_reg P=1 Q=1 M=1 N=1  → all map to all_ones for attention
        for reg_name, keep_ds, bypass_ds in [
            ("weight_reg", ds_w, [ds_i, ds_o]),
            ("input_activation_reg", ds_i, [ds_w, ds_o]),
            ("output_activation_reg", ds_o, [ds_w, ds_i]),
        ]:
            reg = spec.architecture.find(reg_name)
            reg.constraints.dataspace["keep"] = ProblemDataspaceList([keep_ds])
            reg.constraints.dataspace["bypass"] = ProblemDataspaceList(bypass_ds)
            reg.constraints.temporal["factors"] = Factors(all_ones)

    # ------------------------------------------------------------------
    elif arch == "simba_like":
        # Weight Stationary: PEWeightBuffer keeps Inputs2 (keys/values).
        # Following the official Simba design (JSSC 2020 / MICRO 2019):
        #
        # Spatial tiling assignment (consistent with projection):
        #   PE (meshX):              red_dim (D or K — reduction)
        #   reg_mac (Y):             out_ch  (K or D — output channel)
        #   distributed_buffers (Y): out_ch  (K or D — output channel)
        #
        # Buffer-level constraints follow the official Simba WS dataflow:
        #   GLB: pins out_ch (compiler hint for accumulation)
        #   PEWeightBuffer: weight dims free, output/batch dims pinned
        #   PEAccuBuffer: only out_ch free (accumulation)
        #   Conv ref permutations: [P,Q→Q inner, N→B,H outer]

        # PE spatial: tile red_dim (reduction dimension)
        red_pe = min(red_val, pe_x)
        pe_node = spec.architecture.find("PE")
        pe_node.constraints.spatial["factors"] = Factors(
            [f"{red_dim}={red_pe}", "B=1", "H=1", "Q=1", f"{out_ch}=1"]
        )
        pe_node.constraints.spatial["permutation"] = Permutation(
            [red_dim, out_ch, "Q", "H", "B"]
        )

        # distributed_buffers spatial: tile out_ch (like M in conv)
        out_ch_dist = min(out_ch_val, pe_y)
        dist_node = spec.architecture.find("distributed_buffers")
        dist_node.constraints.spatial["factors"] = Factors(
            [f"{out_ch}={out_ch_dist}", "B=1", "H=1", "Q=1", f"{red_dim}=1"]
        )
        dist_node.constraints.spatial["permutation"] = Permutation(
            [out_ch, red_dim, "Q", "H", "B"]
        )

        # reg_mac spatial: tile out_ch (output channel)
        out_ch_reg = min(out_ch_val, reg_mac_spatial)
        reg_node = spec.architecture.find("reg_mac")
        reg_node.constraints.spatial["factors"] = Factors(
            [f"{out_ch}={out_ch_reg}", "B=1", "H=1", "Q=1", f"{red_dim}=1"]
        )
        reg_node.constraints.spatial["permutation"] = Permutation(
            [out_ch, red_dim, "Q", "H", "B"]
        )

        # shared_glb: keep Inputs1 + Outputs, bypass Inputs2
        # Simba GLB is a data distributor, not an accumulator (unlike Eyeriss).
        # Large PE-level buffers (32KB weight, 3KB accu) handle reuse.
        # Only pin batch dims; let mapper decide output/reduction tiling.
        # (Matches projection code which uses Factors(["B=1"]).)
        glb = spec.architecture.find("shared_glb")
        glb.constraints.dataspace["keep"] = ProblemDataspaceList([ds_i, ds_o])
        glb.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w])
        glb.constraints.temporal["factors"] = Factors(["B=1", "H=1"])

        # PEInputBuffer (8KB): keep Inputs1
        # Pin out_ch=1 (inputs don't contain output-channel)
        pib = spec.architecture.find("PEInputBuffer")
        pib.constraints.dataspace["keep"] = ProblemDataspaceList([ds_i])
        pib.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w, ds_o])
        pib.constraints.temporal["factors"] = Factors([f"{out_ch}=1"])

        # PEWeightBuffer (32KB): keep Inputs2 — weight stationary
        # Conv ref perm: [P,Q,K,N] → Attn: [Q,out_ch,B,H, red(free)]
        # Output dims (Q) inner, batch (B,H) outer, weight dims (red,out_ch) free
        pwb = spec.architecture.find("PEWeightBuffer")
        pwb.constraints.dataspace["keep"] = ProblemDataspaceList([ds_w])
        pwb.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_i, ds_o])
        pwb.constraints.temporal["factors"] = Factors(["B=1", "H=1", "Q=1"])
        pwb.constraints.temporal["permutation"] = Permutation(
            ["Q", out_ch, "B", "H", red_dim]
        )

        # PEAccuBuffer (128 entries): keep Outputs — accumulation
        # Pin batch + reduction + spatial =1 (only out_ch free)
        # No permutation (consistent with projection; matches official repo)
        pab = spec.architecture.find("PEAccuBuffer")
        pab.constraints.dataspace["keep"] = ProblemDataspaceList([ds_o])
        pab.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w, ds_i])
        pab.constraints.temporal["factors"] = Factors(
            ["B=1", "H=1", f"{red_dim}=1", "Q=1"]
        )

        # InputPassthrough: keep Inputs1
        ipt = spec.architecture.find("InputPassthrough")
        ipt.constraints.dataspace["keep"] = ProblemDataspaceList([ds_i])
        ipt.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w, ds_o])

        # PEWeightRegs (depth=1): keep Inputs2
        pwr = spec.architecture.find("PEWeightRegs")
        pwr.constraints.dataspace["keep"] = ProblemDataspaceList([ds_w])
        pwr.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_i, ds_o])
        pwr.constraints.temporal["factors"] = Factors(all_ones)


def _apply_cnn_constraints(spec, arch: str, pe_x: int, pe_y: int, dims: dict):
    """Apply CNN-operator constraints via Approach C.

    Dims: [G, C, M, R, S, N, P, Q]
    Dataspaces:
        Weights = conv filters  (G, C, M, R, S)
        Inputs  = activations   (N, G*Cgroup, C, ...)
        Outputs = feature maps  (N, G*Mgroup, M, P, Q)  [read_write]

    Follows official Timeloop exercise example_designs constraints:
      - eyeriss_like  → Row Stationary
      - gemmini_like  → Output Stationary
      - simba_like    → Weight Stationary

    G is left unconstrained at spatial levels so the mapper can tile it
    for depthwise convolutions (G>1, C=1, M=1).
    """
    dim_order = ["G", "C", "M", "R", "S", "N", "P", "Q"]
    all_ones = [f"{d}=1" for d in dim_order]

    ds_w = "Weights"
    ds_i = "Inputs"
    ds_o = "Outputs"

    N = dims.get("N", 1)

    # ------------------------------------------------------------------
    if arch == "eyeriss_like":
        # Row Stationary — official eyeriss_like exercise constraints
        # with G added as unconstrained dimension.
        #
        # Official Conv7D:
        #   PE_column: perm [N,C,P,R,S,Q,M] factors N=1,C=1,P=1,R=1,S=1 split=999
        #   PE:        perm [N,P,Q,R,S,C,M] factors N=1,P=1,Q=1,R=1       split=0
        #   GLB:       keep [Inputs, Outputs], bypass [Weights]
        #   ifmap:     keep Inputs,  perm [N,M,C,P,Q,R,S], all=1
        #   wspad:     keep Weights, perm [N,M,P,Q,S,C,R], N=1,M=1,P=1,Q=1,S=1
        #   pspad:     keep Outputs, perm [N,C,P,Q,R,S,M], N=1,C=1,R=1,S=1,P=1,Q=1

        # PE_column (meshX): free M, Q, G → tiles M (standard) or G (depthwise)
        pe_col = spec.architecture.find("PE_column")
        pe_col.constraints.spatial["factors"] = Factors(
            ["N=1", "C=1", "P=1", "R=1", "S=1"]
        )
        pe_col.constraints.spatial["split"] = 999
        pe_col.constraints.spatial["permutation"] = Permutation(
            ["N", "G", "C", "P", "R", "S", "Q", "M"]
        )

        # PE (meshY): free C, M, S, G → tiles C (standard) or G (depthwise)
        pe = spec.architecture.find("PE")
        pe.constraints.spatial["factors"] = Factors(
            ["N=1", "P=1", "Q=1", "R=1"]
        )
        pe.constraints.spatial["split"] = 0
        pe.constraints.spatial["permutation"] = Permutation(
            ["N", "G", "P", "Q", "R", "S", "C", "M"]
        )

        # shared_glb: batch>1 → keep all (weight reuse); batch=1 → official
        glb = spec.architecture.find("shared_glb")
        if N > 1:
            glb.constraints.dataspace["keep"] = ProblemDataspaceList(
                [ds_w, ds_i, ds_o]
            )
            glb.constraints.dataspace["bypass"] = ProblemDataspaceList([])
            glb.constraints.temporal["factors"] = Factors([f"N={N}"])
        else:
            glb.constraints.dataspace["keep"] = ProblemDataspaceList(
                [ds_i, ds_o]
            )
            glb.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w])

        # ifmap_spad (12 entries): keep Inputs
        ifmap = spec.architecture.find("ifmap_spad")
        ifmap.constraints.dataspace["keep"] = ProblemDataspaceList([ds_i])
        ifmap.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w, ds_o])
        ifmap.constraints.temporal["factors"] = Factors(all_ones)
        ifmap.constraints.temporal["permutation"] = Permutation(
            ["N", "G", "M", "C", "P", "Q", "R", "S"]
        )

        # weights_spad (224 entries): keep Weights
        wspad = spec.architecture.find("weights_spad")
        wspad.constraints.dataspace["keep"] = ProblemDataspaceList([ds_w])
        wspad.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_i, ds_o])
        wspad.constraints.temporal["factors"] = Factors(
            ["N=1", "M=1", "P=1", "Q=1", "S=1"]
        )
        wspad.constraints.temporal["permutation"] = Permutation(
            ["N", "G", "M", "P", "Q", "S", "C", "R"]
        )

        # psum_spad (24 entries): keep Outputs
        pspad = spec.architecture.find("psum_spad")
        pspad.constraints.dataspace["keep"] = ProblemDataspaceList([ds_o])
        pspad.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w, ds_i])
        pspad.constraints.temporal["factors"] = Factors(
            ["N=1", "C=1", "R=1", "S=1", "P=1", "Q=1"]
        )
        pspad.constraints.temporal["permutation"] = Permutation(
            ["N", "G", "C", "P", "Q", "R", "S", "M"]
        )

    # ------------------------------------------------------------------
    elif arch == "gemmini_like":
        # Output Stationary — official simple_output_stationary exercise.
        #
        # Official:
        #   PE:     spatial perm [C,M] split=1, factors R=1,S=1,P=1,Q=1
        #   pe_spad: keep Outputs, temporal perm [R,S,P,Q]
        #   regs:   each keeps its dataspace, factors pinned to 1

        # PE_column (meshX): tile M/G along X
        pe_col = spec.architecture.find("PE_column")
        pe_col.constraints.spatial["factors"] = Factors(
            ["R=1", "S=1", "P=1", "Q=1", "N=1"]
        )
        pe_col.constraints.spatial["split"] = 999
        pe_col.constraints.spatial["permutation"] = Permutation(["C", "M"])

        # PE (meshY): tile C/G along Y
        pe = spec.architecture.find("PE")
        pe.constraints.spatial["factors"] = Factors(
            ["R=1", "S=1", "P=1", "Q=1", "N=1"]
        )
        pe.constraints.spatial["split"] = 0
        pe.constraints.spatial["permutation"] = Permutation(["C", "M"])

        # shared_glb: keep [Inputs, Outputs], bypass Weights
        glb = spec.architecture.find("shared_glb")
        glb.constraints.dataspace["keep"] = ProblemDataspaceList([ds_i, ds_o])
        glb.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w])
        glb.constraints.temporal["factors"] = Factors([])

        # pe_spad (256 entries): keep Outputs — output stationary
        pe_spad = spec.architecture.find("pe_spad")
        pe_spad.constraints.dataspace["keep"] = ProblemDataspaceList([ds_o])
        pe_spad.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w, ds_i])
        pe_spad.constraints.temporal["permutation"] = Permutation(
            ["R", "S", "P", "Q"]
        )
        if N > 1:
            pe_spad.constraints.temporal["factors"] = Factors([f"N={N}"])

        # Registers (depth=1): each keeps its dataspace
        for reg_name, keep_ds, bypass_ds, factors in [
            ("weight_reg", ds_w, [ds_i, ds_o],
             ["R=1", "S=1", "M=1", "C=1"]),
            ("input_activation_reg", ds_i, [ds_w, ds_o],
             ["P=1", "Q=1", "C=1", "N=1"]),
            ("output_activation_reg", ds_o, [ds_w, ds_i],
             ["P=1", "Q=1", "M=1", "N=1"]),
        ]:
            reg = spec.architecture.find(reg_name)
            reg.constraints.dataspace["keep"] = ProblemDataspaceList([keep_ds])
            reg.constraints.dataspace["bypass"] = ProblemDataspaceList(bypass_ds)
            reg.constraints.temporal["factors"] = Factors(factors)

    # ------------------------------------------------------------------
    elif arch == "simba_like":
        # Weight Stationary — official simba_like exercise constraints.
        #
        # Official:
        #   PE (meshX):          factors R=1,S=1,P=1,Q=1,N=1 perm [M,C,R,S,P,Q,N]
        #   distributed_buffers: factors P=1,Q=1,R=1,S=1,C=1,N=1 perm [M,C,Q,R,S,P,N]
        #   reg_mac:             factors P=1,Q=1,R=1,S=1,M=1,N=1 perm [C,M,Q,R,S,P,N]
        #   GLB:                 keep [Inputs, Outputs], bypass [Weights]
        #   PEInputBuffer:       keep [Inputs]
        #   PEWeightBuffer:      keep [Weights]
        #   PEAccuBuffer:        keep [Outputs]
        #   PEWeightRegs:        keep [Weights]

        # PE (meshX): free M, C, G
        pe_node = spec.architecture.find("PE")
        pe_node.constraints.spatial["factors"] = Factors(
            ["R=1", "S=1", "P=1", "Q=1", "N=1"]
        )
        pe_node.constraints.spatial["permutation"] = Permutation(
            ["M", "C", "R", "S", "P", "Q", "N"]
        )

        # distributed_buffers (meshY): free M, G
        dist_node = spec.architecture.find("distributed_buffers")
        dist_node.constraints.spatial["factors"] = Factors(
            ["P=1", "Q=1", "R=1", "S=1", "C=1", "N=1"]
        )
        dist_node.constraints.spatial["permutation"] = Permutation(
            ["M", "C", "Q", "R", "S", "P", "N"]
        )

        # reg_mac (meshY=4): free C, G
        reg_node = spec.architecture.find("reg_mac")
        reg_node.constraints.spatial["factors"] = Factors(
            ["P=1", "Q=1", "R=1", "S=1", "M=1", "N=1"]
        )
        reg_node.constraints.spatial["permutation"] = Permutation(
            ["C", "M", "Q", "R", "S", "P", "N"]
        )

        # shared_glb: keep [Inputs, Outputs], bypass [Weights]
        glb = spec.architecture.find("shared_glb")
        glb.constraints.dataspace["keep"] = ProblemDataspaceList([ds_i, ds_o])
        glb.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w])
        glb.constraints.temporal["factors"] = Factors(["N=1"])

        # PEInputBuffer (8KB): keep Inputs
        pib = spec.architecture.find("PEInputBuffer")
        pib.constraints.dataspace["keep"] = ProblemDataspaceList([ds_i])
        pib.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w, ds_o])
        pib.constraints.temporal["factors"] = Factors(["M=1"])

        # PEWeightBuffer (32KB): keep Weights — weight stationary
        pwb = spec.architecture.find("PEWeightBuffer")
        pwb.constraints.dataspace["keep"] = ProblemDataspaceList([ds_w])
        pwb.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_i, ds_o])

        # PEAccuBuffer (128 entries): keep Outputs
        pab = spec.architecture.find("PEAccuBuffer")
        pab.constraints.dataspace["keep"] = ProblemDataspaceList([ds_o])
        pab.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w, ds_i])
        if N > 1:
            pab.constraints.temporal["factors"] = Factors([f"N={N}"])

        # InputPassthrough: keep Inputs
        ipt = spec.architecture.find("InputPassthrough")
        ipt.constraints.dataspace["keep"] = ProblemDataspaceList([ds_i])
        ipt.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_w, ds_o])

        # PEWeightRegs (depth=1): keep Weights
        pwr = spec.architecture.find("PEWeightRegs")
        pwr.constraints.dataspace["keep"] = ProblemDataspaceList([ds_w])
        pwr.constraints.dataspace["bypass"] = ProblemDataspaceList([ds_i, ds_o])
        pwr.constraints.temporal["factors"] = Factors(all_ones)


def _apply_softmax_constraints(spec, pe_x: int, dims: dict, op_name: str):
    """Apply softmax-operator constraints on simple_vector (1D) architecture.

    Dims: [B, H, Q, K]
    Softmax sub-operations:
        softmax_max:     Inputs1(B,H,Q,K) -> Outputs(B,H,Q)       (reduce over K)
        softmax_sub_exp: Inputs1(B,H,Q,K), Inputs2(B,H,Q) -> Outputs(B,H,Q,K)
        softmax_sum:     Inputs1(B,H,Q,K) -> Outputs(B,H,Q)       (reduce over K)
        softmax_div:     Inputs1(B,H,Q,K), Inputs2(B,H,Q) -> Outputs(B,H,Q,K)

    Architecture: simple_vector (1D PE array, PE_column meshX only).
    Spatial: tile H along meshX (H is always independent across all softmax ops).
    """
    dim_order = ["B", "H", "Q", "K"]
    all_ones = [f"{d}=1" for d in dim_order]

    H = dims.get("H", 1)
    h_spatial = min(H, pe_x)

    # Determine dataspaces from op type
    is_two_input = "sub_exp" in op_name or "div" in op_name
    if is_two_input:
        inputs = ["Inputs1", "Inputs2"]
    else:
        inputs = ["Inputs1"]
    output = "Outputs"
    all_ds = inputs + [output]

    # ---- DRAM split ----
    dram_i = spec.architecture.find("DRAM_I")
    dram_i.constraints.dataspace["keep"] = ProblemDataspaceList(inputs)
    dram_i.constraints.dataspace["bypass"] = ProblemDataspaceList([output])

    dram_o = spec.architecture.find("DRAM_O")
    dram_o.constraints.dataspace["keep"] = ProblemDataspaceList([output])
    dram_o.constraints.dataspace["bypass"] = ProblemDataspaceList(inputs)

    try:
        dram_bk = spec.architecture.find("DRAM_Backup")
        dram_bk.constraints.dataspace["keep"] = ProblemDataspaceList([])
        dram_bk.constraints.dataspace["bypass"] = ProblemDataspaceList(all_ds)
    except Exception:
        pass

    # ---- shared_glb: keep all dataspaces ----
    glb = spec.architecture.find("shared_glb")
    glb.constraints.dataspace["keep"] = ProblemDataspaceList(all_ds)
    glb.constraints.dataspace["bypass"] = ProblemDataspaceList([])

    # ---- PE_column spatial: tile H along meshX ----
    pe_col = spec.architecture.find("PE_column")
    col_factors = list(all_ones)
    col_factors[1] = f"H={h_spatial}"   # dim_order[1] == "H"
    pe_col.constraints.spatial["factors"] = Factors(col_factors)
    pe_col.constraints.spatial["split"] = 999

    # ---- reg_file: keep all dataspaces, temporal factors all = 1 ----
    reg = spec.architecture.find("reg_file")
    reg.constraints.dataspace["keep"] = ProblemDataspaceList(all_ds)
    reg.constraints.dataspace["bypass"] = ProblemDataspaceList([])
    reg.constraints.temporal["factors"] = Factors(all_ones)


# ===================================================================
# Public API
# ===================================================================

def run_mapper(
    net: str,
    problem: str,
    batch_size: int,
    sequence_length: int = 1,
    mapper_idx: int = 0,
    fused_layer_type: str = "single",
    output_tile_channels: int = 1,
    tp_degree: int = 1,
    arch_target: str = "eyeriss_like",
    glb_scale: float = 1,
    pe_x_scale: float = 1,
    pe_y_scale: float = 1,
    dram_config: dict = {"I": "LPDDR5", "O": "LPDDR5"},
    output_base_dir: Optional[str] = None,
    remove_bw_limit: bool = True,
):
    """Run Timeloop mapper for LLM or CNN BF16 workloads.

    Uses Approach C: all constraints set in Python, no temp YAML files.
    Uses arch_bf.yaml. DRAM dataspace names are overridden in Python.

    Workload type is auto-detected from problem dimensions:
      - CNN: G in dims, B not in dims → _apply_cnn_constraints()
      - LLM attention: attn_qk/attn_v → _apply_attention_constraints()
      - LLM projection: everything else → _apply_projection_constraints()

    Args:
        net:               Network name (e.g. 'llama3.1_8b_prefill_s512'
                           or 'mobilenet_v3_small').
        problem:           Path to the workload YAML file.
        batch_size:        Batch size (multiplied into B for LLM, N for CNN).
        sequence_length:   Sequence length (default 1, for output path only).
        mapper_idx:        0 = energy-first, 1 = delay-first.
        fused_layer_type:  Fusion tag for output path (default 'single').
        output_tile_channels: Output tile channels (for output path).
        tp_degree:         Tensor-parallelism degree.
        arch_target:       Architecture name: eyeriss_like | simba_like | gemmini_like.
        glb_scale:         Scale factor for the global buffer depth (and banks/BW).
        pe_x_scale:        Scale factor for PE X dimension.
        pe_y_scale:        Scale factor for PE Y dimension.
        dram_config:       Dict with 'I' and 'O' keys mapping to DRAM type strings.
        output_base_dir:   Override base directory for outputs.
        remove_bw_limit:   If True, set DRAM bandwidth to near-infinite (1000x).

    Returns:
        (config_id, problem_id, result) or (None, None, None) on error.
    """
    try:
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                problem_name = os.path.basename(problem).split(".")[0]

                # ---- Softmax always uses simple_vector architecture ----
                if "softmax" in problem_name:
                    arch_target = "simple_vector"

                # ---- Jinja parse data (bf16 arch, no mapping file) ----
                jinja_parse_data = {
                    "architecture": arch_target,
                    "use_temp_arch": False,
                    "use_bf_arch": True,
                    "problem": problem,
                    "mapping_path": False,
                }

                # ---- Compute effective PE sizes ----
                bx, by = ARCH_BASE_PE.get(arch_target, (64, 64))
                pe_x = round(bx * pe_x_scale)
                pe_y = round(by * pe_y_scale)

                # ---- Read problem dimensions from YAML ----
                with open(problem, "r") as f:
                    problem_data = yaml.safe_load(f)
                problem_dims = dict(problem_data["problem"]["instance"])

                # ---- Detect workload type from problem dimensions ----
                is_cnn = "G" in problem_dims and "B" not in problem_dims
                is_softmax = "softmax" in problem_name
                is_attention = (
                    not is_cnn
                    and not is_softmax
                    and ("attn_qk" in problem_name or "attn_v" in problem_name)
                )
                is_projection = not is_cnn and not is_softmax and not is_attention

                # Strip layer-index prefix to get bare op name
                op_suffix = problem_name
                for prefix in ["layer0_", "layer1_", "layer2_"]:
                    if problem_name.startswith(prefix):
                        op_suffix = problem_name[len(prefix):]
                        break

                needs_rewrite = False

                if is_cnn:
                    # ---- CNN: batch into N, no TP ----
                    if batch_size > 1:
                        problem_dims["N"] = problem_dims.get("N", 1) * batch_size
                        needs_rewrite = True
                else:
                    # ---- LLM / ViT: TP and batch into B ----
                    tp_config = LLAMA_TP_CONFIG
                    if "qwen" in net.lower():
                        tp_config = QWEN_TP_CONFIG
                    elif "vit" in net.lower():
                        # ViT uses VIT_TP_CONFIG; tp_degree is always 1 for ViT
                        tp_config = VIT_TP_CONFIG
                    tp_dim = tp_config.get(op_suffix)
                    if tp_dim and tp_degree > 1 and tp_dim in problem_dims:
                        problem_dims[tp_dim] = problem_dims[tp_dim] // tp_degree
                    if batch_size > 1:
                        problem_dims["B"] = problem_dims.get("B", 1) * batch_size
                        needs_rewrite = True
                    if tp_degree > 1:
                        needs_rewrite = True

                # ---- Output directory ----
                base_dir = output_base_dir or THIS_SCRIPT_DIR
                proc_id = (
                    f"arch={arch_target}"
                    f"@glb_scale={glb_scale}"
                    f"@pe_x_scale={pe_x_scale}"
                    f"@pe_y_scale={pe_y_scale}"
                )
                dram_config_id = f"{dram_config['I']}@{dram_config['O']}"
                output_dir = (
                    f"{base_dir}/outputs/{net}/{problem_name}/{batch_size}/"
                    f"{sequence_length}/{mapper_idx}/{fused_layer_type}/"
                    f"{output_tile_channels}/{tp_degree}/{proc_id}/{dram_config_id}"
                )

                if os.path.exists(output_dir):
                    for fname in os.listdir(output_dir):
                        fpath = os.path.join(output_dir, fname)
                        if os.path.isfile(fpath):
                            os.remove(fpath)
                os.makedirs(output_dir, exist_ok=True)

                # ---- Rewrite problem file if dims changed ----
                if needs_rewrite:
                    problem_data["problem"]["instance"] = problem_dims
                    modified_problem = os.path.join(
                        output_dir, f"{problem_name}_modified.yaml"
                    )
                    with open(modified_problem, "w") as f:
                        yaml.dump(problem_data, f, default_flow_style=False)
                    jinja_parse_data["problem"] = modified_problem

                # ---- Load spec from Jinja template ----
                spec = Specification.from_yaml_files(
                    TOP_JINJA_PATH, jinja_parse_data=jinja_parse_data
                )
                spec.variables.technology = technology_node
                spec.variables.global_cycle_seconds = cycle_time

                # ---- Scale global buffer ----
                shared_glb = spec.architecture.find("shared_glb")
                shared_glb.attributes["depth"] = round(
                    shared_glb.attributes["depth"] * glb_scale
                )
                if glb_scale != 1.0:
                    if "n_banks" in shared_glb.attributes:
                        shared_glb.attributes["n_banks"] = round(
                            shared_glb.attributes["n_banks"] * glb_scale
                        )
                    if "shared_bandwidth" in shared_glb.attributes:
                        shared_glb.attributes["shared_bandwidth"] = round(
                            shared_glb.attributes["shared_bandwidth"] * glb_scale
                        )

                # ---- Scale PE array ----
                if arch_target in ("eyeriss_like", "gemmini_like"):
                    spec.architecture.find("PE_column").spatial.meshX = pe_x
                    spec.architecture.find("PE").spatial.meshY = pe_y
                elif arch_target == "simba_like":
                    spec.architecture.find("PE").spatial.meshX = pe_x
                    spec.architecture.find("distributed_buffers").spatial.meshY = pe_y
                elif arch_target == "simple_vector":
                    spec.architecture.find("PE_column").spatial.meshX = pe_x

                # ---- Apply mapping constraints (Approach C) ----
                if is_softmax:
                    _apply_softmax_constraints(
                        spec, pe_x, problem_dims, op_suffix
                    )
                elif is_cnn:
                    _apply_cnn_constraints(
                        spec, arch_target, pe_x, pe_y, problem_dims
                    )
                elif is_projection:
                    _apply_projection_constraints(
                        spec, arch_target, pe_x, pe_y, problem_dims
                    )
                else:
                    _apply_attention_constraints(
                        spec, arch_target, pe_x, pe_y, problem_dims, op_suffix
                    )

                # ---- Configure DRAM type and bandwidth ----
                if is_cnn:
                    ds_w, ds_i, ds_o = "Weights", "Inputs", "Outputs"
                elif is_softmax:
                    # Softmax: Inputs1 (+ optional Inputs2) and Outputs
                    is_two_input = "sub_exp" in op_suffix or "div" in op_suffix
                    ds_i = "Inputs1"
                    ds_w = "Inputs2" if is_two_input else "Inputs1"
                    ds_o = "Outputs"
                else:
                    ds_i, ds_w, ds_o = "Inputs1", "Inputs2", "Outputs"
                bw_multiplier = 1000 if remove_bw_limit else 1
                for dram_index in ["I", "O"]:
                    dram = spec.architecture.find(f"DRAM_{dram_index}")
                    dram_info = dram_type_bandwidth_width_dict[dram_config[dram_index]]
                    dram.attributes["type"] = dram_info["timeloop"]
                    dram.attributes["width"] = dram_info["width"]
                    dram.attributes["shared_bandwidth"] = (
                        dram_info["bandwidth"] * bw_multiplier * 8 / word_size / tp_degree
                    )

                # ---- DRAM dataspace constraints ----
                spec.architecture.find("DRAM_I")["constraints"]["dataspace"]["keep"] = (
                    ProblemDataspaceList([ds_i, ds_w])
                )
                spec.architecture.find("DRAM_I")["constraints"]["dataspace"]["bypass"] = (
                    ProblemDataspaceList([ds_o])
                )
                spec.architecture.find("DRAM_O")["constraints"]["dataspace"]["keep"] = (
                    ProblemDataspaceList([ds_o])
                )
                spec.architecture.find("DRAM_O")["constraints"]["dataspace"]["bypass"] = (
                    ProblemDataspaceList([ds_w, ds_i])
                )
                spec.architecture.find("DRAM_Backup")["constraints"]["dataspace"]["keep"] = (
                    ProblemDataspaceList([])
                )
                spec.architecture.find("DRAM_Backup")["constraints"]["dataspace"]["bypass"] = (
                    ProblemDataspaceList([ds_w, ds_i, ds_o])
                )

                # ---- Mapper configuration ----
                mapper = spec.mapper
                mapper["algorithm"] = timeloop_search_strategy
                mapper["timeout"] = timeloop_timeout
                mapper["victory_condition"] = timeloop_victory_condition
                mapper["num_threads"] = timeloop_num_threads
                mapper["diagnostics"] = False
                target = ["energy", "delay"]
                if mapper_idx == 1:
                    target = ["delay", "energy"]
                mapper["optimization_metrics"] = target

                # ---- Run mapper ----
                res = tl.call_mapper(
                    spec, output_dir=output_dir, dump_intermediate_to=output_dir
                )

                # ---- Cleanup output directory (keep only stats) ----
                utility_functions.clean_directory(output_dir)

                # ---- Build identifiers ----
                config_id = (
                    f"{arch_target}@glb{glb_scale}@pe_x_scale{pe_x_scale}"
                    f"@pe_y_scale{pe_y_scale}"
                    f"@{utility_functions.dram_config_to_id(dram_config)}"
                )
                problem_id = (
                    f"{net}@{problem_name}@{batch_size}@{sequence_length}"
                    f"@{mapper_idx}@{fused_layer_type}@{output_tile_channels}"
                    f"@{tp_degree}"
                )

                return config_id, problem_id, res

    except Exception as e:
        with open(os.path.join(THIS_SCRIPT_DIR, "timeloop_errors.log"), "a") as f:
            f.write(
                f"Error running config {net}, {problem}, "
                f"Batch:{batch_size}, Seq:{sequence_length} on {arch_target}: {e}\n"
            )
        return None, None, None
