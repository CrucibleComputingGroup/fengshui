"""Analytical softmax on a compute chiplet's 1-D vector unit.

Replaces the ``simple_vector`` database lookup for non-PIM softmax. PIM softmax keeps its
database rows.

WHY NOT THE DATABASE ROWS
  The simple_vector softmax rows are a Timeloop mapping artifact: _apply_softmax_constraints
  (src/timeloop_experiments/timeloop_helper.py:790) lets only the heads use the vector lanes, so
  every row takes 2^22 cycles (4.19 ms for llama3.1-8B prefill) at any lane count, and each of
  the four sub-ops re-reads the scores from DRAM as 'weights'. The rows exist only at pe_y 1 and
  batch 1, where the evaluator did not look them up for most chiplets (see cal_perf_phy_net).

THE MODEL
  Four passes over rows held in the GLB -- max, sub_exp, sum, div: FuseMax StableSoftmax's
  einsums M, SN, SD and A -- each issuing one element per lane per cycle on dedicated pipelined
  function units. Rows are spread over 64 * pe_x lanes; only when there are fewer rows than
  lanes (decode) is a row split over s lanes, whose partials are combined through the GLB in a
  ceil(log2 s)-level tree. The layer is one roofline, the form of
  cal_perf_phy_net._apply_bw_contention:

      latency = max(sum_p max(compute_p, glb_port_p), dram_in, dram_out) * cycle_time

  Energy = function units + register file + GLB (ERT energy per access at the chiplet's
  glb_scale) + DRAM words x word_size x final_e. The evaluator adds the package crossing from
  i/w/o_access and scales by tp and batch, as for any row.

  The caller states where the scores S come from and where the probabilities P go
  (scores_in_dram, probs_to_dram); cal_perf_phy_net derives both from the stage position and
  the attn_qk / attn_v residency table (attention_residency.csv).

  Checked against 352 Timeloop runs of this schedule with the GLB as backing store (cycles
  exact; access counts and lanes exact at pe_x 1, 2, 4); the same constants with the database's
  mapping reproduce all 65,536 LLM simple_vector rows.

Every constant is required and cited beside its value. [A] marks a named modelling assumption
rather than a sourced number.

Sources:
  M       = the Mozart workspace the database was built in: chiplet_timeloop/ (the Timeloop sweep;
            its arch/ templates are not shipped here, see database_builder.py) and
            micro24-fusemax-artifact/ (the FuseMax artifact)
  ERT     = src/scripts/ert/simple_vector_glb<g>.ERT_summary.yaml: the Timeloop/Accelergy run of
            M/chiplet_timeloop/arch/simple_vector/arch_bf.yaml through
            src/timeloop_experiments/timeloop_helper.py, 14 nm, one file per glb_scale
  FMX_ERT = M/micro24-fusemax-artifact/workspace/inputs/yamls/area_energy/outputs-1d-flat/ERT.yaml
"""
import math

from global_parameter import cycle_time, glb_base_word, pe_x_base_size, word_size

# Clock, word and lanes come from global_parameter: cycle_time (the Timeloop global_cycle_seconds,
# timeloop_helper.py:1011), word_size (bf16; arch_bf.yaml datawidth 16) and pe_x_base_size (lanes
# per pe_x step; arch_bf.yaml meshX 64, timeloop_helper.py:1035-1036).

# ---------------------------------------------------------------- function units (dedicated, pipelined)
# [A] each pass issues one element per lane per cycle. FuseMax StableSoftmax sums one cycle per element
# per einsum: "Multi-cycle operations (exp, div) can be pipelined" (M/micro24-fusemax-artifact/workspace/
# src/accel/stable_softmax.py:125); parse_stats.add_vector_unit charges every lane for dedicated
# max/exp/div units; Timeloop and the database charge one lane-cycle per element per sub-op.
ISSUE_CYCLES_PER_ELEMENT = 1
ALU_BASE_PJ = 0.293341  # ERT mac.compute: bf16mac = bf16adder + bf16multiplier (M/chiplet_timeloop/arch/_components/bf16mac.yaml:20-26)
# Energy per operation relative to a MAC, from FuseMax's Accelergy ERT of its 1-D function units (FMX_ERT:
# max 0.028036, add 8.3395, exponentiatial 94.6475, divide 36.251153, mac 17.1645 pJ); the same ratios are
# M/chiplet_timeloop/flashattn/fusemax/accel/proposal.py:127-136. [A] the ratios transfer to this bf16 MAC.
OP_ENERGY_RATIO = {"max": 0.001633, "add": 0.486, "exp": 5.514, "divide": 2.112}
# Operations per element in each pass: FuseMax StableSoftmax compute_cost (stable_softmax.py:34-39):
# M {max}, SN {add, exponentiatial}, SD {add}, A {divide}.
# (name, ops per element, S-sized GLB reads, per-row GLB reads, S-sized GLB writes, per-row GLB writes, reduction op)
# max: read S, write m | sub_exp: read S and m, write E | sum: read E, write l | div: read E and l, write P
PASSES = (
    ("max", ("max",), 1, 0, 0, 1, "max"),
    ("sub_exp", ("add", "exp"), 1, 1, 1, 0, None),
    ("sum", ("add",), 1, 0, 0, 1, "add"),
    ("div", ("divide",), 1, 1, 1, 0, None),
)
# [A] a split row (decode) combines its s partials in a tree. arch_bf.yaml has no inter-lane network; the
# only level the lanes share is shared_glb, so each combine is 1 GLB write + 1 GLB read (the partner's
# partial) + 1 op, 2 RF reads + 2 RF writes, and one tree level takes these three steps in sequence.
# Timeloop charges nothing for it.
COMBINE_CYCLES_PER_LEVEL = 3

# ---------------------------------------------------------------- GLB geometry
# The GLB holds glb_base_word words per glb_scale (global_parameter; arch_bf.yaml shared_glb depth
# 1048576 x width 64 / datawidth 16).
GLB_PORT_WORDS_PER_CYCLE_PER_SCALE = 256  # arch_bf.yaml shared_bandwidth, x glb_scale (timeloop_helper.py:1022-1026);
                                          # reads, fills and updates share it (Timeloop shared_glb throttling)
GLB_BLOCK_WORDS = 4                       # arch_bf.yaml width 64 / datawidth 16; stats "Block size : 4"

# ---------------------------------------------------------------- energy per action, pJ (ERT)
RF_READ_PJ = 0.153201   # ERT reg_file.read (smartbuffer_RF, depth 8 x 16 b)
RF_WRITE_PJ = 0.147751  # ERT reg_file.write == update
GLB_READ_PJ_PER_BLOCK = {1: 32.1007, 4: 61.9732, 9: 93.2124, 16: 137.508}   # ERT shared_glb.read (CactiSRAM)
GLB_WRITE_PJ_PER_BLOCK = {1: 29.0396, 4: 58.9121, 9: 89.7824, 16: 131.492}  # ERT shared_glb.write == update
                                                                            # (M/chiplet_timeloop/arch/_components/smartbuffer_SRAM.yaml:51)


def _require(cond, msg):
    if not cond:
        raise ValueError(msg)


def lane_mapping(rows, k_len, lanes):
    """Rows over lanes; a row is split over s lanes only when rows < lanes (decode).
    Returns (s, cycles of one S-sized pass without the combine tree, combine-tree levels)."""
    s = max(1, lanes // rows)
    row_slots = lanes // s
    elem_cycles = math.ceil(rows / row_slots) * math.ceil(k_len / s) * ISSUE_CYCLES_PER_ELEMENT
    levels = math.ceil(math.log2(s)) if s > 1 else 0
    return s, elem_cycles, levels


def softmax_row(*, heads, q_len, k_len, batch_in_problem, tp, pe_x_scale, glb_scale,
                scores_in_dram, probs_to_dram, dram_i, dram_o, dram_table):
    """Per-chip, per-item softmax row for the evaluator (the meaning of a database row at batch 1).

    heads, q_len, k_len, batch_in_problem: the layer's problem instance H, Q, K, B
        (LayerConfig.problem_data['instance']); B must be 1, the evaluator scales by batch.
    tp: tensor-parallel degree; each chip holds heads / tp heads.
    pe_x_scale, glb_scale: the chiplet (64 * pe_x_scale lanes; an ERT exists for glb_scale 1, 4, 9, 16).
    scores_in_dram: S is read once from DRAM (dram_i) and filled into the GLB, instead of read
        from the GLB tile attn_qk left there.
    probs_to_dram: P is written once to DRAM (dram_o) and softmax books its GLB staging write;
        otherwise P stays in the GLB and attn_v books P's GLB write (its fill) and port time.
    dram_i, dram_o: the DRAM types at this layer's boundaries.
    dram_table: global_parameter.dram_type_bandwidth_width_dict ('bandwidth' GB/s, 'final_e' pJ/bit).

    The row carries dynamic_energy (J), latency (s) and i/w/o_access (words) and no
    static_power: the vector unit's leakage is the evaluator's (parse_stats.add_vector_unit).
    """
    _require(batch_in_problem == 1, f"softmax problem B must be 1 (the evaluator scales by batch), got {batch_in_problem}")
    _require(heads % tp == 0, f"H={heads} is not divisible by tp={tp}")
    _require(isinstance(scores_in_dram, bool) and isinstance(probs_to_dram, bool),
             f"scores_in_dram / probs_to_dram must be bool, got {scores_in_dram!r} / {probs_to_dram!r}")
    _require(glb_scale in GLB_READ_PJ_PER_BLOCK, f"no ERT for glb_scale {glb_scale}")
    _require(float(pe_x_scale).is_integer() and pe_x_scale >= 1, f"pe_x_scale must be a positive integer, got {pe_x_scale}")
    lanes = int(pe_x_base_size * pe_x_scale)
    rows = (heads // tp) * q_len
    n = rows * k_len
    s, elem_cycles, levels = lane_mapping(rows, k_len, lanes)
    if scores_in_dram:   # S streams from DRAM in row tiles: the tile of rows in flight must fit the GLB
        _require(min(rows, lanes // s) * k_len <= glb_scale * glb_base_word, "streamed S tile exceeds the GLB")

    port = GLB_PORT_WORDS_PER_CYCLE_PER_SCALE * glb_scale
    onchip_cycles = 0
    glb_reads = glb_writes = rf_reads = rf_writes = 0
    ops = {k: 0 for k in OP_ENERGY_RATIO}
    passes = {}
    for name, pass_ops, rn, rr, wn, wr, red in PASSES:
        combines = rows * (s - 1) if red else 0
        writes_s = wn if (name != "div" or probs_to_dram) else 0     # P's GLB write: here only if P goes to DRAM
        port_words = n * rn + rows * rr + n * writes_s + rows * wr + 2 * combines
        if name == "max" and scores_in_dram:
            port_words += n                                           # the DRAM -> GLB fill of S
        compute = elem_cycles + (levels * COMBINE_CYCLES_PER_LEVEL if red else 0)
        glb_cyc = math.ceil(port_words / port)
        cyc = max(compute, glb_cyc)
        onchip_cycles += cyc
        p_ops = {op: n for op in pass_ops}
        if red:
            p_ops[red] = p_ops.get(red, 0) + combines
            # Timeloop's counts with K innermost: I1 1 R + 1 W per element; O (N - R*s) R + N W;
            # each combine 2 R + 2 W
            p_rf_r, p_rf_w = 2 * n - rows * s + 2 * combines, 2 * n + 2 * combines
        else:
            # I1 1 R + 1 W per element; I2 N R + R*s W (m or l multicast to a row's s lanes); O N W
            p_rf_r, p_rf_w = 2 * n, 2 * n + rows * s
        p_glb_r = n * rn + rows * rr + combines
        p_glb_w = n * writes_s + rows * wr + combines
        for op, c in p_ops.items():
            ops[op] += c
        rf_reads += p_rf_r; rf_writes += p_rf_w; glb_reads += p_glb_r; glb_writes += p_glb_w
        passes[name] = dict(compute=compute, elem=elem_cycles, glb=glb_cyc, port_words=port_words, cycles=cyc,
                            combines=combines, ops=p_ops, rf_r=p_rf_r, rf_w=p_rf_w, glb_r=p_glb_r, glb_w=p_glb_w)
    if scores_in_dram:
        glb_writes += n                                               # the one fill of S

    words_in = n if scores_in_dram else 0
    words_out = n if probs_to_dram else 0
    # [A] DRAM transfers overlap the passes (row tiling); words per cycle as _apply_bw_contention
    bw_i = dram_table[dram_i]["bandwidth"] * 8 / word_size / tp
    bw_o = dram_table[dram_o]["bandwidth"] * 8 / word_size / tp
    cyc_in = math.ceil(words_in / bw_i) if words_in else 0
    cyc_out = math.ceil(words_out / bw_o) if words_out else 0
    cycles = max(onchip_cycles, cyc_in, cyc_out)

    e_glb_r = GLB_READ_PJ_PER_BLOCK[glb_scale] / GLB_BLOCK_WORDS
    e_glb_w = GLB_WRITE_PJ_PER_BLOCK[glb_scale] / GLB_BLOCK_WORDS
    e_alu = sum(cnt * OP_ENERGY_RATIO[op] * ALU_BASE_PJ for op, cnt in ops.items())
    e_rf = rf_reads * RF_READ_PJ + rf_writes * RF_WRITE_PJ
    e_glb = glb_reads * e_glb_r + glb_writes * e_glb_w
    e_dram = (words_in * word_size * dram_table[dram_i]["final_e"]
              + words_out * word_size * dram_table[dram_o]["final_e"])
    return {
        "latency": cycles * cycle_time,
        "dynamic_energy": (e_alu + e_rf + e_glb + e_dram) * 1e-12,
        "i_access": float(words_in),     # S is an activation; the evaluator's crossing reads i_access
        "w_access": 0.0,
        "o_access": float(words_out),
        # diagnostics (the evaluator does not read them)
        "_n": n, "_rows": rows, "_lanes": lanes, "_split": s,
        "_cycles": dict(onchip=onchip_cycles, dram_in=cyc_in, dram_out=cyc_out, passes=passes),
        "_pJ": dict(alu=e_alu, rf=e_rf, glb=e_glb, dram=e_dram),
        "_counts": dict(ops=ops, rf_r=rf_reads, rf_w=rf_writes, glb_r=glb_reads, glb_w=glb_writes),
    }
