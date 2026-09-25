#!/usr/bin/env python3
"""ViT per-operator energy/latency breakdown + PIM-vs-conventional analysis.

Answers two rebuttal questions for the ViT case study (Fig 14 / W7):
  (1) How much of ViT's per-frame energy goes to softmax (and the whole
      attention pipeline qk->softmax->av)?
  (2) Why does near-bank PIM give ~zero benefit for ViT?

Method (REUSES the framework so numbers match the committed av_vision_results.csv
by construction; no hand-rolled energy model):
  * Reproduce fig14's homo selection (best single non-PIM chiplet from the v5
    energy pool, summed across the 4 vision workloads) by capturing the genes
    that generate_paper_fig14._run_ga_single produces.
  * Decompose the homo-optimal gene per fusion group: per-group energy is
    func.evaluate(latency) where func is the chosen convex-hull Function and
    latency is the network operating point (cal_opt_val_fused sums exactly these,
    so sum == reported total). Validate against the committed CSV.
  * Per-op clean breakdown via the no-fusion gene ('1'*num_layers): each group is
    one op.
  * PIM: re-decompose the no-fusion net on (a) the homo chiplet @GDDR7 and
    (b) the PIM chiplet @GDDR7, plus the full hetero pool @GDDR7, to see whether
    PIM is ever the per-op winner and by how much it loses.

Read-only: does not modify the paper or any DB.
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, SCRIPTS)
os.chdir(SCRIPTS)

# fig14 imports seaborn at module top (only used by the plotting fn we never call).
import types
if 'seaborn' not in sys.modules:
    _sns = types.ModuleType('seaborn')
    _sns.set_palette = lambda *a, **k: None
    _sns.barplot = lambda *a, **k: None
    sys.modules['seaborn'] = _sns
import generate_paper_fig14 as fig14
import global_parameter as gp
from chiplet_dataclass import ChipletConfig
from network_dataclass import VirtualNetwork, create_physical_network_from_gene
from genetic_algo_opt_phy_net import evaluate_gene
import cal_perf_phy_net as cpn
import pandas as pd

COMMITTED = {  # av_vision_results.csv, cost_aware=False (energy)
    'ViT-L/16': 5.159076e-03,
    'ViT-H/14': 1.035523e-02,
}
# NOTE: fig14 ALL_WORKLOADS loads ViT with sequence_length=1 (the s197/s257 patch
# count is baked into the layer YAML dims + DB rows; the DB lookup key uses seq=1).
VIT_WORKLOADS = {'ViT-L/16': ('vit_l16_s197', 1), 'ViT-H/14': ('vit_h14_s257', 1)}

# ------------------------------------------------------------------ helpers
def _build_chiplet_inputs(pool, net_name, db_file):
    """Replicate the GA's chiplets_data construction."""
    cdata = []
    for c in pool:
        cdata.append(cpn.get_chiplet_data(db_file, c.arch_target,
                                      c.global_buffer_size_scale,
                                      c.pe_x_scale, c.pe_y_scale, net_name))
    return cdata

def decompose(vn, pool, gene, db_file, cost_aware=False):
    """Return (total, latency, [(op_names, energy, latency_x1, chiplet_id), ...])."""
    cdata = _build_chiplet_inputs(pool, vn.network_name, db_file)
    g, fitness, cfg, err = evaluate_gene(gene, vn, pool, cdata, db_file,
                                         'energy', False, cost_aware)
    if cfg is None:
        return float('inf'), None, [], err
    phys = create_physical_network_from_gene(vn, gene)
    lat = cfg['latency']
    funcs = cfg['functions']
    groups = phys.fusion_groups
    rows = []
    if len(funcs) != len(groups):
        # group(s) dropped (no valid data) -> report mismatch; map by index up to min
        pass
    n = min(len(funcs), len(groups))
    for i in range(n):
        ops = [l.name for l in groups[i].layers]
        e = funcs[i].evaluate(lat)
        rows.append((ops, e, funcs[i].x1, funcs[i].id))
    return fitness, lat, rows, (len(funcs), len(groups))

def nofusion_gene(num_layers, buffer_config):
    return {'binary_string': '1' * num_layers, 'buffer_config': list(buffer_config)}

def classify(op):
    if 'softmax' in op: return 'softmax'
    if 'attn_qk' in op: return 'attn_qk'
    if 'attn_v' in op:  return 'attn_v'
    if op.endswith('q_proj') or op.endswith('k_proj') or op.endswith('v_proj') or op.endswith('o_proj'):
        return 'proj_qkvo'
    if 'gate_proj' in op or 'down_proj' in op or 'up_proj' in op: return 'mlp'
    if 'lm_head' in op: return 'lm_head'
    return 'other:' + op

# physical multiplicities in the real model (framework materializes q_proj once;
# attn/softmax once; mlp once; lm_head once). True ViT block: q,k,v,o = 4 projs.
TRUE_MULT = {'proj_qkvo': 4, 'attn_qk': 1, 'softmax': 1, 'attn_v': 1, 'mlp': 1, 'lm_head': 1}

# ------------------------------------------------------------------ main
print("Loading v5 energy pool (fig14 constants) ...")
energy_csv = fig14._find_latest_csv(fig14.V5_ENERGY_DIR)
print("  pool csv:", energy_csv)
pool_energy = ChipletConfig.from_csv_for_n_chiplets(8, energy_csv)
print("  pool:", [c.get_identifier() for c in pool_energy])
pim_chip = next((c for c in pool_energy if c.arch_target == 'PIM'), None)
print("  PIM in pool:", pim_chip.get_identifier() if pim_chip else None)

# Capture every gene produced during fig14's evaluation
records = []
_orig_run = fig14._run_ga_single
def _capture(vn, chiplet_list, results_file, cost_aware, **kw):
    val, gene = _orig_run(vn, chiplet_list, results_file, cost_aware, **kw)
    records.append({'net': vn.network_name,
                    'chips': tuple(c.get_identifier() for c in chiplet_list),
                    'val': val, 'gene': gene, 'cost_aware': cost_aware})
    return val, gene
fig14._run_ga_single = _capture

print("\nRunning fig14.evaluate_all_workloads (energy, cost_aware=False) ...")
results = fig14.evaluate_all_workloads(pool_energy, False, "Energy")
print("\nfig14 reported per-workload:")
for r in results:
    print(f"   {r['network']}: homo={r['homo_energy']:.6e} hetero={r['hetero_energy']:.6e}")

# Identify the homo-best chiplet = single-chiplet candidate minimizing summed val over the 4 workloads
homo_recs = [r for r in records if len(r['chips']) == 1 and r['val'] != float('inf')]
from collections import defaultdict
tot = defaultdict(float); count = defaultdict(int)
for r in homo_recs:
    tot[r['chips'][0]] += r['val']; count[r['chips'][0]] += 1
# only chiplets evaluated on all 4 workloads
nwl = max(count.values())
homo_best_id = min((c for c in tot if count[c] == nwl), key=lambda c: tot[c])
print(f"\nHomo-best chiplet (min summed energy over {nwl} workloads): {homo_best_id}")
homo_best = next(c for c in pool_energy if c.get_identifier() == homo_best_id)

db_file = fig14._build_database_for_workload('vit_l16_s197', True)

for disp, (net_name, seq) in VIT_WORKLOADS.items():
    print("\n" + "=" * 92)
    print(f"### {disp}  ({net_name}, seq={seq})   committed homo energy = {COMMITTED[disp]:.6e}")
    print("=" * 92)
    _db = pd.read_csv(db_file)
    db_layers = set(_db[_db['net'] == net_name]['layer_name'].unique())
    vn = VirtualNetwork(net_name, batch_size=1, sequence_length=seq)
    vn.load_from_dir(os.path.join(fig14.NET_DIR, net_name), db_layers=db_layers)
    num_layers = len(vn.layers)
    print(f"materialized layers ({num_layers}): {[l.name for l in vn.layers]}")

    # homo-optimal gene for THIS workload on the homo-best chiplet
    homo_gene = None
    for r in homo_recs:
        if r['chips'] == (homo_best_id,) and r['net'] == net_name:
            homo_gene = r['gene']; homo_val = r['val']; break
    print(f"homo gene buffer_config: {homo_gene['buffer_config']}")
    print(f"homo gene binary_string: {homo_gene['binary_string']}")

    # (A) faithful decomposition of the homo-optimal gene (validates total)
    tot_e, lat, rows, lens = decompose(vn, [homo_best], homo_gene, db_file)
    ssum = sum(e for _, e, _, _ in rows)
    print(f"\n[A] homo-optimal gene: total={tot_e:.6e}  sum(per-group)={ssum:.6e}  "
          f"committed={COMMITTED[disp]:.6e}  (funcs,groups)={lens}")
    print(f"    match committed: {abs(tot_e-COMMITTED[disp])/COMMITTED[disp]*100:.3f}% diff")

    # (B) clean per-op via no-fusion gene (same DRAM choices as homo gene)
    nf = nofusion_gene(num_layers, homo_gene['buffer_config'])
    tot_nf, lat_nf, rows_nf, lens_nf = decompose(vn, [homo_best], nf, db_file)
    print(f"\n[B] no-fusion gene: total={tot_nf:.6e}  (funcs,groups)={lens_nf}  "
          f"vs homo-opt {tot_e:.6e} ({(tot_nf/tot_e-1)*100:+.2f}%)")
    # aggregate by class
    agg_e = defaultdict(float); agg_l = defaultdict(float)
    for ops, e, x1, cid in rows_nf:
        cls = classify(ops[0]) if len(ops) == 1 else '+'.join(classify(o) for o in ops)
        agg_e[cls] += e; agg_l[cls] += x1
    Etot = sum(agg_e.values()); Ltot = sum(agg_l.values())
    if Etot == 0:
        print(f"    !! no per-op data (decompose failed): {lens_nf}")
        continue
    print(f"    per-op (framework-counted, q_proj once): energy total={Etot:.6e}")
    print(f"    {'op-class':<12} {'energy':>12} {'E%':>7} {'lat(x1)':>12} {'lat%':>7}")
    for cls in sorted(agg_e, key=lambda c: -agg_e[c]):
        print(f"    {cls:<12} {agg_e[cls]:>12.4e} {agg_e[cls]/Etot*100:>6.2f}% "
              f"{agg_l[cls]:>12.4e} {agg_l[cls]/Ltot*100:>6.2f}%")
    # true-model-weighted (q/k/v/o x4)
    twe = {c: agg_e[c] * TRUE_MULT.get(c, 1) for c in agg_e}
    twl = {c: agg_l[c] * TRUE_MULT.get(c, 1) for c in agg_l}
    TWE = sum(twe.values()); TWL = sum(twl.values())
    print(f"    --- true-model-weighted (q/k/v/o x4): energy total={TWE:.6e} ---")
    for cls in sorted(twe, key=lambda c: -twe[c]):
        print(f"    {cls:<12} {twe[cls]:>12.4e} {twe[cls]/TWE*100:>6.2f}% "
              f"{twl[cls]:>12.4e} {twl[cls]/TWL*100:>6.2f}%")
    # headline shares (true-model-weighted)
    def share(d, T, *cls): return sum(d.get(c, 0) for c in cls) / T * 100
    print(f"    >> softmax E% = {share(twe,TWE,'softmax'):.2f}% (fw {share(agg_e,Etot,'softmax'):.2f}%)")
    print(f"    >> attention-pipeline (qk+softmax+av) E% = "
          f"{share(twe,TWE,'attn_qk','softmax','attn_v'):.2f}% (fw {share(agg_e,Etot,'attn_qk','softmax','attn_v'):.2f}%)")
    print(f"    >> projections q/k/v/o E% = {share(twe,TWE,'proj_qkvo'):.2f}%   "
          f"MLP E% = {share(twe,TWE,'mlp'):.2f}%")

    # (C) PIM analysis: no-fusion @ all-GDDR7 on homo chiplet vs PIM vs full pool
    g7 = ['GDDR7'] * (num_layers + 1)
    nf7 = nofusion_gene(num_layers, g7)
    _, _, rows_conv, _ = decompose(vn, [homo_best], nf7, db_file)
    _, _, rows_pim, lens_pim = decompose(vn, [pim_chip], nf7, db_file)
    _, _, rows_full, _ = decompose(vn, list(pool_energy), nf7, db_file)
    conv_e = {classify(ops[0]): e for ops, e, _, _ in rows_conv}
    pim_e = {classify(ops[0]): e for ops, e, _, _ in rows_pim}
    full_win = {classify(ops[0]): cid for ops, e, _, cid in rows_full}
    print(f"\n[C] PIM vs conventional, per op (all @GDDR7, no fusion). "
          f"PIM feasible (funcs,groups)={lens_pim}")
    print(f"    {'op-class':<12} {'conv@GDDR7':>12} {'PIM@GDDR7':>12} {'PIM/conv':>9}  winner-in-full-pool")
    sum_conv = sum_pim = 0.0
    for cls in sorted(conv_e, key=lambda c: -conv_e.get(c, 0)):
        ce = conv_e.get(cls, float('nan')); pe = pim_e.get(cls, float('nan'))
        ratio = pe / ce if ce else float('nan')
        win = full_win.get(cls, '?').split('@')[0]
        sum_conv += (ce if ce == ce else 0); sum_pim += (pe if pe == pe else 0)
        print(f"    {cls:<12} {ce:>12.4e} {pe:>12.4e} {ratio:>8.2f}x  {win}")
    print(f"    TOTAL all-conv@GDDR7={sum_conv:.6e}  all-PIM@GDDR7={sum_pim:.6e}  "
          f"PIM/conv={sum_pim/sum_conv:.2f}x")
    pim_wins = sum(1 for c, cid in full_win.items() if cid.startswith('PIM@'))
    print(f"    PIM selected for {pim_wins}/{len(full_win)} ops by the full-pool optimizer")

    # check hetero winner gene for this workload: does it ever pick PIM?
    het_recs = [r for r in records if len(r['chips']) > 1 and r['net'] == net_name and r['val'] != float('inf')]
    if het_recs:
        best_het = min(het_recs, key=lambda r: r['val'])
        _, _, rows_het, _ = decompose(vn, list(pool_energy), best_het['gene'], db_file)
        het_chips = set(cid.split('@')[0] for _, _, _, cid in rows_het)
        print(f"    hetero-optimal gene (val={best_het['val']:.6e}) uses chiplets: {het_chips}")

print("\nDONE.")
