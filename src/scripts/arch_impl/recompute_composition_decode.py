#!/usr/bin/env python3
"""
Recompute the §5.1 decode-specific composition numbers on the CORRECTED DB + v7 pools:
  - PIM benefit  : LLaMA-8B/70B batch-1 decode ENERGY on energy-chain N=1 (no PIM) vs N=2 (+PIM)
  - Switch benefit: Qwen3-30B/235B decode EDP on edp-chain N=2 (no switch) vs N=3 (+switch)
Run from scripts/ in conda `mozart`:  python3 arch_impl/recompute_composition_decode.py
"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/
import pandas as pd
from network_dataclass import VirtualNetwork
from chiplet_pool_env import ChipletConfig
from chiplet_sel import run_single_optimization, NET_DIR

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'unified_database.csv')
ARCHGYM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'archgym_results')
# Pool generation to read. 'ae' is the deterministic pool behind the camera-ready
# main results; 'v7' is the superseded generation this diagnostic originally used.
_CHAIN_VER = os.environ.get('FENGSHUI_CHAIN_VERSION', 'ae')


def pool(obj, n):
    f = sorted(glob.glob(f'{ARCHGYM}/{_CHAIN_VER}_{obj}_chain/chain_summary_*.json'))[-1]
    return [ChipletConfig.from_identifier(s) for s in json.load(open(f))[f'n{n}']['best_chiplets']]


def make_nets(specs):
    nets = [VirtualNetwork(name, batch_size=b, sequence_length=s) for (name, b, s) in specs]
    db = pd.read_csv(DB)
    layers = {net: set(db[db['net'] == net]['layer_name'].unique()) for net in db['net'].unique()}
    for vn in nets:
        vn.load_from_dir(os.path.join(NET_DIR, vn.network_name), db_layers=layers.get(vn.network_name))
    return [vn for vn in nets if len(vn.layers) > 0]


def val(nets, group, objective):
    _, res = run_single_optimization(nets, group, objective=objective, results_file=DB,
                                     cost_aware=False, use_dag_cp=True, v_het_batch=True)
    return {k: v['min_value'] for k, v in res.items()}


def main():
    # ---- PIM benefit: LLaMA batch-1 decode ENERGY, energy chain N=1 -> N=2 ----
    llama = make_nets([('llama3.1_8b_decode_kv1024', 1, 1), ('llama3.1_70b_decode_kv1024', 1, 1)])
    e_n1 = val(llama, pool('energy', 1), 'energy')
    e_n2 = val(llama, pool('energy', 2), 'energy')
    print("\n=== PIM benefit (batch-1 decode ENERGY, energy chain N=1 -> N=2) ===")
    for k in sorted(e_n1):
        drop = 100 * (1 - e_n2[k] / e_n1[k])
        print(f"  {k:42s}  N1={e_n1[k]:.3e}  N2={e_n2[k]:.3e}  drop={drop:5.1f}%")

    # ---- Switch benefit: Qwen decode EDP, edp chain N=2 -> N=3 (range over b1/b8) ----
    qwen = make_nets([('qwen3_30b_a3b_decode_kv1024', 1, 1), ('qwen3_30b_a3b_decode_kv1024', 8, 1),
                      ('qwen3_235b_a22b_decode_kv1024', 1, 1), ('qwen3_235b_a22b_decode_kv1024', 8, 1)])
    d_n2 = val(qwen, pool('edp', 2), 'edp')
    d_n3 = val(qwen, pool('edp', 3), 'edp')
    print("\n=== Switch benefit (decode EDP, edp chain N=2 -> N=3) ===")
    for k in sorted(d_n2):
        drop = 100 * (1 - d_n3[k] / d_n2[k])
        print(f"  {k:42s}  N2={d_n2[k]:.3e}  N3={d_n3[k]:.3e}  drop={drop:5.1f}%")


if __name__ == '__main__':
    main()
