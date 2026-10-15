"""The 20 networks of Figures 8 and 9 (4 LLMs x prefill/decode x batch 1/8, and 2 CNNs at
batch 1/8), loaded with the database for the Figure 9 evaluation (generate_paper_fig10.py)."""
import os

from global_parameter import NET_DIR
from network_dataclass import VirtualNetwork
from cal_perf_phy_net import preload_database


def _build_virtual_nets(database_file="unified_database.csv"):
    """Build virtual networks for baseline evaluation.
    Matches setup_virtual_nets in run_archgym_chiplet.py (b1/b8, include CNN)."""
    virtual_nets = []
    for model in ['llama3.1_8b', 'llama3.1_70b', 'qwen3_30b_a3b', 'qwen3_235b_a22b']:
        virtual_nets.append(VirtualNetwork(f"{model}_prefill_s1024", batch_size=1, sequence_length=1024))
        virtual_nets.append(VirtualNetwork(f"{model}_prefill_s1024", batch_size=8, sequence_length=1024))
        virtual_nets.append(VirtualNetwork(f"{model}_decode_kv1024", batch_size=1, sequence_length=1))
        virtual_nets.append(VirtualNetwork(f"{model}_decode_kv1024", batch_size=8, sequence_length=1))
    for cnn in ['mobilenet_v3_small', 'replknet31b']:
        virtual_nets.append(VirtualNetwork(cnn, batch_size=1, sequence_length=1))
        virtual_nets.append(VirtualNetwork(cnn, batch_size=8, sequence_length=1))

    import pandas as pd
    _db = pd.read_csv(database_file)
    _db_layers_per_net = {
        net: set(_db[_db['net'] == net]['layer_name'].unique())
        for net in _db['net'].unique()
    }
    del _db   # free the whole table before preload_database reads the file again (lower peak memory)
    for vn in virtual_nets:
        db_layers = _db_layers_per_net.get(vn.network_name, None)
        try:
            vn.load_from_dir(os.path.join(NET_DIR, vn.network_name), db_layers=db_layers)
        except Exception as e:
            print(f"Warning: failed to load {vn.network_name}: {e}")
    virtual_nets = [vn for vn in virtual_nets if len(vn.layers) > 0]
    needed_nets = set(vn.network_name for vn in virtual_nets)
    preload_database(database_file, needed_nets=needed_nets)
    return virtual_nets
