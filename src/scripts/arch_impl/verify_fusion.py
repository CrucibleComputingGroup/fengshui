import os, sys
THIS=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(THIS,".."))
import ablation_study as A
from chiplet_sel import run_single_optimization
from cal_perf_phy_net import preload_database
import global_parameter as gp
from network_dataclass import VirtualNetwork
import pandas as pd
DB=A.DEFAULT_DB; NET="mobilenet_v3_small"
_db=pd.read_csv(DB); dl=set(_db[_db['net']==NET]['layer_name'].unique())
vn=VirtualNetwork(NET,batch_size=1,sequence_length=1); vn.load_from_dir(os.path.join(gp.NET_DIR,NET),db_layers=dl)
preload_database(DB,needed_nets={NET})
pool,_,_=A.load_pool("energy",False)
print(f"Net={vn.get_unique_name()} ({len(vn.layers)} layers)\n")
for label,key in [("Full Fengshui","full"),("w/o Layer Fusion","no_fusion")]:
    with A.toggle(key):
        _,res=run_single_optimization(virtual_nets=[vn],chiplet_group=pool,objective="energy",
            results_file=DB,cost_aware=False,use_sequential=True,n_workers=1,use_dag_cp=True,v_het_batch=True)
    g=res[vn.get_unique_name()]["best_gene"]; e=res[vn.get_unique_name()]["min_value"]
    bs=g.get("binary_string"); ngroups=bs.count("1"); nfused=bs.count("0")
    print(f"### {label}")
    print(f"    energy        = {e:.4e}")
    print(f"    binary_string = {bs}")
    print(f"    #groups(1s)={ngroups}  #fused-in(0s)={nfused}   {'<-- FUSION ACTIVE' if nfused>0 else '<-- no fusion (all separate)'}\n")
