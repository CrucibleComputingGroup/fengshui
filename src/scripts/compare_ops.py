import pandas as pd
import pickle

with open("../.db_cache/remap_unified_database.csv_4870553777_1775413477_70b.pkl", "rb") as f:
    db = pickle.load(f)

net = "llama3.1_70b_decode_kv1024"
bs, sl = 1, 1

gemmini = db[(db["net"]==net) & (db["batch_size"]==bs) & (db["sequence_length"]==sl) &
             (db["arch_target"]=="gemmini_like") & (db["glb_scale"]==1) &
             (db["pe_x_scale"]==3) & (db["pe_y_scale"]==4)]

pim = db[(db["net"]==net) & (db["batch_size"]==bs) & (db["sequence_length"]==sl) &
         (db["arch_target"]=="PIM")]

print("=== decode_kv1024: gemmini_like@3x4 vs PIM (per operator) ===")
hdr = f"{'layer':>20s}  {'gem_E':>10s}  {'gem_L':>10s}  {'gem_util':>8s}  {'pim_E':>10s}  {'pim_L':>10s}  {'pim_util':>8s}  {'E_rat':>6s}  {'L_rat':>6s}"
print(hdr)

layers = sorted(gemmini["layer_name"].unique())
for lname in layers:
    g = gemmini[gemmini["layer_name"]==lname]
    p = pim[pim["layer_name"]==lname]
    if g.empty or p.empty:
        continue
    gb = g.loc[g["dynamic_energy"].idxmin()]
    pb = p.loc[p["dynamic_energy"].idxmin()]
    er = float(gb["dynamic_energy"]) / float(pb["dynamic_energy"])
    lr = float(gb["latency"]) / float(pb["latency"])
    ge = f"{float(gb['dynamic_energy']):.3e}"
    gl = f"{float(gb['latency']):.3e}"
    gu = f"{float(gb['utilization']):.4f}"
    pe = f"{float(pb['dynamic_energy']):.3e}"
    pl = f"{float(pb['latency']):.3e}"
    pu = f"{float(pb['utilization']):.4f}"
    print(f"{lname:>20s}  {ge:>10s}  {gl:>10s}  {gu:>8s}  {pe:>10s}  {pl:>10s}  {pu:>8s}  {er:>5.1f}x  {lr:>5.1f}x")
