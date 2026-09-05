# Figure 9 cost-panel assets

`generate_paper_fig10.py` (paper **Figure 9**) reads the two JSON files here for the
cost-bearing panels (Energy x $ and EDP x $), where each design's cost is scaled by the
device count needed to reach iso-throughput:

| File | Contents | Consumed at |
|---|---|---|
| `homo_latency.json`   | best homogeneous (Gemini-style) design + its per-network value/latency | `generate_paper_fig10.py:600` |
| `pernet_latency.json` | best-per-network homogeneous design value/latency | `generate_paper_fig10.py:601` |

Both are **regenerable** from the shipped database — they are not hand-entered constants:

```bash
# from anywhere; needs src/unified_database.csv in place (~30 min each, CPU only)
python3 src/scripts/arch_impl/fig11_a2_assets/homo_latency.py
python3 src/scripts/arch_impl/fig11_a2_assets/pernet_latency.py
```

Each script sweeps all homogeneous candidate designs (eyeriss/simba/gemmini x GLB x PE
scales) with memory homogenized to GDDR7, and records the winning design's per-network
value and latency. Env overrides: `FENGSHUI_DB` (database path), `FENGSHUI_A2_OUT`
(output directory, default: this directory).

> Regenerating overwrites the shipped JSON in place. Back it up first if you want to
> diff shipped vs. regenerated.
