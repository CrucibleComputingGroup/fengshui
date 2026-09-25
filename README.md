# Fengshui

**Demystifying Chiplet Ecosystem and Bespoke Neural Network Accelerator Codesign**
MICRO 2026 · [Zenodo artifact 10.5281/zenodo.21524898](https://doi.org/10.5281/zenodo.21524898)

Fengshui jointly optimizes two things that are normally decided separately: **which chiplets to
build** (a reusable pool, amortizing NRE across applications) and **how to compose them** into a
network-specific accelerator for each workload. The two are circularly dependent — a pool's value
depends on the accelerators built from it, while accelerator quality is bounded by the available
chiplets — so Fengshui searches them together, over operator-level disaggregation, memory
heterogeneity, tensor fusion, and pipeline/tensor/expert parallelism, with place-and-route
validation for physical implementability.

---

## Quick start

```bash
git clone https://github.com/CrucibleComputingGroup/fengshui.git
cd fengshui

# Fetch the 4.6 GB database from Zenodo into src/ (~271 MB compressed download)
bash tools/download_data.sh

# One environment, analysis stack + gym
pip install -r requirements.txt        # or: docker build -f docker/Dockerfile.analysis -t fengshui .

jupyter lab notebooks/reproduce_all.ipynb
```

The database must end up at `src/unified_database.csv` (26.7 M rows, sha256 `a1f5849d…`). Scripts
under `src/scripts/` resolve it as `../unified_database.csv`; every driver also honours a
`FENGSHUI_DB` environment override.

CPU only. No GPU and no commercial EDA license are needed — GPU measurements are pre-computed and
shipped in `src/gpu/benchmarks_bf16/`.

---

## What reproduces what

| Paper artifact | Script | Key inputs |
|---|---|---|
| Fig. 8 `fig:pool_size` | `src/scripts/generate_paper_fig6_chain.py` | `src/archgym_results/{ae_*_chain, v7_*_sa}/` |
| Fig. 9 `fig:big_eval` + `constants.tex` | `src/scripts/generate_paper_fig10.py` | `ae_*_chain_converged/`, `arch_impl/competing_ae_raw.csv`, `arch_impl/fig11_a2_assets/`, `src/gpu/benchmarks_bf16/` |
| Fig. 10 `fig:nre_cost` | `src/cost_model/cost_model/generate_nre_cost_figure.py` | `fig10_areas.json`, `fig10_power.json` |
| Fig. 11 `fig:generalization` | `src/train_test/` — see its `README.md` | `corrected_v2_runs/`, `finetune_runs/` |
| Table III `tab:ablation` | `src/scripts/arch_impl/generate_ablation_heatmap.py` | `ablation_factorial_ae_summary.csv` |
| Table V `tab:competitors` | `src/scripts/arch_impl/build_competitor_table.py` | `competing_ae_raw.csv` |
| Pool convergence endpoints | `src/scripts/run_convergence_all.sh` | `ae_*_chain/` |

Everything reads the pre-computed database; no Timeloop run is required. Regenerating the database
from raw Timeloop sweeps takes about a month and is out of scope here.

---

## Determinism

The inner mapping GA is deterministic. `deterministic_ga_rng()` in `src/scripts/chiplet_sel.py`
seeds it per network from a stable hash of the network name, so results do not depend on worker
count or call order. Set `FENGSHUI_DETERMINISTIC=1` (the default) and optionally `FENGSHUI_GA_SEED`.

The **outer** pool search is not bit-reproducible. Re-running a chain reproduces the training pool
and its cross-evaluation exactly, but the held-out own-optimum search can move ~0.3% in the I-SAEO
phase (n > `--saeo-end`). That is an order of magnitude below the precision any reported number
carries, but it means chain values are not expected to match to the last digit.

`tools/verify_determinism.py` checks the inner-GA guarantee.

## Pool size is a result, not a parameter

Figure 9's "unconstrained" reference is the point where growing the shared pool stops paying —
marginal improvement below 1% for two consecutive steps. That endpoint differs per objective:

```
$ bash src/scripts/run_convergence_all.sh --assess-only     # ~6 s, no heavy compute

energy       N=10   already converged (4 trailing sub-1% steps)
energy_cost  N=12   required extension from N=10
edp          N=10   already converged (3 trailing sub-1% steps)
edp_cost     N=13   required extension from N=10
```

The shipped `ae_*_chain_converged/` directories record these endpoints and the full marginal curve.
Re-deriving the two that needed extending takes roughly four hours; `--assess-only` reports the
status of all four either way.

## Grouped-query attention: KV cache sized by KV heads

The attention workloads (`layer0_attn_qk.yaml`, `layer0_attn_v.yaml`) give K and V the query-head
count H, because a Timeloop problem has a single H. For grouped-query-attention models the KV cache
therefore came out g = `num_attention_heads / num_key_value_heads` times too large: the DRAM reads
of K (attn_qk) and V (attn_v), their DRAM energy, the DRAM-bandwidth bound of those rows' latency,
and the KV capacity that sizes each fusion group's DRAM. FLOPs were right.

| model | g | source |
|---|---|---|
| llama3.1-8B | 32 / 8 = 4 | `NETWORK.yaml`, HF `meta-llama/Llama-3.1-8B` config.json |
| llama3.1-70B | 64 / 8 = 8 | `NETWORK.yaml`, HF `meta-llama/Llama-3.1-70B` config.json |
| qwen3-30B-A3B | 32 / 4 = 8 | `NETWORK.yaml`, HF `Qwen/Qwen3-30B-A3B` config.json |
| qwen3-235B-A22B | 64 / 4 = 16 | `NETWORK.yaml`, HF `Qwen/Qwen3-235B-A22B` config.json |
| ViT-B/16, L/16, H/14 | 1 (multi-head attention) | `NETWORK.yaml` |
| OPT, CNNs, other `network_analysis.csv` entries | 1 | `KV_GROUP_WITHOUT_NETWORK_YAML` in `src/scripts/gqa_kv.py` |

`src/scripts/gqa_kv.py` corrects this in the evaluator, once, where `cal_perf_phy_net` builds its
row dicts, so every search and figure script that prices through the evaluator sees it. In each
non-PIM attn_qk / attn_v row that reads K or V from DRAM, the KV words (`i_access`) are divided by
g, the matching DRAM energy is removed, and the latency is re-derived with postprocess_bw's
roofline, max(compute-only cycles, DRAM bound), from the reduced words. The compute-only cycles
are shipped in `src/scripts/attn_compute_cycles.csv`, which `tools/build_attn_compute_cycles.py`
extracts from the Timeloop stats and checks against every attention row of the database. The KV
capacity (`weight_mem` of the same ops in `network_analysis.csv`) is divided by g. Nothing is
optional: a network without a g, a missing table, a missing table key, or an attention op of a GQA
network that is not one of the two corrected ops raises.

Not changed: PIM rows (the CENT model gives lumped latency and energy with no DRAM access counts),
every row of a network with g = 1, and the database's attention rows in the middle or at the end of
a fusion group. For those, the fusion split (`parse_stats.py`) zeroes `i_access`, which for these
ops is the K / V tensor, so the rows carry no KV read and there is nothing to rescale. Physically no
producer in the fusion group holds the KV cache on chip, so the evaluator does not use attn_v's
middle and end rows as they are: `_apply_attention_rows` in `cal_perf_phy_net.py` rebuilds them from
attn_v's corrected single row, keeping its V read, now sized with `num_key_value_heads`. attn_qk's
middle and end rows still read no K; they occur only on the linear path, since on the DAG-CP path
attn_qk always opens its stage. The effects below were measured at the GQA correction, before that
rebuild; restoring the V read then, as a diagnostic, moved the Fengshui (Full) decode EDP change
from −50.79% to −49.01% (b1) and from −9.14% to −6.09% (b8).

Scripts that read database rows directly instead of pricing through the evaluator do not see the
correction: `remap.py`, `run_remap_cross_eval_v2.py`, `compare_ops.py`, `generate_pnr_config.py`,
`reoptimize_pnr_config.py`, `chiplet_pruning.py` (only with `--prune-pct` > 0) and
`arch_impl/c6_*.py`. Of these, `arch_impl/c6_op_walkthrough.py` and `arch_impl/c6_weighted.py` also
price softmax from the database's `simple_vector` rows, a Timeloop mapping artifact the evaluator no
longer reads (`src/scripts/softmax_vector.py`). None of them is used by
`notebooks/reproduce_all.ipynb`.

**The published MICRO 2026 numbers and the shipped results (`archgym_results/`, `arch_impl/*.csv`)
were produced before this correction.** Re-running the evaluator now gives lower (better) EDP and
energy for the GQA models, mostly in decode, so live re-runs no longer match the shipped CSVs for
those workloads. `notebooks/reproduce_all.ipynb` therefore no longer compares its live re-runs
(cells 10 and 13) against the shipped CSVs.

For llama3.1-8B, the correction changes the published comparison (cost-unaware) by (+ = worse,
− = better; each column is relative to the same code without the correction, same framework):

| cell | EDP, Fengshui (Full) | EDP, Gemini-style | energy, Fengshui (Full) | energy, Gemini-style |
|---|---|---|---|---|
| prefill b1 | −0.32% | −0.06% | −0.07% | −0.08% |
| prefill b8 | −0.06% | −0.06% | −0.07% | −0.07% |
| decode b1 | −50.79% | −0.50% | −12.95% | −1.02% |
| decode b8 | −9.14% | −2.61% | −10.51% | −5.64% |

Fengshui gains more than the Gemini-style baseline, so the published llama3.1-8B decode comparisons
understate Fengshui's advantage. Over the 20-net suite, with the published pools, the correction
moves Fengshui (Full)'s geomean by −3.80% (energy), −7.71% (EDP), −7.80% (energy × cost) and
−9.09% (EDP × cost); it does not change the CNN workloads. Relative to the published values, a
re-run also includes two earlier model changes: inter-chiplet communication charged per bit
(+0.30% energy, +0.31% EDP on this geomean) and the CATCH cost-model port (−12.5% energy × cost,
−11.0% EDP × cost). Net of all three, the geomean is −3.51%, −7.42%, −19.12% and −18.80% against
the published values.

---

## Layout

```
src/scripts/          core model, search, and figure generators
src/train_test/       held-out generalization study (Fig. 11)
src/cost_model/       NRE / cost model (Fig. 10)
src/archgym_results/  pool-search chains: ae_*_chain, ae_*_chain_converged, v7_*_sa
src/workloads/        42 workload operator-shape descriptions
src/gpu/              pre-computed bf16 GPU measurements
src/timeloop*/        database build pipeline (not needed to reproduce figures)
notebooks/            reproduce_all.ipynb
tools/                download_data.sh, verify_determinism.py, build_attn_compute_cycles.py,
                      build_attention_residency.py
docker/               analysis and Timeloop images
```

## License and attribution

GPLv3 — see `LICENSE`. Portions derive from arch-gym (Apache-2.0) and the Accelergy/CACTI
ecosystem; see `NOTICE`.

If you use Fengshui, please cite the MICRO 2026 paper — see `CITATION.cff`.
