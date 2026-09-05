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

## This repository vs. the Zenodo record

**Use this repository.** It is the current code.

The Zenodo record is the frozen snapshot submitted for artifact evaluation, plus the 4.6 GB
characterization database. Two things landed here *after* that snapshot was cut, and both change
results:

| Change | Effect if you use the Zenodo code instead |
|---|---|
| I-SAEO bootstrap fix in `src/scripts/run_archgym_chiplet.py` | The seeded pool could carry a duplicate PIM or switch chiplet. Figure 11's MoE EDP reads 2.41× instead of 1.1× |
| Converged pool chains in `src/archgym_results/ae_*_chain_converged/` | Figure 9's `poolVsUnconGap` reads 4.8% instead of the paper's 4.1% |

Zenodo remains the right source for the **database** — it is far too large for git. Everything else,
take from here.

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
tools/                download_data.sh, verify_determinism.py
docker/               analysis and Timeloop images
```

## License and attribution

GPLv3 — see `LICENSE`. Portions derive from arch-gym (Apache-2.0) and the Accelergy/CACTI
ecosystem; see `NOTICE`.

If you use Fengshui, please cite the MICRO 2026 paper — see `CITATION.cff`.
