# Fengshui

**Demystifying Chiplet Ecosystem and Bespoke Neural Network Accelerator Codesign**\
MICRO 2026 · [Zenodo artifact 10.5281/zenodo.21524898](https://doi.org/10.5281/zenodo.21524898)

Fengshui jointly optimizes two things that are normally decided separately: **which chiplets to
build** (a reusable pool, amortizing NRE across applications) and **how to compose them** into a
network-specific accelerator for each workload. The two are circularly dependent — a pool's value
depends on the accelerators built from it, while accelerator quality is bounded by the available
chiplets — so Fengshui searches them together, over operator-level disaggregation, memory
heterogeneity, tensor fusion, and pipeline/tensor/expert parallelism.

This repository holds the code and results for the paper's Figure 8 (chiplet-pool size) and
Figure 9 (the pool against GPU, homogeneous and unconstrained designs).

---

## Quick start

```bash
git clone https://github.com/CrucibleComputingGroup/fengshui.git
cd fengshui

# Fetch the 4.6 GB database from Zenodo into src/ (~271 MB compressed download)
bash tools/download_data.sh

pip install -r requirements.txt        # or Docker: see docker/Dockerfile.analysis for the commands

jupyter lab notebooks/reproduce_all.ipynb
```

The database must end up at `src/unified_database.csv` (26.7 M rows, sha256 `a1f5849d…`). Run the
scripts from `src/scripts/`. `run_saeo_isaeo_chain.py` and `run_archgym_chiplet.py` read the database
as `../unified_database.csv` by default; `converge_chain.py` and `generate_paper_fig10.py
--recompute` take `--database`, and `run_convergence_all.sh` honours `FENGSHUI_DB`.

CPU only. No GPU and no commercial EDA license are needed — GPU measurements are pre-computed and
shipped in `src/gpu/benchmarks_bf16/`.

---

## What reproduces what

| Paper artifact | Script (`src/scripts/`) | Key inputs |
|---|---|---|
| Fig. 8 `fig:pool_size` | `generate_paper_fig6_chain.py` | `src/archgym_results/{ae_*_chain, v7_*_sa}/` |
| Fig. 8 pool searches | `run_saeo_isaeo_chain.py` (SAEO+I-SAEO); `run_archgym_chiplet.py --method incremental --algorithms sa --n-end 10` (SA); both `--n-evals 500 --seed 42` | database |
| Pool convergence endpoints | `run_convergence_all.sh` (writes `src/archgym_results/repro_*_chain_converged/`) | `src/archgym_results/ae_*_chain/` |
| Fig. 9 `fig:big_eval` | `generate_paper_fig10.py` (replot) | `src/archgym_results/figure9/` |
| Fig. 9 cells | `generate_paper_fig10.py --recompute --database ../unified_database.csv --workers 16` (22 min on 16 workers: about 5 CPU-hours, 48 GB of memory; exits 1 unless all 400 cells equal the shipped ones) | database, the one chain CSV in each `src/archgym_results/ae_*_chain_converged/`, `src/gpu/benchmarks_bf16/` |

The searches and `--recompute` read the pre-computed database, and the replots read only the
shipped results; no Timeloop run is required. The database build pipeline (Timeloop/Accelergy
sweeps) is not in this repository. An earlier version of its sweep drivers, parsers and
`network_analysis.csv` builders is in the Zenodo code archive (`fengshui-code.tar.gz`), and
comments here that cite those files refer to it; the Timeloop architecture templates (`arch/`)
that some comments cite are not in that archive.

---

## Changes since the MICRO 2026 results

The paper's numbers predate these changes. The shipped results (`src/archgym_results/`) come from
a re-run with them, with the pools re-searched.

- GQA KV-cache sizing
- Per-bit inter-chiplet communication
- CATCH cost model
- DDR5 energy charged once
- Package-crossing energy scaled with tp and batch
- Tensor sizes from `network_analysis.csv`
- Softmax on the vector unit
- V read in fused attention
- Off-path op priced at its batch
- Stage-boundary DRAM charged once
- MoE expert stages costed as their EP chiplets
- PIM priced as a whole CENT device (cost and area of 8 dies, full-device standby)
- MoE prefill expert work counted per expert: B·E database rows per layer instead of B·k (decode and
  PIM rows keep B·k)

Effect on the abstract's claims (geomean over the 20 networks of Figure 9, Chiplet Pool vs
homogeneous accelerators):

| Claim | Paper | Re-run (pools re-searched) |
|---|---|---|
| Energy reduction | 48.5% | 45.0% |
| Energy × $ reduction | 88.1% | 51.7% |
| EDP reduction | 93.0% | 95.4% |
| EDP × $ reduction | 97.8% | 98.0% |
| Gap to unconstrained designs (worst metric) | 4.1% | 4.5% |

---

## Layout

```
src/scripts/            performance model, pool search, Figure 8 and 9 scripts
src/archgym_results/    pool-search results (ae_*_chain, ae_*_chain_converged, v7_*_sa) and the
                        Figure 9 cells (figure9/)
src/workloads/          operator shapes of the 10 workloads (20 networks at batch 1 and 8)
src/gpu/                pre-computed bf16 GPU measurements
src/timeloop/           mem_spec.py (memory model, imported by the performance model)
src/timeloop_experiments/  compute_area.py (chiplet area, imported by the performance model)
notebooks/              reproduce_all.ipynb
tools/                  download_data.sh; build_attention_residency.py and
                        build_attn_compute_cycles.py (how two shipped tables were built from
                        the raw Timeloop outputs, which are not shipped)
docker/                 analysis image
LICENSES/               Apache-2.0 text (for the parts of get_cost.py ported from CATCH)
```

## License and attribution

GPLv3 — see `LICENSE`. `src/scripts/get_cost.py` ports parts of the CATCH cost model (Apache-2.0,
`LICENSES/Apache-2.0.txt`); the database was produced with Timeloop and Accelergy/CACTI. See `NOTICE`.

If you use Fengshui, please cite the MICRO 2026 paper (also in `CITATION.cff`):

```bibtex
@inproceedings{jin2026fengshui,
  title     = {Fengshui: Demystifying Chiplet Ecosystem and Bespoke Neural Network Accelerator Codesign},
  author    = {Jin, Haoran and Yang, Jirong and Zhang, Zhiheng and Shin, Justin and Lyu, Barry and
               Zhang, Kangqi and Liu, Yunpeng and Bleier, Nathan},
  booktitle = {Proceedings of the 59th IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  year      = {2026}
}
```
