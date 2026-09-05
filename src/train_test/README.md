# Held-out generalization study (train/test) — paper **Figure 11**

Does a chiplet pool designed on one set of workloads generalize to **unseen** workloads?
This experiment splits the workloads (excluding **efficientnet_b0, gpt, and resnet**;
**ViT is included**) into a **train** and a **test** set, runs the full co-design pool search on each,
and then evaluates the **train-derived pool on the test set** with the pool held fixed
(a shipped tape-out / ASIC — only the per-workload inner mapping is re-optimized). A
`generalization_ratio` near 1.0 means the train-derived pool is about as good on unseen
workloads as a pool designed on them directly; >1.0 quantifies the loss from not having
seen the test set.

Each `(workload, batch)` pair (batch ∈ {1, 8}) is one virtual network, matching the
framework's default. This reuses the framework's own optimizer (`run_single_optimization`),
so it stays in lock-step with the rest of the artifact — no logic is re-implemented.

The evaluated architecture space contains Eyeriss-like, Simba-like, and
Gemmini-like compute chiplets, plus PIM and the network switch. **FEATHER is
excluded.** Both the driver and the full-run orchestrator fail immediately if
FEATHER appears in the candidate space or a fixed Fengshui pool.

> **Tier:** like the pool search (notebook T2), this needs the database **and** `gym`
> (it imports `run_archgym_chiplet`). Run it in the analysis environment with the DB in
> place (`bash ../../tools/download_data.sh`, or the notebook's fetch step).

## Which split does Figure 11 use?

**Figure 11 is *not* a random 70/30 split.** It is five **held-out-family** splits: for each
family, every workload of that family is the test set and everything else is the train set.
`plot_v2_cross.py` declares exactly these five result-directory names:

| Figure-11 group | `--split-mode` | test set (held out) | results directory the plotter reads |
|---|---|---|---|
| MoE (Qwen) held-out | `moe_test` | `qwen*` (MoE) | `results/moe_test_v2/` |
| Decode held-out | `decode_test` | all `*decode*` | `results/decode_test_v2/` |
| Vision held-out | `vision_test` | `mobilenet*`, `replknet*`, `vit*` | `results/vision_test_v2/` |
| Prefill held-out | `prefill_test` | all `*prefill*` | `results/prefill_test_v2/` |
| Seq-2048 held-out | `seq2048_test` | all `*s2048*` / `*kv2048*` | `results/seq2048_test_v2/` |

⚠️ `--split-mode` **defaults to `random`** (`run_train_test.py:312`). A command without
`--split-mode` produces a random 70/30 split, which is a *different* experiment and yields
**nothing that appears in Figure 11**. The `_v2` suffix on the directory names is a naming
convention of the shipped results only — it is *not* part of the `--split-mode` value, and
the driver does not create it by itself. The `run_all_v2.sh` orchestrator creates the
expected layout.

## Run

Run from this `src/train_test/` directory. Use `run_all_v2.sh` for both smoke and full
recomputation; it pins the author-recorded `_v2` settings, verifies the corrected database
SHA-256 and all four AE pool directories, schedules jobs without 20-way oversubscription, and
consolidates every output into an isolated run directory.

Set the database and pool roots explicitly in the AE package:

```bash
export FENGSHUI_DB="$(realpath ../unified_database.csv)"
export FENGSHUI_ARCHGYM="$(realpath ../archgym_results)"
```

### Validate inputs without launching a search

```bash
bash run_all_v2.sh --validate-only
```

This checks the runner and Python syntax, the four `ae_*_chain` pool directories, and the
corrected database hash. Hashing the 4.9 GB database is performed once per invocation.

### Held-out `_v2` smoke test — setup validation, not a numerical Figure-11 reproduction

```bash
bash run_all_v2.sh --smoke
```

The smoke path uses the full author-recorded workload protocol: ViT remains included,
sequences 1024/4096 are dropped, MoE is held out (26 train / 16 test virtual networks),
pruning is disabled, and the seed and batch sizes are pinned. It deliberately reduces the
search to `energy_nocost`, n=1..2, and 8 evaluations, so it validates train → test-own →
cross-evaluation plumbing but does not reproduce the n=8 bars. Smoke mode skips plotting.
The notebook runs this mode by default.

### Full corrected-database run

```bash
bash run_all_v2.sh
```

With no option, the orchestrator runs all five held-out splits and four objective configurations
using SAEO(n=1..6) → I-SAEO(n=7..8), 500 evaluations, 2,000 surrogate candidates, no pruning,
sequences 1024/4096 dropped, seed 42, and batches 1/8. It runs one split at a time and launches
that split's four configurations in parallel; launching all 20 combinations together is unsafe
because each configuration already forks many workers.

The notebook keeps the full run opt-in: set `FENGSHUI_RUN_FULL_FIG11=1` before starting its
kernel. The same notebook cell then invokes `run_all_v2.sh` with no option. Optional runner
controls include `FENGSHUI_V2_OUTPUT_ROOT`, `FENGSHUI_V2_RUN_ID`, and `FENGSHUI_PYTHON`.

The shipped `summary.json` files were generated with the pre-DRAM-fix database and legacy
stochastic inner GA. New runs use the corrected database and deterministic inner GA, so fresh
absolute values are expected to differ. Treat the corrected run as the AE result; do not expect
stale absolute values to match.

### Isolated consolidation and plotting

Every invocation writes under `corrected_v2_runs/<run-id>/` by default:

```text
corrected_v2_runs/<run-id>/
  raw/                         timestamped driver outputs
  results/<family>_test_v2/   complete consolidated split/config trees
  logs/                        one log per split/config (+ plot log)
  plot/                        full-run Figure 11 and CSV
  run_manifest.txt             inputs, hash, flags, and source directories
  SHA256SUMS                   hashes of consolidated results and plots
```

The runner checks every required sweep/summary file, verifies that `split.json` is identical
across the four configurations, preserves each complete config directory, and builds the combined
`summary_all.json`. It never copies into or overwrites the shipped `results/` tree.

A full run generates the plot automatically. To replot any completed run into another isolated
location, use the plotter's explicit paths:

```bash
RUN=corrected_v2_runs/<run-id>
python3 plot_v2_cross.py --results-root "$RUN/results" --output-dir "$RUN/replot"
```

## Outputs

### What each raw driver job writes (`corrected_v2_runs/<run-id>/raw/.../<timestamp>/`)

| File | Meaning |
|---|---|
| `split.json` | train/test virtual-network split + seed + workload list |
| `<config>/train_sweep_phase1_saeo.csv`, `<config>/train_sweep_phase2_isaeo.csv` | framework result on the **train** set (per n), one file per chain phase (with `--algorithm` ≠ `chain` it is a single `train_sweep.csv`) |
| `<config>/test_sweep_phase{1,2}_*.csv` | same for the **test** set (its own optimum) |
| `<config>/cross_eval.csv` | **train pool evaluated on test set** + `generalization_ratio = cross / test_own`, plus per-workload columns |
| `<config>/summary.json` | per-config recap: `train`, `test_own`, `cross`, `fengshui_cross` (8-point series over n=1..8) |
| `summary_all.json` | recap across all configs run, including `split` and the optimizer `args` |

`<config>` ∈ `energy_nocost, energy_cost, edp_nocost, edp_cost`.

### What is **shipped** in this bundle

`results/` contains **exactly 20 files** — `results/<family>_test_v2/<config>/summary.json`
for the 5 families × 4 configs, and nothing else. The per-run `split.json`, the
`*_sweep_phase*.csv` sweeps, `cross_eval.csv` and `summary_all.json` were **deliberately
stripped** to keep the code bundle small (the sweep CSVs carry a row per candidate pool and
dominate the tarball); do not go hunting for them — a fresh run regenerates all of them.

Consequence, stated plainly: each shipped `summary.json` holds only the four 8-point series
(`train`, `test_own`, `cross`, and `mozart_cross` — the older on-disk name for what the
current code writes as `fengshui_cross`). It records **no seed, no split membership, no pool
identity and no workload list**, so the 640 numbers behind Figure 11 carry no in-file
provenance. The author-recorded settings are documented above. A fresh `run_all_v2.sh` invocation
records its exact database hash, flags, paths, and source timestamps in `run_manifest.txt` and
retains each regenerated `split.json` for direct inspection.

Other shipped artifacts: `generalization_v2_cross.csv`, `generalization_combined.csv`,
`figures/*.pdf`, `ratio_random_split.pdf`, `finetune_results/finetune_master_table.csv`.

## Plotting

`plot_v2_cross.py` accepts explicit input and output roots. Use them to keep replots separate
from both shipped data and corrected reruns.

**Replot the shipped Figure-11 summaries without overwriting them:**

```bash
mkdir -p replots/shipped
python3 plot_v2_cross.py --results-root "$PWD/results" --output-dir "$PWD/replots/shipped"
```

This writes `replots/shipped/generalization_v2_cross.csv` and
`replots/shipped/figures/generalization_v2_cross.{png,pdf}`. A full `run_all_v2.sh` invocation
already writes the corresponding corrected-data artifacts under its isolated `plot/` directory;
use the command in the previous section to create additional replots.

`plot_v2_finetune.py` remains a shipped-data supplement: it reads
`finetune_results/finetune_master_table.csv` and writes under `figures/`. It is not part of the
corrected `_v2` orchestrator.

**Need outputs a full run produces — these do NOT work on the shipped bundle:**

| Script | Missing input | How to enable |
|---|---|---|
| `plot_combined.py` | `fengshui_cross.json` (not shipped) **and** `results/{random_split,decode_test,moe_test}/<config>/summary.json` (note: no `_v2` suffix) | run the random split plus the decode/moe splits into those directory names, then generate `fengshui_cross.json` with `eval_fengshui_pools.py` |
| `plot_ratio.py` | `results/random_split/<config>/{test_sweep_phase2_isaeo.csv, cross_eval.csv}` (sweep CSVs are stripped from the bundle) | run the **random** split (`--split-mode random`, the default) into `results/random_split/` and keep its CSVs; the shipped `ratio_random_split.pdf` is its output |

Both fail immediately with `FileNotFoundError` on the shipped tree — that is expected, not a
bug in the bundle.

## Extras (secondary)

- `run_finetune.py` / `summarize_finetune.py` — the **fine-tune** variant (re-optimize the
  train pool a little on the test set, n=8 → n=9) and its summary. Its shipped output
  `finetune_results/finetune_master_table.csv` is what `plot_v2_finetune.py` plots.
- `run_moe_vision.sh`, `run_prefill_seq.sh` — historical wrappers that produced the shipped
  results. They are retained for provenance; use `run_all_v2.sh` for new AE runs because it
  validates inputs, schedules safely, and consolidates the complete result tree automatically.
- `eval_fengshui_pools.py` — evaluate the paper's shipped `../archgym_results/ae_*_chain`
  pools on a split's test set; writes `fengshui_cross.json` (needed by `plot_combined.py`).
  Stage the `split.json` files under `$SPLITS_DIR`, default `./splits`.
