#!/usr/bin/env python3
"""
V2 generalization figure (n=8 only): train-pool transferred onto held-out test.

3 held-out splits (MoE / Decode / Vision) x 4 configs = 12 bars, one figure.
Each bar = generalization ratio at n_chiplets=8:

    ratio = cross["8"] / test_own["8"]
          = (train-derived 8-chiplet pool's geomean on the test set)
            / (test set's OWN-optimal 8-chiplet pool geomean)

ratio = 1.0 (dashed) -> train pool matches the test-own ceiling. Log y.

Reads : <results-root>/<split>/<config>/summary.json
Writes: <output-dir>/figures/generalization_v2_cross.png/.pdf and
        <output-dir>/generalization_v2_cross.csv

With no arguments, ``results-root`` and ``output-dir`` retain the historical
in-tree locations, so existing notebook and command-line usage is unchanged.
"""
import argparse
import os, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default=os.path.join(ROOT, "results"),
        help="Result tree containing <split>/<config>/summary.json "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=ROOT,
        help="Output base directory; writes the CSV here and figures under "
             "<output-dir>/figures (default: %(default)s)",
    )
    return parser.parse_args()


args = parse_args()
RES = os.path.abspath(os.path.expanduser(args.results_root))
OUTDIR = os.path.abspath(os.path.expanduser(args.output_dir))
FIGDIR = os.path.join(OUTDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

SPLITS = [("moe_test_v2", "MoE (Qwen)\nheld-out"),
          ("decode_test_v2", "Decode\nheld-out"),
          ("vision_test_v2", "Vision\nheld-out"),
          ("prefill_test_v2", "Prefill\nheld-out"),
          ("seq2048_test_v2", "Seq-2048\nheld-out")]
CONFIGS = [("energy_nocost", "Energy", "#6baed6"),
           ("energy_cost", "Energy×$", "#2171b5"),
           ("edp_nocost", "EDP", "#fb6a4a"),
           ("edp_cost", "EDP×$", "#cb181d")]
N = "8"

# ---- gather ----
data = {}   # data[(split, config)] = ratio @ n=8
rows = []
for sk, _ in SPLITS:
    for ck, _, _ in CONFIGS:
        path = os.path.join(RES, sk, ck, "summary.json")
        if not os.path.exists(path):
            continue  # config still running (e.g. prefill energy_nocost)
        summ = json.load(open(path))
        cross, own = summ["cross"][N], summ["test_own"][N]
        r = cross / own
        data[(sk, ck)] = r
        rows.append(dict(split=sk, config=ck, cross=cross, test_own=own, ratio=r))

if not rows:
    raise FileNotFoundError(
        f"no summary.json files found under results root: {RES}"
    )

csv_out = os.path.join(OUTDIR, "generalization_v2_cross.csv")
with open(csv_out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# ---- plot ----
fig, ax = plt.subplots(figsize=(14, 5.5))
bw = 0.2
group_gap = 1.2
centers = []
seen_labels = set()
for si, (sk, slabel) in enumerate(SPLITS):
    base = si * group_gap
    for ci, (ck, clabel, color) in enumerate(CONFIGS):
        if (sk, ck) not in data:
            continue  # config still running -> no bar
        x = base + (ci - 1.5) * bw
        r = data[(sk, ck)]
        lab = clabel if clabel not in seen_labels else None
        seen_labels.add(clabel)
        ax.bar(x, r, bw, color=color, label=lab)
        ax.text(x, r * 1.03, f"{r:.2f}×", ha="center", va="bottom", fontsize=8)
    centers.append(base)

ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.8)
ax.text(centers[-1] + 0.4, 1.0, " optimum", va="center", ha="left",
        fontsize=9, color="gray")
ax.set_yscale("log")
ax.set_ylabel("Generalization ratio\n(train pool on test / test-own optimum)")
ax.set_xticks(centers)
ax.set_xticklabels([s for _, s in SPLITS], fontsize=11)
ax.set_title("Train-pool generalization to held-out test sets "
             "(8-chiplet pools, v2)", fontsize=12)
ax.legend(title="Objective", fontsize=9, ncol=4, loc="upper center",
          bbox_to_anchor=(0.5, -0.08), frameon=False)
ax.grid(axis="y", which="both", alpha=0.25)
plt.tight_layout()
for ext in ("png", "pdf"):
    out = os.path.join(FIGDIR, f"generalization_v2_cross.{ext}")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
print("wrote", csv_out)

print(f"\n{'split':16}{'config':14}{'cross/own @n=8':>16}")
for sk, _ in SPLITS:
    for ck, _, _ in CONFIGS:
        v = data.get((sk, ck))
        print(f"{sk:16}{ck:14}{'(running)' if v is None else f'{v:16.2f}'}")
