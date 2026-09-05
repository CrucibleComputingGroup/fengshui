#!/usr/bin/env python3
"""Test softmax constraints on simple_vector architecture.

Randomly selects ~32 softmax workloads across different models, phases,
sequence lengths, and softmax op types, runs the Timeloop mapper,
and reports results.
"""
import sys
import os
import random
import time
import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in [os.path.join(_THIS_DIR, "..", "scripts"),
                   os.path.join(_THIS_DIR, "..")]:
    if (os.path.isfile(os.path.join(_candidate, "utility_functions.py"))
            and _candidate not in sys.path):
        sys.path.insert(0, _candidate)

# Ensure Timeloop shared libraries are on LD_LIBRARY_PATH (Docker env)
_tl_lib = "/workspace/accelergy-timeloop-infrastructure/src/timeloop/lib"
if os.path.isdir(_tl_lib):
    os.environ["LD_LIBRARY_PATH"] = _tl_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import timeloop_helper
from global_parameter import NET_DIR

# ============================================================
# Collect all softmax workload files
# ============================================================
all_softmax = []
for net_name in sorted(os.listdir(NET_DIR)):
    net_dir = os.path.join(NET_DIR, net_name)
    if not os.path.isdir(net_dir):
        continue
    for fname in sorted(os.listdir(net_dir)):
        if "softmax" in fname and fname.endswith(".yaml"):
            all_softmax.append((net_name, os.path.join(net_dir, fname)))

print(f"Found {len(all_softmax)} softmax workloads total.")

# ============================================================
# Randomly sample ~32 workloads (ensure all 4 op types covered)
# ============================================================
random.seed(42)
op_types = ["softmax_max", "softmax_sub_exp", "softmax_sum", "softmax_div"]

# Ensure at least 2 of each op type
selected = []
for op in op_types:
    candidates = [(n, p) for n, p in all_softmax if op in os.path.basename(p)]
    selected.extend(random.sample(candidates, min(4, len(candidates))))

# Fill remaining slots randomly
remaining = [x for x in all_softmax if x not in selected]
n_more = max(0, 32 - len(selected))
if remaining and n_more > 0:
    selected.extend(random.sample(remaining, min(n_more, len(remaining))))

# Deduplicate and shuffle
selected = list(dict.fromkeys(selected))
random.shuffle(selected)
print(f"Selected {len(selected)} workloads for testing.\n")

# ============================================================
# Run mapper on each selected workload
# ============================================================
output_base = os.path.join(_THIS_DIR, "test_softmax_outputs")
os.makedirs(output_base, exist_ok=True)

results = []
total = len(selected)

for idx, (net_name, problem_path) in enumerate(selected, 1):
    problem_name = os.path.basename(problem_path).replace(".yaml", "")

    # Read dims for display
    with open(problem_path) as f:
        pdata = yaml.safe_load(f)
    dims = pdata["problem"]["instance"]

    print(f"[{idx}/{total}] {net_name} / {problem_name}")
    print(f"         dims: {dims}")

    t0 = time.time()
    config_id, problem_id, res = timeloop_helper.run_mapper(
        net=net_name,
        problem=problem_path,
        batch_size=1,
        sequence_length=1,
        mapper_idx=0,
        arch_target="simple_vector",   # will be overridden inside for softmax
        output_base_dir=output_base,
        remove_bw_limit=True,
    )
    elapsed = time.time() - t0

    if res is None:
        print(f"         FAILED ({elapsed:.1f}s)")
        results.append({
            "net": net_name, "problem": problem_name,
            "dims": dims, "status": "FAILED", "elapsed": elapsed,
            "cycles": None, "energy": None, "utilization": None,
        })
    else:
        # Extract key stats
        stats = res[0] if isinstance(res, (list, tuple)) else res
        try:
            cycles = stats.cycles
            energy = stats.energy
            utilization = getattr(stats, 'utilization', None)
        except AttributeError:
            # Try dict-style access
            try:
                cycles = stats["cycles"]
                energy = stats["energy"]
                utilization = stats.get("utilization", None)
            except (TypeError, KeyError):
                cycles = "?"
                energy = "?"
                utilization = "?"

        print(f"         OK ({elapsed:.1f}s) cycles={cycles}, energy={energy}")
        results.append({
            "net": net_name, "problem": problem_name,
            "dims": dims, "status": "OK", "elapsed": elapsed,
            "cycles": cycles, "energy": energy, "utilization": utilization,
        })
    print()

# ============================================================
# Summary
# ============================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)
n_ok = sum(1 for r in results if r["status"] == "OK")
n_fail = sum(1 for r in results if r["status"] == "FAILED")
print(f"Total: {len(results)}, OK: {n_ok}, FAILED: {n_fail}")
print()

if n_ok > 0:
    print(f"{'Network':<40} {'Problem':<25} {'Dims':<30} {'Cycles':<12} {'Energy':<15}")
    print("-" * 122)
    for r in results:
        if r["status"] == "OK":
            dim_str = f"B={r['dims'].get('B','?')} H={r['dims'].get('H','?')} Q={r['dims'].get('Q','?')} K={r['dims'].get('K','?')}"
            print(f"{r['net']:<40} {r['problem']:<25} {dim_str:<30} {str(r['cycles']):<12} {str(r['energy']):<15}")

if n_fail > 0:
    print(f"\nFailed workloads:")
    for r in results:
        if r["status"] == "FAILED":
            print(f"  {r['net']} / {r['problem']}")
