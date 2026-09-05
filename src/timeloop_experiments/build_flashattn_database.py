#!/usr/bin/env python3
r"""Parse-only builder for the FlashAttn (FuseMax) performance CSV.

Turns the EXISTING Timeloop results under ``flashattn_outputs/`` into the
rich-schema FlashAttn CSV **without re-running Timeloop or Accelergy**.  The
full sweep (``run_flashattn_sweep.py``) took 28.8 h; all 147,456 mapper stats
files and 24,576 ``energy_estimation.yaml`` files are already on disk, so the
only work left is parsing + the DRAM post-processing math.

How it works
------------
Reuses ``run_flashattn_sweep.run_one_config`` *verbatim* (identical FuseMax
waterfall + ``postprocess_dram`` over the 16 DRAM combos), but monkeypatches the
TWO Timeloop/Accelergy *execution* entry points into parse-only no-ops:

  * ``Cascade.run_mapper``           -> skip ``tl.call_mapper``; require stats.txt
  * ``Cascade.run_accelergy_energy`` -> skip the accelergy subprocess; require
                                        ``{2d,1d}/energy_estimation.yaml``

Everything downstream (``collect_mem_traffic``, ``collect_latency``,
``collect_mem_traffic_split``, the ``eval()`` waterfall, ``read_energy``,
``postprocess_dram``) only READS on-disk files, so the emitted rows are
byte-identical to what a successful real sweep would have written.

Run INSIDE Docker ``my_timeloop_env_v2`` (needs ``timeloopfe`` + ``src.utils.stats``):

  docker exec my_timeloop_env_v2 bash -c \
    "export LD_LIBRARY_PATH=/workspace/accelergy-timeloop-infrastructure/src/timeloop/lib:\$LD_LIBRARY_PATH && \
     cd /workspace/chiplet_timeloop/timeloop_experiments && \
     python3 build_flashattn_database.py --n-jobs 32 --output-csv flashattn_database.csv"

  # quick subset check first:
  python3 build_flashattn_database.py --models llama --archs gemmini_like --limit 8 \
      --n-jobs 4 --output-csv flashattn_smoke.csv
"""
import sys
import os
import argparse
import math
import time
from pathlib import Path

import joblib  # noqa: E402

# Import the proven sweep module — this also wires up sys.path for
# global_parameter and installs the 'src' -> fusemax alias.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import run_flashattn_sweep as fa  # noqa: E402


# ---------------------------------------------------------------------------
# Parse-only monkeypatches (idempotent; re-applied inside each joblib worker)
# ---------------------------------------------------------------------------

def _apply_parse_only_patches():
    """Neutralise the Timeloop/Accelergy execution calls so the existing
    on-disk results are parsed instead of recomputed.

    Idempotent and safe to call in every worker process (joblib loky workers
    do NOT inherit monkeypatches from the parent), so the worker wrapper calls
    this before every config.

    IMPORTANT — class identity: ``proposal.py`` does
    ``from src.accel.cascade import Cascade``, so ``Proposal`` inherits from
    ``src.accel.cascade.Cascade``.  Patching ``accel.cascade.Cascade`` (a
    *different* module object for the same file) silently MISSES and the real
    mapper runs.  We therefore patch the methods directly on the leaf
    ``Proposal`` class that ``run_one_config`` actually instantiates — MRO then
    guarantees the no-ops win regardless of the base-module path."""
    fa._ensure_src_module()
    from accel.proposal import Proposal

    def _noop_run_mapper(self, einsum, output_dir, arch_yaml=None,
                         spec_callback=None, mapper_timeout=300,
                         mapper_victory=100):
        # output_dir is the per-einsum dir; the mapper stats must already exist.
        stats = Path(output_dir) / "timeloop-mapper.stats.txt"
        if not stats.is_file() or stats.stat().st_size == 0:
            raise FileNotFoundError(
                f"parse-only: missing/empty mapper stats {stats}")
        # Skip tl.call_mapper — the stats are read later by collect_*().
        return

    def _noop_run_accelergy_energy(self, output_dir, spec_callback=None):
        # output_dir is the per-config dir; energy was already estimated.
        for sub in ("2d", "1d"):
            ee = Path(output_dir) / sub / "energy_estimation.yaml"
            if not ee.is_file():
                raise FileNotFoundError(
                    f"parse-only: missing energy_estimation {ee}")
        # Skip the accelergy subprocess — read_energy() reads the YAMLs above.
        return

    Proposal.run_mapper = _noop_run_mapper
    Proposal.run_accelergy_energy = _noop_run_accelergy_energy

    # Safety net: make ANY stray Timeloop/Accelergy execution fail LOUD (the
    # config is then dropped by run_one_config's try/except) instead of
    # silently re-running the 28.8 h sweep and overwriting the on-disk results.
    def _blocked(*_a, **_k):
        raise RuntimeError(
            "parse-only safety net: Timeloop/Accelergy execution is blocked")

    for _mod_name in ("timeloopfe.v4", "pytimeloop.timeloopfe.v4"):
        _m = sys.modules.get(_mod_name)
        if _m is not None:
            for _fn in ("call_mapper", "call_model", "call_accelergy_verbose"):
                if hasattr(_m, _fn):
                    setattr(_m, _fn, _blocked)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def parse_one_config(fa_dir, wl, arch, px, py, glb, output_base_dir,
                     target_drams, tp, td, mapper_timeout, mapper_victory):
    """Patch (in this process) then reuse run_one_config to parse one config."""
    _apply_parse_only_patches()
    return fa.run_one_config(
        fa_dir, wl, arch, px, py, glb,
        output_base_dir, target_drams,
        mapper_timeout=mapper_timeout,
        mapper_victory=mapper_victory,
        tp_degree=tp, tp_dim=td,
    )


# ---------------------------------------------------------------------------
# Row validity filter
# ---------------------------------------------------------------------------

_POSITIVE_FIELDS = ("latency_s", "inf_bw_latency_s", "dynamic_energy_pj",
                    "inf_bw_energy_pj", "traffic")


def _row_is_valid(row):
    for k in _POSITIVE_FIELDS:
        v = row.get(k, None)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(v) or v <= 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Parse existing flashattn_outputs/ into the FlashAttn CSV "
                    "(no Timeloop/Accelergy reruns).")
    ap.add_argument("--models", type=str, default="all",
                    choices=["llama", "qwen", "all"])
    ap.add_argument("--archs", type=str, default=None,
                    help="Comma-separated arch subset (default: all)")
    ap.add_argument("--drams", type=str, default=",".join(fa.dram_options),
                    help=f"DRAM types (default: {','.join(fa.dram_options)})")
    ap.add_argument("--tp-dims", type=str, default="H")
    ap.add_argument("--output-csv", type=str, default="flashattn_database.csv")
    ap.add_argument("--output-base-dir", type=str, default=_THIS_DIR,
                    help="Base dir holding flashattn_outputs/ (default: script dir)")
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--backend", type=str, default="loky",
                    choices=["loky", "threading"],
                    help="joblib backend (loky=processes, fast; threading=fallback)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Parse only the first N configs (smoke test)")
    ap.add_argument("--mapper-timeout", type=int, default=300)
    ap.add_argument("--mapper-victory", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.models == "llama":
        models = fa.LLAMA_MODELS
    elif args.models == "qwen":
        models = fa.QWEN_MODELS
    else:
        models = fa.LLAMA_MODELS + fa.QWEN_MODELS

    target_drams = [d.strip() for d in args.drams.split(",")]
    tp_dims = [d.strip().upper() for d in args.tp_dims.split(",")]
    arch_filter = ([a.strip() for a in args.archs.split(",")]
                   if args.archs else None)

    configs = fa.build_configs(models, tp_dims=tp_dims, arch_filter=arch_filter)
    if args.limit is not None:
        configs = configs[:args.limit]

    n_combos = len(target_drams) ** 2
    print("=" * 64)
    print("  FlashAttn (FuseMax) — parse-only DB build")
    print("=" * 64)
    print(f"  Models:        {models}")
    print(f"  Archs:         {arch_filter or fa.ARCHS}")
    print(f"  Target DRAMs:  {target_drams}  ({n_combos} combos/config)")
    print(f"  Configs:       {len(configs)}")
    print(f"  Max rows:      {len(configs) * n_combos}")
    print(f"  Backend:       {args.backend} (n_jobs={args.n_jobs})")
    print(f"  Output CSV:    {args.output_csv}")
    if args.dry_run:
        print("  [dry-run] not parsing.")
        return

    # Apply in the parent too (covers threading backend / serial paths).
    _apply_parse_only_patches()

    t0 = time.time()
    raw = joblib.Parallel(n_jobs=args.n_jobs, backend=args.backend, verbose=10)(
        joblib.delayed(parse_one_config)(
            fa_dir, wl, arch, px, py, glb,
            args.output_base_dir, target_drams,
            tp, td, args.mapper_timeout, args.mapper_victory,
        )
        for fa_dir, wl, arch, px, py, glb, tp, td in configs
    )
    elapsed = time.time() - t0

    all_rows, ok = [], 0
    for res in raw:
        if res:
            ok += 1
            all_rows.extend(res)

    valid = [r for r in all_rows if _row_is_valid(r)]
    dropped = len(all_rows) - len(valid)

    print(f"\nParsed in {elapsed/60:.1f} min "
          f"({elapsed/max(len(configs),1):.3f}s/config).")
    print(f"  Configs OK:   {ok}/{len(configs)}")
    print(f"  Rows total:   {len(all_rows)}  (valid {len(valid)}, dropped {dropped})")

    if valid:
        fa.write_csv(valid, args.output_csv)
        fa.write_compat_csv(valid, args.output_csv)
    else:
        print("  No valid rows — nothing written.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
