#!/usr/bin/env python3
"""Fengshui AE — determinism self-test (with positive and negative controls).

The artifact's reproducibility rests on the per-network GA reseed
(`chiplet_sel.deterministic_ga_rng`), which seeds Python's global `random`
stream from a stable hash of each network's unique name.  This script checks
that mechanism with three separate probes and says exactly what each one does
and does not prove:

  CHECK 1 (gating) — worker-count invariance.
      One GA remap on a fixed pool, run with 1 worker and with 4 workers.
      Results must be byte-identical.  PROVES: your core count does not change
      the numbers.  DOES NOT PROVE: that the reseed is active — on a small net
      subset the GA optimum can be RNG-insensitive, so this check can pass even
      with FENGSHUI_DETERMINISTIC=0.  That is why CHECK 2 exists.

  CHECK 2 (gating) — positive control: the seed actually matters.
      Re-runs with a different seed family (FENGSHUI_GA_SEED) and requires at
      least one per-network value to change.  PROVES: the reseed is wired in and
      live — the GA really is consuming the seeded stream, so CHECK 1's equality
      is a property of the mechanism and not a vacuous tautology.  If the first
      alternate seed changes nothing, the script tries further seeds and then a
      larger network subset before declaring failure, and reports what it took.

  CHECK 3 (reported, NOT gating) — negative control: mechanism off.
      Two back-to-back runs with FENGSHUI_DETERMINISTIC=0 (legacy stochastic
      path).  If they diverge, the harness demonstrably can detect
      nondeterminism.  If they agree, the GA optimum is simply RNG-insensitive
      on this subset — informative, but not a failure, so it does not affect the
      exit code.

Exit code: 0 only if CHECK 1 and CHECK 2 both pass; 1 otherwise (2 on setup
error).  Runtime: ~2-3 min per run; 4-5 runs total (~10-15 min) with the
defaults, plus the one-time database load.

Run inside the submission image, from `$DL/fengshui_AE` (same mount as README §1):

  docker run --rm -v "$PWD:/fengshui" -w /fengshui \
    fengshui-analysis:latest \
    python3 tools/verify_determinism.py

Environment knobs:
  FENGSHUI_SCRIPTS   framework `scripts/` dir (default: <repo>/src/scripts)
  VERIFY_OBJ         objective / pool to use          (default: energy)
  VERIFY_DB          database path, relative to SCRIPTS (default: ../unified_database.csv)
  VERIFY_N_NETS      networks in the subset           (default: 6)
  VERIFY_MAX_NETS    escalation size for CHECK 2      (default: 12)
  VERIFY_ALT_SEEDS   comma-separated alternate seeds  (default: 7,13,29)
  VERIFY_NEG_CONTROL run CHECK 3 (1/0)                (default: 1)
  VERIFY_CHAIN       chiplet-pool chain version       (default: ae)
"""
import os
import sys
import contextlib

# ── Locate the framework relative to THIS file so the script works both inside
#    the container (repo mounted at /fengshui) and from a plain checkout. ──────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SCRIPTS = os.path.join(os.path.dirname(_HERE), 'src', 'scripts')
SCRIPTS = os.path.abspath(os.environ.get('FENGSHUI_SCRIPTS', _DEFAULT_SCRIPTS))

if not os.path.isdir(SCRIPTS):
    sys.stderr.write(
        f"[verify] ERROR: framework scripts dir not found: {SCRIPTS}\n"
        f"[verify] expected the shipped layout <repo>/src/scripts; override with\n"
        f"[verify]   -e FENGSHUI_SCRIPTS=/path/to/scripts\n")
    sys.exit(2)

sys.path.insert(0, SCRIPTS)
os.chdir(SCRIPTS)

import chiplet_sel                                            # noqa: E402
from baseline import _build_virtual_nets                      # noqa: E402
from chiplet_sel import run_single_optimization               # noqa: E402
from arch_impl.ablation_study import load_pool                # noqa: E402

OBJ = os.environ.get('VERIFY_OBJ', 'energy')
DB = os.environ.get('VERIFY_DB', '../unified_database.csv')
N_NETS = int(os.environ.get('VERIFY_N_NETS', '6'))
MAX_NETS = int(os.environ.get('VERIFY_MAX_NETS', '12'))
ALT_SEEDS = [int(s) for s in os.environ.get('VERIFY_ALT_SEEDS', '7,13,29').split(',') if s.strip()]
NEG_CONTROL = os.environ.get('VERIFY_NEG_CONTROL', '1') != '0'
CHAIN = os.environ.get('VERIFY_CHAIN', 'ae')


@contextlib.contextmanager
def ga_env(deterministic, seed_base):
    """Select the determinism knobs for one run, restoring them afterwards.

    NOTE: `chiplet_sel._GA_SEED_BASE` is read from FENGSHUI_GA_SEED at *import*
    time, so setting the environment variable alone has no effect within an
    already-imported process; we patch the module attribute as well (
    `_stable_seed` reads the module global on every call).  Worker processes are
    forked, so they inherit the patched value.  FENGSHUI_DETERMINISTIC *is* read
    per call inside `deterministic_ga_rng`, so the env var suffices there.
    """
    saved_env = {k: os.environ.get(k) for k in ('FENGSHUI_DETERMINISTIC', 'FENGSHUI_GA_SEED')}
    saved_base = chiplet_sel._GA_SEED_BASE
    os.environ['FENGSHUI_DETERMINISTIC'] = '1' if deterministic else '0'
    os.environ['FENGSHUI_GA_SEED'] = str(seed_base)
    chiplet_sel._GA_SEED_BASE = seed_base
    try:
        yield
    finally:
        chiplet_sel._GA_SEED_BASE = saved_base
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_once(vnets, pool, workers):
    _, results = run_single_optimization(
        virtual_nets=vnets,
        chiplet_group=pool,
        objective=OBJ,
        results_file=DB,
        cost_aware=False,
        use_dag_cp=True,
        v_het_batch=True,
        n_workers=workers,
        use_sequential=True,
    )
    # repr() of the float: determinism should be exact, so compare the full repr
    return {k: repr(results[k]['min_value']) for k in sorted(results)}


def diff_keys(r1, r2):
    """Networks whose value differs between two runs (missing keys count)."""
    return sorted(k for k in set(r1) | set(r2) if r1.get(k) != r2.get(k))


def show(r1, r2, label2):
    for k in sorted(r1):
        flag = '' if r1.get(k) == r2.get(k) else f'   <-- DIFFERS ({label2}={r2.get(k)})'
        print(f"    {k:42s} {r1[k]}{flag}")


def main():
    print("[verify] Fengshui determinism self-test")
    print(f"[verify] scripts={SCRIPTS}")
    print(f"[verify] obj={OBJ}  chain={CHAIN}  db={DB}  n_nets={N_NETS}"
          f"  alt_seeds={ALT_SEEDS}  neg_control={NEG_CONTROL}")
    print("[verify] expect ~2-3 min per run, 4-5 runs total, after the one-time DB load\n")

    all_vnets = _build_virtual_nets(database_file=DB)
    vnets = all_vnets[:N_NETS]
    pool, pool_file, _ = load_pool(OBJ, cost_aware=False, n=8, chain_version=CHAIN)
    print(f"[verify] pool_file={pool_file}")
    print(f"[verify] nets={[v.get_unique_name() for v in vnets]}\n")

    # ── CHECK 1: worker-count invariance (gating) ───────────────────────────
    print("[check 1] worker-count invariance (FENGSHUI_DETERMINISTIC=1, seed 0)")
    print("[check 1]   run A: 1 worker ...")
    with ga_env(True, 0):
        rA = run_once(vnets, pool, 1)
    print("[check 1]   run B: 4 workers ...")
    with ga_env(True, 0):
        rB = run_once(vnets, pool, 4)
    show(rA, rB, '4w')
    check1 = (rA == rB)
    print(f"[check 1] {'PASS' if check1 else 'FAIL'}: results are "
          f"{'identical' if check1 else 'NOT identical'} across 1 and 4 workers.")
    print("[check 1] proves: core count does not change the numbers.")
    print("[check 1] does NOT prove: that the reseed is active (see check 2).\n")

    # ── CHECK 2: positive control — the seed must matter (gating) ───────────
    print("[check 2] positive control: does FENGSHUI_GA_SEED change any value?")
    hit_seed, hit_diffs, base = None, [], rA
    for seed in ALT_SEEDS:
        print(f"[check 2]   run with seed {seed} (1 worker) ...")
        with ga_env(True, seed):
            rC = run_once(vnets, pool, 1)
        d = diff_keys(base, rC)
        print(f"[check 2]   seed {seed}: {len(d)}/{len(base)} network value(s) changed")
        if d:
            hit_seed, hit_diffs = seed, d
            show(base, rC, f'seed{seed}')
            break

    if hit_seed is None and len(vnets) < min(MAX_NETS, len(all_vnets)):
        n_big = min(MAX_NETS, len(all_vnets))
        print(f"[check 2]   no seed changed anything on {len(vnets)} nets — "
              f"escalating the subset to {n_big} nets (the signal is weak on the "
              f"small subset; this is expected to be reported, not hidden)")
        vnets_big = all_vnets[:n_big]
        with ga_env(True, 0):
            base = run_once(vnets_big, pool, 1)
        for seed in ALT_SEEDS:
            print(f"[check 2]   run with seed {seed} on {n_big} nets (1 worker) ...")
            with ga_env(True, seed):
                rC = run_once(vnets_big, pool, 1)
            d = diff_keys(base, rC)
            print(f"[check 2]   seed {seed}: {len(d)}/{len(base)} network value(s) changed")
            if d:
                hit_seed, hit_diffs = seed, d
                show(base, rC, f'seed{seed}')
                break

    check2 = hit_seed is not None
    if check2:
        print(f"[check 2] PASS: seed {hit_seed} changed {len(hit_diffs)} value(s) "
              f"on {len(base)} nets ({', '.join(hit_diffs)}).")
        print("[check 2] proves: the per-network reseed is wired in and live, so "
              "check 1's equality is a property of the mechanism, not a vacuous "
              "PASS. Note the GA optimum is only weakly seed-sensitive on this "
              "subset — a small number of changed values is the expected signal.")
    else:
        print("[check 2] FAIL: no alternate seed changed any value, even after "
              "escalation. The reseed cannot be observed, so check 1 alone is "
              "not evidence that the determinism mechanism works. Try a larger "
              "VERIFY_N_NETS/VERIFY_MAX_NETS or more VERIFY_ALT_SEEDS before "
              "concluding the mechanism is broken.")
    print()

    # ── CHECK 3: negative control — mechanism off (reported, not gating) ────
    if NEG_CONTROL:
        print("[check 3] negative control: two runs with FENGSHUI_DETERMINISTIC=0")
        with ga_env(False, 0):
            rD = run_once(vnets, pool, 1)
            rE = run_once(vnets, pool, 1)
        d_off = diff_keys(rD, rE)
        d_vs_det = diff_keys(rA, rD)
        print(f"[check 3]   legacy run 1 vs legacy run 2: {len(d_off)}/{len(rD)} differ")
        print(f"[check 3]   legacy run 1 vs deterministic run A: {len(d_vs_det)}/{len(rD)} differ")
        if d_off:
            print("[check 3] observed: with the reseed disabled the two runs diverge, "
                  "so this harness demonstrably detects nondeterminism.")
        else:
            print("[check 3] observed: even with the reseed disabled the two runs agree "
                  "— on this subset the GA optimum is largely RNG-insensitive. That is "
                  "informative, NOT a failure, and it is exactly why check 2 (not check "
                  "1) is what establishes the mechanism. Not gating; exit code unaffected.")
        print()
    else:
        print("[check 3] skipped (VERIFY_NEG_CONTROL=0)\n")

    ok = check1 and check2
    print(f"[verify] check 1 (worker-count invariance): {'PASS' if check1 else 'FAIL'}")
    print(f"[verify] check 2 (seed sensitivity / mechanism live): {'PASS' if check2 else 'FAIL'}")
    print(f"[verify] RESULT: {'PASS' if ok else 'FAIL'} (exit {0 if ok else 1})")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
