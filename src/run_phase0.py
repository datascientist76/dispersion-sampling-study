"""
Phase 0 runner: clipping fractions per kernel and L, pooled over seeds.

Edit ONLY initial_conditions() to use your frozen Phase-10-R2 construction.
Until then the default placeholder is used and results are NOT Paper 1 numbers.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from clip_counter import ClipCounter
from phase0_clipping_check import (ETA, LOG_KERNELS, STEPS,
                                   analytic_threshold, make_initial_conditions)

CHECKPOINTS = {0, 1, 5, 10, 20, 40, 60, 80, 100, 120}
PAPER1_REF = {("gaussian", 2): 0.118454, ("gaussian", 6): 0.021391}


def initial_conditions(n, sigma, rng):
    # >>> REPLACE this body with your frozen construction: it must return
    # >>> (x0 centered states of length n, w0 edge strengths in
    # >>> np.triu_indices(n, 1) order).
    return make_initial_conditions(n, sigma, rng)


def run_one(kernel, L, n, sigma, seed, counter, mode):
    rng = np.random.default_rng(seed)
    x, w0 = initial_conditions(n, sigma, rng)
    iu = np.triu_indices(n, 1)
    logk = LOG_KERNELS[kernel]
    base = logk(np.abs(x[iu[0]] - x[iu[1]]))
    wbar0 = w0.mean()
    for t in range(STEPS + 1):
        d = np.abs(x[iu[0]] - x[iu[1]])
        r = logk(d) - base
        if (mode == "all" and t < STEPS) or (mode == "checkpoints" and t in CHECKPOINTS):
            counter.update(r, d, L)
        if t == STEPS:
            break
        wt = w0 * np.exp(np.clip(r, -L, L))
        w = wt * (wbar0 / wt.mean())
        sgn = np.sign(x[iu[1]] - x[iu[0]])
        f = np.zeros(n)
        np.add.at(f, iu[0], w * sgn)
        np.add.at(f, iu[1], -w * sgn)
        x = x + (ETA / n) * f
        x -= x.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--clips", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--mode", choices=["all", "checkpoints"], default="all",
                    help="count every step, or only stored checkpoints")
    ap.add_argument("--out", default="results/phase0_clipping.json")
    a = ap.parse_args()

    rows = []
    print(f"n={a.n} seeds={a.seeds} mode={a.mode} sigma={a.sigma}")
    print(f"{'kernel':<12}{'L':>3}{'clip frac':>12}{'max dist':>10}{'threshold':>11}  can clip?")
    for kernel in LOG_KERNELS:
        for L in a.clips:
            c = ClipCounter()
            for s in range(a.seeds):
                run_one(kernel, L, a.n, a.sigma, s, c, a.mode)
            sm = c.summary()
            thr = analytic_threshold(kernel, L)
            can = sm["max_distance"] > thr
            rows.append(dict(kernel=kernel, L=L, **sm, threshold=thr, can_clip=bool(can)))
            print(f"{kernel:<12}{L:>3}{sm['clip_fraction']:>12.6f}"
                  f"{sm['max_distance']:>10.3f}{thr:>11.3f}  {'yes' if can else 'NO'}")

    print("\nValidation against Paper 1 (Gaussian, Phase 16):")
    for (k, L), ref in PAPER1_REF.items():
        got = [r["clip_fraction"] for r in rows if r["kernel"] == k and r["L"] == L]
        if got:
            print(f"  {k} L={L}: reference {ref:.6f}  this run {got[0]:.6f}  diff {got[0]-ref:+.6f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(args=vars(a), rows=rows), fh, indent=2)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
