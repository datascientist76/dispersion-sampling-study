"""
Phase 0, step 2: clipping check for Paper 1 dynamics.

Question: for each kernel and LOG_CLIP value L, how often does the raw
log-kernel change exceed the clip bound?  If clipping (almost) never binds
for a kernel, that kernel's invariance across L is trivial, not evidence of
kernel specificity.

Two parts
---------
1. Analytic reachability bounds (no simulation needed).
   The raw change is r = log K(d_t) - log K(d_0).  For |r| to exceed L:
     exponential: |d_t - d_0| > L           -> needs max(d_0, d_t) > L
     gaussian   : |d_t^2 - d_0^2| > L       -> needs max(d_0, d_t) > sqrt(L)
     inverse    : |log((1+d_t)/(1+d_0))| > L -> needs max(d_0, d_t) > e^L - 1
   Compare these thresholds with the largest inter-node distance in YOUR
   populations.  If max distance < threshold, clipping cannot bind.

2. Reference simulator following the Methods of Paper 1 (Sections 3.5-3.7).
   IMPORTANT: the initial state scale (SIGMA), the edge-strength pool, and
   the NLOW/NHIGH x LOCAL/DISTANT assignment below are PLACEHOLDERS.
   Numbers from this script are NOT Paper 1 numbers until you replace
   `make_initial_conditions` with the frozen Phase-10-R2 construction, or
   instead paste the counting lines (see INSTRUMENT_YOUR_CODE) into your own
   simulator, which is the preferred route.

INSTRUMENT_YOUR_CODE (preferred): inside your existing time loop, after
computing the raw change r (upper-triangle edge vector) and before clipping:
    clipped_count += int(np.sum(np.abs(r) > L))
    total_count   += r.size
    max_dist_seen  = max(max_dist_seen, d_t.max())
Then report clipped_count / total_count per (kernel, L).
"""
import argparse
import json
import numpy as np

ETA = 0.03
STEPS = 120

LOG_KERNELS = {
    "exponential": lambda d: -d,
    "gaussian": lambda d: -d ** 2,
    "inverse": lambda d: -np.log1p(d),
}


def analytic_threshold(kernel, L):
    """Smallest value that max(d_0, d_t) must exceed for |r| > L to be possible."""
    if kernel == "exponential":
        return float(L)
    if kernel == "gaussian":
        return float(np.sqrt(L))
    if kernel == "inverse":
        return float(np.expm1(L))
    raise ValueError(kernel)


def make_initial_conditions(n, sigma, rng):
    """PLACEHOLDER. Replace with the frozen Phase-10-R2 construction."""
    x0 = rng.normal(0.0, sigma, n)
    x0 -= x0.mean()
    m = n * (n - 1) // 2
    w0 = rng.lognormal(mean=0.0, sigma=1.0, size=m)  # positive, heterogeneous
    return x0, w0


def simulate(kernel, L, n, sigma, seed):
    rng = np.random.default_rng(seed)
    x, w0 = make_initial_conditions(n, sigma, rng)
    iu = np.triu_indices(n, 1)
    logk = LOG_KERNELS[kernel]
    d0 = np.abs(x[iu[0]] - x[iu[1]])
    base = logk(d0)
    wbar0 = w0.mean()
    clipped = total = 0
    max_d = d0.max()
    for _ in range(STEPS):
        d = np.abs(x[iu[0]] - x[iu[1]])
        max_d = max(max_d, d.max())
        r = logk(d) - base
        clipped += int(np.sum(np.abs(r) > L))
        total += r.size
        wt = w0 * np.exp(np.clip(r, -L, L))
        w = wt * (wbar0 / wt.mean())
        sgn = np.sign(x[iu[1]] - x[iu[0]])
        f = np.zeros(n)
        np.add.at(f, iu[0], w * sgn)
        np.add.at(f, iu[1], -w * sgn)
        x = x + (ETA / n) * f
        x -= x.mean()
    return clipped, total, max_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="PLACEHOLDER initial state scale; set to your frozen value")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--clips", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    rows = []
    print(f"n={args.n} sigma={args.sigma} seeds={args.seeds}  (placeholder initial conditions)")
    print(f"{'kernel':<12}{'L':>3}{'clip frac':>12}{'max dist':>10}{'threshold':>11}  can clip?")
    for kernel in LOG_KERNELS:
        for L in args.clips:
            c = t = 0
            md = 0.0
            for s in range(args.seeds):
                ci, ti, m = simulate(kernel, L, args.n, args.sigma, s)
                c += ci
                t += ti
                md = max(md, m)
            thr = analytic_threshold(kernel, L)
            can = md > thr
            rows.append(dict(kernel=kernel, L=L, clip_fraction=c / t,
                             max_distance=md, threshold=thr, can_clip=bool(can)))
            print(f"{kernel:<12}{L:>3}{c / t:>12.6f}{md:>10.3f}{thr:>11.3f}  {'yes' if can else 'NO'}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
