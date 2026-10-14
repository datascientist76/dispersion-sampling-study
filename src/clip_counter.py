"""Accumulates clipping statistics across time steps and seeds."""
import numpy as np


class ClipCounter:
    def __init__(self):
        self.clipped = 0
        self.total = 0
        self.max_d = 0.0

    def update(self, r, d, L):
        """r: raw log-kernel change per edge (before clipping); d: edge distances."""
        self.clipped += int(np.sum(np.abs(r) > L))
        self.total += int(r.size)
        self.max_d = max(self.max_d, float(np.max(d)))

    def summary(self):
        frac = self.clipped / self.total if self.total else float("nan")
        return {"clipped": self.clipped, "total": self.total,
                "clip_fraction": frac, "max_distance": self.max_d}
