"""Driftless columns: preserved-layer counts and the longest hiatus."""
import numpy as np

from stratigraphic_universality import increments, mean_records, preserved
from stratigraphic_universality.io import save, write_csv

LAWS = ["gauss", "laplace", "stable1.5", "stable0.8", "fbm0.3", "fbm0.7"]
M = 4000
BATCH = 500


def main():
    """Count preserved layers for columns of 10 to 10^4 layers."""
    rng = np.random.default_rng(2)
    ns = np.unique(np.logspace(1, 4, 13).astype(int))
    nmax = ns[-1]
    out = {"Ns": ns, "exact": mean_records(ns - 1)}
    for law in LAWS:
        acc = np.zeros(len(ns))
        frac = []
        for _ in range(M // BATCH):
            h = np.cumsum(increments(law, nmax, rng, size=BATCH), axis=1)
            for i, n in enumerate(ns):
                acc[i] += preserved(h[:, :n]).sum()
            for row in preserved(h):
                idx = np.flatnonzero(row)
                g = np.diff(np.concatenate([[-1], idx]))
                frac.append(g.max() / nmax)
        out[law] = acc / M
        out["lmax_" + law] = np.array(frac)
        print(law, np.round(out[law] / out["exact"], 3),
              "longest/N", round(np.mean(frac), 4), flush=True)
    save("driftless", **out)
    write_csv("driftless_counts", ["N", "exact"] + LAWS,
              [ns, out["exact"]] + [out[k] for k in LAWS])
    write_csv("driftless_longest_fraction",
              ["law", "mean", "std", "standard_error"],
              [LAWS, [out["lmax_" + k].mean() for k in LAWS],
               [out["lmax_" + k].std() for k in LAWS],
               [out["lmax_" + k].std() / np.sqrt(M) for k in LAWS]])


if __name__ == "__main__":
    main()
