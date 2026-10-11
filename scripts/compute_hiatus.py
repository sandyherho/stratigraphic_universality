"""Hiatus durations: driftless densities and drifted counts per thickness."""
import numpy as np
from scipy.integrate import quad

from stratigraphic_universality import (drift_from_epsilon, increments,
                                        levy_density, preserved)
from stratigraphic_universality.io import save, write_csv

LAWS = ["gauss", "laplace", "stable1.5", "stable0.8", "fbm0.3", "fbm0.7"]


def main():
    """Pool driftless hiatuses, then count drifted hiatuses per thickness."""
    rng = np.random.default_rng(3)
    bins = np.unique(np.logspace(0, 4, 33).astype(int)).astype(float)
    centers = np.sqrt(bins[1:] * bins[:-1])
    out = {"bins": bins}
    for law in LAWS:
        gaps = []
        for _ in range(8):
            h = np.cumsum(increments(law, 2 ** 15, rng, size=250), axis=1)
            for row in preserved(h):
                gaps.append(np.diff(np.flatnonzero(row)))
        g = np.concatenate(gaps)
        hist, _ = np.histogram(g, bins)
        dens = hist / np.diff(bins) / g.size
        out["gd_" + law] = dens
        m = (centers > 5) & (centers < 1000) & (dens > 0)
        out["slope_" + law] = np.polyfit(np.log(centers[m]),
                                         np.log(dens[m]), 1)[0]
        print(law, "density slope", round(float(out["slope_" + law]), 3))

    eps = 0.01
    mu = drift_from_epsilon("gauss", eps)
    D, v = 0.5, mu
    tc = D / v ** 2
    ell = np.unique(np.logspace(0, np.log10(12 * tc), 30).astype(int))
    cnt = np.zeros(len(ell))
    thick = 0.0
    for _ in range(10):
        h = np.cumsum(mu + increments("gauss", 2 ** 21, rng))
        p = preserved(h)
        idx = np.flatnonzero(p[:len(h) - int(50 * tc)])
        g = np.diff(idx)
        thick += h[idx[-1]] - h[idx[0]]
        cnt += np.array([(g >= L).sum() for L in ell])
    rate = cnt / thick
    exact = np.array([quad(levy_density, L - 0.5, np.inf, args=(D, tc))[0]
                      for L in ell])
    print("rate/exact", np.round(rate / exact, 3))
    out.update(ell=ell, rate=rate, exact_rate=exact, tc=tc)
    save("hiatus", **out)
    write_csv("hiatus_density_driftless", ["tau_center"] + LAWS,
              [centers] + [out["gd_" + k] for k in LAWS])
    write_csv("hiatus_rate_drift", ["ell_steps", "ell_over_tc",
                                    "rate_sim", "rate_exact"],
              [ell, ell / tc, rate, exact])


if __name__ == "__main__":
    main()
