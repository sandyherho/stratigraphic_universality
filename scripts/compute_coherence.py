"""Depth-domain coherence of a periodic forcing, and its line shape."""
import numpy as np

from stratigraphic_universality import (ages_on_grid, coherence,
                                        drift_from_epsilon, increments, psi)
from stratigraphic_universality.io import save, write_csv

LAWS = ["gauss", "laplace", "student3", "stable1.5", "fbm0.3", "fbm0.7"]
EPS = 0.005
TCS = 1 / EPS


def fit(dz, c, k0, lo=0.15, hi=0.9):
    """Half-width, apparent wavenumber and decay exponent of a coherence."""
    r = np.abs(c)
    m = (r > lo) & (r < hi) & (dz > 0)
    if m.sum() < 4:
        return np.nan, np.nan, np.nan
    a = -np.polyfit(dz[m], np.log(r[m]), 1)[0]
    ph = np.unwrap(np.angle(c * np.exp(-1j * k0 * dz)))
    b = k0 + np.polyfit(dz[r > lo], ph[r > lo], 1)[0]
    g = np.polyfit(np.log(dz[m]), np.log(-np.log(r[m])), 1)[0]
    return a / k0, b / k0, g


def main():
    """Simulate long columns and measure the coherence at 12 frequencies."""
    rng = np.random.default_rng(5)
    us = np.logspace(np.log10(0.05), np.log10(3), 12)
    out = {"us": us}
    for law in LAWS:
        mu = drift_from_epsilon(law, EPS)
        nrep = 2 if law.startswith("fbm") else 1
        n = 2 ** 21 if nrep == 2 else 2 ** 22
        dz = mu * 5
        ages = [ages_on_grid(np.cumsum(mu + increments(law, n, rng)), dz,
                             int(60 * TCS)) for _ in range(nrep)]
        fa, fb, fg = [], [], []
        for i, u in enumerate(us):
            om = u / TCS
            lags = np.unique(np.logspace(
                0, np.log10(min(len(a) for a in ages) // 8), 60).astype(int))
            c = np.mean([coherence(a, om, lags) for a in ages], axis=0)
            d = lags * dz
            out["curve_%s_%d_dz" % (law, i)] = d
            out["curve_%s_%d_C" % (law, i)] = c
            a, b, g = fit(d, c, om / mu)
            fa.append(a)
            fb.append(b)
            fg.append(g)
        out["fa_" + law] = np.array(fa)
        out["fb_" + law] = np.array(fb)
        out["fg_" + law] = np.array(fg)
        print(law, "a/k0", np.round(fa, 3), flush=True)
    p = psi(us / TCS, np.sqrt(EPS / 2), 0.5)
    k0 = us / TCS / np.sqrt(EPS / 2)
    out["ex_a"] = p.real / k0
    out["ex_b"] = -p.imag / k0
    save("coherence", **out)
    cols = [us, out["ex_a"], out["ex_b"]]
    head = ["omega_tc", "halfwidth_exact", "wavenumber_exact"]
    for law in LAWS:
        cols += [out["fa_" + law], out["fb_" + law]]
        head += ["halfwidth_" + law, "wavenumber_" + law]
    write_csv("line_shape", head, cols)


if __name__ == "__main__":
    main()
