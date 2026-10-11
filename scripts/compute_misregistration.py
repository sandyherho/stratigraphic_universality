"""Scaling of the age misregistration with thickness."""
import numpy as np

from stratigraphic_universality import (drift_from_epsilon, increments,
                                        preserved)
from stratigraphic_universality.io import save, write_csv

LAWS = ["gauss", "laplace", "stable1.5", "fbm0.3", "fbm0.7"]
EPS = 0.005
TCS = 1 / EPS


def main():
    """Interquartile range and variance of misregistration against depth."""
    rng = np.random.default_rng(6)
    out = {}
    lags = np.unique(np.logspace(0, 4.1, 36).astype(int))
    for law in LAWS:
        mu = drift_from_epsilon(law, EPS)
        dz = mu * 5
        var, iqr = [], []
        for _ in range(8):
            n = 2 ** 21
            h = np.cumsum(mu + increments(law, n, rng))
            p = preserved(h)
            idx = np.flatnonzero(p[:n - int(60 * TCS)])
            hp = h[idx]
            z = np.arange(hp[0], hp[-1], dz)
            ages = idx[np.searchsorted(hp, z, "right") - 1].astype(float)
            v, q = [], []
            for k in lags:
                d = ages[k:] - ages[:-k] - k * dz / mu
                v.append(d.var())
                q.append(np.subtract(*np.percentile(d, [75, 25])))
            var.append(v)
            iqr.append(q)
        var = np.mean(var, 0)
        iqr = np.mean(iqr, 0)
        tau = lags * dz / mu
        m = tau > 10 * TCS
        out["tau_" + law] = tau / TCS
        out["var_" + law] = var / TCS ** 2
        out["iqr_" + law] = iqr / TCS
        out["svar_" + law] = np.polyfit(np.log(tau[m]), np.log(var[m]), 1)[0]
        out["siqr_" + law] = np.polyfit(np.log(tau[m]), np.log(iqr[m]), 1)[0]
        print(law, "var slope", round(float(out["svar_" + law]), 3),
              "iqr slope", round(float(out["siqr_" + law]), 3), flush=True)
        if law == "gauss":
            out["var_ratio_gauss"] = var / (2 * TCS * tau)
    save("misregistration", **out)
    write_csv("misregistration", ["dz_over_v_tc"]
              + ["iqr_" + k for k in LAWS] + ["var_" + k for k in LAWS],
              [out["tau_gauss"]] + [out["iqr_" + k] for k in LAWS]
              + [out["var_" + k] for k in LAWS])


if __name__ == "__main__":
    main()
