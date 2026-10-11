"""Completeness against reduced resolution for nine surface laws."""
import numpy as np

from stratigraphic_universality import (completeness, increments,
                                        drift_from_epsilon,
                                        spitzer_completeness)
from stratigraphic_universality.io import save, write_csv

LAWS = ["gauss", "laplace", "student3", "stable1.5", "stable1.2",
        "fbm0.3", "fbm0.5", "fbm0.7"]
N = 2 ** 21


def run_law(law, eps_values, reps, rng):
    """Mean completeness over ``reps`` realizations at each resolution."""
    out = []
    for eps in eps_values:
        mu = drift_from_epsilon(law, eps)
        cut = int(min(N // 4, 200 / eps))
        vals = [completeness(np.cumsum(mu + increments(law, N, rng)), cut)
                for _ in range(reps)]
        out.append(np.mean(vals))
    return np.array(out)


def main():
    """Run all laws, then refine the stable laws with more realizations."""
    eps = np.logspace(-3, 1, 13)
    res = {}
    rng = np.random.default_rng(1)
    for law in LAWS:
        res[law] = run_law(law, eps, 3, rng)
        print(law, np.round(res[law] / np.sqrt(eps), 3), flush=True)
    rng = np.random.default_rng(7)
    for law in ["stable1.5", "stable1.2"]:
        res[law] = run_law(law, eps, 12, rng)
        print(law, "(12 realizations)", np.round(res[law] / np.sqrt(eps), 3))
    spit = np.array([spitzer_completeness(np.sqrt(e / 2)) for e in eps])
    save("completeness", eps=eps, spitzer=spit, **res)
    write_csv("completeness", ["eps", "spitzer_exact"] + LAWS,
              [eps, spit] + [res[k] for k in LAWS])


if __name__ == "__main__":
    main()
