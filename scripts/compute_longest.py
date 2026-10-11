"""Longest hiatus against record duration, with the exact distribution."""
import numpy as np
from scipy.optimize import brentq

from stratigraphic_universality import (drift_from_epsilon, hiatus_rate,
                                        increments, preserved)
from stratigraphic_universality.io import save, write_csv


def main():
    """Simulate records of 30 to 10^4 crossover times."""
    rng = np.random.default_rng(4)
    eps = 0.02
    mu = drift_from_epsilon("gauss", eps)
    D, v = 0.5, mu
    tc = D / v ** 2
    ts = np.array([30, 100, 300, 1000, 3000, 10000])
    out = {"Ts": ts, "tc": tc}
    med_sim, med_ex, q1, q3 = [], [], [], []
    for T in ts:
        n = int(T * tc)
        reps = min(max(40, int(4e7 / n)), 2000)
        longest = []
        for _ in range(reps):
            h = np.cumsum(mu + increments("gauss", n + int(40 * tc), rng))
            idx = np.flatnonzero(preserved(h)[:n])
            longest.append(np.diff(idx).max(initial=0))
        longest = np.array(longest)
        thick = v * n
        me = brentq(lambda ell: thick * hiatus_rate(ell, D, tc) - np.log(2),
                    1, 1e7)
        med_sim.append(np.median(longest))
        med_ex.append(me)
        q1.append(np.percentile(longest, 25))
        q3.append(np.percentile(longest, 75))
        out["L_%d" % T] = longest
        print(T, reps, np.median(longest) / tc, me / tc, flush=True)
    out.update(med_sim=np.array(med_sim), med_ex=np.array(med_ex))
    save("longest", **out)
    write_csv("longest_hiatus", ["T_over_tc", "median_sim_over_tc",
                                 "q1_over_tc", "q3_over_tc",
                                 "median_exact_over_tc"],
              [ts, np.array(med_sim) / tc, np.array(q1) / tc,
               np.array(q3) / tc, np.array(med_ex) / tc])


if __name__ == "__main__":
    main()
