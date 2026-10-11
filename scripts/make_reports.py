"""Write plain-text reports of every number quoted in the manuscript."""
import os

import numpy as np
from scipy.optimize import brentq

from stratigraphic_universality import (chronometric_information,
                                        completeness_asymptote,
                                        spitzer_completeness)
from stratigraphic_universality.io import REPORT, load


def write(name, lines):
    """Write a report file."""
    path = os.path.join(REPORT, name + ".txt")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", path)


def fmt(arr, nd=3):
    """Format an array compactly."""
    return " ".join(f"{x:.{nd}f}" for x in np.atleast_1d(arr))


def slope(x, y, mask):
    """Least-squares slope in log-log coordinates."""
    return np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0]


def completeness_report():
    """Completeness, exponents and the two-term asymptote."""
    d = load("completeness")
    eps = d["eps"]
    m = eps <= 1e-2
    lines = ["COMPLETENESS", "=" * 60,
             "eps = dt/t_c: " + fmt(eps, 4), "",
             "Gaussian simulation / exact Spitzer sum:",
             "  " + fmt(d["gauss"] / d["spitzer"]), "",
             "Two-term asymptote / exact Spitzer sum:"]
    for e in [1e-4, 1e-3, 1e-2, 1e-1]:
        r = completeness_asymptote(e) / spitzer_completeness(np.sqrt(e / 2))
        lines.append(f"  eps={e:g}: relative error {abs(r - 1):.2e}")
    lines += ["", "c / sqrt(eps) at eps=1e-3 and fitted exponent "
              "(eps <= 1e-2):"]
    for law in ["gauss", "laplace", "student3", "stable1.5", "stable1.2",
                "fbm0.3", "fbm0.5", "fbm0.7"]:
        lines.append(f"  {law:10s} ratio {d[law][0] / np.sqrt(eps[0]):.3f}"
                     f"   exponent {slope(eps, d[law], m):.3f}")
    write("completeness", lines)


def driftless_report():
    """Preserved counts and the longest hiatus without net accumulation."""
    e = load("driftless")
    ns = e["Ns"]
    lines = ["DRIFTLESS COLUMNS", "=" * 60, "N: " + " ".join(map(str, ns)),
             "", "mean preserved / exact Sparre Andersen formula:"]
    devs = []
    for law in ["gauss", "laplace", "stable1.5", "stable0.8"]:
        r = e[law] / e["exact"]
        devs.append(np.max(np.abs(r - 1)))
        lines.append(f"  {law:10s} " + fmt(r))
    lines.append(f"  maximum deviation {max(devs):.4f}")
    lines += ["", "growth exponent of preserved count (N >= 100):"]
    for law in ["fbm0.3", "fbm0.7"]:
        lines.append(f"  {law:10s} {slope(ns, e[law], ns >= 100):.3f}")
    lines += ["", "longest hiatus / N (mean, standard error):"]
    for law in ["gauss", "laplace", "stable1.5", "stable0.8", "fbm0.3",
                "fbm0.7"]:
        x = e["lmax_" + law]
        se = x.std() / np.sqrt(x.size)
        lines.append(f"  {law:10s} {x.mean():.4f}  {se:.4f}")
    write("driftless", lines)


def hiatus_report():
    """Hiatus densities, counts per thickness and the longest hiatus."""
    g = load("hiatus")
    lines = ["HIATUS STATISTICS", "=" * 60,
             "driftless density slopes (5 < tau < 1000 steps):"]
    for law in ["gauss", "laplace", "stable1.5", "stable0.8", "fbm0.3",
                "fbm0.7"]:
        lines.append(f"  {law:10s} {float(g['slope_' + law]):.3f}")
    r = g["rate"] / g["exact_rate"]
    x = g["ell"] / g["tc"]
    lines += ["", "drifted Gaussian, eps = 0.01: simulated / exact rate",
              "  ell/t_c   ratio"]
    lines += [f"  {a:7.3f}   {b:.3f}" for a, b in zip(x, r)]
    ok = (x >= 0.09) & (x <= 6)
    lines.append(f"  max deviation for 0.09 <= ell/t_c <= 6: "
                 f"{np.max(np.abs(r[ok] - 1)):.3f}")
    f = load("longest")
    tc = float(f["tc"])
    lines += ["", "median longest hiatus / t_c (eps = 0.02):",
              "  T/t_c   simulation   exact"]
    for T, a, b in zip(f["Ts"], f["med_sim"], f["med_ex"]):
        lines.append(f"  {T:6d}   {a / tc:8.2f}   {b / tc:8.2f}")
    write("hiatus", lines)


def coherence_report():
    """Line shape and misregistration."""
    h = load("coherence")
    us = h["us"]
    m = us >= 0.15
    lines = ["DEPTH-DOMAIN COHERENCE", "=" * 60,
             "omega t_c: " + fmt(us), "",
             "fitted / exact, for omega t_c >= 0.15:"]
    for law in ["gauss", "laplace", "student3"]:
        ra = h["fa_" + law][m] / h["ex_a"][m]
        rb = h["fb_" + law][m] / h["ex_b"][m]
        lines.append(f"  {law:10s} half-width {ra.min():.3f}..{ra.max():.3f}"
                     f"   wavenumber {rb.min():.3f}..{rb.max():.3f}")
    mi = load("misregistration")
    lines += ["", "misregistration scaling exponents (dz/(v t_c) >= 10):",
              "  law         variance   IQR"]
    for law in ["gauss", "laplace", "stable1.5", "fbm0.3", "fbm0.7"]:
        lines.append(f"  {law:10s}  {float(mi['svar_' + law]):.3f}     "
                     f"{float(mi['siqr_' + law]):.3f}")
    vr = mi["var_ratio_gauss"]
    lines.append("  Gaussian variance / (2 t_c dz / v), last five lags: "
                 + fmt(vr[-5:]))
    write("coherence", lines)


def information_report():
    """Chronometric information and the coherence horizon."""
    lines = ["CHRONOMETRIC INFORMATION", "=" * 60,
             "  omega t_c   bits at 1 cycle   cycles to 0.1 bit   N_coh"]
    for u in [0.003, 0.01, 0.03, 0.1]:
        i1 = chronometric_information(1, u)
        n01 = brentq(lambda n: chronometric_information(n, u) - 0.1, 0.1,
                     5000)
        lines.append(f"  {u:8.3f}   {i1:12.3f}   {n01:15.2f}   "
                     f"{1 / (2 * np.pi * u):8.2f}")
    lines += ["", "t_c for ten coherent cycles, P/(40 pi^2), kyr:"]
    for p in [20, 41, 100, 405]:
        lines.append(f"  P = {p:4d} kyr: t_c <= {p / (40 * np.pi ** 2):.3f}")
    write("information", lines)


def main():
    """Write all reports."""
    completeness_report()
    driftless_report()
    hiatus_report()
    coherence_report()
    information_report()


if __name__ == "__main__":
    main()
