"""Figure 3: hiatus durations, counts per thickness, and the longest gap."""
import numpy as np

from stratigraphic_universality.io import load, save_figure
from stratigraphic_universality.plotting import COLORS, LABELS, plt, tag


def main():
    """Draw the three hiatus panels."""
    g = load("hiatus")
    b = g["bins"]
    cb = np.sqrt(b[1:] * b[:-1])
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))
    for law in ["gauss", "laplace", "stable1.5", "stable0.8", "fbm0.3",
                "fbm0.7"]:
        dd = g["gd_" + law]
        m = dd > 0
        fbm = law.startswith("fbm")
        ax[0].loglog(cb[m], dd[m], "s" if fbm else "o", ms=2.5,
                     mfc=COLORS[law] if fbm else "none", color=COLORS[law],
                     label=LABELS[law])
    tt = np.array([3, 3000.])
    ax[0].loglog(tt, 0.35 * tt ** -1.5, "k-", lw=0.8)
    ax[0].text(300, 0.35 * 300 ** -1.5 * 2.2, r"$\tau^{-3/2}$")
    ax[0].set_xlabel(r"hiatus $\tau$ (steps)")
    ax[0].set_ylabel("probability density")
    ax[0].legend(frameon=False, loc="lower left", fontsize=5.8)
    tag(ax[0], "(a)")
    tc = float(g["tc"])
    ax[1].loglog(g["ell"] / tc, g["rate"], "o", ms=3, mfc="none",
                 color=COLORS["gauss"], label="simulation")
    ax[1].loglog(g["ell"] / tc, g["exact_rate"], "k-", lw=0.9,
                 label="Lévy measure, exact")
    ax[1].set_xlabel(r"$\ell/t_c$")
    ax[1].set_ylabel(r"hiatuses $>\ell$ per unit thickness")
    ax[1].legend(frameon=False, loc="lower left")
    tag(ax[1], "(b)")
    f = load("longest")
    ts = f["Ts"]
    tc4 = float(f["tc"])
    q1 = [np.percentile(f["L_%d" % T], 25) / tc4 for T in ts]
    q3 = [np.percentile(f["L_%d" % T], 75) / tc4 for T in ts]
    ax[2].fill_between(ts, q1, q3, color=COLORS["gauss"], alpha=0.2, lw=0,
                       label="simulation, quartiles")
    ax[2].plot(ts, f["med_sim"] / tc4, "o", ms=3, mfc="none",
               color=COLORS["gauss"], label="simulation, median")
    ax[2].plot(ts, f["med_ex"] / tc4, "k-", lw=0.9, label="exact median")
    tf = np.logspace(1.4, 4.1, 50)
    ax[2].plot(tf, 4 * (np.log(tf) - 1.5 * np.log(np.log(tf))), "k:",
               lw=0.9, label=r"$4[\ln\hat T-\frac{3}{2}\ln\ln\hat T]$")
    ax[2].set_xscale("log")
    ax[2].set_xlabel(r"record duration $T/t_c$")
    ax[2].set_ylabel(r"longest hiatus $/t_c$")
    ax[2].legend(frameon=False, loc="upper left")
    tag(ax[2], "(c)")
    fig.tight_layout()
    save_figure(fig, "fig03_hiatus")
    plt.close(fig)


if __name__ == "__main__":
    main()
