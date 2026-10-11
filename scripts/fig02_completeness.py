"""Figure 2: completeness, its universality, and the driftless limit."""
import numpy as np

from stratigraphic_universality import (completeness_asymptote,
                                        spitzer_completeness)
from stratigraphic_universality.io import load, save_figure, write_csv
from stratigraphic_universality.plotting import COLORS, LABELS, plt, tag

LEG = dict(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.26),
           ncol=2, fontsize=5.8, columnspacing=0.6, handletextpad=0.3)


def main():
    """Draw the three completeness panels."""
    d = load("completeness")
    eps = d["eps"]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 3.1))
    xf = np.logspace(-3, 1, 200)
    sp = np.array([spitzer_completeness(np.sqrt(z / 2)) for z in xf])
    ax[0].plot(xf, sp / np.sqrt(xf), "k-", lw=1.0,
               label="Spitzer, exact (Gaussian)")
    ax[0].plot(xf, completeness_asymptote(xf) / np.sqrt(xf), "k:", lw=1.0,
               label="two-term asymptote")
    laws_a = ["gauss", "laplace", "student3", "stable1.5", "stable1.2"]
    for law in laws_a:
        ax[0].plot(eps, d[law] / np.sqrt(eps), "o", ms=3, mfc="none",
                   color=COLORS[law], label=LABELS[law])
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$\Delta t/t_c$")
    ax[0].set_ylabel(r"$c\,(\Delta t/t_c)^{-1/2}$")
    ax[0].set_ylim(0, 1.25)
    ax[0].legend(**LEG)
    tag(ax[0], "(a)")
    for law, hu in [("fbm0.3", 0.3), ("fbm0.5", 0.5), ("fbm0.7", 0.7)]:
        ax[1].loglog(eps, d[law], "o", ms=3, mfc="none", color=COLORS[law],
                     label=LABELS[law])
        m = eps <= 1e-2
        amp = np.exp(np.mean(np.log(d[law][m]) - (1 - hu) * np.log(eps[m])))
        sel = eps < 0.3
        ax[1].loglog(eps[sel], amp * eps[sel] ** (1 - hu), "-",
                     color=COLORS[law], lw=0.8)
    ax[1].set_xlabel(r"$\Delta t/t_c$")
    ax[1].set_ylabel("completeness $c$")
    ax[1].legend(**LEG)
    tag(ax[1], "(b)")
    e = load("driftless")
    ns = e["Ns"]
    for law in ["gauss", "laplace", "stable1.5", "stable0.8"]:
        ax[2].loglog(ns, e[law], "o", ms=3, mfc="none", color=COLORS[law],
                     label=LABELS[law])
    ax[2].loglog(ns, e["exact"], "k-", lw=0.9, label="exact")
    for law in ["fbm0.3", "fbm0.7"]:
        ax[2].loglog(ns, e[law], "s", ms=2.5, color=COLORS[law],
                     label=LABELS[law])
    ax[2].set_xlabel("layers deposited $N$")
    ax[2].set_ylabel("mean layers preserved")
    ax[2].legend(**LEG)
    tag(ax[2], "(c)")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.40,
                        wspace=0.42)
    save_figure(fig, "fig02_completeness")
    plt.close(fig)
    write_csv("fig02a_ratio", ["eps"] + laws_a + ["spitzer"],
              [eps] + [d[k] / np.sqrt(eps) for k in laws_a]
              + [d["spitzer"] / np.sqrt(eps)])
    write_csv("fig02b_fbm", ["eps", "fbm0.3", "fbm0.5", "fbm0.7"],
              [eps, d["fbm0.3"], d["fbm0.5"], d["fbm0.7"]])


if __name__ == "__main__":
    main()
