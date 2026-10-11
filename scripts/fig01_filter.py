"""Figure 1: the stratigraphic filter and the resulting column."""
import numpy as np

from stratigraphic_universality import (drift_from_epsilon, future_minimum,
                                        preserved)
from stratigraphic_universality.io import save_figure, write_csv
from stratigraphic_universality.plotting import COLORS, plt, tag


def main():
    """Draw one Gaussian surface, its future minimum and the column."""
    rng = np.random.default_rng(11)
    eps, n = 0.05, 700
    mu = drift_from_epsilon("gauss", eps)
    h = np.cumsum(mu + rng.standard_normal(n))
    p = preserved(h)
    fut = future_minimum(h)
    t = np.arange(n)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6),
                           gridspec_kw={"width_ratios": [2.2, 1]})
    ax[0].plot(t, h, color="0.6", lw=0.7, label=r"surface $h(t)$")
    ax[0].plot(t, fut, color=COLORS["gauss"], lw=1.2,
               label=r"future minimum $m(t)$")
    ax[0].plot(t[p], h[p], ".", ms=2.2, color="k", label="preserved")
    ax[0].set_xlabel(r"time (steps $\Delta t$)")
    ax[0].set_ylabel(r"elevation $h/\sigma$")
    ax[0].legend(frameon=False, loc="upper left")
    ax[0].text(0.97, 0.05, f"$\\Delta t/t_c={eps}$, completeness "
               f"{p.mean():.2f}", transform=ax[0].transAxes, ha="right")
    tag(ax[0], "(a)")
    idx = np.flatnonzero(p)
    hp = h[idx]
    ax[1].step(idx, hp, where="post", color=COLORS["gauss"], lw=1.2,
               label=r"recorded age $L(z)$")
    ax[1].plot([0, n], [0, mu * n], "--", color="0.4", lw=0.8,
               label=r"nominal $z/v$")
    gaps = np.diff(idx)
    gi = np.argmax(gaps)
    ax[1].annotate("longest hiatus", xy=(idx[gi] + gaps[gi] / 2, hp[gi]),
                   xytext=(30, hp[gi] + 8), fontsize=6.5,
                   arrowprops=dict(arrowstyle="->", lw=0.6))
    ax[1].set_xlabel("age of layer (steps)")
    ax[1].set_ylabel(r"height in column $z/\sigma$")
    ax[1].legend(frameon=False, loc="lower right")
    tag(ax[1], "(b)")
    fig.tight_layout()
    save_figure(fig, "fig01_filter")
    plt.close(fig)
    write_csv("fig01a_surface", ["step", "elevation", "future_minimum",
                                 "preserved"], [t, h, fut, p.astype(int)])
    write_csv("fig01b_column", ["age_step", "height"], [idx, hp])


if __name__ == "__main__":
    main()
