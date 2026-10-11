"""Animation 3: reading a finished column from base to top.

Left: the age misregistration ``L(z) - z/v``, the error made by dating the
column with its long-term accumulation rate. It rises in jumps, each a
hiatus, and wanders like a random walk; the shaded band is plus or minus one
exact standard deviation. Middle: hiatuses longer than ``ell`` per unit
thickness, converging to the exact Levy measure. Right: the longest hiatus
found so far against the exact median for a column of that thickness.
"""
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.optimize import brentq

from stratigraphic_universality import (drift_from_epsilon, hiatus_rate,
                                        preserved)
from stratigraphic_universality.anim_style import (ACCENT, CYAN, FG, MUTED,
                                                   ROSE, dark_axes,
                                                   new_figure, save_gif)

EPS = 0.01
N = 2 ** 19
FRAMES = 150


def main():
    """Render the hiatus animation."""
    rng = np.random.default_rng(23)
    mu = drift_from_epsilon("gauss", EPS)
    D, v = 0.5, mu
    tc = D / v ** 2
    h = np.cumsum(mu + rng.standard_normal(N))
    idx = np.flatnonzero(preserved(h)[:N - int(50 * tc)])
    hp = h[idx] - h[idx[0]]
    age = (idx - idx[0]) / tc
    mis = age - hp / (v * tc)
    gaps = np.diff(idx)
    gap_top = hp[1:]
    ell = np.logspace(np.log10(0.05 * tc), np.log10(10 * tc), 40)
    exact = hiatus_rate(ell, D, tc)
    zmax = hp[-1]
    zf = np.linspace(0, zmax, 300)
    sd = np.sqrt(2 * tc * zf / v) / tc

    def median_longest(z):
        """Exact median of the longest hiatus in thickness ``z``."""
        return brentq(lambda x: z * hiatus_rate(x, D, tc) - np.log(2),
                      1e-3, 1e7) / tc

    zm = np.linspace(zmax / 40, zmax, 60)
    med = np.array([median_longest(z) for z in zm])

    fig, ax = new_figure(1, 3, figsize=(7.6, 3.4),
                         gridspec_kw={"width_ratios": [1.5, 1, 1]})
    for a in ax:
        dark_axes(a)
    ax[0].fill_between(zf / (v * tc), -sd, sd, color=CYAN, alpha=0.15, lw=0)
    mline, = ax[0].plot([], [], color=CYAN, lw=0.8)
    ax[0].set_xlim(0, zmax / (v * tc))
    ax[0].set_ylim(min(mis.min(), -sd.max()) * 1.1,
                   max(mis.max(), sd.max()) * 1.1)
    ax[0].axhline(0, color=MUTED, lw=0.5)
    ax[0].set_xlabel(r"height in column $z/(v t_c)$")
    ax[0].set_ylabel(r"age misregistration $/t_c$")
    ax[0].text(0.02, 0.95, r"band: $\pm$1 exact s.d.",
               transform=ax[0].transAxes, va="top", color=MUTED,
               fontsize=6.5)
    ax[1].loglog(ell / tc, exact, color=FG, lw=1.0, ls="--", label="exact")
    hpts, = ax[1].plot([], [], "o", ms=2.8, mfc="none", color=ACCENT,
                       label="column so far")
    ax[1].set_xlabel(r"$\ell/t_c$")
    ax[1].set_ylabel(r"hiatuses $>\ell$ per unit thickness")
    ax[1].set_ylim(exact.min() * 0.3, exact.max() * 3)
    leg = ax[1].legend(frameon=False, fontsize=7, loc="lower left")
    for t in leg.get_texts():
        t.set_color(FG)
    ax[2].plot(zm / (v * tc), med, color=FG, lw=1.0, ls="--",
               label="exact median")
    lline, = ax[2].plot([], [], color=ROSE, lw=1.2, label="longest so far")
    ax[2].set_xlim(0, zmax / (v * tc))
    ax[2].set_ylim(0, max(med.max(), gaps.max() / tc) * 1.15)
    ax[2].set_xlabel(r"height in column $z/(v t_c)$")
    ax[2].set_ylabel(r"longest hiatus $/t_c$")
    leg = ax[2].legend(frameon=False, fontsize=7, loc="lower right")
    for t in leg.get_texts():
        t.set_color(FG)
    fig.suptitle("Hiatuses are the jumps of the age-depth map",
                 color=FG, fontsize=9.5)
    fig.text(0.5, 0.015, r"Gaussian surface, $\Delta t/t_c=0.01$; the "
             "column is read upward from its base", ha="center", color=MUTED,
             fontsize=7)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.2,
                        wspace=0.42)
    tops = np.linspace(zmax / 40, zmax, FRAMES)
    run_max = np.maximum.accumulate(gaps) / tc

    def update(k):
        z = tops[k]
        j = np.searchsorted(hp, z, "right")
        mline.set_data(hp[:j] / (v * tc), mis[:j])
        g = gaps[gap_top <= z]
        rate = np.array([(g >= L).sum() for L in ell]) / max(z, 1e-9)
        ok = rate > 0
        hpts.set_data(ell[ok] / tc, rate[ok])
        jj = max(np.searchsorted(gap_top, z, "right"), 1)
        lline.set_data(gap_top[:jj] / (v * tc), run_max[:jj])
        return mline, hpts, lline

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    save_gif(anim, "anim03_hiatus")
    print("saved anim03_hiatus; final longest", round(run_max[-1], 2),
          "exact median", round(med[-1], 2))


if __name__ == "__main__":
    main()
