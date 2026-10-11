"""Animation 2: four very different surfaces, one record statistic.

Top: driftless surfaces with Gaussian, Laplace and two symmetric stable
fluctuation laws, with the layers that survive at the current moment. The
stable surfaces are dominated by rare, very large jumps. Bottom: the mean
number of surviving layers over 1000 surfaces of each law, against the exact
Sparre Andersen result, which does not depend on the law.
"""
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import NullFormatter, ScalarFormatter

from stratigraphic_universality import increments, mean_records, preserved
from stratigraphic_universality.anim_style import (FG, MUTED, dark_axes,
                                                   save_gif)
from stratigraphic_universality.plotting import plt

LAWS = [("gauss", "Gaussian", "#5fd3f3"), ("laplace", "Laplace", "#b5e48c"),
        ("stable1.5", r"stable $\alpha=1.5$", "#ffb347"),
        ("stable0.8", r"stable $\alpha=0.8$", "#ff6f91")]
N = 2000
M = 1000
FRAMES = 140


def main():
    """Render the universality animation."""
    rng = np.random.default_rng(22)
    steps = np.unique(np.linspace(10, N, FRAMES).astype(int))
    single, mean = {}, {}
    for law, _, _ in LAWS:
        single[law] = np.cumsum(increments(law, N, rng))
        ens = np.cumsum(increments(law, N, rng, size=M), axis=1)
        mean[law] = np.array([preserved(ens[:, :s]).sum(axis=1).mean()
                              for s in steps])
    exact = mean_records(steps - 1)

    fig = plt.figure(figsize=(7.4, 4.6))
    fig.patch.set_facecolor("#0b1220")
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.45,
                          wspace=0.32, left=0.07, right=0.98, top=0.88,
                          bottom=0.14)
    top = [fig.add_subplot(gs[0, i]) for i in range(4)]
    bot = fig.add_subplot(gs[1, :])
    lines, dots = [], []
    for ax, (law, lab, col) in zip(top, LAWS):
        dark_axes(ax)
        h = single[law]
        lo, hi = np.percentile(h, [0, 100])
        pad = 0.05 * (hi - lo)
        ax.set_xlim(0, N)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(lab, fontsize=8, color=col)
        ax.set_xticks([0, N // 2, N])
        ax.tick_params(labelsize=6)
        ln, = ax.plot([], [], color=MUTED, lw=0.6)
        dt, = ax.plot([], [], "o", ms=1.8, color=col)
        lines.append(ln)
        dots.append(dt)
    top[0].set_ylabel("elevation", fontsize=7)
    dark_axes(bot)
    bot.plot(steps, exact, color=FG, lw=1.0, ls="--",
             label="exact, any symmetric law")
    curves = []
    for law, lab, col in LAWS:
        c, = bot.plot([], [], "o", ms=2.6, mfc="none", color=col, label=lab)
        curves.append(c)
    bot.set_xscale("log")
    bot.set_yscale("log")
    bot.set_xlim(steps[0] * 0.9, N * 1.1)
    bot.set_ylim(2, 70)
    bot.yaxis.set_major_formatter(ScalarFormatter())
    bot.yaxis.set_minor_formatter(NullFormatter())
    bot.set_yticks([3, 10, 30])
    bot.set_xlabel("layers deposited")
    bot.set_ylabel("mean layers surviving")
    leg = bot.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    for txt in leg.get_texts():
        txt.set_color(FG)
    fig.suptitle("Wildly different erosion, identical record statistics",
                 color=FG, fontsize=9.5)
    fig.text(0.5, 0.012, "no net accumulation; dots on the surfaces are the "
             "layers that survive at the current moment", ha="center",
             color=MUTED, fontsize=7)

    def update(k):
        s = steps[k]
        for law_i, (law, _, _) in enumerate(LAWS):
            h = single[law][:s]
            idx = np.flatnonzero(preserved(h))
            lines[law_i].set_data(np.arange(s), h)
            dots[law_i].set_data(idx, h[idx])
            curves[law_i].set_data(steps[:k + 1], mean[law][:k + 1])
        return lines + dots + curves

    anim = FuncAnimation(fig, update, frames=len(steps), blit=False)
    save_gif(anim, "anim02_universality")
    dev = max(np.max(np.abs(mean[law] / exact - 1)) for law, _, _ in LAWS)
    print("saved anim02_universality; max relative deviation", round(dev, 3))


if __name__ == "__main__":
    main()
