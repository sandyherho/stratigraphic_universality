"""Animation 1: a column built and eroded by a fluctuating surface.

Left: the surface elevation through time, with the layers that survive at
the current moment colored by their age. Right: the column that exists at
that moment, colored by the age of the material at each height. White ticks
mark hiatuses longer than the crossover time. The color scale spans exactly
the ages present in the simulation.
"""
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

from stratigraphic_universality import drift_from_epsilon, preserved
from stratigraphic_universality.anim_style import (FG, MUTED, dark_axes,
                                                   new_figure, save_gif)

EPS = 0.02
N = 1500
FRAMES = 150


def main():
    """Render the column animation."""
    rng = np.random.default_rng(21)
    tc = 1 / EPS
    mu = drift_from_epsilon("gauss", EPS)
    h = np.cumsum(mu + rng.standard_normal(N))
    t = np.arange(N) / tc
    norm = Normalize(vmin=0.0, vmax=t[-1])
    cmap = "turbo"
    zlo, zhi = h.min() - 1, h.max() + 1
    zgrid = np.linspace(zlo, zhi, 600)

    fig, (ax, axc) = new_figure(1, 2, figsize=(7.4, 3.9),
                                gridspec_kw={"width_ratios": [3.2, 1]})
    dark_axes(ax)
    dark_axes(axc)
    line, = ax.plot([], [], color=MUTED, lw=0.7)
    pts = ax.scatter([], [], s=5, c=[], cmap=cmap, norm=norm, zorder=3)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(zlo, zhi)
    ax.set_xlabel(r"time $t/t_c$")
    ax.set_ylabel(r"elevation $h/\sigma$")
    info = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top",
                   color=FG, fontsize=7.5, family="monospace")
    img = axc.imshow(np.full((len(zgrid), 1), np.nan), aspect="auto",
                     origin="lower", extent=(0, 1, zlo, zhi), cmap=cmap,
                     norm=norm, interpolation="nearest")
    axc.set_xticks([])
    axc.set_ylabel(r"height in column $z/\sigma$")
    axc.set_title("column now", fontsize=8, color=FG)
    ticks = axc.hlines([], 0, 1, color="w", lw=0.8)
    cb = fig.colorbar(img, ax=axc, fraction=0.12, pad=0.08)
    cb.set_label(r"age of layer $t/t_c$", color=FG, fontsize=7)
    cb.ax.tick_params(colors=FG, labelsize=6.5)
    cb.outline.set_edgecolor(MUTED)
    fig.suptitle("A stratigraphic column keeps only the future minimum of "
                 "its surface", color=FG, fontsize=9.5)
    fig.text(0.5, 0.015, r"Gaussian surface, $\Delta t/t_c = 0.02$; white "
             r"ticks mark hiatuses longer than $t_c$", ha="center",
             color=MUTED, fontsize=7)
    fig.subplots_adjust(left=0.08, right=0.93, top=0.88, bottom=0.15,
                        wspace=0.35)
    steps = np.linspace(20, N - 1, FRAMES).astype(int)

    def update(k):
        now = steps[k]
        hh = h[:now + 1]
        idx = np.flatnonzero(preserved(hh))
        line.set_data(t[:now + 1], hh)
        pts.set_offsets(np.c_[t[idx], hh[idx]])
        pts.set_array(t[idx])
        hp = hh[idx]
        col = np.full(len(zgrid), np.nan)
        inside = (zgrid >= hp[0]) & (zgrid <= hp[-1])
        j = np.searchsorted(hp, zgrid[inside], "left")
        col[inside] = t[idx[np.clip(j, 0, len(idx) - 1)]]
        img.set_data(col[:, None])
        gaps = np.diff(idx)
        big = np.flatnonzero(gaps > tc)
        ticks.set_segments([[[0, hp[i]], [1, hp[i]]] for i in big])
        longest = gaps.max(initial=0) / tc
        info.set_text(f"elapsed       {t[now]:6.1f} t_c\n"
                      f"layers kept   {len(idx):6d} of {now + 1}\n"
                      f"completeness  {len(idx) / (now + 1):6.2f}\n"
                      f"longest gap   {longest:6.2f} t_c")
        return line, pts, img, info, ticks

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    save_gif(anim, "anim01_column")
    print("saved anim01_column; color range", norm.vmin, norm.vmax)


if __name__ == "__main__":
    main()
