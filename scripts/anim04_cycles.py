"""Animation 4: an astronomical cycle as the column records it.

Top: two stripes for the same stretch of column. The upper stripe is the
forcing that a complete record would show at the nominal age ``z/v``; the
lower stripe is the forcing actually stored in the preserved layers. The
window climbs through the column and the two drift out of step. Bottom left:
the phase lag of the record, a random walk in height, with the exact band of
one standard deviation. Bottom right: the depth spectrum of the record,
averaged over independent columns as they accumulate, against the exact
expected periodogram (the Lorentzian line seen through a finite segment).
The color scale is the forcing value, which spans exactly minus one to one.
"""
import numpy as np
from matplotlib.animation import FuncAnimation

from stratigraphic_universality import (ages_on_grid, drift_from_epsilon,
                                        psi)
from stratigraphic_universality.anim_style import (ACCENT, CYAN, FG, MUTED,
                                                   dark_axes, save_gif)
from stratigraphic_universality.plotting import plt

EPS = 0.005
U = 0.05
N = 2 ** 21
FRAMES = 160
WINDOW = 12.0
COLUMNS = 40


def main():
    """Render the cycle animation."""
    rng = np.random.default_rng(24)
    tcs = 1 / EPS
    mu = drift_from_epsilon("gauss", EPS)
    D, v = 0.5, mu
    om = U / tcs
    k0 = om / v
    lam = 2 * np.pi / k0
    dz = lam / 64
    h = np.cumsum(mu + rng.standard_normal(N))
    ages = ages_on_grid(h, dz, int(60 * tcs))
    ages -= ages[0]
    z = np.arange(len(ages)) * dz
    ncyc = z / lam
    rec = np.cos(om * ages)
    nominal = np.cos(k0 * z)
    phase = om * ages - k0 * z
    p = psi(om, v, D)
    seg = 2048
    kk = 2 * np.pi * np.fft.fftfreq(seg, dz)
    order = np.argsort(kk)
    kk = kk[order]
    band = (kk > 0.5 * k0) & (kk < 1.5 * k0)
    m = np.arange(-seg + 1, seg)
    cm = np.exp(-np.abs(m) * dz * p)
    cm[m < 0] = np.conj(cm[m < 0])
    lor = np.array([(dz / (2 * np.pi)) * np.real(np.sum(
        (1 - np.abs(m) / seg) * cm * np.exp(-1j * q * m * dz)))
        for q in kk[band]])
    psds = []
    for _ in range(COLUMNS):
        hh = np.cumsum(mu + rng.standard_normal(N))
        aa = ages_on_grid(hh, dz, int(60 * tcs))
        for j in range(len(aa) // seg):
            x = np.exp(1j * om * aa[j * seg:(j + 1) * seg])
            psds.append((np.abs(np.fft.fft(x) * dz) ** 2
                         / (2 * np.pi * seg * dz))[order][band])
    psds = np.array(psds)
    cum = np.cumsum(psds, axis=0) / np.arange(1, len(psds) + 1)[:, None]

    fig = plt.figure(figsize=(7.4, 4.6))
    fig.patch.set_facecolor("#0b1220")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.2], hspace=0.55,
                          wspace=0.3, left=0.08, right=0.93, top=0.87,
                          bottom=0.12)
    axs = fig.add_subplot(gs[0, :])
    axp = fig.add_subplot(gs[1, 0])
    axk = fig.add_subplot(gs[1, 1])
    for a in (axs, axp, axk):
        dark_axes(a)
    npts = int(WINDOW * 64)
    img = axs.imshow(np.zeros((2, npts)), aspect="auto", cmap="RdYlBu_r",
                     vmin=-1, vmax=1, extent=(0, WINDOW, 0, 2),
                     interpolation="nearest")
    axs.set_yticks([0.5, 1.5])
    axs.set_yticklabels(["recorded", "complete"], fontsize=7)
    axs.set_xlabel("nominal cycles from window base")
    stripe_txt = axs.set_title("", fontsize=8, color=FG)
    cb = fig.colorbar(img, ax=axs, fraction=0.03, pad=0.01)
    cb.set_label("forcing", color=FG, fontsize=7)
    cb.ax.tick_params(colors=FG, labelsize=6)
    total = ncyc[-1]
    sdv = np.sqrt(4 * np.pi * U * np.linspace(0, total, 200))
    axp.fill_between(np.linspace(0, total, 200), -sdv, sdv, color=CYAN,
                     alpha=0.15, lw=0)
    pl, = axp.plot([], [], color=CYAN, lw=0.8)
    axp.set_xlim(0, total)
    axp.set_ylim(min(phase.min(), -sdv.max()) * 1.1,
                 max(phase.max(), sdv.max()) * 1.1)
    axp.set_xlabel("nominal cycles")
    axp.set_ylabel("phase lag (rad)")
    axp.text(0.02, 0.95, r"band: $\pm$1 exact s.d.", transform=axp.transAxes,
             va="top", color=MUTED, fontsize=6.5)
    axk.semilogy(kk[band] / k0, lor, color=FG, lw=1.0, ls="--",
                 label="exact")
    sl, = axk.semilogy([], [], color=ACCENT, lw=1.0, label="simulated")
    ntxt = axk.text(0.02, 0.95, "", transform=axk.transAxes, va="top",
                    color=MUTED, fontsize=6.5)
    axk.axvline(1.0, color=MUTED, lw=0.6, ls=":")
    axk.set_xlim(0.5, 1.5)
    axk.set_ylim(lor.min() * 0.3, lor.max() * 3)
    axk.set_xlabel(r"wavenumber $/k_0$")
    axk.set_ylabel("expected periodogram")
    leg = axk.legend(frameon=False, fontsize=7, loc="upper right")
    for t in leg.get_texts():
        t.set_color(FG)
    fig.suptitle("Unmarked hiatuses make an astronomical cycle lose its "
                 "phase in depth", color=FG, fontsize=9.5)
    fig.text(0.5, 0.015, r"Gaussian surface, $\omega t_c=0.05$: about "
             "three coherent cycles", ha="center", color=MUTED, fontsize=7)
    starts = np.linspace(0, len(ages) - npts - 1, FRAMES).astype(int)

    def update(k):
        s0 = starts[k]
        img.set_data(np.vstack([nominal[s0:s0 + npts], rec[s0:s0 + npts]]))
        stripe_txt.set_text("column window starting %.0f cycles above the "
                            "base" % ncyc[s0])
        top = s0 + npts
        pl.set_data(ncyc[:top], phase[:top])
        j = int(np.ceil((k + 1) / FRAMES * len(psds))) - 1
        sl.set_data(kk[band] / k0, cum[j])
        ntxt.set_text("%d segments of 32 cycles" % (j + 1))
        return img, pl, sl, ntxt

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    save_gif(anim, "anim04_cycles")
    err = np.max(np.abs(cum[-1] / lor - 1)[lor > lor.max() / 10])
    print("saved anim04_cycles; segments", len(psds), "cycles",
          round(total, 1), "max rel. error near peak", round(err, 3))


if __name__ == "__main__":
    main()
