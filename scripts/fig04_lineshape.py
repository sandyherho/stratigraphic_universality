"""Figure 4: a periodic forcing seen in depth, and age misregistration."""
import numpy as np

from stratigraphic_universality import psi
from stratigraphic_universality.io import load, save_figure
from stratigraphic_universality.plotting import COLORS, LABELS, plt, tag


def main():
    """Draw line width, apparent wavenumber and misregistration scaling."""
    h5 = load("coherence")
    us = h5["us"]
    uf = np.logspace(-1.5, 0.6, 200)
    pf = psi(uf, 1.0, 1.0)
    sel = us >= 0.1
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))
    ax[0].loglog(uf, pf.real / uf, "k-", lw=0.9,
                 label=r"exact, $\mathrm{Re}\,\Psi/k_0$")
    ax[0].loglog(uf, uf, "k:", lw=0.9, label=r"$\omega t_c$")
    for law in ["gauss", "laplace", "student3"]:
        ax[0].loglog(us[sel], h5["fa_" + law][sel], "o", ms=3, mfc="none",
                     color=COLORS[law], label=LABELS[law])
    ax[0].set_xlabel(r"$\omega t_c$")
    ax[0].set_ylabel(r"line half-width / $k_0$")
    ax[0].legend(frameon=False, loc="upper left")
    tag(ax[0], "(a)")
    ax[1].semilogx(uf, -pf.imag / uf, "k-", lw=0.9, label="exact")
    ax[1].semilogx(uf, 1 - 2 * uf ** 2, "k:", lw=0.9,
                   label=r"$1-2(\omega t_c)^2$")
    for law in ["gauss", "laplace", "student3"]:
        ax[1].semilogx(us[sel], h5["fb_" + law][sel], "o", ms=3, mfc="none",
                       color=COLORS[law])
    ax[1].set_ylim(0.3, 1.05)
    ax[1].set_xlabel(r"$\omega t_c$")
    ax[1].set_ylabel(r"apparent wavenumber / $k_0$")
    ax[1].legend(frameon=False, loc="lower left")
    tag(ax[1], "(b)")
    m6 = load("misregistration")
    for law in ["gauss", "laplace", "stable1.5", "fbm0.3", "fbm0.7"]:
        t = m6["tau_" + law]
        q = m6["iqr_" + law]
        mm = t > 0.5
        fbm = law.startswith("fbm")
        ax[2].loglog(t[mm], q[mm], "s" if fbm else "o", ms=2.5,
                     mfc=COLORS[law] if fbm else "none", color=COLORS[law],
                     label=LABELS[law])
    ax[2].set_xlabel(r"$\Delta z/(v t_c)$")
    ax[2].set_ylabel(r"IQR of age misregistration $/t_c$")
    ax[2].legend(frameon=False, loc="lower right")
    tag(ax[2], "(c)")
    fig.tight_layout()
    save_figure(fig, "fig04_lineshape")
    plt.close(fig)


if __name__ == "__main__":
    main()
