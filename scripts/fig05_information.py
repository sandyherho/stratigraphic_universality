"""Figure 5: chronometric information and the coherence horizon."""
import numpy as np

from stratigraphic_universality import chronometric_information
from stratigraphic_universality.io import save_figure, write_csv
from stratigraphic_universality.plotting import plt, tag

US = [(0.003, "#1f4e79"), (0.01, "#2e8b57"), (0.03, "#8b6914"),
      (0.1, "#b22222")]
PERIODS = [(20, "precession"), (41, "obliquity"),
           (100, "short eccentricity"), (405, "long eccentricity")]


def main():
    """Draw information decay and the coherent-cycle map."""
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.5))
    nc = np.logspace(-1, 3, 60)
    cols = [nc]
    for u, col in US:
        info = [chronometric_information(n, u, points=4001,
                                         harmonics=400) for n in nc]
        cols.append(info)
        ax[0].semilogx(nc, info, color=col, label=f"$\\omega t_c={u}$")
    ax[0].set_xlabel(r"depth separation (nominal cycles $\Delta z\,k_0/2\pi$)")
    ax[0].set_ylabel("chronometric information (bits)")
    ax[0].legend(frameon=False)
    ax[0].set_ylim(0, None)
    tag(ax[0], "(a)")
    tcg = np.logspace(-3, 1, 200)
    pg = np.logspace(np.log10(5), np.log10(2000), 200)
    tt, pp = np.meshgrid(tcg, pg)
    ncoh = pp / (4 * np.pi ** 2 * tt)
    cs = ax[1].contourf(tt, pp, np.log10(ncoh),
                        levels=np.arange(-1, 5.01, 0.5), cmap="cividis",
                        extend="both")
    ax[1].contour(tt, pp, np.log10(ncoh), levels=[0, 1, 2], colors="w",
                  linewidths=0.7)
    for period, lab in PERIODS:
        ax[1].axhline(period, color="w", lw=0.5, ls="--")
        ax[1].text(1.2e-3, period * 1.07, lab, color="w", fontsize=6)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$t_c$ (kyr)")
    ax[1].set_ylabel("forcing period $P$ (kyr)")
    cb = fig.colorbar(cs, ax=ax[1])
    cb.set_label(r"$\log_{10}$ coherent cycles $N_\mathrm{coh}$")
    tag(ax[1], "(b)")
    fig.tight_layout()
    save_figure(fig, "fig05_information")
    plt.close(fig)
    write_csv("fig05a_information", ["ncycles"]
              + ["bits_u%g" % u for u, _ in US], cols)


if __name__ == "__main__":
    main()
