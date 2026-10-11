"""Shared plotting style, colors and labels."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "lines.linewidth": 1.1,
    "font.family": "serif", "mathtext.fontset": "cm",
})

COLORS = {
    "gauss": "#1f4e79", "laplace": "#2e8b57", "student3": "#8b6914",
    "stable1.5": "#b22222", "stable1.2": "#e07b39", "stable0.8": "#7b3f9e",
    "fbm0.3": "#00838f", "fbm0.5": "#555555", "fbm0.7": "#c2185b",
}

LABELS = {
    "gauss": "Gaussian", "laplace": "Laplace", "student3": r"Student $t_3$",
    "stable1.5": r"stable $\alpha=1.5$", "stable1.2": r"stable $\alpha=1.2$",
    "stable0.8": r"stable $\alpha=0.8$", "fbm0.3": "fBm $H=0.3$",
    "fbm0.5": "fBm $H=0.5$", "fbm0.7": "fBm $H=0.7$",
}


def tag(ax, s):
    """Write a bold panel label above the top-left corner."""
    ax.set_title(s, loc="left", fontsize=8, fontweight="bold", pad=3)
