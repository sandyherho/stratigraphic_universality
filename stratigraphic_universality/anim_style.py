"""Dark style shared by the animations."""
import os

from matplotlib.animation import PillowWriter

from .io import ANIM
from .plotting import plt

BG = "#0b1220"
PANEL = "#111a2e"
FG = "#e6edf7"
MUTED = "#8fa1bf"
ACCENT = "#ffb347"
CYAN = "#5fd3f3"
ROSE = "#ff6f91"
LIME = "#b5e48c"


def dark_axes(ax):
    """Apply the dark style to an axes."""
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(which="both", colors=FG, labelsize=7)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def new_figure(*args, **kwargs):
    """Create a figure with the dark background."""
    fig, ax = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(BG)
    return fig, ax


def save_gif(anim, name, fps=15, dpi=110):
    """Write an animation to ``outputs/animations/<name>.gif``."""
    path = os.path.join(ANIM, name + ".gif")
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi,
              savefig_kwargs={"facecolor": BG})
    return path
