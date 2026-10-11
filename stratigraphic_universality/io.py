"""Output locations and small helpers shared by the scripts."""
import csv
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUT, "figures")
ANIM = os.path.join(OUT, "animations")
DATA = os.path.join(OUT, "data")
REPORT = os.path.join(OUT, "reports")
CACHE = os.path.join(OUT, "cache")

for _d in (FIG, ANIM, DATA, REPORT, CACHE):
    os.makedirs(_d, exist_ok=True)


def cache_path(name):
    """Path of a cached ``.npz`` result."""
    return os.path.join(CACHE, name + ".npz")


def load(name):
    """Load a cached result as a dictionary."""
    with np.load(cache_path(name)) as d:
        return {k: d[k] for k in d.files}


def save(name, **arrays):
    """Save a result to the cache."""
    np.savez(cache_path(name), **arrays)


def write_csv(name, header, columns):
    """Write equal-length columns to ``outputs/data/<name>.csv``."""
    path = os.path.join(DATA, name + ".csv")
    rows = zip(*[np.asarray(c).tolist() for c in columns])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def save_figure(fig, name):
    """Save a figure as vector PDF and 600 dpi PNG."""
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=600)
