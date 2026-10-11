"""Regenerate every result, figure, report and animation in order.

Usage::

    python scripts/run_all.py            # everything
    python scripts/run_all.py --no-anim  # skip the animations

The computations take roughly half an hour on two cores; cached results in
``outputs/cache`` are reused by the figure and report scripts.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

COMPUTE = ["compute_completeness", "compute_driftless", "compute_hiatus",
           "compute_longest", "compute_coherence", "compute_misregistration"]
FIGURES = ["fig01_filter", "fig02_completeness", "fig03_hiatus",
           "fig04_lineshape", "fig05_information"]
ANIMATIONS = ["anim01_column", "anim02_universality", "anim03_hiatus",
              "anim04_cycles"]


def run(name):
    """Run one script and report its duration."""
    t0 = time.time()
    subprocess.run([sys.executable, os.path.join(HERE, name + ".py")],
                   check=True)
    print(f"[{name}] {time.time() - t0:.0f} s", flush=True)


def main():
    """Run the pipeline."""
    steps = COMPUTE + FIGURES + ["make_reports"]
    if "--no-anim" not in sys.argv:
        steps += ANIMATIONS
    for name in steps:
        run(name)


if __name__ == "__main__":
    main()
