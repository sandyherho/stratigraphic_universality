"""Universality and information loss in the stratigraphic record.

Supplementary code for an idealized model in which the preserved
stratigraphic column is the future minimum of a stochastic depositional
surface.
"""
from .generators import (drift_from_epsilon, fgn, increments, noise, stable,
                         surface)
from .record import (ages_on_grid, coherence, column_at, completeness,
                     future_minimum, hiatuses, preserved)
from .theory import (chronometric_information, completeness_asymptote,
                     hiatus_rate, levy_density, longest_hiatus_cdf,
                     mean_records, psi, spitzer_completeness)

__version__ = "0.1.0"

__all__ = [
    "ages_on_grid", "chronometric_information", "coherence", "column_at",
    "completeness", "completeness_asymptote", "drift_from_epsilon", "fgn",
    "future_minimum", "hiatus_rate", "hiatuses", "increments",
    "levy_density", "longest_hiatus_cdf", "mean_records", "noise",
    "preserved", "psi", "spitzer_completeness", "stable", "surface",
]
