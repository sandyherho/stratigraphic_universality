"""Exact results used to verify the simulations.

Notation follows the manuscript: ``v`` is the long-term accumulation rate,
``D`` the fluctuation diffusivity, ``t_c = D / v**2`` the crossover time and
``eps = dt / t_c`` the reduced resolution.
"""
import numpy as np
from scipy.special import gammaincc, gammaln, ndtr, zeta


def spitzer_completeness(a, nmax=None):
    """Exact completeness for Gaussian steps with ``mu/sigma = a``."""
    if nmax is None:
        nmax = int(max(200, 60.0 / a ** 2))
    n = np.arange(1, nmax + 1)
    return np.exp(-np.sum(ndtr(-a * np.sqrt(n)) / n))


def completeness_asymptote(eps):
    """Two-term small-resolution form, ``sqrt(eps) exp(zeta(1/2) ...)``."""
    eps = np.asarray(eps, float)
    return np.sqrt(eps) * np.exp(zeta(0.5) * np.sqrt(eps)
                                 / (2 * np.sqrt(np.pi)))


def mean_records(n):
    """Mean number of records of an ``n``-step symmetric walk (with start)."""
    n = np.asarray(n, float)
    return np.exp(np.log(2 * n + 1) + gammaln(2 * n + 1)
                  - 2 * gammaln(n + 1) - 2 * n * np.log(2.0))


def levy_density(tau, D, tc):
    """Levy measure density of the age-depth subordinator."""
    return (4 * np.pi * D * tau ** 3) ** -0.5 * np.exp(-tau / (4 * tc))


def upper_gamma_minus_half(y):
    """Upper incomplete gamma ``Gamma(-1/2, y)`` for ``y > 0``."""
    y = np.asarray(y, float)
    return 2 * np.exp(-y) / np.sqrt(y) - 2 * np.sqrt(np.pi) * gammaincc(0.5, y)


def hiatus_rate(ell, D, tc):
    """Return the number of hiatuses longer than ``ell`` per thickness."""
    return (upper_gamma_minus_half(np.asarray(ell) / (4 * tc))
            / np.sqrt(4 * np.pi * D * 4 * tc))


def longest_hiatus_cdf(ell, thickness, D, tc):
    """``P(longest hiatus < ell)`` in a column of the given thickness."""
    return np.exp(-thickness * hiatus_rate(ell, D, tc))


def psi(omega, v, D):
    """Characteristic exponent: ``E exp(i w dL) = exp(-dz Psi(w))``."""
    tc = D / v ** 2
    return (v / (2 * D)) * (np.sqrt(1 - 4j * omega * tc + 0j) - 1)


def chronometric_information(ncycles, u, harmonics=600, points=8001):
    """Chronometric information in bits between horizons ``ncycles`` apart.

    ``u = omega t_c``; separation is in nominal cycles of the forcing.
    """
    k = np.arange(1, harmonics + 1)
    phi = np.linspace(-np.pi, np.pi, points)
    ck = np.exp(-ncycles * (np.pi / u) * (np.sqrt(1 - 4j * k * u) - 1))
    f = (1 + 2 * np.real(np.exp(-1j * np.outer(phi, k)) @ ck)) / (2 * np.pi)
    f = np.clip(f, 1e-300, None)
    return np.trapezoid(f * np.log2(2 * np.pi * f), phi)
