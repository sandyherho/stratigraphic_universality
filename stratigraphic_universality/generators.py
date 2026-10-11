"""Stochastic generators for the elevation of a depositional surface.

Every law is normalized so that the reduced resolution
``epsilon = dt / t_c`` has the same meaning across laws (see
:func:`drift_from_epsilon`).
"""
import numpy as np

FINITE_VARIANCE = ("gauss", "laplace", "student3")


def stable(alpha, shape, rng):
    """Draw symmetric alpha-stable variates (Chambers-Mallows-Stuck).

    The characteristic function is ``exp(-|k|**alpha)``.
    """
    v = rng.uniform(-np.pi / 2, np.pi / 2, shape)
    w = rng.exponential(1.0, shape)
    if alpha == 1.0:
        return np.tan(v)
    return (np.sin(alpha * v) / np.cos(v) ** (1 / alpha)
            * (np.cos(v - alpha * v) / w) ** ((1 - alpha) / alpha))


def noise(law, n, rng, size=None):
    """Draw independent fluctuations of unit variance (or unit stable scale).

    Parameters
    ----------
    law : str
        ``"gauss"``, ``"laplace"``, ``"student3"`` or ``"stableA"`` with
        ``A`` the stability index, for example ``"stable1.5"``.
    n : int
        Number of steps.
    rng : numpy.random.Generator
        Random number generator.
    size : int, optional
        Number of independent series; a 2D array is returned if given.
    """
    shape = (n,) if size is None else (size, n)
    if law == "gauss":
        return rng.standard_normal(shape)
    if law == "laplace":
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), shape)
    if law == "student3":
        return rng.standard_t(3, shape) / np.sqrt(3.0)
    if law.startswith("stable"):
        return stable(float(law[6:]), shape, rng)
    raise ValueError(law)


def fgn(hurst, n, rng, size=None):
    """Generate unit-variance fractional Gaussian noise (Davies-Harte)."""
    k = np.arange(n + 1)
    g = 0.5 * (np.abs(k + 1) ** (2 * hurst) - 2 * np.abs(k) ** (2 * hurst)
               + np.abs(k - 1) ** (2 * hurst))
    row = np.concatenate([g, g[-2:0:-1]])
    lam = np.fft.fft(row).real
    lam[lam < 0] = 0.0
    m = row.size
    sz = 1 if size is None else size
    z = (rng.standard_normal((sz, m)) + 1j * rng.standard_normal((sz, m)))
    x = np.fft.fft(np.sqrt(lam / m) * z, axis=1).real[:, :n]
    return x[0] if size is None else x


def increments(law, n, rng, size=None):
    """Return fluctuations for any supported law, including ``"fbmH"``."""
    if law.startswith("fbm"):
        return fgn(float(law[3:]), n, rng, size)
    return noise(law, n, rng, size)


def drift_from_epsilon(law, eps):
    """Return the per-step drift for reduced resolution ``eps = dt/t_c``.

    Finite variance: ``eps = 2 mu**2``. Stable: ``eps = mu**(a/(a-1))``.
    Fractional: ``eps = (sqrt(2) mu)**(1/(1-H))``. All reduce to the
    Gaussian definition at ``a = 2`` and ``H = 1/2``.
    """
    if law in FINITE_VARIANCE:
        return np.sqrt(eps / 2.0)
    if law.startswith("stable"):
        a = float(law[6:])
        return eps ** ((a - 1) / a)
    if law.startswith("fbm"):
        h = float(law[3:])
        return eps ** (1 - h) / np.sqrt(2.0)
    raise ValueError(law)


def surface(law, n, eps, rng, size=None):
    """Return the elevation history ``h_n = h_{n-1} + mu + xi_n``."""
    mu = drift_from_epsilon(law, eps) if eps > 0 else 0.0
    return np.cumsum(mu + increments(law, n, rng, size), axis=-1)
