"""The stratigraphic filter and the observables of the preserved column."""
import numpy as np


def preserved(h):
    """Mark layers that survive: ``h_n < min_{m>n} h_m`` (strict).

    Works on the last axis, so a 2D array holds independent surfaces.
    """
    fut = np.minimum.accumulate(h[..., ::-1], axis=-1)[..., ::-1]
    out = np.ones(h.shape, bool)
    out[..., :-1] = h[..., :-1] < fut[..., 1:]
    return out


def future_minimum(h):
    """Return the future minimum ``m_n = min_{m>=n} h_m``."""
    return np.minimum.accumulate(h[::-1])[::-1]


def column_at(h, now):
    """Return the column that exists at step ``now`` (indices of layers).

    A layer deposited at ``n <= now`` is present at ``now`` if every later
    elevation up to ``now`` lies strictly above it.
    """
    return np.flatnonzero(preserved(h[:now + 1]))


def completeness(h, cut):
    """Fraction of preserved layers, discarding the last ``cut`` steps."""
    p = preserved(h)
    return p[:len(h) - cut].mean()


def hiatuses(idx):
    """Durations (in steps) between consecutive preserved layers."""
    return np.diff(idx)


def ages_on_grid(h, dz, cut):
    """Age ``L(z)`` sampled on a uniform height grid of spacing ``dz``.

    Each height receives the age of the preserved layer just below it.
    """
    p = preserved(h)
    idx = np.flatnonzero(p[:len(h) - cut])
    hp = h[idx]
    z = np.arange(hp[0], hp[-1], dz)
    return idx[np.searchsorted(hp, z, "right") - 1].astype(float)


def coherence(ages, omega, lags):
    """Complex coherence ``E exp(i omega [L(z+dz) - L(z)])`` at grid lags."""
    e = np.exp(1j * omega * ages)
    return np.array([np.mean(e[k:] * np.conj(e[:-k])) for k in lags])
