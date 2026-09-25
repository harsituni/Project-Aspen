"""Chapman-Richards stand growth.

    V(T) = a * (1 - e^(-b*T))^c          merchantable volume, m3/ha
    MAI(T) = V(T) / T                    mean annual increment
    CAI(T) = dV/dT                       current annual increment

The stand's mean annual increment peaks exactly where the current annual
increment crosses it from above; that crossing is the biological rotation
age a forester would quote, and it is the benchmark the economic rotations
in rotation.py are measured against. Monetary figures are real CAD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass(frozen=True)
class GrowthParams:
    """Chapman-Richards parameters for one aggregate yield curve."""

    a: float  # asymptotic volume, m3/ha
    b: float  # growth rate, per year
    c: float  # shape, dimensionless
    max_age: int = 150

    @classmethod
    def from_config(cls) -> "GrowthParams":
        return cls(
            a=config.num("growth.chapman_richards_a"),
            b=config.num("growth.chapman_richards_b"),
            c=config.num("growth.chapman_richards_c"),
            max_age=int(config.num("growth.max_modelled_age")),
        )


def volume(age, p: GrowthParams):
    """Merchantable volume in m3/ha at the given age (scalar or array)."""
    t = np.asarray(age, dtype=float)
    v = p.a * np.power(1.0 - np.exp(-p.b * np.maximum(t, 0.0)), p.c)
    return float(v) if np.isscalar(age) or np.ndim(age) == 0 else v


def mai(age, p: GrowthParams):
    """Mean annual increment, m3/ha/yr. Defined as 0 at age 0."""
    t = np.asarray(age, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(t > 0, volume(t, p) / np.where(t > 0, t, 1.0), 0.0)
    return float(out) if np.isscalar(age) or np.ndim(age) == 0 else out


def cai(age, p: GrowthParams):
    """Current annual increment, m3/ha/yr, i.e. the analytic derivative dV/dT."""
    t = np.asarray(age, dtype=float)
    u = np.exp(-p.b * np.maximum(t, 0.0))
    out = p.a * p.b * p.c * np.power(1.0 - u, p.c - 1.0) * u
    return float(out) if np.isscalar(age) or np.ndim(age) == 0 else out


def age_grid(p: GrowthParams, step: float = 1.0) -> np.ndarray:
    """Ages from `step` to max_age inclusive."""
    n = int(round(p.max_age / step))
    return np.linspace(step, p.max_age, n)


def volume_table(p: GrowthParams, step: float = 1.0):
    """(ages, volume, MAI, CAI) arrays for plotting and for the Excel table."""
    ages = age_grid(p, step)
    return ages, volume(ages, p), mai(ages, p), cai(ages, p)


def max_mai_age(p: GrowthParams, step: float = 0.01) -> float:
    """Age at which mean annual increment culminates (fine grid search)."""
    ages = age_grid(p, step)
    return float(ages[int(np.argmax(mai(ages, p)))])
