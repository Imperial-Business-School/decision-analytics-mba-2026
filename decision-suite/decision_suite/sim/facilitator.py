"""
dsim.facilitator — the ergonomic layer students actually type.

`dsim.core` defines the rigorous distribution classes (Triangular, Normal,
Uniform, DiscreteUniform) and the Model class. This module adds lowercase
convenience constructors for the distributions, so a notebook cell reads

    load_factor = Parameter("load_factor", base=0.6, distribution=triangular(0.5, 0.6, 0.7))

rather than requiring the capitalized class name every time. Purely sugar —
every function here does nothing but construct a dsim.core class.
"""

from __future__ import annotations

from .core import DiscreteUniform, Empirical, Normal, Triangular, Uniform


def triangular(low: float, mode: float, high: float) -> Triangular:
    """Shorthand for `Triangular(low, mode, high)`."""
    return Triangular(low, mode, high)


def normal(mean: float, std: float) -> Normal:
    """Shorthand for `Normal(mean, std)`."""
    return Normal(mean, std)


def uniform(low: float, high: float) -> Uniform:
    """Shorthand for `Uniform(low, high)`."""
    return Uniform(low, high)


def discrete_uniform(values: list[float]) -> DiscreteUniform:
    """Shorthand for `DiscreteUniform(values)`."""
    return DiscreteUniform(values)


def empirical(points: list[tuple[float, float]]) -> Empirical:
    """Shorthand for `Empirical(points)` — (cumulative_probability, value) pairs."""
    return Empirical(points)
