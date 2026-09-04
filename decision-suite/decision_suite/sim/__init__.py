"""
decision_suite.sim (formerly dsim) — a small scaffold for teaching Monte
Carlo simulation and decision analysis: one Model class wrapping a
student's own model function, plus best-case/worst-case and Monte Carlo
simulation capabilities.

Deliberately minimal — see SIM_API_MANUAL.md for the "what's a package method
vs. what's composed in the notebook" test this package's surface follows.

    from decision_suite.sim import Model, Parameter, Triangular, Normal, Uniform, DiscreteUniform, Empirical
    from decision_suite.sim import triangular, normal, uniform, discrete_uniform, empirical  # lowercase shorthand

© 2026 Reza Skandari and Imperial College Business School. Internal,
educational use only, see ../LICENSE.md.
"""

from .core import (
    # Exceptions
    DsimError,
    ValidationError,
    # Distributions
    Triangular,
    Normal,
    Uniform,
    DiscreteUniform,
    Empirical,
    # Parameters & scenarios
    Parameter,
    Scenario,
    # Results
    RunResult,
    ScenarioComparison,
    SimulationResult,
    # Model
    Model,
)

# Lowercase convenience constructors
from .facilitator import (
    triangular,
    normal,
    uniform,
    discrete_uniform,
    empirical,
)

__all__ = [
    "DsimError", "ValidationError",
    "Triangular", "Normal", "Uniform", "DiscreteUniform", "Empirical",
    "Parameter", "Scenario",
    "RunResult", "ScenarioComparison",
    "SimulationResult",
    "Model",
    "triangular", "normal", "uniform", "discrete_uniform", "empirical",
]
