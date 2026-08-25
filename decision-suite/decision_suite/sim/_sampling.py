"""
dsim._sampling — private: the Monte Carlo sampling engine.

Handles independent sampling (each distributed parameter drawn on its own)
and correlated sampling (a Gaussian-copula transform: draw correlated
standard normals, then push each one through its own parameter's inverse-CDF
so the marginal distributions are unchanged but the joint draws respect the
given correlation structure). This is what reproduces the Excel/@Risk
originals' RiskCorrmat behaviour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .core import SimulationResult, ValidationError

if TYPE_CHECKING:
    from .core import Model


def _distributed_parameters(model: "Model") -> list[str]:
    return [name for name, p in model.parameters.items() if p.distribution is not None]


def _independent_draws(model: "Model", names: list[str], n_iterations: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    draws = {}
    for name in names:
        dist = model.parameters[name].distribution
        draws[name] = dist.rvs(size=n_iterations, random_state=rng)
    return draws


def _correlated_draws(
    model: "Model", names: list[str], n_iterations: int, rng: np.random.Generator,
    correlations: pd.DataFrame | np.ndarray,
) -> dict[str, np.ndarray]:
    if isinstance(correlations, pd.DataFrame):
        missing = set(names) - set(correlations.columns)
        if missing:
            raise ValidationError(
                f"Correlation matrix is missing columns for: {sorted(missing)}.",
                rule="correlation_missing_parameters",
            )
        matrix = correlations.loc[names, names].to_numpy()
    else:
        matrix = np.asarray(correlations)
        if matrix.shape != (len(names), len(names)):
            raise ValidationError(
                f"Correlation matrix shape {matrix.shape} does not match the "
                f"{len(names)} distributed parameters {names}. If passing a "
                f"plain array, its row/column order must match this list, or "
                f"pass a labeled pandas DataFrame instead.",
                rule="correlation_shape_mismatch",
            )

    if not np.allclose(matrix, matrix.T, atol=1e-8):
        raise ValidationError("Correlation matrix must be symmetric.", rule="matrix_not_symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-8):
        raise ValidationError("Correlation matrix diagonal must be all 1.0.", rule="matrix_bad_diagonal")

    try:
        chol = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValidationError(
            "Correlation matrix is not positive semi-definite — check that the "
            "correlations given are mutually consistent.",
            rule="matrix_not_psd",
        ) from exc

    # Sample independent standard normals, correlate them via the Cholesky
    # factor, then convert each column to a uniform via the standard-normal
    # CDF, and finally through that parameter's own inverse-CDF (ppf). This
    # is the standard Gaussian-copula construction: marginals are exactly
    # each parameter's own distribution, but the joint draws are correlated.
    from scipy.stats import norm

    z = rng.standard_normal(size=(n_iterations, len(names)))
    correlated_z = z @ chol.T
    uniforms = norm.cdf(correlated_z)

    draws = {}
    for i, name in enumerate(names):
        dist = model.parameters[name].distribution
        if not hasattr(dist, "ppf"):
            raise ValidationError(
                f"Distribution for '{name}' ({dist.describe()}) does not support "
                f"correlated sampling (no .ppf()) — DiscreteUniform cannot be correlated.",
                parameter=name, rule="distribution_not_correlatable",
            )
        draws[name] = dist.ppf(uniforms[:, i])
    return draws


def run_simulation(
    model: "Model",
    n_iterations: int,
    seed: int | None,
    correlations: pd.DataFrame | np.ndarray | None,
) -> SimulationResult:
    names = _distributed_parameters(model)
    rng = np.random.default_rng(seed)
    if correlations is None:
        draws = _independent_draws(model, names, n_iterations, rng)
    else:
        # Only the parameters actually named in the correlation matrix are
        # sampled jointly; every other distributed parameter is still drawn
        # independently, matching the Excel original (only load_factor and
        # ticket_price used RiskCorrmat — charter_price, hours_flown, and
        # operating_cost kept their own independent RiskTriang/RiskNormal).
        correlated_names = list(correlations.columns) if isinstance(correlations, pd.DataFrame) else names
        independent_names = [n for n in names if n not in correlated_names]

        draws = _correlated_draws(model, correlated_names, n_iterations, rng, correlations)
        draws.update(_independent_draws(model, independent_names, n_iterations, rng))

    base_params = model.base_values()
    outcomes: dict[str, np.ndarray] | None = None
    for i in range(n_iterations):
        params = dict(base_params)
        for name in names:
            params[name] = draws[name][i]
        run_outcomes = model.func(**params)
        if outcomes is None:
            outcomes = {key: np.empty(n_iterations) for key in run_outcomes}
        for key, value in run_outcomes.items():
            outcomes[key][i] = value

    return SimulationResult(
        objectives=outcomes[model.objective],
        objective_name=model.objective,
        outcomes=outcomes,
        parameter_samples=draws,
        n_iterations=n_iterations,
        seed=seed,
        correlated=correlations is not None,
    )
