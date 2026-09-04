"""
dsim.core — data model, distributions, and the Model class.

One Model class, many analysis methods (compare dtree's single DecisionTree
class with .rollback()/.sensitivity()/.flip()) — what varies between a base
case, a scenario, a sensitivity sweep, and a Monte Carlo simulation is how
richly each Parameter is populated (base only; base+range; base+range+
distribution), not which class you construct.

No sampling-engine or interactive-plotting internals live here — those are
private modules (dsim._sampling, dsim._sensitivity, dsim._plotting) that
Model's methods call into. This module is the public surface: what a
student imports and constructs directly. Sensitivity sweeps (one-way,
tornado, two-way), risk metrics (VaR/CVaR), and non-interactive charts are
deliberately *not* provided here — each is a few lines a student composes
directly in the notebook against `model.run()` and a result's raw data
(`.objective`, `.outcomes`, `.objectives`), not something worth hiding
behind a package method. What stays (`.best_worst()`, `.simulate()`'s
correlated sampling) is the opposite: numerically subtle enough that a
fresh, ungoverned attempt is genuinely likely to get it wrong.

Import from here (or, more conveniently, from decision_suite.sim directly):

    from decision_suite.sim import Model, Parameter, Triangular, Normal, Uniform, DiscreteUniform, Empirical
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


# ===========================================================================
# EXCEPTIONS
# ===========================================================================

class DsimError(Exception):
    """Base class for all dsim errors."""


class ValidationError(DsimError):
    """
    Raised when a distribution, parameter, or correlation matrix is
    constructed with inconsistent values.

    Attributes
    ----------
    parameter : str | None
        Name of the offending parameter, or None for model-level errors.
    rule : str
        Short machine-readable rule identifier, e.g. "low_above_high",
        "missing_parameter", "matrix_not_symmetric".
    """
    def __init__(self, message: str, parameter: str | None = None, rule: str = "") -> None:
        super().__init__(message)
        self.parameter = parameter
        self.rule = rule


# ===========================================================================
# DISTRIBUTIONS
# ===========================================================================
# Deliberately narrow vocabulary — matches what the Excel/@Risk originals
# actually used (RiskTriang, RiskNormal, RiskUniform, RiskDUniform), plus
# Empirical (below), added because a real case (Dynatron) states its
# uncertainty as elicited quantile points, not a low/mode/high shape. No
# Lognormal/Pert/Beta/etc: still no full scipy.stats catalogue, only shapes
# an actual case has needed — see SIM_API_MANUAL.md's "design conversation, not
# a code change" note for why this list stays deliberately short.

class Triangular:
    """
    Triangular distribution — a range with a most-likely value (mode).

    Use when a parameter has a plausible range and a single most-likely
    value inside it, and values further from the mode are progressively
    less likely (e.g. "$240 most likely, could be as low as $200 or as
    high as $300").

    Parameters
    ----------
    low : float
        Lowest plausible value (pessimistic extreme).
    mode : float
        Most likely value (matches the deterministic base case).
    high : float
        Highest plausible value (optimistic extreme).
    """
    def __init__(self, low: float, mode: float, high: float) -> None:
        if not (low <= mode <= high):
            raise ValidationError(
                f"Triangular requires low <= mode <= high, got low={low}, mode={mode}, high={high}.",
                rule="triangular_out_of_order",
            )
        self.low = float(low)
        self.mode = float(mode)
        self.high = float(high)

    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3

    def rvs(self, size: int = 1, random_state: np.random.Generator | None = None) -> np.ndarray:
        rng = random_state or np.random.default_rng()
        # scipy's triang uses a 0-1 shape parameter c = (mode-low)/(high-low)
        c = 0.0 if self.high == self.low else (self.mode - self.low) / (self.high - self.low)
        from scipy import stats
        return stats.triang.rvs(c, loc=self.low, scale=self.high - self.low, size=size, random_state=rng)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        from scipy import stats
        c = 0.0 if self.high == self.low else (self.mode - self.low) / (self.high - self.low)
        return stats.triang.cdf(x, c, loc=self.low, scale=self.high - self.low)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """Inverse CDF — used by correlated sampling to transform standard-normal draws."""
        from scipy import stats
        c = 0.0 if self.high == self.low else (self.mode - self.low) / (self.high - self.low)
        return stats.triang.ppf(q, c, loc=self.low, scale=self.high - self.low)

    def pdf_curve(self, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(self.low, self.high, n)
        from scipy import stats
        c = 0.0 if self.high == self.low else (self.mode - self.low) / (self.high - self.low)
        y = stats.triang.pdf(x, c, loc=self.low, scale=self.high - self.low)
        return x, y

    def range(self) -> tuple[float, float]:
        return self.low, self.high

    def describe(self) -> str:
        return f"Triangular(low={self.low:g}, mode={self.mode:g}, high={self.high:g})"

    def __repr__(self) -> str:
        return self.describe()


class Normal:
    """
    Normal (Gaussian) distribution — a symmetric range around a mean.

    Use when a parameter's uncertainty is "the best guess could be wrong by
    about X in either direction, and values further from the guess become
    steadily less likely, with no hard cutoff" (e.g. "$1,200 could be wrong
    by $50, depending on fuel prices").

    Parameters
    ----------
    mean : float
        Best-guess / expected value (matches the deterministic base case).
    std : float
        Standard deviation. If you only have a "could be wrong by ±X"
        statement, a common rule of thumb is std ≈ X / 2 (so ±X covers
        roughly a 95% range).
    """
    def __init__(self, mean: float, std: float) -> None:
        if std <= 0:
            raise ValidationError(f"Normal requires std > 0, got {std}.", rule="non_positive_std")
        self.mean_ = float(mean)
        self.std = float(std)

    def mean(self) -> float:
        return self.mean_

    def rvs(self, size: int = 1, random_state: np.random.Generator | None = None) -> np.ndarray:
        rng = random_state or np.random.default_rng()
        return rng.normal(self.mean_, self.std, size=size)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        from scipy import stats
        return stats.norm.cdf(x, loc=self.mean_, scale=self.std)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        from scipy import stats
        return stats.norm.ppf(q, loc=self.mean_, scale=self.std)

    def pdf_curve(self, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(self.mean_ - 4 * self.std, self.mean_ + 4 * self.std, n)
        from scipy import stats
        y = stats.norm.pdf(x, loc=self.mean_, scale=self.std)
        return x, y

    def range(self) -> tuple[float, float]:
        """5th/95th percentile range — used as the default sensitivity sweep range."""
        from scipy import stats
        return (
            stats.norm.ppf(0.05, loc=self.mean_, scale=self.std),
            stats.norm.ppf(0.95, loc=self.mean_, scale=self.std),
        )

    def describe(self) -> str:
        return f"Normal(mean={self.mean_:g}, std={self.std:g})"

    def __repr__(self) -> str:
        return self.describe()


class Uniform:
    """
    Uniform distribution — every value in a range is equally likely.

    Use when a parameter has a plausible range but no reason to think any
    value in it is more likely than another (e.g. "somewhere between $1,600
    and $2,200 per hour, no idea which is more likely").

    Parameters
    ----------
    low : float
        Lowest plausible value.
    high : float
        Highest plausible value.
    """
    def __init__(self, low: float, high: float) -> None:
        if not (low < high):
            raise ValidationError(f"Uniform requires low < high, got low={low}, high={high}.", rule="uniform_out_of_order")
        self.low = float(low)
        self.high = float(high)

    def mean(self) -> float:
        return (self.low + self.high) / 2

    def rvs(self, size: int = 1, random_state: np.random.Generator | None = None) -> np.ndarray:
        rng = random_state or np.random.default_rng()
        return rng.uniform(self.low, self.high, size=size)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(x) - self.low) / (self.high - self.low), 0, 1)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        return self.low + np.asarray(q) * (self.high - self.low)

    def pdf_curve(self, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(self.low, self.high, n)
        y = np.full_like(x, 1.0 / (self.high - self.low))
        return x, y

    def range(self) -> tuple[float, float]:
        return self.low, self.high

    def describe(self) -> str:
        return f"Uniform(low={self.low:g}, high={self.high:g})"

    def __repr__(self) -> str:
        return self.describe()


class DiscreteUniform:
    """
    Discrete uniform distribution — a fixed set of equally-likely values.

    Use for genuinely discrete, equally-likely outcomes (a die face, a coin
    flip, a shortlist of equally plausible scenarios). This is the
    distribution used for the warm-up dice exercise before the real model.

    Parameters
    ----------
    values : list[float]
        The equally-likely values, e.g. [1, 2, 3, 4, 5, 6] for a die.
    """
    def __init__(self, values: list[float]) -> None:
        if len(values) == 0:
            raise ValidationError("DiscreteUniform requires at least one value.", rule="empty_values")
        self.values = list(values)

    def mean(self) -> float:
        return float(np.mean(self.values))

    def rvs(self, size: int = 1, random_state: np.random.Generator | None = None) -> np.ndarray:
        rng = random_state or np.random.default_rng()
        return rng.choice(self.values, size=size)

    def range(self) -> tuple[float, float]:
        return min(self.values), max(self.values)

    def describe(self) -> str:
        return f"DiscreteUniform({self.values})"

    def __repr__(self) -> str:
        return self.describe()


class Empirical:
    """
    Empirical (quantile-elicited) distribution — a CDF built by connecting
    a handful of stated percentile points with straight lines.

    Use when a range and a single most-likely value (Triangular) isn't rich
    enough to capture what was actually estimated — e.g. "the median is
    150,000; we're sure it's between 50,000 and 300,000; there's a 1-in-4
    chance it's below 125,000 and a 1-in-4 chance it's at least 190,000."
    That's five stated points, not a three-point shape, and forcing it into
    Triangular(50_000, 150_000, 300_000) would silently discard the two
    quartile points.

    Parameters
    ----------
    points : list[tuple[float, float]]
        (cumulative_probability, value) pairs, e.g.
        [(0.0, 50_000), (0.25, 125_000), (0.5, 150_000), (0.75, 190_000), (1.0, 300_000)].
        Must include probability 0.0 and 1.0 (the stated min and max);
        probabilities and values must both be strictly increasing.
    """
    def __init__(self, points: list[tuple[float, float]]) -> None:
        if len(points) < 2:
            raise ValidationError("Empirical requires at least 2 points.", rule="too_few_points")
        sorted_points = sorted(points, key=lambda point: point[0])
        probabilities = [probability for probability, _ in sorted_points]
        values = [value for _, value in sorted_points]
        if probabilities[0] != 0.0 or probabilities[-1] != 1.0:
            raise ValidationError(
                "Empirical requires points at probability 0.0 and 1.0 (the "
                f"stated min and max), got {probabilities[0]:g} and {probabilities[-1]:g}.",
                rule="missing_endpoints",
            )
        for lower, upper in zip(probabilities, probabilities[1:]):
            if not (upper > lower):
                raise ValidationError(
                    "Empirical requires strictly increasing probabilities.",
                    rule="probabilities_not_increasing",
                )
        for lower, upper in zip(values, values[1:]):
            if not (upper > lower):
                raise ValidationError(
                    "Empirical requires strictly increasing values as probability increases.",
                    rule="values_not_increasing",
                )
        self.probabilities = np.array(probabilities, dtype=float)
        self.values = np.array(values, dtype=float)

    def mean(self) -> float:
        # E[X] via the midpoint of each linear CDF segment, weighted by that
        # segment's probability mass — exact for a piecewise-linear CDF.
        segment_mass = np.diff(self.probabilities)
        segment_midpoint = (self.values[:-1] + self.values[1:]) / 2
        return float(np.sum(segment_mass * segment_midpoint))

    def rvs(self, size: int = 1, random_state: np.random.Generator | None = None) -> np.ndarray:
        rng = random_state or np.random.default_rng()
        uniform_draws = rng.uniform(0.0, 1.0, size=size)
        return self.ppf(uniform_draws)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.interp(x, self.values, self.probabilities), 0.0, 1.0)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """Inverse CDF — piecewise-linear interpolation through the stated points."""
        return np.interp(q, self.probabilities, self.values)

    def pdf_curve(self, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        # The density implied by a piecewise-linear CDF is a step function:
        # constant within each segment, at that segment's probability mass
        # divided by its width.
        x = np.linspace(self.values[0], self.values[-1], n)
        segment_density = np.diff(self.probabilities) / np.diff(self.values)
        segment_index = np.clip(np.searchsorted(self.values, x, side="right") - 1, 0, len(segment_density) - 1)
        y = segment_density[segment_index]
        return x, y

    def range(self) -> tuple[float, float]:
        return float(self.values[0]), float(self.values[-1])

    def describe(self) -> str:
        stated_points = ", ".join(f"{p:g}->{v:g}" for p, v in zip(self.probabilities, self.values))
        return f"Empirical({stated_points})"

    def __repr__(self) -> str:
        return self.describe()


Distribution = Triangular | Normal | Uniform | DiscreteUniform | Empirical


# ===========================================================================
# PARAMETERS & SCENARIOS
# ===========================================================================

@dataclass
class Parameter:
    """
    One input to the model — carries everything that's known about it, so
    `Model` doesn't need separate classes for separate kinds of analysis
    (compare `dtree`'s `Value`, which similarly carries a base, optional
    scenarios, and optional uncertainty on one object).

    A Parameter always has a `base` value (used by `Model.run()`) and may
    optionally carry a `distribution` (used only by `Model.simulate()`)
    and/or a `low`/`high` range (used by `Model.best_worst()` when no
    explicit range is passed in, and by hand-written sensitivity sweeps
    composed in the notebook).

    Parameters
    ----------
    name : str
        Must match the keyword argument name in the student's model function.
    base : float
        Base-case value.
    low, high : float | None
        Plausible range, for sensitivity analysis. If omitted and a
        distribution is set, the distribution's own `.range()` is used.
    n_points : int
        Suggested number of points for a hand-composed sensitivity sweep
        of this parameter (e.g. `np.linspace(p.low, p.high, p.n_points)`)
        — a documented default a notebook cell can read, not consumed by
        anything in the package itself.
    distribution : Distribution | None
        Set only for parameters that will be sampled by `Model.simulate()`.
    unit : str
        Free-text unit label, shown in tables/plots (e.g. "$/hour", "%").
    description : str
        One-line business-meaning note, shown in `model.parameters`'s table.
    """
    name: str
    base: float
    low: float | None = None
    high: float | None = None
    n_points: int = 6
    distribution: Distribution | None = None
    unit: str = ""
    description: str = ""

    def has_range(self) -> bool:
        """True if this parameter has enough information to derive a low/high range."""
        return (self.low is not None and self.high is not None) or self.distribution is not None

    def range(self) -> tuple[float, float]:
        if self.low is not None and self.high is not None:
            return self.low, self.high
        if self.distribution is not None:
            return self.distribution.range()
        raise ValidationError(
            f"Parameter '{self.name}' has no low/high range and no distribution to derive one from.",
            parameter=self.name, rule="missing_range",
        )


@dataclass
class Scenario:
    """
    A named, user-authored parameter combination, e.g.
    Scenario('pessimistic', overrides={'load_factor': 0.5}, weight=1.0).

    `weight` is only used by `Model.scenario_expected_value()` — weights are
    normalised across every scenario added to the same Model, so equal
    weights (the default) mean "equally plausible," not "certain."
    """
    name: str
    overrides: dict[str, float]
    weight: float = 1.0


def _html_section(title: str, meta_lines: list[str], table_label: str, table_html: str) -> str:
    """Shared layout for every result class's `_repr_html_`: a bold title,
    a few labeled one-line facts, then a labeled table — so every dsim
    result reads the same way instead of each being a bare table."""
    parts = [f"<div style='margin-bottom:0.3em'><b>{title}</b></div>"]
    for line in meta_lines:
        parts.append(f"<div style='margin-bottom:0.2em'>{line}</div>")
    parts.append(f"<div style='margin-top:0.4em'><b>{table_label}</b></div>")
    parts.append(table_html)
    return f"<div>{''.join(parts)}</div>"


def _outcomes_table_html(outcomes: dict[str, float]) -> str:
    rows = "".join(
        f"<tr><td style='text-align:left'>{name}</td>"
        f"<td style='text-align:right'>{value:,.2f}</td></tr>"
        for name, value in outcomes.items()
    )
    return f"<table><tbody>{rows}</tbody></table>"


class Outcomes(dict):
    """
    A plain dict of a model function's returned outcomes, with a nicer
    Jupyter table display — still a normal dict everywhere else (`.get()`,
    indexing, `dict(outcomes)`, `pd.Series(outcomes)` all still work).
    """

    def _repr_html_(self) -> str:
        return _html_section("Outcomes", [], "Values:", _outcomes_table_html(self))


# ===========================================================================
# RESULTS
# ===========================================================================

@dataclass
class RunResult:
    """
    The result of one deterministic model run.

    Attributes
    ----------
    params : dict[str, float]
        The full parameter set used for this run.
    outcomes : dict[str, float]
        Everything the student's model function returned (all intermediate
        outcomes, not just the objective) — this is why the model function
        should return a dict, not a single number.
    objective : float
        The single headline number (e.g. NPV) — `outcomes[objective_name]`.
    objective_name : str
        The key in `outcomes` that `objective` came from (e.g. "npv") —
        lets a RunResult label itself in charts without needing the model
        it came from passed back in.
    label : str
        Optional name (e.g. a scenario name), blank for a plain run.
    """
    params: dict[str, float]
    outcomes: dict[str, float]
    objective: float
    objective_name: str
    label: str = ""

    def __repr__(self) -> str:
        tag = f" [{self.label}]" if self.label else ""
        return f"RunResult{tag}(objective={self.objective:,.2f})"

    def _repr_html_(self) -> str:
        title = f"Run result [{self.label}]" if self.label else "Run result"
        meta = [f"Objective: <code>{self.objective_name}</code> = {self.objective:,.2f}"]
        return _html_section(title, meta, "Outcomes:", _outcomes_table_html(self.outcomes))


@dataclass
class ScenarioComparison:
    """Result of `Model.run_scenarios()` — one row per named scenario."""
    results: list[RunResult]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"scenario": r.label, **r.outcomes}
            for r in self.results
        )

    def _repr_html_(self) -> str:
        meta = [f"{len(self.results)} scenario(s): " + ", ".join(r.label for r in self.results)]
        return _html_section("Scenario comparison", meta, "Outcomes:", self.to_frame()._repr_html_())


@dataclass
class SimulationResult:
    """
    Result of `Model.simulate()`.

    Attributes
    ----------
    objectives : np.ndarray
        One simulated objective value per iteration (same as
        `outcomes[model.objective]`) — kept as its own attribute since it's
        the array most risk metrics and charts composed against this
        result actually use.
    objective_name : str
        The key in `outcomes` that `objectives` came from (e.g. "npv").
    outcomes : dict[str, np.ndarray]
        Every outcome the model function returned, not just the objective —
        one array per outcome key, each of length `n_iterations`. Lets you
        inspect the simulated distribution of e.g. `profit` or `tax`, not
        only the headline NPV.
    parameter_samples : dict[str, np.ndarray]
        The sampled draws for each parameter that had a distribution.
    n_iterations : int
    seed : int | None
    correlated : bool
        Whether correlated sampling was used.
    """
    objectives: np.ndarray
    objective_name: str
    outcomes: dict[str, np.ndarray]
    parameter_samples: dict[str, np.ndarray]
    n_iterations: int
    seed: int | None
    correlated: bool = False

    def outcomes_frame(self) -> pd.DataFrame:
        """Every simulated outcome, one column per outcome, one row per iteration."""
        return pd.DataFrame(self.outcomes)

    def _repr_html_(self) -> str:
        meta = [
            f"Objective: <code>{self.objective_name}</code>",
            f"{self.n_iterations:,} iterations"
            + (" (correlated sampling)" if self.correlated else " (independent sampling)"),
        ]
        return _html_section("Simulation result", meta, "Outcome summary:", self.outcomes_frame().describe()._repr_html_())

    def plot_interactive(self, **kwargs):
        """Interactive histogram with a draggable low/high range slider — see dsim._plotting.interactive_risk_range."""
        from ._plotting import interactive_risk_range
        return interactive_risk_range(self, **kwargs)


class ParameterDict(dict):
    """
    `Model.parameters` — a plain `dict[str, Parameter]` (so `.parameters[name]`,
    `.parameters[name].low = ...` etc. all work exactly as before), with a
    nicer Jupyter table display: name, base, low/high, distribution, unit.
    """

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for p in self.values():
            try:
                low, high = p.range()
            except ValidationError:
                low, high = None, None
            rows.append(
                {
                    "parameter": p.name,
                    "base": p.base,
                    "low": low,
                    "high": high,
                    "distribution": repr(p.distribution) if p.distribution is not None else None,
                    "unit": p.unit,
                    "description": p.description,
                }
            )
        return pd.DataFrame(rows)

    def _repr_html_(self) -> str:
        return self.to_frame()._repr_html_()


class ScenarioList(list):
    """
    `Model.scenarios` — a plain `list[Scenario]` (append/iterate as usual),
    with a nicer Jupyter table display: each scenario's name and its full
    parameter set (base values with that scenario's overrides applied).
    """

    def __init__(self, *args, model: "Model | None" = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._model = model

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"scenario": s.name, **{**self._model.base_values(), **s.overrides}}
            for s in self
        )

    def _repr_html_(self) -> str:
        return self.to_frame()._repr_html_()


# ===========================================================================
# MODEL
# ===========================================================================

class Model:
    """
    Wraps a student-written business model function and gives it scenario,
    best/worst, and Monte Carlo simulation capabilities — one object, a
    deliberately small set of methods, the same shape as dtree's
    `DecisionTree` (`.rollback()`, `.sensitivity()`, `.flip()` all live on
    one tree; here `.run()`, `.add_scenario()`, `.best_worst()`,
    `.simulate()` all live on one Model). Sensitivity sweeps, risk metrics,
    and charts are deliberately not methods here — compose them in the
    notebook against `.run()` and a result's raw data instead; see
    `SIM_API_MANUAL.md`.

    The function itself — the actual NPV/profit/cost formula — is always
    the student's own code. Model never sees or generates that logic; it
    only calls the function repeatedly with different parameter values and
    organizes the results.

    What varies between a base-case run, a scenario, a sensitivity sweep,
    and a Monte Carlo simulation isn't which class you use — it's how
    richly each `Parameter` is populated (base only; base + low/high;
    base + low/high + distribution). See `Parameter`.

    Parameters
    ----------
    func : Callable[..., dict[str, float]]
        The student's own model function. Must accept the parameter names
        as keyword arguments and return a dict of named outcomes (not just
        a single number) — e.g. {"revenue": ..., "npv": ...}.
    parameters : dict[str, Parameter | float]
        One entry per argument `func` expects. A plain float is shorthand
        for `Parameter(name, base=value)` (no range, no distribution) — use
        this for parameters that are genuinely fixed constants, not
        uncertain inputs.
    objective : str
        The key in the returned dict that is the headline number to
        analyze (e.g. "npv").
    """

    def __init__(
        self,
        func: Callable[..., dict[str, float]],
        parameters: dict[str, Parameter | float],
        objective: str,
    ) -> None:
        self.func = func
        self.objective = objective
        self.parameters = ParameterDict(
            (name, p if isinstance(p, Parameter) else Parameter(name=name, base=p))
            for name, p in parameters.items()
        )
        self.scenarios = ScenarioList(model=self)

    def _repr_html_(self) -> str:
        parts = [
            f"<div style='margin-bottom:0.3em'><b>Outcome engine:</b> <code>{self.func.__name__}</code></div>",
            f"<div style='margin-bottom:0.3em'><b>Objective:</b> <code>{self.objective}</code></div>",
        ]
        if self.scenarios:
            names = ", ".join(s.name for s in self.scenarios)
            parts.append(f"<div style='margin-bottom:0.3em'><b>Scenarios ({len(self.scenarios)}):</b> {names}</div>")
        parts.append(f"<div><b>Parameters:</b></div>{self.parameters.to_frame()._repr_html_()}")
        return f"<div>{''.join(parts)}</div>"

    # -- running -------------------------------------------------------------

    def base_values(self) -> dict[str, float]:
        return {name: p.base for name, p in self.parameters.items()}

    def run(self, **overrides: float) -> RunResult:
        """Run the model once, at base-case values except for any overrides given."""
        params = {**self.base_values(), **overrides}
        outcomes = Outcomes(self.func(**params))
        if self.objective not in outcomes:
            raise ValidationError(
                f"Model function did not return '{self.objective}' among its outcomes "
                f"(got keys: {list(outcomes)}).",
                rule="missing_objective",
            )
        return RunResult(params=params, outcomes=outcomes, objective=outcomes[self.objective], objective_name=self.objective)

    # -- scenarios -------------------------------------------------------------

    def add_scenario(self, name: str, weight: float = 1.0, **overrides: float) -> "Model":
        """Add a named, user-authored scenario. Chainable — returns self.

        Re-adding an existing name replaces it rather than duplicating it,
        so re-running a notebook cell with tweaked values doesn't pile up
        repeated scenarios.
        """
        self.scenarios[:] = [s for s in self.scenarios if s.name != name]
        self.scenarios.append(Scenario(name=name, overrides=overrides, weight=weight))
        return self

    def run_scenarios(self) -> ScenarioComparison:
        """Run every added scenario and collect the results side by side."""
        if not self.scenarios:
            raise ValidationError("No scenarios added yet — call .add_scenario(name, **overrides) first.", rule="no_scenarios")
        results = []
        for s in self.scenarios:
            r = self.run(**s.overrides)
            r.label = s.name
            results.append(r)
        return ScenarioComparison(results=results)

    def scenario_expected_value(self) -> float:
        """
        Weight-normalised expected objective across every added scenario.
        Equal weights (the default) mean "equally plausible," not
        "certain" — set explicit weights on `.add_scenario()` to reflect
        actual likelihoods if you have them.
        """
        if not self.scenarios:
            raise ValidationError("No scenarios added yet — call .add_scenario(name, **overrides) first.", rule="no_scenarios")
        total_weight = sum(s.weight for s in self.scenarios)
        if total_weight <= 0:
            raise ValidationError("Scenario weights must sum to a positive number.", rule="non_positive_total_weight")
        return sum(s.weight * self.run(**s.overrides).objective for s in self.scenarios) / total_weight

    # -- sensitivity -------------------------------------------------------------

    def _resolve_ranges(self, ranges: dict[str, tuple[float, float]] | None) -> dict[str, tuple[float, float]]:
        ranges = ranges or {name: p.range() for name, p in self.parameters.items() if p.has_range()}
        if not ranges:
            raise ValidationError(
                "No parameters have a range to analyze — set low/high or a distribution "
                "on at least one Parameter, or pass ranges= explicitly.",
                rule="no_ranges",
            )
        return ranges

    def best_worst(self, ranges: dict[str, tuple[float, float]] | None = None) -> tuple[RunResult, RunResult]:
        """
        Construct the mechanical best-case / worst-case parameter
        combination: for each parameter, whichever range endpoint improves
        the objective goes into "best," whichever worsens it goes into
        "worst" (tested one at a time, holding others at base). This is a
        bound, not a realistic scenario — all parameters landing at their
        best value simultaneously is not a forecast. If `ranges` is
        omitted, every parameter with a range is included.
        """
        from ._sensitivity import best_worst_extremes
        return best_worst_extremes(self, ranges=self._resolve_ranges(ranges))

    # -- simulation ------------------------------------------------------

    def simulate(
        self,
        n_iterations: int = 5000,
        seed: int | None = None,
        correlations: pd.DataFrame | np.ndarray | None = None,
    ) -> SimulationResult:
        """
        Monte Carlo simulation: draw `n_iterations` samples from every
        parameter that has a `distribution` set, run the model on each draw,
        and collect the resulting objective values.

        Parameters
        ----------
        n_iterations : int
            Number of samples to draw (5,000 matches the Excel/@Risk original).
        seed : int | None
            Random seed, for reproducible results.
        correlations : DataFrame | ndarray | None
            Optional correlation matrix (indexed/ordered by parameter name)
            over a *subset* of the distributed parameters — if given,
            those parameters are sampled with a Gaussian-copula transform
            instead of independently; every other distributed parameter is
            still sampled independently.
        """
        if not any(p.distribution is not None for p in self.parameters.values()):
            raise ValidationError(
                "No parameters have a distribution set — nothing to simulate. "
                "Set distribution= on at least one Parameter first.",
                rule="no_distributions",
            )
        from ._sampling import run_simulation
        return run_simulation(self, n_iterations=n_iterations, seed=seed, correlations=correlations)

    # -- display -----------------------------------------------------------

    def plot_interactive(self, parameters: list[str] | None = None, **kwargs):
        """One slider per parameter, live-updating the outcome waterfall as you drag — see dsim._plotting.interactive_parameters."""
        from ._plotting import interactive_parameters
        return interactive_parameters(self, parameters=parameters, **kwargs)

    def __repr__(self) -> str:
        return f"Model(objective={self.objective!r}, parameters={list(self.parameters)})"
