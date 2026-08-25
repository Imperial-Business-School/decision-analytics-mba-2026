"""
dsim._sensitivity — private: best/worst sensitivity analysis over a `Model`.

One-way, tornado, and two-way sensitivity sweeps are not implemented here —
they're each a short loop over `model.run(**overrides)` a student composes
directly in the notebook, not something an LLM would reliably get wrong if
asked to write it fresh. `best_worst_extremes` stays: the naive
one-parameter-at-a-time heuristic gives a *wrong* answer whenever one
parameter's effect on the objective depends on another's value, which is
easy to miss and not obviously wrong-looking, exactly the kind of thing
worth protecting in a tested package function instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core import RunResult, ValidationError

if TYPE_CHECKING:
    from .core import Model


def best_worst_extremes(model: "Model", ranges: dict[str, tuple[float, float]]) -> tuple[RunResult, RunResult]:
    """
    The true best-case / worst-case objective over the box defined by
    `ranges`: every combination of each parameter's low/high endpoint is
    evaluated (2**len(ranges) runs), and the actual best- and worst-scoring
    corner is returned.

    Exhaustive rather than one-parameter-at-a-time: picking each parameter's
    favorable direction in isolation (holding the rest at base) and then
    combining them is only correct if no parameter's effect on the
    objective depends on another parameter's value. Business models often
    don't have that property — e.g. whether more flight hours help or hurt
    NPV here depends on whether revenue-per-hour exceeds operating
    cost-per-hour, which itself depends on ticket price, load factor, and
    charter price. Checking every corner is exact wherever the model is
    linear in each parameter (as this one is), and a closer approximation
    than the one-at-a-time heuristic in general.
    """
    import itertools

    names = list(ranges)
    if len(names) > 20:
        raise ValidationError(
            f"best_worst() checks every corner of the parameter box (2**n runs) — "
            f"{len(names)} ranged parameters means {2**len(names):,} runs, too many "
            f"to run here. Narrow `ranges` to fewer parameters.",
            rule="too_many_ranges",
        )

    best_run, worst_run = None, None
    for corner in itertools.product(*(ranges[name] for name in names)):
        overrides = dict(zip(names, corner))
        run = model.run(**overrides)
        if best_run is None or run.objective > best_run.objective:
            best_run = run
        if worst_run is None or run.objective < worst_run.objective:
            worst_run = run

    best_run.label = "best case"
    worst_run.label = "worst case"
    return best_run, worst_run
