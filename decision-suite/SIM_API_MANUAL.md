# decision_suite.sim — API manual for the GenAI copilot

This document is not for students. It's a reference for **Claude Code (or
any GenAI copilot)** working inside a student's notebook, so that when a
student asks for something in plain English, the copilot can write correct
`decision_suite.sim` code immediately, without guessing at the API or inventing methods
that don't exist.

## The one rule that matters more than any method list

**Students should never write or edit code themselves.** Every change to
the notebook happens because a student described what they want and the
copilot wrote it. That means this manual has to cover not just "what
methods exist" but "how to turn an arbitrary request into a cell," because
most requests won't match a method name one-for-one.

`decision_suite.sim` is deliberately a small set of **primitives**, not a library with a
pre-built feature for every possible request. The test for whether
something belongs in `decision_suite.sim` is not "is this useful" but **"would an LLM
asked cold reliably get this right if it just wrote it inline, and is
getting it wrong actually costly."** Two things make wrongness costly
enough to protect in a tested package method rather than re-derive per
session: a wrong answer that's silently plausible-looking (not visually
obvious the way a chart rendering strangely is), and/or a correct approach
that's genuinely non-obvious (a naive version exists, looks reasonable,
and is wrong). Everything else — sensitivity sweeps, scenario/parameter
tables, risk metrics, non-interactive charts — is a few lines composed
directly in the notebook against `model.run()` and a result's raw data,
not a package method.

When a request matches an existing method, use it — don't re-derive what
`decision_suite.sim` already computes. When it doesn't, **write a few lines of code in
the notebook cell** using the raw data `decision_suite.sim` exposes (`pandas`,
`matplotlib`, plain Python) — see "Worked translations" below for the
expected shape of that code, including house style (no list
comprehensions, explicit `for` loops — see "Code style," and split a
computation cell from its plot cell rather than combining them). Do
**not** add a new method to the `decision_suite.sim` package to satisfy a one-off
request — that's scope creep in the package, and it hides the student's
actual analytical choice inside a canned function instead of leaving it
visible as code in their own notebook. If a request seems to need a
genuinely new capability the primitives can't reach (rare), stop and ask
before extending the package.

## Code style for composed cells

Per `CLAUDE.md.student-template.md`: no list/dict comprehensions, even
when shorter — use an explicit `for` loop with a named accumulator
variable. Comprehensions read as "clever" to a non-programmer audience.

```python
# Not this:
objectives = [model.run(ticket_price=v).objective for v in values]

# This:
objectives = []
for v in values:
    result = model.run(ticket_price=v)
    objectives.append(result.objective)
```

Also split one step into multiple cells rather than cramming computation
and plotting together — a compute cell (the loop, building the table),
then a separate plot cell, matching every other step in these notebooks.

## The core object: `Model`

```python
model = Model(func, parameters={...}, objective="npv")
```

- `func` — the student's own model function. Takes parameter names as
  keyword arguments, **returns a dict** of every named outcome (not just
  one number) — e.g. `{"revenue": ..., "profit": ..., "npv": ...}`. `decision_suite.sim`
  never sees or writes this function; it's the student's model.
- `parameters` — `dict[str, Parameter | float]`. A plain float is
  shorthand for "no range, no distribution, just a base value."
- `objective` — the dict key that's the headline number for every
  analysis (best/worst, simulation). **"Change my objective to X" = just
  change this string**, or re-run `Model(...)` with a different
  `objective=`. No other code changes needed if `X` is already one of the
  keys `func` returns.

`model.parameters` is a dict (`ParameterDict`) — mutate it directly, e.g.
`model.parameters["ticket_price"].low = 200`. It displays as a table on
its own in Jupyter.

`model.scenarios` is a list (`ScenarioList`) of every scenario added via
`add_scenario()`. Also displays as a table on its own.

**Scenarios vs. decisions — don't conflate these.** A scenario is a state
of nature the student doesn't control (pessimistic/base/optimistic demand).
A decision alternative is a choice someone in the case is actually
proposing (produce 150,000 units vs. 200,000 vs. 225,000). They're both
"a named dict of parameter overrides fed to `model.run()`," but they answer
different questions, and a request to compare decisions **across**
scenarios needs both axes kept separate, not merged into one
`model.scenarios` list. Keep decisions as a plain Python dict in the
notebook (`decisions = {"sales_reps": {...}, "gassman": {...}}`), never
`add_scenario()` calls — see "Worked translations" for the comparison
pattern. This isn't a package feature (no `Model.add_decision()`,
deliberately); it's a notebook-level naming convention.

### `Parameter`

```python
Parameter(name="ticket_price", base=240, low=200, high=300,
          distribution=triangular(200, 240, 300), unit="$/hr",
          description="scheduled-flight ticket price")
```

- `base` — required, the deterministic value.
- `low`/`high` — optional range, used by `best_worst()` when no explicit
  range is passed in, and read directly by hand-composed sensitivity
  sweeps.
- `n_points` — a suggested sweep density (default 6) a composed cell can
  read (e.g. `np.linspace(p.low, p.high, p.n_points)`); not consumed by
  anything in the package itself.
- `distribution` — optional, used only by `simulate()`.
- A parameter can have any subset of these — base only, base + range, or
  base + range + distribution. That's the whole model of "how much
  risk-thinking has been attached so far," not three different classes.

### Distributions

```python
from decision_suite.sim import triangular, normal, uniform, discrete_uniform, empirical
```
Lowercase constructors for `Triangular(low, mode, high)`, `Normal(mean, std)`,
`Uniform(low, high)`, `DiscreteUniform(values)`, `Empirical(points)`. This is
the full set — `decision_suite.sim` deliberately doesn't support every `scipy.stats`
distribution, only shapes an actual case has needed. Don't reach for
`scipy.stats` directly for a distribution shape; if a student wants
something outside this list, that's a design conversation, not a code
change.

`Empirical(points)` takes `(cumulative_probability, value)` pairs and
builds a piecewise-linear CDF through them — use it when a case states its
uncertainty as several elicited quantile points rather than a low/mode/high
shape (this is exactly the Dynatron demand estimate: "median 150,000...
1 chance in 4 it's below 125,000... 1 chance in 4 it's at least 190,000...
certainly between 50,000 and 300,000" → five points, not a triangle):
```python
from decision_suite.sim import empirical

demand_distribution = empirical([
    (0.0, 50_000),    # certain floor
    (0.25, 125_000),  # 1-in-4 chance below this
    (0.5, 150_000),   # median
    (0.75, 190_000),  # 1-in-4 chance at or above this
    (1.0, 300_000),   # certain ceiling
])
```
Points must include probability `0.0` and `1.0` (the stated min and max),
and both probabilities and values must be strictly increasing — don't
silently fit a `Triangular` to a case that actually gave more than three
points, that discards real information the case stated.

## `Model` methods

| Method | Returns | What it does |
|---|---|---|
| `model.run(**overrides)` | `RunResult` | One deterministic run, base values plus any overrides — the one safe primitive everything else, including composed notebook code, is built on. |
| `model.base_values()` | `dict` | Every parameter's base value as a plain dict. |
| `model.add_scenario(name, weight=1.0, **overrides)` | `Model` (chainable) | Add/replace a named scenario. Re-adding a name replaces it, doesn't duplicate. |
| `model.run_scenarios()` | `ScenarioComparison` | Run every added scenario, one row each. |
| `model.scenario_expected_value()` | `float` | Probability-weighted objective across scenarios (uses each scenario's `weight`). |
| `model.best_worst(ranges=None)` | `(RunResult, RunResult)` | True best/worst corner of the parameter box — checks every combination, not a per-parameter guess (see note below). |
| `model.simulate(n_iterations=5000, seed=None, correlations=None)` | `SimulationResult` | Monte Carlo over every parameter with a `.distribution` set; supports correlated (Gaussian-copula) sampling. |
| `model.plot_interactive(parameters=None)` | ipywidgets | Slider per parameter, live-updating the outcome waterfall. |

**Note on `best_worst()`:** it checks all `2**n` corners of the ranged
parameters' box, not each parameter's favorable direction in isolation —
the latter is wrong whenever one parameter's effect on the objective
depends on another's value (proven in the Eagle Airlines model:
`hours_flown`'s good/bad direction flips depending on the other five
parameters). Don't "optimize" this into a per-parameter heuristic even if
it looks slow — it isn't, for any parameter count this course uses (≤20).
This, and `simulate()`'s correlated-sampling path (Cholesky decomposition,
Gaussian-copula transform), are the two places in `decision_suite.sim` doing something
genuinely non-obvious — everything else in this manual's "worked
translations" is deliberately *not* a package method.

There is no `.one_way()`, `.tornado()`, `.two_way()`, or `.scenario(name)`
method, and no `RiskSummary`/`OneWaySensitivityResult`/`TornadoResult`/
`TwoWaySensitivityResult` class. Each is a short loop over `model.run()` a
notebook cell composes directly — see "Worked translations."

## What every result object exposes (for composing custom requests)

This is the part that matters most for open-ended requests. Every result
object below is a normal Python object — index into it, pass it to
`pandas`/`matplotlib` directly, don't wait for a `decision_suite.sim` method to exist.

| Object | Key raw-data attributes |
|---|---|
| `RunResult` | `.params` (dict), `.outcomes` (dict, every key `func` returned), `.objective` (float), `.objective_name` (str), `.label` (str) |
| `ScenarioComparison` | `.results` (list of `RunResult`), `.to_frame()` (DataFrame, one row per scenario) |
| `SimulationResult` | `.objectives` (array, one per iteration), `.objective_name`, `.outcomes` (dict of arrays, every outcome key, not just the objective), `.outcomes_frame()` (DataFrame), `.parameter_samples` (dict of arrays, the actual draws used), `.n_iterations`, `.correlated` |

Every one of these also has a rich Jupyter display (`_repr_html_`) — if a
request is just "show me X," check whether displaying the object itself
already answers it before writing a new table/print statement.

## Existing `.plot()` methods (use before writing new plotting code)

Only the two interactive, ipywidgets-based charts are package methods —
real, hard-won debugging effort went into making these render *and stay
interactive* reliably in VS Code's Jupyter extension, exactly the kind of
subtle, expensive-to-rediscover problem worth protecting. Every other
chart (histogram, bar, line, tornado, waterfall, CDF, distribution
preview) is composed fresh in the notebook — see "Worked translations."

| Call | Shows |
|---|---|
| `sim.plot_interactive()` | Draggable-range histogram (reads off P(low ≤ objective ≤ high)). |
| `model.plot_interactive()` | Slider per parameter, live outcome waterfall. |

**Always call `plt.close(fig)` before the cell's final `fig` reference**
in any composed chart cell — otherwise matplotlib's automatic inline
display *and* Jupyter's last-expression display both fire, rendering the
same chart twice. This is a real, recurring bug this exact pattern has
hit more than once — don't skip it.

## Worked translations: student request → notebook cell

These are examples of the actual judgment call this manual exists for:
does the request match something above, or does it need a few new lines
using raw data? All follow the code-style and multi-cell conventions
above.

**"Change my main objective to profit instead of NPV."**
`func` already returns a `"profit"` key (every intermediate outcome is in
the dict) → just change the `objective=` argument when constructing (or
re-constructing) `Model`. No new method, no new code beyond that one
argument.

**"Show me the parameters/scenarios I've created."**
Already covered — display `model.parameters` or `model.scenarios` in a
cell, don't write a new table from scratch.

**"Sweep ticket_price and show me how NPV changes" (one-way sensitivity)**
Compute cell:
```python
values = [200, 220, 240, 260, 280, 300]
objectives = []
for v in values:
    result = model.run(ticket_price=v)
    objectives.append(result.objective)

import pandas as pd
sweep = pd.DataFrame({"ticket_price": values, "objective": objectives})
sweep
```
Plot cell:
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(values, objectives, color="#376092", marker="o")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("ticket_price")
ax.set_ylabel("objective")
fig.tight_layout()
plt.close(fig)
fig
```

**"Which parameters matter most?" (tornado)**
Compute cell — evaluate each ranged parameter at its low/high endpoint,
holding others at base, and rank by swing:
```python
rows = []
for name, p in model.parameters.items():
    if not p.has_range():
        continue
    low, high = p.range()
    low_objective = model.run(**{name: low}).objective
    high_objective = model.run(**{name: high}).objective
    swing = abs(high_objective - low_objective)
    rows.append({"parameter": name, "low_value": low, "high_value": high,
                 "low_objective": low_objective, "high_objective": high_objective,
                 "swing": swing})

tornado = pd.DataFrame(rows).sort_values("swing", ascending=False)
tornado
```
Plot cell — horizontal bars, widest at top:
```python
fig, ax = plt.subplots(figsize=(8, 0.5 * len(tornado) + 1.5))
sorted_rows = tornado.sort_values("swing")
lefts = sorted_rows[["low_objective", "high_objective"]].min(axis=1)
ax.barh(sorted_rows["parameter"], sorted_rows["swing"], left=lefts, color="#a9c2dc", edgecolor="#376092")
ax.axvline(model.run().objective, color="#b5433c", linestyle="--", label="base case")
ax.legend(frameon=False)
fig.tight_layout()
plt.close(fig)
fig
```

**"Grid of NPV over ticket_price and load_factor" (two-way)**
Compute cell:
```python
import numpy as np

x_values = [200, 220, 240, 260, 280, 300]
y_values = [0.5, 0.55, 0.6, 0.65, 0.7]
grid = np.empty((len(y_values), len(x_values)))
for i, y in enumerate(y_values):
    for j, x in enumerate(x_values):
        result = model.run(ticket_price=x, load_factor=y)
        grid[i, j] = result.objective

two_way = pd.DataFrame(grid, index=y_values, columns=x_values)
two_way
```
Plot cell:
```python
fig, ax = plt.subplots(figsize=(7, 5.5))
mesh = ax.pcolormesh(x_values, y_values, grid, shading="auto", cmap="RdYlGn")
fig.colorbar(mesh, ax=ax, label="objective")
if grid.min() < 0 < grid.max():
    ax.contour(x_values, y_values, grid, levels=[0], colors="black", linewidths=1.5)
ax.set_xlabel("ticket_price")
ax.set_ylabel("load_factor")
fig.tight_layout()
plt.close(fig)
fig
```

**"What's the risk profile of the simulation?" (VaR/CVaR/percentiles)**
```python
objectives = sim.objectives
mean = objectives.mean()
p5 = np.percentile(objectives, 5)
p95 = np.percentile(objectives, 95)
prob_negative = (objectives < 0).mean()
var_5 = np.percentile(objectives, 5)
tail = objectives[objectives <= var_5]
cvar_5 = tail.mean()

print(f"Mean: {mean:,.2f}")
print(f"P(objective < 0): {prob_negative:.1%}")
print(f"VaR (5%): {var_5:,.2f}")
print(f"CVaR (5%): {cvar_5:,.2f}")
```

**"Plot ticket price against profit from the simulation."**
```python
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(sim.parameter_samples["ticket_price"], sim.outcomes["profit"])
ax.set_xlabel("ticket_price")
ax.set_ylabel("profit")
fig.tight_layout()
plt.close(fig)
fig
```

**"What's the probability profit (not NPV) is negative?"**
`sim.outcomes["profit"]` is a plain array → `(sim.outcomes["profit"] < 0).mean()`.
Don't try to force this through a risk-summary helper — it's one line.

**"Compare the pessimistic and optimistic scenarios side by side."**
`model.run_scenarios().to_frame()` already has every scenario as a row —
filter it: `df[df["scenario"].isin(["pessimistic", "optimistic"])]`.

**"Compare a few managers' proposed decisions across the pessimistic/base/
optimistic scenarios" (decision × scenario matrix)**
Two separate plain dicts, one per axis, never `add_scenario()` for the
decisions — see the "Scenarios vs. decisions" note above. Nested loop over
`model.run()`, not `run_scenarios()` (that method only sweeps one axis):
```python
decisions = {
    "sales_reps": {"std_qty": 130_000, "super_qty": 95_000},
    "production_mgr": {"std_qty": 80_000, "super_qty": 70_000},
    "gassman": {"std_qty": 115_000, "super_qty": 85_000},
}
scenarios = {
    "pessimistic": {"demand": 90_000, "super_share": 0.30},
    "base": {"demand": 150_000, "super_share": 0.40},
    "optimistic": {"demand": 220_000, "super_share": 0.60},
}

rows = []
for decision_name, decision_overrides in decisions.items():
    for scenario_name, scenario_overrides in scenarios.items():
        result = model.run(**decision_overrides, **scenario_overrides)
        rows.append({"decision": decision_name, "scenario": scenario_name,
                     "objective": result.objective})

matrix = pd.DataFrame(rows).pivot(index="decision", columns="scenario", values="objective")
matrix
```
Same shape as the two-way sweep above, just crossing two named-override
sets instead of two numeric ranges — still a plain loop over `model.run()`,
not a new package method.

**"Compare those same decisions on their simulated profit distributions"
(overlay + side-by-side stats)**
One `model.simulate()` per decision, holding that decision's parameters
fixed and letting the uncertain parameters (the ones with a
`.distribution`) vary — collect each `SimulationResult`, then a stats table
and an overlaid histogram:
```python
sims = {}
for decision_name, decision_overrides in decisions.items():
    sims[decision_name] = model.simulate(n_iterations=5000, seed=42, **decision_overrides)

stats_rows = []
for decision_name, sim_result in sims.items():
    objectives = sim_result.objectives
    stats_rows.append({
        "decision": decision_name,
        "mean": objectives.mean(),
        "std": objectives.std(),
        "p_negative": (objectives < 0).mean(),
        "var_5": np.percentile(objectives, 5),
    })
stats_table = pd.DataFrame(stats_rows)
stats_table
```
Plot cell — overlay, not subplots, so the spreads are directly comparable:
```python
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = ["#376092", "#b5433c", "#4c9a5c"]
for (decision_name, sim_result), color in zip(sims.items(), colors):
    ax.hist(sim_result.objectives, bins=40, alpha=0.45, label=decision_name, color=color)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("objective")
ax.legend(frameon=False)
fig.tight_layout()
plt.close(fig)
fig
```
If a student wants this overlay interactive (toggle a decision on/off,
hover for stats), that's a real candidate for a new `decision_suite.sim` method — the
static version above isn't, it's a plain loop and a `plt.hist` call.

**"Only simulate 10,000 iterations instead of 5,000."**
`model.simulate(n_iterations=10000, seed=...)` — just a different argument
value, not a new capability.

**"Preview this parameter's distribution before simulating."**
```python
dist = model.parameters["ticket_price"].distribution
x, y = dist.pdf_curve()  # every distribution class exposes this

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(x, y, color="#376092")
ax.fill_between(x, y, color="#376092", alpha=0.25)
ax.axvline(dist.mean(), color="#b5433c", linestyle="--", label=f"mean = {dist.mean():,.3g}")
ax.set_title(dist.describe())
ax.legend(frameon=False)
fig.tight_layout()
plt.close(fig)
fig
```

## Where the real ceiling is

If a request needs something no combination of the above can produce
(e.g. a genuinely new sampling scheme, a new distribution family) —
that's a real package change, not a notebook cell. Stop and flag it
rather than bolting on a one-off method quietly; those decisions belong
to a deliberate design conversation, the same way `Model.display()`, the
two-way heatmap `.plot()`, and the one-way/tornado/two-way/risk-summary
methods were all cut from this package for being exactly that kind of
addition an LLM could reliably compose fresh instead.
