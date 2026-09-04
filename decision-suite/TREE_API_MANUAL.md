# decision_suite.tree — API manual for the GenAI copilot

This document is not for students. It's a reference for **Claude Code (or
any GenAI copilot)** working inside a student's notebook, so that when a
student asks for something in plain English about a decision tree, the
copilot can write correct `decision_suite.tree` code immediately, without
guessing at the API or inventing methods that don't exist.

## The one rule that matters more than any method list

**Students should never write or edit code themselves.** Every change to
the notebook happens because a student described what they want and the
copilot wrote it. `decision_suite.tree` (import path `decision_suite.tree`,
package formerly known as `dtree`) is a real, fairly complete modeling
library, not a thin set of primitives the way `decision_suite.sim` is —
building a tree, solving it, reading off the optimal policy, drawing a risk
profile, running a sensitivity sweep, and computing EVPI/EVSI are all
first-class package features. The judgment call this manual exists for is
narrower than in the `sim` manual: mostly "which existing method/class does
this request map to," and occasionally "this needs a few lines of plain
Python composed around the tree's raw objects (a branch, a probability
list, a rollback result)" when a request is genuinely bespoke (e.g. a
custom tornado across named parameters that aren't all `Value`/`Prob`
objects with `Range` uncertainty — see the Texaco worked example below).

Do **not** add a new method to the `decision_suite.tree` package to satisfy
a one-off notebook request. If a request seems to need real new capability
the classes below can't reach, stop and ask before extending the package.

## Import surface

```python
from decision_suite.tree import (
    ChanceNode, DecisionNode, LeafNode, LogicNode, DecisionTree,
    Branch, Prob, Value, Range, Scenario, Context,
    GlobalSettings, TreeSettings, settings,
    RollbackResult, RiskProfile, RiskProfileCollection,
    SensitivityResult, RiskAttitudeSensitivityResult,
    DtreeError, ValidationError, FlipError, RollbackError, SerializationError,
)
```

In worked notebooks, the everyday import is just:
```python
from decision_suite.tree import ChanceNode, DecisionNode, LeafNode, DecisionTree
```
`Range` (for sensitivity analysis) and `RiskProfile`/`RiskProfileCollection`
(for building custom risk-profile overlays) are usually imported later,
from `decision_suite.tree.core`, right when the notebook first needs them
— both import paths work, `decision_suite.tree.core` is the module they're
actually defined in.

`ChanceNode`, `DecisionNode`, `LeafNode`, `LogicNode` imported from
`decision_suite.tree` are the **facilitator** versions — they carry the
mutation methods (`.add()`, `.set_probs()`, `.force()`, etc.) used to build
and edit a tree. The same names imported from `decision_suite.tree.core`
are the bare versions with no mutation methods, useful only for type
annotations or fully declarative construction (passing `branches=[...]`
directly). Always import the node classes from `decision_suite.tree`
itself for normal notebook work.

## The core objects

### `Branch`

The edge leaving any node. Rarely constructed directly in a notebook —
`.add()` builds one for you — but every method that returns "the thing you
loop over" (`node.branches`) returns a list of these.

```python
Branch(child=some_node, value=1200.0, time=0, active=True, label="heavy",
       on_enter=None)
```

- `child` — the node this branch leads to, or `None` for a branch that
  ends immediately (equivalent to an implicit leaf worth `value`).
- `value` — the edge payoff: a plain float, or a `Value(...)` object when
  the edge needs scenarios or uncertainty attached (see below).
- `time` — years from t=0 this edge's payoff accrues, discounted by the
  tree's `discount_rate` exactly like a leaf's `time`.
- `active` — `False` excludes the branch from rollback; this is what
  `.force()`/`.deactivate()` toggle.
- `label` — the name shown in `display()`/`save_svg()` and looked up by
  `.force(label)`, `.deactivate(label)`, `sensitivity()`'s crossover
  output, and `risk_profile_by_branch()`.
- `on_enter` — an optional zero-arg callable fired when this branch is
  taken during rollback, for updating a shared `Context` (see below);
  not called for inactive branches.

### `DecisionNode`

A node where the decision maker picks a branch — rollback chooses the
branch with the best (max, or min if `maximize=False`) expected value.

```python
harvest = DecisionNode("harvest")
harvest.add("harvest now", value=harvest_now_value, next=LeafNode("i"))
harvest.add("wait", next=rain)  # rain is a ChanceNode built earlier
```

`.add(label, value=0, time=0, next=None)` appends one branch and returns
`self`, so calls chain. `next=` is the child node — this is how sequential
structure is built: pass another `DecisionNode` as `next=` to model a
second decision made after nature moves (genuine recourse), the same way
`next=` can point to a `ChanceNode` or `LeafNode`.

Other mutation methods, all facilitator-only:
- `.force(label)` — deactivate every branch except the named one. Used to
  compute EVPC-style "what if this branch were certain" scenarios (see
  Freemark Abbey's spores example below) as well as simple what-ifs.
- `.unforce()` — reactivate every branch.
- `.deactivate(label_or_list)` / `.activate(label_or_list)` — toggle one
  or more branches without touching the rest.

### `ChanceNode`

A node where nature picks a branch according to probabilities.

```python
rain = ChanceNode("rain")
rain.add("heavy", prob=0.5, next=sell_as)
rain.add("light", prob=0.5, next=spores)
```

`.add(label, prob=None, value=0, time=0, next=None)` — same shape as
`DecisionNode.add()` plus `prob=`. Passing `prob=` on every `.add()` call
builds up `node.probs.base` incrementally; alternatively add all branches
first (`prob=None`) and set every probability at once:

```python
appeal = ChanceNode("appeal")
appeal.add("high", next=LeafNode("a"))
appeal.add("medium", next=LeafNode("b"))
appeal.add("low", next=LeafNode("c"))
appeal.set_probs([0.20, 0.50, 0.30])
```

`.set_probs()` validates the length matches the branch count; if the
values don't sum to 1.0 it normalises and warns (or raises, in strict
mode) rather than silently using unnormalised weights.

Probabilities live on `node.probs`, a `Prob` object: `node.probs.base` is
the plain `list[float]`, in branch order — read or mutate this list
directly for a manual sensitivity sweep (see the Freemark and Texaco
examples). `node.probs.uncertainty` is where a `Range(...)` is attached
for `tree.sensitivity()` to auto-detect (see "Sensitivity analysis"
below).

`redistribute=True` (the default) means deactivating one outcome
renormalises the remaining active probabilities rather than leaving them
summing to less than 1.

Same `.force()` / `.unforce()` / `.deactivate()` / `.activate()` methods
as `DecisionNode`, with the same label-based lookup — `.force(label)` on a
`ChanceNode` is how "assume this outcome is certain" scenarios (perfect
control, EVPC) are built:

```python
spores_evpc.force("forms")             # only this branch is active
spores_evpc.probs.base = [1.0, 0.0]    # so the display shows p=1.00, not the original odds
```
Note the second line: `.force()` only toggles `active`, it does not touch
`probs.base` — for a display that shows the forced certainty as an honest
probability (rather than the original odds with one branch struck through),
set `probs.base` directly as well, as done in the Freemark Abbey EVPC
worked solution.

### `LeafNode`

Terminal node, carries the payoff for one root-to-leaf path.

```python
LeafNode("i")                       # value defaults to 0 — all payoff already on the edges
LeafNode("g", value=125_000)        # cumulative mode: a plain float
LeafNode("h", value=lambda **kw: kw["price"] * kw["quantity"])   # formula mode
```

- `value` — either a float (cumulative mode: this leaf just adds a fixed
  amount to whatever accumulated on the branches leading to it) or a
  callable (formula mode). The callable receives one keyword argument per
  ancestor node name along the path taken to reach it (e.g. `price=...,
  quantity=...` if those are ancestor node names), plus a reserved
  `_path` kwarg holding `{node_name: branch_label}` for the path taken.
  `**kwargs` is forbidden in the callable signature — that's a
  `ValidationError` at tree-construction time, not a silent runtime bug.
- `time` — years from t=0 the leaf payoff occurs, discounted like a
  branch edge value. **Branch edge values and the leaf value are both
  accumulated, they are not alternatives** — put a cost that occurs
  partway down the path on the branch, and the terminal payoff on the
  leaf; don't double up the same cash flow in both places.

Leaf names only need to be unique among a node's own path (labels like
`"a"`, `"b"`, `"c"` reused across unrelated subtrees are normal and appear
throughout the worked examples) — but every name must still be unique
among its own ancestors on any single root-to-leaf path, exactly like any
other node.

### `LogicNode`

A node that routes by a boolean condition instead of by expected value or
chance — for structure driven by an if/then rule (e.g. "if cumulative cost
so far exceeds budget, only the cheap option is available") rather than a
probability or a choice. Less commonly needed than the other three; use it
only when a request is genuinely "which branch applies depends on a
condition evaluated during rollback," not as a decision or chance
substitute.

```python
budget_check = LogicNode(
    "budget_check",
    condition=lambda **ctx: ctx["spend_so_far"] < 100_000,  # binary shorthand
    children=[under_budget_node, over_budget_node],
    values=[0, 0],
    labels=["under", "over"],
)
```
`condition=` is binary shorthand (exactly 2 branches: `True` →
`branches[0]`, `False` → `branches[1]`); `conditions=[...]` gives one
callable per branch for more than two branches, and if more than one
condition evaluates `True`, the EV used is the **average** across all
`True` branches, not an error. Zero `True` conditions raises
`RollbackError("no_true_condition")` at rollback time. `.deactivate()` /
`.activate()` work the same as the other node types; there's no `.force()`
or `.add()` on `LogicNode`.

## `DecisionTree`

```python
tree = DecisionTree(root=harvest, maximize=True, discount_rate=0.0)
```

- `root` — the top node, usually a `DecisionNode`.
- `maximize` — `True` (default): decision nodes pick the branch with the
  highest EV. `False`: lowest (e.g. minimizing cost).
- `discount_rate` — annual rate; every branch/leaf `time` is discounted as
  `value / (1 + discount_rate) ** time`.
- `scenarios` — optional tree-level list of `Scenario(name, value=dict,
  weight=1.0)` for coordinated multi-node what-ifs, selected later via
  `tree.rollback(scenario="name")`.

Construction runs validation immediately (`ValidationError` on the first
error found — cycles, name collisions, missing probabilities in strict
mode, etc.) unless the tree is malformed in a way `check()` reports as a
warning rather than an error, in which case a default was silently
applied. Call `tree.check()` to get every `ModelIssue` (errors and
warnings) without raising, if a student wants to inspect what's wrong
before fixing it.

### Solving: `tree.rollback()`

```python
result = tree.rollback()
```

Folds the tree back and returns a `RollbackResult` (a fresh object every
call — nothing here is cached across mutations, so re-run `rollback()`
after changing any probability, value, or active flag). Optional
arguments, all for risk-attitude modeling:

- `utility="exponential"` or `"logarithmic"`, plus `risk_tolerance=<float>`
  — folds back on expected utility instead of expected value, and
  populates `result.ce` (the certainty equivalent) as well as `result.ev`
  (which then reports in CE units, not raw EU).
- `utility=<callable>` — a custom utility function, with `inverse=` also
  supplied if a CE is wanted back in payoff units.
- `scenario="name"` — solve using one of `tree.scenarios`'s named
  parameter overrides instead of the base case.

With no arguments, this is plain risk-neutral expected-value rollback —
the case used in every one of the worked examples below.

### Reading `RollbackResult`

```python
result.ev                  # expected value at the root (or CE, if a utility was supplied)
result.ce                  # certainty equivalent, or None if no utility was supplied
result.optimal_path        # e.g. ["harvest", "wait", "rain", "heavy", "sell as", "label"]
result.node_values["rain"] # EV at any named node, not just the root
result.distribution        # sorted [(payoff, probability), ...] for the optimal strategy, summing to 1.0
result.policy              # a DecisionTree with every non-optimal decision branch deactivated
```

`result.node_values` is the answer to "what's the EV of choosing branch X"
— e.g. comparing `harvest_now_value` against `result.node_values["rain"]`
to show why "wait" wins, exactly as done in both the Freemark Abbey and
Texaco solutions:
```python
result = tree.rollback()
print(f"EV(harvest now) = ${harvest_now_value:,.0f}")
print(f"EV(wait)        = ${result.node_values['rain']:,.0f}")
print(f"Optimal path: {result.optimal_path}")
```

`result.percentile(p)`, `result.var(alpha)` (same as `percentile`),
`result.cvar(alpha)` — read off the outcome distribution directly, no
need to rebuild a table from `result.distribution` by hand for a simple
percentile/VaR/CVaR request.

### Displaying a tree or a result

```python
tree.display(compact=True)                     # unsolved structure
result.display(compact=True)                   # solved tree + outcome distribution
result.display(policy_only=True)               # only the optimal path at each decision node
result.display(reveal=["sugar", "acidity"])    # fold-back walkthrough: only these nodes annotated as solved
```
`compact=True` (the default in every worked example) drops the "→ target"
arrows and the separate leaf-node row for a denser tree, meant for slides
— use it unless a student specifically wants the expanded form.
`show_distribution=False` omits the outcome-distribution table (useful
when the tree and the distribution are being exported as separate
images). `reveal=[...]` is specifically for walking a class through
solving a tree one node at a time: call `display()`/`save_svg()`
repeatedly with a progressively longer `reveal` list, as in the Freemark
Abbey fold-back sequence:
```python
fold_steps = [["sugar"], ["sugar", "acidity"], ["sugar", "acidity", "price"], ...]
for i, revealed in enumerate(fold_steps, start=1):
    result.save_svg(f"fold-{i}.svg", title=f"Folding back: {revealed[-1]}",
                     compact=True, show_distribution=False, reveal=revealed)
```

`tree.save_svg(path, ...)` / `result.save_svg(path, ...)` take the same
display arguments (`max_depth`, `expand_shared`, `width`, `title`,
`compact`, plus `show_distribution`/`reveal`/`policy_only` on the result
version) and write an SVG file instead of printing — this is how every
tree image on the slides in the worked examples is produced, always into
an `export_dir` under the class's `slides/images/` folder.

`dtree.settings` (imported as `settings`) controls session-wide display
defaults: `settings.formatter = lambda v: f"${v:,.0f}"` for how numbers
render, `settings.colors[...]` for the rich-markup palette, and
`settings.background` for light/dark/transparent SVG backgrounds — reach
for these only if a student explicitly asks to change how the tree looks,
not by default.

`tree.plot()`, `tree.plot_mermaid()`, `tree.to_dict()`,
`DecisionTree.from_dict()`, and `result.plot()`/`result.plot_mermaid()`
all raise `NotImplementedError` — they are declared but not built. Don't
suggest them, and don't invent a workaround that pretends they exist; use
`display()`/`save_svg()` for visualizing the tree itself.

## Risk profiles

```python
profiles = result.risk_profile_by_branch()          # defaults to the root DecisionNode
profiles = result.risk_profile_by_branch(node="sell as")  # any named DecisionNode
```
Returns a `RiskProfileCollection`: one `RiskProfile` (`.label`,
`.distribution`, `.ev`) per branch of that decision node, keyed by branch
label (`profiles["wait"]`, `profiles.profiles` for the raw dict).
`profiles.plot(view="both" | "cdf" | "histogram")` returns a matplotlib
`Figure` overlaying every branch's outcome distribution — this is the
standard "risk profile" chart in both the Freemark Abbey and Texaco
solutions:
```python
profiles = result.risk_profile_by_branch()
fig = profiles.plot(view="both")
```

To build a **custom** risk-profile comparison (e.g. net of a cost that
only applies to one branch, or comparing a no-information baseline
against a solved subtree from a different rollback), construct
`RiskProfile`/`RiskProfileCollection` directly rather than trying to force
`risk_profile_by_branch()` to do it — both classes are plain public
constructors for exactly this:
```python
from decision_suite.tree.core import RiskProfile, RiskProfileCollection

wait_net = [(payoff - SPORES_COST, prob) for payoff, prob in wait_raw.distribution]
profiles = RiskProfileCollection(profiles={
    "harvest now": profiles_raw["harvest now"],
    "wait, net of spores cost": RiskProfile(
        label="wait, net of spores cost", distribution=wait_net, ev=wait_raw.ev - SPORES_COST,
    ),
})
fig = profiles.plot(view="both")
```
`result.plot_distribution(view="both" | "cdf" | "histogram")` is the
single-strategy version (just the optimal policy's own distribution, no
branch comparison) — use it when there's nothing to compare against yet.

## Sensitivity analysis

Attach a `Range` to whatever should vary, then call `tree.sensitivity()`.

```python
from decision_suite.tree.core import Range

rain.probs.uncertainty = Range(0.0, 1.0, n=101)     # a probability
sugar.branches[0].value = Value(sugar_high_value, uncertainty=Range(5, 9, step=0.5))  # a leaf/edge value
```
`Range(low, high, step=...)` or `Range(low, high, n=...)` — exactly one of
`step`/`n`. It satisfies the same `.mean()`/`.rvs()` protocol
`decision_suite.sim` distributions use, and exposes `.points` (the
evenly-spaced sweep values) for anything composed by hand.

```python
sr = tree.sensitivity(include=["rain"])                 # 1-way
sr = tree.sensitivity(include=["rain", "spores"])        # 2-way (also runs both 1-way sweeps)
sr = tree.sensitivity()                                  # n-way: auto-detects every node with a Range attached
```
`include=None` auto-detects every uncertain node; the method
(`"1-way"`/`"2-way"`/`"n-way"`) is inferred from `len(include)` unless
passed explicitly. Returns a `SensitivityResult`:

- `sr.sweeps` — `{node_name: [(x, ev, optimal_branch_label), ...]}`, one
  1-way sweep per included variable.
- `sr.grid` / `sr.grid_names` — populated only for `"2-way"`: a 2D grid
  keyed by `(x1, x2)` giving `(ev, optimal_branch)`.
- `sr.base_ev`, `sr.base_branch`, `sr.base_values` — the base case, for
  drawing a reference line/marker.
- `sr.strategy_region()` — `{node_name: {"crossovers": [x, ...]}}`, the
  parameter values where the optimal branch changes. This is the direct
  answer to "at what probability does the decision flip":
  ```python
  rain_crossover = sr.strategy_region()["rain"]["crossovers"][0]
  ```

`sr.plot(type=..., node=..., nodes=..., view="ev")`:
- `type="one_way"`, `node="rain"` — EV vs. the parameter, needs `node=`.
- `type="strategy_region"` — 1-way policy-region bands (`node=`) or a 2-way
  heatmap (`nodes=[n1, n2]`), exactly the two-way chart in the Freemark
  Abbey worked example: `sr_2way.plot(type="strategy_region", nodes=["rain", "spores"])`.
- `type="tornado"` — every included variable ranked by EV swing.
- `type="spider"` — every variable normalised to % change from base.

**When a request doesn't fit the `Value`/`Prob` + `Range` shape** — e.g.
"tornado across these eight named parameters, including branch values that
aren't wrapped in `Value(uncertainty=...)`" — don't force it through
`tree.sensitivity()`. Compose it directly against `tree.rollback()`,
exactly as the Texaco worked solution does: a small `set_branch_value`/
`set_prob` helper, a `±10%` sweep per parameter, then a plain matplotlib
horizontal-bar tornado. This is the tree-module analogue of `decision_suite.sim`'s
"a few lines against `model.run()`" philosophy — `tree.sensitivity()`
covers the common, well-typed case; a bespoke sweep over arbitrary
branches/probabilities is still just a loop over `tree.rollback()`.

### Risk-attitude sensitivity

```python
ra = tree.risk_attitude_sensitivity(risk_tolerance=(1_000, 500_000))   # sweep mode
ra = tree.risk_attitude_sensitivity(risk_tolerance=some_distribution)   # uncertain mode
ra = tree.risk_attitude_sensitivity(utility_factory=make_utility_fn)    # factory mode
```
Answers "does the recommended decision change as the decision maker
becomes more/less risk averse." Returns a `RiskAttitudeSensitivityResult`:
`.ce_by_branch`, `.optimal_branch_by_sample`, `.crossovers` (sweep mode
only), `.reversal_probability` (uncertain/factory modes),
`.risk_tolerances`. `.plot()` renders whichever chart fits the mode (CE
curves with crossovers marked / mean CE with percentile bands / overlaid
CE histograms).

## Value of information: `.flip()`

`tree.flip(decision=..., chance=...)` returns a **new** `DecisionTree`
(the original is untouched) with information placed before the named
decision — this is how EVPI and EVSI are computed without hand-building
the flipped structure. It's real work worth using rather than re-deriving:
Bayes' rule, marginal probabilities, and correctly grafting a copied
subtree onto every signal branch are all handled internally.

**EVPI** — perfect information, `test=None, likelihood=None`:
```python
vopi_tree = tree.flip(decision="harvest", chance="rain")
vopi_result = vopi_tree.rollback()
evpi = vopi_result.ev - ev_no_information
```
The named chance node (`"rain"`) becomes the new root; each of its
outcome branches leads to a copy of the original decision subtree where
that chance node is degenerate (probability 1 on the now-known state).

**EVSI** — imperfect information from an actual signal, `test=` +
`likelihood=` supplied:
```python
test_node = ChanceNode("superdoppler")
test_node.add("predicted heavy", next=LeafNode("t1"))
test_node.add("predicted light", next=LeafNode("t2"))

likelihood = [
    [P_PRED_HEAVY_GIVEN_HEAVY, P_PRED_LIGHT_GIVEN_HEAVY],   # given heavy rain (row = true state)
    [P_PRED_HEAVY_GIVEN_LIGHT, P_PRED_LIGHT_GIVEN_LIGHT],   # given light rain
]                                                             # columns = test outcomes, same order as test_node.branches

vosi_tree = tree.flip(decision="harvest", chance="rain", test=test_node, likelihood=likelihood)
vosi_result = vosi_tree.rollback()
evsi = vosi_result.ev - ev_no_information
```
`likelihood` rows must match `chance` node's branch order and sum to 1.0
each; it also accepts a `dict` form (`{state_label: [p...]}` or
`{state_label: Prob([...])}`, the `Prob` form specifically so likelihood
values can themselves be swept in a sensitivity analysis). `test`'s branch
labels and edge values are carried into the flipped tree; only its
probabilities are replaced with the computed marginals.

Always sanity-check a `.flip()` result against a hand-built tree the first
time it's used in a new case (as both VOPI and VOSI worked solutions do
with an `assert abs(auto.ev - manual.ev) < 1e-6`) — it's cheap insurance
and a good thing to show a student once, not something to repeat in every
cell afterward.

**`.flip()` needs a plain chance node with `probs.base` set** — if the
decision named isn't the root, it copies the whole tree and grafts the
flipped structure back in at the right point, so `decision=` doesn't have
to be the tree's own root.

## Copying subtrees

`node.copy(independent=False, deep=True)` — every node type has this.
`deep=True` (default) copies the whole subtree; `independent=True` gives
every copied node a fresh `event_id` (needed when the same subtree
structure is reused in two different branches of the same tree and must
be tracked separately during rollback, e.g. `sell_as.copy()` reused under
both a "predicted heavy" and a "predicted light" branch in the VOSI
walkthrough). Reuse `.copy()` rather than reconstructing an identical
subtree by hand when a case genuinely reuses the same downstream structure
in more than one place.

## Common mistakes to avoid

- **Don't forget `result = tree.rollback()` returns a new object every
  call.** Mutating a probability or value and reading a stale `result`
  gives the old answer — always re-roll back after a mutation.
- **`.force()` only changes `active`, not `probs.base`.** If the display
  should show the forced branch as probability 1 (not just the other
  branches struck through), set `probs.base` explicitly too — see the
  `ChanceNode.force()` example above.
- **Edge value vs. leaf value are both accumulated, never a substitute for
  each other.** A cost partway down a path belongs on the branch that
  incurs it; the final payoff belongs on the leaf. Putting the same cash
  flow in both double-counts it.
- **`node.probs.base` is a plain list in branch order** — mutate index by
  index for a manual sweep (`node.probs.base[0] = x; node.probs.base[1] =
  1 - x`), always restoring the original values afterward (wrap the sweep
  in `try/finally`, as both worked sensitivity examples do).
- **`tree.plot()`, `plot_mermaid()`, `to_dict()`, `from_dict()` are not
  implemented.** Use `display()` / `save_svg()` for visualizing a tree;
  don't propose serializing a tree to/from a dict.
- **There is no `Model`-style single "run the case" object** — a
  `DecisionTree` is built once per case (occasionally with a
  `build_base_tree()` helper function returning the key nodes, so the same
  structure can be reused for the base case, an EVPC scenario, and a VOPI
  flip, as in the Freemark Abbey notebooks) and solved with `.rollback()`
  as many times as needed.

## Where the real ceiling is

If a request needs something no combination of the above can produce —
serializing a tree to JSON, an interactive `plot()`/`plot_mermaid()`
(both declared but intentionally unimplemented), a genuinely new solving
mode — that's a real package change, not a notebook cell. Stop and flag
it rather than working around a missing feature with a fragile
reimplementation.
