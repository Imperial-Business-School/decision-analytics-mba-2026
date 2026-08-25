"""
dtree.core — bare node classes and all public types.

No mutation methods (.add, .set_probs, .force, etc.) live here.
Those are on the facilitator subclasses in dtree.facilitator.

Import from here for type annotations or declarative tree construction.
"""

from __future__ import annotations

import types as _types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, runtime_checkable
from uuid import UUID, uuid4

import numpy as np


# ===========================================================================
# EXCEPTIONS
# ===========================================================================

class DtreeError(Exception):
    """Base class for all dtree errors."""


class ValidationError(DtreeError):
    """
    Raised at connection time or composition time when the tree structure
    violates a rule that cannot be deferred.

    Attributes
    ----------
    node : str | None
        Name of the offending node, or None for tree-level errors.
    rule : str
        Short machine-readable rule identifier, e.g. "cycle_detected",
        "name_collision", "missing_probs".
    """
    def __init__(self, message: str, node: str | None = None, rule: str = "") -> None:
        super().__init__(message)
        self.node = node
        self.rule = rule


class FlipError(DtreeError):
    """
    Raised by flip() when the signal structure is inconsistent.

    Attributes
    ----------
    reason : str
        "label_mismatch"     : sample outcome labels ≠ source outcome labels
        "asymmetric_signals" : signal ChanceNodes under sample have different branch labels
        "invalid_likelihood" : a signal ChanceNode's probs do not sum to 1.0 (±1e-6)
        "ambiguous_before"   : before=None but multiple DecisionNode ancestors found
    """
    def __init__(self, message: str, reason: str = "") -> None:
        super().__init__(message)
        self.reason = reason


class RollbackError(DtreeError):
    """
    Raised during rollback when the tree cannot be evaluated.

    Attributes
    ----------
    node : str | None
        Name of the node that triggered the error.
    reason : str
        "no_true_condition" : a LogicNode has no True condition at rollback time
        "missing_probs"     : a ChanceNode has no probabilities (strict mode)
    """
    def __init__(self, message: str, node: str | None = None, reason: str = "") -> None:
        super().__init__(message)
        self.node = node
        self.reason = reason


class SerializationError(DtreeError):
    """Raised by to_dict() when the tree cannot be serialised to JSON."""


# ===========================================================================
# DISTRIBUTION PROTOCOLS
# ===========================================================================

@runtime_checkable
class DistributionProtocol(Protocol):
    """
    Structural protocol for univariate distributions.
    Any object with .mean() → float and .rvs(size=n) → ndarray satisfies this.
    scipy.stats frozen distributions and Range both qualify.
    """
    def mean(self) -> float: ...
    def rvs(self, size: int = 1) -> np.ndarray: ...


@runtime_checkable
class DirichletProtocol(Protocol):
    """
    Structural protocol for multivariate distributions over probability vectors.
    .mean() returns ndarray shape (k,) — distinguishes it from DistributionProtocol.
    scipy.stats.dirichlet([...]) satisfies this automatically.
    """
    def mean(self) -> np.ndarray: ...
    def rvs(self, size: int = 1) -> np.ndarray: ...


# ===========================================================================
# SUPPORTING TYPES
# ===========================================================================

@dataclass
class Scenario:
    """
    A named scenario for sensitivity analysis.

    value type depends on context:
      Value.scenarios       → float
      Prob.scenarios        → list[float]
      DecisionTree.scenarios → dict[str, float]

    Weights are normalised across all Scenario objects in the same list.
    Equal weights (default 1) → equally plausible scenarios.
    Raises ValidationError if all weights in a list are 0.
    """
    name: str
    value: float | list[float] | dict[str, float]
    weight: float = 1.0


class Value:
    """
    Full specification for a single edge value.

    Parameters
    ----------
    base : float
        Base-case edge value.
    scenarios : list[Scenario] | None
        Named deterministic overrides for what-if analysis.
    uncertainty : DistributionProtocol | None
        Any object with .mean() and .rvs(size=) for Monte Carlo / SA.

    A plain float is accepted wherever Value is expected (handled at call sites).
    """
    base: float
    scenarios: list[Scenario] | None
    uncertainty: DistributionProtocol | None

    def __init__(
        self,
        base: float,
        scenarios: list[Scenario] | None = None,
        uncertainty: DistributionProtocol | None = None,
    ) -> None:
        self.base = base
        self.scenarios = scenarios
        self.uncertainty = uncertainty
        if scenarios is not None and all(s.weight == 0 for s in scenarios):
            raise ValidationError(
                "All scenario weights are 0 — at least one must be non-zero.",
                rule="zero_weights",
            )


class Prob:
    """
    Probability specification for a ChanceNode or LogicNode.

    Parameters
    ----------
    base : list[float] | None
        Base-case probabilities. Dimension validated at connection time.
    scenarios : list[Scenario] | None
        Named probability-vector overrides for what-if analysis.
    uncertainty : DirichletProtocol | None
        Distribution over probability vectors for Monte Carlo.
    """
    base: list[float] | None
    scenarios: list[Scenario] | None
    uncertainty: DirichletProtocol | None

    def __init__(
        self,
        base: list[float] | None = None,
        scenarios: list[Scenario] | None = None,
        uncertainty: DirichletProtocol | None = None,
    ) -> None:
        self.base = base
        self.scenarios = scenarios
        self.uncertainty = uncertainty
        if scenarios is not None and all(s.weight == 0 for s in scenarios):
            raise ValidationError(
                "All scenario weights are 0 — at least one must be non-zero.",
                rule="zero_weights",
            )


class Range:
    """
    A discrete or continuous uniform sweep over a value range.
    Satisfies DistributionProtocol: .mean() and .rvs(size=).

    Exactly one of step or n must be supplied.
    Raises ValidationError if (high - low) is not divisible by step.
    """
    low: float
    high: float
    step: float | None
    n: int | None

    def __init__(
        self,
        low: float,
        high: float,
        step: float | None = None,
        n: int | None = None,
    ) -> None:
        if (step is None) == (n is None):
            raise ValidationError(
                "Exactly one of step or n must be supplied to Range.",
                rule="range_spec",
            )
        self.low = low
        self.high = high
        self.step = step
        self.n = n

        if step is not None:
            span = high - low
            remainder = abs(span % step)
            if remainder > 1e-9 * abs(step):
                raise ValidationError(
                    f"(high - low) = {span} is not divisible by step = {step}.",
                    rule="range_step",
                )
            n_points = round(span / step) + 1
            self._points: np.ndarray = np.linspace(low, high, n_points)
        else:
            self._points = np.linspace(low, high, n)

    def mean(self) -> float:
        return (self.low + self.high) / 2.0

    def rvs(self, size: int = 1) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=size)

    @property
    def points(self) -> np.ndarray:
        """Evenly-spaced sweep points for tornado / spider plots."""
        return self._points


@dataclass(frozen=True)
class ModelIssue:
    """
    A single validation finding from tree.check().

    level   : "error" means rollback() will raise; "warning" means a default was applied.
    node    : name of the offending node, or None for tree-level issues.
    message : human-readable description of the violated constraint.
    rule    : machine-readable identifier matching ValidationError.rule vocabulary.
    """
    level: Literal["error", "warning"]
    node: str | None
    message: str
    rule: str = ""


# ===========================================================================
# CONTEXT
# ===========================================================================

class Context:
    """
    Simple container for path state shared between on_enter callbacks and
    downstream LogicNode conditions or LeafNode callables.

    Attribute names and initial values are set as keyword arguments.
    Thin wrapper over types.SimpleNamespace; exposed under this name so
    user code never needs to import from types.

    Reset to known defaults before each rollback — the engine resets nothing.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._ns = _types.SimpleNamespace(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ns, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_ns":
            super().__setattr__(name, value)
        else:
            setattr(self._ns, name, value)

    def set(self, **kwargs: Any) -> None:
        """Set one or more attributes in a single expression — usable in lambdas."""
        for k, v in kwargs.items():
            setattr(self._ns, k, v)


# ===========================================================================
# BRANCH
# ===========================================================================

@dataclass
class Branch:
    """
    A single branch from a node. Carries only edge-level properties.
    Probabilities (ChanceNode) and conditions (LogicNode) live on the parent node.

    child    : child node, or None for an implicit terminal.
    value    : edge payoff — scalar or Value with scenarios/uncertainty.
    time     : accrual time in years for PV calculation.
    active   : if False, excluded from rollback.
    label    : display label shown in tree output and plots.
    on_enter : optional side-effect called when this branch is taken during rollback.
               Use to update shared Context state before downstream nodes evaluate.
               Not called for inactive branches.
    """
    child: Node | None = None
    value: float | Value = 0
    time: float = 0
    active: bool = True
    label: str | None = None
    on_enter: Callable[[], None] | None = None


# ===========================================================================
# COPY HELPERS  (module-level so all node types can share them)
# ===========================================================================

def _copy_branch(b: Branch, independent: bool, deep: bool, memo: dict) -> Branch:
    child = _copy_node_impl(b.child, independent, deep, memo) if deep else None
    return Branch(
        child=child,
        value=b.value,
        time=b.time,
        active=b.active,
        label=b.label,
        on_enter=b.on_enter,
    )


def _copy_node_impl(node: Node | None, independent: bool, deep: bool, memo: dict) -> Node | None:
    if node is None:
        return None
    node_id = id(node)
    if node_id in memo:
        return memo[node_id]
    return node._copy_impl(independent, deep, memo)


# ===========================================================================
# NODES
# ===========================================================================

class Node(ABC):
    """
    Abstract base for all node types.

    name     : must be unique among all ancestors on any root-to-leaf path.
    event_id : UUID assigned at construction; inherited by copy() unless independent=True.
    """
    name: str
    event_id: UUID

    @abstractmethod
    def display(self, depth: int | None = None) -> None:
        """Print this node's structure to stdout."""

    @abstractmethod
    def copy(self: Node, independent: bool = False, deep: bool = True) -> Node:
        """Copy this node, optionally including its entire subtree."""

    @abstractmethod
    def _copy_impl(self, independent: bool, deep: bool, memo: dict) -> Node:
        """Internal recursive copy used by copy() and _copy_node_impl()."""


# ---------------------------------------------------------------------------
# DecisionNode
# ---------------------------------------------------------------------------

class DecisionNode(Node):
    """
    A node where the decision maker chooses one branch.
    Optimal branch determined by max (or min) EU during rollback.

    Accept either branches= (primary) or parallel lists (children/values/times/active).
    Mutually exclusive — ValidationError if both supplied.
    """
    branches: list[Branch]

    def __init__(
        self,
        name: str,
        branches: list[Branch] | None = None,
        *,
        children: list[Node | None] | None = None,
        values: list[float | Value] | None = None,
        times: list[float] | None = None,
        active: list[bool] | None = None,
    ) -> None:
        parallel = any(x is not None for x in (children, values, times, active))
        if branches is not None and parallel:
            raise ValidationError(
                f"DecisionNode '{name}': branches= and parallel-list params are mutually exclusive.",
                node=name, rule="ambiguous_construction",
            )
        self.name = name
        self.event_id = uuid4()

        if branches is not None:
            self.branches = list(branches)
        elif children is not None:
            n = len(children)
            vals  = values if values is not None else [0.0] * n
            times_ = times  if times  is not None else [0.0] * n
            acts  = active if active is not None else [True] * n
            self.branches = [
                Branch(child=c, value=v, time=t, active=a)
                for c, v, t, a in zip(children, vals, times_, acts)
            ]
        else:
            self.branches = []

    def display(self, depth: int | None = None) -> None:
        from ._display import display_node
        display_node(self, depth)

    def copy(self, independent: bool = False, deep: bool = True) -> DecisionNode:
        return self._copy_impl(independent, deep, {})

    def _copy_impl(self, independent: bool, deep: bool, memo: dict) -> DecisionNode:
        node_id = id(self)
        if node_id in memo:
            return memo[node_id]
        new = type(self).__new__(type(self))
        memo[node_id] = new
        new.name = self.name
        new.event_id = uuid4() if independent else self.event_id
        new.branches = [_copy_branch(b, independent, deep, memo) for b in self.branches]
        return new


# ---------------------------------------------------------------------------
# ChanceNode
# ---------------------------------------------------------------------------

class ChanceNode(Node):
    """
    A node where nature selects a branch according to probabilities.

    probs        : Prob object carrying base probs, scenarios, and uncertainty.
    redistribute : if True, deactivating an outcome renormalises remaining probs.
    """
    branches: list[Branch]
    probs: Prob | None
    redistribute: bool

    def __init__(
        self,
        name: str,
        branches: list[Branch] | None = None,
        probs: Prob | None = None,
        redistribute: bool = True,
        *,
        children: list[Node | None] | None = None,
        values: list[float | Value] | None = None,
        times: list[float] | None = None,
        active: list[bool] | None = None,
    ) -> None:
        parallel = any(x is not None for x in (children, values, times, active))
        if branches is not None and parallel:
            raise ValidationError(
                f"ChanceNode '{name}': branches= and parallel-list params are mutually exclusive.",
                node=name, rule="ambiguous_construction",
            )
        self.name = name
        self.event_id = uuid4()
        self.probs = probs
        self.redistribute = redistribute

        if branches is not None:
            self.branches = list(branches)
        elif children is not None:
            n = len(children)
            vals   = values if values is not None else [0.0] * n
            times_ = times  if times  is not None else [0.0] * n
            acts   = active if active is not None else [True] * n
            self.branches = [
                Branch(child=c, value=v, time=t, active=a)
                for c, v, t, a in zip(children, vals, times_, acts)
            ]
        else:
            self.branches = []

    def display(self, depth: int | None = None) -> None:
        from ._display import display_node
        display_node(self, depth)

    def copy(self, independent: bool = False, deep: bool = True) -> ChanceNode:
        return self._copy_impl(independent, deep, {})

    def _copy_impl(self, independent: bool, deep: bool, memo: dict) -> ChanceNode:
        node_id = id(self)
        if node_id in memo:
            return memo[node_id]
        new = type(self).__new__(type(self))
        memo[node_id] = new
        new.name = self.name
        new.event_id = uuid4() if independent else self.event_id
        new.redistribute = self.redistribute
        new.probs = (
            Prob(
                base=list(self.probs.base) if self.probs.base is not None else None,
                scenarios=self.probs.scenarios,
                uncertainty=self.probs.uncertainty,
            )
            if self.probs is not None else None
        )
        new.branches = [_copy_branch(b, independent, deep, memo) for b in self.branches]
        return new


# ---------------------------------------------------------------------------
# LeafNode
# ---------------------------------------------------------------------------

class LeafNode(Node):
    """
    Terminal node. Carries the payoff for one root-to-leaf path.

    value : float (cumulative mode) or callable (formula mode).
            Callable receives named params matching ancestor node names;
            reserved param _path receives {node_name: branch_label} dict.
            **kwargs is forbidden in callable — ValidationError at composition time.
    time  : periods from t=0 at which the terminal payoff occurs. Discounted
            by the tree's discount_rate exactly like a branch edge value.
            Note: branch edge values and the leaf value are both accumulated —
            they represent different cash flows (e.g., a drilling cost on the
            branch and the oil revenue on the leaf). Do not put the same
            payoff in both places.
    """
    value: float | Callable[..., float]
    time: float

    def __init__(self, name: str, value: float | Callable[..., float] = 0, time: float = 0) -> None:
        self.name = name
        self.event_id = uuid4()
        self.value = value
        self.time = time

    def display(self, depth: int | None = None) -> None:
        from ._display import display_node
        display_node(self, depth)

    def copy(self, independent: bool = False, deep: bool = True) -> LeafNode:
        return self._copy_impl(independent, deep, {})

    def _copy_impl(self, independent: bool, deep: bool, memo: dict) -> LeafNode:
        node_id = id(self)
        if node_id in memo:
            return memo[node_id]
        new = type(self).__new__(type(self))
        memo[node_id] = new
        new.name = self.name
        new.event_id = uuid4() if independent else self.event_id
        new.value = self.value
        new.time = self.time
        return new


# ---------------------------------------------------------------------------
# LogicNode
# ---------------------------------------------------------------------------

class LogicNode(Node):
    """
    A node that routes to branches by boolean conditions rather than EV.

    conditions : one callable per branch, evaluated at rollback time.
        Multiple True  → average EV of all True branches.
        None True      → RollbackError("no_true_condition").

    Binary shorthand: condition= (single callable, exactly 2 branches).
        True → branches[0]; False → branches[1].
        Normalised to conditions=[condition, negation] at construction time.

    Shorthand without explicit branches: pass children/values/labels alongside
        condition= or conditions=; converted to Branch records at construction.
    """
    branches: list[Branch]
    conditions: list[Callable[..., bool]]

    def __init__(
        self,
        name: str,
        branches: list[Branch] | None = None,
        conditions: list[Callable[..., bool]] | None = None,
        *,
        condition: Callable[..., bool] | None = None,
        children: list[Node | None] | None = None,
        values: list[float | Value] | None = None,
        labels: list[str] | None = None,
    ) -> None:
        if conditions is not None and condition is not None:
            raise ValidationError(
                f"LogicNode '{name}': condition= and conditions= are mutually exclusive.",
                node=name, rule="ambiguous_construction",
            )
        self.name = name
        self.event_id = uuid4()

        # Build branches from shorthand if needed
        if branches is None and children is not None:
            n = len(children)
            vals   = values if values is not None else [0.0] * n
            labs   = labels if labels is not None else [None] * n
            branches = [Branch(child=c, value=v, label=l)
                        for c, v, l in zip(children, vals, labs)]

        self.branches = list(branches) if branches is not None else []

        # Normalise binary shorthand → conditions list
        if condition is not None:
            if len(self.branches) != 2:
                raise ValidationError(
                    f"LogicNode '{name}': binary shorthand (condition=) requires exactly 2 branches.",
                    node=name, rule="binary_condition_branch_count",
                )
            _cond = condition
            self.conditions = [_cond, lambda *a, **kw: not _cond(*a, **kw)]
        elif conditions is not None:
            self.conditions = list(conditions)
        else:
            self.conditions = []

        if self.conditions and len(self.conditions) != len(self.branches):
            raise ValidationError(
                f"LogicNode '{name}': len(conditions)={len(self.conditions)} "
                f"!= len(branches)={len(self.branches)}.",
                node=name, rule="conditions_branch_mismatch",
            )

    def display(self, depth: int | None = None) -> None:
        from ._display import display_node
        display_node(self, depth)

    def copy(self, independent: bool = False, deep: bool = True) -> LogicNode:
        return self._copy_impl(independent, deep, {})

    def _copy_impl(self, independent: bool, deep: bool, memo: dict) -> LogicNode:
        node_id = id(self)
        if node_id in memo:
            return memo[node_id]
        new = type(self).__new__(type(self))
        memo[node_id] = new
        new.name = self.name
        new.event_id = uuid4() if independent else self.event_id
        new.conditions = list(self.conditions)
        new.branches = [_copy_branch(b, independent, deep, memo) for b in self.branches]
        return new


# ===========================================================================
# SETTINGS
# ===========================================================================

@dataclass
class GlobalSettings:
    """
    Session-wide defaults. Access as ``dtree.settings``.

    Attributes
    ----------
    strict : bool
        False (default) — infer defaults where possible.
        True — all omissions are ValidationErrors.
    style : str
        "rich" (default), "auto", or "plain".
    formatter : callable(float) -> str | None
        Number formatter used for all displayed values and EVs.
        None (default) uses the built-in comma-separated integer formatter.
        Example: ``dtree.settings.formatter = lambda v: f"${v:,.0f}"``
    colors : dict[str, str]
        Rich markup color/style strings keyed by semantic role.
        Keys: "ev", "value", "prob", "path_prob", "optimal", "inactive",
              "decision", "chance", "leaf", "logic".
        Example: ``dtree.settings.colors["ev"] = "blue"``
    background : str
        Background for interactive display() in a real Jupyter kernel.
        "transparent" (default) — no forced background; automatically
            switches between a light-safe and a dark-safe palette using the
            notebook front-end's own light/dark setting (CSS
            prefers-color-scheme), so it looks right either way.
        "light" — force a white background, regardless of notebook theme.
        "dark"  — force a dark background, regardless of notebook theme.
        Any CSS color string (e.g. "#f0e6d2") — force that exact background;
            text color is chosen automatically for contrast.
        save_svg() also honours this setting, except "transparent" (no page
            to toggle against for a static file) resolves to "light" there.
        Has no effect on plain terminal output.
        Example: ``dtree.settings.background = "light"``
    """

    def __init__(self) -> None:
        self.strict: bool = False
        self.style: str = "rich"
        self.background: str = "transparent"
        self.formatter: Callable[[float], str] | None = None
        self.colors: dict[str, str] = {
            "ev":        "dim cyan",
            "value":     "yellow",
            "prob":      "cyan",
            "path_prob": "cyan",
            "optimal":   "bold green",
            "inactive":  "dim red",
            "decision":  "bold green",
            "chance":    "bold red",
            "leaf":      "bold blue",
            "logic":     "bold magenta",
        }


@dataclass(frozen=True)
class TreeSettings:
    """Per-tree settings. Read-only after construction."""
    discount_rate: float = 0.0
    maximize: bool = True


settings = GlobalSettings()


# ===========================================================================
# DECISION TREE
# ===========================================================================

class DecisionTree:
    """
    The root object. Composition-time validation runs in __init__.

    Parameters
    ----------
    root          : root node (usually a DecisionNode)
    discount_rate : annual discount rate; PV = value / (1 + rate)^time
    maximize      : True → DecisionNodes pick max EU; False → pick min EU
    scenarios     : coordinated multi-node what-if scenarios (tree-level)
    """
    settings: TreeSettings

    def __init__(
        self,
        root: Node,
        discount_rate: float = 0.0,
        maximize: bool = True,
        scenarios: list[Scenario] | None = None,
        _skip_validation: bool = False,
    ) -> None:
        self.root = root
        self.settings = TreeSettings(discount_rate=discount_rate, maximize=maximize)
        self.scenarios = scenarios

        if not _skip_validation:
            issues = self.check()
            errors = [i for i in issues if i.level == "error"]
            if errors:
                e = errors[0]
                raise ValidationError(e.message, node=e.node, rule=e.rule)

    @classmethod
    def _from_root(cls, root: Node, tree_settings: TreeSettings) -> DecisionTree:
        """Private factory — builds a DecisionTree without re-running validation."""
        obj = cls.__new__(cls)
        obj.root = root
        obj.settings = tree_settings
        obj.scenarios = None
        return obj

    def check(self) -> list[ModelIssue]:
        """
        Run all composition-time validation rules without raising.
        Returns all issues (errors and warnings). Empty list means valid.
        """
        from ._engine import validate_tree
        return validate_tree(self)

    def rollback(
        self,
        utility: Literal["exponential", "logarithmic"] | Callable[..., float] | None = None,
        risk_tolerance: float | None = None,
        inverse: Callable[[float], float] | None = None,
        scenario: str | None = None,
    ) -> RollbackResult:
        """Evaluate and fold back the tree. Returns a fresh RollbackResult."""
        from ._engine import run_rollback
        return run_rollback(self, utility=utility, risk_tolerance=risk_tolerance,
                            inverse=inverse, scenario=scenario)

    def flip(
        self,
        decision: str,
        chance: str,
        test: "ChanceNode | None" = None,
        likelihood: "list[list[float]] | dict[str, Prob] | None" = None,
    ) -> "DecisionTree":
        """
        Return a new tree with information placed before the reference decision.

        EVPI (test=None, likelihood=None):
            The true-state chance node becomes the root. Each state branch leads to
            a copy of the decision subtree where that chance node carries a degenerate
            distribution (probability 1 on the known state, named
            '{chance}_posterior').

        EVSI (test + likelihood provided):
            The test/signal node is placed at the root with marginal probabilities.
            Each signal branch leads to a copy of the decision subtree where the
            chance node carries posterior probabilities computed via Bayes' theorem.

        Parameters
        ----------
        decision    : name of the reference DecisionNode.
        chance      : name of the true-state ChanceNode that follows the decision.
        test        : ChanceNode whose branches define test outcomes (EVSI only).
                      Branch labels and edge values are copied; probs are replaced
                      with computed marginals.
        likelihood  : P(test_outcome_j | true_state_i).
            list    : [[p00, p01, …], [p10, p11, …], …]
                      rows = true states (order matches chance.branches)
                      cols = test outcomes (order matches test.branches)
            dict    : {state_label: Prob([p0, p1, …]) or [p0, p1, …]}
                      Prob form allows sensitivity analysis on likelihood values.
            None    : identity matrix — EVPI (perfect information).
        """
        from ._flip import run_flip
        return run_flip(self, decision_name=decision, chance_name=chance,
                        test=test, likelihood=likelihood)

    def display(self, max_depth: int | None = None, expand_shared: bool = True,
                compact: bool = True) -> None:
        """Print the tree structure to stdout — no rollback values.

        expand_shared=False collapses shared subtrees inline and lists them
        once in a footer section below the main tree.
        compact=True drops the "→ target" hint on every branch and the
        separate leaf-node row, for a denser tree meant for slides.
        """
        from ._display import display_tree
        display_tree(self, max_depth, expand_shared=expand_shared, compact=compact)

    def save_svg(
        self,
        path: str,
        max_depth: int | None = None,
        expand_shared: bool = True,
        width: int = 120,
        title: str = "Decision Tree",
        compact: bool = True,
    ) -> None:
        """Save the tree display as an SVG file.

        compact=True drops the "→ target" hint on every branch and the
        separate leaf-node row, for a denser tree meant for slides.
        """
        from ._display import save_svg_tree
        save_svg_tree(self, path, max_depth=max_depth, expand_shared=expand_shared,
                      width=width, title=title, compact=compact)

    def sensitivity(
        self,
        result=None,
        include: list[str] | None = None,
        method: str | None = None,
        n: int = 1000,
        scenarios: list[str] | None = None,
    ) -> "SensitivityResult":
        """
        Sensitivity analysis over uncertain nodes.

        Auto-detects nodes with Value(uncertainty=Range(...)) or
        Prob(uncertainty=Range(...)) when include=None.

        Method inference (when method=None):
            1 variable  → "1-way"
            2 variables → "2-way"  (runs 1-way sweeps + 2D strategy-region grid)
            ≥3 variables → "n-way"  (tornado + spider)

        Parameters
        ----------
        result : RollbackResult | None
            When supplied, the utility function from this rollback is reused.
        include : list[str] | None
            Node names to sweep. None = auto-detect all uncertain nodes.
        method : str | None
            "1-way" | "2-way" | "n-way". Inferred from len(include) when None.
        """
        from ._sensitivity import run_sensitivity
        return run_sensitivity(self, result=result, include=include,
                               method=method, n=n, scenarios=scenarios)

    def risk_attitude_sensitivity(
        self,
        result: "RollbackResult | None" = None,
        risk_tolerance=None,
        utility=None,
        utility_factory=None,
        n: int = 50,
    ) -> "RiskAttitudeSensitivityResult":
        """
        Analyse how the optimal decision and CE vary with risk attitude.

        Modes (mutually exclusive):
          sweep   : risk_tolerance=(low, high) or None (auto-range from payoff spread)
          uncertain: risk_tolerance=DistributionProtocol  (sample n values)
          factory  : utility_factory=callable  (called n times, returns utility fn)

        utility : "exponential" (default) | "logarithmic" | callable
        n       : sweep points or sample count (default 50)
        """
        from ._sensitivity import run_risk_attitude_sensitivity
        return run_risk_attitude_sensitivity(
            self, result=result, risk_tolerance=risk_tolerance,
            utility=utility, utility_factory=utility_factory, n=n,
        )

    def plot(self, collapse: list[str] | None = None) -> None:
        raise NotImplementedError("plot() not yet implemented")

    def plot_mermaid(self, collapse: list[str] | None = None) -> str:
        raise NotImplementedError("plot_mermaid() not yet implemented")

    def to_dict(self) -> dict:
        raise NotImplementedError("to_dict() not yet implemented")

    @classmethod
    def from_dict(cls, spec: dict) -> DecisionTree:
        raise NotImplementedError("from_dict() not yet implemented")


# ===========================================================================
# RESULT OBJECTS
# ===========================================================================

class RollbackResult:
    """
    Return value of tree.rollback(). Treat as immutable — re-run rollback() to recompute.

    Attributes
    ----------
    ev              : expected value at the root (CE units if utility was supplied)
    ce              : certainty equivalent; None if no utility was supplied
    scenario        : name of the scenario used, or None for base case
    optimal_path    : node names on the optimal path, root to leaf
    distribution    : sorted (payoff, probability) pairs summing to 1.0
    node_values     : EV at every node in the tree, keyed by node name
    policy          : copy of tree with non-optimal decision branches deactivated
    """
    ev: float
    ce: float | None
    scenario: str | None
    optimal_path: list[str]
    distribution: list[tuple[float, float]]
    node_values: dict[str, float]
    policy: DecisionTree

    def __init__(
        self,
        ev: float,
        ce: float | None,
        optimal_path: list[str],
        distribution: list[tuple[float, float]],
        node_values: dict[str, float],
        policy: DecisionTree,
        scenario: str | None = None,
        _utility_fn=None,
        _tree: "DecisionTree | None" = None,
    ) -> None:
        self.ev = ev
        self.ce = ce
        self.scenario = scenario
        self.optimal_path = optimal_path
        self.distribution = distribution
        self.node_values = node_values
        self.policy = policy
        self._utility_fn = _utility_fn   # preserved for sensitivity re-rollbacks
        self._tree = _tree               # preserved for risk_profile_by_branch

    def percentile(self, p: float) -> float:
        """Payoff at cumulative probability p (linear interpolation)."""
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}")
        cumulative = 0.0
        for payoff, prob in self.distribution:
            cumulative += prob
            if cumulative >= p - 1e-12:
                return payoff
        return self.distribution[-1][0]

    def var(self, alpha: float) -> float:
        """Value at Risk: payoff x such that P(payoff ≤ x) = alpha."""
        return self.percentile(alpha)

    def cvar(self, alpha: float) -> float:
        """Conditional VaR (expected shortfall) at level alpha."""
        threshold = self.var(alpha)
        tail_payoffs = [(v, p) for v, p in self.distribution if v <= threshold]
        tail_prob = sum(p for _, p in tail_payoffs)
        if tail_prob < 1e-12:
            return threshold
        return sum(v * p for v, p in tail_payoffs) / tail_prob

    def display(self, view: Literal["ev", "ce", "eu"] = "ev",
                max_depth: int | None = None,
                expand_shared: bool = True,
                policy_only: bool = False,
                compact: bool = True,
                show_distribution: bool = True,
                reveal: list[str] | None = None) -> None:
        """Print the annotated tree and outcome distribution summary.

        policy_only=True removes sub-optimal branches so only the optimal
        decision path at each decision node is shown.
        expand_shared=False collapses shared subtrees into a footer.
        compact=True drops the "→ target" hint on every branch and the
        separate leaf-node row, for a denser tree meant for slides.
        show_distribution=False omits the outcome-distribution table, so the
        tree and the distribution can be shown/exported separately.
        reveal=[...] shows the full tree, but only annotates the named
        chance/decision nodes (and their optimal-branch arrows) as solved —
        everything else still renders as if unsolved (raw probabilities and
        payoffs, no EV). For walking a class through folding a tree back one
        piece at a time: pass progressively larger name lists across a
        sequence of calls. reveal=None (default) shows the whole solve.
        """
        from ._display import display_result
        display_result(self, view=view, max_depth=max_depth,
                       expand_shared=expand_shared, policy_only=policy_only,
                       compact=compact, show_distribution=show_distribution,
                       reveal=reveal)

    def save_svg(
        self,
        path: str,
        view: str = "ev",
        max_depth: int | None = None,
        expand_shared: bool = True,
        width: int = 120,
        title: str = "Decision Tree",
        compact: bool = True,
        show_distribution: bool = True,
        reveal: list[str] | None = None,
        policy_only: bool = False,
    ) -> None:
        """Save the annotated rollback display as an SVG file.

        compact=True drops the "→ target" hint on every branch and the
        separate leaf-node row, for a denser tree meant for slides.
        show_distribution=False omits the outcome-distribution table, so the
        tree and the distribution can be exported separately.
        reveal=[...] shows the full tree, but only annotates the named
        chance/decision nodes (and their optimal-branch arrows) as solved —
        see RollbackResult.display() for the full explanation.
        policy_only=True removes sub-optimal decision branches, leaving only
        the optimal path at each decision node — a compact "policy tree"
        showing just the recommended actions, with chance nodes left fully
        expanded (they represent outcome uncertainty, not a choice to prune).
        """
        from ._display import save_svg_result
        save_svg_result(self, path, view=view, max_depth=max_depth,
                        expand_shared=expand_shared, width=width, title=title,
                        compact=compact, show_distribution=show_distribution,
                        reveal=reveal, policy_only=policy_only)

    def plot_distribution(self, view: Literal["both", "cdf", "histogram"] = "both"):
        """
        Plot the outcome distribution of the optimal strategy.

        view : "both" (default) — CDF and PMF side by side
               "cdf"            — cumulative distribution only
               "histogram"      — probability mass only
        Returns the matplotlib Figure.
        """
        from ._sensitivity import plot_distribution
        return plot_distribution(self, view=view)

    def risk_profile_by_branch(self, node: str | None = None) -> "RiskProfileCollection":
        """
        One RiskProfile per branch of a DecisionNode.

        node : None → root node (must be a DecisionNode)
               str  → name of any DecisionNode in the tree
        """
        from ._sensitivity import run_risk_profile_by_branch
        return run_risk_profile_by_branch(self, node_name=node)

    def plot(self, collapse: list[str] | None = None) -> None:
        raise NotImplementedError("plot() not yet implemented")

    def plot_mermaid(self, collapse: list[str] | None = None) -> str:
        raise NotImplementedError("plot_mermaid() not yet implemented")


@dataclass(frozen=True)
class RiskProfile:
    """Outcome distribution for one branch (used in RiskProfileCollection)."""
    label: str
    distribution: list[tuple[float, float]]
    ev: float = 0.0


class RiskProfileCollection:
    """Result of result.risk_profile_by_branch(). Keyed by branch label."""

    def __init__(self, profiles: dict) -> None:
        self.profiles = profiles

    def __getitem__(self, branch: str) -> RiskProfile:
        return self.profiles[branch]

    def __repr__(self) -> str:
        labels = list(self.profiles)
        return f"RiskProfileCollection({labels})"

    def plot(self, view: str = "cdf"):
        """
        Overlay risk profiles for all branches.

        view : "cdf" (default) | "histogram" | "both"
        Returns the matplotlib Figure.
        """
        from ._sensitivity import plot_risk_profile
        return plot_risk_profile(self, view=view)


class SensitivityResult:
    """
    Result of tree.sensitivity().

    Attributes
    ----------
    sweeps : dict[str, list[tuple[float, float, str]]]
        {node_name: [(param_value, ev_at_root, optimal_branch), ...]}
        One entry per included variable; each list is the 1-way sweep.
    grid : dict[tuple[float, float], tuple[float, str]] | None
        2-way grid data: {(x1, x2): (ev, optimal_branch)}.
        Only populated when method="2-way".
    grid_names : tuple[str, str] | None
        Names of the two variables in the grid axes (x, y).
    base_ev : float
        EV at the base case (all params at their base values).
    base_branch : str
        Optimal branch label at the root node in the base case.
    base_values : dict[str, float]
        Base parameter value for each variable (centre of spider chart).
    method : str
        "1-way" | "2-way" | "n-way"
    variable_types : dict[str, str]
        "leaf_value" or "prob" for each variable.
    """

    def __init__(
        self,
        sweeps: dict,
        grid: dict | None,
        grid_names: tuple | None,
        base_ev: float,
        base_branch: str,
        base_values: dict,
        method: str,
        variable_types: dict,
    ) -> None:
        self.sweeps         = sweeps
        self.grid           = grid
        self.grid_names     = grid_names
        self.base_ev        = base_ev
        self.base_branch    = base_branch
        self.base_values    = base_values
        self.method         = method
        self.variable_types = variable_types

    def plot(
        self,
        type: str,
        node: str | None = None,
        nodes: list[str] | None = None,
        view: str = "ev",
    ):
        """
        Render a sensitivity chart and return the matplotlib Figure.

        type="one_way"         node= required.  EV vs parameter value.
        type="strategy_region" node= for 1-way bands; nodes=[n1,n2] for 2-way heatmap.
        type="tornado"         All variables ranked by EV swing.
        type="spider"          All variables normalised to % change from base.
        """
        from ._sensitivity import (
            plot_one_way, plot_strategy_region, plot_tornado, plot_spider,
        )
        if type == "one_way":
            if node is None:
                raise ValidationError("node= is required for type='one_way'.",
                                      rule="missing_node")
            return plot_one_way(self, node, view)
        if type == "strategy_region":
            return plot_strategy_region(self, node=node, nodes=nodes)
        if type == "tornado":
            return plot_tornado(self, view)
        if type == "spider":
            return plot_spider(self, view)
        raise ValidationError(
            f"Unknown plot type '{type}'. "
            "Use 'one_way', 'strategy_region', 'tornado', or 'spider'.",
            rule="unknown_plot_type",
        )

    def strategy_region(self) -> dict:
        """
        Return crossover points for each variable.
        {node_name: {"crossovers": [x, ...]}} where x is the parameter value
        at which the optimal decision first changes.
        """
        out: dict = {}
        for name, pts in self.sweeps.items():
            crossovers: list[float] = []
            prev_b: str | None = None
            for x, ev, b in pts:
                if prev_b is not None and b != prev_b:
                    crossovers.append(x)
                prev_b = b
            out[name] = {"crossovers": crossovers}
        return out


class RiskAttitudeSensitivityResult:
    """
    Result of tree.risk_attitude_sensitivity().

    Attributes
    ----------
    mode : "sweep" | "uncertain" | "factory"
    ce_by_branch : dict[str, list[float]]
        {branch_label: [CE at each evaluation]}
    optimal_branch_by_sample : list[str]
        Optimal branch label at each evaluation.
    crossovers : list[float] | None
        Risk-tolerance values where the optimal branch changes (sweep mode only).
    reversal_probability : float | None
        P(optimal ≠ base-case optimal) across samples (uncertain/factory only).
    risk_tolerances : list[float] | None
        The evaluated risk-tolerance values (sweep and uncertain modes).
    """

    def __init__(
        self,
        mode: str,
        ce_by_branch: dict,
        optimal_branch_by_sample: list,
        crossovers,
        reversal_probability,
        risk_tolerances,
    ) -> None:
        self.mode = mode
        self.ce_by_branch = ce_by_branch
        self.optimal_branch_by_sample = optimal_branch_by_sample
        self.crossovers = crossovers
        self.reversal_probability = reversal_probability
        self.risk_tolerances = risk_tolerances

    def plot(self):
        """
        Sweep mode    : CE per branch vs risk tolerance; crossovers marked.
        Uncertain mode: mean CE ± 10th–90th percentile bands; P(reversal) annotated.
        Factory mode  : overlaid CE histograms per branch.
        Returns the matplotlib Figure.
        """
        from ._sensitivity import plot_risk_attitude
        return plot_risk_attitude(self)
