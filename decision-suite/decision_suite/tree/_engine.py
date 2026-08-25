"""
dtree._engine — rollback computation and tree validation.
Private module; do not import directly from user code.
"""

from __future__ import annotations

import inspect
import math
import warnings
from typing import TYPE_CHECKING

from .core import (
    ChanceNode,
    DecisionNode,
    LeafNode,
    LogicNode,
    ModelIssue,
    RollbackError,
    RollbackResult,
    ValidationError,
    Value,
    settings as global_settings,
)

if TYPE_CHECKING:
    from .core import DecisionTree


# ===========================================================================
# VALIDATION
# ===========================================================================

def validate_tree(tree: DecisionTree) -> list[ModelIssue]:
    """Collect all composition-time issues without raising."""
    issues: list[ModelIssue] = []
    _check_subtree(tree.root, path_names=[], path_ids=set(), issues=issues)
    return issues


def _check_subtree(node, path_names: list[str], path_ids: set, issues: list[ModelIssue]) -> None:
    if node is None:
        return

    # Cycle: node already on the current root-to-leaf path
    if id(node) in path_ids:
        issues.append(ModelIssue(
            level="error", node=node.name, rule="cycle_detected",
            message=f"Cycle detected: node '{node.name}' is its own descendant.",
        ))
        return  # stop here to avoid infinite recursion

    # Name collision on this path (siblings sharing names is fine)
    if node.name in path_names:
        issues.append(ModelIssue(
            level="error", node=node.name, rule="name_collision",
            message=(
                f"Name collision: '{node.name}' appears more than once on a "
                f"root-to-leaf path. Node names must be unique per path."
            ),
        ))

    if isinstance(node, ChanceNode):
        _check_chance_node(node, issues)
    elif isinstance(node, LeafNode):
        _check_leaf_node(node, issues)
    elif isinstance(node, LogicNode):
        _check_logic_node(node, issues)

    new_names = path_names + [node.name]
    new_ids = path_ids | {id(node)}
    for branch in getattr(node, "branches", []):
        _check_subtree(branch.child, new_names, new_ids, issues)


def _check_chance_node(node: ChanceNode, issues: list[ModelIssue]) -> None:
    n = len(node.branches)
    if n == 0:
        return

    if node.probs is None or node.probs.base is None:
        level = "error" if global_settings.strict else "warning"
        issues.append(ModelIssue(
            level=level, node=node.name, rule="missing_probs",
            message=(
                f"ChanceNode '{node.name}' has no probabilities. "
                f"{'Raises at rollback (strict mode).' if level == 'error' else f'Equal weights (1/{n}) will be used at rollback.'}"
            ),
        ))
        return

    if len(node.probs.base) != n:
        issues.append(ModelIssue(
            level="error", node=node.name, rule="probs_length_mismatch",
            message=(
                f"ChanceNode '{node.name}': len(probs.base)={len(node.probs.base)} "
                f"!= len(branches)={n}."
            ),
        ))
        return

    total = sum(node.probs.base)
    if abs(total - 1.0) > 1e-6:
        level = "error" if global_settings.strict else "warning"
        issues.append(ModelIssue(
            level=level, node=node.name, rule="probs_not_normalized",
            message=(
                f"ChanceNode '{node.name}': probs sum to {total:.6f}, not 1.0. "
                f"{'Error in strict mode.' if level == 'error' else 'Will be normalised at rollback.'}"
            ),
        ))


def _check_leaf_node(node: LeafNode, issues: list[ModelIssue]) -> None:
    if not callable(node.value):
        return
    try:
        sig = inspect.signature(node.value)
    except (ValueError, TypeError):
        return
    for pname, param in sig.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            issues.append(ModelIssue(
                level="error", node=node.name, rule="kwargs_forbidden",
                message=f"LeafNode '{node.name}': callable uses **kwargs, which is forbidden.",
            ))
            break


def _check_logic_node(node: LogicNode, issues: list[ModelIssue]) -> None:
    if len(node.conditions) != len(node.branches):
        issues.append(ModelIssue(
            level="error", node=node.name, rule="conditions_branch_mismatch",
            message=(
                f"LogicNode '{node.name}': len(conditions)={len(node.conditions)} "
                f"!= len(branches)={len(node.branches)}."
            ),
        ))


# ===========================================================================
# ROLLBACK
# ===========================================================================

def run_rollback(
    tree: DecisionTree,
    utility=None,
    risk_tolerance: float | None = None,
    inverse=None,
    scenario: str | None = None,
) -> RollbackResult:
    """Entry point called by DecisionTree.rollback()."""
    from .core import DecisionTree as DT

    utility_fn = _make_utility_fn(utility, risk_tolerance)
    node_vals: dict = {}         # event_id → float (avoids name-collision for cloned subtrees)
    opt_labels: dict = {}        # event_id → chosen branch label

    # Build scenario context: tree-level overrides + name for per-node lookup
    sc: dict | None = None
    if scenario is not None:
        overrides: dict = {}
        if tree.scenarios:
            for s in tree.scenarios:
                if s.name == scenario and isinstance(s.value, dict):
                    overrides = dict(s.value)
                    break
        sc = {"name": scenario, "overrides": overrides}

    eu, dist, path = _rollback_node(
        tree.root, 0.0, {}, {}, tree.settings, utility_fn, node_vals, opt_labels, sc,
    )

    distribution = sorted(dist.items())

    ce: float | None = None
    if utility_fn is not None:
        ce = _compute_ce(eu, utility_fn, inverse, distribution)

    policy = _build_policy(tree, opt_labels)

    # ev is reported in value units: CE if available, raw EU otherwise
    ev_reported = ce if ce is not None else eu

    # Public node_values keyed by name (last-write-wins for duplicate names).
    # _node_values_by_id keyed by event_id for accurate per-copy display.
    public_node_values = _node_vals_by_name(tree.root, node_vals)

    result = RollbackResult(
        ev=ev_reported,
        ce=ce,
        optimal_path=path,
        distribution=distribution,
        node_values=public_node_values,
        policy=policy,
        scenario=scenario,
        _utility_fn=utility_fn,
        _tree=tree,
    )
    result._node_values_by_id = node_vals
    return result


def _rollback_node(
    node,
    acc_pv: float,
    edge_vals: dict,
    path_lbls: dict,
    settings,
    utility_fn,
    node_vals: dict,
    opt_labels: dict,
    sc: dict | None = None,
) -> tuple[float, dict, list[str]]:
    """
    Recursively evaluate the tree. Returns (eu, dist, path).

    eu   : expected (utility-of) payoff at this node
    dist : {payoff: probability} over all reachable terminals via optimal strategy
    path : node names on the optimal path from this node inclusive
    sc   : scenario context dict {"name": str, "overrides": {node_name: float}} or None
    """
    if isinstance(node, LeafNode):
        return _rollback_leaf(node, acc_pv, edge_vals, path_lbls, settings, utility_fn, node_vals, sc)

    if isinstance(node, ChanceNode):
        return _rollback_chance(node, acc_pv, edge_vals, path_lbls, settings, utility_fn, node_vals, opt_labels, sc)

    if isinstance(node, DecisionNode):
        return _rollback_decision(node, acc_pv, edge_vals, path_lbls, settings, utility_fn, node_vals, opt_labels, sc)

    if isinstance(node, LogicNode):
        return _rollback_logic(node, acc_pv, edge_vals, path_lbls, settings, utility_fn, node_vals, opt_labels, sc)

    raise RollbackError(f"Unknown node type: {type(node).__name__}", node=getattr(node, "name", "?"))


def _eval_branch_child(branch, acc_pv, edge_vals, path_lbls, settings, utility_fn, node_vals, opt_labels, sc=None):
    """Apply edge PV and recurse into a branch's child (or produce implicit terminal)."""
    edge_pv = _pv_scalar(_resolve_edge_value(branch.value, sc), branch.time, settings.discount_rate)
    new_acc = acc_pv + edge_pv

    if branch.on_enter is not None:
        branch.on_enter()

    if branch.child is None:
        payoff = new_acc
        eu = utility_fn(payoff) if utility_fn else payoff
        return eu, {payoff: 1.0}, []

    return _rollback_node(branch.child, new_acc, edge_vals, path_lbls, settings, utility_fn, node_vals, opt_labels, sc)


def _rollback_leaf(node: LeafNode, acc_pv, edge_vals, path_lbls, settings, utility_fn, node_vals, sc=None):
    if callable(node.value):
        payoff = float(_call_with_injection(node.value, edge_vals, path_lbls))
    else:
        leaf_v = _resolve_leaf_value(node.value, node.name, sc)
        payoff = acc_pv + _pv_scalar(leaf_v, node.time, settings.discount_rate)
    eu = utility_fn(payoff) if utility_fn else payoff
    node_vals[node.event_id] = eu
    return eu, {payoff: 1.0}, [node.name]


def _rollback_chance(node: ChanceNode, acc_pv, edge_vals, path_lbls, settings, utility_fn, node_vals, opt_labels, sc=None):
    active = [b for b in node.branches if b.active]
    if not active:
        raise RollbackError(f"ChanceNode '{node.name}' has no active branches.", node=node.name)

    probs = _get_active_probs(node, active, sc)

    eu = 0.0
    dist: dict[float, float] = {}

    for branch, prob in zip(active, probs):
        raw = _resolve_edge_value(branch.value, sc)
        new_edge_vals = {**edge_vals, node.name: raw}
        new_path_lbls = {**path_lbls, node.name: branch.label or ""}

        child_eu, child_dist, _ = _eval_branch_child(
            branch, acc_pv, new_edge_vals, new_path_lbls, settings, utility_fn, node_vals, opt_labels, sc,
        )
        eu += prob * child_eu
        for payoff, p in child_dist.items():
            dist[payoff] = dist.get(payoff, 0.0) + prob * p

    node_vals[node.event_id] = eu
    return eu, dist, [node.name]


def _rollback_decision(node: DecisionNode, acc_pv, edge_vals, path_lbls, settings, utility_fn, node_vals, opt_labels, sc=None):
    active = [b for b in node.branches if b.active]
    if not active:
        raise RollbackError(f"DecisionNode '{node.name}' has no active branches.", node=node.name)

    best_eu: float | None = None
    best_branch = None
    best_dist: dict | None = None
    best_path: list | None = None

    for branch in active:
        raw = _resolve_edge_value(branch.value, sc)
        new_edge_vals = {**edge_vals, node.name: raw}
        new_path_lbls = {**path_lbls, node.name: branch.label or ""}

        child_eu, child_dist, child_path = _eval_branch_child(
            branch, acc_pv, new_edge_vals, new_path_lbls, settings, utility_fn, node_vals, opt_labels, sc,
        )

        is_better = (
            best_eu is None
            or (settings.maximize and child_eu > best_eu)
            or (not settings.maximize and child_eu < best_eu)
        )
        if is_better:
            best_eu = child_eu
            best_branch = branch
            best_dist = child_dist
            best_path = child_path

    opt_labels[node.event_id] = best_branch.label
    node_vals[node.event_id] = best_eu
    return best_eu, best_dist, [node.name] + best_path


def _rollback_logic(node: LogicNode, acc_pv, edge_vals, path_lbls, settings, utility_fn, node_vals, opt_labels, sc=None):
    true_results: list[tuple[float, dict, list]] = []

    for branch, cond in zip(node.branches, node.conditions):
        if not branch.active:
            continue
        if _call_with_injection(cond, edge_vals, path_lbls):
            raw = _resolve_edge_value(branch.value, sc)
            new_edge_vals = {**edge_vals, node.name: raw}
            new_path_lbls = {**path_lbls, node.name: branch.label or ""}

            child_eu, child_dist, child_path = _eval_branch_child(
                branch, acc_pv, new_edge_vals, new_path_lbls, settings, utility_fn, node_vals, opt_labels, sc,
            )
            true_results.append((child_eu, child_dist, child_path))

    if not true_results:
        raise RollbackError(
            f"LogicNode '{node.name}' has no True condition at rollback time.",
            node=node.name, reason="no_true_condition",
        )

    n = len(true_results)
    eu = sum(r[0] for r in true_results) / n
    dist: dict[float, float] = {}
    w = 1.0 / n
    for _, d, _ in true_results:
        for payoff, p in d.items():
            dist[payoff] = dist.get(payoff, 0.0) + w * p

    node_vals[node.event_id] = eu
    return eu, dist, [node.name] + true_results[0][2]


# ===========================================================================
# POLICY TREE
# ===========================================================================

def _build_policy(tree: DecisionTree, opt_labels: dict) -> DecisionTree:
    """Deep-copy the tree and deactivate non-optimal decision branches."""
    from .core import DecisionTree as DT
    root_copy = tree.root.copy(deep=True)
    _deactivate_suboptimal(root_copy, opt_labels, visited=set())
    return DT._from_root(root_copy, tree.settings)


def _deactivate_suboptimal(node, opt_labels: dict, visited: set) -> None:
    if node is None or id(node) in visited:
        return
    visited.add(id(node))

    if isinstance(node, DecisionNode) and node.event_id in opt_labels:
        opt_label = opt_labels[node.event_id]
        for branch in node.branches:
            if branch.label != opt_label:
                branch.active = False

    for branch in getattr(node, "branches", []):
        _deactivate_suboptimal(branch.child, opt_labels, visited)


# ===========================================================================
# HELPERS
# ===========================================================================

def _node_vals_by_name(root, node_vals_by_id: dict) -> dict:
    """Walk the tree and build a name-keyed {name: ev} dict from event_id-keyed node_vals."""
    result: dict[str, float] = {}
    _walk_for_names(root, node_vals_by_id, result, visited=set())
    return result


def _walk_for_names(node, by_id, by_name, visited):
    if node is None or id(node) in visited:
        return
    visited.add(id(node))
    if node.event_id in by_id:
        by_name[node.name] = by_id[node.event_id]
    for b in getattr(node, "branches", []):
        _walk_for_names(b.child, by_id, by_name, visited)


def _pv(value, time: float, discount_rate: float) -> float:
    v = _raw_value(value)
    if discount_rate == 0.0 or time == 0.0:
        return v
    return v / (1.0 + discount_rate) ** time


def _pv_scalar(v: float, time: float, discount_rate: float) -> float:
    if discount_rate == 0.0 or time == 0.0:
        return v
    return v / (1.0 + discount_rate) ** time


def _raw_value(value) -> float:
    """Extract the base float from a float | Value."""
    return value.base if isinstance(value, Value) else float(value)


def _resolve_edge_value(value, sc: dict | None) -> float:
    """Return scenario-adjusted float for a branch edge value (float | Value)."""
    if sc:
        sname = sc.get("name")
        if sname and isinstance(value, Value) and value.scenarios:
            for sv in value.scenarios:
                if sv.name == sname:
                    return float(sv.value)
    return value.base if isinstance(value, Value) else float(value)


def _resolve_leaf_value(value, node_name: str, sc: dict | None) -> float:
    """Return scenario-adjusted scalar for a non-callable leaf value."""
    if sc:
        sname = sc.get("name")
        overrides = sc.get("overrides", {})
        if node_name in overrides:
            return float(overrides[node_name])
        if sname and isinstance(value, Value) and value.scenarios:
            for sv in value.scenarios:
                if sv.name == sname:
                    return float(sv.value)
    return value.base if isinstance(value, Value) else float(value)


def _resolve_prob_vector(node, sc: dict | None) -> list[float] | None:
    """Return alternate prob vector if a named scenario matches on node.probs, else None."""
    if not sc:
        return None
    sname = sc.get("name")
    if not sname or node.probs is None or not node.probs.scenarios:
        return None
    for sv in node.probs.scenarios:
        if sv.name == sname:
            return list(sv.value)
    return None


def _get_active_probs(node: ChanceNode, active_branches: list, sc: dict | None = None) -> list[float]:
    """Return probabilities aligned to active_branches, redistributed if needed."""
    n_total = len(node.branches)

    scenario_base = _resolve_prob_vector(node, sc)

    if node.probs is None or node.probs.base is None:
        if scenario_base is not None:
            base = scenario_base
        elif global_settings.strict:
            raise RollbackError(
                f"ChanceNode '{node.name}' has no probabilities (strict mode).",
                node=node.name, reason="missing_probs",
            )
        else:
            warnings.warn(
                f"ChanceNode '{node.name}' has no probabilities; using equal weights.",
                stacklevel=4,
            )
            base = [1.0 / n_total] * n_total
    else:
        base = scenario_base if scenario_base is not None else list(node.probs.base)

    active_set = {id(b) for b in active_branches}
    active_probs = [p for b, p in zip(node.branches, base) if id(b) in active_set]

    if node.redistribute:
        total = sum(active_probs)
        if total > 1e-12:
            active_probs = [p / total for p in active_probs]

    return active_probs


def _call_with_injection(fn, edge_vals: dict, path_lbls: dict):
    """
    Call fn injecting named parameters from edge values and path labels.
    Zero-arg callables (closures over Context) are called directly.
    """
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return fn()

    params = sig.parameters
    if not params:
        return fn()

    kwargs: dict = {}
    for pname, param in params.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if pname == "_path":
            kwargs["_path"] = path_lbls
        elif pname in edge_vals:
            kwargs[pname] = edge_vals[pname]
        # else: use default if available; let Python raise naturally if not

    return fn(**kwargs)


def _make_utility_fn(utility, risk_tolerance):
    if utility is None:
        return None
    if utility == "exponential":
        if risk_tolerance is None:
            raise ValidationError(
                "risk_tolerance is required for exponential utility.", rule="missing_risk_tolerance"
            )
        rt = risk_tolerance
        return lambda x: 1.0 - math.exp(-x / rt)
    if utility == "logarithmic":
        if risk_tolerance is None:
            raise ValidationError(
                "risk_tolerance is required for logarithmic utility.", rule="missing_risk_tolerance"
            )
        rt = risk_tolerance
        return lambda x: math.log(x + rt)
    if callable(utility):
        return utility
    raise ValidationError(f"Unknown utility: {utility!r}. Use 'exponential', 'logarithmic', or a callable.", rule="unknown_utility")


def _compute_ce(eu: float, utility_fn, inverse, distribution: list) -> float | None:
    if inverse is not None:
        return inverse(eu)
    try:
        import scipy.optimize as opt
        payoffs = [p for p, _ in distribution]
        lo, hi = min(payoffs) - 1.0, max(payoffs) + 1.0
        # widen until we bracket the root
        for _ in range(64):
            if utility_fn(lo) <= eu <= utility_fn(hi):
                break
            lo -= abs(lo) + 1.0
            hi += abs(hi) + 1.0
        return float(opt.brentq(lambda x: utility_fn(x) - eu, lo, hi))
    except Exception:
        return None
