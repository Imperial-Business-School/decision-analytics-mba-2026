"""
dtree._sensitivity — sensitivity analysis computation and plotting.
Private module; do not import directly from user code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core import ChanceNode, DecisionNode, LeafNode, Value, ValidationError
from ._display import _fv

if TYPE_CHECKING:
    from .core import DecisionTree, SensitivityResult


# ---------------------------------------------------------------------------
# Figure creation — always force a light, opaque background
# ---------------------------------------------------------------------------

def _subplots(*args, **kwargs):
    """Drop-in replacement for plt.subplots() that forces a white, opaque
    figure/axes background regardless of any surrounding dark-mode or
    transparent-savefig configuration (some notebook front-ends auto-set
    transparent figure backgrounds for dark themes, which combined with
    matplotlib's default black text/axes makes these plots unreadable)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor("white")
    axes = ax.flat if hasattr(ax, "flat") else (ax if isinstance(ax, (list, tuple)) else [ax])
    for a in axes:
        a.set_facecolor("white")
    return fig, ax


# ---------------------------------------------------------------------------
# Node lookup
# ---------------------------------------------------------------------------

def _find_node(root, name: str):
    """Walk the tree (DAG-safe) and return the first node whose .name matches."""
    return _find_walk(root, name, set())


def _find_walk(node, name: str, visited: set):
    if node is None or id(node) in visited:
        return None
    visited.add(id(node))
    if node.name == name:
        return node
    for b in getattr(node, "branches", []):
        found = _find_walk(b.child, name, visited)
        if found is not None:
            return found
    return None


# ---------------------------------------------------------------------------
# Variable info — what gets swept and what range
# ---------------------------------------------------------------------------

def _get_var_info(node) -> tuple[float, list[float], str]:
    """
    Return (base_param, sweep_points, var_type).

    var_type:
      "leaf_value"  — LeafNode.value.base
      "prob"        — ChanceNode.probs.base[0] (others redistributed)
    """
    if isinstance(node, LeafNode):
        if not isinstance(node.value, Value) or node.value.uncertainty is None:
            raise ValidationError(
                f"LeafNode '{node.name}' needs Value(uncertainty=Range(...)) for sensitivity.",
                node=node.name, rule="no_uncertainty",
            )
        return float(node.value.base), list(node.value.uncertainty._points), "leaf_value"

    if isinstance(node, ChanceNode):
        if node.probs is None or node.probs.base is None:
            raise ValidationError(
                f"ChanceNode '{node.name}' has no probabilities.",
                node=node.name, rule="no_probs",
            )
        if node.probs.uncertainty is None:
            raise ValidationError(
                f"ChanceNode '{node.name}' needs Prob(uncertainty=Range(...)) for sensitivity.",
                node=node.name, rule="no_uncertainty",
            )
        return float(node.probs.base[0]), list(node.probs.uncertainty._points), "prob"

    raise ValidationError(
        f"Node '{node.name}' ({type(node).__name__}) is not supported for sensitivity. "
        "Use LeafNode (leaf value) or ChanceNode (first probability).",
        node=node.name, rule="unsupported_node_type",
    )


# ---------------------------------------------------------------------------
# Temporary parameter mutation (in-place, try/finally safe)
# ---------------------------------------------------------------------------

def _get_param(node, vtype: str):
    if vtype == "leaf_value":
        return node.value.base if isinstance(node.value, Value) else float(node.value)
    return list(node.probs.base)   # prob: full copy of base list


def _set_param(node, v: float, vtype: str) -> None:
    if vtype == "leaf_value":
        if isinstance(node.value, Value):
            node.value.base = v
        else:
            node.value = v
        return
    # prob: sweep first branch; redistribute remainder proportionally
    probs = node.probs.base
    n = len(probs)
    probs[0] = v
    rest_sum = sum(probs[1:])
    remaining = 1.0 - v
    if n == 2:
        probs[1] = remaining
    elif rest_sum > 1e-12:
        for i in range(1, n):
            probs[i] = probs[i] / rest_sum * remaining
    else:
        for i in range(1, n):
            probs[i] = remaining / (n - 1)


def _restore_param(node, original, vtype: str) -> None:
    if vtype == "leaf_value":
        if isinstance(node.value, Value):
            node.value.base = original
        else:
            node.value = original
    else:
        for i, v in enumerate(original):
            node.probs.base[i] = v


# ---------------------------------------------------------------------------
# Lightweight rollback — no policy tree, no CE computation
# ---------------------------------------------------------------------------

def _quick_rollback(tree, utility_fn) -> tuple[float, str]:
    from ._engine import _rollback_node
    node_vals: dict = {}
    opt_labels: dict = {}
    eu, _, _ = _rollback_node(
        tree.root, 0.0, {}, {}, tree.settings, utility_fn,
        node_vals, opt_labels,
    )
    return eu, opt_labels.get(tree.root.event_id, "")


# ---------------------------------------------------------------------------
# Auto-detection of uncertain nodes
# ---------------------------------------------------------------------------

def _collect_uncertain_names(root) -> list[str]:
    names: list[str] = []
    _collect_walk(root, names, set())
    return names


def _collect_walk(node, names: list, visited: set) -> None:
    if node is None or id(node) in visited:
        return
    visited.add(id(node))
    if isinstance(node, LeafNode):
        if isinstance(node.value, Value) and node.value.uncertainty is not None:
            names.append(node.name)
    elif isinstance(node, ChanceNode):
        if node.probs is not None and node.probs.uncertainty is not None:
            names.append(node.name)
    for b in getattr(node, "branches", []):
        _collect_walk(b.child, names, visited)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def run_sensitivity(tree, result, include, method, n, scenarios):
    from .core import SensitivityResult

    utility_fn = getattr(result, "_utility_fn", None) if result is not None else None

    # Resolve variable list
    if include is None:
        var_names = _collect_uncertain_names(tree.root)
        if not var_names:
            raise ValidationError(
                "No uncertain nodes found in tree. "
                "Set Value(uncertainty=Range(...)) or Prob(uncertainty=Range(...)).",
                rule="no_uncertain_nodes",
            )
    else:
        var_names = list(include)

    # Validate & look up nodes
    node_map: dict = {}
    for name in var_names:
        node = _find_node(tree.root, name)
        if node is None:
            raise ValidationError(
                f"Node '{name}' not found in tree.", rule="node_not_found"
            )
        node_map[name] = node

    # Infer method
    if method is None:
        method = {1: "1-way", 2: "2-way"}.get(len(var_names), "n-way")

    # Gather sweep info
    var_types: dict[str, str] = {}
    base_values: dict[str, float] = {}
    sweep_pts: dict[str, list[float]] = {}
    for name, node in node_map.items():
        base, pts, vtype = _get_var_info(node)
        var_types[name] = vtype
        base_values[name] = base
        sweep_pts[name] = pts

    # Base-case EV
    base_ev, base_branch = _quick_rollback(tree, utility_fn)

    # ── 1-way sweeps for every variable (always needed for tornado/spider) ──
    sweeps: dict[str, list[tuple[float, float, str]]] = {}
    for name, node in node_map.items():
        vtype = var_types[name]
        original = _get_param(node, vtype)
        pts_data: list[tuple[float, float, str]] = []
        try:
            for x in sweep_pts[name]:
                _set_param(node, x, vtype)
                ev, opt = _quick_rollback(tree, utility_fn)
                pts_data.append((x, ev, opt))
        finally:
            _restore_param(node, original, vtype)
        sweeps[name] = pts_data

    # ── 2-way grid (only when method="2-way" and ≥2 variables) ──
    grid: dict[tuple[float, float], tuple[float, str]] | None = None
    grid_names: tuple[str, str] | None = None
    if method == "2-way" and len(var_names) >= 2:
        n1, n2 = var_names[0], var_names[1]
        nd1, nd2 = node_map[n1], node_map[n2]
        vt1, vt2 = var_types[n1], var_types[n2]
        orig1, orig2 = _get_param(nd1, vt1), _get_param(nd2, vt2)
        grid = {}
        try:
            for x1 in sweep_pts[n1]:
                _set_param(nd1, x1, vt1)
                for x2 in sweep_pts[n2]:
                    _set_param(nd2, x2, vt2)
                    ev, opt = _quick_rollback(tree, utility_fn)
                    grid[(x1, x2)] = (ev, opt)
        finally:
            _restore_param(nd1, orig1, vt1)
            _restore_param(nd2, orig2, vt2)
        grid_names = (n1, n2)

    return SensitivityResult(
        sweeps=sweeps,
        grid=grid,
        grid_names=grid_names,
        base_ev=base_ev,
        base_branch=base_branch,
        base_values=base_values,
        method=method,
        variable_types=var_types,
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _branch_colors(branch_labels: list[str]) -> dict[str, tuple]:
    import matplotlib.pyplot as plt
    unique = list(dict.fromkeys(branch_labels))
    cmap = plt.get_cmap("tab10")
    return {b: cmap(i / max(len(unique), 1)) for i, b in enumerate(unique)}


def _crossover_xs(pts: list[tuple[float, float, str]]) -> list[float]:
    """Return x values where optimal branch changes."""
    xs = []
    prev_b = None
    for x, ev, b in pts:
        if prev_b is not None and b != prev_b:
            xs.append(x)
        prev_b = b
    return xs


# ---------------------------------------------------------------------------
# plot_one_way
# ---------------------------------------------------------------------------

def plot_one_way(sr, node_name: str, view: str = "ev"):
    import matplotlib.pyplot as plt

    pts = sr.sweeps.get(node_name)
    if pts is None:
        raise ValueError(f"No sweep data for '{node_name}'.")

    xs  = [x  for x, ev, _ in pts]
    ys  = [ev for _, ev, _ in pts]
    brs = [b  for _, _,  b in pts]

    fig, ax = _subplots(figsize=(7, 4))
    ax.plot(xs, ys, color="steelblue", lw=2, marker="o", ms=4, label=view.upper())
    ax.axhline(sr.base_ev, color="gray", lw=1, ls="--",
               label=f"Base {view.upper()} = {_fv(sr.base_ev)}")
    ax.axvline(sr.base_values[node_name], color="gray", lw=1, ls=":")

    for cx in _crossover_xs(pts):
        ax.axvline(cx, color="crimson", lw=1.2, ls="--", alpha=0.8)
        ax.text(cx, ax.get_ylim()[0], f" {_fv(cx)}", color="crimson", fontsize=8,
                va="bottom", rotation=90)

    ax.set_xlabel(node_name)
    ax.set_ylabel(view.upper())
    ax.set_title(f"One-way sensitivity: {node_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_strategy_region  (1-way and 2-way dispatch)
# ---------------------------------------------------------------------------

def plot_strategy_region(sr, node: str | None = None, nodes: list[str] | None = None):
    if nodes is not None and len(nodes) == 2:
        return _plot_sr_2way(sr, nodes)
    if node is not None:
        return _plot_sr_1way(sr, node)
    # Default: 2-way if grid exists, else 1-way on first var
    if sr.grid is not None and sr.grid_names is not None:
        return _plot_sr_2way(sr, list(sr.grid_names))
    first = next(iter(sr.sweeps))
    return _plot_sr_1way(sr, first)


def _plot_sr_1way(sr, node_name: str):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    pts = sr.sweeps.get(node_name)
    if pts is None:
        raise ValueError(f"No sweep data for '{node_name}'.")

    xs  = [x  for x, _,  _ in pts]
    evs = [ev for _, ev, _ in pts]
    brs = [b  for _, _,  b in pts]
    colors = _branch_colors(brs)

    fig, ax = _subplots(figsize=(8, 4))

    # Shaded bands per region (which policy is optimal)
    segs: list[tuple[float, float, str]] = []
    seg_x0, seg_b = xs[0], brs[0]
    for x, b in zip(xs[1:], brs[1:]):
        if b != seg_b:
            segs.append((seg_x0, x, seg_b))
            seg_x0, seg_b = x, b
    segs.append((seg_x0, xs[-1], seg_b))

    for x0, x1, b in segs:
        ax.axvspan(x0, x1, alpha=0.25, color=colors[b], lw=0)

    # EV curve, on top of the policy shading
    ax.plot(xs, evs, color="black", lw=2, marker="o", ms=3, label="EV")

    # Crossover dashed lines + labels
    for cx in _crossover_xs(pts):
        ax.axvline(cx, color="black", lw=1.2, ls="--")
        ax.text(cx, 0.98, f"{_fv(cx)}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8)

    ax.axvline(sr.base_values[node_name], color="gray", lw=1, ls=":")

    patches = [mpatches.Patch(color=colors[b], alpha=0.5, label=b)
               for b in dict.fromkeys(brs)]
    ev_handle, ev_label = ax.get_legend_handles_labels()
    ax.legend(handles=list(ev_handle) + patches,
              labels=list(ev_label) + [p.get_label() for p in patches],
              loc="best", fontsize=8)
    ax.set_xlabel(node_name)
    ax.set_ylabel("EV")
    ax.set_title(f"Strategy region: {node_name}")
    fig.tight_layout()
    return fig


def _plot_sr_2way(sr, node_names: list[str]):
    import matplotlib.pyplot as plt
    import numpy as np

    if sr.grid is None:
        raise ValueError("No 2-way grid. Run sensitivity(method='2-way') first.")

    n1, n2 = node_names[0], node_names[1]
    xs = sorted(set(k[0] for k in sr.grid))
    ys = sorted(set(k[1] for k in sr.grid))

    all_branches = [b for _, b in sr.grid.values()]
    unique = list(dict.fromkeys(all_branches))
    bidx = {b: i for i, b in enumerate(unique)}

    Z_ev = np.array([[sr.grid[(x, y)][0] for x in xs] for y in ys], dtype=float)
    Z_branch = np.array([[bidx[sr.grid[(x, y)][1]] for x in xs] for y in ys], dtype=float)

    fig, ax = _subplots(figsize=(7, 5))

    # EV as a continuous heat map (the "ev" half of "policy and ev").
    mesh = ax.pcolormesh(xs, ys, Z_ev, cmap="viridis", shading="nearest")
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("EV")

    # Policy boundary/boundaries as contour lines on top (the "policy" half),
    # each region labelled with its optimal branch at that region's centroid.
    if len(unique) > 1:
        levels = [i + 0.5 for i in range(len(unique) - 1)]
        ax.contour(xs, ys, Z_branch, levels=levels, colors="white", linewidths=2)
    X, Y = np.meshgrid(xs, ys)
    for b, i in bidx.items():
        mask = Z_branch == i
        if not mask.any():
            continue
        cx, cy = X[mask].mean(), Y[mask].mean()
        ax.text(cx, cy, b, ha="center", va="center", fontsize=10, fontweight="bold",
                color="white", bbox=dict(boxstyle="round", fc="black", alpha=0.45, lw=0))

    # Base-case crosshair
    ax.axvline(sr.base_values[n1], color="white", lw=1.5, ls="--", alpha=0.8)
    ax.axhline(sr.base_values[n2], color="white", lw=1.5, ls="--", alpha=0.8)
    ax.scatter([sr.base_values[n1]], [sr.base_values[n2]],
               color="white", edgecolor="black", s=60, zorder=5, label="base case")
    ax.legend(fontsize=8)

    ax.set_xlabel(n1)
    ax.set_ylabel(n2)
    ax.set_title(f"2-way strategy region: {n1}  ×  {n2}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_tornado
# ---------------------------------------------------------------------------

def plot_tornado(sr, view: str = "ev"):
    import matplotlib.pyplot as plt

    entries = []
    for name, pts in sr.sweeps.items():
        evs = [ev for _, ev, _ in pts]
        lo, hi = min(evs), max(evs)
        entries.append((name, lo, hi))
    entries.sort(key=lambda e: e[2] - e[1], reverse=True)

    names = [e[0] for e in entries]
    fig, ax = _subplots(figsize=(8, max(3, len(names) * 0.55 + 1.2)))

    # Padding scaled to the chart's overall span, not a fixed absolute
    # offset: a constant like "+1" is negligible against freemark-sized
    # (tens-of-thousands) values but can dwarf a single bar's own width at
    # smaller scales (e.g. this case's low single-digit $bn figures),
    # pushing the value label past the axis and into the category label.
    overall_span = max(e[2] for e in entries) - min(e[1] for e in entries)
    pad = overall_span * 0.02 if overall_span > 0 else 1

    for i, (name, lo, hi) in enumerate(entries):
        ax.barh(i, hi - lo, left=lo,
                color="steelblue", alpha=0.75, height=0.55, edgecolor="white")
        ax.text(hi + pad, i, f"{_fv(hi)}", va="center", fontsize=8)
        ax.text(lo - pad, i, f"{_fv(lo)}", va="center", ha="right", fontsize=8)

    ax.axvline(sr.base_ev, color="black", lw=1.5, label=f"Base = {_fv(sr.base_ev)}")
    # Explicit left-side clearance: the "lo" value labels sit right at the
    # axes' left edge, exactly where matplotlib draws the y-tick category
    # labels outside the axes — the two can visually collide there even
    # though the padding above already separates the label from its own
    # bar, since that padding doesn't account for the y-tick labels' own
    # extent. A fixed extra margin on the left only (not applied on the
    # right, where there's no competing text) keeps them apart regardless
    # of how long a category label (default or caller-overridden) is.
    x_min, x_max = min(e[1] for e in entries) - pad, max(e[2] for e in entries) + pad
    ax.set_xlim(x_min - (x_max - x_min) * 0.12, x_max)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel(view.upper())
    ax.set_title("Tornado chart")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_spider
# ---------------------------------------------------------------------------

def plot_spider(sr, view: str = "ev"):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    fig, ax = _subplots(figsize=(8, 5))

    for i, (name, pts) in enumerate(sr.sweeps.items()):
        base_x = sr.base_values[name]
        if abs(base_x) < 1e-12:
            continue    # can't express % deviation from zero base
        pct_xs = [(x - base_x) / abs(base_x) * 100 for x, _, _ in pts]
        ys     = [ev for _, ev, _ in pts]
        color  = cmap(i / max(len(sr.sweeps), 1))
        ax.plot(pct_xs, ys, color=color, lw=2, marker="o", ms=3, label=name)

    ax.axhline(sr.base_ev, color="gray", lw=1, ls="--", alpha=0.6)
    ax.axvline(0, color="gray", lw=1, ls="--", alpha=0.6)
    ax.scatter([0], [sr.base_ev], color="black", zorder=5, s=60)

    ax.set_xlabel("% change from base value")
    ax.set_ylabel(view.upper())
    ax.set_title("Spider chart")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ===========================================================================
# RISK PROFILE BY BRANCH
# ===========================================================================

def run_risk_profile_by_branch(result, node_name: str | None):
    """
    Force each branch of a DecisionNode in turn, run rollback, collect distributions.
    Returns a RiskProfileCollection.
    """
    from .core import DecisionNode, RiskProfile, RiskProfileCollection, ValidationError
    from ._engine import _rollback_node, _compute_ce

    tree = result._tree
    utility_fn = result._utility_fn

    # Locate target node
    target = _find_node(tree.root, node_name) if node_name else tree.root
    if target is None:
        raise ValidationError(f"Node '{node_name}' not found.", rule="node_not_found")
    if not isinstance(target, DecisionNode):
        raise ValidationError(
            f"'{getattr(target, 'name', node_name)}' is not a DecisionNode.",
            rule="not_decision_node",
        )

    branches = target.branches
    original_actives = [b.active for b in branches]
    profiles: dict = {}

    for branch in branches:
        # Force this branch
        for b in branches:
            b.active = (b is branch)
        try:
            eu, dist, _ = _rollback_node(
                tree.root, 0.0, {}, {}, tree.settings, utility_fn, {}, {}
            )
            distribution = sorted(dist.items())
            ce = _compute_ce(eu, utility_fn, None, distribution) if utility_fn else None
            ev_reported = ce if ce is not None else eu
        finally:
            for b, active in zip(branches, original_actives):
                b.active = active

        profiles[branch.label] = RiskProfile(
            label=branch.label,
            distribution=distribution,
            ev=ev_reported,
        )

    return RiskProfileCollection(profiles=profiles)


# ---------------------------------------------------------------------------
# plot_risk_profile  (used by RiskProfileCollection.plot)
# ---------------------------------------------------------------------------

def plot_risk_profile(collection, view: str = "cdf"):
    import matplotlib.pyplot as plt

    fig, axes = _subplots(1, 2, figsize=(11, 4))
    cmap = plt.get_cmap("tab10")

    # Shared bar width across all branches, so a branch with a single
    # (certain) payoff still gets a visibly-wide bar instead of falling
    # back to a width of 1 dollar against a tens-of-thousands axis scale.
    all_payoffs = [p for rp in collection.profiles.values() for p, _ in rp.distribution]
    x_min, x_max = min(all_payoffs), max(all_payoffs)
    all_span = (x_max - x_min) if len(all_payoffs) > 1 else 1
    w = all_span * 0.04 if all_span > 0 else 1

    for i, (label, rp) in enumerate(collection.profiles.items()):
        color = cmap(i / max(len(collection.profiles), 1))
        payoffs = [p for p, _ in rp.distribution]
        probs   = [p for _, p in rp.distribution]

        # CDF: inclusive cumulative at each payoff, extended out to the
        # shared x-range on both ends so the curve reads as an actual CDF,
        # flat at 0 before the lowest payoff and flat at 1 after the
        # highest, not just a bare jump with nothing plotted around it
        # (the extension also fixes a single-payoff/certain branch, which
        # would otherwise be one invisible point with no line at all).
        ax1 = axes[0]
        cumulative = []
        running = 0.0
        for prob in probs:
            running += prob
            cumulative.append(running)
        step_xs = [x_min] + payoffs + [x_max]
        step_ys = [0.0] + cumulative + [cumulative[-1]]
        ax1.step(step_xs, step_ys, where="post", color=color, lw=2,
                 label=f"{label}  (EV={_fv(rp.ev)})")

        # PMF bars (slightly offset to show overlaps)
        ax2 = axes[1]
        offset = (i - len(collection.profiles) / 2) * w * 0.3
        ax2.bar([p + offset for p in payoffs], probs,
                width=w * 0.5, color=color, alpha=0.7, label=label)

    for ax, title, ylabel in [
        (axes[0], "Cumulative distribution", "Cumulative probability"),
        (axes[1], "Probability distribution", "Probability"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Payoff")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    fig.suptitle("Risk profiles by branch")
    fig.tight_layout()
    return fig


# ===========================================================================
# DISTRIBUTION PLOT  (for RollbackResult.plot_distribution)
# ===========================================================================

def plot_distribution(result, view: str = "both"):
    import matplotlib.pyplot as plt

    dist     = result.distribution
    payoffs  = [p for p, _ in dist]
    probs    = [p for _, p in dist]
    ev       = result.ev

    if view == "both":
        fig, (ax_cdf, ax_pmf) = _subplots(1, 2, figsize=(11, 4))
    elif view == "cdf":
        fig, ax_cdf = _subplots(figsize=(6, 4))
        ax_pmf = None
    else:
        fig, ax_pmf = _subplots(figsize=(6, 4))
        ax_cdf = None

    # ── CDF ──────────────────────────────────────────────────────────────────
    if ax_cdf is not None:
        cumul = 0.0
        xs, ys = [payoffs[0]], [0.0]
        for pay, prob in zip(payoffs, probs):
            xs.append(pay)
            ys.append(cumul)
            cumul += prob
            xs.append(pay)
            ys.append(cumul)
        xs.append(payoffs[-1])
        ys.append(cumul)

        ax_cdf.plot(xs, ys, color="steelblue", lw=2)
        ax_cdf.axvline(ev, color="crimson", lw=1.5, ls="--", label=f"EV = {_fv(ev)}")
        ax_cdf.axhline(0.5, color="gray", lw=1, ls=":", alpha=0.7)
        ax_cdf.fill_between(xs, ys, alpha=0.08, color="steelblue")
        ax_cdf.set_xlabel("Payoff")
        ax_cdf.set_ylabel("Cumulative probability")
        ax_cdf.set_title("CDF")
        ax_cdf.legend(fontsize=8)

    # ── PMF ──────────────────────────────────────────────────────────────────
    if ax_pmf is not None:
        span = max(payoffs) - min(payoffs) if len(payoffs) > 1 else 1
        w = span * 0.06
        ax_pmf.bar(payoffs, probs, width=w, color="steelblue", alpha=0.75,
                   edgecolor="white")
        ax_pmf.axvline(ev, color="crimson", lw=1.5, ls="--", label=f"EV = {_fv(ev)}")
        ax_pmf.set_xlabel("Payoff")
        ax_pmf.set_ylabel("Probability")
        ax_pmf.set_title("Distribution")
        ax_pmf.legend(fontsize=8)

    fig.suptitle(f"Outcome distribution, EV = {_fv(ev)}")
    fig.tight_layout()
    return fig


# ===========================================================================
# RISK ATTITUDE SENSITIVITY
# ===========================================================================

def run_risk_attitude_sensitivity(tree, result, risk_tolerance, utility, utility_factory, n):
    from .core import DecisionNode, RiskAttitudeSensitivityResult, ValidationError
    from ._engine import _rollback_node, _make_utility_fn, _compute_ce
    import numpy as np

    root = tree.root
    if not isinstance(root, DecisionNode):
        raise ValidationError(
            "risk_attitude_sensitivity requires a DecisionNode at the tree root.",
            rule="root_not_decision",
        )

    branches = [b for b in root.branches if b.active]
    branch_labels = [b.label for b in branches]
    original_actives = [b.active for b in root.branches]
    utility_type = utility or "exponential"
    if isinstance(utility_type, list):
        utility_type = utility_type[0]

    # ── Determine mode ────────────────────────────────────────────────────────
    if utility_factory is not None:
        mode = "factory"
    elif hasattr(risk_tolerance, "rvs"):
        mode = "uncertain"
    else:
        mode = "sweep"

    # ── Auto-range ────────────────────────────────────────────────────────────
    if mode == "sweep" and risk_tolerance is None:
        _, base_dist, _ = _rollback_node(
            root, 0.0, {}, {}, tree.settings, None, {}, {}
        )
        payoffs = sorted(base_dist.keys())
        if len(payoffs) < 2:
            raise ValidationError(
                "Cannot auto-range: fewer than 2 distinct payoffs.",
                rule="insufficient_outcomes",
            )
        spread = payoffs[-1] - payoffs[0]
        risk_tolerance = (0.1 * spread, 5.0 * spread)

    # ── Generate evaluation points ────────────────────────────────────────────
    if mode == "sweep":
        rt_low, rt_high = risk_tolerance
        rt_values = list(np.linspace(rt_low, rt_high, n))
    elif mode == "uncertain":
        rt_values = list(float(v) for v in risk_tolerance.rvs(size=n))
    else:
        rt_values = None   # factory: no shared parameter axis

    # ── Evaluate CE per branch at each point ──────────────────────────────────
    ce_by_branch: dict[str, list[float]] = {lbl: [] for lbl in branch_labels}
    optimal_by_sample: list[str] = []

    def _eval_one(utility_fn) -> tuple[dict, str]:
        branch_ces: dict[str, float] = {}
        try:
            for branch in branches:
                for b in root.branches:
                    b.active = (b is branch)
                eu, dist, _ = _rollback_node(
                    root, 0.0, {}, {}, tree.settings, utility_fn, {}, {}
                )
                dist_sorted = sorted(dist.items())
                ce = _compute_ce(eu, utility_fn, None, dist_sorted)
                branch_ces[branch.label] = ce if ce is not None else eu
        finally:
            for b, active in zip(root.branches, original_actives):
                b.active = active
        opt = max(branch_ces, key=lambda k: branch_ces[k])
        return branch_ces, opt

    if mode in ("sweep", "uncertain"):
        for rt in rt_values:
            if callable(utility_type) and not isinstance(utility_type, str):
                utility_fn = utility_type
            else:
                utility_fn = _make_utility_fn(utility_type, rt)
            branch_ces, opt = _eval_one(utility_fn)
            for lbl in branch_labels:
                ce_by_branch[lbl].append(branch_ces[lbl])
            optimal_by_sample.append(opt)
    else:  # factory
        for _ in range(n):
            utility_fn = utility_factory()
            branch_ces, opt = _eval_one(utility_fn)
            for lbl in branch_labels:
                ce_by_branch[lbl].append(branch_ces[lbl])
            optimal_by_sample.append(opt)

    # ── Crossovers (sweep only) ───────────────────────────────────────────────
    crossovers: list[float] | None = None
    if mode == "sweep":
        crossovers = []
        prev = None
        for rt, opt in zip(rt_values, optimal_by_sample):
            if prev is not None and opt != prev:
                crossovers.append(float(rt))
            prev = opt

    # ── Reversal probability (uncertain / factory) ────────────────────────────
    reversal_probability: float | None = None
    if mode in ("uncertain", "factory"):
        base_opt = None
        if result is not None:
            for b in result.policy.root.branches:
                if b.active:
                    base_opt = b.label
                    break
        if base_opt is not None:
            n_reversals = sum(1 for o in optimal_by_sample if o != base_opt)
            reversal_probability = n_reversals / len(optimal_by_sample)

    return RiskAttitudeSensitivityResult(
        mode=mode,
        ce_by_branch=ce_by_branch,
        optimal_branch_by_sample=optimal_by_sample,
        crossovers=crossovers,
        reversal_probability=reversal_probability,
        risk_tolerances=rt_values,
    )


# ---------------------------------------------------------------------------
# plot_risk_attitude
# ---------------------------------------------------------------------------

def plot_risk_attitude(ra):
    import matplotlib.pyplot as plt
    import numpy as np

    cmap = plt.get_cmap("tab10")
    branch_labels = list(ra.ce_by_branch.keys())

    # ── Sweep mode ────────────────────────────────────────────────────────────
    if ra.mode == "sweep":
        fig, ax = _subplots(figsize=(8, 5))
        rts = ra.risk_tolerances
        for i, lbl in enumerate(branch_labels):
            ax.plot(rts, ra.ce_by_branch[lbl],
                    color=cmap(i / max(len(branch_labels), 1)),
                    lw=2, label=lbl)
        for cx in (ra.crossovers or []):
            ax.axvline(cx, color="black", lw=1, ls="--", alpha=0.7)
            ax.text(cx, ax.get_ylim()[0], f" {_fv(cx)}",
                    fontsize=8, va="bottom", rotation=90, color="black")
        ax.set_xlabel("Risk tolerance")
        ax.set_ylabel("Certainty equivalent")
        ax.set_title("Risk attitude sensitivity")
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    # ── Uncertain mode ────────────────────────────────────────────────────────
    if ra.mode == "uncertain":
        rts = np.array(ra.risk_tolerances)
        order = np.argsort(rts)
        fig, ax = _subplots(figsize=(8, 5))
        for i, lbl in enumerate(branch_labels):
            ces = np.array(ra.ce_by_branch[lbl])[order]
            rts_s = rts[order]
            # bin into ~20 groups for smooth bands
            bins = min(20, len(rts_s))
            edges = np.array_split(np.arange(len(rts_s)), bins)
            xs = [rts_s[g].mean() for g in edges]
            means  = [ces[g].mean()              for g in edges]
            p10    = [np.percentile(ces[g], 10)  for g in edges]
            p90    = [np.percentile(ces[g], 90)  for g in edges]
            color = cmap(i / max(len(branch_labels), 1))
            ax.plot(xs, means, color=color, lw=2, label=lbl)
            ax.fill_between(xs, p10, p90, color=color, alpha=0.15)
        if ra.reversal_probability is not None:
            ax.text(0.97, 0.03,
                    f"P(reversal) = {ra.reversal_probability:.1%}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        ax.set_xlabel("Risk tolerance (sampled)")
        ax.set_ylabel("Certainty equivalent")
        ax.set_title("Risk attitude sensitivity (uncertain risk tolerance)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    # ── Factory mode ──────────────────────────────────────────────────────────
    fig, ax = _subplots(figsize=(8, 5))
    for i, lbl in enumerate(branch_labels):
        color = cmap(i / max(len(branch_labels), 1))
        ax.hist(ra.ce_by_branch[lbl], bins=30, color=color,
                alpha=0.55, label=lbl, edgecolor="white")
    ax.set_xlabel("Certainty equivalent")
    ax.set_ylabel("Count")
    ax.set_title("Risk attitude sensitivity (utility factory)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
