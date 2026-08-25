"""
dtree._display — text display helpers for nodes, trees, and rollback results.
Private module; do not import directly from user code.
"""

from __future__ import annotations

from . import core as _core
from .core import ChanceNode, DecisionNode, LeafNode, LogicNode, Value

# ---------------------------------------------------------------------------
# Constants — plain text rendering
# ---------------------------------------------------------------------------

_MID      = "├── "
_LAST     = "└── "
_OPT_MID  = "├─> "
_OPT_LAST = "└─> "
_CONT     = "│   "
_SKIP     = "    "


# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------

def _raw(v) -> float:
    return v.base if isinstance(v, Value) else float(v)


def _fv(v: float) -> str:
    """Format a value using dtree.settings.formatter, or the built-in default."""
    fn = _core.settings.formatter
    if fn is not None:
        return fn(v)
    if abs(v - round(v)) < 1e-9:
        return f"{round(v):,}"
    return f"{v:,.2f}"


def _markup(text: str, key: str) -> str:
    """Wrap *text* in the rich color markup for semantic *key* from settings."""
    color = _core.settings.colors.get(key, "")
    if not color:
        return text
    return f"[{color}]{text}[/{color}]"


def _node_prefix(node) -> str:
    if isinstance(node, DecisionNode):
        return "[D]"
    if isinstance(node, ChanceNode):
        return "[C]"
    if isinstance(node, LeafNode):
        return "[T]"
    if isinstance(node, LogicNode):
        return "[L]"
    return "[?]"


def _branch_probs(node) -> list[float] | None:
    if isinstance(node, ChanceNode) and node.probs is not None and node.probs.base is not None:
        return list(node.probs.base)
    return None


# ---------------------------------------------------------------------------
# Optimal-choice extraction from policy tree
# ---------------------------------------------------------------------------

def _collect_opt_choices(root) -> dict[str, str | None]:
    choices: dict[str, str | None] = {}
    _walk(root, choices, visited=set())
    return choices


def _collect_revealed_ids(root, names: set) -> set:
    """event_ids of every node (by traversal) whose .name is in `names`."""
    ids: set = set()
    _walk_reveal(root, names, ids, visited=set())
    return ids


def _walk_reveal(node, names: set, ids: set, visited: set) -> None:
    if node is None or id(node) in visited:
        return
    visited.add(id(node))
    if getattr(node, "name", None) in names:
        ids.add(node.event_id)
    for b in getattr(node, "branches", []):
        _walk_reveal(b.child, names, ids, visited)


def _walk(node, choices: dict, visited: set) -> None:
    if node is None or id(node) in visited:
        return
    visited.add(id(node))
    if isinstance(node, DecisionNode):
        for b in node.branches:
            if b.active:
                choices[node.event_id] = b.label
                break
    for b in getattr(node, "branches", []):
        _walk(b.child, choices, visited)


# ---------------------------------------------------------------------------
# Shared-node detection
# ---------------------------------------------------------------------------

def _collect_shared_ids(root) -> set[int]:
    """Return ids of nodes referenced from more than one branch in the tree."""
    counts: dict[int, int] = {}
    _count_node_refs(root, counts, visited=set())
    return {nid for nid, c in counts.items() if c > 1}


def _count_node_refs(node, counts: dict, visited: set) -> None:
    if node is None or id(node) in visited:
        return
    visited.add(id(node))
    for b in getattr(node, "branches", []):
        if b.child is not None:
            counts[id(b.child)] = counts.get(id(b.child), 0) + 1
    for b in getattr(node, "branches", []):
        _count_node_refs(b.child, counts, visited)


# ---------------------------------------------------------------------------
# Style detection
# ---------------------------------------------------------------------------

def _use_rich() -> bool:
    from .core import settings as _settings
    if _settings.style == "plain":
        return False
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        if _settings.style == "rich":
            import warnings
            warnings.warn(
                "dtree: 'rich' is not installed — falling back to plain text. "
                "Run `pip install rich` to enable colored output.",
                stacklevel=4,
            )
        return False


# ---------------------------------------------------------------------------
# Plain-text recursive display
# ---------------------------------------------------------------------------

def _display_subtree(
    node,
    base: str,
    node_vals: dict | None,
    opt_choices: dict,
    visited: set,
    depth: int,
    max_depth: int | None,
    show_event_id: bool = False,
    show_inactive: bool = True,
    shared_ids: set | None = None,
    encountered: dict | None = None,
    compact: bool = True,
) -> None:
    if node is None:
        return

    if id(node) in visited:
        print(f"{base}{_node_prefix(node)} {node.name}  ↑ (shared — see above)")
        return
    visited.add(id(node))

    _lbl     = "v" if isinstance(node, LeafNode) else "EV"
    ev_part  = f"  {_lbl}={_fv(node_vals[node.event_id])}" if node_vals and node.event_id in node_vals else ""
    eid_part = f"  (event_id: {str(node.event_id)[:4]}…)" if show_event_id else ""
    print(f"{base}{_node_prefix(node)} {node.name}{ev_part}{eid_part}")

    all_branches = getattr(node, "branches", [])
    branches = [b for b in all_branches if b.active or show_inactive]
    if not branches:
        return

    probs_all   = _branch_probs(node)
    probs       = [probs_all[i] for i, b in enumerate(all_branches) if b.active or show_inactive] if probs_all else None
    can_recurse = max_depth is None or depth < max_depth

    for i, b in enumerate(branches):
        is_last = (i == len(branches) - 1)
        is_opt  = (
            node_vals is not None
            and isinstance(node, DecisionNode)
            and opt_choices.get(node.event_id) == b.label
        )

        if is_opt:
            connector = _OPT_LAST if is_last else _OPT_MID
        else:
            connector = _LAST if is_last else _MID
        cont = _SKIP if is_last else _CONT

        prob  = probs[i] if probs and i < len(probs) else None
        label = f'"{b.label}"' if b.label is not None else f"[branch {i}]"

        # Determine if this child is a shared ref shown in the footer
        is_shared_ref = (
            shared_ids is not None
            and b.child is not None
            and id(b.child) in shared_ids
        )

        parts = [label]
        if prob is not None:
            parts.append(f"p={prob:.2f}")
        val = _raw(b.value)
        parts.append(f"v={_fv(val)}")
        if b.time != 0:
            parts.append(f"time={b.time}")

        if is_shared_ref and encountered is not None:
            encountered.setdefault(id(b.child), b.child)

        if not compact:
            if b.child is None:
                parts.append("→ [terminal]")
            elif is_shared_ref:
                parts.append(f"→ [{b.child.name}]")   # bracketed = see footer
            else:
                parts.append(f"→ {b.child.name}")

        if not b.active and show_inactive:
            parts.append("[inactive]")

        print(f"{base}{connector}{'  '.join(parts)}")

        skip_leaf = compact and isinstance(b.child, LeafNode) and not is_shared_ref
        if can_recurse and b.child is not None and not is_shared_ref and not skip_leaf:
            _display_subtree(
                b.child,
                base=base + cont,
                node_vals=node_vals,
                opt_choices=opt_choices,
                visited=visited,
                depth=depth + 1,
                max_depth=max_depth,
                show_inactive=show_inactive,
                shared_ids=shared_ids,
                encountered=encountered,
                compact=compact,
            )


def _print_shared_footer(encountered: dict, node_vals, opt_choices) -> None:
    if not encountered:
        return
    w = 48
    print()
    print("── Shared subtrees " + "─" * w)
    for node in encountered.values():
        print()
        _display_subtree(
            node,
            base="", node_vals=node_vals, opt_choices=opt_choices,
            visited=set(), depth=0, max_depth=None,
            show_inactive=True,
        )


def _print_distribution(result) -> None:
    dist = result.distribution
    if not dist:
        return

    payoff_w = max(len(_fv(p)) for p, _ in dist)
    payoff_w = max(payoff_w, 9)

    print()
    print("Outcome distribution (optimal strategy):")
    print(f"  {'Payoff':>{payoff_w}}    {'Prob':>7}    Cumulative")
    print(f"  {'-' * payoff_w}    {'-' * 7}    ----------")

    cumulative = 0.0
    for payoff, prob in dist:
        cumulative += prob
        print(f"  {_fv(payoff):>{payoff_w}}    {prob * 100:>6.1f}%    {cumulative * 100:.1f}%")

    print()
    print(f"  EV = {_fv(result.ev)}")
    if result.ce is not None:
        print(f"  CE = {_fv(result.ce)}")


# ---------------------------------------------------------------------------
# Rich rendering backend
# ---------------------------------------------------------------------------

def _rich_symbol_color(node) -> tuple[str, str]:
    """Return (symbol, rich_markup_color) for a node type."""
    c = _core.settings.colors
    if isinstance(node, DecisionNode):
        return "■", c.get("decision", "bold green")
    if isinstance(node, ChanceNode):
        return "●", c.get("chance", "bold red")
    if isinstance(node, LeafNode):
        return "▶", c.get("leaf", "bold blue")
    if isinstance(node, LogicNode):
        return "◆", c.get("logic", "bold magenta")
    return "?", "white"


def _rich_node_header(
    node, node_vals: dict | None,
    show_event_id: bool = False,
    path_prob: float | None = None,
    dimmed: bool = False,
) -> str:
    sym, color = _rich_symbol_color(node)
    if node_vals and node.event_id in node_vals:
        _v = node_vals[node.event_id]
        if isinstance(node, LeafNode):
            ev_str = "  " + _markup(f"v={_fv(_v)}", "value")
        else:
            ev_str = "  " + _markup(f"EV={_fv(_v)}", "ev")
    else:
        ev_str = ""
    eid_str = (
        f"  [dim](id: {str(node.event_id)[:8]})[/dim]"
        if show_event_id and hasattr(node, "event_id") else ""
    )
    # Path probability: only on leaf nodes in result display
    prob_str = ""
    if path_prob is not None and isinstance(node, LeafNode):
        prob_str = "  " + _markup(f"{path_prob:.1%}", "path_prob")
    header = f"[{color}]{sym}[/{color}] [bold]{node.name}[/bold]{ev_str}{prob_str}{eid_str}"
    # Nested [dim] combines with the inner colors rather than replacing them,
    # so a header inside an unreached subtree still reads as muted, not
    # selectively re-highlighted (e.g. a nested decision's own optimal pick).
    return f"[dim]{header}[/dim]" if dimmed else header


def _rich_branch_label(
    b, prob: float | None, is_opt: bool, show_inactive: bool,
    is_shared_ref: bool = False, compact: bool = True,
    dimmed: bool = False, path_prob: float | None = None,
    leaf_total: float | None = None,
) -> str:
    label = b.label if b.label is not None else "[branch]"
    force_dim = dimmed or (not b.active and show_inactive)

    if is_opt and not force_dim:
        label_part = _markup(f"➤ {label}", "optimal")
    else:
        label_part = label

    parts = [label_part]
    if prob is not None:
        parts.append(_markup(f"p={prob:.2f}", "prob"))
    val = _raw(b.value)
    is_terminal = b.child is None or isinstance(b.child, LeafNode)
    if val != 0 or is_terminal:
        parts.append(_markup(f"v={_fv(val)}", "value"))
    # Cumulative path total, only shown when it differs from this branch's
    # own edge value (i.e. other edges earlier on the path also carried a
    # nonzero value) — avoids redundant clutter on single-value paths.
    if leaf_total is not None and abs(leaf_total - val) > 1e-9:
        parts.append(_markup(f"total={_fv(leaf_total)}", "value"))
    # Cumulative path probability, compact-mode leaves only (non-compact
    # mode shows this on the leaf's own row via _rich_node_header instead).
    if path_prob is not None:
        parts.append(_markup(f"{path_prob:.1%}", "path_prob"))
    if b.time != 0:
        parts.append(f"[dim]time={b.time}[/dim]")

    if not compact:
        if b.child is None:
            parts.append("[dim]→ ◾ terminal[/dim]")
        elif is_shared_ref:
            parts.append(f"[dim]→ \\[{b.child.name}][/dim]")   # bracketed = see footer
        else:
            parts.append(f"[dim]→ {b.child.name}[/dim]")

    line = "  ".join(parts)
    if not b.active and show_inactive:
        line += "  " + _markup("[inactive]", "inactive")

    return f"[dim]{line}[/dim]" if force_dim else line


def _build_rich_subtree(
    node,
    rich_parent,
    node_vals: dict | None,
    opt_choices: dict,
    visited: set,
    depth: int,
    max_depth: int | None,
    show_event_id: bool = False,
    show_inactive: bool = True,
    shared_ids: set | None = None,
    encountered: dict | None = None,
    acc_prob: float = 1.0,
    compact: bool = True,
    on_optimal_path: bool = True,
) -> None:
    if node is None:
        return

    if id(node) in visited:
        sym, color = _rich_symbol_color(node)
        rich_parent.add(
            f"[{color}]{sym}[/{color}] [bold]{node.name}[/bold]  [dim]↑ (shared — see above)[/dim]"
        )
        return
    visited.add(id(node))

    all_branches = getattr(node, "branches", [])
    branches     = [b for b in all_branches if b.active or show_inactive]
    probs_all    = _branch_probs(node)
    probs        = [probs_all[i] for i, b in enumerate(all_branches) if b.active or show_inactive] if probs_all else None
    can_recurse  = max_depth is None or depth < max_depth
    is_decision  = isinstance(node, DecisionNode)
    # Per-node, not just "do we have any EV data at all": with a partial
    # reveal, node_vals is non-None but only carries entries for the
    # revealed nodes, so a decision must have its own entry before we know
    # which of its branches to mark optimal (and dim the rest).
    has_ev       = node_vals is not None and node.event_id in node_vals

    for i, b in enumerate(branches):
        is_opt = (
            has_ev
            and is_decision
            and opt_choices.get(node.event_id) == b.label
        )

        is_shared_ref = (
            shared_ids is not None
            and b.child is not None
            and id(b.child) in shared_ids
        )

        # A branch is unreached under the current policy either because an
        # ancestor branch already was (propagated via on_optimal_path), or
        # because this is itself a losing choice at a decision node. Once
        # off-path, the whole subtree below stays dimmed, including any
        # nested decision node's own (locally-optimal-but-moot) pick.
        branch_dimmed = (not on_optimal_path) or (is_decision and has_ev and not is_opt)
        child_on_optimal_path = not branch_dimmed

        if is_opt and not branch_dimmed:
            guide = _core.settings.colors.get("optimal", "bold green")
        elif branch_dimmed:
            # An explicit color, not the bare "dim" attribute: rich's guide_style
            # stacks additively down the tree, so a bare "dim" only reduces the
            # brightness of whatever color was inherited from an ancestor (e.g.
            # the root's own green, once connected to it) rather than replacing
            # it — every connector under a dimmed branch would stay tinted green.
            # An explicit color always wins over an inherited one.
            guide = "grey50"
        else:
            guide = ""

        prob = probs[i] if probs and i < len(probs) else None

        # Accumulate path probability: multiply at chance branches, unchanged elsewhere.
        # Non-optimal decision branches get None — path prob is meaningless there.
        if isinstance(node, ChanceNode) and prob is not None:
            child_acc_prob = acc_prob * prob if acc_prob is not None else None
        elif is_decision and (not has_ev or not is_opt):
            # Not on the optimal path, or this decision isn't revealed yet
            # (an unrevealed decision has no meaningful path probability to
            # show on any of its branches, optimal or not).
            child_acc_prob = None
        else:
            child_acc_prob = acc_prob
        # Surface path_prob on child header only in result display and only for leaves
        child_path_prob = child_acc_prob if node_vals is not None else None

        # In compact mode, a leaf child adds no separate row, so its path
        # probability and (if it differs from this branch's own edge value)
        # cumulative path total are folded onto the branch line instead.
        is_compact_leaf = compact and isinstance(b.child, LeafNode) and not is_shared_ref
        leaf_total = None
        if is_compact_leaf and node_vals is not None and b.child.event_id in node_vals:
            leaf_total = node_vals[b.child.event_id]

        # A losing branch is only marked inactive/greyed once this decision
        # itself is revealed (has_ev) — otherwise a not-yet-revealed
        # decision would leak its answer through styling alone, even with
        # no EV shown anywhere.
        branch_label = _rich_branch_label(
            b, prob, is_opt, show_inactive and has_ev, is_shared_ref, compact,
            dimmed=branch_dimmed,
            path_prob=(child_path_prob if is_compact_leaf else None),
            leaf_total=leaf_total,
        )
        branch_tree = rich_parent.add(branch_label, guide_style=guide)

        # In compact mode, a leaf child adds no information beyond what's
        # already on the branch line (its name is never shown), so skip it.
        if is_compact_leaf:
            continue

        if is_shared_ref:
            if encountered is not None:
                encountered.setdefault(id(b.child), b.child)
        elif can_recurse and b.child is not None:
            child_header = _rich_node_header(
                b.child, node_vals, show_event_id,
                path_prob=child_path_prob, dimmed=branch_dimmed,
            )
            if id(b.child) in visited:
                branch_tree.add(child_header + "  [dim]↑ (shared — see above)[/dim]")
            else:
                # Decision nodes in result view get the optimal guide style so their
                # own branch connectors (├── / └──) are also highlighted, but only
                # when this region is itself still on the optimal path.
                child_guide = (
                    _core.settings.colors.get("optimal", "bold green")
                    if (not branch_dimmed
                        and has_ev
                        and isinstance(b.child, DecisionNode)
                        and b.child.event_id in opt_choices)
                    else guide
                )
                child_tree = branch_tree.add(child_header, guide_style=child_guide)
                _build_rich_subtree(
                    b.child,
                    child_tree,
                    node_vals,
                    opt_choices,
                    visited,
                    depth + 1,
                    max_depth,
                    show_event_id,
                    show_inactive,
                    shared_ids,
                    encountered,
                    child_acc_prob,
                    compact,
                    child_on_optimal_path,
                )


def _rich_shared_footer(encountered: dict, node_vals, opt_choices, console) -> None:
    if not encountered:
        return
    from rich.rule import Rule
    from rich.tree import Tree as RichTree
    console.print()
    console.print(Rule("Shared subtrees", style="dim"))
    for node in encountered.values():
        rich_tree = RichTree(_rich_node_header(node, node_vals))
        _build_rich_subtree(
            node, rich_tree,
            node_vals=node_vals, opt_choices=opt_choices,
            visited=set(), depth=0, max_depth=None,
            show_inactive=True,
        )
        console.print()
        console.print(rich_tree)


def _rich_print_distribution(result, console) -> None:
    from rich.table import Table

    dist = result.distribution
    if not dist:
        return

    table = Table(title="Outcome distribution (optimal strategy)", show_lines=False)
    table.add_column("Payoff",     justify="right",  style="bold")
    table.add_column("Prob",       justify="right",  style="cyan")
    table.add_column("Cumulative", justify="right",  style="dim")

    cumulative = 0.0
    for payoff, prob in dist:
        cumulative += prob
        table.add_row(_fv(payoff), f"{prob * 100:.1f}%", f"{cumulative * 100:.1f}%")

    console.print()
    console.print(table)
    console.print()
    console.print(f"  [bold]EV[/bold] = [bold]{_fv(result.ev)}[/bold]")
    if result.ce is not None:
        console.print(f"  [bold]CE[/bold] = [bold]{_fv(result.ce)}[/bold]")


# ---------------------------------------------------------------------------
# Public entry points  (dispatch plain / rich)
# ---------------------------------------------------------------------------

def _rich_console(record: bool = False):
    from rich.console import Console
    from .core import settings as _settings
    force = (_settings.style == "rich")
    return Console(force_terminal=force, record=record)


def _light_palette():
    return {
        "background": (255, 255, 255), "foreground": (51, 51, 51), "fg_css": "#333333",
        "normal": [
            (30, 30, 30), (176, 43, 43), (56, 128, 60), (161, 110, 8),
            (34, 82, 156), (140, 45, 140), (24, 116, 132), (100, 100, 100),
        ],
        "bright": [
            (0, 0, 0), (200, 50, 50), (60, 150, 70), (185, 130, 10),
            (40, 100, 190), (160, 55, 160), (25, 140, 160), (130, 130, 130),
        ],
    }


def _dark_palette():
    # Rich's own SVG_EXPORT_THEME: the exact palette save_svg() already uses
    # (and this session's dark-background tree exports all along), rather
    # than a hand-picked one. Rich's "dim" style blends toward black, which
    # washes out low-contrast custom colors on a dark background; this
    # theme's colors are specifically tuned to still read well when dimmed.
    from rich.console import SVG_EXPORT_THEME as _t
    return {
        "background": tuple(_t.background_color), "foreground": tuple(_t.foreground_color),
        "fg_css": "#{:02x}{:02x}{:02x}".format(*_t.foreground_color),
        "normal": [tuple(_t.ansi_colors[i]) for i in range(8)],
        "bright": [tuple(_t.ansi_colors[i]) for i in range(8, 16)],
    }


def _terminal_theme_from_palette(palette: dict):
    from rich.terminal_theme import TerminalTheme
    return TerminalTheme(palette["background"], palette["foreground"], palette["normal"], palette["bright"])


def _parse_hex(color: str) -> tuple[int, int, int] | None:
    color = color.lstrip("#")
    if len(color) == 3:
        color = "".join(c * 2 for c in color)
    if len(color) != 6:
        return None
    try:
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return None


def _is_light_hex(color: str) -> bool:
    rgb = _parse_hex(color)
    if rgb is None:
        return True  # unrecognised format: default to a light-safe palette
    r, g, b = rgb
    return (0.299 * r + 0.587 * g + 0.114 * b) > 140


def _ensure_span_colors(html: str, fg_css: str) -> str:
    """Give every <span> an explicit color, not just an inherited one.

    Rich only colors spans that carry an explicit style (e.g. the green
    decision-node marker); plain text like a bare bold node name has no
    color of its own and relies on inheriting from the <pre>/<code>
    ancestor. That's fragile if the host page's own CSS ever wins that
    inheritance chain (observed in VS Code), so make every span explicit.
    """
    import re

    def repl(match: "re.Match[str]") -> str:
        style = match.group(1)
        if re.search(r"(?<!text-decoration-)color:", style):
            return match.group(0)
        sep = ";" if style and not style.endswith(";") else ""
        return f'<span style="{style}{sep}color:{fg_css};">'

    return re.sub(r'<span style="([^"]*)">', repl, html)


def _export_body(console, palette, bg_css: str | None = None) -> tuple[str, str, str]:
    """Export a recorded console's contents with the given palette, as just
    the inner <pre>/<code> HTML (no surrounding <html>/<body>).

    bg_css overrides the CSS background string (e.g. a custom color the
    caller passed in directly), while palette["background"] still supplies
    the RGB triple rich's theme mapping needs internally.
    """
    theme = _terminal_theme_from_palette(palette)
    html = console.export_html(theme=theme, inline_styles=True, clear=False)
    body = html.split("<body>", 1)[1].rsplit("</body>", 1)[0]
    # Set background/color inline on the <pre>/<code> tags themselves, not
    # just a wrapping <div>: some notebook front-ends (e.g. VS Code) apply
    # their own background straight to <pre>/<code> elements, which wins over
    # a background set only on an ancestor (background-color doesn't
    # inherit, so a more specific rule on the element itself overrides it).
    bg_css = bg_css or "#{:02x}{:02x}{:02x}".format(*palette["background"])
    fg_css = palette["fg_css"]
    body = body.replace(
        '<pre style="', f'<pre style="background:{bg_css};color:{fg_css};', 1,
    ).replace(
        '<code style="font-family:inherit"',
        f'<code style="font-family:inherit;background:{bg_css};color:{fg_css}"',
        1,
    )
    body = _ensure_span_colors(body, fg_css)
    return body, bg_css, fg_css


def _jupyter_display_html(console) -> None:
    """Render a buffered console's contents via IPython.display, honouring
    dtree.settings.background:

    "transparent" (default) — no forced background; embeds both a light-safe
        and a dark-safe render and lets CSS prefers-color-scheme pick the
        right one for the viewer's notebook theme.
    "light" / "dark" — force that background regardless of notebook theme.
    any CSS color — force that exact background; text colour auto-picked
        for contrast.
    """
    from IPython.display import display, HTML
    from .core import settings as _settings
    bg_setting = getattr(_settings, "background", "transparent")

    if bg_setting == "transparent":
        # Genuinely transparent: text colors switch per light/dark palette
        # for contrast, but the box itself has no forced fill, so it blends
        # with whatever the notebook page's real background is.
        light_body, _, _ = _export_body(console, _light_palette(), bg_css="transparent")
        dark_body, _, _ = _export_body(console, _dark_palette(), bg_css="transparent")
        uid = f"dtree-{id(console):x}"
        # Three fallbacks, in order, since no single one is reliable across
        # notebook front-ends:
        #   1. VS Code's documented body class (vscode-dark/vscode-light) —
        #      the most reliable signal *when* our output lands in the same
        #      document as that class, which isn't guaranteed if VS Code
        #      renders this output inside a further-nested sandboxed iframe.
        #   2. Walk up the DOM for the nearest actual non-transparent
        #      background and measure its luminance (works when the
        #      surrounding page's real background is reachable in our DOM).
        #   3. CSS prefers-color-scheme (@media below, and matchMedia as a
        #      script-side check too) — a last resort, since it often
        #      doesn't propagate into a sandboxed output webview at all.
        html_out = (
            f"<style>.{uid}-light, .{uid}-dark {{ overflow-x: auto; }}"
            f".{uid}-dark {{ display: none; }}"
            f"@media (prefers-color-scheme: dark) {{"
            f".{uid}-light {{ display: none; }} .{uid}-dark {{ display: block; }}"
            f"}}</style>"
            f'<div id="{uid}">'
            f'<div class="{uid}-light">{light_body}</div>'
            f'<div class="{uid}-dark">{dark_body}</div>'
            f"</div>"
            f"<script>(function(){{"
            f'var root=document.getElementById("{uid}");if(!root)return;'
            f"var dark=null;"
            f"var cls=(document.body&&document.body.className)||\"\";"
            f'if(/vscode-dark|vscode-high-contrast/.test(cls))dark=true;'
            f'else if(/vscode-light/.test(cls))dark=false;'
            f"if(dark===null){{"
            f"var el=root.parentElement,bg=null;"
            f"while(el){{"
            f"var c=getComputedStyle(el).backgroundColor;"
            f'if(c&&c!=="rgba(0, 0, 0, 0)"&&c!=="transparent"){{bg=c;break;}}'
            f"el=el.parentElement;"
            f"}}"
            f"if(bg){{"
            f"var m=bg.match(/[\\d.]+/g);"
            f"if(m&&m.length>=3)dark=(0.299*m[0]+0.587*m[1]+0.114*m[2])<140;"
            f"}}"
            f"}}"
            f"if(dark===null&&window.matchMedia){{"
            f'dark=window.matchMedia("(prefers-color-scheme: dark)").matches;'
            f"}}"
            f"if(dark!==null){{"
            f'var light=root.querySelector(".{uid}-light"),d=root.querySelector(".{uid}-dark");'
            f'if(dark){{light.style.display="none";d.style.display="block";}}'
            f'else{{light.style.display="block";d.style.display="none";}}'
            f"}}"
            f"}})();</script>"
        )
        display(HTML(html_out))
        return

    if bg_setting == "light":
        palette = _light_palette()
    elif bg_setting == "dark":
        palette = _dark_palette()
    else:
        base = _light_palette() if _is_light_hex(bg_setting) else _dark_palette()
        palette = dict(base)
        custom_rgb = _parse_hex(bg_setting)
        if custom_rgb is not None:
            palette["background"] = custom_rgb

    body, bg_css, fg_css = _export_body(console, palette, bg_css=bg_setting if bg_setting not in ("light", "dark") else None)

    display(HTML(
        f'<div style="background:{bg_css};color:{fg_css};padding:10px 16px;'
        f'border-radius:6px;display:inline-block;overflow-x:auto;">' + body + "</div>"
    ))


def _interactive_console():
    """Console for interactive display() calls.

    In a real Jupyter kernel, prints are buffered (not written to stdout) so
    they can be rendered afterward as an explicit white-background HTML block
    via _jupyter_display_html, instead of Jupyter's raw-ANSI black
    terminal-style rendering (force_terminal=True piped to stdout would
    otherwise still get the black-terminal treatment from Jupyter's own
    ANSI-to-HTML conversion).

    Returns (console, is_jupyter).
    """
    import io
    from rich.console import Console
    from .core import settings as _settings
    if Console().is_jupyter:
        # force_jupyter=False stops rich from *also* auto-publishing its own
        # (unstyled, no-background) Jupyter output as a side effect of
        # print(), on top of the one we explicitly build and display below.
        return Console(record=True, force_terminal=True, force_jupyter=False,
                       file=io.StringIO(), width=120), True
    force = (_settings.style == "rich")
    return Console(force_terminal=force), False


def display_node(node, depth: int | None, expand_shared: bool = True,
                 _console=None) -> None:
    """Implements node.display(depth, expand_shared)."""
    max_d      = 0 if depth is None else depth
    shared_ids = None if expand_shared else _collect_shared_ids(node)
    encountered: dict = {}

    if _use_rich():
        from rich.tree import Tree as RichTree
        owns_console = _console is None
        console, is_jupyter = (_console, False) if _console is not None else _interactive_console()
        rich_tree = RichTree(_rich_node_header(node, None, show_event_id=True))
        _build_rich_subtree(
            node, rich_tree,
            node_vals=None, opt_choices={},
            visited=set(), depth=0, max_depth=max_d,
            show_event_id=True,
            shared_ids=shared_ids, encountered=encountered,
        )
        console.print(rich_tree)
        _rich_shared_footer(encountered, None, {}, console)
        if owns_console and is_jupyter:
            _jupyter_display_html(console)
    else:
        _display_subtree(
            node,
            base="", node_vals=None, opt_choices={},
            visited=set(), depth=0, max_depth=max_d,
            show_event_id=True,
            shared_ids=shared_ids, encountered=encountered,
        )
        _print_shared_footer(encountered, None, {})


def display_tree(tree, max_depth: int | None, expand_shared: bool = True,
                 compact: bool = True, _console=None) -> None:
    """Implements tree.display(max_depth, expand_shared, compact)."""
    shared_ids = None if expand_shared else _collect_shared_ids(tree.root)
    encountered: dict = {}

    if _use_rich():
        from rich.tree import Tree as RichTree
        owns_console = _console is None
        console, is_jupyter = (_console, False) if _console is not None else _interactive_console()
        rich_tree = RichTree(_rich_node_header(tree.root, None))
        _build_rich_subtree(
            tree.root, rich_tree,
            node_vals=None, opt_choices={},
            visited=set(), depth=0, max_depth=max_depth,
            show_inactive=True,
            shared_ids=shared_ids, encountered=encountered,
            compact=compact,
        )
        console.print(rich_tree)
        _rich_shared_footer(encountered, None, {}, console)
        if owns_console and is_jupyter:
            _jupyter_display_html(console)
    else:
        _display_subtree(
            tree.root,
            base="", node_vals=None, opt_choices={},
            visited=set(), depth=0, max_depth=max_depth,
            show_inactive=True,
            shared_ids=shared_ids, encountered=encountered,
            compact=compact,
        )
        _print_shared_footer(encountered, None, {})


def display_result(
    result, view: str = "ev", max_depth: int | None = None,
    expand_shared: bool = True, policy_only: bool = False,
    compact: bool = True, show_distribution: bool = True,
    reveal: list[str] | None = None, _console=None,
) -> None:
    """Implements result.display(view, max_depth, expand_shared, policy_only,
    compact, show_distribution, reveal).

    policy_only=True removes sub-optimal decision branches, leaving only the
    optimal path at each decision node.
    compact=True drops the "→ target" hint on every branch and the separate
    leaf-node row (its value already sits on the branch line), for a denser
    tree meant for slides rather than debugging.
    show_distribution=False omits the outcome-distribution table, so the
    tree and the distribution can be exported/displayed separately.
    reveal=[...] restricts EV annotations and optimal-branch arrows to the
    named chance/decision nodes; every other node renders as unsolved.
    """
    if view not in ("ev",):
        print(f"[dtree] view='{view}' not yet supported; showing 'ev'.")

    opt_choices = _collect_opt_choices(result.policy.root)
    shared_ids  = None if expand_shared else _collect_shared_ids(result.policy.root)
    encountered: dict = {}
    show_inactive = not policy_only

    node_vals = getattr(result, "_node_values_by_id", None) or result.node_values

    if reveal is not None:
        revealed_ids = _collect_revealed_ids(result.policy.root, set(reveal))
        node_vals = {k: v for k, v in node_vals.items() if k in revealed_ids}
        opt_choices = {k: v for k, v in opt_choices.items() if k in revealed_ids}

    if _use_rich():
        from rich.tree import Tree as RichTree
        owns_console = _console is None
        console, is_jupyter = (_console, False) if _console is not None else _interactive_console()
        # A decision root's own optimal choice is recorded in opt_choices too, so
        # its connectors (into all its children, per rich's one-guide-per-parent
        # model) get the same green treatment nested decision nodes already get —
        # otherwise the optimal-path highlight only starts one level down, looking
        # disconnected from the root. A chance root has no "choice" to mark in the
        # general case, but in policy_only mode every sub-optimal decision branch
        # has already been pruned, so everything still shown (including a chance
        # root's own branches) *is* the policy — leaving the root ungreened there
        # would make it look disconnected from the fully-green tree below it.
        root_is_greened_decision = (
            isinstance(result.policy.root, DecisionNode)
            and result.policy.root.event_id in opt_choices
        )
        root_is_policy_chance = policy_only and isinstance(result.policy.root, ChanceNode)
        root_guide = (
            _core.settings.colors.get("optimal", "bold green")
            if (root_is_greened_decision or root_is_policy_chance)
            else "tree.line"
        )
        rich_tree = RichTree(_rich_node_header(result.policy.root, node_vals), guide_style=root_guide)
        _build_rich_subtree(
            result.policy.root, rich_tree,
            node_vals=node_vals, opt_choices=opt_choices,
            visited=set(), depth=0, max_depth=max_depth,
            show_inactive=show_inactive,
            shared_ids=shared_ids, encountered=encountered,
            compact=compact,
        )
        console.print(rich_tree)
        _rich_shared_footer(encountered, node_vals, opt_choices, console)
        if show_distribution:
            _rich_print_distribution(result, console)
        if owns_console and is_jupyter:
            _jupyter_display_html(console)
    else:
        _display_subtree(
            result.policy.root,
            base="", node_vals=node_vals, opt_choices=opt_choices,
            visited=set(), depth=0, max_depth=max_depth,
            show_inactive=show_inactive,
            shared_ids=shared_ids, encountered=encountered,
            compact=compact,
        )
        _print_shared_footer(encountered, node_vals, opt_choices)
        if show_distribution:
            _print_distribution(result)


# ---------------------------------------------------------------------------
# SVG export
# ---------------------------------------------------------------------------

def _make_recording_console(width: int = 120):
    from rich.console import Console
    # force_jupyter=False: this console is only ever used to build up
    # content for export_svg()/save_svg() — printing to it should not also
    # auto-publish a second, unstyled Jupyter display as a side effect when
    # called from inside a real kernel (the same issue _interactive_console
    # guards against for the interactive display() path).
    return Console(record=True, force_terminal=True, force_jupyter=False, width=width)


def save_svg_tree(tree, path: str, max_depth: int | None = None,
                  expand_shared: bool = True, width: int = 120,
                  title: str = "Decision Tree", compact: bool = True) -> None:
    console = _make_recording_console(width)
    display_tree(tree, max_depth, expand_shared=expand_shared, compact=compact,
                _console=console)
    _write_svg(console, path, title)


def save_svg_result(result, path: str, view: str = "ev",
                    max_depth: int | None = None, expand_shared: bool = True,
                    width: int = 120, title: str = "Decision Tree",
                    compact: bool = True, show_distribution: bool = True,
                    reveal: list[str] | None = None, policy_only: bool = False) -> None:
    console = _make_recording_console(width)
    display_result(result, view=view, max_depth=max_depth,
                   expand_shared=expand_shared, policy_only=policy_only, compact=compact,
                   show_distribution=show_distribution, reveal=reveal,
                   _console=console)
    _write_svg(console, path, title)


def _write_svg(console, path: str, title: str) -> None:
    from .core import settings as _settings

    # A saved SVG is a static file, there is no surrounding page to toggle
    # against, so "transparent" (the interactive-display default) resolves
    # to "light" here rather than embedding an unusable JS toggle.
    bg_setting = getattr(_settings, "background", "transparent")
    if bg_setting in ("transparent", "light"):
        palette = _light_palette()
    elif bg_setting == "dark":
        palette = _dark_palette()
    else:
        palette = _light_palette() if _is_light_hex(bg_setting) else _dark_palette()
        custom_rgb = _parse_hex(bg_setting)
        if custom_rgb is not None:
            palette = dict(palette)
            palette["background"] = custom_rgb

    theme = _terminal_theme_from_palette(palette)
    svg = console.export_svg(title=title, theme=theme)
    svg = _strip_svg_chrome(svg)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"Saved: {path}")


def _strip_svg_chrome(svg_text: str, padding: int = 8) -> str:
    """
    Remove rich terminal chrome (title bar, window buttons), shift content up,
    and trim blank space on the right.
    """
    import xml.etree.ElementTree as ET
    import re

    NS  = "http://www.w3.org/2000/svg"
    NSP = f"{{{NS}}}"
    ET.register_namespace("", NS)

    root = ET.fromstring(svg_text)

    # ── 1. Find content group and its current y-offset (header height) ────────
    header_h   = 41.0   # rich default
    content_grp = None
    for child in root:
        m = re.search(r"translate\(9,\s*(\d+(?:\.\d+)?)\)", child.get("transform", ""))
        if m and child.tag == f"{NSP}g":
            header_h    = float(m.group(1))
            content_grp = child
            break
    shift = header_h - padding   # pixels to remove from height

    # ── 2. Remove title text and buttons group ────────────────────────────────
    for child in list(root):
        tag = child.tag
        if tag == f"{NSP}text" and "title" in child.get("class", ""):
            root.remove(child)
        elif tag == f"{NSP}g" and re.search(r"translate\(26", child.get("transform", "")):
            root.remove(child)

    # ── 3. Shift content group up ─────────────────────────────────────────────
    if content_grp is not None:
        content_grp.set("transform", f"translate(9, {padding})")

    # ── 4. Find actual rightmost content (skip line-end elements past clip) ───
    clip_w = None
    for clip in root.iter(f"{NSP}clipPath"):
        for rect in clip.iter(f"{NSP}rect"):
            w = rect.get("width")
            if w:
                clip_w = float(w)
                break
        if clip_w is not None:
            break

    max_x = 0.0
    for text_el in root.iter(f"{NSP}text"):
        x = float(text_el.get("x", 0))
        if clip_w is not None and x >= clip_w:   # line-end padding element
            continue
        tl = float(text_el.get("textLength", 0))
        if tl > 0:
            max_x = max(max_x, x + tl)

    # left-margin=9 from translate(9,...) + 2 for 1px borders on each side
    new_w = round(9 + max_x + padding + 2, 1) if max_x > 0 else None

    # ── 5. Adjust background rect ─────────────────────────────────────────────
    for child in root:
        if child.tag == f"{NSP}rect":
            h = float(child.get("height", 0))
            child.set("height", str(round(h - shift, 1)))
            if new_w is not None:
                child.set("width", str(round(new_w - 2, 1)))
            break

    # ── 6. Adjust viewBox ─────────────────────────────────────────────────────
    vb = root.get("viewBox", "").split()
    if len(vb) == 4:
        new_h = round(float(vb[3]) - shift, 1)
        w_str = str(round(new_w, 1)) if new_w is not None else vb[2]
        root.set("viewBox", f"0 0 {w_str} {new_h}")

    return ET.tostring(root, encoding="unicode")
