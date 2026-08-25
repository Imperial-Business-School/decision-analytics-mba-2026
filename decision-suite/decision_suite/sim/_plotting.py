"""
dsim._plotting — private: interactive (ipywidgets) chart rendering only.

Non-interactive charts (histograms, bar charts, line charts, waterfalls,
tornado bars) aren't implemented here — each is a few lines of standard
matplotlib a student composes directly in the notebook against a result
object's already-exposed raw data (`.objectives`, `.values`, `.grid`, etc.),
not something an LLM would reliably get wrong if asked to write it fresh.

The two interactive widgets here are the exception: real, hard-won
debugging effort went into making ipywidgets render *and stay interactive*
reliably in VS Code's Jupyter extension (redraw correctness, avoiding
`Output`/`clear_output` bugs that caused stacking instead of replacing),
exactly the kind of subtle, non-obvious, expensive-to-rediscover problem
worth protecting in a tested package function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .core import Model, RunResult, SimulationResult

_ACCENT = "#376092"
_ACCENT_LIGHT = "#a9c2dc"
_BAD = "#b5433c"
_GOOD = "#4a8c5f"


def outcome_waterfall(result: "RunResult", steps: list[tuple[str, float]] | None = None, figsize: tuple[float, float] = (8, 4.5)):
    """
    Waterfall chart showing how signed components add up to the objective —
    e.g. +revenue, -operating cost, -fixed cost, -tax, = NPV. Internal
    helper used by `interactive_parameters` to redraw on every slider move;
    not exposed as a standalone `.plot()` method (composing a one-off
    waterfall directly in the notebook is straightforward when needed).

    Parameters
    ----------
    steps : list[tuple[str, float]] | None
        Ordered (label, signed_value) pairs — positive values raise the
        running total, negative values lower it, e.g.
        ``[("revenue", result.outcomes["revenue"]),
           ("operating costs", -result.outcomes["operating_costs"]), ...]``.
        You choose the sign, since `result.outcomes` stores costs as plain
        positive magnitudes (business-readable), not signed deltas. If
        omitted, falls back to a plain bar per outcome (no cumulative
        stacking) — an arbitrary outcomes dict mixes independent deltas
        (revenue, operating_costs) with running subtotals (profit,
        taxable_profit), so a safe default can't assume every value is a
        delta to stack.
    """
    if steps is None:
        labels = list(result.outcomes.keys())
        values = [result.outcomes[k] for k in labels]
        fig, ax = plt.subplots(figsize=figsize)
        colors = [_ACCENT if label == result.objective_name else (_GOOD if v >= 0 else _BAD) for label, v in zip(labels, values)]
        ax.bar(range(len(labels)), values, color=colors)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Outcomes for this run")
        fig.tight_layout()
        plt.close(fig)
        return fig

    labels = [label for label, _ in steps]
    values = [v for _, v in steps]
    scale = max((abs(v) for v in values), default=1) or 1

    fig, ax = plt.subplots(figsize=figsize)
    running = 0.0
    for i, (label, v) in enumerate(zip(labels, values)):
        bottom = min(running, running + v)
        ax.bar(i, abs(v), bottom=bottom, color=(_GOOD if v >= 0 else _BAD), width=0.6)
        ax.text(i, running + v + np.sign(v) * scale * 0.02, f"{v:+,.0f}",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        running += v

    ax.bar(len(labels), running, color=_ACCENT, width=0.6)
    ax.text(len(labels), running + np.sign(running or 1) * scale * 0.02, f"{running:,.0f}",
             ha="center", va="bottom" if running >= 0 else "top", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(labels) + 1))
    ax.set_xticklabels(labels + [result.objective_name], rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"How the outcomes build up to {result.objective_name}")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ===========================================================================
# INTERACTIVE (ipywidgets — classic DOM widgets, reliably interactive across
# Jupyter frontends including VS Code; Plotly's newer FigureWidget (built on
# anywidget) renders but does not respond to mouse input in VS Code's
# notebook renderer, so it's deliberately not used here)
# ===========================================================================

def interactive_risk_range(result: "SimulationResult", figsize: tuple[float, float] = (7, 4.5)):
    """
    Interactive version of the histogram: a two-handled range slider lets
    you drag a low/high threshold across the simulated objective, with the
    shaded region and P(low <= objective <= high) readout updating live —
    the Jupyter equivalent of @Risk's draggable percentile lines on its
    Cumulative Ascending graph.

    The slider is the control; the *chart itself* draws a bracket between
    the current low/high (inside the same Axes as the histogram), which is
    what actually stays aligned with the data — a separate HTML slider can
    never be pixel-matched to a matplotlib image reliably.

    Requires ipywidgets and a Jupyter frontend that renders it (VS Code's
    Jupyter extension does). Returns the widget — display it by leaving
    the call as a cell's last line (Jupyter auto-displays it), the normal
    way ipywidgets are used, rather than this function displaying it itself.

    Redraws update an `Image` widget's `.value` (raw PNG bytes) rather than
    using an `Output` widget's `display()`/`clear_output()` — some Jupyter
    frontends don't reliably honour `clear_output`, causing each redraw to
    stack a new image instead of replacing the old one. Setting `.value` on
    an Image widget is a plain trait update, so there's nothing to stack.
    """
    import ipywidgets as widgets

    objectives = result.objectives
    lo, hi = float(objectives.min()), float(objectives.max())

    slider = widgets.FloatRangeSlider(
        value=[float(np.percentile(objectives, 5)), float(np.percentile(objectives, 95))],
        min=lo, max=hi, step=(hi - lo) / 200,
        description="range", continuous_update=True,
        layout=widgets.Layout(width=f"{int(figsize[0] * plt.rcParams['figure.dpi'])}px"),
        style={"description_width": "initial"},
    )
    image = widgets.Image(format="png")

    def render_png(low: float, high: float) -> bytes:
        fig, ax = plt.subplots(figsize=figsize)
        counts, *_ = ax.hist(objectives, bins=50, color=_ACCENT_LIGHT, edgecolor="white")
        ax.axvline(low, color=_BAD, linewidth=1.5)
        ax.axvline(high, color=_BAD, linewidth=1.5)
        ax.axvspan(low, high, color=_BAD, alpha=0.15)

        # A bracket drawn *inside this same Axes* is the authoritative
        # "where is the selected range" indicator — guaranteed to line
        # up exactly with the two vertical lines above, since they share
        # one coordinate system.
        y_bracket = counts.max() * 1.08
        ax.annotate(
            "", xy=(low, y_bracket), xytext=(high, y_bracket),
            arrowprops=dict(arrowstyle="<->", color=_BAD, linewidth=1.5),
        )
        ax.set_ylim(top=y_bracket * 1.15)

        p_between = float(((objectives >= low) & (objectives <= high)).mean())
        p_below = float((objectives < low).mean())
        p_above = float((objectives > high).mean())
        ax.set_xlabel("objective")
        ax.set_ylabel("iterations")
        ax.set_title(
            f"P({low:,.0f} ≤ objective ≤ {high:,.0f}) = {p_between:.1%}\n"
            f"(below range: {p_below:.1%}, above range: {p_above:.1%})"
        )
        fig.tight_layout()

        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()

    def redraw(_change=None):
        low, high = slider.value
        image.value = render_png(low, high)

    slider.observe(redraw, names="value")
    redraw()

    return widgets.VBox([slider, image], layout=widgets.Layout(align_items="center"))


def interactive_parameters(model: "Model", parameters: list[str] | None = None, figsize: tuple[float, float] = (7, 4.5)):
    """
    One slider per parameter, live-updating the outcome waterfall as you
    drag any of them — the Jupyter equivalent of typing a new value into
    an Excel cell and watching the sheet recalculate.

    Parameters
    ----------
    parameters : list[str] | None
        Which parameters get a slider. If omitted, every parameter on
        `model` that has a low/high range is included — fixed parameters
        (no range) are excluded automatically, since there'd be nothing
        to drag.

    Redraws update an `Image` widget's `.value` (raw PNG bytes) rather than
    using an `Output` widget's `display()`/`clear_output()` — see
    `interactive_risk_range`'s docstring for why.
    """
    import io
    import ipywidgets as widgets

    names = parameters or [name for name, p in model.parameters.items() if p.has_range()]
    if not names:
        raise ValueError(
            "No parameters have a range to explore — set low/high or a distribution "
            "on at least one Parameter, or pass parameters= explicitly."
        )

    sliders = {}
    for name in names:
        p = model.parameters[name]
        low, high = p.range()
        sliders[name] = widgets.FloatSlider(
            value=p.base, min=low, max=high, step=(high - low) / 100,
            description=name, continuous_update=True,
            style={"description_width": "120px"},
            layout=widgets.Layout(width="260px"),
        )

    image = widgets.Image(format="png")

    def redraw(_change=None):
        overrides = {name: s.value for name, s in sliders.items()}
        result = model.run(**overrides)
        fig = outcome_waterfall(result, figsize=figsize)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        image.value = buf.getvalue()

    for s in sliders.values():
        s.observe(redraw, names="value")
    redraw()

    controls = widgets.VBox(list(sliders.values()))
    return widgets.HBox([controls, image])
