import math
import random
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path

from src.utils.enums import Dataset


def plot_histogram(
    data: pd.Series | np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    tick_fontsize: int = 10,
    annotation_fontsize: int = 10,
    legend_fontsize: int = 12,
    label_fontsize: int = 12,
    title_fontsize: int = 14,
    bins: int = 10,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (8, 4),
    annotate: bool = False,
    median: float | None = None,
    median_label: str = "Median",
    median_color: str = "red",
    median_linestyle: str = "--",
    q1: float | None = None,
    q3: float | None = None,
    mean: float | None = None,
    mean_label: str = "Mean",
    mean_color: str = "green",
    mean_linestyle: str = ":",
    std: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    color: str | None = None,
) -> Figure:
    """Plot a histogram using seaborn.

    Args:
        data (pd.Series | np.ndarray): Data to plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        ax (Axes | None): Axes to draw into. If None, a new figure is created.
        figsize: Figure size as (width, height). Only used when ax is None. Default is (8, 4).
        annotate (bool): If True, annotates each bar with its height in bold.
        median (float | None): If provided and not NaN, plots a dashed vertical line at
            this value.
        median_label (str): Legend label for the median line. Default is "Median".
        q1 (float | None): First quartile. If both q1 and q3 are provided and not NaN,
            plots a shaded region spanning [q1, q3] to visualise the IQR.
        q3 (float | None): Third quartile. See q1.
        mean (float | None): If provided and not NaN, plots a dotted vertical line at
            this value.
        mean_label (str): Legend label for the mean line. Default is "Mean".
        std (float | None): If provided and not NaN, plots a shaded region for
            mean ± std. Requires mean to also be provided and not NaN.
        xlim (tuple[float, float] | None): (min, max) range for the bins. If None,
            the range is inferred from the data.
        ylim (tuple[float, float] | None): (min, max) limits for the y-axis. If None,
            the limits are inferred from the data.
        color (str | None): Bar color. If provided, the median/IQR overlay uses the
            color 1/3 around the HSV wheel (120°) and the mean/std overlay uses the
            color 2/3 around the wheel (240°), forming a triadic scheme with the bars.
            If None, median/IQR defaults to red and mean/std defaults to green.
    """
    if owns_figure := (ax is None):
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Automatically set x-axis limits
    x_min, x_max = xlim if xlim is not None else (data.min(), data.max())
    sns.histplot(
        data=data,
        ax=ax,
        bins=bins,
        binrange=(x_min, x_max),
        color=color,
    )

    # Set x-axis limits
    ax.set_xlim(x_min, x_max)

    # Set y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(*ylim)

    # Set title, x-axis label, y-axis label, and font sizes
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    # Set font sizes for tick labels
    plt.setp(ax.get_xticklabels(), fontsize=tick_fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=tick_fontsize)

    # Optionally annotate bars with their heights
    if annotate:
        for bar in ax.patches:
            if (height := bar.get_height()) == 0:
                continue
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=annotation_fontsize,
            )

    # Optionally plot reference lines and shaded regions for median ± IQR
    has_overlay = False

    valid_median = median is not None and not math.isnan(median)
    if valid_median:
        ax.axvline(
            median,
            color=median_color,
            linestyle=median_linestyle,
            label=f"{median_label}: {median:.3g}",
        )
        has_overlay = True

        valid_q1 = q1 is not None and not math.isnan(q1)
        valid_q3 = q3 is not None and not math.isnan(q3)
        if valid_q1 and valid_q3:
            ax.axvspan(
                q1,
                q3,
                alpha=0.2,
                color=median_color,
                label=f"IQR: {q3 - q1:.3g}",
            )

    # Optionally plot reference lines and shaded regions for mean ± std
    valid_mean = mean is not None and not math.isnan(mean)
    if valid_mean:
        ax.axvline(
            mean,
            color=mean_color,
            linestyle=mean_linestyle,
            label=f"{mean_label}: {mean:.3g}",
        )
        has_overlay = True

        valid_std = std is not None and not math.isnan(std)
        if valid_std:
            ax.axvspan(
                mean - std,
                mean + std,
                alpha=0.1,
                color=mean_color,
                label=f"±1 Std: {std:.3g}",
            )

    # Add legend if any reference lines or shaded regions are present
    if has_overlay:
        ax.legend(fontsize=legend_fontsize)

    # Perform layout adjustment if the figure was created by this function
    if owns_figure:
        fig.tight_layout()

    return fig


def plot_stripplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
    hue: str | None = None,
    order: list | None = None,
    hue_order: list | None = None,
    hue_legend: bool = False,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (8, 5),
    palette: str = "tab10",
    size: int = 6,
    jitter: bool = True,
    median: float | None = None,
    median_label: str = "Median",
    median_color: str = "black",
    median_linestyle: str = "--",
    median_linewidth: float = 1.5,
    mean: float | None = None,
    mean_label: str = "Mean",
    mean_color: str = "grey",
    mean_linestyle: str = ":",
    mean_linewidth: float = 1.5,
    x_means: dict[str, float] | None = None,
    x_means_label: str = "Mean",
    x_means_linewidth: float = 3,
    x_means_color: str = "black",
    ylim: tuple[float, float] | None = None,
    rotation: float = 0,
    tick_fontsize: int = 10,
    legend_fontsize: int = 12,
    label_fontsize: int = 12,
    title_fontsize: int = 14,
    legend_line_factor: float = 0.75,
    num_models: int | None = 10,
    seed: int = 42,
) -> Figure:
    """Plot a categorical strip plot using seaborn.

    Each category on the x-axis is represented by a column of dots, where
    every dot is a single observation. Dots are optionally jittered
    horizontally to reduce overplotting and color-coded by the ``hue``
    column. Three kinds of reference lines are supported: a dashed horizontal
    line at the median, a dotted horizontal line at the mean (both spanning
    the full width), and short horizontal tick-marks at a per-category value
    (``x_means``).

    Args:
        data: DataFrame in long format with at least the ``x`` and ``y``
            columns present.
        x: Column name for the categorical x-axis groups.
        y: Column name for the numeric y-axis values.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        title: Title of the plot.
        hue: Column name used to color-code dots. Defaults to ``x`` if None,
            so each category gets a distinct color.
        order: Display order of x-axis categories. Inferred from the data
            if None.
        hue_order: Display order of hue categories, used to match palette
            colors to levels. When ``hue`` equals ``x`` this defaults to
            ``order``; otherwise it is inferred by sorting the unique values
            of the hue column.
        hue_legend: If True, adds one dot handle per hue category to the
            legend so the color mapping is visible. Useful when ``hue``
            differs from ``x``.
        ax: Axes to draw into. If None, a new figure and axes are created.
        figsize: Figure size as ``(width, height)``. Only used when ``ax``
            is None. Default is ``(8, 5)``.
        palette: Seaborn palette name used to color the dots.
        size: Marker radius in points for each dot.
        jitter: Whether to add horizontal jitter to separate overlapping dots.
        median: If provided and not NaN, draws a horizontal dashed reference
            line at this value spanning the full plot width.
        median_label: Legend label for the median reference line.
        median_linewidth: Line width of the median reference line.
        mean: If provided and not NaN, draws a horizontal dotted reference
            line at this value spanning the full plot width.
        mean_label: Legend label for the mean reference line.
        mean_linewidth: Line width of the mean reference line.
        x_means: Mapping from each x-axis category name to a reference value.
            For each entry a short horizontal line is drawn across that
            category's column, making it easy to compare dots to a
            per-category baseline.
        x_means_label: Legend label for the per-category reference lines.
        x_means_linewidth: Line width of the per-category reference lines.
        x_means_color: Color of the per-category reference lines.
        ylim: ``(min, max)`` limits for the y-axis. Inferred from data
            if None.
        rotation: Rotation angle in degrees for x-axis tick labels.
        tick_fontsize: Font size for axis tick labels.
        label_fontsize: Font size for axis labels.
        title_fontsize: Font size for the plot title.
        num_models: Number of models to sample. If None, all models are used.
        seed: Seed for the random number generator.

    Returns:
        The matplotlib Figure containing the strip plot.
    """
    if owns_figure := (ax is None):
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Determines which column is used to color-code the dots
    hue_col = hue if hue is not None else x

    # Randomly sample a subset of models (identified by hue column) if requested
    if num_models is not None:
        all_models = sorted(data[hue_col].unique())
        if len(all_models) > num_models:
            rng = random.Random(seed)
            sampled = set(rng.sample(all_models, num_models))
            data = data[data[hue_col].isin(sampled)]
            if order is not None:
                order = [v for v in order if v in sampled]
            if hue_order is not None:
                hue_order = [v for v in hue_order if v in sampled]

    # Controls the left-to-right order of categories on the x-axis
    display_order = order if order is not None else sorted(data[x].unique())

    # Controls which color from the palette is used for each strip in the plot
    default_hue_order = (
        display_order if hue_col == x else sorted(data[hue_col].unique())
    )
    display_hue_order = hue_order if hue_order is not None else default_hue_order

    sns.stripplot(
        data=data,
        x=x,
        y=y,
        hue=hue_col,
        order=display_order,
        hue_order=display_hue_order,
        ax=ax,
        palette=palette,
        size=size,
        jitter=jitter,
        legend=False,
    )

    # Set y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(*ylim)

    # Set title, x-axis label, y-axis label, and font sizes
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    # Set font size and rotation for ticks
    plt.setp(
        ax.get_xticklabels(),
        rotation=rotation,
        ha="right" if rotation > 0 else "center",
        fontsize=tick_fontsize,
    )
    plt.setp(
        ax.get_yticklabels(),
        fontsize=tick_fontsize,
    )

    legend_handles = []

    # Manually add legend handles for each hue category if requested
    # NOTE: We do this because Seaborn's automatic legends can be inconsistent
    # or include unwanted entries
    if hue_legend:
        palette_colors = sns.color_palette(
            palette,
            n_colors=len(display_hue_order),
        )
        for color, label in zip(palette_colors, display_hue_order):
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=size,
                    label=label,
                )
            )

    # Optionally plot reference line for global median
    valid_median = median is not None and not math.isnan(median)
    if valid_median:
        ax.axhline(
            median,
            color=median_color,
            linestyle=median_linestyle,
            linewidth=median_linewidth,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=median_color,
                linestyle=median_linestyle,
                linewidth=median_linewidth * legend_line_factor,
                label=f"{median_label}: {median:.3g}",
            )
        )

    # Optionally plot reference line for global mean
    valid_mean = mean is not None and not math.isnan(mean)
    if valid_mean:
        ax.axhline(
            mean,
            color=mean_color,
            linestyle=mean_linestyle,
            linewidth=mean_linewidth,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=mean_color,
                linestyle=mean_linestyle,
                linewidth=mean_linewidth * legend_line_factor,
                label=f"{mean_label}: {mean:.3g}",
            )
        )

    # Optionally plot reference lines for per-category means
    valid_x_means = x_means is not None and all(
        not math.isnan(x) for x in x_means.values()
    )
    if valid_x_means:
        for i, category in enumerate(display_order):
            if category in x_means:
                ax.hlines(
                    x_means[category],
                    xmin=i - 0.3,
                    xmax=i + 0.3,
                    colors=x_means_color,
                    linewidth=x_means_linewidth,
                )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=x_means_color,
                linewidth=x_means_linewidth * legend_line_factor,
                label=x_means_label,
            )
        )

    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=legend_fontsize)

    if owns_figure:
        fig.tight_layout()
    return fig


def plot_capability_tree(
    dataset: Dataset,
    root: dict,
    model: str | None = None,
    num_levels: int = 2,
    max_characters: int = 120,
) -> "plt.Figure":
    """Display the first num_levels levels of a capability tree as a styled diagram.

    Renders each node as a dark-blue rounded box showing the capability description,
    instance count, and (optionally) model score and CI. Style mirrors the EvalTree
    web demo.

    Args:
        dataset: The dataset to plot the capability tree for.
        root: The root node of the capability tree (from load_capability_tree).
        model: The model name whose score to display at each node. If None, model
            performance and CI are not shown.
        num_levels: Maximum depth to display (root = level 1).
        max_characters: Maximum characters of the capability description to wrap
            inside each node.

    Returns:
        The matplotlib Figure.
    """
    # ── colours ───────────────────────────────────────────────────────────────
    C_FIG = "#dce4f0"  # figure background
    C_NODE = "#2d4e8c"  # node fill
    C_BORDER = "#3d6aad"  # node border
    C_TITLE = "#f5a623"  # distinction text for non-root nodes (orange)
    C_BODY = "#ffffff"  # capability description text (white)
    C_INST_BG = "#1a3575"  # instance count pill background
    C_PERF_BG = "#b055c0"  # accuracy / CI pill background (purple)
    C_PILL_TXT = "#ffffff"  # all pill text (white)
    C_EDGE = "#7ab0d4"  # bezier edge colour

    # ── geometry ──────────────────────────────────────────────────────────────
    show_model = model is not None
    X_GAP = 1.5
    NW = 1.3
    NH = 2.0 if show_model else 1.6
    Y_GAP = NH + 1.2
    PILL_H = 0.18  # height of score / CI pills
    INST_H = 0.17  # height of instance count pill
    LINE_H = NH * 0.055  # line pitch
    WRAP_W = 36

    # ── helpers ───────────────────────────────────────────────────────────────
    def _score(node: dict) -> float | None:
        if not show_model:
            return None
        return next((s for m, s in node.get("ranking", []) if m == model), None)

    def _ci(node: dict) -> tuple | None:
        if not show_model:
            return None
        ci = node.get("CI", {})
        return tuple(ci[model]) if isinstance(ci, dict) and model in ci else None

    # ── step 1: collect nodes ─────────────────────────────────────────────────
    nodes: list[dict] = []
    children: dict[int, list[int]] = {}

    def _collect(node: dict, depth: int, parent: int | None) -> None:
        idx = len(nodes)
        cap = textwrap.shorten(
            node.get("capability", ""), width=max_characters, placeholder="…"
        )
        cap_lines = textwrap.wrap(cap, width=WRAP_W)[:3]
        distinction = node.get("distinction")
        dist_lines = textwrap.wrap(distinction, width=WRAP_W)[:2] if distinction else []
        nodes.append(
            {
                "cap_lines": cap_lines,
                "dist_lines": dist_lines,
                "size": node.get("size", "?"),
                "score": _score(node),
                "ci": _ci(node),
                "depth": depth,
                "parent": parent,
                "x": 0.0,
            }
        )
        children[idx] = []
        if parent is not None:
            children[parent].append(idx)
        if depth < num_levels:
            subtrees = node.get("subtrees", [])
            if isinstance(subtrees, list):
                for child in subtrees:
                    _collect(child, depth + 1, idx)

    _collect(root, 1, None)

    # ── step 2: assign x positions (centred over leaves) ──────────────────────
    leaf_x = [0]

    def _assign_x(idx: int) -> None:
        kids = children[idx]
        if not kids:
            nodes[idx]["x"] = float(leaf_x[0]) * X_GAP
            leaf_x[0] += 1
        else:
            for k in kids:
                _assign_x(k)
            nodes[idx]["x"] = sum(nodes[k]["x"] for k in kids) / len(kids)

    _assign_x(0)
    n_leaves = leaf_x[0]

    # ── step 3: build figure ──────────────────────────────────────────────────
    fig_w = max(12, n_leaves * 3.4)
    fig_h = max(5, num_levels * Y_GAP + NH + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(C_FIG)
    ax.set_facecolor(C_FIG)
    x_pad = NW * 0.6 + 0.2
    ax.set_xlim(-x_pad, (n_leaves - 1) * X_GAP + x_pad)
    ax.set_ylim(-(num_levels - 1) * Y_GAP - NH * 0.6, NH * 0.6)
    ax.axis("off")

    # ── step 4: curved bezier edges (drawn before nodes) ──────────────────────
    for idx, node in enumerate(nodes):
        if node["parent"] is None:
            continue
        px = nodes[node["parent"]]["x"]
        py = -(nodes[node["parent"]]["depth"] - 1) * Y_GAP
        cx_ = node["x"]
        cy = -(node["depth"] - 1) * Y_GAP
        y0, y1 = py - NH / 2, cy + NH / 2
        ym = (y0 + y1) / 2
        verts = [(px, y0), (px, ym), (cx_, ym), (cx_, y1)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        ax.add_patch(
            PathPatch(
                Path(verts, codes),
                facecolor="none",
                edgecolor=C_EDGE,
                lw=1.4,
                zorder=1,
            )
        )

    # ── step 5: draw nodes ────────────────────────────────────────────────────
    PILL_MARGIN = 0.03
    pw = NW - 2 * PILL_MARGIN  # full-width pills

    for idx, node in enumerate(nodes):
        x = node["x"]
        y = -(node["depth"] - 1) * Y_GAP
        xL = x - NW / 2
        yB = y - NH / 2

        # Background box (uniform border on all nodes)
        ax.add_patch(
            FancyBboxPatch(
                (xL, yB),
                NW,
                NH,
                boxstyle="round,pad=0.04",
                facecolor=C_NODE,
                edgecolor=C_BORDER,
                linewidth=1.0,
                zorder=2,
            )
        )

        dist_lines = node["dist_lines"]
        cap_lines = node["cap_lines"]

        # Text: distinction (orange, bold) then capability (white), stacked from top
        text_top = yB + NH * 0.93
        row = 0.0
        metric = Dataset(dataset).metric

        for line in dist_lines:
            ax.text(
                x,
                text_top - row * LINE_H,
                line,
                ha="center",
                va="top",
                fontsize=11,
                fontweight="bold",
                color=C_TITLE,
                zorder=3,
            )
            row += 1.0
        if dist_lines:
            row += 0.1  # small gap between title and description

        for line in cap_lines:
            ax.text(
                x,
                text_top - row * LINE_H,
                line,
                ha="center",
                va="top",
                fontsize=10,
                fontweight="normal",
                color=C_BODY,
                zorder=3,
            )
            row += 1.0

        # Instance pill: dynamically positioned just below text
        text_end_y = text_top - (row + 0.5) * LINE_H
        inst_y = text_end_y - INST_H / 2
        inst_y_min = yB + NH * (0.24 if show_model else 0.10)
        inst_y_max = yB + NH * 0.82
        inst_y = max(inst_y_min, min(inst_y, inst_y_max))

        ax.add_patch(
            FancyBboxPatch(
                (x - pw / 2, inst_y - INST_H / 2),
                pw,
                INST_H,
                boxstyle="round,pad=0.01",
                facecolor=C_INST_BG,
                edgecolor="none",
                zorder=3,
            )
        )
        ax.text(
            x,
            inst_y,
            f"{node['size']} instances",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=C_PILL_TXT,
            zorder=4,
        )

        # Score + CI combined into one full-width pill, just below instance pill
        if node["score"] is not None or node["ci"] is not None:
            perf_y = inst_y - INST_H / 2 - PILL_H / 2 - 0.04
            score_str = (
                f"{node['score']:.3f} {metric.capitalize()}"
                if node["score"] is not None
                else ""
            )
            ci_str = (
                f"95% CI: [{node['ci'][0]:.3f}, {node['ci'][1]:.3f}]"
                if node["ci"] is not None
                else ""
            )
            perf_label = "   ".join(filter(None, [score_str, ci_str]))
            ax.add_patch(
                FancyBboxPatch(
                    (x - pw / 2, perf_y - PILL_H / 2),
                    pw,
                    PILL_H,
                    boxstyle="round,pad=0.01",
                    facecolor=C_PERF_BG,
                    edgecolor="none",
                    zorder=3,
                )
            )
            ax.text(
                x,
                perf_y,
                perf_label,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=C_PILL_TXT,
                zorder=4,
            )

    title = (
        f"{dataset.pretty_name} Capability Tree: {model}"
        if show_model
        else f"{dataset.pretty_name} Capability Tree"
    )
    fig.suptitle(title, fontsize=12, color="#333333", y=0.99)
    plt.tight_layout()
    plt.close(fig)
    return fig
