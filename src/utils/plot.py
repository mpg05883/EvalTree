import colorsys
import math
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


def _complementary_color(color: str) -> tuple:
    """Return a color roughly 120° away on the HSV wheel (triadic offset).

    A 120° rotation mirrors the blue-red relationship — somewhat opposite but
    not fully complementary (which would be 180°).
    """
    r, g, b = plt.matplotlib.colors.to_rgb(color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + 1 / 3) % 1.0
    return colorsys.hsv_to_rgb(h, s, v)


def plot_histogram(
    data: pd.Series | np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (8, 4),
    annotate: bool = False,
    mean: float | None = None,
    mean_label: str = "Mean",
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
        mean (float | None): If provided and not NaN, plots a vertical line at this value.
        std (float | None): If provided and not NaN, plots a shaded region for mean ± std.
            Requires mean to also be provided and not NaN.
        xlim (tuple[float, float] | None): (min, max) range for the bins. If None,
            the range is inferred from the data.
        ylim (tuple[float, float] | None): (min, max) limits for the y-axis. If None,
            the limits are inferred from the data.
        color (str | None): Bar color. If provided, the mean/std overlay uses the
            complementary color (opposite on the color wheel). If None, uses seaborn's
            default color for the bars and red for the mean/std overlay.
    """
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    mean_color = _complementary_color(color) if color is not None else "red"

    data_min, data_max = xlim if xlim is not None else (data.min(), data.max())
    sns.histplot(data=data, ax=ax, bins=10, binrange=(data_min, data_max), color=color)
    ax.set_xlim(data_min, data_max)

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
            )

    # Optionally plot vertical line at mean and a shaded region for ±1 std
    if mean is not None and not math.isnan(mean):
        ax.axvline(
            mean,
            color=mean_color,
            linestyle="--",
            label=f"{mean_label}: {mean:.3g}",
        )
        if std is not None and not math.isnan(std):
            ax.axvspan(
                mean - std,
                mean + std,
                alpha=0.2,
                color=mean_color,
                label=f"±1 Std: {std:.3g}",
            )
        ax.legend()

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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
    mean: float | None = None,
    mean_label: str = "Mean",
    x_means: dict[str, float] | None = None,
    x_means_label: str = "Mean",
    x_means_linewidth: float = 3,
    x_means_color: str = "black",
    ylim: tuple[float, float] | None = None,
    rotation: float = 0,
    tick_fontsize: int = 11,
    label_fontsize: int = 13,
    title_fontsize: int = 14,
) -> Figure:
    """Plot a categorical strip plot using seaborn.

    Each category on the x-axis is represented by a column of dots, where
    every dot is a single observation. Dots are optionally jittered
    horizontally to reduce overplotting and color-coded by the ``hue``
    column. Two kinds of reference lines are supported: a single horizontal
    dashed line spanning the full width (``mean``), and short horizontal
    tick-marks drawn at a per-category value (``x_means``).

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
        mean: If provided and not NaN, draws a horizontal dashed reference
            line at this value spanning the full plot width.
        mean_label: Legend label for the full-width reference line.
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

    Returns:
        The matplotlib Figure containing the strip plot.
    """
    owns_figure = ax is None
    if owns_figure:
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    hue_col = hue if hue is not None else x
    resolved_order = order if order is not None else sorted(data[x].unique())
    resolved_hue_order = (
        hue_order
        if hue_order is not None
        else (resolved_order if hue_col == x else sorted(data[hue_col].unique()))
    )
    sns.stripplot(
        data=data,
        x=x,
        y=y,
        hue=hue_col,
        order=resolved_order,
        hue_order=resolved_hue_order,
        ax=ax,
        palette=palette,
        size=size,
        jitter=jitter,
        legend=False,
    )

    legend_handles = []

    if hue_legend:
        palette_colors = sns.color_palette(palette, n_colors=len(resolved_hue_order))
        for color, label in zip(palette_colors, resolved_hue_order):
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

    if mean is not None and not math.isnan(mean):
        ax.axhline(mean, color="red", linestyle="--")
        legend_handles.append(
            Line2D([0], [0], color="red", linestyle="--", label=f"{mean_label}: {mean:.3g}")
        )

    if x_means is not None:
        for i, category in enumerate(resolved_order):
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
                linewidth=x_means_linewidth * 0.75,
                label=x_means_label,
            )
        )

    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=tick_fontsize)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    plt.setp(
        ax.get_xticklabels(),
        rotation=rotation,
        ha="right" if rotation > 0 else "center",
        fontsize=tick_fontsize,
    )
    plt.setp(ax.get_yticklabels(), fontsize=tick_fontsize)

    if owns_figure:
        fig.tight_layout()
    return fig


def plot_barplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
    ax: Axes | None = None,
    figsize: tuple[int, int] | None = None,
    annotate: bool = False,
    mean: float | None = None,
    mean_label: str = "Mean",
    std: float | None = None,
    edgecolor: str = "black",
    linewidth: float = 1,
    rotation: int = 0,
    tick_fontsize: int = 12,
    annotation_fontsize: int = 12,
    legend_fontsize: int = 14,
    label_fontsize: int = 14,
    title_fontsize: int = 16,
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """Plot a bar chart using seaborn.

    Args:
        data (pd.DataFrame): DataFrame containing the data to plot.
        x (str): Column name to use for the x-axis categories.
        y (str): Column name to use for the y-axis values.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        ax (Axes | None): Axes to draw into. If None, a new figure is created.
        figsize: Figure size as (width, height). Only used when ax is None. Default is (8, 5).
        annotate (bool): If True, annotates each bar with its value in bold.
        mean (float | None): If provided and not NaN, plots a horizontal line at this value.
        mean_label (str): Label for the mean line in the legend. Default is "Mean".
        std (float | None): If provided and not NaN, plots a shaded region for mean ± std.
            Requires mean to also be provided and not NaN.
        edgecolor (str): Bar edge color. Default is "black".
        linewidth (float): Bar edge line width. Default is 1.
        rotation (int): Rotation angle for x-axis tick labels. Default is 0.
        ylim (tuple[float, float] | None): (min, max) limits for the y-axis. If None,
            the limits are inferred from the data.
    """
    owns_figure = ax is None
    xsize = len(data[x].unique())
    figsize = figsize if figsize is not None else (max(6, xsize * 1.5), 6)
    if owns_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.barplot(
        data=data,
        x=x,
        y=y,
        ax=ax,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

    if annotate:
        ax.bar_label(
            ax.containers[0],
            fontweight="bold",
            fontsize=annotation_fontsize,
        )

    # Optionally plot horizontal line at mean and a shaded region for ±1 std
    if mean is not None and not math.isnan(mean):
        ax.axhline(
            mean,
            color="red",
            linestyle="--",
            label=f"{mean_label}: {int(mean)}",
        )
        if std is not None and not math.isnan(std):
            ax.axhspan(
                mean - std,
                mean + std,
                alpha=0.2,
                color="red",
                label=f"±1 Std: {std:.3g}",
            )
        ax.legend(fontsize=legend_fontsize)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis="x", rotation=rotation, labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
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
