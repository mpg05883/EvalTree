import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure


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
    """
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    data_min, data_max = xlim if xlim is not None else (data.min(), data.max())
    sns.histplot(data=data, ax=ax, bins=10, binrange=(data_min, data_max))
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
            color="red",
            linestyle="--",
            label=f"{mean_label}: {mean:.3g}",
        )
        if std is not None and not math.isnan(std):
            ax.axvspan(
                mean - std,
                mean + std,
                alpha=0.2,
                color="red",
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
