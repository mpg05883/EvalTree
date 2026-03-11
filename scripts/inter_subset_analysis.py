from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import kendalltau
from tqdm import tqdm

from src.utils.data import (
    get_metadata_mask,
    get_unique_metadata_values,
    load_dataset,
)
from src.utils.enums import Dataset
from src.utils.model import load_model_scores
from src.utils.path import build_data_path, build_plot_path
from src.utils.plot import plot_histogram


def subset_agreement_analysis(
    dataset: Dataset,
    dataset_df: pd.DataFrame,
    model_scores_df: pd.DataFrame,
    subsets: list[str],
    **kwargs,
) -> pd.DataFrame:
    """Assess how faithfully each subset preserves the benchmark-level model ranking.

    For each subset, computes Kendall's Tau between the global model ranking (derived
    from mean scores across the full benchmark) and the subset-level ranking.
    - A high tau indicates the subset reproduces the same relative ordering of models as the
    full benchmark.
    - A low or negative tau suggests the subset yields a different ordering.
    Results are saved as a histogram showing the distribution of tau values across all subsets.
    """
    global_scores = model_scores_df.mean()

    kwargs = {
        "desc": "Computing Kendall's taus",
        "total": len(subsets),
        "unit": dataset.plural,
    }

    results = []

    for subset in tqdm(subsets, **kwargs):
        mask = (
            get_metadata_mask(dataset_df, dataset.subset_col, subset)
            if dataset == Dataset.DS_1000
            else dataset_df[dataset.subset_col] == subset
        )
        subset_scores = model_scores_df[mask].mean()
        kendall_tau, _ = kendalltau(global_scores, subset_scores)
        results.append(
            {
                "subset": subset,
                "scores": subset_scores,
                "kendall_tau": kendall_tau,
                "num_instances": len(model_scores_df[mask]),
            }
        )

    return pd.DataFrame(results)


def plot_subset_agreement_histogram(
    df: pd.DataFrame,
    dataset: Dataset,
    num_models: int,
    num_subsets: int,
    **kwargs,
) -> plt.Figure:
    """Plot the distribution of Kendall's Tau values across subsets as a histogram.

    Takes the output of subset_agreement_analysis and visualises how consistently
    each subset reproduces the benchmark-level model ranking. The x-axis shows
    Kendall's Tau (ranging from -1 to 1 or 0 to 1 depending on the data), and
    the histogram is annotated with the mean and standard deviation across all subsets.
    """
    data = df["kendall_tau"]
    xlabel = r"Kendall's $\tau$"
    ylabel = f"Number of {dataset.plural.title()}"
    title = (
        f"{dataset.pretty_name}: {dataset.subset_col.title()} Agreement with Full Benchmark"
        f"\n({num_models} models, {num_subsets} {dataset.plural})"
    )
    annotate = True
    mean = data.mean()
    mean_label = f"Mean {xlabel}"
    std = data.std()
    xlim = (-1, 1) if min(data) < 0 else (0, 1)
    ylim = {
        Dataset.DS_1000: (0, num_subsets),
        Dataset.MATH: (0, num_subsets),
        Dataset.MMLU: (0, num_subsets // 2),
    }[dataset]

    return plot_histogram(
        data,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        annotate=annotate,
        mean=mean,
        mean_label=mean_label,
        std=std,
        xlim=xlim,
        ylim=ylim,
    )


def subset_model_performance_analysis(
    dataset: Dataset,
    dataset_df: pd.DataFrame,
    model_scores_df: pd.DataFrame,
    subsets: list[str],
    **kwargs,
) -> None:
    """Compute model performance across subsets.

    For each model, computes the mean score on every subset and plots the distribution
    of those per-subset means as a histogram. A vertical reference line marks the
    benchmark-level mean, making it easy to see which subsets are easier or harder
    than average. Results are saved as a multi-panel figure with one panel per model.
    """
    subset_to_scores = {}
    for subset in subsets:
        mask = (
            get_metadata_mask(dataset_df, dataset.subset_col, subset)
            if dataset == Dataset.DS_1000
            else dataset_df[dataset.subset_col] == subset
        )
        subset_to_scores[subset] = model_scores_df[mask].mean()
    df = pd.DataFrame(subset_to_scores).T
    df.index.name = "subset"
    return df


def plot_subset_model_performance_strip_plot(
    dataset: Dataset,
    model_scores_df: pd.DataFrame,
    subset_scores_df: pd.DataFrame,
    num_models: int,
    num_subsets: int,
    palette: str = "tab10",
    size: int = 10,
    linewidth: float = 3,
    rotation: float = 30,
    axis_fontsize: float = 12,
    label_fontsize: float = 14,
    title_fontsize: float = 16,
    **kwargs,
) -> plt.Figure:
    """Visualize per-subset accuracy for each model as a vertical strip plot.

    Renders one column per model on the x-axis, with each dot representing a single
    subset's mean accuracy on the y-axis. Each model's dots and benchmark line share
    the same color from the tab10 palette. A short horizontal line marks the
    benchmark-level (global) mean accuracy for each model, making it easy to see
    which subsets fall above or below the overall benchmark score.
    """
    models = subset_scores_df.columns.tolist()
    global_means = model_scores_df.mean()
    colors = sns.color_palette(palette, n_colors=num_models)

    # Melt to long format: one row per (subset, model) pair
    long_df = subset_scores_df.reset_index().melt(
        id_vars="subset",
        var_name="model",
        value_name="score",
    )

    fig, ax = plt.subplots(
        figsize=(
            max(8, num_models * 1.5),
            5,
        )
    )

    sns.stripplot(
        data=long_df,
        x="model",
        y="score",
        hue="model",
        palette=colors,
        order=models,
        hue_order=models,
        ax=ax,
        jitter=True,
        size=size,
        legend=False,
    )

    # Draw a short horizontal line at the benchmark-level mean for each model column,
    # using black
    for i, model in enumerate(models):
        ax.hlines(
            global_means[model],
            xmin=i - 0.3,
            xmax=i + 0.3,
            colors="black",
            linewidth=linewidth,
        )

    ax.set_xlabel(None, fontsize=label_fontsize)
    ax.set_ylabel(dataset.metric.title(), fontsize=label_fontsize)
    ax.set_ylim(0, 1)
    plt.setp(
        ax.get_xticklabels(),
        rotation=rotation,
        ha="right",
        fontsize=axis_fontsize,
    )
    plt.setp(
        ax.get_yticklabels(),
        fontsize=axis_fontsize,
    )

    legend_handle = Line2D(
        [0],
        [0],
        color="black",
        linewidth=linewidth * 0.75,
        label=f"Benchmark-Level {dataset.metric}",
    )
    ax.legend(
        handles=[legend_handle],
        fontsize=axis_fontsize,
    )

    ax.set_title(
        f"{dataset.pretty_name}: {dataset.subset_col.title()} "
        f"{dataset.metric.title()} vs Full Benchmark"
        f"\n({num_subsets} {dataset.plural})",
        fontsize=title_fontsize,
    )

    plt.tight_layout()
    return fig


def main(dataset: Dataset, experiment: str) -> None:
    dataset_df = load_dataset(dataset)
    model_scores_df = load_model_scores(dataset)
    assert len(dataset_df) == len(model_scores_df)

    subsets = (
        get_unique_metadata_values(dataset_df, dataset.subset_col)
        if dataset == Dataset.DS_1000
        else dataset_df[dataset.subset_col].unique()
    )

    num_subsets = len(subsets)
    num_models = len(model_scores_df.columns)
    num_instances = len(dataset_df)

    shared = dict(
        dataset=dataset,
        dataset_df=dataset_df,
        model_scores_df=model_scores_df,
        subsets=subsets,
        num_subsets=num_subsets,
        num_models=num_models,
        num_instances=num_instances,
        experiment=experiment,
    )

    subset_agreement_df = subset_agreement_analysis(**shared)
    data_name = f"subset_agreement_num_models={num_models}"
    data_path = build_data_path(dataset, experiment, data_name)
    subset_agreement_df.to_csv(data_path, index=False)
    print(f"Saved data to {data_path}")

    subset_agreement_fig = plot_subset_agreement_histogram(
        subset_agreement_df, **shared
    )
    plot_name = f"subset_agreement_histogram_num_models={num_models}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    subset_agreement_fig.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

    subset_scores_df = subset_model_performance_analysis(**shared)
    data_name = "subset_model_performance"
    data_path = build_data_path(dataset, experiment, data_name)
    subset_scores_df.to_csv(data_path, index=False)
    print(f"Saved data to {data_path}")

    subset_model_performance_fig = plot_subset_model_performance_strip_plot(
        subset_scores_df=subset_scores_df,
        **shared,
    )
    plot_name = "subset_model_performance_strip_plot"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    subset_model_performance_fig.savefig(plot_path)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    # Only MATH, MMLU, and DS-1000 have pre-defined subsets
    datasets = [Dataset.MATH, Dataset.MMLU, Dataset.DS_1000]
    experiment = Path(__file__).stem

    for dataset in datasets:
        main(dataset, experiment)
