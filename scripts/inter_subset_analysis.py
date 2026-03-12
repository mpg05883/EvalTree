import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
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
from src.utils.plot import plot_histogram, plot_stripplot

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def subset_external_agreement_analysis(
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
        "desc": "Computing agreement with full benchmark",
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
        tau, _ = kendalltau(global_scores, subset_scores)
        results.append(
            {
                "subset": subset,
                "scores": subset_scores,
                "kendall_tau": tau,
                "num_instances": len(model_scores_df[mask]),
            }
        )

    return pd.DataFrame(results)


def plot_subset_external_agreement_histogram(
    df: pd.DataFrame,
    dataset: Dataset,
    num_models: int,
    num_subsets: int,
    **kwargs,
) -> plt.Figure:
    """Plot the distribution of Kendall's Tau values across subsets as a histogram.

    Takes the output of subset_external_agreement_analysis and visualises how consistently
    each subset reproduces the benchmark-level model ranking. The x-axis shows
    Kendall's Tau (ranging from -1 to 1 or 0 to 1 depending on the data), and
    the histogram is annotated with the median and IQR across all subsets.
    """
    data = df["kendall_tau"]
    xlabel = r"Kendall's $\tau$"
    ylabel = f"Number of {dataset.plural.title()}"
    title = (
        f"{dataset.pretty_name}: {dataset.subset_col.title()} Agreement with Full Benchmark"
        f"\n({num_models} models, {num_subsets} {dataset.plural})"
    )
    annotate = True
    median = data.median()
    median_label = f"Median {xlabel}"
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
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
        median=median,
        median_label=median_label,
        q1=q1,
        q3=q3,
        xlim=xlim,
        ylim=ylim,
    )


def subset_performance_analysis(
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


def plot_subset_performance_stripplot(
    dataset: Dataset,
    model_scores_df: pd.DataFrame,
    subset_scores_df: pd.DataFrame,
    num_models: int,
    num_subsets: int,
    **kwargs,
) -> plt.Figure:
    """Visualize per-subset accuracy for each model as a vertical strip plot.

    Renders one column per model on the x-axis, with each dot representing a
    single subset's mean accuracy on the y-axis. A short horizontal black line
    marks the benchmark-level (global) mean accuracy for each model, making it
    easy to see which subsets fall above or below the overall benchmark score.

    Args:
        dataset: The dataset being analysed, used for axis labels and title.
        model_scores_df: Full benchmark model scores (instances × models),
            used to compute the per-model global mean reference lines.
        subset_scores_df: Per-subset mean model scores (subsets × models),
            as returned by :func:`subset_performance_analysis`.
        num_models: Number of models, used for the figure width.
        num_subsets: Number of subsets, used for the plot title.

    Returns:
        The matplotlib Figure containing the strip plot.
    """
    models = subset_scores_df.columns.tolist()
    global_means = model_scores_df.mean().to_dict()

    long_df = subset_scores_df.reset_index().melt(
        id_vars="subset",
        var_name="model",
        value_name="score",
    )

    x = "model"
    y = "score"
    xlabel = ""
    ylabel = dataset.metric.title()
    title = (
        f"{dataset.pretty_name}: {dataset.subset_col.title()} "
        f"{dataset.metric.title()} vs Full Benchmark"
        f"\n({num_subsets} {dataset.plural})"
    )
    hue = "model"
    order = models
    palette = "tab10"
    size = 10
    x_means = global_means
    x_means_label = f"Full Benchmark {dataset.metric.title()}"
    figsize = (max(8, num_models * 1.5), 5)
    ylim = (0, 1)
    rotation = 30

    return plot_stripplot(
        data=long_df,
        x=x,
        y=y,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        hue=hue,
        order=order,
        palette=palette,
        size=size,
        x_means=x_means,
        x_means_label=x_means_label,
        figsize=figsize,
        ylim=ylim,
        rotation=rotation,
    )


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

    subset_external_agreement_df = subset_external_agreement_analysis(**shared)
    data_name = f"subset__external_agreement__{num_models=}"
    data_path = build_data_path(dataset, experiment, data_name)
    subset_external_agreement_df.to_csv(data_path, index=False)
    logger.info(f"Saved data to {data_path}")

    subset_external_agreement_fig = plot_subset_external_agreement_histogram(
        subset_external_agreement_df,
        **shared,
    )
    plot_name = f"subset__external_agreement__histogram__{num_models=}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    subset_external_agreement_fig.savefig(plot_path)
    plt.close(subset_external_agreement_fig)
    logger.info(f"Saved plot to {plot_path}")

    subset_scores_df = subset_performance_analysis(**shared)
    data_name = "subset__performance"
    data_path = build_data_path(dataset, experiment, data_name)
    subset_scores_df.to_csv(data_path, index=False)
    logger.info(f"Saved data to {data_path}")

    subset_performance_fig = plot_subset_performance_stripplot(
        subset_scores_df=subset_scores_df,
        **shared,
    )
    plot_name = "subset__performance__stripplot"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    subset_performance_fig.savefig(plot_path)
    plt.close(subset_performance_fig)
    logger.info(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    # Only MATH, MMLU, and DS-1000 have pre-defined subsets
    datasets = [Dataset.MATH, Dataset.MMLU, Dataset.DS_1000]
    experiment = Path(__file__).stem
    num_datasets = len(datasets)

    for i, dataset in enumerate(datasets):
        print(f"{'-'*80} Dataset {i+1}/{num_datasets}: {dataset.pretty_name} {'-'*80}")
        main(dataset, experiment)
