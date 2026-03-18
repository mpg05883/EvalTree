import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau
from tqdm import tqdm

from src.utils.data import (
    get_metadata_mask,
    get_unique_metadata_values,
    load_dataset,
)
from src.utils.enums import Dataset
from src.utils.metrics import kendallw
from src.utils.model import load_model_scores
from src.utils.path import build_data_path, build_plot_path
from src.utils.plot import plot_histogram

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def split_halves_analysis(
    dataset: Dataset,
    dataset_df: pd.DataFrame,
    model_scores_df: pd.DataFrame,
    subsets,
    num_trials: int,
    **kwargs,
) -> pd.DataFrame:
    """Measure intra-subset ranking consistency using split-halves Kendall's Tau.

    For each subset, randomly splits instances into two halves ``num_trials``
    times, computes mean model scores on each half, and measures Kendall's Tau
    between the two resulting rankings. A high mean tau indicates that any
    random half of the subset reliably reproduces the same model ordering as
    the other half, suggesting the subset is internally consistent.

    Args:
        dataset: The dataset being analysed, used for subset masking and labels.
        dataset_df: Full benchmark instance-level DataFrame.
        model_scores_df: Per-instance model scores (instances × models).
        subsets: Iterable of subset identifiers to analyse.
        num_trials: Number of random split-halves trials per subset.

    Returns:
        A DataFrame with one row per subset and columns
        ``["subset", "mean_kendall_tau", "std_kendall_tau", "num_instances"]``.
    """
    tqdm_kwargs = {
        "desc": "Computing split-halves Kendall's taus",
        "total": len(subsets),
        "unit": dataset.plural,
    }

    results = []
    for subset in tqdm(subsets, **tqdm_kwargs):
        mask = (
            get_metadata_mask(dataset_df, dataset.subset_col, subset)
            if dataset == Dataset.DS_1000
            else dataset_df[dataset.subset_col] == subset
        )
        subset_scores = model_scores_df[mask]
        subset_size = len(subset_scores)

        taus = []
        for trial in range(num_trials):
            rng = np.random.default_rng(trial)
            shuffled_indices = rng.permutation(subset_size)
            half = subset_size // 2
            scores_a = subset_scores.iloc[shuffled_indices[:half]].mean()
            scores_b = subset_scores.iloc[shuffled_indices[half:]].mean()
            tau, _ = kendalltau(scores_a, scores_b)
            taus.append(tau)

        results.append(
            {
                "subset": subset,
                "mean_kendall_tau": np.mean(taus),
                "std_kendall_tau": np.std(taus),
                "num_instances": subset_size,
            }
        )

    return pd.DataFrame(results)


def plot_split_halves_histogram(
    df: pd.DataFrame,
    dataset: Dataset,
    num_models: int,
    num_subsets: int,
    num_instances: int,
    num_trials: int,
    **kwargs,
) -> plt.Figure:
    """Plot the distribution of split-halves Kendall's Tau values as a histogram.

    Takes the output of :func:`split_halves_analysis` and visualises how
    consistently each subset reproduces the same model ranking across random
    splits. The x-axis spans (-1, 1) or (0, 1) depending on whether any
    negative tau values are present.

    Args:
        df: DataFrame returned by :func:`split_halves_analysis`.
        dataset: The dataset being analysed, used for the plot title.
        num_models: Number of models in the benchmark.
        num_subsets: Number of subsets analysed.
        num_instances: Total number of benchmark instances.
        num_trials: Number of split-halves trials used per subset.

    Returns:
        The matplotlib Figure containing the histogram.
    """
    data = df["mean_kendall_tau"]
    xlabel = r"Kendall's $\tau$"
    ylabel = f"Number of {dataset.plural.capitalize()}"
    title = (
        f"{dataset.pretty_name}: {dataset.subset_col.title()} Internal Agreement"
        f"\n({num_models} models, {num_subsets} {dataset.plural}, {num_trials} trials)"
    )
    annotate = True
    median = data.median()
    median_label = f"Median {xlabel}"
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    xlim = (-1, 1) if min(data) < 0 else (0, 1)
    ylim = {
        Dataset.BBH: (0, num_subsets),
        Dataset.DS_1000: (0, num_subsets),
        Dataset.GPQA_DIAMOND: (0, num_subsets),
        Dataset.MATH: (0, num_subsets),
        Dataset.MATH_LVL_5: (0, num_subsets),
        Dataset.MMLU: (0, num_subsets // 2),
        Dataset.MMLU_PRO: (0, num_subsets),
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


def bootstrap_analysis(
    dataset: Dataset,
    dataset_df: pd.DataFrame,
    model_scores_df: pd.DataFrame,
    subsets,
    num_trials: int,
    **kwargs,
) -> pd.DataFrame:
    """Measure intra-subset ranking stability using bootstrapped Kendall's Tau.

    For each subset, computes the reference model ranking from all instances
    in the subset, then draws ``num_trials`` bootstrap samples and measures
    Kendall's Tau between each bootstrap ranking and the reference. A high
    mean tau indicates that resampled subsets reliably recover the same model
    ordering, reflecting stable estimation within the subset.

    NOTE: Bootstrapping can produce upwardly biased estimates. See here for
    more details: https://stats.stackexchange.com/questions/96739/what-is-the-632-rule-in-bootstrapping

    Args:
        dataset: The dataset being analysed, used for subset masking and labels.
        dataset_df: Full benchmark instance-level DataFrame.
        model_scores_df: Per-instance model scores (instances × models).
        subsets: Iterable of subset identifiers to analyse.
        num_trials: Number of bootstrap trials per subset.

    Returns:
        A DataFrame with one row per subset and columns
        ``["subset", "mean_kendall_tau", "std_kendall_tau", "num_instances"]``.
    """
    tqdm_kwargs = {
        "desc": "Computing bootstrap Kendall's taus",
        "total": len(subsets),
        "unit": dataset.plural,
    }

    results = []
    for subset in tqdm(subsets, **tqdm_kwargs):
        mask = (
            get_metadata_mask(dataset_df, dataset.subset_col, subset)
            if dataset == Dataset.DS_1000
            else dataset_df[dataset.subset_col] == subset
        )
        subset_scores = model_scores_df[mask]
        n = len(subset_scores)
        reference_scores = subset_scores.mean()

        taus = []
        for b in range(num_trials):
            rng = np.random.default_rng(b)
            bootstrap_indices = rng.integers(0, n, size=n)
            bootstrap_scores = subset_scores.iloc[bootstrap_indices].mean()
            tau, _ = kendalltau(reference_scores, bootstrap_scores)
            taus.append(tau)

        results.append(
            {
                "subset": subset,
                "mean_kendall_tau": np.mean(taus),
                "std_kendall_tau": np.std(taus),
                "num_instances": n,
            }
        )

    return pd.DataFrame(results)


def plot_bootstrap_histogram(
    df: pd.DataFrame,
    dataset: Dataset,
    num_models: int,
    num_subsets: int,
    num_instances: int,
    num_trials: int,
    **kwargs,
) -> plt.Figure:
    """Plot the distribution of bootstrapped Kendall's Tau values as a histogram.

    Takes the output of :func:`bootstrap_analysis` and visualises how stably
    each subset's model ranking is estimated under resampling. The x-axis spans
    (-1, 1) or (0, 1) depending on whether any negative tau values are present.

    Args:
        df: DataFrame returned by :func:`bootstrap_analysis`.
        dataset: The dataset being analysed, used for the plot title.
        num_models: Number of models in the benchmark.
        num_subsets: Number of subsets analysed.
        num_instances: Total number of benchmark instances.
        num_trials: Number of bootstrap trials used per subset.

    Returns:
        The matplotlib Figure containing the histogram.
    """
    data = df["mean_kendall_tau"]
    xlabel = r"Kendall's $\tau$"
    ylabel = f"Number of {dataset.plural.capitalize()}"
    title = (
        f"{dataset.pretty_name}: {dataset.subset_col.title()} Internal Agreement (Bootstrap {xlabel})"
        f"\n({num_models} models, {num_subsets} {dataset.plural}, {num_trials} trials)"
    )
    annotate = True
    median = data.median()
    median_label = f"Median {xlabel}"
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    xlim = (-1, 1) if min(data) < 0 else (0, 1)
    ylim = {
        Dataset.BBH: (0, num_subsets),
        Dataset.GPQA_DIAMOND: (0, num_subsets),
        Dataset.MATH_LVL_5: (0, num_subsets),
        Dataset.MMLU_PRO: (0, num_subsets),
        Dataset.DS_1000: (0, num_subsets),
        Dataset.MATH: (0, num_subsets),
        Dataset.MMLU: (0, num_subsets),
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


def minibatch_analysis(
    dataset: Dataset,
    dataset_df: pd.DataFrame,
    model_scores_df: pd.DataFrame,
    subsets,
    num_trials: int,
    num_folds: int,
    **kwargs,
) -> pd.DataFrame:
    """Measure intra-subset ranking agreement across folds using Kendall's W.

    For each subset, randomly shuffles instances and divides them into
    ``num_folds`` equal folds ``num_trials`` times. Each trial computes
    Kendall's W across the fold-level model rankings, measuring how concordant
    the rankings are across folds. A W near 1 indicates strong agreement
    between folds; a W near 0 indicates near-random disagreement.

    Args:
        dataset: The dataset being analysed, used for subset masking and labels.
        dataset_df: Full benchmark instance-level DataFrame.
        model_scores_df: Per-instance model scores (instances × models).
        subsets: Iterable of subset identifiers to analyse.
        num_trials: Number of random shuffle trials per subset.
        num_folds: Number of folds to split instances into per trial.

    Returns:
        A DataFrame with one row per subset and columns
        ``["subset", "mean_kendallw", "std_kendallw", "num_instances"]``.
    """
    tqdm_kwargs = {
        "desc": "Computing mini-batch Kendall's Ws",
        "total": len(subsets),
        "unit": dataset.plural,
    }

    results = []
    for subset in tqdm(subsets, **tqdm_kwargs):
        mask = (
            get_metadata_mask(dataset_df, dataset.subset_col, subset)
            if dataset == Dataset.DS_1000
            else dataset_df[dataset.subset_col] == subset
        )
        subset_scores = model_scores_df[mask]
        subset_size = len(subset_scores)

        kendallw_values = []
        for trial in range(num_trials):
            rng = np.random.default_rng(trial)
            shuffled_indices = rng.permutation(subset_size)
            folds = np.array_split(shuffled_indices, num_folds)

            # Each row is one fold's mean model scores: shape (num_folds, num_models)
            fold_scores = np.array(
                [subset_scores.iloc[fold].mean().values for fold in folds]
            )
            kendallw_values.append(kendallw(fold_scores))

        results.append(
            {
                "subset": subset,
                "mean_kendallw": np.mean(kendallw_values),
                "std_kendallw": np.std(kendallw_values),
                "num_instances": subset_size,
            }
        )

    return pd.DataFrame(results)


def plot_minibatch_histogram(
    df: pd.DataFrame,
    dataset: Dataset,
    num_models: int,
    num_subsets: int,
    num_instances: int,
    num_trials: int,
    num_folds: int,
    **kwargs,
) -> plt.Figure:
    """Plot the distribution of mini-batch Kendall's W values as a histogram.

    Takes the output of :func:`minibatch_analysis` and visualises how
    concordant model rankings are across random folds within each subset.
    The x-axis always spans (0, 1) since Kendall's W is non-negative.

    Args:
        df: DataFrame returned by :func:`minibatch_analysis`.
        dataset: The dataset being analysed, used for the plot title.
        num_models: Number of models in the benchmark.
        num_subsets: Number of subsets analysed.
        num_instances: Total number of benchmark instances.
        num_trials: Number of shuffle trials used per subset.
        num_folds: Number of folds used per trial.

    Returns:
        The matplotlib Figure containing the histogram.
    """
    data = df["mean_kendallw"]
    xlabel = "Kendall's W"
    ylabel = f"Number of {dataset.plural.capitalize()}"
    title = (
        f"{dataset.pretty_name}: {dataset.subset_col.title()} Internal Agreement (Mini-Batch Kendall's W)"
        f"\n({num_models} models, {num_subsets} {dataset.plural}, {num_trials} trials, {num_folds} folds)"
    )
    annotate = True
    median = data.median()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    xlim = (0, 1)
    ylim = {
        Dataset.BBH: (0, num_subsets),
        Dataset.GPQA_DIAMOND: (0, num_subsets),
        Dataset.MATH_LVL_5: (0, num_subsets),
        Dataset.MMLU_PRO: (0, num_subsets),
        Dataset.DS_1000: (0, num_subsets),
        Dataset.MATH: (0, num_subsets),
        Dataset.MMLU: (0, num_subsets),
    }[dataset]
    median_label = "Median Kendall's W"

    return plot_histogram(
        data,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        annotate=annotate,
        median=median,
        q1=q1,
        q3=q3,
        xlim=xlim,
        ylim=ylim,
        median_label=median_label,
    )


def plot_per_subset_performance_histograms(
    dataset: Dataset,
    dataset_df: pd.DataFrame,
    model_scores_df: pd.DataFrame,
    subsets,
    num_models: int,
    num_subsets: int,
    **kwargs,
) -> plt.Figure:
    """Plot a grid of histograms showing model performance distributions per subset.

    Each subplot corresponds to one subset and shows a histogram of per-model
    mean scores computed over the instances belonging to that subset.

    The subplots are arranged in a grid as close to square as possible. When a
    perfect square is not possible the grid is wider than it is tall.

    Args:
        dataset: The dataset being analysed, used for labels and titles.
        dataset_df: Full benchmark instance-level DataFrame.
        model_scores_df: Per-instance model scores (instances x models).
        subsets: Iterable of subset identifiers to plot.
        num_models: Number of models in the benchmark.
        num_subsets: Number of subsets.

    Returns:
        The matplotlib Figure containing the subplot grid.
    """
    annotation_fontsize = 12
    tick_fontsize = annotation_fontsize + 2
    legend_fontsize = tick_fontsize
    label_fontsize = tick_fontsize + 2
    title_fontsize = tick_fontsize + 2
    suptitle_fontsize = title_fontsize + 2
    raw_annotation_color = "red"
    linestyle = "--"
    linewidth = 3
    alpha = 0.3
    subplot_size = (6, 4)

    # Compute per-subset mean model scores (subsets x models)
    subset_scores = {}
    for subset in subsets:
        mask = (
            get_metadata_mask(dataset_df, dataset.subset_col, subset)
            if dataset == Dataset.DS_1000
            else dataset_df[dataset.subset_col] == subset
        )
        subset_scores[subset] = model_scores_df[mask].mean()
    subset_scores = pd.DataFrame(subset_scores).T
    subset_scores.index.name = "subset"

    subsets = sorted(subset_scores.index.tolist())
    n = len(subsets)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    if nrows > ncols:
        ncols += 1
        nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(subplot_size[0] * ncols, subplot_size[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    bins = np.linspace(0, 1, 11)
    for idx, subset_name in enumerate(subsets):
        ax = axes_flat[idx]
        data = subset_scores.loc[subset_name]

        sns.histplot(
            data=data,
            bins=bins,
            ax=ax,
            color=sns.color_palette("tab10")[0],
        )
        ax.set_xlim(0, 1)

        # Annotate bar heights
        for patch in ax.patches:
            if patch.get_height() > 0:
                ax.text(
                    patch.get_x() + patch.get_width() / 2,
                    patch.get_height(),
                    f"{int(patch.get_height())}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=annotation_fontsize,
                )

        # Median line
        median_val = data.median()
        ax.axvline(
            median_val,
            color=raw_annotation_color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=f"Median: {median_val:.2f}",
        )

        # IQR shaded region
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        ax.axvspan(
            q1,
            q3,
            alpha=alpha,
            color=raw_annotation_color,
            label=f"IQR: {iqr:.2f}",
        )

        ax.legend(fontsize=legend_fontsize - 4)
        ax.tick_params(labelsize=tick_fontsize - 2)
        ax.set_title(subset_name, fontsize=title_fontsize - 2)
        ax.set_xlabel("Score (0\u20131)", fontsize=label_fontsize - 2)
        ax.set_ylabel("Number of Models", fontsize=label_fontsize - 2)

    # Hide unused axes
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    suptitle = (
        f"{dataset.pretty_name}: Distribution of Raw "
        f"{dataset.metric.title()} Scores\n"
        f"by {dataset.subset_col.title()}"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.suptitle(suptitle, fontsize=suptitle_fontsize, y=0.98)

    return fig


def main(dataset: Dataset, experiment: str, num_trials: int, num_folds: int) -> None:
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
        num_trials=num_trials,
        num_folds=num_folds,
        experiment=experiment,
    )

    split_halves_df = split_halves_analysis(**shared)
    data_name = f"split_halves__internal_agreement__{num_models=}__{num_trials=}"
    data_path = build_data_path(dataset, experiment, data_name)
    split_halves_df.to_csv(data_path, index=False)
    logger.info(f"Saved data to {data_path}")

    split_halves_fig = plot_split_halves_histogram(split_halves_df, **shared)
    plot_name = (
        f"split_halves__internal_agreement__histogram__{num_models=}__{num_trials=}"
    )
    plot_path = build_plot_path(dataset, experiment, plot_name)
    split_halves_fig.savefig(plot_path)
    plt.close(split_halves_fig)
    logger.info(f"Saved plot to {plot_path}")

    per_subset_fig = plot_per_subset_performance_histograms(**shared)
    plot_name = f"per_subset__performance__histograms__{num_models=}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    per_subset_fig.savefig(plot_path)
    plt.close(per_subset_fig)
    logger.info(f"Saved plot to {plot_path}")

    # bootstrap_df = bootstrap_analysis(**shared)
    # data_name = f"bootstrap__internal_agreement__{num_models=}__{num_trials=}"
    # data_path = build_data_path(dataset, experiment, data_name)
    # bootstrap_df.to_csv(data_path, index=False)
    # logger.info(f"Saved data to {data_path}")

    # bootstrap_fig = plot_bootstrap_histogram(bootstrap_df, **shared)
    # plot_name = (
    #     f"bootstrap__internal_agreement__histogram__{num_models=}__{num_trials=}"
    # )
    # plot_path = build_plot_path(dataset, experiment, plot_name)
    # bootstrap_fig.savefig(plot_path)
    # plt.close(bootstrap_fig)
    # logger.info(f"Saved plot to {plot_path}")

    # minibatch_df = minibatch_analysis(**shared)
    # data_name = (
    #     f"minibatch__internal_agreement__{num_models=}__{num_folds=}__{num_trials=}"
    # )
    # data_path = build_data_path(dataset, experiment, data_name)
    # minibatch_df.to_csv(data_path, index=False)
    # logger.info(f"Saved data to {data_path}")

    # minibatch_fig = plot_minibatch_histogram(minibatch_df, **shared)
    # plot_name = f"minibatch__internal_agreement__histogram__{num_models=}__{num_folds=}__{num_trials=}"
    # plot_path = build_plot_path(dataset, experiment, plot_name)
    # minibatch_fig.savefig(plot_path)
    # plt.close(minibatch_fig)
    # logger.info(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    datasets = [
        Dataset.MATH,
        Dataset.MMLU,
        Dataset.DS_1000,
        Dataset.BBH,
        Dataset.GPQA_DIAMOND,
        Dataset.MATH_LVL_5,
        Dataset.MMLU_PRO,
    ]
    experiment = Path(__file__).stem
    num_trial_values = [500]
    num_fold_values = [5]

    for i, dataset in enumerate(datasets):
        for num_trials in num_trial_values:
            for num_folds in num_fold_values:
                print(
                    f"{'-'*80} Dataset {i+1}/{len(datasets)}: {dataset.pretty_name}, "
                    f"{num_trials=}, {num_folds=} {'-'*80}"
                )
                main(dataset, experiment, num_trials, num_folds)
