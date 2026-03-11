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
        "desc": "Computing split-halves Kendall's Taus",
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
    ylabel = f"{dataset.subset_col.capitalize()} Count"
    title = (
        f"{dataset.pretty_name}: Split-Halves Kendall's Tau Across {dataset.plural.capitalize()}"
        f"\n({num_models} models, {num_subsets} {dataset.plural}, {num_instances} instances, {num_trials} trials)"
    )
    mean = data.mean()
    std = data.std()
    xlim = (-1, 1) if min(data) < 0 else (0, 1)
    ylim = {
        Dataset.DS_1000: (0, num_subsets),
        Dataset.MATH: (0, num_subsets),
        Dataset.MMLU: (0, num_subsets // 2),
    }[dataset]
    color = sns.color_palette("tab10")[0]

    return plot_histogram(
        data,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        annotate=True,
        mean=mean,
        std=std,
        xlim=xlim,
        ylim=ylim,
        color=color,
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
        "desc": "Computing bootstrapped Kendall's Taus",
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
    ylabel = f"{dataset.subset_col.capitalize()} Count"
    title = (
        f"{dataset.pretty_name}: Bootstrapped Kendall's Tau Across {dataset.plural.capitalize()}"
        f"\n({num_models} models, {num_subsets} {dataset.plural}, {num_instances} instances, {num_trials} trials)"
    )
    mean = data.mean()
    std = data.std()
    xlim = (-1, 1) if min(data) < 0 else (0, 1)
    ylim = {
        Dataset.DS_1000: (0, num_subsets),
        Dataset.MATH: (0, num_subsets),
        Dataset.MMLU: (0, num_subsets),
    }[dataset]
    color = sns.color_palette("tab10")[1]

    return plot_histogram(
        data,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        annotate=True,
        mean=mean,
        std=std,
        xlim=xlim,
        ylim=ylim,
        color=color,
    )


def minibatch_w_analysis(
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
        "desc": "Computing mini-batch Kendall's W",
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


def plot_minibatch_w_histogram(
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

    Takes the output of :func:`minibatch_w_analysis` and visualises how
    concordant model rankings are across random folds within each subset.
    The x-axis always spans (0, 1) since Kendall's W is non-negative.

    Args:
        df: DataFrame returned by :func:`minibatch_w_analysis`.
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
    ylabel = f"{dataset.subset_col.capitalize()} Count"
    title = (
        f"{dataset.pretty_name}: Mini-Batch Kendall's W Across {dataset.plural.capitalize()}"
        f"\n({num_models} models, {num_subsets} {dataset.plural}, {num_instances} instances, {num_trials} trials, {num_folds} folds)"
    )
    mean = data.mean()
    std = data.std()
    xlim = (0, 1)
    ylim = {
        Dataset.DS_1000: (0, num_subsets),
        Dataset.MATH: (0, num_subsets),
        Dataset.MMLU: (0, num_subsets),
    }[dataset]
    color = sns.color_palette("tab10")[2]

    return plot_histogram(
        data,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        annotate=True,
        mean=mean,
        std=std,
        xlim=xlim,
        ylim=ylim,
        color=color,
    )


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
    data_name = f"split-halves_internal-agreement_num-trials={num_trials}"
    data_path = build_data_path(dataset, experiment, data_name)
    split_halves_df.to_csv(data_path, index=False)
    print(f"Saved data to {data_path}")

    split_halves_fig = plot_split_halves_histogram(split_halves_df, **shared)
    plot_name = f"split-halves_internal-agreement_histogram_num-trials={num_trials}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    split_halves_fig.savefig(plot_path)
    plt.close(split_halves_fig)
    print(f"Saved plot to {plot_path}")

    bootstrap_df = bootstrap_analysis(**shared)
    data_name = f"bootstrap_internal-agreement_num-trials={num_trials}"
    data_path = build_data_path(dataset, experiment, data_name)
    bootstrap_df.to_csv(data_path, index=False)
    print(f"Saved data to {data_path}")

    bootstrap_fig = plot_bootstrap_histogram(bootstrap_df, **shared)
    plot_name = f"bootstrap_internal-agreement_histogram_num-trials={num_trials}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    bootstrap_fig.savefig(plot_path)
    plt.close(bootstrap_fig)
    print(f"Saved plot to {plot_path}")

    minibatch_w_df = minibatch_w_analysis(**shared)
    data_name = (
        f"mini-batch_internal-agreement_num-folds={num_folds}_num-trials={num_trials}"
    )
    data_path = build_data_path(dataset, experiment, data_name)
    minibatch_w_df.to_csv(data_path, index=False)
    print(f"Saved data to {data_path}")

    minibatch_w_fig = plot_minibatch_w_histogram(minibatch_w_df, **shared)
    plot_name = f"mini-batch_internal-agreement_histogram_num-folds={num_folds}_num-trials={num_trials}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    minibatch_w_fig.savefig(plot_path)
    plt.close(minibatch_w_fig)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    # Only MATH, MMLU, and DS-1000 have pre-defined subsets
    datasets = [Dataset.MATH, Dataset.MMLU, Dataset.DS_1000]
    experiment = Path(__file__).stem
    num_trial_values = [500]
    num_fold_values = [5]

    for dataset in datasets:
        for num_trials in num_trial_values:
            for num_folds in num_fold_values:
                main(dataset, experiment, num_trials, num_folds)
        print(f"{'-'*200}")
