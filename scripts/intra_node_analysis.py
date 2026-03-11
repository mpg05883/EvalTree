import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from tqdm import tqdm

from src.utils.capability_tree import (
    Node,
    collect_nodes,
    load_capability_tree,
)
from src.utils.enums import Dataset
from src.utils.metrics import kendallw
from src.utils.model import load_model_scores
from src.utils.path import build_data_path, build_plot_path
from src.utils.plot import plot_histogram, plot_stripplot

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def split_halves_analysis(
    nodes: list[Node],
    model_scores_df: pd.DataFrame,
    num_trials: int,
    **kwargs,
) -> pd.DataFrame:
    """Measure intra-node ranking consistency using split-halves Kendall's Tau.

    For each node, randomly splits its instances into two halves ``num_trials``
    times, computes mean model scores on each half, and measures Kendall's Tau
    between the two resulting rankings. A high mean tau indicates that any
    random half of the node reliably reproduces the same model ordering as the
    other half, suggesting the node's instances are internally consistent.

    Args:
        nodes: Qualifying nodes collected from the capability tree.
        model_scores_df: Per-instance model scores (instances × models).
        num_trials: Number of random split-halves trials per node.

    Returns:
        A DataFrame with one row per node and columns
        ``["node", "depth", "mean_kendall_tau", "std_kendall_tau", "num_instances"]``.
    """
    tqdm_kwargs = {
        "desc": "Computing split-halves Kendall's taus",
        "total": len(nodes),
        "unit": "nodes",
    }

    results = []
    for node in tqdm(nodes, **tqdm_kwargs):
        indices = node.get_indices()
        node_scores = model_scores_df.iloc[indices]
        node_size = len(node_scores)

        taus = []
        for trial in range(num_trials):
            rng = np.random.default_rng(trial)
            shuffled_indices = rng.permutation(node_size)
            half = node_size // 2
            scores_a = node_scores.iloc[shuffled_indices[:half]].mean()
            scores_b = node_scores.iloc[shuffled_indices[half:]].mean()
            tau, _ = kendalltau(scores_a, scores_b)
            taus.append(tau)

        results.append(
            {
                "node": node.capability,
                "depth": node.depth,
                "mean_kendall_tau": np.mean(taus),
                "std_kendall_tau": np.std(taus),
                "num_instances": node_size,
            }
        )

    return pd.DataFrame(results)


def plot_all_nodes_split_halves_histogram(
    df: pd.DataFrame,
    dataset: Dataset,
    num_models: int,
    num_nodes: int,
    min_instances: int,
    num_trials: int,
    **kwargs,
) -> plt.Figure:
    """Plot the distribution of split-halves Kendall's Tau values across all nodes.

    Takes the output of :func:`split_halves_analysis` and visualises how
    consistently each node reproduces the same model ranking across random
    splits. The x-axis spans (-1, 1) or (0, 1) depending on whether any
    negative tau values are present.

    Args:
        df: DataFrame returned by :func:`split_halves_analysis`.
        dataset: The dataset being analysed, used for the plot title.
        num_models: Number of models in the benchmark.
        num_nodes: Total number of qualifying nodes analysed.
        min_instances: Instance threshold used when collecting nodes.
        num_trials: Number of split-halves trials used per node.

    Returns:
        The matplotlib Figure containing the histogram.
    """
    data = df["mean_kendall_tau"]
    xlabel = r"Kendall's $\tau$"
    min_instance_label = r"$n_{\mathrm{min}}$"
    ylabel = "Number of Nodes"
    title = (
        f"{dataset.pretty_name}: Internal Node Agreement (Split-Halves {xlabel})"
        f"\n({num_models} models, {num_nodes} nodes, {num_trials} trials, {min_instance_label}={min_instances})"
    )
    median = data.median()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    xlim = (-1, 1) if min(data) < 0 else (0, 1)
    ylim = (0, num_nodes)

    return plot_histogram(
        data,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        annotate=True,
        median=median,
        q1=q1,
        q3=q3,
        xlim=xlim,
        ylim=ylim,
    )


def plot_per_level_split_halves_stripplot(
    df: pd.DataFrame,
    dataset: Dataset,
    num_models: int,
    min_instances: int,
    num_trials: int,
    **kwargs,
) -> plt.Figure:
    """Plot split-halves Kendall's Tau values per capability tree level as a strip plot.

    Takes the output of :func:`split_halves_analysis` and produces a single
    strip plot where each x-axis tick corresponds to a capability tree level
    and every dot represents one node. Dots are color-coded by level using the
    tab10 palette.

    Args:
        df: DataFrame returned by :func:`split_halves_analysis`, which
            includes a ``depth`` column used to assign nodes to levels.
        dataset: The dataset being analysed, used for the plot title.
        num_models: Number of models in the benchmark.
        min_instances: Instance threshold used when collecting nodes.
        num_trials: Number of split-halves trials used per node.

    Returns:
        The matplotlib Figure containing the strip plot.
    """
    plot_df = df.copy()
    plot_df["level"] = plot_df["depth"].apply(lambda d: f"Level {int(d)}")
    order = [f"Level {int(d)}" for d in sorted(df["depth"].unique())]

    xlabel = "Capability Tree Level"
    ylabel = r"Kendall's $\tau$"
    min_instance_label = r"$n_{\mathrm{min}}$"
    title = (
        f"{dataset.pretty_name}: Internal Node Agreement (Split-Halves {ylabel})"
        f"\n({num_models} models, {num_trials} trials, {min_instance_label}={min_instances})"
    )
    ylim = (-1, 1) if df["mean_kendall_tau"].min() < 0 else (0, 1)

    return plot_stripplot(
        data=plot_df,
        x="level",
        y="mean_kendall_tau",
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        hue="level",
        order=order,
        palette="tab10",
        ylim=ylim,
        rotation=30,
    )


def bootstrap_analysis(
    nodes: list[Node],
    model_scores_df: pd.DataFrame,
    num_trials: int,
    **kwargs,
) -> pd.DataFrame:
    """Measure intra-node ranking stability using bootstrap Kendall's Tau.

    For each node, computes the reference model ranking from all of the node's
    instances, then draws ``num_trials`` bootstrap samples and measures
    Kendall's Tau between each bootstrap ranking and the reference. A high
    mean tau indicates that resampled instances reliably recover the same model
    ordering, reflecting stable estimation within the node.

    Args:
        nodes: Qualifying nodes collected from the capability tree.
        model_scores_df: Per-instance model scores (instances × models).
        num_trials: Number of bootstrap trials per node.

    Returns:
        A DataFrame with one row per node and columns
        ``["node", "depth", "mean_kendall_tau", "std_kendall_tau", "num_instances"]``.
    """
    tqdm_kwargs = {
        "desc": "Computing bootstrap Kendall's taus",
        "total": len(nodes),
        "unit": "nodes",
    }

    results = []
    for node in tqdm(nodes, **tqdm_kwargs):
        indices = node.get_indices()
        node_scores = model_scores_df.iloc[indices]
        n = len(node_scores)
        reference_scores = node_scores.mean()

        taus = []
        for b in range(num_trials):
            rng = np.random.default_rng(b)
            bootstrap_indices = rng.integers(0, n, size=n)
            bootstrap_scores = node_scores.iloc[bootstrap_indices].mean()
            tau, _ = kendalltau(reference_scores, bootstrap_scores)
            taus.append(tau)

        results.append(
            {
                "node": node.capability,
                "depth": node.depth,
                "mean_kendall_tau": np.mean(taus),
                "std_kendall_tau": np.std(taus),
                "num_instances": n,
            }
        )

    return pd.DataFrame(results)


def plot_all_nodes_bootstrap_histogram(
    df: pd.DataFrame,
    dataset: Dataset,
    num_models: int,
    num_nodes: int,
    min_instances: int,
    num_trials: int,
    **kwargs,
) -> plt.Figure:
    """Plot the distribution of bootstrap Kendall's Tau values across all nodes.

    Takes the output of :func:`bootstrap_analysis` and visualises how stably
    each node's model ranking is estimated under resampling. The x-axis spans
    (-1, 1) or (0, 1) depending on whether any negative tau values are present.

    Args:
        df: DataFrame returned by :func:`bootstrap_analysis`.
        dataset: The dataset being analysed, used for the plot title.
        num_models: Number of models in the benchmark.
        num_nodes: Total number of qualifying nodes analysed.
        min_instances: Instance threshold used when collecting nodes.
        num_trials: Number of bootstrap trials used per node.

    Returns:
        The matplotlib Figure containing the histogram.
    """
    data = df["mean_kendall_tau"]
    xlabel = r"Kendall's $\tau$"
    min_instance_label = r"$n_{\mathrm{min}}$"
    ylabel = "Number of Nodes"
    title = (
        f"{dataset.pretty_name}: Internal Node Agreement (Bootstrap {xlabel})"
        f"\n({num_models} models, {num_nodes} nodes, {num_trials} trials, {min_instance_label}={min_instances})"
    )
    median = data.median()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    xlim = (-1, 1) if min(data) < 0 else (0, 1)
    ylim = (0, num_nodes)

    return plot_histogram(
        data,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        annotate=True,
        median=median,
        q1=q1,
        q3=q3,
        xlim=xlim,
        ylim=ylim,
    )


def plot_per_level_bootstrap_stripplot(
    df: pd.DataFrame,
    dataset: Dataset,
    num_models: int,
    min_instances: int,
    num_trials: int,
    **kwargs,
) -> plt.Figure:
    """Plot bootstrap Kendall's Tau values per capability tree level as a strip plot.

    Takes the output of :func:`bootstrap_analysis` and produces a single strip
    plot where each x-axis tick corresponds to a capability tree level and
    every dot represents one node. Dots are color-coded by level using the
    tab10 palette.

    Args:
        df: DataFrame returned by :func:`bootstrap_analysis`, which includes a
            ``depth`` column used to assign nodes to levels.
        dataset: The dataset being analysed, used for the plot title.
        num_models: Number of models in the benchmark.
        min_instances: Instance threshold used when collecting nodes.
        num_trials: Number of bootstrap trials used per node.

    Returns:
        The matplotlib Figure containing the strip plot.
    """
    plot_df = df.copy()
    plot_df["level"] = plot_df["depth"].apply(lambda d: f"Level {int(d)}")
    order = [f"Level {int(d)}" for d in sorted(df["depth"].unique())]

    xlabel = "Capability Tree Level"
    ylabel = r"Kendall's $\tau$"
    min_instance_label = r"$n_{\mathrm{min}}$"
    title = (
        f"{dataset.pretty_name}: Internal Node Agreement (Bootstrap {ylabel})"
        f"\n({num_models} models, {num_trials} trials, {min_instance_label}={min_instances})"
    )
    ylim = (-1, 1) if df["mean_kendall_tau"].min() < 0 else (0, 1)

    return plot_stripplot(
        data=plot_df,
        x="level",
        y="mean_kendall_tau",
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        hue="level",
        order=order,
        palette="tab10",
        ylim=ylim,
        rotation=30,
    )


def minibatch_w_analysis(
    nodes: list[Node],
    model_scores_df: pd.DataFrame,
    num_trials: int,
    num_folds: int,
    **kwargs,
) -> pd.DataFrame:
    """Measure intra-node ranking agreement across folds using Kendall's W.

    For each node, randomly shuffles its instances and divides them into
    ``num_folds`` equal folds ``num_trials`` times. Each trial computes
    Kendall's W across the fold-level model rankings, measuring how concordant
    the rankings are across folds. A W near 1 indicates strong agreement
    between folds; a W near 0 indicates near-random disagreement.

    Args:
        nodes: Qualifying nodes collected from the capability tree.
        model_scores_df: Per-instance model scores (instances × models).
        num_trials: Number of random shuffle trials per node.
        num_folds: Number of folds to split instances into per trial.

    Returns:
        A DataFrame with one row per node and columns
        ``["node", "depth", "mean_kendall_w", "std_kendall_w", "num_instances"]``.
    """
    tqdm_kwargs = {
        "desc": "Computing mini-batch Kendall's Ws",
        "total": len(nodes),
        "unit": "nodes",
    }

    results = []
    for node in tqdm(nodes, **tqdm_kwargs):
        indices = node.get_indices()
        node_scores = model_scores_df.iloc[indices]
        node_size = len(node_scores)

        kendallw_values = []
        for trial in range(num_trials):
            rng = np.random.default_rng(trial)
            shuffled_indices = rng.permutation(node_size)
            folds = np.array_split(shuffled_indices, num_folds)

            # Each row is one fold's mean model scores: shape (num_folds, num_models)
            fold_scores = np.array(
                [node_scores.iloc[fold].mean().values for fold in folds]
            )
            kendallw_values.append(kendallw(fold_scores))

        results.append(
            {
                "node": node.capability,
                "depth": node.depth,
                "mean_kendall_w": np.mean(kendallw_values),
                "std_kendall_w": np.std(kendallw_values),
                "num_instances": node_size,
            }
        )

    return pd.DataFrame(results)


def plot_all_nodes_minibatch_w_histogram(
    df: pd.DataFrame,
    dataset: Dataset,
    num_models: int,
    num_nodes: int,
    min_instances: int,
    num_trials: int,
    num_folds: int,
    **kwargs,
) -> plt.Figure:
    """Plot the distribution of mini-batch Kendall's W values across all nodes.

    Takes the output of :func:`minibatch_w_analysis` and visualises how
    concordant model rankings are across random folds within each node. The
    x-axis always spans (0, 1) since Kendall's W is non-negative.

    Args:
        df: DataFrame returned by :func:`minibatch_w_analysis`.
        dataset: The dataset being analysed, used for the plot title.
        num_models: Number of models in the benchmark.
        num_nodes: Total number of qualifying nodes analysed.
        min_instances: Instance threshold used when collecting nodes.
        num_trials: Number of shuffle trials used per node.
        num_folds: Number of folds used per trial.

    Returns:
        The matplotlib Figure containing the histogram.
    """
    data = df["mean_kendall_w"]
    xlabel = "Kendall's W"
    min_instance_label = r"$n_{\mathrm{min}}$"
    ylabel = "Number of Nodes"
    title = (
        f"{dataset.pretty_name}: Internal Node Agreement (Mini-Batch Kendall's W)"
        f"\n({num_models} models, {num_nodes} nodes, {num_trials} trials, {num_folds} folds, {min_instance_label}={min_instances})"
    )
    median = data.median()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    xlim = (0, 1)
    ylim = (0, num_nodes)

    return plot_histogram(
        data,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        annotate=True,
        median=median,
        q1=q1,
        q3=q3,
        xlim=xlim,
        ylim=ylim,
    )


def plot_per_level_minibatch_w_stripplot(
    df: pd.DataFrame,
    dataset: Dataset,
    num_models: int,
    min_instances: int,
    num_trials: int,
    num_folds: int,
    **kwargs,
) -> plt.Figure:
    """Plot mini-batch Kendall's W values per capability tree level as a strip plot.

    Takes the output of :func:`minibatch_w_analysis` and produces a single
    strip plot where each x-axis tick corresponds to a capability tree level
    and every dot represents one node. Dots are color-coded by level using the
    tab10 palette.

    Args:
        df: DataFrame returned by :func:`minibatch_w_analysis`, which includes
            a ``depth`` column used to assign nodes to levels.
        dataset: The dataset being analysed, used for the plot title.
        num_models: Number of models in the benchmark.
        min_instances: Instance threshold used when collecting nodes.
        num_trials: Number of shuffle trials used per node.
        num_folds: Number of folds used per trial.

    Returns:
        The matplotlib Figure containing the strip plot.
    """
    plot_df = df.copy()
    plot_df["level"] = plot_df["depth"].apply(lambda d: f"Level {int(d)}")
    order = [f"Level {int(d)}" for d in sorted(df["depth"].unique())]

    xlabel = "Capability Tree Level"
    ylabel = "Kendall's W"
    min_instance_label = r"$n_{\mathrm{min}}$"
    title = (
        f"{dataset.pretty_name}: Internal Node Agreement (Mini-Batch Kendall's W)"
        f"\n({num_models} models, {num_trials} trials, {num_folds} folds, {min_instance_label}={min_instances})"
    )

    return plot_stripplot(
        data=plot_df,
        x="level",
        y="mean_kendall_w",
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        hue="level",
        order=order,
        palette="tab10",
        ylim=(0, 1),
        rotation=30,
    )


def main(
    dataset: Dataset,
    min_instances: int,
    experiment: str,
    num_trials: int,
    num_folds: int,
) -> None:
    model_scores_df = load_model_scores(dataset)
    num_instances = len(model_scores_df)
    num_models = len(model_scores_df.columns)

    root = load_capability_tree(dataset)
    nodes = collect_nodes(root, min_instances)
    num_nodes = len(nodes)

    shared = dict(
        dataset=dataset,
        nodes=nodes,
        model_scores_df=model_scores_df,
        num_nodes=num_nodes,
        num_models=num_models,
        num_instances=num_instances,
        num_trials=num_trials,
        num_folds=num_folds,
        min_instances=min_instances,
        experiment=experiment,
    )

    split_halves_df = split_halves_analysis(**shared)
    data_name = f"all-nodes_split-halves_internal-agreement_num-models={num_models}_min-instances={min_instances}_num-trials={num_trials}"
    data_path = build_data_path(dataset, experiment, data_name)
    split_halves_df.to_csv(data_path, index=False)
    logger.info(f"Saved data to {data_path}")

    split_halves_all_nodes_fig = plot_all_nodes_split_halves_histogram(
        split_halves_df,
        **shared,
    )
    plot_name = f"all-nodes_split-halves_internal-agreement_histogram_min-instances={min_instances}_num-trials={num_trials}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    split_halves_all_nodes_fig.savefig(plot_path)
    plt.close(split_halves_all_nodes_fig)
    logger.info(f"Saved plot to {plot_path}")

    split_halves_per_level_fig = plot_per_level_split_halves_stripplot(
        split_halves_df,
        **shared,
    )
    plot_name = f"per-level_split-halves_internal-agreement_stripplot_min-instances={min_instances}_num-trials={num_trials}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    split_halves_per_level_fig.savefig(plot_path)
    plt.close(split_halves_per_level_fig)
    logger.info(f"Saved plot to {plot_path}")

    bootstrap_df = bootstrap_analysis(**shared)
    data_name = f"all-nodes_bootstrap_internal-agreement_num-models={num_models}_min-instances={min_instances}_num-trials={num_trials}"
    data_path = build_data_path(dataset, experiment, data_name)
    bootstrap_df.to_csv(data_path, index=False)
    logger.info(f"Saved data to {data_path}")

    bootstrap_all_nodes_fig = plot_all_nodes_bootstrap_histogram(
        bootstrap_df,
        **shared,
    )
    plot_name = f"all-nodes_bootstrap_internal-agreement_histogram_min-instances={min_instances}_num-trials={num_trials}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    bootstrap_all_nodes_fig.savefig(plot_path)
    plt.close(bootstrap_all_nodes_fig)
    logger.info(f"Saved plot to {plot_path}")

    bootstrap_per_level_fig = plot_per_level_bootstrap_stripplot(
        bootstrap_df,
        **shared,
    )
    plot_name = f"per-level_bootstrap_internal-agreement_stripplot_min-instances={min_instances}_num-trials={num_trials}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    bootstrap_per_level_fig.savefig(plot_path)
    plt.close(bootstrap_per_level_fig)
    logger.info(f"Saved plot to {plot_path}")

    minibatch_w_df = minibatch_w_analysis(**shared)
    data_name = f"all-nodes_mini-batch_internal-agreement_num-models={num_models}_min-instances={min_instances}_num-trials={num_trials}_num-folds={num_folds}"
    data_path = build_data_path(dataset, experiment, data_name)
    minibatch_w_df.to_csv(data_path, index=False)
    logger.info(f"Saved data to {data_path}")

    minibatch_w_all_nodes_fig = plot_all_nodes_minibatch_w_histogram(
        minibatch_w_df,
        **shared,
    )
    plot_name = f"all-nodes_mini-batch_internal-agreement_histogram_min-instances={min_instances}_num-trials={num_trials}_num-folds={num_folds}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    minibatch_w_all_nodes_fig.savefig(plot_path)
    plt.close(minibatch_w_all_nodes_fig)
    logger.info(f"Saved plot to {plot_path}")

    minibatch_w_per_level_fig = plot_per_level_minibatch_w_stripplot(
        minibatch_w_df,
        **shared,
    )
    plot_name = f"per-level_mini-batch_internal-agreement_stripplot_min-instances={min_instances}_num-trials={num_trials}_num-folds={num_folds}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    minibatch_w_per_level_fig.savefig(plot_path)
    plt.close(minibatch_w_per_level_fig)
    logger.info(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    # NOTE: We ignore the following datasets:
    # - Chatbot-Arena and Chatbot-Arena (New) don't have per-instance scores
    # - WildChat-10K only has evaluation results for two models
    datasets = [Dataset.DS_1000, Dataset.MATH, Dataset.MMLU]
    experiment = Path(__file__).stem
    num_trial_values = [50, 500, 1000]
    num_fold_values = [2, 5, 10]

    for i, dataset in enumerate(datasets):
        one_tenth = dataset.num_instances // 10
        min_instance_values = [0, 50, one_tenth]

        for min_instances in min_instance_values:
            for num_trials in num_trial_values:
                for num_folds in num_fold_values:
                    print(
                        f"{'-'*80} Dataset {i+1}/{len(datasets)}: {dataset.pretty_name}, "
                        f"{min_instances=}, {num_trials=}, {num_folds=} {'-'*80}"
                    )
                    main(
                        dataset,
                        min_instances,
                        experiment,
                        num_trials,
                        num_folds,
                    )
