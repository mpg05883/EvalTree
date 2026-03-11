import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau
from tqdm import tqdm

from src.utils.capability_tree import (
    Level,
    Node,
    collect_levels,
    load_capability_tree,
)
from src.utils.enums import Dataset
from src.utils.metrics import kendallw
from src.utils.model import load_model_scores
from src.utils.path import build_plot_path
from src.utils.plot import plot_histogram


def split_halves_kendall_tau(
    dataset: Dataset,
    nodes_with_levels: list[tuple[Node, int]],
    levels: list[int],
    model_scores_df: pd.DataFrame,
    num_trials: int,
    num_nodes: int,
    num_models: int,
    num_instances: int,
    analysis: str,
    color,
) -> None:
    """Compute and plot split-halves Kendall's Tau across all nodes and by level.

    For each node, the instances are randomly split in half num_trials times.
    Kendall's Tau is computed between the mean model scores of each half, and
    the mean Tau across trials is recorded.

    Args:
        dataset: The dataset being analysed.
        nodes_with_levels: List of (node, level) pairs to analyse.
        levels: Sorted list of depth levels present in the tree.
        model_scores_df: DataFrame of per-instance model scores.
        num_trials: Number of random split-half trials per node.
        num_nodes: Total number of nodes being analysed.
        num_models: Number of models in the dataset.
        num_instances: Total number of instances in the dataset.
        analysis: Analysis name used when building plot file paths.
        color: Matplotlib colour for histogram bars.
    """
    kwargs = {
        "desc": "Computing split-half Kendall's Taus",
        "total": num_nodes,
        "unit": "nodes",
    }

    results = []

    for node, level in tqdm(nodes_with_levels, **kwargs):
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
                "level": level,
                "mean_kendall_tau": np.mean(taus),
                "std_kendall_tau": np.std(taus),
                "num_instances": node_size,
            }
        )

    df = pd.DataFrame(results)

    xlim = (-1, 1) if min(df["mean_kendall_tau"]) < 0 else (0, 1)

    ylim = {
        Dataset.CHATBOT_ARENA: (0, num_nodes),
        Dataset.CHATBOT_ARENA_NEW: (0, num_nodes),
        Dataset.DS_1000: (0, num_nodes),
        Dataset.MATH: (0, num_nodes),
        Dataset.MMLU: (0, num_nodes // 2),
        Dataset.WILDCHAT_10K: (0, num_nodes),
    }[dataset]

    plot_histogram(
        df["mean_kendall_tau"],
        xlabel="Kendall's Tau",
        ylabel="Node Count",
        title=(
            f"{dataset}: Split-Halves Kendall's Tau Across Nodes"
            f"\n({num_models} models, {num_nodes} nodes, {num_instances} instances, {num_trials} trials)"
        ),
        annotate=True,
        mean=df["mean_kendall_tau"].mean(),
        std=df["mean_kendall_tau"].std(),
        xlim=xlim,
        ylim=ylim,
        color=color,
    )

    file_path = build_plot_path(
        dataset,
        analysis,
        plot_name=f"split-halves_kendall-tau_histogram_num-trials={num_trials}",
        sub_dirs=["all_nodes"],
    )
    plt.savefig(file_path)
    print(f"Saved plot to {file_path}")
    plt.close()

    _, axes = plt.subplots(len(levels), 1, figsize=(8, 4 * len(levels)))
    axes = np.atleast_1d(axes)

    for i, level in enumerate(levels):
        level_data = df[df["level"] == level]["mean_kendall_tau"]
        plot_histogram(
            level_data,
            xlabel="Kendall's Tau",
            ylabel="Node Count",
            title=f"Level {level} ({len(level_data)} nodes, {num_trials} trials)",
            ax=axes[i],
            annotate=True,
            mean=level_data.mean(),
            std=level_data.std(),
            xlim=xlim,
            color=color,
        )

    plt.suptitle(
        f"{dataset}: Split-Halves Kendall's Tau by Level"
        f"\n({num_models} models, {num_nodes} nodes, {num_instances} instances, {num_trials} trials)",
        y=1.0,
    )
    plt.tight_layout()
    file_path = build_plot_path(
        dataset,
        analysis,
        plot_name=f"split-halves_kendall-tau_num-trials={num_trials}",
        sub_dirs=["by_level"],
    )
    plt.savefig(file_path, bbox_inches="tight")
    print(f"Saved plot to {file_path}")
    plt.close()


def bootstrapped_kendall_tau(
    dataset: Dataset,
    nodes_with_levels: list[tuple[Node, int]],
    levels: list[int],
    model_scores_df: pd.DataFrame,
    num_trials: int,
    num_nodes: int,
    num_models: int,
    num_instances: int,
    analysis: str,
    color,
) -> None:
    """Compute and plot bootstrapped Kendall's Tau across all nodes and by level.

    For each node, bootstrap samples are drawn num_trials times. Kendall's Tau
    is computed between the reference ranking (mean across all instances) and
    the bootstrap ranking, and the mean Tau across trials is recorded.

    Args:
        dataset: The dataset being analysed.
        nodes_with_levels: List of (node, level) pairs to analyse.
        levels: Sorted list of depth levels present in the tree.
        model_scores_df: DataFrame of per-instance model scores.
        num_trials: Number of bootstrap trials per node.
        num_nodes: Total number of nodes being analysed.
        num_models: Number of models in the dataset.
        num_instances: Total number of instances in the dataset.
        analysis: Analysis name used when building plot file paths.
        color: Matplotlib colour for histogram bars.
    """
    kwargs = {
        "desc": "Computing bootstrapped Kendall's Taus",
        "total": num_nodes,
        "unit": "nodes",
    }

    results = []

    for node, level in tqdm(nodes_with_levels, **kwargs):
        indices = node.get_indices()
        node_scores = model_scores_df.iloc[indices]
        n = len(node_scores)

        # Reference ranking: model ranking across all instances in the node
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
                "level": level,
                "mean_kendall_tau": np.mean(taus),
                "std_kendall_tau": np.std(taus),
                "num_instances": n,
            }
        )

    df = pd.DataFrame(results)

    xlim = (-1, 1) if min(df["mean_kendall_tau"]) < 0 else (0, 1)

    ylim = {
        Dataset.CHATBOT_ARENA: (0, num_nodes),
        Dataset.CHATBOT_ARENA_NEW: (0, num_nodes),
        Dataset.DS_1000: (0, num_nodes),
        Dataset.MATH: (0, num_nodes),
        Dataset.MMLU: (0, num_nodes),
        Dataset.WILDCHAT_10K: (0, num_nodes),
    }[dataset]

    plot_histogram(
        df["mean_kendall_tau"],
        xlabel="Kendall's Tau",
        ylabel="Node Count",
        title=(
            f"{dataset}: Bootstrapped Kendall's Tau Across Nodes"
            f"\n({num_models} models, {num_nodes} nodes, {num_instances} instances, {num_trials} trials)"
        ),
        annotate=True,
        mean=df["mean_kendall_tau"].mean(),
        std=df["mean_kendall_tau"].std(),
        xlim=xlim,
        ylim=ylim,
        color=color,
    )

    file_path = build_plot_path(
        dataset,
        analysis,
        plot_name=f"bootstrap_kendall-tau_histogram_num-trials={num_trials}",
        sub_dirs=["all_nodes"],
    )
    plt.savefig(file_path)
    print(f"Saved plot to {file_path}")
    plt.close()

    _, axes = plt.subplots(len(levels), 1, figsize=(8, 4 * len(levels)))
    axes = np.atleast_1d(axes)

    for i, level in enumerate(levels):
        level_data = df[df["level"] == level]["mean_kendall_tau"]
        plot_histogram(
            level_data,
            xlabel="Kendall's Tau",
            ylabel="Node Count",
            title=f"Level {level} ({len(level_data)} nodes, {num_trials} trials)",
            ax=axes[i],
            annotate=True,
            mean=level_data.mean(),
            std=level_data.std(),
            xlim=xlim,
            color=color,
        )

    plt.suptitle(
        f"{dataset}: Bootstrapped Kendall's Tau by Level"
        f"\n({num_models} models, {num_nodes} nodes, {num_instances} instances, {num_trials} trials)",
        y=1.0,
    )
    plt.tight_layout()
    file_path = build_plot_path(
        dataset,
        analysis,
        plot_name=f"bootstrap_kendall-tau_num-trials={num_trials}",
        sub_dirs=["by_level"],
    )
    plt.savefig(file_path, bbox_inches="tight")
    print(f"Saved plot to {file_path}")
    plt.close()


def minibatch_kendall_w(
    dataset: Dataset,
    nodes_with_levels: list[tuple[Node, int]],
    levels: list[int],
    model_scores_df: pd.DataFrame,
    num_trials: int,
    num_folds: int,
    num_nodes: int,
    num_models: int,
    num_instances: int,
    analysis: str,
    color,
) -> None:
    """Compute and plot mini-batch Kendall's W across all nodes and by level.

    For each node, instances are randomly shuffled and split into num_folds
    folds num_trials times. Kendall's W is computed across the per-fold mean
    model scores, and the mean W across trials is recorded.

    Args:
        dataset: The dataset being analysed.
        nodes_with_levels: List of (node, level) pairs to analyse.
        levels: Sorted list of depth levels present in the tree.
        model_scores_df: DataFrame of per-instance model scores.
        num_trials: Number of random shuffle trials per node.
        num_folds: Number of folds to split instances into per trial.
        num_nodes: Total number of nodes being analysed.
        num_models: Number of models in the dataset.
        num_instances: Total number of instances in the dataset.
        analysis: Analysis name used when building plot file paths.
        color: Matplotlib colour for histogram bars.
    """
    kwargs = {
        "desc": "Computing mini-batch Kendall's Ws",
        "total": num_nodes,
        "unit": "nodes",
    }

    results = []

    for node, level in tqdm(nodes_with_levels, **kwargs):
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
                "level": level,
                "mean_kendallw": np.mean(kendallw_values),
                "std_kendallw": np.std(kendallw_values),
                "num_instances": node_size,
            }
        )

    df = pd.DataFrame(results)

    xlim = (0, 1)
    ylim = (0, num_nodes)

    plot_histogram(
        df["mean_kendallw"],
        xlabel="Kendall's W",
        ylabel="Node Count",
        title=(
            f"{dataset}: Mini-Batch Kendall's W Across Nodes"
            f"\n({num_models} models, {num_nodes} nodes, {num_instances} instances, {num_trials} trials, {num_folds} folds)"
        ),
        annotate=True,
        mean=df["mean_kendallw"].mean(),
        std=df["mean_kendallw"].std(),
        xlim=xlim,
        ylim=ylim,
        color=color,
    )

    file_path = build_plot_path(
        dataset,
        analysis,
        plot_name=f"mini-batch_kendall-w_histogram_num-folds={num_folds}_num-trials={num_trials}",
        sub_dirs=["all_nodes"],
    )
    plt.savefig(file_path)
    print(f"Saved plot to {file_path}")
    plt.close()

    _, axes = plt.subplots(len(levels), 1, figsize=(8, 4 * len(levels)))
    axes = np.atleast_1d(axes)

    for i, level in enumerate(levels):
        level_data = df[df["level"] == level]["mean_kendallw"]
        plot_histogram(
            level_data,
            xlabel="Kendall's W",
            ylabel="Node Count",
            title=f"Level {level} ({len(level_data)} nodes, {num_trials} trials, {num_folds} folds)",
            ax=axes[i],
            annotate=True,
            mean=level_data.mean(),
            std=level_data.std(),
            xlim=xlim,
            color=color,
        )

    plt.suptitle(
        f"{dataset}: Mini-Batch Kendall's W by Level"
        f"\n({num_models} models, {num_nodes} nodes, {num_instances} instances, {num_trials} trials, {num_folds} folds)",
        y=1.0,
    )
    plt.tight_layout()
    file_path = build_plot_path(
        dataset,
        analysis,
        plot_name=f"mini-batch_kendall-w_num-folds={num_folds}_num-trials={num_trials}",
        sub_dirs=["by_level"],
    )
    plt.savefig(file_path, bbox_inches="tight")
    print(f"Saved plot to {file_path}")
    plt.close()


def main(dataset: Dataset) -> None:
    model_scores_df = load_model_scores(dataset)
    models = model_scores_df.columns.tolist()

    root = load_capability_tree(dataset)
    tree_levels: list[Level] = collect_levels(root)
    levels = [lv.depth for lv in tree_levels]
    nodes_with_levels = [(node, lv.depth) for lv in tree_levels for node in lv.nodes]

    colors = sns.color_palette("tab10")

    num_instances = len(model_scores_df)
    num_models = len(models)
    num_nodes = len(nodes_with_levels)
    num_trials = 500
    num_folds = 5
    analysis = "intra_node_analysis"

    shared = dict(
        dataset=dataset,
        nodes_with_levels=nodes_with_levels,
        levels=levels,
        model_scores_df=model_scores_df,
        num_trials=num_trials,
        num_nodes=num_nodes,
        num_models=num_models,
        num_instances=num_instances,
        analysis=analysis,
    )

    split_halves_kendall_tau(**shared, color=colors[0])
    bootstrapped_kendall_tau(**shared, color=colors[1])
    minibatch_kendall_w(**shared, num_folds=num_folds, color=colors[2])


if __name__ == "__main__":
    # NOTE: We ignore these datasets for the following reasons:
    # - Chatbot-Arena and Chatbot-Arena (New) don't have per-instance scores
    # - WildChat-10K only has evaluation results for two models

    datasets = [Dataset.DS_1000, Dataset.MATH, Dataset.MMLU]

    for dataset in datasets:
        main(dataset)
