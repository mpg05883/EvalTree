import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau
from tqdm import tqdm

from src.utils.capability_tree import (
    collect_nodes_by_level,
    get_node_indices,
    load_capability_tree,
)
from src.utils.enums import Dataset
from src.utils.metrics import kendallw
from src.utils.model import load_model_scores
from src.utils.path import build_plot_path
from src.utils.plot import plot_histogram


def main(dataset: Dataset) -> None:
    model_scores_df = load_model_scores(dataset)
    models = model_scores_df.columns.tolist()

    root = load_capability_tree(dataset)
    nodes_by_level = collect_nodes_by_level(root)
    levels = sorted(nodes_by_level.keys())
    nodes_with_levels = [
        (node, level) for level in levels for node in nodes_by_level[level]
    ]
    nodes = [node for node, _ in nodes_with_levels]
    colors = sns.color_palette("tab10")

    num_instances = len(model_scores_df)
    num_models = len(models)
    num_nodes = len(nodes)
    num_trials = 500
    analysis = "intra_node_analysis"

    # -------------------------------------------------------------------------
    # 1. Split-Halves Kendall's Tau
    # -------------------------------------------------------------------------
    kwargs = {
        "desc": "Computing split-half Kendall's Taus",
        "total": num_nodes,
        "unit": "nodes",
    }

    split_half_results = []

    for node, level in tqdm(nodes_with_levels, **kwargs):
        indices = get_node_indices(node)
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

        split_half_results.append(
            {
                "node": node["capability"],
                "level": level,
                "mean_kendall_tau": np.mean(taus),
                "std_kendall_tau": np.std(taus),
                "num_instances": node_size,
            }
        )

    split_half_df = pd.DataFrame(split_half_results)

    xlim = (-1, 1) if min(split_half_df["mean_kendall_tau"]) < 0 else (0, 1)

    ylim = {
        Dataset.CHATBOT_ARENA: (0, num_nodes),
        Dataset.CHATBOT_ARENA_NEW: (0, num_nodes),
        Dataset.DS_1000: (0, num_nodes),
        Dataset.MATH: (0, num_nodes),
        Dataset.MMLU: (0, num_nodes // 2),
        Dataset.WILDCHAT_10K: (0, num_nodes),
    }[dataset]

    plot_histogram(
        split_half_df["mean_kendall_tau"],
        xlabel="Kendall's Tau",
        ylabel="Node Count",
        title=(
            f"{dataset}: Split-Halves Kendall's Tau Across Nodes"
            f"\n({num_models} models, {num_nodes} nodes, {num_instances} instances)"
        ),
        annotate=True,
        mean=split_half_df["mean_kendall_tau"].mean(),
        std=split_half_df["mean_kendall_tau"].std(),
        xlim=xlim,
        ylim=ylim,
        color=colors[0],
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
        level_data = split_half_df[split_half_df["level"] == level]["mean_kendall_tau"]
        plot_histogram(
            level_data,
            xlabel="Kendall's Tau",
            ylabel="Node Count",
            title=f"Level {level} ({len(level_data)} nodes)",
            ax=axes[i],
            annotate=True,
            mean=level_data.mean(),
            std=level_data.std(),
            xlim=xlim,
            color=colors[0],
        )

    plt.suptitle(
        f"{dataset}: Split-Halves Kendall's Tau by Level"
        f"\n({num_models} models, {num_nodes} nodes, {num_instances} instances)",
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

    # -------------------------------------------------------------------------
    # 2. Bootstrapped Kendall's Tau
    # -------------------------------------------------------------------------
    kwargs = {
        "desc": "Computing bootstrapped Kendall's Taus",
        "total": num_nodes,
        "unit": "nodes",
    }

    bootstrap_results = []

    for node, level in tqdm(nodes_with_levels, **kwargs):
        indices = get_node_indices(node)
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

        bootstrap_results.append(
            {
                "node": node["capability"],
                "level": level,
                "mean_kendall_tau": np.mean(taus),
                "std_kendall_tau": np.std(taus),
                "num_instances": n,
            }
        )

    bootstrap_df = pd.DataFrame(bootstrap_results)

    xlim = (-1, 1) if min(bootstrap_df["mean_kendall_tau"]) < 0 else (0, 1)

    ylim = {
        Dataset.CHATBOT_ARENA: (0, num_nodes),
        Dataset.CHATBOT_ARENA_NEW: (0, num_nodes),
        Dataset.DS_1000: (0, num_nodes),
        Dataset.MATH: (0, num_nodes),
        Dataset.MMLU: (0, num_nodes),
        Dataset.WILDCHAT_10K: (0, num_nodes),
    }[dataset]

    plot_histogram(
        bootstrap_df["mean_kendall_tau"],
        xlabel="Kendall's Tau",
        ylabel="Node Count",
        title=(
            f"{dataset}: Bootstrapped Kendall's Tau Across Nodes"
            f"\n({num_models} models, {num_nodes} nodes, {num_instances} instances)"
        ),
        annotate=True,
        mean=bootstrap_df["mean_kendall_tau"].mean(),
        std=bootstrap_df["mean_kendall_tau"].std(),
        xlim=xlim,
        ylim=ylim,
        color=colors[1],
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
        level_data = bootstrap_df[bootstrap_df["level"] == level]["mean_kendall_tau"]
        plot_histogram(
            level_data,
            xlabel="Kendall's Tau",
            ylabel="Node Count",
            title=f"Level {level} ({len(level_data)} nodes)",
            ax=axes[i],
            annotate=True,
            mean=level_data.mean(),
            std=level_data.std(),
            xlim=xlim,
            color=colors[1],
        )

    plt.suptitle(
        f"{dataset}: Bootstrapped Kendall's Tau by Level"
        f"\n({num_models} models, {num_nodes} nodes, {num_instances} instances)",
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

    # -------------------------------------------------------------------------
    # 3. Mini-Batch Kendall's W
    # -------------------------------------------------------------------------
    num_folds = 5

    kwargs = {
        "desc": "Computing mini-batch Kendall's Ws",
        "total": num_nodes,
        "unit": "nodes",
    }

    minibatch_w_results = []

    for node, level in tqdm(nodes_with_levels, **kwargs):
        indices = get_node_indices(node)
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

        minibatch_w_results.append(
            {
                "node": node["capability"],
                "level": level,
                "mean_kendallw": np.mean(kendallw_values),
                "std_kendallw": np.std(kendallw_values),
                "num_instances": node_size,
            }
        )

    minibatch_w_df = pd.DataFrame(minibatch_w_results)

    xlim = (0, 1)

    ylim = (0, num_nodes)

    plot_histogram(
        minibatch_w_df["mean_kendallw"],
        xlabel="Kendall's W",
        ylabel="Node Count",
        title=(
            f"{dataset}: Mini-Batch Kendall's W Across Nodes"
            f"\n({num_models} models, {num_nodes} nodes, {num_instances} instances)"
        ),
        annotate=True,
        mean=minibatch_w_df["mean_kendallw"].mean(),
        std=minibatch_w_df["mean_kendallw"].std(),
        xlim=xlim,
        ylim=ylim,
        color=colors[2],
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
        level_data = minibatch_w_df[minibatch_w_df["level"] == level]["mean_kendallw"]
        plot_histogram(
            level_data,
            xlabel="Kendall's W",
            ylabel="Node Count",
            title=f"Level {level} ({len(level_data)} nodes)",
            ax=axes[i],
            annotate=True,
            mean=level_data.mean(),
            std=level_data.std(),
            xlim=xlim,
            color=colors[2],
        )

    plt.suptitle(
        f"{dataset}: Mini-Batch Kendall's W by Level"
        f"\n({num_models} models, {num_nodes} nodes, {num_instances} instances)",
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


if __name__ == "__main__":
    # NOTE: We ignore these datasets for the following reasons:
    # - Chatbot-Arena and Chatbot-Arena (New) don't have per-instance scores
    # - WildChat-10K only has evaluation results for two models

    datasets = [Dataset.DS_1000, Dataset.MATH, Dataset.MMLU]

    for dataset in datasets:
        main(dataset)
