import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau
from tqdm import tqdm

from src.utils.capability_tree import (
    collect_nodes,
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
    nodes = collect_nodes(root)
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

    for node in tqdm(nodes, **kwargs):
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
                "mean_kendall_tau": np.mean(taus),
                "std_kendall_tau": np.std(taus),
                "num_instances": node_size,
            }
        )

    split_half_df = pd.DataFrame(split_half_results)

    xlim = {
        Dataset.CHATBOT_ARENA: (-1, 1),
        Dataset.CHATBOT_ARENA_NEW: (-1, 1),
        Dataset.DS_1000: (-1, 1),
        Dataset.MATH: (-1, 1),
        Dataset.MMLU: (-1, 1),
        Dataset.WILDCHAT_10K: (-1, 1),
    }[dataset]

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
    )
    plt.savefig(file_path)
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

    for node in tqdm(nodes, **kwargs):
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
                "mean_kendall_tau": np.mean(taus),
                "std_kendall_tau": np.std(taus),
                "num_instances": n,
            }
        )

    bootstrap_df = pd.DataFrame(bootstrap_results)

    xlim = {
        Dataset.CHATBOT_ARENA: (-1, 1),
        Dataset.CHATBOT_ARENA_NEW: (-1, 1),
        Dataset.DS_1000: (-1, 1),
        Dataset.MATH: (-1, 1),
        Dataset.MMLU: (-1, 1),
        Dataset.WILDCHAT_10K: (-1, 1),
    }[dataset]

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
    )
    plt.savefig(file_path)
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

    for node in tqdm(nodes, **kwargs):
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
                "mean_kendallw": np.mean(kendallw_values),
                "std_kendallw": np.std(kendallw_values),
                "num_instances": node_size,
            }
        )

    minibatch_w_df = pd.DataFrame(minibatch_w_results)

    xlim = {
        Dataset.CHATBOT_ARENA: (0, 1),
        Dataset.CHATBOT_ARENA_NEW: (0, 1),
        Dataset.DS_1000: (0, 1),
        Dataset.MATH: (0, 1),
        Dataset.MMLU: (0, 1),
        Dataset.WILDCHAT_10K: (0, 1),
    }[dataset]

    ylim = {
        Dataset.CHATBOT_ARENA: (0, num_nodes),
        Dataset.CHATBOT_ARENA_NEW: (0, num_nodes),
        Dataset.DS_1000: (0, num_nodes),
        Dataset.MATH: (0, num_nodes),
        Dataset.MMLU: (0, num_nodes // 2),
        Dataset.WILDCHAT_10K: (0, num_nodes),
    }[dataset]

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
    )
    plt.savefig(file_path)
    print(f"Saved plot to {file_path}")
    plt.close()


if __name__ == "__main__":
    # NOTE: We ignore these datasets for the following reasons:
    # - Chatbot-Arena and Chatbot-Arena (New) don't have per-instance scores
    # - WildChat-10K only has evaluation results for two models

    datasets = [Dataset.DS_1000, Dataset.MATH, Dataset.MMLU]
    
    for dataset in datasets:
        main(dataset)
