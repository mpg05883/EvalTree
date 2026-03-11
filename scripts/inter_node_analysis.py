import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from tqdm import tqdm

from src.utils.capability_tree import (
    align_rankings,
    collect_nodes_dfs,
    load_capability_tree,
)
from src.utils.enums import Dataset
from src.utils.path import build_plot_path
from src.utils.plot import plot_histogram


def main(dataset: Dataset, min_instances: int) -> None:
    root = load_capability_tree(dataset)
    global_ranking = {model: score for model, score in root["ranking"]}
    nodes = collect_nodes_dfs(root, min_instances)

    num_models, num_nodes = len(global_ranking), len(nodes)

    # -------------------------------------------------------------------------
    # 1. Compute distribution of Kendall's Tau across nodes
    # -------------------------------------------------------------------------
    taus = np.zeros(len(nodes))

    kwargs = {
        "desc": "Computing Kendall's Taus",
        "total": len(nodes),
        "unit": "node",
    }

    for i, node in tqdm(enumerate(nodes), **kwargs):
        if node.ranking is None:
            continue
        aligned_global, aligned_local = align_rankings(
            global_ranking,
            node.ranking,
        )
        tau, _ = kendalltau(aligned_global, aligned_local)
        taus[i] = tau

    xlabel = "Kendall's Tau"
    ylabel = "Node Count"
    title = (
        f"{dataset}: Distribution of Kendall's Tau Across Nodes"
        f"\n({num_models} models, {num_nodes} nodes, {min_instances} instances minimum)"
    )
    annotate = True
    xlim = (-1, 1) if min(taus) < 0 else (0, 1)

    plot_histogram(
        taus,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        annotate=annotate,
        mean=taus.mean(),
        std=taus.std(),
        xlim=xlim,
    )

    analysis = "inter_node_analysis"
    plot_name = f"kendall-tau_histogram_min-instances={min_instances}"
    plot_path = build_plot_path(
        dataset,
        analysis=analysis,
        plot_name=plot_name,
    )
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

    # -------------------------------------------------------------------------
    # 2. Model Performance Analysis
    # -------------------------------------------------------------------------
    global_scores_dict = global_ranking

    node_to_scores = {}
    for i, node in enumerate(nodes):
        if node.ranking is None:
            continue
        node_to_scores[i] = node.ranking

    node_scores_df = pd.DataFrame(node_to_scores).T
    node_scores_df.index.name = "node"

    _, axes = plt.subplots(num_models, 1, figsize=(8, 3 * num_models))
    metric = Dataset(dataset).metric

    xlim = {
        Dataset.DS_1000: (0, 1),
        Dataset.MATH: (0, 1),
        Dataset.MMLU: (0, 1),
        Dataset.WILDCHAT_10K: (0, 1),
    }.get(dataset)

    for i, model in enumerate(node_scores_df.columns):
        plot_histogram(
            node_scores_df[model].dropna(),
            xlabel=f"Mean {metric.title()}",
            ylabel="Node Count",
            title=f"{model}",
            ax=axes[i] if num_models > 1 else axes,
            annotate=True,
            mean=global_scores_dict[model],
            mean_label=f"Benchmark-level {metric}",
            xlim=xlim,
        )

    plt.suptitle(
        f"{dataset}: Mean {metric.title()} Across Nodes"
        f"\n({num_nodes} nodes, {min_instances} instances minimum)",
        y=1.0,
    )
    plt.tight_layout()
    plot_name = f"performance_histogram_min-instances={min_instances}"
    plot_path = build_plot_path(dataset, analysis, plot_name)
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    min_instance_counts = [50]
    datasets = [d.value for d in Dataset]
    for dataset in datasets:
        for min_instances in min_instance_counts:
            main(dataset, min_instances)
