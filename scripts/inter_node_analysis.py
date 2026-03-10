from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from tqdm import tqdm

from src.utils.capability_tree import (
    align_rankings,
    collect_nodes,
    load_capability_tree,
)
from src.utils.enums import Dataset
from src.utils.path import build_plot_path
from src.utils.plot import plot_barplot, plot_histogram


def main(dataset: Dataset, min_instances: int) -> None:
    root = load_capability_tree(dataset)
    global_ranking = root["ranking"]
    nodes = collect_nodes(root, min_instances)

    num_models, num_nodes = len(global_ranking), len(nodes)

    # -------------------------------------------------------------------------
    # 1. Kendall's Tau
    # -------------------------------------------------------------------------\
    taus = np.zeros(len(nodes))

    kwargs = {
        "desc": "Computing Kendall's Taus",
        "total": len(nodes),
        "unit": "node",
    }

    for i, node in tqdm(enumerate(nodes), **kwargs):
        if node["ranking"] is None:
            continue
        aligned_global, aligned_local = align_rankings(
            global_ranking,
            node["ranking"],
        )
        tau, _ = kendalltau(aligned_global, aligned_local)
        taus[i] = tau

    xlabel = "Kendall's Tau"
    ylabel = "Node Count"
    title = (
        f"{dataset}: Distribution of Kendall's Tau Across Nodes"
        f"\n({num_models} models, {num_nodes} nodes)"
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
    # 2. Model Rankings by Cumulative Node Size
    # -------------------------------------------------------------------------
    instance_counter = Counter()
    for node in nodes:
        if node["ranking"] is None:
            continue
        ranking = " > ".join(name for name, _ in node["ranking"])
        instance_counter[ranking] += node["size"]

    ranking_counts_df = pd.DataFrame(
        instance_counter.most_common(),
        columns=["ranking", "instance_count"],
    )

    print(f"Number of unique rankings: {(num_rankings:=len(ranking_counts_df))}")
    ranking_counts_df.head()

    # TODO: Explain why there are repeat instances in the cumulative node sizes
    topk = min(10, len(instance_counter))
    num_instances = root["size"]

    # Rename rankings to prevent text from overflowing
    plot_df = ranking_counts_df.head(topk).copy()
    plot_df["ranking"] = [f"Ranking {i + 1}" for i in range(topk)]

    x = "ranking"
    y = "instance_count"
    xlabel = "Model Ranking"
    ylabel = "Cumulative Node Size"
    title = (
        f"{dataset}: Top {topk} Model Rankings by Cumulative Node Size"
        f"\n({num_instances} instances, {num_nodes} nodes, {num_models} models, {num_rankings} rankings)"
    )
    annotate = True

    plot_barplot(
        plot_df,
        x=x,
        y=y,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        annotate=annotate,
        mean=plot_df["instance_count"].mean(),
        std=plot_df["instance_count"].std(),
    )

    plot_name = f"top-{topk}-rankings_barplot_min-instances={min_instances}"
    plot_path = build_plot_path(
        dataset,
        analysis=analysis,
        plot_name=plot_name,
    )
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    min_instance_counts = [50]
    datasets = [d.value for d in Dataset]
    for dataset in datasets:
        for min_instances in min_instance_counts:
            main(dataset, min_instances)
