import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import kendalltau
from tqdm import tqdm

from utils.capability_tree import (
    align_rankings,
    collect_nodes,
    load_capability_tree,
)
from utils.data import Dataset
from utils.path import build_plot_path


def main(dataset: Dataset, min_instances: int) -> None:
    root = load_capability_tree(dataset)
    nodes = collect_nodes(root, min_instances)
    global_ranking = root["ranking"]
    num_models = len(global_ranking)
    print(f"Number of models: {num_models}")
    taus = []

    kwargs = {
        "desc": "Computing Kendall's Taus",
        "total": len(nodes),
        "unit": "node",
    }

    for node in tqdm(nodes, **kwargs):
        if node["ranking"] is None:
            continue
        global_vec, local_vec = align_rankings(global_ranking, node["ranking"])
        tau, _ = kendalltau(global_vec, local_vec)
        taus.append(tau)

    # Compute mean and std
    taus_arr = np.array(taus)
    mean_tau = taus_arr.mean()
    std_tau = taus_arr.std()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(taus, ax=ax, bins=10)

    # Annotate bar heights
    for bar in ax.patches:
        if (height := bar.get_height()) == 0:
            continue
        ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Add vertical line at mean and shaded region for +/- 1 std (if not NaN)
    if not (np.isnan(mean_tau) and np.isnan(std_tau)):
        ax.axvline(
            mean_tau,
            color="red",
            linestyle="--",
            label=f"Mean: {mean_tau:.3g}",
        )
        ax.axvspan(
            mean_tau - std_tau,
            mean_tau + std_tau,
            alpha=0.2,
            color="red",
            label=f"±1 Std: {std_tau:.3g}",
        )
        ax.legend()

    ax.set_xlabel("Kendall's Tau")
    ax.set_ylabel("Node Count")
    ax.set_title(
        f"{dataset}: Distribution of Kendall's Tau Across Nodes"
        f"\n({num_models} models, {len(nodes)} nodes, min_instances={min_instances})"
    )
    plt.tight_layout()

    plot_path = build_plot_path(
        dataset,
        analysis="inter_node_analysis",
        plot_name=f"kendall_tau_distribution-min_instances={min_instances}",
    )
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    min_instance_counts = [0, 50]
    datasets = [d.value for d in Dataset]
    for dataset in datasets:
        for min_instances in min_instance_counts:
            main(dataset, min_instances)
