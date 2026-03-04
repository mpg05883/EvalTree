import argparse
import json

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau

from utils.capability_tree import align_rankings, collect_nodes
from utils.path import build_plot_path, resolve_capability_tree_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset name (e.g. DS-1000, MATH, MMLU)")
    args = parser.parse_args()

    data_path = resolve_capability_tree_path(args.dataset)
    with open(data_path) as f:
        root = json.load(f)

    global_ranking = root["ranking"]
    nodes = collect_nodes(root)
    print(f"Found {len(nodes)} qualifying nodes (non-root, size > 50)")

    taus = []
    for node in nodes:
        global_vec, local_vec = align_rankings(global_ranking, node["ranking"])
        tau, _ = kendalltau(global_vec, local_vec)
        taus.append(tau)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(taus, ax=ax)
    ax.set_xlabel("Kendall's Tau")
    ax.set_ylabel("Node Count")
    ax.set_title(f"Distribution of Kendall's Tau – {args.dataset}")
    plt.tight_layout()

    plot_path = build_plot_path(
        args.dataset, "inter_node_analysis", "kendall_tau_distribution"
    )
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
