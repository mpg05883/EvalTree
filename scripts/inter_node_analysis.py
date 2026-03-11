from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kendalltau
from tqdm import tqdm

from src.utils.capability_tree import (
    Level,
    Node,
    align_rankings,
    collect_levels,
    collect_nodes,
    load_capability_tree,
)
from src.utils.enums import Dataset
from src.utils.path import build_data_path, build_plot_path
from src.utils.plot import plot_histogram, plot_stripplot


def all_nodes_agreement_analysis(
    nodes: list[Node],
    global_ranking: dict[str, float],
    **kwargs,
) -> pd.DataFrame:
    """Measure how consistently each node's local model ranking matches the
    global model ranking across all nodes in the capability tree.

    For each node with a non-null ranking, computes Kendall's Tau between the
    global benchmark ranking and the node-level ranking. A tau near 1 means
    the node preserves the global ordering of models; a tau near -1 means it
    reverses it. Nodes without a ranking (``None``) are silently skipped and
    excluded from the returned DataFrame.

    Args:
        nodes: Qualifying nodes collected from the capability tree.
        global_ranking: Benchmark-level model scores as a dict mapping model
            name to mean score, ordered descending by score.

    Returns:
        A DataFrame with one row per ranked node and columns
        ``["size", "depth", "capability", "distinction", "kendall_tau"]``.
    """
    tqdm_kwargs = {
        "desc": "Computing Kendall's taus",
        "total": len(nodes),
        "unit": "nodes",
    }

    results = []
    for node in tqdm(nodes, **tqdm_kwargs):
        if node.ranking is None:
            continue
        aligned_global, aligned_local = align_rankings(global_ranking, node.ranking)
        tau, _ = kendalltau(aligned_global, aligned_local)
        results.append(
            {
                "size": node.size,
                "depth": node.depth,
                "capability": node.capability,
                "distinction": node.distinction,
                "kendall_tau": tau,
            }
        )

    return pd.DataFrame(results)


def plot_all_nodes_agreement_histogram(
    df: pd.DataFrame,
    dataset: Dataset,
    num_models: int,
    num_nodes: int,
    min_instances: int,
    **kwargs,
) -> plt.Figure:
    """Plot the distribution of Kendall's Tau values across nodes as a histogram.

    Takes the output of :func:`all_nodes_agreement_analysis` and visualises how
    consistently each node reproduces the global model ranking. The x-axis
    spans either (0, 1) or (-1, 1) depending on whether any negative tau
    values are present, and the histogram is annotated with the mean and
    standard deviation.

    Args:
        df: DataFrame returned by :func:`all_nodes_agreement_analysis`.
        dataset: The dataset being analysed, used for the plot title.
        num_models: Number of models in the benchmark.
        num_nodes: Total number of qualifying nodes analysed.
        min_instances: Instance threshold used when collecting nodes.

    Returns:
        The matplotlib Figure containing the histogram.
    """
    data = df["kendall_tau"]
    xlim = (-1, 1) if min(data) < 0 else (0, 1)
    ylim = {
        Dataset.CHATBOT_ARENA: (0, num_nodes),
        Dataset.CHATBOT_ARENA_NEW: (0, num_nodes),
        Dataset.DS_1000: (0, num_nodes),
        Dataset.MATH: (0, num_nodes),
        Dataset.MMLU: (0, num_nodes),
        Dataset.WILDCHAT_10K: (0, num_nodes),
    }.get(dataset)
    xlabel = r"Kendall's $\tau$"
    min_instance_label = r"$n_{\mathrm{min}}$"
    ylabel = "Number of Nodes"
    title = (
        f"{dataset.pretty_name}: Node Agreement with Full Benchmark (All Nodes)"
        f"\n({num_models} models, {num_nodes} nodes, {min_instance_label}={min_instances})"
    )
    annotate = True
    mean = data.mean()
    mean_label = f"Mean {xlabel}"
    std = data.std()

    return plot_histogram(
        data,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        annotate=annotate,
        mean=mean,
        mean_label=mean_label,
        std=std,
        xlim=xlim,
        ylim=ylim,
    )


def plot_per_level_agreement_strip_plot(
    df: pd.DataFrame,
    dataset: Dataset,
    num_models: int,
    min_instances: int,
    **kwargs,
) -> plt.Figure:
    """Plot per-node Kendall's Tau values for each capability tree level as a strip plot.

    Takes the output of :func:`all_nodes_agreement_analysis` and produces a
    single strip plot where each x-axis tick corresponds to a capability tree
    level and every dot represents one node. Dots are color-coded by level
    using the tab10 palette. A horizontal dashed line marks the overall mean
    Kendall's Tau across all nodes and levels.

    Args:
        df: DataFrame returned by :func:`all_nodes_agreement_analysis`,
            which includes a ``depth`` column used to assign nodes to levels.
        dataset: The dataset being analysed, used for the plot title.
        num_models: Number of models in the benchmark.
        min_instances: Instance threshold used when collecting nodes.

    Returns:
        The matplotlib Figure containing the strip plot.
    """
    plot_df = df.copy()
    plot_df["level"] = plot_df["depth"].apply(lambda d: f"Level {d}")
    order = [f"Level {d}" for d in sorted(df["depth"].unique())]

    x = "level"
    y = "kendall_tau"
    xlabel = "Level"
    ylabel = r"Kendall's $\tau$"
    min_instance_label = r"$n_{\mathrm{min}}$"
    title = (
        f"{dataset.pretty_name}: Node Agreement with Full Benchmark (Per Level)"
        f"\n({num_models} models, {min_instance_label}={min_instances})"
    )
    hue = "level"
    order = order
    palette = "tab10"
    ylim = (-1, 1) if df["kendall_tau"].min() < 0 else (0, 1)
    rotation = 30

    return plot_stripplot(
        data=plot_df,
        x=x,
        y=y,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        hue=hue,
        order=order,
        palette=palette,
        ylim=ylim,
        rotation=rotation,
    )


def all_nodes_performance_analysis(
    nodes: list[Node],
    **kwargs,
) -> pd.DataFrame:
    """Collect per-node mean model scores into a node × model DataFrame.

    For each node with a non-null ranking, stores its dict of model scores as
    a row in the output DataFrame. Nodes without a ranking are skipped. The
    resulting DataFrame has one row per ranked node and one column per model,
    making it straightforward to examine how each model's accuracy varies
    across nodes.

    Args:
        nodes: Qualifying nodes collected from the capability tree.

    Returns:
        A DataFrame indexed by node (int position) with one column per model,
        containing each node's mean score for that model.
    """
    node_to_scores = {}
    for i, node in enumerate(nodes):
        if node.ranking is None:
            continue
        node_to_scores[i] = node.ranking

    df = pd.DataFrame(node_to_scores).T
    df.index.name = "node"
    return df


def plot_all_nodes_performance_strip_plot(
    df: pd.DataFrame,
    dataset: Dataset,
    global_ranking: dict[str, float],
    num_models: int,
    num_nodes: int,
    min_instances: int,
    **kwargs,
) -> plt.Figure:
    """Plot per-node accuracy for each model as a strip plot.

    Renders one column per model on the x-axis, with each dot representing a
    single node's mean accuracy on the y-axis. A short horizontal black line
    marks the benchmark-level mean accuracy for each model, making it easy to
    see which nodes fall above or below the overall benchmark score.

    Args:
        df: DataFrame returned by :func:`all_nodes_performance_analysis`.
        dataset: The dataset being analysed, used for axis labels and title.
        global_ranking: Benchmark-level model scores used as reference lines.
        num_models: Number of models, used for the figure width.
        num_nodes: Total number of qualifying nodes, used for the plot title.
        min_instances: Instance threshold used when collecting nodes.

    Returns:
        The matplotlib Figure containing the strip plot.
    """
    ylim = {
        Dataset.DS_1000: (0, 1),
        Dataset.MATH: (0, 1),
        Dataset.MMLU: (0, 1),
        Dataset.WILDCHAT_10K: (0, 1),
    }.get(dataset)

    models = list(df.columns)
    long_df = df.reset_index().melt(
        id_vars=["node"],
        value_vars=models,
        var_name="model",
        value_name=dataset.metric,
    )

    x = "model"
    y = dataset.metric
    xlabel = ""
    ylabel = dataset.metric.title()
    min_instance_label = r"$n_{\mathrm{min}}$"
    title = (
        f"{dataset.pretty_name}: Node {dataset.metric.title()} vs Full Benchmark (All Nodes)"
        f"\n({num_nodes} nodes, {min_instance_label}={min_instances})"
    )
    hue = "model"
    order = models
    palette = "tab10"
    x_means = global_ranking
    x_means_label = f"Full Benchmark {dataset.metric.title()}"
    figsize = (max(8, num_models * 1.5), 5)
    ylim = ylim
    rotation = 30

    return plot_stripplot(
        data=long_df,
        x=x,
        y=y,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        hue=hue,
        order=order,
        palette=palette,
        x_means=x_means,
        x_means_label=x_means_label,
        figsize=figsize,
        ylim=ylim,
        rotation=rotation,
    )


def per_level_performance_analysis(
    levels: list[Level],
    **kwargs,
) -> pd.DataFrame:
    """Collect per-node mean model scores into a node × model DataFrame, with
    each row tagged by the capability tree level the node belongs to.

    For each node with a non-null ranking in each level, stores its dict of
    model scores and its level depth as a row in the output DataFrame. Nodes
    without a ranking are skipped. The ``depth`` column allows the per-level
    plotting function to group rows into one subplot column per level.

    Args:
        levels: Qualifying levels collected from the capability tree, each
            holding the nodes at a single depth.

    Returns:
        A DataFrame with one row per ranked node, a ``depth`` column
        indicating the level, and one additional column per model containing
        each node's mean score for that model.
    """
    rows = []
    for level in levels:
        for node in level.nodes:
            if node.ranking is None:
                continue
            rows.append({"depth": level.depth, **node.ranking})

    df = pd.DataFrame(rows)
    df.index.name = "node"
    return df


def plot_per_level_performance_strip_plot(
    df: pd.DataFrame,
    levels: list[Level],
    dataset: Dataset,
    global_ranking: dict[str, float],
    num_models: int,
    min_instances: int,
    **kwargs,
) -> plt.Figure:
    """Plot per-node accuracy for each model, faceted by capability tree level.

    Produces one strip plot panel per capability tree level, arranged in a
    single column. Within each panel the x-axis shows models and the y-axis
    shows mean node accuracy, with one jittered dot per node. A short
    horizontal black line marks the benchmark-level mean for each model.
    Each panel title reports the number of nodes and the number of unique
    dataset instances at that level.

    Args:
        df: DataFrame returned by :func:`per_level_performance_analysis`,
            which includes a ``depth`` column used to assign each node to a
            level.
        levels: Qualifying levels from the capability tree, used to look up
            the node count and deduplicated instance count for each level.
        dataset: The dataset being analysed, used for axis labels and title.
        global_ranking: Benchmark-level model scores used as reference lines.
        num_models: Number of models, used for the figure width.
        min_instances: Instance threshold used when collecting nodes.

    Returns:
        The matplotlib Figure containing one strip plot panel per level.
    """
    metric = dataset.metric
    ylim = {
        Dataset.DS_1000: (0, 1),
        Dataset.MATH: (0, 1),
        Dataset.MMLU: (0, 1),
        Dataset.WILDCHAT_10K: (0, 1),
    }.get(dataset)

    nodes_by_depth = {level.depth: level.num_nodes for level in levels}
    instances_by_depth = {level.depth: level.num_instances for level in levels}

    models = [col for col in df.columns if col != "depth"]
    long_df = df.reset_index().melt(
        id_vars=["node", "depth"],
        value_vars=models,
        var_name="model",
        value_name=metric,
    )
    depths = sorted(df["depth"].unique())
    num_levels = len(depths)
    min_instance_label = r"$n_{\mathrm{min}}$"

    fig, axes = plt.subplots(
        num_levels,
        1,
        figsize=(max(8, num_models * 1.5), 4 * num_levels),
        squeeze=False,
    )

    for i, depth in enumerate(depths):
        level_data = long_df[long_df["depth"] == depth]
        num_nodes_at_level = nodes_by_depth.get(depth, 0)
        num_instances_at_level = instances_by_depth.get(depth, 0)

        x = "model"
        y = metric
        xlabel = ""
        ylabel = metric.title()
        title = f"Level {depth} ({num_nodes_at_level} nodes, {num_instances_at_level} instances)"
        hue = "model"
        order = models
        palette = "tab10"
        x_means = global_ranking
        x_means_label = f"Full Benchmark {metric.title()}"
        ylim = ylim
        rotation = 30

        plot_stripplot(
            data=level_data,
            x=x,
            y=y,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            hue=hue,
            order=order,
            palette=palette,
            ax=axes[i, 0],
            x_means=x_means,
            x_means_label=x_means_label,
            ylim=ylim,
            rotation=rotation,
        )

    plt.suptitle(
        f"{dataset.pretty_name}: Node {metric.title()} vs Full Benchmark (Per Level)"
        f"\n({min_instance_label}={min_instances})",
        y=1.0,
    )
    plt.tight_layout()
    return fig


def main(dataset: Dataset, min_instances: int, experiment: str) -> None:
    root = load_capability_tree(dataset)
    global_ranking = {model: score for model, score in root["ranking"]}
    nodes = collect_nodes(root, min_instances)
    levels = collect_levels(root, min_instances)
    num_models = len(global_ranking)
    num_nodes = len(nodes)
    num_levels = len(levels)

    shared = dict(
        dataset=dataset,
        global_ranking=global_ranking,
        nodes=nodes,
        levels=levels,
        num_models=num_models,
        num_nodes=num_nodes,
        num_levels=num_levels,
        min_instances=min_instances,
        experiment=experiment,
    )

    all_nodes_agreement_df = all_nodes_agreement_analysis(**shared)
    data_name = f"all-nodes_agreement_min-instances={min_instances}"
    data_path = build_data_path(dataset, experiment, data_name)
    all_nodes_agreement_df.to_csv(data_path)
    print(f"Saved data to {data_path}")

    all_nodes_agreement_fig = plot_all_nodes_agreement_histogram(
        all_nodes_agreement_df,
        **shared,
    )
    plot_name = f"all-nodes_agreement_histogram_min-instances={min_instances}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    all_nodes_agreement_fig.savefig(plot_path)
    plt.close(all_nodes_agreement_fig)
    print(f"Saved plot to {plot_path}")

    per_level_agreement_fig = plot_per_level_agreement_strip_plot(
        all_nodes_agreement_df,
        **shared,
    )
    plot_name = f"per-level_agreement_strip-plot_min-instances={min_instances}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    per_level_agreement_fig.savefig(plot_path)
    plt.close(per_level_agreement_fig)
    print(f"Saved plot to {plot_path}")

    all_nodes_performance_df = all_nodes_performance_analysis(**shared)
    data_name = f"all-nodes_performance_min-instances={min_instances}"
    data_path = build_data_path(dataset, experiment, data_name)
    all_nodes_performance_df.to_csv(data_path)
    print(f"Saved data to {data_path}")

    all_nodes_performance_fig = plot_all_nodes_performance_strip_plot(
        all_nodes_performance_df,
        **shared,
    )
    plot_name = f"all-nodes_performance_strip-plot_min-instances={min_instances}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    all_nodes_performance_fig.savefig(plot_path)
    plt.close(all_nodes_performance_fig)
    print(f"Saved plot to {plot_path}")

    per_level_performance_df = per_level_performance_analysis(**shared)
    data_name = f"per-level_performance_min-instances={min_instances}"
    data_path = build_data_path(dataset, experiment, data_name)
    per_level_performance_df.to_csv(data_path)
    print(f"Saved data to {data_path}")

    per_level_performance_fig = plot_per_level_performance_strip_plot(
        per_level_performance_df,
        **shared,
    )
    plot_name = f"per-level_performance_strip-plot_min-instances={min_instances}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    per_level_performance_fig.savefig(plot_path)
    plt.close(per_level_performance_fig)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    min_instance_values = [0, 50]
    datasets = [Dataset(d.value) for d in Dataset]
    experiment = Path(__file__).stem

    for dataset in datasets:
        for min_instances in min_instance_values:
            main(dataset, min_instances, experiment)
