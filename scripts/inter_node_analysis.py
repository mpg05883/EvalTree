import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kendalltau
from tqdm import tqdm

from src.utils.capability_tree import (
    Node,
    align_rankings,
    collect_nodes,
    load_capability_tree,
)
from src.utils.enums import Dataset
from src.utils.path import build_data_path, build_plot_path
from src.utils.plot import plot_histogram, plot_stripplot

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def all_nodes_external_agreement_analysis(
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
        "desc": "Computing agreement with full benchmark",
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


def plot_all_nodes_external_agreement_histogram(
    df: pd.DataFrame,
    dataset: Dataset,
    num_models: int,
    num_nodes: int,
    min_instances: int,
    **kwargs,
) -> plt.Figure:
    """Plot the distribution of external Kendall's Tau values across nodes as a histogram.

    Takes the output of :func:`all_nodes_external_agreement_analysis` and visualises how
    consistently each node reproduces the global model ranking. The x-axis
    spans either (0, 1) or (-1, 1) depending on whether any negative tau
    values are present, and the histogram is annotated with the median and
    IQR.

    Args:
        df: DataFrame returned by :func:`all_nodes_external_agreement_analysis`.
        dataset: The dataset being analysed, used for the plot title.
        num_models: Number of models in the benchmark.
        num_nodes: Total number of qualifying nodes analysed.
        min_instances: Instance threshold used when collecting nodes.

    Returns:
        The matplotlib Figure containing the histogram.
    """
    data = df["kendall_tau"]
    xlim = (-1, 1) if min(data) < 0 else (0, 1)

    xlabel = r"Kendall's $\tau$"
    min_instance_label = r"$n_{\mathrm{min}}$"
    ylabel = "Number of Nodes"
    title = (
        f"{dataset.pretty_name}: Node Agreement with Full Benchmark"
        f"\n({num_models} models, {num_nodes} nodes, {min_instance_label}={min_instances})"
    )
    annotate = True
    median = data.median()
    median_label = f"Median {xlabel}"
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)

    return plot_histogram(
        data,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        annotate=annotate,
        median=median,
        median_label=median_label,
        q1=q1,
        q3=q3,
        xlim=xlim,
    )


def plot_per_level_external_agreement_stripplot(
    df: pd.DataFrame,
    dataset: Dataset,
    num_models: int,
    min_instances: int,
    **kwargs,
) -> plt.Figure:
    """Plot per-node Kendall's Tau values for each capability tree level as a strip plot.

    Takes the output of :func:`all_nodes_external_agreement_analysis` and produces a
    single strip plot where each x-axis tick corresponds to a capability tree
    level and every dot represents one node. Dots are color-coded by level
    using the tab10 palette. A horizontal dashed line marks the overall mean
    Kendall's Tau across all nodes and levels.

    Args:
        df: DataFrame returned by :func:`all_nodes_external_agreement_analysis`,
            which includes a ``depth`` column used to assign nodes to levels.
        dataset: The dataset being analysed, used for the plot title.
        num_models: Number of models in the benchmark.
        min_instances: Instance threshold used when collecting nodes.

    Returns:
        The matplotlib Figure containing the strip plot.
    """
    plot_df = df.copy()
    plot_df["level"] = plot_df["depth"].apply(lambda d: f"Level {int(d)}")
    order = [f"Level {int(d)}" for d in sorted(df["depth"].unique())]

    x = "level"
    y = "kendall_tau"
    xlabel = "Capability Tree Level"
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
    median_tau = df["kendall_tau"].median()

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
        median=median_tau,
        median_label=r"Full Benchmark Kendall's $\tau$",
    )


def all_nodes_performance_analysis(
    nodes: list[Node],
    **kwargs,
) -> pd.DataFrame:
    """Collect per-node mean model scores into a node × model DataFrame.

    For each node with a non-null ranking, stores its depth and dict of model
    scores as a row in the output DataFrame. Nodes without a ranking are
    skipped. The resulting DataFrame has one row per ranked node, a ``depth``
    column, and one additional column per model, making it straightforward to
    examine how each model's accuracy varies across nodes and levels.

    Args:
        nodes: Qualifying nodes collected from the capability tree.

    Returns:
        A DataFrame indexed by node (int position) with a ``depth`` column
        and one column per model containing each node's mean score.
    """
    tqdm_kwargs = {
        "desc": "Collecting node model scores",
        "total": len(nodes),
        "unit": "nodes",
    }

    node_to_scores = {}
    for i, node in tqdm(enumerate(nodes), **tqdm_kwargs):
        if node.ranking is None:
            continue
        node_to_scores[i] = {"depth": node.depth, **node.ranking}

    df = pd.DataFrame(node_to_scores).T
    df.index.name = "node"
    return df


def plot_all_nodes_performance_stripplot(
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

    models = [col for col in df.columns if col != "depth"]
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
        f"{dataset.pretty_name}: Node {dataset.metric.title()} vs Full Benchmark"
        f"\n({num_nodes} nodes, {min_instance_label}={min_instances})"
    )
    hue = "model"
    order = models
    palette = "tab10"
    x_means = global_ranking
    x_means_label = f"Full Benchmark {dataset.metric.title()}"
    figsize = (max(8, num_models * 1.5), 5)
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


def plot_per_level_performance_stripplot(
    all_nodes_performance_df: pd.DataFrame,
    dataset: Dataset,
    global_ranking: dict[str, float],
    num_models: int,
    min_instances: int,
    size: int = 10,
    tick_fontsize: int = 12,
    legend_fontsize: int = 14,
    label_fontsize: int = 14,
    title_fontsize: int = 16,
    suptitle_fontsize: int = 18,
    **kwargs,
) -> plt.Figure:
    """Plot per-node model scores for each level, faceted by level.

    Produces one strip plot panel per capability tree level, arranged in a
    single column. Within each panel the x-axis shows models and the y-axis
    shows mean node scores, with one jittered dot per node. A short horizontal
    black line marks the full-benchmark score for each model, making it easy
    to see which nodes fall above or below the global baseline.

    Args:
        all_nodes_performance_df: DataFrame returned by
            :func:`all_nodes_performance_analysis`, which includes a
            ``depth`` column used to assign each node to a level.
        dataset: The dataset being analysed, used for axis labels and title.
        global_ranking: Benchmark-level model scores used as per-model
            reference lines.
        num_models: Number of models, used to size the figure width.
        min_instances: Instance threshold used when collecting nodes.

    Returns:
        The matplotlib Figure containing one strip plot panel per level.
    """
    df = all_nodes_performance_df
    ylim = {
        Dataset.DS_1000: (0, 1),
        Dataset.MATH: (0, 1),
        Dataset.MMLU: (0, 1),
        Dataset.WILDCHAT_10K: (0, 1),
    }.get(dataset)

    models = [col for col in df.columns if col != "depth"]
    depths = sorted(df["depth"].unique())
    num_levels = len(depths)

    long_df = df.reset_index().melt(
        id_vars=["node", "depth"],
        value_vars=models,
        var_name="model",
        value_name=dataset.metric,
    )
    long_df["level"] = long_df["depth"].apply(lambda d: f"Level {int(d)}")

    min_instance_label = r"$n_{\mathrm{min}}$"

    fig, axes = plt.subplots(
        num_levels,
        1,
        figsize=(max(8, num_models * 1.5), 4 * num_levels),
        squeeze=False,
    )

    for i, depth in enumerate(depths):
        level_label = f"Level {int(depth)}"
        level_data = long_df[long_df["level"] == level_label]

        plot_stripplot(
            data=level_data,
            size=size,
            x="model",
            y=dataset.metric,
            xlabel="",
            ylabel=dataset.metric.title(),
            title=level_label,
            hue="model",
            order=models,
            palette="tab10",
            ax=axes[i, 0],
            x_means=global_ranking,
            x_means_label=f"Full Benchmark {dataset.metric.title()}",
            ylim=ylim,
            rotation=30,
            tick_fontsize=tick_fontsize,
            legend_fontsize=legend_fontsize,
            label_fontsize=label_fontsize,
            title_fontsize=title_fontsize,
        )

    plt.suptitle(
        f"{dataset.pretty_name}: Node {dataset.metric.title()} vs Full Benchmark (Per Level)"
        f"\n({min_instance_label}={min_instances})",
        y=1.0,
        fontsize=suptitle_fontsize,
    )
    plt.tight_layout()
    return fig


def main(dataset: Dataset, min_instances: int, experiment: str) -> None:
    root = load_capability_tree(dataset)
    global_ranking = {model: score for model, score in root["ranking"]}
    nodes = collect_nodes(root, min_instances)
    num_models, num_nodes = len(global_ranking), len(nodes)

    shared = dict(
        dataset=dataset,
        global_ranking=global_ranking,
        nodes=nodes,
        num_models=num_models,
        num_nodes=num_nodes,
        min_instances=min_instances,
        experiment=experiment,
    )

    all_nodes_external_agreement_df = all_nodes_external_agreement_analysis(**shared)
    data_name = f"all_nodes__external_agreement__{min_instances=}"
    data_path = build_data_path(dataset, experiment, data_name)
    all_nodes_external_agreement_df.to_csv(data_path)
    logger.info(f"Saved data to {data_path}")

    all_nodes_external_agreement_fig = plot_all_nodes_external_agreement_histogram(
        all_nodes_external_agreement_df,
        **shared,
    )
    plot_name = f"all_nodes__external_agreement__histogram__{min_instances=}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    all_nodes_external_agreement_fig.savefig(plot_path)
    plt.close(all_nodes_external_agreement_fig)
    logger.info(f"Saved plot to {plot_path}")

    per_level_external_agreement_fig = plot_per_level_external_agreement_stripplot(
        all_nodes_external_agreement_df,
        **shared,
    )
    plot_name = f"per_level__external_agreement__stripplot__{min_instances=}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    per_level_external_agreement_fig.savefig(plot_path)
    plt.close(per_level_external_agreement_fig)
    logger.info(f"Saved plot to {plot_path}")

    all_nodes_performance_df = all_nodes_performance_analysis(**shared)
    data_name = f"all_nodes__performance__{min_instances=}"
    data_path = build_data_path(dataset, experiment, data_name)
    all_nodes_performance_df.to_csv(data_path)
    logger.info(f"Saved data to {data_path}")

    all_nodes_performance_fig = plot_all_nodes_performance_stripplot(
        all_nodes_performance_df,
        **shared,
    )
    plot_name = f"all_nodes__performance__stripplot__{min_instances=}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    all_nodes_performance_fig.savefig(plot_path)
    plt.close(all_nodes_performance_fig)
    logger.info(f"Saved plot to {plot_path}")

    per_level_performance_fig = plot_per_level_performance_stripplot(
        all_nodes_performance_df,
        **shared,
    )
    plot_name = f"per_level__performance__stripplot__{min_instances=}"
    plot_path = build_plot_path(dataset, experiment, plot_name)
    per_level_performance_fig.savefig(plot_path)
    plt.close(per_level_performance_fig)
    logger.info(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    datasets = [Dataset(d.value) for d in Dataset]
    experiment = Path(__file__).stem

    for i, dataset in enumerate(datasets):
        one_tenth = dataset.num_instances // 10
        min_instance_values = [50]
        for min_instances in min_instance_values:
            print(
                f"{'-'*80} Dataset {i+1}/{len(datasets)}: {dataset.pretty_name}, "
                f"{min_instances=} {'-'*80}"
            )
            main(dataset, min_instances, experiment)
