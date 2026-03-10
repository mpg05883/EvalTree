import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata
from tqdm import tqdm

from src.utils.data import (
    get_metadata_mask,
    get_unique_metadata_values,
    load_dataset,
)
from src.utils.enums import Dataset
from src.utils.model import load_model_scores
from src.utils.path import build_plot_path
from src.utils.plot import plot_histogram


def main(dataset: Dataset) -> None:
    dataset_df = load_dataset(dataset)
    model_scores_df = load_model_scores(dataset)
    assert len(dataset_df) == len(model_scores_df)

    # -------------------------------------------------------------------------
    # 1. Compute number of subsets that differ from global ranking
    # -------------------------------------------------------------------------
    global_scores = model_scores_df.mean()
    global_ranking = rankdata(global_scores)

    num_different = 0

    subset_col = dataset.subset_col
    is_ds_1000 = dataset == Dataset.DS_1000
    subsets = (
        get_unique_metadata_values(dataset_df, subset_col)
        if is_ds_1000
        else dataset_df[subset_col].unique()
    )

    for subset in subsets:
        mask = (
            get_metadata_mask(dataset_df, subset_col, subset)
            if is_ds_1000
            else dataset_df[subset_col] == subset
        )
        subset_scores = model_scores_df[mask].mean()
        subset_ranking = rankdata(subset_scores)
        different_ranking = not np.array_equal(subset_ranking, global_ranking)
        num_different += 1 if different_ranking else 0

    plural = "libraries" if is_ds_1000 else f"{subset_col}s"
    print(
        f"Total {plural}: {(num_subsets := len(subsets))}. "
        f"{plural.capitalize()} that differ from global ranking: {num_different}"
    )

    # -------------------------------------------------------------------------
    # 2. Compute distribution of Kendall's Tau across subsets
    # -------------------------------------------------------------------------
    kwargs = {
        "desc": "Computing Kendall's taus",
        "total": num_subsets,
        "unit": "subset",
    }

    kendall_tau_results = []

    for subset in tqdm(subsets, **kwargs):
        mask = (
            get_metadata_mask(dataset_df, subset_col, subset)
            if is_ds_1000
            else dataset_df[subset_col] == subset
        )
        subset_scores = model_scores_df[mask].mean()
        kendall_tau, _ = kendalltau(global_scores, subset_scores)
        kendall_tau_results.append(
            {
                "subset": subset,
                "scores": subset_scores,
                "kendall_tau": kendall_tau,
                "num_instances": len(model_scores_df[mask]),
            }
        )

    kendall_tau_df = pd.DataFrame(kendall_tau_results)

    num_models = len(model_scores_df.columns)
    num_instances = len(dataset_df)

    xlim = {
        Dataset.DS_1000: (0, 1),
        Dataset.MATH: (0, 1),
        Dataset.MMLU: (0, 1),
    }[dataset]

    ylim = {
        Dataset.DS_1000: (0, num_subsets),
        Dataset.MATH: (0, num_subsets),
        Dataset.MMLU: (0, num_subsets // 2),
    }[dataset]

    plot_histogram(
        kendall_tau_df["kendall_tau"],
        xlabel="Kendall's Tau",
        ylabel=f"{subset_col.capitalize()} Count",
        title=(
            f"{dataset}: Kendall's Tau Across {plural.capitalize()}"
            f"\n({num_models} models, {num_subsets} subsets, {num_instances} instances)"
        ),
        annotate=True,
        mean=kendall_tau_df["kendall_tau"].mean(),
        std=kendall_tau_df["kendall_tau"].std(),
        xlim=xlim,
        ylim=ylim,
    )

    analysis = "inter_subset_analysis"
    plot_name = "kendall-tau_histogram"
    file_path = build_plot_path(dataset, analysis, plot_name)
    plt.savefig(file_path)
    print(f"Saved plot to {file_path}")

    # -------------------------------------------------------------------------
    # 3. Mean Accuracy
    # -------------------------------------------------------------------------
    # Compute each subset's mean model scores
    subset_to_scores = {}
    for subset in subsets:
        mask = (
            get_metadata_mask(dataset_df, subset_col, subset)
            if is_ds_1000
            else dataset_df[subset_col] == subset
        )
        subset_to_scores[subset] = model_scores_df[mask].mean()
    subset_scores_df = pd.DataFrame(subset_to_scores).T
    subset_scores_df.index.name = "subset"

    # Plot the subset scores distribution
    global_means = model_scores_df.mean()
    _, axes = plt.subplots(num_models, 1, figsize=(8, 3 * num_models))

    xlim = {
        Dataset.DS_1000: (0, 1),
        Dataset.MATH: (0, 1),
        Dataset.MMLU: (0, 1),
    }[dataset]

    ylim = {
        Dataset.DS_1000: (0, num_subsets),
        Dataset.MATH: (0, num_subsets),
        Dataset.MMLU: (0, num_subsets // 2),
    }[dataset]

    for i, model in enumerate(subset_scores_df.columns):
        plot_histogram(
            subset_scores_df[model],
            xlabel="Mean Accuracy",
            ylabel=f"{subset_col.capitalize()} Count",
            title=f"{model}",
            ax=axes[i] if num_models > 1 else axes,
            annotate=True,
            mean=global_means[model],
            mean_label="Benchmark-level accuracy",
            xlim=xlim,
            ylim=ylim,
        )

    plt.suptitle(
        f"{dataset}: Mean Accuracy Across {plural.capitalize()}"
        f"\n({num_subsets} {plural}, {num_instances} instances)",
        y=1.01 if dataset == Dataset.MMLU else None,
    )
    plt.tight_layout()
    plot_name = "accuracy_histogram"
    file_path = build_plot_path(dataset, analysis, plot_name)
    plt.savefig(file_path)
    print(f"Saved plot to {file_path}")


if __name__ == "__main__":
    # Only MATH, MMLU, and DS-1000 have pre-defined subsets
    datasets = [Dataset.MATH, Dataset.MMLU, Dataset.DS_1000]

    for dataset in datasets:
        main(dataset)
