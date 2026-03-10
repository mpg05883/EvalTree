import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau
from tqdm import tqdm

from src.utils.data import (
    get_metadata_mask,
    get_unique_metadata_values,
    load_dataset,
)
from src.utils.enums import Dataset
from src.utils.metrics import kendallw
from src.utils.model import load_model_scores
from src.utils.path import build_plot_path
from src.utils.plot import plot_histogram


def main(dataset: Dataset) -> None:
    dataset_df = load_dataset(dataset)
    model_scores_df = load_model_scores(dataset)
    assert len(dataset_df) == len(model_scores_df)

    subset_col = dataset.subset_col
    is_ds_1000 = dataset == Dataset.DS_1000
    subsets = (
        get_unique_metadata_values(dataset_df, subset_col)
        if is_ds_1000
        else dataset_df[subset_col].unique()
    )

    plural = "libraries" if is_ds_1000 else f"{subset_col}s"
    num_subsets = len(subsets)
    num_models = len(model_scores_df.columns)
    num_instances = len(dataset_df)
    num_trials = 500
    analysis = "intra_subset_analysis"
    colors = sns.color_palette("tab10")

    # -------------------------------------------------------------------------
    # 1. Split-Halves Kendall's Tau
    # -------------------------------------------------------------------------
    kwargs = {
        "desc": "Computing split-half Kendall's Taus",
        "total": num_subsets,
        "unit": plural,
    }

    split_half_results = []

    for subset in tqdm(subsets, **kwargs):
        mask = (
            get_metadata_mask(dataset_df, subset_col, subset)
            if is_ds_1000
            else dataset_df[subset_col] == subset
        )
        subset_scores = model_scores_df[mask]
        subset_size = len(subset_scores)

        taus = []
        for trial in range(num_trials):
            rng = np.random.default_rng(trial)
            shuffled_indices = rng.permutation(subset_size)
            half = subset_size // 2
            scores_a = subset_scores.iloc[shuffled_indices[:half]].mean()
            scores_b = subset_scores.iloc[shuffled_indices[half:]].mean()
            tau, _ = kendalltau(scores_a, scores_b)
            taus.append(tau)

        split_half_results.append(
            {
                "subset": subset,
                "mean_kendall_tau": np.mean(taus),
                "std_kendall_tau": np.std(taus),
                "num_instances": subset_size,
            }
        )

    split_half_df = pd.DataFrame(split_half_results)

    ylim = {
        Dataset.DS_1000: (0, num_subsets),
        Dataset.MATH: (0, num_subsets),
        Dataset.MMLU: (0, num_subsets // 2),
    }[dataset]

    plot_histogram(
        split_half_df["mean_kendall_tau"],
        xlabel="Kendall's Tau",
        ylabel=f"{subset_col.capitalize()} Count",
        title=(
            f"{dataset}: Split-Halves Kendall's Tau Across {plural.capitalize()}"
            f"\n({num_models} models, {num_subsets} subsets, {num_instances} instances, {num_trials} trials)"
        ),
        annotate=True,
        mean=split_half_df["mean_kendall_tau"].mean(),
        std=split_half_df["mean_kendall_tau"].std(),
        xlim=(0, 1),
        ylim=ylim,
        color=colors[0],
    )

    file_path = build_plot_path(
        dataset, analysis, plot_name="split_halves-kendall_tau-histogram"
    )
    plt.savefig(file_path)
    print(f"Saved plot to {file_path}")
    plt.close()

    # -------------------------------------------------------------------------
    # 2. Bootstrapped Kendall's Tau
    # -------------------------------------------------------------------------
    kwargs = {
        "desc": "Computing bootstrapped Kendall's Taus",
        "total": num_subsets,
        "unit": plural,
    }

    bootstrap_results = []

    for subset in tqdm(subsets, **kwargs):
        mask = (
            get_metadata_mask(dataset_df, subset_col, subset)
            if is_ds_1000
            else dataset_df[subset_col] == subset
        )
        subset_scores = model_scores_df[mask]
        n = len(subset_scores)

        # Reference ranking: model ranking across all instances in the subset
        reference_scores = subset_scores.mean()

        taus = []
        for b in range(num_trials):
            rng = np.random.default_rng(b)
            bootstrap_indices = rng.integers(0, n, size=n)
            bootstrap_scores = subset_scores.iloc[bootstrap_indices].mean()
            tau, _ = kendalltau(reference_scores, bootstrap_scores)
            taus.append(tau)

        bootstrap_results.append(
            {
                "subset": subset,
                "mean_kendall_tau": np.mean(taus),
                "std_kendall_tau": np.std(taus),
                "num_instances": n,
            }
        )

    bootstrap_df = pd.DataFrame(bootstrap_results)

    ylim = {
        Dataset.DS_1000: (0, num_subsets),
        Dataset.MATH: (0, num_subsets),
        Dataset.MMLU: (0, num_subsets),
    }[dataset]

    plot_histogram(
        bootstrap_df["mean_kendall_tau"],
        xlabel="Kendall's Tau",
        ylabel=f"{subset_col.capitalize()} Count",
        title=(
            f"{dataset}: Bootstrapped Kendall's Tau Across {plural.capitalize()}"
            f"\n({num_models} models, {num_subsets} subsets, {num_instances} instances, {num_trials} trials)"
        ),
        annotate=True,
        mean=bootstrap_df["mean_kendall_tau"].mean(),
        std=bootstrap_df["mean_kendall_tau"].std(),
        xlim=(0, 1),
        ylim=ylim,
        color=colors[1],
    )

    file_path = build_plot_path(
        dataset, analysis, plot_name="bootstrap-kendall_tau-histogram"
    )
    plt.savefig(file_path)
    print(f"Saved plot to {file_path}")
    plt.close()

    # -------------------------------------------------------------------------
    # 3. Mini-Batch Kendall's W
    # -------------------------------------------------------------------------
    num_folds = 5

    kwargs = {
        "desc": "Computing mini-batch Kendall's W",
        "total": num_subsets,
        "unit": plural,
    }

    minibatch_w_results = []

    for subset in tqdm(subsets, **kwargs):
        mask = (
            get_metadata_mask(dataset_df, subset_col, subset)
            if is_ds_1000
            else dataset_df[subset_col] == subset
        )
        subset_scores = model_scores_df[mask]
        subset_size = len(subset_scores)

        kendallw_values = []
        for trial in range(num_trials):
            rng = np.random.default_rng(trial)
            shuffled_indices = rng.permutation(subset_size)
            folds = np.array_split(shuffled_indices, num_folds)

            # Each row is one fold's mean model scores: shape (num_folds, num_models)
            fold_scores = np.array(
                [subset_scores.iloc[fold].mean().values for fold in folds]
            )

            kendallw_values.append(kendallw(fold_scores))

        minibatch_w_results.append(
            {
                "subset": subset,
                "mean_kendallw": np.mean(kendallw_values),
                "std_kendallw": np.std(kendallw_values),
                "num_instances": subset_size,
            }
        )

    minibatch_w_df = pd.DataFrame(minibatch_w_results)

    ylim = {
        Dataset.DS_1000: (0, num_subsets),
        Dataset.MATH: (0, num_subsets),
        Dataset.MMLU: (0, num_subsets),
    }[dataset]

    plot_histogram(
        minibatch_w_df["mean_kendallw"],
        xlabel="Kendall's W",
        ylabel=f"{subset_col.capitalize()} Count",
        title=(
            f"{dataset}: Mini-Batch Kendall's W Across {plural.capitalize()}"
            f"\n({num_models} models, {num_subsets} subsets, {num_instances} instances, {num_trials} trials)"
        ),
        annotate=True,
        mean=minibatch_w_df["mean_kendallw"].mean(),
        std=minibatch_w_df["mean_kendallw"].std(),
        xlim=(0, 1),
        ylim=ylim,
        color=colors[2],
    )

    file_path = build_plot_path(
        dataset, analysis, plot_name="mini-batch-kendall_w-histogram"
    )
    plt.savefig(file_path)
    print(f"Saved plot to {file_path}")
    plt.close()


if __name__ == "__main__":
    # Only MATH, MMLU, and DS-1000 have pre-defined subsets
    datasets = [Dataset.MATH, Dataset.MMLU, Dataset.DS_1000]

    for dataset in datasets:
        main(dataset)
