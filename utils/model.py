from typing import Literal

import numpy as np
import pandas as pd

from utils.common import DATASETS
from utils.path import resolve_eval_results_dir


def load_eval_results(dataset: DATASETS) -> pd.DataFrame:
    """Load the evaluation results for all models on a given dataset."""
    file_path = resolve_eval_results_dir(dataset) / "all.csv"
    return pd.read_csv(file_path)


def compute_rankings(
    df: pd.DataFrame,
    ascending: bool = False,
    method: Literal["min", "max", "average", "first", "dense"] = "average",
) -> np.ndarray:
    """Compute model rankings from performance data.

    Args:
        df: DataFrame where rows are instances and columns are models.
            Values should be performance scores (e.g., 0/1 for correct/incorrect).
        ascending: Whether to rank in ascending or descending order. Default is
        False.
        - If True, lower accuracies are ranked higher.
        - If False, higher accuracies are ranked higher.
        method: Method to break ties when models have the same accuracy.
        Default is "average". See examples below.
            Example with accuracies [0.80, 0.75, 0.75, 0.60]:
            - "min": [1, 2, 2, 4] - Tied models share the lowest rank they'd occupy.
            - "max": [1, 3, 3, 4] - Tied models share the highest rank they'd occupy.
            - "average": [1, 2.5, 2.5, 4] - Tied models get the average of their ranks.
            - "first": [1, 2, 3, 4] - First occurrence gets lower rank (no ties).
            - "dense": [1, 2, 2, 3] - Like min, but next rank doesn't skip.

    Returns:
        numpy array of rankings (1 = best) in the same order as df columns.
    """
    average_accuracies = df.mean(axis=0)
    rankings = average_accuracies.rank(
        ascending=ascending,
        method=method,
    ).astype(int)
    return rankings.values
