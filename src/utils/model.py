from typing import Literal

import pandas as pd

from src.utils.enums import Dataset
from src.utils.path import resolve_model_scores_path


def load_model_scores(dataset: Dataset) -> pd.DataFrame:
    """Load the evaluation results for all models on a given dataset."""
    file_path = resolve_model_scores_path(dataset)
    return pd.read_csv(file_path)


def compute_model_ranking(
    data: pd.DataFrame | list[dict[str, float]],
    ascending: bool = False,
    method: Literal["min", "max", "average", "first", "dense"] = "average",
) -> dict[str, float]:
    """Compute model ranks from accuracy data.

    Args:
        data: One of:
            - DataFrame where rows are instances and columns are models.
              Values should be accuracy scores (e.g., 0/1 for correct/incorrect).
            - List of dicts mapping model names to scores, one dict per instance.
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
        Dict mapping model names to their ranks (1 = best), sorted alphabetically by model name.
    """
    # Convert data to mean scores, and then a dict of model names to scores
    model_scores = (
        data.select_dtypes(include="number").mean(axis=0).to_dict()
        if isinstance(data, pd.DataFrame)
        else pd.DataFrame(data).mean(axis=0).to_dict()
    )

    # Rank the models by their scores
    model_ranks = (
        pd.Series(model_scores)
        .rank(
            ascending=ascending,
            method=method,
        )
        .to_dict()
    )

    # Return dict of model names to ranks, sorted alphabetically by model name
    return dict(sorted(model_ranks.items()))
