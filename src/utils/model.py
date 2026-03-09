import pandas as pd

from src.utils.enums import Dataset
from src.utils.path import resolve_model_scores_path


def load_model_scores(dataset: Dataset) -> pd.DataFrame:
    """Load the evaluation results for all models on a given dataset."""
    file_path = resolve_model_scores_path(dataset)
    return pd.read_csv(file_path)
