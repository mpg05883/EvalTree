import pandas as pd

from utils.common import DATASETS
from utils.path import resolve_eval_results_dir


def load_eval_results(dataset: DATASETS) -> pd.DataFrame:
    """Load the evaluation results for all models on a given dataset."""
    file_path = resolve_eval_results_dir(dataset) / "all.csv"
    return pd.read_csv(file_path)
