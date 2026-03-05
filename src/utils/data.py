import pandas as pd

from src.utils.enum import Dataset, Library, MetadataKey, PerturbationType
from src.utils.path import resolve_dataset_path


def load_dataset(dataset: Dataset) -> pd.DataFrame:
    """Load a dataset from the local directory."""
    file_path = resolve_dataset_path(dataset)
    return pd.read_json(file_path)


def get_unique_metadata_values(
    df: pd.DataFrame,
    key: MetadataKey,
    col: str = "metadata",
) -> set:
    """Get all unique values for a given key from a metadata column.

    Args:
        df: DataFrame containing a metadata column.
        key: The key to extract values for from each dict.
        col: Name of the metadata column. Defaults to "metadata".

    Returns:
        Set of unique values for the given key.
    """
    return set(d[key] for d in df[col] if key in d)


def get_metadata_mask(
    df: pd.DataFrame,
    key: MetadataKey,
    value: Library | PerturbationType,
    col: str = "metadata",
) -> pd.Series:
    """Get boolean mask for rows where metadata[key] equals value.

    Args:
        df: DataFrame containing a metadata column.
        key: The key to check in each metadata dict.
        value: The value to filter for.
        col: Name of the metadata column. Defaults to "metadata".

    Returns:
        Boolean Series indicating which rows match the condition.
    """
    return df[col].apply(lambda d: d.get(key) == value)
