from enum import StrEnum

import pandas as pd


class Dataset(StrEnum):
    DS_1000 = "DS-1000"
    MATH = "MATH"
    MMLU = "MMLU"

    @property
    def num_instances(self) -> int:
        if self == Dataset.DS_1000:
            return 1000
        elif self == Dataset.MATH:
            return 5000
        elif self == Dataset.MMLU:
            return 14042
        else:
            raise ValueError(f"Unknown dataset: {self}")


class Library(StrEnum):
    MATPLOTLIB = "Matplotlib"
    NUMPY = "Numpy"
    PANDAS = "Pandas"
    PYTORCH = "Pytorch"
    SCIPY = "Scipy"
    SKLEARN = "Sklearn"
    TENSORFLOW = "Tensorflow"


class PerturbationType(StrEnum):
    ORIGIN = "Origin"
    SEMANTIC = "Semantic"
    DIFFICULT_REWRITE = "Difficult-Rewrite"
    SURFACE = "Surface"


class MetadataKey(StrEnum):
    LIBRARY = "library"
    PERTURBATION_TYPE = "perturbation_type"


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
