import json
from enum import StrEnum
from functools import cached_property
from pathlib import Path


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


class Dataset(StrEnum):
    CHATBOT_ARENA = "Chatbot-Arena"
    CHATBOT_ARENA_NEW = "Chatbot-Arena_NEW"
    DS_1000 = "DS-1000"
    MATH = "MATH"
    MMLU = "MMLU"
    WILDCHAT_10K = "WildChat10K"

    @property
    def pretty_name(self) -> str:
        """Properly formatted name of the dataset."""
        return {
            Dataset.CHATBOT_ARENA: "Chatbot Arena",
            Dataset.CHATBOT_ARENA_NEW: "Chatbot Arena (NEW)",
            Dataset.DS_1000: "DS-1000",
            Dataset.MATH: "MATH",
            Dataset.MMLU: "MMLU",
            Dataset.WILDCHAT_10K: "WildChat",
        }[self]

    @property
    def num_instances(self) -> int:
        """Number of instances in the dataset. Values obtained from EvalTree's
        web demo:

        https://zhiyuan-zeng.github.io/EvalTree/
        """
        return {
            Dataset.CHATBOT_ARENA: 44230,
            Dataset.CHATBOT_ARENA_NEW: 40273,
            Dataset.DS_1000: 1000,
            Dataset.MATH: 5000,
            Dataset.MMLU: 14042,
            Dataset.WILDCHAT_10K: 10000,
        }[self]

    @cached_property
    def metadata(self) -> dict[str, str | list[str]]:
        """Metadata for the dataset. Values were obtained from meta.json."""
        root_dir = Path(__file__).resolve().parent.parent.parent
        path = root_dir / "data" / "meta.json"
        with open(path, "r") as f:
            metadata = json.load(f)
        return metadata[self]

    @property
    def metric(self) -> str:
        """Metric used to evaluate model performance on the dataset. Values
        were obtained from meta.json.
        """
        return self.metadata["metrics"]

    @property
    def models(self) -> list[str]:
        """Models that were evaluated on the dataset. Values were obtained from
        meta.json.
        """
        return self.metadata["models"]

    @property
    def subset_col(self) -> str:
        """Column name for pre-defined subsets in the dataset.

        NOTE: Only MATH, MMLU, and DS-1000 have pre-defined subsets.
        """
        allowed = {Dataset.DS_1000, Dataset.MATH, Dataset.MMLU}

        if self not in allowed:
            raise ValueError(
                f"Dataset {self} does not have pre-defined subsets: "
                f"allowed datasets are {allowed}"
            )

        return {
            Dataset.DS_1000: MetadataKey.LIBRARY,  # Can be either "library" or "perturbation_type"
            Dataset.MATH: "subset",
            Dataset.MMLU: "subject",
        }[self]

    @property
    def plural(self) -> str:
        """Plural form of the dataset's subsets.

        NOTE: Only applies to MATH, MMLU, and DS-1000.
        """
        allowed = {Dataset.DS_1000, Dataset.MATH, Dataset.MMLU}

        if self not in allowed:
            raise ValueError(
                f"Dataset {self} does not have pre-defined subsets: "
                f"allowed datasets are {allowed}"
            )

        return {
            Dataset.DS_1000: "libraries",
            Dataset.MATH: "subsets",
            Dataset.MMLU: "subjects",
        }[self]
