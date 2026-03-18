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
    BBH = "BBH"
    CHATBOT_ARENA = "Chatbot-Arena"
    CHATBOT_ARENA_NEW = "Chatbot-Arena_NEW"
    DS_1000 = "DS-1000"
    GPQA_DIAMOND = "GPQA-Diamond"
    MATH = "MATH"
    MATH_LVL_5 = "MATH-Lvl-5"
    MMLU = "MMLU"
    MMLU_PRO = "MMLU-Pro"
    WILDCHAT_10K = "WildChat10K"

    @property
    def pretty_name(self) -> str:
        """Properly formatted name of the dataset."""
        return {
            Dataset.BBH: "BIG-Bench Hard",
            Dataset.CHATBOT_ARENA: "Chatbot Arena",
            Dataset.CHATBOT_ARENA_NEW: "Chatbot Arena (NEW)",
            Dataset.DS_1000: "DS-1000",
            Dataset.GPQA_DIAMOND: "GPQA Diamond",
            Dataset.MATH: "MATH",
            Dataset.MATH_LVL_5: "MATH Level 5",
            Dataset.MMLU: "MMLU",
            Dataset.MMLU_PRO: "MMLU-Pro",
            Dataset.WILDCHAT_10K: "WildChat",
        }[self]

    @property
    def num_instances(self) -> int:
        """Number of instances in the dataset. Values obtained from EvalTree's
        web demo:

        https://zhiyuan-zeng.github.io/EvalTree/
        """
        return {
            Dataset.BBH: 5759,
            Dataset.CHATBOT_ARENA: 44230,
            Dataset.CHATBOT_ARENA_NEW: 40273,
            Dataset.DS_1000: 1000,
            Dataset.GPQA_DIAMOND: 198,
            Dataset.MATH: 5000,
            Dataset.MATH_LVL_5: 1324,
            Dataset.MMLU: 14042,
            Dataset.MMLU_PRO: 12032,
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
        """Column name for pre-defined subsets in the dataset."""
        return {
            Dataset.BBH: "subset",
            Dataset.DS_1000: MetadataKey.LIBRARY,  # Can be either "library" or "perturbation_type"
            Dataset.GPQA_DIAMOND: "Subdomain",  # Can be either "Subdomain" or "High-level domain"
            Dataset.MATH: "subset",
            Dataset.MATH_LVL_5: "type",
            Dataset.MMLU: "subject",
            Dataset.MMLU_PRO: "category",
        }[self]

    @property
    def plural(self) -> str:
        """Plural form of the dataset's subsets."""

        return {
            Dataset.BBH: "subsets",
            Dataset.DS_1000: "libraries",
            Dataset.GPQA_DIAMOND: "Subdomains",
            Dataset.MATH: "subsets",
            Dataset.MATH_LVL_5: "types",
            Dataset.MMLU: "subjects",
            Dataset.MMLU_PRO: "categories",
        }[self]
