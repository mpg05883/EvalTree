from enum import StrEnum


class Dataset(StrEnum):
    CHATBOT_ARENA = "Chatbot-Arena"
    CHATBOT_ARENA_NEW = "Chatbot-Arena_NEW"
    DS_1000 = "DS-1000"
    MATH = "MATH"
    MMLU = "MMLU"
    WILDCHAT_10K = "WildChat10K"

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

    @property
    def metric(self) -> str:
        return {
            Dataset.CHATBOT_ARENA: "elo Score",
            Dataset.CHATBOT_ARENA_NEW: "elo score",
            Dataset.DS_1000: "accuracy",
            Dataset.MATH: "accuracy",
            Dataset.MMLU: "accuracy",
            Dataset.WILDCHAT_10K: "win-rate",
        }[self]


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
