from pathlib import Path

from utils.common import DATASETS


def resolve_data_dir() -> Path:
    """Resolve the absolute file path to the data directory.

    NOTE: This directory contains data that's already been processed and is
    ready to be used to create a capability tree.
    """
    return Path(__file__).resolve().parent.parent / "data"


def resolve_evaltree_dir() -> Path:
    """Resolve the absolute file path to the EvalTree directory."""
    return Path(__file__).resolve().parent.parent / "EvalTree"


def resolve_datasets_dir() -> Path:
    """Resolve the absolute file path to the EvalTree/Datasets directory."""
    return resolve_evaltree_dir() / "Datasets"


def resolve_dataset_path(dataset: DATASETS) -> Path:
    """Resolve the absolute file path to the raw dataset file."""
    return resolve_datasets_dir() / dataset / "dataset.json"


def resolve_eval_results_dir(dataset: DATASETS) -> Path:
    """Resolve the absolute file path to the model eval results directory."""
    return resolve_eval_results_dir(dataset) / "eval_results"


def resolve_capability_tree_path(dataset: DATASETS) -> Path:
    """Resolve the absolute file path to the capability tree file."""
    return (
        resolve_evaltree_dir()
        / "Datasets"
        / dataset
        / "EvalTree"
        / "stage3-RecursiveClustering"
        / "[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]_[stage4-CapabilityDescription-model=gpt-4o].json"
    )
