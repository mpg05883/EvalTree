from pathlib import Path

from utils.data import Dataset


def resolve_capability_tree_path(dataset: Dataset) -> Path:
    """Resolve the absolute file path to the processed capability tree file."""
    return Path(__file__).resolve().parent.parent / "data" / f"{dataset}.json"


def resolve_evaltree_dir() -> Path:
    """Resolve the absolute file path to the EvalTree directory."""
    return Path(__file__).resolve().parent.parent / "EvalTree"


def resolve_dataset_path(dataset: Dataset) -> Path:
    """Resolve the absolute file path to the raw dataset file."""
    return resolve_evaltree_dir() / "Datasets" / dataset / "dataset.json"


def resolve_eval_results_dir(dataset: Dataset) -> Path:
    """Resolve the absolute file path to the model eval results directory."""
    return resolve_evaltree_dir() / "Datasets" / dataset / "eval_results"


def resolve_plots_dir() -> Path:
    """Resolve the absolute file path to the plots directory."""
    return Path(__file__).resolve().parent.parent / "plots"


def build_plot_path(
    dataset: Dataset,
    analysis: str,
    plot_name: str,
    extension: str = "png",
) -> Path:
    """Build the absolute file path to the plot file and create the directory
    if it doesn't exist."""
    path = resolve_plots_dir() / analysis / dataset / f"{plot_name}.{extension}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
