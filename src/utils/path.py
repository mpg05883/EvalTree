from pathlib import Path

from src.utils.data import Dataset


def resolve_root_dir() -> Path:
    """Resolve the absolute file path to the root directory."""
    return Path(__file__).resolve().parent.parent.parent


def resolve_data_dir() -> Path:
    """Resolve the absolute file path to the data directory."""
    return resolve_root_dir() / "data"


def resolve_dataset_dir(dataset: Dataset) -> Path:
    """Resolve the absolute file path to the dataset directory."""
    return resolve_data_dir() / dataset


def resolve_dataset_path(dataset: Dataset) -> Path:
    """Resolve the absolute file path to the dataset file."""
    # Chatbot Arena (New) uses the same dataset as Chatbot Arena
    dataset = Dataset.CHATBOT_ARENA if dataset == Dataset.CHATBOT_ARENA_NEW else dataset
    return resolve_dataset_dir(dataset) / "dataset.json"


def resolve_capability_tree_path(dataset: Dataset) -> Path:
    """Resolve the absolute file path to the capability tree file."""
    return resolve_dataset_dir(dataset) / "capability_tree.json"


def resolve_model_scores_path(dataset: Dataset) -> Path:
    """Resolve the absolute file path to the file containing each model's
    per-instance scores."""
    return resolve_dataset_dir(dataset) / "model_scores.csv"


def resolve_eval_results_dir(dataset: Dataset) -> Path:
    """Resolve the absolute file path to the eval results directory."""
    return resolve_dataset_dir(dataset) / "eval_results"


def resolve_results_dir() -> Path:
    """Resolve the absolute file path to the results directory."""
    return resolve_root_dir() / "results"


def resolve_plots_dir() -> Path:
    """Resolve the absolute file path to the plots directory."""
    return resolve_results_dir() / "plots"


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
