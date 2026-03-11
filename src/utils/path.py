from pathlib import Path

from src.utils.enums import Dataset


def resolve_root_dir() -> Path:
    """Resolve the absolute file path to the root directory."""
    return Path(__file__).resolve().parent.parent.parent


def resolve_dataset_dir(dataset: Dataset) -> Path:
    """Resolve the absolute file path to the dataset directory."""
    return resolve_root_dir() / "data" / dataset


def resolve_dataset_path(dataset: Dataset) -> Path:
    """Resolve the absolute file path to the dataset file."""
    # Chatbot Arena (New) uses the same dataset as Chatbot Arena
    dataset = Dataset.CHATBOT_ARENA if dataset == Dataset.CHATBOT_ARENA_NEW else dataset
    return resolve_dataset_dir(dataset) / "dataset.json"


def resolve_metadata_path() -> Path:
    """Resolve the absolute file path to the metadata file."""
    return resolve_root_dir() / "data" / "metadata.json"


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


def resolve_results_dir(experiment: str) -> Path:
    """Resolve the absolute file path to the results directory for a given
    experiment."""
    return resolve_root_dir() / "results" / experiment


def resolve_plots_dir(experiment: str) -> Path:
    """Resolve the absolute file path to the plots directory for a given
    experiment."""
    return resolve_results_dir(experiment) / "plots"


def resolve_data_dir(experiment: str) -> Path:
    """Resolve the absolute file path to the data directory for a given
    experiment."""
    return resolve_results_dir(experiment) / "data"


def build_plot_path(
    dataset: Dataset,
    experiment: str,
    plot_name: str,
    sub_dirs: list[str] | None = None,
    extension: str = "png",
) -> Path:
    """Build the absolute file path to the plot file and create the directory
    if it doesn't exist."""
    path = resolve_plots_dir(experiment) / dataset / f"{plot_name}.{extension}"
    if sub_dirs:
        path = path.parent.joinpath(*sub_dirs) / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_data_path(
    dataset: Dataset,
    experiment: str,
    data_name: str,
    sub_dirs: list[str] | None = None,
    extension: str = "csv",
) -> Path:
    """Build the absolute file path to the data file and create the directory
    if it doesn't exist."""
    path = resolve_data_dir(experiment) / dataset / f"{data_name}.{extension}"
    if sub_dirs:
        path = path.parent.joinpath(*sub_dirs) / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
