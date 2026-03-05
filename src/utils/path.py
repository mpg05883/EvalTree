from pathlib import Path

from utils.data import Dataset

_SRC_DIR = Path(__file__).resolve().parent.parent   # → <root>/src/
_ROOT_DIR = _SRC_DIR.parent                         # → <root>/


def resolve_capability_tree_path(dataset: Dataset) -> Path:
    """Resolve the absolute file path to the processed capability tree file."""
    return _ROOT_DIR / "data" / f"{dataset}.json"


def resolve_evaltree_dir() -> Path:
    """Resolve the absolute file path to the EvalTree directory."""
    return _SRC_DIR / "EvalTree"


def resolve_dataset_path(dataset: Dataset) -> Path:
    """Resolve the absolute file path to the raw dataset file."""
    # Chatbot Arena (New) uses the same dataset as Chatbot Arena
    dataset = Dataset.CHATBOT_ARENA if dataset == Dataset.CHATBOT_ARENA_NEW else dataset
    return resolve_evaltree_dir() / "Datasets" / dataset / "dataset.json"


def resolve_eval_results_dir(dataset: Dataset) -> Path:
    """Resolve the absolute file path to the model eval results directory."""
    return resolve_evaltree_dir() / "Datasets" / dataset / "eval_results"


def resolve_plots_dir() -> Path:
    """Resolve the absolute file path to the plots directory."""
    return _ROOT_DIR / "plots"


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
