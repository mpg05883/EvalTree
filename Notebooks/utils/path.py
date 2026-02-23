from pathlib import Path


def resolve_root_dir() -> Path:
    """
    Resolve the absolute path to the project root directory from this file's
    location.

    Returns:
        Path: Absolute path to the project root directory.
    """
    return Path(__file__).resolve().parent.parent.parent


def resolve_datasets_dir() -> Path:
    """
    Resolve the absolute path to the Datasets directory from this file's
    location.

    Returns:
        Path: Absolute path to the Datasets directory.
    """
    dirpath = resolve_root_dir() / "Datasets"
    dirpath.mkdir(exist_ok=True)
    return dirpath


def resolve_eval_results_dir(dataset: str, real: bool = True) -> Path:
    """
    Resolve the absolute path to the evaluation results directory for a given
    dataset.

    Returns:
        Path: Absolute path to the evaluation results directory.
    """
    last_dir = "real" if real else "synthetic"
    dirpath = resolve_datasets_dir() / dataset / "eval_results" / last_dir
    dirpath.mkdir(exist_ok=True)
    return dirpath


def resolve_results_dir() -> Path:
    """
    Resolve the absolute path to the results directory from this file's
    location.

    Returns:
        Path: Absolute path to the results directory.
    """
    dirpath = resolve_root_dir() / "Results"
    dirpath.mkdir(exist_ok=True)
    return dirpath


def resolve_plots_dir() -> Path:
    """
    Resolve the absolute path to the plots directory from this file's
    location.

    Returns:
        Path: Absolute path to the plots directory.
    """
    dirpath = resolve_root_dir() / "Plots"
    dirpath.mkdir(exist_ok=True)
    return dirpath
