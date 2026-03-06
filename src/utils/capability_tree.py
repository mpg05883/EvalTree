import json
from typing import Any

from src.utils.enums import Dataset
from src.utils.path import resolve_capability_tree_path


def load_capability_tree(dataset: Dataset) -> dict[str, Any]:
    """Load the capability tree for a specified dataset.

    Args:
        dataset: The name of the dataset.

    Returns:
        The root node of the capability tree.
    """
    file_path = resolve_capability_tree_path(dataset)
    with open(file_path) as f:
        return json.load(f)


def collect_nodes(root: dict, min_instances: int = 50) -> list[dict]:
    """Iteratively collect all non-root nodes with more than `min_instances`
    instances.

    Args:
        root: The root node of the capability tree.
        min_instances: The minimum number of instances required for a node to be
        collected.

    Returns:
        A list of nodes with more than `min_instances` instances.
    """
    nodes = []
    stack = [(root, True)]
    while stack:
        node, is_root = stack.pop()
        if not is_root and node["size"] > min_instances:
            nodes.append(node)
        if isinstance(node["subtrees"], list):
            for child in node["subtrees"]:
                stack.append((child, False))
    return nodes


def align_rankings(
    global_ranking: list[list[str | float]],
    local_ranking: list[list[str | float]],
) -> tuple[list[float], list[float]]:
    """Align the rankings of the node and the global ranking.

    Args:
        global_ranking: The global ranking of the models, as a list of
            [model_name, score] pairs.
        local_ranking: The rankings of the models in the node, as a list of
            [model_name, score] pairs.

    Returns:
        The aligned rankings of the models in the node and the global ranking.
    """
    global_scores = {model: score for model, score in global_ranking}
    local_scores = {model: score for model, score in local_ranking}
    models = [m for m in global_scores if m in local_scores]
    aligned_global_ranking = [global_scores[m] for m in models]
    aligned_local_ranking = [local_scores[m] for m in models]
    return aligned_global_ranking, aligned_local_ranking
