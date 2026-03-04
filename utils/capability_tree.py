import json
from typing import Any

from utils.data import Dataset
from utils.path import resolve_capability_tree_path


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
    global_ranking: list[list],
    local_ranking: list[list],
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
    models = list(global_scores.keys())
    global_vec = [global_scores[m] for m in models]
    local_vec = [local_scores[m] for m in models]
    return global_vec, local_vec
