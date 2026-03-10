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
    stack = []
    if isinstance(root["subtrees"], list):
        stack.extend(root["subtrees"])

    while stack:
        node = stack.pop()
        if node["size"] <= min_instances:
            continue
        nodes.append(node)
        if isinstance(node["subtrees"], list):
            stack.extend(node["subtrees"])

    return nodes


def collect_nodes_by_level(
    root: dict,
    min_instances: int = 50,
) -> dict[int, list[dict]]:
    """Iteratively collect all non-root nodes with more than `min_instances`
    instances, grouped by their depth level in the tree (root's children = 1).

    Args:
        root: The root node of the capability tree.
        min_instances: The minimum number of instances required for a node to be
        collected.

    Returns:
        A dict mapping level (int) to the list of nodes at that level with
        more than `min_instances` instances.
    """
    nodes_by_level: dict[int, list[dict]] = {}
    stack: list[tuple[dict, int]] = []
    if isinstance(root["subtrees"], list):
        stack.extend((child, 1) for child in root["subtrees"])

    while stack:
        node, level = stack.pop()
        if node["size"] <= min_instances:
            continue
        nodes_by_level.setdefault(level, []).append(node)
        if isinstance(node["subtrees"], list):
            stack.extend((child, level + 1) for child in node["subtrees"])

    return nodes_by_level


def get_node_indices(node: dict) -> list[int]:
    """Iteratively collect all dataset row indices for instances in a node.

    Each leaf node's subtrees field is an integer — the direct row index into
    model_scores_df / dataset_df. This function collects all such indices from
    a node and its descendants.

    Args:
        node: A capability tree node (from load_capability_tree or collect_nodes).

    Returns:
        A list of integer row indices into the dataset.
    """
    indices = []
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current["subtrees"], int):
            indices.append(current["subtrees"])
        else:
            stack.extend(current["subtrees"])
    return indices


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
