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


def preview_tree(root: dict, model: str, k: int = 3, max_cap_len: int = 80) -> None:
    """Print the first k levels of a capability tree with a model's score per node.

    Useful for manually comparing computed tree values against the EvalTree web demo.

    Args:
        root: The root node of the capability tree (from load_capability_tree).
        model: The model name whose score to display at each node.
        k: Maximum depth to display (root = level 1).
        max_cap_len: Maximum characters to show of the capability description.

    Example:
        >>> tree = load_capability_tree(Dataset.CHATBOT_ARENA)
        >>> preview_tree(tree, model="gpt-4-1106-preview", k=3)
        [L1] size=44230 | score=1247.2614 | Synthesizing and integrating culturally...
          [L2] size=30167 | score=1242.5119 | Synthesizing and optimizing interdiscipl...
    """
    def _score(node: dict) -> float | None:
        return next((s for m, s in node.get("ranking", []) if m == model), None)

    def _traverse(node: dict, depth: int) -> None:
        if depth > k:
            return
        indent = "  " * (depth - 1)
        cap = node.get("capability", "")[:max_cap_len]
        size = node.get("size", "?")
        score = _score(node)
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"{indent}[L{depth}] size={size:>6} | score={score_str} | {cap}")
        subtrees = node.get("subtrees", [])
        if isinstance(subtrees, list):
            for child in subtrees:
                _traverse(child, depth + 1)

    _traverse(root, depth=1)


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
