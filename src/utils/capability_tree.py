import json
from collections import deque
from dataclasses import dataclass
from typing import Any

from src.utils.enums import Dataset
from src.utils.path import resolve_capability_tree_path


@dataclass
class Node:
    """A node in a capability tree.

    Attributes:
        capability: Natural-language description of the capability this node
            represents.
        size: Total number of dataset instances contained in this subtree.
        depth: Depth of this node in the tree (root is 1).
        subtrees: Child nodes if this is an internal node, or a single integer
            dataset row index if this is a leaf node.
        ranking: Model performance ranking for this node, as a dict mapping
            each model name to its mean score, ordered descending by score.
        CI: Bootstrap confidence interval for each model's score, as a dict
            mapping model name to a [lower, upper] bound pair. Only present on
            nodes with enough instances for reliable estimation.
        distinction: Short label distinguishing this node from its siblings
            (e.g. "Combinatorial and statistical structure analysis"). Absent
            on the root.
    """

    capability: str
    size: int
    depth: int
    subtrees: "list[Node] | int"
    ranking: dict[str, float] | None
    CI: dict[str, list[float]] | None = None
    distinction: str | None = None
    input: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Node":
        subtrees = data["subtrees"]
        if isinstance(subtrees, list):
            subtrees = [cls.from_dict(child) for child in subtrees]
        return cls(
            capability=data["capability"],
            size=data["size"],
            depth=data["depth"],
            subtrees=subtrees,
            ranking=(
                {model: score for model, score in data["ranking"]}
                if data["ranking"] is not None
                else None
            ),
            CI=data.get("CI"),
            distinction=data.get("distinction"),
            input=data.get("input"),
        )

    def get_indices(self) -> list[int]:
        """Return all dataset row indices for instances in this subtree.

        Traverses descendant nodes and collects the integer index stored at
        each leaf (where ``subtrees`` is an int rather than a list).

        Returns:
            A list of integer row indices into the dataset.
        """
        indices = []
        stack: list[Node] = [self]
        while stack:
            node = stack.pop()
            if isinstance(node.subtrees, int):
                indices.append(node.subtrees)
            else:
                stack.extend(node.subtrees)
        return indices


@dataclass
class Level:
    """A single depth level in a capability tree.

    Attributes:
        depth: Depth of this level in the tree, where 1 means the root's
            immediate children.
        nodes: All qualifying nodes collected at this depth level.
    """

    depth: int
    nodes: "list[Node]"

    @property
    def num_nodes(self) -> int:
        """Total number of nodes at this depth level."""
        return len(self.nodes)

    @property
    def indices(self) -> list[int]:
        """Unique dataset row indices of all instances linked to at least one
        node at this level.

        Deduplicates across nodes so that any instance appearing in more than
        one node is counted only once.
        """
        seen: set[int] = set()
        result: list[int] = []
        for node in self.nodes:
            for idx in node.get_indices():
                if idx not in seen:
                    seen.add(idx)
                    result.append(idx)
        return result

    @property
    def num_instances(self) -> int:
        """Number of unique instances linked to at least one node at this level."""
        return len(self.indices)


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


def collect_nodes(root: dict, min_instances: int = 50) -> "list[Node]":
    """Collect all non-root nodes depth-first, returning a flat list of Node objects.

    Uses an explicit stack to traverse the tree in depth-first order. Nodes
    whose size is below ``min_instances`` are skipped and their entire
    subtree is pruned — no deeper descendants are visited.

    Args:
        root: The raw capability-tree dict returned by :func:`load_capability_tree`.
        min_instances: Minimum number of instances a node must contain to be
            included. Nodes whose size is greater than or equal to this value
            are kept. Defaults to 50.

    Returns:
        A flat ``list[Node]`` in depth-first traversal order.
    """
    nodes: list[Node] = []
    stack: list[dict] = []
    if isinstance(root["subtrees"], list):
        stack.extend(root["subtrees"])

    while stack:
        node = stack.pop()
        if node["size"] < min_instances:
            continue
        nodes.append(Node.from_dict(node))
        if isinstance(node["subtrees"], list):
            stack.extend(node["subtrees"])

    return nodes


def collect_levels(root: dict, min_instances: int = 50) -> "list[Level]":
    """Collect all non-root nodes breadth-first, returning one Level per depth.

    Uses a queue to traverse the tree in breadth-first order so that all
    nodes at a shallower level are visited before any node at the next level.
    Qualifying nodes are grouped into :class:`Level` objects, one per depth,
    where level 1 is the root's immediate children. Nodes whose size is below
    ``min_instances`` are skipped and their entire subtree is pruned.

    Args:
        root: The raw capability-tree dict returned by :func:`load_capability_tree`.
        min_instances: Minimum number of instances a node must contain to be
            included. Nodes whose size is greater than or equal to this value
            are kept. Defaults to 50.

    Returns:
        A ``list[Level]`` ordered by ascending depth, where each :class:`Level`
        holds the qualifying nodes at that depth and their shared metadata.
    """
    nodes_by_depth: dict[int, list[Node]] = {}
    queue: deque[tuple[dict, int]] = deque()
    if isinstance(root["subtrees"], list):
        queue.extend((child, 1) for child in root["subtrees"])

    while queue:
        node, depth = queue.popleft()
        if node["size"] < min_instances:
            continue
        nodes_by_depth.setdefault(depth, []).append(Node.from_dict(node))
        if isinstance(node["subtrees"], list):
            queue.extend((child, depth + 1) for child in node["subtrees"])

    return [Level(depth=d, nodes=nodes) for d, nodes in sorted(nodes_by_depth.items())]


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
    global_ranking: dict[str, float],
    local_ranking: dict[str, float],
) -> tuple[list[float], list[float]]:
    """Align the rankings of the node and the global ranking.

    Returns scores for only the models present in both dicts, in the order
    they appear in ``global_ranking``.

    Args:
        global_ranking: The global model ranking, as a dict mapping each model
            name to its benchmark-level mean score.
        local_ranking: The node-level model ranking, as a dict mapping each
            model name to its mean score within the node.

    Returns:
        A pair of aligned score lists ``(global_scores, local_scores)``,
        one entry per model that appears in both rankings.
    """
    models = [m for m in global_ranking if m in local_ranking]
    aligned_global = [global_ranking[m] for m in models]
    aligned_local = [local_ranking[m] for m in models]
    return aligned_global, aligned_local
