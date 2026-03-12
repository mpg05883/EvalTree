from src.utils.capability_tree import (
    Level,
    Node,
    collect_levels,
    collect_nodes,
    get_node_indices,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_node(size: int, subtrees, depth: int = 2, capability: str = "") -> dict:
    """Return a minimal capability-tree node dict with all fields required by
    Node.from_dict()."""
    return {
        "capability": capability or f"cap_{size}",
        "size": size,
        "depth": depth,
        "subtrees": subtrees,
        "ranking": None,
    }


# =============================================================================
# collect_nodes
# =============================================================================


def test_dfs_root_excluded():
    """The root node is never included in results, even if its size exceeds
    the threshold and it has no children."""
    root = make_node(100, [], depth=1)
    assert collect_nodes(root) == []


def test_dfs_single_qualifying_child():
    """A child whose size exceeds the default threshold (50) is collected and
    returned as a Node instance."""
    child = make_node(100, 5)
    root = make_node(200, [child], depth=1)
    result = collect_nodes(root)
    assert len(result) == 1
    assert isinstance(result[0], Node)
    assert result[0].size == 100


def test_dfs_min_instances_boundary():
    """The threshold check is greater-than-or-equal, so a node whose size
    equals min_instances exactly is included."""
    root = make_node(200, [make_node(50, 3)], depth=1)
    result = collect_nodes(root)
    assert len(result) == 1
    assert result[0].size == 50


def test_dfs_all_non_root_nodes_collected():
    """In a two-level tree where every non-root node exceeds the threshold,
    all four non-root nodes are returned."""
    grandchildren = [
        make_node(80, 3, depth=3),
        make_node(60, 2, depth=3),
    ]
    child_a = make_node(150, grandchildren, depth=2)
    child_b = make_node(100, 5, depth=2)
    root = make_node(200, [child_a, child_b], depth=1)
    result = collect_nodes(root)
    assert len(result) == 4
    assert {n.size for n in result} == {150, 100, 80, 60}


def test_dfs_partial_qualification():
    """Among sibling nodes, only those that exceed the threshold are collected;
    siblings that fall below it are excluded."""
    qualifying = make_node(100, 3)
    disqualified = make_node(30, 2)
    root = make_node(200, [qualifying, disqualified], depth=1)
    result = collect_nodes(root)
    assert len(result) == 1
    assert result[0].size == 100


def test_dfs_custom_min_instances():
    """A lower min_instances value allows nodes that would otherwise be
    excluded to pass the threshold."""
    node_a = make_node(100, 3)
    node_b = make_node(30, 2)
    root = make_node(200, [node_a, node_b], depth=1)
    result = collect_nodes(root, min_instances=20)
    assert len(result) == 2
    assert {n.size for n in result} == {100, 30}


def test_dfs_pruning_on_disqualified_node():
    """When a node fails the threshold, its entire subtree is pruned: children
    and deeper descendants are never visited, even if they would qualify."""
    large_child = make_node(100, 2, depth=3)
    small_parent = make_node(30, [large_child], depth=2)
    root = make_node(200, [small_parent], depth=1)
    assert collect_nodes(root) == []


def test_dfs_leaf_root_returns_empty():
    """When the root's subtrees field is an int rather than a list, there are
    no children to traverse and the result is empty."""
    root = make_node(200, 10, depth=1)
    assert collect_nodes(root) == []


def test_dfs_returns_node_instances():
    """All items in the returned list are Node dataclass instances."""
    child = make_node(100, 5)
    root = make_node(200, [child], depth=1)
    result = collect_nodes(root)
    assert all(isinstance(n, Node) for n in result)


# =============================================================================
# collect_levels
# =============================================================================


def test_bfs_root_excluded():
    """The root node is never included in results, even if its size exceeds
    the threshold and it has no children."""
    root = make_node(100, [], depth=1)
    assert collect_levels(root) == []


def test_bfs_single_qualifying_child():
    """A single qualifying child of the root is returned as a Level at depth 1
    whose nodes list contains one Node instance."""
    child = make_node(100, 5)
    root = make_node(200, [child], depth=1)
    result = collect_levels(root)
    assert len(result) == 1
    assert isinstance(result[0], Level)
    assert result[0].depth == 1
    assert len(result[0].nodes) == 1
    assert isinstance(result[0].nodes[0], Node)
    assert result[0].nodes[0].size == 100


def test_bfs_min_instances_boundary():
    """The threshold check is greater-than-or-equal, so a node whose size
    equals min_instances exactly is included."""
    root = make_node(200, [make_node(50, 3)], depth=1)
    result = collect_levels(root)
    assert len(result) == 1
    assert result[0].nodes[0].size == 50


def test_bfs_correct_levels():
    """Nodes are assigned to the correct Level: root's children at depth 1,
    grandchildren at depth 2."""
    grandchildren = [
        make_node(80, 3, depth=3),
        make_node(60, 2, depth=3),
    ]
    child = make_node(150, grandchildren, depth=2)
    root = make_node(200, [child], depth=1)
    result = collect_levels(root)
    assert len(result) == 2
    assert result[0].depth == 1
    assert result[0].num_nodes == 1
    assert result[0].nodes[0].size == 150
    assert result[1].depth == 2
    assert {n.size for n in result[1].nodes} == {80, 60}


def test_bfs_multiple_children_same_level():
    """Sibling nodes that both qualify are both placed in the same Level."""
    child_a = make_node(100, 3)
    child_b = make_node(80, 2)
    root = make_node(200, [child_a, child_b], depth=1)
    result = collect_levels(root)
    assert len(result) == 1
    assert result[0].depth == 1
    assert {n.size for n in result[0].nodes} == {100, 80}


def test_bfs_partial_qualification():
    """Among sibling nodes only the qualifying one is collected; the
    disqualified sibling is excluded from the Level."""
    qualifying = make_node(100, 3)
    disqualified = make_node(30, 2)
    root = make_node(200, [qualifying, disqualified], depth=1)
    result = collect_levels(root)
    assert len(result) == 1
    assert result[0].num_nodes == 1
    assert result[0].nodes[0].size == 100


def test_bfs_custom_min_instances():
    """A lower min_instances value allows nodes that would otherwise be
    excluded to pass the threshold and appear in the Level."""
    node_a = make_node(100, 3)
    node_b = make_node(30, 2)
    root = make_node(200, [node_a, node_b], depth=1)
    result = collect_levels(root, min_instances=20)
    assert len(result) == 1
    assert {n.size for n in result[0].nodes} == {100, 30}


def test_bfs_pruning_on_disqualified_node():
    """When a node fails the threshold, its entire subtree is pruned:
    children are never visited even if they would qualify."""
    large_child = make_node(100, 2, depth=3)
    small_parent = make_node(30, [large_child], depth=2)
    root = make_node(200, [small_parent], depth=1)
    assert collect_levels(root) == []


def test_bfs_leaf_root_returns_empty():
    """When the root's subtrees field is an int rather than a list, there are
    no children to traverse and the result is an empty list."""
    root = make_node(200, 10, depth=1)
    assert collect_levels(root) == []


def test_bfs_three_levels():
    """A three-level tree produces two Level objects at depths 1 and 2."""
    grandchild = make_node(60, 1, depth=3)
    child = make_node(120, [grandchild], depth=2)
    root = make_node(200, [child], depth=1)
    result = collect_levels(root)
    assert len(result) == 2
    assert result[0].depth == 1
    assert result[0].nodes[0].size == 120
    assert result[1].depth == 2
    assert result[1].nodes[0].size == 60


def test_bfs_deep_level_skipped_when_parent_pruned():
    """A qualifying node at depth 3 is never reached when its depth-2 parent
    fails the threshold, because pruning stops traversal at the parent."""
    deep_node = make_node(100, 1, depth=4)
    pruned_mid = make_node(20, [deep_node], depth=3)
    qualifying_top = make_node(150, [pruned_mid], depth=2)
    root = make_node(200, [qualifying_top], depth=1)
    result = collect_levels(root)
    assert len(result) == 1
    assert result[0].depth == 1
    assert result[0].nodes[0].size == 150


def test_bfs_returns_level_instances():
    """All items in the returned list are Level dataclass instances."""
    child = make_node(100, 5)
    root = make_node(200, [child], depth=1)
    result = collect_levels(root)
    assert all(isinstance(lv, Level) for lv in result)


def test_bfs_level_order():
    """BFS visits all depth-1 nodes before any depth-2 node, so the returned
    list is ordered by ascending depth."""
    grandchild = make_node(70, 1, depth=3)
    child_a = make_node(120, [grandchild], depth=2)
    child_b = make_node(110, 2, depth=2)
    root = make_node(200, [child_a, child_b], depth=1)
    result = collect_levels(root)
    assert [lv.depth for lv in result] == [1, 2]


# =============================================================================
# Level properties
# =============================================================================


def test_level_num_nodes():
    """num_nodes returns the count of nodes stored in the Level."""
    child_a = make_node(100, 3)
    child_b = make_node(80, 2)
    root = make_node(200, [child_a, child_b], depth=1)
    result = collect_levels(root)
    assert result[0].num_nodes == 2


def test_level_num_instances_leaf_nodes():
    """num_instances counts the unique leaf indices across all nodes in the Level.
    Each leaf node contributes a single integer index via get_indices()."""
    child_a = make_node(100, 3)  # subtrees=3 → one leaf at index 3
    child_b = make_node(80, 7)  # subtrees=7 → one leaf at index 7
    root = make_node(200, [child_a, child_b], depth=1)
    result = collect_levels(root)
    assert result[0].num_instances == 2


def test_level_indices_deduplication():
    """indices deduplicates row indices that appear in more than one node
    at the same level."""
    shared_leaf = {
        "size": 10,
        "subtrees": 42,
        "capability": "leaf",
        "depth": 3,
        "ranking": None,
    }
    node_a = make_node(100, [shared_leaf])
    node_b = make_node(80, [shared_leaf])
    root = make_node(200, [node_a, node_b], depth=1)
    result = collect_levels(root)
    # index 42 appears in both nodes but should be counted only once
    assert result[0].indices.count(42) == 1
    assert result[0].num_instances == 1


# =============================================================================
# get_node_indices
# =============================================================================


def test_gni_leaf_node():
    """A node whose subtrees field is a single int returns a list containing
    just that index."""
    node = {"size": 10, "subtrees": 7}
    assert get_node_indices(node) == [7]


def test_gni_single_level_children():
    """A node with a list of leaf children returns all their integer indices."""
    node = {
        "size": 30,
        "subtrees": [
            {"size": 10, "subtrees": 0},
            {"size": 10, "subtrees": 1},
            {"size": 10, "subtrees": 2},
        ],
    }
    assert sorted(get_node_indices(node)) == [0, 1, 2]


def test_gni_multi_level():
    """A multi-level subtree yields the indices of all leaf descendants,
    regardless of depth."""
    leaf_a = {"size": 10, "subtrees": 3}
    leaf_b = {"size": 10, "subtrees": 7}
    mid = {"size": 20, "subtrees": [leaf_a, leaf_b]}
    leaf_c = {"size": 10, "subtrees": 11}
    node = {"size": 40, "subtrees": [mid, leaf_c]}
    assert sorted(get_node_indices(node)) == [3, 7, 11]


def test_gni_chain_of_non_leaves():
    """A chain of non-leaf nodes with a single leaf at the bottom returns
    the single leaf index."""
    leaf = {"size": 10, "subtrees": 42}
    mid = {"size": 20, "subtrees": [leaf]}
    top = {"size": 30, "subtrees": [mid]}
    assert get_node_indices(top) == [42]


def test_gni_all_indices_returned():
    """Every leaf index in a wide subtree is present in the result."""
    leaves = [{"size": 10, "subtrees": i} for i in range(5)]
    node = {"size": 50, "subtrees": leaves}
    assert sorted(get_node_indices(node)) == list(range(5))


def test_gni_returns_list():
    """The return type is always a list, even for a single-index result."""
    node = {"size": 10, "subtrees": 99}
    result = get_node_indices(node)
    assert isinstance(result, list)


def test_gni_index_zero():
    """An index of 0 is a valid leaf value and is correctly collected."""
    node = {"size": 10, "subtrees": 0}
    assert get_node_indices(node) == [0]
