from src.utils.capability_tree import collect_nodes, collect_nodes_by_level, get_node_indices


def test_root_excluded():
    """The root node is never included in results, even if its size exceeds
    the threshold and it has no children."""
    root = {"size": 100, "subtrees": []}
    assert collect_nodes(root) == []


def test_single_qualifying_child():
    """A child whose size exceeds the default threshold (50) is collected."""
    child = {"size": 100, "subtrees": 5}
    root = {"size": 200, "subtrees": [child]}
    assert collect_nodes(root) == [child]


def test_min_instances_boundary():
    """The threshold check is strictly greater-than, so a node whose size
    equals min_instances exactly is excluded."""
    root = {"size": 200, "subtrees": [{"size": 50, "subtrees": 3}]}
    assert collect_nodes(root) == []


def test_all_non_root_nodes_collected():
    """In a two-level tree where every non-root node exceeds the threshold,
    all four non-root nodes are returned."""
    grandchildren = [
        {"size": 80, "subtrees": 3},
        {"size": 60, "subtrees": 2},
    ]
    child_a = {"size": 150, "subtrees": grandchildren}
    child_b = {"size": 100, "subtrees": 5}
    root = {"size": 200, "subtrees": [child_a, child_b]}
    result = collect_nodes(root)
    assert len(result) == 4
    assert set(id(n) for n in result) == {
        id(child_a),
        id(child_b),
        *[id(gc) for gc in grandchildren],
    }


def test_partial_qualification():
    """Among sibling nodes, only those that exceed the threshold are collected;
    siblings that fall below it are excluded."""
    qualifying = {"size": 100, "subtrees": 3}
    disqualified = {"size": 30, "subtrees": 2}
    root = {"size": 200, "subtrees": [qualifying, disqualified]}
    assert collect_nodes(root) == [qualifying]


def test_custom_min_instances():
    """A lower min_instances value allows nodes that would otherwise be
    excluded to pass the threshold."""
    node_a = {"size": 100, "subtrees": 3}
    node_b = {"size": 30, "subtrees": 2}
    root = {"size": 200, "subtrees": [node_a, node_b]}
    result = collect_nodes(root, min_instances=20)
    assert len(result) == 2
    assert set(id(n) for n in result) == {id(node_a), id(node_b)}


def test_pruning_on_disqualified_node():
    """When a node fails the threshold, its entire subtree is pruned: children
    and deeper descendants are never visited, even if they would qualify."""
    large_child = {"size": 100, "subtrees": 2}
    small_parent = {"size": 30, "subtrees": [large_child]}
    root = {"size": 200, "subtrees": [small_parent]}
    assert collect_nodes(root) == []


def test_leaf_root_returns_empty():
    """When the root's subtrees field is an int rather than a list, there are
    no children to traverse and the result is empty."""
    root = {"size": 200, "subtrees": 10}
    assert collect_nodes(root) == []


# =============================================================================
# collect_nodes_by_level
# =============================================================================


def test_cnbl_root_excluded():
    """The root node is never included in results, even if its size exceeds
    the threshold and it has no children."""
    root = {"size": 100, "subtrees": []}
    assert collect_nodes_by_level(root) == {}


def test_cnbl_single_qualifying_child():
    """A single qualifying child of the root is placed at level 1."""
    child = {"size": 100, "subtrees": 5}
    root = {"size": 200, "subtrees": [child]}
    result = collect_nodes_by_level(root)
    assert list(result.keys()) == [1]
    assert result[1] == [child]


def test_cnbl_min_instances_boundary():
    """The threshold check is strictly greater-than, so a node whose size
    equals min_instances exactly is excluded and the result is empty."""
    root = {"size": 200, "subtrees": [{"size": 50, "subtrees": 3}]}
    assert collect_nodes_by_level(root) == {}


def test_cnbl_correct_levels():
    """Nodes are assigned to the correct level: root's children at level 1,
    grandchildren at level 2."""
    grandchildren = [
        {"size": 80, "subtrees": 3},
        {"size": 60, "subtrees": 2},
    ]
    child = {"size": 150, "subtrees": grandchildren}
    root = {"size": 200, "subtrees": [child]}
    result = collect_nodes_by_level(root)
    assert set(result.keys()) == {1, 2}
    assert result[1] == [child]
    assert set(id(n) for n in result[2]) == {id(gc) for gc in grandchildren}


def test_cnbl_multiple_children_same_level():
    """Sibling nodes that both qualify are both placed at level 1."""
    child_a = {"size": 100, "subtrees": 3}
    child_b = {"size": 80, "subtrees": 2}
    root = {"size": 200, "subtrees": [child_a, child_b]}
    result = collect_nodes_by_level(root)
    assert set(result.keys()) == {1}
    assert set(id(n) for n in result[1]) == {id(child_a), id(child_b)}


def test_cnbl_partial_qualification():
    """Among sibling nodes only the qualifying one is collected; the
    disqualified sibling is excluded from the level."""
    qualifying = {"size": 100, "subtrees": 3}
    disqualified = {"size": 30, "subtrees": 2}
    root = {"size": 200, "subtrees": [qualifying, disqualified]}
    result = collect_nodes_by_level(root)
    assert result == {1: [qualifying]}


def test_cnbl_custom_min_instances():
    """A lower min_instances value allows nodes that would otherwise be
    excluded to pass the threshold and appear in the result."""
    node_a = {"size": 100, "subtrees": 3}
    node_b = {"size": 30, "subtrees": 2}
    root = {"size": 200, "subtrees": [node_a, node_b]}
    result = collect_nodes_by_level(root, min_instances=20)
    assert set(result.keys()) == {1}
    assert set(id(n) for n in result[1]) == {id(node_a), id(node_b)}


def test_cnbl_pruning_on_disqualified_node():
    """When a node fails the threshold, its entire subtree is pruned:
    children are never visited even if they would qualify."""
    large_child = {"size": 100, "subtrees": 2}
    small_parent = {"size": 30, "subtrees": [large_child]}
    root = {"size": 200, "subtrees": [small_parent]}
    assert collect_nodes_by_level(root) == {}


def test_cnbl_leaf_root_returns_empty():
    """When the root's subtrees field is an int rather than a list, there are
    no children to traverse and the result is an empty dict."""
    root = {"size": 200, "subtrees": 10}
    assert collect_nodes_by_level(root) == {}


def test_cnbl_three_levels():
    """A three-level tree assigns nodes at levels 1, 2, and 3 respectively."""
    grandchild = {"size": 60, "subtrees": 1}
    child = {"size": 120, "subtrees": [grandchild]}
    root = {"size": 200, "subtrees": [child]}
    result = collect_nodes_by_level(root)
    assert set(result.keys()) == {1, 2}
    assert result[1] == [child]
    assert result[2] == [grandchild]


def test_cnbl_deep_level_skipped_when_parent_pruned():
    """A qualifying node at level 3 is never reached when its level-2 parent
    fails the threshold, because pruning stops traversal at the parent."""
    deep_node = {"size": 100, "subtrees": 1}
    pruned_mid = {"size": 20, "subtrees": [deep_node]}
    qualifying_top = {"size": 150, "subtrees": [pruned_mid]}
    root = {"size": 200, "subtrees": [qualifying_top]}
    result = collect_nodes_by_level(root)
    assert set(result.keys()) == {1}
    assert result[1] == [qualifying_top]


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
