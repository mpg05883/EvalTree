from src.utils.capability_tree import collect_nodes


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
