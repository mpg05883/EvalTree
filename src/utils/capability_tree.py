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


def preview_tree(
    root: dict, model: str, num_levels: int = 2, max_characters: int = 120,
) -> "plt.Figure":
    """Display the first num_levels levels of a capability tree as a styled diagram.

    Renders each node as a dark-blue rounded box showing the capability description,
    instance count, model score, and CI (where available). The highest-scoring node
    gets an orange highlight border. Style mirrors the EvalTree web demo.

    Args:
        root: The root node of the capability tree (from load_capability_tree).
        model: The model name whose score to display at each node.
        num_levels: Maximum depth to display (root = level 1).
        max_characters: Maximum characters of the capability description to wrap
            inside each node.

    Returns:
        The matplotlib Figure.
    """
    import textwrap
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, PathPatch
    from matplotlib.path import Path

    # ── colours ───────────────────────────────────────────────────────────────
    C_FIG    = "#dce4f0"   # figure background
    C_NODE   = "#2d4e8c"   # node fill
    C_BORDER = "#3d6aad"   # default node border
    C_HILITE = "#f5a623"   # orange border for best-scoring node
    C_TITLE  = "#f5a623"   # first capability line (bold orange)
    C_BODY   = "#dce8ff"   # remaining capability lines
    C_INST   = "#40d0c0"   # instance count (teal)
    C_PIL_BG = "#1a3575"   # pill background
    C_SCORE  = "#f5a623"   # score pill text
    C_CI     = "#e070d0"   # CI pill text (pink)
    C_EDGE   = "#7ab0d4"   # bezier edge colour

    # ── geometry ──────────────────────────────────────────────────────────────
    X_GAP  = 1.0    # horizontal gap between leaf centres
    Y_GAP  = 2.4    # vertical gap between level centres
    NW     = 0.86   # node box width
    NH     = 1.35   # node box height
    PILL_H = 0.16   # height of score / CI pills

    # ── helpers ───────────────────────────────────────────────────────────────
    def _score(node: dict) -> float | None:
        return next((s for m, s in node.get("ranking", []) if m == model), None)

    def _ci(node: dict) -> tuple | None:
        ci = node.get("CI", {})
        return tuple(ci[model]) if isinstance(ci, dict) and model in ci else None

    # ── step 1: collect nodes ─────────────────────────────────────────────────
    nodes: list[dict] = []
    children: dict[int, list[int]] = {}

    def _collect(node: dict, depth: int, parent: int | None) -> None:
        idx = len(nodes)
        cap = textwrap.shorten(node.get("capability", ""), width=max_characters, placeholder="…")
        cap_lines = textwrap.wrap(cap, width=28)[:4]
        nodes.append({
            "lines":  cap_lines,
            "size":   node.get("size", "?"),
            "score":  _score(node),
            "ci":     _ci(node),
            "depth":  depth,
            "parent": parent,
            "x":      0.0,
        })
        children[idx] = []
        if parent is not None:
            children[parent].append(idx)
        if depth < num_levels:
            subtrees = node.get("subtrees", [])
            if isinstance(subtrees, list):
                for child in subtrees:
                    _collect(child, depth + 1, idx)

    _collect(root, 1, None)

    # ── step 2: assign x positions (centred over leaves) ──────────────────────
    leaf_x = [0]

    def _assign_x(idx: int) -> None:
        kids = children[idx]
        if not kids:
            nodes[idx]["x"] = float(leaf_x[0]) * X_GAP
            leaf_x[0] += 1
        else:
            for k in kids:
                _assign_x(k)
            nodes[idx]["x"] = sum(nodes[k]["x"] for k in kids) / len(kids)

    _assign_x(0)
    n_leaves = leaf_x[0]

    # ── step 3: find best-scoring node for orange border ──────────────────────
    scored   = [(i, n["score"]) for i, n in enumerate(nodes) if n["score"] is not None]
    best_idx = max(scored, key=lambda t: t[1])[0] if scored else -1

    # ── step 4: build figure ──────────────────────────────────────────────────
    fig_w = max(12, n_leaves * 2.4)
    fig_h = max(5, num_levels * Y_GAP + NH + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(C_FIG)
    ax.set_facecolor(C_FIG)
    x_pad = NW * 0.6 + 0.2
    ax.set_xlim(-x_pad, (n_leaves - 1) * X_GAP + x_pad)
    ax.set_ylim(-(num_levels - 1) * Y_GAP - NH * 0.6, NH * 0.6)
    ax.axis("off")

    # ── step 5: curved bezier edges (drawn before nodes) ──────────────────────
    for idx, node in enumerate(nodes):
        if node["parent"] is None:
            continue
        px  = nodes[node["parent"]]["x"]
        py  = -(nodes[node["parent"]]["depth"] - 1) * Y_GAP
        cx_ = node["x"]
        cy  = -(node["depth"] - 1) * Y_GAP
        y0, y1 = py - NH / 2, cy + NH / 2
        ym = (y0 + y1) / 2
        verts = [(px, y0), (px, ym), (cx_, ym), (cx_, y1)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        ax.add_patch(PathPatch(
            Path(verts, codes), facecolor="none", edgecolor=C_EDGE, lw=1.4, zorder=1,
        ))

    # ── step 6: draw nodes ────────────────────────────────────────────────────
    for idx, node in enumerate(nodes):
        x  = node["x"]
        y  = -(node["depth"] - 1) * Y_GAP
        xL = x - NW / 2
        yB = y - NH / 2

        # Background box
        ax.add_patch(FancyBboxPatch(
            (xL, yB), NW, NH,
            boxstyle="round,pad=0.04",
            facecolor=C_NODE,
            edgecolor=C_HILITE if idx == best_idx else C_BORDER,
            linewidth=2.5 if idx == best_idx else 1.0,
            zorder=2,
        ))

        has_ci  = node["ci"] is not None
        n_lines = len(node["lines"])

        # Capability text: fixed-pitch lines stacked from the top of the node
        LINE_H   = NH * 0.155   # fixed line pitch
        text_top = yB + NH * 0.93
        for i, line in enumerate(node["lines"]):
            ly = text_top - i * LINE_H
            ax.text(
                x, ly, line,
                ha="center", va="top",
                fontsize=7 if i == 0 else 6.5,
                fontweight="bold" if i == 0 else "normal",
                color=C_TITLE if i == 0 else C_BODY,
                zorder=3,
            )

        # Instance count
        inst_y = yB + NH * (0.30 if has_ci else 0.33)
        ax.text(
            x, inst_y, f"{node['size']} instances",
            ha="center", va="center", fontsize=7, fontweight="bold",
            color=C_INST, zorder=3,
        )

        # Score pill
        if node["score"] is not None:
            score_y = yB + NH * (0.18 if has_ci else 0.16)
            pw = NW * 0.82
            ax.add_patch(FancyBboxPatch(
                (x - pw / 2, score_y - PILL_H / 2), pw, PILL_H,
                boxstyle="round,pad=0.01",
                facecolor=C_PIL_BG, edgecolor="none", zorder=3,
            ))
            ax.text(
                x, score_y, f"{node['score']:.3f} Score",
                ha="center", va="center", fontsize=7, fontweight="bold",
                color=C_SCORE, zorder=4,
            )

        # CI pill
        if has_ci:
            ci_y = yB + NH * 0.06
            lo, hi = node["ci"]
            pw = NW * 0.82
            ax.add_patch(FancyBboxPatch(
                (x - pw / 2, ci_y - PILL_H / 2), pw, PILL_H,
                boxstyle="round,pad=0.01",
                facecolor=C_PIL_BG, edgecolor="none", zorder=3,
            ))
            ax.text(
                x, ci_y, f"95% CI: [{lo:.3f}, {hi:.3f}]",
                ha="center", va="center", fontsize=7, fontweight="bold",
                color=C_CI, zorder=4,
            )

    fig.suptitle(
        f"Capability Tree — {model}  ({num_levels} levels)",
        fontsize=10, color="#333333", y=0.99,
    )
    plt.tight_layout()
    return fig


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
