import os
import re
import random
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# =====================================================================
# 1. PARSING & UTILITIES
# =====================================================================


def parse_man_topo(filepath):
    """Extracts base adjacency, nodes, MAN code, and EdgeString."""
    filepath = Path(filepath)

    # Matches the MAN code, then captures everything after the first underscore
    match = re.match(r"^([0-9A-Z]+)_(.+)$", filepath.stem)
    if not match:
        raise ValueError(f"Filename {filepath.name} does not match MAN convention.")

    man_code, edge_string = match.groups()

    with filepath.open("r") as f:
        delimiter = "," if "," in f.readline() else r"\s+"

    df = pd.read_csv(filepath, sep=delimiter).replace({2: -1})
    G = nx.from_pandas_edgelist(
        df, source="Source", target="Target", edge_attr="Type", create_using=nx.DiGraph
    )

    return nx.to_numpy_array(G, weight="Type"), list(G.nodes()), man_code, edge_string


def save_edgelist(adj, node_names, filepath):
    """Converts a dense array to an edgelist and saves to file."""
    df = pd.DataFrame(adj, index=node_names, columns=node_names)
    edgelist = df.stack()[df.stack() != 0].reset_index()
    edgelist.columns = ["Source", "Target", "Type"]
    edgelist["Type"] = edgelist["Type"].astype(int)
    edgelist.to_csv(filepath, sep=" ", index=False)


def get_random_spanning_backbone(allowed_mask):
    """
    Extracts a Weakly Connected Spanning Forest using randomized weights.
    Mathematically guarantees connectivity without introducing 'star-hub' artifacts.
    """
    G = nx.from_numpy_array(allowed_mask, create_using=nx.DiGraph)
    G_un = G.to_undirected(as_view=False)

    # Assign random weights to prevent lexicographical bias
    for u, v in G_un.edges():
        G_un[u][v]["weight"] = random.random()

    forest = nx.Graph()
    for comp in nx.connected_components(G_un):
        tree = nx.minimum_spanning_tree(G_un.subgraph(comp), weight="weight")
        forest.add_edges_from(tree.edges())

    protected_edges = []
    # Map the undirected tree back to valid directed edges
    for u, v in forest.edges():
        valid_dirs = []
        if G.has_edge(u, v):
            valid_dirs.append((u, v))
        if G.has_edge(v, u):
            valid_dirs.append((v, u))
        if valid_dirs:
            protected_edges.append(random.choice(valid_dirs))

    return protected_edges


def plot_motif_coherence_numberline(df, base_filepath="structural_coherence"):
    """
    Generates a gradient bar for MeanCoh with a Nord diverging color palette.
    Annotates the MAN-PNStr motifs and their exact values below the bar
    with improved collision-avoidance staggering.
    """
    # Attempt to set Roboto; falls back to standard sans-serif if not installed
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Roboto", "Arial", "Helvetica", "DejaVu Sans"]

    # Slightly wider and taller figure to accommodate wider text strings
    fig, ax = plt.subplots(figsize=(16, 6))

    # Define Nord diverging palette: Red (-1) -> Snow (0) -> Blue (+1)
    nord_red = "#BF616A"
    nord_zero = "#ECEFF4"
    nord_blue = "#5E81AC"
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "nord_diverging", [nord_red, nord_zero, nord_blue]
    )

    bar_height = 0.15
    bar_y_start = 0

    # 1. Draw the gradient bar
    gradient = np.linspace(-1, 1, 500).reshape(1, -1)
    ax.imshow(
        gradient,
        aspect="auto",
        cmap=cmap,
        extent=[-1, 1, bar_y_start, bar_y_start + bar_height],
        zorder=1,
    )

    # 2. Add border to the bar
    rect = patches.Rectangle(
        (-1, bar_y_start),
        2,
        bar_height,
        fill=False,
        edgecolor="#2E3440",
        lw=2,
        zorder=2,
    )
    ax.add_patch(rect)

    # 3. Add Top Annotations ("Structural Coherence", "-1", "+1")
    ax.text(
        0,
        bar_height + 0.05,
        "Structural Coherence",
        ha="center",
        va="bottom",
        fontsize=16,
        color="#2E3440",
        weight="bold",
    )
    ax.text(
        -1,
        bar_height + 0.05,
        "-1",
        ha="center",
        va="bottom",
        fontsize=14,
        color="#BF616A",
        weight="bold",
    )
    ax.text(
        1,
        bar_height + 0.05,
        "+1",
        ha="center",
        va="bottom",
        fontsize=14,
        color="#5E81AC",
        weight="bold",
    )

    # Add a center 0 tick on the bar for reference
    ax.plot(
        [0, 0],
        [bar_y_start, bar_y_start + bar_height],
        color="#2E3440",
        lw=1.5,
        ls="--",
        zorder=3,
    )
    ax.text(
        0, bar_y_start - 0.05, "0", ha="center", va="top", fontsize=12, color="#2E3440"
    )

    # 4. Filter, Sort, and Plot Motifs
    df_plot = (
        df.drop_duplicates(subset=["MAN-PNStr"])
        .sort_values("MeanCoh")
        .reset_index(drop=True)
    )

    # Deeper and more spaced out y_levels to handle wider text and extreme clustering at 1.0/-1.0
    y_levels = [-0.15, -0.35, -0.55, -0.75, -0.95, -1.15, -1.35, -1.55]
    last_x_at_level = {i: -999 for i in range(len(y_levels))}

    for _, row in df_plot.iterrows():
        x = row["MeanCoh"]
        # Append the value to the text string
        txt = f"{row['MAN-PNStr']} ({x:.2f})"

        # Dynamic collision detection
        best_level_idx = 0
        for i in range(len(y_levels)):
            # Increased threshold from 0.08 to 0.18 because the text is wider now
            if abs(x - last_x_at_level[i]) > 0.18:
                best_level_idx = i
                break

        y_target = y_levels[best_level_idx]
        last_x_at_level[best_level_idx] = x

        # Draw connector line
        ax.plot([x, x], [bar_y_start, y_target + 0.05], color="#4C566A", lw=1, zorder=2)

        # Draw the marker point
        ax.plot(x, bar_y_start, marker="v", color="#2E3440", markersize=6, zorder=4)

        # Annotate the motif string + value (alpha=1.0 for solid background)
        ax.text(
            x,
            y_target,
            txt,
            ha="center",
            va="top",
            fontsize=9,
            color="#2E3440",
            bbox=dict(
                facecolor="#ECEFF4",
                edgecolor="#D8DEE9",
                boxstyle="round,pad=0.3",
                alpha=1.0,
            ),
            zorder=5,
        )

    # 5. Clean up axes limits and remove grid/spines
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(min(y_levels) - 0.2, bar_height + 0.3)
    ax.axis("off")

    plt.tight_layout()

    # Save with solid white background instead of transparent
    plt.savefig(
        f"{base_filepath}.svg", format="svg", bbox_inches="tight", facecolor="white"
    )
    plt.savefig(
        f"{base_filepath}.png",
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def plot_diverging_coherence_numberline(
    df, motif_col="MAN-PNStr", val_col="MeanCoh", base_filepath="coherence_numberline"
):
    """
    Generates a horizontal diverging gradient bar (-1 to +1) for structural coherence.
    Utilizes a topological bipartite mapping algorithm to distribute text annotations
    into pre-calculated slots around the perimeter of the figure.
    Guarantees zero line crossings while preserving a clean, radial fan-out aesthetic.
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Roboto", "Arial", "Helvetica", "DejaVu Sans"]

    # Extra wide figure to give massive breathing room for the text columns and long lines
    fig, ax = plt.subplots(figsize=(20, 8))

    # Nord Colors
    nord_red = "#BF616A"
    nord_white = "#ECEFF4"
    nord_blue = "#5E81AC"
    nord_dark = "#2E3440"

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "nord_diverging", [nord_red, nord_white, nord_blue]
    )
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

    bar_height = 0.15
    bar_edge = bar_height / 2

    # 1. Draw the horizontal gradient bar
    gradient = np.linspace(-1, 1, 500).reshape(1, -1)
    ax.imshow(
        gradient,
        aspect="auto",
        cmap=cmap,
        extent=[-1, 1, -bar_edge, bar_edge],
        zorder=1,
    )

    # Sharp, crisp border around the bar
    rect = patches.Rectangle(
        (-1, -bar_edge),
        2,
        bar_height,
        fill=False,
        edgecolor=nord_dark,
        lw=2.5,
        zorder=2,
    )
    ax.add_patch(rect)

    # Center Zero Marker
    # ax.plot([0, 0], [-bar_edge, bar_edge], color=nord_dark, lw=1.5, ls="--", zorder=3)
    ax.text(
        0,
        bar_edge + 0.04,
        "0.0",
        ha="center",
        va="bottom",
        fontsize=16,
        color=nord_dark,
        # weight="bold",
    )

    # 2. Setup Sequential Fanning Logic (Topological Mapping)
    df_plot = (
        df.drop_duplicates(subset=[motif_col])
        .sort_values(val_col, ascending=True)
        .reset_index(drop=True)
    )

    # Split items strictly by value to keep left side negative and right side positive
    left_df = df_plot[df_plot[val_col] < 0].copy()
    right_df = df_plot[df_plot[val_col] >= 0].copy()

    def get_y_slots(n):
        """Generates uniformly spaced Y-coordinates, explicitly leaving the center open."""
        if n == 0:
            return []
        if n == 1:
            return [0.6]  # Safe default off-center

        n_bot = n // 2
        n_top = n - n_bot

        def safe_linspace(start, end, num):
            if num == 0:
                return []
            if num == 1:
                return [(start + end) / 2]
            return list(np.linspace(start, end, num))

        bot_ys = safe_linspace(-1.3, -0.3, n_bot)
        top_ys = safe_linspace(0.3, 1.3, n_top)

        return bot_ys + top_ys

    # Mathematical Guarantee for No Crossings:
    # Left hemisphere: Origins ascending, Targets ascending.
    # Right hemisphere: Origins ascending, Targets descending.
    left_ys = sorted(get_y_slots(len(left_df)), reverse=False)
    right_ys = sorted(get_y_slots(len(right_df)), reverse=True)

    def plot_hemisphere(data, y_slots, is_left):
        for i, (_, row) in enumerate(data.iterrows()):
            val = row[val_col]
            tgt_y = y_slots[i]

            # X Coordinates for the kink routing
            text_x = -1.45 if is_left else 1.45
            kink_x = -1.35 if is_left else 1.35
            ha = "right" if is_left else "left"

            color = cmap(norm(val))

            # Draw the radial slant and the horizontal kink
            ax.plot(
                [val, kink_x, text_x],
                [0, tgt_y, tgt_y],
                color=color,
                lw=2.5,
                zorder=4,
                solid_joinstyle="round",
                solid_capstyle="round",
            )

            # Anchor dot exactly on the number line
            ax.plot(
                val,
                0,
                marker="o",
                markersize=10,
                color=color,
                markeredgecolor=nord_dark,
                zorder=5,
            )

            # Text box formatting
            label = f"{str(row[motif_col]).replace('_', '-')} ({val:.2f})"

            # Small offset to push text away from the absolute end of the line
            text_offset_x = text_x + (0.02 if ha == "left" else -0.02)

            ax.text(
                text_offset_x,
                tgt_y,
                label,
                ha=ha,
                va="center",
                fontsize=11,
                color=nord_dark,
                bbox=dict(
                    facecolor=nord_white,
                    edgecolor="#D8DEE9",
                    boxstyle="round,pad=0.4",
                    alpha=1.0,
                ),
                zorder=6,
            )

    # 3. Plotting Execution
    plot_hemisphere(left_df, left_ys, is_left=True)
    plot_hemisphere(right_df, right_ys, is_left=False)

    # 4. Dynamic scaling to ensure SVG headroom and text margins
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{base_filepath}.svg", format="svg", bbox_inches="tight", facecolor="white"
    )
    plt.savefig(
        f"{base_filepath}.png",
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


# =====================================================================
# 2. CORE NETWORK GENERATORS
# =====================================================================


def calculate_hierarchy_indices(base_dim, scale, proportions):
    """Calculates robust node boundaries for hierarchical subgroups."""
    raw_counts = np.array(proportions) * scale
    node_counts = np.round(raw_counts).astype(int)

    # Ensure minimum 1 node per layer if proportion > 0
    for i in range(len(node_counts)):
        if node_counts[i] == 0 and proportions[i] > 0:
            node_counts[i] = 1

    # Rebalance to ensure total matches scale
    diff = scale - node_counts.sum()
    if diff != 0:
        target_idx = (
            np.argmax(node_counts)
            if node_counts[np.argmax(node_counts)] + diff >= 1
            else 0
        )
        node_counts[target_idx] += diff

    boundaries = [0] + np.cumsum(node_counts * base_dim).tolist()

    # Create the sorting permutation array (P)
    grouped_indices = [[] for _ in proportions]
    start_nodes = np.cumsum([0] + list(node_counts))
    for k in range(len(proportions)):
        for i in range(base_dim):
            grouped_indices[k].extend(
                range(i * scale + start_nodes[k], i * scale + start_nodes[k + 1])
            )

    P = np.concatenate(grouped_indices).astype(int)
    return grouped_indices, boundaries, P


def generate_er_network(base_adj, scale, global_density, seed=None):
    """Generates a connected ER network using random sparsification."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    dense_adj = np.kron(base_adj, np.ones((scale, scale)))
    allowed_mask = (dense_adj != 0).astype(float)

    # 1. Establish connectivity backbone
    protected_edges = get_random_spanning_backbone(allowed_mask)
    er_adj = np.zeros_like(dense_adj)
    for u, v in protected_edges:
        er_adj[u, v] = dense_adj[u, v]

    # 2. Random Fill up to target density
    target_edges = int(np.count_nonzero(allowed_mask) * global_density)
    needed = max(0, target_edges - len(protected_edges))

    if needed > 0:
        empty_slots = [
            (u, v) for u, v in zip(*np.where(allowed_mask == 1)) if er_adj[u, v] == 0
        ]
        chosen = random.sample(empty_slots, min(needed, len(empty_slots)))
        for u, v in chosen:
            er_adj[u, v] = dense_adj[u, v]

    return er_adj


def generate_hi_network(
    base_adj, scale, proportions, h_matrix, density_constraints, seed=None
):
    """Generates a hierarchical network with Same-Team Rescue logic."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    dense_adj = np.kron(base_adj, np.ones((scale, scale)))
    num_layers = len(proportions)
    get_team = lambda node: node // scale

    grouped_indices, boundaries, P = calculate_hierarchy_indices(
        base_adj.shape[0], scale, proportions
    )

    # 1. Build Layer-Restricted Allowed Mask
    hi_allowed_mask = np.zeros_like(dense_adj)
    for i in range(num_layers):
        for j in range(num_layers):
            if h_matrix[i, j] < 1.0:  # Rule permits connection
                for u in grouped_indices[i]:
                    for v in grouped_indices[j]:
                        if dense_adj[u, v] != 0:  # Base topology permits connection
                            hi_allowed_mask[u, v] = 1.0

    # 2. Establish connectivity backbone within the strict hierarchy
    protected_edges = get_random_spanning_backbone(hi_allowed_mask)
    hi_adj = np.zeros_like(dense_adj)
    for u, v in protected_edges:
        hi_adj[u, v] = dense_adj[u, v]

    # 3. Calculate Budgets per Hierarchical Block
    block_budgets = {}
    for i in range(num_layers):
        for j in range(num_layers):
            if h_matrix[i, j] >= 1.0:
                continue

            slots = sum(
                1
                for u in grouped_indices[i]
                for v in grouped_indices[j]
                if hi_allowed_mask[u, v] > 0
            )
            target_edges = int(slots * density_constraints[i, j])
            current_edges = sum(
                1
                for u in grouped_indices[i]
                for v in grouped_indices[j]
                if hi_adj[u, v] != 0
            )
            block_budgets[(i, j)] = max(0, target_edges - current_edges)

    # 4. Same-Team Rescue Operation
    for j in range(1, num_layers):
        targets = grouped_indices[j].copy()
        random.shuffle(targets)
        upstream = [i for i in range(j) if h_matrix[i, j] < 1.0]

        for v in targets:
            if any(get_team(u) == get_team(v) for u in np.where(hi_adj[:, v] != 0)[0]):
                continue

            feasible = []
            for i in upstream:
                if block_budgets[(i, j)] <= 0:
                    continue
                for u in grouped_indices[i]:
                    if (
                        get_team(u) == get_team(v)
                        and hi_adj[u, v] == 0
                        and hi_allowed_mask[u, v] > 0
                    ):
                        feasible.append((i, u))
            if feasible:
                i, u = random.choice(feasible)
                hi_adj[u, v] = dense_adj[u, v]
                block_budgets[(i, j)] -= 1

    # 5. Random Fill Remaining Budget
    for (i, j), needed in block_budgets.items():
        if needed <= 0:
            continue
        empty_slots = [
            (u, v)
            for u in grouped_indices[i]
            for v in grouped_indices[j]
            if hi_allowed_mask[u, v] > 0 and hi_adj[u, v] == 0
        ]
        if empty_slots:
            chosen = random.sample(empty_slots, min(needed, len(empty_slots)))
            for u, v in chosen:
                hi_adj[u, v] = dense_adj[u, v]

    return hi_adj, P, boundaries


# =====================================================================
# 3. PLOTTING & VERIFICATION
# =====================================================================

NORD_COLORS = {
    "dark": "#2e3440",
    "gray": "#3b4252",
    "red": "#bf616a",
    "blue": "#5e81ac",
    "green": "#a3be8c",
    "yellow": "#ebcb8b",
    "purple": "#b48ead",
    "orange": "#d08770",
    "cyan": "#8fbcbb",
    "light_blue": "#88c0d0",
}

NORD_PALETTE = [
    NORD_COLORS["red"],
    NORD_COLORS["blue"],
    NORD_COLORS["green"],
    NORD_COLORS["yellow"],
    NORD_COLORS["purple"],
    NORD_COLORS["cyan"],
    NORD_COLORS["orange"],
]


def set_global_nord_style():
    """Configures Matplotlib global settings for the Nord aesthetic."""
    plt.style.use("default")

    # Let Seaborn dynamically scale all fonts, lines, and markers for a larger canvas.
    # 'talk' is usually perfect for presentations. Use 'poster' if it's still too small.
    sns.set_context("paper", font_scale=1.6)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Roboto", "Arial", "DejaVu Sans"],
            # You can remove the hardcoded font.size and labelsize here,
            # as sns.set_context handles the hierarchy automatically.
            "text.color": NORD_COLORS["dark"],
            "axes.labelcolor": NORD_COLORS["dark"],
            "axes.titlecolor": NORD_COLORS["dark"],
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": NORD_COLORS["gray"],
            "axes.linewidth": 1.5,
            "xtick.color": NORD_COLORS["dark"],
            "ytick.color": NORD_COLORS["dark"],
            # "grid.color": NORD_COLORS["gray"],
            # "grid.alpha": 0.3,
            "axes.grid": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": True,
        }
    )


def plot_adjacency_with_density(
    adj,
    title,
    out_filepath,
    boundaries=None,
    sort_idx=None,
    labels=["IN", "MID", "OUT"],
):
    """Visualizes matrix. Sorts by hierarchy and overlays density boxes if provided."""
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    plot_adj = adj[sort_idx, :][:, sort_idx] if sort_idx is not None else adj

    cmap = mcolors.ListedColormap(["#FF3333", "#FFFFFF", "#3366FF"])
    norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    img = ax.imshow(
        plot_adj, cmap=cmap, norm=norm, origin="upper", interpolation="none"
    )
    plt.colorbar(img, ticks=[-1, 0, 1], shrink=0.8)

    actual_edges = np.count_nonzero(adj)
    density_val = actual_edges / (adj.shape[0] * adj.shape[1])

    # Overlay Hierarchical Boxes & Inner Densities
    if boundaries is not None and sort_idx is not None:
        num_lvls = len(boundaries) - 1
        for i in range(num_lvls):
            for j in range(num_lvls):
                r_start, r_end = boundaries[i], boundaries[i + 1]
                c_start, c_end = boundaries[j], boundaries[j + 1]
                if r_start == r_end or c_start == c_end:
                    continue

                submat = plot_adj[r_start:r_end, c_start:c_end]
                block_edges = np.count_nonzero(submat)
                block_density = block_edges / submat.size if submat.size > 0 else 0.0

                width, height = c_end - c_start, r_end - r_start
                ax.add_patch(
                    patches.Rectangle(
                        (c_start - 0.5, r_start - 0.5),
                        width,
                        height,
                        lw=2.5,
                        edgecolor="#2e3440",
                        facecolor="none",
                    )
                )

                if block_edges > 0:
                    lbl_str = (
                        f"{labels[i]}->{labels[j]}\n"
                        if i < len(labels) and j < len(labels)
                        else ""
                    )
                    ax.text(
                        c_start + width / 2 - 0.5,
                        r_start + height / 2 - 0.5,
                        f"{lbl_str}{block_density:.2f}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="#eceff4",
                        bbox=dict(
                            facecolor="#2e3440", alpha=0.8, edgecolor="none", pad=2
                        ),
                    )

    plt.title(
        f"{title}\nNodes: {adj.shape[0]} | Edges: {actual_edges} | Global Grid Density: {density_val:.3f}"
    )
    plt.tight_layout()
    plt.savefig(out_filepath, dpi=150)
    plt.close()


def get_representative_rows(group):
    """
    Directly selects specific base motifs based on a hardcoded list,
    bypassing algorithmic filtering, while preserving the original
    output format and sorting.
    """
    target_motifs = [
        "030C-N00PN0",
        "100-NN",
        "300-NNNNPP",
        "030T-0N0NP0",
        "030T-0P0NP0",
        "300-NNNNNN",
        "030C-N00NN0",
        "300-NNPNPP",
        "300-PNNPPP",
        "100-PN",
        "300-NPNPNP",
    ]

    # Filter the original group (which contains both SA and NS variants)
    # using the hardcoded list of base motifs.
    result_df = group[group["MAN-PNStr"].isin(target_motifs)].copy()

    # Sort to maintain hierarchical readability
    return result_df.sort_values(["MAN-PNStr", "SelfActivation", "MeanCoh"])


def generate_scaled_networks(
    base_files,
    output_dir,
    scales=[10, 20, 30, 40, 50],
    global_densities=np.round(np.arange(0.1, 1.1, 0.1), 1),
    max_replicates=10,
    proportions=[0.05, 0.05, 0.90],
    h_matrix=1.0 - np.array([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
    hi_density_overrides=np.array(
        [[-1.0, 0.5, -1.0], [-1.0, 1.0, 0.85], [-1.0, -1.0, -1.0]]
    ),
    enable_plotting=False,
):
    """
    Generates scaled ER and Hierarchical networks from base MAN motifs.

    Args:
        base_files (list): List of Paths to the base .topo files.
        output_dir (Union[str, Path]): Directory to save the scaled topologies.
        scales (list): List of scale factors to apply.
        global_densities (np.array): Array of global densities to iterate over.
        max_replicates (int): Maximum number of random replicates for densities < 1.0.
        proportions (list): List of proportions for hierarchical layers.
        h_matrix (np.array): Matrix dictating allowed cross-layer connections.
        hi_density_overrides (np.array): Matrix dictating density constraints per block.
        enable_plotting (bool): Toggles the generation of PNG plots (hardcoded to scale 10, gd 0.5, rep 1).
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "Plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    if enable_plotting:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------ GROUPING BY MAN CODE ------------------
    man_groups = defaultdict(list)
    for base_file in base_files:
        if not base_file.exists():
            print(f"Warning: File not found {base_file}")
            continue
        try:
            _, _, man_code, _ = parse_man_topo(base_file)
            man_groups[man_code].append(base_file)
        except ValueError as e:
            print(f"Error parsing {base_file}: {e}")

    # ------------------ BATCHED EXECUTION LOOP ------------------
    for man_code, files_in_group in tqdm(
        man_groups.items(), desc="Processing MAN Groups"
    ):
        man_cache = {}

        for base_file in files_in_group:
            # Parse the already specific base file (SA or NS variant)
            base_adj, node_list, _, edge_str = parse_man_topo(base_file)
            unsigned_base_adj = np.abs(base_adj)

            for scale in scales:
                node_names = [
                    f"T{i + 1}_{j:03d}"
                    for i in range(base_adj.shape[0])
                    for j in range(1, scale + 1)
                ]

                dense_signed_adj = np.kron(base_adj, np.ones((scale, scale)))

                for gd in global_densities:
                    num_reps = 1 if gd == 1.0 else max_replicates

                    for rep in range(1, num_reps + 1):
                        density_prefix = f"{int(gd * 100):03d}D"

                        cache_key = (edge_str, scale, gd, rep)

                        if cache_key not in man_cache:
                            er_mask = generate_er_network(
                                unsigned_base_adj, scale, gd, seed=rep
                            )
                            hi_mask, sort_idx, boundaries = generate_hi_network(
                                unsigned_base_adj,
                                scale,
                                proportions,
                                h_matrix,
                                hi_density_overrides,
                                seed=rep,
                            )
                            man_cache[cache_key] = (
                                er_mask,
                                hi_mask,
                                sort_idx,
                                boundaries,
                            )

                        er_mask, hi_mask, sort_idx, boundaries = man_cache[cache_key]

                        # ER
                        er_name = f"{man_code}_{edge_str}_{scale:03d}N_{density_prefix}_ER_{rep:03d}R"
                        er_adj = er_mask * dense_signed_adj
                        save_edgelist(
                            er_adj, node_names, output_dir / f"{er_name}.topo"
                        )

                        # HI
                        hi_name = f"{man_code}_{edge_str}_{scale:03d}N_{density_prefix}_HI_{rep:03d}R"
                        hi_adj = hi_mask * dense_signed_adj
                        save_edgelist(
                            hi_adj, node_names, output_dir / f"{hi_name}.topo"
                        )

                        # --- CONDITIONAL PLOTTING ---
                        if enable_plotting and scale == 10 and gd == 0.5 and rep == 1:
                            plot_adjacency_with_density(
                                er_adj,
                                er_name,
                                plots_dir / f"{er_name}.png",
                                boundaries=boundaries,
                            )
                            plot_adjacency_with_density(
                                hi_adj,
                                hi_name,
                                plots_dir / f"{hi_name}.png",
                                boundaries=boundaries,
                                sort_idx=sort_idx,
                            )


def plot_motif_coherence_matrix(
    motif_name, base_dir, out_base_dir="/home/csb/Projects00/MotifFinding/AllUniqueNets"
):
    """
    Reads a coherence matrix and team mapping, reorders the matrix by teams,
    and plots it as a heatmap with black boxes highlighting the diagonal team blocks.
    Saves outputs to separate SVG and PNG directories without a colorbar.
    X-axis ticks are set to a standard 0-degree horizontal rotation.
    """
    set_global_nord_style()

    base_dir = Path(base_dir)
    motif_dir = base_dir / motif_name

    # Establish output directories
    out_base_dir = Path(out_base_dir)
    png_dir = out_base_dir / "CohMat_PNG"
    svg_dir = out_base_dir / "CohMat_SVG"

    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)

    cohmat_path = motif_dir / f"{motif_name}_CohMat.parquet"
    teams_path = motif_dir / f"{motif_name}_Teams.csv"

    if not cohmat_path.exists() or not teams_path.exists():
        print(f"Missing data files for {motif_name} in {motif_dir}. Skipping...")
        return

    # 1. Load and process the Teams mapping
    teams_df = pd.read_csv(teams_path)

    # Sort by Group first to clump teams together, then alphabetically by Node
    teams_df = teams_df.sort_values(by=["Group", "Node"]).reset_index(drop=True)
    ordered_nodes = teams_df["Node"].tolist()

    # Calculate exact block boundary coordinates for the bounding boxes
    group_sizes = teams_df.groupby("Group", sort=False).size().tolist()
    boundaries = [0] + np.cumsum(group_sizes).tolist()

    # 2. Load and reorder the Coherence Matrix
    cohmat_df = pd.read_parquet(cohmat_path)

    # Failsafe: Ensure index is correctly set to node names if not natively preserved
    if cohmat_df.index.name != "Node" and cohmat_df.index[0] not in ordered_nodes:
        if (
            cohmat_df.columns[0] in ordered_nodes
            or cohmat_df.iloc[0, 0] in ordered_nodes
        ):
            cohmat_df = cohmat_df.set_index(cohmat_df.columns[0])

    # Subselect and symmetrically reorder rows & columns using the hierarchical mapping
    reordered_cohmat = cohmat_df.loc[ordered_nodes, ordered_nodes]

    fig, ax = plt.subplots(figsize=(3, 3))

    # Nord Diverging Palette: Red (-1) -> White (0) -> Blue (+1)
    nord_red = "#BF616A"
    nord_white = "#FFFFFF"
    nord_blue = "#5E81AC"
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "nord_diverging", [nord_red, nord_white, nord_blue]
    )

    # 4. Plot the Heatmap
    sns.heatmap(
        reordered_cohmat,
        ax=ax,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        center=0,
        annot=False,
        linewidths=1.2,
        linecolor="white",
        square=True,
        cbar=False,  # Colorbar removed
    )

    # 5. Overlay Black Bounding Boxes for Teams (Diagonal Blocks)
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        size = group_sizes[i]

        rect = patches.Rectangle(
            (start, start),
            size,
            size,
            fill=False,
            edgecolor="#2E3440",
            linewidth=3.5,
            zorder=10,
            clip_on=False,
        )
        ax.add_patch(rect)

    # 6. Aesthetics and Labels
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Rename axis labels to Source and Target
    ax.set_ylabel("Source", labelpad=5)
    ax.set_xlabel("Target", labelpad=5)

    # Turn the x-ticks to horizontal (0 degrees)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    # Clean the motif name for the title
    clean_title = (
        motif_name.replace("_SA", " (SA)").replace("_NS", " (NS)").replace("_", "-")
    )
    plt.title(
        clean_title,
        pad=15,
        fontsize=8,
        color="#2E3440",
    )

    plt.tight_layout()

    # 7. Save outputs to designated directories
    png_path = png_dir / f"{motif_name}_heatmap.png"
    svg_path = svg_dir / f"{motif_name}_heatmap.svg"

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(svg_path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close()


def plot_motif_coherence_matrix(
    motif_name, base_dir, out_base_dir="/home/csb/Projects00/MotifFinding/AllUniqueNets"
):
    """
    Reads a coherence matrix and team mapping, reorders the matrix by teams,
    and plots it as a heatmap with black boxes highlighting the diagonal team blocks.
    Saves outputs to separate SVG and PNG directories without a colorbar.
    X-axis ticks are set to a standard 0-degree horizontal rotation.
    NaN values are explicitly colored with a Nord light grey to distinguish them.
    """
    base_dir = Path(base_dir)
    motif_dir = base_dir / motif_name

    # Establish output directories
    out_base_dir = Path(out_base_dir)
    png_dir = out_base_dir / "CohMat_PNG"
    svg_dir = out_base_dir / "CohMat_SVG"

    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)

    cohmat_path = motif_dir / f"{motif_name}_CohMat.parquet"
    teams_path = motif_dir / f"{motif_name}_Teams.csv"

    if not cohmat_path.exists() or not teams_path.exists():
        print(f"Missing data files for {motif_name} in {motif_dir}. Skipping...")
        return

    # 1. Load and process the Teams mapping
    teams_df = pd.read_csv(teams_path)

    # Sort by Group first to clump teams together, then alphabetically by Node
    teams_df = teams_df.sort_values(by=["Group", "Node"]).reset_index(drop=True)
    ordered_nodes = teams_df["Node"].tolist()

    # Calculate exact block boundary coordinates for the bounding boxes
    group_sizes = teams_df.groupby("Group", sort=False).size().tolist()
    boundaries = [0] + np.cumsum(group_sizes).tolist()

    # 2. Load and reorder the Coherence Matrix
    cohmat_df = pd.read_parquet(cohmat_path)

    # Failsafe: Ensure index is correctly set to node names if not natively preserved
    if cohmat_df.index.name != "Node" and cohmat_df.index[0] not in ordered_nodes:
        if (
            cohmat_df.columns[0] in ordered_nodes
            or cohmat_df.iloc[0, 0] in ordered_nodes
        ):
            cohmat_df = cohmat_df.set_index(cohmat_df.columns[0])

    # Subselect and symmetrically reorder rows & columns using the hierarchical mapping
    reordered_cohmat = cohmat_df.loc[ordered_nodes, ordered_nodes]

    # 3. Setup Plotting and Colors
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Roboto", "Arial", "Helvetica", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(3, 3))

    # Nord Diverging Palette: Red (-1) -> White (0) -> Blue (+1)
    nord_red = "#BF616A"
    nord_white = "#FFFFFF"
    nord_blue = "#5E81AC"
    nord_nan_grey = "#4c566a"  # Nord Light Grey for NaN values

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "nord_diverging", [nord_red, nord_white, nord_blue]
    )
    cmap.set_bad(color=nord_nan_grey)

    # Set axes background to match the NaN color (as seaborn leaves NaNs transparent)
    ax.set_facecolor(nord_nan_grey)

    # 4. Plot the Heatmap
    sns.heatmap(
        reordered_cohmat,
        ax=ax,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        center=0,
        annot=False,
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar=False,
    )

    # 5. Overlay Black Bounding Boxes for Teams (Diagonal Blocks)
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        size = group_sizes[i]

        rect = patches.Rectangle(
            (start, start),
            size,
            size,
            fill=False,
            edgecolor="#2E3440",
            linewidth=3.5,
            zorder=10,
            clip_on=False,
        )
        ax.add_patch(rect)

    # 6. Aesthetics and Labels
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Rename axis labels to Source and Target
    ax.set_ylabel("Source", fontsize=9, labelpad=20)
    ax.set_xlabel("Target", fontsize=9, labelpad=20)

    # Turn the x-ticks to horizontal (0 degrees)
    ax.tick_params(axis="x", rotation=0, labelsize=20)
    ax.tick_params(axis="y", rotation=0, labelsize=20)

    # Clean the motif name for the title
    clean_title = (
        motif_name.replace("_SA", " (SA)").replace("_NS", " (NS)").replace("_", "-")
    )
    plt.title(
        clean_title,
        pad=15,
        fontsize=8,
        color="#2E3440",
    )

    plt.tight_layout()

    # 7. Save outputs to designated directories
    png_path = png_dir / f"{motif_name}_heatmap.png"
    svg_path = svg_dir / f"{motif_name}_heatmap.svg"

    plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(svg_path, format="svg", bbox_inches="tight", transparent=True)
    plt.close()


# =====================================================================
# 4. MAIN PIPELINE
# =====================================================================

if __name__ == "__main__":
    input_dir = Path("./AllUniqueNets/Topologies/")
    output_dir = Path("./ScaledTopos")
    plots_dir = output_dir / "Plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    ####################################################################
    ## Read the Labelled Circuit List
    circuit_df = pd.read_csv("./unified_labeled_circuits.csv")
    ## Read the compiled cohmat data for the circuits
    circuits_cohmat_df = pd.read_csv("./MotifCohResults/CompiledMotifSummary.csv")
    circuits_cohmat_df["TopoName_match"] = circuits_cohmat_df["TopoName"].str.replace(
        "_", "-", regex=False
    )
    circuit_df = pd.merge(
        circuit_df,
        circuits_cohmat_df,
        on="TopoName",
        how="inner",
    )
    cols_to_drop = [col for col in circuit_df.columns if col.endswith("_y")]
    circuit_df = circuit_df.drop(columns=cols_to_drop)
    circuit_df.columns = [col.replace("_x", "") for col in circuit_df.columns]
    print(circuit_df)
    # Get the subset of newtorks
    # MeanCoh == MeanCoh is used to filter out NaN vlaues of MeanCoh
    subset_df = circuit_df.query(
        "Category == 'Holistic' and MiddleNode == True and MeanCoh == MeanCoh"
    ).copy()
    subset_df = (
        subset_df.groupby("MAN", group_keys=False)
        .apply(get_representative_rows)
        .reset_index(drop=True)
    )
    # print(subset_df)
    print(subset_df.columns)
    print(f"Sampled representative motifs: {len(subset_df)}")
    print(subset_df[["MAN-PNStr", "MeanCoh", "NumGroups", "TopoName"]])
    print(
        subset_df[subset_df["SelfActivation"] == "NS"][
            ["MAN_code", "MAN-PNStr", "MeanCoh", "NumGroups", "TopoName"]
        ].sort_values(by=["MAN_code", "MeanCoh"])
    )
    print(
        subset_df[subset_df["SelfActivation"] == "SA"][
            ["MAN_code", "MAN-PNStr", "MeanCoh", "NumGroups", "TopoName"]
        ].sort_values(by=["MAN_code", "MeanCoh"])
    )

    # plot_diverging_coherence_numberline(
    #     df=subset_df[subset_df["SelfActivation"] == "NS"],
    #     motif_col="MAN-PNStr",
    #     val_col="MeanCoh",
    #     base_filepath="coherence_numberline_ns",
    # )

    # for nt in subset_df[subset_df["SelfActivation"] == "NS"]["TopoName"]:
    #     plot_motif_coherence_matrix(
    #         motif_name=nt,
    #         base_dir="./MotifCohResults",
    #     )

    ###################################################################

    # Extract the base file paths directly from the TopoName column
    base_files = [
        input_dir / f"{topo_name}.topo" for topo_name in subset_df["TopoName"]
    ]

    generate_scaled_networks(
        base_files=base_files,
        output_dir=output_dir,
        scales=[10, 30, 50],
        enable_plotting=True,
        max_replicates=50,
    )
