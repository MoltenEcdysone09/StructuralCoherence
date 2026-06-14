import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import re
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator
from itertools import combinations
from io import StringIO

# --- Organism Name Mapping ---
NETWORK_TO_ORGANISM = {
    "196627_v2020_s21_regNetwork_Strong": "Corynebacterium glutamicum",
    "83332_v2018_s15-16_regNetwork": "Mycobacterium tuberculosis",
    "224308_v2022_sSW22_regNetwork": "Bacillus subtilis",
    "511145_v2022_sRDB22_eStrong_regNetwork_Strong": "Escherichia coli",
    "208964_v2020_sRPA20_regNetwork_Strong": "Pseudomonas aeruginosa",
    "100226_v2019_sA22-DBSCR15_eStrong_regNetwork": "Streptomyces coelicolor",
}

# =====================================================================
# 1. STYLE & DESIGN PARAMETERS (Nord Theme)
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
    NORD_COLORS["light_blue"],
]


def set_global_nord_style():
    """Configures Matplotlib global settings for the Nord aesthetic."""
    plt.style.use("default")
    sns.set_context("paper", font_scale=1.6)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Roboto", "Arial", "DejaVu Sans"],
            "text.color": NORD_COLORS["dark"],
            "axes.labelcolor": NORD_COLORS["dark"],
            "axes.titlecolor": NORD_COLORS["dark"],
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": NORD_COLORS["gray"],
            "axes.linewidth": 1.5,
            "xtick.color": NORD_COLORS["dark"],
            "ytick.color": NORD_COLORS["dark"],
            "grid.color": NORD_COLORS["gray"],
            "grid.alpha": 0.3,
            "legend.frameon": False,
        }
    )


set_global_nord_style()


###################################################################################################
# DATA PREPROCESSING
###################################################################################################


def imo_classify_node_coh(row):
    out_val = row["OutAbsCohSum"]
    in_val = row["InAbsCohSum"]

    if pd.isna(out_val) and pd.isna(in_val):
        return np.nan

    out_val = out_val if pd.notna(out_val) else 0
    in_val = in_val if pd.notna(in_val) else 0

    if out_val == 0 and in_val == 0:
        return "Isolated"
    elif out_val == 0 and in_val != 0:
        return "Output"
    elif in_val == 0 and out_val != 0:
        return "Input"
    else:
        return "Middle"


def imo_classify_node(row):
    out_val = row["OutWalkSum"] if pd.notna(row["OutWalkSum"]) else 0
    in_val = row["InWalkSum"] if pd.notna(row["InWalkSum"]) else 0

    if out_val == 0 and in_val != 0:
        return "Output"
    elif in_val == 0 and out_val != 0:
        return "Input"
    else:
        return "Middle"


def preprocess_network_data(absy_dir, gene_info_dir, topos_dir):
    """
    Iterates through coherence matrices and metadata to extract structural module statistics.
    Returns clean DataFrames ready for plotting.
    """
    cohmat_list = sorted(list(absy_dir.glob("*/*_CohMat.parquet")))

    all_intrainter_density = []
    nodelvl_module_compo = []
    all_module_sizes = []

    for cm in cohmat_list:
        print(f"Processing: {cm.stem}")
        cmat = pd.read_parquet(cm)

        net_sum_df = pd.DataFrame(
            {
                "OutAbsCohSum": cmat.abs().sum(axis=1, min_count=1),
                "InAbsCohSum": cmat.abs().sum(axis=0, min_count=1),
                "OutAbsCohMean": cmat.abs().mean(axis=1),
                "InAbsCohMean": cmat.abs().mean(axis=0),
                "OutCohSum": cmat.sum(axis=1, min_count=1),
                "InCohSum": cmat.sum(axis=0, min_count=1),
                "OutCohMean": cmat.mean(axis=1),
                "InCohMean": cmat.mean(axis=0),
            }
        )
        net_sum_df = net_sum_df.reset_index(names="Node")
        net_sum_df["OutCoh_Consistency"] = (
            net_sum_df["OutCohSum"] / net_sum_df["OutAbsCohSum"]
        )
        net_sum_df["InCoh_Consistency"] = (
            net_sum_df["InCohSum"] / net_sum_df["InAbsCohSum"]
        )
        net_sum_df["NodeLevel"] = net_sum_df.apply(imo_classify_node_coh, axis=1)
        net_sum_df = net_sum_df[net_sum_df["NodeLevel"] != "Isolated"]

        # --- Module Annotation ---
        clean_stem = re.sub(r"_regNetwork(?:_Strong)?_CohMat$", "", cm.stem)
        gene_info_path = (
            gene_info_dir
            / f"{clean_stem}_regNet-genes-modules"
            / f"{clean_stem}_geneInformation.tsv"
        )

        gene_info_df = pd.read_csv(gene_info_path, sep="\t", engine="python").rename(
            columns={"Gene_name": "Node"}
        )
        node_info_df = pd.merge(net_sum_df, gene_info_df, how="left", on="Node")

        module_stats = (
            node_info_df.groupby(["NDA_component", "NodeLevel"])
            .size()
            .reset_index(name="Count")
        )
        component_totals = node_info_df.groupby("NDA_component").size()
        module_stats["Component_Total"] = module_stats["NDA_component"].map(
            component_totals
        )
        module_stats["Fraction"] = (
            module_stats["Count"] / module_stats["Component_Total"]
        )
        module_stats = module_stats[
            ["NDA_component", "NodeLevel", "Count", "Component_Total", "Fraction"]
        ]
        module_stats["Network"] = Path(cm).stem
        nodelvl_module_compo.append(module_stats)

        component_sizes_df = component_totals.reset_index(name="Total_Nodes")
        component_sizes_df["Network"] = Path(cm).stem
        all_module_sizes.append(component_sizes_df)

        nodelvl_order = ["Input", "Middle", "Output"]

        topo_file = topos_dir / f"{Path(cm).stem.replace('_CohMat', '')}.topo"
        if topo_file.exists():
            topo_df = pd.read_csv(
                topo_file,
                sep=r"\s+",
                header=None,
                usecols=[0, 1],
                names=["SourceNode", "TargetNode"],
            )

            topo_merged = (
                topo_df.merge(
                    node_info_df[["Node", "NDA_component", "NodeLevel"]],
                    left_on="SourceNode",
                    right_on="Node",
                )
                .rename(
                    columns={
                        "NDA_component": "Source_Module",
                        "NodeLevel": "Source_Level",
                    }
                )
                .drop(columns=["Node"])
            )

            topo_merged = (
                topo_merged.merge(
                    node_info_df[["Node", "NDA_component", "NodeLevel"]],
                    left_on="TargetNode",
                    right_on="Node",
                    suffixes=("", "_Tgt"),
                )
                .rename(
                    columns={
                        "NDA_component": "Target_Module",
                        "NodeLevel": "Target_Level",
                    }
                )
                .drop(columns=["Node"])
            )

            actual_intra = topo_merged[
                topo_merged["Source_Module"] == topo_merged["Target_Module"]
            ]
            actual_inter = topo_merged[
                topo_merged["Source_Module"] != topo_merged["Target_Module"]
            ]

            node_counts_mat = module_stats.pivot(
                index="NDA_component", columns="NodeLevel", values="Count"
            ).fillna(0)
            node_counts_mat = node_counts_mat.reindex(
                columns=nodelvl_order, fill_value=0
            )

            intra_capacity = pd.DataFrame(
                index=nodelvl_order, columns=nodelvl_order, dtype=float
            )
            for src_lvl in nodelvl_order:
                for tgt_lvl in nodelvl_order:
                    intra_capacity.at[src_lvl, tgt_lvl] = (
                        node_counts_mat[src_lvl] * node_counts_mat[tgt_lvl]
                    ).sum()

            total_counts = node_counts_mat.sum()
            total_capacity = pd.DataFrame(
                index=nodelvl_order, columns=nodelvl_order, dtype=float
            )
            for src_lvl in nodelvl_order:
                for tgt_lvl in nodelvl_order:
                    total_capacity.at[src_lvl, tgt_lvl] = (
                        total_counts[src_lvl] * total_counts[tgt_lvl]
                    )

            inter_capacity = total_capacity - intra_capacity

            def get_density_matrix(actual_df, capacity_df):
                counts = (
                    actual_df.groupby(["Source_Level", "Target_Level"])
                    .size()
                    .reset_index(name="Count")
                )
                actual_mat = counts.pivot(
                    index="Source_Level", columns="Target_Level", values="Count"
                )
                actual_mat = actual_mat.reindex(
                    index=nodelvl_order, columns=nodelvl_order
                ).fillna(0)
                density = actual_mat / capacity_df.replace(0, 1)
                return density

            intra_density = get_density_matrix(actual_intra, intra_capacity)
            inter_density = get_density_matrix(actual_inter, inter_capacity)

            intra_long = intra_density.stack().reset_index()
            intra_long.columns = ["Source_Level", "Target_Level", "Density"]
            intra_long["Type"] = "Intra"
            intra_long["Network"] = Path(cm).stem

            inter_long = inter_density.stack().reset_index()
            inter_long.columns = ["Source_Level", "Target_Level", "Density"]
            inter_long["Type"] = "Inter"
            inter_long["Network"] = Path(cm).stem

            network_summary = pd.concat([intra_long, inter_long], ignore_index=True)
            all_intrainter_density.append(network_summary)

    # 1. Clean Intra/Inter Density
    all_intrainter_density = pd.concat(all_intrainter_density, ignore_index=True)
    intrainter_density = all_intrainter_density[
        all_intrainter_density["Density"] > 0
    ].copy()
    intrainter_density["Interaction"] = (
        intrainter_density["Source_Level"].str[0]
        + " - "
        + intrainter_density["Target_Level"].str[0]
    )

    level_order = ["Input", "Middle", "Output"]
    interaction_order = [
        f"{src[0]} - {tgt[0]}" for src in level_order for tgt in level_order
    ]
    existing_interactions = [
        i for i in interaction_order if i in intrainter_density["Interaction"].unique()
    ]

    # 2. Clean Node Composition
    nodelvl_module_compo_df = pd.concat(nodelvl_module_compo, ignore_index=True)

    # 3. Clean Module Sizes
    all_module_sizes_df = pd.concat(all_module_sizes, ignore_index=True)

    return (
        intrainter_density,
        existing_interactions,
        nodelvl_module_compo_df,
        all_module_sizes_df,
    )


############################################################################################
# PLOTTING FUNCTIONS
############################################################################################


def plot_intra_inter_comparison_F(intrainter_density, existing_interactions, plots_dir):
    """
    Grouped Boxplot comparing Intra vs Inter density with Nord Styling.
    Uses Orange (Intra) and Green (Inter) from the user's palette.
    """
    comparison_palette = [NORD_COLORS["yellow"], NORD_COLORS["blue"]]

    plt.figure(figsize=(6, 5))

    ax = sns.boxplot(
        data=intrainter_density,
        x="Interaction",
        y="Density",
        hue="Type",
        order=existing_interactions,
        hue_order=["Intra", "Inter"],
        palette=comparison_palette,
        showfliers=False,
        width=0.5,
    )

    sns.stripplot(
        data=intrainter_density,
        x="Interaction",
        y="Density",
        hue="Type",
        order=existing_interactions,
        hue_order=["Intra", "Inter"],
        palette=comparison_palette,
        dodge=True,
        alpha=0.8,
        edgecolor=NORD_COLORS["dark"],
        linewidth=1.0,
        ax=ax,
        zorder=3,
        legend=False,
    )

    pairs = [
        ((interaction, "Intra"), (interaction, "Inter"))
        for interaction in existing_interactions
    ]

    try:
        annotator = Annotator(
            ax,
            pairs,
            data=intrainter_density,
            x="Interaction",
            y="Density",
            hue="Type",
            order=existing_interactions,
            hue_order=["Intra", "Inter"],
        )
        annotator.configure(
            test="Mann-Whitney",
            text_format="star",
            loc="inside",
            comparisons_correction="Benjamini-Hochberg",
            color=NORD_COLORS["dark"],
            verbose=False,
        )
        annotator.apply_and_annotate()
    except Exception as e:
        print(f"Stats annotation failed: {e}")

    plt.title("Structural Density: Intra- vs Inter-Module Connectivity", pad=15)
    plt.tick_params(axis="x")
    plt.xlabel("Interaction Type")
    plt.ylabel("Connection Density (Normalized)")
    plt.xticks(ha="center")

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top * 1.05)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])

    plt.legend(
        title="Connection Type",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
    )

    plt.tight_layout()
    save_path = plots_dir / "AllNets_InterIntraComparison.png"
    save_path_svg = save_path.with_suffix(".svg")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    plt.close()


def plot_separate_intra_inter_distributions_F(
    intrainter_density, existing_interactions, plots_dir
):
    """
    Generates two separate plots:
    1. Only Intra-Module Density (colored by Interaction)
    2. Only Inter-Module Density (colored by Interaction)
    """

    def _plot_subset(density_type, filename_suffix):
        subset_df = intrainter_density[
            intrainter_density["Type"] == density_type
        ].copy()

        if subset_df.empty:
            print(f"No data for {density_type}, skipping plot.")
            return

        plt.figure(figsize=(5, 5))

        ax = sns.boxplot(
            data=subset_df,
            x="Interaction",
            y="Density",
            order=existing_interactions,
            palette=NORD_PALETTE,
            showfliers=False,
            width=0.4,
        )

        sns.stripplot(
            data=subset_df,
            x="Interaction",
            y="Density",
            order=existing_interactions,
            hue="Interaction",
            hue_order=existing_interactions,
            palette=NORD_PALETTE,
            dodge=False,
            edgecolor=NORD_COLORS["dark"],
            linewidth=1.0,
            size=8,
            alpha=0.7,
            jitter=True,
            ax=ax,
            zorder=3,
            legend=False,
        )

        pairs = list(combinations(existing_interactions, 2))
        if pairs:
            try:
                annotator = Annotator(
                    ax,
                    pairs,
                    data=subset_df,
                    x="Interaction",
                    y="Density",
                    order=existing_interactions,
                )
                annotator.configure(
                    test="Mann-Whitney",
                    text_format="star",
                    loc="inside",
                    comparisons_correction="Benjamini-Hochberg",
                    color=NORD_COLORS["dark"],
                    line_width=1.5,
                    verbose=False,
                )
                annotator.apply_and_annotate()
            except Exception as e:
                print(f"Stats failed for {density_type}: {e}")

        plt.title(f"Structural Density: {density_type}-Module Only", pad=10)
        plt.xlabel("Interaction Type")
        plt.ylabel("Density (Normalized)")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(NORD_COLORS["dark"])

        plt.tight_layout()
        save_path = plots_dir / f"AllNets_{filename_suffix}_Distribution.png"
        save_path_svg = save_path.with_suffix(".svg")
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
        plt.savefig(save_path_svg, dpi=300, transparent=True)
        plt.close()
        print(f"Saved plot: {save_path}")

    _plot_subset("Intra", "IntraOnly")
    _plot_subset("Inter", "InterOnly")


def plot_all_nets_node_level_fraction_F(nodelvl_module_compo, plots_dir):
    """
    Violin + Stripplot of Node Level Fractions across ALL networks.
    Uses Nord Styling (Red/Blue/Green).
    """
    level_order = ["Input", "Middle", "Output"]
    nord_color_list = [NORD_PALETTE[0], NORD_PALETTE[1], NORD_PALETTE[2]]

    plt.figure(figsize=(6, 5))

    ax = sns.violinplot(
        data=nodelvl_module_compo,
        x="NodeLevel",
        y="Fraction",
        order=level_order,
        palette=nord_color_list,
        density_norm="width",
        cut=0,
    )

    for poly in ax.collections:
        poly.set_alpha(0.6)

    sns.stripplot(
        data=nodelvl_module_compo,
        x="NodeLevel",
        y="Fraction",
        order=level_order,
        hue="NodeLevel",
        hue_order=level_order,
        palette=nord_color_list,
        color="black",
        alpha=0.6,
        s=8,
        jitter=True,
        edgecolor=NORD_COLORS["dark"],
        linewidth=0.1,
        dodge=False,
        zorder=1,
    )

    try:
        annotator = Annotator(
            ax,
            pairs=list(combinations(level_order, 2)),
            data=nodelvl_module_compo,
            x="NodeLevel",
            y="Fraction",
            order=level_order,
        )
        annotator.configure(
            test="Mann-Whitney",
            text_format="star",
            loc="inside",
            comparisons_correction="Benjamini-Hochberg",
            color=NORD_COLORS["dark"],
            line_offset_to_group=0.1,
            line_offset=0.3,
            verbose=False,
        )
        annotator.apply_and_annotate()
    except Exception as e:
        print(f"Stats annotation failed: {e}")

    plt.title("All Nets Distribution of Node Levels across NDA Components", pad=10)
    plt.ylabel("Fraction of Nodes in Level")
    plt.xlabel("Network Layer")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])

    if ax.get_legend():
        ax.get_legend().remove()

    plt.tight_layout()
    save_path = plots_dir / "AllNets_Frac_NodeLvl_Module.png"
    save_path_svg = save_path.with_suffix(".svg")
    plt.savefig(save_path, dpi=300, transparent=True)
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_module_sizes_by_network_F(module_sizes_df, meta_df, plots_dir):
    """
    Plots module size distributions with annotations for Mean and Legend with Coverage.
    """
    sorted_networks = meta_df.sort_values("Completeness", ascending=True)[
        "Organism"
    ].tolist()
    n_nets = len(NETWORK_TO_ORGANISM)

    nord_colors = ["#5E81AC", "#A3BE8C", "#EBCB8B", "#D08770", "#BF616A", "#B48EAD"]
    my_cmap = LinearSegmentedColormap.from_list("NordSeq", nord_colors, N=n_nets)
    my_palette = [my_cmap(i) for i in np.linspace(0, 1, n_nets)]
    palette_dict = dict(zip(NETWORK_TO_ORGANISM.values(), my_palette))

    missing_keys = set(module_sizes_df["Network"].unique()) - set(palette_dict.keys())
    if missing_keys:
        print("The following Networks are in the dataframe but not the palette:")
        print(missing_keys)

    plt.figure(figsize=(10, 5))

    ax = sns.violinplot(
        data=module_sizes_df,
        x="Network",
        y="Total_Nodes",
        hue="Network",
        palette=palette_dict,
        density_norm="width",
        linewidth=1.5,
        dodge=False,
        cut=0,
    )

    for poly in ax.collections:
        poly.set_alpha(0.5)

    sns.stripplot(
        data=module_sizes_df,
        x="Network",
        y="Total_Nodes",
        hue="Network",
        order=sorted_networks,
        palette=palette_dict,
        size=8,
        alpha=0.7,
        jitter=True,
        zorder=1,
        ax=ax,
        legend=False,
        linewidth=0.5,
    )

    plt.ylim(0, 35)

    for i, net in enumerate(sorted_networks):
        subset = module_sizes_df[module_sizes_df["Network"] == net]
        if subset.empty:
            continue
        mean_val = subset["Total_Nodes"].mean()
        max_val = subset["Total_Nodes"].max()

        ax.text(
            x=i,
            y=max_val + (max_val * 0.20),
            s=f"μ={mean_val:.1f}",
            ha="center",
            va="bottom",
            color="#2E3440",
        )

    plt.title("Module Size Distribution per Network", pad=15)
    plt.ylabel("Module Size (Total Nodes)")
    plt.xlabel("Networks")

    ax.set_xticklabels([])
    ax.tick_params(axis="x", which="both", length=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#2E3440")

    legend_handles = []
    for net in sorted_networks:
        color = palette_dict.get(net, "#000000")
        try:
            cov = meta_df.loc[meta_df["Organism"] == net, "Completeness"].values[0]
            label_text = f"{net} ({cov}%)"
        except IndexError:
            label_text = net

        legend_handles.append(mpatches.Patch(color=color, label=label_text))

    plt.legend(
        handles=legend_handles,
        title="Organism (Network Completeness)",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
    )

    plt.tight_layout()
    save_path = plots_dir / "AllNets_ModuleSize_PerNetwork_Violin.png"
    save_path_svg = save_path.with_suffix(".svg")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    print(f"Saved plot: {save_path}")
    plt.close()


def plot_global_nodelevel_density_boxplot_F(wt_base_dir, topos_dir, plot_dir):
    """
    Calculates the connection density (Actual Topo Edges / Possible Edges)
    between Input, Middle, and Output levels for all organisms.
    Plots them as a single grouped boxplot ordered by median density,
    matching the exact values and significance from the original analysis.
    """
    from statannotations.Annotator import Annotator

    set_global_nord_style()

    print("\nProcessing Global NodeLevel Connectivity Boxplots (Topo-based)...")

    level_join_densities = []

    # 1. Extract and compute densities for all organisms
    for grn, org_name in NETWORK_TO_ORGANISM.items():
        wt_path = wt_base_dir / grn / f"{grn}_CohMat.parquet"
        topo_file = topos_dir / f"{grn}.topo"

        if not wt_path.exists() or not topo_file.exists():
            continue

        # --- A. Load CohMat to Classify Nodes ---
        cmat = pd.read_parquet(wt_path)
        num_mat = cmat.select_dtypes(include="number")

        row_sums = np.nanmean(num_mat.abs(), axis=1)
        col_sums = np.nanmean(num_mat.abs(), axis=0)

        # Handle MultiIndex safely
        node_names = (
            num_mat.index.get_level_values(1)
            if isinstance(num_mat.index, pd.MultiIndex)
            else num_mat.index
        )
        node_names = [n[1] if isinstance(n, tuple) else str(n) for n in node_names]

        inout_coh = pd.DataFrame(
            {
                "Node": node_names,
                "MeanAbsOutCoh": row_sums,
                "MeanAbsInCoh": col_sums,
            }
        )

        def classify_node(row):
            out_val = row["MeanAbsOutCoh"] if pd.notna(row["MeanAbsOutCoh"]) else 0
            in_val = row["MeanAbsInCoh"] if pd.notna(row["MeanAbsInCoh"]) else 0
            if out_val == 0 and in_val != 0:
                return "Output"
            elif in_val == 0 and out_val != 0:
                return "Input"
            else:
                return "Middle"

        inout_coh["NodeLevel"] = inout_coh.apply(classify_node, axis=1)
        node_level_dict = inout_coh.set_index("Node")["NodeLevel"].to_dict()

        # --- B. Load Topo File to Count Actual Physical Edges ---
        topo_df = pd.read_csv(
            topo_file,
            sep=r"\s+",
            usecols=[0, 1],
            names=["Source", "Target"],
            header=None,
        )
        topo_df["Source"] = topo_df["Source"].astype(str)
        topo_df["Target"] = topo_df["Target"].astype(str)

        topo_df["SourceLevel"] = topo_df["Source"].map(node_level_dict)
        topo_df["TargetLevel"] = topo_df["Target"].map(node_level_dict)
        topo_df = topo_df.dropna(subset=["SourceLevel", "TargetLevel"])

        node_order = ["Input", "Middle", "Output"]

        # Calculate Possible Edges (Nodes in A * Nodes in B)
        all_nodes = pd.concat(
            [
                topo_df[["Source", "SourceLevel"]].rename(
                    columns={"Source": "Node", "SourceLevel": "Level"}
                ),
                topo_df[["Target", "TargetLevel"]].rename(
                    columns={"Target": "Node", "TargetLevel": "Level"}
                ),
            ]
        )
        node_counts = all_nodes.drop_duplicates().groupby("Level")["Node"].count()

        possible_edges = pd.DataFrame(index=node_order, columns=node_order, dtype=float)
        for A in node_order:
            for B in node_order:
                cA = node_counts.get(A, 0)
                cB = node_counts.get(B, 0)
                possible_edges.loc[A, B] = cA * cB

        # Calculate Actual Edges
        heatmap_counts = pd.crosstab(
            topo_df["SourceLevel"],
            topo_df["TargetLevel"],
        ).reindex(index=node_order, columns=node_order, fill_value=0)

        # Density = Actual / Possible
        heatmap_density = heatmap_counts / possible_edges.replace(0, np.nan)

        # Flatten into long form
        long_density = heatmap_density.stack().reset_index()
        long_density.columns = ["Source_Level", "Target_Level", "Density"]
        long_density["Network"] = org_name
        long_density["Connection Density"] = (
            long_density["Source_Level"].str[0]
            + " - "
            + long_density["Target_Level"].str[0]
        )

        level_join_densities.append(long_density)

    if not level_join_densities:
        print("No density data processed.")
        return

    df = pd.concat(level_join_densities, ignore_index=True)

    # 2. Compute median stats to order the plot and filter out zero-median connections
    density_stats = (
        df.groupby("Connection Density")["Density"].agg(["median", "std"]).reset_index()
    )
    density_stats = density_stats[density_stats["median"] > 0]

    filtered_df = df[
        df["Connection Density"].isin(density_stats["Connection Density"])
    ].copy()
    density_order = density_stats.sort_values("median", ascending=False)[
        "Connection Density"
    ].tolist()

    # 3. Plotting Setup
    fig, ax = plt.subplots(figsize=(5, 5))

    sns.boxplot(
        data=filtered_df,
        x="Connection Density",
        y="Density",
        hue="Connection Density",
        order=density_order,
        palette=NORD_PALETTE,
        linewidth=1.5,
        width=0.4,
        ax=ax,
        showfliers=False,
    )

    sns.stripplot(
        data=filtered_df,
        x="Connection Density",
        y="Density",
        order=density_order,
        hue="Connection Density",
        palette=NORD_PALETTE,
        ax=ax,
        linewidth=1.3,
        size=9,
        jitter=True,
        alpha=0.8,
        legend=False,
    )

    # 4. Stats Annotation
    levels = sorted(filtered_df["Connection Density"].unique())
    levels = [l for l in density_order if l in levels]

    pairs = [
        (levels[i], levels[j])
        for i in range(len(levels))
        for j in range(i + 1, len(levels))
    ]

    if pairs:
        try:
            annotator = Annotator(
                ax=ax,
                pairs=pairs,
                data=filtered_df,
                x="Connection Density",
                y="Density",
                order=density_order,
            )
            annotator.configure(
                test="Mann-Whitney",
                text_format="star",
                loc="inside",
                verbose=False,
                color=NORD_COLORS["dark"],
                line_width=1.5,
                hide_non_significant=True,
            )
            annotator.apply_and_annotate()
        except Exception as e:
            print(f"Stats annotation failed: {e}")

    # 5. Formatting & Layout
    ax.tick_params(axis="x")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top * 1.05)

    plt.xticks(rotation=0, ha="center")
    plt.title("Inter/Intra-Level Structural Connection Density", pad=15, fontsize=16)
    plt.ylabel("Density")
    plt.xlabel("Connection Type")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])

    plt.tight_layout()

    # 6. Save
    filename = "Global_NodeLevel_Topo_Density_Boxplot"
    fig.savefig(
        plot_dir / f"{filename}.png",
        format="png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    fig.savefig(
        plot_dir / f"{filename}.svg",
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)


def plot_average_node_level_distribution_F(
    nodelvl_module_compo_df, plots_dir, nord_colors
):
    """
    Calculates the average percentage of Input, Middle, and Output nodes
    across all networks and plots a bar chart with standard deviation error bars.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Calculate per-network percentages from the component dataframe
    net_level_counts = (
        nodelvl_module_compo_df.groupby(["Network", "NodeLevel"])["Count"]
        .sum()
        .reset_index()
    )
    net_totals = net_level_counts.groupby("Network")["Count"].transform("sum")
    net_level_counts["Percentage"] = (net_level_counts["Count"] / net_totals) * 100

    # 2. Calculate mean and std across networks
    mean_vals = (
        net_level_counts.groupby("NodeLevel")["Percentage"]
        .mean()
        .reindex(["Input", "Middle", "Output"])
    )
    std_vals = (
        net_level_counts.groupby("NodeLevel")["Percentage"]
        .std()
        .reindex(["Input", "Middle", "Output"])
    )

    # 3. Map colors
    nord_map = {
        "Input": nord_colors["red"],
        "Middle": nord_colors["blue"],
        "Output": nord_colors["green"],
    }
    bar_colors = [nord_map.get(idx, nord_colors["gray"]) for idx in mean_vals.index]

    # 4. Plot Setup
    plt.figure(figsize=(5, 5.5))
    bars = plt.bar(
        mean_vals.index,
        mean_vals.values,
        yerr=std_vals.values,
        capsize=5,
        color=bar_colors,
        edgecolor=nord_colors["dark"],
        linewidth=1.5,
        width=0.4,
        error_kw=dict(ecolor=nord_colors["dark"], lw=1.5, capthick=1.5),
    )

    # 5. Styling
    plt.ylabel("Percentage of Nodes (%)")
    plt.xlabel("Node Level")
    plt.title("Average Node Level Distribution Across Networks", pad=15)

    # Adjust Y-limit to fit annotations (Adding 1.3x headroom)
    plt.ylim(0, max(mean_vals.values + std_vals.fillna(0).values) * 1.1)

    # Annotate bars
    for bar, mean, std in zip(bars, mean_vals.values, std_vals.values):
        std_val = std if pd.notna(std) else 0.0
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std_val + 1,  # Place just above the error bar
            f"{mean:.1f} ± {std_val:.1f}%",
            ha="center",
            va="bottom",
            color=nord_colors["dark"],
            fontsize=13,
        )

    # Spines
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(nord_colors["dark"])

    plt.tight_layout()

    save_path = plots_dir / "NodeLevelDist.png"
    save_path_svg = save_path.with_suffix(".svg")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(save_path_svg, dpi=300, transparent=True, bbox_inches="tight")
    plt.close()

    print(f"Saved Average Node Level Distribution Plot to: {save_path}")


###################################################################################################
# EXECUTION BLOCK
###################################################################################################

if __name__ == "__main__":
    # 1. Define Execution Directories
    absy_dir = Path("./AbasyCohResults_Targeted/")
    gene_info_dir = Path("./AbasyNets/")
    topos_dir = Path("./AbasyTOPOS_Targeted/")

    plots_dir = Path("./GRN_Plots/Fig6/")
    plots_dir.mkdir(exist_ok=True, parents=True)

    # 2. Run Central Preprocessing
    (
        intrainter_density,
        existing_interactions,
        nodelvl_module_compo_df,
        all_module_sizes_df,
    ) = preprocess_network_data(absy_dir, gene_info_dir, topos_dir)

    # 3. Call Modular Plotting Functions
    plot_intra_inter_comparison_F(intrainter_density, existing_interactions, plots_dir)
    plot_separate_intra_inter_distributions_F(
        intrainter_density, existing_interactions, plots_dir
    )
    plot_all_nets_node_level_fraction_F(nodelvl_module_compo_df, plots_dir)

    # 4. Process Module Size Metadata & Plot
    data_string = """Regulatory_Network,Organism,Version,Genomic_Coverage,Completeness
        100226_v2019_sA22-DBSCR15_eStrong_regNetwork,Streptomyces coelicolor,2019,5.2,2.3
        224308_v2022_sSW22_regNetwork,Bacillus subtilis,2022,58.2,49.4
        511145_v2022_sRDB22_eStrong_regNetwork_Strong,Escherichia coli,2022,51.3,56.0
        196627_v2020_s21_regNetwork_Strong,Corynebacterium glutamicum,2020,71.7,42.6
        83332_v2018_s15-16_regNetwork,Mycobacterium tuberculosis,2018,62.1,67.3
        208964_v2020_sRPA20_regNetwork_Strong,Pseudomonas aeruginosa,2020,18.4,13.7
        """
    meta_df = pd.read_csv(StringIO(data_string))

    # Clean the trailing "_CohMat" off the stems so they match the dictionary keys perfectly
    all_module_sizes_df["Network"] = all_module_sizes_df["Network"].str.replace(
        "_CohMat", "", regex=False
    )

    # Map the full IDs directly to the Organism names to hook into the Palette successfully
    all_module_sizes_df["Network"] = all_module_sizes_df["Network"].replace(
        NETWORK_TO_ORGANISM
    )

    # Filter Outliers (e.g., giant basal machinery modules to keep violins legible)
    threshold = all_module_sizes_df["Total_Nodes"].quantile(0.95)
    filtered_df = all_module_sizes_df[
        all_module_sizes_df["Total_Nodes"] <= threshold
    ].copy()

    plot_module_sizes_by_network_F(filtered_df, meta_df, plots_dir)

    plot_global_nodelevel_density_boxplot_F(
        wt_base_dir=absy_dir,
        topos_dir=topos_dir,
        plot_dir=plots_dir,
    )

    plot_average_node_level_distribution_F(
        nodelvl_module_compo_df=nodelvl_module_compo_df,
        plots_dir=plots_dir,
        nord_colors=NORD_COLORS,
    )
