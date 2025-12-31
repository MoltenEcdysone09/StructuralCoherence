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


###################################################################################################
### Plotting Settings
###################################################################################################

# --- GLOBAL CONSTANTS ---
# Nord Palette Definition
NORD_COLORS = {
    "dark": "#2e3440",  # nord0: Darkest black/grey
    "gray": "#3b4252",  # nord1: Dark gray
    "red": "#bf616a",  # nord11
    "blue": "#5e81ac",  # nord10
    "green": "#a3be8c",  # nord14
    "yellow": "#ebcb8b",  # nord13
    "purple": "#b48ead",  # nord15
}

# Define the categorical palette list for use in all plots
NORD_PALETTE = [
    NORD_COLORS["red"],
    NORD_COLORS["blue"],
    NORD_COLORS["green"],
    NORD_COLORS["yellow"],
    NORD_COLORS["purple"],
]


def set_global_nord_style():
    """
    Configures Matplotlib global settings for the entire script run.
    """
    # Reset to default first to clear any conflicting states
    plt.style.use("default")

    # Update global parameters
    plt.rcParams.update(
        {
            # 1. Fonts
            "font.family": "sans-serif",
            "font.sans-serif": ["Roboto", "Arial", "DejaVu Sans"],
            "font.size": 16,
            # "font.weight": "bold",
            # 2. Text Colors
            "text.color": NORD_COLORS["dark"],
            "axes.labelcolor": NORD_COLORS["dark"],
            "axes.titlecolor": NORD_COLORS["dark"],
            # "axes.labelweight": "bold",
            # "axes.titleweight": "bold",
            # 3. Axes & Spines (The "Closed Box" Look)
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": NORD_COLORS["dark"],
            "axes.linewidth": 2.0,  # Thick Spines
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,  # No Grid
            # 4. Ticks
            "xtick.color": NORD_COLORS["dark"],
            "ytick.color": NORD_COLORS["dark"],
            "xtick.major.width": 2.0,
            "ytick.major.width": 2.0,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            # 5. Legend Defaults
            "legend.edgecolor": NORD_COLORS["dark"],
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": False,
            # 6. Global Line/Patch widths (helps with consistency)
            "lines.linewidth": 2.0,
            "patch.linewidth": 2.0,
        }
    )


# --- APPLY THE SETTINGS ---
set_global_nord_style()

###################################################################################################
###################################################################################################


def get_priority(filename):
    # Priority 1: Must contain BOTH "eStrong" AND "Strong"
    if "_eStrong" in filename and "_Strong" in filename:
        return 1
    # Priority 2: Contains "Strong" (but not "eStrong", as it was caught above)
    if "_Strong" in filename:
        return 2
    # Priority 3: Contains neither
    return 3


# Function to classify nodes
def imo_classify_node(row):
    # Treat NaN as 0 for this logic
    out_val = row["OutWalkSum"] if pd.notna(row["OutWalkSum"]) else 0
    in_val = row["InWalkSum"] if pd.notna(row["InWalkSum"]) else 0

    if out_val == 0 and in_val != 0:
        return "Output"
    elif in_val == 0 and out_val != 0:
        return "Input"
    else:
        # This catches cases where both are 0/NaN
        # or both are non-zero
        return "Middle"


# Abasy network data
absy_dir = Path("./AbasyCohResults/")
gene_info_dir = Path("./AbasyNets/")
topos_dir = Path("./AbasyTOPOS/")
# cohmat_list = list(absy_dir.glob("*/*_Strong_CohMat.parquet"))
cohmat_list = sorted(list(absy_dir.glob("*/*_CohMat.parquet")))

# Getting the unique netowrk code
net_codes = set([str(nc.stem).split("_")[0] for nc in cohmat_list])
print(set(net_codes))
print(len(net_codes))

# nunduplicate cohmat list
nondup_cohmat_list = []

for nc in net_codes:
    print(nc)
    # Subet the networks which have th same names
    sub_list = sorted([n for n in cohmat_list if n.stem.startswith(nc)])
    if not sub_list:
        print(f"No networks found for {nc}")
        continue
    sel_network = min(sub_list, key=lambda path: get_priority(path.stem))
    if "100266" not in sel_network.name:
        nondup_cohmat_list.append(sel_network)
        print(f"Selected Network: {sel_network.name}")
    else:
        print("Ignoring 100266")

gene_info_dir = Path("./AbasyNets/")
# PLto direcotry
plots_dir = absy_dir / "ModdulePlots"
plots_dir.mkdir(exist_ok=True)

all_intrainter_density = []
nodelvl_module_compo = []
permodule_inter_denisty = []
all_module_sizes = []

for cm in sorted(nondup_cohmat_list):
    print(cm)
    # Read coherence matrix
    cmat = pd.read_parquet(cm)
    # print(cmat.columns)
    # print(cmat.columns.get_level_values("TargetNode"))
    wmat_order = cmat.columns.get_level_values("TargetNode")
    # Drop the group level
    cmat = cmat.droplevel("Group", axis=0).droplevel("Group", axis=1)
    # read the walk frac matrix
    wmat = pd.read_parquet(str(cm).replace("_CohMat", "_WalkMats"))["NumWalks"]
    wmat = wmat.groupby(level="SourceNode").sum()
    wmat = wmat.reindex(wmat_order)[wmat_order]
    # Sum info datframe
    net_sum_df = pd.DataFrame(
        {
            # "Node": list(wmat_order),
            "OutWalkSum": wmat.sum(axis=1),
            "InWalkSum": wmat.sum(axis=0),
            "OutWalkMean": wmat.mean(axis=1),
            "InWalkMean": wmat.mean(axis=0),
            "OutAbsCohSum": cmat.abs().sum(axis=1),
            "InAbsCohSum": cmat.abs().sum(axis=0),
            "OutAbsCohMean": cmat.abs().mean(axis=1),
            "InAbsCohMean": cmat.abs().mean(axis=0),
            "OutCohSum": cmat.sum(axis=1),
            "InCohSum": cmat.sum(axis=0),
            "OutCohMean": cmat.mean(axis=1),
            "InCohMean": cmat.mean(axis=0),
        }
    )
    # Reset index to name the index as node
    net_sum_df = net_sum_df.reset_index(names="Node")
    # Get the ratio of the OutCohSum to OutAbsCohSum
    net_sum_df["OutCoh_Consistency"] = (
        net_sum_df["OutCohSum"] / net_sum_df["OutAbsCohSum"]
    )
    net_sum_df["InCoh_Consistency"] = net_sum_df["InCohSum"] / net_sum_df["InAbsCohSum"]
    # Annotating the nodes wiht thier IMO
    net_sum_df["NodeLevel"] = net_sum_df.apply(imo_classify_node, axis=1)
    ##############################################
    #### Node Level Consistency Dist
    ##############################################
    # 1. Setup Order & Colors
    level_order = ["Input", "Middle", "Output"]

    # Strict Color List (Red -> Blue -> Green)
    nord_color_list = [
        NORD_PALETTE[0],  # Red
        NORD_PALETTE[1],  # Blue
        NORD_PALETTE[2],  # Green
    ]

    # 2. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

    # --- Plot 1: Out-Consistency ---
    sns.kdeplot(
        data=net_sum_df,
        x="OutCoh_Consistency",
        hue="NodeLevel",
        hue_order=level_order,  # Enforce order
        palette=nord_color_list,  # Enforce Nord colors
        ax=axes[0],
        common_norm=False,
        fill=True,
        alpha=0.2,  # Lighter fill for clarity
        linewidth=2.5,  # Thicker lines matching styling
        legend=False,
    )
    axes[0].set_title("Out-Consistency Distribution", fontweight="bold")
    axes[0].set_xlabel("Out-Consistency", fontweight="bold")
    axes[0].set_ylabel("Density", fontweight="bold")

    # --- Plot 2: In-Consistency ---
    sns.kdeplot(
        data=net_sum_df,
        x="InCoh_Consistency",
        hue="NodeLevel",
        hue_order=level_order,
        palette=nord_color_list,
        ax=axes[1],
        common_norm=False,
        fill=True,
        alpha=0.2,
        linewidth=2.5,
    )
    axes[1].set_title("In-Consistency Distribution", fontweight="bold")
    axes[1].set_xlabel("In-Consistency", fontweight="bold")

    # --- Styling ---
    # Apply box spines to both axes
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(NORD_COLORS["dark"])
            spine.set_linewidth(2.0)

    # Move legend outside
    sns.move_legend(
        axes[1],
        "upper left",
        bbox_to_anchor=(1.02, 1),
        title="Network Layer",
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(
        plots_dir / f"{Path(cm).stem}_InOutConsistency.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        plots_dir / f"{Path(cm).stem}_InOutConsistency.svg",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    # plt.show()
    plt.close()

    ##############################################
    ## Module Annotation
    ##############################################
    clean_stem = re.sub(
        r"_regNetwork(?:_Strong)?_CohMat$",  # matches both Strong and non-Strong
        "",
        cm.stem,
    )

    gene_info_path = (
        gene_info_dir
        / f"{clean_stem}_regNet-genes-modules"
        / f"{clean_stem}_geneInformation.tsv"
    )

    gene_info_df = pd.read_csv(gene_info_path, sep="\t", engine="python").rename(
        columns={"Gene_name": "Node"}
    )
    # Merge the node info witht he inout_coh dataframe
    node_info_df = pd.merge(net_sum_df, gene_info_df, how="left", on="Node")
    # get the fraction of genes in differnt levels info
    module_stats = (
        node_info_df.groupby(["NDA_component", "NodeLevel"])
        .size()
        .reset_index(name="Count")
    )
    component_totals = node_info_df.groupby("NDA_component").size()
    module_stats["Component_Total"] = module_stats["NDA_component"].map(
        component_totals
    )
    module_stats["Fraction"] = module_stats["Count"] / module_stats["Component_Total"]
    module_stats = module_stats[
        ["NDA_component", "NodeLevel", "Count", "Component_Total", "Fraction"]
    ]
    print(module_stats.head())
    module_stats["Network"] = Path(cm).stem
    nodelvl_module_compo.append(module_stats)
    # This is simply the calculated totals reset to a dataframe
    component_sizes_df = component_totals.reset_index(name="Total_Nodes")
    component_sizes_df["Network"] = Path(cm).stem
    all_module_sizes.append(component_sizes_df)
    # Settingthe node order
    nodelvl_order = ["Input", "Middle", "Output"]
    ##############################################
    # Plotting the fractoin of node levels
    ##############################################
    # 1. Setup Order & Colors
    level_order = ["Input", "Middle", "Output"]

    # Strict Color List
    nord_color_list = [
        NORD_PALETTE[0],  # Red
        NORD_PALETTE[1],  # Blue
        NORD_PALETTE[2],  # Green
    ]

    # 2. Setup Plot
    plt.figure(figsize=(6, 6))

    # --- Layer 1: Boxplot (Background) ---
    ax = sns.boxplot(
        data=module_stats,
        x="NodeLevel",
        y="Fraction",
        order=level_order,
        palette=nord_color_list,
        showfliers=False,
        width=0.5,
    )

    # --- Layer 2: Stripplot (Points) ---
    sns.stripplot(
        data=module_stats,
        x="NodeLevel",
        y="Fraction",
        order=level_order,
        hue="NodeLevel",
        hue_order=level_order,
        palette=nord_color_list,
        jitter=True,
        alpha=0.8,
        edgecolor=NORD_COLORS["dark"],
        linewidth=1.0,
        dodge=False,
        size=6,
        zorder=3,
    )

    annotator = Annotator(
        ax,
        pairs=list(combinations(level_order, 2)),
        data=module_stats,
        x="NodeLevel",
        y="Fraction",
        order=level_order,
    )
    annotator.configure(
        test="Mann-Whitney",
        text_format="star",
        loc="inside",
        comparisons_correction="Benjamini-Hochberg",
        color=NORD_COLORS["dark"],  # Nord dark text
        line_width=1.5,
    )
    annotator.apply_and_annotate()

    # Avoid clashing of the annaotations with the boundary
    ax.set_ylim(-0.1, 1.6)

    plt.title("Distribution of Node Levels across Modules")
    plt.ylabel("Fraction of Nodes in Level")
    plt.xlabel("Network Layer")

    # Box Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])
        spine.set_linewidth(2.0)

    # Remove legend
    if ax.get_legend():
        ax.get_legend().remove()

    plt.tight_layout()

    save_path = plots_dir / f"{Path(cm).stem}_Frac_NodeLvl_Module.png"
    save_path_svg = save_path.with_suffix(".svg")
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    # plt.show()
    plt.close()
    ##############################################
    ##############################################

    ##############################################
    # Plotting and seeing the componenet sizes
    ##############################################

    upper_bound = component_sizes_df["Total_Nodes"].quantile(0.95)
    sns.kdeplot(
        data=component_sizes_df[component_sizes_df["Total_Nodes"] <= upper_bound],
        x="Total_Nodes",
        fill=True,
    )
    plt.savefig(plots_dir / f"{Path(cm).stem}_ModuleSize_Hist.png", dpi=300)
    plt.savefig(
        plots_dir / f"{Path(cm).stem}_ModuleSize_Hist.svg", dpi=300, transparent=True
    )
    plt.close()

    ##############################################
    ##############################################

    topo_file = topos_dir / f"{Path(cm).stem.replace('_CohMat', '')}.topo"

    if topo_file.exists():
        topo_df = pd.read_csv(
            topo_file,
            sep=r"\s+",
            header=None,
            usecols=[0, 1],
            names=["SourceNode", "TargetNode"],
        )

        # Merge Source Metadata
        topo_merged = (
            topo_df.merge(
                node_info_df[["Node", "NDA_component", "NodeLevel"]],
                left_on="SourceNode",
                right_on="Node",
            )
            .rename(
                columns={"NDA_component": "Source_Module", "NodeLevel": "Source_Level"}
            )
            .drop(columns=["Node"])
        )

        # Merge Target Metadata
        topo_merged = (
            topo_merged.merge(
                node_info_df[["Node", "NDA_component", "NodeLevel"]],
                left_on="TargetNode",
                right_on="Node",
                suffixes=("", "_Tgt"),
            )
            .rename(
                columns={"NDA_component": "Target_Module", "NodeLevel": "Target_Level"}
            )
            .drop(columns=["Node"])
        )

        # Split Actual Edges into Intra and Inter
        actual_intra = topo_merged[
            topo_merged["Source_Module"] == topo_merged["Target_Module"]
        ]
        actual_inter = topo_merged[
            topo_merged["Source_Module"] != topo_merged["Target_Module"]
        ]

        # We need the count of nodes in every module/level to calculate capacity
        # Pivot module_stats: Index=Module, Cols=Level, Values=Count
        # Ensure we have 0s for missing levels
        node_counts_mat = module_stats.pivot(
            index="NDA_component", columns="NodeLevel", values="Count"
        ).fillna(0)

        # Ensure strict column order [Input, Middle, Output]
        nodelvl_order = ["Input", "Middle", "Output"]
        node_counts_mat = node_counts_mat.reindex(columns=nodelvl_order, fill_value=0)

        # 1. Calculate INTRA Capacity: Sum of (N_A * N_B) for each module
        # Result is a 3x3 matrix where cell (A, B) = sum(Node_A_m * Node_B_m)
        intra_capacity = pd.DataFrame(
            index=nodelvl_order, columns=nodelvl_order, dtype=float
        )
        for src_lvl in nodelvl_order:
            for tgt_lvl in nodelvl_order:
                # Dot product of the two columns across all modules
                intra_capacity.at[src_lvl, tgt_lvl] = (
                    node_counts_mat[src_lvl] * node_counts_mat[tgt_lvl]
                ).sum()

        # 2. Calculate TOTAL Capacity: (Total N_A) * (Total N_B)
        total_counts = node_counts_mat.sum()  # Series: Input=TotalInput, etc.
        total_capacity = pd.DataFrame(
            index=nodelvl_order, columns=nodelvl_order, dtype=float
        )
        for src_lvl in nodelvl_order:
            for tgt_lvl in nodelvl_order:
                total_capacity.at[src_lvl, tgt_lvl] = (
                    total_counts[src_lvl] * total_counts[tgt_lvl]
                )

        # 3. Calculate INTER Capacity: Total - Intra
        inter_capacity = total_capacity - intra_capacity

        # --- C. Calculate Densities ---

        def get_density_matrix(actual_df, capacity_df):
            # Count actual edges
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

            # Divide by capacity (avoid division by zero)
            density = actual_mat / capacity_df.replace(
                0, 1
            )  # Replace 0 with 1 to avoid NaN (result will be 0/1=0)
            return density

        intra_density = get_density_matrix(actual_intra, intra_capacity)
        inter_density = get_density_matrix(actual_inter, inter_capacity)
        intra_long = intra_density.stack().reset_index()
        intra_long.columns = ["Source_Level", "Target_Level", "Density"]
        intra_long["Type"] = "Intra"
        intra_long["Network"] = Path(cm).stem  # Capture Network ID

        # 2. Process INTER-module densities
        inter_long = inter_density.stack().reset_index()
        inter_long.columns = ["Source_Level", "Target_Level", "Density"]
        inter_long["Type"] = "Inter"
        inter_long["Network"] = Path(cm).stem

        # 3. Combine and Append
        network_summary = pd.concat([intra_long, inter_long], ignore_index=True)
        all_intrainter_density.append(network_summary)

        ##############################################
        # Plot Intra Density
        ##############################################
        cmap_intra = LinearSegmentedColormap.from_list(
            "NordYellowSeq", ["black", NORD_PALETTE[3]]
        )

        # Using NORD_PALETTE[4] (Purple) for Inter
        cmap_inter = LinearSegmentedColormap.from_list(
            "NordPurpleSeq", ["black", NORD_PALETTE[4]]
        )

        # 2. Setup Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # --- Plot 1: Intra Density (Yellow) ---
        sns.heatmap(
            intra_density,
            annot=True,
            fmt=".4f",
            cmap=cmap_intra,  # Apply custom Nord Yellow cmap
            ax=axes[0],
            linewidths=1,
            linecolor="white",  # White lines contrast best inside the heatmap
            vmin=0,
            cbar_kws={"label": "Connection Density (Edge / Possible)"},
            # annot_kws={"size": 12},  # Ensure annotations are readable
        )
        axes[0].set_title("INTRA-Module Density\n(Normalized)")
        axes[0].set_ylabel("Source Level")
        axes[0].set_xlabel("Target Level")

        # --- Plot 2: Inter Density (Purple) ---
        sns.heatmap(
            inter_density,
            annot=True,
            fmt=".4f",
            cmap=cmap_inter,  # Apply custom Nord Purple cmap
            ax=axes[1],
            linewidths=1,
            linecolor="white",
            vmin=0,
            cbar_kws={"label": "Connection Density (Edge / Possible)"},
            # annot_kws={"size": 12},
        )
        axes[1].set_title("INTER-Module Density\n(Normalized)")
        axes[1].set_ylabel("Source Level")
        axes[1].set_xlabel("Target Level")

        # 3. Styling & Saving
        plt.suptitle(
            f"Structural Density Profile: {Path(cm).stem}",
            fontsize=16,
            fontweight="bold",
            y=1.05,
        )

        plt.tight_layout()

        save_path = plots_dir / f"{Path(cm).stem}_IntraInterDensity.png"
        save_path_svg = save_path.with_suffix(".svg")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.savefig(save_path_svg, dpi=300, transparent=True)
        # plt.show()
        plt.close()

        ##############################################
        ##############################################

        # Pre processign to filter modules with all levle of nodes
        active_nodes = module_stats[module_stats["Count"] > 0]
        valid_modules_mask = (
            active_nodes.groupby("NDA_component")["NodeLevel"].nunique() == 3
        )
        valid_module_list = valid_modules_mask[valid_modules_mask].index
        module_stats_subset = module_stats[
            module_stats["NDA_component"].isin(valid_module_list)
        ].copy()
        actual_intra_subset = actual_intra[
            actual_intra["Source_Module"].isin(valid_module_list)
        ].copy()
        intra_capacity_df = module_stats_subset.merge(
            module_stats_subset, on="NDA_component", suffixes=("_Src", "_Tgt")
        )
        intra_capacity_df = module_stats.merge(
            module_stats, on="NDA_component", suffixes=("_Src", "_Tgt")
        )
        intra_capacity_df["Capacity"] = (
            intra_capacity_df["Count_Src"] * intra_capacity_df["Count_Tgt"]
        )
        intra_capacity_df = intra_capacity_df.rename(
            columns={
                "NDA_component": "Module",
                "NodeLevel_Src": "Source_Level",
                "NodeLevel_Tgt": "Target_Level",
            }
        )[["Module", "Source_Level", "Target_Level", "Capacity"]]
        intra_counts_df = (
            actual_intra.groupby(["Source_Module", "Source_Level", "Target_Level"])
            .size()
            .reset_index(name="Actual_Edges")
            .rename(columns={"Source_Module": "Module"})
        )
        intra_density_per_module = intra_capacity_df.merge(
            intra_counts_df, on=["Module", "Source_Level", "Target_Level"], how="left"
        ).fillna(0)  # Fill 0 for level pairs that exist but have no edges
        intra_density_per_module["Density"] = intra_density_per_module[
            "Actual_Edges"
        ] / intra_density_per_module["Capacity"].replace(0, 1)
        print(intra_density_per_module.head())
        print(intra_density_per_module.columns)

        # Optional: Add metadata
        intra_density_per_module["Type"] = "Intra"
        intra_density_per_module["Network"] = Path(cm).stem

        # Result:
        # DataFrame with columns: [Module, Source_Level, Target_Level, Capacity, Actual_Edges, Density]
        print("Intra Density Per Module Head:")
        print(intra_density_per_module.head())

        # Adding the COnnection Type
        intra_density_per_module["Interaction"] = (
            intra_density_per_module["Source_Level"]
            + " - "
            + intra_density_per_module["Target_Level"]
        )
        # Filtering the 0 denissties
        intra_density_per_module = intra_density_per_module[
            intra_density_per_module["Density"] > 0
        ]
        permodule_inter_denisty.append(intra_density_per_module)
        # Order of interactions to plot
        sorted_interactions = (
            intra_density_per_module.groupby("Interaction")["Density"]
            .mean()
            .sort_values(ascending=True)
            .index.tolist()
        )

        ##############################################
        ## Intra module desnity distribution
        ##############################################
        full_nord_palette = [
            "#d08770",  # Nord 12: Orange (Added)
            "#a3be8c",  # Nord 14: Green
            "#b48ead",  # Nord 15: Purple
            "#ebcb8b",  # Nord 13: Yellow
            "#88c0d0",  # Nord 8:  Ice Blue (Added)
            "#bf616a",  # Nord 11: Red
            "#81a1c1",  # Nord 9:  Sky Blue (Added)
            "#8fbcbb",  # Nord 7:  Teal / Frost Green (Added)
            "#5e81ac",  # Nord 10: Deep Blue
        ]

        # 2. Setup Plot
        plt.figure(
            figsize=(9, 6)
        )  # Slightly wider to accommodate more colors/categories

        # --- Layer 1: Boxplot (Background) ---
        ax = sns.boxplot(
            data=intra_density_per_module,
            x="Interaction",
            y="Density",
            palette=full_nord_palette,  # Use the full list
            order=sorted_interactions,
            showfliers=False,
            width=0.4,
        )

        # --- Layer 2: Stripplot (Points) ---
        sns.stripplot(
            data=intra_density_per_module,
            x="Interaction",
            y="Density",
            order=sorted_interactions,
            hue="Interaction",
            hue_order=sorted_interactions,
            palette=full_nord_palette,  # Matches boxplot colors
            alpha=0.9,
            jitter=True,
            edgecolor=NORD_COLORS["dark"],
            linewidth=1.0,
            dodge=False,
            # size=6,
            ax=ax,
            zorder=3,
        )

        # 3. Statistical Annotations
        pairs = list(combinations(sorted_interactions, 2))

        if pairs:
            try:
                annotator = Annotator(
                    ax,
                    pairs,
                    data=intra_density_per_module,
                    x="Interaction",
                    y="Density",
                    order=sorted_interactions,
                )

                annotator.configure(
                    test="Mann-Whitney",
                    text_format="star",
                    loc="inside",
                    comparisons_correction="Benjamini-Hochberg",
                    verbose=False,
                    color=NORD_COLORS["dark"],
                    line_width=1.5,
                )

                annotator.apply_and_annotate()
            except Exception as e:
                print(f"Stats annotation failed: {e}")

        # 4. Styling
        plt.title(
            f"Intra-Module Density Distribution: {Path(cm).stem}",
            pad=15,
        )
        plt.xlabel("Interaction Type")
        plt.ylabel("Density")

        # Expand Y-limit for stats
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top * 1.15)

        # Box Spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(NORD_COLORS["dark"])
            spine.set_linewidth(2.0)

        # Remove legend
        if ax.get_legend():
            ax.get_legend().remove()

        plt.tight_layout()

        # 5. Save
        save_path = plots_dir / f"{Path(cm).stem}_IntraModuleDensity.png"
        save_path_svg = save_path.with_suffix(".svg")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.savefig(save_path_svg, dpi=300, transparent=True)
        # plt.show()
        plt.close()
        print(f"Saved plot: {save_path}")
        ##############################################
        ##############################################

# PLotting the boxplots of INtra and inter dneisities
all_intrainter_density = pd.concat(all_intrainter_density, ignore_index=True)
intrainter_density = all_intrainter_density[
    all_intrainter_density["Density"] > 0
].copy()

# 2. Create "Interaction" Column
intrainter_density["Interaction"] = (
    intrainter_density["Source_Level"] + " - " + intrainter_density["Target_Level"]
)

# 3. Define Logic for Order (Input->Middle->Output)
level_order = ["Input", "Middle", "Output"]
interaction_order = [f"{src} - {tgt}" for src in level_order for tgt in level_order]
existing_interactions = [
    i for i in interaction_order if i in intrainter_density["Interaction"].unique()
]


############################################################################################
### Plotting The INter vs Intra boxplots across all the newtorks
############################################################################################
def plot_intra_inter_comparison(intrainter_density, existing_interactions, plots_dir):
    """
    Grouped Boxplot comparing Intra vs Inter density with Nord Styling.
    Uses Orange (Intra) and Green (Inter) from the user's palette.
    """

    # 1. Setup Colors (User's specific order)
    full_nord_palette = [
        "#d08770",  # Nord 12: Orange (Intra)
        "#a3be8c",  # Nord 14: Green  (Inter)
        "#b48ead",  # Nord 15: Purple
        "#ebcb8b",  # Nord 13: Yellow
        "#88c0d0",  # Nord 8:  Ice Blue
        "#bf616a",  # Nord 11: Red
        "#81a1c1",  # Nord 9:  Sky Blue
        "#8fbcbb",  # Nord 7:  Teal
        "#5e81ac",  # Nord 10: Deep Blue
    ]

    # We only need 2 colors for the 'hue' (Intra vs Inter)
    # Seaborn will automatically pick the first two, but let's be explicit for clarity
    comparison_palette = full_nord_palette[:2]

    # 2. Setup Plot
    plt.figure(figsize=(8, 5))

    # --- Layer 1: Boxplot (Background) ---
    ax = sns.boxplot(
        data=intrainter_density,
        x="Interaction",
        y="Density",
        hue="Type",
        order=existing_interactions,
        hue_order=["Intra", "Inter"],
        palette=comparison_palette,
        showfliers=False,
        width=0.7,
    )

    # --- Layer 2: Stripplot (Overlay Points) ---
    # Note: We use the SAME palette so dots match their box color
    sns.stripplot(
        data=intrainter_density,
        x="Interaction",
        y="Density",
        hue="Type",
        order=existing_interactions,
        hue_order=["Intra", "Inter"],
        palette=comparison_palette,
        dodge=True,  # Crucial: Aligns points with split boxes
        alpha=0.8,  # Solid points
        edgecolor=NORD_COLORS["dark"],
        linewidth=1.0,  # Black outline
        jitter=True,
        # size=5,
        ax=ax,
        zorder=3,
        legend=False,  # Hide strip legend (redundant)
    )

    # 3. Statistical Annotations
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
            line_width=1.5,
            verbose=False,
        )
        annotator.apply_and_annotate()
    except Exception as e:
        print(f"Stats annotation failed: {e}")

    # 4. Styling
    plt.title(
        "Structural Density: Intra- vs Inter-Module Connectivity",
        pad=15,
    )
    plt.tick_params(axis="x", labelsize=12)
    plt.xlabel("Interaction Type")
    plt.ylabel("Connection Density (Normalized)")
    plt.xticks(ha="center")

    # Expand Y-limit for stats
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top * 1.05)

    # Box Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])
        spine.set_linewidth(2.0)

    # Legend
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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    # plt.show()
    plt.close()
    print(f"Saved plot: {save_path}")


plot_intra_inter_comparison(intrainter_density, existing_interactions, plots_dir)


def plot_separate_intra_inter_distributions(
    intrainter_density, existing_interactions, plots_dir
):
    """
    Generates two separate plots:
    1. Only Intra-Module Density (colored by Interaction)
    2. Only Inter-Module Density (colored by Interaction)
    """

    # 1. Full Spectrum Nord Palette for Interaction Types
    full_nord_palette = [
        "#bf616a",  # Red
        "#d08770",  # Orange
        "#ebcb8b",  # Yellow
        "#a3be8c",  # Green
        "#8fbcbb",  # Teal
        "#88c0d0",  # Ice Blue
        "#81a1c1",  # Sky Blue
        "#5e81ac",  # Deep Blue
        "#b48ead",  # Purple
    ]

    # Helper function to plot a single type (Intra or Inter)
    def _plot_subset(density_type, filename_suffix):
        # Filter data
        subset_df = intrainter_density[
            intrainter_density["Type"] == density_type
        ].copy()

        if subset_df.empty:
            print(f"No data for {density_type}, skipping plot.")
            return

        plt.figure(figsize=(8, 6))

        # --- Boxplot ---
        ax = sns.boxplot(
            data=subset_df,
            x="Interaction",
            y="Density",
            order=existing_interactions,
            palette=full_nord_palette,
            showfliers=False,
            width=0.4,
        )

        # --- Stripplot ---
        sns.stripplot(
            data=subset_df,
            x="Interaction",
            y="Density",
            order=existing_interactions,
            hue="Interaction",
            hue_order=existing_interactions,
            palette=full_nord_palette,
            dodge=False,
            edgecolor=NORD_COLORS["dark"],
            jitter=True,
            ax=ax,
            zorder=3,
            legend=False,
        )

        # --- Stats ---
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

        # --- Styling ---
        plt.title(f"Structural Density: {density_type}-Module Only", pad=10)
        plt.xlabel("Interaction Type")
        plt.ylabel("Density (Normalized)")

        # Expand Y for stats
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top * 1.2)

        # Spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(NORD_COLORS["dark"])
            spine.set_linewidth(2.0)

        plt.tight_layout()

        save_path = plots_dir / f"AllNets_{filename_suffix}_Distribution.png"
        save_path_svg = save_path.with_suffix(".svg")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.savefig(save_path_svg, dpi=300, transparent=True)
        plt.close()
        print(f"Saved plot: {save_path}")

    # 2. Generate the two plots
    _plot_subset("Intra", "IntraOnly")
    _plot_subset("Inter", "InterOnly")


# Call this function alongside your main comparison plotter
plot_separate_intra_inter_distributions(
    intrainter_density, existing_interactions, plots_dir
)

############################################################################################
############################################################################################

############################################################################################
### Node distribtuion acorss modules
############################################################################################

# Make a df for the node compositions
nodelvl_module_compo = pd.concat(nodelvl_module_compo)
print(nodelvl_module_compo)
print(nodelvl_module_compo.columns)


def plot_all_nets_node_level_fraction(nodelvl_module_compo, plots_dir):
    """
    Violin + Stripplot of Node Level Fractions across ALL networks.
    Uses Nord Styling (Red/Blue/Green).
    """

    # 1. Setup Order & Colors
    level_order = ["Input", "Middle", "Output"]

    # Strict Color List (Red -> Blue -> Green)
    nord_color_list = [
        NORD_PALETTE[0],  # Red
        NORD_PALETTE[1],  # Blue
        NORD_PALETTE[2],  # Green
    ]

    # 2. Setup Plot
    plt.figure(figsize=(6, 6))

    # --- Layer 1: Violin Plot (Distribution Shape) ---
    ax = sns.violinplot(
        data=nodelvl_module_compo,
        x="NodeLevel",
        y="Fraction",
        order=level_order,
        palette=nord_color_list,
        # inner=None,  # Hide inner box/quartiles (stripplot will show data)
        density_norm="width",  # Normalize width
        cut=0,
    )

    # Manual alpha adjustment for violin bodies
    for poly in ax.collections:
        poly.set_alpha(0.6)

    # --- Layer 2: Stripplot (Individual Points) ---
    sns.stripplot(
        data=nodelvl_module_compo,
        x="NodeLevel",
        y="Fraction",
        order=level_order,
        hue="NodeLevel",
        hue_order=level_order,
        palette=nord_color_list,
        color="black",  # Fallback, but hue overrides it
        # alpha=0.6,
        # s=5,  # Point size
        jitter=True,
        edgecolor=NORD_COLORS["dark"],
        dodge=False,
        zorder=1,
    )

    # 3. Statistical Annotations
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
            line_width=1.5,
            line_offset_to_group=0.1,
            line_offset=0.3,
        )
        annotator.apply_and_annotate()
    except Exception as e:
        print(f"Stats annotation failed: {e}")

    # 4. Styling
    # Adjust Y-limits to fit stats
    ax.set_ylim(-0.2, 1.6)

    plt.title(
        "All Nets Distribution of Node Levels across NDA Components",
        pad=10,
    )
    plt.ylabel("Fraction of Nodes in Level")
    plt.xlabel("Network Layer")

    # Box Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])
        spine.set_linewidth(2.0)

    # Remove legend
    if ax.get_legend():
        ax.get_legend().remove()

    plt.tight_layout()

    save_path = plots_dir / "AllNets_Frac_NodeLvl_Module.png"
    save_path_svg = save_path.with_suffix(".svg")
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    # plt.show()
    plt.close()
    print(f"Saved plot: {save_path}")


plot_all_nets_node_level_fraction(nodelvl_module_compo, plots_dir)

############################################################################################
############################################################################################


# Onatenate all the per module density
permodule_inter_density = pd.concat(permodule_inter_denisty)
sorted_interactions = (
    permodule_inter_density.groupby("Interaction")["Density"]
    .median()
    .sort_values(ascending=False)
    .index.tolist()
)

############################################################################################
### Intra module density dstribution
############################################################################################


def plot_per_module_density_distribution(
    permodule_inter_density, sorted_interactions, cm, plots_dir
):
    """
    Boxplot + Stripplot of Per-Module Density with FULL Nord Styling.
    Uses the expanded Nord palette to cover multiple interaction types.
    """

    # 1. Expanded Nord Palette
    # Red -> Orange -> Yellow -> Green -> Teal -> Cyan -> Blue -> Dark Blue -> Purple
    full_nord_palette = [
        "#bf616a",  # Nord 11: Red
        "#d08770",  # Nord 12: Orange
        "#ebcb8b",  # Nord 13: Yellow
        "#a3be8c",  # Nord 14: Green
        "#8fbcbb",  # Nord 7:  Teal
        "#88c0d0",  # Nord 8:  Ice Blue
        "#81a1c1",  # Nord 9:  Sky Blue
        "#5e81ac",  # Nord 10: Deep Blue
        "#b48ead",  # Nord 15: Purple
    ]

    # 2. Setup Plot
    plt.figure(figsize=(8, 6))

    # --- Layer 1: Boxplot (Background) ---
    ax = sns.violinplot(
        data=permodule_inter_density,
        x="Interaction",
        y="Density",
        palette=full_nord_palette,
        order=sorted_interactions,
        density_norm="width",  # Normalize width
        cut=0,
    )

    # Manual alpha adjustment for violin bodies
    for poly in ax.collections:
        poly.set_alpha(0.6)

    # --- Layer 2: Stripplot (Points) ---
    sns.stripplot(
        data=permodule_inter_density,
        x="Interaction",
        y="Density",
        order=sorted_interactions,
        hue="Interaction",  # Match hue to x
        hue_order=sorted_interactions,
        palette=full_nord_palette,
        jitter=True,
        edgecolor=NORD_COLORS["dark"],
        dodge=False,
        ax=ax,
        zorder=1,
    )

    # 3. Statistical Annotations
    pairs = list(combinations(sorted_interactions, 2))

    if pairs:
        try:
            annotator = Annotator(
                ax,
                pairs,
                data=permodule_inter_density,
                x="Interaction",
                y="Density",
                order=sorted_interactions,
            )

            annotator.configure(
                test="Mann-Whitney",
                text_format="star",
                loc="inside",
                comparisons_correction="Benjamini-Hochberg",
                verbose=False,
                color=NORD_COLORS["dark"],
                line_width=1.5,
            )

            annotator.apply_and_annotate()
        except Exception as e:
            print(f"Stats annotation failed: {e}")

    # 4. Styling
    plt.title("Intra-Module Density Distribution", pad=10)
    plt.xlabel("Interaction Type")
    plt.ylabel("Density (Normalised)")

    # Expand Y-limit for stats
    bottom, top = ax.get_ylim()
    # Ensure bottom is not below 0 if density is non-negative
    bottom = min(bottom, -0.05)
    ax.set_ylim(bottom, top * 1.25)  # Give 25% headroom for stars

    # Box Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])
        spine.set_linewidth(2.0)

    # Remove legend
    if ax.get_legend():
        ax.get_legend().remove()

    plt.tight_layout()

    # 5. Save
    # Assuming plots_dir is defined in your scope
    if "plots_dir" in locals():
        save_path = plots_dir / "All_IntraModuleDensity_Distribution.png"
        save_path_svg = save_path.with_suffix(".svg")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.savefig(save_path_svg, dpi=300, transparent=True)
        print(f"Saved plot: {save_path}")

    # plt.show()
    plt.close()


plot_per_module_density_distribution(
    permodule_inter_density, sorted_interactions, cm, plots_dir
)

############################################################################################
############################################################################################

############################################################################################
### All Networks Module Size Distribution (Violin Plot)
############################################################################################


# def plot_module_sizes_by_network(module_sizes_df, plots_dir):
#     """
#     Plots module size distributions with one violin per network.
#     - Uses Nord Colors.
#     - Log-scaled Y-axis to show outliers clearly.
#     - Custom Legend.
#     """
#
#     # 1. Determine Order (Sort by Median Size)
#     sorted_networks = (
#         module_sizes_df.groupby("Network")["Total_Nodes"]
#         .median()
#         .sort_values(ascending=True)
#         .index.tolist()
#     )
#
#     n_nets = len(sorted_networks)
#
#     # 2. Setup Nord Palette for Many Categories
#     # If we have many networks, we interpolate the Nord colors to create a custom map
#     # Or simply repeat the NORD_PALETTE if distinctness is preferred over gradient.
#     # Here, a LinearSegmentedColormap approach is cleanest for many violin plots.
#     from matplotlib.colors import LinearSegmentedColormap
#
#     # Create a gradient from Blue -> Green -> Yellow -> Orange -> Red
#     nord_gradient = [
#         NORD_COLORS["blue"],
#         NORD_COLORS["green"],
#         NORD_COLORS["yellow"],
#         "#d08770",  # Orange
#         NORD_COLORS["red"],
#         NORD_COLORS["purple"],
#     ]
#
#     # Create distinct colors for each network from this gradient
#     # (Using sns.color_palette with a ListedColormap logic)
#     my_cmap = LinearSegmentedColormap.from_list("NordSeq", nord_gradient, N=n_nets)
#     my_palette = [my_cmap(i) for i in np.linspace(0, 1, n_nets)]
#
#     palette_dict = dict(zip(sorted_networks, my_palette))
#
#     # 3. Setup Plot
#     plt.figure(figsize=(14, 6))
#
#     # --- Violin Plot ---
#     ax = sns.violinplot(
#         data=module_sizes_df,
#         x="Network",
#         y="Total_Nodes",
#         hue="Network",
#         order=sorted_networks,
#         palette=palette_dict,
#         density_norm="width",
#         linewidth=1.5,
#         dodge=False,
#         cut=0,
#     )
#
#     # Manual alpha adjustment for violin bodies
#     for poly in ax.collections:
#         poly.set_alpha(0.6)
#         poly.set_edgecolor(NORD_COLORS["dark"])
#
#     # --- Stripplot ---
#     sns.stripplot(
#         data=module_sizes_df,
#         x="Network",
#         y="Total_Nodes",
#         hue="Network",
#         order=sorted_networks,
#         palette=palette_dict,
#         size=10,
#         alpha=0.7,
#         jitter=True,
#         zorder=1,
#         ax=ax,
#         legend=False,
#         edgecolor="white",
#     )
#
#     plt.title("Module Size Distribution per Network", pad=5)
#     plt.ylabel("Module Size (Total Nodes)")
#     plt.xlabel("Networks")
#
#     # Hide X-axis text (often too crowded)
#     ax.set_xticklabels([])
#     ax.tick_params(axis="x", which="both", length=0)  # Hide x ticks too if preferred
#
#     # Spines
#     for spine in ax.spines.values():
#         spine.set_visible(True)
#         spine.set_color(NORD_COLORS["dark"])
#         spine.set_linewidth(2.0)
#
#     # 5. Legend
#     # Manually Create Legend Handles
#     legend_handles = [
#         mpatches.Patch(color=palette_dict[net], label=net) for net in sorted_networks
#     ]
#
#     plt.legend(
#         handles=legend_handles,
#         title="Organism",
#         bbox_to_anchor=(1.02, 1),
#         loc="upper left",
#         borderaxespad=0,
#         frameon=False,  # Cleaner look
#     )
#
#     plt.tight_layout()
#
#     save_path = plots_dir / "AllNets_ModuleSize_PerNetwork_Violin.png"
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     print(f"Saved plot: {save_path}")
#     plt.close()
#
#
# data_string = """
# Regulatory_Network,Organism,Version,Genomic_Coverage,Completeness
# 100226_v2019_sA22-DBSCR15_eStrong,Streptomyces coelicolor,2019,5.2,2.3
# 224308_v2022_sSW22,Bacillus subtilis,2022,58.2,49.4
# 511145_v2022_sRDB22_eStrong,Escherichia coli,2022,51.3,56.0
# 196627_v2020_s21_eStrong,Corynebacterium glutamicum,2020,71.7,42.6
# 83332_v2018_s15-16,Mycobacterium tuberculosis,2018,62.1,67.3
# 208964_v2020_sRPA20_eStrong,Pseudomonas aeruginosa,2020,18.4,13.7
# """
# meta_df = pd.read_csv(StringIO(data_string))
#
# # --- 2. PREPARE MAIN DATA ---
# # (Assuming 'all_module_sizes' is your list of dataframes)
# all_module_sizes_df = pd.concat(all_module_sizes, ignore_index=True)
#
# # --- 3. PREPROCESSING: SWAP ID -> ORGANISM NAME ---
# # Iterate through metadata to find matches and replace strings
# for index, row in meta_df.iterrows():
#     search_string = row["Regulatory_Network"]
#     organism_name = row["Organism"]
#
#     # Logic: If 'search_string' is inside the 'Network' column string -> Replace with 'organism_name'
#     mask = all_module_sizes_df["Network"].str.contains(search_string, regex=False)
#     all_module_sizes_df.loc[mask, "Network"] = organism_name
#
# threshold = all_module_sizes_df["Total_Nodes"].quantile(0.95)
# filtered_df = all_module_sizes_df[
#     all_module_sizes_df["Total_Nodes"] <= threshold
# ].copy()
# original_count = len(all_module_sizes_df)
# new_count = len(filtered_df)
# print(
#     f"Removed {original_count - new_count} outlier modules ({(1 - new_count / original_count):.1%} of data)"
# )
#
# # --- Run it ---
# # Ensure you have run the aggregation line first:
# plot_module_sizes_by_network(filtered_df, plots_dir)


all_module_sizes_df = pd.concat(all_module_sizes, ignore_index=True)

data_string = """
Regulatory_Network,Organism,Version,Genomic_Coverage,Completeness
100226_v2019_sA22-DBSCR15_eStrong,Streptomyces coelicolor,2019,5.2,2.3
224308_v2022_sSW22,Bacillus subtilis,2022,58.2,49.4
511145_v2022_sRDB22_eStrong,Escherichia coli,2022,51.3,56.0
196627_v2020_s21_eStrong,Corynebacterium glutamicum,2020,71.7,42.6
83332_v2018_s15-16,Mycobacterium tuberculosis,2018,62.1,67.3
208964_v2020_sRPA20_eStrong,Pseudomonas aeruginosa,2020,18.4,13.7
"""
meta_df = pd.read_csv(StringIO(data_string))

# (Simulating your data loading here)
# all_module_sizes_df = pd.concat(all_module_sizes, ignore_index=True)

# --- 2. PREPROCESSING: SWAP ID -> ORGANISM NAME ---
for index, row in meta_df.iterrows():
    search_string = row["Regulatory_Network"]
    organism_name = row["Organism"]

    # Check if column contains the ID, replace with Organism Name
    mask = all_module_sizes_df["Network"].str.contains(search_string, regex=False)
    all_module_sizes_df.loc[mask, "Network"] = organism_name

# --- 3. FILTER OUTLIERS ---
threshold = all_module_sizes_df["Total_Nodes"].quantile(0.95)
filtered_df = all_module_sizes_df[
    all_module_sizes_df["Total_Nodes"] <= threshold
].copy()


# --- 4. PLOTTING FUNCTION ---
def plot_module_sizes_by_network(module_sizes_df, meta_df, plots_dir):
    """
    Plots module size distributions with annotations for Mean and Legend with Coverage.
    """

    # A. Determine Order (Sort by Median Size)
    sorted_networks = meta_df.sort_values("Completeness", ascending=True)[
        "Organism"
    ].tolist()
    n_nets = len(sorted_networks)

    # B. Setup Colors (Nord-ish Gradient)
    # Defining colors manually to ensure variables exist
    nord_colors = ["#5E81AC", "#A3BE8C", "#EBCB8B", "#D08770", "#BF616A", "#B48EAD"]

    # Create gradient
    my_cmap = LinearSegmentedColormap.from_list("NordSeq", nord_colors, N=n_nets)
    my_palette = [my_cmap(i) for i in np.linspace(0, 1, n_nets)]
    palette_dict = dict(zip(sorted_networks, my_palette))

    # C. Setup Plot
    plt.figure(figsize=(14, 6))

    # --- Violin Plot ---
    ax = sns.violinplot(
        data=module_sizes_df,
        x="Network",
        y="Total_Nodes",
        hue="Network",
        order=sorted_networks,
        palette=palette_dict,
        density_norm="width",
        linewidth=1.5,
        dodge=False,
        cut=0,
    )

    # Transparency for violins
    for poly in ax.collections:
        poly.set_alpha(0.5)

    # --- Stripplot ---
    sns.stripplot(
        data=module_sizes_df,
        x="Network",
        y="Total_Nodes",
        hue="Network",
        order=sorted_networks,
        palette=palette_dict,
        size=10,
        jitter=True,
        zorder=1,
        ax=ax,
        legend=False,
        linewidth=0.5,
    )

    plt.ylim(0, 35)

    # --- ANNOTATE MEANS ---
    # Iterate over the sorted networks to place text at the correct x-index
    for i, net in enumerate(sorted_networks):
        # Calculate statistics
        subset = module_sizes_df[module_sizes_df["Network"] == net]
        mean_val = subset["Total_Nodes"].mean()
        max_val = subset["Total_Nodes"].max()

        # Place text slightly above the highest point of that violin
        # (Using max_val + offset ensures it doesn't overlap data points)
        ax.text(
            x=i,
            y=max_val + (max_val * 0.20),  # 5% buffer above max
            s=f"μ={mean_val:.1f}",
            ha="center",
            va="bottom",
            color="#2E3440",
            fontsize=18,
        )

    # D. Styling
    plt.title("Module Size Distribution per Network", pad=15, fontweight="bold")
    plt.ylabel("Module Size (Total Nodes)")
    plt.xlabel("Networks")

    # Hide X labels
    ax.set_xticklabels([])
    ax.tick_params(axis="x", which="both", length=0)

    # Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#2E3440")
        spine.set_linewidth(2.0)

    # E. Custom Legend with Coverage
    legend_handles = []
    for net in sorted_networks:
        # distinct color
        color = palette_dict[net]

        # Lookup Coverage in Meta DF
        # (Assumes 'Network' in df matches 'Organism' in meta_df due to preprocessing)
        try:
            cov = meta_df.loc[meta_df["Organism"] == net, "Completeness"].values[0]
            label_text = f"{net} ({cov}%)"
        except IndexError:
            label_text = net  # Fallback if no match found

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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    print(f"Saved plot: {save_path}")
    plt.close()


# --- Run it ---
# Pass 'filtered_df' and 'meta_df'
plot_module_sizes_by_network(filtered_df, meta_df, plots_dir)
