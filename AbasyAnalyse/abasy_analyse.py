import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from statannotations.Annotator import Annotator
import re
import os


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
def classify_node(row):
    # Treat NaN as 0 for this logic
    out_val = row["MeanAbsOutCoh"] if pd.notna(row["MeanAbsOutCoh"]) else 0
    in_val = row["MeanAbsInCoh"] if pd.notna(row["MeanAbsInCoh"]) else 0

    if out_val == 0 and in_val != 0:
        return "Output"
    elif in_val == 0 and out_val != 0:
        return "Input"
    else:
        # This catches cases where both are 0/NaN
        # or both are non-zero
        return "Middle"


# Function to classify nodes
def classify_node_totcoh(row):
    # Treat NaN as 0 for this logic
    out_val = row["AbsOutCoh"] if pd.notna(row["AbsOutCoh"]) else 0
    in_val = row["AbsInCoh"] if pd.notna(row["AbsInCoh"]) else 0

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


# Create Plots folder if it doesn't exist
plots_dir = absy_dir / "Plots"
plots_dir.mkdir(exist_ok=True)
boxplt_dir = plots_dir / "Box"
boxplt_dir.mkdir(exist_ok=True)
cohhist_dir = plots_dir / "CohHist"
cohhist_dir.mkdir(exist_ok=True)
heiheat_dir = plots_dir / "Hei_Heatmap"
heiheat_dir.mkdir(exist_ok=True)

inout_coh_all = []
all_distributions = []
level_join_densities = []
level_join_walk_densities = []
tot_inout_coh_df = []

for cm in sorted(nondup_cohmat_list):
    print(cm)
    # Read coherence matrix
    cmat = pd.read_parquet(cm)
    # print(cmat)
    num_mat = cmat.select_dtypes(include="number")

    # Reading the walk mat
    wmat = pd.read_parquet(str(cm).replace("_CohMat", "_WalkMats"))["NumWalks"]
    # Sum the walk values
    wmat = wmat.groupby("SourceNode").sum()
    # reorder the soruce nodes to that of target nodes
    wmat = wmat.reindex(wmat.columns.tolist())
    # replace all the 0s with nan for eaiser handling
    wmat = wmat.replace(0, np.nan)
    wmat.index.name = "SourceNode"
    # print(wmat.loc["SCO1489", "SCO1489"])
    # print(wmat.shape)

    # --- 1. Histogram of all walk values ---
    values = wmat.values.flatten()
    values = values[values != 0]
    # print(len(values))
    values = values[~np.isnan(values)]

    ###########################################################################################
    ### Walk Histograms
    ###########################################################################################
    plt.figure(figsize=(6, 4))
    plt.hist(
        values,
        bins=40,
        color=NORD_COLORS["blue"],
        edgecolor=NORD_COLORS["dark"],
        alpha=0.8,  # High opacity for clarity
        linewidth=1.2,  # Slightly thicker edge for the cartoon-ish Nord look
    )
    plt.title(f"{Path(cm).stem} - All Coherence Values", pad=5)
    plt.xlabel("Walk Value")
    plt.ylabel("Frequency")
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])
        spine.set_linewidth(2.0)
    plt.tight_layout()
    save_path = cohhist_dir / f"{Path(cm).stem}_AllWalkHist.png"
    save_path_svg = save_path.with_suffix(".svg")
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    # plt.show()
    plt.close()
    ###########################################################################################
    ###########################################################################################

    ###########################################################################################
    ### Histogram of all coherence values
    ###########################################################################################
    values = num_mat.values.flatten()
    values = values[~np.isnan(values)]

    plt.figure(figsize=(5, 6))
    plt.hist(
        values,
        bins=40,
        color=NORD_COLORS["blue"],
        edgecolor=NORD_COLORS["dark"],
        alpha=0.8,
        linewidth=1.2,
    )
    # --- Styling ---
    plt.title(f"{Path(cm).stem} - All Coherence Values", pad=5)
    plt.xlabel("Coherence Value")
    plt.ylabel("Frequency")
    # Box Spines
    ax = plt.gca()
    # e power convetion
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])
        spine.set_linewidth(2.0)
    plt.tight_layout()
    save_path = cohhist_dir / f"{Path(cm).stem}_AllCohHist.png"
    save_path_svg = save_path.with_suffix(".svg")
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    plt.close()
    ###########################################################################################
    ###########################################################################################

    # Row-wise and column-wise absolute sums
    # row_sums = num_mat.abs().sum(axis=1)
    # col_sums = num_mat.abs().sum(axis=0)
    row_sums = np.nanmean(num_mat.abs(), axis=1)
    col_sums = np.nanmean(num_mat.abs(), axis=0)
    wlk_row_sums = np.nanmean(wmat, axis=1)
    wlk_col_sums = np.nanmean(wmat, axis=0)

    # Create summary DataFrame
    inout_coh = pd.DataFrame(
        # {"Node": num_mat.index, "OutCoh": row_sums.values, "InCoh": col_sums.values}
        {
            "Node": num_mat.index.get_level_values(1),
            "MeanAbsOutCoh": row_sums,
            "MeanAbsInCoh": col_sums,
            "WalkMeanOutVal": wlk_row_sums,
            "WalkMeanInVal": wlk_col_sums,
            # "AbsInCoh": num_mat.abs().sum(axis=1),
            # "AbsOutCoh": num_mat.abs().sum(axis=0),
        }
    )

    # Extract only the gene name
    inout_coh["Node"] = inout_coh["Node"].apply(
        lambda x: x[1] if isinstance(x, tuple) else str(x)
    )
    inout_coh["NodeLevel"] = inout_coh.apply(classify_node, axis=1)

    # sns.scatterplot(data=inout_coh, x="WalkMeanOutVal", y="WalkMeanInVal")
    # plt.show()

    # # Another version of tot_inout_coh
    # tot_inout_coh = inout_coh.copy()
    # tot_inout_coh["Network"] = Path(cm).stem.replace("_CohMat", "")
    # tot_inout_coh_df.append(tot_inout_coh)

    # Extractin diagnoal values from the dataframe
    diag_vals = pd.Series(np.diag(cmat), index=cmat.index, name="DiagCohval").to_frame()
    diag_vals = diag_vals.reset_index()
    diag_vals = diag_vals.drop(columns="Group")

    inout_coh = pd.merge(
        inout_coh, diag_vals, how="left", left_on="Node", right_on="SourceNode"
    )
    inout_coh = inout_coh.drop(columns="SourceNode")

    clean_stem = re.sub(
        r"_regNetwork(?:_Strong)?_CohMat$",  # matches both Strong and non-Strong
        "",
        cm.stem,
    )

    # Store percentage distribution
    dist = inout_coh["NodeLevel"].value_counts(normalize=True) * 100
    dist.name = Path(cm).stem
    all_distributions.append(dist)

    ###########################################################################################
    ### Create a single figure with two boxplots side by side
    ###########################################################################################

    fig, axes = plt.subplots(figsize=(8, 10))

    # Fixed order for NodeLevel
    node_order = ["Input", "Middle", "Output"]

    cohdf_melted = inout_coh.melt(
        id_vars=["NodeLevel"],  # Column(s) to keep
        value_vars=["MeanAbsInCoh", "MeanAbsOutCoh"],  # Columns to unpivot
        var_name="CoherenceType",  # Name for the new column of 'In'/'Out'
        value_name="MeanAbsCoh",  # Name for the new column of values
    )

    cohdf_melted["CoherenceType"] = cohdf_melted["CoherenceType"].map(
        {"MeanAbsInCoh": "Incoming", "MeanAbsOutCoh": "Outgoing"}
    )

    sns.stripplot(
        data=cohdf_melted,
        x="NodeLevel",  # Group by Node Level on the x-axis
        y="MeanAbsCoh",  # The values are on the y-axis
        hue="CoherenceType",  # Split the boxes by 'Incoming' vs 'Outgoing'
        order=node_order,
        dodge=True,
        ax=axes,
        # A new palette for the new hue
        palette={"Incoming": "#6baed6", "Outgoing": "#31a354"},
    )

    # --- 4. Set Labels and Title ---
    # (Assuming 'cm' is a Path object as in your original code)
    fig.suptitle(f"{cm.stem} - Coherence by Node Level")
    axes.set_xlabel("Node Level")
    axes.set_ylabel("Mean Absolute Coherence")
    axes.legend(title="Coherence Type")
    plt.tight_layout()
    # Save the combined figure
    save_path = boxplt_dir / f"{Path(cm).stem}_InOutCohBox.png"
    save_path_svg = save_path.with_suffix(".svg")
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    plt.close(fig)
    ###########################################################################################
    ###########################################################################################

    ###########################################################################################
    ### Create a single figure with two boxplots side by side
    ###########################################################################################
    fig, axes = plt.subplots(figsize=(8, 10))

    # Fixed order for NodeLevel
    node_order = ["Input", "Middle", "Output"]

    walkdf_melted = inout_coh.melt(
        id_vars=["NodeLevel"],  # Column(s) to keep
        value_vars=["WalkMeanInVal", "WalkMeanOutVal"],  # Columns to unpivot
        var_name="WalkType",  # Name for the new column of 'In'/'Out'
        value_name="MeanWalkVal",  # Name for the new column of values
    )

    walkdf_melted["WalkType"] = walkdf_melted["WalkType"].map(
        {"WalkMeanInVal": "Incoming", "WalkMeanOutVal": "Outgoing"}
    )

    sns.stripplot(
        data=walkdf_melted,
        x="NodeLevel",  # Group by Node Level on the x-axis
        y="MeanWalkVal",  # The values are on the y-axis
        hue="WalkType",  # Split the boxes by 'Incoming' vs 'Outgoing'
        order=node_order,
        dodge=True,
        ax=axes,
        # A new palette for the new hue
        palette={"Incoming": "#6baed6", "Outgoing": "#31a354"},
    )

    # --- 4. Set Labels and Title ---
    # (Assuming 'cm' is a Path object as in your original code)
    fig.suptitle(f"{cm.stem} - WalkVal by Node Level")
    axes.set_xlabel("Node Level")
    axes.set_ylabel("Mean Walk Value")
    axes.legend(title="Walk Type")
    plt.tight_layout()
    # Save the combined figure
    save_path = boxplt_dir / f"{Path(cm).stem}_InOutWalkBox.png"
    save_path_svg = save_path.with_suffix(".svg")
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    plt.close(fig)
    ###########################################################################################
    ###########################################################################################

    # Making Abs In and out coh df
    # Row-wise and column-wise absolute sums
    row_sums = num_mat.abs().sum(axis=1)
    col_sums = num_mat.abs().sum(axis=0)
    # Create summary DataFrame
    tot_inout_coh = pd.DataFrame(
        # {"Node": num_mat.index, "OutCoh": row_sums.values, "InCoh": col_sums.values}
        {
            "Node": num_mat.index.get_level_values(1),
            "AbsOutCoh": row_sums,
            "AbsInCoh": col_sums,
        }
    )
    # Extract only the gene name
    tot_inout_coh["Node"] = tot_inout_coh["Node"].apply(
        lambda x: x[1] if isinstance(x, tuple) else str(x)
    )
    tot_inout_coh["NodeLevel"] = tot_inout_coh.apply(classify_node_totcoh, axis=1)
    tot_inout_coh["Network"] = Path(cm).stem.replace("_CohMat", "")

    # sns.scatterplot(data=tot_inout_coh, x="AbsInCoh", y="AbsOutCoh", hue="NodeLevel")
    # plt.show()

    tot_inout_coh_df.append(tot_inout_coh)

    ###########################################################################################
    #### Heatmap of connection density based on NodeLevel
    ###########################################################################################
    topo_file = topos_dir / f"{Path(cm).stem.replace('_CohMat', '')}.topo"

    if topo_file.exists():
        # Read the topo file
        topo_df = pd.read_csv(
            topo_file,
            sep=r"\s+",
            # usecols=[0, 1, 2],
            # names=["Source", "Target"],
            # header=None,
        )

        # Map Source and Target nodes to NodeLevel
        node_level_dict = inout_coh.set_index("Node")["NodeLevel"].to_dict()
        topo_df["SourceLevel"] = topo_df["Source"].map(node_level_dict)
        topo_df["TargetLevel"] = topo_df["Target"].map(node_level_dict)
        # print(topo_df)
        # print(topo_df.columns)

        # Drop edges where either node is missing in the node_level_dict
        topo_df = topo_df.dropna(subset=["SourceLevel", "TargetLevel"])

        # Create 3x3 matrix for counts
        node_order = ["Input", "Middle", "Output"]
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

        # 2. Compute possible edges A x B
        possible_edges = pd.DataFrame(index=node_order, columns=node_order, dtype=float)
        for A in node_order:
            for B in node_order:
                possible_edges.loc[A, B] = node_counts[A] * node_counts[B]

        # 3. Actual counts
        heatmap_counts = pd.crosstab(
            topo_df["SourceLevel"],
            topo_df["TargetLevel"],
            rownames=["Source"],
            colnames=["Target"],
        ).reindex(index=node_order, columns=node_order, fill_value=0)

        # 4. Density = actual edges / possible edges
        heatmap_density = heatmap_counts / possible_edges

        # Converting the levels to long form
        long_heatmap_density = heatmap_density.stack().reset_index()
        long_heatmap_density.columns = ["Source_Level", "Target_Level", "Density"]
        long_heatmap_density["Network"] = str(Path(cm).stem)
        # print(long_heatmap_density)
        # Append to the list
        level_join_densities.append(long_heatmap_density)

        # Add a small epsilon to avoid log(0)
        epsilon = 1e-9
        log_density = np.log10(heatmap_density + epsilon)  # log(1 + value)

        ###########################################################################################
        ###########################################################################################
        nord_blue_cmap = LinearSegmentedColormap.from_list(
            "NordBlueSeq",
            [
                NORD_COLORS["dark"],
                NORD_COLORS["blue"],
            ],
        )

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            log_density,
            annot=True,
            fmt=".2f",
            cmap=nord_blue_cmap,  # Apply custom map
            cbar_kws={"label": "Log10 Density"},
            linewidths=0.5,  # Add grid lines for "blocky" look
            # linecolor="white",  # White lines separate cells cleanly
            # annot_kws={"color": NORD_COLORS["dark"]},  # Ensure text is Nord Dark
        )
        plt.title(f"{Path(cm).stem}", pad=10)  # Clean title, no bold
        # Heatmaps often hide spines, but if you want the border:
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(NORD_COLORS["dark"])
            spine.set_linewidth(2.0)
        plt.tight_layout()
        save_path = heiheat_dir / f"{Path(cm).stem}_NodeLevelHeatmapLog.png"
        save_path_svg = save_path.with_suffix(".svg")
        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path_svg, dpi=300, transparent=True)
        plt.close()
    ###########################################################################################
    ###########################################################################################

    ######## walk ma tplot
    # 2. Get the list of levels and the mapping
    node_level_dict = inout_coh.set_index("Node")["NodeLevel"].to_dict()
    levels = ["Input", "Middle", "Output"]

    ####### Map the levels as index

    # 1. Create a binary membership matrix (Rows=Nodes, Cols=Levels)
    # This creates a matrix where 1 indicates the node belongs to that level
    dummies = pd.get_dummies(pd.Series(node_level_dict))

    # Align dummies to your wmat index to ensure order is perfect
    # (Fillna 0 because if a node isn't in the dict, it belongs to no level)
    dummies = dummies.reindex(wmat.index).fillna(0)

    # 2. Matrix Multiplication
    # Transpose(Dummies) * WMat * Dummies
    # resulting shape: (Levels x Levels)
    level_sums = dummies.T.dot(wmat.fillna(0)).dot(dummies)

    # 1. Count how many nodes exist in each level
    # dummies is (Nodes x Levels), so summing gives a Series of counts
    level_node_counts = dummies.sum()

    # 2. Calculate the total area (matrix size) for each block
    # This performs: Count(Level_A) * Count(Level_B) for every combination
    # Result is a (Levels x Levels) matrix of divisors
    block_sizes = pd.DataFrame(
        np.outer(level_node_counts, level_node_counts),
        index=level_node_counts.index,
        columns=level_node_counts.index,
    )

    # 3. Calculate "Normal" Mean (Total Sum / Total Block Area)
    level_means = level_sums / block_sizes
    level_means = level_means.astype(float)

    ###########################################################################################
    ### Mean Communicability Heatmap
    ###########################################################################################
    nord_blue_cmap = LinearSegmentedColormap.from_list(
        "NordBlueSeq",
        [
            NORD_COLORS["dark"],
            NORD_COLORS["blue"],
        ],
    )
    # Set up the figure size (adjust 10, 8 as needed based on your number of levels)
    plt.figure(figsize=(6, 5))
    # Create the heatmap
    sns.heatmap(
        level_means,
        annot=True,
        fmt=".2f",
        cmap=nord_blue_cmap,  # Apply custom map
        cbar_kws={"label": "Log10 Mean Normalised Communicability"},
        linewidths=0.5,  # Add grid lines for "blocky" look
        # linecolor="white",  # White lines separate cells cleanly
        # annot_kws={"color": NORD_COLORS["dark"]},  # Ensure text is Nord Dark
    )
    # plt.title(f"{Path(cm).stem}", pad=10)  # Clean title, no bold
    plt.ylabel("Source Level")
    plt.xlabel("Target Level")
    # Heatmaps often hide spines, but if you want the border:
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])
        spine.set_linewidth(2.0)
    # # Rotate tick labels if they are long
    # plt.xticks(rotation=45, ha="right")
    # plt.yticks(rotation=0)
    plt.tight_layout()
    # plt.show()
    plt.savefig(heiheat_dir / f"{Path(cm).stem}_WalkNodeLevelHeatmapLog.png", dpi=300)
    plt.savefig(
        heiheat_dir / f"{Path(cm).stem}_WalkNodeLevelHeatmapLog.svg",
        dpi=300,
        transparent=True,
    )
    plt.close()
    ###########################################################################################
    ###########################################################################################

    # Min max noramlise the WalkValue Columns
    g_max = level_means.max().max()
    g_min_nonzero = level_means.replace(0, np.nan).min().min()
    level_means = (level_means - g_min_nonzero) / (g_max - g_min_nonzero)
    # Make the level into lonf form and add ntowrk name
    level_means = level_means.stack().reset_index()
    level_means.columns = ["Source_Level", "Target_Level", "WalkValue (MinMax)"]
    level_means["Network"] = Path(cm).stem
    level_join_walk_densities.append(level_means)


# --- Bar plot across networks for NodeLevel percentages ---
dist_df = pd.DataFrame(all_distributions).fillna(0)
mean_vals = dist_df.mean().reindex(["Input", "Middle", "Output"])
std_vals = dist_df.std().reindex(["Input", "Middle", "Output"])

###########################################################################################
###########################################################################################
nord_map = {
    "Input": NORD_COLORS["red"],
    "Middle": NORD_COLORS["blue"],
    "Output": NORD_COLORS["green"],
}
# Create list of colors matching the data index
bar_colors = [nord_map.get(idx, NORD_COLORS["gray"]) for idx in mean_vals.index]
plt.figure(figsize=(5, 6))
# 2. Plot Bars
bars = plt.bar(
    mean_vals.index,
    mean_vals.values,
    yerr=std_vals.values,
    capsize=5,
    color=bar_colors,  # Apply consistent Nord colors
    edgecolor=NORD_COLORS["dark"],  # Dark borders
    linewidth=1.5,
    error_kw=dict(ecolor=NORD_COLORS["dark"], lw=1.5, capthick=1.5),  # Style error bars
)
# 3. Styling
plt.ylabel("Percentage of Nodes (%)")
plt.xlabel("Node Level")
plt.title("Average Node Level Distribution Across Networks", pad=15)
# Adjust Y-limit to fit annotations
# Adding a bit more headroom (1.3x) for the text
plt.ylim(0, max(mean_vals.values + std_vals.values) * 1.3)
# Annotate bars
for bar, mean, std in zip(bars, mean_vals.values, std_vals.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + std + 1,  # Place just above the error bar
        f"{mean:.1f} ± {std:.1f}%",
        ha="center",
        va="bottom",
        color=NORD_COLORS["dark"],
        fontsize=14,
    )
# Spines
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color(NORD_COLORS["dark"])
    spine.set_linewidth(2.0)
plt.tight_layout()
save_path = plots_dir / "NodeLevelDist.png"
save_path_svg = save_path.with_suffix(".svg")
plt.savefig(save_path, dpi=300)
plt.savefig(save_path_svg, dpi=300, transparent=True)
plt.close()
###########################################################################################
###########################################################################################


# Plot the bar plot of the join probabilities
level_join_densities = pd.concat(level_join_densities)

# For Bar Plot join the level source target
level_join_densities["Connection Density"] = (
    level_join_densities["Source_Level"] + " - " + level_join_densities["Target_Level"]
)
# print(level_join_densities)
# print(level_join_densities.columns)

# # PLotting the boxplot with the order increasing mean density of connections
# Barplot with mean and SD
# ---- 1. Compute stats ----
stats = (
    level_join_densities.groupby("Connection Density")["Density"]
    .agg(["median", "std"])
    .reset_index()
)

# ---- 2. Filter categories with mean == 0 ----
stats = stats[stats["median"] != 0]
filtered_df = level_join_densities[
    level_join_densities["Connection Density"].isin(stats["Connection Density"])
]

# ---- 3. ORDER BY MEAN (descending) ----
order = stats.sort_values("median", ascending=False)["Connection Density"]

###########################################################################################
## Connectviton Density Plot
###########################################################################################
# 1. Setup Plot
fig, ax = plt.subplots(figsize=(6, 6))
sns.boxplot(
    data=filtered_df,
    x="Connection Density",
    y="Density",
    hue="Connection Density",
    order=order,
    palette=NORD_PALETTE,  # Nord colors for categories
    linewidth=1.5,
    width=0.4,
    ax=ax,
    showfliers=False,
)
sns.stripplot(
    data=filtered_df,
    x="Connection Density",
    y="Density",
    order=order,
    hue="Connection Density",  # Match hue to x
    palette=NORD_PALETTE,  # Same palette
    ax=ax,
    # edgecolor=NORD_COLORS["dark"],  # Dark borders on points
    linewidth=1.0,
    size=8,
    jitter=True,
    alpha=0.8,
    legend=False,
)
# 2. Stats Annotation
levels = sorted(filtered_df["Connection Density"].unique())
# Ensure levels follow the specified 'order' if provided
if order is not None:
    levels = [l for l in order if l in levels]
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
            order=order,
        )
        annotator.configure(
            test="Mann-Whitney",
            text_format="star",
            loc="inside",
            verbose=False,
            color=NORD_COLORS["dark"],  # Dark stats text
            line_width=1.5,
        )
        annotator.apply_and_annotate()
    except Exception as e:
        print(f"Stats annotation failed: {e}")
ax.tick_params(axis="x", labelsize=12)
# Expand Y-limit for stats
bottom, top = ax.get_ylim()
ax.set_ylim(bottom, top * 1.10)
plt.xticks(rotation=0, ha="center")
# Spines
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color(NORD_COLORS["dark"])
    spine.set_linewidth(2.0)
plt.tight_layout()
save_path = plots_dir / "LevelContactBox.png"
save_path_svg = save_path.with_suffix(".svg")
plt.savefig(save_path, dpi=300)
plt.savefig(save_path_svg, dpi=300, transparent=True)
plt.close()
###########################################################################################
###########################################################################################

# Plot the bar plot of the join probabilities
level_join_walk_densities = pd.concat(level_join_walk_densities)

# For Bar Plot join the level source target
level_join_walk_densities["Connection Density"] = (
    level_join_walk_densities["Source_Level"]
    + " - "
    + level_join_walk_densities["Target_Level"]
)
# print(level_join_walk_densities)
# print(level_join_walk_densities.columns)

# # PLotting the boxplot with the order increasing mean density of connections
# Barplot with mean and SD
# ---- 1. Compute stats ----
stats = (
    level_join_walk_densities.groupby("Connection Density")["WalkValue (MinMax)"]
    .agg(["median", "std"])
    .reset_index()
)

# ---- 2. Filter categories with mean == 0 ----
stats = stats[stats["median"] > 0]
filtered_df = level_join_walk_densities[
    level_join_walk_densities["Connection Density"].isin(stats["Connection Density"])
]

# ---- 3. ORDER BY MEAN (descending) ----
order = stats.sort_values("median", ascending=False)["Connection Density"]

###########################################################################################
###########################################################################################
# 1. Setup Plot
fig, ax = plt.subplots(figsize=(6, 6))
sns.boxplot(
    data=filtered_df,
    x="Connection Density",
    y="WalkValue (MinMax)",
    hue="Connection Density",
    order=order,
    palette=NORD_PALETTE,  # Consistent Nord colors
    linewidth=1.5,
    width=0.4,
    ax=ax,
    showfliers=False,
)
sns.stripplot(
    data=filtered_df,
    x="Connection Density",
    y="WalkValue (MinMax)",
    hue="Connection Density",  # <--- Match hue to boxplot
    order=order,
    palette=NORD_PALETTE,  # Same palette
    ax=ax,
    # edgecolor=NORD_COLORS["dark"],  # Dark borders
    linewidth=1.0,
    size=8,
    jitter=True,
    alpha=0.8,
    legend=False,
)
# 2. Stats Annotation
levels = sorted(filtered_df["Connection Density"].unique())
if order is not None:
    levels = [l for l in order if l in levels]
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
            y="WalkValue (MinMax)",
            order=order,
        )
        annotator.configure(
            test="Mann-Whitney",
            text_format="star",
            loc="inside",
            verbose=False,
            color=NORD_COLORS["dark"],  # Dark stats text
            line_width=1.5,
        )
        annotator.apply_and_annotate()
    except Exception as e:
        print(f"Stats annotation failed: {e}")

# 3. Styling
bottom, top = ax.get_ylim()
ax.set_ylim(bottom, top * 1.15)  # Expand Y for stats

plt.xticks(rotation=0, ha="center")
plt.xlabel("Connection Density")
plt.ylabel("Mean Norm. Communicability (MinMax)")
# plt.title("...", pad=5) # Add title if needed with pad=5
# Spines
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color(NORD_COLORS["dark"])
    spine.set_linewidth(2.0)

plt.tight_layout()
save_path = plots_dir / "Walk_LevelContactBox.png"
save_path_svg = save_path.with_suffix(".svg")
plt.savefig(save_path, dpi=300)
plt.savefig(save_path_svg, dpi=300, transparent=True)
plt.close()
# fig, ax = plt.subplots(figsize=(8, 8))
#
# sns.boxplot(
#     data=filtered_df,
#     x="Connection Density",
#     y="WalkValue (MinMax)",
#     linewidth=1.2,
#     width=0.4,
#     order=order,  # <--- ORDER HERE
#     ax=ax,
#     showfliers=False,
# )
#
# sns.stripplot(
#     data=filtered_df,
#     x="Connection Density",
#     y="WalkValue (MinMax)",
#     order=order,
#     ax=ax,
#     color="black",  # or let seaborn choose a default
#     size=8,
#     jitter=True,
#     alpha=0.7,  # transparency so boxes remain visible
# )
#
# levels = sorted(filtered_df["Connection Density"].unique())
# pairs = [
#     (levels[i], levels[j])
#     for i in range(len(levels))
#     for j in range(i + 1, len(levels))
# ]
#
# # 2. Setup the Annotator
# annotator = Annotator(
#     ax=plt.gca(),
#     pairs=pairs,
#     data=filtered_df,  # Use the FULL dataframe here if possible, or the one used for boxplot
#     x="Connection Density",
#     y="WalkValue (MinMax)",
# )
#
# # 3. Configure and Plot
# annotator.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=2)
# annotator.apply_and_annotate()
#
# plt.tight_layout()
#
# plt.xticks(rotation=0, ha="center")
# plt.tight_layout()
# plt.savefig(plots_dir / "Walk_LevelContactBox.png", dpi=300)
# # plt.show()
# plt.close()
###########################################################################################
###########################################################################################


# PLotting the in and out coh across newtorks
tot_inout_coh_df = pd.concat(tot_inout_coh_df).reset_index(drop=True)
# Getting the InCOh ratio and out coh ratio
tot_inout_coh_df["FracInCoh"] = tot_inout_coh_df["AbsInCoh"] / (
    tot_inout_coh_df["AbsInCoh"] + tot_inout_coh_df["AbsOutCoh"]
)
tot_inout_coh_df["FracOutCoh"] = 1 - tot_inout_coh_df["FracInCoh"]
tot_inout_coh_df["Diff_InOutCoh"] = (
    tot_inout_coh_df["FracInCoh"] - tot_inout_coh_df["FracOutCoh"]
)
tot_inout_coh_df["TotCoh"] = (
    tot_inout_coh_df["AbsInCoh"] + tot_inout_coh_df["AbsOutCoh"]
)

tot_inout_coh_df.to_csv(absy_dir / "Tot_InOutCoh.csv", sep="\t", index=False)

###########################################################################################
###########################################################################################
# 1. Define Color Mapping
palette_node_level = {
    "Input": NORD_COLORS["red"],
    "Middle": NORD_COLORS["blue"],
    "Output": NORD_COLORS["green"],
}

# 2. Setup Plot
plt.figure(figsize=(7, 6))

# --- Scatterplot ---
sns.scatterplot(
    data=tot_inout_coh_df,
    x="AbsInCoh",
    y="AbsOutCoh",
    hue="NodeLevel",
    hue_order=["Input", "Middle", "Output"],
    palette=palette_node_level,
    edgecolor=NORD_COLORS["dark"],
    alpha=0.8,
    s=60,
)

# 3. Styling
plt.title("Absolute In vs Out Coherence", pad=5)
plt.xlabel("Absolute Incoming Coherence")
plt.ylabel("Absolute Outgoing Coherence")

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    borderaxespad=0,
    frameon=True,
    title="Node Level",
)

# Spines
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color(NORD_COLORS["dark"])
    spine.set_linewidth(2.0)

plt.tight_layout()
save_path = plots_dir / "AbsInvsOutCoh.png"
save_path_svg = save_path.with_suffix(".svg")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.savefig(save_path_svg, dpi=300, bbox_inches="tight", transparent=True)
plt.close()
###########################################################################################
###########################################################################################


levels = sorted(tot_inout_coh_df["NodeLevel"].unique())  # ['Input', 'Middle', 'Output']
fig, axes = plt.subplots(nrows=len(levels), ncols=1, figsize=(8, 10), sharex=True)

###########################################################################################
###########################################################################################
palette_node_level = {
    "Input": NORD_COLORS["red"],
    "Middle": NORD_COLORS["blue"],
    "Output": NORD_COLORS["green"],
}
# 2. Iterate and Plot
for ax, level in zip(axes, levels):
    subset = tot_inout_coh_df[tot_inout_coh_df["NodeLevel"] == level]
    # Get specific color for this level
    level_color = palette_node_level.get(level, NORD_COLORS["gray"])
    sns.histplot(
        data=subset,
        x="Diff_InOutCoh",
        bins=30,
        ax=ax,
        color=level_color,  # Dynamic Nord Color
        edgecolor=NORD_COLORS["dark"],  # Dark border
        linewidth=1.2,  # Slightly thicker edge
        alpha=0.8,
    )
    ax.set_title(f"Difference (In - Out) for {level} Nodes", pad=5)
    ax.set_ylabel("Count")
    ax.set_xlabel("Difference (InCoh - OutCoh)")  # Set label per axis
    # Remove grid (Nord style is usually clean) or keep strictly controlled
    ax.grid(False)
    # Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])
        spine.set_linewidth(2.0)
plt.tight_layout()
save_path = plots_dir / "DiffInOutDistribution.png"
save_path_svg = save_path.with_suffix(".svg")
plt.savefig(save_path, dpi=300)
plt.savefig(save_path_svg, dpi=300, transparent=True)
plt.close()
# # 2. Iterate and Plot
# for ax, level in zip(axes, levels):
#     subset = tot_inout_coh_df[tot_inout_coh_df["NodeLevel"] == level]
#
#     sns.histplot(
#         data=subset,
#         x="Diff_InOutCoh",
#         # OPTION 1: Set exact width (e.g., 0.05 units)
#         # binwidth=0.05,
#         # OPTION 2: Set total number of bins (e.g., 50)
#         bins=30,
#         # kde=True,  # Optional: Add a density curve
#         ax=ax,
#         color="teal",
#         edgecolor="black",
#     )
#
#     ax.set_title(f"Difference (In - Out) for {level} Nodes")
#     ax.set_ylabel("Count")
#     ax.grid(True, alpha=0.3)
#
# plt.xlabel("Difference (InCoh - OutCoh)")
# plt.tight_layout()
# plt.savefig(plots_dir / "DiffInOutDistribution.png", dpi=100)
# # plt.show()
# plt.close()
###########################################################################################
###########################################################################################
