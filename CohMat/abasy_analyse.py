import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import re
import os

# Nord Colors
nord_colors = [
    "#D08770",
    "#8FBCBB",
    "#B48EAD",
    "#A3BE8C",
    "#5E81AC",
    "#BF616A",
    "#88C0D0",
    "#EBCB8B",
    "#81A1C1",
]

plt.rcParams.update(
    {
        "axes.prop_cycle": plt.cycler(color=nord_colors),
        "font.family": "Roboto",
        "legend.fontsize": 14,
        "legend.title_fontsize": 14,
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.25,
    }
)
plt.style.use("seaborn-v0_8-deep")
sns.set_palette(sns.color_palette(nord_colors))


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


# def classify_node(row):
#     if row["OutCoh"] == 0 and row["InCoh"] != 0:
#         return "Output"
#     elif row["InCoh"] == 0 and row["OutCoh"] != 0:
#         return "Input"
#     else:
#         return "Middle"


# Abasy network data
absy_dir = Path("../../AbasyCohResults/")
gene_info_dir = Path("../../../AbasyNets/")
# cohmat_list = list(absy_dir.glob("*/*_Strong_CohMat.parquet"))
cohmat_list = list(absy_dir.glob("*/*_CohMat.parquet"))


# Create Plots folder if it doesn't exist
plots_dir = absy_dir / "Plots"
plots_dir.mkdir(exist_ok=True)
boxplt_dir = plots_dir / "Box"
boxplt_dir.mkdir(exist_ok=True)
cohhist_dir = plots_dir / "CohHist"
cohhist_dir.mkdir(exist_ok=True)
heiheat_dir = plots_dir / "Hei_Heatmap"
heiheat_dir.mkdir(exist_ok=True)

all_distributions = []
level_join_densities = []
tf_enrichments = []

for cm in cohmat_list:
    # Read coherence matrix
    cmat = pd.read_parquet(cm)
    # print(cmat)
    num_mat = cmat.select_dtypes(include="number")

    # --- 1. Histogram of all coherence values ---
    values = num_mat.values.flatten()
    values = values[~np.isnan(values)]

    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=40, color="steelblue", edgecolor="black", alpha=0.7)
    plt.title(f"{Path(cm).stem} - All Coherence Values")
    plt.xlabel("Coherence Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(cohhist_dir / f"{Path(cm).stem}_AllCohHist.png", dpi=300)
    plt.close()

    # Row-wise and column-wise absolute sums
    # row_sums = num_mat.abs().sum(axis=1)
    # col_sums = num_mat.abs().sum(axis=0)
    row_sums = np.nanmean(num_mat.abs(), axis=1)
    col_sums = np.nanmean(num_mat.abs(), axis=0)

    # Create summary DataFrame
    inout_coh = pd.DataFrame(
        # {"Node": num_mat.index, "OutCoh": row_sums.values, "InCoh": col_sums.values}
        {
            "Node": num_mat.index.get_level_values(1),
            "MeanAbsOutCoh": row_sums,
            "MeanAbsInCoh": col_sums,
        }
    )

    # Extract only the gene name
    inout_coh["Node"] = inout_coh["Node"].apply(
        lambda x: x[1] if isinstance(x, tuple) else str(x)
    )
    inout_coh["NodeLevel"] = inout_coh.apply(classify_node, axis=1)

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

    gene_info_path = (
        gene_info_dir
        / f"{clean_stem}_regNet-genes-modules"
        / f"{clean_stem}_geneInformation.tsv"
    )

    gene_info_df = pd.read_csv(gene_info_path, sep="\t", engine="python").rename(
        columns={"Gene_name": "Node"}
    )

    # # Import the gene information
    # gene_info_df = pd.read_csv(
    #     gene_info_dir
    #     / cm.stem.replace("_regNetwork_Strong_CohMat", "_regNet-genes-modules")
    #     / f"{cm.stem.replace('_regNetwork_Strong_CohMat', '')}_geneInformation.tsv",
    #     sep="\t",
    #     engine="python",
    # ).rename(columns={"Gene_name": "Node"})
    print(gene_info_df)

    # Merge the node info witht he inout_coh dataframe
    inout_coh = pd.merge(inout_coh, gene_info_df, how="left", on="Node")

    # Make a new column for transcriptional factor or not
    inout_coh["TF"] = inout_coh["Product_function"].str.contains(
        "transcriptional", case=False, na=False
    )
    print(inout_coh["TF"].value_counts(normalize=True))

    tf_percentage = inout_coh.groupby("NodeLevel")["TF"].mean() * 100
    print(tf_percentage)

    # print(inout_coh)
    # print(inout_coh.columns)
    # print(inout_coh.dtypes)

    # Get odds ratio and other metrics
    # tflevel_enrich_df = []

    for level in inout_coh["NodeLevel"].unique():
        # Build contingency table
        a = inout_coh[(inout_coh["NodeLevel"] == level) & (inout_coh["TF"])].shape[0]
        b = inout_coh[(inout_coh["NodeLevel"] == level) & (~inout_coh["TF"])].shape[0]
        c = inout_coh[(inout_coh["NodeLevel"] != level) & (inout_coh["TF"])].shape[0]
        d = inout_coh[(inout_coh["NodeLevel"] != level) & (~inout_coh["TF"])].shape[0]

        # fisher exact test (testing TF enrichment)
        oddsratio, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")

        tf_enrichments.append(
            {
                "NodeLevel": level,
                "TF_in_group": a,
                "NonTF_in_group": b,
                "TF_outside": c,
                "NonTF_outside": d,
                "Odds_ratio": oddsratio,
                "P_value": pvalue,
                "TF_percentage": (a / (a + b)) * 100,
                "Network": cm.stem,
            }
        )

    # tflevel_enrich_df = pd.DataFrame(tflevel_enrich_df).sort_values(by="NodeLevel")
    # print(tflevel_enrich_df)

    # Store percentage distribution
    dist = inout_coh["NodeLevel"].value_counts(normalize=True) * 100
    dist.name = Path(cm).stem
    all_distributions.append(dist)

    # --- Create a single figure with two boxplots side by side ---
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

    # # Outgoing Coherence boxplot
    # sns.boxplot(
    #     data=inout_coh,
    #     x="NodeLevel",
    #     y="MeanAbsOutCoh",
    #     hue="NodeLevel",
    #     order=node_order,
    #     palette={"Input": "#31a354", "Middle": "#fd8d3c", "Output": "#6baed6"},
    #     ax=axes[0],
    # )
    # # axes[0].set_title(f"{Path(cm).stem} - Outgoing Coherence")
    # axes[0].set_xlabel("Node Level")
    # axes[0].set_ylabel("Outgoing Coherence")
    #
    # # Incoming Coherence boxplot
    # sns.boxplot(
    #     data=inout_coh,
    #     x="NodeLevel",
    #     y="MeanAbsInCoh",
    #     hue="NodeLevel",
    #     order=node_order,
    #     palette={"Input": "#31a354", "Middle": "#fd8d3c", "Output": "#6baed6"},
    #     ax=axes[1],
    # )
    # # axes[1].set_title(f"{Path(cm).stem} - Incoming Coherence")
    # axes[1].set_xlabel("Node Level")
    # axes[1].set_ylabel("Incoming Coherence")
    #
    # fig.suptitle(f"{Path(cm).stem} - Incoming Coherence")
    #
    plt.tight_layout()

    # Save the combined figure
    save_path = boxplt_dir / f"{Path(cm).stem}_InOutCohBox.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    # --- 4. Heatmap of connection density based on NodeLevel ---
    topos_dir = Path("../../AbasyTOPOS")
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
        # print(topo_df)
        # print(topo_df.columns)

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

        # heatmap_counts = pd.crosstab(
        #     topo_df["SourceLevel"],
        #     topo_df["TargetLevel"],
        #     rownames=["Source"],
        #     colnames=["Target"],
        # ).reindex(index=node_order, columns=node_order, fill_value=0)
        #
        # # Optionally normalize by total edges to get density
        # heatmap_density = heatmap_counts / heatmap_counts.sum().sum()
        # print(heatmap_counts)

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

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            log_density,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar_kws={"label": "Log10 Density"},
        )
        plt.title(f"{Path(cm).stem}")
        plt.tight_layout()
        # plt.show()
        plt.savefig(heiheat_dir / f"{Path(cm).stem}_NodeLevelHeatmapLog.png", dpi=300)
        plt.close()


# --- Bar plot across networks for NodeLevel percentages ---
dist_df = pd.DataFrame(all_distributions).fillna(0)
mean_vals = dist_df.mean().reindex(["Input", "Middle", "Output"])
std_vals = dist_df.std().reindex(["Input", "Middle", "Output"])

plt.figure(figsize=(4, 7))
bars = plt.bar(
    mean_vals.index,
    mean_vals.values,
    yerr=std_vals.values,
    capsize=5,
    edgecolor="black",
)
plt.ylabel("Percentage of Nodes (%)")
plt.xlabel("Node Level")
plt.title("Average Node Level Distribution Across Networks")
plt.ylim(0, max(mean_vals.values + std_vals.values) * 1.2)

# Annotate bars
for bar, mean, std in zip(bars, mean_vals.values, std_vals.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 8,
        f"{mean:.1f} ± {std:.1f}%",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig(plots_dir / "NodeLevelDist.png", dpi=300)
plt.close()
# plt.show()


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

fig, ax = plt.subplots(figsize=(8, 8))

sns.boxplot(
    data=filtered_df,
    x="Connection Density",
    y="Density",
    linewidth=1.2,
    width=0.4,
    order=order,  # <--- ORDER HERE
    ax=ax,
)

sns.stripplot(
    data=filtered_df,
    x="Connection Density",
    y="Density",
    order=order,
    ax=ax,
    color="black",  # or let seaborn choose a default
    size=8,
    jitter=True,
    alpha=0.7,  # transparency so boxes remain visible
)

plt.xticks(rotation=0, ha="center")
plt.tight_layout()
plt.savefig(plots_dir / "LevelContactBox.png", dpi=300)
# plt.show()
plt.close()


# PLotting the TF enrichment figure
tf_enrichments = pd.DataFrame(tf_enrichments)
print(tf_enrichments)
print(tf_enrichments.columns)
# Sort for nicer plotting
df = tf_enrichments.sort_values(["NodeLevel", "Odds_ratio"], ascending=[True, False])

# Create significance column
df["P_value_FDR"] = multipletests(df["P_value"], method="fdr_bh")[1]
df["Significant FDR"] = df["P_value_FDR"] < 0.05

# Palette for significance (True = sig)
palette = {True: "black", False: "white"}

plt.figure(figsize=(6, 8))

# --- Boxplot first ---
sns.boxplot(
    data=df,
    x="NodeLevel",
    y="Odds_ratio",
    showcaps=True,
    width=0.6,
    # boxprops={"facecolor": "white", "edgecolor": "black"},
    # medianprops={"color": "black"},
    # whiskerprops={"color": "black"},
)

# --- Overlay stripplot ---
sns.stripplot(
    data=df,
    x="NodeLevel",
    y="Odds_ratio",
    hue="Significant FDR",
    palette=palette,
    dodge=False,
    jitter=True,
    alpha=0.8,
    edgecolor="black",  # outline for hollow markers
    linewidth=1,
    size=7,
    marker="o",
)

plt.yscale("log")
plt.ylabel("Odds Ratio (log scale)")
plt.xlabel("Node Level")
plt.title("TF Enrichment Across Node Levels")

# Fix legend for significance
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(
    handles,
    ["p >= 0.05", "p < 0.05"],
    title="Enrichment\nSignificance FDR",
    loc="upper right",
)

plt.tight_layout()
plt.savefig(plots_dir / "TF_Enrichment.png", dpi=300)
plt.close()
# plt.show()

# # log2 OR (handle OR=0)
# df["log2_OR"] = np.log2(df["Odds_ratio"].replace(0, np.nan))
#
# # significance label
# df["sig"] = df["P_value"].apply(lambda p: "*" if p < 0.05 else "ns")
#
# # pivot for heatmap
# heat = df.pivot(index="NodeLevel", columns="Network", values="log2_OR")
#
# # annotation matrix
# anno = df.pivot(index="NodeLevel", columns="Network", values="sig")
#
# # Format annotation: log2(OR) + newline + sig
# combined_anno = df.apply(
#     lambda r: f"{r['log2_OR']:.2f}\n({r['sig']})", axis=1
# ).values.reshape(heat.shape)
#
# # ---------------------------------------------------------
# # Plot heatmap
# # ---------------------------------------------------------
#
# plt.figure(figsize=(1.2 * heat.shape[1], 4))
#
# ax = sns.heatmap(
#     heat,
#     cmap="coolwarm",
#     center=0,
#     annot=combined_anno,
#     fmt="",
#     linewidths=0.5,
#     cbar_kws={"label": "log2(Odds Ratio)"},
# )
#
# ax.set_xlabel("Network")
# ax.set_ylabel("Node Level")
# ax.set_title("TF Enrichment Across Node Levels and Networks")
#
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()
