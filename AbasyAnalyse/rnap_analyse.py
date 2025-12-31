import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy.stats import fligner
from statannotations.Annotator import Annotator
from itertools import combinations
from scipy import stats

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


def get_separator(header_line):
    # List the candidates you want to check
    potential_seps = [",", ";", "\t", "|"]
    # Create a dictionary counting occurrences of each: {',': 2, ';': 0, ...}
    counts = {sep: header_line.count(sep) for sep in potential_seps}
    # Find the key with the highest value
    best_sep = max(counts, key=counts.get)
    return best_sep


# Specifying the data directory
data_dir = Path("./lf2c_rnap/")

# Plotting folder
plot_dir = data_dir / "Plots"
plot_dir.mkdir(exist_ok=True)

# Specifyingthe regulondb associated results
regdb_dir = Path("./AbasyCohResults/511145_v2022_sRDB22_eStrong_regNetwork_Strong/")
# Load up the regulondb tsv
reg_df = pd.read_csv(
    regdb_dir / "511145_v2022_sRDB22_eStrong_GOInformation.tsv", sep="\t"
)
print(reg_df)
print(reg_df.columns)

print(list(data_dir.glob("*.csv")))

for fil in data_dir.glob("*.csv"):
    with open(fil) as f:
        # Finding the seperator
        sep = get_separator(f.readline())
    # print(fil.name)
    expdf = pd.read_csv(fil, sep=sep)
    if "Unnamed: 0" in expdf.columns:
        expdf = expdf.drop(columns="Unnamed: 0")

    pval_col = "padj" if "padj" in expdf.columns else "pvalue"

    # if "padj" in expdf:
    #     expdf = expdf[expdf["padj"] <= 0.05]
    #     # print(expdf[expdf["padj"] <= 0.05])
    # elif "pvalue" in expdf:
    #     expdf = expdf[expdf["pvalue"] <= 0.05]
    #     # print(expdf[expdf["pvalue"] <= 0.05])
    # # print(expdf.columns)
    # # print(expdf)

    # Heck if this is time data set or not
    if "min" not in fil.name and "X" in fil.name:
        # Label the differentially epxressed genes with the condition column
        condition_name = str(fil.name).split("_DE_")[1].replace(".csv", "")
        reg_df[condition_name] = reg_df["Node"].isin(expdf["gname"])
        # mapping the expression values
        lfc_map = expdf.set_index("gname")["log2FoldChange"]
        reg_df[f"LF2C_{condition_name}"] = reg_df["Node"].map(lfc_map)
        # Mappingthe pvalue
        pval_map = expdf.set_index("gname")[pval_col]
        reg_df[f"Pval_{condition_name}"] = reg_df["Node"].map(pval_map)
        # Adding a column for pvalue singificance
        sig_genes = expdf[expdf[pval_col] <= 0.05]["gname"]
        reg_df[condition_name] = reg_df["Node"].isin(sig_genes)

        # Merge the two on teh Node and gname
        expdf = pd.merge(expdf, reg_df, left_on="gname", right_on="Node", how="inner")
        print(expdf)
        print(expdf.columns)

        # 1. Histograms (Out Coherence)
        plt.figure(figsize=(6, 5))
        sns.histplot(
            data=reg_df,
            x="MeanAbsOutCoh",
            hue=condition_name,
            palette=NORD_PALETTE,  # Nord colors
            edgecolor=NORD_COLORS["dark"],
            multiple="stack",
        )
        plt.title(f"{condition_name} - Out Coherence", pad=10)
        plt.savefig(plot_dir / f"{condition_name}_MeanAbsOutCoh_Hist.png")
        plt.close()

        # 2. Histograms (In Coherence)
        plt.figure(figsize=(6, 5))
        sns.histplot(
            data=reg_df,
            x="MeanAbsInCoh",
            hue=condition_name,
            palette=NORD_PALETTE,
            edgecolor=NORD_COLORS["dark"],
            multiple="stack",
        )
        plt.title(f"{condition_name} - In Coherence", pad=10)
        plt.savefig(plot_dir / f"{condition_name}_MeanAbsInCoh_Hist.png")
        plt.close()

        # 3. Boxplot (In Coherence vs Node Level)
        plt.figure(figsize=(8, 6))
        sns.boxplot(
            data=reg_df,
            y="MeanAbsInCoh",
            x="NodeLevel",
            palette=NORD_PALETTE,  # Consistent Nord categorical colors
            boxprops=dict(edgecolor=NORD_COLORS["dark"], linewidth=1.5),
            whiskerprops=dict(color=NORD_COLORS["dark"], linewidth=1.5),
            capprops=dict(color=NORD_COLORS["dark"], linewidth=1.5),
            medianprops=dict(color=NORD_COLORS["dark"], linewidth=2),
        )
        plt.savefig(plot_dir / f"{condition_name}_MeanAbsInCoh_NodeLvl_Box.png")
        plt.close()

        # 4. Boxplot (Out Coherence vs Node Level)
        plt.figure(figsize=(8, 6))
        sns.boxplot(
            data=reg_df,
            y="MeanAbsOutCoh",
            x="NodeLevel",
            palette=NORD_PALETTE,
            boxprops=dict(edgecolor=NORD_COLORS["dark"], linewidth=1.5),
            whiskerprops=dict(color=NORD_COLORS["dark"], linewidth=1.5),
            capprops=dict(color=NORD_COLORS["dark"], linewidth=1.5),
            medianprops=dict(color=NORD_COLORS["dark"], linewidth=2),
        )
        plt.savefig(plot_dir / f"{condition_name}_MeanAbsOutCoh_NodeLvl_Box.png")
        plt.close()

        # 5. Barplot (Percentage of DE Genes)
        perc_df = (
            reg_df.groupby("NodeLevel")[condition_name]
            .value_counts(normalize=True)
            .mul(100)
            .rename("Percentage")
            .reset_index()
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=perc_df,
            x="NodeLevel",
            y="Percentage",
            hue=condition_name,
            palette=NORD_PALETTE,
            edgecolor=NORD_COLORS["dark"],
            linewidth=1.5,
        )
        plt.title(f"Percentage of DE Genes by Node Level ({condition_name})")
        plt.ylabel("Percentage (%)")
        plt.legend(frameon=False)
        plt.savefig(plot_dir / f"{condition_name}_NodeLvlPercent.png")
        plt.close()

        # 6. Scatterplots (Coherence vs LFC)
        # Using a subset of Nord colors for 'hue' (Is_TF_GO: True/False)
        # tf_palette = {True: NORD_COLORS["red"], False: NORD_COLORS["blue"]}
        tf_palette = {
            True: NORD_COLORS["red"],
            False: NORD_COLORS["blue"],
            "True": NORD_COLORS["red"],  # Add string key
            "False": NORD_COLORS["blue"],  # Add string key
        }

        plt.figure(figsize=(6, 6))
        sns.scatterplot(
            data=expdf,
            y="MeanAbsInCoh",
            x="log2FoldChange",
            hue="Is_TF_GO",
            palette=tf_palette,
            edgecolor=NORD_COLORS["dark"],
            alpha=0.8,
        )
        plt.savefig(plot_dir / f"{condition_name}_MeanAbsInCoh_LFC_TF.png")
        plt.close()

        plt.figure(figsize=(6, 6))
        sns.scatterplot(
            data=expdf,
            y="MeanAbsOutCoh",
            x="log2FoldChange",
            hue="Is_TF_GO",
            palette=tf_palette,
            edgecolor=NORD_COLORS["dark"],
            alpha=0.8,
        )
        plt.savefig(plot_dir / f"{condition_name}_MeanAbsOutCoh_LFC_TF.png")
        plt.close()

        # 7. Boxplot (LFC vs TF status)
        plt.figure(figsize=(6, 6))
        sns.boxplot(
            data=expdf,
            y="log2FoldChange",
            x="Is_TF_GO",
            palette=tf_palette,
            boxprops=dict(edgecolor=NORD_COLORS["dark"], linewidth=1.5),
            whiskerprops=dict(color=NORD_COLORS["dark"], linewidth=1.5),
            capprops=dict(color=NORD_COLORS["dark"], linewidth=1.5),
            medianprops=dict(color=NORD_COLORS["dark"], linewidth=2),
        )
        plt.savefig(plot_dir / f"{condition_name}_LFC_TF.png")
        plt.close()

        # # Plotting the in coherence of the diff expressed genes
        # sns.histplot(data=reg_df, x="MeanAbsOutCoh", hue=condition_name)
        # plt.savefig(plot_dir / f"{condition_name}_MeanAbsOutCoh_Hist.png")
        # plt.close()
        # sns.histplot(data=reg_df, x="MeanAbsInCoh", hue=condition_name)
        # plt.savefig(plot_dir / f"{condition_name}_MeanAbsInCoh_Hist.png")
        # plt.close()
        # sns.boxplot(data=reg_df, y="MeanAbsInCoh", x="NodeLevel")
        # plt.savefig(plot_dir / f"{condition_name}_MeanAbsInCoh_NodeLvl_Box.png")
        # plt.close()
        # sns.boxplot(data=reg_df, y="MeanAbsOutCoh", x="NodeLevel")
        # plt.savefig(plot_dir / f"{condition_name}_MeanAbsOutCoh_NodeLvl_Box.png")
        # plt.close()
        # # Plotting the percentage of node level differntially epxressed
        # perc_df = (
        #     reg_df.groupby("NodeLevel")[condition_name]
        #     .value_counts(normalize=True)
        #     .mul(100)
        #     .rename("Percentage")
        #     .reset_index()
        # )
        # # 2. Plotting
        # plt.figure(figsize=(10, 6))  # Optional: make the plot a bit larger
        # sns.barplot(data=perc_df, x="NodeLevel", y="Percentage", hue=condition_name)
        # plt.title(f"Percentage of DE Genes by Node Level ({condition_name})")
        # plt.ylabel("Percentage (%)")
        # plt.savefig(plot_dir / f"{condition_name}_NodeLvlPercent.png")
        # plt.close()
        #
        # sns.scatterplot(
        #     data=expdf, y="MeanAbsInCoh", x="log2FoldChange", hue="Is_TF_GO"
        # )
        # plt.savefig(plot_dir / f"{condition_name}_MeanAbsInCoh_LFC_TF.png")
        # plt.close()
        # sns.scatterplot(
        #     data=expdf, y="MeanAbsOutCoh", x="log2FoldChange", hue="Is_TF_GO"
        # )
        # plt.savefig(plot_dir / f"{condition_name}_MeanAbsOutCoh_LFC_TF.png")
        # plt.close()
        # sns.boxplot(data=expdf, y="log2FoldChange", x="Is_TF_GO")
        # plt.savefig(plot_dir / f"{condition_name}_LFC_TF.png")
        # plt.close()
        # # plt.show()

    else:
        continue

    # break


# Plotting on the merged data
print(reg_df.columns)

# Mapping of dilution value to conditions
conditions = {"0.75X_vs_1.0X": 0.75, "0.5X_vs_1.0X": 0.50, "0.25X_vs_1.0X": 0.25}

plot_long = []

for cond_name, dosage in conditions.items():
    lfc_col = f"LF2C_{cond_name}"
    pval_col = f"Pval_{cond_name}"
    # Filteringthe significant genes
    mask = reg_df[pval_col] <= 0.05
    subset = reg_df.loc[mask].copy()
    # Settingthe condition name and dosage
    subset["Condition"] = cond_name
    subset["Dosage"] = dosage
    subset["LFC"] = subset[lfc_col]
    # Keep the static network metrics
    cols_to_keep = [
        "Node",
        "MeanAbsOutCoh",
        "MeanAbsInCoh",
        "NodeLevel",
        "Is_TF_GO",
        "Condition",
        "Dosage",
        "LFC",
    ]
    plot_long.append(subset[cols_to_keep])

# COncat into a single dataframe
plot_long = pd.concat(plot_long)
# Getting the Abssolute LFC
plot_long["AbsLFC"] = np.abs(plot_long["LFC"])

print(plot_long)
print(plot_long.columns)

g = sns.lmplot(
    data=plot_long,
    x="MeanAbsInCoh",
    y="LFC",
    hue="Condition",
    palette=NORD_PALETTE,  # Swapped viridis_r for Nord
    height=6,
    aspect=1.5,
    scatter_kws={"alpha": 0.4, "s": 15, "edgecolor": "none"},  # Clean scatter
    line_kws={"linewidth": 2.5},
)
plt.title("Does High Input Coherence Correlate with LFC Magnitude?")
plt.xlabel("Input Coherence (MeanAbsInCoh)")
plt.ylabel("Log2 Fold Change")
plt.savefig(plot_dir / "AbsInCoh_LFC_Corr.png")
plt.close()

# 2. Lmplot (In Coh vs AbsLFC)
# g = sns.lmplot(
#     data=plot_long,
#     x="MeanAbsInCoh",
#     y="AbsLFC",
#     hue="Condition",
#     palette=NORD_PALETTE,
#     height=6,
#     aspect=1.5,
#     scatter_kws={"alpha": 0.8, "s": 50},
#     line_kws={"linewidth": 2.5},
# )
# plt.xlabel("Absolute Incoming Coherence")
# plt.ylabel("Aabsolute $log_10$ Fold Change")
# plt.savefig(plot_dir / "AbsInCoh_AbsLFC_Corr.png")
# plt.close()

hue_order = sorted(plot_long["Condition"].unique())

new_labels = []
for cond in hue_order:
    subset = plot_long[plot_long["Condition"] == cond]
    # CLEANING: Drop rows where x or y is NaN or Infinite just for the math
    valid_data = (
        subset[["MeanAbsInCoh", "AbsLFC"]].replace([np.inf, -np.inf], np.nan).dropna()
    )
    if len(valid_data) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid_data["MeanAbsInCoh"], valid_data["AbsLFC"]
        )
        # Determine stars based on p-value
        if p_value < 0.001:
            stars = "***"
        elif p_value < 0.01:
            stars = "**"
        elif p_value < 0.05:
            stars = "*"
        else:
            stars = ""
        label = f"{cond.replace('_', ' ').replace('X', 'x')}\n($r={r_value:.2f}, p-value={p_value:.2f})"
    else:
        label = f"{cond} (N/A)"
    new_labels.append(label)

fig_w, fig_h = 7.5, 7

g = sns.lmplot(
    data=plot_long,
    x="MeanAbsInCoh",
    y="AbsLFC",
    hue="Condition",
    hue_order=hue_order,  # Vital: ensures colors match our new labels
    palette=NORD_PALETTE,
    height=fig_h,
    aspect=fig_w / fig_h,
    legend=False,  # We build a custom one below
    scatter_kws={
        "alpha": 0.8,
        "s": 50,
    },
    # line_kws={"linewidth": 2.5},
)

ax = g.ax
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color(NORD_COLORS["dark"])
    spine.set_linewidth(2.0)

ax.set_xlabel("Absolute Incoming Coherence")
ax.set_ylabel(r"Absolute $\log_{10}$ Fold Change")

handles, _ = ax.get_legend_handles_labels()

if len(handles) > len(hue_order):
    handles = handles[: len(hue_order)]

plt.legend(
    handles=handles,
    labels=new_labels,
    title="Condition",
    bbox_to_anchor=(0.5, -0.2),
    loc="upper center",
    ncol=3,
    borderaxespad=0,
    frameon=True,
)

plt.tight_layout()
plt.savefig(plot_dir / "AbsInCoh_AbsLFC_Corr.png", bbox_inches="tight")
plt.savefig(
    plot_dir / "AbsInCoh_AbsLFC_Corr.svg", bbox_inches="tight", transparent=True
)
plt.close()

# fig_w, fig_h = 7.5, 5
#
# # 2. Lmplot
# g = sns.lmplot(
#     data=plot_long,
#     x="MeanAbsInCoh",
#     y="AbsLFC",
#     hue="Condition",
#     palette=NORD_PALETTE,
#     height=fig_h,
#     aspect=fig_w / fig_h,
#     legend=False,  # We will manually add the legend to match previous style if needed
#     scatter_kws={
#         "alpha": 0.5,
#         "s": 20,
#     },
#     line_kws={"linewidth": 2.5},
# )
#
# # 3. Styling the Spines (The "Box")
# # g.ax gives you the single matplotlib axes object inside the FacetGrid
# ax = g.ax
#
# # Force the "Box" appearance (Top and Right lines)
# for spine in ax.spines.values():
#     spine.set_visible(True)  # Make sure top/right are shown
#     spine.set_color(NORD_COLORS["dark"])  # Color them
#     spine.set_linewidth(2.0)  # Thicken them
#
# # 4. Labels and Title
# ax.set_xlabel("Absolute Incoming Coherence")
# ax.set_ylabel(r"Absolute $\log_{10}$ Fold Change")  # Fixed LaTeX syntax
# ax.set_title(
#     "Correlation: InCoh vs LFC", pad=15, color=NORD_COLORS["dark"], fontweight="bold"
# )
#
# # 5. Legend
# # Recreating the legend manually to match the specific position of your previous plot
# plt.legend(
#     title="Condition",
#     bbox_to_anchor=(1.02, 1),
#     loc="upper left",
#     borderaxespad=0,
#     frameon=True,
# )
#
# plt.tight_layout()
# plt.savefig(
#     plot_dir / "AbsInCoh_AbsLFC_Corr.png", bbox_inches="tight"
# )  # bbox_inches prevents legend clipping
# plt.close()


# 3. Lmplot (Out Coh vs LFC)
g = sns.lmplot(
    data=plot_long,
    x="MeanAbsOutCoh",
    y="LFC",
    hue="Condition",
    palette=NORD_PALETTE,
    height=6,
    aspect=1.5,
    scatter_kws={"alpha": 0.4, "s": 15, "edgecolor": "none"},
    line_kws={"linewidth": 2.5},
)
plt.title("Does High Input Coherence Correlate with LFC Magnitude?")
plt.xlabel("Input Coherence (MeanAbsOutCoh)")
plt.ylabel("Log2 Fold Change")
plt.savefig(plot_dir / "AbsOutCoh_LFC_Corr.png")
plt.close()

# 4. Lmplot (Dilution Trajectory)
# Using 'palette' instead of 'viridis_r'
g = sns.lmplot(
    data=plot_long,
    x="MeanAbsOutCoh",
    y="AbsLFC",
    hue="Condition",
    palette=NORD_PALETTE,
    height=6,
)

plt.gca().invert_xaxis()
plt.title("Expression Trajectory as Concentration Decreases")
plt.ylabel("Log2 Fold Change")
plt.xlabel("Concentration (Dosage)")
plt.savefig(plot_dir / "TF_Dilution.png")
plt.close()


nord_diverging = LinearSegmentedColormap.from_list(
    "nord_div",
    [NORD_COLORS["blue"], "white", NORD_COLORS["red"]],  # Blue -> White -> Red
)

cols = ["LF2C_0.75X_vs_1.0X", "LF2C_0.5X_vs_1.0X", "LF2C_0.25X_vs_1.0X"]
heatmap_data = reg_df.set_index("Node")[cols].dropna()

# 5. Clustermap
sns.clustermap(
    heatmap_data,
    cmap=nord_diverging,  # Use custom Nord map
    center=0,
    z_score=None,
    col_cluster=False,
    figsize=(10, 12),
    yticklabels=False,
    linewidths=0.0,  # Smooth look
    tree_kws=dict(linewidths=1.5, colors=NORD_COLORS["dark"]),  # Dark dendrogram lines
)
plt.savefig(plot_dir / "Heatmap_DiffGeneexpression.png")
plt.close()

# 1. Define Orders and Colors
condition_order = ["0.25X_vs_1.0X", "0.5X_vs_1.0X", "0.75X_vs_1.0X"]
node_order = ["Input", "Middle", "Output"]

# Palette for Node Level (Red/Blue/Green)
palette_node_level = {
    "Input": NORD_COLORS["red"],
    "Middle": NORD_COLORS["blue"],
    "Output": NORD_COLORS["green"],
}
# Palette for Conditions (Blue/Orange/Green)
palette_conditions = [NORD_COLORS["blue"], "#d08770", NORD_COLORS["green"]]

# ==========================================
# PLOT A: Condition vs LFC (Grouped by NodeLevel)
# ==========================================
fig, ax = plt.subplots(figsize=(12, 7))

# 1. Plot Boxplot
sns.boxplot(
    data=plot_long,
    x="Condition",
    y="LFC",
    hue="NodeLevel",
    order=condition_order,
    hue_order=node_order,
    palette=palette_node_level,
    boxprops=dict(alpha=0.8, edgecolor=NORD_COLORS["dark"]),
    whiskerprops=dict(color=NORD_COLORS["dark"]),
    capprops=dict(color=NORD_COLORS["dark"]),
    medianprops=dict(color=NORD_COLORS["dark"], linewidth=1.5),
    showfliers=False,
    ax=ax,
)

# 2. Add Stats (Input vs Middle vs Output WITHIN each Condition)
pairs_plot_a = []
for cond in condition_order:
    pairs_plot_a.append(((cond, "Input"), (cond, "Middle")))
    pairs_plot_a.append(((cond, "Middle"), (cond, "Output")))
    pairs_plot_a.append(((cond, "Input"), (cond, "Output")))

try:
    annotator = Annotator(
        ax,
        pairs_plot_a,
        data=plot_long,
        x="Condition",
        y="LFC",
        hue="NodeLevel",
        order=condition_order,
        hue_order=node_order,
    )
    annotator.configure(
        test="Levene",
        text_format="star",
        loc="inside",
        color=NORD_COLORS["dark"],
        line_width=1.5,
        verbose=False,
    )
    annotator.apply_and_annotate()
except Exception as e:
    print(f"Stats failed for Plot A: {e}")

# 3. Styling
plt.axhline(0, color=NORD_COLORS["dark"], linestyle="--", linewidth=2)
plt.title("Variance Differences: Input vs Middle vs Output (Levene Test)", pad=15)
plt.ylabel("Log2 Fold Change")
plt.xlabel("Condition")
plt.legend(
    title="Node Level", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left"
)

# Spines
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color(NORD_COLORS["dark"])
    spine.set_linewidth(2.0)

plt.tight_layout()
plt.savefig(plot_dir / "ConditionNodeLvl_LFC_Box.png", dpi=300)
plt.close()


# ==========================================
# PLOT B: NodeLevel vs LFC (Grouped by Condition)
# ==========================================
fig, ax = plt.subplots(figsize=(12, 7))

# 1. Plot Boxplot
sns.boxplot(
    data=plot_long,
    x="NodeLevel",
    y="LFC",
    hue="Condition",
    order=node_order,
    hue_order=condition_order,
    palette=palette_conditions,
    boxprops=dict(alpha=0.8, edgecolor=NORD_COLORS["dark"]),
    whiskerprops=dict(color=NORD_COLORS["dark"]),
    capprops=dict(color=NORD_COLORS["dark"]),
    medianprops=dict(color=NORD_COLORS["dark"], linewidth=1.5),
    showfliers=False,
    ax=ax,
)

# 2. Add Stats (Conditions compared WITHIN each NodeLevel)
pairs_plot_b = []
for node in node_order:
    pairs_plot_b.append(((node, "0.25X_vs_1.0X"), (node, "0.5X_vs_1.0X")))
    pairs_plot_b.append(((node, "0.5X_vs_1.0X"), (node, "0.75X_vs_1.0X")))
    pairs_plot_b.append(((node, "0.25X_vs_1.0X"), (node, "0.75X_vs_1.0X")))

try:
    annotator = Annotator(
        ax,
        pairs_plot_b,
        data=plot_long,
        x="NodeLevel",
        y="LFC",
        hue="Condition",
        order=node_order,
        hue_order=condition_order,
    )
    annotator.configure(
        test="Levene",
        text_format="star",
        loc="inside",
        color=NORD_COLORS["dark"],
        line_width=1.5,
        verbose=False,
    )
    annotator.apply_and_annotate()
except Exception as e:
    print(f"Stats failed for Plot B: {e}")

# 3. Styling
plt.axhline(0, color=NORD_COLORS["dark"], linestyle="--", linewidth=2)
plt.title("Gene Expression Variance: Levene Test", pad=15)
plt.ylabel("Log2 Fold Change")
plt.xlabel("Node Level")
plt.legend(title="Condition", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

# Spines
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color(NORD_COLORS["dark"])
    spine.set_linewidth(2.0)

plt.tight_layout()
plt.savefig(plot_dir / "NodeLvlCondition_LFC_Box.png", dpi=300)
plt.close()
