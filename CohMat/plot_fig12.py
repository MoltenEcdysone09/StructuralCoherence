import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import collections
import os
from scipy.signal import savgol_filter
from matplotlib import cm, colors
from matplotlib.patches import Patch
from scipy.stats import pearsonr, linregress
import matplotlib as mpl
from statannotations.Annotator import Annotator
import itertools
from pathlib import Path
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
from scipy import stats
import math

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
            "font.size": 18,
            # "font.weight": 450,
            # 2. Text Colors
            "text.color": NORD_COLORS["dark"],
            "axes.labelcolor": NORD_COLORS["dark"],
            "axes.titlecolor": NORD_COLORS["dark"],
            "axes.labelsize": 20,
            # "axes.labelweight": 500,
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
            # Latex Formatting
            "mathtext.default": "regular",
        }
    )


# --- APPLY THE SETTINGS ---
set_global_nord_style()

###################################################################################################
###################################################################################################


###################################################################################################
### Pre Processing
###################################################################################################

# Creating the PLots Folder
fig_dir = Path("../../FigPlots")
fig_dir.mkdir(exist_ok=True)

# CohRes file
cohres_df = pd.read_parquet("../../../ArtNetCohResultsERHI.parquet")
print(cohres_df.columns)
print(cohres_df.dtypes)

# Replacing the TN within basenet
cohres_df["BaseNet"] = cohres_df["BaseNet"].str.replace("TN", "")

# Finding the base newtorks
base_df = cohres_df[
    (cohres_df["NumNodesPerGroup"] == 10)
    & (cohres_df["Density"] == 100)
    & (cohres_df["NetType"] == "ER")
]
print(base_df)

# # Some newtorks may be missing nodes because they are
# # not completely connecnted and nodes will be ignored due to pandas ops
# print(cohres_df["NumNodes"].value_counts())
# cohres_df = cohres_df[cohres_df["NumNodes"] % 10 == 0]
# print(cohres_df["NumNodes"].value_counts())
# cohres_df["NumNodes"] = cohres_df["NumNodes"].round(-1)
# print(cohres_df["NumNodes"].value_counts())

# Getting the base net and numgroups dict
basenet_numgroups_dict = dict(zip(base_df["BaseNet"], base_df["NumGroups"]))
print(basenet_numgroups_dict)

# Normalising the number of groups by the num groups in base net
# cohres_df["FoldChangeNumTeams"] = (
#     cohres_df["NumGroups"] - cohres_df["BaseNet"].map(basenet_numgroups_dict)
# ) / (cohres_df["NumNodes"])
cohres_df["FoldChangeNumTeams"] = cohres_df["NumGroups"] / cohres_df["BaseNet"].map(
    basenet_numgroups_dict
)
cohres_df["MinMaxFoldChangeNumTeams"] = cohres_df.groupby("NumNodes")[
    "FoldChangeNumTeams"
].transform(lambda x: (x - x.min()) / (x.max() - x.min()))


# Binning the WalkVals to get smoother data
cohres_df["WalkValBin"] = np.floor(cohres_df["AbsMeanWalkVal"])

# rename the connectivity term
cohres_df = cohres_df.rename(columns={"AbsMeanWalkVal": "NormMeanComm"})

# Min-Max normlaisation for the Mean Walk Walk
cohres_df["MinMaxCommunicability"] = cohres_df.groupby("NumNodes")[
    "NormMeanComm"
].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

cohres_df["MinMaxCommunicabilityBin"] = cohres_df["MinMaxCommunicability"].round(1)

# Create a scale Column
cohres_df["Scale"] = cohres_df["NumNodesPerGroup"].astype(str) + "x"

print(cohres_df.columns)
print(cohres_df.dtypes)

print(
    cohres_df[
        (cohres_df["Scale"] == "10x")
        & (cohres_df["FoldChangeNumTeams"] > 1)
        & (cohres_df["BaseNet"] == "2_2_0TN")
    ]
)

###################################################################################################
###################################################################################################


###################################################################################################
###  MeanCoherenceValue / FoldChangeNumTeams Vs MeanCommunicability
###################################################################################################


def create_stacked_scatterplots_comm(df, basenet_dict, save_path):
    """
    Generates vertically stacked scatterplots.
    - Points have transparent fill (alpha=0.7)
    - Points have transparent white edges
    - Smaller points (s=70)
    - No titles
    - Legend outside
    """
    dark_color = NORD_COLORS["dark"]
    # Define transparent white (R, G, B, Alpha)
    transparent_white_edge = (1, 1, 1, 0.1)
    # transparent_white_edge = (0.26, 0.30, 0.37, 0.2)

    for bn in basenet_dict.keys():
        print(f"Plotting for BaseNet: {bn}")

        # Subset Data
        bn_cdf = df[(df["BaseNet"] == bn) & (df["NetType"] == "ER")]

        if bn_cdf.empty:
            continue

        # Initialize Figure (2 Rows, 1 Column)
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 9.65), sharex=False)

        # --- Plot 1 (Top): Communicability vs FoldChangeNumTeams ---
        sns.scatterplot(
            data=bn_cdf,
            x="NormMeanComm",
            y="FoldChangeNumTeams",
            # hue="NumNodesPerGroup",
            hue="Scale",
            palette=NORD_PALETTE,
            s=100,  # Smaller size
            alpha=0.8,  # Transparent Fill
            edgecolor=transparent_white_edge,  # Transparent White Edge
            # linewidth=0
            linewidth=0.1,  # Thin edge line
            ax=axes[0],
        )
        axes[0].set_ylabel("Fold Change (Team Count)")
        axes[0].set_xlabel(r"$\log_{10}$(Mean Normalized Communicability)")
        axes[0].set_ylim(-1, 65)

        # --- Plot 2 (Bottom): Communicability vs MeanCoh ---
        sns.scatterplot(
            data=bn_cdf,
            x="NormMeanComm",
            y="MeanCoh",
            # hue="NumNodesPerGroup",
            hue="Scale",
            palette=NORD_PALETTE,
            s=100,  # Smaller size
            alpha=0.8,  # Transparent Fill
            edgecolor=transparent_white_edge,  # Transparent White Edge
            linewidth=0.1,  # Thin edge line
            ax=axes[1],
        )
        axes[1].set_ylabel("Structural Coherence")
        axes[1].set_ylim(-0.75, 1.15)
        axes[1].set_xlabel(r"$\log_{10}$(Mean Normalized Communicability)")

        # --- Legend Customization ---
        axes[0].legend_.remove()
        axes[1].legend_.remove()

        handles, labels = axes[0].get_legend_handles_labels()

        # Add legend outside top plot
        leg = axes[0].legend(
            handles,
            labels,
            title="Scale",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            edgecolor=dark_color,
        )

        # Thicken the legend border
        leg.get_frame().set_linewidth(2.5)

        fig.suptitle(bn, y=0.98, fontsize=18, fontweight=500)

        plt.tight_layout()
        # plt.show()
        plt.savefig(save_path / f"{bn}_meancomm.png", dpi=300)
        plt.savefig(save_path / f"{bn}_meancomm.svg", dpi=300, transparent=True)
        plt.close()


# --- Run ---
save_path = fig_dir / "Fig1"
save_path.mkdir(exist_ok=True)
create_stacked_scatterplots_comm(cohres_df, basenet_numgroups_dict, save_path)

###################################################################################################
###################################################################################################

###################################################################################################
###  MeanCoherenceValue / FoldChangeNumTeams Vs Density
###################################################################################################


def create_stacked_scatterplots_density(df, basenet_dict, save_path):
    """
    Generates vertically stacked scatterplots.
    - Points have transparent fill (alpha=0.7)
    - Points have transparent white edges
    - Smaller points (s=70)
    - No titles
    - Legend outside
    """
    dark_color = NORD_COLORS["dark"]
    # Define transparent white (R, G, B, Alpha)
    transparent_white_edge = (1, 1, 1, 0.1)
    # transparent_white_edge = (0.26, 0.30, 0.37, 0.2)

    for bn in basenet_dict.keys():
        print(f"Plotting for BaseNet: {bn}")

        # Subset Data
        bn_cdf = df[(df["BaseNet"] == bn) & (df["NetType"] == "ER")]

        if bn_cdf.empty:
            continue

        # Initialize Figure (2 Rows, 1 Column)
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 9), sharex=True)

        # --- Plot 1 (Top): Communicability vs FoldChangeNumTeams ---
        sns.scatterplot(
            data=bn_cdf,
            x="Density",
            y="FoldChangeNumTeams",
            # hue="NumNodesPerGroup",
            hue="Scale",
            palette=NORD_PALETTE,
            s=100,  # Smaller size
            alpha=0.8,  # Transparent Fill
            edgecolor=transparent_white_edge,  # Transparent White Edge
            linewidth=0.1,  # Thin edge line
            ax=axes[0],
        )
        axes[0].set_ylabel("Fold Change (Team Count)")

        # --- Plot 2 (Bottom): Communicability vs MeanCoh ---
        sns.scatterplot(
            data=bn_cdf,
            x="Density",
            y="MeanCoh",
            # hue="NumNodesPerGroup",
            hue="Scale",
            palette=NORD_PALETTE,
            s=100,  # Smaller size
            alpha=0.8,  # Transparent Fill
            edgecolor=transparent_white_edge,  # Transparent White Edge
            linewidth=0.1,  # Thin edge line
            ax=axes[1],
        )
        axes[1].set_ylabel("Structural Coherence")
        axes[1].set_xlabel(r"Density")

        # --- Legend Customization ---
        axes[0].legend_.remove()
        axes[1].legend_.remove()

        handles, labels = axes[0].get_legend_handles_labels()

        # Add legend outside top plot
        leg = axes[0].legend(
            handles,
            labels,
            title="Scale",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            edgecolor=dark_color,
        )

        # Thicken the legend border
        leg.get_frame().set_linewidth(2.5)

        plt.tight_layout()
        # plt.show()
        plt.savefig(save_path / f"{bn}_density.png", dpi=300)
        plt.savefig(save_path / f"{bn}_density.svg", dpi=300, transparent=True)
        plt.close()


# --- Run ---
save_path = fig_dir / "Fig1"
save_path.mkdir(exist_ok=True)
create_stacked_scatterplots_density(cohres_df, basenet_numgroups_dict, save_path)

###################################################################################################
###################################################################################################

###################################################################################################
###  MeanCommunicability Vs Density
###################################################################################################


def create_density_communicability_plot(df, save_path):
    """
    Creates a single enhanced scatterplot for Density vs MeanCommunicability.
    Uses BOTH color and shape for NumNodes.
    """
    # Filter for ER networks only
    er_df = df[df["NetType"] == "ER"]

    # Setup Colors
    dark_color = NORD_COLORS["dark"] if "NORD_COLORS" in globals() else "#2E3440"
    # transparent_black_edge = (0, 0, 0, 0.5)
    transparent_black_edge = (0.26, 0.30, 0.37, 0.1)

    # Define 9 Distinct Markers
    custom_markers = ["o", "s", "D", "^", "v", "X", "p", "*", ">"]

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.scatterplot(
        data=er_df,
        x="MeanComm",
        y="Density",
        hue="NumNodes",  # Add Color back
        style="NumNodes",  # Keep Shape
        palette=NORD_PALETTE,  # Your palette (will cycle if <9 colors)
        markers=custom_markers,  # Your custom shapes
        s=80,
        alpha=0.9,
        edgecolor=transparent_black_edge,  # Black "Highlighter" border
        linewidth=1.5,
        ax=ax,
    )

    ax.set_ylabel("Network Density")
    ax.set_xlabel(r"$\log_{10}$(Mean Communicability)")

    # --- Legend Customization ---
    leg = ax.legend(
        title="Network Size\n(Number of Nodes)",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        edgecolor=dark_color,
    )
    leg.get_frame().set_linewidth(2.5)

    plt.tight_layout()
    plt.savefig(save_path / "meancomm_vs_density.png", dpi=300)
    plt.savefig(save_path / "meancomm_vs_density.svg", dpi=300, transparent=True)
    # plt.show()
    plt.close()


# --- Run ---
save_path = fig_dir / "Fig1"
create_density_communicability_plot(cohres_df, save_path)

###################################################################################################
###  MeanCoh vs FoldChangeNumTeams
###################################################################################################


def plot_slope_regression(df, x_col, y_col, hue_col, save_path):
    # --- 1. Setup Data & Colors ---
    palette = (
        NORD_PALETTE if "NORD_PALETTE" in globals() else sns.color_palette("viridis")
    )
    dark_color = NORD_COLORS["dark"] if "NORD_COLORS" in globals() else "#2E3440"

    groups = sorted(df[hue_col].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- 2. Iterate Through Groups ---
    for i, group in enumerate(groups):
        # A. Subset Data
        subset = df[df[hue_col] == group]
        if len(subset) < 2:
            continue

        # B. Linear Regression (Slope & Intercept)
        # dropping NaNs ensures clean calculation
        clean_sub = subset[[x_col, y_col]].dropna()
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            clean_sub[x_col], clean_sub[y_col]
        )

        # Format Label: e.g., "10 Nodes (m=0.45)"
        # We can also add significance like "m=0.45*" if p < 0.05
        # sig = "*" if p_value < 0.05 else ""
        label_text = f"{group} ($m={slope:.3f}$)"
        # label_text = f"{group} ($m={slope:.3f} {sig}$)"

        # C. Get Color
        color = palette[i % len(palette)]

        # D. Plot Scatter (Background)
        sns.scatterplot(
            data=subset,
            x=x_col,
            y=y_col,
            color=color,
            s=70,
            alpha=0.7,
            edgecolor=(0, 0, 0, 0.1),  # Light black edge on points
            # edgecolor=(0.26, 0.30, 0.37, 0.5),
            linewidth=0.3,
            ax=ax,
            legend=False,  # We build legend manually via the line
        )

        # E. Plot Regression Line MANUALLY
        # We generate x values to draw a smooth line
        x_vals = np.linspace(clean_sub[x_col].min(), clean_sub[x_col].max(), 100)
        y_vals = slope * x_vals + intercept

        # Plot the line
        # Note: We add the 'label' here so it shows up in the legend
        (line,) = ax.plot(x_vals, y_vals, color=color, label=label_text, linewidth=2.5)

        # --- THE TRICK FOR VISIBILITY ---
        # Add a "Path Effect" to create an outline around the line
        # 'linewidth=5' determines how thick the border is.
        # 'foreground="white"' creates a white gap between points and the line.
        # Change foreground to 'black' if you want a dark border.
        line.set_path_effects(
            [
                pe.withStroke(linewidth=5, foreground="black", alpha=0.7),  # The Border
                pe.Normal(),  # The Core Line itself
            ]
        )

    # --- 3. Custom Styling ---
    ax.set_xlabel("Fold Change (Team Count)")
    ax.set_ylabel("Structural Coherence")

    # Customize Legend
    leg = ax.legend(
        title="Scale (Slope, $m$)",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        edgecolor=dark_color,
    )
    leg.get_frame().set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig(save_path / "fcnumteams_vs_meancoh_linreg.png", dpi=300)
    plt.savefig(
        save_path / "fcnumteams_vs_meancoh_linreg.svg", dpi=300, transparent=True
    )
    # plt.show()
    plt.close()


# --- Example Run ---
save_path = fig_dir / "Fig1"
save_path.mkdir(exist_ok=True)
plot_slope_regression(
    cohres_df[cohres_df["NetType"] == "ER"],
    "FoldChangeNumTeams",
    "MeanCoh",
    # "NumNodesPerGroup",
    "Scale",
    save_path,
)

###################################################################################################
###################################################################################################

####################################################################################################
#### MinMaxCommunicabilityBin vs MinMaxFoldChangeNumTeams
####################################################################################################


def plot_refined_coherence_pointplot(
    plot_df, base_df, x_col, y_col, hue_col, metric_col="MeanCoh", save_path=Path(".")
):
    # --- 1. Calculate the Mapping Metric (Mean Coherence) ---
    coherence_map = base_df.groupby(hue_col)[metric_col].mean().to_dict()

    # --- 2. Build Colormap (Pure Black -> Nord Orange) ---
    custom_cmap = colors.LinearSegmentedColormap.from_list(
        # "black_to_orange", ["#000000", "#d08770"]
        "black_to_orange",
        ["#000000", "#b48ead"],
    )

    # --- 3. Generate Palette Dictionary ---
    vals = list(coherence_map.values())
    norm = colors.Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))

    # Create a dict { 'BaseNetName': (R, G, B, A) }
    palette_dict = {name: custom_cmap(norm(val)) for name, val in coherence_map.items()}

    # --- 4. Setup Plot ---
    plt.figure(figsize=(10, 6))

    # --- 5. Draw Pointplot ---
    ax = sns.pointplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        estimator="mean",
        errorbar="sd",
        capsize=0.1,
        markers="o",
        scale=1.1,  # Increased marker size slightly
        dodge=0.4,
        palette=palette_dict,
        linewidth=2.5,  # Thicker lines (was 1.0)
        errwidth=2.0,  # Thicker error bars (was 1.0)
        legend=False,
    )

    # Note: "Halo" path effects section has been removed completely.

    # --- 6. Aesthetics ---
    ax.set_xlabel("Min-Max Normalized Mean Communicability")
    ax.set_ylabel("Min-Max Normalized Team Count")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.subplots_adjust(right=0.75)

    # --- 7. Create Sorted Legend ---
    sorted_items = sorted(coherence_map.items(), key=lambda item: item[1])

    legend_elements = [
        Patch(
            facecolor=palette_dict[name],
            edgecolor="none",
            label=f"{name.replace('TN', '')} : {val:.2f}",
        )
        for name, val in sorted_items
    ]
    title_str = "   Base    :   Structural\nNetwork   Coherence"
    ax.legend(
        handles=legend_elements,
        # title="Base Network : Strcutural Coherence",
        title=title_str,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
        fontsize=14,  # Increased from 9 to 12
        title_fontsize=15,  # Increased title size slightly
    )

    plt.tight_layout()
    plt.savefig(save_path / "minmax_communicability_vs_fcnumteams.png", dpi=300)
    plt.savefig(
        save_path / "minmax_communicability_vs_fcnumteams.svg",
        dpi=300,
        transparent=True,
    )
    # plt.show()
    plt.close()


# --- Example Run ---
save_path = fig_dir / "Fig1"
save_path.mkdir(exist_ok=True)
plot_refined_coherence_pointplot(
    plot_df=cohres_df[cohres_df["NetType"] == "ER"],
    base_df=base_df,
    x_col="MinMaxCommunicabilityBin",
    y_col="MinMaxFoldChangeNumTeams",
    hue_col="BaseNet",
    metric_col="MeanCoh",
    save_path=save_path,
)


####################################################################################################
####################################################################################################

####################################################################################################
#### Heatmap - Curavture and Slope
####################################################################################################

er_cohres_df = cohres_df[cohres_df["NetType"] == "ER"]

# Getting the roundede off NormMeanComm
er_cohres_df["NormMeanCommBin"] = er_cohres_df.loc[:, "NormMeanComm"].round(1)


# Subsettingthe resuqiured columns
er_cohres_df = er_cohres_df[
    [
        "NumNodesPerGroup",
        "BaseNet",
        "Replicate",
        "FoldChangeNumTeams",
        "NormMeanCommBin",
    ]
]

# Initialize list to store results
results_list = []

# Loop through your groups
for (group_size, basenet), group_df in er_cohres_df.groupby(
    ["NumNodesPerGroup", "BaseNet"]
):
    # --- 1. PRE-PROCESSING ---
    group_cols = [
        c for c in group_df.columns if c not in ["Replicate", "FoldChangeNumTeams"]
    ]
    avg_df = group_df.groupby(group_cols)["FoldChangeNumTeams"].mean().reset_index()
    avg_df = avg_df.sort_values("NormMeanCommBin")

    # --- 2. DEFINE ANCHORS & SLOPE (Used for Classification Logic) ---
    # Start: Average of first 3
    least_3 = avg_df.head(5)
    x1, y1 = (
        least_3["NormMeanCommBin"].mean(),
        least_3["FoldChangeNumTeams"].mean(),
    )
    # End: Last 1
    max_row = avg_df.tail(1)
    x2, y2 = (
        max_row["NormMeanCommBin"].values[0],
        max_row["FoldChangeNumTeams"].values[0],
    )
    denom = x2 - x1
    if denom == 0:
        m = 0
    else:
        m = (y2 - y1) / denom
    c_intercept = y1 - (m * x1)

    # --- 2a. CALCULATE REPORTED SLOPE (Min/Max FoldChange) ---
    # Find the row with the global MINIMUM FoldChange in this group
    row_min_y = avg_df.loc[avg_df["FoldChangeNumTeams"].idxmin()]
    # Find the row with the global MAXIMUM FoldChange in this group
    row_max_y = avg_df.loc[avg_df["FoldChangeNumTeams"].idxmax()]

    # Extract coordinates
    x_at_min, y_min = (
        row_min_y["NormMeanCommBin"],
        row_min_y["FoldChangeNumTeams"],
    )
    x_at_max, y_max = (
        row_max_y["NormMeanCommBin"],
        row_max_y["FoldChangeNumTeams"],
    )

    # Calculate slope between these two extreme points
    denom_reported = x_at_max - x_at_min
    if denom_reported == 0:
        reported_slope = 0  # Handle vertical line edge case
    else:
        # reported_slope = (y_max - y_min) / denom_reported
        reported_slope = m

    # --- 3. CALCULATE MIDDLE METRICS (Uses original 'm') ---
    n = len(avg_df)
    mid_idx = n // 2
    middle_two_rows = avg_df.iloc[mid_idx - 1 : mid_idx + 1].copy()
    mid_avg_actual_y = middle_two_rows["FoldChangeNumTeams"].mean()

    mid_avg_line_y = (m * middle_two_rows["NormMeanCommBin"].mean()) + c_intercept

    if mid_avg_line_y != 0:
        bulge_ratio = mid_avg_actual_y / mid_avg_line_y
    else:
        bulge_ratio = 1.0

    # --- 4. CLASSIFICATION LOGIC ---
    if abs(y1 - y2) <= 0.1:
        classification = "Flat"
    elif 0.75 <= bulge_ratio <= 1.25:
        classification = "Linear"
    elif bulge_ratio < 0.75:
        classification = "Concave"
    elif bulge_ratio > 1.25:
        classification = "Convex"
    else:
        classification = "Undefined"

    # --- 5. STORE RESULTS ---
    results_list.append(
        {
            "NumNodesPerGroup": group_size,
            "BaseNet": basenet,
            "Slope": reported_slope,
            "Start_Y": y1,
            "End_Y": y2,
            "Bulge_Ratio": bulge_ratio,
            "Classification": classification,
        }
    )

# --- 7. CREATE FINAL DATAFRAME ---
final_classification_df = pd.DataFrame(results_list)
final_classification_df["Scale"] = (
    final_classification_df["NumNodesPerGroup"].astype(str) + "x"
)


def plot_nord_curvature_heatmap(df, base_df=None, save_path=None):
    """
    Plots a curvature heatmap using the Nord color palette.
    Distinguishes between 'Linear' and 'Flat' classifications.

    Args:
        df (pd.DataFrame): Final classification dataframe.
        base_df (pd.DataFrame, optional): For sorting X-axis by 'MedianCoh'.
        output_path (str): Save path.
    """
    print("--- Generating Nord Curvature Heatmap ---")

    # 1. Pivot Data
    slope_pivot = df.pivot(index="BaseNet", columns="Scale", values="Slope")
    class_pivot = df.pivot(index="BaseNet", columns="Scale", values="Classification")

    # 2. Sort Y-axis (BaseNet)
    # The sorting logic now applies to the Index (axis=0), not columns
    if base_df is not None:
        # Assuming you want to sort by MeanCoh/MedianCoh
        sorted_basenets = base_df.sort_values("MeanCoh")["BaseNet"].tolist()

        # Filter to keep only basenets present in the pivot table
        valid_order = [b for b in sorted_basenets if b in slope_pivot.index]

        # Reindex the rows
        slope_pivot = slope_pivot.reindex(valid_order)
        class_pivot = class_pivot.reindex(valid_order)
    else:
        # Default alphabetical sort for rows
        slope_pivot = slope_pivot.sort_index(axis=0)
        class_pivot = class_pivot.sort_index(axis=0)

    # Ensure X-axis (NumNodesPerGroup) is sorted numerically
    slope_pivot = slope_pivot.sort_index(axis=1)
    class_pivot = class_pivot.sort_index(axis=1)

    # 3. Nord Color Palette Definition
    nord_colors = {
        "red": "#BF616A",  # Nord11 - Concave
        "grey": "#D8DEE9",  # Nord4  - Flat
        "blue": "#81A1C1",  # Nord9  - Linear
        "green": "#A3BE8C",  # Nord14 - Convex
    }

    # 4. Map Classifications to Integers
    # We need 4 distinct integers for 4 colors
    class_map = {
        "Flat": 0,
        "Concave": 1,
        "Linear": 2,
        "Convex": 3,
        "Vertical Data": 1,  # Treat as Flat/Neutral
        "Not Enough Data": 1,
    }

    # Apply mapping
    data_for_heatmap = class_pivot.replace(class_map).fillna(1).astype(int)

    # 5. Create Discrete Colormap
    # Order must match the integer mapping: 0, 1, 2, 3
    cmap_list = [
        nord_colors["grey"],
        nord_colors["red"],
        nord_colors["blue"],
        nord_colors["green"],
    ]
    cmap = colors.ListedColormap(cmap_list)

    # Define boundaries for the discrete bins
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # 6. Plotting
    fig, ax = plt.subplots(figsize=(8, 10))

    sns.heatmap(
        data=data_for_heatmap,
        annot=slope_pivot,  # Annotate with Slope
        fmt=".2f",  # 2 decimal places
        cmap=cmap,
        norm=norm,  # Use discrete normalization
        linewidths=0.5,
        linecolor="#2E3440",  # Nord0 (Dark Grey) for grid lines
        cbar_kws={"ticks": [0, 1, 2, 3], "shrink": 0.6, "pad": 0.08},
        annot_kws={"size": 14},
        ax=ax,
    )

    # 7. Customize Colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(["Flat", "Concave", "Linear", "Convex"])
    cbar.ax.tick_params(size=0)

    # 8. Titles and Labels
    # ax.set_title(
    #     "Curvature Analysis (Nord Theme)\nColor=Shape, Text=Slope",
    #     fontsize=16,
    #     color="#2E3440",
    # )
    ax.set_ylabel("Base Network (Ordered By Structural Coherence)")
    ax.set_xlabel("Scale")

    # plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path / "heatmap_curvature_classif.png", dpi=300)
    plt.savefig(save_path / "heatmap_curvature_classif.svg", dpi=300, transparent=True)
    plt.close()


# --- Usage ---
save_path = fig_dir / "Fig1"
save_path.mkdir(exist_ok=True)
plot_nord_curvature_heatmap(
    final_classification_df, base_df=base_df, save_path=save_path
)

####################################################################################################
####################################################################################################

####################################################################################################
### Slope distribtion
####################################################################################################


def calculate_slope_distributions(df):
    """
    Calculates the slope for each unique replicate using a custom 3-point method:
    Slope = (Avg(Last 2 Y) - First Y) / (Avg(Last 2 X) - First X)
    """
    print("--- Calculating Slopes per Replicate (First vs Avg Last 2) ---")

    slope_data = []

    # Group by the unique identifier for a single trajectory
    grouper = df.groupby(["BaseNet", "NumNodesPerGroup", "Replicate"])

    for (basenet, groupsize, repl), group_df in grouper:
        # Drop NaNs
        clean_df = group_df.dropna(subset=["NormMeanCommBin", "FoldChangeNumTeams"])

        # IMPORTANT: Sort by X axis to ensure 'First' and 'Last' are correct
        clean_df = clean_df.sort_values(by="NormMeanCommBin")

        # We need at least 2 points to have a "First" and "Last"
        if len(clean_df) < 2:
            continue

        # Extract arrays for easier indexing
        x = clean_df["NormMeanCommBin"].values
        y = clean_df["FoldChangeNumTeams"].values

        # --- Custom Method ---

        # 1. Identify the First Point (Start)
        x_start = x[0]
        y_start = y[0]

        # 2. Identify the Average of the Last Two Points (End)
        # We average both X and Y to find the geometric center of the last segment
        x_end_avg = (x[-1] + x[-2]) / 2.0
        y_end_avg = (y[-1] + y[-2]) / 2.0

        # Safety check: avoid division by zero if start and end X are identical
        if (x_end_avg - x_start) == 0:
            continue

        # 3. Calculate Slope
        slope = (y_end_avg - y_start) / (x_end_avg - x_start)

        slope_data.append(
            {
                "BaseNet": basenet,
                "NumNodesPerGroup": groupsize,
                "Replicate": repl,
                "Slope": slope,
            }
        )

    return pd.DataFrame(slope_data)


# --- EXECUTION ---
slope_dist_df = calculate_slope_distributions(er_cohres_df)

# Check the output
print(f"Calculated slopes for {len(slope_dist_df)} replicates.")
print(slope_dist_df.head())


def plot_slope_distributions(slope_df, base_df=None):
    """
    Plots the distribution of calculated slopes using boxplots.
    x-axis: BaseNet (sorted by MeanCoh from base_df)
    hue: NumNodesPerGroup
    """
    plt.figure(figsize=(14, 8))

    # --- Sorting Logic ---
    # Determine the order of BaseNets for the X-axis
    order = None
    if base_df is not None:
        # Sort base_df by 'MeanCoh' and extract the BaseNet names in that order
        # Ensure 'MeanCoh' exists in base_df, otherwise change to your specific sorting column
        if "MeanCoh" in base_df.columns:
            sorted_basenets = base_df.sort_values("MeanCoh")["BaseNet"].tolist()

            # Filter the order list to only include BaseNets actually present in slope_df
            # (Prevents errors if base_df has networks not in the current slope data)
            existing_nets = set(slope_df["BaseNet"].unique())
            order = [net for net in sorted_basenets if net in existing_nets]
        else:
            print(
                "Warning: 'MeanCoh' not found in base_df. X-axis will not be sorted by metric."
            )

    # --- Plotting ---
    # Flipped: X is BaseNet, Hue is NumNodesPerGroup
    sns.boxplot(
        data=slope_df,
        x="BaseNet",
        y="Slope",
        hue="NumNodesPerGroup",
        order=order,  # Apply the sorted order here
        palette="viridis",
        showfliers=False,
    )

    # Reference line at 0
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)

    plt.title("Distribution of Slopes (First vs. Avg Last 2)", fontsize=16)
    plt.ylabel("Slope (Rate of Change)", fontsize=12)
    plt.xlabel("Base Network", fontsize=12)

    # Rotate X-axis labels 90 degrees
    plt.xticks(rotation=90)

    plt.legend(title="Nodes Per Group", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# plot_slope_distributions(slope_dist_df, base_df)


def plot_slope_trends(slope_df, base_df=None):
    """
    Plots the trend of Slope vs NumNodesPerGroup for each BaseNet.
    """
    plt.figure(figsize=(14, 8))

    # --- Sorting Logic (Optional but recommended) ---
    # We sort the hue order based on MeanCoh just like before so the legend is organized
    hue_order = None
    if base_df is not None and "MeanCoh" in base_df.columns:
        sorted_nets = base_df.sort_values("MeanCoh", ascending=False)[
            "BaseNet"
        ].tolist()
        existing_nets = set(slope_df["BaseNet"].unique())
        hue_order = [net for net in sorted_nets if net in existing_nets]

    # --- Plotting ---
    # sns.lineplot automatically aggregates replicates:
    # - The solid line is the Mean
    # - The shaded area is the Confidence Interval (95% by default)
    sns.lineplot(
        data=slope_df,
        x="NumNodesPerGroup",
        y="Slope",
        hue="BaseNet",
        hue_order=hue_order,
        palette="viridis",
        marker="o",  # Adds dots at the data points
        linewidth=2,
    )

    # Reference line at 0 (No change)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    plt.title("Trend of Fold Change Slope vs. Group Size", fontsize=16)
    plt.xlabel("Number of Nodes Per Group", fontsize=12)
    plt.ylabel("Slope (Rate of Change)", fontsize=12)

    # Place legend outside because there might be many BaseNets
    plt.legend(title="Base Network", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# --- EXECUTION ---
# Ensure 'NumNodesPerGroup' is numeric for a proper line plot
slope_dist_df["NumNodesPerGroup"] = pd.to_numeric(slope_dist_df["NumNodesPerGroup"])
# plot_slope_trends(slope_dist_df, base_df)


def plot_coherence_vs_slope(slope_df, base_df):
    """
    Plots a scatter plot of Mean Coherence (from base_df) vs. Slope (from slope_df).
    Points are colored by NumNodesPerGroup.
    """
    # 1. Merge the slope data with the base_df to get 'MeanCoh' for each BaseNet
    # We use a left join to keep all slope replicates
    merged_df = pd.merge(
        slope_df, base_df[["BaseNet", "MeanCoh"]], on="BaseNet", how="left"
    )

    # Check if merge was successful (avoid empty plot if names don't match)
    if merged_df["MeanCoh"].isnull().all():
        print("Error: Could not match BaseNet names between slope_df and base_df.")
        return

    plt.figure(figsize=(10, 7))

    # 2. Create the Scatter Plot
    # hue: distinct colors for group sizes
    # style: distinct shapes for group sizes (helps accessibility)
    sns.scatterplot(
        data=merged_df,
        x="MeanCoh",
        y="Slope",
        hue="NumNodesPerGroup",
        palette="viridis",
        style="NumNodesPerGroup",
        s=100,  # Marker size
        alpha=0.8,  # Slight transparency to see overlaps
    )

    # 3. Add a Global Trend Line (Optional but helpful)
    # This fits a simple linear regression to ALL points to show the general direction
    sns.regplot(
        data=merged_df,
        x="MeanCoh",
        y="Slope",
        scatter=False,  # We already plotted scatter points above
        color="black",
        line_kws={"linestyle": "--", "linewidth": 1.5, "alpha": 0.6},
        ci=None,  # Turn off confidence interval shading for clarity
    )

    # 4. Calculate and display correlation coefficient
    # Drop NaNs just for calculation
    calc_df = merged_df.dropna(subset=["MeanCoh", "Slope"])
    if len(calc_df) > 1:
        corr_val = calc_df["MeanCoh"].corr(calc_df["Slope"])
        plt.text(
            0.05,
            0.95,
            f"Pearson r = {corr_val:.3f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.title("Correlation: Network Coherence vs. Slope", fontsize=16)
    plt.xlabel("Structural Coherence (BaseNet)", fontsize=12)
    plt.ylabel("Slope (Rate of Change)", fontsize=12)
    plt.legend(title="Nodes Per Group", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# --- EXECUTION ---
# Ensure base_df has 'MeanCoh' column
# plot_coherence_vs_slope(slope_dist_df, base_df)

####################################################################################################
####################################################################################################


###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################


###########################################################################################################
### PLotting the ER vs HI stats
###########################################################################################################


def plot_er_hi_scatter_per_basenet(df, save_dir):
    """
    Generates side-by-side scatterplots (ER vs HI) for each BaseNet.
    Plots NumGroups vs MeanCoh, colored by NumNodesPerGroup.
    """
    print("--- Generating ER vs HI Scatterplots ---")

    # Ensure directory exists
    save_dir.mkdir(exist_ok=True, parents=True)

    # Get unique basenets
    basenets = df["BaseNet"].unique()

    for basenet in basenets:
        # Subset for this basenet
        bn_df = df[df["BaseNet"] == basenet]

        # Setup Figure: 1 Row, 2 Columns (ER left, HI right)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)

        # Common Plotting Args
        plot_args = {
            "x": "NumGroups",
            "y": "MeanCoh",
            "hue": "NumNodesPerGroup",
            "palette": NORD_PALETTE,
            "s": 80,
            "alpha": 0.8,
            "edgecolor": NORD_COLORS["dark"],
            "linewidth": 0.5,
        }

        # --- ER PLOT ---
        sns.scatterplot(
            data=bn_df[bn_df["NetType"] == "ER"],
            ax=axes[0],
            legend=False,  # Hide legend on first plot
            **plot_args,
        )
        axes[0].set_title(f"{basenet} - ER", fontweight="bold")
        axes[0].set_ylim(-1.05, 1.05)
        axes[0].set_ylabel("Structural Coherence")

        # --- HI PLOT ---
        sns.scatterplot(
            data=bn_df[bn_df["NetType"] == "HI"],
            ax=axes[1],
            legend=True,  # Show legend here
            **plot_args,
        )
        axes[1].set_title(f"{basenet} - HI", fontweight="bold")
        axes[1].set_ylabel("")  # Hide Y label on shared axis

        # Clean Legend
        axes[1].legend(title="Nodes/Group", bbox_to_anchor=(1.02, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(save_dir / f"{basenet}_ER_vs_HI_scatter.png", dpi=300)
        plt.savefig(
            save_dir / f"{basenet}_ER_vs_HI_scatter.svg", dpi=300, transparent=True
        )
        plt.close()


def plot_er_hi_comparison_boxplot(data, x, y, hue, save_dir):
    """
    Creates a Nord-themed boxplot comparing metrics across categories (NetType)
    with Mann-Whitney statistical annotations.
    """
    print(f"--- Plotting Boxplot: {y} by {x} (hue={hue}) ---")
    save_dir.mkdir(exist_ok=True, parents=True)

    label_dict = {
        "MeanCoh": "Structural Coherence",
        "NumGroups": "Team Count",
        "NormMeanComm": r"$\log_{10}$(Mean Normalized Communicability)",
        "MeanCommNZ": "MeanCommNZ",
    }

    # 1. Setup Colors (Red vs Blue for ER/HI contrast)
    # Using specific Nord colors for binary comparison
    comparison_palette = [NORD_COLORS["red"], NORD_COLORS["blue"], NORD_COLORS["green"]]

    # 2. Initialize Figure
    fig, ax = plt.subplots(figsize=(8, 7))

    # 3. Define Dark Border Properties (Nord Style)
    dark_color = NORD_COLORS["dark"]
    plot_props = {
        "boxprops": {"edgecolor": dark_color, "linewidth": 2.0},
        "whiskerprops": {"color": dark_color, "linewidth": 2.0},
        "capprops": {"color": dark_color, "linewidth": 2.0},
        "medianprops": {"color": dark_color, "linewidth": 2.0},
        "flierprops": {"markeredgecolor": dark_color},
    }

    # 4. Create Boxplot
    sns.boxplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        palette=comparison_palette,
        showfliers=False,
        width=0.7,
        ax=ax,
        **plot_props,
    )

    # 5. Add Zero Line if data spans 0
    if data[y].min() < 0 < data[y].max():
        ax.axhline(y=0, linewidth=2.0, color=dark_color, linestyle="--", alpha=0.6)

    # 6. Statistical Annotations
    # Generate pairs: Compare hues (ER vs HI) within each x-group
    x_order = sorted(data[x].unique())
    hue_order = sorted(data[hue].unique())

    # Only annotate if we have exactly 2 hues (e.g., ER vs HI) per group
    if len(hue_order) >= 2:
        box_pairs = [
            ((x_val, h1), (x_val, h2))
            for x_val in x_order
            for h1, h2 in itertools.combinations(hue_order, 2)
        ]

        if box_pairs:
            annotator = Annotator(ax, box_pairs, data=data, x=x, y=y, hue=hue)
            annotator.configure(
                test="Mann-Whitney",
                text_format="star",
                loc="inside",
                verbose=0,
                line_width=1.5,
                color=dark_color,
            )
            annotator.apply_and_annotate()

    # 7. Final Polish
    ax.set_title(f"Comparison: {y}")
    ax.set_xlabel(x)
    ax.set_ylabel(label_dict[y])

    # Legend outside
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Network Type")

    plt.tight_layout()
    plt.savefig(save_dir / f"Boxplot_{y}_by_{x}.png", dpi=300)
    plt.savefig(save_dir / f"Boxplot_{y}_by_{x}.svg", dpi=300, transparent=True)
    plt.close()


# --- EXECUTION ---

# 1. Define Save Path for these specific plots
er_hi_plot_dir = fig_dir / "ER_vs_HI"

# 2. Run Scatterplots (Per BaseNet)
# plot_er_hi_scatter_per_basenet(cohres_df, er_hi_plot_dir)

# 3. Run Boxplots (Aggregated Stats)
metrics_to_plot = ["MeanCoh", "NumGroups", "NormMeanComm"]

# Add MeanCommNZ only if it exists in dataframe
if "MeanCommNZ" in cohres_df.columns:
    metrics_to_plot.append("MeanCommNZ")

for metric in metrics_to_plot:
    plot_er_hi_comparison_boxplot(
        data=cohres_df,
        x="Scale",
        y=metric,
        hue="NetType",
        save_dir=er_hi_plot_dir,
    )

print(cohres_df)
print(cohres_df.dtypes)

###################################################################################################
###################################################################################################

###################################################################################################
#### Plotting ratio at 10 density
###################################################################################################


def get_er_hi_ratio_dataframe(df, metric="MeanCoh"):
    print(f"--- Building Summary Dataframe for {metric} ---")

    # 1. Define grouping columns for aggregation
    # We include NetType here because we need to calculate means for ER and HI separately
    group_cols = ["BaseNet", "NumNodesPerGroup", "NetType", "Density"]

    # 2. Group and Aggregate
    agg_series = df.groupby(group_cols)[metric].mean()
    agg_series.name = f"{metric}_Mean"

    # 3. Reset Index
    df_agg = agg_series.reset_index()

    # 4. Pivot
    pivot_index = [c for c in group_cols if c != "NetType"]

    df_pivot = df_agg.pivot(
        index=pivot_index,  # Rows: BaseNet, Density, etc.
        columns="NetType",  # Columns: ER vs HI
        values=f"{metric}_Mean",  # Values: The aggregated mean
    ).reset_index()

    # Clean up column index name
    df_pivot.columns.name = None

    # 5. Calculate Ratio
    if "ER" in df_pivot.columns and "HI" in df_pivot.columns:
        df_pivot[f"{metric}_ER_Mean"] = df_pivot["ER"]
        df_pivot[f"{metric}_HI_Mean"] = df_pivot["HI"]

        # Calculate Ratio
        df_pivot[f"{metric}_Ratio"] = df_pivot["HI"] / df_pivot["ER"]
    else:
        print("Error: Could not find both 'ER' and 'HI' in the data.")
        return None

    return df_pivot


# Generate the dataframe
ratio_df = get_er_hi_ratio_dataframe(cohres_df, metric="NumGroups")
ratio_df["Scale"] = ratio_df["NumNodesPerGroup"].astype(str) + "x"

# Inspect the result
print(ratio_df)
print(ratio_df.dtypes)

for bn in ratio_df["BaseNet"].unique():
    rdf = ratio_df[ratio_df["BaseNet"] == bn]

    plt.figure(figsize=(7.5, 5))

    # Determine logical order for the hue (NumNodesPerGroup)
    # Sorting ensures the legend goes from Small -> Large (or vice versa)
    hue_order = sorted(rdf["Scale"].unique())

    # 2. Point Plot
    sns.pointplot(
        data=rdf,
        x="Density",
        y="NumGroups_Ratio",
        hue="Scale",
        hue_order=hue_order,  # Enforce sorted order
        palette=NORD_PALETTE,  # Nord Categorical Colors
        # Styling the points and lines
        markers="o",
        # scale=1.2,
        # err_kws={"linewidth": 1.5},  # Error bar width
        capsize=0.1,  # Small caps on error bars
        dodge=True,  # Separate points slightly for clarity
    )
    plt.ylim(-0.1, 4.2)

    # 3. Styling
    # plt.title(f"No. Teams Ratio for {bn}", pad=10)  # No bold, simple padding
    plt.xlabel("Density")
    plt.ylabel("Team Count Ratio (HI / ER)")

    # Legend
    plt.legend(
        title="Scale",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
    )
    plt.axhline(
        y=1,
        color=NORD_COLORS["dark"],
        linestyle="-.",
        linewidth=2.0,
        alpha=0.7,
        zorder=0,
    )

    # Spines
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])
        spine.set_linewidth(2.0)

    plt.xticks(rotation=90)
    plt.title(bn)
    plt.tight_layout()
    # Save or Show
    save_path = er_hi_plot_dir / f"FoldChangeRatio_{bn}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(
        save_path.with_name(save_path.name.replace("png", "svg")),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    # plt.show()
    plt.close()

# 1. Filter Data for Density == 10
plot_df = ratio_df[ratio_df["Density"] == 10].copy()

# ==============================================================================
# 1. SORTING LOGIC (Using base_df)
# ==============================================================================
order = None

# Ensure base_df is defined and has the metric
if "base_df" in locals() and base_df is not None:
    # Sort base_df by 'MeanCoh' (or your specific metric)
    if "MeanCoh" in base_df.columns:
        sorted_basenets = base_df.sort_values("MeanCoh")["BaseNet"].tolist()

        # Filter the order list to only include BaseNets actually present in the PLOTTING dataframe
        # (Using ratio_df here based on previous context)
        existing_nets = set(ratio_df["BaseNet"].unique())

        # Create the final order list
        order = [net for net in sorted_basenets if net in existing_nets]
    else:
        print("Warning: 'MeanCoh' not found in base_df. Defaulting to alphabetical.")
        order = sorted(ratio_df["BaseNet"].unique())
else:
    # Fallback if base_df isn't available
    order = sorted(ratio_df["BaseNet"].unique())


def plot_numgroups_ratio(ratio_df, base_df=None, density_val=10):
    """
    Plots the Number of Groups Ratio sorted by Mean Coherence.
    - Stripplot for raw data (colored linearly).
    - Pointplot for mean (Black, on top).
    - X-axis rotated 90 degrees.
    """

    # --- 1. SORTING LOGIC ---
    # Default to alphabetical if no base_df or metric found
    order = sorted(ratio_df["BaseNet"].unique())

    if base_df is not None:
        # Check for common metric names
        metric_col = None
        for col in ["MeanCoh", "MeanAbsOutCoh", "Coherence"]:
            if col in base_df.columns:
                metric_col = col
                break

        if metric_col:
            # Sort base_df by the metric
            sorted_basenets = base_df.sort_values(metric_col)["BaseNet"].tolist()
            # Intersect with networks present in ratio_df
            existing_nets = set(ratio_df["BaseNet"].unique())
            order = [net for net in sorted_basenets if net in existing_nets]
        else:
            print(
                "Warning: No coherence metric found in base_df. Using alphabetical order."
            )

    # --- 2. DATA FILTERING ---
    plot_df = ratio_df[ratio_df["Density"] == density_val].copy()

    # --- 3. CUSTOM LINEAR NORD PALETTE ---
    # n_colors matches the number of networks to ensure a distinct linear mapping
    n_nets = len(order)

    # Create a gradient. You can add intermediate colors if you want (e.g. Blue -> Green -> Red)
    # Using Dark -> Purple as in your snippet, or Blue -> Red for high contrast
    # nord_linear_cmap = colors.LinearSegmentedColormap.from_list(
    #     "NordLinear", [NORD_COLORS["blue"], NORD_COLORS["purple"], NORD_COLORS["red"]]
    # )
    nord_linear_cmap = colors.LinearSegmentedColormap.from_list(
        "NordLinear", [NORD_COLORS["dark"], "#b48ead"]
    )

    # Generate the specific colors from this map dynamically based on n_nets
    linear_palette = [nord_linear_cmap(i) for i in np.linspace(0, 1, n_nets)]

    # --- 4. PLOTTING ---
    plt.figure(figsize=(10, 6))

    # LAYER 1: STRIPPLOT (Raw Data)
    sns.stripplot(
        data=plot_df,
        x="BaseNet",
        y="NumGroups_Ratio",
        order=order,
        hue="BaseNet",
        hue_order=order,
        palette=linear_palette,
        legend=False,
        size=14,
        jitter=True,
        zorder=0,
    )

    # LAYER 2: POINTPLOT (Mean Marker)
    # Strictly Black for visibility, placed on top
    sns.pointplot(
        data=plot_df,
        x="BaseNet",
        y="NumGroups_Ratio",
        order=order,
        color="black",  # Black mean marker
        join=False,
        errorbar=None,
        markers="_",  # Horizontal line
        scale=2.0,  # Make it big/thick
        zorder=10,  # Force it to the very top
    )

    # --- 5. STYLING ---
    # plt.title(f"Number of Teams Ratio (Density = {density_val})", pad=20)
    plt.ylabel("Team Count Ratio (HI / ER)")
    plt.xlabel("Base Network (Sorted by Structural Coherence)")

    # Baseline at 1.0
    plt.axhline(
        1, color=NORD_COLORS["dark"], linestyle="--", linewidth=1.5, alpha=0.5, zorder=0
    )

    # Rotate X-Axis Labels
    plt.xticks(rotation=90)

    # Nord Spines
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])
        spine.set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig(er_hi_plot_dir / "10Density_BaseNetNumGorupsRatio.png", dpi=300)
    plt.savefig(
        er_hi_plot_dir / "10Density_BaseNetNumGorupsRatio.svg",
        dpi=300,
        transparent=True,
    )
    # plt.show()
    plt.close()


# Example Call:
plot_numgroups_ratio(ratio_df, base_df)

###################################################################################################
###################################################################################################


def plot_numgroups_trajectory(ratio_df, base_df=None):
    """
    Plots the Trajectory of NumGroups_Ratio across Densities.
    - X-axis: Density (shows the progression).
    - Y-axis: NumGroups_Ratio.
    - Hue: BaseNet (colored linearly by Coherence).
    - Style: Pointplot (lines connected).
    """

    order = sorted(ratio_df["BaseNet"].unique())
    coherence_map = {}
    metric_used = "BaseNet"  # Default title if no metric found

    if base_df is not None:
        # Check for common metric names
        metric_col = None
        for col in ["MeanCoh", "MeanAbsOutCoh", "Coherence"]:
            if col in base_df.columns:
                metric_col = col
                break

        if metric_col:
            # 1. Sort base_df by the metric
            sorted_basenets = base_df.sort_values(metric_col)["BaseNet"].tolist()

            # 2. Intersect with networks present in ratio_df
            existing_nets = set(ratio_df["BaseNet"].unique())
            order = [net for net in sorted_basenets if net in existing_nets]

            # 3. Create a map of {Network: Value} for the legend
            coherence_map = base_df.set_index("BaseNet")[metric_col].to_dict()
            metric_used = metric_col
        else:
            print("Warning: No metric found. Using alphabetical order.")

    n_nets = len(order)
    nord_linear_cmap = colors.LinearSegmentedColormap.from_list(
        "NordLinear", [NORD_COLORS["dark"], "#b48ead"]
    )
    # Generate colors matching the number of networks
    linear_palette = [nord_linear_cmap(i) for i in np.linspace(0, 1, n_nets)]
    color_map = dict(zip(order, linear_palette))

    # --- 3. PLOTTING ---
    plt.figure(figsize=(11, 6))

    # Point Plot: Draws lines connecting the densities for each network
    sns.pointplot(
        data=ratio_df,
        x="Density",
        y="NumGroups_Ratio",
        hue="BaseNet",
        hue_order=order,  # Sort hues by Coherence
        palette=linear_palette,  # Apply linear gradient
        markers=".",  # Small markers to emphasize the line
        # scale=0.8,
        # err_width=None,  # Remove error bars for cleaner "trajectory" look
        errorbar=None,
        # errorbar="sd",
        # capsize=0.15,
        # err_kws={"linewidth": 1.2},
        dodge=False,  # Keep lines straight (no offset)
        alpha=1.0,
    )

    legend_elements = []
    for net_name in order:
        # Get color and value
        c = color_map[net_name]
        val = coherence_map.get(net_name, 0.0)

        # Format label depending on whether we found a metric
        if coherence_map:
            label_text = f"{net_name} : {val:.2f}"
        else:
            label_text = f"{net_name}"

        legend_elements.append(Patch(facecolor=c, edgecolor="none", label=label_text))

    # Add the custom legend
    plt.legend(
        handles=legend_elements,
        title="Base Network : Structural Coherence",  # Dynamic title
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
        fontsize=12,  # Readable size
        title_fontsize=13,  # Slightly larger title
    )

    # --- 4. STYLING ---
    plt.title("Trajectory of Teams Ratios Across Densities", pad=15)
    plt.ylabel("Team Count Ratio (HI / ER)")
    plt.xlabel("Density")

    # Baseline at 1.0
    plt.axhline(
        1, color=NORD_COLORS["dark"], linestyle="--", linewidth=1.5, alpha=0.5, zorder=0
    )
    # Nord Spines
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])
        spine.set_linewidth(2.0)

    # Clean up X-axis (if many densities, rotate labels)
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(er_hi_plot_dir / "AllDensity_BaseNetNumGorupsRatio.png", dpi=300)
    plt.savefig(
        er_hi_plot_dir / "AllDensity_BaseNetNumGorupsRatio.svg",
        dpi=300,
        transparent=True,
    )
    # plt.show()
    plt.close()


# Example Call:
plot_numgroups_trajectory(ratio_df, base_df)

###################################################################################################
#### ER vs HI MeanCoh vs FoldChangeNumTeams
###################################################################################################


def plot_er_hi_scatter(df, x_col, y_col, hue_col, save_path):
    """
    Plots Scatter for ER vs HI with HI points layered ON TOP of ER points.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # 1. Define Colors
    palette = {"ER": NORD_COLORS["purple"], "HI": NORD_COLORS["red"]}

    # 2. SORTING LOGIC (Critical Step)
    # Sort so 'ER' comes first (drawn at bottom) and 'HI' comes last (drawn on top)
    # Since 'E' < 'H', ascending sort works perfectly.
    df_sorted = df.sort_values(hue_col, ascending=True)

    # 3. Plot Scatter
    sns.scatterplot(
        data=df_sorted,  # <--- Use the sorted dataframe
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=palette,
        hue_order=["ER", "HI"],  # Ensure legend stays in specific order
        s=60,
        alpha=0.8,
        edgecolor=NORD_COLORS["dark"],
        linewidth=0.3,
        ax=ax,
        legend=True,
    )

    # 4. Styling
    ax.set_xlabel("Fold Change (Team Count)")
    ax.set_ylabel("Structural Coherence")
    # ax.set_title(f"Scatter: {x_col} vs {y_col}", pad=15)

    # Move Legend
    sns.move_legend(
        ax,
        "upper left",
        bbox_to_anchor=(1.02, 1),
        title="Network\nType",
        frameon=True,
    )

    # Legend Styling
    legend = ax.get_legend()
    legend.get_frame().set_edgecolor(NORD_COLORS["dark"])
    legend.get_frame().set_linewidth(1.5)

    # Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])
        spine.set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig(er_hi_plot_dir / "fcnumteams_vs_meancoh_scatter.png", dpi=300)
    plt.savefig(
        er_hi_plot_dir / "fcnumteams_vs_meancoh_scatter.svg", dpi=300, transparent=True
    )
    # plt.show()
    plt.close()


def plot_er_hi_scatter_minmax(df, x_col, y_col, hue_col, save_path):
    """
    Plots Scatter for ER vs HI with HI points layered ON TOP of ER points.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # 1. Define Colors
    palette = {"ER": NORD_COLORS["purple"], "HI": NORD_COLORS["red"]}

    # 2. SORTING LOGIC (Critical Step)
    # Sort so 'ER' comes first (drawn at bottom) and 'HI' comes last (drawn on top)
    # Since 'E' < 'H', ascending sort works perfectly.
    df_sorted = df.sort_values(hue_col, ascending=True)

    # 3. Plot Scatter
    sns.scatterplot(
        data=df_sorted,  # <--- Use the sorted dataframe
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=palette,
        hue_order=["ER", "HI"],  # Ensure legend stays in specific order
        s=60,
        alpha=0.8,
        edgecolor=NORD_COLORS["dark"],
        linewidth=0.3,
        ax=ax,
        legend=True,
    )

    # 4. Styling
    ax.set_xlabel("Min-Max Normalized Fold Change (Team Count)")
    ax.set_ylabel("Structural Coherence")
    # ax.set_title(f"Scatter: {x_col} vs {y_col}", pad=15)

    # Move Legend
    sns.move_legend(
        ax,
        "upper left",
        bbox_to_anchor=(1.02, 1),
        title="Network\nType",
        frameon=True,
    )

    # Legend Styling
    legend = ax.get_legend()
    legend.get_frame().set_edgecolor(NORD_COLORS["dark"])
    legend.get_frame().set_linewidth(1.5)

    # Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])
        spine.set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig(er_hi_plot_dir / "minmax_fcnumteams_vs_meancoh_scatter.png", dpi=300)
    plt.savefig(
        er_hi_plot_dir / "minmax_fcnumteams_vs_meancoh_scatter.svg",
        dpi=300,
        transparent=True,
    )
    # plt.show()
    plt.close()


def plot_er_hi_scatter_with_regression(df, x_col, y_col, hue_col, save_path):
    """
    Plots Scatter for ER vs HI with Regression Lines and Slope (m).

    Features:
    1. Layers HI (Top) over ER (Bottom).
    2. Calculates linear regression slope (m).
    3. CLAMPS the regression line so it does not extend below y = -0.8
       (prevents lines from shooting into empty space).
    """
    # 1. Setup Figure
    fig, ax = plt.subplots(figsize=(7, 6))

    # 2. Define Colors & Order (ER first = Bottom, HI second = Top)
    groups = ["ER", "HI"]
    palette = {"ER": NORD_COLORS["purple"], "HI": NORD_COLORS["red"]}

    # 3. Iterate to Plot Scatter + Line for each group
    for group in groups:
        # A. Subset Data
        subset = df[df[hue_col] == group]
        if len(subset) < 2:
            continue

        # B. Linear Regression (Slope & Intercept)
        clean_sub = subset[[x_col, y_col]].dropna()
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            clean_sub[x_col], clean_sub[y_col]
        )

        # C. Create Label with Slope
        label_text = f"{group} ($m={slope:.2f}$)"
        color = palette[group]

        # D. Plot Scatter (Points)
        sns.scatterplot(
            data=subset,
            x=x_col,
            y=y_col,
            color=color,
            s=60,
            alpha=0.6,
            edgecolor=NORD_COLORS["dark"],
            linewidth=0.2,
            ax=ax,
            legend=False,
        )

        # E. Calculate Clamped Line Coordinates
        # ---------------------------------------------------------
        # 1. Start with the data's actual X-boundaries
        x_start = clean_sub[x_col].min()
        x_end = clean_sub[x_col].max()

        # 2. Calculate X where line hits the Y-Floor ( x = (y - c) / m )
        #    This finds the exact X-coordinate where the line crosses Y=-0.8
        if group == "HI":
            Y_FLOOR_LIMIT = -0.3
            Y_CEIL_LIMIT = 1.1
        else:
            Y_FLOOR_LIMIT = -0.75
            Y_CEIL_LIMIT = 1.1
        if abs(slope) > 1e-5:  # Avoid division by zero
            x_at_y_floor = (Y_FLOOR_LIMIT - intercept) / slope
            x_at_y_ceil = (Y_CEIL_LIMIT - intercept) / slope
        else:
            x_at_y_floor = x_end if slope < 0 else x_start
            x_at_y_ceil = x_start

        # 3. Apply the Cut (Logic depends on slope direction)
        if slope < 0:
            # Negative Slope: Line goes DOWN as X goes RIGHT.
            # Cut the right side (End) if it goes below floor.
            x_end = min(x_end, x_at_y_floor)

            # Cut the left side (Start) if it starts above ceiling (rare but safe)
            x_start = max(x_start, x_at_y_ceil)
        else:
            # Positive Slope: Line goes UP as X goes RIGHT.
            # Cut the left side (Start) if it starts below floor.
            x_start = max(x_start, x_at_y_floor)

            # Cut the right side (End) if it goes above ceiling.
            x_end = min(x_end, x_at_y_ceil)

        # 4. Generate Points
        if x_start < x_end:
            x_vals = np.linspace(x_start, x_end, 100)
            y_vals = slope * x_vals + intercept

            # F. Plot the Line
            (line,) = ax.plot(
                x_vals, y_vals, color=color, label=label_text, linewidth=3.0
            )

            # G. Add Path Effect (Outline)
            line.set_path_effects(
                [
                    pe.withStroke(
                        linewidth=5, foreground=NORD_COLORS["dark"], alpha=0.8
                    ),
                    pe.Normal(),
                ]
            )

    # 4. Styling
    ax.set_xlabel("Min-Max Normalized\nFold Change (Team Count)")
    ax.set_ylabel("Structural Coherence")

    # Custom Legend
    legend = ax.legend(
        title="Network Type (Slope, m)",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
    )

    # Legend Styling
    legend.get_frame().set_edgecolor(NORD_COLORS["dark"])
    legend.get_frame().set_linewidth(1.5)

    # Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])
        spine.set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig(er_hi_plot_dir / "minmax_fcnumteams_vs_meancoh_regr.png", dpi=300)
    plt.savefig(
        er_hi_plot_dir / "minmax_fcnumteams_vs_meancoh_regr.svg",
        dpi=300,
        transparent=True,
    )
    # plt.show()
    plt.close()


# --- Run ---
plot_er_hi_scatter(
    cohres_df,
    "FoldChangeNumTeams",
    "MeanCoh",
    "NetType",
    save_path,
)

plot_er_hi_scatter_with_regression(
    cohres_df,
    "MinMaxFoldChangeNumTeams",
    "MeanCoh",
    "NetType",
    save_path,
)

# plot_er_hi_scatter_minmax(
#     cohres_df,
#     "MinMaxFoldChangeNumTeams",
#     "MeanCoh",
#     "NetType",
#     save_path,
# )

###################################################################################################
###################################################################################################

###################################################################################################
###  MeanCoherenceValue / FoldChangeNumTeams Vs MeanCommunicability
###################################################################################################


def create_stacked_scatterplots_comm_hi(df, basenet_dict, save_path):
    """
    Generates vertically stacked scatterplots.
    - Points have transparent fill (alpha=0.7)
    - Points have transparent white edges
    - Smaller points (s=70)
    - No titles
    - Legend outside
    """
    dark_color = NORD_COLORS["dark"]
    # Define transparent white (R, G, B, Alpha)
    # transparent_white_edge = (0, 0, 0, 0.1)
    transparent_white_edge = (0.26, 0.30, 0.37, 0.2)

    for bn in basenet_dict.keys():
        print(f"Plotting for BaseNet: {bn}")

        # Subset Data
        bn_cdf = df[(df["BaseNet"] == bn) & (df["NetType"] == "HI")]

        if bn_cdf.empty:
            continue

        # Initialize Figure (2 Rows, 1 Column)
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True)

        # --- Plot 1 (Top): Communicability vs FoldChangeNumTeams ---
        sns.scatterplot(
            data=bn_cdf,
            x="NormMeanComm",
            y="FoldChangeNumTeams",
            hue="NumNodesPerGroup",
            palette=NORD_PALETTE,
            s=50,  # Smaller size
            alpha=0.8,  # Transparent Fill
            edgecolor=transparent_white_edge,  # Transparent White Edge
            linewidth=1.0,  # Thin edge line
            ax=axes[0],
        )
        axes[0].set_ylabel("Fold Change (No. Teams)")

        # --- Plot 2 (Bottom): Communicability vs MeanCoh ---
        sns.scatterplot(
            data=bn_cdf,
            x="NormMeanComm",
            y="MeanCoh",
            hue="NumNodesPerGroup",
            palette=NORD_PALETTE,
            s=50,  # Smaller size
            alpha=0.8,  # Transparent Fill
            edgecolor=transparent_white_edge,  # Transparent White Edge
            linewidth=1.0,  # Thin edge line
            ax=axes[1],
        )
        axes[1].set_ylabel("Structural Coherence")
        axes[1].set_xlabel(r"$\log_{10}$(Mean Normalized Communicability)")

        # --- Legend Customization ---
        axes[0].legend_.remove()
        axes[1].legend_.remove()

        handles, labels = axes[0].get_legend_handles_labels()

        # Add legend outside top plot
        leg = axes[0].legend(
            handles,
            labels,
            title="Scale",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            edgecolor=dark_color,
        )

        # Thicken the legend border
        leg.get_frame().set_linewidth(2.5)

        plt.tight_layout()
        # plt.show()
        plt.savefig(save_path / f"{bn}_meancomm_HI.png", dpi=300)
        plt.savefig(save_path / f"{bn}_meancomm_HI.svg", dpi=300, transparent=True)
        plt.close()


# --- Run ---
save_path = fig_dir / "Fig2"
save_path.mkdir(exist_ok=True)
# create_stacked_scatterplots_comm_hi(cohres_df, basenet_numgroups_dict, save_path)

############################################################################################################
############################################################################################################

###################################################################################################
###  MeanCoherenceValue / FoldChangeNumTeams Vs Density
###################################################################################################


def create_stacked_lineplots_density_hi(df, basenet_dict, save_path):
    """
    Generates vertically stacked lineplots with:
    - Numeric X-axis (Density)
    - Transparent Error Bars (via layering)
    - Large borderless dots
    - Thinner connecting lines
    """
    dark_color = NORD_COLORS["dark"]

    for bn in basenet_dict.keys():
        print(f"Plotting for BaseNet: {bn}")

        # 1. Subset Data
        bn_cdf = df[(df["BaseNet"] == bn) & (df["NetType"] == "HI")].copy()

        # Filter specific Nodes and Density
        # bn_cdf = bn_cdf[bn_cdf["NumNodesPerGroup"].isin([10, 30, 50])]
        bn_cdf = bn_cdf[bn_cdf["Density"].isin(np.append(np.arange(10, 101, 10), 100))]

        if bn_cdf.empty:
            continue

        # bn_cdf["Density"] = pd.to_numeric(bn_cdf["Density"])

        # Initialize Figure
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 9.5), sharex=True)

        # --- Helper Function for Consistent Styling ---
        def plot_styled_layer(ax, y_col, add_legend=False):
            # Layer 1: Background (Transparent Error Bars)
            sns.lineplot(
                data=bn_cdf,
                x="Density",
                y=y_col,
                hue="Scale",
                palette=NORD_PALETTE,
                ax=ax,
                marker="o",
                err_style="bars",
                errorbar="sd",
                err_kws={
                    "capsize": 3,  # Width of the cap in pixels (Try 3-6)
                    "capthick": 2.0,  # Thickness of the cap line
                    "linewidth": 2.0,  # Thickness of the vertical bar
                },
                alpha=0.5,  # 30% Opacity for bars and line shadow
                linewidth=1.5,  # Match the foreground line width
                markersize=10,  # Match foreground size
                markeredgewidth=0,
                legend=False,  # Never show legend for background layer
                zorder=1,
            )

            # Layer 2: Foreground (Solid Line & Dots)
            sns.lineplot(
                data=bn_cdf,
                x="Density",
                y=y_col,
                hue="Scale",
                palette=NORD_PALETTE,
                ax=ax,
                marker="o",
                # Disable error bars here (already drawn in Layer 1)
                errorbar=None,
                # Visuals for Foreground
                alpha=1.0,  # Fully opaque
                linewidth=2.5,  # Thinner line as requested
                markersize=10,  # Large dots
                markeredgewidth=0,  # No border on dots
                legend=add_legend,  # Only add legend if requested
                zorder=2,
            )

        # --- Plot 1 (Top): FoldChangeNumTeams ---
        plot_styled_layer(axes[0], "FoldChangeNumTeams", add_legend=True)
        axes[0].set_ylim(-0.5, 15)
        axes[0].set_ylabel("Fold Change (Team Count)")
        axes[0].set_xlabel(None)  # Hide x-label for top plot

        # --- Plot 2 (Bottom): Mean Coherence ---
        plot_styled_layer(axes[1], "MeanCoh", add_legend=False)
        axes[1].set_ylim(-0.1, 1.10)
        axes[1].set_ylabel("Structural Coherence")
        axes[1].set_xlabel("Density")

        # --- Legend Customization ---
        # We enabled the legend on axes[0], now we move it outside
        if axes[0].get_legend():
            sns.move_legend(
                axes[0],
                "upper left",
                bbox_to_anchor=(1.02, 1),
                title="Scale",
                frameon=True,
            )
            # Thicken legend border
            axes[0].get_legend().get_frame().set_edgecolor(dark_color)
            axes[0].get_legend().get_frame().set_linewidth(2.5)

        fig.suptitle(f"{bn}", color=dark_color, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        plt.tight_layout()

        # Save
        # plt.show()
        plt.savefig(save_path / f"{bn}_density_HI.png", dpi=300)
        plt.savefig(save_path / f"{bn}_density_HI.svg", dpi=300, transparent=True)
        plt.close()


def create_stacked_scatterplots_density_hi(df, basenet_dict, save_path):
    """
    Generates vertically stacked scatterplots.
    - Points have transparent fill (alpha=0.7)
    - Points have transparent white edges
    - Smaller points (s=70)
    - No titles
    - Legend outside
    """
    dark_color = NORD_COLORS["dark"]
    # Define transparent white (R, G, B, Alpha)
    # transparent_white_edge = (0, 0, 0, 0.1)
    transparent_white_edge = (0.26, 0.30, 0.37, 0.2)

    for bn in basenet_dict.keys():
        print(f"Plotting for BaseNet: {bn}")

        # Subset Data
        bn_cdf = df[(df["BaseNet"] == bn) & (df["NetType"] == "HI")]

        # SUbet only the 10 and 50 noe networks
        bn_cdf = bn_cdf[bn_cdf["NumNodesPerGroup"].isin([10, 50])]
        bn_cdf = bn_cdf[bn_cdf["Density"].isin(np.append(np.arange(20, 101, 10), 100))]

        if bn_cdf.empty:
            continue

        # Initialize Figure (2 Rows, 1 Column)
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True)

        # --- Plot 1 (Top): Communicability vs FoldChangeNumTeams ---
        sns.violinplot(
            data=bn_cdf,
            x="Density",
            y="FoldChangeNumTeams",
            hue="NumNodesPerGroup",
            palette=NORD_PALETTE,
            linewidth=0.5,  # Thin edge line
            saturation=0.7,  # Slightly desaturated for the background
            ax=axes[0],
            inner=None,
        )
        for art in axes[0].collections:
            if isinstance(art, collections.PolyCollection):
                art.set_alpha(0.4)

        # 2. The Rain (Strip for raw data)
        sns.stripplot(
            data=bn_cdf,
            x="Density",
            y="FoldChangeNumTeams",
            hue="NumNodesPerGroup",
            palette=NORD_PALETTE,
            dodge=True,  # Crucial: Aligns the dots with the hue groups
            jitter=True,  # Adds random noise to x-axis
            size=5,  # Adjust size as needed
            alpha=0.9,  # Transparency for the dots
            linewidth=1.5,  # Thin white edge helps dots pop
            edgecolor=transparent_white_edge,
            ax=axes[0],
            legend=False,  # Prevent duplicate legend entries
            zorder=1,
        )
        axes[0].set_ylabel("Fold Change (Teams)")

        # --- Plot 2 (Bottom): Communicability vs MeanCoh ---
        # 1. The Cloud
        sns.violinplot(
            data=bn_cdf,
            x="Density",
            y="MeanCoh",
            hue="NumNodesPerGroup",
            palette=NORD_PALETTE,
            linewidth=0.5,
            saturation=0.7,
            ax=axes[1],
            inner=None,
        )
        for art in axes[1].collections:
            if isinstance(art, collections.PolyCollection):
                art.set_alpha(0.4)

        # 2. The Rain
        sns.stripplot(
            data=bn_cdf,
            x="Density",
            y="MeanCoh",
            hue="NumNodesPerGroup",
            palette=NORD_PALETTE,
            dodge=True,
            jitter=True,
            size=5,
            alpha=0.9,
            linewidth=1.5,
            edgecolor=transparent_white_edge,
            ax=axes[1],
            legend=False,
            zorder=1,
        )
        # sns.violinplot(
        #     data=bn_cdf,
        #     x="Density",
        #     y="FoldChangeNumTeams",
        #     hue="NumNodesPerGroup",
        #     palette=NORD_PALETTE,
        #     # s=50,  # Smaller size
        #     # alpha=0.8,  # Transparent Fill
        #     # edgecolor=transparent_white_edge,  # Transparent White Edge
        #     linewidth=1.2,  # Thin edge line
        #     # dodge=True,
        #     # split=True,
        #     ax=axes[0],
        # )
        # axes[0].set_ylabel("Fold Change (Teams)")
        #
        # # --- Plot 2 (Bottom): Communicability vs MeanCoh ---
        # sns.violinplot(
        #     data=bn_cdf,
        #     x="Density",
        #     y="MeanCoh",
        #     hue="NumNodesPerGroup",
        #     palette=NORD_PALETTE,
        #     # s=50,  # Smaller size
        #     # alpha=0.8,  # Transparent Fill
        #     # edgecolor=transparent_white_edge,  # Transparent White Edge
        #     linewidth=1.2,  # Thin edge line
        #     # dodge=True,
        #     # split=True,
        #     ax=axes[1],
        # )
        axes[1].set_ylabel("Structural Coherence")
        axes[1].set_xlabel(r"Density")

        # --- Legend Customization ---
        axes[0].legend_.remove()
        axes[1].legend_.remove()

        handles, labels = axes[0].get_legend_handles_labels()

        # Add legend outside top plot
        leg = axes[0].legend(
            handles,
            labels,
            title="Scale",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            edgecolor=dark_color,
        )

        # Thicken the legend border
        leg.get_frame().set_linewidth(2.5)

        plt.tight_layout()
        # plt.show()
        plt.savefig(save_path / f"{bn}_density_HI.png", dpi=300)
        plt.savefig(save_path / f"{bn}_density_HI.svg", dpi=300, transparent=True)
        plt.close()


# --- Run ---
save_path = fig_dir / "Fig2"
save_path.mkdir(exist_ok=True)
# create_stacked_scatterplots_density_hi(cohres_df, basenet_numgroups_dict, save_path)
create_stacked_lineplots_density_hi(cohres_df, basenet_numgroups_dict, save_path)

###################################################################################################
###################################################################################################

###################################################################################################
###  MeanCommunicability Vs Density
###################################################################################################


def create_density_communicability_plot_hi(df, save_path):
    """
    Creates a single enhanced scatterplot for Density vs MeanCommunicability.
    Uses BOTH color and shape for NumNodes.
    """
    # Filter for ER networks only
    er_df = df[df["NetType"] == "HI"]

    # Setup Colors
    dark_color = NORD_COLORS["dark"] if "NORD_COLORS" in globals() else "#2E3440"
    # transparent_black_edge = (0, 0, 0, 0.5)
    transparent_black_edge = (0.26, 0.30, 0.37, 0.1)

    # Define 9 Distinct Markers
    custom_markers = ["o", "s", "D", "^", "v", "X", "p", "*", ">"]

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.scatterplot(
        data=er_df,
        x="MeanComm",
        y="Density",
        hue="NumNodes",  # Add Color back
        style="NumNodes",  # Keep Shape
        palette=NORD_PALETTE,  # Your palette (will cycle if <9 colors)
        markers=custom_markers,  # Your custom shapes
        s=80,
        alpha=0.9,
        edgecolor=transparent_black_edge,  # Black "Highlighter" border
        linewidth=1.5,
        ax=ax,
    )

    ax.set_ylabel("Network Density")
    ax.set_xlabel(r"$\log_{10}$(Mean Communicability)")

    # --- Legend Customization ---
    leg = ax.legend(
        title="Network Size\n(Number of Nodes)",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        edgecolor=dark_color,
    )
    leg.get_frame().set_linewidth(2.5)

    plt.tight_layout()
    plt.savefig(save_path / "meancomm_vs_density_HI.png", dpi=300)
    plt.savefig(save_path / "meancomm_vs_density_HI.svg", dpi=300, transparent=True)
    # plt.show()
    plt.close()


# --- Run ---
save_path = fig_dir / "Fig2"
create_density_communicability_plot_hi(cohres_df, save_path)

####################################################################################################
####  MeanCoh vs FoldChangeNumTeams
####################################################################################################
#
#
# def plot_slope_regression_hi(df, x_col, y_col, hue_col, save_path):
#    # --- 1. Setup Data & Colors ---
#    palette = (
#        NORD_PALETTE if "NORD_PALETTE" in globals() else sns.color_palette("viridis")
#    )
#    dark_color = NORD_COLORS["dark"] if "NORD_COLORS" in globals() else "#2E3440"
#
#    groups = sorted(df[hue_col].unique())
#
#    fig, ax = plt.subplots(figsize=(9, 6))
#
#    # --- 2. Iterate Through Groups ---
#    for i, group in enumerate(groups):
#        # A. Subset Data
#        subset = df[df[hue_col] == group]
#        if len(subset) < 2:
#            continue
#
#        # B. Linear Regression (Slope & Intercept)
#        # dropping NaNs ensures clean calculation
#        clean_sub = subset[[x_col, y_col]].dropna()
#        slope, intercept, r_value, p_value, std_err = stats.linregress(
#            clean_sub[x_col], clean_sub[y_col]
#        )
#
#        # Format Label: e.g., "10 Nodes (m=0.45)"
#        # We can also add significance like "m=0.45*" if p < 0.05
#        sig = "*" if p_value < 0.05 else ""
#        label_text = f"{group} ($m={slope:.3f} {sig}$)"
#
#        # C. Get Color
#        color = palette[i % len(palette)]
#
#        # D. Plot Scatter (Background)
#        sns.scatterplot(
#            data=subset,
#            x=x_col,
#            y=y_col,
#            color=color,
#            s=70,
#            alpha=0.7,
#            edgecolor=(0, 0, 0, 0.1),  # Light black edge on points
#            # edgecolor=(0.26, 0.30, 0.37, 0.5),
#            linewidth=1,
#            ax=ax,
#            legend=False,  # We build legend manually via the line
#        )
#
#        # E. Plot Regression Line MANUALLY
#        # We generate x values to draw a smooth line
#        x_vals = np.linspace(clean_sub[x_col].min(), clean_sub[x_col].max(), 100)
#        y_vals = slope * x_vals + intercept
#
#        # Plot the line
#        # Note: We add the 'label' here so it shows up in the legend
#        (line,) = ax.plot(x_vals, y_vals, color=color, label=label_text, linewidth=2.5)
#
#        # --- THE TRICK FOR VISIBILITY ---
#        # Add a "Path Effect" to create an outline around the line
#        # 'linewidth=5' determines how thick the border is.
#        # 'foreground="white"' creates a white gap between points and the line.
#        # Change foreground to 'black' if you want a dark border.
#        line.set_path_effects(
#            [
#                pe.withStroke(linewidth=5, foreground="black", alpha=0.7),  # The Border
#                pe.Normal(),  # The Core Line itself
#            ]
#        )
#
#    # --- 3. Custom Styling ---
#    ax.set_xlabel("Fold Change (Num Teams)")
#    ax.set_ylabel("Mean Coherence")
#
#    # Customize Legend
#    leg = ax.legend(
#        title="Scale\n(Linear Slope $m$)",
#        bbox_to_anchor=(1.02, 1),
#        loc="upper left",
#        borderaxespad=0,
#        edgecolor=dark_color,
#    )
#    leg.get_frame().set_linewidth(2.0)
#
#    plt.tight_layout()
#    plt.savefig(save_path / "fcnumteams_vs_meancoh_linreg_HI.png", dpi=300)
#    # plt.show()
#    plt.close()
#
#
# save_path = fig_dir / "Fig2"
# save_path.mkdir(exist_ok=True)
## plot_slope_regression_hi(
##     cohres_df[cohres_df["NetType"] == "HI"],
##     "FoldChangeNumTeams",
##     "MeanCoh",
##     "NumNodesPerGroup",
##     save_path,
## )
#
####################################################################################################
####################################################################################################
#
#####################################################################################################
##### MinMaxCommunicabilityBin vs MinMaxFoldChangeNumTeams
#####################################################################################################
#
#
# def plot_refined_coherence_pointplot_hi(
#    plot_df, base_df, x_col, y_col, hue_col, metric_col="MeanCoh", save_path=Path(".")
# ):
#    """
#    Plots a pointplot where colors are mapped to the MEAN coherence of the BaseNet.
#    - Darkest color is pure Black (#000000).
#    - Lightest color is Nord Orange (#d08770).
#    - Lines are thicker, no white borders.
#    - Legend text is larger.
#    """
#
#    # --- 1. Calculate the Mapping Metric (Mean Coherence) ---
#    coherence_map = base_df.groupby(hue_col)[metric_col].mean().to_dict()
#
#    # --- 2. Build Colormap (Pure Black -> Nord Orange) ---
#    custom_cmap = colors.LinearSegmentedColormap.from_list(
#        "black_to_orange", ["#000000", "#d08770"]
#    )
#
#    # --- 3. Generate Palette Dictionary ---
#    vals = list(coherence_map.values())
#    norm = colors.Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))
#
#    # Create a dict { 'BaseNetName': (R, G, B, A) }
#    palette_dict = {name: custom_cmap(norm(val)) for name, val in coherence_map.items()}
#
#    # --- 4. Setup Plot ---
#    plt.figure(figsize=(10, 6))
#
#    # --- 5. Draw Pointplot ---
#    ax = sns.pointplot(
#        data=plot_df,
#        x=x_col,
#        y=y_col,
#        hue=hue_col,
#        estimator="mean",
#        errorbar="sd",
#        capsize=0.1,
#        markers="o",
#        scale=1.1,  # Increased marker size slightly
#        dodge=0.3,
#        palette=palette_dict,
#        linewidth=2.5,  # Thicker lines (was 1.0)
#        errwidth=2.0,  # Thicker error bars (was 1.0)
#        legend=False,
#    )
#
#    # Note: "Halo" path effects section has been removed completely.
#
#    # --- 6. Aesthetics ---
#    ax.set_xlabel("Min-Max Normalized Mean Communicability")
#    ax.set_ylabel("Min-Max Normalized Team Count")
#    ax.grid(True, linestyle="--", alpha=0.3)
#    plt.subplots_adjust(right=0.75)
#
#    # --- 7. Create Sorted Legend ---
#    sorted_items = sorted(coherence_map.items(), key=lambda item: item[1])
#
#    legend_elements = [
#        Patch(
#            facecolor=palette_dict[name], edgecolor="none", label=f"{name} : {val:.2f}"
#        )
#        for name, val in sorted_items
#    ]
#
#    ax.legend(
#        handles=legend_elements,
#        title=f"{hue_col} : Mean Coherence",
#        bbox_to_anchor=(1.02, 1),
#        loc="upper left",
#        borderaxespad=0,
#        frameon=True,
#        fontsize=14,  # Increased from 9 to 12
#        title_fontsize=15,  # Increased title size slightly
#    )
#
#    plt.tight_layout()
#    plt.savefig(save_path / "minmax_communicability_vs_fcnumteams_HI.png", dpi=300)
#    # plt.show()
#    plt.close()
#
#
## --- Example Run ---
# save_path = fig_dir / "Fig2"
# save_path.mkdir(exist_ok=True)
## plot_refined_coherence_pointplot_hi(
##     plot_df=cohres_df[cohres_df["NetType"] == "HI"],
##     base_df=base_df,
##     x_col="MinMaxCommunicabilityBin",
##     y_col="MinMaxFoldChangeNumTeams",
##     hue_col="BaseNet",
##     metric_col="MeanCoh",
##     save_path=save_path,
## )
#
#
#####################################################################################################
#####################################################################################################
